from __future__ import annotations

import os
import pickle
import re
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm.auto import tqdm

# EXPANDED: Labels to match your new Taxonomy (Items 1, 1A, 3, 7, 8)
SECTION_LABELS = {
    "section_1": "Business Description",
    "section_1a": "Risk Factors",
    "section_3": "Legal Proceedings",
    "section_7": "MD&A",
    "section_8": "Financial Statements",
}


SECTION_PATTERNS = {
    "section_1a": re.compile(
        r"item\s*1a\.?\s*risk\s*factors(.*?)(?=item\s*1b\.?|item\s*2\.?)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    "section_7": re.compile(
        r"item\s*7\.?\s*management'?s\s*discussion\s*and\s*analysis(.*?)(?=item\s*7a\.?|item\s*8\.?)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
    "section_3": re.compile(
        r"item\s*3\.?\s*legal\s*proceedings(.*?)(?=item\s*4\.?|item\s*5\.?)",
        flags=re.IGNORECASE | re.DOTALL,
    ),
}


def _clean_text(text: object) -> str:
    """Safely clean text by handling non-string types and whitespace."""
    # If it's not a string, or it's null, return empty string
    if not isinstance(text, str):
        return ""
    
    # Perform regex substitution for multiple whitespaces
    cleaned = re.sub(r"\s+", " ", text)
    return cleaned.strip()


def extract_sections_from_text(raw_text: str) -> Dict[str, str]:
    """Extract key SEC sections using robust regex fallbacks."""
    found = {}
    if not raw_text:
        return found

    for section_type, pattern in SECTION_PATTERNS.items():
        match = pattern.search(raw_text)
        if match:
            section_text = _clean_text(match.group(1))
            if section_text:
                found[section_type] = section_text

    return found

# --- PARSING LOGIC ---
def parse_sections(df_filings: pd.DataFrame, target_sections: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Parses the expanded set of sections required for the Credit Signal Taxonomy.
    """
    if target_sections is None:
        target_sections = list(SECTION_LABELS.keys())

    rows: List[Dict[str, object]] = []
    
    for row in tqdm(df_filings.itertuples(index=False), total=len(df_filings), desc="Parsing sections"):
        for section_type in target_sections:
            val = getattr(row, section_type, "")
            section_text = str(val) if val is not None else ""
            
            if len(section_text.strip()) < 50: # Increased threshold for signal quality
                continue
                
            rows.append({
                "ticker": row.ticker,
                "year": int(row.year),
                "section_type": section_type,
                "section_text": _clean_text(section_text),
                # TAXONOMY ENHANCEMENT: Store raw length for Meta Signal #18
                "section_char_len": len(section_text) 
            })

    return pd.DataFrame(rows)


#Previous
# def parse_sections(df_filings: pd.DataFrame, target_sections: Optional[Sequence[str]] = None) -> pd.DataFrame:
#     """
#     Build a section-level DataFrame from raw filings.

#     Output columns: ticker, year, section_type, section_text.
#     """
#     if target_sections is None:
#         target_sections = list(SECTION_PATTERNS.keys())

#     rows: List[Dict[str, object]] = []
#     for row in tqdm(df_filings.itertuples(index=False), total=len(df_filings), desc="Parsing sections"):
#         section_map = extract_sections_from_text(row.raw_text)
#         for section_type in target_sections:
#             section_text = section_map.get(section_type, "")
#             if not section_text:
#                 continue
#             rows.append(
#                 {
#                     "ticker": row.ticker,
#                     "year": int(row.year),
#                     "section_type": section_type,
#                     "section_text": section_text,
#                 }
#             )

#     return pd.DataFrame(rows)


def section_coverage_stats(df_sections: pd.DataFrame) -> pd.DataFrame:
    if df_sections.empty:
        return pd.DataFrame(columns=["section_type", "coverage"])
    stats = (
        df_sections.groupby("section_type", as_index=False)
        .size()
        .rename(columns={"size": "coverage"})
        .sort_values("coverage", ascending=False)
    )
    return stats


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if not text:
        return []

    tokens = text.split()
    if not tokens:
        return []

    chunks = []
    step = max(1, chunk_size - chunk_overlap)
    for start in range(0, len(tokens), step):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            break
        chunks.append(" ".join(chunk_tokens))
        if end >= len(tokens):
            break
    return chunks


# --- CHUNKING LOGIC ---
def _chunk_row(args: Tuple[Dict[str, object], int, int]) -> List[Dict[str, object]]:
    """
    Enhanced chunker that attaches 'Section Meta-Signals' to every chunk.
    """
    row, chunk_size, chunk_overlap = args
    section_chunks = _chunk_text(str(row["section_text"]), chunk_size, chunk_overlap)

    out: List[Dict[str, object]] = []
    for idx, chunk_text in enumerate(section_chunks):
        out.append(
            {
                "ticker": row["ticker"],
                "year": int(row["year"]),
                "section_type": row["section_type"],
                "chunk_id": f"{row['ticker']}-{row['year']}-{row['section_type']}-{idx}",
                "text": chunk_text,
                # TAXONOMY METADATA: 
                # This helps the AI identify 'Disclosure Length' shifts (Signal 18)
                "meta_section_len": row["section_char_len"], 
                "label": SECTION_LABELS.get(str(row["section_type"]), "General")
            }
        )
    return out

def build_chunks(
    df_sections: pd.DataFrame,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
    cache_path: str = "data/chunks.pkl",
    use_multiprocessing: bool = True,
) -> List[Dict[str, object]]:
    """Create section-aware chunks and cache to disk."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    records = df_sections.to_dict(orient="records")
    if not records:
        return []

    args = [(row, chunk_size, chunk_overlap) for row in records]
    all_chunks: List[Dict[str, object]] = []

    if use_multiprocessing and len(args) > 50:
        workers = max(1, min(cpu_count(), 8))
        with Pool(processes=workers) as pool:
            for part in tqdm(pool.imap(_chunk_row, args), total=len(args), desc="Chunking sections"):
                all_chunks.extend(part)
    else:
        for arg in tqdm(args, desc="Chunking sections"):
            all_chunks.extend(_chunk_row(arg))

    with open(cache_path, "wb") as f:
        pickle.dump(all_chunks, f)

    return all_chunks


def chunk_stats(all_chunks: List[Dict[str, object]]) -> pd.DataFrame:
    if not all_chunks:
        return pd.DataFrame(columns=["metric", "value"])

    lengths = [len(str(c["text"]).split()) for c in all_chunks]
    stats = pd.DataFrame(
        {
            "metric": ["count", "min_tokens", "mean_tokens", "max_tokens"],
            "value": [
                len(lengths),
                min(lengths),
                round(sum(lengths) / len(lengths), 2),
                max(lengths),
            ],
        }
    )
    return stats
