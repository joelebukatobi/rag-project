from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple
import io
import requests
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import tarfile
import json

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm

from edgar import Company, set_identity
import pandas as pd

# REQUIRED: The SEC will block you without a User-Agent string
set_identity("cg77@fordham.edu")

def _safe_year(value: object) -> Optional[int]:
    """Extract a 4-digit year from mixed date formats."""
    if value is None:
        return None
    text = str(value)
    match = re.search(r"(19|20)\d{2}", text)
    if not match:
        return None
    return int(match.group(0))


def _pick_first(record: dict, keys: Sequence[str], default: str = "") -> str:
    for key in keys:
        if key in record and record[key] is not None:
            val = str(record[key]).strip()
            if val:
                return val
    return default


def _build_raw_text(record: dict) -> str:
    text_keys = ["text", "filing_text", "raw_text", "document"]
    text = _pick_first(record, text_keys)
    if text:
        return text

    section_keys = ["section_1", "section_1a", "section_3", "section_7", "section_7a", "section_8"]
    section_blobs = []
    for key in section_keys:
        val = record.get(key)
        if isinstance(val, str) and val.strip():
            section_blobs.append(f"{key.upper()}\n{val.strip()}")
    return "\n\n".join(section_blobs)

#Uses edgar tools d
def load_live_filings(tickers, year_range):
    rows = []
    year_min, year_max = year_range

    for ticker in tqdm(tickers, desc="Processing Tickers"):
        try:
            company = Company(ticker)
            filings = company.get_filings(form="10-K")
            if not filings: continue

            for filing in filings:
                f_year = filing.filing_date.year
                if year_min <= f_year <= year_max:
                    
                    # 1. Get the structured TenK object
                    tenk = filing.obj()
                    
                    # 2. Extract Sections (Item 1A, 7, and 8)
                    # We use .get() or dictionary access to avoid AttributeErrors
                    risk_factors = getattr(tenk, "risk_factors", "") or tenk["Item 1A"] or ""
                    mda = getattr(tenk, "management_discussion", "") or tenk["Item 7"] or ""
                    
                    # Item 8 (Financials) is where the Debt Tables live
                    financials = tenk["Item 8"] or ""
                    
                    # Item 3 (Legal)
                    legal = tenk["Item 3"] or ""

                    rows.append({
                        "ticker": ticker.upper(),
                        "year": f_year,
                        "company": filing.company,
                        "section_1a": str(risk_factors),
                        "section_7": str(mda),
                        "section_8": str(financials),
                        "section_3": str(legal)
                    })
        except Exception as e:
            print(f"Skipping {ticker} year {f_year if 'f_year' in locals() else 'unknown'} due to error: {e}")
            continue

    return pd.DataFrame(rows)


# Used Parquet
def load_raw_filings_hardcoded(
    tickers: Iterable[str],
    year_range: Tuple[int, int],
    filing_type: str = "10-K",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    The 'No-Fail' Loader: Downloads the raw compressed corpus directly 
    from the repo, bypassing all 'load_dataset' script logic.
    """
    ticker_set = {t.upper() for t in tickers}
    year_min, year_max = year_range
    rows = []

    print("Downloading raw corpus file (this may take a minute)...")
    # This repo usually stores its primary data in 'edgar-corpus.tar.gz' or similar.
    # We download the specific file directly.
    try:
        local_path = hf_hub_download(
            repo_id="eloukas/edgar-corpus",
            filename="corpus.tar.gz",  # If this fails, try 'data.tar.gz'
            repo_type="dataset"
        )
    except Exception as e:
        return f"Could not find the raw file: {e}. Please check the 'Files' tab on the HF repo."

    with tarfile.open(local_path, "r:gz") as tar:
        # Most SEC datasets on HF are collections of JSON files or one giant JSONL
        for member in tqdm(tar.getmembers(), desc="Parsing Tarball"):
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            
            f = tar.extractfile(member)
            if f is None: continue
            
            # Load the record
            record = json.load(f)
            
            # --- Apply your existing filter logic ---
            ticker = _pick_first(record, ["ticker", "symbol"]).upper()
            if ticker_set and ticker not in ticker_set:
                continue

            # Filtering and date logic exactly as you had it...
            year = _safe_year(_pick_first(record, ["year", "filing_date"]))
            if year is None or year < year_min or year > year_max:
                continue

            rows.append({
                "ticker": ticker,
                "company_name": record.get("company", ticker),
                "year": year,
                "raw_text": _build_raw_text(record),
                "section_7": record.get("section_7", ""), 
            })

            if limit and len(rows) >= limit:
                break

    return pd.DataFrame(rows)

#Uses edgar-corpus.py
def load_raw_filings(
    tickers: Iterable[str],
    year_range: Tuple[int, int],
    filing_type: str = "10-K",
    split: str = "train",
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load and filter filings from eloukas/edgar-corpus into a normalized DataFrame.

    Output columns: ticker, company_name, year, raw_text.
    """
    ticker_set = {t.upper() for t in tickers}
    year_min, year_max = year_range

    dataset = load_dataset("eloukas/edgar-corpus", split=split)
    rows = []

    iterator = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))

    for record in tqdm(iterator, desc="Loading filings"):
        ticker = _pick_first(record, ["ticker", "symbol", "stock_ticker"]).upper()
        if ticker_set and ticker not in ticker_set:
            continue

        record_filing_type = _pick_first(record, ["filing_type", "form", "type"]).upper()
        if filing_type and record_filing_type and filing_type.upper() != record_filing_type:
            continue

        year = _safe_year(
            _pick_first(record, ["year", "filing_date", "report_date", "period_of_report"]) or None
        )
        if year is None or year < year_min or year > year_max:
            continue

        raw_text = _build_raw_text(record)
        if not raw_text.strip():
            continue

        company_name = _pick_first(record, ["company", "company_name", "name"], default=ticker)

        rows.append(
            {
                "ticker": ticker,
                "company_name": company_name,
                "year": year,
                "raw_text": raw_text,
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["ticker", "year"]).reset_index(drop=True)
    return df


def filing_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return filing counts by ticker/year for quick quality checks."""
    if df.empty:
        return pd.DataFrame(columns=["ticker", "year", "count"])
    stats = (
        df.groupby(["ticker", "year"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["ticker", "year"])
    )
    return stats
