from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.retrieve import HybridRetriever


@dataclass
class EvalCase:
    query: str
    ticker: str
    section_type: str
    year: int
    expected_keywords: List[str]


def build_test_set(
    tickers: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    max_cases: int = 20,
) -> List[EvalCase]:
    """Create a representative test set from the given tickers and years."""
    templates = [
        ("supply chain risk changes", "section_1a", ["supply", "vendor"]),
        ("regulatory and compliance exposure", "section_1a", ["regulation", "compliance"]),
        ("demand trends by segment", "section_7", ["segment", "demand"]),
        ("litigation and legal actions", "section_3", ["litigation", "legal"]),
    ]
    if tickers is None:
        tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "TSLA","CCL"]
    if years is None:
        years = [2019, 2020, 2021, 2022]

    out: List[EvalCase] = []
    for t in tickers:
        for y in years:
            q, section, kws = templates[(y - years[0]) % len(templates)]
            out.append(EvalCase(query=q, ticker=t, section_type=section, year=y, expected_keywords=kws))
    return out[:max_cases]


def recall_at_k(results: List[Dict[str, object]], expected_keywords: List[str], k: int = 5) -> float:
    top = results[:k]
    if not top:
        return 0.0
    joined = " ".join([str(r.get("text", "")).lower() for r in top])
    hit = any(kw.lower() in joined for kw in expected_keywords)
    return 1.0 if hit else 0.0


def mrr(results: List[Dict[str, object]], expected_keywords: List[str]) -> float:
    for rank, item in enumerate(results, start=1):
        text = str(item.get("text", "")).lower()
        if any(kw.lower() in text for kw in expected_keywords):
            return 1.0 / rank
    return 0.0


def citation_grounding_score(structured_output: Dict[str, object]) -> float:
    citations = structured_output.get("citations", [])
    if not isinstance(citations, list) or not citations:
        return 0.0
    valid = [c for c in citations if isinstance(c, str) and len(c.strip()) > 0]
    return len(valid) / len(citations)


def structured_output_accuracy(structured_output: Dict[str, object]) -> float:
    required = [
        "company",
        "filing_year_a",
        "filing_year_b",
        "risk_changes",
        "revenue_drivers",
        "litigation_changes",
        "tone_shift",
        "citations",
    ]
    ok = all(k in structured_output for k in required)
    return 1.0 if ok else 0.0


def run_retrieval_eval(
    retriever: HybridRetriever,
    test_set: List[EvalCase],
    top_k: int = 5,
    use_bm25: bool = True,
    use_metadata_filter: bool = True,
    label: str = "Running retrieval eval",
) -> pd.DataFrame:
    rows = []
    for case in tqdm(test_set, desc=label):
        res = retriever.retrieve(
            query=case.query,
            ticker=case.ticker,
            year=case.year,
            section_type=case.section_type,
            top_k=top_k,
            use_bm25=use_bm25,
            use_metadata_filter=use_metadata_filter,
        )
        rows.append(
            {
                "query": case.query,
                "ticker": case.ticker,
                "year": case.year,
                "section_type": case.section_type,
                "Recall@5": recall_at_k(res, case.expected_keywords, k=5),
                "MRR": mrr(res, case.expected_keywords),
            }
        )

    return pd.DataFrame(rows)


def ablation_table(
    retriever: HybridRetriever,
    test_set: List[EvalCase],
    top_k: int = 5,
) -> pd.DataFrame:
    """Run retrieval eval under different configurations for a real ablation study."""
    configs = [
        {"setup": "Hybrid + Metadata Filter", "use_bm25": True, "use_metadata_filter": True},
        {"setup": "No BM25 (Semantic Only)", "use_bm25": False, "use_metadata_filter": True},
        {"setup": "No Metadata Filtering", "use_bm25": True, "use_metadata_filter": False},
        {"setup": "Semantic Only, No Filter", "use_bm25": False, "use_metadata_filter": False},
    ]

    rows = []
    for cfg in configs:
        df = run_retrieval_eval(
            retriever,
            test_set,
            top_k=top_k,
            use_bm25=cfg["use_bm25"],
            use_metadata_filter=cfg["use_metadata_filter"],
            label=cfg["setup"],
        )
        rows.append({
            "setup": cfg["setup"],
            "Recall@5": round(float(df["Recall@5"].mean()), 3),
            "MRR": round(float(df["MRR"].mean()), 3),
        })

    return pd.DataFrame(rows)
