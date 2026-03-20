from __future__ import annotations

import re
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm


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
