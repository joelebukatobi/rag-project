from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple


def _form_str(f: Any) -> str:
    s = str(getattr(f, "form", None) or getattr(f, "filing_type", None) or "").upper()
    return s.replace(" ", "")


def is_amended_10k_filing(f: Any) -> bool:
    form = _form_str(f)
    return "10-K/A" in form or ("/A" in form and "10-K" in form)


def is_base_10k_filing(f: Any) -> bool:
    form = _form_str(f)
    if "10-K" not in form:
        return False
    return not is_amended_10k_filing(f)


def _por_str(f: Any) -> str:
    por = getattr(f, "period_of_report", None) or getattr(f, "report_date", None)
    return str(por) if por is not None else ""


def _filing_date_str(f: Any) -> str:
    fd = getattr(f, "filing_date", None) or getattr(f, "date", None)
    if fd is None:
        return ""
    return str(fd)


def _accession_str(f: Any) -> str:
    return str(getattr(f, "accession_no", None) or "")


def accession_to_slug(accession: str, max_len: int = 64) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", (accession or "").strip())
    s = s.strip("_") or "no_accession"
    return s[:max_len]


def sort_key_deterministic(f: Any) -> Tuple[str, str, str]:
    return (_por_str(f), _filing_date_str(f), _accession_str(f))


def pick_filing_for_year(
    all_filings: Any,
    fiscal_year: int,
    *,
    use_amended_10k: bool,
) -> Tuple[Optional[Any], str]:
    """
    Pick a single filing deterministically (no 'longest text' tournament).
    Default: only plain 10-K filings. If use_amended_10k, include 10-K/A as well.
    """
    try:
        seq = list(all_filings)
    except Exception:
        return None, "No filings in list."
    if not seq:
        return None, "No filings in list."

    base_pool = [f for f in seq if is_base_10k_filing(f)]
    amend_pool = [f for f in seq if is_amended_10k_filing(f)]

    if use_amended_10k:
        pool = base_pool + amend_pool
    else:
        pool = base_pool

    if not pool:
        if not use_amended_10k and amend_pool:
            return None, (
                f"No non-amended 10-K found for FY{fiscal_year}. "
                f"Enable 'Use amended 10-K/A' if you need an amended filing."
            )
        return None, f"No eligible 10-K filings for FY{fiscal_year}."

    cands: List[Any] = []
    for f in pool:
        por_s = _por_str(f)
        if por_s.startswith(f"{fiscal_year}-"):
            cands.append(f)
    if not cands:
        for f in pool:
            fd = _filing_date_str(f)
            if fd.startswith(f"{fiscal_year}-") or fd.startswith(f"{fiscal_year + 1}-"):
                cands.append(f)
    if not cands:
        cands = list(pool)

    cands = sorted(cands, key=sort_key_deterministic, reverse=True)
    return cands[0], ""
