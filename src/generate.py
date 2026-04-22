from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List, Optional

from openai import OpenAI
from src.retrieve import HybridRetriever
from src.view_schemas import (
    REPORT_PROMPT_VERSION,
    build_schema_for_view,
    normalize_view,
    schema_as_json_str,
    view_role_block,
)

def _build_context(chunks: List[Dict[str, object]], label: str) -> str:
    """Builds a structured context window for the model."""
    lines = [f"### {label} DATA"]
    for i, c in enumerate(chunks, start=1):
        # We now include the meta_section_len to help with Signal #18
        meta = f" (Section Length: {c.get('meta_section_len', 'N/A')} chars)"
        lines.append(
            f"[{i}] {c['ticker']} {c['year']} | {c['section_type']} | ID: {c['chunk_id']}{meta}"
        )
        lines.append(str(c["text"])[:2000])
    return "\n".join(lines)


# --- THE CREDIT SCORING RUBRIC (The "Law of the Land") ---
_UNDERWRITING_MATERIALITY_GUIDE = """
ANALYSIS GUIDELINES (NO SCORING):

1. LIQUIDITY CONTEXT (The 5x Rule):
   - If [Total Liquidity] > 5x the exposure of a specific risk (e.g., a lawsuit or vendor loss), 
     classify that risk as 'LOW MATERIALITY'. 
   - Reason: The company's balance sheet provides a sufficient safety buffer.

2. DISCLOSURE COMPLEXITY (Meta-Signal #18):
   - If YoY Disclosure Size increases by >20%, classify as 'MEDIUM MATERIALITY'.
   - Tag as: 'Increased Reporting Complexity'. 

3. FINANCIAL NOISE VS. STRESS:
   - Mark-to-market debt value changes (Fair Value) are 'MARKET MATERIALITY'.
   - Do NOT flag as 'High Materiality' unless paired with 'Refinancing' or 'Covenant' warnings.

4. LEGAL THRESHOLDS:
   - Classify lawsuits as 'LOW MATERIALITY' unless the contingency exceeds 10% of cash reserves.
   - If it exceeds 10%, classify as 'HIGH MATERIALITY'.

5. STRATEGIC SHIFTS (Item 1):
   - Any brand sunset, market exit, or segment consolidation is 'MEDIUM MATERIALITY'.
"""

# --- TAXONOMY-DRIVEN PROMPTS ---
_TAXONOMY_INSTRUCTIONS = """
Analyze the provided SEC excerpts using the Credit Signal Taxonomy (v1). 
Focus on identifying the following signal clusters:
1. BUSINESS FRAGILITY: Customer concentration, supply chain shifts, competitive pressure.
2. RISK INTENSITY: New risk factors, expanded risk language, regulatory escalation.
3. LIQUIDITY STRESS: Phrases like 'may not be sufficient', refinancing risk, capital allocation shifts.
4. MANAGEMENT TONE: Increased hedging language, specificity shifts (vague to specific), or sentiment decay.
5. CROSS-SECTION CONTRADICTIONS: Mismatches between management optimism and risk disclosures.

STRICT CREDIT ANALYSIS RULES (FAIR VALUE GATEKEEPER):
- If you detect a decrease in the 'Fair Value' of debt in Item 8:
  1. CROSS-CHECK Item 7 (MD&A) and Item 1A (Risk) for terms like 'liquidity', 'refinancing', 'covenant', or 'facility access'.
  2. IF AND ONLY IF those terms indicate actual stress, flag it as a 'Credit Stress Signal'.
  3. OTHERWISE, categorize it as 'Market Risk (Interest Rate-driven)' and explicitly state that no firm-specific liquidity stress was found.

STRICT LEGAL ANALYSIS RULES (ITEM 3):
- Detect expansion in legal proceedings (more cases, higher complexity, or new jurisdictions).
- CROSS-CHECK Cash & Cash Equivalents in Item 8.
- GATEKEEPER: Do NOT flag legal proceedings as 'Liquidity Stress' unless:
  1. Estimated loss contingencies are quantified and exceed 10% of current cash/ST investments.
  2. Management explicitly mentions legal settlements as a risk to debt covenants.
  3. There is an 'Emphasis of Matter' regarding legal going-concern risks.
- OTHERWISE: Label as 'Increased Regulatory/Legal Headwinds' or 'Business Fragility' without implying a credit liquidity event.

STRICT CREDIT ANALYSIS RULES:
{_UNDERWRITING_MATERIALITY_GUIDE}
"""

def compare(
    retriever: HybridRetriever,
    ticker: str,
    section_type: str,
    year_a: int,
    year_b: int,
    query: str,
    top_k: int = 10, # Increased k to capture more taxonomy signals
    model: str = "gpt-4o-mini",
) -> Dict[str, object]:
    client = OpenAI()

    # Retrieve comparative chunks
    chunks_a = retriever.retrieve(query=query, ticker=ticker, year=year_a, section_type=section_type, top_k=top_k)
    chunks_b = retriever.retrieve(query=query, ticker=ticker, year=year_b, section_type=section_type, top_k=top_k)

    prompt = f"""
        ROLE: Senior Credit Underwriter
        TASK: Delta Analysis of SEC filings for {ticker} ({year_a} vs {year_b}).
        INSTRUCTIONS:
        {_TAXONOMY_INSTRUCTIONS}

        EVIDENCE REQUIREMENTS:
        - Cite specific chunk IDs for every signal detected.
        - If a section length (Meta Signal #18) has changed significantly, flag it.
        - Distinguish between 'Boilerplate' and 'Material' changes.
        - Distinguish between 'Market Risk' and 'Credit Stress' per the Rules above.

        VERIFIABILITY MANDATE (STRICT):
        - Every finding MUST include at least one specific, Google-verifiable data point.
        - Include exact dollar amounts (e.g., "$31.9B total debt"), percentages (e.g., "revenue up 37% YoY"),
          dates, entity/counterparty names, regulatory body names, case numbers, or geographic jurisdictions.
        - Quote exact phrases from the filing where possible, in single quotes.
        - If the filing states a specific figure, include it verbatim — do NOT paraphrase numbers.
        - Preference: a reader should be able to copy a key phrase from your finding, paste it into Google,
          and find a corroborating source (earnings report, news article, court filing).

        ANTI-VAGUENESS RULE:
        - NEVER output a finding that merely restates boilerplate filing language.
        - If the filing mentions a regulation, NAME the specific regulation (e.g., "SEC Rule 15c3-3", "EU GDPR", "CCPA").
        - If the filing mentions lawsuits, NAME the parties, jurisdiction, and amounts if available.
        - If the filing mentions a market/product, NAME the specific market, product line, or geography.
        - If the source text is genuinely vague, you MUST still add context: state WHAT the vague language
          most likely refers to based on the company's known business, and flag it as 'Boilerplate — low
          analytical value' so the reader knows it lacks specificity.
        - BAD example: "The regulation of certain transactions involving virtual goods and cryptocurrencies"
        - GOOD example: "Nike's 2023 10-K added new risk language around digital asset regulation, likely
          referencing Nike's .SWOOSH NFT platform launched in Nov 2022. No specific regulatory action cited —
          this appears to be precautionary boilerplate ahead of anticipated SEC digital asset rulemaking."

    {_build_context(chunks_a, f'BASE YEAR {year_a}')}

    {_build_context(chunks_b, f'TARGET YEAR {year_b}')}
    """.strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a specialized Credit Analytics Engine (CreditDelta)."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    cids_a = sorted(
        {str(c.get("chunk_id", "")) for c in chunks_a if c.get("chunk_id")}
    )
    cids_b = sorted(
        {str(c.get("chunk_id", "")) for c in chunks_b if c.get("chunk_id")}
    )
    return {
        "ticker": ticker,
        "section_type": section_type,
        "year_a": year_a,
        "year_b": year_b,
        "raw_diff": response.choices[0].message.content,
        "chunk_ids_a": cids_a,
        "chunk_ids_b": cids_b,
    }


def _report_cache_path(cache_dir: str, key: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{key}.pkl")


def _load_report_cache(cache_dir: str, key: str) -> Optional[Dict[str, Any]]:
    path = _report_cache_path(cache_dir, key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _save_report_cache(cache_dir: str, key: str, report: Dict[str, Any]) -> None:
    path = _report_cache_path(cache_dir, key)
    with open(path, "wb") as f:
        pickle.dump(report, f)


def _fingerprint_key(payload: Dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def generate_structured_output(
    diff_result: Dict[str, object],
    model: str = "gpt-4o-mini",
    *,
    view: str = "Risk Analyst",
    section_type: str = "",
) -> Dict[str, Any]:
    client = OpenAI()
    v = normalize_view(view)
    schema_hint = build_schema_for_view(v)
    role = view_role_block(v)

    board_cap = ""
    if v == "Board":
        board_cap = "Limit 'findings' to at most 5 items. Prioritize highest materiality. "

    prompt = f"""
{role}
TASK: Convert the Delta Analysis into a structured report for view="{v}".

{board_cap}
UNDERWRITING GATEKEEPERS:
{_UNDERWRITING_MATERIALITY_GUIDE}

INSTRUCTIONS:
1. For every finding, apply the 5x Liquidity and 10% Legal thresholds.
2. If a risk is mitigated by liquidity (5x Rule), mark as 'LOW MATERIALITY' where applicable.
3. Categorize Interest-Rate debt fluctuations as 'MARKET MATERIALITY' when market-driven.
4. Provide the 'strategic_outlook' consistent with the evidence.
5. Every 'evidence' field must contain at least one verifiable fact, quote, or number from the analysis, with chunk ID(s) in 'source' where possible.
6. For view-specific keys (e.g. judge_scores, omission_analysis, meta_signals), only fill if part of the schema; use numbers 0-100 for scores where requested.

ANALYSIS DATA TO CONVERT:
{diff_result.get('raw_diff', '')}

OUTPUT FORMAT (strict JSON; match this structure, including optional keys for this view):
{schema_as_json_str(schema_hint)}
""".strip()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a specialized Credit Analytics Engine. Output only valid JSON matching the requested view schema.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )

    return json.loads(response.choices[0].message.content)


def _enrich_post(
    report: Dict[str, Any],
    view: str,
    *,
    section_len_a: int,
    section_len_b: int,
) -> Dict[str, Any]:
    v = normalize_view(view)
    if v == "Board":
        find = report.get("findings")
        if isinstance(find, list) and len(find) > 5:
            report["findings"] = find[:5]
    if v == "Research":
        ms = report.get("meta_signals")
        if not isinstance(ms, dict):
            report["meta_signals"] = {}
            ms = report["meta_signals"]
        growth = None
        if section_len_a:
            growth = (section_len_b - section_len_a) / float(section_len_a) * 100.0
        ms["section_char_len_year_a"] = int(section_len_a)
        ms["section_char_len_year_b"] = int(section_len_b)
        ms["disclosure_growth_pct"] = None if growth is None else round(growth, 2)
    return report


def _merge_provenance(
    report: Dict[str, Any],
    *,
    view: str,
    model: str,
    top_k: int,
    acc_a: str,
    acc_b: str,
    fin_fp: str,
) -> None:
    rep = report.get("provenance")
    if not isinstance(rep, dict):
        rep = {}
    rep.update(
        {
            "prompt_version": REPORT_PROMPT_VERSION,
            "view": normalize_view(view),
            "model": model,
            "top_k": top_k,
            "filing_accession_no_year_a": acc_a,
            "filing_accession_no_year_b": acc_b,
            "evidence_fingerprint_sha256": fin_fp,
        }
    )
    report["provenance"] = rep


def generate_report(
    retriever: HybridRetriever,
    ticker: str,
    section_type: str,
    year_a: int,
    year_b: int,
    query: str,
    top_k: int = 10,
    model: str = "gpt-4o-mini",
    view: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    section_len_a: int = 0,
    section_len_b: int = 0,
    cache_dir: str = "data/reports",
) -> Dict[str, Any]:
    """
    Two-step flow: compare() then view-specific generate_structured_output().
    Caches the final report JSON on evidence + view + version (see Implementation.md).
    """
    v = normalize_view(view)
    pa = str((provenance or {}).get("year_a", {}).get("accession_no", "")) if isinstance(provenance, dict) else ""  # noqa: E501
    pb = str((provenance or {}).get("year_b", {}).get("accession_no", "")) if isinstance(provenance, dict) else ""  # noqa: E501

    diff = compare(
        retriever=retriever,
        ticker=ticker,
        section_type=section_type,
        year_a=year_a,
        year_b=year_b,
        query=query,
        top_k=top_k,
        model=model,
    )
    cids_a = list(diff.get("chunk_ids_a") or [])
    cids_b = list(diff.get("chunk_ids_b") or [])

    fin_payload: Dict[str, Any] = {
        "report_prompt_version": REPORT_PROMPT_VERSION,
        "model": model,
        "view": v,
        "ticker": str(ticker).upper(),
        "section_type": section_type,
        "year_a": int(year_a),
        "year_b": int(year_b),
        "query": query,
        "top_k": int(top_k),
        "chunk_ids_a": cids_a,
        "chunk_ids_b": cids_b,
        "accession_a": pa,
        "accession_b": pb,
    }
    fin_fp = _fingerprint_key(fin_payload)
    cache_key = _fingerprint_key({"fingerprint": fin_fp, "role": "final_report_v1"})

    cached = _load_report_cache(cache_dir, cache_key)
    if cached is not None and isinstance(cached, dict) and cached.get("executive_summary") is not None:  # noqa: E501
        return cached

    struct = generate_structured_output(
        diff,
        model,
        view=v,
        section_type=section_type,
    )
    if not isinstance(struct, dict):
        struct = {"executive_summary": "", "findings": [], "strategic_outlook": {}}
    struct = _enrich_post(
        struct, v, section_len_a=section_len_a, section_len_b=section_len_b
    )
    struct["view"] = v

    if isinstance(provenance, dict):
        struct["filing"] = {
            "year_a": provenance.get("year_a"),
            "year_b": provenance.get("year_b"),
        }
    _merge_provenance(
        struct,
        view=v,
        model=model,
        top_k=top_k,
        acc_a=pa,
        acc_b=pb,
        fin_fp=fin_fp,
    )

    _save_report_cache(cache_dir, cache_key, struct)
    return struct
