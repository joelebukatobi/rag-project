from __future__ import annotations

import json
from typing import Dict, List, Tuple

from openai import OpenAI

from src.retrieve import HybridRetriever


def _build_context(chunks: List[Dict[str, object]], label: str) -> str:
    lines = [f"### {label}"]
    for i, c in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] ticker={c['ticker']} year={c['year']} section={c['section_type']} chunk_id={c['chunk_id']}"
        )
        lines.append(str(c["text"])[:1600])
    return "\n".join(lines)


_SECTION_PROMPTS = {
    "section_1a": (
        "You are analyzing SEC 10-K Risk Factors (Item 1A) filings.\n"
        "Compare the two sets of excerpts and provide findings for ALL of the following:\n"
        "1) Risk Changes: new risks added, risks removed, risks expanded or reduced\n"
        "2) Revenue Drivers: any mentions of revenue segments, demand trends, or growth drivers\n"
        "3) Litigation Changes: any legal proceedings, lawsuits, or regulatory actions mentioned\n"
        "4) Tone Shifts: changes in cautious vs optimistic language\n"
    ),
    "section_7": (
        "You are analyzing SEC 10-K MD&A (Item 7) filings.\n"
        "Compare the two sets of excerpts and provide findings for ALL of the following:\n"
        "1) Revenue Drivers: changes in revenue segments, product lines, geographic markets, demand trends\n"
        "2) Risk Changes: operational or financial risks discussed in the management analysis\n"
        "3) Litigation Changes: any legal or regulatory matters impacting financial performance\n"
        "4) Tone Shifts: changes in management's outlook, confidence, or forward-looking language\n"
    ),
    "section_3": (
        "You are analyzing SEC 10-K Legal Proceedings (Item 3) filings.\n"
        "Compare the two sets of excerpts and provide findings for ALL of the following:\n"
        "1) Litigation Changes: lawsuits added, resolved, ongoing, or escalated\n"
        "2) Risk Changes: legal risks that could impact the business\n"
        "3) Revenue Drivers: any financial impact or settlements affecting revenue\n"
        "4) Tone Shifts: changes in how the company characterizes legal exposure\n"
    ),
}


def compare(
    retriever: HybridRetriever,
    ticker: str,
    section_type: str,
    year_a: int,
    year_b: int,
    query: str,
    top_k: int = 5,
    model: str = "gpt-4o",
) -> Dict[str, object]:
    """Retrieve same-section chunks across years and produce a structured narrative diff."""
    client = OpenAI()

    chunks_a = retriever.retrieve(query=query, ticker=ticker, year=year_a, section_type=section_type, top_k=top_k)
    chunks_b = retriever.retrieve(query=query, ticker=ticker, year=year_b, section_type=section_type, top_k=top_k)

    section_guidance = _SECTION_PROMPTS.get(section_type, _SECTION_PROMPTS["section_1a"])

    prompt = f"""
{section_guidance}
For each finding, cite the relevant chunk_id. If a category has no findings, explicitly state "No significant changes found."

{_build_context(chunks_a, f'Year {year_a}')}

{_build_context(chunks_b, f'Year {year_b}')}
""".strip()

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a rigorous SEC filings analyst."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    text = response.output_text
    return {
        "ticker": ticker,
        "section_type": section_type,
        "year_a": year_a,
        "year_b": year_b,
        "chunks_a": chunks_a,
        "chunks_b": chunks_b,
        "raw_diff": text,
    }


def _extract_json(text: str) -> Dict[str, object]:
    """Best-effort JSON parsing with fallback for fenced output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()
    return json.loads(cleaned)


def _validate_output(payload: Dict[str, object]) -> Tuple[bool, str]:
    required_top = [
        "company",
        "filing_year_a",
        "filing_year_b",
        "risk_changes",
        "revenue_drivers",
        "litigation_changes",
        "tone_shift",
        "citations",
    ]
    for key in required_top:
        if key not in payload:
            return False, f"Missing key: {key}"

    if not isinstance(payload.get("risk_changes"), list):
        return False, "risk_changes must be a list"
    if not isinstance(payload.get("citations"), list):
        return False, "citations must be a list"

    tone = payload.get("tone_shift", {})
    if not isinstance(tone, dict) or "direction" not in tone:
        return False, "tone_shift must include direction"

    return True, "ok"


def generate_structured_output(
    diff_result: Dict[str, object],
    model: str = "gpt-4o",
) -> Dict[str, object]:
    """Generate strict JSON output from comparative diff, with robust error handling."""
    client = OpenAI()

    schema_hint = {
        "company": "string",
        "filing_year_a": "int",
        "filing_year_b": "int",
        "risk_changes": [
            {
                "risk": "string",
                "change_type": "Expanded | New | Removed | Unchanged",
                "evidence": "string",
                "confidence": "float",
            }
        ],
        "revenue_drivers": [
            {
                "segment": "string",
                "trend": "Increased | Decreased | Stable",
                "evidence": "string",
            }
        ],
        "litigation_changes": [
            {
                "item": "string",
                "change_type": "Added | Resolved | Ongoing",
                "evidence": "string",
            }
        ],
        "tone_shift": {
            "direction": "More Optimistic | More Cautious | Neutral",
            "evidence": "string",
            "confidence": "float",
        },
        "citations": ["string"],
    }

    section_type = diff_result.get("section_type", "section_1a")
    section_names = {
        "section_1a": "Risk Factors (Item 1A)",
        "section_7": "MD&A (Item 7)",
        "section_3": "Legal Proceedings (Item 3)",
    }
    section_name = section_names.get(section_type, "SEC Filing")

    prompt = f"""
Transform the following {section_name} comparison into strict JSON matching this schema exactly.
ALL array fields (risk_changes, revenue_drivers, litigation_changes) MUST be populated with
any relevant findings from the comparison. Extract and categorize every finding into the
appropriate field. Do not leave arrays empty if the comparison text contains relevant information.
Do not add markdown. Return JSON only.

Schema:
{json.dumps(schema_hint, indent=2)}

Comparison:
{diff_result.get('raw_diff', '')}
""".strip()

    try:
        response = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": "Return strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        payload = _extract_json(response.output_text)
        valid, reason = _validate_output(payload)
        if not valid:
            return {
                "error": "validation_failed",
                "details": reason,
                "raw": response.output_text,
            }
        return payload
    except json.JSONDecodeError as exc:
        return {
            "error": "malformed_json",
            "details": str(exc),
            "raw": response.output_text if "response" in locals() else "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "error": "generation_failed",
            "details": str(exc),
            "raw": "",
        }
