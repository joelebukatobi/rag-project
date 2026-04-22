from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

# Aligned with Implementation.md (five views only; no custom personas).
VIEWS: Tuple[str, ...] = (
    "Board",
    "Risk Analyst",
    "Research",
    "Legal & Compliance",
    "Regulatory / Auditor",
)

DEFAULT_VIEW = "Risk Analyst"

REPORT_PROMPT_VERSION = "v5_views_2026-04-22"


def normalize_view(view: Optional[str]) -> str:
    v = (view or DEFAULT_VIEW).strip()
    if v in VIEWS:
        return v
    return DEFAULT_VIEW


def _base_findings() -> List[Dict[str, Any]]:
    return [
        {
            "category": "STRATEGIC | REGULATORY | FINANCIAL | GOVERNANCE",
            "materiality": "HIGH | MEDIUM | LOW | MARKET",
            "title": "Finding Headline",
            "evidence": "Verifiable fact with exact figures or short quotes; cite chunk ID(s) in source.",
            "verdict": "Conclusion citing applicable gatekeeper",
            "source": "Chunk ID(s)",
        }
    ]


def build_schema_for_view(
    view: str,
) -> Dict[str, Any]:
    base = {
        "view": view,
        "executive_summary": "BLUF; include figures where supported by evidence.",
        "strategic_outlook": {
            "primary_driver": "Key delta with specific figures or explicit uncertainty",
            "net_posture": "STABLE | CAUTIONARY | IMPROVING",
            "liquidity_buffer": "5x / 10% status per gatekeepers (or 'not applicable')",
        },
        "findings": _base_findings(),
    }

    if view == "Board":
        return {
            **base,
            "findings": _base_findings(),
        }

    if view == "Research":
        return {
            **base,
            "meta_signals": {
                "disclosure_growth_pct": "Filled by system from section lengths; echo here",
                "section_char_len_year_a": "number",
                "section_char_len_year_b": "number",
                "trend_narrative": "2–4 sentences on multi-year pattern; cite sources",
            },
        }

    if view == "Legal & Compliance":
        return {
            **base,
            "judge_scores": {
                "citation_chunk_coverage_0_100": "integer 0-100: share of claims tied to provided chunk text",
                "verbatim_quote_or_number_presence_0_100": "integer 0-100",
                "speculative_language_flag": "LOW | MEDIUM | HIGH",
                "rationale": "1–2 sentences, grounded in evidence",
            },
        }

    if view == "Regulatory / Auditor":
        return {
            **base,
            "omission_analysis": {
                "removed_or_slimmed_topics": [
                    {
                        "topic": "string",
                        "evidence": "What Year A had vs B; quote or figure; source chunk ID(s)",
                    }
                ],
                "added_or_expanded_topics": [
                    {
                        "topic": "string",
                        "evidence": "string",
                    }
                ],
            },
        }

    return base


def view_role_block(view: str) -> str:
    if view == "Board":
        return (
            "ROLE: Board / IC reader. Be concise, decision-ready, BLUF first. "
            "Cap findings at 5; prioritize strategic and high materiality. "
        )
    if view == "Research":
        return (
            "ROLE: Research / trends. Emphasize multi-year pattern and disclosure size evolution (Meta #18). "
        )
    if view == "Legal & Compliance":
        return (
            "ROLE: Model-risk / faithfulness. Score how well claims are grounded; "
            "flag speculation; require chunk-backed sources. "
        )
    if view == "Regulatory / Auditor":
        return (
            "ROLE: Disclosures / audit lens. Emphasize omissions, removals, and expansion of topic coverage year-over-year. "
        )
    return (
        "ROLE: Credit risk analyst. Emphasize material changes and evidence, with full findings list. "
    )


def schema_as_json_str(schema: Dict[str, Any]) -> str:
    return json.dumps(schema, ensure_ascii=False, indent=2)
