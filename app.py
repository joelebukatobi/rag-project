from __future__ import annotations

import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import textwrap
from edgar import Company, set_identity

# Custom Logic Imports
from src.chunk import build_chunks
from src.embed import embed_chunks
from src.filing_select import accession_to_slug, pick_filing_for_year
from src.generate import generate_report
from src.retrieve import HybridRetriever

from dotenv import load_dotenv

load_dotenv()

# SEC Identity - Required for API access
set_identity("cg77@fordham.edu")
CACHE_DIR = "data/filings"
os.makedirs(CACHE_DIR, exist_ok=True)

# --- UI CONFIGURATION ---
st.set_page_config(page_title="SEC Credit Intelligence", layout="centered")


def inject_executive_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Merriweather:wght@700&family=Space+Grotesk:wght@400;500;600&display=swap');

            /* Simple black & white canvas */
            html, body, [data-testid="stAppViewContainer"] { background: #ffffff !important; }
            html, body, [class*="st-"] { font-family: 'Space Grotesk', sans-serif; color: #0a0a0a; }
            h1, h2, h3, h4, h5 { font-family: 'Merriweather', serif; color: #0a0a0a; }

            /* Hide default chrome spacing; keep it airy */
            [data-testid="stHeader"] { background: transparent; }
            footer { visibility: hidden; }
            .block-container { padding-bottom: 3.5rem; }

            /* Fixed author footer */
            .author-footer {
                position: fixed;
                bottom: 0;
                left: 50%;
                transform: translateX(-50%);
                width: 100%;
                max-width: 960px;
                text-align: center;
                font-size: 0.82rem;
                color: #6b7280;
                background: #ffffff;
                border-top: 1px solid #e5e5e5;
                padding: 0.6rem 1rem;
                z-index: 999;
                font-family: 'Space Grotesk', sans-serif;
                box-sizing: border-box;
            }
            .author-footer a {
                color: #0a0a0a;
                font-weight: 600;
                text-decoration: none;
            }
            .author-footer a:hover { text-decoration: underline; }
            .block-container { padding-top: 3.25rem; max-width: 960px; margin: 0 auto; }

            /* Form border to match card radius */
            [data-testid="stForm"] {
                border: 1px solid #e5e5e5 !important;
                border-radius: 14px !important;
                padding: 1.25rem !important;
            }

            .muted { color: #5a5a5a; }

            /* ---- FORM CONTROLS: force white bg + black text ---- */
            /* Labels */
            label, .stTextInput label, .stTextArea label, .stSelectbox label {
                color: #0a0a0a !important;
            }

            /* Text inputs + textarea */
            [data-testid="stTextInput"] input,
            [data-testid="stTextArea"] textarea,
            .stTextArea textarea,
            textarea {
                background: #ffffff !important;
                color: #0a0a0a !important;
                border: 1px solid #e5e5e5 !important;
                border-radius: 12px !important;
            }
            [data-testid="stTextArea"] textarea:focus,
            .stTextArea textarea:focus,
            textarea:focus {
                outline: none !important;
                box-shadow: 0 0 0 2px rgba(0,0,0,0.08) !important;
            }

            /* Selectbox (BaseWeb) */
            [data-baseweb="select"] > div {
                background: #ffffff !important;
                color: #0a0a0a !important;
                border: 1px solid #e5e5e5 !important;
                border-radius: 12px !important;
            }
            [data-baseweb="select"] span,
            [data-baseweb="select"] input,
            [data-baseweb="select"] div {
                color: #0a0a0a !important;
            }

            /* Dropdown menu options — let the light theme handle colors */
            [data-baseweb="menu"] [role="option"]:hover,
            [role="listbox"] [role="option"]:hover {
                background: #f3f4f6 !important;
            }

            /* Placeholder text */
            input::placeholder, textarea::placeholder {
                color: #6b7280 !important;
            }

            /* Button should stay black */
            .stButton > button {
                background: #0a0a0a !important;
                color: #ffffff !important;
                border: 1px solid #0a0a0a !important;
                border-radius: 12px !important;
                font-weight: 600 !important;
            }

            /* Executive summary + findings */
            .bluf-card { border: 1px solid #e5e5e5; border-radius: 14px; padding: 20px; background: #ffffff; }
            .bluf-title { font-family: 'Merriweather', serif; font-size: 1.1rem; margin-bottom: 10px; }
            .bluf-content { font-size: 1.0rem; line-height: 1.65; color: #111; }

            .exec-card { border: 1px solid #e5e5e5; padding: 16px; margin-bottom: 12px; border-radius: 12px; background: #ffffff; }
            .card-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
            .card-label { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
            .card-status { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; padding: 2px 10px; border: 1px solid #0a0a0a; border-radius: 999px; }
            .card-body { font-size: 0.95rem; line-height: 1.55; }

            /* Materiality tags: colored backgrounds + white text */
            .status-high { background: #dc2626; color: #ffffff; border-color: #dc2626; }
            .status-medium { background: #f59e0b; color: #ffffff; border-color: #f59e0b; }
            .status-low { background: #16a34a; color: #ffffff; border-color: #16a34a; }
            .status-market { background: #2563eb; color: #ffffff; border-color: #2563eb; }

            /* Posture tags */
            .status-cautionary { background: #f59e0b; color: #ffffff; border-color: #f59e0b; }
            .status-stable { background: #16a34a; color: #ffffff; border-color: #16a34a; }
            .status-improving { background: #2563eb; color: #ffffff; border-color: #2563eb; }

            /* Results anchor spacing */
            .results-anchor { scroll-margin-top: 24px; }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_exec_card(data: Dict[str, Any], *, show_category_label: bool = True):
    materiality = str(data.get("materiality", "Low")).upper()
    m_class = f"status-{materiality.lower()}"
    category_html = ""
    if show_category_label:
        category_html = f"<span class='card-label'>{str(data.get('category', 'General')).upper()}</span>"
    # Important: no leading indentation on lines, otherwise Streamlit may render as a code block.
    html = (
        f"<div class=\"exec-card\">"
        f"<div class=\"card-header\">"
        f"{category_html}"
        f"<span class=\"card-status {m_class}\">{materiality} MATERIALITY</span>"
        f"</div>"
        f"<div class=\"card-body\">{data.get('evidence', 'No specific snippet found.')}</div>"
        f"<div style=\"margin-top: 10px; padding: 8px; background: #f8fafc; border-left: 3px solid #1c2e4a; "
        f"font-size: 0.85rem; font-weight: 600;\">"
        f"{data.get('verdict', 'Monitor for developments.')}"
        f"</div>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def _extract_section_text_with_tier(tenk_obj, section_id: str):
    """Return (text, tier_name) for provenance."""
    item_id = section_id.replace("section_", "Item ").upper()
    attr_map = {
        "section_1": "business",
        "section_1a": "risk_factors",
        "section_3": "legal_proceedings",
        "section_7": "management_discussion",
        "section_8": "financial_statements",
    }
    raw_content = getattr(tenk_obj, attr_map.get(section_id, ""), "")
    tier = "attr"
    if not raw_content or len(str(raw_content)) < 500:
        try:
            raw_content = tenk_obj[item_id]
            tier = "item_index"
        except Exception:
            raw_content = tenk_obj.section(item_id)
            tier = "section"
    return str(raw_content), tier


@st.cache_data(show_spinner=False)
def get_filing_data(
    ticker: str,
    year: int,
    section_id: str,
    use_amended_10k: bool = False,
):
    def _provenance_dict(filing, extraction_tier: str, raw_len: int) -> Dict[str, Any]:
        por = getattr(filing, "period_of_report", None) or getattr(
            filing, "report_date", None
        )
        fd = getattr(filing, "filing_date", None) or getattr(filing, "date", None)
        acc = getattr(filing, "accession_no", None)
        form = str(getattr(filing, "form", None) or "")
        return {
            "ticker": ticker.upper(),
            "fiscal_year": int(year),
            "section_id": section_id,
            "accession_no": str(acc) if acc is not None else "",
            "form": form,
            "period_of_report": str(por) if por is not None else "",
            "filing_date": str(fd) if fd is not None else "",
            "extraction_tier": extraction_tier,
            "section_char_len": int(raw_len),
        }

    def _load_legacy() -> Any:
        file_id = f"{ticker}_{year}_{section_id}"
        chunk_path = f"{CACHE_DIR}/{file_id}_chunks.pkl"
        vec_path = f"{CACHE_DIR}/{file_id}_vecs.pkl"
        if not (os.path.exists(chunk_path) and os.path.exists(vec_path)):
            return None
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)
        with open(vec_path, "rb") as f:
            p = pickle.load(f)
            vecs = p["vectors"] if isinstance(p, dict) else p[0]
        prov: Dict[str, Any] = {
            "ticker": ticker.upper(),
            "fiscal_year": int(year),
            "section_id": section_id,
            "accession_no": "",
            "form": "",
            "period_of_report": "",
            "filing_date": "",
            "extraction_tier": "legacy_cache",
            "section_char_len": int(
                next((m.get("meta_section_len", 0) for m in chunks), 0) or 0
            ),
            "cache_note": "Pre-accession cache file; re-fetch will pin accession.",
        }
        return {
            "chunks": chunks,
            "vecs": vecs,
            "error": None,
            "provenance": prov,
        }

    try:
        company = Company(ticker)
        if use_amended_10k:
            f10 = company.get_filings(form="10-K")
            f10a = company.get_filings(form="10-K/A")
            all_filings = list(f10) + list(f10a)
        else:
            all_filings = list(company.get_filings(form="10-K"))
        if not all_filings:
            return {
                "chunks": None,
                "vecs": None,
                "error": f"No 10-K filings found for {ticker}.",
                "provenance": None,
            }

        filing, pick_err = pick_filing_for_year(
            all_filings, int(year), use_amended_10k=use_amended_10k
        )
        if not filing:
            leg = _load_legacy()
            if leg is not None:
                return leg
            return {
                "chunks": None,
                "vecs": None,
                "error": pick_err
                or f"Could not select a 10-K for FY{year} ({ticker}).",
                "provenance": None,
            }

        acc = getattr(filing, "accession_no", None) or "unknown"
        acc_slug = accession_to_slug(str(acc))
        file_id = f"{ticker}_{year}_{section_id}_{acc_slug}"
        chunk_path = f"{CACHE_DIR}/{file_id}_chunks.pkl"
        vec_path = f"{CACHE_DIR}/{file_id}_vecs.pkl"
        prov_path = f"{CACHE_DIR}/{file_id}_provenance.json"

        if os.path.exists(chunk_path) and os.path.exists(vec_path):
            with open(chunk_path, "rb") as f:
                chunks = pickle.load(f)
            with open(vec_path, "rb") as f:
                p = pickle.load(f)
                vecs = p["vectors"] if isinstance(p, dict) else p[0]
            provenance: Dict[str, Any]
            if os.path.exists(prov_path):
                with open(prov_path, "r", encoding="utf-8") as f:
                    provenance = json.load(f)
            else:
                provenance = {
                    "ticker": ticker.upper(),
                    "fiscal_year": int(year),
                    "section_id": section_id,
                    "accession_no": str(acc),
                    "extraction_tier": "cache_hit",
                    "section_char_len": int(
                        next((m.get("meta_section_len", 0) for m in chunks), 0) or 0
                    ),
                }
            return {
                "chunks": chunks,
                "vecs": vecs,
                "error": None,
                "provenance": provenance,
            }

        tenk = filing.obj()
        raw_text, ext_tier = _extract_section_text_with_tier(tenk, section_id)
        if len(raw_text) < 200:
            por = getattr(filing, "period_of_report", None) or getattr(
                filing, "report_date", None
            )
            fd = getattr(filing, "filing_date", None) or getattr(filing, "date", None)
            acc2 = getattr(filing, "accession_no", None)
            return {
                "chunks": None,
                "vecs": None,
                "error": (
                    f"Extracted text too short for {ticker} FY{year} {section_id} "
                    f"(len={len(raw_text)}). Filing: accession={acc2}, period_of_report={por}, "
                    f"filing_date={fd}. Enable 'Use amended 10-K/A' only if you intend to use an amended filing."
                ),
                "provenance": _provenance_dict(filing, ext_tier, len(raw_text)),
            }

        prov_save = _provenance_dict(filing, ext_tier, len(raw_text))
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov_save, f, indent=2)

        df_temp = pd.DataFrame(
            [
                {
                    "ticker": ticker,
                    "year": year,
                    "section_type": section_id,
                    "section_text": raw_text,
                    "section_char_len": len(raw_text),
                }
            ]
        )

        chunks = build_chunks(df_temp, cache_path=chunk_path)
        vecs, metadata = embed_chunks(chunks, cache_path=vec_path)
        return {
            "chunks": metadata,
            "vecs": vecs,
            "error": None,
            "provenance": prov_save,
        }

    except Exception as e:
        print(f"Error fetching {ticker} {section_id}: {e}")
        leg = _load_legacy()
        if leg is not None:
            return leg
        return {
            "chunks": None,
            "vecs": None,
            "error": f"Exception: {type(e).__name__}: {e}",
            "provenance": None,
        }


# --- APP INTERFACE ---
inject_executive_css()

st.markdown(
    "<div class='author-footer'>"
    "By <a href='https://github.com/gisemba' target='_blank'>Claudia Gisemba</a>"
    " and <a href='https://github.com/joelebukatobi' target='_blank'>Joel Onwuanaku</a>"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='font-size: 2.1rem; margin-bottom: 0.25rem; text-align: center;'>Credit Intelligence Terminal</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='muted' style='margin-bottom: 0.25rem; text-align: center;'>"
    "Configure your analysis and generate an underwriting-style comparative report."
    "</div>"
    "<div style='font-size: 0.82rem; color: #6b7280; margin-bottom: 1.25rem; text-align: center;'>"
    "Pre-indexed: <strong>AAPL, NKE</strong> (2020–2025) — these load instantly. "
    "Any other ticker will be indexed on first run."
    "</div>",
    unsafe_allow_html=True,
)

SECTION_OPTIONS = [
    "Business (Item 1)",
    "Risk Factors (Item 1A)",
    "MD&A (Item 7)",
    "Legal Proceedings (Item 3)",
    "Financial Statements (Item 8)",
]

with st.container():

    with st.form("controls", clear_on_submit=False):
        c1, c2 = st.columns([1, 1], gap="large")
        with c1:
            ticker = st.text_input(
                "Ticker",
                value="NKE",
                placeholder="e.g., AAPL, GOOG, NKE",
            ).strip().upper()
        with c2:
            section_label = st.selectbox("Section", SECTION_OPTIONS, index=1)

        view_label = st.selectbox(
            "View",
            [
                "Board",
                "Risk Analyst",
                "Research",
                "Legal & Compliance",
                "Regulatory / Auditor",
            ],
            index=1,
            help="Each view uses a different report shape and UI emphasis; see Implementation.md.",
        )
        use_amended_10k = st.checkbox(
            "Use amended 10-K/A (optional; default is base 10-K only)",
            value=False,
        )

        y1, y2 = st.columns(2, gap="large")
        years = list(range(2018, 2026))
        with y1:
            y_a = st.selectbox("Base Year", years, index=years.index(2020))
        with y2:
            y_b = st.selectbox("Target Year", years, index=years.index(2023))

        query = st.text_area(
            "Underwriting focus",
            value="major risk, liquidity, and covenant changes",
            height=90,
            placeholder="What should the model focus on?",
        )

        run = st.form_submit_button("Generate report")

section_map = {
    "Business (Item 1)": "section_1",
    "Risk Factors (Item 1A)": "section_1a",
    "MD&A (Item 7)": "section_7",
    "Legal Proceedings (Item 3)": "section_3",
    "Financial Statements (Item 8)": "section_8",
}


def _scroll_to_results() -> None:
    components.html(
        """
        <script>
          const el = window.parent.document.getElementById('results');
          if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
        """,
        height=0,
    )


if run:
    if not ticker or not ticker.isalnum() or len(ticker) > 6:
        st.error("Please enter a valid ticker symbol (1–6 letters/numbers).")
        st.stop()
    if y_a == y_b:
        st.error("Please choose two different years.")
        st.stop()

    with st.spinner(f"Analyzing {ticker} FY{y_a} vs FY{y_b}..."):
        a = get_filing_data(
            ticker, y_a, section_map[section_label], use_amended_10k=use_amended_10k
        )
        b = get_filing_data(
            ticker, y_b, section_map[section_label], use_amended_10k=use_amended_10k
        )
        chunks_a, vecs_a = a.get("chunks"), a.get("vecs")
        chunks_b, vecs_b = b.get("chunks"), b.get("vecs")

        if chunks_a is None or chunks_b is None:
            st.error("Filing data unavailable for one or both years/sections.")
            if a.get("error"):
                st.caption(f"Base year error (FY{y_a}, {section_map[section_label]}): {a['error']}")
            if b.get("error"):
                st.caption(f"Target year error (FY{y_b}, {section_map[section_label]}): {b['error']}")
            st.stop()

        # Build Hybrid Retriever (Fast because model is already in RAM)
        retriever = HybridRetriever(
            vectors=np.vstack([vecs_a, vecs_b]), metadata=chunks_a + chunks_b
        )

        prov_a = a.get("provenance") or {}
        prov_b = b.get("provenance") or {}
        len_a = int(
            prov_a.get("section_char_len", 0)
            or next((m.get("meta_section_len", 0) for m in chunks_a), 0)
        )
        len_b = int(
            prov_b.get("section_char_len", 0)
            or next((m.get("meta_section_len", 0) for m in chunks_b), 0)
        )

        report = generate_report(
            retriever=retriever,
            ticker=ticker,
            section_type=section_map[section_label],
            year_a=int(y_a),
            year_b=int(y_b),
            query=query,
            view=view_label,
            provenance={"year_a": prov_a, "year_b": prov_b},
            section_len_a=len_a,
            section_len_b=len_b,
        )

        # --- RENDER REPORT ---
        st.markdown(
            "<div id='results' class='results-anchor'></div>", unsafe_allow_html=True
        )
        _scroll_to_results()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e5e5; margin: 2rem 0;'>", unsafe_allow_html=True)

        st.markdown(
            f"<h2 style='font-size: 1.4rem; margin-top: 0;'>Comparative Review: {ticker} (FY{y_a} vs FY{y_b})</h2>"
            f"<div class='muted' style='margin: 0.25rem 0 0.5rem 0;'>View: <strong>{view_label}</strong></div>",
            unsafe_allow_html=True,
        )

        with st.expander("Provenance & filing", expanded=False):
            st.json(
                {
                    "year_a_filing": prov_a,
                    "year_b_filing": prov_b,
                    "report_provenance": report.get("provenance", {}),
                }
            )

        if len_a > 0:
            growth = ((len_b - len_a) / len_a) * 100
            st.markdown(
                f"<div class='muted' style='margin: 0.25rem 0 1.25rem 0;'>"
                f"<strong>Meta-Signal #18:</strong> disclosure size changed by <strong>{growth:+.1f}%</strong> "
                f"({len_a:,} → {len_b:,} chars)."
                f"</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div class="bluf-card"><div class="bluf-title">Executive Summary</div>'
            f'<div class="bluf-content">{report.get("executive_summary", "N/A")}</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e5e5; margin: 2rem 0;'>", unsafe_allow_html=True)

        # --- ROW LAYOUT (Outlook first, then deltas) ---
        st.markdown(
            "<h3 style='font-size: 1.1rem; margin-top: 0;'>Strategic Outlook</h3>",
            unsafe_allow_html=True,
        )
        outlook = report.get("strategic_outlook", {})
        posture = str(outlook.get("net_posture", "STABLE")).upper()
        posture_class = f"status-{posture.lower()}"
        driver = outlook.get("primary_driver", "N/A")
        outlook_html = (
            f"<div class=\"exec-card\">"
            f"<div class=\"card-header\">"
            f"<span class=\"card-label\">OUTLOOK</span>"
            f"<span class=\"card-status {posture_class}\">{posture}</span>"
            f"</div>"
            f"<div class=\"card-body\">{driver}</div>"
            f"</div>"
        )
        st.markdown(outlook_html, unsafe_allow_html=True)

        st.markdown(
            "<hr style='border: none; border-top: 1px solid #e5e5e5; margin: 1.5rem 0;'>",
            unsafe_allow_html=True,
        )

        st.markdown(
            "<h3 style='font-size: 1.1rem; margin-top: 0;'>Operational & Risk Deltas</h3>",
            unsafe_allow_html=True,
        )
        findings = list(report.get("findings", []) or [])
        category_order = ["STRATEGIC", "FINANCIAL", "REGULATORY", "GOVERNANCE"]
        # Order requested: highest risk first (red -> orange -> green -> market).
        materiality_order = ["HIGH", "MEDIUM", "LOW", "MARKET"]
        materiality_rank = {m: i for i, m in enumerate(materiality_order)}

        def _normalize_category(raw: object) -> str:
            s = str(raw or "").upper()
            # If the model outputs combined strings like "STRATEGIC | REGULATORY", pick a primary bucket.
            for key in category_order:
                if key in s:
                    return key
            return "OTHER"

        # Group by category, but order category sections by highest materiality present (Option 1).
        cat_buckets = {c: [] for c in category_order}
        cat_buckets["OTHER"] = []
        for f in findings:
            c = _normalize_category(f.get("category", ""))
            cat_buckets[c].append(f)

        def _bucket_best_rank(bucket: List[Dict[str, Any]]) -> int:
            if not bucket:
                return 999
            ranks = [
                materiality_rank.get(str(x.get("materiality", "")).upper().strip(), 999)
                for x in bucket
            ]
            return min(ranks) if ranks else 999

        ordered_categories = sorted(
            [c for c in category_order if cat_buckets.get(c)],
            key=lambda c: _bucket_best_rank(cat_buckets[c]),
        )
        if cat_buckets.get("OTHER"):
            ordered_categories.append("OTHER")

        for c in ordered_categories:
            c_bucket = cat_buckets.get(c, [])
            if not c_bucket:
                continue

            c_label = c if c != "OTHER" else "OTHER / UNCATEGORIZED"
            st.markdown(
                f"<div class='muted' style='margin: 0.25rem 0 0.75rem 0; font-weight: 800;'>"
                f"{c_label}</div>",
                unsafe_allow_html=True,
            )

            # Order findings within each category by materiality (highest risk first).
            c_bucket.sort(
                key=lambda x: materiality_rank.get(
                    str(x.get("materiality", "")).upper().strip(), 999
                )
            )
            for finding in c_bucket:
                # Avoid repeating the category inside each card since we're already grouped.
                render_exec_card(finding, show_category_label=False)

        if view_label == "Research" and report.get("meta_signals"):
            st.markdown(
                "<h3 style='font-size: 1.1rem; margin-top: 1.5rem;'>Multi-year meta-signals</h3>",
                unsafe_allow_html=True,
            )
            st.json(report["meta_signals"])

        if view_label == "Legal & Compliance" and report.get("judge_scores"):
            st.markdown(
                "<h3 style='font-size: 1.1rem; margin-top: 1.5rem;'>Faithfulness (model check)</h3>",
                unsafe_allow_html=True,
            )
            st.json(report["judge_scores"])

        if view_label == "Regulatory / Auditor" and report.get("omission_analysis"):
            st.markdown(
                "<h3 style='font-size: 1.1rem; margin-top: 1.5rem;'>Omission and gap analysis</h3>",
                unsafe_allow_html=True,
            )
            st.json(report["omission_analysis"])
else:
    st.markdown(
        "<div class='muted'>Enter your inputs above and generate a report.</div>",
        unsafe_allow_html=True,
    )

