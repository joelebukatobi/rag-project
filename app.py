from __future__ import annotations

import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Any

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from edgar import Company, set_identity

# Custom Logic Imports
from src.chunk import build_chunks
from src.embed import embed_chunks
from src.generate import compare, generate_structured_output
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

            /* Materiality tags: monochrome */
            .status-high { background: #0a0a0a; color: #ffffff; border-color: #0a0a0a; }
            .status-medium { background: #ffffff; color: #0a0a0a; border-color: #0a0a0a; }
            .status-low { background: #ffffff; color: #0a0a0a; border-color: #d0d0d0; }

            /* Results anchor spacing */
            .results-anchor { scroll-margin-top: 24px; }
        </style>
    """,
        unsafe_allow_html=True,
    )


def render_exec_card(data: Dict[str, Any]):
    materiality = str(data.get("materiality", "Low")).upper()
    m_class = f"status-{materiality.lower()}"
    st.markdown(
        f"""
        <div class="exec-card">
            <div class="card-header">
                <span class="card-label">{str(data.get("category", "General")).upper()}</span>
                <span class="card-status {m_class}">{materiality} MATERIALITY</span>
            </div>
            <div class="card-body">{data.get("evidence", "No specific snippet found.")}</div>
            <div style="margin-top: 10px; padding: 8px; background: #f8fafc; border-left: 3px solid #1c2e4a; font-size: 0.85rem; font-weight: 600;">
                {data.get("verdict", "Monitor for developments.")}
            </div>
        </div>
    """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def get_filing_data(ticker: str, year: int, section_id: str):
    file_id = f"{ticker}_{year}_{section_id}"
    chunk_path = f"{CACHE_DIR}/{file_id}_chunks.pkl"
    vec_path = f"{CACHE_DIR}/{file_id}_vecs.pkl"

    # 1. Instant Cache Return
    if os.path.exists(chunk_path) and os.path.exists(vec_path):
        with open(chunk_path, "rb") as f:
            chunks = pickle.load(f)
        with open(vec_path, "rb") as f:
            p = pickle.load(f)
            return (chunks, p["vectors"]) if isinstance(p, dict) else (chunks, p[0])

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K").filter(
            date=f"{year}-01-01:{year}-12-31"
        )
        if not filings:
            return None, None

        # Get the TenK object
        tenk = filings[0].obj()

        # Define the Item search string (e.g., 'Item 1' for 'section_1')
        item_id = section_id.replace("section_", "Item ").upper()

        # Tier 1: Try direct edgartools attribute
        attr_map = {
            "section_1": "business",
            "section_1a": "risk_factors",
            "section_3": "legal_proceedings",
            "section_7": "management_discussion",
            "section_8": "financial_statements",
        }
        raw_content = getattr(tenk, attr_map.get(section_id, ""), "")

        # Tier 2: If empty or too short, use the 'Item' indexer (The "NKE/TSLA" fix)
        if not raw_content or len(str(raw_content)) < 500:
            try:
                # This scans the actual document map for the Item header
                raw_content = tenk[item_id]
            except:
                # Tier 3: Hard slice
                raw_content = tenk.section(item_id)

        raw_text = str(raw_content)
        if len(raw_text) < 200:
            return None, None

        # --- PROCESS PIPELINE ---
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
        return metadata, vecs

    except Exception as e:
        print(f"Error fetching {ticker} {section_id}: {e}")
        return None, None


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

        y1, y2 = st.columns(2, gap="large")
        years = list(range(2018, 2024))
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
        chunks_a, vecs_a = get_filing_data(ticker, y_a, section_map[section_label])
        chunks_b, vecs_b = get_filing_data(ticker, y_b, section_map[section_label])

        if chunks_a is None or chunks_b is None:
            st.error(
                f"Filing data unavailable for {ticker}. SEC may not have indexed this section yet."
            )
            st.stop()

        # Build Hybrid Retriever (Fast because model is already in RAM)
        retriever = HybridRetriever(
            vectors=np.vstack([vecs_a, vecs_b]), metadata=chunks_a + chunks_b
        )

        diff = compare(
            retriever=retriever,
            ticker=ticker,
            section_type=section_map[section_label],
            year_a=int(y_a),
            year_b=int(y_b),
            query=query,
        )
        report = generate_structured_output(diff)

        # Meta-Signal Calculation
        len_a = next((m["meta_section_len"] for m in chunks_a), 0)
        len_b = next((m["meta_section_len"] for m in chunks_b), 0)

        # --- RENDER REPORT ---
        st.markdown(
            "<div id='results' class='results-anchor'></div>", unsafe_allow_html=True
        )
        _scroll_to_results()

        st.markdown("<hr style='border: none; border-top: 1px solid #e5e5e5; margin: 2rem 0;'>", unsafe_allow_html=True)

        st.markdown(
            f"<h2 style='font-size: 1.4rem; margin-top: 0;'>Comparative Review: {ticker} (FY{y_a} vs FY{y_b})</h2>",
            unsafe_allow_html=True,
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

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(
                "<h3 style='font-size: 1.1rem; margin-top: 0;'>Operational & Risk Deltas</h3>",
                unsafe_allow_html=True,
            )
            for finding in report.get("findings", []):
                render_exec_card(finding)

        with c_right:
            st.markdown(
                "<h3 style='font-size: 1.1rem; margin-top: 0;'>Strategic Outlook</h3>",
                unsafe_allow_html=True,
            )
            outlook = report.get("strategic_outlook", {})
            posture = str(outlook.get("net_posture", "STABLE")).upper()
            driver = outlook.get("primary_driver", "N/A")
            st.markdown(
                f"""
                <div class="exec-card">
                    <div class="card-header">
                        <span class="card-label">OUTLOOK</span>
                        <span class="card-status">{posture}</span>
                    </div>
                    <div class="card-body">{driver}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        "<div class='muted'>Enter your inputs above and generate a report.</div>",
        unsafe_allow_html=True,
    )

