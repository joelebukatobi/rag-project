from __future__ import annotations

import os
import json
from typing import Dict, List, Any

import dill as pickle
import numpy as np
import streamlit as st

from src.chunk import build_chunks, parse_sections
from src.embed import embed_chunks
from src.evaluate import build_test_set, run_retrieval_eval
from src.generate import compare, generate_structured_output
from src.ingest import load_live_filings
from src.retrieve import HybridRetriever

from dotenv import load_dotenv
load_dotenv()

# --- CONFIGURATION & PERMANENT UI SHIELD ---
st.set_page_config(page_title="SEC Fillings Credit Intelligence", layout="wide")

def inject_executive_css():
    """Injects a high-density, materiality-based UI. Removes all rating/scoring language."""
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville&family=Inter:wght@400;600;700&display=swap');
            
            .main { background-color: #f4f5f7; }
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }

            /* Top-Level Summary Card */
            .bluf-card {
                background-color: #ffffff;
                border-top: 4px solid #1c2e4a;
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .bluf-title { font-family: 'Libre Baskerville', serif; font-size: 1.4rem; color: #1c2e4a; margin-bottom: 10px; }
            .bluf-content { font-family: 'Libre Baskerville', serif; font-size: 1.1rem; line-height: 1.6; color: #333; }

            /* Strategic Delta Cards */
            .exec-card {
                background-color: #ffffff;
                border: 1px solid #e1e4e8;
                padding: 18px;
                margin-bottom: 12px;
                border-radius: 2px;
            }
            .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f0f0f0; padding-bottom: 8px; margin-bottom: 12px; }
            .card-label { font-size: 0.85rem; font-weight: 700; color: #1c2e4a; text-transform: uppercase; letter-spacing: 0.02em; }
            
            /* New Materiality Tags (Replacing Status) */
            .card-status { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border: 1px solid #1c2e4a; border-radius: 3px; }
            
            .card-body { font-size: 0.9rem; line-height: 1.5; color: #2d3748; }
            .card-evidence { font-size: 0.8rem; color: #718096; font-style: italic; margin-top: 10px; padding-top: 8px; border-top: 1px dashed #e1e4e8; }
            
            /* Materiality Level Styling - REPAIRED TO MATCH RENDERER */
            .status-high { color: #b91c1c; border-color: #b91c1c; background-color: #fef2f2; }    /* Significant Delta */
            .status-medium { color: #d97706; border-color: #d97706; background-color: #fffbeb; }  /* Notable Delta */
            .status-low { color: #047857; border-color: #047857; background-color: #f0fdf4; }     /* Routine Delta */
            .status-market { color: #2563eb; border-color: #2563eb; background-color: #eff6ff; }  /* External Delta */

            /* Sidebar Overrides */
            [data-testid="stSidebar"] { background-color: #1c2e4a !important; }
            .stSelectbox label, .stTextArea label, .stTextInput label { color: white !important; font-weight: 600; }

            .verdict-box { 
                margin-top: 10px; padding: 8px; background: #f8fafc; 
                border-left: 3px solid #1c2e4a; font-size: 0.85rem; 
                font-weight: 600; color: #1e293b; 
            }               
        </style>
    """, unsafe_allow_html=True)

def render_exec_card(data: Dict[str, object]):
    """
    Renders an Underwriting Finding card based on Materiality.
    """
    materiality = str(data.get("materiality", "Low")).upper()
    category = str(data.get("category", "General")).upper()
    
    # CSS class mapping - FIXED to match status classes in CSS block
    m_class = f"status-{materiality.lower()}"

    st.markdown(f"""
        <div class="exec-card">
            <div class="card-header">
                <span class="card-label">{category}</span>
                <span class="card-status {m_class}">{materiality} MATERIALITY</span>
            </div>
            <div class="card-body">
                {data.get('evidence', 'No specific snippet found.')}
            </div>
            <div class="verdict-box">
                {data.get('verdict', 'Monitor for further developments.')}
            </div>
            <div class="card-evidence">Source: {data.get('source', 'Internal Document')}</div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def initialize_pipeline() -> Dict[str, object]:
    """Loads the hybrid retriever using the new incremental payload structure."""
    CHUNKS_PATH, EMBEDDINGS_PATH = "data/chunks.pkl", "data/embeddings.pkl"
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
        with open(CHUNKS_PATH, "rb") as f: 
            chunks = pickle.load(f)
        with open(EMBEDDINGS_PATH, "rb") as f: 
            payload = pickle.load(f)
        return {
            "retriever": HybridRetriever(
                vectors=np.array(payload['vectors']), 
                metadata=payload['metadata']
            )
        }
    return {"retriever": None}

# --- APPLICATION INTERFACE ---
inject_executive_css()

st.markdown("<h1 style='font-size: 1.8rem; border-bottom: 2px solid #1c2e4a;'>Credit Intelligence Terminal</h1>", unsafe_allow_html=True)
st.caption("Strategic Underwriting Review | Proprietary & Confidential")

with st.sidebar:
    st.markdown("### Analysis Configuration")
    ticker = st.selectbox("Ticker Symbol", ["TSLA", "AAPL", "AMZN", "NVDA", "CCL"], index=4)
    
    section_label = st.selectbox("Section", [
        "Business (Item 1)",
        "Risk Factors (Item 1A)", 
        "MD&A (Item 7)", 
        "Legal Proceedings (Item 3)",
        "Financial Statements (Item 8)"
    ], index=0)
    
    col1, col2 = st.columns(2)
    y_a = col1.selectbox("Base Year", range(2018, 2027), index=2)
    y_b = col2.selectbox("Target Year", range(2018, 2027), index=7)
    
    query = st.text_area("Underwriting Focus", value="major risk, liquidity, and covenant changes", height=80)
    run = st.button("Generate Report", type="primary", use_container_width=True)

section_map = {
    "Business (Item 1)": "section_1",
    "Risk Factors (Item 1A)": "section_1a",
    "MD&A (Item 7)": "section_7",
    "Legal Proceedings (Item 3)": "section_3",
    "Financial Statements (Item 8)": "section_8",
}

if run:
    with st.spinner("Executing multi-period delta analysis..."):
        try:
            state = initialize_pipeline()
            if not state["retriever"]:
                st.error("Vector Store not found. Please run the ingestion pipeline first.")
                st.stop()

            credit_logic_query = f"""
            {query}
            
            STRICT CREDIT ANALYSIS RULES:
            If you detect a decrease in the 'Fair Value' of debt in Item 8:
            1. CROSS-CHECK Item 7 (MD&A) and Item 1A (Risk) for terms like 'liquidity,' 
               'refinancing,' 'covenant,' or 'credit facility access.'
            2. IF AND ONLY IF those terms indicate actual stress, flag it as a 'Credit Stress Signal.'
            3. OTHERWISE, categorize it as 'Market Risk (Interest Rate-driven)' and 
               explicitly state that no firm-specific liquidity stress was found.
            """

            diff = compare(
                retriever=state["retriever"],
                ticker=ticker,
                section_type=section_map[section_label],
                year_a=int(y_a),
                year_b=int(y_b),
                query=credit_logic_query,
                model="gpt-4o-mini"
            )
            report = generate_structured_output(diff, model="gpt-4o-mini")
            
            meta = state["retriever"].metadata
            len_a = next((m['meta_section_len'] for m in meta if m['ticker']==ticker and m['year']==y_a and m['section_type']==section_map[section_label]), 0)
            len_b = next((m['meta_section_len'] for m in meta if m['ticker']==ticker and m['year']==y_b and m['section_type']==section_map[section_label]), 0)
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            st.stop()

    # --- EXECUTIVE REPORT VIEW ---
    st.markdown(f"Comparative Review: {ticker} (FY{y_a} vs FY{y_b})")
    
    if len_a > 0 and len_b > 0:
        growth = ((len_b - len_a) / len_a) * 100
        color = "#b91c1c" if growth > 15 else "#047857" if growth < -5 else "#4b5563"
        st.markdown(f"""
            <div style="background: white; padding: 10px 15px; border-left: 5px solid {color}; margin-bottom: 20px; font-size: 0.9rem;">
                <strong>Meta-Signal #18:</strong> Disclosure size changed by <span style="color:{color}; font-weight:bold;">{growth:+.1f}%</span> 
                ({len_a:,} → {len_b:,} chars). Large increases typically correlate with increased risk complexity.
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="bluf-card">
            <div class="bluf-title">Executive Summary</div>
            <div class="bluf-content">{report.get('executive_summary', 'Synthesis complete.')}</div>
        </div>
    """, unsafe_allow_html=True)

    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown("Operational & Risk Deltas")
        # UPDATED: Use 'findings' key from new schema
        for finding in report.get("findings", []):
            render_exec_card(finding)

    with c_right:
        st.markdown("Financial & Structural Outlook")
        
        # --- FIXED OUTLOOK BOX: Materiality Counting Logic ---
        findings = report.get("findings", [])
        m_levels = [f.get("materiality", "").upper() for f in findings]
        high_count = m_levels.count("HIGH")
        med_count = m_levels.count("MEDIUM")
        
        outlook_data = report.get("strategic_outlook", {})
        net_posture = str(outlook_data.get("net_posture", "STABLE")).upper()
        
        status_color = "#b91c1c" if high_count > 0 or net_posture == "CAUTIONARY" else "#d97706" if med_count > 0 else "#047857"
            
        st.markdown(f"""
            <div style="background-color: #1c2e4a; color: white; padding: 25px; border-radius: 4px;">
                <div style="font-size: 0.8rem; text-transform: uppercase; opacity: 0.8;">Materiality Profile</div>
                <div style="font-size: 2.2rem; font-weight: 700; margin-top: 5px;">
                    {high_count} High / {med_count} Med
                </div>
                <hr style="opacity: 0.2; margin: 15px 0;">
                <div style="font-size: 1.4rem; font-weight: 700; color: {status_color};">{net_posture}</div>
                <div style="font-size: 0.85rem; margin-top: 10px; opacity: 0.9;">
                    Primary Driver: {outlook_data.get('primary_driver', 'N/A')}
                </div>
                <div style="font-size: 0.75rem; margin-top: 5px; font-style: italic; opacity: 0.7;">
                    {outlook_data.get('liquidity_buffer', '')}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with st.expander("Record of Evidence (Raw RAG Context)"):
        st.text(diff.get("raw_diff", "No raw context retrieved."))

else:
    st.info("Select a ticker and target years in the sidebar to initiate the multi-period underwriting review.")