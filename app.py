from __future__ import annotations

import os
import json
import pickle
import pandas as pd
from typing import Dict, List, Any

import numpy as np
import streamlit as st
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
st.set_page_config(page_title="SEC Credit Intelligence", layout="wide")

def inject_executive_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville&family=Inter:wght@400;600;700&display=swap');
            .main { background-color: #f4f5f7; }
            html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
            .bluf-card { background-color: #ffffff; border-top: 4px solid #1c2e4a; padding: 25px; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .bluf-title { font-family: 'Libre Baskerville', serif; font-size: 1.4rem; color: #1c2e4a; margin-bottom: 10px; }
            .bluf-content { font-family: 'Libre Baskerville', serif; font-size: 1.1rem; line-height: 1.6; color: #333; }
            .exec-card { background-color: #ffffff; border: 1px solid #e1e4e8; padding: 18px; margin-bottom: 12px; border-radius: 2px; }
            .card-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #f0f0f0; padding-bottom: 8px; margin-bottom: 12px; }
            .card-label { font-size: 0.85rem; font-weight: 700; color: #1c2e4a; text-transform: uppercase; letter-spacing: 0.02em; }
            .card-status { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; padding: 2px 8px; border: 1px solid #1c2e4a; border-radius: 3px; }
            .card-body { font-size: 0.9rem; line-height: 1.5; color: #2d3748; }
            .status-high { color: #b91c1c; border-color: #b91c1c; background-color: #fef2f2; }
            .status-medium { color: #d97706; border-color: #d97706; background-color: #fffbeb; }
            .status-low { color: #047857; border-color: #047857; background-color: #f0fdf4; }
            [data-testid="stSidebar"] { background-color: #1c2e4a !important; }
            .stSelectbox label, .stTextArea label, .stTextInput label { color: white !important; }
        </style>
    """, unsafe_allow_html=True)

def render_exec_card(data: Dict[str, Any]):
    materiality = str(data.get("materiality", "Low")).upper()
    m_class = f"status-{materiality.lower()}"
    st.markdown(f"""
        <div class="exec-card">
            <div class="card-header">
                <span class="card-label">{str(data.get('category', 'General')).upper()}</span>
                <span class="card-status {m_class}">{materiality} MATERIALITY</span>
            </div>
            <div class="card-body">{data.get('evidence', 'No specific snippet found.')}</div>
            <div style="margin-top: 10px; padding: 8px; background: #f8fafc; border-left: 3px solid #1c2e4a; font-size: 0.85rem; font-weight: 600;">
                {data.get('verdict', 'Monitor for developments.')}
            </div>
        </div>
    """, unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def get_filing_data(ticker: str, year: int, section_id: str):
    file_id = f"{ticker}_{year}_{section_id}"
    chunk_path = f"{CACHE_DIR}/{file_id}_chunks.pkl"
    vec_path = f"{CACHE_DIR}/{file_id}_vecs.pkl"

    # 1. Instant Cache Return
    if os.path.exists(chunk_path) and os.path.exists(vec_path):
        with open(chunk_path, "rb") as f: chunks = pickle.load(f)
        with open(vec_path, "rb") as f: 
            p = pickle.load(f)
            return (chunks, p["vectors"]) if isinstance(p, dict) else (chunks, p[0])

    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K").filter(date=f"{year}-01-01:{year}-12-31")
        if not filings: return None, None
        
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
            "section_8": "financial_statements"
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
        df_temp = pd.DataFrame([{
            "ticker": ticker, "year": year, "section_type": section_id,
            "section_text": raw_text, "section_char_len": len(raw_text)
        }])
        
        chunks = build_chunks(df_temp, cache_path=chunk_path)
        vecs, metadata = embed_chunks(chunks, cache_path=vec_path)
        return metadata, vecs

    except Exception as e:
        print(f"Error fetching {ticker} {section_id}: {e}")
        return None, None

# --- APP INTERFACE ---
inject_executive_css()

st.markdown("<h1 style='font-size: 1.8rem; border-bottom: 2px solid #1c2e4a;'>Credit Intelligence Terminal</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Analysis Configuration")
    ticker = st.text_input("Ticker Symbol", value="NKE").upper()
    section_label = st.selectbox("Section", [
        "Business (Item 1)", "Risk Factors (Item 1A)", "MD&A (Item 7)", 
        "Legal Proceedings (Item 3)", "Financial Statements (Item 8)"
    ], index=1)
    
    col1, col2 = st.columns(2)
    y_a = col1.selectbox("Base Year", range(2018, 2027), index=2)
    y_b = col2.selectbox("Target Year", range(2018, 2027), index=7)
    
    query = st.text_area("Underwriting Focus", value="major risk, liquidity, and covenant changes", height=80)
    run = st.button("Generate Report", type="primary", use_container_width=True)

section_map = {
    "Business (Item 1)": "section_1", "Risk Factors (Item 1A)": "section_1a",
    "MD&A (Item 7)": "section_7", "Legal Proceedings (Item 3)": "section_3",
    "Financial Statements (Item 8)": "section_8",
}

if run:
    with st.spinner(f"Analyzing {ticker} FY{y_a} vs FY{y_b}..."):
        chunks_a, vecs_a = get_filing_data(ticker, y_a, section_map[section_label])
        chunks_b, vecs_b = get_filing_data(ticker, y_b, section_map[section_label])

        if chunks_a is None or chunks_b is None:
            st.error(f"Filing data unavailable for {ticker}. SEC may not have indexed this section yet.")
            st.stop()

        # Build Hybrid Retriever (Fast because model is already in RAM)
        retriever = HybridRetriever(vectors=np.vstack([vecs_a, vecs_b]), metadata=chunks_a + chunks_b)

        diff = compare(retriever=retriever, ticker=ticker, section_type=section_map[section_label], 
                       year_a=int(y_a), year_b=int(y_b), query=query)
        report = generate_structured_output(diff)
        
        # Meta-Signal Calculation
        len_a = next((m['meta_section_len'] for m in chunks_a), 0)
        len_b = next((m['meta_section_len'] for m in chunks_b), 0)
        
        # --- RENDER REPORT ---
        st.markdown(f"### Comparative Review: {ticker}")
        
        if len_a > 0:
            growth = ((len_b - len_a) / len_a) * 100
            color = "#b91c1c" if growth > 15 else "#047857"
            st.markdown(f'<div style="border-left: 5px solid {color}; padding-left: 10px; font-size: 0.9rem; margin-bottom: 20px;">'
                        f'<strong>Meta-Signal #18:</strong> Disclosure size changed by <strong>{growth:+.1f}%</strong>.</div>', unsafe_allow_html=True)

        st.markdown(f'<div class="bluf-card"><div class="bluf-title">Executive Summary</div>'
                    f'<div class="bluf-content">{report.get("executive_summary", "N/A")}</div></div>', unsafe_allow_html=True)

        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown("#### Operational & Risk Deltas")
            for finding in report.get("findings", []): render_exec_card(finding)

        with c_right:
            st.markdown("#### Strategic Outlook")
            outlook = report.get("strategic_outlook", {})
            st.markdown(f'<div style="background-color: #1c2e4a; color: white; padding: 25px; border-radius: 4px;">'
                        f'<div style="font-size: 1.4rem; font-weight: 700;">{str(outlook.get("net_posture", "STABLE")).upper()}</div>'
                        f'<div style="font-size: 0.85rem; margin-top: 10px; opacity: 0.9;">Primary Driver: {outlook.get("primary_driver", "N/A")}</div></div>', unsafe_allow_html=True)
else:
    st.info("Input a Ticker to begin analysis.")