from __future__ import annotations

import os
import json
from typing import Dict

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

st.set_page_config(page_title="SEC Filings RAG", page_icon="📄", layout="wide")

@st.cache_resource(show_spinner=False)
def initialize_pipeline() -> Dict[str, object]:
    CHUNKS_PATH = "data/chunks.pkl"
    EMBEDDINGS_PATH = "data/embeddings.pkl"
    os.makedirs("data", exist_ok=True)

    # Initialize variables to None to prevent NameErrors later
    all_chunks, vectors, metadata = None, None, None
    df, df_sections = None, None
    loaded_from_cache = False

    # 1. ATTEMPT CACHE LOAD
    if os.path.exists(CHUNKS_PATH) and os.path.exists(EMBEDDINGS_PATH):
            try:
                with open(CHUNKS_PATH, "rb") as f:
                    all_chunks = pickle.load(f)
                with open(EMBEDDINGS_PATH, "rb") as f:
                    raw_data = pickle.load(f)
                
                # --- Dictionary-Safe Unpacking ---
                if isinstance(raw_data, dict):
                    vectors = raw_data.get("vectors")
                    metadata = raw_data.get("metadata")
                else:
                    # Fallback if it was saved as a tuple (vectors, metadata)
                    vectors, metadata = raw_data
                
                # CRITICAL: Convert the actual data to numpy, not the string keys!
                vectors = np.array(vectors)
                
                loaded_from_cache = True
                st.sidebar.success(f"Loaded data from local cache.")
            except Exception as e:
                st.sidebar.warning(f" Cache corrupted or incompatible: {e}")
                loaded_from_cache = False

    # 2. FALLBACK TO INGESTION (If cache failed or doesn't exist)
    if not loaded_from_cache:
        st.info("🔄 Cache missing or invalid. Starting fresh SEC ingestion...")
        
        tickers = ["TSLA", "AAPL", "CCL", "AMZN", "NVDA"]
        year_range = (2018, 2023)
        
        # Run the full pipeline
        df = load_live_filings(tickers=tickers, year_range=year_range)
        df_sections = parse_sections(df)
        all_chunks = build_chunks(df_sections, cache_path=CHUNKS_PATH)
        vectors, metadata = embed_chunks(all_chunks, cache_path=EMBEDDINGS_PATH)
        
        vectors = np.array(vectors)

    # 3. BUILD RETRIEVER
    retriever = HybridRetriever(
        vectors=vectors, 
        metadata=metadata, 
        semantic_weight=0.6, 
        lexical_weight=0.4
    )

    return {
        "df": df,
        "df_sections": df_sections,
        "all_chunks": all_chunks,
        "vectors": vectors,
        "metadata": metadata,
        "retriever": retriever,
    }

st.title("Analyst-Ready SEC Filings RAG System")
st.caption("Structured comparative intelligence across 10-K filing years")

with st.sidebar:
    st.header("Controls")
    ticker = st.selectbox("Company ticker", ["TSLA", "AAPL", "AMZN", "NVDA", "CCL"], index=0)
    section_label = st.selectbox("Section", ["Risk Factors", "MD&A", "Legal Proceedings"], index=0)
    year_a = st.selectbox("Year A", [2018, 2019, 2020, 2021, 2022, 2023], index=2)
    year_b = st.selectbox("Year B", [2018, 2019, 2020, 2021, 2022, 2023], index=5)
    query = st.text_input("Focus query", value="major risk and revenue shifts")
    run = st.button("Run comparative analysis", type="primary")

section_map = {
    "Risk Factors": "section_1a",
    "MD&A": "section_7",
    "Legal Proceedings": "section_3",
}

if run:
    if year_a == year_b:
        st.error("Please select two different years.")
        st.stop()

    with st.spinner("Loading filings, retrieving evidence, and generating structured output..."):
        try:
            state = initialize_pipeline()
            retriever = state["retriever"]

            diff = compare(
                retriever=retriever,
                ticker=ticker,
                section_type=section_map[section_label],
                year_a=int(year_a),
                year_b=int(year_b),
                query=query,
                top_k=5,
                model="gpt-4o",
            )
            structured = generate_structured_output(diff, model="gpt-4o")
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)
            st.stop()

    if "error" in structured:
        st.warning("Structured generation returned an error. Displaying diagnostics.")
        st.json(structured)
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Risk Deltas")
            st.dataframe(structured.get("risk_changes", []), use_container_width=True)

            st.subheader("Revenue Drivers")
            st.dataframe(structured.get("revenue_drivers", []), use_container_width=True)

        with c2:
            st.subheader("Litigation Changes")
            st.dataframe(structured.get("litigation_changes", []), use_container_width=True)

            st.subheader("Tone Shift")
            st.json(structured.get("tone_shift", {}))

        st.subheader("Citations")
        st.write(structured.get("citations", []))

        with st.expander("Raw JSON output"):
            st.code(json.dumps(structured, indent=2), language="json")

    with st.expander("Raw comparative narrative"):
        st.write(diff.get("raw_diff", ""))

with st.expander("Quick retrieval health check"):
    if st.button("Run tiny evaluation sample"):
        state = initialize_pipeline()
        retriever = state["retriever"]
        test_set = build_test_set()[:5]
        eval_df = run_retrieval_eval(retriever, test_set, top_k=5)
        st.dataframe(eval_df, use_container_width=True)
        st.metric("Mean Recall@5", round(float(eval_df["Recall@5"].mean()), 3))
        st.metric("Mean MRR", round(float(eval_df["MRR"].mean()), 3))
