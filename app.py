from __future__ import annotations

import json
from typing import Dict

import streamlit as st

from src.chunk import build_chunks, parse_sections
from src.embed import embed_chunks
from src.evaluate import build_test_set, run_retrieval_eval
from src.generate import compare, generate_structured_output
from src.ingest import load_raw_filings
from src.retrieve import HybridRetriever

st.set_page_config(page_title="SEC Filings RAG", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def initialize_pipeline() -> Dict[str, object]:
    tickers = ["TSLA", "AAPL", "MSFT", "AMZN", "NVDA"]
    year_range = (2018, 2023)

    df = load_raw_filings(tickers=tickers, year_range=year_range, filing_type="10-K")
    df_sections = parse_sections(df)
    all_chunks = build_chunks(df_sections, chunk_size=800, chunk_overlap=200, cache_path="data/chunks.pkl")
    vectors, metadata = embed_chunks(all_chunks, cache_path="data/embeddings.pkl")
    retriever = HybridRetriever(vectors=vectors, metadata=metadata, semantic_weight=0.6, lexical_weight=0.4)

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
    ticker = st.selectbox("Company ticker", ["TSLA", "AAPL", "MSFT", "AMZN", "NVDA"], index=0)
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
