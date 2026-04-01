import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from edgar import Company, set_identity
from src.chunk import build_chunks
from src.embed import embed_chunks

set_identity("cg77@fordham.edu") 

CACHE_DIR = "data/filings"
os.makedirs(CACHE_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def get_filing_data(ticker: str, year: int, section_id: str):
    """
    High-Performance Logic: RAM Cache -> Disk Cache -> SEC Fetch.
    Under 6 seconds for cached data; optimized fetch for new data.
    """
    file_id = f"{ticker}_{year}_{section_id}"
    chunk_path = f"{CACHE_DIR}/{file_id}_chunks.pkl"
    vec_path = f"{CACHE_DIR}/{file_id}_vecs.pkl"

    # 1. DISK HIT: Check if we already processed this ticker/year/section
    if os.path.exists(chunk_path) and os.path.exists(vec_path):
        try:
            with open(chunk_path, "rb") as f: chunks = pickle.load(f)
            with open(vec_path, "rb") as f: 
                data = pickle.load(f)
                # Handle both Tuple or Dict return formats from your embed.py
                vecs = data["vectors"] if isinstance(data, dict) else data[0]
            return chunks, vecs
        except Exception as e:
            print(f"Cache read error for {file_id}: {e}")

    # 2. SEC FETCH: Only happens if not on disk
    try:
        company = Company(ticker)
        # Search for 10-K filings in that specific year
        filings = company.get_filings(form="10-K").filter(date=f"{year}-01-01:{year}-12-31")
        if not filings:
            return None, None
        
        tenk = filings[0].obj()
        
        # Mapping your IDs to edgartools logic
        attr_map = {
            "section_1": "business",
            "section_1a": "risk_factors",
            "section_3": "legal_proceedings",
            "section_7": "management_discussion",
            "section_8": "financial_statements"
        }
        
        attr_name = attr_map.get(section_id)
        # Try primary attribute
        raw_text = getattr(tenk, attr_name, "")
        
        # FALLBACK: If attribute is missing/empty, try manual item lookup
        if not raw_text or len(str(raw_text)) < 200:
            # Map section_1a -> 'Item 1A'
            item_id = section_id.replace("section_", "Item ").upper()
            try:
                raw_text = tenk.section(item_id)
            except:
                raw_text = ""

        if not raw_text or len(str(raw_text)) < 100:
            return None, None

        # 3. PROCESSING BRIDGE
        df_temp = pd.DataFrame([{
            "ticker": ticker,
            "year": year,
            "section_type": section_id,
            "section_text": str(raw_text),
            "section_char_len": len(str(raw_text))
        }])
        
        # Generate chunks using your src logic
        chunks = build_chunks(df_temp, cache_path=chunk_path)
        
        # Generate vectors
        # Note: We pass vec_path to your embed_chunks to ensure it saves to the right file
        vecs, metadata = embed_chunks(chunks, cache_path=vec_path)
        
        return metadata, vecs

    except Exception as e:
        print(f"Critical error processing {ticker} FY{year}: {str(e)}")
        return None, None