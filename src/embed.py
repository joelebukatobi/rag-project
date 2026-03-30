from __future__ import annotations

import os
import pickle
from multiprocessing import cpu_count
from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


def _load_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_chunks(
    all_chunks: List[Dict[str, object]],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 128,
    cache_path: str = "data/embeddings.pkl",
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """
    Embed chunk text and cache vectors. 
    Now supports incremental updates (only embeds what is missing).
    """
    existing_vectors = np.array([]).reshape(0, 0)
    existing_metadata = []

    # 1. Load existing cache if it exists
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
            existing_vectors = data["vectors"]
            existing_metadata = data["metadata"]
        
        # Create a set of unique IDs already embedded to avoid duplicates
        # Using chunk_id as the unique key
        seen_ids = {m.get("chunk_id") for m in existing_metadata if m.get("chunk_id")}
        
        # Filter all_chunks to only those NOT in seen_ids
        chunks_to_embed = [c for c in all_chunks if c.get("chunk_id") not in seen_ids]
        
        if not chunks_to_embed:
            print("✅ All tickers/chunks already present in cache. Skipping embedding.")
            return existing_vectors, existing_metadata
        
        print(f"🔄 Found {len(chunks_to_embed)} new chunks to add to cache.")
    else:
        chunks_to_embed = all_chunks

    if not chunks_to_embed:
        return existing_vectors, existing_metadata

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    model = _load_model(model_name)

    # 2. Prepare Metadata for NEW chunks
    new_texts = [str(c["text"]) for c in chunks_to_embed]
    new_metadata = []
    for c in chunks_to_embed:
        new_metadata.append({
            "ticker": c["ticker"],
            "year": int(c["year"]),
            "section_type": c["section_type"],
            "chunk_id": c.get("chunk_id", ""),
            "source": c.get("source", f"{c['ticker']}_{c['year']}_10k"),
            "text": c["text"],
            "meta_section_len": c.get("meta_section_len", 0), 
            "label": c.get("label", "General")
        })

    # 3. Generate NEW Vectors
    new_vectors = model.encode(
        new_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True, 
    ).astype("float32")

    # 4. Merge and Save
    if existing_vectors.size > 0:
        final_vectors = np.vstack([existing_vectors, new_vectors])
        final_metadata = existing_metadata + new_metadata
    else:
        final_vectors = new_vectors
        final_metadata = new_metadata

    payload = {"vectors": final_vectors, "metadata": final_metadata}
    
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)

    return payload["vectors"], payload["metadata"]

def embedding_info(vectors: np.ndarray) -> Dict[str, int]:
    """Diagnostic info for the CreditDelta vector store."""
    if vectors.size == 0:
        return {"count": 0, "dimension": 0, "workers": cpu_count()}

    return {
        "count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "workers": cpu_count(),
    }