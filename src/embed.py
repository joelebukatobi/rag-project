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
    batch_size: int = 256,
    cache_path: str = "data/embeddings.pkl",
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    """Embed chunk text and cache vectors with metadata."""
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data["vectors"], data["metadata"]

    if not all_chunks:
        return np.array([]), []

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    model = _load_model(model_name)

    texts = [str(c["text"]) for c in all_chunks]
    metadata = [
        {
            "ticker": c["ticker"],
            "year": int(c["year"]),
            "section_type": c["section_type"],
            "chunk_id": c["chunk_id"],
            "source": c["source"],
            "text": c["text"],
        }
        for c in all_chunks
    ]

    # SentenceTransformer internally batches efficiently; process pool speeds up tokenization.
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    payload = {"vectors": vectors, "metadata": metadata}
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f)

    return vectors, metadata


def embedding_info(vectors: np.ndarray) -> Dict[str, int]:
    if vectors.size == 0:
        return {"count": 0, "dimension": 0, "workers": cpu_count()}

    return {
        "count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]),
        "workers": cpu_count(),
    }
