from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import faiss
import numpy as np
import streamlit as st # Added for caching
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Re-use the cached model loader to save 15 seconds per run
@st.cache_resource
def _get_embedding_model(model_name: str):
    return SentenceTransformer(model_name)

def safe_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / (norms + 1e-10)

@dataclass
class HybridRetriever:
    vectors: np.ndarray
    metadata: List[Dict[str, Any]]
    embed_model_name: str = "all-MiniLM-L6-v2"
    semantic_weight: float = 0.6
    lexical_weight: float = 0.4

    def __post_init__(self) -> None:
        # FIX: Use cached model instead of re-loading every time
        self.embed_model = _get_embedding_model(self.embed_model_name)
        
        self._last_filter_key = None
        self._cached_sub_index = None
        self._cached_bm25 = None
        self._cached_meta = []

    def _get_filtered_subset(
        self, 
        ticker: Optional[str], 
        year: Optional[int], 
        section_type: Optional[str]
    ) -> tuple[np.ndarray, List[Dict[str, Any]]]:
        indices = []
        for i, item in enumerate(self.metadata):
            # String matching for robustness
            t_match = not ticker or str(item.get("ticker", "")).upper() == ticker.upper()
            y_match = not year or int(item.get("year", 0)) == int(year)
            s_match = not section_type or str(item.get("section_type", "")).lower() == section_type.lower()
            
            if t_match and y_match and s_match:
                indices.append(i)

        if not indices:
            return np.array([]).reshape(0, self.vectors.shape[1] if self.vectors.size > 0 else 384), []
            
        return self.vectors[indices].astype("float32"), [self.metadata[i] for i in indices]

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        section_type: Optional[str] = None,
        top_k: int = 8,
        use_bm25: bool = True,
        use_metadata_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        
        filter_key = (ticker, year, section_type) if use_metadata_filter else ("ALL", None, None)
        
        if filter_key != self._last_filter_key:
            sub_vecs, sub_meta = self._get_filtered_subset(
                ticker if use_metadata_filter else None,
                year if use_metadata_filter else None,
                section_type if use_metadata_filter else None
            )
            
            if sub_vecs.size > 0:
                norm_vecs = safe_normalize(sub_vecs)
                index = faiss.IndexFlatIP(norm_vecs.shape[1])
                index.add(norm_vecs)
                self._cached_sub_index = index
                
                if use_bm25:
                    tokenized = [str(m.get("text", "")).lower().split() for m in sub_meta]
                    self._cached_bm25 = BM25Okapi(tokenized)
            else:
                self._cached_sub_index = None
                self._cached_bm25 = None
            
            self._cached_meta = sub_meta
            self._last_filter_key = filter_key

        if not self._cached_meta or self._cached_sub_index is None:
            return []

        # 2. Semantic Search
        # Explicitly use the cached model's encode
        q_vec = self.embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        
        search_breadth = min(top_k * 5, len(self._cached_meta))
        sem_scores, sem_ids = self._cached_sub_index.search(q_vec, search_breadth)
        semantic_map = {int(i): float(s) for i, s in zip(sem_ids[0], sem_scores[0])}

        # 3. Lexical Search
        bm25_scores = []
        b_min, b_max = 0.0, 1.0
        if use_bm25 and self._cached_bm25:
            bm25_scores = self._cached_bm25.get_scores(query.lower().split())
            if len(bm25_scores) > 0:
                b_min, b_max = np.min(bm25_scores), np.max(bm25_scores)
        
        denom = (b_max - b_min) if (b_max - b_min) > 1e-9 else 1.0
            
        # 4. Hybrid Reranking
        merged = []
        for idx, meta in enumerate(self._cached_meta):
            sem = semantic_map.get(idx, 0.0)
            lex = (bm25_scores[idx] - b_min) / denom if (use_bm25 and len(bm25_scores) > 0) else 0.0

            final_score = (self.semantic_weight * sem) + (self.lexical_weight * lex)

            merged.append({
                **meta,
                "semantic_score": sem,
                "bm25_score": lex,
                "final_score": final_score,
            })

        # Deterministic ordering: break ties by chunk_id for stable caching/output.
        merged.sort(
            key=lambda x: (
                -float(x.get("final_score", 0.0)),
                str(x.get("chunk_id", "")),
            )
        )
        return merged[:top_k]