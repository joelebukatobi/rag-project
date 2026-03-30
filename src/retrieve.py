from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

def safe_normalize(vecs: np.ndarray) -> np.ndarray:
    """Helper to replace faiss.normalize_L2 using NumPy for consistency."""
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
        self.embed_model = SentenceTransformer(self.embed_model_name)
        # Cache for the filtered subset to optimize cross-year comparisons
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
        """
        Filters the global vector store down to specific 10-K sections.
        Ensures Credit Taxonomy metadata (meta_section_len) is preserved.
        """
        indices = []
        for i, item in enumerate(self.metadata):
            if ticker and str(item.get("ticker", "")).upper() != ticker.upper():
                continue
            if year and int(item.get("year", 0)) != int(year):
                continue
            if section_type and str(item.get("section_type", "")).lower() != section_type.lower():
                continue
            indices.append(i)

        if not indices:
            return np.array([]), []
            
        return self.vectors[indices].astype("float32"), [self.metadata[i] for i in indices]

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        section_type: Optional[str] = None,
        top_k: int = 8,  # Increased default for better signal coverage
        use_bm25: bool = True,
        use_metadata_filter: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieval combining Semantic (FAISS) and Lexical (BM25) search.
        Designed to catch keyword-heavy credit signals (e.g., 'liquidity', 'defaults').
        """
        
        # 1. Sub-Index Management (Filtering)
        filter_key = (ticker, year, section_type) if use_metadata_filter else ("ALL", None, None)
        
        if filter_key != self._last_filter_key:
            sub_vecs, sub_meta = self._get_filtered_subset(
                ticker if use_metadata_filter else None,
                year if use_metadata_filter else None,
                section_type if use_metadata_filter else None
            )
            
            if sub_vecs.size > 0:
                # Normalize and build temporary FAISS index for the specific filing
                norm_vecs = safe_normalize(sub_vecs)
                index = faiss.IndexFlatIP(norm_vecs.shape[1])
                index.add(norm_vecs)
                self._cached_sub_index = index
                
                # Build BM25 for precise keyword matching (Credit Signals #9, #10)
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

        # 2. Semantic Search (Dense)
        q_vec = self.embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        # Search breadly to find nuance, then re-rank via hybrid scores
        search_breadth = min(top_k * 5, len(self._cached_meta))
        sem_scores, sem_ids = self._cached_sub_index.search(q_vec, search_breadth)
        semantic_map = {int(i): float(s) for i, s in zip(sem_ids[0], sem_scores[0])}

        # 3. Lexical Search (Sparse - BM25)
        bm25_scores = []
        if use_bm25 and self._cached_bm25:
            bm25_scores = self._cached_bm25.get_scores(query.lower().split())
            b_min, b_max = np.min(bm25_scores), np.max(bm25_scores)
            denom = (b_max - b_min) if (b_max - b_min) > 1e-9 else 1.0
            
        # 4. Hybrid Reranking
        merged = []
        for idx, meta in enumerate(self._cached_meta):
            # Get Semantic score or default to a baseline
            sem = semantic_map.get(idx, 0.0)
            
            # Get Lexical score (Normalized)
            lex = 0.0
            if use_bm25 and len(bm25_scores) > 0:
                lex = (bm25_scores[idx] - b_min) / denom

            # CreditDelta weighting: Semantic 60% / Lexical 40%
            s_w = self.semantic_weight if use_bm25 else 1.0
            l_w = self.lexical_weight if use_bm25 else 0.0
            final_score = (s_w * sem) + (l_w * lex)

            # Preserve all meta-signals (meta_section_len) for the LLM
            merged.append({
                **meta,
                "semantic_score": sem,
                "bm25_score": lex,
                "final_score": final_score,
            })

        # Return top results sorted by the hybrid credit-weighting
        merged.sort(key=lambda x: x["final_score"], reverse=True)
        return merged[:top_k]