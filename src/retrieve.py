from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class HybridRetriever:
    vectors: np.ndarray
    metadata: List[Dict[str, object]]
    embed_model_name: str = "all-MiniLM-L6-v2"
    semantic_weight: float = 0.6
    lexical_weight: float = 0.4

    def __post_init__(self) -> None:
        self.embed_model = SentenceTransformer(self.embed_model_name)
        if self.vectors.size > 0:
            vecs = self.vectors.astype("float32")
            faiss.normalize_L2(vecs)
            self.faiss_index = faiss.IndexFlatIP(vecs.shape[1])
            self.faiss_index.add(vecs)
        else:
            self.faiss_index = None

    def _candidate_indices(
        self,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        section_type: Optional[str] = None,
    ) -> List[int]:
        indices = []
        for i, item in enumerate(self.metadata):
            if ticker and str(item["ticker"]).upper() != ticker.upper():
                continue
            if year and int(item["year"]) != int(year):
                continue
            if section_type and str(item["section_type"]).lower() != section_type.lower():
                continue
            indices.append(i)
        return indices

    def retrieve(
        self,
        query: str,
        ticker: Optional[str] = None,
        year: Optional[int] = None,
        section_type: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, object]]:
        """
        Hybrid retrieval with metadata filtering first, then FAISS+BM25 reranking.
        """
        candidates = self._candidate_indices(ticker=ticker, year=year, section_type=section_type)
        if not candidates:
            return []

        candidate_meta = [self.metadata[i] for i in candidates]
        candidate_vecs = self.vectors[candidates].astype("float32")

        # Semantic scoring on filtered candidates.
        q_vec = self.embed_model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(candidate_vecs)
        sub_index = faiss.IndexFlatIP(candidate_vecs.shape[1])
        sub_index.add(candidate_vecs)

        sem_scores, sem_ids = sub_index.search(q_vec, min(top_k * 4, len(candidate_meta)))
        semantic_map = {int(i): float(s) for i, s in zip(sem_ids[0], sem_scores[0])}

        # Lexical BM25 scoring on filtered candidates.
        tokenized = [str(m["text"]).lower().split() for m in candidate_meta]
        bm25 = BM25Okapi(tokenized)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Min-max normalization keeps semantic and lexical scores comparable.
        bm25_min = float(np.min(bm25_scores))
        bm25_max = float(np.max(bm25_scores))

        def norm_bm25(v: float) -> float:
            if bm25_max - bm25_min < 1e-9:
                return 0.0
            return (float(v) - bm25_min) / (bm25_max - bm25_min)

        merged = []
        for local_idx, meta in enumerate(candidate_meta):
            sem = semantic_map.get(local_idx, 0.0)
            lex = norm_bm25(float(bm25_scores[local_idx]))
            final = self.semantic_weight * sem + self.lexical_weight * lex
            merged.append(
                {
                    **meta,
                    "semantic_score": sem,
                    "bm25_score": lex,
                    "final_score": final,
                }
            )

        merged.sort(key=lambda x: x["final_score"], reverse=True)
        return merged[:top_k]
