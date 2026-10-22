"""Hybrid retrieval model blending Sparse (BM25) and Dense (Sentence Embedding) using Reciprocal Rank Fusion (RRF)."""
import logging
from typing import List, Dict, Optional
from collections import defaultdict

from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Hybrid ranking model blending sparse lexical retrieval and dense vector retrieval using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{m in {sparse, dense}} (weight_m / (rrf_k + rank_m(d)))
    """

    def __init__(
        self,
        sparse_retriever: Optional[BaseRetriever] = None,
        dense_retriever: Optional[BaseRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        tfidf_retriever: Optional[TfidfRetriever] = None,
        rrf_k: int = 60,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        # Support both explicit sparse/dense parameters and legacy bm25/tfidf arguments
        self.sparse = sparse_retriever or bm25_retriever
        self.dense = dense_retriever or tfidf_retriever
        self.rrf_k = rrf_k
        self.preprocessor = preprocessor or TextPreprocessor()

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Performs two-stage retrieval blending sparse and dense models with Reciprocal Rank Fusion."""
        if not query or not query.strip():
            return []

        candidate_pool_k = max(top_k * 4, 25)

        # 1. Sparse retrieval results (BM25)
        sparse_results: List[SearchResult] = []
        if self.sparse is not None:
            try:
                sparse_results = self.sparse.retrieve(query, top_k=candidate_pool_k)
            except Exception as e:
                logger.warning(f"Sparse retriever failed: {e}")

        # 2. Dense semantic vector retrieval results (Sentence Embedding)
        dense_results: List[SearchResult] = []
        if self.dense is not None:
            try:
                dense_results = self.dense.retrieve(query, top_k=candidate_pool_k)
            except Exception as e:
                logger.warning(f"Dense retriever failed: {e}")

        if not sparse_results and not dense_results:
            return []

        # 3. Reciprocal Rank Fusion calculation
        rrf_scores: Dict[int, float] = defaultdict(float)
        record_map: Dict[int, SearchResult] = {}

        # Sparse contribution
        for rank, res in enumerate(sparse_results, 1):
            rrf_scores[res.record_id] += 1.0 / (self.rrf_k + rank)
            record_map[res.record_id] = res

        # Dense contribution
        for rank, res in enumerate(dense_results, 1):
            rrf_scores[res.record_id] += 1.0 / (self.rrf_k + rank)
            if res.record_id not in record_map:
                record_map[res.record_id] = res

        # Optional entity reinforcement
        entity_info = self.preprocessor.extract_entities_and_keywords(query)
        entities_lower = [e.lower() for e in entity_info["entities"]]
        if entities_lower:
            for record_id, res in record_map.items():
                page_lower = res.page_title.lower()
                if any(ent in page_lower or page_lower in ent for ent in entities_lower):
                    rrf_scores[record_id] *= 1.20

        # 4. Sort and build final ranked list
        sorted_candidates = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        fused_results: List[SearchResult] = []
        for rank, (record_id, fused_score) in enumerate(sorted_candidates, 1):
            base_item = record_map[record_id]
            fused_results.append(SearchResult(
                record_id=base_item.record_id,
                doc_id=base_item.doc_id,
                sentence_id=base_item.sentence_id,
                text=base_item.text,
                page_title=base_item.page_title,
                score=float(fused_score),
                rank=rank,
                retrieval_method="Hybrid (RRF: Sparse + Dense)"
            ))

        return fused_results
