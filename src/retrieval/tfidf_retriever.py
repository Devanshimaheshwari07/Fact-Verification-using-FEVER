"""TF-IDF and Cosine Similarity retrieval model."""
import logging
from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.indexing.inverted_index import InvertedIndex
from src.indexing.build_index import IndexManager
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class TfidfRetriever(BaseRetriever):
    """Retrieves and ranks evidence sentences using Vector Space Model (TF-IDF + Cosine Similarity)."""

    def __init__(
        self,
        index_manager: IndexManager,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        self.index_manager = index_manager
        self.preprocessor = preprocessor or TextPreprocessor()

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Calculates cosine similarity between query and all indexed documents."""
        if not query or not query.strip():
            return []

        if self.index_manager.tfidf_vectorizer is None or self.index_manager.tfidf_matrix is None:
            logger.warning("TF-IDF vectorizer or matrix not loaded.")
            return []

        # Vectorize claim query
        try:
            query_vec = self.index_manager.tfidf_vectorizer.transform([query])
        except Exception as e:
            logger.error(f"Error vectorizing query '{query}': {e}")
            return []

        if query_vec.nnz == 0:
            # Query contains no known vocabulary terms
            return []

        # Cosine similarity is the dot product of L2-normalized vectors
        # Shape: (num_docs, 1) -> flatten to (num_docs,)
        similarity_scores = (self.index_manager.tfidf_matrix @ query_vec.T).toarray().ravel()

        # Get top-k indices
        if len(similarity_scores) == 0:
            return []

        # Find documents with score > 0
        non_zero_indices = np.where(similarity_scores > 1e-6)[0]
        if len(non_zero_indices) == 0:
            return []

        # Sort non-zero scores descending
        top_indices = non_zero_indices[np.argsort(-similarity_scores[non_zero_indices])[:top_k]]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            record_id = self.index_manager.record_ids[idx]
            record = self.index_manager.inverted_index.get_record(record_id)
            if not record:
                continue

            results.append(SearchResult(
                record_id=record.record_id,
                doc_id=record.doc_id,
                sentence_id=record.sentence_id,
                text=record.text,
                page_title=record.page_title,
                score=float(similarity_scores[idx]),
                rank=rank,
                retrieval_method="TF-IDF"
            ))

        return results
