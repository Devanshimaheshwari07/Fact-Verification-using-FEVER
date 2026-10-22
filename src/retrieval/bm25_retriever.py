"""Okapi BM25 probabilistic retrieval model."""
import logging
import math
from typing import List, Optional, Dict
import numpy as np

from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 (Best Matching 25) probabilistic ranking model with configurable k1 and b parameters.

    Implements the Okapi BM25 scoring function using the InvertedIndex postings lists.
    """

    def __init__(
        self,
        index: InvertedIndex,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        preprocessor: Optional[TextPreprocessor] = None
    ):
        self.index = index
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.preprocessor = preprocessor or TextPreprocessor()

    def _compute_idf(self, term: str) -> float:
        """Computes Robertson-Spärck Jones Inverse Document Frequency (IDF).

        Formula:
            IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1.0 )
        """
        N = self.index.total_documents
        df = self.index.get_doc_frequency(term)
        if df == 0 or N == 0:
            return 0.0

        # Non-negative Robertson-Spärck Jones IDF variant
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1.0)
        return max(idf, self.epsilon)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Calculates BM25 relevance scores for all candidate documents matching query terms."""
        if not query or not query.strip():
            return []

        query_tokens = self.preprocessor.preprocess(query)
        if not query_tokens:
            return []

        avg_dl = self.index.avg_doc_length
        if avg_dl == 0:
            avg_dl = 1.0

        components = self.preprocessor.extract_claim_components(query)
        attribute_tokens = components.get("attribute_tokens", set())

        doc_scores: Dict[int, float] = {}

        for term in query_tokens:
            postings = self.index.get_postings(term)
            if not postings:
                continue

            idf = self._compute_idf(term)
            # Boost specific claim attributes (e.g., vegan, Fox, Apollo) so sentences matching the key assertion rank higher
            is_attr = term in attribute_tokens
            weight_mult = 1.6 if is_attr else 1.0

            for record_id, tf in postings:
                doc_len = self.index.doc_lengths.get(record_id, 1)

                # BM25 term weight
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / avg_dl))
                term_score = weight_mult * idf * (numerator / denominator)

                doc_scores[record_id] = doc_scores.get(record_id, 0.0) + term_score

        if not doc_scores:
            return []

        # Sort documents descending by BM25 score
        sorted_records = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []
        for rank, (record_id, score) in enumerate(sorted_records, 1):
            record = self.index.get_record(record_id)
            if not record:
                continue

            results.append(SearchResult(
                record_id=record.record_id,
                doc_id=record.doc_id,
                sentence_id=record.sentence_id,
                text=record.text,
                page_title=record.page_title,
                score=float(score),
                rank=rank,
                retrieval_method="BM25"
            ))

        return results
