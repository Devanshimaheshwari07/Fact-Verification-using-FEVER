"""Boolean Retrieval implementation supporting AND, OR, and NOT operations."""
import logging
import re
from typing import List, Set, Optional

from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.indexing.inverted_index import InvertedIndex
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class BooleanRetriever(BaseRetriever):
    """Executes Boolean queries (AND, OR, NOT) against the InvertedIndex."""

    def __init__(self, index: InvertedIndex, preprocessor: Optional[TextPreprocessor] = None):
        self.index = index
        self.preprocessor = preprocessor or TextPreprocessor()

    def _get_term_docs(self, term: str) -> Set[int]:
        """Returns set of record IDs containing the term."""
        processed_terms = self.preprocessor.preprocess(term)
        if not processed_terms:
            return set()
        matched_records = set()
        for t in processed_terms:
            postings = self.index.get_postings(t)
            matched_records.update(record_id for record_id, _ in postings)
        return matched_records

    def evaluate_query(self, query: str) -> Set[int]:
        """Evaluates a Boolean query containing AND, OR, NOT operators.

        Examples:
        - "Turing AND Enigma"
        - "Armstrong OR Aldrin"
        - "Apollo NOT Collins"
        """
        all_docs = set(self.index.records.keys())
        query = query.strip()
        if not query:
            return set()

        # Check for explicit Boolean operators
        has_and = " AND " in query
        has_or = " OR " in query
        has_not = " NOT " in query

        if has_or:
            parts = [p.strip() for p in query.split(" OR ")]
            result_docs = set()
            for part in parts:
                result_docs.update(self.evaluate_query(part))
            return result_docs

        if has_and:
            parts = [p.strip() for p in query.split(" AND ")]
            if not parts:
                return set()
            result_docs = self.evaluate_query(parts[0])
            for part in parts[1:]:
                result_docs.intersection_update(self.evaluate_query(part))
            return result_docs

        if has_not:
            parts = [p.strip() for p in query.split(" NOT ")]
            if not parts:
                return set()
            result_docs = self.evaluate_query(parts[0])
            for part in parts[1:]:
                excluded_docs = self.evaluate_query(part)
                result_docs.difference_update(excluded_docs)
            return result_docs

        # Default fallback: Free-text query. Treat as OR if multiple terms,
        # but prioritize records with more term matches.
        terms = self.preprocessor.preprocess(query)
        if not terms:
            return set()

        matching_docs: Set[int] = set()
        for t in terms:
            postings = self.index.get_postings(t)
            matching_docs.update(record_id for record_id, _ in postings)

        return matching_docs

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves and ranks records matching the Boolean expression."""
        matched_doc_ids = self.evaluate_query(query)
        if not matched_doc_ids:
            return []

        # Score matched documents by number of query terms they contain
        query_terms = self.preprocessor.preprocess(query)
        scored_records = []

        term_counts: dict = {}
        for t in query_terms:
            for r_id, _ in self.index.get_postings(t):
                term_counts[r_id] = term_counts.get(r_id, 0) + 1

        for doc_id in matched_doc_ids:
            record = self.index.get_record(doc_id)
            if not record:
                continue

            score = float(term_counts.get(doc_id, 1.0))
            scored_records.append((doc_id, score, record))

        # Sort descending by match score, then by record_id
        scored_records.sort(key=lambda x: (x[1], -x[0]), reverse=True)

        results = []
        for rank, (doc_id, score, record) in enumerate(scored_records[:top_k], 1):
            results.append(SearchResult(
                record_id=record.record_id,
                doc_id=record.doc_id,
                sentence_id=record.sentence_id,
                text=record.text,
                page_title=record.page_title,
                score=score,
                rank=rank,
                retrieval_method="Boolean"
            ))

        return results
