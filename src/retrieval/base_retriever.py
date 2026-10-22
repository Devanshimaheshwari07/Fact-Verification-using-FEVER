"""Base retriever interface and data models."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass
class SearchResult:
    """Represents a ranked evidence sentence retrieved from the knowledge base."""
    record_id: int
    doc_id: str             # Wikipedia page ID (e.g., "Nikolaj_Coster-Waldau")
    sentence_id: int        # Sentence ID (0, 1, 2, ...)
    text: str               # Original sentence text
    page_title: str         # Readable title (e.g., "Nikolaj Coster-Waldau")
    score: float            # Ranking score (TF-IDF cosine similarity, BM25, etc.)
    rank: int               # 1-indexed rank position
    retrieval_method: str   # Name of retrieval algorithm used

    def to_dict(self) -> Dict[str, Any]:
        """Converts result to serializable dictionary."""
        return {
            "record_id": self.record_id,
            "doc_id": self.doc_id,
            "sentence_id": self.sentence_id,
            "text": self.text,
            "page_title": self.page_title,
            "score": round(float(self.score), 4),
            "rank": self.rank,
            "retrieval_method": self.retrieval_method
        }


class BaseRetriever(ABC):
    """Abstract base class for all retrieval methods."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves top-k evidence sentences matching the query.

        Args:
            query: Natural language factual claim or query.
            top_k: Number of evidence candidates to return.

        Returns:
            List of SearchResult objects ordered by descending score.
        """
        pass
