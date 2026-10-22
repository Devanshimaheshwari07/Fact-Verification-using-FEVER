"""Inverted Index implementation for sentence-level evidence retrieval."""
import json
import logging
import math
import os
import pickle
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Metadata record for an indexed evidence sentence."""
    record_id: int
    doc_id: str             # Wikipedia page ID (e.g. "Alan_Turing")
    sentence_id: int        # Sentence ID on page (0, 1, 2, ...)
    text: str               # Original text
    page_title: str         # Human-readable title
    token_count: int        # Number of tokens in sentence


class InvertedIndex:
    """Inverted index mapping terms to postings lists with term frequencies and collection stats."""

    def __init__(self):
        # term -> List of (record_id, term_frequency)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        # record_id -> DocumentRecord
        self.records: Dict[int, DocumentRecord] = {}
        # record_id -> token length
        self.doc_lengths: Dict[int, int] = {}
        # Total tokens across all indexed sentences
        self.total_tokens: int = 0
        # Average document length
        self.avg_doc_length: float = 0.0

    @property
    def total_documents(self) -> int:
        return len(self.records)

    @property
    def vocabulary_size(self) -> int:
        return len(self.postings)

    def add_sentence(
        self,
        record_id: int,
        doc_id: str,
        sentence_id: int,
        text: str,
        page_title: str,
        tokens: List[str]
    ) -> None:
        """Indexes a single sentence record."""
        token_count = len(tokens)
        record = DocumentRecord(
            record_id=record_id,
            doc_id=doc_id,
            sentence_id=sentence_id,
            text=text,
            page_title=page_title,
            token_count=token_count
        )
        self.records[record_id] = record
        self.doc_lengths[record_id] = token_count
        self.total_tokens += token_count

        # Count frequencies in this document
        tf_map: Dict[str, int] = {}
        for token in tokens:
            tf_map[token] = tf_map.get(token, 0) + 1

        # Append to postings
        for token, count in tf_map.items():
            if token not in self.postings:
                self.postings[token] = []
            self.postings[token].append((record_id, count))

    def finalize(self) -> None:
        """Computes summary statistics after all documents are added."""
        if self.total_documents > 0:
            self.avg_doc_length = self.total_tokens / self.total_documents
        else:
            self.avg_doc_length = 0.0

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """Returns postings list for a term, or empty list if not in vocabulary."""
        return self.postings.get(term, [])

    def get_doc_frequency(self, term: str) -> int:
        """Returns number of documents containing term."""
        return len(self.postings.get(term, []))

    def get_record(self, record_id: int) -> Optional[DocumentRecord]:
        """Returns metadata for a record ID."""
        return self.records.get(record_id)

    def get_statistics(self) -> Dict[str, Any]:
        """Returns comprehensive index diagnostics."""
        total_postings = sum(len(p) for p in self.postings.values())
        return {
            "total_documents": self.total_documents,
            "vocabulary_size": self.vocabulary_size,
            "total_postings_entries": total_postings,
            "average_document_length": round(self.avg_doc_length, 2),
            "total_indexed_tokens": self.total_tokens
        }

    def save(self, output_dir: str) -> None:
        """Serializes the index and records to disk."""
        target_path = Path(output_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        index_file = target_path / "inverted_index.pkl"
        meta_file = target_path / "index_metadata.json"

        data = {
            "postings": self.postings,
            "records": self.records,
            "doc_lengths": self.doc_lengths,
            "total_tokens": self.total_tokens,
            "avg_doc_length": self.avg_doc_length
        }

        with open(index_file, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

        stats = self.get_statistics()
        stats["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved InvertedIndex with {self.total_documents} records to {target_path}")

    @classmethod
    def load(cls, index_dir: str) -> "InvertedIndex":
        """Loads a serialized InvertedIndex from disk."""
        target_path = Path(index_dir)
        index_file = target_path / "inverted_index.pkl"

        if not index_file.exists():
            raise FileNotFoundError(f"Index file not found at {index_file}")

        with open(index_file, "rb") as f:
            data = pickle.load(f)

        idx = cls()
        idx.postings = data["postings"]
        idx.records = data["records"]
        idx.doc_lengths = data["doc_lengths"]
        idx.total_tokens = data["total_tokens"]
        idx.avg_doc_length = data["avg_doc_length"]

        logger.info(f"Loaded InvertedIndex with {idx.total_documents} records from {target_path}")
        return idx
