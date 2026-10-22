"""Indexing package for FEVER knowledge base."""
from src.indexing.inverted_index import InvertedIndex, DocumentRecord
from src.indexing.build_index import IndexManager

__all__ = ["InvertedIndex", "DocumentRecord", "IndexManager"]
