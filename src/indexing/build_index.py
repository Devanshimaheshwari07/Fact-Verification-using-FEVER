"""Index building pipeline for Wikipedia knowledge base."""
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz, load_npz

from src.data.load_data import EvidenceSentence, stream_wiki_sentences
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex
from src.utils.config_loader import get_storage_dir

logger = logging.getLogger(__name__)


class IndexManager:
    """Manages the creation, persistence, and loading of lexical and TF-IDF indexes."""

    def __init__(
        self,
        index_dir: str = "artifacts/indexes",
        preprocessor: Optional[TextPreprocessor] = None
    ):
        self.raw_index_dir = index_dir
        self.index_dir = get_storage_dir(index_dir, for_writing=False)
        self.preprocessor = preprocessor or TextPreprocessor()
        self.inverted_index: Optional[InvertedIndex] = None
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.record_ids: List[int] = []

    def build_from_corpus(
        self,
        corpus_path: Any,
        max_pages: Optional[int] = None,
        save_artifacts: bool = True
    ) -> Dict[str, Any]:
        """Builds inverted index and TF-IDF representations from one or more corpus files/directories."""
        paths = corpus_path if isinstance(corpus_path, (list, tuple)) else [corpus_path]
        logger.info(f"Building indexes from {paths}...")
        start_time = time.time()

        inv_index = InvertedIndex()
        raw_texts: List[str] = []
        record_ids: List[int] = []

        record_id = 0
        seen_pages = set()
        for p in paths:
            for sent in stream_wiki_sentences(str(p)):
                # Tokens for inverted index (with stemming and normalization)
                tokens = self.preprocessor.preprocess(sent.text)
                inv_index.add_sentence(
                    record_id=record_id,
                    doc_id=sent.doc_id,
                    sentence_id=sent.sentence_id,
                    text=sent.text,
                    page_title=sent.page_title,
                    tokens=tokens
                )
                raw_texts.append(sent.text)
                record_ids.append(record_id)
                record_id += 1
                seen_pages.add(sent.doc_id)
                if max_pages and len(seen_pages) >= max_pages:
                    break
            if max_pages and len(seen_pages) >= max_pages:
                break

        inv_index.finalize()
        self.inverted_index = inv_index
        self.record_ids = record_ids

        # Build TF-IDF vectorizer and sparse matrix
        logger.info(f"Fitting TF-IDF vectorizer on {len(raw_texts)} sentences...")
        self.tfidf_vectorizer = TfidfVectorizer(
            tokenizer=self.preprocessor.preprocess,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=50000,
            token_pattern=None
        )
        if raw_texts:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(raw_texts)
        else:
            self.tfidf_matrix = None

        build_duration = time.time() - start_time
        logger.info(f"Index construction completed in {build_duration:.2f}s ({len(record_ids)} sentences).")

        stats = inv_index.get_statistics()
        stats["build_duration_seconds"] = round(build_duration, 3)

        if save_artifacts:
            self.save()

        return stats

    def save(self) -> None:
        """Saves inverted index, TF-IDF vectorizer, and matrix to disk."""
        self.index_dir = get_storage_dir(self.raw_index_dir, for_writing=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.inverted_index:
            self.inverted_index.save(str(self.index_dir))

        if self.tfidf_vectorizer:
            with open(self.index_dir / "tfidf_vectorizer.pkl", "wb") as f:
                pickle.dump(self.tfidf_vectorizer, f)

        if self.tfidf_matrix is not None:
            save_npz(self.index_dir / "tfidf_matrix.npz", self.tfidf_matrix)

        with open(self.index_dir / "record_ids.pkl", "wb") as f:
            pickle.dump(self.record_ids, f)

        logger.info(f"Successfully saved all index artifacts to {self.index_dir}")

    def load(self) -> None:
        """Loads inverted index, TF-IDF vectorizer, and matrix from disk."""
        if not self.index_dir.exists():
            raise FileNotFoundError(f"Index directory does not exist: {self.index_dir}")

        self.inverted_index = InvertedIndex.load(str(self.index_dir))

        vec_path = self.index_dir / "tfidf_vectorizer.pkl"
        if vec_path.exists():
            with open(vec_path, "rb") as f:
                self.tfidf_vectorizer = pickle.load(f)

        mat_path = self.index_dir / "tfidf_matrix.npz"
        if mat_path.exists():
            self.tfidf_matrix = load_npz(mat_path)

        ids_path = self.index_dir / "record_ids.pkl"
        if ids_path.exists():
            with open(ids_path, "rb") as f:
                self.record_ids = pickle.load(f)

        logger.info(f"Loaded all index artifacts from {self.index_dir}")

    def is_indexed(self) -> bool:
        """Checks if pre-built indexes exist on disk."""
        return (self.index_dir / "inverted_index.pkl").exists()
