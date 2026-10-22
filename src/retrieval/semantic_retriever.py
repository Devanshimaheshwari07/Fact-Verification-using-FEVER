"""Semantic retrieval model using Sentence Transformers or dense semantic vector projection."""
import logging
from typing import List, Optional
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.indexing.inverted_index import InvertedIndex
from src.indexing.build_index import IndexManager

logger = logging.getLogger(__name__)


class SemanticRetriever(BaseRetriever):
    """Dense semantic vector retrieval model.

    Implements dense semantic vector representations for passage retrieval:
    - Pretrained Sentence Transformers (e.g. all-MiniLM-L6-v2) when torch/transformers available.
    - Dense Latent Semantic Vector Space (TruncatedSVD projection) with cosine similarity.
    """

    def __init__(
        self,
        index_manager: IndexManager,
        model_name: str = "all-MiniLM-L6-v2",
        dense_dim: int = 64
    ):
        self.index_manager = index_manager
        self.model_name = model_name
        self.dense_dim = dense_dim
        self.transformer_model = None
        self.is_neural_active = False

        # Dense latent vector structures
        self.svd_model: Optional[TruncatedSVD] = None
        self.doc_dense_embeddings: Optional[np.ndarray] = None

        self._init_model()

    def _init_model(self) -> None:
        """Attempts to load sentence-transformers model; falls back gracefully if unavailable."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self.transformer_model = SentenceTransformer(self.model_name)
            self.is_neural_active = True
            logger.info("SentenceTransformer model loaded successfully.")
        except (ImportError, Exception) as e:
            logger.info(
                f"SentenceTransformer not available ({e}). "
                "SemanticRetriever will operate using Dense Semantic Vector Projection (LSA)."
            )
            self.transformer_model = None
            self.is_neural_active = False
            self._ensure_dense_index()

    def _ensure_dense_index(self) -> None:
        """Constructs dense semantic vector index over indexed passages."""
        if self.doc_dense_embeddings is not None:
            return

        if self.index_manager.tfidf_matrix is not None:
            matrix = self.index_manager.tfidf_matrix
            n_features = matrix.shape[1]
            n_components = min(self.dense_dim, n_features - 1) if n_features > 2 else 2
            try:
                svd = TruncatedSVD(n_components=n_components, random_state=42)
                dense_embs = svd.fit_transform(matrix)
                self.doc_dense_embeddings = normalize(dense_embs, norm='l2', axis=1)
                self.svd_model = svd
                logger.info(f"Built dense semantic vector index: shape {self.doc_dense_embeddings.shape}")
            except Exception as e:
                logger.warning(f"Could not build dense vector projection: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieves semantically similar evidence sentences using dense vector similarity."""
        if not query or not query.strip():
            return []

        if self.index_manager.inverted_index is None:
            return []

        num_records = self.index_manager.inverted_index.total_documents
        if num_records == 0:
            return []

        # 1. Pretrained Sentence Transformer Dense Vector Model
        if self.is_neural_active and self.transformer_model is not None:
            try:
                sentences = [
                    self.index_manager.inverted_index.get_record(r_id).text
                    for r_id in self.index_manager.record_ids
                ]
                query_emb = self.transformer_model.encode([query], normalize_embeddings=True)
                doc_embs = self.transformer_model.encode(sentences, normalize_embeddings=True)

                scores = np.dot(doc_embs, query_emb.T).ravel()
                top_indices = np.argsort(-scores)[:top_k]

                results = []
                for rank, idx in enumerate(top_indices, 1):
                    record_id = self.index_manager.record_ids[idx]
                    record = self.index_manager.inverted_index.get_record(record_id)
                    results.append(SearchResult(
                        record_id=record.record_id,
                        doc_id=record.doc_id,
                        sentence_id=record.sentence_id,
                        text=record.text,
                        page_title=record.page_title,
                        score=float(scores[idx]),
                        rank=rank,
                        retrieval_method="Sentence Embedding (Transformer)"
                    ))
                return results
            except Exception as e:
                logger.warning(f"Neural encoding failed: {e}. Falling back to Dense Semantic Projection.")

        # 2. Dense Semantic Vector Projection Model
        self._ensure_dense_index()
        if (
            self.svd_model is not None
            and self.doc_dense_embeddings is not None
            and self.index_manager.tfidf_vectorizer is not None
        ):
            query_sparse = self.index_manager.tfidf_vectorizer.transform([query])
            if query_sparse.nnz == 0:
                return []

            query_dense = self.svd_model.transform(query_sparse)
            query_dense_norm = normalize(query_dense, norm='l2', axis=1)

            # Cosine similarity in dense semantic embedding space
            scores = (self.doc_dense_embeddings @ query_dense_norm.T).ravel()
            top_indices = np.argsort(-scores)[:top_k]

            results = []
            for rank, idx in enumerate(top_indices, 1):
                if scores[idx] <= 1e-4:
                    continue
                record_id = self.index_manager.record_ids[idx]
                record = self.index_manager.inverted_index.get_record(record_id)
                results.append(SearchResult(
                    record_id=record.record_id,
                    doc_id=record.doc_id,
                    sentence_id=record.sentence_id,
                    text=record.text,
                    page_title=record.page_title,
                    score=float(scores[idx]),
                    rank=rank,
                    retrieval_method="Sentence Embedding (Dense Vector)"
                ))
            return results

        return []
