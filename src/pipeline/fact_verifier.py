"""End-to-end Fact Verifier pipeline orchestrator."""
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.indexing.inverted_index import InvertedIndex
from src.indexing.build_index import IndexManager
from src.retrieval.base_retriever import BaseRetriever, SearchResult
from src.retrieval.boolean_retriever import BooleanRetriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.semantic_retriever import SemanticRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.verification.inference import VerificationEngine, VerificationOutput
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.train_classifier import VerificationModelTrainer
from src.utils.database import ExperimentDatabase

logger = logging.getLogger(__name__)


class FactVerifierPipeline:
    """End-to-end orchestrator for claim preprocessing, retrieval, and evidence-grounded verification."""

    def __init__(
        self,
        index_manager: Optional[IndexManager] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        db_path: str = "artifacts/results/fever_ir.db",
        model_dir: str = "artifacts/models"
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.index_manager = index_manager or IndexManager(preprocessor=self.preprocessor)
        self.db = ExperimentDatabase(db_path=db_path)
        self.model_dir = Path(model_dir)

        # Retrieval models
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.verification_engine: Optional[VerificationEngine] = None

        self._init_components()

    def _init_components(self) -> None:
        """Initializes all retrieval engines and verification models."""
        if not self.index_manager.inverted_index:
            try:
                self.index_manager.load()
            except Exception as e:
                logger.info(f"Could not load pre-existing index ({e}). Auto-building from sample corpus...")
                sample_path = Path("data/sample/wiki_sample.jsonl")
                if sample_path.exists():
                    try:
                        self.index_manager.build_from_corpus(
                            corpus_path=[sample_path],
                            save_artifacts=False
                        )
                        logger.info("Auto-built in-memory inverted index and TF-IDF representations.")
                    except Exception as build_err:
                        logger.error(f"Failed to auto-build index: {build_err}")

        if not self.index_manager.inverted_index:
            logger.warning("No index available; local retrieval disabled.")
            return

        inv_index = self.index_manager.inverted_index

        # Register retrievers
        bm25 = BM25Retriever(index=inv_index, preprocessor=self.preprocessor)
        tfidf = TfidfRetriever(index_manager=self.index_manager, preprocessor=self.preprocessor)
        boolean = BooleanRetriever(index=inv_index, preprocessor=self.preprocessor)
        semantic = SemanticRetriever(index_manager=self.index_manager)
        hybrid = HybridRetriever(sparse_retriever=bm25, dense_retriever=semantic, preprocessor=self.preprocessor)

        self.retrievers = {
            "bm25": bm25,
            "tfidf": tfidf,
            "boolean": boolean,
            "semantic": semantic,
            "hybrid": hybrid
        }

        # Initialize verification classifier
        trainer = VerificationModelTrainer(model_dir=str(self.model_dir))
        classifier_model = None
        try:
            trainer.load()
            classifier_model = trainer.model
        except Exception:
            logger.info("Trained classifier artifact not yet available; operating with rule-grounded verifier.")

        self.verification_engine = VerificationEngine(
            model=classifier_model,
            feature_extractor=VerificationFeatureExtractor(self.preprocessor),
            preprocessor=self.preprocessor
        )

    def is_ready(self) -> bool:
        """Returns True if index is loaded and retrieval engines are active."""
        return len(self.retrievers) > 0 and self.verification_engine is not None

    def get_retriever(self, method: str = "bm25") -> BaseRetriever:
        """Returns the requested retrieval engine."""
        method_lower = method.lower()
        if method_lower not in self.retrievers:
            logger.warning(f"Retriever '{method}' not found. Defaulting to 'bm25'.")
            return self.retrievers.get("bm25")
        return self.retrievers[method_lower]

    def verify_claim(
        self,
        claim_text: str,
        retrieval_method: str = "bm25",
        top_k: int = 5,
        log_to_db: bool = True,
        enable_live_wiki: bool = True
    ) -> VerificationOutput:
        """Runs the complete verification pipeline for a user-entered claim."""
        if not self.is_ready():
            self._init_components()
            if not self.is_ready():
                raise RuntimeError("System index is not initialized. Please build or load the index first.")

        retriever = self.get_retriever(retrieval_method)

        # 1. Retrieve candidate evidence sentences from local index
        start_time = time.time()
        retrieved_evidence = retriever.retrieve(claim_text, top_k=top_k)
        retrieval_duration = time.time() - start_time

        # 2. Dynamic Live Wikipedia Fallback if local matches are weak
        max_score = retrieved_evidence[0].score if retrieved_evidence else 0.0
        if enable_live_wiki and (not retrieved_evidence or max_score < 3.5):
            try:
                from src.retrieval.wiki_live import live_search
                live_evidence = live_search(claim_text, preprocessor=self.preprocessor, top_k=top_k)
                if live_evidence:
                    if not retrieved_evidence or (live_evidence[0].score > max_score):
                        retrieved_evidence = live_evidence
            except Exception as e:
                logger.warning(f"Live Wikipedia search encountered error: {e}")

        # 3. Fact verification and explanation generation
        output = self.verification_engine.verify(claim_text, retrieved_evidence)

        # 3. Log to SQLite database for audit and caching
        if log_to_db:
            try:
                self.db.log_claim_verification(
                    claim_text=claim_text,
                    retrieval_method=retrieval_method,
                    top_k=top_k,
                    predicted_label=output.predicted_label,
                    confidence=output.confidence,
                    explanation=output.explanation,
                    evidence=[e.to_dict() for e in output.evidence]
                )
            except Exception as e:
                logger.warning(f"Could not log verification to DB: {e}")

        return output
