"""Model training script and manager for fact verification classifiers."""
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from src.data.load_data import FeverClaim, load_fever_claims
from src.retrieval.base_retriever import BaseRetriever
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.label_mapping import LABEL_TO_ID, ID_TO_LABEL
from src.utils.config_loader import get_storage_dir

logger = logging.getLogger(__name__)


class VerificationModelTrainer:
    """Trains and serializes the evidence-grounded claim verification classifier."""

    def __init__(
        self,
        classifier_type: str = "logistic_regression",
        model_dir: str = "artifacts/models",
        random_state: int = 42
    ):
        self.classifier_type = classifier_type
        self.raw_model_dir = model_dir
        self.model_dir = get_storage_dir(model_dir, for_writing=False)
        self.random_state = random_state
        self.feature_extractor = VerificationFeatureExtractor()
        self.model = None

    def _init_classifier(self):
        """Initializes calibrated sklearn estimator using Platt scaling / sigmoid calibration."""
        if self.classifier_type == "random_forest":
            base = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                class_weight="balanced",
                random_state=self.random_state
            )
        else:
            base = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                C=1.0,
                random_state=self.random_state
            )
        return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)

    def prepare_dataset_features(
        self,
        claims: List[FeverClaim],
        retriever: BaseRetriever,
        top_k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts feature vectors and target labels from claims by running retrieval."""
        X_list = []
        y_list = []

        logger.info(f"Extracting features for {len(claims)} claims using {retriever.__class__.__name__}...")

        for claim in claims:
            # Retrieve evidence without exposing gold label
            evidence = retriever.retrieve(claim.claim, top_k=top_k)
            feat = self.feature_extractor.extract_features(claim.claim, evidence)
            label_id = LABEL_TO_ID[claim.label]

            X_list.append(feat)
            y_list.append(label_id)

        X = np.vstack(X_list)
        y = np.array(y_list, dtype=np.int64)
        return X, y

    def train(
        self,
        train_claims: List[FeverClaim],
        retriever: BaseRetriever,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Trains verification model on training claims."""
        start_time = time.time()
        X_train, y_train = self.prepare_dataset_features(train_claims, retriever, top_k=top_k)

        self.model = self._init_classifier()
        self.model.fit(X_train, y_train)

        train_preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, train_preds)
        macro_f1 = f1_score(y_train, train_preds, average="macro", zero_division=0)

        duration = time.time() - start_time
        logger.info(f"Model ({self.classifier_type}) trained in {duration:.2f}s | Train Acc: {acc:.4f} | Macro F1: {macro_f1:.4f}")

        # Save artifacts
        self.save()

        return {
            "classifier_type": self.classifier_type,
            "train_samples": len(train_claims),
            "train_accuracy": round(float(acc), 4),
            "train_macro_f1": round(float(macro_f1), 4),
            "training_time_seconds": round(duration, 3)
        }

    def save(self) -> None:
        """Saves model and metadata."""
        self.model_dir = get_storage_dir(self.raw_model_dir, for_writing=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.model_dir / "fever_classifier.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "classifier_type": self.classifier_type,
                "feature_names": self.feature_extractor.feature_names
            }, f)
        logger.info(f"Saved trained classifier to {model_path}")

    def load(self) -> None:
        """Loads trained model from disk."""
        model_path = self.model_dir / "fever_classifier.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found at {model_path}")

        with open(model_path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.classifier_type = data.get("classifier_type", self.classifier_type)
        logger.info(f"Loaded trained classifier ({self.classifier_type}) from {model_path}")
