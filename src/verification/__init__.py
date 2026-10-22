"""Fact verification package."""
from src.verification.label_mapping import LABEL_TO_ID, ID_TO_LABEL, ALL_LABELS, normalize_label
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.train_classifier import VerificationModelTrainer
from src.verification.inference import VerificationEngine, VerificationOutput

__all__ = [
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "ALL_LABELS",
    "normalize_label",
    "VerificationFeatureExtractor",
    "VerificationModelTrainer",
    "VerificationEngine",
    "VerificationOutput"
]
