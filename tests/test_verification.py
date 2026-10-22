"""Unit tests for feature extraction and fact verification engine."""
import pytest
import numpy as np

from src.retrieval.base_retriever import SearchResult
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.inference import VerificationEngine
from src.verification.label_mapping import LABEL_TO_ID, ID_TO_LABEL, normalize_label


def test_label_mappings():
    assert LABEL_TO_ID["SUPPORTS"] == 0
    assert LABEL_TO_ID["REFUTES"] == 1
    assert LABEL_TO_ID["NOT ENOUGH INFO"] == 2
    assert ID_TO_LABEL[0] == "SUPPORTS"
    assert normalize_label("supports") == "SUPPORTS"
    assert normalize_label("refutes") == "REFUTES"
    assert normalize_label("unsupported") == "NOT ENOUGH INFO"


def test_feature_extractor_dimensions():
    extractor = VerificationFeatureExtractor()
    claim = "Albert Einstein won the Nobel Prize in 1921."
    evidence = [
        SearchResult(
            record_id=0,
            doc_id="Albert_Einstein",
            sentence_id=3,
            text="He received the 1921 Nobel Prize in Physics for his discovery of the photoelectric effect.",
            page_title="Albert Einstein",
            score=12.5,
            rank=1,
            retrieval_method="BM25"
        )
    ]

    features = extractor.extract_features(claim, evidence)
    assert isinstance(features, np.ndarray)
    assert len(features) == len(extractor.feature_names)

    feat_dict = dict(zip(extractor.feature_names, features))
    assert feat_dict["numeric_overlap_ratio"] == 1.0  # 1921 matched
    assert feat_dict["numeric_mismatch"] == 0.0
    assert feat_dict["max_claim_containment"] > 0.5


def test_verification_engine_supports():
    engine = VerificationEngine(model=None)
    claim = "Marie Curie won two Nobel Prizes."
    evidence = [
        SearchResult(
            record_id=0,
            doc_id="Marie_Curie",
            sentence_id=1,
            text="She was the first person to win a Nobel Prize twice, in two scientific fields.",
            page_title="Marie Curie",
            score=10.0,
            rank=1,
            retrieval_method="BM25"
        )
    ]

    output = engine.verify(claim, evidence)
    assert output.predicted_label in ["SUPPORTS", "NOT ENOUGH INFO"]
    assert len(output.evidence) == 1
    assert "Marie Curie" in output.explanation


def test_verification_engine_empty_evidence():
    engine = VerificationEngine(model=None)
    output = engine.verify("An unverifiable assertion", [])
    assert output.predicted_label == "NOT ENOUGH INFO"
    assert output.confidence >= 0.90
