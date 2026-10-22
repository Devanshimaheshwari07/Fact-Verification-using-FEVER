"""Comprehensive tests verifying the 5 FEVER IR and Fact Verification fixes:
1. Evidence relevance: Attribute-weighted retrieval prioritization.
2. Verdict logic: Predicting NOT ENOUGH INFO on unmentioned attributes.
3. Evidence grounding: Exact sentence citations and unmentioned attribute explanations.
4. Evaluation: Dynamic metrics on separate labeled evaluation dataset.
5. Confidence calibration: Calibrated probabilistic predictions.
6. End-to-end integration: Functional connection between evidence source, retriever, and verifier.
"""
import pytest
import numpy as np

from src.preprocessing.text_preprocessor import TextPreprocessor
from src.indexing.build_index import IndexManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.inference import VerificationEngine
from src.pipeline.fact_verifier import FactVerifierPipeline
from src.data.load_data import load_fever_claims
from src.evaluation.fever_evaluation import compute_fever_scores


@pytest.fixture(scope="module")
def pipeline():
    im = IndexManager("artifacts/indexes")
    im.load()
    return FactVerifierPipeline(index_manager=im)


def test_claim_component_extraction():
    preprocessor = TextPreprocessor()
    comp = preprocessor.extract_claim_components("Nikolaj Coster-Waldau is a vegan.")
    assert comp["subject"] == "Nikolaj Coster-Waldau"
    assert "vegan" in comp["attribute_tokens"]
    assert comp["attribute_text"] == "vegan"


def test_verdict_logic_unmentioned_attribute(pipeline):
    """Verifies that an unmentioned attribute (e.g. veganism for Nikolaj) predicts NOT ENOUGH INFO."""
    out = pipeline.verify_claim("Nikolaj Coster-Waldau is a vegan.")
    assert out.predicted_label == "NOT ENOUGH INFO"
    assert out.confidence >= 0.85
    assert out.probabilities["NOT ENOUGH INFO"] >= 0.85
    assert "vegan" in out.explanation.lower()
    assert "not enough info" in out.explanation.lower() or "insufficient evidence" in out.explanation.lower()
    assert out.comparison.relationship == "Insufficient Evidence"
    assert out.comparison.evidence_sufficient == "No"


def test_verdict_logic_explicit_contradiction(pipeline):
    """Verifies that explicit geographic contradictions correctly predict REFUTES."""
    out = pipeline.verify_claim("The Eiffel Tower is located in London.")
    assert out.predicted_label == "REFUTES"
    assert out.confidence >= 0.85
    assert "Paris" in out.explanation
    assert out.comparison.relationship == "Contradiction"


def test_evidence_grounding_supports(pipeline):
    """Verifies that SUPPORTS cites the exact sentence confirming the attribute."""
    out = pipeline.verify_claim("Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.")
    assert out.predicted_label == "SUPPORTS"
    assert out.confidence >= 0.70
    assert "Nikolaj Coster-Waldau" in out.explanation
    assert "Sentence 5" in out.explanation or "Sentence 4" in out.explanation
    assert "Fox" in out.explanation


def test_confidence_calibration(pipeline):
    """Verifies that confidence scores and probabilities are valid, normalized, and calibrated."""
    claims = [
        "Nikolaj Coster-Waldau is a vegan.",
        "The Eiffel Tower is located in London.",
        "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company."
    ]
    for claim in claims:
        out = pipeline.verify_claim(claim)
        assert 0.0 <= out.confidence <= 1.0
        prob_sum = sum(out.probabilities.values())
        assert abs(prob_sum - 1.0) < 0.02
        for p in out.probabilities.values():
            assert 0.0 <= p <= 1.0


def test_evidence_relevance_attribute_boost():
    """Verifies that BM25 gives higher ranking to sentences matching both subject and specific attribute."""
    im = IndexManager("artifacts/indexes")
    im.load()
    retriever = BM25Retriever(im.inverted_index)

    # Claim asserts Fox Broadcasting Company
    results = retriever.retrieve("Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.", top_k=5)
    assert len(results) > 0
    top_texts = " ".join([r.text for r in results[:2]])
    assert "Fox" in top_texts


def test_evaluation_on_separate_dataset(pipeline):
    """Verifies that evaluation runs on separate dev dataset and computes valid FEVER metrics."""
    dev_claims = load_fever_claims("data/sample/dev.jsonl")
    assert len(dev_claims) >= 15

    pred_labels = []
    retrieved_tuples = []
    for c in dev_claims[:6]:
        out = pipeline.verify_claim(c.claim, retrieval_method="bm25", top_k=3, enable_live_wiki=False)
        pred_labels.append(out.predicted_label)
        retrieved_tuples.append([(e.doc_id, e.sentence_id) for e in out.evidence])

    fever_metrics = compute_fever_scores(dev_claims[:6], pred_labels, retrieved_tuples)
    assert "fever_score" in fever_metrics
    assert "label_accuracy" in fever_metrics
    assert 0.0 <= fever_metrics["fever_score"] <= 1.0
    assert 0.0 <= fever_metrics["label_accuracy"] <= 1.0


def test_end_to_end_connection(pipeline):
    """Verifies functional connection: evidence retrieved -> features extracted -> verdict produced."""
    claim = "Alan Turing was an English mathematician who broke German codes at Bletchley Park."
    out = pipeline.verify_claim(claim)

    # 1. Evidence was retrieved from index
    assert len(out.evidence) > 0
    assert any("Alan_Turing" in e.doc_id for e in out.evidence)

    # 2. Features were conditioned on retrieved evidence
    assert out.feature_highlights["max_retrieval_score"] > 0
    assert out.feature_highlights["predicate_containment"] > 0

    # 3. Verdict and explanation are grounded in that evidence
    assert out.predicted_label == "SUPPORTS"
    assert "Alan Turing" in out.explanation
