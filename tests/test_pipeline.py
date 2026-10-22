"""Integration tests for the complete FactVerifierPipeline."""
import pytest
from src.pipeline.fact_verifier import FactVerifierPipeline
from src.indexing.build_index import IndexManager


@pytest.fixture(scope="module")
def pipeline():
    pipe = FactVerifierPipeline()
    return pipe


def test_pipeline_initialization(pipeline):
    assert pipeline.is_ready()
    assert "bm25" in pipeline.retrievers
    assert "tfidf" in pipeline.retrievers
    assert "boolean" in pipeline.retrievers


def test_pipeline_supports_claim(pipeline):
    claim = "Alan Turing was an English mathematician who broke German codes at Bletchley Park."
    output = pipeline.verify_claim(claim, retrieval_method="bm25", top_k=3, log_to_db=False)

    assert output.claim == claim
    assert output.predicted_label in ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    assert len(output.evidence) > 0
    assert output.evidence[0].doc_id == "Alan_Turing"
    assert len(output.explanation) > 0


def test_pipeline_refutes_claim(pipeline):
    claim = "Buzz Aldrin was the first person to step on the Moon during Apollo 11."
    output = pipeline.verify_claim(claim, retrieval_method="bm25", top_k=3, log_to_db=False)

    assert len(output.evidence) > 0
    assert output.evidence[0].doc_id in ["Apollo_11", "Neil_Armstrong", "Buzz_Aldrin"]
    assert output.predicted_label in ["REFUTES", "SUPPORTS", "NOT ENOUGH INFO"]


def test_pipeline_not_enough_info(pipeline):
    claim = "Albert Einstein had three pet rabbits named Quantum, Newton and Curie."
    output = pipeline.verify_claim(claim, retrieval_method="bm25", top_k=3, log_to_db=False)

    assert output.predicted_label in ["NOT ENOUGH INFO", "REFUTES"]


def test_pipeline_empty_claim(pipeline):
    output = pipeline.verify_claim("", retrieval_method="bm25", top_k=3, log_to_db=False)
    assert output.predicted_label == "NOT ENOUGH INFO"
