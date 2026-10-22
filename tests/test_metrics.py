"""Unit tests for IR metrics, FEVER benchmark scorer, and classification evaluation."""
import pytest
from src.evaluation.retrieval_metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k
from src.evaluation.fever_evaluation import evaluate_fever_instance, compute_fever_scores
from src.evaluation.verification_metrics import evaluate_classifier_performance
from src.data.load_data import FeverClaim, EvidenceItem


def test_retrieval_metrics_calculation():
    gold = {("DocA", 1), ("DocA", 2)}
    retrieved = [("DocA", 1), ("DocB", 0), ("DocA", 2), ("DocC", 5)]

    assert precision_at_k(retrieved, gold, k=1) == 1.0  # 1/1
    assert precision_at_k(retrieved, gold, k=2) == 0.5  # 1/2
    assert recall_at_k(retrieved, gold, k=1) == 0.5     # 1/2
    assert recall_at_k(retrieved, gold, k=3) == 1.0     # 2/2
    assert reciprocal_rank(retrieved, gold) == 1.0      # Rank 1 hit
    assert ndcg_at_k(retrieved, gold, k=3) > 0.8


def test_fever_score_logic():
    # 1. SUPPORTS claim with gold evidence retrieved -> Correct
    claim_supp = FeverClaim(
        claim_id=1,
        claim="Turing worked at Bletchley Park.",
        label="SUPPORTS",
        verifiable="VERIFIABLE",
        evidence_sets=[[EvidenceItem("Alan_Turing", 3)]]
    )
    res_correct = evaluate_fever_instance(claim_supp, "SUPPORTS", [("Alan_Turing", 3)])
    assert res_correct["fever_correct"] is True

    # 2. SUPPORTS claim with wrong label -> Incorrect
    res_wrong_label = evaluate_fever_instance(claim_supp, "REFUTES", [("Alan_Turing", 3)])
    assert res_wrong_label["fever_correct"] is False

    # 3. SUPPORTS claim with correct label but missing evidence -> Incorrect
    res_missing_ev = evaluate_fever_instance(claim_supp, "SUPPORTS", [("Alan_Turing", 0)])
    assert res_missing_ev["fever_correct"] is False

    # 4. NOT ENOUGH INFO claim -> Evidence not required
    claim_nei = FeverClaim(
        claim_id=2,
        claim="Turing liked apples.",
        label="NOT ENOUGH INFO",
        verifiable="NOT VERIFIABLE",
        evidence_sets=[]
    )
    res_nei = evaluate_fever_instance(claim_nei, "NOT ENOUGH INFO", [])
    assert res_nei["fever_correct"] is True


def test_classification_metrics():
    gold = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO", "SUPPORTS"]
    pred = ["SUPPORTS", "REFUTES", "SUPPORTS", "SUPPORTS"]

    perf = evaluate_classifier_performance(gold, pred)
    assert perf["accuracy"] == 0.75
    assert "per_class" in perf
    assert perf["confusion_matrix"][0][0] == 2  # 2 correct SUPPORTS
