"""Evaluation metrics package."""
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval_batch
)
from src.evaluation.verification_metrics import evaluate_classifier_performance
from src.evaluation.fever_evaluation import (
    is_evidence_set_satisfied,
    evaluate_fever_instance,
    compute_fever_scores
)
from src.evaluation.error_analysis import perform_error_analysis

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "ndcg_at_k",
    "evaluate_retrieval_batch",
    "evaluate_classifier_performance",
    "is_evidence_set_satisfied",
    "evaluate_fever_instance",
    "compute_fever_scores",
    "perform_error_analysis"
]
