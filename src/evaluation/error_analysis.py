"""Fine-grained error analysis and categorization for IR fact verification."""
from typing import List, Dict, Any, Tuple
from src.data.load_data import FeverClaim
from src.evaluation.fever_evaluation import evaluate_fever_instance


def perform_error_analysis(
    claims: List[FeverClaim],
    predicted_labels: List[str],
    retrieved_evidence_tuples: List[List[Tuple[str, int]]]
) -> Dict[str, Any]:
    """Categorizes model errors into retrieval vs verification causes."""
    categories: Dict[str, List[Dict[str, Any]]] = {
        "retrieval_failure": [],         # Gold evidence missing from top-k
        "verification_failure": [],      # Evidence retrieved, but wrong label predicted
        "polarity_confusion": [],        # Predicted REFUTES for SUPPORTS, or vice-versa
        "spurious_verdict_on_nei": [],   # Predicted SUPPORTS/REFUTES when ground truth was NEI
        "false_nei": []                  # Predicted NEI when verifiable evidence was present
    }

    correct_instances = []

    for claim, pred_label, retrieved in zip(claims, predicted_labels, retrieved_evidence_tuples):
        eval_res = evaluate_fever_instance(claim, pred_label, retrieved)

        item_meta = {
            "claim_id": claim.claim_id,
            "claim": claim.claim,
            "gold_label": claim.label,
            "predicted_label": pred_label,
            "retrieved_count": len(retrieved),
            "evidence_satisfied": eval_res["evidence_satisfied"]
        }

        if eval_res["fever_correct"]:
            correct_instances.append(item_meta)
            continue

        # Error cases
        if claim.label != "NOT ENOUGH INFO" and not eval_res["evidence_satisfied"]:
            categories["retrieval_failure"].append(item_meta)

        if eval_res["evidence_satisfied"] and not eval_res["label_correct"]:
            categories["verification_failure"].append(item_meta)

        if (claim.label == "SUPPORTS" and pred_label == "REFUTES") or \
           (claim.label == "REFUTES" and pred_label == "SUPPORTS"):
            categories["polarity_confusion"].append(item_meta)

        if claim.label == "NOT ENOUGH INFO" and pred_label in {"SUPPORTS", "REFUTES"}:
            categories["spurious_verdict_on_nei"].append(item_meta)

        if claim.label in {"SUPPORTS", "REFUTES"} and pred_label == "NOT ENOUGH INFO":
            categories["false_nei"].append(item_meta)

    summary_counts = {cat: len(items) for cat, items in categories.items()}
    summary_counts["correct_fever_claims"] = len(correct_instances)
    summary_counts["total_evaluated"] = len(claims)

    return {
        "counts": summary_counts,
        "sample_cases": {cat: items[:5] for cat, items in categories.items()}
    }
