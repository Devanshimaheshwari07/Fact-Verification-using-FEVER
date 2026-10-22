"""Official FEVER benchmark evaluation procedure (FEVER Score & Evidence Set Completeness)."""
from typing import List, Set, Tuple, Dict, Any
from src.data.load_data import FeverClaim, EvidenceItem


def is_evidence_set_satisfied(
    retrieved_items: Set[Tuple[str, int]],
    evidence_set: List[EvidenceItem]
) -> bool:
    """Checks whether every sentence required by an evidence set is present in retrieved items."""
    if not evidence_set:
        return False

    for item in evidence_set:
        if item.page_id is None or item.sentence_id is None:
            continue
        if (item.page_id, item.sentence_id) not in retrieved_items:
            return False
    return True


def evaluate_fever_instance(
    claim: FeverClaim,
    predicted_label: str,
    retrieved_evidence: List[Tuple[str, int]]
) -> Dict[str, bool]:
    """Evaluates a single claim against FEVER benchmark scoring criteria.

    Returns:
        dict with:
            - label_correct: bool
            - evidence_satisfied: bool
            - fever_correct: bool
    """
    label_correct = (predicted_label.upper() == claim.label.upper())

    if claim.label == "NOT ENOUGH INFO":
        # For NEI, evidence is not required under FEVER rules
        return {
            "label_correct": label_correct,
            "evidence_satisfied": True,
            "fever_correct": label_correct
        }

    # For SUPPORTS and REFUTES, at least one complete evidence set must be retrieved
    retrieved_set = set(retrieved_evidence)
    evidence_satisfied = False

    for ev_set in claim.evidence_sets:
        if is_evidence_set_satisfied(retrieved_set, ev_set):
            evidence_satisfied = True
            break

    fever_correct = label_correct and evidence_satisfied

    return {
        "label_correct": label_correct,
        "evidence_satisfied": evidence_satisfied,
        "fever_correct": fever_correct
    }


def compute_fever_scores(
    claims: List[FeverClaim],
    predictions: List[str],
    retrieved_lists: List[List[Tuple[str, int]]]
) -> Dict[str, Any]:
    """Computes overall FEVER score, Label Accuracy, and Evidence Recall across a test split."""
    total = len(claims)
    if total == 0:
        return {
            "total_claims": 0,
            "label_accuracy": 0.0,
            "evidence_recall": 0.0,
            "fever_score": 0.0
        }

    label_correct_count = 0
    evidence_satisfied_count = 0
    fever_correct_count = 0
    verifiable_count = 0

    for claim, pred_label, retrieved in zip(claims, predictions, retrieved_lists):
        res = evaluate_fever_instance(claim, pred_label, retrieved)

        if res["label_correct"]:
            label_correct_count += 1

        if claim.label != "NOT ENOUGH INFO":
            verifiable_count += 1
            if res["evidence_satisfied"]:
                evidence_satisfied_count += 1

        if res["fever_correct"]:
            fever_correct_count += 1

    label_acc = label_correct_count / total
    ev_recall = evidence_satisfied_count / verifiable_count if verifiable_count > 0 else 0.0
    fever_score = fever_correct_count / total

    return {
        "total_claims": total,
        "verifiable_claims": verifiable_count,
        "label_accuracy": round(float(label_acc), 4),
        "evidence_recall": round(float(ev_recall), 4),
        "fever_score": round(float(fever_score), 4),
        "label_correct_count": label_correct_count,
        "fever_correct_count": fever_correct_count
    }
