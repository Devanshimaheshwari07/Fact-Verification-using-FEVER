"""Classification evaluation metrics for fact verification models."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

from src.verification.label_mapping import ALL_LABELS, LABEL_TO_ID, normalize_label


def evaluate_classifier_performance(
    gold_labels: List[str],
    pred_labels: List[str]
) -> Dict[str, Any]:
    """Computes comprehensive multi-class classification metrics."""
    normalized_gold = [normalize_label(g) for g in gold_labels]
    normalized_pred = [normalize_label(p) for p in pred_labels]

    acc = accuracy_score(normalized_gold, normalized_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        normalized_gold,
        normalized_pred,
        labels=ALL_LABELS,
        zero_division=0
    )

    per_class_metrics = {}
    for i, label in enumerate(ALL_LABELS):
        per_class_metrics[label] = {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1_score": round(float(f1[i]), 4),
            "support": int(support[i])
        }

    macro_f1 = float(np.mean(f1))
    weighted_f1 = float(np.average(f1, weights=support)) if sum(support) > 0 else 0.0

    cm = confusion_matrix(normalized_gold, normalized_pred, labels=ALL_LABELS)

    report_str = classification_report(
        normalized_gold,
        normalized_pred,
        labels=ALL_LABELS,
        zero_division=0
    )

    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "labels": ALL_LABELS,
        "classification_report_text": report_str
    }
