"""Information Retrieval evaluation metrics (Precision@K, Recall@K, MRR, nDCG@K)."""
import math
from typing import List, Set, Tuple, Dict, Any, Optional
import numpy as np


def precision_at_k(
    retrieved: List[Tuple[str, int]],
    gold_evidence: Set[Tuple[str, int]],
    k: int = 5
) -> float:
    """Computes Precision@K: fraction of top-k retrieved sentences that are gold evidence."""
    if k <= 0 or not retrieved:
        return 0.0

    top_k_retrieved = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in gold_evidence)
    return relevant_retrieved / len(top_k_retrieved)


def recall_at_k(
    retrieved: List[Tuple[str, int]],
    gold_evidence: Set[Tuple[str, int]],
    k: int = 5
) -> float:
    """Computes Recall@K: fraction of gold evidence sentences retrieved in top-k."""
    if not gold_evidence:
        return 1.0  # If no gold evidence needed, recall is vacuously satisfied

    top_k_retrieved = retrieved[:k]
    relevant_retrieved = sum(1 for item in top_k_retrieved if item in gold_evidence)
    return relevant_retrieved / len(gold_evidence)


def reciprocal_rank(
    retrieved: List[Tuple[str, int]],
    gold_evidence: Set[Tuple[str, int]]
) -> float:
    """Computes Reciprocal Rank (RR): 1 / rank of first relevant retrieved sentence."""
    if not gold_evidence:
        return 0.0

    for rank, item in enumerate(retrieved, 1):
        if item in gold_evidence:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: List[Tuple[str, int]],
    gold_evidence: Set[Tuple[str, int]],
    k: int = 5
) -> float:
    """Computes Normalized Discounted Cumulative Gain (nDCG@K)."""
    if not gold_evidence or k <= 0:
        return 0.0

    top_k_retrieved = retrieved[:k]
    dcg = 0.0
    for rank, item in enumerate(top_k_retrieved, 1):
        if item in gold_evidence:
            dcg += 1.0 / math.log2(rank + 1)

    # Ideal DCG: perfect ranking with all gold items at the top
    ideal_hits = min(k, len(gold_evidence))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision(
    retrieved: List[Tuple[str, int]],
    gold_evidence: Set[Tuple[str, int]],
    k: Optional[int] = 10
) -> float:
    """Computes Average Precision (AP@K) for a single query:
    AP = sum(P@rank * rel(rank)) / min(len(gold_evidence), k)
    """
    if not gold_evidence or not retrieved:
        return 0.0

    target_list = retrieved[:k] if k is not None else retrieved
    num_relevant = 0
    precision_sum = 0.0

    for rank, item in enumerate(target_list, 1):
        if item in gold_evidence:
            num_relevant += 1
            precision_sum += num_relevant / rank

    normalizer = min(len(gold_evidence), len(target_list))
    return precision_sum / normalizer if normalizer > 0 else 0.0


def evaluate_retrieval_batch(
    retrieved_per_query: List[List[Tuple[str, int]]],
    gold_per_query: List[Set[Tuple[str, int]]],
    latencies: Optional[List[float]] = None,
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Calculates macro-averaged IR metrics (Precision@K, Recall@K, MAP, nDCG@K, MRR)."""
    total_queries = len(retrieved_per_query)
    if total_queries == 0:
        return {}

    metrics: Dict[str, Any] = {}

    for k in k_list:
        p_at_k = [precision_at_k(r, g, k) for r, g in zip(retrieved_per_query, gold_per_query) if g]
        r_at_k = [recall_at_k(r, g, k) for r, g in zip(retrieved_per_query, gold_per_query) if g]
        ndcg_k = [ndcg_at_k(r, g, k) for r, g in zip(retrieved_per_query, gold_per_query) if g]

        metrics[f"precision@{k}"] = round(float(np.mean(p_at_k)), 4) if p_at_k else 0.0
        metrics[f"recall@{k}"] = round(float(np.mean(r_at_k)), 4) if r_at_k else 0.0
        metrics[f"ndcg@{k}"] = round(float(np.mean(ndcg_k)), 4) if ndcg_k else 0.0

    # Mean Average Precision (MAP)
    maps = [average_precision(r, g, k=10) for r, g in zip(retrieved_per_query, gold_per_query) if g]
    metrics["map"] = round(float(np.mean(maps)), 4) if maps else 0.0

    # Mean Reciprocal Rank (MRR)
    mrrs = [reciprocal_rank(r, g) for r, g in zip(retrieved_per_query, gold_per_query) if g]
    metrics["mrr"] = round(float(np.mean(mrrs)), 4) if mrrs else 0.0

    # Precision-Recall Curve points across cutoff depths
    pr_curve_k = [1, 2, 3, 5, 8, 10]
    pr_curve = []
    for k_val in pr_curve_k:
        p = np.mean([precision_at_k(r, g, k_val) for r, g in zip(retrieved_per_query, gold_per_query) if g])
        r = np.mean([recall_at_k(r, g, k_val) for r, g in zip(retrieved_per_query, gold_per_query) if g])
        pr_curve.append({"k": k_val, "precision": round(float(p), 4), "recall": round(float(r), 4)})
    metrics["pr_curve"] = pr_curve

    if latencies:
        metrics["mean_latency_ms"] = round(float(np.mean(latencies)) * 1000, 2)
        metrics["p95_latency_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 2)

    return metrics
