"""Comprehensive evaluation runner for Information Retrieval and Fact Verification benchmarks."""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logger
from src.data.load_data import load_fever_claims, FeverClaim
from src.indexing.build_index import IndexManager
from src.pipeline.fact_verifier import FactVerifierPipeline
from src.evaluation.retrieval_metrics import evaluate_retrieval_batch
from src.evaluation.fever_evaluation import compute_fever_scores
from src.evaluation.verification_metrics import evaluate_classifier_performance
from src.evaluation.error_analysis import perform_error_analysis
from src.utils.database import ExperimentDatabase

logger = setup_logger("run_evaluation")


def evaluate_retrievers(
    claims: List[FeverClaim],
    pipeline: FactVerifierPipeline,
    k_list: List[int] = [1, 3, 5, 10]
) -> Dict[str, Any]:
    """Runs comparative evaluation across all available retrieval algorithms."""
    results: Dict[str, Any] = {}
    # The 4 core ranking models from the system architecture + boolean baseline
    methods = ["tfidf", "bm25", "semantic", "hybrid"]

    # Filter to claims with gold evidence for IR evaluation
    verifiable_claims = [c for c in claims if c.label != "NOT ENOUGH INFO" and len(c.evidence_sets) > 0]
    logger.info(f"Evaluating retrieval methods on {len(verifiable_claims)} verifiable claims...")

    for method in methods:
        retriever = pipeline.get_retriever(method)
        retrieved_lists: List[List[Tuple[str, int]]] = []
        gold_sets: List[set] = []
        latencies: List[float] = []

        for c in verifiable_claims:
            start_t = time.time()
            res = retriever.retrieve(c.claim, top_k=max(k_list))
            duration = time.time() - start_t

            retrieved_tuples = [(r.doc_id, r.sentence_id) for r in res]
            retrieved_lists.append(retrieved_tuples)
            gold_sets.append(c.get_flat_gold_evidence())
            latencies.append(duration)

        metrics = evaluate_retrieval_batch(
            retrieved_per_query=retrieved_lists,
            gold_per_query=gold_sets,
            latencies=latencies,
            k_list=k_list
        )
        results[method] = metrics
        logger.info(f"Method '{method}': MRR={metrics.get('mrr', 0):.4f} | R@5={metrics.get('recall@5', 0):.4f} | Latency={metrics.get('mean_latency_ms', 0):.2f}ms")

    return results


def evaluate_verification_pipeline(
    claims: List[FeverClaim],
    pipeline: FactVerifierPipeline,
    retrieval_method: str = "bm25",
    top_k: int = 5
) -> Dict[str, Any]:
    """Evaluates end-to-end fact verification accuracy, FEVER score, and error taxonomy."""
    logger.info(f"Evaluating end-to-end verification pipeline (method={retrieval_method}, top_k={top_k})...")

    gold_labels = [c.label for c in claims]
    pred_labels: List[str] = []
    retrieved_evidence_tuples: List[List[Tuple[str, int]]] = []

    for c in claims:
        out = pipeline.verify_claim(
            c.claim,
            retrieval_method=retrieval_method,
            top_k=top_k,
            log_to_db=False,
            enable_live_wiki=False
        )
        pred_labels.append(out.predicted_label)
        retrieved_evidence_tuples.append([(e.doc_id, e.sentence_id) for e in out.evidence])

    # 1. Classification metrics
    clf_metrics = evaluate_classifier_performance(gold_labels, pred_labels)

    # 2. Official FEVER score
    fever_metrics = compute_fever_scores(claims, pred_labels, retrieved_evidence_tuples)

    # 3. Error analysis
    err_analysis = perform_error_analysis(claims, pred_labels, retrieved_evidence_tuples)

    return {
        "classification": clf_metrics,
        "fever_benchmark": fever_metrics,
        "error_analysis": err_analysis
    }


def resolve_dev_file(user_path: str = None) -> Path:
    """Resolves the best available FEVER development claims file, prioritizing official data."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    from src.utils.config_loader import get_storage_dir
    raw_dir = get_storage_dir("data/raw")
    official_dev = raw_dir / "shared_task_dev.jsonl"
    if official_dev.exists():
        return official_dev
    sample = Path("data/sample/dev.jsonl")
    if sample.exists():
        return sample
    return official_dev


def main():
    parser = argparse.ArgumentParser(description="Run complete evaluation benchmark suite on official FEVER data.")
    parser.add_argument("--dev_file", default=None,
                        help="Path to development claims JSONL.")
    parser.add_argument("--index_dir", default="artifacts/indexes",
                        help="Path to index artifacts.")
    parser.add_argument("--results_dir", default="artifacts/results",
                        help="Output directory for results JSON.")
    parser.add_argument("--max_claims", type=int, default=500,
                        help="Maximum claims to evaluate (default: 500).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k evidence sentences.")

    args = parser.parse_args()

    from src.utils.config_loader import get_storage_dir
    results_path = get_storage_dir(args.results_dir, for_writing=True)
    results_path.mkdir(parents=True, exist_ok=True)

    # Load claims
    dev_file = resolve_dev_file(args.dev_file)
    logger.info(f"Loading official FEVER development claims from: {dev_file}")
    dev_claims = load_fever_claims(str(dev_file), max_records=args.max_claims)
    logger.info(f"Loaded {len(dev_claims)} claims for evaluation from {dev_file}.")

    # Initialize pipeline
    pipeline = FactVerifierPipeline(
        index_manager=IndexManager(index_dir=args.index_dir)
    )

    # 1. Comparative IR Evaluation
    ir_comparison = evaluate_retrievers(dev_claims, pipeline)

    # 2. End-to-end Verification Evaluation with BM25
    e2e_bm25 = evaluate_verification_pipeline(dev_claims, pipeline, retrieval_method="bm25", top_k=args.top_k)

    # 3. End-to-end Verification Evaluation with Hybrid
    e2e_hybrid = evaluate_verification_pipeline(dev_claims, pipeline, retrieval_method="hybrid", top_k=args.top_k)

    all_results = {
        "dataset_split": "dev",
        "num_claims": len(dev_claims),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ir_comparison": ir_comparison,
        "verification_bm25": e2e_bm25,
        "verification_hybrid": e2e_hybrid
    }

    # Save to JSON
    json_out = results_path / "evaluation_results.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Saved evaluation results to {json_out}")

    # Log to SQLite DB
    db = ExperimentDatabase()
    db.log_evaluation_run(
        run_name="BM25_Dev_Benchmark",
        dataset_split="dev",
        retrieval_method="bm25",
        classifier_type="logistic_regression",
        num_samples=len(dev_claims),
        metrics=e2e_bm25
    )
    db.log_evaluation_run(
        run_name="Hybrid_Dev_Benchmark",
        dataset_split="dev",
        retrieval_method="hybrid",
        classifier_type="logistic_regression",
        num_samples=len(dev_claims),
        metrics=e2e_hybrid
    )

    logger.info("=== Overall FEVER Evaluation Summary (BM25) ===")
    logger.info(f"  FEVER Score:      {e2e_bm25['fever_benchmark']['fever_score']:.4f}")
    logger.info(f"  Label Accuracy:   {e2e_bm25['fever_benchmark']['label_accuracy']:.4f}")
    logger.info(f"  Evidence Recall:  {e2e_bm25['fever_benchmark']['evidence_recall']:.4f}")
    logger.info(f"  Macro F1:         {e2e_bm25['classification']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
