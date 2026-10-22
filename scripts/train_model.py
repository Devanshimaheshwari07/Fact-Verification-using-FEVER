"""Script to train and evaluate the evidence-grounded claim verification classifier on official FEVER data."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logger
from src.utils.config_loader import get_storage_dir
from src.data.load_data import load_fever_claims
from src.indexing.build_index import IndexManager
from src.retrieval.bm25_retriever import BM25Retriever
from src.verification.train_classifier import VerificationModelTrainer

logger = setup_logger("train_model")


def resolve_train_file(user_path: str = None) -> Path:
    """Resolves the best available FEVER training claims file, prioritizing official data."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return p
    raw_dir = get_storage_dir("data/raw")
    official_train = raw_dir / "train.jsonl"
    if official_train.exists():
        return official_train
    sample = Path("data/sample/train.jsonl")
    if sample.exists():
        return sample
    return official_train


def main():
    parser = argparse.ArgumentParser(description="Train fact verification classifier on official FEVER dataset.")
    parser.add_argument("--train_file", default=None,
                        help="Path to training claims JSONL.")
    parser.add_argument("--index_dir", default="artifacts/indexes",
                        help="Path to indexed knowledge base artifacts.")
    parser.add_argument("--model_dir", default="artifacts/models",
                        help="Output directory to save trained model artifact.")
    parser.add_argument("--classifier", choices=["logistic_regression", "random_forest"],
                        default="logistic_regression", help="Classifier architecture.")
    parser.add_argument("--max_claims", type=int, default=2000,
                        help="Maximum training claims to condition classifier on (default: 2000).")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of retrieved sentences to condition features on.")

    args = parser.parse_args()

    # Load index manager
    manager = IndexManager(index_dir=args.index_dir)
    manager.load()

    # Initialize BM25 retriever for feature extraction during training
    retriever = BM25Retriever(index=manager.inverted_index)

    # Load claims
    train_file = resolve_train_file(args.train_file)
    logger.info(f"Loading official FEVER training claims from: {train_file}")
    train_claims = load_fever_claims(str(train_file), max_records=args.max_claims)
    logger.info(f"Loaded {len(train_claims)} training claims for model training.")

    # Train classifier
    trainer = VerificationModelTrainer(
        classifier_type=args.classifier,
        model_dir=args.model_dir,
        random_state=42
    )
    results = trainer.train(train_claims, retriever=retriever, top_k=args.top_k)

    logger.info("=== Model Training Completed Successfully ===")
    for k, v in results.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
