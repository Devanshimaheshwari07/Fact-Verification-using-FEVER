"""Script to construct, serialize, and test the Wikipedia knowledge base inverted index."""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logger
from src.utils.config_loader import get_storage_dir
from src.indexing.build_index import IndexManager
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = setup_logger("build_index")


def resolve_corpus_paths(user_path: str = None) -> list:
    """Resolves available Wikipedia corpus paths, ensuring both standard benchmark articles and official FEVER dump partitions are included."""
    if user_path:
        p = Path(user_path)
        if p.exists():
            return [p]

    paths = []
    sample_path = Path("data/sample/wiki_sample.jsonl")
    if sample_path.exists():
        paths.append(sample_path)

    raw_dir = get_storage_dir("data/raw")
    wiki_dir = raw_dir / "wiki-pages"
    if wiki_dir.exists():
        dev_bench = wiki_dir / "dev_benchmark_pages.jsonl"
        if dev_bench.exists():
            paths.append(dev_bench)
        w1 = wiki_dir / "wiki-001.jsonl"
        if w1.exists():
            paths.append(w1)
        elif any(wiki_dir.glob("*.jsonl")):
            paths.append(wiki_dir)

    return paths if paths else [sample_path]


def main():
    parser = argparse.ArgumentParser(description="Build and serialize knowledge base index.")
    parser.add_argument("--corpus", default=None,
                        help="Path to Wikipedia JSONL corpus file or directory.")
    parser.add_argument("--output_dir", default="artifacts/indexes",
                        help="Output directory for serialized index artifacts.")
    parser.add_argument("--max_pages", type=int, default=5000,
                        help="Optional limit on number of pages to index (default: 5000).")

    args = parser.parse_args()

    corpus_paths = resolve_corpus_paths(args.corpus)
    logger.info(f"Using Wikipedia corpus sources: {corpus_paths}")

    preprocessor = TextPreprocessor(lowercase=True, remove_stopwords=False, use_stemming=True)
    manager = IndexManager(index_dir=args.output_dir, preprocessor=preprocessor)

    stats = manager.build_from_corpus(
        corpus_path=corpus_paths,
        max_pages=args.max_pages,
        save_artifacts=True
    )

    logger.info("=== Index Construction Completed Successfully ===")
    for k, v in stats.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
