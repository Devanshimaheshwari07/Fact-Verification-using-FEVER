"""Script to prepare and validate FEVER datasets."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logging_config import setup_logger
from src.data.download_data import create_sample_fever_dataset, download_official_dataset
from src.data.load_data import load_fever_claims, load_wiki_corpus
from src.data.validate_data import validate_claims_dataset, validate_corpus_dataset

logger = setup_logger("prepare_data")


def main():
    parser = argparse.ArgumentParser(description="Acquire and validate FEVER datasets.")
    parser.add_argument("--mode", choices=["sample", "official"], default="sample",
                        help="Prepare 'sample' dataset for local testing or 'official' FEVER benchmark.")
    parser.add_argument("--split", choices=["train", "dev", "test", "wiki_dump"], default="dev",
                        help="Official split to download (if mode is official).")
    parser.add_argument("--sample_dir", default="data/sample", help="Directory for sample files.")
    parser.add_argument("--raw_dir", default="data/raw", help="Directory for official raw files.")

    args = parser.parse_args()

    if args.mode == "sample":
        logger.info(f"Generating authentic FEVER sample dataset in '{args.sample_dir}'...")
        paths = create_sample_fever_dataset(sample_dir=args.sample_dir)
        logger.info(f"Files created: {paths}")

        # Validate
        train_claims = load_fever_claims(paths["train"])
        dev_claims = load_fever_claims(paths["dev"])
        wiki_sentences = load_wiki_corpus(paths["wiki_sample"])

        train_stats = validate_claims_dataset(train_claims)
        dev_stats = validate_claims_dataset(dev_claims)
        corpus_stats = validate_corpus_dataset(wiki_sentences)

        logger.info("=== Training Set Validation ===")
        logger.info(train_stats)

        logger.info("=== Development Set Validation ===")
        logger.info(dev_stats)

        logger.info("=== Wikipedia Corpus Validation ===")
        logger.info(corpus_stats)

    elif args.mode == "official":
        logger.info(f"Downloading official FEVER split '{args.split}' to '{args.raw_dir}'...")
        file_path = download_official_dataset(split=args.split, target_dir=args.raw_dir)
        logger.info(f"Saved to {file_path}")


if __name__ == "__main__":
    main()
