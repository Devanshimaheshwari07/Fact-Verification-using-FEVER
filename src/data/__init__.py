"""Data loading, validation, and acquisition package."""
from src.data.load_data import (
    FeverClaim,
    EvidenceSentence,
    EvidenceItem,
    load_fever_claims,
    load_wiki_corpus,
    stream_wiki_sentences,
    normalize_page_id,
    format_page_title
)
from src.data.validate_data import validate_claims_dataset, validate_corpus_dataset
from src.data.download_data import download_official_dataset, create_sample_fever_dataset

__all__ = [
    "FeverClaim",
    "EvidenceSentence",
    "EvidenceItem",
    "load_fever_claims",
    "load_wiki_corpus",
    "stream_wiki_sentences",
    "normalize_page_id",
    "format_page_title",
    "validate_claims_dataset",
    "validate_corpus_dataset",
    "download_official_dataset",
    "create_sample_fever_dataset"
]
