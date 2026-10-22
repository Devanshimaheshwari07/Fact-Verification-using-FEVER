"""Data validation and diagnostic summary utilities for FEVER datasets."""
import collections
from typing import Any, Dict, List
from src.data.load_data import FeverClaim, EvidenceSentence


def validate_claims_dataset(claims: List[FeverClaim]) -> Dict[str, Any]:
    """Inspects and validates a list of FEVER claims, returning distribution metrics.

    Args:
        claims: List of FeverClaim objects.

    Returns:
        Dictionary with statistical distribution and validation flags.
    """
    total = len(claims)
    if total == 0:
        return {
            "total_claims": 0,
            "error": "Dataset is empty"
        }

    class_dist = collections.Counter(c.label for c in claims)
    verifiable_dist = collections.Counter(c.verifiable for c in claims)

    claims_with_evidence = sum(1 for c in claims if len(c.evidence_sets) > 0)
    claims_multi_hop = sum(1 for c in claims if any(len(s) > 1 for s in c.evidence_sets))
    claims_multi_sets = sum(1 for c in claims if len(c.evidence_sets) > 1)

    lengths = [len(c.claim.split()) for c in claims]
    avg_length = sum(lengths) / total if total > 0 else 0.0

    return {
        "total_claims": total,
        "class_distribution": dict(class_dist),
        "verifiable_distribution": dict(verifiable_dist),
        "claims_with_gold_evidence": claims_with_evidence,
        "claims_with_multi_hop_evidence": claims_multi_hop,
        "claims_with_multiple_evidence_sets": claims_multi_sets,
        "average_claim_token_length": round(avg_length, 2),
        "min_claim_tokens": min(lengths) if lengths else 0,
        "max_claim_tokens": max(lengths) if lengths else 0,
    }


def validate_corpus_dataset(sentences: List[EvidenceSentence]) -> Dict[str, Any]:
    """Inspects and validates an indexed Wikipedia corpus collection."""
    total = len(sentences)
    if total == 0:
        return {
            "total_sentences": 0,
            "error": "Corpus is empty"
        }

    unique_docs = len(set(s.doc_id for s in sentences))
    lengths = [len(s.text.split()) for s in sentences]
    avg_length = sum(lengths) / total if total > 0 else 0.0

    return {
        "total_sentences": total,
        "unique_documents": unique_docs,
        "average_sentence_length_tokens": round(avg_length, 2),
        "min_sentence_length": min(lengths) if lengths else 0,
        "max_sentence_length": max(lengths) if lengths else 0,
    }
