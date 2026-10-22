"""Feature extraction for claim-evidence verification classification."""
import re
from typing import List, Dict, Any, Optional
import numpy as np

from src.retrieval.base_retriever import SearchResult
from src.preprocessing.text_preprocessor import TextPreprocessor, NEGATION_TOKENS


class VerificationFeatureExtractor:
    """Extracts interpretable IR, lexical, syntactic, and numeric features from (claim, evidence) pairs."""

    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        self.preprocessor = preprocessor or TextPreprocessor()

        self.feature_names = [
            "max_retrieval_score",
            "mean_retrieval_score",
            "score_margin",
            "max_jaccard_similarity",
            "max_claim_containment",
            "claim_has_negation",
            "evidence_has_negation",
            "negation_mismatch",
            "has_claim_numbers",
            "numeric_overlap_ratio",
            "numeric_mismatch",
            "entity_overlap_ratio",
            "predicate_containment",
            "evidence_count",
            "token_length_ratio"
        ]

    def extract_features(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> np.ndarray:
        """Extracts fixed-length feature vector for claim and retrieved evidence."""
        claim_tokens = set(self.preprocessor.preprocess(claim_text))
        claim_raw_words = set(re.findall(r"\b[a-z']+\b", claim_text.lower()))

        # Claim diagnostics
        claim_entities_info = self.preprocessor.extract_entities_and_keywords(claim_text)
        claim_numbers = set(claim_entities_info["numbers"])
        claim_has_neg = float(any(w in claim_raw_words for w in NEGATION_TOKENS))

        if not retrieved_evidence:
            # If no evidence was retrieved, return default zeroed features
            vec = np.zeros(len(self.feature_names), dtype=np.float32)
            vec[self.feature_names.index("claim_has_negation")] = claim_has_neg
            if claim_numbers:
                vec[self.feature_names.index("has_claim_numbers")] = 1.0
                vec[self.feature_names.index("numeric_mismatch")] = 1.0
            return vec

        scores = [res.score for res in retrieved_evidence]
        raw_max = float(max(scores)) if scores else 0.0
        max_score = float(np.log1p(max(raw_max, 0.0)))
        raw_mean = float(np.mean(scores)) if scores else 0.0
        mean_score = float(np.log1p(max(raw_mean, 0.0)))
        score_margin = float(np.log1p(max(scores[0] - scores[1], 0.0))) if len(scores) > 1 else max_score

        # Combine text of all top evidence
        combined_evidence_text = " ".join(res.text for res in retrieved_evidence).lower()
        combined_evidence_words = set(re.findall(r"\b[a-z']+\b", combined_evidence_text))

        # Check evidence negation on the top relevant sentence for the claim's attribute
        components = self.preprocessor.extract_claim_components(claim_text)
        attr_tokens = components.get("attribute_tokens", set())

        best_ev_text = ""
        best_overlap = -1
        for res in retrieved_evidence:
            ev_toks = set(self.preprocessor.preprocess(res.text))
            hits = len(ev_toks.intersection(attr_tokens)) if attr_tokens else len(ev_toks.intersection(claim_tokens))
            if hits > best_overlap:
                best_overlap = hits
                best_ev_text = res.text.lower()

        best_ev_words = set(re.findall(r"\b[a-z']+\b", best_ev_text))
        evidence_has_neg = float(any(w in best_ev_words for w in NEGATION_TOKENS))

        # Negation mismatch is only meaningful if the best matching sentence for the claim has opposite polarity
        if claim_has_neg and not evidence_has_neg and best_overlap > 0:
            negation_mismatch = 1.0
        elif not claim_has_neg and evidence_has_neg and best_overlap > 0:
            has_targeted_neg = any(
                f"{neg} {aw}" in best_ev_text or f"{neg} a {aw}" in best_ev_text
                for aw in components.get("attribute_text", "").split()
                for neg in ["not", "never", "no", "neither", "refused", "failed"]
            )
            negation_mismatch = 1.0 if has_targeted_neg else 0.0
        else:
            negation_mismatch = 0.0

        # Jaccard and containment
        max_jaccard = 0.0
        max_containment = 0.0
        for res in retrieved_evidence:
            ev_tokens = set(self.preprocessor.preprocess(res.text))
            intersection = claim_tokens.intersection(ev_tokens)
            union = claim_tokens.union(ev_tokens)

            jaccard = len(intersection) / len(union) if union else 0.0
            containment = len(intersection) / len(claim_tokens) if claim_tokens else 0.0

            if jaccard > max_jaccard:
                max_jaccard = jaccard
            if containment > max_containment:
                max_containment = containment

        # Number matching
        evidence_numbers = set(re.findall(r"\b\d+(?:[\.,]\d+)?\b", combined_evidence_text))
        has_claim_nums = 1.0 if claim_numbers else 0.0
        if claim_numbers:
            matching_nums = claim_numbers.intersection(evidence_numbers)
            numeric_overlap = len(matching_nums) / len(claim_numbers)
            numeric_mismatch = 1.0 if len(matching_nums) == 0 else 0.0
        else:
            numeric_overlap = 1.0
            numeric_mismatch = 0.0

        # Entity overlap
        entities = claim_entities_info["entities"]
        if entities:
            matched_entities = sum(
                1 for e in entities if (e.lower() in combined_evidence_text or any(sub.lower() in combined_evidence_text for sub in re.findall(r"\b[A-Za-z0-9]+\b", e) if len(sub) > 2))
            )
            entity_overlap = matched_entities / len(entities)
        else:
            entity_overlap = 1.0

        # Predicate / assertion containment: measure overlap of key claim attributes (excluding entities and stopwords)
        components = self.preprocessor.extract_claim_components(claim_text)
        attr_tokens = components.get("attribute_tokens", set())

        if attr_tokens:
            attr_matched = sum(
                1 for at in attr_tokens
                if any(at in self.preprocessor.preprocess(res.text) for res in retrieved_evidence)
            )
            predicate_containment = attr_matched / len(attr_tokens)
        else:
            entity_tokens = set()
            for e in entities:
                entity_tokens.update(self.preprocessor.preprocess(e))
                for sub in re.findall(r"\b[a-zA-Z0-9]+\b", e.lower()):
                    entity_tokens.add(self.preprocessor.stem(sub) if self.preprocessor.use_stemming else sub)
            predicate_tokens = claim_tokens - entity_tokens
            if predicate_tokens:
                pred_matched = sum(
                    1 for pt in predicate_tokens
                    if any(pt in self.preprocessor.preprocess(res.text) for res in retrieved_evidence)
                )
                predicate_containment = pred_matched / len(predicate_tokens)
            else:
                predicate_containment = max_containment

        # Evidence count and length ratio
        ev_count = float(len(retrieved_evidence))
        top_ev_len = len(retrieved_evidence[0].text.split()) if retrieved_evidence else 1
        claim_len = max(len(claim_text.split()), 1)
        length_ratio = min(claim_len / max(top_ev_len, 1), 3.0)

        feature_dict = {
            "max_retrieval_score": max_score,
            "mean_retrieval_score": mean_score,
            "score_margin": score_margin,
            "max_jaccard_similarity": max_jaccard,
            "max_claim_containment": max_containment,
            "claim_has_negation": claim_has_neg,
            "evidence_has_negation": evidence_has_neg,
            "negation_mismatch": negation_mismatch,
            "has_claim_numbers": has_claim_nums,
            "numeric_overlap_ratio": numeric_overlap,
            "numeric_mismatch": numeric_mismatch,
            "entity_overlap_ratio": entity_overlap,
            "predicate_containment": predicate_containment,
            "evidence_count": ev_count,
            "token_length_ratio": length_ratio
        }

        return np.array([feature_dict[name] for name in self.feature_names], dtype=np.float32)
