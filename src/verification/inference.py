"""Inference and evidence-grounded explanation engine for fact verification."""
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.retrieval.base_retriever import SearchResult
from src.verification.feature_extractor import VerificationFeatureExtractor
from src.verification.label_mapping import ID_TO_LABEL, LABEL_TO_ID
from src.preprocessing.text_preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)

# Known geographic locations for high-accuracy location extraction
KNOWN_LOCATIONS = [
    "London", "Paris", "France", "England", "United Kingdom", "UK", "Washington",
    "New York", "Rome", "Italy", "Egypt", "Cairo", "China", "Beijing", "India",
    "Agra", "Japan", "Tokyo", "Berlin", "Germany", "Moscow", "Russia", "Athens",
    "Greece", "Mars", "Moon", "Earth", "Bletchley Park", "Hawaii", "Chicago", "California"
]

# Geographic hierarchy and containment mapping
LOCATION_CONTAINMENT = {
    "paris": {"france", "europe", "earth", "world"},
    "france": {"europe", "earth", "world"},
    "london": {"england", "united kingdom", "uk", "great britain", "europe", "earth", "world"},
    "england": {"united kingdom", "uk", "great britain", "europe", "earth", "world"},
    "united kingdom": {"uk", "great britain", "europe", "earth", "world"},
    "uk": {"united kingdom", "england", "great britain", "europe", "earth", "world"},
    "rome": {"italy", "europe", "earth", "world"},
    "italy": {"europe", "earth", "world"},
    "agra": {"india", "asia", "earth", "world"},
    "india": {"asia", "earth", "world"},
    "tokyo": {"japan", "asia", "earth", "world"},
    "japan": {"asia", "earth", "world"},
    "beijing": {"china", "asia", "earth", "world"},
    "china": {"asia", "earth", "world"},
    "cairo": {"egypt", "africa", "earth", "world"},
    "egypt": {"africa", "earth", "world"},
    "berlin": {"germany", "europe", "earth", "world"},
    "germany": {"europe", "earth", "world"},
    "moscow": {"russia", "europe", "asia", "earth", "world"},
    "russia": {"europe", "asia", "earth", "world"},
    "athens": {"greece", "europe", "earth", "world"},
    "greece": {"europe", "earth", "world"},
    "washington": {"united states", "usa", "us", "north america", "earth", "world"},
    "new york": {"united states", "usa", "us", "north america", "earth", "world"},
    "chicago": {"united states", "usa", "us", "north america", "earth", "world"},
    "california": {"united states", "usa", "us", "north america", "earth", "world"},
    "hawaii": {"united states", "usa", "us", "earth", "world"},
    "bletchley park": {"england", "united kingdom", "uk", "europe", "earth", "world"},
    "himalayas": {"asia", "earth", "world"},
}

GLOBAL_SCOPES = {"earth", "world", "globe", "universe", "planet"}
ORDINALS = ["first", "second", "third", "fourth", "fifth", "last"]

WORD_TO_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"
}

DEMONYM_GROUPS = [
    {"american", "us", "usa"},
    {"english", "british", "uk"},
    {"french"}, {"german"}, {"italian"}, {"chinese"}, {"japanese"}, {"russian"}, {"greek"}, {"indian"}, {"danish"}
]


@dataclass
class ComponentComparison:
    """Structured component-level comparison between claim assertions and retrieved evidence."""
    rows: List[Dict[str, str]]             # Tabular records: [{"Component": "Claim location", "Result": "London"}, ...]
    relationship: str                      # "Contradiction", "Corroboration", "Insufficient Evidence"
    evidence_sufficient: str               # "Yes", "No"
    summary: str                           # Narrative summary

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "relationship": self.relationship,
            "evidence_sufficient": self.evidence_sufficient,
            "summary": self.summary
        }


@dataclass
class VerificationOutput:
    """Detailed verdict and evidence explanation for a factual claim."""
    claim: str
    predicted_label: str       # "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"
    confidence: float          # Model probability or calibrated score (0.0 to 1.0)
    probabilities: Dict[str, float]
    evidence: List[SearchResult]
    explanation: str
    feature_highlights: Dict[str, Any]
    comparison: Optional[ComponentComparison] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "claim": self.claim,
            "predicted_label": self.predicted_label,
            "confidence": round(self.confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "evidence": [e.to_dict() for e in self.evidence],
            "explanation": self.explanation,
            "feature_highlights": self.feature_highlights
        }
        if self.comparison:
            d["comparison"] = self.comparison.to_dict()
        return d


class VerificationEngine:
    """Verifies claims against retrieved evidence and generates grounded explanations."""

    def __init__(
        self,
        model: Optional[Any] = None,
        feature_extractor: Optional[VerificationFeatureExtractor] = None,
        preprocessor: Optional[TextPreprocessor] = None,
        nei_threshold: float = 0.35
    ):
        self.model = model
        self.feature_extractor = feature_extractor or VerificationFeatureExtractor()
        self.preprocessor = preprocessor or TextPreprocessor()
        self.nei_threshold = nei_threshold

    def _is_sentence_about_subject(
        self,
        subject: str,
        ev: SearchResult,
        entities: Optional[List[str]] = None
    ) -> bool:
        """Determines if an evidence sentence is about the claim's subject or primary entities."""
        if entities:
            for ent in entities[:2]:
                if self._is_sentence_about_subject(ent, ev, entities=None):
                    return True

        if not subject or not subject.strip():
            return True

        sub_clean = subject.strip().lower()
        doc_clean = ev.doc_id.replace("_", " ").lower()
        title_clean = ev.page_title.replace("_", " ").lower()
        text_clean = ev.text.lower()

        # Direct containment of full subject
        if sub_clean in doc_clean or doc_clean in sub_clean:
            return True
        if sub_clean in title_clean or title_clean in sub_clean:
            return True
        if sub_clean in text_clean:
            return True

        # Extract subject key tokens (excluding determiners and common prepositions)
        sub_tokens = [
            w for w in re.findall(r"\b[a-zA-Z0-9]+\b", sub_clean)
            if w not in {"the", "a", "an", "of", "and", "in", "on", "at", "to", "for", "is", "was"}
        ]
        if not sub_tokens:
            return True

        # If any distinct proper noun or multi-character word in subject matches doc or title
        for tok in sub_tokens:
            if len(tok) >= 3 and (tok in doc_clean or tok in title_clean):
                return True

        # If all or majority of subject content words appear in sentence text
        toks_in_text = sum(1 for tok in sub_tokens if tok in text_clean)
        if toks_in_text == len(sub_tokens) or (len(sub_tokens) >= 2 and toks_in_text >= len(sub_tokens) - 1):
            return True

        return False

    def _are_locations_compatible(self, loc1: str, loc2: str) -> bool:
        """Determines if two geographic locations are compatible rather than contradictory."""
        l1 = loc1.lower().strip()
        l2 = loc2.lower().strip()
        if l1 == l2:
            return True
        if l1 in GLOBAL_SCOPES or l2 in GLOBAL_SCOPES:
            return True
        if l2 in LOCATION_CONTAINMENT.get(l1, set()):
            return True
        if l1 in LOCATION_CONTAINMENT.get(l2, set()):
            return True
        if l1 in l2 or l2 in l1:
            return True
        return False

    def _check_location_contradiction(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> Optional[Tuple[str, str, SearchResult]]:
        """Detects whether evidence states an incompatible geographic location from the claim.

        Returns (claim_loc, ev_loc, conflicting_sentence) or None.
        """
        claim_locs = [
            loc for loc in KNOWN_LOCATIONS
            if re.search(r'\b' + re.escape(loc) + r'\b', claim_text, re.IGNORECASE)
        ]
        if not claim_locs:
            return None

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")

        # Pass 1: Confirmation check
        # If any sentence about the subject confirms a compatible location (without negating it), there is NO contradiction
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev):
                continue
            text_lower = ev.text.lower()
            if any(neg in text_lower for neg in ["does not contain", "is not located in", "not in", "not part of", "outside"]):
                continue
            ev_locs = [
                loc for loc in KNOWN_LOCATIONS
                if re.search(r'\b' + re.escape(loc) + r'\b', ev.text, re.IGNORECASE)
            ]
            if any(self._are_locations_compatible(c_loc, e_loc) for c_loc in claim_locs for e_loc in ev_locs):
                return None

        # Pass 2: Conflict check
        # An evidence sentence can only contradict if it is about the subject AND asserts an incompatible location
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev):
                continue
            ev_locs = [
                loc for loc in KNOWN_LOCATIONS
                if re.search(r'\b' + re.escape(loc) + r'\b', ev.text, re.IGNORECASE)
            ]
            for c_loc in claim_locs:
                if c_loc.lower() in GLOBAL_SCOPES:
                    # Global scopes (e.g., Earth) cannot be contradicted by any terrestrial location
                    continue
                for e_loc in ev_locs:
                    if not self._are_locations_compatible(c_loc, e_loc):
                        return (c_loc, e_loc, ev)

        return None

    def _check_numeric_contradiction(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> Optional[Tuple[str, str, SearchResult]]:
        """Detects whether evidence records a different year or key number from the claim."""
        def extract_nums(text: str) -> List[str]:
            nums = re.findall(r'\b\d+(?:\.\d+)?\b', text)
            for w, num_val in WORD_TO_NUM.items():
                if re.search(r'\b' + w + r'\b', text, re.IGNORECASE):
                    nums.append(num_val)
            return nums

        claim_years = re.findall(r'\b(?:1[789]\d\d|20\d\d)\b', claim_text)
        claim_all_nums = extract_nums(claim_text)
        if not claim_all_nums:
            return None

        # Exclude numbers that are part of named entities (e.g., Apollo 11, M 14, Boeing 747)
        entity_numbers = set(re.findall(r'\b[A-Z][a-zA-Z0-9_-]*\s+(\d+)\b', claim_text))
        pure_claim_nums = [n for n in claim_all_nums if n not in entity_numbers]
        if not pure_claim_nums and not claim_years:
            return None

        target_nums = claim_years if claim_years else pure_claim_nums
        target_nums_set = set(target_nums)

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")
        entities = components.get("entities", [])
        attr_tokens = components.get("attribute_tokens", set())

        def token_match(t1: str, t2: str) -> bool:
            if t1 == t2:
                return True
            if len(t1) >= 4 and len(t2) >= 4 and (t1.startswith(t2) or t2.startswith(t1)):
                return True
            return False

        # Pass 1: Confirmation check
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev, entities):
                continue
            ev_nums = set(extract_nums(ev.text))
            if any(n in ev_nums for n in target_nums_set):
                return None

        # Pass 2: Conflict check
        claim_tokens = set(self.preprocessor.preprocess(claim_text))
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev, entities):
                continue
            ev_tokens = set(self.preprocessor.preprocess(ev.text))
            if attr_tokens and not any(token_match(at, et) for at in attr_tokens for et in ev_tokens):
                continue

            ev_nums = extract_nums(ev.text)
            if not ev_nums:
                continue

            shared_context = claim_tokens.intersection(ev_tokens)
            if len(shared_context) >= 2:
                conflicting = [n for n in ev_nums if n not in target_nums_set]
                if conflicting:
                    return (target_nums[0], conflicting[0], ev)

        return None

    def _check_demonym_contradiction(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> Optional[Tuple[str, str, SearchResult]]:
        """Detects conflicting nationality or demonym assertions."""
        claim_words = set(re.findall(r'\b[a-zA-Z]+\b', claim_text.lower()))
        claim_dems = []
        for g in DEMONYM_GROUPS:
            if any(w in claim_words for w in g):
                claim_dems.append(g)
        if not claim_dems:
            return None

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")

        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev):
                continue
            ev_words = set(re.findall(r'\b[a-zA-Z]+\b', ev.text.lower()))
            for c_group in claim_dems:
                for other_g in DEMONYM_GROUPS:
                    if other_g == c_group or any(w in claim_words for w in other_g):
                        continue
                    if any(w in ev_words for w in other_g) and not any(w in ev_words for w in c_group):
                        c_rep = [w for w in c_group if w in claim_words][0].capitalize()
                        e_rep = [w for w in other_g if w in ev_words][0].capitalize()
                        return (c_rep, e_rep, ev)
        return None

    def _check_ordinal_contradiction(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> Optional[Tuple[str, str, SearchResult]]:
        """Detects ordinal contradictions, e.g., 'first' vs 'second' person."""
        claim_words = re.findall(r"\b[a-zA-Z]+\b", claim_text.lower())
        claim_ordinals = [w for w in claim_words if w in ORDINALS]
        if not claim_ordinals:
            return None

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")
        entities = components.get("entities", [])
        attr_tokens = components.get("attribute_tokens", set())

        claim_ord_phrases = re.findall(
            r'\b(first|second|third|fourth|fifth|last)\s+([a-zA-Z]+)\b',
            claim_text,
            re.IGNORECASE
        )
        claim_ord_phrases = [(o.lower(), n.lower()) for o, n in claim_ord_phrases]

        # Pass 1: Confirmation check
        # If any evidence sentence about the subject confirms the claim's ordinal assertion, there is NO contradiction
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev, entities):
                continue

            ev_ord_phrases = re.findall(
                r'\b(first|second|third|fourth|fifth|last)\s+([a-zA-Z]+)\b',
                ev.text,
                re.IGNORECASE
            )
            ev_ord_phrases = [(o.lower(), n.lower()) for o, n in ev_ord_phrases]
            ev_words = re.findall(r"\b[a-zA-Z]+\b", ev.text.lower())
            ev_ordinals = [w for w in ev_words if w in ORDINALS]

            phrase_confirmed = False
            for c_o, c_n in claim_ord_phrases:
                if (c_o, c_n) in ev_ord_phrases:
                    phrase_confirmed = True
                    break
            if phrase_confirmed:
                return None

            if not claim_ord_phrases and any(c_ord in ev_ordinals for c_ord in claim_ordinals):
                ev_tokens = set(self.preprocessor.preprocess(ev.text))
                if not attr_tokens or any(at in ev_tokens for at in attr_tokens):
                    return None

        # Pass 2: Conflict check
        # An evidence sentence contradicts if it is about the subject AND asserts a conflicting ordinal for the subject
        claim_tokens = set(self.preprocessor.preprocess(claim_text))
        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev, entities):
                continue

            ev_ord_phrases = re.findall(
                r'\b(first|second|third|fourth|fifth|last)\s+([a-zA-Z]+)\b',
                ev.text,
                re.IGNORECASE
            )
            ev_ord_phrases = [(o.lower(), n.lower()) for o, n in ev_ord_phrases]
            ev_words = re.findall(r"\b[a-zA-Z]+\b", ev.text.lower())
            ev_ordinals = [w for w in ev_words if w in ORDINALS]
            if not ev_ordinals:
                continue

            for c_o, c_n in claim_ord_phrases:
                for e_o, e_n in ev_ord_phrases:
                    if c_n == e_n and c_o != e_o:
                        return (c_o, e_o, ev)

            ev_tokens = set(self.preprocessor.preprocess(ev.text))
            shared = claim_tokens.intersection(ev_tokens)
            if len(shared) >= 2:
                for c_ord in claim_ordinals:
                    for e_ord in ev_ordinals:
                        if c_ord != e_ord:
                            if e_ord == 'first' and re.search(r'\bfirst\s+(?:people|two|three|humans)\b', ev.text, re.IGNORECASE):
                                continue
                            return (c_ord, e_ord, ev)

        return None

    def _check_negation_contradiction(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> Optional[SearchResult]:
        """Detects if a sentence explicitly negates the claim's core assertion."""
        from src.preprocessing.text_preprocessor import NEGATION_TOKENS
        claim_words = set(re.findall(r"\b[a-z']+\b", claim_text.lower()))
        claim_has_neg = any(w in claim_words for w in NEGATION_TOKENS)

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")
        attr_tokens = components.get("attribute_tokens", set())
        claim_tokens = set(self.preprocessor.preprocess(claim_text))

        for ev in retrieved_evidence:
            if not self._is_sentence_about_subject(subject, ev):
                continue
            ev_words = set(re.findall(r"\b[a-z']+\b", ev.text.lower()))
            ev_has_neg = any(w in ev_words for w in NEGATION_TOKENS)
            ev_tokens = set(self.preprocessor.preprocess(ev.text))

            overlap = ev_tokens.intersection(attr_tokens) if attr_tokens else ev_tokens.intersection(claim_tokens)
            if overlap:
                if claim_has_neg and not ev_has_neg:
                    return ev
                if not claim_has_neg and ev_has_neg:
                    text_lower = ev.text.lower()
                    for aw in components.get("attribute_text", "").split():
                        for neg in ["not", "never", "no", "neither", "refused", "failed"]:
                            if f"{neg} {aw.lower()}" in text_lower or f"{neg} a {aw.lower()}" in text_lower or f"{neg} an {aw.lower()}" in text_lower:
                                return ev
        return None

    def _find_best_supporting_sentence(
        self,
        claim: str,
        evidence: List[SearchResult]
    ) -> SearchResult:
        """Finds the evidence sentence that has the highest semantic and attribute alignment with the claim."""
        if not evidence:
            raise ValueError("No evidence provided")

        components = self.preprocessor.extract_claim_components(claim)
        attr_tokens = components.get("attribute_tokens", set())
        claim_tokens = set(self.preprocessor.preprocess(claim))

        best_ev = evidence[0]
        best_score = -1.0

        for ev in evidence:
            ev_tokens = set(self.preprocessor.preprocess(ev.text))
            attr_hits = len(ev_tokens.intersection(attr_tokens)) if attr_tokens else 0
            overlap = len(ev_tokens.intersection(claim_tokens))
            score = attr_hits * 3.0 + overlap
            if score > best_score:
                best_score = score
                best_ev = ev

        return best_ev

    def _generate_explanation(
        self,
        claim: str,
        label: str,
        evidence: List[SearchResult],
        features: Dict[str, float]
    ) -> str:
        """Constructs an evidence-grounded textual explanation citing the exact sentence that supports the verdict."""
        if not evidence:
            return (
                "Insufficient Evidence: No relevant sentences were found in the knowledge base. "
                "The system cannot verify or refute this claim with the available data."
            )

        components = self.preprocessor.extract_claim_components(claim)
        subject = components.get("subject", "the subject")
        attr_text = components.get("attribute_text", "")
        top_ev = evidence[0]

        if label == "SUPPORTS":
            grounding_ev = self._find_best_supporting_sentence(claim, evidence)
            containment_pct = features.get("max_claim_containment", 0.0) * 100
            return (
                f"Evidence Grounding: The claim is supported by evidence from '{grounding_ev.page_title}' "
                f"(Sentence {grounding_ev.sentence_id}): \"{grounding_ev.text}\". "
                f"The evidence directly confirms the asserted fact with {containment_pct:.1f}% claim alignment."
            )

        elif label == "REFUTES":
            loc_contra = self._check_location_contradiction(claim, evidence)
            num_contra = self._check_numeric_contradiction(claim, evidence)
            ord_contra = self._check_ordinal_contradiction(claim, evidence)
            dem_contra = self._check_demonym_contradiction(claim, evidence)
            neg_contra = self._check_negation_contradiction(claim, evidence)

            if loc_contra:
                c_loc, e_loc, ev = loc_contra
                return (
                    f"Contradiction Detected: The claim asserts the location is {c_loc}, but evidence from "
                    f"'{ev.page_title}' (Sentence {ev.sentence_id}) states: \"{ev.text}\", confirming the location is {e_loc}."
                )
            elif num_contra:
                c_num, e_num, ev = num_contra
                return (
                    f"Contradiction Detected: The claim asserts the date/number is {c_num}, but evidence from "
                    f"'{ev.page_title}' (Sentence {ev.sentence_id}) states: \"{ev.text}\", identifying {e_num} as the verified record."
                )
            elif ord_contra:
                c_ord, e_ord, ev = ord_contra
                return (
                    f"Contradiction Detected: The claim asserts {subject} was the {c_ord}, but evidence from "
                    f"'{ev.page_title}' (Sentence {ev.sentence_id}) states: \"{ev.text}\", confirming {subject} was the {e_ord}."
                )
            elif dem_contra:
                c_dem, e_dem, ev = dem_contra
                return (
                    f"Contradiction Detected: The claim asserts the nationality is {c_dem}, but evidence from "
                    f"'{ev.page_title}' (Sentence {ev.sentence_id}) states: \"{ev.text}\", identifying {e_dem} as the verified record."
                )
            elif neg_contra:
                return (
                    f"Contradiction Detected: The claim polarity directly contradicts evidence from "
                    f"'{neg_contra.page_title}' (Sentence {neg_contra.sentence_id}): \"{neg_contra.text}\"."
                )
            else:
                return (
                    f"Contradiction Detected: The claim is refuted by evidence from '{top_ev.page_title}' "
                    f"(Sentence {top_ev.sentence_id}): \"{top_ev.text}\". Analysis detected substantive factual contradiction."
                )

        else:  # NOT ENOUGH INFO
            if attr_text and features.get("predicate_containment", 0.0) == 0.0:
                return (
                    f"Insufficient Evidence: Retrieved passages from '{top_ev.page_title}' discuss {subject}, "
                    f"but contain no information regarding '{attr_text}'. "
                    "Under the FEVER benchmark protocol, an assertion cannot be verified or refuted without "
                    "explicit textual evidence, resulting in NOT ENOUGH INFO."
                )
            else:
                return (
                    f"Insufficient Evidence: While retrieved passages from '{top_ev.page_title}' discuss related context, "
                    "they do not provide explicit confirmation or refutation of the specific assertion in the claim."
                )

    def verify(
        self,
        claim_text: str,
        retrieved_evidence: List[SearchResult]
    ) -> VerificationOutput:
        """Predicts verification verdict and generates an interpretable explanation."""
        if not claim_text or not claim_text.strip():
            return VerificationOutput(
                claim="",
                predicted_label="NOT ENOUGH INFO",
                confidence=1.0,
                probabilities={"SUPPORTS": 0.0, "REFUTES": 0.0, "NOT ENOUGH INFO": 1.0},
                evidence=[],
                explanation="Empty claim provided. Cannot verify.",
                feature_highlights={}
            )

        # 1. Feature extraction
        feat_vec = self.feature_extractor.extract_features(claim_text, retrieved_evidence)
        feat_dict = dict(zip(self.feature_extractor.feature_names, feat_vec.tolist()))

        # 2. Heuristic check for total lack of evidence
        if not retrieved_evidence or feat_dict["max_retrieval_score"] < 0.01:
            comparison = self._build_comparison(claim_text, [], "NOT ENOUGH INFO", feat_dict)
            return VerificationOutput(
                claim=claim_text,
                predicted_label="NOT ENOUGH INFO",
                confidence=0.95,
                probabilities={"SUPPORTS": 0.02, "REFUTES": 0.03, "NOT ENOUGH INFO": 0.95},
                evidence=retrieved_evidence,
                explanation="No relevant evidence sentences could be retrieved from the Wikipedia knowledge base.",
                feature_highlights=feat_dict,
                comparison=comparison
            )

        components = self.preprocessor.extract_claim_components(claim_text)
        subject = components.get("subject", "")
        attr_tokens = components.get("attribute_tokens", set())
        pred_containment = feat_dict.get("predicate_containment", 0.0)

        # Explicit contradiction checks
        loc_contra = self._check_location_contradiction(claim_text, retrieved_evidence)
        num_contra = self._check_numeric_contradiction(claim_text, retrieved_evidence)
        ord_contra = self._check_ordinal_contradiction(claim_text, retrieved_evidence)
        dem_contra = self._check_demonym_contradiction(claim_text, retrieved_evidence)
        neg_contra = self._check_negation_contradiction(claim_text, retrieved_evidence)

        # Check for direct supporting evidence about the subject
        subject_evs = [ev for ev in retrieved_evidence if self._is_sentence_about_subject(subject, ev)]
        entities = components.get("entities", [])

        def token_match(t1: str, t2: str) -> bool:
            if t1 == t2:
                return True
            if len(t1) >= 4 and len(t2) >= 4 and (t1.startswith(t2) or t2.startswith(t1)):
                return True
            return False

        direct_support_ev = None
        if subject_evs and not neg_contra:
            subject_tokens = set()
            for ev in subject_evs:
                subject_tokens.update(self.preprocessor.preprocess(ev.text))

            if attr_tokens:
                matched_attr = sum(1 for at in attr_tokens if any(token_match(at, st) for st in subject_tokens))
                attr_hits_ratio = matched_attr / len(attr_tokens)
            else:
                attr_hits_ratio = 1.0

            if len(attr_tokens) >= 2 and attr_hits_ratio < 0.34:
                ent_relational_ok = False
            else:
                ent_relational_ok = False
                for ev in subject_evs:
                    ev_full = (ev.text + " " + ev.page_title).lower()
                    ents_in_ev = sum(
                        1 for e in entities
                        if e.lower() in ev_full or any(sub.lower() in ev_full for sub in e.split() if len(sub) > 2)
                    )
                    if len(entities) >= 3 and ents_in_ev / len(entities) >= 0.8:
                        ent_relational_ok = True
                        break

            if attr_hits_ratio >= 0.5 or ent_relational_ok:
                direct_support_ev = self._find_best_supporting_sentence(claim_text, subject_evs)

        # 3. Model prediction (if available)
        predicted_label = "NOT ENOUGH INFO"
        confidence = 0.85
        prob_dict = {"SUPPORTS": 0.05, "REFUTES": 0.05, "NOT ENOUGH INFO": 0.90}

        if self.model is not None and hasattr(self.model, "predict_proba"):
            try:
                probs = self.model.predict_proba(feat_vec.reshape(1, -1))[0]
                prob_dict = {
                    "SUPPORTS": float(probs[LABEL_TO_ID["SUPPORTS"]]),
                    "REFUTES": float(probs[LABEL_TO_ID["REFUTES"]]),
                    "NOT ENOUGH INFO": float(probs[LABEL_TO_ID["NOT ENOUGH INFO"]])
                }
                pred_idx = int(np.argmax(probs))
                predicted_label = ID_TO_LABEL[pred_idx]
                confidence = float(probs[pred_idx])
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Using rule fallback.")
                predicted_label, confidence, prob_dict = self._rule_based_verification(feat_dict)
        else:
            predicted_label, confidence, prob_dict = self._rule_based_verification(feat_dict)

        # =========================================================================
        # 4. Strict FEVER Predicate Grounding & Contradiction Guard
        # =========================================================================
        # Case A: Explicit factual contradictions strictly result in REFUTES
        if loc_contra:
            predicted_label = "REFUTES"
            confidence = max(confidence, 0.92)
            prob_dict = {"SUPPORTS": 0.02, "REFUTES": 0.92, "NOT ENOUGH INFO": 0.06}
        elif num_contra:
            predicted_label = "REFUTES"
            confidence = max(confidence, 0.90)
            prob_dict = {"SUPPORTS": 0.03, "REFUTES": 0.90, "NOT ENOUGH INFO": 0.07}
        elif ord_contra:
            predicted_label = "REFUTES"
            confidence = max(confidence, 0.92)
            prob_dict = {"SUPPORTS": 0.02, "REFUTES": 0.92, "NOT ENOUGH INFO": 0.06}
        elif dem_contra:
            predicted_label = "REFUTES"
            confidence = max(confidence, 0.90)
            prob_dict = {"SUPPORTS": 0.03, "REFUTES": 0.90, "NOT ENOUGH INFO": 0.07}
        elif neg_contra:
            predicted_label = "REFUTES"
            confidence = max(confidence, 0.88)
            prob_dict = {"SUPPORTS": 0.04, "REFUTES": 0.88, "NOT ENOUGH INFO": 0.08}

        # Case B: Direct supporting evidence about the subject strictly results in SUPPORTS
        elif direct_support_ev is not None:
            predicted_label = "SUPPORTS"
            confidence = max(confidence, 0.90)
            prob_dict = {"SUPPORTS": 0.90, "REFUTES": 0.05, "NOT ENOUGH INFO": 0.05}

        # Case C: Absence of direct supporting evidence or unmentioned attributes MUST BE NOT ENOUGH INFO
        else:
            predicted_label = "NOT ENOUGH INFO"
            confidence = max(confidence, 0.90)
            prob_dict = {"SUPPORTS": 0.05, "REFUTES": 0.05, "NOT ENOUGH INFO": 0.90}

        explanation = self._generate_explanation(claim_text, predicted_label, retrieved_evidence, feat_dict)
        comparison = self._build_comparison(claim_text, retrieved_evidence, predicted_label, feat_dict)

        return VerificationOutput(
            claim=claim_text,
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities=prob_dict,
            evidence=retrieved_evidence,
            explanation=explanation,
            feature_highlights=feat_dict,
            comparison=comparison
        )

    def _build_comparison(
        self,
        claim: str,
        evidence: List[SearchResult],
        label: str,
        features: Dict[str, float]
    ) -> ComponentComparison:
        """Constructs a structured component-level comparison between the claim and top evidence."""
        if not evidence:
            return ComponentComparison(
                rows=[
                    {"Component": "Claim query", "Result": claim},
                    {"Component": "Retrieved evidence", "Result": "None found"},
                    {"Component": "Relationship", "Result": "Insufficient Evidence"},
                    {"Component": "Evidence sufficient?", "Result": "No"}
                ],
                relationship="Insufficient Evidence",
                evidence_sufficient="No",
                summary="No relevant evidence was retrieved to verify this claim."
            )

        top_ev = evidence[0]
        claim_clean = claim.strip()
        components = self.preprocessor.extract_claim_components(claim_clean)
        subject = components.get("subject", "Claim subject")
        attr_text = components.get("attribute_text", "asserted attribute")

        loc_contra = self._check_location_contradiction(claim_clean, evidence)
        num_contra = self._check_numeric_contradiction(claim_clean, evidence)
        ord_contra = self._check_ordinal_contradiction(claim_clean, evidence)
        dem_contra = self._check_demonym_contradiction(claim_clean, evidence)

        rows: List[Dict[str, str]] = []

        # Case 1: Location contradiction / corroboration
        if loc_contra and label == "REFUTES":
            c_loc, e_loc, ev = loc_contra
            rows.append({"Component": "Claim location", "Result": c_loc})
            rows.append({"Component": "Evidence location", "Result": e_loc})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The evidence identifies {e_loc} as the location, contradicting the claim that the subject is in {c_loc}."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 2: Number / Date contradiction
        if num_contra and label == "REFUTES":
            c_num, e_num, ev = num_contra
            rows.append({"Component": "Claim numeric assertion", "Result": c_num})
            rows.append({"Component": "Evidence record", "Result": str(e_num)})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The evidence specifies {e_num}, contradicting the claim's numeric assertion of {c_num}."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 2b: Ordinal contradiction
        if ord_contra and label == "REFUTES":
            c_ord, e_ord, ev = ord_contra
            rows.append({"Component": "Claim ordinal assertion", "Result": c_ord})
            rows.append({"Component": "Evidence record", "Result": e_ord})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The evidence specifies {e_ord}, contradicting the claim's ordinal assertion of {c_ord}."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 2c: Demonym / Nationality contradiction
        if dem_contra and label == "REFUTES":
            c_dem, e_dem, ev = dem_contra
            rows.append({"Component": "Claim nationality assertion", "Result": c_dem})
            rows.append({"Component": "Evidence record", "Result": e_dem})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The evidence identifies {e_dem}, contradicting the claim's nationality assertion of {c_dem}."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 3: Negation mismatch
        if label == "REFUTES" and features.get("negation_mismatch", 0) > 0.5:
            claim_neg = features.get("claim_has_negation", 0) > 0.5
            rows.append({"Component": "Claim polarity", "Result": "Negative assertion" if claim_neg else "Affirmative assertion"})
            rows.append({"Component": "Evidence polarity", "Result": "Affirmative fact" if claim_neg else "Negative/Contradictory fact"})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = "The evidence asserts a factual polarity directly opposite to the claim."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 4: General REFUTES
        if label == "REFUTES":
            rows.append({"Component": "Claim assertion", "Result": claim_clean})
            rows.append({"Component": "Evidence fact", "Result": f"\"{top_ev.text}\" ({top_ev.page_title})"})
            rows.append({"Component": "Relationship", "Result": "Contradiction"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The retrieved Wikipedia evidence from '{top_ev.page_title}' contradicts the claim assertion regarding {subject}."
            return ComponentComparison(rows=rows, relationship="Contradiction", evidence_sufficient="Yes", summary=summary)

        # Case 5: General SUPPORTS
        if label == "SUPPORTS":
            grounding_ev = self._find_best_supporting_sentence(claim_clean, evidence)
            rows.append({"Component": "Claim assertion", "Result": claim_clean})
            rows.append({"Component": "Corroborating evidence", "Result": f"\"{grounding_ev.text}\" ({grounding_ev.page_title})"})
            rows.append({"Component": "Relationship", "Result": "Corroboration"})
            rows.append({"Component": "Evidence sufficient?", "Result": "Yes"})
            summary = f"The retrieved Wikipedia evidence from '{grounding_ev.page_title}' directly corroborates the claim."
            return ComponentComparison(rows=rows, relationship="Corroboration", evidence_sufficient="Yes", summary=summary)

        # Case 6: NOT ENOUGH INFO
        rows.append({"Component": f"Claim assertion ({subject})", "Result": claim_clean})
        rows.append({"Component": "Target attribute", "Result": attr_text if attr_text else "Key assertion"})
        rows.append({"Component": "Retrieved evidence status", "Result": f"Passages from '{top_ev.page_title}' do not mention '{attr_text}'"})
        rows.append({"Component": "Relationship", "Result": "Insufficient Evidence"})
        rows.append({"Component": "Evidence sufficient?", "Result": "No"})
        summary = f"The retrieved evidence from '{top_ev.page_title}' discusses {subject}, but contains no information regarding '{attr_text}'."
        return ComponentComparison(rows=rows, relationship="Insufficient Evidence", evidence_sufficient="No", summary=summary)

    def _rule_based_verification(self, feat_dict: Dict[str, float]) -> Tuple[str, float, Dict[str, float]]:
        """Interpretable fallback logic when model is not yet fitted."""
        containment = feat_dict["max_claim_containment"]
        pred_containment = feat_dict["predicate_containment"]
        neg_mismatch = feat_dict["negation_mismatch"]
        num_mismatch = feat_dict["numeric_mismatch"]

        if neg_mismatch > 0.5 or num_mismatch > 0.5:
            return "REFUTES", 0.88, {"SUPPORTS": 0.05, "REFUTES": 0.88, "NOT ENOUGH INFO": 0.07}

        if pred_containment == 0.0 or containment < 0.25:
            return "NOT ENOUGH INFO", 0.90, {"SUPPORTS": 0.04, "REFUTES": 0.06, "NOT ENOUGH INFO": 0.90}

        return "SUPPORTS", 0.86, {"SUPPORTS": 0.86, "REFUTES": 0.07, "NOT ENOUGH INFO": 0.07}
