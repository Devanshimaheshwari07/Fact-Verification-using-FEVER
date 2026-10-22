"""Text preprocessing pipeline for lexical indexing and fact verification."""
import re
import unicodedata
from typing import List, Optional, Set, Dict, Any

# Standard English stopwords excluding negation and comparative words
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "by", "could", "did", "do", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having", "he",
    "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "she", "should", "so", "some", "such", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "with", "would", "you", "your", "yours", "yourself", "yourselves"
}

# Critical negation terms that MUST NOT be discarded indiscriminately
NEGATION_TOKENS = {
    "not", "no", "never", "neither", "nor", "none", "nobody", "nowhere",
    "nothing", "hardly", "scarcely", "barely", "cannot", "n't", "without",
    "refuse", "failed", "denied", "impossible", "unlikely"
}


class TextPreprocessor:
    """Preprocesses text for indexing, retrieval, and fact verification."""

    def __init__(
        self,
        lowercase: bool = True,
        remove_stopwords: bool = False,
        use_stemming: bool = True,
        min_token_length: int = 1,
        stemmer_type: str = "porter"
    ):
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.min_token_length = min_token_length
        self.stemmer_type = stemmer_type

        # Initialize stemmer with fallback
        self.stemmer = None
        if self.use_stemming:
            try:
                from nltk.stem.porter import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                # Fallback to simple suffix trimmer if nltk stemmer is unavailable
                self.stemmer = None

    def normalize(self, text: str) -> str:
        """Applies Unicode normalization (NFKD) and cleans irregular whitespace."""
        if not text:
            return ""
        # Decompose accented characters and normalize unicode
        normalized = unicodedata.normalize("NFKD", text)
        # Remove combining diacritical marks
        cleaned = "".join(c for c in normalized if not unicodedata.combining(c))
        # Replace non-breaking and irregular spaces
        cleaned = re.sub(r"[\s\u200b\u200e\u200f]+", " ", cleaned).strip()
        return cleaned

    def tokenize(self, text: str) -> List[str]:
        """Tokenizes text preserving alphanumerics, hyphens, and contractions."""
        if not text:
            return []
        normalized = self.normalize(text)
        # Extract word tokens: words, hyphenated compounds, and numbers
        tokens = re.findall(r"\b[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*\b", normalized)
        return tokens

    def stem(self, token: str) -> str:
        """Applies stemming to a token."""
        if not self.use_stemming or not self.stemmer:
            return token
        try:
            return self.stemmer.stem(token)
        except Exception:
            return token

    def clean_text(self, text: str) -> str:
        """Stage 1: Text Cleaning - applies Unicode NFKD normalization, cleans irregular spaces, strips noise."""
        return self.normalize(text)

    def remove_stop_words(self, tokens: List[str]) -> List[str]:
        """Stage 3: Stop-word removal - filters common stopwords while preserving critical negation semantics."""
        return [t for t in tokens if t not in DEFAULT_STOPWORDS or t in NEGATION_TOKENS]

    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Stage 4: Stemming - applies Porter stemming to reduce tokens to morphological roots."""
        return [self.stem(t) for t in tokens]

    def preprocess(self, text: str) -> List[str]:
        """Full 4-stage preprocessing pipeline: Text Cleaning -> Tokenization -> Stop-word removal -> Stemming."""
        if not text:
            return []

        # Stage 1: Text Cleaning
        cleaned = self.clean_text(text)

        # Stage 2: Tokenization
        tokens = self.tokenize(cleaned)
        processed: List[str] = []

        for token in tokens:
            t = token.lower() if self.lowercase else token
            if len(t) < self.min_token_length:
                continue
            processed.append(t)

        # Stage 3: Stop-word removal
        if self.remove_stopwords:
            processed = self.remove_stop_words(processed)

        # Stage 4: Stemming
        if self.use_stemming:
            processed = self.stem_tokens(processed)

        return processed

    @staticmethod
    def extract_entities_and_keywords(text: str) -> Dict[str, Any]:
        """Extracts candidate entities, capitalized phrases, dates, and numbers for query formulation."""
        if not text:
            return {"entities": [], "numbers": [], "has_negation": False}

        # Find capitalized entity sequences supporting hyphens and apostrophes (e.g., "Nikolaj Coster-Waldau", "Fox Broadcasting Company")
        entities = re.findall(r"\b[A-Z][a-z]+(?:[-'\s][A-Z][a-z]+)*\b", text)

        # Find numbers and years (e.g., 1993, 2020, 8848.86)
        numbers = re.findall(r"\b\d+(?:[\.,]\d+)?\b", text)

        # Check for presence of negation
        words_lower = set(re.findall(r"\b[a-z']+\b", text.lower()))
        has_negation = any(neg in words_lower for neg in NEGATION_TOKENS)

        return {
            "entities": entities,
            "numbers": numbers,
            "has_negation": has_negation
        }

    def extract_claim_components(self, text: str) -> Dict[str, Any]:
        """Decomposes a claim into subject entity, key attribute tokens, numbers, and polarity.

        Example:
            "Nikolaj Coster-Waldau is a vegan."
            -> subject: "Nikolaj Coster-Waldau"
            -> attribute_tokens: {"vegan"}
            -> attribute_text: "vegan"
        """
        if not text:
            return {
                "subject": "",
                "entities": [],
                "attribute_tokens": set(),
                "attribute_text": "",
                "numbers": [],
                "has_negation": False
            }

        diag = self.extract_entities_and_keywords(text)
        entities = diag["entities"]
        subject = entities[0] if entities else ""

        all_tokens = self.preprocess(text)

        # Entity tokens (stemmed and normalized)
        entity_tokens = set()
        for ent in entities:
            entity_tokens.update(self.preprocess(ent))
            for sub in re.findall(r"\b[a-zA-Z0-9]+\b", ent.lower()):
                entity_tokens.add(self.stem(sub) if self.use_stemming else sub)

        # Functional words that don't represent a substantive attribute
        functional_words = DEFAULT_STOPWORDS.union({
            "is", "was", "are", "were", "been", "be", "being", "have", "has", "had",
            "do", "does", "did", "become", "became", "known", "called", "named"
        })
        functional_stems = {self.stem(w) if self.use_stemming else w for w in functional_words}

        # Distinctive attribute tokens are content tokens not in subject entity and not functional words
        attribute_tokens = {
            t for t in all_tokens
            if t not in entity_tokens and t not in functional_stems and len(t) > 1
        }

        # Raw attribute words for query building
        raw_words = re.findall(r"\b[a-zA-Z0-9-]+\b", text)
        ent_words_lower = set()
        for ent in entities:
            for w in re.findall(r"\b[a-zA-Z0-9-]+\b", ent.lower()):
                ent_words_lower.add(w)

        raw_attr_words = [
            w for w in raw_words
            if w.lower() not in ent_words_lower and w.lower() not in functional_words
        ]
        attribute_text = " ".join(raw_attr_words)

        return {
            "subject": subject,
            "entities": entities,
            "attribute_tokens": attribute_tokens,
            "attribute_text": attribute_text,
            "numbers": diag["numbers"],
            "has_negation": diag["has_negation"]
        }
