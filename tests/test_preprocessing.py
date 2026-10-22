"""Unit tests for text preprocessing and entity extraction."""
import pytest
from src.preprocessing.text_preprocessor import TextPreprocessor, NEGATION_TOKENS


def test_unicode_normalization():
    prep = TextPreprocessor()
    text = "Nikolaj Coster\u2013Waldau café resume\u0301"
    normalized = prep.normalize(text)
    assert "cafe" in normalized
    assert "resume" in normalized


def test_tokenization_and_stemming():
    prep = TextPreprocessor(lowercase=True, use_stemming=True)
    text = "The lunar module landed on July 20, 1969."
    tokens = prep.preprocess(text)
    assert "lunar" in tokens
    assert "modul" in tokens  # Stemmed form of module
    assert "land" in tokens   # Stemmed form of landed
    assert "1969" in tokens


def test_negation_preservation():
    # Negation words must not be discarded even if remove_stopwords is enabled
    prep = TextPreprocessor(lowercase=True, remove_stopwords=True, use_stemming=False)
    text = "The machine was not cracked by Americans"
    tokens = prep.preprocess(text)
    assert "not" in tokens
    assert "cracked" in tokens


def test_entity_and_number_extraction():
    prep = TextPreprocessor()
    text = "Alan Turing arrived at Bletchley Park in 1939 with 45 colleagues."
    extracted = prep.extract_entities_and_keywords(text)
    assert "Alan Turing" in extracted["entities"]
    assert "Bletchley Park" in extracted["entities"]
    assert "1939" in extracted["numbers"]
    assert "45" in extracted["numbers"]
    assert not extracted["has_negation"]


def test_empty_input_handling():
    prep = TextPreprocessor()
    assert prep.preprocess("") == []
    assert prep.normalize("") == ""
    assert prep.tokenize("") == []
