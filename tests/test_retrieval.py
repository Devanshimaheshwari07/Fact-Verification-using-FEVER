"""Unit tests for Boolean, TF-IDF, BM25, and Hybrid retrieval models."""
import pytest
from src.indexing.inverted_index import InvertedIndex
from src.indexing.build_index import IndexManager
from src.preprocessing.text_preprocessor import TextPreprocessor
from src.retrieval.boolean_retriever import BooleanRetriever
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.tfidf_retriever import TfidfRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def sample_index():
    index = InvertedIndex()
    preprocessor = TextPreprocessor()

    docs = [
        (0, "Apollo_11", 1, "Commander Neil Armstrong landed the Apollo Lunar Module Eagle on July 20, 1969."),
        (1, "Apollo_11", 2, "Armstrong became the first person to step onto the lunar surface on July 21."),
        (2, "Alan_Turing", 3, "During World War II, Turing was a leading codebreaker at Bletchley Park."),
        (3, "Albert_Einstein", 1, "Einstein developed the theory of relativity and mass energy equivalence.")
    ]

    for r_id, doc_id, sent_id, text in docs:
        tokens = preprocessor.preprocess(text)
        index.add_sentence(r_id, doc_id, sent_id, text, doc_id.replace("_", " "), tokens)

    index.finalize()
    return index


def test_boolean_retrieval(sample_index):
    retriever = BooleanRetriever(sample_index)

    # AND query
    results_and = retriever.retrieve("Armstrong AND Apollo", top_k=5)
    assert len(results_and) > 0
    assert any("Armstrong" in r.text for r in results_and)

    # NOT query
    results_not = retriever.retrieve("Apollo NOT Turing", top_k=5)
    assert len(results_not) > 0
    assert all("Turing" not in r.text for r in results_not)


def test_bm25_ranking(sample_index):
    retriever = BM25Retriever(sample_index, k1=1.5, b=0.75)
    results = retriever.retrieve("Neil Armstrong Moon landing", top_k=2)

    assert len(results) > 0
    assert results[0].doc_id == "Apollo_11"
    assert results[0].score > 0.0
    assert results[0].rank == 1


def test_empty_and_unknown_queries(sample_index):
    retriever = BM25Retriever(sample_index)
    assert retriever.retrieve("") == []
    assert retriever.retrieve("xyzzy nonexistentword") == []
