"""Unit tests for inverted index construction and persistence."""
import pytest
import tempfile
import shutil
from pathlib import Path

from src.indexing.inverted_index import InvertedIndex
from src.indexing.build_index import IndexManager
from src.preprocessing.text_preprocessor import TextPreprocessor


def test_inverted_index_basic_operations():
    index = InvertedIndex()
    index.add_sentence(
        record_id=0,
        doc_id="Alan_Turing",
        sentence_id=0,
        text="Alan Turing was an English computer scientist.",
        page_title="Alan Turing",
        tokens=["alan", "turing", "english", "comput", "scientist"]
    )
    index.add_sentence(
        record_id=1,
        doc_id="Nikolaj_Coster-Waldau",
        sentence_id=0,
        text="Nikolaj Coster-Waldau is a Danish actor.",
        page_title="Nikolaj Coster-Waldau",
        tokens=["nikolaj", "coster-waldau", "danish", "actor"]
    )
    index.finalize()

    assert index.total_documents == 2
    assert index.vocabulary_size == 9
    assert index.get_doc_frequency("turing") == 1
    assert index.get_doc_frequency("unknown_word") == 0

    postings = index.get_postings("turing")
    assert len(postings) == 1
    assert postings[0][0] == 0  # record_id 0


def test_index_save_and_reload(tmp_path):
    index = InvertedIndex()
    index.add_sentence(
        record_id=0,
        doc_id="Apollo_11",
        sentence_id=1,
        text="Neil Armstrong landed on the Moon.",
        page_title="Apollo 11",
        tokens=["neil", "armstrong", "land", "moon"]
    )
    index.finalize()

    save_dir = tmp_path / "test_index"
    index.save(str(save_dir))

    loaded_index = InvertedIndex.load(str(save_dir))
    assert loaded_index.total_documents == 1
    assert loaded_index.get_doc_frequency("armstrong") == 1
    rec = loaded_index.get_record(0)
    assert rec.doc_id == "Apollo_11"
    assert rec.sentence_id == 1
