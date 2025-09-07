"""
Pytest configuration and fixtures for pybiber tests.
"""

import pytest
import spacy
import polars as pl
from pybiber import spacy_parse
from .test_data import SAMPLE_TEXTS


@pytest.fixture(scope="session")
def nlp_model():
    """Load spaCy model once for all tests."""
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        pytest.skip("en_core_web_sm model not available")


@pytest.fixture
def sample_corpus():
    """Create a sample corpus from test data."""
    doc_ids = list(SAMPLE_TEXTS.keys())
    texts = list(SAMPLE_TEXTS.values())

    return pl.DataFrame({
        "doc_id": doc_ids,
        "text": texts
    })


@pytest.fixture
def parsed_corpus(sample_corpus, nlp_model):
    """Parse the sample corpus using spaCy."""
    return spacy_parse(sample_corpus, nlp_model, n_process=1, batch_size=10)


@pytest.fixture
def sample_feature_matrix():
    """Create a sample feature matrix for BiberAnalyzer testing."""
    return pl.DataFrame({
        "doc_id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "category": ["A", "A", "B", "B", "C"],
        "f_01_feature": [10.5, 12.3, 8.7, 9.1, 15.2],
        "f_02_feature": [20.3, 18.9, 25.1, 22.4, 19.8],
        "f_03_feature": [5.7, 6.2, 4.8, 5.9, 7.1]
    })


@pytest.fixture
def micusp_mini_corpus():
    """Load the micusp mini corpus for integration tests."""
    try:
        return pl.read_parquet("pybiber/data/micusp_mini.parquet")
    except Exception:
        pytest.skip("micusp_mini.parquet not available")
