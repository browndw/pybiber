"""Tests for the high-level PybiberPipeline wrapper."""

import polars as pl
import pytest

from pybiber import PybiberPipeline


@pytest.mark.skipif(True, reason="placeholder")
def _placeholder():
    # Placeholder to ensure the file is recognized as a test module
    pass


def test_pipeline_run_returns_features_and_tokens(sample_corpus, nlp_model):
    pipeline = PybiberPipeline(
        nlp=nlp_model, disable_ner=True, n_process=1, batch_size=8
    )

    feats, tokens = pipeline.run(sample_corpus, return_tokens=True)

    # Features: must include doc_id and at least one numeric feature
    assert "doc_id" in feats.columns
    num_cols = feats.select(pl.selectors.numeric()).columns
    assert len(num_cols) > 0

    # Tokens: basic schema presence and sentence_id should be non-boolean
    for col in [
        "doc_id",
        "sentence_id",
        "token_id",
        "token",
        "lemma",
        "pos",
        "tag",
        "head_token_id",
        "dep_rel",
    ]:
        assert col in tokens.columns

    dtype = tokens.get_column("sentence_id").dtype
    assert dtype not in (pl.Boolean,), (
        f"unexpected dtype for sentence_id: {dtype}"
    )
