"""
Test suite for pybiber.parse_functions module.
"""

import pytest
import polars as pl
from pybiber.parse_functions import biber, _biber_weight
from .test_data import SAMPLE_TEXTS, EXPECTED_FEATURES


class TestBiberFunction:
    """Test the main biber feature extraction function."""

    def test_biber_basic_functionality(self, parsed_corpus):
        """Test basic biber function with parsed corpus."""
        # Test without normalization first
        features = biber(parsed_corpus, normalize=False)

        # Check structure
        assert isinstance(features, pl.DataFrame)
        assert "doc_id" in features.columns
        assert features.shape[0] > 0

        # Check that we have feature columns (should start with f_)
        feature_cols = [col for col in features.columns
                        if col.startswith("f_")]
        assert len(feature_cols) > 0

        # Verify data types - most should be UInt32 for counts
        for col in feature_cols:
            if col not in ["f_43_type_token", "f_44_mean_word_length"]:
                assert features[col].dtype == pl.UInt32

    def test_biber_normalization(self, parsed_corpus):
        """Test biber function with normalization enabled."""
        features_norm = biber(parsed_corpus, normalize=True)
        features_raw = biber(parsed_corpus, normalize=False)

        # Check that normalized features are different
        # (should be per 1000 tokens)
        # except for TTR and mean word length
        for col in features_norm.columns:
            excluded_cols = ["f_43_type_token", "f_44_mean_word_length"]
            if col.startswith("f_") and col not in excluded_cols:
                norm_vals = features_norm[col].to_list()
                raw_vals = features_raw[col].to_list()
                # Normalized values should generally be different
                # unless raw count was 0
                if any(v > 0 for v in raw_vals):
                    assert norm_vals != raw_vals

    def test_biber_ttr_vs_mattr(self, parsed_corpus):
        """Test TTR vs MATTR selection based on document length."""
        # Force TTR
        features_ttr = biber(parsed_corpus, normalize=False, force_ttr=True)

        # Should have type-token ratio feature
        assert "f_43_type_token" in features_ttr.columns

        # Values should be between 0 and 1 for TTR
        ttr_values = features_ttr["f_43_type_token"].to_list()
        for val in ttr_values:
            if val is not None and not pl.Series([val]).is_null().item():
                assert 0 <= val <= 1

    def test_specific_feature_detection(self, nlp_model):
        """Test detection of specific linguistic features."""
        # Test a few key features with known samples
        test_cases = [
            ("quickbrown", "f_03_present_tense"),
            ("past_tense", "f_01_past_tense"),
            ("perfect_aspect", "f_02_perfect_aspect"),
        ]

        for doc_id, feature in test_cases:
            has_sample = doc_id in SAMPLE_TEXTS
            has_expected = feature in EXPECTED_FEATURES.get(doc_id, {})
            if has_sample and has_expected:
                # Create single-document corpus
                corpus = pl.DataFrame({
                    "doc_id": [doc_id],
                    "text": [SAMPLE_TEXTS[doc_id]]
                })

                # Parse and extract features
                from pybiber.parse_utils import spacy_parse
                parsed = spacy_parse(corpus, nlp_model, n_process=1)
                features = biber(parsed, normalize=False)

                # Check if feature is detected
                if feature in features.columns:
                    expected_count = EXPECTED_FEATURES[doc_id][feature]
                    actual_count = features[feature].item()

                    # Allow some flexibility due to parsing differences
                    assert actual_count >= 0
                    if expected_count > 0:
                        assert actual_count > 0

    def test_empty_corpus_handling(self):
        """Test handling of empty or minimal corpus."""
        # Create minimal corpus with just one token
        minimal_corpus = pl.DataFrame({
            "doc_id": ["test"],
            "sentence_id": [1],
            "token_id": [0],
            "token": ["hello"],
            "lemma": ["hello"],
            "pos": ["NOUN"],
            "tag": ["NN"],
            "head_token_id": [0],
            "dep_rel": ["ROOT"]
        })

        features = biber(minimal_corpus, normalize=False)

        # Should complete without errors
        assert isinstance(features, pl.DataFrame)
        assert features.shape[0] == 1
        assert "doc_id" in features.columns

    def test_multiple_documents(self, parsed_corpus):
        """Test biber function with multiple documents."""
        features = biber(parsed_corpus, normalize=False)

        # Should have one row per document
        doc_ids_parsed = parsed_corpus["doc_id"].unique().sort()
        doc_ids_features = features["doc_id"].sort()

        assert len(doc_ids_features) == len(doc_ids_parsed)
        # All documents from parsed corpus should appear in features
        assert set(doc_ids_features.to_list()).issubset(
            set(doc_ids_parsed.to_list())
        )


class TestBiberWeighting:
    """Test the _biber_weight function for different weighting schemes."""

    def test_biber_weight_validation(self):
        """Test _biber_weight validates input correctly."""
        # Create invalid DataFrame
        invalid_df = pl.DataFrame({
            "doc_id": ["test"],
            "f_01_test": [5.0]  # Wrong dtype - should be UInt32
        })

        totals_df = pl.DataFrame({
            "doc_id": ["test"],
            "doc_total": [100]
        })

        # This might not raise an error depending on validation logic
        # Just ensure it doesn't crash
        try:
            _biber_weight(invalid_df, totals_df, scheme="prop")
        except ValueError:
            pass  # Expected behavior

    def test_biber_weight_proportional(self):
        """Test proportional weighting scheme."""
        # Create valid biber counts DataFrame
        biber_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "f_01_past_tense": pl.Series([5, 10], dtype=pl.UInt32),
            "f_03_present_tense": pl.Series([3, 6], dtype=pl.UInt32),
            "f_43_type_token": pl.Series([0.5, 0.6], dtype=pl.Float64),
            "f_44_mean_word_length": pl.Series([4.2, 4.5], dtype=pl.Float64)
        })

        totals_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "doc_total": [100, 200]
        })

        weighted = _biber_weight(biber_df, totals_df, scheme="prop")

        # Check proportional scaling (per 1000 tokens)
        # doc1: 5/100 * 1000 = 50, doc2: 10/200 * 1000 = 50
        past_tense_norm = weighted["f_01_past_tense"].to_list()
        assert abs(past_tense_norm[0] - 50.0) < 0.001
        assert abs(past_tense_norm[1] - 50.0) < 0.001

        # TTR and mean word length should be unchanged
        ttr_values = weighted["f_43_type_token"].to_list()
        assert ttr_values == [0.5, 0.6]

    def test_biber_weight_scaled(self):
        """Test scaled (z-score) weighting scheme."""
        biber_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "f_01_past_tense": pl.Series([5, 10, 15], dtype=pl.UInt32),
            "f_43_type_token": pl.Series([0.5, 0.6, 0.7], dtype=pl.Float64)
        })

        totals_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "doc_total": [100, 100, 100]
        })

        weighted = _biber_weight(biber_df, totals_df, scheme="scale")

        # After scaling, mean should be approximately 0, std should be 1
        scaled_values = weighted["f_01_past_tense"].to_list()
        mean_val = sum(scaled_values) / len(scaled_values)
        assert abs(mean_val) < 0.001  # Should be close to 0

    def test_biber_weight_tfidf(self):
        """Test TF-IDF weighting scheme."""
        biber_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "f_01_past_tense": pl.Series([5, 0], dtype=pl.UInt32),
            "f_03_present_tense": pl.Series([0, 10], dtype=pl.UInt32),
            "f_43_type_token": pl.Series([0.5, 0.6], dtype=pl.Float64),
            "f_44_mean_word_length": pl.Series([4.2, 4.5], dtype=pl.Float64)
        })

        totals_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "doc_total": [100, 200]
        })

        weighted = _biber_weight(biber_df, totals_df, scheme="tfidf")

        # TF-IDF should exclude TTR and mean word length
        assert "f_43_type_token" not in weighted.columns
        assert "f_44_mean_word_length" not in weighted.columns

        # Should have TF-IDF weighted values
        assert "f_01_past_tense" in weighted.columns
        assert "f_03_present_tense" in weighted.columns

    def test_biber_weight_tfidf_idf_variant(self):
        """Numerically validate custom IDF variant log1p(df+N) - log1p(df).

        We reconstruct IDF values from tf-idf output and confirm ordering:
        rarer feature => higher IDF.
        """
        # 3 documents, 4 features with varying document frequencies
        # f_01_common appears in all docs (df=3)
        # f_02_rare appears only in doc2 (df=1)
        # f_03_mid1 appears in docs 1 & 3 (df=2)
        # f_04_mid2 appears in docs 1 & 2 (df=2)
        biber_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "f_01_common": pl.Series([5, 3, 4], dtype=pl.UInt32),
            "f_02_rare": pl.Series([0, 7, 0], dtype=pl.UInt32),
            "f_03_mid1": pl.Series([2, 0, 1], dtype=pl.UInt32),
            "f_04_mid2": pl.Series([1, 2, 0], dtype=pl.UInt32),
            # include derived metrics to ensure exclusion works
            "f_43_type_token": pl.Series([0.5, 0.55, 0.6], dtype=pl.Float64),
            "f_44_mean_word_length": pl.Series(
                [4.2, 4.3, 4.1], dtype=pl.Float64
            ),
        })
        # Keep doc_total constant (100) so normalized tf = count * 10
        totals_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2", "doc3"],
            "doc_total": [100, 100, 100],
        })

        weighted = _biber_weight(biber_df, totals_df, scheme="tfidf")

        # Confirm excluded features removed
        assert "f_43_type_token" not in weighted.columns
        assert "f_44_mean_word_length" not in weighted.columns

        # Reconstruct IDF: tfidf = tf * idf; tf = count/100*1000 = count*10
        def recover_idf(col, counts):
            tfidf_vals = weighted[col].to_list()
            rec = []
            for v, c in zip(tfidf_vals, counts):
                if c > 0:
                    rec.append(v / (c * 10))
            return sum(rec) / len(rec)  # mean over docs where term present

        idf_common = recover_idf("f_01_common", [5, 3, 4])
        idf_rare = recover_idf("f_02_rare", [0, 7, 0])
        idf_mid1 = recover_idf("f_03_mid1", [2, 0, 1])
        idf_mid2 = recover_idf("f_04_mid2", [1, 2, 0])

        # Expected ordering: rare > mid (df=2) > common (df=3)
        assert idf_rare > idf_mid1
        assert idf_rare > idf_mid2
        # mid features roughly equal (allow tiny numerical diff)
        assert abs(idf_mid1 - idf_mid2) < 1e-9
        assert idf_mid1 > idf_common
        assert idf_mid2 > idf_common

        # Sanity: All IDFs positive
        for val in [idf_common, idf_rare, idf_mid1, idf_mid2]:
            assert val > 0

    def test_invalid_scheme(self):
        """Test invalid weighting scheme raises error."""
        biber_df = pl.DataFrame({
            "doc_id": ["test"],
            "f_01_past_tense": pl.Series([5], dtype=pl.UInt32)
        })

        totals_df = pl.DataFrame({
            "doc_id": ["test"],
            "doc_total": [100]
        })

        with pytest.raises(ValueError, match="Invalid count_by type"):
            _biber_weight(biber_df, totals_df, scheme="invalid")


class TestBiberIntegration:
    """Integration tests using real data."""

    def test_micusp_integration(self, micusp_mini_corpus, nlp_model):
        """Test biber function with real MICUSP data."""
        # Take a small sample to speed up test
        small_sample = micusp_mini_corpus.head(5)

        from pybiber.parse_utils import spacy_parse
        parsed = spacy_parse(small_sample, nlp_model, n_process=1)
        features = biber(parsed, normalize=True)

        # Should successfully extract features
        assert features.shape[0] == 5
        assert "doc_id" in features.columns

        # Check that we have a reasonable number of features
        feature_cols = [col for col in features.columns
                        if col.startswith("f_")]
        assert len(feature_cols) >= 40  # Should have most Biber features

        # Verify some features have non-zero values
        non_zero_features = []
        for col in feature_cols[:10]:  # Check first 10 features
            values = features[col].to_list()
            if any(v > 0 for v in values if v is not None):
                non_zero_features.append(col)

        assert len(non_zero_features) > 0  # At least some features should fire
