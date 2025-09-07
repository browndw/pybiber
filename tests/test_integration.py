"""
Snapshot and integration tests for pybiber.

These tests verify that the overall behavior of the package
remains consistent across versions.
"""

import polars as pl
from pybiber import (
    spacy_parse, biber, get_noun_phrases, BiberAnalyzer
)


class TestSnapshotBehavior:
    """Test that outputs remain consistent (snapshot testing)."""

    def test_biber_output_structure_consistency(self, parsed_corpus):
        """Test that biber output structure remains consistent."""
        features_raw = biber(parsed_corpus, normalize=False, force_ttr=True)
        features_norm = biber(parsed_corpus, normalize=True, force_ttr=True)

        # Check consistent structure
        assert isinstance(features_raw, pl.DataFrame)
        assert isinstance(features_norm, pl.DataFrame)

        # Same number of documents
        assert features_raw.shape[0] == features_norm.shape[0]

        # Same columns
        assert features_raw.columns == features_norm.columns

        # doc_id column should be identical
        raw_ids = features_raw["doc_id"].to_list()
        norm_ids = features_norm["doc_id"].to_list()
        assert raw_ids == norm_ids

    def test_feature_column_consistency(self, parsed_corpus):
        """Test that feature columns are consistent."""
        features = biber(parsed_corpus, normalize=False, force_ttr=True)

        # Should have doc_id column
        assert "doc_id" in features.columns

        # Should have Biber features (starting with f_)
        feature_cols = [col for col in features.columns
                        if col.startswith("f_")]

        # Should have a reasonable number of features (Biber has 67 features)
        assert len(feature_cols) >= 40  # At least most features
        assert len(feature_cols) <= 70  # Not too many unexpected ones

        # Specific important features should be present
        important_features = [
            "f_01_past_tense",
            "f_03_present_tense",
            "f_02_perfect_aspect",
            "f_43_type_token",
            "f_44_mean_word_length"
        ]

        for feature in important_features:
            assert feature in features.columns

    def test_spacy_parse_output_structure(self, sample_corpus, nlp_model):
        """Test spacy_parse output structure consistency."""
        parsed = spacy_parse(sample_corpus.head(3), nlp_model, n_process=1)

        expected_columns = [
            "doc_id", "sentence_id", "token_id", "token", "lemma",
            "pos", "tag", "head_token_id", "dep_rel"
        ]

        assert parsed.columns == expected_columns

        # Check data types
        assert parsed["sentence_id"].dtype == pl.UInt32
        assert parsed["token_id"].dtype == pl.Int64
        assert parsed["head_token_id"].dtype == pl.Int64
        assert parsed["token"].dtype == pl.String
        assert parsed["lemma"].dtype == pl.String

    def test_noun_phrases_output_structure(self, sample_corpus, nlp_model):
        """Test get_noun_phrases output structure consistency."""
        phrases = get_noun_phrases(sample_corpus.head(2), nlp_model,
                                   n_process=1)

        expected_columns = [
            "doc_id", "phrase_text", "phrase_tags", "phrase_len",
            "root_text", "root_tag", "root_idx", "start_idx", "end_idx"
        ]

        assert phrases.columns == expected_columns

        # Check data types
        assert phrases["phrase_len"].dtype in [pl.Int64, pl.UInt32, pl.UInt64]
        assert phrases["root_idx"].dtype in [pl.Int64, pl.UInt32, pl.UInt64]
        assert phrases["start_idx"].dtype in [pl.Int64, pl.UInt32, pl.UInt64]
        assert phrases["end_idx"].dtype in [pl.Int64, pl.UInt32, pl.UInt64]


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_with_sample_data(self, nlp_model):
        """Test complete pipeline from text to features."""
        # Create sample texts
        texts = {
            "doc1.txt": "The cat sat on the mat. It was comfortable.",
            "doc2.txt": "I think that you are right about this matter.",
            "doc3.txt": "Running quickly, she reached the finish line."
        }

        # Simulate file reading
        doc_ids = list(texts.keys())
        text_content = list(texts.values())

        corpus = pl.DataFrame({
            "doc_id": doc_ids,
            "text": text_content
        })

        # Parse with spaCy
        parsed = spacy_parse(corpus, nlp_model, n_process=1)

        # Extract Biber features
        features = biber(parsed, normalize=True, force_ttr=True)

        # Get noun phrases
        phrases = get_noun_phrases(corpus, nlp_model, n_process=1)

        # Initialize analyzer
        features_with_category = features.with_columns(
            pl.lit("test").alias("category")
        )
        analyzer = BiberAnalyzer(features_with_category, id_column=True)

        # Verify complete pipeline
        assert features.shape[0] == 3
        assert phrases.shape[0] >= 1  # Should find some phrases
        assert analyzer.doc_ids is not None
        assert len(analyzer.doc_ids) == 3

    def test_micusp_integration_sample(self, micusp_mini_corpus, nlp_model):
        """Test integration with actual MICUSP data sample."""
        # Use small sample for speed
        sample = micusp_mini_corpus.head(3)

        # Full pipeline
        parsed = spacy_parse(sample, nlp_model, n_process=1)
        features = biber(parsed, normalize=True)
        phrases = get_noun_phrases(sample, nlp_model, n_process=1)

        # Should complete without errors
        assert features.shape[0] == 3
        assert phrases.shape[0] >= 1

        # Features should have reasonable values
        feature_cols = [col for col in features.columns
                        if col.startswith("f_")]

        # At least some features should have non-zero values
        non_zero_count = 0
        for col in feature_cols[:10]:  # Check first 10 features
            values = features[col].to_list()
            if any(v > 0 for v in values if v is not None):
                non_zero_count += 1

        assert non_zero_count >= 3  # At least some features should fire

    def test_empty_text_handling_pipeline(self, nlp_model):
        """Test pipeline handles empty/minimal texts gracefully."""
        corpus = pl.DataFrame({
            "doc_id": ["empty", "minimal", "normal"],
            "text": ["", "Hi.", "This is a normal sentence with content."]
        })

        # Should complete without crashing
        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should have results for non-empty documents only (empty texts are filtered out)
        assert features.shape[0] == 2

        # Normal text should have more features than empty/minimal
        normal_doc = features.filter(pl.col("doc_id") == "normal")
        feature_cols = [col for col in features.columns
                        if col.startswith("f_")]

        normal_feature_sum = sum(
            normal_doc[col].item() or 0 for col in feature_cols[:10]
        )

        # Normal document should have some features
        assert normal_feature_sum > 0


class TestErrorRecovery:
    """Test error handling and recovery in integration scenarios."""

    def test_malformed_text_handling(self, nlp_model):
        """Test handling of malformed or unusual text."""
        corpus = pl.DataFrame({
            "doc_id": ["special1", "special2", "special3"],
            "text": [
                "!@#$%^&*()",  # Only punctuation
                "word" * 1000,  # Very repetitive
                "Mix3d c4s3s w1th numb3rs"  # Mixed alphanumeric
            ]
        })

        # Should handle without crashing
        try:
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)

            assert features.shape[0] == 3
            assert "doc_id" in features.columns

        except Exception as e:
            # If it fails, should be a reasonable error
            assert isinstance(e, (ValueError, TypeError, RuntimeError))

    def test_very_long_text_handling(self, nlp_model):
        """Test handling of very long texts."""
        # Create a long text (but not too long to slow down tests)
        long_text = "This is a test sentence. " * 1000  # ~25k chars

        corpus = pl.DataFrame({
            "doc_id": ["long_doc"],
            "text": [long_text]
        })

        # Should handle long texts (with document splitting)
        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        assert features.shape[0] == 1
        assert "long_doc" in features["doc_id"].to_list()

    def test_unicode_handling(self, nlp_model):
        """Test handling of unicode characters."""
        corpus = pl.DataFrame({
            "doc_id": ["unicode_test"],
            "text": ["Café naïve résumé façade coöperation"]
        })

        # Should handle unicode gracefully
        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        assert features.shape[0] == 1
        assert isinstance(features["doc_id"].item(), str)


class TestVersionConsistency:
    """Test consistency across different usage patterns."""

    def test_batch_vs_single_processing(self, nlp_model):
        """Test batch processing gives same results as single processing."""
        texts = [
            "The first document has some content.",
            "The second document has different content.",
            "The third document has more content."
        ]

        # Process as batch
        batch_corpus = pl.DataFrame({
            "doc_id": [f"doc_{i}" for i in range(3)],
            "text": texts
        })
        batch_parsed = spacy_parse(batch_corpus, nlp_model, n_process=1,
                                   batch_size=3)
        batch_features = biber(batch_parsed, normalize=False, force_ttr=True)

        # Process individually and combine
        individual_features = []
        for i, text in enumerate(texts):
            single_corpus = pl.DataFrame({
                "doc_id": [f"doc_{i}"],
                "text": [text]
            })
            single_parsed = spacy_parse(single_corpus, nlp_model, n_process=1)
            single_feature = biber(single_parsed, normalize=False,
                                   force_ttr=True)
            individual_features.append(single_feature)

        combined_features = pl.concat(individual_features)

        # Results should be comparable (allowing for minor differences)
        assert batch_features.shape == combined_features.shape

        # doc_ids should match
        batch_ids = set(batch_features["doc_id"].to_list())
        combined_ids = set(combined_features["doc_id"].to_list())
        assert batch_ids == combined_ids
