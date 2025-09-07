"""
Feature-specific tests for pybiber, similar to the R package tests.

These tests verify that specific linguistic features are correctly
identified in test sentences.
"""

import polars as pl
from pybiber import spacy_parse, biber


class TestSpecificFeatures:
    """Test detection of specific Biber features in known text samples."""

    def test_present_tense_detection(self, nlp_model):
        """Test detection of present tense verbs."""
        text = "The quick brown fox jumps over the lazy dog."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "jumps" as present tense
        if "f_03_present_tense" in features.columns:
            present_count = features["f_03_present_tense"].item()
            assert present_count >= 1

    def test_past_tense_detection(self, nlp_model):
        """Test detection of past tense verbs."""
        text = "The cat walked slowly down the street."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "walked" as past tense
        if "f_01_past_tense" in features.columns:
            past_count = features["f_01_past_tense"].item()
            assert past_count >= 1

    def test_perfect_aspect_detection(self, nlp_model):
        """Test detection of perfect aspect verbs."""
        text = "I have written this sentence."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "have written" as perfect aspect
        if "f_02_perfect_aspect" in features.columns:
            perfect_count = features["f_02_perfect_aspect"].item()
            assert perfect_count >= 1

    def test_predicative_adjectives(self, nlp_model):
        """Test detection of predicative adjectives."""
        text = "The horse is big."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "big" as predicative adjective
        if "f_41_adj_pred" in features.columns:
            pred_adj_count = features["f_41_adj_pred"].item()
            assert pred_adj_count >= 1

    def test_demonstrative_pronouns(self, nlp_model):
        """Test detection of demonstrative pronouns."""
        text = "That is an example sentence."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "That" as demonstrative pronoun
        if "f_10_demonstrative_pronoun" in features.columns:
            demo_count = features["f_10_demonstrative_pronoun"].item()
            assert demo_count >= 1

    def test_wh_questions(self, nlp_model):
        """Test detection of WH-questions."""
        text = "When are you leaving?"
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "When" as WH-question word
        if "f_13_wh_question" in features.columns:
            wh_count = features["f_13_wh_question"].item()
            assert wh_count >= 1

    def test_existential_there(self, nlp_model):
        """Test detection of existential 'there'."""
        text = "There is a feature in this sentence."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect existential "There"
        if "f_20_existential_there" in features.columns:
            exist_count = features["f_20_existential_there"].item()
            assert exist_count >= 1

    def test_that_verb_complements(self, nlp_model):
        """Test detection of that-verb complements."""
        text = "I said that he went."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "said that" as that-verb complement
        if "f_21_that_verb_comp" in features.columns:
            that_verb_count = features["f_21_that_verb_comp"].item()
            assert that_verb_count >= 1

    def test_that_adj_complements(self, nlp_model):
        """Test detection of that-adjective complements."""
        text = "I'm glad that you like it."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "glad that" as that-adjective complement
        if "f_22_that_adj_comp" in features.columns:
            that_adj_count = features["f_22_that_adj_comp"].item()
            assert that_adj_count >= 1

    def test_present_participles(self, nlp_model):
        """Test detection of present participles."""
        text = "Stuffing his mouth with cookies, Joe ran out the door."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "Stuffing" as present participle
        if "f_25_present_participle" in features.columns:
            participle_count = features["f_25_present_participle"].item()
            assert participle_count >= 1

    def test_sentence_relatives(self, nlp_model):
        """Test detection of sentence relatives."""
        text = ("Bob likes fried mangoes, which is the most "
                "disgusting thing I've ever heard of.")
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "which" as sentence relative
        if "f_34_sentence_relatives" in features.columns:
            rel_count = features["f_34_sentence_relatives"].item()
            assert rel_count >= 1

    def test_agentless_passives(self, nlp_model):
        """Test detection of agentless passive constructions."""
        text = "The task was done."
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect "was done" as agentless passive
        if "f_17_agentless_passives" in features.columns:
            passive_count = features["f_17_agentless_passives"].item()
            assert passive_count >= 1

        # Should NOT detect by-passive
        if "f_18_by_passives" in features.columns:
            by_passive_count = features["f_18_by_passives"].item()
            assert by_passive_count == 0

    def test_multiple_features_in_complex_text(self, nlp_model):
        """Test detection of multiple features in a complex sentence."""
        text = ("The students have completed their assignments, "
                "which were quite challenging.")
        corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})

        parsed = spacy_parse(corpus, nlp_model, n_process=1)
        features = biber(parsed, normalize=False, force_ttr=True)

        # Should detect multiple features
        detected_features = []

        for col in features.columns:
            if col.startswith("f_") and features[col].item() > 0:
                detected_features.append(col)

        # Should detect at least a few features in this complex sentence
        assert len(detected_features) >= 2


class TestFeatureConsistency:
    """Test feature detection consistency across similar texts."""

    def test_similar_constructions_consistency(self, nlp_model):
        """Test that similar constructions are detected consistently."""
        texts = [
            "I think he went.",
            "She believes they left.",
            "We assume it worked."
        ]

        results = []
        for i, text in enumerate(texts):
            corpus = pl.DataFrame({"doc_id": [f"test_{i}"], "text": [text]})
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)
            results.append(features)

        # All should have similar pattern for that-deletion
        # (though detection may vary based on parsing)
        if "f_60_that_deletion" in results[0].columns:
            deletion_counts = [r["f_60_that_deletion"].item() for r in results]
            # At least some should be detected (allow for parsing variations)
            assert any(count > 0 for count in deletion_counts)

    def test_tense_consistency(self, nlp_model):
        """Test consistency in tense detection."""
        past_texts = ["He walked.", "She ran.", "They jumped."]
        present_texts = ["He walks.", "She runs.", "They jump."]

        past_counts = []
        present_counts = []

        for text in past_texts:
            corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)
            if "f_01_past_tense" in features.columns:
                past_counts.append(features["f_01_past_tense"].item())

        for text in present_texts:
            corpus = pl.DataFrame({"doc_id": ["test"], "text": [text]})
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)
            if "f_03_present_tense" in features.columns:
                present_counts.append(features["f_03_present_tense"].item())

        # Past tense texts should generally have past tense features
        if past_counts:
            assert any(count > 0 for count in past_counts)

        # Present tense texts should generally have present tense features
        if present_counts:
            assert any(count > 0 for count in present_counts)


class TestFeatureRobustness:
    """Test robustness of feature detection."""

    def test_punctuation_handling(self, nlp_model):
        """Test that punctuation doesn't interfere with feature detection."""
        texts = [
            "I think he went",
            "I think he went.",
            "I think he went!",
            "I think, he went."
        ]

        results = []
        for i, text in enumerate(texts):
            corpus = pl.DataFrame({"doc_id": [f"test_{i}"], "text": [text]})
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)
            results.append(features)

        # Feature detection should be robust to punctuation differences
        # (This is more of a smoke test - exact behavior may vary)
        for result in results:
            assert isinstance(result, pl.DataFrame)
            assert result.shape[0] == 1

    def test_capitalization_handling(self, nlp_model):
        """Test handling of different capitalization."""
        texts = [
            "The cat walked down the street.",
            "the cat walked down the street.",
            "THE CAT WALKED DOWN THE STREET."
        ]

        results = []
        for i, text in enumerate(texts):
            corpus = pl.DataFrame({"doc_id": [f"test_{i}"], "text": [text]})
            parsed = spacy_parse(corpus, nlp_model, n_process=1)
            features = biber(parsed, normalize=False, force_ttr=True)
            results.append(features)

        # Should handle different capitalizations
        for result in results:
            assert isinstance(result, pl.DataFrame)
            # Past tense should be detected regardless of capitalization
            if "f_01_past_tense" in result.columns:
                past_count = result["f_01_past_tense"].item()
                assert past_count >= 1
