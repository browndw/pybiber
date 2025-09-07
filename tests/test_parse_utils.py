"""
Test suite for pybiber.parse_utils module.
"""

import pytest
import tempfile
from pathlib import Path
import polars as pl
from pybiber.parse_utils import (
    get_text_paths,
    readtext,
    corpus_from_folder,
    spacy_parse,
    get_noun_phrases,
    TextPreprocessor,
    TextChunker
)


class TestTextProcessingUtilities:
    """Test basic text processing utility functions."""

    def test_str_squish(self):
        """Test _str_squish removes extra whitespace."""
        assert TextPreprocessor.squish_whitespace("  hello   world  \n  ") == "hello world"  # noqa: E501
        assert TextPreprocessor.squish_whitespace("single") == "single"
        assert TextPreprocessor.squish_whitespace("") == ""
        assert TextPreprocessor.squish_whitespace("  \n\t  ") == ""

    def test_replace_curly_quotes(self):
        """Test _replace_curly_quotes converts unicode quotes."""
        # Test left and right double quotes
        text = u"He said \u201chello\u201d and \u2018world\u2019."
        expected = 'He said "hello" and \'world\'.'
        assert TextPreprocessor.replace_curly_quotes(text) == expected

        # Test just double quotes
        text = u"\u201chello\u201d"
        expected = '"hello"'
        assert TextPreprocessor.replace_curly_quotes(text) == expected

    def test_split_docs(self):
        """Test _split_docs splits documents appropriately."""
        text = "First sentence. Second sentence! Third question? Fourth."
        chunks = TextChunker.split_document(text, 2)
        assert len(chunks) == 2
        assert all(isinstance(chunk, str) for chunk in chunks)

        # Test with single chunk
        chunks = TextChunker.split_document(text, 1)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_pre_process_corpus(self):
        """Test _pre_process_corpus cleans text appropriately."""
        df = pl.DataFrame({
            "doc_id": ["test1", "test2"],
            "text": ["  Hello   world  ", "Café naïve résumé"]
        })

        processed = TextPreprocessor.preprocess_corpus(df)

        # Check whitespace is cleaned
        assert "Hello world" in processed.get_column("text")[0]

        # Check unicode normalization occurs
        text2 = processed.get_column("text")[1]
        assert isinstance(text2, str)


class TestFileOperations:
    """Test file reading and path operations."""

    def test_get_text_paths_non_recursive(self):
        """Test get_text_paths finds .txt files non-recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.txt").write_text("content1")
            (Path(tmpdir) / "test2.txt").write_text("content2")
            (Path(tmpdir) / "not_text.py").write_text("python code")

            # Create subdirectory with txt file
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "test3.txt").write_text("content3")

            paths = get_text_paths(tmpdir, recursive=False)

            # Should find 2 files in main directory
            assert len(paths) == 2
            assert all(path.endswith(".txt") for path in paths)
            assert all("subdir" not in path for path in paths)

    def test_get_text_paths_recursive(self):
        """Test get_text_paths finds .txt files recursively."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.txt").write_text("content1")

            # Create subdirectory with txt file
            subdir = Path(tmpdir) / "subdir"
            subdir.mkdir()
            (subdir / "test2.txt").write_text("content2")

            paths = get_text_paths(tmpdir, recursive=True)

            # Should find files in both directories
            assert len(paths) == 2
            assert any("subdir" in path for path in paths)

    def test_readtext(self):
        """Test readtext creates proper DataFrame from text files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "doc1.txt"
            file2 = Path(tmpdir) / "doc2.txt"
            file1.write_text("First document content.")
            file2.write_text("Second document content.")

            paths = [str(file1), str(file2)]
            df = readtext(paths)

            # Check structure
            assert df.shape[0] == 2
            assert df.columns == ["doc_id", "text"]

            # Check content
            doc_ids = df.get_column("doc_id").to_list()
            assert "doc1.txt" in doc_ids
            assert "doc2.txt" in doc_ids

            texts = df.get_column("text").to_list()
            assert "First document content." in texts
            assert "Second document content." in texts

    def test_corpus_from_folder(self):
        """Test corpus_from_folder creates corpus from directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            (Path(tmpdir) / "test1.txt").write_text("Content of test1.")
            (Path(tmpdir) / "test2.txt").write_text("Content of test2.")

            df = corpus_from_folder(tmpdir)

            assert df.shape[0] == 2
            assert df.columns == ["doc_id", "text"]
            expected_ids = ["test1.txt", "test2.txt"]
            assert df.get_column("doc_id").to_list() == expected_ids

    def test_corpus_from_folder_empty_directory(self):
        """Test corpus_from_folder raises error for empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create only non-txt files
            (Path(tmpdir) / "test.py").write_text("python code")

            with pytest.raises(ValueError, match="No text files found"):
                corpus_from_folder(tmpdir)


class TestSpacyIntegration:
    """Test spaCy-dependent functions."""

    def test_spacy_parse_validation(self, nlp_model):
        """Test spacy_parse validates input correctly."""
        # Test invalid DataFrame structure
        invalid_df = pl.DataFrame({"wrong": ["col"]})

        with pytest.raises(ValueError, match="Invalid DataFrame"):
            spacy_parse(invalid_df, nlp_model)

    def test_spacy_parse_basic(self, nlp_model, sample_corpus):
        """Test basic spacy_parse functionality."""
        # Take just first few samples to speed up test
        small_corpus = sample_corpus.head(3)

        parsed = spacy_parse(small_corpus, nlp_model, n_process=1,
                             batch_size=5)

        # Check structure
        expected_cols = [
            "doc_id", "sentence_id", "token_id", "token", "lemma",
            "pos", "tag", "head_token_id", "dep_rel"
        ]
        assert parsed.columns == expected_cols

        # Check we have tokens for each document
        doc_ids = parsed.get_column("doc_id").unique().to_list()
        assert len(doc_ids) == 3

        # Check data types
        assert parsed.get_column("sentence_id").dtype == pl.UInt32
        assert parsed.get_column("token_id").dtype == pl.Int64

    def test_get_noun_phrases_validation(self, nlp_model):
        """Test get_noun_phrases validates input correctly."""
        invalid_df = pl.DataFrame({"wrong": ["col"]})

        with pytest.raises(ValueError, match="Invalid DataFrame"):
            get_noun_phrases(invalid_df, nlp_model)

    def test_get_noun_phrases_basic(self, nlp_model):
        """Test basic get_noun_phrases functionality."""
        corpus = pl.DataFrame({
            "doc_id": ["test1"],
            "text": ["The big red house stood on the hill."]
        })

        phrases = get_noun_phrases(corpus, nlp_model, n_process=1,
                                   batch_size=5)

        # Check structure
        expected_cols = [
            "doc_id", "phrase_text", "phrase_tags", "phrase_len",
            "root_text", "root_tag", "root_idx", "start_idx", "end_idx"
        ]
        assert phrases.columns == expected_cols

        # Should find noun phrases like "The big red house" and "the hill"
        assert phrases.shape[0] >= 1
        phrase_texts = phrases.get_column("phrase_text").to_list()
        assert any("house" in phrase.lower() for phrase in phrase_texts)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_text_handling(self, nlp_model):
        """Test handling of empty or whitespace-only texts."""
        corpus = pl.DataFrame({
            "doc_id": ["empty1", "empty2", "whitespace"],
            "text": ["", "   ", "  \n\t  "]
        })

        parsed = spacy_parse(corpus, nlp_model, n_process=1)

        # Should handle empty texts gracefully
        assert isinstance(parsed, pl.DataFrame)
        # Might have minimal tokens (like added spaces)

    def test_long_document_splitting(self, nlp_model):
        """Test that very long documents are split appropriately."""
        # Create a document longer than 500000 characters
        long_text = "This is a test sentence. " * 25000  # ~625k chars

        corpus = pl.DataFrame({
            "doc_id": ["long_doc"],
            "text": [long_text]
        })

        parsed = spacy_parse(corpus, nlp_model, n_process=1, batch_size=5)

        # Should successfully parse without memory issues
        assert parsed.shape[0] > 0
        assert "long_doc" in parsed.get_column("doc_id").unique().to_list()

    def test_special_characters(self, nlp_model):
        """Test handling of special characters and unicode."""
        corpus = pl.DataFrame({
            "doc_id": ["special"],
            "text": ["Hello! This has émojis 😀 and symbols @#$%^&*()"]
        })

        parsed = spacy_parse(corpus, nlp_model, n_process=1)

        # Should handle special characters without errors
        assert parsed.shape[0] > 0
        tokens = parsed.get_column("token").to_list()
        assert len(tokens) > 0
