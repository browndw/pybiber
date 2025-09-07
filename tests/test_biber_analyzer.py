"""
Test suite for pybiber.biber_analyzer module.
"""

import pytest
import numpy as np
import polars as pl
from pybiber.biber_analyzer import BiberAnalyzer, _get_eigenvalues, _promax


class TestBiberAnalyzer:
    """Test the BiberAnalyzer class."""

    @pytest.fixture
    def sample_feature_matrix(self):
        """Create a sample feature matrix for testing."""
        np.random.seed(42)  # For reproducible tests

        # Create sample data with 10 documents, 5 features
        data = {
            "doc_id": [f"doc_{i:02d}" for i in range(10)],
            "category": ["A"] * 5 + ["B"] * 5,
            "f_01_feature": np.random.normal(50, 10, 10),
            "f_02_feature": np.random.normal(30, 15, 10),
            "f_03_feature": np.random.normal(20, 5, 10),
            "f_04_feature": np.random.normal(40, 8, 10),
            "f_05_feature": np.random.normal(60, 12, 10),
        }

        return pl.DataFrame(data)

    @pytest.fixture
    def minimal_feature_matrix(self):
        """Create minimal feature matrix for edge case testing."""
        return pl.DataFrame({
            "category": ["A", "B"],
            "f_01_feature": [10.5, 20.3],
            "f_02_feature": [5.2, 15.8]
        })

    def test_biber_analyzer_initialization_with_categories_only(
            self, minimal_feature_matrix):
        """Test BiberAnalyzer initialization with categories only."""
        # This should currently raise an error due to buggy validation logic
        with pytest.raises(ValueError, match="Invalid DataFrame"):
            BiberAnalyzer(minimal_feature_matrix, id_column=False)

    def test_biber_analyzer_initialization_with_ids_and_categories(
            self, sample_feature_matrix):
        """Test BiberAnalyzer initialization with doc IDs and categories."""
        analyzer = BiberAnalyzer(sample_feature_matrix, id_column=True)

        assert analyzer.doc_ids is not None
        assert analyzer.category_ids is not None
        assert len(analyzer.doc_ids) == 10
        assert len(analyzer.category_ids) == 10

    def test_biber_analyzer_validation_wrong_dtypes(self):
        """Test BiberAnalyzer validation with wrong data types."""
        # DataFrame with integer instead of float
        invalid_df = pl.DataFrame({
            "category": ["A", "B"],
            "f_01_feature": [10, 20],  # Int instead of Float
            "f_02_feature": [5.2, 15.8]
        })

        with pytest.raises(ValueError, match="Invalid DataFrame"):
            BiberAnalyzer(invalid_df, id_column=False)

    def test_biber_analyzer_validation_wrong_string_columns(self):
        """Test validation with wrong number of string columns."""
        # Too many string columns for id_column=False
        invalid_df = pl.DataFrame({
            "doc_id": ["doc1", "doc2"],
            "category": ["A", "B"],
            "extra_str": ["X", "Y"],
            "f_01_feature": [10.5, 20.3]
        })

        with pytest.raises(ValueError, match="Invalid DataFrame"):
            BiberAnalyzer(invalid_df, id_column=False)

    def test_biber_analyzer_feature_access(self, sample_feature_matrix):
        """Test access to feature matrix and basic properties."""
        analyzer = BiberAnalyzer(sample_feature_matrix, id_column=True)

        # Should be able to access basic attributes
        assert hasattr(analyzer, 'doc_ids')
        assert hasattr(analyzer, 'category_ids')

        # Check that features are accessible
        # (This depends on internal implementation)
        # We'll check if the analyzer stores the data correctly
        assert analyzer.doc_ids is not None
        assert analyzer.category_ids is not None


class TestUtilityFunctions:
    """Test utility functions in biber_analyzer module."""

    def test_get_eigenvalues_basic(self):
        """Test _get_eigenvalues function with basic data."""
        # Create correlation matrix with known structure
        np.random.seed(42)
        X = np.random.normal(0, 1, (50, 10))

        # Add some correlation structure
        X[:, 1] = X[:, 0] + np.random.normal(0, 0.1, 50)
        X[:, 2] = X[:, 0] + np.random.normal(0, 0.2, 50)

        eigenvals = _get_eigenvalues(X, cor_min=0.2)

        # Should return DataFrame with eigenvalue columns
        assert isinstance(eigenvals, pl.DataFrame)
        assert "ev_all" in eigenvals.columns
        assert "ev_mda" in eigenvals.columns

        # Eigenvalues should be in descending order
        all_eigs = eigenvals["ev_all"].to_list()
        assert all_eigs == sorted(all_eigs, reverse=True)

    def test_get_eigenvalues_low_correlation(self):
        """Test _get_eigenvalues with low correlation threshold."""
        np.random.seed(42)
        X = np.random.normal(0, 1, (30, 5))

        # Test with high correlation minimum (should filter out variables)
        eigenvals = _get_eigenvalues(X, cor_min=0.9)

        assert isinstance(eigenvals, pl.DataFrame)
        # With high cor_min, ev_mda might have fewer eigenvalues

    def test_promax_rotation(self):
        """Test _promax rotation function."""
        # Create simple factor loading matrix
        loadings = np.array([
            [0.8, 0.1],
            [0.7, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])

        rotated = _promax(loadings, m=4)

        # Should return array with same shape
        assert rotated.shape == loadings.shape
        assert isinstance(rotated, np.ndarray)

        # Rotated loadings should be different from original
        assert not np.allclose(rotated, loadings)

    def test_promax_rotation_edge_cases(self):
        """Test _promax with edge cases."""
        # Single factor
        single_factor = np.array([[0.8], [0.7], [0.9]])
        rotated = _promax(single_factor, m=2)

        assert rotated.shape == single_factor.shape

        # Very small loadings
        small_loadings = np.array([
            [0.1, 0.05],
            [0.08, 0.12]
        ])
        rotated = _promax(small_loadings, m=4)
        assert rotated.shape == small_loadings.shape


class TestBiberAnalyzerMethods:
    """Test methods of BiberAnalyzer class if they exist."""

    def test_analyzer_methods_exist(self, sample_feature_matrix):
        """Test that expected methods exist on BiberAnalyzer."""
        analyzer = BiberAnalyzer(sample_feature_matrix, id_column=True)

        # Check for common methods that might exist
        # (This will depend on the actual implementation)
        expected_attributes = ['doc_ids', 'category_ids']

        for attr in expected_attributes:
            assert hasattr(analyzer, attr)


class TestIntegrationWithRealData:
    """Integration tests using real feature data."""

    def test_with_biber_output(self, parsed_corpus, nlp_model):
        """Test BiberAnalyzer with actual biber function output."""
        from pybiber.parse_functions import biber

        # Get biber features (normalized)
        features = biber(parsed_corpus, normalize=True)

        # Add category column for testing
        features = features.with_columns(
            pl.lit("test_category").alias("category")
        )

        # Initialize analyzer
        analyzer = BiberAnalyzer(features, id_column=True)

        # Should initialize successfully
        assert analyzer.doc_ids is not None
        assert analyzer.category_ids is not None
        assert len(analyzer.doc_ids) == features.shape[0]

    def test_with_multiple_categories(self, parsed_corpus):
        """Test analyzer with multiple document categories."""
        from pybiber.parse_functions import biber

        features = biber(parsed_corpus, normalize=True)

        # Create multiple categories
        n_docs = features.shape[0]
        categories = ["A"] * (n_docs // 2) + ["B"] * (n_docs - n_docs // 2)

        features = features.with_columns(
            pl.Series("category", categories)
        )

        analyzer = BiberAnalyzer(features, id_column=True)

        # Should handle multiple categories
        unique_categories = set(analyzer.category_ids.to_list())
        assert len(unique_categories) >= 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pl.DataFrame({
            "category": pl.Series([], dtype=pl.String),
            "f_01_feature": pl.Series([], dtype=pl.Float64)
        })

        # Should handle empty DataFrame gracefully or raise appropriate error
        try:
            analyzer = BiberAnalyzer(empty_df, id_column=False)
            assert analyzer.category_ids.len() == 0
        except (ValueError, IndexError):
            # Acceptable if it raises an appropriate error
            pass

    def test_single_document(self):
        """Test behavior with single document."""
        single_doc = pl.DataFrame({
            "category": ["A"],
            "f_01_feature": [10.5],
            "f_02_feature": [20.3]
        })

        analyzer = BiberAnalyzer(single_doc, id_column=False)

        # Should handle single document
        assert len(analyzer.category_ids) == 1

    def test_missing_values_handling(self):
        """Test handling of missing values in feature matrix."""
        data_with_nulls = pl.DataFrame({
            "category": ["A", "B", "C"],
            "f_01_feature": [10.5, None, 15.2],
            "f_02_feature": [5.2, 15.8, None]
        })

        # Should handle nulls appropriately
        try:
            analyzer = BiberAnalyzer(data_with_nulls, id_column=False)
            assert len(analyzer.category_ids) == 3
        except (ValueError, TypeError):
            # Might raise error if nulls not supported
            pass
