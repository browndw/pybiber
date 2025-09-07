"""
Enhanced error handling and validation utilities for pybiber.

This module provides comprehensive error classes, validation functions,
and user-friendly error messages with actionable suggestions for common
corpus processing issues.

Exception Classes:
    PybiberError: Base exception for all pybiber errors
    CorpusValidationError: Corpus structure and content validation
    ModelValidationError: spaCy model compatibility validation
    DataFormatError: Data format and schema validation
    ParameterValidationError: Function parameter validation
    FileSystemError: File and directory access validation
    ValidationWarning: Non-fatal validation warnings
    PerformanceWarning: Performance-related warnings

Validation Functions:
    validate_corpus_dataframe: Validate corpus DataFrame structure
    validate_tokens_dataframe: Validate processed tokens structure
    validate_directory_path: Validate directory existence and access
    validate_text_files_in_directory: Validate text file availability
    warn_about_performance_settings: Performance optimization warnings
    suggest_alternatives_for_empty_results: Alternative suggestions

Example:
    Using validation functions::

        import polars as pl
        from pybiber.validation import validate_corpus_dataframe

        corpus = pl.DataFrame({
            'doc_id': ['doc1', 'doc2'],
            'text': ['First text', 'Second text']
        })

        try:
            validate_corpus_dataframe(corpus)
            print("Corpus is valid!")
        except CorpusValidationError as e:
            print(f"Validation failed: {e}")

    Custom error handling::

        from pybiber.validation import PybiberError
        from pybiber.parse_utils import CorpusProcessor

        try:
            # Your pybiber operations
            result = CorpusProcessor().process_corpus(...)
        except PybiberError as e:
            # Handle all pybiber-specific errors
            print(f"Pybiber error: {e}")

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

import warnings
from typing import List, Union
from pathlib import Path
import polars as pl
from collections import OrderedDict
from spacy.language import Language
from .config import CONFIG


class PybiberError(Exception):
    """
    Base exception class for all pybiber errors.

    This is the parent class for all custom exceptions in the pybiber
    package. It allows for catching all pybiber-specific errors with
    a single except clause.

    Example:
        Catch all pybiber errors::

            from pybiber.validation import PybiberError

            try:
                # Your pybiber operations
                result = CorpusProcessor().process_corpus(...)
            except PybiberError as e:
                print(f"Pybiber processing failed: {e}")
    """

    pass


class CorpusValidationError(PybiberError):
    """
    Raised when corpus DataFrame validation fails.

    This exception is raised when the input corpus doesn't meet the
    required structure or content standards for processing.

    Common causes:
        - Missing required columns ('doc_id', 'text')
        - Empty corpus or missing data
        - Invalid data types
        - Duplicate document IDs

    Example:
        >>> import polars as pl
        >>> from pybiber.validation import validate_corpus_dataframe
        >>>
        >>> # This will raise CorpusValidationError
        >>> invalid_corpus = pl.DataFrame({'wrong_col': ['data']})
        >>> validate_corpus_dataframe(invalid_corpus)
        CorpusValidationError: Corpus DataFrame has insufficient columns...
    """

    pass


class ModelValidationError(PybiberError):
    """Raised when spaCy model validation fails."""

    pass


class DataFormatError(PybiberError):
    """Raised when data format is incorrect."""

    pass


class ParameterValidationError(PybiberError):
    """Raised when function parameters are invalid."""

    pass


class FileSystemError(PybiberError):
    """Raised when file system operations fail."""

    pass


class ValidationWarning(UserWarning):
    """Warning for potentially problematic but non-fatal issues."""

    pass


class PerformanceWarning(UserWarning):
    """Warning for performance-related issues."""

    pass


def validate_corpus_dataframe(corp: pl.DataFrame, context: str = "") -> None:
    """
    Comprehensive validation of corpus DataFrame.

    :param corp: DataFrame to validate
    :param context: Context for error messages
        (e.g., "in CorpusProcessor().process_corpus()")
    """
    if corp is None:
        raise CorpusValidationError(
            f"Corpus DataFrame is None {context}. "
            "Please provide a valid DataFrame with 'doc_id' and 'text' "
            "columns."
        )

    if corp.height == 0:
        raise CorpusValidationError(
            f"Corpus DataFrame is empty {context}. "
            "Please provide a DataFrame with at least one row of data."
        )

    # Check schema
    expected_schema = OrderedDict([("doc_id", pl.String), ("text", pl.String)])

    actual_schema = corp.collect_schema()

    if len(actual_schema) < 2:
        raise CorpusValidationError(
            f"Corpus DataFrame has insufficient columns {context}. "
            f"Expected 2 columns (doc_id, text), found {len(actual_schema)}. "
            "Please ensure your DataFrame has 'doc_id' and 'text' columns."
        )

    if actual_schema != expected_schema:
        missing_cols = set(expected_schema.keys()) - set(actual_schema.keys())
        extra_cols = set(actual_schema.keys()) - set(expected_schema.keys())
        wrong_types = {
            col: (expected_schema[col], actual_schema.get(col))
            for col in expected_schema
            if (col in actual_schema and actual_schema[col] != expected_schema[col])  # noqa: E501
        }

        error_msg = f"Invalid corpus DataFrame schema {context}.\n"

        if missing_cols:
            error_msg += f"Missing columns: {', '.join(missing_cols)}\n"

        if extra_cols:
            error_msg += f"Unexpected columns: {', '.join(extra_cols)}\n"

        if wrong_types:
            error_msg += "Incorrect column types:\n"
            for col, (expected, actual) in wrong_types.items():
                error_msg += f"  {col}: expected {expected}, got {actual}\n"

        error_msg += "\nExpected schema: doc_id (String), text (String)"

        raise CorpusValidationError(error_msg)

    # Check for common data issues
    null_doc_ids = corp.filter(pl.col("doc_id").is_null()).height
    if null_doc_ids > 0:
        raise CorpusValidationError(
            f"Found {null_doc_ids} rows with null doc_id {context}. "
            "All documents must have a valid document ID."
        )

    duplicate_ids = corp.group_by("doc_id").len().filter(pl.col("len") > 1)
    if duplicate_ids.height > 0:
        duplicates = duplicate_ids.get_column("doc_id").to_list()[:5]
        error_msg = (
            "Found duplicate document IDs "
            f"{context}: {', '.join(duplicates)}"
        )
        if len(duplicates) == 5:
            error_msg += "... (and more)"
        error_msg += "\nEach document must have a unique ID."
        raise CorpusValidationError(error_msg)

    # Warnings for potentially problematic data
    null_texts = corp.filter(pl.col("text").is_null()).height
    if null_texts > 0:
        warnings.warn(
            f"Found {null_texts} documents with null text {context}. "
            "These will be filtered out during processing.",
            ValidationWarning,
        )

    empty_texts = corp.filter(
        pl.col("text").is_not_null() & (pl.col("text").str.strip_chars() == "")
    ).height
    if empty_texts > 0:
        warnings.warn(
            f"Found {empty_texts} documents with empty text {context}. "
            "These may not produce meaningful analysis results.",
            ValidationWarning,
        )

    # Performance warnings
    if corp.height > 10000:
        warnings.warn(
            f"Large corpus detected ({corp.height:,} documents) {context}. "
            "Consider enabling memory optimization or processing in batches "
            "for better performance.",
            PerformanceWarning,
        )

    very_long_texts = corp.filter(
        pl.col("text").str.len_chars() > 100000
    ).height  # noqa: E501
    if very_long_texts > 0:
        warnings.warn(
            f"Found {very_long_texts} very long documents (>100k characters) {context}. "  # noqa: E501
            "These may be automatically chunked during processing.",
            PerformanceWarning,
        )


def validate_tokens_dataframe(
    tokens_table: pl.DataFrame, context: str = ""
) -> None:
    """
    Comprehensive validation of tokens DataFrame.

    :param tokens_table: DataFrame to validate
    :param context: Context for error messages
    """
    if tokens_table is None:
        raise DataFormatError(
            f"Tokens DataFrame is None {context}. "
            "Expected a DataFrame produced by "
            "CorpusProcessor().process_corpus()."
        )

    if tokens_table.height == 0:
        raise DataFormatError(
            f"Tokens DataFrame is empty {context}. "
            "Please provide a DataFrame with token data."
        )

    # Expected schema for token-level output produced by
    # CorpusProcessor().process_corpus()
    expected_schema = OrderedDict(
        [
            ("doc_id", pl.String),
            ("sentence_id", pl.UInt32),
            ("token_id", pl.Int64),
            ("token", pl.String),
            ("lemma", pl.String),
            ("pos", pl.String),
            ("tag", pl.String),
            ("head_token_id", pl.Int64),
            ("dep_rel", pl.String),
        ]
    )

    actual_schema = tokens_table.collect_schema()

    if actual_schema != expected_schema:
        missing_cols = set(expected_schema.keys()) - set(actual_schema.keys())
        extra_cols = set(actual_schema.keys()) - set(expected_schema.keys())
        wrong_types = {
            col: (expected_schema[col], actual_schema.get(col))
            for col in expected_schema
            if (
                col in actual_schema
                and actual_schema[col] != expected_schema[col]
            )
        }

        error_msg = f"Invalid tokens DataFrame schema {context}.\n"

        if missing_cols:
            error_msg += (
                "Missing columns: "
                + ", ".join(sorted(missing_cols))
                + "\n"
            )
        if extra_cols:
            error_msg += (
                "Unexpected columns: "
                + ", ".join(sorted(extra_cols))
                + "\n"
            )
        if wrong_types:
            error_msg += "Incorrect column types:\n"
            for col, (expected, actual) in wrong_types.items():
                error_msg += f"  {col}: expected {expected}, got {actual}\n"

        error_msg += (
            "\nExpected a DataFrame produced by "
            "CorpusProcessor().process_corpus() with columns in this order: "
            "doc_id (String), sentence_id (UInt32), token_id (Int64), "
            "token (String), lemma (String), pos (String), tag (String), "
            "head_token_id (Int64), dep_rel (String)"
        )

        if "doc_id" not in actual_schema:
            error_msg += (
                "\n\nTip: If you have a raw text corpus, "
                "use CorpusProcessor().process_corpus() first."
            )

        raise DataFormatError(error_msg)


def validate_spacy_model(nlp_model: Language, context: str = "") -> None:
    """Validate that a spaCy model is compatible with pybiber.

    Ensures the model matches the expected prefix (e.g., en_core_web_) and
    raises ModelValidationError with a helpful message if not.

    :param nlp_model: spaCy Language model instance
    :param context: Optional context string for error messages
    :raises ModelValidationError: If the model name does not match expectations
    """
    model_name = nlp_model.lang + "_" + nlp_model.meta.get("name", "")
    if not model_name.startswith(CONFIG.EXPECTED_MODEL_PREFIX):
        raise ModelValidationError(
            "Invalid spaCy model "
            + context
            + ". Expected a model starting with "
            + f"'{CONFIG.EXPECTED_MODEL_PREFIX}'. "
            + "For information and instructions see: https://spacy.io/models"
        )


def validate_directory_path(
        directory: Union[str, Path], context: str = ""
) -> Path:
    """
    Validate directory path and provide helpful error messages.

    :param directory: Directory path to validate
    :param context: Context for error messages
    :return: Validated Path object
    """
    if directory is None:
        raise FileSystemError(
            f"Directory path is None {context}. "
            "Please provide a valid directory path."
        )

    if isinstance(directory, str) and directory.strip() == "":
        raise FileSystemError(
            f"Directory path is empty {context}. "
            "Please provide a valid directory path."
        )

    path = Path(directory)

    if not path.exists():
        # Try to provide helpful suggestions
        parent = path.parent
        if parent.exists():
            similar_dirs = [
                d.name
                for d in parent.iterdir()
                if d.is_dir()
                and d.name.lower().startswith(path.name[:3].lower())  # noqa: E501
            ]
            suggestion = ""
            if similar_dirs:
                suggestion = f"\nDid you mean: {', '.join(similar_dirs[:3])}?"
        else:
            suggestion = f"\nParent directory also doesn't exist: {parent}"

        raise FileSystemError(
            f"Directory does not exist {context}: {path}{suggestion}\n"
            "Please check the path and ensure the directory exists."
        )

    if not path.is_dir():
        raise FileSystemError(
            f"Path is not a directory {context}: {path}\n"
            "Please provide a path to a directory, not a file."
        )

    return path


def validate_text_files_in_directory(
        directory: Path, context: str = ""
) -> List[Path]:
    """
    Validate that directory contains text files.

    :param directory: Directory to check
    :param context: Context for error messages
    :return: List of text file paths
    """
    text_files = list(directory.glob("*.txt"))

    if len(text_files) == 0:
        # Check for other text-like files
        other_files = list(directory.glob("*"))
        text_like = [
            f
            for f in other_files
            if f.suffix.lower() in [".text", ".doc", ".docx", ".pdf", ".md"]
        ]

        error_msg = (
            f"No .txt files found in directory {context}: {directory}\n"  # noqa: E501
        )

        if len(other_files) == 0:
            error_msg += "The directory is empty."
        elif text_like:
            error_msg += (
                f"Found {len(text_like)} files with text-like extensions: "
                f"{', '.join(f.suffix for f in text_like[:5])}\n"
                "Only .txt files are supported. Consider converting your files to .txt format."  # noqa: E501
            )
        else:
            error_msg += (
                "Found "
                f"{len(other_files)} files, but none with .txt extension.\n"
                "Only .txt files are supported."
            )

        raise FileSystemError(error_msg)

    # Check for common issues
    large_files = [
        f for f in text_files if f.stat().st_size > 10 * 1024 * 1024
    ]  # 10MB  # noqa: E501
    if large_files:
        warnings.warn(
            f"Found {len(large_files)} large files (>10MB) {context}. "
            "These may take longer to process or be automatically chunked.",
            PerformanceWarning,
        )

    return text_files


def warn_about_performance_settings():
    """
    Provide warnings about performance settings that might affect results.
    """
    from .config import CONFIG

    if not CONFIG.ENABLE_CACHING:
        warnings.warn(
            "Caching is disabled. Repeated operations will be slower. "
            "Consider enabling caching for better performance.",
            PerformanceWarning,
        )

    if CONFIG.MEMORY_EFFICIENT_MODE:
        warnings.warn(
            "Memory efficient mode is enabled. This may slow down processing "
            "but will use less memory for large datasets.",
            PerformanceWarning,
        )


def suggest_alternatives_for_empty_results(operation: str, **kwargs):
    """Provide suggestions when operations return empty results."""
    suggestions = []

    if operation == "ngrams":
        min_freq = kwargs.get("min_frequency", 10)
        if min_freq > 5:
            suggestions.append(
                f"Try reducing min_frequency (currently {min_freq})"
            )

        span = kwargs.get("span", 2)
        if span > 3:
            suggestions.append(f"Try using a smaller span (currently {span})")

    elif operation == "collocations":
        node_word = kwargs.get("node_word", "")
        if node_word:
            suggestions.append(f"Check if '{node_word}' exists in your corpus")
            suggestions.append("Try a more common word as the node")

    elif operation == "clusters":
        node_word = kwargs.get("node_word", "")
        span = kwargs.get("span", 2)
        if node_word and span > 2:
            suggestions.append("Try using span=2 for more results")

    if suggestions:
        warning_msg = f"No results found for {operation}. Suggestions:\n"
        warning_msg += "\n".join(f"- {s}" for s in suggestions)
        warnings.warn(warning_msg, ValidationWarning)
