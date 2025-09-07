"""
Utilities for processessing text data using a spaCy instance.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import os
import math
import unicodedata
import warnings
from pathlib import Path
from typing import List, Tuple, Dict, Any

import polars as pl
from spacy.tokens import Doc
from spacy.language import Language
from spacy.util import filter_spans

from .config import CONFIG, PATTERNS
from .performance import PerformanceMonitor, ProgressTracker, MemoryOptimizer
from .validation import (
    ModelValidationError,
    CorpusValidationError,
    validate_spacy_model,
    validate_corpus_dataframe
)


class TextPreprocessor:
    """
    Handles text cleaning and preprocessing for NLP analysis.

    This class provides various text normalization and cleaning methods
    to prepare raw text for spaCy processing. All methods can be used
    independently or combined through the main preprocessing pipeline.

    Example:
        Use individual preprocessing methods::

            from pybiber.processors import TextPreprocessor

            preprocessor = TextPreprocessor()

            # Clean whitespace
            clean_text = preprocessor.squish_whitespace("Hello    world\\n\\n")
            # Result: "Hello world"

            # Fix curly quotes
            fixed_quotes = preprocessor.replace_curly_quotes("He said "hello"")
            # Result: 'He said "hello"'

        Use the complete preprocessing pipeline::

            import polars as pl

            corpus = pl.DataFrame({
                'doc_id': ['doc1'],
                'text': ['He said "hello"    with   extra spaces']
            })

            preprocessor = TextPreprocessor()
            clean_corpus = preprocessor.preprocess_corpus(corpus)
    """

    @staticmethod
    def squish_whitespace(text: str) -> str:
        """
        Remove extra spaces, returns, tabs, and other whitespace from text.

        Normalizes all whitespace sequences to single spaces and strips
        leading/trailing whitespace.

        Args:
            text: Input text that may contain irregular whitespace.

        Returns:
            Cleaned text with normalized whitespace.

        Example:
            >>> TextPreprocessor.squish_whitespace("Hello    world\\n\\ttest")
            'Hello world test'
        """
        return " ".join(text.split())

    @staticmethod
    def replace_curly_quotes(text: str) -> str:
        """
        Replace curly/smart quotes with straight ASCII quotes.

        Converts Unicode curly quotes (left and right, single and double)
        to standard ASCII quote characters for consistent processing.

        Args:
            text: Input text that may contain curly quotes.

        Returns:
            Text with curly quotes replaced by straight quotes.

        Example:
            >>> TextPreprocessor.replace_curly_quotes("He said "hello"")
            'He said "hello"'
        """
        replacements = {
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201C": '"',  # Left double quote
            "\u201D": '"',  # Right double quote
        }

        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """
        Normalize unicode characters and convert to ASCII when possible.

        Applies NFKD (compatibility decomposition) normalization and
        converts to ASCII, removing characters that can't be represented.
        This helps standardize text from different sources.

        Args:
            text: Input text that may contain non-ASCII unicode characters.

        Returns:
            ASCII-normalized text.

        Example:
            >>> TextPreprocessor.normalize_unicode("café naïve")
            'cafe naive'
        """
        return (
            unicodedata.normalize("NFKD", text)
            .encode("ascii", errors="ignore")
            .decode("utf-8")
        )

    @staticmethod
    def preprocess_corpus(corp: pl.DataFrame) -> pl.DataFrame:
        """
        Apply all preprocessing steps to a corpus DataFrame.

        Runs the complete preprocessing pipeline on the 'text' column
        of a corpus DataFrame, applying all cleaning and normalization
        steps in the optimal order.

        Args:
            corp: A polars DataFrame with 'doc_id' and 'text' columns.

        Returns:
            DataFrame with preprocessed text column.

        Example:
            >>> import polars as pl
            >>> corpus = pl.DataFrame({
            ...     'doc_id': ['doc1'],
            ...     'text': ['He said "hello"    with   extra spaces']
            ... })
            >>> preprocessor = TextPreprocessor()
            >>> clean_corpus = preprocessor.preprocess_corpus(corpus)
        """
        return (
            corp.with_columns(
                pl.col("text").map_elements(
                    TextPreprocessor.squish_whitespace, return_dtype=pl.String
                    )
            )
            .with_columns(
                pl.col("text").map_elements(
                    TextPreprocessor.replace_curly_quotes,
                    return_dtype=pl.String,
                )
            )
            .with_columns(
                pl.col("text").map_elements(
                    TextPreprocessor.normalize_unicode, return_dtype=pl.String
                    )
            )
            .with_columns(
                pl.col("text").map_elements(
                    TextPreprocessor.squish_whitespace, return_dtype=pl.String
                    )
            )
        )


class TextChunker:
    """
    Handles splitting large texts into manageable chunks for processing.

    This class provides intelligent text chunking that attempts to split
    on natural boundaries (sentences, then words) to preserve linguistic
    context while keeping chunks within memory limits for spaCy processing.

    Example:
        Split a long document::

            from pybiber.processors import TextChunker

            chunker = TextChunker()
            long_text = "First sentence. Second sentence. " * 1000

            # Split into 3 chunks
            chunks = chunker.split_document(long_text, 3)
            print(f"Split into {len(chunks)} chunks")

        Process a corpus with automatic chunking::

            import polars as pl

            corpus = pl.DataFrame({
                'doc_id': ['long_doc.txt'],
                'text': ['Very long document text...']
            })

            chunker = TextChunker()
            chunked_corpus = chunker.prepare_chunked_corpus(corpus)
    """

    @staticmethod
    def split_document(doc_txt: str, n_chunks: float) -> List[str]:
        """
        Split documents into chunks, preferring natural boundaries.

        Attempts to split on sentence boundaries first, then word boundaries,
        and finally character boundaries as fallbacks. This preserves
        linguistic context while ensuring manageable chunk sizes.

        Args:
            doc_txt: The document text to be split.
            n_chunks: Number of chunks to create (can be fractional).

        Returns:
            List of text chunks, ideally split on natural boundaries.

        Example:
            >>> chunker = TextChunker()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> chunks = chunker.split_document(text, 2)
            >>> len(chunks)
            2

        Note:
            If n_chunks <= 1, returns the original text as a single chunk.
        """
        if n_chunks <= 0:
            return []  # Return empty list for empty documents
        if n_chunks <= 1:
            return [doc_txt]  # Return single chunk for short documents

        doc_len = len(doc_txt)
        chunk_idx = [
            math.ceil(i / n_chunks * doc_len)
            for i in range(1, int(n_chunks))
        ]

        # Try to split on sentence boundaries first
        try:
            split_idx = [
                (
                    PATTERNS.SENTENCE_BOUNDARY.search(doc_txt[idx:])
                    .span()[1]
                    + (idx - 1)
                )
                for idx in chunk_idx
            ]
            split_idx.insert(0, 0)
            doc_chunks = [
                doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:] + [None])
            ]

            if len(doc_chunks) == n_chunks:
                return doc_chunks
        except (AttributeError, IndexError):
            # Fall back to word boundaries if sentence splitting fails
            pass

        # Fallback to word boundaries
        try:
            split_idx = [
                PATTERNS.WORD_BOUNDARY.search(doc_txt[idx:]).span()[0] + idx
                for idx in chunk_idx
            ]
            split_idx.insert(0, 0)
            doc_chunks = [
                doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:] + [None])
            ]
            return doc_chunks
        except (AttributeError, IndexError):
            # If all else fails, just split at character boundaries
            split_idx = chunk_idx[:]
            split_idx.insert(0, 0)
            split_idx.append(len(doc_txt))
            return [doc_txt[i:j] for i, j in zip(split_idx, split_idx[1:])]

    def prepare_chunked_corpus(self, corp: pl.DataFrame) -> pl.DataFrame:
        """
        Split long texts into chunks and prepare for processing.

        Automatically determines which documents need chunking based on
        their length, splits them appropriately, and creates a new corpus
        DataFrame with chunked documents that have unique identifiers.

        Args:
            corp: A polars DataFrame with 'doc_id' and 'text' columns.

        Returns:
            DataFrame with potentially more rows (due to chunking) where
            each chunk has a unique doc_id formed by combining chunk number
            and original doc_id.

        Example:
            >>> import polars as pl
            >>> corpus = pl.DataFrame({
            ...     'doc_id': ['doc1.txt', 'very_long_doc.txt'],
            ...     'text': ['Short text', 'Very long text that needs chunking...']
            ... })
            >>> chunker = TextChunker()
            >>> chunked = chunker.prepare_chunked_corpus(corpus)
            # May result in doc_ids like: 'doc1.txt', '0::very_long_doc.txt', '1::very_long_doc.txt'
        """  # noqa: E501
        return (
            corp.with_columns(
                n_chunks=pl.Expr.ceil(
                    pl.col("text").str.len_chars().truediv(
                        CONFIG.MAX_CHUNK_SIZE
                    )
                ).cast(pl.UInt32, strict=False)
            )
            .with_columns(
                # Guarantee at least 1 chunk so empty texts are retained
                pl.when(pl.col("n_chunks") < 1)
                .then(pl.lit(1))
                .otherwise(pl.col("n_chunks"))
                .alias("n_chunks")
            )
            .with_columns(chunk_id=pl.int_ranges("n_chunks"))
            .with_columns(
                pl.struct(["text", "n_chunks"])
                .map_elements(
                    lambda x: self.split_document(x["text"], x["n_chunks"]),
                    return_dtype=pl.List(pl.String),
                )
                .alias("text")
            )
            .explode("text", "chunk_id")
            .filter(pl.col("text").is_not_null())
            .with_columns(pl.col("text").str.strip_chars() + " ")
            .with_columns(
                pl.concat_str(
                    [pl.col("chunk_id"), pl.col("doc_id")],
                    separator=CONFIG.CHUNK_ID_SEPARATOR,
                ).alias("doc_id")
            )
            .drop(["n_chunks", "chunk_id"])
        )


class SpacyProcessor:
    """Handles spaCy NLP pipeline processing."""

    @staticmethod
    def prepare_text_tuples(
        corp: pl.DataFrame,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Convert corpus DataFrame to spaCy-compatible tuple format."""
        text_tuples = []
        for item in corp.to_dicts():
            text_tuples.append((item["text"], {"doc_id": item["doc_id"]}))
        return text_tuples

    @staticmethod
    def setup_doc_extension() -> None:
        """Setup spaCy Doc extension for document IDs."""
        if not Doc.has_extension("doc_id"):
            Doc.set_extension("doc_id", default=None)

    def process_with_spacy(
        self,
        text_tuples: List[Tuple[str, Dict[str, Any]]],
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        disable_ner: bool = True,
        show_progress: bool = False,
    ) -> List[pl.DataFrame]:
        """Process text tuples through spaCy pipeline."""
        self.setup_doc_extension()

        if show_progress:
            progress = ProgressTracker(len(text_tuples), "spaCy processing")

        if disable_ner and "ner" in getattr(nlp_model, "pipe_names", []):
            with nlp_model.select_pipes(disable=["ner"]):
                doc_tuples = nlp_model.pipe(
                    text_tuples, as_tuples=True,
                    n_process=n_process, batch_size=batch_size
                )
        else:
            doc_tuples = nlp_model.pipe(
                text_tuples, as_tuples=True,
                n_process=n_process, batch_size=batch_size
            )

        df_list = []
        processed_count = 0

        for doc, context in doc_tuples:
            doc._.doc_id = context["doc_id"]

            # Extract token information
            sentence_id_list = [token.is_sent_start for token in doc]
            token_id_list = [token.i for token in doc]
            token_list = [token.text for token in doc]
            lemma_list = [token.lemma_ for token in doc]
            pos_list = [token.pos_ for token in doc]
            tag_list = [token.tag_ for token in doc]
            head_list = [token.head.i for token in doc]
            dependency_list = [token.dep_ for token in doc]

            df = pl.DataFrame({
                "doc_id": doc._.doc_id,
                "sentence_id": sentence_id_list,
                "token_id": token_id_list,
                "token": token_list,
                "lemma": lemma_list,
                "pos": pos_list,
                "tag": tag_list,
                "head_token_id": head_list,
                "dep_rel": dependency_list
            })

            df_list.append(df)

            processed_count += 1
            if show_progress:
                progress.update()

        if show_progress:
            progress.finish()

        return df_list

    def process_noun_phrases(
        self,
        text_tuples: List[Tuple[str, Dict[str, Any]]],
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        disable_ner: bool = True,
        show_progress: bool = False,
    ) -> List[pl.DataFrame]:
        """Process text tuples through spaCy pipeline."""
        self.setup_doc_extension()

        if show_progress:
            progress = ProgressTracker(len(text_tuples), "spaCy processing")

        if disable_ner and "ner" in getattr(nlp_model, "pipe_names", []):
            with nlp_model.select_pipes(disable=["ner"]):
                doc_tuples = nlp_model.pipe(
                    text_tuples, as_tuples=True,
                    n_process=n_process, batch_size=batch_size
                )
        else:
            doc_tuples = nlp_model.pipe(
                text_tuples, as_tuples=True,
                n_process=n_process, batch_size=batch_size
            )

        df_list = []
        processed_count = 0

        for doc, context in doc_tuples:
            doc._.doc_id = context["doc_id"]
            phrase_text = []
            phrase_tags = []
            phrase_len = []
            root_text = []
            root_tag = []
            root_idx = []
            start_idx = []
            end_idx = []
            spans = []
            # get spans for all noun chunks
            for nc in doc.noun_chunks:
                nc_span = doc[nc.root.left_edge.i:nc.root.right_edge.i+1]
                spans.append(nc_span)
            # filter non-overlapping noun chunks
            filtered_nps = filter_spans(spans)
            # gather attributes
            for nc in filtered_nps:
                nc_span = doc[nc.root.left_edge.i:nc.root.right_edge.i+1]
                phrase_text.append(
                    nc_span.text
                    )
                phrase_tags.append(
                    " | ".join([t.tag_ for t in nc_span])
                    )
                # Count tokens with uppercase POS tags (faster than regex)
                phrase_len.append(
                    sum((t.tag_[:1].isupper() for t in nc_span))
                )
                start_idx.append(nc.root.left_edge.i)
                end_idx.append(nc.root.right_edge.i)
                root_text.append(nc.root.text)
                root_tag.append(doc[nc.root.i].tag_)
                root_idx.append(nc.root.i)

            df = pl.DataFrame({
                "doc_id": doc._.doc_id,
                "phrase_text": phrase_text,
                "phrase_tags": phrase_tags,
                "phrase_len": phrase_len,
                "root_text": root_text,
                "root_tag": root_tag,
                "root_idx": root_idx,
                "start_idx": start_idx,
                "end_idx": end_idx
            })
            df_list.append(df)

            processed_count += 1
            if show_progress:
                progress.update()

        if show_progress:
            progress.finish()

        return df_list


class DataFrameTransformer:
    """Handles DataFrame transformations and formatting."""

    @staticmethod
    def recombine_chunks(
        df: pl.DataFrame, original_doc_order: List[str] = None
    ) -> pl.DataFrame:
        """Recombine chunked documents and preserve original document order."""
        # Split chunk info from doc_id
        df_split = (
            df.with_columns(
                pl.col("doc_id").str.split_exact(CONFIG.CHUNK_ID_SEPARATOR, 1)
            )
            .unnest("doc_id")
            .rename({"field_0": "chunk_id", "field_1": "doc_id"})
            .with_columns(pl.col("chunk_id").cast(pl.UInt32, strict=False))
        )

        # If we have original document order, preserve it
        if original_doc_order is not None:
            # Create order mapping
            doc_order_map = {doc_id: i for i, doc_id in enumerate(original_doc_order)}  # noqa: E501

            # Add order column and sort by it, then by chunk_id
            df_with_order = df_split.with_columns(
                pl.col("doc_id").map_elements(
                    # Unknown docs go to end
                    lambda x: doc_order_map.get(x, 999999),
                    return_dtype=pl.UInt32,
                ).alias("doc_order")
            )

            result = (
                df_with_order
                .sort(["doc_order", "chunk_id"], descending=[False, False])
                .drop(["chunk_id", "doc_order"])
            )
        else:
            # Fallback: sort by doc_id and chunk_id
            # (preserves alphabetical order)
            result = (
                df_split
                .sort(["doc_id", "chunk_id"], descending=[False, False])
                .drop("chunk_id")
            )

        return result

    def transform_spacy_output(
        self, df_list: List[pl.DataFrame], original_doc_order: List[str] = None
    ) -> pl.DataFrame:
        """Apply all transformations to spaCy output."""
        if not df_list:
            return pl.DataFrame({
                "doc_id": [],
                "sentence_id": [],
                "token_id": [],
                "token": [],
                "lemma": [],
                "pos": [],
                "tag": [],
                "head_token_id": [],
                "dep_rel": []
            })
        # Concatenate all DataFrames
        df = pl.concat(df_list)

        # Apply transformations in sequence
        df = self.recombine_chunks(df, original_doc_order=original_doc_order)

        # convert boolean to numerical id
        df = (df
              .with_columns(
                pl.col("sentence_id").cum_sum().over("doc_id")
                ))

        return df

    def transform_phrase_output(
        self, df_list: List[pl.DataFrame], original_doc_order: List[str] = None
    ) -> pl.DataFrame:
        """Transform noun-phrase outputs (no sentence_id logic)."""
        if not df_list:
            return pl.DataFrame({
                "doc_id": [],
                "phrase_text": [],
                "phrase_tags": [],
                "phrase_len": [],
                "root_text": [],
                "root_tag": [],
                "root_idx": [],
                "start_idx": [],
                "end_idx": [],
            })
        df = pl.concat(df_list)
        df = self.recombine_chunks(df, original_doc_order=original_doc_order)
        return df


class CorpusProcessor:
    """Main class that orchestrates corpus processing pipeline."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.chunker = TextChunker()
        self.spacy_processor = SpacyProcessor()
        self.transformer = DataFrameTransformer()

    def process_corpus(
        self,
        corp: pl.DataFrame,
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        disable_ner: bool = True,
        show_progress: bool = None,
    ) -> pl.DataFrame:
        """
        Process a corpus using the complete pipeline.

        :param corp: A polars DataFrame containing 'doc_id' and 'text' columns.
        :param nlp_model: An 'en_core_web' instance.
        :param n_process: The number of parallel processes to use
                         during parsing.
        :param batch_size: The batch size to use during parsing.
        :param show_progress: Whether to show progress for large corpora.
                             If None, will auto-determine based on corpus size.
        :return: A polars DataFrame with full dependency parses.
        """
        with PerformanceMonitor("Corpus processing"):
            # Validate inputs
            # Validate inputs
            validate_corpus_dataframe(corp, "in CorpusProcessor")
            validate_spacy_model(nlp_model, "in CorpusProcessor")

            # Filter out null texts only
            # (keep empty so legacy behavior preserved via replacement)
            corp = corp.filter(pl.col("text").is_not_null())

            # Capture original document order before any transformations
            original_doc_order = corp["doc_id"].to_list()

            # Determine if we should show progress
            if show_progress is None:
                show_progress = len(corp) > CONFIG.PROGRESS_THRESHOLD

            # Initialize progress tracker
            if show_progress:
                progress = ProgressTracker(len(corp), "Processing corpus")

            # Check if this is a large corpus that needs memory optimization
            if MemoryOptimizer.is_large_corpus_size(len(corp)):
                # Enable memory efficient mode temporarily
                original_mode = CONFIG.MEMORY_EFFICIENT_MODE
                CONFIG.MEMORY_EFFICIENT_MODE = True

            try:
                # Preprocess text
                corp = self.preprocessor.preprocess_corpus(corp)
                # Match legacy behavior:
                # replace empty / whitespace-only with single space
                corp = corp.with_columns(
                    pl.when(pl.col("text").str.len_chars() == 0)
                    .then(pl.lit(" "))
                    .otherwise(pl.col("text"))
                    .alias("text")
                )
                if show_progress:
                    progress.update(len(corp) // 4)

                # Chunk large texts
                corp = self.chunker.prepare_chunked_corpus(corp)
                if show_progress:
                    progress.update(len(corp) // 4)

                # Prepare for spaCy processing
                text_tuples = self.spacy_processor.prepare_text_tuples(corp)

                # Process with spaCy
                df_list = self.spacy_processor.process_with_spacy(
                    text_tuples,
                    nlp_model,
                    n_process,
                    batch_size,
                    disable_ner=disable_ner,
                    show_progress=show_progress,
                )
                if show_progress:
                    progress.update(len(corp) // 4)

                # Transform and finalize
                result = self.transformer.transform_spacy_output(
                    df_list, original_doc_order=original_doc_order
                )
                if show_progress:
                    progress.update(len(corp) // 4)
                    progress.finish()

                # Optimize result DataFrame
                result = MemoryOptimizer.optimize_dataframe(result)

                return result

            finally:
                # Restore original memory mode if we changed it
                if MemoryOptimizer.is_large_corpus_size(len(corp)):
                    CONFIG.MEMORY_EFFICIENT_MODE = original_mode

    def extract_noun_phrases(
        self,
        corp: pl.DataFrame,
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        disable_ner: bool = True,
        show_progress: bool = None,
    ) -> pl.DataFrame:
        """
        Process a corpus using the complete pipeline.

        :param corp: A polars DataFrame containing 'doc_id' and 'text' columns.
        :param nlp_model: An 'en_core_web_sm' instance.
        :param n_process: The number of parallel processes to use
                         during parsing.
        :param batch_size: The batch size to use during parsing.
        :param show_progress: Whether to show progress for large corpora.
                             If None, will auto-determine based on corpus size.
        :return: A polars DataFrame with full dependency parses.
        """
        with PerformanceMonitor("Corpus processing"):
            # Validate inputs
            validate_corpus_dataframe(corp, "in CorpusProcessor")
            validate_spacy_model(nlp_model, "in CorpusProcessor")

            # Filter out null texts only
            # (keep empty so legacy behavior preserved via replacement)
            corp = corp.filter(pl.col("text").is_not_null())

            # Capture original document order before any transformations
            original_doc_order = corp["doc_id"].to_list()

            # Determine if we should show progress
            if show_progress is None:
                show_progress = len(corp) > CONFIG.PROGRESS_THRESHOLD

            # Initialize progress tracker
            if show_progress:
                progress = ProgressTracker(len(corp), "Processing corpus")

            # Check if this is a large corpus that needs memory optimization
            if MemoryOptimizer.is_large_corpus_size(len(corp)):
                # Enable memory efficient mode temporarily
                original_mode = CONFIG.MEMORY_EFFICIENT_MODE
                CONFIG.MEMORY_EFFICIENT_MODE = True

            try:
                # Preprocess text
                corp = self.preprocessor.preprocess_corpus(corp)
                # Match legacy behavior:
                # replace empty / whitespace-only with single space
                corp = corp.with_columns(
                    pl.when(pl.col("text").str.len_chars() == 0)
                    .then(pl.lit(" "))
                    .otherwise(pl.col("text"))
                    .alias("text")
                )
                if show_progress:
                    progress.update(len(corp) // 4)

                # Chunk large texts
                corp = self.chunker.prepare_chunked_corpus(corp)
                if show_progress:
                    progress.update(len(corp) // 4)

                # Prepare for spaCy processing
                text_tuples = self.spacy_processor.prepare_text_tuples(corp)

                # Process with spaCy
                df_list = self.spacy_processor.process_noun_phrases(
                    text_tuples,
                    nlp_model,
                    n_process,
                    batch_size,
                    disable_ner=disable_ner,
                    show_progress=show_progress,
                )
                if show_progress:
                    progress.update(len(corp) // 4)

                # Transform and finalize (phrase-aware)
                result = self.transformer.transform_phrase_output(
                    df_list, original_doc_order=original_doc_order
                )
                if show_progress:
                    progress.update(len(corp) // 4)
                    progress.finish()

                # Optimize result DataFrame
                result = MemoryOptimizer.optimize_dataframe(result)

                return result

            finally:
                # Restore original memory mode if we changed it
                if MemoryOptimizer.is_large_corpus_size(len(corp)):
                    CONFIG.MEMORY_EFFICIENT_MODE = original_mode

    # Back-compat convenience alias for tests and clients
    def spacy_parse(
        self,
        corp: pl.DataFrame,
        nlp_model: Language,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        disable_ner: bool = True,
        show_progress: bool = None,
    ) -> pl.DataFrame:
        """Alias of process_corpus for parity with legacy naming.

        Provided to ease migration and satisfy tests that call
        CorpusProcessor.spacy_parse().
        """
        return self.process_corpus(
            corp=corp,
            nlp_model=nlp_model,
            n_process=n_process,
            batch_size=batch_size,
            disable_ner=disable_ner,
            show_progress=show_progress,
        )


def spacy_parse(
    corp: pl.DataFrame,
    nlp_model: Language,
    n_process: int = 1,
    batch_size: int = 25,
    disable_ner: bool = True,
) -> pl.DataFrame:
    """Parse a corpus (legacy public API).

    This function is maintained for backward compatibility and now delegates
    to the new class-based pipeline (`CorpusProcessor.process_corpus`).

    Parameters
    ----------
    corp : pl.DataFrame
        DataFrame with 'doc_id' and 'text'.
    nlp_model : Language
        spaCy model instance (e.g., 'en_core_web_sm').
    n_process : int
        Number of processes passed to spaCy's pipe.
    batch_size : int
        Batch size for spaCy pipe.

    Returns
    -------
    pl.DataFrame
        Token-level dependency parse output identical in schema to the legacy
        implementation.
    """
    warnings.warn(
        "spacy_parse() is deprecated and will be removed in a future release. "
        "Use CorpusProcessor().process_corpus() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    processor = CorpusProcessor()
    try:
        # Legacy behavior: replace empty/whitespace-only texts
        # with single space
        if "text" in corp.columns:
            # Track original empty documents (after stripping)
            empty_ids = (
                corp
                .with_columns(pl.col("text").str.strip_chars().alias("_t"))
                .filter(pl.col("_t").str.len_chars() == 0)
                .get_column("doc_id")
                .to_list()
            )
            corp = corp.with_columns(
                pl.when(pl.col("text").str.len_chars() == 0)
                .then(pl.lit(" "))
                .otherwise(pl.col("text"))
                .alias("text")
            )
        # Delegate (auto-detect progress inside the processor)
        result = processor.process_corpus(
            corp=corp,
            nlp_model=nlp_model,
            n_process=n_process,
            batch_size=batch_size,
            disable_ner=disable_ner,
            show_progress=None,
        )
        # Drop originally empty documents to maintain
        # legacy integration expectation
        if "empty_ids" in locals() and empty_ids:
            result = result.filter(~pl.col("doc_id").is_in(empty_ids))
        return result
    except CorpusValidationError as e:  # Preserve legacy interface
        raise ValueError(
            "Invalid DataFrame. Expected a DataFrame with 2 columns "
            "(doc_id & text)."
        ) from e
    except ModelValidationError as e:  # Preserve legacy interface
        raise ValueError(str(e)) from e


def get_noun_phrases(corp: pl.DataFrame,
                     nlp_model: Language,
                     n_process=1,
                     batch_size=25,
                     disable_ner: bool = True) -> pl.DataFrame:
    """
    Extract expanded noun phrases using the 'en_core_web_sm' model.

    Parameters
    ----------
    corp:
        A polars DataFrame
        conataining a 'doc_id' column and a 'text' column.
    nlp_model:
        An 'en_core_web_sm' instance.
    n_process:
        The number of parallel processes
        to use during parsing.
    batch_size:
        The batch size to use during parsing.

    Returns
    -------
    pl.DataFrame
        a polars DataFrame with,
        noun phrases and their assocated part-of-speech tags.

    Notes
    -----
    Noun phrases can be extracted directly from the
    [noun_chunks](https://spacy.io/api/doc#noun_chunks)
    attribute. However, per spaCy's documentation
    the attribute does not permit nested noun phrases,
    for example when a prepositional phrases modifies
    a preceding noun phrase. This function extracts
    elatorated noun phrases in their complete form.

    """
    warnings.warn(
        "get_noun_phrases() is deprecated and will be removed in a future "
        "release. Use CorpusProcessor().extract_noun_phrases() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    processor = CorpusProcessor()
    try:
        # Legacy behavior: replace empty/whitespace-only texts
        # with single space
        if "text" in corp.columns:
            # Track original empty documents (after stripping)
            empty_ids = (
                corp
                .with_columns(pl.col("text").str.strip_chars().alias("_t"))
                .filter(pl.col("_t").str.len_chars() == 0)
                .get_column("doc_id")
                .to_list()
            )
            corp = corp.with_columns(
                pl.when(pl.col("text").str.len_chars() == 0)
                .then(pl.lit(" "))
                .otherwise(pl.col("text"))
                .alias("text")
            )
    # Delegate to phrase extractor (auto-detect progress)
        result = processor.extract_noun_phrases(
            corp=corp,
            nlp_model=nlp_model,
            n_process=n_process,
            batch_size=batch_size,
            disable_ner=disable_ner,
            show_progress=None,
        )
        # Drop originally empty documents to maintain
        # legacy integration expectation
        if "empty_ids" in locals() and empty_ids:
            result = result.filter(~pl.col("doc_id").is_in(empty_ids))
        return result
    except CorpusValidationError as e:  # Preserve legacy interface
        raise ValueError(
            "Invalid DataFrame. Expected a DataFrame with 2 columns "
            "(doc_id & text)."
        ) from e
    except ModelValidationError as e:  # Preserve legacy interface
        raise ValueError(str(e)) from e


def get_text_paths(directory: str,
                   recursive=False) -> List:
    """Get a list of full paths for all text files.

    Parameters
    ----------
    directory:
        A string indictating a path to a directory.
    recursive:
        Whether to search subdirectories.

    Returns
    -------
    List
        A list of full paths.

    """
    full_paths = []
    if recursive is True:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt'):
                    full_paths.append(os.path.join(root, file))
    else:
        for file in Path(directory).glob("*.txt"):
            full_paths.append(str(file))
    return full_paths


def readtext(paths: List) -> pl.DataFrame:
    """Import all text files from a list of paths.

    Parameters
    ----------
    paths:
        A list of paths of text files returned by get_text_paths.

    Returns
    -------
    List
        A list of full paths.

    Notes
    -----
    Modeled on the R function
    [readtext](https://readtext.quanteda.io/articles/readtext_vignette.html).

    """
    # Get a list of the file basenames
    doc_ids = [os.path.basename(path) for path in paths]
    # Create a list collapsing each text file into one element in a string
    texts = [open(path).read() for path in paths]
    df = pl.DataFrame({
        "doc_id": doc_ids,
        "text": texts
    })
    df = (
        df
        .with_columns(
            pl.col("text").str.strip_chars()
        )
        .sort("doc_id", descending=False)
    )
    return df


def corpus_from_folder(directory: str) -> pl.DataFrame:
    """Import all text files from a directory.

    Parameters
    ----------
    directory:
        A directory containing text files.

    Returns
    -------
    pl.DataFrame
        A polars DataFrame.

    """
    text_files = get_text_paths(directory)
    if len(text_files) == 0:
        raise ValueError("""
                    No text files found in directory.
                    """)
    df = readtext(text_files)
    return df
