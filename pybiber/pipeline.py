"""
High-level orchestration pipeline for pybiber.

This module provides a simple, user-friendly interface to:
 1) read a corpus from a folder of .txt files,
 2) parse the corpus with spaCy (dependency parsing required),
 3) extract Biber features,
 4) optionally hand off to BiberAnalyzer for visualization/statistics.

The pipeline prefers deterministic, memory-conscious defaults and keeps
NER disabled by default for speed and reproducibility. You can re-enable
NER by setting disable_ner=False.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import polars as pl
import spacy
from spacy.language import Language

from .config import CONFIG
from .parse_utils import (
    corpus_from_folder,
    get_text_paths,
    readtext,
    CorpusProcessor,
)
from .parse_functions import biber
from .biber_analyzer import BiberAnalyzer
from .validation import validate_spacy_model, validate_corpus_dataframe


class PybiberPipeline:
    """End-to-end convenience wrapper for common pybiber workflows.

    Parameters
    ----------
    nlp : Optional[Language]
        Pre-loaded spaCy model. If None, the model named by `model` will be
        loaded lazily on first use.
    model : str
        Name of the spaCy model to load when `nlp` is None. Defaults to
        "en_core_web_sm".
    disable_ner : bool
        When True, disable spaCy's NER component for speed and stability.
        Parser is still enabled and required. Defaults to True.
    n_process : int
        Number of processes to use for spaCy's pipe.
    batch_size : int
        Batch size for spaCy's pipe.
    show_progress : Optional[bool]
        Whether to show internal progress indicators when processing.
        If None, it is determined based on corpus size.
    """

    def __init__(
        self,
        nlp: Optional[Language] = None,
        model: str = "en_core_web_sm",
        disable_ner: bool = True,
        n_process: int = CONFIG.DEFAULT_N_PROCESS,
        batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
        show_progress: Optional[bool] = None,
    ) -> None:
        self._nlp = nlp
        self._model_name = model
        self.disable_ner = disable_ner
        self.n_process = n_process
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._processor = CorpusProcessor()

    def _get_nlp(self) -> Language:
        if self._nlp is None:
            self._nlp = spacy.load(self._model_name)
        # Validate model (en_core_web_* expected)
        validate_spacy_model(self._nlp, context="in PybiberPipeline")
        return self._nlp

    # ---- Step 1: Read corpus ----
    def from_folder(
        self, directory: str, recursive: bool = False
    ) -> pl.DataFrame:
        """Read .txt files from a folder into a corpus DataFrame."""
        if not recursive:
            return corpus_from_folder(directory)
        paths = get_text_paths(directory, recursive=True)
        return readtext(paths)

    # ---- Step 2: Parse corpus ----
    def parse(self, corpus: pl.DataFrame) -> pl.DataFrame:
        """Parse a corpus with spaCy using the configured settings."""
        validate_corpus_dataframe(corpus, context="in PybiberPipeline.parse")
        nlp = self._get_nlp()
        return self._processor.spacy_parse(
            corp=corpus,
            nlp_model=nlp,
            n_process=self.n_process,
            batch_size=self.batch_size,
            disable_ner=self.disable_ner,
            show_progress=self.show_progress,
        )

    # ---- Step 3: Extract features ----
    def features(
        self,
        tokens: pl.DataFrame,
        normalize: Optional[bool] = True,
        force_ttr: Optional[bool] = False,
    ) -> pl.DataFrame:
        """Compute Biber features from token-level parses."""
        return biber(tokens, normalize=normalize, force_ttr=force_ttr)

    # ---- End-to-end helpers ----
    def run_from_folder(
        self,
        directory: str,
        recursive: bool = False,
        return_tokens: bool = False,
        normalize: Optional[bool] = True,
        force_ttr: Optional[bool] = False,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Read, parse, and compute features from a folder of .txt files."""
        corpus = (
            corpus_from_folder(directory)
            if not recursive
            else self.from_folder(directory, recursive=True)
        )
        tokens = self.parse(corpus)
        features_df = self.features(
            tokens, normalize=normalize, force_ttr=force_ttr
        )
        return (features_df, tokens) if return_tokens else features_df

    def run(
        self,
        corpus: pl.DataFrame,
        return_tokens: bool = False,
        normalize: Optional[bool] = True,
        force_ttr: Optional[bool] = False,
    ) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
        """Parse and compute features from an in-memory corpus DataFrame."""
        tokens = self.parse(corpus)
        features_df = self.features(
            tokens, normalize=normalize, force_ttr=force_ttr
        )
        return (features_df, tokens) if return_tokens else features_df

    # ---- Optional analysis handoff ----
    def to_analyzer(self, biber_df: pl.DataFrame) -> BiberAnalyzer:
        """Create a BiberAnalyzer from a Biber feature matrix."""
        return BiberAnalyzer(biber_df)


# Convenience one-liners -----------------------------------------------------

def run_biber_from_folder(
    directory: str,
    model: str = "en_core_web_sm",
    disable_ner: bool = True,
    n_process: int = CONFIG.DEFAULT_N_PROCESS,
    batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
    recursive: bool = False,
    return_tokens: bool = False,
    normalize: Optional[bool] = True,
    force_ttr: Optional[bool] = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """One-liner: read -> parse -> biber() from a folder of .txt files."""
    pipeline = PybiberPipeline(
        nlp=None,
        model=model,
        disable_ner=disable_ner,
        n_process=n_process,
        batch_size=batch_size,
        show_progress=None,
    )
    return pipeline.run_from_folder(
        directory=directory,
        recursive=recursive,
        return_tokens=return_tokens,
        normalize=normalize,
        force_ttr=force_ttr,
    )


def run_biber(
    corpus: pl.DataFrame,
    nlp: Optional[Language] = None,
    model: str = "en_core_web_sm",
    disable_ner: bool = True,
    n_process: int = CONFIG.DEFAULT_N_PROCESS,
    batch_size: int = CONFIG.DEFAULT_BATCH_SIZE,
    return_tokens: bool = False,
    normalize: Optional[bool] = True,
    force_ttr: Optional[bool] = False,
) -> Union[pl.DataFrame, Tuple[pl.DataFrame, pl.DataFrame]]:
    """One-liner: parse -> biber() from an in-memory corpus DataFrame."""
    pipeline = PybiberPipeline(
        nlp=nlp,
        model=model,
        disable_ner=disable_ner,
        n_process=n_process,
        batch_size=batch_size,
        show_progress=None,
    )
    return pipeline.run(
        corpus=corpus,
        return_tokens=return_tokens,
        normalize=normalize,
        force_ttr=force_ttr,
    )
