# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.3.0] - 2026-01-10

- Features: Added configurable MATTR window size via `mattr_window` for `biber()` and pipeline helpers; if the requested window exceeds the shortest document length, the window is reduced to the shortest length with a warning.
- Refactor: Removed SciPy/FactorAnalyzer runtime dependency from MDA; implemented NumPy-only maximum-likelihood factor analysis and rotations.
- Parity: Restored PCA parity with legacy artifacts (stable sign conventions and consistent loadings/scores definitions).
- Parity: Aligned MDA extraction/rotation pipeline with legacy 0.2.0 (ML extraction → varimax → stats::factanal-style promax conversion; deterministic ordering).
- Optimizer: Improved bounded ML optimization stability near active bounds (projected gradient) and made multi-start behavior stable (prefer primary SMC start; restarts as fallback).
- Tests: Added regression coverage against cached MICUSP parquet artifacts (no spaCy parse required).

## [0.2.0] - 2025-09-08

- Refactor: Introduced class-based `CorpusProcessor` pipeline with validation → preprocessing → chunking → spaCy parse → shaping.
- Determinism: Numeric cumulative `sentence_id` per doc; stable `doc_id` sort; single-pass lag/lead; right-join zero-fill helpers.
- Validation: Centralized validators and error messages in `pybiber/validation.py`; added `validate_spacy_model`.
- Features: Modularized `biber()` into cohesive blocks; extracted regex-driven features; added split/coordination blocks; preserved normalization rules.
- Defaults: NER disabled by default with toggle; empty text normalization preserved for legacy parity.
- Tests: Expanded coverage; removed XFAIL for sentence ID parity; all tests passing on Python 3.10–3.12.
- Packaging: Include `pybiber/data/*.parquet` in wheels; corrected license classifier to MIT.

## [0.1.1] - 2024-xx-xx

- Previous release.
