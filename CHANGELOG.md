# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.3.1] - 2026-03-06

- Fix: Preserved token boundaries during unicode normalization by mapping structural punctuation (for example, em/en dashes) to ASCII separators before ASCII encoding; this prevents unintended joins such as `simultaneouslyincluding` while preserving accent normalization (for example, `cafe` from `cafe`).
- Fix: Tightened `f_13_wh_question` detection logic to reduce false positives in declarative sentences and avoid sentence-level leakage patterns; aligned the heuristic with WH + auxiliary + sentence-start/punctuation context.
- Adjustment: Added configurable `strict_be_main_verb` handling for `f_19_be_main_verb` in `biber()` and pipeline helpers; strict mode now defaults on and counts only finite `be` sentence roots, while compatibility mode (`strict_be_main_verb=False`) counts finite non-auxiliary `be` constructions (including embedded predicatives).
- Defaults: Changed `strict_be_main_verb` default to `True` across `biber()` and pipeline convenience methods for more intuitive main-verb behavior out of the box.
- Tests: Added regression tests for dash-boundary normalization and for WH-question overcounting, including negative declarative examples and a cross-document `sentence_id` leakage guard.
- Tests: Added regression coverage for `f_19_be_main_verb` strict vs compatibility behavior and pipeline passthrough of `strict_be_main_verb`.

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
