# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

- Refactor: Introduced class-based `CorpusProcessor` pipeline with validation → preprocessing → chunking → spaCy parse → shaping.
- Determinism: Numeric cumulative `sentence_id` per doc; stable `doc_id` sort; single-pass lag/lead; right-join zero-fill helpers.
- Validation: Centralized validators and error messages in `pybiber/validation.py`; added `validate_spacy_model`.
- Features: Modularized `biber()` into cohesive blocks; extracted regex-driven features; added split/coordination blocks; preserved normalization rules.
- Defaults: NER disabled by default with toggle; empty text normalization preserved for legacy parity.
- Tests: Expanded coverage; removed XFAIL for sentence ID parity; all tests passing on Python 3.10–3.12.
- Packaging: Include `pybiber/data/*.parquet` in wheels; corrected license classifier to MIT.

## [0.1.1] - 2024-xx-xx

- Previous release.
