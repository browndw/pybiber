"""
Configuration constants for pybiber.

.. codeauthor:: David Brown <dwb2@andrew.cmu.edu>
"""

from dataclasses import dataclass
import re


@dataclass
class ProcessingConfig:
    """Configuration constants for corpus processing."""

    # Text chunking
    MAX_CHUNK_SIZE: int = 500000
    CHUNK_ID_SEPARATOR: str = "@"

    # Normalization factors
    FREQUENCY_NORMALIZATION_FACTOR: int = 1000

    # spaCy processing defaults
    DEFAULT_N_PROCESS: int = 1
    DEFAULT_BATCH_SIZE: int = 25

    # Validation
    EXPECTED_MODEL_PREFIX: str = "en_core_web"

    # Performance optimization settings
    ENABLE_CACHING: bool = True
    CACHE_MAX_SIZE: int = 128
    MEMORY_EFFICIENT_MODE: bool = False
    BATCH_PROCESSING_THRESHOLD: int = 1000  # rows
    PROGRESS_THRESHOLD: int = 5000  # rows to show progress

    # Memory optimization thresholds
    LARGE_CORPUS_THRESHOLD: int = 10000  # documents
    STREAMING_THRESHOLD: int = 50000  # documents


@dataclass
class RegexPatterns:
    """Compiled regex patterns for text processing."""

    SENTENCE_BOUNDARY = re.compile(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s")  # noqa: E501
    WORD_BOUNDARY = re.compile(r" ")


# Global configuration instance
CONFIG = ProcessingConfig()
PATTERNS = RegexPatterns()
