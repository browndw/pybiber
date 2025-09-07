# flake8: noqa

# Set version ----
try:
    from importlib.metadata import version as _v
    __version__ = _v("pybiber")  # type: ignore
    del _v
except Exception:  # pragma: no cover
    __version__ = "0.0.0+local"

# Imports ----
from .parse_utils import (
    corpus_from_folder,
    CorpusProcessor,
    get_noun_phrases,
    get_text_paths,
    readtext,
    spacy_parse,
)

from .parse_functions import biber

from .biber_analyzer import BiberAnalyzer
from .pipeline import PybiberPipeline, run_biber_from_folder, run_biber

__all__ = ['get_text_paths', 'readtext', 'corpus_from_folder',
           'spacy_parse', 'get_noun_phrases', 'biber', 'BiberAnalyzer',
           'CorpusProcessor', 'PybiberPipeline', 'run_biber_from_folder',
           'run_biber']
