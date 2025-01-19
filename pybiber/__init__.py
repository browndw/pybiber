# flake8: noqa

# Set version ----
from importlib_metadata import version as _v

__version__ = _v("pybiber")

del _v

# Imports ----
from .parse_utils import (
    corpus_from_folder,
    get_noun_phrases,
    get_text_paths,
    readtext,
    spacy_parse,
)

from .parse_functions import biber

from .biber_analyzer import BiberAnalyzer

__all__ = ['get_text_paths', 'readtext', 'corpus_from_folder',
           'spacy_parse', 'get_noun_phrases', 'biber', 'BiberAnalyzer']
