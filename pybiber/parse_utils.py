"""
Utilities for processessing text data using a spaCy instance.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import os
import re
import math
import unicodedata
from pathlib import Path
from collections import OrderedDict
from typing import List

import polars as pl
from spacy.tokens import Doc
from spacy.language import Language
from spacy.util import filter_spans


def _str_squish(text: str) -> str:
    """Remove extra spaces, returns, etc. from a string.

    Parameters
    ----------
    text:
        A string.

    Returns
    -------
    str
        A string.

    """
    return " ".join(text.split())


def _replace_curly_quotes(text: str) -> str:
    """Replaces curly quotes with straight quotes.

    Parameters
    ----------
    text:
        A string.ß

    Returns
    -------
    str
        A string.

    """
    text = text.replace(u'\u2018', "'")  # Left single quote
    text = text.replace(u'\u2019', "'")  # Right single quote
    text = text.replace(u'\u201C', '"')  # Left double quote
    text = text.replace(u'\u201D', '"')  # Right double quote
    return text


def _split_docs(doc_txt: str,
                n_chunks: float) -> List:
    """Split documents that will exhaust spaCy's memory into smaller chunks.

    Parameters
    ----------
    doc_txt:
        A string.
    n_chunks:
        The number of chunks for splitting.

    Returns
    -------
    List
        A list of strings.

    """
    sent_boundary = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s')
    doc_len = len(doc_txt)
    chunk_idx = [math.ceil(i/n_chunks*doc_len) for i in range(1, n_chunks)]
    split_idx = [sent_boundary.search(
        doc_txt[idx:]
        ).span()[1] + (idx-1) for idx in chunk_idx]
    split_idx.insert(0, 0)
    doc_chunks = [doc_txt[i:j] for i, j in zip(
        split_idx, split_idx[1:] + [None]
        )]
    if len(doc_chunks) == n_chunks:
        return doc_chunks
    else:
        split_idx = [re.search(
            ' ', doc_txt[idx:]
            ).span()[0] + idx for idx in chunk_idx]
        split_idx.insert(0, 0)
        doc_chunks = [doc_txt[i:j] for i, j in zip(
            split_idx, split_idx[1:] + [None]
            )]
        return doc_chunks


def _pre_process_corpus(corp: pl.DataFrame) -> pl.DataFrame:
    """Format texts to increase spaCy tagging accuracy.

    Parameters
    ----------
    corp:
        A polars DataFrame with 'doc_id' and 'text' columns.

    Returns
    -------
    pl.DataFrame
        A polars DataFrame.

    """
    df = (
        corp
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: _str_squish(x),
                          return_dtype=pl.String)
                        )
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: _replace_curly_quotes(x),
                          return_dtype=pl.String)
                        )
        .with_columns(
            pl.col('text')
            .map_elements(lambda x: unicodedata.normalize('NFKD', x)
                          .encode('ascii', errors='ignore')
                          .decode('utf-8'), return_dtype=pl.String)
                        )
    )
    return df


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


def spacy_parse(corp: pl.DataFrame,
                nlp_model: Language,
                n_process=1,
                batch_size=25) -> pl.DataFrame:
    """Parse a corpus using the 'en_core_web_sm' model.

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
        A polars DataFrame with,
        token sequencies identified by part-of-speech tags
        and dependency parses.

    """
    validation = OrderedDict([('doc_id', pl.String),
                              ('text', pl.String)])
    if corp.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame with 2 columns (doc_id & text).
                         """)
    if (nlp_model.lang != 'en' and
        all(item in nlp_model.pipe_names for item in ['tagger',
                                                      'parser',
                                                      'lemmatizer']
            )):
        raise ValueError("""
                         Invalid spaCy model. Expected a pipeline with \
                         tagger, parser and lemmatizer like 'en_core_web_sm'.
                         For information and instructions see:
                         https://spacy.io/models/en
                         """)

    corp = _pre_process_corpus(corp)
    # split long texts (> 500000 chars) into chunks
    corp = (corp
            .with_columns(
                n_chunks=pl.Expr.ceil(
                    pl.col("text").str.len_chars().truediv(500000)
                )
                .cast(pl.UInt32, strict=False)
                )
            .with_columns(
                chunk_id=pl.int_ranges("n_chunks")
                )
            .with_columns(
                pl.struct(['text', 'n_chunks'])
                .map_elements(lambda x: _split_docs(x['text'], x['n_chunks']),
                              return_dtype=pl.List(pl.String))
                .alias("text")
                )
            .explode("text", "chunk_id")
            .with_columns(
                pl.col("text").str.strip_chars() + " "
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("chunk_id"),
                        pl.col("doc_id")
                        ], separator="@",
                        ).alias("doc_id")
                    )
            .drop(["n_chunks", "chunk_id"])
            )
    # tuple format for spaCy
    text_tuples = []
    for item in corp.to_dicts():
        text_tuples.append((item['text'], {"doc_id": item['doc_id']}))
    # add doc_id as custom attribute
    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    # create pipeline
    doc_tuples = nlp_model.pipe(text_tuples,
                                as_tuples=True,
                                n_process=n_process,
                                batch_size=batch_size)
    # process corpus and gather into a DataFrame
    df_list = []
    for doc, context in doc_tuples:
        doc._.doc_id = context["doc_id"]
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
    # contatenate list of DataFrames
    df = pl.concat(df_list)
    # convert boolean to numerical id
    df = (
        df
        .with_columns(
            pl.col("doc_id")
            .str.split_exact("@", 1)
            )
        .unnest("doc_id")
        .rename({"field_0": "chunk_id", "field_1": "doc_id"})
        .with_columns(
            pl.col("chunk_id")
            .cast(pl.UInt32, strict=False)
            )
        .sort(["doc_id", "chunk_id"], descending=[False, False])
        .drop("chunk_id")
        .with_columns(
            pl.col("sentence_id").cum_sum().over("doc_id")
            )
        )
    return df


def get_noun_phrases(corp: pl.DataFrame,
                     nlp_model: Language,
                     n_process=1,
                     batch_size=25) -> pl.DataFrame:
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
    validation = OrderedDict([('doc_id', pl.String),
                              ('text', pl.String)])
    if corp.collect_schema() != validation:
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame with 2 columns (doc_id & text).
                         """)
    if (nlp_model.lang != 'en' and
        all(item in nlp_model.pipe_names for item in ['tagger']
            )):
        raise ValueError("""
                         Invalid spaCy model. Expected a pipeline with
                         tagger, parser and lemmatizer like 'en_core_web_sm'.
                         For information and instructions see:
                         https://spacy.io/models/en
                         """)

    corp = _pre_process_corpus(corp)
    # split long texts (> 500000 chars) into chunks
    corp = (corp
            .with_columns(
                n_chunks=pl.Expr.ceil(
                    pl.col("text").str.len_chars().truediv(500000)
                )
                .cast(pl.UInt32, strict=False)
                )
            .with_columns(
                chunk_id=pl.int_ranges("n_chunks")
                )
            .with_columns(
                pl.struct(['text', 'n_chunks'])
                .map_elements(lambda x: _split_docs(x['text'], x['n_chunks']),
                              return_dtype=pl.List(pl.String))
                .alias("text")
                )
            .explode("text", "chunk_id")
            .with_columns(
                pl.col("text").str.strip_chars() + " "
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("chunk_id"),
                        pl.col("doc_id")
                        ], separator="@",
                        ).alias("doc_id")
                    )
            .drop(["n_chunks", "chunk_id"])
            )
    text_tuples = []
    for item in corp.to_dicts():
        text_tuples.append((item['text'], {"doc_id": item['doc_id']}))
    # add doc_id as custom attribute
    if not Doc.has_extension("doc_id"):
        Doc.set_extension("doc_id", default=None)
    # create pipeline
    doc_tuples = nlp_model.pipe(text_tuples,
                                as_tuples=True,
                                n_process=n_process,
                                batch_size=batch_size)
    # process corpus and gather into a DataFrame
    df_list = []
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
            phrase_len.append(
                sum([bool(re.match("^[A-Z]", t.tag_)) for t in nc_span])
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
    df = pl.concat(df_list)

    df = (
        df
        .with_columns(
            pl.col("doc_id")
            .str.split_exact("@", 1)
            )
        .unnest("doc_id")
        .rename({"field_0": "chunk_id", "field_1": "doc_id"})
        .with_columns(
            pl.col("chunk_id")
            .cast(pl.UInt32, strict=False)
            )
        .sort(["doc_id", "chunk_id"], descending=[False, False])
        .drop("chunk_id")
        )
    return df
