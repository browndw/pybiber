"""Extract Biber features from a corpus parsed and annotated by spaCy.

Logging
-------
Uses the module logger ``pybiber.parse``. Set level via::

    import logging
    logging.getLogger("pybiber.parse").setLevel(logging.INFO)

Performance
-----------
Regex patterns are precompiled once at import for speed.
"""

import re
import logging
from typing import Optional, Dict, Pattern

import polars as pl

from .biber_dict import FEATURES, WORDLISTS

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger("pybiber.parse")
if not logger.handlers:  # avoid duplicate handlers in interactive sessions
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Precompile regex patterns for FEATURES for performance
# ---------------------------------------------------------------------------
FEATURE_PATTERNS: Dict[str, Pattern[str]] = {
    k: re.compile("|".join(v)) for k, v in FEATURES.items()
}

# ---------------------------------------------------------------------------
# Small helpers to assemble multiple features efficiently
# ---------------------------------------------------------------------------


def _ensure_full_doc_index_multi(
        df: pl.DataFrame,
        ids_df: pl.DataFrame,
        col_names: list[str]
) -> pl.DataFrame:
    """Right-join to ids, zero-fill listed columns,
    sort by doc_id, and drop doc_id.

    Returns only the feature columns aligned by ascending doc_id.
    """
    ids_sorted = ids_df.sort("doc_id", descending=False)
    out = (
        ids_sorted
        .join(df, on="doc_id", how="left")
        .with_columns([
            pl.col(c).fill_null(strategy="zero")
            for c in col_names
        ])
        .sort("doc_id", descending=False)
    )
    return out.select(col_names)


def _block_aux_tense(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Features sharing lemma/aux patterns computed together.

    Returns columns: f_02_perfect_aspect, f_12_proverb_do
    """
    f02 = (
        tokens
        .filter(
            (pl.col("lemma") == "have")
            & (pl.col("dep_rel").str.contains("aux"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_02_perfect_aspect")
    )
    f12 = (
        tokens
        .filter(
            (pl.col("lemma") == "do")
            & (~pl.col("dep_rel").str.contains("aux"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_12_proverb_do")
    )
    df = f02.join(f12, on="doc_id", how="outer")
    return _ensure_full_doc_index_multi(
        df,
        ids,
        ["f_02_perfect_aspect", "f_12_proverb_do"],
    )


def _block_lexical_membership(
    tokens: pl.DataFrame,
    ids: pl.DataFrame
) -> pl.DataFrame:
    """Lexical membership and nominal form features computed together.

    Returns columns:
    - f_10_demonstrative_pronoun
    - f_14_nominalizations
    - f_15_gerunds
    - f_16_other_nouns (depends on f_14 and gerunds used as nouns)
    """
    f10 = (
        tokens
        .with_columns(pl.col("tag").shift(1).over("doc_id").alias("tag_1"))
        .filter(
            (pl.col("tag") == "DT")
            & (
                (pl.col("tag_1").is_null())
                | (~pl.col("tag_1").str.contains("^N|^CD|DT"))
            )
            & (pl.col("dep_rel").str.contains("nsubj|dobj|pobj"))
        )
        .filter(
            pl.col("token")
            .str.to_lowercase()
            .is_in(WORDLISTS['pronoun_matchlist'])
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_10_demonstrative_pronoun")
    )
    f14 = (
        tokens
        .filter(
            (pl.col("pos") == "NOUN")
            & (
                pl.col("token")
                .str.to_lowercase()
                .str.contains(
                    "tion$|tions$|ment$|ments$|" "ness$|nesses$|ity$|ities$"
                )
            )
        )
        .filter(
            ~pl.col("token")
            .str.to_lowercase()
            .is_in(WORDLISTS['nominalization_stoplist'])
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_14_nominalizations")
    )
    # f_15: gerunds (participial forms functioning as nouns)
    f15_base = (
        tokens
        .filter(
            pl.col("token").str.to_lowercase().str.contains("ing$|ings$")
            & pl.col("dep_rel").str.contains("nsub|dobj|pobj")
        )
        .filter(
            ~pl.col("token").str.to_lowercase()
            .is_in(WORDLISTS["gerund_stoplist"])
        )
    )

    f15 = (
        f15_base
        .group_by("doc_id", maintain_order=True)
        .len(name="f_15_gerunds")
    )

    # gerunds counted as NOUN for subtraction from other nouns (f16)
    gerunds_n = (
        f15_base
        .filter(pl.col("pos") == "NOUN")
        .group_by("doc_id", maintain_order=True)
        .len(name="gerunds_n")
    )

    # total other nouns before subtraction
    f16_raw = (
        tokens
        .filter((pl.col("pos") == "NOUN") | (pl.col("pos") == "PROPN"))
        .filter(~pl.col("token").str.contains("-"))
        .group_by("doc_id", maintain_order=True)
        .len(name="f_16_other_nouns")
    )

    # Join all parts and compute adjusted f_16
    # f_51: demonstratives used as determiners
    f51 = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase()
             .is_in(WORDLISTS["pronoun_matchlist"])),
            (pl.col("dep_rel") == "det")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_51_demonstratives")
    )

    df = (
        ids
        .join(f10, on="doc_id", how="left")
        .join(f14, on="doc_id", how="left")
        .join(f15, on="doc_id", how="left")
        .join(f16_raw, on="doc_id", how="left")
        .join(gerunds_n, on="doc_id", how="left")
        .join(f51, on="doc_id", how="left")
        .with_columns([
            pl.col("f_10_demonstrative_pronoun").fill_null(strategy="zero"),
            pl.col("f_14_nominalizations").fill_null(strategy="zero"),
            pl.col("f_15_gerunds").fill_null(strategy="zero"),
            pl.col("f_16_other_nouns").fill_null(strategy="zero"),
            pl.col("gerunds_n").fill_null(strategy="zero"),
            pl.col("f_51_demonstratives").fill_null(strategy="zero"),
        ])
        .with_columns(
            (
                pl.col("f_16_other_nouns")
                - pl.col("gerunds_n")
                - pl.col("f_14_nominalizations")
            ).alias("f_16_other_nouns")
        )
        .sort("doc_id", descending=False)
    )

    return df.select([
        "f_10_demonstrative_pronoun",
        "f_14_nominalizations",
        "f_15_gerunds",
        "f_16_other_nouns",
        "f_51_demonstratives",
    ])


def _block_adj_and_prepositions(
    tokens: pl.DataFrame,
    ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute prepositions and adjective features together.

    Returns columns:
    - f_39_prepositions
    - f_40_adj_attr
    - f_41_adj_pred
    """
    f39 = (
        tokens
        .filter(pl.col("dep_rel") == "prep")
        .group_by("doc_id", maintain_order=True)
        .len(name="f_39_prepositions")
    )

    f40 = (
        tokens
        .filter(
            (pl.col("pos") == "ADJ")
            & (
                (pl.col("pos_lag_-1") == "NOUN")
                | (pl.col("pos_lag_-1") == "ADJ")
                | (
                    (pl.col("tok_lag_-1") == ",")
                    & (pl.col("pos_lag_-2") == "ADJ")
                )
            )
        )
        .filter(~pl.col("token").str.contains("-"))
        .group_by("doc_id", maintain_order=True)
        .len(name="f_40_adj_attr")
    )

    f41 = (
        tokens
        .filter(
            (pl.col("pos") == "ADJ")
            & (
                (pl.col("pos_lag_1") == "VERB")
                | (pl.col("pos_lag_1") == "AUX")
            )
            & (
                (pl.col("lem_lag_1").is_in(WORDLISTS["linking_matchlist"]))
                & (pl.col("pos_lag_-1") != "NOUN")
                & (pl.col("pos_lag_-1") != "ADJ")
                & (pl.col("pos_lag_-1") != "ADV")
            )
        )
        .filter(~pl.col("token").str.contains("-"))
        .group_by("doc_id", maintain_order=True)
        .len(name="f_41_adj_pred")
    )

    f61 = (
        tokens
        .filter(
            (pl.col("tag") == "IN"),
            (
                (pl.col("dep_rel") == "prep")
                & (pl.col("tag_lag_-1").str.contains("^[[:punct:]]$"))
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_61_stranded_preposition")
    )

    df = (
        ids
        .join(f39, on="doc_id", how="left")
        .join(f40, on="doc_id", how="left")
        .join(f41, on="doc_id", how="left")
        .join(f61, on="doc_id", how="left")
        .with_columns([
            pl.col("f_39_prepositions").fill_null(strategy="zero"),
            pl.col("f_40_adj_attr").fill_null(strategy="zero"),
            pl.col("f_41_adj_pred").fill_null(strategy="zero"),
            pl.col("f_61_stranded_preposition").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )
    return df.select([
        "f_39_prepositions",
        "f_40_adj_attr",
        "f_41_adj_pred",
        "f_61_stranded_preposition",
    ])


def _block_clause_embedding(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute clause/embedding related features in one pass.

    Returns columns:
    - f_21_that_verb_comp
    - f_22_that_adj_comp
    - f_23_wh_clause
    - f_29_that_subj
    - f_30_that_obj
    - f_31_wh_subj
    - f_32_wh_obj
    - f_34_sentence_relatives
    - f_35_because
    - f_38_other_adv_sub
    """
    f21 = (
        tokens
        .filter(
            (pl.col("token") == "that")
            & (pl.col("pos") == "SCONJ")
            & (pl.col("pos_lag_1") == "VERB")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_21_that_verb_comp")
    )

    f22 = (
        tokens
        .filter(
            (pl.col("token") == "that")
            & (pl.col("pos") == "SCONJ")
            & (pl.col("pos_lag_1") == "ADJ")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_22_that_adj_comp")
    )

    f23 = (
        tokens
        .filter(
            pl.col("tag").str.contains("^W")
            & (pl.col("token") != "which")
            & (pl.col("pos_lag_1") == "VERB")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_23_wh_clause")
    )

    f29 = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "that")
            & (pl.col("dep_rel").str.contains("nsubj"))
            & (pl.col("tag_lag_1").str.contains("^N|^CD|DT"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_29_that_subj")
    )

    f30 = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "that")
            & (pl.col("dep_rel").str.contains("dobj"))
            & (pl.col("tag_lag_1").str.contains("^N|^CD|DT"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_30_that_obj")
    )

    f31 = (
        tokens
        .filter(
            pl.col("tag").str.contains("^W")
            & (pl.col("lem_lag_2") != "ask")
            & (pl.col("lem_lag_2") != "tell")
            & (
                pl.col("tag_lag_1").str.contains("^N|^CD|DT")
                | (
                    (pl.col("pos_lag_1") == "PUNCT")
                    & (pl.col("tag_lag_2").str.contains("^N|^CD|DT"))
                    & (pl.col("token") == "who")
                )
            )
        )
        .filter(
            (pl.col("token") != "that")
            & (pl.col("dep_rel").str.contains("nsubj"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_31_wh_subj")
    )

    f32 = (
        tokens
        .filter(
            pl.col("tag").str.contains("^W")
            & (pl.col("lem_lag_2") != "ask")
            & (pl.col("lem_lag_2") != "tell")
            & (
                pl.col("tag_lag_1").str.contains("^N|^CD|DT")
                | (
                    (pl.col("pos_lag_1") == "PUNCT")
                    & (pl.col("tag_lag_2").str.contains("^N|^CD|DT"))
                    & (pl.col("token") == "who")
                )
            )
        )
        .filter(
            (pl.col("token") != "that")
            & (pl.col("dep_rel").str.contains("obj"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_32_wh_obj")
    )

    f34 = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "which")
            & (pl.col("pos_lag_1") == "PUNCT")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_34_sentence_relatives")
    )

    f35 = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "because")
            & (pl.col("tok_lag_-1").str.to_lowercase() != "of")
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_35_because")
    )

    f38 = (
        tokens
        .filter(
            (pl.col("pos") == "SCONJ")
            & (pl.col("dep_rel") == "mark")
            & (pl.col("token").str.to_lowercase() != "because")
            & (pl.col("token").str.to_lowercase() != "if")
            & (pl.col("token").str.to_lowercase() != "unless")
            & (pl.col("token").str.to_lowercase() != "though")
            & (pl.col("token").str.to_lowercase() != "although")
            & (pl.col("token").str.to_lowercase() != "tho")
        )
        .filter(
            ~(
                (pl.col("token").str.to_lowercase() == "that")
                & (pl.col("dep_lag_1") != "ADV")
            )
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_38_other_adv_sub")
    )

    f60 = (
        tokens
        .filter(
            (pl.col("lemma").is_in(WORDLISTS["verb_matchlist"])),
            (pl.col("pos") == "VERB"),
            (
                (
                    (pl.col("dep_lag_-1") == "nsubj")
                    & (pl.col("pos_lag_-2") == "VERB")
                    & (pl.col("tag_lag_-1") != "WP")
                    & (pl.col("tag_lag_-2") != "VBG")
                )
                | (
                    (pl.col("tag_lag_-1") == "DT")
                    & (pl.col("dep_lag_-2") == "nsubj")
                    & (pl.col("pos_lag_-3") == "VERB")
                )
                | (
                    (pl.col("tag_lag_-1") == "DT")
                    & (pl.col("dep_lag_-2") == "amod")
                    & (pl.col("dep_lag_-3") == "nsubj")
                    & (pl.col("pos_lag_-4") == "VERB")
                )
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_60_that_deletion")
    )

    # Join to ids and zero-fill
    df = (
        ids
        .join(f21, on="doc_id", how="left")
        .join(f22, on="doc_id", how="left")
        .join(f23, on="doc_id", how="left")
        .join(f29, on="doc_id", how="left")
        .join(f30, on="doc_id", how="left")
        .join(f31, on="doc_id", how="left")
        .join(f32, on="doc_id", how="left")
        .join(f34, on="doc_id", how="left")
        .join(f35, on="doc_id", how="left")
        .join(f38, on="doc_id", how="left")
        .join(f60, on="doc_id", how="left")
        .with_columns([
            pl.col("f_21_that_verb_comp").fill_null(strategy="zero"),
            pl.col("f_22_that_adj_comp").fill_null(strategy="zero"),
            pl.col("f_23_wh_clause").fill_null(strategy="zero"),
            pl.col("f_29_that_subj").fill_null(strategy="zero"),
            pl.col("f_30_that_obj").fill_null(strategy="zero"),
            pl.col("f_31_wh_subj").fill_null(strategy="zero"),
            pl.col("f_32_wh_obj").fill_null(strategy="zero"),
            pl.col("f_34_sentence_relatives").fill_null(strategy="zero"),
            pl.col("f_35_because").fill_null(strategy="zero"),
            pl.col("f_38_other_adv_sub").fill_null(strategy="zero"),
            pl.col("f_60_that_deletion").fill_null(strategy="zero"),
        ])
        # Guard: avoid double-counting WH clauses and WH relatives
        .with_columns(
            (
                pl.col("f_23_wh_clause")
                - pl.col("f_31_wh_subj")
                - pl.col("f_32_wh_obj")
            ).alias("_f23_adj")
        )
        .with_columns(
            pl.when(pl.col("_f23_adj") > 0)
            .then(pl.col("_f23_adj"))
            .otherwise(0)
            .alias("f_23_wh_clause")
        )
        .drop("_f23_adj")
        .sort("doc_id", descending=False)
    )

    return df.select([
        "f_21_that_verb_comp",
        "f_22_that_adj_comp",
        "f_23_wh_clause",
        "f_29_that_subj",
        "f_30_that_obj",
        "f_31_wh_subj",
        "f_32_wh_obj",
        "f_34_sentence_relatives",
        "f_35_because",
        "f_38_other_adv_sub",
        "f_60_that_deletion",
    ])


def _block_sentence_level(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute sentence-level features together (B5).

    Currently includes:
    - f_13_wh_question
    """
    f13 = (
        tokens
        .filter((pl.col("tag").str.contains("^W")) & (pl.col("pos") != "DET"))
        .filter(
            pl.col("sentence_id").is_in(
                tokens.filter(pl.col("token") == "?").get_column("sentence_id")
            )
            | (pl.col("dep_lag_-1") == "aux")
            | ((pl.col("token_id") < 3) & (pl.col("pos_lag_1") == "AUX"))
            | (
                pl.col("sentence_id").is_in(
                    tokens
                    .filter(pl.col("pos") == "AUX")
                    .get_column("sentence_id")
                )
                & (~pl.col("dep_lag_1").is_in(["ccomp", "advcl"]))
            )
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_13_wh_question")
    )

    return _ensure_full_doc_index_multi(
        f13,
        ids,
        ["f_13_wh_question"],
    )


def _block_regex_features(tokens: pl.DataFrame) -> pl.DataFrame:
    """Compute regex-based Biber features from token/tag sequences.

    This reproduces the legacy regex counting over concatenated token_tag
    blobs per document. Returns a DataFrame with doc_id and all regex
    feature columns (f_01, f_03, f_04, ..., as defined in FEATURES).
    """
    biber_tkns = (
        tokens
        .filter((pl.col("token") != " ") & (pl.col("tag") != "_SP"))
        # generic tag for punctuation
        .with_columns(
            pl.when(pl.col("dep_rel") == "punct")
            .then(pl.lit("_punct"))
            .otherwise(pl.col("token"))
            .alias("token")
        )
        .with_columns(
            pl.when(pl.col("dep_rel") == "punct")
            .then(pl.lit(""))
            .otherwise(pl.col("tag"))
            .alias("tag")
        )
        # replace ampersand used as CC with "and"
        .with_columns(
            pl.when((pl.col("token") == "&") & (pl.col("tag") == "CC"))
            .then(pl.lit("and"))
            .otherwise(pl.col("token"))
            .alias("token")
        )
        # join tokens and tags and collapse per doc
        .with_columns(
            pl.concat_str(
                [
                    pl.col("token").str.to_lowercase(),
                    pl.col("tag").str.to_lowercase(),
                ],
                separator="_",
            ).alias("tokens")
        )
        .select(["doc_id", "tokens"])
        .group_by("doc_id", maintain_order=True)
        .agg(pl.col("tokens").str.concat(" "))
    )

    counts: list[pl.DataFrame] = []
    for row in biber_tkns.iter_rows(named=True):
        feature_counts: dict[str, int] = {}
        token_blob = row["tokens"]
        for key, pattern in FEATURE_PATTERNS.items():
            feature_counts[key] = sum(1 for _ in pattern.finditer(token_blob))
        df = pl.from_dict(feature_counts)
        df.insert_column(0, pl.Series("doc_id", [row["doc_id"]]))
        counts.append(df)

    base_regex = (
        pl.concat(counts)
        .sort("doc_id", descending=False)
        .with_columns(pl.all().exclude("doc_id").cast(pl.UInt32, strict=True))
    )
    return base_regex


def _block_derived_metrics(
        tokens: pl.DataFrame,
        ids: pl.DataFrame,
        use_ttr: bool,
) -> pl.DataFrame:
    """Compute derived lexical metrics together.

    Returns columns:
    - f_43_type_token (TTR or MATTR depending on use_ttr)
    - f_44_mean_word_length
    """
    if not use_ttr:
        f43 = (
            tokens
            .filter(
                pl.col("token")
                .str.to_lowercase()
                .str.contains("^[a-z]+$")
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token").str.to_lowercase(),
                        pl.col("tag").str.to_lowercase(),
                    ],
                    separator="_",
                ).alias("token")
            )
            .select(["doc_id", "token"])
            .rolling(
                pl.int_range(pl.len()).alias("index"),
                period="100i",
                group_by="doc_id",
            )
            .agg(
                pl.when(pl.len() == 100)
                .then(pl.col("token").n_unique().truediv(100))
                .alias("f_43_type_token"),
            )
            .drop("index")
            .group_by("doc_id", maintain_order=True)
            .agg(pl.col("f_43_type_token").mean())
            .select(["doc_id", "f_43_type_token"])
        )
    else:
        f43 = (
            tokens
            .filter(
                pl.col("token")
                .str.to_lowercase()
                .str.contains("^[a-z]+$")
            )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token").str.to_lowercase(),
                        pl.col("tag").str.to_lowercase(),
                    ],
                    separator="_",
                ).alias("token")
            )
            .select(["doc_id", "token"])
            .group_by("doc_id", maintain_order=True)
            .agg(
                pl.col("token").n_unique().truediv(pl.col("token").len())
                .alias("f_43_type_token")
            )
            .select(["doc_id", "f_43_type_token"])
        )

    f44 = (
        tokens
        .with_columns(pl.col("token").str.to_lowercase())
        .filter(pl.col("token").str.contains("^[a-z]+$"))
        .select(["doc_id", "token"])
        .with_columns(
            pl.col("token")
            .str.len_chars()
            .alias("f_44_mean_word_length")
        )
        .group_by("doc_id", maintain_order=True)
        .agg(pl.col("f_44_mean_word_length").mean())
        .select(["doc_id", "f_44_mean_word_length"])
    )

    df = (
        ids
        .join(f43, on="doc_id", how="left")
        .join(f44, on="doc_id", how="left")
        .with_columns([
            pl.col("f_43_type_token").fill_null(strategy="zero"),
            pl.col("f_44_mean_word_length").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )

    return df.select(["f_43_type_token", "f_44_mean_word_length"])


def _block_passive_voice(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute passive and stative verb features together.

    Returns columns:
    - f_17_agentless_passives
    - f_18_by_passives
    - f_19_be_main_verb
    """
    f17 = (
        tokens
        .filter(
            (pl.col("dep_rel") == "auxpass")
            & (
                (
                    pl.col("tok_lag_-2").is_null()
                    | (pl.col("tok_lag_-2") != "by")
                )
                & (
                    pl.col("tok_lag_-3").is_null()
                    | (pl.col("tok_lag_-3") != "by")
                )
            )
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_17_agentless_passives")
    )

    f18 = (
        tokens
        .filter(
            (pl.col("dep_rel") == "auxpass")
            & (
                (pl.col("tok_lag_-2") == "by")
                | (pl.col("tok_lag_-3") == "by")
            )
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_18_by_passives")
    )

    f19 = (
        tokens
        .filter(
            (pl.col("lemma") == "be")
            & (~pl.col("dep_rel").str.contains("aux"))
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_19_be_main_verb")
    )

    df = (
        ids
        .join(f17, on="doc_id", how="left")
        .join(f18, on="doc_id", how="left")
        .join(f19, on="doc_id", how="left")
        .with_columns([
            pl.col("f_17_agentless_passives").fill_null(strategy="zero"),
            pl.col("f_18_by_passives").fill_null(strategy="zero"),
            pl.col("f_19_be_main_verb").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )

    return df.select([
        "f_17_agentless_passives",
        "f_18_by_passives",
        "f_19_be_main_verb",
    ])


def _block_participial_clauses(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute participial/adverbial and whiz/reduced relative features.

    Returns columns:
    - f_25_present_participle
    - f_26_past_participle
    - f_27_past_participle_whiz
    - f_28_present_participle_whiz
    """
    f25 = (
        tokens
        .filter(
            (pl.col("tag") == "VBG")
            & ((pl.col("dep_rel") == "advcl") | (pl.col("dep_rel") == "ccomp"))
        )
        .filter(pl.col("dep_lag_1") == "punct")
        .group_by("doc_id", maintain_order=True)
        .len(name="f_25_present_participle")
    )

    f26 = (
        tokens
        .filter(
            (pl.col("tag") == "VBN")
            & ((pl.col("dep_rel") == "advcl") | (pl.col("dep_rel") == "ccomp"))
        )
        .filter(pl.col("dep_lag_1") == "punct")
        .group_by("doc_id", maintain_order=True)
        .len(name="f_26_past_participle")
    )

    f27 = (
        tokens
        .filter(
            (pl.col("tag") == "VBN"),
            (pl.col("dep_rel") == "acl"),
            (pl.col("pos_lag_1") == "NOUN"),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_27_past_participle_whiz")
    )

    f28 = (
        tokens
        .filter(
            (pl.col("tag") == "VBG"),
            (pl.col("dep_rel") == "acl"),
            (pl.col("pos_lag_1") == "NOUN"),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_28_present_participle_whiz")
    )

    df = (
        ids
        .join(f25, on="doc_id", how="left")
        .join(f26, on="doc_id", how="left")
        .join(f27, on="doc_id", how="left")
        .join(f28, on="doc_id", how="left")
        .with_columns([
            pl.col("f_25_present_participle").fill_null(strategy="zero"),
            pl.col("f_26_past_participle").fill_null(strategy="zero"),
            pl.col("f_27_past_participle_whiz").fill_null(strategy="zero"),
            pl.col("f_28_present_participle_whiz").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )

    return df.select([
        "f_25_present_participle",
        "f_26_past_participle",
        "f_27_past_participle_whiz",
        "f_28_present_participle_whiz",
    ])


def _block_split_constructions(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute split infinitive and split auxiliary features together.

    Returns columns:
    - f_62_split_infinitive
    - f_63_split_auxiliary
    """
    f62 = (
        tokens
        .filter(
            (pl.col("tag") == "TO"),
            (
                (pl.col("tag_lag_-1") == "RB")
                & (pl.col("tag_lag_-2") == "VB")
            )
            | (
                (pl.col("tag_lag_-1") == "RB")
                & (pl.col("tag_lag_-2") == "RB")
                & (pl.col("tag_lag_-3") == "VB")
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_62_split_infinitive")
    )

    f63 = (
        tokens
        .filter(
            (pl.col("dep_rel").str.contains("aux")),
            (
                (pl.col("pos_lag_-1") == "ADV")
                & (pl.col("pos_lag_-2") == "VERB")
            )
            | (
                (pl.col("pos_lag_-1") == "ADV")
                & (pl.col("pos_lag_-2") == "ADV")
                & (pl.col("pos_lag_-3") == "VERB")
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_63_split_auxiliary")
    )

    df = (
        ids
        .join(f62, on="doc_id", how="left")
        .join(f63, on="doc_id", how="left")
        .with_columns([
            pl.col("f_62_split_infinitive").fill_null(strategy="zero"),
            pl.col("f_63_split_auxiliary").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )
    return df.select([
        "f_62_split_infinitive",
        "f_63_split_auxiliary",
    ])


def _block_coordination(
        tokens: pl.DataFrame,
        ids: pl.DataFrame
) -> pl.DataFrame:
    """Compute phrasal and clausal coordination features together.

    Returns columns:
    - f_64_phrasal_coordination
    - f_65_clausal_coordination
    """
    f64 = (
        tokens
        .filter(
            (pl.col("tag") == "CC"),
            (
                (pl.col("pos_lag_-1") == "NOUN")
                & (pl.col("pos_lag_1") == "NOUN")
            )
            | (
                (pl.col("pos_lag_-1") == "VERB")
                & (pl.col("pos_lag_1") == "VERB")
            )
            | (
                (pl.col("pos_lag_-1") == "ADJ")
                & (pl.col("pos_lag_1") == "ADJ")
            )
            | (
                (pl.col("pos_lag_-1") == "ADV")
                & (pl.col("pos_lag_1") == "ADV")
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_64_phrasal_coordination")
    )

    f65 = (
        tokens
        .filter(
            (pl.col("tag") == "CC"),
            (pl.col("dep_rel") != "ROOT"),
            (
                (pl.col("dep_lag_-1") == "nsubj")
                | (pl.col("dep_lag_-2") == "nsubj")
                | (pl.col("dep_lag_-3") == "nsubj")
            ),
        )
        .group_by("doc_id", maintain_order=True)
        .len(name="f_65_clausal_coordination")
    )

    df = (
        ids
        .join(f64, on="doc_id", how="left")
        .join(f65, on="doc_id", how="left")
        .with_columns([
            pl.col("f_64_phrasal_coordination").fill_null(strategy="zero"),
            pl.col("f_65_clausal_coordination").fill_null(strategy="zero"),
        ])
        .sort("doc_id", descending=False)
    )
    return df.select([
        "f_64_phrasal_coordination",
        "f_65_clausal_coordination",
    ])


def _biber_weight(
        biber_counts: pl.DataFrame,
        totals: pl.DataFrame,
        scheme="prop"
 ) -> pl.DataFrame:

    if (
        not all(
            x == pl.UInt32 for x in biber_counts.collect_schema().dtypes()[1:]
            ) and
        biber_counts.columns[0] != "doc_id"
    ):
        raise ValueError("""
                         Invalid DataFrame.
                         Expected a DataFrame produced by biber.
                         """)

    scheme_types = ['prop', 'scale', 'tfidf']
    if scheme not in scheme_types:
        raise ValueError("""scheme_types
                         Invalid count_by type. Expected one of: %s
                         """ % scheme_types)

    dtm = biber_counts.join(totals, on="doc_id")

    # Normalize per 1000 tokens (exclude derived metrics and doc_total)
    weighted_df = (
        dtm.with_columns(
            pl.selectors.numeric()
            .exclude(['f_43_type_token', 'f_44_mean_word_length', 'doc_total'])
            .truediv(pl.col("doc_total"))
            .mul(1000)
        )
        .drop("doc_total")
    )

    if scheme == "prop":
        return weighted_df.sort("doc_id", descending=False)
    elif scheme == "scale":
        weighted_df = (
            weighted_df
            .with_columns(
                pl.selectors.numeric()
                .sub(pl.selectors.numeric().mean())
                .truediv(pl.selectors.numeric().std())
            )
        )
        return weighted_df.sort("doc_id", descending=False)
    else:  # scheme == 'tfidf'
        drop_cols = [
            c for c in ["f_43_type_token", "f_44_mean_word_length"]
            if c in weighted_df.columns
        ]
        tfidf_df = weighted_df.drop(drop_cols) if drop_cols else weighted_df
        tfidf_df = (
            tfidf_df
            .transpose(include_header=True,
                       header_name="Tag",
                       column_names="doc_id")
            .with_columns(
                (
                    (pl.sum_horizontal(pl.selectors.numeric().ge(0)) +
                     pl.sum_horizontal(pl.selectors.numeric().gt(0)))
                    .log1p()
                    - pl.sum_horizontal(pl.selectors.numeric().gt(0)).log1p()
                ).alias("IDF")
            )
            .with_columns(
                pl.selectors.numeric().exclude("IDF").mul(pl.col("IDF"))
            )
            .drop("IDF")
            .transpose(include_header=True,
                       header_name="doc_id",
                       column_names="Tag")
        )
        for col in ["f_43_type_token", "f_44_mean_word_length", "doc_total"]:
            if col in tfidf_df.columns:
                tfidf_df = tfidf_df.drop(col)
        logger.info(
            "Excluded from tf-idf matrix: "
            "f_43_type_token and f_44_mean_word_length"
        )
        return tfidf_df.sort("doc_id", descending=False)


def biber(
    tokens: pl.DataFrame,
    normalize: Optional[bool] = True,
    force_ttr: Optional[bool] = False
) -> pl.DataFrame:
    """Extract Biber features from a parsed corpus.

    Parameters
    ----------
    tokens:
        A polars DataFrame
        with the output of the spacy_parse function.
    normalize:
        Normalize counts per 1000 tokens.
    force_ttr:
        Force the calcuation of type-token ratio
        rather than moving average type-token ratio.

    Returns
    -------
    pl.DataFrame
        A polars DataFrame with,
        counts of feature frequencies.

    Notes
    -----
    MATTR is the default as it is less sensitive than TTR
    to variations in text lenghth. However, the
    function will automatically use TTR if any of the
    corpus texts are less than 200 words.
    Thus, forcing TTR can be necessary when processing multiple
    corpora that you want to be consistent.

    """
    doc_totals = (
        tokens
        .filter(
            ~(pl.col("token").str.contains("^[[:punct:]]+$"))
            )
        .group_by("doc_id", maintain_order=True)
        .len(name="doc_total")
        )

    doc_len_min = (
        tokens
        .filter(pl.col("token")
                .str.to_lowercase().str.contains("^[a-z]+$"))
        ).group_by("doc_id", maintain_order=True
                   ).agg(
                       pl.col("token").len()
                       ).min().get_column("token").item()

    if doc_len_min > 200 and force_ttr is False:
        force_ttr = False
        logger.info("Using MATTR for f_43_type_token")
    else:
        force_ttr = True
        logger.info("Using TTR for f_43_type_token")

    ids = tokens.select("doc_id").unique()
    regex_counts = _block_regex_features(tokens)

    # add lead/lag columns for feature detection (single pass)
    tokens = (
            tokens
            .with_columns([
                    # dep_rel lags
                    *(
                        pl.col("dep_rel")
                        .shift(i, fill_value="punct")
                        .over("doc_id")
                        .alias(f"dep_lag_{i}")
                        for i in range(-3, 1 + 1)
                    ),
                    # lemma lags
                    *(
                        pl.col("lemma")
                        .shift(i, fill_value="punct")
                        .over("doc_id")
                        .alias(f"lem_lag_{i}")
                        for i in range(0, 2 + 1)
                    ),
                    # pos lags
                    *(
                        pl.col("pos")
                        .shift(i)
                        .over("doc_id")
                        .alias(f"pos_lag_{i}")
                        for i in range(-4, 2 + 1)
                    ),
                    # tag lags
                    *(
                        pl.col("tag")
                        .shift(i, fill_value="PUNCT")
                        .over("doc_id")
                        .alias(f"tag_lag_{i}")
                        for i in range(-3, 2 + 1)
                    ),
                    # token lags
                    *(
                        pl.col("token")
                        .shift(i)
                        .over("doc_id")
                        .alias(f"tok_lag_{i}")
                        for i in range(-3, 1 + 1)
                    ),
            ])
            .drop(pl.selectors.contains("_lag_0"))
    )

    # Aux/tense block (reused across features)
    _aux_block = _block_aux_tense(tokens, ids)
    f_02_perfect_aspect = _aux_block.select("f_02_perfect_aspect")

    # Lexical membership block (reused)
    _lex_block = _block_lexical_membership(tokens, ids)
    f_10_demonstrative_pronoun = _lex_block.select(
        "f_10_demonstrative_pronoun"
    )

    f_12_proverb_do = _aux_block.select("f_12_proverb_do")

    # Sentence-level block (B5)
    _sent_block = _block_sentence_level(tokens, ids)
    f_13_wh_question = _sent_block.select("f_13_wh_question")

    # f_15 and f_16 come from lexical membership block (f_16 depends on f_14)
    f_14_nominalizations = _lex_block.select("f_14_nominalizations")
    f_15_gerunds = _lex_block.select("f_15_gerunds")
    f_16_other_nouns = _lex_block.select("f_16_other_nouns")
    f_51_demonstratives = _lex_block.select("f_51_demonstratives")

    # Clause/embedding block
    _clause_block = _block_clause_embedding(tokens, ids)
    f_21_that_verb_comp = _clause_block.select("f_21_that_verb_comp")
    f_22_that_adj_comp = _clause_block.select("f_22_that_adj_comp")
    f_23_wh_clause = _clause_block.select("f_23_wh_clause")
    f_29_that_subj = _clause_block.select("f_29_that_subj")
    f_30_that_obj = _clause_block.select("f_30_that_obj")
    f_31_wh_subj = _clause_block.select("f_31_wh_subj")
    f_32_wh_obj = _clause_block.select("f_32_wh_obj")
    f_34_sentence_relatives = _clause_block.select("f_34_sentence_relatives")
    f_35_because = _clause_block.select("f_35_because")
    f_38_other_adv_sub = _clause_block.select("f_38_other_adv_sub")
    f_60_that_deletion = _clause_block.select("f_60_that_deletion")

    # Adjective and preposition block (combined features)
    _adj_prep_block = _block_adj_and_prepositions(tokens, ids)
    f_39_prepositions = _adj_prep_block.select("f_39_prepositions")
    f_40_adj_attr = _adj_prep_block.select("f_40_adj_attr")
    f_41_adj_pred = _adj_prep_block.select("f_41_adj_pred")
    f_61_stranded_preposition = _adj_prep_block.select(
        "f_61_stranded_preposition"
    )

    # Passive voice and stative verb features
    _passive_block = _block_passive_voice(tokens, ids)
    f_17_agentless_passives = _passive_block.select("f_17_agentless_passives")
    f_18_by_passives = _passive_block.select("f_18_by_passives")
    f_19_be_main_verb = _passive_block.select("f_19_be_main_verb")

    # Participial/adverbial clauses and reduced relatives
    _part_block = _block_participial_clauses(tokens, ids)
    f_25_present_participle = _part_block.select("f_25_present_participle")
    f_26_past_participle = _part_block.select("f_26_past_participle")
    f_27_past_participle_whiz = _part_block.select("f_27_past_participle_whiz")
    f_28_present_participle_whiz = _part_block.select(
        "f_28_present_participle_whiz"
    )

    # Derived metrics block (f_43, f_44)
    _derived_block = _block_derived_metrics(tokens, ids, use_ttr=force_ttr)
    f_43_type_token = _derived_block.select("f_43_type_token")
    f_44_mean_word_length = _derived_block.select("f_44_mean_word_length")

    # Split constructions
    _split_block = _block_split_constructions(tokens, ids)
    f_62_split_infinitive = _split_block.select("f_62_split_infinitive")
    f_63_split_auxiliary = _split_block.select("f_63_split_auxiliary")

    # Coordination
    _coord_block = _block_coordination(tokens, ids)
    f_64_phrasal_coordination = _coord_block.select(
        "f_64_phrasal_coordination"
    )
    f_65_clausal_coordination = _coord_block.select(
        "f_65_clausal_coordination"
    )

    blocks = [
        regex_counts,
        f_02_perfect_aspect,
        f_10_demonstrative_pronoun,
        f_12_proverb_do,
        f_13_wh_question,
        f_14_nominalizations,
        f_15_gerunds,
        f_16_other_nouns,
        f_17_agentless_passives,
        f_18_by_passives,
        f_19_be_main_verb,
        f_21_that_verb_comp,
        f_22_that_adj_comp,
        f_23_wh_clause,
        f_25_present_participle,
        f_26_past_participle,
        f_27_past_participle_whiz,
        f_28_present_participle_whiz,
        f_29_that_subj,
        f_30_that_obj,
        f_31_wh_subj,
        f_32_wh_obj,
        f_34_sentence_relatives,
        f_35_because,
        f_38_other_adv_sub,
        f_39_prepositions,
        f_40_adj_attr,
        f_41_adj_pred,
        f_43_type_token,
        f_44_mean_word_length,
        f_51_demonstratives,
        f_60_that_deletion,
        f_61_stranded_preposition,
        f_62_split_infinitive,
        f_63_split_auxiliary,
        f_64_phrasal_coordination,
        f_65_clausal_coordination,
    ]

    biber_counts = pl.concat(blocks, how="horizontal")

    biber_counts = biber_counts.select(sorted(biber_counts.columns))

    if normalize is False:
        return biber_counts

    if normalize is True:
        logger.info(
            "All features normalized per 1000 tokens except: "
            "f_43_type_token and f_44_mean_word_length"
        )
        biber_counts = _biber_weight(biber_counts,
                                     totals=doc_totals,
                                     scheme="prop")
        return biber_counts
