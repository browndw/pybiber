"""
Extract Biber features from a corpus parsed and annotated by spaCy.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import re
from typing import Optional
from textwrap import dedent

import polars as pl

from .biber_dict import FEATURES, WORDLISTS


def _biber_weight(biber_counts: pl.DataFrame,
                  totals: pl.DataFrame,
                  scheme="prop"):

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

    weighted_df = (
        dtm
        .with_columns(
            pl.selectors.numeric().exclude(
                ['f_43_type_token',
                 'f_44_mean_word_length',
                 'doc_total']
                 ).truediv(
                     pl.col("doc_total")
                 ).mul(1000)
        )
        .drop("doc_total")
    )

    if scheme == "prop":
        print(dedent(
            """
            All features normalized per 1000 tokens except:
            f_43_type_token and f_44_mean_word_length
            """
            ))
        return weighted_df

    elif scheme == "scale":
        weighted_df = (
            weighted_df
            .with_columns(
                pl.selectors.numeric()
                .sub(
                    pl.selectors.numeric().mean()
                    )
                .truediv(
                    pl.selectors.numeric().std()
                    )
                )
        )
        return weighted_df

    else:
        weighted_df = (
            weighted_df
            .drop(['f_43_type_token', 'f_44_mean_word_length'])
            .transpose(include_header=True,
                       header_name="Tag",
                       column_names="doc_id")
            # log(1 + N/(1+df)) = log((1+df+N)/(1+df)) =
            # log(1+df+N) - log(1+df) = log1p(df+N) - log1p(df)
            .with_columns(
                pl.sum_horizontal(pl.selectors.numeric().ge(0))
                .add(pl.sum_horizontal(pl.selectors.numeric().gt(0))).log1p()
                .sub(pl.sum_horizontal(pl.selectors.numeric().gt(0)).log1p())
                .alias("IDF")
            )
            # multiply normalized frequencies by IDF
            .with_columns(
                pl.selectors.numeric().exclude("IDF").mul(pl.col("IDF"))
            )
            .drop("IDF")
            .transpose(include_header=True,
                       header_name="doc_id",
                       column_names="Tag")
            )
        print(dedent(
            """
            Excluded from tf-idf matrix:
            f_43_type_token and f_44_mean_word_length
            """))
        return weighted_df


def biber(tokens: pl.DataFrame,
          normalize: Optional[bool] = True,
          force_ttr: Optional[bool] = False) -> pl.DataFrame:
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
        print("Using MATTR for f_43_type_token")
    else:
        force_ttr = True
        print("Using TTR for f_43_type_token")

    ids = tokens.select("doc_id").unique()
    biber_tkns = (
        tokens
        .filter(
            (pl.col("token") != " ") & (pl.col("tag") != "_SP")
            )
        # create generic tag for punctuation
        .with_columns(
                pl.when(
                    pl.col("dep_rel") == "punct"
                )
                .then(pl.lit("_punct"))
                .otherwise(pl.col("token"))
                .alias("token")
                )
        .with_columns(
                pl.when(
                    pl.col("dep_rel") == "punct"
                )
                .then(pl.lit(""))
                .otherwise(pl.col("tag"))
                .alias("tag")
                )
        # replace ampersand
        .with_columns(
                pl.when(
                    (pl.col("token") == "&") &
                    (pl.col("tag") == "CC")
                )
                .then(pl.lit("and"))
                .otherwise(pl.col("token"))
                .alias("token")
                )
        # join tokens and tags and collapse
        .with_columns(
            pl.concat_str(
                [
                    pl.col("token").str.to_lowercase(),
                    pl.col("tag").str.to_lowercase(),
                ],
                separator="_",
                ).alias("tokens")
                ).select(["doc_id", "tokens"])
        .group_by(
            "doc_id", maintain_order=True
            ).agg(pl.col("tokens").str.concat(" "))
        )

    counts = []
    for row in biber_tkns.iter_rows(named=True):
        feature_counts = {}
        for key in FEATURES.keys():
            pattern = re.compile('|'.join(FEATURES[key]))
            count = len(pattern.findall(row['tokens']))
            feature_counts[key] = count
        df = pl.from_dict(feature_counts)
        s = pl.Series("doc_id", [row['doc_id']])
        df.insert_column(0, s)
        counts.append(df)

    biber_1 = pl.concat(counts).sort(
        "doc_id", descending=False
        ).with_columns(
            pl.all().exclude("doc_id").cast(pl.UInt32, strict=True)
            )

    # add lead/lag columns for feature detection
    tokens = (
        tokens
        .with_columns(
            pl.col("dep_rel")
            .shift(i, fill_value="punct").over("doc_id").alias(
                f"dep_lag_{i}") for i in range(
                    -3, 1 + 1)
            )
        .with_columns(
            pl.col("lemma")
            .shift(i, fill_value="punct").over("doc_id").alias(
                f"lem_lag_{i}") for i in range(
                    0, 2 + 1)
            )
        .with_columns(
            pl.col("pos")
            .shift(i).over("doc_id").alias(
                f"pos_lag_{i}") for i in range(
                    -4, 2 + 1)
            )
        .with_columns(
            pl.col("tag")
            .shift(i, fill_value="PUNCT").over("doc_id").alias(
                f"tag_lag_{i}") for i in range(
                    -3, 2 + 1)
            )
        .with_columns(
            pl.col("token")
            .shift(i).over("doc_id").alias(
                f"tok_lag_{i}") for i in range(
                    -3, 1 + 1)
            )
        .drop(pl.selectors.contains("_lag_0"))
        )

    f_02_perfect_aspect = (
        tokens
        .filter(
            (pl.col("lemma") == "have") &
            (pl.col("dep_rel").str.contains("aux"))
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_02_perfect_aspect")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_02_perfect_aspect").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_02_perfect_aspect")
    )

    f_10_demonstrative_pronoun = (
        tokens
        .with_columns(
                    pl.col("tag")
                    .shift(1).over("doc_id")
                    .alias("tag_1")
                )
        .filter(
            (pl.col("tag") == "DT") &
            (~pl.col("tag_1").str.contains("^N|^CD|DT")) &
            (pl.col("dep_rel").str.contains("nsubj|dobj|pobj"))
        )
        .filter(
            pl.col("token").str.to_lowercase()
            .is_in(WORDLISTS['pronoun_matchlist'])
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_10_demonstrative_pronoun")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_10_demonstrative_pronoun").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_10_demonstrative_pronoun")
        )

    f_12_proverb_do = (
        tokens
        .filter(
            (pl.col("lemma") == "do") &
            (~pl.col("dep_rel").str.contains("aux"))
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_12_proverb_do")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_12_proverb_do").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_12_proverb_do")
    )

    f_13_wh_question = (
        tokens
        .filter(
            (pl.col("tag").str.contains("^W")) &
            (pl.col("pos") != "DET") &
            (pl.col("dep_lag_-1") == "aux")
        )
        .filter(
            (pl.col("pos_lag_1") == "PUNCT") |
            (pl.col("pos_lag_2") == "PUNCT")
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_13_wh_question")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_13_wh_question").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_13_wh_question")
    )

    f_14_nominalizations = (
        tokens
        .filter(
            (pl.col("pos") == "NOUN") &
            (pl.col("token").str.to_lowercase().str.contains(
                "tion$|tions$|ment$|ments$|ness$|nesses$|ity$|ities$"
                ))
        )
        .filter(
            ~pl.col("token").str.to_lowercase()
            .is_in(WORDLISTS['nominalization_stoplist'])
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_14_nominalizations")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_14_nominalizations").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_14_nominalizations")
        )

    f_15_gerunds = (
        tokens
        .filter(
            pl.col("token").str.to_lowercase().str.contains(
                "ing$|ings$"
                ) &
            pl.col("dep_rel").str.contains("nsub|dobj|pobj")
            )
        .filter(
            ~pl.col("token").str.to_lowercase()
            .is_in(WORDLISTS['gerund_stoplist'])
            )
        )

    gerunds_n = (
        f_15_gerunds
        .filter(
            pl.col("pos") == "NOUN"
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="gerunds_n")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("gerunds_n").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("gerunds_n")
        )

    f_15_gerunds = (
        f_15_gerunds
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_15_gerunds")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_15_gerunds").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_15_gerunds")
        )

    f_16_other_nouns = (
        tokens
        .filter(
            (pl.col("pos") == "NOUN") |
            (pl.col("pos") == "PROPN")
        )
        .filter(
            ~pl.col("token").str.contains("-")
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_16_other_nouns")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_16_other_nouns").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .with_columns(
            gerunds_n=gerunds_n.to_series()
        )
        .with_columns(
            nominalizations_n=f_14_nominalizations.to_series()
        )
        .with_columns(
            (
                pl.col("f_16_other_nouns") -
                pl.col("gerunds_n") -
                pl.col("nominalizations_n")
                )
            .alias("f_16_other_nouns")
        )
        .select("f_16_other_nouns")
    )

    f_17_agentless_passives = (
        tokens
        .filter(
            (pl.col("dep_rel") == "auxpass") &
            (
                (pl.col("tok_lag_-2") != "by") |
                (pl.col("tok_lag_-3") != "by")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_17_agentless_passives")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_17_agentless_passives").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_17_agentless_passives")
    )

    f_18_by_passives = (
        tokens
        .filter(
            (pl.col("dep_rel") == "auxpass") &
            (
                (pl.col("tok_lag_-2") == "by") |
                (pl.col("tok_lag_-3") == "by")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_18_by_passives")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_18_by_passives").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_18_by_passives")
    )

    f_19_be_main_verb = (
        tokens
        .filter(
            (pl.col("lemma") == "be") &
            (~pl.col("dep_rel").str.contains("aux"))
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_19_be_main_verb")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_19_be_main_verb").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_19_be_main_verb")
    )

    f_21_that_verb_comp = (
        tokens
        .filter(
            (pl.col("token") == "that"),
            (pl.col("pos") == "SCONJ"),
            (pl.col("pos_lag_1") == "VERB")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_21_that_verb_comp")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_21_that_verb_comp").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_21_that_verb_comp")
    )

    f_22_that_adj_comp = (
        tokens
        .filter(
            (pl.col("token") == "that"),
            (pl.col("pos") == "SCONJ"),
            (pl.col("pos_lag_1") == "ADJ")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_22_that_adj_comp")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_22_that_adj_comp").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_22_that_adj_comp")
    )

    f_23_wh_clause = (
        tokens
        .filter(
            (pl.col("tag").str.contains("^W")),
            (pl.col("token") != "which"),
            (pl.col("pos_lag_1") == "VERB")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_23_wh_clause")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_23_wh_clause").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_23_wh_clause")
    )

    f_25_present_participle = (
        tokens
        .filter(
            (pl.col("tag") == "VBG") &
            (
                (pl.col("dep_rel") == "advcl") |
                (pl.col("dep_rel") == "ccomp")
            )
            )
        .filter(
            pl.col("dep_lag_1") == "punct"
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_25_present_participle")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_25_present_participle").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_25_present_participle")
    )

    f_26_past_participle = (
        tokens
        .filter(
            (pl.col("tag") == "VBN") &
            (
                (pl.col("dep_rel") == "advcl") |
                (pl.col("dep_rel") == "ccomp")
            )
            )
        .filter(
            pl.col("dep_lag_1") == "punct"
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_26_past_participle")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_26_past_participle").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_26_past_participle")
    )

    f_27_past_participle_whiz = (
        tokens
        .filter(
            (pl.col("tag") == "VBN"),
            (pl.col("dep_rel") == "acl"),
            (pl.col("pos_lag_1") == "NOUN")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_27_past_participle_whiz")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_27_past_participle_whiz").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_27_past_participle_whiz")
    )

    f_28_present_participle_whiz = (
        tokens
        .filter(
            (pl.col("tag") == "VBG"),
            (pl.col("dep_rel") == "acl"),
            (pl.col("pos_lag_1") == "NOUN")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_28_present_participle_whiz")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_28_present_participle_whiz").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_28_present_participle_whiz")
    )

    f_29_that_subj = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "that"),
            (pl.col("dep_rel").str.contains("nsubj")),
            (pl.col("tag_lag_1").str.contains("^N|^CD|DT"))
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_29_that_subj")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_29_that_subj").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_29_that_subj")
    )

    f_30_that_obj = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "that"),
            (pl.col("dep_rel").str.contains("dobj")),
            (pl.col("tag_lag_1").str.contains("^N|^CD|DT"))
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_30_that_obj")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_30_that_obj").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_30_that_obj")
    )

    f_31_wh_subj = (
        tokens
        .filter(
            (pl.col("tag").str.contains("^W")),
            (pl.col("lem_lag_2") != "ask"),
            (pl.col("lem_lag_2") != "tell"),
            (
                (pl.col("tag_lag_1").str.contains("^N|^CD|DT")) |
                (
                    (pl.col("pos_lag_1") == "PUNCT") &
                    (pl.col("tag_lag_2").str.contains("^N|^CD|DT")) &
                    (pl.col("token") == "who")
                )
            )
            )
        .filter(
            (pl.col("token") != "that"),
            (pl.col("dep_rel").str.contains("nsubj"))
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_31_wh_subj")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_31_wh_subj").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_31_wh_subj")
    )

    f_32_wh_obj = (
        tokens
        .filter(
            (pl.col("tag").str.contains("^W")),
            (pl.col("lem_lag_2") != "ask"),
            (pl.col("lem_lag_2") != "tell"),
            (
                (pl.col("tag_lag_1").str.contains("^N|^CD|DT")) |
                (
                    (pl.col("pos_lag_1") == "PUNCT") &
                    (pl.col("tag_lag_2").str.contains("^N|^CD|DT")) &
                    (pl.col("token") == "who")
                )
            )
            )
        .filter(
            (pl.col("token") != "that"),
            (pl.col("dep_rel").str.contains("obj"))
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_32_wh_obj")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_32_wh_obj").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_32_wh_obj")
    )

    f_34_sentence_relatives = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "which"),
            (pl.col("pos_lag_1") == "PUNCT")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_34_sentence_relatives")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_34_sentence_relatives").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_34_sentence_relatives")
    )

    f_35_because = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase() == "because"),
            (pl.col("tok_lag_-1").str.to_lowercase() != "of")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_35_because")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_35_because").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_35_because")
    )

    f_38_other_adv_sub = (
        tokens
        .filter(
            (pl.col("pos") == "SCONJ"),
            (pl.col("dep_rel") == "mark"),
            (pl.col("token").str.to_lowercase() != "because"),
            (pl.col("token").str.to_lowercase() != "if"),
            (pl.col("token").str.to_lowercase() != "unless"),
            (pl.col("token").str.to_lowercase() != "though"),
            (pl.col("token").str.to_lowercase() != "although"),
            (pl.col("token").str.to_lowercase() != "tho")
            )
        .filter(
            ~(
                (pl.col("token").str.to_lowercase() == "that") &
                (pl.col("dep_lag_1") != "ADV")
            )
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_38_other_adv_sub")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_38_other_adv_sub").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_38_other_adv_sub")
    )

    f_39_prepositions = (
        tokens
        .filter(
            (pl.col("dep_rel") == "prep")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_39_prepositions")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_39_prepositions").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_39_prepositions")
    )

    f_40_adj_attr = (
        tokens
        .filter(
            (pl.col("pos") == "ADJ"),
            (
                (pl.col("pos_lag_-1") == "NOUN") |
                (pl.col("pos_lag_-1") == "ADJ") |
                (
                    (pl.col("tok_lag_-1") == ",") &
                    (pl.col("pos_lag_-2") == "ADJ")
                )
            )
            )
        .filter(
            ~pl.col("token").str.contains("-")
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_40_adj_attr")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_40_adj_attr").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_40_adj_attr")
    )

    f_41_adj_pred = (
        tokens
        .filter(
            (pl.col("pos") == "ADJ"),
            (
                (pl.col("pos_lag_1") == "VERB") |
                (pl.col("pos_lag_1") == "AUX")
            ),
            (
                (pl.col("lem_lag_1").is_in(WORDLISTS["linking_matchlist"])),
                (pl.col("pos_lag_-1") != "NOUN"),
                (pl.col("pos_lag_-1") != "ADJ"),
                (pl.col("pos_lag_-1") != "ADV")
            )
            )
        .filter(
            ~pl.col("token").str.contains("-")
        )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_41_adj_pred")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_41_adj_pred").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_41_adj_pred")
    )

    if force_ttr is False:
        f_43_type_token = (
            tokens
            .filter(
                pl.col("token").str.to_lowercase().str.contains("^[a-z]+$")
                )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token").str.to_lowercase(),
                        pl.col("tag").str.to_lowercase(),
                        ], separator="_",
                        ).alias("token")
                        ).select(["doc_id", "token"])
            .rolling(
                pl.int_range(pl.len()).alias("index"),
                period="100i",
                group_by="doc_id"
                )
            .agg(
                pl.when(
                    pl.len() == 100
                    ).then(
                        pl.col("token").n_unique().truediv(100)
                        ).alias("f_43_type_token"),
            )
            .drop("index")
        ).group_by(
            "doc_id", maintain_order=True
            ).agg(pl.col("f_43_type_token").mean()
                  ).select("f_43_type_token")

    else:
        f_43_type_token = (
            tokens
            .filter(
                pl.col("token").str.to_lowercase().str.contains("^[a-z]+$")
                )
            .with_columns(
                pl.concat_str(
                    [
                        pl.col("token").str.to_lowercase(),
                        pl.col("tag").str.to_lowercase(),
                        ], separator="_",
                        ).alias("token")
                        ).select(["doc_id", "token"])
        ).group_by(
            "doc_id", maintain_order=True
            ).agg(
                pl.col("token").n_unique().truediv(
                    pl.col("token").len()
                    ).alias("f_43_type_token")
                ).select("f_43_type_token")

    f_44_mean_word_length = (
        tokens
        .with_columns(
            pl.col("token").str.to_lowercase()
        )
        .filter(pl.col("token").str.contains("^[a-z]+$"))
        .select(["doc_id", "token"])
        .with_columns(
            pl.col("token").str.len_chars().alias("f_44_mean_word_length")
        )
        ).group_by(
            "doc_id", maintain_order=True
            ).agg(pl.col("f_44_mean_word_length").mean()
                  ).select("f_44_mean_word_length")

    f_51_demonstratives = (
        tokens
        .filter(
            (pl.col("token").str.to_lowercase()
             .is_in(WORDLISTS["pronoun_matchlist"])),
            (pl.col("dep_rel") == "det")
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_51_demonstratives")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_51_demonstratives").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_51_demonstratives")
    )

    f_60_that_deletion = (
        tokens
        .filter(
                (pl.col("lemma").is_in(WORDLISTS["verb_matchlist"])),
                (pl.col("pos") == "VERB"),
               (
                   (pl.col("dep_lag_-1") == "nsubj") &
                   (pl.col("pos_lag_-2") == "VERB") &
                   (pl.col("tag_lag_-1") != "WP") &
                   (pl.col("tag_lag_-2") != "VBG")
                   ) |
               (
                   (pl.col("tag_lag_-1") == "DT") &
                   (pl.col("dep_lag_-2") == "nsubj") &
                   (pl.col("pos_lag_-3") == "VERB")
                   ) |
               (
                   (pl.col("tag_lag_-1") == "DT") &
                   (pl.col("dep_lag_-2") == "amod") &
                   (pl.col("dep_lag_-3") == "nsubj") &
                   (pl.col("pos_lag_-4") == "VERB")
                   )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_60_that_deletion")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_60_that_deletion").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_60_that_deletion")
    )

    f_61_stranded_preposition = (
        tokens
        .filter(
            (pl.col("tag") == "IN"),
            (
                (pl.col("dep_rel") == "prep") &
                (pl.col("tag_lag_-1").str.contains("^[[:punct:]]$"))
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_61_stranded_preposition")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_61_stranded_preposition").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_61_stranded_preposition")
    )

    f_62_split_infinitive = (
        tokens
        .filter(
            (pl.col("tag") == "TO"),
            (
                (pl.col("tag_lag_-1") == "RB") &
                (pl.col("tag_lag_-2") == "VB")
            ) |
            (
                (pl.col("tag_lag_-1") == "RB") &
                (pl.col("tag_lag_-2") == "RB") &
                (pl.col("tag_lag_-3") == "VB")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_62_split_infinitive")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_62_split_infinitive").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_62_split_infinitive")
    )

    f_63_split_auxiliary = (
        tokens
        .filter(
            (pl.col("dep_rel").str.contains("aux")),
            (
                (pl.col("pos_lag_-1") == "ADV") &
                (pl.col("pos_lag_-2") == "VERB")
            ) |
            (
                (pl.col("pos_lag_-1") == "ADV") &
                (pl.col("pos_lag_-2") == "ADV") &
                (pl.col("pos_lag_-3") == "VERB")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_63_split_auxiliary")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_63_split_auxiliary").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_63_split_auxiliary")
    )

    f_64_phrasal_coordination = (
        tokens
        .filter(
            (pl.col("tag") == "CC"),
            (
                (pl.col("pos_lag_-1") == "NOUN") &
                (pl.col("pos_lag_1") == "NOUN")
            ) |
            (
                (pl.col("pos_lag_-1") == "VERB") &
                (pl.col("pos_lag_1") == "VERB")
            ) |
            (
                (pl.col("pos_lag_-1") == "ADJ") &
                (pl.col("pos_lag_1") == "ADJ")
            ) |
            (
                (pl.col("pos_lag_-1") == "ADV") &
                (pl.col("pos_lag_1") == "ADV")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_64_phrasal_coordination")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_64_phrasal_coordination").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_64_phrasal_coordination")
    )

    f_65_clausal_coordination = (
        tokens
        .filter(
            (pl.col("tag") == "CC"),
            (pl.col("dep_rel") != "ROOT"),
            (
                (pl.col("dep_lag_-1") == "nsubj") |
                (pl.col("dep_lag_-2") == "nsubj") |
                (pl.col("dep_lag_-3") == "nsubj")
            )
            )
        .group_by(
            "doc_id", maintain_order=True
            ).len(name="f_65_clausal_coordination")
        .join(ids, on="doc_id", how="right", coalesce=True)
        .with_columns(
            pl.col("f_65_clausal_coordination").fill_null(strategy="zero")
            )
        .sort("doc_id", descending=False)
        .select("f_65_clausal_coordination")
    )

    biber_counts = pl.concat([
        biber_1, f_02_perfect_aspect, f_10_demonstrative_pronoun,
        f_12_proverb_do, f_13_wh_question, f_14_nominalizations,
        f_15_gerunds, f_16_other_nouns, f_17_agentless_passives,
        f_18_by_passives, f_19_be_main_verb, f_21_that_verb_comp,
        f_22_that_adj_comp, f_23_wh_clause, f_25_present_participle,
        f_26_past_participle, f_27_past_participle_whiz,
        f_28_present_participle_whiz, f_29_that_subj, f_30_that_obj,
        f_31_wh_subj, f_32_wh_obj, f_34_sentence_relatives,
        f_35_because, f_38_other_adv_sub, f_39_prepositions,
        f_40_adj_attr, f_41_adj_pred, f_43_type_token,
        f_44_mean_word_length, f_51_demonstratives, f_60_that_deletion,
        f_61_stranded_preposition, f_62_split_infinitive, f_63_split_auxiliary,
        f_64_phrasal_coordination, f_65_clausal_coordination
        ], how="horizontal")

    biber_counts = biber_counts.select(sorted(biber_counts.columns))

    if normalize is False:
        return biber_counts

    if normalize is True:
        biber_counts = _biber_weight(biber_counts,
                                     totals=doc_totals,
                                     scheme="prop")
        return biber_counts
