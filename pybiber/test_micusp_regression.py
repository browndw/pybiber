from __future__ import annotations

import numpy as np

from pybiber.biber_analyzer import BiberAnalyzer

from .micusp_baseline_embedded import (
    BIBER_FEATURES,
    MDA_DIM_SCORES,
    MDA_GROUP_MEANS,
    MDA_LOADINGS,
    MDA_SUMMARY,
    PCA_COORDINATES,
    PCA_LOADINGS,
    PCA_VARIABLE_CONTRIBUTION,
    PCA_VARIANCE_EXPLAINED,
    df_from_embedded,
)


def test_micusp_pca_matches_020_baseline():
    """Regression test: PCA outputs match the 0.2.0 baseline artifacts.

    This uses the cached MICUSP feature table (no spaCy parse) and compares
    against the stored parquet outputs under comparisons/micusp_old.
    """

    features = df_from_embedded(BIBER_FEATURES)

    analyzer = BiberAnalyzer(features, id_column=True)
    analyzer.pca()

    # Coordinates: join on doc_id to avoid ordering issues
    coord_base = df_from_embedded(PCA_COORDINATES)
    coord_new = analyzer.pca_coordinates
    assert coord_new is not None
    pc_cols = [c for c in coord_base.columns if c.startswith("PC_")]

    joined = coord_base.join(coord_new, on="doc_id", how="inner", suffix="_new")  # noqa: E501
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Loadings: join on feature
    load_base = df_from_embedded(PCA_LOADINGS)
    load_new = analyzer.pca_loadings
    assert load_new is not None
    joined = load_base.join(load_new, on="feature", how="inner", suffix="_new")
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Contribution: join on feature
    contrib_base = df_from_embedded(PCA_VARIABLE_CONTRIBUTION)
    contrib_new = analyzer.pca_variable_contribution
    assert contrib_new is not None
    joined = contrib_base.join(contrib_new, on="feature", how="inner", suffix="_new")  # noqa: E501
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Variance explained: join on Dim
    ve_base = df_from_embedded(PCA_VARIANCE_EXPLAINED)
    ve_new = analyzer.pca_variance_explained
    assert ve_new is not None
    joined = ve_base.join(ve_new, on="Dim", how="inner", suffix="_new")
    np.testing.assert_allclose(
        joined.get_column("VE (%)").to_numpy(),
        joined.get_column("VE (%)_new").to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        joined.get_column("VE (Total)").to_numpy(),
        joined.get_column("VE (Total)_new").to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_micusp_mda_matches_020_baseline():
    """Regression test: MDA outputs match the 0.2.0 baseline artifacts.

    Uses the cached MICUSP feature table (no spaCy parse) and compares the MDA
    parquets under comparisons/micusp_old.
    """

    features = df_from_embedded(BIBER_FEATURES)

    analyzer = BiberAnalyzer(features, id_column=True)
    analyzer.mda(
        n_factors=6, cor_min=0.2, threshold=0.35,
        ml_n_starts=1, ml_random_state=0
        )

    # Dim scores: join on doc_id
    dim_base = df_from_embedded(MDA_DIM_SCORES)
    dim_new = analyzer.mda_dim_scores
    assert dim_new is not None
    factor_cols = [c for c in dim_base.columns if c.startswith("factor_")]
    joined = dim_base.join(dim_new, on="doc_id", how="inner", suffix="_new")
    np.testing.assert_allclose(
        joined.select(factor_cols).to_numpy(),
        joined.select([f"{c}_new" for c in factor_cols]).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    # Loadings: join on feature (allow tiny numeric drift)
    load_base = df_from_embedded(MDA_LOADINGS)
    load_new = analyzer.mda_loadings
    assert load_new is not None
    joined = load_base.join(load_new, on="feature", how="inner", suffix="_new")
    np.testing.assert_allclose(
        joined.select(factor_cols).to_numpy(),
        joined.select([f"{c}_new" for c in factor_cols]).to_numpy(),
        rtol=0.0,
        atol=1e-4,
    )

    # Group means: join on doc_cat
    gm_base = df_from_embedded(MDA_GROUP_MEANS)
    gm_new = analyzer.mda_group_means
    assert gm_new is not None
    joined = gm_base.join(gm_new, on="doc_cat", how="inner", suffix="_new")
    np.testing.assert_allclose(
        joined.select(factor_cols).to_numpy(),
        joined.select([f"{c}_new" for c in factor_cols]).to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )

    # Summary: join on Factor
    sum_base = df_from_embedded(MDA_SUMMARY)
    sum_new = analyzer.mda_summary
    assert sum_new is not None
    joined = sum_base.join(sum_new, on="Factor", how="inner", suffix="_new")
    numeric_cols = [c for c in sum_base.columns if sum_base[c].dtype.is_numeric()]  # noqa: E501
    np.testing.assert_allclose(
        joined.select(numeric_cols).to_numpy(),
        joined.select([f"{c}_new" for c in numeric_cols]).to_numpy(),
        rtol=0.0,
        atol=1e-8,
    )
