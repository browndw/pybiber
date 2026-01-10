from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from pybiber.biber_analyzer import BiberAnalyzer


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _baseline_dir() -> Path:
    return _repo_root() / "comparisons" / "micusp_old"


@pytest.mark.parametrize(
    "artifact",
    [
        "biber_features.parquet",
        "mda_dim_scores.parquet",
        "mda_loadings.parquet",
        "mda_group_means.parquet",
        "mda_summary.parquet",
        "pca_coordinates.parquet",
        "pca_loadings.parquet",
        "pca_variable_contribution.parquet",
        "pca_variance_explained.parquet",
    ],
)
def test_micusp_baseline_artifacts_present(artifact: str):
    path = _baseline_dir() / artifact
    if not path.exists():
        pytest.skip(f"MICUSP baseline artifact missing: {path}")


def test_micusp_pca_matches_020_baseline():
    """Regression test: PCA outputs match the 0.2.0 baseline artifacts.

    This uses the cached MICUSP feature table (no spaCy parse) and compares
    against the stored parquet outputs under comparisons/micusp_old.
    """

    base = _baseline_dir()
    if not base.exists():
        pytest.skip("MICUSP baseline directory not present")

    features_path = base / "biber_features.parquet"
    if not features_path.exists():
        pytest.skip("MICUSP baseline feature table not present")

    features = pl.read_parquet(features_path)

    analyzer = BiberAnalyzer(features, id_column=True)
    analyzer.pca()

    # Coordinates: join on doc_id to avoid ordering issues
    coord_base = pl.read_parquet(base / "pca_coordinates.parquet")
    coord_new = analyzer.pca_coordinates
    assert coord_new is not None
    pc_cols = [c for c in coord_base.columns if c.startswith("PC_")]

    joined = coord_base.join(coord_new, on="doc_id", how="inner", suffix="_new")  # noqa: E501
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Loadings: join on feature
    load_base = pl.read_parquet(base / "pca_loadings.parquet")
    load_new = analyzer.pca_loadings
    assert load_new is not None
    joined = load_base.join(load_new, on="feature", how="inner", suffix="_new")
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Contribution: join on feature
    contrib_base = pl.read_parquet(base / "pca_variable_contribution.parquet")
    contrib_new = analyzer.pca_variable_contribution
    assert contrib_new is not None
    joined = contrib_base.join(contrib_new, on="feature", how="inner", suffix="_new")  # noqa: E501
    a = joined.select(pc_cols).to_numpy()
    b = joined.select([f"{c}_new" for c in pc_cols]).to_numpy()
    np.testing.assert_allclose(a, b, rtol=0.0, atol=1e-10)

    # Variance explained: join on Dim
    ve_base = pl.read_parquet(base / "pca_variance_explained.parquet")
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

    base = _baseline_dir()
    if not base.exists():
        pytest.skip("MICUSP baseline directory not present")

    features_path = base / "biber_features.parquet"
    if not features_path.exists():
        pytest.skip("MICUSP baseline feature table not present")

    features = pl.read_parquet(features_path)

    analyzer = BiberAnalyzer(features, id_column=True)
    analyzer.mda(
        n_factors=6, cor_min=0.2, threshold=0.35,
        ml_n_starts=1, ml_random_state=0
        )

    # Dim scores: join on doc_id
    dim_base = pl.read_parquet(base / "mda_dim_scores.parquet")
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
    load_base = pl.read_parquet(base / "mda_loadings.parquet")
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
    gm_base = pl.read_parquet(base / "mda_group_means.parquet")
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
    sum_base = pl.read_parquet(base / "mda_summary.parquet")
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
