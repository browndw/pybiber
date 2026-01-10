import numpy as np
import polars as pl

from pybiber.biber_analyzer import (
    BiberAnalyzer,
    _promax_r,
    _safe_standardize,
    _ml_factor_loadings_from_corr,
    _sort_loadings,
    _varimax,
)


def reference_mda(
    feature_matrix: pl.DataFrame,
    n_factors: int = 3,
    cor_min: float = 0.2,
    threshold: float = 0.35,
):
    """Reference NumPy pipeline mirroring the analyzer implementation."""
    variables = feature_matrix.select(pl.selectors.numeric())

    # filter out non-correlating variables
    X = variables.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        m_cor = np.corrcoef(X, rowvar=False)
    np.fill_diagonal(m_cor, 0)
    with np.errstate(invalid="ignore"):
        abs_max = np.nanmax(np.abs(m_cor), axis=0)
    abs_max = np.nan_to_num(abs_max, nan=0.0)
    keep = abs_max > cor_min
    if not keep.any():
        keep[:] = True
    m_trim = variables.select(
        [c for c, k in zip(variables.columns, keep) if k]
    )

    # scale variables (ddof=1)
    x = m_trim.to_numpy()
    m_z, _ = _safe_standardize(x, ddof=1)

    n_factors = min(n_factors, m_trim.width)
    corr_trim = np.corrcoef(m_trim.to_numpy(), rowvar=False)
    corr_trim = np.nan_to_num(corr_trim, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr_trim, 1.0)
    base_loadings = _ml_factor_loadings_from_corr(
        corr_trim, n_factors=n_factors, n_starts=5, random_state=0
    )
    if n_factors > 1:
        varimax_loadings = _varimax(base_loadings.copy(), normalize=True)
        varimax_loadings = _sort_loadings(varimax_loadings)
        promax_loadings = _promax_r(varimax_loadings)
    else:
        promax_loadings = base_loadings

    # aggregate dimension scores
    pos = (promax_loadings > threshold).T
    neg = (promax_loadings < -threshold).T

    dim_scores = []
    for i in range(n_factors):
        pos_sum = np.sum(m_z.T[pos[i]], axis=0)
        neg_sum = np.sum(m_z.T[neg[i]], axis=0)
        scores = pos_sum - neg_sum
        dim_scores.append(scores)

    dim_scores = pl.from_numpy(
        np.array(dim_scores).T,
        schema=[f"factor_{i}" for i in range(1, n_factors + 1)],
    )

    # Attach ids/categories in the same way
    cats = feature_matrix.select(pl.selectors.string()).to_series()
    dim_scores = dim_scores.select(cats.alias("doc_cat"), pl.all())

    group_means = (
        dim_scores
        .group_by("doc_cat", maintain_order=True)
        .mean()
    )

    loadings = pl.from_numpy(
        promax_loadings,
        schema=[f"factor_{i}" for i in range(1, n_factors + 1)],
    )
    loadings = loadings.select(
        pl.Series(m_trim.columns).alias("feature"),
        pl.all(),
    )

    return dim_scores, loadings, group_means


def test_mda_dim_scores_compatibility():
    # Deterministic synthetic dataset
    rng = np.random.default_rng(42)
    n_docs = 60
    n_features = 15
    # Create two groups
    cats = ["A"] * (n_docs // 2) + ["B"] * (n_docs - n_docs // 2)
    # Features with slight group shift to avoid zero var / degenerate cases
    X = rng.normal(0, 1, size=(n_docs, n_features))
    X[n_docs // 2:, :5] += 0.2  # small shift in first 5 features for group B

    df = pl.DataFrame(
        {**{f"f_{i}": X[:, i] for i in range(n_features)}, "doc_cat": cats}
    )

    # Current implementation
    analyzer = BiberAnalyzer(df, id_column=False)
    analyzer.mda(n_factors=3, cor_min=0.2, threshold=0.35)
    current_scores = analyzer.mda_dim_scores.select(pl.selectors.numeric())
    current_loadings = analyzer.mda_loadings
    current_group_means = analyzer.mda_group_means

    # Reference implementation
    legacy_scores, legacy_loadings, legacy_group_means = reference_mda(
        df, n_factors=3, cor_min=0.2, threshold=0.35
    )

    # Compare numeric arrays within tolerance
    legacy_scores_num = legacy_scores.select(
        pl.selectors.numeric()
    ).to_numpy()
    np.testing.assert_allclose(
        current_scores.to_numpy(),
        legacy_scores_num,
        rtol=1e-6,
        atol=1e-8,
    )

    # Compare loadings (ensure same feature order)
    assert current_loadings.get_column("feature").to_list() == \
        legacy_loadings.get_column("feature").to_list()
    np.testing.assert_allclose(
        current_loadings.select(pl.selectors.numeric()).to_numpy(),
        legacy_loadings.select(pl.selectors.numeric()).to_numpy(),
        rtol=1e-6, atol=1e-8
    )

    # Compare group means after sorting by doc_cat
    cur_gm = current_group_means.sort("doc_cat")
    leg_gm = legacy_group_means.sort("doc_cat")
    assert cur_gm.get_column("doc_cat").to_list() == \
        leg_gm.get_column("doc_cat").to_list()
    np.testing.assert_allclose(
        cur_gm.select(pl.selectors.numeric()).to_numpy(),
        leg_gm.select(pl.selectors.numeric()).to_numpy(),
        rtol=1e-6, atol=1e-8
    )
