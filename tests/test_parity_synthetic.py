import numpy as np
import polars as pl
from factor_analyzer import FactorAnalyzer, Rotator
from sklearn.linear_model import LinearRegression

from pybiber.biber_analyzer import (
    BiberAnalyzer,
    _ml_factor_loadings_from_corr,
    _promax_r,
    _sort_loadings,
    _safe_standardize,
    _varimax,
)


def _synthetic_matrix(seed: int = 1234,
                      n_docs: int = 240,
                      n_features: int = 12,
                      n_factors: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base_loadings = rng.normal(scale=0.6, size=(n_features, n_factors))
    factor_scores = rng.normal(size=(n_docs, n_factors))
    unique_var = rng.uniform(0.25, 0.65, size=n_features)
    noise = rng.normal(scale=np.sqrt(unique_var), size=(n_docs, n_features))
    X = factor_scores @ base_loadings.T + noise
    return X


def _align_matrix(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Align candidate loadings to reference via least-squares rotation."""
    coef, *_ = np.linalg.lstsq(candidate, reference, rcond=None)
    return candidate @ coef


# adapted from the R stats package
# https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/factanal.R
def _reference_promax(x: np.array, m=4):
    Q = x * np.abs(x)**(m-1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, Q)
    U = model.coef_.T
    d = np.diag(np.linalg.inv(np.dot(U.T, U)))
    U = U * np.sqrt(d)
    promax_loadings = np.dot(x, U)
    return promax_loadings


def test_ml_factor_loadings_matches_factor_analyzer():
    X = _synthetic_matrix()
    X_z, _ = _safe_standardize(X, ddof=1)
    corr = np.corrcoef(X_z, rowvar=False)
    ours = _ml_factor_loadings_from_corr(
        corr, n_factors=4, n_starts=3, random_state=0
        )

    fa = FactorAnalyzer(n_factors=4, rotation=None, method="ml")
    fa.fit(X_z)
    fa_loadings = fa.loadings_

    aligned = _align_matrix(ours, fa_loadings)
    assert np.max(np.abs(aligned - fa_loadings)) < 5e-3
    assert np.mean(np.abs(aligned - fa_loadings)) < 1e-3


def test_rotations_match_factor_analyzer():
    X = _synthetic_matrix(seed=4321)
    X_z, _ = _safe_standardize(X, ddof=1)
    corr = np.corrcoef(X_z, rowvar=False)
    ours_unrot = _ml_factor_loadings_from_corr(
        corr, n_factors=3, n_starts=3, random_state=1
        )

    fa = FactorAnalyzer(n_factors=3, rotation=None, method="ml")
    fa.fit(X_z)
    fa_unrot = fa.loadings_

    rotator = Rotator(method="varimax")
    fa_varimax = rotator.fit_transform(fa_unrot.copy())
    our_varimax = _varimax(ours_unrot.copy())
    aligned_varimax = _align_matrix(our_varimax, fa_varimax)
    assert np.max(np.abs(aligned_varimax - fa_varimax)) < 5e-3

    rotator = Rotator(method="promax")
    # Use reference promax for comparison to
    # reduce dependency on Rotator internals
    fa_promax = _reference_promax(fa_unrot.copy())
    our_promax = _promax_r(ours_unrot.copy())
    aligned_promax = _align_matrix(our_promax, fa_promax)
    assert np.max(np.abs(aligned_promax - fa_promax)) < 1e-2


def test_mda_loadings_track_factor_analyzer_promax():
    # rng = np.random.default_rng(777)
    n_docs = 180
    X = _synthetic_matrix(seed=777, n_docs=n_docs, n_features=10, n_factors=3)
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    categories = [f"cat_{i % 3}" for i in range(n_docs)]
    data = {"doc_id": doc_ids, "doc_cat": categories}
    for idx in range(X.shape[1]):
        data[f"f_{idx:02d}"] = X[:, idx]
    df = pl.DataFrame(data)

    analyzer = BiberAnalyzer(df, id_column=True)
    analyzer.mda(
        n_factors=3, cor_min=0.0, ml_n_starts=3, ml_random_state=5
        )
    loadings = analyzer.mda_loadings
    features = loadings.get_column("feature").to_list()
    factor_cols = [c for c in loadings.columns if c.startswith("factor_")]
    ours = loadings.select(factor_cols).to_numpy()

    X_trim = df.select(features).to_numpy()
    X_trim_z, _ = _safe_standardize(X_trim, ddof=1)
    fa = FactorAnalyzer(n_factors=3, rotation=None, method="ml")
    fa.fit(X_trim_z)
    fa_promax = _sort_loadings(_reference_promax(fa.loadings_.copy()))

    aligned = _align_matrix(ours, fa_promax)
    assert np.max(np.abs(aligned - fa_promax)) < 1.5e-2


def test_ml_factor_loadings_handles_near_collinear_features():
    rng = np.random.default_rng(999)
    X = _synthetic_matrix(seed=888, n_docs=500, n_features=6, n_factors=3)
    collinear = X[:, 0] * 0.97 + 0.03 * rng.normal(size=X.shape[0])
    X[:, 3] = collinear
    X[:, 4] = X[:, 1] * 0.94 + 0.06 * rng.normal(size=X.shape[0])
    X[:, 5] = X[:, 2] * 0.9 + 0.1 * rng.normal(size=X.shape[0])
    X_z, _ = _safe_standardize(X, ddof=1)
    corr = np.corrcoef(X_z, rowvar=False)
    ours = _ml_factor_loadings_from_corr(
        corr, n_factors=3, n_starts=6, random_state=7
        )

    fa = FactorAnalyzer(n_factors=3, rotation=None, method="ml")
    fa.fit(X_z)
    aligned = _align_matrix(ours, fa.loadings_)
    assert np.max(np.abs(aligned - fa.loadings_)) < 2e-2


def test_ml_factor_loadings_survives_feature_filtering():
    rng = np.random.default_rng(1312)
    base = _synthetic_matrix(seed=1312, n_docs=320, n_features=15, n_factors=5)
    weak = rng.normal(scale=0.02, size=(base.shape[0], 5))
    X = np.concatenate([base, weak], axis=1)
    X_z, _ = _safe_standardize(X, ddof=1)
    corr = np.corrcoef(X_z, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    with np.errstate(invalid="ignore"):
        abs_max = np.nanmax(np.abs(corr), axis=0)
    abs_max = np.nan_to_num(abs_max, nan=0.0)
    cor_min = 0.35
    keep = abs_max > cor_min
    if not keep.any():
        keep[:] = True
    trimmed = X_z[:, keep]
    corr_trim = np.corrcoef(trimmed, rowvar=False)
    ours = _ml_factor_loadings_from_corr(
        corr_trim, n_factors=4, n_starts=5, random_state=9
        )

    fa = FactorAnalyzer(n_factors=4, rotation=None, method="ml")
    fa.fit(trimmed)
    aligned = _align_matrix(ours, fa.loadings_)
    assert np.max(np.abs(aligned - fa.loadings_)) < 1.5e-2


def test_mda_feature_filter_parity_with_factor_analyzer():
    rng = np.random.default_rng(2025)
    n_docs = 240
    signal = _synthetic_matrix(
        seed=2025, n_docs=n_docs, n_features=6, n_factors=3
        )
    noise = rng.normal(scale=0.02, size=(n_docs, 4))
    X = np.concatenate([signal, noise], axis=1)
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    categories = [f"cat_{i % 4}" for i in range(n_docs)]
    data = {"doc_id": doc_ids, "doc_cat": categories}
    for idx in range(X.shape[1]):
        data[f"f_{idx:02d}"] = X[:, idx]
    df = pl.DataFrame(data)

    cor_min = 0.45
    analyzer = BiberAnalyzer(df, id_column=True)
    analyzer.mda(
        n_factors=3, cor_min=cor_min, ml_n_starts=4, ml_random_state=3
        )
    loadings = analyzer.mda_loadings
    features = loadings.get_column("feature").to_list()

    feature_cols = [c for c in df.columns if c.startswith("f_")]
    matrix = df.select(feature_cols).to_numpy()
    corr = np.corrcoef(matrix, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    with np.errstate(invalid="ignore"):
        abs_max = np.nanmax(np.abs(corr), axis=0)
    abs_max = np.nan_to_num(abs_max, nan=0.0)
    expected_keep = [c for keep, c in zip(abs_max > cor_min, feature_cols) if keep]  # noqa: E501
    assert features == expected_keep

    factor_cols = [c for c in loadings.columns if c.startswith("factor_")]
    ours = loadings.select(factor_cols).to_numpy()
    X_trim = df.select(features).to_numpy()
    X_trim_z, _ = _safe_standardize(X_trim, ddof=1)
    fa = FactorAnalyzer(n_factors=3, rotation=None, method="ml")
    fa.fit(X_trim_z)
    fa_promax = _sort_loadings(_reference_promax(fa.loadings_.copy()))
    aligned = _align_matrix(ours, fa_promax)
    assert np.max(np.abs(aligned - fa_promax)) < 2e-2


def test_mda_dim_scores_match_manual_reference():
    n_docs = 150
    X = _synthetic_matrix(seed=4242, n_docs=n_docs, n_features=9, n_factors=3)
    doc_ids = [f"doc_{i}" for i in range(n_docs)]
    categories = [f"cat_{i % 5}" for i in range(n_docs)]
    data = {"doc_id": doc_ids, "doc_cat": categories}
    for idx in range(X.shape[1]):
        data[f"f_{idx:02d}"] = X[:, idx]
    df = pl.DataFrame(data)

    threshold = 0.30
    analyzer = BiberAnalyzer(df, id_column=True)
    analyzer.mda(
        n_factors=3, threshold=threshold, cor_min=0.0,
        ml_n_starts=4, ml_random_state=11
        )
    loadings = analyzer.mda_loadings
    features = loadings.get_column("feature").to_list()
    factor_cols = [c for c in loadings.columns if c.startswith("factor_")]
    loadings_arr = loadings.select(factor_cols).to_numpy()

    X_trim = df.select(features).to_numpy()
    X_trim_z, _ = _safe_standardize(X_trim, ddof=1)
    pos = (loadings_arr > threshold).T
    neg = (loadings_arr < -threshold).T
    manual_scores = []
    for idx in range(loadings_arr.shape[1]):
        pos_sum = np.sum(
            X_trim_z[:, pos[idx]], axis=1
            ) if pos[idx].any() else np.zeros(n_docs)
        neg_sum = np.sum(
            X_trim_z[:, neg[idx]], axis=1
            ) if neg[idx].any() else np.zeros(n_docs)
        manual_scores.append(pos_sum - neg_sum)
    manual_scores = np.column_stack(manual_scores)

    dim_scores = analyzer.mda_dim_scores.select(
        [c for c in analyzer.mda_dim_scores.columns if c.startswith("factor_")]
        )
    ours = dim_scores.to_numpy()
    assert ours.shape == manual_scores.shape
    assert np.max(np.abs(ours - manual_scores)) < 1e-10
