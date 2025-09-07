"""Additional tests for statistical analysis (MDA/PCA) and
zero-variance handling."""

import numpy as np
import polars as pl
import pytest
from pybiber.biber_analyzer import BiberAnalyzer


@pytest.fixture
def synthetic_cluster_matrix():
    """Create a feature matrix with two clear clusters and a zero-variance col.

    Cluster 1: f_01, f_02 strongly correlated.
    Cluster 2: f_03, f_04 strongly correlated.
    f_05 constant (zero variance) to test safe standardization.
    """
    rng = np.random.default_rng(42)
    n = 40
    base1 = rng.normal(0, 1, n)
    base2 = rng.normal(0, 1, n)
    f1 = base1 * 10 + rng.normal(0, 0.1, n)
    f2 = base1 * 9 + rng.normal(0, 0.1, n)
    f3 = base2 * 7 + rng.normal(0, 0.1, n)
    f4 = base2 * 6 + rng.normal(0, 0.1, n)
    f5 = np.ones(n) * 3.14  # zero variance

    cats = ["A"] * (n // 2) + ["B"] * (n - n // 2)

    df = pl.DataFrame({
        "doc_id": [f"d{i:02d}" for i in range(n)],
        "category": cats,
        "f_01": f1,
        "f_02": f2,
        "f_03": f3,
        "f_04": f4,
        "f_05": f5,
    })
    return df


def test_mda_cluster_loadings(synthetic_cluster_matrix, caplog):
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    analyzer.mda(n_factors=2, cor_min=0.2, threshold=0.35)

    load = analyzer.mda_loadings
    # Expect cluster separation:
    # f_01,f_02 high on one factor; f_03,f_04 on the other
    cluster1 = load.filter(pl.col("feature").is_in(["f_01", "f_02"]))
    cluster2 = load.filter(pl.col("feature").is_in(["f_03", "f_04"]))

    # Determine dominant factor per cluster by mean absolute loading
    def dominant_factor(df):
        cols = [c for c in df.columns if c.startswith("factor_")]
        means = {c: df[c].abs().mean() for c in cols}
        return max(means, key=means.get)

    dom1 = dominant_factor(cluster1)
    dom2 = dominant_factor(cluster2)
    assert dom1 != dom2  # Different dominant factors

    # Zero-variance feature (f_05) may be dropped by correlation mask.
    # If present, its loadings should be near zero; if absent,
    # that's acceptable.
    zero_var = load.filter(pl.col("feature") == "f_05")
    if zero_var.height == 1:
        assert (
            zero_var.select(pl.all().exclude("feature").abs() < 0.2)
            .to_series()
            .all()
        )


def test_pca_zero_variance_contribution(synthetic_cluster_matrix):
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    analyzer.pca()

    contrib = analyzer.pca_variable_contribution
    z = contrib.filter(pl.col("feature") == "f_05")
    # Contribution should be near 0 across PCs (allow tiny noise)
    for c in z.columns:
        if c.startswith("PC_"):
            assert abs(z[c].item()) < 1e-6

    # Cumulative variance explained should be ~100%
    total = analyzer.pca_variance_explained.get_column("VE (Total)").max()
    assert 99.0 < total <= 100.01


def test_mda_logs_zero_variance(synthetic_cluster_matrix, caplog):
    caplog.set_level("INFO")
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    analyzer.mda(n_factors=2)
    # Expect log mentioning zero-variance or retained features
    msgs = " ".join(r.message for r in caplog.records)
    assert "Zero-variance" in msgs or "retained" in msgs


def test_pca_logs_zero_variance(synthetic_cluster_matrix, caplog):
    caplog.set_level("INFO")
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    analyzer.pca()
    msgs = " ".join(r.message for r in caplog.records)
    assert "Zero-variance" in msgs or "retained" in msgs


def test_eigenvalues_guards_and_fallback():
    """Exercise _get_eigenvalues guard cases and fallback logic via init."""
    import polars as pl
    # Single document -> expect empty eigenvalues
    df_single_doc = pl.DataFrame({
        "category": ["A"],
        "f_01": [1.0],
        "f_02": [2.0],
    })
    ev_single = BiberAnalyzer(df_single_doc).eigenvalues
    assert ev_single.height == 0

    # Single variable (multi-doc) -> expect empty eigenvalues
    df_single_var = pl.DataFrame({
        "category": ["A", "B", "A"],
        "f_01": [1.0, 2.0, 3.0],
    })
    ev_single_var = BiberAnalyzer(df_single_var).eigenvalues
    assert ev_single_var.height == 0

    # All variables below very high cor_min for eigenvalue MDA subset fallback
    rng = np.random.default_rng(0)
    data = {"category": ["A", "B", "A", "B", "A"]}
    for i in range(1, 6):
        data[f"f_{i:02d}"] = rng.normal(0, 1, 5)
    df_uncorr = pl.DataFrame(data)
    analyzer_uncorr = BiberAnalyzer(df_uncorr)
    # With default cor_min=0.2 some may drop; simulate high threshold manually
    # by calling internal _get_eigenvalues with high cor_min through reassign
    from pybiber.biber_analyzer import _get_eigenvalues as _gev  # local import
    ev_high = _gev(analyzer_uncorr.variables.to_numpy(), cor_min=0.999)
    # Fallback: lengths in ev_all and ev_mda should match if present
    if ev_high.height > 0:
        assert (
            ev_high.get_column("ev_all").len() ==
            ev_high.get_column("ev_mda").len()
        )


def test_significance_labeling():
    """Factor ANOVA summary should show significance for clear group diff."""
    import polars as pl
    rng = np.random.default_rng(123)
    n_per = 20
    group_A = rng.normal(10, 1, n_per)
    group_B = rng.normal(-10, 1, n_per)
    # Strongly correlated second feature
    f2_A = group_A * 0.9 + rng.normal(0, 0.1, n_per)
    f2_B = group_B * 0.9 + rng.normal(0, 0.1, n_per)
    df = pl.DataFrame({
        "category": ["A"] * n_per + ["B"] * n_per,
        "f_01": np.concatenate([group_A, group_B]),
        "f_02": np.concatenate([f2_A, f2_B]),
    })
    analyzer = BiberAnalyzer(df)
    analyzer.mda(n_factors=1)
    signif = analyzer.mda_summary.get_column("Signif").to_list()
    assert signif and signif[0] != "NS"


def test_visualization_guards_and_success(synthetic_cluster_matrix, caplog):
    caplog.set_level("WARNING")
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    # Guards before analysis
    assert analyzer.mdaviz_groupmeans() is None
    assert analyzer.pcaviz_contrib() is None
    guard_msgs = " ".join(r.message for r in caplog.records)
    assert "Have you executed" in guard_msgs
    caplog.clear()
    # After analyses
    analyzer.mda(n_factors=2)
    analyzer.pca()
    fig1 = analyzer.mdaviz_groupmeans()
    fig2 = analyzer.pcaviz_contrib()
    assert hasattr(fig1, "axes") and hasattr(fig2, "axes")


def test_mda_cluster_sign_consistency(synthetic_cluster_matrix):
    analyzer = BiberAnalyzer(synthetic_cluster_matrix, id_column=True)
    analyzer.mda(n_factors=2, cor_min=0.2, threshold=0.35)
    load = analyzer.mda_loadings
    # Identify factor columns
    factor_cols = [c for c in load.columns if c.startswith("factor_")]
    # Determine dominant factor for cluster1 (f_01,f_02)
    cluster1 = load.filter(pl.col("feature").is_in(["f_01", "f_02"]))
    dom = max(factor_cols, key=lambda c: cluster1[c].abs().mean())
    signs = cluster1[dom].to_numpy()
    # Non-zero signs should be consistent (product positive)
    if np.all(signs != 0):
        assert np.sign(signs[0]) == np.sign(signs[1])
