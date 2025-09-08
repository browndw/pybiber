import numpy as np
import polars as pl
from sklearn import decomposition


def _safe_standardize(x: np.ndarray, ddof: int = 1, eps: float = 1e-12):
    """Standardize columns with ddof and guard against zero variance."""
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0, ddof=ddof)
    std = np.where(std < eps, 1.0, std)
    return (x - mean) / std


def _compute_contrib(X: np.ndarray, n_components: int) -> np.ndarray:
    """Compute variable contributions per component (FactoMineR style).

    For standardized inputs, the contribution of variable j to component k is:
    100 * (e_{jk}^2), where e_{jk} is the eigenvector loading (component axis)
    for variable j on component k. In sklearn, components_ has shape
    (n_components, n_features) and rows are unit-norm eigenvectors.
    """
    pca = decomposition.PCA(n_components=n_components)
    pca.fit(X)
    # components_: (k, p). Contribution matrix: (p, k)
    contrib = 100.0 * (pca.components_ ** 2).T
    return contrib


def test_pca_variable_contribution_matches_r_output():
    # Load feature matrix (documents x features)
    df = pl.read_csv("tests/test_data/df_biber.csv")
    # Numeric feature columns (exclude doc_id)
    feat_cols = [c for c, dt in df.schema.items() if dt != pl.Utf8]
    X = df.select(feat_cols).to_numpy()

    # Standardize with ddof=1 (R/FactoMineR convention)
    Z = _safe_standardize(X, ddof=1)

    # Compute contributions for first 5 PCs (to match contrib.csv)
    n_components = 5
    contrib_py = _compute_contrib(Z, n_components=n_components)

    # Wrap in DataFrame with feature labels for alignment
    contrib_py_df = pl.DataFrame(
        contrib_py,
        schema=[f"PC_{i}" for i in range(1, n_components + 1)],
    ).with_columns(pl.Series("feature", feat_cols)).select(
        [
            "feature",
            *[f"PC_{i}" for i in range(1, n_components + 1)],
        ]
    )

    # Load R-generated contributions
    contrib_r_df = pl.read_csv("tests/test_data/contrib.csv")

    # Align on feature name (doc_id column holds feature labels in the R file)
    merged = contrib_r_df.join(
        contrib_py_df, on="feature", how="inner", suffix="_py"
    )

    # Ensure we matched all features present in R file
    assert merged.height == contrib_r_df.height, (
        "Feature mismatch between R and Python outputs"
    )

    # Compare each PC column within a small tolerance
    tol = 1e-5
    for i in range(1, n_components + 1):
        pc_r = merged.get_column(f"PC_{i}").to_numpy()
        pc_py = merged.get_column(f"PC_{i}_py").to_numpy()
        # R contributions are non-negative; Python uses the same convention
        # (no polarity). Compare absolute differences.
        diff = np.abs(pc_r - pc_py)
        max_diff = float(diff.max()) if diff.size else 0.0
        assert np.allclose(pc_r, pc_py, atol=tol, rtol=0), (
            f"Contrib mismatch on PC_{i}: max |Δ|={max_diff:.3e} (tol={tol})"
        )
