"""
Carry our specific implemations of exploratory factor analysis
from a parsed corpus.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import warnings
import math
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import logging
import importlib.resources as resources

from adjustText import adjust_text
from collections import Counter
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic handler (library users can reconfigure)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


_ML_PSI_LOWER = 5e-3
_ML_PSI_UPPER = 0.995

# ML factor analysis notes:
# - We use a bounded optimizer over uniquenesses (psi).
# - The primary start is FactorAnalyzer-style (SMC-based). If it converges, we
#   prefer it to avoid drifting into alternate local optima on innocuous input
#   changes (e.g., feature filtering). Random restarts are used only as a
#   recovery mechanism when the primary start fails.


def _svd_flip(
    u: np.ndarray,
    vt: np.ndarray,
    u_based_decision: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically flip SVD signs (NumPy port of sklearn's svd_flip).

    For PCA parity with sklearn, choose signs based on `vt` rows
    (`u_based_decision=False`).
    """
    u = np.asarray(u)
    vt = np.asarray(vt)
    if u.ndim != 2 or vt.ndim != 2:
        return u, vt
    if u.shape[1] == 0:
        return u, vt
    if u_based_decision:
        max_abs_rows = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_rows, np.arange(u.shape[1])])
    else:
        max_abs_cols = np.argmax(np.abs(vt), axis=1)
        signs = np.sign(vt[np.arange(vt.shape[0]), max_abs_cols])
    signs[signs == 0] = 1.0
    u = u * signs
    vt = vt * signs[:, None]
    return u, vt


def _safe_standardize(
        x: np.ndarray,
        ddof: int = 1,
        eps: float = 1e-12
):
    """Standardize array columns safely.

    Replaces zero (or extremely small) standard deviations with 1 to avoid
    division warnings while logging the affected variable indices.

    Parameters
    ----------
    x : np.ndarray
        2D data matrix (observations x variables).
    ddof : int, default 1
        Delta degrees of freedom passed to std.
    eps : float, default 1e-12
        Threshold below which a standard deviation is considered zero.

    Returns
    -------
    (np.ndarray, list[int])
        Standardized array and list of zero-variance variable indices.
    """
    std = np.std(x, axis=0, ddof=ddof)
    zero_var_idx = np.where(std < eps)[0]
    if zero_var_idx.size:
        logger.debug(
            "Zero-variance (or near zero) features encountered; indices=%s",
            zero_var_idx.tolist(),
        )
        # Replace zeros with 1 to keep columns (avoid NaNs / infs)
        std[zero_var_idx] = 1.0
    mean = np.mean(x, axis=0)
    return (x - mean) / std, zero_var_idx.tolist()


def _get_eigenvalues(
        x: np.ndarray,
        cor_min: float = 0.2
):
    """Compute eigenvalues for all features and the MDA-filtered subset."""
    # Guard against insufficient data (need at least 2 observations & 2 vars)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return pl.DataFrame({"ev_all": [], "ev_mda": []})
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            m_cor = np.corrcoef(x.T)
        np.fill_diagonal(m_cor, 0)
        mask = (
            pl.from_numpy(m_cor)
            .with_columns(pl.all().abs())
            .max_horizontal()
            .gt(cor_min)
            .to_list()
        )
        if not any(mask):
            mask = [True] * x.shape[1]
        y = x.T[mask].T
        x_z, _ = _safe_standardize(x, ddof=0)
        y_z, _ = _safe_standardize(y, ddof=0)
        r_all = np.cov(x_z, rowvar=False, ddof=0)
        r_mda = np.cov(y_z, rowvar=False, ddof=0)
        e_all, _ = np.linalg.eigh(r_all)
        e_mda, _ = np.linalg.eigh(r_mda)
        df_all = pl.DataFrame({'ev_all': e_all[::-1]})
        df_mda = pl.DataFrame({'ev_mda': e_mda[::-1]})
        return pl.concat([df_all, df_mda], how="horizontal")
    except Exception:
        return pl.DataFrame({"ev_all": [], "ev_mda": []})


def _principal_loadings(
    x: np.ndarray,
    n_factors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return principal-component loadings and eigenvalues."""
    if x.ndim != 2:
        raise ValueError(
            """
            Input array must be 2-dimensional for factor extraction
            """
            )
    cov = np.cov(x, rowvar=False, ddof=1)
    cov = np.nan_to_num(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    k = min(n_factors, eigvecs.shape[1])
    eigvals = eigvals[:k]
    eigvecs = eigvecs[:, :k]
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    loadings = eigvecs * np.sqrt(eigvals)
    return loadings, eigvals


# TODO(dwb): Keep the analytic gradient in lockstep with the objective.
# It is intentionally backed by numerical-gradient fallbacks in the optimizer.


def _ml_objective_with_gradient(
    psi: np.ndarray,
    corr_mtx: np.ndarray,
    n_factors: int,
) -> tuple[float, np.ndarray]:
    """Return ML objective value and analytic gradient for uniquenesses."""
    psi = np.asarray(psi, dtype=float)
    grad = np.full_like(psi, np.nan)
    if psi.ndim != 1 or psi.size != corr_mtx.shape[0]:
        return float(np.inf), grad
    if not np.all(np.isfinite(psi)) or np.any(psi <= 0):
        return float(np.inf), grad
    inv_sqrt = 1.0 / np.sqrt(psi)
    if not np.all(np.isfinite(inv_sqrt)):
        return float(np.inf), grad
    sstar = (inv_sqrt[:, None] * corr_mtx) * inv_sqrt[None, :]
    try:
        eigvals, eigvecs = np.linalg.eigh(sstar)
    except np.linalg.LinAlgError:
        return float(np.inf), grad
    eigvals = eigvals[::-1]
    eigvecs = eigvecs[:, ::-1]
    if eigvals.size <= n_factors:
        return float(np.inf), grad
    tail_vals = eigvals[n_factors:]
    if tail_vals.size == 0 or np.any(tail_vals <= 0):
        return float(np.inf), grad
    objective = -(np.sum(np.log(tail_vals) - tail_vals) - n_factors + corr_mtx.shape[0])  # noqa: E501
    tail_vecs = eigvecs[:, n_factors:]
    weights = 1.0 - 1.0 / tail_vals
    inv_sqrt_cubed = inv_sqrt ** 3
    z = inv_sqrt[:, None] * tail_vecs
    rd = corr_mtx @ z
    contrib = (tail_vecs * rd) * weights
    grad = -inv_sqrt_cubed * np.sum(contrib, axis=1)
    grad = np.asarray(grad, dtype=float)
    return float(objective), grad


def _numerical_gradient(
    func,
    x: np.ndarray,
    eps: float = 1e-8,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    # Working: fallback finite-difference
    # gradient used when analytic forms fail.
    """Central-difference gradient with optional bound-aware projections."""

    def _apply_bounds(vec: np.ndarray) -> np.ndarray:
        if bounds is None:
            return vec
        lower, upper = bounds
        return np.minimum(np.maximum(vec, lower), upper)

    grad = np.zeros_like(x)
    for i in range(x.size):
        step = eps * max(1.0, abs(x[i]))
        x_forward = x.copy()
        x_backward = x.copy()
        x_forward[i] += step
        x_backward[i] -= step
        x_forward = _apply_bounds(x_forward)
        x_backward = _apply_bounds(x_backward)
        f_forward = func(x_forward)
        f_backward = func(x_backward)
        if not np.isfinite(f_forward) or not np.isfinite(f_backward):
            grad[i] = 0.0
        else:
            grad[i] = (f_forward - f_backward) / (2.0 * step)
    return grad


def _bfgs_minimize(
    objective,
    x0: np.ndarray,
    max_iter: int = 800,
    tol: float = 1e-6,
    line_search_max: int = 30,
    history: int = 10,
    bounds: tuple[np.ndarray | float, np.ndarray | float] | None = None,
    gradient=None,
) -> tuple[np.ndarray, float, np.ndarray]:
    # Needs follow-up: deterministic
    # but line search may be too loose per MICUSP delta.
    """Limited-memory BFGS with safeguarded Armijo-Wolfe line search."""
    x = np.asarray(x0, dtype=float)
    lower_arr = upper_arr = None
    if bounds is not None:
        lower_raw, upper_raw = bounds
        lower_arr = np.asarray(lower_raw, dtype=float)
        upper_arr = np.asarray(upper_raw, dtype=float)
        if lower_arr.ndim == 0:
            lower_arr = np.full_like(x, lower_arr)
        if upper_arr.ndim == 0:
            upper_arr = np.full_like(x, upper_arr)

        def _apply_bounds(vec: np.ndarray) -> np.ndarray:
            return np.minimum(np.maximum(vec, lower_arr), upper_arr)

        x = _apply_bounds(x)
    else:
        def _apply_bounds(vec: np.ndarray) -> np.ndarray:
            return vec

    def _project_gradient(vec: np.ndarray, grad_vec: np.ndarray) -> np.ndarray:
        """Projected gradient for simple bound constraints.

        Components that would move outside the feasible region are set to 0,
        approximating the behavior of L-BFGS-B near active bounds.
        """
        if lower_arr is None or upper_arr is None:
            return grad_vec
        eps = 1e-12
        pg = np.array(grad_vec, dtype=float, copy=True)
        at_lower = vec <= (lower_arr + eps)
        at_upper = vec >= (upper_arr - eps)
        pg[at_lower & (pg > 0)] = 0.0
        pg[at_upper & (pg < 0)] = 0.0
        return pg

    def _eval_grad(vec: np.ndarray) -> np.ndarray:
        if gradient is not None:
            grad_vec = np.asarray(gradient(vec), dtype=float)
        else:
            grad_vec = _numerical_gradient(
                objective,
                vec,
                bounds=(lower_arr, upper_arr) if bounds else None,
            )
        return grad_vec

    f = objective(x)
    if not np.isfinite(f):
        return x, float(np.inf), np.full_like(x, np.nan)
    g = _project_gradient(x, _eval_grad(x))
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    rho_hist: list[float] = []
    for _ in range(max_iter):
        if not np.all(np.isfinite(g)):
            break
        grad_norm = np.linalg.norm(g, ord=np.inf)
        if grad_norm < tol:
            break

        # Two-loop recursion for L-BFGS direction
        q = g.copy()
        alpha: list[float] = []
        if s_hist:
            hist_triplets = list(zip(reversed(s_hist), reversed(y_hist), reversed(rho_hist)))  # noqa: E501
            for s_vec, y_vec, rho in hist_triplets:
                a = rho * np.dot(s_vec, q)
                alpha.append(a)
                q = q - a * y_vec
            last_s = s_hist[-1]
            last_y = y_hist[-1]
            gamma = np.dot(last_s, last_y) / max(np.dot(last_y, last_y), 1e-12)
            r = gamma * q
            for (s_vec, y_vec, rho), a in zip(reversed(hist_triplets), reversed(alpha)):  # noqa: E501
                beta = rho * np.dot(y_vec, r)
                r = r + s_vec * (a - beta)
            direction = -r
        else:
            direction = -g

        directional_deriv = np.dot(g, direction)
        if not np.isfinite(directional_deriv) or directional_deriv >= 0:
            direction = -g
            directional_deriv = np.dot(g, direction)
        if directional_deriv >= 0:
            break

        step = 1.0
        armijo = 1e-4
        wolfe = 0.9
        accepted = False
        best_candidate = None
        best_f_candidate = np.inf
        best_g_candidate = None
        for _ in range(line_search_max):
            candidate = _apply_bounds(x + step * direction)
            f_candidate = objective(candidate)
            if not np.isfinite(f_candidate):
                step *= 0.5
                continue
            g_candidate = _project_gradient(candidate, _eval_grad(candidate))
            if np.isfinite(f_candidate) and f_candidate < best_f_candidate:
                best_candidate = candidate
                best_f_candidate = f_candidate
                best_g_candidate = g_candidate
            if (
                f_candidate <= f + armijo * step * directional_deriv
                and np.dot(g_candidate, direction) >= wolfe * directional_deriv
            ):
                accepted = True
                break
            step *= 0.5
        if not accepted:
            # Fallback: if we found any improving step (even without
            # satisfying curvature), take the best improvement and continue.
            if best_candidate is None or not np.isfinite(best_f_candidate) or best_f_candidate >= f:  # noqa: E501
                break
            candidate = best_candidate
            f_candidate = best_f_candidate
            g_candidate = best_g_candidate if best_g_candidate is not None else _project_gradient(candidate, _eval_grad(candidate))  # noqa: E501

        s_vec = candidate - x
        y_vec = g_candidate - g
        ys = np.dot(y_vec, s_vec)
        if ys > 1e-10 and np.all(np.isfinite(y_vec)):
            rho = 1.0 / ys
            if len(s_hist) == history:
                s_hist.pop(0)
                y_hist.pop(0)
                rho_hist.pop(0)
            s_hist.append(s_vec)
            y_hist.append(y_vec)
            rho_hist.append(rho)
        x, f, g = candidate, f_candidate, g_candidate
    return x, f, g


def _normalize_ml_loadings(solution: np.ndarray,
                           corr_mtx: np.ndarray,
                           n_factors: int) -> np.ndarray:
    # Working: parity-tested normalization from stats::factanal.
    """Normalize ML solution to loadings (ported from R factanal)."""
    psi = np.clip(solution, 1e-8, None)
    sc = np.diag(1.0 / np.sqrt(psi))
    sstar = sc @ corr_mtx @ sc
    try:
        eigvals, eigvecs = np.linalg.eigh(sstar)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Unable to normalize ML loadings due to singular matrix"
            )
    eigvals = eigvals[::-1][:n_factors]
    eigvecs = eigvecs[:, ::-1][:, :n_factors]
    eigvals = np.maximum(eigvals - 1.0, 0.0)
    loadings = eigvecs * np.sqrt(eigvals)
    return np.diag(np.sqrt(psi)) @ loadings


def _initial_psi(
    corr_mtx: np.ndarray,
    lower: float = _ML_PSI_LOWER,
    upper: float = _ML_PSI_UPPER,
    n_starts: int = 1,
    random_state: int | None = None,
) -> np.ndarray:
    # Working: reproduces FactorAnalyzer
    # start heuristics with optional RNG seed.
    """FactorAnalyzer-style starting uniquenesses with optional restarts."""

    p = corr_mtx.shape[0]
    diag_vals = np.array(np.diag(corr_mtx), dtype=float, copy=True)
    diag_vals[~np.isfinite(diag_vals)] = 1.0
    try:
        inv = np.linalg.pinv(corr_mtx)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(corr_mtx + np.eye(p) * 1e-6)
    smc = 1.0 - 1.0 / np.diag(inv)
    base = np.clip(diag_vals - smc, lower, upper)

    starts = np.empty((max(1, n_starts), p), dtype=float)
    starts[0] = base
    if n_starts > 1:
        rng = np.random.default_rng(random_state)
        random_vals = rng.uniform(lower, upper, size=(n_starts - 1, p))
        starts[1:] = random_vals
    return starts


def _ml_factor_loadings_from_corr(
    corr_mtx: np.ndarray,
    n_factors: int,
    max_iter: int = 1000,
    tol: float = 1e-8,
    n_starts: int = 1,
    random_state: int | None = None,
) -> np.ndarray:
    """Estimate ML factor loadings from a correlation matrix via ML FA.

    Uses a bounded quasi-Newton solver over uniquenesses (psi) with an analytic
    gradient and numerical-gradient fallbacks. The primary (SMC) start is
    preferred when it converges; additional starts are for recovery only.
    """
    n_features = corr_mtx.shape[0]
    if n_factors >= n_features:
        raise ValueError(
            "n_factors must be less than number of features for ML FA"
            )
    if not np.all(np.isfinite(corr_mtx)):
        raise ValueError(
            "Correlation matrix contains non-finite entries"
            )
    psi_bounds = (_ML_PSI_LOWER, _ML_PSI_UPPER)
    starts = _initial_psi(
        corr_mtx,
        lower=psi_bounds[0],
        upper=psi_bounds[1],
        n_starts=max(1, n_starts),
        random_state=random_state,
    )

    lower_arr = np.full(n_features, psi_bounds[0], dtype=float)
    upper_arr = np.full(n_features, psi_bounds[1], dtype=float)

    cached_point: np.ndarray | None = None
    cached_value: float | None = None
    cached_grad: np.ndarray | None = None

    def _evaluate_and_cache(vec: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal cached_point, cached_value, cached_grad
        value, grad = _ml_objective_with_gradient(vec, corr_mtx, n_factors)
        cached_point = np.array(vec, dtype=float, copy=True)
        cached_value = float(value)
        cached_grad = np.array(grad, dtype=float, copy=True)
        return cached_value, cached_grad

    def obj(psi: np.ndarray) -> float:
        nonlocal cached_point, cached_value
        psi = np.asarray(psi, dtype=float)
        if (
            cached_point is not None
            and psi.shape == cached_point.shape
            and np.array_equal(cached_point, psi)
            and cached_value is not None
        ):
            return cached_value
        value, _ = _evaluate_and_cache(psi)
        return value

    def grad(psi: np.ndarray) -> np.ndarray:
        nonlocal cached_point, cached_grad
        psi = np.asarray(psi, dtype=float)
        if (
            cached_point is not None
            and psi.shape == cached_point.shape
            and np.array_equal(cached_point, psi)
            and cached_grad is not None
        ):
            return cached_grad
        _, grad_vec = _evaluate_and_cache(psi)
        return grad_vec

    def _solve_from_start(
        psi_start: np.ndarray
    ) -> tuple[np.ndarray, float] | None:
        psi_seed = np.clip(psi_start, psi_bounds[0], psi_bounds[1])
        psi_opt, obj_val, _ = _bfgs_minimize(
            obj,
            psi_seed,
            max_iter=max_iter,
            tol=tol,
            bounds=(lower_arr, upper_arr),
            gradient=grad,
        )
        if not np.isfinite(obj_val):
            return None

        # Alternate path using numerical gradients only; helps recover
        # FactorAnalyzer-equivalent optima on datasets where the analytic
        # gradient or line search stalls at a different stationary point.
        psi_num, obj_num, _ = _bfgs_minimize(
            obj,
            psi_seed,
            max_iter=max_iter,
            tol=tol,
            bounds=(lower_arr, upper_arr),
            gradient=None,
        )
        if np.isfinite(obj_num) and obj_num < obj_val - 1e-9:
            psi_opt, obj_val = psi_num, obj_num

        # Optional polish pass with numerical gradients to avoid analytic
        # gradient drift on harder real-data cases (e.g., MICUSP parity).
        psi_polish, obj_polish, _ = _bfgs_minimize(
            obj,
            psi_opt,
            max_iter=max(120, max_iter // 8),
            tol=max(tol * 0.1, 1e-10),
            bounds=(lower_arr, upper_arr),
            gradient=lambda v: _numerical_gradient(
                obj,
                v,
                bounds=(lower_arr, upper_arr),
                eps=1e-6,
            ),
        )
        if np.isfinite(obj_polish) and obj_polish < obj_val - 1e-9:
            psi_opt, obj_val = psi_polish, obj_polish
        try:
            loadings = _normalize_ml_loadings(psi_opt, corr_mtx, n_factors)
        except ValueError:
            return None
        return loadings, float(obj_val)

    # Always attempt the primary (SMC-based) start first; this mirrors
    # FactorAnalyzer behavior and keeps solutions stable across innocuous
    # changes (e.g., feature filtering) where alternative
    # starts can converge to a different local optimum.
    primary_result = _solve_from_start(starts[0])
    if primary_result is not None:
        primary_loadings, _ = primary_result
        return primary_loadings

    # Recovery path: if the primary start fails, try alternate starts and take
    # the best objective value.
    best = None
    best_obj = np.inf
    best_idx = -1
    for idx, psi0 in enumerate(starts[1:], start=1):
        result = _solve_from_start(psi0)
        if result is None:
            continue
        loadings, obj_val = result
        if (
            obj_val < best_obj
            or (
                np.isclose(obj_val, best_obj, rtol=1e-9, atol=1e-9)
                and idx < best_idx
            )
        ):
            best = loadings
            best_obj = obj_val
            best_idx = idx

    if best is None:
        raise ValueError(
            "ML factor analysis failed to converge for any start value"
            )
    return best


def _varimax(
    loadings: np.ndarray,
    normalize: bool = True,
    max_iter: int = 500,
    tol: float = 1e-5,
) -> np.ndarray:
    # Working: mirrors FactorAnalyzer Rotator; no parity regressions observed.
    """Varimax rotation mirroring factor_analyzer's Rotator."""
    if loadings.size == 0 or loadings.shape[1] < 2:
        return loadings
    X = loadings.copy()
    norms = None
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
    rotation = np.eye(X.shape[1])
    delta = 0.0
    for _ in range(max_iter):
        prev = delta
        basis = X @ rotation
        diag = np.diag(np.sum(basis**2, axis=0))
        transformed = X.T @ (basis**3 - basis @ diag / X.shape[0])
        U, S, Vt = np.linalg.svd(transformed)
        rotation = U @ Vt
        delta = S.sum()
        if delta < prev * (1.0 + tol):
            break
    X = X @ rotation
    if normalize and norms is not None:
        X = X * norms
    return X


def _promax_r(loadings: np.ndarray, m: int = 4) -> np.ndarray:
    """Promax conversion matching the legacy 0.2.0 analyzer (R factanal).

    This is the "varimax -> promax" conversion step used in stats::factanal
    (and in scripts/biber_analyzer_old.py). It assumes the input loadings have
    already been varimax-rotated.
    """
    x = np.asarray(loadings, dtype=float)
    if x.size == 0 or x.shape[1] < 2:
        return x
    q = x * np.abs(x) ** (m - 1)
    # Solve x @ u ~= q in least squares (equivalent to LinearRegression
    # without intercept).
    u, *_ = np.linalg.lstsq(x, q, rcond=None)
    # u returned is (k, k). Normalize columns.
    try:
        d = np.diag(np.linalg.inv(u.T @ u))
    except np.linalg.LinAlgError:
        d = np.diag(np.linalg.pinv(u.T @ u))
    u = u * np.sqrt(np.clip(d, 0.0, None))
    return x @ u


def _sort_loadings(loadings: np.ndarray) -> np.ndarray:
    # Working: ensures deterministic column ordering/signs
    # for downstream diffs.
    """Replicate stats::factanal ordering/sign convention."""
    if loadings.size == 0 or loadings.shape[1] == 1:
        return loadings
    ssq = np.sum(loadings**2, axis=0)
    order = np.argsort(-ssq)
    sorted_loadings = loadings[:, order]
    col_sums = np.sum(sorted_loadings, axis=0)
    neg = col_sums < 0
    if np.any(neg):
        sorted_loadings[:, neg] *= -1.0
    return sorted_loadings


def _betacf(a: float, b: float, x: float,
            max_iter: int = 200, tol: float = 3e-7) -> float:
    # Working: Numerical Recipes continued fraction used by F-test helper.
    """Continued fraction for incomplete beta (Numerical Recipes)."""
    FPMIN = 1e-30
    m2 = 0
    aa = 0.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1.0)
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < tol:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    # Working: regularized incomplete beta builds on _betacf for ANOVA stats.
    """Regularized incomplete beta function."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _f_sf(F: float, df1: int, df2: int) -> float:
    """Survival function (1-CDF) for the F distribution."""
    if df1 <= 0 or df2 <= 0 or not np.isfinite(F) or F < 0:
        return np.nan
    z = (df1 * F) / (df1 * F + df2)
    return 1.0 - _betai(df1 / 2.0, df2 / 2.0, z)


def _anova_one_way(values: list[float], groups: list[str]) -> dict[str, float]:
    # Working: NumPy-only fallback matching scipy.stats outputs in tests.
    """Compute one-way ANOVA stats using NumPy only."""
    y = np.asarray(values, dtype=float)
    g = np.asarray(groups)
    mask = np.isfinite(y)
    y = y[mask]
    g = g[mask]
    n = y.size
    unique, inverse = np.unique(g, return_inverse=True)
    k = unique.size
    if n < 2 or k < 2:
        return {"F": np.nan, "df1": 0, "df2": 0, "p": np.nan, "R2": np.nan}
    overall_mean = np.mean(y)
    ss_between = 0.0
    ss_within = 0.0
    for idx, label in enumerate(unique):
        grp = y[inverse == idx]
        if grp.size == 0:
            continue
        mean = np.mean(grp)
        ss_between += grp.size * (mean - overall_mean) ** 2
        ss_within += np.sum((grp - mean) ** 2)
    df1 = max(k - 1, 0)
    df2 = max(n - k, 0)
    if df1 == 0 or df2 == 0:
        return {"F": np.nan, "df1": df1, "df2": df2, "p": np.nan, "R2": np.nan}
    ms_between = ss_between / df1 if df1 else np.nan
    ms_within = ss_within / df2 if df2 else np.nan
    if ms_within == 0 or not np.isfinite(ms_within):
        F_stat = np.nan
    else:
        F_stat = ms_between / ms_within
    p_value = _f_sf(F_stat, df1, df2) if np.isfinite(F_stat) else np.nan
    total_ss = ss_between + ss_within
    r_squared = ss_between / total_ss if total_ss > 0 else np.nan
    return {"F": F_stat, "df1": df1, "df2": df2, "p": p_value, "R2": r_squared}


class BiberAnalyzer:

    def __init__(self,
                 feature_matrix: pl.DataFrame,
                 id_column: bool = False):

        d_types = Counter(feature_matrix.schema.dtypes())

        if set(d_types) != {pl.Float64, pl.String}:
            raise ValueError("""
                Invalid DataFrame.
                Expected a DataFrame with normalized frequenices and ids.
                    """)
        if id_column is False and d_types[pl.String] != 1:
            raise ValueError("""
                Invalid DataFrame.
                Expected a DataFrame with a column of document categories.
                """)
        if id_column is True and d_types[pl.String] != 2:
            raise ValueError("""
                Invalid DataFrame.
                Expected a DataFrame with a column of document ids \
                and a column of document categories.
                """)

        # sort string columns
        if d_types[pl.String] == 2:
            str_cols = feature_matrix.select(
                pl.selectors.string()
                ).with_columns(
                    pl.all().n_unique()
                    ).head(1).transpose(
                        include_header=True).sort("column_0", descending=True)

            doc_ids = feature_matrix.get_column(str_cols['column'][0])
            category_ids = feature_matrix.get_column(str_cols['column'][1])
            self.doc_ids = doc_ids
            self.category_ids = category_ids
        else:
            category_ids = feature_matrix.select(
                pl.selectors.string()
                ).to_series()
            self.doc_ids = None
            self.category_ids = category_ids

        self.feature_matrix = feature_matrix
        self.variables = self.feature_matrix.select(pl.selectors.numeric())
        self.eigenvalues = _get_eigenvalues(self.variables.to_numpy())
        self.doc_cats = sorted(self.category_ids.unique().to_list())
        # default matrices to None
        self.mda_summary = None
        self.mda_loadings = None
        self.mda_dim_scores = None
        self.mda_group_means = None
        self.pca_coordinates = None
        self.pca_variance_explained = None
        self.pca_variable_contribution = None
        self.pca_loadings = None

        # check grouping variable
        # Only raise if there are multiple documents and every document
        # has a unique category (i.e., grouping variable ineffective)
        if (self.feature_matrix.height > 1 and
                len(self.doc_cats) == self.feature_matrix.height):
            raise ValueError("""
                Invalid DataFrame.
                Expected a column of document categories (not one
                unique category per doc).
                """)

    def mdaviz_screeplot(self,
                         width=6,
                         height=3,
                         dpi=150,
                         mda=True) -> Figure:
        """Generate a scree plot for determining factors.

        Parameters
        ----------
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.
        mda:
            Whether or not non-colinear features should be
            filter out per Biber's multi-dimensional analysis procedure.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        if mda is True:
            x = self.eigenvalues['ev_mda']
        else:
            x = self.eigenvalues['ev_all']
        # SCREEPLOT # Cutoff >= 1
        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
        ax.plot(range(1, self.eigenvalues.height+1),
                x,
                linewidth=.5,
                color='black')
        ax.scatter(range(1, self.eigenvalues.height+1),
                   x,
                   marker='o',
                   facecolors='none',
                   edgecolors='black')
        ax.axhline(y=1, color='r', linestyle='--')
        ax.set(xlabel='Factors', ylabel='Eigenvalues', title="Scree Plot")
        return fig

    def mdaviz_groupmeans(self,
                          factor=1,
                          width=3,
                          height=7,
                          dpi=150) -> Figure:
        """Generate a stick plot of the group means for a factor.

        Parameters
        ----------
        factor:
            The factor or dimension to plot.
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        factor_col = "factor_" + str(factor)
        if self.mda_group_means is None:
            logger.warning("No factors to plot. Have you executed mda()?")
            return None
        if self.mda_group_means is not None:
            max_factor = self.mda_group_means.width - 1
        if self.mda_group_means is not None and factor > max_factor:
            logger.warning(
                "Must specify a factor between 1 and %s", str(max_factor)
            )
            return None
        else:
            x = np.repeat(0, self.mda_group_means.height)
            x_label = np.repeat(-0.05, self.mda_group_means.height)
            y = self.mda_group_means.get_column(factor_col).to_numpy()
            z = self.mda_group_means.get_column('doc_cat').to_list()

            fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
            ax.axes.get_xaxis().set_visible(False)
            ax.scatter(x[y > 0],
                       y[y > 0],
                       marker='o',
                       facecolors='#440154',
                       edgecolors='black',
                       alpha=0.75)
            ax.scatter(x[y < 0],
                       y[y < 0],
                       marker='o',
                       facecolors='#fde725',
                       edgecolors='black',
                       alpha=0.75)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.yaxis.set_label_position('right')
            ax.yaxis.set_ticks_position('right')

            texts = []
            for i, txt in enumerate(z):
                texts += [ax.text(
                    x_label[i], y[i], txt, fontsize=8, ha='right', va='center'
                    )]

            adjust_text(texts,
                        avoid_self=False,
                        target_x=x,
                        target_y=y,
                        only_move='y+',
                        expand=(1, 1.5),
                        arrowprops=dict(arrowstyle="-", lw=0.25))
            return fig

    def pcaviz_groupmeans(self,
                          pc=1,
                          width=8,
                          height=4,
                          dpi=150) -> Figure:
        """Generate a scatter plot of the group means along 2 components.

        Parameters
        ----------
        pc:
            The principal component for the x-axis.
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.

        Returns
        -------
        Figure
            A matplotlib figure.

        """
        if self.pca_coordinates is None:
            logger.warning("No component to plot. Have you executed pca()?")
            return None
        if self.pca_coordinates is not None:
            max_pca = self.pca_coordinates.width - 1
        if self.pca_coordinates is not None and pc + 1 > max_pca:
            logger.warning(
                "Must specify a pc between 1 and %s", str(max_pca - 1)
            )
            return None

        x_col = "PC_" + str(pc)
        y_col = "PC_" + str(pc + 1)
        means = (self.pca_coordinates
                 .group_by('doc_cat', maintain_order=True)
                 .mean())
        x = means.get_column(x_col).to_numpy()
        y = means.get_column(y_col).to_numpy()
        labels = means.get_column('doc_cat').to_list()

        x_title = ("Dim" +
                   str(pc) +
                   " (" +
                   str(
                       (self.pca_variance_explained[pc - 1]
                        .get_column("VE (%)")
                        .round(1)
                        .item())
                       ) +
                   "%)")
        y_title = ("Dim" +
                   str(pc + 1) +
                   " (" +
                   str(
                       (self.pca_variance_explained[pc]
                        .get_column("VE (%)")
                        .round(1)
                        .item())
                       ) +
                   "%)")

        xlimit = means.get_column(x_col).abs().ceil().max()
        ylimit = means.get_column(y_col).abs().ceil().max()

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        ax.scatter(x=x, y=y,
                   marker='o',
                   edgecolor='black',
                   facecolors='#21918c',
                   alpha=0.75)

        ax.axhline(y=0, color='gray', linestyle='-', linewidth=.5)
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=.5)

        ax.set_xlim([-xlimit, xlimit])
        ax.set_ylim([-ylimit, ylimit])

        ax.set_xlabel(x_title)
        ax.set_ylabel(y_title)

        texts = []
        for i, txt in enumerate(labels):
            texts += [ax.text(
                x[i], y[i], txt, fontsize=8, ha='center', va='center'
                )]

        adjust_text(texts,
                    expand=(2, 3),
                    arrowprops=dict(arrowstyle="-", lw=0.25))

        return fig

    def pcaviz_contrib(self,
                       pc=1,
                       width=8,
                       height=4,
                       dpi=150) -> Figure:
        """Generate a bar plot of variable contributions to a component.

        Parameters
        ----------
        pc:
            The principal component.
        width:
            The width of the plot.
        height:
            The height of the plot.
        dpi:
            The resolution of the plot.

        Returns
        -------
        Figure
            A matplotlib figure.

        Notes
        -----
        Modeled on the R function
        [fviz_contrib](https://search.r-project.org/CRAN/refmans/factoextra/html/fviz_contrib.html).

        """
        pc_col = "PC_" + str(pc)

        if self.pca_variable_contribution is None or self.pca_loadings is None:
            logger.warning("No component to plot. Have you executed pca()?")
            return None
        if self.pca_variable_contribution is not None:
            max_pca = self.pca_variable_contribution.width - 1
        if self.pca_variable_contribution is not None and pc > max_pca:
            logger.warning(
                "Must specify a pc between 1 and %s", str(max_pca)
            )
            return None

        # Merge contributions with loadings to apply polarity for visualization
        df_plot = (
            self.pca_variable_contribution
            .select('feature', pc_col)
            .join(
                self.pca_loadings.select(
                    'feature',
                    pl.col(pc_col).alias('loading')
                ),
                on='feature',
                how='inner',
            )
            .with_columns([
                pl.col(pc_col).alias('abs_contrib'),
                pl.when(pl.col('loading') > 0).then(1)
                  .when(pl.col('loading') < 0).then(-1)
                  .otherwise(0)
                  .alias('sign'),
            ])
            .with_columns(
                (
                    pl.col('abs_contrib') * pl.col('sign')
                ).alias('signed_contrib')
            )
        )

        # keep only variables with contribution above mean (by absolute value)
        mean_abs_contrib = float(
            df_plot
            .select(pl.col('abs_contrib').abs().mean().alias('m'))
            .to_series(0)[0]
        )
        df_plot = (
            df_plot
            .filter(pl.col('abs_contrib').abs() > mean_abs_contrib)
            .with_columns(pl.col('signed_contrib').alias(pc_col))
            .select('feature', pc_col)
            .sort(pc_col, descending=True)
            .with_columns(
                pl.col('feature').str.replace(r"f_\d+_", "").alias('feature')
            )
            .with_columns(
                pl.col('feature').str.replace_all('_', ' ').alias('feature')
            )
        )

        feature = df_plot['feature'].to_numpy()
        contribution = df_plot[pc_col].to_numpy()
        ylimit = df_plot.get_column(pc_col).abs().ceil().max()

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        ax.bar(
            feature[contribution > 0],
            contribution[contribution > 0],
            color='#440154', edgecolor='black', linewidth=.5,
        )
        ax.bar(
            feature[contribution < 0],
            contribution[contribution < 0],
            color='#21918c', edgecolor='black', linewidth=.5,
        )

        # Despine
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=.5)

        ax.tick_params(axis="x", which="both", labelrotation=90)
        ax.grid(axis='x', color='gray', linestyle=':', linewidth=.5)
        ax.grid(axis='y', color='w', linestyle='--', linewidth=.5)
        ax.set_ylim([-ylimit, ylimit])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel("Contribution (% x polarity)")

        return fig

    def mda(self,
            n_factors: int = 3,
            cor_min: float = 0.2,
            threshold: float = 0.35,
            ml_n_starts: int = 5,
            ml_random_state: int | None = 0):

        """Execute Biber's multi-dimensional anlaysis.

        Parameters
        ----------
        n_factors:
            The number of factors to extract.
        cor_min:
            The minimum correlation at which to drop variables.
        threshold:
            The factor loading threshold (in absolute value)
            used to calculate dimension scores.
        ml_n_starts:
            Number of random uniqueness seeds for ML estimation (>=1).
        ml_random_state:
            Optional seed for reproducible ML restarts.

        """
        # filter out non-correlating variables
        X = self.variables.to_numpy()
        # Correlation matrix (variables x variables)
        with np.errstate(divide="ignore", invalid="ignore"):
            m_cor = np.corrcoef(X, rowvar=False)
        # Remove self-correlation so it doesn't force retention
        np.fill_diagonal(m_cor, 0.0)
        # Max absolute off-diagonal correlation per variable
        with np.errstate(invalid="ignore"):
            abs_max = np.nanmax(np.abs(m_cor), axis=0)
        # Treat all-NaN columns (e.g. zero variance) as having 0 max corr
        abs_max = np.nan_to_num(abs_max, nan=0.0)
        keep = abs_max > cor_min
        if not keep.any():  # fallback – retain all if threshold is too strict
            logger.warning(
                "Correlation filter (cor_min=%.2f) would drop all %d "
                "variables; keeping all instead.",
                cor_min,
                X.shape[1],
            )
            keep[:] = True
        else:
            dropped = [
                c for c, k in zip(self.variables.columns, keep) if not k
            ]
            if dropped:
                logger.info(
                    "Dropping %d variable(s) with max |r| <= %.2f: %s",
                    len(dropped),
                    cor_min,
                    dropped,
                )
        m_trim = self.variables.select(
            [c for c, k in zip(self.variables.columns, keep) if k]
        )
        # Log zero-variance features (in full set) for transparency
        full_stds = self.variables.to_numpy().std(axis=0, ddof=1)
        zero_full = [
            self.variables.columns[i]
            for i, s in enumerate(full_stds)
            if s == 0
        ]
        if zero_full:
            logger.info(
                "Zero-variance feature(s) may be dropped in MDA: %s",
                zero_full,
            )

        if n_factors < 1:
            raise ValueError("n_factors must be >= 1")

        # scale variables (safe)
        x = m_trim.to_numpy()
        m_z, zero_var_idx = _safe_standardize(x, ddof=1)
        if zero_var_idx:
            logger.info(
                "Zero-variance features retained (neutral scaling) in MDA: %s",
                [m_trim.columns[i] for i in zero_var_idx],
            )
        # m_z = zscore(m_trim.to_numpy(), ddof=1, nan_policy='omit')

        if n_factors > m_trim.width:
            logger.warning(
                """
                Requested %d factors exceeds variable count (%d); truncating.
                """,
                n_factors,
                m_trim.width,
            )
            n_factors = m_trim.width

        base_loadings = None
        try:
            corr_trim = np.corrcoef(m_trim.to_numpy(), rowvar=False)
            corr_trim = np.nan_to_num(corr_trim, nan=0.0, posinf=0.0, neginf=0.0)  # noqa: E501
            np.fill_diagonal(corr_trim, 1.0)
            base_loadings = _ml_factor_loadings_from_corr(
                corr_trim,
                n_factors,
                n_starts=max(1, ml_n_starts),
                random_state=ml_random_state,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning(
                """
                ML factor extraction failed (%s);
                using principal component loadings.
                """,
                exc,
            )
            base_loadings, _ = _principal_loadings(m_z, n_factors)

        # Legacy 0.2.0 path: ML extraction -> varimax -> (R-style) promax.
        # This matches scripts/biber_analyzer_old.py (FactorAnalyzer
        # rotation="varimax" then stats::factanal promax conversion).
        if n_factors > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                varimax_loadings = _varimax(base_loadings.copy(), normalize=True)  # noqa: E501
                # FactorAnalyzer returns varimax loadings with factors ordered
                # by descending sum-of-squares and with column sums positive.
                # Apply that convention before the legacy promax conversion so
                # downstream dim-scores align with the 0.2.0 baseline.
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
            schema=["factor_" + str(i) for i in range(1, n_factors + 1)],
        )

        if self.doc_ids is not None:
            dim_scores = dim_scores.select(
                pl.Series(self.doc_ids).alias("doc_id"),
                pl.Series(self.category_ids).alias("doc_cat"),
                pl.all()
                )
        else:
            dim_scores = dim_scores.select(
                pl.Series(self.category_ids).alias("doc_cat"),
                pl.all()
                )

        group_means = (
            dim_scores
            .group_by("doc_cat", maintain_order=True)
            .mean()
            )

        if self.doc_ids is not None:
            group_means = group_means.drop("doc_id")

        loadings = pl.from_numpy(
            promax_loadings, schema=[
                "factor_" + str(i) for i in range(1, n_factors + 1)
                ]
            )

        loadings = loadings.select(
            pl.Series(m_trim.columns).alias("feature"),
            pl.all()
            )

        summary = []
        for i in range(1, n_factors + 1):
            factor_col = "factor_" + str(i)

            y = dim_scores.get_column(factor_col).to_list()
            X = dim_scores.get_column('doc_cat').to_list()

            stats = _anova_one_way(y, X)
            factor_summary = pl.DataFrame({
                'Factor': [factor_col],
                'F': [stats['F']],
                'df': [[stats['df1'], stats['df2']]],
                'PR(>F)': [stats['p']],
                'R2': [stats['R2']],
            })
            summary.append(factor_summary)
        summary = pl.concat(summary)
        summary = summary.with_columns(
            pl.when(pl.col("PR(>F)") < 0.001).then(pl.lit("*** p < 0.001"))
            .when(pl.col("PR(>F)") < 0.01).then(pl.lit("** p < 0.01"))
            .when(pl.col("PR(>F)") < 0.05).then(pl.lit("* p < 0.05"))
            .otherwise(pl.lit("NS")).alias("Signif")
        ).select(['Factor', 'F', "df", "PR(>F)", "Signif", "R2"])
        self.mda_summary = summary
        self.mda_loadings = loadings
        self.mda_dim_scores = dim_scores
        self.mda_group_means = group_means

    def mda_biber(
            self,
            threshold: float = 0.35
            ):

        """Project results onto Biber's dimensions.

        Parameters
        ----------
        threshold:
            The factor loading threshold (in absolute value)
            used to calculate dimension scores.

        """
        # Load packaged Biber promax loadings
        try:
            with resources.as_file(
                resources.files("pybiber.data").joinpath("biber_loadings.csv")
            ) as p:
                loadings_df = pl.read_csv(str(p))
        except Exception as e:
            raise FileNotFoundError(
                "Could not load 'biber_loadings.csv' from pybiber.data"
            ) from e

        # Identify factor columns and intersecting features
        factor_cols = [
            c for c in loadings_df.columns if c.startswith("factor_")
        ]
        if not factor_cols:
            raise ValueError("Biber loadings file has no factor_* columns.")
        n_factors = len(factor_cols)

        # Align features present in the user matrix and in the loadings
        user_feats = set(self.variables.columns)
        common_feats = [
            f for f in loadings_df.get_column("feature").to_list()
            if f in user_feats
        ]
        if not common_feats:
            raise ValueError(
                "No overlapping features between data and Biber loadings."
            )

        # Trim to common features (preserve loading order)
        m_trim = self.variables.select(common_feats)
        L = (
            loadings_df
            .filter(pl.col("feature").is_in(common_feats))
            .select(factor_cols)
            .to_numpy()
        )

        # Standardize counts (z-score per feature) using safe routine
        x = m_trim.to_numpy()
        m_z, zero_var_idx = _safe_standardize(x, ddof=1)
        if zero_var_idx:
            logger.info(
                "Zero-variance features retained (neutral scaling) in "
                "projection: %s",
                [m_trim.columns[i] for i in zero_var_idx],
            )

        # Thresholded sum/difference per Biber MDA convention
        pos = (L > threshold).T  # shape: (k, p)
        neg = (L < -threshold).T

        dim_scores = []
        for i in range(n_factors):
            pos_sum = (
                np.sum(m_z[:, pos[i]], axis=1)
                if pos[i].any() else np.zeros(m_z.shape[0])
            )
            neg_sum = (
                np.sum(m_z[:, neg[i]], axis=1)
                if neg[i].any() else np.zeros(m_z.shape[0])
            )
            scores = pos_sum - neg_sum
            dim_scores.append(scores)

        dim_scores = pl.from_numpy(
            np.array(dim_scores).T,
            schema=["factor_" + str(i) for i in range(1, n_factors + 1)],
        )

        # Attach identifiers
        if self.doc_ids is not None:
            dim_scores = dim_scores.select(
                pl.Series(self.doc_ids).alias("doc_id"),
                pl.Series(self.category_ids).alias("doc_cat"),
                pl.all(),
            )
        else:
            dim_scores = dim_scores.select(
                pl.Series(self.category_ids).alias("doc_cat"),
                pl.all(),
            )

        group_means = (
            dim_scores.group_by("doc_cat", maintain_order=True).mean()
        )
        if self.doc_ids is not None:
            group_means = group_means.drop("doc_id")

        # Loadings returned for the actually used features (aligned)
        loadings = (
            loadings_df
            .filter(pl.col("feature").is_in(common_feats))
            .select(["feature", *factor_cols])
        )

        # Simple ANOVA summary per factor (same style as mda())
        summary = []
        for i in range(1, n_factors + 1):
            factor_col = "factor_" + str(i)
            y = dim_scores.get_column(factor_col).to_list()
            X = dim_scores.get_column("doc_cat").to_list()
            stats = _anova_one_way(y, X)
            factor_summary = pl.DataFrame({
                'Factor': [factor_col],
                'F': [stats['F']],
                'df': [[stats['df1'], stats['df2']]],
                'PR(>F)': [stats['p']],
                'R2': [stats['R2']],
            })
            summary.append(factor_summary)
        summary = pl.concat(summary)
        summary = (
            summary.with_columns(
                pl.when(pl.col("PR(>F)") < 0.001)
                .then(pl.lit("*** p < 0.001"))
                .when(pl.col("PR(>F)") < 0.01)
                .then(pl.lit("** p < 0.01"))
                .when(pl.col("PR(>F)") < 0.05)
                .then(pl.lit("* p < 0.05"))
                .otherwise(pl.lit("NS")).alias("Signif")
            )
            .select([
                "Factor", "F", "df", "PR(>F)", "Signif", "R2"
            ])  # noqa: W605
        )

        # Assign results
        self.mda_summary = summary
        self.mda_loadings = loadings
        self.mda_dim_scores = dim_scores
        self.mda_group_means = group_means

    def pca(self):
        """Execute principal component analysis.

        Notes
        -----
        Variable contribution is adapted from the FactoMineR function
        [fviz_contrib](https://search.r-project.org/CRAN/refmans/factoextra/html/fviz_contrib.html).

        """
        # scale variables
        x = self.variables.to_numpy()
        df, zero_var_idx = _safe_standardize(x, ddof=1)
        if zero_var_idx:
            logger.info(
                "Zero-variance features retained (neutral scaling) in PCA: %s",
                [self.variables.columns[i] for i in zero_var_idx],
            )

        n_samples, n_features = df.shape
        n = min(n_samples, n_features)

        U, S, Vt = np.linalg.svd(df, full_matrices=False)
        U, Vt = _svd_flip(U, Vt, u_based_decision=False)
        scores = U[:, :n] * S[:n]
        pca_df = pl.DataFrame(
            scores, schema=["PC_" + str(i) for i in range(1, n + 1)]
        )

        # Derive loadings so that standardized data
        # satisfies df @ loadings = scores.
        # This matches sklearn PCA's transform relationship and is robust to
        # numerical quirks (e.g., zero-variance columns).
        loadings, *_ = np.linalg.lstsq(df, scores, rcond=None)
        if zero_var_idx:
            loadings[zero_var_idx, :] = 0.0
        loadings_df = pl.DataFrame(
            loadings, schema=["PC_" + str(i) for i in range(1, n + 1)]
        ).select(
            pl.Series(self.variables.columns).alias("feature"),
            pl.all(),
        )

        contrib = 100.0 * (loadings ** 2)
        if zero_var_idx:
            contrib[zero_var_idx, :] = 0.0
        contrib_df = pl.DataFrame(
            contrib, schema=["PC_" + str(i) for i in range(1, n + 1)]
        ).select(
            pl.Series(self.variables.columns).alias("feature"),
            pl.all(),
        )

        if self.doc_ids is not None:
            pca_df = pca_df.select(
                        pl.Series(self.doc_ids).alias("doc_id"),
                        pl.Series(self.category_ids).alias("doc_cat"),
                        pl.all()
                        )
        else:
            pca_df = pca_df.select(
                        pl.Series(self.category_ids).alias("doc_cat"),
                        pl.all()
                        )

        denom = max(n_samples - 1, 1)
        explained_variance = (S[:n] ** 2) / denom
        total_variance = explained_variance.sum() if explained_variance.size else 0.0  # noqa: E501
        if total_variance == 0:
            var_ratio = np.zeros_like(explained_variance)
        else:
            var_ratio = explained_variance / total_variance
        ve = pl.DataFrame({
            'Dim': ["PC_" + str(i) for i in range(1, n + 1)],
            'VE (%)': var_ratio.tolist(),
        })
        ve = (
            ve
            .with_columns(pl.col('VE (%)').mul(100))
            .with_columns(pl.col('VE (%)').cum_sum().alias('VE (Total)'))
        )

        self.pca_coordinates = pca_df
        self.pca_variance_explained = ve
        self.pca_variable_contribution = contrib_df
        self.pca_loadings = loadings_df
