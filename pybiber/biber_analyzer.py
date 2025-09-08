"""
Carry our specific implemations of exploratory factor analysis
from a parsed corpus.
.. codeauthor:: David Brown <dwb2d@andrew.cmu.edu>
"""

import warnings
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import statsmodels.api as sm
import logging
import importlib.resources as resources

from adjustText import adjust_text
from collections import Counter
from factor_analyzer import FactorAnalyzer
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
from statsmodels.formula.api import ols

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic handler (library users can reconfigure)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _safe_standardize(x: np.ndarray, ddof: int = 1, eps: float = 1e-12):
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


def _get_eigenvalues(x: np.array, cor_min=0.2):
    # Guard against insufficient data (need at least 2 observations & 2 vars)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 2:
        return pl.DataFrame({"ev_all": [], "ev_mda": []})
    try:
        with np.errstate(divide="ignore", invalid="ignore"):
            m_cor = np.corrcoef(x.T)
        np.fill_diagonal(m_cor, 0)
        t = (
            pl.from_numpy(m_cor)
            .with_columns(pl.all().abs())
            .max_horizontal()
            .gt(cor_min)
            .to_list()
        )
        # If filtering removes all columns, fallback to original set
        if not any(t):
            t = [True] * x.shape[1]
        y = x.T[t].T
    # Standardize using safe routine (ddof=0 for backward compatibility)
        x_z, _ = _safe_standardize(x, ddof=0)
        y_z, _ = _safe_standardize(y, ddof=0)
        r_1 = np.cov(x_z, rowvar=False, ddof=0)
        r_2 = np.cov(y_z, rowvar=False, ddof=0)
        e_1, _ = np.linalg.eigh(r_1)
        e_2, _ = np.linalg.eigh(r_2)
        e_1 = pl.DataFrame({'ev_all': e_1[::-1]})
        e_2 = pl.DataFrame({'ev_mda': e_2[::-1]})
        df = pl.concat([e_1, e_2], how="horizontal")
        return df
    except Exception:
        # Fallback: empty result on numerical failure
        return pl.DataFrame({"ev_all": [], "ev_mda": []})


# adapted from the R stats package
# https://github.com/SurajGupta/r-source/blob/master/src/library/stats/R/factanal.R
def _promax(x: np.array, m=4):
    Q = x * np.abs(x)**(m-1)
    model = LinearRegression(fit_intercept=False)
    model.fit(x, Q)
    U = model.coef_.T
    d = np.diag(np.linalg.inv(np.dot(U.T, U)))
    U = U * np.sqrt(d)
    promax_loadings = np.dot(x, U)
    return promax_loadings


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
            threshold: float = 0.35):

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

        # scale variables (safe)
        x = m_trim.to_numpy()
        m_z, zero_var_idx = _safe_standardize(x, ddof=1)
        if zero_var_idx:
            logger.info(
                "Zero-variance features retained (neutral scaling) in MDA: %s",
                [m_trim.columns[i] for i in zero_var_idx],
            )
        # m_z = zscore(m_trim.to_numpy(), ddof=1, nan_policy='omit')

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            if n_factors == 1:
                warnings.filterwarnings(
                    "ignore",
                    message="No rotation will be performed",
                    category=UserWarning,
                )
            fa = FactorAnalyzer(
                n_factors=n_factors,
                rotation="varimax",
                method="ml"
            )
            fa.fit(m_trim.to_numpy())

        # convert varimax to promax
        promax_loadings = _promax(fa.loadings_)

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

            model = ols(
                "response ~ group", data={"response": y, "group": X}
                ).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            factor_summary = (pl.DataFrame(
                anova_table
                )
                .cast({"df": pl.UInt32})
                .with_columns(
                    pl.col('df')
                    .shift(-1).alias('df2').cast(pl.UInt32)
                    )
                .with_columns(df=pl.concat_list("df", "df2"))
                .with_columns(R2=pl.lit(model.rsquared))
                .with_columns(Factor=pl.lit(factor_col))
                .select(['Factor', 'F', "df", "PR(>F)", "R2"])
                ).head(1)
            summary.append(factor_summary)
        summary = pl.concat(summary)
        summary = (
            summary.with_columns(
                pl.when(
                    pl.col("PR(>F)") < 0.05,
                    pl.col("PR(>F)") > 0.01
                    )
                .then(pl.lit("* p < 0.05"))
                .when(
                    pl.col("PR(>F)") < 0.01,
                    pl.col("PR(>F)") > 0.001
                    )
                .then(pl.lit("** p < 0.01"))
                .when(
                    pl.col("PR(>F)") < 0.001,
                    )
                .then(pl.lit("*** p < 0.001"))
                .otherwise(pl.lit("NS"))
                .alias("Signif")
                ).select(['Factor', 'F', "df", "PR(>F)", "Signif", "R2"])
                )
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
            try:
                model = ols(
                    "response ~ group", data={"response": y, "group": X}
                ).fit()
                anova_table = sm.stats.anova_lm(model, typ=2)
                factor_summary = (
                    pl.DataFrame(anova_table)
                    .cast({"df": pl.UInt32})
                    .with_columns(
                        pl.col("df").shift(-1).alias("df2").cast(pl.UInt32)
                    )
                    .with_columns(df=pl.concat_list("df", "df2"))
                    .with_columns(R2=pl.lit(model.rsquared))
                    .with_columns(Factor=pl.lit(factor_col))
                    .select([
                        "Factor", "F", "df", "PR(>F)", "R2"
                    ])  # noqa: W605
                ).head(1)
            except Exception:
                # Fallback in edge cases with single group etc.
                factor_summary = pl.DataFrame({
                    "Factor": [factor_col],
                    "F": [np.nan],
                    "df": [[0, 0]],
                    "PR(>F)": [np.nan],
                    "R2": [np.nan],
                })
            summary.append(factor_summary)
        summary = pl.concat(summary)
        summary = (
            summary.with_columns(
                pl.when(pl.col("PR(>F)") < 0.05, pl.col("PR(>F)") > 0.01)
                .then(pl.lit("* p < 0.05"))
                .when(pl.col("PR(>F)") < 0.01, pl.col("PR(>F)") > 0.001)
                .then(pl.lit("** p < 0.01"))
                .when(pl.col("PR(>F)") < 0.001)
                .then(pl.lit("*** p < 0.001"))
                .otherwise(pl.lit("NS"))
                .alias("Signif")
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
        This is largely a convenience function as most of its outputs
        are produced by wrappers for sklearn. However,
        variable contribution is adapted from the FactoMineR function
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

        # get number of components
        n = min(df.shape)

        pca = decomposition.PCA(n_components=n)
        pca_result = pca.fit_transform(df)
        pca_df = pl.DataFrame(
            pca_result, schema=["PC_" + str(i) for i in range(1, n + 1)]
        )

        # Raw loadings (eigenvectors), shape: (features, components)
        loadings = pca.components_.T
        # Zero-variance features should not contribute to any component.
        # Force their loadings to 0 to avoid arbitrary basis vectors
        # in the null space producing 100% contribution on a late PC.
        if zero_var_idx:
            loadings[zero_var_idx, :] = 0.0
        loadings_df = pl.DataFrame(
            loadings, schema=["PC_" + str(i) for i in range(1, n + 1)]
        ).select(
            pl.Series(self.variables.columns).alias("feature"),
            pl.all(),
        )

        # Variable contribution for correlation PCA (FactoMineR):
        # contrib[j, k] = 100 * loading[j, k]^2
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

        ve = np.array(pca.explained_variance_ratio_).tolist()
        ve = pl.DataFrame(
            {'Dim': [
                "PC_" + str(i) for i in range(1, len(ve) + 1)
                ], 'VE (%)': ve}
            )
        ve = (ve
              .with_columns(pl.col('VE (%)').mul(100))
              .with_columns(pl.col('VE (%)').cum_sum().alias('VE (Total)'))
              )

        self.pca_coordinates = pca_df
        self.pca_variance_explained = ve
        self.pca_variable_contribution = contrib_df
        self.pca_loadings = loadings_df
