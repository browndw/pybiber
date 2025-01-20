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

from adjustText import adjust_text
from textwrap import dedent
from collections import Counter
from factor_analyzer import FactorAnalyzer
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn import decomposition
from statsmodels.formula.api import ols


def _get_eigenvalues(x: np.array, cor_min=0.2):
    m_cor = np.corrcoef(x.T)
    np.fill_diagonal(m_cor, 0)
    t = pl.from_numpy(m_cor).with_columns(
        pl.all().abs()
        ).max_horizontal().gt(cor_min).to_list()
    y = x.T[t].T
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=0)
    y = (y - np.mean(y, axis=0)) / np.std(y, axis=0, ddof=0)
    r_1 = np.cov(x, rowvar=False, ddof=0)
    r_2 = np.cov(y, rowvar=False, ddof=0)
    e_1, _ = np.linalg.eigh(r_1)
    e_2, _ = np.linalg.eigh(r_2)
    e_1 = pl.DataFrame({'ev_all': e_1[::-1]})
    e_2 = pl.DataFrame({'ev_mda': e_2[::-1]})
    df = pl.concat([e_1, e_2], how="horizontal")
    return df


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

        # check grouping variable
        if (
            len(self.doc_cats) == self.feature_matrix.height
        ):
            raise ValueError("""
                Invalid DataFrame.
                Expected a column of document categories.
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
            return print(dedent(
                """
                No factors to plot. Have you executed mda()?
                """
                ))
        if self.mda_group_means is not None:
            max_factor = self.mda_group_means.width - 1
        if self.mda_group_means is not None and factor > max_factor:
            return print(dedent(
                f"""
                Must specify a factor between 1 and {str(max_factor)}
                """
                ))
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
            return print(dedent(
                """
                No component to plot. Have you executed pca()?
                """
                ))
        if self.pca_coordinates is not None:
            max_pca = self.pca_coordinates.width - 1
        if self.pca_coordinates is not None and pc + 1 > max_pca:
            return print(dedent(
                f"""
                Must specify a pc between 1 and {str(max_pca - 1)}
                """
                ))

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

        if self.pca_variable_contribution is None:
            return print(dedent(
                """
                No component to plot. Have you executed pca()?
                """
                ))
        if self.pca_variable_contribution is not None:
            max_pca = self.pca_variable_contribution.width - 1
        if self.pca_variable_contribution is not None and pc > max_pca:
            return print(dedent(
                f"""
                Must specify a pc between 1 and {str(max_pca)}
                """
                ))

        df_plot = (
            self.pca_variable_contribution
            .select('feature', pc_col)
            .sort(pc_col, descending=True)
            .with_columns(
                pl.col(pc_col)
                .abs()
                .mean()
                .lt(pl.col(pc_col)
                    .abs())
                .alias('GT Mean')
                )
            .filter(pl.col('GT Mean'))
            .with_columns(
                pl.col('feature').str.replace(r"f_\d+_", "").alias("feature")
                )
            .with_columns(
                pl.col('feature').str.replace_all("_", " ").alias("feature")
                )
            )

        feature = df_plot['feature'].to_numpy()
        contribuution = df_plot[pc_col].to_numpy()
        ylimit = df_plot.get_column(pc_col).abs().ceil().max()

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        ax.bar(feature[contribuution > 0],
               contribuution[contribuution > 0],
               color='#440154', edgecolor='black', linewidth=.5)
        ax.bar(feature[contribuution < 0],
               contribuution[contribuution < 0],
               color='#21918c', edgecolor='black', linewidth=.5)

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
        m_cor = np.corrcoef(self.variables.to_numpy().T)
        np.fill_diagonal(m_cor, 0)
        t = pl.from_numpy(m_cor).with_columns(
            pl.all().abs()
            ).max_horizontal().gt(cor_min).to_list()
        m_trim = self.variables[t]

        # scale variables
        x = m_trim.to_numpy()
        m_z = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)
        # m_z = zscore(m_trim.to_numpy(), ddof=1, nan_policy='omit')

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=FutureWarning)
            fa = FactorAnalyzer(n_factors=n_factors,
                                rotation="varimax",
                                method="ml")
            fa.fit(m_trim.to_numpy())

        # convert varimax to promax
        promax_loadings = _promax(fa.loadings_)

        # aggrgate dimension scores
        pos = (promax_loadings > threshold).T
        neg = (promax_loadings < -threshold).T

        dim_scores = []
        for i in range(n_factors):
            pos_sum = np.sum(m_z.T[pos[i]], axis=0)
            neg_sum = np.sum(m_z.T[neg[i]], axis=0)
            scores = pos_sum - neg_sum
            dim_scores.append(scores)

        dim_scores = pl.from_numpy(
            np.array(dim_scores).T, schema=[
                "factor_" + str(i) for i in range(1, n_factors + 1)
                ]
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
        df = (x - np.mean(x, axis=0)) / np.std(x, axis=0, ddof=1)

        # get number of components
        n = min(df.shape)

        pca = decomposition.PCA(n_components=n)
        pca_result = pca.fit_transform(df)
        pca_df = pl.DataFrame(
            pca_result, schema=["PC_" + str(i) for i in range(1, n + 1)]
            )

        # calculate variable contribution
        # this is a Python implementation adapted from FactoMineR
        # https://github.com/husson/FactoMineR/blob/master/R/PCA.R
        sdev = pca_df.std(ddof=0).to_numpy().T
        contrib = []
        for i in range(0, len(sdev)):
            coord = pca.components_[i] * sdev[i]
            polarity = np.divide(coord, abs(coord))
            coord = np.square(coord)
            coord = np.divide(coord, sum(coord))*100
            coord = np.multiply(coord, polarity)
            contrib.append(coord)

        contrib_df = pl.DataFrame(
            contrib, schema=["PC_" + str(i) for i in range(1, n + 1)]
            )
        contrib_df = contrib_df.select(
                    pl.Series(self.variables.columns).alias("feature"),
                    pl.all()
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
