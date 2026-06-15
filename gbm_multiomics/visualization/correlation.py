"""
visualization/correlation.py — Cross-omics correlation visualizations.

Generates correlation matrices, scatter plots, and circos-style
multi-omics integration figures.

References
----------
  Ceccarelli et al. (2016) Cell 164:550-563 — GBM multi-omics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    OMICS_COLORS,
    figure_size,
    save_figure,
    set_publication_style,
)


def cross_omics_scatter(
    x_data: pd.Series,
    y_data: pd.Series,
    x_label: str = "RNA Expression",
    y_label: str = "Copy Number",
    title: str = "Cross-Omics Correlation",
    highlight_genes: list[str] | None = None,
    output_dir: Path | None = None,
    filename: str = "cross_omics_scatter",
) -> plt.Figure:
    """
    Scatter plot of two omics measurements per gene.

    Parameters
    ----------
    x_data, y_data : pd.Series
        Gene-indexed values from two omics layers.
    x_label, y_label : str
    title : str
    highlight_genes : list[str], optional
        Genes to label on the plot.
    output_dir : Path, optional
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    set_publication_style()

    common = x_data.index.intersection(y_data.index)
    x = x_data.loc[common].astype(float)
    y = y_data.loc[common].astype(float)

    valid = x.notna() & y.notna()
    x, y = x[valid], y[valid]

    r, p = pearsonr(x, y)
    rho, p_s = spearmanr(x, y)

    fig, ax = plt.subplots(figsize=figure_size("single_column", aspect=1.0))

    ax.scatter(x, y, s=4, alpha=0.3, rasterized=True, color="#333333")

    # Trend line
    from numpy.polynomial.polynomial import polyfit
    try:
        b, m = polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b + m * xs, color="#E41A1C", linewidth=1.5, alpha=0.8)
    except Exception:
        pass

    # Highlight specific genes
    if highlight_genes:
        for gene in highlight_genes:
            if gene in common:
                ax.scatter(x[gene], y[gene], s=50, color="#FF7F00",
                           edgecolors="black", linewidths=0.5, zorder=5)
                ax.annotate(
                    gene,
                    (x[gene], y[gene]),
                    fontsize=6, fontweight="bold",
                    ha="center", va="bottom",
                )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\nPearson r = {r:.3f}, Spearman ρ = {rho:.3f}",
                 fontsize=10)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig


def omics_variance_decomposition(
    variance_df: pd.DataFrame,
    title: str = "Variance Explained per Omics Layer",
    output_dir: Path | None = None,
    filename: str = "variance_decomposition",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Stacked bar chart showing variance decomposition across omics layers.

    Parameters
    ----------
    variance_df : pd.DataFrame
        Factors × omics layers, values = variance explained (0-1).
    title : str
    output_dir : Path, optional
    filename : str
    figsize : tuple, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    if figsize is None:
        figsize = (len(variance_df.columns) * 1.5 + 4,
                   max(4, len(variance_df) * 0.5 + 2))

    fig, ax = plt.subplots(figsize=figsize)

    colors = [OMICS_COLORS.get(c, f"C{i}")
              for i, c in enumerate(variance_df.columns)]

    variance_df.plot(
        kind="barh", stacked=True, ax=ax,
        color=colors,
        edgecolor="white", linewidth=0.5,
    )

    ax.set_xlabel("Variance Explained")
    ax.set_ylabel("Factor")
    ax.set_title(title)
    ax.legend(
        fontsize=8, title="Omics Layer",
        title_fontsize=9, loc="lower right",
    )
    ax.set_xlim(0, variance_df.sum(axis=1).max() * 1.15)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig


def omics_correlation_heatmap(
    corr_dict: dict[str, pd.DataFrame],
    title: str = "Cross-Omics Correlation Summary",
    output_dir: Path | None = None,
    filename: str = "omics_correlation_heatmap",
) -> plt.Figure:
    """
    Summary heatmap of correlations across omics pairs.

    Parameters
    ----------
    corr_dict : dict[str, pd.DataFrame]
        From cross_omics_correlation(). Keys like "rna_cnv", "rna_methylation", etc.
    title : str
    output_dir : Path, optional
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    set_publication_style()

    # Build summary matrix: omics pair → median |r|
    summary = {}
    for pair_name, df in corr_dict.items():
        if "pearson_r" in df.columns:
            summary[pair_name] = df["pearson_r"].abs().median()
        elif isinstance(df, pd.DataFrame) and not df.empty:
            # Correlation matrix case
            vals = df.values.flatten()
            vals = vals[~np.isnan(vals)]
            summary[pair_name] = np.median(np.abs(vals))

    if not summary:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No correlations to display", ha="center", va="center")
        return fig

    # Build pair matrix
    omics = sorted(set(
        [p.split("_")[0] for p in summary] +
        [p.split("_")[1] for p in summary if "_" in p]
    ))

    matrix = pd.DataFrame(np.nan, index=omics, columns=omics)
    for pair_name, corr_val in summary.items():
        parts = pair_name.split("_")
        if len(parts) >= 2:
            a, b = parts[0], parts[1]
            if a in matrix.index and b in matrix.columns:
                matrix.loc[a, b] = corr_val
                matrix.loc[b, a] = corr_val

    fig, ax = plt.subplots(figsize=(len(omics) * 1.5 + 2, len(omics) * 1.5 + 1))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".3f",
        cmap="YlOrRd", vmin=0, vmax=1,
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Median |r|", "shrink": 0.8},
    )
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig
