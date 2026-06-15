"""
visualization/heatmap.py — Publication-quality complex heatmaps.

Generates heatmaps with clinical annotations, row/column clustering,
and configurable color scales. Designed for GBM multi-omics data.

References
----------
  Gu et al. (2016) Bioinformatics 32:2847-2849 — ComplexHeatmap (R)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    CLINICAL_COLORS,
    SUBTYPE_COLORS,
    figure_size,
    save_figure,
    set_publication_style,
)


def complex_heatmap(
    data: pd.DataFrame,
    column_annotations: pd.DataFrame | None = None,
    row_annotations: pd.DataFrame | None = None,
    annotation_colors: dict[str, dict[str, str]] | None = None,
    n_top_genes: int = 50,
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    show_row_names: bool = False,
    show_col_names: bool = False,
    row_names_fontsize: int = 6,
    z_score: bool = True,
    cmap: str = "RdBu_r",
    center: float = 0.0,
    title: str = "GBM Multi-Omics Heatmap",
    output_dir: Path | None = None,
    filename: str = "complex_heatmap",
    figsize: tuple[float, float] | None = None,
    rasterized: bool = True,
) -> plt.Figure:
    """
    Generate a publication-quality complex heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Genes/features × samples matrix to display.
    column_annotations : pd.DataFrame, optional
        Samples × annotation columns. Each column adds a color bar.
    row_annotations : pd.DataFrame, optional
        Features × annotation columns.
    annotation_colors : dict, optional
        {annotation_col: {category: color}}.
    n_top_genes : int
        If data has more rows, use top N variable rows.
    cluster_rows, cluster_cols : bool
        Apply hierarchical clustering.
    show_row_names, show_col_names : bool
    row_names_fontsize : int
    z_score : bool
        Z-score normalize rows before plotting.
    cmap : str
        Matplotlib colormap.
    center : float
        Center value for divergence colormap.
    title : str
    output_dir : Path, optional
    filename : str
    figsize : tuple, optional
    rasterized : bool
        Rasterize the heatmap cells for smaller file size.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    set_publication_style()

    # Select top variable genes if needed
    if n_top_genes and len(data) > n_top_genes:
        var = data.var(axis=1)
        top_idx = var.nlargest(n_top_genes).index
        data = data.loc[top_idx]

    # Z-score normalize rows
    if z_score:
        data = data.sub(data.mean(axis=1), axis=0)
        std = data.std(axis=1).replace(0, 1)
        data = data.div(std, axis=0)

    # Clustering
    row_order = list(range(len(data)))
    col_order = list(range(len(data.columns)))

    if cluster_rows:
        row_linkage = linkage(pdist(data.values), method="ward")
        row_order = leaves_list(row_linkage)
        data = data.iloc[row_order]

    if cluster_cols:
        col_linkage = linkage(pdist(data.T.values), method="ward")
        col_order = leaves_list(col_linkage)
        data = data.iloc[:, col_order]

    # Reorder annotations to match
    if column_annotations is not None:
        column_annotations = column_annotations.loc[data.columns]

    # Determine figure size
    if figsize is None:
        w = max(8, len(data.columns) * 0.12)
        h = max(5, len(data) * 0.12)
        # Add annotation height
        if column_annotations is not None:
            h += 0.3 * len(column_annotations.columns)
        figsize = (min(w, 20), min(h, 16))

    # Setup grid for annotations
    n_annot_rows = 0
    if column_annotations is not None:
        n_annot_rows = len(column_annotations.columns)

    if n_annot_rows > 0:
        gs_kw = {"height_ratios": [0.3] * n_annot_rows + [10]}
        fig, axes = plt.subplots(
            n_annot_rows + 1, 1,
            figsize=figsize,
            gridspec_kw=gs_kw,
            squeeze=False,
        )
        ax_heatmap = axes[-1][0]
        annot_axes = [axes[i][0] for i in range(n_annot_rows)]
    else:
        fig, ax_heatmap = plt.subplots(figsize=figsize)
        annot_axes = []

    # Plot heatmap
    im = ax_heatmap.imshow(
        data.values,
        aspect="auto",
        cmap=cmap,
        vmin=-center if center != 0 else None,
        vmax=center if center != 0 else None,
        rasterized=rasterized,
    )

    # Row/column labels
    if show_row_names and len(data) <= 100:
        ax_heatmap.set_yticks(range(len(data)))
        ax_heatmap.set_yticklabels(data.index, fontsize=row_names_fontsize)
    else:
        ax_heatmap.set_yticks([])

    if show_col_names and len(data.columns) <= 100:
        ax_heatmap.set_xticks(range(len(data.columns)))
        ax_heatmap.set_xticklabels(
            data.columns,
            fontsize=5,
            rotation=90,
            ha="center",
        )
    else:
        ax_heatmap.set_xticks([])

    # Colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
    cbar.set_label("Z-score" if z_score else "Expression", fontsize=9)

    # Column annotations
    if column_annotations is not None:
        for i, col in enumerate(column_annotations.columns):
            ax = annot_axes[i]
            values = column_annotations[col]
            unique_vals = values.dropna().unique()

            # Get colors
            colors = annotation_colors.get(col, {}) if annotation_colors else {}
            if not colors:
                # Auto-detect from built-in palettes
                if col in SUBTYPE_COLORS:
                    colors = SUBTYPE_COLORS
                elif col in CLINICAL_COLORS:
                    colors = CLINICAL_COLORS
                else:
                    # Generate from tab10
                    cm = plt.get_cmap("tab10")
                    colors = {v: cm(j % 10) for j, v in enumerate(unique_vals)}

            # Create annotation heatmap (1 row per annotation)
            annot_array = np.zeros((1, len(values)))
            for j, v in enumerate(unique_vals):
                annot_array[0, values == v] = j

            ax.imshow(annot_array, aspect="auto", cmap=plt.get_cmap("tab10"),
                      vmin=0, vmax=max(len(unique_vals) - 1, 1))
            ax.set_ylabel(col, fontsize=8, rotation=0, ha="right", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

    ax_heatmap.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax_heatmap.set_xlabel(f"Samples (n={len(data.columns)})")
    ax_heatmap.set_ylabel(f"Features (n={len(data)})")

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig


def correlation_heatmap_plot(
    corr_matrix: pd.DataFrame,
    title: str = "Cross-Omics Correlation",
    cmap: str = "RdBu_r",
    center: float = 0.0,
    annotate: bool = False,
    annotate_threshold: float = 0.0,
    output_dir: Path | None = None,
    filename: str = "correlation_heatmap",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Generate a publication-quality correlation heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Square or rectangular correlation matrix.
    title, cmap, center : standard options
    annotate : bool
        If True, annotate cells with correlation values.
    annotate_threshold : float
        Only annotate cells with |r| > threshold.
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
    import seaborn as sns

    set_publication_style()

    if figsize is None:
        n = max(len(corr_matrix), len(corr_matrix.columns))
        size = min(max(6, n * 0.3), 20)
        figsize = (size, size * 0.9)

    fig, ax = plt.subplots(figsize=figsize)

    # Annotation matrix
    annot = None
    if annotate and annotate_threshold > 0:
        annot = corr_matrix.where(
            lambda x: abs(x) >= annotate_threshold
        ).round(2).astype(str)
        annot[annot == "nan"] = ""

    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=cmap,
        center=center,
        annot=annot,
        fmt="",
        annot_kws={"fontsize": 7} if annot is not None else None,
        linewidths=0.1,
        linecolor="white",
        xticklabels=len(corr_matrix.columns) <= 50,
        yticklabels=len(corr_matrix) <= 50,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )

    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.tick_params(axis="both", labelsize=7)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig
