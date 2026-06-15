"""
volcano.py — Publication-quality volcano plots for GBM differential expression.

Generates EnhancedVolcano-style plots with labeled top genes, GBM driver
highlighting, and consistent academic styling.

References
----------
  Blighe et al. (2024) EnhancedVolcano R package
  Love et al. (2014) Genome Biology 15:550 — DESeq2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    DE_COLORS,
    GBM_DRIVER_GENES,
    figure_size,
    save_figure,
    set_publication_style,
)


def volcano_plot(
    de_results: pd.DataFrame,
    padj_col: str = "padj",
    lfc_col: str = "log2FoldChange",
    gene_col: str | None = None,
    padj_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    n_label: int = 20,
    title: str = "GBM Differential Expression",
    highlight_drivers: bool = True,
    output_dir: Path | None = None,
    filename: str = "volcano_plot",
    figsize: tuple[float, float] | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """
    Generate a publication-quality volcano plot.

    Parameters
    ----------
    de_results : pd.DataFrame
        DE results with padj_col, lfc_col. Index = gene IDs/names.
    padj_col : str
        Column with adjusted p-values.
    lfc_col : str
        Column with log2 fold changes.
    gene_col : str, optional
        Column with gene symbols for labeling. If None, uses DataFrame index.
    padj_threshold : float
        FDR cutoff for significance.
    lfc_threshold : float
        |log2FC| cutoff for biological significance.
    n_label : int
        Number of top genes to label.
    title : str
    highlight_drivers : bool
        If True, color GBM driver genes in orange.
    output_dir : Path, optional
    filename : str
        Base filename (without extension).
    figsize : tuple, optional

    Returns
    -------
    (fig, labeled_df)
        fig : matplotlib Figure
        labeled_df : DataFrame of genes that were labeled on the plot
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    df = de_results.copy()

    # Ensure required columns exist
    if padj_col not in df.columns:
        raise ValueError(f"Column '{padj_col}' not found in DE results.")
    if lfc_col not in df.columns:
        raise ValueError(f"Column '{lfc_col}' not found in DE results.")

    # Drop NA
    df = df.dropna(subset=[padj_col, lfc_col])

    # Compute -log10(padj), handling zeros
    df["neg_log10_padj"] = -np.log10(df[padj_col].clip(lower=1e-300))

    # Classify genes
    is_sig = df[padj_col] < padj_threshold
    is_up = is_sig & (df[lfc_col] > lfc_threshold)
    is_down = is_sig & (df[lfc_col] < -lfc_threshold)

    df["direction"] = "NS"
    df.loc[is_up, "direction"] = "UP"
    df.loc[is_down, "direction"] = "DOWN"

    # Identify GBM drivers
    gene_names = df[gene_col] if gene_col else df.index
    df["is_driver"] = [str(g) in GBM_DRIVER_GENES for g in gene_names]

    # ── Select genes to label ────────────────────────────────────────────
    # Top by padj
    top_padj = df.nsmallest(n_label, padj_col).index
    # Top by |lfc|
    top_lfc = df.loc[is_sig].copy()
    top_lfc["abs_lfc"] = top_lfc[lfc_col].abs()
    top_lfc = top_lfc.nlargest(n_label, "abs_lfc").index
    # GBM drivers that are significant
    driver_sig = df[df["is_driver"] & is_sig].index

    label_idx = set(top_padj) | set(top_lfc) | set(driver_sig)
    df["label"] = [str(g) if i in label_idx else "" for i, g in zip(df.index, gene_names)]

    # ── Plot ──────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = figure_size("double_column", aspect=1.0)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot NS genes (grey)
    ns = df[df["direction"] == "NS"]
    ax.scatter(ns[lfc_col], ns["neg_log10_padj"],
               c=DE_COLORS["NS"], s=3, alpha=0.3, rasterized=True,
               label="NS")

    # Plot UP genes (red)
    up = df[df["direction"] == "UP"]
    ax.scatter(up[lfc_col], up["neg_log10_padj"],
               c=DE_COLORS["UP"], s=8, alpha=0.5, rasterized=True,
               label=f"UP ({len(up)})")

    # Plot DOWN genes (blue)
    down = df[df["direction"] == "DOWN"]
    ax.scatter(down[lfc_col], down["neg_log10_padj"],
               c=DE_COLORS["DOWN"], s=8, alpha=0.5, rasterized=True,
               label=f"DOWN ({len(down)})")

    # Highlight GBM drivers
    if highlight_drivers:
        drivers = df[df["is_driver"] & is_sig]
        ax.scatter(drivers[lfc_col], drivers["neg_log10_padj"],
                   c=DE_COLORS["GBM_driver"], s=25, alpha=0.9,
                   edgecolors="black", linewidths=0.5, zorder=5,
                   label=f"GBM driver ({len(drivers)})")

    # ── Threshold lines ──────────────────────────────────────────────────
    max_lfc = df[lfc_col].abs().max() * 1.1
    ax.axhline(-np.log10(padj_threshold), color="grey", linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.axvline(lfc_threshold, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.7)
    ax.axvline(-lfc_threshold, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.7)

    # ── Gene labels ──────────────────────────────────────────────────────
    labeled = df[df["label"] != ""]
    try:
        from adjustText import adjust_text
        texts = []
        for _, row in labeled.iterrows():
            color = DE_COLORS["GBM_driver"] if row["is_driver"] else "black"
            weight = "bold" if row["is_driver"] else "normal"
            texts.append(ax.text(
                row[lfc_col], row["neg_log10_padj"],
                row["label"], fontsize=6, color=color,
                fontweight=weight, ha="center", va="bottom",
            ))
        if texts:
            adjust_text(
                texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="grey", lw=0.5),
                force_text=(0.5, 1.0),
                expand=(1.2, 1.5),
            )
    except ImportError:
        # Fallback: label top 5 without adjustment
        for _, row in labeled.head(5).iterrows():
            ax.annotate(
                row["label"],
                (row[lfc_col], row["neg_log10_padj"]),
                fontsize=7, fontweight="bold",
                ha="center", va="bottom",
            )

    # ── Labels & title ───────────────────────────────────────────────────
    n_up = len(up)
    n_down = len(down)
    n_total = len(df)

    ax.set_xlabel("log$_2$ Fold Change")
    ax.set_ylabel("−log$_{10}$(adjusted p-value)")
    ax.set_title(f"{title}\n"
                 f"FDR < {padj_threshold}, |log$_2$FC| > {lfc_threshold} "
                 f"({n_up} ⬆, {n_down} ⬇, of {n_total} total)",
                 fontsize=10)

    ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
              markerscale=0.8)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    labeled_df = labeled[["label", lfc_col, padj_col, "direction"]].copy()

    return fig, labeled_df


def multi_volcano(
    de_results: dict[str, pd.DataFrame],
    padj_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
    n_label: int = 10,
    output_dir: Path | None = None,
    n_cols: int = 2,
) -> plt.Figure:
    """
    Generate a multi-panel volcano plot for multiple comparisons.

    Parameters
    ----------
    de_results : dict
        {comparison_label: de_results_DataFrame}
    padj_threshold : float
    lfc_threshold : float
    n_label : int
    output_dir : Path, optional
    n_cols : int
        Number of columns in the multi-panel layout.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    n = len(de_results)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 4.5, n_rows * 4),
        squeeze=False,
    )

    for idx, (label, df) in enumerate(de_results.items()):
        ax = axes[idx // n_cols][idx % n_cols]

        df = df.dropna(subset=["padj", "log2FoldChange"])
        df["neg_log10_padj"] = -np.log10(df["padj"].clip(lower=1e-300))

        is_sig = df["padj"] < padj_threshold
        is_up = is_sig & (df["log2FoldChange"] > lfc_threshold)
        is_down = is_sig & (df["log2FoldChange"] < -lfc_threshold)

        n_up = is_up.sum()
        n_down = is_down.sum()

        # NS
        ns = df[~is_sig | ((df["log2FoldChange"].abs() <= lfc_threshold) & is_sig)]
        ax.scatter(ns["log2FoldChange"], ns["neg_log10_padj"],
                   c=DE_COLORS["NS"], s=2, alpha=0.3, rasterized=True)

        # UP
        up = df[is_up]
        ax.scatter(up["log2FoldChange"], up["neg_log10_padj"],
                   c=DE_COLORS["UP"], s=6, alpha=0.5, rasterized=True)

        # DOWN
        down = df[is_down]
        ax.scatter(down["log2FoldChange"], down["neg_log10_padj"],
                   c=DE_COLORS["DOWN"], s=6, alpha=0.5, rasterized=True)

        # Label top genes
        top = df[is_sig].nsmallest(n_label, "padj")
        for gene, row in top.iterrows():
            ax.annotate(
                str(gene),
                (row["log2FoldChange"], row["neg_log10_padj"]),
                fontsize=5, ha="center", va="bottom",
            )

        # Threshold lines
        ax.axhline(-np.log10(padj_threshold), color="grey", linestyle="--",
                   linewidth=0.5, alpha=0.5)
        ax.axvline(lfc_threshold, color="grey", linestyle="--",
                   linewidth=0.5, alpha=0.5)
        ax.axvline(-lfc_threshold, color="grey", linestyle="--",
                   linewidth=0.5, alpha=0.5)

        ax.set_xlabel("log$_2$FC")
        ax.set_ylabel("−log$_{10}$(padj)")
        ax.set_title(f"{label}\n({n_up} ⬆, {n_down} ⬇)", fontsize=9)

    # Hide unused subplots
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, "multi_volcano", output_dir)

    return fig
