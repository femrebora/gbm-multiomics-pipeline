"""
visualization/oncoprint.py — Mutation landscape (oncoprint/waterfall) plots.

Visualizes the mutation landscape of top GBM driver genes across samples,
sorted by molecular subtype or IDH status.

References
----------
  Gu et al. (2016) Bioinformatics 32:2847-2849 — ComplexHeatmap oncoprint
  Brennan et al. (2013) Cell 155:462-477 — GBM mutation landscape
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    MUTATION_COLORS,
    SUBTYPE_COLORS,
    figure_size,
    save_figure,
    set_publication_style,
)


def oncoprint(
    mutation_matrix: pd.DataFrame,
    clinical: pd.DataFrame | None = None,
    top_genes: int = 20,
    sort_by: str | None = None,
    title: str = "GBM Mutation Landscape",
    output_dir: Path | None = None,
    filename: str = "oncoprint",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Generate an oncoprint (mutation landscape) plot.

    Parameters
    ----------
    mutation_matrix : pd.DataFrame
        Genes × samples. Values should be mutation types
        (e.g. "Missense_Mutation", "Nonsense_Mutation", etc.)
        or NaN/empty for no mutation.
    clinical : pd.DataFrame, optional
        Sample annotations for top bar.
    top_genes : int
        Number of top mutated genes to show.
    sort_by : str, optional
        Column in clinical to sort samples by.
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
    from matplotlib.patches import Rectangle

    set_publication_style()

    # Select top mutated genes
    mutation_freq = mutation_matrix.notna().sum(axis=1)
    top = mutation_freq.nlargest(top_genes).index
    mut = mutation_matrix.loc[top].copy()

    # Sort samples
    sample_order = mut.columns.tolist()
    if sort_by and clinical is not None and sort_by in clinical.columns:
        clinical = clinical.set_index(
            clinical.columns[0] if clinical.columns[0] != sort_by else clinical.columns[0]
        )
        common = [s for s in sample_order if s in clinical.index]
        sample_order = clinical.loc[common].sort_values(sort_by).index.tolist()

    mut = mut[sample_order]

    # Determine figure size
    if figsize is None:
        n_samples = len(sample_order)
        w = max(10, n_samples * 0.08)
        h = max(4, top_genes * 0.3 + 1)
        figsize = (min(w, 20), min(h, 14))

    fig, ax = plt.subplots(figsize=figsize)

    # Map mutation types to numeric codes and colors
    type_to_code: dict[str, int] = {}
    type_to_color: dict[str, str] = {}
    code = 0
    for _, row in mut.iterrows():
        for val in row.dropna().unique():
            if val not in type_to_code:
                type_to_code[val] = code
                type_to_color[val] = MUTATION_COLORS.get(val, "#BDBDBD")
                code += 1

    # Draw mutation rectangles
    n_genes = len(mut)
    n_samples = len(sample_order)

    for i, (gene, row) in enumerate(mut.iterrows()):
        for j, sample in enumerate(sample_order):
            val = row.get(sample)
            if pd.notna(val) and val:
                color = type_to_color.get(val, "#BDBDBD")
                rect = Rectangle(
                    (j, i), 1, 0.8,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.add_patch(rect)

    # Gene labels
    ax.set_yticks(np.arange(n_genes) + 0.4)
    ax.set_yticklabels(mut.index, fontsize=8)

    # Sample labels (hidden if too many)
    if n_samples <= 60:
        ax.set_xticks(np.arange(n_samples) + 0.5)
        ax.set_xticklabels(
            [s[:16] + "…" if len(s) > 16 else s for s in sample_order],
            rotation=90, fontsize=5, ha="center",
        )
    else:
        ax.set_xticks([])

    # Axes limits
    ax.set_xlim(0, n_samples)
    ax.set_ylim(0, n_genes)
    ax.invert_yaxis()

    # Legend
    legend_patches = [
        Rectangle((0, 0), 1, 1, facecolor=type_to_color[t], edgecolor="white")
        for t in type_to_code
    ]
    ax.legend(
        legend_patches, list(type_to_code.keys()),
        fontsize=7, loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        title="Mutation Type",
        title_fontsize=8,
    )

    # Mutation frequency bar on the right
    freq_pct = mut.notna().sum(axis=1) / n_samples * 100
    ax_freq = ax.twiny()
    ax_freq.barh(
        np.arange(n_genes) + 0.4, freq_pct.values,
        height=0.7, color="#2166ac", alpha=0.7,
    )
    ax_freq.set_xlim(0, max(freq_pct.max() * 1.2, 101))
    ax_freq.set_xlabel("% Mutated", fontsize=8)

    # TMB bar on top
    if n_samples <= 100:
        ax_tmb = ax.twinx()
        # Count mutations per sample
        tmb = mut.notna().sum(axis=0)
        ax_tmb.bar(
            np.arange(n_samples) + 0.5, tmb.values,
            width=0.8, color="#999999", alpha=0.5,
        )
        ax_tmb.set_ylabel("# Mutations", fontsize=8)

    ax.set_title(f"{title}\nTop {top_genes} Mutated Genes "
                 f"({n_samples} samples)",
                 fontsize=11, fontweight="bold")

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig
