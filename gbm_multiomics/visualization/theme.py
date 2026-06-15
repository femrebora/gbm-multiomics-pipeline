"""
theme.py — Publication-quality visualization theme for GBM multiomics figures.

Provides consistent color palettes, font settings, DPI, and figure sizing
for all pipeline visualizations. Designed for direct journal submission.

Color palettes
--------------
  GBM subtypes: Classical (red), Mesenchymal (blue), Proneural (green), Neural (purple)
  Clinical: IDH-wildtype (orange), IDH-mutant (teal), MGMT methylated (blue), unmethylated (pink)
  DE direction: UP (red), DOWN (blue), NS (grey)
  Survival: High risk (red), Low risk (blue)

References
----------
  Verhaak et al. (2010) Cancer Cell 17:98-110 — subtype colors from TCGA
  Hegi et al. (2005) NEJM 352:997-1003 — MGMT methylation
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── Color palettes ───────────────────────────────────────────────────────────

# GBM molecular subtypes (Verhaak 2010 / TCGA 2013)
SUBTYPE_COLORS: dict[str, str] = {
    "Classical":   "#E41A1C",  # red
    "Mesenchymal": "#377EB8",  # blue
    "Proneural":   "#4DAF4A",  # green
    "Neural":      "#984EA3",  # purple
}

# Clinical / molecular groups
CLINICAL_COLORS: dict[str, str] = {
    "IDH_wildtype":      "#D55E00",  # orange-red
    "IDH_mutant":        "#009E73",  # teal
    "MGMT_methylated":   "#0072B2",  # blue
    "MGMT_unmethylated": "#CC79A7",  # pink
    "Tumor":             "#E41A1C",
    "Normal":            "#377EB8",
}

# Differential expression
DE_COLORS: dict[str, str] = {
    "UP":               "#E41A1C",  # red
    "DOWN":             "#377EB8",  # blue
    "NS":               "#BDBDBD",  # grey
    "GBM_driver":       "#FF7F00",  # orange highlight
}

# Survival risk groups
RISK_COLORS: dict[str, str] = {
    "High": "#E41A1C",
    "Low":  "#377EB8",
    "High Risk": "#E41A1C",
    "Low Risk":  "#377EB8",
}

# Mutation types (oncoprint)
MUTATION_COLORS: dict[str, str] = {
    "Missense_Mutation":      "#2E8B57",  # sea green
    "Nonsense_Mutation":      "#000000",  # black
    "Frame_Shift_Del":        "#E41A1C",  # red
    "Frame_Shift_Ins":        "#FF7F00",  # orange
    "In_Frame_Del":           "#377EB8",  # blue
    "In_Frame_Ins":           "#984EA3",  # purple
    "Splice_Site":            "#FFD700",  # gold
    "Translation_Start_Site": "#8B008B",  # dark magenta
    "Nonstop_Mutation":       "#00CED1",  # dark turquoise
    "Multi_Hit":              "#A9A9A9",  # dark grey
    "Other":                  "#BDBDBD",  # light grey
}

# Omics layer colors
OMICS_COLORS: dict[str, str] = {
    "RNA-seq":     "#E41A1C",
    "Methylation": "#377EB8",
    "CNV":         "#4DAF4A",
    "Mutations":   "#984EA3",
    "miRNA":       "#FF7F00",
}


# ── Matplotlib global style ─────────────────────────────────────────────────

def set_publication_style(
    font_family: str = "DejaVu Sans",
    font_size: int = 10,
    dpi: int = 300,
) -> None:
    """
    Configure matplotlib for publication-quality output.

    Call once at the start of figure generation.

    Parameters
    ----------
    font_family : str
        Font family. Use 'Arial' or 'Helvetica' for journal requirements.
    font_size : int
        Base font size in points.
    dpi : int
        Figure resolution.
    """
    matplotlib.rcParams.update({
        "font.family":         font_family,
        "font.size":           font_size,
        "axes.titlesize":      font_size + 2,
        "axes.labelsize":      font_size,
        "xtick.labelsize":     font_size - 1,
        "ytick.labelsize":     font_size - 1,
        "legend.fontsize":     font_size - 1,
        "figure.dpi":          dpi,
        "savefig.dpi":         dpi,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.05,
        "axes.linewidth":      0.8,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "grid.alpha":          0.3,
        "grid.linestyle":      "--",
        "lines.linewidth":     1.2,
        "errorbar.capsize":    2.0,
    })


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 300,
    close: bool = True,
) -> list[Path]:
    """
    Save a matplotlib figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
    filename : str
        Base filename without extension (e.g. "Fig1_volcano").
    output_dir : Path
        Output directory.
    formats : tuple
        File formats (e.g. "pdf", "png", "svg").
    dpi : int
        Resolution for raster formats.
    close : bool
        Close figure after saving.

    Returns
    -------
    list[Path]
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        path = output_dir / f"{filename}.{fmt}"
        fig.savefig(path, dpi=dpi, format=fmt, bbox_inches="tight")
        saved.append(path)

    if close:
        plt.close(fig)

    return saved


# ── Color helpers ────────────────────────────────────────────────────────────

def get_subtype_color(subtype: str) -> str:
    """Return the color for a GBM molecular subtype."""
    return SUBTYPE_COLORS.get(subtype, "#999999")


def get_clinical_color(group: str) -> str:
    """Return the color for a clinical group."""
    return CLINICAL_COLORS.get(group, "#999999")


def get_de_color(direction: str) -> str:
    """Return the color for a DE direction."""
    return DE_COLORS.get(direction, "#BDBDBD")


def subtype_palette() -> dict[str, str]:
    """Return subtype color palette (for seaborn, etc.)."""
    return dict(SUBTYPE_COLORS)


# ── Figure size presets ──────────────────────────────────────────────────────

def figure_size(
    style: str = "single_column",
    aspect: float = 4/3,
) -> tuple[float, float]:
    """
    Return figure dimensions for common journal layouts.

    Parameters
    ----------
    style : str
        "single_column" — ~86mm (3.4in) wide
        "double_column" — ~178mm (7.0in) wide
        "full_page"     — ~178mm × 230mm
        "half_page"     — ~178mm × 115mm
        "square"        — ~86mm × 86mm
    aspect : float
        Height / width ratio (used for single/double column).

    Returns
    -------
    (width_inches, height_inches)
    """
    presets = {
        "single_column": (3.4, 3.4 * aspect),
        "double_column": (7.0, 7.0 * aspect),
        "full_page":     (7.0, 9.0),
        "half_page":     (7.0, 4.5),
        "square":        (3.4, 3.4),
    }
    return presets.get(style, (7.0, 5.0))


# ── Annotation helpers for plots ─────────────────────────────────────────────

def add_pvalue_annotation(
    ax: plt.Axes,
    p_value: float,
    x1: float, x2: float,
    y: float,
    h: float = 0.02,
    fontsize: int = 9,
) -> None:
    """
    Add a p-value bracket annotation to a plot.

    Parameters
    ----------
    ax : matplotlib Axes
    p_value : float
        P-value to display.
    x1, x2 : float
        x-positions of the two groups.
    y : float
        y-position for the bracket.
    h : float
        Height of the bracket.
    fontsize : int
    """
    if p_value < 0.0001:
        text = "p < 0.0001"
    elif p_value < 0.001:
        text = f"p = {p_value:.3f}"
    elif p_value < 0.01:
        text = f"p = {p_value:.4f}"
    else:
        text = f"p = {p_value:.4f}"

    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=1.0, color="black")
    ax.text((x1 + x2) * 0.5, y + h, text,
            ha="center", va="bottom", fontsize=fontsize)


def add_n_numbers(
    ax: plt.Axes,
    n_dict: dict[str, int],
    x_positions: dict[str, float] | None = None,
    y: float = -0.12,
    fontsize: int = 8,
) -> None:
    """
    Add n=XX labels below the x-axis of a plot.

    Parameters
    ----------
    ax : matplotlib Axes
    n_dict : dict
        {group_label: count}
    x_positions : dict, optional
        {group_label: x_position}. Defaults to sequential integers.
    y : float
        y-position in axis coordinates.
    fontsize : int
    """
    if x_positions is None:
        x_positions = {k: i for i, k in enumerate(n_dict.keys())}

    for group, count in n_dict.items():
        x = x_positions.get(group, 0)
        ax.text(x, y, f"n={count}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=fontsize, style="italic")


# ── GBM-specific annotation ─────────────────────────────────────────────────

# Known GBM driver genes to highlight in plots
GBM_DRIVER_GENES: set[str] = {
    "IDH1", "IDH2", "TERT", "EGFR", "PTEN", "TP53",
    "RB1", "CDKN2A", "NF1", "PIK3CA", "PIK3R1",
    "MGMT", "PDGFRA", "CDK4", "MDM2", "ATRX",
    "BRAF", "MET", "FGFR3", "CDK6", "CCND1",
    "CCND2", "MDM4", "MYC", "VEGFA", "HIF1A",
    "CHI3L1", "OLIG2", "SOX2", "NES", "PROM1",
    # Prognostic biomarkers from literature
    "YKL-40", "GFAP", "VIM", "FN1", "CD44",
    "MMP2", "MMP9", "SERPINE1", "ANGPT2",
    "S100A4", "LGALS3", "TNC", "LOX",
}

# MGMT promoter CpG probes (Illumina 450k)
MGMT_PROBES: list[str] = [
    "cg12434587", "cg12981137", "cg23998421", "cg07342387",
    "cg14452433", "cg01341292", "cg02635545", "cg25513782",
]


def highlight_drivers(
    genes: list[str] | pd.Index,
) -> np.ndarray:
    """
    Return a boolean mask indicating which genes are known GBM drivers.

    Parameters
    ----------
    genes : list or Index

    Returns
    -------
    np.ndarray
        Boolean array, True for driver genes.
    """
    return np.array([g in GBM_DRIVER_GENES for g in genes])


def is_driver(gene: str) -> bool:
    """Check if a gene is in the GBM driver set."""
    return gene in GBM_DRIVER_GENES
