"""
visualization — Publication-quality figure generation for GBM multiomics analysis.

Modules
-------
theme         — Color palettes, font settings, DPI, figure sizing
volcano       — Enhanced volcano plots with gene labeling
survival      — Kaplan-Meier curves with at-risk tables
forest        — Forest plots for Cox hazard ratios
heatmap       — Complex heatmaps with clinical annotations
nomogram      — Prognostic nomogram for survival prediction
oncoprint     — Mutation landscape visualization
correlation   — Cross-omics correlation plots
gene_spotlight — Per-gene thesis figures (expression violin, KM, dashboard)
"""

from gbm_multiomics.visualization.theme import (
    set_publication_style,
    save_figure,
    SUBTYPE_COLORS,
    CLINICAL_COLORS,
    DE_COLORS,
    RISK_COLORS,
    GBM_DRIVER_GENES,
    figure_size,
    get_subtype_color,
    get_clinical_color,
    get_de_color,
)

from gbm_multiomics.visualization.volcano import volcano_plot, multi_volcano

# gene_spotlight requires matplotlib — imported lazily
try:
    from gbm_multiomics.visualization.gene_spotlight import (
        gene_expression_violin,
        gene_survival_km,
        gene_multiomics_dashboard,
    )
    _has_spotlight = True
except ImportError:
    _has_spotlight = False

    def gene_expression_violin(*args, **kwargs):
        raise ImportError("matplotlib required for gene spotlight figures")

    def gene_survival_km(*args, **kwargs):
        raise ImportError("matplotlib required for gene spotlight figures")

    def gene_multiomics_dashboard(*args, **kwargs):
        raise ImportError("matplotlib required for gene spotlight figures")


__all__ = [
    # theme
    "set_publication_style",
    "save_figure",
    "SUBTYPE_COLORS",
    "CLINICAL_COLORS",
    "DE_COLORS",
    "RISK_COLORS",
    "GBM_DRIVER_GENES",
    "figure_size",
    "get_subtype_color",
    "get_clinical_color",
    "get_de_color",
    # volcano
    "volcano_plot",
    "multi_volcano",
    # gene_spotlight
    "gene_expression_violin",
    "gene_survival_km",
    "gene_multiomics_dashboard",
]
