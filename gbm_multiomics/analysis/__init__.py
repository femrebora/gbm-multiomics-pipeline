"""
analysis — Downstream analysis modules for TCGA-GBM multiomics data.

Modules
-------
differential_expression  — DESeq2-based DE, multiple comparisons, batch correction
prognostic              — Genome-wide Cox, Lasso-Cox, multivariate model, risk score
survival                — Kaplan-Meier, Cox univariate/multivariate, expression split
pathway_enrichment      — ORA, GSEA, custom GBM gene sets
subtype                 — Verhaak centroid classification, NMF clustering, WHO 2021
multiomics              — Cross-omics correlation, MOFA, SNF clustering
immune                  — ESTIMATE scores, immune-prognostic correlation
network                 — STRING PPI networks, hub genes, co-expression modules
"""

__all__ = [
    "differential_expression",
    "prognostic",
    "pathway_enrichment",
    "survival",
    "subtype",
    "multiomics",
    "immune",
    "network",
]
