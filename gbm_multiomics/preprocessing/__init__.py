"""
__init__.py — Preprocessing modules for TCGA-GBM multiomics data.

Modules
-------
annotation    — ENSG → HGNC gene symbol mapping, gene biotype filtering
normalization — VST, TPM, CPM normalization with log transformation
qc            — PCA, sample correlation, outlier detection
clinical      — Merge clinical, CDR, molecular annotations into unified metadata
"""

from gbm_multiomics.preprocessing.annotation import (
    annotate_genes,
    filter_protein_coding,
    filter_low_expression,
    map_ensg_to_symbol,
)
from gbm_multiomics.preprocessing.normalization import (
    normalize_cpm,
    normalize_tpm,
    normalize_vst,
    batch_correct,
)
from gbm_multiomics.preprocessing.qc import (
    pca_plot,
    sample_correlation_heatmap,
    detect_outliers,
    library_size_distribution,
    qc_report,
)
from gbm_multiomics.preprocessing.clinical import (
    build_unified_metadata,
    load_clinical_data,
    merge_molecular_features,
)

__all__ = [
    # annotation
    "annotate_genes",
    "filter_protein_coding",
    "filter_low_expression",
    "map_ensg_to_symbol",
    # normalization
    "normalize_cpm",
    "normalize_tpm",
    "normalize_vst",
    "batch_correct",
    # qc
    "pca_plot",
    "sample_correlation_heatmap",
    "detect_outliers",
    "library_size_distribution",
    "qc_report",
    # clinical
    "build_unified_metadata",
    "load_clinical_data",
    "merge_molecular_features",
]
