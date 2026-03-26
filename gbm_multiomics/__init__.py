"""
gbm-multiomics
==============
Download and analyse TCGA-GBM multiomics data from the NCI GDC portal.

Data types
----------
  rna-seq     : STAR-Counts gene expression matrix
  methylation : DNA methylation beta values (450k/EPIC array)
  mutations   : Somatic mutations (Masked MAF, WXS)
  cnv         : Copy number segments (Genotyping Array)
  mirna       : miRNA expression (miRNA-Seq RPM)

Analysis modules (requires: pip install 'gbm-multiomics[analysis]')
--------------------------------------------------------------------
  differential_expression : pydeseq2 or R-ready output
  pathway_enrichment      : ORA and GSEA via gseapy
  survival                : KM curves, Cox regression (lifelines)
  subtype                 : Verhaak centroid classification + NMF
"""

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import GBM_PROJECT_ID, ALL_DATA_TYPES

__version__ = "0.1.0"
__all__ = ["GBMClient", "GDCError", "GBM_PROJECT_ID", "ALL_DATA_TYPES"]
