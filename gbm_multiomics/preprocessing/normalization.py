"""
normalization.py — Count normalization and batch correction for GBM RNA-seq data.

Methods
-------
  normalize_vst    — DESeq2 variance-stabilizing transformation
  normalize_tpm    — Transcripts Per Million
  normalize_cpm    — Counts Per Million with log2
  batch_correct    — ComBat batch effect correction

References
----------
  Love et al. (2014) Genome Biology 15:550 — DESeq2 / VST
  Johnson et al. (2007) Biostatistics 8:118-127 — ComBat
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_cpm(
    counts: pd.DataFrame,
    log_transform: bool = True,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Normalize raw counts to CPM (Counts Per Million), optionally log2.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, integer raw counts.
    log_transform : bool
        If True, apply log2(CPM + pseudocount).
    pseudocount : float
        Added before log to avoid log(0).

    Returns
    -------
    pd.DataFrame
        CPM-normalized (and optionally log2) expression matrix.
    """
    lib_sizes = counts.sum(axis=0)
    cpm = counts.div(lib_sizes, axis=1) * 1e6

    if log_transform:
        cpm = np.log2(cpm + pseudocount)

    return cpm.astype(np.float32)


def normalize_tpm(
    counts: pd.DataFrame,
    gene_lengths: pd.Series | None = None,
    log_transform: bool = True,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """
    Normalize raw counts to TPM (Transcripts Per Million).

    TPM = (counts / gene_length) / Σ(counts / gene_length) × 1e6

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, integer raw counts.
    gene_lengths : pd.Series, optional
        Gene lengths in bp (index: gene IDs). If None, uses
        constant length (effectively RPKM-like normalization).
    log_transform : bool
        If True, apply log2(TPM + pseudocount).
    pseudocount : float

    Returns
    -------
    pd.DataFrame
        TPM-normalized expression matrix.
    """
    if gene_lengths is None:
        # Without gene lengths, TPM = CPM
        return normalize_cpm(counts, log_transform=log_transform,
                             pseudocount=pseudocount)

    # Align gene lengths
    common = counts.index.intersection(gene_lengths.index)
    if len(common) < len(counts):
        print(f"  ⚠   Gene lengths available for {len(common)}/{len(counts)} genes. "
              f"Using CPM for remaining.")
        counts = counts.loc[common]
        gene_lengths = gene_lengths.loc[common]

    # RPK = reads per kilobase
    rpk = counts.div(gene_lengths.values / 1000, axis=0)
    # TPM = RPK / sum(RPK) * 1e6
    tpm = rpk.div(rpk.sum(axis=0), axis=1) * 1e6

    if log_transform:
        tpm = np.log2(tpm + pseudocount)

    return tpm.astype(np.float32)


def normalize_vst(
    counts: pd.DataFrame,
    n_cpus: int = 4,
) -> pd.DataFrame:
    """
    DESeq2 Variance Stabilizing Transformation.

    Uses pydeseq2's VST. Falls back to simple log2(CPM+1) if pydeseq2
    is not available.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, integer raw counts.
    n_cpus : int

    Returns
    -------
    pd.DataFrame
        VST-normalized expression matrix (float32).
    """
    try:
        from pydeseq2.dds import DeseqDataSet

        counts_T = counts.T.astype(int)
        # Minimal metadata for VST only
        dummy_meta = pd.DataFrame(
            {"dummy": ["A"] * counts_T.shape[0]},
            index=counts_T.index,
        )

        dds = DeseqDataSet(
            counts=counts_T,
            metadata=dummy_meta,
            design_factors="dummy",
            n_cpus=n_cpus,
        )
        dds.vst()
        vst = dds.vst_df.T  # back to genes × samples

        print(f"  ✅  VST normalization: {vst.shape[0]} genes × {vst.shape[1]} samples.")
        return vst.astype(np.float32)

    except ImportError:
        print("  ℹ   pydeseq2 not available. Using log2(CPM+1) instead.")
        return normalize_cpm(counts, log_transform=True, pseudocount=1.0)


def batch_correct(
    expr_matrix: pd.DataFrame,
    batch: pd.Series,
    covariates: pd.DataFrame | None = None,
    parametric: bool = True,
) -> pd.DataFrame:
    """
    Apply ComBat batch effect correction.

    Uses pycombat (Python) if available, otherwise attempts rpy2 → sva::ComBat.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes × samples, normalized expression (log2 scale).
    batch : pd.Series
        Batch labels, index = sample IDs matching expr_matrix columns.
    covariates : pd.DataFrame, optional
        Biological covariates to preserve (e.g. condition, age).
        Columns = covariates, index = sample IDs.
    parametric : bool
        If True, use parametric ComBat. False = non-parametric.

    Returns
    -------
    pd.DataFrame
        Batch-corrected expression matrix (same shape as input).
    """
    # Align batch with expression columns
    common = expr_matrix.columns.intersection(batch.index)
    if len(common) < len(expr_matrix.columns):
        print(f"  ⚠   Batch labels available for {len(common)}/{len(expr_matrix.columns)} samples.")
    batch = batch.loc[common]
    expr = expr_matrix[common].copy()

    if batch.nunique() < 2:
        print("  ℹ   Only one batch detected. No correction needed.")
        return expr_matrix.copy()

    # Try Python implementation first
    try:
        from combat.pycombat import pycombat
        corrected = pycombat(expr, batch)
        print(f"  ✅  ComBat correction applied "
              f"({batch.nunique()} batches, {expr.shape[0]} genes).")
        return pd.DataFrame(corrected, index=expr.index, columns=expr.columns)

    except ImportError:
        pass

    # Fallback: rpy2 → sva::ComBat
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()
        sva = importr("sva")

        r_expr = pandas2ri.py2rpy(expr)
        r_batch = ro.StrVector(batch.astype(str).tolist())

        if covariates is not None:
            cov_aligned = covariates.loc[common]
            r_cov = pandas2ri.py2rpy(cov_aligned)
            r_corrected = sva.ComBat(r_expr, batch=r_batch, mod=r_cov,
                                      par_prior=parametric)
        else:
            r_corrected = sva.ComBat(r_expr, batch=r_batch,
                                      par_prior=parametric)

        corrected = pandas2ri.rpy2py(r_corrected)
        corrected.index = expr.index
        corrected.columns = expr.columns

        print(f"  ✅  ComBat correction (R/sva) applied "
              f"({batch.nunique()} batches).")
        return corrected.astype(np.float32)

    except ImportError:
        print("  ⚠   Neither pycombat nor rpy2/sva available. "
              "Skipping batch correction.")
        return expr_matrix.copy()
