"""
differential_expression.py — Differential expression analysis for TCGA-GBM.

Supports two modes:
  1. pydeseq2  — Python DESeq2 implementation (pip install pydeseq2)
  2. r-ready   — Write DESeq2-ready input files for use in R

Common comparisons for GBM
--------------------------
  Tumor vs Normal (matched or unmatched)
  IDH-mutant vs IDH-wildtype (requires mutations data)
  GBM subtype comparisons (Classical vs Mesenchymal, etc.)
  MGMT methylated vs unmethylated (requires methylation data)
  High vs low expression tertile of a gene of interest

Usage
-----
    from gbm_multiomics.analysis.differential_expression import run_deseq2_py

    results = run_deseq2_py(
        counts   = count_matrix,     # genes × samples int DataFrame
        metadata = sample_meta,      # rows=samples, must have 'condition' col
        condition_col = "condition",
        reference     = "Normal",
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _check_pydeseq2() -> None:
    try:
        import pydeseq2  # noqa: F401
    except ImportError:
        raise ImportError(
            "pydeseq2 is required for DE analysis.\n"
            "Install with: pip install 'gbm-multiomics[analysis]'"
        )


def prepare_deseq2_inputs(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_col: str,
    sample_col: str = "sample_submitter_id",
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align counts and metadata for DESeq2 input.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples raw count matrix (ENSG index, sample barcodes as cols).
    metadata : pd.DataFrame
        Must contain sample_col and condition_col.
    condition_col : str
        Column in metadata that defines groups (e.g. "is_tumor", "IDH_status").
    sample_col : str
        Column in metadata with sample barcodes matching counts columns.
    output_dir : Path, optional
        If given, writes counts_for_deseq2.tsv and coldata_for_deseq2.tsv.

    Returns
    -------
    (counts_aligned, coldata)
    """
    # Keep only samples present in both counts and metadata
    meta = metadata.set_index(sample_col)
    common = [s for s in counts.columns if s in meta.index]

    if not common:
        raise ValueError(
            f"No overlapping samples between counts columns and metadata['{sample_col}']."
        )

    counts_aligned = counts[common].copy()
    coldata = meta.loc[common, [condition_col]].copy()
    coldata.index.name = "sample"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        counts_aligned.to_csv(output_dir / "counts_for_deseq2.tsv", sep="\t")
        coldata.to_csv(output_dir / "coldata_for_deseq2.tsv", sep="\t")
        print(f"  ✅  DESeq2 input files written to {output_dir}")

    return counts_aligned, coldata


def run_deseq2_py(
    counts: pd.DataFrame,
    metadata: pd.DataFrame,
    condition_col: str,
    reference: str,
    sample_col: str = "sample_submitter_id",
    n_cpus: int = 1,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run differential expression analysis using pydeseq2.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples raw count matrix.
    metadata : pd.DataFrame
        Sample metadata with condition_col.
    condition_col : str
        Column defining groups (e.g. "is_tumor").
    reference : str
        Reference level for comparison (e.g. "False" for is_tumor comparison).
    sample_col : str
        Column in metadata with sample IDs matching count matrix columns.
    n_cpus : int
        Number of CPUs for pydeseq2.
    output_dir : Path, optional
        If given, writes DE results TSV.

    Returns
    -------
    pd.DataFrame
        One row per gene with: baseMean, log2FoldChange, lfcSE, stat, pvalue, padj.
    """
    _check_pydeseq2()
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    counts_aligned, coldata = prepare_deseq2_inputs(
        counts, metadata, condition_col, sample_col
    )

    # pydeseq2 expects samples as rows, genes as columns
    counts_T = counts_aligned.T.astype(int)
    coldata_str = coldata[[condition_col]].astype(str)

    print(f"  🧬  Running pydeseq2 — "
          f"{counts_T.shape[1]} genes × {counts_T.shape[0]} samples...")

    dds = DeseqDataSet(
        counts       = counts_T,
        metadata     = coldata_str,
        design_factors = condition_col,
        ref_level     = [condition_col, str(reference)],
        n_cpus        = n_cpus,
        refit_cooks   = True,
    )
    dds.deseq2()

    stat_res = DeseqStats(dds, n_cpus=n_cpus)
    stat_res.summary()

    results = stat_res.results_df.copy()
    results.index.name = "gene_id"
    results = results.sort_values("padj")

    n_sig = (results["padj"] < 0.05).sum()
    print(f"  ✅  DE complete: {n_sig:,} genes at FDR < 0.05 "
          f"(|log2FC| > 1: {((results['padj'] < 0.05) & (results['log2FoldChange'].abs() > 1)).sum():,})")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"de_results_{condition_col}.tsv"
        results.to_csv(out_path, sep="\t")
        print(f"  📄  Results written to {out_path.name}")

    return results


def write_r_deseq2_script(
    counts_path: Path,
    coldata_path: Path,
    condition_col: str,
    reference: str,
    output_dir: Path,
) -> Path:
    """
    Write a ready-to-run R script for DESeq2 analysis.

    Use when you prefer running DESeq2 in R rather than pydeseq2.
    """
    script = f"""\
# DESeq2 analysis script — generated by gbm-multiomics
# Run: Rscript deseq2_run.R

library(DESeq2)
library(ggplot2)
library(EnhancedVolcano)

# ── Load data ─────────────────────────────────────────────────────────────────
counts  <- read.table("{counts_path}", sep="\\t", header=TRUE, row.names=1)
coldata <- read.table("{coldata_path}", sep="\\t", header=TRUE, row.names=1)

# Ensure sample order matches
counts <- counts[, rownames(coldata)]

# ── DESeq2 ────────────────────────────────────────────────────────────────────
dds <- DESeqDataSetFromMatrix(
  countData = round(counts),
  colData   = coldata,
  design    = ~ {condition_col}
)
dds${condition_col} <- relevel(dds${condition_col}, ref = "{reference}")
dds <- DESeq(dds, parallel = TRUE)

# ── Results ───────────────────────────────────────────────────────────────────
res <- results(dds, alpha = 0.05)
res_df <- as.data.frame(res)
res_df <- res_df[order(res_df$padj), ]
write.table(res_df, file = "{output_dir}/de_results_{condition_col}.tsv",
            sep = "\\t", quote = FALSE)

# ── MA plot ───────────────────────────────────────────────────────────────────
pdf("{output_dir}/ma_plot.pdf")
plotMA(res, ylim = c(-5, 5))
dev.off()

# ── Volcano plot ─────────────────────────────────────────────────────────────
pdf("{output_dir}/volcano_plot.pdf", width=10, height=8)
EnhancedVolcano(res_df,
  lab    = rownames(res_df),
  x      = "log2FoldChange",
  y      = "padj",
  title  = "GBM Differential Expression ({condition_col})",
  pCutoff = 0.05,
  FCcutoff = 1.0
)
dev.off()

message("Done. Significant genes: ", sum(res_df$padj < 0.05, na.rm=TRUE))
"""
    out_path = output_dir / "deseq2_run.R"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")
    print(f"  📄  R script written to {out_path.name}")
    return out_path


def filter_significant(
    de_results: pd.DataFrame,
    padj_threshold: float = 0.05,
    lfc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Filter DE results to significant up/downregulated genes."""
    sig = de_results[
        (de_results["padj"] < padj_threshold) &
        (de_results["log2FoldChange"].abs() > lfc_threshold)
    ].copy()
    sig["direction"] = np.where(sig["log2FoldChange"] > 0, "UP", "DOWN")
    return sig.sort_values("log2FoldChange", ascending=False)
