"""
pathway_enrichment.py — Pathway enrichment analysis for TCGA-GBM.

Supports:
  - ORA  (Over-Representation Analysis) on gene lists via gseapy
  - GSEA (Gene Set Enrichment Analysis) using pre-ranked gene list
  - GBM-specific gene sets: RTK/PI3K pathway, p53, RB, mismatch repair, etc.

Gene set databases used
-----------------------
  MSigDB Hallmarks (H)              — 50 canonical biological processes
  MSigDB C2 KEGG                    — KEGG pathways
  MSigDB C5 GO Biological Process   — GO terms
  MSigDB C6 Oncogenic Signatures    — cancer-relevant
  Custom GBM gene sets              — RTK signalling, IDH biology

Requires: pip install gseapy

Usage
-----
    from gbm_multiomics.analysis.pathway_enrichment import run_ora, run_gsea

    ora_results = run_ora(
        gene_list  = sig_genes,           # list of HGNC symbols
        gene_sets  = "MSigDB_Hallmark_2020",
        background = all_expressed_genes,
    )

    gsea_results = run_gsea(
        ranked_genes = de_results["log2FoldChange"],  # Series: gene → stat
        gene_sets    = "MSigDB_Hallmark_2020",
    )
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ── GBM-specific gene sets ─────────────────────────────────────────────────────
GBM_GENE_SETS: dict[str, list[str]] = {
    "RTK_PI3K_AKT_pathway": [
        "EGFR", "PDGFRA", "MET", "FGFR1", "ERBB2",
        "PIK3CA", "PIK3R1", "AKT1", "AKT2", "AKT3",
        "PTEN", "MTOR", "TSC1", "TSC2",
    ],
    "p53_RB_cell_cycle": [
        "TP53", "MDM2", "MDM4", "CDKN2A", "RB1",
        "CDK4", "CDK6", "CCND1", "CCND2", "E2F1",
        "CCNE1", "CDK2",
    ],
    "IDH_and_epigenetic": [
        "IDH1", "IDH2", "TET2", "DNMT3A", "DNMT3B",
        "ATRX", "DAXX", "H3F3A", "HIST1H3B",
    ],
    "GBM_invasion": [
        "MMP2", "MMP9", "MMP14", "ADAMTS4",
        "CD44", "VIM", "FN1", "ITGB1", "ITGAV",
        "CHI3L1", "SERPINE1",
    ],
    "GBM_angiogenesis": [
        "VEGFA", "VEGFB", "VEGFC", "KDR", "FLT1",
        "ANGPT1", "ANGPT2", "TEK", "HIF1A", "EPAS1",
    ],
    "GBM_stem_cell_markers": [
        "SOX2", "NES", "PROM1", "CD44", "ITGA6",
        "OLIG2", "MYC", "NANOG", "POU5F1",
    ],
    "NF1_RAS_MAPK": [
        "NF1", "KRAS", "HRAS", "NRAS",
        "RAF1", "BRAF", "MAP2K1", "MAP2K2",
        "MAPK1", "MAPK3", "SPRY2",
    ],
}


def _check_gseapy() -> None:
    try:
        import gseapy  # noqa: F401
    except ImportError:
        raise ImportError(
            "gseapy is required for pathway enrichment.\n"
            "Install with: pip install 'gbm-multiomics[analysis]'"
        )


def run_ora(
    gene_list: list[str],
    gene_sets: str | list[str] | dict = "MSigDB_Hallmark_2020",
    background: list[str] | None = None,
    organism: str = "Human",
    cutoff: float = 0.05,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Over-Representation Analysis on a gene list.

    Parameters
    ----------
    gene_list : list[str]
        HGNC gene symbols to test (e.g. significant DE genes).
    gene_sets : str | list[str] | dict
        gseapy gene set database name(s) OR a dict of custom gene sets.
        Popular options: "MSigDB_Hallmark_2020", "KEGG_2021_Human",
        "GO_Biological_Process_2021", "Reactome_2022"
    background : list[str], optional
        Background gene universe. Defaults to all genes in the gene set.
    organism : str
        "Human" or "Mouse".
    cutoff : float
        Adjusted p-value cutoff for reporting.
    output_dir : Path, optional
        Directory to write results and bar chart.

    Returns
    -------
    pd.DataFrame  Enrichment results sorted by adjusted p-value.
    """
    _check_gseapy()
    import gseapy as gp

    if isinstance(gene_sets, dict):
        # Use GBM_GENE_SETS or user-provided dict
        enr = gp.enrich(
            gene_list  = gene_list,
            gene_sets  = gene_sets,
            background = background,
            outdir     = str(output_dir) if output_dir else None,
            verbose    = verbose,
        )
    else:
        enr = gp.enrichr(
            gene_list  = gene_list,
            gene_sets  = gene_sets,
            organism   = organism,
            background = background,
            outdir     = str(output_dir) if output_dir else None,
            cutoff     = cutoff,
            verbose    = verbose,
        )

    results = enr.results
    if results is None or results.empty:
        print("  ⚠  No enriched pathways found.")
        return pd.DataFrame()

    # Sort by adjusted p-value
    p_col = next(
        (c for c in results.columns if "adj" in c.lower() or "fdr" in c.lower()),
        results.columns[-1],
    )
    results = results.sort_values(p_col)
    sig = results[results[p_col] < cutoff]

    if verbose:
        print(f"  ✅  ORA: {len(sig)} significant pathways at FDR < {cutoff}")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "ora_results.tsv", sep="\t", index=False)
        _plot_top_pathways(sig.head(20), p_col, output_dir / "ora_barplot.pdf",
                           title="ORA — Top Enriched Pathways")

    return results


def run_gsea(
    ranked_genes: pd.Series,
    gene_sets: str | list[str] | dict = "MSigDB_Hallmark_2020",
    min_size: int = 15,
    max_size: int = 500,
    permutation_num: int = 1000,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Pre-ranked GSEA on a gene × statistic Series.

    Parameters
    ----------
    ranked_genes : pd.Series
        Index: HGNC gene symbols.  Values: ranking stat (e.g. log2FC, -log10p * sign(FC)).
        Series should be sorted descending.
    gene_sets : str | list[str] | dict
        Gene set database (same as run_ora).
    min_size / max_size : int
        Gene set size filters.
    permutation_num : int
        Permutations for p-value estimation (1000 recommended; use 100 for testing).
    output_dir : Path, optional
        Directory for results and enrichment plots.

    Returns
    -------
    pd.DataFrame  GSEA results (NES, pval, fdr).
    """
    _check_gseapy()
    import gseapy as gp

    # Sort descending by stat
    ranked = ranked_genes.dropna().sort_values(ascending=False)

    pre_res = gp.prerank(
        rnk          = ranked,
        gene_sets    = gene_sets,
        min_size     = min_size,
        max_size     = max_size,
        permutation_num = permutation_num,
        outdir       = str(output_dir) if output_dir else None,
        verbose      = verbose,
    )

    results = pre_res.res2d
    if results is None or results.empty:
        print("  ⚠  No GSEA results returned.")
        return pd.DataFrame()

    sig = results[results["fdr"] < 0.25]  # GSEA convention: FDR < 0.25
    if verbose:
        print(f"  ✅  GSEA: {len(sig)} pathways at FDR < 0.25 "
              f"(enriched: {(sig['NES'] > 0).sum()}, "
              f"depleted: {(sig['NES'] < 0).sum()})")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "gsea_results.tsv", sep="\t", index=False)

    return results


def run_gbm_custom_ora(
    gene_list: list[str],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Run ORA using the built-in GBM_GENE_SETS (no internet required)."""
    return run_ora(
        gene_list  = gene_list,
        gene_sets  = GBM_GENE_SETS,
        output_dir = output_dir,
        verbose    = True,
    )


def make_ranked_list(
    de_results: pd.DataFrame,
    stat_col: str = "stat",
    lfc_col: str = "log2FoldChange",
    padj_col: str = "padj",
) -> pd.Series:
    """
    Convert DE results to a pre-ranked gene list for GSEA.

    Ranking statistic: Wald stat (preferred) or sign(log2FC) * -log10(padj).
    """
    if stat_col in de_results.columns:
        ranked = de_results[stat_col].dropna()
    else:
        import numpy as np
        sign = de_results[lfc_col].apply(lambda x: 1 if x >= 0 else -1)
        ranked = sign * (-de_results[padj_col].clip(lower=1e-300).apply(
            lambda p: float(f"{-1 * pd.np.log10(p):.4f}") if not pd.isna(p) else 0
        ))

    ranked.index = de_results.index
    return ranked.sort_values(ascending=False)


def _plot_top_pathways(
    results: pd.DataFrame,
    p_col: str,
    out_path: Path,
    title: str = "Top Enriched Pathways",
) -> None:
    """Save a horizontal bar chart of the top enriched pathways."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.35)))
        terms = results["Term"].str[:60] if "Term" in results.columns else results.index.str[:60]
        pvals = -np.log10(results[p_col].clip(lower=1e-300))
        ax.barh(range(len(terms)), pvals[::-1], color="#2166ac")
        ax.set_yticks(range(len(terms)))
        ax.set_yticklabels(list(terms)[::-1], fontsize=8)
        ax.set_xlabel("-log10(adjusted p-value)")
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass  # plotting is optional
