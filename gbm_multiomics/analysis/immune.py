"""
immune.py — Immune infiltration analysis for TCGA-GBM.

Computes ESTIMATE scores (StromalScore, ImmuneScore, ESTIMATEScore)
and correlates with prognostic gene expression.

References
----------
  Yoshihara et al. (2013) Nat Commun 4:2612 — ESTIMATE
  Thorsson et al. (2018) Immunity 48:812-830 — TCGA immune landscape
  Wang et al. (2017) Cancer Cell 32:42-56 — GBM immune subtypes
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── ESTIMATE signature genes (Yoshihara et al. 2013) ────────────────────────

ESTIMATE_STROMAL_GENES: list[str] = [
    "ABCA8", "ADAM12", "ANGPTL2", "ASPN", "CCL2", "CCL11", "CCL17",
    "COL1A1", "COL1A2", "COL3A1", "COL4A1", "COL4A2", "COL5A1",
    "COL5A2", "COL6A1", "COL6A2", "COL6A3", "COL8A1", "COL10A1",
    "CXCL12", "DCN", "DPT", "FAP", "FBN1", "FN1", "GREM1",
    "IGFBP7", "LAMB1", "LOX", "LOXL2", "LRRC15", "LUM", "MFAP4",
    "MMP2", "MMP11", "MMP14", "MXRA5", "POSTN", "SERPINH1",
    "SPARC", "SPON1", "SFRP2", "SULF1", "TAGLN", "THBS2",
    "TNC", "VCAN", "VIM",
]

ESTIMATE_IMMUNE_GENES: list[str] = [
    "BTK", "CCL5", "CCR5", "CD2", "CD3D", "CD3E", "CD3G",
    "CD4", "CD8A", "CD8B", "CD19", "CD27", "CD28", "CD37",
    "CD38", "CD40", "CD40LG", "CD48", "CD52", "CD53", "CD69",
    "CD79A", "CD79B", "CD86", "CD96", "CXCR3", "CXCR6",
    "GZMA", "GZMB", "GZMK", "HLA-DMA", "HLA-DMB", "HLA-DOA",
    "HLA-DOB", "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQA2",
    "HLA-DRA", "HLA-DRB1", "IFNG", "IL2RB", "IL2RG", "IL7R",
    "IRF4", "ITK", "LCK", "LCP2", "NKG7", "PRF1", "PTPRC",
    "SLAMF1", "TNFRSF1B", "ZAP70",
]


def estimate_scores(
    expr: pd.DataFrame,
    stromal_genes: list[str] | None = None,
    immune_genes: list[str] | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compute ESTIMATE StromalScore, ImmuneScore, and ESTIMATEScore.

    ESTIMATE = StromalScore + ImmuneScore
    Tumor purity = cos(0.604987 + 0.000146 * ESTIMATEScore)

    Parameters
    ----------
    expr : pd.DataFrame
        Genes × samples, normalized expression (log2 scale).
        Index should be HGNC gene symbols.
    stromal_genes : list[str], optional
        Defaults to ESTIMATE stromal signature.
    immune_genes : list[str], optional
        Defaults to ESTIMATE immune signature.
    output_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        sample | StromalScore | ImmuneScore | ESTIMATEScore | TumorPurity
    """
    if stromal_genes is None:
        stromal_genes = ESTIMATE_STROMAL_GENES
    if immune_genes is None:
        immune_genes = ESTIMATE_IMMUNE_GENES

    # Find present genes
    stromal_present = [g for g in stromal_genes if g in expr.index]
    immune_present = [g for g in immune_genes if g in expr.index]

    if not stromal_present and not immune_present:
        print("  ⚠   No ESTIMATE signature genes found in expression data.")
        return pd.DataFrame()

    print(f"  🧬  ESTIMATE: {len(stromal_present)}/{len(stromal_genes)} "
          f"stromal genes, {len(immune_present)}/{len(immune_genes)} "
          f"immune genes detected.")

    # Compute scores as mean expression of signature genes
    scores = pd.DataFrame(index=expr.columns)

    if stromal_present:
        scores["StromalScore"] = expr.loc[stromal_present].mean(axis=0).values
    else:
        scores["StromalScore"] = 0

    if immune_present:
        scores["ImmuneScore"] = expr.loc[immune_present].mean(axis=0).values
    else:
        scores["ImmuneScore"] = 0

    scores["ESTIMATEScore"] = scores["StromalScore"] + scores["ImmuneScore"]

    # Tumor purity formula from Yoshihara et al. (2013)
    scores["TumorPurity"] = np.cos(0.604987 + 0.000146 * scores["ESTIMATEScore"])

    scores.index.name = "sample"

    print(f"  ✅  ESTIMATE: Stromal={scores['StromalScore'].mean():.1f}, "
          f"Immune={scores['ImmuneScore'].mean():.1f}, "
          f"ESTIMATE={scores['ESTIMATEScore'].mean():.1f}. "
          f"Mean purity={scores['TumorPurity'].mean():.2f}.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scores.to_csv(output_dir / "estimate_scores.tsv", sep="\t")

    return scores


def immune_survival_split(
    immune_scores: pd.DataFrame,
    clinical: pd.DataFrame,
    score_col: str = "ImmuneScore",
    duration_col: str = "cdr_OS.time",
    event_col: str = "cdr_OS",
    output_dir: Path | None = None,
) -> dict:
    """
    Stratify samples by immune score (high/low) and run KM survival.

    Parameters
    ----------
    immune_scores : pd.DataFrame
        From estimate_scores(), index = sample IDs.
    clinical : pd.DataFrame
    score_col : str
    duration_col, event_col : str
    output_dir : Path, optional

    Returns
    -------
    dict
        KM results.
    """
    from gbm_multiomics.analysis.survival import kaplan_meier

    # Merge immune scores with clinical
    merged = immune_scores.join(
        clinical[[duration_col, event_col]].set_index("case_submitter_id"),
        how="inner",
    )

    median = merged[score_col].median()
    merged[f"{score_col}_group"] = np.where(
        merged[score_col] >= median, f"{score_col}_High", f"{score_col}_Low"
    )

    print(f"  🧬  Immune survival split: {score_col} "
          f"(median={median:.2f}).")

    return kaplan_meier(
        merged,
        duration_col=duration_col,
        event_col=event_col,
        group_col=f"{score_col}_group",
        title=f"GBM Survival — {score_col}",
        output_dir=output_dir,
    )


def immune_prognostic_correlation(
    immune_scores: pd.DataFrame,
    prognostic_genes: list[str],
    expr: pd.DataFrame,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Correlate immune scores with prognostic gene expression.

    Parameters
    ----------
    immune_scores : pd.DataFrame
        From estimate_scores().
    prognostic_genes : list[str]
        Gene symbols from prognostic analysis.
    expr : pd.DataFrame
        Genes × samples, normalized expression.
    output_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        gene | StromalScore_r | ImmuneScore_r | ESTIMATEScore_r | p_value
    """
    common = [g for g in prognostic_genes if g in expr.index]
    if not common:
        return pd.DataFrame()

    common_samples = sorted(
        set(immune_scores.index) & set(expr.columns)
    )

    rows = []
    for gene in common:
        gene_expr = expr.loc[gene, common_samples].astype(float)
        for score_col in ["StromalScore", "ImmuneScore", "ESTIMATEScore"]:
            score_vals = immune_scores.loc[common_samples, score_col]
            r = gene_expr.corr(score_vals)
            rows.append({
                "gene": gene,
                "score_type": score_col,
                "pearson_r": round(r, 4),
            })

    result = pd.DataFrame(rows).pivot_table(
        index="gene", columns="score_type", values="pearson_r",
    ).reset_index()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "immune_prognostic_correlation.tsv",
                      sep="\t", index=False)

    # Highlight strong correlations
    strong_stromal = (result.get("StromalScore", pd.Series()).abs() > 0.3).sum()
    strong_immune = (result.get("ImmuneScore", pd.Series()).abs() > 0.3).sum()
    print(f"  ✅  Immune-prognostic correlation: "
          f"{strong_stromal} genes |r| > 0.3 with StromalScore, "
          f"{strong_immune} with ImmuneScore.")

    return result
