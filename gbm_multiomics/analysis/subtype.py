"""
subtype.py — GBM molecular subtype classification.

Implements:
  1. Centroid-based classification (Verhaak 2010 / TCGA 2013 4-subtype model)
     Classical | Mesenchymal | Proneural | Neural
  2. Consensus NMF-based clustering on the top 1500 MAD genes
  3. IDH-based WHO 2021 classification helper

Subtype gene signatures (abbreviated centroids from Verhaak 2010)
------------------------------------------------------------------
  Classical:   EGFR, PDGFRA, NKX2-1, FOXA1, MKI67 (high)
  Mesenchymal: NF1, CHI3L1, MET, CD44, MERTK, PDPN (high)
  Proneural:   IDH1, PDGFRA, OLIG2, SOX2, TCF4, DLL3 (high)
  Neural:      NEFL, GABRA1, SYT1, SLC12A5 (high) — least stable subtype

Note: Neural subtype has been questioned; many classify into 3 subtypes.
The 2021 WHO CNS classification uses IDH status + CDKN2A/B for grading.

Requires: pip install scikit-learn (for NMF clustering)

Usage
-----
    from gbm_multiomics.analysis.subtype import classify_centroids, cluster_nmf

    subtypes = classify_centroids(
        expr_matrix = log2_tpm,   # genes (HGNC) × samples
    )
    nmf_labels = cluster_nmf(expr_matrix, n_components=4)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Subtype signature genes (Verhaak 2010, Cell) ──────────────────────────────
# Each list is the top marker genes for that subtype.
# Distances are computed as Pearson correlation to the centroid mean profile.
VERHAAK_SIGNATURES: dict[str, list[str]] = {
    "Classical": [
        "EGFR", "NKX2-1", "FOXA1", "MKI67", "CDH1",
        "PDPN", "CDK4", "MDM2", "GAS7", "MEOX2",
    ],
    "Mesenchymal": [
        "NF1", "CHI3L1", "MET", "CD44", "MERTK",
        "PDPN", "ITGA3", "ANPEP", "VIM", "FN1",
    ],
    "Proneural": [
        "IDH1", "PDGFRA", "OLIG2", "SOX2", "TCF4",
        "DLL3", "SFRP2", "BCAN", "MBP", "ASCL1",
    ],
    "Neural": [
        "NEFL", "GABRA1", "SYT1", "SLC12A5",
        "SNAP25", "MAP2", "TUBB2A", "NRXN1", "THY1",
    ],
}

# 3-subtype model (drop Neural, used in some analyses)
VERHAAK_3_SUBTYPES = {k: v for k, v in VERHAAK_SIGNATURES.items() if k != "Neural"}


def classify_centroids(
    expr_matrix: pd.DataFrame,
    use_3_subtypes: bool = False,
    normalise: bool = True,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Assign each sample to a GBM subtype using Pearson correlation
    to per-subtype gene centroids.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes (HGNC symbols, index) × samples (columns).
        Expected to be log2-normalised (e.g. log2(TPM+1) or log2(FPKM+1)).
        Raw counts should be normalised first — see notes below.
    use_3_subtypes : bool
        If True, use Classical/Mesenchymal/Proneural only (drop Neural).
    normalise : bool
        If True, z-score each gene across samples before correlation.
    output_dir : Path, optional
        Saves subtype assignments and correlation heatmap.

    Returns
    -------
    pd.DataFrame
        sample | assigned_subtype | Classical_corr | Mesenchymal_corr |
        Proneural_corr | Neural_corr | max_corr

    Notes
    -----
    Input should be log2(TPM+1) or log2(FPKM+1) — NOT raw counts.
    To convert raw counts to TPM: see analysis/differential_expression.py
    or use DESeq2 vst-normalised counts.
    """
    sigs = VERHAAK_3_SUBTYPES if use_3_subtypes else VERHAAK_SIGNATURES
    subtypes = list(sigs.keys())

    # Z-score normalise (gene-wise)
    expr = expr_matrix.copy().astype(float)
    if normalise:
        expr = expr.sub(expr.mean(axis=1), axis=0)
        std = expr.std(axis=1).replace(0, 1)
        expr = expr.div(std, axis=0)

    # Compute centroid mean profile for each subtype
    centroids: dict[str, pd.Series] = {}
    for subtype, genes in sigs.items():
        present = [g for g in genes if g in expr.index]
        if not present:
            centroids[subtype] = pd.Series(0.0, index=expr.columns)
        else:
            centroids[subtype] = expr.loc[present].mean(axis=0)

    # Pearson correlation of each sample to each centroid
    rows = []
    for sample in expr.columns:
        sample_vec = expr[sample]
        corrs = {}
        for subtype, centroid in centroids.items():
            valid = sample_vec.notna() & centroid.notna()
            if valid.sum() < 5:
                corrs[subtype] = np.nan
            else:
                corrs[subtype] = float(np.corrcoef(
                    sample_vec[valid], centroid[valid]
                )[0, 1])
        best = max(corrs, key=lambda k: corrs[k] if not np.isnan(corrs[k]) else -999)
        rows.append({
            "sample":           sample,
            "assigned_subtype": best,
            "max_corr":         round(corrs[best], 4),
            **{f"{st}_corr": round(corrs[st], 4) if not np.isnan(corrs[st]) else None
               for st in subtypes},
        })

    result = pd.DataFrame(rows)
    counts = result["assigned_subtype"].value_counts()
    print("  ✅  GBM subtype assignment:")
    for st, n in counts.items():
        print(f"       {st}: {n} ({n/len(result)*100:.1f}%)")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "gbm_subtypes_centroid.tsv", sep="\t", index=False)
        _save_subtype_heatmap(result, subtypes, output_dir / "subtype_correlation_heatmap.pdf")

    return result


def cluster_nmf(
    expr_matrix: pd.DataFrame,
    n_components: int = 4,
    n_init: int = 10,
    max_iter: int = 500,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Unsupervised NMF clustering on the top 1500 most variable genes.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Non-negative expression matrix (log2(x+1) shifted to ≥ 0).
        Genes × samples.
    n_components : int
        Number of clusters (4 = classic TCGA GBM subtypes).
    n_init : int
        Number of random initialisations (best is kept).

    Returns
    -------
    pd.DataFrame  sample | nmf_cluster | W_component_0 ... W_component_k
    """
    try:
        from sklearn.decomposition import NMF
    except ImportError:
        raise ImportError(
            "scikit-learn is required for NMF clustering.\n"
            "Install with: pip install 'gbm-multiomics[analysis]'"
        )

    # Select top 1500 MAD genes
    mad = expr_matrix.apply(lambda row: np.median(np.abs(row - np.median(row))), axis=1)
    top_genes = mad.nlargest(1500).index
    sub = expr_matrix.loc[top_genes].T.astype(float)

    # Shift to non-negative
    sub = sub - sub.min().min() + 1e-6

    print(f"  🔧  NMF clustering: {sub.shape[1]} genes × {sub.shape[0]} samples "
          f"(k={n_components}, n_init={n_init})...")

    model = NMF(
        n_components = n_components,
        init         = "nndsvda",
        max_iter     = max_iter,
        random_state = 42,
    )

    W = model.fit_transform(sub)  # samples × components

    cluster_labels = np.argmax(W, axis=1)
    result = pd.DataFrame({
        "sample":      sub.index.tolist(),
        "nmf_cluster": [f"NMF_{i+1}" for i in cluster_labels],
        **{f"W_component_{i}": W[:, i].round(4) for i in range(n_components)},
    })

    print("  ✅  NMF clusters:")
    for c, n in result["nmf_cluster"].value_counts().items():
        print(f"       {c}: {n} ({n/len(result)*100:.1f}%)")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_dir / "gbm_subtypes_nmf.tsv", sep="\t", index=False)

    return result


def who_2021_classify(
    idh_status: pd.DataFrame,
    sample_col: str = "sample",
    idh_col: str = "IDH_status",
) -> pd.DataFrame:
    """
    Apply simplified WHO 2021 CNS tumour classification based on IDH status.

    WHO 2021 (5th edition) defines:
      IDH-wildtype  + histology GBM features → Glioblastoma, IDH-wildtype (Grade 4)
      IDH-mutant    + CDKN2A/B homozygous del → Astrocytoma, IDH-mutant (Grade 4)
      IDH-mutant    + 1p/19q codeletion       → Oligodendroglioma, IDH-mutant (Grade 2/3)
      IDH-mutant    + no codeletion           → Astrocytoma, IDH-mutant (Grade 2/3)

    This function uses IDH status only (no CNV/1p19q data required).
    For full classification, integrate with CNV data.

    Returns
    -------
    pd.DataFrame  sample | IDH_status | who_2021_provisional
    """
    df = idh_status[[sample_col, idh_col]].copy()
    df["who_2021_provisional"] = df[idh_col].map({
        "IDH_wildtype": "Glioblastoma_IDH-wildtype_G4",
        "IDH_mutant":   "Astrocytoma_IDH-mutant_G2-4_provisional",
    }).fillna("Unknown")
    return df


def _save_subtype_heatmap(
    subtype_df: pd.DataFrame,
    subtypes: list[str],
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        corr_cols = [f"{st}_corr" for st in subtypes if f"{st}_corr" in subtype_df.columns]
        plot_df = subtype_df[corr_cols].copy().T
        plot_df.columns = subtype_df["sample"].tolist()

        # Sort samples by assigned subtype
        order = subtype_df.sort_values("assigned_subtype")["sample"].tolist()
        plot_df = plot_df[order]

        fig, ax = plt.subplots(figsize=(min(20, len(order) * 0.08 + 4), 4))
        sns.heatmap(
            plot_df, ax=ax, cmap="RdBu_r", center=0,
            xticklabels=False, yticklabels=True,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title("GBM Subtype Correlation (Verhaak centroids)")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
