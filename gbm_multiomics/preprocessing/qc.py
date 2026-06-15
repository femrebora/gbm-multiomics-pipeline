"""
qc.py — Quality control: PCA, sample correlation, outlier detection.

Provides QC visualizations and outlier detection for RNA-seq count data
before downstream analysis.

References
----------
  Conesa et al. (2016) Genome Biology 17:13 — RNA-seq QC best practices
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def pca_plot(
    expr_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    color_col: str | None = None,
    label_col: str | None = None,
    n_components: int = 2,
    output_dir: Path | None = None,
    title: str = "PCA — GBM RNA-seq",
) -> pd.DataFrame:
    """
    Compute PCA and optionally generate a scatter plot.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes × samples, normalized expression (log2 scale).
    metadata : pd.DataFrame, optional
        Sample metadata. Index must match expr_matrix columns.
    color_col : str, optional
        Column in metadata to color points by.
    label_col : str, optional
        Column in metadata to label points.
    n_components : int
        Number of principal components to compute.
    output_dir : Path, optional
        If provided, saves PCA plot as PDF.
    title : str

    Returns
    -------
    pd.DataFrame
        Samples × PCs, with variance_explained attribute.
    """
    from sklearn.decomposition import PCA

    # Center genes
    expr_T = expr_matrix.T  # samples × genes
    expr_centered = expr_T - expr_T.mean(axis=0)

    pca = PCA(n_components=min(n_components, min(expr_T.shape)))
    scores = pca.fit_transform(expr_centered)

    columns = [f"PC{i + 1}" for i in range(scores.shape[1])]
    pc_df = pd.DataFrame(scores, index=expr_T.index, columns=columns)

    # Add metadata columns if provided
    if color_col is not None and metadata is not None:
        pc_df[color_col] = metadata.loc[expr_T.index, color_col].values
    if label_col is not None and metadata is not None:
        pc_df[label_col] = metadata.loc[expr_T.index, label_col].values

    pc_df.attrs["variance_explained"] = pca.explained_variance_ratio_
    pc_df.attrs["n_components"] = n_components

    # Print summary
    var_pct = pca.explained_variance_ratio_ * 100
    print(f"  📊  PCA: PC1={var_pct[0]:.1f}%, PC2={var_pct[1]:.1f}% variance "
          f"(total top {n_components}: {var_pct[:n_components].sum():.1f}%)")

    # Plot if output_dir
    if output_dir is not None and scores.shape[1] >= 2:
        _save_pca_plot(pc_df, color_col, label_col, title, output_dir,
                       var_pct[0], var_pct[1])

    return pc_df


def sample_correlation_heatmap(
    expr_matrix: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    annotation_col: str | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Compute pairwise sample Pearson correlation matrix.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes × samples, normalized expression.
    metadata : pd.DataFrame, optional
    annotation_col : str, optional
        Column to annotate heatmap with.
    output_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        Sample × sample correlation matrix.
    """
    corr = expr_matrix.corr(method="pearson")

    median_corr = np.median(corr.values[np.triu_indices_from(corr, k=1)])
    min_corr = corr.values[np.triu_indices_from(corr, k=1)].min()
    print(f"  📊  Sample correlation: median={median_corr:.3f}, "
          f"min={min_corr:.3f}")

    if min_corr < 0.7:
        n_low = (corr.values[np.triu_indices_from(corr, k=1)] < 0.7).sum()
        print(f"  ⚠   {n_low} sample pairs with r < 0.7 — "
              f"check for batch effects or mislabeling.")

    if output_dir is not None:
        _save_correlation_heatmap(corr, metadata, annotation_col, output_dir)

    return corr


def detect_outliers(
    expr_matrix: pd.DataFrame,
    n_components: int = 5,
    iqr_multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Detect outlier samples using PCA-based IQR on PC scores.

    Parameters
    ----------
    expr_matrix : pd.DataFrame
        Genes × samples, normalized expression.
    n_components : int
        Number of PCs to check.
    iqr_multiplier : float
        IQR multiplier for outlier threshold (default 3.0 = Tukey's fence).

    Returns
    -------
    pd.DataFrame
        sample | is_outlier | outlier_pcs | max_deviation
    """
    from sklearn.decomposition import PCA

    expr_T = expr_matrix.T
    expr_centered = expr_T - expr_T.mean(axis=0)

    pca = PCA(n_components=min(n_components, min(expr_T.shape)))
    scores = pca.fit_transform(expr_centered)

    outliers = pd.DataFrame(index=expr_T.index)
    outliers["is_outlier"] = False
    outlier_pcs_list: list[list[str]] = [[] for _ in range(len(outliers))]
    max_deviations = np.zeros(len(outliers))

    for pc_idx in range(scores.shape[1]):
        pc_scores = scores[:, pc_idx]
        q1, q3 = np.percentile(pc_scores, [25, 75])
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        pc_outliers = (pc_scores < lower) | (pc_scores > upper)
        for i, is_out in enumerate(pc_outliers):
            if is_out:
                outliers.iloc[i, 0] = True
                outlier_pcs_list[i].append(f"PC{pc_idx + 1}")
                dev = max(abs(pc_scores[i] - q3) / iqr if pc_scores[i] > q3
                          else abs(pc_scores[i] - q1) / iqr, max_deviations[i])
                max_deviations[i] = dev

    outliers["outlier_pcs"] = [", ".join(pcs) if pcs else "" for pcs in outlier_pcs_list]
    outliers["max_deviation"] = max_deviations.round(2)

    n_out = outliers["is_outlier"].sum()
    if n_out > 0:
        print(f"  ⚠   {n_out} outlier sample(s) detected "
              f"(IQR × {iqr_multiplier}):")
        for sample in outliers[outliers["is_outlier"]].index:
            info = outliers.loc[sample]
            print(f"       {sample}: {info['outlier_pcs']} "
                  f"(deviation={info['max_deviation']})")
    else:
        print(f"  ✅  No outliers detected (IQR × {iqr_multiplier}).")

    return outliers


def library_size_distribution(
    counts: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """
    Compute library size summary statistics.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, raw integer counts.
    output_dir : Path, optional

    Returns
    -------
    dict
        {median, mean, min, max, cv} of library sizes in millions.
    """
    lib_sizes = counts.sum(axis=0) / 1e6  # millions of reads

    stats = {
        "median": float(lib_sizes.median()),
        "mean": float(lib_sizes.mean()),
        "min": float(lib_sizes.min()),
        "max": float(lib_sizes.max()),
        "cv": float(lib_sizes.std() / lib_sizes.mean()),
        "n_samples": len(lib_sizes),
    }

    print(f"  📚  Library sizes: median={stats['median']:.1f}M, "
          f"mean={stats['mean']:.1f}M, CV={stats['cv']:.3f}")

    if stats["cv"] > 0.5:
        print(f"  ⚠   High library size variability (CV={stats['cv']:.2f}). "
              f"Consider normalization.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.Series(stats).to_csv(output_dir / "library_size_summary.tsv", sep="\t")

    return stats


def qc_report(
    counts: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
    color_col: str | None = None,
    output_dir: Path | None = None,
) -> dict:
    """
    Run all QC checks and generate a summary report.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, raw integer counts.
    metadata : pd.DataFrame, optional
    color_col : str, optional
    output_dir : Path, optional

    Returns
    -------
    dict
        QC summary with keys: library_stats, n_outliers, pca_variance, sample_corr
    """
    print(f"\n{'='*60}")
    print("  QC Report — GBM RNA-seq")
    print(f"{'='*60}\n")

    # 1. Library sizes
    lib_stats = library_size_distribution(counts)

    # 2. Gene detection
    n_detected = (counts > 0).sum(axis=0)
    gene_detection_rate = n_detected / len(counts)
    print(f"  🧬  Gene detection: {gene_detection_rate.median():.1%} "
          f"(median, range {gene_detection_rate.min():.1%}–{gene_detection_rate.max():.1%})")

    # 3. Normalize for PCA
    from gbm_multiomics.preprocessing.normalization import normalize_cpm
    log2_cpm = normalize_cpm(counts, log_transform=True)

    # 4. PCA
    pca_result = pca_plot(log2_cpm, metadata=metadata, color_col=color_col)

    # 5. Outlier detection
    outliers = detect_outliers(log2_cpm)
    n_outliers = outliers["is_outlier"].sum()

    # 6. Sample correlation
    corr = sample_correlation_heatmap(log2_cpm, metadata=metadata,
                                       annotation_col=color_col)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save all figures
        pca_plot(log2_cpm, metadata=metadata, color_col=color_col,
                 output_dir=output_dir)
        sample_correlation_heatmap(log2_cpm, metadata=metadata,
                                    annotation_col=color_col,
                                    output_dir=output_dir)
        library_size_distribution(counts, output_dir=output_dir)

        # Save outlier report
        outliers.to_csv(output_dir / "outlier_report.tsv", sep="\t")

        # QC summary
        summary = {
            **lib_stats,
            "median_gene_detection": float(gene_detection_rate.median()),
            "n_outliers": int(n_outliers),
            "pca_var_pc1": float(pca_result.attrs["variance_explained"][0]),
            "pca_var_pc2": float(pca_result.attrs["variance_explained"][1]),
            "median_sample_corr": float(np.median(
                corr.values[np.triu_indices_from(corr, k=1)])),
        }
        pd.Series(summary).to_csv(output_dir / "qc_summary.tsv", sep="\t")

    print(f"\n{'='*60}")
    print(f"  QC Summary: {lib_stats['n_samples']} samples, "
          f"{n_outliers} outliers, "
          f"{len(counts)} genes")
    print(f"{'='*60}\n")

    return {
        "library_stats": lib_stats,
        "n_outliers": n_outliers,
        "pca_variance": pca_result.attrs.get("variance_explained", []),
        "sample_corr": corr,
    }


# ── Internal plotting helpers ────────────────────────────────────────────────

def _save_pca_plot(
    pc_df: pd.DataFrame,
    color_col: str | None,
    label_col: str | None,
    title: str,
    output_dir: Path,
    var_pc1: float,
    var_pc2: float,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))

        if color_col and color_col in pc_df.columns:
            groups = pc_df[color_col].unique()
            for grp in groups:
                mask = pc_df[color_col] == grp
                ax.scatter(
                    pc_df.loc[mask, "PC1"], pc_df.loc[mask, "PC2"],
                    label=str(grp), s=60, alpha=0.8, edgecolors="k",
                    linewidths=0.5,
                )
            ax.legend(title=color_col, fontsize=9, title_fontsize=10)
        else:
            ax.scatter(pc_df["PC1"], pc_df["PC2"], s=60, alpha=0.8,
                       edgecolors="k", linewidths=0.5)

        ax.set_xlabel(f"PC1 ({var_pc1:.1f}% variance)")
        ax.set_ylabel(f"PC2 ({var_pc2:.1f}% variance)")
        ax.set_title(title)
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)

        plt.tight_layout()
        fig.savefig(output_dir / "pca_plot.pdf", dpi=300)
        plt.close(fig)
    except Exception:
        pass


def _save_correlation_heatmap(
    corr: pd.DataFrame,
    metadata: pd.DataFrame | None,
    annotation_col: str | None,
    output_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        figsize = min(20, max(8, len(corr) * 0.3))
        fig, ax = plt.subplots(figsize=(figsize, figsize))
        sns.heatmap(
            corr, ax=ax, cmap="RdBu_r", center=0.5,
            xticklabels=False, yticklabels=False,
            cbar_kws={"label": "Pearson r"},
        )
        ax.set_title("Sample–Sample Pearson Correlation")
        plt.tight_layout()
        fig.savefig(output_dir / "sample_correlation_heatmap.pdf", dpi=150)
        plt.close(fig)
    except Exception:
        pass
