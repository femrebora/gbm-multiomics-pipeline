"""
gene_spotlight.py — Per-gene publication figures for GBM thesis.

Generates ready-to-use individual gene figures:
  1. Expression violin/boxplot with jittered points
  2. Kaplan-Meier survival curve (high vs low expression)
  3. Multi-omics 2×2 dashboard panel

All figures follow the publication theme from visualization/theme.py.

References
----------
  Verhaak et al. (2010) Cancer Cell 17:98-110 — GBM subtypes
  Hegi et al. (2005) NEJM 352:997-1003 — MGMT methylation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    CLINICAL_COLORS,
    SUBTYPE_COLORS,
    figure_size,
    get_clinical_color,
    get_subtype_color,
    save_figure,
    set_publication_style,
)


def gene_expression_violin(
    expr: pd.DataFrame,
    gene: str,
    clinical: pd.DataFrame | None = None,
    group_col: str | None = None,
    title: str | None = None,
    output_dir: Path | None = None,
    filename: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> list[Path]:
    """
    Violin plot of gene expression across groups.

    Parameters
    ----------
    expr : pd.DataFrame
        Genes × samples, normalized expression.
    gene : str
        Gene symbol (must be in expr.index).
    clinical : pd.DataFrame, optional
        Sample metadata for grouping.
    group_col : str, optional
        Column in clinical defining groups.
        If None, plots all samples without grouping.
    title : str, optional
        Default: "{gene} Expression".
    output_dir : Path, optional
    filename : str, optional
    figsize : tuple, optional

    Returns
    -------
    list[Path]
        Saved figure paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    if gene not in expr.index:
        raise ValueError(f"Gene '{gene}' not found in expression data.")

    gene_vals = expr.loc[gene].astype(float).dropna()

    # Prepare data
    if clinical is not None and group_col is not None:
        # Find sample ID column
        sample_col = next(
            (c for c in clinical.columns if "sample" in c.lower()), clinical.columns[0]
        )
        common = list(set(gene_vals.index) & set(clinical[sample_col]))
        if not common:
            raise ValueError(
                f"No overlapping samples between expression and clinical['{sample_col}']."
            )

        clinical_idx = clinical.set_index(sample_col)
        groups = clinical_idx.loc[common, group_col].dropna()
        common = list(groups.index)
        gene_vals = gene_vals[common]
        group_labels = groups.values
    else:
        group_labels = np.array(["All"] * len(gene_vals))

    unique_groups = sorted(set(str(g) for g in group_labels))

    # Figure
    if figsize is None:
        figsize = figure_size("single_column", aspect=1.2)

    fig, ax = plt.subplots(figsize=figsize)

    # Colors
    colors = []
    for g in unique_groups:
        c = CLINICAL_COLORS.get(g) or SUBTYPE_COLORS.get(g) or get_clinical_color(g)
        colors.append(c)

    # Violin plot
    parts = ax.violinplot(
        [gene_vals[group_labels == g] for g in unique_groups],
        positions=range(len(unique_groups)),
        showmeans=True,
        showmedians=True,
    )
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)

    # Jittered points
    for i, g in enumerate(unique_groups):
        vals = gene_vals[group_labels == g]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter, vals,
            s=10, alpha=0.4, color=colors[i % len(colors)],
            edgecolors="white", linewidths=0.3,
        )

    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels(unique_groups, fontsize=8)
    ax.set_ylabel("Expression (log2)", fontsize=9)
    ax.set_title(title or f"{gene} Expression", fontsize=11, fontweight="bold")

    # Add n= labels
    for i, g in enumerate(unique_groups):
        n = (group_labels == g).sum()
        ax.text(i, ax.get_ylim()[0] - 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"n={n}", ha="center", fontsize=7, style="italic")

    plt.tight_layout()

    saved = []
    if output_dir is not None:
        fname = filename or f"spotlight_{gene}_expression"
        saved = save_figure(fig, fname, output_dir)
    else:
        plt.close(fig)

    return saved


def gene_survival_km(
    expr: pd.DataFrame,
    gene: str,
    clinical: pd.DataFrame,
    duration_col: str = "cdr_OS.time",
    event_col: str = "cdr_OS",
    split_method: str = "median",
    title: str | None = None,
    output_dir: Path | None = None,
    filename: str | None = None,
) -> list[Path]:
    """
    Kaplan-Meier curve for high vs low expression of a single gene.

    Wraps existing expression_survival_split() and km_plot().

    Parameters
    ----------
    expr, gene, clinical : as above
    duration_col, event_col : str
    split_method : str
        "median", "tertile", or "quartile".
    title : str, optional
    output_dir : Path, optional
    filename : str, optional

    Returns
    -------
    list[Path]
    """
    from gbm_multiomics.analysis.survival import expression_survival_split
    from gbm_multiomics.visualization.survival import km_plot

    if gene not in expr.index:
        raise ValueError(f"Gene '{gene}' not found in expression data.")

    # Find sample ID column
    sample_col = next(
        (c for c in clinical.columns if "sample" in c.lower() or "barcode" in c.lower()),
        clinical.columns[0],
    )

    # Align expression with clinical
    common = list(set(expr.columns) & set(clinical[sample_col]))
    if not common:
        clinical_idx = clinical.set_index(sample_col)
        common = list(set(expr.columns) & set(clinical_idx.index))
        if not common:
            raise ValueError("No overlapping samples between expression and clinical.")
        clinical_to_use = clinical_idx.loc[common]
    else:
        clinical_to_use = clinical.set_index(sample_col).loc[common]

    gene_series = expr.loc[gene, common]

    # Use expression_survival_split for KM fitting
    km_result = expression_survival_split(
        df=clinical_to_use.reset_index(),
        gene_expr=gene_series,
        duration_col=duration_col,
        event_col=event_col,
        split=split_method,
        gene_name=gene,
        output_dir=output_dir,
    )

    return []


def gene_multiomics_dashboard(
    expr: pd.DataFrame,
    gene: str,
    clinical: pd.DataFrame | None = None,
    cnv: pd.DataFrame | None = None,
    methylation: pd.DataFrame | None = None,
    mutations: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    filename: str | None = None,
) -> list[Path]:
    """
    2×2 multi-omics dashboard figure for a single gene.

    Panels:
      Top-left:  Expression violin by group (or boxplot if no groups)
      Top-right: KM survival (high vs low expression)
      Bottom-left: CNV vs expression scatter (if CNV available)
      Bottom-right: Mutation summary or methylation correlation

    Parameters
    ----------
    expr, gene : as above
    clinical : pd.DataFrame, optional
    cnv, methylation, mutations : pd.DataFrame, optional
    output_dir : Path, optional
    filename : str, optional

    Returns
    -------
    list[Path]
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    if gene not in expr.index:
        raise ValueError(f"Gene '{gene}' not found in expression data.")

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # ── Panel 1: Expression violin ───────────────────────────────────────
    ax = axes[0, 0]
    gene_vals = expr.loc[gene].astype(float).dropna()

    if clinical is not None:
        sample_col = next(
            (c for c in clinical.columns if "sample" in c.lower()), clinical.columns[0]
        )
        if "is_tumor" in clinical.columns:
            common = list(set(gene_vals.index) & set(clinical[sample_col]))
            if common:
                clin_idx = clinical.set_index(sample_col).loc[common]
                groups = clin_idx["is_tumor"].map({True: "Tumor", False: "Normal"})
                for label, color in [("Tumor", "#E41A1C"), ("Normal", "#377EB8")]:
                    vals = gene_vals[groups[groups == label].index]
                    if len(vals) > 0:
                        ax.boxplot(
                            [vals], positions=[0 if label == "Tumor" else 1],
                            widths=0.4, patch_artist=True,
                            boxprops={"facecolor": color, "alpha": 0.5},
                        )
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Tumor", "Normal"])
                n_t = (groups == "Tumor").sum()
                n_n = (groups == "Normal").sum()
                ax.set_title(f"{gene} Expression\n(n={n_t} Tumor, n={n_n} Normal)", fontsize=9)
            else:
                ax.hist(gene_vals, bins=20, color="#2166ac", alpha=0.7)
                ax.set_title(f"{gene} Expression Distribution", fontsize=9)
        else:
            ax.hist(gene_vals, bins=20, color="#2166ac", alpha=0.7)
            ax.set_title(f"{gene} Expression (n={len(gene_vals)})", fontsize=9)
    else:
        ax.hist(gene_vals, bins=20, color="#2166ac", alpha=0.7)
        ax.set_title(f"{gene} Expression (n={len(gene_vals)})", fontsize=9)
    ax.set_ylabel("log2 Expression")

    # ── Panel 2: KM survival ────────────────────────────────────────────
    ax = axes[0, 1]
    if clinical is not None and "cdr_OS.time" in clinical.columns:
        try:
            from lifelines import KaplanMeierFitter

            sample_col2 = next(
                (c for c in clinical.columns if "sample" in c.lower()), clinical.columns[0]
            )
            common = list(set(gene_vals.index) & set(clinical[sample_col2]))
            if len(common) >= 10:
                clin_df = clinical.set_index(sample_col2).loc[common]
                median_expr = gene_vals[common].median()
                high_mask = gene_vals[common] >= median_expr

                kmf_high = KaplanMeierFitter()
                kmf_low = KaplanMeierFitter()

                high_idx = [s for s, m in zip(common, high_mask) if m]
                low_idx = [s for s, m in zip(common, high_mask) if not m]

                if high_idx and "cdr_OS.time" in clin_df.columns:
                    kmf_high.fit(
                        clin_df.loc[high_idx, "cdr_OS.time"].astype(float),
                        event_observed=clin_df.loc[high_idx, "cdr_OS"].astype(float),
                        label=f"{gene} High",
                    )
                if low_idx and "cdr_OS.time" in clin_df.columns:
                    kmf_low.fit(
                        clin_df.loc[low_idx, "cdr_OS.time"].astype(float),
                        event_observed=clin_df.loc[low_idx, "cdr_OS"].astype(float),
                        label=f"{gene} Low",
                    )

                kmf_high.plot_survival_function(ax=ax, color="#E41A1C")
                kmf_low.plot_survival_function(ax=ax, color="#377EB8")

                # Log-rank
                from lifelines.statistics import multivariate_logrank_test
                all_groups = np.where(high_mask, f"{gene}_High", f"{gene}_Low")
                lr = multivariate_logrank_test(
                    clin_df.loc[common, "cdr_OS.time"].astype(float),
                    all_groups,
                    clin_df.loc[common, "cdr_OS"].astype(float),
                )
                ax.set_title(
                    f"{gene} — Overall Survival\nlog-rank p = {lr.p_value:.4f}",
                    fontsize=9,
                )

                ax.set_ylabel("Survival Probability")
                ax.set_xlabel("Time (days)")
                ax.set_ylim(0, 1.02)
                ax.legend(fontsize=7)
        except Exception:
            ax.text(0.5, 0.5, "KM not available", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{gene} Survival", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No survival data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(f"{gene} Survival", fontsize=9)

    # ── Panel 3: CNV vs Expression ──────────────────────────────────────
    ax = axes[1, 0]
    if cnv is not None and gene in cnv.index:
        common = list(set(expr.columns) & set(cnv.columns))
        if len(common) >= 10:
            x = cnv.loc[gene, common].astype(float)
            y = expr.loc[gene, common].astype(float)
            valid = x.notna() & y.notna()
            r = x[valid].corr(y[valid])
            ax.scatter(x[valid], y[valid], s=10, alpha=0.5, color="#333333")
            # Trend line
            from numpy.polynomial.polynomial import polyfit
            try:
                b, m = polyfit(x[valid], y[valid], 1)
                xs = np.linspace(x[valid].min(), x[valid].max(), 50)
                ax.plot(xs, b + m * xs, color="#E41A1C", linewidth=1.5)
            except Exception:
                pass
            ax.set_xlabel("Copy Number (log2 ratio)")
            ax.set_ylabel(f"{gene} Expression")
            ax.set_title(f"CNV vs Expression (r = {r:.3f})", fontsize=9)
        else:
            ax.text(0.5, 0.5, "Insufficient samples", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.set_title("CNV vs Expression", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No CNV data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title("CNV vs Expression", fontsize=9)

    # ── Panel 4: Mutation summary ───────────────────────────────────────
    ax = axes[1, 1]
    if mutations is not None and gene in mutations.index:
        mut_row = mutations.loc[gene].dropna()
        n_mut = len(mut_row)
        n_total = len(mutations.columns)
        freq = n_mut / n_total if n_total > 0 else 0

        if n_mut > 0:
            var_counts = mut_row.value_counts().head(6)
            colors = plt.get_cmap("tab10")(range(len(var_counts)))
            ax.barh(range(len(var_counts)), var_counts.values, color=colors, edgecolor="white")
            ax.set_yticks(range(len(var_counts)))
            ax.set_yticklabels(var_counts.index, fontsize=7)
            ax.set_xlabel("Count")
            ax.set_title(f"{gene} Mutations\n({n_mut}/{n_total}, {freq:.1%})", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"No mutations in {n_total} samples",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)
            ax.set_title(f"{gene} Mutations (0%)", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No mutation data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(f"{gene} Mutations", fontsize=9)

    # Final
    fig.suptitle(f"{gene} — Multi-Omics Dashboard", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    saved = []
    if output_dir is not None:
        fname = filename or f"spotlight_{gene}_dashboard"
        saved = save_figure(fig, fname, output_dir)
    else:
        plt.close(fig)

    return saved
