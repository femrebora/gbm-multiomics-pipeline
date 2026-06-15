"""
visualization/forest.py — Forest plots for Cox proportional hazards results.

Generates publication-quality forest plots showing hazard ratios with
95% confidence intervals, sorted by effect size or significance.

References
----------
  Cox (1972) JRSS-B 34:187-220
  Lewis & Clarke (2001) BMJ 322:1479-1481 — forest plot guidelines
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    figure_size,
    save_figure,
    set_publication_style,
)


def forest_plot(
    cox_results: pd.DataFrame,
    hr_col: str = "HR",
    ci_lower_col: str = "HR_lower_95",
    ci_upper_col: str = "HR_upper_95",
    p_col: str = "p_value",
    label_col: str = "covariate",
    sort_by: str = "HR",
    title: str = "Cox Proportional Hazards — Forest Plot",
    xlabel: str = "Hazard Ratio (95% CI)",
    sig_threshold: float = 0.05,
    output_dir: Path | None = None,
    filename: str = "forest_plot",
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Generate a publication-quality forest plot.

    Parameters
    ----------
    cox_results : pd.DataFrame
        One row per covariate with HR, CI columns.
    hr_col, ci_lower_col, ci_upper_col, p_col, label_col : str
        Column names.
    sort_by : str
        "HR", "p_value", or None (preserve input order).
    title, xlabel : str
    sig_threshold : float
        P-value threshold for coloring significant results.
    output_dir : Path, optional
    filename : str
    figsize : tuple, optional

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    df = cox_results.copy()

    # Sort
    if sort_by == "HR":
        df = df.sort_values(hr_col, ascending=False)
    elif sort_by == "p_value":
        df = df.sort_values(p_col, ascending=True)

    # Truncate long labels
    df["plot_label"] = df[label_col].apply(
        lambda x: str(x)[:45] + "…" if len(str(x)) > 45 else str(x)
    )

    n = len(df)
    if figsize is None:
        figsize = (8, max(3, n * 0.4))

    fig, ax = plt.subplots(figsize=figsize)

    y_positions = list(range(n))

    # Plot each covariate
    for i, (_, row) in enumerate(df.iterrows()):
        hr = row[hr_col]
        ci_lo = row[ci_lower_col]
        ci_hi = row[ci_upper_col]
        p = row[p_col]

        is_sig = p < sig_threshold
        color = "#E41A1C" if (is_sig and hr > 1) else "#377EB8" if is_sig else "#999999"
        size = 60 if is_sig else 40

        ax.scatter(hr, i, color=color, s=size, zorder=3, edgecolors="white",
                   linewidths=0.5)
        ax.hlines(i, ci_lo, ci_hi, color=color, linewidth=2.0 if is_sig else 1.0,
                  alpha=0.9)

        # Add whisker caps
        ax.hlines(i, ci_lo, ci_lo, color=color, linewidth=3.0 if is_sig else 1.5)
        ax.hlines(i, ci_hi, ci_hi, color=color, linewidth=3.0 if is_sig else 1.5)

    # Reference line at HR=1
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)

    # Labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(df["plot_label"].tolist(), fontsize=9)

    # Add HR and p-value as right-side labels
    for i, (_, row) in enumerate(df.iterrows()):
        hr_text = f"{row[hr_col]:.2f} ({row[ci_lower_col]:.2f}–{row[ci_upper_col]:.2f})"
        p_text = f"p={row[p_col]:.4f}" if row[p_col] >= 0.0001 else "p<0.0001"
        ax.text(
            1.02, i, f"{hr_text}  {p_text}",
            transform=ax.get_yaxis_transform(),
            fontsize=7, va="center", ha="left",
            color="#333333",
        )

    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11, fontweight="bold")

    # Log scale for HR
    ax.set_xscale("log")
    ax.set_xlim(
        left=df[ci_lower_col].min() * 0.5,
        right=df[ci_upper_col].max() * 2.0,
    )

    # X ticks at sensible HR values
    from matplotlib.ticker import FixedLocator
    hr_ticks = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    hr_labels = ["0.25", "0.5", "0.75", "1.0", "1.5", "2.0", "3.0", "5.0", "10.0"]
    tick_vals = [t for t in hr_ticks
                 if t > df[ci_lower_col].min() * 0.5
                 and t < df[ci_upper_col].max() * 2.0]
    tick_labels = [l for t, l in zip(hr_ticks, hr_labels)
                   if t > df[ci_lower_col].min() * 0.5
                   and t < df[ci_upper_col].max() * 2.0]
    ax.xaxis.set_major_locator(FixedLocator(tick_vals, nbins=6))

    # Invert y-axis so first covariate is at top
    ax.invert_yaxis()

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig


def forest_plot_multivariate(
    cox_model,
    title: str = "Multivariate Cox — Forest Plot",
    output_dir: Path | None = None,
    filename: str = "forest_multivariate",
) -> plt.Figure:
    """
    Forest plot from a fitted lifelines CoxPHFitter model.

    Parameters
    ----------
    cox_model : lifelines.CoxPHFitter
        Fitted model (from cox_multivariate or prognostic module).
    title : str
    output_dir : Path, optional
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    summary = cox_model.summary

    results = []
    for cov in summary.index:
        hr = np.exp(summary.loc[cov, "coef"])
        ci_lo = np.exp(summary.loc[cov, "coef lower 95%"])
        ci_hi = np.exp(summary.loc[cov, "coef upper 95%"])
        p = summary.loc[cov, "p"]
        results.append({
            "covariate": cov,
            "HR": round(hr, 3),
            "HR_lower_95": round(ci_lo, 3),
            "HR_upper_95": round(ci_hi, 3),
            "p_value": p,
        })

    df = pd.DataFrame(results)
    fig = forest_plot(
        df,
        sort_by="HR",
        title=title,
        output_dir=output_dir,
        filename=filename,
    )

    if output_dir is not None:
        df.to_csv(
            Path(output_dir) / f"{filename}_data.tsv",
            sep="\t", index=False,
        )

    return fig
