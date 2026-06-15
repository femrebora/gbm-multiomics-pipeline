"""
visualization/survival.py — Publication-quality Kaplan-Meier curves.

Features:
  - KM curves with 95% CI bands
  - At-risk tables below the plot
  - Log-rank p-value annotation
  - Consistent color palette for GBM clinical groups

References
----------
  Kaplan & Meier (1958) JASA 53:457-481
  Liu et al. (2018) Cell 173:400-416 — PanCanAtlas CDR endpoints
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import (
    CLINICAL_COLORS,
    RISK_COLORS,
    SUBTYPE_COLORS,
    figure_size,
    get_clinical_color,
    save_figure,
    set_publication_style,
)


def km_plot(
    fitters: dict,
    group_col: str,
    p_value: float,
    duration_col: str = "Time",
    title: str = "Kaplan-Meier Survival",
    xlabel: str = "Time (days)",
    ylabel: str = "Survival probability",
    show_ci: bool = True,
    show_at_risk: bool = True,
    max_time: float | None = None,
    x_breaks: float = 365,
    color_map: dict[str, str] | None = None,
    output_dir: Path | None = None,
    filename: str = "km_plot",
    figsize: tuple[float, float] | None = None,
) -> None:
    """
    Generate publication-quality Kaplan-Meier survival curves.

    Parameters
    ----------
    fitters : dict
        {group_label: lifelines.KaplanMeierFitter}
    group_col : str
        Name of the grouping variable (for filename and annotation).
    p_value : float
        Log-rank p-value.
    duration_col : str
        Label for the x-axis.
    title : str
    xlabel, ylabel : str
    show_ci : bool
        Show 95% CI as shaded bands.
    show_at_risk : bool
        Show at-risk table below the plot.
    max_time : float, optional
        Maximum time to show on x-axis.
    x_breaks : float
        Tick interval on x-axis (default 365 = yearly).
    color_map : dict, optional
        {group: color}. Auto-detects from clinical/subtype colors.
    output_dir : Path, optional
    filename : str
    figsize : tuple, optional
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lifelines.plotting import add_at_risk_counts

    set_publication_style()

    n_groups = len(fitters)
    if figsize is None:
        figsize = figure_size("double_column", aspect=0.85)

    if show_at_risk:
        # Split figure: upper 80% for KM, lower 20% for at-risk
        fig = plt.figure(figsize=(figsize[0], figsize[1] * 1.3))
        gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.05)
        ax = fig.add_subplot(gs[0])
        ax_at_risk = fig.add_subplot(gs[1])
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax_at_risk = None

    # Detect colors
    if color_map is None:
        # Try subtype colors, then clinical, then risk
        color_map = {}
        for grp in fitters:
            if grp in SUBTYPE_COLORS:
                color_map[grp] = SUBTYPE_COLORS[grp]
            elif grp in CLINICAL_COLORS:
                color_map[grp] = CLINICAL_COLORS[grp]
            elif grp in RISK_COLORS:
                color_map[grp] = RISK_COLORS[grp]
            else:
                color_map[grp] = get_clinical_color(grp)

    # Plot each group
    for label, kmf in sorted(fitters.items()):
        color = color_map.get(label, "#333333")
        kmf.plot_survival_function(
            ax=ax,
            ci_show=show_ci,
            color=color,
            linewidth=1.8,
            ci_alpha=0.15,
        )

    # Add at-risk table
    if show_at_risk and ax_at_risk is not None:
        add_at_risk_counts(*fitters.values(), ax=ax_at_risk)
        ax_at_risk.set_xlabel("")
        ax_at_risk.set_ylabel("")
        # Hide at-risk axis ticks/labels
        ax_at_risk.tick_params(axis="both", which="both", length=0)
        # Match x limits
        if max_time:
            ax_at_risk.set_xlim(0, max_time)

    # Styling
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)

    if max_time:
        ax.set_xlim(0, max_time)
    if x_breaks:
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_major_locator(MultipleLocator(x_breaks))

    # P-value annotation
    if p_value < 0.0001:
        p_text = f"log-rank p < 0.0001"
    elif p_value < 0.001:
        p_text = f"log-rank p = {p_value:.4f}"
    else:
        p_text = f"log-rank p = {p_value:.4f}"

    ax.set_title(f"{title}\n{p_text}", fontsize=11)

    # Legend
    ax.legend(
        title=group_col,
        fontsize=8,
        title_fontsize=9,
        frameon=True,
        framealpha=0.9,
        loc="lower left" if p_value < 0.05 else "best",
    )

    # Grid
    ax.grid(True, alpha=0.2, linestyle="--")

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    plt.close(fig)


def km_with_at_risk(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    title: str = "GBM Survival",
    output_dir: Path | None = None,
    **kwargs,
) -> dict:
    """
    Complete KM analysis + publication plot in one call.

    Uses lifelines internally, wraps km_plot for figure generation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain duration_col, event_col, group_col.
    duration_col : str
    event_col : str
    group_col : str
    title : str
    output_dir : Path, optional
    **kwargs : passed to km_plot()

    Returns
    -------
    dict
        {logrank_pvalue, median_survival (per group), n_per_group}
    """
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    fitters: dict[str, KaplanMeierFitter] = {}
    groups = df[group_col].dropna().unique()

    for grp in sorted(groups):
        mask = df[group_col] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, duration_col],
            event_observed=df.loc[mask, event_col],
            label=str(grp),
        )
        fitters[str(grp)] = kmf

    # Log-rank
    lr_result = multivariate_logrank_test(
        df[duration_col], df[group_col], df[event_col],
    )
    p_val = lr_result.p_value

    # Median survival
    medians = {g: float(f.median_survival_time_) for g, f in fitters.items()}
    n_groups = df.groupby(group_col).size().to_dict()

    # Plot
    km_plot(
        fitters=fitters,
        group_col=group_col,
        p_value=p_val,
        duration_col=duration_col,
        title=f"{title} — stratified by {group_col}",
        output_dir=output_dir,
        filename=f"km_{group_col}",
        **kwargs,
    )

    # Save summary
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame([{
            "group_col": group_col,
            "logrank_p": p_val,
            **{f"median_{g}_days": round(m, 0) for g, m in medians.items()},
            **{f"n_{g}": n for g, n in n_groups.items()},
        }])
        summary.to_csv(output_dir / f"logrank_{group_col}.tsv", sep="\t", index=False)

    return {
        "logrank_pvalue": p_val,
        "median_survival": medians,
        "n_per_group": n_groups,
        "fitters": fitters,
    }
