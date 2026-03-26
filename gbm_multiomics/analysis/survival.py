"""
survival.py — Survival analysis for TCGA-GBM.

Implements:
  - Kaplan-Meier curves with log-rank test (lifelines)
  - Univariate Cox proportional hazards regression
  - Multivariate Cox regression
  - High/low expression split for a gene of interest

Survival endpoints (from PanCanAtlas CDR, prefix "cdr_"):
  OS      Overall Survival          (1 = dead)
  OS.time Time to death or censor (days)
  PFI     Progression-Free Interval (1 = progression/death)
  PFI.time
  DSS     Disease-Specific Survival
  DSS.time

GBM-specific notes
------------------
  Median OS ~15 months in GBM.
  IDH-wildtype GBM: median OS ~13 months.
  IDH-mutant grade 4 astrocytoma: median OS ~36+ months.
  MGMT-methylated tumours: better response to temozolomide → longer OS.

Requires: pip install lifelines

Usage
-----
    from gbm_multiomics.analysis.survival import kaplan_meier, cox_univariate

    km = kaplan_meier(
        df          = merged_df,
        duration_col = "cdr_OS.time",
        event_col    = "cdr_OS",
        group_col    = "IDH_status",
        output_dir  = Path("results/survival"),
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _check_lifelines() -> None:
    try:
        import lifelines  # noqa: F401
    except ImportError:
        raise ImportError(
            "lifelines is required for survival analysis.\n"
            "Install with: pip install 'gbm-multiomics[analysis]'"
        )


def prepare_survival_data(
    clinical: pd.DataFrame,
    molecular: pd.DataFrame | None = None,
    sample_col: str = "case_submitter_id",
    duration_col: str = "cdr_OS.time",
    event_col: str = "cdr_OS",
) -> pd.DataFrame:
    """
    Merge clinical and optional molecular data, filter to valid survival rows.

    Returns a DataFrame with at minimum: sample, duration, event columns.
    Removes rows with missing or non-positive duration.
    """
    df = clinical.copy()

    if molecular is not None:
        merge_key = next(
            (c for c in molecular.columns if "sample" in c.lower() or "barcode" in c.lower()),
            molecular.columns[0],
        )
        df = df.merge(molecular, left_on=sample_col, right_on=merge_key, how="left")

    # Coerce survival columns to numeric
    for col in [duration_col, event_col]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing/invalid survival data
    df = df[df[duration_col].notna() & (df[duration_col] > 0)].copy()
    df = df[df[event_col].isin([0, 1])].copy()

    return df.reset_index(drop=True)


def kaplan_meier(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    title: str = "",
    output_dir: Path | None = None,
    show_ci: bool = True,
    at_risk_counts: bool = True,
) -> dict:
    """
    Kaplan-Meier survival curves with log-rank test between groups.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain duration_col, event_col, group_col.
    duration_col, event_col, group_col : str
    title : str
        Plot title.
    output_dir : Path, optional
        Saves KM plot as PDF and log-rank results as TSV.
    show_ci : bool
        Show 95% confidence intervals.
    at_risk_counts : bool
        Show at-risk table below the KM plot.

    Returns
    -------
    dict with: "logrank_pvalue", "median_os" (per group), "n_per_group"
    """
    _check_lifelines()
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    groups = df[group_col].dropna().unique()
    fitters: dict[str, KaplanMeierFitter] = {}

    for grp in sorted(groups):
        mask = df[group_col] == grp
        kmf = KaplanMeierFitter()
        kmf.fit(
            df.loc[mask, duration_col],
            event_observed = df.loc[mask, event_col],
            label = str(grp),
        )
        fitters[str(grp)] = kmf

    # Log-rank test
    results = multivariate_logrank_test(
        df[duration_col],
        df[group_col],
        df[event_col],
    )
    p_value = results.p_value

    # Median survival per group
    medians = {grp: float(fitters[grp].median_survival_time_) for grp in fitters}
    n_per_group = df.groupby(group_col).size().to_dict()

    print(f"  📊  KM [{group_col}] — log-rank p={p_value:.4e}  "
          f"| medians: " + "  ".join(f"{g}={m:.0f}d" for g, m in medians.items()))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_km_plot(
            fitters, group_col, p_value, title or f"KM: {group_col}",
            duration_col, df, at_risk_counts, show_ci,
            output_dir / f"km_{group_col}.pdf",
        )
        # Save log-rank summary
        summary = pd.DataFrame([{
            "group_col":  group_col,
            "p_value":    p_value,
            "test_statistic": results.test_statistic,
            **{f"median_{g}": m for g, m in medians.items()},
            **{f"n_{g}": n for g, n in n_per_group.items()},
        }])
        summary.to_csv(output_dir / f"logrank_{group_col}.tsv", sep="\t", index=False)

    return {
        "logrank_pvalue":  p_value,
        "median_survival": medians,
        "n_per_group":     n_per_group,
        "fitters":         fitters,
    }


def cox_univariate(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Run univariate Cox proportional hazards regression for each covariate.

    Parameters
    ----------
    covariates : list[str]
        Column names in df to test individually.

    Returns
    -------
    pd.DataFrame
        covariate | HR | HR_lower_95 | HR_upper_95 | p_value | concordance
    """
    _check_lifelines()
    from lifelines import CoxPHFitter

    rows = []
    for cov in covariates:
        sub = df[[duration_col, event_col, cov]].dropna().copy()
        if len(sub) < 10 or sub[event_col].sum() < 5:
            continue

        # Encode booleans / strings
        if sub[cov].dtype == bool or sub[cov].dtype == object:
            sub[cov] = pd.Categorical(sub[cov]).codes.astype(float)

        try:
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(sub, duration_col=duration_col, event_col=event_col)
            summary = cph.summary
            hr    = float(np.exp(summary.loc[cov, "coef"]))
            ci_lo = float(np.exp(summary.loc[cov, "coef lower 95%"]))
            ci_hi = float(np.exp(summary.loc[cov, "coef upper 95%"]))
            pval  = float(summary.loc[cov, "p"])
            rows.append({
                "covariate":   cov,
                "HR":          round(hr, 3),
                "HR_lower_95": round(ci_lo, 3),
                "HR_upper_95": round(ci_hi, 3),
                "p_value":     pval,
                "concordance": round(cph.concordance_index_, 3),
                "n":           len(sub),
                "events":      int(sub[event_col].sum()),
            })
        except Exception:
            continue

    results = pd.DataFrame(rows).sort_values("p_value")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_dir / "cox_univariate.tsv", sep="\t", index=False)
        _save_forest_plot(results, output_dir / "cox_univariate_forest.pdf")

    sig = results[results["p_value"] < 0.05]
    print(f"  ✅  Cox univariate: {len(sig)} significant covariates (p < 0.05)")
    return results


def cox_multivariate(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    covariates: list[str],
    output_dir: Path | None = None,
) -> "CoxPHFitter":
    """
    Multivariate Cox proportional hazards regression.

    Parameters
    ----------
    covariates : list[str]
        Columns to include simultaneously in the model.

    Returns
    -------
    lifelines.CoxPHFitter  fitted model (call .print_summary() for details)
    """
    _check_lifelines()
    from lifelines import CoxPHFitter

    cols = [duration_col, event_col] + covariates
    sub = df[cols].dropna().copy()

    # Encode categoricals
    for cov in covariates:
        if sub[cov].dtype == bool or sub[cov].dtype == object:
            sub[cov] = pd.Categorical(sub[cov]).codes.astype(float)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub, duration_col=duration_col, event_col=event_col)

    print(f"  ✅  Cox multivariate — concordance: {cph.concordance_index_:.3f}")
    cph.print_summary()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cph.summary.to_csv(output_dir / "cox_multivariate.tsv", sep="\t")

    return cph


def expression_survival_split(
    df: pd.DataFrame,
    gene_expr: pd.Series,
    duration_col: str,
    event_col: str,
    split: str = "median",
    gene_name: str = "gene",
    output_dir: Path | None = None,
) -> dict:
    """
    Split samples into high/low expression groups and run KM + log-rank.

    Parameters
    ----------
    df : pd.DataFrame
        Clinical/survival data. Index must match gene_expr.index.
    gene_expr : pd.Series
        Normalised expression values for one gene (e.g. log2(TPM+1)).
    split : str
        "median" | "tertile" | "quartile". Tertile drops middle third.
        Quartile uses Q1 vs Q3+.
    gene_name : str
        Used for plot labels and filenames.

    Returns
    -------
    dict with KM results (same as kaplan_meier return).
    """
    if split == "median":
        threshold = gene_expr.median()
        labels = {True: f"{gene_name}_high", False: f"{gene_name}_low"}
        grp = gene_expr.map(lambda v: labels[v >= threshold])
    elif split == "tertile":
        t33, t67 = gene_expr.quantile([1/3, 2/3])
        grp = gene_expr.apply(
            lambda v: f"{gene_name}_high" if v >= t67
            else (f"{gene_name}_low" if v <= t33 else None)
        )
        grp = grp.dropna()
    elif split == "quartile":
        q1, q3 = gene_expr.quantile([0.25, 0.75])
        grp = gene_expr.apply(
            lambda v: f"{gene_name}_high" if v >= q3
            else (f"{gene_name}_low" if v <= q1 else None)
        )
        grp = grp.dropna()
    else:
        raise ValueError(f"Unknown split strategy: {split!r}. Choose median/tertile/quartile.")

    merged = df.copy()
    merged[f"{gene_name}_group"] = grp
    merged = merged[merged[f"{gene_name}_group"].notna()]

    return kaplan_meier(
        merged,
        duration_col = duration_col,
        event_col    = event_col,
        group_col    = f"{gene_name}_group",
        title        = f"{gene_name} expression — {split} split",
        output_dir   = output_dir,
    )


def _save_km_plot(
    fitters: dict,
    group_col: str,
    p_value: float,
    title: str,
    duration_col: str,
    df: pd.DataFrame,
    at_risk: bool,
    show_ci: bool,
    out_path: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from lifelines.plotting import add_at_risk_counts

        fig, ax = plt.subplots(figsize=(10, 6))
        for label, kmf in fitters.items():
            kmf.plot_survival_function(ax=ax, ci_show=show_ci)

        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Survival probability")
        ax.set_title(f"{title}\nlog-rank p = {p_value:.4e}")
        ax.set_ylim(0, 1.05)

        if at_risk:
            add_at_risk_counts(*fitters.values(), ax=ax)

        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass


def _save_forest_plot(results: pd.DataFrame, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, max(4, len(results) * 0.4)))
        y = range(len(results))
        ax.scatter(results["HR"], y, zorder=3, color="#d6604d", s=50)
        for i, row in results.iterrows():
            ax.hlines(list(y)[list(results.index).index(i)],
                      row["HR_lower_95"], row["HR_upper_95"],
                      color="#d6604d", linewidth=1.5)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_yticks(list(y))
        ax.set_yticklabels(results["covariate"].tolist(), fontsize=9)
        ax.set_xlabel("Hazard Ratio")
        ax.set_title("Univariate Cox — Forest Plot")
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception:
        pass
