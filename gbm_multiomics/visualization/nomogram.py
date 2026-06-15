"""
visualization/nomogram.py — Prognostic nomogram for individual survival prediction.

Generates a Cox regression-based nomogram that integrates gene expression
risk score with clinical variables (age, IDH status, MGMT status).

References
----------
  Iasonos et al. (2008) J Clin Oncol 26:1364-1370 — nomogram construction
  Harrell (2015) Regression Modeling Strategies — rms package
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.visualization.theme import save_figure, set_publication_style


def nomogram(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    features: list[str],
    feature_labels: dict[str, str] | None = None,
    prediction_times: tuple[int, ...] = (365, 730, 1095),
    title: str = "GBM Prognostic Nomogram",
    output_dir: Path | None = None,
    filename: str = "nomogram",
    figsize: tuple[float, float] = (12, 8),
) -> plt.Figure:
    """
    Build and plot a Cox regression nomogram for survival prediction.

    Uses the R rms package via rpy2 for nomogram generation.
    Falls back to a Python matplotlib approximation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain duration_col, event_col, and all feature columns.
    duration_col : str
    event_col : str
    features : list[str]
        Columns to include in the nomogram.
    feature_labels : dict, optional
        {column_name: display_label}
    prediction_times : tuple
        Survival probability prediction times (days).
    title : str
    output_dir : Path, optional
    filename : str
    figsize : tuple

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    # Try R rms nomogram via rpy2
    fig = _nomogram_r(
        df, duration_col, event_col, features,
        feature_labels, prediction_times, title,
        output_dir, filename, figsize,
    )

    if fig is None:
        # Fallback to Python approximation
        fig = _nomogram_python(
            df, duration_col, event_col, features,
            feature_labels, prediction_times, title,
            output_dir, filename, figsize,
        )

    return fig


def _nomogram_r(
    df, duration_col, event_col, features,
    feature_labels, prediction_times, title,
    output_dir, filename, figsize,
) -> plt.Figure | None:
    """Attempt R rms nomogram via rpy2."""
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        pandas2ri.activate()

        rms = importr("rms")
        base = importr("base")

        # Prepare formula
        formula_str = f"Surv({duration_col}, {event_col}) ~ {' + '.join(features)}"
        r_df = pandas2ri.py2rpy(df[[duration_col, event_col] + features])

        # Fit Cox model
        ro.r(f'datadist <- datadist({r_df.r_repr()})')
        ro.r('options(datadist = "datadist")')
        fit = rms.cph(
            ro.r(formula_str),
            data=r_df,
            surv=True,
        )
        surv_obj = rms.Survival(fit)

        # Build nomogram
        nom = rms.nomogram(
            fit,
            fun=[lambda x: surv_obj(prediction_times[0], x)],
            funlabel=[f"{prediction_times[0]//365}-Year Survival"],
            lp=False,
        )

        # Save via R graphics
        output_dir = Path(output_dir) if output_dir else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        r_pdf = str(output_dir / f"{filename}.pdf")

        ro.r(f'pdf("{r_pdf}", width={figsize[0]}, height={figsize[1]})')
        ro.r(f"plot({nom.r_repr()})")
        ro.r("dev.off()")

        print(f"  ✅  Nomogram saved via R/rms to {r_pdf}")

        # Read back as matplotlib figure
        import matplotlib.image as mpimg
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(np.ones((100, 100, 3)))  # placeholder
        ax.set_title(f"{title}\n(see {filename}.pdf for full nomogram)", fontsize=11)
        ax.axis("off")

        return fig

    except Exception as exc:
        print(f"  ℹ   R/rms nomogram not available ({exc}). Using Python fallback.")
        return None


def _nomogram_python(
    df, duration_col, event_col, features,
    feature_labels, prediction_times, title,
    output_dir, filename, figsize,
) -> plt.Figure:
    """Python matplotlib approximation of a nomogram."""
    import matplotlib.pyplot as plt
    from lifelines import CoxPHFitter

    # Fit Cox model
    cols = [duration_col, event_col] + features
    sub = df[cols].dropna().copy()

    # Encode categorical features
    encoded_features: list[str] = []
    for feat in features:
        if sub[feat].dtype == object or sub[feat].dtype == bool:
            encoded_name = f"{feat}_encoded"
            sub[encoded_name] = pd.Categorical(sub[feat]).codes.astype(float)
            encoded_features.append(encoded_name)
        else:
            encoded_features.append(feat)

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(sub, duration_col=duration_col, event_col=event_col)

    # Extract coefficients
    coefs = cph.params_

    # Build nomogram visualization
    n_features = len(encoded_features)
    fig, axes = plt.subplots(n_features + 1, 1, figsize=figsize,
                              gridspec_kw={"height_ratios": [1] * n_features + [1.5]})

    points_total = 0

    for idx, feat in enumerate(encoded_features):
        ax = axes[idx]
        label = feature_labels.get(feat, feat) if feature_labels else feat
        coef = coefs.get(feat, 0)
        values = sub[feat].values

        # Create a horizontal bar showing the feature range
        vmin, vmax = values.min(), values.max()
        xs = np.linspace(vmin, vmax, 100)
        points = (xs - vmin) / (vmax - vmin + 1e-10) * 100 * abs(coef)
        ax.plot(points, np.zeros_like(xs), linewidth=3, color="#2166ac")

        # Ticks at quartiles
        for q_val, q_label in [(0.25, "Q1"), (0.5, "Median"), (0.75, "Q3")]:
            q = np.quantile(values, q_val)
            ax.axvline((q - vmin) / (vmax - vmin + 1e-10) * 100 * abs(coef),
                       color="grey", linestyle=":", linewidth=0.5, alpha=0.5)

        ax.set_yticks([])
        ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center",
                      labelpad=60)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("Points" if idx == n_features - 1 else "")

    # Total points → survival probability
    ax_total = axes[-1]
    ax_total.set_title("Total Points → Survival Probability", fontsize=10)
    # Simplified: show linear predictor range
    lp_range = np.linspace(-3, 3, 100)
    surv_1yr = np.exp(-np.exp(lp_range) * prediction_times[0] / prediction_times[0])  # placeholder
    ax_total.plot(lp_range, surv_1yr, linewidth=2, color="#E41A1C",
                  label=f"{prediction_times[0]//365}-Year")
    ax_total.set_xlabel("Total Points (Linear Predictor)")
    ax_total.set_ylabel("Survival Probability")
    ax_total.legend(fontsize=8)
    ax_total.set_ylim(0, 1)

    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig


def calibration_plot(
    predicted: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
    prediction_time_label: str = "1-Year",
    title: str = "Nomogram Calibration",
    output_dir: Path | None = None,
    filename: str = "calibration_plot",
) -> plt.Figure:
    """
    Calibration plot: predicted vs observed survival probabilities.

    Parameters
    ----------
    predicted : np.ndarray
        Predicted survival probabilities.
    observed : np.ndarray
        Observed survival (0/1).
    n_bins : int
        Number of bins for calibration.
    prediction_time_label : str
    title : str
    output_dir : Path, optional
    filename : str

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    set_publication_style()

    fig, ax = plt.subplots(figsize=(5, 5))

    # Bin predictions and compute observed proportions
    bins = np.percentile(predicted, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    observed_rates = []

    for i in range(n_bins):
        mask = (predicted >= bins[i]) & (predicted < bins[i + 1])
        if mask.sum() > 0:
            bin_centers.append(predicted[mask].mean())
            observed_rates.append(observed[mask].mean())

    ax.plot(bin_centers, observed_rates, "o-", color="#2166ac",
            linewidth=2, markersize=8, label="Observed")
    ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1, alpha=0.7,
            label="Perfect calibration")

    ax.set_xlabel(f"Predicted {prediction_time_label} Survival")
    ax.set_ylabel(f"Observed {prediction_time_label} Survival")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_dir is not None:
        save_figure(fig, filename, output_dir)

    return fig
