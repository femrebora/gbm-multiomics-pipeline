"""
config.py — YAML-based pipeline configuration.

Loads pipeline parameters from a YAML file, environment variables,
and CLI overrides. Provides typed access to all configuration sections.

Usage
-----
    from gbm_multiomics.config import load_config, PipelineConfig

    cfg = load_config("config/pipeline_config.yaml")
    print(cfg.paths.data_dir)
    print(cfg.differential_expression.padj_threshold)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ── Default config path ─────────────────────────────────────────────────────
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pipeline_config.yaml"


# ── Dataclass hierarchy — immutable configuration objects ────────────────────

@dataclass(frozen=True)
class PathsConfig:
    data_dir: str = "~/gbm_multiomics_data"
    results_dir: str = "results"
    figures_dir: str = "results/figures"
    tables_dir: str = "results/tables"
    report_dir: str = "results/report"
    cache_dir: str = "~/.cache/gbm_pipeline"

    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    def resolved_results_dir(self) -> Path:
        return Path(self.results_dir).expanduser().resolve()


@dataclass(frozen=True)
class DownloadConfig:
    data_types: tuple[str, ...] = ("rna-seq", "methylation", "mutations", "cnv", "mirna")
    include_cdr: bool = True
    fresh: bool = False
    token_file: str = ""


@dataclass(frozen=True)
class PreprocessingConfig:
    min_count: int = 10
    min_samples_expressed: float = 0.2
    protein_coding_only: bool = True
    remove_ribosomal: bool = True
    remove_mitochondrial: bool = True
    normalization_method: str = "vst"
    log_transform: bool = True
    pseudocount: float = 1.0
    batch_correction: bool = False
    batch_column: str = "plate_id"
    outlier_iqr_multiplier: float = 3.0
    max_missing_clinical: float = 0.3


@dataclass(frozen=True)
class DEComparison:
    condition: str
    reference: str
    label: str


@dataclass(frozen=True)
class DifferentialExpressionConfig:
    comparisons: tuple[DEComparison, ...] = (
        DEComparison("is_tumor", "False", "Tumor_vs_Normal"),
    )
    padj_threshold: float = 0.05
    lfc_threshold: float = 1.0
    n_cpus: int = 4
    batch_aware: bool = False
    batch_column: str = "plate_id"


@dataclass(frozen=True)
class UnivariateCoxConfig:
    padj_threshold: float = 0.05
    min_events_per_group: int = 10


@dataclass(frozen=True)
class LassoCoxConfig:
    n_folds: int = 10
    n_lambda: int = 100
    alpha: float = 1.0
    max_features: int = 50


@dataclass(frozen=True)
class MultivariateCoxConfig:
    clinical_covariates: tuple[str, ...] = ("age_at_diagnosis", "IDH_status", "MGMT_status", "gender")
    test_ph_assumption: bool = True


@dataclass(frozen=True)
class RiskGroupConfig:
    n_groups: int = 2
    split_method: str = "median"


@dataclass(frozen=True)
class TimeROCConfig:
    times: tuple[int, ...] = (365, 730, 1095)


@dataclass(frozen=True)
class DepMapConfig:
    enabled: bool = True
    depmap_version: str = "24Q2"


@dataclass(frozen=True)
class PrognosticConfig:
    endpoints: tuple[str, ...] = ("OS", "PFI", "DSS")
    univariate: UnivariateCoxConfig = field(default_factory=UnivariateCoxConfig)
    lasso: LassoCoxConfig = field(default_factory=LassoCoxConfig)
    multivariate: MultivariateCoxConfig = field(default_factory=MultivariateCoxConfig)
    risk_groups: RiskGroupConfig = field(default_factory=RiskGroupConfig)
    time_roc: TimeROCConfig = field(default_factory=TimeROCConfig)
    depmap: DepMapConfig = field(default_factory=DepMapConfig)


@dataclass(frozen=True)
class MOFAConfig:
    enabled: bool = True
    n_factors: int = 10
    n_top_features: int = 5000


@dataclass(frozen=True)
class SNFConfig:
    enabled: bool = True
    n_clusters_range: tuple[int, ...] = (2, 3, 4, 5, 6)
    n_neighbors: int = 20


@dataclass(frozen=True)
class CrossOmicsCorrelationConfig:
    promoter_window_bp: int = 2000
    cis_window_bp: int = 1_000_000


@dataclass(frozen=True)
class MultiOmicsConfig:
    mofa: MOFAConfig = field(default_factory=MOFAConfig)
    snf: SNFConfig = field(default_factory=SNFConfig)
    cross_omics_correlation: CrossOmicsCorrelationConfig = field(default_factory=CrossOmicsCorrelationConfig)


@dataclass(frozen=True)
class ImmuneConfig:
    estimate_enabled: bool = True


@dataclass(frozen=True)
class NetworkConfig:
    string_enabled: bool = True
    string_score_threshold: int = 700
    n_hub_genes: int = 20


@dataclass(frozen=True)
class TMBConfig:
    tmb_genome_size_mb: float = 38.0
    tmb_high_threshold: float = 10.0


@dataclass(frozen=True)
class SupportingConfig:
    immune: ImmuneConfig = field(default_factory=ImmuneConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    tumor_mutation_burden: TMBConfig = field(default_factory=TMBConfig)


@dataclass(frozen=True)
class ThemeConfig:
    style: str = "publication"
    dpi: int = 300
    format: tuple[str, ...] = ("pdf", "png")
    font_family: str = "DejaVu Sans"
    font_size_base: int = 10
    color_palette: str = "GBM_custom"


@dataclass(frozen=True)
class VolcanoConfig:
    n_label: int = 20
    lfc_cutoff: float = 1.0
    padj_cutoff: float = 0.05


@dataclass(frozen=True)
class KMConfig:
    show_ci: bool = True
    show_at_risk: bool = True
    x_breaks_days: int = 365
    max_time_days: int = 1825


@dataclass(frozen=True)
class HeatmapConfig:
    n_top_genes: int = 50
    cluster_rows: bool = True
    cluster_cols: bool = True
    show_row_names: bool = False


@dataclass(frozen=True)
class ForestConfig:
    sort_by: str = "HR"
    show_pvalue: bool = True


@dataclass(frozen=True)
class VisualizationConfig:
    theme: ThemeConfig = field(default_factory=ThemeConfig)
    volcano: VolcanoConfig = field(default_factory=VolcanoConfig)
    km: KMConfig = field(default_factory=KMConfig)
    heatmap: HeatmapConfig = field(default_factory=HeatmapConfig)
    forest: ForestConfig = field(default_factory=ForestConfig)


@dataclass(frozen=True)
class ReportConfig:
    format: str = "pdf"
    include_code: bool = False
    include_supplementary: bool = True
    citation_style: str = "cell"
    output_basename: str = "gbm_prognostic_report"


@dataclass(frozen=True)
class PipelineConfig:
    """Top-level immutable configuration."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    differential_expression: DifferentialExpressionConfig = field(default_factory=DifferentialExpressionConfig)
    prognostic: PrognosticConfig = field(default_factory=PrognosticConfig)
    multiomics: MultiOmicsConfig = field(default_factory=MultiOmicsConfig)
    supporting: SupportingConfig = field(default_factory=SupportingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    _meta: dict[str, Any] = field(default_factory=dict)


# ── Config loader ────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _apply_env_overrides(config_dict: dict, prefix: str = "GBM_") -> dict:
    """Override config values from environment variables like GBM_PATHS__DATA_DIR."""
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        # GBM_PATHS__DATA_DIR → ["paths", "data_dir"]
        key_path = env_key[len(prefix):].lower().split("__")
        target = config_dict
        for part in key_path[:-1]:
            if part not in target:
                target[part] = {}
            target = target[part]
        # Type coercion
        raw = env_val
        if raw.lower() in ("true", "false"):
            raw = raw.lower() == "true"
        elif raw.isdigit():
            raw = int(raw)
        elif raw.replace(".", "", 1).isdigit():
            raw = float(raw)
        target[key_path[-1]] = raw
    return config_dict


def _build_config(raw: dict) -> PipelineConfig:
    """Convert raw dict to typed PipelineConfig."""

    # Paths
    paths_raw = raw.get("paths", {})
    paths = PathsConfig(**{k: v for k, v in paths_raw.items()
                           if k in PathsConfig.__dataclass_fields__})

    # Download
    dl = raw.get("download", {})
    dl["data_types"] = tuple(dl.get("data_types", ["rna-seq"]))
    download = DownloadConfig(**{k: v for k, v in dl.items()
                                 if k in DownloadConfig.__dataclass_fields__})

    # Preprocessing
    pp = raw.get("preprocessing", {})
    preprocessing = PreprocessingConfig(**{k: v for k, v in pp.items()
                                           if k in PreprocessingConfig.__dataclass_fields__})

    # DE
    de_raw = raw.get("differential_expression", {})
    comparisons_raw = de_raw.pop("comparisons", [])
    comparisons = tuple(DEComparison(**c) for c in comparisons_raw)
    de = DifferentialExpressionConfig(
        comparisons=comparisons,
        **{k: v for k, v in de_raw.items()
           if k in DifferentialExpressionConfig.__dataclass_fields__ and k != "comparisons"},
    )

    # Prognostic
    prog_raw = raw.get("prognostic", {})
    uni = UnivariateCoxConfig(**prog_raw.get("univariate", {}))
    lasso = LassoCoxConfig(**prog_raw.get("lasso", {}))
    mv_raw = prog_raw.get("multivariate", {})
    mv_raw["clinical_covariates"] = tuple(mv_raw.get("clinical_covariates", []))
    mv = MultivariateCoxConfig(**mv_raw)
    rg = RiskGroupConfig(**prog_raw.get("risk_groups", {}))
    tr_raw = prog_raw.get("time_roc", {})
    tr_raw["times"] = tuple(tr_raw.get("times", []))
    tr = TimeROCConfig(**tr_raw)
    dm = DepMapConfig(**prog_raw.get("depmap", {}))
    prognostic = PrognosticConfig(
        endpoints=tuple(prog_raw.get("endpoints", ["OS"])),
        univariate=uni, lasso=lasso, multivariate=mv,
        risk_groups=rg, time_roc=tr, depmap=dm,
    )

    # Multi-omics
    mo_raw = raw.get("multiomics", {})
    mofa = MOFAConfig(**mo_raw.get("mofa", {}))
    snf_raw = mo_raw.get("snf", {})
    snf_raw["n_clusters_range"] = tuple(snf_raw.get("n_clusters_range", [2, 3, 4, 5]))
    snf = SNFConfig(**snf_raw)
    co = CrossOmicsCorrelationConfig(**mo_raw.get("cross_omics_correlation", {}))
    multiomics = MultiOmicsConfig(mofa=mofa, snf=snf, cross_omics_correlation=co)

    # Supporting
    supp_raw = raw.get("supporting", {})
    immune = ImmuneConfig(**supp_raw.get("immune", {}))
    net = NetworkConfig(**supp_raw.get("network", {}))
    tmb = TMBConfig(**supp_raw.get("tumor_mutation_burden", {}))
    supporting = SupportingConfig(immune=immune, network=net, tumor_mutation_burden=tmb)

    # Visualization
    vis_raw = raw.get("visualization", {})
    theme = ThemeConfig(**vis_raw.get("theme", {}))
    volc = VolcanoConfig(**vis_raw.get("volcano", {}))
    km = KMConfig(**vis_raw.get("km", {}))
    heatmap = HeatmapConfig(**vis_raw.get("heatmap", {}))
    forest = ForestConfig(**vis_raw.get("forest", {}))
    visualization = VisualizationConfig(theme=theme, volcano=volc, km=km,
                                         heatmap=heatmap, forest=forest)

    # Report
    report = ReportConfig(**raw.get("report", {}))

    return PipelineConfig(
        paths=paths,
        download=download,
        preprocessing=preprocessing,
        differential_expression=de,
        prognostic=prognostic,
        multiomics=multiomics,
        supporting=supporting,
        visualization=visualization,
        report=report,
        _meta=raw.get("pipeline", {}),
    )


def load_config(config_path: str | Path | None = None) -> PipelineConfig:
    """
    Load pipeline configuration from a YAML file.

    Resolution order (last wins):
      1. Built-in defaults (dataclass defaults)
      2. YAML config file (~/gbm_multiomics_data/config.yaml or CONFIG_PATH env)
      3. Environment variables (GBM_ prefix, double-underscore separator)

    Parameters
    ----------
    config_path : str or Path, optional
        Path to YAML config file. Defaults to config/pipeline_config.yaml
        in the package root, then falls back to built-in defaults.

    Returns
    -------
    PipelineConfig
        Immutable typed configuration.
    """
    # Start with empty (defaults will fill in)
    merged: dict[str, Any] = {}

    # Determine config file path
    if config_path is not None:
        cfg_path = Path(config_path).expanduser()
    elif "GBM_CONFIG" in os.environ:
        cfg_path = Path(os.environ["GBM_CONFIG"]).expanduser()
    elif DEFAULT_CONFIG_PATH.exists():
        cfg_path = DEFAULT_CONFIG_PATH
    else:
        # No config file — use env overrides on defaults
        merged = _apply_env_overrides(merged)
        return _build_config(merged)

    if cfg_path.exists():
        with open(cfg_path, "r") as fh:
            file_config = yaml.safe_load(fh) or {}
        merged = _deep_merge(merged, file_config)

    # Apply environment overrides
    merged = _apply_env_overrides(merged)

    return _build_config(merged)


# ── CLI override helper ──────────────────────────────────────────────────────

def apply_cli_overrides(config: PipelineConfig, **overrides: Any) -> PipelineConfig:
    """
    Apply individual CLI overrides by re-creating config with changes.

    Example
    -------
    cfg = apply_cli_overrides(cfg, padj_threshold=0.01, n_cpus=8)
    """
    raw = {
        "paths": config.paths.__dict__,
        "download": config.download.__dict__,
        "preprocessing": {**config.preprocessing.__dict__},
        "differential_expression": {
            **config.differential_expression.__dict__,
            "comparisons": [c.__dict__ for c in config.differential_expression.comparisons],
        },
        "prognostic": {
            "endpoints": list(config.prognostic.endpoints),
            "univariate": config.prognostic.univariate.__dict__,
            "lasso": config.prognostic.lasso.__dict__,
            "multivariate": {
                **config.prognostic.multivariate.__dict__,
                "clinical_covariates": list(config.prognostic.multivariate.clinical_covariates),
            },
            "risk_groups": config.prognostic.risk_groups.__dict__,
            "time_roc": {**config.prognostic.time_roc.__dict__,
                         "times": list(config.prognostic.time_roc.times)},
            "depmap": config.prognostic.depmap.__dict__,
        },
    }
    # Apply overrides at top level
    for key, value in overrides.items():
        # Walk nested dicts to find the key
        _set_nested(raw, key, value)
    return _build_config(raw)


def _set_nested(d: dict, key: str, value: Any) -> None:
    """Set a nested key in dict, creating intermediate dicts as needed."""
    if "__" in key:
        parts = key.split("__")
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    else:
        # Top-level — try to find which section
        for section in d.values():
            if isinstance(section, dict) and key in section:
                section[key] = value
                return
        d[key] = value
