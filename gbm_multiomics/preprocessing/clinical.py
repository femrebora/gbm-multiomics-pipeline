"""
clinical.py — Clinical data integration for GBM multiomics.

Merges GDC clinical data, PanCanAtlas CDR survival annotations,
molecular features (IDH status, MGMT methylation, subtypes)
into a unified per-sample metadata DataFrame.

References
----------
  Liu et al. (2018) Cell 173:400-416 — PanCanAtlas CDR
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_clinical_data(
    data_dir: Path,
    project_id: str = "TCGA-GBM",
) -> pd.DataFrame:
    """
    Load and merge GDC clinical + CDR data for all samples.

    Parameters
    ----------
    data_dir : Path
        Directory containing RNA-seq output (with CDR data).
    project_id : str

    Returns
    -------
    pd.DataFrame
        Per-sample clinical annotations.
    """
    data_dir = Path(data_dir)
    clinical_df = pd.DataFrame()

    # Try loading CDR-merged file first (most complete)
    cdr_files = list((data_dir / "rna_seq").glob("*cdr*.tsv"))
    if cdr_files:
        clinical_df = pd.read_csv(cdr_files[0], sep="\t", dtype=str, low_memory=False)
        print(f"  📂  Loaded CDR-merged clinical data: {len(clinical_df)} samples.")
        return clinical_df

    # Fallback: load metadata
    meta_path = data_dir / "rna_seq" / "rna_seq_metadata.tsv"
    if meta_path.exists():
        clinical_df = pd.read_csv(meta_path, sep="\t", dtype=str)
        print(f"  📂  Loaded RNA-seq metadata: {len(clinical_df)} samples.")
        return clinical_df

    print("  ⚠   No clinical data found. Run download first.")
    return clinical_df


def merge_molecular_features(
    clinical_df: pd.DataFrame,
    data_dir: Path,
    sample_col: str = "case_submitter_id",
) -> pd.DataFrame:
    """
    Merge molecular features (IDH, MGMT, subtypes) into clinical DataFrame.

    Looks for:
      - mutations/idh_status.tsv (IDH1/IDH2 mutation status)
      - methylation/mgmt_probe_summary.tsv (MGMT promoter methylation)
      - analysis/subtype/gbm_subtypes_centroid.tsv (Verhaak subtypes)

    Parameters
    ----------
    clinical_df : pd.DataFrame
        Base clinical DataFrame.
    data_dir : Path
        Root data directory.
    sample_col : str
        Column to merge on (case or sample submitter ID).

    Returns
    -------
    pd.DataFrame
        Clinical DataFrame with added molecular feature columns.
    """
    df = clinical_df.copy()
    data_dir = Path(data_dir)

    # ── IDH status ────────────────────────────────────────────────────────
    idh_path = data_dir / "mutations" / "idh_status.tsv"
    if idh_path.exists():
        idh_df = pd.read_csv(idh_path, sep="\t", dtype=str)
        merge_key = next(
            (c for c in idh_df.columns if sample_col in c.lower()), None
        )
        if merge_key is None:
            merge_key = idh_df.columns[0] if len(idh_df.columns) > 0 else None

        if merge_key and merge_key in idh_df.columns:
            df = df.merge(
                idh_df[[merge_key, "IDH_status"]],
                left_on=sample_col, right_on=merge_key, how="left",
            )
            df = df.drop(columns=[merge_key], errors="ignore")
            print(f"  🧬  Merged IDH status: "
                  f"{df['IDH_status'].notna().sum()} samples annotated.")

    # ── MGMT methylation ─────────────────────────────────────────────────
    mgmt_path = data_dir / "methylation" / "mgmt_probe_summary.tsv"
    if mgmt_path.exists():
        mgmt_df = pd.read_csv(mgmt_path, sep="\t", dtype=str)
        merge_key = next(
            (c for c in mgmt_df.columns if sample_col in c.lower() or "sample" in c.lower()),
            None,
        )
        if merge_key is None:
            merge_key = mgmt_df.columns[0] if len(mgmt_df.columns) > 0 else None

        if merge_key and merge_key in mgmt_df.columns:
            mgmt_cols = [merge_key]
            if "MGMT_status" in mgmt_df.columns:
                mgmt_cols.append("MGMT_status")
            if "MGMT_mean_beta" in mgmt_df.columns:
                mgmt_cols.append("MGMT_mean_beta")
            df = df.merge(
                mgmt_df[mgmt_cols],
                left_on=sample_col, right_on=merge_key, how="left",
            )
            df = df.drop(columns=[merge_key], errors="ignore")
            print(f"  🧬  Merged MGMT methylation: "
                  f"{df.get('MGMT_status', pd.Series(dtype=str)).notna().sum()} samples annotated.")

    # ── Subtype ──────────────────────────────────────────────────────────
    subtype_path = data_dir / "analysis" / "subtype" / "gbm_subtypes_centroid.tsv"
    if not subtype_path.exists():
        subtype_path = data_dir / "subtype" / "gbm_subtypes_centroid.tsv"

    if subtype_path.exists():
        sub_df = pd.read_csv(subtype_path, sep="\t", dtype=str)
        merge_key = next(
            (c for c in sub_df.columns if "sample" in c.lower() or "barcode" in c.lower()),
            "sample",
        )
        if merge_key in sub_df.columns:
            df = df.merge(
                sub_df[[merge_key, "assigned_subtype"]],
                left_on=sample_col, right_on=merge_key, how="left",
            )
            df = df.drop(columns=[merge_key], errors="ignore")
            print(f"  🧬  Merged Verhaak subtypes: "
                  f"{df['assigned_subtype'].notna().sum()} samples classified.")

    return df


def build_unified_metadata(
    data_dir: Path,
    project_id: str = "TCGA-GBM",
    sample_col: str = "case_submitter_id",
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build unified per-sample metadata with clinical + molecular features.

    This is the main entry point for clinical data integration.
    It loads clinical/CDR data, merges all molecular features,
    and writes the unified table.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing all omics output.
    project_id : str
        TCGA project ID.
    sample_col : str
        Column to use as sample identifier for merging.
    output_dir : Path, optional
        If provided, writes unified_metadata.tsv.

    Returns
    -------
    pd.DataFrame
        Unified sample metadata with columns:
        - Clinical: submitter_id, gender, age_at_diagnosis, vital_status, ...
        - CDR survival: cdr_OS, cdr_OS.time, cdr_PFI, cdr_PFI.time, ...
        - Molecular: IDH_status, MGMT_status, assigned_subtype
        - QC flags: cdr_matched, cdr_survival_complete
    """
    print(f"\n{'='*60}")
    print("  Building Unified Sample Metadata")
    print(f"{'='*60}\n")

    # Load clinical base
    df = load_clinical_data(data_dir, project_id)

    # Merge molecular features
    df = merge_molecular_features(df, data_dir, sample_col)

    # Coerce numeric columns
    numeric_cols = [
        "age_at_diagnosis", "cdr_OS", "cdr_OS.time",
        "cdr_PFI", "cdr_PFI.time", "cdr_DSS", "cdr_DSS.time",
        "cdr_DFI", "cdr_DFI.time",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Add survival data quality flags
    if "cdr_OS" in df.columns and "cdr_OS.time" in df.columns:
        df["has_survival_data"] = (
            df["cdr_OS"].notna() &
            df["cdr_OS.time"].notna() &
            (df["cdr_OS.time"] > 0) &
            df["cdr_OS"].isin([0, 1])
        )
        n_surv = df["has_survival_data"].sum()
        print(f"  📊  Samples with complete OS data: {n_surv}/{len(df)}")

    # Add age group
    if "age_at_diagnosis" in df.columns:
        age = pd.to_numeric(df["age_at_diagnosis"], errors="coerce")
        df["age_group"] = pd.cut(
            age,
            bins=[0, 40, 55, 70, 200],
            labels=["<40", "40-55", "55-70", ">70"],
        ).astype(str)

    # Summary
    n_total = len(df)
    print(f"\n  ✅  Unified metadata: {n_total} samples, "
          f"{len(df.columns)} columns.")
    print(f"  Columns: {', '.join(df.columns.tolist()[:15])}...")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / "unified_metadata.tsv", sep="\t", index=False)
        print(f"  📄  Saved to {output_dir / 'unified_metadata.tsv'}")

    return df
