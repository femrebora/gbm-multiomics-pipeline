"""
mirna.py — TCGA-GBM miRNA-seq downloader.

Downloads open-access miRNA Expression Quantification TXT files
(BCGSC miRNA Profiling pipeline) and assembles a
miRNA × sample RPM (reads per million mapped miRNA) matrix.

GBM-relevant miRNAs
--------------------
  miR-21-5p   — oncomiR; promotes GBM proliferation / anti-apoptosis
  miR-10b-5p  — promotes invasion; enriched in GBM vs normal brain
  miR-182-5p  — targets FOXO3; associated with poor survival
  miR-128-3p  — tumour suppressor; lost in GBM
  miR-7-5p    — targets EGFR/IRS-1; tumour suppressor in GBM
  miR-34a-5p  — p53 target; lost in GBM with TP53 mutation

Output files
------------
  mirna/raw/                        extracted TXT files
  mirna/mirna_rpm.tsv               miRNA × samples RPM matrix
  mirna/mirna_metadata.tsv
  mirna/gbm_mirna_summary.tsv       GBM-relevant miRNA expression summary

GDC miRNA TXT format (per sample, tab-separated with header):
  miRNA_ID | read_count | reads_per_million_miRNA_mapped | cross-mapped
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import MIRNA_FILTERS, GBM_PROJECT_ID

# GBM-relevant miRNAs for the summary output
GBM_MIRNAS = (
    "hsa-mir-21",
    "hsa-mir-10b",
    "hsa-mir-182",
    "hsa-mir-128-1",
    "hsa-mir-128-2",
    "hsa-mir-7-1",
    "hsa-mir-7-2",
    "hsa-mir-34a",
    "hsa-mir-9-1",
    "hsa-mir-9-2",
    "hsa-mir-9-3",
    "hsa-mir-155",
    "hsa-mir-221",
    "hsa-mir-222",
)

RPM_COL = "reads_per_million_miRNA_mapped"
READ_COL = "read_count"
ID_COL   = "miRNA_ID"


def discover(client: GBMClient, project_id: str = GBM_PROJECT_ID) -> list[dict]:
    print(f"  🔍  Discovering miRNA-seq files for {project_id}...")
    records = client.discover_files(project_id, MIRNA_FILTERS)
    print(f"  ✅  {len(records)} miRNA files found.")
    return records


def build_metadata(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        for case in rec.get("cases", []):
            for samp in case.get("samples", []):
                sub_id = samp.get("submitter_id", "")
                code = sub_id.split("-")[3][:2] if sub_id.count("-") >= 3 else ""
                rows.append({
                    "file_id":             rec.get("file_id", ""),
                    "file_name":           rec.get("file_name", ""),
                    "case_submitter_id":   case.get("submitter_id", ""),
                    "sample_submitter_id": sub_id,
                    "sample_type":         samp.get("sample_type", ""),
                    "is_tumor":            code in {
                        "01","02","03","04","05","06","07",
                        "08","09","40","50","60","61"
                    },
                })
    return pd.DataFrame(rows)


def build_rpm_matrix(
    raw_dir: Path,
    metadata: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse per-sample miRNA TXT files and assemble a miRNA × samples RPM matrix.

    Uses reads_per_million_miRNA_mapped as the expression measure.
    Excludes cross-mapped reads (cross-mapped == "Y").

    Returns
    -------
    pd.DataFrame  miRNA_ID rows × sample columns, float64 RPM values.
    """
    file_id_to_sample = metadata.set_index("file_id")["sample_submitter_id"].to_dict()
    file_name_to_sample = metadata.set_index("file_name")["sample_submitter_id"].to_dict()

    txt_files = list(raw_dir.rglob("*.txt"))
    if verbose:
        print(f"  🔧  Parsing {len(txt_files)} miRNA TXT files...")

    columns: dict[str, pd.Series] = {}
    skipped = 0

    for txt in txt_files:
        uid_dir = txt.parent.name
        sample = (
            file_id_to_sample.get(uid_dir)
            or file_name_to_sample.get(txt.name)
        )
        if sample is None:
            skipped += 1
            continue

        try:
            df = pd.read_csv(txt, sep="\t", dtype=str, low_memory=False)

            if ID_COL not in df.columns or RPM_COL not in df.columns:
                skipped += 1
                continue

            # Filter out cross-mapped entries if the column exists
            if "cross-mapped" in df.columns:
                df = df[df["cross-mapped"].str.upper() != "Y"].copy()

            df = df.set_index(ID_COL)
            rpm = pd.to_numeric(df[RPM_COL], errors="coerce")
            columns[sample] = rpm

        except Exception:
            skipped += 1
            continue

    if not columns:
        raise GDCError(
            "No valid miRNA TXT files could be parsed.",
            fix=f"Check {raw_dir} for extracted miRNA quantification files.",
            step="mirna parse",
        )

    matrix = pd.DataFrame(columns)
    matrix.index.name = "miRNA_ID"
    if verbose:
        print(f"  ✅  RPM matrix: {matrix.shape[0]} miRNAs × {matrix.shape[1]} samples "
              f"({skipped} skipped).")
    return matrix


def summarise_gbm_mirnas(rpm_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return mean ± std RPM for GBM-relevant miRNAs across all tumour samples.
    """
    present = [m for m in GBM_MIRNAS if m in rpm_matrix.index]
    if not present:
        return pd.DataFrame(columns=["miRNA_ID", "mean_rpm", "std_rpm", "n_samples"])

    sub = rpm_matrix.loc[present].T
    summary = pd.DataFrame({
        "miRNA_ID":  sub.columns.tolist(),
        "mean_rpm":  sub.mean().round(2).values,
        "std_rpm":   sub.std().round(2).values,
        "n_samples": sub.notna().sum().values,
    })
    return summary.sort_values("mean_rpm", ascending=False).reset_index(drop=True)


def run(
    client: GBMClient,
    output_dir: Path,
    project_id: str = GBM_PROJECT_ID,
    skip_download: bool = False,
) -> dict:
    records  = discover(client, project_id)
    metadata = build_metadata(records)
    raw_dir  = output_dir / "mirna" / "raw"
    out_dir  = output_dir / "mirna"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        file_ids = [r["file_id"] for r in records]
        client.batch_download(file_ids, raw_dir, label="mirna")

    rpm     = build_rpm_matrix(raw_dir, metadata, verbose=True)
    summary = summarise_gbm_mirnas(rpm)

    rpm_path     = out_dir / "mirna_rpm.tsv"
    meta_path    = out_dir / "mirna_metadata.tsv"
    summary_path = out_dir / "gbm_mirna_summary.tsv"

    rpm.to_csv(rpm_path,         sep="\t")
    metadata.to_csv(meta_path,   sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)

    return {
        "rpm":          rpm,
        "metadata":     metadata,
        "gbm_summary":  summary,
        "rpm_path":     rpm_path,
        "meta_path":    meta_path,
        "summary_path": summary_path,
    }
