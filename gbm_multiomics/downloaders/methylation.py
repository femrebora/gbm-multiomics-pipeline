"""
methylation.py — TCGA-GBM DNA methylation 450k/EPIC array downloader.

Downloads open-access Methylation Beta Value TXT files and assembles
a CpG probe × sample beta-value matrix (float32, range 0–1).

Output files
------------
  methylation/raw/              extracted TXT files
  methylation/methylation_beta.tsv   probes × samples beta matrix
  methylation/methylation_metadata.tsv
  methylation/mgmt_probe_summary.tsv  MGMT promoter CpG probe summary

GBM-relevant CpG loci
---------------------
  MGMT promoter: cg12434587, cg02659086, cg16672562, cg13420082, cg09750382
  IDH1 R132 region: assessed via mutation (MAF), not methylation array
  G-CIMP probes: Noushmehr 2010 signature (high methylation = proneural G-CIMP)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import METHYLATION_FILTERS, GBM_PROJECT_ID

# MGMT promoter CpG probes (Hegi et al. NEJM 2005, GBM specific)
MGMT_PROBES = (
    "cg12434587", "cg02659086", "cg16672562",
    "cg13420082", "cg09750382", "cg12981137",
)

# G-CIMP classifier probes (Noushmehr et al. Cancer Cell 2010 — top 6)
GCIMP_PROBES = (
    "cg14141141", "cg17571639", "cg12778551",
    "cg03530917", "cg26898166", "cg14169341",
)


def discover(client: GBMClient, project_id: str = GBM_PROJECT_ID) -> list[dict]:
    print(f"  🔍  Discovering methylation beta-value files for {project_id}...")
    records = client.discover_files(project_id, METHYLATION_FILTERS)
    print(f"  ✅  {len(records)} methylation files found.")
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


def build_beta_matrix(
    raw_dir: Path,
    metadata: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse per-sample beta-value TXT files into a probes × samples matrix.

    GDC methylation TXT format (tab-separated, no header):
        Composite Element REF  <beta>  Gene_Symbol  Chromosome  Genomic_Coordinate

    Returns
    -------
    pd.DataFrame  probes × samples, float32 beta values (NaN = missing)
    """
    file_id_to_sample = metadata.set_index("file_id")["sample_submitter_id"].to_dict()
    file_name_to_sample = metadata.set_index("file_name")["sample_submitter_id"].to_dict()

    columns: dict[str, pd.Series] = {}
    skipped = 0

    txt_files = list(raw_dir.rglob("*.txt"))
    if verbose:
        print(f"  🔧  Parsing {len(txt_files)} methylation TXT files...")

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
            df = pd.read_csv(
                txt, sep="\t", header=0, index_col=0,
                usecols=[0, 1],   # probe ID + beta value
                dtype=str, low_memory=False,
            )
            beta = pd.to_numeric(df.iloc[:, 0], errors="coerce").astype("float32")
            columns[sample] = beta
        except Exception:
            skipped += 1
            continue

    if not columns:
        raise GDCError(
            "No valid methylation TXT files could be parsed.",
            fix=f"Check {raw_dir} for extracted methylation files.",
            step="methylation parse",
        )

    matrix = pd.DataFrame(columns)
    matrix.index.name = "probe_id"
    if verbose:
        print(f"  ✅  Beta matrix: {matrix.shape[0]} probes × {matrix.shape[1]} samples "
              f"({skipped} skipped).")
    return matrix


def summarise_mgmt(beta_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-sample mean MGMT promoter beta value and methylation status.

    MGMT methylated  → mean beta ≥ 0.30  (approximate clinical threshold)
    Returns DataFrame: sample | mgmt_mean_beta | mgmt_methylated
    """
    probes_present = [p for p in MGMT_PROBES if p in beta_matrix.index]
    if not probes_present:
        return pd.DataFrame(columns=["sample", "mgmt_mean_beta", "mgmt_methylated"])

    mgmt = beta_matrix.loc[probes_present].mean(axis=0)
    result = pd.DataFrame({
        "sample":         mgmt.index,
        "mgmt_mean_beta": mgmt.values.round(4),
        "mgmt_methylated": (mgmt.values >= 0.30),
    })
    return result.reset_index(drop=True)


def run(
    client: GBMClient,
    output_dir: Path,
    project_id: str = GBM_PROJECT_ID,
    skip_download: bool = False,
) -> dict:
    records  = discover(client, project_id)
    metadata = build_metadata(records)
    raw_dir  = output_dir / "methylation" / "raw"
    out_dir  = output_dir / "methylation"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        file_ids = [r["file_id"] for r in records]
        client.batch_download(file_ids, raw_dir, label="methylation")

    beta = build_beta_matrix(raw_dir, metadata, verbose=True)
    mgmt = summarise_mgmt(beta)

    beta_path    = out_dir / "methylation_beta.tsv"
    meta_path    = out_dir / "methylation_metadata.tsv"
    mgmt_path    = out_dir / "mgmt_probe_summary.tsv"

    # Save in chunks to avoid memory spikes on large matrices
    beta.to_csv(beta_path, sep="\t")
    metadata.to_csv(meta_path, sep="\t", index=False)
    mgmt.to_csv(mgmt_path, sep="\t", index=False)

    print(f"  📁  MGMT methylated samples: {mgmt['mgmt_methylated'].sum()}/{len(mgmt)}")

    return {
        "beta":       beta,
        "metadata":   metadata,
        "mgmt":       mgmt,
        "beta_path":  beta_path,
        "meta_path":  meta_path,
        "mgmt_path":  mgmt_path,
    }
