"""
rna_seq.py — TCGA-GBM RNA-seq STAR-Counts downloader.

Downloads open-access STAR unstranded count TSV files and assembles
a gene × sample count matrix (int64 raw counts, ENSG row index,
TCGA sample barcode column labels).

Output files
------------
  raw_counts/                   extracted TSV files (one dir per file UUID)
  rna_seq_counts.tsv            genes × samples raw count matrix
  rna_seq_metadata.tsv          per-sample metadata (case, sample type, etc.)
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import RNA_SEQ_FILTERS, GBM_PROJECT_ID


# Columns accepted as "unstranded raw count" depending on TSV version
_UNSTRANDED_COLS = ("unstranded", "stranded_unspecific", "count",
                    "expected_count", "raw_count")
# Rows to skip in STAR count files (summary rows beginning with "__")
_STAR_SKIP_PREFIX = "__"


def discover(client: GBMClient, project_id: str = GBM_PROJECT_ID) -> list[dict]:
    """Return file metadata for all STAR-Counts TSV files in project."""
    print(f"  🔍  Discovering RNA-seq STAR-Counts files for {project_id}...")
    records = client.discover_files(project_id, RNA_SEQ_FILTERS)
    print(f"  ✅  {len(records)} RNA-seq files found.")
    return records


def download(
    client: GBMClient,
    records: list[dict],
    output_dir: Path,
) -> Path:
    """Download and extract all STAR-Counts TSV files. Returns raw_counts dir."""
    raw_dir = output_dir / "raw_counts"
    file_ids = [r["file_id"] for r in records]
    client.batch_download(file_ids, raw_dir, label="rna_seq")
    return raw_dir


def build_metadata(records: list[dict]) -> pd.DataFrame:
    """
    Flatten GDC file metadata into a tidy DataFrame.

    Columns: file_id, file_name, case_id, case_submitter_id,
             sample_id, sample_submitter_id, sample_type, tissue_type,
             is_tumor
    """
    rows = []
    for rec in records:
        for case in rec.get("cases", []):
            for samp in case.get("samples", []):
                sample_type = samp.get("sample_type", "")
                # TCGA 4th barcode field: 01=tumor, 11=solid normal, etc.
                sub_id = samp.get("submitter_id", "")
                code = sub_id.split("-")[3][:2] if sub_id.count("-") >= 3 else ""
                rows.append({
                    "file_id":             rec.get("file_id", ""),
                    "file_name":           rec.get("file_name", ""),
                    "file_size":           rec.get("file_size", 0),
                    "case_id":             case.get("case_id", ""),
                    "case_submitter_id":   case.get("submitter_id", ""),
                    "sample_id":           samp.get("sample_id", ""),
                    "sample_submitter_id": sub_id,
                    "sample_type":         sample_type,
                    "tissue_type":         samp.get("tissue_type", ""),
                    "sample_type_code":    code,
                    "is_tumor":            code in {
                        "01","02","03","04","05","06","07","08","09","40","50","60","61"
                    },
                })
    return pd.DataFrame(rows)


def build_count_matrix(
    raw_dir: Path,
    metadata: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse all STAR-Counts TSV files and assemble a genes × samples matrix.

    Returns
    -------
    pd.DataFrame
        Index: Ensembl gene IDs (ENSG...), columns: sample_submitter_id.
        Values are integer raw counts.
    """
    file_id_to_sample = metadata.set_index("file_id")["sample_submitter_id"].to_dict()
    # Fallback: match by filename stem if UUID dir not present
    file_name_to_sample = metadata.set_index("file_name")["sample_submitter_id"].to_dict()

    columns: dict[str, pd.Series] = {}
    skipped = 0

    tsv_files = list(raw_dir.rglob("*.tsv"))
    if verbose:
        print(f"  🔧  Parsing {len(tsv_files)} TSV files...")

    for tsv in tsv_files:
        # Resolve sample label: prefer UUID directory name, fallback to filename
        uid_dir = tsv.parent.name
        sample = (
            file_id_to_sample.get(uid_dir)
            or file_name_to_sample.get(tsv.name)
        )
        if sample is None:
            skipped += 1
            continue

        try:
            df = pd.read_csv(tsv, sep="\t", comment="#", index_col=0,
                             dtype=str, low_memory=False)
        except Exception:
            skipped += 1
            continue

        # Find the unstranded count column
        col_name = next(
            (c for c in df.columns if c.lower() in _UNSTRANDED_COLS), None
        )
        if col_name is None:
            skipped += 1
            continue

        counts = df[col_name]
        # Drop STAR summary rows (__not_aligned, etc.)
        counts = counts[~counts.index.str.startswith(_STAR_SKIP_PREFIX)]
        # Drop non-ENSG rows (gene_name, gene_type header rows in some versions)
        counts = counts[counts.index.str.startswith("ENSG")]
        columns[sample] = counts.astype(int)

    if not columns:
        raise GDCError(
            "No valid STAR-Counts TSV files could be parsed.",
            fix=f"Check that {raw_dir} contains extracted TSV subdirectories.",
            step="count matrix",
        )

    matrix = pd.DataFrame(columns)
    matrix.index.name = "gene_id"
    if verbose:
        print(f"  ✅  Count matrix: {matrix.shape[0]} genes × {matrix.shape[1]} samples "
              f"({skipped} files skipped).")
    return matrix


def run(
    client: GBMClient,
    output_dir: Path,
    project_id: str = GBM_PROJECT_ID,
    skip_download: bool = False,
) -> dict[str, pd.DataFrame | Path]:
    """
    Full RNA-seq pipeline: discover → download → parse.

    Returns
    -------
    dict with keys: "counts" (DataFrame), "metadata" (DataFrame),
                    "raw_dir" (Path), "counts_path" (Path), "metadata_path" (Path)
    """
    records  = discover(client, project_id)
    metadata = build_metadata(records)
    raw_dir  = output_dir / "rna_seq" / "raw_counts"

    if not skip_download:
        file_ids = [r["file_id"] for r in records]
        client.batch_download(file_ids, raw_dir, label="rna_seq")

    counts = build_count_matrix(raw_dir, metadata, verbose=True)

    counts_path   = output_dir / "rna_seq" / "rna_seq_counts.tsv"
    metadata_path = output_dir / "rna_seq" / "rna_seq_metadata.tsv"
    (output_dir / "rna_seq").mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, sep="\t")
    metadata.to_csv(metadata_path, sep="\t", index=False)

    return {
        "counts":        counts,
        "metadata":      metadata,
        "raw_dir":       raw_dir,
        "counts_path":   counts_path,
        "metadata_path": metadata_path,
    }
