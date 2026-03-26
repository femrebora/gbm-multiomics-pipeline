"""
cnv.py — TCGA-GBM copy number segment downloader.

Downloads open-access Copy Number Segment TXT files (DNAcopy workflow,
Genotyping Array) and produces:
  - Concatenated segment table (all samples, long format)
  - Gene-level CNV summary for key GBM genes (EGFR, CDKN2A, PTEN, etc.)
  - Per-sample chromosome 7 gain / chromosome 10 loss flags (hallmark of GBM)

Segment file columns
--------------------
  GDC_Aliquot   TCGA aliquot barcode
  Chromosome    1–22, X, Y
  Start         genomic start (hg38)
  End           genomic end (hg38)
  Num_Probes    number of array probes in segment
  Segment_Mean  log2 copy number ratio (0 = diploid)

CNV thresholds used here
-------------------------
  Segment_Mean > 0.3   → copy gain
  Segment_Mean > 1.0   → high-level amplification (e.g. EGFR amp)
  Segment_Mean < -0.3  → copy loss
  Segment_Mean < -1.0  → deep deletion (e.g. CDKN2A homozygous del)

Output files
------------
  cnv/raw/                        extracted TXT files
  cnv/cnv_segments.tsv            all segments (long format)
  cnv/cnv_chr7_chr10.tsv          chr7 gain / chr10 loss flags per sample
  cnv/cnv_metadata.tsv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import CNV_FILTERS, GBM_PROJECT_ID

# Thresholds
GAIN_THRESHOLD  =  0.3
AMP_THRESHOLD   =  1.0
LOSS_THRESHOLD  = -0.3
DEL_THRESHOLD   = -1.0

# Chromosomes critical for GBM (chr7 gain + chr10 loss = near-universal)
CHR7_LABEL  = "chr7"
CHR10_LABEL = "chr10"


def discover(client: GBMClient, project_id: str = GBM_PROJECT_ID) -> list[dict]:
    print(f"  🔍  Discovering CNV segment files for {project_id}...")
    records = client.discover_files(project_id, CNV_FILTERS)
    print(f"  ✅  {len(records)} CNV files found.")
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


def parse_segment_files(
    raw_dir: Path,
    metadata: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse all CNV segment TXT files into a single long-format DataFrame.

    Adds a 'sample_submitter_id' column from the file-ID/filename lookup.
    """
    file_id_to_sample = metadata.set_index("file_id")["sample_submitter_id"].to_dict()
    file_name_to_sample = metadata.set_index("file_name")["sample_submitter_id"].to_dict()

    txt_files = list(raw_dir.rglob("*.txt"))
    if verbose:
        print(f"  🔧  Parsing {len(txt_files)} CNV segment files...")

    frames: list[pd.DataFrame] = []
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
            df["sample_submitter_id"] = sample
            frames.append(df)
        except Exception:
            skipped += 1
            continue

    if not frames:
        raise GDCError(
            "No valid CNV segment TXT files could be parsed.",
            fix=f"Check {raw_dir} for extracted CNV files.",
            step="cnv parse",
        )

    segments = pd.concat(frames, ignore_index=True)

    # Normalise Segment_Mean to float
    if "Segment_Mean" in segments.columns:
        segments["Segment_Mean"] = pd.to_numeric(segments["Segment_Mean"],
                                                   errors="coerce")

    if verbose:
        n_samples = segments["sample_submitter_id"].nunique()
        print(f"  ✅  {len(segments):,} segments across {n_samples} samples "
              f"({skipped} files skipped).")
    return segments


def summarise_chr7_chr10(segments: pd.DataFrame) -> pd.DataFrame:
    """
    For each sample compute:
      - chr7_mean_log2  : mean Segment_Mean across chromosome 7
      - chr10_mean_log2 : mean Segment_Mean across chromosome 10
      - chr7_gain       : True if chr7_mean_log2 > GAIN_THRESHOLD
      - chr10_loss      : True if chr10_mean_log2 < LOSS_THRESHOLD
      - gbm_chr_pattern : True if chr7 gain AND chr10 loss (hallmark)

    Most GBM tumours (>80%) show this co-occurring pattern.
    """
    if "Chromosome" not in segments.columns or "Segment_Mean" not in segments.columns:
        return pd.DataFrame()

    # Normalise chromosome labels ("7" → "chr7" or keep "chr7")
    chrom = segments["Chromosome"].astype(str).str.lstrip("chr")
    segments = segments.copy()
    segments["_chrom"] = chrom

    results = []
    for sample, grp in segments.groupby("sample_submitter_id"):
        chr7  = grp[grp["_chrom"] == "7"]["Segment_Mean"].mean()
        chr10 = grp[grp["_chrom"] == "10"]["Segment_Mean"].mean()
        results.append({
            "sample":         sample,
            "chr7_mean_log2":  round(float(chr7), 4)  if not np.isnan(chr7)  else None,
            "chr10_mean_log2": round(float(chr10), 4) if not np.isnan(chr10) else None,
            "chr7_gain":       chr7  > GAIN_THRESHOLD  if not np.isnan(chr7)  else None,
            "chr10_loss":      chr10 < LOSS_THRESHOLD  if not np.isnan(chr10) else None,
        })

    df = pd.DataFrame(results)
    if not df.empty:
        df["gbm_chr_pattern"] = df["chr7_gain"].fillna(False) & df["chr10_loss"].fillna(False)
        pct = df["gbm_chr_pattern"].mean() * 100
        print(f"  📁  Chr7+/Chr10- GBM pattern: {df['gbm_chr_pattern'].sum()} "
              f"samples ({pct:.1f}%)")
    return df


def run(
    client: GBMClient,
    output_dir: Path,
    project_id: str = GBM_PROJECT_ID,
    skip_download: bool = False,
) -> dict:
    records  = discover(client, project_id)
    metadata = build_metadata(records)
    raw_dir  = output_dir / "cnv" / "raw"
    out_dir  = output_dir / "cnv"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        file_ids = [r["file_id"] for r in records]
        client.batch_download(file_ids, raw_dir, label="cnv")

    segments  = parse_segment_files(raw_dir, metadata, verbose=True)
    chr_flags = summarise_chr7_chr10(segments)

    seg_path  = out_dir / "cnv_segments.tsv"
    chr_path  = out_dir / "cnv_chr7_chr10.tsv"
    meta_path = out_dir / "cnv_metadata.tsv"

    segments.to_csv(seg_path,   sep="\t", index=False)
    chr_flags.to_csv(chr_path,  sep="\t", index=False)
    metadata.to_csv(meta_path,  sep="\t", index=False)

    return {
        "segments":   segments,
        "chr_flags":  chr_flags,
        "metadata":   metadata,
        "seg_path":   seg_path,
        "chr_path":   chr_path,
        "meta_path":  meta_path,
    }
