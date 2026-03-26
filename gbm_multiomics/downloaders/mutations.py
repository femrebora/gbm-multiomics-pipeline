"""
mutations.py — TCGA-GBM somatic mutation (MAF) downloader.

Downloads open-access Masked Somatic Mutation MAF files (WXS) and
produces:
  - Long-format mutation table (all variants)
  - Per-sample driver gene summary (presence/absence matrix)
  - IDH1/IDH2 mutation status per sample (critical GBM classifier)

MAF column glossary (GDC Masked Somatic Mutation)
---------------------------------------------------
  Hugo_Symbol             gene name
  Variant_Classification  Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, etc.
  Variant_Type            SNP, DNP, INS, DEL
  HGVSc                   coding DNA change
  HGVSp_Short             protein change (e.g. p.R132H)
  Tumor_Sample_Barcode    TCGA sample barcode (matches RNA-seq metadata)
  t_alt_count             tumour alt allele depth
  t_ref_count             tumour ref allele depth
  n_alt_count             normal alt allele depth
  FILTER                  PASS = high-confidence variant

Output files
------------
  mutations/raw/                    extracted MAF files
  mutations/mutations_all.tsv       all variants (long format)
  mutations/mutations_drivers.tsv   driver gene presence/absence matrix
  mutations/idh_status.tsv          IDH1/IDH2 mutation status per sample
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gbm_multiomics.client import GBMClient, GDCError
from gbm_multiomics.constants import MUTATIONS_FILTERS, GBM_PROJECT_ID, GBM_DRIVER_GENES

# MAF columns to keep (reduce file size; add more as needed)
MAF_COLUMNS_KEEP = [
    "Hugo_Symbol",
    "Chromosome", "Start_Position", "End_Position",
    "Strand",
    "Variant_Classification", "Variant_Type",
    "Reference_Allele", "Tumor_Seq_Allele1", "Tumor_Seq_Allele2",
    "Tumor_Sample_Barcode",
    "HGVSc", "HGVSp_Short",
    "t_depth", "t_ref_count", "t_alt_count",
    "n_depth", "n_ref_count", "n_alt_count",
    "FILTER",
    "IMPACT",           # HIGH / MODERATE / LOW / MODIFIER (VEP)
    "BIOTYPE",
]


def discover(client: GBMClient, project_id: str = GBM_PROJECT_ID) -> list[dict]:
    print(f"  🔍  Discovering somatic mutation MAF files for {project_id}...")
    records = client.discover_files(project_id, MUTATIONS_FILTERS)
    print(f"  ✅  {len(records)} MAF files found.")
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


def parse_maf_files(
    raw_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Parse all MAF files into a single long-format DataFrame.

    Keeps only PASS-filtered variants and a curated set of columns.
    """
    maf_files = list(raw_dir.rglob("*.maf")) + list(raw_dir.rglob("*.maf.gz"))
    if verbose:
        print(f"  🔧  Parsing {len(maf_files)} MAF files...")

    frames: list[pd.DataFrame] = []
    skipped = 0

    for maf in maf_files:
        try:
            # MAF files start with comment lines (#), skip them
            df = pd.read_csv(
                maf, sep="\t", comment="#",
                low_memory=False, dtype=str,
            )
            # Keep only PASS variants (masked MAFs are already filtered but double-check)
            if "FILTER" in df.columns:
                df = df[df["FILTER"].str.upper() == "PASS"].copy()

            # Retain only known columns (silently drop missing ones)
            keep = [c for c in MAF_COLUMNS_KEEP if c in df.columns]
            frames.append(df[keep])
        except Exception:
            skipped += 1
            continue

    if not frames:
        raise GDCError(
            "No valid MAF files could be parsed.",
            fix=f"Check {raw_dir} for extracted .maf files.",
            step="mutations parse",
        )

    mutations = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"  ✅  {len(mutations):,} PASS variants across "
              f"{mutations['Tumor_Sample_Barcode'].nunique()} samples "
              f"({skipped} files skipped).")
    return mutations


def build_driver_matrix(mutations: pd.DataFrame) -> pd.DataFrame:
    """
    Build a driver gene × sample presence/absence matrix (1 = mutated).

    Rows: GBM driver genes, Columns: TCGA sample barcodes.
    """
    driver_muts = mutations[mutations["Hugo_Symbol"].isin(GBM_DRIVER_GENES)].copy()

    if driver_muts.empty:
        return pd.DataFrame(index=list(GBM_DRIVER_GENES))

    matrix = (
        driver_muts
        .groupby(["Hugo_Symbol", "Tumor_Sample_Barcode"])
        .size()
        .unstack(fill_value=0)
        .clip(upper=1)   # presence/absence
    )
    # Ensure all driver genes are represented
    for gene in GBM_DRIVER_GENES:
        if gene not in matrix.index:
            matrix.loc[gene] = 0

    return matrix.loc[list(GBM_DRIVER_GENES)].sort_index(axis=1)


def extract_idh_status(mutations: pd.DataFrame) -> pd.DataFrame:
    """
    Extract IDH1/IDH2 mutation status per sample.

    Returns DataFrame: sample | IDH1_mutated | IDH1_variant |
                                IDH2_mutated | IDH2_variant | IDH_status
    Where IDH_status is: "IDH_mutant" / "IDH_wildtype"

    IDH1 R132H is the canonical GBM hotspot (p.Arg132His).
    """
    idh_genes = {"IDH1", "IDH2"}
    idh_muts = mutations[mutations["Hugo_Symbol"].isin(idh_genes)].copy()

    # Collect all samples seen in the mutation file
    all_samples = mutations["Tumor_Sample_Barcode"].unique()

    rows = []
    for sample in all_samples:
        sample_idh = idh_muts[idh_muts["Tumor_Sample_Barcode"] == sample]

        idh1_row = sample_idh[sample_idh["Hugo_Symbol"] == "IDH1"]
        idh2_row = sample_idh[sample_idh["Hugo_Symbol"] == "IDH2"]

        idh1_mut  = len(idh1_row) > 0
        idh2_mut  = len(idh2_row) > 0
        idh1_var  = idh1_row["HGVSp_Short"].iloc[0] if idh1_mut else ""
        idh2_var  = idh2_row["HGVSp_Short"].iloc[0] if idh2_mut else ""

        rows.append({
            "sample":        sample,
            "IDH1_mutated":  idh1_mut,
            "IDH1_variant":  idh1_var,
            "IDH2_mutated":  idh2_mut,
            "IDH2_variant":  idh2_var,
            "IDH_status":   "IDH_mutant" if (idh1_mut or idh2_mut) else "IDH_wildtype",
        })

    return pd.DataFrame(rows)


def run(
    client: GBMClient,
    output_dir: Path,
    project_id: str = GBM_PROJECT_ID,
    skip_download: bool = False,
) -> dict:
    records  = discover(client, project_id)
    metadata = build_metadata(records)
    raw_dir  = output_dir / "mutations" / "raw"
    out_dir  = output_dir / "mutations"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not skip_download:
        file_ids = [r["file_id"] for r in records]
        client.batch_download(file_ids, raw_dir, label="mutations")

    mutations   = parse_maf_files(raw_dir, verbose=True)
    driver_mat  = build_driver_matrix(mutations)
    idh_status  = extract_idh_status(mutations)

    mut_path    = out_dir / "mutations_all.tsv"
    driver_path = out_dir / "mutations_drivers.tsv"
    idh_path    = out_dir / "idh_status.tsv"
    meta_path   = out_dir / "mutations_metadata.tsv"

    mutations.to_csv(mut_path,    sep="\t", index=False)
    driver_mat.to_csv(driver_path, sep="\t")
    idh_status.to_csv(idh_path,   sep="\t", index=False)
    metadata.to_csv(meta_path,    sep="\t", index=False)

    idh_wt  = (idh_status["IDH_status"] == "IDH_wildtype").sum()
    idh_mut = (idh_status["IDH_status"] == "IDH_mutant").sum()
    print(f"  📁  IDH wildtype: {idh_wt}  |  IDH mutant: {idh_mut}")

    return {
        "mutations":    mutations,
        "driver_matrix": driver_mat,
        "idh_status":   idh_status,
        "metadata":     metadata,
        "mut_path":     mut_path,
        "driver_path":  driver_path,
        "idh_path":     idh_path,
        "meta_path":    meta_path,
    }
