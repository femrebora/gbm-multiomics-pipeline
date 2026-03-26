"""
cli.py — Command-line interface for gbm-multiomics.

Commands
--------
gbm-download  — Download one or more GBM data types from TCGA/GDC
gbm-analyse   — Run downstream analysis on downloaded data

Download usage
--------------
  gbm-download --data-type rna-seq
  gbm-download --data-type rna-seq methylation mutations
  gbm-download --data-type all
  gbm-download --data-type rna-seq --dry-run
  gbm-download --data-type rna-seq --output ~/gbm_data
  gbm-download --data-type mutations --token ~/gdc-user-token.txt
  gbm-download --data-type rna-seq --fresh

Analysis usage
--------------
  gbm-analyse --analysis de --data-dir ~/gbm_data
  gbm-analyse --analysis de pathway --condition IDH_status --reference IDH_wildtype
  gbm-analyse --analysis survival --endpoint OS --group IDH_status
  gbm-analyse --analysis subtype --data-dir ~/gbm_data
  gbm-analyse --analysis all --data-dir ~/gbm_data
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from gbm_multiomics.client import GDCError
from gbm_multiomics.constants import ALL_DATA_TYPES, GBM_PROJECT_ID


# ── Shared defaults ────────────────────────────────────────────────────────────
DEFAULT_OUTPUT = Path.home() / "gbm_multiomics_data"
ANALYSIS_CHOICES = ("de", "pathway", "survival", "subtype", "all")


# ── Download parser ────────────────────────────────────────────────────────────

def _build_download_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gbm-download",
        description=(
            "Download TCGA-GBM multiomics data from the NCI GDC portal.\n\n"
            "Data types: " + ", ".join(ALL_DATA_TYPES)
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--data-type", "-d",
        nargs="+",
        metavar="TYPE",
        choices=list(ALL_DATA_TYPES) + ["all"],
        required=True,
        help=(
            "One or more data types to download. Use 'all' for every type.\n"
            f"Choices: {', '.join(ALL_DATA_TYPES)}, all"
        ),
    )
    p.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--token", "-t",
        metavar="TOKEN_FILE",
        help="Path to GDC authentication token file (for controlled-access data)",
    )
    p.add_argument(
        "--project",
        metavar="PROJECT_ID",
        default=GBM_PROJECT_ID,
        help=f"GDC project ID (default: {GBM_PROJECT_ID})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover files and report counts without downloading",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and start from scratch",
    )
    p.add_argument(
        "--no-cdr",
        action="store_true",
        help="Skip PanCanAtlas CDR annotation download (RNA-seq only)",
    )
    p.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return p


# ── Analysis parser ────────────────────────────────────────────────────────────

def _build_analyse_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gbm-analyse",
        description="Run downstream analysis on downloaded TCGA-GBM data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--analysis", "-a",
        nargs="+",
        choices=ANALYSIS_CHOICES,
        required=True,
        help="Analysis type(s) to run. 'all' runs every analysis.",
    )
    p.add_argument(
        "--data-dir", "-d",
        metavar="DIR",
        default=str(DEFAULT_OUTPUT),
        help=f"Directory containing downloaded data (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--output", "-o",
        metavar="DIR",
        default=None,
        help="Output directory for analysis results (default: <data-dir>/analysis)",
    )
    p.add_argument(
        "--condition",
        metavar="COLUMN",
        default="is_tumor",
        help="Metadata column defining comparison groups for DE (default: is_tumor)",
    )
    p.add_argument(
        "--reference",
        metavar="VALUE",
        default="False",
        help="Reference level for DE comparison (default: False = normal samples)",
    )
    p.add_argument(
        "--endpoint",
        choices=("OS", "PFI", "DSS"),
        default="OS",
        help="Survival endpoint to use (default: OS = Overall Survival)",
    )
    p.add_argument(
        "--group",
        metavar="COLUMN",
        default="IDH_status",
        help="Metadata column to stratify KM curves by (default: IDH_status)",
    )
    p.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return p


# ── Download runner ────────────────────────────────────────────────────────────

def run_download(args: argparse.Namespace) -> None:
    """Execute download pipeline for the requested data types."""
    import pandas as pd

    from gbm_multiomics.client import GBMClient
    from gbm_multiomics.checkpoint import Checkpoint
    from gbm_multiomics.downloaders import rna_seq, methylation, mutations, cnv, mirna

    # Resolve data types
    requested: list[str] = (
        list(ALL_DATA_TYPES)
        if "all" in args.data_type
        else args.data_type
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cp = Checkpoint(output_dir)
    if args.fresh:
        cp.reset_all()
        print("  🔄  Starting fresh — checkpoint cleared.")

    # Build client
    client = (
        GBMClient.from_file(args.token)
        if args.token
        else GBMClient()
    )

    print(f"\n  🌐  Checking GDC API connectivity...")
    if not client.check_connectivity():
        print("  ⚠   Cannot reach GDC API. Check your internet connection.")
        sys.exit(1)
    print(f"  ✅  GDC API reachable.")
    print(f"  📂  Output directory: {output_dir}")
    print(f"  🧬  Project: {args.project}")
    print(f"  📦  Data types: {', '.join(requested)}\n")

    # --- Dispatch per data type ---
    dispatch = {
        "rna-seq":     rna_seq,
        "methylation": methylation,
        "mutations":   mutations,
        "cnv":         cnv,
        "mirna":       mirna,
    }

    for dtype in requested:
        step_key = f"downloaded_{dtype.replace('-', '_')}"
        print(f"\n{'─' * 60}")
        print(f"  Data type: {dtype.upper()}")
        print(f"{'─' * 60}")

        if cp.is_done(step_key) and not args.fresh:
            saved = cp.get(step_key)
            print(f"  ✅  {dtype}: already downloaded "
                  f"({saved.get('n_files', '?')} files).")
            continue

        module = dispatch[dtype]

        if args.dry_run:
            records = module.discover(client, args.project)
            metadata = module.build_metadata(records)
            total_size_mb = sum(
                r.get("file_size", 0) for r in records
            ) / (1024 ** 2)
            print(f"\n  ℹ   Dry run — {len(records)} files "
                  f"({total_size_mb:.0f} MB estimated). No download performed.")
            continue

        try:
            result = module.run(
                client     = client,
                output_dir = output_dir,
                project_id = args.project,
            )
            cp.save(step_key, {
                "n_files": len(result.get("metadata", pd.DataFrame())),
                "output_dir": str(output_dir / dtype.replace("-", "_")),
            })
        except GDCError as exc:
            print(f"\n{exc.formatted()}")
            print(f"  ⚠   Skipping {dtype} due to error. Other types will continue.")
            continue

    if not args.dry_run:
        print(f"\n\n  🎉  Download complete! Data saved to: {output_dir}")
        print("  Run `gbm-analyse --help` for downstream analysis options.")


# ── Analysis runner ────────────────────────────────────────────────────────────

def run_analyse(args: argparse.Namespace) -> None:
    """Execute analysis pipeline for the requested analysis types."""
    import pandas as pd

    from gbm_multiomics.analysis import (
        differential_expression as de_mod,
        pathway_enrichment as pe_mod,
        survival as surv_mod,
        subtype as sub_mod,
    )

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output) if args.output else data_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    requested: list[str] = (
        [a for a in ANALYSIS_CHOICES if a != "all"]
        if "all" in args.analysis
        else args.analysis
    )

    print(f"\n  📊  GBM Analysis Pipeline")
    print(f"  Data directory:   {data_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Analyses:         {', '.join(requested)}\n")

    # ── Load RNA-seq data if available ────────────────────────────────────────
    counts_path = data_dir / "rna_seq" / "rna_seq_counts.tsv"
    meta_path   = data_dir / "rna_seq" / "rna_seq_metadata.tsv"
    counts = metadata = None

    if counts_path.exists() and meta_path.exists():
        print("  📂  Loading RNA-seq count matrix...")
        counts   = pd.read_csv(counts_path,   sep="\t", index_col=0)
        metadata = pd.read_csv(meta_path,     sep="\t", dtype=str)
        print(f"  ✅  {counts.shape[0]} genes × {counts.shape[1]} samples loaded.")
    else:
        if any(a in requested for a in ("de", "pathway", "subtype")):
            print("  ⚠   RNA-seq data not found. Run `gbm-download --data-type rna-seq` first.")

    # ── Load clinical/IDH data if available ───────────────────────────────────
    idh_path = data_dir / "mutations" / "idh_status.tsv"
    idh_df   = None
    if idh_path.exists():
        idh_df = pd.read_csv(idh_path, sep="\t", dtype=str)

    # ── Differential expression ───────────────────────────────────────────────
    if "de" in requested and counts is not None and metadata is not None:
        print(f"\n{'─' * 60}")
        print("  Differential Expression Analysis")
        print(f"{'─' * 60}")
        de_out = output_dir / "differential_expression"

        # Write R script regardless (always useful)
        de_mod.write_r_deseq2_script(
            counts_path  = counts_path,
            coldata_path = meta_path,
            condition_col = args.condition,
            reference    = args.reference,
            output_dir   = de_out,
        )

        # Run pydeseq2 if available
        try:
            de_results = de_mod.run_deseq2_py(
                counts       = counts,
                metadata     = metadata,
                condition_col = args.condition,
                reference    = args.reference,
                output_dir   = de_out,
            )
        except ImportError as exc:
            print(f"  ℹ   {exc}")
            print("  ℹ   Use the generated R script at {de_out}/deseq2_run.R instead.")
            de_results = None

    # ── Pathway enrichment ────────────────────────────────────────────────────
    if "pathway" in requested:
        print(f"\n{'─' * 60}")
        print("  Pathway Enrichment Analysis")
        print(f"{'─' * 60}")
        pe_out = output_dir / "pathway_enrichment"

        de_results_path = output_dir / "differential_expression" / f"de_results_{args.condition}.tsv"
        if de_results_path.exists():
            de_df = pd.read_csv(de_results_path, sep="\t", index_col=0)
            sig_genes = de_mod.filter_significant(de_df)
            up_genes   = sig_genes[sig_genes["direction"] == "UP"].index.tolist()
            down_genes = sig_genes[sig_genes["direction"] == "DOWN"].index.tolist()

            print(f"  🧬  Running ORA on {len(up_genes)} upregulated genes...")
            pe_mod.run_gbm_custom_ora(up_genes,   output_dir=pe_out / "up_custom")
            pe_mod.run_gbm_custom_ora(down_genes, output_dir=pe_out / "down_custom")

            # Try MSigDB Hallmarks (requires internet + gseapy)
            try:
                pe_mod.run_ora(
                    gene_list  = up_genes,
                    gene_sets  = "MSigDB_Hallmark_2020",
                    output_dir = pe_out / "up_hallmarks",
                )
            except Exception as exc:
                print(f"  ⚠   MSigDB ORA skipped: {exc}")
        else:
            print("  ⚠   DE results not found. Run `gbm-analyse --analysis de` first.")

    # ── Survival analysis ─────────────────────────────────────────────────────
    if "survival" in requested:
        print(f"\n{'─' * 60}")
        print("  Survival Analysis")
        print(f"{'─' * 60}")
        surv_out = output_dir / "survival"
        duration_col = f"cdr_{args.endpoint}.time"
        event_col    = f"cdr_{args.endpoint}"

        # Find a merged clinical file (CDR annotations)
        cdr_path = data_dir / "rna_seq" / f"TCGA-GBM_full_merged_with_cdr.tsv"
        if not cdr_path.exists():
            # Try any merged file
            cdr_files = list((data_dir / "rna_seq").glob("*cdr*.tsv"))
            cdr_path = cdr_files[0] if cdr_files else None

        if cdr_path and cdr_path.exists():
            clin_df = pd.read_csv(cdr_path, sep="\t", dtype=str)
            for col in [duration_col, event_col]:
                if col in clin_df.columns:
                    clin_df[col] = pd.to_numeric(clin_df[col], errors="coerce")

            surv_df = surv_mod.prepare_survival_data(
                clin_df, molecular=idh_df,
                duration_col=duration_col, event_col=event_col,
            )

            if args.group in surv_df.columns:
                surv_mod.kaplan_meier(
                    surv_df,
                    duration_col = duration_col,
                    event_col    = event_col,
                    group_col    = args.group,
                    output_dir   = surv_out,
                )
                surv_mod.cox_univariate(
                    surv_df,
                    duration_col = duration_col,
                    event_col    = event_col,
                    covariates   = [args.group],
                    output_dir   = surv_out,
                )
            else:
                print(f"  ⚠   Column '{args.group}' not found in merged data.")
        else:
            print("  ⚠   CDR-merged data not found. Download RNA-seq data first (includes CDR).")

    # ── Subtype classification ────────────────────────────────────────────────
    if "subtype" in requested and counts is not None:
        print(f"\n{'─' * 60}")
        print("  GBM Subtype Classification")
        print(f"{'─' * 60}")
        sub_out = output_dir / "subtype"

        # Convert raw counts to log2(CPM+1) for centroid correlation
        print("  🔧  Normalising counts to log2(CPM+1)...")
        cpm = counts.div(counts.sum(axis=0) / 1e6)
        import numpy as np
        log2_cpm = np.log2(cpm + 1)

        sub_mod.classify_centroids(log2_cpm, output_dir=sub_out)

        # WHO 2021 classification if IDH status is available
        if idh_df is not None:
            who = sub_mod.who_2021_classify(idh_df)
            who.to_csv(sub_out / "who_2021_classification.tsv", sep="\t", index=False)
            print(f"  📄  WHO 2021 provisional classification written.")

    print(f"\n\n  🎉  Analysis complete! Results saved to: {output_dir}")


# ── Entry points ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: gbm-download"""
    parser = _build_download_parser()
    args   = parser.parse_args()
    try:
        run_download(args)
    except GDCError as exc:
        print(f"\n{exc.formatted()}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n  ⛔  Cancelled. Re-run to resume (checkpoint saved).")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ❌  Unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


def main_analyse() -> None:
    """Entry point: gbm-analyse"""
    parser = _build_analyse_parser()
    args   = parser.parse_args()
    try:
        run_analyse(args)
    except KeyboardInterrupt:
        print("\n\n  ⛔  Cancelled.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ❌  Unexpected error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
