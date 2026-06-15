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

import numpy as np
import pandas as pd

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


# ── Pipeline parser (new unified entry point) ─────────────────────────────────

def _build_pipeline_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gbm-pipeline",
        description="GBM Multi-Omics Prognostic Biomarker Discovery Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", help="Pipeline step")

    # ── download ──────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Download TCGA-GBM data from GDC")
    dl.add_argument("--data-type", "-d", nargs="+",
                    choices=list(ALL_DATA_TYPES) + ["all"],
                    default=["rna-seq"],
                    help="Data types to download")
    dl.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT))
    dl.add_argument("--dry-run", action="store_true")
    dl.add_argument("--fresh", action="store_true")
    dl.add_argument("--token")

    # ── preprocess ─────────────────────────────────────────────────────────
    prep = sub.add_parser("preprocess", help="QC and normalize downloaded data")
    prep.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    prep.add_argument("--output", "-o", default=None)
    prep.add_argument("--method", choices=["vst", "tpm", "cpm"], default="vst")
    prep.add_argument("--protein-coding-only", action="store_true", default=True)
    prep.add_argument("--skip-annotation", action="store_true",
                      help="Skip gene symbol annotation")

    # ── analyze ────────────────────────────────────────────────────────────
    ana = sub.add_parser("analyze", help="Run analysis modules")
    ana.add_argument("--module", "-m", nargs="+",
                     choices=["de", "prognostic", "pathway", "survival", "subtype",
                              "multiomics", "immune", "network", "all"],
                     default=["all"])
    ana.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    ana.add_argument("--output", "-o", default=None)
    ana.add_argument("--config", "-c", default=None,
                     help="Path to YAML config file")

    # ── report ─────────────────────────────────────────────────────────────
    rep = sub.add_parser("report", help="Generate thesis PDF report")
    rep.add_argument("--results-dir", "-r", default="results")
    rep.add_argument("--output", "-o", default=None)
    rep.add_argument("--include-code", action="store_true")

    # ── figures ────────────────────────────────────────────────────────────
    fig = sub.add_parser("figures", help="Regenerate all publication figures")
    fig.add_argument("--results-dir", "-r", default="results")
    fig.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))

    # ── gene-focus (gene spotlight for thesis) ──────────────────────────────
    gf = sub.add_parser("gene-focus", help="Full gene focus report for thesis genes")
    gf.add_argument("--genes", "-g", required=True,
                    help="Comma-separated gene symbols (e.g., EGFR,PTEN,TP53)")
    gf.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    gf.add_argument("--output", "-o", default=None)

    gl = sub.add_parser("gene-lookup", help="Single-gene lookup across all omics")
    gl.add_argument("--gene", "-g", required=True, help="Gene symbol to query")
    gl.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))

    gs = sub.add_parser("gene-spotlight", help="Generate per-gene thesis spotlight figures")
    gs.add_argument("--genes", "-g", required=True,
                    help="Comma-separated gene symbols")
    gs.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    gs.add_argument("--output", "-o", default=None)
    gs.add_argument("--group-by", default="is_tumor",
                    help="Group column for expression violin")

    gr = sub.add_parser("gene-rank", help="Rank genes against genome-wide prognostic results")
    gr.add_argument("--genes", "-g", required=True,
                    help="Comma-separated gene symbols")
    gr.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    gr.add_argument("--output", "-o", default=None)

    # ── run (full pipeline) ────────────────────────────────────────────────
    run = sub.add_parser("run", help="Run the complete pipeline")
    run.add_argument("--config", "-c", default=None,
                     help="Path to YAML config file")
    run.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    run.add_argument("--output", "-o", default=None)
    run.add_argument("--skip-download", action="store_true")
    run.add_argument("--skip-report", action="store_true")

    return p


def main_pipeline() -> None:
    """Entry point: gbm-pipeline (unified pipeline CLI)."""
    parser = _build_pipeline_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    try:
        if args.command == "download":
            # Forward to existing download runner
            dl_args = argparse.Namespace(
                data_type=args.data_type,
                output=args.output,
                dry_run=args.dry_run if hasattr(args, "dry_run") else False,
                fresh=args.fresh if hasattr(args, "fresh") else False,
                token=args.token if hasattr(args, "token") else None,
                project=GBM_PROJECT_ID,
            )
            run_download(dl_args)

        elif args.command == "preprocess":
            _run_preprocess(args)

        elif args.command == "analyze":
            _run_analyze_pipeline(args)

        elif args.command == "report":
            _run_report(args)

        elif args.command == "figures":
            _run_regenerate_figures(args)

        elif args.command == "gene-focus":
            _run_gene_focus(args)

        elif args.command == "gene-lookup":
            _run_gene_lookup(args)

        elif args.command == "gene-spotlight":
            _run_gene_spotlight(args)

        elif args.command == "gene-rank":
            _run_gene_rank(args)

        elif args.command == "run":
            _run_full_pipeline(args)

    except KeyboardInterrupt:
        print("\n\n  ⛔  Cancelled.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  ❌  Error: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)


# ── Pipeline step runners ─────────────────────────────────────────────────────

def _run_preprocess(args: argparse.Namespace) -> None:
    """Run QC and normalization on downloaded RNA-seq data."""
    from gbm_multiomics.preprocessing import (
        qc_report, annotate_genes, normalize_vst, normalize_cpm,
        filter_protein_coding, filter_low_expression,
    )

    data_dir = Path(args.data_dir)
    counts_path = data_dir / "rna_seq" / "rna_seq_counts.tsv"
    meta_path = data_dir / "rna_seq" / "rna_seq_metadata.tsv"
    out_dir = Path(args.output) if args.output else data_dir / "preprocessed"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not counts_path.exists():
        print(f"  ❌  Count matrix not found at {counts_path}")
        print("  Run `gbm-pipeline download --data-type rna-seq` first.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  GBM Preprocessing Pipeline")
    print(f"{'='*60}\n")

    # Load data
    counts = pd.read_csv(counts_path, sep="\t", index_col=0)
    metadata = pd.read_csv(meta_path, sep="\t") if meta_path.exists() else None
    print(f"  📂  Loaded: {counts.shape[0]} genes × {counts.shape[1]} samples.")

    # Filter low expression
    counts = filter_low_expression(counts, min_count=10, min_samples=0.2)

    # Annotate genes
    if not args.skip_annotation:
        counts = annotate_genes(counts, cache_dir=out_dir)

    # Protein-coding filter
    if args.protein_coding_only:
        counts = filter_protein_coding(counts)

    # Normalize
    if args.method == "vst":
        norm = normalize_vst(counts)
    elif args.method == "tpm":
        norm = normalize_cpm(counts, log_transform=True)  # fallback
    else:
        norm = normalize_cpm(counts, log_transform=True)

    # Save normalized data
    norm.to_csv(out_dir / "normalized_expression.tsv", sep="\t")
    print(f"  📄  Normalized expression: {norm.shape[0]} genes × {norm.shape[1]} samples → {out_dir}")

    # QC report
    qc_report(counts, metadata=metadata, output_dir=out_dir / "qc")

    print(f"\n  ✅  Preprocessing complete. Output: {out_dir}")


def _run_analyze_pipeline(args: argparse.Namespace) -> None:
    """Run specified analysis modules."""
    from pathlib import Path

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else data_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    modules = args.module
    if "all" in modules:
        modules = ["de", "prognostic", "pathway", "survival", "subtype",
                    "multiomics", "immune", "network"]

    # Load normalized expression
    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    counts_path = data_dir / "rna_seq" / "rna_seq_counts.tsv"

    if not norm_path.exists() and not counts_path.exists():
        print("  ❌  No expression data found.")
        print("  Run `gbm-pipeline preprocess` first.")
        return

    # Forward to existing gbm-analyse for backward compat modules
    analyse_args = argparse.Namespace(
        analysis=["all" if m in ["de", "pathway", "survival", "subtype"] else m
                   for m in modules if m in ["de", "pathway", "survival", "subtype"]],
        data_dir=str(data_dir),
        output=str(out_dir),
        condition="is_tumor",
        reference="False",
        endpoint="OS",
        group="IDH_status",
    )

    if any(m in modules for m in ["de", "pathway", "survival", "subtype"]):
        filtered_modules = [m for m in modules if m in ["de", "pathway", "survival", "subtype"]]
        analyse_args.analysis = filtered_modules if "all" not in filtered_modules else ["all"]
        run_analyse(analyse_args)

    # New modules: prognostic, multiomics, immune, network
    for module in modules:
        if module == "prognostic":
            _run_prognostic_module(data_dir, out_dir)
        elif module == "multiomics":
            _run_multiomics_module(data_dir, out_dir)
        elif module == "immune":
            _run_immune_module(data_dir, out_dir)
        elif module == "network":
            _run_network_module(data_dir, out_dir)


def _run_prognostic_module(data_dir: Path, out_dir: Path) -> None:
    """Run the full prognostic biomarker discovery pipeline."""
    print(f"\n{'─'*60}")
    print("  Prognostic Biomarker Discovery")
    print(f"{'─'*60}")

    from gbm_multiomics.analysis.prognostic import run_prognostic_pipeline
    from gbm_multiomics.preprocessing.clinical import build_unified_metadata
    from gbm_multiomics.visualization.forest import forest_plot
    from gbm_multiomics.visualization.survival import km_plot

    # Load normalized expression
    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    if not norm_path.exists():
        from gbm_multiomics.preprocessing.normalization import normalize_cpm
        counts = pd.read_csv(data_dir / "rna_seq" / "rna_seq_counts.tsv",
                             sep="\t", index_col=0)
        expr = normalize_cpm(counts, log_transform=True)
    else:
        expr = pd.read_csv(norm_path, sep="\t", index_col=0)

    # Build unified clinical metadata
    clinical = build_unified_metadata(data_dir)

    # Run pipeline
    result = run_prognostic_pipeline(
        expr=expr,
        clinical=clinical,
        output_dir=out_dir / "prognostic",
    )

    # Generate figures
    if "multivariate" in result:
        mv = result["multivariate"]
        if "summary" in mv and not mv["summary"].empty:
            forest_plot(
                pd.DataFrame({
                    "covariate": mv["summary"].index.tolist(),
                    "HR": np.exp(mv["summary"]["coef"]).tolist(),
                    "HR_lower_95": np.exp(mv["summary"]["coef lower 95%"]).tolist(),
                    "HR_upper_95": np.exp(mv["summary"]["coef upper 95%"]).tolist(),
                    "p_value": mv["summary"]["p"].tolist(),
                }),
                title="GBM Multivariate Cox — Forest Plot",
                output_dir=out_dir / "prognostic",
            )

    # KM for risk groups
    if "risk_score" in result and "model_df" in result:
        from gbm_multiomics.analysis.survival import kaplan_meier
        risk_df = result["model_df"].copy()
        risk_df["risk_score"] = result["risk_score"]["risk_score"].values[:len(risk_df)]
        median = risk_df["risk_score"].median()
        risk_df["risk_group"] = np.where(
            risk_df["risk_score"] >= median, "High Risk", "Low Risk"
        )
        kaplan_meier(
            risk_df, duration_col="cdr_OS.time", event_col="cdr_OS",
            group_col="risk_group", title="GBM Prognostic Risk Score",
            output_dir=out_dir / "prognostic",
        )

    print("  ✅  Prognostic analysis complete.")


def _run_multiomics_module(data_dir: Path, out_dir: Path) -> None:
    """Run multi-omics integration analysis."""
    print(f"\n{'─'*60}")
    print("  Multi-Omics Integration")
    print(f"{'─'*60}")

    # Load available omics data
    rna_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    if not rna_path.exists():
        print("  ⚠   RNA-seq data not found. Skipping multi-omics.")
        return

    rna = pd.read_csv(rna_path, sep="\t", index_col=0)

    # Try loading other omics
    views = {"rna": rna}
    cnv, methylation, mutations, mirna = None, None, None, None

    cnv_path = data_dir / "cnv" / "cnv_segments.tsv"
    if cnv_path.exists():
        cnv = pd.read_csv(cnv_path, sep="\t")
        print(f"  📂  CNV data loaded: {len(cnv)} segments.")

    meth_path = data_dir / "methylation" / "methylation_beta.tsv"
    if meth_path.exists():
        methylation = pd.read_csv(meth_path, sep="\t", index_col=0)
        views["methylation"] = methylation

    mut_path = data_dir / "mutations" / "mutations_drivers.tsv"
    if mut_path.exists():
        mutations = pd.read_csv(mut_path, sep="\t", index_col=0)

    # Cross-omics correlation
    from gbm_multiomics.analysis.multiomics import cross_omics_correlation
    corr_results = cross_omics_correlation(
        rna_expr=rna, cnv=cnv, methylation=methylation,
        mutations=mutations, output_dir=out_dir / "multiomics",
    )

    # Correlation heatmap
    if corr_results:
        from gbm_multiomics.visualization.correlation import omics_correlation_heatmap
        omics_correlation_heatmap(
            corr_results,
            output_dir=out_dir / "multiomics",
        )

    print("  ✅  Multi-omics integration complete.")


def _run_immune_module(data_dir: Path, out_dir: Path) -> None:
    """Run immune infiltration analysis."""
    print(f"\n{'─'*60}")
    print("  Immune Infiltration Analysis")
    print(f"{'─'*60}")

    from gbm_multiomics.analysis.immune import estimate_scores, immune_survival_split
    from gbm_multiomics.preprocessing.clinical import build_unified_metadata

    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    if not norm_path.exists():
        print("  ⚠   Normalized expression data not found. Skipping immune analysis.")
        return

    expr = pd.read_csv(norm_path, sep="\t", index_col=0)
    clinical = build_unified_metadata(data_dir)

    # ESTIMATE
    scores = estimate_scores(expr, output_dir=out_dir / "immune")

    # Immune survival
    if not scores.empty and "case_submitter_id" in clinical.columns:
        immune_survival_split(
            scores, clinical,
            score_col="ImmuneScore",
            output_dir=out_dir / "immune",
        )

    print("  ✅  Immune analysis complete.")


def _run_network_module(data_dir: Path, out_dir: Path) -> None:
    """Run PPI network analysis on prognostic genes."""
    print(f"\n{'─'*60}")
    print("  Protein-Protein Interaction Network")
    print(f"{'─'*60}")

    # Load prognostic genes
    prog_path = data_dir / "analysis" / "prognostic" / "prognostic_univariate.tsv"
    if not prog_path.exists():
        print("  ⚠   Prognostic results not found. Run prognostic module first.")
        return

    prog_df = pd.read_csv(prog_path, sep="\t")
    sig_genes = prog_df[prog_df["padj"] < 0.05]["gene"].head(100).tolist()

    from gbm_multiomics.analysis.network import build_ppi_network, identify_hub_genes, detect_modules
    network = build_ppi_network(sig_genes, output_dir=out_dir / "network")

    if not network["edges"].empty:
        hubs = identify_hub_genes(network)
        if network["graph"] is not None:
            detect_modules(network["graph"], output_dir=out_dir / "network")

    print("  ✅  Network analysis complete.")


def _run_report(args: argparse.Namespace) -> None:
    """Generate Quarto thesis report."""
    from gbm_multiomics.report.quarto_generator import generate_report, export_thesis_figures

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output) if args.output else results_dir / "report"

    # Export thesis-ready figures
    export_thesis_figures(results_dir)

    # Generate report
    report_path = generate_report(
        results_dir=results_dir,
        output_dir=output_dir,
        include_code=args.include_code if hasattr(args, "include_code") else False,
    )

    print(f"\n  ✅  Report generated: {report_path}")
    print(f"  ℹ   Compile with: quarto render {report_path} --to pdf")


def _run_regenerate_figures(args: argparse.Namespace) -> None:
    """Regenerate all publication-quality figures from cached results."""
    print(f"\n{'='*60}")
    print("  Regenerating Publication Figures")
    print(f"{'='*60}")

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)

    # This would re-run figure generation from cached analysis results
    # For now, print guidance
    print("\n  ℹ   Run individual analysis modules to regenerate figures:")
    print("       gbm-pipeline analyze --module de")
    print("       gbm-pipeline analyze --module prognostic")
    print("       gbm-pipeline analyze --module multiomics")

    print(f"\n  ✅  Use existing figures in: {results_dir}/figures/")


def _run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the complete pipeline from download to report."""
    print(f"\n{'='*60}")
    print("  GBM Multi-Omics Prognostic Pipeline — FULL RUN")
    print(f"{'='*60}\n")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else data_dir / "analysis"

    steps = []

    # 1. Download
    if not args.skip_download:
        steps.append("download")
        dl_args = argparse.Namespace(
            data_type=["all"], output=str(data_dir),
            dry_run=False, fresh=False, token=None,
            project=GBM_PROJECT_ID,
        )
        run_download(dl_args)

    # 2. Preprocess
    steps.append("preprocess")
    prep_args = argparse.Namespace(
        data_dir=str(data_dir), output=str(data_dir / "preprocessed"),
        method="vst", protein_coding_only=True, skip_annotation=False,
    )
    _run_preprocess(prep_args)

    # 3. Analyze (all modules)
    steps.append("analyze")
    ana_args = argparse.Namespace(
        module=["all"], data_dir=str(data_dir),
        output=str(out_dir), config=args.config,
    )
    _run_analyze_pipeline(ana_args)

    # 4. Report
    if not args.skip_report:
        steps.append("report")
        rep_args = argparse.Namespace(
            results_dir=str(out_dir), output=str(out_dir / "report"),
            include_code=False,
        )
        _run_report(rep_args)

    print(f"\n{'='*60}")
    print(f"  🎉  Full pipeline complete!")
    print(f"  Steps completed: {' → '.join(steps)}")
    print(f"  Data: {data_dir}")
    print(f"  Results: {out_dir}")
    print(f"{'='*60}\n")


# ── Gene focus runners ─────────────────────────────────────────────────────────

def _run_gene_focus(args: argparse.Namespace) -> None:
    """Full gene focus report for thesis genes."""
    from gbm_multiomics.analysis.gene_focus import GeneFocus

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else data_dir / "analysis" / "gene_focus"
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]

    # Load expression
    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    counts_path = data_dir / "rna_seq" / "rna_seq_counts.tsv"
    if norm_path.exists():
        expr = pd.read_csv(norm_path, sep="\t", index_col=0)
    elif counts_path.exists():
        from gbm_multiomics.preprocessing.normalization import normalize_cpm
        counts = pd.read_csv(counts_path, sep="\t", index_col=0)
        expr = normalize_cpm(counts, log_transform=True)
    else:
        print("  ❌  No expression data found. Run `gbm-pipeline preprocess` first.")
        return

    # Load clinical
    from gbm_multiomics.preprocessing.clinical import build_unified_metadata
    clinical = build_unified_metadata(data_dir)

    # Load prognostic results if available
    prog_results = None
    prog_path = data_dir / "analysis" / "prognostic" / "prognostic_univariate.tsv"
    if prog_path.exists():
        prog_results = pd.read_csv(prog_path, sep="\t")

    # Load DepMap if available
    depmap_results = None
    dm_path = data_dir / "analysis" / "prognostic" / "depmap_validation.tsv"
    if dm_path.exists():
        depmap_results = pd.read_csv(dm_path, sep="\t")

    # Build GeneFocus
    gf = GeneFocus(
        genes=genes,
        expr=expr,
        clinical=clinical,
        prognostic_results=prog_results,
        depmap_results=depmap_results,
    )

    # Generate report
    print(f"\n{'='*60}")
    print(f"  Gene Focus Report — {len(genes)} genes")
    print(f"{'='*60}\n")

    # 1. Multi-gene comparison
    print("─" * 40)
    print("  1. Multi-Gene Comparison Report")
    print("─" * 40)
    report = gf.gene_report(output_dir=out_dir)

    # 2. Rank against genome
    print("\n" + "─" * 40)
    print("  2. Gene Ranking vs Genome-Wide Results")
    print("─" * 40)
    ranking = gf.rank_against_genome()
    if not ranking.empty:
        ranking.to_csv(out_dir / "gene_ranking.tsv", sep="\t", index=False)

    # 3. DepMap comparison
    print("\n" + "─" * 40)
    print("  3. DepMap Cross-Validation")
    print("─" * 40)
    depmap_comp = gf.compare_depmap()
    if not depmap_comp.empty:
        depmap_comp.to_csv(out_dir / "gene_depmap_comparison.tsv", sep="\t", index=False)

    # 4. Spotlight figures
    print("\n" + "─" * 40)
    print("  4. Thesis Spotlight Figures")
    print("─" * 40)
    gf.generate_spotlight_figures(output_dir=out_dir / "spotlight")

    print(f"\n  ✅  Gene focus report complete: {out_dir}")


def _run_gene_lookup(args: argparse.Namespace) -> None:
    """Single-gene lookup printed to terminal."""
    import json
    from gbm_multiomics.analysis.gene_focus import GeneFocus

    data_dir = Path(args.data_dir)
    gene = args.gene.strip()

    # Load minimal data
    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    counts_path = data_dir / "rna_seq" / "rna_seq_counts.tsv"
    if norm_path.exists():
        expr = pd.read_csv(norm_path, sep="\t", index_col=0)
    elif counts_path.exists():
        from gbm_multiomics.preprocessing.normalization import normalize_cpm
        counts = pd.read_csv(counts_path, sep="\t", index_col=0)
        expr = normalize_cpm(counts, log_transform=True)
    else:
        print(f"  ❌  No expression data found.")
        return

    from gbm_multiomics.preprocessing.clinical import build_unified_metadata
    clinical = build_unified_metadata(data_dir)

    prog_results = None
    prog_path = data_dir / "analysis" / "prognostic" / "prognostic_univariate.tsv"
    if prog_path.exists():
        prog_results = pd.read_csv(prog_path, sep="\t")

    gf = GeneFocus(genes=[gene], expr=expr, clinical=clinical,
                    prognostic_results=prog_results)

    summary = gf.gene_summary(gene)

    if not summary.get("found"):
        print(f"  ❌  Gene '{gene}' not found in expression data.")
        return

    print(f"\n{'='*60}")
    print(f"  Gene Lookup: {gene}")
    print(f"{'='*60}")

    for section, data in summary.items():
        if section in ("gene", "found"):
            continue
        if data:
            print(f"\n  ── {section.upper()} ──")
            if isinstance(data, dict):
                for k, v in data.items():
                    print(f"     {k}: {v}")
            else:
                print(f"     {data}")

    print(f"\n{'='*60}")


def _run_gene_spotlight(args: argparse.Namespace) -> None:
    """Generate per-gene thesis spotlight figures."""
    from gbm_multiomics.analysis.gene_focus import GeneFocus

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else data_dir / "analysis" / "gene_focus" / "spotlight"

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]

    norm_path = data_dir / "preprocessed" / "normalized_expression.tsv"
    if not norm_path.exists():
        print("  ❌  No expression data. Run preprocess first.")
        return

    expr = pd.read_csv(norm_path, sep="\t", index_col=0)
    from gbm_multiomics.preprocessing.clinical import build_unified_metadata
    clinical = build_unified_metadata(data_dir)

    gf = GeneFocus(genes=genes, expr=expr, clinical=clinical)
    gf.generate_spotlight_figures(output_dir=out_dir, group_col=args.group_by)


def _run_gene_rank(args: argparse.Namespace) -> None:
    """Rank user's genes against genome-wide prognostic results."""
    from gbm_multiomics.analysis.gene_focus import GeneFocus

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output) if args.output else data_dir / "analysis" / "gene_focus"
    out_dir.mkdir(parents=True, exist_ok=True)

    genes = [g.strip() for g in args.genes.split(",") if g.strip()]

    prog_path = data_dir / "analysis" / "prognostic" / "prognostic_univariate.tsv"
    if not prog_path.exists():
        print("  ❌  No prognostic results. Run `gbm-pipeline analyze --module prognostic` first.")
        return

    prog_results = pd.read_csv(prog_path, sep="\t")

    gf = GeneFocus(genes=genes, expr=pd.DataFrame(), clinical=pd.DataFrame(),
                    prognostic_results=prog_results)

    ranking = gf.rank_against_genome()
    if not ranking.empty:
        ranking.to_csv(out_dir / "gene_ranking.tsv", sep="\t", index=False)
        print(f"\n  📄  Ranking saved to: {out_dir / 'gene_ranking.tsv'}")

        # Print top-ranked
        print(f"\n  {'Gene':<15} {'Rank':<8} {'%ile':<8} {'HR':<8} {'padj':<10} {'Sig':<6}")
        print(f"  {'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*10} {'─'*6}")
        for _, row in ranking.iterrows():
            if pd.notna(row["prognostic_rank"]):
                print(f"  {row['gene']:<15} {int(row['prognostic_rank']):>6,}  "
                      f"{row['percentile']:>5.1f}%  "
                      f"{row['HR']:>6.3f}  "
                      f"{row['padj']:>8.2e}  "
                      f"{'✓' if row['is_significant'] else ' '}")


def main_preprocess() -> None:
    """Entry point: gbm-preprocess"""
    parser = argparse.ArgumentParser(
        prog="gbm-preprocess",
        description="QC and normalize TCGA-GBM RNA-seq data.",
    )
    parser.add_argument("--data-dir", "-d", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--method", choices=["vst", "tpm", "cpm"], default="vst")
    parser.add_argument("--skip-annotation", action="store_true")
    args = parser.parse_args()
    try:
        _run_preprocess(args)
    except Exception as exc:
        print(f"\n  ❌  Error: {exc}")
        sys.exit(1)


def main_report() -> None:
    """Entry point: gbm-report"""
    parser = argparse.ArgumentParser(
        prog="gbm-report",
        description="Generate GBM thesis report.",
    )
    parser.add_argument("--results-dir", "-r", default="results")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--include-code", action="store_true")
    args = parser.parse_args()
    try:
        _run_report(args)
    except Exception as exc:
        print(f"\n  ❌  Error: {exc}")
        sys.exit(1)
