"""
gene_focus.py — Gene Focus Module for TCGA-GBM thesis.

Provides a GeneFocus class that aggregates per-gene data from all omics
and analysis modules, enabling:
  1. Single-gene lookup across all omics
  2. Multi-gene comparison report
  3. Ranking user's genes against genome-wide prognostic results
  4. Thesis spotlight figure generation

Usage
-----
    from gbm_multiomics.analysis.gene_focus import GeneFocus

    gf = GeneFocus(
        genes=["EGFR", "PTEN", "TP53", "IDH1", "NF1"],
        expr=log2_cpm,
        clinical=metadata,
        cnv=cnv_df,
        methylation=meth_df,
        mutations=mut_df,
        prognostic_results=cox_results,
    )

    # Single-gene lookup
    summary = gf.gene_summary("EGFR")

    # Multi-gene report
    report = gf.gene_report(output_dir=Path("results/gene_focus"))

    # Rank genes
    ranking = gf.rank_against_genome()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class GeneFocus:
    """
    Aggregate and query per-gene data across all omics layers.

    Parameters
    ----------
    genes : list[str]
        Gene symbols to focus on.
    expr : pd.DataFrame
        Genes × samples, normalized expression (log2 scale).
    clinical : pd.DataFrame
        Sample metadata with survival endpoints, clinical covariates.
        Must have sample IDs matching expr columns.
    cnv : pd.DataFrame, optional
        Genes × samples copy number values.
    methylation : pd.DataFrame, optional
        Probes × samples beta values.
    mutations : pd.DataFrame, optional
        Genes × samples mutation matrix (mutation type strings or 0/1).
    prognostic_results : pd.DataFrame, optional
        From univariate_cox_genome_wide(). Must have: gene, HR, HR_lower_95,
        HR_upper_95, p_value, padj columns.
    network_data : dict, optional
        From build_ppi_network() — edges, nodes, graph.
    immune_scores : pd.DataFrame, optional
        From estimate_scores() — StromalScore, ImmuneScore, ESTIMATEScore.
    depmap_results : pd.DataFrame, optional
        From validate_depmap().
    """

    def __init__(
        self,
        genes: list[str],
        expr: pd.DataFrame,
        clinical: pd.DataFrame,
        cnv: pd.DataFrame | None = None,
        methylation: pd.DataFrame | None = None,
        mutations: pd.DataFrame | None = None,
        prognostic_results: pd.DataFrame | None = None,
        network_data: dict | None = None,
        immune_scores: pd.DataFrame | None = None,
        depmap_results: pd.DataFrame | None = None,
    ):
        self.genes = [g for g in genes if g in expr.index]
        if not self.genes:
            raise ValueError(
                f"None of the requested genes found in expression data. "
                f"Requested: {genes[:5]}... Available (sample): "
                f"{list(expr.index[:10])}..."
            )
        missing = set(genes) - set(self.genes)
        if missing:
            print(f"  ℹ   {len(missing)} gene(s) not in expression data: "
                  f"{', '.join(sorted(missing)[:10])}")

        self.expr = expr
        self.clinical = clinical
        self.cnv = cnv
        self.methylation = methylation
        self.mutations = mutations
        self.prognostic_results = prognostic_results
        self.network_data = network_data
        self.immune_scores = immune_scores
        self.depmap_results = depmap_results

        # Detect sample ID column in clinical
        self._sample_col = self._detect_sample_col()

    def _detect_sample_col(self) -> str:
        """Find the sample ID column in clinical data."""
        for col in self.clinical.columns:
            if "sample" in col.lower() and "submitter" in col.lower():
                return col
        # Fallback: first column
        return self.clinical.columns[0]

    # ── Single-Gene Lookup ──────────────────────────────────────────────────

    def gene_summary(self, gene: str) -> dict[str, Any]:
        """
        Everything we know about one gene across all omics.

        Returns a dict with keys:
          - expression: mean, median, sd, tumor_mean, normal_mean, log2FC
          - prognostic: HR, HR_lower_95, HR_upper_95, p_value, padj, rank, percentile, direction
          - mutations: frequency, top_variants
          - cnv: mean_cn, gain_freq, loss_freq
          - methylation: mean_beta, expr_methylation_r (if probe mapping available)
          - network: degree_centrality, betweenness_centrality, top_neighbors
          - depmap: mean_dependency, is_essential, n_cell_lines
          - immune: stromal_r, immune_r, estimate_r
        """
        if gene not in self.expr.index:
            return {"gene": gene, "found": False, "error": "Not in expression data."}

        result: dict[str, Any] = {"gene": gene, "found": True}

        # ── Expression ──────────────────────────────────────────────────
        expr_vals = self.expr.loc[gene].astype(float).dropna()
        result["expression"] = {
            "mean": round(float(expr_vals.mean()), 3),
            "median": round(float(expr_vals.median()), 3),
            "sd": round(float(expr_vals.std()), 3),
            "min": round(float(expr_vals.min()), 3),
            "max": round(float(expr_vals.max()), 3),
            "n_samples": len(expr_vals),
        }

        # Tumor vs Normal
        common_samples = list(set(expr_vals.index) & set(self.clinical[self._sample_col]))
        if common_samples:
            clinical_idx = self.clinical.set_index(self._sample_col)
            if "is_tumor" in clinical_idx.columns:
                tumor_mask = clinical_idx.loc[common_samples, "is_tumor"].astype(bool)
                tumor_vals = expr_vals[expr_vals.index.isin(
                    common_samples[np.where(tumor_mask)[0]]
                    if isinstance(tumor_mask, pd.Series)
                    else [s for s, t in zip(common_samples, tumor_mask) if t]
                )]
                normal_vals = expr_vals[expr_vals.index.isin(
                    common_samples[~np.array(tumor_mask)]
                    if hasattr(tumor_mask, '__array__')
                    else [s for s, t in zip(common_samples, tumor_mask) if not t]
                )]
                tumor_mean = float(tumor_vals.mean()) if len(tumor_vals) > 0 else np.nan
                normal_mean = float(normal_vals.mean()) if len(normal_vals) > 0 else np.nan
                result["expression"]["tumor_mean"] = round(tumor_mean, 3)
                result["expression"]["normal_mean"] = round(normal_mean, 3)
                if not np.isnan(tumor_mean) and not np.isnan(normal_mean) and normal_mean != 0:
                    result["expression"]["log2FC_tumor_vs_normal"] = round(
                        tumor_mean - normal_mean, 3
                    )

        # ── Prognostic ──────────────────────────────────────────────────
        if self.prognostic_results is not None and gene in self.prognostic_results["gene"].values:
            prog = self.prognostic_results[self.prognostic_results["gene"] == gene].iloc[0]
            result["prognostic"] = {
                "HR": float(prog["HR"]),
                "HR_lower_95": float(prog["HR_lower_95"]),
                "HR_upper_95": float(prog["HR_upper_95"]),
                "p_value": float(prog["p_value"]),
                "padj": float(prog["padj"]),
                "direction": str(prog.get("direction", "unknown")),
            }
            # Rank
            sorted_idx = self.prognostic_results.sort_values("padj").index
            rank = sorted_idx.get_loc(
                self.prognostic_results[
                    self.prognostic_results["gene"] == gene
                ].index[0]
            ) + 1
            result["prognostic"]["rank"] = rank
            result["prognostic"]["percentile"] = round(
                rank / len(self.prognostic_results) * 100, 2
            )
            result["prognostic"]["is_top_1pct"] = result["prognostic"]["percentile"] <= 1
            result["prognostic"]["is_top_5pct"] = result["prognostic"]["percentile"] <= 5
            result["prognostic"]["is_significant"] = float(prog["padj"]) < 0.05

        # ── Mutations ───────────────────────────────────────────────────
        if self.mutations is not None and gene in self.mutations.index:
            mut_row = self.mutations.loc[gene]
            n_mut = mut_row.notna().sum()
            freq = n_mut / len(mut_row) if len(mut_row) > 0 else 0
            result["mutations"] = {
                "n_mutated": int(n_mut),
                "frequency": round(float(freq), 4),
                "n_total": len(mut_row),
            }
            # Top variant types
            if n_mut > 0:
                var_counts = mut_row.dropna().value_counts()
                result["mutations"]["top_variants"] = {
                    str(k): int(v) for k, v in var_counts.head(5).items()
                }

        # ── CNV ─────────────────────────────────────────────────────────
        if self.cnv is not None and gene in self.cnv.index:
            cnv_vals = self.cnv.loc[gene].astype(float).dropna()
            if len(cnv_vals) > 0:
                result["cnv"] = {
                    "mean_cn": round(float(cnv_vals.mean()), 3),
                    "sd_cn": round(float(cnv_vals.std()), 3),
                    "gain_frequency": round(float((cnv_vals > 0.3).mean()), 4),
                    "loss_frequency": round(float((cnv_vals < -0.3).mean()), 4),
                    "n_samples": len(cnv_vals),
                }

        # ── Methylation ─────────────────────────────────────────────────
        if self.methylation is not None:
            # Look for probes associated with this gene (name-based match for now)
            gene_probes = [p for p in self.methylation.index
                           if gene.upper() in p.upper().split("_")]
            if gene_probes:
                meth_gene = self.methylation.loc[gene_probes].mean(axis=0)
                result["methylation"] = {
                    "mean_beta": round(float(meth_gene.mean()), 4),
                    "n_probes": len(gene_probes),
                }
                # Correlation with expression
                common = list(set(self.expr.columns) & set(meth_gene.index))
                if len(common) >= 10:
                    r = self.expr.loc[gene, common].corr(meth_gene[common])
                    result["methylation"]["expr_methylation_r"] = round(float(r), 4)

        # ── Network ─────────────────────────────────────────────────────
        if self.network_data is not None:
            nodes = self.network_data.get("nodes", pd.DataFrame())
            if not nodes.empty and gene in nodes["gene"].values:
                node_row = nodes[nodes["gene"] == gene].iloc[0]
                result["network"] = {
                    "degree_centrality": round(float(node_row.get("degree_centrality", 0)), 4),
                    "betweenness_centrality": round(float(node_row.get("betweenness_centrality", 0)), 4),
                }
                # Top neighbors
                edges = self.network_data.get("edges", pd.DataFrame())
                if not edges.empty:
                    neighbors_a = edges[edges["preferredName_A"] == gene]["preferredName_B"]
                    neighbors_b = edges[edges["preferredName_B"] == gene]["preferredName_A"]
                    all_neighbors = pd.concat([neighbors_a, neighbors_b]).unique()
                    result["network"]["n_neighbors"] = len(all_neighbors)
                    result["network"]["top_neighbors"] = list(all_neighbors[:10])

        # ── DepMap ──────────────────────────────────────────────────────
        if self.depmap_results is not None and gene in self.depmap_results["gene"].values:
            dm = self.depmap_results[self.depmap_results["gene"] == gene].iloc[0]
            result["depmap"] = {
                "mean_dependency": float(dm["mean_dependency_score"]),
                "is_essential": bool(dm["is_essential"]),
                "n_cell_lines": int(dm.get("n_cell_lines", 0)),
            }

        # ── Immune ──────────────────────────────────────────────────────
        if self.immune_scores is not None:
            common = list(
                set(self.expr.columns) & set(self.immune_scores.index)
            )
            if len(common) >= 10:
                expr_gene = self.expr.loc[gene, common]
                immune_corrs = {}
                for score_col in ["StromalScore", "ImmuneScore", "ESTIMATEScore"]:
                    if score_col in self.immune_scores.columns:
                        r = expr_gene.corr(self.immune_scores.loc[common, score_col])
                        immune_corrs[f"{score_col}_r"] = round(float(r), 4)
                if immune_corrs:
                    result["immune"] = immune_corrs

        return result

    # ── Multi-Gene Report ───────────────────────────────────────────────────

    def gene_report(self, output_dir: Path | None = None) -> pd.DataFrame:
        """
        Generate a comprehensive multi-gene comparison table.

        Returns a DataFrame with one row per gene and columns for:
        expression stats, prognostic metrics, mutation frequency,
        CNV status, network centrality, DepMap dependency.

        Parameters
        ----------
        output_dir : Path, optional
            If provided, saves report TSV and generates comparison heatmap.

        Returns
        -------
        pd.DataFrame
            Genes × metrics.
        """
        rows = []
        for gene in self.genes:
            summary = self.gene_summary(gene)
            if not summary.get("found"):
                continue

            row: dict[str, Any] = {"gene": gene}

            # Expression
            expr = summary.get("expression", {})
            row["expr_mean"] = expr.get("mean")
            row["expr_sd"] = expr.get("sd")
            row["log2FC_tumor_vs_normal"] = expr.get("log2FC_tumor_vs_normal")

            # Prognostic
            prog = summary.get("prognostic", {})
            row["prognostic_HR"] = prog.get("HR")
            row["prognostic_padj"] = prog.get("padj")
            row["prognostic_rank"] = prog.get("rank")
            row["prognostic_percentile"] = prog.get("percentile")
            row["is_prognostic_top_5pct"] = prog.get("is_top_5pct", False)

            # Mutations
            mut = summary.get("mutations", {})
            row["mutation_frequency"] = mut.get("frequency")
            row["n_mutated"] = mut.get("n_mutated")

            # CNV
            cnv = summary.get("cnv", {})
            row["cnv_gain_freq"] = cnv.get("gain_frequency")
            row["cnv_loss_freq"] = cnv.get("loss_frequency")

            # Network
            net = summary.get("network", {})
            row["network_degree"] = net.get("degree_centrality")
            row["network_n_neighbors"] = net.get("n_neighbors")

            # DepMap
            dm = summary.get("depmap", {})
            row["depmap_dependency"] = dm.get("mean_dependency")
            row["depmap_is_essential"] = dm.get("is_essential")

            # Immune
            imm = summary.get("immune", {})
            row["immune_stromal_r"] = imm.get("StromalScore_r")
            row["immune_estimate_r"] = imm.get("ESTIMATEScore_r")

            rows.append(row)

        report = pd.DataFrame(rows)

        # Summary statistics
        if not report.empty:
            n_has_prog = report["prognostic_HR"].notna().sum()
            n_top5 = report["is_prognostic_top_5pct"].sum()
            n_essential = report.get("depmap_is_essential", pd.Series()).sum()
            print(f"  📊  Gene Report: {len(report)}/{len(self.genes)} genes found.")
            if n_has_prog > 0:
                print(f"       {n_top5} in top 5% prognostic, "
                      f"{n_essential} essential in DepMap.")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            report.to_csv(output_dir / "gene_focus_report.tsv", sep="\t", index=False)
            _save_gene_report_heatmap(report, output_dir)

        return report

    # ── Rank & Validate ─────────────────────────────────────────────────────

    def rank_against_genome(self) -> pd.DataFrame:
        """
        Show where each gene ranks in the genome-wide prognostic results.

        Returns
        -------
        pd.DataFrame
            gene | prognostic_rank | total_genes | percentile |
            HR | padj | is_top_1pct | is_top_5pct
        """
        if self.prognostic_results is None or self.prognostic_results.empty:
            print("  ⚠   No prognostic results available. "
                  "Run prognostic analysis first.")
            return pd.DataFrame()

        total = len(self.prognostic_results)
        sorted_prog = self.prognostic_results.sort_values("padj")

        rows = []
        for gene in self.genes:
            if gene not in sorted_prog["gene"].values:
                rows.append({
                    "gene": gene,
                    "prognostic_rank": None,
                    "total_genes": total,
                    "percentile": None,
                    "HR": None,
                    "padj": None,
                    "status": "Not in prognostic results",
                })
                continue

            match = sorted_prog[sorted_prog["gene"] == gene].iloc[0]
            rank = sorted_prog.index.get_loc(
                sorted_prog[sorted_prog["gene"] == gene].index[0]
            ) + 1
            pct = rank / total * 100

            rows.append({
                "gene": gene,
                "prognostic_rank": rank,
                "total_genes": total,
                "percentile": round(pct, 2),
                "HR": float(match["HR"]),
                "padj": float(match["padj"]),
                "is_top_1pct": pct <= 1,
                "is_top_5pct": pct <= 5,
                "is_top_10pct": pct <= 10,
                "is_significant": float(match["padj"]) < 0.05,
                "direction": str(match.get("direction", "unknown")),
            })

        result = pd.DataFrame(rows).sort_values("prognostic_rank", na_position="last")

        # Summary
        n_ranked = result["prognostic_rank"].notna().sum()
        n_top1 = result["is_top_1pct"].sum()
        n_top5 = result["is_top_5pct"].sum()
        n_top10 = result["is_top_10pct"].sum()

        print(f"  📊  Gene Ranking (among {total:,} genes):")
        print(f"       {n_top1} top 1%, {n_top5} top 5%, {n_top10} top 10%")
        print(f"       {n_ranked}/{len(self.genes)} genes found in results.")

        if n_ranked > 0:
            top_gene = result[result["prognostic_rank"].notna()].iloc[0]
            print(f"       Best ranked: {top_gene['gene']} "
                  f"(rank #{int(top_gene['prognostic_rank']):,}, "
                  f"top {top_gene['percentile']:.1f}%)")

        return result

    def compare_depmap(self) -> pd.DataFrame:
        """
        Side-by-side comparison of prognostic HR vs DepMap dependency.

        Returns
        -------
        pd.DataFrame
            gene | prognostic_HR | prognostic_padj | depmap_dependency |
            depmap_is_essential | concordant (both prognostic AND essential)
        """
        if self.depmap_results is None or self.depmap_results.empty:
            print("  ⚠   No DepMap results available.")
            return pd.DataFrame()

        rows = []
        for gene in self.genes:
            row: dict[str, Any] = {"gene": gene}

            # Prognostic
            if self.prognostic_results is not None and gene in self.prognostic_results["gene"].values:
                p = self.prognostic_results[self.prognostic_results["gene"] == gene].iloc[0]
                row["prognostic_HR"] = float(p["HR"])
                row["prognostic_padj"] = float(p["padj"])
                row["prognostic_significant"] = float(p["padj"]) < 0.05
            else:
                row["prognostic_HR"] = None
                row["prognostic_padj"] = None
                row["prognostic_significant"] = False

            # DepMap
            if gene in self.depmap_results["gene"].values:
                dm = self.depmap_results[self.depmap_results["gene"] == gene].iloc[0]
                row["depmap_dependency"] = float(dm["mean_dependency_score"])
                row["depmap_is_essential"] = bool(dm["is_essential"])
            else:
                row["depmap_dependency"] = None
                row["depmap_is_essential"] = False

            # Concordance: prognostic (HR>1, padj<0.05) AND essential (dep< -0.5)
            row["concordant"] = (
                row.get("prognostic_significant", False)
                and row.get("depmap_is_essential", False)
            )

            rows.append(row)

        result = pd.DataFrame(rows).sort_values("depmap_dependency", na_position="last")

        n_concordant = result["concordant"].sum()
        print(f"  📊  DepMap Comparison: {n_concordant} genes concordant "
              f"(prognostic + essential).")

        return result

    # ── Thesis Spotlight Figures ────────────────────────────────────────────

    def generate_spotlight_figures(
        self,
        output_dir: Path,
        group_col: str = "is_tumor",
        duration_col: str = "cdr_OS.time",
        event_col: str = "cdr_OS",
    ) -> list[Path]:
        """
        Generate per-gene publication figures for thesis.

        For each gene:
          1. Expression violin/boxplot by group
          2. KM survival curve (high vs low expression)
          3. Multi-omics dashboard (if CNV + mutations available)

        Parameters
        ----------
        output_dir : Path
        group_col : str
            Column in clinical for expression grouping.
        duration_col, event_col : str
            Survival endpoint columns.

        Returns
        -------
        list[Path]
            Paths to generated figure files.
        """
        from gbm_multiomics.visualization.gene_spotlight import (
            gene_expression_violin,
            gene_survival_km,
            gene_multiomics_dashboard,
        )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated: list[Path] = []

        for gene in self.genes:
            print(f"  🧬  Generating spotlight for: {gene}")

            # 1. Expression violin
            try:
                paths = gene_expression_violin(
                    expr=self.expr,
                    gene=gene,
                    clinical=self.clinical,
                    group_col=group_col,
                    output_dir=output_dir,
                )
                generated.extend(paths)
            except Exception as exc:
                print(f"       ⚠   Expression violin failed: {exc}")

            # 2. KM survival
            try:
                paths = gene_survival_km(
                    expr=self.expr,
                    gene=gene,
                    clinical=self.clinical,
                    duration_col=duration_col,
                    event_col=event_col,
                    output_dir=output_dir,
                )
                generated.extend(paths)
            except Exception as exc:
                print(f"       ⚠   KM survival failed: {exc}")

            # 3. Multi-omics dashboard (if data available)
            if self.cnv is not None or self.mutations is not None:
                try:
                    paths = gene_multiomics_dashboard(
                        expr=self.expr,
                        gene=gene,
                        clinical=self.clinical,
                        cnv=self.cnv,
                        mutations=self.mutations,
                        output_dir=output_dir,
                    )
                    generated.extend(paths)
                except Exception as exc:
                    print(f"       ⚠   Dashboard failed: {exc}")

        print(f"  ✅  {len(generated)} spotlight figures generated "
              f"for {len(self.genes)} genes.")
        return generated


# ── Internal plotting helper ────────────────────────────────────────────────

def _save_gene_report_heatmap(report: pd.DataFrame, output_dir: Path) -> None:
    """Save a gene × metrics heatmap for the gene report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Select numeric columns
        numeric_cols = report.select_dtypes(include=[np.number]).columns.tolist()
        if "gene" in report.columns:
            plot_df = report.set_index("gene")[numeric_cols]
        else:
            plot_df = report[numeric_cols]

        if plot_df.empty or len(plot_df) < 2:
            return

        # Z-score columns for comparison
        plot_z = (plot_df - plot_df.mean()) / plot_df.std().replace(0, 1)

        fig, ax = plt.subplots(
            figsize=(max(8, len(numeric_cols) * 0.8),
                     max(4, len(plot_df) * 0.4))
        )
        sns.heatmap(
            plot_z, ax=ax, cmap="RdBu_r", center=0,
            annot=plot_df.round(3) if len(numeric_cols) <= 6 else False,
            fmt=".3f" if len(numeric_cols) <= 6 else "",
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "Z-score"},
        )
        ax.set_title("Gene Focus — Multi-Omics Comparison", fontsize=11)
        plt.tight_layout()
        fig.savefig(output_dir / "gene_report_heatmap.pdf", dpi=300)
        plt.close(fig)
    except Exception:
        pass
