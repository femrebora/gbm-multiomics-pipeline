"""
test_gene_focus.py — Tests for the GeneFocus module and gene spotlight figures.

All tests use synthetic data from conftest.py — no real TCGA data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestGeneFocus:
    @pytest.fixture
    def gene_focus_data(self, large_expr_matrix, prognostic_clinical_data):
        """Build GeneFocus with synthetic expression + clinical + prognostic data."""
        expr = large_expr_matrix.copy()
        clinical = prognostic_clinical_data.copy()

        # Build synthetic prognostic results
        genes_in_expr = [g for g in expr.index if g in [
            "EGFR", "PTEN", "TP53", "IDH1", "NF1", "VEGFA", "SOX2", "MYC", "CHI3L1"
        ]]
        prog_rows = []
        for i, gene in enumerate(expr.index):
            hr = 2.0 if gene == "EGFR" else 0.5 if gene == "PTEN" else 1.0 + np.random.default_rng(i).normal(0, 0.3)
            pval = 0.0001 if gene in ("EGFR", "PTEN", "TP53") else np.random.default_rng(i).uniform(0.001, 0.99)
            prog_rows.append({
                "gene": gene,
                "HR": hr,
                "HR_lower_95": hr * 0.7,
                "HR_upper_95": hr * 1.4,
                "coef": np.log(hr),
                "se_coef": 0.2,
                "p_value": pval,
                "padj": min(pval * len(expr.index), 0.99),
                "concordance": 0.65,
                "direction": "high-risk" if hr > 1 else "protective",
            })

        prog_results = pd.DataFrame(prog_rows).sort_values("padj")

        return {"expr": expr, "clinical": clinical, "prog": prog_results}

    def test_init_filters_missing_genes(self, gene_focus_data):
        """GeneFocus filters out genes not in expression data."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR", "PTEN", "NONEXISTENT_GENE_XYZ"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=gene_focus_data["prog"],
        )

        assert "EGFR" in gf.genes
        assert "PTEN" in gf.genes
        assert "NONEXISTENT_GENE_XYZ" not in gf.genes

    def test_init_empty_genes_raises(self, gene_focus_data):
        """ValueError when no genes are found in expression data."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        with pytest.raises(ValueError, match="None of the requested genes"):
            GeneFocus(
                genes=["NONEXISTENT_1", "NONEXISTENT_2"],
                expr=gene_focus_data["expr"],
                clinical=gene_focus_data["clinical"],
            )

    def test_gene_summary_returns_all_sections(self, gene_focus_data):
        """gene_summary returns expected keys for a gene in the data."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=gene_focus_data["prog"],
        )

        summary = gf.gene_summary("EGFR")
        assert summary["found"] is True
        assert summary["gene"] == "EGFR"
        assert "expression" in summary
        assert "mean" in summary["expression"]
        assert "sd" in summary["expression"]
        assert "prognostic" in summary
        assert "HR" in summary["prognostic"]
        assert "rank" in summary["prognostic"]
        assert "percentile" in summary["prognostic"]

    def test_gene_summary_missing_gene(self, gene_focus_data):
        """gene_summary returns found=False for gene not in data."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
        )

        summary = gf.gene_summary("NONEXISTENT_GENE")
        assert summary["found"] is False

    def test_gene_report_generates_dataframe(self, gene_focus_data):
        """gene_report returns a DataFrame with expected columns."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        genes = ["EGFR", "PTEN", "TP53", "VEGFA", "SOX2"]
        gf = GeneFocus(
            genes=genes,
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=gene_focus_data["prog"],
        )

        report = gf.gene_report()
        assert isinstance(report, pd.DataFrame)
        assert "gene" in report.columns
        assert len(report) >= 1

    def test_gene_report_saves_to_output_dir(self, gene_focus_data, tmp_path):
        """gene_report writes TSV and heatmap to output_dir."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR", "PTEN", "TP53"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=gene_focus_data["prog"],
        )

        report = gf.gene_report(output_dir=tmp_path)
        assert (tmp_path / "gene_focus_report.tsv").exists()

    def test_rank_against_genome(self, gene_focus_data):
        """rank_against_genome shows correct ranking information."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR", "PTEN", "TP53"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=gene_focus_data["prog"],
        )

        ranking = gf.rank_against_genome()
        assert isinstance(ranking, pd.DataFrame)
        assert "prognostic_rank" in ranking.columns
        assert "percentile" in ranking.columns
        assert "HR" in ranking.columns
        assert "is_top_5pct" in ranking.columns

        # EGFR should be highly ranked (injected signal: pval=0.0001)
        egfr_row = ranking[ranking["gene"] == "EGFR"]
        if not egfr_row.empty:
            assert egfr_row.iloc[0]["is_significant"]  # padj is tight but should be sig

    def test_rank_against_genome_without_prognostic(self, gene_focus_data):
        """rank_against_genome returns empty DataFrame without prognostic data."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
            prognostic_results=None,
        )

        ranking = gf.rank_against_genome()
        assert ranking.empty

    def test_compare_depmap_without_data(self, gene_focus_data):
        """compare_depmap returns empty when no DepMap data available."""
        from gbm_multiomics.analysis.gene_focus import GeneFocus

        gf = GeneFocus(
            genes=["EGFR"],
            expr=gene_focus_data["expr"],
            clinical=gene_focus_data["clinical"],
        )

        result = gf.compare_depmap()
        assert result.empty


class TestGeneSpotlightFigures:
    def test_expression_violin_creates_figure(self, large_expr_matrix, sample_metadata, tmp_path):
        """Expression violin generates a PDF file."""
        pytest.importorskip("matplotlib")
        from gbm_multiomics.visualization.gene_spotlight import gene_expression_violin

        # Pick a gene in the matrix
        gene = large_expr_matrix.index[0]

        paths = gene_expression_violin(
            expr=large_expr_matrix,
            gene=gene,
            clinical=None,
            output_dir=tmp_path,
        )

        assert len(paths) > 0
        assert any(p.suffix == ".pdf" for p in paths)

    def test_expression_violin_missing_gene_raises(self, large_expr_matrix):
        """ValueError when gene is not in expression data."""
        pytest.importorskip("matplotlib")
        from gbm_multiomics.visualization.gene_spotlight import gene_expression_violin

        with pytest.raises(ValueError, match="not found"):
            gene_expression_violin(
                expr=large_expr_matrix,
                gene="NONEXISTENT_GENE_ABC",
            )

    def test_expression_violin_with_groups(self, large_expr_matrix, sample_metadata, tmp_path):
        """Expression violin with clinical groups works."""
        pytest.importorskip("matplotlib")
        from gbm_multiomics.visualization.gene_spotlight import gene_expression_violin

        # Build metadata matching expression columns
        meta = sample_metadata.copy()
        meta["sample_submitter_id"] = large_expr_matrix.columns[:len(meta)]

        gene = "EGFR" if "EGFR" in large_expr_matrix.index else large_expr_matrix.index[0]

        paths = gene_expression_violin(
            expr=large_expr_matrix,
            gene=gene,
            clinical=meta,
            group_col="is_tumor",
            output_dir=tmp_path,
        )

        assert len(paths) > 0

    def test_multiomics_dashboard_creates_figure(
        self, large_expr_matrix, mutation_matrix, tmp_path
    ):
        """Multi-omics dashboard generates a PDF figure."""
        pytest.importorskip("matplotlib")
        from gbm_multiomics.visualization.gene_spotlight import gene_multiomics_dashboard

        gene = "EGFR" if "EGFR" in large_expr_matrix.index else large_expr_matrix.index[0]

        paths = gene_multiomics_dashboard(
            expr=large_expr_matrix,
            gene=gene,
            mutations=mutation_matrix,
            output_dir=tmp_path,
        )

        assert len(paths) > 0
        assert any(p.suffix == ".pdf" for p in paths)

    def test_multiomics_dashboard_missing_gene(self, large_expr_matrix):
        """ValueError when gene missing."""
        pytest.importorskip("matplotlib")
        from gbm_multiomics.visualization.gene_spotlight import gene_multiomics_dashboard

        with pytest.raises(ValueError, match="not found"):
            gene_multiomics_dashboard(
                expr=large_expr_matrix,
                gene="NONEXISTENT",
            )
