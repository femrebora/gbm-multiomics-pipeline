"""
test_analysis.py — Unit tests for analysis modules.

All tests use synthetic data, no real TCGA data or API calls required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ── Differential expression ───────────────────────────────────────────────────

class TestDifferentialExpression:
    def test_prepare_deseq2_inputs_alignment(
        self, small_count_matrix, sample_metadata, tmp_path
    ):
        from gbm_multiomics.analysis.differential_expression import prepare_deseq2_inputs

        counts_aligned, coldata = prepare_deseq2_inputs(
            small_count_matrix,
            sample_metadata,
            condition_col    = "is_tumor",
            sample_col       = "sample_submitter_id",
            output_dir       = tmp_path,
        )
        # All columns in counts_aligned must be in metadata
        assert set(counts_aligned.columns).issubset(
            set(sample_metadata["sample_submitter_id"])
        )
        assert "is_tumor" in coldata.columns
        assert (tmp_path / "counts_for_deseq2.tsv").exists()
        assert (tmp_path / "coldata_for_deseq2.tsv").exists()

    def test_prepare_deseq2_inputs_no_overlap_raises(self, small_count_matrix):
        from gbm_multiomics.analysis.differential_expression import prepare_deseq2_inputs

        bad_meta = pd.DataFrame({
            "sample_submitter_id": ["WRONG-001", "WRONG-002"],
            "is_tumor": [True, False],
        })
        with pytest.raises(ValueError, match="No overlapping samples"):
            prepare_deseq2_inputs(small_count_matrix, bad_meta, "is_tumor")

    def test_filter_significant(self):
        from gbm_multiomics.analysis.differential_expression import filter_significant

        de = pd.DataFrame({
            "log2FoldChange": [3.0, -2.5, 0.1, -0.05, 4.0],
            "padj":           [0.001, 0.01, 0.9, 0.3, 0.0001],
        }, index=["GENE1", "GENE2", "GENE3", "GENE4", "GENE5"])

        sig = filter_significant(de, padj_threshold=0.05, lfc_threshold=1.0)
        assert "GENE3" not in sig.index
        assert "GENE4" not in sig.index
        assert "UP" in sig["direction"].values
        assert "DOWN" in sig["direction"].values

    def test_write_r_script(self, tmp_path):
        from gbm_multiomics.analysis.differential_expression import write_r_deseq2_script

        script_path = write_r_deseq2_script(
            counts_path   = tmp_path / "counts.tsv",
            coldata_path  = tmp_path / "coldata.tsv",
            condition_col = "is_tumor",
            reference     = "False",
            output_dir    = tmp_path,
        )
        assert script_path.exists()
        content = script_path.read_text()
        assert "DESeqDataSetFromMatrix" in content
        assert "is_tumor" in content


# ── Pathway enrichment ────────────────────────────────────────────────────────

class TestPathwayEnrichment:
    def test_gbm_gene_sets_defined(self):
        from gbm_multiomics.analysis.pathway_enrichment import GBM_GENE_SETS

        assert "RTK_PI3K_AKT_pathway" in GBM_GENE_SETS
        assert "p53_RB_cell_cycle" in GBM_GENE_SETS
        assert "GBM_invasion" in GBM_GENE_SETS
        assert len(GBM_GENE_SETS) >= 6

    def test_make_ranked_list(self):
        from gbm_multiomics.analysis.pathway_enrichment import make_ranked_list

        de = pd.DataFrame({
            "log2FoldChange": [3.0, -2.0, 0.5],
            "padj":           [0.001, 0.01, 0.5],
            "stat":           [10.0, -8.0, 1.0],
        }, index=["EGFR", "PTEN", "TP53"])

        ranked = make_ranked_list(de)
        assert ranked.index[0] == "EGFR"  # highest stat first
        assert ranked.is_monotonic_decreasing


# ── Survival analysis ─────────────────────────────────────────────────────────

class TestSurvivalAnalysis:
    def test_prepare_survival_data_filters_invalid(self, clinical_data):
        from gbm_multiomics.analysis.survival import prepare_survival_data

        # Add invalid rows
        clinical_data.loc[0, "cdr_OS.time"] = np.nan
        clinical_data.loc[1, "cdr_OS.time"] = -10
        clinical_data.loc[2, "cdr_OS"]      = 5   # invalid event code

        result = prepare_survival_data(
            clinical_data,
            duration_col = "cdr_OS.time",
            event_col    = "cdr_OS",
        )
        assert (result["cdr_OS.time"] > 0).all()
        assert result["cdr_OS"].isin([0, 1]).all()

    def test_kaplan_meier_runs(self, clinical_data, tmp_path):
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.survival import kaplan_meier, prepare_survival_data

        df = prepare_survival_data(
            clinical_data, duration_col="cdr_OS.time", event_col="cdr_OS"
        )
        result = kaplan_meier(
            df, "cdr_OS.time", "cdr_OS", "IDH_status",
            output_dir=tmp_path,
        )
        assert "logrank_pvalue" in result
        assert 0.0 <= result["logrank_pvalue"] <= 1.0

    def test_cox_univariate_runs(self, clinical_data, tmp_path):
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.survival import cox_univariate, prepare_survival_data

        df = prepare_survival_data(
            clinical_data, duration_col="cdr_OS.time", event_col="cdr_OS"
        )
        result = cox_univariate(
            df, "cdr_OS.time", "cdr_OS",
            covariates=["IDH_status"],
            output_dir=tmp_path,
        )
        assert isinstance(result, pd.DataFrame)

    def test_expression_survival_split_median(self, clinical_data):
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.survival import (
            expression_survival_split, prepare_survival_data
        )
        df = prepare_survival_data(
            clinical_data, duration_col="cdr_OS.time", event_col="cdr_OS"
        )
        # Assign dummy expression values
        gene_expr = pd.Series(
            np.random.default_rng(1).random(len(df)),
            index=df.index,
        )
        result = expression_survival_split(
            df, gene_expr, "cdr_OS.time", "cdr_OS",
            split="median", gene_name="EGFR",
        )
        assert "logrank_pvalue" in result


# ── Subtype classification ────────────────────────────────────────────────────

class TestSubtypeClassification:
    def _make_expr_matrix(self, n_samples: int = 20) -> pd.DataFrame:
        """Synthetic log2(CPM+1) matrix with Verhaak signature genes."""
        from gbm_multiomics.analysis.subtype import VERHAAK_SIGNATURES
        all_genes = list({g for genes in VERHAAK_SIGNATURES.values() for g in genes})
        all_genes += [f"BACKGROUND_{i}" for i in range(200)]
        rng = np.random.default_rng(99)
        data = rng.random((len(all_genes), n_samples)) * 10
        samples = [f"TCGA-06-{i:04d}-01A" for i in range(n_samples)]
        return pd.DataFrame(data, index=all_genes, columns=samples)

    def test_classify_centroids_returns_all_samples(self, tmp_path):
        from gbm_multiomics.analysis.subtype import classify_centroids

        expr = self._make_expr_matrix(n_samples=12)
        result = classify_centroids(expr, output_dir=tmp_path)
        assert len(result) == 12
        assert "assigned_subtype" in result.columns
        assert set(result["assigned_subtype"]).issubset(
            {"Classical", "Mesenchymal", "Proneural", "Neural"}
        )

    def test_classify_centroids_3_subtypes(self):
        from gbm_multiomics.analysis.subtype import classify_centroids

        expr = self._make_expr_matrix()
        result = classify_centroids(expr, use_3_subtypes=True)
        assert "Neural" not in result["assigned_subtype"].values

    def test_who_2021_classify(self, idh_status_data):
        from gbm_multiomics.analysis.subtype import who_2021_classify

        result = who_2021_classify(idh_status_data)
        assert "who_2021_provisional" in result.columns
        wt_rows = result[result["IDH_status"] == "IDH_wildtype"]
        assert (wt_rows["who_2021_provisional"] == "Glioblastoma_IDH-wildtype_G4").all()
        mut_rows = result[result["IDH_status"] == "IDH_mutant"]
        assert (mut_rows["who_2021_provisional"].str.contains("Astrocytoma")).all()
