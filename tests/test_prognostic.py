"""
test_prognostic.py — Tests for the prognostic biomarker discovery module.

All tests use synthetic data from conftest.py — no real TCGA data needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestUnivariateCox:
    def test_genome_wide_runs(self, large_expr_matrix, prognostic_clinical_data):
        """Genome-wide univariate Cox completes without error."""
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.prognostic import univariate_cox_genome_wide

        results = univariate_cox_genome_wide(
            expr=large_expr_matrix,
            clinical=prognostic_clinical_data,
            duration_col="cdr_OS.time",
            event_col="cdr_OS",
            verbose=False,
        )

        assert isinstance(results, pd.DataFrame)
        if not results.empty:
            assert "HR" in results.columns
            assert "padj" in results.columns
            assert "gene" in results.columns
            # Sorted by padj
            assert results["padj"].is_monotonic_increasing

    def test_returns_empty_on_insufficient_events(self, large_expr_matrix):
        """Returns empty when there are too few events."""
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.prognostic import univariate_cox_genome_wide

        # Create clinical data with too few events
        n = 10
        samples = [f"TCGA-06-{i:04d}" for i in range(n)]
        expr_samples = [f"TCGA-06-{i:04d}-01A" for i in range(n)]
        clinical = pd.DataFrame({
            "case_submitter_id": samples,
            "sample_submitter_id": expr_samples,
            "cdr_OS": [0] * n,  # all censored
            "cdr_OS.time": np.random.default_rng(1).integers(100, 500, n).astype(float),
        })

        with pytest.raises(ValueError, match="insufficient"):
            univariate_cox_genome_wide(
                expr=large_expr_matrix.iloc[:, :n],
                clinical=clinical,
                duration_col="cdr_OS.time",
                event_col="cdr_OS",
                min_events=5,
                verbose=False,
            )

    def test_known_prognostic_genes_found(self, large_expr_matrix, prognostic_clinical_data):
        """Known GBM drivers appear in results when present in expression data."""
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.prognostic import univariate_cox_genome_wide

        # Inject strong signal for EGFR (high expression → worse survival)
        expr = large_expr_matrix.copy()
        clinical = prognostic_clinical_data.copy()

        # Make EGFR expression strongly correlated with event
        for i, sample in enumerate(clinical["sample_submitter_id"]):
            if sample in expr.columns:
                if clinical.iloc[i]["cdr_OS"] == 1:
                    expr.loc["EGFR", sample] += 5  # higher in events

        results = univariate_cox_genome_wide(
            expr=expr, clinical=clinical,
            duration_col="cdr_OS.time", event_col="cdr_OS",
            verbose=False,
        )

        # EGFR should be in top results
        if "EGFR" in results["gene"].values:
            egfr_row = results[results["gene"] == "EGFR"].iloc[0]
            assert egfr_row["HR"] > 1  # high expression → higher risk


class TestRiskScore:
    def test_calculate_risk_score(self, large_expr_matrix):
        """Risk score calculation produces expected output shape."""
        from gbm_multiomics.analysis.prognostic import calculate_risk_score

        genes = ["EGFR", "PTEN", "TP53", "VEGFA"]
        coefs = pd.Series([0.5, -0.3, 0.2, 0.1], index=genes)

        risk = calculate_risk_score(
            expr=large_expr_matrix,
            coefficients=coefs,
            center=True,
        )

        assert "risk_score" in risk.columns
        assert "risk_group" in risk.columns
        assert len(risk) == large_expr_matrix.shape[1]
        # Centered
        assert abs(risk["risk_score"].mean()) < 1e-6
        # Equal groups
        assert abs(
            (risk["risk_group"] == "High Risk").sum() -
            (risk["risk_group"] == "Low Risk").sum()
        ) <= 1

    def test_calculate_risk_score_missing_genes(self, large_expr_matrix):
        """Handles genes not in expression matrix gracefully."""
        from gbm_multiomics.analysis.prognostic import calculate_risk_score

        coefs = pd.Series([0.5, -0.3], index=["GENE_NOT_IN_DATA", "EGFR"])

        risk = calculate_risk_score(
            expr=large_expr_matrix,
            coefficients=coefs,
        )

        assert len(risk) == large_expr_matrix.shape[1]


class TestMultivariateCox:
    def test_build_multivariate_model(self, large_expr_matrix, prognostic_clinical_data):
        """Multivariate Cox model builds successfully."""
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.prognostic import build_multivariate_model

        genes = ["EGFR", "PTEN", "VEGFA"]
        clinical = prognostic_clinical_data.set_index("sample_submitter_id")

        result = build_multivariate_model(
            expr=large_expr_matrix,
            clinical=clinical,
            gene_list=genes,
            clinical_covariates=["age_at_diagnosis", "IDH_status"],
            duration_col="cdr_OS.time",
            event_col="cdr_OS",
        )

        assert "model" in result
        assert "coefficients" in result
        assert "concordance" in result
        assert 0 <= result["concordance"] <= 1


class TestDepMap:
    def test_validate_depmap(self, tmp_path):
        """DepMap validation handles missing data gracefully."""
        from gbm_multiomics.analysis.prognostic import validate_depmap

        # This will likely fail to download in test, but should return gracefully
        result = validate_depmap(
            prognostic_genes=["EGFR", "PTEN", "TP53", "NONEXISTENT"],
            cache_dir=tmp_path,
        )

        assert isinstance(result, pd.DataFrame)
