"""
test_multiomics.py — Tests for multi-omics integration, immune, and network modules.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestCrossOmicsCorrelation:
    def test_rna_cnv_correlation(self, large_expr_matrix, tmp_path):
        """RNA-CNV correlation computes successfully."""
        pytest.importorskip("scipy")
        from gbm_multiomics.analysis.multiomics import cross_omics_correlation

        # Create synthetic CNV data matching expression genes
        cnv = pd.DataFrame(
            np.random.default_rng(42).normal(0, 0.5, size=large_expr_matrix.shape),
            index=large_expr_matrix.index,
            columns=large_expr_matrix.columns,
        )

        results = cross_omics_correlation(
            rna_expr=large_expr_matrix,
            cnv=cnv,
            output_dir=tmp_path,
        )

        assert "rna_cnv" in results
        corr_df = results["rna_cnv"]
        assert "pearson_r" in corr_df.columns
        assert "gene" in corr_df.columns

    def test_mutation_expr_correlation(self, large_expr_matrix, mutation_matrix, tmp_path):
        """Mutation-expression correlation runs."""
        from gbm_multiomics.analysis.multiomics import cross_omics_correlation

        # Align mutation samples with expression columns
        common_samples = sorted(
            set(large_expr_matrix.columns) & set(mutation_matrix.columns)
        )
        if len(common_samples) < 3:
            # Re-index mutation to match expression samples
            mutation_matrix.columns = large_expr_matrix.columns[:len(mutation_matrix.columns)]

        results = cross_omics_correlation(
            rna_expr=large_expr_matrix,
            mutations=mutation_matrix,
            output_dir=tmp_path,
        )

        # May be empty if genes don't overlap, but shouldn't error
        assert isinstance(results, dict)


class TestSNFCluster:
    def test_snf_cluster_placeholder(self):
        """SNF clustering gracefully handles missing snfpy."""
        from gbm_multiomics.analysis.multiomics import snf_cluster

        views = {
            "rna": pd.DataFrame(
                np.random.default_rng(1).normal(0, 1, (100, 20)),
                index=[f"Gene_{i}" for i in range(100)],
                columns=[f"Sample_{i}" for i in range(20)],
            ),
        }

        result = snf_cluster(views)
        # Should return empty dict if snfpy not available
        assert isinstance(result, dict)


class TestImmuneAnalysis:
    def test_estimate_scores(self, tmp_path):
        """ESTIMATE scores compute successfully."""
        from gbm_multiomics.analysis.immune import estimate_scores, ESTIMATE_STROMAL_GENES, ESTIMATE_IMMUNE_GENES

        # Build expression matrix with ESTIMATE signature genes
        all_genes = ESTIMATE_STROMAL_GENES[:20] + ESTIMATE_IMMUNE_GENES[:20] + ["GENE_EXTRA"]
        rng = np.random.default_rng(42)
        n_samples = 20
        expr = pd.DataFrame(
            rng.normal(8, 2, size=(len(all_genes), n_samples)),
            index=all_genes,
            columns=[f"Sample_{i}" for i in range(n_samples)],
        )

        scores = estimate_scores(expr, output_dir=tmp_path)

        assert "StromalScore" in scores.columns
        assert "ImmuneScore" in scores.columns
        assert "ESTIMATEScore" in scores.columns
        assert "TumorPurity" in scores.columns
        assert len(scores) == n_samples
        assert scores["TumorPurity"].between(0, 1).all()

    def test_immune_survival_split(self, large_expr_matrix, prognostic_clinical_data, tmp_path):
        """Immune score survival stratification runs."""
        pytest.importorskip("lifelines")
        from gbm_multiomics.analysis.immune import estimate_scores, immune_survival_split

        scores = estimate_scores(large_expr_matrix)
        # Set index to match clinical merge
        scores.index = prognostic_clinical_data["sample_submitter_id"].values[:len(scores)]

        clinical = prognostic_clinical_data.copy()
        clinical["case_submitter_id"] = clinical["sample_submitter_id"]

        result = immune_survival_split(
            scores, clinical,
            score_col="ImmuneScore",
            output_dir=tmp_path,
        )

        assert "logrank_pvalue" in result


class TestNetworkAnalysis:
    def test_build_ppi_network(self):
        """PPI network builds from gene list."""
        from gbm_multiomics.analysis.network import build_ppi_network

        genes = ["EGFR", "PTEN", "TP53", "NF1", "IDH1"]
        network = build_ppi_network(genes, score_threshold=900)

        assert "edges" in network
        assert "nodes" in network
        # May be empty if no STRING access, but should not error

    def test_identify_hub_genes_empty(self):
        """Hub gene identification handles empty network."""
        from gbm_multiomics.analysis.network import identify_hub_genes

        hubs = identify_hub_genes({"nodes": pd.DataFrame(), "edges": pd.DataFrame()})
        assert hubs.empty

    def test_coexpression_network(self, large_expr_matrix, tmp_path):
        """Co-expression network builds from expression data."""
        from gbm_multiomics.analysis.network import coexpression_network

        # Use a small subset for speed
        expr_subset = large_expr_matrix.iloc[:50]
        edges = coexpression_network(
            expr_subset,
            min_correlation=0.5,
            output_dir=tmp_path,
        )

        assert isinstance(edges, pd.DataFrame)
        if not edges.empty:
            assert "source" in edges.columns
            assert "target" in edges.columns
            assert "weight" in edges.columns
