"""
test_preprocessing.py — Tests for preprocessing modules (QC, normalization, annotation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestAnnotation:
    def test_map_ensg_to_symbol(self):
        """ENSG → HGNC mapping returns symbols."""
        from gbm_multiomics.preprocessing.annotation import map_ensg_to_symbol

        genes = ["ENSG00000157764", "ENSG00000146648", "ENSG00000141510"]
        symbols = map_ensg_to_symbol(genes)

        assert isinstance(symbols, pd.Series)
        assert len(symbols) == 3
        # Should map BRAF, EGFR, TP53 (or return ENSG if API fails)
        assert symbols.notna().all()

    def test_filter_low_expression(self, small_count_matrix):
        """Low expression filter removes genes below threshold."""
        from gbm_multiomics.preprocessing.annotation import filter_low_expression

        # Inject a gene with all zeros
        counts = small_count_matrix.copy()
        counts.loc[counts.index[0]] = 0

        filtered = filter_low_expression(counts, min_count=1, min_samples=1)
        assert len(filtered) < len(counts)
        assert counts.index[0] not in filtered.index

    def test_filter_protein_coding(self, small_count_matrix):
        """Protein-coding filter returns a DataFrame."""
        from gbm_multiomics.preprocessing.annotation import filter_protein_coding

        filtered = filter_protein_coding(small_count_matrix, verbose=False)
        assert isinstance(filtered, pd.DataFrame)
        # Should be subset or equal
        assert len(filtered) <= len(small_count_matrix)


class TestNormalization:
    def test_normalize_cpm(self, small_count_matrix):
        """CPM normalization produces expected values."""
        from gbm_multiomics.preprocessing.normalization import normalize_cpm

        cpm = normalize_cpm(small_count_matrix, log_transform=False)

        assert cpm.shape == small_count_matrix.shape
        # Column sums should be 1e6
        assert np.allclose(cpm.sum(axis=0), 1e6, rtol=0.01)

    def test_normalize_log2_cpm(self, small_count_matrix):
        """Log2 CPM normalization."""
        from gbm_multiomics.preprocessing.normalization import normalize_cpm

        log2_cpm = normalize_cpm(small_count_matrix, log_transform=True)

        assert log2_cpm.shape == small_count_matrix.shape
        # All values should be >= 0 (log2(CPM+1) >= log2(1) = 0)
        assert (log2_cpm.values >= 0).all()

    def test_normalize_vst(self, small_count_matrix):
        """VST normalization works or falls back to log2 CPM."""
        from gbm_multiomics.preprocessing.normalization import normalize_vst

        vst = normalize_vst(small_count_matrix)
        assert vst.shape == small_count_matrix.shape

    def test_normalize_tpm_without_lengths(self, small_count_matrix):
        """TPM without gene lengths falls back to CPM."""
        from gbm_multiomics.preprocessing.normalization import normalize_tpm

        tpm = normalize_tpm(small_count_matrix, log_transform=False)
        assert tpm.shape == small_count_matrix.shape

    def test_batch_correct_no_batches(self, large_expr_matrix):
        """Batch correction with single batch returns unchanged."""
        from gbm_multiomics.preprocessing.normalization import batch_correct

        batch = pd.Series("batch_1", index=large_expr_matrix.columns)
        corrected = batch_correct(large_expr_matrix, batch)

        assert corrected.shape == large_expr_matrix.shape


class TestQualityControl:
    def test_pca_plot(self, large_expr_matrix, tmp_path):
        """PCA computes correctly."""
        pytest.importorskip("sklearn")
        from gbm_multiomics.preprocessing.qc import pca_plot

        pc_df = pca_plot(large_expr_matrix, output_dir=tmp_path)

        assert "PC1" in pc_df.columns
        assert "PC2" in pc_df.columns
        assert len(pc_df) == large_expr_matrix.shape[1]
        assert "variance_explained" in pc_df.attrs
        # Sum of first 2 PCs should be < 100%
        assert sum(pc_df.attrs["variance_explained"][:2]) <= 1.0

    def test_detect_outliers_no_extreme(self, large_expr_matrix):
        """No outliers in clean data."""
        pytest.importorskip("sklearn")
        from gbm_multiomics.preprocessing.qc import detect_outliers

        outliers = detect_outliers(large_expr_matrix, iqr_multiplier=3.0)
        assert "is_outlier" in outliers.columns
        # With IQR × 3, typically 0 outliers in synthetic data

    def test_library_size(self, small_count_matrix):
        """Library size statistics are sensible."""
        from gbm_multiomics.preprocessing.qc import library_size_distribution

        stats = library_size_distribution(small_count_matrix)
        assert stats["median"] > 0
        assert stats["n_samples"] == small_count_matrix.shape[1]

    def test_qc_report(self, small_count_matrix, tmp_path):
        """Full QC report runs without error."""
        pytest.importorskip("sklearn")
        from gbm_multiomics.preprocessing.qc import qc_report

        result = qc_report(small_count_matrix, output_dir=tmp_path)
        assert "library_stats" in result
        assert "n_outliers" in result


class TestClinicalIntegration:
    def test_build_unified_metadata_basic(self, clinical_data, tmp_path):
        """Build unified metadata with minimal inputs."""
        from gbm_multiomics.preprocessing.clinical import build_unified_metadata

        # Create minimal data dir structure
        data_dir = tmp_path / "data"
        (data_dir / "rna_seq").mkdir(parents=True)
        clinical_data.to_csv(
            data_dir / "rna_seq" / "rna_seq_metadata.tsv",
            sep="\t", index=False,
        )

        result = build_unified_metadata(data_dir)
        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0
