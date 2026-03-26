"""
test_downloaders.py — Unit tests for downloader metadata builders and parsers.

All tests are unit tests (no GDC API calls).
Integration tests (marked @pytest.mark.integration) are excluded from CI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gbm_multiomics.downloaders import rna_seq, methylation, mutations, cnv, mirna


# ── rna_seq ───────────────────────────────────────────────────────────────────

class TestRnaSeqMetadata:
    def test_build_metadata_columns(self):
        records = [
            {
                "file_id": "uuid-1",
                "file_name": "file_1.tsv",
                "file_size": 1000,
                "cases": [{
                    "case_id":      "case-1",
                    "submitter_id": "TCGA-06-0001",
                    "samples": [{
                        "sample_id":    "samp-1",
                        "submitter_id": "TCGA-06-0001-01A",
                        "sample_type":  "Primary Tumor",
                        "tissue_type":  "Tumor",
                    }],
                }],
            }
        ]
        meta = rna_seq.build_metadata(records)
        assert "file_id" in meta.columns
        assert "sample_submitter_id" in meta.columns
        assert "is_tumor" in meta.columns
        assert meta["is_tumor"].iloc[0] == True  # noqa: E712

    def test_tumor_normal_classification(self):
        tumor_sample   = "TCGA-06-0001-01A"
        normal_sample  = "TCGA-06-0001-11A"
        records = [
            {
                "file_id": f"uuid-{i}", "file_name": f"f{i}.tsv", "file_size": 0,
                "cases": [{"case_id": f"c{i}", "submitter_id": f"TCGA-06-000{i}",
                            "samples": [{"sample_id": f"s{i}",
                                         "submitter_id": s,
                                         "sample_type": st, "tissue_type": ""}]}],
            }
            for i, (s, st) in enumerate([
                (tumor_sample, "Primary Tumor"),
                (normal_sample, "Solid Tissue Normal"),
            ])
        ]
        meta = rna_seq.build_metadata(records)
        assert meta[meta["sample_submitter_id"] == tumor_sample]["is_tumor"].iloc[0] == True   # noqa: E712
        assert meta[meta["sample_submitter_id"] == normal_sample]["is_tumor"].iloc[0] == False  # noqa: E712

    def test_build_count_matrix(self, tmp_path, small_count_matrix, sample_metadata):
        # Create fake TSV files in UUID-named subdirs
        for _, row in sample_metadata.iterrows():
            uid_dir = tmp_path / row["file_id"]
            uid_dir.mkdir()
            tsv = uid_dir / row["file_name"]
            # Write STAR-like TSV: gene_id + unstranded column
            genes = small_count_matrix.index.tolist()
            counts = small_count_matrix[row["sample_submitter_id"]].tolist()
            lines = ["gene_id\tunstranded\n"]
            lines += [f"{g}\t{c}\n" for g, c in zip(genes, counts)]
            tsv.write_text("".join(lines))

        matrix = rna_seq.build_count_matrix(tmp_path, sample_metadata, verbose=False)
        assert matrix.shape[0] == len(small_count_matrix.index)
        assert set(matrix.columns) == set(sample_metadata["sample_submitter_id"])


# ── methylation ───────────────────────────────────────────────────────────────

class TestMethylationMetadata:
    def test_build_metadata_returns_dataframe(self):
        records = [{
            "file_id": "uuid-m1", "file_name": "beta.txt", "file_size": 100,
            "cases": [{"case_id": "c1", "submitter_id": "TCGA-06-0001",
                       "samples": [{"sample_id": "s1",
                                    "submitter_id": "TCGA-06-0001-01A",
                                    "sample_type": "Primary Tumor",
                                    "tissue_type": ""}]}],
        }]
        meta = methylation.build_metadata(records)
        assert len(meta) == 1
        assert meta["is_tumor"].iloc[0] == True  # noqa: E712

    def test_summarise_mgmt_with_probes(self):
        import pandas as pd
        probes = list(methylation.MGMT_PROBES)
        data = {
            "TCGA-06-0001-01A": [0.8, 0.7, 0.9, 0.75, 0.8, 0.85],
            "TCGA-06-0002-01A": [0.1, 0.05, 0.2, 0.1, 0.15, 0.1],
        }
        beta = pd.DataFrame(data, index=probes)
        summary = methylation.summarise_mgmt(beta)
        assert len(summary) == 2
        assert summary.loc[summary["sample"] == "TCGA-06-0001-01A", "mgmt_methylated"].iloc[0] == True   # noqa: E712
        assert summary.loc[summary["sample"] == "TCGA-06-0002-01A", "mgmt_methylated"].iloc[0] == False  # noqa: E712

    def test_summarise_mgmt_no_probes(self):
        beta = pd.DataFrame({"sample1": [0.5]}, index=["cg99999999"])
        summary = methylation.summarise_mgmt(beta)
        assert summary.empty


# ── mutations ─────────────────────────────────────────────────────────────────

class TestMutationsParser:
    def _make_maf(self, tmp_path: Path, rows: list[dict]) -> Path:
        import pandas as pd
        df = pd.DataFrame(rows)
        p = tmp_path / "test.maf"
        p.write_text(df.to_csv(sep="\t", index=False))
        return tmp_path

    def test_parse_maf_filters_pass(self, tmp_path):
        rows = [
            {"Hugo_Symbol": "IDH1", "Tumor_Sample_Barcode": "TCGA-06-0001-01A",
             "FILTER": "PASS", "Variant_Classification": "Missense_Mutation",
             "Variant_Type": "SNP", "HGVSp_Short": "p.R132H",
             "HGVSc": "c.394C>T", "IMPACT": "HIGH",
             "Chromosome": "2", "Start_Position": "209113112", "End_Position": "209113112",
             "Strand": "+", "Reference_Allele": "C", "Tumor_Seq_Allele1": "C",
             "Tumor_Seq_Allele2": "T", "t_depth": "50", "t_ref_count": "25",
             "t_alt_count": "25", "n_depth": "40", "n_ref_count": "40",
             "n_alt_count": "0", "BIOTYPE": "protein_coding"},
            {"Hugo_Symbol": "TP53", "Tumor_Sample_Barcode": "TCGA-06-0001-01A",
             "FILTER": "clustered_events", "Variant_Classification": "Nonsense_Mutation",
             "Variant_Type": "SNP", "HGVSp_Short": "p.R248W",
             "HGVSc": "c.742C>T", "IMPACT": "HIGH",
             "Chromosome": "17", "Start_Position": "7674220", "End_Position": "7674220",
             "Strand": "+", "Reference_Allele": "C", "Tumor_Seq_Allele1": "C",
             "Tumor_Seq_Allele2": "T", "t_depth": "30", "t_ref_count": "15",
             "t_alt_count": "15", "n_depth": "25", "n_ref_count": "25",
             "n_alt_count": "0", "BIOTYPE": "protein_coding"},
        ]
        maf_dir = self._make_maf(tmp_path, rows)
        result = mutations.parse_maf_files(maf_dir, verbose=False)
        # Only PASS variant should survive
        assert len(result) == 1
        assert result.iloc[0]["Hugo_Symbol"] == "IDH1"

    def test_idh_status_extraction(self):
        df = pd.DataFrame({
            "Hugo_Symbol":          ["IDH1", "TP53"],
            "Tumor_Sample_Barcode": ["TCGA-06-0001-01A", "TCGA-06-0002-01A"],
            "HGVSp_Short":          ["p.R132H", "p.R248W"],
            "FILTER":               ["PASS", "PASS"],
        })
        status = mutations.extract_idh_status(df)
        row1 = status[status["sample"] == "TCGA-06-0001-01A"].iloc[0]
        row2 = status[status["sample"] == "TCGA-06-0002-01A"].iloc[0]
        assert row1["IDH_status"] == "IDH_mutant"
        assert row2["IDH_status"] == "IDH_wildtype"

    def test_driver_matrix_shape(self):
        from gbm_multiomics.constants import GBM_DRIVER_GENES
        df = pd.DataFrame({
            "Hugo_Symbol":          ["IDH1", "EGFR", "IDH1"],
            "Tumor_Sample_Barcode": ["S1", "S1", "S2"],
            "FILTER": ["PASS"] * 3,
            "HGVSp_Short": ["p.R132H", "p.A289V", "p.R132H"],
        })
        mat = mutations.build_driver_matrix(df)
        assert "IDH1" in mat.index
        assert mat.loc["IDH1", "S1"] == 1
        assert mat.loc["IDH1", "S2"] == 1
        assert mat.loc["EGFR", "S1"] == 1


# ── cnv ───────────────────────────────────────────────────────────────────────

class TestCNVParser:
    def test_summarise_chr7_chr10_flags(self):
        import pandas as pd
        segments = pd.DataFrame({
            "sample_submitter_id": ["S1", "S1", "S2", "S2"],
            "Chromosome":          ["7", "10", "7", "10"],
            "Segment_Mean":        [0.8, -0.9, -0.1, 0.1],
        })
        result = cnv.summarise_chr7_chr10(segments)
        s1 = result[result["sample"] == "S1"].iloc[0]
        s2 = result[result["sample"] == "S2"].iloc[0]
        assert s1["chr7_gain"]  == True   # noqa: E712
        assert s1["chr10_loss"] == True   # noqa: E712
        assert s1["gbm_chr_pattern"] == True   # noqa: E712
        assert s2["gbm_chr_pattern"] == False  # noqa: E712

    def test_chr7_chr10_missing_data(self):
        import pandas as pd
        segments = pd.DataFrame({
            "sample_submitter_id": ["S1"],
            "Chromosome":          ["5"],
            "Segment_Mean":        [0.5],
        })
        result = cnv.summarise_chr7_chr10(segments)
        assert result.empty or result["gbm_chr_pattern"].isna().all() \
               or not result["gbm_chr_pattern"].iloc[0]


# ── mirna ─────────────────────────────────────────────────────────────────────

class TestMirnaParser:
    def test_build_rpm_matrix(self, tmp_path):
        import pandas as pd

        meta = pd.DataFrame({
            "file_id":             ["uuid-1", "uuid-2"],
            "file_name":           ["s1_mirna.txt", "s2_mirna.txt"],
            "sample_submitter_id": ["TCGA-06-0001-01A", "TCGA-06-0002-01A"],
            "is_tumor":            [True, True],
        })

        for fname, sample in [("s1_mirna.txt", "TCGA-06-0001-01A"),
                               ("s2_mirna.txt", "TCGA-06-0002-01A")]:
            uid_dir = tmp_path / (
                "uuid-1" if fname == "s1_mirna.txt" else "uuid-2"
            )
            uid_dir.mkdir(exist_ok=True)
            txt = uid_dir / fname
            txt.write_text(
                "miRNA_ID\tread_count\treads_per_million_miRNA_mapped\tcross-mapped\n"
                "hsa-mir-21\t1000\t5000.0\tN\n"
                "hsa-mir-10b\t200\t1000.0\tN\n"
                "hsa-mir-cross\t50\t250.0\tY\n"  # cross-mapped, should be excluded
            )

        matrix = mirna.build_rpm_matrix(tmp_path, meta, verbose=False)
        assert "hsa-mir-21" in matrix.index
        assert "hsa-mir-10b" in matrix.index
        assert "hsa-mir-cross" not in matrix.index
        assert matrix.shape[1] == 2
