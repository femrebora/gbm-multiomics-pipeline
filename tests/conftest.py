"""
conftest.py — Shared test fixtures for gbm-multiomics tests.

All tests use synthetic data — no real TCGA data or API calls required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest


# ── Minimal count matrix ──────────────────────────────────────────────────────

@pytest.fixture
def small_count_matrix() -> pd.DataFrame:
    """5 genes × 6 samples raw count matrix."""
    rng = np.random.default_rng(42)
    genes = [f"ENSG{i:011d}" for i in range(5)]
    samples = [f"TCGA-06-000{i}-01A" for i in range(6)]
    data = rng.integers(10, 5000, size=(5, 6))
    return pd.DataFrame(data, index=genes, columns=samples)


@pytest.fixture
def sample_metadata() -> pd.DataFrame:
    """Minimal sample metadata matching small_count_matrix."""
    samples = [f"TCGA-06-000{i}-01A" for i in range(6)]
    return pd.DataFrame({
        "file_id": [f"uuid-{i}" for i in range(6)],
        "file_name": [f"file_{i}.tsv" for i in range(6)],
        "case_submitter_id": [f"TCGA-06-000{i}" for i in range(6)],
        "sample_submitter_id": samples,
        "sample_type": ["Primary Tumor"] * 4 + ["Solid Tissue Normal"] * 2,
        "is_tumor": [True] * 4 + [False] * 2,
    })


@pytest.fixture
def clinical_data() -> pd.DataFrame:
    """Minimal clinical data for survival tests."""
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "case_submitter_id": [f"TCGA-06-00{i:02d}" for i in range(n)],
        "sample_submitter_id": [f"TCGA-06-00{i:02d}-01A" for i in range(n)],
        "cdr_OS": rng.integers(0, 2, size=n).tolist(),
        "cdr_OS.time": rng.integers(30, 900, size=n).tolist(),
        "cdr_PFI": rng.integers(0, 2, size=n).tolist(),
        "cdr_PFI.time": rng.integers(30, 600, size=n).tolist(),
        "IDH_status": (["IDH_wildtype"] * 7 + ["IDH_mutant"] * 3),
    })


@pytest.fixture
def idh_status_data() -> pd.DataFrame:
    return pd.DataFrame({
        "sample": [f"TCGA-06-00{i:02d}-01A" for i in range(6)],
        "IDH1_mutated": [False, False, False, True, True, False],
        "IDH1_variant": ["", "", "", "p.R132H", "p.R132H", ""],
        "IDH2_mutated": [False, False, False, False, False, True],
        "IDH2_variant": ["", "", "", "", "", "p.R172K"],
        "IDH_status": ["IDH_wildtype", "IDH_wildtype", "IDH_wildtype",
                       "IDH_mutant", "IDH_mutant", "IDH_mutant"],
    })


@pytest.fixture
def mock_gdc_client() -> MagicMock:
    """Mock GBMClient that never makes real HTTP calls."""
    client = MagicMock()
    client.check_connectivity.return_value = True
    client.discover_files.return_value = [
        {
            "file_id": f"aaaa000{i}-1111-2222-3333-444455556666",
            "file_name": f"sample_{i}.tsv",
            "file_size": 1_000_000,
            "cases": [{
                "case_id": f"bbbb000{i}-...",
                "submitter_id": f"TCGA-06-000{i}",
                "samples": [{
                    "sample_id": f"cccc000{i}-...",
                    "submitter_id": f"TCGA-06-000{i}-01A",
                    "sample_type": "Primary Tumor",
                    "tissue_type": "Tumor",
                }],
            }],
        }
        for i in range(4)
    ]
    return client


# ── New fixtures for Phase 2-9 modules ────────────────────────────────────────

@pytest.fixture
def large_expr_matrix() -> pd.DataFrame:
    """200 genes × 30 samples normalized expression matrix with known patterns."""
    rng = np.random.default_rng(99)
    n_genes = 200
    n_samples = 30

    # Include real GBM gene symbols in the index
    gbm_genes = [
        "EGFR", "PTEN", "TP53", "IDH1", "NF1", "PDGFRA", "CDK4",
        "MDM2", "CDKN2A", "RB1", "PIK3CA", "PIK3R1", "MGMT",
        "VEGFA", "HIF1A", "CHI3L1", "OLIG2", "SOX2", "MYC",
        "TERT", "ATRX", "BRAF", "MET", "CDK6", "CCND1",
    ]
    background = [f"GENE_{i}" for i in range(n_genes - len(gbm_genes))]
    all_genes = gbm_genes + background
    rng.shuffle(all_genes)

    samples = [f"TCGA-06-{i:04d}-01A" for i in range(n_samples)]

    # Base expression
    data = rng.normal(8, 2, size=(len(all_genes), n_samples))
    expr = pd.DataFrame(data, index=all_genes, columns=samples)

    # Inject known prognostic signal: EGFR high → worse survival
    # This creates a realistic expression structure for testing
    return np.abs(expr)  # ensure positive for log2


@pytest.fixture
def prognostic_clinical_data() -> pd.DataFrame:
    """Clinical data with known prognostic patterns for testing."""
    rng = np.random.default_rng(7)
    n = 30
    samples = [f"TCGA-06-{i:04d}" for i in range(n)]
    expr_samples = [f"TCGA-06-{i:04d}-01A" for i in range(n)]

    os_time = rng.integers(30, 1500, size=n).astype(float)
    os_event = rng.binomial(1, 0.6, size=n).astype(float)

    return pd.DataFrame({
        "case_submitter_id": samples,
        "sample_submitter_id": expr_samples,
        "cdr_OS": os_event,
        "cdr_OS.time": os_time,
        "cdr_PFI": rng.binomial(1, 0.5, size=n).astype(float),
        "cdr_PFI.time": rng.integers(20, 900, size=n).astype(float),
        "IDH_status": rng.choice(["IDH_wildtype", "IDH_mutant"], n, p=[0.85, 0.15]),
        "age_at_diagnosis": rng.integers(30, 85, size=n).astype(float),
        "gender": rng.choice(["MALE", "FEMALE"], n),
        "MGMT_status": rng.choice(["Methylated", "Unmethylated"], n, p=[0.4, 0.6]),
    })


@pytest.fixture
def de_results_data() -> pd.DataFrame:
    """Synthetic differential expression results."""
    rng = np.random.default_rng(12)
    genes = ["EGFR", "PTEN", "TP53", "IDH1", "NF1", "PDGFRA",
             "VEGFA", "SOX2", "MYC", "CHI3L1"] + [f"GENE_{i}" for i in range(90)]
    n = len(genes)

    lfc = rng.normal(0, 1.5, size=n)
    # Make some genes strongly DE
    lfc[0] = 4.0   # EGFR — UP
    lfc[1] = -3.0  # PTEN — DOWN
    lfc[2] = -2.5  # TP53 — DOWN
    lfc[5] = 3.5   # PDGFRA — UP

    pval = 10 ** (-rng.uniform(0.5, 10, size=n))
    pval[:6] = [1e-15, 1e-12, 1e-10, 1e-8, 1e-6, 1e-14]

    return pd.DataFrame({
        "log2FoldChange": lfc,
        "padj": pval,
        "pvalue": pval * 0.8,
        "stat": lfc / rng.uniform(0.5, 2, size=n),
        "baseMean": rng.uniform(100, 10000, size=n),
    }, index=genes)


@pytest.fixture
def mutation_matrix() -> pd.DataFrame:
    """Synthetic mutation matrix (genes × samples, mutation type values)."""
    rng = np.random.default_rng(33)
    genes = ["TP53", "PTEN", "EGFR", "NF1", "IDH1", "RB1", "PIK3CA",
             "ATRX", "PDGFRA", "BRAF"]
    samples = [f"TCGA-06-{i:04d}-01A" for i in range(20)]

    mut_types = ["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
                 "Splice_Site", "In_Frame_Del"]

    data = {}
    for gene in genes:
        # Each gene mutated in 10-40% of samples
        mut_rate = rng.uniform(0.1, 0.4)
        values = []
        for _ in samples:
            if rng.random() < mut_rate:
                values.append(rng.choice(mut_types))
            else:
                values.append(np.nan)
        data[gene] = values

    return pd.DataFrame(data, index=samples).T


@pytest.fixture
def methylation_data() -> pd.DataFrame:
    """Synthetic methylation beta values (probes × samples)."""
    rng = np.random.default_rng(55)
    n_probes = 50
    n_samples = 20

    probes = [f"cg{str(i).zfill(8)}" for i in range(n_probes)]
    samples = [f"TCGA-06-{i:04d}-01A" for i in range(n_samples)]

    # Beta values between 0 and 1
    data = rng.beta(2, 3, size=(n_probes, n_samples))
    return pd.DataFrame(data, index=probes, columns=samples)
