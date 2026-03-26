"""
conftest.py — Shared test fixtures for gbm-multiomics tests.
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
    genes   = [f"ENSG{i:011d}" for i in range(5)]
    samples = [f"TCGA-06-000{i}-01A" for i in range(6)]
    data = rng.integers(10, 5000, size=(5, 6))
    return pd.DataFrame(data, index=genes, columns=samples)


@pytest.fixture
def sample_metadata() -> pd.DataFrame:
    """Minimal sample metadata matching small_count_matrix."""
    samples = [f"TCGA-06-000{i}-01A" for i in range(6)]
    return pd.DataFrame({
        "file_id":             [f"uuid-{i}" for i in range(6)],
        "file_name":           [f"file_{i}.tsv" for i in range(6)],
        "case_submitter_id":   [f"TCGA-06-000{i}" for i in range(6)],
        "sample_submitter_id": samples,
        "sample_type":         ["Primary Tumor"] * 4 + ["Solid Tissue Normal"] * 2,
        "is_tumor":            [True] * 4 + [False] * 2,
    })


@pytest.fixture
def clinical_data() -> pd.DataFrame:
    """Minimal clinical data for survival tests."""
    rng = np.random.default_rng(0)
    n = 10
    return pd.DataFrame({
        "case_submitter_id": [f"TCGA-06-00{i:02d}" for i in range(n)],
        "sample_submitter_id": [f"TCGA-06-00{i:02d}-01A" for i in range(n)],
        "cdr_OS":            rng.integers(0, 2, size=n).tolist(),
        "cdr_OS.time":       rng.integers(30, 900, size=n).tolist(),
        "cdr_PFI":           rng.integers(0, 2, size=n).tolist(),
        "cdr_PFI.time":      rng.integers(30, 600, size=n).tolist(),
        "IDH_status":        (["IDH_wildtype"] * 7 + ["IDH_mutant"] * 3),
    })


@pytest.fixture
def idh_status_data() -> pd.DataFrame:
    return pd.DataFrame({
        "sample":       [f"TCGA-06-00{i:02d}-01A" for i in range(6)],
        "IDH1_mutated": [False, False, False, True, True, False],
        "IDH1_variant": ["", "", "", "p.R132H", "p.R132H", ""],
        "IDH2_mutated": [False, False, False, False, False, True],
        "IDH2_variant": ["", "", "", "", "", "p.R172K"],
        "IDH_status":   ["IDH_wildtype", "IDH_wildtype", "IDH_wildtype",
                         "IDH_mutant",   "IDH_mutant",   "IDH_mutant"],
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
                "case_id":    f"bbbb000{i}-...",
                "submitter_id": f"TCGA-06-000{i}",
                "samples": [{
                    "sample_id":    f"cccc000{i}-...",
                    "submitter_id": f"TCGA-06-000{i}-01A",
                    "sample_type":  "Primary Tumor",
                    "tissue_type":  "Tumor",
                }],
            }],
        }
        for i in range(4)
    ]
    return client
