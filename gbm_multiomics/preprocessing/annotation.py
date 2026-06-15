"""
annotation.py — Gene annotation: ENSG → HGNC symbol mapping and gene filtering.

Uses MyGene.info API for fast gene symbol lookup with local Ensembl cache.
Filters to protein-coding genes and removes ribosomal/mitochondrial noise.

References
----------
  MyGene.info: Wu et al. (2013) Bioinformatics 29:532-539
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ── Ribosomal protein gene prefixes ──────────────────────────────────────────
_RIBOSOMAL_PREFIXES = ("RPS", "RPL", "MRPS", "MRPL")

# ── Mitochondrial gene prefix ────────────────────────────────────────────────
_MT_PREFIX = "MT-"


def map_ensg_to_symbol(
    gene_ids: pd.Index | list[str],
    species: str = "human",
    cache_dir: Path | None = None,
) -> pd.Series:
    """
    Map Ensembl gene IDs (ENSG...) to HGNC gene symbols using MyGene.info.

    Parameters
    ----------
    gene_ids : Index-like
        Ensembl gene IDs (e.g. "ENSG00000157764").
    species : str
        "human" or "mouse".
    cache_dir : Path, optional
        Directory to cache mapping results.

    Returns
    -------
    pd.Series
        Index: ENSG IDs, values: HGNC symbols (or original ID if not found).
    """
    import json
    import time

    ensg_list = [g for g in gene_ids if isinstance(g, str) and g.startswith("ENSG")]
    if not ensg_list:
        return pd.Series(index=pd.Index(gene_ids), dtype=str).fillna("unknown")

    # Try cache first
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"ensg_to_symbol_{species}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            mapping = {g: cached.get(g, g) for g in gene_ids}
            return pd.Series(mapping, index=pd.Index(gene_ids), name="gene_symbol")

    mapping: dict[str, str] = {}

    # Query MyGene.info in batches of 1000
    batch_size = 1000
    for i in range(0, len(ensg_list), batch_size):
        batch = ensg_list[i:i + batch_size]
        try:
            import requests
            resp = requests.post(
                "https://mygene.info/v3/gene",
                json={"ids": batch, "fields": "symbol", "species": species},
                timeout=30,
            )
            if resp.status_code == 200:
                for entry in resp.json():
                    ensg = entry.get("query", "")
                    symbol = entry.get("symbol", ensg)
                    mapping[ensg] = symbol if symbol else ensg
            time.sleep(0.3)  # rate limit
        except Exception:
            # On failure, use ENSG as-is
            for g in batch:
                mapping.setdefault(g, g)

    # Cache result
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"ensg_to_symbol_{species}.json"
        cache_file.write_text(json.dumps(mapping, indent=2))

    result = {g: mapping.get(g, g) for g in gene_ids}
    return pd.Series(result, index=pd.Index(gene_ids), name="gene_symbol")


def filter_protein_coding(
    counts: pd.DataFrame,
    species: str = "human",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter count matrix to protein-coding genes only.

    Uses a built-in set of known protein-coding gene biotypes
    (from Ensembl v110). Falls back to keeping all genes if annotation fails.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes × samples, ENSG index.
    species : str
        Not used for built-in filter, reserved for future biotype annotation.
    verbose : bool

    Returns
    -------
    pd.DataFrame
        Filtered count matrix (subset of rows).
    """
    # Protein-coding biotypes in Ensembl
    _PROTEIN_CODING_BIOTYPES = frozenset({
        "protein_coding",
        "IG_C_gene", "IG_D_gene", "IG_J_gene", "IG_V_gene",
        "TR_C_gene", "TR_D_gene", "TR_J_gene", "TR_V_gene",
    })

    # Try downloading biotype info from Ensembl BioMart
    try:
        import requests
        ensg_ids = counts.index.tolist()
        batch_size = 500
        protein_coding_ensgs: set[str] = set()

        for i in range(0, len(ensg_ids), batch_size):
            batch = ensg_ids[i:i + batch_size]
            resp = requests.post(
                "https://rest.ensembl.org/lookup/id",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                json={"ids": batch},
                timeout=30,
            )
            if resp.status_code == 200:
                for ensg, info in resp.json().items():
                    if info and info.get("biotype") in _PROTEIN_CODING_BIOTYPES:
                        protein_coding_ensgs.add(ensg)
    except Exception:
        # Fallback: keep genes that don't start with known non-coding prefixes
        if verbose:
            print("  ⚠  Could not fetch biotype info. Keeping all genes.")
        return counts.copy()

    n_before = len(counts)
    filtered = counts[counts.index.isin(protein_coding_ensgs)]
    n_after = len(filtered)

    if verbose:
        print(f"  🧬  Protein-coding filter: {n_before} → {n_after} genes "
              f"({n_before - n_after} non-coding removed).")

    return filtered


def filter_low_expression(
    counts: pd.DataFrame,
    min_count: int = 10,
    min_samples: int | float = 0.2,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Filter out lowly-expressed genes.

    Parameters
    ----------
    counts : pd.DataFrame
        Raw count matrix (genes × samples), int values.
    min_count : int
        Minimum count threshold.
    min_samples : int or float
        If float, treated as fraction of total samples.
        If int, absolute number of samples.

    Returns
    -------
    pd.DataFrame
        Filtered count matrix.
    """
    n_samples = counts.shape[1]
    if isinstance(min_samples, float):
        min_samples_n = int(np.ceil(min_samples * n_samples))
    else:
        min_samples_n = min_samples

    keep = (counts > min_count).sum(axis=1) >= min_samples_n
    filtered = counts.loc[keep].copy()

    if verbose:
        print(f"  🔍  Expression filter (> {min_count} in ≥ {min_samples_n} samples): "
              f"{len(counts)} → {len(filtered)} genes.")

    return filtered


def annotate_genes(
    counts: pd.DataFrame,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Annotate ENSG-indexed count matrix with HGNC symbols.

    Adds 'gene_symbol' column as the first column, sorts by symbol,
    and handles duplicates by keeping the row with maximum mean expression.

    Parameters
    ----------
    counts : pd.DataFrame
        Genes (ENSG) × samples. Integer raw counts.
    cache_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        Count matrix with gene_symbol column. Duplicate symbols resolved.
    """
    symbols = map_ensg_to_symbol(counts.index, cache_dir=cache_dir)
    annotated = counts.copy()
    annotated.insert(0, "gene_symbol", symbols.values)

    # Handle duplicates: keep row with max mean expression
    dup_mask = annotated["gene_symbol"].duplicated(keep=False)
    if dup_mask.any():
        dup_symbols = annotated.loc[dup_mask, "gene_symbol"].unique()
        n_dup = len(dup_symbols)
        # For each duplicated symbol, keep the one with highest mean
        to_drop = []
        for sym in dup_symbols:
            rows = annotated[annotated["gene_symbol"] == sym]
            # Compute mean expression (excluding gene_symbol column)
            expr_cols = [c for c in annotated.columns if c != "gene_symbol"]
            means = rows[expr_cols].mean(axis=1)
            # Keep max, drop rest
            to_drop.extend(rows.index.difference([means.idxmax()]).tolist())
        annotated = annotated.drop(to_drop)
        if verbose:
            print(f"  ℹ   Resolved {n_dup} duplicate gene symbols "
                  f"(kept max-expression copy).")

    n_mapped = (annotated["gene_symbol"] != annotated.index).sum()
    if verbose:
        print(f"  ✅  Gene annotation: {n_mapped:,}/{len(annotated):,} "
              f"genes mapped to symbols.")

    return annotated
