"""
network.py — Protein-protein interaction network analysis for GBM prognostic genes.

Builds PPI networks from STRING DB for prognostic gene sets,
identifies hub genes, and performs module detection.

References
----------
  Szklarczyk et al. (2023) NAR 51:D638-D646 — STRING v12
  Langfelder & Horvath (2008) BMC Bioinformatics 9:559 — WGCNA
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_ppi_network(
    genes: list[str],
    species: int = 9606,  # human
    score_threshold: int = 700,
    output_dir: Path | None = None,
) -> dict:
    """
    Query STRING database and build a PPI network.

    Parameters
    ----------
    genes : list[str]
        HGNC gene symbols.
    species : int
        NCBI taxonomy ID (9606 = human).
    score_threshold : int
        STRING combined score threshold (0-1000).
    output_dir : Path, optional

    Returns
    -------
    dict
        {edges: DataFrame, nodes: DataFrame, graph: networkx.Graph}
    """
    import io

    string_url = "https://string-db.org/api/tsv/network"

    print(f"  🌐  STRING PPI: {len(genes)} genes, score ≥ {score_threshold}.")

    try:
        import requests

        resp = requests.get(
            string_url,
            params={
                "identifiers": "\r".join(genes),
                "species": species,
                "required_score": score_threshold,
            },
            timeout=60,
        )

        if resp.status_code != 200:
            print(f"  ⚠   STRING API error (HTTP {resp.status_code}).")
            return {"edges": pd.DataFrame(), "nodes": pd.DataFrame()}

        edges = pd.read_csv(io.StringIO(resp.text), sep="\t")

    except Exception as exc:
        print(f"  ⚠   STRING API failed: {exc}")
        return {"edges": pd.DataFrame(), "nodes": pd.DataFrame()}

    if edges.empty:
        print("  ⚠   No interactions found at this score threshold.")
        return {"edges": pd.DataFrame(), "nodes": pd.DataFrame()}

    # Build node list
    nodes_in = set(edges["preferredName_A"].unique())
    nodes_out = set(edges["preferredName_B"].unique())
    all_nodes = nodes_in | nodes_out

    nodes = pd.DataFrame({
        "gene": sorted(all_nodes),
    })
    nodes["is_seed"] = nodes["gene"].isin(genes)

    n_seeds_found = nodes["is_seed"].sum()
    print(f"  ✅  STRING network: {len(edges)} edges, {len(nodes)} nodes "
          f"({n_seeds_found}/{len(genes)} seeds connected).")

    # Build networkx graph
    try:
        import networkx as nx

        G = nx.Graph()
        G.add_nodes_from(nodes["gene"].tolist())
        for _, row in edges.iterrows():
            G.add_edge(
                row["preferredName_A"], row["preferredName_B"],
                weight=row.get("score", 0) / 1000,
            )

        # Compute centrality
        degree_cent = pd.Series(nx.degree_centrality(G)).sort_values(ascending=False)
        between_cent = pd.Series(nx.betweenness_centrality(G)).sort_values(ascending=False)

        nodes["degree_centrality"] = nodes["gene"].map(degree_cent).fillna(0)
        nodes["betweenness_centrality"] = nodes["gene"].map(between_cent).fillna(0)
        nodes = nodes.sort_values("degree_centrality", ascending=False)

    except ImportError:
        G = None
        print("  ℹ   networkx not available. Centrality not computed.")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        edges.to_csv(output_dir / "ppi_edges.tsv", sep="\t", index=False)
        nodes.to_csv(output_dir / "ppi_nodes.tsv", sep="\t", index=False)

    return {
        "edges": edges,
        "nodes": nodes,
        "graph": G,
    }


def identify_hub_genes(
    network: dict,
    n_top: int = 20,
    min_degree_centrality: float = 0.05,
) -> pd.DataFrame:
    """
    Identify hub genes from PPI network centrality.

    Parameters
    ----------
    network : dict
        From build_ppi_network().
    n_top : int
        Number of top hub genes to return.
    min_degree_centrality : float
        Minimum degree centrality to be considered a hub.

    Returns
    -------
    pd.DataFrame
        gene | degree_centrality | betweenness_centrality | is_hub | is_seed
    """
    nodes = network.get("nodes", pd.DataFrame())
    if nodes.empty:
        return pd.DataFrame()

    hubs = nodes[
        (nodes["degree_centrality"] >= min_degree_centrality)
    ].head(n_top).copy()

    hubs["is_hub"] = True
    if hubs.empty:
        print("  ℹ   No hub genes identified at the given threshold.")
    else:
        print(f"  🧬  Hub genes: {len(hubs)} identified.")
        for _, row in hubs.iterrows():
            print(f"       {row['gene']}: DC={row['degree_centrality']:.3f}, "
                  f"BC={row['betweenness_centrality']:.3f} "
                  f"{'(seed)' if row['is_seed'] else ''}")

    return hubs


def detect_modules(
    graph,
    min_module_size: int = 5,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Detect communities/modules in PPI network using Louvain clustering.

    Parameters
    ----------
    graph : networkx.Graph
    min_module_size : int
    output_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        gene | module | module_size
    """
    if graph is None:
        return pd.DataFrame()

    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(graph)

        modules = pd.DataFrame({
            "gene": list(partition.keys()),
            "module": [f"Module_{m + 1}" for m in partition.values()],
        })
        modules["module_size"] = modules.groupby("module")["gene"].transform("count")

        # Filter small modules
        modules = modules[modules["module_size"] >= min_module_size]

        print(f"  🧬  Modules: {modules['module'].nunique()} detected "
              f"(size ≥ {min_module_size}).")
        for mod, size in modules["module"].value_counts().items():
            print(f"       {mod}: {size} genes")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            modules.to_csv(output_dir / "ppi_modules.tsv", sep="\t", index=False)

        return modules

    except ImportError:
        print("  ℹ   python-louvain not available. Skipping community detection.")
        return pd.DataFrame()


def coexpression_network(
    expr: pd.DataFrame,
    min_correlation: float = 0.6,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build a gene co-expression network (WGCNA-like).

    Computes pairwise Pearson correlations, thresholded at min_correlation.
    For large gene sets, use top variable genes only.

    Parameters
    ----------
    expr : pd.DataFrame
        Genes × samples, normalized expression.
    min_correlation : float
        Minimum absolute Pearson r for an edge.
    output_dir : Path, optional

    Returns
    -------
    pd.DataFrame
        source | target | weight (edges of the co-expression network).
    """
    n_genes = len(expr)
    if n_genes > 2000:
        # Use top variable genes
        var = expr.var(axis=1)
        top_genes = var.nlargest(2000).index
        expr = expr.loc[top_genes]
        print(f"  ℹ   Using top {len(expr)} variable genes for co-expression.")

    # Compute correlation matrix
    corr = expr.T.corr(method="pearson")

    # Extract edges above threshold
    edges_list = []
    for i, gene_a in enumerate(corr.index):
        for gene_b in corr.columns[i + 1:]:
            r = corr.loc[gene_a, gene_b]
            if abs(r) >= min_correlation:
                edges_list.append({
                    "source": gene_a,
                    "target": gene_b,
                    "weight": round(abs(r), 4),
                    "direction": "positive" if r > 0 else "negative",
                })

    edges = pd.DataFrame(edges_list)
    if edges.empty:
        print(f"  ⚠   No edges at |r| ≥ {min_correlation}.")
        return edges

    print(f"  🧬  Co-expression network: {len(edges):,} edges "
          f"at |r| ≥ {min_correlation} "
          f"({len(expr)} genes, {expr.shape[1]} samples).")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        edges.to_csv(output_dir / "coexpression_network.tsv", sep="\t", index=False)

    return edges
