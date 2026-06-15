"""
multiomics.py — Multi-omics integration for TCGA-GBM.

Integrates RNA-seq, methylation, CNV, and mutation data to find genes
consistently altered across omics layers.

Methods
-------
  cross_omics_correlation  — Pairwise correlation between omics layers
  run_mofa                 — Multi-Omics Factor Analysis (MOFA2)
  snf_cluster              — Similarity Network Fusion clustering
  integrated_prognostic    — Build prognostic model from multi-omics features

References
----------
  Argelaguet et al. (2018) Mol Syst Biol 14:e8124 — MOFA
  Wang et al. (2014) Nat Methods 11:333-337 — SNF
  Ceccarelli et al. (2016) Cell 164:550-563 — GBM multi-omics
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def cross_omics_correlation(
    rna_expr: pd.DataFrame,
    cnv: pd.DataFrame | None = None,
    methylation: pd.DataFrame | None = None,
    mutations: pd.DataFrame | None = None,
    mirna_expr: pd.DataFrame | None = None,
    gene_symbol_map: dict | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Compute cross-omics pairwise correlations.

    Parameters
    ----------
    rna_expr : pd.DataFrame
        Genes × samples, normalized RNA-seq expression.
    cnv : pd.DataFrame, optional
        Genes × samples, copy number values (log2 ratio or GISTIC scores).
    methylation : pd.DataFrame, optional
        Probes × samples, beta values.
    mutations : pd.DataFrame, optional
        Genes × samples, binary mutation matrix (0/1).
    mirna_expr : pd.DataFrame, optional
        miRNAs × samples, normalized miRNA expression.
    gene_symbol_map : dict, optional
        {gene_id: gene_symbol} mapping (for merging across data types).
    output_dir : Path, optional

    Returns
    -------
    dict[str, pd.DataFrame]
        Each key is an omics pair (e.g. "rna_cnv"), value is correlation results:
        gene | pearson_r | spearman_r | p_value
    """
    results: dict[str, pd.DataFrame] = {}

    # ── RNA ~ CNV cis-correlation ────────────────────────────────────────
    if cnv is not None:
        common_genes = sorted(set(rna_expr.index) & set(cnv.index))
        common_samples = sorted(set(rna_expr.columns) & set(cnv.columns))

        if len(common_genes) > 10 and len(common_samples) > 10:
            print(f"  🧬  RNA ~ CNV: {len(common_genes)} genes "
                  f"× {len(common_samples)} samples.")

            rows = []
            for gene in common_genes:
                rna_vals = rna_expr.loc[gene, common_samples].astype(float)
                cnv_vals = cnv.loc[gene, common_samples].astype(float)
                valid = rna_vals.notna() & cnv_vals.notna()
                if valid.sum() < 10:
                    continue
                pearson = rna_vals[valid].corr(cnv_vals[valid], method="pearson")
                spearman = rna_vals[valid].corr(cnv_vals[valid], method="spearman")
                rows.append({
                    "gene": gene,
                    "pearson_r": round(pearson, 4),
                    "spearman_r": round(spearman, 4),
                    "n": int(valid.sum()),
                })

            corr_df = pd.DataFrame(rows).sort_values("pearson_r", ascending=False)
            results["rna_cnv"] = corr_df

            # Summary
            n_cis = (corr_df["pearson_r"] > 0.3).sum()
            print(f"  ✅  RNA-CNV: {n_cis} genes with r > 0.3 "
                  f"(median r = {corr_df['pearson_r'].median():.3f}).")

    # ── RNA ~ Methylation (promoter) ─────────────────────────────────────
    if methylation is not None:
        common_samples = sorted(set(rna_expr.columns) & set(methylation.columns))
        if len(common_samples) > 10:
            print(f"  🧬  RNA ~ Methylation: {len(methylation.index)} probes "
                  f"× {len(common_samples)} samples.")

            # Simplified: correlate each probe (promoter) with all genes
            # In practice, would need probe-to-gene mapping
            # For now, compute probe-gene correlation matrix (sampled)
            n_probes = min(len(methylation.index), 1000)
            probe_sample = np.random.default_rng(42).choice(
                methylation.index, n_probes, replace=False,
            )

            meth_sub = methylation.loc[probe_sample, common_samples].astype(float)
            rna_sub = rna_expr.loc[rna_expr.index[:500], common_samples].astype(float)

            # Compute correlation for each probe with each gene (can be large)
            # Use a fast matrix correlation
            corr_matrix = pd.DataFrame(
                np.corrcoef(meth_sub.values, rna_sub.values)[:n_probes, n_probes:],
                index=probe_sample,
                columns=rna_sub.index,
            )

            results["rna_methylation_corr"] = corr_matrix

            # Top anti-correlations (methylation → silencing)
            top_neg = corr_matrix.stack().nsmallest(20)
            print(f"  ✅  RNA-Methylation: top anti-correlations "
                  f"(median r = {corr_matrix.stack().median():.3f}).")

    # ── Mutation → Expression ────────────────────────────────────────────
    if mutations is not None:
        common_genes = sorted(set(rna_expr.index) & set(mutations.index))
        common_samples = sorted(set(rna_expr.columns) & set(mutations.columns))

        if len(common_genes) > 5 and len(common_samples) > 10:
            common_samples_arr = np.array(common_samples)
            rows = []
            for gene in common_genes:
                mut_vals = pd.to_numeric(
                    mutations.loc[gene, common_samples], errors="coerce"
                )
                mut_samples_mask = mut_vals > 0
                mut_samples_raw = list(common_samples_arr[mut_samples_mask.values])
                wt_samples = [s for s in common_samples if s not in mut_samples_raw]

                if len(mut_samples_raw) < 3 or len(wt_samples) < 3:
                    continue

                from scipy.stats import mannwhitneyu
                mut_expr = rna_expr.loc[gene, mut_samples_raw].astype(float)
                wt_expr = rna_expr.loc[gene, wt_samples].astype(float)

                lfc = mut_expr.mean() - wt_expr.mean()
                stat, p = mannwhitneyu(mut_expr, wt_expr, alternative="two-sided")

                rows.append({
                    "gene": gene,
                    "log2FC": round(lfc, 4),
                    "n_mutated": len(mut_samples_raw),
                    "n_wt": len(wt_samples),
                    "p_value": round(p, 6),
                })

            if rows:
                mut_df = pd.DataFrame(rows).sort_values("p_value")
                results["mutation_expr"] = mut_df

                n_sig = (mut_df["p_value"] < 0.05).sum()
                print(f"  ✅  Mutation-Expression: {n_sig} genes with "
                      f"p < 0.05 (Mann-Whitney).")

    # ── miRNA → Target mRNA ──────────────────────────────────────────────
    if mirna_expr is not None:
        common_samples = sorted(set(rna_expr.columns) & set(mirna_expr.columns))
        if len(common_samples) > 10:
            # Top variable miRNAs vs top variable mRNAs
            mirna_top = mirna_expr.loc[
                mirna_expr.var(axis=1).nlargest(100).index,
                common_samples,
            ]
            rna_top = rna_expr.loc[
                rna_expr.var(axis=1).nlargest(500).index,
                common_samples,
            ]

            corr_mirna = pd.DataFrame(
                np.corrcoef(mirna_top, rna_top)[:100, 100:],
                index=mirna_top.index,
                columns=rna_top.index,
            )

            results["mirna_mrna_corr"] = corr_mirna

            # Known GBM miRNA targets
            gbm_mirnas = [m for m in mirna_top.index
                          if any(k in m.upper() for k in
                                 ["MIR21", "MIR10B", "MIR128", "MIR7",
                                  "MIR221", "MIR222", "MIR181", "MIR34A"])]
            if gbm_mirnas:
                print(f"  📊  GBM miRNA targets: {len(gbm_mirnas)} known miRNAs "
                      f"analyzed.")
                results["gbm_mirna_targets"] = corr_mirna.loc[gbm_mirnas] if gbm_mirnas else pd.DataFrame()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in results.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                df.to_csv(output_dir / f"cross_omics_{name}.tsv", sep="\t")

    return results


def run_mofa(
    views: dict[str, pd.DataFrame],
    n_factors: int = 10,
    n_iter: int = 1000,
    convergence_mode: str = "fast",
    output_dir: Path | None = None,
) -> dict:
    """
    Multi-Omics Factor Analysis (MOFA2).

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        {omics_name: features × samples DataFrame}
        e.g. {"rna": rna_df, "methylation": meth_df, "cnv": cnv_df}
    n_factors : int
        Number of latent factors.
    n_iter : int
        Maximum iterations.
    convergence_mode : str
        "fast", "medium", or "slow".
    output_dir : Path, optional

    Returns
    -------
    dict
        {factors, weights, variance_explained, metadata}
    """
    try:
        import mofapy2
        from mofapy2.run.entry_point import entry_point

        print(f"  🧬  MOFA: {len(views)} omics views, K={n_factors} factors.")

        # Prepare data for MOFA
        mofa_data = entry_point()
        mofa_data.set_data_options(
            scale_groups=False,
            scale_views=True,
        )
        mofa_data.set_model_options(
            factors=n_factors,
            spikeslab_weights=True,
            ard_factors=True,
        )
        mofa_data.set_train_options(
            iter=n_iter,
            convergence_mode=convergence_mode,
            drop_factor_threshold=0.02,
            verbose=False,
        )

        # Add views
        for name, df in views.items():
            # MOFA expects samples × features
            mofa_data.set_data_matrix(
                view_name=name,
                matrix=df.T.values,
                features_names=df.index.tolist(),
                samples_names=df.columns.tolist(),
            )

        # Build and train
        mofa_data.build()
        mofa_data.run()

        # Extract results
        factors = mofa_data.model.getFactors()
        weights = mofa_data.model.getWeights()
        var_explained = mofa_data.model.calculate_variance_explained()

        print(f"  ✅  MOFA complete. Factors: {factors.shape[1]}.")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(factors).to_csv(
                output_dir / "mofa_factors.tsv", sep="\t",
            )

        return {
            "factors": factors,
            "weights": weights,
            "variance_explained": var_explained,
            "model": mofa_data,
        }

    except ImportError:
        print("  ⚠   mofapy2 not available. Install with: pip install mofapy2")
        return {}


def snf_cluster(
    views: dict[str, pd.DataFrame],
    n_neighbors: int = 20,
    n_clusters_range: tuple[int, ...] = (2, 3, 4, 5),
    output_dir: Path | None = None,
) -> dict:
    """
    Similarity Network Fusion for multi-omics clustering.

    Parameters
    ----------
    views : dict[str, pd.DataFrame]
        {omics_name: features × samples}.
    n_neighbors : int
        Number of neighbors for SNF graph.
    n_clusters_range : tuple
        Cluster numbers to evaluate.
    output_dir : Path, optional

    Returns
    -------
    dict
        {clusters, affinity_matrix, silhouette_scores}
    """
    # SNF requires SNFtool or snfpy package
    try:
        from snf import snf
        from sklearn.cluster import spectral_clustering
        from sklearn.metrics import silhouette_score

        print(f"  🧬  SNF clustering: {len(views)} omics, "
              f"K ∈ {n_clusters_range}.")

        # Build affinity matrices per view
        affinities = {}
        for name, df in views.items():
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaled = StandardScaler().fit_transform(df.T)
            # Compute Euclidean distance → affinity
            from sklearn.metrics import pairwise_distances
            dist = pairwise_distances(scaled, metric="euclidean")
            # Convert to affinity using scaled exponential similarity kernel
            sigma = np.median(dist[dist > 0])
            aff = np.exp(-dist ** 2 / (2 * sigma ** 2))
            affinities[name] = pd.DataFrame(aff, index=df.columns, columns=df.columns)

        # Fuse networks
        fused = snf(list(affinities.values()), K=n_neighbors)

        # Find optimal number of clusters
        best_k = n_clusters_range[0]
        best_score = -1
        scores = {}

        for k in n_clusters_range:
            labels = spectral_clustering(fused, n_clusters=k, random_state=42)
            if len(set(labels)) > 1:
                score = silhouette_score(fused, labels, metric="precomputed")
                scores[k] = score
                if score > best_score:
                    best_score = score
                    best_k = k

        # Final clustering
        final_labels = spectral_clustering(fused, n_clusters=best_k, random_state=42)
        sample_labels = {s: f"SNF_{l + 1}" for s, l in zip(views[list(views.keys())[0]].columns, final_labels)}

        clusters = pd.DataFrame({
            "sample": list(sample_labels.keys()),
            "snf_cluster": list(sample_labels.values()),
        })

        print(f"  ✅  SNF: optimal K={best_k} "
              f"(silhouette={best_score:.3f}).")
        for k, v in clusters["snf_cluster"].value_counts().items():
            print(f"       {k}: {v} samples.")

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            clusters.to_csv(output_dir / "snf_clusters.tsv", sep="\t", index=False)

        return {
            "clusters": clusters,
            "affinity_matrix": fused,
            "silhouette_scores": scores,
            "optimal_k": best_k,
        }

    except ImportError:
        print("  ⚠   snfpy not available. Install with: pip install snfpy")
        return {}
