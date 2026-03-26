# gbm-multiomics

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

A command-line toolkit to download and analyse **TCGA-GBM** (Glioblastoma Multiforme) multiomics data directly from the [NCI GDC portal](https://portal.gdc.cancer.gov) REST API, and prepare it for downstream analysis including differential expression, pathway enrichment, survival analysis, and molecular subtype classification.

Built as an analysis-focused companion to [tcga-gdc-downloader](https://github.com/onedimkurt/tcga-gdc-downloader), specialised entirely for GBM biology and the WHO 2021 CNS tumour classification framework.

---

## Features

- Downloads five data types for TCGA-GBM directly via the GDC REST API — no `gdc-client` required
- GBM-specific postprocessing built into each downloader (IDH status, MGMT methylation, Chr7+/Chr10− flags, driver gene matrix, GBM miRNA signatures)
- Checkpoint system: safely resumes interrupted downloads
- Four downstream analysis modules: differential expression, pathway enrichment, survival, molecular subtype classification
- R script generation for DESeq2 (when Python pydeseq2 is not available)
- Built-in GBM pathway gene sets (RTK/PI3K, p53/RB, IDH biology, invasion, angiogenesis, stem cells) — offline ORA without internet access to MSigDB

---

## Data Types

| Flag | Data type | GDC assay | Key output |
|------|-----------|-----------|------------|
| `rna-seq` | STAR-Counts gene expression | RNA-Seq | genes × samples raw count matrix |
| `methylation` | Beta values | Methylation Array (450k/EPIC) | probes × samples + MGMT promoter summary |
| `mutations` | Masked somatic mutations | WXS | PASS-filtered MAF + IDH1/IDH2 status per sample |
| `cnv` | Copy number segments | Genotyping Array | DNAcopy segments + Chr7 gain / Chr10 loss flags |
| `mirna` | miRNA expression | miRNA-Seq | miRNA × samples RPM matrix + GBM miRNA summary |

---

## Installation

Requires Python 3.10 or newer.

```bash
git clone https://github.com/femrebora/gbm-multiomics.git
cd gbm-multiomics
pip install .
```

For downstream analysis (differential expression, pathway enrichment, survival, subtype):

```bash
pip install ".[analysis]"
```

This installs: `pydeseq2`, `lifelines`, `gseapy`, `scikit-learn`, `scipy`, `matplotlib`, `seaborn`, `statsmodels`.

---

## Quick Start

### 1. Download data

```bash
# Single data type
gbm-download --data-type rna-seq

# Multiple data types in one run
gbm-download --data-type rna-seq mutations methylation

# All five data types
gbm-download --data-type all

# Discover files without downloading (recommended first step)
gbm-download --data-type rna-seq --dry-run

# Custom output directory
gbm-download --data-type rna-seq --output ~/my_gbm_data

# Resume an interrupted download (checkpoint is automatic)
gbm-download --data-type rna-seq

# Start completely fresh
gbm-download --data-type rna-seq --fresh
```

### 2. Run downstream analysis

```bash
# Differential expression: tumor vs normal
gbm-analyse --analysis de --condition is_tumor --reference False

# Survival analysis stratified by IDH status
gbm-analyse --analysis survival --endpoint OS --group IDH_status

# GBM molecular subtype classification (Verhaak 2010)
gbm-analyse --analysis subtype

# Pathway enrichment on DE results
gbm-analyse --analysis pathway

# All analyses in one command
gbm-analyse --analysis all --data-dir ~/gbm_multiomics_data
```

---

## Output Structure

After a full `--data-type all` download:

```
~/gbm_multiomics_data/
├── rna_seq/
│   ├── raw_counts/                     # extracted STAR-Counts TSV files
│   ├── rna_seq_counts.tsv              # genes × samples raw count matrix
│   ├── rna_seq_metadata.tsv            # per-sample metadata (tumor/normal, barcodes)
│   └── metadata_with_cdr.tsv          # enriched with PanCanAtlas CDR survival + subtypes
├── methylation/
│   ├── raw/                            # extracted beta-value TXT files
│   ├── methylation_beta.tsv            # CpG probes × samples (float32 beta values)
│   ├── methylation_metadata.tsv
│   └── mgmt_probe_summary.tsv         # MGMT promoter mean beta + methylated flag
├── mutations/
│   ├── raw/                            # extracted MAF files
│   ├── mutations_all.tsv              # all PASS-filtered variants (long format)
│   ├── mutations_drivers.tsv          # driver gene × sample presence/absence matrix
│   ├── idh_status.tsv                 # IDH1/IDH2 mutation status per sample
│   └── mutations_metadata.tsv
├── cnv/
│   ├── raw/                            # extracted segment TXT files
│   ├── cnv_segments.tsv               # all segments (long format)
│   ├── cnv_chr7_chr10.tsv             # Chr7 gain / Chr10 loss flags per sample
│   └── cnv_metadata.tsv
├── mirna/
│   ├── raw/                            # extracted miRNA TXT files
│   ├── mirna_rpm.tsv                  # miRNA × samples RPM matrix
│   ├── gbm_mirna_summary.tsv          # mean/std RPM for GBM-relevant miRNAs
│   └── mirna_metadata.tsv
└── gbm_pipeline_checkpoint.json       # checkpoint — delete to restart from scratch
```

After `gbm-analyse --analysis all`:

```
~/gbm_multiomics_data/analysis/
├── differential_expression/
│   ├── de_results_is_tumor.tsv        # log2FC, pvalue, padj per gene
│   ├── deseq2_run.R                   # ready-to-run R script (DESeq2 + EnhancedVolcano)
│   ├── counts_for_deseq2.tsv
│   └── coldata_for_deseq2.tsv
├── pathway_enrichment/
│   ├── up_custom/ora_results.tsv      # GBM gene set ORA (upregulated)
│   ├── down_custom/ora_results.tsv    # GBM gene set ORA (downregulated)
│   └── up_hallmarks/ora_results.tsv  # MSigDB Hallmarks ORA (requires internet)
├── survival/
│   ├── km_IDH_status.pdf              # Kaplan-Meier curve
│   ├── logrank_IDH_status.tsv         # log-rank test results
│   └── cox_univariate.tsv             # hazard ratios + 95% CI
└── subtype/
    ├── gbm_subtypes_centroid.tsv      # Verhaak subtype + per-subtype correlations
    └── who_2021_classification.tsv    # provisional WHO 2021 classification
```

---

## GBM-Specific Biology

### Why these data types?

| Data type | GBM relevance |
|-----------|--------------|
| RNA-seq | Defines Classical / Mesenchymal / Proneural / Neural subtypes (Verhaak 2010); RTK amplification signatures |
| Methylation | **MGMT promoter methylation** — predictive of temozolomide response (Hegi 2005 NEJM); **G-CIMP** — IDH-mutant hypermethylation phenotype |
| Mutations | **IDH1 R132H** — defines WHO 2021 grade; **TERT promoter** — poor prognosis; EGFR/PTEN/NF1/TP53 driver landscape |
| CNV | **Chr7 gain + Chr10 loss** — near-universal in IDH-wildtype GBM (~85%); EGFR amplification; CDKN2A/B deletion |
| miRNA | miR-21 (oncomiR); miR-10b (invasion); miR-128/miR-7 (tumour suppressors) |

### WHO 2021 CNS Classification (simplified)

```
IDH-wildtype  → Glioblastoma, IDH-wildtype, Grade 4
IDH-mutant    + CDKN2A/B homozygous deletion → Astrocytoma, IDH-mutant, Grade 4
IDH-mutant    + 1p/19q codeletion            → Oligodendroglioma, IDH-mutant
IDH-mutant    (no above)                     → Astrocytoma, IDH-mutant, Grade 2/3
```

This tool outputs IDH status from MAF files and provisional WHO classification. Full classification requires integrating CNV (CDKN2A/B) and 1p/19q data manually.

---

## Analysis Modules

### Differential Expression

Supports two backends:
- **pydeseq2** (Python) — runs directly if installed
- **R/DESeq2** — always writes a ready-to-run `deseq2_run.R` script

```python
from gbm_multiomics.analysis.differential_expression import run_deseq2_py, filter_significant

results = run_deseq2_py(
    counts        = count_matrix,     # genes × samples int DataFrame
    metadata      = sample_meta,      # must have 'condition' column
    condition_col = "IDH_status",
    reference     = "IDH_wildtype",
    output_dir    = Path("results/de"),
)
sig = filter_significant(results, padj_threshold=0.05, lfc_threshold=1.0)
```

### Pathway Enrichment

```python
from gbm_multiomics.analysis.pathway_enrichment import (
    run_ora, run_gsea, run_gbm_custom_ora, GBM_GENE_SETS
)

# Offline ORA using built-in GBM gene sets
run_gbm_custom_ora(gene_list=sig_genes, output_dir=Path("results/pathway"))

# ORA with MSigDB Hallmarks (requires internet + gseapy)
run_ora(gene_list=sig_genes, gene_sets="MSigDB_Hallmark_2020")

# Pre-ranked GSEA
run_gsea(ranked_genes=de_results["stat"], gene_sets="KEGG_2021_Human")
```

Built-in `GBM_GENE_SETS`:
- `RTK_PI3K_AKT_pathway`
- `p53_RB_cell_cycle`
- `IDH_and_epigenetic`
- `GBM_invasion`
- `GBM_angiogenesis`
- `GBM_stem_cell_markers`
- `NF1_RAS_MAPK`

### Survival Analysis

```python
from gbm_multiomics.analysis.survival import (
    kaplan_meier, cox_univariate, expression_survival_split
)

# Kaplan-Meier with log-rank test
km = kaplan_meier(df, "cdr_OS.time", "cdr_OS", "IDH_status", output_dir=Path("results"))

# Gene expression survival split (EGFR high vs low)
expression_survival_split(
    df, gene_expr=log2_tpm.loc["EGFR"], "cdr_OS.time", "cdr_OS",
    split="median", gene_name="EGFR",
)
```

Survival endpoints from PanCanAtlas CDR (downloaded automatically with RNA-seq):
- **OS** — Overall Survival
- **PFI** — Progression-Free Interval
- **DSS** — Disease-Specific Survival

### Subtype Classification

```python
from gbm_multiomics.analysis.subtype import classify_centroids, cluster_nmf, who_2021_classify

# Verhaak 2010 centroid-based classification
subtypes = classify_centroids(log2_cpm_matrix, output_dir=Path("results/subtype"))

# Unsupervised NMF (k=4)
nmf_labels = cluster_nmf(log2_cpm_matrix, n_components=4)

# WHO 2021 provisional classification from IDH status
who = who_2021_classify(idh_status_df)
```

---

## Python API

```python
from pathlib import Path
from gbm_multiomics import GBMClient
from gbm_multiomics.downloaders import rna_seq, mutations, methylation

client = GBMClient()   # or GBMClient(token="...") for controlled-access

# Discover files (no download)
records = rna_seq.discover(client)
print(f"Found {len(records)} RNA-seq files")

# Full download + parse pipeline
result = rna_seq.run(client, output_dir=Path("~/gbm_data"))
counts   = result["counts"]    # pd.DataFrame: genes × samples
metadata = result["metadata"]  # pd.DataFrame: per-sample metadata

# IDH status from mutations
mut_result = mutations.run(client, output_dir=Path("~/gbm_data"))
idh_status = mut_result["idh_status"]
```

---

## Requirements

**Core** (download only):
```
python >= 3.10
requests >= 2.31
pandas >= 2.0
numpy >= 1.26
tqdm >= 4.66
openpyxl >= 3.1
```

**Analysis** (`pip install ".[analysis]"`):
```
pydeseq2 >= 0.4
lifelines >= 0.27
gseapy >= 1.1
scikit-learn >= 1.3
scipy >= 1.11
statsmodels >= 0.14
matplotlib >= 3.7
seaborn >= 0.13
```

---

## GDC Data Access

All data types downloaded by this tool use **open-access** GDC files — no authentication token is required for standard TCGA-GBM data.

A GDC token is only needed for controlled-access data (e.g. raw sequencing reads, germline variants). Download a token at [portal.gdc.cancer.gov](https://portal.gdc.cancer.gov) (Login → click username → Download Token) and pass it with `--token ~/gdc-user-token.txt`.

---

## Citation

If you use this tool in your research, please cite the NCI GDC:

> Grossman RL, Heath AP, Ferretti V, et al. Toward a Shared Vision for Cancer Genomic Data. *N Engl J Med.* 2016;375(12):1109-1112.

And the PanCanAtlas Clinical Data Resource (downloaded automatically for survival annotations):

> Liu J, Lichtenberg T, Hoadley KA, et al. An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. *Cell.* 2018;173(2):400-416.e11.

---

## License

MIT — see [LICENSE](LICENSE) for details.
