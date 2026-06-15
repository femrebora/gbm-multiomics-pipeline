# gbm-multiomics

![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Tests](https://img.shields.io/badge/tests-50%20passed-green)

**Prognostic biomarker discovery pipeline for Glioblastoma (GBM) using TCGA multi-omics data.** Downloads RNA-seq, methylation, mutations, CNV, and miRNA data from the NCI GDC, then identifies genes essential for GBM survival with publication-ready figures.

---

## Quick Start

```bash
# 1. Install
git clone https://github.com/femrebora/gbm-multiomics-pipeline.git
cd gbm-multiomics-pipeline
pip install -e ".[analysis]"

# 2. Download TCGA-GBM data
gbm-pipeline download --data-type rna-seq

# 3. Preprocess & QC
gbm-pipeline preprocess

# 4. Find prognostic genes
gbm-pipeline analyze --module prognostic

# 5. Look up a gene of interest
gbm-pipeline gene-lookup --gene PTGS1

# 6. Generate figures
gbm-pipeline gene-spotlight --genes EGFR,PTEN,TP53,IDH1

# 7. Full pipeline (download → figures → report)
gbm-pipeline run
```

---

## Installation

Requires **Python 3.10+**.

```bash
git clone https://github.com/femrebora/gbm-multiomics-pipeline.git
cd gbm-multiomics-pipeline

# Core (download only)
pip install -e .

# Full analysis (DE, survival, prognostic, multi-omics, figures)
pip install -e ".[analysis]"
```

### Data Available 

| Tool | Purpose |
|------|---------|
| [tcga-gdc-downloader](https://github.com/onedimkurt/tcga-gdc-downloader) | Download TCGA RNA-seq data for any of 33 cancer types with CDR survival annotations, used as the foundation for this pipeline |

Install separately if you need pan-cancer TCGA downloads outside of GBM:

```bash
pip install tcga-gdc-downloader
```

### Docker (optional)

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml run pipeline run
```

---

## Commands

All commands are available under `gbm-pipeline`:

### Download

```bash
gbm-pipeline download --data-type rna-seq           # single data type
gbm-pipeline download --data-type all               # all 5 omics types
gbm-pipeline download --data-type rna-seq --dry-run # check first, don't download
gbm-pipeline download --data-type rna-seq --fresh   # restart from scratch
```

**Data types:** `rna-seq` `methylation` `mutations` `cnv` `mirna`

### Preprocess

```bash
gbm-pipeline preprocess                  # VST normalize + QC
gbm-pipeline preprocess --method tpm     # TPM normalization instead
```

### Analyze

```bash
gbm-pipeline analyze --module prognostic   # genome-wide Cox → Lasso → risk score
gbm-pipeline analyze --module de           # differential expression
gbm-pipeline analyze --module multiomics   # cross-omics correlation
gbm-pipeline analyze --module immune       # ESTIMATE immune scores
gbm-pipeline analyze --module network      # PPI networks, hub genes
gbm-pipeline analyze --module all          # everything
```

### Gene Focus (thesis spotlight)

```bash
gbm-pipeline gene-lookup --gene EGFR                          # single gene across all omics
gbm-pipeline gene-rank --genes EGFR,PTEN,TP53                 # rank vs genome-wide results
gbm-pipeline gene-spotlight --genes EGFR,PTEN,TP53,IDH1,NF1   # per-gene publication figures
gbm-pipeline gene-focus --genes EGFR,PTEN,TP53,IDH1,NF1       # full report + ranking + figures
```

### Report

```bash
gbm-pipeline report   # generate thesis PDF via Quarto
```

---

## What the pipeline does

```
TCGA-GBM GDC API
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  DOWNLOAD                                               │
│  RNA-seq · Methylation · Mutations · CNV · miRNA       │
│  + PanCanAtlas CDR survival annotations                 │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  PREPROCESS                                             │
│  Gene annotation · VST/TPM normalization · PCA QC      │
│  Outlier detection · Clinical data integration          │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  ANALYSIS                                               │
│  Differential expression · Pathway enrichment           │
│  Genome-wide Cox regression · Lasso-Cox selection       │
│  Multivariate modeling · Risk score · Time-ROC          │
│  Multi-omics integration (MOFA, SNF)                    │
│  Immune infiltration (ESTIMATE) · PPI networks          │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  GENE FOCUS                                             │
│  Single-gene lookup · Multi-gene report                 │
│  Genome-wide ranking · DepMap validation                │
│  Thesis spotlight figures                               │
└─────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│  OUTPUT                                                 │
│  Publication-ready figures (PDF/PNG, 300 DPI)           │
│  Tables · Heatmaps · Volcano · KM curves · Forest plots │
│  Quarto thesis report                                   │
└─────────────────────────────────────────────────────────┘
```

---

## Output Structure

```
~/gbm_multiomics_data/
├── rna_seq/           # count matrix + metadata + CDR annotations
├── methylation/       # beta values + MGMT promoter summary
├── mutations/         # MAF + driver matrix + IDH status
├── cnv/               # segments + Chr7/Chr10 flags
├── mirna/             # RPM matrix + GBM miRNA summary
├── preprocessed/      # normalized expression + QC reports
└── analysis/
    ├── differential_expression/
    ├── prognostic/         # Cox results, risk scores, DepMap
    ├── pathway_enrichment/
    ├── survival/
    ├── multiomics/
    ├── immune/
    ├── network/
    └── gene_focus/         # spotlight figures, gene reports

results/
├── figures/           # publication-ready PDFs and PNGs
├── tables/            # all result tables
└── report/            # Quarto thesis PDF
```

---

## Key References

The pipeline cites these papers (see `references.bib`):

| Paper | Topic |
|-------|-------|
| Verhaak et al. (2010) *Cancer Cell* | GBM molecular subtypes |
| Brennan et al. (2013) *Cell* | TCGA GBM genomic landscape |
| Hegi et al. (2005) *NEJM* | MGMT methylation & temozolomide |
| Louis et al. (2021) *Neuro-Oncology* | WHO 2021 CNS classification |
| Liu et al. (2018) *Cell* | PanCanAtlas CDR survival data |
| Love et al. (2014) *Genome Biology* | DESeq2 |
| Tibshirani (1997) *Stat Med* | Lasso-Cox |
| Argelaguet et al. (2018) *Mol Syst Biol* | MOFA |
| Meyers et al. (2017) *Nat Genet* | DepMap CRISPR screens |
| Yoshihara et al. (2013) *Nat Commun* | ESTIMATE |

---

## Citation

If you use this pipeline in your research:

> **GBM Multi-Omics Pipeline** (2025). TCGA-GBM prognostic biomarker discovery toolkit.
> https://github.com/femrebora/gbm-multiomics-pipeline

Please also cite the NCI GDC and PanCanAtlas CDR:

> Grossman RL, et al. (2016). *N Engl J Med*, 375(12):1109-1112.
>
> Liu J, et al. (2018). *Cell*, 173(2):400-416.e11.
