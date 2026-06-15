"""
quarto_generator.py — Automated Quarto report generation for GBM thesis.

Generates a comprehensive .qmd report file that compiles to PDF
with all analysis results, figures, tables, and citations.

References
----------
  Quarto: https://quarto.org
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def generate_report(
    results_dir: Path,
    output_dir: Path | None = None,
    include_code: bool = False,
    citation_style: str = "cell",
    title: str = "Prognostic Biomarker Discovery in Glioblastoma",
    author: str = "",
    date: str | None = None,
) -> Path:
    """
    Generate a Quarto .qmd report file from pipeline results.

    Parameters
    ----------
    results_dir : Path
        Directory containing all pipeline output (figures/, tables/).
    output_dir : Path, optional
        Where to write the .qmd file (default: results_dir/report/).
    include_code : bool
        If True, include analysis code blocks.
    citation_style : str
        CSL style for citations ("cell", "nature", "apa", etc.).
    title : str
    author : str
    date : str, optional

    Returns
    -------
    Path
        Path to the generated .qmd file.
    """
    results_dir = Path(results_dir)
    output_dir = output_dir or (results_dir / "report")
    output_dir.mkdir(parents=True, exist_ok=True)

    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    # Discover available results
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"

    figure_files = sorted(figures_dir.glob("*.pdf")) if figures_dir.exists() else []
    table_files = sorted(tables_dir.glob("*.tsv")) if tables_dir.exists() else []

    # Build the QMD content
    qmd = _build_qmd_header(title, author, date, citation_style)
    qmd += _build_abstract()
    qmd += _build_introduction()
    qmd += _build_methods_section()
    qmd += _build_results_section(results_dir, figure_files, table_files, include_code)
    qmd += _build_discussion_section()
    qmd += _build_references_section()

    # Write file
    report_path = output_dir / "gbm_prognostic_report.qmd"
    report_path.write_text(qmd, encoding="utf-8")

    print(f"  📄  Report written to: {report_path}")
    print(f"  ℹ   To compile: cd {output_dir} && quarto render {report_path.name} --to pdf")

    return report_path


def _build_qmd_header(title: str, author: str, date: str, citation_style: str) -> str:
    return f"""---
title: "{title}"
subtitle: "TCGA-GBM Multi-Omics Integration and Essential Gene Discovery"
author: "{author}"
date: "{date}"
format:
  pdf:
    documentclass: article
    papersize: a4
    toc: true
    toc-depth: 3
    number-sections: true
    colorlinks: true
    cite-method: biblatex
    include-in-header:
      - text: \\\\usepackage{{booktabs}}
      - text: \\\\usepackage{{longtable}}
bibliography: references.bib
csl: {citation_style}.csl
execute:
  echo: false
  warning: false
---

"""


def _build_abstract() -> str:
    return r"""
# Abstract {.unnumbered}

Glioblastoma multiforme (GBM) is the most common and aggressive primary brain tumor,
with a median overall survival of approximately 15 months despite maximal therapy
[@brennan2013somatic; @verhaak2010integrated]. While large-scale genomic studies have
characterized the molecular landscape of GBM, translating these findings into clinically
actionable prognostic biomarkers remains a critical challenge.

This study presents a comprehensive multi-omics analysis of TCGA-GBM data, integrating
RNA-seq, DNA methylation, copy number variation, somatic mutations, and miRNA expression
to identify genes essential for glioblastoma progression and prognosis. Using
genome-wide Cox regression with Lasso regularization [@tibshirani1997lasso], we
identify a prognostic gene signature that predicts overall survival independent of
established clinical factors including IDH mutation status [@louis2021cns] and
MGMT promoter methylation [@hegi2005mgmt].

Multi-Omics Factor Analysis (MOFA) [@argelaguet2018mofa] and cross-omics correlation
analysis reveal genes consistently altered across multiple regulatory layers, while
external validation against DepMap CRISPR dependency screens [@meyers2017computational]
confirms the functional essentiality of top candidate genes in GBM cell lines.

Our integrated analysis identifies [NEEDS_COMPLETION] genes with strong prognostic
value and multi-omics support, representing potential therapeutic targets for
glioblastoma treatment.

**Keywords:** glioblastoma, TCGA, multi-omics, prognostic biomarker, Cox regression,
Lasso, MOFA, DepMap
\n
"""


def _build_introduction() -> str:
    return r"""
# Introduction

Glioblastoma (World Health Organization Grade 4) is the most prevalent primary
malignant brain tumor in adults, accounting for approximately 49% of all malignant
brain tumors [@brennan2013somatic]. Despite advances in surgical resection,
radiotherapy, and temozolomide chemotherapy, the median overall survival remains
dismal at approximately 15 months, with fewer than 5% of patients surviving beyond
5 years [@hegi2005mgmt].

## Molecular Classification of GBM

The 2021 WHO Classification of Central Nervous System Tumors fundamentally
reorganized glioma classification around molecular biomarkers, particularly
IDH mutation status [@louis2021cns]. IDH-wildtype glioblastoma is now defined
by the presence of TERT promoter mutation, EGFR amplification, and/or combined
gain of chromosome 7 and loss of chromosome 10.

Transcriptomic profiling has identified distinct molecular subtypes of GBM:
Classical (EGFR amplification), Mesenchymal (NF1 mutation, immune infiltration),
Proneural (IDH1 mutation, PDGFRA alteration), and Neural [@verhaak2010integrated].
These subtypes have implications for prognosis and treatment response, though
intratumoral heterogeneity can lead to subtype switching upon recurrence
[@wang2017tumor].

## Prognostic Biomarkers in GBM

The most well-established prognostic biomarker in GBM is MGMT promoter methylation,
which predicts response to temozolomide chemotherapy [@hegi2005mgmt]. IDH1/2
mutations define a distinct molecular subclass with markedly better prognosis
[@ceccarelli2016molecular]. However, beyond these canonical markers, there remains
a critical need for additional prognostic biomarkers that can refine risk
stratification and identify novel therapeutic targets.

## Study Aims

This study aims to:
1. Identify genes whose expression is significantly associated with overall
   survival in GBM through genome-wide Cox regression analysis
2. Build a multivariate prognostic model using Lasso-regularized Cox regression
3. Integrate multi-omics data (RNA-seq, methylation, CNV, mutations, miRNA) to
   identify genes consistently altered across regulatory layers
4. Validate prognostic genes against external CRISPR dependency screens (DepMap)
5. Characterize the immune microenvironment and its relationship to prognostic
   gene expression

\n
"""


def _build_methods_section() -> str:
    return r"""
# Methods

## Data Acquisition

TCGA-GBM multi-omics data were downloaded from the NCI Genomic Data Commons (GDC)
[@grossman2016toward] using the gbm-multiomics-pipeline. Five data types were
acquired:

- **RNA-seq**: STAR-Counts gene expression quantification (all open-access samples)
- **DNA Methylation**: Illumina 450k/EPIC array beta values
- **Copy Number Variation**: Genotyping array-derived copy number segments
- **Somatic Mutations**: WXS masked somatic mutation calls (MAF format)
- **miRNA Expression**: miRNA-Seq quantification

Clinical annotations, including the PanCanAtlas Clinical Data Resource (CDR)
survival endpoints [@liu2018integrated], were merged with molecular data.

## Differential Expression Analysis

Differential expression analysis was performed using DESeq2 [@love2014moderated]
with the Wald test. Multiple comparisons were evaluated: Tumor vs Normal,
IDH-mutant vs IDH-wildtype, and MGMT-methylated vs unmethylated. Genes with
adjusted p-value < 0.05 and |log2 fold change| > 1 were considered significantly
differentially expressed.

## Prognostic Biomarker Discovery

### Genome-wide Univariate Cox Regression

For each expressed gene, a univariate Cox proportional hazards model was fitted
with overall survival (OS) as the endpoint. Benjamini-Hochberg correction was
applied to control the false discovery rate at 5%.

### Lasso-Cox Feature Selection

The top 500 genes from univariate analysis were subjected to Lasso-regularized
Cox regression (α = 1.0) with 10-fold cross-validation
[@tibshirani1997lasso; @simon2011regularization]. The optimal regularization
parameter λ was selected at the minimum cross-validation error.

### Multivariate Cox Model

A multivariate Cox model was constructed incorporating the Lasso-selected gene
expression features and clinical covariates (age at diagnosis, IDH status,
MGMT methylation status). The proportional hazards assumption was tested using
Schoenfeld residuals. A prognostic risk score was calculated as the linear
predictor from the multivariate model.

### Model Evaluation

Model performance was assessed using:
- Concordance index (Harrell's C-index)
- Time-dependent ROC AUC at 1, 2, and 3 years [@heagerty2000survival]
- Kaplan-Meier analysis of high-risk vs low-risk groups (median split)

## Multi-Omics Integration

Cross-omics correlations were computed for all gene-wise omics pairs.
Multi-Omics Factor Analysis (MOFA2) [@argelaguet2018mofa] was used to identify
latent factors capturing coordinated variation across omics layers.

## External Validation

Prognostic genes were cross-referenced with the DepMap Achilles CRISPR dependency
screen (release 24Q2) [@meyers2017computational]. Gene effect scores (CERES) in
CNS/brain cancer cell lines were used to assess the functional essentiality of
prognostic candidates.

## Immune Microenvironment Analysis

ESTIMATE scores (StromalScore, ImmuneScore) were computed from RNA-seq expression
data [@yoshihara2013inferring]. Correlation between immune scores and prognostic
gene expression was assessed.

## Network Analysis

Protein-protein interaction networks were constructed using STRING v12
[@szklarczyk2023string] for the prognostic gene signature. Hub genes were
identified by degree and betweenness centrality. Community detection was
performed using the Louvain algorithm.

\n
"""


def _build_results_section(
    results_dir: Path,
    figure_files: list[Path],
    table_files: list[Path],
    include_code: bool,
) -> str:
    text = r"""
# Results

"""

    # Cohort characteristics
    text += r"""
## Cohort Characteristics

[NEEDS_COMPLETION: Summary of TCGA-GBM cohort — N samples, age distribution,
IDH status distribution, MGMT status, survival outcomes.]

"""

    # Differential Expression
    text += r"""
## Differential Expression Analysis

### Tumor vs Normal Comparison

Differential expression analysis identified [NEEDS_COMPLETION] significantly
differentially expressed genes (FDR < 0.05, |log2FC| > 1) between GBM tumor
and normal brain tissue.

"""

    volcano_files = [f for f in figure_files if "volcano" in f.stem.lower()]
    for vf in sorted(volcano_files):
        text += f"![Volcano plot]({vf.relative_to(results_dir)})\n\n"

    # Prognostic Analysis
    text += r"""
## Prognostic Biomarker Discovery

### Genome-wide Univariate Cox Regression

[NEEDS_COMPLETION: Number of prognostic genes, top hits with HR and p-values,
GBM driver gene overlap.]

### Lasso-Cox Feature Selection

[NEEDS_COMPLETION: Number of selected genes, optimal λ, cross-validation results.]

### Multivariate Prognostic Model

[NEEDS_COMPLETION: Concordance index, significant covariates, forest plot.]

### Risk Stratification

"""

    km_files = [f for f in figure_files if "km" in f.stem.lower() or "risk" in f.stem.lower()]
    for kf in sorted(km_files):
        text += f"![Kaplan-Meier]({kf.relative_to(results_dir)})\n\n"

    forest_files = [f for f in figure_files if "forest" in f.stem.lower()]
    for ff in sorted(forest_files):
        text += f"![Forest Plot]({ff.relative_to(results_dir)})\n\n"

    # Multi-Omics
    text += r"""
## Multi-Omics Integration

### Cross-Omics Correlations

[NEEDS_COMPLETION: Summary of significant cross-omics correlations.]

### MOFA Factor Analysis

[NEEDS_COMPLETION: Number of factors, variance explained per omics, top feature genes.]

"""

    heatmap_files = [f for f in figure_files if "heatmap" in f.stem.lower() or "mofa" in f.stem.lower()]
    for hf in sorted(heatmap_files):
        text += f"![Heatmap]({hf.relative_to(results_dir)})\n\n"

    # External Validation
    text += r"""
## External Validation — DepMap CRISPR Screens

[NEEDS_COMPLETION: Number of prognostic genes with strong dependency scores
in GBM cell lines.]

"""

    # Immune
    text += r"""
## Immune Microenvironment

[NEEDS_COMPLETION: ESTIMATE scores, immune-prognostic correlations.]

"""

    # Network
    text += r"""
## Protein-Protein Interaction Network

[NEEDS_COMPLETION: Network statistics, hub genes, functional modules.]

"""
    return text


def _build_discussion_section() -> str:
    return r"""
# Discussion

## Prognostic Gene Signature

[NEEDS_COMPLETION: Interpretation of prognostic genes, biological significance,
comparison with existing signatures in the literature.]

## Multi-Omics Convergence

[NEEDS_COMPLETION: Genes consistently altered across omics, regulatory mechanisms.]

## Clinical Implications

[NEEDS_COMPLETION: Potential for clinical translation, nomogram utility,
comparison with existing GBM prognostic tools.]

## Limitations

This study has several limitations:
1. TCGA-GBM represents a single cohort; external validation in independent
   datasets (e.g., CGGA, REMBRANDT) is needed
2. The retrospective nature of TCGA data limits causal inference
3. Bulk RNA-seq cannot resolve intratumoral heterogeneity
4. DepMap validation is limited to in vitro CRISPR screens

## Conclusions

[NEEDS_COMPLETION: Summary of key findings, proposed essential genes,
future directions.]

\n
"""


def _build_references_section() -> str:
    return r"""
# References {.unnumbered}

::: {#refs}
:::

\n
"""


def export_thesis_figures(
    results_dir: Path,
    output_dir: Path | None = None,
) -> list[Path]:
    """
    Copy all figures to a thesis-ready output directory with consistent naming.

    Parameters
    ----------
    results_dir : Path
    output_dir : Path, optional

    Returns
    -------
    list[Path]
        Paths to exported figure files.
    """
    results_dir = Path(results_dir)
    output_dir = output_dir or (results_dir / "figures" / "export")
    output_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = results_dir / "figures"
    if not figures_dir.exists():
        print("  ⚠   No figures directory found.")
        return []

    import shutil

    exported = []
    for fig_file in sorted(figures_dir.glob("*.pdf")):
        # Standardize naming
        new_name = fig_file.stem.replace(" ", "_").replace("(", "").replace(")", "")
        dest = output_dir / f"{new_name}.pdf"
        shutil.copy2(fig_file, dest)
        exported.append(dest)

    print(f"  📊  Exported {len(exported)} figures to {output_dir}")

    return exported
