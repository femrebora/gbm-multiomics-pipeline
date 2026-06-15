# GBM Multi-Omics Prognostic Biomarker Pipeline
# Docker image with Python 3.11 + R 4.3 for full analysis reproducibility
#
# Build:  docker build -t gbm-pipeline -f docker/Dockerfile .
# Run:    docker run --rm -v $(pwd)/data:/data -v $(pwd)/results:/results gbm-pipeline
#
# References:
#   Love et al. (2014) Genome Biology 15:550 — DESeq2
#   Tibshirani (1997) Stat Med 16:385-395 — Lasso-Cox
#   Argelaguet et al. (2018) Mol Syst Biol 14:e8124 — MOFA

FROM python:3.11-slim

LABEL maintainer="GBM Multi-Omics Pipeline"
LABEL description="TCGA-GBM prognostic biomarker discovery pipeline"
LABEL version="0.2.0"

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # R installation
    r-base=4.3.* \
    r-base-dev \
    libcurl4-openssl-dev \
    libssl-dev \
    libxml2-dev \
    libfontconfig1-dev \
    libharfbuzz-dev \
    libfribidi-dev \
    libfreetype6-dev \
    libpng-dev \
    libtiff5-dev \
    libjpeg-dev \
    libgsl-dev \
    libopenblas-dev \
    # Build tools
    build-essential \
    git \
    curl \
    wget \
    # For Quarto report generation (optional)
    pandoc \
    # Clean up
    && rm -rf /var/lib/apt/lists/*

# ── Install R packages ──────────────────────────────────────────────────────
RUN R -e 'install.packages(c("BiocManager"), repos="https://cloud.r-project.org")' \
    && R -e 'BiocManager::install(c( \
        "DESeq2", \
        "EnhancedVolcano", \
        "ComplexHeatmap", \
        "survminer", \
        "glmnet", \
        "survival", \
        "timeROC", \
        "rms", \
        "impute", \
        "preprocessCore", \
        "sva", \
        "limma" \
    ), ask=FALSE, update=FALSE)'

# ── Python dependencies ─────────────────────────────────────────────────────
WORKDIR /app

COPY pyproject.toml .
COPY gbm_multiomics/__init__.py gbm_multiomics/

RUN pip install --no-cache-dir -e ".[analysis]" \
    && pip install --no-cache-dir \
        pyyaml>=6.0 \
        scikit-survival>=0.22 \
        plotly>=5.18 \
        adjustText>=1.1 \
        networkx>=3.2 \
        rpy2>=3.5 \
        jupyter>=1.0 \
        notebook>=7.0 \
        quarto-cli>=1.4

# ── Install MOFA2 separately (has complex deps) ─────────────────────────────
RUN pip install --no-cache-dir mofapy2>=0.7 || echo "MOFA2 will be installed via R/Bioc if needed"

# ── Copy application code ───────────────────────────────────────────────────
COPY . .

RUN pip install --no-cache-dir -e ".[analysis,all]"

# ── Runtime configuration ───────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV MPLBACKEND=Agg

# Data and results volumes
VOLUME ["/data", "/results"]

# Default entry point
ENTRYPOINT ["python", "-m", "gbm_multiomics.cli"]
CMD ["--help"]
