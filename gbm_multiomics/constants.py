"""
constants.py — GBM-specific project IDs, GDC filter specs, and field lists.

TCGA-GBM data types available on GDC (open access):
  rna-seq     : STAR-Counts TSV  (RNA-Seq, Gene Expression Quantification)
  methylation : Beta-value TXT   (Methylation Array, Illumina 450k/EPIC)
  mutations   : Masked MAF       (WXS, Masked Somatic Mutation)
  cnv         : Segment TXT      (Genotyping Array, Copy Number Segment)
  mirna       : quant TXT        (miRNA-Seq, miRNA Expression Quantification)

All downloads use:
  POST https://api.gdc.cancer.gov/data
  Body: {"ids": ["file_uuid", ...]}
"""

from __future__ import annotations

# ── Project ───────────────────────────────────────────────────────────────────
GBM_PROJECT_ID = "TCGA-GBM"

# ── GDC API endpoints ─────────────────────────────────────────────────────────
GDC_BASE               = "https://api.gdc.cancer.gov"
GDC_FILES_ENDPOINT     = f"{GDC_BASE}/files"
GDC_CASES_ENDPOINT     = f"{GDC_BASE}/cases"
GDC_DATA_ENDPOINT      = f"{GDC_BASE}/data"
GDC_PROJECTS_ENDPOINT  = f"{GDC_BASE}/projects"
GDC_STATUS_ENDPOINT    = f"{GDC_BASE}/status"

# ── Pagination / networking ───────────────────────────────────────────────────
GDC_PAGE_SIZE          = 2_000
MAX_TOTAL_FILES        = 50_000
REQUEST_TIMEOUT_SHORT  = 60
REQUEST_TIMEOUT_LONG   = 900
MAX_RETRIES            = 3
RETRY_WAIT_SECONDS     = 10
CHUNK_SIZE_BYTES       = 1024 * 1024   # 1 MB
GDC_DOWNLOAD_BATCH_SIZE = 50           # keep batches under ~100 MB GDC limit

# ── Checkpoint filename ───────────────────────────────────────────────────────
CHECKPOINT_FILE = "gbm_pipeline_checkpoint.json"

# ── Shared field list for /files queries ─────────────────────────────────────
BASE_FILE_FIELDS = (
    "file_id,file_name,file_size,"
    "cases.case_id,cases.submitter_id,"
    "cases.samples.sample_id,cases.samples.submitter_id,"
    "cases.samples.sample_type,cases.samples.tissue_type"
)

# ── Clinical fields ───────────────────────────────────────────────────────────
CLINICAL_FIELDS = ",".join([
    "case_id", "submitter_id",
    "demographic.gender", "demographic.age_at_index", "demographic.race",
    "demographic.vital_status", "demographic.days_to_death",
    "diagnoses.age_at_diagnosis", "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_death", "diagnoses.primary_diagnosis",
    "diagnoses.tumor_grade", "diagnoses.morphology",
    "diagnoses.tissue_or_organ_of_origin",
])

# ─────────────────────────────────────────────────────────────────────────────
# GDC filter specs — one list per data type.
# Each is composed with a project_id filter at query time.
# ─────────────────────────────────────────────────────────────────────────────

RNA_SEQ_FILTERS: list[dict] = [
    {"op": "=", "content": {"field": "data_type",
                             "value": "Gene Expression Quantification"}},
    {"op": "=", "content": {"field": "experimental_strategy",
                             "value": "RNA-Seq"}},
    {"op": "=", "content": {"field": "analysis.workflow_type",
                             "value": "STAR - Counts"}},
    {"op": "=", "content": {"field": "data_format",
                             "value": "TSV"}},
    {"op": "=", "content": {"field": "access",
                             "value": "open"}},
]

METHYLATION_FILTERS: list[dict] = [
    {"op": "=", "content": {"field": "data_type",
                             "value": "Methylation Beta Value"}},
    {"op": "=", "content": {"field": "experimental_strategy",
                             "value": "Methylation Array"}},
    {"op": "=", "content": {"field": "data_format",
                             "value": "TXT"}},
    {"op": "=", "content": {"field": "access",
                             "value": "open"}},
]

MUTATIONS_FILTERS: list[dict] = [
    {"op": "=", "content": {"field": "data_type",
                             "value": "Masked Somatic Mutation"}},
    {"op": "=", "content": {"field": "experimental_strategy",
                             "value": "WXS"}},
    {"op": "=", "content": {"field": "data_format",
                             "value": "MAF"}},
    {"op": "=", "content": {"field": "access",
                             "value": "open"}},
]

CNV_FILTERS: list[dict] = [
    {"op": "=", "content": {"field": "data_type",
                             "value": "Copy Number Segment"}},
    {"op": "=", "content": {"field": "experimental_strategy",
                             "value": "Genotyping Array"}},
    {"op": "=", "content": {"field": "data_format",
                             "value": "TXT"}},
    {"op": "=", "content": {"field": "access",
                             "value": "open"}},
]

MIRNA_FILTERS: list[dict] = [
    {"op": "=", "content": {"field": "data_type",
                             "value": "miRNA Expression Quantification"}},
    {"op": "=", "content": {"field": "experimental_strategy",
                             "value": "miRNA-Seq"}},
    {"op": "=", "content": {"field": "data_format",
                             "value": "TXT"}},
    {"op": "=", "content": {"field": "access",
                             "value": "open"}},
]

DATA_TYPE_FILTERS: dict[str, list[dict]] = {
    "rna-seq":      RNA_SEQ_FILTERS,
    "methylation":  METHYLATION_FILTERS,
    "mutations":    MUTATIONS_FILTERS,
    "cnv":          CNV_FILTERS,
    "mirna":        MIRNA_FILTERS,
}

ALL_DATA_TYPES: tuple[str, ...] = tuple(DATA_TYPE_FILTERS.keys())

# ── GBM molecular subtypes (Verhaak 2010 / TCGA 2013) ────────────────────────
GBM_SUBTYPES = ("Classical", "Mesenchymal", "Proneural", "Neural")

# ── Key GBM driver genes (for mutation summary) ───────────────────────────────
GBM_DRIVER_GENES = (
    "IDH1", "IDH2",        # WHO 2021: defines grade 4 GBM vs astrocytoma
    "TERT",                # TERT promoter mutation
    "EGFR",                # amplification / vIII deletion
    "PTEN",                # deletion / mutation
    "TP53",                # mutation
    "RB1",                 # deletion
    "CDKN2A",              # p16 deletion
    "NF1",                 # mutation (mesenchymal subtype)
    "PIK3CA", "PIK3R1",    # PI3K pathway
    "MGMT",                # methylation silencing (not a mut, but listed for context)
    "PDGFRA",              # amplification (proneural)
    "CDK4", "MDM2",        # co-amplification with EGFR
    "ATRX",                # loss in IDH-mutant
)

# ── PanCanAtlas CDR (survival annotations) ───────────────────────────────────
CDR_FILE_UUID     = "1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
CDR_CACHE_FILENAME = "TCGA_CDR_SupplementalTableS1.xlsx"
CDR_JOIN_KEY      = "bcr_patient_barcode"
CDR_SURVIVAL_COLS = ("OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time")
