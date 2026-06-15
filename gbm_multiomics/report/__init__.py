"""
report — Automated thesis report generation.

Uses Quarto to generate a publication-ready PDF report
with all analysis results, figures, tables, and citations.
"""

from gbm_multiomics.report.quarto_generator import generate_report

__all__ = ["generate_report"]
