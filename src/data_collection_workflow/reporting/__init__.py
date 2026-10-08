"""Deterministic reporting utilities for completed Data Collection Workflow sessions."""

from .report_facts import build_report_facts
from .run_settings import build_run_settings
from .source_catalog import build_source_catalog
from .unified_report import write_unified_report


def write_final_reports(session_dir):
    """Compatibility entry point for the single English session report."""
    paths = write_unified_report(session_dir)
    return {**paths, "english_report": paths["final_report_english"],
            "facts_json": paths["report_snapshot_json"]}


__all__ = ["build_report_facts", "build_run_settings", "build_source_catalog",
           "write_unified_report", "write_final_reports"]
