"""Markdown renderer for deterministic final workflow reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from data_collection_workflow.text_encoding import repair_mojibake_text

from .report_facts import (
    FINAL_REPORT_DIAGNOSTICS_JSON,
    FINAL_REPORT_FACTS_JSON,
    NOT_AVAILABLE,
    build_report_facts,
)


ENGLISH_FINAL_REPORT = "final_report.md"


def _preview(value: Any, limit: int = 180) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value if value is not None else "")
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _cell(value: Any) -> str:
    text = _preview(value).replace("|", "/")
    return text if text else NOT_AVAILABLE


def _table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(value) for value in row) + " |")
    return lines


def _kv_table(mapping: dict) -> list[str]:
    return _table(["metric", "value"], [[key, value] for key, value in mapping.items()])


def _records_table(rows: list[dict], headers: list[str]) -> list[str]:
    if not rows:
        return ["No rows available."]
    return _table(headers, [[row.get(header, NOT_AVAILABLE) for header in headers] for row in rows])


def _artifact_table(rows: list[dict]) -> list[str]:
    return _table(
        ["artifact", "available", "path"],
        [[row.get("artifact"), row.get("available"), row.get("path")] for row in rows],
    )


def _quality_reason_rows(reason_counts: dict) -> list[list[Any]]:
    if not reason_counts:
        return [["none", 0]]
    return [[reason, count] for reason, count in reason_counts.items()]


def _single_source_validation_note(source_corroboration_status: dict) -> list[str]:
    """Render the source validation note."""
    official_single_source = int(
        source_corroboration_status.get("official_single_source_not_cross_validated", 0)
        or 0
    )
    non_official_single_source = int(
        source_corroboration_status.get("non_official_single_source_unverified", 0) or 0
    )
    multi_source = int(
        source_corroboration_status.get("multi_source_corroborated", 0) or 0
    )
    if official_single_source <= 0 or multi_source > 0:
        return []
    return [
        "**Cross-source validation note:** Source-backed primary records exist, but cross-source validation is incomplete.",
        f"official_single_source_not_cross_validated=`{official_single_source}`; non_official_single_source_unverified=`{non_official_single_source}`.",
        "",
    ]


def _readable_workflow_sections(facts: dict) -> list[str]:
    outputs = facts.get("readable_outputs") or {}
    reviewable_rows = outputs.get("reviewable_evidence_matrix") or []
    reviewable_total = int(
        outputs.get("reviewable_evidence_total_count")
        or facts["executive_summary"].get("reviewable_records_count")
        or 0
    )
    reviewable_shown = len(reviewable_rows)
    run_summary = {
        "run_status": facts["executive_summary"].get("run_status"),
        "real_collection_status": facts["executive_summary"].get(
            "real_collection_status"
        ),
        "accepted_records": facts["executive_summary"].get("accepted_records_count"),
        "reviewable_evidence": reviewable_total,
        "quarantined_records": facts["executive_summary"].get(
            "quarantined_records_count"
        ),
        "sources": facts["data_source_summary"].get("total_source_candidates"),
        "fetches": facts["data_source_summary"].get("fetched_source_count"),
        "root_failure_class": facts["executive_summary"].get(
            "one_sentence_main_limitation"
        ),
    }
    note = "reviewable evidence is useful output, not failure: it preserves source-backed evidence that is not yet safe for the final primary dataset."
    return [
        "## Run Summary",
        "",
        *_kv_table(run_summary),
        "",
        note,
        "",
        "## Reviewable Evidence Matrix",
        "",
        f"showing {reviewable_shown} of {reviewable_total}",
        "",
        *_records_table(
            reviewable_rows,
            [
                "evidence_id",
                "record_id",
                "source_id",
                "source_product_type",
                "evidence_role",
                "candidate_type",
                "location",
                "case_count",
                "death_count",
                "quality_status",
                "review_reason",
                "source_url",
                "evidence_quote",
            ],
        ),
        "",
        "## Case Evidence Bundles",
        "",
        *_records_table(
            outputs.get("case_evidence_bundles") or [],
            [
                "case_evidence_bundle_id",
                "bundle_type",
                "supporting_source_count",
                "verified_authority_source_count",
                "source_product_types",
                "main_blocking_reason",
                "recommended_review_action",
                "best_evidence_quote",
                "source_urls",
            ],
        ),
        "",
        "## Source Registry Profile",
        "",
        *_records_table(
            outputs.get("source_registry_profile") or [],
            [
                "source_id",
                "source_name",
                "source_product_type",
                "source_type",
                "authority_score",
                "task_specificity",
                "time_window_fit",
                "machine_readability",
                "screening_decision",
            ],
        ),
        "",
        "## Fetch Manifest",
        "",
        *_records_table(
            outputs.get("fetch_manifest") or [],
            [
                "source_id",
                "fetch_selected",
                "data_product_type",
                "task_specificity",
                "skip_reason",
                "source_url",
            ],
        ),
        "",
        "## Record Provenance",
        "",
        *_records_table(
            outputs.get("record_provenance") or [],
            [
                "record_or_evidence_id",
                "source_id",
                "source_url",
                "source_product_type",
                "quality_status",
                "review_reason",
                "evidence_quote",
            ],
        ),
        "",
        "## Source-to-Evidence Funnel",
        "",
        *_kv_table(outputs.get("source_to_evidence_funnel") or {}),
        "",
        "## High-Confidence Exact-Page Recall",
        "",
        *_kv_table(outputs.get("high_confidence_exact_page_recall") or {}),
        "",
        "## High-Confidence Sources With No Extracted Record",
        "",
        *_records_table(
            outputs.get("high_confidence_sources_no_record") or [],
            [
                "source_id",
                "source_name",
                "source_type",
                "source_product_type",
                "source_to_evidence_status",
                "target_chunk_count",
                "target_record_count",
                "source_only_reason",
                "extraction_failure_substage",
                "source_url",
            ],
        ),
        "",
        "## Disease-Local Rejection Counts",
        "",
        *_kv_table(outputs.get("disease_local_rejection_counts") or {}),
        "",
        "## Case-Field Completeness Summary",
        "",
        *_kv_table(outputs.get("case_field_completeness_summary") or {}),
        "",
        "## LLM Empty-Output Recovery",
        "",
        *_kv_table(outputs.get("llm_empty_output_recovery") or {}),
        "",
        "## Case-Span Extraction Coverage",
        "",
        *_kv_table(outputs.get("case_span_extraction_coverage") or {}),
        "",
        "## Failure Funnel",
        "",
        *_kv_table(outputs.get("failure_funnel") or {}),
        "",
    ]


def render_final_report_english(facts: dict) -> str:
    executive = facts["executive_summary"]
    task_scope = facts["task_scope"]
    source_summary = facts["data_source_summary"]
    final_stats = facts["final_statistics"]
    quality = facts["quality_and_trustworthiness"]
    validation = facts["validation_readiness"]
    readiness = facts.get("collection_readiness") or {}
    diagnostics = facts["diagnostics"]

    lines: list[str] = [
        "# Final Public Health Data Collection Report",
        "",
        "This report is generated from deterministic workflow artifacts only. It does not use an LLM to create counts, sources, records, or validation conclusions.",
        "",
        *_readable_workflow_sections(facts),
        "## 1. Executive Summary",
        "",
        *_kv_table(executive),
        "",
    ]
    if executive["suitable_as_final_epidemiological_dataset"] == "no":
        lines.extend(
            [
                "**Final epidemiological use warning:** suitable_as_final_epidemiological_dataset is `no`; accepted records, if any, should be treated as an auditable evidence product rather than a final epidemiological dataset.",
                "",
            ]
        )
    lines.extend(
        [
            "## 2. Task Scope",
            "",
            *_kv_table(task_scope),
            "",
            "## 3. Collection Funnel",
            "",
            *_kv_table(facts["collection_funnel"]),
            "",
            "## 4. Case Study Readiness",
            "",
            *_kv_table(readiness or {"collection_readiness_status": NOT_AVAILABLE}),
            "",
        ]
    )
    if diagnostics["missing_report_metrics"]:
        lines.extend(
            [
                "Missing report metrics were written to diagnostics as `missing_report_metric` entries.",
                "",
            ]
        )
    lines.extend(
        [
            "## 5. Data Source Summary",
            "",
            *_kv_table(source_summary),
            "",
        ]
    )
    if source_summary.get("warnings"):
        lines.extend(
            [
                "**Source warning:** "
                + ", ".join(str(item) for item in source_summary["warnings"]),
                "",
            ]
        )
    lines.extend(
        [
            "## 6. Final Statistical Results",
            "",
            "Primary statistics below use accepted records only. Pending review and quarantined records are excluded.",
            "",
            "### Accepted Primary Dataset Table",
            "",
            *_records_table(
                final_stats["accepted_primary_dataset"],
                [
                    "record_id",
                    "date_or_reporting_period",
                    "location",
                    "cases",
                    "deaths",
                    "statistical_count_type",
                    "source_name",
                    "source_url",
                    "evidence_quote",
                    "quality_status",
                ],
            ),
            "",
            "### Aggregation",
            "",
            *_kv_table(final_stats["aggregation"]),
            "",
        ]
    )
    if final_stats["provenance_warnings"]:
        lines.extend(
            [
                "### Provenance Warnings",
                "",
                *_table(
                    ["record_id", "warnings"],
                    [
                        [row.get("record_id"), ", ".join(row.get("warnings") or [])]
                        for row in final_stats["provenance_warnings"]
                    ],
                ),
                "",
            ]
        )
    lines.extend(
        [
            "## 7. Quality and Trustworthiness",
            "",
            *_kv_table(
                {
                    "accepted_count": quality["accepted_count"],
                    "pending_review_count": quality["pending_review_count"],
                    "quarantined_count": quality["quarantined_count"],
                    "source_corroboration_status": quality[
                        "source_corroboration_status"
                    ],
                    "provenance_completeness": quality["provenance_completeness"],
                }
            ),
            "",
            *_single_source_validation_note(
                quality["source_corroboration_status"]
            ),
            "### main_quarantine_reasons",
            "",
            *_table(["reason", "count"], _quality_reason_rows(quality["main_quarantine_reasons"])),
            "",
            "### main_pending_review_reasons",
            "",
            *_table(["reason", "count"], _quality_reason_rows(quality["main_pending_review_reasons"])),
            "",
            "## 8. Validation / Comparison Readiness",
            "",
            *_kv_table(validation),
            "",
            "If `evaluation_rows_count` is `not_available` or `0`, this report does not claim benchmark validation.",
            "",
            "## 9. Human Review Tasks",
            "",
            *_records_table(
                facts["human_review_tasks"],
                [
                    "priority",
                    "review_task_type",
                    "record_id_source_id_packet_id",
                    "reason",
                    "expected_human_action",
                ],
            ),
            "",
            "## 10. Exported Artifacts",
            "",
            *_artifact_table(facts["exported_artifacts"]),
            "",
            "## Diagnostics",
            "",
            *_kv_table(
                {
                    "missing_report_metric": diagnostics["missing_report_metrics"],
                    "reporting_warnings": diagnostics["reporting_warnings"],
                    "legacy_reports_treated_as_diagnostics": diagnostics[
                        "legacy_reports_treated_as_diagnostics"
                    ],
                    "generated_from_artifacts_only": facts["report_metadata"][
                        "generated_from_artifacts_only"
                    ],
                    "llm_called_for_report": facts["report_metadata"][
                        "llm_called_for_report"
                    ],
                }
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def _copy_legacy_reports_to_diagnostics(session_dir: Path) -> list[str]:
    diagnostics_dir = session_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in (
        "workflow_run_report.md",
        "workflow_interpretive_report.md",
        "workflow_interpretive_report_summary.json",
    ):
        source = session_dir / name
        if not source.exists():
            continue
        target = diagnostics_dir / name
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
        copied.append(str(target))
    return copied


def write_final_reports(session_dir: Path | str) -> dict:
    """Write final_report.md and fact diagnostics."""

    session = Path(session_dir)
    session.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = session / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    facts = build_report_facts(session)
    legacy_diagnostic_paths = _copy_legacy_reports_to_diagnostics(session)
    if legacy_diagnostic_paths:
        facts["diagnostics"]["legacy_report_diagnostic_paths"] = legacy_diagnostic_paths

    english_path = session / ENGLISH_FINAL_REPORT
    facts_path = session / FINAL_REPORT_FACTS_JSON
    diagnostics_path = diagnostics_dir / FINAL_REPORT_DIAGNOSTICS_JSON

    english_path.write_text(
        repair_mojibake_text(render_final_report_english(facts)),
        encoding="utf-8",
    )
    facts_path.write_text(
        json.dumps(facts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    diagnostics_payload = {
        "missing_report_metric": facts["diagnostics"]["missing_report_metrics"],
        "reporting_warnings": facts["diagnostics"]["reporting_warnings"],
        "legacy_reports_treated_as_diagnostics": facts["diagnostics"][
            "legacy_reports_treated_as_diagnostics"
        ],
        "legacy_report_diagnostic_paths": facts["diagnostics"].get(
            "legacy_report_diagnostic_paths", []
        ),
        "generated_from_artifacts_only": True,
        "llm_called_for_report": False,
        "search_called_for_report": False,
        "fetch_called_for_report": False,
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "english_report": str(english_path),
        "facts_json": str(facts_path),
        "diagnostics_json": str(diagnostics_path),
    }
