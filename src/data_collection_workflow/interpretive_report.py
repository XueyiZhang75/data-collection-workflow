"""Deterministic interpretive report builder for completed workflow sessions.

This module reads existing session artifacts only. It does not call an LLM,
search provider, fetcher, graph node, or human-review mutator.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from data_collection_workflow.text_encoding import repair_mojibake, repair_mojibake_text


ENGLISH_REPORT = "workflow_interpretive_report.md"
SUMMARY_JSON = "workflow_interpretive_report_summary.json"

_COUNT_FIELDS = {
    "final_case_dataset_count": ("final_case_dataset",),
    "global_outbreak_event_dataset_count": ("global_outbreak_event_dataset",),
    "regional_surveillance_dataset_count": ("regional_surveillance_dataset",),
    "country_year_aggregate_dataset_count": ("country_year_aggregate_dataset",),
    "official_alert_dataset_count": ("official_alert_dataset",),
    "final_dataset_count": ("final_dataset",),
    "final_dataset_pre_quality_gate_count": ("final_dataset_pre_quality_gate",),
    "final_dataset_post_review_count": ("final_dataset_post_review",),
    "zero_case_statement_count": ("zero_case_statements",),
    "exposure_monitoring_record_count": ("exposure_monitoring_records",),
    "surveillance_summary_record_count": ("surveillance_summary_records",),
    "outbreak_summary_record_count": ("outbreak_summary_records",),
    "context_record_count": ("context_records",),
    "unclassified_observation_count": ("unclassified_observation_records",),
    "non_primary_observation_count": ("non_primary_observations",),
    "quarantined_record_count": ("quarantined_records",),
    "pending_review_record_count": ("pending_review_records",),
    "claim_count": ("claims",),
    "corroborated_event_count": ("corroborated_events",),
}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _read_collection_rows(session_dir: Path, stem: str, warnings: list[str]) -> list[dict]:
    collection_dir = session_dir / "collection"
    json_value = _read_json(collection_dir / f"{stem}.json")
    if isinstance(json_value, list):
        return [row for row in json_value if isinstance(row, dict)]
    if json_value is not None:
        warnings.append(f"collection/{stem}.json exists but is not a list")
    csv_rows = _read_csv(collection_dir / f"{stem}.csv")
    if csv_rows:
        return csv_rows
    return []


def _read_diagnostic_dict(session_dir: Path, stem: str, warnings: list[str]) -> dict:
    value = _read_json(session_dir / "diagnostics" / f"{stem}.json")
    if isinstance(value, dict):
        return value
    if value is not None:
        warnings.append(f"diagnostics/{stem}.json exists but is not an object")
    return {}


def _as_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value) -> list:
    return value if isinstance(value, list) else []


def _first_present(*values, default=None):
    for value in values:
        if value not in (None, "", []):
            return value
    return default


def _count(
    name: str,
    *,
    run_summary: dict,
    run_quality: dict,
    final_quality: dict,
    collections: dict[str, list[dict]],
) -> int:
    value = _first_present(
        run_summary.get(name),
        run_quality.get(name),
        final_quality.get(name),
    )
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    stems = _COUNT_FIELDS.get(name) or ()
    for stem in stems:
        if stem in collections:
            return len(collections[stem])
    return 0


def _source_text(record: dict) -> str:
    return _first_present(
        record.get("source_title"),
        record.get("title"),
        record.get("source_id"),
        default="unknown source",
    )


def _publisher_text(record: dict) -> str:
    return _first_present(
        record.get("actual_publisher"),
        record.get("publisher"),
        record.get("source_publisher"),
        default="unknown publisher",
    )


def _case_counts(record: dict) -> str:
    fields = [
        "cases_confirmed",
        "cases_probable",
        "cases_suspected",
        "cases_unspecified",
        "deaths",
        "hospitalizations",
    ]
    parts = [
        f"{field}={record.get(field)}"
        for field in fields
        if record.get(field) not in (None, "", [])
    ]
    return ", ".join(parts) if parts else "count fields unavailable"


def _evidence(record: dict, limit: int = 220) -> str:
    text = " ".join(str(record.get("evidence_quote") or "").split())
    if not text:
        text = " ".join(str(record.get("evidence") or "").split())
    if not text:
        return "evidence quote unavailable"
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _sample_records(records: list[dict], limit: int = 5) -> list[dict]:
    return list(records[:limit])


def _format_markdown_table(rows: Iterable[Iterable[Any]]) -> list[str]:
    rows = [[str(cell if cell is not None else "") for cell in row] for row in rows]
    if not rows:
        return []
    header = rows[0]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in rows[1:]:
        lines.append("| " + " | ".join(cell.replace("|", "/") for cell in row) + " |")
    return lines


def _validation_limited(status: str | None, run_quality: dict, summary: dict) -> bool:
    if run_quality.get("validation_limited") is True:
        return True
    if summary.get("validation_limited") is True:
        return True
    return str(status or "") in {
        "no_task_compatible_validation_source",
        "incompatible_validation_source_disabled",
        "validation_limited_no_compatible_source",
    }


def _llm_stages_enabled(run_summary: dict) -> bool:
    if "all_three_llm_stages_enabled" in run_summary:
        return bool(run_summary.get("all_three_llm_stages_enabled"))
    stages = _as_dict(run_summary.get("llm_stage_summary"))
    return any(bool(_as_dict(value).get("enabled")) for value in stages.values())


def _task_fields(run_summary: dict, run_quality: dict, session_dir: Path) -> dict:
    return {
        "session_id": run_summary.get("session_id") or session_dir.name,
        "disease": _first_present(
            run_summary.get("task_disease"),
            run_quality.get("task_disease"),
            default="unknown",
        ),
        "location": _first_present(
            run_summary.get("task_location"),
            run_quality.get("task_location"),
            default="unknown",
        ),
        "start_date": _first_present(
            run_summary.get("task_start_date"),
            run_quality.get("task_start_date"),
            default="unknown",
        ),
        "end_date": _first_present(
            run_summary.get("task_end_date"),
            run_quality.get("task_end_date"),
            default="unknown",
        ),
        "user_request": run_summary.get("user_request") or "",
        "collection_mode": run_quality.get("final_dataset_mode") or "unknown",
    }


def load_interpretive_report_artifacts(session_dir: Path | str) -> dict:
    """Load existing artifacts from a completed session directory."""

    session_dir = Path(session_dir)
    warnings: list[str] = []
    run_summary = _as_dict(_read_json(session_dir / "workflow_run_summary.json"))
    if not run_summary:
        warnings.append("workflow_run_summary.json missing or unreadable")
    package = _as_dict(_read_json(session_dir / "collection" / "final_package.json"))
    if not package:
        warnings.append("collection/final_package.json missing or unreadable")

    collection_names = [
        "final_case_dataset",
        "global_outbreak_event_dataset",
        "regional_surveillance_dataset",
        "country_year_aggregate_dataset",
        "official_alert_dataset",
        "final_dataset",
        "final_dataset_pre_quality_gate",
        "final_dataset_post_review",
        "zero_case_statements",
        "exposure_monitoring_records",
        "surveillance_summary_records",
        "outbreak_summary_records",
        "context_records",
        "unclassified_observation_records",
        "non_primary_observations",
        "quarantined_records",
        "pending_review_records",
        "claims",
        "claim_comparisons",
        "corroborated_events",
        "source_identity_assessments",
        "record_inclusion_decisions",
    ]
    collections: dict[str, list[dict]] = {}
    for name in collection_names:
        rows = _read_collection_rows(session_dir, name, warnings)
        if not rows and isinstance(package.get(name), list):
            rows = [row for row in package.get(name) if isinstance(row, dict)]
        collections[name] = rows

    diagnostics = {
        name: _read_diagnostic_dict(session_dir, name, warnings)
        for name in (
            "run_quality_summary",
            "final_dataset_quality_summary",
            "observation_type_dataset_summary",
            "corroboration_summary",
            "source_identity_summary",
            "validation_source_compatibility_summary",
            "anomaly_summary",
            "human_review_application_summary",
            "workflow_summaries",
            "source_coverage_audit",
            "structured_extraction_summary",
        )
    }
    for name in (
        "run_quality_summary",
        "final_dataset_quality_summary",
        "observation_type_dataset_summary",
        "corroboration_summary",
        "source_identity_summary",
        "validation_source_compatibility_summary",
        "anomaly_summary",
        "human_review_application_summary",
        "source_coverage_audit",
        "structured_extraction_summary",
    ):
        if not diagnostics[name] and isinstance(package.get(name), dict):
            diagnostics[name] = package.get(name) or {}

    return {
        "session_dir": session_dir,
        "run_summary": run_summary,
        "package": package,
        "collections": collections,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def build_interpretive_report_summary(session_dir: Path | str) -> dict:
    artifacts = load_interpretive_report_artifacts(session_dir)
    run_summary = artifacts["run_summary"]
    diagnostics = artifacts["diagnostics"]
    collections = artifacts["collections"]
    run_quality = diagnostics["run_quality_summary"]
    final_quality = diagnostics["final_dataset_quality_summary"]
    corroboration = diagnostics["corroboration_summary"]
    source_identity = diagnostics["source_identity_summary"]
    validation = diagnostics["validation_source_compatibility_summary"]
    anomaly = diagnostics["anomaly_summary"]
    human_review = diagnostics["human_review_application_summary"]
    task = _task_fields(run_summary, run_quality, artifacts["session_dir"])

    final_case_count = _count(
        "final_case_dataset_count",
        run_summary=run_summary,
        run_quality=run_quality,
        final_quality=final_quality,
        collections=collections,
    )
    global_outbreak_count = _count(
        "global_outbreak_event_dataset_count",
        run_summary=run_summary,
        run_quality=run_quality,
        final_quality=final_quality,
        collections=collections,
    )
    regional_surveillance_count = _count(
        "regional_surveillance_dataset_count",
        run_summary=run_summary,
        run_quality=run_quality,
        final_quality=final_quality,
        collections=collections,
    )
    country_year_aggregate_count = _count(
        "country_year_aggregate_dataset_count",
        run_summary=run_summary,
        run_quality=run_quality,
        final_quality=final_quality,
        collections=collections,
    )
    official_alert_count = _count(
        "official_alert_dataset_count",
        run_summary=run_summary,
        run_quality=run_quality,
        final_quality=final_quality,
        collections=collections,
    )
    validation_status = _first_present(
        run_summary.get("validation_source_compatibility_status"),
        validation.get("compatibility_status"),
        default="unknown",
    )
    validation_mode = _first_present(
        run_summary.get("validation_mode"),
        validation.get("validation_mode"),
        default="held_out_file",
    )
    validation_is_limited = _validation_limited(
        validation_status,
        run_quality,
        run_summary,
    )
    corroborated_primary = int(
        _first_present(
            run_summary.get("corroborated_primary_case_event_count"),
            corroboration.get("corroborated_primary_case_event_count"),
            run_quality.get("corroborated_primary_case_event_count"),
            default=0,
        )
        or 0
    )
    task_location = str(task.get("location") or "").lower()
    global_task = task_location in {"global", "worldwide", "world", "all"}
    task_aware_product_count = (
        final_case_count
        + global_outbreak_count
        + regional_surveillance_count
        + country_year_aggregate_count
        + official_alert_count
    )
    suitable = (
        task_aware_product_count > 0
        and (corroborated_primary > 0 or global_task)
        and not validation_is_limited
        and str(run_quality.get("run_quality_status") or run_summary.get("run_quality_status"))
        not in {"failed_quality_gate", "no_records_extracted"}
    )
    if final_case_count == 0 and global_task and task_aware_product_count:
        conclusion_en = (
            "This global workflow run did not produce a local line-list style primary case dataset, "
            f"but it produced task-aware outputs: {global_outbreak_count} global outbreak event records, "
            f"{regional_surveillance_count} regional surveillance records, and {official_alert_count} official alert records."
        )
    elif final_case_count == 0:
        conclusion_en = (
            "The workflow technically completed search, fetch, extraction, filtering, and export steps, "
            "but no accepted primary case dataset records were found; therefore the output is not suitable as a final epidemiological dataset for the target location."
        )
    else:
        plural = "" if final_case_count == 1 else "s"
        conclusion_en = (
            f"The workflow produced {final_case_count} accepted primary case dataset record{plural}; "
            "these records still require expert review of source identity, corroboration status, and human review outcomes before final use."
        )

    key_artifacts = {
        ENGLISH_REPORT: ENGLISH_REPORT,
        "workflow_run_report.md": "workflow_run_report.md",
        SUMMARY_JSON: SUMMARY_JSON,
        "workflow_run_summary.json": "workflow_run_summary.json",
        "collection/final_case_dataset.csv": "collection/final_case_dataset.csv",
        "collection/final_case_dataset.json": "collection/final_case_dataset.json",
        "collection/global_outbreak_event_dataset.csv": "collection/global_outbreak_event_dataset.csv",
        "collection/regional_surveillance_dataset.csv": "collection/regional_surveillance_dataset.csv",
        "collection/country_year_aggregate_dataset.csv": "collection/country_year_aggregate_dataset.csv",
        "collection/official_alert_dataset.csv": "collection/official_alert_dataset.csv",
        "collection/final_dataset.csv": "collection/final_dataset.csv",
        "collection/final_dataset_pre_quality_gate.csv": "collection/final_dataset_pre_quality_gate.csv",
        "collection/zero_case_statements.csv": "collection/zero_case_statements.csv",
        "collection/exposure_monitoring_records.csv": "collection/exposure_monitoring_records.csv",
        "collection/context_records.csv": "collection/context_records.csv",
        "collection/quarantined_records.csv": "collection/quarantined_records.csv",
        "diagnostics/run_quality_summary.json": "diagnostics/run_quality_summary.json",
        "diagnostics/corroboration_summary.json": "diagnostics/corroboration_summary.json",
        "diagnostics/source_identity_summary.json": "diagnostics/source_identity_summary.json",
        "diagnostics/validation_source_compatibility_summary.json": "diagnostics/validation_source_compatibility_summary.json",
    }
    recommended_steps = [
        "Inspect final_case_dataset before using any case-count output.",
        "Review quarantined_records and record_inclusion_decisions for excluded evidence.",
        "Review source_identity_summary and source_identity_assessments for publisher uncertainty.",
        "Review corroboration_summary before treating any claim as cross-source supported.",
        "Apply human review decisions in a separate review pass if needed.",
    ]
    if final_case_count == 0:
        recommended_steps.insert(0, "Do not treat this run as a final primary case dataset.")
    if validation_is_limited:
        if validation_mode == "live_cross_source":
            recommended_steps.append(
                "Add or discover task-compatible live validation sources for cross-source validation."
            )
        else:
            recommended_steps.append("Add or identify a task-compatible held-out validation source.")

    summary = {
        "session_id": task["session_id"],
        "task_disease": task["disease"],
        "task_location": task["location"],
        "task_start_date": task["start_date"],
        "task_end_date": task["end_date"],
        "collection_mode": task["collection_mode"],
        "live_search_enabled": bool(run_summary.get("live_search_enabled")),
        "live_fetch_enabled": bool(run_summary.get("live_fetch_enabled")),
        "llm_stages_enabled": _llm_stages_enabled(run_summary),
        "one_sentence_conclusion_en": conclusion_en,
        "final_case_dataset_count": final_case_count,
        "global_outbreak_event_dataset_count": global_outbreak_count,
        "regional_surveillance_dataset_count": regional_surveillance_count,
        "country_year_aggregate_dataset_count": country_year_aggregate_count,
        "official_alert_dataset_count": official_alert_count,
        "task_aware_data_product_count": task_aware_product_count,
        "final_dataset_count": _count("final_dataset_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "final_dataset_pre_quality_gate_count": _count("final_dataset_pre_quality_gate_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "final_dataset_post_review_count": _count("final_dataset_post_review_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "zero_case_statement_count": _count("zero_case_statement_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "exposure_monitoring_record_count": _count("exposure_monitoring_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "surveillance_summary_record_count": _count("surveillance_summary_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "outbreak_summary_record_count": _count("outbreak_summary_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "context_record_count": _count("context_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "unclassified_observation_count": _count("unclassified_observation_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "non_primary_observation_count": _count("non_primary_observation_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "quarantined_record_count": _count("quarantined_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "pending_review_record_count": _count("pending_review_record_count", run_summary=run_summary, run_quality=run_quality, final_quality=final_quality, collections=collections),
        "claim_count": int(_first_present(run_summary.get("claim_count"), corroboration.get("claim_count"), default=len(collections["claims"])) or 0),
        "claim_comparison_count": int(_first_present(run_summary.get("claim_comparison_count"), corroboration.get("claim_comparison_count"), default=len(collections["claim_comparisons"])) or 0),
        "corroborated_event_count": int(_first_present(run_summary.get("corroborated_event_count"), corroboration.get("corroborated_event_count"), default=len(collections["corroborated_events"])) or 0),
        "corroborated_primary_case_event_count": corroborated_primary,
        "validation_source_compatibility_status": validation_status,
        "validation_mode": validation_mode,
        "validation_limited": validation_is_limited,
        "source_identity_assessed_count": int(_first_present(run_summary.get("source_identity_assessed_count"), source_identity.get("identity_assessed_count"), default=len(collections["source_identity_assessments"])) or 0),
        "actual_publisher_unknown_count": int(_first_present(source_identity.get("unknown_publisher_count"), run_summary.get("actual_publisher_unknown_count"), default=0) or 0),
        "human_review_item_count": int(_first_present(run_summary.get("human_review_item_count"), human_review.get("human_review_item_count"), human_review.get("review_item_count"), default=0) or 0),
        "anomaly_count": int(_first_present(run_summary.get("anomaly_result_count"), anomaly.get("anomaly_count"), default=0) or 0),
        "run_quality_status": _first_present(run_summary.get("run_quality_status"), run_quality.get("run_quality_status"), default="unknown"),
        "primary_case_dataset_status": _first_present(run_summary.get("primary_case_dataset_status"), run_quality.get("primary_case_dataset_status"), default="unknown"),
        "suitable_as_final_epidemiological_dataset": bool(suitable),
        "recommended_next_steps": recommended_steps,
        "key_artifacts": key_artifacts,
        "warnings": artifacts["warnings"],
        "generated_from_artifacts_only": True,
        "llm_called_for_report": False,
        "search_called_for_report": False,
        "fetch_called_for_report": False,
    }
    return summary


def _status_rows(summary: dict) -> list[list[Any]]:
    fields = [
        "final_case_dataset_count",
        "global_outbreak_event_dataset_count",
        "regional_surveillance_dataset_count",
        "country_year_aggregate_dataset_count",
        "official_alert_dataset_count",
        "task_aware_data_product_count",
        "final_dataset_count",
        "final_dataset_pre_quality_gate_count",
        "zero_case_statement_count",
        "exposure_monitoring_record_count",
        "surveillance_summary_record_count",
        "outbreak_summary_record_count",
        "context_record_count",
        "unclassified_observation_count",
        "non_primary_observation_count",
        "quarantined_record_count",
        "pending_review_record_count",
        "final_dataset_post_review_count",
        "run_quality_status",
        "primary_case_dataset_status",
        "suitable_as_final_epidemiological_dataset",
    ]
    return [["Field", "Value"], *[[field, summary.get(field)] for field in fields]]


def _primary_case_lines(records: list[dict]) -> list[str]:
    lines: list[str] = []
    if not records:
        return [
            "No accepted primary case dataset records were produced.",
            "Evidence found in zero-case, exposure monitoring, context, quarantined, or other views may be useful, but it does not directly answer the primary case-data question.",
        ]
    for record in _sample_records(records):
        lines.append(
            "- "
            + "; ".join(
                [
                    f"disease={record.get('disease') or 'unknown'}",
                    f"location={record.get('location') or record.get('country') or 'unknown'}",
                    f"date={record.get('date_reported') or record.get('period') or 'unknown'}",
                    _case_counts(record),
                    f"source={_source_text(record)}",
                    f"publisher={_publisher_text(record)}",
                    f"url={record.get('source_url') or 'unavailable'}",
                    f"corroboration_status={record.get('corroboration_status') or 'unknown'}",
                    f"independent_source_count={record.get('independent_source_count') or 0}",
                    f"evidence={_evidence(record)}",
                ]
            )
        )
    return lines


def _task_aware_dataset_lines(collections: dict[str, list[dict]]) -> list[str]:
    labels = [
        ("global_outbreak_event_dataset", "global outbreak event records"),
        ("regional_surveillance_dataset", "regional surveillance records"),
        ("country_year_aggregate_dataset", "country-year aggregate records"),
        ("official_alert_dataset", "official alert records"),
    ]
    lines: list[str] = []
    for key, label in labels:
        rows = collections.get(key) or []
        if not rows:
            continue
        lines.append(f"- {key}: {len(rows)} " + "record(s).")
        for row in _sample_records(rows, limit=3):
            lines.append(
                "  - "
                + "; ".join(
                    [
                        f"source={_source_text(row)}",
                        f"publisher={_publisher_text(row)}",
                        _case_counts(row),
                        f"url={row.get('source_url') or 'unavailable'}",
                    ]
                )
            )
    if not lines:
        return ["No global/task-aware dataset view contains records."]
    return lines


def _non_case_lines(collections: dict[str, list[dict]]) -> list[str]:
    labels = [
        ("zero_case_statements", "zero-case statements"),
        ("exposure_monitoring_records", "exposure monitoring records"),
        ("surveillance_summary_records", "surveillance summaries"),
        ("outbreak_summary_records", "outbreak summaries"),
        ("context_records", "context/background records"),
        ("unclassified_observation_records", "unclassified observations"),
        ("non_primary_observations", "non-primary observations"),
    ]
    lines = []
    for key, en_label in labels:
        count = len(collections.get(key) or [])
        if count <= 0:
            continue
        lines.append(
            f"- {en_label}: {count}. These are useful public-health observations but not confirmed cases unless separately accepted as primary case records."
        )
    if not lines:
        lines.append("No non-case observation view contains records.")
    lines.append(
        "Zero-case and exposure-monitoring observations are not confirmed cases; context/background evidence is useful but not a case count."
    )
    return lines


def _coverage_extraction_lines(diagnostics: dict) -> list[str]:
    coverage = _as_dict(diagnostics.get("source_coverage_audit"))
    content_fetch = _as_dict(diagnostics.get("content_fetch_summary"))
    document_quality = _as_dict(diagnostics.get("document_quality_summary"))
    evidence_chunking = _as_dict(diagnostics.get("evidence_chunking_summary"))
    extraction = _as_dict(diagnostics.get("structured_extraction_summary"))
    if not coverage and (not extraction) and (not content_fetch):
        return ["No source coverage audit artifact is available."]
    requirements = _as_list(coverage.get("requirements"))
    fetch_failed_count = int(
        coverage.get("fetch_failed_requirement_count")
        or content_fetch.get("fetch_failures_blocking_count")
        or 0
    )
    unusable_count = int(
        coverage.get("unusable_requirement_count")
        or document_quality.get("unusable_count")
        or 0
    )
    total_chunks = int(
        evidence_chunking.get("total_chunk_count")
        or extraction.get("input_chunk_count")
        or 0
    )
    raw_records = int(extraction.get("raw_record_count") or 0)
    accepted_records = int(coverage.get("accepted_record_count") or 0)
    parsed_not_extracted = [
        req
        for req in requirements
        if req.get("parsed") is True
        and (not req.get("extracted"))
        and (int(req.get("extracted_record_count") or 0) == 0)
    ]
    failures = _as_list(extraction.get("official_extraction_failures"))
    lines = [
        f"- coverage_status: `{coverage.get('coverage_status') or 'unknown'}`",
        f"- source verification chain: predicted={len(requirements)}, discovered={coverage.get('discovered_requirement_count', 0)}, fetched={coverage.get('fetched_requirement_count', 0)}, fetch_failed={fetch_failed_count}, parsed={coverage.get('parsed_requirement_count', 0)}, unusable={unusable_count}, chunks={total_chunks}, records={raw_records}, extracted={coverage.get('extracted_requirement_count', 0)}, accepted={coverage.get('accepted_requirement_count', 0)}",
    ]
    if fetch_failed_count and (not int(coverage.get("fetched_requirement_count") or 0)):
        lines.append(
            "- failure_stage: `all_target_fetch_failed` before evidence chunking/extraction."
        )
    elif unusable_count and total_chunks <= 0:
        lines.append("- failure_stage: `parsed_unusable_or_no_chunks`.")
    elif total_chunks <= 0 and raw_records <= 0:
        lines.append("- failure_stage: `no_chunks`.")
    elif raw_records <= 0:
        lines.append("- failure_stage: `no_records_extracted`.")
    elif accepted_records <= 0:
        lines.append("- failure_stage: `records_quarantined`.")
    if parsed_not_extracted:
        lines.append(
            f"- {len(parsed_not_extracted)} official sources were discovered/fetched/parsed but produced no extracted records."
        )
    if failures:
        reason_counts: dict[str, int] = {}
        for failure in failures:
            reason = str(failure.get("reason") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        lines.append(f"- official_extraction_failure_reasons: `{reason_counts}`")
    return lines


def build_interpretive_report_english(session_dir: Path | str) -> str:
    artifacts = load_interpretive_report_artifacts(session_dir)
    summary = build_interpretive_report_summary(session_dir)
    collections = artifacts["collections"]
    diagnostics = artifacts["diagnostics"]
    run_summary = artifacts["run_summary"]
    validation = diagnostics["validation_source_compatibility_summary"]
    source_identity = diagnostics["source_identity_summary"]
    source_search = _as_dict(run_summary.get("source_search_execution_summary"))
    lines: list[str] = [
        "# Data Collection Result Interpretation Report",
        "",
        "## 1. Task",
        "",
        f"- disease: `{summary['task_disease']}`",
        f"- location: `{summary['task_location']}`",
        f"- date range: `{summary['task_start_date']} to {summary['task_end_date']}`",
        "- target fields: cases, deaths, dates, locations, source URLs, source types, evidence quotes",
        f"- collection mode: `{summary['collection_mode']}`",
        f"- session id: `{summary['session_id']}`",
        f"- live search: `{summary['live_search_enabled']}`",
        f"- live fetch: `{summary['live_fetch_enabled']}`",
        f"- LLM stages: `{summary['llm_stages_enabled']}`",
        f"- search provider: `{run_summary.get('source_search_provider') or source_search.get('search_provider') or 'unknown'}`",
        "",
        "## 2. One-sentence conclusion",
        "",
        summary["one_sentence_conclusion_en"],
        "",
        "## 3. Final data status",
        "",
        *_format_markdown_table(_status_rows(summary)),
        "",
        "## 3.1 Coverage and extraction status",
        "",
        *_coverage_extraction_lines(diagnostics),
        "",
        "## 4. Primary case dataset findings",
        "",
        *_primary_case_lines(collections["final_case_dataset"]),
        "",
        "## 5. Global/task-aware dataset views",
        "",
        *_task_aware_dataset_lines(collections),
        "",
        "## 5. Useful non-case public-health observations",
        "",
        *_non_case_lines(collections),
        "",
        "## 6. Cross-source corroboration",
        "",
        f"- claim_count: `{summary['claim_count']}`",
        f"- claim_comparison_count: `{summary['claim_comparison_count']}`",
        f"- corroborated_event_count: `{summary['corroborated_event_count']}`",
        f"- corroborated_primary_case_event_count: `{summary['corroborated_primary_case_event_count']}`",
        f"- conflicting_claim_count: `{diagnostics['corroboration_summary'].get('conflicting_claim_count', 0)}`",
        f"- single_source_unverified_count: `{diagnostics['corroboration_summary'].get('single_source_unverified_count', 0)}`",
        "These fields describe cross-source support, single-source unverified evidence, or conflicts. They do not establish automatic truth determination.",
        "",
        "## 7. Source quality and credibility",
        "",
        f"- source_candidate_count: `{run_summary.get('source_registry_count', 0)}`",
        f"- fetched_document_count: `{run_summary.get('document_count', 0)}`",
        f"- source_identity_assessed_count: `{summary['source_identity_assessed_count']}`",
        f"- actual_publisher_unknown_count: `{summary['actual_publisher_unknown_count']}`",
        f"- source_type_counts: `{source_identity.get('source_type_counts') or {}}`",
        f"- source_critic_assessed_count: `{(_as_dict(run_summary.get('llm_stage_summary')).get('source_critic') or {}).get('assessed_source_count', 0)}`",
        "Sources are described as official, news, context-only, or unknown only when the source identity artifacts support that label.",
        "",
        "## 8. Validation status",
        "",
        f"- validation_source_compatibility_status: `{summary['validation_source_compatibility_status']}`",
        f"- validation_mode: `{summary.get('validation_mode')}`",
        f"- active_validation_record_count: `{validation.get('active_validation_record_count', run_summary.get('active_validation_record_count', 0))}`",
        f"- inactive_validation_record_count: `{validation.get('inactive_validation_record_count', run_summary.get('inactive_validation_record_count', 0))}`",
        f"- validation_limited: `{summary['validation_limited']}`",
    ]
    if summary["validation_limited"]:
        lines.append(
            "validation is limited because no task-compatible held-out validation source was available; this does not prove that no case occurred."
        )
    if summary["validation_limited"] and summary.get("validation_mode") == "live_cross_source":
        lines = [
            line for line in lines if "held-out validation source" not in line
        ]
        lines.append(
            "Live cross-source validation was limited because this run did not find a task-compatible validation source. This does not prove absence of cases; it means the live search/fetch set did not include enough independent validation evidence."
        )
    lines.extend(
        [
            "",
            "## 9. Excluded / quarantined evidence",
            "",
            f"- quarantined_record_count: `{summary['quarantined_record_count']}`",
            f"- pending_review_record_count: `{summary['pending_review_record_count']}`",
            "These records did not enter the primary case dataset. Inspect quarantined_records and record_inclusion_decisions for exclusion reasons.",
            "",
            "## 10. Human review priorities",
            "",
            f"- human_review_item_count: `{summary['human_review_item_count']}`",
            "- Review source scope mismatch, validation limitation, publisher uncertainty, and single-source unverified claims first.",
        ]
    )
    for item in _sample_records(collections.get("human_review_items") or _as_list(artifacts["package"].get("human_review_items")), 5):
        lines.append(f"- {item.get('review_id') or 'review'}: {item.get('reason') or item.get('item_type') or 'needs review'}")
    lines.extend(
        [
            "",
            "## 11. Can this be used as a final epidemiological dataset?",
            "",
            f"- suitable_as_final_epidemiological_dataset: `{summary['suitable_as_final_epidemiological_dataset']}`",
            "If false, the output should not be used directly as a final case dataset. It can still be used for evidence audit and expert review.",
            "",
            "## 12. Recommended next steps",
            "",
            *[f"- {step}" for step in summary["recommended_next_steps"]],
            "",
            "## 13. Key artifact index",
            "",
            *[f"- `{value}`" for value in summary["key_artifacts"].values()],
            "",
            "## 14. Important disclaimer",
            "",
            "Note: This report interprets evidence collected by the workflow. It is not an official surveillance conclusion, medical advice, or automatic truth determination. Expert review is still required.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_interpretive_reports(session_dir: Path | str) -> dict:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    english_path = session_dir / ENGLISH_REPORT
    summary_path = session_dir / SUMMARY_JSON
    english_path.write_text(
        build_interpretive_report_english(session_dir),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(
            repair_mojibake(build_interpretive_report_summary(session_dir)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "english_report": str(english_path),
        "summary_json": str(summary_path),
    }
