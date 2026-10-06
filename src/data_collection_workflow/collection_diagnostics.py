"""Open-pilot diagnostics for case-study readiness.

These helpers compare open-run workflow artifacts against repo-local benchmark
metadata without injecting benchmark records into collection outputs. They are
intentionally deterministic and offline-safe.
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit


_MISSING = {"", "none", "null", "nan", "n/a", "na", "unknown", "publisher_unknown"}
_OFFICIAL_SOURCE_TYPES = {
    "official_public_health_agency",
    "national_public_health_agency",
    "state_or_local_public_health_agency",
    "international_public_health_agency",
    "international_organization_report",
    "structured_database",
}
_ANNUAL_COUNT_TYPES = {
    "annual",
    "annual_total",
    "annual total",
    "annual cumulative",
    "cumulative_annual",
    "year_total",
}
_NEWLY_REPORTED_TYPES = {"newly_reported", "newly reported", "incident_report"}


def classify_single_source_record(record: dict) -> str:
    """Return a credibility-aware single-source taxonomy label."""

    status = _norm(record.get("corroboration_status"))
    if status in {"corroborated", "cross_source_supported", "multi_source_corroborated"}:
        return "multi_source_corroborated"
    if status in {"conflicting_claims", "conflicting_sources"}:
        return "conflicting_sources"
    if record.get("requires_human_review") or record.get("human_review_reason"):
        return "needs_human_review"
    has_provenance = _has_value(record.get("source_url")) and _has_value(
        record.get("evidence_quote")
    )
    publisher = _publisher(record)
    source_type = _norm(record.get("source_type_final") or record.get("source_type"))
    official = source_type in _OFFICIAL_SOURCE_TYPES or _known_official_publisher(
        publisher
    )
    if has_provenance and official:
        return "official_single_source_not_cross_validated"
    return "non_official_single_source_unverified"


def compare_open_pilot_records(
    *,
    workflow_records: list[dict],
    benchmark_records: list[dict],
    source_candidates: list[dict],
    quarantined_records: list[dict] | None = None,
    evidence_chunks: list[dict] | None = None,
    raw_records: list[dict] | None = None,
    normalized_records: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    """Compare open-run records with benchmark records using diagnostic statuses."""

    rows: list[dict] = []
    matched_workflow_ids: set[str] = set()
    quarantined_records = quarantined_records or []
    evidence_chunks = evidence_chunks or []
    raw_records = raw_records or []
    normalized_records = normalized_records or workflow_records

    for benchmark in benchmark_records:
        candidates = [
            record
            for record in workflow_records
            if _same_disease(record, benchmark)
            and _same_location(record, benchmark)
            and _same_period(record, benchmark)
        ]
        status = "benchmark_missing_from_workflow"
        workflow = candidates[0] if candidates else None
        review_required = True
        reason = "No comparable workflow record was found for the benchmark row."
        extraction_miss_subtype = None

        if candidates:
            workflow = _best_workflow_match(candidates, benchmark)
            matched_workflow_ids.add(_record_id(workflow))
            if not _compatible_count_type(workflow, benchmark):
                status = "not_comparable_count_type"
                reason = "Disease, location, and period align but count semantics differ."
            elif _same_case_count(workflow, benchmark):
                if _source_url(workflow) == _source_url(benchmark):
                    status = "match"
                    review_required = False
                    reason = "Workflow and benchmark agree on source and count."
                elif _same_official_publisher(workflow, benchmark):
                    status = "source_equivalent_match"
                    review_required = False
                    reason = (
                        "Workflow and benchmark agree on disease, geography, period, "
                        "count, and official agency, but use different URLs."
                    )
                else:
                    status = "strong_partial_match"
                    reason = (
                        "Workflow and benchmark agree on disease, geography, period, "
                        "and count, but source equivalence needs review."
                    )
            else:
                status = "count_mismatch_review_required"
                reason = "Comparable workflow and benchmark records have different counts."
        elif _is_vessel_or_travel_benchmark(benchmark):
            if _has_quarantined_vessel_record(quarantined_records, benchmark):
                status = "quality_gate_or_semantics_miss"
                extraction_miss_subtype = classify_extraction_miss_subtype(
                    benchmark_record=benchmark,
                    source_candidates=source_candidates,
                    evidence_chunks=evidence_chunks,
                    raw_records=raw_records,
                    normalized_records=normalized_records,
                    quarantined_records=quarantined_records,
                )
                reason = (
                    "A vessel/travel-associated candidate record exists outside the "
                    "accepted workflow records."
                )
            elif _has_relevant_official_source_candidate(source_candidates, benchmark):
                status = "extraction_miss"
                extraction_miss_subtype = classify_extraction_miss_subtype(
                    benchmark_record=benchmark,
                    source_candidates=source_candidates,
                    evidence_chunks=evidence_chunks,
                    raw_records=raw_records,
                    normalized_records=normalized_records,
                    quarantined_records=quarantined_records,
                )
                reason = (
                    "A relevant official source candidate was discovered, but no "
                    "accepted benchmark-comparable workflow record was extracted. "
                    f"subtype={extraction_miss_subtype}."
                )
            else:
                status = "source_discovery_miss"
                reason = (
                    "No WHO/CDC/DON/HAN-like official source candidate was present "
                    "for the vessel/travel-associated benchmark."
                )

        rows.append(
            _comparison_row(
                benchmark=benchmark,
                workflow=workflow,
                status=status,
                reason=reason,
                human_review_flag=review_required,
                extraction_miss_subtype=extraction_miss_subtype,
            )
        )

    for workflow in workflow_records:
        workflow_id = _record_id(workflow)
        if workflow_id in matched_workflow_ids:
            continue
        status = (
            "workflow_extra_supported"
            if _supported_workflow_extra(workflow)
            else "workflow_extra_needs_review"
        )
        rows.append(
            _comparison_row(
                benchmark=None,
                workflow=workflow,
                status=status,
                reason=(
                    "Workflow found a source-backed record outside the current "
                    "benchmark set; retain for human review or benchmark expansion."
                ),
                human_review_flag=True,
            )
        )

    status_counts = Counter(row["open_pilot_match_status"] for row in rows)
    subtype_counts = Counter(
        row.get("extraction_miss_subtype")
        for row in rows
        if row.get("extraction_miss_subtype")
    )
    summary = {
        "evaluation_method": "open_pilot_diagnostic_matcher_v1",
        "workflow_record_count": len(workflow_records),
        "benchmark_record_count": len(benchmark_records),
        "evaluation_row_count": len(rows),
        "open_pilot_match_status_counts": dict(status_counts),
        "extraction_miss_subtype_counts": dict(subtype_counts),
        "workflow_extra_supported_count": status_counts.get(
            "workflow_extra_supported", 0
        ),
        "source_discovery_miss_count": status_counts.get("source_discovery_miss", 0),
        "extraction_miss_count": status_counts.get("extraction_miss", 0),
        "quality_gate_or_semantics_miss_count": status_counts.get(
            "quality_gate_or_semantics_miss", 0
        ),
        "human_review_flagged_row_count": sum(
            1 for row in rows if row.get("human_review_flag")
        ),
    }
    return rows, summary


def classify_extraction_miss_subtype(
    *,
    benchmark_record: dict,
    source_candidates: list[dict],
    evidence_chunks: list[dict] | None = None,
    raw_records: list[dict] | None = None,
    normalized_records: list[dict] | None = None,
    quarantined_records: list[dict] | None = None,
) -> str:
    """Classify why an official-source benchmark row is not comparable.

    The function only reads workflow artifacts; it never injects benchmark data
    into collection outputs.
    """

    evidence_chunks = evidence_chunks or []
    raw_records = raw_records or []
    normalized_records = normalized_records or []
    quarantined_records = quarantined_records or []

    relevant_sources = [
        source
        for source in source_candidates
        if _candidate_relevant_to_benchmark(source, benchmark_record)
    ]
    relevant_source_ids = {
        str(source.get("source_id") or "")
        for source in relevant_sources
        if source.get("source_id")
    }
    relevant_chunks = [
        chunk
        for chunk in evidence_chunks
        if _artifact_relevant_to_benchmark(
            chunk,
            benchmark_record,
            relevant_source_ids=relevant_source_ids,
        )
    ]
    if relevant_sources and not relevant_chunks:
        return "source_found_but_no_relevant_chunks"
    if not relevant_sources and not relevant_chunks:
        return "source_found_but_no_relevant_chunks"

    relevant_raw = [
        record
        for record in raw_records
        if _artifact_relevant_to_benchmark(
            record,
            benchmark_record,
            relevant_source_ids=relevant_source_ids,
        )
    ]
    if not relevant_raw:
        return "relevant_chunks_found_but_no_raw_record"

    relevant_normalized = [
        record
        for record in normalized_records
        if _artifact_relevant_to_benchmark(
            record,
            benchmark_record,
            relevant_source_ids=relevant_source_ids,
        )
    ]
    if not relevant_normalized:
        return "raw_record_extracted_but_schema_invalid"

    if any(_outside_time_window(record) for record in relevant_normalized):
        return "outside_time_window"
    if any(not _compatible_count_type(record, benchmark_record) for record in relevant_normalized):
        return "incompatible_count_type"
    if any(_is_non_primary_observation(record) for record in relevant_normalized):
        return "non_primary_observation_only"

    relevant_quarantined = [
        record
        for record in quarantined_records
        if _artifact_relevant_to_benchmark(
            record,
            benchmark_record,
            relevant_source_ids=relevant_source_ids,
        )
    ]
    if relevant_quarantined:
        return "normalized_record_but_quality_gate_blocked"

    return "record_extracted_but_not_benchmark_comparable"


def discover_benchmark_source_candidates(benchmark_records: list[dict]) -> list[dict]:
    """Return benchmark source candidates without adding them to collection."""

    candidates: list[dict] = []
    for index, record in enumerate(benchmark_records, start=1):
        text = _record_text(record)
        mentions_who_don = "who" in text and (
            "disease outbreak news" in text or re.search(r"\bdon\s*\d+", text)
        )
        mentions_vessel = _is_vessel_or_travel_benchmark(record)
        if not (mentions_who_don or mentions_vessel):
            continue
        source_url = str(record.get("source_url") or "").strip()
        candidates.append(
            {
                "candidate_id": f"benchmark_source_candidate_{index:03d}",
                "benchmark_id": record.get("benchmark_id")
                or record.get("record_id")
                or f"benchmark_{index:03d}",
                "candidate_source_url": source_url or None,
                "candidate_publisher": "World Health Organization"
                if "who" in text or "who.int" in source_url.lower()
                else None,
                "candidate_source_type": "official_public_health_agency"
                if "who" in text or "who.int" in source_url.lower()
                else None,
                "benchmark_source_identified": "candidate",
                "requires_human_confirmation": True,
                "missing_source_url": not bool(source_url),
                "possible_don_id_mismatch": _possible_don_id_mismatch(text, source_url),
                "source_discovery_use": "diagnostic_only_not_collection_input",
            }
        )
    return candidates


def write_benchmark_source_candidates(
    candidates: list[dict],
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Benchmark Source Candidates",
        "",
        "These candidates are diagnostic-only and must not be injected into collection output.",
        "",
    ]
    if not candidates:
        lines.append("No benchmark source candidates were identified.")
    for candidate in candidates:
        lines.extend(
            [
                f"## {candidate.get('candidate_id')}",
                f"- benchmark_id: `{candidate.get('benchmark_id')}`",
                f"- benchmark_source_identified: `{candidate.get('benchmark_source_identified')}`",
                f"- requires_human_confirmation: `{candidate.get('requires_human_confirmation')}`",
                f"- missing_source_url: `{candidate.get('missing_source_url')}`",
                f"- possible_don_id_mismatch: `{candidate.get('possible_don_id_mismatch')}`",
                f"- candidate_source_url: `{candidate.get('candidate_source_url') or ''}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_open_pilot_evaluation_outputs(
    rows: list[dict],
    summary: dict,
    output_dir: Path | str,
    *,
    prefix: str = "open_rerun",
) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{prefix}_evaluation_report.csv"
    json_path = out_dir / f"{prefix}_evaluation_summary.json"
    md_path = out_dir / f"{prefix}_readable_evaluation_report.md"
    fieldnames = _fieldnames(rows)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(_evaluation_markdown(rows, summary), encoding="utf-8")
    return {
        "evaluation_report_csv": str(csv_path),
        "evaluation_summary_json": str(json_path),
        "readable_evaluation_report_md": str(md_path),
    }


def _comparison_row(
    *,
    benchmark: dict | None,
    workflow: dict | None,
    status: str,
    reason: str,
    human_review_flag: bool,
    extraction_miss_subtype: str | None = None,
) -> dict:
    return {
        "benchmark_id": (benchmark or {}).get("benchmark_id")
        or (benchmark or {}).get("record_id")
        or "",
        "workflow_record_id": _record_id(workflow or {}),
        "disease": (workflow or benchmark or {}).get("disease") or "",
        "location": _display_location(workflow or benchmark or {}),
        "reporting_period": (workflow or benchmark or {}).get("reporting_period")
        or "",
        "workflow_case_count": _case_count(workflow or {}),
        "benchmark_case_count": _case_count(benchmark or {}),
        "workflow_source_url": _source_url(workflow or {}),
        "benchmark_source_url": _source_url(benchmark or {}),
        "workflow_publisher": _publisher(workflow or {}),
        "benchmark_publisher": _publisher(benchmark or {}),
        "workflow_count_type": _count_type(workflow or {}),
        "benchmark_count_type": _count_type(benchmark or {}),
        "open_pilot_match_status": status,
        "extraction_miss_subtype": extraction_miss_subtype or "",
        "human_review_flag": human_review_flag,
        "review_reason": reason,
    }


def _best_workflow_match(candidates: list[dict], benchmark: dict) -> dict:
    def score(record: dict) -> tuple[int, int, int]:
        return (
            int(_same_case_count(record, benchmark)),
            int(_same_official_publisher(record, benchmark)),
            int(_source_url(record) == _source_url(benchmark)),
        )

    return sorted(candidates, key=score, reverse=True)[0]


def _same_disease(a: dict, b: dict) -> bool:
    return "hanta" in _norm(a.get("disease")) and "hanta" in _norm(b.get("disease"))


def _same_location(a: dict, b: dict) -> bool:
    if _is_vessel_or_travel_benchmark(a) and _is_vessel_or_travel_benchmark(b):
        return True
    a_values = _location_values(a)
    b_values = _location_values(b)
    return bool(a_values and b_values and (a_values & b_values))


def _same_period(a: dict, b: dict) -> bool:
    return bool(_period(a) and _period(a) == _period(b))


def _same_case_count(a: dict, b: dict) -> bool:
    return _case_count(a) is not None and _case_count(a) == _case_count(b)


def _compatible_count_type(a: dict, b: dict) -> bool:
    a_type = _count_type(a)
    b_type = _count_type(b)
    if not a_type or not b_type:
        return True
    if a_type == b_type:
        return True
    if a_type in _ANNUAL_COUNT_TYPES and b_type in _ANNUAL_COUNT_TYPES:
        return True
    if a_type in _NEWLY_REPORTED_TYPES or b_type in _NEWLY_REPORTED_TYPES:
        return False
    return True


def _same_official_publisher(a: dict, b: dict) -> bool:
    publisher_a = _publisher(a)
    publisher_b = _publisher(b)
    return (
        _has_value(publisher_a)
        and publisher_a == publisher_b
        and _known_official_publisher(publisher_a)
    )


def _supported_workflow_extra(record: dict) -> bool:
    return (
        _has_value(record.get("source_url"))
        and _has_value(record.get("evidence_quote"))
        and _has_value(_publisher(record))
    )


def _candidate_relevant_to_benchmark(candidate: dict, benchmark: dict) -> bool:
    text = _record_text(candidate)
    benchmark_text = _record_text(benchmark)
    if any(
        token in text
        for token in (
            "who.int",
            "cdc.gov",
            "paho.org",
            "ecdc.europa.eu",
            "disease outbreak news",
            " don",
            " han",
            "epidemiological alert",
        )
    ):
        if "mv hondius" not in benchmark_text:
            return True
        return "hondius" in text or any(
            token in text for token in ("cruise ship", "vessel", "travel associated")
        )
    if "mv hondius" in benchmark_text and "hondius" in text:
        return True
    return False


def _artifact_relevant_to_benchmark(
    artifact: dict,
    benchmark: dict,
    *,
    relevant_source_ids: set[str],
) -> bool:
    source_id = str(artifact.get("source_id") or "")
    if source_id and source_id in relevant_source_ids:
        return True
    text = _record_text(artifact)
    benchmark_text = _record_text(benchmark)
    if "mv hondius" in benchmark_text:
        return "hondius" in text or (
            "cruise ship" in text and "hantavirus" in text
        )
    return "hantavirus" in text or "hps" in text or "andes virus" in text


def _outside_time_window(record: dict) -> bool:
    status = _norm(
        record.get("period_overlap_status")
        or record.get("record_period_fit_status")
        or record.get("record_task_fit_status")
        or record.get("quarantine_reason")
    )
    return "outside time window" in status or "outside_time_window" in status


def _is_non_primary_observation(record: dict) -> bool:
    if record.get("primary_case_dataset_eligible") is True:
        return False
    values = []
    if record.get("observation_type"):
        values.append(str(record.get("observation_type")))
    values.extend(str(value) for value in record.get("observation_types") or [])
    text = _norm(" ".join(values))
    return any(
        token in text
        for token in (
            "surveillance summary",
            "outbreak summary",
            "background context",
            "exposure monitoring",
            "non primary",
        )
    )


def _has_relevant_official_source_candidate(
    source_candidates: list[dict],
    benchmark: dict,
) -> bool:
    benchmark_text = _record_text(benchmark)
    for candidate in source_candidates:
        text = _record_text(candidate)
        if any(token in text for token in ("who.int", "cdc.gov", "disease outbreak news", " don", " han")):
            return True
        if "mv hondius" in benchmark_text and "hondius" in text:
            return True
    return False


def _has_quarantined_vessel_record(records: list[dict], benchmark: dict) -> bool:
    benchmark_text = _record_text(benchmark)
    for record in records:
        text = _record_text(record)
        if "mv hondius" in benchmark_text and "hondius" in text:
            return True
        if _is_vessel_or_travel_benchmark(record) and _same_period(record, benchmark):
            return True
    return False


def _is_vessel_or_travel_benchmark(record: dict) -> bool:
    text = _record_text(record)
    return any(
        token in text
        for token in (
            "mv hondius",
            "vessel",
            "cruise ship",
            "travel-associated",
            "travel associated",
        )
    )


def _possible_don_id_mismatch(text: str, source_url: str) -> bool:
    combined = f"{text} {source_url}".lower()
    don_ids = set(re.findall(r"\bdon\s*[-_ ]?(\d+)\b", combined))
    url_ids = set(re.findall(r"\bdon\s*[-_ ]?(\d+)\b", source_url.lower()))
    return bool(len(don_ids) > 1 or (don_ids and url_ids and don_ids != url_ids))


def _location_values(record: dict) -> set[str]:
    values: set[str] = set()
    for key in ("locality", "subnational_location", "country", "geographic_scope", "location"):
        value = _norm(record.get(key))
        if value and value not in _MISSING:
            values.add(value)
    return values


def _display_location(record: dict) -> str:
    for key in ("locality", "subnational_location", "geographic_scope", "country", "location"):
        value = record.get(key)
        if _has_value(value):
            return str(value)
    return ""


def _period(record: dict) -> str:
    for key in ("reporting_period", "date_anchor", "event_start_date", "date_reported"):
        value = record.get(key)
        if _has_value(value):
            text = str(value)
            year_match = re.search(r"\b(20\d{2})\b", text)
            return year_match.group(1) if year_match else _norm(text)
    return ""


def _count_type(record: dict) -> str:
    return _norm(record.get("statistical_count_type"))


def _case_count(record: dict):
    for key in ("cases_confirmed", "cases_unspecified", "cases_probable", "cases_suspected"):
        value = record.get(key)
        if _has_value(value):
            return _number(value)
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return int(number) if number.is_integer() else number


def _publisher(record: dict) -> str:
    for key in ("actual_publisher", "publisher", "source_publisher"):
        value = _norm(record.get(key))
        if value and value not in _MISSING:
            return value
    return ""


def _known_official_publisher(publisher: str) -> bool:
    return publisher in {
        "centers for disease control and prevention",
        "world health organization",
        "pan american health organization",
        "european centre for disease prevention and control",
        "new mexico department of health",
        "california department of public health",
        "virginia department of health",
    }


def _source_url(record: dict) -> str:
    return str(record.get("source_url") or record.get("canonical_url") or "").strip()


def _record_id(record: dict) -> str:
    return str(record.get("record_id") or record.get("source_record_id") or "").strip()


def _record_text(record: dict) -> str:
    values = []
    for key, value in record.items():
        if isinstance(value, (str, int, float)):
            values.append(str(value))
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
    url = _source_url(record)
    if url:
        parts = urlsplit(url)
        values.append(parts.netloc)
        values.append(parts.path)
    return _norm(" ".join(values))


def _norm(value) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    return " ".join(text.split())


def _has_value(value) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return _norm(value) not in _MISSING
    return True


def _csv_value(value) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def _fieldnames(rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    for row in rows:
        for key in row:
            if key not in ordered:
                ordered.append(key)
    return ordered or ["open_pilot_match_status"]


def _evaluation_markdown(rows: list[dict], summary: dict) -> str:
    lines = [
        "# Open Rerun Evaluation Report",
        "",
        "This diagnostic compares open-run workflow artifacts with repo-local benchmark metadata.",
        "",
        "## Summary",
        "",
    ]
    for key in (
        "workflow_record_count",
        "benchmark_record_count",
        "evaluation_row_count",
        "human_review_flagged_row_count",
    ):
        lines.append(f"- {key}: `{summary.get(key, 0)}`")
    lines.extend(["", "## Match Status Counts", ""])
    for status, count in sorted((summary.get("open_pilot_match_status_counts") or {}).items()):
        lines.append(f"- {status}: `{count}`")
    lines.extend(["", "## Rows", ""])
    for row in rows[:25]:
        lines.append(
            "- "
            f"{row.get('open_pilot_match_status')}: "
            f"benchmark={row.get('benchmark_id') or 'none'}, "
            f"workflow={row.get('workflow_record_id') or 'none'}, "
            f"reason={row.get('review_reason')}"
        )
    return "\n".join(lines)
