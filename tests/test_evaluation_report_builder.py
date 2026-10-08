"""Tests for deterministic masked-validation evaluation reporting."""

from __future__ import annotations

import subprocess
import sys
import json
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow.evaluation_report_builder import (  # noqa: E402
    EVALUATION_FIELDNAMES,
    build_evaluation_report,
    build_evaluation_review_items,
    write_evaluation_outputs,
)


def _record(**overrides) -> dict:
    record = {
        "record_id": "rec_001",
        "linked_event_id": "event_001",
        "disease": "Hantavirus disease",
        "virus_or_syndrome": None,
        "country": "Country X",
        "subnational_location": None,
        "date_anchor": "2023",
        "date_anchor_field": "date_reported",
        "date_reported": "2023",
        "reporting_period": None,
        "statistical_count_type": None,
        "cases_unspecified": 12,
        "deaths": 2,
        "source_id": "src_collection",
        "source_url": "https://example.org/report",
        "evidence_quote": "Country X reported 12 cases and 2 deaths.",
        "supporting_chunk_id": "chunk_001",
    }
    record.update(overrides)
    return record


def test_evaluation_report_builder_smoke():
    rows, summary = build_evaluation_report(
        collection_records=[_record()],
        validation_records=[_record(record_id="val_001", source_id="src_validation")],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["masking_compliance_status"] == "passed"
    assert row["overall_match_status"] == "match"
    assert row["human_review_flag"] is False
    assert summary["human_review_flagged_row_count"] == 0


def test_evaluation_report_builder_flags_reserved_source_leakage():
    rows, summary = build_evaluation_report(
        collection_records=[_record(source_id="src_reserved")],
        validation_records=[_record(record_id="val_001", source_id="src_validation")],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert (
        row["masking_compliance_status"]
        == "failed_validation_source_in_collection"
    )
    assert row["human_review_flag"] is True
    assert summary["reserved_source_leakage_source_ids"] == ["src_reserved"]


def test_evaluation_report_builder_missing_collection_record():
    rows, summary = build_evaluation_report(
        collection_records=[],
        validation_records=[_record(record_id="val_001", source_id="src_reserved")],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["overall_match_status"] == "missing_collection_record"
    assert row["human_review_flag"] is True
    assert (
        row["provenance_completeness_status"]
        == "not_applicable_no_collection_record"
    )
    assert summary["human_review_flagged_row_count"] == 1


def test_evaluation_review_items_are_built_from_flagged_rows():
    rows, summary = build_evaluation_report(
        collection_records=[_record(cases_unspecified=12, deaths=2)],
        validation_records=[
            _record(
                record_id="val_001",
                source_id="src_validation",
                cases_unspecified=13,
                deaths=2,
            )
        ],
        reserved_source_ids={"src_reserved"},
    )

    items = build_evaluation_review_items(rows)

    assert summary["human_review_flagged_row_count"] == 1
    assert len(items) == 1
    item = items[0]
    assert item["review_id"] == "review_validation_eval_001"
    assert item["item_type"] == "masked_validation"
    assert item["status"] == "pending"
    assert "eval_001" in item["related_ids"]
    assert "case_count" in item["reason"]
    assert "differ" in item["reason"].lower()
    assert item["review_packet"]["packet_sections"]["evaluation_row"] == rows[0]


def test_evaluation_report_builder_not_comparable_case_prevents_clean_match():
    rows, _summary = build_evaluation_report(
        collection_records=[_record(cases_unspecified=12, deaths=2)],
        validation_records=[
            _record(record_id="val_001", source_id="src_validation_a", cases_unspecified=12),
            _record(record_id="val_002", source_id="src_validation_b", cases_unspecified=13),
        ],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert "case_count_not_comparable" in row["field_level_match_status"]
    assert "death_count_match" in row["field_level_match_status"]
    assert row["overall_match_status"] == "partial_match_not_comparable"
    assert row["human_review_flag"] is True
    assert "not comparable" in row["review_reason"]


def test_evaluation_report_builder_all_numeric_fields_not_comparable():
    rows, _summary = build_evaluation_report(
        collection_records=[
            _record(record_id="rec_001", cases_unspecified=12, deaths=None),
            _record(record_id="rec_002", cases_unspecified=13, deaths=None),
        ],
        validation_records=[
            _record(record_id="val_001", source_id="src_validation", cases_unspecified=None, deaths=None),
        ],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["overall_match_status"] == "insufficient_comparable_values"
    assert row["human_review_flag"] is True
    assert "Neither side has comparable numeric values." in row["review_reason"]


def test_evaluation_report_builder_clean_match_still_no_review():
    rows, _summary = build_evaluation_report(
        collection_records=[_record(cases_unspecified=12, deaths=2)],
        validation_records=[
            _record(record_id="val_001", source_id="src_validation", cases_unspecified=12, deaths=2),
        ],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["overall_match_status"] == "match"
    assert row["human_review_flag"] is False
    assert row["review_reason"] == ""


def test_annual_collection_record_matches_validation_by_reporting_period_even_if_reported_later():
    collection = _record(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2026-03-12",
        date_anchor_field="date_reported",
        reporting_period="2025",
        statistical_count_type="annual",
        cases_unspecified=7,
        deaths=3,
        source_id="src_collection",
    )
    validation = _record(
        record_id="val_001",
        source_id="src_validation",
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2025",
        date_anchor_field="reporting_period",
        reporting_period="2025",
        statistical_count_type="annual",
        cases_unspecified=7,
        deaths=3,
    )

    rows, summary = build_evaluation_report(
        collection_records=[collection],
        validation_records=[validation],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["collection_case_count"] == "7"
    assert row["validation_case_count"] == "7"
    assert row["collection_death_count"] == "3"
    assert row["validation_death_count"] == "3"
    assert "case_count_match" in row["field_level_match_status"]
    assert "death_count_match" in row["field_level_match_status"]
    assert row["overall_match_status"] == "match"
    assert row["human_review_flag"] is False
    assert summary["rows_with_both_collection_and_validation_evidence_count"] == 1


def test_annual_alignment_does_not_merge_newly_reported_record():
    collection = _record(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2026-03-12",
        date_anchor_field="date_reported",
        reporting_period="2026",
        statistical_count_type="newly_reported",
        cases_confirmed=1,
        cases_unspecified=None,
        deaths=None,
        source_id="src_collection",
    )
    validation = _record(
        record_id="val_001",
        source_id="src_validation",
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2025",
        date_anchor_field="reporting_period",
        reporting_period="2025",
        statistical_count_type="annual",
        cases_unspecified=7,
        deaths=3,
    )

    rows, summary = build_evaluation_report(
        collection_records=[collection],
        validation_records=[validation],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 2
    assert summary["overall_match_status_counts"] == {
        "missing_collection_record": 1,
        "missing_validation_record": 1,
    }


def test_annual_alignment_preserves_masking_compliance():
    collection = _record(
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2026-03-12",
        date_anchor_field="date_reported",
        reporting_period="2025",
        statistical_count_type="annual",
        cases_unspecified=7,
        deaths=3,
        source_id="src_reserved",
    )
    validation = _record(
        record_id="val_001",
        source_id="src_validation",
        disease="Hantavirus disease",
        virus_or_syndrome="HPS",
        country="United States of America",
        subnational_location="New Mexico",
        date_reported="2026-03-12",
        date_anchor="2025",
        date_anchor_field="reporting_period",
        reporting_period="2025",
        statistical_count_type="annual",
        cases_unspecified=7,
        deaths=3,
    )

    rows, summary = build_evaluation_report(
        collection_records=[collection],
        validation_records=[validation],
        reserved_source_ids={"src_reserved"},
    )

    assert len(rows) == 1
    row = rows[0]
    assert (
        row["masking_compliance_status"]
        == "failed_validation_source_in_collection"
    )
    assert row["overall_match_status"] == "match"
    assert row["human_review_flag"] is True
    assert summary["reserved_source_leakage_count"] == 1


def test_write_evaluation_outputs_creates_files(tmp_path):
    rows, summary = build_evaluation_report(
        collection_records=[_record()],
        validation_records=[_record(record_id="val_001", source_id="src_validation")],
        reserved_source_ids={"src_reserved"},
    )

    manifest = write_evaluation_outputs(rows, summary, tmp_path)

    report_csv = tmp_path / "evaluation_report.csv"
    summary_json = tmp_path / "evaluation_summary.json"
    markdown = tmp_path / "readable_evaluation_report.md"
    assert report_csv.exists()
    assert summary_json.exists()
    assert markdown.exists()
    assert set(manifest) == {
        "evaluation_report_csv",
        "evaluation_summary_json",
        "readable_evaluation_report_md",
    }
    header = report_csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[: len(EVALUATION_FIELDNAMES)] == EVALUATION_FIELDNAMES
    assert "Masked Validation Evaluation Report" in markdown.read_text(encoding="utf-8")


def test_readable_evaluation_report_includes_collection_and_validation_preview(tmp_path):
    rows, summary = build_evaluation_report(
        collection_records=[_record(source_id="src_collection", cases_unspecified=12)],
        validation_records=[
            _record(record_id="val_001", source_id="src_validation", cases_unspecified=13)
        ],
        reserved_source_ids={"src_reserved"},
    )

    # The readable view must use the same limitations as the machine summary,
    # including task-specific compatibility findings.
    summary["workflow_limitations"] = ["The supplied reference covers only March."]
    write_evaluation_outputs(rows, summary, tmp_path)

    report = (tmp_path / "readable_evaluation_report.md").read_text(encoding="utf-8")
    assert "## Evaluation Row Preview" in report
    assert "collection_source_ids=src_collection" in report
    assert "validation_source_ids=src_validation" in report
    assert "overall_match_status=mismatch" in report
    assert "human_review_flag=true" in report
    assert "collection_evidence_quote_preview=" in report
    assert "validation_evidence_quote_preview=" in report
    assert summary["workflow_limitations"][0] in report


def test_configured_workflow_script_uses_config_without_runtime_confirmations(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_workflow.py"
    assert script.exists(), f"missing {script}"

    # This tests CLI/config execution, not a paid live discovery run. Disabling
    # fetch alone intentionally allows search-only use, so provide explicit
    # offline discovery configuration instead of inheriting mutable user config.
    from data_collection_workflow.runtime_profile import default_workflow_run_config
    config = default_workflow_run_config()
    config['structured_task'] = {"disease": "measles", "location": "Canada", "start_date": "2024-01-01", "end_date": "2024-12-31"}
    config['source_search'].update(enabled=False, mode='disabled')
    config_path = tmp_path / 'offline-cli.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')
    session_id = "test_session"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--disable-live-fetch",
            "--disable-all-llm",
            "--write-run-notebook",
            "--output-dir",
            str(tmp_path),
            "--session-id",
            session_id,
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    session_dir = tmp_path / "sessions" / session_id
    assert (session_dir / "final_report.html").exists()
    assert not (session_dir / "workflow_run_report.md").exists()
    assert not list(session_dir.glob("*_chinese.md"))
    assert (session_dir / "workflow_run_summary.json").exists()
    notebook = json.loads((session_dir / 'workflow_replay_notebook.ipynb').read_text(encoding='utf-8'))
    links = next(cell['source'] for cell in notebook['cells'] if '## Artifact Links' in cell['source'])
    assert 'final_report.html' in links and 'session_report.zip' in links
    assert (session_dir / "workflow_console" / "data_collection_workflow_console.html").exists()
    review_items = json.loads(
        (session_dir / "collection" / "human_review_items.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(review_items, list)
    summary = json.loads(
        (session_dir / "workflow_run_summary.json").read_text(encoding="utf-8")
    )
    assert summary["human_review_item_count"] == len(review_items)
    assert summary["human_review_enabled"] is False
    assert summary["validation_mode"] == "live_cross_source"
    assert "Pass --allow-live-fetch" not in result.stderr


def test_configured_workflow_script_prints_sanitized_config_without_running():
    project_root = Path(__file__).resolve().parents[1]
    script = project_root / "scripts" / "run_workflow.py"

    result = subprocess.run(
        [sys.executable, str(script), "--print-config-only"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "sanitized_environment:" in result.stdout
    assert "studio_minimal_input:" in result.stdout
    assert "ENABLE_LIVE_FETCH" in result.stdout
    assert "ENABLE_LLM_SOURCE_PLANNING" in result.stdout
    assert "ANTHROPIC_API_KEY" not in result.stdout
