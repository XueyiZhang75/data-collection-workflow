from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data_collection_workflow import llm_clients  # noqa: E402
from data_collection_workflow.models import LLMExtractedRecord, LLMExtractionOutput  # noqa: E402
from data_collection_workflow.nodes.extraction import (  # noqa: E402
    schema_validation_and_repair,
    structured_extraction,
)
from data_collection_workflow.nodes.content_processing import (  # noqa: E402
    document_quality_check,
    evidence_chunking_and_data_presence_flagging,
)


def _flu_state(chunks: list[dict], **overrides) -> dict:
    state = {
        "user_request": "Collect FLU surveillance data for Virginia.",
        "structured_task": {
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "Virginia",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
            "time_window": "2024-10-01 to 2024-10-10",
            "collection_mode": "direct_collection",
        },
        "disease_intelligence": {
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "pathogen_terms": ["influenza", "influenza A", "influenza B"],
            "syndrome_terms": ["influenza-like illness"],
        },
        "evidence_chunks": chunks,
        "must_fetch_sources": [
            {
                "source_id": "src_vdh_week_40",
                "must_fetch": True,
                "must_fetch_reason": "target jurisdiction official weekly surveillance report",
                "coverage_requirement_ids": ["req_vdh_week_40"],
            }
        ],
        "source_registry": [
            {
                "source_id": "src_vdh_week_40",
                "canonical_url": "https://www.vdh.virginia.gov/report.pdf",
                "title": "VDH Weekly RDS Report Week 40",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "must_fetch": True,
                "credibility_level": "high",
            }
        ],
        "collection_trace": [],
    }
    state.update(overrides)
    return state


def _chunk(source_id: str, index: int, **overrides) -> dict:
    row = {
        "chunk_id": f"chunk_{source_id}_{index}",
        "source_id": source_id,
        "text": (
            "Virginia influenza surveillance reported 42 positive laboratory "
            "tests and 7 hospitalizations during MMWR week 40."
        ),
        "contains_target_data": True,
        "data_types": ["testing_count", "hospitalization_count"],
        "disease_relevance_status": "target_disease_match",
        "extraction_eligible_for_task_disease": True,
        "source_url": f"https://example.org/{source_id}.pdf",
        "source_type": "official_public_health_agency",
        "source_type_final": "state_or_local_public_health_agency",
        "source_role_final": "collection",
        "fetch_purpose": "data_extraction",
        "chunk_kind": "text",
        "title": "Weekly surveillance report",
        "publisher": "Virginia Department of Health",
        "chunk_index": index,
    }
    row.update(overrides)
    return row


def _fake_output(chunk: dict, _policy) -> LLMExtractionOutput:
    return LLMExtractionOutput(
        records=[
            LLMExtractedRecord(
                disease="Seasonal influenza",
                virus_or_syndrome="influenza",
                country="United States of America",
                subnational_location="Virginia",
                date_reported="2024-10-05",
                reporting_period="MMWR week 40, 2024",
                tests_positive=42.0,
                hospitalizations=7.0,
                count_semantics="weekly aggregate surveillance count",
                aggregation_level="subnational",
                geographic_scope="Virginia",
                geographic_scope_type="subnational",
            )
        ],
        chunk_is_relevant=True,
    )


def _enable_direct_llm(
    monkeypatch,
    *,
    max_chunks: int = 3,
    safety_max_calls: int | None = None,
) -> None:
    monkeypatch.setenv("COLLECTION_MODE", "direct_collection")
    monkeypatch.setenv("ENABLE_LLM_EXTRACTION", "true")
    monkeypatch.setenv("LLM_MAX_CHUNKS", str(max_chunks))
    monkeypatch.setenv(
        "LLM_EXTRACTION_SAFETY_MAX_CALLS",
        str(safety_max_calls if safety_max_calls is not None else max_chunks),
    )
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")


def test_direct_collection_prioritizes_must_fetch_official_chunks_before_safety_cap(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=3, safety_max_calls=3)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk("src_forum_context", i, source_type="forum", publisher="Forum")
        for i in range(1, 8)
    ] + [
        _chunk("src_vdh_week_40", 1),
        _chunk("src_vdh_week_40", 2),
    ]

    result = structured_extraction(_flu_state(chunks))
    summary = result["structured_extraction_summary"]

    assert calls == [
        "src_vdh_week_40",
        "src_vdh_week_40",
        "src_forum_context",
    ]
    assert summary["llm_call_count"] == 3
    assert summary["scheduler_model_call_count"] == 3
    assert summary["scheduler_cache_hit_count"] == 2
    assert summary["extraction_budget_ledger"]["actual_model_call_count"] == 3
    assert summary["extraction_budget_ledger"]["rolling_actual_model_calls"] == 3
    vdh_budget = summary["extraction_budget_by_source"]["src_vdh_week_40"]
    assert vdh_budget["attempted_count"] == 2
    assert vdh_budget["budget_bucket"] == "verified_target_collection"
    assert vdh_budget["target_fit_status"] == "verified_target"
    assert vdh_budget["attempted_before_target_sources"] is False
    assert summary["official_extraction_queue"][0]["source_id"] == "src_vdh_week_40"


def test_structured_extraction_executes_selected_primary_spans_concurrently(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=8, safety_max_calls=8)
    monkeypatch.setenv("LLM_EXTRACTION_MAX_CONCURRENCY", "4")
    active = 0
    peak_active = 0
    lock = threading.Lock()

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = []
    source_registry = []
    for index in range(1, 5):
        source_id = f"src_vdh_concurrent_{index}"
        chunks.append(
            _chunk(
                source_id,
                index,
                text=(
                    "Virginia influenza surveillance reported "
                    f"{40 + index} positive laboratory tests and "
                    f"{index} hospitalizations during MMWR week 40."
                ),
            )
        )
        source_registry.append(
            {
                "source_id": source_id,
                "canonical_url": f"https://www.vdh.virginia.gov/report-{index}.pdf",
                "title": f"VDH Weekly RDS Report {index}",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "credibility_level": "high",
            }
        )

    result = structured_extraction(
        _flu_state(chunks, source_registry=source_registry, must_fetch_sources=[])
    )

    assert result["structured_extraction_summary"]["scheduler_model_call_count"] >= 4
    assert peak_active >= 2


def test_direct_collection_skips_context_extraction_when_target_source_unusable(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=3)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    cdc_context_chunk = _chunk(
        "src_cdc_week_42_context",
        1,
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-42.html",
        canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-42.html",
        title="CDC FluView Week 42 national report",
        publisher="CDC",
        source_type="official_public_health_agency",
        source_type_final="national_public_health_agency",
        source_role_final="context",
    )
    state = _flu_state(
        [cdc_context_chunk],
        source_registry=[
            {
                "source_id": "src_cdc_week_42_context",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-42.html",
                "title": "CDC FluView Week 42 national report",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "context",
                "credibility_level": "high",
            }
        ],
        source_coverage_audit={
            "coverage_status": "target_official_source_unusable",
            "accepted_record_count": 0,
        },
    )

    result = structured_extraction(state)

    summary = result["structured_extraction_summary"]
    assert calls == []
    assert summary["raw_record_count"] == 0
    assert summary["no_task_collection_document"] is True
    assert summary["extraction_blocking_reason"] == "no_task_collection_document"
    assert summary["skipped_context_extraction_count"] == 1
    budget = summary["extraction_budget_by_source"]["src_cdc_week_42_context"]
    assert budget["budget_bucket"] == "official_or_high_trust"
    assert budget["target_fit_status"] == "non_target_or_context"


def test_direct_collection_skips_context_extraction_when_coverage_is_partial(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=3)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    cdc_context_chunk = _chunk(
        "src_cdc_week_43_context",
        1,
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-43.html",
        canonical_url="https://www.cdc.gov/fluview/surveillance/2024-week-43.html",
        title="CDC FluView Week 43 national report",
        publisher="CDC",
        source_type="official_public_health_agency",
        source_type_final="national_public_health_agency",
        source_role_final="context",
    )
    state = _flu_state(
        [cdc_context_chunk],
        source_registry=[
            {
                "source_id": "src_cdc_week_43_context",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-43.html",
                "title": "CDC FluView Week 43 national report",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "context",
                "credibility_level": "high",
            }
        ],
        source_coverage_audit={
            "coverage_status": "partial_target_coverage",
            "coverage_completeness_status": "partial_target_coverage",
            "accepted_record_count": 17,
            "missing_requirement_ids": [
                "virginia_influenza_official_week_43_2024"
            ],
        },
    )

    result = structured_extraction(state)

    summary = result["structured_extraction_summary"]
    assert calls == []
    assert summary["raw_record_count"] == 0
    assert summary["skipped_context_extraction_count"] == 1
    budget = summary["extraction_budget_by_source"]["src_cdc_week_43_context"]
    assert budget["budget_bucket"] == "official_or_high_trust"
    assert budget["target_fit_status"] == "non_target_or_context"


def test_direct_collection_llm_extraction_receives_task_acceptance_contract(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)
    seen_contracts: list[dict] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        seen_contracts.append(dict(chunk.get("task_acceptance_contract") or {}))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    state = _flu_state(
        [_chunk("src_vdh_week_40", 1)],
        task_acceptance_contract={
            "contract_version": "v1",
            "record_acceptance_rules": ["must_match_task_location"],
            "context_or_quarantine_rules": [
                "national_or_broader_aggregate_without_target_location_fit"
            ],
        },
    )

    result = structured_extraction(state)

    assert seen_contracts == [state["task_acceptance_contract"]]
    assert result["record_task_fit_assessments"]
    assert result["structured_extraction_summary"][
        "record_task_fit_assessment_count"
    ] == 1


def test_direct_collection_extracts_ed_visit_percent_as_generic_metric(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)

    def fake_extract(_chunk: dict, _policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    date_reported=None,
                    reporting_period="MMWR week 40, 2024",
                    metric_name="nssp_ed_visit_percent",
                    metric_value=0.2,
                    metric_unit="percent",
                    metric_category="ed_visit_percent",
                    metric_denominator="emergency_department_visits",
                    metric_period_start="2024-09-29",
                    metric_period_end="2024-10-05",
                    positivity_rate=None,
                    count_semantics="weekly ED visit percent",
                    aggregation_level="national",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunk = _chunk(
        "src_cdc_week_40",
        1,
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        title="CDC FluView Week 40",
        publisher="CDC",
        text="NSSP emergency department visits for influenza were 0.2% in week 40.",
    )

    result = structured_extraction(
        _flu_state(
            [chunk],
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
        )
    )

    record = result["raw_records"][0]
    assert record["metric_name"] == "nssp_ed_visit_percent"
    assert record["metric_value"] == 0.2
    assert record["metric_unit"] == "percent"
    assert record["metric_category"] == "ed_visit_percent"
    assert record["metric_denominator"] == "emergency_department_visits"
    assert record["metric_period_end"] == "2024-10-05"
    assert record["positivity_rate"] is None
    assessment = result["record_task_fit_assessments"][0]
    assert assessment["count_semantics_fit"] == "has_interpretable_metric"
    assert assessment["date_fit"] == "has_date_or_period"
    assert result["metric_extraction_plan"]["metric_record_count"] == 1
    assert result["metric_row_extraction_audit"][0]["metric_category"] == "ed_visit_percent"


def test_direct_collection_fills_missing_metric_period_from_verified_source(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)

    def fake_extract(_chunk: dict, _policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    date_reported=None,
                    reporting_period="MMWR week 40, 2024",
                    metric_name="clinical_lab_positive_specimens",
                    metric_value=380,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    metric_denominator="clinical_laboratory_specimens",
                    count_semantics="weekly clinical laboratory positives",
                    aggregation_level="national",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunk = _chunk(
        "src_cdc_week_40",
        1,
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        title="CDC FluView Week 40, 2024",
        publisher="CDC",
    )

    result = structured_extraction(
        _flu_state(
            [chunk],
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            must_fetch_sources=[{"source_id": "src_cdc_week_40", "must_fetch": True}],
            source_registry=[
                {
                    "source_id": "src_cdc_week_40",
                    "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                    "title": "CDC FluView Week 40, 2024",
                    "publisher": "CDC",
                    "source_type": "official_public_health_agency",
                    "source_type_final": "national_public_health_agency",
                    "source_role_final": "collection",
                    "must_fetch": True,
                    "credibility_level": "high",
                }
            ],
        )
    )

    record = result["raw_records"][0]
    audit = result["metric_row_extraction_audit"][0]
    assert record["metric_period_start"] == "2024-09-29"
    assert record["metric_period_end"] == "2024-10-05"
    assert record["metric_period_source"] == "filled_from_source_reporting_period"
    assert "filled_metric_period_from_source_reporting_period" in record["semantic_warnings"]
    assert audit["metric_period_source"] == "filled_from_source_reporting_period"


def test_markdown_metric_table_rows_create_table_chunks_without_parsed_tables():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_cdc_week_40",
                "source_id": "src_cdc_week_40",
                "url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "clean_text": (
                    "Influenza surveillance table\n\n"
                    "| Metric | Value | Week |\n"
                    "| --- | ---: | --- |\n"
                    "| Clinical lab positive specimens | 380 | 40 |\n"
                    "| Percent of emergency department visits for influenza | 0.2% | 40 |\n"
                ),
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    table_chunks = [
        chunk
        for chunk in result["evidence_chunks"]
        if chunk.get("chunk_kind") == "metric_row"
    ]

    assert table_chunks
    assert result["evidence_chunking_summary"]["markdown_metric_row_chunk_count"] >= 2
    assert all("markdown_table_row" in (chunk.get("context_types") or []) for chunk in table_chunks)
    assert all(chunk.get("row_id") for chunk in table_chunks)
    assert all(chunk.get("row_quote") for chunk in table_chunks)
    assert any("Clinical lab positive specimens" in chunk["text"] for chunk in table_chunks)


def test_markdown_metric_table_rows_preserve_header_column_labels():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_cdc_week_40",
                "source_id": "src_cdc_week_40",
                "url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "must_fetch": True,
                "reporting_period_start": "2024-09-29",
                "reporting_period_end": "2024-10-05",
                "clean_text": (
                    "Clinical Laboratories\n\n"
                    "| Metric | Current Week | Previous Week |\n"
                    "| --- | ---: | ---: |\n"
                    "| No. of positive specimens (%) | 380 (0.7%) | 264 (0.5%) |\n"
                ),
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    rows = [
        chunk
        for chunk in result["evidence_chunks"]
        if chunk.get("chunk_kind") == "metric_row"
        and "positive specimens" in (chunk.get("row_quote") or "")
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["source_column_labels"] == ["Current Week", "Previous Week"]
    assert row["table_header"] == "| Metric | Current Week | Previous Week |"
    assert row["heading_context"] == "Clinical Laboratories"


def test_plain_markdown_metric_lines_create_metric_row_chunks_without_pipe_table():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_vdh_week_40",
                "source_id": "src_vdh_week_40",
                "url": "https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-40.pdf",
                "title": "Virginia Department of Health Weekly RDS Report Week 40",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "must_fetch": True,
                "reporting_period_start": "2024-09-29",
                "reporting_period_end": "2024-10-05",
                "clean_text": (
                    "Virginia Respiratory Disease Surveillance Report\n\n"
                    "Influenza-like illness: 2.1% of emergency department visits\n"
                    "Influenza-associated outbreaks reported: 3\n"
                    "Positive influenza specimens: 42 (4.2%)\n"
                    "Hospital admissions for influenza: 7\n"
                ),
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    metric_rows = [
        chunk
        for chunk in result["evidence_chunks"]
        if chunk.get("chunk_kind") == "metric_row"
    ]

    assert len(metric_rows) >= 4
    assert result["evidence_chunking_summary"]["markdown_metric_row_chunk_count"] >= 4
    assert all("markdown_metric_line" in (chunk.get("context_types") or []) for chunk in metric_rows)
    assert all(chunk.get("contains_target_data") is True for chunk in metric_rows)
    assert all(chunk.get("extraction_eligible_for_task_disease") is True for chunk in metric_rows)
    assert any("Positive influenza specimens" in chunk["row_quote"] for chunk in metric_rows)


def test_navigation_date_under_metric_heading_is_not_a_metric_row():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_navigation_noise",
                "source_id": "src_navigation_noise",
                "url": "https://example.gov/flu/report",
                "title": "Influenza surveillance report",
                "publisher": "Example health authority",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "clean_text": (
                    "Influenza cases\n"
                    "Last updated 10 October 2024\n"
                    "About this website\n"
                    "Contact 555-0100\n"
                    "Confirmed influenza cases: 12\n"
                ),
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    row_quotes = {
        chunk.get("row_quote")
        for chunk in result["evidence_chunks"]
        if chunk.get("chunk_kind") == "metric_row"
    }

    assert "Confirmed influenza cases: 12" in row_quotes
    assert "Last updated 10 October 2024" not in row_quotes
    assert "Contact 555-0100" not in row_quotes


def test_evidence_chunks_expose_scheduler_quality_signals():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_case_signal",
                "source_id": "src_case_signal",
                "url": "https://example.gov/flu/case-update",
                "title": "Influenza case update",
                "publisher": "Example health authority",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "clean_text": "Patient 2 had influenza confirmed by PCR on 5 May 2024.",
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    chunk = next(row for row in result["evidence_chunks"] if row["contains_target_data"])

    assert chunk["signal_strength"] in {"strong", "moderate"}
    assert chunk["local_disease_status"] == "target_disease_match"
    assert 0.0 <= chunk["boilerplate_ratio"] <= 1.0
    assert chunk["extraction_priority"] <= 2


def test_verified_collection_metric_rows_inherit_document_task_relevance():
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_cdc_week_40",
                "source_id": "src_cdc_week_40",
                "url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_markdown",
                "quality_status": "usable",
                "document_disease_relevance_status": "target_disease_match",
                "must_fetch": True,
                "clean_text": (
                    "| Metric | Current week | Previous week |\n"
                    "| --- | ---: | ---: |\n"
                    "| No. of specimens tested | 53,699 | 53,424 |\n"
                    "| No. of positive specimens | 1,319 | 1,248 |\n"
                ),
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)
    rows = [
        chunk
        for chunk in result["evidence_chunks"]
        if chunk.get("chunk_kind") == "metric_row"
    ]

    assert rows
    assert all(row["contains_target_data"] is True for row in rows)
    assert all(row["extraction_eligible_for_task_disease"] is True for row in rows)
    assert all(row["disease_relevance_status"] == "target_disease_match" for row in rows)
    assert all("inherited_document_task_relevance" in row["context_types"] for row in rows)


def test_direct_collection_batches_metric_rows_and_preserves_row_quotes(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=10)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("chunk_id")))
        assert chunk.get("chunk_kind") == "metric_row_batch"
        assert len(chunk.get("metric_rows") or []) == 3
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    subnational_location="Virginia",
                    metric_name="Clinical lab positive specimens",
                    metric_value=42,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    metric_period_start="2024-10-01",
                    metric_period_end="2024-10-05",
                    source_row_id="row_positive",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="subnational",
                    geographic_scope="Virginia",
                    geographic_scope_type="subnational",
                ),
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    subnational_location="Virginia",
                    metric_name="Hospitalizations",
                    metric_value=7,
                    metric_unit="count",
                    metric_category="hospitalization_count",
                    metric_period_start="2024-10-01",
                    metric_period_end="2024-10-05",
                    source_row_id="row_hosp",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="subnational",
                    geographic_scope="Virginia",
                    geographic_scope_type="subnational",
                ),
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_vdh_week_40",
            1,
            chunk_id="chunk_row_positive",
            chunk_kind="metric_row",
            row_id="row_positive",
            table_id="table_lab",
            row_quote="| Clinical lab positive specimens | 42 | Week 40 |",
            text="| Clinical lab positive specimens | 42 | Week 40 |",
            context_types=["markdown_table_row"],
        ),
        _chunk(
            "src_vdh_week_40",
            2,
            chunk_id="chunk_row_hosp",
            chunk_kind="metric_row",
            row_id="row_hosp",
            table_id="table_lab",
            row_quote="| Hospitalizations | 7 | Week 40 |",
            text="| Hospitalizations | 7 | Week 40 |",
            context_types=["markdown_table_row"],
        ),
        _chunk(
            "src_vdh_week_40",
            3,
            chunk_id="chunk_row_ed",
            chunk_kind="metric_row",
            row_id="row_ed",
            table_id="table_lab",
            row_quote="| ED visit percent | 0.2% | Week 40 |",
            text="| ED visit percent | 0.2% | Week 40 |",
            context_types=["markdown_table_row"],
        ),
    ]

    result = structured_extraction(_flu_state(chunks))

    assert calls == ["batch_src_vdh_week_40_table_lab_1"]
    summary = result["structured_extraction_summary"]
    plan = result["metric_extraction_plan"]
    assert summary["llm_call_count"] == 1
    assert summary["batched_metric_row_call_count"] == 1
    assert plan["batched_metric_row_call_count"] == 1
    assert plan["metric_row_chunk_count"] == 3
    by_metric = {record["metric_name"]: record for record in result["raw_records"]}
    assert by_metric["Clinical lab positive specimens"]["evidence_quote"] == (
        "| Clinical lab positive specimens | 42 | Week 40 |"
    )
    assert by_metric["Hospitalizations"]["evidence_quote"] == (
        "| Hospitalizations | 7 | Week 40 |"
    )
    assert by_metric["Clinical lab positive specimens"]["supporting_chunk_id"] == (
        "chunk_row_positive"
    )
    assert by_metric["Clinical lab positive specimens"]["chunk_kind"] == "metric_row"


def test_direct_collection_deterministically_splits_count_percent_metric_rows(monkeypatch):
    llm_calls: list[dict] = []

    def fake_extract_chunk_with_llm(chunk, _policy):
        llm_calls.append(chunk)
        return LLMExtractionOutput(records=[])

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_vdh_week_40",
        1,
        chunk_id="chunk_positive_percent",
        chunk_kind="metric_row",
        row_id="row_positive_percent",
        table_id="clinical_labs",
        text="| **No. of positive specimens (%)** | 223 (0.5%) | 352,980 (8.5%) |",
        row_quote="| **No. of positive specimens (%)** | 223 (0.5%) | 352,980 (8.5%) |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        source_column_labels=["Current Week", "Season to Date / Cumulative"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(_flu_state([row]))

    records = result["raw_records"]
    assert llm_calls == []
    assert len(records) == 4
    by_category = {}
    for record in records:
        by_category.setdefault(
            (record["metric_category"], record["source_column_label"]),
            record,
        )

    assert by_category[("lab_positive_count", "Current Week")]["metric_value"] == 223.0
    assert by_category[("lab_positivity_percent", "Current Week")]["metric_value"] == 0.5
    assert (
        by_category[("lab_positive_count", "Season to Date / Cumulative")][
            "metric_value"
        ]
        == 352980.0
    )
    assert (
        by_category[("lab_positivity_percent", "Season to Date / Cumulative")][
            "metric_value"
        ]
        == 8.5
    )
    assert (
        by_category[("lab_positive_count", "Season to Date / Cumulative")][
            "statistical_count_type"
        ]
        == "cumulative"
    )
    assert (
        by_category[("lab_positive_count", "Season to Date / Cumulative")][
            "reporting_period"
        ]
        == "Season to Date / Cumulative through MMWR week 40, 2024"
    )
    assert all(record["source_row_id"] == "row_positive_percent" for record in records)
    assert all(record["metric_row_binding_status"] == "resolved" for record in records)
    audit_rows = result["metric_row_extraction_audit"]
    assert len(audit_rows) == 4
    assert all(
        audit["row_quote"]
        == "| **No. of positive specimens (%)** | 223 (0.5%) | 352,980 (8.5%) |"
        for audit in audit_rows
    )
    assert all(audit["evidence_quote"] == audit["row_quote"] for audit in audit_rows)
    assert result["structured_extraction_summary"]["deterministic_metric_row_record_count"] == 4


def test_metric_row_splitter_marks_unlabeled_multi_value_columns_ambiguous(monkeypatch):
    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(records=[])

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_positive_percent",
        chunk_kind="metric_row",
        row_id="row_positive_percent",
        table_id="clinical_labs",
        text="| **No. of positive specimens (%)** | 56 (8.8%) | 154 (13.8%) |",
        row_quote="| **No. of positive specimens (%)** | 56 (8.8%) | 154 (13.8%) |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(_flu_state([row]))

    records = result["raw_records"]
    assert records
    assert {record["source_column_label"] for record in records} == {
        "Column 1",
        "Column 2",
    }
    assert all(
        record["metric_column_semantics_status"] == "ambiguous"
        for record in records
    )
    assert all(
        record["resolved_column_period_type"] == "ambiguous_column"
        for record in records
    )
    assert all(
        "ambiguous_metric_column_semantics" in record["semantic_warnings"]
        for record in records
    )
    assert {
        audit["metric_column_semantics_status"]
        for audit in result["metric_row_extraction_audit"]
    } == {"ambiguous"}


def test_metric_row_splitter_recovers_weekly_lab_comparison_columns(monkeypatch):
    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(records=[])

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_positive_percent",
        chunk_kind="metric_row",
        row_id="row_positive_percent",
        table_id="clinical_labs",
        text="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        row_quote="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        heading_context="Results of tests from Clinical Laboratories",
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(_flu_state([row]))

    records = result["raw_records"]
    assert len(records) == 4
    by_category_and_label = {
        (record["metric_category"], record["source_column_label"]): record
        for record in records
    }
    current_positive = by_category_and_label[
        ("lab_positive_count", "Current Week")
    ]
    current_percent = by_category_and_label[
        ("lab_positivity_percent", "Current Week")
    ]
    previous_positive = by_category_and_label[
        ("lab_positive_count", "Previous Week")
    ]
    previous_percent = by_category_and_label[
        ("lab_positivity_percent", "Previous Week")
    ]

    assert current_positive["metric_value"] == 380.0
    assert current_percent["metric_value"] == 0.7
    assert current_positive["resolved_column_period_type"] == "current_period"
    assert current_percent["metric_period_start"] == "2024-09-29"
    assert current_percent["metric_period_end"] == "2024-10-05"

    assert previous_positive["metric_value"] == 264.0
    assert previous_percent["metric_value"] == 0.5
    assert previous_positive["resolved_column_period_type"] == "previous_period"
    assert previous_percent["metric_period_start"] == "2024-09-22"
    assert previous_percent["metric_period_end"] == "2024-09-28"
    assert all(
        record["metric_column_semantics_status"] == "resolved"
        for record in records
    )

    tested_row = _chunk(
        "src_cdc_week_40",
        2,
        chunk_id="chunk_specimens_tested",
        chunk_kind="metric_row",
        row_id="row_specimens_tested",
        table_id="clinical_labs",
        text="| **No. of specimens tested** | 53,699 | 53,424 |",
        row_quote="| **No. of specimens tested** | 53,699 | 53,424 |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        heading_context="Results of tests from Clinical Laboratories",
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    tested_result = structured_extraction(_flu_state([tested_row]))
    tested_records = tested_result["raw_records"]
    assert len(tested_records) == 2
    tested_by_label = {
        record["source_column_label"]: record
        for record in tested_records
    }
    current_tested = tested_by_label["Current Week"]
    previous_tested = tested_by_label["Previous Week"]

    assert current_tested["metric_category"] == "lab_test_count"
    assert current_tested["metric_value"] == 53699.0
    assert current_tested["resolved_column_period_type"] == "current_period"
    assert current_tested["metric_period_start"] == "2024-09-29"
    assert current_tested["metric_period_end"] == "2024-10-05"

    assert previous_tested["metric_category"] == "lab_test_count"
    assert previous_tested["metric_value"] == 53424.0
    assert previous_tested["resolved_column_period_type"] == "previous_period"
    assert previous_tested["metric_period_start"] == "2024-09-22"
    assert previous_tested["metric_period_end"] == "2024-09-28"
    assert all(
        record["metric_column_semantics_status"] == "resolved"
        for record in tested_records
    )
    assert {
        audit["column_semantics_resolution_method"]
        for audit in result["metric_row_extraction_audit"]
    } == {"column_label"}


def test_metric_row_splitter_resolves_previous_week_period_from_column_label(monkeypatch):
    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(records=[])

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_positive_percent",
        chunk_kind="metric_row",
        row_id="row_positive_percent",
        table_id="clinical_labs",
        text="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        row_quote="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        source_column_labels=["Current Week", "Previous Week"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(_flu_state([row]))

    by_label = {record["source_column_label"]: record for record in result["raw_records"]}
    assert by_label["Current Week"]["metric_period_start"] == "2024-09-29"
    assert by_label["Current Week"]["metric_period_end"] == "2024-10-05"
    assert by_label["Current Week"]["resolved_column_period_type"] == "current_period"
    assert by_label["Current Week"]["metric_column_semantics_status"] == "resolved"
    assert by_label["Previous Week"]["metric_period_start"] == "2024-09-22"
    assert by_label["Previous Week"]["metric_period_end"] == "2024-09-28"
    assert (
        by_label["Previous Week"]["metric_period_source"]
        == "filled_from_previous_column_label"
    )
    assert by_label["Previous Week"]["resolved_column_period_type"] == "previous_period"
    assert by_label["Previous Week"]["metric_column_semantics_status"] == "resolved"


def test_narrative_week_metric_resolves_current_period_semantics(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=3)

    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                    reporting_period="MMWR week 40, 2024",
                    metric_name=(
                        "Percentage of ED visits with discharge diagnosis of influenza"
                    ),
                    metric_value=0.2,
                    metric_unit="percent",
                    metric_category="ed_visit_percent",
                    count_semantics="weekly emergency department visit percent",
                    statistical_count_type="weekly",
                    source_row_id="row_ed_visit_percent",
                    observation_type="surveillance_summary",
                    observation_types=["surveillance_summary"],
                    primary_case_dataset_eligible=False,
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_ed_visit_percent",
        chunk_kind="metric_row",
        row_id="row_ed_visit_percent",
        table_id="markdown_metric_lines_0",
        text=(
            "The percentage of emergency department (ED) visits with a "
            "discharge diagnosis of influenza reported in NSSP was 0.2% "
            "overall during Week 40."
        ),
        row_quote=(
            "The percentage of emergency department (ED) visits with a "
            "discharge diagnosis of influenza reported in NSSP was 0.2% "
            "overall during Week 40."
        ),
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["public_health_metric", "rate_or_percent_metric"],
        context_types=["markdown_metric_line", "syndromic_surveillance_metric"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(
        _flu_state(
            [row],
            user_request="Collect flu surveillance metrics for United States.",
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "time_window": "2024-09-29 to 2024-10-05",
                "collection_mode": "direct_collection",
            },
        )
    )

    record = result["raw_records"][0]
    assert record["metric_category"] == "ed_visit_percent"
    assert record["metric_column_semantics_status"] == "resolved"
    assert record["resolved_column_period_type"] == "current_period"
    assert record["metric_period_start"] == "2024-09-29"
    assert record["metric_period_end"] == "2024-10-05"
    assert record["metric_period_source"] == "filled_from_narrative_week_phrase"
    audit = result["metric_row_extraction_audit"][0]
    assert audit["row_context_type"] == "markdown_metric_line"
    assert audit["column_semantics_resolution_method"] == "narrative_week_phrase"


def test_narrative_week_zero_death_metric_resolves_current_period_semantics(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=3)

    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                    reporting_period="MMWR week 40, 2024",
                    metric_name="Influenza-associated pediatric deaths",
                    metric_value=0,
                    metric_unit="count",
                    metric_category="death_count",
                    count_semantics="weekly death count",
                    statistical_count_type="weekly",
                    source_row_id="row_week_40_deaths",
                    observation_type="surveillance_summary",
                    observation_types=["surveillance_summary"],
                    primary_case_dataset_eligible=False,
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_week_40_deaths",
        chunk_kind="metric_row",
        row_id="row_week_40_deaths",
        table_id="markdown_metric_lines_0",
        text=(
            "No influenza-associated pediatric deaths occurring during the "
            "2024-2025 season were reported to CDC during Week 40."
        ),
        row_quote=(
            "No influenza-associated pediatric deaths occurring during the "
            "2024-2025 season were reported to CDC during Week 40."
        ),
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["public_health_metric", "death_count"],
        context_types=["markdown_metric_line"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
    )

    result = structured_extraction(
        _flu_state(
            [row],
            user_request="Collect flu surveillance metrics for United States.",
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "time_window": "2024-09-29 to 2024-10-05",
                "collection_mode": "direct_collection",
            },
        )
    )

    record = result["raw_records"][0]
    assert record["metric_category"] == "death_count"
    assert record["metric_column_semantics_status"] == "resolved"
    assert record["resolved_column_period_type"] == "current_period"
    assert record["metric_period_source"] == "filled_from_narrative_week_phrase"


def test_direct_metric_row_records_inherit_task_geography_before_schema_validation(monkeypatch):
    def fake_extract_chunk_with_llm(chunk, _policy):
        return LLMExtractionOutput(records=[])

    monkeypatch.setattr(
        llm_clients,
        "extract_chunk_with_llm",
        fake_extract_chunk_with_llm,
    )

    row = _chunk(
        "src_cdc_week_40",
        1,
        chunk_id="chunk_positive_percent",
        chunk_kind="metric_row",
        row_id="row_positive_percent",
        table_id="clinical_labs",
        text="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        row_quote="| **No. of positive specimens (%)** | 380 (0.7%) | 264 (0.5%) |",
        contains_target_data=True,
        extraction_eligible_for_task_disease=True,
        disease_relevance_status="target_disease_match",
        data_types=["laboratory_metric"],
        context_types=["table", "metric_row"],
        source_column_labels=["Current Week", "Season to Date / Cumulative"],
        reporting_period_start="2024-09-29",
        reporting_period_end="2024-10-05",
        reporting_period_label="MMWR week 40, 2024",
        source_url="https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
        source_type="official_public_health_agency",
        source_type_final="national_public_health_agency",
        title="CDC FluView Week 40",
        publisher="Centers for Disease Control and Prevention",
        source_role_final="collection",
        credibility_level="high",
    )
    state = _flu_state(
        [row],
        user_request="Collect flu surveillance metrics for United States.",
        structured_task={
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "time_window": "2024-09-29 to 2024-10-05",
            "collection_mode": "direct_collection",
        },
        must_fetch_sources=[
            {
                "source_id": "src_cdc_week_40",
                "must_fetch": True,
                "coverage_requirement_ids": ["req_cdc_week_40"],
            }
        ],
        source_registry=[
            {
                "source_id": "src_cdc_week_40",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40",
                "publisher": "Centers for Disease Control and Prevention",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "must_fetch": True,
                "credibility_level": "high",
            }
        ],
        source_coverage_requirements=[
            {
                "requirement_id": "req_cdc_week_40",
                "location": "United States",
                "official_candidate_urls": [
                    "https://www.cdc.gov/fluview/surveillance/2024-week-40.html"
                ],
                "reporting_period_start": "2024-09-29",
                "reporting_period_end": "2024-10-05",
                "reporting_period_label": "MMWR week 40, 2024",
            }
        ],
    )

    extraction_result = structured_extraction(state)
    validation_result = schema_validation_and_repair(
        {**state, "raw_records": extraction_result["raw_records"]}
    )

    positivity_records = [
        record
        for record in validation_result["validated_records"]
        if record.get("metric_category") == "lab_positivity_percent"
    ]
    assert positivity_records
    assert all(record["country"] == "United States" for record in positivity_records)
    assert all(record["geographic_scope"] == "United States" for record in positivity_records)
    assert all(record["metric_period_start"] == "2024-09-29" for record in positivity_records)
    assert all(record["metric_period_end"] == "2024-10-05" for record in positivity_records)
    assert not [
        record
        for record in validation_result["rejected_records"]
        if record.get("metric_category") == "lab_positivity_percent"
    ]


def test_metric_record_with_start_after_end_is_rejected_by_schema_validation():
    raw_record = {
        "record_id": "rec_invalid_period",
        "disease": "Seasonal influenza",
        "virus_or_syndrome": "influenza",
        "country": "United States",
        "geographic_scope": "United States",
        "geographic_scope_type": "country",
        "source_url": "https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
        "source_type": "official_public_health_agency",
        "source_id": "src_cdc_week_41",
        "evidence_quote": "| Percent positive specimens | 0.8% | Week 41 |",
        "metric_name": "Percent positive specimens",
        "metric_value": 0.8,
        "metric_unit": "percent",
        "metric_category": "lab_positivity_percent",
        "metric_period_start": "2024-10-06",
        "metric_period_end": "2024-10-05",
        "metric_period_source": "filled_from_source_reporting_period",
        "source_row_id": "row_positive_percent",
        "metric_row_binding_status": "resolved",
        "schema_status": None,
        "provenance_status": None,
        "missing_fields": [],
    }

    result = schema_validation_and_repair(
        _flu_state([], raw_records=[raw_record])
    )

    assert result["validated_records"] == []
    rejected = result["rejected_records"][0]
    assert rejected["schema_status"] == "rejected"
    assert "metric_period_invalid_start_after_end" in rejected["validation_errors"]


def test_metric_record_with_ambiguous_multi_week_quote_is_rejected_without_row_period_metadata():
    raw_record = {
        "record_id": "rec_ambiguous_multi_week",
        "disease": "Seasonal influenza",
        "virus_or_syndrome": "influenza",
        "country": "United States",
        "geographic_scope": "United States",
        "geographic_scope_type": "country",
        "source_url": "https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
        "source_type": "official_public_health_agency",
        "source_id": "src_cdc_week_41",
        "evidence_quote": (
            "No influenza-associated pediatric deaths were reported during "
            "Week 40. No influenza-associated pediatric deaths were reported "
            "during Week 41."
        ),
        "metric_name": "Influenza-associated pediatric deaths",
        "metric_value": 0,
        "metric_unit": "count",
        "metric_category": "death_count",
        "metric_period_start": "2024-10-06",
        "metric_period_end": "2024-10-12",
        "metric_period_source": "filled_from_source_reporting_period",
        "metric_row_binding_status": "resolved",
        "schema_status": None,
        "provenance_status": None,
        "missing_fields": [],
    }

    result = schema_validation_and_repair(
        _flu_state([], raw_records=[raw_record])
    )

    assert result["validated_records"] == []
    rejected = result["rejected_records"][0]
    assert rejected["schema_status"] == "rejected"
    assert "ambiguous_multi_period_metric_quote" in rejected["validation_errors"]


def test_direct_collection_fills_metric_period_from_source_reporting_period(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=10)

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        source_id = str(chunk.get("source_id") or "")
        row_id = "week_40_positive" if source_id == "src_cdc_week_40" else "week_41_positive"
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    metric_name="Clinical lab positive specimens",
                    metric_value=42 if source_id == "src_cdc_week_40" else 84,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    source_row_id=row_id,
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="national",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_cdc_week_40",
            1,
            chunk_id="chunk_week_40_positive",
            chunk_kind="metric_row",
            row_id="week_40_positive",
            table_id="table_lab",
            row_quote="| Clinical lab positive specimens | 42 | Week 40 |",
            text="| Clinical lab positive specimens | 42 | Week 40 |",
            reporting_period_start="2024-09-29",
            reporting_period_end="2024-10-05",
            reporting_period_label="MMWR week 40, 2024",
            period_basis="week_ending_saturday",
        ),
        _chunk(
            "src_cdc_week_41",
            1,
            chunk_id="chunk_week_41_positive",
            chunk_kind="metric_row",
            row_id="week_41_positive",
            table_id="table_lab",
            row_quote="| Clinical lab positive specimens | 84 | Week 41 |",
            text="| Clinical lab positive specimens | 84 | Week 41 |",
            reporting_period_start="2024-10-06",
            reporting_period_end="2024-10-12",
            reporting_period_label="MMWR week 41, 2024",
            period_basis="week_ending_saturday",
        ),
    ]

    result = structured_extraction(
        _flu_state(
            chunks,
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-12",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-12",
                "time_window": "2024-09-29 to 2024-10-12",
                "collection_mode": "direct_collection",
            },
            must_fetch_sources=[
                {
                    "source_id": "src_cdc_week_40",
                    "must_fetch": True,
                    "coverage_requirement_ids": ["req_cdc_week_40"],
                },
                {
                    "source_id": "src_cdc_week_41",
                    "must_fetch": True,
                    "coverage_requirement_ids": ["req_cdc_week_41"],
                },
            ],
            source_registry=[
                {
                    "source_id": "src_cdc_week_40",
                    "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                    "title": "CDC FluView Week 40",
                    "publisher": "Centers for Disease Control and Prevention",
                    "source_type": "official_public_health_agency",
                    "source_role_final": "collection",
                    "must_fetch": True,
                    "credibility_level": "high",
                },
                {
                    "source_id": "src_cdc_week_41",
                    "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
                    "title": "CDC FluView Week 41",
                    "publisher": "Centers for Disease Control and Prevention",
                    "source_type": "official_public_health_agency",
                    "source_role_final": "collection",
                    "must_fetch": True,
                    "credibility_level": "high",
                },
            ],
        )
    )

    records = {record["source_id"]: record for record in result["raw_records"]}
    assert records["src_cdc_week_40"]["metric_period_start"] == "2024-09-29"
    assert records["src_cdc_week_40"]["metric_period_end"] == "2024-10-05"
    assert records["src_cdc_week_41"]["metric_period_start"] == "2024-10-06"
    assert records["src_cdc_week_41"]["metric_period_end"] == "2024-10-12"
    assert {
        records["src_cdc_week_40"]["metric_period_source"],
        records["src_cdc_week_41"]["metric_period_source"],
    } == {"filled_from_source_reporting_period"}


def test_direct_collection_marks_cumulative_column_as_cumulative_not_current_week(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=10)

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    metric_name="No. of positive specimens",
                    metric_value=779,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    source_row_id="row_positive_cumulative",
                    source_column_label="column_2",
                    metric_column_label="Season-to-date through Week 40",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="national",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_cdc_week_40",
            1,
            chunk_id="chunk_positive_cumulative",
            chunk_kind="metric_row",
            row_id="row_positive_cumulative",
            table_id="table_lab",
            row_quote="| No. of positive specimens | 359 | 779 |",
            text="| No. of positive specimens | 359 | 779 |",
            source_column_label="Cumulative since season start",
            metric_column_label="Season-to-date through Week 40",
            reporting_period_start="2024-09-29",
            reporting_period_end="2024-10-05",
            reporting_period_label="MMWR week 40, 2024",
            period_basis="week_ending_saturday",
        )
    ]

    result = structured_extraction(
        _flu_state(
            chunks,
            structured_task={
                "disease": "FLU",
                "location": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
            collection_spec={
                "disease": "FLU",
                "geography": "United States",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
                "collection_mode": "direct_collection",
            },
        )
    )

    record = result["raw_records"][0]
    assert record["metric_period_source"] == "filled_from_column_label"
    assert record["statistical_count_type"] == "cumulative"
    assert record["count_semantics"] == "cumulative"
    assert record["reporting_period"] == "Season-to-date through Week 40"
    assert "inferred_cumulative_metric_from_column_label" in record[
        "semantic_warnings"
    ]


def test_direct_collection_rebinds_metric_record_to_matching_row_quote(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=10)

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    metric_name="Number of positive specimens",
                    metric_value=359,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    metric_period_start="2024-09-29",
                    metric_period_end="2024-10-05",
                    source_row_id="row_tested",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="national",
                    geographic_scope="United States",
                    geographic_scope_type="country",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_cdc_week_40",
            1,
            chunk_id="chunk_tested",
            chunk_kind="metric_row",
            row_id="row_tested",
            table_id="table_lab",
            row_quote="| No. of specimens tested | 46,025 | 107,292 |",
            text="| No. of specimens tested | 46,025 | 107,292 |",
        ),
        _chunk(
            "src_cdc_week_40",
            2,
            chunk_id="chunk_positive",
            chunk_kind="metric_row",
            row_id="row_positive",
            table_id="table_lab",
            row_quote="| No. of positive specimens | 359 | 779 |",
            text="| No. of positive specimens | 359 | 779 |",
        ),
    ]

    result = structured_extraction(_flu_state(chunks))

    record = result["raw_records"][0]
    assert record["source_row_id"] == "row_positive"
    assert record["supporting_chunk_id"] == "chunk_positive"
    assert record["evidence_quote"] == "| No. of positive specimens | 359 | 779 |"
    assert record["metric_row_binding_status"] == "rebinding_resolved"
    assert "metric_row_rebound_from_llm_source_row_id" in record["semantic_warnings"]


def test_direct_collection_skips_text_fallback_after_enough_row_records(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=10)
    monkeypatch.setenv("DIRECT_MIN_TARGET_METRIC_RECORDS", "2")
    monkeypatch.setenv("DIRECT_ENABLE_TEXT_FALLBACK_AFTER_ROW_EXTRACTION", "false")
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("chunk_id")))
        assert chunk.get("chunk_kind") == "metric_row_batch"
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    subnational_location="Virginia",
                    metric_name="Clinical lab positive specimens",
                    metric_value=42,
                    metric_unit="count",
                    metric_category="lab_positive_count",
                    metric_period_start="2024-10-01",
                    metric_period_end="2024-10-05",
                    source_row_id="row_positive",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="subnational",
                    geographic_scope="Virginia",
                    geographic_scope_type="subnational",
                ),
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    subnational_location="Virginia",
                    metric_name="Hospitalizations",
                    metric_value=7,
                    metric_unit="count",
                    metric_category="hospitalization_count",
                    metric_period_start="2024-10-01",
                    metric_period_end="2024-10-05",
                    source_row_id="row_hosp",
                    count_semantics="weekly aggregate surveillance metric",
                    aggregation_level="subnational",
                    geographic_scope="Virginia",
                    geographic_scope_type="subnational",
                ),
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_vdh_week_40",
            1,
            chunk_id="chunk_row_positive",
            chunk_kind="metric_row",
            row_id="row_positive",
            table_id="table_lab",
            row_quote="| Clinical lab positive specimens | 42 | Week 40 |",
            text="| Clinical lab positive specimens | 42 | Week 40 |",
            context_types=["markdown_table_row"],
        ),
        _chunk(
            "src_vdh_week_40",
            2,
            chunk_id="chunk_row_hosp",
            chunk_kind="metric_row",
            row_id="row_hosp",
            table_id="table_lab",
            row_quote="| Hospitalizations | 7 | Week 40 |",
            text="| Hospitalizations | 7 | Week 40 |",
            context_types=["markdown_table_row"],
        ),
        _chunk(
            "src_vdh_week_40",
            3,
            chunk_id="chunk_text_fallback_1",
            chunk_kind="text",
            text="Long narrative text repeats the same two metrics.",
        ),
        _chunk(
            "src_vdh_week_40",
            4,
            chunk_id="chunk_text_fallback_2",
            chunk_kind="text",
            text="Another long narrative text repeats the same two metrics.",
        ),
    ]

    result = structured_extraction(_flu_state(chunks))

    assert calls == ["batch_src_vdh_week_40_table_lab_1"]
    summary = result["structured_extraction_summary"]
    plan = result["metric_extraction_plan"]
    assert summary["llm_call_count"] == 1
    assert summary["fallback_text_chunk_call_count"] == 0
    assert plan["fallback_text_chunk_call_count"] == 0
    assert plan["skipped_text_fallback_after_row_record_count"] == 2


def test_context_metric_rows_do_not_suppress_target_source_text_fallback(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=4)
    monkeypatch.setenv("DIRECT_MIN_TARGET_METRIC_RECORDS", "1")
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_cdc_week_41_context",
            1,
            chunk_id="chunk_cdc_context_row",
            chunk_kind="metric_row",
            row_id="row_cdc_ed",
            table_id="table_ed",
            row_quote="| Emergency department visits | 0.3% | Week 41 |",
            text="| Emergency department visits | 0.3% | Week 41 |",
            source_url="https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
            title="CDC FluView Week 41 national summary",
            publisher="CDC",
            source_type_final="national_public_health_agency",
            source_role_final="validation",
        ),
        _chunk(
            "src_vdh_week_41",
            1,
            chunk_id="chunk_vdh_text_1",
            chunk_kind="text",
            text=(
                "Virginia respiratory disease surveillance for MMWR week 41 "
                "reported influenza activity and laboratory detections."
            ),
            source_url="https://www.vdh.virginia.gov/epidemiology/influenza.pdf",
            title="VDH Weekly Respiratory Disease Surveillance Week 41",
            publisher="Virginia Department of Health",
            source_type_final="state_or_local_public_health_agency",
            source_role_final="collection",
        ),
    ]
    state = _flu_state(
        chunks,
        structured_task={
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
            "time_window": "2024-10-06 to 2024-10-12",
            "collection_mode": "direct_collection",
        },
        must_fetch_sources=[
            {
                "source_id": "src_vdh_week_41",
                "must_fetch": True,
                "coverage_requirement_ids": ["req_vdh_week_41"],
            }
        ],
        source_registry=[
            {
                "source_id": "src_cdc_week_41_context",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
                "title": "CDC FluView Week 41 national summary",
                "publisher": "CDC",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "validation",
                "target_fit_status": "non_target_or_context",
                "credibility_level": "high",
            },
            {
                "source_id": "src_vdh_week_41",
                "canonical_url": "https://www.vdh.virginia.gov/epidemiology/influenza.pdf",
                "title": "VDH Weekly Respiratory Disease Surveillance Week 41",
                "publisher": "Virginia Department of Health",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "must_fetch": True,
                "coverage_requirement_ids": ["req_vdh_week_41"],
                "credibility_level": "high",
            },
        ],
    )

    result = structured_extraction(state)
    summary = result["structured_extraction_summary"]
    budget = summary["extraction_budget_by_source"]

    assert calls == ["src_cdc_week_41_context", "src_vdh_week_41"]
    assert budget["src_vdh_week_41"]["attempted_count"] == 1
    assert budget["src_vdh_week_41"]["record_count"] == 1
    assert summary["metric_row_record_count_by_source"] == {
        "src_cdc_week_41_context": 1
    }
    assert summary["fallback_text_chunk_call_count"] == 1
    assert summary["official_extraction_failures"] == []


def test_task_record_candidate_metric_rows_are_extracted_without_complete_coverage(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=4)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Tuberculosis",
                    virus_or_syndrome="tuberculosis",
                    country="India",
                    metric_name="TB incidence rate",
                    metric_value=195,
                    metric_unit="per 100,000 population",
                    metric_category="incidence_rate",
                    metric_period_start="2023-01-01",
                    metric_period_end="2023-12-31",
                    source_row_id="row_tb_incidence",
                    count_semantics="annual public health metric",
                    aggregation_level="national",
                    geographic_scope="India",
                    geographic_scope_type="national",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_india_tb_report",
            1,
            chunk_id="chunk_tb_incidence",
            chunk_kind="metric_row",
            row_id="row_tb_incidence",
            table_id="table_tb_annual",
            row_quote="| India | TB incidence rate | 195 per 100,000 | 2023 |",
            text="| India | TB incidence rate | 195 per 100,000 | 2023 |",
            source_url="https://tbcindia.mohfw.gov.in/reports/tb-india-2024.pdf",
            title="India TB Report 2024",
            publisher="Central TB Division, Ministry of Health and Family Welfare",
            source_type_final="national_public_health_agency",
            source_role_final="collection",
            target_fit_status="task_record_collection_candidate",
            triage_role="task_record_collection_candidate",
            fetch_purpose="data_extraction",
            context_types=["table", "metric_row"],
        )
    ]
    state = _flu_state(
        chunks,
        user_request="Collect tuberculosis annual metrics for India in 2023.",
        structured_task={
            "disease": "Tuberculosis",
            "location": "India",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "India",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "time_window": "2023-01-01 to 2023-12-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "Tuberculosis",
            "disease_standard_name": "Tuberculosis",
            "aliases": ["tuberculosis", "TB"],
            "pathogen_terms": ["mycobacterium tuberculosis"],
            "syndrome_terms": [],
        },
        source_coverage_audit={
            "coverage_status": "target_official_source_missing",
            "coverage_completeness_status": "no_target_coverage",
        },
        must_fetch_sources=[],
        source_registry=[
            {
                "source_id": "src_india_tb_report",
                "canonical_url": "https://tbcindia.mohfw.gov.in/reports/tb-india-2024.pdf",
                "title": "India TB Report 2024",
                "publisher": "Central TB Division, Ministry of Health and Family Welfare",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "target_fit_status": "task_record_collection_candidate",
                "triage_role": "task_record_collection_candidate",
                "disease_fit": "match",
                "geography_fit": "match",
                "date_fit": "match",
                "credibility_level": "high",
            }
        ],
    )

    result = structured_extraction(state)
    summary = result["structured_extraction_summary"]

    assert calls == ["src_india_tb_report"]
    assert summary["no_task_collection_document"] is False
    assert summary["metric_row_batch_record_count"] == 1
    assert result["raw_records"][0]["metric_category"] == "incidence_rate"


def test_direct_collection_text_fallback_extracts_core_annual_burden_metrics(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=2)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return LLMExtractionOutput(records=[], chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunk = _chunk(
        "src_cdc_tb_annual",
        1,
        chunk_id="chunk_cdc_tb_burden",
        chunk_kind="text",
        text=(
            "In 2024, the United States reported 10,388 tuberculosis cases, "
            "corresponding to an incidence rate of 3.1 cases per 100,000 persons."
        ),
        source_url="https://www.cdc.gov/tb/surveillance/2024/index.html",
        title="Reported Tuberculosis in the United States, 2024",
        publisher="Centers for Disease Control and Prevention",
        source_type_final="national_public_health_agency",
        source_role_final="collection",
        target_fit_status="task_record_collection_candidate",
        triage_role="task_record_collection_candidate",
        fetch_purpose="data_extraction",
        data_types=["case_count", "incidence_rate"],
        context_types=["narrative_metric"],
        country="United States of America",
        geographic_scope="United States",
        geographic_scope_type="country",
        reporting_period_start="2024-01-01",
        reporting_period_end="2024-12-31",
        reporting_period_label="2024",
    )
    state = _flu_state(
        [chunk],
        user_request="Collect tuberculosis annual burden metrics for the United States in 2024.",
        structured_task={
            "disease": "Tuberculosis",
            "location": "United States",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "United States",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "time_window": "2024-01-01 to 2024-12-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "Tuberculosis",
            "disease_standard_name": "Tuberculosis",
            "aliases": ["tuberculosis", "TB"],
            "pathogen_terms": ["mycobacterium tuberculosis"],
            "syndrome_terms": [],
        },
        task_evidence_contract={
            "time_granularity": "annual",
            "requirements": [
                {
                    "requirement_id": "united_states_tuberculosis_annual_2024",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                    "geography": "United States",
                    "disease": "Tuberculosis",
                }
            ],
        },
        source_coverage_audit={
            "coverage_status": "partial_target_coverage",
            "coverage_completeness_status": "partial_target_coverage",
        },
        must_fetch_sources=[],
        source_registry=[
            {
                "source_id": "src_cdc_tb_annual",
                "canonical_url": "https://www.cdc.gov/tb/surveillance/2024/index.html",
                "title": "Reported Tuberculosis in the United States, 2024",
                "publisher": "Centers for Disease Control and Prevention",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "collection",
                "target_fit_status": "task_record_collection_candidate",
                "triage_role": "task_record_collection_candidate",
                "disease_fit": "match",
                "geography_fit": "match",
                "date_fit": "match",
                "credibility_level": "high",
                "reporting_period_start": "2024-01-01",
                "reporting_period_end": "2024-12-31",
                "reporting_period_label": "2024",
            }
        ],
    )

    result = structured_extraction(state)
    summary = result["structured_extraction_summary"]
    records = result["raw_records"]
    categories = {record.get("metric_category") for record in records}

    assert calls == ["src_cdc_tb_annual"]
    assert summary["llm_empty_output_count"] == 1
    assert summary["rule_based_fallback_record_count"] >= 2
    assert "case_count" in categories
    assert "incidence_rate" in categories
    incidence = next(
        record for record in records if record.get("metric_category") == "incidence_rate"
    )
    assert incidence["metric_value"] == 3.1
    assert incidence["metric_unit"] == "per 100,000 population"
    assert incidence["metric_period_start"] == "2024-01-01"
    assert incidence["metric_period_end"] == "2024-12-31"


def test_direct_collection_reports_core_metric_gap_when_task_source_yields_no_records(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return LLMExtractionOutput(records=[], chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunk = _chunk(
        "src_vdh_tb_annual",
        1,
        chunk_id="chunk_vdh_tb_gap",
        chunk_kind="text",
        text=(
            "Virginia tuberculosis annual surveillance report for 2025. "
            "The report discusses TB case counts, incidence, mortality, and "
            "treatment outcomes for Virginia residents."
        ),
        source_url="https://www.vdh.virginia.gov/tuberculosis/data-reports",
        title="Virginia Tuberculosis Annual Surveillance Report 2025",
        publisher="Virginia Department of Health",
        source_type_final="state_or_local_public_health_agency",
        source_role_final="collection",
        target_fit_status="task_record_collection_candidate",
        triage_role="task_record_collection_candidate",
        disease_fit="match",
        geography_fit="match",
        date_fit="match",
        fetch_purpose="data_extraction",
        data_types=["case_count", "incidence_rate", "death_count"],
        context_types=["narrative_metric"],
        country="United States of America",
        subnational_location="Virginia",
        geographic_scope="Virginia",
        geographic_scope_type="subnational",
        reporting_period_start="2025-01-01",
        reporting_period_end="2025-12-31",
        reporting_period_label="2025",
    )
    state = _flu_state(
        [chunk],
        user_request="Collect tuberculosis annual burden metrics for Virginia in 2025.",
        structured_task={
            "disease": "Tuberculosis",
            "location": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "Tuberculosis",
            "geography": "Virginia",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "time_window": "2025-01-01 to 2025-12-31",
            "collection_mode": "direct_collection",
        },
        disease_intelligence={
            "disease_input": "Tuberculosis",
            "disease_standard_name": "Tuberculosis",
            "aliases": ["tuberculosis", "TB"],
            "pathogen_terms": ["mycobacterium tuberculosis"],
            "syndrome_terms": [],
        },
        task_evidence_contract={
            "time_granularity": "annual",
            "requirements": [
                {
                    "requirement_id": "virginia_tuberculosis_annual_2025",
                    "period_start": "2025-01-01",
                    "period_end": "2025-12-31",
                    "geography": "Virginia",
                    "disease": "Tuberculosis",
                }
            ],
        },
        source_registry=[
            {
                "source_id": "src_vdh_tb_annual",
                "canonical_url": "https://www.vdh.virginia.gov/tuberculosis/data-reports",
                "title": "Virginia Tuberculosis Annual Surveillance Report 2025",
                "publisher": "Virginia Department of Health",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "target_fit_status": "task_record_collection_candidate",
                "triage_role": "task_record_collection_candidate",
                "disease_fit": "match",
                "geography_fit": "match",
                "date_fit": "match",
                "credibility_level": "high",
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31",
                "reporting_period_label": "2025",
            }
        ],
    )

    result = structured_extraction(state)
    summary = result["structured_extraction_summary"]
    plan = result["metric_extraction_plan"]

    assert calls == ["src_vdh_tb_annual"]
    assert result["raw_records"] == []
    assert summary["core_metric_extraction_gap_count"] == 1
    assert summary["core_metric_extraction_gaps"][0]["source_id"] == "src_vdh_tb_annual"
    assert plan["core_metric_extraction_gap_count"] == 1
    assert plan["task_source_extraction_attempted_count"] == 1
    assert plan["context_extraction_skipped_count"] == 0
    assert plan["target_text_fallback_attempted_count"] == 1
    assert plan["best_available_extraction_count"] == 0


def test_direct_collection_prioritizes_target_official_chunks_before_context_official_cap(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_cdc_2025_week_04",
            1,
            source_url="https://www.cdc.gov/fluview/surveillance/2025-week-04.html",
            title="CDC FluView week 04, 2025",
            publisher="CDC",
        ),
        _chunk(
            "src_ny_week_40",
            1,
            source_url=(
                "https://www.health.ny.gov/diseases/communicable/influenza/"
                "surveillance/2024-2025/archive/2024-10-05_flu_report.pdf"
            ),
            title="New York State Influenza Surveillance Report",
            publisher="New York State Department of Health",
        ),
    ]
    state = _flu_state(
        chunks,
        structured_task={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-10-01",
            "end_date": "2024-10-02",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "New York",
            "start_date": "2024-10-01",
            "end_date": "2024-10-02",
            "time_window": "2024-10-01 to 2024-10-02",
            "collection_mode": "direct_collection",
        },
        must_fetch_sources=[],
        source_registry=[
            {
                "source_id": "src_cdc_2025_week_04",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2025-week-04.html",
                "title": "CDC FluView week 04, 2025",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "source_type_final": "national_public_health_agency",
                "source_role_final": "context",
                "credibility_level": "high",
            },
            {
                "source_id": "src_ny_week_40",
                "canonical_url": (
                    "https://www.health.ny.gov/diseases/communicable/influenza/"
                    "surveillance/2024-2025/archive/2024-10-05_flu_report.pdf"
                ),
                "title": "New York State Influenza Surveillance Report",
                "publisher": "New York State Department of Health",
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "credibility_level": "high",
            },
        ],
    )

    result = structured_extraction(state)

    assert calls == ["src_ny_week_40"]
    budget = result["structured_extraction_summary"]["extraction_budget_by_source"]
    assert budget["src_ny_week_40"]["attempted_count"] == 1
    assert budget["src_cdc_2025_week_04"]["skipped_due_to_cap_count"] == 1
    assert budget["src_ny_week_40"]["budget_bucket"] == "verified_target_collection"
    assert budget["src_cdc_2025_week_04"]["budget_bucket"] == "official_or_high_trust"
    assert budget["src_cdc_2025_week_04"]["target_fit_status"] == "non_target_or_context"


def test_direct_collection_must_fetch_chunk_does_not_bypass_disease_gate(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)
    calls: list[str] = []

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        calls.append(str(chunk.get("source_id")))
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunk = _chunk(
        "src_vdh_week_40",
        1,
        disease_relevance_status="incompatible_disease",
        extraction_eligible_for_task_disease=False,
    )

    result = structured_extraction(_flu_state([chunk]))

    assert calls == []
    assert result["raw_records"] == []
    summary = result["structured_extraction_summary"]
    assert summary["must_fetch_disease_gate_bypass_count"] == 0
    assert summary["skipped_disease_mismatch_chunk_count"] == 1


def test_structured_extraction_summary_preserves_adaptive_budget_diagnostics(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=1)

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)

    result = structured_extraction(_flu_state([_chunk("src_vdh_week_40", 1)]))
    summary = result["structured_extraction_summary"]
    llm_summary = result["llm_extraction_summary"]

    for diagnostics in (summary, llm_summary):
        assert diagnostics["primary_llm_call_count"] == 1
        assert diagnostics["metric_primary_call_limit"] >= 0
        assert diagnostics[
            "metric_row_batch_deferred_for_case_coverage_count"
        ] >= 0
        assert diagnostics["extraction_budget_ledger"]["primary_call_count"] == 1
        assert diagnostics["llm_transport_success_count"] == 1
        assert diagnostics["llm_non_empty_output_count"] == 1
        assert diagnostics["valid_record_count"] >= 1
        assert diagnostics["unique_valid_record_count"] >= 1
        assert diagnostics["focused_recovery_valid_record_count"] == 0
        assert diagnostics["focused_recovery_field_gain_count"] == 0


def test_direct_collection_document_quality_keeps_mismatch_as_context_only():
    text = (
        "COVID-19 surveillance reported 42 cases and 7 hospitalizations. "
        "This mixed document should remain parse-usable so the LLM team can "
        "decide task relevance later. "
    ) * 5
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_mixed",
                "source_id": "src_mixed",
                "url": "https://example.org/mixed",
                "title": "Mixed respiratory surveillance update",
                "publisher": "Example Department of Health",
                "source_type": "official_public_health_agency",
                "source_role_final": "collection",
                "fetch_status": "fetched",
                "parse_status": "parsed_html",
                "clean_text": text,
                "tables": [],
                "is_offline_stub": False,
                "is_fixture_document": False,
            }
        ],
    )

    result = document_quality_check(state)
    doc = result["documents"][0]

    assert doc["quality_status"] == "not_task_relevant"
    assert doc["not_extractable_for_task_disease"] is True
    assert "disease_mismatch_not_task_relevant" in doc["quality_issues"]
    assert "disease_mismatch_audit_only" not in doc["quality_issues"]


def test_direct_collection_chunk_relevance_blocks_non_target_disease_data():
    text = (
        "COVID-19 surveillance reported 42 cases, 7 hospitalizations, "
        "and 2 deaths for the week ending October 5, 2024. "
        "The same public-health report may contain other respiratory tables. "
    ) * 4
    state = _flu_state(
        [],
        documents=[
            {
                "document_id": "doc_mixed",
                "source_id": "src_mixed",
                "url": "https://example.org/mixed",
                "title": "Mixed respiratory surveillance update",
                "publisher": "Example Department of Health",
                "source_type": "official_public_health_agency",
                "source_role_final": "collection",
                "fetch_purpose": "data_extraction",
                "fetch_status": "fetched",
                "parse_status": "parsed_html",
                "quality_status": "usable",
                "not_extractable_for_task_disease": False,
                "clean_text": text,
                "tables": [],
                "is_offline_stub": False,
            }
        ],
    )

    result = evidence_chunking_and_data_presence_flagging(state)

    assert result["evidence_chunks"]
    assert result["evidence_chunks"][0]["contains_target_data"] is False
    assert result["evidence_chunks"][0]["extraction_eligible_for_task_disease"] is False
    assert result["chunk_relevance_assessments"][0]["decision_owner"] == (
        "llm_chunk_relevance_agent"
    )
    assert result["evidence_chunking_summary"]["disease_mismatch_chunk_count"] >= 1


def test_llm_extraction_preserves_public_health_surveillance_count_fields(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=1)
    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", _fake_output)

    result = structured_extraction(_flu_state([_chunk("src_vdh_week_40", 1)]))

    record = result["raw_records"][0]
    assert record["tests_positive"] == 42.0
    assert record["hospitalizations"] == 7.0


def test_direct_collection_resolves_must_fetch_failure_by_equivalent_official_alias(
    monkeypatch,
):
    _enable_direct_llm(monkeypatch, max_chunks=2)

    def fake_extract(chunk: dict, policy) -> LLMExtractionOutput:
        if chunk.get("source_id") == "src_ny_short_alias":
            return LLMExtractionOutput(records=[], chunk_is_relevant=True)
        return _fake_output(chunk, policy)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_extract)
    chunks = [
        _chunk(
            "src_ny_short_alias",
            1,
            source_url=(
                "https://www.health.ny.gov/diseases/communicable/influenza/"
                "surveillance/2024-25/archive/2024-11-02_flu_report.pdf"
            ),
            title="New York State Influenza Surveillance Report",
            publisher="New York State Department of Health",
            text=(
                "New York State Influenza Surveillance Report archive landing "
                "page for the 2024-2025 season."
            ),
        ),
        _chunk(
            "src_ny_long_alias",
            1,
            source_url=(
                "https://www.health.ny.gov/diseases/communicable/influenza/"
                "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf"
            ),
            title="New York State Influenza Surveillance Report",
            publisher="New York State Department of Health",
        ),
    ]
    state = _flu_state(
        chunks,
        structured_task={
            "disease": "FLU",
            "location": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "collection_mode": "direct_collection",
        },
        collection_spec={
            "disease": "FLU",
            "geography": "New York",
            "start_date": "2024-11-01",
            "end_date": "2024-11-03",
            "time_window": "2024-11-01 to 2024-11-03",
            "collection_mode": "direct_collection",
        },
        must_fetch_sources=[
            {"source_id": "src_ny_short_alias", "must_fetch": True},
            {"source_id": "src_ny_long_alias", "must_fetch": True},
        ],
        source_registry=[
            {
                "source_id": "src_ny_short_alias",
                "canonical_url": chunks[0]["source_url"],
                "title": chunks[0]["title"],
                "publisher": chunks[0]["publisher"],
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "must_fetch": True,
                "credibility_level": "high",
            },
            {
                "source_id": "src_ny_long_alias",
                "canonical_url": chunks[1]["source_url"],
                "title": chunks[1]["title"],
                "publisher": chunks[1]["publisher"],
                "source_type": "official_public_health_agency",
                "source_type_final": "state_or_local_public_health_agency",
                "source_role_final": "collection",
                "must_fetch": True,
                "credibility_level": "high",
            },
        ],
    )

    result = structured_extraction(state)
    summary = result["structured_extraction_summary"]

    assert result["raw_records"]
    assert summary["official_extraction_failures"] == []
    assert summary["official_extraction_resolved_by_equivalent_source_count"] == 1
    short_budget = summary["extraction_budget_by_source"]["src_ny_short_alias"]
    long_budget = summary["extraction_budget_by_source"]["src_ny_long_alias"]
    assert short_budget["official_report_key"] == long_budget["official_report_key"]


def test_must_fetch_extraction_failure_includes_attempted_chunk_preview(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=1)

    def empty_extract(_chunk: dict, _policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(records=[], chunk_is_relevant=True)

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", empty_extract)
    chunk = _chunk(
        "src_vdh_week_40",
        1,
        text="Positive influenza specimens: 42 (4.2%)",
        chunk_kind="text",
    )

    result = structured_extraction(_flu_state([chunk]))
    failures = result["structured_extraction_summary"]["official_extraction_failures"]

    assert failures
    assert failures[0]["sample_chunks"][0]["chunk_id"] == "chunk_src_vdh_week_40_1"
    assert failures[0]["sample_chunks"][0]["chunk_kind"] == "text"
    assert "Positive influenza specimens" in failures[0]["sample_chunks"][0]["text_preview"]


def test_llm_outbreak_count_is_not_written_as_unspecified_cases(monkeypatch):
    _enable_direct_llm(monkeypatch, max_chunks=1)

    def fake_outbreak_output(chunk: dict, _policy) -> LLMExtractionOutput:
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Seasonal influenza",
                    virus_or_syndrome="influenza",
                    country="United States of America",
                    subnational_location="New York",
                    date_reported="2024-11-02",
                    reporting_period="week ending November 2, 2024",
                    cases_unspecified=13.0,
                    count_semantics=(
                        "Season-to-date total outbreaks reported from hospitals "
                        "and nursing homes."
                    ),
                    aggregation_level="subnational",
                    geographic_scope="New York",
                    geographic_scope_type="subnational",
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(llm_clients, "extract_chunk_with_llm", fake_outbreak_output)
    chunk = _chunk(
        "src_ny_week_44",
        1,
        text=(
            "There were 13 outbreaks reported from hospitals and nursing "
            "homes season to date in New York State."
        ),
        source_url=(
            "https://www.health.ny.gov/diseases/communicable/influenza/"
            "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf"
        ),
        title="New York State Influenza Surveillance Report",
        publisher="New York State Department of Health",
    )

    result = structured_extraction(_flu_state([chunk]))

    record = result["raw_records"][0]
    assert record["cases_unspecified"] is None
    assert record["cumulative_count"] == 13.0
    assert record["observation_type"] == "outbreak_summary"
    assert record["primary_case_dataset_eligible"] is False
    assert "outbreak_count_not_case_count" in record["semantic_warnings"]
