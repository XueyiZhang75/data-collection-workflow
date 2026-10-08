from __future__ import annotations

import json
import sys
from pathlib import Path


_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "official_outbreak_chunks"


def _chunk(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _context() -> dict:
    return {
        "collection_mode": "direct_collection",
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2025-01-01",
            "end_date": "2026-05-08",
        },
        "collection_spec": {
            "disease": "Hantavirus disease",
            "geography": "Global",
            "time_window": "2025-01-01 to 2026-05-08",
            "target_population": "human",
        },
        "disease_intelligence": {
            "disease_standard_name": "Hantavirus disease",
            "aliases": ["hantavirus", "hantavirus pulmonary syndrome"],
            "abbreviations": ["HPS"],
            "pathogen_terms": ["Andes virus", "ANDV", "orthohantavirus"],
            "syndrome_terms": ["hantavirus pulmonary syndrome"],
        },
        "source_registry_by_id": {},
        "must_fetch_source_ids": set(),
    }


def _extract(chunks: list[dict]):
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import (
        extract_official_outbreak_records_from_chunks,
    )

    policy = StructuredExtractionPolicy(**load_structured_extraction_policy())
    return extract_official_outbreak_records_from_chunks(
        chunks,
        policy=policy,
        context=_context(),
    )


def test_who_don_like_vessel_outbreak_extracts_reviewable_primary_candidate():
    records, diagnostics = _extract([_chunk("who_don_mv_hondius_chunk.json")])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["cases_unspecified"] == 8
    assert record["deaths"] == 3
    assert record["primary_case_dataset_eligible"] is True
    assert record["requires_human_review"] is True
    assert record["human_review_reason"] == "official_single_source_not_cross_validated"
    assert record["record_schema"] == "official_outbreak_page_extraction_schema"
    assert record["location_type"] == "vessel_associated"
    assert record["locality"] == "MV Hondius"
    assert record["record_geography_fit_status"] == (
        "vessel_or_travel_associated_within_global_scope"
    )
    assert record["disease_alias_match"] is True
    assert record["matched_alias"] in {"Andes virus", "ANDV", "hantavirus"}
    assert record["canonical_disease"] == "Hantavirus disease"
    assert record["case_count_text_span"]
    assert record["death_count_text_span"]
    assert record["date_anchor_type"] in {"as_of_date", "report_date"}
    assert record["source_url"]
    assert record["evidence_quote"]
    assert diagnostics[0]["extraction_attempted"] is True
    assert diagnostics[0]["primary_candidate_count"] == 1


def test_cdc_han_like_official_alert_extracts_case_death_candidate():
    chunk = {
        "source_id": "fixture_cdc_han",
        "chunk_id": "chunk_fixture_cdc_han_001",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.cdc.gov/han/php/notices/fixture.html",
        "title": "2026 Multi-country Hantavirus Cluster Linked to Cruise Ship | HAN | CDC",
        "publisher": "Centers for Disease Control and Prevention",
        "actual_publisher": "Centers for Disease Control and Prevention",
        "source_type": "official_public_health_agency",
        "source_type_final": "national_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "CDC HAN Health Advisory. As of May 8, 2026, CDC is aware of "
            "8 hantavirus pulmonary syndrome cases, including 3 deaths, "
            "associated with cruise ship travel. Andes virus infection was confirmed."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["cases_unspecified"] == 8
    assert record["deaths"] == 3
    assert record["source_url"].startswith("https://www.cdc.gov/han/")
    assert record["publisher"] == "Centers for Disease Control and Prevention"
    assert record["requires_human_review"] is True
    assert diagnostics[0]["failure_substage"] is None


def test_official_case_sentence_extracts_line_list_detail_fields():
    chunk = {
        "source_id": "fixture_who_don599",
        "chunk_id": "chunk_fixture_who_don599_001",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        "title": "Disease Outbreak News; Hantavirus disease - multi-country",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Case 4: An adult female, with presentation of pneumonia, died on "
            "2 May 2026. The onset of symptoms was on 28 April, with fever "
            "and a general feeling of being unwell. Andes virus infection was confirmed."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["workflow_case_label"] == "Case 4"
    assert record["gender"] == "female"
    assert record["outcome"] == "death"
    assert record["date_onset"] == "2026-04-28"
    assert record["date_death"] == "2026-05-02"
    assert "pneumonia" in record["symptoms"]
    assert "fever" in record["symptoms"]
    assert diagnostics[0]["primary_candidate_count"] == 1


def test_adjacent_case_spans_do_not_transfer_fields_between_cases():
    chunk = {
        "source_id": "fixture_who_don599",
        "chunk_id": "chunk_fixture_who_don599_adjacent_cases",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        "title": "Disease Outbreak News; Hantavirus disease - multi-country",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Case 3: An adult male presented to the ship's doctor on "
            "24 April 2026 with febrile illness, shortness of breath and "
            "signs of pneumonia. Case 4: An adult female, with onset of "
            "symptoms on 28 April 2026 and presentation of pneumonia, died "
            "on 2 May 2026. Andes virus infection was confirmed."
        ),
    }

    records, diagnostics = _extract([chunk])

    rows = {record.model_dump()["workflow_case_label"]: record.model_dump() for record in records}
    assert set(rows) == {"Case 3", "Case 4"}
    assert rows["Case 3"]["gender"] == "male"
    assert rows["Case 3"]["outcome"] is None
    assert rows["Case 3"]["date_death"] is None
    assert rows["Case 4"]["gender"] == "female"
    assert rows["Case 4"]["outcome"] == "death"
    assert rows["Case 4"]["date_onset"] == "2026-04-28"
    assert rows["Case 4"]["date_death"] == "2026-05-02"
    assert diagnostics[0]["primary_candidate_count"] == 2


def test_case_span_extracts_developed_symptoms_and_reported_onset_dates():
    chunk = {
        "source_id": "fixture_who_don599",
        "chunk_id": "chunk_fixture_who_don599_onset_variants",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        "title": "Disease Outbreak News; Hantavirus disease - multi-country",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Case 1 developed symptoms on 6 April 2026 and died on "
            "11 April 2026. Case 5 reported onset of symptoms on 30 April "
            "2026 and later recovered. Andes virus infection was confirmed."
        ),
    }

    records, diagnostics = _extract([chunk])

    rows = {record.model_dump()["workflow_case_label"]: record.model_dump() for record in records}
    assert rows["Case 1"]["date_onset"] == "2026-04-06"
    assert rows["Case 1"]["date_death"] == "2026-04-11"
    assert rows["Case 1"]["outcome"] == "death"
    assert rows["Case 5"]["date_onset"] == "2026-04-30"
    assert rows["Case 5"]["outcome"] == "recovered"
    assert rows["Case 5"]["date_death"] is None
    assert diagnostics[0]["primary_candidate_count"] == 2


def test_case_span_extracts_confirmation_role_care_and_contact_fields():
    chunk = {
        "source_id": "fixture_who_don599",
        "chunk_id": "chunk_fixture_who_don599_rich_line_list_fields",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        "title": "Disease Outbreak News; Hantavirus disease - multi-country",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Case 6: A male passenger developed symptoms on 7 May 2026, "
            "was hospitalized and isolated, and was confirmed by PCR on "
            "10 May 2026. He was a close contact of Case 4 aboard the "
            "cruise ship MV Hondius. Andes virus infection was confirmed."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["workflow_case_label"] == "Case 6"
    assert record["gender"] == "male"
    assert record["cruise_passenger_guest"] is True
    assert record["cruise_crew"] is False
    assert record["hospitalized"] is True
    assert record["isolated"] is True
    assert record["contact_with_case"] == "Case 4"
    assert record["contact_setting"] == "aboard cruise ship"
    assert record["confirmation_method"] == "PCR"
    assert record["date_confirmation"] == "2026-05-10"
    assert diagnostics[0]["primary_candidate_count"] == 1


def test_deterministic_case_span_fields_include_field_level_provenance():
    quote = (
        "Case 4: An 80-year-old German female developed fever on 28 April "
        "2026 and died on 2 May 2026. Andes virus infection was confirmed."
    )
    chunk = {
        "source_id": "fixture_who_case_4",
        "chunk_id": "chunk_fixture_who_case_4",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/example-outbreak",
        "title": "Andes virus outbreak update",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": quote,
    }

    records, _diagnostics = _extract([chunk])

    assert len(records) == 1
    row = records[0].model_dump()
    provenance = json.loads(row["field_provenance_json"])
    assert provenance["age"]["extracted_value"] == "80"
    assert provenance["gender"]["case_span_id"] == row["case_span_id"]
    assert provenance["date_death"]["supporting_quote"] == row["case_span_quote"]
    assert provenance["date_death"]["extraction_method"] == (
        "deterministic_case_label_span"
    )


def test_pdf_spaced_hyphens_preserve_numeric_age_nationality_role_and_dates():
    chunk = {
        "source_id": "fixture_who_rra_pdf",
        "chunk_id": "chunk_fixture_who_rra_pdf_case_5",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/example-rra.pdf",
        "title": "Rapid risk assessment - Andes hantavirus outbreak",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Case 5: A 41 -year -old Dutch male national, serving as the "
            "ship's doctor, reported onset of symptoms on 30 April 2026, "
            "including fever, fatigue, and myalgia. He was hospitalised in "
            "an intensive care unit and isolated. PCR testing confirmed "
            "Andes virus infection on 7 May 2026."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["workflow_case_label"] == "Case 5"
    assert record["age"] == "41"
    assert record["gender"] == "male"
    assert record["nationality"] == "Dutch"
    assert record["occupation_or_role"] == "ship doctor"
    assert record["date_onset"] == "2026-04-30"
    assert record["date_confirmation"] == "2026-05-07"
    assert record["confirmation_method"] == "PCR"
    assert record["hospitalized"] is True
    assert record["intensive_care"] is True
    assert record["isolated"] is True
    assert "fever" in record["symptoms"]
    assert "fatigue" in record["symptoms"]
    assert "myalgia" in record["symptoms"]
    assert diagnostics[0]["primary_candidate_count"] == 1


def test_confirmation_date_patterns_cover_official_report_word_order_variants():
    from data_collection_workflow.nodes.extraction import _official_line_list_details

    variants = [
        (
            "Case 5: His samples confirmed PCR positivity for Andes virus on 6 May 2026.",
            "PCR",
            "2026-05-06",
        ),
        (
            "Case 4: Hantavirus infection was confirmed on 8 May 2026.",
            "laboratory testing",
            "2026-05-08",
        ),
        (
            "Case 2: She died on 26 April 2026. On 4 May 2026, the case was "
            "subsequently confirmed by PCR with hantavirus infection.",
            "PCR",
            "2026-05-04",
        ),
    ]

    for text, method, date_confirmation in variants:
        details = _official_line_list_details(text)
        assert details["confirmation_method"] == method
        assert details["date_confirmation"] == date_confirmation


def test_llm_chunk_order_round_robins_high_value_event_sources(monkeypatch):
    from data_collection_workflow.nodes.extraction import _ordered_llm_chunks

    monkeypatch.setenv("LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE", "2")
    chunks = []
    for source_id, count in (("src_a", 5), ("src_b", 2), ("src_c", 1)):
        for index in range(count):
            chunks.append(
                {
                    "source_id": source_id,
                    "chunk_id": f"{source_id}_{index}",
                    "chunk_index": index,
                    "chunk_kind": "text",
                    "text": "Hantavirus MV Hondius case evidence.",
                    "contains_target_data": True,
                    "usable_for_task_collection": True,
                    "source_role_final": "collection",
                }
            )

    ordered = _ordered_llm_chunks(chunks, _context())

    assert [chunk["source_id"] for chunk in ordered[:4]] == [
        "src_a",
        "src_b",
        "src_c",
        "src_a",
    ]


def test_llm_chunk_order_gives_every_high_value_source_one_slot_before_context(
    monkeypatch,
):
    from data_collection_workflow.nodes.extraction import _ordered_llm_chunks

    monkeypatch.setenv("LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE", "3")
    monkeypatch.setenv("LLM_OFFICIAL_EXTRACTION_MAX_CHUNKS", "30")
    chunks = [
        {
            "source_id": f"event_{index:02d}",
            "chunk_id": f"event_{index:02d}_chunk",
            "chunk_index": 0,
            "chunk_kind": "text",
            "text": "Andes virus MV Hondius case evidence.",
            "contains_target_data": True,
            "usable_for_task_collection": True,
            "source_role_final": "collection",
        }
        for index in range(40)
    ]
    chunks.extend(
        {
            "source_id": f"context_{index:02d}",
            "chunk_id": f"context_{index:02d}_chunk",
            "chunk_index": 0,
            "chunk_kind": "text",
            "text": "General background page.",
            "contains_target_data": False,
            "source_role_final": "context",
        }
        for index in range(50)
    )

    ordered = _ordered_llm_chunks(chunks, _context())

    assert {chunk["source_id"] for chunk in ordered[:40]} == {
        f"event_{index:02d}" for index in range(40)
    }


def test_llm_chunk_order_prefers_explicit_case_span_over_navigation(monkeypatch):
    from data_collection_workflow.nodes.extraction import _ordered_llm_chunks

    monkeypatch.setenv("LLM_MUST_FETCH_MIN_CHUNKS_PER_SOURCE", "2")
    chunks = [
        {
            "source_id": "src_event",
            "chunk_id": "chunk_navigation",
            "chunk_index": 0,
            "chunk_kind": "text",
            "text": "Home About News Contact Publications",
            "contains_target_data": True,
            "usable_for_task_collection": True,
            "source_role_final": "collection",
        },
        {
            "source_id": "src_event",
            "chunk_id": "chunk_case_4",
            "chunk_index": 1,
            "chunk_kind": "text",
            "text": (
                "Case 4: an adult female developed fever on 28 April 2026 "
                "and died on 2 May 2026 from Andes virus infection."
            ),
            "contains_target_data": True,
            "usable_for_task_collection": True,
            "source_role_final": "collection",
        },
    ]

    ordered = _ordered_llm_chunks(chunks, _context())

    assert ordered[0]["chunk_id"] == "chunk_case_4"


def test_multi_case_chunk_is_split_before_llm_extraction():
    from data_collection_workflow.nodes.extraction import _expand_case_span_chunks_for_llm

    chunk = {
        "source_id": "src_who_multi_case",
        "chunk_id": "chunk_who_multi_case",
        "chunk_kind": "text",
        "text": (
            "Case 3: an adult male developed fever on 24 April 2026. "
            "Case 4: an adult female died on 2 May 2026. Andes virus infection "
            "was confirmed."
        ),
        "source_url": "https://www.who.int/multi-case",
        "title": "Andes virus outbreak update",
        "contains_target_data": True,
    }

    expanded = _expand_case_span_chunks_for_llm([chunk])

    assert len(expanded) == 2
    assert [row["case_span_id"] for row in expanded] == [
        "case_span_001_case_3",
        "case_span_002_case_4",
    ]
    assert "Case 4" not in expanded[0]["text"]
    assert "Case 3" not in expanded[1]["text"]
    assert expanded[0]["parent_chunk_id"] == "chunk_who_multi_case"
    assert expanded[1]["parent_chunk_id"] == "chunk_who_multi_case"


def test_llm_case_only_record_is_retained_with_span_provenance():
    from data_collection_workflow.config import load_llm_structured_extraction_policy
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output

    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    quote = (
        "Case 4: an adult female developed fever on 28 April 2026 and died "
        "on 2 May 2026 from Andes virus infection."
    )
    chunk = {
        "source_id": "src_who_case_4",
        "chunk_id": "chunk_who_case_4",
        "chunk_kind": "text",
        "text": quote,
        "source_url": "https://www.who.int/example-case-4",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "source_role_final": "collection",
        "title": "Andes virus outbreak update",
        "publisher": "World Health Organization",
        "contains_target_data": True,
    }
    llm_record = LLMExtractedRecord(
        disease="Hantavirus disease",
        virus_or_syndrome="Andes virus",
        workflow_case_label="Case 4",
        age="adult",
        gender="female",
        symptoms="fever",
        date_onset="2026-04-28",
        date_death="2026-05-02",
        outcome="death",
        case_span_id="case_span_001_case_4",
        case_span_quote=quote,
        case_span_start=0,
        case_span_end=len(quote),
        case_span_extraction_method="llm_case_span",
    )

    record = _build_record_from_llm_output(
        llm_record,
        chunk,
        1,
        policy,
        {"provider": "anthropic", "model": "test-model"},
        _context(),
    )

    assert record is not None
    row = record.model_dump()
    assert row["workflow_case_label"] == "Case 4"
    assert row["case_span_id"] == "case_span_001_case_4"
    assert row["case_span_quote"] == quote
    provenance = json.loads(row["field_provenance_json"])
    assert provenance["gender"]["case_span_id"] == "case_span_001_case_4"
    assert provenance["gender"]["supporting_quote"] == quote


def test_llm_case_fields_are_rebound_to_the_explicit_local_span():
    from data_collection_workflow.config import load_llm_structured_extraction_policy
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output

    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    quote = (
        "Case 4: an adult female developed fever on 28 April 2026 and died "
        "on 2 May 2026 from Andes virus infection."
    )
    chunk = {
        "source_id": "src_who_case_4",
        "chunk_id": "chunk_who_case_4",
        "chunk_kind": "text",
        "text": quote,
        "case_span_id": "case_span_001_case_4",
        "case_span_quote": quote,
        "case_span_start": 0,
        "case_span_end": len(quote),
        "case_span_extraction_method": "deterministic_case_label_span",
        "source_url": "https://www.who.int/example-case-4",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "source_role_final": "collection",
        "title": "Andes virus outbreak update",
        "publisher": "World Health Organization",
        "contains_target_data": True,
    }
    llm_record = LLMExtractedRecord(
        disease="Hantavirus disease",
        virus_or_syndrome="Andes virus",
        workflow_case_label="Case 3",
        gender="male",
        case_span_id="case_span_001_case_4",
        case_span_quote=quote,
    )

    record = _build_record_from_llm_output(
        llm_record,
        chunk,
        1,
        policy,
        {"provider": "anthropic", "model": "test-model"},
        _context(),
    )

    assert record is not None
    row = record.model_dump()
    assert row["workflow_case_label"] == "Case 4"
    assert row["gender"] == "female"
    assert set(row["unsupported_case_fields"]) == {
        "gender",
        "workflow_case_label",
    }


def test_llm_case_span_recovers_explicit_fields_omitted_by_the_model():
    from data_collection_workflow.config import load_llm_structured_extraction_policy
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output

    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    quote = (
        "Case 5: A 41-year-old Dutch male national serving as the ship's "
        "doctor developed fever and fatigue on 30 April 2026. PCR confirmed "
        "Andes virus infection on 7 May 2026."
    )
    chunk = {
        "source_id": "src_case_report",
        "chunk_id": "chunk_case_report_case_5",
        "chunk_kind": "text",
        "text": quote,
        "case_span_id": "case_span_001_case_5",
        "case_span_quote": quote,
        "case_span_start": 0,
        "case_span_end": len(quote),
        "case_span_extraction_method": "deterministic_case_label_span",
        "source_url": "https://example.org/case-report",
        "source_type": "peer_reviewed_literature",
        "source_type_final": "peer_reviewed_literature",
        "source_role_final": "collection",
        "title": "Andes virus case report",
        "publisher": "Example Journal",
        "contains_target_data": True,
    }
    llm_record = LLMExtractedRecord(
        disease="Hantavirus disease",
        virus_or_syndrome="Andes virus",
        workflow_case_label="Case 5",
        case_span_id="case_span_001_case_5",
        case_span_quote=quote,
    )

    record = _build_record_from_llm_output(
        llm_record,
        chunk,
        1,
        policy,
        {"provider": "anthropic", "model": "test-model"},
        _context(),
    )

    assert record is not None
    row = record.model_dump()
    assert row["age"] == "41"
    assert row["gender"] == "male"
    assert row["nationality"] == "Dutch"
    assert row["occupation_or_role"] == "ship doctor"
    assert row["date_onset"] == "2026-04-30"
    assert row["date_confirmation"] == "2026-05-07"
    assert row["confirmation_method"] == "PCR"
    assert "fever" in row["symptoms"]
    assert "fatigue" in row["symptoms"]
    provenance = json.loads(row["field_provenance_json"])
    assert provenance["age"]["extraction_method"] == (
        "deterministic_case_span_field_recovery"
    )


def test_strong_case_signal_empty_output_uses_separate_focused_recovery_budget(
    monkeypatch,
):
    from data_collection_workflow.config import (
        load_llm_structured_extraction_policy,
        load_structured_extraction_policy,
    )
    from data_collection_workflow.models import (
        LLMExtractedRecord,
        LLMExtractionOutput,
        LLMStructuredExtractionPolicy,
        StructuredExtractionPolicy,
    )
    from data_collection_workflow.nodes import extraction

    monkeypatch.setenv("LLM_MAX_CHUNKS", "1")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_MAX_CALLS", "1")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_MAX_SOURCES", "1")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_SPANS_PER_SOURCE", "1")
    monkeypatch.setattr(
        extraction.llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    calls: list[dict] = []

    def _extract(chunk, _policy):
        calls.append(chunk)
        if len(calls) == 1:
            return LLMExtractionOutput(
                records=[],
                chunk_is_relevant=True,
                extraction_notes="No record returned in broad pass.",
            )
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    virus_or_syndrome="Andes virus",
                    workflow_case_label="Patient 2",
                    age="64",
                    gender="male",
                    cruise_passenger_guest=True,
                    date_confirmation="2026-05-05",
                    confirmation_method="PCR",
                    case_span_id="focused_span_001",
                    case_span_quote=chunk["text"],
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(extraction.llm_clients, "extract_chunk_with_llm", _extract)
    monkeypatch.setattr(
        extraction,
        "_rule_based_extract_records_from_chunks",
        lambda *_args, **_kwargs: ([], {}),
    )
    llm_policy = LLMStructuredExtractionPolicy(
        **load_llm_structured_extraction_policy()
    )
    deterministic_policy = StructuredExtractionPolicy(
        **load_structured_extraction_policy()
    )
    chunk = {
        "source_id": "src_scientific_case_report",
        "chunk_id": "chunk_scientific_patient_2",
        "chunk_kind": "text",
        "text": (
            "Patient 2 was a 64-year-old male passenger on MV Hondius. "
            "On 5 May 2026, PCR confirmed Andes virus infection."
        ),
        "source_url": "https://example.org/andv-case-report",
        "source_type": "peer_reviewed_literature",
        "source_type_final": "peer_reviewed_literature",
        "source_role_final": "collection",
        "fetch_purpose": "data_extraction",
        "title": "Andes virus case report",
        "publisher": "Example Journal",
        "contains_target_data": True,
        "usable_for_task_collection": True,
    }

    records, stats = extraction._llm_extract_records_from_chunks(
        [chunk],
        llm_policy,
        deterministic_policy,
        fallback_to_rule_based=False,
        context=_context(),
    )

    assert len(calls) == 2
    assert len(records) == 1
    assert records[0].workflow_case_label == "Patient 2"
    assert records[0].focused_recovery_status == "focused_retry_succeeded"
    assert stats["focused_recovery_call_count"] == 1
    assert stats["focused_retry_succeeded_count"] == 1
    assert stats["llm_empty_output_reasons"]["llm_empty_strong_signal"] == 1


def test_adaptive_budget_continues_past_soft_cap_for_uncovered_high_value_source(
    monkeypatch,
):
    from data_collection_workflow.config import (
        load_llm_structured_extraction_policy,
        load_structured_extraction_policy,
    )
    from data_collection_workflow.models import (
        LLMExtractedRecord,
        LLMExtractionOutput,
        LLMStructuredExtractionPolicy,
        StructuredExtractionPolicy,
    )
    from data_collection_workflow.nodes import extraction

    monkeypatch.setenv("LLM_MAX_CHUNKS", "1")
    monkeypatch.setenv("LLM_SOFT_PRIMARY_CALLS", "1")
    monkeypatch.setenv("LLM_HARD_PRIMARY_CALLS", "2")
    monkeypatch.setenv("LLM_HIGH_VALUE_SOURCE_MIN_SPANS", "1")
    monkeypatch.setattr(
        extraction.llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    monkeypatch.setattr(
        extraction,
        "extract_official_outbreak_records_from_chunks",
        lambda *_args, **_kwargs: ([], []),
    )
    calls: list[str] = []

    def _extract(chunk, _policy):
        calls.append(str(chunk.get("source_id")))
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    virus_or_syndrome="Andes virus",
                    cases=1,
                    evidence_quote=chunk["text"],
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(extraction.llm_clients, "extract_chunk_with_llm", _extract)
    policy = LLMStructuredExtractionPolicy(
        **load_llm_structured_extraction_policy()
    )
    deterministic = StructuredExtractionPolicy(
        **load_structured_extraction_policy()
    )
    chunks = [
        {
            "source_id": source_id,
            "chunk_id": f"{source_id}_chunk",
            "chunk_kind": "text",
            "text": (
                f"Andes virus was confirmed in patient {patient_number} "
                "on 5 May 2026."
            ),
            "source_url": f"https://{source_id}.example/report",
            "source_type": "official_public_health_agency",
            "source_type_final": "national_public_health_agency",
            "source_role_final": "collection",
            "fetch_purpose": "data_extraction",
            "contains_target_data": True,
            "usable_for_task_collection": True,
        }
        for patient_number, source_id in enumerate(
            ("authority_a", "authority_b"),
            start=1,
        )
    ]

    _, stats = extraction._llm_extract_records_from_chunks(
        chunks,
        policy,
        deterministic,
        fallback_to_rule_based=False,
        context=_context(),
    )

    assert calls == ["authority_a", "authority_b"]
    assert stats["primary_llm_call_count"] == 2
    assert stats["extraction_budget_ledger"]["soft_primary_calls"] == 1
    assert stats["extraction_budget_ledger"]["hard_primary_calls"] == 2
    assert stats["extraction_budget_ledger"]["soft_cap_extension_call_count"] == 1


def test_productive_source_empty_case_span_remains_eligible_for_recovery(monkeypatch):
    from data_collection_workflow.config import (
        load_llm_structured_extraction_policy,
        load_structured_extraction_policy,
    )
    from data_collection_workflow.models import (
        LLMExtractedRecord,
        LLMExtractionOutput,
        LLMStructuredExtractionPolicy,
        StructuredExtractionPolicy,
    )
    from data_collection_workflow.nodes import extraction

    monkeypatch.setenv("LLM_MAX_CHUNKS", "2")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_MAX_CALLS", "1")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_MAX_SOURCES", "1")
    monkeypatch.setenv("LLM_FOCUSED_RECOVERY_SPANS_PER_SOURCE", "2")
    monkeypatch.setattr(
        extraction.llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    monkeypatch.setattr(
        extraction,
        "extract_official_outbreak_records_from_chunks",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        extraction,
        "_rule_based_extract_records_from_chunks",
        lambda *_args, **_kwargs: ([], {}),
    )
    calls: list[str] = []

    def _extract(chunk, _policy):
        calls.append(str(chunk.get("chunk_id")))
        if len(calls) == 1:
            return LLMExtractionOutput(
                records=[
                    LLMExtractedRecord(
                        disease="Hantavirus disease",
                        workflow_case_label="Patient 1",
                        cases=1,
                        evidence_quote=chunk["text"],
                    )
                ],
                chunk_is_relevant=True,
            )
        if len(calls) == 2:
            return LLMExtractionOutput(records=[], chunk_is_relevant=True)
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    workflow_case_label="Patient 2",
                    cases=1,
                    evidence_quote=chunk["text"],
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(extraction.llm_clients, "extract_chunk_with_llm", _extract)
    policy = LLMStructuredExtractionPolicy(
        **load_llm_structured_extraction_policy()
    )
    deterministic = StructuredExtractionPolicy(
        **load_structured_extraction_policy()
    )
    chunks = [
        {
            "source_id": "case_series",
            "chunk_id": f"patient_{number}",
            "chunk_kind": "text",
            "text": (
                f"Patient {number} was confirmed with Andes virus by PCR "
                "on 5 May 2026."
            ),
            "source_url": "https://example.org/case-series",
            "source_type": "peer_reviewed_literature",
            "source_type_final": "peer_reviewed_literature",
            "source_role_final": "collection",
            "fetch_purpose": "data_extraction",
            "contains_target_data": True,
            "usable_for_task_collection": True,
        }
        for number in (1, 2)
    ]

    records, stats = extraction._llm_extract_records_from_chunks(
        chunks,
        policy,
        deterministic,
        fallback_to_rule_based=False,
        context=_context(),
    )

    assert len(calls) == 3
    assert {record.workflow_case_label for record in records} == {
        "Patient 1",
        "Patient 2",
    }
    assert stats["focused_recovery_call_count"] == 1


def test_case_span_budget_is_reserved_before_metric_row_batches(monkeypatch):
    from data_collection_workflow.config import (
        load_llm_structured_extraction_policy,
        load_structured_extraction_policy,
    )
    from data_collection_workflow.models import (
        LLMExtractedRecord,
        LLMExtractionOutput,
        LLMStructuredExtractionPolicy,
        StructuredExtractionPolicy,
    )
    from data_collection_workflow.nodes import extraction

    monkeypatch.setenv("LLM_MAX_CHUNKS", "1")
    monkeypatch.setattr(
        extraction.llm_clients,
        "get_llm_settings",
        lambda: {"provider": "anthropic", "model": "test-model"},
    )
    monkeypatch.setattr(
        extraction,
        "extract_official_outbreak_records_from_chunks",
        lambda *_args, **_kwargs: ([], []),
    )
    monkeypatch.setattr(
        extraction,
        "_deterministic_metric_row_records",
        lambda *_args, **_kwargs: ([], []),
    )
    calls: list[str] = []

    def _extract(chunk, _policy):
        calls.append(str(chunk.get("chunk_id")))
        return LLMExtractionOutput(
            records=[
                LLMExtractedRecord(
                    disease="Hantavirus disease",
                    workflow_case_label="Case 4",
                    cases=1,
                    evidence_quote=chunk["text"],
                )
            ],
            chunk_is_relevant=True,
        )

    monkeypatch.setattr(extraction.llm_clients, "extract_chunk_with_llm", _extract)
    policy = LLMStructuredExtractionPolicy(
        **load_llm_structured_extraction_policy()
    )
    deterministic = StructuredExtractionPolicy(
        **load_structured_extraction_policy()
    )
    chunks = [
        {
            "source_id": "weekly_table",
            "chunk_id": "metric_row_1",
            "chunk_kind": "metric_row",
            "row_id": "row_1",
            "row_quote": "Hantavirus tests | 12 | 1",
            "text": "Hantavirus tests | 12 | 1",
            "source_url": "https://example.gov/weekly",
            "source_type_final": "national_public_health_agency",
            "source_role_final": "collection",
            "fetch_purpose": "data_extraction",
            "contains_target_data": True,
            "extraction_eligible_for_task_disease": True,
        },
        {
            "source_id": "case_report",
            "chunk_id": "case_span_4",
            "chunk_kind": "text",
            "text": "Case 4 had Andes virus confirmed by PCR on 5 May 2026.",
            "source_url": "https://example.org/case-report",
            "source_type_final": "peer_reviewed_literature",
            "source_role_final": "collection",
            "fetch_purpose": "data_extraction",
            "contains_target_data": True,
            "extraction_eligible_for_task_disease": True,
        },
    ]

    _records, stats = extraction._llm_extract_records_from_chunks(
        chunks,
        policy,
        deterministic,
        fallback_to_rule_based=False,
        context=_context(),
    )

    assert len(calls) == 1
    assert calls[0].startswith("case_span_4__case_span_")
    assert stats["batched_metric_row_call_count"] == 0
    assert stats["metric_row_batch_deferred_for_case_coverage_count"] == 1


def test_official_grouped_case_sentence_preserves_group_label_and_vessel_context():
    chunk = {
        "source_id": "fixture_who_don599",
        "chunk_id": "chunk_fixture_who_don599_002",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON599",
        "title": "Disease Outbreak News; Hantavirus disease - multi-country",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Cases 1 and 2 had travelled in South America, including Argentina, "
            "before they boarded the cruise ship on 1 April 2026. Andes virus "
            "infection was confirmed."
        ),
    }

    records, _diagnostics = _extract([chunk])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["workflow_case_label"] == "Cases 1 and 2"
    assert record["cases_unspecified"] == 2
    assert "South America" in record["travel_from"]
    assert "Argentina" in record["travel_from"]
    assert "cruise ship" in record["travel_or_vessel_context"].lower()
    assert record["ship_board_date"] == "2026-04-01"


def test_paho_regional_alert_extracts_regional_primary_candidate():
    records, diagnostics = _extract([_chunk("paho_americas_alert_chunk.json")])

    assert len(records) == 1
    record = records[0].model_dump()
    assert record["cases_unspecified"] == 229
    assert record["deaths"] == 59
    assert record["geographic_scope"] == "Region of the Americas"
    assert record["geographic_scope_type"] == "region"
    assert record["primary_case_dataset_eligible"] is True
    assert record["requires_human_review"] is True
    assert diagnostics[0]["primary_candidate_count"] == 1


def test_official_outbreak_numeric_guardrails_do_not_count_context_numbers():
    chunk = {
        "source_id": "fixture_guardrail",
        "chunk_id": "chunk_fixture_guardrail_001",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.who.int/emergencies/disease-outbreak-news/item/fixture-guardrail",
        "title": "Hantavirus operational update",
        "publisher": "World Health Organization",
        "actual_publisher": "World Health Organization",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "Hantavirus operational update. The response involved 8 countries, "
            "14 ports, 90 crew, 120 passengers, 47 contacts, 25 specimens, "
            "12 tests, epidemiological week 47, 50% follow-up completion, "
            "and ages from 20 to 65 years. 3. Zero Preventable Maternal Deaths."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert records == []
    assert diagnostics[0]["extraction_attempted"] is True
    assert diagnostics[0]["failure_substage"] in {
        "ambiguous_numeric_context",
        "no_case_or_death_signal",
    }


def test_plain_hantavirus_surveillance_page_does_not_trigger_han_outbreak_extractor():
    chunk = {
        "source_id": "fixture_ecdc_surveillance",
        "chunk_id": "chunk_fixture_ecdc_surveillance_001",
        "chunk_kind": "text",
        "contains_target_data": True,
        "source_url": "https://www.ecdc.europa.eu/en/hantavirus-infection/surveillance-and-disease-data",
        "title": "Surveillance and updates for hantavirus",
        "publisher": "European Centre for Disease Prevention and Control",
        "actual_publisher": "European Centre for Disease Prevention and Control",
        "source_type": "official_public_health_agency",
        "source_type_final": "international_public_health_agency",
        "fetch_purpose": "collection",
        "text": (
            "For 2023, countries reported 1885 hantavirus infections. "
            "This surveillance page is not an outbreak alert."
        ),
    }

    records, diagnostics = _extract([chunk])

    assert records == []
    assert diagnostics == []


def test_partial_time_window_overlap_is_preserved_for_review_not_dropped():
    records, _diagnostics = _extract([_chunk("who_don_mv_hondius_chunk.json")])

    record = records[0].model_dump()
    assert record["event_start_date"] == "2025-12-29"
    assert record["event_end_date"] == "2026-05-10"
    assert record["period_overlap_status"] == "partial_overlap"
    assert record["requires_human_review"] is True
    assert "partial_time_window_overlap" in record["extraction_warnings"]
