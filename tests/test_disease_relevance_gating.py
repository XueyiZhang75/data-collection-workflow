from __future__ import annotations

import json
from pathlib import Path

from data_collection_workflow.disease_relevance import (
    assess_record_disease_compatibility,
    assess_source_disease_relevance,
    build_disease_relevance_context,
)
from data_collection_workflow.nodes.content_processing import (
    document_quality_check,
    evidence_chunking_and_data_presence_flagging,
)
from data_collection_workflow.nodes.extraction import (
    schema_validation_and_repair,
    structured_extraction,
)
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.nodes.normalization import record_normalization


def _hantavirus_state(**overrides) -> dict:
    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Shanghai",
            "start_date": "2024-01-01",
            "end_date": "2026-06-09",
        },
        "collection_spec": {
            "disease": "hantavirus",
            "geography": "Shanghai",
            "time_window": "2024-01-01 to 2026-06-09",
            "target_population": "human",
        },
        "disease_intelligence": {
            "disease_input": "hantavirus",
            "disease_standard_name": "Hantavirus disease",
            "aliases": ["hantavirus", "hantavirus disease"],
            "abbreviations": ["HPS", "HFRS"],
            "pathogen_terms": ["hantavirus", "orthohantavirus"],
            "syndrome_terms": [
                "hantavirus pulmonary syndrome",
                "hemorrhagic fever with renal syndrome",
            ],
        },
        "collection_trace": [],
        "human_review_queue": [],
    }
    state.update(overrides)
    return state


def _flu_state(**overrides) -> dict:
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "Virginia",
            "time_window": "2024-10-06 to 2024-10-12",
            "target_population": "human",
        },
        "disease_intelligence": {
            "disease_input": "FLU",
            "disease_standard_name": "Seasonal influenza",
            "aliases": ["flu", "influenza", "seasonal influenza"],
            "abbreviations": ["ILI"],
            "pathogen_terms": ["influenza"],
            "syndrome_terms": ["influenza-like illness"],
        },
        "collection_trace": [],
        "human_review_queue": [],
    }
    state.update(overrides)
    return state


def _covid_source() -> dict:
    return {
        "source_id": "src_yahoo_covid_shanghai",
        "canonical_url": "https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
        "url": "https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
        "title": "EXPLAINER-Shanghai death numbers raise questions over its COVID accounting",
        "publisher": "Yahoo Finance",
        "source_type": "news_and_situation_report",
        "snippet": "Shanghai reported COVID-19 deaths and confirmed infections.",
        "query_used": '"HFRS Shanghai" cases deaths public health shanghai 2024',
        "status": "candidate",
    }


def _covid_chunk(**overrides) -> dict:
    chunk = {
        "chunk_id": "chunk_src_yahoo_covid_shanghai_001",
        "source_id": "src_yahoo_covid_shanghai",
        "text": (
            "Shanghai had reported no COVID-19 deaths for more than a month. "
            "The city has now reported 285 COVID-related fatalities from "
            "around 500,000 confirmed cases."
        ),
        "contains_target_data": True,
        "data_types": ["case_count", "death_count", "location"],
        "context_types": [],
        "confidence": 0.95,
        "document_type": "html",
        "fetch_purpose": "data_extraction",
        "source_url": "https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
        "canonical_url": "https://finance.yahoo.com/news/explainer-shanghai-death-numbers-raise-063847555.html",
        "title": "EXPLAINER-Shanghai death numbers raise questions over its COVID accounting",
        "publisher": "Yahoo Finance",
        "source_type": "news_and_situation_report",
        "source_role": "data_source",
        "source_role_final": "collection",
        "quality_status": "usable",
        "chunk_index": 1,
        "chunk_kind": "text",
    }
    chunk.update(overrides)
    return chunk


def test_source_relevance_ignores_query_used_as_disease_proof():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_source_disease_relevance(_covid_source(), context)

    assert assessment["status"] == "unrelated_disease"
    assert "COVID-19" in assessment["incompatible_disease_terms_found"]
    assert "query_used" not in " ".join(assessment["evidence_fields_used"])


def test_document_quality_marks_unrelated_covid_document_not_task_relevant():
    result = document_quality_check(
        _hantavirus_state(
            documents=[
                {
                    "source_id": "src_yahoo_covid_shanghai",
                    "document_type": "html",
                    "clean_text": _covid_chunk()["text"] * 5,
                    "tables": [],
                    "metadata": {},
                    "parse_status": "parsed_html",
                    "quality_status": None,
                    "quality_issues": [],
                    "url": _covid_source()["url"],
                    "canonical_url": _covid_source()["canonical_url"],
                    "title": _covid_source()["title"],
                    "publisher": "Yahoo Finance",
                    "source_type": "news_and_situation_report",
                    "source_role": "data_source",
                    "fetch_purpose": "data_extraction",
                    "is_live_fetched": True,
                    "is_offline_stub": False,
                }
            ]
        )
    )

    doc = result["documents"][0]
    assert doc["quality_status"] == "not_task_relevant"
    assert doc["not_extractable_for_task_disease"] is True
    assert doc["document_disease_relevance_status"] == "unrelated_disease"
    assert result["document_quality_summary"]["disease_relevance_status_counts"][
        "unrelated_disease"
    ] == 1


def test_chunking_suppresses_unrelated_covid_data_for_hantavirus_task():
    result = evidence_chunking_and_data_presence_flagging(
        _hantavirus_state(
            documents=[
                {
                    "source_id": "src_yahoo_covid_shanghai",
                    "document_type": "html",
                    "clean_text": _covid_chunk()["text"] * 4,
                    "tables": [],
                    "metadata": {},
                    "parse_status": "parsed_html",
                    "quality_status": "usable",
                    "quality_issues": [],
                    "url": _covid_source()["url"],
                    "canonical_url": _covid_source()["canonical_url"],
                    "title": _covid_source()["title"],
                    "publisher": "Yahoo Finance",
                    "source_type": "news_and_situation_report",
                    "source_role": "data_source",
                    "fetch_purpose": "data_extraction",
                    "is_live_fetched": True,
                    "is_offline_stub": False,
                }
            ]
        )
    )

    chunks = result["evidence_chunks"]
    assert chunks
    assert all(chunk["contains_target_data"] is False for chunk in chunks)
    assert all(
        chunk["disease_relevance_status"] == "unrelated_disease"
        for chunk in chunks
    )
    assert result["data_presence_summary"]["target_data_chunk_count"] == 0
    assert result["data_presence_summary"]["disease_mismatch_chunk_count"] >= 1


def test_structured_extraction_skips_disease_mismatch_chunk_even_if_marked_target():
    result = structured_extraction(
        _hantavirus_state(evidence_chunks=[_covid_chunk()])
    )

    assert result["raw_records"] == []
    summary = result["structured_extraction_summary"]
    assert summary["raw_record_count"] == 0
    assert summary["skipped_disease_mismatch_chunk_count"] == 1
    assert "src_yahoo_covid_shanghai" in summary["skipped_disease_mismatch_source_ids"]


def test_schema_validation_rejects_llm_record_from_incompatible_covid_evidence():
    raw_record = {
        "record_id": "rec_bad_covid_as_hantavirus",
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "virus_or_syndrome": "SARS-CoV-2",
        "pathogen_or_syndrome": "SARS-CoV-2",
        "country": "China",
        "subnational_location": "Shanghai",
        "date_reported": "2022-04-28",
        "cases_confirmed": 500000,
        "deaths": 285,
        "source_id": "src_yahoo_covid_shanghai",
        "source_url": _covid_source()["url"],
        "source_type": "news_and_situation_report",
        "evidence_quote": _covid_chunk()["text"],
        "supporting_chunk_id": "chunk_src_yahoo_covid_shanghai_001",
        "extraction_confidence": 0.9,
        "extraction_method": "llm_structured_output_extractor",
    }

    result = schema_validation_and_repair(
        _hantavirus_state(raw_records=[raw_record])
    )

    assert result["validated_records"] == []
    assert len(result["rejected_records"]) == 1
    rejected = result["rejected_records"][0]
    assert rejected["schema_status"] == "rejected"
    assert rejected["record_disease_compatibility_status"] == "incompatible_disease"
    assert "disease_mismatch" in rejected["validation_errors"]


def test_normalization_quarantines_incompatible_validated_record():
    validated_record = {
        "record_id": "rec_bad_covid_validated",
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "virus_or_syndrome": "SARS-CoV-2",
        "pathogen_or_syndrome": "SARS-CoV-2",
        "country": "China",
        "subnational_location": "Shanghai",
        "date_reported": "2022-04-28",
        "cases_confirmed": 500000,
        "deaths": 285,
        "source_id": "src_yahoo_covid_shanghai",
        "source_url": _covid_source()["url"],
        "source_type": "news_and_situation_report",
        "evidence_quote": _covid_chunk()["text"],
        "supporting_chunk_id": "chunk_src_yahoo_covid_shanghai_001",
        "schema_status": "valid",
        "provenance_status": "verified",
        "extraction_confidence": 0.9,
    }

    result = record_normalization(
        _hantavirus_state(validated_records=[validated_record])
    )

    assert result["normalized_records"] == []
    summary = result["record_normalization_summary"]
    assert summary["disease_mismatch_quarantined_record_count"] == 1
    assert len(result["disease_mismatch_records"]) == 1


def test_record_compatibility_accepts_target_hantavirus_evidence():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "virus_or_syndrome": "HFRS",
            "evidence_quote": (
                "Shanghai reported one HFRS case caused by hantavirus in 2025."
            ),
        },
        context,
    )

    assert assessment["status"] == "compatible"
    assert assessment["target_disease_terms_found"]


def test_record_compatibility_accepts_influenza_row_from_mixed_respiratory_report():
    context = build_disease_relevance_context(_flu_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Influenza",
            "disease_standard_name": "Seasonal influenza",
            "virus_or_syndrome": "influenza",
            "metric_name": "Influenza positive lab reports",
            "metric_category": "lab_positive_count",
            "count_semantics": "weekly influenza laboratory positive specimens",
            "evidence_quote": (
                "Influenza positive lab reports: 119. The same respiratory "
                "disease surveillance report also tracks COVID-19 and RSV."
            ),
        },
        context,
    )

    assert assessment["status"] == "compatible"
    assert assessment["reject_record"] is False
    assert "influenza" in {
        str(value).lower() for value in assessment["target_disease_terms_found"]
    }
    assert "COVID-19" in assessment["incompatible_disease_terms_found"]
    assert assessment["record_level_target_identity_override"] is True


def test_record_compatibility_rejects_non_target_count_in_mixed_outbreak_page():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 93,
            "deaths": 58,
            "source_title": (
                "WHO Director-General's opening remarks at the media briefing "
                "on outbreaks of Ebola and hantavirus"
            ),
            "evidence_quote": (
                "Since 2014, up to 7 May 2026, 93 confirmed human cases of "
                "avian influenza A(H5N6), including 58 deaths, have been "
                "reported in the WHO Western Pacific Region."
            ),
        },
        context,
    )

    assert assessment["reject_record"] is True
    assert assessment["status"] == "incompatible_disease"
    assert assessment["record_local_disease_relevance_status"] in {
        "unrelated_disease",
        "incompatible_disease",
    }
    assert "influenza" in {
        str(value).lower()
        for value in assessment["record_local_incompatible_terms_found"]
    }


def test_record_compatibility_rejects_non_target_numeric_sentence_even_when_quote_mentions_target_later():
    context = build_disease_relevance_context(
        _hantavirus_state(
            disease_intelligence={
                "disease_input": "hantavirus",
                "disease_standard_name": "Hantavirus disease",
                "aliases": ["hantavirus", "hantavirus disease"],
                "abbreviations": ["HPS", "HFRS"],
                "pathogen_terms": ["hantavirus", "orthohantavirus", "Andes virus"],
                "syndrome_terms": ["hantavirus pulmonary syndrome"],
                "case_count_terms": ["cases", "confirmed cases", "human cases"],
                "death_terms": ["deaths", "fatal cases"],
            }
        )
    )
    assert "cases" not in {
        str(value).lower() for value in context["target_disease_identity_terms"]
    }
    assert "human cases" not in {
        str(value).lower() for value in context["target_disease_identity_terms"]
    }
    assert "deaths" not in {
        str(value).lower() for value in context["target_disease_identity_terms"]
    }
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 93,
            "deaths": 58,
            "source_title": "WHO outbreaks under monitoring",
            "evidence_quote": (
                "Since 2014, up to 7 May 2026, 93 confirmed human cases of "
                "avian influenza A(H5N6), including 58 deaths, have been "
                "reported in the WHO Western Pacific Region. Separately, "
                "public health authorities continue monitoring the MV Hondius "
                "hantavirus outbreak."
            ),
        },
        context,
    )

    assert assessment["reject_record"] is True
    assert assessment["status"] == "incompatible_disease"
    assert assessment["record_local_disease_relevance_status"] == "incompatible_disease"
    assert assessment["record_local_numeric_disease_status"] == "incompatible_disease"
    assert assessment["record_local_numeric_target_terms_found"] == []
    assert {
        str(value).lower()
        for value in assessment["record_local_numeric_incompatible_terms_found"]
    } & {"influenza", "avian influenza", "h5n6", "influenza a"}
    assert "local_evidence_disease_mismatch" in assessment["reason"]


def test_record_compatibility_rejects_ebola_numeric_sentence_despite_later_target_mention():
    context = build_disease_relevance_context(
        _hantavirus_state(
            disease_intelligence={
                "disease_input": "hantavirus",
                "disease_standard_name": "Hantavirus disease",
                "aliases": ["hantavirus", "hantavirus disease"],
                "abbreviations": ["HPS", "HFRS"],
                "pathogen_terms": ["hantavirus", "Andes virus"],
                "case_count_terms": ["cases", "human cases"],
                "death_terms": ["deaths"],
            }
        )
    )

    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 600,
            "source_title": "WHO media briefing on Ebola and hantavirus",
            "evidence_quote": (
                "Ebola vaccine could take nine months as death toll rises further, "
                "WHO warns. There have now been 139 suspected deaths and 600 "
                "cases. The same briefing also discussed hantavirus."
            ),
        },
        context,
    )

    assert assessment["reject_record"] is True
    assert assessment["status"] == "incompatible_disease"
    assert assessment["record_local_numeric_disease_status"] == "incompatible_disease"
    assert assessment["record_local_numeric_target_terms_found"] == []
    assert "Ebola" in assessment["record_local_numeric_incompatible_terms_found"]
    assert "local_evidence_disease_mismatch" in assessment["reason"]


def test_record_compatibility_rejects_non_target_numeric_sentence_without_sentence_spacing():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 93,
            "deaths": 58,
            "source_title": "GOV.UK outbreaks under monitoring",
            "evidence_quote": (
                "Since 2014, up to 7 May 2026, 93 confirmed human cases of "
                "avian influenza A(H5N6), including 58 deaths, have been "
                "reported in the WHO Western Pacific Region.She was hospitalised "
                "on 23 April 2026 and died on 3 May 2026.The same page later "
                "mentions the MV Hondius hantavirus outbreak."
            ),
        },
        context,
    )

    assert assessment["reject_record"] is True
    assert assessment["status"] == "incompatible_disease"
    assert assessment["record_local_disease_relevance_status"] == "incompatible_disease"
    assert "local_evidence_disease_mismatch" in assessment["reason"]


def test_record_compatibility_rejects_generic_count_sentence_without_local_target_identity():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 448,
            "deaths": 3,
            "source_title": "Global health update",
            "evidence_quote": (
                "Cases 95% have been unvaccinated or have unknown vaccination "
                "statuses and 1% have been hospitalized. Since the previous "
                "update, 448 confirmed incident cases and 3 deaths were "
                "reported, including cases for the first time in Madagascar."
            ),
        },
        context,
    )

    assert assessment["reject_record"] is True
    assert assessment["status"] == "incompatible_disease"
    assert assessment["record_local_numeric_disease_status"] == "incompatible_disease"
    assert assessment["record_local_numeric_target_terms_found"] == []
    assert "local_evidence_disease_mismatch" in assessment["reason"]


def test_record_local_numeric_gate_does_not_use_page_title_as_disease_evidence():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 448,
            "deaths": 3,
            "source_title": "Hantavirus outbreak update",
            "evidence_quote": (
                "Since the previous update, 448 confirmed cases and 3 deaths "
                "were reported."
            ),
        },
        context,
    )

    assert assessment["status"] == "incompatible_disease"
    assert assessment["reject_record"] is True
    assert assessment["record_local_numeric_disease_status"] == (
        "incompatible_disease"
    )
    assert assessment["record_local_numeric_target_terms_found"] == []
    assert "source_title" not in assessment["record_local_evidence_fields_used"]


def test_record_local_numeric_gate_accepts_target_disease_in_local_table_header():
    context = build_disease_relevance_context(_hantavirus_state())
    assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "disease_standard_name": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 448,
            "deaths": 3,
            "source_title": "Weekly surveillance report",
            "table_header": "Hantavirus cases | Deaths",
            "evidence_quote": "Confirmed cases | 448 | Deaths | 3",
        },
        context,
    )

    assert assessment["status"] == "compatible"
    assert assessment["reject_record"] is False
    assert assessment["record_local_numeric_disease_status"] == (
        "target_disease_match"
    )
    assert "table_header" in assessment["record_local_evidence_fields_used"]


def test_target_table_heading_cannot_override_explicit_incompatible_disease_row():
    from data_collection_workflow.nodes.content_processing import (
        _assess_metric_row_local_disease_relevance,
    )

    context = build_disease_relevance_context(_hantavirus_state())
    row = {
        "text": "Avian influenza A(H5N6) confirmed human cases | 93",
        "row_quote": "Avian influenza A(H5N6) confirmed human cases | 93",
        "heading_context": "Hantavirus surveillance",
        "table_header": "Disease | Metric | Count",
        "row_context_type": "structured_table",
    }

    chunk_assessment = _assess_metric_row_local_disease_relevance(row, context)
    record_assessment = assess_record_disease_compatibility(
        {
            "disease": "Hantavirus disease",
            "metric_category": "case_count",
            "cases_unspecified": 93,
            "heading_context": row["heading_context"],
            "table_header": row["table_header"],
            "evidence_quote": row["row_quote"],
            "row_context_type": row["row_context_type"],
        },
        context,
    )

    assert chunk_assessment["status"] in {
        "unrelated_disease",
        "incompatible_disease",
    }
    assert record_assessment["reject_record"] is True
    assert record_assessment["record_local_numeric_disease_status"] == (
        "incompatible_disease"
    )
    assert "local_evidence_disease_mismatch" in record_assessment["reason"]


def test_llm_extraction_policy_requires_task_disease_match():
    policy = json.loads(
        Path("src/data_collection_workflow/resources/llm_structured_extraction_policy.json").read_text(
            encoding="utf-8"
        )
    )
    text = policy["system_prompt"] + " ".join(policy["required_output_rules"])

    assert "target task disease" in text
    assert "incompatible disease" in text


def test_final_package_includes_disease_relevance_summary():
    result = final_data_package_builder(
        _hantavirus_state(
            normalized_records=[],
            source_registry=[],
            linked_events=[],
            event_clusters=[],
            duplicate_clusters=[],
            validation_cases=[],
            validation_comparisons=[],
            validation_results=[],
            anomaly_results=[],
            conflicts=[],
            disease_relevance_summary={
                "target_disease": "Hantavirus disease",
                "rejected_incompatible_record_count": 1,
            },
        )
    )

    package = result["final_data_package"]
    summaries = package["workflow_summaries"]
    assert summaries["disease_relevance_summary"]["target_disease"] == (
        "Hantavirus disease"
    )


def test_final_package_quarantines_local_non_target_numeric_claim_from_reviewable_candidates():
    record = {
        "record_id": "rec_govuk_h5n6_mixed_page",
        "disease": "Hantavirus disease",
        "disease_standard_name": "Hantavirus disease",
        "canonical_disease": "Hantavirus disease",
        "metric_category": "case_count",
        "observation_type": "unspecified_case_record",
        "primary_case_dataset_eligible": True,
        "country": "Global",
        "date_reported": "2026-05-10",
        "cases_unspecified": 93,
        "deaths": 58,
        "source_id": "src_govuk_mixed_outbreaks",
        "source_url": (
            "https://www.gov.uk/government/publications/"
            "outbreaks-under-monitoring-in-2026/"
            "outbreaks-under-monitoring-week-19-week-ending-10-may-2026"
        ),
        "source_title": "Outbreaks under monitoring: week 19",
        "source_type": "official_public_health_agency",
        "source_type_final": "national_public_health_agency",
        "publisher": "GOV.UK",
        "evidence_quote": (
            "Since 2014, up to 7 May 2026, 93 confirmed human cases of "
            "avian influenza A(H5N6), including 58 deaths, have been "
            "reported in the WHO Western Pacific Region."
        ),
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "record_schema": "official_outbreak_page_extraction_schema",
        "extraction_confidence": 0.9,
    }
    source = {
        "source_id": "src_govuk_mixed_outbreaks",
        "canonical_url": record["source_url"],
        "title": record["source_title"],
        "publisher": "GOV.UK",
        "source_type_final": "national_public_health_agency",
        "authority_bucket": "official_authority",
        "source_role_final": "collection",
        "status": "ready_for_content_fetch",
        "data_product_type": "event_outbreak_report",
        "task_specificity": "event_specific",
    }

    result = final_data_package_builder(
        _hantavirus_state(
            structured_task={
                "disease": "hantavirus",
                "location": "Global",
                "start_date": "2026-04-01",
                "end_date": "2026-06-30",
            },
            collection_spec={
                "disease": "hantavirus",
                "geography": "Global",
                "time_window": "2026-04-01 to 2026-06-30",
                "target_population": "human",
            },
            normalized_records=[record],
            source_registry=[source],
            documents=[],
            evidence_chunks=[],
            validation_results=[],
            anomaly_results=[],
        )
    )

    package = result["final_data_package"]
    reviewable_ids = {
        row.get("record_id") for row in package["reviewable_case_dataset"]
    }
    quarantined = {
        row.get("record_id"): row for row in package["quarantined_records"]
    }

    assert "rec_govuk_h5n6_mixed_page" not in reviewable_ids
    assert "rec_govuk_h5n6_mixed_page" in quarantined
    assert quarantined["rec_govuk_h5n6_mixed_page"][
        "record_local_disease_relevance_status"
    ] == "incompatible_disease"
    assert "local_evidence_disease_mismatch" in "; ".join(
        quarantined["rec_govuk_h5n6_mixed_page"].get("quality_gate_reasons") or []
    )
