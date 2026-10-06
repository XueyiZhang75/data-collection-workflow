"""Reported dates use source roles and precision; cumulative periods keep source wording."""
import pytest
from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.nodes.extraction import _build_record_from_llm_output, _cumulative_reporting_period_label
from test_acquisition_evidence import assess, evidence


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
@pytest.mark.parametrize("disease,country", [("measles", "Canada"), ("pertussis", "France")])
@pytest.mark.parametrize("date_text", ["16 January 2025", "2025-01-16", "Thursday 16 January 2025"])
def test_report_date_english_source_conversion(field, disease, country, date_text):
    text = f"{country} reported 4 confirmed {disease} cases on {date_text}."
    q = assess(text, disease=disease, country=country, cases_confirmed=4, **{field: "2025-01-16"})
    assert q.status == "qualified", q.reasons
    assert all(text[f.locator["char_start"]:f.locator["char_end"]] == f.quote for f in q.field_evidence)


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
@pytest.mark.parametrize("text", [
    "France a signal\u00e9 4 cas confirm\u00e9s de rougeole le 16 janvier 2025.",
    "Le jeudi 16 janvier 2025, France a notifi\u00e9 4 cas confirm\u00e9s de rougeole.",
])
def test_report_date_french_source_conversion(field, text):
    q = assess(text, disease="rougeole", country="France", cases_confirmed=4, **{field: "2025-01-16"})
    assert q.status == "qualified", q.reasons


@pytest.mark.parametrize("field", ["date_reported", "report_date"])
@pytest.mark.parametrize("text", [
    "Published on 16 January 2025. Canada reported 4 confirmed measles cases during 2024.",
    "Publication date: 2025-01-16. Canada reported 4 confirmed measles cases during 2024.",
    "Canada reported 4 confirmed measles cases with symptom onset on 16 January 2025.",
    "Canada reported 4 confirmed measles cases with a death on 16 January 2025.",
    "Canada did not report 4 confirmed measles cases on 16 January 2025.",
    "Canada may report 4 confirmed measles cases on 16 January 2025.",
    "As of 16 January 2025, Canada had 4 confirmed measles cases.",
])
def test_other_date_roles_and_nonaffirmative_reporting_do_not_support_report_date(field, text):
    q = assess(text, disease="measles", country="Canada", cases_confirmed=4, **{field: "2025-01-16"})
    assert q.status == "candidate"
    assert not next(f.supported for f in q.field_evidence if f.field == field)


@pytest.mark.parametrize("text", [
    "Publi\u00e9 le 16 janvier 2025. France a signal\u00e9 4 cas confirm\u00e9s de rougeole en 2024.",
    "France n'a pas signal\u00e9 4 cas confirm\u00e9s de rougeole le 16 janvier 2025.",
])
def test_french_publication_or_negative_date_is_not_report_date(text):
    q = assess(text, disease="rougeole", country="France", cases_confirmed=4, date_reported="2025-01-16")
    assert not next(f.supported for f in q.field_evidence if f.field == "date_reported")


@pytest.mark.parametrize("period", ["January 10 to date", "depuis le 10 janvier", "2025"])
def test_evidence_period_fallback_does_not_invent_season(monkeypatch, period):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    assert _cumulative_reporting_period_label({"reporting_period": period}, {}, {}) == period


def test_explicit_season_column_and_legacy_fallback_are_preserved(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    assert _cumulative_reporting_period_label({"reporting_period": "Week 40", "metric_column_label": "Season-to-date"}, {}, {}) == "Season-to-date through Week 40"
    monkeypatch.setenv("PIPELINE_MODE", "legacy")
    assert _cumulative_reporting_period_label({"reporting_period": "Week 40"}, {}, {}) == "Season-to-date through Week 40"


@pytest.mark.parametrize("disease,country", [("measles", "Canada"), ("pertussis", "France")])
def test_actual_llm_builder_keeps_narrative_cumulative_period(monkeypatch, disease, country):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    text = f"From January 10 to date, {country} reported 17 cumulative confirmed {disease} cases."
    _, _, chunk = evidence(text)
    chunk.update(source_url="https://surveillance.example/report", source_type="unknown", confidence=0.9)
    response = LLMExtractedRecord(disease=disease, country=country, reporting_period="January 10 to date",
                                 statistical_count_type="cumulative", cases_confirmed=17, metric_name="cases_confirmed", metric_value=17, evidence_quote=text)
    result = _build_record_from_llm_output(response, chunk, 1,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {"disease_standard_name": disease, "is_hantavirus": False})
    assert result.reporting_period == "January 10 to date"
    assert "Season" not in (result.metric_period_label or "")


def test_existing_iso_typed_table_report_date_is_preserved():
    from data_collection_workflow.evidence_qualification import _supports
    assert _supports("date_reported", "2025-01-16", "Date reported | Country\n2025-01-16 | Canada", {})


@pytest.mark.parametrize("disease,country", [("measles", "Canada"), ("pertussis", "France")])
def test_reported_date_survives_actual_builder_schema_and_normalization(monkeypatch, disease, country):
    from data_collection_workflow.nodes.extraction import schema_validation_and_repair
    from data_collection_workflow.nodes.normalization import record_normalization
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    text = f"{country} reported 4 confirmed {disease} cases on 16 January 2025."
    _, document, chunk = evidence(text)
    chunk.update(source_url="https://surveillance.example/report", source_type="unknown", confidence=0.9)
    output = LLMExtractedRecord(disease=disease, country=country, date_reported="2025-01-16", cases_confirmed=4, evidence_quote=text)
    built = _build_record_from_llm_output(output, chunk, 1,
        LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()), {},
        {"disease_standard_name": disease, "is_hantavirus": False})
    state = {"structured_task": {"disease": disease, "location": country},
             "collection_spec": {"disease": disease, "geography": country},
             "raw_records": [built.model_dump(mode="json")], "documents": [document], "evidence_chunks": [chunk],
             "source_registry": [{"source_id": "source", "url": chunk["source_url"], "source_type": "unknown"}],
             "collection_trace": [], "human_review_queue": []}
    state.update(schema_validation_and_repair(state))
    state.update(record_normalization(state))
    assert len(state["normalized_records"]) == 1
    row = state["normalized_records"][0]
    assert row["date_reported"] == "2025-01-16"
    result = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    assert result.status == "qualified", result.reasons
