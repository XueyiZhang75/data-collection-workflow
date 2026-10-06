"""Real-source failure shapes, varied across diseases and countries, without live calls."""
import hashlib
import io

import pytest

from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.nodes.extraction import (
    _build_record_from_llm_output,
    _official_best_count_mentions,
)
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.evidence_qualification import assess_record_evidence


@pytest.fixture(autouse=True)
def enable_evidence_pipeline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def evidence(text, **facts):
    digest = hashlib.sha256(text.encode()).hexdigest()
    document = {
        "source_id": "source", "document_id": "document", "clean_text": text,
        "content_hash": digest, "text_hash": digest, "parser_version": "test-native/1",
    }
    chunk = {
        "source_id": "source", "chunk_id": "span", "document_id": "document",
        "document_hash": digest, "text": text, "char_start": 0, "char_end": len(text),
    }
    row = {"record_id": "observation", "source_id": "source", "supporting_chunk_id": "span", **facts}
    return row, document, chunk


def assess(text, **facts):
    row, document, chunk = evidence(text, **facts)
    return assess_record_evidence(
        row, contract={}, evidence_index={"documents": [document], "evidence_chunks": {"span": chunk}}
    )


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("pertussis", "Canada")])
def test_manuscript_line_number_before_case_heading_is_not_a_count(disease, country):
    text = f"124 Case investigation from the earliest confirmed {disease} patient preceding the 2025 outbreak in {country}"
    mention, _, _ = _official_best_count_mentions(text)
    assert mention is None, "A manuscript line number is not an epidemiological quantity."


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("measles", "France")])
def test_line_number_candidate_cannot_pass_numeric_evidence_admission(disease, country):
    text = f"124 Case investigation from the earliest confirmed {disease} patient preceding the 2025 outbreak in {country}"
    qualification = assess(text, disease=disease, country=country, reporting_period="2025", cases_unspecified=124)
    count = next(field for field in qualification.field_evidence if field.field == "cases_unspecified")
    assert not count.supported


@pytest.mark.parametrize("separator", [" ", "\u00a0", "\u202f"])
@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("dengue", "Brazil")])
def test_grouped_integer_is_preserved_through_rule_and_evidence(separator, disease, country):
    text = f"{country} reported 18{separator}000 suspected {disease} cases during 2025."
    mention, _, _ = _official_best_count_mentions(text)
    assert mention is not None and mention["value"] == 18000
    qualification = assess(text, disease=disease, country=country, reporting_period="2025", cases_suspected=18000)
    assert qualification.status == "qualified", qualification.reasons


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("dengue", "Brazil")])
def test_lower_bound_grouped_count_is_never_an_exact_count(disease, country):
    text = f"{country} reported more than 18 000 suspected {disease} cases during 2025."
    for value in (18, 18000):
        qualification = assess(text, disease=disease, country=country, reporting_period="2025", cases_suspected=value)
        assert qualification.status == "candidate"
        assert any(reason.startswith("cases_suspected:") for reason in qualification.reasons)


@pytest.mark.parametrize("disease,target,owner", [("mpox", "Sierra Leone", "Nigeria"), ("measles", "France", "Germany")])
def test_comparison_with_another_country_does_not_transfer_its_count(disease, target, owner):
    text = (
        f"The {disease} epidemic in {target} in 2025 has been surpassed only by {owner}, "
        f"with 3,771 suspected {disease} cases reported in 2017 through 2024."
    )
    qualification = assess(text, disease=disease, country=target, reporting_period="2017 through 2024", cases_suspected=3771)
    assert qualification.status == "candidate"
    assert any(reason.startswith("cases_suspected:") for reason in qualification.reasons)


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("pertussis", "Canada")])
def test_explicit_3011_interval_supports_normalized_period_and_end_date(disease, country):
    text = f"Between January and 23 May 2025, {country} reported a total of 3,011 confirmed {disease} cases."
    qualification = assess(
        text, disease=disease, country=country, reporting_period="January to 23 May 2025",
        metric_period_end="2025-05-23", cases_confirmed=3011,
    )
    assert qualification.status == "qualified", qualification.reasons


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("pertussis", "Canada")])
def test_month_precision_does_not_invent_first_day_or_full_year(disease, country):
    text = f"Between January and 23 May 2025, {country} reported a total of 3,011 confirmed {disease} cases."
    for date in ("2025-01-01", "2025-12-31"):
        qualification = assess(
            text, disease=disease, country=country, reporting_period="January to 23 May 2025",
            metric_period_start=date, cases_confirmed=3011,
        )
        assert qualification.status == "candidate"
        assert any(reason.startswith("metric_period_start:") for reason in qualification.reasons)


def test_3011_without_country_scope_remains_unbound():
    text = "Between January and 23 May 2025, a total of 3,011 confirmed mpox cases have been reported nationwide."
    qualification = assess(text, disease="mpox", country="Sierra Leone", reporting_period="January to 23 May 2025", cases_confirmed=3011)
    assert qualification.status == "candidate"
    assert any(reason.startswith("country:") for reason in qualification.reasons)
    assert any(reason.startswith("cases_confirmed:") for reason in qualification.reasons)


def test_national_total_is_not_a_subnational_total():
    text = "Sierra Leone reported 3,011 confirmed mpox cases during 2025, with the majority originating from Freetown."
    qualification = assess(text, disease="mpox", country="Sierra Leone", subnational_location="Freetown", reporting_period="2025", cases_confirmed=3011)
    assert qualification.status == "candidate"
    assert any(reason.startswith("cases_confirmed:") for reason in qualification.reasons)


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("measles", "France")])
def test_pdf_normalized_quote_resolves_to_original_line_numbered_location(tmp_path, disease, country):
    from reportlab.pdfgen.canvas import Canvas

    output = io.BytesIO()
    canvas = Canvas(output)
    canvas.drawString(36, 760, f"97 {country} reported 3,011 confirmed {disease}")
    canvas.drawString(36, 740, "98 cases during 2025.")
    canvas.save()
    document = parse_response(output.getvalue(), url="https://offline.invalid/report.pdf", source_id="source", session_dir=tmp_path, content_type="application/pdf")
    raw_text = document["clean_text"]
    quote = f"{country} reported 3,011 confirmed {disease} cases during 2025."
    chunk = {"source_id": "source", "chunk_id": "span", "document_hash": document["content_hash"], "text": raw_text, "char_start": 0, "char_end": len(raw_text)}
    facts = {"disease": disease, "country": country, "reporting_period": "2025", "cases_confirmed": 3011}
    row = {"record_id": "r", "source_id": "source", "supporting_chunk_id": "span", **facts, "field_provenance_json": {field: {"quote": quote} for field in facts}}
    qualification = assess_record_evidence(row, contract={}, evidence_index={"documents": [document], "evidence_chunks": {"span": chunk}})
    assert qualification.status == "qualified", qualification.reasons
    for field in qualification.field_evidence:
        locator = field.locator
        assert raw_text[locator["char_start"]:locator["char_end"]] == field.quote
        assert field.document_hash == document["content_hash"]


@pytest.mark.parametrize("disease,country", [("mpox", "Sierra Leone"), ("measles", "France")])
def test_normalized_quote_cannot_substitute_a_different_country(disease, country):
    text = f"97 {country} reported 3,011 confirmed {disease}\n98 cases during 2025."
    row, document, chunk = evidence(text, disease=disease, country="Brazil", reporting_period="2025", cases_confirmed=3011)
    quote = f"Brazil reported 3,011 confirmed {disease} cases during 2025."
    row["field_provenance_json"] = {"country": {"quote": quote}, "cases_confirmed": {"quote": quote}}
    qualification = assess_record_evidence(row, contract={}, evidence_index={"documents": [document], "evidence_chunks": {"span": chunk}})
    assert qualification.status == "candidate"
    assert any(reason.startswith("country:") for reason in qualification.reasons)


@pytest.mark.parametrize("disease,country,unrelated", [("mpox", "Sierra Leone", "lymphadenopathy"), ("measles", "France", "pneumonia")])
def test_builder_does_not_fill_pathogen_from_unrelated_chunk_sentence(disease, country, unrelated):
    quote = f"{country} reported 3,011 confirmed {disease} cases during 2025."
    text = quote + f" A separate clinical subgroup had {unrelated}."
    _, _, chunk = evidence(text)
    chunk.update(source_url="https://offline.invalid/report", source_type="academic_or_peer_reviewed_source", confidence=0.9)
    output = LLMExtractedRecord(disease=disease, country=country, reporting_period="2025", cases_confirmed=3011, evidence_quote=quote)
    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    built = _build_record_from_llm_output(output, chunk, 1, policy, {}, {"disease_standard_name": disease, "is_hantavirus": False, "pathogen_terms": [unrelated]})
    assert built is not None
    assert built.pathogen_or_syndrome is None
    assert built.virus_or_syndrome is None


# French assertions use the same binding contract as English. These are synthetic
# statements, varied by disease and jurisdiction; none is a live factual claim.
_FRENCH_TASKS = [
    ("chikungunya", "La R\u00e9union", "France"),
    ("dengue", "Guyane", "France"),
    ("mpox", "Qu\u00e9bec", "Canada"),
]


@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS)
@pytest.mark.parametrize("label,field", [
    ("confirm\u00e9s", "cases_confirmed"),
    ("suspects", "cases_suspected"),
    ("probables", "cases_probable"),
])
def test_french_grouped_integer_and_postposed_case_qualifier_share_binding(disease, region, country, label, field):
    from data_collection_workflow.nodes.extraction import _classify_case_bucket

    text = f"En 2025, {region} ({country}) a notifi\u00e9 32\u202f456 cas {label} de {disease}."
    mention, _, _ = _official_best_count_mentions(text)
    assert mention is not None and mention["value"] == 32456
    assert _classify_case_bucket(mention["span"]) == field
    qualification = assess(text, disease=disease, country=country, subnational_location=region,
                           reporting_period="2025", **{field: 32456})
    assert qualification.status == "qualified", qualification.reasons


@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS)
def test_french_day_month_interval_binds_iso_dates_without_losing_original_locations(disease, region, country):
    text = (f"\u00c0 {region} ({country}), 32 456 cas confirm\u00e9s de {disease} ont \u00e9t\u00e9 notifi\u00e9s "
            "du 1er janvier au 23 mai 2025.")
    qualification = assess(text, disease=disease, country=country, subnational_location=region,
        reporting_period="du 1er janvier au 23 mai 2025", metric_period_start="2025-01-01",
        metric_period_end="2025-05-23", cases_confirmed=32456)
    assert qualification.status == "qualified", qualification.reasons
    for field in qualification.field_evidence:
        assert field.supported
        assert text[field.locator["char_start"]:field.locator["char_end"]] == field.quote


@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS)
def test_french_lower_bound_never_becomes_an_exact_confirmed_count(disease, region, country):
    text = f"En 2025, {region} ({country}) a notifi\u00e9 plus de 32 456 cas confirm\u00e9s de {disease}."
    for value in (32, 32456):
        qualification = assess(text, disease=disease, country=country, subnational_location=region,
                               reporting_period="2025", cases_confirmed=value)
        assert qualification.status == "candidate"
        count = next(field for field in qualification.field_evidence if field.field == "cases_confirmed")
        assert not count.supported


@pytest.mark.parametrize("disease,target,owner,country", [
    ("chikungunya", "La R\u00e9union", "Mayotte", "France"),
    ("mpox", "Qu\u00e9bec", "Ontario", "Canada"),
])
@pytest.mark.parametrize("use_owner", [False, True], ids=["wrong_jurisdiction", "actual_jurisdiction"])
def test_french_comparison_keeps_count_with_its_local_jurisdiction(disease, target, owner, country, use_owner):
    text = (f"En 2025, la surveillance de {disease} se poursuit \u00e0 {target} ({country}). "
            f"En 2025, {owner} ({country}) a notifi\u00e9 8 432 cas confirm\u00e9s de {disease}.")
    qualification = assess(text, disease=disease, country=country,
        subnational_location=owner if use_owner else target, reporting_period="2025", cases_confirmed=8432)
    if use_owner:
        assert qualification.status == "qualified", qualification.reasons
    else:
        assert qualification.status == "candidate"
        count = next(field for field in qualification.field_evidence if field.field == "cases_confirmed")
        assert not count.supported


@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS)
def test_french_country_total_with_majority_in_region_is_not_that_regions_total(disease, region, country):
    text = (f"En 2025, {country} a notifi\u00e9 32 456 cas confirm\u00e9s de {disease}, "
            f"dont la majorit\u00e9 \u00e0 {region}.")
    qualification = assess(text, disease=disease, country=country, subnational_location=region,
                           reporting_period="2025", cases_confirmed=32456)
    assert qualification.status == "candidate"
    count = next(field for field in qualification.field_evidence if field.field == "cases_confirmed")
    assert not count.supported


@pytest.mark.parametrize("date_field,wrong_value", [
    ("metric_period_start", "2025-04-05"), ("metric_period_end", "2025-03-02"),
])
def test_french_date_occurrence_cannot_swap_interval_roles(date_field, wrong_value):
    text = "Guyane (France) a notifi\u00e9 32 456 cas confirm\u00e9s de dengue du 2 mars au 5 avril 2025."
    qualification = assess(text, disease="dengue", country="France", subnational_location="Guyane",
        reporting_period="du 2 mars au 5 avril 2025", cases_confirmed=32456, **{date_field: wrong_value})
    field = next(field for field in qualification.field_evidence if field.field == date_field)
    assert not field.supported


def test_french_publication_date_cannot_supply_a_missing_observation_endpoint():
    text = "Publi\u00e9 le 23 mai 2025. Qu\u00e9bec (Canada) a notifi\u00e9 32 456 cas confirm\u00e9s de mpox."
    qualification = assess(text, disease="mpox", country="Canada", subnational_location="Qu\u00e9bec",
                           metric_period_end="2025-05-23", cases_confirmed=32456)
    assert qualification.status == "candidate"
    field = next(field for field in qualification.field_evidence if field.field == "metric_period_end")
    assert not field.supported


@pytest.mark.parametrize("disease,region", [("chikungunya", "La R\u00e9union"), ("mpox", "Qu\u00e9bec")])
def test_french_explicit_subnational_scope_is_usable_without_inventing_a_country(disease, region):
    text = f"En 2025, {region} a notifi\u00e9 32 456 cas confirm\u00e9s de {disease}."
    qualification = assess(text, disease=disease, subnational_location=region,
                           reporting_period="2025", cases_confirmed=32456)
    assert qualification.status == "qualified", qualification.reasons
    assert not any(field.field == "country" for field in qualification.field_evidence)


def test_french_normalized_quote_maps_line_labels_accents_and_grouping_back_to_raw_span():
    text = ("119 \u00c0 La R\u00e9union (France), 32\u202f456 cas confirm\u00e9s de chikungunya\n"
            "120 ont \u00e9t\u00e9 notifi\u00e9s du 1er janvier au 23 mai 2025.")
    quote = ("\u00c0 La R\u00e9union (France), 32 456 cas confirm\u00e9s de chikungunya "
             "ont \u00e9t\u00e9 notifi\u00e9s du 1er janvier au 23 mai 2025.")
    facts = {"disease": "chikungunya", "country": "France", "subnational_location": "La R\u00e9union",
             "reporting_period": "du 1er janvier au 23 mai 2025", "metric_period_start": "2025-01-01",
             "metric_period_end": "2025-05-23", "cases_confirmed": 32456}
    row, document, chunk = evidence(text, **facts)
    row["field_provenance_json"] = {field: {"quote": quote} for field in facts}
    qualification = assess_record_evidence(row, contract={}, evidence_index={"documents": [document], "evidence_chunks": {"span": chunk}})
    assert qualification.status == "qualified", qualification.reasons
    for field in qualification.field_evidence:
        assert field.document_hash == document["content_hash"]
        assert text[field.locator["char_start"]:field.locator["char_end"]] == field.quote



@pytest.mark.parametrize("label,wrong_field", [
    ("suspects", "cases_confirmed"), ("confirm\u00e9s", "cases_suspected"),
])
@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS[:2])
def test_french_case_qualifier_cannot_be_relabelled(label, wrong_field, disease, region, country):
    text = f"En 2025, {region} ({country}) a notifi\u00e9 32 456 cas {label} de {disease}."
    qualification = assess(text, disease=disease, country=country, subnational_location=region,
                           reporting_period="2025", **{wrong_field: 32456})
    field = next(field for field in qualification.field_evidence if field.field == wrong_field)
    assert not field.supported


@pytest.mark.parametrize("disease,region,country", _FRENCH_TASKS[:2])
def test_french_month_precision_cannot_invent_first_day(disease, region, country):
    text = (f"\u00c0 {region} ({country}), 32 456 cas confirm\u00e9s de {disease} ont \u00e9t\u00e9 notifi\u00e9s "
            "de janvier au 23 mai 2025.")
    qualification = assess(text, disease=disease, country=country, subnational_location=region,
        reporting_period="de janvier au 23 mai 2025", metric_period_start="2025-01-01", cases_confirmed=32456)
    field = next(field for field in qualification.field_evidence if field.field == "metric_period_start")
    assert not field.supported
