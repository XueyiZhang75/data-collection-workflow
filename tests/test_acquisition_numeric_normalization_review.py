"""Independent full-flow controls for preserved numeric claims and admission."""
import socket

import pytest

from test_acquisition_numeric_normalization import normalize
from test_acquisition_evidence import evidence
from data_collection_workflow.evidence_qualification import qualify_records


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    def denied(*args, **kwargs):
        raise AssertionError("external network forbidden")
    monkeypatch.setattr(socket.socket, "connect", denied)


def checked(text, disease, country, field, value):
    row, qualification = normalize(text, disease, country, field, value)
    _, document, chunk = evidence(text)
    groups = qualify_records([row], contract={}, evidence_index={"documents": [document], "evidence_chunks": {"span": chunk}})
    assert row[field] == value, "Preserve the claim for audit even when evidence rejects it."
    return qualification, groups


@pytest.mark.parametrize("disease,country,field,claim", [
    ("measles", "Canada", "cases_confirmed", "4 156 new confirmed measles cases"),
    ("dengue", "Brazil", "cases_suspected", "4 156 newly reported suspected dengue cases"),
    ("chikungunya", "France", "cases_unspecified", "4 156 nouveaux cas de chikungunya"),
    ("measles", "Canada", "cases_confirmed", "4\u202f156 nouveaux cas confirm\u00e9s de measles"),
])
def test_explicit_new_count_modifiers_survive_full_flow(disease, country, field, claim):
    text = f"{country}, 2025: {claim}, an increase of 16%."
    qualification, groups = checked(text, disease, country, field, 4156)
    assert qualification.status == "qualified", qualification.reasons
    assert len(groups["qualified_records"]) == 1


@pytest.mark.parametrize("field,value,text", [
    ("cases_unspecified", 16, "Canada reported 4 156 new measles cases during 2025, an increase of 16%."),
    ("cases_unspecified", 124, "Page 124\nNew cases of measles in Canada during 2025 are discussed in this report."),
    ("cases_confirmed", 16, "In Canada during 2025, a measles patient aged 16 years was treated."),
    ("cases_confirmed", 4156, "Canada reported 4 156 new suspected measles cases during 2025."),
    ("cases_suspected", 4156, "Canada reported 4 156 new confirmed measles cases during 2025."),
    ("deaths", 4156, "Canada reported 4 156 new measles cases and 2 deaths during 2025."),
    ("cases_unspecified", 4156, "Canada reported more than 4 156 new measles cases during 2025."),
    ("cases_unspecified", 4156, "Canada reported 4 156-4 200 new measles cases during 2025."),
    ("cases_confirmed", 4156, "Canada did not report 4 156 new confirmed measles cases during 2025."),
])
def test_retained_wrong_count_never_enters_qualified_records(field, value, text):
    qualification, groups = checked(text, "measles", "Canada", field, value)
    count = next(item for item in qualification.field_evidence if item.field == field)
    assert not count.supported, qualification.reasons
    assert qualification.status != "qualified"
    assert groups["qualified_records"] == []


@pytest.mark.parametrize("disease,country,other", [("measles", "Canada", "France"), ("dengue", "Brazil", "Peru")])
def test_new_count_does_not_borrow_other_country_or_nearby_year(disease, country, other):
    texts = [
        f"{country} monitored {disease} during 2025. {other} reported 4 156 new confirmed {disease} cases during 2025.",
        f"{country} monitored {disease} during 2025. {country} reported 4 156 new confirmed {disease} cases during 2024.",
    ]
    for text in texts:
        qualification, groups = checked(text, disease, country, "cases_confirmed", 4156)
        assert qualification.status != "qualified", (text, qualification.reasons)
        assert groups["qualified_records"] == []


@pytest.mark.parametrize("claim", ["n'a pas d\u00e9clar\u00e9 4 156 nouveaux cas", "a d\u00e9clar\u00e9 au moins 4 156 nouveaux cas", "a d\u00e9clar\u00e9 16% de nouveaux cas"])
def test_french_polarity_bound_and_percentage_remain_nonfacts(claim):
    value = 16 if "16%" in claim else 4156
    text = f"France, en 2025, {claim} de chikungunya."
    qualification, groups = checked(text, "chikungunya", "France", "cases_unspecified", value)
    assert qualification.status != "qualified", qualification.reasons
    assert groups["qualified_records"] == []


@pytest.mark.parametrize("role", ["Page", "Ligne", "Num\u00e9ro"])
def test_french_explicit_number_roles_do_not_become_new_case_counts(role):
    text = f"{role} 124\nNouveaux cas de chikungunya en France durant 2025."
    qualification, groups = checked(text, "chikungunya", "France", "cases_unspecified", 124)
    assert not next(item for item in qualification.field_evidence if item.field == "cases_unspecified").supported
    assert groups["qualified_records"] == []


@pytest.mark.parametrize("disease,country,claim", [
    ("measles", "Canada", "4 156\nnew measles cases"),
    ("chikungunya", "France", "4\u202f156\nnouveaux cas de chikungunya"),
])
def test_real_grouped_count_keeps_newline_before_new_case_label(disease, country, claim):
    text = f"{country}, 2025: {claim}."
    qualification, groups = checked(text, disease, country, "cases_unspecified", 4156)
    assert qualification.status == "qualified", qualification.reasons
    assert len(groups["qualified_records"]) == 1
