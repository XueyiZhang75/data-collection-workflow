"""Independent source-bound count meaning and evidence-span identity controls."""
from copy import deepcopy
from dataclasses import asdict

import pytest

from data_collection_workflow.evidence_qualification import assess_record_evidence
from test_acquisition_evidence import evidence

SEMANTIC_FIELDS = ("count_semantics", "statistical_count_type")


def _assess(text, *, value=43, semantics=None, disease="measles", country="Canada", **extra):
    fields = {"disease": disease, "country": country, "reporting_period": "2025", **extra}
    if value is not None:
        fields["cases_confirmed"] = value
    if semantics is not None:
        fields.update(count_semantics=semantics, statistical_count_type=semantics)
    row, doc, chunk = evidence(text, **fields)
    original = deepcopy(row)
    result = assess_record_evidence(
        row, contract={},
        evidence_index={"documents": [doc], "evidence_chunks": {"span": chunk}},
    )
    assert row == original, "Qualification must not remove fields or substitute counts."
    return result


def _semantic_support(result):
    fields = [f for f in result.field_evidence if f.field in SEMANTIC_FIELDS]
    assert len(fields) == 2
    return [f.supported for f in fields]


@pytest.mark.parametrize("disease,country", [("measles", "Canada"), ("pertussis", "France")])
@pytest.mark.parametrize("semantics,template", [
    ("subset", "During 2025, {country} reported 43 confirmed {disease} cases out of 187 national cases."),
    ("cumulative", "In 2025, {country} reported a total of 43 confirmed {disease} cases since January."),
    ("newly_reported", "During 2025, {country} reported 43 newly reported confirmed {disease} cases."),
])
def test_canonical_meaning_is_supported_by_the_same_english_count(disease, country, semantics, template):
    result = _assess(template.format(disease=disease, country=country),
                     semantics=semantics, disease=disease, country=country)
    assert all(f.supported for f in result.field_evidence if f.field not in SEMANTIC_FIELDS), result.reasons
    assert _semantic_support(result) == [True, True]
    assert result.status == "qualified", result.reasons
    assert result.product_kind == "aggregate"


@pytest.mark.parametrize("semantics,text,value", [
    ("subset", "En 2025, la France a signalé 43 cas confirmés de chikungunya, dont 5 cas confirmés chez les enfants.", 5),
    ("cumulative", "En 2025, la France a signalé un total cumulé de 43 cas confirmés de chikungunya depuis janvier.", 43),
    ("newly_reported", "En 2025, la France a signalé 43 nouveaux cas confirmés de chikungunya.", 43),
])
def test_explicit_french_count_meanings_are_not_literal_english_enums(semantics, text, value):
    result = _assess(text, value=value, semantics=semantics, disease="chikungunya", country="France")
    assert all(f.supported for f in result.field_evidence if f.field not in SEMANTIC_FIELDS), result.reasons
    assert _semantic_support(result) == [True, True]
    assert result.status == "qualified", result.reasons


@pytest.mark.parametrize("semantics,text", [
    ("cumulative", "During 2025, Canada reported a total of 43 confirmed measles cases."),
    ("cumulative", "Canada reported 43 confirmed measles cases on 31 May 2025."),
    ("cumulative", "Canada reported a total of 43 confirmed measles cases during 2025, since diagnostic samples were available."),
    ("annual", "Canada reported 43 confirmed measles cases on 31 May 2025."),
    ("subset", "During 2025, Canada reported 43 confirmed measles cases."),
    ("newly_reported", "During 2025, Canada reported 43 confirmed measles cases."),
])
def test_total_reporting_verb_or_one_date_does_not_supply_count_meaning(semantics, text):
    result = _assess(text, semantics=semantics)
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("semantics,text", [
    ("newly_reported", "Canada reported 43 confirmed measles cases during 2025; Canada also reported 5 new confirmed measles cases during 2025."),
    ("cumulative", "Canada reported 43 confirmed measles cases during 2025; Canada recorded a cumulative total of 5 confirmed measles cases since January 2025."),
    ("subset", "Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases among children."),
])
def test_another_count_modifier_cannot_label_the_record_count(semantics, text):
    result = _assess(text, value=43, semantics=semantics)
    numeric = next(f for f in result.field_evidence if f.field == "cases_confirmed")
    assert numeric.supported, result.reasons
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


def test_including_supports_the_included_count_but_not_the_total():
    text = "Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases among children."
    included = _assess(text, value=5, semantics="subset")
    total = _assess(text, value=43, semantics="subset")
    assert _semantic_support(included) == [True, True]
    assert included.status == "qualified", included.reasons
    assert _semantic_support(total) == [False, False]


@pytest.mark.parametrize("semantics,foreign", [
    ("newly_reported", "France reported 43 new confirmed measles cases during 2025."),
    ("cumulative", "France reported a cumulative total of 43 confirmed measles cases since January 2025."),
    ("subset", "France reported 43 confirmed measles cases out of 187 national cases during 2025."),
])
@pytest.mark.parametrize("separator", [". ", ", while "])
def test_same_number_in_foreign_statement_cannot_supply_semantics(semantics, foreign, separator):
    text = "Canada reported 43 confirmed measles cases during 2025" + separator + foreign
    result = _assess(text, semantics=semantics)
    assert next(f for f in result.field_evidence if f.field == "cases_confirmed").supported
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("semantics,text", [
    ("newly_reported", "Canada did not report 43 new confirmed measles cases during 2025."),
    ("cumulative", "Canada did not report a cumulative total of 43 confirmed measles cases in 2025."),
    ("subset", "Canada did not report 43 confirmed measles cases out of 187 national cases in 2025."),
    ("newly_reported", "In a hypothetical scenario, Canada would report 43 new confirmed measles cases in 2025."),
    ("subset", "In a hypothetical scenario, Canada would report 43 confirmed measles cases out of 187 national cases in 2025."),
])
def test_negated_or_hypothetical_meaning_is_not_observed_evidence(semantics, text):
    result = _assess(text, semantics=semantics)
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("semantics", ["subset", "cumulative", "newly_reported"])
def test_test_denominator_is_not_a_case_count_with_semantics(semantics):
    result = _assess(
        "During 2025 in Canada, among 200 tests for measles, 5 new confirmed measles cases were reported.",
        value=200, semantics=semantics,
    )
    assert not next(f for f in result.field_evidence if f.field == "cases_confirmed").supported
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("unrecognized", ["unsupported_count_enum", "lifetime_total", "subsampled_future"])
def test_unrecognized_enum_is_not_replaced_with_a_supported_known_meaning(unrecognized):
    result = _assess(
        "During 2025, Canada reported a cumulative total of 43 confirmed measles cases since January.",
        semantics=unrecognized,
    )
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("span_field", ["case_span_id", "evidence_span_id"])
@pytest.mark.parametrize("number", [1, 17])
def test_source_span_identity_does_not_turn_an_aggregate_into_a_patient(span_field, number):
    result = _assess(f"Canada reported {number} confirmed measles cases during 2025.",
                     value=number, **{span_field: "source-local-locator"})
    assert result.status == "qualified", result.reasons
    assert result.product_kind == "aggregate"
    assert not any(f.field in {"case_id", "patient_id", "case_label", "case_identifier", "workflow_case_label"}
                   for f in result.field_evidence)


@pytest.mark.parametrize("span_field", ["case_span_id", "evidence_span_id"])
def test_a_span_without_numeric_or_patient_identity_remains_context(span_field):
    result = _assess("Canada monitored measles surveillance during 2025.", value=None,
                     **{span_field: "source-local-locator"})
    assert result.product_kind != "case_level"
    assert not any(f.field in {"case_id", "patient_id", "case_label", "case_identifier", "workflow_case_label"}
                   for f in result.field_evidence)


@pytest.mark.parametrize("text,country,extra", [
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases in Quebec.", "Canada", {"subnational_location": "Quebec"}),
    ("43 confirmed measles cases were reported during 2025, including 5 confirmed measles cases in France.", "France", {}),
])
def test_parent_count_cannot_borrow_the_child_geography(text, country, extra):
    result = _assess(text, value=43, semantics="subset", country=country, **extra)
    assert not next(f for f in result.field_evidence if f.field == "cases_confirmed").supported
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("text,disease,country,period", [
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases in France.", "measles", "Canada", "2025"),
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed dengue cases.", "measles", "Canada", "2025"),
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases from 2024.", "measles", "Canada", "2025"),
])
def test_explicit_child_scope_overrides_parent_scope_instead_of_combining(text, disease, country, period):
    result = _assess(text, value=5, semantics="subset", disease=disease, country=country, reporting_period=period)
    assert not next(f for f in result.field_evidence if f.field == "cases_confirmed").supported
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("text,disease,country,period", [
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases in France.", "measles", "France", "2025"),
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed dengue cases.", "dengue", "Canada", "2025"),
    ("Canada reported 43 confirmed measles cases during 2025, including 5 confirmed measles cases from 2024.", "measles", "Canada", "2024"),
])
def test_child_explicit_scope_and_remaining_parent_scope_are_source_bound(text, disease, country, period):
    result = _assess(text, value=5, semantics="subset", disease=disease, country=country, reporting_period=period)
    assert result.status == "qualified", result.reasons
    assert _semantic_support(result) == [True, True]


@pytest.mark.parametrize("boundary", [
    ", including France reported 5 confirmed measles cases.",
    ", including the laboratory reported 5 confirmed measles cases.",
    "; 5 confirmed measles cases were reported.",
    ". Five observations were investigated, including 5 confirmed measles cases.",
    ". Including 5 confirmed measles cases among children.",
])
def test_new_subject_reporting_predicate_or_sentence_cannot_inherit_parent_scope(boundary):
    result = _assess("Canada reported 43 confirmed measles cases during 2025" + boundary,
                     value=5, semantics="subset")
    assert not next(f for f in result.field_evidence if f.field == "cases_confirmed").supported
    assert _semantic_support(result) == [False, False]
    assert result.status == "candidate"


@pytest.mark.parametrize("disease,country,locations,counts", [
    ("mpox", "Sierra Leone", ("Western Urban Area", "Western Rural Area"), (43, 5)),
    ("measles", "Canada", ("Quebec", "Ontario"), (11, 7)),
])
def test_qualified_study_subsets_preserve_geography_and_are_not_national_totals(disease, country, locations, counts):
    from data_collection_workflow.evidence_products import build_evidence_products

    rows, documents, chunks = [], [], {}
    for i, (location, count) in enumerate(zip(locations, counts)):
        text = (f"During 2025, a study in {location}, {country} reported "
                f"{count} confirmed {disease} cases out of 187 study cases.")
        row, doc, chunk = evidence(text, disease=disease, country=country,
            subnational_location=location, reporting_period="2025", cases_confirmed=count,
            count_semantics="subset", statistical_count_type="subset", evidence_span_id=f"study-span-{i}")
        row.update(record_id=f"study-row-{i}", supporting_chunk_id=f"span-{i}")
        doc["document_id"] = f"study-document-{i}"
        chunk.update(chunk_id=f"span-{i}", document_id=doc["document_id"])
        index = {"documents": [doc], "evidence_chunks": {f"span-{i}": chunk}}
        qualification = assess_record_evidence(row, contract={}, evidence_index=index)
        assert qualification.status == "qualified", qualification.reasons
        assert qualification.product_kind == "aggregate"
        row["evidence_qualification"] = asdict(qualification)
        rows.append(row)
        documents.append(doc)
        chunks.update(index["evidence_chunks"])

    products = build_evidence_products(rows, evidence_index={"documents": documents, "evidence_chunks": chunks})
    assert not products["excluded_observations"]
    assert not products["case_entities"]
    assert not products["derivations"]
    assert len(products["aggregate_groups"]) == 2
    assert len(products["time_series_observations"]) == 2
    assert {series["value"] for series in products["time_series_observations"]} == set(counts)
    for observation in products["aggregate_groups"] + products["time_series_observations"]:
        scope = observation["scope"]
        assert scope["subnational_location"] in locations
        assert scope["count_semantics"] == scope["statistical_count_type"] == "subset"
        assert scope.get("count_basis") != "cumulative"
    assert all(series["value"] != sum(counts) for series in products["time_series_observations"])
