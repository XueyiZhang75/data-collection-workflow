"""Known geographic aliases retain their source role at the contract boundary."""
import pytest
from data_collection_workflow.geography import explicit_country_key
from data_collection_workflow.evidence_qualification import _constraint_reasons

@pytest.mark.parametrize("field", ["subnational_location", "geographic_scope"])
def test_existing_cldr_territory_alias_applies_to_noncountry_field(field):
    record = {"country": "France", field: "La R\u00e9union", "geographic_scope_type": "subnational"}
    assert explicit_country_key("La R\u00e9union") == explicit_country_key("R\u00e9union")
    assert not _constraint_reasons(record, {"location": "R\u00e9union", "geography": "R\u00e9union"})

@pytest.mark.parametrize("value,target", [("Allemagne", "Germany"), ("Canad\u00e1", "Canada")])
def test_existing_cldr_cross_language_scope_alias(value, target):
    assert explicit_country_key(value) == explicit_country_key(target)
    assert not _constraint_reasons({"geographic_scope": value, "geographic_scope_type": "country"}, {"location": target})

@pytest.mark.parametrize("record,target", [
    ({"country": "France", "subnational_location": "La R\u00e9union"}, "Martinique"),
    ({"country": "France"}, "R\u00e9union"),
    ({"subnational_location": "La R\u00e9union"}, "France"),
    ({"subnational_location": "La Example"}, "Example"),
    ({"country": "United States", "locality": "G\u00e9orgie", "geographic_scope": "G\u00e9orgie", "geographic_scope_type": "county"}, "Georgia"),
    ({"country": "United States", "geographic_scope": "G\u00e9orgie", "geographic_scope_type": "county"}, "Georgia"),
])
def test_alias_match_does_not_infer_parent_strip_article_or_promote_locality(record, target):
    assert "location:contract_mismatch" in _constraint_reasons(record, {"location": target})


def test_unknown_exact_location_remains_supported_without_alias_invention():
    assert not _constraint_reasons({"locality": "Example District", "geographic_scope_type": "locality"}, {"location": "Example District"})


@pytest.mark.parametrize("level,field", [("state", "subnational_location"), ("county", "locality"), ("city", "locality")])
def test_explicit_parent_country_keeps_same_name_local_target(level, field):
    record = {"country": "United States", field: "Georgia", "geographic_scope": "Georgia", "geographic_scope_type": level}
    assert not _constraint_reasons(record, {"country": "United States", "location": "Georgia", field: "Georgia"})
    assert "country:contract_mismatch" in _constraint_reasons(record, {"country": "Georgia", "location": "Georgia"})


@pytest.mark.parametrize("level", ["county", "city", "state", "province"])
def test_country_alias_does_not_translate_explicit_local_scope(level):
    record = {"country": "United States", "subnational_location": "Géorgie", "geographic_scope": "Géorgie", "geographic_scope_type": level}
    assert "location:contract_mismatch" in _constraint_reasons(record, {"location": "Georgia", "country": "United States"})


def test_known_country_alias_never_supplies_missing_country_role():
    record = {"subnational_location": "Allemagne", "geographic_scope_type": "subnational"}
    assert "country:contract_mismatch" in _constraint_reasons(record, {"country": "Germany", "location": "Germany"})


def test_same_local_name_in_wrong_parent_country_stays_mismatched():
    record = {"country": "Canada", "locality": "Example District", "geographic_scope_type": "county"}
    assert "country:contract_mismatch" in _constraint_reasons(record, {"country": "United States", "location": "Example District"})


def test_alias_only_changes_contract_match_not_source_facts():
    import copy
    import hashlib
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    text = "Pertussis in La Réunion: 12 confirmed cases during 2024."
    record = {"record_id": "r", "source_id": "s", "chunk_id": "c", "disease": "Pertussis", "subnational_location": "La Réunion", "reporting_period": "2024", "cases_confirmed": 12}
    original = copy.deepcopy(record)
    doc = {"source_id": "s", "clean_text": text, "content_hash": hashlib.sha256(text.encode()).hexdigest()}
    chunk = {"chunk_id": "c", "source_id": "s", "text": text, "char_start": 0, "char_end": len(text)}
    index = build_evidence_index({"documents": [doc], "evidence_chunks": [chunk]})
    qualified = assess_record_evidence(record, contract={"location": "Réunion"}, evidence_index=index)
    assert qualified.status == "qualified", qualified.reasons
    assert record == original
    unsupported = dict(record, country="France")
    q = assess_record_evidence(unsupported, contract={"location": "Réunion"}, evidence_index=index)
    assert q.status == "candidate"
    assert "country:unbound_field_value" in q.reasons
    assert unsupported["country"] == "France"


def test_broader_territory_alias_remains_usable_for_local_observation():
    record = {"country": "France", "subnational_location": "La Réunion", "locality": "Saint-Denis", "geographic_scope": "Saint-Denis", "geographic_scope_type": "city"}
    assert not _constraint_reasons(record, {"location": "Réunion", "country": "France"})


def test_untyped_literal_preserves_existing_match_without_guessing_a_country():
    record = {"country": "United States", "subnational_location": "Georgia", "geographic_scope_type": "state"}
    assert not _constraint_reasons(record, {"location": "Georgia"})
    assert record["country"] == "United States"
