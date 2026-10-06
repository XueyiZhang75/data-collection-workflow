from copy import deepcopy

import pytest

from data_collection_workflow.source_coverage import annotate_source_coverage


TASKS = [
    ("mpox", "United States", "https://t.cdc.gov/short"),
    ("measles", "Canada", "https://www.canada.ca/public-health/short"),
    ("dengue", "Brazil", "https://www.gov.br/saude/short"),
]


def _state(disease, location):
    return {"structured_task": {"disease": disease, "location": location,
             "start_date": "2025-01-01", "end_date": "2025-12-31"}}


def _candidate(url, **overrides):
    return {"source_id": "current_source", "canonical_url": url,
            "title": "Using nicotine replacement therapy",
            "source_type": "official_public_health_agency",
            "source_role": "data_source", "source_role_final": "collection",
            "target_fit_status": "task_record_collection_candidate",
            "triage_role": "task_record_collection_candidate",
            "disease_fit": "candidate", "geography_fit": "candidate",
            "date_fit": "candidate", "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True, **overrides}


@pytest.mark.parametrize("disease,location,url", TASKS)
@pytest.mark.parametrize("negative", [
    {"screening_decision": "exclude"},
    {"critic_decision": "exclude"},
    {"source_role": "irrelevant_source"},
    {"source_role_fit": "irrelevant_source"},
])
def test_generic_official_candidate_does_not_override_explicit_rejection(disease, location, url, negative):
    source = _candidate(url, **negative)
    original = deepcopy(source)
    rows, _, _ = annotate_source_coverage([source], _state(disease, location))
    assert not rows[0].get("must_fetch")
    assert rows[0]["final_screening_decision"] == "exclude"
    assert rows[0]["ready_for_content_fetch"] is False
    assert source == original


@pytest.mark.parametrize("negative", [
    {"source_role_fit": "irrelevant_source", "screening_decision": "exclude"},
    {"source_disease_relevance_status": "unrelated_disease"},
    {"geography_fit": "wrong_geography"},
])
def test_repeated_annotation_revokes_stale_coverage_priority_after_negative_fit(negative):
    state = _state("pertussis", "United Kingdom")
    source = _candidate("https://www.gov.uk/report", title="Pertussis United Kingdom 2025")
    first, requirements, _ = annotate_source_coverage([source], state)
    assert first[0]["must_fetch"] is True
    later = {**first[0], **negative}
    second, _, _ = annotate_source_coverage([later], state)
    assert second[0]["must_fetch"] is False
    assert second[0]["coverage_requirement_ids"] == []
    assert second[0]["blocked_from_fetch"] is True
    assert "target_official_must_fetch" not in second[0].get("routing_flags", [])
    third, _, _ = annotate_source_coverage(second, state)
    assert second == third


@pytest.mark.parametrize("disease,location,url", TASKS)
def test_supported_target_remains_must_fetch_with_review_routing(disease, location, url):
    source = _candidate(url, title=f"{disease} surveillance in {location} 2025",
                        screening_decision="needs_human_review",
                        final_screening_decision="needs_human_review",
                        ready_for_content_fetch=False)
    rows, _, _ = annotate_source_coverage([source], _state(disease, location))
    assert rows[0]["must_fetch"] is True
    assert rows[0]["final_screening_decision"] == "include_for_content_fetch"
    assert rows[0]["ready_for_content_fetch"] is True


def test_concrete_target_metadata_can_repair_weak_excluded_decision():
    source = _candidate("https://www.gov.uk/pertussis-2025", title="Pertussis United Kingdom 2025",
                        screening_decision="exclude", blocked_from_fetch_reason="source_identity_recommended_excluded")
    rows, _, _ = annotate_source_coverage([source], _state("pertussis", "United Kingdom"))
    assert rows[0]["must_fetch"] is True
    assert rows[0]["ready_for_content_fetch"] is True


@pytest.mark.parametrize("disease,location,url", TASKS)
@pytest.mark.parametrize("fit", ["candidate", "possible"])
def test_included_official_page_with_only_uncertain_fit_remains_ordinary_candidate(disease, location, url, fit):
    source = _candidate(url, title="Public information portal", screening_decision="include",
                        disease_fit=fit, geography_fit=fit, date_fit=fit)
    rows, _, _ = annotate_source_coverage([source], _state(disease, location))
    assert not rows[0].get("must_fetch")
    assert rows[0]["final_screening_decision"] == "include_for_content_fetch"
    assert rows[0]["ready_for_content_fetch"] is True
    assert not rows[0].get("blocked_from_fetch")


def test_stale_candidate_only_priority_reverts_to_ordinary_fetch_without_exclusion():
    state = _state("pertussis", "United Kingdom")
    source = _candidate("https://www.gov.uk/report", title="Pertussis United Kingdom 2025")
    first, _, _ = annotate_source_coverage([source], state)
    assert first[0]["must_fetch"] is True
    later = {**first[0], "title": "Public information portal"}
    second, _, _ = annotate_source_coverage([later], state)
    assert second[0]["must_fetch"] is False
    assert second[0]["coverage_requirement_ids"] == []
    assert second[0]["ready_for_content_fetch"] is True
    assert second[0]["final_screening_decision"] == "include_for_content_fetch"
    assert second[0]["blocked_from_fetch"] is False


@pytest.mark.parametrize("disease,location,url", TASKS)
def test_confirmed_fit_flags_can_establish_mandatory_priority(disease, location, url):
    source = _candidate(url, title="Surveillance download", disease_fit="match",
                        geography_fit="match", date_fit="match")
    rows, _, _ = annotate_source_coverage([source], _state(disease, location))
    assert rows[0]["must_fetch"] is True
    assert rows[0]["ready_for_content_fetch"] is True


def test_fetched_source_text_can_confirm_fit_without_using_task_as_evidence():
    source = _candidate("https://www.gov.uk/report", title="Surveillance archive")
    state = _state("pertussis", "United Kingdom")
    candidate, _, _ = annotate_source_coverage([source], state)
    assert not candidate[0].get("must_fetch")
    document = {"source_id": source["source_id"], "parse_status": "parsed",
                "fetch_status": "success", "usable_for_task_collection": True,
                "clean_text": "Pertussis observations for United Kingdom in 2025."}
    supported, _, _ = annotate_source_coverage([source], state, documents=[document])
    assert supported[0]["must_fetch"] is True
    assert "snippet" not in supported[0]
    unrelated, _, _ = annotate_source_coverage([source], state, documents=[{**document, "source_id": "another_source"}])
    assert not unrelated[0].get("must_fetch")


@pytest.mark.parametrize("last_version", [
    {"fetch_status": "deferred_budget", "parse_status": "parse_deferred", "clean_text": ""},
    {"fetch_status": "success", "parse_status": "parsed", "clean_text": "Dengue observations in Brazil during 2023."},
])
def test_same_source_document_versions_are_order_independent(last_version):
    from itertools import permutations

    source = _candidate("https://www.gov.uk/report", title="Surveillance archive")
    state = _state("pertussis", "United Kingdom")
    matched = {"source_id": source["source_id"], "fetch_status": "success",
               "parse_status": "parsed", "usable_for_task_collection": True,
               "clean_text": "Pertussis observations for United Kingdom in 2025."}
    other = {"source_id": source["source_id"], **last_version}
    outputs = []
    for versions in permutations([matched, other]):
        rows, _, _ = annotate_source_coverage([source], state, documents=list(versions))
        assert rows[0].get("must_fetch") is True
        outputs.append(rows)
    assert outputs[0] == outputs[1]


def test_partial_matches_from_different_document_versions_are_not_combined():
    from itertools import permutations

    source = _candidate("https://www.gov.uk/report", title="Surveillance archive")
    state = _state("pertussis", "United Kingdom")
    versions = [{"source_id": source["source_id"], "fetch_status": "success",
                 "parse_status": "parsed", "usable_for_task_collection": True,
                 "clean_text": text} for text in (
                     "Pertussis observations.", "United Kingdom surveillance.", "Data for 2025."
                 )]
    for ordered in permutations(versions):
        rows, _, _ = annotate_source_coverage([source], state, documents=list(ordered))
        assert not rows[0].get("must_fetch")
        assert rows[0]["ready_for_content_fetch"] is True
        assert not rows[0].get("blocked_from_fetch")


def test_unreadable_document_cannot_confirm_priority_even_with_retained_text():
    source = _candidate("https://www.gov.uk/report", title="Surveillance archive")
    document = {"source_id": source["source_id"], "fetch_status": "success",
                "parse_status": "parsed", "content_readable": False,
                "clean_text": "Pertussis observations for United Kingdom in 2025."}
    rows, _, _ = annotate_source_coverage([source], _state("pertussis", "United Kingdom"), documents=[document])
    assert not rows[0].get("must_fetch")
