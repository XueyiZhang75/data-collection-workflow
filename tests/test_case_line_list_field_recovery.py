from __future__ import annotations

import json

from data_collection_workflow.case_field_recovery import recover_line_list_rows


def test_recovery_fills_explicit_case_span_fields_without_benchmark_data():
    bundle_rows = [
        {
            "workflow_row_id": "WFL-0001",
            "bundle_id": "case_bundle_0001",
            "bundle_type": "individual_case_like",
            "workflow_case_label": "Case 5",
            "case_label_normalized": "case_5",
            "case_status": "confirmed_case_record",
            "case_count": "1",
            "age": "",
            "gender": "male",
            "nationality": "",
            "occupation_or_role": "",
            "date_onset": "",
            "date_confirmation": "",
            "symptoms": "fever",
        }
    ]
    candidate_rows = [
        {
            "workflow_candidate_id": "WFC-0001",
            "candidate_id": "candidate_case_5",
            "bundle_id": "case_bundle_0001",
            "candidate_type": "individual_case_candidate",
            "source_id": "src_who",
            "workflow_case_label": "Case 5",
            "case_label_normalized": "case_5",
            "age": "",
            "gender": "male",
            "nationality": "",
            "occupation_or_role": "",
            "date_onset": "",
            "date_confirmation": "",
            "symptoms": "fever",
            "case_span_id": "case_span_case_5",
            "case_span_quote": (
                "Case 5: A 41 -year -old Dutch male national, serving as the "
                "ship's doctor, reported onset of symptoms on 30 April 2026, "
                "including fever, fatigue, and myalgia. PCR testing confirmed "
                "Andes virus infection on 6 May 2026."
            ),
            "evidence_quote": "",
        }
    ]

    recovered_bundles, recovered_candidates, summary = recover_line_list_rows(
        bundle_rows,
        candidate_rows,
    )

    candidate = recovered_candidates[0]
    assert candidate["age"] == "41"
    assert candidate["nationality"] == "Dutch"
    assert candidate["occupation_or_role"] == "ship doctor"
    assert candidate["date_onset"] == "2026-04-30"
    assert candidate["date_confirmation"] == "2026-05-06"
    assert candidate["confirmation_method"] == "PCR"
    assert candidate["symptoms"] == "fever; fatigue; myalgia"
    assert "age" in candidate["recovered_fields"].split("; ")
    provenance = json.loads(candidate["field_provenance_json"])
    assert provenance["age"]["source_id"] == "src_who"
    assert provenance["age"]["case_span_id"] == "case_span_case_5"
    assert provenance["age"]["supporting_quote"] == candidate["case_span_quote"]

    bundle = recovered_bundles[0]
    assert bundle["age"] == "41"
    assert bundle["nationality"] == "Dutch"
    assert bundle["occupation_or_role"] == "ship doctor"
    assert bundle["date_confirmation"] == "2026-05-06"
    assert json.loads(bundle["field_source_ids_json"])["age"] == ["src_who"]
    assert json.loads(bundle["field_conflicts_json"]) == {}
    assert summary["recovered_candidate_field_count"] >= 6
    assert summary["recovered_bundle_field_count"] >= 4
    assert "Gh_ID" not in bundle


def test_recovery_does_not_assign_individual_fields_to_aggregate_bundles():
    bundles = [
        {
            "workflow_row_id": "WFL-0002",
            "bundle_id": "case_bundle_0002",
            "bundle_type": "aggregate_event",
            "case_count": "13",
            "age": "",
            "gender": "",
        }
    ]
    candidates = [
        {
            "workflow_candidate_id": "WFC-0002",
            "candidate_id": "candidate_snapshot",
            "bundle_id": "case_bundle_0002",
            "candidate_type": "aggregate_event_candidate",
            "source_id": "src_who",
            "case_span_quote": "The outbreak included 13 confirmed cases and 3 deaths.",
            "evidence_quote": "The outbreak included 13 confirmed cases and 3 deaths.",
        }
    ]

    recovered_bundles, _, _ = recover_line_list_rows(bundles, candidates)

    assert recovered_bundles[0]["age"] == ""
    assert recovered_bundles[0]["gender"] == ""
    assert recovered_bundles[0]["field_recovery_method"] == (
        "not_applicable_for_non_individual_bundle"
    )
