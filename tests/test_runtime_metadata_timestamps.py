"""Runtime event dates must not inherit the bundled fixture timestamp."""

from datetime import datetime, timezone

import pytest

from data_collection_workflow.config import load_final_package_policy
from data_collection_workflow.human_review_application import apply_human_review_decisions
from data_collection_workflow.models import FinalPackagePolicy
from data_collection_workflow.nodes.finalization import _build_package_metadata
from data_collection_workflow.nodes.human_review import human_review


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    assert parsed.utcoffset().total_seconds() == 0
    return parsed


@pytest.mark.parametrize("state,contains_fixture", [
    ({"pipeline_mode": "evidence"}, False),
    ({"pipeline_mode": "evidence"}, True),
    ({"content_fetch_summary": {"live_fetch_enabled": True}}, True),
    ({"source_search_execution_summary": {"provider": "tavily", "executed_query_count": 1}}, True),
    ({}, False),
])
def test_real_package_generation_records_current_utc(state, contains_fixture):
    before = datetime.now(timezone.utc)
    metadata = _build_package_metadata(
        state, FinalPackagePolicy(**load_final_package_policy()),
        contains_fixture, None, [], False,
    )
    after = datetime.now(timezone.utc)
    assert before <= _timestamp(metadata["generated_at"]) <= after


def test_pure_offline_fixture_generation_retains_configured_fixed_time():
    policy = FinalPackagePolicy(**load_final_package_policy())
    policy.fixed_generated_at = "2001-02-03T04:05:06Z"
    metadata = _build_package_metadata({}, policy, True, "Synthetic data", [], False)
    assert metadata["generated_at"] == policy.fixed_generated_at


@pytest.mark.parametrize("decided_at", [None, "2026-06-01T00:00:00Z"])
def test_review_recording_fills_current_time_and_preserves_supplied_date(decided_at, monkeypatch):
    monkeypatch.setenv("HUMAN_REVIEW_APPLY_DECISIONS", "false")
    state = {
        "human_review_queue": [{"review_id": "review_1", "item_type": "source_screening", "related_ids": [], "reason": "Check source", "status": "pending"}],
        "human_review_decisions": [{"review_id": "review_1", "decision": "accept", "reviewer_id": "reviewer", "decided_at": decided_at}],
        "collection_trace": [],
    }
    before = datetime.now(timezone.utc)
    result = human_review(state)
    after = datetime.now(timezone.utc)
    actual = result["human_review_queue"][0]["decided_at"]
    if decided_at:
        assert actual == decided_at
    else:
        assert before <= _timestamp(actual) <= after


def test_review_application_and_field_audit_record_current_utc(monkeypatch):
    monkeypatch.setenv("HUMAN_REVIEW_APPLY_DECISIONS", "false")
    decision_time = "2026-06-01T00:00:00Z"
    state = {
        "normalized_records": [{"record_id": "record_1", "cases_unspecified": 4}],
        "human_review_decisions": [{
            "decision_id": "decision_1", "review_id": "review_1",
            "decision_type": "correct_fields", "reviewer_id": "reviewer",
            "decided_at": decision_time, "target_type": "record",
            "target_ids": ["record_1"], "reason": "Verified count",
            "patch": {"cases_unspecified": 5}, "apply_decision": True,
        }],
    }
    before = datetime.now(timezone.utc)
    result = apply_human_review_decisions(state)
    after = datetime.now(timezone.utc)
    applied = result["applied_human_review_decisions"][0]
    audits = result["human_review_audit_trail"]
    assert audits
    assert result["final_dataset_post_review"][0]["cases_unspecified"] == 5
    assert state["normalized_records"][0]["cases_unspecified"] == 4
    for event in [applied, *audits]:
        assert event["decided_at"] == decision_time
        assert before <= _timestamp(event["applied_at"]) <= after
