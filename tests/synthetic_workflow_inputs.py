"""Small, task-parameterized inputs created only inside pytest temporary paths."""
from __future__ import annotations

import json
from html import escape
from pathlib import Path


def write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_workflow_config(
    tmp_path: Path, *, disease: str = "Example disease", location: str = "Example region",
    year: str = "2024", phase: str = "task",
) -> Path:
    """Write one task, at most one search hit, one table, and one review decision."""
    from data_collection_workflow.nodes.source_discovery import _source_id_from_canonical_url

    tmp_path.mkdir(parents=True, exist_ok=True)
    search_enabled = phase != "task"
    fetch_enabled = phase in {"fetch", "collection", "review"}
    url = "https://health.example.gov/surveillance/report"
    title = f"{location} {disease} surveillance report {year}"
    search_path = write_json(tmp_path / "search.json", {"queries": [{
        "match_terms": [disease], "results": [{"url": url, "title": title,
            "snippet": f"{location} reported 12 {disease} cases and 1 death in {year}.",
            "published_date": f"{year}-06-01", "source": "Example Department of Health", "rank": 1}]}]})
    page = tmp_path / "report.html"
    page.write_text(
        f'<html><head><title>{escape(title)}</title>'
        f'<meta name="date" content="{year}-06-01"></head><body><h1>{escape(title)}</h1>'
        f'<p>Official public health surveillance in {escape(location)}, United States. '
        f'This test table reports {escape(disease)} cases, deaths, and hospitalizations '
        f'for the week ending {year}-06-01.</p>'
        '<table><tr><th>Week ending</th><th>Location</th><th>Cases</th><th>Deaths</th>'
        f'<th>Hospitalizations</th></tr><tr><td>{year}-06-01</td><td>{escape(location)}</td>'
        '<td>12</td><td>1</td><td>3</td></tr></table></body></html>', encoding="utf-8")
    map_path = write_json(tmp_path / "content.json", {"fixtures": [{
        "canonical_url": url, "fixture_path": str(page),
        "content_type": "text/html; charset=utf-8", "http_status_code": 200}]})
    decision_path = write_json(tmp_path / "decisions.json", {"decisions": [{
        "decision_id": "test_source_review", "review_id": "review_source",
        "decision_type": "approve_source_role", "reviewer_id": "test_reviewer",
        "decided_at": f"{year}-06-02T00:00:00Z", "target_type": "source",
        "target_ids": [_source_id_from_canonical_url(url)], "reason": "Test explicit source approval.",
        "patch": {"source_role_final": "collection"}, "apply_decision": True}]})
    config = {
        "profile_name": "temporary_test_task", "pipeline_mode": "standard",
        "user_request": f"Collect {disease} data for {location} in {year}.",
        "workflow": {"collection_mode": "standard", "use_fixture_documents": False},
        "structured_task": {"disease": disease, "location": location, "start_date": year,
            "end_date": year, "target_fields": ["cases_confirmed", "cases_unspecified", "deaths",
                "hospitalizations", "date_reported", "source_url", "source_type", "evidence_quote"],
            "source_preferences": ["official_public_health_agency", "structured_database"],
            "collection_mode": "standard"},
        "live_web": {"enabled": False},
        "source_search": {"enabled": search_enabled, "mode": "fixture" if search_enabled else "disabled",
            "provider": "fixture", "fixture_path": str(search_path), "max_queries": 3,
            "max_results_per_query": 5, "max_total_results": 15, "combine_with_seed_catalog": False,
            "cache_enabled": False, "iterative": {"enabled": False}, "authority_gap_retry": {"enabled": False}},
        "content_fetch": {"fetch_search_derived_sources": fetch_enabled, "max_search_derived_sources": 1,
            "max_total_sources": 2, "min_credibility_score": 0.55, "allow_needs_review": False,
            "allowed_final_roles": ["collection", "collection_support", "context"],
            "content_fixture_map_path": str(map_path), "external_fetch": {"enabled": False}},
        "llm": {"source_planning_enabled": False, "source_critic_enabled": False,
            "structured_extraction_enabled": False, "source_credibility": {"enabled": False},
            "source_identity": {"enabled": False, "require_llm": False, "allow_deterministic_fallback": True}},
        "disease_intelligence": {"llm_enabled": False, "force_llm": False, "fallback_to_curated": True},
        "human_review": {"decisions_path": str(decision_path) if phase == "review" else None,
            "apply_decisions": phase == "review", "require_reviewer_id": True},
        "source_sets": {"source_id_allowlist_enabled": False, "workflow_source_ids": []},
        "output": {"sessionized": True, "run_output_root": str(tmp_path / "runs")},
    }
    return write_json(tmp_path / "task.jsonc", config)
