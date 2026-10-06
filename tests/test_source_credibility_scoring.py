"""Stage 6 tests for source credibility scoring and source-role assignment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


STAGE6_ENV_KEYS = [
    "COLLECTION_MODE",
    "ENABLE_LLM_SOURCE_CRITIC",
    "ENABLE_LLM_SOURCE_CREDIBILITY",
    "LLM_SOURCE_CREDIBILITY_MAX_SOURCES",
    "LLM_SOURCE_CREDIBILITY_SOURCE_ID_ALLOWLIST",
    "SOURCE_ROLE_POLICY_OVERLAY_PATH",
    "SEARCH_MODE",
    "SEARCH_PROVIDER",
    "SEARCH_FIXTURE_PATH",
    "ENABLE_LIVE_SEARCH",
    "SEARCH_MAX_QUERIES",
    "SEARCH_MAX_RESULTS_PER_QUERY",
    "SEARCH_MAX_TOTAL_RESULTS",
    "SEARCH_COMBINE_WITH_SEED_CATALOG",
]


ALLOWED_FINAL_ROLES = {
    "collection",
    "validation",
    "context",
    "collection_support",
    "search_endpoint",
    "excluded",
    "needs_human_review",
}


def _clear_stage6_env(monkeypatch) -> None:
    for key in STAGE6_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _spec(disease: str = "COVID-19", location: str = "New York", year: str = "2024") -> dict:
    return {
        "disease": disease,
        "geography": location,
        "time_window": year,
        "start_date": year,
        "end_date": year,
        "required_fields": [
            "cases_confirmed",
            "deaths",
            "date_reported",
            "source_url",
            "evidence_quote",
        ],
    }


def _disease_intelligence(disease: str = "COVID-19") -> dict:
    lowered = disease.lower()
    if "dengue" in lowered:
        return {
            "disease_standard_name": "dengue",
            "aliases": ["dengue", "dengue fever"],
            "abbreviations": ["DENV"],
            "pathogen_terms": ["dengue virus"],
            "surveillance_terms": ["arbovirus surveillance", "dengue surveillance"],
            "case_count_terms": ["cases"],
            "death_terms": ["deaths"],
        }
    return {
        "disease_standard_name": "COVID-19",
        "aliases": ["COVID-19", "coronavirus disease 2019"],
        "abbreviations": ["SARS-CoV-2"],
        "pathogen_terms": ["SARS-CoV-2"],
        "surveillance_terms": ["COVID-19 surveillance"],
        "case_count_terms": ["cases"],
        "death_terms": ["deaths"],
    }


def _entry(
    source_id: str,
    *,
    title: str,
    url: str,
    publisher: str | None,
    source_type: str,
    snippet: str | None = None,
    expected_fields: list[str] | None = None,
    role_hint: str | None = None,
    discovery_method: str = "fixture_test",
) -> dict:
    return {
        "source_id": source_id,
        "canonical_url": url,
        "title": title,
        "publisher": publisher,
        "source_type": source_type,
        "snippet": snippet,
        "status": "registered",
        "expected_fields": expected_fields or ["cases", "deaths", "date", "location"],
        "matched_terms": [],
        "role_hint": role_hint,
        "discovery_method": discovery_method,
    }


def _run_source_scoring(
    entries: list[dict],
    *,
    disease: str = "COVID-19",
    location: str = "New York",
    year: str = "2024",
) -> dict:
    from data_collection_workflow.nodes.source_screening import (
        source_critic_and_uncertainty_routing,
        source_screening,
    )

    state = {
        "collection_spec": _spec(disease, location, year),
        "disease_intelligence": _disease_intelligence(disease),
        "source_registry": entries,
        "human_review_queue": [],
        "collection_trace": [],
    }
    state.update(source_screening(state))
    state.update(source_critic_and_uncertainty_routing(state))
    return state


def _run_full_graph_from_config(config_name: str) -> dict:
    from data_collection_workflow.graph import build_graph
    from data_collection_workflow.workflow_run_config import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config_path = _PROJECT_ROOT / "configs" / "examples" / config_name
    config = load_workflow_run_config(config_path)
    env_updates = workflow_run_env_from_config(config)
    with temporary_workflow_env(env_updates):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def _search_derived_entries(result: dict) -> list[dict]:
    return [
        entry
        for entry in result.get("source_registry") or []
        if entry.get("discovery_method") in {"fixture_search_result", "live_search_result"}
    ]


def test_deterministic_credibility_scoring_exists_for_all_registry_entries(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_official_ny",
                title="COVID-19 cases and deaths in New York",
                url="https://health.ny.gov/covid/data",
                publisher="New York State Department of Health",
                source_type="official_public_health_agency",
            ),
            _entry(
                "src_unknown_blog",
                title="Unverified COVID notes",
                url="https://example-blog.invalid/covid",
                publisher=None,
                source_type="news_and_situation_report",
            ),
        ]
    )

    assessments = result.get("source_credibility_assessments") or []
    summary = result.get("source_credibility_summary") or {}
    registry = result.get("source_registry") or []

    assert len(assessments) == len(registry) == 2
    assert summary["assessed_source_count"] == 2
    for entry in registry:
        assert isinstance(entry.get("credibility_score"), float)
        assert entry.get("credibility_level") in {
            "high",
            "medium",
            "low",
            "excluded",
            "needs_review",
        }
        assert entry.get("source_role_final") in ALLOWED_FINAL_ROLES
        assert entry.get("source_role_recommendation") in ALLOWED_FINAL_ROLES
        assert entry.get("assessment_method") == "deterministic_source_credibility_v1"
        assert entry.get("final_score_explanation")


def test_official_health_department_source_scores_higher_than_low_quality_source(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_health_department",
                title="COVID-19 confirmed cases and deaths New York 2024",
                url="https://health.ny.gov/diseases/covid-19/data",
                publisher="New York State Department of Health",
                source_type="official_public_health_agency",
            ),
            _entry(
                "src_low_quality",
                title="Rumor thread about something viral",
                url="https://unknown.example/rumor",
                publisher="Unknown blog",
                source_type="news_and_situation_report",
                expected_fields=[],
            ),
        ]
    )
    by_id = {entry["source_id"]: entry for entry in result["source_registry"]}

    official = by_id["src_health_department"]
    low_quality = by_id["src_low_quality"]
    assert official["authority_score"] > low_quality["authority_score"]
    assert official["credibility_score"] > low_quality["credibility_score"]
    assert low_quality["source_role_final"] in {"excluded", "needs_human_review", "collection_support"}


def test_official_health_department_news_url_can_remain_collection(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_official_press_release",
                title="COVID-19 cases and deaths in New York 2024",
                url="https://health.ny.gov/news/covid-19-cases-2024",
                publisher="New York State Department of Health",
                source_type="official_public_health_agency",
            )
        ]
    )

    entry = result["source_registry"][0]

    assert entry["source_role_final"] == "collection"
    assert entry["human_review_recommended"] is False
    assert "secondary_news_or_media_source" not in entry["risk_flags"]


def test_covid19_search_derived_source_credibility_is_disease_aware(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_full_graph_from_config("covid19/offline_search.jsonc")
    search_entries = _search_derived_entries(result)
    summary = result.get("source_credibility_summary") or {}

    assert search_entries
    assert summary["search_derived_assessed_count"] > 0
    for entry in search_entries:
        text = " ".join(
            str(entry.get(key) or "")
            for key in ("title", "snippet", "query_used", "publisher", "domain")
        ).lower()
        assert entry["disease_relevance_score"] > 0
        assert "hantavirus pulmonary syndrome" not in text
        assert entry["source_role_final"] in ALLOWED_FINAL_ROLES


def test_dengue_search_derived_source_credibility_is_disease_aware(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_full_graph_from_config("dengue/offline_search.jsonc")
    search_entries = _search_derived_entries(result)
    summary = result.get("source_credibility_summary") or {}

    assert search_entries
    assert summary["search_derived_assessed_count"] > 0
    assert any(entry["disease_relevance_score"] > 0 for entry in search_entries)
    for entry in search_entries:
        text = " ".join(
            str(entry.get(key) or "")
            for key in ("title", "snippet", "query_used", "publisher", "domain")
        ).lower()
        assert "hantavirus pulmonary syndrome" not in text
        if entry.get("source_type") == "news_and_situation_report" or "news" in text:
            assert entry["source_role_final"] in {"collection_support", "needs_human_review"}
            assert entry["authority_score"] < 0.85


def test_context_only_source_does_not_become_collection_source(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_context",
                title="Dengue prevention and mosquito control fact sheet",
                url="https://www.cdc.gov/dengue/prevention",
                publisher="CDC",
                source_type="official_public_health_agency",
                snippet="Prevention, symptoms, treatment, and mosquito control guidance.",
                expected_fields=["disease", "case_definition"],
            )
        ],
        disease="dengue",
        location="Florida",
        year="2025",
    )
    entry = result["source_registry"][0]

    assert entry["source_role_final"] in {"context", "excluded", "needs_human_review"}
    assert entry["source_role_final"] != "collection"


def test_search_endpoint_remains_search_endpoint_or_deferred(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_pubmed_search",
                title="PubMed dengue Florida search results",
                url="https://pubmed.ncbi.nlm.nih.gov/?term=dengue+Florida+2025",
                publisher="PubMed",
                source_type="peer_reviewed_literature",
                expected_fields=[],
            )
        ],
        disease="dengue",
        location="Florida",
        year="2025",
    )
    entry = result["source_registry"][0]

    assert entry["source_role_final"] == "search_endpoint"
    assert entry["source_role_final"] != "collection"
    assert entry["ready_for_content_fetch"] is False


def test_validation_reserved_source_remains_separated_when_policy_says_so(monkeypatch):
    _clear_stage6_env(monkeypatch)
    monkeypatch.setenv("COLLECTION_MODE", "masked_validation")

    result = _run_source_scoring(
        [
            _entry(
                "src_cdc_reported_cases",
                title="CDC hantavirus reported cases",
                url="https://www.cdc.gov/hantavirus/surveillance/reporting.html",
                publisher="CDC",
                source_type="official_public_health_agency",
                expected_fields=["cases", "deaths", "date", "location"],
            )
        ],
        disease="hantavirus",
        location="New Mexico",
        year="2025",
    )
    entry = result["source_registry"][0]

    assert entry["source_role_final"] == "validation"
    assert entry["source_role"] == "validation_reserved"
    assert entry["ready_for_content_fetch"] is False
    assert "blocked_from_collection" in entry["routing_flags"]


def test_human_review_trigger_for_ambiguous_source(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_source_scoring(
        [
            _entry(
                "src_ambiguous",
                title="Dengue case report",
                url="https://example.org/dengue-case",
                publisher=None,
                source_type="news_and_situation_report",
                snippet="A possible dengue case was mentioned, but location and date are unclear.",
                expected_fields=["cases"],
            )
        ],
        disease="dengue",
        location="Florida",
        year="2025",
    )
    entry = result["source_registry"][0]
    queue = result.get("human_review_queue") or []

    assert entry["human_review_recommended"] is True
    assert entry["human_review_reason"]
    assert any(
        item.get("item_type") == "source_credibility"
        and "src_ambiguous" in (item.get("related_ids") or [])
        for item in queue
    )


def test_optional_llm_source_credibility_success_is_mocked(monkeypatch):
    _clear_stage6_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CREDIBILITY", "true")
    monkeypatch.setenv("LLM_SOURCE_CREDIBILITY_MAX_SOURCES", "1")

    llm_clients = importlib.import_module("data_collection_workflow.llm_clients")

    def mock_llm(*args, **kwargs):  # noqa: ANN002, ANN003
        return {
            "source_role_recommendation": "collection_support",
            "credibility_level": "medium",
            "risk_flags": ["llm_requests_human_review"],
            "human_review_recommended": True,
            "explanation": "Mocked LLM suggests review because source is secondary.",
            "confidence": 0.66,
        }

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_llm)

    result = _run_source_scoring(
        [
            _entry(
                "src_news",
                title="Florida reports dengue case in 2025",
                url="https://example-news.test/dengue-florida",
                publisher="Example News",
                source_type="news_and_situation_report",
                expected_fields=["cases", "date", "location"],
            )
        ],
        disease="dengue",
        location="Florida",
        year="2025",
    )
    entry = result["source_registry"][0]
    assessment = result["source_credibility_assessments"][0]

    assert entry["llm_used"] is True
    assert entry["llm_failed"] is False
    assert assessment["llm_used"] is True
    assert entry["source_role_final"] in ALLOWED_FINAL_ROLES
    assert isinstance(entry["credibility_score"], float)
    assert "llm_requests_human_review" in entry["risk_flags"]


def test_optional_llm_source_credibility_failure_falls_back(monkeypatch):
    _clear_stage6_env(monkeypatch)
    monkeypatch.setenv("ENABLE_LLM_SOURCE_CREDIBILITY", "true")

    llm_clients = importlib.import_module("data_collection_workflow.llm_clients")

    def mock_llm_failure(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("mock provider failure")

    monkeypatch.setattr(llm_clients, "run_pydantic_structured_llm", mock_llm_failure)

    result = _run_source_scoring(
        [
            _entry(
                "src_official",
                title="COVID-19 cases New York 2024",
                url="https://health.ny.gov/covid",
                publisher="New York State Department of Health",
                source_type="official_public_health_agency",
            )
        ]
    )
    entry = result["source_registry"][0]

    assert isinstance(entry["credibility_score"], float)
    assert entry["llm_used"] is False
    assert entry["llm_failed"] is True
    assert entry["llm_error_type"] == "RuntimeError"
    assert any("llm_source_credibility_failed" in warning for warning in entry["warnings"])


def test_full_graph_covid19_fixture_search_source_credibility_smoke(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_full_graph_from_config("covid19/offline_search.jsonc")
    package = result.get("final_data_package") or {}
    workflow_summaries = package.get("workflow_summaries") or {}
    search_entries = _search_derived_entries(result)

    assert package
    assert workflow_summaries.get("source_credibility_summary")
    assert result.get("source_credibility_summary", {}).get("search_derived_assessed_count") > 0
    assert search_entries
    assert all(isinstance(entry.get("credibility_score"), float) for entry in search_entries)
    assert all(entry.get("source_role_final") in ALLOWED_FINAL_ROLES for entry in search_entries)


def test_full_graph_dengue_fixture_search_source_credibility_smoke(monkeypatch):
    _clear_stage6_env(monkeypatch)

    result = _run_full_graph_from_config("dengue/offline_search.jsonc")
    package = result.get("final_data_package") or {}
    workflow_summaries = package.get("workflow_summaries") or {}
    search_entries = _search_derived_entries(result)

    assert package
    assert workflow_summaries.get("source_credibility_summary")
    assert result.get("source_credibility_summary", {}).get("search_derived_assessed_count") > 0
    assert search_entries
    assert all(isinstance(entry.get("credibility_score"), float) for entry in search_entries)
    assert all(entry.get("source_role_final") in ALLOWED_FINAL_ROLES for entry in search_entries)


def test_config_maps_optional_llm_source_credibility_controls_to_env(monkeypatch):
    _clear_stage6_env(monkeypatch)

    from data_collection_workflow.workflow_run_config import workflow_run_env_from_config

    config = {
        "workflow": {"use_fixture_documents": True},
        "live_web": {"enabled": False},
        "llm": {
            "source_planning_enabled": False,
            "source_critic_enabled": False,
            "structured_extraction_enabled": False,
            "source_credibility": {
                "enabled": True,
                "max_sources": 2,
                "source_id_allowlist": ["src_a", "src_b"],
            },
        },
        "source_sets": {"workflow_source_ids": ["src_a", "src_b", "src_c"]},
    }

    env = workflow_run_env_from_config(config)

    assert env["ENABLE_LLM_SOURCE_CREDIBILITY"] == "true"
    assert env["LLM_SOURCE_CREDIBILITY_MAX_SOURCES"] == "2"
    assert env["LLM_SOURCE_CREDIBILITY_SOURCE_ID_ALLOWLIST"] == "src_a,src_b"


def test_console_stage_payload_exposes_source_credibility_summary():
    from scripts.build_console import _stage_payload

    stages = _stage_payload(
        {
            "collection_trace": [
                {"node_name": "source_screening", "message": "screened sources"}
            ],
            "source_credibility_summary": {
                "assessed_source_count": 2,
                "role_counts": {"collection": 1, "validation": 1},
            },
            "source_credibility_assessments": [
                {
                    "source_id": "src_a",
                    "source_role_final": "collection",
                    "credibility_score": 0.91,
                }
            ],
        }
    )

    matching = [
        stage
        for stage in stages
        if "source_credibility_summary" in (stage.get("show") or {})
    ]

    assert matching
    assert "source_credibility_assessments" in matching[0]["state_writes"]
