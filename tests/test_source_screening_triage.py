from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def test_source_triage_results_are_task_specific_not_hantavirus_templates():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "California",
            "start_date": "2024-10-01",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "California",
            "start_date": "2024-10-01",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_cdph_weekly_flu",
                "canonical_url": "https://www.cdph.ca.gov/Programs/CID/DCDC/Pages/Immunization/Influenza.aspx",
                "title": "California influenza surveillance weekly report",
                "publisher": "California Department of Public Health",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "expected_fields": ["tests_positive", "hospitalizations", "date", "location"],
                "matched_terms": ["flu", "influenza", "California"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_flu_california",
                "query_used": "California influenza surveillance weekly report",
                "role_hint": "collection",
                "priority": 1,
                "search_rank": 1,
            }
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = result["source_triage_results"][0]
    reason_text = " ".join(str(value or "") for value in triage.values()).lower()

    assert triage["source_url"].startswith("https://www.cdph.ca.gov/")
    assert triage["source_title"] == "California influenza surveillance weekly report"
    assert triage["task_disease"] == "FLU"
    assert triage["task_location"] == "California"
    assert "hantavirus" not in reason_text
    assert "influenza" in reason_text or "flu" in reason_text


def test_direct_source_triage_downgrades_wrong_year_or_week_sources():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "United States",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_cdc_week_40_2024",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-40.html",
                "title": "CDC FluView Week 40, 2024",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "expected_fields": ["tests_positive", "hospitalizations"],
                "matched_terms": ["flu", "influenza", "week 40", "2024"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_week_40",
                "query_used": "CDC FluView 2024 week 40 influenza",
                "search_rank": 1,
            },
            {
                "source_id": "src_cdc_week_40_2025",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2025-week-40.html",
                "title": "CDC FluView Week 40, 2025",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "expected_fields": ["tests_positive", "hospitalizations"],
                "matched_terms": ["flu", "influenza", "week 40", "2025"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_week_40",
                "query_used": "CDC FluView 2024 week 40 influenza",
                "search_rank": 2,
            },
            {
                "source_id": "src_cdc_week_06_2026",
                "canonical_url": "https://www.cdc.gov/fluview/surveillance/2026-week-06.html",
                "title": "CDC FluView Week 06, 2026",
                "publisher": "CDC",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "expected_fields": ["tests_positive", "hospitalizations"],
                "matched_terms": ["flu", "influenza", "week 6", "2026"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_week_40",
                "query_used": "CDC FluView 2024 week 40 influenza",
                "search_rank": 3,
            },
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = {row["source_id"]: row for row in result["source_triage_results"]}
    registry = {row["source_id"]: row for row in result["source_registry"]}

    assert triage["src_cdc_week_40_2024"]["triage_role"] == "verified_target_collection"
    assert triage["src_cdc_week_40_2024"]["target_verification_status"] == "verified_target"

    for source_id in ("src_cdc_week_40_2025", "src_cdc_week_06_2026"):
        assert triage[source_id]["triage_role"] in {"context_only", "excluded"}
        assert triage[source_id]["target_verification_status"] == "temporal_mismatch"
        assert triage[source_id]["date_fit"] == "mismatch"
        assert registry[source_id]["source_role"] != "data_source"
        assert registry[source_id]["screening_decision"] != "include"


def test_direct_source_triage_persists_task_candidate_role_to_registry():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_vdh_flu_landing",
                "canonical_url": "https://www.vdh.virginia.gov/epidemiology/influenza-flu-in-virginia",
                "title": "Influenza (Flu) in Virginia - Epidemiology",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "expected_fields": ["tests_positive", "date", "location"],
                "matched_terms": ["flu", "influenza", "Virginia"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_vdh_fallback",
                "query_used": "Virginia influenza surveillance week 40 2024",
                "role_hint": "collection",
                "search_rank": 2,
            }
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = result["source_triage_results"][0]
    registry = result["source_registry"][0]

    assert triage["triage_role"] == "task_record_collection_candidate"
    assert triage["target_verification_status"] == "unverified_candidate"
    assert triage["date_fit"] == "candidate"
    assert registry["triage_role"] == "task_record_collection_candidate"
    assert registry["target_fit_status"] == "task_record_collection_candidate"
    assert registry["target_verification_status"] == "candidate_task_record_source"
    assert registry["date_fit"] == "candidate"


def test_must_fetch_does_not_verify_an_unrelated_disease_source():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "hantavirus",
            "geography": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_h5n6_must_fetch",
                "canonical_url": "https://example.gov/h5n6-update",
                "title": "H5N6 avian influenza outbreak update",
                "publisher": "Example Ministry of Health",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "must_fetch": True,
                "coverage_requirement_ids": ["official_update"],
                "source_disease_relevance_status": "unrelated_disease",
                "source_incompatible_disease_terms_found": ["h5n6", "avian influenza"],
            }
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = result["source_triage_results"][0]
    registry = result["source_registry"][0]

    assert triage["target_verification_status"] == "unrelated_disease"
    assert triage["triage_role"] in {"context_only", "excluded"}
    assert triage["triage_decision"] != "include"
    assert registry["target_fit_status"] != "verified_target_collection"


def test_must_fetch_preserves_fetch_obligation_without_promoting_generic_source():
    from data_collection_workflow.source_product_profile import profile_source_product

    state = {
        "structured_task": {
            "disease": "influenza",
            "location": "Example State",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "influenza",
            "geography": "Example State",
            "start_date": "2025-01-01",
            "end_date": "2025-01-31",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_required_portal",
                "canonical_url": "https://health.example.gov/respiratory-data",
                "title": "Respiratory disease data portal",
                "publisher": "Example State Department of Health",
                "source_type": "official_public_health_agency",
                "status": "registered",
                "must_fetch": True,
                "coverage_requirement_ids": ["required_portal"],
            }
        ],
        "collection_trace": [],
    }

    source = state["source_registry"][0]
    profile = profile_source_product(source, state)

    assert source["must_fetch"] is True
    assert profile["data_product_type"] == "unrelated_or_other"
    assert profile["task_specificity"] == "unrelated"


def test_direct_source_triage_routes_news_and_social_candidates_to_review():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "India",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "India",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_ndtv_flu",
                "canonical_url": "https://www.ndtv.com/india-news/no-abnormal-rise-in-seasonal-flu-cases",
                "title": "No abnormal rise in seasonal flu cases in India, Health Ministry says",
                "publisher": "NDTV",
                "source_type": "news_media",
                "status": "registered",
                "matched_terms": ["flu", "India", "2024", "cases"],
                "expected_fields": ["case_count", "date", "location"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_flu_india",
                "query_used": "India seasonal flu 2024 cases official data",
                "role_hint": "collection",
                "search_rank": 1,
            },
            {
                "source_id": "src_youtube_flu",
                "canonical_url": "https://www.youtube.com/watch?v=YE-dJl4lhKw",
                "title": "Seasonal influenza situation in India 2024",
                "publisher": "YouTube",
                "source_type": "social_media",
                "status": "registered",
                "matched_terms": ["flu", "India", "2024"],
                "expected_fields": ["case_count", "date", "location"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_flu_india",
                "query_used": "India seasonal flu 2024 cases official data",
                "role_hint": "collection",
                "search_rank": 2,
            },
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = {row["source_id"]: row for row in result["source_triage_results"]}
    registry = {row["source_id"]: row for row in result["source_registry"]}

    for source_id in ("src_ndtv_flu", "src_youtube_flu"):
        assert triage[source_id]["triage_role"] != "task_record_collection_candidate"
        assert registry[source_id]["target_fit_status"] != "task_record_collection_candidate"
        assert registry[source_id]["source_role_final"] == "needs_human_review"
        assert registry[source_id]["requires_human_review"] is True
        assert "source_trust_requires_human_review" in registry[source_id]["screening_flags"]


def test_source_triage_results_preserve_identity_and_review_metadata():
    from data_collection_workflow.nodes.source_screening import source_screening

    state = {
        "structured_task": {
            "disease": "Tuberculosis",
            "location": "United Kingdom",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        "collection_spec": {
            "disease": "Tuberculosis",
            "geography": "United Kingdom",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "collection_mode": "direct_collection",
        },
        "source_registry": [
            {
                "source_id": "src_ukhsa_tb_2025",
                "canonical_url": "https://www.gov.uk/government/statistics/tuberculosis-in-england-2025",
                "title": "Tuberculosis in England: annual report 2025",
                "publisher": "UK Health Security Agency",
                "actual_publisher": "UK Health Security Agency",
                "source_type": "official_public_health_agency",
                "source_type_final": "official_public_health_agency",
                "source_role_final": "collection",
                "status": "registered",
                "expected_fields": ["case_count", "incidence_rate", "date", "location"],
                "matched_terms": ["tuberculosis", "United Kingdom", "2025"],
                "discovery_method": "fixture_search_result",
                "query_id": "q_tb_uk_2025",
                "query_used": "United Kingdom tuberculosis annual report 2025",
                "role_hint": "collection",
                "search_rank": 1,
                "requires_human_review": False,
            }
        ],
        "collection_trace": [],
    }

    result = source_screening(state)
    triage = result["source_triage_results"][0]

    assert triage["source_type_final"] == "official_public_health_agency"
    assert triage["source_type"] == "official_public_health_agency"
    assert triage["actual_publisher"] == "UK Health Security Agency"
    assert triage["publisher"] == "UK Health Security Agency"
    assert triage["source_role_final"] == "collection"
    assert triage["requires_human_review"] is False
    assert triage["human_review_reason"] in (None, "")


def test_evidence_literature_articles_are_fetched_and_search_endpoints_stay_deferred(monkeypatch):
    import importlib
    from data_collection_workflow.nodes.source_screening import source_screening, source_critic_and_uncertainty_routing
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_CRITIC', 'false')
    monkeypatch.setenv('ENABLE_LLM_SOURCE_IDENTITY', 'false')
    discovery = importlib.import_module('data_collection_workflow.nodes.source_discovery')
    urls = {
        'article': ('https://pubmed.ncbi.nlm.nih.gov/12345678/', 'PubMed'),
        'pmc_article': ('https://europepmc.org/article/MED/12345678', 'Europe PMC'),
        'endpoint': ('https://pubmed.ncbi.nlm.nih.gov/?term=measles', 'PubMed'),
        'pmc_endpoint': ('https://europepmc.org/search?query=measles', 'Europe PMC'),
    }
    state = {'structured_task': {'disease': 'measles', 'location': 'Canada',
                                'collection_mode': 'direct_collection'},
             'source_candidates': [
        {'source_id': key, 'title': 'Measles Canada cases' if 'article' in key else 'Measles search results',
         'url': url, 'publisher': publisher, 'source_type': 'peer_reviewed_literature',
         'snippet': 'Human measles cases in Canada 2025', 'discovery_method': 'live_search_result'}
        for key, (url, publisher) in urls.items()
    ]}
    for node in (discovery.source_dedup_and_registry, source_screening, source_critic_and_uncertainty_routing):
        state.update(node(state))
    registry = {row['source_id']: row for row in state['source_registry']}
    assert registry['article']['ready_for_content_fetch'] is True
    assert registry['pmc_article']['ready_for_content_fetch'] is True
    assert registry['endpoint']['ready_for_content_fetch'] is False
    assert registry['pmc_endpoint']['ready_for_content_fetch'] is False

    from data_collection_workflow.nodes.source_screening import _select_llm_source_critic_candidates
    selected, _, _ = _select_llm_source_critic_candidates(
        state['source_registry'], allowlist=None, max_sources=None)
    assert selected == {'article', 'pmc_article'}
