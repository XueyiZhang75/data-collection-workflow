from __future__ import annotations

from pathlib import Path

from data_collection_workflow.graph import build_graph
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.nodes.linking_validation import record_linking


def _record(record_id: str, **overrides) -> dict:
    record = {
        "record_id": record_id,
        "disease": "COVID-19",
        "disease_standard_name": "COVID-19",
        "virus_or_syndrome": "COVID-19",
        "country": "United States of America",
        "subnational_location": "New York",
        "date_reported": "2024-06-01",
        "cases_unspecified": 100.0,
        "deaths": 2.0,
        "hospitalizations": 5.0,
        "statistical_count_type": "annual",
        "count_semantics": "annual",
        "source_id": f"src_{record_id}",
        "source_url": f"https://example.org/{record_id}",
        "source_title": "Example public health source",
        "source_type": "official_public_health_agency",
        "publisher": "Example Department of Health",
        "source_role_final": "collection",
        "credibility_score": 0.9,
        "credibility_level": "high",
        "discovery_method": "fixture_search_result",
        "search_provider": "fixture",
        "query_id": "q_fixture_001",
        "query_used": "COVID-19 cases deaths New York 2024",
        "evidence_quote": (
            "New York reported 100 COVID-19 cases, 2 deaths, and "
            "5 hospitalizations on 2024-06-01."
        ),
        "supporting_chunk_id": f"chunk_{record_id}",
        "schema_status": "valid",
        "provenance_status": "verified",
        "normalization_status": "normalized",
        "record_schema": "generic_public_health_record",
        "requires_human_review": False,
    }
    record.update(overrides)
    return record


def _link(records: list[dict]) -> dict:
    return record_linking(
        {
            "normalized_records": records,
            "human_review_queue": [],
            "collection_trace": [],
        }
    )


def _records_by_id(records: list[dict]) -> dict[str, dict]:
    return {record["record_id"]: record for record in records}


def _run_fixture_config(config_name: str) -> dict:
    from data_collection_workflow.runtime_profile import (
        load_workflow_run_config,
        temporary_workflow_env,
        workflow_initial_state_from_config,
        workflow_run_env_from_config,
    )

    config = load_workflow_run_config(Path("configs") / "examples" / config_name)
    with temporary_workflow_env(workflow_run_env_from_config(config)):
        return build_graph().invoke(workflow_initial_state_from_config(config))


def test_every_normalized_record_receives_event_cluster_fields():
    result = _link([
        _record("rec_a"),
        _record("rec_b", date_reported="2024-06-02", supporting_chunk_id="chunk_b"),
    ])

    records = result["normalized_records"]
    assert result["event_clustering_summary"]["input_normalized_record_count"] == 2
    assert result["event_clusters"]
    for record in records:
        assert record["event_cluster_id"]
        assert isinstance(record["countable"], bool)
        assert record["event_member_status"] in {
            "representative",
            "countable",
            "non_countable_duplicate",
            "related_not_merged",
            "conflicting_member",
            "singleton",
            "not_comparable",
            "invalid",
        }
        assert record["duplicate_detection_method"] == "deterministic_event_clusterer"
        assert record["duplicate_detection_reason"]


def test_exact_duplicate_same_source_and_chunk_clusters_and_counts_once():
    original = _record(
        "rec_001_original",
        source_id="src_same",
        source_url="https://example.org/same",
        supporting_chunk_id="chunk_same",
    )
    duplicate = _record(
        "rec_002_duplicate",
        source_id="src_same",
        source_url="https://example.org/same",
        supporting_chunk_id="chunk_same",
    )

    result = _link([duplicate, original])
    records = _records_by_id(result["normalized_records"])

    assert records["rec_001_original"]["event_cluster_id"] == records["rec_002_duplicate"]["event_cluster_id"]
    assert records["rec_001_original"]["countable"] is True
    assert records["rec_002_duplicate"]["countable"] is False
    assert records["rec_002_duplicate"]["duplicate_of_record_id"] == "rec_001_original"
    assert records["rec_001_original"]["event_member_status"] == "representative"
    assert records["rec_002_duplicate"]["event_member_status"] == "non_countable_duplicate"

    cluster = result["event_clusters"][0]
    assert cluster["cluster_status"] == "duplicate_cluster"
    assert cluster["representative_record_id"] == "rec_001_original"
    assert cluster["countable_record_ids"] == ["rec_001_original"]
    assert cluster["non_countable_duplicate_record_ids"] == ["rec_002_duplicate"]
    assert result["duplicate_detection_summary"]["duplicate_cluster_count"] == 1
    assert result["duplicate_detection_summary"]["non_countable_duplicate_count"] == 1


def test_same_event_official_and_news_support_selects_official_representative():
    official = _record(
        "rec_official",
        source_type="official_public_health_agency",
        source_role_final="collection",
        credibility_score=0.95,
        publisher="State Department of Health",
    )
    news = _record(
        "rec_news",
        source_id="src_news",
        source_url="https://news.example.org/covid",
        source_type="news_and_situation_report",
        source_role_final="collection_support",
        credibility_score=0.55,
        publisher="Example News",
        evidence_quote="News coverage repeated the official count of 100 COVID-19 cases and 2 deaths.",
    )

    result = _link([news, official])
    records = _records_by_id(result["normalized_records"])
    cluster = result["event_clusters"][0]

    assert cluster["cluster_status"] == "duplicate_cluster"
    assert cluster["representative_record_id"] == "rec_official"
    assert records["rec_official"]["countable"] is True
    assert records["rec_news"]["countable"] is False
    assert records["rec_news"]["duplicate_of_record_id"] == "rec_official"
    assert "representative_selection_reason" in cluster
    assert "higher_source_priority" in cluster["representative_selection_reason"]


def test_different_dates_do_not_merge():
    result = _link([
        _record("rec_day_1", date_reported="2024-06-01"),
        _record("rec_day_2", date_reported="2024-07-01"),
    ])
    records = result["normalized_records"]

    assert len({record["event_cluster_id"] for record in records}) == 2
    assert all(record["countable"] is True for record in records)
    assert result["duplicate_detection_summary"]["duplicate_cluster_count"] == 0


def test_different_locations_do_not_merge():
    result = _link([
        _record("rec_ny", subnational_location="New York"),
        _record("rec_fl", subnational_location="Florida"),
    ])
    records = result["normalized_records"]

    assert len({record["event_cluster_id"] for record in records}) == 2
    assert all(record["countable"] is True for record in records)
    assert result["duplicate_detection_summary"]["duplicate_cluster_count"] == 0


def test_different_diseases_do_not_merge():
    result = _link([
        _record("rec_covid", disease="COVID-19", disease_standard_name="COVID-19"),
        _record(
            "rec_dengue",
            disease="Dengue",
            disease_standard_name="Dengue",
            virus_or_syndrome="Dengue",
        ),
    ])
    records = result["normalized_records"]

    assert len({record["event_cluster_id"] for record in records}) == 2
    assert all(record["countable"] is True for record in records)


def test_cumulative_vs_newly_reported_is_related_not_duplicate_and_reviewed():
    cumulative = _record(
        "rec_cumulative",
        statistical_count_type="cumulative",
        count_semantics="cumulative",
        cases_unspecified=1000,
    )
    newly = _record(
        "rec_new",
        statistical_count_type="newly_reported",
        count_semantics="newly_reported",
        cases_unspecified=100,
    )

    result = _link([cumulative, newly])
    records = _records_by_id(result["normalized_records"])

    assert records["rec_cumulative"]["event_cluster_id"] != records["rec_new"]["event_cluster_id"]
    assert records["rec_cumulative"]["countable"] is True
    assert records["rec_new"]["countable"] is True
    assert records["rec_cumulative"]["event_member_status"] == "related_not_merged"
    assert records["rec_new"]["event_member_status"] == "related_not_merged"
    assert result["event_clustering_summary"]["related_cluster_count"] >= 2
    assert any(
        item["item_type"] == "duplicate_event_clustering"
        and "count_semantics" in item["reason"]
        for item in result["human_review_queue"]
    )


def test_annual_summary_vs_single_update_is_not_merged():
    annual = _record(
        "rec_annual",
        date_reported="2025",
        reporting_period="2025",
        statistical_count_type="annual",
        count_semantics="annual",
        cases_unspecified=500,
    )
    weekly = _record(
        "rec_weekly",
        date_reported="2025-08-01",
        statistical_count_type="weekly",
        count_semantics="weekly",
        cases_unspecified=12,
    )

    result = _link([annual, weekly])
    records = _records_by_id(result["normalized_records"])

    assert records["rec_annual"]["event_cluster_id"] != records["rec_weekly"]["event_cluster_id"]
    assert records["rec_annual"]["countable"] is True
    assert records["rec_weekly"]["countable"] is True
    assert result["duplicate_detection_summary"]["non_countable_duplicate_count"] == 0


def test_conflicting_counts_same_event_route_to_human_review():
    official = _record("rec_conflict_official", cases_unspecified=100, deaths=2)
    secondary = _record(
        "rec_conflict_secondary",
        source_id="src_secondary",
        source_url="https://example.org/secondary",
        source_type="news_and_situation_report",
        credibility_score=0.6,
        cases_unspecified=130,
        deaths=2,
    )

    result = _link([official, secondary])
    cluster = result["event_clusters"][0]
    records = _records_by_id(result["normalized_records"])

    assert cluster["cluster_status"] == "conflict_needs_review"
    assert cluster["needs_human_review"] is True
    assert records["rec_conflict_official"]["duplicate_review_required"] is True
    assert records["rec_conflict_secondary"]["duplicate_review_required"] is True
    assert any(
        item["item_type"] == "duplicate_event_clustering"
        and cluster["event_cluster_id"] in item["related_ids"]
        for item in result["human_review_queue"]
    )


def test_human_review_queue_receives_uncertain_duplicate_item():
    one = _record("rec_uncertain_1", statistical_count_type=None, count_semantics=None)
    two = _record(
        "rec_uncertain_2",
        source_id="src_uncertain_2",
        source_url="https://example.org/uncertain-2",
        statistical_count_type=None,
        count_semantics=None,
    )

    result = _link([one, two])

    assert any(
        item["item_type"] == "duplicate_event_clustering"
        and item.get("review_packet", {}).get("event_cluster_id")
        and item.get("review_packet", {}).get("member_record_ids")
        for item in result["human_review_queue"]
    )


def test_final_package_exports_clustering_artifacts():
    linked = _link([
        _record("rec_pkg_a", source_id="src_pkg", supporting_chunk_id="chunk_pkg"),
        _record("rec_pkg_b", source_id="src_pkg", supporting_chunk_id="chunk_pkg"),
    ])
    result = final_data_package_builder(
        {
            "normalized_records": linked["normalized_records"],
            "source_registry": [],
            "linked_events": linked["linked_events"],
            "event_clusters": linked["event_clusters"],
            "duplicate_clusters": linked["duplicate_clusters"],
            "conflicts": [],
            "human_review_queue": linked["human_review_queue"],
            "collection_trace": linked["collection_trace"],
            "event_clustering_summary": linked["event_clustering_summary"],
            "duplicate_detection_summary": linked["duplicate_detection_summary"],
        }
    )
    package = result["final_data_package"]

    assert package["event_clusters"]
    assert package["duplicate_clusters"]
    assert package["workflow_summaries"]["event_clustering_summary"]
    assert package["workflow_summaries"]["duplicate_detection_summary"]
    assert package["final_dataset"][0]["event_cluster_id"]
    assert "countable" in package["final_dataset"][0]


def test_full_graph_covid19_fixture_extraction_clustering_smoke():
    result = _run_fixture_config("covid19/offline_collection.jsonc")
    records = result["normalized_records"]
    package = result["final_data_package"]

    assert records
    assert all(record["disease"] == "COVID-19" for record in records)
    assert all(record.get("event_cluster_id") for record in records)
    assert all("countable" in record for record in records)
    assert result["event_clustering_summary"]["input_normalized_record_count"] == len(records)
    assert package["event_clusters"]
    assert package["workflow_summaries"]["event_clustering_summary"]


def test_full_graph_dengue_fixture_extraction_clustering_smoke():
    result = _run_fixture_config("dengue/offline_collection.jsonc")
    records = result["normalized_records"]
    package = result["final_data_package"]

    assert records
    assert all(record["disease"] == "Dengue" for record in records)
    assert all(record.get("event_cluster_id") for record in records)
    assert all("countable" in record for record in records)
    assert result["event_clustering_summary"]["input_normalized_record_count"] == len(records)
    assert package["event_clusters"]
    assert package["workflow_summaries"]["duplicate_detection_summary"]


def test_hantavirus_new_mexico_compatibility_clustering():
    result = _link([
        _record(
            "rec_hps_nm",
            disease="Hantavirus disease",
            disease_standard_name="Hantavirus disease",
            virus_or_syndrome="HPS",
            pathogen_or_syndrome="Sin Nombre virus",
            subnational_location="New Mexico",
            date_reported="2024",
            cases_unspecified=1,
            deaths=None,
            source_id="src_nmdoh_hps_2024_first_case",
            source_url="https://www.nmhealth.org/news/safety/2024/2?view=2065",
            source_title="New Mexico reports first hantavirus case of 2024",
            publisher="New Mexico Department of Health",
            legacy_record_type="HantavirusRecord",
        )
    ])

    record = result["normalized_records"][0]
    assert record["disease"] == "Hantavirus disease"
    assert record["linked_event_id"] == record["event_cluster_id"]
    assert record["legacy_record_type"] == "HantavirusRecord"
    assert result["linked_events"]
    assert result["event_clusters"]
