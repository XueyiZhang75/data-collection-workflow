from __future__ import annotations

from data_collection_workflow.source_coverage import (
    annotate_source_coverage,
    build_authority_source_coverage_summary,
    build_source_coverage_audit,
    build_source_coverage_requirements,
    build_task_evidence_contract,
)


def test_virginia_flu_date_range_requires_week_40_and_41_reports():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "FLU",
                "location": "VIRGINIA",
                "start_date": "2024-10-01",
                "end_date": "2024-10-10",
            },
            "collection_spec": {
                "disease": "FLU",
                "geography": "VIRGINIA",
                "start_date": "2024-10-01",
                "end_date": "2024-10-10",
            },
        }
    )

    ids = {item["requirement_id"] for item in requirements}

    assert "virginia_influenza_official_week_40_2024" in ids
    assert "virginia_influenza_official_week_41_2024" in ids
    assert all(item["source_type"] == "official_weekly_surveillance_report" for item in requirements)
    assert all("vdh.virginia.gov" in item["official_domains"] for item in requirements)


def test_new_york_flu_date_range_requires_target_week_official_reports():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "FLU",
                "location": "NEW YORK",
                "start_date": "2024-10-01",
                "end_date": "2024-11-01",
            },
            "collection_spec": {
                "disease": "FLU",
                "geography": "NEW YORK",
                "start_date": "2024-10-01",
                "end_date": "2024-11-01",
            },
        }
    )

    ids = {item["requirement_id"] for item in requirements}

    assert "new_york_influenza_official_week_40_2024" in ids
    assert "new_york_influenza_official_week_43_2024" in ids
    assert any("health.ny.gov" in item["official_domains"] for item in requirements)
    week_40 = next(
        item
        for item in requirements
        if item["requirement_id"] == "new_york_influenza_official_week_40_2024"
    )
    assert any(
        url.endswith("/2024-10-05_flu_report.pdf")
        for url in week_40["official_candidate_urls"]
    )


def test_united_states_flu_date_range_requires_cdc_fluview_target_week():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "FLU",
                "location": "UNITED STATES",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
            },
            "collection_spec": {
                "disease": "FLU",
                "geography": "UNITED STATES",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
            },
        }
    )

    ids = {item["requirement_id"] for item in requirements}

    assert "united_states_influenza_official_week_40_2024" in ids
    week_40 = next(
        item
        for item in requirements
        if item["requirement_id"] == "united_states_influenza_official_week_40_2024"
    )
    assert "cdc.gov" in week_40["official_domains"]
    assert week_40["agency"] == "Centers for Disease Control and Prevention"
    assert week_40["official_candidate_urls"] == [
        "https://www.cdc.gov/fluview/surveillance/2024-week-40.html"
    ]


def test_generic_annual_national_task_builds_source_coverage_requirement():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
                "target_fields": [
                    "incidence rate",
                    "case count",
                    "death count",
                    "treatment coverage",
                ],
            },
            "collection_spec": {
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2023-01-01",
                "end_date": "2024-12-31",
            },
        }
    )

    ids = {item["requirement_id"] for item in requirements}

    assert "india_tuberculosis_annual_2023" in ids
    assert "india_tuberculosis_annual_2024" in ids
    for requirement in requirements:
        assert requirement["source_type"] == "task_relevant_public_health_evidence"
        assert requirement["period_basis"] == "annual"
        assert requirement["location"] == "India"
        assert requirement["geography"] == "India"
        assert requirement["disease"] == "tuberculosis"
        assert requirement["period_start"] in {"2023-01-01", "2024-01-01"}
        assert requirement["period_end"] in {"2023-12-31", "2024-12-31"}
        assert requirement["reporting_period_start"] == requirement["period_start"]
        assert requirement["reporting_period_end"] == requirement["period_end"]
        assert "incidence_rate" in requirement["accepted_metric_categories"]
        assert "case_count" in requirement["accepted_metric_categories"]
        assert "death_count" in requirement["accepted_metric_categories"]


def test_source_coverage_requirement_projects_task_evidence_contract_fields():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "Tuberculosis",
                "location": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
            "collection_spec": {
                "disease": "Tuberculosis",
                "geography": "India",
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
            },
        }
    )

    requirement = requirements[0]

    assert requirement["time_granularity"] == "annual"
    assert requirement["accepted_metric_families"] == requirement[
        "accepted_metric_categories"
    ]
    assert "source_provenance_verified" in requirement["strict_final_conditions"]
    assert "wrong_period_or_broader_than_task" in requirement[
        "best_available_conditions"
    ]
    assert "source_trust_boundary" in requirement["human_review_conditions"]


def test_calendar_year_exclusive_end_date_builds_annual_requirement():
    state = {
        "structured_task": {
            "disease": "Measles",
            "location": "United States",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
        },
        "collection_spec": {
            "disease": "Measles",
            "geography": "United States",
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
        },
    }

    requirements = build_source_coverage_requirements(state)
    contract = build_task_evidence_contract(state)
    ids = {item["requirement_id"] for item in requirements}

    assert ids == {"united_states_measles_annual_2023"}
    requirement = requirements[0]
    assert requirement["period_basis"] == "annual"
    assert requirement["reporting_period_start"] == "2023-01-01"
    assert requirement["reporting_period_end"] == "2023-12-31"
    assert contract["time_granularity"] == "annual"
    assert contract["requirements"] == requirements
    assert "case_count" in contract["accepted_metric_families"]


def test_country_adjective_location_is_canonicalized_for_generic_contract():
    state = {
        "structured_task": {
            "disease": "Measles",
            "location": "German",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
        },
        "collection_spec": {
            "disease": "Measles",
            "geography": "German",
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
        },
    }

    requirements = build_source_coverage_requirements(state)
    contract = build_task_evidence_contract(state)

    assert requirements[0]["location"] == "Germany"
    assert requirements[0]["geography"] == "Germany"
    assert requirements[0]["requirement_id"] == "germany_measles_annual_2025"
    assert contract["location"] == "Germany"


def test_generic_subnational_short_window_builds_task_window_requirement():
    requirements = build_source_coverage_requirements(
        {
            "structured_task": {
                "disease": "FLU",
                "location": "California",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
            },
            "collection_spec": {
                "disease": "FLU",
                "geography": "California",
                "start_date": "2024-09-29",
                "end_date": "2024-10-05",
            },
        }
    )

    ids = {item["requirement_id"] for item in requirements}

    assert "california_flu_task_window_2024_09_29_2024_10_05" in ids
    assert "california_flu_annual_2024" not in ids
    requirement = next(
        item
        for item in requirements
        if item["requirement_id"] == "california_flu_task_window_2024_09_29_2024_10_05"
    )
    assert requirement["period_basis"] == "task_window"
    assert requirement["reporting_period_start"] == "2024-09-29"
    assert requirement["reporting_period_end"] == "2024-10-05"
    assert requirement["location"] == "California"
    assert requirement["source_type"] == "task_relevant_public_health_evidence"
    assert "lab_positive_count" in requirement["accepted_metric_categories"]


def test_generic_task_record_candidate_can_satisfy_task_window_coverage():
    registry = [
        {
            "source_id": "src_cdph_dashboard",
            "canonical_url": "https://data.chhs.ca.gov/dataset/influenza-surveillance",
            "title": "California Influenza Surveillance Open Data",
            "publisher": "California Department of Public Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "target_fit_status": "task_record_collection_candidate",
            "triage_role": "task_record_collection_candidate",
            "disease_fit": "match",
            "geography_fit": "match",
            "date_fit": "candidate",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        }
    ]
    documents = [
        {
            "source_id": "src_cdph_dashboard",
            "fetch_status": "fetched",
            "http_status_code": 200,
            "parse_status": "parsed_html",
            "quality_status": "usable",
            "usable_for_task_collection": True,
            "clean_text": (
                "California influenza surveillance data for the week ending "
                "October 5, 2024 includes laboratory positives and ILI metrics."
            ),
        }
    ]
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "California",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "California",
            "start_date": "2024-09-29",
            "end_date": "2024-10-05",
        },
    }

    updated, requirements, audit = annotate_source_coverage(
        registry,
        state,
        documents=documents,
    )

    assert requirements
    source = updated[0]
    assert source["coverage_requirement_ids"] == [
        "california_flu_task_window_2024_09_29_2024_10_05"
    ]
    assert source["ready_for_content_fetch"] is True
    assert audit["coverage_status"] == "parsed_no_records"
    assert audit["coverage_completeness_status"] == "no_target_coverage"
    assert audit["complete_requirement_count"] == 0
    assert audit["missing_requirement_ids"] == [
        "california_flu_task_window_2024_09_29_2024_10_05"
    ]


def test_fetch_failed_error_page_does_not_satisfy_source_coverage():
    requirement = {
        "requirement_id": "united_states_influenza_official_week_1_2025",
        "disease": "influenza",
        "location": "United States",
        "official_domains": ["cdc.gov"],
    }
    registry = [
        {
            "source_id": "src_cdc_week_1_generated",
            "coverage_requirement_ids": [requirement["requirement_id"]],
            "discovery_method": "official_coverage_requirement",
            "must_fetch": True,
        }
    ]
    documents = [
        {
            "source_id": "src_cdc_week_1_generated",
            "fetch_status": "fetch_failed",
            "http_status_code": 404,
            "parse_status": "parsed_html",
            "title": "Page Not Found | CDC",
            "clean_text": "Page Not Found. The page you requested was not found.",
            "quality_status": "unusable",
        }
    ]

    audit = build_source_coverage_audit([requirement], registry, documents)

    row = audit["requirements"][0]
    assert row["discovered"] is True
    assert row["fetched"] is False
    assert row["parsed"] is False
    assert row["fetch_failed"] is True
    assert row["unusable"] is True
    assert audit["coverage_status"] == "target_official_source_fetch_failed"


def test_partial_multi_requirement_coverage_reports_missing_requirement_ids():
    requirements = [
        {
            "requirement_id": "virginia_influenza_official_week_41_2024",
            "disease": "influenza",
            "location": "Virginia",
            "official_domains": ["vdh.virginia.gov"],
        },
        {
            "requirement_id": "virginia_influenza_official_week_42_2024",
            "disease": "influenza",
            "location": "Virginia",
            "official_domains": ["vdh.virginia.gov"],
        },
    ]
    registry = [
        {
            "source_id": "src_vdh_week41_bad",
            "coverage_requirement_ids": [requirements[0]["requirement_id"]],
            "must_fetch": True,
        },
        {
            "source_id": "src_vdh_week42_good",
            "coverage_requirement_ids": [requirements[1]["requirement_id"]],
            "must_fetch": True,
        },
    ]
    documents = [
        {
            "source_id": "src_vdh_week41_bad",
            "fetch_status": "fetched",
            "http_status_code": 200,
            "parse_status": "parsed_html",
            "quality_status": "unusable",
            "title": "Page not found",
            "clean_text": "Page not found",
        },
        {
            "source_id": "src_vdh_week42_good",
            "fetch_status": "fetched",
            "http_status_code": 200,
            "parse_status": "parsed_pdf",
            "quality_status": "usable",
            "title": "VDH Weekly Respiratory Disease Surveillance Report",
            "clean_text": "Influenza surveillance metrics for Week 42.",
        },
    ]

    audit = build_source_coverage_audit(requirements, registry, documents)

    assert audit["coverage_status"] == "parsed_no_records"
    assert audit["coverage_completeness_status"] == "no_target_coverage"
    assert audit["complete_requirement_count"] == 0
    assert audit["partial_requirement_count"] == 2
    assert audit["missing_requirement_ids"] == [
        "virginia_influenza_official_week_41_2024",
        "virginia_influenza_official_week_42_2024",
    ]
    assert audit["requirements"][0]["missing_reason"] == "target_alias_error_page"


def test_parsed_context_document_does_not_complete_target_coverage():
    requirement = {
        "requirement_id": "virginia_influenza_official_week_40_2024",
        "disease": "influenza",
        "location": "Virginia",
        "official_domains": ["vdh.virginia.gov"],
    }
    registry = [
        {
            "source_id": "src_vdh_end_of_season",
            "coverage_requirement_ids": [requirement["requirement_id"]],
            "target_fit_status": "best_available_context_candidate",
            "triage_role": "best_available_context_candidate",
            "source_role_final": "context",
        }
    ]
    documents = [
        {
            "source_id": "src_vdh_end_of_season",
            "fetch_status": "fetched",
            "http_status_code": 200,
            "parse_status": "parsed_pdf",
            "quality_status": "usable",
            "usable_for_task_collection": False,
            "title": "Virginia 2023-2024 Influenza End of Season Report",
            "clean_text": "Virginia 2023-2024 influenza season summary.",
        }
    ]

    audit = build_source_coverage_audit([requirement], registry, documents)

    row = audit["requirements"][0]
    assert row["fetched"] is True
    assert row["parsed"] is False
    assert row["missing_reason"] == "no_task_collection_document"
    assert audit["coverage_status"] == "target_official_source_unusable"
    assert audit["coverage_completeness_status"] == "no_target_coverage"


def test_wrong_period_document_with_requirement_id_does_not_complete_coverage():
    requirement = {
        "requirement_id": "virginia_influenza_official_week_40_2024",
        "disease": "influenza",
        "location": "Virginia",
        "official_domains": ["vdh.virginia.gov"],
        "week": 40,
        "year": 2024,
        "reporting_period_start": "2024-09-29",
        "reporting_period_end": "2024-10-05",
    }
    registry = [
        {
            "source_id": "src_vdh_week_42_wrong_period",
            "canonical_url": (
                "https://www.vdh.virginia.gov/content/uploads/sites/3/2024/10/"
                "2024-25_Weekly-RDS-Report_Week-42.pdf"
            ),
            "title": "VDH Weekly Respiratory Disease Surveillance Report Week 42",
            "coverage_requirement_ids": [requirement["requirement_id"]],
            "source_role_final": "collection",
            "target_fit_status": "search_verified_target_collection",
        }
    ]
    documents = [
        {
            "source_id": "src_vdh_week_42_wrong_period",
            "fetch_status": "fetched",
            "http_status_code": 200,
            "parse_status": "parsed_pdf",
            "quality_status": "usable",
            "title": "Weekly Respiratory Disease Surveillance Report Week 42",
            "clean_text": "VDH respiratory disease surveillance report for Week 42, 2024.",
        }
    ]

    audit = build_source_coverage_audit([requirement], registry, documents)

    row = audit["requirements"][0]
    assert row["fetched"] is True
    assert row["parsed"] is False
    assert row["period_mismatch"] is True
    assert row["missing_reason"] == "source_period_mismatch"
    assert audit["coverage_status"] == "target_source_period_mismatch"
    assert audit["coverage_completeness_status"] == "no_target_coverage"


def test_wrong_year_virginia_weekly_report_does_not_satisfy_target_requirement():
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
        },
    }
    registry = [
        {
            "source_id": "src_vdh_week_41_2023",
            "canonical_url": (
                "https://www.vdh.virginia.gov/content/uploads/sites/3/2023/11/"
                "Weekly-RDS-Report_Week-41.pdf"
            ),
            "title": "Weekly Respiratory Disease Surveillance Report Week 41",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        },
        {
            "source_id": "src_vdh_week_41_2025",
            "canonical_url": (
                "https://www.vdh.virginia.gov/content/uploads/sites/3/"
                "2025-26_Weekly-RDS-Report_Week-41.pdf"
            ),
            "title": "Weekly Respiratory Disease Surveillance Report Week 41",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        },
    ]

    annotated, requirements, audit = annotate_source_coverage(registry, state)

    assert any(
        item["requirement_id"] == "virginia_influenza_official_week_41_2024"
        for item in requirements
    )
    assert all(not row.get("must_fetch") for row in annotated)
    assert audit["discovered_requirement_count"] == 0
    assert audit["coverage_status"] == "target_official_source_missing"


def test_source_discovery_injects_new_york_target_official_candidates_first():
    from data_collection_workflow.nodes.source_discovery import source_dedup_and_registry, source_discovery

    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "NEW YORK",
            "start_date": "2024-10-01",
            "end_date": "2024-10-02",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "NEW YORK",
            "start_date": "2024-10-01",
            "end_date": "2024-10-02",
        },
        "search_query_inventory": [],
        "collection_trace": [],
    }

    discovery = source_discovery(state)
    candidates = discovery["source_candidates"]

    assert candidates[0]["discovery_method"] == "official_coverage_requirement"
    assert candidates[0]["url"].endswith("/2024-10-05_flu_report.pdf")

    registry_result = source_dedup_and_registry({**state, **discovery})
    annotated, _, audit = annotate_source_coverage(registry_result["source_registry"], state)
    target = next(
        row
        for row in annotated
        if str(row.get("canonical_url") or "").endswith("/2024-10-05_flu_report.pdf")
    )
    assert target["must_fetch"] is True
    assert audit["discovered_requirement_count"] == 1


def test_source_registry_deduplicates_new_york_weekly_report_season_aliases():
    from data_collection_workflow.nodes.source_discovery import source_dedup_and_registry

    state = {
        "source_candidates": [
            {
                "source_id": "src_official_short_season",
                "url": (
                    "https://www.health.ny.gov/diseases/communicable/influenza/"
                    "surveillance/2024-25/archive/2024-11-02_flu_report.pdf"
                ),
                "title": "NYSDOH week ending 2024-11-02 flu report",
                "publisher": "New York State Department of Health",
                "source_type": "official_public_health_agency",
                "discovery_method": "official_coverage_requirement",
                "coverage_requirement_ids": [
                    "new_york_influenza_official_week_44_2024"
                ],
                "must_fetch": True,
                "must_fetch_reason": "target official weekly report",
            },
            {
                "source_id": "src_search_long_season",
                "url": (
                    "https://www.health.ny.gov/diseases/communicable/influenza/"
                    "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf"
                ),
                "title": "New York State Influenza Surveillance Report",
                "publisher": "New York State Department of Health",
                "source_type": "official_public_health_agency",
                "discovery_method": "live_search_result",
                "search_rank": 1,
                "target_fit_status": "verified_target_collection",
                "target_verification_status": "verified",
                "triage_role": "verified_target_collection",
            },
        ],
        "collection_trace": [],
    }

    result = source_dedup_and_registry(state)

    registry = result["source_registry"]
    assert len(registry) == 1
    assert result["source_registry_summary"]["duplicate_count"] == 1
    assert result["source_registry_summary"]["official_report_alias_duplicate_count"] == 1
    assert registry[0]["source_id"] == "src_search_long_season"
    assert registry[0]["must_fetch"] is True
    assert registry[0]["coverage_requirement_ids"] == [
        "new_york_influenza_official_week_44_2024"
    ]
    assert registry[0]["official_report_alias_source_ids"] == [
        "src_official_short_season",
        "src_search_long_season",
    ]
    assert registry[0]["official_report_alias_urls"] == [
        "https://www.health.ny.gov/diseases/communicable/influenza/"
        "surveillance/2024-25/archive/2024-11-02_flu_report.pdf",
        "https://www.health.ny.gov/diseases/communicable/influenza/"
        "surveillance/2024-2025/archive/2024-11-02_flu_report.pdf",
    ]


def test_source_registry_prefers_search_verified_vdh_alias_over_generated_url():
    from data_collection_workflow.nodes.source_discovery import source_dedup_and_registry

    state = {
        "source_candidates": [
            {
                "source_id": "src_official_generated_week42",
                "url": (
                    "https://www.vdh.virginia.gov/content/uploads/sites/13/"
                    "2024/10/Weekly-RDS-Report_Week-42.pdf"
                ),
                "title": "Virginia Department of Health official week 42 report",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "discovery_method": "official_coverage_requirement",
                "coverage_requirement_ids": [
                    "virginia_influenza_official_week_42_2024"
                ],
                "must_fetch": True,
                "must_fetch_reason": "target official weekly report",
                "target_fit_status": "predicted_target_candidate",
                "target_verification_status": "predicted_unverified",
                "triage_role": "predicted_target_candidate",
            },
            {
                "source_id": "src_search_vdh_week42",
                "url": (
                    "https://www.vdh.virginia.gov/content/uploads/sites/3/"
                    "2024/10/2024-25_Weekly-RDS-Report_Week-42.pdf"
                ),
                "title": "[PDF] WEEKLY RESPIRATORY DISEASE SURVEILLANCE REPORT",
                "publisher": "Virginia Department of Health",
                "source_type": "official_public_health_agency",
                "discovery_method": "live_search_result",
                "search_rank": 1,
                "target_fit_status": "verified_target_collection",
                "target_verification_status": "verified",
                "triage_role": "verified_target_collection",
                "coverage_requirement_ids": [
                    "virginia_influenza_official_week_42_2024"
                ],
            },
        ],
        "collection_trace": [],
    }

    result = source_dedup_and_registry(state)

    registry = result["source_registry"]
    assert len(registry) == 1
    target = registry[0]
    assert target["source_id"] == "src_search_vdh_week42"
    assert target["canonical_url"].endswith(
        "/2024/10/2024-25_Weekly-RDS-Report_Week-42.pdf"
    )
    assert target["must_fetch"] is True
    assert target["target_fit_status"] == "verified_target_collection"
    assert target["target_verification_status"] == "verified"
    assert target["official_report_alias_source_ids"] == [
        "src_official_generated_week42",
        "src_search_vdh_week42",
    ]
    assert target["official_report_alias_preferred_source_id"] == (
        "src_search_vdh_week42"
    )


def test_new_york_official_weekly_flu_pdfs_are_marked_must_fetch():
    registry = [
        {
            "source_id": "src_ny_week_42",
            "canonical_url": "https://www.health.ny.gov/diseases/communicable/influenza/surveillance/2024-2025/archive/2024-10-19_flu_report.pdf",
            "title": "[PDF] New York State Influenza Surveillance Report - NY.gov",
            "publisher": "New York State Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        },
        {
            "source_id": "src_ny_week_43",
            "canonical_url": "https://www.health.ny.gov/diseases/communicable/influenza/surveillance/2024-2025/archive/2024-10-26_flu_report.pdf",
            "title": "[PDF] New York State Influenza Surveillance Report - NY.gov",
            "publisher": "New York State Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "excluded",
            "final_screening_decision": "do_not_fetch",
            "ready_for_content_fetch": False,
            "blocked_from_fetch": True,
            "credibility_level": "excluded",
        },
        {
            "source_id": "src_mississippi_week_41",
            "canonical_url": "https://msdh.ms.gov/msdhsite/index.cfm/14,20797,199,pdf/Flu_Surveillance_2024_41.pdf",
            "title": "[PDF] 2024-2025 Influenza Surveillance Report Week 41",
            "publisher": "Mississippi State Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        },
        {
            "source_id": "src_cdc_week_41",
            "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-41.html",
            "title": "Weekly US Influenza Surveillance Report: Key Updates for Week 41",
            "publisher": "CDC",
            "source_type": "official_public_health_agency",
            "source_role_final": "context",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "high",
        },
    ]
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "NEW YORK",
            "start_date": "2024-10-01",
            "end_date": "2024-11-01",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "NEW YORK",
            "start_date": "2024-10-01",
            "end_date": "2024-11-01",
        },
    }

    updated, requirements, audit = annotate_source_coverage(registry, state)

    by_id = {item["source_id"]: item for item in updated}

    assert requirements
    assert by_id["src_ny_week_42"]["must_fetch"] is True
    assert by_id["src_ny_week_43"]["must_fetch"] is True
    assert by_id["src_ny_week_43"]["source_role_final"] == "collection"
    assert by_id["src_ny_week_43"]["blocked_from_fetch"] is False
    assert "target_official_must_fetch" in by_id["src_ny_week_43"]["routing_flags"]
    assert not by_id["src_mississippi_week_41"].get("must_fetch")
    assert not by_id["src_cdc_week_41"].get("must_fetch")
    assert audit["discovered_requirement_count"] >= 2


def test_coverage_repair_does_not_revive_unrelated_disease_source():
    from data_collection_workflow.source_coverage import _repair_must_fetch_routing

    entry = {
        "source_id": "src_h5n6",
        "source_role_final": "excluded",
        "final_screening_decision": "do_not_fetch",
        "ready_for_content_fetch": False,
        "blocked_from_fetch": True,
        "blocked_from_fetch_reason": "disease_mismatch_source",
        "source_disease_relevance_status": "unrelated_disease",
    }

    repaired = _repair_must_fetch_routing(
        entry,
        ["official_update"],
        "coverage target",
    )

    assert repaired["must_fetch"] is True
    assert repaired["source_role_final"] == "excluded"
    assert repaired["final_screening_decision"] == "do_not_fetch"
    assert repaired["ready_for_content_fetch"] is False
    assert repaired["blocked_from_fetch"] is True
    assert repaired["blocked_from_fetch_reason"] == "disease_mismatch_source"


def test_vdh_week_reports_are_marked_must_fetch_even_after_excluded_routing():
    registry = [
        {
            "source_id": "src_week_40",
            "canonical_url": "https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-40.pdf",
            "title": "Weekly-RDS-Report_Week-40.pdf",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "excluded",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "blocked_from_fetch": True,
            "blocked_from_fetch_reason": "source_identity_recommended_excluded",
            "credibility_level": "excluded",
        },
        {
            "source_id": "src_cdc_context",
            "canonical_url": "https://www.cdc.gov/fluview/surveillance/2024-week-43.html",
            "title": "CDC FluView Week 43",
            "publisher": "CDC",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
        },
    ]
    state = {
        "structured_task": {
            "disease": "flu",
            "location": "Virginia",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
        },
        "collection_spec": {
            "disease": "flu",
            "geography": "Virginia",
            "start_date": "2024-10-01",
            "end_date": "2024-10-10",
        },
    }

    updated, requirements, audit = annotate_source_coverage(registry, state)

    week_40 = next(item for item in updated if item["source_id"] == "src_week_40")
    cdc = next(item for item in updated if item["source_id"] == "src_cdc_context")

    assert requirements
    assert week_40["must_fetch"] is True
    assert week_40["must_fetch_reason"]
    assert week_40["source_role_final"] == "collection"
    assert week_40["credibility_level"] == "high"
    assert week_40["blocked_from_fetch"] is False
    assert "source_role_final:excluded" in week_40["routing_conflict_warnings"]
    assert week_40["coverage_requirement_ids"] == [
        "virginia_influenza_official_week_40_2024"
    ]
    assert not cdc.get("must_fetch")
    assert audit["requirements"][0]["discovered"] is True


def test_vdh_generic_pages_are_not_must_fetch_just_because_query_mentions_week():
    registry = [
        {
            "source_id": "src_vdh_news_tag",
            "canonical_url": "https://www.vdh.virginia.gov/news/tag/flu",
            "title": "flu Archives - Newsroom - Virginia Department of Health",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "query_used": (
                'site:vdh.virginia.gov "respiratory disease surveillance" '
                '"Week-44" "2024"'
            ),
        },
        {
            "source_id": "src_vdh_week_45",
            "canonical_url": "https://www.vdh.virginia.gov/content/uploads/sites/3/2024/11/2024-25_Weekly-RDS-Report_Week-45.pdf",
            "title": "[PDF] WEEKLY RESPIRATORY DISEASE SURVEILLANCE REPORT",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "excluded",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "credibility_level": "excluded",
        },
    ]
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-11-01",
            "end_date": "2024-11-20",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-11-01",
            "end_date": "2024-11-20",
        },
    }

    updated, requirements, audit = annotate_source_coverage(registry, state)

    news_tag = next(item for item in updated if item["source_id"] == "src_vdh_news_tag")
    week_45 = next(item for item in updated if item["source_id"] == "src_vdh_week_45")

    assert requirements
    assert not news_tag.get("must_fetch")
    assert news_tag.get("coverage_requirement_ids") in (None, [])
    assert week_45["must_fetch"] is True
    assert week_45["coverage_requirement_ids"] == [
        "virginia_influenza_official_week_45_2024"
    ]
    assert audit["discovered_requirement_count"] == 1


def test_vdh_wrong_week_report_does_not_satisfy_requested_week():
    registry = [
        {
            "source_id": "src_vdh_week_12",
            "canonical_url": "https://www.vdh.virginia.gov/content/uploads/sites/3/2025/03/2024-25_Weekly-RDS-Report_Week-12.pdf",
            "title": "[PDF] WEEKLY RESPIRATORY DISEASE SURVEILLANCE REPORT",
            "publisher": "Virginia Department of Health",
            "source_type": "official_public_health_agency",
            "source_role_final": "collection",
            "final_screening_decision": "include_for_content_fetch",
            "ready_for_content_fetch": True,
            "query_used": (
                'site:vdh.virginia.gov "respiratory disease surveillance" '
                '"Week-44" "2024"'
            ),
        }
    ]
    state = {
        "structured_task": {
            "disease": "FLU",
            "location": "VIRGINIA",
            "start_date": "2024-11-01",
            "end_date": "2024-11-02",
        },
        "collection_spec": {
            "disease": "FLU",
            "geography": "VIRGINIA",
            "start_date": "2024-11-01",
            "end_date": "2024-11-02",
        },
    }

    updated, _, audit = annotate_source_coverage(registry, state)

    assert not updated[0].get("must_fetch")
    assert audit["discovered_requirement_count"] == 0


def test_query_strategy_adds_virginia_vdh_weekly_report_hints():
    from data_collection_workflow.nodes.task_scope import query_strategy_builder

    result = query_strategy_builder(
        {
            "collection_spec": {
                "disease": "FLU",
                "geography": "VIRGINIA",
                "start_date": "2024-10-01",
                "end_date": "2024-10-10",
                "time_window": "2024-10-01 to 2024-10-10",
            },
            "structured_task": {
                "disease": "FLU",
                "location": "VIRGINIA",
                "start_date": "2024-10-01",
                "end_date": "2024-10-10",
            },
            "disease_intelligence": {
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            "collection_trace": [],
        }
    )

    queries = [item["query"] for item in result["search_query_inventory"]]

    assert any("site:vdh.virginia.gov" in query and "Week-40" in query for query in queries)
    assert any("site:vdh.virginia.gov" in query and "Week-41" in query for query in queries)
    assert any("respiratory disease surveillance" in query.lower() for query in queries)


def test_query_strategy_uses_new_york_official_report_hints_not_virginia_templates():
    from data_collection_workflow.nodes.task_scope import query_strategy_builder

    result = query_strategy_builder(
        {
            "collection_spec": {
                "disease": "FLU",
                "geography": "NEW YORK",
                "start_date": "2024-10-01",
                "end_date": "2024-10-02",
                "time_window": "2024-10-01 to 2024-10-02",
            },
            "structured_task": {
                "disease": "FLU",
                "location": "NEW YORK",
                "start_date": "2024-10-01",
                "end_date": "2024-10-02",
            },
            "disease_intelligence": {
                "disease_input": "FLU",
                "disease_standard_name": "Seasonal influenza",
                "aliases": ["flu", "influenza", "seasonal influenza"],
                "pathogen_terms": ["influenza"],
                "syndrome_terms": ["influenza-like illness"],
            },
            "collection_trace": [],
        }
    )

    queries = [item["query"] for item in result["search_query_inventory"]]

    assert any(
        "site:health.ny.gov" in query and "2024-10-05_flu_report.pdf" in query
        for query in queries
    )
    assert not any(
        "site:health.ny.gov" in query and "Weekly-RDS-Report" in query
        for query in queries
    )


def test_export_writes_source_coverage_diagnostics(tmp_path):
    from data_collection_workflow.export import export_final_data_package

    package = {
        "final_dataset": [],
        "source_registry": [],
        "linked_events": [],
        "conflicts": [],
        "human_review_items": [],
        "official_coverage_candidates": [{"source_id": "src_ny_week_40"}],
        "target_official_fetch_plan": [{"source_id": "src_ny_week_40"}],
        "source_coverage_requirements": [{"requirement_id": "req_1"}],
        "source_coverage_audit": {"requirement_count": 1},
        "must_fetch_sources": [{"source_id": "src_vdh_week_40"}],
        "fetch_failures_blocking": [{"source_id": "src_vdh_week_40"}],
    }

    manifest = export_final_data_package(package, tmp_path)

    files = manifest["files"]
    assert (tmp_path / "official_coverage_candidates.json").exists()
    assert (tmp_path / "target_official_fetch_plan.json").exists()
    assert (tmp_path / "source_coverage_requirements.json").exists()
    assert (tmp_path / "source_coverage_audit.json").exists()
    assert (tmp_path / "must_fetch_sources.json").exists()
    assert (tmp_path / "fetch_failures_blocking.json").exists()
    assert files["source_coverage_audit_json"].endswith("source_coverage_audit.json")
    assert (tmp_path / "task_evidence_contract.json").exists()
    assert files["task_evidence_contract_json"].endswith("task_evidence_contract.json")
    assert files["official_coverage_candidates_json"].endswith(
        "official_coverage_candidates.json"
    )
    assert files["target_official_fetch_plan_json"].endswith(
        "target_official_fetch_plan.json"
    )


def test_langgraph_state_schema_keeps_source_coverage_fields():
    from data_collection_workflow.state import DataCollectionState

    annotations = DataCollectionState.__annotations__

    assert "source_coverage_requirements" in annotations
    assert "task_evidence_contract" in annotations
    assert "source_coverage_audit" in annotations
    assert "official_coverage_candidates" in annotations
    assert "target_official_fetch_plan" in annotations
    assert "must_fetch_sources" in annotations
    assert "fetch_failures_blocking" in annotations


def test_authority_source_coverage_warns_when_only_news_sources_discovered():
    registry = [
        {
            "source_id": "src_news",
            "canonical_url": "https://www.example-news.org/hantavirus-outbreak",
            "source_type": "news_media",
            "source_type_final": "news_media",
            "source_role_final": "collection_support",
        },
        {
            "source_id": "src_tracker",
            "canonical_url": "https://example-tracker.org/hantavirus-dashboard",
            "source_type": "secondary_aggregator",
            "source_type_final": "secondary_aggregator",
            "source_role_final": "context",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)
    audit = build_source_coverage_audit([], registry, [])

    assert summary["coverage_status"] == "authority_source_recall_incomplete"
    assert "authority_source_recall_incomplete" in summary["warnings"]
    assert summary["news_media_source_count"] == 1
    assert summary["unknown_tracker_context_source_count"] == 1
    assert audit["coverage_status"] == "not_required"
    assert audit["authority_source_coverage_summary"]["coverage_status"] == (
        "authority_source_recall_incomplete"
    )


def test_authority_source_coverage_accepts_official_structured_and_peer_sources():
    registry = [
        {
            "source_id": "src_canada",
            "canonical_url": "https://www.canada.ca/en/public-health/services/diseases.html",
            "source_type": "national_public_health_agency",
            "source_type_final": "national_public_health_agency",
        },
        {
            "source_id": "src_pathoplexus",
            "canonical_url": "https://pathoplexus.org/pathogen/hantavirus",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
        {
            "source_id": "src_nejm",
            "canonical_url": "https://www.nejm.org/doi/full/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)
    audit = build_source_coverage_audit([], registry, [])

    assert summary["coverage_status"] == "authority_source_recall_sufficient"
    assert "authority_source_recall_incomplete" not in summary["warnings"]
    assert summary["official_or_public_health_source_count"] == 1
    assert summary["structured_database_source_count"] == 1
    assert summary["peer_reviewed_or_scientific_source_count"] == 1
    assert audit["coverage_status"] == "not_required"


def test_authority_source_coverage_infers_covered_jurisdictions_from_verified_domains():
    registry = [
        {
            "source_id": "src_canada",
            "canonical_url": "https://www.canada.ca/en/public-health/news/example.html",
            "source_type": "national_public_health_agency",
            "source_type_final": "national_public_health_agency",
        },
        {
            "source_id": "src_france",
            "canonical_url": "https://sante.gouv.fr/actualites-presse/example",
            "source_type": "national_public_health_agency",
            "source_type_final": "national_public_health_agency",
        },
        {
            "source_id": "src_pathoplexus",
            "canonical_url": "https://pathoplexus.org/pathogen/hantavirus",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
        {
            "source_id": "src_nejm",
            "canonical_url": "https://www.nejm.org/doi/full/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["detected_event_jurisdictions"] == ["Canada", "France"]
    assert summary["covered_event_jurisdictions"] == ["Canada", "France"]
    assert summary["missing_event_jurisdictions"] == []
    assert summary["event_jurisdiction_authority_coverage_status"] == (
        "event_jurisdiction_authority_complete"
    )


def test_authority_source_coverage_partial_when_aggregate_and_peer_sources_lack_structured_database():
    registry = [
        {
            "source_id": "src_who",
            "canonical_url": "https://www.who.int/emergencies/disease-outbreak-news/item/2026-DON604",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_ecdc",
            "canonical_url": "https://www.ecdc.europa.eu/en/infectious-disease-topics/hantavirus-infection",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_paho",
            "canonical_url": "https://www.paho.org/en/news/example-hantavirus",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_nejm",
            "canonical_url": "https://www.nejm.org/doi/full/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["official_or_public_health_source_count"] == 3
    assert summary["peer_reviewed_or_scientific_source_count"] == 1
    assert summary["structured_database_source_count"] == 0
    assert summary["coverage_status"] == "authority_source_recall_partial"
    assert "structured_database_source_missing" in summary["warnings"]


def test_authority_source_coverage_recovers_known_government_domain_from_unknown_type():
    registry = [
        {
            "source_id": "src_gov_uk_monitoring",
            "canonical_url": "https://www.gov.uk/government/publications/outbreaks-under-monitoring-in-2026/outbreaks-under-monitoring-week-19-week-ending-10-may-2026",
            "source_type": "unknown",
            "source_type_final": "unknown",
        }
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["official_or_public_health_source_count"] == 1
    assert summary["unknown_tracker_context_source_count"] == 0
    assert summary["domains_by_family"]["official_or_public_health"] == ["gov.uk"]


def test_authority_source_coverage_does_not_trust_planned_type_on_generic_domain():
    registry = [
        {
            "source_id": "src_generic_blog",
            "canonical_url": "https://www.woodsidemd.com/post/hantavirus-overview",
            "source_type": "official_public_health_agency",
            "source_type_final": "official_public_health_agency",
        }
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["official_or_public_health_source_count"] == 0
    assert summary["unknown_tracker_context_source_count"] == 1
    assert summary["coverage_status"] == "authority_source_recall_incomplete"


def test_authority_source_coverage_does_not_count_generic_domains_as_structured_database():
    registry = [
        {
            "source_id": "src_congress_context",
            "canonical_url": "https://www.congress.gov/crs_external_products/IF/PDF/IF13230/IF13230.1.pdf",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
        {
            "source_id": "src_ssrn_context",
            "canonical_url": "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=12345",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["structured_database_source_count"] == 0
    assert summary["unknown_tracker_context_source_count"] == 2
    assert summary["coverage_status"] == "authority_source_recall_incomplete"


def test_authority_source_coverage_reports_event_jurisdiction_gaps():
    registry = [
        {
            "source_id": "src_who",
            "canonical_url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_ecdc",
            "canonical_url": "https://www.ecdc.europa.eu/en/example",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_eurosurveillance",
            "canonical_url": "https://www.eurosurveillance.org/content/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
    ]

    summary = build_authority_source_coverage_summary(
        registry,
        detected_event_jurisdictions=["Canada", "Spain"],
    )

    assert summary["international_or_regional_authority_source_count"] == 2
    assert summary["national_or_subnational_authority_source_count"] == 0
    assert summary["event_jurisdiction_authority_coverage_status"] == (
        "event_jurisdiction_authority_missing"
    )
    assert summary["missing_event_jurisdictions"] == ["Canada", "Spain"]
    assert summary["coverage_status"] != "authority_source_recall_sufficient"


def test_authority_source_coverage_is_stage_aware_for_unextracted_authorities():
    registry = [
        {
            "source_id": "src_who",
            "canonical_url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_canada",
            "canonical_url": "https://www.canada.ca/en/public-health/news/example.html",
            "title": "Canada public health update on cruise-ship hantavirus cluster",
            "source_type": "national_public_health_agency",
            "source_type_final": "national_public_health_agency",
        },
        {
            "source_id": "src_nejm",
            "canonical_url": "https://www.nejm.org/doi/full/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
        {
            "source_id": "src_pathoplexus",
            "canonical_url": "https://pathoplexus.org/seq/example",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
    ]
    documents = [
        {
            "source_id": "src_who",
            "fetch_status": "fetched",
            "parse_status": "parsed_text",
            "quality_status": "usable",
        },
        {
            "source_id": "src_nejm",
            "fetch_status": "fetched",
            "parse_status": "parsed_text",
            "quality_status": "usable",
        },
    ]
    evidence_rows = [
        {"source_id": "src_who", "quality_status": "source_only"},
        {"source_id": "src_nejm", "quality_status": "source_only"},
    ]

    summary = build_authority_source_coverage_summary(
        registry,
        documents=documents,
        evidence_rows=evidence_rows,
        detected_event_jurisdictions=["Canada"],
    )

    assert summary["coverage_status"] == "authority_source_recall_partial"
    assert summary["discovered_count"] == 4
    assert summary["fetched_count"] == 2
    assert summary["parsed_count"] == 2
    assert summary["extracted_evidence_count"] == 0
    assert summary["retained_source_only_count"] == 2
    assert summary["covered_event_jurisdictions"] == []
    assert summary["missing_event_jurisdictions"] == ["Canada"]
    assert "event_jurisdiction_authority_not_extracted" in summary["warnings"]


def test_authority_source_coverage_detects_event_jurisdictions_from_source_text():
    registry = [
        {
            "source_id": "src_who",
            "canonical_url": "https://www.who.int/emergencies/disease-outbreak-news/item/example",
            "title": "Hantavirus cruise ship outbreak affecting Canada, France and Spain",
            "snippet": "Passengers returned to Canada, France, Spain, and the Netherlands.",
            "source_type": "international_public_health_agency",
            "source_type_final": "international_public_health_agency",
        },
        {
            "source_id": "src_nejm",
            "canonical_url": "https://www.nejm.org/doi/full/example",
            "source_type": "academic_or_peer_reviewed_source",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
        {
            "source_id": "src_pathoplexus",
            "canonical_url": "https://pathoplexus.org/seq/example",
            "source_type": "structured_database",
            "source_type_final": "structured_database",
        },
    ]

    summary = build_authority_source_coverage_summary(registry)

    assert summary["coverage_status"] == "authority_source_recall_partial"
    assert summary["detected_event_jurisdictions"] == [
        "Canada",
        "France",
        "Spain",
        "Netherlands",
    ]
    assert summary["missing_event_jurisdictions"] == [
        "Canada",
        "France",
        "Spain",
        "Netherlands",
    ]
    assert "event_jurisdiction_authority_missing" in summary["warnings"]
