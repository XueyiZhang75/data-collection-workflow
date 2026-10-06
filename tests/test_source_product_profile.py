from __future__ import annotations

from data_collection_workflow.source_product_profile import profile_source_product


def _state() -> dict:
    return {
        "structured_task": {
            "disease": "hantavirus",
            "location": "Global",
            "start_date": "2026-04-01",
            "end_date": "2026-06-30",
        },
        "disease_intelligence": {
            "aliases": ["hantavirus"],
            "pathogen_terms": ["Andes virus", "ANDV"],
        },
    }


def test_source_product_profile_classifies_generic_background_not_event_source():
    profile = profile_source_product(
        {
            "canonical_url": "https://www.cdc.gov/hantavirus/about/index.html",
            "title": "About Hantavirus",
            "source_type_final": "official_public_health_agency",
        },
        _state(),
    )

    assert profile["data_product_type"] == "background_fact_sheet"
    assert profile["task_specificity"] == "disease_specific_but_context"
    assert profile["expected_evidence_role"] == "source_context"
    assert profile["source_product_profile_reason"] == "background_or_fact_sheet"


def test_source_product_profile_does_not_upgrade_fact_sheet_from_query_terms():
    profile = profile_source_product(
        {
            "canonical_url": "https://www.cdc.gov/hantavirus/about/index.html",
            "title": "About Hantavirus",
            "snippet": "Background information for public health; cases can be severe.",
            "query_used": "hantavirus outbreak cases deaths 2026",
            "source_type_final": "official_public_health_agency",
        },
        _state(),
    )

    assert profile["data_product_type"] == "background_fact_sheet"
    assert profile["task_specificity"] == "disease_specific_but_context"
    assert profile["machine_readability"] == "not_extractable"
    assert profile["expected_evidence_role"] == "source_context"


def test_source_product_profile_classifies_case_report_article():
    profile = profile_source_product(
        {
            "canonical_url": "https://www.nejm.org/doi/full/10.1056/NEJMc2606496",
            "title": "Andes Hantavirus Outbreak on a Cruise Ship, 2026",
            "source_type_final": "academic_or_peer_reviewed_source",
        },
        _state(),
    )

    assert profile["data_product_type"] == "case_report_article"
    assert profile["task_specificity"] == "event_specific"
    assert profile["machine_readability"] == "narrative_extractable"
    assert profile["expected_evidence_role"] == "individual_case_evidence"


def test_source_product_profile_classifies_sequence_record_and_unrelated_database_page():
    seq_profile = profile_source_product(
        {
            "canonical_url": "https://pathoplexus.org/seq/PP_006VZMV.2",
            "title": "Pathoplexus | PP_006VZMV.2 Andes virus",
            "source_type_final": "structured_database",
        },
        _state(),
    )
    unrelated_profile = profile_source_product(
        {
            "canonical_url": "https://pathoplexus.org/ebola-bdbv/search",
            "title": "Ebola Bundibugyo - Browse - Pathoplexus",
            "source_type_final": "structured_database",
        },
        _state(),
    )

    assert seq_profile["data_product_type"] == "sequence_database_record"
    assert seq_profile["expected_evidence_role"] == "individual_case_evidence"
    assert unrelated_profile["data_product_type"] == "unrelated_or_other"
    assert unrelated_profile["task_specificity"] == "unrelated"
    assert unrelated_profile["expected_evidence_role"] == "do_not_extract"


def test_source_product_profile_classifies_canada_biosafety_sheet_as_context():
    profile = profile_source_product(
        {
            "canonical_url": "https://www.canada.ca/en/public-health/services/laboratory-biosafety-biosecurity/pathogen-safety-data-sheets-risk-assessment/hantaan-orthohantavirus.html",
            "title": "Hantaan orthohantavirus - Pathogen Safety Data Sheets",
            "source_type_final": "national_public_health_agency",
        },
        _state(),
    )

    assert profile["data_product_type"] == "background_fact_sheet"
    assert profile["machine_readability"] == "not_extractable"
    assert profile["expected_evidence_role"] == "source_context"


def test_source_product_profile_keeps_must_fetch_target_pdf_task_relevant():
    state = {
        "structured_task": {
            "disease": "influenza",
            "location": "Virginia",
            "start_date": "2024-10-06",
            "end_date": "2024-10-12",
        }
    }
    profile = profile_source_product(
        {
            "source_id": "src_vdh_weekly_pdf",
            "canonical_url": "https://www.vdh.virginia.gov/content/uploads/sites/13/2024/10/Weekly-RDS-Report_Week-41.pdf",
            "title": "Weekly RDS Report Week 41",
            "publisher": "Virginia Department of Health",
            "source_type": "official_health_agency",
            "must_fetch": True,
            "coverage_requirement_ids": ["virginia_influenza_official_week_41_2024"],
            "target_fit_status": "predicted_target_candidate",
        },
        state,
    )

    assert profile["data_product_type"] == "official_surveillance_report"
    assert profile["task_specificity"] != "unrelated"
    assert profile["expected_evidence_role"] == "aggregate_event_evidence"


def test_source_product_profile_does_not_promote_background_page_because_must_fetch():
    profile = profile_source_product(
        {
            "source_id": "src_cdc_about_hantavirus",
            "canonical_url": "https://www.cdc.gov/hantavirus/about/index.html",
            "title": "About Hantavirus",
            "publisher": "Centers for Disease Control and Prevention",
            "source_type": "official_public_health_agency",
            "must_fetch": True,
            "coverage_requirement_ids": [
                "global_hantavirus_task_window_2026_04_01_2026_06_30"
            ],
        },
        _state(),
    )

    assert profile["data_product_type"] == "background_fact_sheet"
    assert profile["task_specificity"] == "disease_specific_but_context"
    assert profile["machine_readability"] == "not_extractable"
    assert profile["expected_evidence_role"] == "source_context"
    assert profile["source_product_profile_reason"] == "background_or_fact_sheet"


def test_source_product_profile_keeps_verified_authority_event_pages_fetchable():
    rivm_profile = profile_source_product(
        {
            "canonical_url": "https://www.rivm.nl/en/news/arrival-and-cleaning-of-cruise-ship-hondius",
            "title": "Arrival and cleaning of cruise ship Hondius",
            "snippet": "RIVM update about the MV Hondius cruise ship in 2026.",
            "source_type_final": "national_public_health_agency",
        },
        _state(),
    )
    govuk_profile = profile_source_product(
        {
            "canonical_url": "https://www.gov.uk/government/publications/outbreaks-under-monitoring/outbreaks-under-monitoring-week-19-2026",
            "title": "Outbreaks under monitoring: week 19, 2026",
            "snippet": "UKHSA monitoring update includes hantavirus and MV Hondius.",
            "source_type_final": "national_public_health_agency",
        },
        _state(),
    )
    sante_profile = profile_source_product(
        {
            "canonical_url": "https://sante.gouv.fr/soins-et-maladies/maladies/maladies-infectieuses/article/hantavirus-point-de-situation-2026",
            "title": "Hantavirus: point de situation MV Hondius",
            "snippet": "DGS urgent update on cas linked to the navire MV Hondius.",
            "source_type_final": "national_public_health_agency",
        },
        _state(),
    )

    for profile in (rivm_profile, govuk_profile, sante_profile):
        assert profile["data_product_type"] == "event_outbreak_report"
        assert profile["task_specificity"] == "event_specific"
        assert profile["machine_readability"] != "not_extractable"
        assert profile["expected_evidence_role"] in {
            "aggregate_event_evidence",
            "non_case_or_monitoring_evidence",
        }
        assert profile["source_product_profile_reason"] != "background_or_fact_sheet"
