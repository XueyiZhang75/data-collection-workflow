"""Offline producer/consumer regressions for comprehensive review rounds 01-06."""
import hashlib

import pytest

from data_collection_workflow.nodes.task_scope import task_intake_and_scope_planning
from data_collection_workflow.source_coverage import build_task_evidence_contract
from data_collection_workflow.evidence_qualification import qualified_coverage
from data_collection_workflow.query_policy import assess_query_task_fit
from data_collection_workflow.source_identity import assess_source_identity, enrich_source_identity_post_fetch
from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")


def task(fields=None):
    return {"disease":"dengue", "location":"Brazil", "start_date":"2025-01-01",
            "end_date":"2025-12-31", "target_fields": fields or []}


def contract(fields):
    state={"structured_task":task(fields)}
    state.update(task_intake_and_scope_planning(state))
    return build_task_evidence_contract(state)


def qualified(**fields):
    return {"record_id":"source-bound-record", "disease":"dengue", "country":"Brazil",
            "reporting_period":"2025", "evidence_qualification":{"status":"qualified", "product_kind":"aggregate"}, **fields}


def test_pass01_death_request_cannot_be_satisfied_by_case_only_record():
    result=qualified_coverage(contract(["deaths"])["requirements"], [qualified(cases_confirmed=12)])
    assert result["coverage_complete"] is False


def test_pass01_each_requested_metric_remains_a_coverage_obligation():
    requirements=contract(["cases_confirmed","deaths"])["requirements"]
    assert not qualified_coverage(requirements,[qualified(cases_confirmed=12)])["coverage_complete"]
    assert qualified_coverage(requirements,[qualified(cases_confirmed=12),qualified(record_id="zero-deaths",deaths=0)])["coverage_complete"]
    assert not qualified_coverage(contract(["cases_confirmed"])["requirements"], [qualified(cases_probable=12)])["coverage_complete"]


@pytest.mark.parametrize("query", ["monkeypox United States 2025 surveillance data", "mpox United States 2025 contact tracing data"])
def test_pass02_legitimate_name_and_retrieval_queries_remain_eligible(query):
    state={"structured_task":{**task(),"disease":"mpox","location":"United States"}}
    assert assess_query_task_fit({"query":query},state)["accepted"]


def test_pass02_unrequested_event_and_wrong_disease_remain_rejected():
    state={"structured_task":{**task(),"disease":"mpox","location":"United States"}}
    for query in ("smallpox United States 2025 surveillance", "mpox United States 2025 cruise Aurora"):
        assert not assess_query_task_fit({"query":query},state)["accepted"]


def test_pass03_same_disease_name_survives_real_chunk_relevance_gate():
    from data_collection_workflow.evidence_chunking import build_evidence_chunks
    text="Monkeypox in United States: 12 confirmed cases during 2025. " + "Surveillance methods are described in the report. "*2
    digest=hashlib.sha256(text.encode()).hexdigest()
    document={"source_id":"source", "document_id":"doc", "content_hash":digest, "text_hash":digest,
              "clean_text":text, "content_readable":True, "parse_status":"parsed_text", "document_type":"text",
              "quality_status":"usable", "extraction_readiness":"ready", "source_type":"official_public_health"}
    result=build_evidence_chunks({"structured_task":{"disease":"mpox"},"documents":[document]})
    assert any(chunk["extraction_eligible_for_task_disease"] for chunk in result["evidence_chunks"])


@pytest.mark.parametrize("title", ["Disease surveillance news", "WHO reports annual disease cases"])
def test_pass04_article_citation_is_upstream_not_host_publisher(title):
    entry={"source_id":"news", "url":"https://reuters.com/world/report", "title":title,
           "publisher":"Reuters", "source_type":"news_and_situation_report"}
    original=assess_source_identity(entry)
    result=enrich_source_identity_post_fetch(entry,original,{"source_id":"news", "title":title,
        "clean_text":"World Health Organization reports annual cases in Brazil.", "metadata":{}},
        collection_spec={"disease":"dengue","geography":"Brazil"})
    assert result["actual_publisher"] == "Reuters"
    assert "World Health Organization" in result["upstream_source_mentions"]
    assert result["source_type_final"] == "news_media"


def test_pass05_source_candidates_alone_never_count_as_qualified_coverage():
    requirements=contract(["deaths"])["requirements"]
    assert not qualified_coverage(requirements,[])["coverage_complete"]
    assert not qualified_coverage(requirements,[qualified(deaths=3,country="Canada")])["coverage_complete"]


def test_pass06_explicitly_blocked_sources_cannot_starve_recovery_fetch_slots():
    sources=[{"source_id":f"a{i:02}","url":f"https://irrelevant.example/{i}","source_role_final":"collection",
              "blocked_from_fetch":True,"blocked_from_fetch_reason":"explicit_user_exclusion"} for i in range(16)]
    sources.append({"source_id":"z_relevant","url":"https://health.example/relevant","source_role_final":"collection",
                    "final_screening_decision":"include_for_content_fetch","ready_for_content_fetch":True})
    state={"source_registry":sources}
    plan=plan_recovery(assess_collection_gaps(state),state=state,
        budget={"remaining":{"fetch":20,"fetch_ordinary":20,"extraction":20,"search":20}})
    assert any(action.target_id=="z_relevant" for action in plan.actions)
    assert not any(action.target_id.startswith("a") for action in plan.actions)


def test_pass01_unspecified_metrics_keep_existing_generic_requirements():
    result=contract([])
    assert not any(row.get("required_metric_fields") for row in result["requirements"])


def test_pass04_llm_advice_cannot_restore_a_cited_agency_as_news_publisher():
    entry={"source_id":"news", "url":"https://reuters.com/world/report", "publisher":"Reuters"}
    original=assess_source_identity(entry)
    result=enrich_source_identity_post_fetch(entry,original,{"source_id":"news",
        "clean_text":"World Health Organization reported cases.", "metadata":{}},
        llm_decision={"actual_publisher":"World Health Organization", "source_type_final":"international_public_health_agency"})
    assert result["actual_publisher"] == "Reuters"
    assert result["source_type_final"] == "news_media"


@pytest.mark.parametrize("end", ["2025-01-07", "2025-01-31"])
def test_pass05_annual_requirement_is_not_satisfied_by_one_qualified_fragment(end):
    requirements=contract(["cases_confirmed"])["requirements"]
    row=qualified(cases_confirmed=12, reporting_period="2025-01",
                  metric_period_start="2025-01-01", metric_period_end=end)
    result=qualified_coverage(requirements,[row])
    assert result["coverage_complete"] is False
    assert result["requirements"][0]["coverage_status"] == "evidence_gap"
    assert row["evidence_qualification"]["status"] == "qualified"
    assert row["metric_period_end"] == end


def test_pass05_qualified_fragments_can_jointly_cover_requested_annual_period():
    requirements=contract(["cases_confirmed"])["requirements"]
    rows=[qualified(record_id="first-half",cases_confirmed=12,reporting_period="2025-H1",
                    metric_period_start="2025-01-01",metric_period_end="2025-06-30"),
          qualified(record_id="second-half",cases_confirmed=14,reporting_period="2025-H2",
                    metric_period_start="2025-07-01",metric_period_end="2025-12-31")]
    assert qualified_coverage(requirements, rows)["coverage_complete"]
    rows[1]["metric_period_start"]="2025-07-02"
    assert not qualified_coverage(requirements, rows)["coverage_complete"]


@pytest.mark.parametrize("fields,quote,values,product_key", [
    ([], "Patient A in Canada had Pertussis during 2025.",
     {"workflow_case_label":"Patient A"}, "case_entities"),
    ([], "Patient A in Canada during 2025 had Pertussis: 1 confirmed case.",
     {"workflow_case_label":"Patient A","cases_confirmed":1}, "case_entities"),
    (["deaths"], "Pertussis in Canada during 2025: 12 confirmed cases.",
     {"cases_confirmed":12}, "aggregate_groups"),
])
def test_pass01_real_contract_preserves_observations_without_closing_other_product_coverage(
        tmp_path, fields, quote, values, product_key):
    from test_evidence_positive_extraction import source_state
    from data_collection_workflow.evidence_qualification import qualify_records, build_evidence_index
    from data_collection_workflow.evidence_products import build_evidence_products
    state={"structured_task":{"disease":"Pertussis","location":"Canada",
           "start_date":"2025-01-01","end_date":"2025-12-31","target_fields":fields}}
    state.update(task_intake_and_scope_planning(state))
    source=source_state(tmp_path,quote.encode(),"text/plain")
    state.update(documents=source["documents"],evidence_chunks=source["evidence_chunks"])
    task_contract=build_task_evidence_contract(state)
    index=build_evidence_index(state)
    chunk=state["evidence_chunks"][0]
    row={"record_id":"actual-source-observation","source_id":"source",
         "supporting_chunk_id":chunk["chunk_id"],"disease":"Pertussis","country":"Canada",
         "reporting_period":"2025","case_span_quote":quote,**values}
    result=qualify_records([row],contract=task_contract,evidence_index=index)
    assert result["qualified_records"], result["record_qualifications"]
    products=build_evidence_products(result["qualified_records"],evidence_index=index)
    assert products[product_key]
    assert not qualified_coverage(task_contract["requirements"],result["qualified_records"])["coverage_complete"]
    wrong_scope={**row,"country":"France"}
    assert not qualify_records([wrong_scope],contract=task_contract,evidence_index=index)["qualified_records"]
