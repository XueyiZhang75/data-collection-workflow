"""General observation producer contract: typed periods and local count evidence."""
import hashlib
import json
import socket
import pytest
from data_collection_workflow.config import load_llm_structured_extraction_policy, load_structured_extraction_policy
from data_collection_workflow.models import LLMStructuredExtractionPolicy, LLMExtractedRecord, StructuredExtractionPolicy
from data_collection_workflow.llm_clients import _build_llm_messages, OUTPUT_LANGUAGE_INSTRUCTION
from data_collection_workflow.nodes.extraction import _build_record_from_llm_output, _official_outbreak_record_from_chunk
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index

@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE","evidence")
    def forbidden(*args,**kwargs):
        raise AssertionError("No network is permitted in producer contract tests")
    monkeypatch.setattr(socket.socket,"connect",forbidden)

def policy():
    return LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())

def messages():
    return _build_llm_messages({"source_id":"s","chunk_id":"c","text":"Evidence only."},policy())

def test_evidence_does_not_instruct_chunk_wide_death_inheritance():
    system=messages()[0]["content"]
    assert "death count is explicitly stated anywhere in the chunk" not in system
    assert "same observation" in system and "deaths" in system

def test_evidence_teaches_metric_period_and_clinical_event_roles():
    system=messages()[0]["content"]
    assert "metric_period_start" in system and "metric_period_end" in system
    assert "event_start_date" in system and "event_end_date" in system
    assert "publication_date" in system and "date_reported" in system
    assert "observation interval" in system and "clinical event" in system

def test_evidence_does_not_force_interval_to_cumulative_or_annual():
    system=messages()[0]["content"]
    assert "does not by itself establish cumulative" in system
    assert "statistical_count_type" in system and "unknown" in system
    assert "task time window" in system

def test_evidence_field_quotes_retain_binding_not_isolated_numbers():
    system=messages()[0]["content"]
    assert "complete local assertion" in system
    assert "field_provenance_json" in system and "time role" in system

@pytest.mark.parametrize("revision",["legacy","standard"])
def test_standard_prompt_preserves_policy_with_english_presentation_instruction(monkeypatch,revision):
    monkeypatch.setenv("PIPELINE_MODE",revision)
    p=policy()
    before=p.model_dump()
    actual=_build_llm_messages({"text":"source"},p)[0]["content"]
    expected=p.system_prompt+"\n\nRequired output rules:\n"+"\n".join(f"{i+1}. {r}" for i,r in enumerate(p.required_output_rules))
    assert actual.count(OUTPUT_LANGUAGE_INSTRUCTION) == 1
    assert actual.replace("\n\n" + OUTPUT_LANGUAGE_INSTRUCTION, "", 1) == expected
    assert p.model_dump()==before
    assert "death count is explicitly stated anywhere in the chunk" in actual

def material(disease,country,noun="fatalities"):
    text=f"From 02 February 2024 to 19 April 2024, {country} reported 41 confirmed {disease} cases, including 3 {noun}."
    digest=hashlib.sha256(text.encode()).hexdigest()
    chunk={"source_id":"s","chunk_id":"c","document_hash":digest,"text":text,
           "char_start":0,"char_end":len(text),"source_url":"https://evidence.invalid/report",
           "source_type":"official_public_health_agency","chunk_kind":"text","contains_target_data":True,
           "fetch_purpose":"data_extraction"}
    doc={"source_id":"s","content_hash":digest,"text_hash":digest,"clean_text":text}
    state={"documents":[doc],"evidence_chunks":[chunk]}
    context={"disease_standard_name":disease,"disease_terms":[disease],"aliases":[disease],"is_hantavirus":False}
    return text,chunk,state,context

def typed_record(disease,country,metric="deaths",noun="fatalities",extra=None):
    text,chunk,state,context=material(disease,country,noun)
    value=3 if metric=="deaths" else 41
    fields={"disease":disease,"country":country,metric:value,"metric_name":metric,"metric_value":value,
            "metric_unit":"count","metric_category":"death_count" if metric=="deaths" else "case_count",
            "metric_period_start":"2024-02-02","metric_period_end":"2024-04-19",
            "reporting_period":"02 February 2024 to 19 April 2024","statistical_count_type":"unknown",
            "count_semantics":"unknown"}
    fields.update(extra or {})
    quotes={k:{"quote":text} for k,v in fields.items() if v is not None}
    llm=LLMExtractedRecord(**fields,evidence_quote=text,field_provenance_json=quotes)
    row=_build_record_from_llm_output(llm,chunk,1,policy(),{"provider":"offline","model":"typed-fixture"},context)
    assert row is not None
    q=assess_record_evidence(row.model_dump(),contract={"disease":disease,"location":country},
                            evidence_index=build_evidence_index(state))
    return row,q,state

@pytest.mark.parametrize("disease,country",[("measles","Canada"),("dengue","Brazil"),("cholera","Peru")])
@pytest.mark.parametrize("metric",["cases_confirmed","deaths"])
def test_correctly_typed_model_observation_qualifies(disease,country,metric):
    row,q,_=typed_record(disease,country,metric)
    assert q.status=="qualified",q.reasons
    assert row.metric_period_start=="2024-02-02" and row.metric_period_end=="2024-04-19"
    assert row.event_start_date is None and row.event_end_date is None
    assert row.statistical_count_type=="unknown"

@pytest.mark.parametrize("extra",[
    {"event_start_date":"2024-02-02"},
    {"event_end_date":"2024-04-19"},
    {"statistical_count_type":"cumulative","count_semantics":"cumulative"},
    {"metric_period_end":"2024-12-31"},
    {"deaths":4,"metric_value":4},
    {"country":"France"},
])
def test_wrong_type_or_unsupported_value_remains_candidate(extra):
    row,q,_=typed_record("measles","Canada",extra=extra)
    assert q.status=="candidate",q.to_dict()
    for key,value in extra.items():
        assert getattr(row,key)==value  # No deleting facts to qualify.

@pytest.mark.parametrize("disease,country",[("measles","Canada"),("dengue","Brazil")])
def test_rule_and_model_agree_on_typed_local_observation(disease,country):
    text,chunk,state,context=material(disease,country)
    p=StructuredExtractionPolicy(**load_structured_extraction_policy())
    row,diag=_official_outbreak_record_from_chunk(chunk,1,p,context)
    assert row is not None,diag
    assert row.cases_confirmed==41 and row.deaths==3
    assert row.metric_period_start=="2024-02-02" and row.metric_period_end=="2024-04-19"
    assert row.event_start_date is None and row.event_end_date is None
    q=assess_record_evidence(row.model_dump(),contract={"disease":disease,"location":country},evidence_index=build_evidence_index(state))
    assert q.status=="qualified",q.reasons

@pytest.mark.parametrize("text",[
    "Canada reported 3 fatalities from cholera in 2024. Brazil reported measles cases in 2024.",
    "Canada reported 3 fatality rates for measles in 2024.",
    "Canada did not report 3 fatalities from measles in 2024.",
    "Canada might report 3 fatalities from measles in 2024.",
    "Canada reported 3% fatalities from measles in 2024.",
])
def test_fatality_synonym_does_not_relax_wrong_scope_or_count(text):
    digest=hashlib.sha256(text.encode()).hexdigest()
    chunk={"source_id":"s","chunk_id":"c","text":text,"char_start":0,"char_end":len(text)}
    row={"source_id":"s","supporting_chunk_id":"c","disease":"measles","country":"Canada",
         "reporting_period":"2024","deaths":3}
    index=build_evidence_index({"documents":[{"source_id":"s","content_hash":digest,"clean_text":text}],"evidence_chunks":[chunk]})
    assert assess_record_evidence(row,contract={},evidence_index=index).status=="candidate"


def test_evidence_keeps_geographic_levels_separate_without_common_knowledge_parent():
    system=messages()[0]["content"]
    assert "country and subnational_location independently" in system
    assert "common geographic knowledge" in system
    assert "explicitly supported locality" in system

@pytest.mark.parametrize("place", ["Ontario","Tasmania","Puerto Rico"])
def test_model_can_preserve_supported_region_without_inventing_parent_country(place):
    text,chunk,state,context=material("measles",place,"deaths")
    fields={"disease":"measles","country":None,"subnational_location":place,
            "geographic_scope":place,"geographic_scope_type":"subnational","cases_confirmed":41,
            "metric_period_start":"2024-02-02","metric_period_end":"2024-04-19",
            "reporting_period":"02 February 2024 to 19 April 2024"}
    model=LLMExtractedRecord(**fields,evidence_quote=text,field_provenance_json={k:{"quote":text} for k,v in fields.items() if v is not None})
    row=_build_record_from_llm_output(model,chunk,1,policy(),{"provider":"offline","model":"typed-fixture"},context)
    assert row.country is None
    assert row.subnational_location==place
    q=assess_record_evidence(row.model_dump(),contract={"disease":"measles","location":place},evidence_index=build_evidence_index(state))
    assert q.status=="qualified",q.reasons
    invented={**row.model_dump(),"country":"Canada"}
    # Even a real-world parent cannot be supplied without source evidence.
    assert assess_record_evidence(invented,contract={},evidence_index=build_evidence_index(state)).status=="candidate"
