"""Field recovery preserves observation identity across generic metric aliases."""
from copy import deepcopy
import socket
import pytest
from test_acquisition_recovery import source_state
from data_collection_workflow.workflow_recovery import RecoveryAction, RecoveryDelta, _repair_candidate_fields, merge_recovery_delta
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for key,value in {"PIPELINE_MODE":"evidence","ENABLE_LLM_EXTRACTION":"false",
                      "ENABLE_LLM_SOURCE_IDENTITY":"false","ENABLE_LIVE_SEARCH":"false",
                      "ENABLE_LIVE_FETCH":"false"}.items():
        monkeypatch.setenv(key,value)
    def forbidden(*args,**kwargs):
        raise AssertionError("Recovery metric tests forbid network")
    monkeypatch.setattr(socket.socket,"connect",forbidden)

def material(disease="measles",country="France",metric="cases_confirmed",name=None):
    text=f"From 02 February 2025 to 19 April 2025, {country} reported 12 confirmed {disease} cases, including 2 deaths."
    state=source_state(text)
    state["structured_task"].update(disease=disease,location=country)
    value=12 if metric=="cases_confirmed" else 2
    row={"record_id":"original","source_id":"s","supporting_chunk_id":"c","disease":disease,"country":country,
         metric:value,"metric_name":name or metric,"metric_value":value,"metric_unit":"count",
         "evidence_quote":text,"field_provenance_json":{metric:{"quote":text}}}
    state.update(raw_records=[row],candidate_records=[row],normalized_records=[row])
    return state,row

def repair(state):
    return _repair_candidate_fields(RecoveryAction("repair_fields","original","repair-original","recover source-backed dates"),state)

def merged(state,update):
    delta=RecoveryDelta(record_updates=[update])
    result={**state,**merge_recovery_delta(state,delta)}
    again={**result,**merge_recovery_delta(result,delta)}
    assert result["raw_records"]==again["raw_records"]
    assert len(result["raw_records"])==1
    return result,result["raw_records"][0]

@pytest.mark.parametrize("disease,country",[("measles","France"),("dengue","Brazil")])
@pytest.mark.parametrize("metric,name",[("cases_confirmed","cases_confirmed"),("cases_confirmed","confirmed_cases"),("deaths","deaths")])
def test_alias_of_explicit_supported_count_allows_field_recovery(disease,country,metric,name):
    state,row=material(disease,country,metric,name)
    original=deepcopy(row)
    before=assess_record_evidence(row,contract=state["structured_task"],evidence_index=build_evidence_index(state))
    assert before.status=="candidate"
    update,reason=repair(state)
    assert update is not None,reason
    assert set(update["fields"])<= {"metric_period_start","metric_period_end","reporting_period","geographic_scope"}
    assert update["fields"]["metric_period_start"]=="2025-02-02"
    assert update["fields"]["metric_period_end"]=="2025-04-19"
    final,result=merged(state,update)
    for field in (metric,"metric_name","metric_value","metric_unit","disease","country","evidence_quote"):
        assert result[field]==original[field]
    assert state["raw_records"][0]==original
    after=assess_record_evidence(result,contract=state["structured_task"],evidence_index=build_evidence_index(final))
    assert after.status=="qualified",after.reasons

@pytest.mark.parametrize("mutation",[
    {"metric_name":"deaths"},
    {"metric_name":"unrecognized_metric"},
    {"metric_value":13},
    {"metric_unit":"percent"},
    {"metric_unit":"rate"},
    {"count_unit":"deaths"},
    {"deaths":13},
    {"cases_confirmed":None,"cases_probable":12},
    {"case_definition":"probable"},
])
def test_conflicting_metric_definition_or_unit_cannot_be_aliased(mutation):
    state,row=material()
    row.update(mutation)
    before=deepcopy(state)
    update,reason=repair(state)
    assert update is None
    assert reason=="different_or_ambiguous_observation_count"
    assert state==before

@pytest.mark.parametrize("bad",[
    {"event_start_date":"2025-02-02"},
    {"event_end_date":"2025-04-19"},
    {"count_semantics":"cumulative","statistical_count_type":"cumulative"},
])
def test_supported_field_gain_does_not_erase_existing_unsupported_claims(bad):
    state,row=material()
    row.update(bad)
    update,reason=repair(state)
    assert update is not None,reason
    final,result=merged(state,update)
    assert all(result[key]==value for key,value in bad.items())
    assert all(key not in update["fields"] for key in bad)
    qualification=assess_record_evidence(result,contract=state["structured_task"],evidence_index=build_evidence_index(final))
    assert qualification.status=="candidate"
    assert any(reason.startswith(tuple(bad)) for reason in qualification.reasons)

def test_alias_repair_legacy_duplicate_identity_still_fails_closed():
    state,row=material()
    state["raw_records"].append(dict(row))
    assert repair(state)==(None,"ambiguous_record_identity")

def test_alias_repair_cannot_use_same_count_in_sibling_country_statement():
    first="France reported 12 confirmed measles cases."
    state=source_state(first+" Germany reported 12 confirmed measles cases during 2025.")
    row={"record_id":"original","source_id":"s","supporting_chunk_id":"c","disease":"measles","country":"France",
         "cases_confirmed":12,"metric_name":"cases_confirmed","metric_value":12,"metric_unit":"count",
         "evidence_quote":first,"field_provenance_json":{"cases_confirmed":{"quote":first}}}
    state["raw_records"]=[row]
    update,reason=repair(state)
    assert update is None
    assert reason=="no_supported_field_gain"

@pytest.mark.parametrize("failure",["hash","span"])
def test_alias_repair_still_requires_original_hash_and_locator(failure):
    state,row=material()
    if failure=="hash":
        state["documents"][0]["clean_text"]+=" Tampered."
    else:
        row["field_provenance_json"]["cases_confirmed"]["quote"]="Invented original statement."
    assert repair(state)==(None,"unresolved_original_observation_locator")

def headed_material():
    import hashlib
    from data_collection_workflow.evidence_chunking import bound_context_for
    state,row=material()
    original=state["evidence_chunks"][0]["text"]
    heading="Measles surveillance in France"
    full=heading+"\nBackground navigation without observations.\n\n"+original
    offset=full.index(original);digest=hashlib.sha256(full.encode()).hexdigest()
    doc=state["documents"][0]
    doc.update(clean_text=full,content_hash=digest,text_hash=digest,
        locator_spans=[{"char_start":0,"char_end":len(heading),"role":"heading","heading_level":1,"section_id":"scope"}])
    chunk=state["evidence_chunks"][0]
    chunk.update(char_start=offset,char_end=offset+len(original),document_hash=digest)
    chunk["bound_context_spans"]=bound_context_for(doc,offset)
    assert chunk["bound_context_spans"]
    return state,row

def test_recovered_field_quote_is_raw_slice_with_structural_context_separate():
    state,row=headed_material()
    update,reason=repair(state)
    assert update is not None,reason
    text=state["documents"][0]["clean_text"]
    for field,provenance in update["field_provenance"].items():
        assert provenance["quote"]==text[provenance["char_start"]:provenance["char_end"]]
        assert provenance["quote"] in text
        assert provenance["bound_context_spans"]==state["evidence_chunks"][0]["bound_context_spans"]
    final,result=merged(state,update)
    assert assess_record_evidence(result,contract=state["structured_task"],evidence_index=build_evidence_index(final)).status=="qualified"

def test_recovered_field_cannot_use_tampered_structural_header():
    state,row=headed_material()
    state["evidence_chunks"][0]["bound_context_spans"][0]["quote"]="Measles surveillance in Germany"
    assert repair(state)==(None,"unresolved_original_observation_locator")
