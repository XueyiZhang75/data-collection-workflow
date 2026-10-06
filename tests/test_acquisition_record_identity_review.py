"""Independent source-position and semantic record identity contracts."""
from copy import deepcopy
from hashlib import sha256
import socket

import pytest

from data_collection_workflow.record_identity import record_identity_payload, stable_record_id
from data_collection_workflow.workflow_recovery import RecoveryDelta, merge_recovery_delta


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")
    monkeypatch.setattr(socket.socket,"connect",lambda *args,**kwargs:pytest.fail("network forbidden"))


def observation():
    text="Canada reported 12 confirmed measles cases during 2025."
    digest=sha256(text.encode()).hexdigest()
    row={"source_id":"source","supporting_chunk_id":"chunk","disease":"measles",
         "country":"Canada","reporting_period":"2025","cases_confirmed":12,"evidence_quote":text}
    chunk={"source_id":"source","chunk_id":"chunk","document_id":"document","document_hash":digest,
           "char_start":0,"char_end":len(text),"text":text,"chunk_kind":"text"}
    return row,chunk


def test_fact_key_order_and_derived_annotations_do_not_change_identity():
    row,chunk=observation()
    annotated=dict(reversed(list(row.items())))
    annotated.update(record_id="batch-local-99",conflict_ids=["conflict-new"],
        record_conflict_status="needs_review",linked_event_id="event-next",
        evidence_qualification={"status":"candidate","reasons":["diagnostic"]},
        extraction_confidence=0.35,normalization_actions=["format_only"],
        requires_human_review=True)
    assert record_identity_payload(row,chunk)==record_identity_payload(annotated,chunk)
    assert stable_record_id(row,chunk)==stable_record_id(annotated,chunk)


def test_model_generated_display_number_is_not_source_observation_identity():
    row,chunk=observation()
    first={**row,"workflow_case_label":"case_001"}
    repeated={**row,"workflow_case_label":"case_019"}
    assert stable_record_id(first,chunk)==stable_record_id(repeated,chunk)


def test_two_identical_clinical_descriptions_at_distinct_source_spans_do_not_collide():
    sentence="The patient had fever and was hospitalized."
    text=sentence+" "+sentence
    chunk={"source_id":"clinical","chunk_id":"two-patients","document_id":"clinical-document",
           "document_hash":sha256(text.encode()).hexdigest(),"text":text,"char_start":0,"char_end":len(text)}
    first={"source_id":"clinical","supporting_chunk_id":"two-patients","disease":"measles",
           "symptoms":["fever"],"hospitalized":True,"evidence_quote":sentence,"case_span_quote":sentence,
           "case_span_start":0,"case_span_end":len(sentence)}
    second={**first,"case_span_start":len(sentence)+1,"case_span_end":len(text)}
    assert stable_record_id(first,chunk)!=stable_record_id(second,chunk)


def test_derived_annotation_replay_merges_once_and_preserves_original_record_id():
    row,chunk=observation()
    original={**row,"record_id":"original-evidence-record"}
    incoming={**row,"record_id":"another-batch-id","extraction_confidence":0.2,
              "conflict_ids":["new-conflict"],"record_consistency_warnings":["review-note"]}
    state={"raw_records":[original],"evidence_chunks":[chunk],"source_registry":[],"documents":[]}
    merged=merge_recovery_delta(state,RecoveryDelta(records=[incoming]))
    assert len(merged["raw_records"])==1
    assert merged["raw_records"][0]["record_id"]=="original-evidence-record"
    assert merge_recovery_delta(merged,RecoveryDelta(records=[incoming]))["raw_records"]==merged["raw_records"]
    assert state["raw_records"]==[original]


def test_conflicting_count_survives_collision_repair_and_repeated_delta():
    row,chunk=observation()
    original={**row,"record_id":"colliding-old-id"}
    different={**row,"record_id":"colliding-old-id","cases_confirmed":13}
    state={"raw_records":[original],"evidence_chunks":[chunk],"source_registry":[],"documents":[]}
    first=merge_recovery_delta(state,RecoveryDelta(records=[different]))
    repeated=merge_recovery_delta(first,RecoveryDelta(records=[different]))
    assert len(repeated["raw_records"])==2
    assert {record["cases_confirmed"] for record in repeated["raw_records"]}=={12,13}
    assert len({record["record_id"] for record in repeated["raw_records"]})==2
    assert next(record for record in repeated["raw_records"] if record["cases_confirmed"]==12)["record_id"]=="colliding-old-id"


def test_different_patient_spans_survive_merge_under_one_original_chunk():
    sentence="The patient had fever and was hospitalized."
    text=sentence+" "+sentence
    chunk={"source_id":"clinical","chunk_id":"two-patients","document_id":"document",
           "document_hash":sha256(text.encode()).hexdigest(),"text":text,"char_start":0,"char_end":len(text)}
    first={"source_id":"clinical","supporting_chunk_id":"two-patients","disease":"measles",
           "symptoms":["fever"],"hospitalized":True,"evidence_quote":sentence,"case_span_quote":sentence,
           "record_id":"patient-A","case_span_start":0,"case_span_end":len(sentence)}
    second={**first,"record_id":"patient-B","case_span_start":len(sentence)+1,"case_span_end":len(text)}
    state={"raw_records":[first],"evidence_chunks":[chunk],"source_registry":[],"documents":[]}
    merged=merge_recovery_delta(state,RecoveryDelta(records=[second]))
    assert len(merged["raw_records"])==2
    assert {record["record_id"] for record in merged["raw_records"]}=={"patient-A","patient-B"}


def test_legacy_merge_keeps_its_existing_annotation_sensitive_behavior(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE","legacy")
    row,chunk=observation()
    original={**row,"record_id":"legacy-1"}
    annotated={**row,"record_id":"legacy-2","conflict_ids":["legacy-conflict"]}
    state={"raw_records":[original],"evidence_chunks":[chunk],"source_registry":[],"documents":[]}
    merged=merge_recovery_delta(state,RecoveryDelta(records=[annotated]))
    assert len(merged["raw_records"])==2
