"""Numeric role evidence is shared across extraction, qualification and scheduling."""
import hashlib
import pytest
from data_collection_workflow.source_assertions import count_mentions, contextual_number_role
from data_collection_workflow.evidence_qualification import _supports, assess_record_evidence
from data_collection_workflow.nodes.extraction import _official_best_count_mentions, _chunk_has_strong_record_signal

@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv("PIPELINE_MODE", "evidence")

@pytest.mark.parametrize("disease,country", [("dengue","Brazil"),("measles","Canada")])
@pytest.mark.parametrize("template,value,role", [
    ("In 2025 {disease} cases were monitored in {country}.",2025,"year"),
    ("After 2 weeks {disease} cases require reassessment in {country} during 2025.",2,"duration"),
    ("Table 12 describes {disease} cases in {country} during 2025.",12,"table"),
    ("3 years since {disease} cases were first detected in {country} during 2025.",3,"duration"),
])
def test_explicit_role_rejects_all_count_consumers(template,value,role,disease,country):
    text=template.format(disease=disease,country=country)
    pos=text.index(str(value))
    assert contextual_number_role(text,pos,pos+len(str(value))) == role
    assert count_mentions(text) == []
    assert _official_best_count_mentions(text)[0] is None
    row={"disease":disease,"country":country,"reporting_period":"2025","cases_unspecified":value}
    assert _supports("cases_unspecified",value,text,row) is False
    digest=hashlib.sha256(text.encode()).hexdigest()
    doc={"source_id":"s","document_id":"d","text_hash":digest,"content_hash":digest,"clean_text":text}
    chunk={"source_id":"s","document_id":"d","chunk_id":"c","document_hash":digest,"text":text,"char_start":0,"char_end":len(text)}
    verdict=assess_record_evidence({**row,"record_id":"r","source_id":"s","supporting_chunk_id":"c"},contract={},evidence_index={"documents":[doc],"evidence_chunks":{"c":chunk}})
    assert verdict.status != "qualified"
    assert not next(e for e in verdict.field_evidence if e.field=="cases_unspecified").supported
    assert _chunk_has_strong_record_signal({"text":text}) is False

@pytest.mark.parametrize("text,value,role", [
    ("During 2025 dengue cases were monitored.",2025,"year"),
    ("Year 2025 dengue cases were monitored.",2025,"year"),
    ("En 2025 des cas ont ete signales.",2025,"year"),
    ("Tableau 12 decrit les cas.",12,"table"),
    ("Figure 12 describes cases.",12,"figure"),
    ("After 2 weeks cases were monitored.",2,"duration"),
    ("In 2025 cases were monitored.",2025,"year"),
])
def test_roles_do_not_bypass_old_direct_numeric_signal(text,value,role):
    pos=text.index(str(value))
    assert contextual_number_role(text,pos,pos+len(str(value))) == role
    assert count_mentions(text) == []
    assert _chunk_has_strong_record_signal({"text":text}) is False

@pytest.mark.parametrize("text,value", [
    ("Brazil reported 2025 dengue cases during 2024.",2025),
    ("The outbreak resulted in 2025 dengue cases in Brazil during 2024.",2025),
    ("Canada reported 12 confirmed measles cases.",12),
    ("Peru reported 3 confirmed cholera cases.",3),
    ("Brazil reported 18 000 suspected dengue cases.",18000),
    ("La Reunion rapporte 12 cas confirmes.",12),
    ("Table 12 describes 14 dengue cases in Brazil during 2025.",14),
])
def test_true_counts_remain_available(text,value):
    assert [m["value"] for m in count_mentions(text)] == [value]
    assert _official_best_count_mentions(text)[0]["value"] == value
    assert _chunk_has_strong_record_signal({"text":text}) is True
