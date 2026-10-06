import pytest
from data_collection_workflow.nodes.extraction import _detect_pathogen_or_syndrome

@pytest.mark.parametrize('term,text', [
 ('CHIK','Chikungunya cases increased.'),('FLU','The report discusses influence.'),
 ('DEN','The denominator was revised.'),('SAR','The research cohort was sampled.'),
 ('XV','The chapter was XVII.'),('AB','The CAB programme reported cases.'),
])
def test_evidence_short_alias_does_not_create_a_pathogen_from_a_larger_word(monkeypatch,term,text):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert _detect_pathogen_or_syndrome(text,{'pathogen_terms':[term]}) is None

@pytest.mark.parametrize('term,text',[
 ('CHIKV','Patients tested positive for CHIKV.'),('XV','XV: 12 cases.'),
 ('AB','The pathogen (AB) was isolated.'),('Zeta virus','Zeta virus was confirmed.'),
 ('Zeta virus','Zeta\nvirus was confirmed.'),('HCoV-OC43','HCoV-OC43 infection'),
 ('甲型流感病毒','检测出甲型流感病毒感染'),
])
def test_evidence_explicit_pathogen_alias_is_preserved(monkeypatch,term,text):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert _detect_pathogen_or_syndrome(text,{'pathogen_terms':[term]}) == term

@pytest.mark.parametrize('context',[None,{}, {'pathogen_terms':[]}])
def test_missing_profile_does_not_invent_pathogen(monkeypatch,context):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert _detect_pathogen_or_syndrome('There were 12 cases.',context) is None

def test_legacy_token_matching_is_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    assert _detect_pathogen_or_syndrome('influence',{'pathogen_terms':['FLU']}) == 'FLU'
