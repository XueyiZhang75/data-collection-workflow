"""Independent boundary review of the narrowly changed pathogen-token helper."""
import pytest
from data_collection_workflow.nodes.extraction import _detect_pathogen_or_syndrome as detect

@pytest.mark.parametrize('term,text',[
 ('CHIK','Chikungunya reported 12 cases.'),('CHIK','antiCHIK compounds were tested.'),
 ('DEN','The density increased.'),('FLU','A fluent discussion followed.'),
 ('SAR','sarcoma appears in the background.'),('AB','AB123 is the sample identifier.'),
 ('AB','123AB is the sample identifier.'),('HMPV','HMPVlike is a compound adjective.'),
])
def test_ascii_alias_not_inside_another_token(monkeypatch,term,text):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert detect(text,{'pathogen_terms':[term]}) is None


@pytest.mark.parametrize('term,text',[
 ('CHIKV','(CHIKV), detected'),('Zeta virus','Zeta\nvirus was isolated.'),
 ('Zeta virus','Zeta-virus was isolated.'),('Zeta virus','Zeta\r\nvirus was isolated.'),
 ('SARS-CoV-2','SARS-\nCoV-2 was detected.'),('HCoV-OC43','HCoV-OC43 was detected.'),
 ('甲型流感病毒','检测出甲型流感病毒感染'),('甲病毒','甲病毒检出'),
 ('Virus X','PCR confirmed VIRUS X.'),
])
def test_complete_terms_typography_and_cjk_remain_supported(monkeypatch,term,text):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert detect(text,{'pathogen_terms':[term]})==term


def test_short_term_order_cannot_shadow_present_long_alias(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert detect('CHIKV positive',{'pathogen_terms':['CHIK','CHIKV']})=='CHIKV'


@pytest.mark.parametrize('term,text',[('CHIK','chikungunya'),('FLU','influence'),('DEN','density')])
def test_legacy_substring_contract_is_preserved(monkeypatch,term,text):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    assert detect(text,{'pathogen_terms':[term]})==term


@pytest.mark.parametrize('ctx',[None,{}, {'pathogen_terms':[]},{'pathogen_terms':['',None]}])
def test_absent_profile_terms_do_not_create_values(monkeypatch,ctx):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert detect('There were 12 cases.',ctx) is None
