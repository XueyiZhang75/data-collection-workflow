"""Cross-task evidence checks for defects observed in the live run."""
import hashlib
import pytest
from data_collection_workflow.evidence_qualification import assess_record_evidence


def assess(text, **changes):
    row=dict(record_id='r',chunk_id='c',disease='Measles',country='France',reporting_period='2024',cases_confirmed=12)
    row.update(changes)
    doc={'source_id':'s','clean_text':text,'content_hash':hashlib.sha256(text.encode()).hexdigest()}
    return assess_record_evidence(row,contract={},evidence_index={'documents':[doc],'evidence_chunks':{'c':{'source_id':'s','chunk_id':'c','text':text}}})


@pytest.mark.parametrize('disease,left,right',[('Measles','France','Germany'),('Pertussis','Canada','Brazil')])
@pytest.mark.parametrize('reverse',[False,True])
def test_multiple_counts_in_one_sentence_do_not_cross_bind(disease,left,right,reverse):
    clauses=[f'{left} reported 12 confirmed cases',f'{right} reported 7 confirmed cases']
    if reverse: clauses.reverse()
    text=f'{disease} during 2024: '+ ' and '.join(clauses)+'.'
    q=assess(text,disease=disease,country=right)
    assert q.status=='candidate'
    assert any(reason.startswith('cases_confirmed:') for reason in q.reasons)


def test_single_explicit_count_remains_qualified():
    assert assess('Measles in France: 12 confirmed cases during 2024.').status=='qualified'


def test_number_embedded_with_explicit_disease_name_remains_bound():
    assert assess('France reported 12 confirmed measles cases during 2024.').status=='qualified'


def test_case_group_cannot_become_one_qualified_patient():
    text='Patients A and B in France had measles during 2024: 2 confirmed cases.'
    q=assess(text,workflow_case_label='Patients A and B',cases_confirmed=2)
    assert q.status=='candidate' or q.product_kind.value=='aggregate'


@pytest.mark.parametrize('disease,country', [('Measles','France'),('Pertussis','Canada')])
def test_official_rule_does_not_borrow_country_from_title(monkeypatch,disease,country):
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    chunk={'source_id':'s','chunk_id':'c','text':f'{disease}: 12 confirmed cases during 2024.',
           'title':f'{country} health agency update','source_url':'https://example.org/report',
           'source_type':'official_public_health_agency'}
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    row,diagnostic=_official_outbreak_record_from_chunk(chunk,1,policy,{'disease_standard_name':disease,'disease_terms':[disease],'structured_task':{'disease':disease}})
    assert row is None or row.country is None


@pytest.mark.parametrize('label,source_name,task_name',[
    ('Mpox (Monkeypox)','mpox','mpox'),
    ('Mpox (Monkeypox)','monkeypox','mpox'),
    ('Whooping cough','pertussis','Pertussis'),
    ('COVID-19','coronavirus disease 2019','COVID-19'),
])
def test_verified_disease_names_share_identity_without_filling_facts(label,source_name,task_name):
    text=f'France reported 12 confirmed cases of {source_name} during 2024.'
    q=assess(text,disease=label)
    assert q.status=='qualified',q.reasons
    from data_collection_workflow.evidence_qualification import _constraint_reasons
    assert _constraint_reasons({'disease':label},{'disease':task_name})==[]


@pytest.mark.parametrize('label,task',[
    ('Mpox (clade I)','mpox'),('Mpox (measles)','mpox'),
    ('Hantavirus pulmonary syndrome','hemorrhagic fever with renal syndrome'),
    ('Influenza-like illness','Influenza'),('H5N1','Influenza'),
    ('Novel disease (unknown)','Novel disease'),
])
def test_disease_names_do_not_merge_qualifiers_pathogens_or_unknown_aliases(label,task):
    from data_collection_workflow.evidence_qualification import _constraint_reasons
    assert 'disease:contract_mismatch' in _constraint_reasons({'disease':label},{'disease':task})


def test_missing_or_wrong_disease_still_fails_source_support():
    assert assess('France reported 12 confirmed cases during 2024.',disease='Mpox (Monkeypox)').status=='candidate'
    assert assess('France reported 12 confirmed measles cases during 2024.',disease='Mpox (Monkeypox)').status=='candidate'


@pytest.mark.parametrize('reverse',[False,True])
def test_official_rule_binds_country_to_selected_count_sentence(monkeypatch,reverse):
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    sentences=['Mpox in Brazil caused 12 confirmed cases during 2025.', 'Mpox in Argentina caused 7 confirmed cases during 2025.']
    if reverse: sentences.reverse()
    chunk={'source_id':'s','chunk_id':'c','text':' '.join(sentences),'title':'United States health agency report','source_url':'https://example.org/report','source_type':'official_public_health_agency'}
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    row,_=_official_outbreak_record_from_chunk(chunk,1,policy,{'disease_standard_name':'Mpox','disease_terms':['mpox']})
    assert row is not None
    assert row.cases_confirmed==12
    assert row.cases_unspecified is None
    assert row.country=='Brazil'


def test_partial_official_rule_result_does_not_block_llm_extraction(monkeypatch):
    from data_collection_workflow.config import load_structured_extraction_policy, load_llm_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy, LLMStructuredExtractionPolicy, LLMExtractionOutput
    from data_collection_workflow.nodes import extraction
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LLM_HARD_PRIMARY_CALLS','2')
    monkeypatch.setenv('LLM_FOCUSED_RECOVERY_ENABLED','false')
    chunk={'source_id':'s','chunk_id':'c','text':'Mpox in Brazil caused 12 confirmed cases during 2025.',
           'title':'Mpox outbreak report', 'source_url':'https://www.who.int/emergencies/disease-outbreak-news/item/local-test',
           'source_type':'official_public_health_agency','chunk_kind':'text','fetch_purpose':'data_extraction',
           'contains_target_data':True,'extraction_eligible_for_task_disease':True,'disease_relevance_status':'target_disease_match',
           'source_role_final':'collection'}
    context={'disease_standard_name':'Mpox','disease_terms':['mpox'],'structured_task':{'disease':'mpox'},'guard_enabled':False}
    calls=[]
    def extract(chunk,policy):
        calls.append(chunk['chunk_id'])
        return LLMExtractionOutput(chunk_is_relevant=False,records=[])
    monkeypatch.setattr(extraction.llm_clients,'extract_chunk_with_llm',extract)
    monkeypatch.setattr(extraction.llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'offline-test'})
    records,stats=extraction._llm_extract_records_from_chunks([chunk],LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),StructuredExtractionPolicy(**load_structured_extraction_policy()),False,context)
    assert stats['official_outbreak_deterministic_record_count']>0
    assert calls,stats


@pytest.mark.parametrize('other', ['Germany 7', 'Germany reported 7 suspected cases', 'germany 7'])
def test_competing_scope_with_elliptical_or_different_metric_is_not_shared(other):
    text='Measles during 2024: France reported 12 confirmed cases and '+other+'.'
    assert assess(text,country='Germany').status=='candidate'


def test_canonical_disease_alias_binds_inside_metric_phrase():
    assert assess('France reported 12 confirmed mpox cases during 2024.',disease='Mpox (Monkeypox)').status=='qualified'

@pytest.mark.parametrize('disease,country,verb', [('Measles','France','identified'), ('Pertussis','Canada','recorded'), ('Dengue','Brazil','detected')])
def test_official_count_cue_accepts_observation_verbs_and_disease_modifiers(monkeypatch,disease,country,verb):
    from data_collection_workflow.nodes.extraction import _official_best_count_mentions
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    text=f'{country} {verb} 17 new {disease} cases during 2027. It reported 4 cases during 2026.'
    case,_,_ = _official_best_count_mentions(text, {'structured_task':{'disease':disease,'start_date':'2027-01-01','end_date':'2027-12-31'}})
    assert case is not None and case['value']==17


def test_contact_number_cannot_become_case_count(monkeypatch):
    from data_collection_workflow.nodes.extraction import _official_best_count_mentions
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    case,_,_ = _official_best_count_mentions('France identified 17 contacts of measles cases during 2027.', {})
    assert case is None


def test_evidence_rule_does_not_add_unproven_statistical_labels(monkeypatch):
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _official_outbreak_record_from_chunk
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    chunk={'source_id':'s','chunk_id':'c','text':'France reported 17 measles cases during 2027.',
           'source_url':'https://health.example/report','source_type':'official_public_health_agency'}
    record,_=_official_outbreak_record_from_chunk(chunk,1,StructuredExtractionPolicy(**load_structured_extraction_policy()),{'disease_standard_name':'Measles','disease_terms':['measles']})
    assert record is not None
    assert record.statistical_count_type is None and record.count_semantics is None
    assert record.case_definition is None


def test_evidence_semantics_are_not_borrowed_from_other_chunk_statements(monkeypatch):
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    quote='France reported 17 measles cases.'
    record={'disease':'Measles','cases_unspecified':17,'evidence_quote':quote}
    chunk={'text':quote+' Separately, cumulative tests totaled 200.'}
    actual=_apply_extraction_semantic_guardrails(record,chunk,{'disease_standard_name':'Measles','is_hantavirus':False})
    assert actual.get('statistical_count_type') is None


def test_evidence_retains_extracted_semantics_for_evidence_assessment(monkeypatch):
    from data_collection_workflow.nodes.extraction import _apply_extraction_semantic_guardrails
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    quote='France reported 17 measles cases during 2027.'
    record={'disease':'Measles','cases_unspecified':17,'evidence_quote':quote,'statistical_count_type':'annual'}
    actual=_apply_extraction_semantic_guardrails(record,{'text':quote},{'disease_standard_name':'Measles','is_hantavirus':False})
    assert actual.get('statistical_count_type')=='annual'
