import hashlib
import pytest
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index


def fixture(text='Pertussis in Washington, United States: 12 confirmed cases during 2024.'):
    doc={'source_id':'s','clean_text':text,'content_hash':hashlib.sha256(text.encode()).hexdigest()}
    chunk={'chunk_id':'c','source_id':'s','text':text,'char_start':0,'char_end':len(text)}
    row={'record_id':'r','source_id':'s','chunk_id':'c','disease':'Pertussis','country':'United States','subnational_location':'Washington','reporting_period':'2024','cases_confirmed':12}
    return row,build_evidence_index({'documents':[doc],'evidence_chunks':[chunk]})


def assess(row,index):
    return assess_record_evidence(row,contract={'disease':'Pertussis','country':'United States'},evidence_index=index)


def test_bound_observation_qualifies_as_aggregate():
    row,index=fixture()
    q=assess(row,index)
    assert q.status=='qualified'
    assert q.product_kind=='aggregate'


@pytest.mark.parametrize('change', [ {'country':'France'}, {'reporting_period':'2023'}, {'cases_confirmed':13} ])
def test_mutated_fact_fails_closed(change):
    row,index=fixture(); row.update(change)
    assert assess(row,index).status!='qualified'


@pytest.mark.parametrize('text',[
    'Pertussis in Washington, United States: 12 deaths during 2024.',
    'Pertussis in Washington, United States: section 12.4 confirmed cases during 2024.',
    'Published 2024. Pertussis in Washington, United States: 12 confirmed cases.',
    'Pertussis in Washington, United States during 2024. '+('Background. '*100)+'12 confirmed cases.',
])
def test_semantic_false_positives(text):
    row,index=fixture(text)
    assert assess(row,index).status!='qualified'


def test_task_and_review_never_supply_evidence(monkeypatch):
    from data_collection_workflow.run_quality_gates import apply_run_quality_gates
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    for enabled in (True,False):
        result=apply_run_quality_gates({'normalized_records':[row],'human_review_enabled':enabled,'structured_task':row})
        assert result['final_dataset']==[]
        assert result['candidate_records']


def test_bound_table_header_qualifies():
    text='Pertussis cases in United States during 2024\nState | Confirmed cases\nWashington | 12'
    row,index=fixture(text)
    index['evidence_chunks']['c'].update(table_id='t',row_id='r',row_quote='Washington | 12',table_header='State | Confirmed cases',heading_context='Pertussis cases in United States during 2024')
    assert assess(row,index).status=='qualified'


def test_hash_is_mandatory():
    row,index=fixture(); index['documents'][0].pop('content_hash')
    assert assess(row,index).status!='qualified'

def test_evidence_final_package_preserves_contract_and_never_creates_cases(monkeypatch):
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    from data_collection_workflow.models import FinalDataPackage
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    result=final_data_package_builder({'normalized_records':[row], 'documents':index['documents'], 'evidence_chunks':list(index['evidence_chunks'].values())})
    assert len(result['final_data_package']['final_dataset'])==1
    assert result['final_case_dataset']==[]
    assert len(result['aggregate_dataset'])==1
    FinalDataPackage(**result['final_data_package'])


def test_tampered_document_hash_fails():
    row,index=fixture(); index['documents'][0]['clean_text']+=' appended'
    assert assess(row,index).status=='candidate'


def test_unknown_and_non_us_geography_stays_uninferred(monkeypatch):
    from data_collection_workflow.geography import resolve_record_geography
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row=resolve_record_geography({}, source_identity={'jurisdiction_name':'Ontario','jurisdiction_level':'subnational'}, task={'location':'Washington'})
    assert not row.get('country')
    assert not row.get('subnational_location')


def test_coverage_cannot_be_filled_by_raw_records(monkeypatch):
    from data_collection_workflow.source_coverage import build_source_coverage_audit
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    req=[{'requirement_id':'wa','disease':'Pertussis','subnational_location':'Washington','reporting_period':'2024'}]
    assert build_source_coverage_audit(req,[],evidence_rows=[row])['accepted_requirement_count']==0
    row['evidence_qualification']=assess(row,index).to_dict()
    assert build_source_coverage_audit(req,[],evidence_rows=[row])['accepted_requirement_count']==1


@pytest.mark.parametrize('mode',['direct','full','fast','standard'])
def test_modes_use_same_qualification(monkeypatch,mode):
    from data_collection_workflow.run_quality_gates import apply_run_quality_gates
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    state={'normalized_records':[row],'documents':index['documents'],'evidence_chunks':list(index['evidence_chunks'].values()),'collection_mode':mode,'human_review_enabled':False}
    assert len(apply_run_quality_gates(state)['final_dataset'])==1
    row['cases_confirmed']=14
    assert apply_run_quality_gates(state)['final_dataset']==[]

def test_unrelated_adjacent_sentences_do_not_bind_disease_to_count():
    row,index=fixture('Pertussis in Washington, United States during 2024. Influenza caused 12 confirmed cases.')
    assert assess(row,index).status!='qualified'


def test_table_wrong_column_is_not_cases():
    text='Pertussis in United States during 2024\nState | Confirmed cases | Deaths\nWashington | 100 | 12'
    row,index=fixture(text)
    assert assess(row,index).status!='qualified'

@pytest.mark.parametrize('extra',[{'age':'99'},{'gender':'female'},{'date_death':'2025-01-01'},{'outcome':'dead'},{'date_onset':'2025-01-01'},{'date_reported':'2025-01-01'}])
def test_all_substantive_fields_need_evidence(extra):
    row,index=fixture();row.update(extra)
    assert assess(row,index).status=='candidate'


def test_competing_disease_same_clause_is_ambiguous():
    row,index=fixture('Pertussis and influenza in Washington, United States during 2024: influenza caused 12 confirmed cases.')
    assert assess(row,index).status=='candidate'


@pytest.mark.parametrize('contract',[{'location':'Virginia'}, {'start_date':'2025-01-01','end_date':'2025-12-31'}, {'task_period_start':'2025-01-01','task_period_end':'2025-12-31'}, {'requirements':[{'location':'Virginia','period_start':'2025-01-01','period_end':'2025-12-31'}]}])
def test_actual_contract_scope_is_enforced(contract):
    row,index=fixture()
    assert assess_record_evidence(row,contract=contract,evidence_index=index).status=='candidate'


@pytest.mark.parametrize('req',[{'location':'Virginia'},{'period_start':'2025-01-01','period_end':'2025-12-31'}, {'accepted_metric_families':['death_count']},{'geography':'Virginia'}])
def test_actual_coverage_constraints_are_enforced(req):
    from data_collection_workflow.evidence_qualification import qualified_coverage
    row,index=fixture();row['evidence_qualification']=assess(row,index).to_dict()
    assert qualified_coverage([{'requirement_id':'x','disease':'Pertussis',**req}],[row])['accepted_requirement_count']==0


def test_review_exclusion_removes_all_admitted_views(monkeypatch):
    from data_collection_workflow.run_quality_gates import apply_run_quality_gates
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    result=apply_run_quality_gates({'normalized_records':[row],'documents':index['documents'],'evidence_chunks':list(index['evidence_chunks'].values()),'records_excluded_by_human_review':[row]})
    assert result['qualified_aggregate_records']==[]
    assert result['qualified_records']==[]


def test_canonical_extraction_chunk_id_is_resolved():
    row,index=fixture();row['supporting_chunk_id']=row.pop('chunk_id')
    assert assess(row,index).status=='qualified'

def test_real_extraction_normalization_qualification(monkeypatch):
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output
    from data_collection_workflow.nodes.normalization import _normalize_record
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy, RecordNormalizationPolicy
    from data_collection_workflow.config import load_llm_structured_extraction_policy, load_record_normalization_policy
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    row,index=fixture()
    extracted=_build_record_from_llm_output(LLMExtractedRecord(**row),index['evidence_chunks']['c'],1,LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),{},context={"disease_standard_name":"Pertussis","is_hantavirus":False})
    assert extracted is not None
    normalized,_=_normalize_record(extracted.model_dump(),RecordNormalizationPolicy(**load_record_normalization_policy()))
    q=assess(normalized,index)
    assert q.status=='qualified',q.reasons


def test_uncertain_ocr_stays_candidate():
    row,index=fixture();index['documents'][0]['quality_issues']=['low_ocr_confidence_candidate']
    assert assess(row,index).status=='candidate'

def test_unknown_coverage_constraint_stays_gap():
    from data_collection_workflow.evidence_qualification import qualified_coverage
    row,index=fixture();row['evidence_qualification']=assess(row,index).to_dict()
    assert qualified_coverage([{'requirement_id':'x','disease':'Pertussis','age_group':'children'}],[row])['accepted_requirement_count']==0

def test_lower_bound_is_not_admitted_as_exact_count():
    row,index=fixture('Pertussis in Washington, United States: at least 12 confirmed cases during 2024.')
    assert assess(row,index).status=='candidate'

@pytest.mark.parametrize('field,value',[('age','12'),('date_death','2024'),('date_onset','2024'),('date_confirmation','2024'),('date_reported','2024'),('gender','Washington'),('outcome','confirmed'),('nationality','United States'),('symptoms','Pertussis'),('workflow_case_label','12')])
def test_clinical_values_cannot_borrow_other_field_tokens(field,value):
    row,index=fixture();row[field]=value
    assert assess(row,index).status=='candidate'


def test_positive_role_bound_clinical_sentence():
    text='Patient A, aged 12, gender: female, in Washington, United States had Pertussis during 2024; symptom onset on 2024-01-02, confirmed on 2024-01-03, died on 2024-01-04.'
    row,index=fixture(text);row.pop('cases_confirmed')
    row.update(workflow_case_label='Patient A',age='12',gender='female',date_onset='2024-01-02',date_confirmation='2024-01-03',date_death='2024-01-04')
    assert assess(row,index).status=='qualified'

@pytest.mark.parametrize('field',['date_onset','date_death','date_confirmation'])
def test_clinical_date_cannot_borrow_report_date(field):
    row,index=fixture('Pertussis in Washington, United States: 12 confirmed cases during 2024, reported on 2024-02-01.')
    row[field]='2024-02-01'
    assert assess(row,index).status=='candidate'


@pytest.mark.parametrize('phrase',['aged 12','age: 12','12 years old','12-year-old'])
def test_positive_explicit_age_roles(phrase):
    row,index=fixture('Patient A, '+phrase+', in Washington, United States had Pertussis during 2024.')
    row.pop('cases_confirmed');row.update(workflow_case_label='Patient A',age='12')
    assert assess(row,index).status=='qualified'


def test_geographic_scope_type_has_evidence_derived_proof():
    row,index=fixture();row.update(geographic_scope='Washington',geographic_scope_type='subnational')
    q=assess(row,index)
    assert q.status=='qualified',q.reasons
    assert any(e.field=='geographic_scope_type' and e.supported for e in q.field_evidence)


def test_structured_partition_requires_explicit_scope_and_exclusion_proof():
    text='Pertussis in Washington, United States during 2024. Population groups Adults and Children are mutually exclusive and exhaustive for All residents; confirmed cases.'
    row,index=fixture(text);row.pop('cases_confirmed')
    row['disjoint_partition']={'axis':'population_scope','members':['Adults','Children'],'target':'All residents','metric':'cases_confirmed','exhaustive':True,'mutually_exclusive':True}
    q=assess(row,index)
    assert q.status=='qualified',q.reasons
    assert any(e.field=='disjoint_partition' and e.supported for e in q.field_evidence)
    row['disjoint_partition']['members']=['Adults','Invented']
    assert assess(row,index).status=='candidate'


def test_patient_links_require_explicit_same_patient_citation():
    text='Patient A in Washington, United States had Pertussis during 2024; Patient A is the same patient as Patient B at https://example.org/b.'
    row,index=fixture(text);row.pop('cases_confirmed');row.update(workflow_case_label='Patient A',case_span_quote=text)
    target='Patient B had Pertussis during 2024.'
    target_hash=hashlib.sha256(target.encode()).hexdigest()
    index['documents'].append({'source_id':'b','content_hash':target_hash,'clean_text':target,'url':'https://example.org/b'})
    row['patient_links']=[{'identity':'Patient B','document_hash':target_hash}]
    q=assess(row,index)
    assert q.status=='qualified',q.reasons
    assert any(e.field=='patient_links' and e.supported for e in q.field_evidence)
    row['patient_links'][0]['identity']='Patient Z'
    assert assess(row,index).status=='candidate'


def test_extraction_models_preserve_structural_proofs():
    from data_collection_workflow.models import LLMExtractedRecord, PublicHealthRecord
    link=[{'identity':'Patient B','document_hash':'a'*64}]
    raw=LLMExtractedRecord(disease='Pertussis',patient_links=link)
    assert raw.model_dump().get('patient_links')==link
    record=PublicHealthRecord(record_id='r',disease='Pertussis',patient_links=link)
    assert record.model_dump().get('patient_links')==link


def test_structural_proof_survives_actual_extraction_and_product_building(monkeypatch):
    from data_collection_workflow.nodes.extraction import _build_record_from_llm_output
    from data_collection_workflow.nodes.normalization import _normalize_record
    from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy, RecordNormalizationPolicy
    from data_collection_workflow.config import load_llm_structured_extraction_policy, load_record_normalization_policy
    from data_collection_workflow.evidence_products import build_evidence_products
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    text='Pertussis in Washington, United States during 2024. Population groups Adults and Children are mutually exclusive and exhaustive for All residents; confirmed cases.'
    row,index=fixture(text);row.pop('cases_confirmed')
    proof={'axis':'population_scope','members':['Adults','Children'],'target':'All residents','metric':'cases_confirmed','exhaustive':True,'mutually_exclusive':True}
    row.update(disjoint_partition=proof,geographic_scope='Washington',geographic_scope_type='subnational')
    extracted=_build_record_from_llm_output(LLMExtractedRecord(**row),index['evidence_chunks']['c'],1,LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy()),{},context={'disease_standard_name':'Pertussis','is_hantavirus':False})
    assert extracted is not None
    normalized,_=_normalize_record(extracted.model_dump(),RecordNormalizationPolicy(**load_record_normalization_policy()))
    assert normalized['disjoint_partition']==proof
    q=assess(normalized,index)
    assert q.status=='qualified',q.reasons
    normalized['evidence_qualification']=q.to_dict()
    product=build_evidence_products([normalized],evidence_index=index)
    assert not product['excluded_observations']


@pytest.mark.parametrize('field,value',[('age','45'),('gender','female'),('country','Canada'),('date_onset','2024-03-01')])
def test_case_fields_cannot_borrow_later_patients(field,value):
    text='Patient A in Washington, United States had Pertussis during 2024. Patient B, aged 45, gender: female, in Ontario, Canada had Pertussis during 2024; symptom onset on 2024-03-01.'
    row,index=fixture(text);row.pop('cases_confirmed');row.update(workflow_case_label='Patient A',case_span_quote=text)
    row[field]=value
    assert assess(row,index).status=='candidate'


@pytest.mark.parametrize('qualifier',['not ', 'possibly ', 'never '])
def test_negated_or_uncertain_structural_proof_is_candidate(qualifier):
    text='Pertussis in Washington, United States during 2024. Adults and Children are '+qualifier+'mutually exclusive and exhaustive for All residents; confirmed cases.'
    row,index=fixture(text);row.pop('cases_confirmed')
    row['disjoint_partition']={'axis':'population_scope','members':['Adults','Children'],'target':'All residents','metric':'cases_confirmed','exhaustive':True,'mutually_exclusive':True}
    assert assess(row,index).status=='candidate'


@pytest.mark.parametrize('relation',['is not the same patient as','may be the same patient as','is possibly the same patient as'])
def test_negative_or_uncertain_patient_relation_never_qualifies(relation):
    text='Patient A in Washington, United States had Pertussis during 2024; Patient A '+relation+' Patient B at https://example.org/b.'
    row,index=fixture(text);row.pop('cases_confirmed');row.update(workflow_case_label='Patient A',case_span_quote=text)
    target='Patient B had Pertussis during 2024.';digest=hashlib.sha256(target.encode()).hexdigest()
    index['documents'].append({'source_id':'b','content_hash':digest,'clean_text':target,'url':'https://example.org/b'})
    row['patient_links']=[{'identity':'Patient B','document_hash':digest}]
    assert assess(row,index).status=='candidate'


def test_patient_comparison_is_not_subject_identity():
    text='Patient A in Washington, United States had Pertussis during 2024. Patient B, unlike Patient A, was aged 45.'
    row,index=fixture(text);row.pop('cases_confirmed');row.update(workflow_case_label='Patient A',age='45',case_span_quote=text)
    assert assess(row,index).status=='candidate'


def test_reparsed_document_resolves_exact_chunk_version():
    row,index=fixture()
    current=index['documents'][0]
    current.update(document_id='parsed-new',text_hash=current['content_hash'],content_hash='a'*64,parser_version='evidence')
    previous={**current,'document_id':'parsed-old','clean_text':'unreadable','text_hash':hashlib.sha256(b'unreadable').hexdigest(),'parser_version':'v2'}
    index['documents'].append(previous)
    index['evidence_chunks']['c'].update(document_id='parsed-new',document_hash='a'*64)
    assert assess(row,index).status=='qualified'
    index['evidence_chunks']['c']['document_id']='missing-version'
    assert assess(row,index).status=='candidate'
