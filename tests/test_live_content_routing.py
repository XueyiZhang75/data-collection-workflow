"""Fetched evidence can correct advisory search metadata without changing qualification."""
import pytest
from data_collection_workflow.nodes.content_processing import document_quality_check, evidence_chunking_and_data_presence_flagging


def state(text='Pertussis in Canada: 12 confirmed cases during 2024.', **source_fields):
    source={'source_id':'s','title':'Surveillance report','source_role':'context_source',
            'source_role_final':'context','fetch_purpose':'context_grounding',
            'source_type_final':'unknown','source_identity_unverified':True,**source_fields}
    doc={**source,'document_id':'d','clean_text':text,'parse_status':'parsed','fetch_status':'success',
         'content_readable':True,'acquisition_status':'readable','request_success':True,'http_status_code':200}
    return {'structured_task':{'disease':'Pertussis','location':'Canada','start_date':'2024-01-01','end_date':'2024-12-31'},
            'documents':[doc],'source_registry':[source]}


def test_actual_data_content_reopens_extraction_without_promoting_trust(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    initial=state()
    update=document_quality_check(initial)
    doc=update['documents'][0]
    assert doc['extraction_readiness']=='ready'
    assert doc['source_role_final']=='collection'
    assert doc['fetch_purpose']=='data_extraction'
    assert doc['metadata']['content_routing_revision']['previous_source_role_final']=='context'
    source=update['source_registry'][0]
    assert source['source_identity_unverified'] is True and source['source_type_final']=='unknown'
    assert source['source_role_final']=='collection'
    chunks=evidence_chunking_and_data_presence_flagging({**initial,**update})['evidence_chunks']
    assert chunks and all(c['fetch_purpose']=='data_extraction' for c in chunks)


@pytest.mark.parametrize('changes',[{'source_role_final':'validation'}, {'source_role_final':'validation_reserved'}, {'source_role':'validation_source'},
                                   {'source_role_final':'excluded'}, {'source_role':'context_only'}, {'routing_flags':['blocked_from_structured_extraction']},
                                   {'routing_flags':['context_only']}])
def test_explicit_extraction_exclusions_remain_protected(monkeypatch,changes):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    initial=state(**changes)
    doc=document_quality_check(initial)['documents'][0]
    assert doc['fetch_purpose']=='context_grounding'


def test_policy_context_only_source_stays_context(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setattr('data_collection_workflow.nodes.content_processing.load_source_role_policy',lambda:{'context_only_source_ids':['s']})
    assert document_quality_check(state())['documents'][0]['fetch_purpose']=='context_grounding'


@pytest.mark.parametrize('text',['Cholera in Canada: 12 confirmed cases during 2024.', 'Pertussis is a respiratory infection.'])
def test_background_or_wrong_disease_cannot_reopen_extraction(monkeypatch,text):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    assert document_quality_check(state(text))['documents'][0]['source_role_final']=='context'


def test_legacy_context_routing_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    assert document_quality_check(state())['documents'][0]['source_role_final']=='context'
