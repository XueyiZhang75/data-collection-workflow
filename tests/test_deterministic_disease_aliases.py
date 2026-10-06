"""Exact shared disease names survive deterministic extraction and schema gates."""
import pytest

from data_collection_workflow.config import load_structured_extraction_policy
from data_collection_workflow.models import StructuredExtractionPolicy
from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
from data_collection_workflow.nodes.extraction import (
    _build_extraction_context, _official_hantavirus_alias_match,
    _official_outbreak_record_from_chunk, structured_extraction, schema_validation_and_repair,
)
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index


@pytest.fixture(autouse=True)
def offline_evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')


def make_state(tmp_path,task,text):
    doc=parse_response(text.encode(),url='https://authority.invalid/bulletin',source_id='source',
                       session_dir=tmp_path,content_type='text/plain')
    doc.update(document_id='document',quality_status='usable',extraction_readiness='ready',
               source_type='official_public_health',fetch_purpose='data_extraction',source_role_final='collection')
    state={'structured_task':{'disease':task,'location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31',
                             'collection_mode':'direct_collection'},'documents':[doc]}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    return state


@pytest.mark.parametrize('task,literal',[('mpox','Monkeypox'),('pertussis','Whooping cough'),('measles','Rubeola')])
def test_exact_alias_reaches_official_helper_and_public_schema(tmp_path,task,literal):
    text=f'{literal} in Canada: 12 cases were reported during 2025.'
    state=make_state(tmp_path,task,text)
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    context=_build_extraction_context(state,policy)
    chunk=next(c for c in state['evidence_chunks'] if text in c['text'])
    record,diagnostic=_official_outbreak_record_from_chunk(chunk,1,policy,context)
    assert record is not None,diagnostic
    schema=schema_validation_and_repair({**state,'raw_records':[record.model_dump()]})
    assert schema['validated_records'],schema['rejected_records']
    assert not any('disease_mismatch' in str(r.get('validation_errors')) for r in schema['validated_records'])
    assert assess_record_evidence(record.model_dump(),contract={},evidence_index=build_evidence_index(state)).status=='qualified'
    public=structured_extraction(state)
    assert public['raw_records']
    assert any(str(r.get('disease_alias_used') or '').casefold()==literal.casefold() for r in public['raw_records'])
    public_schema=schema_validation_and_repair({**state,**public})
    assert public_schema['validated_records'],public_schema['rejected_records']


def test_saved_source_minimal_heading_uses_existing_name_alias():
    # Exact heading from the saved CDC document; no count/reference answer is supplied.
    heading='Monkeypox Virus Surveillance \u2014 United States, 2024\u20132025'
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    context=_build_extraction_context({'structured_task':{'disease':'mpox'}},policy)
    assert _official_hantavirus_alias_match(heading,context)[0]


@pytest.mark.parametrize('literal',['MPXV','Monkeypoxlike syndrome','Measles'])
def test_unproved_or_partial_names_do_not_become_exact_task_aliases(literal):
    policy=StructuredExtractionPolicy(**load_structured_extraction_policy())
    context=_build_extraction_context({'structured_task':{'disease':'mpox'}},policy)
    assert not _official_hantavirus_alias_match(literal+' in Canada: 12 cases during 2025.',context)[0]
