"""Structural titles are context; actual observations still reach extraction."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.evidence_chunking import build_evidence_chunks
from data_collection_workflow.nodes.extraction import extraction_skip_reason
from data_collection_workflow.models import EvidenceChunk


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')


def _case(tmp_path,title):
    html=f'<html><body><h1>{title}</h1><p>Dengue surveillance in Brazil.</p></body></html>'
    doc=parse_response(html.encode(),url='https://public.example/report',source_id='s',
        session_dir=tmp_path,content_type='text/html')
    doc.update(quality_status='usable',extraction_readiness='ready',source_role_final='collection')
    loc=next(l for l in doc['locator_spans'] if l.get('role')=='heading')
    chunk={'chunk_id':'c','source_id':'s','text':doc['clean_text'][loc['char_start']:loc['char_end']],
        'char_start':loc['char_start'],'char_end':loc['char_end'],'document_hash':doc['content_hash'],
        'contains_target_data':True,'extraction_eligible_for_task_disease':True,
        'disease_relevance_status':'target_disease_match','extraction_readiness':'ready',
        'source_role_final':'collection'}
    state={'structured_task':{'disease':'dengue','location':'Brazil','start_date':'2025-01-01',
        'end_date':'2025-12-31'},'documents':[doc]}
    context={'documents_by_source_id':{'s':[doc]},'collection_mode':'direct_collection'}
    return doc,chunk,state,context


@pytest.mark.parametrize('title',[
    'Clinical characterization and severity of dengue cases in Brazil, 2025',
    'Demographics and clinical symptom data for hospitalized dengue patients in Brazil',
    'Situation Report: Dengue outbreak in Brazil - 3 September 2025',
    'Extended Data Fig. 3: Distribution of cluster sizes during the dengue outbreak',
])
def test_structural_title_is_not_independent_observation(tmp_path,title):
    doc,chunk,state,context=_case(tmp_path,title)
    assert extraction_skip_reason(chunk,context)=='structural_heading_without_observation'
    made=build_evidence_chunks(state)['evidence_chunks']
    heading=next(c for c in made if c['text']==chunk['text'])
    assert heading['contains_target_data'] is False
    assert heading['extraction_eligible_for_task_disease'] is False
    assert heading.get('source_is_heading') is True
    # The original heading remains a real, locatable context for later prose.
    assert any(c.get('bound_context_spans') for c in made if c['text']!=chunk['text'])


@pytest.mark.parametrize('title',[
    'Brazil reported 18 dengue cases as of 3 September 2025',
    'Brazil reported 2025 dengue cases as of 3 September 2025',
    'Dengue test positivity was 12% in Brazil during 2025',
    'A dengue patient was hospitalized in Brazil and later recovered',
    'Une patiente atteinte de dengue a été hospitalisée au Brésil',
    'First confirmed dengue case in Brazil',
])
def test_observation_in_heading_remains_eligible(tmp_path,title):
    _,chunk,_,context=_case(tmp_path,title)
    assert extraction_skip_reason(chunk,context) is None


def test_short_narrative_does_not_need_a_numeric_count(tmp_path):
    doc,chunk,_,context=_case(tmp_path,'Dengue surveillance in Brazil, 2025')
    # An uncertain narrative must not be discarded just for its length or lack of digits.
    chunk.update(text='The patient developed a rash and recovered.',char_start=None,char_end=None)
    assert extraction_skip_reason(chunk,context) is None


def test_heading_role_survives_model_roundtrip():
    row=EvidenceChunk(chunk_id='c',source_id='s',text='Dengue surveillance, 2025',source_is_heading=True)
    assert row.model_dump().get('source_is_heading') is True


def test_legacy_extraction_contract_is_unchanged(monkeypatch,tmp_path):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    _,chunk,_,context=_case(tmp_path,'Dengue surveillance in Brazil, 2025')
    assert extraction_skip_reason(chunk,context) is None
