"""Independent parser-to-eligibility controls for actual heading observations."""
import socket
import pytest
from test_acquisition_structural_eligibility import _case
from data_collection_workflow.evidence_chunking import build_evidence_chunks
from data_collection_workflow.nodes.extraction import extraction_skip_reason


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('LANGSMITH_TRACING','false')
    monkeypatch.setenv('LANGCHAIN_TRACING_V2','false')
    def denied(*args, **kwargs): raise AssertionError('External network forbidden')
    monkeypatch.setattr(socket.socket,'connect',denied)


@pytest.mark.parametrize('title',[
    'Brazil reported 2025 dengue cases as of 3 September 2025',
    'Dengue test positivity was 12% in Brazil during 2025',
    'A dengue patient was hospitalized in Brazil and later recovered',
    'Une patiente atteinte de dengue a \u00e9t\u00e9 hospitalis\u00e9e au Br\u00e9sil',
    'First confirmed dengue case in Brazil',
    'No dengue deaths were reported in Brazil during 2025',
])
def test_actual_observation_heading_is_preserved_by_chunk_builder(tmp_path,title):
    doc,original,state,context=_case(tmp_path,title)
    made=build_evidence_chunks(state)['evidence_chunks']
    chunk=next(row for row in made if row['text']==original['text'])
    assert chunk['contains_target_data'] is True
    assert chunk['extraction_eligible_for_task_disease'] is True
    assert extraction_skip_reason(chunk,context) is None
    assert doc['clean_text'][chunk['char_start']:chunk['char_end']]==title
    assert chunk['document_hash']==doc['content_hash']


def test_old_other_document_heading_cannot_reclassify_a_new_span(tmp_path):
    _,chunk,_,context=_case(tmp_path,'Dengue surveillance in Brazil, 2025')
    chunk['document_hash']='different-original-document'
    assert extraction_skip_reason(chunk,context) is None


def test_other_source_heading_cannot_reclassify_same_named_text(tmp_path):
    doc,chunk,_,context=_case(tmp_path,'Dengue surveillance in Brazil, 2025')
    context['documents_by_source_id']={'other-source':[{**doc,'source_id':'other-source'}]}
    assert extraction_skip_reason(chunk,context) is None



def test_potential_observation_heading_does_not_override_wrong_disease(tmp_path):
    _,original,state,context=_case(tmp_path,'Brazil reported 18 measles cases during 2025')
    made=build_evidence_chunks(state)['evidence_chunks']
    chunk=next(row for row in made if row['text']==original['text'])
    assert chunk['contains_target_data'] is False
    assert chunk['extraction_eligible_for_task_disease'] is False
    assert extraction_skip_reason(chunk,context) is not None


@pytest.mark.parametrize('observed',[False,True])
def test_inline_heading_parts_use_their_shared_source_heading(tmp_path, observed):
    from data_collection_workflow.document_acquisition import parse_response
    middle='12 cases' if observed else 'disease severity'
    raw=('<h1>Dengue clinical characterization: <em>'+middle+'</em> in Brazil during 2025</h1>'
         '<p>Dengue surveillance in Brazil.</p>').encode()
    doc=parse_response(raw,url='https://public.example/report',source_id='inline',
        session_dir=tmp_path,content_type='text/html')
    doc.update(quality_status='usable',extraction_readiness='ready',source_role_final='collection')
    state={'structured_task':{'disease':'dengue','location':'Brazil','start_date':'2025-01-01','end_date':'2025-12-31'},'documents':[doc]}
    chunks=build_evidence_chunks(state)['evidence_chunks']
    headings=[c for c in chunks if c.get('source_is_heading')]
    assert len(headings)==3
    for c in headings:
        assert doc['clean_text'][c['char_start']:c['char_end']]==c['text']
    if observed:
        chunk=next(c for c in headings if c['text']==middle)
        assert chunk['contains_target_data'] is True
        assert chunk['extraction_eligible_for_task_disease'] is True
    else:
        assert all(not c['contains_target_data'] for c in headings)
