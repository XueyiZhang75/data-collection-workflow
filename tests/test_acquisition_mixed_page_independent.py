"""Independent routing checks; no provider or historic-session dependencies."""
import pytest
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.nodes.content_processing import document_quality_check
from data_collection_workflow.evidence_chunking import build_evidence_chunks, local_content_skip_reason
from data_collection_workflow.nodes.extraction import extraction_skip_reason

@pytest.fixture(autouse=True)
def revision(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')


def route(tmp_path, body, *, title='Cholera in Ghana', revision='evidence'):
    doc = parse_response(('<html><body><h1>'+title+'</h1>'+body+'</body></html>').encode(),
        url='https://research.example/article', source_id='source', session_dir=tmp_path,
        content_type='text/html')
    doc.update(source_role='data_source', source_role_final='collection', fetch_purpose='data_extraction')
    state={'structured_task':{'disease':'cholera','location':'Ghana','start_date':'2025-01-01','end_date':'2025-12-31',
        'collection_mode':'direct_collection'},'documents':[doc]}
    state.update(document_quality_check(state))
    state.update(build_evidence_chunks(state))
    return state


def test_reference_disease_does_not_silence_original_local_target_observation(tmp_path):
    body='<p>Ghana reported 12 confirmed cholera cases in 2025.</p>'
    base=route(tmp_path/'base',body)
    mixed=route(tmp_path/'mixed',body+'<h2>References</h2><p>Dengue surveillance reported 81 cases in 2024.</p>')
    expected=[c['text'] for c in base['evidence_chunks'] if c['contains_target_data']]
    retained=[c['text'] for c in mixed['evidence_chunks'] if c['contains_target_data']]
    assert 'Ghana reported 12 confirmed cholera cases in 2025.' in expected
    assert set(expected) <= set(retained)
    assert all(extraction_skip_reason(c) is None for c in mixed['evidence_chunks'] if c['text'] == 'Ghana reported 12 confirmed cholera cases in 2025.')
    assert mixed['documents'][0]['document_disease_relevance_status']=='ambiguous_disease'
    assert not any(c['contains_target_data'] for c in mixed['evidence_chunks'] if 'Dengue' in c['text'])


def test_target_heading_does_not_authorize_other_disease_count(tmp_path):
    state=route(tmp_path,'<p>Ghana reported 81 confirmed dengue cases in 2025.</p>')
    foreign=[c for c in state['evidence_chunks'] if '81 confirmed dengue' in c['text']]
    assert foreign
    assert all(not c['contains_target_data'] and not c['extraction_eligible_for_task_disease'] for c in foreign)
    assert all(extraction_skip_reason(c) for c in foreign)


def test_title_from_search_does_not_authorize_wrong_disease_page(tmp_path):
    state=route(tmp_path,'<p>Ghana reported 81 dengue cases in 2025.</p>',title='Dengue surveillance')
    assert state['documents'][0]['extraction_readiness']=='not_ready'
    assert not any(c['contains_target_data'] for c in state['evidence_chunks'])


def test_local_target_table_survives_unrelated_reference(tmp_path):
    state=route(tmp_path,'<table><caption>Cholera in Ghana during 2025</caption>'
        '<tr><th>Disease</th><th>Country</th><th>Confirmed cases</th><th>Year</th></tr>'
        '<tr><td>cholera</td><td>Ghana</td><td>12</td><td>2025</td></tr></table>'
        '<h2>References</h2><p>A dengue outbreak caused 81 cases.</p>')
    rows=[c for c in state['evidence_chunks'] if c.get('row_id') is not None and '12' in c['text']]
    assert rows
    assert any(c['extraction_eligible_for_task_disease'] and extraction_skip_reason(c) is None for c in rows)
    assert all(local_content_skip_reason(c) is None for c in rows)


@pytest.mark.parametrize('text',[
    'The patient had a copy number variant and later recovered.',
    'Cite the patient history: a rash developed after admission.',
    'Author-reported symptoms included fever and vomiting.',
    '患者出现呕吐和脱水，随后康复。',
    'Les manifestations inhabituelles ont persisté après la sortie.',
    'Unrecognized clinical presentation without a numeric count.',
])
def test_unknown_or_clinical_narrative_is_not_ui_metadata(text):
    chunk={'text':text,'chunk_kind':'text','bound_context_spans':[{'role':'heading','quote':'Cholera in Ghana: 12 cases in 2025'}]}
    assert local_content_skip_reason(chunk) is None


@pytest.mark.parametrize('extra',[{'row_id':'r','table_id':'t','chunk_kind':'table_row'},
    {'structured_data_kind':'json_record','chunk_kind':'text'}])
def test_structured_records_do_not_use_ui_text_vocabulary(extra):
    assert local_content_skip_reason({'text':'Copy | 12',**extra}) is None


@pytest.mark.parametrize('text',['Copy','Cite','Find articles by'])
def test_explicit_ui_units_cannot_borrow_target_measurement(text):
    chunk={'text':text,'chunk_kind':'text','contains_target_data':True,
        'extraction_eligible_for_task_disease':True,'disease_relevance_status':'target_disease_match',
        'bound_context_spans':[{'role':'heading','quote':'Cholera in Ghana: 12 cases in 2025'}]}
    assert local_content_skip_reason(chunk)
    assert extraction_skip_reason(chunk)


@pytest.mark.parametrize('text',[
    'Fever and rash developed in Patient A. Find articles by Jane Doe',
    'Find articles by Élise Martin\nLa personne présentait une diarrhée aqueuse sévère.',
    'Find articles by Jane Doe\nThere was pronounced dehydration, with improvement after rehydration.',
    'Author links open overlay panel\nClinical outcomes remained uncertain for this small cohort.',
])
def test_embedded_author_control_cannot_discard_actual_or_unfamiliar_narrative(text):
    assert local_content_skip_reason({'text':text,'chunk_kind':'text'}) is None


@pytest.mark.parametrize('text',['None declared','Not applicable'])
def test_administrative_phrase_under_clinical_heading_is_not_auto_discarded(text):
    assert local_content_skip_reason({'text':text,'chunk_kind':'text',
        'bound_context_spans':[{'role':'heading','quote':'Clinical observation'}]}) is None


@pytest.mark.parametrize('text,section',[
    ('Find articles by John Doe','Author information'),
    ('AMA','Discharge outcome'),
    ('APA','Unmapped category'),
    ('NLM','Unmapped category'),
])
def test_mixed_author_metadata_and_ambiguous_short_labels_remain_candidates(text,section):
    assert local_content_skip_reason({'text':text,'chunk_kind':'text',
        'bound_context_spans':[{'role':'heading','quote':section}]}) is None


def test_explicit_citation_format_control_sequence_can_be_skipped():
    assert local_content_skip_reason({'text':'Copy\nFormat:\nAMA\nAPA\nMLA\nNLM','chunk_kind':'text',
        'bound_context_spans':[{'role':'heading','quote':'Cite'}]})
