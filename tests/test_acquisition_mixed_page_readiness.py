"""Whole-document ambiguity must not erase explicit local disease evidence."""
import copy
import pytest
from data_collection_workflow.nodes import content_processing as cp
from data_collection_workflow.nodes.extraction import extraction_skip_reason
from data_collection_workflow.document_acquisition import parse_response


def pipeline(tmp_path, monkeypatch, disease, body, *, url='https://example.invalid/report', revision='evidence'):
    monkeypatch.setenv('PIPELINE_MODE', revision)
    doc = parse_response(body.encode(), url=url, source_id='source', session_dir=tmp_path, content_type='text/html')
    doc.update(source_role_final='collection', fetch_purpose='data_extraction')
    state = {'structured_task': {'disease': disease, 'location': 'Canada', 'collection_mode': 'direct_collection'},
             'documents': [doc], 'source_registry': [], 'collection_trace': []}
    state.update(cp.document_quality_check(state))
    state.update(cp.evidence_chunking_and_data_presence_flagging(state))
    return state


def eligible(state):
    return [c for c in state['evidence_chunks'] if not extraction_skip_reason(c)]


@pytest.mark.parametrize('target,other', [('chikungunya','dengue'), ('measles','dengue'), ('dengue','influenza')])
@pytest.mark.parametrize('reverse', [False, True])
def test_mixed_page_routes_only_explicit_local_target_observations(tmp_path, monkeypatch, target, other, reverse):
    sections = [f'<h2>{target} observations</h2><p>Canada reported 12 confirmed {target} cases in May 2025.</p>',
                f'<h2>{other} observations</h2><p>Canada reported 93 confirmed {other} cases in June 2025.</p>']
    if reverse: sections.reverse()
    state = pipeline(tmp_path, monkeypatch, target, '<h1>Surveillance report</h1>' + ''.join(sections))
    doc = state['documents'][0]
    assert doc['document_disease_relevance_status'] == 'ambiguous_disease'
    assert doc['extraction_readiness'] == 'local_assessment_required'
    assert doc['task_relevance_status'] != 'target_disease_match'
    assert any('12 confirmed' in c['text'] for c in eligible(state))
    assert not any('93 confirmed' in c['text'] for c in eligible(state))
    for c in state['evidence_chunks']:
        assert doc['clean_text'][c['char_start']:c['char_end']] == c['text']


@pytest.mark.parametrize('target,other', [('chikungunya','dengue'), ('measles','dengue'), ('dengue','influenza')])
def test_other_disease_reference_does_not_remove_existing_local_fact(tmp_path, monkeypatch, target, other):
    body = f'<h1>{target} surveillance</h1><p>Canada reported 12 confirmed {target} cases in May 2025.</p>'
    clean = pipeline(tmp_path/'clean', monkeypatch, target, body)
    mixed = pipeline(tmp_path/'mixed', monkeypatch, target, body + f'<h2>References</h2><p>Annual {other} surveillance report, 2023.</p>')
    assert any('12 confirmed' in c['text'] for c in eligible(clean))
    assert any('12 confirmed' in c['text'] for c in eligible(mixed))
    assert not any('Annual ' in c['text'] for c in eligible(mixed))


@pytest.mark.parametrize('target,other', [('chikungunya','dengue'), ('measles','dengue')])
def test_target_title_does_not_relabel_foreign_disease_count(tmp_path, monkeypatch, target, other):
    state = pipeline(tmp_path, monkeypatch, target,
        f'<h1>{target} surveillance</h1><p>Canada reported 93 confirmed {other} cases in June 2025.</p>')
    assert not any('93 confirmed' in c['text'] for c in eligible(state))


def test_same_local_assertion_remains_ambiguous(tmp_path, monkeypatch):
    state = pipeline(tmp_path, monkeypatch, 'measles',
        '<h1>Mixed surveillance</h1><p>Canada reported 12 measles cases and 93 dengue cases in May 2025.</p>')
    assert not eligible(state)


def test_mixed_background_navigation_does_not_become_data(tmp_path, monkeypatch):
    state = pipeline(tmp_path, monkeypatch, 'measles',
        '<h1>About measles and dengue</h1><p>Read more about measles and dengue prevention.</p><p>Privacy policy</p>')
    assert not eligible(state)


def test_search_shell_remains_navigation(tmp_path, monkeypatch):
    state = pipeline(tmp_path, monkeypatch, 'measles',
        '<h1>Search results</h1><p>Canada reported 12 measles cases in May 2025.</p><p>Read about dengue.</p>',
        url='https://example.invalid/search?q=measles')
    assert state['documents'][0]['page_identity_status'] == 'search_shell'
    assert not eligible(state)


def test_unreadable_mixed_document_stays_blocked(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    doc={'clean_text':'Canada reported 12 measles cases. See dengue cases.', 'tables':[]}
    result=cp._post_fetch_readiness(doc, {'status':'ambiguous_disease'}, 'unusable')
    assert result['extraction_readiness']=='not_ready'


def test_legacy_page_ambiguity_unchanged(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE','standard')
    result=cp._post_fetch_readiness({'clean_text':'Canada reported 12 measles cases. See dengue cases.'},
                                    {'status':'ambiguous_disease'}, 'usable')
    assert result['extraction_readiness']=='not_ready'


@pytest.mark.parametrize('text,section', [
    ('Copy', 'PERMALINK'),
    ('Copy\nDownload .nbib\n.nbib\nFormat:\nAMA\nAPA\nMLA\nNLM', 'Cite'),
    ('View on publisher site\nPDF (1.4 MB)\nCite\nCollections\nPermalink', 'ACTIONS'),
    ('Open in a new tab', 'Table 1.'),
    ('Similar articles\nCited by other articles\nLinks to NCBI Databases', 'RESOURCES'),
    ('Find articles by', 'Jane Example'),
    ('Author links open overlay panel', 'Author information'),
    ('None declared.', 'Use of artificial intelligence tools'),
    ('None declared.', 'Conflict of interest'),
])
def test_explicit_interface_and_author_units_do_not_inherit_observations(text, section):
    from data_collection_workflow.evidence_chunking import local_content_skip_reason
    chunk={'text':text,'bound_context_spans':[{'role':'heading','quote':'Measles outbreak in Canada: 12 cases in 2025'},
                                             {'role':'heading','quote':section}]}
    assert local_content_skip_reason(chunk) is not None


@pytest.mark.parametrize('text,section,extra', [
    ('Canada reported 12 measles cases in May 2025.', 'Related articles', {}),
    ('Open in a new tab\n12 confirmed cases were reported in May 2025.', 'Table 1.', {}),
    ('Patient A developed fever and was hospitalized.', 'Author information', {}),
    ('The patient had no rash and recovered.', 'Case report', {}),
    ('None declared.', 'Clinical observations', {}),
    ('Find articles by John Doe', 'Author information', {}),
    ('Find articles by\nJane Example\n,\nJohn Example', 'Jane Example', {}),
    ('1\nDepartment of Public Health, Canada\nFind articles by\nJane Example', 'Jane Example', {}),
    ('15 nouveaux cas confirmés ont été signalés.', 'Liens associés', {}),
    ('新增确诊病例十二例，患者均已康复。', '相关信息', {}),
    ('May | 12', 'Table 1.', {'table_id':'t','row_id':'r','chunk_kind':'metric_row'}),
    ('{"cases":12}', 'Data', {'structured_data_kind':'json_record'}),
])
def test_interface_filter_keeps_actual_observations_and_unknown_narrative(text, section, extra):
    from data_collection_workflow.evidence_chunking import local_content_skip_reason
    chunk={'text':text,'bound_context_spans':[{'role':'heading','quote':'Measles outbreak in Canada: 12 cases in 2025'},
                                             {'role':'heading','quote':section}],**extra}
    assert local_content_skip_reason(chunk) is None


def test_mixed_page_retains_data_but_skips_copy_control(tmp_path, monkeypatch):
    state=pipeline(tmp_path, monkeypatch, 'measles',
        '<h1>Measles outbreak in Canada: 12 cases in May 2025</h1>'
        '<p>Canada reported 12 confirmed measles cases in May 2025.</p>'
        '<h2>References</h2><p>Dengue surveillance report.</p><h2>PERMALINK</h2><p>Copy</p>')
    assert any('12 confirmed' in c['text'] for c in eligible(state))
    assert not any(c['text']=='Copy' for c in eligible(state))


@pytest.mark.parametrize('label', ['AMA', 'APA', 'MLA', 'NLM'])
def test_ambiguous_short_labels_require_citation_format_context(label):
    from data_collection_workflow.evidence_chunking import local_content_skip_reason
    assert local_content_skip_reason({'text':label,'bound_context_spans':[{'role':'heading','quote':'Discharge outcome'}]}) is None
    assert local_content_skip_reason({'text':'Format:\n'+label,'bound_context_spans':[{'role':'heading','quote':'Cite'}]})
