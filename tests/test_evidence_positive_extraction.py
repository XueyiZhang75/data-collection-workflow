"""Positive evidence parse -> source spans -> extraction model -> qualification tests."""
import io
from copy import deepcopy

import pytest

from data_collection_workflow.config import load_llm_structured_extraction_policy
from data_collection_workflow.evidence_products import build_evidence_products
from data_collection_workflow.models import LLMExtractedRecord, LLMStructuredExtractionPolicy
from data_collection_workflow.nodes.content_processing import evidence_chunking_and_data_presence_flagging
from data_collection_workflow.nodes.extraction import _build_record_from_llm_output
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index


@pytest.fixture(autouse=True)
def evidence(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def source_state(tmp_path, body, content_type, *, disease='Pertussis'):
    doc = parse_response(body, url='https://example.test/bulletin', source_id='source',
                         session_dir=tmp_path, content_type=content_type)
    doc.update(document_id='document', quality_status='usable', extraction_readiness='ready',
               source_type='official_public_health')
    state = {'documents': [doc], 'structured_task': {'disease': disease}}
    state.update(evidence_chunking_and_data_presence_flagging(state))
    return state


def qualified_outputs(state, facts):
    policy = LLMStructuredExtractionPolicy(**load_llm_structured_extraction_policy())
    index = build_evidence_index(state)
    outputs, attempts = [], []
    for position, chunk in enumerate(state['evidence_chunks'], 1):
        if not chunk.get('extraction_eligible_for_task_disease'):
            continue
        # Deterministic stand-in for a correct LLM response; use the real builder.
        model = LLMExtractedRecord(**facts)
        built = _build_record_from_llm_output(model, chunk, position, policy, {},
                    context={'disease_standard_name': facts['disease'], 'is_hantavirus': False})
        if built is None:
            continue
        row = built.model_dump()
        q = assess_record_evidence(row, contract={}, evidence_index=index)
        attempts.append((chunk, q))
        row['evidence_qualification'] = q.to_dict()
        if q.status == 'qualified':
            outputs.append(row)
    return outputs, attempts


FACTS = dict(disease='Pertussis', country='United States', subnational_location='Washington',
             reporting_period='2024', cases_confirmed=12)


def test_multiline_source_text_keeps_exact_resolvable_spans(tmp_path):
    text = 'Pertussis in Washington, United States: 12 confirmed cases\nduring 2024.'
    state = source_state(tmp_path, text.encode(), 'text/plain')
    outputs, attempts = qualified_outputs(state, FACTS)
    assert outputs, [(q.status, q.reasons) for _, q in attempts]
    for chunk in state['evidence_chunks']:
        assert state['documents'][0]['clean_text'][chunk['char_start']:chunk['char_end']] == chunk['text']


def test_long_document_retains_short_exact_positive_observation(tmp_path):
    text = 'Pertussis in Washington, United States: 12 confirmed cases during 2024. ' + 'General surveillance methodology. ' * 60
    state = source_state(tmp_path, text.encode(), 'text/plain')
    outputs, attempts = qualified_outputs(state, FACTS)
    assert outputs, [(len(c['text']), q.reasons) for c, q in attempts]
    assert all(len(e['quote']) <= 900 for r in outputs for e in r['evidence_qualification']['field_evidence'])


def test_html_source_heading_header_and_rows_survive_pipeline(tmp_path):
    html = b'<h1>Pertussis in United States</h1><h2>Reporting period: 2024</h2><table><tr><th>State</th><th>Confirmed cases</th></tr><tr><td>Washington</td><td>12</td></tr><tr><td>Oregon</td><td>7</td></tr></table>'
    state = source_state(tmp_path, html, 'text/html')
    outputs, attempts = qualified_outputs(state, FACTS)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]
    row_chunks = [c for c in state['evidence_chunks'] if c.get('row_id') is not None]
    assert row_chunks
    assert all(c['table_id'] == 'table_1' for c in row_chunks)
    assert any(c['table_header'] == 'State | Confirmed cases' for c in row_chunks)
    assert any(c.get('bound_context_spans') for c in row_chunks)
    bad, _ = qualified_outputs(state, {**FACTS, 'cases_confirmed': 7})
    assert bad == []
    products = build_evidence_products(outputs, evidence_index=build_evidence_index(state))
    assert products['aggregate_groups']
    assert not products['excluded_observations']


def test_csv_structured_field_names_and_own_row_bind_numeric_values(tmp_path):
    csv = b'disease,country,reporting_period,cases_confirmed\nPertussis,France,2024,12\nPertussis,Germany,2024,7\n'
    state = source_state(tmp_path, csv, 'text/csv')
    facts = dict(disease='Pertussis', country='France', reporting_period='2024', cases_confirmed=12)
    outputs, attempts = qualified_outputs(state, facts)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]
    assert qualified_outputs(state, {**facts, 'cases_confirmed': 7})[0] == []


def test_json_object_fields_remain_typed_evidence(tmp_path):
    body = b'[{"disease":"Pertussis","country":"France","reporting_period":"2024","cases_confirmed":12},{"disease":"Pertussis","country":"Germany","reporting_period":"2024","cases_confirmed":7}]'
    state = source_state(tmp_path, body, 'application/json')
    facts = dict(disease='Pertussis', country='France', reporting_period='2024', cases_confirmed=12)
    outputs, attempts = qualified_outputs(state, facts)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]
    assert qualified_outputs(state, {**facts, 'cases_confirmed': 7})[0] == []
    assert any(c.get('json_pointer') == '/0' for c in state['evidence_chunks'])


def test_pdf_native_multiline_text_remains_exact_evidence(tmp_path):
    canvas_module = pytest.importorskip('reportlab.pdfgen.canvas')
    buffer = io.BytesIO()
    canvas = canvas_module.Canvas(buffer)
    canvas.drawString(36, 760, 'Pertussis in Washington, United States: 12 confirmed cases')
    canvas.drawString(36, 740, 'during 2024.')
    canvas.save()
    state = source_state(tmp_path, buffer.getvalue(), 'application/pdf')
    outputs, attempts = qualified_outputs(state, FACTS)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]


def test_bound_heading_does_not_carry_into_next_disease_section(tmp_path):
    text = '# Pertussis in France during 2024\n12 confirmed cases.\n# Measles in Germany during 2024\n7 confirmed cases.'
    state = source_state(tmp_path, text.encode(), 'text/plain')
    facts = dict(disease='Pertussis', country='France', reporting_period='2024', cases_confirmed=12)
    assert qualified_outputs(state, facts)[0]
    assert qualified_outputs(state, {**facts, 'cases_confirmed': 7})[0] == []


def test_distant_source_heading_and_product_locator_share_context_contract(tmp_path):
    html = ('<h1>Pertussis in France during 2024</h1><p>' +
            'Surveillance methodology. ' * 80 + '</p><p>12 confirmed cases.</p>')
    state = source_state(tmp_path, html.encode(), 'text/html')
    facts = dict(disease='Pertussis', country='France', reporting_period='2024', cases_confirmed=12)
    outputs, attempts = qualified_outputs(state, facts)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]
    index = build_evidence_index(state)
    products = build_evidence_products(outputs, evidence_index=index)
    assert products['aggregate_groups'] and not products['excluded_observations']
    bad = deepcopy(outputs)
    for entry in bad[0]['evidence_qualification']['field_evidence']:
        entry['locator']['bound_context_spans'][0]['char_end'] += 1
    assert build_evidence_products(bad, evidence_index=index)['excluded_observations']


def test_field_quote_cannot_borrow_a_different_chunk(tmp_path):
    text = '# Pertussis in France during 2024\n12 confirmed cases.\n# Measles in Germany during 2024\n7 confirmed cases.'
    state = source_state(tmp_path, text.encode(), 'text/plain')
    facts = dict(disease='Pertussis', country='France', reporting_period='2024', cases_confirmed=12)
    outputs, _ = qualified_outputs(state, facts)
    assert outputs
    other = next(c for c in state['evidence_chunks'] if c['text'] == '7 confirmed cases.')
    row = {**outputs[0], 'chunk_id': other['chunk_id'], 'evidence_chunk_id': other['chunk_id'],
           'field_provenance_json': {name: {'quote': '12 confirmed cases.', 'chunk_id': other['chunk_id']}
                                    for name in ('disease','country','reporting_period','cases_confirmed')}}
    q = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(state))
    assert q.status == 'candidate'
    assert any('nonlocal' in reason for reason in q.reasons)


@pytest.mark.parametrize('content_type,body', [
    ('text/plain', b'Pertussis in Washington, United States: 12 confirmed cases during 2024.'),
    ('text/csv', b'disease,country,subnational_location,reporting_period,cases_confirmed\nPertussis,United States,Washington,2024,12\n'),
    ('application/json', b'{"disease":"Pertussis","country":"United States","subnational_location":"Washington","reporting_period":"2024","cases_confirmed":12}'),
])
def test_normalized_count_metric_keeps_explicit_source_semantics(tmp_path, content_type, body):
    state = source_state(tmp_path, body, content_type)
    facts = {**FACTS, 'metric_name':'confirmed_cases', 'metric_value':12,
             'metric_unit':'count', 'metric_category':'case_count'}
    outputs, attempts = qualified_outputs(state, facts)
    assert outputs, [(c['text'], q.reasons) for c, q in attempts]
    assert not qualified_outputs(state, {**facts, 'metric_unit':'percent'})[0]


def test_table_caption_and_note_are_bound_to_their_source_table(tmp_path):
    html = b'<table><caption>Pertussis in France</caption><tr><th>Region</th><th>Confirmed cases</th></tr><tr><td>North</td><td>12</td></tr><tfoot>Reporting period: 2024-W01</tfoot></table>'
    state = source_state(tmp_path, html, 'text/html')
    facts = dict(disease='Pertussis',country='France',subnational_location='North',
                 reporting_period='2024-W01',cases_confirmed=12)
    outputs, attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    assert build_evidence_products(outputs,evidence_index=build_evidence_index(state))['aggregate_groups']


def test_explicit_table_geography_column_controls_shared_country_heading(tmp_path):
    html = b'<h1>Pertussis in France during 2024</h1><table><tr><th>Country</th><th>Confirmed cases</th></tr><tr><td>France</td><td>12</td></tr><tr><td>Germany</td><td>7</td></tr></table>'
    state = source_state(tmp_path, html, 'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    assert qualified_outputs(state,facts)[0]
    assert not qualified_outputs(state,{**facts,'cases_confirmed':7})[0]
    assert qualified_outputs(state,{**facts,'country':'Germany','cases_confirmed':7})[0]


def test_markdown_table_row_without_repeated_metric_label_is_extractable(tmp_path):
    text = '# Pertussis in United States during 2024\n| State | Confirmed cases |\n| --- | --- |\n| Washington | 12 |\n| Oregon | 7 |'
    state = source_state(tmp_path,text.encode(),'text/plain')
    outputs,attempts = qualified_outputs(state,FACTS)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    rows = [c for c in state['evidence_chunks'] if c.get('row_id')]
    assert len(rows) == 2 and all(c['extraction_eligible_for_task_disease'] for c in rows)
    assert not qualified_outputs(state,{**FACTS,'cases_confirmed':7})[0]


def test_field_quote_can_name_a_verified_heading_and_products_resolve_it(tmp_path):
    title = 'Pertussis in France during 2024'
    state = source_state(tmp_path, ('<h1>'+title+'</h1><p>12 confirmed cases.</p>').encode(), 'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12,
                 field_provenance_json={'disease':{'quote':title},'country':{'quote':title},'reporting_period':{'quote':title}})
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    result = build_evidence_products(outputs,evidence_index=build_evidence_index(state))
    assert result['aggregate_groups'] and not result['excluded_observations']


def test_overlong_structural_evidence_stays_candidate(tmp_path):
    title = 'Pertussis in France during 2024 ' + 'methodology ' * 80
    state = source_state(tmp_path, ('<h1>'+title+'</h1><p>12 confirmed cases.</p>').encode(),'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    assert state['evidence_chunks']
    assert not qualified_outputs(state,facts)[0]


def test_unbound_task_metadata_cannot_replace_a_missing_source_title(tmp_path):
    state = source_state(tmp_path,b'12 confirmed cases.','text/plain')
    state['documents'][0].update(title='Pertussis in France during 2024',country='France',reporting_period_label='2024')
    state.update(evidence_chunking_and_data_presence_flagging(state))
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    assert not qualified_outputs(state,facts)[0]


def test_json_reporting_year_can_be_a_source_number(tmp_path):
    body = b'{"disease":"Pertussis","country":"France","reporting_period":2024,"cases_confirmed":12}'
    state = source_state(tmp_path,body,'application/json')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    assert not qualified_outputs(state,{**facts,'reporting_period':'2025'})[0]


def test_numeric_source_heading_remains_extractable_evidence(tmp_path):
    state = source_state(tmp_path,b'<h1>Pertussis in France caused 12 confirmed cases during 2024.</h1>','text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]


@pytest.mark.parametrize('suffix',['',' Patient B with Pertussis in France was aged 20.'])
def test_source_patient_span_keeps_own_attributes_and_products(tmp_path,suffix):
    text = 'Patient A with Pertussis in France was aged 45.' + suffix
    state = source_state(tmp_path,text.encode(),'text/plain')
    facts = dict(disease='Pertussis',country='France',workflow_case_label='Patient A',age='45')
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    products = build_evidence_products(outputs,evidence_index=build_evidence_index(state))
    assert products['case_entities'] and not products['excluded_observations']
    assert not qualified_outputs(state,{**facts,'age':'20'})[0]


def test_death_metric_cannot_read_a_different_cases_column(tmp_path):
    html = b'<h1>Pertussis in France during 2024</h1><table><tr><th>Cases</th><th>Deaths</th></tr><tr><td>12</td><td>0</td></tr></table>'
    state = source_state(tmp_path,html,'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',deaths=0)
    assert qualified_outputs(state,facts)[0]
    assert not qualified_outputs(state,{**facts,'deaths':12})[0]


def test_typed_total_tests_header_cannot_read_positive_tests(tmp_path):
    html = b'<h1>Pertussis in France during 2024</h1><table><tr><th>Positive tests</th><th>Total tests</th></tr><tr><td>12</td><td>100</td></tr></table>'
    state = source_state(tmp_path,html,'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',tests_total=100)
    assert qualified_outputs(state,facts)[0]
    assert not qualified_outputs(state,{**facts,'tests_total':12})[0]


def test_source_disease_can_appear_between_count_modifier_and_noun(tmp_path):
    text = 'Canada reported 12 confirmed pertussis cases during 2025.'
    state = source_state(tmp_path,text.encode(),'text/plain')
    facts = dict(disease='Pertussis',country='Canada',reporting_period='2025',cases_confirmed=12)
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    assert not qualified_outputs(state,{**facts,'disease':'Measles'})[0]


def test_bound_prose_heading_retains_its_hypothetical_context(tmp_path):
    html = b'<h1>Hypothetical scenario: Pertussis in France during 2024.</h1><p>12 confirmed cases.</p>'
    state = source_state(tmp_path,html,'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    assert not qualified_outputs(state,facts)[0]


def test_large_source_table_is_indexed_once_and_retains_all_rows(tmp_path,monkeypatch):
    from data_collection_workflow import evidence_chunking
    original = evidence_chunking._table_spans
    calls = []
    def counted(doc):
        calls.append(1)
        yield from original(doc)
    monkeypatch.setattr(evidence_chunking,'_table_spans',counted)
    csv = 'disease,country,reporting_period,cases_confirmed\n' + '\n'.join(f'Pertussis,Country{i},2024,{i}' for i in range(1000))
    state = source_state(tmp_path,csv.encode(),'text/csv')
    rows = [c for c in state['evidence_chunks'] if c.get('row_id')]
    assert len(rows) == 1000
    assert len({c['char_start'] for c in rows}) == 1000
    assert len(calls) == 1


def test_json_candidate_without_disease_key_does_not_raise(tmp_path):
    state = source_state(tmp_path,b'{"country":"France","reporting_period":"2024","cases_confirmed":12}','application/json')
    chunk = state['evidence_chunks'][0]
    q = assess_record_evidence(dict(source_id='source',chunk_id=chunk['chunk_id'],country='France',
        reporting_period='2024',cases_confirmed=12),contract={},evidence_index=build_evidence_index(state))
    assert q.status == 'candidate'


def test_single_column_source_table_keeps_header_cell_binding(tmp_path):
    html = b'<h1>Pertussis in France during 2024</h1><table><tr><th>Confirmed cases</th></tr><tr><td>12</td></tr></table>'
    state = source_state(tmp_path,html,'text/html')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12)
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    assert not qualified_outputs(state,{**facts,'cases_confirmed':24})[0]
    products = build_evidence_products(outputs,evidence_index=build_evidence_index(state))
    assert products['aggregate_groups'] and not products['excluded_observations']


def test_short_field_quote_uses_its_own_complete_source_sentence(tmp_path):
    text = 'Canada reported 12 confirmed pertussis cases during 2025.'
    state = source_state(tmp_path,text.encode(),'text/plain')
    facts = dict(disease='Pertussis',country='Canada',reporting_period='2025',cases_confirmed=12,
                 evidence_quote=text,field_provenance_json={'cases_confirmed':{'quote':'12 confirmed pertussis cases'}})
    outputs,attempts = qualified_outputs(state,facts)
    assert outputs, [(c['text'],q.reasons) for c,q in attempts]
    assert build_evidence_products(outputs,evidence_index=build_evidence_index(state))['aggregate_groups']


def test_short_field_quote_cannot_borrow_scope_from_a_different_sentence(tmp_path):
    text = 'Pertussis in France during 2024. Germany reported 12 confirmed cases during 2025.'
    state = source_state(tmp_path,text.encode(),'text/plain')
    facts = dict(disease='Pertussis',country='France',reporting_period='2024',cases_confirmed=12,
                 field_provenance_json={'cases_confirmed':{'quote':'12 confirmed cases'}})
    assert not qualified_outputs(state,facts)[0]


def test_background_heading_is_retained_without_an_extraction_call(tmp_path):
    state = source_state(tmp_path,b'<h1>Pertussis surveillance</h1><p>France reported 12 confirmed pertussis cases during 2024.</p>','text/html')
    state['documents'][0]['fetch_purpose']='data_extraction'
    state.update(evidence_chunking_and_data_presence_flagging(state))
    title = next(c for c in state['evidence_chunks'] if c['text']=='Pertussis surveillance')
    assert not title['extraction_eligible_for_task_disease']
    assert any(c['extraction_eligible_for_task_disease'] for c in state['evidence_chunks'])
