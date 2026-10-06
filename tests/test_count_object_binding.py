"""Independent RED reproduction: counted adverse events are not disease cases.
Synthetic source headings supply actual source scope, never task defaults.
No production implementation or original session is modified by this file.
"""
import hashlib
import pytest
from data_collection_workflow.evidence_qualification import assess_record_evidence
from data_collection_workflow.source_assertions import count_mentions

@pytest.fixture(autouse=True)
def evidence(monkeypatch):monkeypatch.setenv('PIPELINE_MODE','evidence')

def assess(body,heading,**facts):
    text=heading+'\n'+body;digest=hashlib.sha256(text.encode()).hexdigest()
    bound=[dict(role='heading',quote=heading,char_start=0,char_end=len(heading),heading_level=1,section_id='h1')]
    doc=dict(source_id='s',document_id='d',content_hash=digest,text_hash=digest,clean_text=text,parser_version='independent-ae/1',locator_spans=bound)
    chunk=dict(source_id='s',chunk_id='c',document_hash=digest,text=body,char_start=len(heading)+1,char_end=len(text),bound_context_spans=bound)
    return assess_record_evidence(dict(record_id='r',source_id='s',supporting_chunk_id='c',**facts),contract={},evidence_index={'documents':[doc],'evidence_chunks':{'c':chunk}})

@pytest.mark.parametrize('disease,country,count',[('chikungunya','France',40),('measles','Canada',27),('dengue','Brazil',82)])
@pytest.mark.parametrize('expression',["cas d’effets indésirables", "cas d'effets indésirables", 'cases of adverse events', 'cases of adverse reactions'])
def test_adverse_event_case_object_is_not_disease_case(disease,country,count,expression):
    body=f'Au total, {count} {expression} ont été déclarés et analysés.'
    q=assess(body,f'{disease} vaccine surveillance in {country}, 2025',disease=disease,country=country,reporting_period='2025',cases_unspecified=count)
    assert not any(f.reason=='invalid_bound_context' for f in q.field_evidence)
    assert q.status=='candidate',q.to_dict()
    assert any(f.field=='cases_unspecified' and not f.supported for f in q.field_evidence)

@pytest.mark.parametrize('body,count',[
    ('Au total, 40 cas d’effets indésirables sur l’ensemble du territoire national, dont 16 graves, ont été déclarés et analysés.',40),
    ('Au total, 81 cas d’effets indésirables,\ndont 22 graves\n, ont été analysés.',81),
    ('Au total, depuis le début de la vaccination avec le vaccin Ixchiq, 62 cas d’effets indésirables ont été déclarés et analysés sur l’ensemble du territoire national, dont 21 graves.',62),
])
def test_saved_adverse_event_phrasing_with_explicit_source_scope(body,count):
    q=assess(body,'Chikungunya surveillance in France, 2025',disease='chikungunya',country='France',reporting_period='2025',cases_unspecified=count)
    assert q.status=='candidate',q.to_dict()
    assert any(f.field=='cases_unspecified' and not f.supported for f in q.field_evidence)

@pytest.mark.parametrize('disease,country',[('chikungunya','France'),('measles','Canada'),('dengue','Brazil')])
def test_genuine_disease_cases_survive_mixed_vaccine_article(disease,country):
    body=f'{country} reported 123 {disease} cases during 2025. The vaccine safety registry recorded 40 cases of adverse events.'
    q=assess(body,f'{disease} surveillance and vaccine safety in {country}, 2025',disease=disease,country=country,reporting_period='2025',cases_unspecified=123)
    assert q.status=='qualified',q.reasons

@pytest.mark.parametrize('disease,country',[('chikungunya','France'),('measles','Canada'),('dengue','Brazil')])
def test_mentions_of_vaccination_alone_do_not_reject_disease_cases(disease,country):
    body=f'Following vaccination, {country} reported 123 confirmed {disease} cases during 2025.'
    q=assess(body,f'{disease} vaccine surveillance in {country}, 2025',disease=disease,country=country,reporting_period='2025',cases_confirmed=123)
    assert q.status=='qualified',q.reasons

@pytest.mark.parametrize('expression',["cas d’effets indésirables", "cas d'effets indésirables",'cases of adverse events','adverse event cases','confirmed cases of adverse reactions'])
def test_rules_do_not_emit_adverse_event_as_disease_case(expression):
    assert not any(m['field'].startswith('cases_') for m in count_mentions(f'France 2025: 40 {expression} were analysed.'))

@pytest.mark.parametrize('label',['Adverse event cases','Cases of adverse events',"Cas d’effets indésirables",'Confirmed adverse reaction cases'])
def test_reversed_label_does_not_prove_disease_case(label):
    q=assess(f'{label}: 40.','Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=40)
    assert q.status=='candidate',q.to_dict()

@pytest.mark.parametrize('header',['Adverse event cases','Cases of adverse events',"Cas d’effets indésirables",'Confirmed adverse reaction cases'])
def test_table_column_object_cannot_count_as_disease_case(header):
    q=assess(f'| Cases | {header} |\n| 123 | 40 |','Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=40)
    assert q.status=='candidate',q.to_dict()
    good=assess(f'| Cases | {header} |\n| 123 | 40 |','Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=123)
    assert good.status=='qualified',good.reasons

@pytest.mark.parametrize('order',['disease_first','adverse_first'])
def test_same_sentence_counts_keep_separate_objects(order):
    first='123 measles cases';second='40 cases of adverse events'
    joined=f'{first} and {second}' if order=='disease_first' else f'{second} and {first}'
    body=f'Canada reported {joined} during 2025.'
    good=assess(body,'Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=123)
    assert good.status=='qualified',good.reasons
    bad=assess(body,'Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=40)
    assert bad.status=='candidate',bad.to_dict()
    assert [m['value'] for m in count_mentions(body) if m['field'].startswith('cases_')]==[123]

@pytest.mark.parametrize('body',[
    '123 confirmed measles cases without adverse events were reported in Canada during 2025.',
    '123 measles cases in Canada during 2025 were investigated for possible adverse events.',
    'In Canada during 2025, the vaccine registry recorded 40 cases of adverse events. Separately, 123 measles cases were confirmed.',
])
def test_unrelated_adverse_event_mentions_do_not_discard_genuine_rule_counts(body):
    assert any(m['value']==123 for m in count_mentions(body))

@pytest.mark.parametrize('header',['Cases without adverse events','Measles cases with adverse events'])
def test_case_population_with_adverse_events_is_not_adverse_event_count(header):
    q=assess(f'| {header} |\n| 123 |','Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=123)
    assert q.status=='qualified',q.reasons

@pytest.mark.parametrize('gap',[' '*240,'\n'+' '*320])
def test_long_layout_spacing_does_not_hide_count_object(gap):
    body=f'40{gap}cases of adverse events were reported.'
    q=assess(body,'Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=40)
    assert q.status=='candidate'
    assert not any(m['field'].startswith('cases_') for m in count_mentions(body))

@pytest.mark.parametrize('object_heading',['Adverse events','Effets indésirables'])
def test_multilevel_table_object_header_binds_its_child_case_column(tmp_path,object_heading):
    from test_evidence_positive_extraction import source_state
    from data_collection_workflow.evidence_qualification import build_evidence_index
    html=('<h1>Pertussis surveillance</h1><table>'
          '<tr><th rowspan="2">Country</th><th rowspan="2">Year</th>'
          f'<th colspan="2">{object_heading}</th><th rowspan="2">Cases</th></tr>'
          '<tr><th>Cases</th><th>Deaths</th></tr>'
          '<tr><td>France</td><td>2027</td><td>40</td><td>3</td><td>123</td></tr></table>')
    state=source_state(tmp_path,html.encode(),'text/html')
    row=next(c for c in state['evidence_chunks'] if c.get('row_id')=='2')
    facts=dict(record_id='r',source_id='source',supporting_chunk_id=row['chunk_id'],disease='Pertussis',country='France',reporting_period='2027',cases_unspecified=40)
    bad=assess_record_evidence(facts,contract={},evidence_index=build_evidence_index(state))
    assert bad.status=='candidate',bad.to_dict()
    assert any(f.field=='cases_unspecified' and not f.supported for f in bad.field_evidence)
    good=assess_record_evidence({**facts,'cases_unspecified':123},contract={},evidence_index=build_evidence_index(state))
    assert good.status=='qualified',good.reasons

@pytest.mark.parametrize('object_heading',['Number of cases of adverse events','Cas d’effets indésirables (n)','Cases of adverse events (n)'])
@pytest.mark.parametrize('multilevel',[False,True])
def test_parsed_table_measure_label_wrappers_keep_the_counted_object(tmp_path,object_heading,multilevel):
    from test_evidence_positive_extraction import source_state
    from data_collection_workflow.evidence_qualification import build_evidence_index
    if multilevel:
        head=('<tr><th rowspan="2">Country</th><th rowspan="2">Year</th>'
              f'<th colspan="2">{object_heading}</th><th rowspan="2">Cases</th></tr>'
              '<tr><th>Cases</th><th>Deaths</th></tr>')
        expected_row='2'
    else:
        head=f'<tr><th>Country</th><th>Year</th><th>{object_heading}</th><th>Deaths</th><th>Cases</th></tr>'
        expected_row='1'
    html='<h1>Pertussis surveillance</h1><table>'+head+'<tr><td>France</td><td>2027</td><td>40</td><td>3</td><td>123</td></tr></table>'
    state=source_state(tmp_path,html.encode(),'text/html')
    row=next(c for c in state['evidence_chunks'] if c.get('row_id')==expected_row)
    facts=dict(record_id='r',source_id='source',supporting_chunk_id=row['chunk_id'],disease='Pertussis',country='France',reporting_period='2027',cases_unspecified=40)
    bad=assess_record_evidence(facts,contract={},evidence_index=build_evidence_index(state))
    assert bad.status=='candidate',bad.to_dict()
    good=assess_record_evidence({**facts,'cases_unspecified':123},contract={},evidence_index=build_evidence_index(state))
    assert good.status=='qualified',good.reasons

@pytest.mark.parametrize('heading',['Number of disease cases with adverse events','Cases without adverse events (n)','Measles cases with adverse events (n)'])
def test_case_population_labels_with_measure_wrappers_stay_disease_cases(heading):
    q=assess(f'| {heading} |\n| 123 |','Measles in Canada, 2025',disease='measles',country='Canada',reporting_period='2025',cases_unspecified=123)
    assert q.status=='qualified',q.reasons
