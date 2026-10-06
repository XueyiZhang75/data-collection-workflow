"""Structural metadata must not outrank local observations or acquire fake metric roles."""
import pytest
from data_collection_workflow.evidence_chunking import local_content_skip_reason
from data_collection_workflow.nodes.extraction import extraction_skip_reason, _chunk_information_priority
from test_acquisition_mixed_page_readiness import pipeline


@pytest.fixture(autouse=True)
def explicit_evidence_revision(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def scope(text,heading,**extra):
    return {'text':text,'contains_target_data':True,'extraction_readiness':'ready',
        'bound_context_spans':[{'role':'heading','quote':'Disease outbreak observations in 2025'},
                               {'role':'heading','quote':heading}],**extra}


@pytest.mark.parametrize('label',['Abstract','References','Authors','Affiliations','Metrics','Data Availability'])
def test_complete_bare_section_heading_does_not_borrow_title_measurement(label):
    c=scope(label,'Disease outbreak observations in 2025',source_is_heading=True)
    assert local_content_skip_reason(c)
    assert extraction_skip_reason(c)


def test_article_view_count_is_not_health_measurement():
    c=scope('#### Total Views 220','### Metrics',source_is_heading=True)
    assert local_content_skip_reason(c)


@pytest.mark.parametrize('text,heading',[
    ('Abstract findings: 12 patients recovered.','Report'),
    ('Plus de 550 hospitalisations','Report'),
    ('Patient A developed fever','Report'),
    ('新增病例十二例，均已康复','Report'),
    ('An unfamiliar syndrome was documented','Report'),
    ('12 confirmed cases','References'),
    ('Total Views 220','Patient visual assessment'),
    ('Metrics of clinical recovery improved','Metrics'),
])
def test_observation_or_unknown_heading_preserved(text,heading):
    assert local_content_skip_reason(scope(text,heading,source_is_heading=True)) is None


@pytest.mark.parametrize('code,definition',[
    ('HAS','Haute Autorité de Santé'),('FDA','Food and Drug Administration'),
    ('ODE','Ordinary Differential Equation'),('CHIKV','Chikungunya virus'),
    ('SEIR-SEI','Susceptible, Exposed, Infected, Recovered-Susceptible, Exposed, Infected'),
    ('EMA','European Medicines Agency'),('PRAC','Pharmacovigilance Risk Assessment Committee'),
    ('DENV','Dengue Virus'),('BCG','Bacillus Calmette-Guérin'),
])
def test_code_expansion_alone_does_not_remove_unknown_table_observations(code,definition):
    c=scope(f'| {code} | {definition} |','Abbreviations',chunk_kind='metric_row',table_id='t',row_id='1')
    assert local_content_skip_reason(c) is None
    assert extraction_skip_reason(c) is None


@pytest.mark.parametrize('text,heading',[
    ('| May | 12 |','References'),
    ('| 2025-05 | <25 | 10 | 13 | [7-21] |','References'),
    ('| HAS | 12 cases |','Abbreviations'),
    ('| PRAC | Patient A recovered |','Abbreviations'),
    ('| XYZ | Unfamiliar clinical syndrome was observed |','Abbreviations'),
    ('| PA | Patient A |','Patient details'),
    ('| CHIKV | Chikungunya virus |','Clinical results'),
    ('| Y | 病例均康复 |','Abbreviations'),
    ('| BCR | Besondere klinische Reaktion |','Unknown clinical findings'),
])
def test_numeric_unknown_or_mixed_table_is_never_deleted(text,heading):
    c=scope(text,heading,chunk_kind='metric_row',table_id='t',row_id='1')
    assert local_content_skip_reason(c) is None


CONSENT=('I confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, '
    'and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) '
    'outside the research group so cannot be used to identify individuals. Yes')
CITATION=('T. Example, A. Sample, Adult mosquito abundance in urban areas. Journal of Vector Studies 11, 2130 (2020). '
    'https://doi.org/10.1234/example.2020.123')


@pytest.mark.parametrize('disease,country',[('measles','Canada'),('dengue','Brazil')])
@pytest.mark.parametrize('text,section',[(CONSENT,'Author Declarations')])
def test_complete_consent_template_retained_but_ranks_after_same_source_observation(tmp_path,monkeypatch,disease,country,text,section):
    body=(f'<h1>{disease} outbreak in {country} in 2025</h1>'
          f'<p>{country} reported 12 confirmed {disease} cases in May 2025.</p>'
          f'<h2>{section}</h2><p>{text}</p>')
    state=pipeline(tmp_path,monkeypatch,disease,body)
    actual=next(c for c in state['evidence_chunks'] if '12 confirmed' in c['text'])
    meta=next(c for c in state['evidence_chunks'] if text in c['text'])
    assert meta['contains_target_data'] is True
    assert extraction_skip_reason(meta) is None
    assert _chunk_information_priority(meta)>_chunk_information_priority(actual)
    assert meta['semantic_span_type']=='context'
    doc=state['documents'][0]
    assert doc['clean_text'][meta['char_start']:meta['char_end']]==meta['text']


@pytest.mark.parametrize('text,section',[
    (CITATION+' The patient developed severe dehydration and recovered.','References'),
    (CONSENT+' Patient A had an unfamiliar clinical syndrome.','Author Declarations'),
    ('Le patient présentait une éruption inhabituelle.','Author Declarations'),
    ('Patient A developed fever. See https://doi.org/10.1234/example for background.','References'),
])
def test_mixed_or_unknown_clinical_paragraph_keeps_priority(tmp_path,monkeypatch,text,section):
    body='<h1>measles outbreak in Canada in 2025</h1><p>Canada reported 12 confirmed measles cases.</p>'+f'<h2>{section}</h2><p>{text}</p>'
    state=pipeline(tmp_path,monkeypatch,'measles',body)
    row=next(c for c in state['evidence_chunks'] if text in c['text'])
    assert row['contains_target_data'] is True
    assert extraction_skip_reason(row) is None
    assert row['extraction_priority']<=3


def test_unproven_glossary_keeps_original_chunk_and_locator(tmp_path,monkeypatch):
    state=pipeline(tmp_path,monkeypatch,'dengue','<h1>Dengue outbreak in Canada in 2025</h1>'
        '<p>Canada reported 12 confirmed dengue cases.</p><h2>Abbreviations</h2>'
        '<table><tr><th>Abbreviation</th><th>Meaning</th></tr><tr><td>DENV</td><td>Dengue Virus</td></tr></table>')
    row=next(c for c in state['evidence_chunks'] if 'DENV | Dengue Virus' in c['text'])
    assert row['contains_target_data']
    assert extraction_skip_reason(row) is None
    doc=state['documents'][0]
    assert doc['clean_text'][row['char_start']:row['char_end']]==row['text']


@pytest.mark.parametrize('text',[
    '[Google Scholar](https://example.invalid/lookup?title=Adult+vector&year=2019)\n\n24\n\nT. Example, A. Sample, Adult mosquito abundance. _Journal of Vector Studies_**11**, 2130 (2020).\n\n[View](https://doi.org/10.1234/example.2020.123)\n\n[PubMed](https://example.invalid/123/)',
    'Example; Sample Fatal Adverse Event in an Elderly Patient: A Case Report. _Journal of Infection._ 2025, 12 (9), ofaf550. doi: 10.1234/example.ofaf550 .',
])
def test_bibliography_shape_alone_keeps_original_priority(tmp_path,monkeypatch,text):
    state=pipeline(tmp_path,monkeypatch,'measles',
        '<h1>Measles outbreak in Canada in 2025</h1><p>Canada reported 12 confirmed measles cases.</p>'
        '<h2>References</h2><p>'+text+'</p>')
    row=next(c for c in state['evidence_chunks'] if 'T. Example' in c['text'] or 'Example; Sample' in c['text'])
    assert extraction_skip_reason(row) is None
    assert row['extraction_priority']<=3
