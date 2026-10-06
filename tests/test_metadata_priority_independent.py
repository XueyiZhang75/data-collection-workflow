"""Independent metadata routing boundaries; never tests fact qualification by counts."""
import pytest
from data_collection_workflow.evidence_chunking import local_content_skip_reason
from data_collection_workflow.nodes.extraction import extraction_skip_reason,_chunk_information_priority
from test_acquisition_mixed_page_readiness import pipeline


def unit(text,section,**extra):
 return dict(text=text,contains_target_data=True,extraction_readiness='ready',bound_context_spans=[{'role':'heading','quote':section}],**extra)


@pytest.mark.parametrize('body',[
 'Clinical <em>Metrics</em>: 12 dengue patients recovered in Canada during 2025',
 '<em>Data Availability</em>: 12 confirmed dengue cases in Canada during 2025',
 'The <em>Authors</em> reported 12 dengue cases in Canada during 2025',
 'Clinical <em>Abstract</em>: First dengue patient recovered in Canada during 2025',
])
def test_inline_heading_metadata_token_cannot_erase_observation_unit(tmp_path,monkeypatch,body):
 state=pipeline(tmp_path,monkeypatch,'dengue','<h1>Dengue outbreak in Canada</h1><h2>'+body+'</h2>')
 headings=[c for c in state['evidence_chunks'] if c.get('source_is_heading') and c['text'] in {'Metrics','Data Availability','Authors','Abstract'}]
 assert headings
 for chunk in headings:
  assert extraction_skip_reason(chunk) is None
  assert chunk['contains_target_data']


@pytest.mark.parametrize('body,section',[
 ('Consent was waived for all patients. Twelve patients developed a rash.','Author Declarations'),
 ('We confirm that institutional permission was granted. Patient A died.','Ethics'),
 ('This report includes 12 confirmed dengue cases in Canada during 2025. https://doi.org/10.1234/test.2025','References'),
 ('J. Author. Clinical study. 2025;12:23. https://doi.org/10.1234/test. The patient developed encephalitis.','References'),
 ('El paciente presentó fiebre y posteriormente se recuperó.','Author Declarations'),
 ('Die Patientin entwickelte Fieber und erholte sich.','References'),
])
def test_mixed_or_unrecognized_narrative_is_not_metadata_deprioritized(tmp_path,monkeypatch,body,section):
 state=pipeline(tmp_path,monkeypatch,'dengue','<h1>Dengue outbreak in Canada during 2025</h1><p>Canada reported 12 confirmed dengue cases.</p><h2>'+section+'</h2><p>'+body+'</p>')
 chunk=next(c for c in state['evidence_chunks'] if body in c['text'])
 assert extraction_skip_reason(chunk) is None
 assert chunk['extraction_priority']<=3


@pytest.mark.parametrize('text',[
 '| PCR | Patients clinically recovered. |',
 '| PR | Patient recovered |',
 '| DHF | Dengue hemorrhagic fever (12 patients) |',
 '| PA | Patient A |',
 '| UI | Une infection observée |',
 '| Y | 患者已康复 |',
 '| CFR | Case fatality ratio 2.5% |',
])
def test_abbreviation_shaped_observation_or_ambiguous_identity_retained(text):
 assert local_content_skip_reason(unit(text,'Abbreviations',chunk_kind='metric_row',table_id='table',row_id='row')) is None


@pytest.mark.parametrize('text,section',[
 ('| Canada | 12 |','References'),
 ('| 2025-05-04 | 18 000 |','Author Declarations'),
 ('| Week 12 | 0 |','Abbreviations'),
 ('| AR | Acute rash |','Clinical findings'),
])
def test_table_under_metadata_section_is_not_blanket_deleted(text,section):
 assert local_content_skip_reason(unit(text,section,chunk_kind='metric_row',table_id='table',row_id='row')) is None


@pytest.mark.parametrize('text,heading',[
 ('This cohort included Authors who became ill.','Report'),
 ('Authors','Patient groups'),
 ('Total Views 220','Clinical visual acuity'),
 ('未知病例报告','Metrics'),
])
def test_nonheading_and_unknown_units_do_not_become_bare_ui(text,heading):
 assert local_content_skip_reason(unit(text,heading)) is None


@pytest.mark.parametrize('body,section',[
 ('I confirm that all necessary patient consent has been obtained. Long afterwards, everyone described altered color perception.','Author Declarations'),
 ('I confirm that all necessary participant consent has been obtained; in the following fortnight appetite disappeared completely.','Author Declarations'),
 ('All necessary institutional forms have been archived. For three nights the participant could not stand unaided.','Ethics'),
 ('J. Author. Clinical study. 2025;12:23. https://doi.org/10.1234/test. Subsequently, vision remained doubled for weeks.','References'),
 ('I confirm that ethical permission was granted. Das Farbsehen war anschließend verändert.','Author Declarations'),
])
def test_recognized_template_with_unknown_continuation_is_not_proven_pure_metadata(tmp_path,monkeypatch,body,section):
 state=pipeline(tmp_path,monkeypatch,'dengue','<h1>Dengue outbreak in Canada during 2025</h1><p>Canada reported 12 confirmed dengue cases.</p><h2>'+section+'</h2><p>'+body+'</p>')
 chunk=next(c for c in state['evidence_chunks'] if body in c['text'])
 assert extraction_skip_reason(chunk) is None
 assert chunk['extraction_priority']<=3
