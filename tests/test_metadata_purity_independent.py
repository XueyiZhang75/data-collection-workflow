import pytest
from data_collection_workflow.nodes.extraction import extraction_skip_reason
from test_acquisition_mixed_page_readiness import pipeline

CONSENT=('I confirm that all necessary patient/participant consent has been obtained and the appropriate institutional forms have been archived, '
 'and that any patient/participant/sample identifiers included were not known to anyone (e.g., hospital staff, patients or participants themselves) '
 'outside the research group so cannot be used to identify individuals. Yes')

@pytest.mark.parametrize('text,section',[
 ('T. Example reported that Patient A developed fever and was hospitalized. Journal of Vector Studies 11, 2130 (2020).','References'),
 ('T. Example, A. Sample. The participant could no longer stand unaided for three nights. Journal of Vector Studies 11, 2130 (2020).','References'),
 ('In accordance with rules for 12 hospitalized patients, formal ethical approval for the cohort was waived. '+CONSENT,'Author Declarations'),
 ('All data processing complied with the protocol; two participants lost vision. '+CONSENT,'Author Declarations'),
])
def test_citation_end_or_admin_template_does_not_prove_middle_is_pure(tmp_path,monkeypatch,text,section):
 state=pipeline(tmp_path,monkeypatch,'dengue','<h1>Dengue outbreak in Canada during 2025</h1><p>Canada reported 12 confirmed dengue cases.</p><h2>'+section+'</h2><p>'+text+'</p>')
 row=next(c for c in state['evidence_chunks'] if text in c['text'])
 assert extraction_skip_reason(row) is None
 assert row['extraction_priority']<=3


@pytest.mark.parametrize('text',['| RF | Recovered fully |','| DR | Died recently |','| DC | Developed convulsions |'])
def test_acronym_initials_do_not_disprove_finite_clinical_observation(text):
 from data_collection_workflow.evidence_chunking import local_content_skip_reason
 chunk={'text':text,'contains_target_data':True,'extraction_readiness':'ready','chunk_kind':'metric_row','table_id':'t','row_id':'r','bound_context_spans':[{'role':'heading','quote':'Abbreviations'}]}
 assert local_content_skip_reason(chunk) is None
