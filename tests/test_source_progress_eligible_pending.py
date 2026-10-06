"""Only genuinely executable spans produce awaiting-extraction source status."""
import pytest
from data_collection_workflow.source_progress import _processing_projection

@pytest.fixture(autouse=True)
def evidence(monkeypatch):monkeypatch.setenv('PIPELINE_MODE','evidence')

def project(chunks,attempted=(),jobs=()):
 doc={'source_id':'s','content_readable':True,'clean_text':'Readable source body','parse_status':'parsed','fetch_status':'success','content_hash':'a'*64}
 group={'ids':{'s'},'urls':{'https://test.example/report'},'entries':[]}
 return _processing_projection(group,[doc],list(jobs),chunks,set(attempted),'none')

def span(cid='c',**extra):
 return dict(chunk_id=cid,source_id='s',text='The patient developed a rash and was hospitalized.',
  contains_target_data=True,extraction_eligible_for_task_disease=True,disease_relevance_status='target_disease_match',**extra)

@pytest.mark.parametrize('flags,expected_reason',[({'contains_target_data':False},'no_target_data'),
 ({'extraction_eligible_for_task_disease':False},'disease_not_eligible')])
def test_only_background_remaining_is_readable_not_awaiting(flags,expected_reason):
 c=span();c.update(flags)
 status,reason=project([c])
 assert status=='readable'
 assert reason.startswith('no_extraction_eligible_spans') and expected_reason in reason

def test_real_unattempted_clinical_span_stays_pending():
 assert project([span()])==('awaiting_extraction','unprocessed_evidence_chunks')

def test_actually_attempted_span_without_output_reports_attempt():
 assert project([span()],['c'])==('extracted_without_evidence','no_output_observation')

def test_unknown_disease_relevance_matches_recovery_planner():
 c=span();c['disease_relevance_status']='unknown'
 status,reason=project([c]);assert status=='readable'
 assert 'disease_relevance_unconfirmed' in reason

def test_consumer_skip_is_not_reported_as_extracted():
 assert project([span(extraction_status='skipped')])==('readable','extraction_skipped_without_attempt')

def test_pending_acquisition_budget_is_separate_from_extraction():
 assert project([span()],jobs=[{'status':'budget_deferred','reason':'source_targets'}])==('budget_deferred','source_targets')

def test_legacy_status_mapping_is_preserved(monkeypatch):
 monkeypatch.setenv('PIPELINE_MODE','standard')
 c=span();c['contains_target_data']=False
 assert project([c])==('awaiting_extraction','unprocessed_evidence_chunks')
