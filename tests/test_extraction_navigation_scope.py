"""Local source spans cannot borrow disease/count signals from web navigation ancestors."""
import hashlib
import pytest
from data_collection_workflow.nodes import extraction
from data_collection_workflow.evidence_chunking import build_evidence_chunks
from data_collection_workflow.workflow_recovery import assess_collection_gaps

@pytest.fixture(autouse=True)
def revision(monkeypatch): monkeypatch.setenv('PIPELINE_MODE','evidence')

def chunk(text, heading='Post navigation', **extra):
 return dict(source_id='s',chunk_id='c',text=text,chunk_kind='text',fetch_purpose='data_extraction',
  contains_target_data=True,extraction_eligible_for_task_disease=True,disease_relevance_status='target_disease_match',
  source_is_heading=True,bound_context_spans=[{'role':'heading','quote':'Measles in Canada: 12 confirmed cases'},
  {'role':'heading','quote':heading}],**extra)

@pytest.mark.parametrize('text',[
 'EUROPEAN UNION PARLIAMENT: SHOW THE EVIDENCE',
 'DR SAMUEL BANGURA STEPS ASIDE, ENDORSES IBRAHIM PRINCE THOLLEY FOR APC TONKOLILI DISTRICT CHAIRMANSHIP',
 'IMF Managing Director Commends President Bio',"SIERRA LEONE’S JUDICIAL TEST",
 '23 September 2026\nthetimes-sierraleone.com\nNews',
 'Read More »\nSeptember 16, 2026\nNo Comments'])
def test_navigation_cannot_inherit_root_disease_or_count(text):
 c=chunk(text)
 assert extraction.extraction_skip_reason(c)
 assert not any(g.kind=='unprocessed_span' for g in assess_collection_gaps({'evidence_chunks':[c]}))

def test_long_hex_payload_is_not_prose():
 c=chunk('23f326e797b2a386e797b063b24336e797b082a382e386e797b'*8+')',heading='SHARES')
 assert extraction.extraction_skip_reason(c)=='encoded_payload_without_text'

@pytest.mark.parametrize('text',[
 'The patient was hospitalized after developing a rash.',
 'La patiente a été hospitalisée après une éruption cutanée.',
 '患者出现发热和皮疹，随后康复。',
 'The outbreak is spreading to new districts.',
 '12 confirmed cases were reported.',
 'Vaccination coverage reached 78%.',
 'An unknown syndrome affected the index patient.',
 'Read more about the patient who developed fever.',
])
def test_real_observations_remain_eligible_even_in_related_articles(text):
 assert extraction.extraction_skip_reason(chunk(text)) is None

@pytest.mark.parametrize('text',['Inflammatory manifestations and systemic involvement',
 'Clinical characteristics of the emerging condition','新出现的疾病相关临床发现'])
def test_unknown_body_without_navigation_is_not_blocked(text):
 c=chunk(text,heading='Results');c['source_is_heading']=False
 assert extraction.extraction_skip_reason(c) is None

def test_table_and_json_local_fields_are_not_navigation():
 c=chunk('District X | 12')
 c.update(chunk_kind='metric_row',row_id='1',table_id='t')
 assert extraction.extraction_skip_reason(c) is None
 c.update(chunk_kind='text',structured_data_kind='json_record',text='{"cases":12}')
 assert extraction.extraction_skip_reason(c) is None

def test_legacy_source_gate_is_unchanged(monkeypatch):
 monkeypatch.setenv('PIPELINE_MODE','legacy')
 assert extraction.extraction_skip_reason(chunk('IMF Managing Director Commends President Bio')) is None

def test_chunk_builder_preserves_navigation_original_but_removes_false_data_signal():
 text='# Measles in Canada: 12 confirmed cases in 2025\n\nPatients were hospitalized.\n\n## Post navigation\n\n### CITY COUNCIL APPROVES TRANSPORT BUDGET\n\nRead More »\nSeptember 16, 2026\nNo Comments'
 doc=dict(source_id='s',document_id='d',content_hash='a'*64,text_hash=hashlib.sha256(text.encode()).hexdigest(),
  clean_text=text,document_type='html',content_readable=True,parse_status='parsed',document_quality='high',quality_status='usable',fetch_purpose='data_extraction')
 state={'structured_task':{'disease':'measles','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},'documents':[doc]}
 result=build_evidence_chunks(state)
 rows=result['evidence_chunks']
 nav=[c for c in rows if 'CITY COUNCIL' in c['text'] or 'Read More' in c['text']]
 assert len(nav)==2
 assert all(not c['contains_target_data'] and not c['extraction_eligible_for_task_disease'] for c in nav)
 assert all(extraction.extraction_skip_reason(c) for c in nav)
 assert any('Patients were hospitalized.' in c['text'] and c['contains_target_data'] for c in rows)


@pytest.mark.parametrize('llm_enabled',[False,True])
def test_initial_consumer_does_not_dispatch_navigation(monkeypatch,llm_enabled):
 monkeypatch.setattr(extraction.llm_clients,'llm_extraction_enabled',lambda:llm_enabled)
 monkeypatch.setattr(extraction.llm_clients,'llm_fallback_to_rule_based',lambda:False)
 monkeypatch.setattr(extraction.llm_clients,'get_llm_settings',lambda:{'provider':'anthropic','model':'offline-test'})
 def forbidden(*a,**k): raise AssertionError('navigation reached paid extraction')
 monkeypatch.setattr(extraction.llm_clients,'extract_chunk_with_llm',forbidden)
 state={'structured_task':{'disease':'measles','location':'Canada','start_date':'2025-01-01','end_date':'2025-12-31'},
        'evidence_chunks':[chunk('IMF Managing Director Commends President Bio')]}
 result=extraction.structured_extraction(state)
 assert result['raw_records']==[]
 assert result['extraction_attempted_chunk_ids']==[]
 assert result['structured_extraction_summary']['input_chunk_count']==0
