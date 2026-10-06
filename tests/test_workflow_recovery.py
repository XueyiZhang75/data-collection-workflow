from data_collection_workflow.workflow_recovery import assess_collection_gaps, plan_recovery, merge_recovery_delta, RecoveryDelta, recovery_control


def test_merge_is_idempotent_keeps_conflicting_observations():
    state={'source_registry':[{'source_id':'s1','canonical_url':'https://example.org/report'}], 'documents':[], 'raw_records':[]}
    delta=RecoveryDelta(sources=[{'source_id':'s2','canonical_url':'https://example.org/report#page=1'}],documents=[{'document_id':'d1','source_id':'s2','content_hash':'abc','clean_text':'A'}],records=[{'record_id':'r1','source_id':'s2','supporting_chunk_id':'c1','cases_confirmed':4},{'record_id':'r2','source_id':'s2','supporting_chunk_id':'c1','cases_confirmed':5}])
    first=merge_recovery_delta(state,delta)
    second=merge_recovery_delta({**state,**first},delta)
    assert first == second
    assert len(first['source_registry'])==1
    assert len(first['documents'])==1
    assert len(first['raw_records'])==2
    assert all(r['source_id']=='s1' for r in first['raw_records'])


def test_recovery_prefers_existing_content_and_respects_remaining_budget():
    state={'documents':[{'document_id':'d1','source_id':'s1','content_hash':'a','clean_text':'measles 4 cases'}], 'evidence_chunks':[], 'source_registry':[], 'structured_task':{'disease':'measles'}}
    gaps=assess_collection_gaps(state)
    plan=plan_recovery(gaps,state=state,budget={'remaining':{'fetch':0,'search':0,'extraction':0}})
    assert plan.actions[0].kind=='reparse'
    assert all(a.kind=='reparse' for a in plan.actions)


def test_no_progress_or_two_rounds_stops():
    state={'recovery_round':1,'recovery_previous_gain':[], 'normalized_records':[], 'source_registry':[], 'documents':[]}
    result=recovery_control(state)
    assert result['recovery_stop_reason']=='no_progress'
    state['recovery_round']=2
    assert recovery_control(state)['recovery_stop_reason']=='round_limit'


def test_successfully_attempted_empty_chunk_not_planned_again():
    state={'documents':[], 'source_registry':[], 'evidence_chunks':[{'chunk_id':'c1','text':'measles cases','content_hash':'a'}], 'extraction_attempted_chunk_ids':['c1']}
    assert not any(g.kind=='unprocessed_span' for g in assess_collection_gaps(state))


def test_source_alias_merge_rebinds_records_and_attempted_span_ids():
    chunk={'chunk_id':'old_c','source_id':'s1','content_hash':'h','text':'4 cases'}
    state={'source_registry':[{'source_id':'s1','canonical_url':'https://example.org/report'}], 'evidence_chunks':[chunk]}
    delta=RecoveryDelta(sources=[{'source_id':'s2','canonical_url':'https://example.org/report'}],chunks=[{**chunk,'chunk_id':'new_c','source_id':'s2'}],records=[{'record_id':'r','source_id':'s2','supporting_chunk_id':'new_c','cases_confirmed':4}],attempted_chunk_ids=['new_c'])
    merged=merge_recovery_delta(state,delta)
    ids={c['chunk_id'] for c in merged['evidence_chunks']}
    assert merged['raw_records'][0]['supporting_chunk_id'] in ids
    assert set(merged['extraction_attempted_chunk_ids']) <= ids


def test_version_bound_chunk_ids_preserve_index(monkeypatch):
    from data_collection_workflow.nodes.content_processing import _make_evidence_chunk
    from data_collection_workflow.evidence_qualification import build_evidence_index
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    def chunk(version,text):
        return _make_evidence_chunk({'source_id':'s','content_hash':version},1,'text',text,0,len(text),None,True,['cases'],[],1.0,'data').model_dump()
    a=chunk('v1','4 cases');b=chunk('v2','5 cases')
    assert a['chunk_id'] != b['chunk_id']
    assert a['document_hash']=='v1'
    merged=merge_recovery_delta({'evidence_chunks':[a]},RecoveryDelta(chunks=[b]))
    assert len({c['chunk_id'] for c in merged['evidence_chunks']})==2
    assert len(build_evidence_index(merged)['evidence_chunks'])==2


def test_recovery_query_targets_requirement_period_and_location():
    from data_collection_workflow.workflow_recovery import _recovery_query
    state={'structured_task':{'disease':'measles','location':'France','start_date':'2024-01-01'},'source_coverage_audit':{'requirements':[{'requirement_id':'r','location':'Paris','period_start':'2024-02-01','period_end':'2024-02-29'}]}}
    query=_recovery_query(state,'r')
    assert 'Paris' in query['query'] and '2024-02-01' in query['query'] and '2024-02-29' in query['query']


def test_alias_merge_nested_provenance_dedup_and_replay():
    import json
    base={'record_id':'a','source_id':'s1','supporting_chunk_id':'old','cases_confirmed':4,'field_provenance_json':json.dumps({'cases_confirmed':{'source_id':'s1','locator':{'chunk_id':'old'}}})}
    chunk={'chunk_id':'old','source_id':'s1','document_hash':'v1','text':'4 cases'}
    state={'source_registry':[{'source_id':'s1','url':'https://example.org'}],'evidence_chunks':[chunk],'raw_records':[base]}
    new={**base,'record_id':'b','source_id':'s2','supporting_chunk_id':'new','field_provenance_json':json.dumps({'cases_confirmed':{'source_id':'s2','locator':{'chunk_id':'new'}}})}
    delta=RecoveryDelta(sources=[{'source_id':'s2','url':'https://example.org'}],chunks=[{**chunk,'source_id':'s2','chunk_id':'new'}],records=[new])
    first=merge_recovery_delta(state,delta)
    assert len(first['raw_records'])==1
    assert merge_recovery_delta(first,delta)==first


def test_execute_recovery_does_not_mark_scheduler_skipped_chunks(monkeypatch,tmp_path):
    from data_collection_workflow.workflow_recovery import execute_recovery,RecoveryPlan,RecoveryAction
    from data_collection_workflow.session_runtime import RunContext
    import data_collection_workflow.nodes.extraction as extraction
    monkeypatch.setattr(extraction,'structured_extraction',lambda state:{'raw_records':[],'extraction_attempted_chunk_ids':[]})
    ctx=RunContext(tmp_path,{'universal':{}})
    delta=execute_recovery(RecoveryPlan([RecoveryAction('extract','c','a','gain')]),context=ctx,artifacts={'evidence_chunks':[{'chunk_id':'c','source_id':'s','text':'4 cases'}]},budget=ctx.ledger)
    assert delta.attempted_chunk_ids==[]
    assert delta.actions[0]['status']=='skipped'
    assert delta.actions[0]['error']=='consumer_did_not_attempt_span'


def test_candidate_or_unsupported_fields_cannot_count_as_progress():
    state={'recovery_round':1,'recovery_previous_gain':[], 'candidate_records':[{'evidence_qualification':{'field_evidence':[{'supported':True,'field':'cases','value':4}]}}], 'qualified_records':[{'evidence_qualification':{'field_evidence':[{'supported':False,'field':'cases','value':5}]}}]}
    assert recovery_control(state)['recovery_stop_reason']=='no_progress'


def test_colliding_legacy_chunk_versions_keep_original_record_reference():
    a={'chunk_id':'same','source_id':'s','document_hash':'v1','text':'4 cases'}
    b={**a,'document_hash':'v2','text':'5 cases'}
    state={'evidence_chunks':[a],'raw_records':[{'record_id':'a','supporting_chunk_id':'same','cases_confirmed':4}]}
    delta=RecoveryDelta(chunks=[b],records=[{'record_id':'b','supporting_chunk_id':'same','cases_confirmed':5}],attempted_chunk_ids=['same'])
    first=merge_recovery_delta(state,delta)
    lookup={c['chunk_id']:c for c in first['evidence_chunks']}
    for record in first['raw_records']:
        assert str(record['cases_confirmed']) in lookup[record['supporting_chunk_id']]['text']
    assert merge_recovery_delta(first,delta)==first


def test_search_recovery_routes_and_fetches_new_provider_source(tmp_path,monkeypatch):
    import importlib
    from data_collection_workflow.models import SourceCandidate
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import execute_recovery,RecoveryPlan,RecoveryAction
    from data_collection_workflow.document_acquisition import parse_response
    discovery=importlib.import_module('data_collection_workflow.nodes.source_discovery')
    acquisition=importlib.import_module('data_collection_workflow.document_acquisition')
    for key,value in {'PIPELINE_MODE':'evidence','SEARCH_MODE':'live','ENABLE_LIVE_SEARCH':'true','SEARCH_PROVIDER':'tavily','ENABLE_LIVE_FETCH':'true','FETCH_SEARCH_DERIVED_SOURCES':'true','ENABLE_LLM_EXTRACTION':'false','ENABLE_LLM_SOURCE_IDENTITY':'false','SOURCE_ID_ALLOWLIST':''}.items(): monkeypatch.setenv(key,value)
    class Provider:
        def search(self,query,**kwargs):
            return {'results':[SourceCandidate(source_id='fresh',url='https://www.cdc.gov/measles/data-research/index.html',title='Measles surveillance 2024 United States',publisher='CDC',source_type='official_public_health_agency',snippet='United States reported 285 confirmed measles cases in 2024.',discovery_method='search',search_provider='offline_test').model_dump()]}
    monkeypatch.setattr(discovery,'_provider_for_settings',lambda settings:Provider())
    fetched=[]
    def acquire(url,**kwargs):
        fetched.append(url)
        return parse_response(b'<h1>Measles surveillance 2024</h1><p>United States reported 285 confirmed measles cases in 2024. The annual surveillance report provides the confirmed case total for the United States from January 1 to December 31, 2024.</p>',url=url,source_id=kwargs['source_id'],session_dir=kwargs['session_dir'],content_type='text/html')
    monkeypatch.setattr(acquisition,'acquire_document',acquire)
    state={'structured_task':{'disease':'measles','location':'United States','start_date':'2024-01-01','end_date':'2024-12-31'},'source_registry':[],'documents':[],'evidence_chunks':[]}
    ctx=RunContext(tmp_path,{'pipeline_mode':'evidence'})
    with ctx.activate():
        delta=execute_recovery(RecoveryPlan([RecoveryAction('search','task','a','gain')]),context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]['status']=='completed',delta.actions
    assert fetched and delta.documents and delta.chunks,delta.sources
    assert delta.sources[0].get('final_screening_decision')
    assert ctx.ledger.snapshot()['used']['search']==1


def test_saved_raw_reparse_rebuilds_text_and_preserves_prior_version(tmp_path,monkeypatch):
    from data_collection_workflow.document_acquisition import parse_response
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.workflow_recovery import execute_recovery,RecoveryPlan,RecoveryAction
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION','false')
    doc=parse_response(b'<h1>Measles</h1><p>United States reported 285 measles cases in 2024.</p>',url='https://example.org/report',source_id='s',session_dir=tmp_path,content_type='text/html')
    doc.update(document_id='old',clean_text='',text_hash='old',parse_status='failed')
    state={'structured_task':{'disease':'measles'},'documents':[doc]}
    ctx=RunContext(tmp_path,{'pipeline_mode':'evidence'})
    with ctx.activate(): delta=execute_recovery(RecoveryPlan([RecoveryAction('reparse','old','a','gain')]),context=ctx,artifacts=state,budget=ctx.ledger)
    assert delta.actions[0]['status']=='completed',delta.actions
    assert '285' in delta.documents[0]['clean_text']
    merged=merge_recovery_delta(state,delta)
    assert len(merged['documents'])==2
    assert doc['clean_text']==''
    assert merge_recovery_delta(merged,delta)==merged


def test_reparse_rejects_outside_session_and_tampered_raw(tmp_path):
    import hashlib
    import pytest
    from data_collection_workflow.workflow_recovery import _reparse_saved_document
    from data_collection_workflow.session_runtime import RunContext,ResumeMismatch
    session=tmp_path/'session'
    ctx=RunContext(session,{})
    outside=tmp_path/'outside.raw';outside.write_bytes(b'evidence')
    doc={'source_id':'s','raw_artifact_path':'../outside.raw','content_hash':hashlib.sha256(b'evidence').hexdigest()}
    with pytest.raises(ResumeMismatch,match='current session'): _reparse_saved_document(doc,ctx)
    inside=session/'inside.raw';inside.write_bytes(b'changed')
    doc['raw_artifact_path']='inside.raw'
    with pytest.raises(ResumeMismatch,match='hash mismatch'): _reparse_saved_document(doc,ctx)


def test_failed_parse_saved_raw_is_recovery_gap_without_refetch_budget():
    state={'source_registry':[{'source_id':'s'}],'documents':[{'source_id':'s','document_id':'d','raw_artifact_path':'artifacts/raw','content_hash':'h','clean_text':''}]}
    plan=plan_recovery(assess_collection_gaps(state),state=state,budget={'remaining':{'fetch':0,'search':0,'extraction':0}})
    assert [(a.kind,a.target_id) for a in plan.actions]==[('reparse','d')]


def test_supported_candidate_fields_count_once_without_duplicate_source_gain():
    import hashlib
    from data_collection_workflow.evidence_qualification import assess_record_evidence
    from data_collection_workflow.workflow_recovery import _gain
    text='Pertussis in Washington, United States: 12 confirmed cases during 2024.'
    digest=hashlib.sha256(text.encode()).hexdigest()
    doc={'source_id':'s','content_hash':digest,'clean_text':text}
    chunk={'chunk_id':'c','source_id':'s','text':text}
    row={'record_id':'r','chunk_id':'c','disease':'Pertussis','country':'United States','subnational_location':'Washington','reporting_period':'2024','cases_confirmed':12,'age':'99'}
    index={'documents':[doc],'evidence_chunks':{'c':chunk}}
    row['evidence_qualification']=assess_record_evidence(row,contract={},evidence_index=index).to_dict()
    assert row['evidence_qualification']['status']=='candidate'
    state={'candidate_records':[row],'documents':[doc],'evidence_chunks':[chunk]}
    original=_gain(state)
    assert original
    other_doc={**doc,'source_id':'mirror'};other_chunk={**chunk,'chunk_id':'mirror-c','source_id':'mirror'}
    duplicate={**row,'record_id':'mirror-r','chunk_id':'mirror-c'}
    index={'documents':[doc,other_doc],'evidence_chunks':{'c':chunk,'mirror-c':other_chunk}}
    duplicate['evidence_qualification']=assess_record_evidence(duplicate,contract={},evidence_index=index).to_dict()
    assert _gain({'candidate_records':[row,duplicate],'documents':[doc,other_doc],'evidence_chunks':[chunk,other_chunk]})==original


def test_recovery_control_recomputes_qualified_coverage_at_loop_boundary():
    record={'record_id':'r','disease':'measles','country':'France','reporting_period':'2024','cases_confirmed':4,'evidence_qualification':{'status':'qualified','product_kind':'aggregate'}}
    state={'qualified_records':[record],'source_coverage_requirements':[{'requirement_id':'annual','disease':'measles','country':'France','reporting_period':'2024'}],
           'source_registry':[{'source_id':'s'}],'documents':[{'source_id':'s','clean_text':'4 cases'}],
           'evidence_chunks':[{'source_id':'s','chunk_id':'c'}],'extraction_attempted_chunk_ids':['c'],
           'source_coverage_audit':{'requirements':[{'requirement_id':'annual','coverage_status':'evidence_gap'}]}}
    result=recovery_control(state)
    assert result['source_coverage_audit']['coverage_status']=='complete'
    assert not result['recovery_gaps']
    assert not result['recovery_plan']['actions']


def test_stop_reason_distinguishes_budget_exhaustion_from_no_action():
    from data_collection_workflow.workflow_recovery import RecoveryGap
    plan=plan_recovery([RecoveryGap('source_missing','r','gap')],state={},budget={'remaining':{'search':0,'fetch':10,'extraction':10}})
    assert plan.stop_reason=='budget_exhausted'
    plan=plan_recovery([],state={},budget={'remaining':{}})
    assert plan.stop_reason=='coverage_satisfied'


def test_alias_document_dedup_rebinds_new_chunk_and_nested_provenance():
    import json
    doc={'document_id':'old_doc','source_id':'s1','content_hash':'raw','text_hash':'text','parser_version':'v1','clean_text':'4 cases'}
    state={'source_registry':[{'source_id':'s1','url':'https://example.org/report'}],'documents':[doc]}
    delta=RecoveryDelta(sources=[{'source_id':'s2','url':'https://example.org/report'}],documents=[{**doc,'source_id':'s2','document_id':'new_doc'}],chunks=[{'source_id':'s2','chunk_id':'new_chunk','document_id':'new_doc','document_hash':'raw','text':'4 cases'}],records=[{'record_id':'r','source_id':'s2','supporting_chunk_id':'new_chunk','document_id':'new_doc','field_provenance_json':json.dumps({'cases_confirmed':{'locator':{'document_id':'new_doc','chunk_id':'new_chunk'}}})}])
    merged=merge_recovery_delta(state,delta)
    assert [d['document_id'] for d in merged['documents']]==['old_doc']
    assert merged['evidence_chunks'][0]['document_id']=='old_doc'
    record=merged['raw_records'][0]
    assert record['document_id']=='old_doc'
    assert json.loads(record['field_provenance_json'])['cases_confirmed']['locator']['document_id']=='old_doc'
    assert merge_recovery_delta(merged,delta)==merged


def test_partial_ocr_remains_a_gap_after_usable_spans_were_extracted():
    state = {'source_registry': [{'source_id': 's'}],
             'documents': [{'source_id': 's', 'document_id': 'd', 'content_hash': 'h', 'clean_text': '12 cases',
                            'content_readable': True, 'acquisition_incomplete': True,
                            'acquisition_status': 'budget_exhausted', 'budget_exhausted_kind': 'ocr',
                            'raw_artifact_path': 'acquisition/raw', 'unprocessed_pages': [2, 4]}],
             'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': 'h'}],
             'extraction_attempted_chunk_ids': ['c']}
    gaps = assess_collection_gaps(state)
    assert len(gaps) == 1
    assert gaps[0].kind == 'acquisition_incomplete'
    assert gaps[0].source_id == 's' and gaps[0].budget_kind == 'ocr'
    assert '2' in gaps[0].reason and '4' in gaps[0].reason
    plan = plan_recovery(gaps, state=state, budget={'remaining': {'ocr': 0, 'fetch': 10, 'fetch_ordinary': 10, 'search': 10, 'extraction': 10}})
    assert plan.actions == []
    assert plan.stop_reason == 'budget_exhausted'


def test_nonreadable_shell_text_does_not_hide_fetch_gap_or_trigger_reparse():
    state = {'source_registry': [{'source_id': 's', 'url': 'https://example.org/report', 'final_screening_decision': 'include_for_content_fetch'}],
             'documents': [{'source_id': 's', 'document_id': 'shell', 'clean_text': 'Dashboard Loading...',
                            'raw_artifact_path': 'acquisition/shell', 'content_readable': False, 'is_shell': True,
                            'acquisition_incomplete': True, 'budget_exhausted_kind': 'browser'}]}
    gaps = assess_collection_gaps(state)
    assert any(g.kind == 'acquisition_incomplete' and g.budget_kind == 'browser' for g in gaps)
    assert not any(g.kind == 'parse_missing' for g in gaps)
    plan = plan_recovery(gaps, state=state, budget={'remaining': {'browser': 0, 'fetch': 10, 'fetch_ordinary': 10, 'extraction': 10}})
    assert not plan.actions and plan.stop_reason == 'budget_exhausted'


def test_ordinary_subpool_blocks_new_discovery_but_preserves_existing_span_extraction():
    from data_collection_workflow.workflow_recovery import RecoveryGap
    state = {'source_registry': [{'source_id': 's', 'url': 'https://example.org/report'}]}
    gaps = [RecoveryGap('fetch_failed', 's', 'missing'), RecoveryGap('source_missing', 'r', 'coverage'),
            RecoveryGap('unprocessed_span', 'c', 'extract existing')]
    plan = plan_recovery(gaps, state=state, budget={'remaining': {'search': 5, 'fetch': 10, 'fetch_ordinary': 0, 'extraction': 10}})
    assert [(action.kind, action.target_id) for action in plan.actions] == [('extract', 'c')]
    assert plan.stop_reason is None


def test_completed_same_content_version_resolves_ocr_deferral_independent_of_document_order():
    deferred = {'source_id': 's', 'document_id': 'partial', 'content_hash': 'h', 'clean_text': '12 cases',
                'content_readable': True, 'acquisition_incomplete': True, 'budget_exhausted_kind': 'ocr', 'unprocessed_pages': [2]}
    complete = {**deferred, 'document_id': 'complete', 'acquisition_incomplete': False,
                'acquisition_status': 'readable', 'budget_exhausted_kind': None, 'unprocessed_pages': []}
    for docs in ([deferred, complete], [complete, deferred]):
        state = {'source_registry': [{'source_id': 's'}], 'documents': docs,
                 'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': 'h'}],
                 'extraction_attempted_chunk_ids': ['c']}
        assert not assess_collection_gaps(state)


def test_changed_content_does_not_silently_resolve_old_partial_version():
    deferred = {'source_id': 's', 'document_id': 'partial', 'content_hash': 'old', 'clean_text': '12 cases',
                'content_readable': True, 'acquisition_incomplete': True, 'budget_exhausted_kind': 'ocr', 'unprocessed_pages': [2]}
    complete = {**deferred, 'document_id': 'different', 'content_hash': 'new', 'acquisition_incomplete': False,
                'acquisition_status': 'readable', 'budget_exhausted_kind': None, 'unprocessed_pages': []}
    state = {'source_registry': [{'source_id': 's'}], 'documents': [deferred, complete],
             'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': 'old'},
                                 {'source_id': 's', 'chunk_id': 'c2', 'document_hash': 'new'}],
             'extraction_attempted_chunk_ids': ['c', 'c2']}
    assert any(g.kind == 'acquisition_incomplete' for g in assess_collection_gaps(state))


def test_recovery_plan_can_reuse_exact_cached_fetch_after_ordinary_pool_exhaustion(tmp_path, monkeypatch):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.document_acquisition import PARSER_VERSION
    from data_collection_workflow.workflow_recovery import RecoveryGap
    monkeypatch.setenv('FETCH_MAX_BYTES', '5000000')
    ctx = RunContext(tmp_path, {'universal': {'budget_limits': {'fetch_ordinary': 1, 'fetch': 1}}})
    source = {'source_id': 's', 'url': 'http://127.0.0.1/report', 'final_screening_decision': 'include_for_content_fetch'}
    payload = {'parser_version': PARSER_VERSION, 'url': source['url'], 'max_bytes': 5000000}
    with ctx.activate():
        ctx.call('fetch_ordinary', payload, lambda: {'body': 'saved-response'})
        before = ctx.ledger.snapshot()
        plan = plan_recovery([RecoveryGap('fetch_failed', 's', 'missing in graph')],
                             state={'source_registry': [source]}, budget=ctx.ledger)
        assert [(action.kind, action.target_id) for action in plan.actions] == [('fetch', 's')]
        assert ctx.ledger.snapshot() == before
        monkeypatch.setenv('FETCH_MAX_BYTES', '5000001')
        changed = plan_recovery([RecoveryGap('fetch_failed', 's', 'changed strategy')],
                                state={'source_registry': [source]}, budget=ctx.ledger)
        assert not changed.actions


def test_recovery_plan_preserves_verified_priority_exception_only():
    from data_collection_workflow.workflow_recovery import RecoveryGap
    source = dict(source_id='s', url='https://www.cdc.gov/report', must_fetch=True,
                  source_identity_unverified=False, metadata_only_identity=False,
                  discovery_method='live_search_result', target_verification_status='verified',
                  target_verification_reason='Matches supported task geography and period.',
                  disease_fit='match', geography_fit='match', date_fit='match')
    budget = {'remaining': {'fetch_ordinary': 0, 'fetch': 2, 'extraction': 4}}
    gaps = [RecoveryGap('fetch_failed', 's', 'missing')]
    plan = plan_recovery(gaps, state={'source_registry': [source]}, budget=budget)
    assert len(plan.actions) == 1 and plan.actions[0].kind == 'fetch'
    source['source_identity_unverified'] = True
    assert not plan_recovery(gaps, state={'source_registry': [source]}, budget=budget).actions


def test_recovery_plan_can_reuse_cached_ocr_but_not_changed_language(tmp_path):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.document_acquisition import PARSER_VERSION
    ctx = RunContext(tmp_path, {'universal': {'budget_limits': {'ocr': 1}, 'acquisition': {'ocr_languages': 'eng'}}})
    (tmp_path / 'raw').write_bytes(b'12 cases')
    doc = {'source_id': 's', 'document_id': 'd', 'content_hash': 'hash', 'raw_artifact_path': 'raw',
           'content_readable': True, 'clean_text': '12 cases', 'acquisition_incomplete': True,
           'budget_exhausted_kind': 'ocr', 'unprocessed_pages': [2]}
    state = {'documents': [doc], 'source_registry': [{'source_id': 's'}],
             'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': 'hash'}],
             'extraction_attempted_chunk_ids': ['c']}
    with ctx.activate():
        ctx.call('ocr', {'parser_version': PARSER_VERSION, 'document_hash': 'hash', 'page': 2, 'languages': 'eng', 'render_scale': 3},
                 lambda: {'text': 'cached page', 'words': []})
        plan = plan_recovery(assess_collection_gaps(state), state=state, budget=ctx.ledger)
        assert [(a.kind, a.target_id) for a in plan.actions] == [('reparse', 'd')]
        # Configuration changes alter the cache identity; they cannot reuse another language response.
        ctx.config['universal']['acquisition']['ocr_languages'] = 'fra'
        assert not plan_recovery(assess_collection_gaps(state), state=state, budget=ctx.ledger).actions


def test_denied_fetch_stub_resolves_when_same_source_pdf_is_acquired_but_ocr_is_partial():
    from data_collection_workflow.workflow_recovery import unresolved_acquisition_documents
    stub = {'source_id': 's', 'url': 'https://example.org/report.pdf', 'acquisition_incomplete': True,
            'budget_exhausted_kind': 'fetch_ordinary', 'request_success': False, 'content_readable': False}
    partial = {**stub, 'content_hash': 'pdf', 'budget_exhausted_kind': 'ocr', 'request_success': True,
               'http_status_code': 200, 'content_readable': True, 'unprocessed_pages': [2]}
    assert unresolved_acquisition_documents({'documents': [stub, partial]}) == [partial]


def test_same_text_complete_reparse_updates_partial_metadata_without_losing_document_identity():
    partial = {'source_id': 's', 'document_id': 'old', 'content_hash': 'raw', 'text_hash': 'unchanged',
               'parser_version': 'same', 'clean_text': 'Native text', 'content_readable': True,
               'acquisition_incomplete': True, 'acquisition_status': 'budget_exhausted',
               'budget_exhausted_kind': 'ocr', 'unprocessed_pages': [2], 'fetch_error': 'ocr budget exhausted'}
    complete = {**partial, 'document_id': 'new', 'acquisition_incomplete': False,
                'acquisition_status': 'readable', 'budget_exhausted_kind': None,
                'unprocessed_pages': [], 'fetch_error': None}
    state = {'source_registry': [{'source_id': 's'}], 'documents': [partial],
             'evidence_chunks': [{'source_id': 's', 'document_id': 'old', 'chunk_id': 'c', 'document_hash': 'raw'}],
             'extraction_attempted_chunk_ids': ['c']}
    merged = merge_recovery_delta(state, RecoveryDelta(documents=[complete]))
    assert len(merged['documents']) == 1
    retained = merged['documents'][0]
    assert retained['document_id'] == 'old'
    assert retained['acquisition_incomplete'] is False
    assert retained['fetch_error'] is None
    assert merged['evidence_chunks'][0]['document_id'] == 'old'
    assert not assess_collection_gaps({**state, **merged})
    # Replaying an older partial result cannot regress a completed parse.
    replayed = merge_recovery_delta(merged, RecoveryDelta(documents=[partial]))
    assert replayed == merged


def test_deferred_http_response_cannot_resolve_itself_or_an_identical_copy():
    from data_collection_workflow.workflow_recovery import unresolved_acquisition_documents
    deferred = {'source_id': 's', 'url': 'https://example.org/report', 'content_hash': 'shell',
                'acquisition_incomplete': True, 'budget_exhausted_kind': 'fetch',
                'request_success': True, 'http_status_code': 200, 'content_readable': False}
    assert len(unresolved_acquisition_documents({'documents': [deferred, dict(deferred)]})) == 1


def test_cached_partial_ocr_recovery_targets_exact_content_version_independent_of_order(tmp_path, monkeypatch):
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.document_acquisition import PARSER_VERSION, parse_response
    from data_collection_workflow.workflow_recovery import execute_recovery
    monkeypatch.setenv('ENABLE_LLM_EXTRACTION', 'false')
    ctx = RunContext(tmp_path, {'universal': {'budget_limits': {'ocr': 1}, 'acquisition': {'ocr_languages': 'eng'}}})
    old = parse_response(b'Measles: 12 confirmed cases in France during 2024.', url='https://example.org/report', source_id='s', session_dir=tmp_path)
    new = parse_response(b'Measles: 13 confirmed cases in France during 2024.', url='https://example.org/report', source_id='s', session_dir=tmp_path)
    new.update(acquisition_incomplete=True, acquisition_status='budget_exhausted', budget_exhausted_kind='ocr', unprocessed_pages=[2])
    with ctx.activate():
        ctx.call('ocr', {'parser_version': PARSER_VERSION, 'document_hash': new['content_hash'], 'page': 2, 'languages': 'eng', 'render_scale': 3}, lambda: {'text': '', 'words': []})
        for documents in ([old, new], [new, old]):
            state = {'source_registry': [{'source_id': 's'}], 'documents': documents,
                     'evidence_chunks': [{'source_id': 's', 'chunk_id': str(i), 'document_hash': d['content_hash']} for i, d in enumerate(documents)],
                     'extraction_attempted_chunk_ids': ['0', '1']}
            plan = plan_recovery(assess_collection_gaps(state), state=state, budget=ctx.ledger)
            assert len(plan.actions) == 1 and plan.actions[0].kind == 'reparse'
            assert plan.actions[0].document_hash == new['content_hash']
            delta = execute_recovery(plan, context=ctx, artifacts=state, budget=ctx.ledger)
            assert delta.actions[0]['status'] == 'completed'
            assert [d['content_hash'] for d in delta.documents] == [new['content_hash']]


def test_fetch_recovery_keeps_discovered_resource_identities(tmp_path,monkeypatch):
    from data_collection_workflow.workflow_recovery import execute_recovery,RecoveryPlan,RecoveryAction
    from data_collection_workflow.session_runtime import RunContext
    from data_collection_workflow.document_acquisition import parse_response
    from data_collection_workflow.nodes import content_processing,extraction
    parent={'source_id':'parent','url':'https://example.invalid/index'}
    child={'source_id':'child','url':'https://example.invalid/report','parent_source_id':'parent'}
    doc=parse_response(b'Pertussis in France: 4 confirmed cases during 2024.',url=child['url'],source_id='child',session_dir=tmp_path)
    monkeypatch.setattr(content_processing,'content_fetch_and_parse',lambda state:{
        'source_registry':[parent,child],'documents':[doc]})
    monkeypatch.setattr(extraction,'structured_extraction',lambda state:{'raw_records':[]})
    state={'structured_task':{'disease':'Pertussis'},'source_registry':[parent]}
    runtime=RunContext(tmp_path,{'pipeline_mode':'evidence'})
    with runtime.activate():
        delta=execute_recovery(RecoveryPlan([RecoveryAction('fetch','parent','fetch-parent','coverage')]),context=runtime,artifacts=state,budget=runtime.ledger)
    assert {source['source_id'] for source in delta.sources}=={'parent','child'}
    merged=merge_recovery_delta(state,delta)
    assert {doc['source_id'] for doc in merged['documents']} <= {source['source_id'] for source in merged['source_registry']}
    assert merge_recovery_delta({**state,**merged},delta)==merged
