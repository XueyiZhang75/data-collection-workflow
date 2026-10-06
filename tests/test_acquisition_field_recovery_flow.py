"""A repair action must actually enrich the original observation from its source."""
import pytest
from test_acquisition_recovery import source_state, adaptive_runtime
from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
from data_collection_workflow.workflow_recovery import (
    RecoveryPlan, assess_collection_gaps, plan_recovery, execute_recovery, merge_recovery_delta,
)


@pytest.mark.parametrize('disease,country', [('measles','France'),('pertussis','Canada')])
def test_field_recovery_updates_original_candidate_and_is_idempotent(tmp_path,monkeypatch,disease,country):
    for key,value in {'PIPELINE_MODE':'evidence','ENABLE_LLM_EXTRACTION':'false',
                      'ENABLE_LLM_SOURCE_IDENTITY':'false','ENABLE_LIVE_FETCH':'false',
                      'ENABLE_LIVE_SEARCH':'false'}.items():
        monkeypatch.setenv(key,value)
    text=f'{country} reported 12 confirmed {disease} cases during 2025.'
    state=source_state(text)
    state['structured_task'].update(disease=disease,location=country)
    original={'record_id':'original-candidate','source_id':'s','supporting_chunk_id':'c',
              'disease':disease,'cases_confirmed':12,'evidence_quote':text}
    original['evidence_qualification']=assess_record_evidence(original,
        contract=state['structured_task'],evidence_index=build_evidence_index(state)).to_dict()
    assert original['evidence_qualification']['status']=='candidate'
    state.update(raw_records=[original],candidate_records=[original],normalized_records=[original],
                 extraction_attempted_chunk_ids=['c'])
    runtime=adaptive_runtime(tmp_path)
    with runtime.activate():
        plan=plan_recovery(assess_collection_gaps(state),state=state,budget=runtime.ledger)
        actions=[action for action in plan.actions if action.target_id in {'original-candidate','c'}]
        assert actions, 'Missing-field diagnosis must produce executable repair work.'
        delta=execute_recovery(RecoveryPlan(actions),context=runtime,artifacts=state,budget=runtime.ledger)
        updated={**state,**merge_recovery_delta(state,delta)}
        repeated={**updated,**merge_recovery_delta(updated,delta)}
    rows=[row for row in updated['raw_records'] if row['record_id']=='original-candidate']
    assert len(rows)==1
    assert rows[0].get('country')==country
    assert rows[0].get('reporting_period')=='2025'
    qualification=assess_record_evidence(rows[0],contract=state['structured_task'],
        evidence_index=build_evidence_index(updated))
    assert qualification.status=='qualified',qualification.to_dict()
    assert len(updated['raw_records'])==1, 'Do not append a new duplicate for the same repaired observation.'
    assert repeated['raw_records']==updated['raw_records']
    assert delta.actions and all(action['status'] not in {'skipped','failed','budget_deferred'}
                                 for action in delta.actions)


def test_actual_recovery_nodes_reach_finalization_with_no_external_services(tmp_path,monkeypatch):
    from data_collection_workflow.workflow_recovery import recovery_control,recovery_execute
    from data_collection_workflow.nodes.extraction import schema_validation_and_repair
    from data_collection_workflow.nodes.normalization import record_normalization
    from data_collection_workflow.nodes.linking_validation import record_linking,cross_source_consistency_check,quality_gate_routing
    from data_collection_workflow.nodes.finalization import final_data_package_builder
    for key,value in {'PIPELINE_MODE':'evidence','ENABLE_LLM_EXTRACTION':'false',
        'ENABLE_LLM_SOURCE_IDENTITY':'false','ENABLE_LLM_DISEASE_INTELLIGENCE':'false',
        'ENABLE_LIVE_FETCH':'false','ENABLE_LIVE_SEARCH':'false','ENABLE_HUMAN_REVIEW':'false'}.items():
        monkeypatch.setenv(key,value)
    state=source_state()
    state['source_registry'][0].update(canonical_url='https://offline.invalid/report',status='fetched')
    state.update(collection_trace=[],human_review_queue=[],collection_spec={
        'disease':'measles','geography':'France','start_date':'2025-01-01','end_date':'2025-12-31'})
    runtime=adaptive_runtime(tmp_path)
    with runtime.activate():
        for _ in range(12):
            state.update(recovery_control(state))
            if not state['recovery_plan']['actions']:
                break
            for node in (recovery_execute,schema_validation_and_repair,record_normalization,
                         record_linking,cross_source_consistency_check,quality_gate_routing):
                state.update(node(state))
        else:
            pytest.fail('Recovery repeated non-executable work instead of reaching finalization.')
        state.update(final_data_package_builder(state))
    assert state['recovery_stop_reason']
    package=state['final_data_package']
    assert len(package['final_dataset'])==1
    assert package['final_dataset'][0]['cases_confirmed']==12
    assert package['final_dataset'][0]['country']=='France'
    assert not any(value for kind,value in runtime.ledger.snapshot()['used'].items()
                   if kind in {'http_requests','search','extraction'})
