"""Configured evidence integration with real local browser/PDF/OCR readiness, no paid calls."""
import argparse
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    spec = importlib.util.spec_from_file_location('universal_integration_runner', ROOT / 'scripts/run_workflow.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _offline_config(tmp_path):
    from data_collection_workflow.runtime_profile import default_workflow_run_config
    config = default_workflow_run_config()
    config['pipeline_mode'] = 'evidence'
    config['user_request'] = 'Collect aggregate measles surveillance in France during 2024.'
    config['structured_task'] = {'disease': 'measles', 'location': 'France', 'start_date': '2024-01-01', 'end_date': '2024-12-31', 'collection_objective': 'aggregate surveillance'}
    config['source_search']['enabled'] = False
    config['live_web']['enabled'] = False
    config['llm'].update(source_planning_enabled=False, source_critic_enabled=False, structured_extraction_enabled=False)
    config['llm']['source_identity']['enabled'] = False
    config['disease_intelligence'].update(llm_enabled=False, force_llm=False)
    config['output'].update(run_output_root=str(tmp_path), sessionized=True, session_id='integration', write_latest_alias=False, write_latest_console_alias=False)
    return config


@pytest.mark.skipif(not (ROOT / '.runtime/acquisition-paths.json').exists(), reason='requires installed local acquisition runtime')
def test_configured_offline_graph_manifest_and_explicit_completed_resume(tmp_path, monkeypatch):
    runner = _runner()
    config = _offline_config(tmp_path)
    monkeypatch.setattr(runner, '_config_with_cli_overrides', lambda args: (None, config))
    import data_collection_workflow.llm_clients as clients
    def forbidden(*args, **kwargs):
        raise AssertionError('offline integration must not dispatch a model')
    monkeypatch.setattr(clients, 'build_chat_model', forbidden)
    args = argparse.Namespace(config=None, resume_session=None, live_status=False, write_run_notebook=False)
    summary = runner.run_workflow(args)
    session = tmp_path / 'sessions/integration'
    readiness = json.loads((session / 'acquisition-preflight.json').read_text())
    assert readiness['ready'] and readiness['dynamic_page'] and readiness['scanned_page']
    manifest = json.loads((session / 'result_manifest.json').read_text())
    assert manifest == summary['result_manifest']
    assert manifest['counts']['discovered_sources'] == 0
    assert manifest['counts']['qualified_observations'] == 0
    assert manifest['data_availability'] == 'no_usable_evidence'
    assert manifest['release_status'] == 'not_evaluated'
    assert manifest['budget']['used'] == {}
    assert not (tmp_path / 'latest').exists()
    events_path = session / 'diagnostics/run_events.ndjson'
    previous_events = events_path.read_bytes()
    args.resume_session = 'integration'
    resumed = runner.run_workflow(args)
    assert resumed['result_manifest'] == manifest
    assert events_path.read_bytes().startswith(previous_events)
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len({row['sequence'] for row in events}) == len(events)
    package = json.loads((session / 'collection/final_package.json').read_text())
    assert package['result_manifest'] == manifest
    assert package['final_dataset'] == []
    assert package['candidate_records'] == []


def test_recovery_alias_merge_rebinds_all_supported_evidence_reference_keys():
    from data_collection_workflow.workflow_recovery import merge_recovery_delta, RecoveryDelta
    chunk = {'chunk_id': 'old', 'source_id': 'a', 'document_hash': 'v1', 'text': '4 cases'}
    state = {'source_registry': [{'source_id': 'a', 'url': 'https://example.org/report'}], 'evidence_chunks': [chunk]}
    delta = RecoveryDelta(sources=[{'source_id': 'b', 'url': 'https://example.org/report'}], chunks=[{**chunk, 'source_id': 'b', 'chunk_id': 'new'}], records=[{'record_id': 'r', 'source_id': 'b', 'evidence_chunk_id': 'new', 'field_provenance_json': {'cases_confirmed': {'evidence_span_id': 'new'}}}])
    merged = merge_recovery_delta(state, delta)
    references = {chunk['chunk_id'] for chunk in merged['evidence_chunks']}
    record = merged['raw_records'][0]
    assert record['evidence_chunk_id'] in references
    assert record['field_provenance_json']['cases_confirmed']['evidence_span_id'] in references


@pytest.mark.skipif(not (ROOT / '.runtime/acquisition-paths.json').exists(), reason='requires installed local acquisition runtime')
def test_configured_explicit_resume_continues_checkpoint_without_reintake(tmp_path, monkeypatch):
    runner = _runner()
    config = _offline_config(tmp_path)
    monkeypatch.setattr(runner, '_config_with_cli_overrides', lambda args: (None, config))
    import data_collection_workflow.graph as graph
    import data_collection_workflow.llm_clients as clients
    def forbidden(*args, **kwargs):
        raise AssertionError('offline integration must not dispatch a model')
    monkeypatch.setattr(clients, 'build_chat_model', forbidden)
    original_intake = graph.task_intake_and_scope_planning
    original_finalize = graph.final_data_package_builder
    visits = {'intake': 0, 'finalize': 0}
    def intake(state):
        visits['intake'] += 1
        return original_intake(state)
    def finalize(state):
        visits['finalize'] += 1
        if visits['finalize'] == 1:
            raise RuntimeError('integration interruption before finalization')
        return original_finalize(state)
    monkeypatch.setattr(graph, 'task_intake_and_scope_planning', intake)
    monkeypatch.setattr(graph, 'final_data_package_builder', finalize)
    args = argparse.Namespace(config=None, resume_session=None, live_status=False, write_run_notebook=False)
    with pytest.raises(RuntimeError, match='integration interruption'):
        runner.run_workflow(args)
    assert visits == {'intake': 1, 'finalize': 1}
    events = tmp_path / 'sessions/integration/diagnostics/run_events.ndjson'
    prior = events.read_bytes()
    args.resume_session = 'integration'
    summary = runner.run_workflow(args)
    assert visits == {'intake': 1, 'finalize': 2}
    assert events.read_bytes().startswith(prior)
    assert summary['result_manifest']['technical_completion'] == 'completed'
    assert summary['result_manifest']['budget']['used'] == {}


def test_recovery_document_alias_preserves_real_field_qualification():
    import hashlib
    from data_collection_workflow.workflow_recovery import merge_recovery_delta, RecoveryDelta
    from data_collection_workflow.evidence_qualification import assess_record_evidence, build_evidence_index
    text = 'Pertussis in Washington, United States: 12 confirmed cases during 2024.'
    digest = hashlib.sha256(text.encode()).hexdigest()
    doc = {'document_id': 'old_doc', 'source_id': 's1', 'content_hash': digest, 'text_hash': digest, 'parser_version': 'v1', 'clean_text': text}
    chunk = {'chunk_id': 'new_chunk', 'source_id': 's2', 'document_id': 'new_doc', 'document_hash': digest, 'text': text, 'char_start': 0, 'char_end': len(text)}
    provenance = {'cases_confirmed': {'source_id': 's2', 'document_id': 'new_doc', 'chunk_id': 'new_chunk', 'quote': text, 'locator': {'document_id': 'new_doc', 'chunk_id': 'new_chunk'}}}
    record = {'record_id': 'r', 'source_id': 's2', 'supporting_chunk_id': 'new_chunk', 'document_id': 'new_doc', 'disease': 'Pertussis', 'country': 'United States', 'subnational_location': 'Washington', 'reporting_period': '2024', 'cases_confirmed': 12, 'field_provenance_json': json.dumps(provenance)}
    state = {'source_registry': [{'source_id': 's1', 'url': 'https://example.org/report'}], 'documents': [doc], 'evidence_chunks': []}
    delta = RecoveryDelta(sources=[{'source_id': 's2', 'url': 'https://example.org/report'}], documents=[{**doc, 'source_id': 's2', 'document_id': 'new_doc'}], chunks=[chunk], records=[record])
    merged = merge_recovery_delta(state, delta)
    row = merged['raw_records'][0]
    field = json.loads(row['field_provenance_json'])['cases_confirmed']
    assert field['document_id'] == field['locator']['document_id'] == 'old_doc'
    qualification = assess_record_evidence(row, contract={}, evidence_index=build_evidence_index(merged))
    assert qualification.status == 'qualified', qualification.to_dict()
    assert all(e.supported and e.locator['document_id'] == 'old_doc' for e in qualification.field_evidence)
    assert merge_recovery_delta(merged, delta) == merged
