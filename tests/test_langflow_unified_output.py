"""Current report links and portable materials through the optional local API."""
import json
from pathlib import Path
from urllib.parse import urljoin

import pytest

from data_collection_workflow.langflow_demo import (
    ArtifactAccessError, RunRegistry, VISUAL_CONTRACT_VERSION,
    build_run_snapshot, create_app, resolve_artifact_path,
)
from data_collection_workflow.reporting import write_unified_report


@pytest.fixture
def completed_session(tmp_path):
    session = tmp_path / 'session'
    package = {
        'final_dataset': [],
        'candidate_records': [{'record_id': 'candidate-one', 'source_id': 'source-one',
                               'source_url': 'https://example.org/report',
                               'evidence_quote': 'Canada reported 2 measles cases.',
                               'evidence_qualification': {'status': 'candidate'}}],
        'source_registry': [{'source_id': 'source-one', 'url': 'https://example.org/report',
                             'title': 'Example report'}],
        'result_manifest': {'task': {'disease': 'measles', 'location': 'Canada',
                                    'start_date': '2025-01-01', 'end_date': '2025-12-31'}},
    }
    paths = write_unified_report(session, package=package)
    summary = {'artifact_paths': paths}
    (session / 'workflow_run_summary.json').write_text(json.dumps(summary), encoding='utf-8')
    registry = RunRegistry()
    registry.register(session_id='session', session_dir=session)
    registry.mark_completed('session', summary)
    from fastapi.testclient import TestClient
    client = TestClient(create_app(registry=registry, api_base_url='http://testserver'))
    return session, paths, client


def test_current_report_and_bundle_are_linked_and_served(completed_session):
    session, paths, client = completed_session
    response = client.get('/runs/session', params={'visual_contract_version': VISUAL_CONTRACT_VERSION})
    assert response.status_code == 200
    snapshot = response.json()
    for key, label in [('final_report_english', 'Final report'), ('report_bundle', 'Report bundle')]:
        url = snapshot['artifact_urls'][key]
        assert f'{label}: {url}' in snapshot['result_location_text']
        downloaded = client.get(url)
        assert downloaded.status_code == 200
        assert downloaded.content == Path(paths[key]).read_bytes()
    assert 'Run report: pending' not in snapshot['result_location_text']
    assert 'Interpretive report: pending' not in snapshot['result_location_text']


def test_report_relative_data_and_evidence_links_use_only_packaged_material(completed_session):
    session, paths, client = completed_session
    report_url = 'http://testserver/runs/session/artifacts/final_report_english'
    evidence = next((session / 'evidence').glob('*_evidence.json'))
    for relative in ['data/final_dataset.csv', 'data/report_snapshot.json', evidence.relative_to(session).as_posix()]:
        response = client.get(urljoin(report_url, relative))
        assert response.status_code == 200
        assert response.content == (session / relative).read_bytes()
    # A file merely present on disk is not a member of the published report.
    (session / 'data/private.json').write_text('{"private": true}', encoding='utf-8')
    for relative in ['data/private.json', 'data/%2e%2e/workflow_run_summary.json',
                     'evidence/%2e%2e/workflow_run_summary.json', 'data/%5c..%5cworkflow_run_summary.json']:
        assert client.get(urljoin(report_url, relative)).status_code == 404


@pytest.mark.parametrize('key', ['final_report_english', 'report_bundle'])
def test_report_artifact_cannot_escape_session(tmp_path, key):
    session = tmp_path / 'session'
    session.mkdir()
    outside = tmp_path / 'outside.html'
    outside.write_text('outside', encoding='utf-8')
    (session / 'workflow_run_summary.json').write_text(
        json.dumps({'artifact_paths': {key: str(outside)}}), encoding='utf-8')
    with pytest.raises(ArtifactAccessError):
        resolve_artifact_path(session, key)
    snapshot = build_run_snapshot(session_id='session', session_dir=session,
                                  api_base_url='http://testserver', status='completed')
    assert key not in snapshot['artifact_urls']


def test_langflow_final_results_component_shows_current_report_links():
    from test_langflow_demo_adapter import _load_langflow_deep_links_component
    module = _load_langflow_deep_links_component(Path(__file__).resolve().parents[1])
    snapshot = {'session_id': 'session', 'status': 'completed', 'artifact_urls': {
        'final_report_english': 'http://testserver/report',
        'report_bundle': 'http://testserver/bundle',
    }}
    text = module.format_deep_links_text(snapshot)
    assert 'http://testserver/report' in text
    assert 'http://testserver/bundle' in text
