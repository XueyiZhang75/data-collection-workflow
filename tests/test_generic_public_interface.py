"""Canonical public entrypoints preserve explicit task identities."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('entrypoint', ['run_workflow.py', 'start_studio.py'])
def test_neutral_entrypoints_support_help_without_starting_work(entrypoint):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts' / entrypoint), '--help'],
                            cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--config' in result.stdout


def test_public_graph_uses_single_canonical_entrypoint():
    graphs = json.loads((ROOT / 'langgraph.json').read_text(encoding='utf-8'))['graphs']
    assert graphs == {'data_collection_workflow': './src/data_collection_workflow/studio_app.py:graph'}


def test_generic_record_keeps_legacy_type_identity_and_disease_value():
    from data_collection_workflow import models
    assert hasattr(models, 'ObservationRecord')
    assert models.HantavirusRecord is models.ObservationRecord
    record = models.ObservationRecord(record_id='test', disease='mpox', cases_confirmed=2)
    assert record.model_dump()['disease'] == 'mpox'
    assert models.ObservationRecord.model_json_schema()['title'] == 'ObservationRecord'
    assert issubclass(models.PublicHealthRecord, models.ObservationRecord)


@pytest.mark.parametrize('disease', [None, 'measles', 'chikungunya'])
def test_source_planning_does_not_invent_a_default_disease(disease):
    from data_collection_workflow.agents.source_planning_agent import _normalize_plan
    plan = _normalize_plan({}, {'disease': disease} if disease else {}, {}, {}, {})
    assert plan['disease'] == (disease or '')


def test_generic_extraction_policy_leaves_unspecified_disease_unresolved():
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _build_extraction_context, _repair_record
    policy = StructuredExtractionPolicy.model_validate(load_structured_extraction_policy())
    assert policy.default_disease == ''
    assert _build_extraction_context({}, policy)['disease_standard_name'] == ''
    repaired, actions = _repair_record({'record_id': 'missing-disease'}, policy)
    assert not repaired.get('disease')
    assert 'set_disease_to_default' not in actions
    assert _build_extraction_context({'structured_task': {'disease': 'mpox'}}, policy)['disease_standard_name'] == 'mpox'


def test_missing_disease_cannot_pass_schema_validation_by_default_repair():
    from data_collection_workflow.config import load_structured_extraction_policy
    from data_collection_workflow.models import StructuredExtractionPolicy
    from data_collection_workflow.nodes.extraction import _validate_record
    policy = StructuredExtractionPolicy.model_validate(load_structured_extraction_policy())
    record = dict(record_id='missing', cases_confirmed=2, country='Canada',
                  date_reported='2025-01-02', source_id='s', source_url='https://example.org',
                  supporting_chunk_id='chunk', evidence_quote='Two cases reported in Canada.')
    repaired, result = _validate_record(record, policy)
    assert not repaired.get('disease')
    assert result.schema_status == 'rejected'
    assert 'disease' in result.missing_fields


def test_advisory_planning_does_not_inject_an_unrequested_case_catalog():
    from data_collection_workflow.agents.source_planning_agent import _build_user_payload
    payload = _build_user_payload('Collect measles in Canada', {'disease':'measles'}, {}, {}, {})
    assert payload['known_seed_source_summary'] == []


def test_advisory_planning_uses_only_explicit_current_run_sources():
    from data_collection_workflow.agents.source_planning_agent import _build_user_payload, _normalize_plan
    strategy = {'current_run_sources':[{'source_id':'current', 'url':'https://example.org/current', 'title':'Current task evidence'}]}
    payload = _build_user_payload('Collect measles in Canada', {'disease':'measles'}, {}, strategy, {})
    assert [s['source_id'] for s in payload['known_seed_source_summary']] == ['current']
    plan = _normalize_plan({'candidate_source_hints':['https://example.org/current', 'https://example.org/unverified']}, {'disease':'measles'}, {}, strategy, {})
    assert [s['agent_proposed_unverified'] for s in plan['candidate_source_hints']] == [False, True]
