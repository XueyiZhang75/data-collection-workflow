"""Positive real-node evidence flow with only the model endpoint replaced."""
import json
from copy import deepcopy

import pytest

from data_collection_workflow import llm_clients
from data_collection_workflow.export import export_final_data_package
from data_collection_workflow.models import Document, LLMExtractionOutput
from data_collection_workflow.nodes.content_processing import (
    document_quality_check, evidence_chunking_and_data_presence_flagging,
)
from data_collection_workflow.nodes.extraction import structured_extraction, schema_validation_and_repair
from data_collection_workflow.nodes.normalization import record_normalization
from data_collection_workflow.nodes.linking_validation import (
    record_linking, cross_source_consistency_check, quality_gate_routing,
)
from data_collection_workflow.nodes.finalization import final_data_package_builder
from data_collection_workflow.document_acquisition import parse_response
from data_collection_workflow.session_runtime import RunContext


PROSE_QUOTE = 'Canada reported 12 confirmed pertussis cases during 2025.'
TABLE_HEADING = 'Pertussis surveillance in Canada during 2025'


@pytest.mark.parametrize('layout', ['prose', 'html_table'])
def test_real_positive_evidence_chain_exports_qualified_aggregate(tmp_path, monkeypatch, layout):
    for key, value in {
        'PIPELINE_MODE': 'evidence',
        'ENABLE_LLM_EXTRACTION': 'true',
        'LLM_FALLBACK_TO_RULE_BASED': 'false',
        'LLM_PROVIDER': 'openai', 'LLM_MODEL': 'offline-response-fixture',
        'ENABLE_LLM_SOURCE_IDENTITY': 'false',
        'ENABLE_LLM_DISEASE_INTELLIGENCE': 'false',
        'ENABLE_HUMAN_REVIEW': 'false',
    }.items():
        monkeypatch.setenv(key, value)
    quote = PROSE_QUOTE if layout == 'prose' else '12'
    html = (f'<html><main><h1>Pertussis surveillance</h1><p>{PROSE_QUOTE}</p></main></html>'
            if layout == 'prose' else
            f'<html><main><h1>{TABLE_HEADING}</h1><table><tr><th>Confirmed cases</th></tr>'
            '<tr><td>12</td></tr></table></main></html>')
    url = 'https://www.canada.ca/en/public-health/offline-positive-fixture'
    source = dict(source_id='positive_source', url=url, canonical_url=url,
                  publisher='Government of Canada', source_type='official_public_health_agency',
                  source_role='data_source', source_role_final='collection', status='fetched',
                  fetch_purpose='data_extraction', credibility_score=0.9, credibility_level='high')
    parsed = parse_response(html.encode(), url=url, source_id=source['source_id'],
                            session_dir=tmp_path / 'parsed', content_type='text/html')
    document = Document(**{**source, **parsed}).model_dump()
    state = {
        'structured_task': {'disease': 'Pertussis', 'location': 'Canada',
                            'start_date': '2025-01-01', 'end_date': '2025-12-31'},
        'collection_spec': {'disease': 'Pertussis', 'geography': 'Canada',
                            'start_date': '2025-01-01', 'end_date': '2025-12-31',
                            'target_population': 'humans'},
        'source_registry': [source], 'documents': [document],
        'human_review_queue': [], 'collection_trace': [],
    }
    response = {'chunk_is_relevant': True, 'records': [{
        'disease': 'Pertussis', 'country': 'Canada', 'reporting_period': '2025',
        'cases_confirmed': 12, 'evidence_quote': quote,
        'source_row_id': '1' if layout == 'html_table' else None,
        'field_provenance_json': {
            'disease': {'quote': 'pertussis' if layout == 'prose' else 'Pertussis'},
            'country': {'quote': 'Canada'}, 'reporting_period': {'quote': '2025'},
            'cases_confirmed': {'quote': '12 confirmed pertussis cases' if layout == 'prose' else '12'},
        },
    }]}
    invocations = []

    class ResponseModel:
        model_name = 'offline-response-fixture'

        def with_structured_output(self, schema):
            assert schema is LLMExtractionOutput
            return self

        def invoke(self, messages, config=None):
            invocations.append(messages)
            evidence_text = messages[-1]['content'].split('Evidence chunk text:', 1)[1].strip()
            has_observation = PROSE_QUOTE in evidence_text if layout == 'prose' else '12' in evidence_text
            return deepcopy(response) if has_observation else {'chunk_is_relevant': False, 'records': []}

    monkeypatch.setattr(llm_clients, 'build_chat_model', lambda *args, **kwargs: ResponseModel())
    runtime = RunContext(tmp_path / 'session', {
        'pipeline_mode': 'evidence',
        'universal': {'budget_limits': {'model': 8, 'extraction': 8}, 'extraction_reserve': 0},
    })
    with runtime.activate():
        for node in (document_quality_check, evidence_chunking_and_data_presence_flagging,
                     structured_extraction, schema_validation_and_repair, record_normalization,
                     record_linking, cross_source_consistency_check, quality_gate_routing,
                     final_data_package_builder):
            state.update(node(state))
        package = state['final_data_package']
        exported = export_final_data_package(package, tmp_path / 'export')

    assert invocations, state.get('llm_extraction_summary')
    assert state['raw_records'], state.get('extraction_summary')
    assert len(package['aggregate_dataset']) == 1, json.dumps([
        record.get('evidence_qualification') for record in package.get('candidate_records', [])
    ], indent=2)
    record = package['aggregate_dataset'][0]
    assert record['cases_confirmed'] == 12
    assert record['country'] == 'Canada'
    assert record['reporting_period'] == '2025'
    assert package['final_case_dataset'] == []
    assert package['result_manifest']['counts']['qualified_observations'] == 1
    assert package['candidate_records'] == []
    assert record['supporting_chunk_id'] in {chunk['chunk_id'] for chunk in state['evidence_chunks']}
    fields = record['evidence_qualification']['field_evidence']
    assert {'disease', 'country', 'reporting_period', 'cases_confirmed'} <= {
        field['field'] for field in fields
    }
    assert all(field['supported'] and field['document_hash'] == parsed['content_hash']
               and field['locator']['chunk_id'] == record['supporting_chunk_id']
               for field in fields)
    retained = json.loads((tmp_path / 'export' / 'aggregate_dataset.json').read_text(encoding='utf-8'))
    assert len(retained) == 1 and retained[0]['cases_confirmed'] == 12
    assert exported
