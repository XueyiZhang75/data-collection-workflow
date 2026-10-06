"""Concise advisory planning preserves retrieval breadth and deterministic context."""
import json

import pytest

from data_collection_workflow.models import ExecutableSourcePlan
from data_collection_workflow.nodes.task_scope import _build_executable_source_plan_prompt, _merge_llm_executable_plan


CORE_QUERY_FIELDS = {'query_id', 'query', 'provider_channel', 'source_type', 'role_hint', 'rationale'}
REPEATED_QUERY_FIELDS = {'expected_fields', 'disease_terms_used', 'location_terms_used', 'time_terms_used'}
DETERMINISTIC_SECTIONS = {'source_discovery_objectives', 'planned_source_categories', 'source_planning_risks'}


@pytest.fixture(autouse=True)
def universal_pipeline(monkeypatch):
    monkeypatch.setenv('PIPELINE_MODE', 'evidence')


def reference_plan(count=13, disease='dengue', country='Brazil'):
    families = [
        ('international_organization_report', 'official_site_search', 'official_site'),
        ('official_public_health_agency', 'official_site_search', 'domain_limited'),
        ('structured_database', 'database_search', 'database'),
        ('peer_reviewed_literature', 'literature_api', 'literature'),
    ]
    fields = ['cases_confirmed', 'cases_probable', 'deaths', 'hospitalizations', 'reporting_period',
              'country', 'subnational', 'age', 'sex', 'test_positivity', 'population_denominator']
    rows = []
    for index in range(count):
        source_type, channel, query_type = families[index % len(families)]
        rows.append({
            'query_id': f'q_{index}', 'query': f'{disease} {country} surveillance cases {2025 + index}',
            'provider_channel': channel, 'query_type': query_type, 'source_type': source_type,
            'role_hint': 'collection' if index % 2 else 'validation', 'priority': 0 if index == 0 else index % 4 + 1,
            'rationale': 'Retrieve a distinct task-grounded surveillance source.',
            'expected_fields': fields, 'disease_terms_used': [disease], 'location_terms_used': [country],
            'time_terms_used': ['2025-01-01', '2025-12-31'], 'query_language': 'pt' if index == 0 else 'en',
            'jurisdiction_hint': country, 'official_domain_hint': 'health.example' if index == 0 else None,
            'localized_source_hint': index == 0,
        })
    return ExecutableSourcePlan(
        plan_id='fixture_plan', disease=disease, location=country, time_window='2025',
        generation_method='deterministic_executable_source_plan', target_fields=fields,
        source_discovery_objectives=[{'objective_id': 'objective', 'objective': 'Find independent surveillance evidence.',
            'source_role_hint': 'collection', 'rationale': 'Task coverage requires independently bound evidence.'}],
        planned_source_categories=[{'source_category_id': f'category_{i}', 'source_type': source_type,
            'role_hint': 'collection', 'expected_fields': fields, 'why_relevant': 'Provides independent evidence.'}
            for i, (source_type, _, _) in enumerate(families)],
        planned_queries=rows,
        source_planning_risks=[{'risk_id': 'reporting_gap', 'risk': 'Reporting may be incomplete.',
            'mitigation': 'Preserve missingness and consult independent sources.'}],
    )


def build_prompt(plan):
    return _build_executable_source_plan_prompt(
        user_request=f'Collect {plan.disease} evidence for {plan.location}.',
        spec={'disease': plan.disease, 'geography': plan.location, 'time_window': plan.time_window,
              'target_fields': plan.target_fields},
        profile={'disease_standard_name': plan.disease}, strategy={}, schema_dict={},
        deterministic_plan=plan, task_acceptance_contract={},
    )


@pytest.mark.parametrize('count,disease,country', [(13, 'dengue', 'Brazil'), (24, 'measles', 'Canada')])
def test_prompt_reference_omits_repeated_fields_without_reducing_query_slots(count, disease, country):
    plan = reference_plan(count, disease, country)
    _, prompt = build_prompt(plan)
    reference = json.loads(prompt)['reference_deterministic_plan_shape']
    assert set(reference) == {'plan_id', 'disease', 'generation_method', 'planned_queries'}
    assert len(reference['planned_queries']) == count
    assert [row['query'] for row in reference['planned_queries']] == [row.query for row in plan.planned_queries]
    assert {row['source_type'] for row in reference['planned_queries']} == {row.source_type for row in plan.planned_queries}
    for compact, original in zip(reference['planned_queries'], plan.planned_queries):
        assert CORE_QUERY_FIELDS <= compact.keys()
        assert not REPEATED_QUERY_FIELDS & compact.keys()
        for key in ('provider_channel', 'source_type', 'role_hint', 'query_type', 'priority',
                    'query_language', 'jurisdiction_hint'):
            assert compact[key] == getattr(original, key)
    first = reference['planned_queries'][0]
    assert first['official_domain_hint'] == 'health.example'
    assert first['localized_source_hint'] is True
    assert first['query_language'] == 'pt'


def test_response_contract_requests_compact_fields_and_retains_query_breadth():
    _, prompt = build_prompt(reference_plan())
    contract = json.loads(prompt)['response_contract']
    assert set(contract['top_level_fields']) == {'plan_id', 'disease', 'generation_method', 'planned_queries'}
    assert set(contract['required_query_fields']) == CORE_QUERY_FIELDS
    assert REPEATED_QUERY_FIELDS <= set(contract['omit_repeated_query_fields'])
    assert DETERMINISTIC_SECTIONS <= set(contract['omit_top_level_sections'])
    assert contract['preserve_query_breadth'] is True
    assert contract['rationale_style'] == 'one short sentence'


def test_minimal_schema_validated_reply_retains_deterministic_objectives_categories_and_risks():
    plan = reference_plan()
    minimal = {'plan_id': 'llm_plan', 'disease': plan.disease, 'generation_method': 'llm_executable_source_plan',
               'planned_queries': [{key: row.model_dump()[key] for key in CORE_QUERY_FIELDS} for row in plan.planned_queries]}
    # The real shared structured helper applies schema defaults before this merge.
    validated = ExecutableSourcePlan.model_validate(minimal).model_dump()
    result = _merge_llm_executable_plan(validated, plan)
    for field in DETERMINISTIC_SECTIONS:
        assert getattr(result, field) == getattr(plan, field)
    assert len(result.planned_queries) == len(plan.planned_queries)
    assert {row.source_type for row in result.planned_queries} == {row.source_type for row in plan.planned_queries}
    assert result.target_fields == plan.target_fields
    assert result.location == plan.location
    assert result.time_window == plan.time_window
