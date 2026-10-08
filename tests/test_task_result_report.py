import json

from data_collection_workflow.task_result_report import build_task_result, render_task_result
from data_collection_workflow.result_manifest import build_result_manifest, write_universal_run_outputs


TASK = {
    'disease': 'mpox', 'location': 'Sierra Leone',
    'start_date': '2025-01-01', 'end_date': '2025-12-31',
    'target_fields': ['cases_confirmed', 'deaths'],
}


def observation(value, *, status='qualified', period='2025', scope='Sierra Leone',
                as_of=None, semantics='cumulative', record_id='one', quote=None):
    quote = quote or f'Mpox in Sierra Leone: {value:,} confirmed cases in 2025.'
    fields = ['disease', 'geographic_scope', 'cases_confirmed']
    if period:
        fields.append('reporting_period')
    return {
        'record_id': record_id, 'disease': 'mpox', 'country': 'Sierra Leone',
        'geographic_scope': scope, 'geographic_scope_type': 'country' if scope == 'Sierra Leone' else 'subnational',
        'reporting_period': period, 'as_of_date': as_of,
        'cases_confirmed': value, 'count_semantics': semantics,
        'source_url': 'https://example.org/report', 'evidence_quote': quote,
        'evidence_qualification': {
            'status': status, 'reasons': [] if status == 'qualified' else ['cases_confirmed:unbound_field_value'],
            'field_evidence': [
                {'field': field, 'document_hash': 'abc123', 'locator': {'chunk_id': 'chunk-1'}, 'quote': quote}
                for field in fields
            ],
        },
    }


def report(qualified=(), candidates=(), task=None):
    package = {'final_dataset': list(qualified), 'candidate_records': list(candidates)}
    return build_task_result(package, {'task': task or TASK, 'coverage_status': 'incomplete'})


def test_full_scope_qualified_total_answers_task_with_source():
    result = report([observation(5442)])
    answer = result['answers']['cases_confirmed']
    assert answer['status'] == 'confirmed'
    assert answer['value'] == 5442
    assert answer['sources'][0]['url'] == 'https://example.org/report'
    assert '5,442' in render_task_result(result)


def test_partial_snapshot_and_unqualified_year_end_lead_do_not_become_annual_answer():
    snapshot = observation(3682, period=None, as_of='2025-06-04', record_id='snapshot')
    lead = observation(
        5442, status='candidate', period='2025-01-09 to 2025-12-17',
        record_id='lead', quote='Since the first mpox case, Sierra Leone reported 5,442 confirmed cases by 17 December 2025.',
    )
    result = report([snapshot], [lead])
    answer = result['answers']['cases_confirmed']
    assert answer['status'] == 'unconfirmed'
    assert answer['value'] is None
    assert answer['candidate_leads'][0]['value'] == 5442
    text = render_task_result(result)
    assert 'No total for the requested scope is confirmed' in text
    assert 'unverified leads' in text
    assert '3,682' in text and '5,442' in text


def test_local_subset_average_and_foreign_claim_cannot_answer_country_total():
    local = observation(43, scope='Western Urban Area', record_id='local')
    daily = observation(29, period='2025', semantics='daily_average', record_id='daily')
    foreign = observation(100, record_id='foreign')
    foreign['country'] = foreign['geographic_scope'] = 'Liberia'
    result = report([local, daily, foreign])
    assert result['answers']['cases_confirmed']['value'] is None
    assert result['answers']['cases_confirmed']['status'] == 'unconfirmed'


def test_conflicting_qualified_totals_require_review():
    a = observation(100, record_id='a')
    b = observation(120, record_id='b')
    b['source_url'] = 'https://example.org/other'
    result = report([a, b])
    assert result['answers']['cases_confirmed']['status'] == 'conflict'
    assert result['answers']['cases_confirmed']['value'] is None
    assert {item['value'] for item in result['answers']['cases_confirmed']['conflicting_values']} == {100, 120}


def test_missing_is_unknown_not_zero_and_unsupported_candidate_is_not_a_lead():
    bad = observation(77, status='candidate', quote='77 people in another country.')
    result = report([], [bad])
    answer = result['answers']['cases_confirmed']
    assert answer['status'] == 'unconfirmed'
    assert answer['value'] is None
    assert answer['candidate_leads'] == []
    assert result['answers']['deaths']['value'] is None


def test_run_writer_keeps_task_json_and_uses_single_unified_report(tmp_path):
    row = observation(5442)
    package = {
        'final_dataset': [row], 'aggregate_dataset': [row], 'final_case_dataset': [],
        'candidate_records': [], 'context_records': [], 'source_registry': [],
    }
    package['result_manifest'] = build_result_manifest(package, {'structured_task': TASK})
    summary = {'artifact_paths': {}}
    write_universal_run_outputs(package, summary, tmp_path)
    artifacts = summary['artifact_paths']
    task_result = json.loads((tmp_path / 'task_result.json').read_text(encoding='utf-8'))
    assert task_result['answers']['cases_confirmed']['value'] == 5442
    assert 'task_result_chinese' not in artifacts
    assert (tmp_path / 'final_report.html').is_file()
    assert not (tmp_path / 'task_result.md').exists()
    assert not (tmp_path / 'collection' / 'final_report.md').exists()
    saved = json.loads((tmp_path / 'collection' / 'final_package.json').read_text(encoding='utf-8'))
    assert saved['artifact_manifest']['files']['final_report_english'] == artifacts['final_report_english']
    assert 'task_result_chinese' not in saved['artifact_manifest']['files']



def test_interactive_completion_displays_task_answer_and_report_path(capsys):
    from scripts import collect as interactive
    interactive._print_task_result_summary({
        'artifact_paths': {'task_result_english': 'outputs/session/task_result.md'},
        'task_result_summary': {'headline': 'Confirmed: confirmed cases 5,442.'},
    })
    text = capsys.readouterr().out
    assert 'task_result_report:' in text
    assert '5,442' in text


def test_year_end_candidate_without_year_in_quote_is_a_lead_and_ranks_above_older_counts():
    older = observation(4400, status='candidate', period='2025', as_of='2025-06-30',
                        record_id='older', quote='Sierra Leone reported over 4400 confirmed mpox cases by June 2025.')
    older['source_url'] = 'https://example.org/older'
    latest = observation(5442, status='candidate', period='2025-01-09 to 2025-12-17',
                         as_of='2025-12-17', record_id='latest',
                         quote='Since the first mpox case was recorded on Jan. 9, Sierra Leone has reported 5,442 confirmed cases.')
    result = report([], [older, latest])
    leads = result['answers']['cases_confirmed']['candidate_leads']
    assert [lead['value'] for lead in leads[:2]] == [5442, 4400]
    assert result['answers']['cases_confirmed']['value'] is None
    assert 'over 4,400' in render_task_result(result)


def test_answer_locator_points_to_count_field_not_disease_span():
    row = observation(5442)
    evidence = row['evidence_qualification']['field_evidence']
    evidence[0]['locator'] = {'chunk_id': 'disease-chunk'}
    evidence[2]['locator'] = {'chunk_id': 'count-chunk'}
    answer = report([row])['answers']['cases_confirmed']
    assert answer['sources'][0]['locator']['chunk_id'] == 'count-chunk'


def test_country_aliases_match_but_explicit_country_conflict_blocks_total():
    row = observation(100)
    row['country'] = row['geographic_scope'] = 'United States'
    task = {**TASK, 'location': 'USA'}
    assert report([row], task=task)['answers']['cases_confirmed']['value'] == 100
    bad = observation(200)
    bad['country'] = 'Liberia'
    assert report([bad])['answers']['cases_confirmed']['value'] is None


def test_unrelated_average_in_other_sentence_does_not_reject_annual_total():
    row = observation(100, quote='Mpox in Sierra Leone: 100 confirmed cases in 2025. Average age was 30 years.')
    assert report([row])['answers']['cases_confirmed']['value'] == 100


def test_custom_numeric_indicator_uses_requested_field_without_unrelated_defaults():
    row = observation(100)
    row['vaccination_coverage_percent'] = 87.5
    row['evidence_qualification']['field_evidence'].append({
        'field': 'vaccination_coverage_percent', 'document_hash': 'abc123',
        'locator': {'chunk_id': 'coverage'}, 'quote': 'Vaccination coverage was 87.5% in 2025.',
    })
    task = {**TASK, 'target_fields': ['vaccination_coverage_percent']}
    result = report([row], task=task)
    assert set(result['answers']) == {'vaccination_coverage_percent'}
    assert result['answers']['vaccination_coverage_percent']['value'] == 87.5
    assert '87.5%' in result['answers']['vaccination_coverage_percent']['sources'][0]['quote']


def test_cfr_later_in_same_claim_does_not_turn_total_into_rate():
    row = observation(100, quote='Mpox in Sierra Leone: 100 confirmed cases in 2025, including 2 deaths (CFR 2%).')
    assert report([row])['answers']['cases_confirmed']['value'] == 100


def test_subgroup_is_not_national_population_total():
    row = observation(426)
    row['population_scope'] = 'people living with HIV'
    assert report([row])['answers']['cases_confirmed']['value'] is None


def test_ambiguous_early_as_of_cannot_support_full_year_answer():
    row = observation(100, as_of='June 2025')
    assert report([row])['answers']['cases_confirmed']['value'] is None


def test_headline_surfaces_latest_lead_without_promoting_it_to_answer():
    lead = observation(5442, status='candidate', period='2025-01-09 to 2025-12-17',
                       as_of='2025-12-17', quote='Mpox in Sierra Leone: 5,442 confirmed cases by 17 December 2025.')
    result = report([], [lead])
    assert '5,442' in result['headline']
    assert 'Unverified leads' in result['headline']
    assert result['answers']['cases_confirmed']['value'] is None


def test_headline_prioritizes_deaths_over_unspecified_case_samples():
    total = observation(5442, status='candidate', as_of='2025-12-17',
                        quote='Mpox in Sierra Leone: 5,442 confirmed cases by 17 December 2025.')
    sample = observation(187, status='candidate', as_of='2025-05-31', record_id='sample',
                         quote='We analyzed 187 mpox case reports from Sierra Leone during 2025.')
    sample['cases_confirmed'] = None
    sample['cases_unspecified'] = 187
    death = observation(60, status='candidate', as_of='2025-12-17', record_id='death',
                        quote='Mpox in Sierra Leone: 60 deaths by 17 December 2025.')
    death['cases_confirmed'] = None
    death['deaths'] = 60
    task = {**TASK, 'target_fields': ['cases_confirmed', 'cases_unspecified', 'deaths']}
    result = report([], [total, sample, death], task=task)
    assert 'deaths 60' in result['headline']
    assert result['answers']['cases_unspecified']['candidate_leads'] == []


def test_generic_numeric_or_age_is_not_confirmed_without_metric_identity():
    row = observation(100)
    row['metric_value'] = 87.5
    row['metric_name'] = 'vaccination coverage'
    row['age'] = 30
    row['evidence_qualification']['field_evidence'] += [
        {'field': 'metric_value', 'document_hash': 'abc123', 'locator': {'chunk_id': 'metric'},
         'quote': 'Vaccination coverage was 87.5% in Sierra Leone in 2025.'},
        {'field': 'age', 'document_hash': 'abc123', 'locator': {'chunk_id': 'age'},
         'quote': 'Patient age was 30 years.'},
    ]
    task = {**TASK, 'target_fields': ['metric_value', 'age']}
    result = report([row], task=task)
    assert result['answers']['metric_value']['value'] is None
    assert result['answers']['age']['value'] is None
    assert 'vaccination coverage' in render_task_result(result)


def test_long_report_points_to_actual_collection_csv_location():
    rows = [observation(i, record_id=f'row-{i}') for i in range(31)]
    text = render_task_result(report(rows))
    assert 'collection/final_dataset.csv' in text


def test_task_result_writer_emits_machine_answer_with_english_headline(tmp_path):
    from data_collection_workflow.task_result_report import write_task_result_artifacts

    result = report([observation(5442)])
    paths = write_task_result_artifacts(result, tmp_path)
    assert set(paths) == {'task_result_json'}
    assert not list(tmp_path.glob('*_chinese.md'))
    saved = json.loads((tmp_path / 'task_result.json').read_text(encoding='utf-8'))
    assert saved['headline'] == 'Confirmed: confirmed cases 5,442.'
    assert 'headline_zh' not in saved and 'headline_en' not in saved
    assert not (tmp_path / 'task_result.md').exists()
    text = render_task_result(saved)
    assert '# Collection task results' in text
    assert not any('\u4e00' <= char <= '\u9fff' for char in text + json.dumps(saved, ensure_ascii=False))


def test_task_result_preserves_original_multilingual_evidence():
    quote = 'Mpox in Sierra Leone: 5,442 confirmed cases in 2025. 原文记录保留。'
    result = report([observation(5442, quote=quote)])
    assert result['answers']['cases_confirmed']['sources'][0]['quote'] == quote
    assert quote in render_task_result(result)
    assert result['headline'] == 'Confirmed: confirmed cases 5,442.'
