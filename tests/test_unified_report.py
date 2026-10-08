"""The reading report must preserve evidence scope and be usable offline."""
import json
import re
import zipfile
from pathlib import Path

import pytest
from bs4 import BeautifulSoup


TASK = {'disease': 'measles', 'location': 'Canada', 'start_date': '2025-01-01',
        'end_date': '2025-12-31', 'target_fields': ['cases_confirmed', 'deaths']}


def observation(value=100, *, record_id='one', status='qualified', start='2025-01-01',
                end='2025-12-31', quote=None):
    quote = quote or f'Canada reported {value} confirmed measles cases from {start} through {end}.'
    row = {'record_id': record_id, 'disease': 'measles', 'country': 'Canada',
           'geographic_scope': 'Canada', 'geographic_scope_type': 'country',
           'cases_confirmed': value, 'count_semantics': 'cumulative',
           'metric_period_start': start, 'metric_period_end': end,
           'reporting_period': f'{start} to {end}', 'as_of_date': end,
           'source_url': f'https://example.org/{record_id}', 'evidence_quote': quote}
    row['evidence_qualification'] = {'status': status, 'reasons': [], 'field_evidence': [
        {'field': field, 'quote': quote, 'document_hash': 'verified-document',
         'locator': {'chunk_id': 'local-paragraph'}, 'supported': True}
        for field in ['disease', 'country', 'geographic_scope', 'cases_confirmed',
                      'metric_period_start', 'metric_period_end', 'reporting_period', 'as_of_date']]}
    return row


def package(*rows, candidates=(), sources=(), task=None):
    return {'final_dataset': list(rows), 'aggregate_dataset': list(rows),
            'candidate_records': list(candidates), 'context_records': [],
            'final_case_dataset': [], 'source_registry': list(sources),
            'result_manifest': {'task': task or TASK, 'coverage_status': 'partial'}}


def generate(tmp_path, data, **kwargs):
    from data_collection_workflow.reporting.unified_report import write_unified_report
    paths = write_unified_report(tmp_path, data, **kwargs)
    text = Path(paths['final_report_english']).read_text(encoding='utf-8')
    soup = BeautifulSoup(text, 'html.parser')
    payload = json.loads(soup.select_one('#report-data').string)
    return paths, text, soup, payload


def test_report_is_single_english_entry_and_recomputes_current_task_answers(tmp_path):
    data = package(observation(100))
    data['result_manifest']['counts'] = {'qualified_observations': 999999, 'candidate_records': 888888}
    data['result_manifest']['dataset_counts'] = {'final_dataset': 999999}
    data['task_result'] = {'headline': 'Stale total 999999', 'answers': {}}
    (tmp_path / 'task_result.json').write_text(json.dumps(data['task_result']))
    paths, text, soup, payload = generate(tmp_path, data, summary={'final_record_count': 999999})
    assert paths['final_report'] == paths['final_report_english']
    assert list(tmp_path.glob('*.html')) == [tmp_path / 'final_report.html']
    assert soup.html['lang'] == 'en'
    assert 'measles in Canada' in soup.h1.get_text()
    assert '999999' not in soup.select_one('#summary').get_text()
    assert payload['snapshot']['results']['cases_confirmed']['value'] == 100
    assert payload['snapshot']['record_counts']['final_dataset'] == 1
    assert payload['snapshot']['record_counts']['individual_cases'] == 0
    assert payload['snapshot']['result_manifest']['counts']['qualified_observations'] == 1
    assert payload['snapshot']['result_manifest']['counts']['candidate_records'] == 0
    assert payload['snapshot']['result_manifest']['dataset_counts']['final_dataset'] == 1
    assert 'not patients' in soup.select_one('#downloads').get_text()
    assert not re.search(r'mpox|Sierra Leone|5,442', text)


def test_supported_zero_is_displayed_but_missing_deaths_are_unconfirmed(tmp_path):
    _, _, soup, payload = generate(tmp_path, package(observation(0)))
    cards = {node['data-field']: node for node in soup.select('[data-field]')}
    assert cards['cases_confirmed'].select_one('.result-value').get_text() == '0'
    assert 'Not confirmed' in cards['deaths'].get_text()
    assert payload['snapshot']['results']['deaths']['value'] is None


def test_scoped_result_is_headline_without_promoting_it_to_calendar_total(tmp_path):
    row = observation(41, start='2025-02-03', end='2025-10-14',
                      quote='The Ministry of Health declared the measles outbreak in Canada over on 2025-10-14. '
                            'The outbreak recorded a total of 41 confirmed measles cases from 2025-02-03 through 2025-10-14.')
    _, _, soup, payload = generate(tmp_path, package(row))
    result = payload['snapshot']['results']['cases_confirmed']
    assert result['value'] == 41 and result['result_kind'] == 'completed_outbreak_total'
    assert 'Completed outbreak total' in soup.select_one('#summary').get_text()
    assert '2025-02-03' in soup.select_one('#summary').get_text()
    assert 'full requested period' in soup.select_one('#gaps').get_text()
    citation = soup.select_one('#summary .citation')
    source = next(s for s in payload['catalog']['sources'] if row['source_url'] in [s['url'], *s.get('alias_urls', [])])
    assert citation['href'] == '#source-' + source['report_source_id']


def test_conflicting_and_candidate_totals_are_never_headline_supported_counts(tmp_path):
    _, _, soup, payload = generate(tmp_path, package(observation(100), observation(120, record_id='two'),
                                               candidates=[observation(700, record_id='lead', status='candidate')]))
    answer = payload['snapshot']['results']['cases_confirmed']
    assert answer['value'] is None and answer['status'] == 'conflict'
    assert 'Conflicting evidence' in soup.select_one('[data-field="cases_confirmed"]').get_text()
    assert not soup.select_one('[data-field="cases_confirmed"] .result-value')
    assert '100' in soup.select_one('#gaps').get_text() and '120' in soup.select_one('#gaps').get_text()


def test_empty_and_candidate_only_reports_do_not_invent_zero_cases(tmp_path):
    for subdir, data in [('empty', package()), ('candidate', package(candidates=[observation(87, status='candidate')]))]:
        _, _, soup, payload = generate(tmp_path / subdir, data)
        assert payload['snapshot']['results']['cases_confirmed']['value'] is None
        assert not soup.select_one('#summary .result-value')
        assert 'Not confirmed' in soup.select_one('#summary').get_text()


def test_html_and_script_injections_are_escaped(tmp_path):
    malicious = '</script><script>alert("bad")</script><img src=x onerror=alert(1)>'
    task = {**TASK, 'disease': malicious}
    sources = [{'source_id': 'evil', 'url': 'javascript:alert(1)', 'title': malicious}]
    _, text, soup, payload = generate(tmp_path, package(sources=sources, task=task))
    assert soup.h1.get_text().startswith(malicious)
    assert not soup.find('img')
    assert not any('alert("bad")' == s.string for s in soup.find_all('script'))
    assert not soup.select('a[href^="javascript:"]')
    assert payload['catalog']['sources'][0]['title'] == malicious
    assert '\\u003c/script' in text


def test_portable_zip_only_contains_explicit_current_data_and_evidence(tmp_path):
    row = observation(12)
    data = package(row)
    data['evidence_chunks'] = [{'chunk_id': 'local-paragraph', 'text': row['evidence_quote']}]
    (tmp_path / 'collection').mkdir()
    original = tmp_path / 'collection' / 'final_dataset.csv'
    original.write_text('do not overwrite this original\n', encoding='utf-8')
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'old_report.md').write_text('obsolete')
    (tmp_path / 'workflow.log').write_text('internal log')
    paths, _, soup, payload = generate(tmp_path, data)
    assert original.read_text(encoding='utf-8') == 'do not overwrite this original\n'
    with zipfile.ZipFile(paths['report_bundle']) as archive:
        names = archive.namelist()
        assert 'final_report.html' in names
        assert 'data/final_dataset.csv' in names and 'data/evidence_chunks.json' in names
        assert 'data/candidate_records.json' in names and 'data/record_inclusion_decisions.json' in names
        assert not any(name.endswith('.md') or 'workflow.log' in name or '..' in name for name in names)
        assert not any('final_package' in name or 'build/' in name or 'collection/' in name for name in names)
        for link in soup.select('a[href]'):
            href = link['href']
            if not href.startswith(('#', 'http:', 'https:')):
                assert href in names, href
        current = json.loads(archive.read('data/final_dataset.json'))
        assert current[0]['cases_confirmed'] == 12


def test_more_than_twenty_sources_remain_available_in_full_catalogue(tmp_path):
    sources = [{'source_id': f'raw-{n}', 'url': f'https://example.org/source-{n}',
                'title': f'Surveillance page {n}'} for n in range(37)]
    paths, _, soup, payload = generate(tmp_path, package(sources=sources))
    assert len(payload['catalog']['sources']) == 37
    assert len(json.loads(Path(paths['source_catalog_json']).read_text())['sources']) == 37
    assert soup.select_one('#page-size option[value="all"]')
    assert soup.select_one('#next') and soup.select_one('#query')
    assert soup.select_one('#settings-groups')


def test_default_inputs_load_from_session_without_mutating_original_package(tmp_path):
    from data_collection_workflow.reporting.unified_report import write_unified_report
    data = package(observation(2))
    collection = tmp_path / 'collection'
    collection.mkdir()
    package_path = collection / 'final_package.json'
    original = json.dumps(data)
    package_path.write_text(original)
    (tmp_path / 'workflow_run_summary.json').write_text(json.dumps({'stop_reason': 'completed'}))
    paths = write_unified_report(tmp_path)
    assert package_path.read_text() == original
    assert json.loads(Path(paths['report_snapshot_json']).read_text())['results']['cases_confirmed']['value'] == 2


def test_saved_config_is_used_and_historical_settings_survive_report_rebuild(tmp_path):
    from data_collection_workflow.reporting.unified_report import write_unified_report
    tmp_path.joinpath('run_config.json').write_text(json.dumps({'pipeline_mode': 'evidence', 'llm': {'model': 'saved-model'}}))
    paths = write_unified_report(tmp_path, package())
    inventory = json.loads(Path(paths['run_settings_json']).read_text())
    model = next(row for group in inventory['groups'] for row in group['rows'] if row['key'] == 'llm.model')
    assert model['configured_display'] == 'saved-model'
    tmp_path.joinpath('run_config.json').unlink()
    write_unified_report(tmp_path, package())
    assert json.loads(Path(paths['run_settings_json']).read_text()) == inventory


def test_actual_run_timestamps_duration_and_recorded_budget_preserve_zero(tmp_path):
    state = {'run_status': {'started_at_utc': '2025-02-01T00:00:00+00:00',
                            'completed_at_utc': '2025-02-01T00:00:03+00:00', 'duration_ms': 2500},
             'run_budget_ledger': {'limits': {'search': 10}, 'used': {'search': 0}},
             'recovery_stop_reason': 'budget_exhausted'}
    _, _, _, payload = generate(tmp_path, package(), state=state)
    run = payload['snapshot']['run_information']
    assert run['runtime']['duration_seconds'] == 2.5
    assert run['budget']['rows'][0]['used'] == 0
    assert run['models']['display_en'] == 'Not recorded'


def test_standalone_rebuild_keeps_effective_settings_absent_from_saved_configuration(tmp_path):
    from data_collection_workflow.reporting.unified_report import write_unified_report
    config = {'llm': {'model': 'configured-model'}}
    tmp_path.joinpath('run_config.json').write_text(json.dumps(config))
    paths = write_unified_report(tmp_path, package(), config=config,
                                 state={'runtime_profile': {'env': {'LLM_MODEL': 'actual-model'}}})
    original = json.loads(Path(paths['run_settings_json']).read_text())
    model = next(row for group in original['groups'] for row in group['rows'] if row['key'] == 'llm.model')
    assert model['effective_display'] == 'actual-model'
    write_unified_report(tmp_path, package(), summary={'stop_reason': 'completed'})
    assert json.loads(Path(paths['run_settings_json']).read_text()) == original


def test_unresolved_acquisition_is_described_even_with_complete_numeric_answer(tmp_path):
    task = {**TASK, 'target_fields': ['cases_confirmed']}
    sources = [{'source_id': 'waiting', 'url': 'https://example.org/waiting',
                'processing_status': 'budget_deferred', 'processing_reason': 'budget_exhausted'}]
    _, _, soup, _ = generate(tmp_path, package(observation(5), sources=sources, task=task))
    assert 'unresolved acquisition' in soup.select_one('#gaps').get_text()


def test_complete_task_without_pending_sources_has_no_unresolved_section(tmp_path):
    task = {**TASK, 'target_fields': ['cases_confirmed']}
    _, _, soup, _ = generate(tmp_path, package(observation(5), task=task))
    assert not soup.select_one('#gaps')
    assert not soup.select('a[href="#gaps"]')


def test_explicit_source_url_wins_over_shared_record_id_when_linking_conclusion(tmp_path):
    row = observation(35)
    row['source_id'] = 'b'
    row['source_url'] = 'https://example.org/b'
    for field in row['evidence_qualification']['field_evidence']:
        field['locator'] = {'chunk_id': 'chunk-a' if field['field'] == 'country' else 'chunk-b'}
    data = package(row, sources=[{'source_id': key, 'url': f'https://example.org/{key}'} for key in ['a', 'b']])
    data['evidence_chunks'] = [{'chunk_id': 'chunk-a', 'source_id': 'a'}, {'chunk_id': 'chunk-b', 'source_id': 'b'}]
    _, _, soup, payload = generate(tmp_path, data)
    target = next(source for source in payload['catalog']['sources'] if 'b' in source['source_ids'])
    assert soup.select_one('#summary .citation')['href'] == '#source-' + target['report_source_id']


def test_standalone_rebuild_loads_saved_evidence_before_attributing_source_records(tmp_path):
    row = observation(21)
    row['source_id'] = 'source'
    data = package(row, sources=[{'source_id': 'source', 'url': row['source_url']}])
    collection = tmp_path / 'collection'
    collection.mkdir()
    collection.joinpath('evidence_chunks.json').write_text(json.dumps([
        {'chunk_id': 'local-paragraph', 'source_id': 'source', 'text': row['evidence_quote']}]))
    _, _, _, payload = generate(tmp_path, data)
    source = payload['catalog']['sources'][0]
    assert source['record_counts']['qualified'] == 1
    assert source['evidence_details'][0]['record_id'] == 'one'
    assert source['evidence_files']


def test_bundle_redacts_credentials_consistently_without_mutating_original_data(tmp_path):
    secret_url = 'https://user:secret-password@example.org/report?API_KEY=secret-query-key'
    row = observation(12)
    row['source_url'] = secret_url
    row['private_token'] = 'secret-row-token'
    data = package(row, sources=[{'source_id': 's', 'url': secret_url}])
    data['result_manifest']['task'] = {**TASK, 'ApiKey': 'secret-task-key'}
    before = json.dumps(data)
    paths, _, soup, payload = generate(tmp_path, data, config={'user_request': 'Collect from ' + secret_url})
    assert json.dumps(data) == before
    assert soup.select_one('#summary .citation')
    with zipfile.ZipFile(paths['report_bundle']) as archive:
        for name in archive.namelist():
            content = archive.read(name).decode('utf-8-sig')
            for secret in ['secret-password', 'secret-query-key', 'secret-row-token', 'secret-task-key']:
                assert secret not in content, (name, secret)


def test_saved_manifest_budget_and_runtime_model_are_used_when_summary_is_absent(tmp_path):
    data = package()
    data['result_manifest']['budget'] = {'limits': {'search': 10}, 'used': {'search': 4}, 'total_cost_usd': 1.25}
    _, _, _, payload = generate(tmp_path, data, state={'runtime_profile': {'env': {'LLM_MODEL': 'historical-model'}}})
    run = payload['snapshot']['run_information']
    assert run['budget']['rows'][0]['limit'] == 10
    assert run['budget']['rows'][0]['used'] == 4
    assert run['budget']['cost_display_en'] == 'USD 1.2500'
    assert run['models']['display_en'] == 'historical-model'


def test_standard_mode_preserves_collected_and_pending_records_without_qualification_claims(tmp_path):
    accepted = observation(11)
    accepted.pop('evidence_qualification')
    pending = {**observation(13, record_id='pending'), 'requires_human_review': True}
    pending.pop('evidence_qualification')
    data = {'final_dataset': [accepted], 'final_case_dataset': [], 'source_registry': [],
            'pending_review_records': [pending], 'quarantined_records': [pending],
            'record_inclusion_decisions': [{'record_id': 'pending', 'record_final_inclusion_status': 'pending_human_review',
                                            'reasons': ['source identity requires review']}]}
    paths, _, soup, payload = generate(tmp_path, data, config={'pipeline_mode': 'standard', 'structured_task': TASK})
    counts = payload['snapshot']['record_counts']
    assert counts['final_dataset'] == 1
    assert counts['qualified_observations'] == 0
    assert counts['candidate_records'] == 1
    assert not soup.select_one('#summary .result-value')
    assert 'Collected observations' in soup.select_one('#downloads').get_text()
    assert json.loads(Path(paths['final_report']).parent.joinpath('data/candidate_records.json').read_text())[0]['record_id'] == 'pending'
    assert sum(source['record_counts'].get('qualified', 0) for source in payload['catalog']['sources']) == 0
    decisions = json.loads(Path(paths['final_report']).parent.joinpath('data/record_inclusion_decisions.json').read_text())
    saved = next(row for row in decisions if row['record_id'] == 'pending')
    assert saved['record_final_inclusion_status'] == 'pending_human_review'
    assert saved['reasons'] == ['source identity requires review']
    with zipfile.ZipFile(paths['report_bundle']) as archive:
        assert 'data/pending_review_records.json' in archive.namelist()
        assert 'data/quarantined_records.json' in archive.namelist()


def test_explicit_empty_candidate_dataset_does_not_revive_stale_aliases(tmp_path):
    data = package()
    data['pending_review_records'] = [observation(99, record_id='stale')]
    _, _, _, payload = generate(tmp_path, data)
    assert payload['snapshot']['record_counts']['candidate_records'] == 0


def test_final_manifest_stop_reason_wins_over_earlier_recovery_state(tmp_path):
    data = package()
    data['result_manifest'].update(collection_status='partial', recovery_stop_reason='provider_account_limit')
    state = {'recovery_stop_reason': 'coverage_satisfied',
             'run_budget_ledger': {'providers': {'anthropic': {'status': 'halted'}}}}
    _, _, _, payload = generate(tmp_path, data, state=state)
    assert payload['snapshot']['run_information']['runtime']['stop_reason_en'] == 'provider account limit'
    assert payload['snapshot']['result_manifest']['collection_status'] == 'partial'


@pytest.mark.parametrize('folder', ['collection', 'diagnostics'])
def test_historical_structured_task_file_supplies_recorded_scope(tmp_path, folder):
    task_path = tmp_path / folder / 'structured_task.json'
    task_path.parent.mkdir()
    task_path.write_text(json.dumps(TASK))
    data = package(observation(7))
    data.pop('result_manifest')
    _, _, soup, payload = generate(tmp_path, data)
    assert payload['snapshot']['task'] == TASK
    assert 'measles in Canada' in soup.h1.get_text()
    assert payload['snapshot']['results']['cases_confirmed']['value'] == 7


def test_browser_pagination_citation_reset_settings_search_and_html_escape(tmp_path):
    playwright = pytest.importorskip('playwright.sync_api')
    sources = [{'source_id': f'raw-{n}', 'url': f'https://example.org/source-{n}',
                'title': f'Surveillance page {n}'} for n in range(37)]
    sources[0]['title'] = '<img src=x onerror=alert(1)> Surveillance page'
    sources[0]['publisher'] = '<script>bad()</script>'
    row = observation(42)
    row['source_url'] = sources[-1]['url']
    collected = observation(3, record_id='collected')
    collected.pop('evidence_qualification')
    collected.update(source_id='raw-0', source_url=sources[0]['url'], record_final_inclusion_status='accepted')
    paths, _, _, payload = generate(tmp_path, package(row, collected, sources=sources))
    with playwright.sync_playwright() as driver:
        executable = next((p for p in [Path(driver.chromium.executable_path),
                         Path('C:/Program Files/Google/Chrome/Application/chrome.exe')] if p.is_file()), None)
        if executable is None:
            pytest.skip('No Chromium installation is available')
        browser = driver.chromium.launch(executable_path=str(executable), headless=True)
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(Path(paths['final_report']).resolve().as_uri())
        assert page.locator('#source-rows > tr:not(.source-detail)').count() == 25
        page.locator('#next').click()
        assert page.locator('#source-rows > tr:not(.source-detail)').count() == 12
        page.locator('#page-size').select_option('all')
        assert page.locator('#source-rows > tr:not(.source-detail)').count() == 37
        assert page.locator('img').count() == 0
        assert page.locator('#contribution-filter option[value="collected"]').count() == 1
        page.locator('#contribution-filter').select_option('collected')
        assert page.locator('#source-rows > tr:not(.source-detail)').count() == 1
        page.locator('#source-rows [data-toggle]').click()
        assert 'not evidence-qualified' in page.locator('.source-detail').inner_text()
        page.locator('#source-rows [data-toggle]').click()
        page.locator('#reset').click()
        page.locator('#query').fill('no-match')
        assert page.locator('#source-rows .empty').count() == 1
        page.locator('#contribution-filter').select_option('candidate')
        page.locator('#summary .citation').first.click()
        assert page.locator('#query').input_value() == ''
        assert page.locator('#contribution-filter').input_value() == 'all'
        assert page.locator('.source-detail').count() == 1
        assert '42' in page.locator('.source-detail').inner_text()
        page.locator('#settings-query').fill('llm.model')
        assert page.locator('.settings-group[open]').count() >= 1
        assert page.locator('tr[data-setting-key="llm.model"]').count() == 1
        assert not errors
        with zipfile.ZipFile(paths['report_bundle']) as archive:
            for href in page.locator('a[href]').evaluate_all('(links)=>links.map(a=>a.getAttribute("href"))'):
                if href.startswith(('data/', 'evidence/')):
                    assert href in archive.namelist(), href
        browser.close()
