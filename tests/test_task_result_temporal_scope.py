"""Task answers keep evidence quality separate from the period a count covers."""
import pytest

from data_collection_workflow.task_result_report import build_task_result


TASK = {'disease': 'measles', 'location': 'Canada', 'start_date': '2025-01-01',
        'end_date': '2025-12-31', 'target_fields': ['cases_confirmed']}


def observation(value=100, *, start='2025-02-03', end='2025-10-14',
                quote=None, semantics='cumulative', status='qualified', record_id='one'):
    quote = quote or f'Canada reported {value} confirmed measles cases from {start} through {end}.'
    row = {'record_id': record_id, 'disease': 'measles', 'country': 'Canada',
           'geographic_scope': 'Canada', 'geographic_scope_type': 'country',
           'cases_confirmed': value, 'count_semantics': semantics,
           'metric_period_start': start, 'metric_period_end': end,
           'reporting_period': f'{start} to {end}', 'as_of_date': end,
           'source_url': f'https://example.org/{record_id}', 'evidence_quote': quote}
    row['evidence_qualification'] = {'status': status, 'reasons': [], 'field_evidence': [
        {'field': field, 'quote': quote, 'document_hash': 'verified-document',
         'locator': {'chunk_id': 'local-paragraph'}, 'supported': True}
        for field in ['disease', 'country', 'geographic_scope', 'cases_confirmed',
                      'metric_period_start', 'metric_period_end', 'reporting_period', 'as_of_date']]}
    return row


def result(*rows, candidates=()):
    return build_task_result({'final_dataset': list(rows), 'candidate_records': list(candidates)},
                             {'task': TASK, 'coverage_status': 'incomplete'})


def test_irregular_qualified_cumulative_is_usable_without_becoming_annual_total():
    output = result(observation())
    answer = output['answers']['cases_confirmed']
    assert answer['status'] == 'unconfirmed' and answer['value'] is None
    scoped = answer['selected_result']
    assert scoped['value'] == 100 and scoped['result_kind'] == 'latest_available'
    assert scoped['period_start'] == '2025-02-03' and scoped['period_end'] == '2025-10-14'
    assert scoped['sources'][0]['locator']['chunk_id'] == 'local-paragraph'
    assert '100' in output['headline'] and 'Latest available' in output['headline']
    assert '2025-10-14' in output['headline']


def test_next_year_publication_can_support_previous_calendar_year_total():
    row = observation(start='2025-01-01', end='2025-12-31')
    row['publication_date'] = '2026-03-18'
    answer = result(row)['answers']['cases_confirmed']
    assert answer['status'] == 'confirmed' and answer['value'] == 100
    assert answer['selected_result']['result_kind'] == 'full_period_total'


def test_explicit_authoritative_closure_is_outbreak_total_not_annual_total():
    quote = ('The Ministry of Health declared the measles outbreak in Canada over on 2025-10-14. '
             'The outbreak recorded a total of 100 confirmed measles cases from 2025-02-03 through 2025-10-14.')
    output = result(observation(quote=quote))
    answer = output['answers']['cases_confirmed']
    assert answer['value'] is None
    assert answer['selected_result']['result_kind'] == 'completed_outbreak_total'
    assert 'Completed outbreak' in output['headline']
    assert 'does not establish zero cases outside this outbreak' in answer['selected_result']['boundary_note']
    assert '100' in output['headline'] and '2025-10-14' in output['headline']


@pytest.mark.parametrize('closure', [
    'The Ministry of Health has not declared the measles outbreak in Canada over.',
    'The Ministry of Health may declare the measles outbreak in Canada over.',
    'If the Ministry of Health declared the measles outbreak in Canada over, surveillance would continue.',
    'The Ministry of Health expects to declare the measles outbreak in Canada over.',
    'The Ministry of Health denied that it declared the measles outbreak in Canada over.',
    'The Ministry of Health declared the cholera outbreak in Canada over.',
    'The Ministry of Health declared the measles outbreak in Brazil over.',
    'A blogger declared the measles outbreak in Canada over.',
    'A blogger who declared the measles outbreak in Canada over was interviewed.',
    'No further measles reports were found for Canada.',
])
def test_negated_hypothetical_unrelated_or_unattributed_closure_does_not_finish_outbreak(closure):
    row = observation(quote=closure + ' Canada reported 100 confirmed measles cases from 2025-02-03 through 2025-10-14.')
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'latest_available'


def test_title_and_unverified_row_quote_cannot_supply_missing_closure_evidence():
    row = observation()
    row['source_title'] = 'Ministry of Health declared the measles outbreak in Canada over'
    row['evidence_quote'] = row['source_title'] + '. ' + row['evidence_quote']
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'latest_available'


def test_candidate_closure_cannot_become_supported_result():
    row = observation(status='candidate', quote='The Ministry of Health declared the measles outbreak in Canada over with 100 confirmed cases from 2025-02-03 through 2025-10-14.')
    answer = result(candidates=[row])['answers']['cases_confirmed']
    assert answer['selected_result'] is None and answer['supported_results'] == []
    assert answer['candidate_leads'][0]['value'] == 100


def test_changing_cumulative_snapshots_are_ordered_not_summed_or_conflicted():
    older = observation(60, end='2025-05-03', record_id='older')
    newer = observation(100, end='2025-10-14', record_id='newer')
    answer = result(older, newer)['answers']['cases_confirmed']
    assert answer['selected_result']['value'] == 100
    assert [item['value'] for item in answer['supported_results']] == [100, 60]
    assert not answer['scoped_result_conflict']


def test_conflicting_same_period_snapshots_remain_visible_without_selecting_a_winner():
    answer = result(observation(100), observation(120, record_id='other'))['answers']['cases_confirmed']
    assert answer['selected_result'] is None and answer['scoped_result_conflict']
    assert {item['value'] for item in answer['supported_results']} == {100, 120}


def test_distinct_completed_outbreaks_are_not_summed_into_year_total():
    rows = [observation(value, start=start, end=end, record_id=str(value), quote=(
        f'The Ministry of Health declared the measles outbreak in Canada over with {value} confirmed cases '
        f'from {start} through {end}.'))
        for value, start, end in [(60, '2025-02-03', '2025-05-03'), (100, '2025-08-01', '2025-10-14')]]
    answer = result(*rows)['answers']['cases_confirmed']
    assert answer['value'] is None
    assert [item['value'] for item in answer['supported_results']] == [100, 60]
    assert all(item['result_kind'] == 'completed_outbreak_total' for item in answer['supported_results'])


def test_irregular_incident_period_is_reported_as_period_count_not_cumulative():
    answer = result(observation(semantics='new'))['answers']['cases_confirmed']
    assert answer['selected_result']['result_kind'] == 'reported_period'
    assert answer['value'] is None


def test_unsupported_date_or_count_and_outside_window_cannot_supply_scoped_answer():
    for fields in [{'cases_confirmed'}, {'reporting_period', 'metric_period_start', 'metric_period_end', 'as_of_date'}]:
        row = observation()
        for entry in row['evidence_qualification']['field_evidence']:
            if entry['field'] in fields:
                entry['supported'] = False
        assert result(row)['answers']['cases_confirmed']['supported_results'] == []
    outside = observation(start='2024-02-03', end='2024-10-14')
    assert result(outside)['answers']['cases_confirmed']['supported_results'] == []


def test_approximate_qualified_count_keeps_qualifier_and_is_not_exact_annual_total():
    row = observation(start='2025-01-01', end='2025-12-31', quote='Canada reported approximately 100 confirmed measles cases in 2025.')
    answer = result(row)['answers']['cases_confirmed']
    assert answer['value'] is None
    assert answer['selected_result']['qualifier'] == 'about'
    assert 'about 100' in result(row)['headline']


def test_verified_closure_can_be_linked_from_same_local_chunk_as_count():
    row = observation(quote='Since the first measles case, Canada recorded 100 confirmed cases from 2025-02-03 through 2025-10-14.')
    row['evidence_qualification']['closure_evidence'] = [{
        'document_hash': 'verified-document', 'locator': {'chunk_id': 'local-paragraph'},
        'supported': True,
        'quote': 'The Health Minister declared the measles outbreak in Canada over on 2025-10-14.'}]
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'completed_outbreak_total'
    assert len(selected['sources']) == 2
    assert 'Health Minister' in selected['sources'][1]['quote']


@pytest.mark.parametrize('patch', [
    {'document_hash': 'another-document'}, {'locator': {'chunk_id': 'other-paragraph'}},
    {'supported': False}, {'supported': None},
])
def test_unlinked_or_unverified_closure_cannot_complete_numeric_observation(patch):
    row = observation(quote='Since the first measles case, Canada recorded 100 confirmed cases from 2025-02-03 through 2025-10-14.')
    row['evidence_qualification']['closure_evidence'] = [{
        'document_hash': 'verified-document', 'locator': {'chunk_id': 'local-paragraph'},
        'supported': True,
        'quote': 'The Health Minister declared the measles outbreak in Canada over on 2025-10-14.', **patch}]
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


def test_as_of_only_snapshot_is_supported_with_start_unresolved():
    row = observation()
    row['metric_period_start'] = row['metric_period_end'] = row['reporting_period'] = None
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'latest_available'
    assert selected['period_start'] is None and selected['period_end'] == '2025-10-14'


def test_weekly_reporting_period_count_is_usable_but_not_a_cumulative_total():
    row = observation(semantics='weekly count', start='2025-10-08', end='2025-10-14',
                      quote='Canada reported 100 confirmed measles cases during the week 2025-10-08 through 2025-10-14.')
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['value'] == 100 and selected['result_kind'] == 'reported_period'


def test_real_qualification_connects_local_closure_to_task_report_without_promoting_candidates():
    import hashlib
    from data_collection_workflow.evidence_qualification import build_evidence_index, qualify_records

    count_quote = ('Canada recorded a cumulative total of 100 confirmed measles cases in this outbreak '
                   'from 2025-02-03 through 2025-10-14.')
    text = 'The Health Ministry declared the measles outbreak in Canada over. ' + count_quote
    digest = hashlib.sha256(text.encode()).hexdigest()
    index = build_evidence_index({
        'documents': [{'source_id': 's', 'clean_text': text, 'content_hash': digest, 'text_hash': digest}],
        'evidence_chunks': [{'source_id': 's', 'chunk_id': 'c', 'document_hash': digest,
                             'text': text, 'char_start': 0, 'char_end': len(text)}]})
    row = observation(quote=count_quote)
    row.pop('evidence_qualification')
    row.pop('as_of_date')
    row.update(source_id='s', supporting_chunk_id='c', field_provenance_json={
        'cases_confirmed': {'quote': count_quote}})
    groups = qualify_records([row], contract=TASK, evidence_index=index)
    assert len(groups['qualified_records']) == 1
    qualified = groups['qualified_records'][0]
    assert qualified['evidence_qualification']['closure_evidence']
    selected = result(qualified)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'completed_outbreak_total' and selected['value'] == 100
    assert len(selected['sources']) == 2
    row['cases_confirmed'] = 999
    bad = qualify_records([row], contract=TASK, evidence_index=index)
    assert not bad['qualified_records'] and bad['candidate_records']
    assert result(candidates=bad['candidate_records'])['answers']['cases_confirmed']['selected_result'] is None


def test_verified_declaration_can_attribute_health_minister_in_following_sentence():
    row = observation(quote=(
        'Canada has declared the end of a measles outbreak in the country. '
        'The declaration on Tuesday was announced by Health Minister Dr. Alex Smith. '
        'Since the first measles case was recorded on Feb. 3, Canada has reported a total of '
        '100 confirmed cases through 2025-10-14.'))
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'completed_outbreak_total'


@pytest.mark.parametrize('attribution', [
    'The declaration was denied by the Health Minister.',
    'The declaration might be made by the Health Minister.',
    'The declaration was criticized by the Health Minister.',
    'The declaration was sent to the Health Minister.',
    'The Health Minister discussed hospital budgets.',
])
def test_unrelated_or_denied_following_attribution_cannot_complete_outbreak(attribution):
    row = observation(quote=(
        'Canada has declared the end of a measles outbreak in the country. '
        + attribution + ' Since the first measles case, Canada reported 100 confirmed cases '
        'from 2025-02-03 through 2025-10-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


def test_month_precision_closure_retains_months_instead_of_inventing_days():
    row = observation(quote=(
        'The Health Ministry declared the measles outbreak in Canada over in October 2025. '
        'The outbreak recorded a cumulative total of 100 confirmed measles cases '
        'from February 2025 through October 2025.'))
    row['metric_period_start'] = row['metric_period_end'] = row['as_of_date'] = None
    row['reporting_period'] = 'February 2025 through October 2025'
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'completed_outbreak_total'
    assert selected['period_start'] == '2025-02' and selected['period_end'] == '2025-10'
    assert '2025-02-01' not in result(row)['headline']


def test_monthly_observation_can_answer_its_own_reporting_interval():
    row = observation(semantics='monthly count')
    row['metric_period_start'] = row['metric_period_end'] = row['as_of_date'] = None
    row['reporting_period'] = '2025-10'
    selected = result(row)['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'reported_period'
    assert selected['period_start'] == selected['period_end'] == '2025-10'


def test_previous_year_outbreak_closure_does_not_close_current_year_counts():
    row = observation(quote=(
        'The Ministry of Health declared the measles outbreak in Canada over in 2024. '
        'Canada reported a cumulative total of 100 confirmed measles cases in a new outbreak '
        'from 2025-02-03 through 2025-10-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


@pytest.mark.parametrize('closure', [
    'The Health Ministry declared the first measles outbreak in Canada over on 2025-03-01.',
    'The Health Ministry in Canada declared the measles outbreak in Brazil over on 2025-10-14.',
    'The Health Ministry discussed measles in Canada while a blogger declared the outbreak over.',
    'The Health Ministry declared the measles outbreak in Canada over in an exercise scenario.',
])
def test_closure_actor_event_geography_and_episode_must_belong_to_count(closure):
    row = observation(start='2025-08-01', quote=closure + (
        ' Canada recorded a cumulative total of 100 confirmed measles cases in the second outbreak '
        'from 2025-08-01 through 2025-10-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


def test_latest_month_precision_snapshot_sorts_by_calendar_bounds():
    day = observation(100, end='2025-10-14', record_id='day')
    month = observation(120, record_id='month', quote='Canada recorded a cumulative total of 120 confirmed measles cases from February 2025 through October 2025.')
    month['metric_period_start'] = month['metric_period_end'] = month['as_of_date'] = None
    month['reporting_period'] = 'February 2025 through October 2025'
    answer = result(day, month)['answers']['cases_confirmed']
    assert answer['selected_result']['value'] == 120
    assert answer['selected_result']['period_end'] == '2025-10'


def test_month_may_is_not_a_hypothetical_closure():
    row = observation(end='2025-05-14', quote=(
        'The Health Ministry declared the measles outbreak in Canada over on May 14, 2025. '
        'The outbreak recorded a cumulative total of 100 confirmed measles cases '
        'from 2025-02-03 through 2025-05-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'completed_outbreak_total'


def test_verified_declaration_attribution_can_report_minister_statement():
    row = observation(quote=(
        'Canada has declared the end of a measles outbreak in the country.\n'
        'The declaration on Tuesday aligns with international health standards, which require a minimum of '
        '42 days without a new confirmed case, Health Minister Alex Smith told a ceremony.\n'
        'Since the first measles case was recorded on Feb. 3, Canada has reported a cumulative total of '
        '100 confirmed cases through 2025-10-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'completed_outbreak_total'


def test_authority_discussing_bloggers_claim_does_not_declare_closure():
    row = observation(quote=(
        'The Health Ministry discussed a blogger who declared the measles outbreak in Canada over. '
        'Since the first measles case, Canada reported 100 confirmed cases '
        'from 2025-02-03 through 2025-10-14.'))
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


def test_cumulative_count_semantics_keeps_cumulative_interpretation():
    selected = result(observation(semantics='cumulative_count'))['answers']['cases_confirmed']['selected_result']
    assert selected['result_kind'] == 'latest_available'


@pytest.mark.parametrize('closure', [
    'The Health Ministry declared the cholera outbreak in Brazil over while Canada investigated a measles outbreak.',
    'The measles outbreak in Canada was declared over by a blogger after a Health Ministry briefing.',
    'Canada has declared the end of a measles outbreak in the country. The declaration was announced by a blogger after a Health Minister briefing.',
    'The Health Ministry declared the measles outbreak in Brazil over while Canada investigated its own outbreak.',
    'The Health Ministry declared the cholera outbreak in Brazil over during a visit to Canada to discuss measles.',
])
def test_other_clause_or_actor_cannot_supply_declaration_scope(closure):
    row = observation(quote=closure + ' Canada recorded a cumulative total of 100 confirmed measles cases in this outbreak from 2025-02-03 through 2025-10-14.')
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


@pytest.mark.parametrize('statement', [
    'The Health Ministry confirmed a measles outbreak in Canada over the weekend.',
    'The Health Ministry announced the measles outbreak in Canada had spread to over 100 communities.',
    'The Health Ministry announced a measles outbreak in Canada at the end of September.',
])
def test_outbreak_announcement_is_not_an_outbreak_end(statement):
    row = observation(quote=statement + ' Canada recorded a cumulative total of 100 confirmed measles cases in this outbreak from 2025-02-03 through 2025-10-14.')
    assert result(row)['answers']['cases_confirmed']['selected_result']['result_kind'] == 'latest_available'


def closure_event_observation():
    row = observation(quote=(
        'Canada has declared the end of a measles outbreak in the country. '
        'The declaration on Tuesday aligns with international health standards, Health Minister Alex Smith told a ceremony. '
        'Since the first measles case was recorded on Jan. 9, Canada has reported a total of 100 confirmed cases.'))
    row['metric_period_start'] = row['metric_period_end'] = row['reporting_period'] = row['as_of_date'] = None
    row['outbreak_closure_date'] = '2025-12-16'
    row['outbreak_closure_derivation'] = {
        'basis': 'derived_relative_date', 'rule': 'previous_weekday_from_publication_date',
        'publication_date': '2025-12-17', 'weekday': 'Tuesday',
        'anchors': [{'role': 'publication_date', 'quote': 'Published December 17, 2025', 'char_start': 0, 'char_end': 27},
                    {'role': 'outbreak_closure', 'quote': 'The declaration on Tuesday', 'char_start': 100, 'char_end': 126}]}
    row['evidence_qualification']['temporal_basis'] = 'outbreak_closure_event'
    row['evidence_qualification']['field_evidence'][3]['value'] = 100
    row['evidence_qualification']['field_evidence'].append({
        'field': 'outbreak_closure_date', 'value': '2025-12-16', 'supported': True,
        'document_hash': 'verified-document', 'locator': {'chunk_id': 'local-paragraph'},
        'quote': row['evidence_quote']})
    return row


def test_verified_closure_event_supports_outbreak_total_with_statistical_dates_unresolved():
    output = result(closure_event_observation())
    answer = output['answers']['cases_confirmed']
    assert answer['value'] is None and answer['status'] == 'unconfirmed'
    selected = answer['selected_result']
    assert selected['result_kind'] == 'completed_outbreak_total' and selected['value'] == 100
    assert selected['temporal_basis'] == 'outbreak_closure_event'
    assert selected['period_start'] is None and selected['period_end'] is None and selected['as_of_date'] is None
    assert selected['closure_date'] == '2025-12-16'
    assert selected['closure_date_source']['derivation']['publication_date'] == '2025-12-17'
    assert 'outbreak closed 2025-12-16' in output['headline']
    assert 'statistical period unresolved' in output['headline']
    assert 'may include cases before the query window' in selected['boundary_note']
    assert 'None' not in output['headline'] and '2025-01-09' not in output['headline']
    assert output['qualified_observations'][0]['closure_date_derived'] is True


@pytest.mark.parametrize('patch', [
    {'supported': False}, {'value': '2025-12-15'}, {'document_hash': 'wrong-document'},
    {'locator': {'chunk_id': 'another-paragraph'}},
])
def test_closure_date_requires_its_own_verified_linked_evidence(patch):
    row = closure_event_observation()
    row['evidence_qualification']['field_evidence'][-1].update(patch)
    assert result(row)['answers']['cases_confirmed']['selected_result'] is None


def test_outside_query_closure_event_does_not_make_unknown_period_in_scope():
    row = closure_event_observation()
    row['outbreak_closure_date'] = '2026-01-06'
    row['evidence_qualification']['field_evidence'][-1]['value'] = '2026-01-06'
    assert result(row)['answers']['cases_confirmed']['selected_result'] is None


def test_candidate_with_closure_anchor_still_is_not_supported():
    row = closure_event_observation()
    row['evidence_qualification']['status'] = 'candidate'
    assert result(candidates=[row])['answers']['cases_confirmed']['selected_result'] is None


def test_closure_event_anchor_does_not_override_unrelated_closure():
    row = closure_event_observation()
    for entry in row['evidence_qualification']['field_evidence']:
        entry['quote'] = entry['quote'].replace('end of a measles outbreak', 'end of a cholera outbreak')
    assert result(row)['answers']['cases_confirmed']['selected_result'] is None


def test_recovered_same_source_claim_is_not_repeated_as_unverified_in_readable_report():
    row = closure_event_observation()
    row['record_id'] = 'recovered-one'
    row['recovered_from_record_id'] = 'original-one'
    original = observation(100, status='candidate', record_id='original-one',
                           quote='Canada reported 100 confirmed measles cases in 2025.')
    original['source_url'] = row['source_url']
    output = result(row, candidates=[original])
    answer = output['answers']['cases_confirmed']
    assert answer['candidate_leads'][0]['superseded_by_qualified_record_ids'] == ['recovered-one']
    assert 'Unverified leads' not in output['headline']


def test_closure_sources_identify_the_accepted_declaration_not_the_entire_chunk():
    selected = result(closure_event_observation())['answers']['cases_confirmed']['selected_result']
    source = selected['sources'][-1]
    assert source['closure_statement_quote'] == 'Canada has declared the end of a measles outbreak in the country.'
    assert source['closure_attribution_quote'].startswith('The declaration on Tuesday')
    assert 'Jan. 9' not in source['closure_statement_quote'] + source['closure_attribution_quote']
