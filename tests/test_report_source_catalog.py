"""The session catalogue preserves source coverage and evidence boundaries."""
from copy import deepcopy

from data_collection_workflow.reporting.source_catalog import build_source_catalog


def catalog(package, state=None):
    return build_source_catalog(package, state)


def by_source(result, source_id):
    return next(row for row in result['sources'] if source_id in row['source_ids'])


def test_complete_catalog_includes_inventory_and_excluded_only_sources_and_aliases():
    package = {
        'source_registry': [
            {'source_id': 'z-main', 'url': 'https://example.org/report'},
            {'source_id': 'a-alias', 'url': 'https://EXAMPLE.org/report/#table'},
            {'source_id': 'separate', 'url': 'https://example.org/report?version=2', 'title': 'Same title'},
        ],
        'source_inventory': [{'source_id': 'inventory', 'url': 'https://example.org/inventory'}],
        'excluded_sources': [{'source_id': 'excluded', 'url': 'https://example.org/excluded',
                             'blocked_from_fetch_reason': 'geography_mismatch'}],
        'final_dataset': [{'record_id': 'q', 'source_id': 'z-main',
                           'evidence_qualification': {'status': 'qualified'}}],
    }
    before = deepcopy(package)
    result = catalog(package)
    assert len(result['sources']) == 4
    assert result['sources'][0]['source_ids'] == ['a-alias', 'z-main']
    assert result['sources'][0]['report_source_id'] == 'S001'
    assert by_source(result, 'excluded')['processing']['code'] == 'screening_excluded'
    assert by_source(result, 'inventory')['processing']['code'] == 'not_attempted'
    assert by_source(result, 'separate')['record_counts']['qualified'] == 0
    assert result['metadata']['record_counts'] == {'qualified': 1, 'candidate': 0, 'context': 0}
    assert package == before
    package['source_registry'].reverse()
    assert catalog(package) == result


def test_explicit_alias_chains_merge_without_dropping_urls_or_saved_status():
    package = {
        'source_registry': [{'source_id': 'a', 'url': 'https://example.org/original',
                             'official_report_alias_source_ids': ['b'],
                             'official_report_alias_urls': ['https://example.org/print']}],
        'source_inventory': [{'source_id': 'b', 'url': 'https://example.org/print',
                              'source_id_aliases': ['c']}],
        'excluded_sources': [{'source_id': 'c', 'url': 'https://example.org/old'}],
        'source_processing_status': [{'source_ids': ['c', 'b', 'a'], 'processing_status': 'acquisition_failed',
                                      'processing_reason': 'http_error', 'acquisition_status': 'failed_or_unreadable'}],
    }
    result = catalog(package)
    assert len(result['sources']) == 1
    source = result['sources'][0]
    assert source['source_ids'] == ['a', 'b', 'c']
    assert {source['url'], *source['alias_urls']} == {
        'https://example.org/original', 'https://example.org/print', 'https://example.org/old'}
    assert source['processing']['code'] == 'acquisition_failed'


def test_failed_and_unread_sources_never_inherit_official_upstream_identity():
    package = {'source_registry': [
        {'source_id': 'failed', 'url': 'https://unknown.example.org/report',
         'publisher': 'unknown', 'upstream_source_mentions': ['National Health Agency'],
         'source_type_final': 'unknown', 'published_date': None},
        {'source_id': 'unread', 'url': 'https://example.org/unread', 'snippet': 'Search metadata only'},
    ]}
    state = {'documents': [{'source_id': 'failed', 'fetch_status': 'fetch_failed',
                            'fetch_error': 'HTTP 503', 'clean_text': 'Service unavailable'}]}
    result = catalog(package, state)
    failed = by_source(result, 'failed')
    assert failed['processing']['code'] == 'acquisition_failed'
    assert failed['publisher']['display'] == 'unknown.example.org'
    assert failed['publisher']['verified'] is False
    assert failed['upstream_mentions'] == ['National Health Agency']
    assert failed['upstream_mentions_verified'] is False
    assert failed['source_type_label'] == 'Source type unverified'
    assert failed['publication_date']['value'] is None
    assert failed['saved_excerpt']['text'] == ''
    unread = by_source(result, 'unread')
    assert unread['search_metadata_excerpt']['text'] == 'Search metadata only'
    assert unread['saved_excerpt']['text'] == ''


def test_current_roles_and_bound_fields_control_contributions_without_candidate_promotion():
    long_quote = 'Full evidence quotation. ' * 100
    package = {
        'source_registry': [{'source_id': 'primary', 'url': 'https://example.org/primary'},
                            {'source_id': 'evidence', 'url': 'https://example.org/evidence',
                             'published_date': '2024-03-07', 'actual_publisher': 'Local Research Institute',
                             'source_identity_unverified': False}],
        'final_dataset': [{'record_id': 'qualified', 'source_id': 'primary', 'evidence_quote': long_quote,
                           'evidence_qualification': {'status': 'qualified', 'product_kind': 'aggregate',
                            'field_evidence': [
                                {'field': 'cases_confirmed', 'value': 12, 'supported': True, 'quote': long_quote,
                                 'locator': {'chunk_id': 'count'}, 'document_hash': 'hash'},
                                {'field': 'reporting_period', 'value': '2024-01', 'supported': True,
                                 'locator': {'chunk_id': 'count'}},
                                {'field': 'deaths', 'value': 99, 'supported': False,
                                 'locator': {'chunk_id': 'wrong'}}]}}],
        'candidate_records': [{'record_id': 'candidate', 'source_id': 'evidence', 'evidence_quote': long_quote,
                               'evidence_qualification': {'status': 'qualified',
                                'reasons': ['missing_geography', 'cases_confirmed:unbound_field_value']}}],
        'context_records': [{'record_id': 'context', 'source_id': 'evidence',
                              'evidence_qualification': {'status': 'qualified', 'product_kind': 'context'}}],
    }
    state = {'evidence_chunks': [{'chunk_id': 'count', 'source_id': 'evidence'},
                                {'chunk_id': 'wrong', 'source_id': 'primary'}],
             'documents': [{'source_id': 'evidence', 'content_readable': True,
                            'clean_text': long_quote, 'artifact_path': 'acquisition/evidence.json'}]}
    result = catalog(package, state)
    source = by_source(result, 'evidence')
    assert source['record_counts'] == {'qualified': 1, 'candidate': 1, 'context': 1}
    assert source['records_by_role'] == {'qualified': ['qualified'], 'candidate': ['candidate'], 'context': ['context']}
    assert by_source(result, 'primary')['record_counts']['qualified'] == 0
    assert source['publisher']['verified'] is True
    assert source['publication_date']['value'] == '2024-03-07'
    assert source['statistics_period']['display'] == '2024-01'
    fields = source['evidence_details'][0]['supported_fields']
    assert {field['field'] for field in fields} == {'cases_confirmed', 'reporting_period'}
    assert fields[0]['quote'] == long_quote
    assert source['candidate_details'][0]['status'] == 'candidate'
    assert source['candidate_details'][0]['evidence_quote'] == long_quote
    assert all(' ' in reason for reason in source['candidate_details'][0]['reason_labels'])
    assert source['saved_documents'][0]['artifact'] == 'acquisition/evidence.json'


def test_statistical_dates_do_not_become_publication_dates_or_validate_candidate_periods():
    package = {'source_registry': [{'source_id': 's', 'url': 'https://example.org/no-date'}],
               'candidate_records': [{'record_id': 'c', 'source_id': 's', 'as_of_date': '2025-05-01',
                                      'date_reported': '2025-05-02', 'reporting_period': '2025',
                                      'evidence_qualification': {'status': 'candidate', 'field_evidence': [
                                          {'field': 'reporting_period', 'value': '2025', 'supported': True}]}}]}
    source = catalog(package)['sources'][0]
    assert source['publication_date']['value'] is None
    assert source['statistics_period']['supported_record_periods'] == []
    assert source['record_counts'] == {'qualified': 0, 'candidate': 1, 'context': 0}


def test_saved_progress_is_joined_by_source_id_and_unsafe_document_paths_are_not_exposed():
    package = {'source_inventory': [{'source_id': 'a', 'url': 'https://example.org/a'},
                                     {'source_id': 'z', 'url': 'https://example.org/z'}],
               'source_processing_status': [
                   {'source_ids': ['z'], 'processing_status': 'budget_deferred', 'processing_reason': 'source_targets',
                    'acquisition_status': 'not_fetched'},
                   {'source_ids': ['a'], 'processing_status': 'readable', 'processing_reason': 'no_extracted_evidence',
                    'acquisition_status': 'readable'}],
               'documents': [{'source_id': 'a', 'clean_text': 'Original source text', 'content_readable': True,
                              'artifact_path': '../../private.json'}]}
    result = catalog(package)
    assert by_source(result, 'z')['processing']['code'] == 'budget_deferred'
    source = by_source(result, 'a')
    assert source['processing']['readable'] is True
    assert source['saved_excerpt']['text'] == 'Original source text'
    assert not source['saved_documents'][0].get('artifact')


def test_empty_catalog_and_record_only_sources_are_retained_without_file_reads():
    assert catalog({})['sources'] == []
    result = catalog({'candidate_records': [{'record_id': 'orphan', 'source_id': 'missing',
                                              'source_url': 'https://example.org/orphan'}]})
    assert result['sources'][0]['source_ids'] == ['missing']
    assert result['sources'][0]['url'] == 'https://example.org/orphan'
    assert result['sources'][0]['records_by_role']['candidate'] == ['orphan']


def test_record_url_without_source_id_remains_available_for_report_citations():
    package = {'final_dataset': [{'record_id': 'one', 'source_url': 'https://example.org/one#confirmed',
                                 'source_title': 'A record-only reference',
                                 'evidence_qualification': {'status': 'qualified', 'field_evidence': [
                                     {'field': 'cases_confirmed', 'value': 12, 'supported': True,
                                      'locator': {'chunk_id': 'unresolved'}}]}}]}
    result = catalog(package)
    assert len(result['sources']) == 1
    assert result['sources'][0]['url'] == 'https://example.org/one'
    assert result['sources'][0]['alias_urls'] == ['https://example.org/one#confirmed']
    assert result['sources'][0]['title'] == 'A record-only reference'
    assert result['sources'][0]['processing']['code'] == 'unknown'
    # Keeping a citation target must not manufacture field-level source binding.
    assert result['metadata']['unresolved_record_ids'] == ['one']
    assert result['sources'][0]['evidence_details'] == []


def test_proven_alias_urls_merge_but_url_only_sources_keep_their_metadata():
    package = {'source_registry': [
        {'source_id': 'a', 'url': 'https://example.org/a',
         'official_report_alias_urls': ['https://example.org/b']},
        {'source_id': 'b', 'url': 'https://example.org/b'},
        {'url': 'https://example.org/url-only', 'title': 'A URL-only source',
         'published_date': '2024-04-01'},
    ]}
    result = catalog(package)
    assert len(result['sources']) == 2
    assert by_source(result, 'a')['source_ids'] == ['a', 'b']
    source = next(row for row in result['sources'] if row['url'].endswith('/url-only'))
    assert source['title'] == 'A URL-only source'
    assert source['publication_date']['value'] == '2024-04-01'


def test_qualified_field_quotes_and_periods_stay_with_the_bound_source():
    package = {
        'source_registry': [{'source_id': 'a', 'url': 'https://example.org/a'},
                            {'source_id': 'b', 'url': 'https://example.org/b'}],
        'evidence_chunks': [{'chunk_id': 'ca', 'source_id': 'a'}, {'chunk_id': 'cb', 'source_id': 'b'}],
        'final_dataset': [{'record_id': 'q', 'source_id': 'a', 'evidence_qualification': {
            'status': 'qualified', 'field_evidence': [
                {'field': 'cases_confirmed', 'value': 12, 'supported': True,
                 'quote': 'Twelve confirmed cases.', 'locator': {'chunk_id': 'ca'}},
                {'field': 'deaths', 'value': 1, 'supported': True,
                 'quote': 'One death.', 'locator': {'chunk_id': 'cb'}},
                {'field': 'reporting_period', 'value': '2024', 'supported': True,
                 'quote': 'An unresolved attribution.', 'locator': {'chunk_id': 'missing'}},
            ]}}],
    }
    result = catalog(package)
    first = by_source(result, 'a')
    second = by_source(result, 'b')
    assert [field['field'] for field in first['evidence_details'][0]['supported_fields']] == ['cases_confirmed']
    assert [field['field'] for field in second['evidence_details'][0]['supported_fields']] == ['deaths']
    assert first['statistics_period']['supported_record_periods'] == []
    assert result['metadata']['record_counts']['qualified'] == 1
    assert result['metadata']['contribution_counts']['qualified'] == 2


def test_standard_accepted_records_are_collected_without_becoming_qualified_evidence():
    accepted = {'record_id': 'accepted', 'source_id': 's', 'cases_confirmed': 17,
                'record_final_inclusion_status': 'accepted', 'requires_human_review': False,
                'evidence_quote': 'Seventeen cases were recorded.', 'reporting_period': '2024',
                'evidence_qualification': None}
    package = {'source_registry': [{'source_id': 's', 'url': 'https://example.org/report'}],
               'final_dataset': [accepted],
               'record_inclusion_decisions': [{'record_id': 'accepted', 'record_final_inclusion_status': 'accepted',
                                                'reasons': ['Standard inclusion checks passed.']}]}
    before = deepcopy(package)
    result = catalog(package)
    source = result['sources'][0]
    assert source['record_counts'] == {'qualified': 0, 'candidate': 0, 'context': 0, 'collected': 1}
    assert source['records_by_role']['qualified'] == []
    assert source['records_by_role']['collected'] == ['accepted']
    assert source['evidence_details'] == []
    detail = source['collected_details'][0]
    assert detail['status'] == 'collected'
    assert detail['record_final_inclusion_status'] == 'accepted'
    assert detail['reasons'] == ['Standard inclusion checks passed.']
    assert detail['evidence_quote'] == accepted['evidence_quote']
    assert detail['supported_fields'] == []
    assert any(field['field'] == 'cases_confirmed' and field['value'] == 17 for field in detail['observed_fields'])
    assert source['statistics_period']['supported_record_periods'] == []
    assert source['contribution_labels'] == ['Collected observations']
    assert source['processing']['raw_reason'] != 'qualified'
    assert result['metadata']['record_counts']['qualified'] == 0
    assert result['metadata']['record_counts']['collected'] == 1
    assert package == before


def test_mixed_current_records_keep_qualified_and_collected_source_roles_separate():
    package = {'source_registry': [{'source_id': 's', 'url': 'https://example.org/report'}],
               'final_dataset': [
                   {'record_id': 'q', 'source_id': 's', 'evidence_qualification': {'status': 'qualified'}},
                   {'record_id': 'collected', 'source_url': 'https://example.org/report#cases',
                    'evidence_qualification': {'status': 'candidate', 'reasons': ['missing_observation_period']}}],
               'candidate_records': [{'record_id': 'pending', 'source_id': 's', 'requires_human_review': True}],
               'record_inclusion_decisions': [{'record_id': 'pending', 'record_final_inclusion_status': 'pending_human_review',
                                                'reasons': ['Source identity requires review.']}]}
    source = catalog(package)['sources'][0]
    assert source['records_by_role'] == {'qualified': ['q'], 'candidate': ['pending'], 'context': [], 'collected': ['collected']}
    assert [detail['record_id'] for detail in source['evidence_details']] == ['q']
    assert source['collected_details'][0]['reasons'] == ['missing_observation_period']
    assert source['candidate_details'][0]['reasons'] == ['Source identity requires review.']
    assert source['candidate_details'][0]['record_final_inclusion_status'] == 'pending_human_review'
