import hashlib
import unicodedata

import pytest

from data_collection_workflow.models import Document
from data_collection_workflow.query_policy import assess_query_task_fit, supported_event_terms


def _state():
    return {'structured_task': {'disease': 'test fever', 'location': 'Example Region'}}


@pytest.mark.parametrize('intent', [
    'downloadable spreadsheet XLSX open data',
    'provincial laboratory confirmed infections',
    'archived bulletin weekly report tables XLS XML',
    'données épidémiologiques bulletin hebdomadaire téléchargement tableaux',
    'datos epidemiológicos boletín semanal descarga laboratorio infecciones',
    'dados epidemiológicos boletim semanal relatório arquivo planilha',
    unicodedata.normalize('NFD', 'données épidémiologiques bulletin hebdomadaire'),
])
def test_generic_multilingual_retrieval_intents_are_not_event_facts(intent):
    result = assess_query_task_fit({'query': f'test fever Example Region {intent}'}, _state())
    assert result['accepted'], result


def test_retrieval_intents_do_not_admit_unbound_event_names():
    result = assess_query_task_fit(
        {'query': 'test fever downloadable XLSX laboratory vessel Orion'}, _state())
    assert not result['accepted']
    assert 'unbound_query_terms' in result['reasons']
    assert 'orion' in result['unbound_terms']


def _fetched_state():
    state = _state()
    text = 'The test fever cases were reported aboard vessel Orion.'
    state['documents'] = [Document(
        source_id='source-1', clean_text=text,
        content_hash=hashlib.sha256(b'<html>raw fetched response</html>').hexdigest(),
        text_hash=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        fetch_status='fetched', parse_status='parsed',
    ).model_dump()]
    state['supported_event_terms'] = [{
        'term': 'vessel Orion', 'source_id': 'source-1', 'quote': text,
        'document_hash': state['documents'][0]['content_hash'],
    }]
    return state


def test_actual_clean_text_document_supports_bound_event_query():
    state = _fetched_state()
    assert supported_event_terms(state) == ['vessel Orion']
    assert assess_query_task_fit(
        {'query': 'test fever vessel Orion cases', 'event_terms': ['vessel Orion']},
        state,
    )['accepted']


@pytest.mark.parametrize('damage', [
    'wrong_document_hash', 'wrong_content_hash', 'wrong_text_hash',
    'altered_quote', 'metadata_only', 'missing_id', 'ambiguous_id',
])
def test_event_support_is_bound_to_one_current_fetched_text(damage):
    state = _fetched_state()
    document = state['documents'][0]
    support = state['supported_event_terms'][0]
    document['document_id'] = support['document_id'] = 'doc-1'
    document['text'] = document['clean_text']
    if damage == 'wrong_document_hash':
        support['document_hash'] = 'stale-version'
    elif damage == 'wrong_content_hash':
        support['content_hash'] = 'stale-version'
    elif damage == 'wrong_text_hash':
        document['text_hash'] = hashlib.sha256(b'different text').hexdigest()
    elif damage == 'altered_quote':
        support['quote'] = 'The test fever cases were reported on vessel Orion.'
    elif damage == 'metadata_only':
        document['title'] = support['quote']
        document['metadata'] = {'description': support['quote']}
        document['clean_text'] = 'No event-specific content in the fetched page.'
        document.pop('text')
        document['text_hash'] = hashlib.sha256(document['clean_text'].encode()).hexdigest()
    elif damage == 'missing_id':
        support.pop('source_id')
        support.pop('document_id')
    elif damage == 'ambiguous_id':
        state['documents'].append(dict(document))
    assert supported_event_terms(state) == []
    assert not assess_query_task_fit({'query': 'test fever vessel Orion cases'}, state)['accepted']


def test_stale_legacy_text_cannot_override_current_clean_text():
    state = _fetched_state()
    document = state['documents'][0]
    document['document_id'] = 'doc-1'
    document['text'] = document['clean_text']
    document['clean_text'] = 'Only general disease information is present.'
    document['text_hash'] = hashlib.sha256(document['clean_text'].encode()).hexdigest()
    state['supported_event_terms'][0]['document_id'] = 'doc-1'
    assert supported_event_terms(state) == []


def test_legacy_document_id_and_exact_quote_remain_supported():
    state = _state()
    quote = 'The test fever cases were reported aboard vessel Orion.'
    state['documents'] = [{'document_id': 'doc-1', 'text': quote, 'content_hash': 'hash'}]
    state['supported_event_terms'] = [
        {'term': 'vessel Orion', 'document_id': 'doc-1', 'quote': quote}]
    assert supported_event_terms(state) == ['vessel Orion']


def test_user_request_can_authorize_an_event_without_a_source():
    state = _state()
    state['structured_task']['user_request'] = 'Collect test fever cases aboard vessel Orion.'
    assert assess_query_task_fit(
        {'query': 'test fever vessel Orion cases', 'event_terms': ['vessel Orion']},
        state,
    )['accepted']


@pytest.mark.parametrize('origin', ['fetched_quote', 'user_request'])
def test_partial_event_name_does_not_count_as_supported(origin):
    state = _fetched_state()
    support = state['supported_event_terms'][0]
    support['term'] = 'vessel Ori'
    if origin == 'user_request':
        state['documents'] = []
        state['structured_task']['user_request'] = 'Collect test fever cases on vessel Orion.'
    assert supported_event_terms(state) == []
    assert not assess_query_task_fit(
        {'query': 'test fever vessel Ori cases', 'event_terms': ['vessel Ori']}, state,
    )['accepted']


def test_maintained_publisher_name_is_valid_retrieval_vocabulary(monkeypatch):
    from data_collection_workflow import source_identity
    monkeypatch.setattr(source_identity,'load_source_identity_registry',lambda:[
        {'publisher_name':'Sentinel Observatory Institute','publisher_aliases':[]}
    ])
    query={'query':'test fever Sentinel Observatory Institute data'}
    assert assess_query_task_fit(query,_state())['accepted']
    monkeypatch.setattr(source_identity,'load_source_identity_registry',lambda:[])
    assert not assess_query_task_fit(query,_state())['accepted']
