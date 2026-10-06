"""Bounded recovery gain preparation without relaxing any field evidence checks."""
from copy import deepcopy
import hashlib

from data_collection_workflow import evidence_products as products
from data_collection_workflow.evidence_qualification import build_evidence_index
from data_collection_workflow.workflow_recovery import _gain
from data_collection_workflow.session_runtime import fingerprint


def gain_state():
    quote = 'During 2025, Canada reported an annual total of 12 confirmed measles cases in adults, including 2 deaths.'
    text = quote + '\n' + ('Unrelated narrative without additional facts. ' * 2000)
    digest = hashlib.sha256(text.encode()).hexdigest()
    doc = {'document_id': 'd', 'source_id': 's', 'content_hash': digest, 'text_hash': digest, 'clean_text': text}
    chunk = {'chunk_id': 'c', 'document_id': 'd', 'source_id': 's', 'document_hash': digest,
             'text': quote, 'char_start': 0, 'char_end': len(quote), 'bound_context_spans': []}
    fields = {'disease': 'measles', 'country': 'Canada', 'reporting_period': '2025',
              'case_definition': 'confirmed', 'population_scope': 'adults',
              'unit': 'cases', 'statistical_count_type': 'annual_total',
              'cases_confirmed': 12, 'deaths': 2}
    proof = {'supported': True, 'document_hash': digest, 'quote': quote,
             'locator': {'document_id': 'd', 'chunk_id': 'c', 'char_start': 0, 'char_end': len(quote)}}
    row = {'record_id': 'r', 'source_id': 's', **fields, 'evidence_qualification': {
        'status': 'candidate', 'field_evidence': [dict(proof, field=k, value=v) for k, v in fields.items()]}}
    # Repeated document objects and substantial unrelated bodies reproduce the
    # costly preparation path without a wall-clock assertion or live material.
    docs = [deepcopy(doc) for _ in range(16)]
    for n in range(24):
        body = f'Unrelated document {n}. ' + 'Background material. ' * 3000
        other_hash = hashlib.sha256(body.encode()).hexdigest()
        docs.append({'document_id': f'other-{n}', 'source_id': f'other-source-{n}',
                     'content_hash': other_hash, 'text_hash': other_hash, 'clean_text': body})
    return {'documents': docs, 'evidence_chunks': [chunk], 'candidate_records': [row]}, fields


def expected_gain(fields):
    scope_names = ('disease', 'country', 'reporting_period', 'case_definition',
                   'population_scope', 'unit', 'statistical_count_type')
    scope = {k: fields[k] for k in scope_names if k in fields}
    return sorted(fingerprint([None, scope, name, value]) for name, value in fields.items())


def document_spy(monkeypatch):
    calls = []
    original = products._documents
    def watched(index):
        calls.append(index)
        return original(index)
    monkeypatch.setattr(products, '_documents', watched)
    return calls


def test_gain_prepares_documents_once_for_many_fields_and_full_output_matches(monkeypatch):
    state, fields = gain_state()
    original = deepcopy(state)
    calls = document_spy(monkeypatch)
    assert _gain(state) == expected_gain(fields)
    assert state == original
    assert len(calls) == 1


def test_gain_without_matching_supported_fields_does_not_prepare_documents(monkeypatch):
    state, _ = gain_state()
    for entry in state['candidate_records'][0]['evidence_qualification']['field_evidence']:
        entry['supported'] = False
    calls = document_spy(monkeypatch)
    assert _gain(state) == []
    assert calls == []


def test_gain_preparation_is_local_to_each_call_and_detects_changed_original(monkeypatch):
    state, fields = gain_state()
    calls = document_spy(monkeypatch)
    assert _gain(state) == expected_gain(fields)
    for doc in state['documents']:
        if doc['document_id'] == 'd':
            doc['clean_text'] += ' changed bytes'
    assert _gain(state) == []
    assert len(calls) == 2


def test_gain_still_checks_each_field_value_against_current_record():
    state, fields = gain_state()
    state['candidate_records'][0]['country'] = 'Elsewhere'
    fields.pop('country')
    assert _gain(state) == expected_gain(fields)


def test_explicit_empty_prepared_documents_do_not_fall_back_to_index():
    state, _ = gain_state()
    entry = state['candidate_records'][0]['evidence_qualification']['field_evidence'][0]
    index = build_evidence_index(state)
    assert products._resolved(entry, index) is True
    assert products._resolved(entry, index, documents=[]) is False


def test_gain_rechecks_original_text_hash_for_every_supported_field(monkeypatch):
    state, fields = gain_state()
    raw = state['documents'][0]['clean_text'].encode()
    checks = []
    original = hashlib.sha256
    def watched(value=b'', **kwargs):
        if value == raw:
            checks.append(value)
        return original(value, **kwargs)
    monkeypatch.setattr(products.hashlib, 'sha256', watched)
    assert _gain(state) == expected_gain(fields)
    assert len(checks) == len(fields)
