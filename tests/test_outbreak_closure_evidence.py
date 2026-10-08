import hashlib

from data_collection_workflow.evidence_qualification import assess_record_evidence
from data_collection_workflow.outbreak_closure import closure_context_evidence


def source():
    text = ('The Health Ministry declared the measles outbreak in Canada over. '
            'Measles in Canada: 10 confirmed cases in 2025.')
    digest = hashlib.sha256(text.encode()).hexdigest()
    doc = {'source_id': 's', 'content_hash': digest, 'text_hash': digest, 'clean_text': text}
    chunk = {'source_id': 's', 'chunk_id': 'c', 'document_hash': digest,
             'text': text, 'char_start': 0, 'char_end': len(text), 'bound_context_spans': []}
    row = {'record_id': 'r', 'source_id': 's', 'supporting_chunk_id': 'c',
           'disease': 'measles', 'country': 'Canada', 'reporting_period': '2025',
           'cases_confirmed': 10, 'evidence_quote': text,
           'field_provenance_json': {'cases_confirmed': {'quote': 'Measles in Canada: 10 confirmed cases in 2025.'}}}
    index = {'documents': [doc], 'evidence_chunks': {'c': chunk}}
    qualification = assess_record_evidence(row, contract={}, evidence_index=index).to_dict()
    assert qualification['status'] == 'qualified'
    return row, qualification, index


def test_verified_closure_context_retains_declaration_outside_numeric_sentence():
    row, qualification, index = source()
    evidence = closure_context_evidence(row, qualification, index)
    assert len(evidence) == 1
    assert 'Health Ministry declared' in evidence[0]['quote']
    assert '10 confirmed cases' in evidence[0]['quote']
    assert evidence[0]['supported'] is True
    assert evidence[0]['locator']['char_end'] == len(index['documents'][0]['clean_text'])


def test_tampered_document_cannot_supply_closure_context():
    row, qualification, index = source()
    index['documents'][0]['clean_text'] += ' altered'
    assert closure_context_evidence(row, qualification, index) == []


def test_unqualified_count_cannot_acquire_closure_support():
    row, qualification, index = source()
    qualification['status'] = 'candidate'
    assert closure_context_evidence(row, qualification, index) == []


def test_other_chunk_cannot_supply_closure_context():
    row, qualification, index = source()
    text = index['documents'][0]['clean_text']
    start = text.index('Measles in Canada:')
    chunk = index['evidence_chunks']['c']
    chunk.update(text=text[start:], char_start=start)
    assert closure_context_evidence(row, qualification, index) == []


def test_changed_value_cannot_reuse_qualified_closure_context():
    row, qualification, index = source()
    row['cases_confirmed'] = 999
    assert closure_context_evidence(row, qualification, index) == []
