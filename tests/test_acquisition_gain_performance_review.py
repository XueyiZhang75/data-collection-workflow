"""Independent integrity controls for call-local recovery document preparation."""
from copy import deepcopy
import hashlib
import pytest
from data_collection_workflow import evidence_products as products
from data_collection_workflow.workflow_recovery import _gain


def material():
    text = 'Pertussis in County A, Country A: 12 confirmed cases during 2025.'
    digest = hashlib.sha256(text.encode()).hexdigest()
    raw_digest = hashlib.sha256(b'original bytes').hexdigest()
    doc = dict(document_id='doc1', source_id='s', content_hash=raw_digest, text_hash=digest, clean_text=text)
    chunk = dict(chunk_id='chunk1', document_id='doc1', source_id='s', document_hash=raw_digest,
                 text=text, char_start=0, char_end=len(text), bound_context_spans=[])
    fields = dict(disease='Pertussis', country='Country A', reporting_period='2025', cases_confirmed=12)
    entries = [dict(field=key,value=value,supported=True,document_hash=raw_digest,quote=text,
                    locator=dict(document_id='doc1',chunk_id='chunk1',char_start=0,char_end=len(text)))
               for key,value in fields.items()]
    row = dict(fields,record_id='r',source_id='s',evidence_qualification=dict(status='candidate',product_kind='aggregate',field_evidence=entries))
    index = dict(documents=[doc],evidence_chunks={'chunk1':chunk})
    return row,index


def state(row,index):
    return dict(candidate_records=[row],documents=index['documents'],evidence_chunks=list(index['evidence_chunks'].values()))


@pytest.mark.parametrize('mutation', ['document_text','text_hash','document_source','locator_document',
    'chunk_hash','chunk_source','missing_chunk','negative_start','past_end','quote','chunk_bounds','bound_context','unsupported'])
def test_preparation_preserves_every_integrity_rejection(mutation):
    row,index=material()
    entry=row['evidence_qualification']['field_evidence'][0]
    doc=index['documents'][0]
    chunk=index['evidence_chunks']['chunk1']
    if mutation=='document_text': doc['clean_text']+=' altered'
    elif mutation=='text_hash': doc['text_hash']='0'*64
    elif mutation=='document_source': doc['source_id']='foreign'
    elif mutation=='locator_document': entry['locator']['document_id']='another-version'
    elif mutation=='chunk_hash': chunk['document_hash']='1'*64
    elif mutation=='chunk_source': chunk['source_id']='foreign'
    elif mutation=='missing_chunk': entry['locator']['chunk_id']='missing'
    elif mutation=='negative_start': entry['locator']['char_start']=-1
    elif mutation=='past_end': entry['locator']['char_end']=10000
    elif mutation=='quote': entry['quote']='Unrelated assertion.'
    elif mutation=='chunk_bounds': chunk['char_start']=10
    elif mutation=='bound_context': entry['locator']['bound_context_spans']=[dict(char_start=0,char_end=4,text='fake')]
    elif mutation=='unsupported': entry['supported']=False
    documents=products._documents(index)
    assert products._resolved(entry,index) is False
    assert products._resolved(entry,index,documents=documents) is False


def test_prepared_empty_collection_cannot_fall_back_to_valid_index():
    row,index=material()
    entry=row['evidence_qualification']['field_evidence'][0]
    assert products._resolved(entry,index)
    assert not products._resolved(entry,index,documents=[])


def test_prepared_reference_does_not_cache_document_integrity_result():
    row,index=material()
    entry=row['evidence_qualification']['field_evidence'][0]
    documents=products._documents(index)
    assert products._resolved(entry,index,documents=documents)
    index['documents'][0]['clean_text']+=' tampered after preparation'
    assert not products._resolved(entry,index,documents=documents)


def test_same_hash_other_document_version_does_not_supply_locator():
    row,index=material()
    original=index['documents'][0]
    other=dict(original,document_id='doc2',source_id='other')
    index['documents']=[other]
    entry=row['evidence_qualification']['field_evidence'][0]
    assert not products._resolved(entry,index,documents=products._documents(index))
    index['documents'].append(original)
    prepared=products._documents(index)
    assert products._resolved(entry,index,documents=prepared)
    entry['locator']['document_id']='doc2'
    assert not products._resolved(entry,index,documents=prepared)


def test_document_order_and_exact_duplicates_preserve_gain():
    row,index=material()
    before=_gain(state(row,index))
    other=dict(index['documents'][0],document_id='other',source_id='foreign')
    index['documents']=[other,index['documents'][0],deepcopy(index['documents'][0])]
    assert _gain(state(row,index))==before
    index['documents'].reverse()
    assert _gain(state(row,index))==before


def test_gain_reprepares_each_call_and_does_not_cross_session_reuse(monkeypatch):
    row,index=material()
    original=deepcopy(index)
    calls=[]
    real=products._documents
    def tracked(value):
        calls.append(value)
        return real(value)
    monkeypatch.setattr(products,'_documents',tracked)
    first=_gain(state(row,index))
    assert len(first)==4
    assert len(calls)==1
    index['documents']=[dict(index['documents'][0],clean_text='tampered')]
    assert _gain(state(row,index))==[]
    assert len(calls)==2
    assert _gain(state(deepcopy(row),original))==first
    assert len(calls)==3
    assert set(original)=={'documents','evidence_chunks'}


def test_stale_field_value_is_not_counted_as_progress():
    row,index=material()
    original=_gain(state(row,index))
    row['cases_confirmed']=99
    assert len(_gain(state(row,index)))==len(original)-1
    row['evidence_qualification']['reasons']=['location:contract_mismatch']
    assert _gain(state(row,index))==[]
