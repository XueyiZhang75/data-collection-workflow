import pytest
from test_focused_retry_semantic_change import run
from data_collection_workflow.nodes import extraction as ex

@pytest.mark.parametrize('quote,expected_calls', [('First patient had fever.',2), ('Second patient died.',1)])
def test_focus_stays_inside_the_previously_submitted_span(monkeypatch, quote, expected_calls):
    full='mpox in Brazil. First patient had fever. Second patient died.'
    prior='mpox in Brazil. First patient had fever.'
    def expanded(chunks, **kwargs):
        parent=chunks[0]
        return [dict(parent,chunk_id='c:case:1', parent_chunk_id='c', text=prior,
          case_span_quote=prior,case_span_start=0,case_span_end=len(prior))]
    monkeypatch.setattr(ex,'_expand_case_span_chunks_for_llm',expanded)
    calls,stats,_=run(monkeypatch,text=full,quote=quote)
    assert len(calls)==expected_calls
    if expected_calls==2:
        assert calls[1]['case_span_quote']==quote
        assert full[calls[1]['case_span_start']:calls[1]['case_span_end']]==quote
    else:
        assert stats['focused_recovery_call_count']==0

@pytest.mark.parametrize('text,quote', [
 ('mpox in Brazil. The patient had fever.', 'mpox in Brazil. The patient had fever. MORE'),
 ('mpox in Brazil. The patient had fever.', ''),
])
def test_focus_cannot_expand_original_input(monkeypatch,text,quote):
    calls,stats,_=run(monkeypatch,text=text,quote=quote)
    assert len(calls)==1


def test_repeated_patient_text_does_not_move_to_another_patient(monkeypatch):
    first='mpox in Brazil. The patient had fever.'
    full=first+' Later patient. The patient had fever.'
    quote='The patient had fever.'
    def expanded(chunks, **kwargs):
        return [dict(chunks[0],chunk_id='c:case:1',parent_chunk_id='c',text=first,
          case_span_quote=first,case_span_start=0,case_span_end=len(first))]
    monkeypatch.setattr(ex,'_expand_case_span_chunks_for_llm',expanded)
    def bad_position(c):
        return dict(c,case_span_start=full.rfind(quote),case_span_end=len(full))
    calls,_,_=run(monkeypatch,text=full,quote=quote,queue_change=bad_position)
    assert len(calls)==2
    assert calls[1]['case_span_start']==full.find(quote)
    assert calls[1]['case_span_end']<=len(first)


def test_parent_span_without_original_coordinates_is_deferred(monkeypatch):
    full='mpox in Brazil. The patient had fever. Later another patient died.'
    first='mpox in Brazil. The patient had fever.'
    def expanded(chunks, **kwargs):
        return [dict(chunks[0],chunk_id='c:case:1',parent_chunk_id='c',text=first,case_span_quote=first)]
    monkeypatch.setattr(ex,'_expand_case_span_chunks_for_llm',expanded)
    calls,_,_=run(monkeypatch,text=full,quote='The patient had fever.')
    assert len(calls)==1
