"""Independent original-parent offset checks for focused v2."""
import pytest
from test_focused_retry_semantic_change import run
from data_collection_workflow.nodes import extraction as ex

@pytest.mark.parametrize('quote,body',[
 ('Second patient had fever.','Case 2. Second patient\n\t had fever. More details.'),
 ('患者於週末住院。 敘述保持原樣。','Case 2. 患者於週末住院。\n\u00a0敘述保持原樣。'),
])
def test_nonzero_expanded_parent_start_preserves_original_unicode_offsets(monkeypatch,quote,body):
    prefix='mpox in Brazil. Case 1. First patient recovered.\n'
    full=prefix+body
    start=len(prefix)
    def expanded(chunks,**kwargs):
        return [dict(chunks[0],chunk_id='c:case:2',parent_chunk_id='c',text=' '.join(body.split()),case_span_quote=' '.join(body.split()),case_span_start=start,case_span_end=len(full))]
    monkeypatch.setattr(ex,'_expand_case_span_chunks_for_llm',expanded)
    calls,stats,_=run(monkeypatch,text=full,quote=quote)
    assert len(calls)==2
    chosen=calls[1]
    a,b=chosen['case_span_start'],chosen['case_span_end']
    assert start<=a<b<=len(full)
    assert full[a:b]==chosen['case_span_quote']
    assert ' '.join(full[a:b].split())==quote
    assert chosen['char_start']==150 and chosen['char_end']==150+len(full)
    assert stats['focused_recovery_call_count']==1


def test_repeated_quote_in_earlier_patient_cannot_attract_second_patient_focus(monkeypatch):
    quote='The patient had fever.'
    first='mpox in Brazil. Case 1. '+quote+'\n'
    second='Case 2. '+quote+' Other details.'
    full=first+second
    def expanded(chunks,**kwargs):
        return [dict(chunks[0],chunk_id='c:case:2',parent_chunk_id='c',text=second,case_span_quote=second,case_span_start=len(first),case_span_end=len(full))]
    monkeypatch.setattr(ex,'_expand_case_span_chunks_for_llm',expanded)
    def earlier(c):return dict(c,case_span_start=full.find(quote),case_span_end=full.find(quote)+len(quote))
    calls,stats,_=run(monkeypatch,text=full,quote=quote,queue_change=earlier)
    assert len(calls)==2
    assert calls[1]['case_span_start']==full.rfind(quote)
    assert full[calls[1]['case_span_start']:calls[1]['case_span_end']]==quote
