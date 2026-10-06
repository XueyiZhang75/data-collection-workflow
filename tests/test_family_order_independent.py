"""Independent queue ordering controls, using only local fixture responses."""
import pytest
from test_dynamic_resource_family import run_pages

@pytest.mark.parametrize('disease,country',[('Measles','Canada'),('Dengue','Brazil'),('Unknown fever','India')])
def test_soft_checkpoint_disabled_does_not_disable_actual_order_updates(monkeypatch,tmp_path,disease,country):
    rt,state,visits,claims,result=run_pages(monkeypatch,tmp_path,disease=disease,country=country,targets=3,soft=0)
    assert state['source_registry'][1]['url'] in visits
    assert rt.ledger.snapshot()['used']['source_targets']==3
    assert len(result['documents'])==3


def test_duplicate_link_occurrences_do_not_consume_additional_targets(monkeypatch,tmp_path):
    rt,state,visits,claims,result=run_pages(monkeypatch,tmp_path,targets=4,links=[('/same.csv','Download case counts'),('/same.csv#table','Download case counts'),('/next.csv','Download case counts')])
    assert len(visits)==len(set(visits))==4
    assert rt.ledger.snapshot()['used']['source_targets']==4
    assert len(result['documents'])==4
    assert sum('/same.csv' in u for u in visits)==1


def test_empty_child_does_not_reduce_remaining_sibling_scores(monkeypatch,tmp_path):
    pages={'https://data.example/series-0.csv':('text/csv',b'')}
    rt,state,visits,claims,result=run_pages(monkeypatch,tmp_path,pages=pages)
    child_claims=[c for c in claims if '/series-' in c['url']]
    assert len(child_claims)==3
    assert child_claims[0]['url']=='https://data.example/series-0.csv'
    assert child_claims[0]['priority']==child_claims[1]['priority']
    assert child_claims[2]['priority']<child_claims[1]['priority']
    assert rt.ledger.snapshot()['used']['source_targets']==5
    failed=next(d for d in result['documents'] if d['url'].endswith('/series-0.csv'))
    assert not failed['content_readable']


def test_task_disease_is_not_replaced_by_queue_family_metadata(monkeypatch,tmp_path):
    rt,state,visits,claims,result=run_pages(monkeypatch,tmp_path,parent_group='independent-publication-group')
    assert state['structured_task']['disease']=='Example fever'
    assert all('independent-publication-group' not in d.get('clean_text','') for d in result['documents'])
    assert all(c['payload']['entry'].get('discovery_method')!='task_resource_link' or not c['payload']['entry'].get('source_independence_group') for c in claims)
