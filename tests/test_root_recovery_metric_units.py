import pytest
from test_recovery_metric_identity import material, repair

@pytest.mark.parametrize('metric,unit',[('cases_confirmed','cases'),('deaths','deaths')])
def test_same_assertion_with_both_metrics_uses_its_own_count_unit(monkeypatch,metric,unit):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    state,row=material(metric=metric)
    row['count_unit']=unit
    update,reason=repair(state)
    assert update is not None,reason
    assert update['fields']['metric_period_start']=='2025-02-02'

@pytest.mark.parametrize('metric,wrong_unit',[('cases_confirmed','deaths'),('deaths','cases')])
def test_same_assertion_does_not_make_other_metric_unit_interchangeable(monkeypatch,metric,wrong_unit):
    monkeypatch.setenv('PIPELINE_MODE','evidence')
    state,row=material(metric=metric)
    row['count_unit']=wrong_unit
    update,reason=repair(state)
    assert update is None and reason=='different_or_ambiguous_observation_count'
