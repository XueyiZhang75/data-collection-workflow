"""Independent stage-cap edges: absence, global override, zero, stage isolation."""
import pytest
from data_collection_workflow.session_runtime import derive_budget_limits, RunContext, BudgetExceeded
KIND="model:SourceCriticAgentOutput"

def config():
    return {"pipeline_mode":"evidence","llm":{"source_critic":{"max_sources":4}},"universal":{"budget_policy":{"version":2,"mode":"adaptive"}}}

@pytest.mark.parametrize("value",[None,"missing"])
def test_absent_cap_preserves_default_model_budget(value):
    c=config()
    if value=="missing":c["llm"]["source_critic"].pop("max_sources")
    else:c["llm"]["source_critic"]["max_sources"]=None
    assert KIND not in derive_budget_limits(c)

def test_global_budget_override_cannot_be_increased_by_stage_or_model_override():
    c=config();c["universal"]["budget_limits"]={KIND:1};c["universal"]["model_limits"]={"SourceCriticAgentOutput":3}
    assert derive_budget_limits(c)[KIND]==1

def test_critic_zero_does_not_dispatch_or_charge(tmp_path):
    c=config();c["universal"]["model_limits"]={"SourceCriticAgentOutput":0}
    ctx=RunContext(tmp_path,c)
    def forbidden():raise AssertionError("provider must not execute")
    with pytest.raises(BudgetExceeded):ctx.call(KIND,{"source":"one"},forbidden,recovery=True)
    snap=ctx.ledger.snapshot()
    assert snap["used"].get(KIND,0)==0
    assert snap["remaining"][KIND]==0

def test_critic_mapping_does_not_change_other_stage_caps():
    c=config();c["llm"]["source_identity"]={"max_sources":12};c["llm"]["source_credibility"]={"max_sources":3}
    limits=derive_budget_limits(c)
    assert limits[KIND]==4
    assert limits["model:SourceIdentityAgentOutput"]==12
    assert limits["model:LLMSourceCredibilitySuggestion"]==3
    assert limits["extraction"]==2400
