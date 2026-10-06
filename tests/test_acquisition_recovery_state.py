"""Recovery search convergence metadata must survive actual graph state updates."""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph,START,END
from data_collection_workflow.state import DataCollectionState


def test_new_evidence_search_streak_boundary_survives_checkpoint_without_resetting_history():
    graph=StateGraph(DataCollectionState)
    graph.add_node('evidence_gain',lambda state:{'recovery_search_streak_start':2})
    graph.add_node('next_control',lambda state:{'recovery_stop_reason':
        'search_after_new_evidence' if state.get('recovery_search_streak_start')==2 else 'stale_stop'})
    graph.add_edge(START,'evidence_gain')
    graph.add_edge('evidence_gain','next_control')
    graph.add_edge('next_control',END)
    compiled=graph.compile(checkpointer=InMemorySaver())
    history=[{'action_id':'prior-search','status':'attempted_empty','attempts':1}]
    config={'configurable':{'thread_id':'one-session'}}
    result=compiled.invoke({'recovery_action_history':history},config)
    assert result['recovery_stop_reason']=='search_after_new_evidence'
    saved=compiled.get_state(config).values
    assert saved['recovery_search_streak_start']==2
    assert saved['recovery_action_history']==history
