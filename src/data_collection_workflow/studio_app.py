"""LangGraph Studio entry point.

LangGraph Studio imports the compiled `graph` object declared here. Runtime
behavior is controlled by environment variables: the same graph can run offline
fixture checks, live web collection, or a workflow runtime profile with all
three LLM stages enabled.
"""

from __future__ import annotations

# Use an absolute import: LangGraph CLI loads this file by path (not as a
# package member), so `from .graph import build_graph` would raise
# "attempted relative import with no known parent package".
from data_collection_workflow.graph import build_graph

graph = build_graph()
