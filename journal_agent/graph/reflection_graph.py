import logging

from langgraph.graph import END, START, StateGraph

from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.graph.nodes.insight_nodes import make_cluster_fragments, make_label_clusters, make_verify_citations
from journal_agent.graph.nodes.stores import (
    make_save_insights,
)
from journal_agent.graph.routing import goto

logger = logging.getLogger(__name__)

from journal_agent.graph.state import ReflectionState
from journal_agent.stores import InsightsRepository


def build_reflection_graph(
        registry: LLMRegistry,
        insights_repo: InsightsRepository,
):
    """todo add comments

    """

    reflection_llm = registry.get("classifier")

    # noinspection PyTypeChecker
    builder = StateGraph(ReflectionState)  # no_qa

    # Reflection pipeline nodes
    builder.add_node("cluster_fragments", make_cluster_fragments())
    builder.add_node("label_clusters", make_label_clusters(llm=reflection_llm))
    builder.add_node("verify_citations", make_verify_citations(llm=reflection_llm, max_concurrency=3))


    # Wiring
    builder.add_edge(START, "cluster_fragments")
    builder.add_conditional_edges("cluster_fragments", goto("label_clusters"))
    builder.add_conditional_edges("label_clusters", goto("verify_citations"))
    builder.add_edge("verify_citations", END)

    compiled = builder.compile()
    return compiled
