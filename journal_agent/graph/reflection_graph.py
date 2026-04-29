import logging

from langgraph.graph import END, START, StateGraph

from journal_agent.comms.llm_registry import LLMRegistry
from journal_agent.graph.nodes.insight_nodes import (
    make_classify_stance,
    make_create_clusters,
    make_label_clusters,
    make_persist_votes,
    make_propose_subject,
    make_route_candidates,
    make_verify_citations,
)
from journal_agent.graph.nodes.stores import (
    make_save_insights,
)
from journal_agent.graph.routing import goto
from journal_agent.graph.state import ReflectionState
from journal_agent.stores import InsightsRepository, SubjectsRepository

logger = logging.getLogger(__name__)


def build_reflection_graph(
        registry: LLMRegistry,
        insights_repo: InsightsRepository,
):
    """Build the Phase 10 cluster-based reflection graph.

    Pipeline: cluster_fragments → label_clusters → verify_citations → END.
    Retained alongside Phase 11 for coexistence; will be retired once the
    new pipeline has proven itself.
    """

    reflection_llm = registry.get("classifier")

    # noinspection PyTypeChecker
    builder = StateGraph(ReflectionState)  # no_qa

    # Reflection pipeline nodes
    builder.add_node("cluster_fragments", make_create_clusters(llm=reflection_llm))
    builder.add_node("label_clusters", make_label_clusters(llm=reflection_llm))
    builder.add_node("verify_citations", make_verify_citations(llm=reflection_llm, max_concurrency=3))


    # Wiring
    builder.add_edge(START, "cluster_fragments")
    builder.add_conditional_edges("cluster_fragments", goto("label_clusters"))
    builder.add_conditional_edges("label_clusters", goto("verify_citations"))
    builder.add_edge("verify_citations", END)

    compiled = builder.compile()
    return compiled


def build_claim_reflection_graph(
        registry: LLMRegistry,
        subjects_repo: SubjectsRepository,
):
    """Build the Phase 11 claim-based reflection graph (do-nothing skeleton).

    Pipeline (per-fragment invocation):
        START
          ↓
        route_candidates    — vector search → state.candidate_subjects
          ↓
        classify_stance     — LLM stance classifier → state.votes
          ↓
        propose_subject     — conditional LLM proposer → state.proposed_subject
          ↓
        persist_votes       — write subjects/claims/votes/processing rows
          ↓
        END

    The claim_regenerator runs on a separate, scheduled path (not part of
    this per-fragment graph) — see SubjectsRepository.vote_count_since +
    config_builder.CLAIM_REGEN_VOTE_GAP for the trigger.

    Skeleton: every node returns shape-correct empty state; no LLM calls,
    no DB writes. Wiring is real. Use to verify the graph compiles and runs.

    Design doc: design/phase11-claim-based-insights.md
    """

    classifier_llm = registry.get("classifier")

    # noinspection PyTypeChecker
    builder = StateGraph(ReflectionState)  # no_qa

    builder.add_node("route_candidates", make_route_candidates(subjects_repo=subjects_repo))
    builder.add_node("classify_stance", make_classify_stance(llm=classifier_llm))
    builder.add_node(
        "propose_subject",
        make_propose_subject(llm=classifier_llm, subjects_repo=subjects_repo),
    )
    builder.add_node("persist_votes", make_persist_votes(subjects_repo=subjects_repo, llm=classifier_llm))

    builder.add_edge(START, "route_candidates")
    builder.add_conditional_edges("route_candidates", goto("classify_stance"))
    builder.add_conditional_edges("classify_stance", goto("propose_subject"))
    builder.add_conditional_edges("propose_subject", goto("persist_votes"))
    builder.add_edge("persist_votes", END)

    compiled = builder.compile()
    return compiled
