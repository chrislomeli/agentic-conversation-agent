import asyncio
import json
from collections import defaultdict
from typing import Callable, Coroutine, Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import HDBSCAN

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure import context_builder
from journal_agent.configure.config_builder import MINIMUM_VERIFIER_SCORE, MINIMUM_CLUSTER_LABEL_SCORE, \
    MINIMUM_CLUSTER_SCORE
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace, logger
from journal_agent.graph.nodes.classifiers import DEFAULT_LLM_CONCURRENCY
from journal_agent.graph.state import ReflectionState
from journal_agent.model.insights import ProposedSubject, Subject, Vote
from journal_agent.model.session import Cluster, Fragment, StatusValue, Insight, PromptKey, InsightDraft, \
    InsightVerifierScore, VerifierStatus, ContextSpecification, ClusterList, FragmentClusterRequest
from journal_agent.stores.subjects_repo import SubjectsRepository

import warnings

warnings.filterwarnings("ignore", message=".*Expected `none`.*", category=UserWarning, module="pydantic")


def score_cluster(
        cluster: Cluster,
        frag_by_id: dict[str, Fragment],
        recency_weight: float = 0.5,
) -> None:
    """Populate ``cluster.score``: size + recency_weight * span_days.

    Size rewards recurrence; span_days rewards themes persisting across time
    vs bursting in one session. Kept as a standalone helper so it can be lifted
    back into a dedicated ``score_clusters`` node if scoring grows.
    """
    timestamps = sorted(frag_by_id[fid].timestamp for fid in cluster.fragment_ids)
    span_days = (timestamps[-1] - timestamps[0]).total_seconds() / 86400
    cluster.score = len(cluster.fragment_ids) + recency_weight * span_days


def make_cluster_fragments_hdb() -> Callable[..., dict]:
    @node_trace("cluster_fragments")
    def cluster_fragments_hdb(state: ReflectionState) -> dict:
        try:
            fragments = state.fragments
            if not fragments:
                return {"clusters": []}

            vectors = np.vstack([f.embedding for f in fragments])

            hdb = HDBSCAN(min_cluster_size=3)
            hdb.fit(vectors)

            # Group fragments by label; -1 = noise, skip those
            groups: dict[int, list] = defaultdict(list)
            outliers: list[Fragment] =[]
            for fragment, label in zip(fragments, hdb.labels_):
                if label != -1:
                    groups[label].append(fragment)
                else:
                    outliers.append(fragment)

            clusters = [
                Cluster(
                    fragment_ids=[f.fragment_id for f in frags],
                    centroid=np.mean([f.embedding for f in frags], axis=0).tolist(),
                )
                for frags in groups.values()
            ]

            frag_by_id = {f.fragment_id: f for f in fragments}
            for cluster in clusters:
                score_cluster(cluster, frag_by_id)

            clusters = [c for c in clusters if c.score >= MINIMUM_CLUSTER_SCORE]

            # diagnostic — remove when clustering is stable
            noise_count = sum(1 for label in hdb.labels_ if label == -1)
            logger.info("HDBSCAN: %d fragments → %d clusters, %d noise", len(fragments), len(clusters), noise_count)
            for i, cluster in enumerate(clusters):
                frags = [frag_by_id[fid] for fid in cluster.fragment_ids]
                tags = sorted({t.tag for f in frags for t in f.tags})
                logger.info(
                    "  cluster %d  size=%d  score=%.1f  tags=%s",
                    i, len(frags), cluster.score, tags,
                )
                for f in frags:
                    logger.info("    [%s] %s", f.fragment_id, f.content[:80])

            return {
                "fragments": fragments,
                "clusters": clusters
            }
        except Exception as e:
            logger.exception("Cluster fragments failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return cluster_fragments_hdb

def make_create_clusters(llm: LLMClient) -> Callable[ ..., Coroutine[Any, Any, dict]]:
    @node_trace("cluster_fragments")
    async def create_clusters(state: ReflectionState) -> dict:
        try:
            fragments = state.fragments
            if not fragments:
                return {"clusters": []}

            fragment_context = json.dumps([
                FragmentClusterRequest(
                    fragment_id=f.fragment_id,
                    content=f.content,
                    tags=[i.tag for i in f.tags],
                    timestamp=f.timestamp,
                ).model_dump(mode="json")
                for f in fragments
            ])

            prompt = get_prompt(key=PromptKey.CREATE_CLUSTERS)
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=fragment_context)
            ]

            # involve the llm
            structured_llm = llm.astructured(ClusterList)
            cluster_list: ClusterList = structured_llm.invoke(messages)

            return {
                "fragments": fragments,
                "clusters": cluster_list.clusters
            }
        except Exception as e:
            logger.exception("Cluster fragments failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return create_clusters


def make_label_clusters(llm: LLMClient, max_concurrency: int = DEFAULT_LLM_CONCURRENCY) -> Callable[
    ..., Coroutine[Any, Any, dict]]:

    @node_trace("label_clusters")
    async def label_clusters(state: ReflectionState) -> dict:
        try:
            clusters = state.clusters
            if not clusters:
                return {"insights": []}

            frag_by_id = {f.fragment_id: f for f in state.fragments}

            system = SystemMessage(get_prompt(PromptKey.LABEL_CLUSTERS))
            structured_llm = llm.astructured(InsightDraft)
            sem = asyncio.Semaphore(max_concurrency)

            async def label_cluster(cluster: Cluster) -> Insight:
                async with sem:
                    payload = {
                        "fragments": [
                            {
                                "id": frag_by_id[fid].fragment_id,
                                "text": frag_by_id[fid].content,
                                "timestamp": frag_by_id[fid].timestamp.isoformat(),
                                "tags": [t.tag for t in frag_by_id[fid].tags],
                            }
                            for fid in cluster.fragment_ids
                        ],
                    }
                    human = HumanMessage(content=json.dumps(payload))
                    draft = await structured_llm.ainvoke([system, human])
                    return Insight(
                        label=draft.label,
                        body=draft.body,
                        label_confidence=draft.vector_score,
                        fragment_ids=cluster.fragment_ids,
                    )

            all_insights = await asyncio.gather(
                *(label_cluster(c) for c in clusters)
            )
            insights = [i for i in all_insights if i.label_confidence >= MINIMUM_CLUSTER_LABEL_SCORE]

            return {"insights": insights}

        except Exception as e:
            logger.exception("label_clusters failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return label_clusters


def make_verify_citations(llm: LLMClient, max_concurrency: int = DEFAULT_LLM_CONCURRENCY) -> Callable[..., Coroutine[Any, Any, dict]]:
    @node_trace("verify_citations")
    async def verify_citations(state: ReflectionState) -> dict:
        try:
            system = SystemMessage(get_prompt(PromptKey.VERIFY_INSIGHTS))
            structured_llm = llm.astructured(InsightVerifierScore)
            sem = asyncio.Semaphore(max_concurrency)

            fragments = state.fragments
            insights = state.insights

            async def worker(insight: Insight) -> Insight:
                async with sem:
                    cited_fragments = [f.content for f in fragments if f.fragment_id in insight.fragment_ids]
                    cited_text = "\n\n".join(f"- {c}" for c in cited_fragments)
                    payload = f"""
                    INSIGHT BEING VERIFIED:
    
                    Label: {insight.label}
                    Body:  {insight.body}
                    
                    FRAGMENTS (fragments cited as evidence):
                    {cited_text}
                    """

                    human = HumanMessage(content=payload)
                    score: InsightVerifierScore = await structured_llm.ainvoke([system, human])
                    if not cited_fragments:
                        return insight.model_copy(update={
                            "verifier_status": VerifierStatus.FAILED,
                            "verifier_score": 0.0,
                            "verifier_comments": "No cited fragments found.",
                        })
                    return insight.model_copy(update={
                        "verifier_score": score.verifier_score,
                        "verifier_comments": score.verifier_comments,
                        "verifier_status": VerifierStatus.VERIFIED if score.verifier_score >= MINIMUM_VERIFIER_SCORE else VerifierStatus.FAILED,
                    })

            verified_insights = await asyncio.gather(
                *(worker(i) for i in insights)
            )

            # print("\nVerified insights:\n")
            # for insight in verified_insights:
            #     print(json.dumps(insight.model_dump(), indent=2, default=str))

            return {"verified_insights": verified_insights}

        except Exception as e:
            logger.exception("verify_citations failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return verify_citations


# ═════════════════════════════════════════════════════════════════════════════
# Phase 11 — Claim-based insights (do-nothing skeleton)
#
# These factories establish the shape of the new reflection pipeline. Bodies
# return shape-correct empty state so the graph compiles and runs end-to-end
# without LLM calls or DB writes. Real implementations land in subsequent PRs.
#
# Design doc: design/phase11-claim-based-insights.md
# ═════════════════════════════════════════════════════════════════════════════


def make_route_candidates(
    subjects_repo: SubjectsRepository,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the route_candidates node.

    Real behavior (TODO):
        For each fragment in state.fragments, embed its content (or use its
        existing embedding) and search the `claims` table by cosine similarity
        on the current claim embeddings. Return the top-K candidate Subjects
        whose current claims sit above ROUTE_CANDIDATES_MIN_SIMILARITY.

    Skeleton behavior:
        Return an empty candidate_subjects list.
    """

    @node_trace("route_candidates")
    async def route_candidates(state: ReflectionState) -> dict:
        # TODO(phase11): vector search over current claim embeddings.
        return {"candidate_subjects": []}

    return route_candidates


def make_classify_stance(
    llm: LLMClient,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the classify_stance node.

    Real behavior (TODO):
        For each (fragment, candidate_subject) pair, call the LLM with the
        stance_classifier prompt and parse a StanceResponse. Filter votes
        below MIN_VOTE_STRENGTH. Convert each StanceVote into a persisted
        Vote (filling fragment_id, fragment_dated_at, processed_at,
        model_signature, claim_version_at_vote). Append to state.votes.

    Skeleton behavior:
        Return an empty votes list.
    """

    @node_trace("classify_stance")
    async def classify_stance(state: ReflectionState) -> dict:
        # TODO(phase11): per-fragment LLM call → StanceResponse → Vote rows.
        return {"votes": []}

    return classify_stance


def make_propose_subject(
    llm: LLMClient,
    subjects_repo: SubjectsRepository,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the propose_subject node.

    Real behavior (TODO):
        Run only when state.votes contains no vote with strength above
        SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH. Call the LLM with the
        subject_proposer prompt over (existing_subjects, fragment); parse
        a ProposerResponse. If non-null, set state.proposed_subject and
        append the bundled initial_vote to state.votes.

    Skeleton behavior:
        Leave proposed_subject None and votes unchanged.
    """

    @node_trace("propose_subject")
    async def propose_subject(state: ReflectionState) -> dict:
        # TODO(phase11): conditional LLM call → ProposerResponse → optional new Subject + Vote.
        return {"proposed_subject": None}

    return propose_subject


def make_persist_votes(
    subjects_repo: SubjectsRepository,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the persist_votes node.

    Real behavior (TODO):
        - If proposed_subject is set: create the Subject + first Claim in DB.
        - Insert all rows in state.votes (subjects.last_activity_at updates).
        - Insert one fragment_processing row per processed fragment with the
          model_signature and the resulting vote_count.
        - Update state.status = StatusValue.PROCESSING on success.

    Skeleton behavior:
        Set status to PROCESSING; do not touch the DB.
    """

    @node_trace("persist_votes")
    async def persist_votes(state: ReflectionState) -> dict:
        # TODO(phase11): write subjects/claims/votes/fragment_processing rows.
        return {"status": StatusValue.PROCESSING}

    return persist_votes


# ── Vote weighting strategy (stub) ────────────────────────────────────────


def compute_traction(votes: list[Vote], strategy: str = "simple_sum") -> float:
    """Aggregate a vote stream into a scalar traction score.

    This is the policy plug point. v1 will implement `simple_sum`. Future
    strategies can layer in:
        - recency decay (exponential or half-life)
        - stance-strength weighting (already in vote.strength)
        - fragment-quality weighting (length, specificity)
        - independence (down-weight votes from same journal entry)
        - asymmetric sensitivity (contradicts may matter more than supports)
        - source/context (signals dict carries the raw data)

    The data model stores raw votes; this function decides what they mean.
    Keeping it pluggable means future improvements compound on existing data
    without migration.

    Args:
        votes: Active (non-invalidated) votes to aggregate. Caller filters
            by subject_id and fragment_dated_at <= as_of.
        strategy: Identifier for the weighting strategy to apply. Currently
            only "simple_sum" is supported; raises NotImplementedError otherwise.

    Returns:
        Scalar traction score. Sign indicates net stance; magnitude indicates
        accumulated evidence weight.
    """
    # TODO(phase11): implement simple_sum and the strategies above.
    raise NotImplementedError(
        f"compute_traction({strategy=!r}) not yet implemented in skeleton."
    )
