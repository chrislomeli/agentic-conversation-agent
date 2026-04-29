import asyncio
import json
import logging
from collections import defaultdict
from typing import Callable, Coroutine, Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import HDBSCAN

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.config_builder import (
    MINIMUM_VERIFIER_SCORE,
    MINIMUM_CLUSTER_LABEL_SCORE,
    MINIMUM_CLUSTER_SCORE,
    MIN_VOTE_STRENGTH,
    ROUTE_CANDIDATES_TOP_K,
    ROUTE_CANDIDATES_MIN_SIMILARITY,
    SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH,
    PROPOSER_DEDUP_SIMILARITY,
    INSIGHT_BATCH_SIZE,
    STANCE_BATCH_SIZE,
)
from journal_agent.configure.prompts import get_prompt, get_prompt_version
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifiers import DEFAULT_LLM_CONCURRENCY
from journal_agent.graph.state import ReflectionState
from journal_agent.model.insights import (
    BatchStanceResponse,
    BatchVerifierResponse,
    CandidateSubject,
    FragmentWorkItem,
    ProposedSubject,
    ProposerResponse,
    Stance,
    Vote,
)
from journal_agent.model.session import Cluster, Fragment, StatusValue, Insight, PromptKey, InsightDraft, \
    VerifierStatus, ClusterList, FragmentClusterRequest
from journal_agent.stores.subjects_repo import SubjectsRepository

import warnings

warnings.filterwarnings("ignore", message=".*Expected `none`.*", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)


def stance_model_signature(llm: LLMClient) -> str:
    """Single source of truth for the model_signature stamped on stance votes.

    The reflect node uses this to ask `fetch_unprocessed_fragments` for work;
    classify_stance and persist_votes use it to write votes and processing rows.
    Any drift between callers makes every fragment look unprocessed forever.
    """
    return f"{llm.model}/stance_classifier/{get_prompt_version(PromptKey.STANCE_CLASSIFIER_BATCH)}"


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


def make_verify_citations(
    llm: LLMClient,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
    batch_size: int = INSIGHT_BATCH_SIZE,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    @node_trace("verify_citations")
    async def verify_citations(state: ReflectionState) -> dict:
        try:
            system = SystemMessage(get_prompt(PromptKey.VERIFY_INSIGHTS_BATCH))
            structured_llm = llm.astructured(BatchVerifierResponse)
            sem = asyncio.Semaphore(max_concurrency)

            frag_content = {f.fragment_id: f.content for f in state.fragments}
            insights = state.insights

            batches = [insights[i:i + batch_size] for i in range(0, len(insights), batch_size)]

            async def verify_batch(batch: list[Insight]) -> list[Insight]:
                async with sem:
                    payload = {
                        "items": [
                            {
                                "insight_id": insight.insight_id,
                                "label": insight.label,
                                "body": insight.body,
                                "fragments": [frag_content[fid] for fid in insight.fragment_ids if fid in frag_content],
                            }
                            for insight in batch
                        ]
                    }
                    try:
                        response: BatchVerifierResponse = await structured_llm.ainvoke(
                            [system, HumanMessage(content=json.dumps(payload))]
                        )
                    except Exception:
                        logger.exception("verify_citations batch LLM call failed")
                        return batch

                    score_map = {r.insight_id: r for r in response.results}
                    verified = []
                    for insight in batch:
                        result = score_map.get(insight.insight_id)
                        cited = [frag_content[fid] for fid in insight.fragment_ids if fid in frag_content]
                        if not cited:
                            verified.append(insight.model_copy(update={
                                "verifier_status": VerifierStatus.FAILED,
                                "verifier_score": 0.0,
                                "verifier_comments": "No cited fragments found.",
                            }))
                        elif result is None:
                            verified.append(insight.model_copy(update={
                                "verifier_status": VerifierStatus.FAILED,
                                "verifier_score": 0.0,
                                "verifier_comments": "Verifier did not return a result for this insight.",
                            }))
                        else:
                            verified.append(insight.model_copy(update={
                                "verifier_score": result.verifier_score,
                                "verifier_comments": result.verifier_comments,
                                "verifier_status": VerifierStatus.VERIFIED if result.verifier_score >= MINIMUM_VERIFIER_SCORE else VerifierStatus.FAILED,
                            }))
                    return verified

            batched_results = await asyncio.gather(*(verify_batch(b) for b in batches))
            verified_insights = [insight for batch in batched_results for insight in batch]

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

    Reads:  state.fragments
    Writes: state.work_items (one FragmentWorkItem per fragment)

    Real behavior (TODO):
        For each fragment in state.fragments (in parallel), use
        fragment.embedding to call subjects_repo.search_candidate_subjects(...)
        and bundle the returned (Subject, Claim, similarity) triples as
        CandidateSubject objects. Wrap in a FragmentWorkItem.

    Skeleton behavior:
        Initialize one empty-candidates work item per fragment so downstream
        nodes have a real list to map over. The shape is correct; only the
        candidate-search step is stubbed.
    """

    @node_trace("route_candidates")
    async def route_candidates(state: ReflectionState) -> dict:
        if not state.fragments:
            return {"work_items": []}

        async def search_one(fragment) -> FragmentWorkItem:
            if not fragment.embedding:
                return FragmentWorkItem(fragment=fragment)
            raw = await asyncio.to_thread(
                subjects_repo.search_candidate_subjects,
                fragment.embedding,
                ROUTE_CANDIDATES_TOP_K,
                ROUTE_CANDIDATES_MIN_SIMILARITY,
            )
            candidates = [
                CandidateSubject(subject=subj, current_claim=claim, similarity=sim)
                for subj, claim, sim in raw
            ]
            return FragmentWorkItem(fragment=fragment, candidates=candidates)

        work_items = await asyncio.gather(*(search_one(f) for f in state.fragments))
        return {"work_items": list(work_items)}

    return route_candidates


def make_classify_stance(
    llm: LLMClient,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
    batch_size: int = STANCE_BATCH_SIZE,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the classify_stance node.

    Reads:  state.work_items[*].fragment + .candidates
    Writes: state.work_items[*].votes

    Sends fragments in batches to reduce API call count. Items with no
    candidates are skipped. Votes below MIN_VOTE_STRENGTH are dropped.
    """

    @node_trace("classify_stance")
    async def classify_stance(state: ReflectionState) -> dict:
        items = state.work_items
        if not items:
            return {}

        sig = stance_model_signature(llm)
        system = SystemMessage(get_prompt(PromptKey.STANCE_CLASSIFIER_BATCH))
        structured_llm = llm.astructured(BatchStanceResponse)
        sem = asyncio.Semaphore(max_concurrency)

        claim_index: dict[str, str] = {
            c.subject.subject_id: c.current_claim.claim_id
            for item in items
            for c in item.candidates
        }
        frag_by_id = {item.fragment.fragment_id: item.fragment for item in items}

        active = [item for item in items if item.candidates]
        batches = [active[i:i + batch_size] for i in range(0, len(active), batch_size)]

        async def classify_batch(batch: list[FragmentWorkItem]) -> list[tuple[str, list]]:
            async with sem:
                payload = {
                    "items": [
                        {
                            "fragment_id": item.fragment.fragment_id,
                            "candidate_subjects": [
                                {"id": c.subject.subject_id, "current_claim": c.current_claim.text}
                                for c in item.candidates
                            ],
                            "fragment": {
                                "dated_at": item.fragment.timestamp.isoformat(),
                                "text": item.fragment.content,
                            },
                        }
                        for item in batch
                    ]
                }
                try:
                    response: BatchStanceResponse = await structured_llm.ainvoke(
                        [system, HumanMessage(content=json.dumps(payload))]
                    )
                    return [(r.fragment_id, r.votes) for r in response.results]
                except Exception:
                    logger.exception("classify_stance batch LLM call failed")
                    return [(item.fragment.fragment_id, []) for item in batch]

        batch_results = await asyncio.gather(*(classify_batch(b) for b in batches))

        votes_by_fragment: dict[str, list[Vote]] = {}
        for batch_result in batch_results:
            for fragment_id, stance_votes in batch_result:
                frag = frag_by_id.get(fragment_id)
                if frag is None:
                    continue
                votes_by_fragment[fragment_id] = [
                    Vote(
                        subject_id=sv.subject_id,
                        claim_id=claim_index.get(sv.subject_id, ""),
                        fragment_id=fragment_id,
                        stance=sv.stance,
                        strength=sv.strength,
                        reasoning=sv.reasoning,
                        fragment_dated_at=frag.timestamp,
                        model_signature=sig,
                    )
                    for sv in stance_votes
                    if sv.strength >= MIN_VOTE_STRENGTH
                ]

        updated = [
            item.model_copy(update={"votes": votes_by_fragment[item.fragment.fragment_id]})
            if item.fragment.fragment_id in votes_by_fragment
            else item
            for item in items
        ]
        return {"work_items": updated}

    return classify_stance


def make_propose_subject(
    llm: LLMClient,
    subjects_repo: SubjectsRepository,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the propose_subject node.

    Reads:  state.work_items[*] (where votes are weak/empty)
            + subjects_repo.list_active_subjects_with_claims() (loaded once
              for the node — the LLM needs to see ALL active subjects, not
              just the routed candidates, to avoid duplicate proposals)
    Writes: state.work_items[*].proposed_subject

    Per-item rather than batched: the proposer's "bias toward not creating"
    rule depends on independent per-fragment evaluation. Batching produced
    sympathetic-creation across items in the same prompt.
    """

    @node_trace("propose_subject")
    async def propose_subject(state: ReflectionState) -> dict:
        items = state.work_items
        if not items:
            return {}

        def needs_proposal(item: FragmentWorkItem) -> bool:
            if not item.votes:
                return True
            return max(v.strength for v in item.votes) < SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH

        needs = [(i, item) for i, item in enumerate(items) if needs_proposal(item)]
        if not needs:
            return {}

        active = await asyncio.to_thread(subjects_repo.list_active_subjects_with_claims)
        existing_summary = [
            {"label": subj.label, "current_claim": claim.text}
            for subj, claim in active
        ]

        system = SystemMessage(get_prompt(PromptKey.SUBJECT_PROPOSER))
        structured_llm = llm.astructured(ProposerResponse)
        sem = asyncio.Semaphore(max_concurrency)

        async def propose_one(item: FragmentWorkItem) -> FragmentWorkItem:
            async with sem:
                payload = {
                    "existing_subjects": existing_summary,
                    "fragment": {
                        "dated_at": item.fragment.timestamp.isoformat(),
                        "text": item.fragment.content,
                    },
                }
                try:
                    response: ProposerResponse = await structured_llm.ainvoke(
                        [system, HumanMessage(content=json.dumps(payload))]
                    )
                except Exception:
                    logger.exception(
                        "propose_subject LLM call failed for fragment %s",
                        item.fragment.fragment_id,
                    )
                    return item
                if response.new_subject is not None:
                    return item.model_copy(update={"proposed_subject": response.new_subject})
                return item

        results = await asyncio.gather(*(propose_one(item) for _, item in needs))

        updated = list(items)
        for (i, _), enriched in zip(needs, results):
            updated[i] = enriched
        return {"work_items": updated}

    return propose_subject


def make_persist_votes(
    subjects_repo: SubjectsRepository,
    llm: LLMClient,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the persist_votes node.

    Reads:  state.work_items (everything)
    Writes: DB only — subjects, claims, votes, fragment_processing rows

    Real behavior (TODO), all in one transaction:
        Step 1 — Dedupe proposed subjects across work items by embedding
            similarity > PROPOSER_DEDUP_SIMILARITY. Pick a canonical proposal;
            redirect followers' initial_votes to the canonical subject_id.
        Step 2 — For each canonical proposal: create_subject_with_claim(...)
            (creates Subject + Claim v1 + materializes initial_vote).
        Step 3 — Bulk insert votes against existing subjects (work_items[*].votes)
            plus any redirected follower initial_votes.
        Step 4 — Insert one fragment_processing row per work item with the
            model_signature and the resulting vote_count.

    Skeleton behavior:
        Set status to PROCESSING; do not touch the DB.
    """

    @node_trace("persist_votes")
    async def persist_votes(state: ReflectionState) -> dict:
        items = state.work_items
        if not items:
            return {"status": StatusValue.COMPLETED}

        sig = stance_model_signature(llm)

        proposals = [
            (i, item) for i, item in enumerate(items) if item.proposed_subject is not None
        ]

        canonical_map: dict[int, int] = {}
        canonical_created: dict[int, tuple] = {}

        if proposals:
            embs = [
                await asyncio.to_thread(
                    subjects_repo.embed_text,
                    f"{item.proposed_subject.label} {item.proposed_subject.initial_claim}",
                )
                for _, item in proposals
            ]

            assigned: set[int] = set()
            for pi in range(len(proposals)):
                if pi in assigned:
                    continue
                item_idx = proposals[pi][0]
                canonical_map[item_idx] = item_idx
                for pj in range(pi + 1, len(proposals)):
                    if pj in assigned:
                        continue
                    norm_i = np.linalg.norm(embs[pi])
                    norm_j = np.linalg.norm(embs[pj])
                    if norm_i == 0 or norm_j == 0:
                        continue
                    sim = float(np.dot(embs[pi], embs[pj]) / (norm_i * norm_j))
                    if sim >= PROPOSER_DEDUP_SIMILARITY:
                        canonical_map[proposals[pj][0]] = item_idx
                        assigned.add(pj)

            for pi, (item_idx, item) in enumerate(proposals):
                if item_idx not in canonical_map or canonical_map[item_idx] != item_idx:
                    continue
                fragment = item.fragment
                first_vote = Vote(
                    subject_id="placeholder",
                    claim_id="placeholder",
                    fragment_id=fragment.fragment_id,
                    stance=item.proposed_subject.initial_vote.stance,
                    strength=item.proposed_subject.initial_vote.strength,
                    reasoning=item.proposed_subject.initial_vote.reasoning,
                    fragment_dated_at=fragment.timestamp,
                    model_signature=sig,
                )
                try:
                    subject, claim, vote = await asyncio.to_thread(
                        subjects_repo.create_subject_with_claim,
                        item.proposed_subject,
                        first_vote,
                    )
                    canonical_created[item_idx] = (subject, claim, vote)
                except Exception:
                    logger.exception(
                        "create_subject_with_claim failed for fragment %s",
                        fragment.fragment_id,
                    )

        extra_votes: list[Vote] = []
        for i, item in proposals:
            canonical_idx = canonical_map.get(i, i)
            if canonical_idx == i:
                continue
            if canonical_idx not in canonical_created:
                continue
            canonical_subject, canonical_claim, _ = canonical_created[canonical_idx]
            extra_votes.append(Vote(
                subject_id=canonical_subject.subject_id,
                claim_id=canonical_claim.claim_id,
                fragment_id=item.fragment.fragment_id,
                stance=item.proposed_subject.initial_vote.stance,
                strength=item.proposed_subject.initial_vote.strength,
                reasoning=item.proposed_subject.initial_vote.reasoning,
                fragment_dated_at=item.fragment.timestamp,
                model_signature=sig,
            ))

        all_votes = [v for item in items for v in item.votes] + extra_votes
        if all_votes:
            await asyncio.to_thread(subjects_repo.insert_votes, all_votes)

        proposal_index = {i: item for i, item in proposals}
        for i, item in enumerate(items):
            vote_count = len(item.votes)
            if i in proposal_index:
                canonical_idx = canonical_map.get(i, i)
                if canonical_idx == i and i in canonical_created:
                    # Canonical: initial_vote was persisted by create_subject_with_claim.
                    vote_count += 1
                elif canonical_idx != i and canonical_idx in canonical_created:
                    # Follower: initial_vote was redirected into extra_votes and persisted.
                    vote_count += 1
            try:
                await asyncio.to_thread(
                    subjects_repo.record_processing,
                    item.fragment.fragment_id,
                    sig,
                    vote_count,
                )
            except Exception:
                logger.exception(
                    "record_processing failed for fragment %s", item.fragment.fragment_id
                )

        return {"status": StatusValue.COMPLETED}

    return persist_votes


# ── Cold-start path: cluster_seed_subjects ────────────────────────────────


def make_cluster_seed_subjects(
    llm: LLMClient,
    subjects_repo: SubjectsRepository,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the cluster_seed_subjects node (cold-start path).

    Reads:  state.fragments
    Writes: DB only — bulk-creates Subjects + Claims + initial Votes; sets
            state.status = COMPLETED so the graph exits without traversing
            the per-fragment path.

    Pipeline (single LLM cluster call + one LLM call per cluster + DB writes):
        1. Cluster all fragments via the CREATE_CLUSTERS prompt (LLM-based —
           we deliberately don't use HDBSCAN here because journal-text
           embeddings don't always separate distinct subjects cleanly).
        2. For each cluster, in parallel: invoke SEED_SUBJECT_FROM_CLUSTER
           → ProposedSubject (or null for incoherent clusters).
        3. For each non-null proposal, persist:
              - the first cluster member anchors create_subject_with_claim
                (which handles Subject + Claim + first Vote in one txn)
              - the rest of the cluster members are bulk-inserted as Votes
                pointing at the new subject_id, sharing the proposed
                initial_vote's stance/strength/reasoning.
              - every cluster member gets a fragment_processing row.
        4. For LLM-rejected clusters (proposed=null): still record
           fragment_processing rows with vote_count=0 so they don't loop
           back next run.
        5. Outliers (fragments not assigned to any cluster) are also
           recorded as processed with vote_count=0.

    Persistence failures: log and skip — affected fragments remain
    unprocessed and will be retried on the next reflect run.

    Design doc: design/phase11-claim-based-insights.md
    """

    @node_trace("cluster_seed_subjects")
    async def cluster_seed_subjects(state: ReflectionState) -> dict:
        try:
            fragments = state.fragments
            if not fragments:
                return {"status": StatusValue.COMPLETED}

            sig = stance_model_signature(llm)
            frag_by_id = {f.fragment_id: f for f in fragments}

            # Step 1 — cluster fragments via LLM
            cluster_payload = json.dumps([
                FragmentClusterRequest(
                    fragment_id=f.fragment_id,
                    content=f.content,
                    tags=[t.tag for t in f.tags],
                    timestamp=f.timestamp,
                ).model_dump(mode="json")
                for f in fragments
            ])
            cluster_system = SystemMessage(get_prompt(PromptKey.CREATE_CLUSTERS))
            cluster_runnable = llm.astructured(ClusterList)
            try:
                cluster_list: ClusterList = await cluster_runnable.ainvoke(
                    [cluster_system, HumanMessage(content=cluster_payload)]
                )
            except Exception:
                logger.exception("cluster_seed_subjects: clustering LLM call failed")
                return {"status": StatusValue.ERROR, "error_message": "clustering failed"}

            clusters = cluster_list.clusters
            logger.info("cluster_seed_subjects: %d fragments → %d clusters", len(fragments), len(clusters))

            # Step 2 — label each cluster (parallel, LLM-bounded)
            seed_system = SystemMessage(get_prompt(PromptKey.SEED_SUBJECT_FROM_CLUSTER))
            seed_runnable = llm.astructured(ProposerResponse)
            sem = asyncio.Semaphore(max_concurrency)

            async def label_cluster(cluster: Cluster) -> tuple[Cluster, ProposedSubject | None]:
                async with sem:
                    payload = {
                        "cluster": [
                            {
                                "fragment_id": fid,
                                "dated_at": frag_by_id[fid].timestamp.isoformat(),
                                "text": frag_by_id[fid].content,
                            }
                            for fid in cluster.fragment_ids if fid in frag_by_id
                        ]
                    }
                    if not payload["cluster"]:
                        return cluster, None
                    try:
                        response: ProposerResponse = await seed_runnable.ainvoke(
                            [seed_system, HumanMessage(content=json.dumps(payload))]
                        )
                        return cluster, response.new_subject
                    except Exception:
                        logger.exception("seed_subject_from_cluster LLM call failed")
                        return cluster, None

            labeled = await asyncio.gather(*(label_cluster(c) for c in clusters))

            # Step 3 — persist per cluster
            clustered_ids: set[str] = set()

            async def _safe_record(fid: str, vote_count: int) -> None:
                try:
                    await asyncio.to_thread(
                        subjects_repo.record_processing, fid, sig, vote_count
                    )
                except Exception:
                    logger.exception("record_processing failed for fragment %s", fid)

            for cluster, proposed in labeled:
                cluster_frag_ids = [fid for fid in cluster.fragment_ids if fid in frag_by_id]
                if not cluster_frag_ids:
                    continue

                if proposed is None:
                    # Cluster incoherent per LLM — drop but mark processed.
                    for fid in cluster_frag_ids:
                        await _safe_record(fid, 0)
                    clustered_ids.update(cluster_frag_ids)
                    continue

                anchor_fid = cluster_frag_ids[0]
                anchor_frag = frag_by_id[anchor_fid]
                first_vote = Vote(
                    subject_id="placeholder",  # rewritten by create_subject_with_claim
                    claim_id="placeholder",
                    fragment_id=anchor_fid,
                    stance=proposed.initial_vote.stance,
                    strength=proposed.initial_vote.strength,
                    reasoning=proposed.initial_vote.reasoning,
                    fragment_dated_at=anchor_frag.timestamp,
                    model_signature=sig,
                )
                try:
                    subject, claim, _ = await asyncio.to_thread(
                        subjects_repo.create_subject_with_claim,
                        proposed,
                        first_vote,
                    )
                except Exception:
                    logger.exception(
                        "create_subject_with_claim failed for cluster anchor %s — fragments stay unprocessed for retry",
                        anchor_fid,
                    )
                    continue

                rest_votes = [
                    Vote(
                        subject_id=subject.subject_id,
                        claim_id=claim.claim_id,
                        fragment_id=fid,
                        stance=proposed.initial_vote.stance,
                        strength=proposed.initial_vote.strength,
                        reasoning=proposed.initial_vote.reasoning,
                        fragment_dated_at=frag_by_id[fid].timestamp,
                        model_signature=sig,
                    )
                    for fid in cluster_frag_ids[1:]
                ]
                if rest_votes:
                    try:
                        await asyncio.to_thread(subjects_repo.insert_votes, rest_votes)
                    except Exception:
                        logger.exception("insert_votes failed for subject %s", subject.subject_id)

                for fid in cluster_frag_ids:
                    await _safe_record(fid, 1)
                clustered_ids.update(cluster_frag_ids)

            # Step 4 — outliers
            for f in fragments:
                if f.fragment_id not in clustered_ids:
                    await _safe_record(f.fragment_id, 0)

            return {"status": StatusValue.COMPLETED}

        except Exception as e:
            logger.exception("cluster_seed_subjects failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return cluster_seed_subjects


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
    if strategy == "simple_sum":
        return sum(
            v.strength if v.stance == Stance.SUPPORT else -v.strength
            for v in votes
        )
    raise NotImplementedError(f"compute_traction: unknown strategy {strategy!r}")
