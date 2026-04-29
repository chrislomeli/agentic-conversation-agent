import asyncio
import json
from collections import defaultdict
from typing import Callable, Coroutine, Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import HDBSCAN

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure import context_builder
from journal_agent.configure.config_builder import (
    MINIMUM_VERIFIER_SCORE,
    MINIMUM_CLUSTER_LABEL_SCORE,
    MINIMUM_CLUSTER_SCORE,
    MIN_VOTE_STRENGTH,
    ROUTE_CANDIDATES_TOP_K,
    ROUTE_CANDIDATES_MIN_SIMILARITY,
    SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH,
    PROPOSER_DEDUP_SIMILARITY,
)
from journal_agent.configure.prompts import get_prompt, get_prompt_version
from journal_agent.graph.node_tracer import node_trace, logger
from journal_agent.graph.nodes.classifiers import DEFAULT_LLM_CONCURRENCY
from journal_agent.graph.state import ReflectionState
from journal_agent.model.insights import (
    CandidateSubject,
    FragmentWorkItem,
    ProposedSubject,
    ProposerResponse,
    Stance,
    StanceResponse,
    Subject,
    Vote,
)
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
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the classify_stance node.

    Reads:  state.work_items[*].fragment + .candidates
    Writes: state.work_items[*].votes

    Real behavior (TODO):
        For each work item with non-empty candidates (in parallel, bounded
        by max_concurrency), call the stance_classifier prompt with
        (fragment, candidates). Parse StanceResponse. For each returned
        StanceVote: convert to a real Vote — filling fragment_id,
        fragment_dated_at (= fragment.timestamp), processed_at,
        model_signature, and claim_version_at_vote (from the matching
        candidate's current_claim.version). Drop votes below
        MIN_VOTE_STRENGTH (defense in depth — the prompt also instructs this).
        Items with no candidates skip the LLM call entirely.

    Skeleton behavior:
        Pass through work_items unchanged.
    """

    @node_trace("classify_stance")
    async def classify_stance(state: ReflectionState) -> dict:
        items = state.work_items
        if not items:
            return {}

        sig = f"{llm.model}/stance_classifier/{get_prompt_version(PromptKey.STANCE_CLASSIFIER)}"
        system = SystemMessage(get_prompt(PromptKey.STANCE_CLASSIFIER))
        structured_llm = llm.astructured(StanceResponse)
        sem = asyncio.Semaphore(max_concurrency)

        claim_index: dict[str, str] = {
            c.subject.subject_id: c.current_claim.claim_id
            for item in items
            for c in item.candidates
        }

        async def classify_item(item: FragmentWorkItem) -> FragmentWorkItem:
            if not item.candidates:
                return item
            async with sem:
                payload = {
                    "candidate_subjects": [
                        {"id": c.subject.subject_id, "current_claim": c.current_claim.text}
                        for c in item.candidates
                    ],
                    "fragment": {
                        "dated_at": item.fragment.timestamp.isoformat(),
                        "text": item.fragment.content,
                    },
                }
                try:
                    response: StanceResponse = await structured_llm.ainvoke(
                        [system, HumanMessage(content=json.dumps(payload))]
                    )
                except Exception:
                    logger.exception("classify_stance LLM call failed for fragment %s", item.fragment.fragment_id)
                    return item

                votes = [
                    Vote(
                        subject_id=sv.subject_id,
                        claim_id=claim_index.get(sv.subject_id, ""),
                        fragment_id=item.fragment.fragment_id,
                        stance=sv.stance,
                        strength=sv.strength,
                        reasoning=sv.reasoning,
                        fragment_dated_at=item.fragment.timestamp,
                        model_signature=sig,
                    )
                    for sv in response.votes
                    if sv.strength >= MIN_VOTE_STRENGTH
                ]
                return item.model_copy(update={"votes": votes})

        updated = await asyncio.gather(*(classify_item(item) for item in items))
        return {"work_items": list(updated)}

    return classify_stance


def make_propose_subject(
    llm: LLMClient,
    subjects_repo: SubjectsRepository,
    max_concurrency: int = DEFAULT_LLM_CONCURRENCY,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the propose_subject node.

    Reads:  state.work_items[*] (where votes are weak/empty)
            + subjects_repo.list_active_subjects() (loaded once for the node,
              not per item — the LLM needs to see ALL active subjects, not
              just the routed candidates, to avoid duplicate proposals)
    Writes: state.work_items[*].proposed_subject

    Real behavior (TODO):
        Filter work items where max(vote.strength) < SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH
        (or votes is empty). For those, in parallel: call the subject_proposer
        prompt. If response.new_subject is non-null, set
        item.proposed_subject = response.new_subject. The bundled initial_vote
        rides inside proposed_subject and gets materialized at persist time.

    Skeleton behavior:
        Pass through work_items unchanged.
    """

    @node_trace("propose_subject")
    async def propose_subject(state: ReflectionState) -> dict:
        items = state.work_items
        if not items:
            return {}

        active = await asyncio.to_thread(subjects_repo.list_active_subjects_with_claims)
        existing_summary = [
            {"label": subj.label, "current_claim": claim.text}
            for subj, claim in active
        ]

        system = SystemMessage(get_prompt(PromptKey.SUBJECT_PROPOSER))
        structured_llm = llm.astructured(ProposerResponse)
        sem = asyncio.Semaphore(max_concurrency)

        def needs_proposal(item: FragmentWorkItem) -> bool:
            if not item.votes:
                return True
            return max(v.strength for v in item.votes) < SUBJECT_PROPOSER_TRIGGER_MAX_STRENGTH

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
                    logger.exception("propose_subject LLM call failed for fragment %s", item.fragment.fragment_id)
                    return item
                if response.new_subject is not None:
                    return item.model_copy(update={"proposed_subject": response.new_subject})
                return item

        needs = [(i, item) for i, item in enumerate(items) if needs_proposal(item)]
        if not needs:
            return {}

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

        sig = f"{llm.model}/stance_classifier/{get_prompt_version(PromptKey.STANCE_CLASSIFIER)}"

        proposals = [
            (i, item) for i, item in enumerate(items) if item.proposed_subject is not None
        ]

        canonical_map: dict[int, int] = {}
        canonical_created: dict[int, tuple] = {}

        if proposals:
            embedder = subjects_repo._embedder
            embs = [
                await asyncio.to_thread(
                    embedder.embed,
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

        for i, item in enumerate(items):
            vote_count = len(item.votes)
            if item.proposed_subject is not None:
                canonical_idx = canonical_map.get(i, i)
                if canonical_idx == i and i in canonical_created:
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
