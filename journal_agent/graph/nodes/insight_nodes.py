import asyncio
import json
from collections import defaultdict
from typing import Callable, Coroutine, Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import HDBSCAN

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.config_builder import MINIMUM_VERIFIER_SCORE, MINIMUM_CLUSTER_LABEL_SCORE, \
    MINIMUM_CLUSTER_SCORE
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace, logger
from journal_agent.graph.nodes.classifiers import DEFAULT_LLM_CONCURRENCY
from journal_agent.graph.state import ReflectionState
from journal_agent.model.session import Cluster, Fragment, StatusValue, Insight, PromptKey, InsightDraft, \
    InsightVerifierScore, VerifierStatus


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


def make_cluster_fragments() -> Callable[..., dict]:
    @node_trace("cluster_fragments")
    def cluster_fragments(state: ReflectionState) -> dict:
        try:
            fragments = state.fragments
            if not fragments:
                return {"clusters": []}

            vectors = np.vstack([f.embedding for f in fragments])

            hdb = HDBSCAN(min_cluster_size=3)
            hdb.fit(vectors)

            # Group fragments by label; -1 = noise, skip those
            groups: dict[int, list] = defaultdict(list)
            for fragment, label in zip(fragments, hdb.labels_):
                if label != -1:
                    groups[label].append(fragment)

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

    return cluster_fragments


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
                        label_confidence=draft.confidence,
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

            print("\nVerified insights:\n")
            for insight in verified_insights:
                print(json.dumps(insight.model_dump(), indent=2, default=str))

            return {"verified_insights": verified_insights}

        except Exception as e:
            logger.exception("verify_citations failed")
            return {"status": StatusValue.ERROR, "error_message": str(e)}

    return verify_citations



