import asyncio
import json
from collections import defaultdict
from typing import Callable, Coroutine, Any

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from sklearn.cluster import HDBSCAN

from journal_agent.comms.llm_client import LLMClient
from journal_agent.configure.prompts import get_prompt
from journal_agent.graph.node_tracer import node_trace, logger
from journal_agent.graph.nodes.classifiers import DEFAULT_LLM_CONCURRENCY
from journal_agent.graph.state import ReflectionState
from journal_agent.model.session import Cluster, Fragment, Status, Insight, PromptKey, InsightDraft
from journal_agent.stores import PgFragmentRepository


def make_collect_window(
        fragment_store: PgFragmentRepository,
) -> Callable[..., dict]:
    @node_trace("generate_insights")
    def collect_window(state: ReflectionState) -> dict:
        try:
            all_fragments = fragment_store.load_all()
            return {"fragments": all_fragments}
        except Exception as e:
            logger.exception("Insight generation failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return collect_window


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
            fragments = state["fragments"]
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

            return {"clusters": clusters}
        except Exception as e:
            logger.exception("Cluster fragments failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return cluster_fragments


def make_label_clusters(llm: LLMClient, max_concurrency: int = DEFAULT_LLM_CONCURRENCY) -> Callable[
    ..., Coroutine[Any, Any, dict]]:

    @node_trace("label_clusters")
    async def label_clusters(state: ReflectionState) -> dict:
        try:
            clusters = state["clusters"]
            if not clusters:
                return {"insights": []}

            frag_by_id = {f.fragment_id: f for f in state["fragments"]}

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
                    draft: InsightDraft = await structured_llm.ainvoke([system, human])
                    return Insight(
                        label=draft.label,
                        body=draft.body,
                        confidence=draft.confidence,
                        fragment_ids=cluster.fragment_ids,
                    )

            insights = await asyncio.gather(
                *(label_cluster(c) for c in clusters)
            )

            return {"insights": list(insights)}

        except Exception as e:
            logger.exception("label_clusters failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return label_clusters
