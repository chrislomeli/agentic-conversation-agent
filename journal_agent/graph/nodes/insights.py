from collections import defaultdict
from typing import Callable

import numpy as np
from sklearn.cluster import HDBSCAN

from journal_agent.graph.node_tracer import node_trace, logger
from journal_agent.graph.state import ReflectionState
from journal_agent.model.session import Cluster, Status
from journal_agent.stores import PgFragmentRepository

def make_collect_window(
    fragment_store: PgFragmentRepository,
) -> Callable[..., dict]:

    @node_trace("generate_insights")
    def collect_window(state: ReflectionState) -> dict:
        try:
            all_fragments = fragment_store.load_all()
            return {"fragments" : all_fragments}
        except Exception as e:
            logger.exception("Insight generation failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return collect_window




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

            return {"clusters": clusters}
        except Exception as e:
            logger.exception("Cluster fragments failed")
            return {"status": Status.ERROR, "error_message": str(e)}

    return cluster_fragments