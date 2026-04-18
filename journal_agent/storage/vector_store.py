import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

import chromadb

from journal_agent.model.session import Fragment, Tag
from journal_agent.storage.utils import resolve_project_root

logger = logging.getLogger(__name__)

# Clamp raw L2 distances above this value to relevance 0.
# With Chroma's default embeddings, real queries rarely exceed 2.0.
_L2_MAX_USEFUL = 2.0


def _relevance_from_l2(distance: float) -> float:
    return max(0.0, 1.0 - distance / _L2_MAX_USEFUL)


def _relevance_from_cosine(distance: float) -> float:
    # Chroma cosine distance ∈ [0, 2]: 0 = identical, 2 = opposite.
    return max(0.0, 1.0 - distance / 2.0)

"""

  - Architecture: inject (match the LLMClient pattern). Not state, not inside TranscriptStore.
  - Code: nearly complete. Add an add_fragments method so the live pipeline can push new fragments after each session.
  - Wire it into both make_retrieve_fragments (new, for the read path — Phase 6) and make_save_fragments (update, add vector write alongside JSON write)
"""

class VectorStore:

    database_name: str = "chroma_db"
    collection_name = "journal"
    client: chromadb.ClientAPI | None = None

    def __init__(self):
        # Get the path
        self._path = resolve_project_root() / "data" / "vector_store"
        if not self._path.exists():
            self._path.mkdir(parents=True, exist_ok=True)

        path = self._path / self.database_name
        self.client = chromadb.PersistentClient(path=path)

        # Create/get the collection (this is your "table")
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

        self._to_relevance = self._select_relevance_fn()

    def _select_relevance_fn(self) -> Callable[[float], float]:
        metadata = self.collection.metadata or {}
        space = metadata.get("hnsw:space", "l2")
        if space == "cosine":
            return _relevance_from_cosine
        if space == "l2":
            return _relevance_from_l2
        # ip ("inner product") and any future metrics: not implemented.
        # Fall back to L2 with a warning so callers still get a 0–1 value.
        logger.warning("Unknown hnsw:space=%r; falling back to L2 normalization.", space)
        return _relevance_from_l2

    def truncate_collection(self):
        # Delete the collection
        self.client.delete_collection(name=self.collection_name)
        # Recreate the collection
        self.collection = self.client.create_collection(name=self.collection_name)

    def rebuild_chroma_from_json(self, full_path: Path):
        # Assuming 'collection' is your collection object
        self.truncate_collection()
        self.add_to_chroma_from_json(full_path)

    def add_to_chroma_from_json(self, full_path: Path):
        # Assuming 'collection' is your collection object
        for json_file in full_path.glob("*.json"):
            objects = json.loads(json_file.read_text())
            _fragments: list[Fragment] = [Fragment(**f) for f in objects]
            self.add_to_chroma_from_fragments(_fragments=_fragments)

    def add_to_chroma_from_fragments(self, _fragments: list[Fragment]):
        # Assuming 'collection' is your collection object
            ids, docs, metas = [], [], []
            for f in _fragments:
                d = self.fragment_to_chroma(f)
                ids.append(d["id"])
                docs.append(d["document"]  + "  TAGS: " + d["metadata"]["tags"])
                metas.append(d["metadata"])
            self.collection.add(ids=ids, documents=docs, metadatas=metas)


    def search_fragments(
        self,
        query_text: str,
        min_relevance: float = 0.3,
        top_k: int = 5,
    ) -> list[tuple[Fragment, float]]:
        matches: list[tuple[Fragment, float]] = []
        try:
            results = self.collection.query(
                query_texts=[query_text],  # ← human's raw message
                n_results=top_k
            )
            result_set = 0
            rows_count = len(results["ids"][result_set])
            for row in range(rows_count):
                distance = results["distances"][result_set][row]
                relevance = self._to_relevance(distance)
                if relevance < min_relevance:
                    continue
                id = results["ids"][result_set][row]
                document = results["documents"][result_set][row]
                metadata = results["metadatas"][result_set][row]

                fragment = VectorStore.fragment_from_chroma({
                    "id": id,
                    "document": document,
                    "metadata": metadata
                })
                matches.append((fragment, relevance))

            return matches
        except Exception as e:
            logger.exception("Error searching fragments: %s", e)
            return []

    @staticmethod
    def fragment_to_chroma(f: Fragment) -> dict:
        return {
            "id": f.fragment_id,
            "document": f.content,
            "metadata": {
                "session_id": f.session_id,
                "exchange_ids": ",".join(f.exchange_ids),
                "tags": json.dumps([t.model_dump() for t in f.tags]),
                "timestamp": f.timestamp.isoformat(),
            }
        }

    @staticmethod
    def fragment_from_chroma(row: dict) -> Fragment:
        meta = row["metadata"]
        return Fragment(
            fragment_id=row["id"],
            content=row["document"],
            session_id=meta["session_id"],
            exchange_ids=meta["exchange_ids"].split(",") if meta["exchange_ids"] else [],
            tags=[Tag(**t) for t in json.loads(meta["tags"])] if meta["tags"] else [],
            timestamp=datetime.fromisoformat(meta["timestamp"]),
        )


def get_vector_store(rebuild: bool = False):
    vs = VectorStore()
    data_folder = resolve_project_root() / "data" / "test"
    if rebuild:
        vs.rebuild_chroma_from_json(data_folder)
    return vs


if __name__ == "__main__":
    v = get_vector_store(rebuild=True)
    matches = v.search_fragments("lets discuss humanity")

    history = json.dumps([{
        "relevance": round(rel, 3),
        "content": f.content,
        "tag": [t.tag for t in f.tags]
    } for f, rel in matches])

    print(history)
    print(matches)