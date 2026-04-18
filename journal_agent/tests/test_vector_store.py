"""Layer 7 tests — VectorStore (storage/vector_store.py).

chromadb is an optional runtime dependency (not installed in this env).
We stub sys.modules["chromadb"] before importing VectorStore so the class
can be loaded and its static/pure-logic methods tested.  Tests that
exercise the collection API use a MagicMock collection injected via the
vector_store fixture — no real embedding model is ever invoked.

If chromadb is installed, the stub is bypassed and the tests run against
the real library.
"""

import importlib.util
import json
import sys
from datetime import datetime
from unittest.mock import MagicMock

import pytest

# ── chromadb stub (no-op when chromadb is already installed) ─────────────────
_chromadb_available = "chromadb" in sys.modules or importlib.util.find_spec("chromadb") is not None
if not _chromadb_available:
    _stub = MagicMock()
    _stub.PersistentClient = MagicMock()
    sys.modules["chromadb"] = _stub

import chromadb  # noqa: E402 — must come after stub injection
from journal_agent.model.session import Fragment, Tag
from journal_agent.storage.vector_store import VectorStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fragment(
    content: str = "some thought",
    session_id: str = "s1",
    tags: tuple[str, ...] = ("philosophy",),
) -> Fragment:
    return Fragment(
        session_id=session_id,
        content=content,
        exchange_ids=["e1"],
        tags=[Tag(tag=t) for t in tags],
        timestamp=datetime.now(),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_collection():
    m = MagicMock()
    m.metadata = {}  # empty metadata → _select_relevance_fn uses l2 formula
    return m


@pytest.fixture
def vector_store(tmp_path, monkeypatch, mock_collection):
    monkeypatch.setattr(
        "journal_agent.storage.vector_store.resolve_project_root",
        lambda: tmp_path,
    )
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    monkeypatch.setattr(chromadb, "PersistentClient", lambda path: mock_client)
    return VectorStore()


# ═══════════════════════════════════════════════════════════════════════════════
# Static serialization — fragment_to_chroma / fragment_from_chroma
# ═══════════════════════════════════════════════════════════════════════════════

class TestFragmentSerialization:
    def test_to_chroma_produces_required_keys(self):
        row = VectorStore.fragment_to_chroma(make_fragment())
        assert {"id", "document", "metadata"} <= row.keys()

    def test_to_chroma_id_matches_fragment_id(self):
        f = make_fragment()
        assert VectorStore.fragment_to_chroma(f)["id"] == f.fragment_id

    def test_to_chroma_document_is_content(self):
        f = make_fragment(content="deep thought")
        assert VectorStore.fragment_to_chroma(f)["document"] == "deep thought"

    def test_to_chroma_tags_are_json_serialized_in_metadata(self):
        f = make_fragment(tags=("philosophy", "creativity"))
        row = VectorStore.fragment_to_chroma(f)
        tag_names = [t["tag"] for t in json.loads(row["metadata"]["tags"])]
        assert "philosophy" in tag_names
        assert "creativity" in tag_names

    def test_round_trip_preserves_id_and_content(self):
        f = make_fragment(content="original thought")
        restored = VectorStore.fragment_from_chroma(VectorStore.fragment_to_chroma(f))
        assert restored.fragment_id == f.fragment_id
        assert restored.content == f.content
        assert restored.session_id == f.session_id

    def test_round_trip_preserves_tags(self):
        f = make_fragment(tags=("philosophy",))
        restored = VectorStore.fragment_from_chroma(VectorStore.fragment_to_chroma(f))
        assert len(restored.tags) == 1
        assert restored.tags[0].tag == "philosophy"

    def test_round_trip_handles_empty_tags(self):
        f = make_fragment(tags=())
        restored = VectorStore.fragment_from_chroma(VectorStore.fragment_to_chroma(f))
        assert restored.tags == []

    def test_round_trip_handles_empty_exchange_ids(self):
        f = Fragment(session_id="s1", content="x", exchange_ids=[], tags=[], timestamp=datetime.now())
        restored = VectorStore.fragment_from_chroma(VectorStore.fragment_to_chroma(f))
        assert restored.exchange_ids == []

    def test_round_trip_preserves_timestamp(self):
        ts = datetime(2024, 6, 1, 12, 0, 0)
        f = Fragment(session_id="s1", content="x", exchange_ids=[], tags=[], timestamp=ts)
        restored = VectorStore.fragment_from_chroma(VectorStore.fragment_to_chroma(f))
        assert restored.timestamp == ts


# ═══════════════════════════════════════════════════════════════════════════════
# Collection operations (mocked collection — no real embeddings)
# ═══════════════════════════════════════════════════════════════════════════════

class TestVectorStoreCollectionOps:
    def test_add_fragments_calls_collection_add(self, vector_store, mock_collection):
        vector_store.add_to_chroma_from_fragments([make_fragment("a"), make_fragment("b")])
        mock_collection.add.assert_called_once()
        _, kwargs = mock_collection.add.call_args
        assert len(kwargs["ids"]) == 2
        assert len(kwargs["documents"]) == 2
        assert len(kwargs["metadatas"]) == 2

    def test_add_fragments_passes_correct_ids(self, vector_store, mock_collection):
        f = make_fragment("thought")
        vector_store.add_to_chroma_from_fragments([f])
        _, kwargs = mock_collection.add.call_args
        assert kwargs["ids"][0] == f.fragment_id

    def test_search_returns_matches_above_min_relevance(self, vector_store, mock_collection):
        f = make_fragment("philosophy")
        row = VectorStore.fragment_to_chroma(f)
        # L2 distance 0.2 → relevance = max(0, 1 - 0.2/2.0) = 0.9
        mock_collection.query.return_value = {
            "ids": [[row["id"]]],
            "distances": [[0.2]],
            "documents": [[row["document"]]],
            "metadatas": [[row["metadata"]]],
        }
        results = vector_store.search_fragments("philosophy", min_relevance=0.5)
        assert len(results) == 1
        fragment, relevance = results[0]
        assert fragment.fragment_id == f.fragment_id
        assert relevance > 0.5

    def test_search_filters_out_low_relevance_matches(self, vector_store, mock_collection):
        f = make_fragment("unrelated")
        row = VectorStore.fragment_to_chroma(f)
        # L2 distance 1.9 → relevance = max(0, 1 - 1.9/2.0) = 0.05
        mock_collection.query.return_value = {
            "ids": [[row["id"]]],
            "distances": [[1.9]],
            "documents": [[row["document"]]],
            "metadatas": [[row["metadata"]]],
        }
        assert vector_store.search_fragments("query", min_relevance=0.3) == []

    def test_search_returns_empty_list_on_collection_exception(
        self, vector_store, mock_collection
    ):
        mock_collection.query.side_effect = RuntimeError("chroma error")
        assert vector_store.search_fragments("query") == []

    def test_truncate_collection_deletes_then_recreates(self, vector_store, mock_collection):
        client = vector_store.client
        vector_store.truncate_collection()
        client.delete_collection.assert_called_once_with(name=VectorStore.collection_name)
        client.create_collection.assert_called_once_with(name=VectorStore.collection_name)
