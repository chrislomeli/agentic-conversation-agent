"""stores — All persistence classes, re-exported for convenience.

Import from here:  ``from journal_agent.stores import TranscriptRepository``
"""

from journal_agent.stores.transcript_repo import TranscriptRepository
from journal_agent.stores.threads_repo import ThreadsRepository
from journal_agent.stores.profile_repo import UserProfileRepository
from journal_agent.stores.insights_repo import InsightsRepository
from journal_agent.stores.fragment_repo import FragmentRepository
from journal_agent.stores.transcript_cache import TranscriptStore
from journal_agent.stores.jsonl_gateway import JsonlGateway
from journal_agent.stores.pg_gateway import PgGateway, get_pg_gateway
from journal_agent.stores.embedder import Embedder
from journal_agent.stores.utils import exchanges_to_messages, resolve_project_root
from journal_agent.stores.checkpointer import make_postgres_checkpointer

__all__ = [
    "TranscriptRepository",
    "ThreadsRepository",
    "UserProfileRepository",
    "InsightsRepository",
    "FragmentRepository",
    "TranscriptStore",
    "JsonlGateway",
    "PgGateway",
    "get_pg_gateway",
    "Embedder",
    "exchanges_to_messages",
    "resolve_project_root",
    "make_postgres_checkpointer",
]