"""repository — All persistence classes, re-exported for convenience.

Import from here:  ``from journal_agent.repository import TranscriptRepository``
"""

from journal_agent.repository.transcript_repo import TranscriptRepository
from journal_agent.repository.threads_repo import ThreadsRepository
from journal_agent.repository.profile_repo import UserProfileRepository
from journal_agent.repository.insights_repo import InsightsRepository
from journal_agent.repository.fragment_repo import PgFragmentRepository
from journal_agent.repository.transcript_cache import TranscriptStore
from journal_agent.repository.jsonl_gateway import JsonlGateway
from journal_agent.repository.pg_gateway import PgGateway, get_pg_gateway
from journal_agent.repository.embedder import Embedder
from journal_agent.repository.utils import exchanges_to_messages, resolve_project_root

__all__ = [
    "TranscriptRepository",
    "ThreadsRepository",
    "UserProfileRepository",
    "InsightsRepository",
    "PgFragmentRepository",
    "TranscriptStore",
    "JsonlGateway",
    "PgGateway",
    "get_pg_gateway",
    "Embedder",
    "exchanges_to_messages",
    "resolve_project_root",
]