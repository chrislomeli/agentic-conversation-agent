"""eos_pipeline.py — Single-node end-of-session ETL pipeline.

The 7 EOS phases run as sequential plain async calls inside one LangGraph
node.  Each phase is produced by the same factory it would have used as a
standalone graph node, so ``node_trace`` observability at each phase
boundary is preserved.

State threading
---------------
``JournalState`` fields like ``threads`` and ``classified_threads`` carry
LangGraph ``add`` reducers — those reducers run at the graph boundary, not
inside a node.  Inside the pipeline, each phase must see the results of the
previous phase, so we thread state forward with ``model_copy(update=...)``
between calls.  The single accumulated dict returned at the end is what
LangGraph applies reducers to.  This is correct because all three reducer
fields (``threads``, ``classified_threads``) start empty when the EOS graph
runs, so append-from-empty is equivalent to assignment.

Error handling
--------------
If a phase returns a ``status=ERROR`` dict, the pipeline logs the failure
and returns immediately.  Downstream phases are skipped — they would
operate on incomplete state and produce corrupt artifacts.
"""

import inspect
import logging
from collections.abc import Callable, Coroutine
from typing import Any

from journal_agent.comms.llm_client import LLMClient
from journal_agent.graph.node_tracer import node_trace
from journal_agent.graph.nodes.classifiers import (
    make_exchange_decomposer,
    make_thread_classifier,
    make_thread_fragment_extractor,
)
from journal_agent.graph.nodes.stores import (
    make_save_classified_threads,
    make_save_fragments,
    make_save_threads,
    make_save_transcript,
)
from journal_agent.graph.state import JournalState
from journal_agent.model.session import StatusValue
from journal_agent.stores import (
    PgFragmentRepository,
    ThreadsRepository,
    TranscriptRepository,
)

logger = logging.getLogger(__name__)


async def _call(fn: Callable, state: JournalState) -> dict:
    """Invoke a sync or async phase function uniformly."""
    result = fn(state)
    if inspect.isawaitable(result):
        return await result
    return result


def make_end_of_session_node(
    *,
    transcript_store: TranscriptRepository,
    thread_store: ThreadsRepository,
    classified_thread_store: ThreadsRepository,
    fragment_store: PgFragmentRepository,
    classifier_llm: LLMClient,
    extractor_llm: LLMClient,
) -> Callable[..., Coroutine[Any, Any, dict]]:
    """Build the single end-of-session pipeline node.

    Runs 7 phases sequentially inside one graph super-step:

        1. save_transcript           — persist raw Exchange transcript
        2. exchange_decomposer       — split transcript into ThreadSegments
        3. save_threads              — persist ThreadSegments
        4. thread_classifier         — assign taxonomy tags to each thread
        5. save_classified_threads   — persist tagged ThreadSegments
        6. thread_fragment_extractor — distill threads into searchable Fragments
        7. save_fragments            — persist + vector-index Fragments

    Each phase function is built from the same ``make_*`` factory it would
    have used as a standalone graph node.  ``node_trace`` tracing fires per
    phase (not just at the outer node boundary) so per-phase latency and
    error signals are still observable.
    """
    phases: list[tuple[str, Callable]] = [
        ("save_transcript",           make_save_transcript(transcript_store)),
        ("exchange_decomposer",       make_exchange_decomposer(classifier_llm)),
        ("save_threads",              make_save_threads(thread_store)),
        ("thread_classifier",         make_thread_classifier(classifier_llm)),
        ("save_classified_threads",   make_save_classified_threads(classified_thread_store)),
        ("thread_fragment_extractor", make_thread_fragment_extractor(extractor_llm)),
        ("save_fragments",            make_save_fragments(fragment_store)),
    ]

    @node_trace("end_of_session")
    async def end_of_session(state: JournalState) -> dict:
        """Execute all EOS phases in sequence, threading state forward."""
        current = state
        accumulated: dict = {}

        for phase_name, phase_fn in phases:
            logger.info("EOS phase: %s (session=%s)", phase_name, state.session_id)
            result = await _call(phase_fn, current)

            if result.get("status") == StatusValue.ERROR:
                logger.error(
                    "EOS phase %s failed (session=%s): %s",
                    phase_name,
                    state.session_id,
                    result.get("error_message", "unknown error"),
                )
                return result  # bail — downstream phases need clean state

            accumulated.update(result)
            # Thread state forward: next phase sees this phase's updates.
            # model_copy replaces fields; reducers (add, add_messages) are
            # applied by LangGraph when the node returns its final dict.
            current = current.model_copy(update=accumulated)

        return accumulated

    return end_of_session
