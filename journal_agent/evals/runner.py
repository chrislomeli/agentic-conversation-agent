"""runner.py — Run the classifier eval suite against a fixture set.

Each classifier is invoked in isolation (not inside a LangGraph graph) using
the same ``make_*`` factory functions the real app uses.  Results are written
as JSONL — one ``EvalRecord`` per line — so they can be diffed between runs.

EOS classifiers are chained: exchange_decomposer feeds thread_classifier,
which feeds thread_fragment_extractor.  This mirrors the real EOS pipeline
order and means the fragment extractor sees realistic classifier output rather
than hand-crafted inputs.

Usage (from the CLI script)::

    results = asyncio.run(run_suite(fixtures, llm))
    write_results(results, output_path)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from journal_agent.configure.prompts import get_prompt_version
from journal_agent.evals.fixtures import Fixture, build_eos_state, build_intent_state, input_hash
from journal_agent.graph.nodes.classifiers import (
    make_exchange_decomposer,
    make_intent_classifier,
    make_thread_classifier,
    make_thread_fragment_extractor,
)
from journal_agent.model.session import PromptKey, StatusValue

logger = logging.getLogger(__name__)


# ── Output record ─────────────────────────────────────────────────────────────


class EvalRecord(BaseModel):
    fixture_id: str
    classifier: str
    prompt_key: str
    prompt_version: str
    input_hash: str
    output: dict[str, Any]
    elapsed_ms: int
    timestamp: str
    error: str | None = None


# ── Per-classifier runners ────────────────────────────────────────────────────


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _node_error(result: dict) -> str | None:
    """Return the error message if the node returned an ERROR status, else None."""
    if result.get("status") == StatusValue.ERROR:
        return result.get("error_message") or "unknown error"
    return None


def _run_intent_classifier(fixture: Fixture, llm: Any) -> EvalRecord:
    state = build_intent_state(fixture)
    h = input_hash(state.session_messages)
    node = make_intent_classifier(llm)
    t0 = perf_counter()
    try:
        result = node(state)
        error = _node_error(result)
        if error:
            output = {}
        else:
            spec = result.get("context_specification")
            output = spec.model_dump(mode="json") if spec else result
    except Exception as exc:
        output = {}
        error = str(exc)
        logger.error("intent_classifier failed for fixture %s: %s", fixture.fixture_id, exc)

    return EvalRecord(
        fixture_id=fixture.fixture_id,
        classifier="intent_classifier",
        prompt_key=PromptKey.INTENT_CLASSIFIER.value,
        prompt_version=get_prompt_version(PromptKey.INTENT_CLASSIFIER),
        input_hash=h,
        output=output,
        elapsed_ms=round((perf_counter() - t0) * 1000),
        timestamp=_now(),
        error=error,
    )


async def _run_eos_pipeline(fixture: Fixture, llm: Any) -> list[EvalRecord]:
    """Run the three EOS classifiers in sequence, threading outputs forward."""
    records: list[EvalRecord] = []
    state = build_eos_state(fixture)

    # ── exchange_decomposer (sync) ─────────────────────────────────────────
    decomposer = make_exchange_decomposer(llm)
    h = input_hash(state.transcript)
    t0 = perf_counter()
    try:
        result = decomposer(state)
        error = _node_error(result)
        if error:
            threads, output = [], {}
        else:
            threads = result.get("threads", [])
            output = {"threads": [t.model_dump(mode="json") for t in threads]}
    except Exception as exc:
        threads = []
        output = {}
        error = str(exc)
        logger.error("exchange_decomposer failed for fixture %s: %s", fixture.fixture_id, exc)

    records.append(EvalRecord(
        fixture_id=fixture.fixture_id,
        classifier="exchange_decomposer",
        prompt_key=PromptKey.DECOMPOSER.value,
        prompt_version=get_prompt_version(PromptKey.DECOMPOSER),
        input_hash=h,
        output=output,
        elapsed_ms=round((perf_counter() - t0) * 1000),
        timestamp=_now(),
        error=error,
    ))

    # ── thread_classifier (async) ──────────────────────────────────────────
    state = state.model_copy(update={"threads": threads})
    classifier = make_thread_classifier(llm)
    h = input_hash(state.threads + state.transcript)
    t0 = perf_counter()
    try:
        result = await classifier(state)
        error = _node_error(result)
        if error:
            classified, output = [], {}
        else:
            classified = result.get("classified_threads", [])
            output = {"classified_threads": [t.model_dump(mode="json") for t in classified]}
    except Exception as exc:
        classified = []
        output = {}
        error = str(exc)
        logger.error("thread_classifier failed for fixture %s: %s", fixture.fixture_id, exc)

    records.append(EvalRecord(
        fixture_id=fixture.fixture_id,
        classifier="thread_classifier",
        prompt_key=PromptKey.THREAD_CLASSIFIER.value,
        prompt_version=get_prompt_version(PromptKey.THREAD_CLASSIFIER),
        input_hash=h,
        output=output,
        elapsed_ms=round((perf_counter() - t0) * 1000),
        timestamp=_now(),
        error=error,
    ))

    # ── thread_fragment_extractor (async) ──────────────────────────────────
    state = state.model_copy(update={"classified_threads": classified})
    extractor = make_thread_fragment_extractor(llm)
    h = input_hash(state.classified_threads + state.transcript)
    t0 = perf_counter()
    try:
        result = await extractor(state)
        error = _node_error(result)
        if error:
            output = {}
        else:
            fragments = result.get("fragments", [])
            output = {"fragments": [f.model_dump(mode="json") for f in fragments]}
    except Exception as exc:
        output = {}
        error = str(exc)
        logger.error("thread_fragment_extractor failed for fixture %s: %s", fixture.fixture_id, exc)

    records.append(EvalRecord(
        fixture_id=fixture.fixture_id,
        classifier="thread_fragment_extractor",
        prompt_key=PromptKey.FRAGMENT_EXTRACTOR.value,
        prompt_version=get_prompt_version(PromptKey.FRAGMENT_EXTRACTOR),
        input_hash=h,
        output=output,
        elapsed_ms=round((perf_counter() - t0) * 1000),
        timestamp=_now(),
        error=error,
    ))

    return records


# ── Suite runner ──────────────────────────────────────────────────────────────


async def run_suite(fixtures: list[Fixture], llm: Any) -> list[EvalRecord]:
    """Run all classifiers against every fixture; return all EvalRecords."""
    all_records: list[EvalRecord] = []
    for fixture in fixtures:
        logger.info("Running fixture %r — %s", fixture.fixture_id, fixture.description)
        all_records.append(_run_intent_classifier(fixture, llm))
        all_records.extend(await _run_eos_pipeline(fixture, llm))
    return all_records


def write_results(records: list[EvalRecord], path: Path) -> None:
    """Write one EvalRecord per line to *path* (JSONL)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(record.model_dump_json() + "\n")
    logger.info("Wrote %d records to %s", len(records), path)


def load_results(path: Path) -> list[EvalRecord]:
    """Load EvalRecords from a JSONL file produced by write_results."""
    records: list[EvalRecord] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(EvalRecord.model_validate_json(line))
    return records
