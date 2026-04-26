"""fixtures.py — Load eval fixtures and build JournalState inputs for each classifier.

A fixture is a small, deterministic session — a list of Exchange objects that
the eval harness uses as input.  The same fixture can be fed to every
classifier in the suite, so results across classifiers are comparable.

Fixtures live in ``data/eval_fixtures.jsonl``; each line is one fixture.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

from journal_agent.graph.state import JournalState
from journal_agent.model.session import Exchange

FIXTURE_PATH = Path(__file__).parent / "data" / "eval_fixtures.jsonl"


@dataclass
class Fixture:
    fixture_id: str
    description: str
    exchanges: list[Exchange]


def load_fixtures(path: Path = FIXTURE_PATH) -> list[Fixture]:
    """Load all fixtures from a JSONL file."""
    fixtures: list[Fixture] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            exchanges = [Exchange.model_validate(e) for e in data["exchanges"]]
            fixtures.append(Fixture(
                fixture_id=data["fixture_id"],
                description=data["description"],
                exchanges=exchanges,
            ))
    return fixtures


def build_intent_state(fixture: Fixture) -> JournalState:
    """Build JournalState suitable for intent_classifier.

    intent_classifier reads ``state.session_messages`` (recent Human/AI
    messages).  We flatten the fixture exchanges into that list.
    """
    messages: list[BaseMessage] = []
    for ex in fixture.exchanges:
        if ex.human:
            messages.append(HumanMessage(content=ex.human.content))
        if ex.ai:
            messages.append(AIMessage(content=ex.ai.content))
    return JournalState(session_id=fixture.fixture_id, session_messages=messages)


def build_eos_state(fixture: Fixture) -> JournalState:
    """Build JournalState suitable for the EOS pipeline.

    exchange_decomposer, thread_classifier, and thread_fragment_extractor all
    read ``state.transcript``.  The subsequent classifiers also read
    ``state.threads`` / ``state.classified_threads``, which the runner fills in
    by carrying forward each step's output.
    """
    return JournalState(session_id=fixture.fixture_id, transcript=fixture.exchanges)


def input_hash(items: list[Any]) -> str:
    """Stable 8-char SHA-256 fingerprint of *items*.

    Used to detect whether the same logical input produced different outputs
    across runs, independent of timestamps or other volatile fields.
    """
    def _serialise(x: Any) -> Any:
        if hasattr(x, "model_dump"):
            return x.model_dump(mode="json")
        if hasattr(x, "content"):
            return {"type": type(x).__name__, "content": x.content}
        return str(x)

    raw = json.dumps([_serialise(x) for x in items], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:8]
