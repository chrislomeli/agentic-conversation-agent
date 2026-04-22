"""
main_replay.py — Replay a saved transcript through the classifier and
extractor nodes without running the full conversation graph.

Use this to iterate on prompts and models: change a prompt, re-run
against the same transcript, diff the outputs between runs.

Usage:
    python -m journal_agent.main_replay [transcript_path]

If transcript_path is omitted, the newest file in data/transcripts/ is
used. Each run writes its output under data/replay/<session>_<timestamp>/
so prior runs survive for comparison.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from journal_agent.comms.llm_registry import build_llm_registry
from journal_agent.configure.config_builder import (
    LLM_ROLE_CONFIG,
    configure_environment,
    models,
)
from journal_agent.graph.nodes.classifiers import (
    make_exchange_decomposer,
    make_thread_classifier,
    make_thread_fragment_extractor,
)
from journal_agent.model.session import Exchange


REPO_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS_DIR = REPO_ROOT / "data" / "test"
REPLAY_DIR = REPO_ROOT / "data" / "replay"


def pick_transcript(arg: str | None) -> Path:
    """Return the transcript path: CLI arg, or newest in data/transcripts."""
    if arg:
        p = Path(arg)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = sorted(TRANSCRIPTS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No transcripts found in {TRANSCRIPTS_DIR}")
    return candidates[-1]


def load_exchanges(path: Path) -> list[Exchange]:
    """Parse a transcript jsonl file into Exchange models."""
    exchanges: list[Exchange] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            exchanges.append(Exchange.model_validate_json(line))
    return exchanges


def write_json(items: list, path: Path) -> None:
    """Write a list of pydantic models as a pretty-printed JSON array."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [item.model_dump(mode="json") for item in items]
    path.write_text(json.dumps(payload, indent=2, default=str))




def main() -> None:
    transcript_path = pick_transcript(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"[replay] Transcript: {transcript_path.name}")

    exchanges = load_exchanges(transcript_path)
    print(f"[replay] Loaded {len(exchanges)} exchanges")

    settings = configure_environment()
    registry = build_llm_registry(
        settings=settings,
        models=models,
        role_config=LLM_ROLE_CONFIG,
    )

    session_id = exchanges[0].session_id or transcript_path.stem
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = REPLAY_DIR / f"{transcript_path.stem}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[replay] Output: {run_dir}")

    # ── Decomposer ────────────────────────────────────────────────────────
    decomposer_fn = make_exchange_decomposer(
        llm=registry.get("classifier"),  # reuse the classifier model for now
    )
    result = decomposer_fn({"session_id": session_id, "transcript": exchanges})
    threads = result.get("threads", [])
    if not threads:
        print(f"[replay] Decomposer returned no records: {result}")
        return
    print(f"[replay] Decomposer produced {len(threads)} record(s)")
    write_json(threads, run_dir / "threads.json")

    # # ── Classifier ────────────────────────────────────────────────────────
    classifier_fn = make_thread_classifier(
        llm=registry.get("classifier"),
    )

    result = classifier_fn({"session_id": session_id, "transcript": exchanges,  "threads": threads})
    threads = result.get("classified_threads", [])
    if not threads:
        print(f"[replay] Classifier returned no records: {result}")
        return
    print(f"[replay] Classifier produced {len(threads)} record(s)")
    write_json(threads, run_dir / "classified_threads.json")

    # # ── Extractor ─────────────────────────────────────────────────────────
    extractor_fn = make_thread_fragment_extractor(llm=registry.get("extractor"))
    result = extractor_fn({"session_id": session_id, "transcript": exchanges,  "classified_threads": threads})
    fragments = result.get("fragments", [])
    if not fragments:
        print(f"[replay] Extractor returned no records: {result}")
        return
    print(f"[replay] Extractor produced {len(fragments)} fragment(s)")
    write_json(fragments, run_dir / "fragments.json")

    print(f"[replay] Done. Inspect: {run_dir}")


if __name__ == "__main__":
    main()
