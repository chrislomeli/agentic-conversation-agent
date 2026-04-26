"""compare.py — Diff two eval run files to detect prompt or output changes.

The comparator joins records by ``(fixture_id, classifier)`` and reports:
  - SAME    — output and prompt version unchanged
  - CHANGED — output or prompt version differs (with a field-level breakdown)
  - NEW     — present in run B but not run A
  - DROPPED — present in run A but not run B

Typical workflow::

    # baseline
    python -m journal_agent.scripts.run_evals --output evals/run_baseline.jsonl

    # bump a prompt VERSION, then:
    python -m journal_agent.scripts.run_evals --output evals/run_v2.jsonl

    # see what changed
    python -m journal_agent.scripts.run_evals --compare evals/run_baseline.jsonl evals/run_v2.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path

from journal_agent.evals.runner import EvalRecord, load_results


def _key(r: EvalRecord) -> tuple[str, str]:
    return (r.fixture_id, r.classifier)


def _output_diff(a: dict, b: dict) -> list[str]:
    """Return human-readable lines describing what changed between two output dicts."""
    lines: list[str] = []
    all_keys = sorted(set(a) | set(b))
    for k in all_keys:
        if k not in a:
            lines.append(f"    + {k}: {json.dumps(b[k], default=str)[:120]}")
        elif k not in b:
            lines.append(f"    - {k}: {json.dumps(a[k], default=str)[:120]}")
        elif a[k] != b[k]:
            a_summary = json.dumps(a[k], default=str)[:80]
            b_summary = json.dumps(b[k], default=str)[:80]
            lines.append(f"    ~ {k}")
            lines.append(f"      A: {a_summary}")
            lines.append(f"      B: {b_summary}")
    return lines


def compare_runs(path_a: Path, path_b: Path) -> str:
    """Compare two run files and return a human-readable report string."""
    records_a = {_key(r): r for r in load_results(path_a)}
    records_b = {_key(r): r for r in load_results(path_b)}
    all_keys = sorted(set(records_a) | set(records_b))

    lines = [
        f"Eval comparison",
        f"  A: {path_a}",
        f"  B: {path_b}",
        "",
    ]

    counts = {"same": 0, "changed": 0, "new": 0, "dropped": 0}

    for key in all_keys:
        fixture_id, classifier = key
        label = f"{classifier} / {fixture_id}"

        if key not in records_a:
            lines.append(f"  NEW     {label}")
            counts["new"] += 1
            continue

        if key not in records_b:
            lines.append(f"  DROPPED {label}")
            counts["dropped"] += 1
            continue

        a, b = records_a[key], records_b[key]
        diffs: list[str] = []

        if a.prompt_version != b.prompt_version:
            diffs.append(f"    prompt_version: {a.prompt_version!r} → {b.prompt_version!r}")

        if a.input_hash != b.input_hash:
            diffs.append(f"    input_hash: {a.input_hash!r} → {b.input_hash!r} (fixture content changed)")

        if a.error != b.error:
            diffs.append(f"    error: {a.error!r} → {b.error!r}")

        if a.output != b.output:
            diffs.append("    output changed:")
            diffs.extend(_output_diff(a.output, b.output))

        if diffs:
            lines.append(f"  CHANGED {label}")
            lines.extend(diffs)
            counts["changed"] += 1
        else:
            counts["same"] += 1

    lines.append("")
    lines.append(
        f"Summary: {counts['same']} same  |  {counts['changed']} changed  |  "
        f"{counts['new']} new  |  {counts['dropped']} dropped"
    )
    return "\n".join(lines)
