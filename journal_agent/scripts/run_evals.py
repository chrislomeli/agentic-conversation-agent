"""run_evals.py — CLI entry point for the classifier eval harness.

Run the full suite against the built-in fixtures::

    uv run python -m journal_agent.scripts.run_evals --output evals/run1.jsonl

Compare two runs::

    uv run python -m journal_agent.scripts.run_evals \\
        --compare evals/run1.jsonl evals/run2.jsonl

Use custom fixtures::

    uv run python -m journal_agent.scripts.run_evals \\
        --fixtures path/to/my_fixtures.jsonl \\
        --output evals/run_custom.jsonl

The script loads settings via ``configure_environment()`` and builds one shared
LLMClient (using the "classifier" role from the registry).  Each classifier is
invoked in isolation — no LangGraph graph, no checkpointer.  Results are JSONL
so they can be versioned and diffed.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


def _build_llm():
    """Load settings and return an LLMClient for the 'classifier' role."""
    from journal_agent.comms.llm_registry import build_llm_registry
    from journal_agent.configure.config_builder import (
        LLM_ROLE_CONFIG,
        configure_environment,
        models,
    )
    settings = configure_environment()
    registry = build_llm_registry(settings=settings, models=models, role_config=LLM_ROLE_CONFIG)
    return registry.get("classifier")


async def _run(fixtures_path: Path, output_path: Path) -> None:
    from journal_agent.evals.fixtures import load_fixtures
    from journal_agent.evals.runner import run_suite, write_results

    llm = _build_llm()
    fixtures = load_fixtures(fixtures_path)
    logger.info("Loaded %d fixture(s) from %s", len(fixtures), fixtures_path)

    records = await run_suite(fixtures, llm)

    errors = [r for r in records if r.error]
    if errors:
        logger.warning("%d record(s) had errors:", len(errors))
        for r in errors:
            logger.warning("  %s / %s — %s", r.classifier, r.fixture_id, r.error)

    write_results(records, output_path)
    logger.info("Done. %d records written to %s", len(records), output_path)


def main() -> None:
    from journal_agent.evals.fixtures import FIXTURE_PATH

    parser = argparse.ArgumentParser(description="Journal agent classifier eval harness")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run the eval suite")
    run_parser.add_argument(
        "--fixtures", type=Path, default=FIXTURE_PATH,
        help="Path to JSONL fixture file (default: built-in eval_fixtures.jsonl)",
    )
    run_parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write JSONL results",
    )

    cmp_parser = subparsers.add_parser("compare", help="Compare two run files")
    cmp_parser.add_argument("run_a", type=Path, help="Baseline run JSONL")
    cmp_parser.add_argument("run_b", type=Path, help="New run JSONL")

    # support legacy flat flags for backwards compat
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Compare two run files (shorthand without subcommand)")
    parser.add_argument("--fixtures", type=Path, default=FIXTURE_PATH)
    parser.add_argument("--output", type=Path)

    args = parser.parse_args()

    if args.compare:
        from journal_agent.evals.compare import compare_runs
        print(compare_runs(Path(args.compare[0]), Path(args.compare[1])))
        return

    if args.command == "compare":
        from journal_agent.evals.compare import compare_runs
        print(compare_runs(args.run_a, args.run_b))
        return

    if args.command == "run" or args.output:
        output = args.output
        fixtures = getattr(args, "fixtures", FIXTURE_PATH)
        asyncio.run(_run(fixtures, output))
        return

    parser.print_help()
    sys.exit(1)


if __name__ == "__main__":
    main()
