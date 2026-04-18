"""
_helpers.py — shared utilities for prompt modules.

Loads the taxonomy once from YAML and provides the schema-block helper
used by every prompt that injects Pydantic schemas into its template.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml
from pydantic import BaseModel

from journal_agent.model.session import Ideation

_TAXONOMY_PATH = Path(__file__).parent.parent / "taxonomy.yaml"


def _load_taxonomy() -> list[Ideation]:
    with open(_TAXONOMY_PATH) as f:
        raw = yaml.safe_load(f)
    return [Ideation(**entry) for entry in raw]


TAXONOMY: list[Ideation] = _load_taxonomy()


def taxonomy_json() -> str:
    """Serialize the taxonomy for injection into prompts."""
    return json.dumps([asdict(t) for t in TAXONOMY], indent=2)


def _schema_block(model_cls: type[BaseModel]) -> str:
    """Return a compact JSON-Schema representation for a Pydantic model."""
    return json.dumps(model_cls.model_json_schema(), indent=2)
