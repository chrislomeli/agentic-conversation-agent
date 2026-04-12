"""
human — Backwards-compatibility shim.

Reusable human protocols have moved to ``commons.human``.
"""

from commons.human import (
    CallHuman,
    ConsoleHuman,
    HumanRequest,
    HumanResponse,
    MockHuman,
)

__all__ = [
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
