"""
human — Protocol-based human interaction layer.

Protocols
---------
CallHuman     Surface a message to a human, get their response back.

Implementations
---------------
ConsoleHuman    CLI-based (input/print).
MockHuman       Deterministic stub for testing.

Swap in any callable that matches the Protocol signature.
"""

from commons.human.console import ConsoleHuman
from commons.human.mock import MockHuman
from commons.human.protocols import (
    CallHuman,
    HumanRequest,
    HumanResponse,
)

__all__ = [
    "CallHuman",
    "HumanRequest",
    "HumanResponse",
    "ConsoleHuman",
    "MockHuman",
]
