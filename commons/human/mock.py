"""
MockHuman — deterministic CallHuman implementation for testing.

Provides canned responses so tests don't block on input().
"""

from __future__ import annotations

from commons.human.protocols import (
    HumanRequest,
    HumanResponse,
)


class MockHuman:
    """
    CallHuman implementation that returns pre-configured responses.

    Cycles through the provided responses in order.  If exhausted,
    returns the fallback response (default: skip).

    Usage:
        human = MockHuman(responses=["yes", "I fixed the goal", "done"])
        human(HumanRequest(prompt="..."))  # -> HumanResponse(content="yes")
        human(HumanRequest(prompt="..."))  # -> HumanResponse(content="I fixed the goal")
        human(HumanRequest(prompt="..."))  # -> HumanResponse(content="done")
        human(HumanRequest(prompt="..."))  # -> HumanResponse(content="", skipped=True)
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        fallback: str | None = None,
    ):
        self._responses = list(responses or [])
        self._fallback = fallback
        self._call_count = 0
        self._requests: list[HumanRequest] = []

    def __call__(self, request: HumanRequest) -> HumanResponse:
        self._requests.append(request)
        idx = self._call_count
        self._call_count += 1

        if idx < len(self._responses):
            return HumanResponse(content=self._responses[idx])

        if self._fallback is not None:
            return HumanResponse(content=self._fallback)

        return HumanResponse(content="", skipped=True)

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def requests(self) -> list[HumanRequest]:
        """All requests received, in order.  Useful for test assertions."""
        return list(self._requests)
