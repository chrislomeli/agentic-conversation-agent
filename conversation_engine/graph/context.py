"""
ConversationContext protocol — the contract between the conversation loop
and any domain that plugs into it.

The loop is domain-agnostic: it validates, reasons, responds, and routes.
The *context* supplies domain-specific validation, finding types, and
state management.  Different domains implement this protocol to reuse
the same conversation loop.

Design principles:
- Protocol, not ABC — structural subtyping, no inheritance required
- Domain types stay behind the boundary — the loop sees Finding, not Assessment
- Immutable findings — the loop reads them, only the context produces them
- The context owns the domain state; the loop owns conversation mechanics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from conversation_engine.models.validation_quiz import ValidationQuiz

# ── Domain-agnostic types ───────────────────────────────────────────
# These are the only types the conversation loop sees.

Severity = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class Finding:
    """
    A single domain-agnostic finding produced by validation.

    This is what the conversation loop works with.  It does not know
    whether the finding came from a graph integrity check, a code lint
    rule, a security scan, or anything else.
    """

    id: str
    finding_type: str
    severity: Severity
    subject_ids: list[str]
    message: str
    evidence: list[str] = field(default_factory=list)
    related_rule_ids: list[str] = field(default_factory=list)
    resolved: bool = False
    conversation_turn_id: str | None = None


@dataclass(frozen=True)
class ValidationResult:
    """
    The complete result of a validation pass.

    Contains all findings (both new and previously-resolved) so the
    loop has a single, consistent snapshot.
    """

    findings: list[Finding]
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Protocol ────────────────────────────────────────────────────────


@runtime_checkable
class ConversationContext(Protocol):
    """
    The contract a domain must satisfy to plug into the conversation loop.

    Implementations hold all domain-specific state (graphs, rules,
    schemas, etc.) and expose only domain-agnostic operations to the
    loop.
    """

    def validate(self, prior_findings: list[Finding]) -> ValidationResult:
        """
        Run domain-specific validation and return findings.

        The loop passes in the previous findings so the context can
        preserve resolved findings and replace unresolved ones with
        fresh results.

        Args:
            prior_findings: Findings from the previous turn (may include
                            resolved findings the context should preserve).

        Returns:
            A ValidationResult with the current set of findings.
        """
        ...

    def format_finding_summary(self, findings: list[Finding]) -> str:
        """
        Produce a human-readable summary of the given findings.

        The loop calls this so the *domain* controls the language, not
        the loop.  A graph-ontology domain will talk about "goals" and
        "requirements"; a code-review domain will talk about "functions"
        and "coverage".

        Args:
            findings: The open (unresolved) findings to summarise.

        Returns:
            A plain-text summary suitable for an AI message.
        """
        ...

    def get_domain_state(self) -> dict[str, Any]:
        """
        Return an opaque snapshot of domain state for checkpointing.

        The loop never inspects this; it just stores and passes it
        through.  Useful for serialisation / replay.
        """
        ...

    @property
    def system_prompt(self) -> str:
        """
        The system prompt that teaches an LLM about this domain.

        Used by the conversation loop's reason node and by the
        pre-flight LLM validator.  The context owns this because
        only the domain knows what the LLM needs to understand.
        """
        ...

    @property
    def preflight_quiz(self) -> list[ValidationQuiz]:
        """
        Quiz questions for pre-run LLM validation.

        The context provides these because the quiz is tightly
        coupled to the system prompt and domain concepts.
        Return an empty list to skip pre-flight validation.
        """
        ...
