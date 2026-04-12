"""
Concrete ConversationContext for the architectural ontology domain.

This adapter bridges the domain-specific models (KnowledgeGraph,
IntegrityRule, RuleEvaluator, Assessment) to the domain-agnostic
ConversationContext protocol that the conversation loop requires.

It owns the domain state and exposes only Findings to the loop.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from conversation_engine.graph.context import (
    ConversationContext,
    Finding,
    ValidationResult,
)
from conversation_engine.models.domain_config import DomainConfig
from conversation_engine.models.project_spec import ProjectSpecification
from conversation_engine.models.rule_node import IntegrityRule

# from conversation_engine.models.query_node import GraphQueryPattern
from conversation_engine.storage.graph import KnowledgeGraph
from conversation_engine.storage.snapshot_facade import snapshot_to_graph
from conversation_engine.validation.evaluator import RuleEvaluator
from conversation_engine.models.validation_quiz import ValidationQuiz
from conversation_engine.infrastructure.llm.architectural_quiz import (
    ARCHITECTURAL_SYSTEM_PROMPT,
    ARCHITECTURAL_QUIZ,
)


# ── Mapping helpers ─────────────────────────────────────────────────

_RULE_ID_TO_FINDING_TYPE: Dict[str, str] = {
    "rule-goal-req": "missing_goal_coverage",
    "rule-req-step": "missing_requirement_realization",
    "rule-step-dep": "missing_step_dependencies",
}


def _default_finding_type(rule_id: str) -> str:
    return _RULE_ID_TO_FINDING_TYPE.get(rule_id, "missing_goal_coverage")


# ── Concrete context ────────────────────────────────────────────────


class ArchitecturalOntologyContext:
    """
    ConversationContext DUCK TYPE implementation backed by the architectural
    ontology's KnowledgeGraph, IntegrityRules, and RuleEvaluator.

    This class satisfies the ConversationContext protocol via structural
    subtyping — no inheritance from the Protocol class is needed.
    """

    def __init__(self, config: DomainConfig) -> None:
        self._config = config
        # Build the graph at runtime from the flat spec via the facade
        if config.project_spec is not None:
            self._graph = snapshot_to_graph(config.project_spec)
        else:
            self._graph = KnowledgeGraph()
        self._rules = config.rules or []
        # self._query_patterns = config.query_patterns or []

    @classmethod
    def from_spec_parts(
        cls,
        graph: KnowledgeGraph,
        rules: List[IntegrityRule],
        # query_patterns: List[GraphQueryPattern] | None = None,
        system_prompt: Optional[str] = None,
        quiz: Optional[List[ValidationQuiz]] = None,
    ) -> "ArchitecturalOntologyContext":
        """
        Convenience factory for callers that don't have a DomainConfig yet.

        Wraps the individual arguments into a DomainConfig and delegates
        to the primary constructor.
        """
        from conversation_engine.storage.snapshot_facade import graph_to_snapshot

        spec = graph_to_snapshot("unnamed", graph)
        return cls(
            DomainConfig(
                project_name="unnamed",
                project_spec=spec,
                rules=rules,
                # query_patterns=query_patterns,
                system_prompt=system_prompt,
                quiz=quiz,
            )
        )

    # ── Protocol implementation ─────────────────────────────────────

    def validate(self, prior_findings: List[Finding]) -> ValidationResult:
        """
        Run RuleEvaluator against the graph and return Findings.

        Previously-resolved findings are preserved.  Unresolved findings
        are replaced with the fresh evaluation.
        """
        evaluator = RuleEvaluator(self._graph)
        violations = evaluator.evaluate_all_rules(self._rules)

        # Preserve resolved findings from prior turns
        resolved = [f for f in prior_findings if f.resolved]

        # Convert violations → domain-agnostic Findings
        new_findings: List[Finding] = []
        for v in violations:
            new_findings.append(
                Finding(
                    id=f"finding-{uuid.uuid4().hex[:8]}",
                    finding_type=_default_finding_type(v.rule_id),
                    severity=v.severity,
                    subject_ids=[v.node_id],
                    message=v.message,
                    evidence=[f"Expected {v.expected_count}, found {v.actual_count}"],
                    related_rule_ids=[v.rule_id],
                )
            )

        return ValidationResult(findings=resolved + new_findings)

    def format_finding_summary(self, findings: List[Finding]) -> str:
        """
        Produce a human-readable summary using architectural language.
        """
        if not findings:
            return "All integrity checks pass. The knowledge graph looks complete."

        lines = [f"Found {len(findings)} issue(s):"]
        for f in findings:
            lines.append(f"  [{f.severity}] {f.message}")
        return "\n".join(lines)

    def get_domain_state(self) -> Dict[str, Any]:
        """
        Return a snapshot of the domain state for checkpointing.
        """
        return {
            "graph": self._graph,
            "rules": self._rules,
            # "query_patterns": self._query_patterns,
        }

    # ── Domain-specific accessors (not part of the protocol) ────────

    @property
    def graph(self) -> KnowledgeGraph:
        return self._graph

    @property
    def rules(self) -> List[IntegrityRule]:
        return self._rules

    # @property
    # def query_patterns(self) -> List[GraphQueryPattern]:
    #     return self._query_patterns

    @property
    def system_prompt(self) -> str:
        return self._config.system_prompt or ARCHITECTURAL_SYSTEM_PROMPT

    @property
    def preflight_quiz(self) -> List[ValidationQuiz]:
        if self._config.quiz is not None:
            return list(self._config.quiz)
        return list(ARCHITECTURAL_QUIZ)
