"""
Validation layer for integrity rule evaluation.

This module provides deterministic validation of graph structure against integrity rules.
"""

from conversation_engine.validation.evaluator import RuleEvaluator, RuleViolation

__all__ = [
    "RuleEvaluator",
    "RuleViolation",
]
