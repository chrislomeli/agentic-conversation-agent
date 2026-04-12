"""
Architectural ontology quiz — domain-specific questions for pre-run LLM validation.

These questions test whether the LLM understands:
  1. The knowledge graph structure (node types, edge types, traceability)
  2. The integrity rules and what violations mean
  3. The conversation loop's role (validate → reason → respond)
  4. How findings work and what the LLM should do about them

The system prompt is also defined here — it's the same prompt the
conversation loop would use when the reason node calls the LLM.
"""

from __future__ import annotations

from conversation_engine.models.validation_quiz import FactualQuiz


# ── System prompt ───────────────────────────────────────────────────

ARCHITECTURAL_SYSTEM_PROMPT = """\
You are an architectural integrity assistant. You help maintain a \
knowledge graph that represents a software system's architecture.

The knowledge graph contains these node types:
- goal: A desired outcome or objective
- requirement: A specific system need (functional or non-functional)
- step: A system implementation step or task
- dependency: An external system, library, or service
- feature: A high-level product offering
- use_case: A user interaction or workflow
- scenario: A concrete instance of a use case
- design_artifact: A design decision or element
- decision: An architectural decision with rationale
- constraint: A limitation or restriction
- guiding_principle: A design or architectural principle
- documentation_artifact: Documentation or explanatory content

Nodes are connected by typed edges that represent traceability:
- SATISFIED_BY: goal → requirement (a goal is satisfied by requirements)
- REALIZED_BY: requirement → step (a requirement is realized by steps)
- DEPENDS_ON: step → dependency (a step depends on something)
- CONSTRAINS: constraint → step (a constraint limits a step)
- DESCRIBED_BY: use_case → scenario (a use case has scenarios)
- DOCUMENTED_BY: any → documentation_artifact
- HAS_SCENARIO: use_case → scenario
- INSTANCE_OF: scenario → use_case
- INFORMS: decision → step

Integrity rules enforce structural constraints:
- Every goal MUST have at least one requirement (via SATISFIED_BY)
- Every requirement MUST have at least one step (via REALIZED_BY)
- Every step SHOULD declare its dependencies (via DEPENDS_ON)

When integrity rules are violated, the system produces "findings" — \
domain-agnostic reports of issues. Each finding has:
- A severity (high, medium, low)
- The subject node(s) involved
- A human-readable message describing the gap
- Evidence explaining what was expected vs. found

Your job in the conversation loop is:
1. Review the findings from the validation pass
2. Explain the issues clearly to the user
3. Suggest concrete actions to resolve each finding
4. Prioritize by severity (high → medium → low)

You do NOT modify the knowledge graph directly. You advise the user \
on what changes would resolve the integrity violations.
"""


# ── Quiz questions ──────────────────────────────────────────────────

ARCHITECTURAL_QUIZ: list[FactualQuiz] = [
    # Q1: Node types
    FactualQuiz(
        id="quiz-node-types",
        name="Node Types Quiz",
        question=("What are the main node types in the knowledge graph? List as many as you can."),
        expected_answer="goal, requirement, step, dependency, feature",
        weight=1.0,
        min_score=0.5,
    ),
    # Q2: Edge types and traceability
    FactualQuiz(
        id="quiz-edge-traceability",
        name="Edge Types and Traceability Quiz",
        question=(
            "How does traceability work in this knowledge graph? "
            "Describe the chain from goals to steps."
        ),
        expected_answer="goal, requirement, step, satisfied_by, realized_by",
        weight=1.5,
        min_score=0.5,
    ),
    # Q3: Integrity rules
    FactualQuiz(
        id="quiz-integrity-rules",
        name="Integrity Rules Quiz",
        question=("What are the integrity rules? What happens when a goal has no requirements?"),
        expected_answer="goal, requirement, violation, finding",
        weight=1.5,
        min_score=0.5,
    ),
    # Q4: Findings and severity
    FactualQuiz(
        id="quiz-findings",
        name="Findings and Severity Quiz",
        question=("What is a 'finding' in this system? What information does a finding contain?"),
        expected_answer="severity, subject, message",
        weight=1.0,
        min_score=0.5,
    ),
    # Q5: Role understanding
    FactualQuiz(
        id="quiz-role",
        name="Role Understanding Quiz",
        question=(
            "What is your role in the conversation loop? "
            "Do you modify the knowledge graph directly?"
        ),
        expected_answer="advise, suggest",
        weight=1.5,
        min_score=0.5,
    ),
    # Q6: Prioritization
    FactualQuiz(
        id="quiz-prioritization",
        name="Prioritization Quiz",
        question=(
            "If there are 3 findings — one high severity, one medium, "
            "and one low — in what order should you address them?"
        ),
        expected_answer="high, medium, low",
        weight=0.5,
        min_score=0.5,
    ),
    # Q7: Edge case — no findings
    FactualQuiz(
        id="quiz-no-findings",
        name="No Findings Quiz",
        question=("What should you tell the user if the validation pass produces zero findings?"),
        expected_answer="complete, pass",
        weight=0.5,
        min_score=0.5,
    ),
]
