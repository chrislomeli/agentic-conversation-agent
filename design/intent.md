# Intent Scoring Design

Three-layer pipeline: **ScoreCard** (LLM) → **Named Intent** (truth table) → **Behavioral Decisions** (lookup).

## Implementation Rubric

| # | What                                                            | Where | Notes | 
|---|-----------------------------------------------------------------|-------|-------|
| 1 | `Domain`, `ScoreCard` Pydantic models                   | `model/session.py` | ✅ Done  All-float scores; 8 fixed domains | 
| 2 | Per-dimension thresholds                                        | `configure/` | One tunable threshold per score dimension; `score > threshold` → bool |
| 3 | `Intent` enum + truth table lookup                              | `model/session.py` or `configure/` | 8 intents from 3 sigmoid-thresholded booleans |
| 4 | Layer 3 behavior lookup (posture, RAG strategy, intent framing) | `configure/` | Static dict mapping `Intent` → posture + RAG + framing string |
| 5 | `"scorer"` prompt template                                      | `configure/prompts.py` | Rubric with anchored scales and fixed domain list |
| 6 | `"socratic"` and `"guide"` posture prompts                      | `configure/prompts.py` | ✅ Done |
| 7 | `make_score_intent(llm)` node                                   | `graph/nodes/scorer.py` | Same factory-closure pattern; calls `llm.structured(ScoreCard)` |
| 8 | `score_card` + `intent` fields on `JournalState`                | `graph/state.py` | `ScoreCard \| None`, `Intent \| None` |
| 9 | Wire node into graph                                            | `graph/graph.py` | `get_user_input → score_intent → retrieve_history → get_ai_response` |
| 10 | Registry entry for scorer LLM                                   | `configure/settings` / registry | Cheap model (e.g. gpt-4o-mini) |
| 11 | Inject intent framing into conversation prompt                  | `configure/context_builder.py` | Prepend `<intent>{framing}</intent>` before posture prompt |
| 12 | Use domain scores in `retrieve_history`                         | `graph/graph.py` or retrieval node | Filter vector search by top-scoring domains |
| 13 | Token-count logging + warning                                   | `configure/context_builder.py` | Log assembled context token count; warn at >2K tokens |
| 14 | Normalize vector distances to 0–1 relevance                     | `storage/vector_store.py` | `search_fragments` returns relevance scores, not raw distances. Behavior table uses universal 0–1 scale. |

---

## Layer 1: ScoreCard (LLM structured output)

The `score_intent` node sends the current human message + last 5 messages to a
cheap LLM (e.g. gpt-4o-mini) with `llm.structured(ScoreCard)`.


```python
class Domain(BaseModel):
    tag: str        # from fixed taxonomy list
    score: float    # 0.0–1.0

class ScoreCard(BaseModel):
    question_score: float       # 0.0–1.0  how much is this a request for information/opinion
    first_person_score: float   # 0.0–1.0  how much is the speaker talking about themselves
    task_score: float           # 0.0–1.0  how much does this contain an explicit directive
    domains: list[Domain]       # scores across all 8 domains
```


Fixed domain list (all 8 always present in output):
- **creativity** — art, writing, music, design, creative process
- **project** — conversations about a specific project (new or existing)
- **humanity** — meaning, ethics, worldview, abstract reasoning
- **music** — music theory, music performance, music production, music history
- **health** — physical/mental wellbeing, fitness, habits, stress
- **relationships** — interpersonal dynamics, family, friendships, conflict
- **technology** — tools, engineering, software, systems
- **journal** — day-to-day activities, daily tasks, daily goals
- **schedule** — daily schedule, to-do list, time management
- **personal_goals** — career goals, personal goals, life goals
- **finance** — money, budgeting, investing, economic decisions


Example ScoreCard output:
```json
{
  "question_score": 0.9,
  "first_person_score": 0.8,
  "task_score": 0.7,
  "domains": [
    { "tag": "creativity", "score": 0.0 },
    { "tag": "project", "score": 0.8 },
    { "tag": "humanity", "score": 0.6 },
    { "tag": "music", "score": 0.0 },
    { "tag": "health", "score": 0.0 },
    { "tag": "relationships", "score": 0.1 },
    { "tag": "technology", "score": 0.0 },
    { "tag": "journal", "score": 0.0 },
    { "tag": "schedule", "score": 0.0 },
    { "tag": "personal_goals", "score": 0.0 },
    { "tag": "finance", "score": 0.0 }
  ]
}
```

---

## Layer 1.5: Thresholding (scores → booleans)

The LLM outputs continuous scores in [0, 1]. A per-dimension threshold converts
each into a boolean for the truth-table lookup.

```python
THRESHOLDS = {
    "question":     0.5,
    "first_person": 0.5,
    "task":         0.5,
}
```

- Thresholds are tunable per dimension — e.g., lower `first_person` if the LLM
  under-reports self-reference, raise `task` if it over-calls directives.
- Raw scores are logged for debugging; only the booleans drive intent resolution.

> **Why not a sigmoid?** An earlier draft squashed each score through
> `sigmoid(x, midpoint, steepness)` before thresholding at 0.5. That reduces
> exactly to `x > midpoint` — sigmoid is monotonic and crosses 0.5 at its
> midpoint, so the squash is invisible once you threshold. The sigmoid only
> earns its keep if the squashed value is used downstream as a soft membership
> (weighted retrieval, blended decisions). Layer 2 collapses straight to
> booleans, so we skip it and threshold directly.

---

## Layer 2: Named Intent (deterministic truth table)

Thresholded scores → 8 named intents. Pure Python, no LLM call.

| question | first_person | task | Intent              | Example                                              |
|----------|--------------|------|---------------------|------------------------------------------------------|
| T        | T            | T    | **seeking_help**    | "Can you help me figure out my career path?"         |
| T        | T            | F    | **self_questioning**| "Am I even cut out for this?"                        |
| T        | F            | T    | **researching**     | "Can you explain how transformers work?"             |
| T        | F            | F    | **curious**         | "What's the deal with stoicism?"                     |
| F        | T            | T    | **planning**        | "I need to build a portfolio site this weekend"      |
| F        | T            | F    | **musing**          | "I've been thinking a lot about what matters to me"  |
| F        | F            | T    | **directing**       | "Summarize the key points of that article"           |
| F        | F            | F    | **observing**       | "The tech industry is shifting toward AI agents"     |

```python
from enum import Enum

class Intent(Enum):
    SEEKING_HELP     = (True,  True,  True)
    SELF_QUESTIONING = (True,  True,  False)
    RESEARCHING      = (True,  False, True)
    CURIOUS          = (True,  False, False)
    PLANNING         = (False, True,  True)
    MUSING           = (False, True,  False)
    DIRECTING        = (False, False, True)
    OBSERVING        = (False, False, False)

# Resolve from scorecard via per-dimension threshold:
q  = card.question_score     > THRESHOLDS["question"]
fp = card.first_person_score > THRESHOLDS["first_person"]
t  = card.task_score         > THRESHOLDS["task"]
intent = Intent((q, fp, t))
```

---

## Layer 3: Behavioral Decisions (deterministic lookup)

Each intent maps to a posture and two independent retrieval decisions. Pure Python.

### Context sources

| Source | State field | What it provides | When to include |
|--------|------------|------------------|----------------|
| **System prompt** | — | Posture + intent framing | Always |
| **Current session** | `session_messages` | The live conversation | Always (pruned by `ContextBuilder` budget) |
| **Recent sessions** | `seed_context` | Continuity: "last time you mentioned X" | When `first_person` is high |
| **Domain fragments** | `retrieved_history` | Topical depth from vector store | When domain scores are above threshold |

### Behavior table

Relevance is normalized 0–1 (1 = perfect match). See [normalization](#relevance-normalization) below.

| Intent               | Posture  | Recent sessions | top_k | min_relevance | Rationale |
|----------------------|----------|-----------------|-------|---------------|-----------|
| **seeking_help**     | Guide    | ✅ include       | 5     | 0.5           | Needs precise, relevant fragments + personal context |
| **self_questioning** | Socratic | ✅ include       | 2     | 0.6           | Light retrieval — mostly reflect, don't overwhelm with data |
| **researching**      | Guide    | skip            | 5     | 0.3           | Cast wider net for topical depth |
| **curious**          | Guide    | skip            | 3     | 0.3           | Moderate retrieval — exploratory, not urgent |
| **planning**         | Guide    | ✅ include       | 5     | 0.5           | Needs actionable past context + personal history |
| **musing**           | Socratic | ✅ include       | 0     | —             | No domain retrieval — let the user wander |
| **directing**        | Guide    | skip            | 3     | 0.5           | Focused task, precise matches only |
| **observing**        | Socratic | skip            | 3     | 0.3           | Surface related past observations, loosely matched |

Relevance reference: >0.7 strong, 0.4–0.7 moderate, <0.4 weak.

### Resolution code

```python
behavior = INTENT_BEHAVIOR[intent]
posture = behavior.posture

# Domain fragments — filter vector search by top-scoring domains
rag_domains = [d.tag for d in card.domains if d.score > threshold] if behavior.top_k > 0 else []
top_k = behavior.top_k
min_relevance = behavior.min_relevance

# Recent sessions — include seed_context for personal continuity
include_recent_sessions = behavior.use_recent_sessions
```

### Relevance normalization

The vector store returns raw distances whose scale depends on the distance
metric and embedding model. Normalize to a universal 0–1 relevance score
at the store boundary so the behavior table never changes when you swap backends.

```python
# Inside VectorStore — convert raw distance to 0–1 relevance.
# The formula depends on which distance metric the collection uses.

def _relevance_from_l2(distance: float, max_useful: float = 2.0) -> float:
    """L2 (Euclidean): 0 = identical, unbounded above.
    Clamp to max_useful then invert to 0–1."""
    return max(0.0, 1.0 - distance / max_useful)

def _relevance_from_cosine(distance: float) -> float:
    """Cosine distance: 0 = identical, 2 = opposite.
    Simple linear inversion."""
    return max(0.0, 1.0 - distance / 2.0)
```

`max_useful` is the one tuning knob — it's the raw L2 distance you consider
"completely irrelevant." With Chroma's default embeddings, 2.0 is a safe
starting point (distances rarely exceed it for real queries). Anything beyond
it clamps to relevance 0.

Callers filter with `if relevance >= min_relevance` instead of
`if distance <= max_distance`.

---

## Graph Position

```
get_user_input → score_intent → retrieve_history → get_ai_response
```

- `score_intent` node: `make_score_intent(llm)` — same factory-closure pattern
- ScoreCard + Intent go into `JournalState`
- `retrieve_history` uses domain scores to narrow vector search
- `get_ai_response` uses posture to select prompt style

---

## Reference: Context Size Guidelines

| Size | Tokens (~) | What fits | Risk |
|------|-----------|-----------|------|
| < 2 KB | ~500 | System prompt + intent framing + posture | Sweet spot for instructions |
| 2–8 KB | 500–2K | + 3–5 focused retrieved fragments | Good. Model attends well |
| 8–16 KB | 2K–4K | + 10+ fragments or long history | Starting to dilute. Diminishing returns |
| 16–32 KB | 4K–8K | Heavy retrieval, full session history | "Lost in the middle" territory |
| > 32 KB | 8K+ | Kitchen sink | Actively harmful to quality |

1 token ≈ 4 chars ≈ 0.004 KB → 1 KB ≈ 250 tokens.

Token budget is split across two concerns:

- **Injected context** (system prompt + recent sessions + domain fragments): target **< 2K tokens**.
  `ContextBuilder` should log the assembled total and warn when it crosses 2K.
- **Current session messages**: grows per turn; managed by existing `ContextBuilder`
  pruning within `max_tokens` (currently 8K). Not counted against the 2K target.