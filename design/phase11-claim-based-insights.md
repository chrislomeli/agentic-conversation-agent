# Phase 11: Claim-Based Insights — Design & Build Plan

## Context

Phase 10 implemented reflective memory as **clustering**: HDBSCAN/LLM-based clustering over fragments → cluster-labeled Insights → verification. The Insight is conceptually a child of a cluster of fragments.

Phase 11 reframes the model. **An insight is not a child of fragments — it is an observation about the user that fragments cast votes on.** Fragments don't "belong" to an insight; they accumulate as evidence over time. The system asks of every new fragment: *which existing ideas does this support, contradict, or sit outside of? Or does it introduce a new idea?*

This shift moves the system from offline batch reflection toward **continuous belief tracking**, drawing on the literature around:

- Stanford Generative Agents (Park et al., 2023) — memory stream + reflections evaluated against new memories
- Belief Revision (AGM theory) — formal model for how beliefs update under new evidence
- Truth Maintenance Systems — beliefs track justifications; evidence propagates
- Argumentation frameworks (Dung) — beliefs as nodes with attack/support edges
- Stance detection / NLI — the ML primitive: does this fragment support or contradict claim X?

---
## Git changes
```bash
~/Source/PROJECTS/agenticAI/journal_agent
design
    phase11-claim-based-insights.md
    pressure_test_insight_prompts.py
journal_agent
    configure
        prompts
            __init__.py
            claim_regenerator.py
            cluster_fragments.py
            stance_classifier.py
            subject_proposer.py
        config_builder.py
    graph
        nodes
          insight_nodes.py
        reflection_graph.py
        state.py
    model
        insights.py
        session.py
    stores
        __init__.py
        pg_gateway.py
        subjects_repo.py
        tests
        test_phase11_skeleton.py
        journal_agent.log
    sql
        schema.sql
    journal_agent.log
````
---

## Conceptual model

Three layers, each with a distinct mutability profile:

| Layer | Mutability | Role |
|---|---|---|
| **Subject** | label/status mutable, identity stable | the stable handle for an idea (e.g., "stance on Buddhism") |
| **Claim** | new versions appended; one is `is_current` | LLM's current best phrasing of the user's position |
| **Vote** | append-only, can be invalidated | timestamped support/contradict evidence from a specific fragment |

Plus one bookkeeping table:

| `FragmentProcessing` | append-only | "we looked at fragment F with model M at time T; produced N votes" |

### Key invariants

- Stance values: `support` or `contradict`. **Neutral is not a vote** — silence produces no row.
- Strength below 0.3 is omitted, not stored. Noise floor.
- Votes attach to **subject**, not claim. Claim text is mutable; subject identity is stable.
- A fragment may vote on multiple subjects, and may cast both `support` and `contradict` on the same subject (genuine ambivalence).
- Constraint: at most one row per `(subject, fragment, stance)` triple.
- `vote.fragment_dated_at` is the user-write timestamp. **All belief queries filter on this**, never on `processed_at`.
- A new subject is always created together with its first vote.
- Vote-weighting / traction is **derived, not stored**. Score function is a swappable strategy; the data model just stores raw evidence.

### Three kinds of "contradiction" (the nuance that matters)

Plain English "contradicts" hides three different signals:

1. **Position revision** — user has genuinely changed (was Buddhist, now Stoic). *Erodes traction.*
2. **Phrasing drift** — claim text is stale; user is more engaged, not less. Example: claim says "user is curious but inconsistent"; new fragment shows daily 2-month practice. *Should trigger claim regeneration; should NOT erode traction.*
3. **Event noise** — bad mood, single rough day, fluctuation around a stable position. Example: "loves walking the dog" not contradicted by "hated walking last Wednesday." *Should mostly produce no vote (strength below threshold).*

**v1 approach: the prompt distinguishes these.** The data model carries enough context (fragment text, reasoning, signals) to support more sophisticated decomposition later without migration. The structural option (two-signal stance: `position_vote` + `claim_drift`) is documented but not built.

The principle: *the data model is the long-lived investment; the prompt is the cheap iteration loop.* Every prompt improvement compounds on top of votes already collected.

---

## Data model

### `subjects`

| Field | Type | Notes |
|---|---|---|
| id | uuid | |
| label | text | short, human-readable: "stance on Buddhism" |
| description | text | optional longer gloss |
| status | enum | `active`, `dormant`, `superseded`, `merged` |
| parent_subject_id | uuid? | when a subject forks |
| merged_into_id | uuid? | soft merge — queries union both |
| created_at | timestamptz | |
| last_activity_at | timestamptz | denormalized; updated on each new vote |

### `claims`

| Field | Type | Notes |
|---|---|---|
| id | uuid | |
| subject_id | fk | |
| text | text | LLM's current best phrasing |
| version | int | monotonic per subject |
| is_current | bool | exactly one per subject |
| embedding | vector(384) | for fragment routing — embed `label + text` |
| regenerated_at_vote_count | int | how many votes existed when this was generated |
| created_at | timestamptz | |

### `votes`

| Field | Type | Notes |
|---|---|---|
| id | uuid | |
| subject_id | fk | votes attach to subject, not claim |
| fragment_id | fk | the evidence |
| stance | enum | `support`, `contradict` |
| strength | float [0,1] | LLM confidence; rows below 0.3 not written |
| reasoning | text | LLM explanation, citing fragment |
| claim_version_at_vote | fk | which phrasing was evaluated against |
| fragment_dated_at | timestamptz | when user wrote it — drives "as-of" queries |
| processed_at | timestamptz | when vote was cast — audit only |
| model_signature | text | model + prompt version |
| signals | jsonb | length, time-of-day, prompt context — store now, weight on later |
| invalidated_at | timestamptz? | for fragment edits/deletes; never hard-delete |
| invalidation_reason | text? | |

Unique constraint: `(subject_id, fragment_id, stance) WHERE invalidated_at IS NULL`.

### `fragment_processing`

| Field | Type | Notes |
|---|---|---|
| id | uuid | |
| fragment_id | fk | |
| processed_at | timestamptz | |
| model_signature | text | pairs with vote.model_signature |
| vote_count | int | 0 is valid: "looked, nothing fired" |
| status | enum | `success`, `error`, `partial` |
| error_detail | text? | |

Query "give me unprocessed fragments" = LEFT JOIN where no successful row exists.

---

## Derived queries (computed, not stored)

```python
def traction(subject_id, as_of_date, strategy="simple_sum"):
    return aggregate(
        votes
            .filter(subject_id=subject_id)
            .filter(fragment_dated_at <= as_of_date)
            .filter(invalidated_at == None),
        strategy=strategy
    )
```

Vote weighting strategy is a **plug point**, stubbed at v1. Dimensions to grow into:

| Dimension | What it captures |
|---|---|
| Recency decay | newer votes count more |
| Stance strength | LLM's confidence |
| Fragment quality | length, specificity, emotional weight |
| Independence | down-weight votes from same journal entry |
| Source/context | morning vs late-night, prompted vs spontaneous |
| Asymmetric sensitivity | one strong contradiction may matter more than ten weak supports |

v1 strategy: `sum(stance_sign * strength)`. Architecture protects against future migration.

---

## Pipeline shape

Deterministic graph (not ReAct), per-fragment invocation. Matches the no-interrupts pattern.

```
       ┌──────────────────┐
fragment ──▶│ route_candidates │  (vector search over current claim embeddings)
       └──────────────────┘
                ▼
       ┌──────────────────┐
       │ classify_stance  │  ← stance_classifier prompt, structured output
       └──────────────────┘
                ▼
       ┌────────────────────┐
       │ propose_subject    │  ← if no strong votes, run subject_proposer
       │   (conditional)    │
       └────────────────────┘
                ▼
       ┌──────────────────┐
       │ persist_votes    │
       │ + processing row │
       └──────────────────┘
```

**Async / scheduled, separate path:**

```
       ┌──────────────────────┐
       │ regenerate_claim     │  ← when vote_count - last_regen_count > N
       │   per stale subject  │     uses claim_regenerator prompt
       └──────────────────────┘
```

---

## The three prompts

All three are single structured-output LLM calls — no tool loops.

### `stance_classifier`

Decides: for each candidate subject, does this fragment vote support or contradict, with what strength?

Load-bearing prompt instructions:
- "Silence is the default" — empty list is valid and common
- Strength scale anchored: 0.9 / 0.6 / 0.3
- "Strengths below 0.3 — omit the vote instead"
- A fragment may vote on multiple subjects; may cast both stances on one subject

**v1 calibration additions** (post pressure-test):
- Vote relative to the user's **underlying position**, not the literal claim text. Stale phrasing is the regenerator's job.
- A single instance of an activity is **not** a stance signal unless explicitly framed as a value, pattern, or commitment.

### `subject_proposer`

When stance classifier returns no votes, decides: should a NEW subject be created?

Load-bearing instructions:
- Four conditions, all must hold (belief/value/pattern, not event; falsifiable; not already covered; expected to recur)
- "Bias toward NOT creating"
- Returns null OR `{label, description, initial_claim, initial_vote}`

### `claim_regenerator`

When a subject has accumulated N new votes since the last regeneration, decides: should the claim text be rewritten?

Load-bearing instructions:
- Reflect user's **current** state, not historical average
- Recent strong evidence dominates older weak evidence
- Falsifiable phrasing
- Fork escape valve: flag `fork_suggested` if evidence has split into two distinct ideas

---

## Pressure-test results (2026-04-28)

5 hand-crafted tests against `claude-sonnet-4-5`. Results:

| # | Test | Result |
|---|---|---|
| 1 | Mundane fragment, stance classifier | ❌ Over-eager: voted support 0.6 on health from a single gym mention. Fix in v1 prompt with calibration text. |
| 2 | Clear support fragment, stance classifier | ⚠️ Surfaced position-vs-phrasing nuance. Fix in v1 prompt. |
| 3 | One-off mood, subject proposer | ✅ Returned null. Bias-toward-NOT-creating held. |
| 4 | Clear new theme, subject proposer | ✅ Proposed "relationship with father" with sharp initial claim. |
| 5 | Trajectory regeneration | ✅ Pitch-perfect. "user has moved away from Buddhist practice…". No averaging. |

Subject proposer and claim regenerator: ship-ready. Stance classifier: needs the v1 calibration additions documented above; re-test after schema lands.

---

## Coexistence with Phase 10

Phase 11 lands **alongside** Phase 10 — does not replace until proven. Specifically:

- Existing `insights` and `insight_fragments` tables remain.
- `InsightsRepository` remains.
- Phase 10 cluster nodes (`make_create_clusters`, `make_label_clusters`, `make_verify_citations`) remain in place but unwired from the new reflection graph.
- The reflect node's behavior changes: per-fragment invocation of the new graph instead of batch clustering.
- Old `load_unprocessed_fragments` (defined as "no row in `insight_fragments`") remains for any old-pipeline replay tooling. The new pipeline uses `fragment_processing` exclusively.

Migration of historical insights into the new model is deferred — decide after we have confidence in the new pipeline.

---

## Build sequence (six PRs)

1. **Schema + models** — new tables, new pydantic models, repos with no callers
2. **Prompts** — three prompt modules + PromptKey entries, smoke tests
3. **Node factories** — new node code, unit-tested against fake LLM clients
4. **Reflection graph** — wire the nodes, end-to-end test against fixture fragments
5. **Reflect node integration** — `make_reflect_node` invokes new graph; old code path dead but present
6. **Cleanup** — delete cluster nodes, old prompts, dead state fields, dead repo code

---

## Working mode

For each PR, lay out the scaffolding first — file stubs, function signatures, docstrings — then fill in. The user reviews the scaffolding and chooses which implementations to write themselves vs hand off. Database access is structured as if it were a tool surface (clean methods, single responsibility), even though it's called directly rather than via tool-use protocol.

---

## Open questions deferred to later phases

- **Claim drift signal** as a structural addition (the two-signal stance approach). v1 handles via prompt; revisit if calibration proves brittle.
- **Subject merging UX** — when two subjects turn out to be the same, soft merge via `merged_into_id`. Implementation deferred; not needed for v1.
- **Subject splitting / forking** — `fork_suggested` from regenerator is recorded but no automated split flow yet.
- **Cross-subject reasoning** — edges between subjects (e.g., "user values discipline" supports "user is drawn to Buddhism"). Argumentation framework territory. Out of v1.
- **Weighting strategy v2+** — recency decay, independence, asymmetric sensitivity. Stub now, build later.
- **Historical cluster-based insight migration** — decide after the new pipeline has proven itself.
