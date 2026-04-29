# Phase 11 — Deferred TODOs

---

## P1 — Fix before publishing

~~All P1 items are resolved.~~

---

~~**Race condition in `regenerate_claim`** — fixed.~~

`SELECT FOR UPDATE` on the subject row now serialises concurrent regenerations. Version is computed via `INSERT … SELECT COALESCE(MAX(version), 0) + 1` inside the same transaction, so no two callers can observe the same version number.

---

## P2 — Deliberate tradeoffs

These items have a "right" solution that we're aware of and consciously not building yet. Each has a one-sentence explanation that holds up in a room.

---

**`propose_subject` loads all active subjects per node call**

The full active-subjects list is sent to the LLM every time the proposer fires, regardless of how many subjects exist. Above ~100 subjects this bloats the prompt and burns tokens on ideas already ruled out by the router.

*The tradeoff:* The router already filters by embedding similarity. Adding a second similarity filter at the proposer step is the right fix, but at v1 with a handful of subjects, the duplicated context costs nothing and simplifies the code. Add the filter when subject count makes it visible.

---

**`compute_traction` produces unnormalized scalars**

`SubjectSnapshot.traction` is a raw `sum(strength × sign)` with no scale ceiling. A subject with 100 weak supports scores 60.0; one with 3 strong supports scores 2.4. The numbers are technically correct but have no shared frame of reference for consumers (UI, LLM narration).

*The tradeoff:* Normalization is policy, and the policy is intentionally a plug point (`compute_traction` dispatch table). Locking in a normalization scheme before the data is real would be premature. The raw values are still directionally useful for dogfooding; revisit once real usage patterns are visible.

---

**Cosine similarity reimplemented in `make_persist_votes`**

The dedup step computes `np.dot / (norm_i × norm_j)` inline instead of sharing a helper with `search_candidate_subjects`. If the Embedder returns L2-normalized vectors (likely), `np.dot` alone is sufficient and the division is wasted.

*The tradeoff:* Two lines of math that work correctly. Not worth a shared utility until there's a third callsite or a verified normalization contract on the Embedder. Noted here so it doesn't get copied a third time.

---

**Phase 10 repos still use the fat-gateway pattern**

`FragmentRepository`, `ThreadsRepository`, `TranscriptRepository`, and `UserProfileRepository` still delegate SQL to `PgGateway` tier-2 methods. The Phase 11 repos own their SQL directly (cleaner). The Phase 10 repos are technically inconsistent.

*The tradeoff:* Phase 10 cluster nodes are scheduled for deletion once Phase 11 proves itself. Refactoring repos that are about to be retired is waste. Retrofit opportunistically — when you next touch one of these repos for a real reason, bring the SQL with you.

---

## P3 — Architectural foresight

These are not bugs or tradeoffs. They are design decisions that were deliberately deferred because the infrastructure for them isn't warranted yet. Knowing these exist — and why they were left out — is what separates an architect from a developer who ran out of time.

In a room of architects, these are the items that open the best conversations.

---

**Weighting strategy v2+**

The current `simple_sum` strategy in `compute_traction` ignores recency, fragment quality, vote independence, and asymmetric sensitivity (one strong contradiction may matter more than ten weak supports). These are all documented in the design doc as known dimensions.

*Why deferred:* The data model already stores everything needed to implement any of these strategies — `fragment_dated_at`, `strength`, `signals` (jsonb), `model_signature`. The plug point is clean. Building a sophisticated weighting scheme before seeing real vote distributions would be model-driven development without data. The architecture protects against migration; the algorithm can compound on top of existing votes when the time comes.

---

**Subject merging and forking**

`merged_into_id` and `parent_subject_id` are in the schema. `fork_suggested` is a valid `RegeneratorAction`. Neither has an implementation behind it.

*Why deferred:* Merging and forking are identity management operations on a graph of evolving beliefs. Getting them wrong silently corrupts the traction history. The schema design (soft merge via `merged_into_id`, union semantics on queries) is already correct for the merge case. Forking requires a UX decision about what happens to votes cast before the split. Both decisions should be driven by observed subject behavior during dogfooding, not by speculation.

---

**Cross-subject reasoning**

Subjects currently accumulate evidence independently. The next layer — "does 'user values discipline' support 'user is drawn to Buddhism'?" — requires edges between subjects with attack/support semantics. This is argumentation framework territory (Dung, 1995; Abstract Argumentation).

*Why deferred:* Single-subject traction isn't proven yet. Adding inter-subject edges before the base layer is calibrated would couple two unvalidated systems. The schema doesn't need to change when this arrives — it's a new table of (subject_id_a, subject_id_b, relationship_type). The design space is well-understood; the trigger is proven single-subject value.

---

*Note: Unit tests for the four Phase 11 graph nodes were completed in the cleanup pass and are no longer deferred — see `tests/test_phase11_nodes.py`.*
