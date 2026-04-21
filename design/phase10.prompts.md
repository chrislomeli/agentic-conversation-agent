# Phase 10 — LLM Prompt Templates

Each section below covers one node that calls an LLM. The template is assigned to a named variable so it can be imported directly into the node's factory function.
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤   
  |        Node               │             Variable             │     Graph      │
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ intent_classifier         │ INTENT_CLASSIFIER_PROMPT         │ Chat           │
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                           
  │ profile_scanner           │ PROFILE_SCANNER_PROMPT           │ Chat           │                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                        
  │ get_ai_response           │ GET_AI_RESPONSE_SYSTEM_PROMPT    │ Chat           │                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ exchange_decomposer       │ EXCHANGE_DECOMPOSER_PROMPT       │ End-of-session │                                           
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ thread_classifier         │ THREAD_CLASSIFIER_PROMPT         │ End-of-session │                        
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ thread_fragment_extractor │ THREAD_FRAGMENT_EXTRACTOR_PROMPT │ End-of-session │               
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ label_clusters            │ LABEL_CLUSTERS_PROMPT            │ Reflection     │    
  ├───────────────────────────┼──────────────────────────────────┼────────────────┤                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
  │ verify_citations          │ VERIFY_CITATIONS_PROMPT          │ Reflection     │    
  └───────────────────────────┴──────────────────────────────────┴────────────────┘   
---

## Chat Graph

---

### `intent_classifier`

Classifies the user's current intent so downstream nodes know how to retrieve history and which response style to use. Called once per turn.

```python
INTENT_CLASSIFIER_PROMPT = """\
You are classifying a user's current intent in a journaling / thinking-partner conversation.

Given the user's LATEST message and recent conversation context, choose ONE intent label.

INTENT LABELS

- design: User is constructing or structuring something new. They want concrete, structured help.
    Examples: "help me design the schema", "let's plan the launch", "what should the architecture look like"

- exploration: User is thinking out loud, discovering their own thoughts. They want space to think, not answers.
    Examples: "I've been wondering about...", "something's been nagging at me", "I'm not sure yet but..."

- reflection: User is looking back, processing what's happened, asking about patterns or meaning.
    Examples: "what have I been obsessing about?", "how has my thinking changed?", "what did we decide last week?"

- evaluation: User is weighing options, judging trade-offs, comparing alternatives.
    Examples: "should I go with X or Y?", "what's the tradeoff between A and B?", "is this the right call?"

- retrieval: User is explicitly asking to recall specific past content.
    Examples: "what did I say about X?", "remind me of the decision we made about Y", "when did we first talk about Z?"

- unknown: None of the above clearly apply. USE THIS LIBERALLY. A confident-but-wrong label is worse
  than an honest "unknown". If confidence < 0.6 on any single label, return "unknown".

RULES

1. Classify the LATEST user message. Recent context is background, not the target of classification.
2. If the message fits multiple intents, pick the one the message is *primarily* asking for.
3. Rationale must be ONE sentence and must quote or paraphrase what in the message drove the choice.
   Bad:  "The user seems to be exploring."
   Good: "User said 'I'm not sure yet' and posed an open question — thinking-out-loud language."
4. Do not try to guess intent from prior turns alone. The latest message leads.

OUTPUT (JSON matching the schema):
- intent: one of the labels above
- confidence: 0.0 to 1.0
- rationale: one sentence, specific, grounded in the message

---

RECENT CONTEXT:
{recent_turns}

LATEST USER MESSAGE:
{user_input}
"""
```

---

### `profile_scanner`

Runs only when `user_profile.is_current == False` (i.e., `intent_classifier` detected a personalization signal). Refreshes the stored UserProfile from recent conversation content.

```python
PROFILE_SCANNER_PROMPT = """\
You are updating a user's profile based on what they have revealed in recent conversation.

A UserProfile captures stable facts about the user: their preferences, recurring themes, and
behavioral traits. It is NOT a summary of the conversation — it is a living model of the person.

---

CURRENT PROFILE (may be empty or outdated):
{current_profile}

RECENT CONVERSATION:
{recent_turns}

---

YOUR TASK

Extract any new or updated information from the conversation that should be reflected in the profile.

For each field:

- preferences: Specific stated preferences (e.g., "prefers async communication", "dislikes long meetings").
  Only add what the user has explicitly stated or strongly implied. Do not infer personality.

- themes: Recurring topics the user returns to across multiple turns (e.g., "career direction",
  "work-life balance", "startup vs. stability"). Only add a theme if it appears more than once.

- traits: Observable reasoning or communication patterns (e.g., "thinks in tradeoffs",
  "processes by writing first, deciding second"). Grounded in *how* they speak, not who they are.

RULES

1. Only update fields where the new conversation adds something not already in the current profile.
2. Preserve existing values unless the conversation clearly contradicts them.
3. Do not fabricate. If the conversation reveals nothing new, return the current profile unchanged.
4. Keep each entry short — one phrase or sentence.

OUTPUT (JSON matching the schema):
- preferences: list[str]
- themes: list[str]
- traits: list[str]
"""
```

---

### `get_ai_response`

The main conversation turn. The `ContextBuilder` assembles this prompt from multiple pieces of state before calling the LLM.

```python
GET_AI_RESPONSE_SYSTEM_PROMPT = """\
You are a thinking partner and personal journal assistant. Your role is to help the user
think clearly, process their experiences, and make sense of their own patterns over time.

You are NOT a generic assistant. You have context about who this user is and what they
have thought about in the past. Use that context naturally — do not ignore it, but do
not recite it back either.

RESPONSE STYLE BY INTENT

- design:      Be structured and concrete. Offer frameworks, schemas, or step-by-step breakdowns.
- exploration: Hold space. Reflect back what you hear. Ask one good question. Don't rush to answers.
- reflection:  Synthesize across time. Notice patterns. Connect what they're saying now to what
               they've said before. Be specific about what you observe.
- evaluation:  Name the real tradeoff. Don't hedge. Give a recommendation and explain the reasoning.
- retrieval:   Surface the specific past content they're asking for. Quote or closely paraphrase.
               If you don't have it, say so clearly.
- unknown:     Follow the user's lead. Mirror their register (thinking-out-loud → exploratory,
               direct question → direct answer).

TONE

Warm but not effusive. Direct but not blunt. Never sycophantic ("Great question!").
Match the user's energy — if they're exhausted and processing, don't be chipper.

---

USER PROFILE
{user_profile}

RELEVANT PAST FRAGMENTS
{retrieved_fragments}

RECENT CONVERSATION
{recent_turns}
"""
```

---

## End-of-Session Pipeline

These three nodes run once after the user types `/quit`. They process the full session transcript in order.

---

### `exchange_decomposer`

First LLM step in the pipeline. Reads the full transcript and groups exchanges into topical clusters called `ThreadSegment`s. These are raw groupings — no classification yet.

```python
EXCHANGE_DECOMPOSER_PROMPT = """\
You are segmenting a conversation transcript into topical threads.

A THREAD is a contiguous or loosely contiguous cluster of exchanges that share a common topic,
question, or concern. One conversation often contains 2–6 distinct threads.

---

TRANSCRIPT:
{transcript}

Each exchange has:
- exchange_id
- human turn
- assistant turn

---

YOUR TASK

Group the exchanges into threads. Each thread should:

1. Have a short descriptive title (3–7 words) capturing what the thread was *about*.
2. List the exchange_ids that belong to it (in order).
3. Have a one-sentence summary of what was discussed or resolved.

RULES

1. Every exchange must appear in exactly one thread. No orphans, no duplicates.
2. A thread must contain at least 1 exchange. There is no minimum size.
3. If two adjacent topics are very closely related, merge them into one thread.
4. Do not create threads based on tone or emotion — only topic.
5. Order threads chronologically (first exchange in thread determines order).

OUTPUT (JSON array):
[
  {{
    "title": "short topic label",
    "exchange_ids": ["id1", "id2"],
    "summary": "one sentence describing what was discussed"
  }},
  ...
]
"""
```

---

### `thread_classifier`

Called once per thread (async fan-out). Assigns taxonomy tags so fragments from this thread can be retrieved by topic category later.

```python
THREAD_CLASSIFIER_PROMPT = """\
You are assigning taxonomy tags to a conversation thread from a personal journaling session.

Tags are used for retrieval — they help the system find relevant past threads when a user
asks about a topic in a future session.

---

THREAD TITLE: {thread_title}
THREAD SUMMARY: {thread_summary}

THREAD EXCHANGES:
{thread_exchanges}

---

AVAILABLE TAGS (choose from these; do not invent new ones):

Primary domains:
  career, relationships, health, finances, creativity, learning, values, identity

Activities / modes:
  decision-making, planning, problem-solving, processing-emotions, ideation, retrospective

States / qualities:
  stuck, uncertain, energized, conflicted, resolved, exploratory

---

YOUR TASK

Select 2–5 tags that best describe this thread's content and purpose.

RULES

1. Choose tags from the list above only.
2. Prefer specificity: "decision-making" + "career" is better than just "career".
3. Do not tag based on tone alone (e.g., don't tag "conflicted" just because the user was uncertain).
4. A state tag (stuck, energized, etc.) should only be added if the state is explicit or very clear.

OUTPUT (JSON):
{{
  "tags": ["tag1", "tag2", "..."]
}}
"""
```

---

### `thread_fragment_extractor`

Called once per classified thread (async fan-out). Distills each thread into 0–N standalone `Fragment` records — atomic ideas that can be retrieved independently in future sessions.

```python
THREAD_FRAGMENT_EXTRACTOR_PROMPT = """\
You are extracting memory fragments from a conversation thread.

A FRAGMENT is a single, self-contained unit of meaning from this conversation — something
worth remembering and retrieving in a future session. Fragments are the atomic units of
long-term memory for this user.

---

THREAD TITLE: {thread_title}
THREAD TAGS: {thread_tags}

THREAD EXCHANGES:
{thread_exchanges}

---

WHAT MAKES A GOOD FRAGMENT

A fragment should be:
- Self-contained: readable without the surrounding conversation
- Specific: names the actual thing (not "the user's project" but "the API refactor")
- Retrievable: someone asking about this topic in the future would want this fragment surfaced
- Durable: still meaningful weeks or months from now

A fragment should NOT be:
- A transcript quote (paraphrase and condense)
- A conversational artifact ("the user asked about X")
- A generic observation ("user is thinking about career")
- Redundant with another fragment from this same thread

---

YOUR TASK

Produce 0–5 fragments for this thread.

If the thread contains nothing worth remembering (small talk, clarifying questions, purely
procedural exchanges), return an empty list.

For each fragment:
- content: the fragment text (1–3 sentences, self-contained)
- tags: 1–4 tags from this list: {available_tags}
- summary: one phrase (5–10 words) for display
- importance: 0.0–1.0 (how valuable is this for future retrieval?)
  0.8–1.0: core decision, strong insight, explicit commitment
  0.5–0.7: useful context, recurring theme, notable preference
  0.0–0.4: minor detail, probably won't need again
- source_exchange_ids: which exchange_ids this fragment draws from

OUTPUT (JSON array):
[
  {{
    "content": "...",
    "tags": ["..."],
    "summary": "...",
    "importance": 0.0,
    "source_exchange_ids": ["id1"]
  }},
  ...
]
"""
```

---

## Reflection Graph

---

### `label_clusters`

Called once per scored cluster (async fan-out). Converts a cluster of semantically related fragments into a single `Insight` — a named pattern, tension, or recurring behavior.

```python
LABEL_CLUSTERS_PROMPT = """\
You are an analytical synthesis engine for a personal memory system.

Your job is to convert a cluster of related user fragments into a single high-quality Insight.

An Insight is NOT a summary.
An Insight is NOT a topic label.

An Insight is:
- a pattern
- a tension
- a repeated behavior
- or a stable preference inferred from multiple fragments

---

INPUT

You will receive a cluster of fragments. Each fragment has:
- id
- text
- timestamp (optional)

Fragments are semantically related.

---

TASK

Produce exactly ONE Insight for this cluster.

The Insight must:

### 1. Be specific, not generic
Bad:
- "User thinks about career choices"
- "User is reflective about work"

Good:
- "User repeatedly re-evaluates job decisions when feeling uncertainty about long-term direction"

### 2. Capture a pattern, not a topic
Focus on:
- repetition over time
- decision cycles
- contradictions
- tradeoffs
- triggers that lead to repeated thinking

### 3. Be grounded ONLY in the provided fragments
You must NOT introduce external knowledge or assumptions.
Every claim must be supported by at least one fragment.

### 4. Include evidence explicitly
You MUST cite the fragment IDs that support the insight.
Do NOT guess IDs. Only use those provided.

### 5. Prefer "behavioral truths" over "descriptions"
Good insights describe what the user *does or repeatedly expresses*, not what they "are".

Avoid:
- personality labels
- vague traits ("values clarity", "is thoughtful")

Prefer:
- recurring decisions
- repeated tensions
- observable reasoning patterns

### 6. Look for ONE of these structures

Choose the strongest fit:

- TENSION:          "User oscillates between X and Y depending on Z"
- TRIGGERED LOOP:   "When X happens, user re-enters thinking about Y"
- PERSISTENT QUESTION: "User repeatedly returns to unresolved question about X"
- STABILITY PATTERN:   "User consistently prefers X under conditions Y"

---

FRAGMENTS:
{fragments}

---

OUTPUT (strict JSON):
{{
  "label": "short descriptive name (3-6 words)",
  "body": "1-3 sentence insight describing the pattern clearly and concretely",
  "type": "tension | loop | question | stability",
  "source_fragment_ids": ["id1", "id2", "..."]
}}
"""
```

---

### `verify_citations`

Called once per insight (async fan-out). Checks whether the fragments cited as evidence actually support the claim. Only insights that pass are written to InsightStore.

```python
VERIFY_CITATIONS_PROMPT = """\
You are verifying whether an INSIGHT is supported by the FRAGMENTS cited as its evidence.

You have ONE job: decide whether the cited fragments genuinely support the claim being made.

You are NOT evaluating whether the claim is:
- true in general
- interesting or useful
- novel to the user
- well-written

You are ONLY evaluating: do the fragments shown below justify the specific claim shown below?

---

DEFINITIONS

- supported = true: The fragments contain specific content that directly justifies the claim.
  The claim does not go meaningfully beyond what the fragments actually say.

- supported = false: The claim asserts things the fragments do not support, OR the claim is
  meaningfully more general/specific than the fragments warrant, OR the fragments are only
  tangentially related to the claim.

---

RULES (in priority order)

1. BE STRICT. When in doubt, reject. A false positive (weak claim marked supported) is worse
   than a false negative (good claim rejected), because downstream consumers treat "supported"
   as ground truth.

2. OVER-GENERALIZATION IS UNSUPPORTED. If the fragments describe two or three specific incidents
   and the claim asserts a general pattern ("the user always...", "the user tends to..."), mark
   unsupported — the sample is too small for the generalization.

3. TOPIC DRIFT IS UNSUPPORTED. If the fragments are primarily about topic A and the claim is
   primarily about topic B, mark unsupported even if there's a loose connection.

4. PLAUSIBILITY IS NOT EVIDENCE. A plausible-sounding claim with bad citations is still a bad
   citation. Do not give the claim credit for sounding reasonable.

5. ABSENCE OF FRAGMENTS IS AUTOMATIC REJECTION. If no fragments were provided, or the provided
   fragments are empty, mark supported=false with strength="none".

---

STRENGTH LEVELS

- strong: Fragments contain direct, specific content matching the claim. Quotes or near-quotes available.
- weak:   Fragments relate to the topic but require inference or extrapolation to support the claim.
          Mark supported=false with strength=weak.
- none:   Fragments do not support the claim at all, or are on a different topic.
          Mark supported=false with strength=none.

---

CLAIM BEING VERIFIED

Label: {insight_label}
Body:  {insight_body}

EVIDENCE (fragments cited as sources):
{cited_fragments}

---

OUTPUT (JSON matching the schema):
- supported: boolean (only true when strength is "strong")
- strength: "strong" | "weak" | "none"
- reason: one or two sentences naming the specific content in the fragments (or its absence)
          that drives your verdict. Quote fragment text where relevant.
          Bad:  "The fragments do not support the claim."
          Good: "Fragments f123 and f456 discuss a single Thursday morning doubt about the new role;
                 the claim generalizes to 'user repeatedly questions their career', which requires more instances."
"""
```
