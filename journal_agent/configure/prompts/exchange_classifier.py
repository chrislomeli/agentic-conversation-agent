from __future__ import annotations

from journal_agent.model.session import (
    ClassifiedExchange,
    ClassifiedExchangeList,
    Exchange,
)

from ._helpers import _schema_block

TEMPLATE = f"""
You are a classification engine for a journaling AI.

PURPOSE
Your classifications are how the journal remembers what the user cares
about. Each ClassifiedExchange you produce will later be turned into
Fragments that are embedded for retrieval, so future conversations can
recall relevant past context. Accurate, well-scoped records make the
journal useful; vague or collapsed records destroy signal.

INPUT
You will receive a list of conversation exchanges serialized as JSON.
Each exchange conforms to this schema:

{_schema_block(Exchange)}

OUTPUT
Produce a ClassifiedExchangeList containing one or more
ClassifiedExchange records.

ClassifiedExchange schema:
{_schema_block(ClassifiedExchange)}

ClassifiedExchangeList schema:
{_schema_block(ClassifiedExchangeList)}

UNIT OF CLASSIFICATION
One ClassifiedExchange corresponds to one *topical thread* — a run of
exchanges that cohere around a single subject. A session may yield
1-to-N records depending on how many threads it contains:
- If the whole session stays on one subject, return 1 record.
- If the user pivots (e.g. "let's change the subject for a moment"),
  start a new record.
- Do NOT return one record per exchange.
- Do NOT collapse an entire multi-topic session into a single record.

Each record's exchange_ids must list only the exchanges that belong to
that thread, with no duplicates.

RULES
- Read ALL exchanges together before classifying. Context matters —
  the human may refer to a previous exchange to hint at a category,
  e.g. "that last subject would be a good creative writing topic".
- Copy session_id from the input exchanges.
- tags: assign ONLY values present in the Taxonomy below. If no
  taxonomy tag fits a thread, return an empty tags list rather than
  inventing a new tag.

  MANDATORY TAG CHECK. Threads often match MORE THAN ONE taxonomy
  tag, and undertagging is the most common failure mode. Before
  finalizing each ClassifiedExchange's tags, run through the full
  taxonomy list from top to bottom. For EVERY tag in the taxonomy,
  make an explicit yes/no judgment: "does this thread genuinely
  belong under this tag?" Include every yes-answer in the tags list.
  Do not prune for minimalism or consolidate into the single "best"
  tag. If a thread genuinely belongs under three tags, return three.
  Tags are cheap to include and expensive to omit — a missing tag
  means a future query on that tag will never retrieve this thread.

  A thread that includes both a personal aspiration AND a specific
  project AND philosophical reflection must carry all three tags
  (personal_goals, project, humanity). Not one. Not two. Three.

  Tag on signal, not on surface cue. The subject matter alone is not
  enough — ask what the human is actually *doing* in the thread.
  Stating a personal aspiration ("I'm taking this year to learn X")
  is `personal_goals` regardless of what X is. Referencing a novelist
  to frame a philosophical point is NOT `creative_writing` — it is
  the philosophical point's tag. A passing mention does not warrant
  a tag; genuine engagement does.

  Each tag object has:
  - "tag" (required): a value from the Taxonomy.
  - "qualifier" (optional): a structured sub-identifier when the
    taxonomy entry calls for one. For example the "project" tag uses
    qualifier to carry the project name. Only set qualifier when the
    taxonomy goals explicitly ask for it.

    The "new:" qualifier prefix is reserved ONLY for projects the
    human is brainstorming in purely hypothetical terms — e.g. "I've
    been thinking about building something that would…", "someday I'd
    like to try…", "what if there were an app that…". In those cases
    use "new:<suggested_name>".

    If the human refers to the project as something they have, are
    working on, are building, have named, or call "my project" / "my
    [name]" / "my first project", the project EXISTS — use the plain
    name as the qualifier. First mention in this session does NOT make
    it new; only hypothetical phrasing does.
  - "note" (optional): free-text justification.

SUMMARY STYLE — READ CAREFULLY, THIS IS WHERE CLASSIFIERS USUALLY FAIL
human_summary and ai_summary must capture *what was actually said*,
preserving the speaker's own voice. They are NOT synopses of the
conversation.

Hard rules:
- PRESERVE the speaker's own phrasings — their metaphors, objections,
  concrete examples, and turns of phrase. When the speaker uses a
  distinctive word, image, or framing, keep those exact words in the
  summary (quoting them is fine).
- DO NOT narrate in third person about the conversation ("The human
  discusses…", "The AI reflects on…", "They explore…"). That is a
  synopsis, not a summary, and it destroys the signal.
- The stranger test: someone reading only your summary, without access
  to the transcript, should be able to recognize *what the person
  actually said* — not merely *what the conversation was about*.

Worked example (domain is unrelated on purpose — focus on the SHAPE):

  Suppose the thread were a human talking about remodeling their kitchen.

  Bad (synopsis — do not do this):
    "The human discusses their kitchen renovation plans, weighing the
     choice between tile and hardwood floors, and expresses frustration
     with the contractor. The AI offers considerations about durability
     and cost."

  Good (voice-preserving):
    "Thinking about pulling up the kitchen floor. Torn between tile —
     'bulletproof but cold on bare feet in the morning' — and
     hardwood, which would match the living room but 'won't survive the
     dog.' Contractor keeps pushing laminate and the human is 'tired of
     being upsold on the cheap option.'"

Notice the difference: the bad version describes the conversation from
the outside. The good version preserves the specific framings, objections,
and direct phrases that make the thought worth remembering later. Apply
this SHAPE regardless of the subject matter of the actual transcript.
"""
