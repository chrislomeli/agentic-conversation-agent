"""seed_fragments.py — load a test corpus of fragments into Postgres.

Run: ``uv run python -m journal_agent.scripts.seed_fragments``

Produces ~70 fragments across 8 themes so HDBSCAN and score_cluster
have something meaningful to work with:

    A — weighing a career pivot (IC → management)         30-day span
    B — marathon training and morning discipline           29-day span
    C — caring for an aging parent                        30-day span
    D — kitchen renovation                                86-day span  ← wide
    E — creative writing sprint                           7-day span   ← tight burst
    F — chronic back pain                                 83-day span  ← wide
    G — friendship drift                                  51-day span  ← medium
    H — low-signal surface noise                          4-day span   ← junk cluster

The span variation means score_cluster produces a real distribution:
    D ≈ 53  F ≈ 51  G ≈ 35  A/B/C ≈ 25  E ≈ 11  H ≈ 6

Each fragment has:
    - fixed fragment_id (seed_<theme>_<nn>) — ON CONFLICT upsert = idempotent
    - deterministic session_id (seed_sess_<nn>) — sessions row auto-created
    - timestamp spread across up to 90 days
    - content + tags

Embeddings are computed in-script by PgFragmentRepository (fastembed, 384-dim).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from journal_agent.model.session import Fragment, Tag
from journal_agent.stores.fragment_repo import FragmentRepository

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Seed corpus — edit content freely.  Keep fragment_ids stable for idempotency.
# ═══════════════════════════════════════════════════════════════════════════════

# Anchor: all timestamps are ANCHOR - days_ago days.
ANCHOR = datetime(2026, 4, 20, 9, 0, tzinfo=timezone.utc)


@dataclass
class Seed:
    fragment_id: str
    session_id: str
    content: str
    tags: list[str]
    days_ago: int  # timestamp = ANCHOR - days_ago


# ── Theme A: career pivot (IC → management) ── span ~29 days, size 10 ─────────
CAREER = [
    Seed("seed_a_01", "seed_sess_01",
         "Had a skip-level with the director today. She asked if I'd ever considered "
         "the management track. I've been circling this for six months and still don't "
         "have a clean answer.",
         ["career", "identity"], days_ago=30),
    Seed("seed_a_02", "seed_sess_02",
         "The thing I'll miss most is head-down coding days. When I'm in flow I lose "
         "track of hours. Managers don't get those hours back.",
         ["career", "identity"], days_ago=28),
    Seed("seed_a_03", "seed_sess_03",
         "Three direct reports is different than zero. Not sure I want to spend my day "
         "unblocking other people instead of building.",
         ["career", "decision"], days_ago=25),
    Seed("seed_a_04", "seed_sess_04",
         "Talked to Pav who made the jump last year. He said the first six months were "
         "brutal — grief for the IC identity.",
         ["career", "identity"], days_ago=22),
    Seed("seed_a_05", "seed_sess_05",
         "If I turn this down twice, the offer won't come a third time. That's the part "
         "that weighs on me.",
         ["career", "decision"], days_ago=18),
    Seed("seed_a_06", "seed_sess_06",
         "Autonomy is the word I keep coming back to. As IC I control my day. As manager "
         "I control the direction. Which one do I actually want?",
         ["career", "decision"], days_ago=14),
    Seed("seed_a_07", "seed_sess_07",
         "Realized I don't want to be a manager — I want to be the kind of manager I "
         "wish I had. Not sure the company is set up for that.",
         ["career", "identity"], days_ago=10),
    Seed("seed_a_08", "seed_sess_08",
         "Comp upside is real. Not the main thing but not nothing.",
         ["career"], days_ago=7),
    Seed("seed_a_09", "seed_sess_09",
         "Wife asked me tonight: 'what would you regret more — saying yes or saying no?' "
         "I didn't have an answer.",
         ["career", "decision"], days_ago=4),
    Seed("seed_a_10", "seed_sess_10",
         "Decided I'll give it 18 months. If I hate it by then I can IC back. People "
         "do it.",
         ["career", "decision"], days_ago=1),
]

# ── Theme B: marathon training / morning discipline ── span ~29 days, size 10 ──
RUNNING = [
    Seed("seed_b_01", "seed_sess_01",
         "Missed the 5am alarm three times this week. The training plan assumes morning "
         "runs and I keep pushing them to evening.",
         ["running", "discipline"], days_ago=29),
    Seed("seed_b_02", "seed_sess_02",
         "Long run Sunday: 16 miles, Z2 the whole way. Pace felt sustainable. First time "
         "I could imagine actually running a marathon.",
         ["running", "training"], days_ago=27),
    Seed("seed_b_03", "seed_sess_03",
         "Split times from today's tempo: 7:45, 7:32, 7:28. Still slowing in the middle.",
         ["running", "training"], days_ago=24),
    Seed("seed_b_04", "seed_sess_04",
         "Knee twinge on mile 9. Going to take tomorrow as a full recovery day, not just "
         "an easy run.",
         ["running", "training"], days_ago=21),
    Seed("seed_b_05", "seed_sess_05",
         "Read that marathon success is 80% the easy miles. I keep wanting to hammer "
         "every run.",
         ["running", "discipline"], days_ago=17),
    Seed("seed_b_06", "seed_sess_06",
         "New shoes came in. Feel weird for the first two miles and then disappear.",
         ["running"], days_ago=13),
    Seed("seed_b_07", "seed_sess_07",
         "Hit a PR in the 10k this morning. Didn't even feel like a hard effort. "
         "That's encouraging.",
         ["running", "training"], days_ago=9),
    Seed("seed_b_08", "seed_sess_08",
         "Two months out from race day. Mileage ramps from 35 to 50 over the next four "
         "weeks.",
         ["running", "training"], days_ago=6),
    Seed("seed_b_09", "seed_sess_09",
         "Morning routine locked in: coffee, banana with nut butter, out the door in 20 "
         "minutes. Keeps me from thinking about it.",
         ["running", "discipline"], days_ago=3),
    Seed("seed_b_10", "seed_sess_10",
         "Talked to coach about nutrition on long runs. I keep bonking around mile 12.",
         ["running", "training"], days_ago=0),
]

# ── Theme C: caring for an aging parent ── span ~30 days, size 10 ─────────────
ELDERCARE = [
    Seed("seed_c_01", "seed_sess_01",
         "Mom's doctor appointment today. The cognitive screening was a half point worse "
         "than last time. Not a cliff but not flat either.",
         ["family", "caregiving"], days_ago=30),
    Seed("seed_c_02", "seed_sess_02",
         "Toured two assisted living places near our house. The second one smelled like "
         "a hospital. She won't go to that one.",
         ["family", "caregiving"], days_ago=26),
    Seed("seed_c_03", "seed_sess_04",
         "Role reversal feels like the right name for it. She used to pack my lunch. "
         "Today I reminded her to take her pills.",
         ["family", "caregiving"], days_ago=23),
    Seed("seed_c_04", "seed_sess_05",
         "She won't admit she can't drive anymore. I'm going to have to have that "
         "conversation this month.",
         ["family", "caregiving"], days_ago=20),
    Seed("seed_c_05", "seed_sess_06",
         "Scheduled her follow-up with the neurologist for the 30th. Siblings said "
         "they'd rotate showing up.",
         ["family", "caregiving"], days_ago=16),
    Seed("seed_c_06", "seed_sess_07",
         "Felt guilty leaving after the visit today. Like I should've stayed another hour.",
         ["family", "caregiving"], days_ago=12),
    Seed("seed_c_07", "seed_sess_08",
         "Found old photos at her place. She was 32 in one of them — my age. Weird "
         "symmetry.",
         ["family"], days_ago=8),
    Seed("seed_c_08", "seed_sess_09",
         "Power of attorney paperwork is sitting on my desk. Keep not filling it in.",
         ["family", "caregiving"], days_ago=5),
    Seed("seed_c_09", "seed_sess_10",
         "She told me 'I don't want to be a burden.' I couldn't respond. Cried on the "
         "drive home.",
         ["family", "caregiving"], days_ago=2),
    Seed("seed_c_10", "seed_sess_10",
         "Sister offered to take next month's appointments. Quietly relieved.",
         ["family", "caregiving"], days_ago=0),
]

# ── Theme D: kitchen renovation ── span ~86 days, size 10 — HIGH score ─────────
RENOVATION = [
    Seed("seed_d_01", "seed_sess_11",
         "Got three contractor quotes for the kitchen. Spread of $28k to $61k for the "
         "same scope. No idea how to evaluate which one is real.",
         ["home", "renovation"], days_ago=88),
    Seed("seed_d_02", "seed_sess_12",
         "Wife wants white oak cabinets. I priced them out — $11k just for the boxes. "
         "We keep bumping into the same ceiling no matter what we cut.",
         ["home", "renovation"], days_ago=80),
    Seed("seed_d_03", "seed_sess_13",
         "Contractor pulled the old tile and found water damage under the subfloor. "
         "Another $4k. This is how it always goes.",
         ["home", "renovation"], days_ago=74),
    Seed("seed_d_04", "seed_sess_14",
         "Spent the weekend cooking in the garage on a hot plate. The novelty wore off "
         "by day two. Kids think it's camping.",
         ["home", "renovation"], days_ago=66),
    Seed("seed_d_05", "seed_sess_15",
         "Tile selection took three hours at the showroom. We left with nothing decided. "
         "Every choice branches into ten more choices.",
         ["home", "renovation"], days_ago=58),
    Seed("seed_d_06", "seed_sess_16",
         "Contractor installed the wrong cabinet pulls. Has to reorder. Another week delay. "
         "I am done caring about cabinet pulls.",
         ["home", "renovation"], days_ago=46),
    Seed("seed_d_07", "seed_sess_17",
         "Countertops are in. The quartz looks better than the sample — that never happens. "
         "First moment I've felt good about this project in weeks.",
         ["home", "renovation"], days_ago=34),
    Seed("seed_d_08", "seed_sess_18",
         "Final punch list has 22 items. Contractor says two days. I'm guessing two weeks. "
         "Experience has taught me to budget for his estimate times three.",
         ["home", "renovation"], days_ago=20),
    Seed("seed_d_09", "seed_sess_11",
         "Came in $9k over budget. Exactly where every contractor horror story ends up. "
         "At least the end is in sight.",
         ["home", "renovation"], days_ago=8),
    Seed("seed_d_10", "seed_sess_12",
         "Kitchen is done. We cooked dinner in it tonight for the first time. "
         "I forget what we fought about.",
         ["home", "renovation"], days_ago=2),
]

# ── Theme E: creative writing sprint ── span ~7 days, size 8 — LOW score ───────
WRITING = [
    Seed("seed_e_01", "seed_sess_13",
         "Started a short story I've been avoiding for two years. Got 800 words down "
         "before the kids woke up. Felt like stealing time.",
         ["creative", "writing"], days_ago=21),
    Seed("seed_e_02", "seed_sess_14",
         "The character isn't working. She keeps doing what I tell her instead of what "
         "she would actually do. Need to back off and listen.",
         ["creative", "writing"], days_ago=20),
    Seed("seed_e_03", "seed_sess_15",
         "2,200 words today. The scene I've been dreading wrote itself. That's the thing "
         "about just starting — the fear is always worse than the thing.",
         ["creative", "writing"], days_ago=19),
    Seed("seed_e_04", "seed_sess_16",
         "Read the draft out loud. The first half is good. The second half is explaining "
         "what the first half already showed. Cut it all.",
         ["creative", "writing"], days_ago=18),
    Seed("seed_e_05", "seed_sess_17",
         "Revised ending three times. Each version is cleaner. Starting to think the "
         "story ends a page before I think it does.",
         ["creative", "writing"], days_ago=17),
    Seed("seed_e_06", "seed_sess_18",
         "Sent it to two readers. The waiting is a specific kind of anxiety. "
         "Completely different from the writing anxiety.",
         ["creative", "writing"], days_ago=16),
    Seed("seed_e_07", "seed_sess_11",
         "First reader said 'I didn't want it to end.' Second reader said 'the ending "
         "felt rushed.' They read a different story.",
         ["creative", "writing"], days_ago=15),
    Seed("seed_e_08", "seed_sess_12",
         "Submitted it. Whatever happens now is out of my hands. The point was always "
         "finishing it, not what happens after.",
         ["creative", "writing"], days_ago=14),
]

# ── Theme F: chronic back pain ── span ~83 days, size 9 — HIGH score ───────────
BACK_PAIN = [
    Seed("seed_f_01", "seed_sess_13",
         "Back went out again lifting a box in the garage. Third time this year. "
         "Doctor wants an MRI. I keep saying I'll book it.",
         ["health", "body"], days_ago=87),
    Seed("seed_f_02", "seed_sess_14",
         "MRI shows two bulging discs at L4-L5. Surgeon says 'not urgent, try PT first.' "
         "PT says 'could take six months.' Six months of this.",
         ["health", "body"], days_ago=79),
    Seed("seed_f_03", "seed_sess_15",
         "PT exercises take 40 minutes a day. I do them maybe four days out of seven. "
         "The days I skip are always the days my back hurts most.",
         ["health", "body", "discipline"], days_ago=71),
    Seed("seed_f_04", "seed_sess_16",
         "Good week — almost no pain. Tempted to stop the PT exercises because I feel "
         "fine. That's exactly the trap the PT warned me about.",
         ["health", "body"], days_ago=61),
    Seed("seed_f_05", "seed_sess_17",
         "Sat through a three-hour flight and couldn't walk off the plane. The mobility "
         "I take for granted when it's there.",
         ["health", "body"], days_ago=50),
    Seed("seed_f_06", "seed_sess_18",
         "Second opinion surgeon says surgery would have 85% success rate. "
         "'What does failure look like?' He paused too long before answering.",
         ["health", "body", "decision"], days_ago=40),
    Seed("seed_f_07", "seed_sess_11",
         "Three months of consistent PT. The exercises I hated are the ones working. "
         "Lesson I keep relearning in every area of my life.",
         ["health", "body", "discipline"], days_ago=28),
    Seed("seed_f_08", "seed_sess_12",
         "Ran a mile for the first time since January. Slow. Didn't care.",
         ["health", "body"], days_ago=12),
    Seed("seed_f_09", "seed_sess_13",
         "Cancelled the surgery consult. PT has worked. Going to keep doing the boring "
         "exercises forever, apparently.",
         ["health", "body"], days_ago=4),
]

# ── Theme G: friendship drift ── span ~51 days, size 8 — MEDIUM score ──────────
FRIENDSHIP = [
    Seed("seed_g_01", "seed_sess_14",
         "Marcus and I used to talk every week. Now it's been two months. "
         "Not a fight — just a slow fade neither of us is naming.",
         ["friendship", "social"], days_ago=55),
    Seed("seed_g_02", "seed_sess_15",
         "Made plans to get dinner. He cancelled the day of. Third time in a row. "
         "I don't know if I'm bothered or relieved.",
         ["friendship", "social"], days_ago=48),
    Seed("seed_g_03", "seed_sess_16",
         "Realized I don't miss the friendship — I miss who I was in it. "
         "We used to have longer conversations than I have with anyone now.",
         ["friendship", "social", "identity"], days_ago=42),
    Seed("seed_g_04", "seed_sess_17",
         "He texted out of nowhere asking how I was doing. I took three days to reply. "
         "That's new for me.",
         ["friendship", "social"], days_ago=36),
    Seed("seed_g_05", "seed_sess_18",
         "We're in different places now. He has the life I thought I wanted at 28. "
         "I'm not sure that's the problem or just a fact.",
         ["friendship", "social", "identity"], days_ago=28),
    Seed("seed_g_06", "seed_sess_11",
         "Had a long call. It was fine — surface level, catching up. "
         "Hung up feeling more lonely than before I called.",
         ["friendship", "social"], days_ago=20),
    Seed("seed_g_07", "seed_sess_12",
         "Wondering if you can grieve a friendship that hasn't ended. "
         "The version of it I miss doesn't exist anymore.",
         ["friendship", "social"], days_ago=12),
    Seed("seed_g_08", "seed_sess_13",
         "Stopped waiting for him to reach out. If I want the friendship I have "
         "to decide that, not just wait to see what happens.",
         ["friendship", "social", "decision"], days_ago=4),
]

# ── Theme H: low-signal surface noise ── span ~4 days, size 4 — LOW score ──────
# Shares vocabulary ("work", "busy", "tired") but no real underlying pattern.
# Designed to score low and test whether the scorer distinguishes signal from noise.
NOISE = [
    Seed("seed_h_01", "seed_sess_14",
         "Really busy at work today. Back-to-back meetings. Tired by 3pm.",
         ["work", "daily"], days_ago=12),
    Seed("seed_h_02", "seed_sess_15",
         "Lots of work stuff going on. Busy week. Hard to focus.",
         ["work", "daily"], days_ago=10),
    Seed("seed_h_03", "seed_sess_16",
         "Work is hectic. Tired. Need a break but can't take one right now.",
         ["work", "daily"], days_ago=9),
    Seed("seed_h_04", "seed_sess_17",
         "Another busy day. Too many meetings. Didn't get to the actual work.",
         ["work", "daily"], days_ago=8),
]

ALL_SEEDS: list[Seed] = CAREER + RUNNING + ELDERCARE + RENOVATION + WRITING + BACK_PAIN + FRIENDSHIP + NOISE


# ═══════════════════════════════════════════════════════════════════════════════
# Loader
# ═══════════════════════════════════════════════════════════════════════════════

def _build_fragments(seeds: list[Seed]) -> list[Fragment]:
    fragments = []
    for s in seeds:
        fragments.append(
            Fragment(
                fragment_id=s.fragment_id,
                session_id=s.session_id,
                content=s.content,
                exchange_ids=[],
                tags=[Tag(tag=t) for t in s.tags],
                timestamp=ANCHOR - timedelta(days=s.days_ago),
            )
        )
    return fragments


def main() -> None:
    fragments = _build_fragments(ALL_SEEDS)
    logger.info(
        "Seeding %d fragments across %d sessions.",
        len(fragments),
        len({f.session_id for f in fragments}),
    )

    repo = FragmentRepository()
    repo.save_fragments(fragments)

    logger.info("Done. Fragments and session rows written.")
    logger.info(
        "Expected score distribution (approx): "
        "D≈53  F≈51  G≈35  A/B/C≈25  E≈11  H≈6"
    )


if __name__ == "__main__":
    main()
