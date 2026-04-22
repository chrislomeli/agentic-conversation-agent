"""seed_fragments.py — load a test corpus of fragments into Postgres.

Run: ``uv run python -m journal_agent.scripts.seed_fragments``

Produces ~30 fragments across 3 distinct themes so HDBSCAN can find them:
    A — weighing a career pivot (IC → management)
    B — marathon training and morning discipline
    C — caring for an aging parent

Each fragment has:
    - fixed fragment_id (seed_<theme>_<nn>) — ON CONFLICT upsert = idempotent
    - deterministic session_id (seed_sess_<nn>) — sessions row auto-created
    - timestamp spread across ~30 days so recency_spread varies per cluster
    - content + tags

Embeddings are computed in-script by PgFragmentRepository (fastembed, 384-dim).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from journal_agent.model.session import Fragment, Tag
from journal_agent.stores.fragment_repo import PgFragmentRepository

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


# ── Theme A: career pivot (IC → management) ───────────────────────────────────
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

# ── Theme B: marathon training / morning discipline ───────────────────────────
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

# ── Theme C: caring for an aging parent ───────────────────────────────────────
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

ALL_SEEDS: list[Seed] = CAREER + RUNNING + ELDERCARE


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
    logger.info("Seeding %d fragments across %d sessions.",
                len(fragments), len({f.session_id for f in fragments}))

    repo = PgFragmentRepository()
    repo.save_fragments(fragments)

    logger.info("Done. Fragments and session rows written.")


if __name__ == "__main__":
    main()
