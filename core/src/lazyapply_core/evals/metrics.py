"""Retrieval and agreement metrics.

Pure arithmetic over hand-labelled data. Two reasons this is its own module:
a metric that is quietly wrong reads as evidence rather than as a bug, and
these are the numbers that go in the README, so they have to be defensible.

Three measurements, answering three different questions:

- **recall@k** — was the right evidence retrieved at all? The model only ever
  sees the top k, so anything below the cutoff is lost regardless of score.
- **MRR** — how far down did the reader have to look? Two orderings can share
  a recall of 1.0 and still differ: first place and third place are not the
  same result.
- **agreement** — did the model's judgment match a person's? Reported with
  direction, because a generous model and a harsh model are different
  problems with different fixes.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from lazyapply_core.domain import MatchStatus

_RANK = {MatchStatus.NONE: 0, MatchStatus.PARTIAL: 1, MatchStatus.STRONG: 2}
"""Statuses are ordered, so a disagreement has a direction."""


def recall_at_k(retrieved: Sequence[str], *, relevant: set[str], k: int) -> float:
    """Fraction of the relevant blocks that appear in the top ``k``.

    Raises:
        ValueError: if nothing is relevant. A requirement with no correct
            answer cannot be scored for recall -- it tests whether the system
            declines to match, which the agreement metric covers instead.
    """
    if not relevant:
        raise ValueError("no relevant evidence to find; recall is undefined")

    found = set(retrieved[:k]) & relevant
    return len(found) / len(relevant)


def reciprocal_rank(retrieved: Sequence[str], *, relevant: set[str]) -> float:
    """``1 / rank`` of the first relevant block, or 0.0 if none was retrieved.

    Only the first hit counts: the question is how far a reader has to look,
    and a second correct block further down does not shorten that.
    """
    for position, evidence_id in enumerate(retrieved, start=1):
        if evidence_id in relevant:
            return 1 / position
    return 0.0


@dataclass(frozen=True)
class Agreement:
    """How often the model's status matched a person's, and how it differed."""

    total: int
    matched: int
    disagreements: tuple[tuple[MatchStatus, MatchStatus], ...]
    """``(expected, actual)`` pairs, in the order they were compared."""

    @property
    def rate(self) -> float:
        return self.matched / self.total

    @staticmethod
    def direction(pair: tuple[MatchStatus, MatchStatus]) -> str:
        """ "generous" or "harsh" for one disagreement.

        Per pair, not from the aggregate counts -- deriving it from the totals
        would label every line in a mixed result the same way.
        """
        expected, actual = pair
        return "generous" if _RANK[actual] > _RANK[expected] else "harsh"

    @property
    def generous(self) -> int:
        """Times the model claimed more than a person would.

        The expensive direction: an inflated match sends someone into an
        interview they cannot defend, and they cannot detect it by reading
        the report.
        """
        return sum(1 for expected, actual in self.disagreements if _RANK[actual] > _RANK[expected])

    @property
    def harsh(self) -> int:
        """Times the model claimed less. Costs opportunities, not credibility."""
        return sum(1 for expected, actual in self.disagreements if _RANK[actual] < _RANK[expected])


def agreement(pairs: Sequence[tuple[MatchStatus, MatchStatus]]) -> Agreement:
    """Compare ``(expected, actual)`` statuses.

    Raises:
        ValueError: if there is nothing to compare.
    """
    if not pairs:
        raise ValueError("nothing to compare")

    disagreements = tuple((expected, actual) for expected, actual in pairs if expected != actual)
    return Agreement(
        total=len(pairs),
        matched=len(pairs) - len(disagreements),
        disagreements=disagreements,
    )


def mean(values: Sequence[float]) -> float | None:
    """Average, or None when there is nothing to average.

    None rather than 0.0 on purpose: "we measured nothing" must not print as
    a bad score.
    """
    if not values:
        return None
    return sum(values) / len(values)
