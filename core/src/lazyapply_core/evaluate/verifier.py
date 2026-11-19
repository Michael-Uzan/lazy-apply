"""Citation verification: every claim must point at evidence we actually sent.

This is the module that makes "grounded" a property of the system rather than
a hope about the prompt. Three things are checked, and each one is a hard
failure:

- A cited evidence id that was not in the pack. The model invented its
  justification; the report would still read convincingly, which is precisely
  why this cannot be a warning.
- A STRONG or PARTIAL with no citation at all. A positive claim with nothing
  behind it is the failure the whole design exists to prevent.
- A requirement id that was never asked about, or answered twice. The model
  drifted from its input and nothing downstream could place the answer.

A requirement the model simply did not answer is *not* an error: the verdict
records it as an explicit NONE. Rejecting the whole response would throw away
the judgments that were right.
"""

from __future__ import annotations

from lazyapply_core.domain import EvidencePack, MatchStatus
from lazyapply_core.evaluate.schema import Judgment

_NEEDS_CITATION = (MatchStatus.STRONG, MatchStatus.PARTIAL)


class CitationError(Exception):
    """A judgment is not grounded in the evidence it was given."""


def verify(judgment: Judgment, pack: EvidencePack) -> None:
    """Raise ``CitationError`` unless every judgment is grounded.

    Citations are checked against the whole pack, not against the single
    requirement's own hits. Blocks are a shared catalogue: a story retrieved
    for one requirement can genuinely satisfy another, and forbidding that
    would discard correct matches.
    """
    known_evidence = {e.evidence_id for e in pack.unique_evidence()}
    hard_facts = {i.requirement.id for i in pack.items if i.requirement.is_hard_fact}
    requirement_ids = {i.requirement.id for i in pack.items}
    seen: set[str] = set()

    for match in judgment.matches:
        rid = match.requirement_id

        if rid not in requirement_ids:
            raise CitationError(f"unknown requirement id: {rid!r}")
        if rid in seen:
            raise CitationError(f"{rid}: judged twice")
        seen.add(rid)

        for evidence_id in match.evidence_ids:
            if evidence_id not in known_evidence:
                raise CitationError(f"{rid}: cites {evidence_id!r}, which was not in the pack")

        # Hard facts are judged against profile.json, which is injected
        # verbatim and has no evidence ids to cite.
        if match.status in _NEEDS_CITATION and not match.evidence_ids and rid not in hard_facts:
            raise CitationError(f"{rid}: {match.status.value} with no evidence cited")
