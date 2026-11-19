"""Turning a Profile and an EvidencePack into prompt text.

Separate from ``prompts.py`` -- that file holds every piece of text the model
reads, this one holds the logic that lays the data out, and the logic is the
part worth unit-testing.

Two layout decisions carry weight:

- The evidence catalogue lists each block once, with its id. Requirements then
  refer to blocks by id. A block retrieved for five requirements therefore
  costs its tokens once, and the model always has an exact id to cite -- which
  is what makes the citation verifier possible.
- A requirement that retrieved nothing says so in words. Absence has to be
  visible, because an unmet requirement is the signal the verdict is built on.
"""

from __future__ import annotations

from lazyapply_core.domain import EvidencePack, Profile
from lazyapply_core.evaluate.prompts import HARD_FACT, NOTHING_RETRIEVED


def render_facts(profile: Profile) -> str:
    """The hard facts, verbatim.

    Injected on every check and never retrieved or paraphrased. Dates, degrees
    and titles are what an LLM invents most confidently, so they are kept out
    of the part of the system that can invent. Empty sections are omitted
    rather than left as headings a model might feel obliged to fill.
    """
    lines = [f"{profile.name} — {profile.headline}", f"Location: {profile.location}"]

    if profile.education:
        lines.append("\nEducation:")
        lines += [
            f"- {e.degree}, {e.institution} (graduated {e.graduated})" for e in profile.education
        ]

    if profile.positions:
        lines.append("\nEmployment:")
        lines += [f"- {p.company} — {p.title} ({p.start} to {p.end})" for p in profile.positions]

    if profile.stack:
        lines.append(f"\nStack: {', '.join(profile.stack)}")

    return "\n".join(lines)


def render_pack(pack: EvidencePack) -> str:
    """The evidence catalogue followed by the requirements to judge."""
    return f"{_catalogue(pack)}\n\n<requirements>\n{_requirements(pack)}\n</requirements>"


def _catalogue(pack: EvidencePack) -> str:
    blocks = pack.unique_evidence()
    if not blocks:
        # A role of pure hard facts retrieves nothing. The prompt must stay
        # coherent rather than carry a dangling heading.
        return "<evidence>\n(no evidence retrieved)\n</evidence>"

    rendered = "\n\n".join(f"[{e.evidence_id}]\n{e.composed_text}" for e in blocks)
    return f"<evidence>\n{rendered}\n</evidence>"


def _requirements(pack: EvidencePack) -> str:
    lines = []
    for item in pack.items:
        requirement = item.requirement
        lines.append(f"{requirement.id} [{requirement.kind.label}] {requirement.text}")

        if requirement.is_hard_fact:
            lines.append(f"  {HARD_FACT}")
        elif not item.hits:
            lines.append(f"  {NOTHING_RETRIEVED}")
        else:
            # Ids only, no similarity scores. A cosine distance is specific to
            # the embedding model and is not calibrated to anything a reader
            # would recognise; showing it would anchor the model to a number
            # instead of to the evidence text. Scores are kept for threshold
            # tuning, where they are compared against each other.
            candidates = ", ".join(h.evidence.evidence_id for h in item.hits)
            lines.append(f"  retrieved: {candidates}")
    return "\n".join(lines)
