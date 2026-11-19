"""Loading a whole directory of evidence files.

The parser handles one document. This handles the collection: deterministic
order, and the cross-file collision check the parser cannot see.
"""

from __future__ import annotations

from pathlib import Path

from lazyapply_core.domain import Evidence
from lazyapply_core.ingest.parser import ParseError, parse_evidence


def load_evidence_dir(directory: Path) -> tuple[Evidence, ...]:
    """Parse every ``*.md`` file in ``directory``, sorted by filename.

    Sorted order matters: ingest must produce the same sequence on every
    machine and every run, or diffing two ingests tells you nothing.

    Raises:
        ParseError: if the directory is missing, holds no ``.md`` files, any
            file fails to parse, or two blocks resolve to the same evidence id.
    """
    if not directory.is_dir():
        raise ParseError(f"evidence directory does not exist: {directory}")

    paths = sorted(directory.glob("*.md"))
    if not paths:
        raise ParseError(f"no .md files in evidence directory: {directory}")

    evidence: list[Evidence] = []
    for path in paths:
        evidence.extend(parse_evidence(path.read_text(), source_file=path.name))

    _reject_duplicate_ids(tuple(evidence))
    return tuple(evidence)


def _reject_duplicate_ids(evidence: tuple[Evidence, ...]) -> None:
    """Two blocks sharing an id would make a citation ambiguous."""
    seen: dict[str, str] = {}
    for ev in evidence:
        if ev.evidence_id in seen:
            raise ParseError(
                f"duplicate evidence id {ev.evidence_id!r}: "
                f"produced by both {seen[ev.evidence_id]} and {ev.source_file}"
            )
        seen[ev.evidence_id] = ev.source_file
