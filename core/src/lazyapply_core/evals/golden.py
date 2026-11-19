"""Loading the hand-labelled golden set.

Strict on purpose. The labels in ``evals/golden.yaml`` are the only thing in
this project that cannot be regenerated -- embeddings, extractions and reports
are all derived -- so a typo in an evidence id must fail here, by name. Left
to leak through, it would appear downstream as a retrieval failure and send
someone hunting a bug in the embedder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.domain import MatchStatus, Verdict

DEFAULT_GOLDEN_PATH = REPO_ROOT / "core/evals/golden.yaml"

SUPPORTED_VERSION = 1
"""Bumped only when the label format changes. Refusing an unknown version
keeps numbers already published from an older format from being mixed in."""

_EVIDENCE_ID = re.compile(r"^ev_[a-z0-9]+(?:-[a-z0-9]+)*__[a-z0-9]+(?:-[a-z0-9]+)*$")


class GoldenError(Exception):
    """The golden set could not be read as written."""


@dataclass(frozen=True)
class GoldenRequirement:
    """One hand-labelled requirement."""

    id: str

    relevant: frozenset[str]
    """Evidence blocks that should be retrieved. Empty means none are -- a
    label, not an omission."""

    status: MatchStatus
    """What a person answers. Compared against the model's answer."""

    note: str | None = None
    """Why this label, for anything non-obvious. Read by humans only."""


@dataclass(frozen=True)
class GoldenCase:
    """One posting and its labels."""

    role: str
    expected_verdict: Verdict
    requirements: tuple[GoldenRequirement, ...]

    @property
    def scoreable(self) -> tuple[GoldenRequirement, ...]:
        """Requirements that can be scored for recall and MRR.

        A requirement with an empty relevant set has no correct answer to
        find, so those metrics are undefined for it. Reported separately
        rather than counted as failures, which would understate retrieval by
        blaming it for questions it was never asked.
        """
        return tuple(r for r in self.requirements if r.relevant)


@dataclass(frozen=True)
class GoldenSet:
    cases: tuple[GoldenCase, ...]


def load_golden(path: Path = DEFAULT_GOLDEN_PATH) -> GoldenSet:
    """Read and validate the golden set.

    Raises:
        GoldenError: on a missing file, malformed YAML, an unsupported
            version, an unknown status or verdict, a malformed evidence id, a
            duplicate requirement id, or a case with no requirements.
    """
    if not path.is_file():
        raise GoldenError(f"golden set not found: {path}")

    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise GoldenError(f"{path.name} is not valid YAML: {exc}") from exc

    version = raw.get("version")
    if version != SUPPORTED_VERSION:
        raise GoldenError(
            f"unsupported golden set version {version!r}, expected {SUPPORTED_VERSION}"
        )

    return GoldenSet(cases=tuple(_case(entry) for entry in raw.get("cases") or ()))


def _case(entry: dict[str, Any]) -> GoldenCase:
    role = str(entry.get("role", "(unnamed)"))
    requirements = tuple(_requirement(r, role) for r in entry.get("requirements") or ())

    if not requirements:
        raise GoldenError(f"{role}: no requirements labelled")

    ids = [r.id for r in requirements]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise GoldenError(f"{role}: duplicate requirement ids: {', '.join(sorted(duplicates))}")

    return GoldenCase(
        role=role,
        expected_verdict=_enum(Verdict, entry.get("expected_verdict"), "verdict", role),
        requirements=requirements,
    )


def _requirement(entry: dict[str, Any], role: str) -> GoldenRequirement:
    requirement_id = str(entry.get("id", "(unnamed)"))
    where = f"{role}/{requirement_id}"

    relevant = entry.get("relevant") or []
    for evidence_id in relevant:
        if not _EVIDENCE_ID.match(str(evidence_id)):
            raise GoldenError(f"{where}: {evidence_id!r} is not a valid evidence id")

    note = entry.get("note")
    return GoldenRequirement(
        id=requirement_id,
        relevant=frozenset(str(e) for e in relevant),
        status=_enum(MatchStatus, entry.get("status"), "status", where),
        note=None if note is None else str(note).strip(),
    )


def _enum(enum_type: Any, value: Any, field: str, where: str) -> Any:
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = ", ".join(member.value for member in enum_type)
        raise GoldenError(
            f"{where}: {value!r} is not a known {field} (expected {allowed})"
        ) from exc
