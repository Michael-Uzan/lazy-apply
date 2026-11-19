"""Role text -> a validated, cached Role.

Extraction is the highest-leverage call in the system: the must-have/
nice-to-have split feeds straight into the verdict formula, and each
requirement's restated text becomes a retrieval query. It therefore runs on
the expensive model, and runs once per posting -- results are cached forever
by a hash of the role text.
"""

from __future__ import annotations

from pathlib import Path

from lazyapply_core.domain import Requirement, Role
from lazyapply_core.hashing import content_hash, normalize
from lazyapply_core.retrieve.base import ExtractionError, Extractor
from lazyapply_core.retrieve.schema import ExtractedRole


def extract_role(role_text: str, *, extractor: Extractor, cache_dir: Path) -> Role:
    """Extract requirements from ``role_text``, using the cache if possible.

    Raises:
        ExtractionError: if the text is empty, or the extraction is unusable.
    """
    if not normalize(role_text):
        raise ExtractionError("role text is empty")

    # Hashed after whitespace normalization, so reflowing a pasted posting
    # does not pay for a second extraction.
    role_id = content_hash(role_text)
    path = cache_dir / f"{role_id}.json"

    extracted = _read_cache(path) or _extract_and_cache(role_text, extractor, path)
    _validate(extracted)

    return Role(
        role_id=role_id,
        title=extracted.title,
        company=extracted.company,
        raw_text=role_text,
        requirements=tuple(
            Requirement(id=r.id, text=r.text, kind=r.kind, category=r.category)
            for r in extracted.requirements
        ),
    )


def _extract_and_cache(role_text: str, extractor: Extractor, path: Path) -> ExtractedRole:
    extracted = extractor.extract(role_text)
    _validate(extracted)

    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(extracted.model_dump_json(indent=2))
    temp.replace(path)
    return extracted


def _read_cache(path: Path) -> ExtractedRole | None:
    if not path.exists():
        return None
    return ExtractedRole.model_validate_json(path.read_text())


def _validate(extracted: ExtractedRole) -> None:
    """Reject extractions that parse but make no sense as a role."""
    if not extracted.requirements:
        raise ExtractionError("extraction returned no requirements")

    ids = [r.id for r in extracted.requirements]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ExtractionError(f"duplicate requirement ids: {', '.join(sorted(duplicates))}")

    # Coverage divides by the must-have count, so zero would be undefined --
    # and a posting with no hard requirement is a misread, not a real role.
    if not any(r.kind.value == "must_have" for r in extracted.requirements):
        raise ExtractionError("extraction found no must-have requirements")


__all__ = ["ExtractionError", "extract_role"]
