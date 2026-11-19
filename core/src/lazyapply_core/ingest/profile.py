"""Loading profile.json — the hard facts.

Dates, degrees, titles and links are injected verbatim into every prompt and
never pass through retrieval or model paraphrase. That removes the most
damaging hallucination class by construction, but it also means a bad file is
the only way these can be wrong — so validation is strict and errors name the
offending field.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from lazyapply_core.domain import Profile
from lazyapply_core.hashing import content_hash
from lazyapply_core.ingest.parser import ParseError


def load_profile(path: Path) -> Profile:
    """Read and validate ``profile.json``.

    Raises:
        ParseError: if the file is missing, is not valid JSON, or violates the
            Profile schema. The message names the path and the bad field.
    """
    if not path.is_file():
        raise ParseError(f"profile not found: {path}")

    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ParseError(f"{path.name} is not valid JSON: {exc}") from exc

    try:
        return Profile.model_validate(raw)
    except ValidationError as exc:
        raise ParseError(f"{path.name}: {_describe(exc)}") from exc


def profile_version(profile: Profile) -> str:
    """Content hash of the facts, used to invalidate cached fit reports.

    Computed from the validated model rather than the file's bytes, with keys
    sorted — so reformatting or reordering the JSON does not invalidate
    anything, while changing an actual fact does.
    """
    canonical = profile.model_dump_json()
    return content_hash(json.dumps(json.loads(canonical), sort_keys=True))


def _describe(exc: ValidationError) -> str:
    """Pydantic errors as one readable line naming each bad field."""
    parts = []
    for error in exc.errors():
        field = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{field}: {error['msg']}")
    return "; ".join(parts)
