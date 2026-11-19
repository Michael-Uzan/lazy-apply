"""The extraction seam."""

from __future__ import annotations

from typing import Protocol

from lazyapply_core.retrieve.schema import ExtractedRole


class ExtractionError(Exception):
    """Extraction failed, or returned something unusable.

    Covers both provider faults (rate limits, outages) and results that
    validate as JSON but make no sense as a role -- no requirements at all,
    duplicate ids, nothing marked must-have.
    """


class Extractor(Protocol):
    def extract(self, role_text: str) -> ExtractedRole:
        """Turn a job posting into structured requirements."""
