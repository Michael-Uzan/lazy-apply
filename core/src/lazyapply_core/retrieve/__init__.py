"""Retrieval layer: role text in, scored evidence out.

`build_extractor` is the only place that decides whether requirements come
from Claude or from a canned fake.
"""

from __future__ import annotations

from lazyapply_core.config import Settings
from lazyapply_core.retrieve.base import ExtractionError, Extractor
from lazyapply_core.retrieve.requirements import extract_role


def build_extractor(settings: Settings, *, fake: bool = False) -> Extractor:
    if fake:
        from lazyapply_core.retrieve.fake import canned_extractor

        return canned_extractor()

    from lazyapply_core.retrieve.claude import ClaudeExtractor

    return ClaudeExtractor(api_key=settings.anthropic_api_key, model=settings.extraction_model)


__all__ = ["ExtractionError", "Extractor", "build_extractor", "extract_role"]
