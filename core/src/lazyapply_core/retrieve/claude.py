"""Requirement extraction via the Claude API.

The only module that talks to Anthropic for extraction. Structured outputs
mean the result arrives as a validated model rather than JSON that has to be
parsed and hoped over.
"""

from __future__ import annotations

from typing import Any

from lazyapply_core.retrieve.base import ExtractionError
from lazyapply_core.retrieve.prompts import EXTRACTION_SYSTEM, EXTRACTION_USER
from lazyapply_core.retrieve.schema import ExtractedRole

DEFAULT_MODEL = "claude-opus-5"

MAX_OUTPUT_TOKENS = 8000
"""Headroom for a long posting: 12 requirements with restated text and ids.
Generous because truncation here loses requirements silently -- the schema
would still validate, just with fewer entries than the posting has."""


class ClaudeExtractor:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
    ) -> None:
        """
        Args:
            client: Pre-built SDK client. Injected by tests so the request
                shape is verified with no key and no network.
        """
        self._model = model
        self._client = client if client is not None else _build_client(api_key)

    def extract(self, role_text: str) -> ExtractedRole:
        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=MAX_OUTPUT_TOKENS,
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": EXTRACTION_USER.format(role_text=role_text)}],
                output_format=ExtractedRole,
            )
        except Exception as exc:
            raise ExtractionError(f"{self._model}: {exc}") from exc

        parsed = response.parsed_output
        if not isinstance(parsed, ExtractedRole):
            # A refusal or a stop before the schema was satisfied. Passing
            # None onward would surface far away as an unrelated error.
            raise ExtractionError(f"{self._model} returned no structured output")
        return parsed


def _build_client(api_key: str | None) -> Any:
    if not api_key:
        raise ExtractionError("ANTHROPIC_API_KEY is not set; add it to .env")

    import anthropic

    return anthropic.Anthropic(api_key=api_key)
