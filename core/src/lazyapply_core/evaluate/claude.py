"""Matching via the Claude API.

The only module that talks to Anthropic for matching. Unlike extraction --
once per posting, cached forever -- this call runs on every fit check, so it
runs on the cheap model. The work is also easier: the hard reading was done at
extraction, and this step reasons over a small, pre-retrieved pack rather than
a raw posting.
"""

from __future__ import annotations

from typing import Any

from lazyapply_core.domain import EvidencePack, Profile
from lazyapply_core.evaluate.base import EvaluationError
from lazyapply_core.evaluate.prompts import MATCHING_SYSTEM, MATCHING_USER
from lazyapply_core.evaluate.render import render_facts, render_pack
from lazyapply_core.evaluate.schema import Judgment

DEFAULT_MODEL = "claude-haiku-4-5"


class ClaudeEvaluator:
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

    def judge(self, pack: EvidencePack, profile: Profile) -> Judgment:
        prompt = MATCHING_USER.format(facts=render_facts(profile), pack=render_pack(pack))

        try:
            response = self._client.messages.parse(
                model=self._model,
                max_tokens=4000,
                system=MATCHING_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                output_format=Judgment,
            )
        except Exception as exc:
            raise EvaluationError(f"{self._model}: {exc}") from exc

        parsed = response.parsed_output
        if not isinstance(parsed, Judgment):
            # A refusal, or a stop before the schema was satisfied. Passing
            # None onward would surface far away as an unrelated error.
            raise EvaluationError(f"{self._model} returned no structured output")
        return parsed


def _build_client(api_key: str | None) -> Any:
    if not api_key:
        raise EvaluationError("ANTHROPIC_API_KEY is not set; add it to .env")

    import anthropic

    return anthropic.Anthropic(api_key=api_key)
