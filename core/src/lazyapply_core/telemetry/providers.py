"""Reading token counts off each provider's response.

Provider-specific knowledge, kept in one place rather than repeated in every
module that makes a call. The two Claude modules need identical code, and the
shapes differ enough between providers that a single generic reader would be
mostly ``getattr`` guesswork.

Counts come from the response, never estimated from prompt length. An
estimated cost report is not worth printing.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

from lazyapply_core.telemetry.base import Telemetry, Usage

ANTHROPIC = "anthropic"
VOYAGE = "voyage"


def elapsed_ms(started: float) -> int:
    """Milliseconds since a ``perf_counter()`` reading."""
    return int((perf_counter() - started) * 1000)


def record_anthropic(
    telemetry: Telemetry,
    response: Any,
    *,
    command: str,
    run: str,
    model: str,
    started: float,
) -> None:
    """Record one Messages API call."""
    usage = getattr(response, "usage", None)
    telemetry.record(
        Usage(
            command=command,
            provider=ANTHROPIC,
            model=model,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            latency_ms=elapsed_ms(started),
            run=run,
        )
    )


def record_voyage(
    telemetry: Telemetry,
    response: Any,
    *,
    command: str,
    run: str,
    model: str,
    started: float,
) -> None:
    """Record one embedding request.

    Embeddings have no output tokens, so ``output_tokens`` stays zero and the
    cost arithmetic is the same shape as every other provider's.
    """
    telemetry.record(
        Usage(
            command=command,
            provider=VOYAGE,
            model=model,
            input_tokens=getattr(response, "total_tokens", 0) or 0,
            output_tokens=0,
            latency_ms=elapsed_ms(started),
            run=run,
        )
    )
