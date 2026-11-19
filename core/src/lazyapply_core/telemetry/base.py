"""The telemetry seam.

A protocol with a no-op default, like the embedder and the store. Every
provider takes a recorder, so no call site needs a ``if telemetry is not
None`` branch -- ``NullTelemetry`` is a real object that does nothing.

Telemetry is an observer and never a participant: it is given what already
happened, and it cannot change or fail the thing it is watching.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol

from lazyapply_core.telemetry.pricing import cost_usd


@dataclass(frozen=True)
class Usage:
    """One provider call, and what it cost."""

    command: str
    """Which CLI command caused it. The unit a user asks about: "what does a
    fit check cost", not "what does a messages.parse cost"."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int

    run: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    """Groups the calls of one command invocation. A fit check makes two
    provider calls -- one embedding, one judgment -- and the number a user
    asks for is what the *check* cost, not what each call cost."""

    cached: bool = False
    """A cache hit made no request. Recorded so the log can show how much the
    cache saved, which is the only place that is visible."""

    at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def cost_usd(self) -> float:
        """What this call cost. Zero for a cache hit: charging for one would
        make the cache look worthless in the report meant to prove its value."""
        if self.cached:
            return 0.0
        return cost_usd(
            self.model, input_tokens=self.input_tokens, output_tokens=self.output_tokens
        )


class Telemetry(Protocol):
    def record(self, event: Usage) -> None:
        """Note that a call happened. Must never raise."""


class NullTelemetry:
    """The default. Discards everything."""

    def record(self, event: Usage) -> None:
        return None
