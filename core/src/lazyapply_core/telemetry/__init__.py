"""Usage and cost telemetry.

``Run`` is the unit everything else is built around: one recorder, one command
name, one id, created once per CLI invocation and handed to each provider. It
travels as a single object because the three always travel together, and
because a fit check's two provider calls have to share an id for the report to
answer "what does a check cost" rather than "what does a call cost".

Switching telemetry off is expressed as a Run whose recorder discards, not as
a flag each provider has to remember to check.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from lazyapply_core.config import Settings
from lazyapply_core.telemetry.base import NullTelemetry, Telemetry, Usage
from lazyapply_core.telemetry.jsonl import JsonlTelemetry, read_usage
from lazyapply_core.telemetry.report import Summary, summarise

__all__ = [
    "JsonlTelemetry",
    "NullTelemetry",
    "Run",
    "Summary",
    "Telemetry",
    "Usage",
    "read_usage",
    "summarise",
]


@dataclass(frozen=True)
class Run:
    """One command invocation's telemetry context."""

    telemetry: Telemetry
    command: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @classmethod
    def for_command(cls, command: str, settings: Settings, *, enabled: bool = True) -> Run:
        """Build the context for ``command``, logging unless switched off."""
        telemetry: Telemetry = JsonlTelemetry(settings.usage_log) if enabled else NullTelemetry()
        return cls(telemetry=telemetry, command=command)

    @classmethod
    def off(cls, command: str = "unknown") -> Run:
        """A context that records nothing. The default inside the library, so
        a test or a caller that does not care about cost needs no setup."""
        return cls(telemetry=NullTelemetry(), command=command)
