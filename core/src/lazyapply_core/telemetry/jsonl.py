"""Append-only JSONL usage log.

One line per provider call, never rewritten. The format is chosen for the
question it has to answer -- "what did this cost" -- and a file you can
``tail``, ``grep`` and ``jq`` beats a schema you have to migrate.

Two rules hold this together:

- **Cost is written, not derived.** The price at the time of the call is baked
  into the line. A report of what last month cost must not change when a
  provider changes a price.
- **A write failure is swallowed.** Telemetry watches; it does not
  participate. Losing a cost line is a nuisance, failing the fit check the
  user actually asked for is not acceptable.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from lazyapply_core.telemetry.base import Usage


class JsonlTelemetry:
    """Appends each call to a JSONL file."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def record(self, event: Usage) -> None:
        line = asdict(event) | {"at": event.at.isoformat(), "cost_usd": event.cost_usd}

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line) + "\n")
        except OSError:
            # A full disk or a read-only path must not take the command down
            # with it. Nothing downstream depends on this line existing.
            return None


def read_usage(path: Path) -> tuple[Usage, ...]:
    """Read the log back.

    A missing file reads as empty -- nobody has run anything yet, which is not
    an error. A malformed line is skipped: an interrupted write should cost
    one line, not the whole history.
    """
    if not path.is_file():
        return ()

    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        event = _parse(line)
        if event is not None:
            events.append(event)
    return tuple(events)


def _parse(line: str) -> Usage | None:
    try:
        raw = json.loads(line)
        return Usage(
            command=raw["command"],
            provider=raw["provider"],
            model=raw["model"],
            input_tokens=raw["input_tokens"],
            output_tokens=raw["output_tokens"],
            latency_ms=raw["latency_ms"],
            run=raw.get("run", ""),
            cached=raw.get("cached", False),
            at=datetime.fromisoformat(raw["at"]),
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
