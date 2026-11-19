"""Shared fixtures. Everything here is offline: no API key, no Qdrant."""

from __future__ import annotations

import pytest

from lazyapply_core.domain import Evidence


@pytest.fixture
def evidence() -> Evidence:
    return Evidence(
        evidence_id="ev_northwind__real-time-trading-dashboard",
        title="Real-time trading dashboard",
        body="Rebuilt the trading dashboard as a React + TypeScript application.",
        composed_text=(
            "Northwind Trading — Real-time trading dashboard\n"
            "tags: react, typescript\n"
            "Rebuilt the trading dashboard as a React + TypeScript application."
        ),
        content_hash="0123456789abcdef",
        company="Northwind Trading",
        role="Senior Full-Stack Engineer",
        start="2022-03",
        end="present",
        tags=("react", "typescript"),
        source_file="evidence/northwind.md",
    )
