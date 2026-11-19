"""Shared fixtures. Everything here is offline: no API key, no Qdrant."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.domain import Evidence

EXAMPLE_CACHE = REPO_ROOT / "data/example/cache/roles"


@pytest.fixture
def seeded_cache(tmp_path: Path) -> Path:
    """A cache directory holding the committed real extractions.

    ``--fake`` swaps in a canned extractor whose requirement ids differ from
    the real ones, and ``core/evals/golden.yaml`` labels the real ids. Seeding
    the cache lets a test run the eval path offline against the ids the labels
    actually refer to.
    """
    shutil.copytree(EXAMPLE_CACHE, tmp_path / "roles")
    return tmp_path


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
