"""Settings and tuned constants.

Everything the system treats as a knob lives here, not in a prompt string. The
verdict thresholds especially: a number inside a prompt cannot be unit-tested,
cannot be swept against the golden set, and cannot be explained to a user.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Thresholds(BaseModel):
    """Verdict boundaries. See ``evaluate.verdict`` for how they combine."""

    skip_below: float = Field(default=0.60, ge=0.0, le=1.0)
    """must_have_coverage under this -> SKIP."""

    generate_at: float = Field(default=0.80, ge=0.0, le=1.0)
    """must_have_coverage at or above this (and score >= min_score) -> GENERATE."""

    min_score: int = Field(default=70, ge=0, le=100)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = ""
    voyage_api_key: str = ""

    qdrant_url: str = "http://localhost:6333"

    # --- Models -------------------------------------------------------------
    # Split by leverage: extraction runs once per role and is cached forever,
    # so it gets the expensive model. Judgment runs on every check.
    extraction_model: str = "claude-opus-5"
    judgment_model: str = "claude-haiku-4-5"
    embedding_model: str = "voyage-3-lite"

    # --- Vector store -------------------------------------------------------
    collection_alias: str = "evidence"
    """Stable name the app always queries."""

    collection_version: str = "evidence_v1"
    """Physical collection the alias points at. Changing chunking or embedding
    model means building v2 alongside, then repointing the alias atomically."""

    vector_size: int = 512
    """voyage-3-lite output dimensions."""

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 3
    """Candidates retrieved per requirement. No score threshold yet — scores
    are collected against the golden set first, then a threshold is chosen
    from that data rather than from intuition."""

    # --- Paths --------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    cache_dir: Path = REPO_ROOT / ".cache"

    thresholds: Thresholds = Thresholds()

    @property
    def evidence_dir(self) -> Path:
        return self.data_dir / "evidence"

    @property
    def profile_path(self) -> Path:
        return self.data_dir / "profile.json"

    @property
    def roles_dir(self) -> Path:
        return self.data_dir / "roles"


def load_settings(data_dir: Path | None = None) -> Settings:
    """Build settings, optionally pointing at a different data directory.

    The override exists so the README quickstart can run against
    ``data/example/`` without anyone editing a config file.
    """
    settings = Settings()
    if data_dir is not None:
        settings = settings.model_copy(update={"data_dir": data_dir})
    return settings
