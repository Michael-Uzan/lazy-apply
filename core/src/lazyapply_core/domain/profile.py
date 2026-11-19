"""Profile: the hard facts.

These never pass through retrieval and are never paraphrased by a model. They
are injected verbatim into every prompt. Dates, degrees and job titles are
exactly the things an LLM invents most confidently, so they are kept out of the
part of the system that can invent.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, EmailStr, HttpUrl

from lazyapply_core.domain.evidence import EndDate, YearMonth


class Link(BaseModel):
    model_config = ConfigDict(frozen=True)

    label: str
    url: HttpUrl


class Education(BaseModel):
    model_config = ConfigDict(frozen=True)

    degree: str
    institution: str
    graduated: YearMonth


class Position(BaseModel):
    model_config = ConfigDict(frozen=True)

    company: str
    title: str
    start: YearMonth
    end: EndDate


class Profile(BaseModel):
    """Verbatim facts about the candidate. Hand-maintained in profile.json."""

    model_config = ConfigDict(frozen=True)

    name: str
    headline: str
    location: str
    email: EmailStr
    links: tuple[Link, ...] = ()
    education: tuple[Education, ...] = ()
    positions: tuple[Position, ...] = ()
    stack: tuple[str, ...] = ()
