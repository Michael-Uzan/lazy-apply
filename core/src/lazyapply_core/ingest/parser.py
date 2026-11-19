"""Parsing hand-authored evidence markdown into Evidence objects.

File shape — YAML front matter describing the context, then one ``##`` block
per self-contained claim:

    ---
    company: Northwind Trading
    role: Senior Full-Stack Engineer
    start: 2022-03
    end: present
    ---

    ## Real-time trading dashboard
    tags: [react, typescript]

    Rebuilt the dashboard as a React application.

Every problem this module can detect is raised, never worked around. A block
that silently parses wrong becomes a wrong vector, which becomes a retrieval
miss nobody can explain weeks later.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from lazyapply_core.domain import EndDate, Evidence, YearMonth
from lazyapply_core.hashing import content_hash
from lazyapply_core.ingest.slug import slugify

_FRONT_MATTER = re.compile(r"\A---\r?\n(?P<meta>.*?)\r?\n---[ \t]*\r?\n", re.DOTALL)
_HEADING = re.compile(r"^##[ \t]+(?P<title>.+?)[ \t]*$", re.MULTILINE)
_TAGS_LINE = re.compile(r"^tags:[ \t]*\[(?P<tags>.*?)\][ \t]*$", re.MULTILINE)

_YEAR_MONTH = TypeAdapter(YearMonth)
_END_DATE = TypeAdapter(EndDate)


class ParseError(Exception):
    """An evidence file could not be read as written."""


def parse_evidence(text: str, *, source_file: str) -> tuple[Evidence, ...]:
    """Parse one evidence markdown document.

    Args:
        text: Full file contents.
        source_file: Path recorded on each Evidence and used, via its stem, as
            the first half of every evidence id.

    Raises:
        ParseError: on missing front matter, no blocks, duplicate titles,
            empty bodies, or malformed dates.
    """
    meta, body = _split_front_matter(text, source_file)
    file_slug = slugify(Path(source_file).stem)

    evidence = tuple(
        _build(title, raw, meta=meta, file_slug=file_slug, source_file=source_file)
        for title, raw in _split_blocks(body, source_file)
    )
    _reject_duplicate_ids(evidence, source_file)
    return evidence


def _split_front_matter(text: str, source_file: str) -> tuple[dict[str, Any], str]:
    match = _FRONT_MATTER.match(text)
    if match is None:
        raise ParseError(f"{source_file}: missing YAML front matter (a '---' block at the top)")

    loaded = yaml.safe_load(match.group("meta")) or {}
    if not isinstance(loaded, dict):
        raise ParseError(f"{source_file}: front matter must be a mapping of keys to values")

    return loaded, text[match.end() :]


def _split_blocks(body: str, source_file: str) -> list[tuple[str, str]]:
    headings = list(_HEADING.finditer(body))
    if not headings:
        raise ParseError(f"{source_file}: no '##' blocks found")

    blocks: list[tuple[str, str]] = []
    for i, heading in enumerate(headings):
        end = headings[i + 1].start() if i + 1 < len(headings) else len(body)
        blocks.append((heading.group("title"), body[heading.end() : end]))
    return blocks


def _build(
    title: str,
    raw: str,
    *,
    meta: dict[str, Any],
    file_slug: str,
    source_file: str,
) -> Evidence:
    tags = _extract_tags(raw)
    prose = _TAGS_LINE.sub("", raw).strip()
    if not prose:
        raise ParseError(f"{source_file}: block '{title}' has an empty body")

    try:
        title_slug = slugify(title)
    except ValueError as exc:
        raise ParseError(f"{source_file}: {exc}") from exc

    company = _as_str(meta.get("company"))
    composed = _compose(company=company, title=title, tags=tags, prose=prose)

    return Evidence(
        evidence_id=f"ev_{file_slug}__{title_slug}",
        title=title,
        body=prose,
        composed_text=composed,
        content_hash=content_hash(composed),
        company=company,
        role=_as_str(meta.get("role")),
        start=_validate(_YEAR_MONTH, meta.get("start"), "start", source_file),
        end=_validate(_END_DATE, meta.get("end"), "end", source_file),
        tags=tags,
        source_file=source_file,
    )


def _compose(*, company: str | None, title: str, tags: tuple[str, ...], prose: str) -> str:
    """Build the text that actually gets embedded.

    Requirement queries are terse ("5+ years of React"), so the company name
    and tags are folded in — prose alone matches them poorly.
    """
    headline = f"{company} — {title}" if company else title
    lines = [headline]
    if tags:
        lines.append(f"tags: {', '.join(tags)}")
    lines.append(prose)
    return "\n".join(lines)


def _extract_tags(raw: str) -> tuple[str, ...]:
    match = _TAGS_LINE.search(raw)
    if match is None:
        return ()
    return tuple(tag.strip() for tag in match.group("tags").split(",") if tag.strip())


def _as_str(value: Any) -> str | None:
    return None if value is None else str(value)


def _validate(adapter: TypeAdapter[Any], value: Any, field: str, source_file: str) -> Any:
    if value is None:
        return None
    try:
        return adapter.validate_python(str(value))
    except ValidationError as exc:
        raise ParseError(f"{source_file}: invalid {field} date {value!r}") from exc


def _reject_duplicate_ids(evidence: tuple[Evidence, ...], source_file: str) -> None:
    seen: set[str] = set()
    for ev in evidence:
        if ev.evidence_id in seen:
            raise ParseError(f"{source_file}: duplicate block title '{ev.title}'")
        seen.add(ev.evidence_id)
