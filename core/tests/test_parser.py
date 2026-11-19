"""Parsing an evidence markdown file into Evidence objects."""

from textwrap import dedent

import pytest

from lazyapply_core.hashing import content_hash
from lazyapply_core.ingest.parser import ParseError, parse_evidence

DOC = dedent("""\
    ---
    company: Northwind Trading
    role: Senior Full-Stack Engineer
    start: 2022-03
    end: present
    ---

    ## Real-time trading dashboard
    tags: [react, typescript]

    Rebuilt the dashboard as a React application.
    Time-to-interactive went from 4.2s to 900ms.

    ## GraphQL federation
    tags: [graphql, nodejs]

    Unified four services behind a federated schema.
    """)


def parse(text: str = DOC, source_file: str = "evidence/northwind.md"):
    return parse_evidence(text, source_file=source_file)


class TestHappyPath:
    def test_one_evidence_per_heading(self) -> None:
        assert len(parse()) == 2

    def test_evidence_id_combines_file_and_title(self) -> None:
        assert parse()[0].evidence_id == "ev_northwind__real-time-trading-dashboard"

    def test_front_matter_applies_to_every_block(self) -> None:
        for ev in parse():
            assert ev.company == "Northwind Trading"
            assert ev.role == "Senior Full-Stack Engineer"
            assert ev.start == "2022-03"
            assert ev.end == "present"

    def test_tags_are_parsed(self) -> None:
        assert parse()[0].tags == ("react", "typescript")

    def test_body_excludes_the_heading_and_tags_line(self) -> None:
        body = parse()[0].body
        assert body.startswith("Rebuilt the dashboard")
        assert "tags:" not in body
        assert "##" not in body

    def test_composed_text_leads_with_company_and_title(self) -> None:
        composed = parse()[0].composed_text
        assert composed.startswith("Northwind Trading — Real-time trading dashboard")
        assert "react, typescript" in composed
        assert "Rebuilt the dashboard" in composed

    def test_content_hash_is_the_hash_of_composed_text(self) -> None:
        ev = parse()[0]
        assert ev.content_hash == content_hash(ev.composed_text)

    def test_source_file_is_recorded(self) -> None:
        assert parse()[0].source_file == "evidence/northwind.md"


class TestOptionalParts:
    def test_tags_line_is_optional(self) -> None:
        doc = dedent("""\
            ---
            company: Orchid Labs
            ---

            ## Mentoring
            Onboarded four junior engineers.
            """)
        ev = parse(doc, "evidence/orchid.md")[0]
        assert ev.tags == ()
        assert ev.body == "Onboarded four junior engineers."

    def test_dates_are_optional(self) -> None:
        doc = "---\ncompany: Orchid Labs\n---\n\n## Mentoring\nOnboarded engineers.\n"
        ev = parse(doc, "evidence/orchid.md")[0]
        assert ev.start is None
        assert ev.end is None


class TestFailsLoudly:
    def test_rejects_missing_front_matter(self) -> None:
        with pytest.raises(ParseError, match="front matter"):
            parse("## A heading\nSome prose.\n", "evidence/x.md")

    def test_rejects_file_with_no_blocks(self) -> None:
        with pytest.raises(ParseError, match="no '##' blocks"):
            parse("---\ncompany: X\n---\n\nJust prose, no heading.\n", "evidence/x.md")

    def test_rejects_duplicate_titles_in_one_file(self) -> None:
        doc = dedent("""\
            ---
            company: X
            ---

            ## Same title
            First body.

            ## Same title
            Second body.
            """)
        with pytest.raises(ParseError, match="duplicate"):
            parse(doc, "evidence/x.md")

    def test_rejects_block_with_empty_body(self) -> None:
        doc = "---\ncompany: X\n---\n\n## Empty block\n\n## Real block\nProse.\n"
        with pytest.raises(ParseError, match="empty body"):
            parse(doc, "evidence/x.md")

    def test_rejects_invalid_date_in_front_matter(self) -> None:
        doc = "---\ncompany: X\nstart: 2022-13\n---\n\n## Block\nProse.\n"
        with pytest.raises(ParseError, match="2022-13"):
            parse(doc, "evidence/x.md")
