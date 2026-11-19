"""Loading a directory of evidence files."""

from pathlib import Path
from textwrap import dedent

import pytest

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.ingest.loader import load_evidence_dir
from lazyapply_core.ingest.parser import ParseError

EXAMPLE_DIR = REPO_ROOT / "data/example/evidence"


def write(directory: Path, name: str, title: str, body: str = "Some prose.") -> None:
    directory.joinpath(name).write_text(
        dedent(f"""\
            ---
            company: Example Co
            ---

            ## {title}
            {body}
            """)
    )


class TestLoading:
    def test_loads_every_markdown_file(self, tmp_path: Path) -> None:
        write(tmp_path, "alpha.md", "First story")
        write(tmp_path, "beta.md", "Second story")
        assert len(load_evidence_dir(tmp_path)) == 2

    def test_is_ordered_deterministically(self, tmp_path: Path) -> None:
        """Ingest order must not depend on filesystem iteration order."""
        write(tmp_path, "zulu.md", "Z story")
        write(tmp_path, "alpha.md", "A story")
        ids = [ev.evidence_id for ev in load_evidence_dir(tmp_path)]
        assert ids == ["ev_alpha__a-story", "ev_zulu__z-story"]

    def test_ignores_non_markdown_files(self, tmp_path: Path) -> None:
        write(tmp_path, "alpha.md", "First story")
        tmp_path.joinpath("notes.txt").write_text("not evidence")
        tmp_path.joinpath("README").write_text("not evidence")
        assert len(load_evidence_dir(tmp_path)) == 1


class TestFailsLoudly:
    def test_rejects_missing_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match="does not exist"):
            load_evidence_dir(tmp_path / "nope")

    def test_rejects_empty_directory(self, tmp_path: Path) -> None:
        with pytest.raises(ParseError, match=r"no \.md files"):
            load_evidence_dir(tmp_path)

    def test_rejects_same_title_in_two_files_with_same_stem_slug(self, tmp_path: Path) -> None:
        """'my-co.md' and 'my_co.md' both slug to 'my-co', so their blocks
        would collide into one id. A citation must map to exactly one block."""
        write(tmp_path, "my-co.md", "Shared title")
        write(tmp_path, "my_co.md", "Shared title")
        with pytest.raises(ParseError, match="duplicate evidence id"):
            load_evidence_dir(tmp_path)

    def test_names_the_offending_file(self, tmp_path: Path) -> None:
        write(tmp_path, "good.md", "Fine story")
        tmp_path.joinpath("broken.md").write_text("## No front matter\nProse.\n")
        with pytest.raises(ParseError, match=r"broken\.md"):
            load_evidence_dir(tmp_path)


class TestAgainstCommittedSample:
    def test_example_evidence_loads(self) -> None:
        """The sample data is the quickstart; it must stay parseable."""
        evidence = load_evidence_dir(EXAMPLE_DIR)
        assert len(evidence) == 6

    def test_example_ids_are_readable(self) -> None:
        ids = {ev.evidence_id for ev in load_evidence_dir(EXAMPLE_DIR)}
        assert "ev_northwind__real-time-trading-dashboard" in ids
        assert "ev_orchid__internal-search-over-research-documents" in ids

    def test_source_file_is_relative_to_the_directory(self) -> None:
        sources = {ev.source_file for ev in load_evidence_dir(EXAMPLE_DIR)}
        assert sources == {"northwind.md", "orchid.md"}
