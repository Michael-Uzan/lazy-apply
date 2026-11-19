"""`lazyapply eval` — the measured numbers.

Runs offline with the fake embedder and the score-based evaluator, against
the real extractions seeded from ``data/example/cache/``. The recall and MRR
numbers in that mode are meaningless -- the fake embedder carries no
semantics -- so what is checked here is that the command runs, reports both
metrics, and distinguishes "nothing to measure" from "measured zero". The
real numbers come from running it without ``--fake``.
"""

from pathlib import Path

from typer.testing import CliRunner

from lazyapply_core.cli import app
from lazyapply_core.config import REPO_ROOT

runner = CliRunner()
EXAMPLE = str(REPO_ROOT / "data/example")


def run(*args: str, cache: Path) -> object:
    return runner.invoke(
        app,
        ["eval", *args, "--data-dir", EXAMPLE, "--cache-dir", str(cache), "--fake", "--in-memory"],
    )


class TestOutput:
    def test_exits_cleanly(self, seeded_cache: Path) -> None:
        result = run(cache=seeded_cache)
        assert result.exit_code == 0, result.output

    def test_reports_recall_and_mrr(self, seeded_cache: Path) -> None:
        out = run(cache=seeded_cache).output.lower()
        assert "recall" in out
        assert "mrr" in out

    def test_names_both_postings(self, seeded_cache: Path) -> None:
        out = run(cache=seeded_cache).output
        assert "strong-match" in out
        assert "bad-match" in out

    def test_reports_agreement_with_the_labels(self, seeded_cache: Path) -> None:
        assert "agreement" in run(cache=seeded_cache).output.lower()

    def test_says_how_many_requirements_had_nothing_to_find(self, seeded_cache: Path) -> None:
        """Ten of twenty-one are excluded from recall. Hiding that would
        overstate what the number covers."""
        assert "no relevant evidence" in run(cache=seeded_cache).output.lower()

    def test_retrieval_only_skips_the_model(self, seeded_cache: Path) -> None:
        out = run("--retrieval-only", cache=seeded_cache).output.lower()
        assert "recall" in out
        assert "agreement" not in out


class TestErrors:
    def test_a_missing_golden_file_fails_cleanly(self, seeded_cache: Path) -> None:
        result = run("--golden", str(seeded_cache / "nope.yaml"), cache=seeded_cache)
        assert result.exit_code != 0
        assert "not found" in result.output
        assert "Traceback" not in result.output


def retrieval_rows(output: str, posting: str = "strong-match") -> list[str]:
    """Rows from the Retrieval table only -- the Thresholds table also names
    each posting."""
    section = output.split("Thresholds")[0]
    return [line for line in section.splitlines() if posting in line]


class TestSweep:
    def test_reports_every_cutoff(self, seeded_cache: Path) -> None:
        """One pack scored at k=1..5. A vector does not depend on k, so the
        sweep costs no extra embedding requests."""
        out = run("--sweep", "--retrieval-only", cache=seeded_cache).output
        assert len(retrieval_rows(out)) == 5

    def test_the_cutoffs_are_labelled(self, seeded_cache: Path) -> None:
        out = run("--sweep", "--retrieval-only", cache=seeded_cache).output
        cutoffs = [line.split()[1] for line in retrieval_rows(out)]
        assert cutoffs == ["1", "2", "3", "4", "5"]

    def test_a_single_cutoff_by_default(self, seeded_cache: Path) -> None:
        out = run("--retrieval-only", cache=seeded_cache).output
        assert len(retrieval_rows(out)) == 1


class TestFakeExtractionHint:
    def test_explains_how_to_get_the_real_requirement_ids(self, tmp_path: Path) -> None:
        """With an empty cache, --fake swaps in a canned extractor whose
        requirement ids differ from the labelled ones. The drift error is
        correct but says nothing about the cause, so it carries the fix."""
        result = runner.invoke(
            app,
            [
                "eval",
                "--data-dir",
                EXAMPLE,
                "--cache-dir",
                str(tmp_path),
                "--fake",
                "--in-memory",
                "--retrieval-only",
            ],
        )
        assert result.exit_code != 0
        assert "data/example/cache" in result.output
        assert "Traceback" not in result.output
