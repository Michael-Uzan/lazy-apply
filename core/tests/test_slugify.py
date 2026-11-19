"""Slug rules decide what an evidence_id looks like, so they are pinned hard."""

import pytest

from lazyapply_core.ingest.slug import slugify


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Real-time trading dashboard", "real-time-trading-dashboard"),
        ("GraphQL federation across four services", "graphql-federation-across-four-services"),
        ("Python ETL for clinical data", "python-etl-for-clinical-data"),
        # Punctuation is dropped, not transliterated into noise.
        ("CI/CD pipeline", "ci-cd-pipeline"),
        ("Mentoring & code review culture", "mentoring-code-review-culture"),
        ("Cut p99 latency by 40%", "cut-p99-latency-by-40"),
        # Em-dashes and smart quotes come free with copy-paste from docs.
        ("Search — relevance tuning", "search-relevance-tuning"),
        ("Rebuilt the “dashboard”", "rebuilt-the-dashboard"),
        # Runs of separators collapse; edges are trimmed.
        ("  Leading and trailing  ", "leading-and-trailing"),
        ("Multiple   spaces", "multiple-spaces"),
        ("Hyphen--heavy---title", "hyphen-heavy-title"),
    ],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_rejects_title_with_no_usable_characters() -> None:
    """A heading of pure punctuation cannot produce a valid id, so it must fail
    loudly at ingest rather than silently yield an empty slug."""
    with pytest.raises(ValueError, match="no usable characters"):
        slugify("!!! ???")
