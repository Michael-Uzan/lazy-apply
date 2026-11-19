"""Role text -> structured, checkable requirements.

The highest-leverage call in the pipeline. Misclassifying "fintech a plus" as
a must-have drags a good candidate to SKIP, so the extraction result is
validated hard and cached forever by role hash.
"""

from pathlib import Path

import pytest

from lazyapply_core.config import REPO_ROOT
from lazyapply_core.domain import RequirementCategory, RequirementKind
from lazyapply_core.retrieve.fake import FakeExtractor
from lazyapply_core.retrieve.requirements import ExtractionError, extract_role
from lazyapply_core.retrieve.schema import ExtractedRequirement, ExtractedRole

ROLE_TEXT = (REPO_ROOT / "data/example/roles/strong-match.txt").read_text()


def extracted(*reqs: ExtractedRequirement) -> ExtractedRole:
    return ExtractedRole(
        title="Senior Full-Stack Engineer", company="Meridian", requirements=list(reqs)
    )


def req(rid: str, kind: RequirementKind = RequirementKind.MUST_HAVE) -> ExtractedRequirement:
    return ExtractedRequirement(
        id=rid, text=f"Did {rid}", kind=kind, category=RequirementCategory.SKILL
    )


class TestMapping:
    def test_builds_a_role_with_the_extracted_fields(self, tmp_path: Path) -> None:
        role = extract_role(
            ROLE_TEXT, extractor=FakeExtractor(extracted(req("react"))), cache_dir=tmp_path
        )
        assert role.title == "Senior Full-Stack Engineer"
        assert role.company == "Meridian"
        assert [r.id for r in role.requirements] == ["react"]

    def test_keeps_the_raw_text(self, tmp_path: Path) -> None:
        role = extract_role(
            ROLE_TEXT, extractor=FakeExtractor(extracted(req("react"))), cache_dir=tmp_path
        )
        assert role.raw_text == ROLE_TEXT

    def test_role_id_is_the_hash_of_the_raw_text(self, tmp_path: Path) -> None:
        from lazyapply_core.hashing import content_hash

        role = extract_role(
            ROLE_TEXT, extractor=FakeExtractor(extracted(req("react"))), cache_dir=tmp_path
        )
        assert role.role_id == content_hash(ROLE_TEXT)

    def test_splits_must_haves_from_nice_to_haves(self, tmp_path: Path) -> None:
        role = extract_role(
            ROLE_TEXT,
            extractor=FakeExtractor(
                extracted(req("react"), req("fintech", RequirementKind.NICE_TO_HAVE))
            ),
            cache_dir=tmp_path,
        )
        assert [r.id for r in role.must_haves] == ["react"]
        assert [r.id for r in role.nice_to_haves] == ["fintech"]


class TestValidation:
    def test_rejects_an_extraction_with_no_requirements(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError, match="no requirements"):
            extract_role(ROLE_TEXT, extractor=FakeExtractor(extracted()), cache_dir=tmp_path)

    def test_rejects_an_extraction_with_no_must_haves(self, tmp_path: Path) -> None:
        """Coverage divides by the must-have count; zero would be undefined,
        and a posting with no hard requirement is a misread, not a real role."""
        with pytest.raises(ExtractionError, match="no must-have"):
            extract_role(
                ROLE_TEXT,
                extractor=FakeExtractor(extracted(req("x", RequirementKind.NICE_TO_HAVE))),
                cache_dir=tmp_path,
            )

    def test_rejects_duplicate_requirement_ids(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError, match="duplicate"):
            extract_role(
                ROLE_TEXT,
                extractor=FakeExtractor(extracted(req("react"), req("react"))),
                cache_dir=tmp_path,
            )

    def test_rejects_empty_role_text(self, tmp_path: Path) -> None:
        with pytest.raises(ExtractionError, match="empty"):
            extract_role("   ", extractor=FakeExtractor(extracted(req("x"))), cache_dir=tmp_path)


class TestCaching:
    def test_first_call_reaches_the_extractor(self, tmp_path: Path) -> None:
        fake = FakeExtractor(extracted(req("react")))
        extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        assert fake.calls == 1

    def test_second_call_does_not(self, tmp_path: Path) -> None:
        fake = FakeExtractor(extracted(req("react")))
        extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        assert fake.calls == 1

    def test_cached_role_is_identical(self, tmp_path: Path) -> None:
        fake = FakeExtractor(extracted(req("react"), req("ts", RequirementKind.NICE_TO_HAVE)))
        first = extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        assert extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path) == first

    def test_a_different_role_is_a_cache_miss(self, tmp_path: Path) -> None:
        fake = FakeExtractor(extracted(req("react")))
        extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        extract_role("A different posting entirely.", extractor=fake, cache_dir=tmp_path)
        assert fake.calls == 2

    def test_whitespace_changes_do_not_bust_the_cache(self, tmp_path: Path) -> None:
        """Reflowing a pasted posting must not pay for a second extraction."""
        fake = FakeExtractor(extracted(req("react")))
        extract_role(ROLE_TEXT, extractor=fake, cache_dir=tmp_path)
        extract_role(ROLE_TEXT.replace("\n", "\n\n"), extractor=fake, cache_dir=tmp_path)
        assert fake.calls == 1

    def test_survives_a_new_process(self, tmp_path: Path) -> None:
        extract_role(
            ROLE_TEXT, extractor=FakeExtractor(extracted(req("react"))), cache_dir=tmp_path
        )
        second = FakeExtractor(extracted(req("react")))
        extract_role(ROLE_TEXT, extractor=second, cache_dir=tmp_path)
        assert second.calls == 0
