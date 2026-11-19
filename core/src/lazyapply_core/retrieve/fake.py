"""A canned extractor for tests.

Returns whatever it was constructed with, and counts calls so tests can prove
the cache prevented work rather than merely returned the right answer.
"""

from __future__ import annotations

from lazyapply_core.retrieve.schema import ExtractedRole


class FakeExtractor:
    def __init__(self, result: ExtractedRole) -> None:
        self._result = result
        self.calls = 0

    def extract(self, role_text: str) -> ExtractedRole:
        self.calls += 1
        return self._result


def canned_extractor() -> FakeExtractor:
    """A plausible extraction for the sample full-stack role.

    Lets the whole command path run with no key and no cost. The shape is
    realistic; the content is not meant to be correct for arbitrary postings.
    """
    from lazyapply_core.domain import RequirementCategory as C
    from lazyapply_core.domain import RequirementKind as K
    from lazyapply_core.retrieve.schema import ExtractedRequirement

    def r(rid: str, text: str, kind: K, category: C) -> ExtractedRequirement:
        return ExtractedRequirement(id=rid, text=text, kind=kind, category=category)

    return FakeExtractor(
        ExtractedRole(
            title="Senior Full-Stack Engineer",
            company="Meridian Systems",
            requirements=[
                r(
                    "react",
                    "Built production web applications with React and TypeScript",
                    K.MUST_HAVE,
                    C.SKILL,
                ),
                r("api_design", "Designed and evolved APIs, GraphQL or REST", K.MUST_HAVE, C.SKILL),
                r("nodejs", "Ran Node.js services in production", K.MUST_HAVE, C.SKILL),
                r("postgres", "Modelled data in PostgreSQL", K.MUST_HAVE, C.SKILL),
                r("realtime", "Delivered real-time data to the browser", K.MUST_HAVE, C.EXPERIENCE),
                r(
                    "performance",
                    "Delivered measurable performance improvements",
                    K.MUST_HAVE,
                    C.EXPERIENCE,
                ),
                r("fintech", "Worked on fintech or trading systems", K.NICE_TO_HAVE, C.DOMAIN),
                r(
                    "event_driven",
                    "Built event-driven systems with message queues",
                    K.NICE_TO_HAVE,
                    C.EXPERIENCE,
                ),
                r("mentoring", "Mentored engineers or led a team", K.NICE_TO_HAVE, C.OTHER),
            ],
        )
    )
