"""The Claude-backed extractor's own wiring, with the SDK client injected."""

from dataclasses import dataclass, field
from typing import Any

import pytest

from lazyapply_core.domain import RequirementCategory, RequirementKind
from lazyapply_core.retrieve.base import ExtractionError
from lazyapply_core.retrieve.claude import ClaudeExtractor
from lazyapply_core.retrieve.schema import ExtractedRequirement, ExtractedRole

RESULT = ExtractedRole(
    title="Senior Full-Stack Engineer",
    company="Meridian",
    requirements=[
        ExtractedRequirement(
            id="react",
            text="Built production web applications with React",
            kind=RequirementKind.MUST_HAVE,
            category=RequirementCategory.SKILL,
        )
    ],
)


@dataclass
class StubMessages:
    calls: list[dict[str, Any]] = field(default_factory=list)
    error: Exception | None = None

    def parse(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return type("Response", (), {"parsed_output": RESULT})()


@dataclass
class StubClient:
    messages: StubMessages = field(default_factory=StubMessages)


@pytest.fixture
def stub() -> StubClient:
    return StubClient()


class TestRequest:
    def test_uses_the_configured_model(self, stub: StubClient) -> None:
        ClaudeExtractor(client=stub, model="claude-opus-5").extract("A posting.")
        assert stub.messages.calls[0]["model"] == "claude-opus-5"

    def test_sends_the_role_text(self, stub: StubClient) -> None:
        ClaudeExtractor(client=stub).extract("Needs 5 years of React.")
        content = stub.messages.calls[0]["messages"][0]["content"]
        assert "Needs 5 years of React." in content

    def test_asks_for_the_extraction_schema(self, stub: StubClient) -> None:
        """Structured output: the result arrives typed, not JSON-parsed and prayed over."""
        ClaudeExtractor(client=stub).extract("A posting.")
        assert stub.messages.calls[0]["output_format"] is ExtractedRole

    def test_sends_a_system_prompt(self, stub: StubClient) -> None:
        ClaudeExtractor(client=stub).extract("A posting.")
        system = stub.messages.calls[0]["system"].lower()
        assert "must_have" in system and "nice_to_have" in system

    def test_returns_the_parsed_output(self, stub: StubClient) -> None:
        assert ClaudeExtractor(client=stub).extract("A posting.") is RESULT


class TestFailures:
    def test_provider_errors_become_extraction_errors(self, stub: StubClient) -> None:
        stub.messages.error = RuntimeError("rate limit exceeded")
        with pytest.raises(ExtractionError, match="rate limit"):
            ClaudeExtractor(client=stub).extract("A posting.")

    def test_the_original_error_is_kept_as_the_cause(self, stub: StubClient) -> None:
        original = RuntimeError("boom")
        stub.messages.error = original
        with pytest.raises(ExtractionError) as info:
            ClaudeExtractor(client=stub).extract("A posting.")
        assert info.value.__cause__ is original

    def test_a_missing_api_key_is_reported_clearly(self) -> None:
        with pytest.raises(ExtractionError, match="ANTHROPIC_API_KEY"):
            ClaudeExtractor(api_key="")

    def test_a_refusal_is_reported_rather_than_returning_none(self, stub: StubClient) -> None:
        """Opus 5 can decline; parsed_output is then None and must not be
        passed on as if it were a role."""
        stub.messages.parse = lambda **kw: type("R", (), {"parsed_output": None})()
        with pytest.raises(ExtractionError, match="no structured output"):
            ClaudeExtractor(client=stub).extract("A posting.")
