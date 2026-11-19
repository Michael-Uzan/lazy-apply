"""The Haiku-backed evaluator's wiring, with the SDK client injected.

Every assertion here runs with no API key and no network. What is being
verified is the request shape -- model, prompt contents, schema -- and that
provider failures arrive as our own error type rather than leaking a traceback
out of the CLI.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from lazyapply_core.domain import (
    Evidence,
    EvidencePack,
    MatchStatus,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    ScoredEvidence,
)
from lazyapply_core.evaluate.base import EvaluationError
from lazyapply_core.evaluate.claude import ClaudeEvaluator
from lazyapply_core.evaluate.schema import JudgedRequirement, Judgment

RESULT = Judgment(
    matches=[
        JudgedRequirement(
            requirement_id="react",
            status=MatchStatus.STRONG,
            evidence_ids=["ev_test__dashboard"],
            rationale="Rebuilt a trading dashboard in React.",
        )
    ]
)

PROFILE = Profile(
    name="Dana Okonkwo",
    headline="Senior Full-Stack Engineer",
    location="Lisbon",
    email="dana@example.com",
)

PACK = EvidencePack(
    role_id="r1",
    items=(
        RequirementEvidence(
            requirement=Requirement(
                id="react",
                text="Built production web applications with React",
                kind=RequirementKind.MUST_HAVE,
                category=RequirementCategory.SKILL,
            ),
            hits=(
                ScoredEvidence(
                    evidence=Evidence(
                        evidence_id="ev_test__dashboard",
                        title="Dashboard",
                        body="Rebuilt the trading dashboard in React.",
                        composed_text="Rebuilt the trading dashboard in React.",
                        content_hash="0" * 16,
                        source_file="test.md",
                    ),
                    score=0.56,
                ),
            ),
        ),
    ),
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


def judge(stub: StubClient, **kwargs: Any) -> Judgment:
    return ClaudeEvaluator(client=stub, **kwargs).judge(PACK, PROFILE)


class TestRequest:
    def test_uses_the_configured_model(self, stub: StubClient) -> None:
        """Matching runs on every check, unlike extraction, so it runs on the
        cheap model."""
        judge(stub, model="claude-haiku-4-5")
        assert stub.messages.calls[0]["model"] == "claude-haiku-4-5"

    def test_sends_the_hard_facts(self, stub: StubClient) -> None:
        judge(stub)
        content = stub.messages.calls[0]["messages"][0]["content"]
        assert "Dana Okonkwo" in content

    def test_sends_the_evidence_with_its_citable_id(self, stub: StubClient) -> None:
        judge(stub)
        content = stub.messages.calls[0]["messages"][0]["content"]
        assert "ev_test__dashboard" in content
        assert "Rebuilt the trading dashboard in React." in content

    def test_sends_the_requirements(self, stub: StubClient) -> None:
        judge(stub)
        content = stub.messages.calls[0]["messages"][0]["content"]
        assert "react" in content
        assert "must-have" in content

    def test_asks_for_the_judgment_schema(self, stub: StubClient) -> None:
        judge(stub)
        assert stub.messages.calls[0]["output_format"] is Judgment

    def test_the_system_prompt_states_the_three_statuses(self, stub: StubClient) -> None:
        judge(stub)
        system = stub.messages.calls[0]["system"]
        assert "STRONG" in system and "PARTIAL" in system and "NONE" in system

    def test_returns_the_parsed_output(self, stub: StubClient) -> None:
        assert judge(stub) is RESULT


class TestFailures:
    def test_provider_errors_become_evaluation_errors(self, stub: StubClient) -> None:
        stub.messages.error = RuntimeError("rate limit exceeded")
        with pytest.raises(EvaluationError, match="rate limit"):
            judge(stub)

    def test_the_original_error_is_kept_as_the_cause(self, stub: StubClient) -> None:
        original = RuntimeError("boom")
        stub.messages.error = original
        with pytest.raises(EvaluationError) as info:
            judge(stub)
        assert info.value.__cause__ is original

    def test_a_missing_api_key_is_reported_clearly(self) -> None:
        with pytest.raises(EvaluationError, match="ANTHROPIC_API_KEY"):
            ClaudeEvaluator(api_key="")

    def test_a_refusal_is_reported_rather_than_returning_none(self, stub: StubClient) -> None:
        stub.messages.parse = lambda **kw: type("R", (), {"parsed_output": None})()
        with pytest.raises(EvaluationError, match="no structured output"):
            judge(stub)
