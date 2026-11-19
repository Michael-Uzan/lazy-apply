"""Providers record what they spent.

Every provider takes a recorder and reports its own tokens and latency. Done
at the provider boundary rather than in the CLI because that is the only
place the real token counts exist -- anything further out would be an
estimate, and an estimated cost report is not worth printing.
"""

from dataclasses import dataclass, field
from typing import Any

import pytest

from lazyapply_core.domain import (
    Evidence,
    EvidencePack,
    Profile,
    Requirement,
    RequirementCategory,
    RequirementEvidence,
    RequirementKind,
    ScoredEvidence,
)
from lazyapply_core.evaluate.claude import ClaudeEvaluator
from lazyapply_core.evaluate.schema import Judgment
from lazyapply_core.retrieve.base import ExtractionError
from lazyapply_core.retrieve.claude import ClaudeExtractor
from lazyapply_core.retrieve.schema import ExtractedRequirement, ExtractedRole
from lazyapply_core.telemetry.base import Usage


class RecordingTelemetry:
    def __init__(self) -> None:
        self.events: list[Usage] = []

    def record(self, event: Usage) -> None:
        self.events.append(event)


@dataclass
class StubUsage:
    input_tokens: int = 1500
    output_tokens: int = 600


@dataclass
class StubMessages:
    parsed: Any = None
    error: Exception | None = None

    def parse(self, **kwargs: Any) -> Any:
        if self.error:
            raise self.error
        return type("Response", (), {"parsed_output": self.parsed, "usage": StubUsage()})()


@dataclass
class StubClient:
    messages: StubMessages = field(default_factory=StubMessages)


EXTRACTION = ExtractedRole(
    title="Engineer",
    requirements=[
        ExtractedRequirement(
            id="react",
            text="Built React applications",
            kind=RequirementKind.MUST_HAVE,
            category=RequirementCategory.SKILL,
        )
    ],
)

PROFILE = Profile(name="Dana", headline="Engineer", location="Lisbon", email="d@example.com")
PACK = EvidencePack(
    role_id="r1",
    items=(
        RequirementEvidence(
            requirement=Requirement(
                id="react",
                text="Built React applications",
                kind=RequirementKind.MUST_HAVE,
                category=RequirementCategory.SKILL,
            ),
            hits=(
                ScoredEvidence(
                    evidence=Evidence(
                        evidence_id="ev_test__dashboard",
                        title="Dashboard",
                        body="body",
                        composed_text="body",
                        content_hash="0" * 16,
                        source_file="t.md",
                    ),
                    score=0.5,
                ),
            ),
        ),
    ),
)


@pytest.fixture
def telemetry() -> RecordingTelemetry:
    return RecordingTelemetry()


class TestExtractor:
    def test_records_one_event_per_call(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=EXTRACTION))
        extractor = ClaudeExtractor(client=client, telemetry=telemetry, command="requirements")
        extractor.extract("A job.")
        assert len(telemetry.events) == 1

    def test_records_the_real_token_counts(self, telemetry: RecordingTelemetry) -> None:
        """From the response, never estimated from the prompt length."""
        client = StubClient(StubMessages(parsed=EXTRACTION))
        ClaudeExtractor(client=client, telemetry=telemetry).extract("A job.")
        event = telemetry.events[0]
        assert event.input_tokens == 1500
        assert event.output_tokens == 600

    def test_records_the_model_and_command(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=EXTRACTION))
        ClaudeExtractor(
            client=client, telemetry=telemetry, command="requirements", model="claude-opus-5"
        ).extract("A job.")
        event = telemetry.events[0]
        assert event.model == "claude-opus-5"
        assert event.command == "requirements"
        assert event.provider == "anthropic"

    def test_records_latency(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=EXTRACTION))
        ClaudeExtractor(client=client, telemetry=telemetry).extract("A job.")
        assert telemetry.events[0].latency_ms >= 0

    def test_computes_the_cost(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=EXTRACTION))
        ClaudeExtractor(client=client, telemetry=telemetry, model="claude-opus-5").extract("A job.")
        # 1500 in at $5/MTok + 600 out at $25/MTok
        assert telemetry.events[0].cost_usd == pytest.approx(0.0075 + 0.015)

    def test_a_failed_request_records_nothing(self, telemetry: RecordingTelemetry) -> None:
        """No response means no usage to read, and inventing one would put
        phantom spend in the report."""
        client = StubClient(StubMessages(error=RuntimeError("rate limit")))
        with pytest.raises(ExtractionError):
            ClaudeExtractor(client=client, telemetry=telemetry).extract("A job.")
        assert telemetry.events == []

    def test_a_refusal_is_still_recorded(self, telemetry: RecordingTelemetry) -> None:
        """The request succeeded and the tokens were spent -- the model just
        declined to fill the schema. Dropping it would understate the bill,
        and a refusal loop is exactly what someone would look at this report
        to find."""
        client = StubClient(StubMessages(parsed=None))
        with pytest.raises(ExtractionError):
            ClaudeExtractor(client=client, telemetry=telemetry).extract("A job.")
        assert len(telemetry.events) == 1
        assert telemetry.events[0].input_tokens == 1500

    def test_works_without_telemetry(self) -> None:
        """The recorder is optional everywhere; the default discards."""
        client = StubClient(StubMessages(parsed=EXTRACTION))
        assert ClaudeExtractor(client=client).extract("A job.") is EXTRACTION


class TestEvaluator:
    def test_records_one_event_per_judgment(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=Judgment(matches=[])))
        ClaudeEvaluator(client=client, telemetry=telemetry, command="fit").judge(PACK, PROFILE)
        assert len(telemetry.events) == 1

    def test_records_the_judgment_model(self, telemetry: RecordingTelemetry) -> None:
        client = StubClient(StubMessages(parsed=Judgment(matches=[])))
        ClaudeEvaluator(
            client=client, telemetry=telemetry, model="claude-haiku-4-5", command="fit"
        ).judge(PACK, PROFILE)
        event = telemetry.events[0]
        assert event.model == "claude-haiku-4-5"
        assert event.command == "fit"


class TestSharedRun:
    def test_both_calls_of_one_command_share_a_run_id(self, telemetry: RecordingTelemetry) -> None:
        """So the report can say what a fit check costs rather than what each
        provider call costs."""
        run = "shared-run"
        extractor_client = StubClient(StubMessages(parsed=EXTRACTION))
        evaluator_client = StubClient(StubMessages(parsed=Judgment(matches=[])))

        ClaudeExtractor(client=extractor_client, telemetry=telemetry, run=run).extract("A job.")
        ClaudeEvaluator(client=evaluator_client, telemetry=telemetry, run=run).judge(PACK, PROFILE)

        assert {e.run for e in telemetry.events} == {run}


class TestVoyage:
    def client(self, total_tokens: int = 240) -> Any:
        class StubVoyage:
            def embed(self, texts: list[str], **kwargs: Any) -> Any:
                return type(
                    "R",
                    (),
                    {"embeddings": [[0.1] * 512 for _ in texts], "total_tokens": total_tokens},
                )()

        return StubVoyage()

    def test_records_one_event_per_request(self, telemetry: RecordingTelemetry) -> None:
        from lazyapply_core.embed.voyage import VoyageEmbedder

        VoyageEmbedder(client=self.client(), telemetry=telemetry, command="ingest").embed_documents(
            ["a", "b"]
        )
        assert len(telemetry.events) == 1

    def test_records_the_tokens_the_provider_counted(self, telemetry: RecordingTelemetry) -> None:
        from lazyapply_core.embed.voyage import VoyageEmbedder

        VoyageEmbedder(client=self.client(240), telemetry=telemetry).embed_documents(["a"])
        event = telemetry.events[0]
        assert event.input_tokens == 240
        assert event.output_tokens == 0
        assert event.provider == "voyage"

    def test_costs_what_voyage_charges(self, telemetry: RecordingTelemetry) -> None:
        from lazyapply_core.embed.voyage import VoyageEmbedder

        VoyageEmbedder(client=self.client(1_000_000), telemetry=telemetry).embed_documents(["a"])
        assert telemetry.events[0].cost_usd == pytest.approx(0.02)


class TestCacheHits:
    def test_a_cache_hit_is_recorded_as_free(self, telemetry: RecordingTelemetry, tmp_path) -> None:
        """Recorded rather than ignored: the saving is only visible if the
        avoided request is in the log."""
        from lazyapply_core.embed.cache import CachedEmbedder
        from lazyapply_core.embed.fake import FakeEmbedder

        inner = FakeEmbedder(dimensions=8)
        cached = CachedEmbedder(inner, cache_dir=tmp_path, telemetry=telemetry, command="ingest")

        cached.embed_documents(["a", "b"])
        telemetry.events.clear()
        cached.embed_documents(["a", "b"])

        assert len(telemetry.events) == 1
        assert telemetry.events[0].cached is True
        assert telemetry.events[0].cost_usd == 0.0

    def test_a_partial_hit_records_only_what_was_served(
        self, telemetry: RecordingTelemetry, tmp_path
    ) -> None:
        from lazyapply_core.embed.cache import CachedEmbedder
        from lazyapply_core.embed.fake import FakeEmbedder

        cached = CachedEmbedder(FakeEmbedder(dimensions=8), cache_dir=tmp_path, telemetry=telemetry)
        cached.embed_documents(["a"])
        telemetry.events.clear()
        cached.embed_documents(["a", "b"])

        hits = [e for e in telemetry.events if e.cached]
        assert len(hits) == 1

    def test_no_event_when_nothing_was_cached(
        self, telemetry: RecordingTelemetry, tmp_path
    ) -> None:
        from lazyapply_core.embed.cache import CachedEmbedder
        from lazyapply_core.embed.fake import FakeEmbedder

        cached = CachedEmbedder(FakeEmbedder(dimensions=8), cache_dir=tmp_path, telemetry=telemetry)
        cached.embed_documents(["a", "b"])
        assert [e for e in telemetry.events if e.cached] == []
