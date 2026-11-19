"""Token prices and the cost arithmetic.

Prices are published numbers, so the risk here is not the maths -- it is a
stale or invented figure. Every entry carries the date it was checked, and an
unknown model raises rather than silently costing nothing: a cost report that
under-reports is worse than one that refuses to guess.
"""

import pytest

from lazyapply_core.telemetry.pricing import (
    PRICES,
    UnknownModelError,
    cost_usd,
    price_for,
)


class TestPriceTable:
    def test_knows_the_extraction_model(self) -> None:
        price = price_for("claude-opus-5")
        assert price.input_per_million == 5.0
        assert price.output_per_million == 25.0

    def test_knows_the_judgment_model(self) -> None:
        price = price_for("claude-haiku-4-5")
        assert price.input_per_million == 1.0
        assert price.output_per_million == 5.0

    def test_knows_the_embedding_model(self) -> None:
        """Embeddings have no output tokens, so the output price is zero
        rather than absent -- the arithmetic stays uniform."""
        price = price_for("voyage-3-lite")
        assert price.input_per_million == 0.02
        assert price.output_per_million == 0.0

    def test_an_unknown_model_raises_rather_than_costing_nothing(self) -> None:
        with pytest.raises(UnknownModelError, match="claude-imaginary-9"):
            price_for("claude-imaginary-9")

    def test_every_entry_records_when_it_was_checked(self) -> None:
        """A price with no date cannot be audited, and these go in a README."""
        assert all(price.checked for price in PRICES.values())

    def test_the_fake_models_are_free(self) -> None:
        """The offline fakes make no request, so they must not inflate a
        report with imaginary spend."""
        assert cost_usd("fake", input_tokens=1_000_000, output_tokens=1_000_000) == 0.0


class TestCost:
    def test_a_million_input_tokens_costs_the_input_price(self) -> None:
        assert cost_usd("claude-opus-5", input_tokens=1_000_000, output_tokens=0) == 5.0

    def test_a_million_output_tokens_costs_the_output_price(self) -> None:
        assert cost_usd("claude-opus-5", input_tokens=0, output_tokens=1_000_000) == 25.0

    def test_input_and_output_are_added(self) -> None:
        assert cost_usd("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=1_000_000) == 6.0

    def test_a_realistic_call(self) -> None:
        """One fit check: roughly 2k in, 800 out on Haiku."""
        got = cost_usd("claude-haiku-4-5", input_tokens=2_000, output_tokens=800)
        assert got == pytest.approx(0.006)

    def test_zero_tokens_costs_nothing(self) -> None:
        assert cost_usd("claude-opus-5", input_tokens=0, output_tokens=0) == 0.0

    def test_output_tokens_dominate(self) -> None:
        """Five times the price on every model here, which is why the
        extraction prompt caps its own output."""
        cheap = cost_usd("claude-haiku-4-5", input_tokens=1000, output_tokens=0)
        dear = cost_usd("claude-haiku-4-5", input_tokens=0, output_tokens=1000)
        assert dear == pytest.approx(5 * cheap)
