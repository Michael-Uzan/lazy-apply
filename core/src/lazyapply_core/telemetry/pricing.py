"""Published token prices, and the cost arithmetic over them.

The risk in this module is not the multiplication -- it is a stale or invented
number. So every entry records the date it was checked against Anthropic's and
Voyage's own pricing pages, and an unrecognised model raises instead of
costing nothing. A cost report that quietly under-reports is worse than one
that refuses to guess.

Output tokens cost five times input on every Claude model here. That ratio is
why the extraction prompt asks for 5-12 requirements rather than an
exhaustive list, and why the matching prompt asks for one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

CHECKED = "2026-11-19"
"""When these were last read from the providers' pricing pages."""

FREE_MODELS = frozenset({"fake"})
"""The offline fakes. They make no request, so they must not show up as spend."""


class UnknownModelError(Exception):
    """No published price for this model."""


@dataclass(frozen=True)
class Price:
    """USD per million tokens."""

    input_per_million: float
    output_per_million: float
    checked: str
    """Date this was verified. A price with no date cannot be audited."""


PRICES: dict[str, Price] = {
    # platform.claude.com/docs/en/about-claude/pricing
    "claude-opus-5": Price(5.0, 25.0, CHECKED),
    "claude-sonnet-5": Price(2.0, 10.0, CHECKED),
    "claude-haiku-4-5": Price(1.0, 5.0, CHECKED),
    # docs.voyageai.com/docs/pricing -- embeddings produce no output tokens,
    # so the output price is zero rather than absent, keeping the arithmetic
    # uniform across providers.
    "voyage-3-lite": Price(0.02, 0.0, CHECKED),
}


def price_for(model: str) -> Price:
    """Look up a model's published price.

    Raises:
        UnknownModelError: if the model is not in the table. Changing a model
            in config without adding its price should break the cost report
            loudly, not shift it silently.
    """
    try:
        return PRICES[model]
    except KeyError as exc:
        known = ", ".join(sorted(PRICES))
        raise UnknownModelError(f"no published price for {model!r}; known: {known}") from exc


def cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float:
    """What one call cost, in USD."""
    if model in FREE_MODELS:
        return 0.0

    price = price_for(model)
    return (
        input_tokens * price.input_per_million + output_tokens * price.output_per_million
    ) / 1_000_000
