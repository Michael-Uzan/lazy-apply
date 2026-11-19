"""Estimating token counts before a request is made.

Rough on purpose. It answers "cents or dollars?" before spending, and prices
the requests a cache avoided -- a hit has no provider count to read, because
there was no request. Anything actually billed is recorded from the provider's
own count instead.
"""

from __future__ import annotations

from collections.abc import Sequence

CHARS_PER_TOKEN = 4
"""Rough bytes-per-token for English prose."""


def estimate_tokens(texts: Sequence[str]) -> int:
    """Approximate tokens in ``texts``."""
    return int(sum(len(text) for text in texts) / CHARS_PER_TOKEN)
