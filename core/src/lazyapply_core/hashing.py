"""Content hashing.

Two different hashes do two different jobs in this system, and conflating them
is a bug waiting to happen:

- ``content_hash`` identifies *the text of a piece of evidence*. It drives the
  embedding cache: same text, same vector, no API call.
- ``role_id`` identifies *a job description*. It drives the requirement
  extraction cache: same role posting, same requirements, no API call.

Neither is an identity. Evidence identity is a readable slug (see
``domain.evidence``) precisely so that rewording a block does not break the
citations pointing at it.
"""

from __future__ import annotations

import hashlib
import re

_WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """Collapse whitespace so trivial formatting edits do not bust caches."""
    return _WHITESPACE.sub(" ", text).strip()


def content_hash(text: str) -> str:
    """Stable 16-hex-char digest of normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:16]
