"""Turning human headings into machine-safe, readable id fragments.

Evidence ids are read by humans — in fit reports, in the golden eval set, in
debugging output — so they are slugs rather than hashes. That makes the slug
rules part of the data contract: change them and existing ids change with them.
"""

from __future__ import annotations

import re

_NOT_ALLOWED = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    """Lowercase ``text`` and reduce it to ``a-z``, ``0-9`` and single hyphens.

    Anything else — punctuation, em-dashes, smart quotes, accented or
    non-Latin letters — is treated as a separator and collapsed away.

    Raises:
        ValueError: if nothing usable survives. A heading that cannot produce a
            slug cannot produce a citable id, so ingest stops rather than
            inventing one.
    """
    slug = _NOT_ALLOWED.sub("-", text.lower()).strip("-")
    if not slug:
        raise ValueError(f"title has no usable characters for a slug: {text!r}")
    return slug
