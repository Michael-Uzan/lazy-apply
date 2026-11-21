#!/usr/bin/env python3
"""Generate the README architecture diagrams, light and dark.

GitHub renders markdown on either a light or a dark ground and gives a diagram
no way to adapt, so each diagram ships as two files and the README picks
between them with a ``<picture>`` element.

Both files come from one source: ``setup.svg`` and ``check.svg`` are authored
once using ``var(--token)`` colours, and this script resolves those tokens
against each palette. Editing a diagram means editing one file, not four.

    python docs/diagrams/build.py

Fonts are system stacks rather than webfonts: GitHub serves these through an
``<img>``, which cannot fetch anything.
"""

from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).parent

SOURCES = ("setup", "check")

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "surface": "#FBFCFB",
        "node": "#FFFFFF",
        "chip": "#F1F4F2",
        "ink": "#13181C",
        "muted": "#5D6A70",
        "faint": "#8A959A",
        "rule": "#D5DCD9",
        "rule-soft": "#E4E9E6",
        "llm": "#6D3FC4",
        "embed": "#1D5FD6",
        "store": "#0F766E",
        "facts": "#A85412",
        "code": "#475A66",
        "go": "#166534",
        "hold": "#92650A",
        "stop": "#A42A22",
    },
    "dark": {
        "surface": "#151B1E",
        "node": "#1B2327",
        "chip": "#1E272B",
        "ink": "#E7ECEA",
        "muted": "#97A4A7",
        "faint": "#6E7C80",
        "rule": "#2C373B",
        "rule-soft": "#232D31",
        "llm": "#B49BF0",
        "embed": "#6FA6F5",
        "store": "#2DD4BF",
        "facts": "#F3A74B",
        "code": "#9BAEB8",
        "go": "#4ADE80",
        "hold": "#F0C14B",
        "stop": "#F07A6E",
    },
}

SANS = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

STYLE = """
  .n-title {{ font-family: {sans}; font-size: 14px; font-weight: 600; fill: {ink}; }}
  .n-sub   {{ font-family: {mono}; font-size: 11px; fill: {muted}; }}
  .n-band  {{ font-family: {mono}; font-size: 11px; letter-spacing: 0.12em; fill: {faint}; }}
  .n-edge  {{ font-family: {mono}; font-size: 10.5px; fill: {muted}; }}
  .n-data  {{ font-family: {mono}; font-size: 11px; fill: {ink}; }}
  .n-box   {{ fill: {node}; stroke: {rule}; stroke-width: 1; }}
  .n-flow  {{ stroke: {rule}; stroke-width: 1.5; fill: none; }}
"""

_TOKEN = re.compile(r"var\(--([a-z-]+)\)")


def resolve(svg: str, palette: dict[str, str]) -> str:
    """Replace every ``var(--token)`` with its hex value.

    Raises:
        KeyError: on a token with no entry in the palette. A silent fallback
            would render as black-on-black in exactly one theme, which is the
            bug this whole two-file arrangement exists to avoid.
    """

    def swap(match: re.Match[str]) -> str:
        return palette[match.group(1)]

    return _TOKEN.sub(swap, svg)


def build(name: str, theme: str) -> Path:
    palette = PALETTES[theme]
    svg = (HERE / f"{name}.svg").read_text(encoding="utf-8")

    style = STYLE.format(sans=SANS, mono=MONO, **palette)
    body = resolve(svg, palette)

    # Paint the ground explicitly: an <img> on GitHub composites over whichever
    # theme the reader has, and a transparent diagram borrows it.
    opening, rest = body.split(">", 1)
    body = (
        f"{opening}>"
        f'<style>{style}</style>'
        f'<rect width="100%" height="100%" fill="{palette["surface"]}"/>'
        f"{rest}"
    )

    out = HERE / f"{name}-{theme}.svg"
    out.write_text(body, encoding="utf-8")
    return out


def main() -> None:
    for name in SOURCES:
        for theme in PALETTES:
            print(f"wrote {build(name, theme).relative_to(HERE.parents[1])}")


if __name__ == "__main__":
    main()
