"""Helpers for generating a simple HTML index for visualization outputs."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Iterable, Optional


def write_image_index_html(
    image_paths: Iterable[str | Path],
    out_html: str | Path,
    title: str = "Visualization Index",
    rel_to: Optional[str | Path] = None,
) -> Path:
    """Write a minimal HTML page that displays images.

    Args:
        image_paths: Iterable of image file paths.
        out_html: Output HTML path.
        title: Page title.
        rel_to: If provided, image links are made relative to this directory.

    Returns:
        Path to the written HTML file.
    """

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    rel_base = Path(rel_to) if rel_to is not None else out_html.parent

    items = []
    for p in image_paths:
        p = Path(p)
        href = str(p.relative_to(rel_base) if p.is_absolute() and rel_base in p.parents else p)
        items.append(href)

    body_lines = [
        "<!doctype html>",
        "<html>",
        "<head>",
        "  <meta charset='utf-8' />",
        f"  <title>{html.escape(title)}</title>",
        "  <style>",
        "    body { font-family: sans-serif; }",
        "    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 12px; }",
        "    .card { border: 1px solid #ddd; padding: 8px; }",
        "    img { width: 100%; height: auto; }",
        "    .path { font-family: monospace; font-size: 12px; color: #333; word-break: break-all; }",
        "  </style>",
        "</head>",
        "<body>",
        f"<h1>{html.escape(title)}</h1>",
        "<div class='grid'>",
    ]

    for href in items:
        esc = html.escape(href)
        body_lines += [
            "  <div class='card'>",
            f"    <div class='path'>{esc}</div>",
            f"    <a href='{esc}' target='_blank' rel='noopener'>",
            f"      <img src='{esc}' loading='lazy' />",
            "    </a>",
            "  </div>",
        ]

    body_lines += ["</div>", "</body>", "</html>"]

    out_html.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    return out_html
