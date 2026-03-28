#!/usr/bin/env python3
"""Generate a simple HTML index for a directory of PNG visualizations.

This is an auxiliary script (doesn't change existing user entrypoints).

Defaults:
  - input: outputs/viz
  - output: outputs/viz/index.html

Usage:
  python scripts/generate_viz_html_index.py [--input-dir DIR] [--out-html PATH]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.platform.viz.html_index import write_image_index_html  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=Path("outputs/viz"))
    ap.add_argument("--out-html", type=Path, default=Path("outputs/viz/index.html"))
    args = ap.parse_args()

    input_dir: Path = args.input_dir
    out_html: Path = args.out_html

    pngs = sorted(input_dir.rglob("*.png"))
    if not pngs:
        print(f"No PNGs found under: {input_dir}")

    write_image_index_html(pngs, out_html, title=f"Viz Index - {input_dir}", rel_to=out_html.parent)
    print(f"Wrote: {out_html} ({len(pngs)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
