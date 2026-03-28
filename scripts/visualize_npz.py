#!/usr/bin/env python3
"""Wrapper for backwards-compatible entrypoint.

This file remains at scripts/visualize_npz.py for stable CLI usage.
Implementation lives in scripts/export/diffusion_planner/viz/visualize_npz.py.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = (
        Path(__file__).resolve().parent
        / "export"
        / "diffusion_planner"
        / "viz"
        / "visualize_npz.py"
    )
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
