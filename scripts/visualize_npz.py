#!/usr/bin/env python3
"""Visualize NPZ data as PNG.

NOTE: Keep this script as the stable CLI entrypoint.
Reusable logic lives in `src/platform/viz/npz_viz.py`.
"""

import sys
from pathlib import Path


# Ensure project root is importable when running as `python scripts/visualize_npz.py ...`.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.platform.viz.npz_viz import visualize_npz  # noqa: E402

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_npz.py <input.npz> [output.png]")
        sys.exit(1)
    
    npz_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    visualize_npz(npz_path, output_path)
