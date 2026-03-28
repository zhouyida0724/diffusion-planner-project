#!/usr/bin/env python3
"""Smoke test for scripts/visualize_npz.py.

This intentionally exercises the *script* entrypoint to ensure imports and CLI
behavior remain intact.

It will:
  - locate one pre-generated single_npz under validation_output2/
  - run scripts/visualize_npz.py <npz> <out_png>
  - assert the PNG exists

Output goes to outputs/viz/smoke/ by default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    candidates = sorted(repo_root.glob("validation_output2/boston50w_viz_50_*/slice*/single_npz/*.npz"))
    if not candidates:
        raise SystemExit("No candidate NPZ files found under validation_output2/boston50w_viz_50_*/...")

    npz_path = candidates[0]
    out_dir = repo_root / "outputs/viz/smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{npz_path.stem}.png"

    import sys

    cmd = [sys.executable, str(repo_root / "scripts/visualize_npz.py"), str(npz_path), str(out_png)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd, cwd=repo_root)

    if not out_png.exists():
        raise SystemExit(f"Expected output PNG missing: {out_png}")

    print("OK - wrote", out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
