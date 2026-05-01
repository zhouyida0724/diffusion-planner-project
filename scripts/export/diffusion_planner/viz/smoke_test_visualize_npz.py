#!/usr/bin/env python3
"""Smoke test for scripts/visualize_npz.py.

This intentionally exercises the *script* entrypoint to ensure imports and CLI
behavior remain intact.

It will:
  - locate one pre-generated single_npz under validation_output2/
  - run scripts/visualize_npz.py <npz> <out_png>
  - assert the PNG exists

Additionally, it will run once with debug overlays enabled via env vars to
ensure the optional visualization paths do not crash.

Output goes to outputs/viz/smoke/ by default.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[4]

    candidates = sorted(repo_root.glob("validation_output2/boston50w_viz_50_*/slice*/single_npz/*.npz"))
    if not candidates:
        # Fallback for dev machines where validation_output2 is not present.
        candidates = sorted(repo_root.glob("data/test_preprocess_output/*.npz"))
    if not candidates:
        raise SystemExit(
            "No candidate NPZ files found under validation_output2/... or data/test_preprocess_output/*.npz"
        )

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

    # Run once with overlay env vars enabled.
    out_png2 = out_dir / f"{npz_path.stem}.overlay.png"
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    env.update(
        {
            "NPZ_VIZ_SHOW_LANE_DIR": "1",
            "NPZ_VIZ_SHOW_TRAFFIC_LIGHTS": "1",
            "NPZ_VIZ_SHOW_NEIGHBOR_HEADING": "1",
            "NPZ_VIZ_SHOW_ACC": "1",
        }
    )
    cmd2 = [sys.executable, str(repo_root / "scripts/visualize_npz.py"), str(npz_path), str(out_png2)]
    print("Running (overlays):", " ".join(cmd2))
    subprocess.check_call(cmd2, cwd=repo_root, env=env)
    if not out_png2.exists():
        raise SystemExit(f"Expected overlay output PNG missing: {out_png2}")
    print("OK - wrote", out_png2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
