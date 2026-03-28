#!/usr/bin/env python3
"""Wrapper for backwards-compatible entrypoint.

Historically users ran: python3 scripts/run_nuboard.py ...
Now the implementation is a shell script under scripts/nuboard/run_nuboard.sh.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


def main() -> None:
    script = Path(__file__).resolve().parent / "nuboard" / "run_nuboard.sh"
    raise SystemExit(subprocess.call([str(script), *os.sys.argv[1:]]))


if __name__ == "__main__":
    main()
