"""Export manifest helpers.

These utilities are shared across export entrypoints. Keep them lightweight and
free of NuPlan-specific imports so scripts can import them inside containers.
"""

from __future__ import annotations

import json
from typing import Any, TextIO


def write_jsonl_line(fp: TextIO, obj: dict[str, Any]) -> None:
    """Write one JSON object as a single line (UTF-8).

    Note: ensure_ascii=False matches exporter behavior for readability.
    """

    fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
