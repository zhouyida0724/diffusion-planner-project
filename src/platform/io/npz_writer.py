"""NPZ writing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def save_npz_compressed(path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write arrays to a compressed NPZ at path."""

    np.savez_compressed(Path(path), **dict(arrays))
