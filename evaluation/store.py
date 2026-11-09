"""Mirror result files to a second location (e.g. Google Drive on Colab) as soon as they are written.

Set RESULTS_MIRROR=/content/drive/MyDrive/llm-scratch/results; unset = no-op.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def mirror(*paths: str | Path) -> None:
    dest = os.getenv("RESULTS_MIRROR")
    if not dest:
        return
    Path(dest).mkdir(parents=True, exist_ok=True)
    for p in map(Path, paths):
        if p.exists():
            shutil.copy(p, Path(dest) / p.name)
