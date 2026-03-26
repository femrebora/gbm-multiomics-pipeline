"""
checkpoint.py — Simple JSON checkpoint for resumable download pipelines.

Stores completed step names and their result payloads in a JSON file.
Re-running skips already-completed steps automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gbm_multiomics.constants import CHECKPOINT_FILE


class Checkpoint:
    """Read/write a JSON checkpoint file in the output directory."""

    def __init__(self, output_dir: Path):
        self._path = output_dir / CHECKPOINT_FILE
        self._data: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def _flush(self) -> None:
        self._path.write_text(
            json.dumps(self._data, indent=2, default=str), encoding="utf-8"
        )

    def is_done(self, step: str) -> bool:
        return step in self._data

    def save(self, step: str, payload: Any) -> None:
        self._data[step] = payload
        self._flush()

    def get(self, step: str) -> Any:
        return self._data.get(step, {})

    def reset_from(self, step: str) -> None:
        """Clear *step* and all subsequent steps (simple: clear everything from step onward)."""
        keys = list(self._data.keys())
        found = False
        for k in keys:
            if k == step:
                found = True
            if found:
                del self._data[k]
        self._flush()

    def reset_all(self) -> None:
        self._data = {}
        self._flush()
