"""JSONL run traces: agent steps, tool calls, retrieval ids, token usage."""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag.store import project_root


def runs_dir() -> Path:
    d = project_root() / "storage" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RunTrace:
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    path: Path | None = None
    lines: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = runs_dir() / f"{self.run_id}.jsonl"

    def log(self, event: str, **payload: Any) -> dict[str, Any]:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **payload}
        self.lines.append(row)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        return row

    def as_list(self) -> list[dict[str, Any]]:
        return list(self.lines)
