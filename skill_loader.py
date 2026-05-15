"""Load a single skill's SKILL.md (YAML frontmatter + body) on demand."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parent


def load_skill(skill_folder: str) -> dict[str, Any]:
    """skill_folder is e.g. 'brand_voice' under skills/."""
    path = project_root() / "skills" / skill_folder / "SKILL.md"
    raw = path.read_text(encoding="utf-8")
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            body = parts[2].strip()
            return {"meta": meta, "body": body, "path": str(path)}
    return {"meta": {}, "body": raw.strip(), "path": str(path)}


def skill_prompt(skill_folder: str) -> str:
    s = load_skill(skill_folder)
    name = s["meta"].get("name", skill_folder)
    desc = s["meta"].get("description", "")
    return f"## Skill: {name}\n{desc}\n\n{s['body']}"
