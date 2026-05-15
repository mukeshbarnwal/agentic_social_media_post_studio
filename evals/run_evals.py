"""Run small offline eval suite; writes evals/latest_results.json and prints a score table."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Force deterministic evals regardless of host shell env
os.environ["MOCK_MODELS"] = "true"

from graph.state import StudioState
from graph.workflow import run_studio


def _score_case(case: dict, state: dict) -> dict:
    ex = case.get("expects", {})
    post = (state.get("post") or {}) if isinstance(state, dict) else {}
    slides = state.get("slides") or []
    markers = post.get("source_markers") or []
    hashtags = post.get("hashtags") or []
    scores: dict[str, float] = {}
    notes: list[str] = []

    if "min_hashtags" in ex:
        ok = len(hashtags) >= int(ex["min_hashtags"])
        scores["hashtags_min"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("hashtag count too low")
    if "max_hashtags" in ex:
        ok = len(hashtags) <= int(ex["max_hashtags"])
        scores["hashtags_max"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("too many hashtags")

    if ex.get("requires_source_markers"):
        ok = len(markers) > 0
        scores["grounding_markers"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("missing source_markers")

    if "min_slides" in ex:
        ok = len(slides) >= int(ex["min_slides"])
        scores["slides"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("not enough slides")

    if ex.get("critic_pass"):
        rep = state.get("critic_report") or {}
        ok = bool(rep.get("pass"))
        scores["critic"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("critic did not pass")

    if ex.get("tone"):
        ok = bool(post.get("hook")) and bool(post.get("body"))
        scores["tone_stub"] = 1.0 if ok else 0.0
        if not ok:
            notes.append("missing post content for tone case")

    overall = sum(scores.values()) / max(len(scores), 1)
    return {"scores": scores, "overall": overall, "notes": notes}


def main() -> None:
    root = Path(__file__).resolve().parent
    cases_path = root / "eval_cases.json"
    out_path = root / "latest_results.json"
    cases = json.loads(cases_path.read_text())
    rows = []
    for case in cases:
        init: StudioState = {
            "topic": case.get("topic", "Eval topic"),
            "tone": case.get("tone", "casual"),
            "target_length": "medium",
            "num_slides": int(case.get("num_slides", 3)),
            "brand_color": "#457b9d",
            "pdf_ids": [],
            "image_paths": [],
            "url_or_query": case.get("url_or_query", ""),
            "rerun_scope": "full",
            "critic_iterations": 0,
        }
        out, trace = run_studio(init)
        r = _score_case(case, dict(out))
        rows.append({"id": case["id"], **r, "trace_path": str(trace.path) if trace.path else None})

    summary = {
        "mean_overall": sum(x["overall"] for x in rows) / max(len(rows), 1),
        "rows": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"{'case':30} {'overall':>7} {'notes'}")
    for x in rows:
        print(f"{x['id']:30} {x['overall']:7.2f} {', '.join(x.get('notes', []))}")
    print("\nMean overall:", round(summary["mean_overall"], 3))
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
