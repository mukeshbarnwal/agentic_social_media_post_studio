#!/usr/bin/env python3
"""Summarize storage/interactions/interactions.jsonl for batch QA."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from observability.interaction_log import load_interactions


def main() -> None:
    rows = load_interactions()
    if not rows:
        print("No interactions logged yet. Run the app or evals first.")
        print("Expected file: storage/interactions/interactions.jsonl")
        return

    n = len(rows)
    errors = [r for r in rows if r.get("error")]
    ok = [r for r in rows if not r.get("error")]

    critic_pass = sum(1 for r in ok if (r.get("output") or {}).get("critic_pass"))
    with_markers = sum(
        1
        for r in ok
        if (r.get("output") or {}).get("source_markers")
    )
    latencies = [r["duration_ms"] for r in rows if r.get("duration_ms") is not None]
    issues: Counter[str] = Counter()
    for r in ok:
        for issue in (r.get("output") or {}).get("critic_issues") or []:
            if isinstance(issue, str):
                issues[issue[:80]] += 1

    routes: Counter[str] = Counter(
        (r.get("output") or {}).get("critic_route") or "unknown" for r in ok
    )
    sources: Counter[str] = Counter(r.get("source", "unknown") for r in rows)
    rerun: Counter[str] = Counter(
        (r.get("input") or {}).get("rerun_scope", "full") for r in rows
    )

    print(f"Total runs:     {n}")
    print(f"Errors:         {len(errors)} ({100 * len(errors) / n:.1f}%)")
    print(f"Critic pass:    {critic_pass}/{len(ok)} ({100 * critic_pass / max(len(ok), 1):.1f}%)")
    print(f"Has markers:    {with_markers}/{len(ok)} ({100 * with_markers / max(len(ok), 1):.1f}%)")
    if latencies:
        latencies.sort()
        print(f"Latency ms:     min={latencies[0]:.0f} p50={latencies[len(latencies)//2]:.0f} max={latencies[-1]:.0f}")
    print(f"By source:      {dict(sources)}")
    print(f"By rerun_scope: {dict(rerun)}")
    print(f"Critic routes:  {dict(routes)}")
    if issues:
        print("Top critic issues:")
        for text, count in issues.most_common(8):
            print(f"  {count:4d}  {text}")

    failed = [r for r in ok if not (r.get("output") or {}).get("critic_pass")]
    if failed:
        print("\nRecent critic failures (run_id):")
        for r in failed[-5:]:
            inp = r.get("input") or {}
            print(f"  {r.get('run_id')}  topic={str(inp.get('topic', ''))[:50]!r}")


if __name__ == "__main__":
    main()
