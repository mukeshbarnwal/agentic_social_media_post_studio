"""Compile LangGraph workflow: planner → research → copywriter → visual → critic (loop) → assemble."""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from langgraph.graph import END, StateGraph

from graph.nodes import (
    assemble_manifest,
    copywriter_node,
    critic_node,
    planner_node,
    research_node,
    visual_node,
)
from graph.state import StudioState
from observability.interaction_log import log_interaction
from observability.langsmith_setup import build_invoke_config, configure_langsmith
from observability.trace_logger import RunTrace

try:
    from langsmith import traceable
except ImportError:  # pragma: no cover
    def traceable(*_args, **_kwargs):  # type: ignore[misc]
        def _decorator(fn):
            return fn

        return _decorator


def _route_after_critic(state: StudioState, trace: RunTrace | None = None) -> Literal["assemble", "copywriter", "visual", "research"]:
    rep = state.get("critic_report") or {}
    it = int(state.get("critic_iterations") or 0)
    max_reached = it >= 3
    if rep.get("pass") or max_reached:
        if trace:
            trace.log(
                "critic_routing",
                destination="assemble",
                critic_iteration=it,
                max_retries_reached=max_reached,
                critic_passed=bool(rep.get("pass")),
                issues=rep.get("issues", []),
            )
        return "assemble" # if maximum retry is reached, we return the assmble which assembles the previous agents outputs
    route = rep.get("route") or "copywriter" 
    if route not in ("copywriter", "visual", "research"):
        route = "copywriter"
    if trace:
        trace.log(
            "critic_routing",
            destination=route,
            critic_iteration=it,
            max_retries_reached=False,
            critic_passed=False,
            issues=rep.get("issues", []),
            scores=rep.get("scores", {}),
        )
    return route  # type: ignore[return-value]


_AGENT_LABELS = {
    "planner":    ("📋", "Planner",    "Breaking down the topic into a content plan…"),
    "research":   ("🔍", "Research",   "Fetching web results and querying RAG…"),
    "copywriter": ("✍️", "Copywriter", "Drafting the LinkedIn post…"),
    "visual":     ("🖼️", "Visual",     "Generating carousel slides…"),
    "critic":     ("🧐", "Critic",     "Evaluating copy and visuals…"),
    "assemble":   ("📦", "Assemble",   "Assembling the final manifest…"),
}


def build_workflow(trace: RunTrace, status_callback=None):
    g = StateGraph(StudioState)

    def wrap(name: str, fn):
        """LangGraph node with an explicit LangSmith span name (planner, research, …)."""

        @traceable(name=name, run_type="chain", tags=["agent", name])
        def _inner(state: StudioState) -> dict[str, Any]:
            icon, label, detail = _AGENT_LABELS.get(name, ("⚙️", name.title(), ""))
            print(f"[AGENT]  ▶ {name.upper()} starting …", flush=True)
            trace.log("agent_start", agent=name)
            if status_callback:
                status_callback(f"{icon} **{label}** — {detail}")
            result = fn(state, trace)
            print(f"[AGENT]  ✓ {name.upper()} done", flush=True)
            return result

        _inner.__name__ = name
        return _inner

    @traceable(name="critic_router", run_type="chain", tags=["agent", "critic"])
    def critic_router(state: StudioState) -> str:
        """Critic conditional edge — replaces anonymous RunnableCallable in traces."""
        return _route_after_critic(state, trace)  # type: ignore[return-value]

    g.add_node("planner", wrap("planner", planner_node))
    g.add_node("research", wrap("research", research_node))
    g.add_node("copywriter", wrap("copywriter", copywriter_node))
    g.add_node("visual", wrap("visual", visual_node))
    g.add_node("critic", wrap("critic", critic_node))
    g.add_node("assemble", wrap("assemble", assemble_manifest))

    g.set_entry_point("planner")
    g.add_edge("planner", "research")
    g.add_edge("research", "copywriter")
    g.add_edge("copywriter", "visual")
    g.add_edge("visual", "critic")
    g.add_conditional_edges(
        "critic",
        critic_router,
        {
            "assemble": "assemble",
            "copywriter": "copywriter",
            "visual": "visual",
            "research": "research",
        },
    )
    g.add_edge("assemble", END)
    try:
        return g.compile(name="agentic_social_post_studio")
    except TypeError:
        return g.compile()


def run_studio(
    initial: StudioState,
    status_callback=None,
    *,
    log_source: str = "app",
    case_id: str | None = None,
) -> tuple[StudioState, RunTrace]:
    configure_langsmith()
    trace = RunTrace()
    trace.log("run_start", topic=initial.get("topic"), rerun_scope=initial.get("rerun_scope", "full"))
    data = dict(initial)
    for drop in ("trace", "manifest", "slides", "sources", "token_usage"):
        data.pop(drop, None)
    app = build_workflow(trace, status_callback=status_callback)
    data.setdefault("critic_iterations", 0)
    data.setdefault("trace", [])
    data.setdefault("token_usage", {})
    mock = os.getenv("MOCK_MODELS", "").lower() in ("1", "true", "yes")
    invoke_config = build_invoke_config(
        run_name="agentic_social_post_studio",
        metadata={
            "topic": initial.get("topic"),
            "rerun_scope": initial.get("rerun_scope", "full"),
            "num_slides": initial.get("num_slides"),
            "pdf_ids": initial.get("pdf_ids") or [],
            "mock_models": mock,
            "jsonl_run_id": trace.run_id,
            "jsonl_trace": str(trace.path) if trace.path else None,
        },
        tags=["studio", "mock" if mock else "live"],
    )
    t0 = time.perf_counter()
    err: str | None = None
    out: StudioState = {}
    try:
        out = app.invoke(data, config=invoke_config) if invoke_config else app.invoke(data)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        duration_ms = (time.perf_counter() - t0) * 1000
        log_interaction(
            trace.run_id,
            dict(initial),
            dict(out) if out else {},
            trace_path=str(trace.path) if trace.path else None,
            source=log_source,
            case_id=case_id,
            error=err,
            duration_ms=duration_ms,
        )
    return out, trace
