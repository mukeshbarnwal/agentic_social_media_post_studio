"""Compile LangGraph workflow: planner → research → copywriter → visual → critic (loop) → assemble."""

from __future__ import annotations

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
from observability.trace_logger import RunTrace


def _route_after_critic(state: StudioState) -> Literal["assemble", "copywriter", "visual", "research"]:
    rep = state.get("critic_report") or {}
    it = int(state.get("critic_iterations") or 0)
    if rep.get("pass") or it >= 3:
        return "assemble"
    route = rep.get("route") or "copywriter"
    if route in ("copywriter", "visual", "research"):
        return route  # type: ignore[return-value]
    return "copywriter"


def build_workflow(trace: RunTrace):
    g = StateGraph(StudioState)

    def wrap(name, fn):
        def _inner(state: StudioState) -> dict[str, Any]:
            trace.log("agent_start", agent=name)
            return fn(state, trace)

        return _inner

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
        _route_after_critic,
        {
            "assemble": "assemble",
            "copywriter": "copywriter",
            "visual": "visual",
            "research": "research",
        },
    )
    g.add_edge("assemble", END)
    return g.compile()


def run_studio(initial: StudioState) -> tuple[StudioState, RunTrace]:
    trace = RunTrace()
    trace.log("run_start", topic=initial.get("topic"), rerun_scope=initial.get("rerun_scope", "full"))
    data = dict(initial)
    for drop in ("trace", "manifest", "slides", "sources", "token_usage"):
        data.pop(drop, None)
    app = build_workflow(trace)
    data.setdefault("critic_iterations", 0)
    data.setdefault("trace", [])
    data.setdefault("token_usage", {})
    out = app.invoke(data)
    return out, trace
