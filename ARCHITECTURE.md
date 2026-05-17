# Architecture

## Sequence (happy path)

```mermaid
sequenceDiagram
  participant U as User (Streamlit)
  participant G as LangGraph
  participant P as Planner
  participant R as Research
  participant C as Copywriter
  participant V as Visual
  participant Q as Critic
  participant M as MCP tools (runtime)
  participant DB as Chroma

  U->>G: topic + files + prefs
  G->>P: plan request
  P->>M: list_sources()
  M->>DB: peek metadata
  P-->>G: slide plan JSON

  G->>R: research
  R->>M: pdf_query / fetch_url / web_search
  M->>DB: retrieve / upsert web chunks
  R-->>G: research_chunks[]

  G->>C: draft post (skills loaded)
  C-->>G: post JSON + source_markers

  G->>V: slides + assets
  V-->>G: slides[] + rendered PNG paths

  G->>Q: rubric scoring
  Q-->>G: pass/fail + route + critic_iteration

  alt critic pass OR iteration >= 3
    G->>U: manifest + trace (critic_routing → assemble)
  else critic fail → copywriter
    G->>C: targeted rewrite
    Note over G,Q: loops back through copywriter → visual → critic
  else critic fail → visual
    G->>V: visual rewrite
    Note over G,Q: loops back through visual → critic
  else critic fail → research
    G->>R: re-research
    Note over G,Q: loops back through research → copywriter → visual → critic
  end
```

## Agent roster & responsibilities

| Agent | Responsibility | Tools / skills |
|------|----------------|----------------|
| Planner | Slide plan, whether web grounding is needed | `list_sources`, `brand_voice` excerpt |
| Research | Grounding pack | `pdf_query`, `fetch_url`, `web_search`, caption helper for uploads |
| Copywriter | Hook/body/hashtags/CTA + `per_slide_captions` + `per_slide_bullets` (grounded, no placeholders) | `linkedin_formatting`, `citation`, `brand_voice` |
| Visual | Treatment decision (uploaded > PDF figure > mock), alt text, real asset path or mock render | `image_prompting` |
| Critic | Rubric + routing | `critic_rubric` |

## Skills discovery / loading

1. Skills live in `skills/<folder>/SKILL.md`.
2. `skill_loader.load_skill("<folder>")` parses optional YAML frontmatter (`name`, `description`) and markdown body.
3. Each graph node pulls **only** the skills it needs (see `graph/nodes.py`).

## Critic loop & observability

The critic runs after every visual pass. Its routing logic in `graph/workflow.py` emits two JSONL trace events per iteration:

| Event | Key fields |
|---|---|
| `agent_end` (critic) | `critic_iteration`, `max_retries_reached`, `passed`, `route`, `scores`, `issues` |
| `critic_routing` | `destination`, `critic_iteration`, `max_retries_reached`, `critic_passed`, `issues` |

The loop is hard-capped at **3 iterations** (`critic_iterations >= 3` → unconditional assemble). When `passed=true`, `issues` is always empty — the Python pre-check overrides any contradictory LLM output so the trace is never misleading.

## Grounding contract

Research emits chunks with stable `chunk_id` values. The copywriter returns `source_markers` referencing those ids. The UI sources panel is derived from `research_chunks` metadata (`modality`, `page`, `path`).
