# Agentic Social Media Post Studio

Proof-of-concept for a **multi-agent LinkedIn post studio** with a **custom MCP server**, **on-demand `SKILL.md` skills**, **Chroma multimodal RAG**, **LangGraph handoffs**, **JSONL tracing**, and a **Streamlit** UI. Heavy models can be disabled with **`MOCK_MODELS=true`** while still exercising the full control flow.

## Quickstart (local)

```bash
cd agentic-social-post-studio
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Optional: add OPENAI_API_KEY and/or TAVILY_API_KEY
export MOCK_MODELS=true
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`.

## Quickstart (Docker)

```bash
cp .env.example .env
docker compose up --build
```

- **Streamlit UI:** `http://localhost:8501`
- **MCP (streamable HTTP):** `http://localhost:8765/mcp`

> Chroma is embedded (persistent volume under `./storage/chroma`). This satisfies the "real vector store" requirement without a separate vector container.

> **Hot-reload:** Source code is volume-mounted (`.:/app`) so file saves on the host are instantly live. Streamlit auto-reloads `app/streamlit_app.py`. For any other module (`graph/`, `rag/`, `mcp_server/`), run `docker compose restart app` to flush Python's module cache. Only run `docker compose up --build` when `requirements.txt` or a `Dockerfile` changes.

| File changed | Action |
|---|---|
| `app/streamlit_app.py` | Nothing — Streamlit auto-reloads |
| `graph/*.py`, `rag/*.py`, `mcp_server/tool_runtime.py` | `docker compose restart app` |
| `mcp_server/server.py` | `docker compose restart mcp` |
| `requirements.txt` or `Dockerfile` | `docker compose up --build` |

## MOCK_MODELS mode

Set `MOCK_MODELS=true` in `.env` (or `export MOCK_MODELS=true` locally) to swap in deterministic stubs for all LLM calls, embeddings, and web search. The full agent pipeline still executes end-to-end. Every mock output is labelled **MOCK** in logs and on slide images so nothing is silently faked.

## Required user flows

| Flow | Inputs | What the system does |
|---|---|---|
| **1 — PDF carousel** | PDF + topic + N slides | Indexes PDF into Chroma, planner generates document-aware queries, research retrieves grounded chunks, copywriter writes slide-by-slide bullets from actual content, visual renders slides |
| **2 — URL / web search** | Topic + URL or search query | `fetch_url` strips and chunks the page into Chroma (URL) or `web_search` returns snippets (query); copywriter grounds post in web content |
| **3 — Single image post** | Image + topic | Research captions the image; copywriter writes post around the caption; visual uses the real uploaded image; `num_slides` auto-set to 1 |
| **4 — Partial rerun** | Edited hook/body | Keeps prior plan + research; re-runs copywriter → visual → critic only |

## Latest automated evals

Run:

```bash
PYTHONPATH=. python evals/run_evals.py
```

Results are written to `evals/latest_results.json`.

**Last run — MOCK mode, 10 cases (flows 1–4 + format/grounding/tone checks):**

| Case | Metrics checked | Overall |
|---|---|---|
| `flow_pdf_carousel` | slides ≥ 4, hashtags ≥ 1, source_markers | 1.00 |
| `flow_web_only` | hashtags ≥ 1 | 1.00 |
| `flow_image_post` | slides = 1 (single-image), hashtags ≥ 1, source_markers | 1.00 |
| `flow_partial_rerun` | source_markers, critic pass | 1.00 |
| `flow_topic_only` | hashtags 1–8 | 1.00 |
| `format_hashtags` | hashtags 3–6 | 1.00 |
| `grounding_markers` | source_markers present | 1.00 |
| `tone_formal` | hook + body present | 1.00 |
| `carousel_length` | slides ≥ 5 | 1.00 |
| `critic_pass_stub` | critic pass | 1.00 |
| **Mean** | | **1.00** |

> All scores are 1.00 in MOCK mode — deterministic stubs always satisfy structural checks. The eval harness's value is catching regressions (a score drop flags a broken agent step or schema change), not measuring LLM quality in isolation.

## Architecture (short)

- **Shared state:** `graph/state.py` (`StudioState`) — TypedDict blackboard; `trace` and `token_usage` use reducers.
- **Graph:** `graph/workflow.py` — `planner → research → copywriter → visual → critic` with critic loops routing back to `copywriter`, `visual`, or `research` until pass or iteration cap (max 3). Each routing decision is logged as a `critic_routing` trace event with destination, iteration count, and whether the cap was hit.
- **MCP tools** (implemented in `mcp_server/tool_runtime.py`, exposed via FastMCP in `mcp_server/server.py`):
  - `web_search(query, max_results=5)` → ranked `{title,url,snippet}` (MOCK unless `TAVILY_API_KEY`, with DuckDuckGo lite fallback)
  - `fetch_url(url)` → `{markdown, image_urls, web_source_id}` — chunks and embeds into Chroma
  - `pdf_query(pdf_id, question, k)` → top-k chunks `{chunk_id,text,metadata}`
  - `index_pdf(file_path)` → `{pdf_id}` (must be under `storage/uploads`)
  - `list_sources()` → chunk listing + distinct `pdf_id`s
- **Skills:** folders under `skills/<name>/SKILL.md` with YAML frontmatter; loaded **per step** via `skill_loader.py`.
- **RAG:** PyMuPDF text + table blocks + embedded figures (saved under `storage/extracted_images`); Chroma persistence; BM25 re-rank in `rag/retriever.py`.
- **Visuals:** decision order — **uploaded image > PDF figure > MOCK slide** (Pillow renders real content onto a branded canvas). When a real asset exists the PNG path is served directly; mock slide is only the final fallback.
- **Observability:** `observability/trace_logger.py` writes JSONL under `storage/runs/<run_id>.jsonl`. Terminal prints structured `[AGENT] IN/OUT` logs per agent step. The UI exposes the latest trace download. Critic runs emit two events per iteration: `agent_end` (with `critic_iteration`, `max_retries_reached`, `passed`, `scores`, `issues`) and `critic_routing` (with `destination` and reason), making the bounded retry loop fully visible in the trace.
- **UI progress:** Each agent step fires a `status_callback` updating an inline caption in real-time — no page refresh or clicking required.
- **Grounded copywriter:** System prompt explicitly forbids generic text; every claim must come from retrieved chunks. Outputs `per_slide_bullets` (real facts per slide) alongside `per_slide_captions`.
- **Dynamic PDF queries:** Planner generates 3-5 document-aware queries (adapts to resume / paper / spec / any doc type). Research deduplicates across queries.
- **Auto single-slide:** Image-only input (no PDF, no URL) auto-sets `num_slides=1` for a single-image post per the spec.
- **Action-style topics:** If the topic field is an instruction rather than a subject (e.g. `"write linkedin summary"`, `"summarize this"`), the Planner and Copywriter treat the uploaded image or retrieved content as the post subject — the topic is the user's intent verb, not the post theme.

See `ARCHITECTURE.md` for a sequence diagram and `PRODUCTION.md` for ops notes.

## Skills

| Skill folder | Loaded by | Purpose |
|---|---|---|
| `skills/brand_voice/` | Planner, Copywriter | Tone rules, do/don't examples for chosen voice |
| `skills/linkedin_formatting/` | Copywriter | Hook patterns, line-break rhythm, hashtag rules, character limits |
| `skills/citation/` | Copywriter | How to attach `source_id`s to claims and build the sources panel |
| `skills/image_prompting/` | Visual | Turn slide intent into a concrete image prompt; write alt text for real assets |
| `skills/critic_rubric/` | Critic | Scoring criteria: factuality vs sources, brand voice, length, accessibility |

Skills are discovered automatically — each agent calls `skill_loader.skill_prompt("<folder>")` for only the skills it needs. No skill is included in every prompt.

## MCP smoke test (no UI)

```bash
PYTHONPATH=. python scripts/smoke_tools.py
```

### Full curl smoke test against the live Docker stack

Make sure `docker compose up --build` is running, then:

**Step 1 — Initialize session**

```bash
SESSION=$(curl -s -i -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"curl-test","version":"1.0"}}}' \
  | grep -i "mcp-session-id" | awk '{print $2}' | tr -d '\r')
echo "Session: $SESSION"
```

**Step 2 — List all tools**

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'
```

Returns all 5 tools: `web_search`, `fetch_url`, `pdf_query`, `index_pdf`, `list_sources`.

**Tool 1 — `list_sources`**

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"list_sources","arguments":{}}}'
```

**Tool 2 — `web_search`**

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"web_search","arguments":{"query":"LangGraph multi-agent patterns","max_results":3}}}'
```

**Tool 3 — `fetch_url`**

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"fetch_url","arguments":{"url":"https://example.com"}}}'
```

**Tool 4 — `index_pdf`** *(copy a PDF into `storage/uploads/` first)*

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"index_pdf","arguments":{"file_path":"storage/uploads/sample.pdf"}}}'
```

**Tool 5 — `pdf_query`** *(replace `pdf_id` with the value returned by `index_pdf`)*

```bash
curl -s -X POST http://localhost:8765/mcp \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -H "mcp-session-id: $SESSION" \
  -d '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"pdf_query","arguments":{"pdf_id":"pdf_abc123","question":"What are the key findings?","k":5}}}'
```

> **Note:** `curl http://localhost:8765/` returns `404` and `curl http://localhost:8765/mcp` without correct headers returns `406` — both expected. The MCP protocol requires `POST` to `/mcp` with `Accept: application/json, text/event-stream`.

## Connecting the MCP server to a client

```json
{
  "mcpServers": {
    "social-post-studio": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Point Claude Desktop, Cursor, or MCP Inspector at the URL above once `docker compose up` is running.

## Trade-off (what we optimised for)

1. The system prioritizes agent orchestration and transparency over maximum model complexity. Focus was placed on multi-agent coordination, MCP-based tooling, shared state, critic rerouting, and partial reruns instead of heavier multimodal models.
This made the workflow easier to debug, evaluate, and reason about while still supporting multimodal grounding.
2. Given the 4-day scope, the project prioritized clear multi-agent orchestration and observable workflow design over maximum model sophistication.
The system focused on explicit agent boundaries, shared state, critic-driven rerouting, multimodal grounding, and partial reruns.
For reproducibility and local execution, heavy vision/image models were replaced with deterministic mocks using MOCK_MODELS=true.

## What we would extend with two more days

- Once a user posts content on LinkedIn, displaying what all questions or comments might he or she expect from users on LinkedIn on that post.
- Write linkedin posts according to the user style of writing
- We can also give an estimate or prediction of how many likes or comments this Ai generated post might receive on LinkedIn. Based on past historical data- data would have a LinkedIn post and corresponding likes/comments.
- Implement observability in Langsmith for better tracking and testing multiple examples at the same time


## References

Case study: `problem_statement1.md`.
