# Agentic Social Media Post Studio

Proof-of-concept for a **multi-agent LinkedIn post studio** with a **custom MCP server**, **on-demand `SKILL.md` skills**, **Chroma multimodal RAG**, **LangGraph handoffs**, **JSONL + interaction logging**, optional **LangSmith** tracing, and a **Streamlit** UI. Heavy models can be disabled with **`MOCK_MODELS=true`** while still exercising the full control flow.

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

## Interaction logs (query + output)

Every studio run (Streamlit, evals, scripts) appends one row to **`storage/interactions/interactions.jsonl`** and writes **`storage/interactions/<run_id>.json`** with:

- **Input:** topic, tone, slides, `pdf_ids`, images, URL/query, rerun scope, edits  
- **Output:** hook, body, hashtags, slides summary, critic pass/route/issues, chunk ids, token usage  
- **Links:** `trace_path` to the agent JSONL under `storage/runs/`

Summarize 100+ runs without opening LangSmith:

```bash
PYTHONPATH=. python scripts/analyze_interactions.py
```

Disable with `INTERACTION_LOG_DISABLED=true` in `.env`.

## LangSmith observability

1. Add to `.env` (copy from `.env.example`):

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_pt_...   # your key
LANGSMITH_PROJECT=agentic-social-post-studio
```

2. Restart the app (`docker compose restart app` or restart Streamlit locally).

3. Generate a post in the UI. Open [smith.langchain.com](https://smith.langchain.com) → project **agentic-social-post-studio** → **Tracing** → **Runs**.

4. **Open the tree, not the flat list:** click the run named **`agentic_social_post_studio`** (root). In the run detail page, use the **Waterfall** / **Trace** tab — you should see nested spans: `planner` → `research` → `copywriter` → `visual` → `critic` → `critic_router` → … → `assemble`, each with `ChatOpenAI` children where applicable.

The **Runs table** often lists every `ChatOpenAI` call as its own row; those are child spans. Filter or sort by **start time** and open the parent run whose name is **`agentic_social_post_studio`**, not the individual LLM rows.

Each root run has tags `studio` and `mock` or `live`. Agent steps are explicit `@traceable` spans (`planner`, `research`, …). Local JSONL traces under `storage/runs/` are linked via metadata `jsonl_trace` / `jsonl_run_id`.

> With `MOCK_MODELS=true`, the graph still traces in LangSmith but LLM spans are skipped (no real `ChatOpenAI` calls).

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
| **4 — Partial rerun** | Hook/body edit + **Rerun from edited copy only** | Skips planner + research; copywriter applies edits → visual → critic (see below) |

### Partial rerun (flow 4)

1. Run **Generate post** once (URL, PDF, or topic).
2. Enter text in **Hook edit** and/or **Body edit** (e.g. `make it punchier`, or paste replacement copy).
3. Click **Rerun from edited copy only** — not **Generate post** again.

Keeps `plan` and `research_chunks`; only `user_edited_hook` / `user_edited_body` you typed are sent (empty fields are not filled with old hook/body).

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

```text
Planner → Research → Copywriter → Visual → Critic ⟲ (max 3) → Assemble → UI
```

| Component | Summary |
|-----------|---------|
| **State** | `graph/state.py` — `StudioState` blackboard (`plan`, `research_chunks`, `post`, `slides`, …) |
| **Graph** | `graph/workflow.py` — critic routes to `copywriter` / `visual` / `research` or `assemble` |
| **MCP** | `web_search`, `fetch_url`, `pdf_query`, `index_pdf`, `list_sources` — see `mcp_server/tool_runtime.py` |
| **RAG** | PyMuPDF ingest → Chroma; BM25 re-rank in `rag/retriever.py` |
| **Visuals** | uploaded image > PDF figure > mock Pillow slide |
| **Grounding** | `source_markers` must reference retrieved `chunk_id`s (critic pre-check) |

**Observability**

| Layer | Where |
|-------|--------|
| Agent steps + tool calls | `storage/runs/<run_id>.jsonl` — download in UI |
| Query + output (batch eval) | `storage/interactions/interactions.jsonl` — `scripts/analyze_interactions.py` |
| LLM / graph debug | LangSmith — see [LangSmith observability](#langsmith-observability) |

See **`ARCHITECTURE.md`** for diagrams (full + partial rerun), agent roster, and grounding contract. See **`PRODUCTION.md`** for ops notes.

## Environment variables (common)

| Variable | Purpose |
|----------|---------|
| `MOCK_MODELS` | `true` = deterministic LLM/search stubs |
| `OPENAI_API_KEY` | Live chat + embeddings + vision caption |
| `TAVILY_API_KEY` | Live web search (optional) |
| `LANGSMITH_TRACING` / `LANGSMITH_API_KEY` / `LANGSMITH_PROJECT` | LangSmith traces |
| `INTERACTION_LOG_DISABLED` | Set `true` to skip interaction JSONL |
| `CHROMA_PERSIST_DIR` | Vector store path (default `storage/chroma`) |

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

- Predict likely LinkedIn comments/questions on a generated post.
- Personal tone model trained on the user's past posts.
- Engagement estimate (likes/comments) from historical post data.
- Gold-set eval with LLM-as-judge + `@traceable` MCP spans in LangSmith.
- Vision captions for PDF figures at ingest so carousel can use real PDF images reliably.


## References

Case study: `problem_statement1.md`.
