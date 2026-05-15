# Agentic Social Post Studio

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

> Chroma is embedded (persistent volume under `./storage/chroma`). This satisfies the “real vector store” requirement without a separate vector container.

## Latest automated evals

Run:

```bash
PYTHONPATH=. python evals/run_evals.py
```

Results are written to `evals/latest_results.json`. Last run on this branch: **mean overall score 1.00** across 8 cases (MOCK mode).

## Architecture (short)

- **Shared state:** `graph/state.py` (`StudioState`) — TypedDict blackboard; `trace` and `token_usage` use reducers.
- **Graph:** `graph/workflow.py` — `planner → research → copywriter → visual → critic` with critic loops routing back to `copywriter`, `visual`, or `research` until pass or iteration cap.
- **MCP tools (implemented in `mcp_server/tool_runtime.py`, exposed via FastMCP in `mcp_server/server.py`):**
  - `web_search(query, max_results=5)` → ranked `{title,url,snippet}` (MOCK unless `TAVILY_API_KEY`, with DuckDuckGo lite fallback)
  - `fetch_url(url)` → `{markdown, image_urls, web_source_id}` and indexes content into Chroma
  - `pdf_query(pdf_id, question, k)` → top‑k chunks `{chunk_id,text,metadata}`
  - `index_pdf(file_path)` → `{pdf_id}` (must be under `storage/uploads`)
  - `list_sources()` → chunk listing + distinct `pdf_id`s
- **Skills:** folders under `skills/<name>/SKILL.md` with YAML frontmatter; loaded **per step** via `skill_loader.py` (not all skills in every system prompt).
- **RAG:** PyMuPDF text + table-ish blocks + embedded figures (saved under `storage/extracted_images`); Chroma persistence; optional BM25 re-rank in `rag/retriever.py`.
- **Visuals:** decision order documented inline in `graph/nodes.py:visual_node` — **uploaded image > PDF figure > MOCK slide** (Pillow renders prompt text onto a colored canvas).
- **Observability:** `observability/trace_logger.py` writes JSONL under `storage/runs/<run_id>.jsonl` (agent/tool/chunk summaries). The UI exposes the latest trace download.

See `ARCHITECTURE.md` for a sequence diagram and `PRODUCTION.md` for ops notes.

## MCP smoke test (no UI)

```bash
PYTHONPATH=. python scripts/smoke_tools.py
```

## Connecting the MCP server to a client

Streamable HTTP endpoint (FastMCP defaults shown in `.env.example`):

```json
{
  "mcpServers": {
    "social-post-studio": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

Clients differ (Claude Desktop vs Cursor vs MCP Inspector); point them at the URL above once `docker compose up` (or `python -m mcp_server.server`) is running.

## Trade-off (what we optimized for)

We optimized for **a runnable end-to-end path on a laptop**: agents call the same Python tool implementations the MCP server exposes, so Streamlit does not need a fragile in-process HTTP MCP loop. The MCP server remains the **contract surface** (schemas + standalone process) for reviewers and external clients, while the UI and LangGraph stay simple and robust.

## What we would extend with two more days

- True **HTTP MCP client** inside LangGraph tool nodes (streaming + retries), plus golden tests against a live MCP container.
- Stronger **prompt-injection hardening** for `fetch_url` (HTML sanitization policy + CSP-style URL allowlists).

## References

Case study: `problem_statement1.md`.
