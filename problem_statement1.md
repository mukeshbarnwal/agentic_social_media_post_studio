

================ PAGE 1 ================

Al Engineer Case Study

Al Engineer Case Study

Agentic Social Media Post Studio

1. Overview

Build a working proof-of-concept for an Al-powered social media post studio that helps a user produce
LinkedIn-style posts (single image or multi-slide carousels) grounded in their own material — PDFs,
uploaded images, and live web content.

This is an Al Engineer case study, not a general full-stack task. We care most about how you design and
operate the agentic system underneath the UI: how agents collaborate, how tools are exposed, how
knowledge is retrieved, how quality is measured, and how the whole thing is shipped so someone else
can run it.

Timeframe: 4 working days. We do not expect a finished product. We expect deliberate engineering
choices and a system we can actually run.

2. What we want to see

* A multi-agent system with clearly separated roles and explicit handoffs — not one mega-prompt
pretending to be agents.

* A custom MCP (Model Context Protocol) server that you build yourself, exposing tools that your
agents consume. The server should be runnable as a standalone process.

© Agent skills following the SKILL.md pattern: capability-scoped markdown files loaded on demand
rather than dumped into the system prompt.

* Multimodal RAG over PDFs (text, tables, embedded figures), images, and live web pages, with
retrieved sources cited in the final output.

* A simple but real evaluation harness with a small eval set and automated scoring.

Observability: every run should be traceable — agent steps, tool calls, retrieved chunks, token
usage.

A polished Streamlit, Gradio, or Dash UI that an actual user could drive end-to-end.

* A one-command deploy: docker compose up brings up the app, the MCP server, and the vector
store.

3. Functional scope

3.1 Inputs

* Topic prompt from the user (free text).

Page 1 of 8


================ PAGE 2 ================

Al Engineer Case Study

© Zero or more PDFs (resume, whitepaper, brochure, deck export). Must handle text, tables, and
embedded images.

* Zero or more uploaded images the user wants to incorporate.
© Optional URL or web search query for live grounding.

* Style preferences: tone (formal / casual / punchy), target length, number of slides, brand color
(optional).

3.2 Outputs
* A Linkedin-style post: hook, body copy, hashtags, call to action.
* A primary image (uploaded, retrieved from PDF, or generated).
© Optional carousel of additional slides, each with image + caption + alt text.
* A sources panel listing every PDF page, URL, and image that grounded the output.

* A LinkedIn-style live preview and a download (PNG per slide, plus a JSON manifest of the run).

3.3 Required user flows

1. User uploads a PDF, asks for a 5-slide carousel summarizing it; system produces grounded slides
with citations.

2. User gives only a topic and a URL; the web search / fetch path grounds the output.

3. User uploads an image and asks for a single-image post with copy around it; the visual agent writes alt text and the copywriter writes around the visual.

4. User edits a generated caption; downstream agents re-run only what is affected (no full
regeneration).

4. Agentic architecture (required)

You may use any framework — LangGraph, CrewAl, AutoGen, Pydantic Al, smolagents, or hand-rolled.
We do not score framework choice. We score the design.

4.1 Minimum agent roster

At least the following roles must exist as separately addressable agents with their own prompts, tools, and (where useful) memory:

Agent Responsibility | Tools it should have
Planner / Decomposes the user goal into a slide- Internal task graph, ability to
Orchestrator by-slide plan, decides which other agents _ dispatch to other agents

to invoke and in what order, handles re-
planning on failure

Page 2 of 8


================ PAGE 3 ================

Al Engineer Case Study

Research Agent Pulls grounding material from PDFs, MCP server tools (web_search,
uploaded images, and the web via the fetch_url, pdf_query), vector
custom MCP server store retriever

Copywriter Agent Generates the post hook, body, hashtags, _ LLM, style guide skill, RAG
and per-slide captions in the user's brand _ context from the Research
voice Agent

Visual Agent Decides the visual treatment per slide, Stable Diffusion (or mock),
calls the image generator or selects an BLIP2 / LLaVA captioner,

uploaded/retrieved image, writes alt text uploaded asset store

Critic / QA Agent Reviews the assembled post against a Read-only access to all artifacts,
rubric (factuality vs sources, brand voice, eval prompts as a skill
length, accessibility) and sends fixes back
to the relevant agent

Bonus points for a sane re-planning loop where the Critic can send work back to a specific upstream
agent rather than nuking the whole run.

4.2 Shared state and handoffs

* Agents must communicate via a structured shared state (typed object, blackboard, or graph state)
— not by string-concatenating each other's outputs.

© Every handoff should be loggable: which agent ran, what it read, what it wrote, what tools it
called.

* Cyclic flows are fine and expected (Critic > Copywriter > Critic). Make sure you have a stop
condition.

4.3 Skills as SKILL.md files

Adopt the agent-skills pattern: a skill is a folder containing a SKILL.md plus any helper assets. The SKILL.md has a short YAML formatter with a name and a description, and a body explaining when and
how to use the skill. Agents load only the skills relevant to the current step rather than carrying all
instructions in the base prompt.

At minimum, ship these skills:
¢ brand-voice — how to write in a chosen tone, with do/don't examples.
¢ linkedin-formatting — hook patterns, line-break rhythm, hashtag rules, character limits.
* citation — how to attach source ids to claims and produce the sources panel.

* image-prompting — how to turn an abstract slide into a concrete image prompt for the visual
agent.

Page 3 of 8


================ PAGE 4 ================

Al Engineer Case Study

* critic-rubric — the scoring criteria the Critic agent applies.
Document in the README how skills are discovered and loaded. We will look at your loading mechanism
— putting all five into every prompt does not count.

5. Custom MCP server (required)
You must implement your own MCP server. Consuming someone else's hosted MCP server does not

satisfy this requirement. Use the official MCP Python or TypeScript SDK.

5.1 Required tools

Tool name | Purpose

web_search Run a query against a search backend (SerpAPI, Tavily, DuckDuckGo, or a
mock) and return ranked results with snippets and URLs

fetch_url Fetch a URL, strip boilerplate, return clean markdown plus extracted
images
pdf_query Given an already-indexed PDF id and a question, return the top-k retrieved

chunks with page numbers and any associated figures

index_pdf Ingest a new PDF: parse text, tables, and images; embed; store in the
vector DB; return a pdf_id

list_sources Enumerate everything currently in the knowledge base so the planner can
decide what to ground on

You may add more. Keep tool signatures narrow and typed; we will read the schemas.

5.2 Transport and packaging
« Either stdio or streamable HTTP transport is fine. HTTP is easier for us to inspect.

« The server must be runnable standalone (e.g., python -m your_mcp_server) and also wired up as a
service in your docker-compose.yml.

* Include a short mcp-inspector or curl-based smoke test in the README so we can hit each tool
without launching the full app.

5.3 Why we ask for this

MCP is becoming the default way agents talk to tools. Building a small server end-to-end shows that you
understand tool schemas, transports, error surfaces, and the boundary between agent and capability —
not just that you can call an API.

Page 4 of 8


================ PAGE 5 ================

6. Multimodal RAG

6.1 Ingestion

¢ PDFs: extract text by section, extract tables as structured rows, extract embedded figures as
images with surrounding caption text. unstructured.io, pdfplumber + pdfminer, PyMuPDF, or
Docling all acceptable.

© Uploaded images: caption via BLIP2 / LLaVA / a vision model and store the caption alongside the
image embedding.

* Web pages: fetch via the MCP fetch_url tool, strip boilerplate, segment, embed.

6.2 Storage and retrieval

« Use a real vector store (FAISS, Qdrant, Chroma, LanceDB, Weaviate). In-memory dicts do not
count.

* Store enough metadata to reconstruct citations: source_id, page, bbox or anchor, modality.

* Hybrid retrieval (vector + BM25 or vector + keyword filter) is a plus.

6.3 Grounding contract

The Copywriter and Visual agents must receive retrieved chunks with stable ids. Every claim or visual
selection in the final post should be traceable to at least one id, surfaced in the Ul's sources panel.

7. Image generation
* Stable Diffusion via diffusers, an open-source hosted endpoint (Replicate, Together, fal), or a mock
are all acceptable.

© If mocked, return a deterministic placeholder with the prompt rendered onto it so we can see the
agent's intent.

© The Visual agent should choose between (a) using an uploaded image, (b) using an image
extracted from a PDF, or (c) generating one. Document the decision rule.

8. Evaluation harness (required)

Ship a small eval suite. We are not asking for a benchmark — we are asking you to take evals seriously.

© Atleast 5 to 10 hand-crafted test cases covering the four required flows in §3.3.

* Automated scoring for: factual grounding (claims map to retrieved sources), format adherence
(length, structure, hashtag count), and a model-graded brand-voice score.

* Ascripts/run_evals.py (or equivalent) that prints a table of scores and writes a JSON report.

Page 5 of 8


================ PAGE 6 ================

Al Engineer Case Study
¢ README must show the latest eval results.

9. Observability
* Trace every run: agent name, step number, tool calls with inputs/outputs, retrieved chunk ids, token usage.

* Acceptable backends: Langfuse, Phoenix/Arize, LangSmith, OpenTelemetry to a local collector, or a well-structured JSONL run log with a small viewer.

* From the UI, the user should be able to open the trace of the most recent run.

10. Suggested stack (not prescriptive)

¢ Ul: Streamlit, Gradio, or Dash.
¢ Agents: LangGraph, CrewAl, AutoGen, Pydantic Al, smolagents, or hand-rolled.

¢ LLMs: any open-source (Llama, Mistral, Qwen, Gemma) via Ollama / vLLM / HF Inference; or a
hosted API behind an env var. Document your choice.

© MCP: official Python or TypeScript SDK.

© Vector store: FAISS / Qdrant / Chroma / LanceDB / Weaviate.

¢ PDF: unstructured / pdfplumber / PyMuPDF / Docling.

© Vision: BLIP2 / LLaVA / Qwen-VL / Florence-2.

© Image gen: Stable Diffusion via diffusers, or a hosted endpoint, or a stub.

* Tracing: Langfuse / Phoenix / LangSmith / OTel.

11. Deployment (required)
We must be able to run your submission with minimal friction. Please satisfy all of the following:
5. A docker-compose.yml at the repo root that brings up: the UI, the MCP server, the vector store,
and any model server you depend on.

6. A.env.example listing every required environment variable (API keys, model endpoints, etc.) with
safe defaults or stubs.

7. A README quickstart of the form: cp .env.example .env, edit keys if needed, docker compose up
— and the app is reachable at a documented localhost port within five minutes on a laptop.

8. Where heavy models would prevent us from running locally, ship a MOCK_MODELS=true mode
that swaps in deterministic stubs so the full agent flow still executes.

9. A short section in the README titled "Connecting the MCP server to a client" showing the exact
JSON config snippet for plugging your MCP server into Claude Desktop or a similar MCP client.

Page 6 of 8


================ PAGE 7 ================

12. Deliverables

10. Git repository (GitHub / GitLab) with the working code.

11. README covering: architecture diagram, agent roster and handoffs, list of skills, MCP tool list
with schemas, eval results, mock-mode instructions, productionization notes.

12. ARCHITECTURE.md or equivalent with a sequence diagram of one full run (user prompt > final
post).

13. docker-compose.yml plus .env.example.
14. A 2to4 minute demo video (Loom, screen recording, or GIF) walking through one complete run.

15. A short PRODUCTION.md (one page) covering: how you would scale this, where the bottlenecks are, how you would handle PIl in uploaded PDFs, and how you would monitor drift.

13. How we score

Weight What we are looking for

Agentic architecture & multi- 20% Clear agent roles, well-designed handoffs, shared

agent design state, planner/executor split or similar pattern

Custom MCP server 15% Correct MCP protocol implementation, useful tool
surface, clean stdio or HTTP transport, exposed to the
app

Multimodal RAG quality 15% PDF + image + web ingestion, chunking strategy,
retrieval evals, citation back to source

Agent skills (SKILL.md pattern) 10% Skills as composable capability modules, loaded on
demand, not all stuffed in the system prompt

Evaluation harness 10% Actual eval set, automated scoring, results in the
README

Observability & tracing 5% Tracing of agent steps, tool calls, token usage; one-

click view of any run

UI/UX & end-to-end product 10% Streamlit/Gradio/Dash app actually works end-to-end
feel and feels polished
Deployability 10% docker compose up gets us running in under five

minutes, including MCP server

Code quality & 5% Typed, structured, tested where it matters, readable
documentation README

Page 7 of 8


================ PAGE 8 ================

14.

Al Engineer Case Study

Bonus points

Streaming agent traces into the UI in real time.

Human-in-the-loop: the user can approve, reject, or edit any agent output mid-run.

Cost and latency budget per run, surfaced in the UI.

Caching layer for retrieval and LLM calls.

A second MCP transport (e.g., both stdio and HTTP) so we can plug into different clients.
Prompt-injection defense for content fetched from the web, with a documented threat model.

A small Cl workflow that runs the eval harness on every push.

Ground rules

Using Al coding assistants is fine and expected. Be ready to explain any line of code in your follow-
up.

Do not commit secrets. Use .env.

If you mock something, label it MOCK clearly in code and README.

Cite any non-trivial code or design you adapted from a public source.

Submission

Reply to this email with the repo link and show us the demo of the POC.

Include a one-paragraph note describing the single trade-off you are most proud of, and the single
thing you would do differently with another two days.

Good luck — have fun with it.
