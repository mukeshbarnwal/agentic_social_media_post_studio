AI Engineer Case Study
AI Engineer Case Study
Agentic Social Media Post Studio

Overview
Build a working proof-of-concept for an AI-powered social media post studio that helps a user produce LinkedIn-style posts (single image or multi-slide carousels) grounded in their own material — PDFs,uploaded images, and live web content.This is an AI Engineer case study, not a general full-stack task. We care most about how you design andoperate the agentic system underneath the UI: how agents collaborate, how tools are exposed, howknowledge is retrieved, how quality is measured, and how the whole thing is shipped so someone elsecan run it.

Timeframe: 4 working days. We do not expect a finished product. We expect deliberate engineeringchoices and a system we can actually run.

What we want to see
• A multi-agent system with clearly separated roles and explicit handoffs — not one mega-promptpretending to be agents.
• A custom MCP (Model Context Protocol) server that you build yourself, exposing tools that youragents consume. The server should be runnable as a standalone process.
• Agent skills following the SKILL.md pattern: capability-scoped markdown files loaded on demandrather than dumped into the system prompt.
• Multimodal RAG over PDFs (text, tables, embedded figures), images, and live web pages, withretrieved sources cited in the final output.
• A simple but real evaluation harness with a small eval set and automated scoring.• Observability: every run should be traceable — agent steps, tool calls, retrieved chunks, tokenusage.
• A polished Streamlit, Gradio, or Dash UI that an actual user could drive end-to-end.
• A one-command deploy: docker compose up brings up the app, the MCP server, and the vectorstore.

3. Functional scope
3.1 Inputs
• Topic prompt from the user (free text)
• Zero or more PDFs (resume, whitepaper, brochure, deck export). Must handle text, tables, andembedded images.
• Zero or more uploaded images the user wants to incorporate.
• Optional URL or web search query for live grounding.
• Style preferences: tone (formal / casual / punchy), target length, number of slides, brand color(optional)

3.2 Outputs
• A LinkedIn-style post: hook, body copy, hashtags, call to action.• A primary image (uploaded, retrieved from PDF, or generated).• Optional carousel of additional slides, each with image + caption + alt text.• A sources panel listing every PDF page, URL, and image that grounded the output.• A LinkedIn-style live preview and a download (PNG per slide, plus a JSON manifest of the run).


3.3 Required user flows

1.User uploads a PDF, asks for a 5-slide carousel summarizing it; system produces grounded slideswith citations.

2.User gives only a topic and a URL; the web search / fetch path grounds the output.

3.User uploads an image and asks for a single-image post with copy around it; the visual agentwrites alt text and the copywriter writes around the visual.

4.User edits a generated caption; downstream agents re-run only what is affected (no fullregeneration).

4. Agentic architecture (required)
You may use any framework — LangGraph, CrewAI, AutoGen, Pydantic AI, smolagents, or hand-rolled.We do not score framework choice. We score the design.

4.1 Minimum agent roster
At least the following roles must exist as separately addressable agents with their own prompts, tools,and (where useful) memory:Agent Responsibility Tools it should havePlanner /OrchestratorDecomposes the user goal into a slide-by-slide plan, decides which other agentsto invoke and in what order, handles re-planning on failureInternal task graph, ability todispatch to other agentsAI Engineer Case StudyPage 3 of 8Research Agent Pulls grounding material from PDFs,uploaded images, and the web via thecustom MCP serverMCP server tools (web_search,fetch_url, pdf_query), vectorstore retrieverCopywriter Agent Generates the post hook, body, hashtags,and per-slide captions in the user's brandvoiceLLM, style guide skill, RAGcontext from the ResearchAgentVisual Agent Decides the visual treatment per slide,calls the image generator or selects anuploaded/retrieved image, writes alt textStable Diffusion (or mock),BLIP2 / LLaVA captioner,uploaded asset storeCritic / QA Agent Reviews the assembled post against arubric (factuality vs sources, brand voice,length, accessibility) and sends fixes backto the relevant agentRead-only access to all artifacts,eval prompts as a skillBonus points for a sane re-planning loop where the Critic can send work back to a specific upstreamagent rather than nuking the whole run.4.2 Shared state and handoffs• Agents must communicate via a structured shared state (typed object, blackboard, or graph state)— not by string-concatenating each other's outputs.• Every handoff should be loggable: which agent ran, what it read, what it wrote, what tools itcalled.• Cyclic flows are fine and expected (Critic → Copywriter → Critic). Make sure you have a stopcondition.4.3 Skills as SKILL.md filesAdopt the agent-skills pattern: a skill is a folder containing a SKILL.md plus any helper assets. TheSKILL.md has a short YAML frontmatter with a name and a description, and a body explaining when andhow to use the skill. Agents load only the skills relevant to the current step rather than carrying allinstructions in the base prompt.At minimum, ship these skills:• brand-voice — how to write in a chosen tone, with do/don't examples.• linkedin-formatting — hook patterns, line-break rhythm, hashtag rules, character limits.• citation — how to attach source ids to claims and produce the sources panel.• image-prompting — how to turn an abstract slide into a concrete image prompt for the visualagent.AI Engineer Case StudyPage 4 of 8• critic-rubric — the scoring criteria the Critic agent applies.Document in the README how skills are discovered and loaded. We will look at your loading mechanism— putting all five into every prompt does not count.

Custom MCP server (required)You must implement your own MCP server. Consuming someone else's hosted MCP server does notsatisfy this requirement. Use the official MCP Python or TypeScript SDK.5.1 Required toolsTool name Purposeweb_search Run a query against a search backend (SerpAPI, Tavily, DuckDuckGo, or amock) and return ranked results with snippets and URLsfetch_url Fetch a URL, strip boilerplate, return clean markdown plus extractedimagespdf_query Given an already-indexed PDF id and a question, return the top-k retrievedchunks with page numbers and any associated figuresindex_pdf Ingest a new PDF: parse text, tables, and images; embed; store in thevector DB; return a pdf_idlist_sources Enumerate everything currently in the knowledge base so the planner candecide what to ground onYou may add more. Keep tool signatures narrow and typed; we will read the schemas.5.2 Transport and packaging• Either stdio or streamable HTTP transport is fine. HTTP is easier for us to inspect.• The server must be runnable standalone (e.g., python -m your_mcp_server) and also wired up as aservice in your docker-compose.yml.• Include a short mcp-inspector or curl-based smoke test in the README so we can hit each toolwithout launching the full app.5.3 Why we ask for thisMCP is becoming the default way agents talk to tools. Building a small server end-to-end shows that youunderstand tool schemas, transports, error surfaces, and the boundary between agent and capability —not just that you can call an API.AI Engineer Case StudyPage 5 of 8

Multimodal RAG6.1 Ingestion• PDFs: extract text by section, extract tables as structured rows, extract embedded figures asimages with surrounding caption text. unstructured.io, pdfplumber + pdfminer, PyMuPDF, orDocling all acceptable.• Uploaded images: caption via BLIP2 / LLaVA / a vision model and store the caption alongside theimage embedding.• Web pages: fetch via the MCP fetch_url tool, strip boilerplate, segment, embed.6.2 Storage and retrieval• Use a real vector store (FAISS, Qdrant, Chroma, LanceDB, Weaviate). In-memory dicts do notcount.• Store enough metadata to reconstruct citations: source_id, page, bbox or anchor, modality.• Hybrid retrieval (vector + BM25 or vector + keyword filter) is a plus.6.3 Grounding contractThe Copywriter and Visual agents must receive retrieved chunks with stable ids. Every claim or visualselection in the final post should be traceable to at least one id, surfaced in the UI's sources panel.

Image generation• Stable Diffusion via diffusers, an open-source hosted endpoint (Replicate, Together, fal), or a mockare all acceptable.• If mocked, return a deterministic placeholder with the prompt rendered onto it so we can see theagent's intent.• The Visual agent should choose between (a) using an uploaded image, (b) using an imageextracted from a PDF, or (c) generating one. Document the decision rule.

Evaluation harness (required)Ship a small eval suite. We are not asking for a benchmark — we are asking you to take evals seriously.• At least 5 to 10 hand-crafted test cases covering the four required flows in §3.3.• Automated scoring for: factual grounding (claims map to retrieved sources), format adherence(length, structure, hashtag count), and a model-graded brand-voice score.• A scripts/run_evals.py (or equivalent) that prints a table of scores and writes a JSON report.AI Engineer Case StudyPage 6 of 8• README must show the latest eval results.

Observability• Trace every run: agent name, step number, tool calls with inputs/outputs, retrieved chunk ids,token usage.• Acceptable backends: Langfuse, Phoenix/Arize, LangSmith, OpenTelemetry to a local collector, ora well-structured JSONL run log with a small viewer.• From the UI, the user should be able to open the trace of the most recent run.

Suggested stack (not prescriptive)• UI: Streamlit, Gradio, or Dash.• Agents: LangGraph, CrewAI, AutoGen, Pydantic AI, smolagents, or hand-rolled.• LLMs: any open-source (Llama, Mistral, Qwen, Gemma) via Ollama / vLLM / HF Inference; or ahosted API behind an env var. Document your choice.• MCP: official Python or TypeScript SDK.• Vector store: FAISS / Qdrant / Chroma / LanceDB / Weaviate.• PDF: unstructured / pdfplumber / PyMuPDF / Docling.• Vision: BLIP2 / LLaVA / Qwen-VL / Florence-2.• Image gen: Stable Diffusion via diffusers, or a hosted endpoint, or a stub.• Tracing: Langfuse / Phoenix / LangSmith / OTel.

Deployment (required)We must be able to run your submission with minimal friction. Please satisfy all of the following:

A docker-compose.yml at the repo root that brings up: the UI, the MCP server, the vector store,and any model server you depend on.

A .env.example listing every required environment variable (API keys, model endpoints, etc.) withsafe defaults or stubs.

A README quickstart of the form: cp .env.example .env, edit keys if needed, docker compose up— and the app is reachable at a documented localhost port within five minutes on a laptop.

Where heavy models would prevent us from running locally, ship a MOCK_MODELS=true modethat swaps in deterministic stubs so the full agent flow still executes.

A short section in the README titled "Connecting the MCP server to a client" showing the exactJSON config snippet for plugging your MCP server into Claude Desktop or a similar MCP client.AI Engineer Case StudyPage 7 of 8

Deliverables

Git repository (GitHub / GitLab) with the working code.

README covering: architecture diagram, agent roster and handoffs, list of skills, MCP tool listwith schemas, eval results, mock-mode instructions, productionization notes.

ARCHITECTURE.md or equivalent with a sequence diagram of one full run (user prompt → finalpost).

docker-compose.yml plus .env.example.

A 2 to 4 minute demo video (Loom, screen recording, or GIF) walking through one complete run.

A short PRODUCTION.md (one page) covering: how you would scale this, where the bottlenecksare, how you would handle PII in uploaded PDFs, and how you would monitor drift.

How we scoreCategory Weight What we are looking forAgentic architecture & multi-agent design20% Clear agent roles, well-designed handoffs, sharedstate, planner/executor split or similar patternCustom MCP server 15% Correct MCP protocol implementation, useful toolsurface, clean stdio or HTTP transport, exposed to theappMultimodal RAG quality 15% PDF + image + web ingestion, chunking strategy,retrieval evals, citation back to sourceAgent skills (SKILL.md pattern) 10% Skills as composable capability modules, loaded ondemand, not all stuffed in the system promptEvaluation harness 10% Actual eval set, automated scoring, results in theREADMEObservability & tracing 5% Tracing of agent steps, tool calls, token usage; one-click view of any runUI/UX & end-to-end productfeel10% Streamlit/Gradio/Dash app actually works end-to-endand feels polishedDeployability 10% docker compose up gets us running in under fiveminutes, including MCP serverCode quality &documentation5% Typed, structured, tested where it matters, readableREADMEAI Engineer Case StudyPage 8 of 8

Bonus points• Streaming agent traces into the UI in real time.• Human-in-the-loop: the user can approve, reject, or edit any agent output mid-run.• Cost and latency budget per run, surfaced in the UI.• Caching layer for retrieval and LLM calls.• A second MCP transport (e.g., both stdio and HTTP) so we can plug into different clients.• Prompt-injection defense for content fetched from the web, with a documented threat model.• A small CI workflow that runs the eval harness on every push.

Ground rules• Using AI coding assistants is fine and expected. Be ready to explain any line of code in your follow-up.• Do not commit secrets. Use .env.• If you mock something, label it MOCK clearly in code and README.• Cite any non-trivial code or design you adapted from a public source.

Submission• Reply to this email with the repo link and show us the demo of the POC.• Include a one-paragraph note describing the single trade-off you are most proud of, and the singlething you would do differently with another two days.Good luck — have fun with it.