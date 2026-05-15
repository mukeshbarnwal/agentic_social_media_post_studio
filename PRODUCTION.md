# Production notes

## Scaling

- **Bottlenecks:** PDF ingestion + embedding throughput, synchronous LLM calls per node, and Chroma single-node persistence.
- **Scale-out pattern:** move Chroma/Qdrant to a managed service; run MCP tool services statelessly behind a queue; cache `pdf_query` results keyed by `(pdf_id, question_hash)`; use a workflow engine with durable execution (Temporal) for long runs.

## PII in uploaded PDFs

- Run **virus scanning** and **content policy** checks on uploads; block executables.
- Store uploads in **object storage** with per-tenant encryption (SSE-KMS) and short-lived signed URLs.
- Apply **redaction** pipelines for common PII patterns before indexing where feasible; restrict cross-tenant retrieval with strict `where` filters on `pdf_id`/tenant id.

## Drift monitoring

- Track eval harness metrics (`evals/latest_results.json`) over time in a metrics store.
- Log distribution shifts in chunk lengths, hashtag counts, and critic failure routes.
- Canary prompts against new model versions before full rollout.

## Security for web grounding

- `fetch_url` should use **allowlists** for domains in production, SSRF protections, and HTML sanitization tuned for markdown extraction.
- Treat all fetched HTML as **untrusted input** to downstream LLMs (prompt-injection surface).

## Observability

- JSONL traces are a stepping stone; forward the same events to **OpenTelemetry** or **Langfuse** with run/user correlation ids.
