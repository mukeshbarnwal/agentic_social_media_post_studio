---
name: citation
description: How to attach stable source ids to claims and build the sources panel.
---

# Citation

- Every non-obvious factual claim should map to one or more `chunk_id` values from research.
- Return `source_markers` as a JSON array of chunk ids you relied on (not free-text URLs).
- In body copy, prefer bracketed ids like `[pdf_abc:p2:t0]` only when the UI will strip or annotate them; otherwise keep prose clean and rely on `source_markers`.
- The sources panel is built from research metadata (`modality`, `page`, `path`).
