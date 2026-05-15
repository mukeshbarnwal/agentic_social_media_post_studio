---
name: critic-rubric
description: Scoring criteria for grounding, format, voice, and accessibility.
---

# Critic rubric

- Grounding: `source_markers` must intersect retrieved `chunk_id` set unless the post is purely opinion and labeled as such.
- Format: hook present; 3–5 hashtags; body length within brief; carousel captions align with slide count.
- Voice: matches selected tone without contradictions; avoids unverifiable superlatives.
- Accessibility: each slide has non-empty `alt_text`; figure selections include modality `figure` when available.
- Routing: if grounding fails → `research`; if tone/structure fails → `copywriter`; if alt/prompt issues → `visual`.
