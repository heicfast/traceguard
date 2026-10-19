# Deviation Log

Every change made after a freeze (protocol, prompts, thresholds, splits, estimand,
label rules) or every accepted deviation from the SOPs/quality manual is logged
here with date, rationale, and impact BEFORE affected work continues. The log
is append-only.

| Date (UTC) | What changed / deviated | Why | Impact on frozen artifacts | Approved by |
|---|---|---|---|---|
| 2026-10-19 | Blueprint 12-week calendar discarded; phase/gate model adopted | Execution is agent-driven, not calendar-driven; gates are the scientific control points | None (schedule is not a scientific parameter) | project owner (user directive) |
| 2026-10-19 | Episode schema v1.0.0 → v1.1.0: added event types `hidden_instruction`, `invalid_proposal`; optional outcome fields | Runner requires explicit privileged-instruction events and invalid-proposal records; blueprint authorizes schema changes before study freeze | Pre-freeze change; no results existed; schema hash recorded in envelopes | implementation agent + tests |
