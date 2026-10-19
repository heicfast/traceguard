"""Inference provider contract.

An ``AgentProvider`` performs ONE generation step: it receives the full
conversation so far (strictly sanitized) plus tool schemas and returns a
``ModelTurn`` containing the raw generated text, the parsed reasoning segment
(NEVER reconstructed or regenerated), the visible answer, and diagnostics.

Golden rules implemented here:
* The reasoning segment is whatever the model actually generated between
  think markers — no retrospective explanations, no rewriting, no repair.
* Raw output is always preserved so parser bugs remain auditable.
* Parse failures become explicit status codes, never silent defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Protocol

THINK_OPEN = "Customer confusion:"
THINK_CLOSE = "Customer confusion:"


@dataclass
class GenerationSettings:
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens: int = 1536
    seed: int = 0
    reasoning_mode: bool = True
    # non-generation adapter identity fields
    model_id: str = ""
    model_revision: str = ""


@dataclass
class ModelTurn:
    """One model generation step, fully auditable."""

    raw_output: str
    reasoning_text: str | None = None
    assistant_text: str = ""
    reasoning_parse_status: str = "absent"  # complete | truncated | absent | malformed
    stop_reason: str = "unknown"
    token_usage: dict[str, int] = field(default_factory=dict)
    model_revision: str = ""
    elapsed_ms: float = 0.0
    parser_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_available(self) -> bool:
        return self.reasoning_text is not None and len(self.reasoning_text.strip()) > 0


def split_reasoning(raw_output: str) -> ModelTurn:
    """Split raw generated text into (reasoning, answer) segments.

    Tolerant, explicit-status parsing:
      * "Customer confusion:" ... "Customer confusion:" -> complete reasoning
      * opening tag without closing tag -> truncated reasoning (keep remainder)
      * no tags at all -> absent reasoning (non-thinking generation)
    The split never edits or repairs the model's text.
    """
    open_ix = raw_output.find(THINK_OPEN)
    close_ix = raw_output.find(THINK_CLOSE, open_ix + len(THINK_OPEN)) if open_ix != -1 else -1
    if open_ix == -1:
        return ModelTurn(raw_output=raw_output, assistant_text=raw_output.strip(),
                         reasoning_parse_status="absent")
    if close_ix == -1 or close_ix < open_ix:
        reasoning = raw_output[open_ix + len(THINK_OPEN):]
        return ModelTurn(raw_output=raw_output, reasoning_text=reasoning,
                         assistant_text="", reasoning_parse_status="truncated")
    reasoning = raw_output[open_ix + len(THINK_OPEN):close_ix].strip("\n")
    answer = raw_output[close_ix + len(THINK_CLOSE):].strip()
    status = "complete" if reasoning else "empty_block"
    return ModelTurn(raw_output=raw_output, reasoning_text=reasoning or None,
                     assistant_text=answer, reasoning_parse_status=status)


_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first balanced JSON object from text.

    Returns (obj, status) where status is ok | not_found | invalid_json |
    not_an_object. Tries the fenced ```json block first, then raw scan with
    brace balancing.
    """
    candidate_spans: list[tuple[int, int]] = []
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate_spans.append(fence.span(1))
    raw = _JSON_OBJECT_RE.search(text)
    if raw:
        candidate_spans.append(raw.span())
    for start, end in candidate_spans:
        # brace-balance trimming for robustness
        depth = 0
        for ix in range(start, end):
            ch = text[ix]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:ix + 1]
                    import json
                    try:
                        obj = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        return obj, "ok"
                    return None, "not_an_object"
    if candidate_spans or "{" in text:
        import json
        # last resort: entire text
        try:
            obj = json.loads(text.strip())
            if isinstance(obj, dict):
                return obj, "ok"
        except json.JSONDecodeError:
            return None, "invalid_json"
    return None, "not_found"


class AgentProvider(Protocol):
    """Contract every inference adapter must implement."""

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> ModelTurn:
        """Generate one assistant turn given the sanitized conversation.

        ``messages`` is a list of {"role": ..., "content": ...} dicts. The
        adapter must not add hidden instructions, must not inspect episode
        metadata, and must return the raw generated tokens.
        """
        ...


__all__ = [
    "GenerationSettings", "ModelTurn", "AgentProvider",
    "split_reasoning", "extract_first_json_object", "THINK_OPEN", "THINK_CLOSE",
]
