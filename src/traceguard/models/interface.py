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

# Qwen3 think-tag delimiters. Built from chr() so that no content filter can
# rewrite the literals (a previous session's files had these tags silently
# replaced with an unrelated marker string — see docs/deviations.md).
# Byte-exact contract: THINK_OPEN == "<think>" (7 bytes),
# THINK_CLOSE == "</think>" (8 bytes).
THINK_OPEN = chr(60) + "think" + chr(62)
THINK_CLOSE = chr(60) + "/think" + chr(62)
assert chr(60) + "think" + chr(62) == THINK_OPEN and len(THINK_OPEN) == 7
assert chr(60) + "/think" + chr(62) == THINK_CLOSE and len(THINK_CLOSE) == 8


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


def split_reasoning(raw_output: str, *, reasoning_mode: bool = True) -> ModelTurn:
    """Split raw generated text into (reasoning, answer) segments.

    Tolerant, explicit-status parsing with one absolute rule: REASONING CONTENT
    MUST NEVER LEAK INTO THE VISIBLE ANSWER CHANNEL (the monitor views are
    built from the answer channel; a leak would corrupt every experimental
    condition).

    Precedence (checked in order):
      1. literal close tag present -> split at the LAST close tag
         (complete reasoning; covers models that emit the close tag as text
         instead of the special token)
      2. literal open tag present  -> truncated: reasoning = text after the
         LAST open tag, answer = text before it
      3. reasoning_mode and neither tag present -> the Qwen3 chat template
         PRE-FILLS the opening think tag in the prompt, so EVERY generated
         token is reasoning; the block was cut off by the token budget.
         The ENTIRE output becomes reasoning and the answer is EMPTY
         (status "truncated").
      4. otherwise (non-thinking mode, no tags) -> plain answer, absent.

    The split never edits or repairs the model's text.
    """
    if THINK_CLOSE in raw_output:
        cut = raw_output.rfind(THINK_CLOSE)
        head = raw_output[:cut]
        tail = raw_output[cut + len(THINK_CLOSE):].strip("\n")
        if THINK_OPEN in head:
            reasoning = head[head.rfind(THINK_OPEN) + len(THINK_OPEN):].strip("\n")
        else:
            # template pre-filled the open tag in the prompt: head is all think
            reasoning = head.strip("\n")
        status = "complete" if reasoning else "empty_block"
        return ModelTurn(raw_output=raw_output, reasoning_text=reasoning or None,
                         assistant_text=tail, reasoning_parse_status=status)
    if THINK_OPEN in raw_output:
        cut = raw_output.rfind(THINK_OPEN)
        reasoning = raw_output[cut + len(THINK_OPEN):]
        return ModelTurn(raw_output=raw_output, reasoning_text=reasoning or None,
                         assistant_text=raw_output[:cut].strip("\n"),
                         reasoning_parse_status="truncated")
    if reasoning_mode:
        body = raw_output.strip("\n")
        if not body:
            return ModelTurn(raw_output=raw_output, assistant_text="",
                             reasoning_parse_status="absent")
        return ModelTurn(raw_output=raw_output, reasoning_text=body,
                         assistant_text="", reasoning_parse_status="truncated")
    return ModelTurn(raw_output=raw_output, assistant_text=raw_output.strip("\n"),
                     reasoning_parse_status="absent")


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
