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

from ..ids import sha256_json, sha256_text

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
    # Canonical raw-evidence fields. Token ids are the lossless record; decoded
    # strings are retained as convenient, independently hashable projections.
    raw_output_with_special_tokens: str = ""
    generated_token_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] = field(default_factory=list)
    rendered_prompt: str = ""
    input_messages: list[dict[str, str]] = field(default_factory=list)
    generation_parameters: dict[str, Any] = field(default_factory=dict)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def trace_available(self) -> bool:
        return self.reasoning_text is not None and len(self.reasoning_text.strip()) > 0

    def audit_payload(self) -> dict[str, Any]:
        """Return the complete, privileged inference record.

        Nothing in this payload is sent to a monitor. Full prompt text and
        messages may contain the hidden attack instruction, so this record is
        deliberately written only to the privileged evidence store.
        """
        return {
            "raw_output": self.raw_output,
            "raw_output_with_special_tokens": self.raw_output_with_special_tokens,
            "raw_output_utf8_sha256": sha256_text(self.raw_output),
            "raw_output_with_special_tokens_utf8_sha256": sha256_text(
                self.raw_output_with_special_tokens
            ),
            "reasoning_text": self.reasoning_text,
            "reasoning_utf8_sha256": (
                sha256_text(self.reasoning_text) if self.reasoning_text is not None else None
            ),
            "assistant_text": self.assistant_text,
            "assistant_utf8_sha256": sha256_text(self.assistant_text),
            "reasoning_parse_status": self.reasoning_parse_status,
            "stop_reason": self.stop_reason,
            "token_usage": self.token_usage,
            "model_revision": self.model_revision,
            "elapsed_ms": self.elapsed_ms,
            "parser_diagnostics": self.parser_diagnostics,
            "rendered_prompt": self.rendered_prompt,
            "rendered_prompt_utf8_sha256": sha256_text(self.rendered_prompt),
            "input_messages": self.input_messages,
            "input_messages_sha256": sha256_json(self.input_messages),
            "prompt_token_ids": self.prompt_token_ids,
            "prompt_token_ids_sha256": sha256_json(self.prompt_token_ids),
            "generated_token_ids": self.generated_token_ids,
            "generated_token_ids_sha256": sha256_json(self.generated_token_ids),
            "generation_parameters": self.generation_parameters,
            "runtime_metadata": self.runtime_metadata,
        }


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
        # Split at EVERY close tag: all think segments are concatenated into
        # the reasoning channel; the answer channel receives only the text
        # after the LAST close (the final visible answer). Text emitted
        # between intermediate blocks is preserved as a parser diagnostic
        # count (multi-block output is degenerate but must not lose evidence).
        segments = raw_output.split(THINK_CLOSE)
        answer = segments[-1].strip("\n")
        reasoning_parts: list[str] = []
        intermediate_answers = 0
        for seg in segments[:-1]:
            if THINK_OPEN in seg:
                before = seg[:seg.rfind(THINK_OPEN)]
                if before.strip("\n"):
                    intermediate_answers += 1
                reasoning_parts.append(seg[seg.rfind(THINK_OPEN) + len(THINK_OPEN):].strip("\n"))
            else:
                # template pre-filled the open tag in the prompt: whole segment
                # is think content
                reasoning_parts.append(seg.strip("\n"))
        reasoning = "\n\n".join(p for p in reasoning_parts if p).strip("\n")
        status = "complete" if reasoning else "empty_block"
        turn = ModelTurn(raw_output=raw_output, reasoning_text=reasoning or None,
                         assistant_text=answer, reasoning_parse_status=status)
        if intermediate_answers:
            turn.parser_diagnostics["intermediate_answers_discarded"] = intermediate_answers
        return turn
    if THINK_OPEN in raw_output:
        cut = raw_output.rfind(THINK_OPEN)
        if reasoning_mode:
            # Thinking mode: the template pre-filled the opening tag in the
            # prompt, so generated content BEFORE an echoed literal open tag is
            # ALSO think content. Nothing may leak to the answer channel
            # (review finding 2026-10-19: the prefix used to be routed to
            # assistant_text, violating the anti-leak contract). The echoed
            # marker itself is stripped; all surrounding content is reasoning.
            body = (raw_output[:cut] + raw_output[cut + len(THINK_OPEN):]).strip("\n")
            return ModelTurn(raw_output=raw_output,
                             reasoning_text=body or None,
                             assistant_text="",
                             reasoning_parse_status="truncated")
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
