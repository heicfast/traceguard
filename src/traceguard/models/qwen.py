"""Qwen3 inference adapter (Hugging Face transformers).

Implements the model card's exact behavior
(https://huggingface.co/Qwen/Qwen3-4B):

* ``tokenizer.apply_chat_template(..., enable_thinking=...)`` selects
  thinking / non-thinking mode. transformers>=4.51 is REQUIRED
  (older versions raise ``KeyError: 'qwen3'``).
* Thinking output is a ``Customer confusion: ... Customer confusion:`` block;
  the closing tag is token id 151668. Splitting is done at the TOKEN level
  using the LAST occurrence of the closing tag (model-card algorithm), with a
  text-level fallback.
* Conversation history handed back to the model must contain ONLY the final
  answer part (think blocks stripped) — the chat template expects this and we
  enforce it.
* Thinking mode sampling: temperature 0.6, top_p 0.95, top_k 20 (never
  greedy); non-thinking: temperature 0.7, top_p 0.8, top_k 20. The monitor
  uses temperature 0 for reproducible scoring (documented deviation, capped
  output, JSON format).
* Exact model revision (commit SHA) is resolved via huggingface_hub and
  recorded in every ModelTurn.

The adapter is GPU-only and lazily imported; all CPU tests use FakeModel.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..ids import sha256_text
from .interface import THINK_OPEN, GenerationSettings, ModelTurn

QWEN_CLOSE_THINK_TOKEN_ID = 151668


class QwenHFProvider:
    """transformers-based Qwen3 provider for Kaggle GPU sessions."""

    provider_name = "qwen_hf"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device_map: str | dict | None = None,
        torch_dtype: str = "auto",
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_id = model_id
        self.device_map = device_map or {"": 0}
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model_revision = self._resolve_revision(model_id, revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, torch_dtype=torch_dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        self.base_seed = int(seed)

    @staticmethod
    def _resolve_revision(model_id: str, revision: str | None) -> str:
        if revision and len(revision) >= 6 and not revision.startswith("PIN"):
            return revision
        try:
            from huggingface_hub import model_info
            info = model_info(model_id)
            return getattr(info, "sha", "unknown") or "unknown"
        except Exception:  # network-locked environments
            return revision or "unresolved_offline"

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> ModelTurn:
        torch = self._torch
        started = time.perf_counter()
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.reasoning_mode,
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        }
        if settings.temperature > 0:
            gen_kwargs.update({
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
            })
        torch.manual_seed((self.base_seed + settings.seed * 7919) % (2**31))
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:].tolist()

        # Token-level think/answer split (model card algorithm: LAST close tag)
        reasoning_text: str | None = None
        parse_status = "absent"
        try:
            index = len(new_ids) - new_ids[::-1].index(QWEN_CLOSE_THINK_TOKEN_ID)
        except ValueError:
            index = 0
        if index > 0:
            reasoning_ids = new_ids[:index]
            answer_ids = new_ids[index:]
            reasoning_text = self.tokenizer.decode(reasoning_ids, skip_special_tokens=True).strip("\n")
            answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip("\n")
            if reasoning_text:
                parse_status = "complete" if len(reasoning_ids) < settings.max_new_tokens else "truncated"
            else:
                # empty think block: template-prefilled non-thinking turn
                parse_status = "empty_block"
                reasoning_text = None
        else:
            answer_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip("\n")
            if THINK_OPEN in answer_text:
                # model opened a think tag but never closed it
                raw_open = answer_text.find(THINK_OPEN)
                reasoning_text = answer_text[raw_open + len(THINK_OPEN):]
                answer_text = answer_text[:raw_open].strip("\n")
                parse_status = "truncated"

        usage = {
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": len(new_ids),
        }
        return ModelTurn(
            raw_output=self.tokenizer.decode(new_ids, skip_special_tokens=True),
            reasoning_text=reasoning_text,
            assistant_text=answer_text,
            reasoning_parse_status=parse_status,
            stop_reason="length" if len(new_ids) >= settings.max_new_tokens else "eos",
            token_usage=usage,
            model_revision=self.model_revision,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            parser_diagnostics={"provider": self.provider_name,
                                "model_id": self.model_id,
                                "prompt_sha256": sha256_text(prompt_text)[:16]},
        )


def build_provider_from_config(config, seed: int = 0):
    """Factory used by GPU runtimes; returns a loaded provider instance."""
    agent_cfg = config.agent
    return QwenHFProvider(
        model_id=agent_cfg.model_id,
        revision=None if agent_cfg.model_revision.startswith("PIN") else agent_cfg.model_revision,
        device_map=os.environ.get("TRACEGUARD_DEVICE_MAP", '{"": 0}').replace("'", '"'),
        torch_dtype=os.environ.get("TRACEGUARD_TORCH_DTYPE", "auto"),
        seed=seed,
    )


__all__ = ["QwenHFProvider", "build_provider_from_config", "QWEN_CLOSE_THINK_TOKEN_ID"]
