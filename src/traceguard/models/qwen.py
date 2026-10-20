"""Qwen3 inference adapter (Hugging Face transformers).

Implements the model card's exact behavior
(https://huggingface.co/Qwen/Qwen3-4B):

* ``tokenizer.apply_chat_template(..., enable_thinking=...)`` selects
  thinking / non-thinking mode. transformers>=4.51 is REQUIRED
  (older versions raise ``KeyError: 'qwen3'``).
* Thinking output is a think-tag block; the closing tag is token id 151668.
  Splitting is done at the TOKEN level using the LAST occurrence of the
  closing tag (a deliberate divergence from the model card's first-occurrence
  snippet: taking the LAST close keeps the answer channel free of trailing
  think content in degenerate multi-block outputs — keep this), with the
  unified text-level fallback ``split_reasoning`` (which enforces: reasoning
  never leaks into the visible answer channel — thinking mode with a
  pre-filled opening tag and no closing tag means EVERY generated token is
  reasoning).
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
import platform
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

from ..ids import sha256_text
from .interface import GenerationSettings, ModelTurn, split_reasoning

QWEN_CLOSE_THINK_TOKEN_ID = 151668


@contextmanager
def _isolated_generation_rng(torch: Any, device: Any, seed: int) -> Iterator[None]:
    """Seed one generation call without leaking mutations to process RNG state.

    Transformers 5.x validates unknown ``generate`` kwargs against the model
    forward signature and rejects ``generator=``. Sampling still consumes the
    active PyTorch RNG, so use PyTorch's supported save/seed/restore scope.
    TRACEGUARD runs generations sequentially within each isolated GPU worker;
    this process-global scope is therefore deterministic and race-free.
    """
    devices: list[int] = []
    if getattr(device, "type", None) == "cuda":
        # torch.manual_seed seeds every CUDA device. Save/restore all visible
        # CUDA generators as well, so a multi-GPU preflight cannot perturb a
        # sibling device even before workers isolate CUDA_VISIBLE_DEVICES.
        devices = list(range(torch.cuda.device_count()))
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        yield


class QwenHFProvider:
    """transformers-based Qwen3 provider for Kaggle GPU sessions."""

    provider_name = "qwen_hf"

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        tokenizer_revision: str | None = None,
        device_map: str | dict | None = None,
        torch_dtype: str = "auto",
        seed: int = 0,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Environment overrides make the kernel-controlled dtype/device policy
        # apply to EVERY construction path (review finding: these env vars were
        # previously read only by a dead factory function, so the T4 fp16
        # forcing was silently never applied).
        if torch_dtype == "auto":
            torch_dtype = os.environ.get("TRACEGUARD_TORCH_DTYPE", "auto")
        if device_map is None:
            raw_map = os.environ.get("TRACEGUARD_DEVICE_MAP", "").strip()
            if raw_map:
                import json as _json

                try:
                    parsed = _json.loads(raw_map)
                    device_map = parsed if isinstance(parsed, dict) else raw_map
                except _json.JSONDecodeError:
                    device_map = raw_map  # e.g. "cuda:0" or "auto"
        self._torch = torch
        self.model_id = model_id
        self.device_map = device_map or {"": 0}
        self.applied_dtype = torch_dtype
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=tokenizer_revision or revision
        )
        self.model_revision = self._resolve_revision(model_id, revision)
        self.tokenizer_revision = self._resolve_revision(
            model_id, tokenizer_revision or revision
        )
        # transformers v5 renamed `torch_dtype` to `dtype`; support both.
        import transformers
        dtype_value = torch_dtype
        if dtype_value == "float16":
            dtype_value = torch.float16
        elif dtype_value == "bfloat16":
            dtype_value = torch.bfloat16
        major = int(transformers.__version__.split(".")[0])
        load_kwargs = {"revision": revision, "device_map": self.device_map}
        if major >= 5:
            load_kwargs["dtype"] = dtype_value
        else:
            load_kwargs["torch_dtype"] = dtype_value
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
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
        import transformers

        torch = self._torch
        started = time.perf_counter()
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=settings.reasoning_mode,
        )
        inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.model.device)
        effective_seed = (self.base_seed + settings.seed * 7919) % (2**31)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "do_sample": settings.temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id
            or self.tokenizer.eos_token_id,
        }
        # Prefill-logits memory guard (2026-10-20): by default generate computes
        # the FULL [seq_len, vocab] logits tensor during prefill; with ~20K-token
        # monitor prompts and a ~152K vocab that single fp32 allocation is ~12 GiB
        # and OOMs the T4 (observed twice on Kaggle). Keeping only the last
        # position's logits is semantically identical for sampling (we never
        # sample from earlier positions). transformers >=5 renamed
        # num_logits_to_keep -> logits_to_keep and REJECTS unknown generate
        # kwargs, so resolve the supported name from the model's forward
        # signature at runtime.
        gen_kwargs.update(self._prefill_logits_guard())
        if settings.temperature > 0:
            gen_kwargs.update({
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "top_k": settings.top_k,
            })
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with (
            _isolated_generation_rng(torch, inputs["input_ids"].device, effective_seed),
            torch.inference_mode(),
        ):
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        new_ids = output_ids[0][inputs["input_ids"].shape[-1]:].tolist()
        decoded_output = self.tokenizer.decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        raw_output_with_special_tokens = self.tokenizer.decode(
            new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )

        # Token-level think/answer split (model card algorithm: LAST close tag).
        # If the closing special token never appears (token budget exhausted
        # mid-think, or the model emitted the tag as literal text), fall back
        # to split_reasoning, which guarantees reasoning can never leak into
        # the visible answer channel.
        try:
            index = len(new_ids) - new_ids[::-1].index(QWEN_CLOSE_THINK_TOKEN_ID)
        except ValueError:
            index = 0
        if index > 0:
            reasoning_ids = new_ids[:index]
            answer_ids = new_ids[index:]
            reasoning_text = self.tokenizer.decode(reasoning_ids,
                                                   skip_special_tokens=True).strip("\n")
            answer_text = self.tokenizer.decode(answer_ids,
                                                skip_special_tokens=True).strip("\n")
            if reasoning_text:
                parse_status = "complete" if len(reasoning_ids) < settings.max_new_tokens \
                    else "truncated"
            else:
                # empty think block: template-prefilled non-thinking turn
                parse_status = "empty_block"
                reasoning_text = None
        else:
            turn = split_reasoning(decoded_output, reasoning_mode=settings.reasoning_mode)
            reasoning_text = turn.reasoning_text
            answer_text = turn.assistant_text
            parse_status = turn.reasoning_parse_status

        usage = {
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "generated_tokens": len(new_ids),
        }
        runtime_metadata: dict[str, Any] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_build": torch.version.cuda,
            "transformers": transformers.__version__,
            "model_class": type(self.model).__name__,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": getattr(self.tokenizer, "name_or_path", ""),
            "tokenizer_revision": self.tokenizer_revision,
            "tokenizer_vocab_size": len(self.tokenizer),
            "chat_template_sha256": sha256_text(self.tokenizer.chat_template or ""),
            "device": str(inputs["input_ids"].device),
            "device_map": self.device_map,
            "dtype_policy": str(self.applied_dtype),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "rng_strategy": "torch.random.fork_rng+torch.manual_seed",
        }
        if torch.cuda.is_available():
            device_index = inputs["input_ids"].device.index or 0
            props = torch.cuda.get_device_properties(device_index)
            runtime_metadata["gpu"] = {
                "name": props.name,
                "capability": list(torch.cuda.get_device_capability(device_index)),
                "total_memory_bytes": props.total_memory,
            }
        serializable_gen_kwargs = dict(gen_kwargs)
        serializable_gen_kwargs["seed"] = effective_seed
        return ModelTurn(
            raw_output=decoded_output,
            reasoning_text=reasoning_text,
            assistant_text=answer_text,
            reasoning_parse_status=parse_status,
            stop_reason="length" if len(new_ids) >= settings.max_new_tokens else "eos",
            token_usage=usage,
            model_revision=self.model_revision,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
            parser_diagnostics={"provider": self.provider_name,
                                "model_id": self.model_id,
                                "prompt_sha256": sha256_text(prompt_text)},
            raw_output_with_special_tokens=raw_output_with_special_tokens,
            generated_token_ids=[int(token_id) for token_id in new_ids],
            prompt_token_ids=[int(token_id) for token_id in inputs["input_ids"][0].tolist()],
            rendered_prompt=prompt_text,
            input_messages=deepcopy(messages),
            generation_parameters=serializable_gen_kwargs,
            runtime_metadata=runtime_metadata,
        )


    def _prefill_logits_guard(self) -> dict[str, int]:
        """Kwarg that limits prefill logits to the sampled position, if the
        installed transformers accepts one; {} otherwise (never an unknown
        kwarg, which v5's generate validation rejects)."""
        import inspect

        try:
            forward_params = set(inspect.signature(self.model.forward).parameters)
        except (AttributeError, TypeError, ValueError):
            return {}
        for name in ("logits_to_keep", "num_logits_to_keep"):
            if name in forward_params:
                return {name: 1}
        return {}

    def between_episodes(self) -> None:
        """Release cached GPU memory between episodes (housekeeping only).

        This does NOT touch sampling or determinism: generation is already
        complete and every turn's seeds are set explicitly per call.
        """
        import contextlib

        with contextlib.suppress(Exception):
            self._torch.cuda.empty_cache()


def build_provider_from_config(config, seed: int = 0):
    """Factory used by GPU runtimes; returns a loaded provider instance."""
    import json

    agent_cfg = config.agent
    device_map = {"": 0}
    raw_map = os.environ.get("TRACEGUARD_DEVICE_MAP", "").strip()
    if raw_map:
        try:
            device_map = json.loads(raw_map)
            if not isinstance(device_map, dict):
                device_map = {"": 0}
        except json.JSONDecodeError:
            device_map = {"": 0}
    return QwenHFProvider(
        model_id=agent_cfg.model_id,
        revision=None if agent_cfg.model_revision.startswith("PIN") else agent_cfg.model_revision,
        tokenizer_revision=(
            None
            if agent_cfg.tokenizer_revision.startswith("PIN")
            else agent_cfg.tokenizer_revision
        ),
        device_map=device_map,
        torch_dtype=os.environ.get("TRACEGUARD_TORCH_DTYPE", "auto"),
        seed=seed,
    )


__all__ = [
    "QwenHFProvider",
    "build_provider_from_config",
    "QWEN_CLOSE_THINK_TOKEN_ID",
]
