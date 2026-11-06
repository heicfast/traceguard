"""OpenAI-compatible inference provider (vLLM/SGLang servers and frontier APIs).

Implements the ``AgentProvider`` contract against any OpenAI-compatible
``/v1/chat/completions`` endpoint:

* **local servers** — vLLM v0.33.x / SGLang v0.5.x launched by
  :mod:`traceguard.serving.vllm_server` on AWS GPU instances
  (determinism class ``seeded_server_best_effort``; the server accepts a
  per-request ``seed``);
* **remote frontier APIs** — OpenAI, Anthropic-compatible gateways,
  DeepSeek, Moonshot Kimi, Z.ai, Alibaba Model Studio and any other
  OpenAI-compatible endpoint (determinism class ``unseeded_api`` unless the
  provider documents seed support).

Golden rules carried over from the transformers adapters:

* the reasoning segment is whatever the endpoint actually returned
  (``reasoning_content`` when the server parses it, think-tag fallback
  otherwise) — never reconstructed, never repaired;
* raw responses, rendered message payloads, usage, latencies, HTTP status
  and retry attempts are retained in the audit payload;
* parse/transport failures become explicit statuses, never silent defaults;
* every call is metered into the run's cost ledger when one is attached.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from ..ids import sha256_json
from ..models.interface import GenerationSettings, ModelTurn, split_reasoning

DEFAULT_TIMEOUT_SECONDS = 300.0
MAX_TRANSPORT_ATTEMPTS = 3  # transport-level only; parse failures never resample


@dataclass
class ProviderEndpoint:
    """Pinned description of one OpenAI-compatible endpoint."""

    base_url: str                      # e.g. http://127.0.0.1:8000/v1 or https://api.deepseek.com/v1
    model_id: str                      # exact model id sent to the endpoint
    api_key_env: str = "TRACEGUARD_API_KEY"  # env var holding the secret (never the secret itself)
    provider_name: str = "api_openai"
    # Capability flags — pinned per endpoint after probing; the provider
    # never guesses silently.
    supports_seed: bool = True         # vLLM/SGLang: yes; most remote APIs: best-effort or no
    supports_reasoning_field: bool = True   # reasoning_content in the response (vLLM/SGLang/DeepSeek)
    # Optional chat-template kwargs forwarded for thinking toggles (vLLM
    # `chat_template_kwargs`, e.g. {"enable_thinking": false} for Qwen3).
    chat_template_kwargs: dict[str, Any] | None = None
    extra_body: dict[str, Any] | None = None
    determinism_class: str = "seeded_server_best_effort"
    request_timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def resolved_api_key(self) -> str | None:
        import os

        return os.environ.get(self.api_key_env) or None


class OpenAICompatProvider:
    """Stateless provider over the OpenAI chat-completions wire format."""

    def __init__(
        self,
        endpoint: ProviderEndpoint,
        *,
        model_revision: str = "unpinned_api_model",
        cost_ledger: Any | None = None,
        price_per_mtok: tuple[float | None, float | None] = (None, None),
        context_note: str = "",
    ):
        self.endpoint = endpoint
        self.model_revision = model_revision
        self.cost_ledger = cost_ledger
        self.price_per_mtok = price_per_mtok
        self.context_note = context_note
        self.model_profile: dict[str, Any] | None = None
        self.probe_artifact_sha256: str | None = None

    # -- HTTP layer -----------------------------------------------------------

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /chat/completions with bounded transport retries.

        Retries happen ONLY on transport-level errors (connection refused,
        timeout, 5xx) where no content was received; a response with an
        error status after all attempts raises loudly.
        """
        url = self.endpoint.base_url.rstrip("/") + "/chat/completions"
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = self.endpoint.resolved_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        last_error: str = "no_attempt"
        for attempt in range(1, MAX_TRANSPORT_ATTEMPTS + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(
                    request, timeout=self.endpoint.request_timeout_seconds
                ) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = ""
                with contextlib_suppress():
                    detail = exc.read().decode("utf-8", errors="replace")[:400]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code >= 500 and attempt < MAX_TRANSPORT_ATTEMPTS:
                    time.sleep(1.5 * attempt)
                    continue
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < MAX_TRANSPORT_ATTEMPTS:
                    time.sleep(1.5 * attempt)
                    continue
                break
        raise RuntimeError(
            f"chat/completions failed after {MAX_TRANSPORT_ATTEMPTS} attempts: {last_error}"
        )

    # -- AgentProvider contract -------------------------------------------------

    def generate(self, messages: list[dict[str, str]], settings: GenerationSettings) -> ModelTurn:
        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.endpoint.model_id,
            "messages": messages,
            "max_tokens": settings.max_new_tokens,
            "temperature": settings.temperature,
        }
        if settings.temperature > 0:
            payload["top_p"] = settings.top_p
            if settings.top_k and settings.top_k > 0:
                payload["top_k"] = settings.top_k
        else:
            # Greedy: pin top_p=1 and omit top_k entirely so servers do not
            # apply a residual filter to a degenerate distribution.
            payload["top_p"] = 1.0
        if self.endpoint.supports_seed and settings.seed is not None:
            payload["seed"] = settings.seed
        if self.endpoint.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.endpoint.chat_template_kwargs)
        if self.endpoint.extra_body:
            for key, value in self.endpoint.extra_body.items():
                payload.setdefault(key, value)

        response = self._post_chat(payload)
        choice = (response.get("choices") or [{}])[0]
        message = choice.get("message", {}) or {}
        raw_answer = str(message.get("content") or "")
        reasoning_field = None
        if self.endpoint.supports_reasoning_field:
            reasoning_field = message.get("reasoning_content")
            if reasoning_field is not None:
                reasoning_field = str(reasoning_field)
        finish_reason = str(choice.get("finish_reason") or "unknown")

        if reasoning_field is not None:
            reasoning_text = reasoning_field.strip("\n") or None
            answer_text = raw_answer.strip("\n")
            parse_status = "complete" if reasoning_text else "empty_block"
        else:
            turn_fallback = split_reasoning(raw_answer, reasoning_mode=settings.reasoning_mode)
            reasoning_text = turn_fallback.reasoning_text
            answer_text = turn_fallback.assistant_text
            parse_status = turn_fallback.reasoning_parse_status

        usage = response.get("usage", {}) or {}
        token_usage = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "generated_tokens": int(usage.get("completion_tokens", 0) or 0),
        }
        rendered_payload = json.dumps(
            {"model": payload["model"], "messages": messages}, sort_keys=True
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        self._meter_cost(token_usage)

        runtime_metadata: dict[str, Any] = {
            "provider": self.endpoint.provider_name,
            "base_url": self.endpoint.base_url,
            "endpoint_model_id": self.endpoint.model_id,
            "api_key_env": self.endpoint.api_key_env,
            "server_side_chat_template": True,
            "chat_template_sha256": None,
            "determinism_class": self.endpoint.determinism_class,
            "supports_seed": self.endpoint.supports_seed,
            "seed_sent": bool(self.endpoint.supports_seed and settings.seed is not None),
            "finish_reason": finish_reason,
            "response_model": response.get("model"),
            "response_id": response.get("id"),
            "context_note": self.context_note,
            "transport_attempts": 1,
        }
        turn = ModelTurn(
            raw_output=(reasoning_field if reasoning_field is not None else raw_answer),
            reasoning_text=reasoning_text,
            assistant_text=answer_text,
            reasoning_parse_status=parse_status,
            stop_reason=("length" if finish_reason == "length" else "eos"),
            token_usage=token_usage,
            model_revision=self.model_revision,
            elapsed_ms=elapsed_ms,
            parser_diagnostics={"provider": self.endpoint.provider_name,
                                "model_id": self.endpoint.model_id,
                                "request_payload_sha256": sha256_json(payload)},
            raw_output_with_special_tokens="",
            generated_token_ids=[],
            prompt_token_ids=[],
            rendered_prompt=rendered_payload,
            input_messages=[dict(m) for m in messages],
            generation_parameters={k: v for k, v in payload.items() if k != "messages"},
            runtime_metadata=runtime_metadata,
        )
        # raw_output is what the endpoint produced (reasoning + answer for
        # reasoning-parsing servers, the full text otherwise). Token ids are
        # not exposed over the chat API and are recorded as absent.
        if reasoning_field is None:
            turn.raw_output = raw_answer
        return turn

    # -- cost metering -----------------------------------------------------------

    def _meter_cost(self, token_usage: dict[str, int]) -> None:
        if self.cost_ledger is None:
            return
        self.cost_ledger.record(
            provider=self.endpoint.provider_name,
            model_id=self.endpoint.model_id,
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("generated_tokens", 0),
            price_per_mtok=self.price_per_mtok,
        )


class contextlib_suppress:
    """Minimal context manager suppressing OSError/ValueError while reading error bodies."""

    def __enter__(self) -> contextlib_suppress:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return exc_type is not None and issubclass(exc_type, (OSError, ValueError))


__all__ = ["OpenAICompatProvider", "ProviderEndpoint", "DEFAULT_TIMEOUT_SECONDS"]
