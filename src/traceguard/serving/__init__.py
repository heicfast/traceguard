"""AWS-era serving plane: high-throughput inference backends.

v0.7.0 introduces a third execution tier next to the Kaggle/T4 transformers
path: OpenAI-compatible *servers* (vLLM v0.33.x, SGLang v0.5.x) running on
AWS A100/H100 instances, plus remote frontier *API* monitors.

Two providers implement the same ``AgentProvider`` contract as the local
transformers adapters:

* :class:`traceguard.serving.openai_compat.OpenAICompatProvider` — talks to
  any OpenAI-compatible endpoint (a local vLLM/SGLang server or a remote
  frontier API) over HTTP.
* :class:`traceguard.serving.vllm_server.VLLMServerManager` — launches,
  health-checks, and tears down a pinned vLLM server with per-model optimal
  kernel flags, recording the full server argv and environment as evidence.

Determinism contract (docs/INFERENCE_ENGINEERING.md §5): local seeded
generation remains the gold standard ("seeded_local").  vLLM servers accept a
per-request ``seed`` and greedy monitor scoring is deterministic in practice,
but bit-exact reproduction across server versions is NOT guaranteed — runs
record a ``determinism_class`` and reproducibility relies on the retained
raw evidence, exactly as for remote APIs ("seeded_server_best_effort" /
"unseeded_api").  The evidence pipeline keeps complete raw outputs, rendered
messages, usage, and hashes for every turn either way.
"""

from .cost import CostLedger, PriceTable, price_for
from .openai_compat import OpenAICompatProvider
from .vllm_server import ServerSpec, VLLMServerManager, build_server_argv

__all__ = [
    "CostLedger",
    "PriceTable",
    "price_for",
    "OpenAICompatProvider",
    "ServerSpec",
    "VLLMServerManager",
    "build_server_argv",
]
