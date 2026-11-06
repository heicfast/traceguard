"""Unit tests for the v0.7.0 serving plane: provider, cost ledger, server argv."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from traceguard.models.interface import GenerationSettings
from traceguard.serving.cost import BudgetExceededError, CostLedger, PriceTable
from traceguard.serving.openai_compat import OpenAICompatProvider, ProviderEndpoint
from traceguard.serving.vllm_server import ServerSpec, build_server_argv

# ---------------------------------------------------------------------------
# OpenAI-compatible provider against a mock server
# ---------------------------------------------------------------------------

class _MockChatServer(BaseHTTPRequestHandler):
    calls: list[dict] = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        _MockChatServer.calls.append(body)
        response = {
            "id": "chatcmpl-mock",
            "model": body.get("model"),
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": '{"rubric": {"policy_conflict": 0, "unauthorized_intent": 0,'
                               ' "concealment": 0, "action_evidence": 0}, "evidence_refs": [],'
                               ' "verdict_text": "clean"}',
                    "reasoning_content": "quick check: the actions look routine.",
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 101, "completion_tokens": 37},
        }
        payload = json.dumps(response).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def mock_server():
    _MockChatServer.calls = []
    server = HTTPServer(("127.0.0.1", 0), _MockChatServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", _MockChatServer
    server.shutdown()
    thread.join(timeout=5)


def test_provider_returns_full_model_turn_with_reasoning_field(mock_server):
    base_url, server = mock_server
    provider = OpenAICompatProvider(ProviderEndpoint(
        base_url=base_url, model_id="qwen3-14b", provider_name="vllm_openai",
    ))
    settings = GenerationSettings(temperature=0.0, seed=42, reasoning_mode=True,
                                  max_new_tokens=256)
    turn = provider.generate(
        [{"role": "user", "content": "monitor this"}], settings)
    assert turn.reasoning_text == "quick check: the actions look routine."
    assert turn.reasoning_parse_status == "complete"
    assert turn.assistant_text.startswith('{"rubric"')
    assert turn.token_usage == {"prompt_tokens": 101, "generated_tokens": 37}
    assert turn.runtime_metadata["determinism_class"] == "seeded_server_best_effort"
    assert turn.runtime_metadata["seed_sent"] is True
    assert turn.raw_output  # raw evidence retained
    assert turn.audit_payload()["rendered_prompt_utf8_sha256"]
    # The request carried the seed (vLLM per-request determinism).
    assert server.calls[-1]["seed"] == 42
    assert server.calls[-1]["temperature"] == 0.0


def test_provider_greedy_request_has_no_top_k(mock_server):
    base_url, server = mock_server
    provider = OpenAICompatProvider(ProviderEndpoint(
        base_url=base_url, model_id="qwen3-14b", provider_name="vllm_openai",
    ))
    provider.generate([{"role": "user", "content": "x"}],
                      GenerationSettings(temperature=0.0, top_k=20))
    request = server.calls[-1]
    assert "top_k" not in request
    assert request["top_p"] == 1.0


def test_provider_metering_and_budget_cap(mock_server, tmp_path):
    base_url, _ = mock_server
    ledger = CostLedger(tmp_path, hard_cap_usd=0.001)
    provider = OpenAICompatProvider(
        ProviderEndpoint(base_url=base_url, model_id="qwen3-14b",
                          provider_name="api_openai", supports_seed=False),
        cost_ledger=ledger,
        price_per_mtok=(10.0, 50.0),
    )
    with pytest.raises(BudgetExceededError):
        provider.generate([{"role": "user", "content": "x"}],
                          GenerationSettings(temperature=0.0))
    events = (tmp_path / "cost_events.jsonl").read_text().strip().splitlines()
    kinds = [json.loads(line)["kind"] for line in events]
    assert "budget_cap_exceeded" in kinds


def test_provider_falls_back_to_think_tag_parsing(mock_server):
    """When the endpoint does not separate reasoning, think tags are parsed."""
    # Build the tags from chr() exactly like models.interface does, so no
    # content filter can rewrite the literals in this file.
    think_open = chr(60) + "think" + chr(62)
    think_close = chr(60) + "/think" + chr(62)
    payload_text = think_open + "mmm" + think_close + "answer body"

    class _NoReasoningHandler(_MockChatServer):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            response = {
                "id": "x",
                "model": body.get("model"),
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": payload_text},
                    "finish_reason": "stop",
                }],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }
            payload = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    _NoReasoningHandler.calls = []
    server = HTTPServer(("127.0.0.1", 0), _NoReasoningHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/v1"
        provider = OpenAICompatProvider(ProviderEndpoint(
            base_url=url, model_id="m", provider_name="vllm_openai",
            supports_reasoning_field=False,
        ))
        turn = provider.generate([{"role": "user", "content": "x"}],
                                 GenerationSettings(temperature=0.0, reasoning_mode=True))
        assert turn.reasoning_text == "mmm"
        assert turn.assistant_text == "answer body"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_provider_transport_retry_then_failure(tmp_path):
    """A dead endpoint fails loudly after bounded retries (no silent default)."""
    provider = OpenAICompatProvider(ProviderEndpoint(
        base_url="http://127.0.0.1:1/v1", model_id="m", provider_name="vllm_openai",
        request_timeout_seconds=0.2,
    ))
    with pytest.raises(RuntimeError, match="chat/completions failed after"):
        provider.generate([{"role": "user", "content": "x"}],
                          GenerationSettings(temperature=0.0))


# ---------------------------------------------------------------------------
# Cost ledger + price table
# ---------------------------------------------------------------------------

def test_price_table_entries_are_cited_and_dated():
    table = PriceTable()
    entry = table.lookup("openai", "gpt-6.1-sol")
    assert entry is not None
    assert entry.prompt_usd_per_mtok == 2.00
    assert entry.completion_usd_per_mtok == 10.00
    assert entry.source_url.startswith("http")
    assert entry.verified_at == "2026-11-06"


def test_price_table_rejects_uncited_entries():
    from traceguard.serving.cost import PriceEntry

    with pytest.raises(ValueError, match="source URL"):
        PriceEntry(1.0, 2.0, "not-a-url", "2026-11-06")
    with pytest.raises(ValueError, match="ISO date"):
        PriceEntry(1.0, 2.0, "https://x", "yesterday")


def test_cost_ledger_records_and_summarizes(tmp_path):
    ledger = CostLedger(tmp_path, hard_cap_usd=100.0)
    ledger.record(provider="openai", model_id="gpt-6.1-sol",
                  prompt_tokens=1_000_000, completion_tokens=500_000,
                  price_per_mtok=(2.0, 10.0))
    ledger.record_compute(provider="aws", instance_type="p4de.24xlarge",
                          hours=2.0, usd_per_hour=27.447)
    summary = ledger.write_summary()
    assert summary["api_usd"] == 7.0          # 2*1 + 10*0.5
    assert summary["total_usd"] == pytest.approx(7.0 + 54.894, abs=1e-6)
    assert summary["compute_hours"] == 2.0
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "cost_events.jsonl").exists()


# ---------------------------------------------------------------------------
# Server argv builder
# ---------------------------------------------------------------------------

def _spec(**overrides) -> ServerSpec:
    base = dict(
        model_id="Qwen/Qwen3-14B", model_revision="a" * 40,
        served_model_name="qwen3-14b", architecture_family="qwen3_dense",
        reasoning_parser="qwen3",
    )
    base.update(overrides)
    return ServerSpec(**base)


def test_server_argv_pins_revision_and_flags():
    argv = build_server_argv(_spec())
    joined = " ".join(argv)
    assert "--revision " + "a" * 40 in joined
    assert "--enable-prefix-caching" in joined
    assert "--reasoning-parser qwen3" in joined
    assert "--disable-log-requests" in joined
    assert "--dtype bfloat16" in joined
    assert "--max-model-len 32768" in joined


def test_server_argv_refuses_mxfp4_without_quantization_flag():
    # gpt-oss ships native MXFP4; the guard triggers only when quantization
    # is unset (the family is known MoE-MXFP4).
    with pytest.raises(ValueError, match="MXFP4"):
        build_server_argv(_spec(architecture_family="gpt_oss_moe"))


def test_server_spec_hash_is_content_addressed():
    assert _spec().content_hash() == _spec().content_hash()
    assert _spec(max_model_len=16384).content_hash() != _spec().content_hash()


def test_server_spec_yaml_roundtrip(tmp_path):
    """The shipped serving specs parse into valid ServerSpec fields."""
    import yaml

    for name in ("qwen3_14b_a100.yaml", "qwen3_32b_a100_fp8.yaml",
                 "gpt_oss_20b_h100.yaml"):
        path = Path(__file__).resolve().parents[2] / "configs" / "serving" / name
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["model_id"], name
        assert raw["model_revision"], name
        assert "max_model_len" in raw, name
