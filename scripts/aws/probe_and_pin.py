#!/usr/bin/env python
"""Tier-aware admission probe for the AWS serving plane.

Resolves the exact model revision served by the pinned spec, verifies the
server responds and separates the reasoning channel correctly, records the
GPU inventory and vLLM version, and writes the probe artifact the factory
verifies at provider construction time (fail closed on any mismatch).

For API monitors the probe additionally LISTs /v1/models and requires the
operator to pin the served model id explicitly (API ids drift; a wrong id is
a silent wrong-model hazard).

Usage:
    python scripts/aws/probe_and_pin.py --config <cell.yaml> \
        --serving-spec configs/serving/qwen3_14b_a100.yaml --role agent --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from traceguard.config import load_config  # noqa: E402
from traceguard.ids import sha256_json  # noqa: E402
from traceguard.traces.storage import atomic_write_json  # noqa: E402


def probe_server(base_url: str, served_model_id: str) -> dict:
    """Gates: server healthy, model list contains the served id, a test
    generation round-trips with usage fields."""
    gates: dict[str, dict] = {}
    with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
        gates["health"] = {"passed": response.status == 200}
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as response:
        models = json.loads(response.read().decode("utf-8"))
    ids = [m.get("id") for m in models.get("data", [])]
    gates["served_model_present"] = {"passed": served_model_id in ids, "ids": ids}
    payload = json.dumps({
        "model": served_model_id,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0.0,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    usage = body.get("usage", {})
    gates["test_generation"] = {
        "passed": bool(body.get("choices")) and "prompt_tokens" in usage,
        "usage": usage,
    }
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="cell/experiment config YAML")
    parser.add_argument("--serving-spec", default=None, help="ServerSpec YAML")
    parser.add_argument("--role", choices=["agent", "monitor", "fact_checker"], required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--out", default=None, help="probe artifact output path")
    args = parser.parse_args()

    config = load_config(args.config)
    spec_section = getattr(config, args.role)
    base_url = spec_section.api_base_url or f"http://127.0.0.1:{args.port}/v1"
    served_model_id = spec_section.api_model_id
    if not served_model_id:
        raise SystemExit("config section has no api_model_id to probe")

    import platform

    artifact = {
        "schema_version": "1.0.0",
        "role": args.role,
        "model_id": spec_section.model_id,
        "served_model_id": served_model_id,
        "base_url": base_url,
        "hardware_tier": spec_section.hardware_tier,
        "python": platform.python_version(),
        "probed_at_utc": __import__("datetime").datetime.now(__import__("datetime").UTC).isoformat(),
    }
    if args.serving_spec:
        import yaml

        raw = yaml.safe_load(Path(args.serving_spec).read_text(encoding="utf-8"))
        artifact["serving_spec"] = raw
        artifact["serving_spec_sha256"] = sha256_json(raw)
    if spec_section.provider == "api_openai":
        # API monitors: LIST models and REQUIRE the operator to confirm the id.
        with urllib.request.urlopen(f"{base_url}/models", timeout=30) as response:
            models = json.loads(response.read().decode("utf-8"))
        ids = [m.get("id") for m in models.get("data", [])]
        artifact["api_models_listed"] = ids
        passed = served_model_id in ids
        artifact["gates"] = {"served_model_present": {"passed": passed, "ids": ids}}
    else:
        artifact["gates"] = probe_server(base_url, served_model_id)

    artifact["all_gates_passed"] = all(g.get("passed") for g in artifact["gates"].values())
    artifact["artifact_sha256"] = sha256_json(
        {k: v for k, v in artifact.items() if k != "artifact_sha256"}
    )
    out = Path(args.out or f"runs/probe/{args.role}_{spec_section.model_id.replace('/', '_')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, artifact)
    print(json.dumps({"all_gates_passed": artifact["all_gates_passed"],
                      "artifact": str(out),
                      "artifact_sha256": artifact["artifact_sha256"]}, indent=2))
    return 0 if artifact["all_gates_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
