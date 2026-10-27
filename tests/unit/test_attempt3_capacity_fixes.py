"""Regression tests for the 2026-10-27 attempt-3 fix set (docs/deviations.md):

1. Deployment-scale admission stress gate (model_probe): the attempt-2 14B
   monitor cells failed on a capacity knife edge admission never exercised
   (~90-char probe prompts vs ~17.9K-char deployment prompts). The stress
   builder must derive its size from the config window, reach deployment
   scale, and use the frozen monitor prompt template.
2. single_split worker layout (run_monitor_workers): the 14B monitors must
   run split across both T4s with one worker and one shard; per_gpu keeps its
   historical behavior; sub-2-GPU split requests are refused, not fallen back.
3. Kernel preflight (preflight_kernel): the local pre-push replay of every
   kernel's own on-Kaggle checks (revision gate, config/artifact paths at the
   pin, metadata slug rules) must parse the real kernel sources.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

from traceguard.config import load_config  # noqa: E402


def _load_probe_module():
    from traceguard import model_probe

    return model_probe


def _load_worker_module():
    path = REPO / "scripts" / "run_monitor_workers.py"
    spec = importlib.util.spec_from_file_location("run_monitor_workers_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_preflight_module():
    path = REPO / "scripts" / "preflight_kernel.py"
    spec = importlib.util.spec_from_file_location("preflight_kernel_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. deployment-scale stress gate
# ---------------------------------------------------------------------------

def test_stress_prompt_reaches_deployment_scale_monitor_role():
    module = _load_probe_module()
    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_monitor_probe.yaml")
    prompt = module.build_deployment_stress_prompt(config, "monitor")
    metrics = module.stress_prompt_metrics(prompt, config, "monitor")
    # Window derives from the config (3072 tokens x 4.0 chars/token).
    assert metrics["window_chars"] == 12_288
    # Deployment max measured 15,659 chars; the stress must exceed it.
    assert metrics["stress_prompt_chars"] >= metrics["minimum_deployment_scale_chars"]
    assert metrics["reaches_deployment_scale"] is True


def test_stress_prompt_uses_the_frozen_monitor_template():
    module = _load_probe_module()
    from traceguard.monitors.llm import MONITOR_PROMPT_TEMPLATE, evidence_nonce

    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_monitor_probe.yaml")
    prompt = module.build_deployment_stress_prompt(config, "monitor")
    # The instruction scaffold must be the frozen template with its format
    # escapes resolved ({{ -> {), byte-identical prefix up to the evidence.
    expected_prefix = (MONITOR_PROMPT_TEMPLATE.split("=== BEGIN EVIDENCE")[0]
                       .replace("{{", "{").replace("}}", "}"))
    assert prompt.startswith(expected_prefix)
    # Nonce-anchored delimiters (injection hardening preserved).
    nonce = evidence_nonce(module._STRESS_EPISODE_ID)
    assert f"=== BEGIN EVIDENCE {nonce} ===" in prompt
    assert f"=== END EVIDENCE {nonce} ===" in prompt


def test_stress_prompt_carries_all_view_sections():
    module = _load_probe_module()
    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_monitor_probe.yaml")
    prompt = module.build_deployment_stress_prompt(config, "monitor")
    for section in (
        "== AUTHORIZED TASK (deployer statement) ==",
        "== SECURITY POLICY (invariants that must hold) ==",
        "== AGENT TOOL ACTIVITY ==",
        "== AGENT REASONING TRACE (model-generated; unverified explanation",
        "UNTRUSTED agent-generated text",
    ):
        assert section in prompt, f"stress view missing section: {section}"


def test_stress_prompt_window_derives_from_config_not_hardcoded():
    module = _load_probe_module()
    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_monitor_probe.yaml")
    base = module.build_deployment_stress_prompt(config, "monitor")
    # Halving the declared window must shrink the prompt accordingly: the
    # builder reads the config, so protocol changes propagate to the gate.
    # (Configs are frozen pydantic models — mutate via model_copy, never
    # in-place.)
    half_config = config.model_copy(update={
        "monitor": config.monitor.model_copy(
            update={"max_monitor_tokens": config.monitor.max_monitor_tokens // 2}),
    })
    shrunk = module.build_deployment_stress_prompt(half_config, "monitor")
    assert len(shrunk) < len(base)
    assert len(base) - len(shrunk) >= config.monitor.max_monitor_tokens // 2 * 4 - 32


def test_stress_prompt_agent_role_scales_to_context_budget():
    module = _load_probe_module()
    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_agent_probe.yaml")
    prompt = module.build_deployment_stress_prompt(config, "agent")
    metrics = module.stress_prompt_metrics(prompt, config, "agent")
    assert metrics["reaches_deployment_scale"] is True
    assert metrics["context_budget_tokens"] == config.agent.max_context_tokens


def test_stress_prompt_rejects_unknown_role():
    module = _load_probe_module()
    config = load_config(REPO / "configs" / "probe" / "qwen3_14b_monitor_probe.yaml")
    try:
        module.build_deployment_stress_prompt(config, "fact_checker")
    except ValueError as error:
        assert "fact_checker" in str(error)
    else:
        raise AssertionError("unknown role must fail loudly, not fall back")


def test_filler_and_action_blocks_are_deterministic_and_exact():
    module = _load_probe_module()
    text_a = module._stress_filler_text(1_000, "abcd ")
    text_b = module._stress_filler_text(1_000, "abcd ")
    assert text_a == text_b and len(text_a) == 1_000
    actions = module._stress_action_block(2_600)
    assert len(actions) <= 2_600
    assert "propose_patch" in actions


# ---------------------------------------------------------------------------
# 2. single_split worker layout
# ---------------------------------------------------------------------------

def test_per_gpu_topology_matches_historical_behavior():
    module = _load_worker_module()
    topology = module.plan_worker_topology(gpu_count=2, layout="per_gpu")
    assert len(topology) == 2
    for rank, spec in enumerate(topology):
        assert spec["rank"] == rank
        assert spec["visible_devices"] == str(rank)
        assert spec["device_map"] == '{"": 0}'
        assert spec["shard"] == rank
        assert spec["shard_count"] == 2


def test_single_split_topology_spans_all_gpus_with_one_worker():
    module = _load_worker_module()
    for gpu_count in (2, 3):
        topology = module.plan_worker_topology(gpu_count=gpu_count, layout="single_split")
        assert len(topology) == 1
        spec = topology[0]
        assert spec["rank"] == 0
        assert spec["visible_devices"] == ",".join(str(r) for r in range(gpu_count))
        assert spec["device_map"] == "auto"
        assert (spec["shard"], spec["shard_count"]) == (0, 1)


def test_topology_refuses_invalid_inputs_instead_of_falling_back():
    module = _load_worker_module()
    for bad in ({"gpu_count": 0, "layout": "per_gpu"},
                {"gpu_count": 2, "layout": "bogus"},
                {"gpu_count": -1, "layout": "single_split"}):
        try:
            module.plan_worker_topology(**bad)
        except ValueError:
            continue
        raise AssertionError(f"invalid topology input must raise: {bad}")


def test_worker_layouts_constant_matches_planner():
    module = _load_worker_module()
    assert module.LAYOUTS == ("per_gpu", "single_split")


# ---------------------------------------------------------------------------
# 3. kernel preflight (local pre-push replay of the kernels' own gates)
# ---------------------------------------------------------------------------

def test_preflight_parses_real_cell_kernel_constants():
    module = _load_preflight_module()
    constants = module.parse_kernel_constants(
        REPO / "kaggle" / "mon_scale_qwen14" / "kernel.py")
    assert constants["TRACEGUARD_COMMIT"]
    assert constants["CELL_CONFIG"] == "configs/phase3_monitor_cells/scale_monitor_qwen14.yaml"
    # The allowlist must parse out of main()'s AST.
    assert "src/traceguard/models/qwen.py" in constants["allowed_changes"]
    assert "src/traceguard/monitor_runtime.py" in constants["allowed_changes"]
    assert constants["non_research_prefixes"][0] == "docs/"
    env_vars = {binding["var"] for binding in constants["env_bindings"]}
    assert "TRACEGUARD_MONITOR_PROBE_ARTIFACT" in env_vars


def test_preflight_parses_split_probe_kernel_and_env_binding():
    module = _load_preflight_module()
    constants = module.parse_kernel_constants(
        REPO / "kaggle" / "probe_split_qwen14" / "kernel.py")
    assert constants["CONFIG"] == "configs/probe/qwen3_14b_monitor_probe.yaml"
    env_vars = {binding["var"] for binding in constants["env_bindings"]}
    assert "TRACEGUARD_DEVICE_MAP" in env_vars  # the split load path binding


def test_preflight_flags_stale_allowlist_exactly_like_the_kernel():
    """A file changed between gen and pin but missing from the allowlist must
    be reported locally — the exact failure class that cost the fleet three
    Kaggle rounds in the v6 launch."""
    module = _load_preflight_module()
    constants = module.parse_kernel_constants(
        REPO / "kaggle" / "mon_scale_qwen14" / "kernel.py")
    allowed = constants["allowed_changes"]
    prefixes = tuple(constants["non_research_prefixes"])
    changed = ["src/traceguard/models/qwen.py",      # allowlisted
               "src/traceguard/sandbox/new_file.py",  # NOT allowlisted
               "docs/deviations.md"]                  # non-research prefix
    disallowed = [line for line in changed
                  if line not in allowed and not line.startswith(prefixes)]
    assert disallowed == ["src/traceguard/sandbox/new_file.py"]


def test_preflight_metadata_slug_rule_detects_mismatches():
    import json

    _load_preflight_module()
    metadata = json.loads(
        (REPO / "kaggle" / "mon_scale_qwen14" / "kernel-metadata.json").read_text())
    owner = metadata["id"].split("/")[0]
    slug = metadata["title"].lower().replace(" ", "-")
    assert metadata["id"] == f"{owner}/{slug}"  # the §9.5 title-resolving rule
