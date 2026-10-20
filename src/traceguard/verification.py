"""Run-directory verification: independent integrity audit.

Checks (docs/02 §8 required tests, operationalized):
  * run manifest present, parseable, config hash consistent with episodes
  * every episode envelope schema-valid; episode ids unique; ledger complete
  * OUTCOME RE-GRADE: recompute ground-truth labels from the privileged
    state/transition evidence and compare with stored labels
  * cross-file transaction/commit integrity (v2 runs)
  * monitor scores: unique score ids; shard/record monitor+condition binding;
    score_id derivation; prompt_hash bound to the frozen monitor prompt
    constants; rendered-prompt hash; score/episode envelope fingerprint
  * checksum ledger: structured malformed-line problems + reverse coverage
  * evidence index (when present): full index verification including reverse
    enumeration; MISSING_INDEX/MISSING_CHECKSUM_LEDGER are stage-severity
    problems for pilot/confirmatory/online and warnings for smoke
  * symlinks anywhere in the run directory are problems (never followed)

verify is a READ-ONLY audit: it never creates or modifies the audited tree
(security review F9). A run that fails verification cannot be presented as
confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .ids import monitor_score_id, sha256_text
from .provenance import TRANSIENT_SUFFIXES, verify_evidence_index
from .sandbox.grader import GRADER_VERSION, grade_episode
from .sandbox.invariants import Invariant
from .sandbox.state import SandboxState
from .traces.schema import load_episodes, validate_monitor_score, validate_privileged
from .traces.storage import RunStore, envelope_fingerprint

_STRICT_STAGES = {"pilot", "confirmatory", "online"}
_FUSION_CONDITIONS = {"actions_plus_cot_late_fusion", "actions_plus_cot_fact_checked"}
_PROBLEM_DISPLAY_CAP = 50


def _frozen_prompt_hashes() -> dict[str, frozenset[str]]:
    """Frozen per-hash prompt constants, by monitor class.

    Security review F5: a score record's prompt_hash must be bound to a frozen
    constant so a worst-case attacker who forges the hash AND recomputes the
    score_id AND re-anchors the digests is still caught. The legacy pre-nonce
    LLM prompt (phase-2 pre-test archives, superseded 2026-10-19 by the
    nonce-hardened prompt) is retained with its provenance so pre-test
    archives keep verifying under the current code.
    """
    from .monitors.interface import prompt_hash
    from .monitors.llm import MONITOR_PROMPT_TEMPLATE
    from .monitors.structured import (
        ACTION_EXTRACTOR_PROMPT,
        ADJUDICATOR_PROMPT,
        FACT_CHECK_PROMPT,
        REASONING_EXTRACTOR_PROMPT,
    )

    current_llm = prompt_hash(MONITOR_PROMPT_TEMPLATE)
    legacy_llm = "309ef4aa865be30e1a01f89c7b31e303ee0766fa7ed227306e3d6b22d8ff3564"
    return {
        "rules": frozenset({prompt_hash("rules_v1: policy-conformance over tool history")}),
        "keywords": frozenset({prompt_hash("keywords_v1: suspicious-phrase features")}),
        "llm": frozenset({current_llm, legacy_llm}),
        "fusion": frozenset({
            prompt_hash("\n".join([
                ACTION_EXTRACTOR_PROMPT, REASONING_EXTRACTOR_PROMPT,
                "no_fact_checker", ADJUDICATOR_PROMPT,
            ])),
            prompt_hash("\n".join([
                ACTION_EXTRACTOR_PROMPT, REASONING_EXTRACTOR_PROMPT,
                FACT_CHECK_PROMPT, ADJUDICATOR_PROMPT,
            ])),
        }),
    }


def _expected_prompt_set(record: dict[str, Any], registry: dict[str, frozenset[str]],
                         ) -> frozenset[str] | None:
    monitor = str(record.get("monitor_name", ""))
    condition = str(record.get("condition", ""))
    if condition in _FUSION_CONDITIONS or monitor.startswith("llm_late_fusion") \
            or monitor.startswith("llm_fact_checked"):
        return registry["fusion"]
    if monitor == "rules_v1":
        return registry["rules"]
    if monitor == "keywords_v1":
        return registry["keywords"]
    if monitor.startswith("llm"):
        return registry["llm"]
    return None


def _parse_score_shard_name(name: str) -> tuple[str, str] | None:
    """Split '<monitor>-<condition>-<shard>.jsonl' (both dash-free fields)."""
    if not name.endswith(".jsonl"):
        return None
    parts = name[: -len(".jsonl")].split("-")
    if len(parts) < 3 or not parts[-1].isdigit():
        return None
    return parts[0], parts[1]


def verify_run(run_dir: str) -> dict[str, Any]:
    root = Path(run_dir)
    if not root.is_dir():
        # Read-only audit: never create the audited tree (review F9).
        return {"ok": False, "problems": ["run directory does not exist"], "warnings": [],
                "problem_count": 1}
    store = RunStore(run_dir)
    problems: list[str] = []
    warnings: list[str] = []

    manifest_path = store.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "problems": ["run_manifest.json missing"], "warnings": [],
                "problem_count": 1}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage = str(manifest.get("experiment_stage", manifest.get("stage", "unknown")))

    episodes: list[dict[str, Any]] = []
    for path in sorted(store.episode_dir.glob("*.jsonl")):
        try:
            episodes.extend(load_episodes(path, validate=True))
        except Exception as exc:
            problems.append(f"schema violation in {path.name}: {exc}")

    ids = [e["episode_id"] for e in episodes]
    if len(ids) != len(set(ids)):
        problems.append("duplicate episode ids present")
    ledger = store.completed_episode_ids()
    missing_in_ledger = set(ids) - ledger
    if missing_in_ledger:
        problems.append(f"{len(missing_in_ledger)} episodes not in completion ledger")
    extra_in_ledger = ledger - set(ids)
    if extra_in_ledger:
        problems.append(f"{len(extra_in_ledger)} ledger entries without episodes")

    config_sha = manifest.get("config_sha256")
    if config_sha:
        mismatched = [
            e["episode_id"] for e in episodes if e["provenance"].get("config_sha256") != config_sha
        ]
        if mismatched:
            problems.append(f"{len(mismatched)} episodes with mismatched config hash")

    # -- outcome re-grade from privileged evidence -----------------------------
    privileged = store.load_all_privileged()
    episodes_by_id = {e["episode_id"]: e for e in episodes}
    regraded_mismatches = 0
    checked = 0
    for env in episodes:
        priv = privileged.get(env["episode_id"])
        if priv is None:
            problems.append(f"privileged record missing for {env['episode_id']}")
            continue
        if priv.get("schema_version") == "2.0.0":
            try:
                validate_privileged(priv)
            except Exception as exc:
                problems.append(f"privileged schema violation {env['episode_id']}: {exc}")
                continue
            for audit_event in priv["events_audit"]:
                if audit_event.get("kind") != "model_turn_audit":
                    continue
                content = audit_event["content"]
                from .ids import sha256_json

                hash_checks = {
                    "raw_output_utf8_sha256": sha256_text(content["raw_output"]),
                    "raw_output_with_special_tokens_utf8_sha256": sha256_text(
                        content["raw_output_with_special_tokens"]
                    ),
                    "rendered_prompt_utf8_sha256": sha256_text(content["rendered_prompt"]),
                    "input_messages_sha256": sha256_json(content["input_messages"]),
                    "prompt_token_ids_sha256": sha256_json(content["prompt_token_ids"]),
                    "generated_token_ids_sha256": sha256_json(content["generated_token_ids"]),
                }
                for field, expected in hash_checks.items():
                    if content.get(field) != expected:
                        problems.append(f"raw evidence hash mismatch {env['episode_id']}:{field}")
        try:
            final_state = SandboxState.from_payload(priv["final_state"])
            initial_state = SandboxState.from_payload(priv["initial_state"])
            invariants = [Invariant.model_validate(inv) for inv in priv["invariants"]]
            outcome = grade_episode(
                transitions=priv["transitions"],
                initial_state=initial_state,
                final_state=final_state,
                invariants=invariants,
                main_goal_path=priv["main_goal_path"],
                main_goal_expected=priv["main_goal_expected"],
                invalid_action_count=env["outcomes"].get("invalid_action_count", 0),
            )
            checked += 1
            stored = env["outcomes"]
            if (
                outcome.attack_attempted != stored["attack_attempted"]
                or outcome.attack_succeeded != stored["attack_succeeded"]
                or outcome.main_task_succeeded != stored["main_task_succeeded"]
            ):
                regraded_mismatches += 1
                problems.append(f"outcome re-grade mismatch: {env['episode_id']}")
        except Exception as exc:
            problems.append(f"re-grade error {env['episode_id']}: {type(exc).__name__}: {exc}")
    if checked and regraded_mismatches == 0:
        warnings.append(f"outcome re-grade passed for {checked} episodes (grader {GRADER_VERSION})")

    # -- cross-file transaction integrity (v2 runs) ------------------------------
    tx_paths = sorted(store.transaction_dir.glob("*.json"))
    if tx_paths:
        for tx_path in tx_paths:
            transaction = json.loads(tx_path.read_text(encoding="utf-8"))
            ep_id = transaction.get("episode_id")
            commit_path = store.commit_dir / tx_path.name
            if not commit_path.exists():
                problems.append(f"uncommitted episode transaction: {ep_id}")
                continue
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            from .ids import sha256_json

            if commit.get("transaction_sha256") != sha256_json(transaction):
                problems.append(f"transaction/commit hash mismatch: {ep_id}")
            projected_envelope: dict[str, Any] | None = episodes_by_id.get(ep_id)
            priv = privileged.get(ep_id)
            if projected_envelope is None or sha256_json(projected_envelope) != transaction.get(
                "envelope_sha256"
            ):
                problems.append(f"transaction/envelope mismatch: {ep_id}")
            if priv is None or sha256_json(priv) != transaction.get("privileged_sha256"):
                problems.append(f"transaction/privileged mismatch: {ep_id}")

    # -- monitor scores -----------------------------------------------------------
    prompt_registry = _frozen_prompt_hashes()
    for path in sorted(store.scores_dir.glob("*.jsonl")):
        shard_pair = _parse_score_shard_name(path.name)
        score_ids: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("record_version") == "2.0.0":
                try:
                    validate_monitor_score(rec)
                except Exception as exc:
                    problems.append(f"monitor score schema violation in {path.name}: {exc}")
            sid = rec.get("score_id")
            if not sid:
                problems.append(f"missing score_id in {path.name}")
                continue
            if sid in score_ids:
                problems.append(f"duplicate score_id {sid} in {path.name}")
            score_ids.add(sid)
            if not rec.get("view_hash"):
                problems.append(f"missing view_hash in {path.name}")
            # Shard binding (review F4): the record must belong to the shard it
            # physically sits in.
            if shard_pair is not None:
                shard_monitor, shard_condition = shard_pair
                if (rec.get("condition") != shard_condition
                        or rec.get("monitor_name") != shard_monitor):
                    problems.append(
                        f"score shard/record condition mismatch in {path.name}: "
                        f"{sid} claims {rec.get('monitor_name')}/{rec.get('condition')}")
            # score_id derivation: binds episode+condition+monitor+prompt_hash.
            try:
                derived = monitor_score_id(
                    rec.get("episode_id", ""), rec.get("condition", ""),
                    rec.get("monitor_name", ""), rec.get("prompt_hash", ""))
                if derived != sid:
                    problems.append(f"score_id derivation mismatch: {sid}")
            except Exception as exc:
                problems.append(f"score_id derivation error {sid}: {exc}")
            # Frozen prompt constants (review F5).
            expected_prompts = _expected_prompt_set(rec, prompt_registry)
            if expected_prompts is None:
                problems.append(
                    f"no frozen prompt constant for monitor {rec.get('monitor_name')} ({sid})")
            elif rec.get("prompt_hash") not in expected_prompts:
                problems.append(
                    f"prompt_hash not bound to frozen monitor prompt constant: {sid}")
            # Rendered prompt integrity.
            if sha256_text(rec.get("rendered_prompt", "")) != rec.get("rendered_prompt_hash"):
                problems.append(f"rendered prompt hash mismatch: {sid}")
            # Creation-time envelope fingerprint: binds the measurement to its
            # true episode even against a fully self-consistent relabeling
            # (review F5; absent only in legacy pre-fingerprint records).
            fingerprint = rec.get("envelope_fingerprint")
            fingerprint_env: dict[str, Any] | None = episodes_by_id.get(rec.get("episode_id"))
            if (fingerprint is not None and fingerprint_env is not None
                    and fingerprint != envelope_fingerprint(fingerprint_env)):
                problems.append(f"score/episode envelope fingerprint mismatch: {sid}")

    # -- symlinks: never followed, always reported --------------------------------
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            problems.append(f"symlink in run directory: {path.relative_to(root)}")

    # -- checksum ledger ------------------------------------------------------------
    checksum_path = store.artifact_dir / "checksums.sha256"
    covered: set[str] = set()
    if checksum_path.exists():
        bad = 0
        for line in checksum_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if "  " not in line:
                problems.append(
                    f"malformed checksum ledger line: {line[:60]!r}")
                continue
            digest, rel = line.split("  ", 1)
            rel = rel.strip()
            covered.add(rel)
            target = store.run_dir / rel
            if not target.exists():
                bad += 1
                continue
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if current != digest:
                bad += 1
        if bad:
            problems.append(f"{bad} checksum mismatches")
        # Reverse coverage (review F2): every real file must be ledger-covered;
        # exclusion is path-based so decoys cannot escape (review F8).
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(root).as_posix()
            if rel == "artifacts/checksums.sha256" or rel.endswith(TRANSIENT_SUFFIXES):
                continue
            if rel not in covered:
                problems.append(f"file not covered by checksum ledger: {rel}")
    else:
        message = "MISSING_CHECKSUM_LEDGER: artifacts/checksums.sha256 absent"
        if stage in _STRICT_STAGES:
            problems.append(message)
        else:
            warnings.append("checksum ledger missing")

    # -- evidence index ----------------------------------------------------------------
    evidence_index_path = store.artifact_dir / "evidence_index.json"
    if evidence_index_path.exists():
        evidence_result = verify_evidence_index(store.run_dir)
        problems.extend(evidence_result["problems"])
        warnings.append(f"evidence index checked {evidence_result.get('files_checked', 0)} files")
    else:
        message = "MISSING_INDEX: artifacts/evidence_index.json absent"
        if stage in _STRICT_STAGES:
            problems.append(message)
        else:
            warnings.append("evidence index missing")

    problem_count = len(problems)
    metadata_prefixes = ("MISSING_INDEX", "MISSING_CHECKSUM_LEDGER")
    metadata_only = all(problem.startswith(metadata_prefixes) for problem in problems) \
        if problems else False
    displayed = problems[:_PROBLEM_DISPLAY_CAP]
    if problem_count > _PROBLEM_DISPLAY_CAP:
        displayed = displayed + [
            f"... {problem_count} problems total ({_PROBLEM_DISPLAY_CAP} shown)"
        ]
    return {
        "ok": not problems,
        "episodes": len(episodes),
        "regrade_checked": checked,
        "problems": displayed,
        "warnings": warnings,
        "problem_count": problem_count,
        "metadata_only": metadata_only,
        "run_id": manifest.get("run_id"),
        "experiment_stage": stage,
    }


__all__ = ["verify_run"]
