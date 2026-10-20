"""Run-directory verification: independent integrity audit.

Checks (docs/02 §8 required tests, operationalized):
  * run manifest present, parseable, config hash consistent with episodes
  * every episode envelope schema-valid; episode ids unique; ledger complete
  * OUTCOME RE-GRADE: recompute ground-truth labels from the privileged
    state/transition evidence and compare with stored labels — catches any
    drift between grader versions and recorded outcomes
  * monitor score records: unique score ids; view hashes present
  * checksum file, when present, matches current content
A run that fails verification cannot be presented as confirmatory evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .provenance import verify_evidence_index
from .sandbox.grader import GRADER_VERSION, grade_episode
from .sandbox.invariants import Invariant
from .sandbox.state import SandboxState
from .traces.schema import load_episodes, validate_monitor_score, validate_privileged
from .traces.storage import RunStore


def verify_run(run_dir: str) -> dict[str, Any]:
    store = RunStore(run_dir)
    problems: list[str] = []
    warnings: list[str] = []

    manifest_path = store.run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {"ok": False, "problems": ["run_manifest.json missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

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
                from .ids import sha256_json, sha256_text

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
            projected_envelope: dict[str, Any] | None = next(
                (item for item in episodes if item["episode_id"] == ep_id), None
            )
            priv = privileged.get(ep_id)
            if projected_envelope is None or sha256_json(projected_envelope) != transaction.get(
                "envelope_sha256"
            ):
                problems.append(f"transaction/envelope mismatch: {ep_id}")
            if priv is None or sha256_json(priv) != transaction.get("privileged_sha256"):
                problems.append(f"transaction/privileged mismatch: {ep_id}")

    # -- monitor scores -----------------------------------------------------------
    for path in sorted(store.scores_dir.glob("*.jsonl")):
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

    # -- checksums -------------------------------------------------------------------
    checksum_path = store.artifact_dir / "checksums.sha256"
    if checksum_path.exists():
        lines = [ln for ln in checksum_path.read_text(encoding="utf-8").splitlines() if ln]
        bad = 0
        for line in lines:
            digest, rel = line.split("  ", 1)
            target = store.run_dir / rel
            if not target.exists():
                bad += 1
                continue
            current = hashlib.sha256(target.read_bytes()).hexdigest()
            if current != digest:
                bad += 1
        if bad:
            problems.append(f"{bad} checksum mismatches")

    evidence_index_path = store.artifact_dir / "evidence_index.json"
    if evidence_index_path.exists():
        evidence_result = verify_evidence_index(store.run_dir)
        problems.extend(evidence_result["problems"])
        warnings.append(f"evidence index checked {evidence_result.get('files_checked', 0)} files")

    return {
        "ok": not problems,
        "episodes": len(episodes),
        "regrade_checked": checked,
        "problems": problems[:50],
        "warnings": warnings,
        "run_id": manifest.get("run_id"),
    }


__all__ = ["verify_run"]
