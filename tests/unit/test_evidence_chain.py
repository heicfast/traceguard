"""Evidence-chain regression tests for the v0.6 security-review fixes F1-F5.

Port of the review's tamper matrix (docs/reviews/v6_core_security_review.md
§4) against the fixed build, plus the owned MINOR hardening (F6-F12, M6).

Every tamper test runs on a throwaway byte-copy of the real verified pilot
archive `runs/pre-tests-data/pilot/pilot_v0_run` (the original is NEVER
mutated). "Canonical" copies are brought to the fully-indexed §4 state the
same way the operator flow does: index build, then checksum ledger refresh.

No assertion in this file relies on terminal display: all content checks are
programmatic (byte counts, json.loads, hashlib) per QUALITY_MANUAL NH-8.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from traceguard.ids import monitor_score_id, sha256_json
from traceguard.provenance import build_evidence_index, verify_evidence_index
from traceguard.traces.storage import RunStore, envelope_fingerprint
from traceguard.verification import verify_run

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT_RUN = REPO_ROOT / "runs" / "pre-tests-data" / "pilot" / "pilot_v0_run"

needs_pilot = pytest.mark.skipif(not PILOT_RUN.exists(), reason="real pilot archive not present")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pilot_copy(tmp_path: Path, name: str) -> Path:
    """Byte-copy of the real pilot archive under tmp_path (never mutate the original)."""
    destination = tmp_path / name
    shutil.copytree(PILOT_RUN, destination)
    return destination


def canonicalize(root: Path) -> None:
    """Bring a copy to the fully-indexed §4 state (index build -> ledger refresh)."""
    build_evidence_index(root)
    RunStore(root).write_checksums()


def _score_shard(root: Path, name: str) -> Path:
    return root / "monitor_scores" / name


def _rewrite_jsonl(path: Path, records: list[dict]) -> None:
    text = "".join(
        json.dumps(rec, sort_keys=True, ensure_ascii=True) + "\n" for rec in records
    )
    path.write_text(text, encoding="utf-8")


def _first_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


# ---------------------------------------------------------------------------
# F1 (BLOCKER): export must not re-certify tampered archives
# ---------------------------------------------------------------------------


@needs_pilot
def test_export_refuses_one_byte_episode_edit_and_preserves_evidence(tmp_path):
    """Review row a2/k_ship: a 1-byte schema-valid episode edit must make export
    refuse, leave NO shipped archive, and leave the prior index/ledger (the
    tamper evidence) untouched instead of rebuilding over it."""
    from traceguard.release import build_release

    root = pilot_copy(tmp_path, "k_ship")
    canonicalize(root)
    shard = root / "episodes" / "shard-00001.jsonl"
    data = shard.read_bytes()
    needle = b"propose_patch"
    assert needle in data, "fixture drift: episode shard no longer contains tool calls"
    offset = data.index(needle)
    # surgical 1-byte edit: propose_patch -> qropose_patch (JSON still parses)
    tampered = data[:offset] + b"q" + data[offset + 1:]
    assert sum(1 for a, b in zip(data, tampered, strict=True) if a != b) == 1
    shard.write_bytes(tampered)

    index_digest_before = _sha256_file(root / "artifacts" / "evidence_index.json")
    ledger_digest_before = _sha256_file(root / "artifacts" / "checksums.sha256")
    archive = tmp_path / "out" / "k_ship.tar.gz"

    with pytest.raises(ValueError, match="export refused") as exc_info:
        build_release(root, archive_path=archive)
    message = str(exc_info.value)
    assert "transaction/envelope mismatch" in message, message
    # refused before writing anything: no archive, no laundered metadata
    assert not archive.exists()
    assert _sha256_file(root / "artifacts" / "evidence_index.json") == index_digest_before
    assert _sha256_file(root / "artifacts" / "checksums.sha256") == ledger_digest_before


@needs_pilot
def test_export_refuses_deleted_report_file(tmp_path):
    """Review row c: deleting an indexed file must make export refuse and name
    the file (previously the deletion was silently laundered into the rebuild)."""
    from traceguard.release import build_release

    root = pilot_copy(tmp_path, "c_delfile")
    canonicalize(root)
    deleted = root / "reports" / "fig_scores_holdout.png"
    assert deleted.exists()
    deleted.unlink()

    with pytest.raises(ValueError, match="export refused") as exc_info:
        build_release(root, archive_path=tmp_path / "out" / "c.tar.gz")
    message = str(exc_info.value)
    assert "reports/fig_scores_holdout.png" in message, message
    assert not (tmp_path / "out" / "c.tar.gz").exists()


@needs_pilot
def test_clean_canonical_copy_exports_fine(tmp_path):
    """The gate must not block a genuinely clean run: a canonical copy exports,
    records the prior index/ledger digests as history anchors, and the
    reproducible content fingerprint is stable across rebuilds (F10)."""
    from traceguard.release import build_release

    root = pilot_copy(tmp_path, "clean")
    canonicalize(root)
    assert verify_run(str(root))["ok"]

    archive = tmp_path / "out" / "release.tar.gz"
    first = build_release(root, archive_path=archive)
    assert first["ok"]
    assert first["manifest"]["release_complete"] is True
    assert archive.exists()
    anchors = first["manifest"]["prior_integrity_anchors"]
    assert anchors["prior_evidence_index_sha256"], "rebuild must anchor the prior index"
    assert anchors["prior_checksum_ledger_file_sha256"], "rebuild must anchor the prior ledger"
    assert verify_run(str(root))["ok"], "post-export tree must still verify"

    second = build_release(root)
    assert second["evidence_content_index_sha256"] == first["evidence_content_index_sha256"], (
        "reproducible content fingerprint must not change across identical rebuilds"
    )
    assert (
        second["manifest"]["prior_integrity_anchors"]["prior_evidence_index_sha256"]
        == first["evidence_index_sha256"]
    ), "second rebuild must anchor the index the first export produced"


@needs_pilot
def test_export_refuses_tampered_preexisting_index(tmp_path):
    """Review row d: a tampered evidence index must make export refuse instead
    of silently rebuilding over it (laundering the tamper evidence)."""
    from traceguard.release import build_release

    root = pilot_copy(tmp_path, "d_index")
    canonicalize(root)
    index_path = root / "artifacts" / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entry = next(item for item in index["files"] if item["path"] == "episodes/shard-00001.jsonl")
    original_digest = entry["sha256"]
    raw_index = index_path.read_text(encoding="utf-8")
    assert raw_index.count(original_digest) == 1, "fixture drift: digest not unique in index"
    entry["sha256"] = original_digest[:-1] + ("0" if original_digest[-1] != "0" else "1")
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    index_digest_before = _sha256_file(index_path)
    with pytest.raises(ValueError, match="export refused") as exc_info:
        build_release(root)
    assert "evidence index" in str(exc_info.value)
    assert _sha256_file(index_path) == index_digest_before, "refusal must not overwrite the index"


@needs_pilot
def test_export_allow_incomplete_tolerates_only_missing_metadata(tmp_path):
    """--allow-incomplete (explicit checkpoint mode) tolerates ONLY unbuilt
    integrity metadata; content tampering refuses the export regardless."""
    from traceguard.release import build_release

    # pristine archive has no evidence index yet -> incompleteness, not tampering
    checkpoint = pilot_copy(tmp_path, "checkpoint")
    release = build_release(checkpoint, allow_incomplete=True,
                            archive_path=tmp_path / "out" / "checkpoint.tar.gz")
    assert release["manifest"]["allow_incomplete"] is True
    assert release["manifest"]["release_complete"] is False
    assert release["manifest"]["verification"]["ok"] is False
    assert any(p.startswith("MISSING_INDEX") for p in release["manifest"]["verification"]["problems"])

    # but tampering is never tolerated, even in checkpoint mode
    tampered = pilot_copy(tmp_path, "checkpoint_tampered")
    canonicalize(tampered)
    shard = tampered / "episodes" / "shard-00001.jsonl"
    data = shard.read_bytes()
    shard.write_bytes(data.replace(b"propose_patch", b"qropose_patch", 1))
    with pytest.raises(ValueError, match="export refused"):
        build_release(tampered, allow_incomplete=True)


# ---------------------------------------------------------------------------
# F2: unindexed file additions must be caught (reverse enumeration)
# ---------------------------------------------------------------------------


@needs_pilot
def test_verify_catches_unindexed_added_file(tmp_path):
    """An added file anywhere in the run dir must fail verification by name."""
    root = pilot_copy(tmp_path, "b_addfile")
    canonicalize(root)
    injected = root / "episodes" / "injected-episode-notes.md"
    injected.write_text("planted evidence\n", encoding="utf-8")

    report = verify_run(str(root))
    assert not report["ok"]
    assert "unindexed file present: episodes/injected-episode-notes.md" in report["problems"]
    assert "file not covered by checksum ledger: episodes/injected-episode-notes.md" in report["problems"]


@needs_pilot
def test_verify_catches_injected_log_file(tmp_path):
    """Review row b: an injected logs/generation.log used to pass verify silently."""
    root = pilot_copy(tmp_path, "b_log")
    canonicalize(root)
    (root / "logs" / "generation.log").write_text("[monitoring] injected log line\n", encoding="utf-8")

    report = verify_run(str(root))
    assert not report["ok"]
    assert any("logs/generation.log" in problem for problem in report["problems"])


@needs_pilot
def test_index_check_catches_added_file(tmp_path):
    """index-run --check must reverse-enumerate too (the review ran it green on row b)."""
    root = pilot_copy(tmp_path, "b_index")
    canonicalize(root)
    (root / "reports" / "planted_report.json").write_text("{}", encoding="utf-8")
    result = verify_evidence_index(root)
    assert not result["ok"]
    assert "unindexed file present: reports/planted_report.json" in result["problems"]


# ---------------------------------------------------------------------------
# F3: artifacts/ must not be strippable
# ---------------------------------------------------------------------------


@needs_pilot
def test_verify_requires_artifacts_for_pilot_stage(tmp_path):
    """Review row g: rm -rf artifacts/ used to leave verify fully green."""
    root = pilot_copy(tmp_path, "g_strip")
    canonicalize(root)
    assert verify_run(str(root))["ok"]
    shutil.rmtree(root / "artifacts")

    report = verify_run(str(root))
    assert not report["ok"]
    assert any(p.startswith("MISSING_INDEX") for p in report["problems"]), report["problems"]
    assert any(p.startswith("MISSING_CHECKSUM_LEDGER") for p in report["problems"])


def test_smoke_stage_missing_index_is_warning_only(tmp_path):
    """Smoke runs keep the lenient path: a missing index warns, it does not fail."""
    from traceguard.smoke import run_smoke

    out = tmp_path / "run"
    result = run_smoke(None, str(out))
    assert result["ok"]
    root = out / "traceguard_smoke"
    shutil.rmtree(root / "artifacts")

    report = verify_run(str(root))
    assert report["ok"], report["problems"]
    assert any("evidence index missing" in warning for warning in report["warnings"])


# ---------------------------------------------------------------------------
# F4: score shards must be bound to their condition/monitor filename
# ---------------------------------------------------------------------------


def _swap_shard_contents(root: Path) -> None:
    left = _score_shard(root, "rules_v1-actions-00001.jsonl")
    right = _score_shard(root, "rules_v1-actions_plus_noise-00001.jsonl")
    left_bytes, right_bytes = left.read_bytes(), right.read_bytes()
    assert left_bytes and right_bytes and left_bytes != right_bytes
    left.write_bytes(right_bytes)
    right.write_bytes(left_bytes)


@needs_pilot
def test_verify_catches_swapped_score_shards(tmp_path):
    """Review row f2: swapping two conditions' shards used to ship misattributed
    measurements as a fully verified release; here the attacker even re-indexes
    and re-checksums (laundering) and the shard binding still fails verification."""
    root = pilot_copy(tmp_path, "f2_swap")
    canonicalize(root)
    _swap_shard_contents(root)
    canonicalize(root)  # attacker launders the digest anchors

    report = verify_run(str(root))
    assert not report["ok"]
    assert any("score shard/record condition mismatch" in p for p in report["problems"]), (
        report["problems"]
    )


@needs_pilot
def test_load_scores_refuses_swapped_shard_content(tmp_path):
    """load_scores must hard-fail on misattributed measurements (F4 at load time)."""
    root = pilot_copy(tmp_path, "f2_load")
    _swap_shard_contents(root)

    store = RunStore(root)
    with pytest.raises(ValueError, match="condition .* does not match the shard"):
        store.load_scores("rules_v1", "actions")


@needs_pilot
def test_load_scores_clean_copy_returns_bound_records(tmp_path):
    root = pilot_copy(tmp_path, "f2_clean")
    records = RunStore(root).load_scores("rules_v1", "actions")
    assert records
    assert all(rec["condition"] == "actions" and rec["monitor_name"] == "rules_v1"
               for rec in records)


# ---------------------------------------------------------------------------
# F5: score<->episode binding must be verified (dead hash activated)
# ---------------------------------------------------------------------------


@needs_pilot
def test_verify_catches_episode_binding_tamper(tmp_path):
    """Editing a score record's episode binding breaks the recorded score_id
    derivation — caught even after the attacker re-indexes and re-checksums."""
    root = pilot_copy(tmp_path, "f5_binding")
    canonicalize(root)
    shard = _score_shard(root, "rules_v1-actions-00001.jsonl")
    lines = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    other_episode = json.loads(
        (root / "episodes" / "shard-00002.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )["episode_id"]
    assert lines[0]["episode_id"] != other_episode
    lines[0]["episode_id"] = other_episode  # score_id left stale, as a naive edit would
    _rewrite_jsonl(shard, lines)
    canonicalize(root)  # launder the digest anchors

    report = verify_run(str(root))
    assert not report["ok"]
    assert any("score_id derivation mismatch" in p for p in report["problems"]), report["problems"]


@needs_pilot
def test_verify_catches_prompt_hash_tamper_with_recomputed_score_id(tmp_path):
    """Worst-case attacker: change prompt_hash AND recompute score_id AND
    re-anchor the digests — the frozen monitor prompt constant still catches it."""
    root = pilot_copy(tmp_path, "f5_prompt")
    canonicalize(root)
    shard = _score_shard(root, "rules_v1-actions-00001.jsonl")
    lines = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    forged_prompt_hash = "f" * 64
    lines[0]["prompt_hash"] = forged_prompt_hash
    lines[0]["prompt_template_hash"] = forged_prompt_hash
    lines[0]["score_id"] = monitor_score_id(
        lines[0]["episode_id"], lines[0]["condition"], lines[0]["monitor_name"], forged_prompt_hash
    )
    _rewrite_jsonl(shard, lines)
    canonicalize(root)  # launder the digest anchors

    report = verify_run(str(root))
    assert not report["ok"]
    assert any(
        "prompt_hash not bound to frozen monitor prompt constant" in p for p in report["problems"]
    ), report["problems"]
    assert not any("score_id derivation mismatch" in p for p in report["problems"]), (
        "fixture error: tamper must be fully self-consistent except for the frozen constant"
    )


@needs_pilot
def test_verify_catches_rendered_prompt_hash_tamper(tmp_path):
    root = pilot_copy(tmp_path, "f5_rendered")
    canonicalize(root)
    shard = _score_shard(root, "rules_v1-actions-00001.jsonl")
    lines = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    lines[0]["rendered_prompt_hash"] = "e" * 64
    _rewrite_jsonl(shard, lines)
    canonicalize(root)

    report = verify_run(str(root))
    assert not report["ok"]
    assert any("rendered prompt hash mismatch" in p for p in report["problems"])


def test_smoke_records_carry_envelope_fingerprint_and_verify_green(tmp_path):
    """Creation-time wiring (F5): the smoke pipeline records an
    envelope_fingerprint per score, and the clean run verifies green."""
    from traceguard.smoke import run_smoke

    out = tmp_path / "run"
    assert run_smoke(None, str(out))["ok"]
    root = out / "traceguard_smoke"

    report = verify_run(str(root))
    assert report["ok"], report["problems"]

    store = RunStore(root)
    episodes_by_id = {env["episode_id"]: env for env in store.load_all_episodes()}
    records = store.load_scores("rules_v1", "actions")
    assert records
    for rec in records:
        assert rec["envelope_fingerprint"] == envelope_fingerprint(episodes_by_id[rec["episode_id"]])


def test_envelope_fingerprint_catches_laundered_episode_binding(tmp_path):
    """Worst-case attacker on NEW-format records: relabel a score to a different
    episode, recompute score_id, and rewrite the ledger — the creation-time
    envelope fingerprint still binds the measurement to its true episode."""
    from traceguard.smoke import run_smoke

    out = tmp_path / "run"
    assert run_smoke(None, str(out))["ok"]
    root = out / "traceguard_smoke"

    shard = _score_shard(root, "rules_v1-actions-00001.jsonl")
    lines = [json.loads(line) for line in shard.read_text(encoding="utf-8").splitlines() if line]
    episodes = RunStore(root).load_all_episodes()
    other = next(env["episode_id"] for env in episodes if env["episode_id"] != lines[0]["episode_id"])
    # the attacker first removes the genuine record for the target episode so
    # the forged one does not collide with it...
    lines = [rec for rec in lines if rec["episode_id"] != other]
    # ...then relabels a measurement of episode A as a measurement of episode B,
    # recomputing the score_id so the derivation check stays quiet.
    lines[0]["episode_id"] = other
    lines[0]["score_id"] = monitor_score_id(
        other, lines[0]["condition"], lines[0]["monitor_name"], lines[0]["prompt_hash"]
    )
    # the envelope_fingerprint is left pointing at the TRUE (relabelled-away)
    # episode — exactly the dead-weight binding the attacker cannot see.
    _rewrite_jsonl(shard, lines)
    RunStore(root).write_checksums()  # launder the only digest anchor this stage has

    report = verify_run(str(root))
    assert not report["ok"]
    assert any(
        "score/episode envelope fingerprint mismatch" in p for p in report["problems"]
    ), report["problems"]
    assert not any("score_id derivation mismatch" in p for p in report["problems"]), (
        "fixture error: the forged score_id must be fully self-consistent"
    )


# ---------------------------------------------------------------------------
# Owned MINOR hardening (security review F6-F12, developer review M6)
# ---------------------------------------------------------------------------


@needs_pilot
def test_malformed_ledger_line_is_structured_problem(tmp_path):
    """F7: a malformed checksum ledger line used to crash verify with a raw
    ValueError; it must be a structured, named problem instead."""
    root = pilot_copy(tmp_path, "i_ledger")
    canonicalize(root)
    ledger = root / "artifacts" / "checksums.sha256"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write("garbage-no-separator-line\n")

    report = verify_run(str(root))  # must return, not raise
    assert not report["ok"]
    assert any(p.startswith("malformed checksum ledger line") for p in report["problems"])


@needs_pilot
def test_problem_lists_report_true_total(tmp_path):
    """F12: more problems than the display cap must still report the true count."""
    root = pilot_copy(tmp_path, "i_many")
    canonicalize(root)
    ledger = root / "artifacts" / "checksums.sha256"
    with ledger.open("a", encoding="utf-8") as handle:
        for index in range(60):
            handle.write(f"garbage-line-{index}-without-separator\n")

    report = verify_run(str(root))
    assert report["problem_count"] == 60
    assert report["problems"][-1].startswith("... 60 problems total (50 shown)")


def test_verify_does_not_create_run_directory(tmp_path):
    """F9: verify is a read-only audit and must not mkdir the audited tree."""
    missing = tmp_path / "does-not-exist"
    report = verify_run(str(missing))
    assert not report["ok"]
    assert not missing.exists()


@needs_pilot
def test_symlinks_are_refused_not_followed(tmp_path):
    """F6: an in-tree symlink used to be indexed by build and rejected by check;
    now build refuses, and both verifiers name the symlink as a problem."""
    root = pilot_copy(tmp_path, "h_symlink")
    canonicalize(root)
    (root / "notes.txt").symlink_to("/etc/hostname")

    with pytest.raises(ValueError, match="symlink"):
        build_evidence_index(root)
    index_report = verify_evidence_index(root)
    assert not index_report["ok"]
    assert "symlink in run directory: notes.txt" in index_report["problems"]
    verify_report = verify_run(str(root))
    assert not verify_report["ok"]
    assert any("symlink in run directory: notes.txt" in p for p in verify_report["problems"])


@needs_pilot
def test_index_reorder_detected(tmp_path):
    """F10 row e: a reordered files array with recomputed internal digests used
    to pass index-run --check; canonical order is now enforced."""
    root = pilot_copy(tmp_path, "e_reorder")
    canonicalize(root)
    index_path = root / "artifacts" / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop("index_sha256")
    index["files"] = list(reversed(index["files"]))
    index["content_index_sha256"] = sha256_json(index["files"])
    index["index_sha256"] = sha256_json(index)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = verify_evidence_index(root)
    assert not result["ok"]
    assert "evidence index files not in canonical sorted order" in result["problems"]


@needs_pilot
def test_index_metadata_bound_to_run_manifest(tmp_path):
    """F11 row j: a forged index run_id with recomputed self-digest used to pass
    index-run --check; the index metadata is now bound to run_manifest.json."""
    root = pilot_copy(tmp_path, "j_forge")
    canonicalize(root)
    index_path = root / "artifacts" / "evidence_index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index.pop("index_sha256")
    index["run_id"] = "totally-different-run"
    index["index_sha256"] = sha256_json(index)
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    result = verify_evidence_index(root)
    assert not result["ok"]
    assert "index metadata not bound to run manifest: run_id" in result["problems"]


@needs_pilot
def test_transient_lock_files_are_ignored(tmp_path):
    """M6/F2 allow-list: a workflow lock is operational state, not evidence —
    it must be neither indexed nor required by the ledger."""
    root = pilot_copy(tmp_path, "m6_lock")
    canonicalize(root)
    before = verify_evidence_index(root)
    assert before["ok"]
    (root / "workflow.lock").write_text("12345\n", encoding="utf-8")
    canonicalize(root)  # operator refresh over the tree that now contains a lock

    index = json.loads((root / "artifacts" / "evidence_index.json").read_text(encoding="utf-8"))
    assert "workflow.lock" not in json.dumps(index)
    assert index["file_count"] == before["files_checked"]
    after = verify_evidence_index(root)
    assert after["ok"], after["problems"]
    report = verify_run(str(root))
    assert report["ok"], report["problems"]


@needs_pilot
def test_decoy_checksums_file_is_covered_by_ledger(tmp_path):
    """F8: the ledger exclusion must be path-based, so a decoy
    logs/checksums.sha256 cannot escape checksum coverage."""
    root = pilot_copy(tmp_path, "f8_decoy")
    canonicalize(root)
    decoy = root / "logs" / "checksums.sha256"
    decoy.write_text("decoy ledger content\n", encoding="utf-8")
    canonicalize(root)  # operator refresh: index + ledger rebuilt over the tree

    ledger_text = (root / "artifacts" / "checksums.sha256").read_text(encoding="utf-8")
    assert "  logs/checksums.sha256" in ledger_text, "decoy must be ledger-covered"
    report = verify_run(str(root))
    assert report["ok"], report["problems"]
    index = json.loads((root / "artifacts" / "evidence_index.json").read_text(encoding="utf-8"))
    assert any(item["path"] == "logs/checksums.sha256" for item in index["files"])


@needs_pilot
def test_ndjson_media_type_in_index(tmp_path):
    """F15: .jsonl shards are indexed as application/x-ndjson, not octet-stream."""
    root = pilot_copy(tmp_path, "f15_media")
    index = build_evidence_index(root)
    episode_entry = next(item for item in index["files"]
                         if item["path"] == "episodes/shard-00001.jsonl")
    assert episode_entry["media_type"] == "application/x-ndjson"
