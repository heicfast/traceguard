"""TRACEGUARD command-line interface.

Commands (docs/06 command contract):
    doctor    - environment self-check: imports, schemas, templates, config
    smoke     - full CPU end-to-end smoke run with FakeModel
    generate  - run episode generation (provider from env/config; GPU or fake)
    monitor   - score stored episodes under all conditions
    analyze   - metrics, paired bootstrap, figures, reports
    verify    - integrity verification of a run directory (hashes, schema)
    export    - package a run for export (checksums + manifest)
    probe-model / matrix-* / prepare-tuning - multi-model study controls
    study-plan - explicit sample-size, GPU-hour, and storage assumptions
    workflow   - dry-run/run/resume/status with a hash-chained journal
    index-run  - build or verify the role-labeled evidence inventory

Every command writes explicit success/failure results; nothing is silently
swallowed. GPU stages record provider identity; fake providers are allowed
only for smoke stages and are refused by analysis for research stages.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSION, __version__
from .config import load_config


def cmd_doctor(args: argparse.Namespace) -> int:
    """Environment self-check; never mutates state."""
    report: dict[str, object] = {
        "traceguard_version": __version__,
        "schema_version": SCHEMA_VERSION,
        "python": sys.version.split()[0],
        "cwd": str(Path.cwd()),
    }
    ok = True
    try:
        from .tasks.specs import FAMILY_TEMPLATES, TEMPLATES, validate_template_family

        report["families"] = {f: len(t) for f, t in FAMILY_TEMPLATES.items()}
        report["templates_total"] = len(TEMPLATES)
        validation = validate_template_family()
        bad = {k: v for k, v in validation.items() if not v.startswith("OK")}
        report["template_validation"] = validation
        report["template_validation_ok"] = not bad
        ok = ok and not bad
    except Exception as exc:
        report["template_validation_error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    try:
        from .tasks.split import FAMILY_SPLIT, families_for_role, manifest_hash

        report["split_manifest"] = FAMILY_SPLIT
        report["split_manifest_sha256"] = manifest_hash()
        for role in ("development", "validation", "holdout", "shift"):
            if not families_for_role(role):
                report[f"split_empty_{role}"] = True
                ok = False
    except Exception as exc:
        report["split_error"] = f"{type(exc).__name__}: {exc}"
        ok = False
    if args.config:
        try:
            cfg = load_config(args.config)
            report["config_ok"] = True
            report["config_sha256"] = cfg.content_hash()
            report["config"] = cfg.to_payload()
        except Exception as exc:
            report["config_ok"] = False
            report["config_error"] = f"{type(exc).__name__}: {exc}"
            ok = False
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if ok else 1


def cmd_smoke(args: argparse.Namespace) -> int:
    """CPU end-to-end smoke run with the deterministic FakeModel."""
    from .smoke import run_smoke

    result = run_smoke(config_path=args.config, out_dir=args.out)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate episodes for a plan shard. Uses GPU provider when available,
    FakeModel only for smoke stages (enforced)."""
    from .generation_runtime import run_generation

    result = run_generation(
        config_path=args.config,
        run_dir=args.run_dir,
        stage=args.stage,
        shard=args.shard,
        shard_count=args.shard_count,
        provider_name=args.provider,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def cmd_monitor(args: argparse.Namespace) -> int:
    """Score stored episodes under all configured conditions."""
    from .monitor_runtime import run_monitoring

    result = run_monitoring(
        config_path=args.config,
        run_dir=args.run_dir,
        monitor_names=args.monitors,
        conditions=args.conditions or None,
        limit=args.limit,
        shard=args.shard,
        shard_count=args.shard_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Paired analysis with clustered bootstrap + figures + report."""
    from .analysis_runtime import run_analysis

    result = run_analysis(
        config_path=args.config,
        run_dir=args.run_dir,
        out_dir=args.out,
        monitor_name=args.monitor,
    )
    print(json.dumps(result.get("summary", result), indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify run-directory integrity: schema, dedupe, hashes, labels."""
    from .verification import verify_run

    result = verify_run(args.run_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok") else 1


def cmd_export(args: argparse.Namespace) -> int:
    """Validate and package a self-verifying research release."""
    from .release import build_release

    result = build_release(
        args.run_dir,
        archive_path=args.archive,
        allow_incomplete=args.allow_incomplete,
    )
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "manifest_path": result["manifest_path"],
                "archive_path": result["archive_path"],
                "archive_sha256": result["archive_sha256"],
                "evidence_index_sha256": result.get("evidence_index_sha256"),
                "release_complete": result["manifest"]["release_complete"],
                "counts": result["manifest"]["counts"],
                "validation_errors": result["manifest"]["validation_errors"],
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    return 0 if result["ok"] or args.allow_incomplete else 1


def cmd_matrix_plan(args: argparse.Namespace) -> int:
    """Validate a frozen multi-model matrix and emit its deterministic job plan."""
    from .model_matrix import build_schedule, load_model_matrix

    matrix = load_model_matrix(args.matrix)
    result = build_schedule(matrix)
    if args.out:
        from .traces.storage import atomic_write_json

        atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_matrix_materialize(args: argparse.Namespace) -> int:
    """Write immutable per-job configs and a deduplicated schedule."""
    from .model_matrix import load_model_matrix, materialize_matrix

    result = materialize_matrix(load_model_matrix(args.matrix), args.out)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_prepare_tuning(args: argparse.Namespace) -> int:
    """Build a development-only monitor SFT corpus with a hashed manifest."""
    from .tuning import prepare_tuning_corpus

    config = load_config(args.config)
    result = prepare_tuning_corpus(
        config=config,
        run_dir=args.run_dir,
        output_dir=args.out,
        conditions=args.conditions,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_probe_model(args: argparse.Namespace) -> int:
    """Run and persist the hardware/template/reasoning admission gates."""
    from .model_probe import run_model_probe

    result = run_model_probe(args.config, args.role, args.out)
    print(
        json.dumps(
            {
                "all_gates_passed": result["all_gates_passed"],
                "artifact_sha256": result["artifact_sha256"],
                "output": args.out,
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["all_gates_passed"] else 1


def cmd_study_plan(args: argparse.Namespace) -> int:
    """Estimate sample, compute, and storage requirements with explicit assumptions."""
    from dataclasses import fields

    import yaml

    from .study_design import PlanningAssumptions, build_study_plan
    from .traces.storage import atomic_write_json

    values: dict[str, Any] = {}
    if args.assumptions:
        raw = yaml.safe_load(Path(args.assumptions).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("planning assumptions file must be a mapping")
        allowed = {field.name for field in fields(PlanningAssumptions)}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown planning assumptions: {sorted(unknown)}")
        values = raw
    result = build_study_plan(
        load_config(args.config),
        PlanningAssumptions(**values),
        matrix_path=args.matrix,
        stage=args.stage,
    )
    if args.out:
        atomic_write_json(args.out, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    """Plan, run, resume, or inspect a hash-journaled research workflow.

    Operator-facing failures (tamper, lock, validation) surface as structured
    failure-envelope JSON with a clear message — tracebacks go to the work
    dir's log, never to stdout (developer review F9). Exit codes: 0 success,
    1 generic failure, 3 statistical-gate failure (errors.GATE_FAILURE_EXIT_CODE),
    130 operator interrupt.
    """
    from .errors import (
        GATE_FAILURE_EXIT_CODE,
        ErrorCode,
        FailureEnvelope,
        classify_exception,
    )
    from .workflow import (
        JournalTamperError,
        LockHeldError,
        WorkflowBaselineError,
        run_workflow,
        workflow_status,
    )

    def emit(exc: BaseException, code: ErrorCode, exit_code: int) -> int:
        envelope = FailureEnvelope(
            code=code,
            message=str(exc),
            stage=args.workflow_action,
            exception_type=type(exc).__name__,
            retryable=False,
            context={"workflow": args.workflow, "work_dir": args.work_dir},
        )
        # Tracebacks are evidence, not operator output: append them to the
        # work dir log when it exists, never print them (F9).
        if Path(args.work_dir).is_dir():
            with contextlib.suppress(OSError):
                log_dir = Path(args.work_dir) / "logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                with open(
                    log_dir / "workflow_cli_errors.log", "a", encoding="utf-8"
                ) as handle:
                    handle.write(
                        f"--- {datetime.now(UTC).isoformat()} "
                        f"action={args.workflow_action} ---\n{traceback.format_exc()}\n"
                    )
        print(json.dumps(envelope.to_payload(), indent=2, sort_keys=True, default=str))
        return exit_code

    try:
        if args.workflow_action == "status":
            result = workflow_status(args.workflow, args.work_dir)
        else:
            result = run_workflow(
                args.workflow,
                args.work_dir,
                dry_run=args.workflow_action == "plan",
                rebaseline=args.rebaseline,
            )
    except KeyboardInterrupt:
        return emit(
            KeyboardInterrupt("orchestrator interrupted (SIGINT)"),
            ErrorCode.USER_CANCELLED,
            130,
        )
    except JournalTamperError as exc:
        return emit(exc, ErrorCode.INPUT_INTEGRITY, 1)
    except (LockHeldError, WorkflowBaselineError, ValueError) as exc:
        return emit(exc, ErrorCode.CONFIGURATION, 1)
    except Exception as exc:
        return emit(exc, classify_exception(exc, stage=args.workflow_action).code, 1)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if result.get("ok", True):
        return 0
    if result.get("failure_code") == ErrorCode.STATISTICAL_GATE.value:
        return GATE_FAILURE_EXIT_CODE
    return 1


def cmd_index_run(args: argparse.Namespace) -> int:
    """Build or verify the independently checkable evidence inventory."""
    from .provenance import build_evidence_index, verify_evidence_index

    if args.check:
        result = verify_evidence_index(args.run_dir)
        display = result
    else:
        result = build_evidence_index(args.run_dir)
        display = {
            "ok": True,
            "path": str(Path(args.run_dir) / "artifacts" / "evidence_index.json"),
            "files": result["file_count"],
            "total_bytes": result["total_bytes"],
            "role_counts": result["role_counts"],
            "index_sha256": result["index_sha256"],
        }
    print(json.dumps(display, indent=2, sort_keys=True, default=str))
    return 0 if result.get("ok", True) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="traceguard")
    parser.add_argument("--version", action="version", version=f"traceguard {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="environment self-check")
    p.add_argument("--config", default=None)
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("smoke", help="CPU smoke run with FakeModel")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default="runs/smoke")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("generate", help="generate episodes for a plan shard")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--stage",
        default="pilot",
        choices=["smoke", "gpu_smoke", "pilot", "confirmatory", "online"],
    )
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--provider", default="auto", choices=["auto", "qwen", "fake"])
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("monitor", help="score stored episodes")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--monitors", nargs="+", default=["rules_v1", "llm_v1"])
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.set_defaults(func=cmd_monitor)

    p = sub.add_parser("analyze", help="paired analysis and figures")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--monitor", default="llm_qwen_hf")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("verify", help="verify a run directory")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("export", help="checksums and export bundle")
    p.add_argument("--run-dir", required=True)
    p.add_argument(
        "--archive", default=None, help="optional .tar.gz path outside the run directory"
    )
    p.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="export an explicitly marked operational checkpoint",
    )
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("matrix-plan", help="validate and schedule a multi-model study")
    p.add_argument("--matrix", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_matrix_plan)

    p = sub.add_parser("matrix-materialize", help="write frozen configs for matrix jobs")
    p.add_argument("--matrix", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_matrix_materialize)

    p = sub.add_parser("prepare-tuning", help="prepare a development-only secondary QLoRA corpus")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--conditions", nargs="+", default=["actions", "actions_plus_cot"])
    p.set_defaults(func=cmd_prepare_tuning)

    p = sub.add_parser("probe-model", help="run a T4 model admission probe")
    p.add_argument("--config", required=True)
    p.add_argument("--role", choices=["agent", "monitor"], required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_probe_model)

    p = sub.add_parser("study-plan", help="estimate sample, GPU, and storage requirements")
    p.add_argument("--config", required=True)
    p.add_argument("--matrix", default=None)
    p.add_argument("--assumptions", default=None)
    p.add_argument(
        "--stage",
        choices=["smoke", "gpu_smoke", "pilot", "confirmatory", "online"],
        default="pilot",
    )
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_study_plan)

    p = sub.add_parser("workflow", help="plan/run/resume/status a research workflow")
    p.add_argument("workflow_action", choices=["plan", "run", "resume", "status"])
    p.add_argument("--workflow", required=True)
    p.add_argument("--work-dir", required=True)
    p.add_argument(
        "--rebaseline",
        action="store_true",
        help=(
            "explicitly re-baseline the journal after a workflow edit "
            "(journals a rebaselined event; required to resume an edited workflow)"
        ),
    )
    p.set_defaults(func=cmd_workflow)

    p = sub.add_parser("index-run", help="build or verify a run evidence index")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=cmd_index_run)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
