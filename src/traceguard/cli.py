"""TRACEGUARD command-line interface.

Commands (docs/06 command contract):
    doctor    - environment self-check: imports, schemas, templates, config
    smoke     - full CPU end-to-end smoke run with FakeModel
    generate  - run episode generation (provider from env/config; GPU or fake)
    monitor   - score stored episodes under all conditions
    analyze   - metrics, paired bootstrap, figures, reports
    verify    - integrity verification of a run directory (hashes, schema)
    export    - package a run for export (checksums + manifest)

Every command writes explicit success/failure results; nothing is silently
swallowed. GPU stages record provider identity; fake providers are allowed
only for smoke stages and are refused by analysis for research stages.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import SCHEMA_VERSION, __version__
from .config import load_config
from .traces.storage import RunStore


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
    """Write checksums + provenance bundle for a run directory."""
    store = RunStore(args.run_dir)
    manifest = store.read_run_manifest()
    store.write_checksums()
    bundle = {
        "run_manifest": manifest,
        "episode_count": len(store.completed_episode_ids()),
    }
    (Path(args.run_dir) / "artifacts" / "export_bundle.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, **bundle}, indent=2))
    return 0


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
    p.add_argument("--stage", default="pilot",
                   choices=["smoke", "gpu_smoke", "pilot", "confirmatory", "online"])
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--provider", default="auto",
                   choices=["auto", "qwen", "fake"])
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("monitor", help="score stored episodes")
    p.add_argument("--config", required=True)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--monitors", nargs="+", default=["rules_v1", "llm_v1"])
    p.add_argument("--conditions", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
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
    p.set_defaults(func=cmd_export)

    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
