#!/usr/bin/env python3
"""Local pre-push preflight for TRACEGUARD kernels.

The v6 fleet launch lost five Kaggle rounds to defects that were all locally
checkable (allowlist misses, stale config paths, missing probe artifacts,
wrong artifact namespaces, absent install gates). Per the 2026-10-27 owner
directive ("make sure this doesn't happen again"), this script replays every
kernel's OWN on-Kaggle checks against the local repository BEFORE any push:

1. The pinned commit exists and is what the kernel source says it is
   (parsed from the kernel file — never hand-typed).
2. The revision-consistency gate: generation commit is an ancestor of the
   pin, and every changed file between them is either explicitly allowlisted
   in the kernel source (AST-parsed) or a non-research prefix.
3. Every config / probe-artifact path the kernel will read EXISTS at the
   pinned commit (git show, not the working tree).
4. The layout-robust run-manifest discovery glob matches at least one run
   directory when a sample is provided.

Exit code 0 = kernel is safe to push; any failure prints the exact problem the
kernel would have hit on Kaggle, minutes earlier and without a GPU session.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=check)


def parse_kernel_constants(kernel_path: Path) -> dict:
    """Extract the kernel's run parameters via AST (no execution)."""
    source = kernel_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    constants: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for stmt in ast.walk(node):
                if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                        and isinstance(stmt.targets[0], ast.Name)):
                    name = stmt.targets[0].id
                    if name in {"allowed_changes", "non_research_prefixes"}:
                        try:
                            constants[name] = ast.literal_eval(stmt.value)
                        except ValueError:
                            continue
    for match in re.finditer(
            r"""(?:os\.environ|env)\[["']([A-Z_]+)["']\]\s*=\s*(.+)""", source):
        constants.setdefault("env_bindings", []).append(
            {"var": match.group(1), "value": match.group(2).strip()})
    if "TRACEGUARD_COMMIT" not in constants:
        raise SystemExit(f"{kernel_path}: TRACEGUARD_COMMIT constant not found")
    return constants


def check_ref_exists(revision: str) -> bool:
    return git(["cat-file", "-e", f"{revision}^{{commit}}"], check=False).returncode == 0


def check_path_at_revision(path: str, revision: str) -> bool:
    return git(["cat-file", "-e", f"{revision}:{path}"], check=False).returncode == 0


def preflight(kernel_dir: Path, gen_revision: str | None, run_dir: Path | None) -> int:
    kernel_path = kernel_dir / "kernel.py"
    metadata_path = kernel_dir / "kernel-metadata.json"
    problems: list[str] = []
    notes: list[str] = []

    if not kernel_path.exists():
        print(f"FAIL: no kernel.py in {kernel_dir}")
        return 1
    constants = parse_kernel_constants(kernel_path)
    pin = constants["TRACEGUARD_COMMIT"]
    print(f"kernel: {kernel_dir.name}")
    print(f"  pinned commit: {pin}")

    # ---- metadata sanity (owner id / title-derived slug) --------------------
    if metadata_path.exists():
        import json
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        slug = re.sub(r"[^a-z0-9]+", "-", metadata.get("title", "").lower()).strip("-")
        expected_id = f"{metadata.get('id', '').split('/')[0]}/{slug}"
        if metadata.get("id", "") != expected_id:
            problems.append(
                f"metadata id {metadata.get('id')!r} does not match its title slug "
                f"{expected_id!r} (SOP-002 §9.5: title-resolving ids; violations "
                "caused the multi-round 409 failures)")
        for key in ("enable_gpu", "kernel_sources", "dataset_sources"):
            if key in metadata:
                notes.append(f"metadata.{key} = {metadata[key]}")
    else:
        problems.append("kernel-metadata.json missing")

    # ---- 1. pin exists -------------------------------------------------------
    if not check_ref_exists(pin):
        problems.append(f"pinned commit {pin} does not exist locally — push first")
    else:
        head = git(["rev-parse", "HEAD"]).stdout.strip()
        if pin != head:
            notes.append(f"NOTE: pin {pin[:10]} != HEAD {head[:10]} (ok if intentional)")

    # ---- 2. revision-consistency gate (ancestor + diff allowlist) -----------
    if gen_revision:
        if not check_ref_exists(gen_revision):
            problems.append(f"generation revision {gen_revision} not found locally")
        else:
            anc = git(["merge-base", "--is-ancestor", gen_revision, pin], check=False)
            if anc.returncode != 0:
                problems.append(
                    f"generation commit {gen_revision[:10]} is NOT an ancestor of the "
                    f"monitor pin {pin[:10]} — the kernel will refuse to score")
            diff = git(["diff", "--name-only", gen_revision, pin])
            changed = [line for line in diff.stdout.splitlines() if line.strip()]
            allowed = constants.get("allowed_changes", set())
            prefixes = tuple(
                constants.get("non_research_prefixes", ()))
            disallowed = [line for line in changed
                          if line not in allowed and not line.startswith(prefixes)]
            if disallowed:
                problems.append(
                    f"revision gate will REJECT on Kaggle — research-core files in the "
                    f"gen->pin diff missing from the kernel allowlist: {disallowed}")
            else:
                notes.append(f"revision gate OK ({len(changed)} changed files covered)")

    # ---- 3. config + artifact paths exist at the pin -------------------------
    for key in ("CONFIG", "CELL_CONFIG"):
        config_path = constants.get(key)
        if config_path and not check_path_at_revision(config_path, pin):
            problems.append(f"{key} {config_path!r} does NOT exist at the pinned commit")
    artifact_rel = None
    for binding in constants.get("env_bindings", []):
        if binding["var"].endswith("_PROBE_ARTIFACT"):
            match = re.search(r'["\'](runs/[^"\']+)["\']', binding["value"])
            if match:
                artifact_rel = match.group(1)
                if not check_path_at_revision(artifact_rel, pin):
                    problems.append(
                        f"probe artifact {artifact_rel!r} does NOT exist at the pin "
                        "(path-class failure that cost the fleet a round)")
    if artifact_rel:
        notes.append(f"probe artifact at pin: {artifact_rel}")

    # ---- 4. run-manifest discovery glob against a local sample ---------------
    if run_dir:
        import glob as globmod
        candidates = sorted(
            path for path in globmod.glob(str(run_dir / "**" / "run_manifest.json"),
                                           recursive=True)
            if (Path(path).parent / "episodes").is_dir()
        )
        if not candidates:
            problems.append(
                f"run-manifest discovery glob matched NOTHING under {run_dir} — "
                "the kernel would exit with NO RUN MANIFEST FOUND")
        else:
            notes.append(f"run-dir discovery OK: {candidates[0]}")

    # ---- report ----------------------------------------------------------------
    for note in notes:
        print(f"  note: {note}")
    if problems:
        for problem in problems:
            print(f"  FAIL: {problem}")
        print(f"PREFLIGHT FAIL: {kernel_dir.name} — fix before pushing")
        return 1
    print(f"PREFLIGHT PASS: {kernel_dir.name}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-dir", required=True)
    parser.add_argument("--gen-revision", default=None,
                        help="generation commit the run manifest records")
    parser.add_argument("--run-dir", default=None,
                        help="local sample run directory to test the discovery glob")
    args = parser.parse_args()
    kernel_dir = Path(args.kernel_dir)
    if not kernel_dir.is_absolute():
        kernel_dir = REPO / kernel_dir
    return preflight(kernel_dir, args.gen_revision,
                     Path(args.run_dir) if args.run_dir else None)


if __name__ == "__main__":
    raise SystemExit(main())
