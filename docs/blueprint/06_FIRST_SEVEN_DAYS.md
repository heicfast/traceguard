# 06 — First Seven Days: Execution Checklist

**Goal:** by Day 7, have a CPU-tested simulator, a pinned GPU preflight notebook, a model able to propose one parsed tool call with an authentic generated trace, and a documented reproduction of a prior result or metric. Not a research paper yet.

## Day 1 — Found the project and scope

- Create a **new private development Git repository**; do not publish unreviewed generated attack traces automatically.
- Copy these documents into `docs/`; adopt or simplify the proposed `src/traceguard/` structure.
- Read Korbak et al. §1–5, AI Control (Greenblatt et al.), and SLEIGHT-Bench's methods.
- Fill the prior-art matrix: study, observability, measurement timing, baseline, open gap.
- Write a 150-word abstract and `docs/assumptions.md` distinguishing open-weight generated CoT from proprietary private reasoning.
- **Deliverable:** `docs/research_question.md`, research log, source references, and new repository.

## Day 2 — Define tasks and deterministic rules

- Write immutable `TaskSpec` and pure `SandboxEngine.step()`.
- Support `inspect_state`, `propose_patch`, `list_objects`, `submit_task`.
- Create two task families with seeded variations.
- Add hand-designed allowed/forbidden/invalid action fixtures.
- **Deliverable:** 20+ CPU unit tests, all passing; no model needed.

## Day 3 — Evidence schemas and security boundary

- Implement canonical event/episode schema and SHA-256 checksums.
- Store hidden attack instruction and private labels outside monitor observations.
- Implement view A and C; test pre-action cutoff and path/label leakage.
- Build `FakeModel` test driver for deterministic transcript generation.
- **Deliverable:** a complete fake episode with safe monitoring views and replay.

## Day 4 — Kaggle compatibility preflight

- Create Kaggle GPU notebook, enable accelerator, record actual GPU/quota.
- Run CUDA math/precision preflight and record environment versions.
- Load `Qwen/Qwen3-4B` with an appropriate tested dtype or quantization path.
- Verify chat templates and a basic generated `<think>` segment, logging raw tokens.
- **Deliverable:** manifest of GPU/model compatibility and peak VRAM.

## Day 5 — Real agent step

- Connect model to typed simulated tool calls with strict parser.
- Generate one full honest episode and one hidden-side-objective attack episode.
- Score outcomes through deterministic state predicates.
- Inspect raw trace and record parser/truncation problems.
- **Deliverable:** 2 end-to-end genuine model episodes, with no claim about reliability.

## Day 6 — Baseline monitors

- Implement exact-rule baseline and fixed-prompt LLM risk score.
- Compile A/C plus length-matched-noise D.
- Run tests verifying same episode ID and privileged-field isolation across all conditions.
- **Deliverable:** paired monitor scores with full prompt hashes and validity metadata.

## Day 7 — Pilot gate review

- Execute 8-episode GPU smoke test; account for every attempted action and failure.
- Estimate memory/token throughput, approximate Kaggle capacity for a 48-episode pilot.
- Inspect whether attacks actually attempt a violation and whether rule monitor trivializes the task.
- Write `worklog/week01.md`: what failed, confirmed constraints, scientific risk, next changes.
- **Deliverable:** end-to-end smoke with pass/fail report, not cherry-picked successes.

## Standard command contract to implement

```bash
# Once the proposed CLI is implemented:
python -m pip install -e '.[dev,research]'
python -m pytest -q
python -m traceguard.cli doctor
python -m traceguard.cli smoke --config configs/smoke.yaml
python -m traceguard.cli generate --config configs/pilot.yaml --shard 0 --count 8
python -m traceguard.cli monitor --run-dir runs/<run-id> --conditions A,C,D
python -m traceguard.cli analyze --run-dir runs/<run-id> --output reports/smoke
```

This is a target command interface; it does **not** work yet. The design package is not a codebase. Implement commands incrementally and update this runbook if their actual names differ.

## Weekly log template

```text
Period:
Code commit:
Model/tokenizer commit:
Runtime hardware and quota:
Hypothesis / primary metric:
Changes made:
Tests run and results:
Completed episodes / attempted attacks / succeeded attacks:
Invalid traces / timeouts / OOMs:
Unexpected observations:
Threats to validity:
Next gate and action:
```
