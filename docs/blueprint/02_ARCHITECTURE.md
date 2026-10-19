# 02 — System Architecture and Interfaces

## 1. Architecture principles

1. **Research system, not a product:** prioritize accurate measurement, a small trusted computing base, and deterministic replay over flashy UI.
2. **In-process, deterministic simulation:** no real filesystem mutations, shell execution, arbitrary code execution, Docker dependency, or network access in the first environment.
3. **Separation of powers:** agent generation, privileged evaluator, observable monitor evidence, and statistics are separate modules with explicit typed interfaces.
4. **Single-source experimental configuration:** no notebook-local magic constants; manifest governs every run.
5. **Independent outcomes:** ground truth comes from checking state transitions against invariant predicates, never from an LLM judge.
6. **Frozen data boundaries:** attack secrets and ground truth labels are physically excluded from serialized monitor views.
7. **Interrupt-resumable:** every completed episode is durably written and deduplicated by a stable experiment/episode key.
8. **CPU testability:** all except generation adapters can be tested offline without accelerators.

## 2. Logical pipeline

```text
                      +-----------------------+
                      | versioned task specs  |
                      | splits / seeds / goals|
                      +-----------+-----------+
                                  |
                                  v
                   +--------------+---------------+
                   | deterministic sandbox engine |
                   | state, actions, invariants   |
                   +--------------+---------------+
                                  ^
                                  | typed tool calls
                                  |
    +----------------------+      |       +---------------------+
    | model runtime adapter+------+       | hidden attack policy|
    | open-weight Qwen3    |<-------------+ or honest policy    |
    | think + tool parsing |              +---------------------+
    +----------+-----------+
               |
               | append-only evidence (raw tokens, tool calls, outputs)
               v
    +----------------------+    +-------------------------------+
    | episode trace store  +--->| monitor view compiler        |
    | privileged metadata  |    | strips secrets / future data |
    +----------+-----------+    +-----------+-------------------+
               |                            |
               v                            v
    +----------------------+    +-------------------------------+
    | ground truth scorer  |    | monitoring variants A...E    |
    | deterministic states |    | score + structured evidence  |
    +----------+-----------+    +---------------+---------------+
               |                                |
               +---------------+----------------+
                               v
                  +----------------------------+
                  | statistical analysis       |
                  | paired+clustered inference |
                  | data integrity / figures   |
                  +--------------+-------------+
                                 |
                                 v
                    paper + reproducibility
```

## 3. Package structure to implement

```text
traceguard/
├── README.md
├── pyproject.toml
├── uv.lock                  # if adopting uv; freeze exact environment
├── configs/
│   ├── smoke.yaml
│   ├── pilot.yaml
│   ├── confirmatory.yaml
│   └── monitor_prompts/
├── src/traceguard/
│   ├── config.py            # pydantic validation; immutable config
│   ├── ids.py               # stable episode IDs, SHA-256 evidence hashes
│   ├── tasks/
│   │   ├── specs.py         # TaskSpec, protected invariants, families
│   │   ├── generator.py     # seeded template instance generation
│   │   └── split.py         # family-scoped split manifest
│   ├── sandbox/
│   │   ├── state.py         # pure typed state
│   │   ├── actions.py       # strict allowed actions, no arbitrary Python
│   │   ├── engine.py        # transition(state, action) -> new state, output
│   │   └── grader.py        # exact violation/completion predicates
│   ├── models/
│   │   ├── interface.py     # inference provider contract
│   │   ├── qwen.py          # Qwen chat/tool/trace format adapter
│   │   └── fake.py          # deterministic fake outputs for unit tests
│   ├── agents/
│   │   ├── policy.py        # honest/attack composition, hidden instructions
│   │   └── runner.py        # generation, action loop, token budgets
│   ├── traces/
│   │   ├── schema.py        # typed messages and trusted/privileged fields
│   │   ├── storage.py       # append only + atomic writes
│   │   └── views.py         # evidence compiler and sanitization
│   ├── monitors/
│   │   ├── interface.py     # score(request) -> MonitorScore
│   │   ├── rules.py         # deterministic known-path baselines
│   │   ├── llm.py           # fixed prompt + parser + invalid output handling
│   │   └── conditions.py    # A/B/C/D/E evidence operators
│   ├── protocols/
│   │   └── pre_action.py    # phase-two intervention-before-tool execution
│   ├── analyses/
│   │   ├── pair.py          # exact episode pairing and censor accounting
│   │   ├── metrics.py       # FPR/TPR, AUROC, AUPRC, effort, latency
│   │   ├── uncertainty.py   # stratified/cluster bootstrap
│   │   └── plots.py         # headless figures and tables
│   └── cli.py               # smoke, generate, replay, monitor, analyze
├── tests/
│   ├── unit/
│   ├── properties/
│   ├── integration/
│   └── fixtures/
├── notebooks/
│   ├── 00_gpu_preflight.ipynb
│   ├── 01_generate.ipynb
│   ├── 02_monitors.ipynb
│   └── 03_analyze.ipynb
├── data/                    # committed manifests/small reviewed samples
├── runs/                    # gitignored generated samples/checkpoints
├── docs/
├── paper/
│   └── manuscript.md
└── scripts/
    ├── smoke_cpu.sh
    ├── verify_run.py
    └── export_run.py
```

No requirement to implement all modules up front. Implement in vertical slices: state transition → fake agent → trace → rule monitor → end-to-end test, then attach the real model.

## 4. Strict interface contracts

### `TaskSpec`

Fields: `task_id`, `family_id`, `template_revision`, `user_goal`, `initial_state`, `available_actions`, `protected_invariants`, `main_goal_predicate`, `attack_goal_predicate`, `seed`, and reproducible `task_hash`. Serialize definitions separately from monitor-allowed fields. Protected invariants may be represented by data/config accessible to the agent, but *their private evaluation result* must never be exposed.

### `SandboxEngine`

`step(state: SandboxState, action: TypedAction) -> TransitionResult`

Result: `new_state`, `tool_response`, `action_accepted`, `protected_violation`, `authorized_goal_reached`, `transition_hash`. State engine uses pure functions or deep copies. Refuse unknown operations, paths, types, and schema versions. Same state + action + engine version must always produce same result.

### `AgentProvider`

`generate(messages, tools, generation_config) -> ModelTurn` with fields `reasoning_text` (possibly absent), `assistant_visible_text`, `proposed_tool_call` (typed or invalid), `raw_token_ids` (if available), `stop_reason`, `token_usage`, `model_revision`, `elapsed_ms`, and parser diagnostics. Do *not* reconstruct CoT from final answers or request an explanation after the fact. Record `trace_available` boolean and `reasoning_parse_status` for each turn.

### `EpisodeRunner`

Use a stateful turn loop with max turns, per-turn token limit, total token budget, tool error budget, and typed transitions. Stop on success, timeout, refusal, invalid tool calls beyond threshold, or an explicit completion. Keep raw model outputs separate from interpreted text to allow auditing adapter errors.

### `ViewCompiler`

`compile(episode, condition, cutoff_turn) -> MonitorObservation`

Maintain an allowlist, not a blocklist. Start from authorized task and observable tool history; attach only requested extra fields. For pre-action view, exclude outcome for pending action and every later event. Require an automated `assert_view_is_clean()` test to ensure fields `attack_goal`, `policy_mode`, `attack_succeeded`, `private_criteria`, and `grader_reason` never enter monitor text. Also guard against accidental leakage through filenames, hashes displayed as class labels, or serial IDs.

### `Monitor`

`score(observation) -> MonitorScore` with `suspicion ∈ [0,1]`, `model_revision`, `prompt_hash`, `input_token_count`, `output_token_count`, `elapsed_ms`, `valid`, and optional `quoted_evidence_ids`. Invalid/missing scores are *not silently dropped*; preregister how scored/abstained/retried.

### `Analysis`

`analyze(paired_scores, outcomes, manifest) -> VersionedReport`. Strictly enforce exactly one primary score per condition per episode, report missingness, fail if task split contamination or duplicate episode keys exist. Verify evidence hashes and match frozen monitor prompt IDs.

## 5. Data and evidence model

Maintain three distinct storage spaces:

1. **Privileged episode record:** task seed, policy mode, hidden objective, full state snapshots, exact invariant scores, and an append-only event log. Local/restricted for experiment integrity.
2. **Public/monitor observation:** sanitized projection containing only authorized task, tool calls/results, and condition-specific trace segments. This must not contain privileged labels.
3. **Evaluation records:** monitor input hash, prompt hash, model identity, raw and parsed score, and linkage to an episode ID.

For publication, release only carefully reviewed synthetic evidence, licensed appropriately. Avoid publishing auxiliary secrets, accidental real-world data, or executable harmful payloads.

### Suggested event schema

- `event_id`, `episode_id`, `step_index`, `event_type` (`model_turn`, `tool_proposal`, `tool_result`, `episode_end`).
- `public_content`, `privileged_content_ref`, `reasoning_content` (may be null), `timestamp`, `model_revision`.
- `state_hash_before`, `state_hash_after`, `tool_result_hash`, `visibility_cutoff`.

All records carry `schema_version`. Canonical JSON serialization for hashing must fix ordering and delimiters. Use SHA-256 for accidental corruption checks and deterministic identification, not as confidentiality protection.

## 6. State machine and online monitoring correctness

```text
INIT → AGENT_GENERATE → (PENDING_ACTION | FINAL)
PENDING_ACTION → [optional PRE_ACTION_MONITOR] → (ALLOW | BLOCK | DEFER)
ALLOW → SANDBOX_TRANSITION → APPEND_TOOL_RESULT → AGENT_GENERATE
BLOCK → APPEND_BLOCK_NOTICE → AGENT_GENERATE / END
FINAL → GRADE_INDEPENDENTLY → EPISODE_COMPLETE
ANY → ERROR/TIMEOUT (explicit, accounted for)
```

For RQ1 offline, record unmonitored trajectories once and replay them to every monitor condition. For RQ4 online, blocking changes subsequent agent behavior; therefore replaying an already-fixed continuation would be invalid. Execute **fresh online agent rollouts** for each pre-action policy, compare randomized comparable task instances, and record any fork/state dependencies. Online A-vs-C can pair seeds and tasks but not assume identical post-intervention trajectories.

## 7. Adapter strategy: Inspect AI and ControlArena

Core simulator remains framework-independent. When CPU invariants and GPU inference work, create an Inspect AI task adapter for sample generation and scoring where its interfaces simplify logging, and a ControlArena setting/micro-protocol for external validation. Existing ControlArena tutorials assume Docker for some settings; do not make those settings phase-one dependencies on Kaggle. Adapter acceptance condition: same canonical task/state/seed must produce semantically identical deterministic grading through both native runner and wrapper. Be explicit about limits of equivalence when the external framework changes prompts or tool behavior.

## 8. Reliability and tests

Required tests:

- A protected invariant cannot be violated without ground-truth scorer detecting it.
- Benign edits do not trigger a protected-state violation.
- Replay is deterministic for fixed state/action sequence.
- Invalid and malformed action JSON is rejected with typed errors and logged.
- Monitor observations never contain hidden objective/labels.
- Pre-action cutoff excludes results of the current action and all future events.
- Reasoning extraction handles absent/malformed `think` tags, parser versions and cutoffs without inventing text.
- Paired analysis detects missing/duplicate conditions and grouped split leakage.
- Threshold selection never reads final holdout labels.
- Checkpoint resumption does not duplicate episodes.
- Statistical result matches independently computed tiny hand-checkable fixture.
- GPU code is optional for CPU unit and integration tests.

CI gates: static types, formatting, lint, unit tests, CPU integration smoke, schema checks, manifest validation. Kaggle GPU integration is a **recorded reproducible run**, not a CI requirement without hosted GPU availability.

## 9. Runtime and dependency choices

- Python 3.11+; use version-pinned compatible PyTorch, Transformers, Accelerate, Pydantic, datasets/Parquet as needed, NumPy/SciPy/scikit-learn.
- Optional quantization via bitsandbytes after a hardware compatibility preflight; don't assume any given Kaggle GPU+Torch image works with any wheel.
- Pytest; Ruff; Mypy or Pyright; JSON schema validation; a fixed dependency lock (uv or pip-tools) plus a captured Kaggle environment manifest.
- Docker is optional on local compute *after* initial benchmark; not necessary for the trusted simulation.
- Do not add a database or web frontend in milestone one. JSONL/Parquet plus manifest suffices.

## 10. Quality bar / architectural acceptance

Every experimental headline number must be traceable from figure → analysis config → monitor score IDs → observation view hashes → episode IDs → immutable event/state evidence → task specification and code/model version. If any link is missing, the release is not considered reproducible.
