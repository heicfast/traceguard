# AWS setup and operations guide (v0.7.0)

From zero to verified evidence on AWS GPU instances, for the TRACEGUARD
research program. Budget discipline is part of the protocol: every run
carries a cost ledger, and the hard cap fails closed.

**Cost basis (verified 2026-11-06, us-east-1):**

| Instance | GPUs | On-demand | Spot (typical) | Fits |
|---|---|---|---|---|
| `p4de.24xlarge` | 8x A100-80GB | $27.447/hr | ~$13.98–22.46/hr | Qwen3-14B bf16; Qwen3-32B FP8 (tp=2); GLM-4.5-Air-FP8 (tp=8) |
| `p5.48xlarge` | 8x H100-80GB | premium tier | — | gpt-oss-20b MXFP4; FA3-class kernels |

A $3,500 credit buys ~127 on-demand p4de hours or ~160–250 spot hours.
Monitor-scale cells (128 episodes x 4 conditions, ~6K-token prompts) take
single-digit hours on a p4de with vLLM — plan for generous margin, probe
first, scale after measuring.

## 1. Account, credits, IAM

1. **Credits.** Apply to *AWS Cloud Credit for Research*
   (https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research;
   student awards up to $5,000) or *AWS Activate* for startup contexts.
   Credits arrive as promotional credit in the billing console — confirm the
   balance and expiry before launching anything.
2. **IAM (least privilege).** Create a dedicated user/group with:
   `ec2:*` (scoped by the tags below where possible), `s3:PutObject/GetObject/ListBucket`
   on your evidence bucket, `cloudwatch:GetMetricData` for GPU health,
   and billing-alarms read. Never use root. Never put long-lived keys in
   the repository (use env vars / `~/.aws` only).
3. **Budget alarms** (belt and braces beyond the ledger):
   Billing → Budgets → create a monthly budget at 80% and 100% of the
   grant with email alerts.

## 2. Instance lifecycle

```bash
# Launch (EBS gp3 2.5 TB for weights + evidence, cluster placement for EFA):
bash scripts/aws/launch_instance.sh p4de.24xlarge my-key
# or spot (interruptible; the workflow journal + WAL transactions resume):
bash scripts/aws/launch_instance.sh p4de.24xlarge my-key spot

ssh -A ubuntu@<public-dns>
```

The user-data script formats the instance-store NVMe as `/mnt/ebs` (XFS),
mounts it, creates `/mnt/ebs/hf-cache` and `/mnt/ebs/runs`, and dumps the
GPU inventory (`nvidia-smi`). **Fail fast on degraded GPUs** — check the
console output before burning hours.

## 3. Software stack on the instance

```bash
sudo apt-get update && sudo apt-get install -y git python3.11 python3-pip
git clone -b upg https://github.com/heicfast/traceguard.git
cd traceguard
python -m pip install -e '.[research,analysis,serving]'   # vLLM v0.33.x pinned
python -m pytest -q                                       # CPU suite must pass HERE too
```

Rules that travel with you: the CPU suite must pass on the instance before
any GPU hour is spent; `traceguard doctor` should be green; the vLLM
version floor (>=0.33.0) is enforced by the server manager.

Export model access once: `export HF_TOKEN=...` (Hugging Face token for
gated weights — never committed).

## 4. Serving a model (pinned, recorded)

```bash
# Print the deterministic argv first (inspect it — it IS the treatment):
traceguard serve-model argv --spec configs/serving/qwen3_14b_a100.yaml

# Launch (health-gated; launch record lands in the run dir as evidence):
traceguard serve-model launch --spec configs/serving/qwen3_14b_a100.yaml \
  --log-dir runs/serving --port 8000 --out runs/serving/launch_record.json
```

One server per model; ports 8000 (agent model) and 8001 (monitor model) is
the convention in the matrix template. Kernel/flag standard:
docs/INFERENCE_ENGINEERING.md.

## 5. Admission probe (fail-closed, tier-matched)

```bash
python scripts/aws/probe_and_pin.py --config configs/aws_pilot_v0.7.template.yaml \
  --serving-spec configs/serving/qwen3_14b_a100.yaml --role agent --port 8000
```

Gates: server health, served-model presence, test generation with usage
fields; for API monitors, the models list is recorded and the served id must
match. The artifact hash goes into the config (`probe_artifact_sha256`);
the factory refuses to construct providers whose artifact/tier/model
mismatch (see `models/factory.py`).

## 6. Running a matrix cell end-to-end

```bash
bash scripts/aws/run_matrix_cell.sh configs/materialized/<cell>.yaml \
  configs/serving/qwen3_14b_a100.yaml runs/v7/<cell>
```

Stage chain: install → probe&pin → serve → generate → monitor → analyze →
verify → canonicalize → export → cost-report (→ optional
`TG_S3_TARGET=s3://bucket/prefix` evidence sync). Every stage is resumable;
spot interruptions cost at most the current shard.

Budget guard: set `budget_usd_hard_cap` in the config for API arms; meter
compute with:

```bash
traceguard cost-report --ledger-dir runs/v7/<cell>/costs   # after recording hours
```

## 7. Evidence durability (S3)

```bash
traceguard sync-evidence --run-dir runs/v7/<cell> --target s3://<bucket>/v7/<cell>
```

Digest-verified upload (SHA-256 in object metadata + verify-after-write
re-download), producing `artifacts/sync_manifest.json`. Recommended: sync
after every landed stage, not only at the end.

## 8. Teardown (never skip)

```bash
aws ec2 terminate-instances --instance-ids <id>
```

Credits burn while instances EXIST, not while they compute. Spot instances
left running overnight are the classic grant-killer; set an alarm if you
must leave one alive. Verify in the console that nothing is running before
closing the laptop.

## 9. Troubleshooting

| Symptom | First check |
|---|---|
| vLLM exits during startup | `runs/serving/vllm_server.log` — usually `max-model-len` x batch exceeds KV; lower `max_num_seqs` or `max_model_len` (a NEW spec hash, recorded as a treatment change) |
| gpt-oss launch refused | Correct: MXFP4 needs Hopper+. Use the p5 tier or another monitor. |
| Admission probe fails | The gates are the point — do not bypass; fix the cause (wrong served id, stale spec, wrong tier) |
| BudgetExceededError mid-run | Intended fail-closed behavior; raise the cap only as a deliberate, recorded decision |
| Spot interruption | Resume: rerun the same cell script; completed episodes are skipped via the completion ledger |
