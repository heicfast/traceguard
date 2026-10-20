# Raw Evidence and Independent Verification Specification

Version 2.0.0.

## Evidence levels

| Level | Contents | Release policy |
|---|---|---|
| Public episode | task ids, observable events, parsed reasoning, tool activity, outcomes, provenance | release after leakage audit |
| Privileged episode | exact prompts/messages, hidden instruction, token ids, raw generations, state snapshots, transitions, grader inputs | controlled research artifact; redact only by producing a separately hashed derivative |
| Monitor score | exact rendered monitor prompt, view hash, raw response, parsed score/verdict, inference audit | public if its evidence view passes leakage checks |
| Runtime | GPU/driver/framework versions, package inventory, model/tokenizer revisions, kernel source commit, wall/VRAM metrics | public |
| Integrity | write-ahead transactions, commit records, SHA-256 inventory | public unless the embedded privileged transaction is separated |

## Per-generation requirements

Every inference provider's `model_turn_audit` must retain:

- original structured messages and their canonical JSON hash;
- rendered chat-template prompt and UTF-8 hash;
- complete prompt token ids and hash;
- complete generated token ids and hash;
- decoded text with and without special tokens and both hashes;
- unmodified parsed reasoning and visible assistant projections;
- parse status, stop reason, prompt/generated token counts, elapsed time;
- effective seed and full generation parameters;
- model and tokenizer revisions, chat-template hash, library versions, dtype,
  device and GPU capability.
- audited model-profile and admission-probe hashes; exact quantization config,
  attention backend, context cap, optional adapter identity/revision, model
  memory footprint, and per-turn peak allocated/reserved GPU memory.

No field above may be character-capped. Human-friendly excerpts may exist only
as additional fields and must point to the complete record.

Each run manifest additionally retains a secret-free runtime snapshot: complete
installed-package inventory, Python/platform identity, CUDA/PyTorch/cuDNN
versions and determinism flags, GPU UUID/name/driver/memory query, and a strict
allowlist of execution-relevant environment variables. Credential variables
are excluded by construction.

QLoRA artifacts require a development-only corpus manifest, source-view and
target hashes, base-model revision, seed/hyperparameters, training metrics,
package versions, and SHA-256/size for every emitted adapter/tokenizer file.

For multi-stage monitors, the requirement applies independently to every
action extractor, reasoning extractor, fact checker, adjudicator, and retry.
The score record retains the complete prompt list, raw response list, bounded
intermediate reports actually supplied downstream, per-stage model revision,
token accounting, runtime audit payload, and a combined source-view hash. Raw
agent reasoning remains in the source episode audit even though the final
adjudicator deliberately does not receive it.

## Verification procedure

1. Validate public, privileged, and score records against their committed JSON
   Schemas.
2. Recompute every per-field hash from UTF-8 bytes or canonical JSON.
3. Verify transaction and commit hashes, JSONL projections, completion ledger,
   episode uniqueness, and artifact checksum inventory.
4. Re-run the exact state grader from initial state, transitions, invariants,
   and final state; compare all stored outcome labels.
5. Recompile every monitor view from the pinned source revision and compare its
   hash with the score record.
6. Recompute analysis tables/figures from score records and compare emitted
   plot-data hashes.

`traceguard export --run-dir RUN --archive OUT.tar.gz` enforces the minimum
research-release inventory, validates score/privileged schemas and embedded
inference hashes, writes `artifacts/artifact_manifest.json`, regenerates the
checksum ledger, and atomically creates the archive outside the run tree.
`--allow-incomplete` is only for explicitly marked operational checkpoints.

The verifier proves internal consistency and replayability of recorded
observations. It does not prove that a stochastic GPU run would emit the same
tokens on different hardware.

## Privacy and publication

Hidden attack instructions and exact agent prompts are privileged by design.
They may be released for a synthetic benchmark only after a dedicated leakage
and dual-use review. Never overwrite the primary evidence to create a public
bundle. Create a derivative, list every removed field, hash both manifests, and
retain the primary bundle unchanged.

