# Failed monitoring attempts — instrument-v1 contract + allocator forensics (2026-10-20/21)

These are NOT research-grade runs. They are the recorded evidence for the
phase-3 pilot attempt-1 monitor failures, documented in
`docs/deviations.md` (2026-10-21 entry) and superseded by the v2 monitor
instrument (commit 7123a31):

- `anchor_mon_attempt1/` — output of `umar0x/traceguard-monitor-scoring`
  v9 (pin 542d5d4, v1 prompt). Both T4 workers completed every call with
  ZERO generation failures (640/640), but the v1 contract (self-computed
  risk_score == rubric sum) invalidated 69-89% of LLM scores (M2 FAIL).
  Includes the full gpu-00 worker copy (all score files + monitor.log
  showing `ok: true, failures: []`) and the gpu-01 partial. The kernel's
  own summary (in `anchor_kernel_environment.json` + `anchor_stdout.txt`)
  shows monitor_rc 0 / analysis_rc 0 / monitor_gate_rc 3 / verify_rc 1
  (37 uncovered-file problems — the pre-canonicalize chain) / export_rc 1.
  The score audits contain the dominant invalid class verbatim:
  `ValueError: risk_score does not equal rubric sum`.
- `scale_mon_attempt1/` — output of `heicjpg/traceguard-mon-scale-qwen14`
  v4 (14B NF4 per-T4, pre-allocator-fix). Deterministic monitors complete;
  LLM arm lost 13% of calls to fragmentation-driven marginal OOMs
  (440 MiB..2.23 GiB requests failing with up to 2.27 GiB reserved but
  unallocated). The failure list in the worker `monitor.log` files is the
  allocator forensics evidence.

The v1 prompt hashes for these score files (llm `075bc303...`, fusion
`edcd2d17...`/`75a8e950...`) are retained in the frozen-prompt registry so
these records keep verifying under current code.
