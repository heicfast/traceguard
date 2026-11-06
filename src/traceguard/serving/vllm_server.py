"""Pinned vLLM server lifecycle with per-model optimal serving flags.

The server argv is DETERMINED by the audited ``ServerSpec`` (YAML, hashed)
rather than improvised on the command line, so every server launch is
reproducible and recordable as evidence.  Flags follow the engineering
standard in ``docs/INFERENCE_ENGINEERING.md``:

* Hopper (p5, H100): FlashAttention-3-class kernels via the default V1
  engine attention backend; FP8 checkpoints where the model ships them;
* Ampere (p4de, A100-80GB): FlashInfer/FA2-class backends, BF16 or official
  FP8 weights; MXFP4 checkpoints (gpt-oss) are NOT Ampere-compatible and are
  refused here (verified against the vLLM gpt-oss release notes);
* prefix caching (``--enable-prefix-caching``) for the shared-prefix monitor
  workloads; chunked prefill is the V1 default;
* explicit ``--max-model-len`` bound to the config's context budget (never
  the advertised maximum);
* per-request seeds keep episode sampling reproducible in practice
  (determinism caveats documented).

The manager never writes inside run directories; its evidence record (argv,
versions, GPU inventory, health log) is handed back to the caller for
commitment through the normal evidence pipeline.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..ids import sha256_json

VLLM_MIN_VERSION = "0.33.0"  # verified latest line, Oct 2026 (vllm.ai releases)

_ARCHITECTURE_FAMILIES_MXFP4 = {"gpt_oss_moe"}


@dataclass(frozen=True)
class ServerSpec:
    """Immutable, hashable serving specification for one model."""

    model_id: str                       # HF repo id (served weights)
    model_revision: str                 # pinned commit
    served_model_name: str              # model id accepted by the endpoint
    architecture_family: str
    dtype: str = "bfloat16"             # A100/H100 default; fp16 only for T4-class
    tensor_parallel_size: int = 1
    max_model_len: int = 32_768
    gpu_memory_utilization: float = 0.92
    max_num_seqs: int = 64
    enable_prefix_caching: bool = True
    # FP8 weight variant: only when the checkpoint is *officially* released in
    # FP8 (e.g. Qwen3 FP8, GLM-4.5-Air-FP8); quantization is a recorded factor.
    quantization: str | None = None
    # Reasoning parser: "qwen3" for Qwen3 think tags, "deepseek_r1" for
    # R1-distills; None disables server-side reasoning separation.
    reasoning_parser: str | None = None
    download_dir: str | None = None
    extra_args: tuple[tuple[str, str], ...] = ()

    def content_hash(self) -> str:
        return sha256_json(self.__dict__ | {"extra_args": list(self.extra_args)})


def build_server_argv(spec: ServerSpec, *, python_executable: str = "python") -> tuple[str, ...]:
    """Deterministic vLLM serve argv for the spec (fail-closed on mismatches)."""
    if spec.architecture_family in _ARCHITECTURE_FAMILIES_MXFP4 and spec.quantization is None:
        # gpt-oss ships native MXFP4 weights: Hopper (sm90)/Blackwell (sm100)
        # or MI300x-class only — verified against the vLLM gpt-oss support
        # announcement (vllm.ai/blog/2025-08-05-gpt-oss). Ampere p4de is not
        # a valid target; refuse rather than fail at kernel load time.
        raise ValueError(
            f"{spec.model_id} uses native MXFP4 weights and requires Hopper-or-newer "
            "NVIDIA GPUs or MI300x-class AMD GPUs; it cannot be served on Ampere "
            "(p4d/p4de/g5). Use a p5 (H100) instance or choose another monitor."
        )
    if spec.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    argv: list[str] = [
        python_executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", spec.model_id,
        "--revision", spec.model_revision,
        "--served-model-name", spec.served_model_name,
        "--dtype", spec.dtype,
        "--max-model-len", str(spec.max_model_len),
        "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        "--max-num-seqs", str(spec.max_num_seqs),
        "--tensor-parallel-size", str(spec.tensor_parallel_size),
        "--disable-log-requests",  # prompts are privileged evidence, never server logs
    ]
    if spec.enable_prefix_caching:
        argv.append("--enable-prefix-caching")
    if spec.quantization:
        argv.extend(["--quantization", spec.quantization])
    if spec.reasoning_parser:
        argv.extend(["--reasoning-parser", spec.reasoning_parser])
    if spec.download_dir:
        argv.extend(["--download-dir", spec.download_dir])
    for flag, value in spec.extra_args:
        argv.extend([flag, value])
    return tuple(argv)


@dataclass
class ServerLaunchRecord:
    """Evidence record for one server lifecycle (committed by the caller)."""

    spec_sha256: str
    argv: tuple[str, ...]
    vllm_version: str
    torch_version: str
    gpu_inventory: list[dict[str, Any]]
    cuda_version: str
    started_at_utc: str
    ready_after_seconds: float
    health_log: list[dict[str, Any]] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        import dataclasses

        return dataclasses.asdict(self)


class VLLMServerManager:
    """Launch, health-check, and tear down one pinned vLLM server."""

    def __init__(self, spec: ServerSpec, *, log_dir: str | Path, host: str = "127.0.0.1",
                 port: int = 8000, startup_timeout_seconds: float = 1800.0,
                 python_executable: str = "python"):
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.host = host
        self.port = port
        self.startup_timeout_seconds = startup_timeout_seconds
        self.python_executable = python_executable
        self.process: subprocess.Popen | None = None
        self.launch_record: ServerLaunchRecord | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    # -- environment capture ---------------------------------------------------

    def _environment_snapshot(self) -> tuple[str, str, list[dict[str, Any]], str]:
        probe = subprocess.run(
            [self.python_executable, "-c",
             "import torch, vllm; import json; print(json.dumps({"
             "'vllm': vllm.__version__, 'torch': torch.__version__,"
             "'cuda': torch.version.cuda,"
             "'gpus': [{'name': torch.cuda.get_device_properties(i).name,"
             "'total_memory': torch.cuda.get_device_properties(i).total_memory}"
             " for i in range(torch.cuda.device_count())]}))"],
            capture_output=True, text=True, timeout=180,
        )
        if probe.returncode != 0:
            raise RuntimeError(
                "vLLM environment probe failed (is vLLM installed on this image?): "
                + probe.stderr[-400:]
            )
        snapshot = json.loads(probe.stdout)
        version = str(snapshot.get("vllm", "unknown"))
        major_minor = tuple(int(part) for part in version.split(".")[:2])
        minimum = tuple(int(part) for part in VLLM_MIN_VERSION.split(".")[:2])
        if major_minor < minimum:
            raise RuntimeError(
                f"vLLM {version} is older than the audited minimum {VLLM_MIN_VERSION}; "
                "pin the serving image to avoid untracked kernel behavior drift"
            )
        return (
            version,
            str(snapshot.get("torch", "unknown")),
            list(snapshot.get("gpus", [])),
            str(snapshot.get("cuda", "unknown")),
        )

    # -- lifecycle ---------------------------------------------------------------

    def start(self) -> ServerLaunchRecord:
        from datetime import UTC, datetime

        argv = build_server_argv(self.spec, python_executable=self.python_executable)
        vllm_version, torch_version, gpus, cuda = self._environment_snapshot()
        started_at = datetime.now(UTC).isoformat()
        started = time.perf_counter()
        log_path = self.log_dir / "vllm_server.log"
        log_handle = log_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            list(argv), stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,  # own process group: teardown kills the whole tree
        )
        health: list[dict[str, Any]] = []
        deadline = started + self.startup_timeout_seconds
        ready = False
        while time.perf_counter() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"vLLM server exited during startup with code {self.process.returncode}; "
                    f"see {log_path}"
                )
            ok, detail = self._health_once()
            health.append({"t": round(time.perf_counter() - started, 1), "ok": ok,
                           "detail": detail})
            if ok:
                ready = True
                break
            time.sleep(5.0)
        if not ready:
            self.stop()
            raise RuntimeError(
                f"vLLM server did not become healthy within "
                f"{self.startup_timeout_seconds}s; see {log_path}"
            )
        self.launch_record = ServerLaunchRecord(
            spec_sha256=self.spec.content_hash(),
            argv=argv,
            vllm_version=vllm_version,
            torch_version=torch_version,
            gpu_inventory=gpus,
            cuda_version=cuda,
            started_at_utc=started_at,
            ready_after_seconds=round(time.perf_counter() - started, 1),
            health_log=health,
        )
        return self.launch_record

    def _health_once(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(
                f"http://{self.host}:{self.port}/health", timeout=5.0
            ) as response:
                return response.status == 200, f"status={response.status}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return False, type(exc).__name__

    def stop(self) -> None:
        """SIGTERM the server's process group, then SIGKILL if needed."""
        if self.process is None:
            return
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                self.process.terminate()
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    self.process.kill()
                self.process.wait(timeout=30)
        self.process = None

    def __enter__(self) -> VLLMServerManager:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


__all__ = [
    "ServerSpec",
    "build_server_argv",
    "VLLMServerManager",
    "ServerLaunchRecord",
    "VLLM_MIN_VERSION",
]
