from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "run_gpu_workers.py"
    spec = importlib.util.spec_from_file_location("run_gpu_workers", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dual_gpu_worker_mapping_is_disjoint_and_complete():
    module = _module()
    workers = module.worker_specs(gpu_count=2, kernel_shard=1, kernel_shard_count=2)
    assert [worker.global_shard for worker in workers] == [2, 3]
    assert {worker.global_shard_count for worker in workers} == {4}
    assert [worker.physical_gpu for worker in workers] == [0, 1]


def test_single_gpu_mapping_preserves_original_shards():
    module = _module()
    workers = module.worker_specs(gpu_count=1, kernel_shard=1, kernel_shard_count=2)
    assert len(workers) == 1
    assert workers[0].global_shard == 1
    assert workers[0].global_shard_count == 2
