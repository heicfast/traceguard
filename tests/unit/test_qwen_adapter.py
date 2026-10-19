"""Dependency-free regression tests for the optional GPU adapter."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace

from traceguard.models.interface import GenerationSettings
from traceguard.models.qwen import QwenHFProvider


class _Device:
    type = "cpu"
    index = None

    def __str__(self) -> str:
        return "cpu"


class _Row:
    def __init__(self, values: list[int]):
        self.values = values

    def __getitem__(self, item):
        return _Row(self.values[item]) if isinstance(item, slice) else self.values[item]

    def tolist(self) -> list[int]:
        return list(self.values)


class _InputIds:
    shape = (1, 2)
    device = _Device()

    def __getitem__(self, item: int) -> _Row:
        assert item == 0
        return _Row([1, 2])


class _Batch(dict):
    def __init__(self):
        super().__init__(input_ids=_InputIds())

    def to(self, _device: _Device) -> _Batch:
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2
    chat_template = "test-template"
    name_or_path = "test-tokenizer"

    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "rendered prompt"

    def __call__(self, *_args, **_kwargs) -> _Batch:
        return _Batch()

    def decode(self, ids, **_kwargs) -> str:
        values = ids.values if isinstance(ids, _Row) else ids
        return "answer" if values else ""

    def __len__(self) -> int:
        return 100


class _Random:
    def __init__(self, torch: _Torch):
        self.torch = torch

    @contextmanager
    def fork_rng(self, *, devices: list[int], enabled: bool):
        assert devices == []
        assert enabled is True
        self.torch.in_rng_scope = True
        try:
            yield
        finally:
            self.torch.in_rng_scope = False


class _Cuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _Torch:
    __version__ = "test"
    version = SimpleNamespace(cuda=None)
    cuda = _Cuda()

    def __init__(self):
        self.in_rng_scope = False
        self.seed: int | None = None
        self.random = _Random(self)

    def manual_seed(self, seed: int) -> None:
        assert self.in_rng_scope
        self.seed = seed

    @contextmanager
    def inference_mode(self):
        yield

    @staticmethod
    def are_deterministic_algorithms_enabled() -> bool:
        return False


class _Model:
    device = _Device()

    def __init__(self, torch: _Torch):
        self.torch = torch
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        assert self.torch.in_rng_scope
        assert self.torch.seed == 7919
        assert "generator" not in kwargs
        return [_Row([1, 2, 10, 11])]


def test_generate_uses_transformers_v5_compatible_isolated_rng(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(__version__="5.16.1"))
    torch = _Torch()
    provider = QwenHFProvider.__new__(QwenHFProvider)
    provider._torch = torch
    provider.tokenizer = _Tokenizer()
    provider.model = _Model(torch)
    provider.base_seed = 0
    provider.model_id = "test-model"
    provider.model_revision = "model-sha"
    provider.tokenizer_revision = "tokenizer-sha"
    provider.device_map = {"": 0}
    provider.applied_dtype = "float16"

    turn = provider.generate(
        [{"role": "user", "content": "test"}],
        GenerationSettings(seed=1, max_new_tokens=4, temperature=0.6),
    )

    assert provider.model.kwargs is not None
    assert provider.model.kwargs["temperature"] == 0.6
    assert turn.generated_token_ids == [10, 11]
    assert turn.generation_parameters["seed"] == 7919
    assert turn.runtime_metadata["rng_strategy"] == (
        "torch.random.fork_rng+torch.manual_seed"
    )
    assert torch.in_rng_scope is False
