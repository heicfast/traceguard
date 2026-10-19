"""Unit tests: canonical hashing, opaque identifiers, id leak checks."""

from __future__ import annotations

import pytest

from traceguard.ids import (
    canonical_json,
    episode_id,
    is_opaque_identifier,
    monitor_score_id,
    sha256_json,
    task_instance_id,
    task_template_id,
)


def test_canonical_json_is_order_independent():
    a = canonical_json({"a": 1, "b": [2, 3]})
    b = canonical_json({"b": [2, 3], "a": 1})
    assert a == b


def test_canonical_json_rejects_nan():
    import math
    with pytest.raises(ValueError):
        canonical_json({"x": math.nan})


def test_sha256_json_stable_and_sensitive():
    h1 = sha256_json({"x": 1})
    h2 = sha256_json({"x": 1})
    h3 = sha256_json({"x": 2})
    assert h1 == h2 and h1 != h3 and len(h1) == 64


def test_episode_id_is_opaque_and_stable():
    e1 = episode_id("run", "ti-123", "attack", 7)
    e2 = episode_id("run", "ti-123", "attack", 7)
    e3 = episode_id("run", "ti-123", "honest", 7)
    assert e1 == e2 and e1 != e3
    assert is_opaque_identifier(e1), f"episode id leaks a label: {e1}"
    # the honest/attack pair must not be distinguishable from the id alone
    assert len(e1) == len(e3)


def test_episode_id_rejects_unknown_policy():
    with pytest.raises(ValueError):
        episode_id("run", "ti", "sabotage", 1)


def test_monitor_score_id_stable():
    a = monitor_score_id("ep", "actions", "llm", "deadbeef")
    b = monitor_score_id("ep", "actions", "llm", "deadbeef")
    c = monitor_score_id("ep", "actions", "llm", "feedface")
    assert a == b and a != c
    assert is_opaque_identifier(a)


def test_template_id_readable_but_unlabeled():
    tid = task_template_id("config_update", "config_update/flag_flip", 1)
    assert "flag_flip" in tid
    assert is_opaque_identifier(tid)
    assert is_opaque_identifier(task_instance_id(tid, 5))


def test_similar_inputs_hash_apart():
    ids = {episode_id("r", f"ti-{i}", "attack", 1) for i in range(100)}
    assert len(ids) == 100
