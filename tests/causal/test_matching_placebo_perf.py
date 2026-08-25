"""Structural perf guard (ADR-068): placebo_shift's cluster grouping is prepared ONCE, not
re-derived (np.unique + per-cluster masks) on every one of n_seeds iterations."""

import numpy as np

import silly_kicks.causal.matching as _m
from silly_kicks.causal.matching import placebo_shift
from tests._perf_structural import call_counter


def _inputs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    x_base = rng.normal(size=(n, 2))
    x_gk = rng.normal(size=(n, 2))
    z = (rng.uniform(size=n) < 0.5).astype(int)
    y = rng.normal(size=n)
    cluster_ids = np.repeat(np.arange(8), n // 8)  # 8 clusters
    return x_base, x_gk, y, z, cluster_ids


def test_cluster_prep_built_once_not_per_seed(monkeypatch):
    x_base, x_gk, y, z, cluster_ids = _inputs()
    calls = call_counter(monkeypatch, _m, "_prepare_cluster_reassign")
    out = placebo_shift(x_base, x_gk, y, z, n_seeds=25, rng_seed=0, cluster_ids=cluster_ids)
    assert calls["n"] == 1  # once total; pre-ADR-068 np.unique+masks ran inside _cluster_reassign per seed
    assert out["shifts"].shape == (25,)
    assert out["permutation_unit"] == "cluster"


def test_hoist_is_byte_identical_to_direct_reassign():
    # The prepared per-seed path reproduces the standalone _cluster_reassign exactly (same rng draws).
    _x_base, x_gk, _y, _z, cluster_ids = _inputs(n=40)
    rng_a = np.random.default_rng(7)
    direct = _m._cluster_reassign(x_gk, cluster_ids, rng_a)
    rng_b = np.random.default_rng(7)
    prepared = _m._cluster_reassign_prepared(_m._prepare_cluster_reassign(x_gk, cluster_ids), rng_b)
    assert np.array_equal(direct, prepared)
