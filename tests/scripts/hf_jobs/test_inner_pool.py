"""InnerPool: results keyed, completion order recorded, combination order is the caller's (sorted)."""

from __future__ import annotations

from scripts._stages import ordered_values
from scripts.hf_jobs.executor import InnerPool
from tests.scripts._toy_pipeline import _inner_eval


def test_w1_runs_in_sorted_key_order():
    pool = InnerPool(1)
    out = pool.map_keyed(_inner_eval, {"0001": (0, 1, (1.0, 2.0)), "0000": (0, 0, (1.0, 2.0))})
    assert pool.drain_log() == ["0000", "0001"]
    assert ordered_values(out) == [1.0, 2.0]


def test_w2_completes_out_of_order_but_keys_the_results():
    pool = InnerPool(2)
    try:
        items = {f"000{j}": (0, j, (1.0, 2.0, 3.0, 4.0)) for j in range(4)}
        out = pool.map_keyed(_inner_eval, items)
        log = pool.drain_log()
    finally:
        pool.shutdown()
    assert sorted(log) == sorted(items)
    assert log != sorted(items)  # non-vacuity: inner key 0000 sleeps, so it finishes last
    assert ordered_values(out) == [1.0, 2.0, 3.0, 4.0]
