"""Self-test for the scale-guard harness (ADR-073): the harness must be PROVEN to catch the
realistic super-linear shapes -- planted O(n)/O(n*log n)/O(n^2)/mixed-term/in-loop-rebuild/
key-discrimination/non-degeneracy -- before any real adopter relies on it. Registry-exempt."""

import math

import numpy as np
import pandas as pd
import pytest

from tests._perf_structural import assert_subquadratic_growth, rows_scanned_counter

SIZES = (256, 1024, 4096)


# --- harness: growth-exponent gate (Task 2) ---
def test_passes_linear():
    assert_subquadratic_growth(lambda n: n, sizes=SIZES)


def test_passes_nlogn():
    assert_subquadratic_growth(lambda n: n * max(int(math.log2(n)), 1), sizes=SIZES)


def test_catches_pure_quadratic():
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(lambda n: n * n, sizes=SIZES)


def test_red_witness_is_robust_mixed_term_quadratic():
    # RED gate is n^2 + 100n (exp ~1.89), asserted with a >=1.6 margin so a later `sizes` change
    # cannot silently defang the harness's own catch-proof (R1).
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(lambda n: n * n + 100 * n, sizes=SIZES)
    counts = [n * n + 100 * n for n in SIZES]
    exp = math.log(counts[-1] / counts[0]) / math.log(SIZES[-1] / SIZES[0])
    assert exp >= 1.6, f"RED witness margin eroded: exp={exp:.3f}"


def test_reference_boundary_values_not_a_gate():
    # Documented monotonicity/reference (R1) -- NOT the pass/fail gate.
    for fn, expected in [(lambda n: n * n + 1000 * n, 1.51), (lambda n: n * n + 10000 * n, 1.11)]:
        counts = [fn(n) for n in SIZES]
        exp = math.log(counts[-1] / counts[0]) / math.log(SIZES[-1] / SIZES[0])
        assert abs(exp - expected) < 0.02, f"reference drift: {exp:.3f} vs {expected}"


def test_non_degeneracy_enforced():
    with pytest.raises(AssertionError, match="work_floor"):
        assert_subquadratic_growth(lambda n: 0, sizes=SIZES)


def test_degenerate_ok_opt_in_passes():
    assert assert_subquadratic_growth(lambda n: 0, sizes=SIZES, degenerate_ok=True) is None


def test_two_sizes_accepted():
    assert_subquadratic_growth(lambda n: n, sizes=(1500, 10000))


# --- rows_scanned_counter (Task 3) ---
def test_boolean_mask_counts_label_select_does_not():
    df = pd.DataFrame({"a": range(100), "b": range(100)})
    with rows_scanned_counter() as c:
        _ = df["a"]  # label -> 0
        _ = df[["a", "b"]]  # label list -> 0
        _ = df[df["a"] > 50]  # boolean mask -> +100
        _ = df[np.array([True] * 50 + [False] * 50)]  # boolean ndarray -> +100
    assert c["n"] == 200


def test_int_array_getitem_is_label_select_not_scan():
    # r2/M1: df[int_list] is COLUMN selection by label -- only valid on an int-column frame, and it
    # is NOT a row rescan -> counts 0. (A string-column frame raises KeyError on df[[0,1]].)
    df = pd.DataFrame({0: range(100), 1: range(100), 2: range(100)})
    with rows_scanned_counter() as c:
        _ = df[[0, 1]]  # int labels -> column select -> 0
    assert c["n"] == 0


def test_groupby_and_take_count():
    df = pd.DataFrame({"g": [1, 1, 2], "x": [1, 2, 3]})
    with rows_scanned_counter() as c:
        df.groupby("g")  # +3
        df.take([0, 1])  # +2
    assert c["n"] == 5


def _rebuild_in_loop(n):  # the S4 regression: m ~ n items, groupby rebuilt each -> n*n
    df = pd.DataFrame({"g": np.arange(n) % 10, "x": np.arange(n)})
    with rows_scanned_counter() as c:
        for _ in range(n):
            df.groupby("g")
    return c["n"]


def _build_once(n):  # the fixed pattern: groupby once, O(1) lookups -> n
    df = pd.DataFrame({"g": np.arange(n) % 10, "x": np.arange(n)})
    with rows_scanned_counter() as c:
        g = df.groupby("g").indices
        for _ in range(n):
            _ = g  # O(1)
    return c["n"]


def test_in_loop_groupby_rebuild_is_caught():
    with pytest.raises(AssertionError):
        assert_subquadratic_growth(_rebuild_in_loop, sizes=(64, 128, 256), label="in-loop-rebuild")


def test_build_once_passes():
    assert_subquadratic_growth(_build_once, sizes=(64, 128, 256), label="build-once")


def test_relative_growth_is_the_robust_gate():
    # PRIMARY, pandas-version-independent (rev2): a boolean-mask rescan-in-loop GROWS (exp ~2); a
    # column-select-in-loop does NOT. The absolute == 200/0/5 above are SECONDARY and pinned to the
    # observed pandas -- uv.lock spans pandas 2.3.3 (py<3.11) and 3.0.2, whose internal .take
    # routing can differ, so the growth property (not a magic number) is the durable assertion.
    def _mask_in_loop(n):
        df = pd.DataFrame({"k": np.arange(n) % 10, "x": np.arange(n)})
        with rows_scanned_counter() as c:
            for v in range(n):
                _ = df[df["k"] == (v % 10)]
        return c["n"]

    def _colselect_in_loop(n):
        df = pd.DataFrame({"a": np.arange(n), "b": np.arange(n)})
        with rows_scanned_counter() as c:
            for _ in range(n):
                _ = df[["a", "b"]]
        return c["n"]

    with pytest.raises(AssertionError):
        assert_subquadratic_growth(_mask_in_loop, sizes=(64, 128, 256))  # rescan -> exp ~2
    assert_subquadratic_growth(_colselect_in_loop, sizes=(64, 128, 256), degenerate_ok=True)  # 0
