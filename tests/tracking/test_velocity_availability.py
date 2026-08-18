"""ADR-063: the ``zero_velocity_if_unavailable`` edge helper.

Contract (method-aware, fail-fast):
- vx/vy present            -> returned UNCHANGED (same object, no copy, no mutation).
- vx/vy absent + DECLARED   -> a COPY with vx=vy=0 (the zero-velocity positional model);
  ``speed_source == SPEED_SOURCE_UNAVAILABLE`` on every row.
- vx/vy absent + FORGOTTEN + a velocity-requiring method -> RAISES ValueError (caller bug;
  fail fast at the policy edge so an all-NaN column never masquerades as legitimately-absent,
  ADR-043). The input is never mutated.
- vx/vy absent + FORGOTTEN + a velocity-FREE method (voronoi) -> unchanged (voronoi needs no
  velocity, so a missing-velocity frame is not a bug for it).
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking._velocity_availability import zero_velocity_if_unavailable
from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE


def _frame(speed_source: str, *, with_vel: bool) -> pd.DataFrame:
    d = {
        "player_id": [1, 2],
        "team_id": [1, 2],
        "is_ball": [False, False],
        "x": [10.0, 20.0],
        "y": [30.0, 40.0],
        "speed_source": [speed_source, speed_source],
    }
    if with_vel:
        d["vx"] = [1.0, 2.0]
        d["vy"] = [3.0, 4.0]
    return pd.DataFrame(d)


def test_present_velocity_is_returned_unchanged_same_object():
    f = _frame("native", with_vel=True)
    assert zero_velocity_if_unavailable(f) is f  # no copy, no mutation


def test_declared_unavailable_gets_a_zero_velocity_copy():
    f = _frame(SPEED_SOURCE_UNAVAILABLE, with_vel=False)
    out = zero_velocity_if_unavailable(f)
    assert out is not f and "vx" not in f.columns  # input untouched
    assert (out["vx"] == 0.0).all() and (out["vy"] == 0.0).all()
    # The marker is deliberately LEFT in place on the copy (the spec's helper caveat): the
    # copy is internally inconsistent if passed onward, but that is fine for the immediate
    # pitch-control call it is built for.
    assert (out["speed_source"] == SPEED_SOURCE_UNAVAILABLE).all()


@pytest.mark.parametrize("method", ["spearman", "fernandez_bornn"])
def test_forgotten_velocity_raises_for_a_velocity_requiring_method(method):
    f = _frame("native", with_vel=False)  # forgot derive_velocities(): a caller BUG
    with pytest.raises(ValueError, match="requires velocity columns"):
        zero_velocity_if_unavailable(f, method=method)
    assert "vx" not in f.columns  # input never mutated on the raise path


def test_forgotten_velocity_is_unchanged_for_a_velocity_free_method():
    f = _frame("native", with_vel=False)
    out = zero_velocity_if_unavailable(f, method="voronoi")  # voronoi needs no velocity
    assert out is f and "vx" not in out.columns


def test_partially_marked_raises_because_not_all_rows_are_unavailable():
    # velocity_unavailable_by_design requires EVERY row marked; a PARTIAL mark means some genuine
    # velocity-bearing source is also missing its velocity -- the caller-bug path must win.
    f = pd.DataFrame(
        {
            "player_id": [1, 2],
            "team_id": [1, 2],
            "is_ball": [False, False],
            "x": [10.0, 20.0],
            "y": [30.0, 40.0],
            "speed_source": [SPEED_SOURCE_UNAVAILABLE, "native"],
        }
    )
    with pytest.raises(ValueError, match="requires velocity columns"):
        zero_velocity_if_unavailable(f, method="spearman")


def test_empty_frames_returned_unchanged_never_raises():
    # An empty frame set is introspection / no-data, not a forgotten derive_velocities(): it must
    # NOT raise even for a velocity-requiring method (VAEP feature_column_names passes empty frames).
    f = _frame("native", with_vel=False).iloc[0:0]
    out = zero_velocity_if_unavailable(f, method="spearman")
    assert out is f and len(out) == 0


def test_default_method_is_velocity_requiring():
    # The default must be a velocity-requiring method, or a forgotten-velocity frame would
    # slip through unflagged at every call site that omits ``method``.
    f = _frame("native", with_vel=False)
    with pytest.raises(ValueError, match="requires velocity columns"):
        zero_velocity_if_unavailable(f)
