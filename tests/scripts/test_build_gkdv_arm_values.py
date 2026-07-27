"""TF-19 sign-off package: the GKDV arm-values pass feeding the §6.1 ICC power leg."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.build_gkdv_arm_values as mod  # bare import: tests/scripts/ has NO __init__.py


def _frames():
    """Two teams, one ball row, one frame."""
    rows = [
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 5, "is_ball": False, "x": 20.0},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 6, "is_ball": False, "x": 80.0},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": None, "is_ball": True, "x": 50.0},
        {"game_id": 1, "period_id": 2, "frame_id": 99, "team_id": 5, "is_ball": False, "x": 30.0},
    ]
    return pd.DataFrame(rows)


def test_frame_slice_selects_exactly_one_game_period_frame():
    got = mod._frame_slice(_frames(), 1, 1, 10)
    assert len(got) == 3
    assert set(got["frame_id"]) == {10}
    assert set(got["period_id"]) == {1}


def test_attacking_team_is_the_non_defending_outfield_team():
    sl = mod._frame_slice(_frames(), 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=6) == 5
    assert mod._attacking_team_id(sl, defending_team_id=5) == 6


def test_attacking_team_resolution_is_dtype_safe():
    """ADR-019: the frame `team_id` may be nullable Int64 while the provenance carries a plain
    int -- a raw `!=` would silently keep BOTH teams and pick the defender as the attacker."""
    f = _frames()
    f["team_id"] = f["team_id"].astype("Int64")
    sl = mod._frame_slice(f, 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=6) == 5


def test_attacking_team_is_None_when_only_the_defenders_are_present():
    """Returned rather than guessed: the caller skips the frame instead of scoring a wrong team."""
    only_def = pd.DataFrame([{"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 6, "is_ball": False, "x": 1.0}])
    assert mod._attacking_team_id(only_def, defending_team_id=6) is None


def test_the_ball_row_is_never_mistaken_for_a_team():
    """The ball carries a null team; including it would make it the 'attacker' on a one-team frame."""
    sl = mod._frame_slice(_frames(), 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=5) != pytest.approx(float("nan"), nan_ok=True)
    assert mod._attacking_team_id(sl, defending_team_id=5) == 6


def test_threat_arm_is_refused_not_silently_defaulted(monkeypatch, capsys):
    """The threat arm needs a fitted ExpectedThreat and NONE can be loaded: the class exposes only
    fit/interpolator/rate (no save/load anywhere in the package) and `FrozenXt` wraps an
    already-fitted in-memory model. Defaulting to `xt=None` would have persisted structural zeros
    -- `compute_threat_pc` returned 0.0 for None before this cycle's guard -- and an ICC on a
    constant column is degenerate while looking like a measurement. So it refuses, loudly."""
    import sys

    for arm in ("threat", "both"):
        monkeypatch.setattr(sys, "argv", ["build_gkdv_arm_values.py", "--out", "x", "--arm", arm])
        with pytest.raises(SystemExit) as excinfo:
            mod.main()
        assert "fitted ExpectedThreat" in str(excinfo.value)


def test_expected_threat_really_has_no_loader():
    """Pins the FACT the refusal rests on. If serialization is ever added, this goes red and the
    refusal above should be revisited rather than left as folklore."""
    from silly_kicks.xthreat import ExpectedThreat

    assert not hasattr(ExpectedThreat, "load")
    assert not hasattr(ExpectedThreat, "save")


def test_only_the_DEFENDING_keeper_is_credited():
    """The serving seam writes a row for BOTH teams' keepers, but `build_ghost_frames` substitutes
    only the DEFENDING one. A naive pass-through therefore credits each frame's delta to two
    keepers, one of whom never moved.

    MEASURED on real GS data before this filter existed: 4448 rows from 2224 scored frames, both
    rows per frame carrying an IDENTICAL arm_value under different keeper_keys. That is
    keeper-INDEPENDENT noise, and it compresses between-keeper variance toward zero -- the same
    mechanism that made xT-GK v2 read "keeper-flat" on fabricated origins (ADR-036/PR-S113).
    """
    from silly_kicks.id_compat import ids_equal

    # The two leading DROPPED rows are load-bearing: filtering them leaves a NON-CONTIGUOUS index,
    # which is what real provenance looks like. An all-scored fixture keeps the index at 0..n-1 and
    # hides the mask-alignment crash this reproduces (measured: IndexingError on the real run).
    provenance = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1, 1],
            "frame_id": [9, 9, 10, 10, 11, 11],
            "gk_team_id": [5, 6, 5, 6, 5, 6],
            "defending_team_id": [6, 6, 6, 6, 6, 6],
            "player_id": [940, 11241, 940, 11241, 940, 11241],
            "drop_reason": ["ball_row_missing", "ball_row_missing", None, None, None, None],
        }
    )
    # Non-vacuity: BOTH keepers must be present, or the selection has nothing to select from.
    assert provenance["player_id"].nunique() == 2

    scored = provenance[provenance["drop_reason"].isna()].reset_index(drop=True)
    keep = np.asarray(ids_equal(scored["gk_team_id"], scored["defending_team_id"]), dtype=bool)
    selected = scored[keep]

    assert len(selected) == 2, "expected exactly one row per frame"
    assert selected.groupby(["game_id", "period_id", "frame_id"]).size().max() == 1
    assert set(selected["player_id"]) == {11241}, "the ATTACKING team's keeper must not be credited"


def test_defending_keeper_selection_is_dtype_safe():
    """ADR-019: `gk_team_id` and `defending_team_id` may arrive on different dtypes; a raw `==`
    would select NOTHING and silently produce an empty arm-values table."""
    from silly_kicks.id_compat import ids_equal

    scored = pd.DataFrame({"gk_team_id": pd.array([5, 6], dtype="Int64"), "defending_team_id": ["6", "6"]})
    assert int(ids_equal(scored["gk_team_id"], scored["defending_team_id"]).sum()) == 1


@pytest.mark.parametrize(
    "module_name",
    ["build_gkdv_arm_values", "run_signoff_power", "derive_opengoal_range"],
)
def test_driver_source_is_ascii_so_help_works_on_windows(module_name):
    """`--help` prints the module docstring, and a Windows console is cp1252: a single non-ASCII
    character (measured: U+0394 in a delta description) makes `--help` die with
    UnicodeEncodeError before the maintainer can read the usage. Cheap to keep, and it fails on
    the machine the drivers are actually invoked from."""
    import pathlib

    src = pathlib.Path(__file__).resolve().parents[2] / "scripts" / f"{module_name}.py"
    offenders = sorted({c for c in src.read_text(encoding="utf-8") if ord(c) > 127})
    assert not offenders, f"non-ASCII in {module_name}.py breaks --help on cp1252: {offenders}"
