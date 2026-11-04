"""TF-46 identify_corners: provider-aware corner anchoring."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import CORNER_IDENTIFY_COLUMNS, identify_corners

_CORNER_CROSSED = 5
_CORNER_SHORT = 6
_PASS = 0


def _actions(rows: list[dict]) -> pd.DataFrame:
    base = {
        "game_id": "g1",
        "original_event_id": "e",
        "action_id": 0,
        "period_id": 1,
        "time_seconds": 0.0,
        "team_id": "t1",
        "player_id": "p1",
        "start_x": 1.0,
        "start_y": 0.5,
        "end_x": 20.0,
        "end_y": 34.0,
        "type_id": _PASS,
        "result_id": 1,
        "bodypart_id": 0,
    }
    out = []
    for i, r in enumerate(rows):
        d = dict(base)
        d.update(r)
        d["action_id"] = i
        out.append(d)
    return pd.DataFrame(out)


def test_spadl_corner_crossed_is_direct():
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2, "time_seconds": 100.0}])
    out = identify_corners(acts, provider="sportec")
    assert len(out) == 1
    row = out.iloc[0]
    assert row["corner_kind"] == "direct"
    assert row["corner_source"] == "spadl_corner"
    assert row["t_kick"] == 100.0
    assert row["attacking_team_id"] == "t1"
    assert list(out.columns) == list(CORNER_IDENTIFY_COLUMNS)


def test_spadl_corner_short_is_short():
    acts = _actions([{"type_id": _CORNER_SHORT, "start_x": 104.5, "start_y": 67.8}])
    out = identify_corners(acts, provider="sportec")
    assert out.iloc[0]["corner_kind"] == "short"


def test_corner_side_from_kick_y():
    acts = _actions(
        [
            {"type_id": _CORNER_CROSSED, "start_y": 1.0, "time_seconds": 10.0},
            {"type_id": _CORNER_CROSSED, "start_y": 67.0, "time_seconds": 20.0},
        ]
    )
    out = identify_corners(acts, provider="sportec").sort_values("t_kick")
    assert set(out["corner_side"]) == {"left", "right"}


def test_corner_id_is_per_game_chronological():
    acts = _actions(
        [
            {"type_id": _CORNER_CROSSED, "time_seconds": 50.0},
            {"type_id": _PASS, "time_seconds": 55.0},
            {"type_id": _CORNER_CROSSED, "time_seconds": 60.0},
        ]
    )
    out = identify_corners(acts, provider="sportec").sort_values("t_kick")
    assert list(out["corner_id"]) == [0, 1]


def test_skillcorner_source_token():
    acts = _actions([{"type_id": _CORNER_CROSSED}])
    out = identify_corners(acts, provider="skillcorner")
    assert out.iloc[0]["corner_source"] == "skillcorner_marker"


def test_gi_is_deprecated_and_ignored():
    """``gi=`` is deprecated (decision B1): it emits ``DeprecationWarning`` and does NOT refine the corner.

    Under the retired GI refinement this start type marked the corner ``unclear``; it now stays ``direct``.
    """
    acts = _actions([{"type_id": _CORNER_CROSSED, "period_id": 1, "time_seconds": 100.0}])
    gi = pd.DataFrame(
        {
            "game_id": ["g1"],
            "period_id": [1],
            "time_seconds": [100.0],
            "game_interruption_before": ["corner_for"],
            "start_type": ["pass_reception"],  # would have been "unclear" under the old GI refinement
        }
    )
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = identify_corners(acts, provider="skillcorner", gi=gi)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert out.iloc[0]["corner_kind"] == "direct"


def test_no_corners_returns_empty_schema():
    out = identify_corners(_actions([{"type_id": _PASS}]), provider="sportec")
    assert out.empty
    assert list(out.columns) == list(CORNER_IDENTIFY_COLUMNS)


def test_missing_required_column_raises():
    acts = _actions([{"type_id": _CORNER_CROSSED}]).drop(columns=["start_y"])
    try:
        identify_corners(acts, provider="sportec")
    except KeyError as e:
        assert "start_y" in str(e)
    else:
        raise AssertionError("expected KeyError for missing start_y")


def test_pure_does_not_mutate_input():
    acts = _actions([{"type_id": _CORNER_CROSSED}])
    before = acts.copy(deep=True)
    identify_corners(acts, provider="sportec")
    pd.testing.assert_frame_equal(acts, before)


# --- spec section 9.4 coherence (non-vacuous: it was "right" / a placeholder before) ---------------
def test_nan_start_y_yields_nan_side_on_every_provider():
    """A NaN ``start_y`` yields a NaN ``corner_side`` on every provider (spec section 9.4).

    Non-vacuous: a real ``start_y`` below the midline still yields ``"right"`` -- the old rule mapped a NaN
    ``start_y`` to ``"right"`` too (``NaN >= midline`` is False), so the fix must change only the NaN case.
    """
    nan = float("nan")
    for provider in ("sportec", "skillcorner", "statsbomb"):
        acts = _actions(
            [
                {"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": 0.2, "time_seconds": 10.0},
                {"type_id": _CORNER_CROSSED, "start_x": 104.5, "start_y": nan, "time_seconds": 20.0},
            ]
        )
        out = identify_corners(acts, provider=provider).sort_values("t_kick").reset_index(drop=True)
        assert out.loc[0, "corner_side"] == "right"  # rule intact: real start_y below the midline
        assert pd.isna(out.loc[1, "corner_side"])  # the fix: NaN start_y -> NaN side (was "right")


def test_derived_corner_nan_start_yields_all_nan():
    """A derived (events-only) corner has a NaN start, so ``t_kick`` / ``kick_x`` / ``kick_y`` /
    ``corner_side`` are all NaN -- ``t_kick`` is NaN, not the ordering placeholder (spec section 9.4)."""
    nan = float("nan")
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": nan, "start_y": nan, "time_seconds": 100.0}])
    out = identify_corners(acts, provider="skillcorner")
    row = out.iloc[0]
    assert pd.isna(row["t_kick"])  # NOT the ordering placeholder (100.0)
    assert pd.isna(row["kick_x"])
    assert pd.isna(row["kick_y"])
    assert pd.isna(row["corner_side"])
    assert row["corner_source"] == "skillcorner_marker"


def test_nan_t_kick_through_add_corner_windows_is_nan_row_with_reason():
    """A NaN ``t_kick`` passes through ``add_corner_windows`` as a NaN row with a reason (NaN
    ``target_snapshot_source``), never a crash and never a fabricated ``kick_plus_2s`` fallback (section 9.4)."""
    from silly_kicks.corners import add_corner_windows

    nan = float("nan")
    acts = _actions([{"type_id": _CORNER_CROSSED, "start_x": nan, "start_y": nan, "time_seconds": 100.0}])
    corners = identify_corners(acts, provider="skillcorner")
    assert pd.isna(corners.iloc[0]["t_kick"])  # precondition: a derived corner
    frames = pd.DataFrame(
        {c: pd.Series(dtype="object") for c in ("frame_id", "period_id", "time_seconds", "player_id", "team_id")}
    )
    contacts = pd.DataFrame(
        {c: pd.Series(dtype="object") for c in ("period_id", "frame_id", "time_seconds", "player_id")}
    )
    out = add_corner_windows(corners, frames, contacts)
    assert len(out) == 1
    assert pd.isna(out.iloc[0]["t_target"])
    assert pd.isna(out.iloc[0]["target_snapshot_source"])  # the reason: no resolved kick, not kick_plus_2s
