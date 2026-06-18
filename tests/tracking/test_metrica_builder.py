"""tracking.metrica.convert_to_frames --- bronze->canonical frame builder."""

import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import metrica as mt
from silly_kicks.tracking.schema import KLOPPY_TRACKING_FRAMES_COLUMNS


def _bronze(n_frames=4, p2_start=2850.0):
    """Metrica frame-level bronze: 0-1 normalized coords, JSON player columns.

    ``p2_start`` sets the raw P2 clock origin: 2850.0 mimics a CONTINUOUS sample game
    (SG1-like, NOT the nominal 2700 --- so per-period-min rebasing is distinguishable from
    a nominal-offset subtraction); pass 0.0 for a PERIOD-RELATIVE game (SG3-like).
    """
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.04 if period == 1 else p2_start + (f - 2) * 0.04
        # home GK jersey "1" near own goal (left, x~0.05) in P1; away GK jersey "2" (distinct,
        # as in real Metrica where the flat gk_jersey_numbers lists both teams' GK numbers).
        home = {"1": {"x": 0.05, "y": 0.50}, "9": {"x": 0.40, "y": 0.55}}
        away = {"2": {"x": 0.95, "y": 0.50}, "9": {"x": 0.60, "y": 0.45}}
        rows.append(
            {
                "period": period,
                "frame": f,
                "timestamp": ts,
                "ball_x": 0.50,
                "ball_y": 0.50,
                "home_players": json.dumps(home),
                "away_players": json.dumps(away),
                "gk_jersey_numbers": json.dumps(["1", "2"]),
                "frame_rate": 25,
            }
        )
    return pd.DataFrame(rows)


def _roster():
    return {"Home": {"1": "h_gk", "9": "h_fw"}, "Away": {"2": "a_gk", "9": "a_fw"}}


def _bronze_gk_collision(n_frames=4):
    """Teams have DIFFERENT GK numbers (home #1, away #16), each reusing the OTHER's GK
    number on an outfielder (home #16 outfielder, away #1 outfielder) --- the exact case a
    team-agnostic jersey.isin(gk_jersey_numbers) mis-flags."""
    rows = []
    for f in range(n_frames):
        period = 1 if f < 2 else 2
        ts = f * 0.04 if period == 1 else 2850.0 + (f - 2) * 0.04
        home = {"1": {"x": 0.04, "y": 0.50}, "16": {"x": 0.45, "y": 0.55}}  # GK #1 deep, OF #16
        away = {"16": {"x": 0.96, "y": 0.50}, "1": {"x": 0.55, "y": 0.45}}  # GK #16 deep, OF #1
        rows.append(
            {
                "period": period,
                "frame": f,
                "timestamp": ts,
                "ball_x": 0.50,
                "ball_y": 0.50,
                "home_players": json.dumps(home),
                "away_players": json.dumps(away),
                "gk_jersey_numbers": json.dumps(["1", "16"]),
                "frame_rate": 25,
            }
        )
    return pd.DataFrame(rows)


def test_rescale_0_1_to_spadl_no_flip():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    fw = frames[(frames.player_id == "h_fw") & (frames.frame_id == 0)].iloc[0]
    assert fw.x == pytest.approx(0.40 * 105.0) and fw.y == pytest.approx(0.55 * 68.0)


def test_ball_z_is_nan():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert np.isnan(frames[frames.is_ball].iloc[0].z)


def test_gk_derived_positionally():
    # GK comes from positional derivation, NOT the flat jersey list. Home GK (jersey "1",
    # deepest home player) is flagged.
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert frames[frames.player_id == "h_gk"].is_goalkeeper.all()


def test_gk_not_flagged_by_shared_jersey_number():
    # The team-agnostic bug would flag home #16 + away #1 outfielders as GK (both numbers are
    # in the flat gk_jersey_numbers). Positional derivation must flag ONLY the deep players.
    roster = {"Home": {"1": "h_gk", "16": "h_of"}, "Away": {"16": "a_gk", "1": "a_of"}}
    frames, _ = mt.convert_to_frames(
        _bronze_gk_collision(), jersey_to_player_id=roster, output_convention="absolute_frame"
    )
    gk_ids = set(frames[(~frames.is_ball) & frames.is_goalkeeper]["player_id"])
    assert gk_ids == {"h_gk", "a_gk"}, gk_ids  # NOT h_of / a_of


def test_clock_rebased_per_period_min_continuous_game():
    # Continuous raw P2 (starts 2850, NOT nominal 2700) -> per-period-min rebases to ~0.
    # A nominal-2700 subtraction would WRONGLY leave P2 at ~150.
    frames, _ = mt.convert_to_frames(
        _bronze(p2_start=2850.0), jersey_to_player_id=_roster(), output_convention="absolute_frame"
    )
    assert frames[frames.period_id == 2]["time_seconds"].min() == pytest.approx(0.0, abs=0.05)


def test_clock_rebased_per_period_min_period_relative_game():
    # Already period-relative raw P2 (starts 0) -> stays ~0 (no spurious negative times).
    frames, _ = mt.convert_to_frames(
        _bronze(p2_start=0.0), jersey_to_player_id=_roster(), output_convention="absolute_frame"
    )
    p2 = frames[frames.period_id == 2]["time_seconds"]
    assert p2.min() == pytest.approx(0.0, abs=0.05)
    assert (p2 >= -1e-6).all()  # never negative


def test_builder_does_not_iterate_rows():
    # Structural perf guard (no wall-clock assert): the vectorized shape must not iterrows
    # NOR apply(axis=1) --- both are the tracking-scale row-wise cliff.
    import pandas as _pd

    orig_iter, orig_apply = _pd.DataFrame.iterrows, _pd.DataFrame.apply

    def _boom_iter(self):
        raise AssertionError("metrica builder must not call DataFrame.iterrows (tracking-scale anti-pattern)")

    def _spy_apply(self, func, *a, **k):
        axis = k.get("axis", a[0] if a else 0)
        if axis in (1, "columns"):
            raise AssertionError("metrica builder must not call DataFrame.apply(axis=1) (row-wise cliff)")
        return orig_apply(self, func, *a, **k)

    _pd.DataFrame.iterrows = _boom_iter  # type: ignore[reportAttributeAccessIssue]
    _pd.DataFrame.apply = _spy_apply  # type: ignore[reportAttributeAccessIssue]
    try:
        mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    finally:
        _pd.DataFrame.iterrows = orig_iter
        _pd.DataFrame.apply = orig_apply


def test_output_schema_matches_kloppy_variant():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), output_convention="absolute_frame")
    assert list(frames.columns) == list(KLOPPY_TRACKING_FRAMES_COLUMNS)


def test_ltr_orientation_home_low_x_every_period():
    frames, _ = mt.convert_to_frames(_bronze(), jersey_to_player_id=_roster(), home_team_id="Home")
    hgk = frames[(~frames.is_ball) & (frames.player_id == "h_gk")]
    assert (hgk.x < 52.5).all()


def test_missing_input_column_raises():
    bad = _bronze().drop(columns=["home_players"])
    with pytest.raises(ValueError, match="home_players"):
        mt.convert_to_frames(bad, jersey_to_player_id=_roster())
