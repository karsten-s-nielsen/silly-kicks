"""Unit tests for TF-16 xShotOccurrence (xS)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking import _geometry as geo
from silly_kicks.tracking import _xshot_occurrence as xs

_FIELD_LENGTH = 105.0


def test_to_goal_relative_x_home_goal_no_flip():
    # Defending goal at x=0 (home GK end): coords pass through unchanged.
    assert geo.to_goal_relative_x(30.0, goal_x=0.0) == pytest.approx(30.0)
    assert geo.to_goal_relative_vx(2.0, goal_x=0.0) == pytest.approx(2.0)


def test_to_goal_relative_x_away_goal_flips():
    # Defending goal at x=105 (away GK end): x -> 105 - x, vx -> -vx.
    assert geo.to_goal_relative_x(30.0, goal_x=105.0) == pytest.approx(_FIELD_LENGTH - 30.0)
    assert geo.to_goal_relative_vx(2.0, goal_x=105.0) == pytest.approx(-2.0)


def test_to_goal_relative_nan_propagates():
    assert np.isnan(geo.to_goal_relative_x(np.nan, goal_x=0.0))
    assert np.isnan(geo.to_goal_relative_vx(np.nan, goal_x=105.0))


def test_geometry_exposes_pitch_constants():
    """PR-S80: metadata template + TF-38 coordinate guard depend on these constants."""
    assert geo.PITCH_LENGTH == 105.0
    assert geo.PITCH_WIDTH == 68.0
    assert isinstance(geo.GEOMETRY_VERSION, str) and geo.GEOMETRY_VERSION


def test_default_carrier_params_are_shared_constant():
    """xS sources carrier defaults from the single library constant (anti-drift; PR-S80 L1)."""
    import inspect

    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS, infer_ball_carrier

    # xS uses the SAME object, not a re-hardcoded copy.
    assert xs._DEFAULT_CARRIER_PARAMS is DEFAULT_CARRIER_PARAMS
    # The constant carries the 4.7.0 calibrated values.
    assert DEFAULT_CARRIER_PARAMS == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
    # infer_ball_carrier's signature defaults equal the constant (drift guard).
    sig = inspect.signature(infer_ball_carrier).parameters
    assert sig["tolerance_m"].default == DEFAULT_CARRIER_PARAMS["tolerance_m"]
    assert sig["beta"].default == DEFAULT_CARRIER_PARAMS["beta"]
    assert sig["gamma"].default == DEFAULT_CARRIER_PARAMS["gamma"]


# --- Task 2: openGoal obstruction helper ---
# Goal mouth: y in [30.34, 37.66] at x=0 (defended goal). Ball is goal-relative.


def test_open_goal_no_defenders_is_one():
    # No defenders between ball and goal -> fully open.
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=np.empty((0, 2)))
    assert val == pytest.approx(1.0)


def test_open_goal_defender_behind_ball_no_shadow():
    # Defender farther from goal than the ball (x > ball_x) casts no shadow.
    defenders = np.array([[25.0, 34.0]])  # behind ball (ball at x=20)
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(1.0)


def test_open_goal_defender_past_goal_line_no_shadow():
    # Defender beyond the goal line (x < 0) casts no shadow.
    defenders = np.array([[-1.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(1.0)


def test_open_goal_central_wall_reduces():
    # A defender on the ball->goal-centre line obstructs a central chunk: 0 < open < 1.
    defenders = np.array([[10.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert 0.0 < val < 1.0


def test_open_goal_overlapping_shadows_unioned():
    # Two COINCIDENT defenders cast the SAME shadow. Union of identical intervals
    # == the interval, so open is UNCHANGED vs one defender. If the impl summed
    # obstructed lengths (double-counting the overlap), open would shrink. This
    # isolates union-not-sum without the goal-line projection subtlety (a small
    # lateral offset at the defender doubles at the goal line, so offset
    # defenders genuinely widen the union --- not a valid "near-duplicate").
    d1 = np.array([[10.0, 34.0]])
    both = np.array([[10.0, 34.0], [10.0, 34.0]])  # exact duplicate
    open_one = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=d1)
    open_both = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=both)
    assert open_both == pytest.approx(open_one, abs=1e-9)


def test_open_goal_offset_shadows_widen_union():
    # Two laterally-offset defenders cast DISTINCT (partially overlapping) shadows
    # -> union is wider than one -> open is strictly smaller (but not as small as
    # naive summation, which the union test above rules out).
    d1 = np.array([[10.0, 34.0]])
    both = np.array([[10.0, 34.0], [10.0, 34.6]])
    open_one = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=d1)
    open_both = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=both)
    assert open_both < open_one


def test_open_goal_full_wall_is_zero():
    # A dense wall of defenders spanning the whole mouth, just in front of the
    # ball, drives open -> 0.0. Place 30 defenders across y just ahead of the goal.
    ys = np.linspace(28.0, 40.0, 30)
    defenders = np.column_stack([np.full(30, 3.0), ys])  # all at x=3 (between ball@x=20 and goal)
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(0.0, abs=1e-6)


def test_open_goal_grazing_angle():
    # Ball at an extreme wide angle to the goal -> finite, in-bounds (no blow-up).
    val = xs._open_goal_fraction(ball=(5.0, 5.0), defenders=np.array([[3.0, 20.0]]))
    assert np.isnan(val) or (0.0 <= val <= 1.0)


def test_open_goal_bounds_property():
    # openGoal in [0,1] for random configs.
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(0, 6))
        defs = np.column_stack([rng.uniform(-5, 25, n), rng.uniform(20, 48, n)])
        val = xs._open_goal_fraction(ball=(rng.uniform(5, 30), rng.uniform(25, 43)), defenders=defs)
        assert 0.0 <= val <= 1.0 or np.isnan(val)


def test_open_goal_golden_master_single_defender():
    # FIRST-PRINCIPLES reference (R5), not copied from implementation output:
    # Goal mouth y in [30.34, 37.66] (width 7.32) at x=0. Ball at (20, 34).
    # One defender (radius 0.375 m) centred at (10, 34) -- exactly between ball
    # and goal centre. Tangent lines from the ball graze the circle at angular
    # half-width asin(r / d_bd) about the ball->defender bearing, where
    # d_bd = 10. The shadow on the goal line (x=0) spans where those two tangent
    # rays cross x=0. Ball->defender points straight along -x (bearing pi). The
    # two tangents make angle +/- asin(0.375/10) = +/-0.0375 rad with that axis.
    # Ball is 20 m from goal line; each tangent hits x=0 at
    # y = 34 -/+ 20 * tan(0.0375) = 34 -/+ 0.7505 -> shadow ~ [33.25, 34.75],
    # width ~ 1.501 m. Open fraction = 1 - 1.501/7.32 = 0.7950.
    defenders = np.array([[10.0, 34.0]])
    val = xs._open_goal_fraction(ball=(20.0, 34.0), defenders=defenders)
    assert val == pytest.approx(0.7950, abs=0.01)


# --- Task 3: faithful feature extractor ---


def _one_frame():
    """Minimal single-frame DataFrame: ball + GK + 2 defenders + 2 attackers."""
    import pandas as pd

    rows = [
        # is_ball, is_goalkeeper, team_id, x, y, vx, vy
        dict(is_ball=True, is_goalkeeper=False, team_id=-1, x=20.0, y=34.0, vx=3.0, vy=0.0),
        dict(is_ball=False, is_goalkeeper=True, team_id=1, x=2.0, y=34.0, vx=0.0, vy=0.0),  # defending GK
        dict(is_ball=False, is_goalkeeper=False, team_id=1, x=10.0, y=30.0, vx=0.0, vy=0.0),  # defender
        dict(is_ball=False, is_goalkeeper=False, team_id=1, x=12.0, y=38.0, vx=0.0, vy=0.0),  # defender
        dict(is_ball=False, is_goalkeeper=False, team_id=2, x=18.0, y=33.0, vx=1.0, vy=0.0),  # attacker
        dict(is_ball=False, is_goalkeeper=False, team_id=2, x=22.0, y=36.0, vx=1.0, vy=0.0),  # attacker
    ]
    df = pd.DataFrame(rows)
    df["game_id"] = 1
    df["period_id"] = 1
    df["frame_id"] = 100
    df["z"] = 0.0
    return df


def test_extract_features_faithful_shape():
    out = xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0)
    assert list(out.columns) == xs.XSHOT_FEATURE_NAMES_FAITHFUL
    assert len(out) == 1
    assert len(xs.XSHOT_FEATURE_NAMES_FAITHFUL) == 27


def test_extended_raises_not_implemented():
    with pytest.raises(NotImplementedError, match="extended"):
        xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0, feature_set="extended")


def test_extract_features_goal_relative_symmetry():
    import pandas as pd

    # Same scene mirrored to the other end must yield identical features.
    f0 = _one_frame()
    f1 = f0.copy()
    f1["x"] = 105.0 - f1["x"]
    f1["vx"] = -f1["vx"]
    a = xs.extract_xshot_features(f0, gk_team_id=1, goal_x=0.0)
    b = xs.extract_xshot_features(f1, gk_team_id=1, goal_x=105.0)
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9)


def test_fewer_than_5_players_nan_slots():
    out = xs.extract_xshot_features(_one_frame(), gk_team_id=1, goal_x=0.0)
    # Only 2 defenders present -> DefDist_2..4 are NaN.
    assert np.isnan(out["DefDist_2"].iloc[0])
    assert np.isnan(out["DefAngle_4"].iloc[0])
    assert not np.isnan(out["DefDist_0"].iloc[0])


# --- Task 4: label builder (time_seconds window, no linkage) ---


def test_label_horizon_via_time_seconds():
    import pandas as pd

    # 3 in-possession frames (team 2) at t=0.0, 0.5, 1.0 in period 1; one shot at t=1.2.
    fidx = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [0.0, 0.5, 1.0],
            "team_in_possession": [2, 2, 2],
        }
    )
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [1.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    # t=0.0: shot at 1.2 is >1.0 ahead -> 0; t=0.5: 1.2-0.5=0.7 <=1 ->1; t=1.0: 0.2 ->1
    assert list(y) == [0, 1, 1]


def test_label_robust_to_noncontiguous_frame_id():
    import pandas as pd

    # Same times as above but frame_id has a huge gap -- label must be identical
    # (proves no frame_id arithmetic).
    fidx = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "frame_id": [10, 9999, 10000],
            "time_seconds": [0.0, 0.5, 1.0],
            "team_in_possession": [2, 2, 2],
        }
    )
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [1.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0, 1, 1]


def test_label_no_period_bleed():
    import pandas as pd

    # Frame at end of P1; shot just after at start of P2 -- must NOT label P1 frame positive.
    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [45.0], "team_in_possession": [2]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [2], "team_id": [2], "time_seconds": [45.2]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0]


def test_label_inclusive_of_t():
    import pandas as pd

    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [10.0], "team_in_possession": [2]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [2], "time_seconds": [10.0]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [1]


def test_label_turnover_opponent_shot_is_negative():
    import pandas as pd

    # Frame: team 2 in possession at t=5.0. Opponent (team 1) shoots at t=5.5.
    fidx = pd.DataFrame({"game_id": [1], "period_id": [1], "time_seconds": [5.0], "team_in_possession": [2]})
    shots = pd.DataFrame({"game_id": [1], "period_id": [1], "team_id": [1], "time_seconds": [5.5]})
    y = xs.build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    assert list(y) == [0]


# --- Task 5: XShotOccurrenceModel ---


def _toy_xy(n=400, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 27)), columns=xs.XSHOT_FEATURE_NAMES_FAITHFUL)
    # Label correlated with first feature so the model can learn something.
    y = (X["r"] + rng.normal(scale=0.5, size=n) < 0).astype(int)
    return X, pd.Series(y)


def test_model_fit_predict_proba():
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    p = m.predict_proba(X)
    assert p.shape == (len(X),)
    assert np.all((p >= 0) & (p <= 1))


def test_model_deterministic():
    X, y = _toy_xy()
    p1 = xs.XShotOccurrenceModel(params={"random_state": 42}).fit(X, y).predict_proba(X)
    p2 = xs.XShotOccurrenceModel(params={"random_state": 42}).fit(X, y).predict_proba(X)
    np.testing.assert_array_equal(p1, p2)


def test_model_save_load_roundtrip(tmp_path):
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y, carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0})
    m.save(tmp_path / "xs_v1")
    loaded = xs.XShotOccurrenceModel.load(tmp_path / "xs_v1")
    np.testing.assert_allclose(loaded.predict_proba(X), m.predict_proba(X), rtol=1e-9)
    assert loaded.carrier_params == {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
    assert loaded.feature_set == "faithful"


def test_model_sha256_verification(tmp_path):
    X, y = _toy_xy()
    xs.XShotOccurrenceModel().fit(X, y).save(tmp_path / "xs_v1")
    (tmp_path / "xs_v1" / "model.json").write_text("tampered")
    with pytest.raises(xs.IntegrityError):
        xs.XShotOccurrenceModel.load(tmp_path / "xs_v1")


def test_model_carrier_params_default_when_unset():
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    # PR-S80: default now sourced from the shared 4.7.0 constant.
    assert m.carrier_params == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}


def test_fit_sets_base_score_to_positive_rate():
    """N4: calibration must not silently depend on xgboost's auto-intercept."""
    import json

    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    cfg = json.loads(m._booster.save_config())
    # xgboost serializes base_score version-dependently: a plain "0.525" on some versions,
    # a bracketed array string "[5.25E-1]" on others -- strip the brackets before parsing.
    base = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
    assert abs(base - float(y.mean())) < 1e-6


def test_fit_does_not_reweight():
    """P2/M2: the SHIPPED model must be unweighted (scale_pos_weight == 1), asserted on the
    fitted booster — not merely absent from the search space."""
    import json
    import re

    assert "scale_pos_weight" not in xs._pinned_params(None)
    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    cfg = json.dumps(json.loads(m._booster.save_config()))
    # Value may be plain ("1") or a bracketed array string ("[1E0]") depending on xgboost version.
    found = re.findall(r'"scale_pos_weight":\s*"?\[?([0-9.eE+-]+)\]?"?', cfg)
    assert all(abs(float(v) - 1.0) < 1e-9 for v in found), f"model reweights: {found}"


def test_metadata_records_pitch_and_platform(tmp_path):
    import json

    X, y = _toy_xy()
    m = xs.XShotOccurrenceModel().fit(X, y)
    m.save(tmp_path / "v1")
    meta = json.loads((tmp_path / "v1" / "metadata.json").read_text())
    assert meta["pitch_length"] == 105.0 and meta["pitch_width"] == 68.0
    for k in ("geometry_version", "xgboost_version", "training_platform"):
        assert k in meta


def test_load_raises_on_pitch_dimension_mismatch(tmp_path, monkeypatch):
    """M4: a rescale/unit change genuinely skews features -> fail closed, never warn."""
    X, y = _toy_xy()
    xs.XShotOccurrenceModel().fit(X, y).save(tmp_path / "v1")
    monkeypatch.setattr(xs._geo, "PITCH_LENGTH", 100.0)
    with pytest.raises((xs.IntegrityError, ValueError)):
        xs.XShotOccurrenceModel.load(tmp_path / "v1")


def test_load_warns_on_geometry_version_only(tmp_path, monkeypatch):
    """Pure-representation change at identical pitch dims is invariant -> warn, not raise."""
    import warnings

    X, y = _toy_xy()
    xs.XShotOccurrenceModel().fit(X, y).save(tmp_path / "v1")
    monkeypatch.setattr(xs._geo, "GEOMETRY_VERSION", "goal-relative-2")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        xs.XShotOccurrenceModel.load(tmp_path / "v1")  # must NOT raise
    assert any("geometry_version" in str(x.message).lower() for x in w)


def test_from_variant_loads_bundled_dir(tmp_path, monkeypatch):
    """from_variant loads a bundled dir (memoized); unknown variant raises."""
    root = tmp_path / "_xshot_weights"
    (root / "default").mkdir(parents=True)
    X, y = _toy_xy()
    xs.XShotOccurrenceModel().fit(X, y).save(root / "default")
    monkeypatch.setattr(xs, "_XSHOT_WEIGHTS_ROOT", root)
    monkeypatch.setattr(xs, "_VARIANT_CACHE", {})  # honour the temp root, not a cached instance

    m = xs.XShotOccurrenceModel.from_variant("default")
    assert m._booster is not None
    # memoized: a second call returns the SAME instance
    assert xs.XShotOccurrenceModel.from_variant("default") is m
    with pytest.raises(FileNotFoundError):
        xs.XShotOccurrenceModel.from_variant("does-not-exist")


def test_directional_fixture_has_both_classes_and_schema():
    """The committed frozen directional rows (CI liveness tripwire data) are well-formed."""
    import pandas as pd

    df = pd.read_parquet("tests/datasets/tracking/xshot_directional/frozen_rows.parquet")
    assert set(xs.XSHOT_FEATURE_NAMES_FAITHFUL).issubset(df.columns)
    assert "label" in df.columns
    assert df["label"].nunique() == 2
    assert df["label"].sum() >= 3 and (df["label"] == 0).sum() >= 3


def test_xshot_surface_home_team_id_optional():
    """T7: home_team_id is unused (GK-based goal resolution) -> callers may omit it, so the
    factory can sit in a module-level default xfn list."""
    frames = _synthetic_match_frames(n_frames=2)
    m = xs.XShotOccurrenceModel().fit(*_toy_xy())
    out = xs.compute_xshot_occurrence(frames, model=m)  # no home_team_id
    assert "xshot_occurrence" in out.columns
    xfns = xs.xshot_occurrence_xfns()  # buildable with no args
    assert len(xfns) == 1 and getattr(xfns[0], "_frame_aware", False) is True


# --- Task 6: compute_xshot_occurrence ---


def _synthetic_match_frames(n_frames=20, game_id=1):
    """Full-schema tracking frames: ball + 11v11 across n_frames in one period.

    Team 1 defends goal x=0 (GK near x=2); team 2 defends x=105 (GK near x=103).
    The ball sits next to a team-2 outfielder near x=20, so team 2 is in
    possession and attacks team 1's goal (goal_x=0).
    """
    import pandas as pd

    rows = []
    for fi in range(n_frames):
        t = fi * 0.04
        # ball next to a team-2 attacker
        rows.append(
            dict(
                player_id=-1,
                team_id=-1,
                is_ball=True,
                is_goalkeeper=False,
                x=20.0,
                y=34.0,
                vx=2.0,
                vy=0.0,
                speed=2.0,
                frame_id=fi,
                time_seconds=t,
            )
        )
        # team 1 (defending, goal x=0): GK + 10 outfield in own half
        rows.append(
            dict(
                player_id=10,
                team_id=1,
                is_ball=False,
                is_goalkeeper=True,
                x=2.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                speed=0.0,
                frame_id=fi,
                time_seconds=t,
            )
        )
        for k in range(10):
            rows.append(
                dict(
                    player_id=11 + k,
                    team_id=1,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=8.0 + k,
                    y=20.0 + 2 * k,
                    vx=0.0,
                    vy=0.0,
                    speed=0.0,
                    frame_id=fi,
                    time_seconds=t,
                )
            )
        # team 2 (attacking): GK at x=103 + carrier near ball + 9 others
        rows.append(
            dict(
                player_id=20,
                team_id=2,
                is_ball=False,
                is_goalkeeper=True,
                x=103.0,
                y=34.0,
                vx=0.0,
                vy=0.0,
                speed=0.0,
                frame_id=fi,
                time_seconds=t,
            )
        )
        rows.append(
            dict(
                player_id=21,
                team_id=2,
                is_ball=False,
                is_goalkeeper=False,
                x=20.3,
                y=34.0,
                vx=2.0,
                vy=0.0,
                speed=2.0,
                frame_id=fi,
                time_seconds=t,
            )
        )
        for k in range(9):
            rows.append(
                dict(
                    player_id=22 + k,
                    team_id=2,
                    is_ball=False,
                    is_goalkeeper=False,
                    x=25.0 + k,
                    y=20.0 + 2 * k,
                    vx=0.0,
                    vy=0.0,
                    speed=0.0,
                    frame_id=fi,
                    time_seconds=t,
                )
            )
    df = pd.DataFrame(rows)
    df["game_id"] = game_id
    df["period_id"] = 1
    df["z"] = 0.0
    df["frame_rate"] = 25.0
    df["ball_state"] = "alive"
    df["speed_source"] = "native"
    df["team_attacking_direction"] = "ltr"
    df["confidence"] = None
    df["visibility"] = None
    df["source_provider"] = "synthetic"
    df["is_goalkeeper_source"] = "native"
    return df


def test_compute_xshot_model_none_uses_bundled_default():
    # PR-S80: weights now ship, so model=None resolves to the bundled "default" variant
    # (from_variant("default")) instead of raising. compute runs and adds the column.
    frames = _synthetic_match_frames(n_frames=2)
    out = xs.compute_xshot_occurrence(frames, model=None, home_team_id=1)
    assert "xshot_occurrence" in out.columns


def test_inference_uses_metadata_carrier_params(monkeypatch):
    # R3: a model carrying non-default carrier params must drive infer_ball_carrier
    # with THOSE params, not the library defaults.
    captured = {}
    real = xs.infer_ball_carrier

    def spy(frames, *, tolerance_m, beta, gamma):
        captured["params"] = (tolerance_m, beta, gamma)
        return real(frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma)

    monkeypatch.setattr(xs, "infer_ball_carrier", spy)
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y, carrier_params={"tolerance_m": 9.0, "beta": 0.1, "gamma": 0.2})
    frames = _synthetic_match_frames(n_frames=3)
    xs.compute_xshot_occurrence(frames, model=model, home_team_id=1)
    assert captured["params"] == (9.0, 0.1, 0.2)


def test_compute_carrier_runs_on_full_frames(monkeypatch):
    # N-A: with link_frame_ids set, infer_ball_carrier must STILL receive the full
    # frame set (cross-frame hysteresis correctness + train/serve parity).
    seen = {}
    real = xs.infer_ball_carrier

    def spy(frames, **kw):
        seen["n_rows"] = len(frames)
        return real(frames, **kw)

    monkeypatch.setattr(xs, "infer_ball_carrier", spy)
    frames = _synthetic_match_frames(n_frames=40)
    total_rows = len(frames)
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    link_ids = set(int(f) for f in frames["frame_id"].unique()[:3])  # only 3 of 40
    xs.compute_xshot_occurrence(frames, model=model, home_team_id=1, link_frame_ids=link_ids)
    assert seen["n_rows"] == total_rows


def test_compute_preserves_id_dtypes():
    # N-B: returned frames keep original game_id/team_id dtypes (no schema mutation).
    frames = _synthetic_match_frames(n_frames=20)
    gid_dtype, tid_dtype = frames["game_id"].dtype, frames["team_id"].dtype
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=1)
    assert out["game_id"].dtype == gid_dtype
    assert out["team_id"].dtype == tid_dtype
    assert "xshot_occurrence" in out.columns


def test_compute_scores_possessing_team():
    # The in-possession team (team 2 here) gets xS values; defenders stay NaN.
    frames = _synthetic_match_frames(n_frames=10)
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=1)
    team2 = out[(out["team_id"] == 2) & (~out["is_ball"])]
    team1 = out[(out["team_id"] == 1) & (~out["is_ball"])]
    assert team2["xshot_occurrence"].notna().any()
    assert team1["xshot_occurrence"].isna().all()


def test_compute_xshot_one_batched_predict_per_match():
    # P1 STRUCTURAL guard (not wall-clock): the whole match must be scored with a
    # SINGLE batched predict_proba, never one-per-frame. Spying on predict_proba
    # and asserting exactly one call pins the exact regression P1 fixed and can
    # NEVER flake on a slow CI runner -- per feedback_windows_ci_perf_budget, a
    # wall-clock ceiling on a shared runner is fundamentally noisy (this very test
    # failed CI at 501ms vs a 500ms ceiling), so a structural proxy is strictly
    # better when one exists.
    from unittest.mock import patch

    frames = _synthetic_match_frames(n_frames=100)  # 100 scored frames
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    real_predict = model.predict_proba
    with patch.object(model, "predict_proba", wraps=real_predict) as spy:
        out = xs.compute_xshot_occurrence(frames, model=model, home_team_id=1)
    assert "xshot_occurrence" in out.columns
    assert out["xshot_occurrence"].notna().any()
    # ONE batched predict for the whole match -- a regression to per-frame predict
    # would make this ~100.
    assert spy.call_count == 1, (
        f"expected 1 batched predict_proba for the match, got {spy.call_count} (per-frame predict regression?)"
    )


# --- Task 7: add_xshot_occurrence ---


def _actions_and_frames_for_add():
    """SPADL-ish actions linked to synthetic frames; team 2 is in possession."""
    import pandas as pd

    frames = _synthetic_match_frames(n_frames=10)
    # One action by team 2 (possessing) at a real frame's time.
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "team_id": [2, 2],
            "player_id": [21, 21],
            "time_seconds": [0.0, 0.2],
        }
    )
    return actions, frames


def _other_team(actions, frames):
    """Return a defending team id (not the possessing team 2)."""
    return 1


def test_add_xshot_nan_safe_marker():
    from silly_kicks._nan_safety import is_nan_safe_enrichment

    assert is_nan_safe_enrichment(xs.add_xshot_occurrence)


def test_add_xshot_adds_column():
    actions, frames = _actions_and_frames_for_add()
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id=1)
    assert "xshot_occurrence" in out.columns
    assert out["xshot_occurrence"].dtype.kind == "f"
    assert len(out) == len(actions)


def test_add_xshot_dtype_mismatch():
    # P2: int64 actions.team_id + object frames.team_id must not silently miss the join.
    actions, frames = _actions_and_frames_for_add()
    frames = frames.copy()
    frames["team_id"] = frames["team_id"].astype(str)
    frames["game_id"] = frames["game_id"].astype(str)
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id="1")
    assert out["xshot_occurrence"].notna().any()


def test_add_xshot_defensive_action_is_nan():
    # S1: an action by the NON-possessing team at a scored frame gets NaN by design.
    import pandas as pd  # noqa: F401

    actions, frames = _actions_and_frames_for_add()
    actions = actions.copy()
    defending_team = _other_team(actions, frames)
    actions.loc[actions.index[0], "team_id"] = defending_team
    actions.loc[actions.index[0], "player_id"] = 11  # a team-1 outfielder
    X, y = _toy_xy()
    model = xs.XShotOccurrenceModel().fit(X, y)
    out = xs.add_xshot_occurrence(actions, frames, model=model, home_team_id=1)
    assert np.isnan(out.loc[out.index[0], "xshot_occurrence"])


# --- Task 8: xshot_occurrence_xfns ---


def test_xshot_xfns_frame_aware_marker():
    fns = xs.xshot_occurrence_xfns(model=None, home_team_id=1)
    assert len(fns) == 1
    assert getattr(fns[0], "_frame_aware", False) is True


def test_xshot_xfns_introspection_nan():
    import pandas as pd

    fns = xs.xshot_occurrence_xfns(model=None, home_team_id=1)
    states = [pd.DataFrame({"action_id": [1, 2]}) for _ in range(3)]
    out = fns[0](states, None)
    assert list(out.columns) == [
        "xshot_occurrence_a0",
        "xshot_occurrence_a1",
        "xshot_occurrence_a2",
    ]
    assert out.isna().all().all()


# --- Task 13: public exports + dependency-light import ---


def test_public_exports():
    import silly_kicks.tracking as t

    for name in [
        "compute_xshot_occurrence",
        "add_xshot_occurrence",
        "xshot_occurrence_xfns",
        "XShotOccurrenceModel",
        "XShotFeatureSet",
        "extract_xshot_features",
    ]:
        assert hasattr(t, name), name


def test_import_silly_kicks_no_xgboost():
    # P3: dependency-light import must not pull xgboost at top level. Use a FRESH
    # SUBPROCESS (the established idiom, tests/calibration/test_import_isolation.py),
    # NOT in-process importlib.reload (submodules stay cached; sys.modules fiddling
    # is flagged unreliable by this project).
    #
    # Contract scope: `import silly_kicks` (the dependency-light guarantee that
    # matters to consumers). The xS module keeps every `import xgboost`
    # function-local, so it adds NO new leak. NOTE: `import silly_kicks.tracking`
    # already pulls xgboost transitively on clean main (verified by git-stash) --
    # a pre-existing condition out of scope for this PR; asserting on it would
    # fail on a fault not introduced here.
    import subprocess
    import sys

    code = (
        "import sys; import silly_kicks; "
        "bad=[m for m in ('xgboost',) if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"import leaked xgboost: {proc.stdout.strip()}"


def test_xshot_module_xgboost_is_lazy():
    # The xS module itself must not import xgboost at module level (it keeps all
    # `import xgboost` function-local). Importing JUST the module (bypassing the
    # tracking __init__'s pre-existing transitive xgboost pull) stays clean.
    import subprocess
    import sys

    code = (
        "import sys, importlib.util; "
        "spec = importlib.util.find_spec('silly_kicks.tracking._xshot_occurrence'); "
        # Load the module's source and check no top-level `import xgboost`.
        "src = open(spec.origin, encoding='utf-8').read(); "
        "lines = [ln for ln in src.splitlines() if ln.strip() == 'import xgboost as xgb']; "
        # All xgboost imports must be indented (function-local), never at column 0.
        "bad = [ln for ln in lines if not ln.startswith(' ')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, f"xgboost imported at module level: {proc.stdout.strip()}"


def test_build_xshot_labels_bit_identical_after_refactor():
    """R2-L2: refactoring build_xshot_labels onto _build_occurrence_labels must not shift xS
    labels. frames_index uses the real `team_in_possession` schema (NOT team_id). Golden frozen
    from the pre-refactor implementation output."""
    import pandas as pd

    from silly_kicks.tracking._xshot_occurrence import build_xshot_labels

    frames_index = pd.DataFrame(
        {
            "game_id": ["g"] * 5,
            "period_id": [1] * 5,
            "time_seconds": [0.0, 0.5, 1.0, 1.5, 2.0],
            "team_in_possession": ["A"] * 5,  # real xS column (line 299), NOT team_id
        }
    )
    shots = pd.DataFrame({"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.2]})
    y = build_xshot_labels(frames_index, shots, horizon_seconds=1.0)
    np.testing.assert_array_equal(np.asarray(y), np.array([0, 1, 1, 0, 0]))  # GOLDEN (frozen pre-refactor)
    assert np.asarray(y).dtype == np.dtype(int)
