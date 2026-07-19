"""ADR-019: ghost-GK feature extraction must be dtype-safe on team ids.

Gradient Sports frames carry a nullable ``Int64`` ``team_id``; sportec/kloppy-family
providers carry object strings. A raw ``==``/``!=`` between the two silently yields an
empty defending/attacking split and an all-zero ``team_in_possession`` -- a corrupt
feature row rather than an error. Task 5's probe routes EVERY ghost position through
this extractor, so the corruption would land in the substitution measurement.

These tests assert the ADR-019 canonicalization contract: an id expressed as ``1``,
``1.0`` or ``"1"`` denotes the SAME team, so both representations must serve the same
ghost position.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking import serve_ghost_gk_positions
from silly_kicks.tracking._ghost_gk import (
    _extract_all_ghost_gk_features,
    extract_ghost_gk_features,
)
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def _stringify_team_ids(frames: pd.DataFrame) -> pd.DataFrame:
    """Re-express the fixture's float64 team ids as a string-id provider would.

    ``str(int(v))`` -- NOT ``str(v)``. The fixture column is float64, so a bare
    ``str`` would produce ``"1.0"``, which canonicalizes to ``"1.0"`` (genuine strings
    pass through ``_canonical`` unchanged) and is therefore a DIFFERENT id, not the
    same id in another dtype. That would test a rename, not a dtype mismatch.
    """
    out = frames.copy()
    out["team_id"] = out["team_id"].map(lambda v: None if pd.isna(v) else str(int(v)))
    return out


def test_carrier_team_id_of_another_dtype_still_sets_team_in_possession():
    """``team_in_possession``, via the documented public ``carrier=`` cache kwarg.

    A caller-supplied carrier table's ``ball_carrier_team_id`` need not share the
    frames' id dtype (GS emits ``Int64``; sportec/kloppy frames carry object strings).
    A raw ``==`` against an object ``gk_team_id`` is unconditionally False, so
    ``team_in_possession`` pins to 0 with no error raised.

    Two deliberate choices, both of which an "obvious" simplification would break:

    1. The carrier is supplied EXPLICITLY rather than inferred. ``_map_result``
       preserves the source ``team_id`` dtype, so the internally-inferred carrier is
       self-consistent and an inferred-carrier version of this test passes both before
       and after the fix -- proving nothing.
    2. The assertion is on the extracted FEATURE ROWS, not on served positions.
       ``_extract_all_ghost_gk_features`` is the exact seam ``_serve_positions_core``
       calls, and a feature difference is deterministic. Served positions are not a
       usable signal here: the shared 10-estimator synthetic fixture model does not
       split on ``team_in_possession``, so it maps both feature rows to the same
       output. Asserting positions would be a test that cannot fail.
    """
    numeric = _make_ghost_gk_frames()
    stringy = _stringify_team_ids(numeric)

    # Away team (2) carries the ball -- so the AWAY GK's row is the one that flips.
    carrier = pd.DataFrame(
        {
            "game_id": ["100"],
            "period_id": [1],
            "frame_id": [1],
            "ball_carrier_team_id": pd.array([2], dtype="Int64"),
        }
    )

    feats_num, meta_num = _extract_all_ghost_gk_features(numeric, home_team_id=1, carrier=carrier)
    feats_str, meta_str = _extract_all_ghost_gk_features(stringy, home_team_id="1", carrier=carrier)

    assert len(meta_num) == len(meta_str) == 2, "expected one row per GK (home + away)"
    assert feats_num["team_in_possession"].tolist() == [0.0, 1.0], (
        "reference path lost team_in_possession -- fixture or carrier-key regression"
    )
    pd.testing.assert_frame_equal(
        feats_num.reset_index(drop=True),
        feats_str.reset_index(drop=True),
    )


def test_public_serving_seam_accepts_string_team_ids():
    """End-to-end smoke on the public seam: string ids must not shrink the output.

    Complements the feature-level assertions above -- it pins the row count and keys
    that Task 5's probe joins on, which a broken split would silently change.
    """
    numeric = _make_ghost_gk_frames()
    stringy = _stringify_team_ids(numeric)

    a = serve_ghost_gk_positions(numeric, model=_fitted_model()[0], home_team_id=1)
    b = serve_ghost_gk_positions(stringy, model=_fitted_model()[0], home_team_id="1")

    assert len(a) == len(b) > 0, "string-id path produced a different row count"
    assert a["ghost_gr_x"].notna().all() and b["ghost_gr_x"].notna().all()


def test_extractor_splits_survive_a_scalar_gk_team_id_of_another_dtype():
    """The extractor itself: a numeric ``gk_team_id`` against string frame ids.

    ``extract_ghost_gk_features`` takes ``gk_team_id`` as a caller-supplied scalar, so
    the two sides are NOT guaranteed to share a dtype. A raw compare empties the
    defending split -- ``defensive_line_x`` and friends go NaN and
    ``attackers_in_box`` counts the whole pitch -- with no error raised.
    """
    numeric = _make_ghost_gk_frames()
    stringy = _stringify_team_ids(numeric)

    ref = extract_ghost_gk_features(numeric, gk_team_id=1.0, goal_x=0.0)
    mixed = extract_ghost_gk_features(stringy, gk_team_id=1, goal_x=0.0)

    assert ref["defensive_line_x"].notna().all(), "reference row is degenerate -- fixture regression"
    pd.testing.assert_frame_equal(ref, mixed)


# --- score-lookup team identity (ADR-043) ----------------------------------------------


def _goal_actions(team_dtype: str):
    """Three goals in one game: team 1 scores twice, team 2 once -> home score_diff = +1."""
    from silly_kicks.spadl import config as cfg

    df = pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "time_seconds": [10.0, 20.0, 30.0],
            "team_id": [1, 1, 2],
            "type_id": [cfg.actiontypes.index("shot")] * 3,
            "result_id": [cfg.results.index("success")] * 3,
        }
    )
    df["team_id"] = df["team_id"].astype(team_dtype)
    return df


@pytest.mark.parametrize("team_dtype", ["int64", "Int64", "float64", "string"])
@pytest.mark.parametrize("home_scalar", [1, "1", 1.0])
def test_score_lookup_is_id_dtype_invariant(team_dtype, home_scalar):
    """`_build_score_lookup` must agree on team identity across id representations.

    It previously compared `str(t) == str(home_team_id)`, which renders a float-backed id as
    "1.0" against a scalar "1". EVERY goal then fell to the away side: this fixture returned
    **-3 instead of +1**, a four-goal swing on `score_diff` -- one of the 26 TRAINED ghost-GK
    features. Renaming the scalar to `home_team_id_norm` also hid it from the AST lint.
    """
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    fn = _build_score_lookup(_goal_actions(team_dtype), home_team_id=home_scalar)
    assert fn(1, 40.0) == 1.0, f"score_diff wrong for column={team_dtype}, scalar={home_scalar!r}"


def test_score_lookup_fixture_would_expose_an_all_away_classification():
    """Non-vacuity: the fixture must DISCRIMINATE, not merely return something.

    If every goal were classified away the answer would be -3, and if every goal were
    classified home it would be +3. Both differ from the correct +1, so a broken
    implementation cannot pass the test above by accident.
    """
    from silly_kicks.tracking._ghost_gk import _build_score_lookup

    correct = _build_score_lookup(_goal_actions("int64"), home_team_id=1)(1, 40.0)
    assert correct == 1.0
    assert correct not in (-3.0, 3.0), "fixture cannot distinguish correct from degenerate"
