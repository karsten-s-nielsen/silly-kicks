"""select_gkdv_scored_frames + build_ghost_frames(scored_frame_keys=): inject a whole-unit
stride survivor set for windowed Stage-2 scoring (ADR-119).

The gkdv ``possession_stride`` (``GkdvParams.possession_stride=5``) keeps every Nth *eligible*
frame, ranked by a ``cumcount`` over the globally-ordered eligible frames of each
``(game_id, period_id)`` (``_engine._apply_domain``). That rank is a WHOLE-UNIT quantity: window
the scoring and the eligible set the rank runs over shrinks, so the phase shifts and a DIFFERENT
set of frames survives. Injecting a whole-unit GoalMap + carrier (the 4.132 seam) does not fix it --
the stride rank is independent of both.

``select_gkdv_scored_frames`` exposes the whole-unit survivor selection (cheap -- no pitch-control /
DAS grids), and ``build_ghost_frames(scored_frame_keys=...)`` accepts it, replacing the internal
stride with "keep the eligible frames whose key is in the injected set". ``scored_frame_keys=None``
keeps the current internal stride -> byte-identical for every existing caller.

These tests pin: the default path is unchanged (None byte-identical), the selection equals the
engine's internal one, an injected subset scores exactly that subset (intersected with the per-frame
domain), and -- the end-to-end proof -- a union of windows scored under one injected survivor set is
byte-identical to the whole-unit scoring.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.gkdv import build_ghost_frames, select_gkdv_scored_frames
from silly_kicks.tracking import infer_ball_carrier, resolve_defended_goals
from tests.gkdv._fixtures import multi_frame_in_domain
from tests.tracking.test_ghost_gk import _fitted_model

_FRAME_KEYS = ["game_id", "period_id", "frame_id"]


def _build(frames: pd.DataFrame, **kwargs):
    return build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1, **kwargs)


def _scored_keys(prov: pd.DataFrame) -> pd.DataFrame:
    """The scored (non-dropped) frame keys of a provenance frame, sorted and de-duplicated."""
    scored = prov.loc[prov["drop_reason"].isna(), _FRAME_KEYS].drop_duplicates()
    return scored.sort_values(_FRAME_KEYS).reset_index(drop=True)


def test_scored_frame_keys_none_is_byte_identical_to_the_internal_stride():
    frames = multi_frame_in_domain(n_frames=6)

    cf_default, prov_default, report_default = _build(frames)
    cf_none, prov_none, report_none = _build(frames, scored_frame_keys=None)

    # Non-vacuity: 6 in-domain frames, stride 5 -> ranks 0 and 5 survive -> exactly 2 scored.
    assert report_default.n_frames_scored == 2, report_default.drop_reasons

    assert cf_none.equals(cf_default)
    assert prov_none.equals(prov_default)
    assert report_none.drop_reasons == report_default.drop_reasons


def test_select_matches_the_engine_internal_selection():
    frames = multi_frame_in_domain(n_frames=6)

    selected = select_gkdv_scored_frames(frames)
    _cf, prov_default, _report = _build(frames)

    # The public selection is exactly the frames the internal stride would score.
    expected = _scored_keys(prov_default)
    got = selected.sort_values(_FRAME_KEYS).reset_index(drop=True)
    assert got.equals(expected)
    # Non-vacuity + the stride phase we expect (frames 0 and 5 of the 6).
    assert got["frame_id"].tolist() == [0, 5]


def test_injected_subset_scores_exactly_that_subset():
    frames = multi_frame_in_domain(n_frames=6)
    carrier = infer_ball_carrier(frames)
    goal_map = resolve_defended_goals(frames)

    # Inject a single survivor (frame 3) -- a frame the DEFAULT stride (0, 5) does not keep,
    # proving the injected set drives selection rather than the internal cumcount.
    survivors = frames.loc[frames["frame_id"] == 3, _FRAME_KEYS].drop_duplicates()
    _cf, prov, report = _build(frames, carrier=carrier, goal_map=goal_map, scored_frame_keys=survivors)

    assert _scored_keys(prov)["frame_id"].tolist() == [3]
    assert report.n_frames_scored == 1, report.drop_reasons
    # The other eligible frames are stride-skipped, not silently dropped -- conservation holds.
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    assert report.drop_reasons.get("stride_skipped", 0) == 5


def test_injected_key_outside_the_domain_is_not_scored():
    """An injected key whose frame fails the per-frame §4.1 domain is NOT resurrected.

    The per-frame domain still runs under injection; ``scored_frame_keys`` only replaces the
    global stride. A key for a non-existent / out-of-domain frame therefore scores nothing.
    """
    frames = multi_frame_in_domain(n_frames=6)
    carrier = infer_ball_carrier(frames)
    goal_map = resolve_defended_goals(frames)

    survivors = frames.loc[frames["frame_id"] == 0, _FRAME_KEYS].drop_duplicates().copy()
    phantom = survivors.copy()
    phantom["frame_id"] = 999  # no such frame
    injected = pd.concat([survivors, phantom], ignore_index=True)

    _cf, prov, _report = _build(frames, carrier=carrier, goal_map=goal_map, scored_frame_keys=injected)
    assert _scored_keys(prov)["frame_id"].tolist() == [0]


def test_windowed_union_is_byte_identical_to_whole_unit():
    """The neutrality proof: union of per-window scoring == whole-unit scoring.

    Pin carrier + goal_map whole-unit (4.132 seam), compute the survivor selection once
    (``select_gkdv_scored_frames``), then score two disjoint windows under that injected set. The
    union of the scored provenance must match the whole-unit scored provenance byte-for-byte.
    """
    frames = multi_frame_in_domain(n_frames=6)
    carrier = infer_ball_carrier(frames)
    goal_map = resolve_defended_goals(frames)
    survivors = select_gkdv_scored_frames(frames, carrier=carrier, goal_map=goal_map)

    _cf_whole, prov_whole, _report_whole = _build(frames, carrier=carrier, goal_map=goal_map)
    whole_scored = prov_whole.loc[prov_whole["drop_reason"].isna()]

    cmp_cols = [*_FRAME_KEYS, "gk_team_id", "ghost_x", "ghost_y", "displacement_m"]
    parts = []
    for lo, hi in ((0, 3), (3, 6)):
        window = frames[(frames["frame_id"] >= lo) & (frames["frame_id"] < hi)].copy()
        _cf, prov_w, _report = _build(window, carrier=carrier, goal_map=goal_map, scored_frame_keys=survivors)
        parts.append(prov_w.loc[prov_w["drop_reason"].isna(), cmp_cols])

    union = pd.concat(parts, ignore_index=True).sort_values([*_FRAME_KEYS, "gk_team_id"]).reset_index(drop=True)
    whole = whole_scored[cmp_cols].sort_values([*_FRAME_KEYS, "gk_team_id"]).reset_index(drop=True)

    # Non-vacuity: the union actually scored the whole-unit survivors (frames 0 and 5).
    assert sorted(union["frame_id"].unique().tolist()) == [0, 5]
    pd.testing.assert_frame_equal(union, whole, check_like=False)
