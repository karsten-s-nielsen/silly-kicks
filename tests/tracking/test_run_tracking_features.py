from __future__ import annotations

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import (
    add_defending_gk_player_id,
    apply_keeper_identities_to_frames,
    run_tracking_features,
)

ROSTER = {10: 901, 20: 902}


def _sb360_fixture():
    """A shot by team 10 with a freeze-frame carrying BOTH teams' keepers (team 20 = the defending
    keeper the shot needs). ``snapshot_to_tracking_frames`` numbers the rows and stamps
    ``speed_source == 'unavailable'`` (velocity-less by construction)."""
    # ``type_id`` is REQUIRED, not decoration: the ``_pre_shot_gk_position`` kernel gates on
    # ``actions["type_id"]`` (a real SPADL shot carries it), so WITHOUT it ``pre_shot_gk_x`` is
    # all-NaN and the keystone's R1 non-vacuity assertion (``pre_shot_gk_x.notna().any()``) would
    # pass VACUOUSLY. The plan's fixture text omitted it; it is added here (matching the Task-4
    # placement-helper fixture) so the producer genuinely unlocks a real GK position.
    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [5.0],
            "team_id": [10],
            "player_id": [101],
            "type_name": ["shot"],
            "type_id": [spadlconfig.actiontype_id["shot"]],
            "start_x": [90.0],
            "start_y": [34.0],
        }
    )
    # Freeze-frame players: shooter (team 10), a team-10 field player, and team 20's keeper near its goal.
    snapshots = pd.DataFrame(
        {
            "action_id": [0, 0, 0],
            "team_id": [10, 10, 20],
            "x": [90.0, 80.0, 104.0],
            "y": [34.0, 40.0, 34.0],
            "is_goalkeeper": [False, False, True],
        }
    )
    frames, _links = T.snapshot_to_tracking_frames(snapshots, actions)
    return actions, frames


def test_producer_equals_composing_the_add_star_calls_after_the_same_resolution():
    actions, frames = _sb360_fixture()
    # Baseline: resolve identity -> stamp defending_gk_player_id on ACTIONS -> BRIDGE identity onto the
    # FRAME keeper rows -> add_pre_shot_gk_position. F6 + R1: the baseline INCLUDES the resolver step
    # AND the frame bridge; without the frame bridge, add_pre_shot_gk_position finds no keeper row on
    # the synthetically-numbered SB360 frames and returns NaN -- and the equality would pass VACUOUSLY
    # (NaN == NaN). The single-sourced helpers guarantee baseline == producer.
    m, _ = T.resolve_keeper_identities(actions, frames, identity="roster", roster=ROSTER)
    base = T.add_pre_shot_gk_position(
        add_defending_gk_player_id(actions, m),
        apply_keeper_identities_to_frames(frames, m),
    )
    out, _report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER, families=["add_pre_shot_gk_position"]
    )
    added = base.columns.difference(actions.columns)
    # NON-VACUITY (R1): the bridge must produce a REAL keeper position, not NaN==NaN. A shot with the
    # defending keeper in-frame MUST yield a populated pre_shot_gk_x, or the keystone proves nothing.
    assert out["pre_shot_gk_x"].notna().any(), (
        "identity->frame bridge failed: pre_shot_gk_x is all-NaN, so the SB360 GK feature did not "
        "unlock -- the cycle's headline deliverable. (This assertion is what makes the equality below "
        "non-vacuous.)"
    )
    pd.testing.assert_frame_equal(out[added].reset_index(drop=True), base[added].reset_index(drop=True))


def test_report_conserves():
    actions, frames = _sb360_fixture()
    _out, report = run_tracking_features(
        actions,
        frames,
        identity="roster",
        roster=ROSTER,
        families=["add_pre_shot_gk_position", "add_team_shape"],
    )
    assert report.n_families_run + report.n_families_skipped == report.n_families_in


def test_absent_model_skips_the_family_not_fabricates():
    actions, frames = _sb360_fixture()
    _out, report = run_tracking_features(
        actions,
        frames,
        identity="roster",
        roster=ROSTER,
        families=["add_xt_gk"],  # xt not supplied
    )
    assert "add_xt_gk" in report.family_status
    assert report.family_status["add_xt_gk"].startswith("skipped")


def test_naming_the_keeper_does_not_make_velocity_metrics_score_on_sb360():
    """Trap 2 non-vacuity (ADR-063): the keeper IS named (non-vacuity -- the metric COULD have moved),
    yet DAS stays NaN on velocity-less frames. `add_das` self-degrades (catches DasUnscoreableError):
    it EMITS the `das_*` columns as NaN with `das_source == 'unscoreable_frame'` -- it does NOT raise
    or get skipped, so the witness is a real NaN VALUE, not an absent column. Note the column is
    `das_diff`, not `das`."""
    actions, frames = _sb360_fixture()
    out, _report = run_tracking_features(
        actions,
        frames,
        identity="roster",
        roster=ROSTER,
        families=["add_pre_shot_gk_position", "add_das"],
    )
    assert out["defending_gk_player_id"].notna().any(), "the keeper WAS named (non-vacuity)"
    assert out["das_diff"].isna().all(), "DAS stays NaN on velocity-less frames even though the keeper is named"
    assert (out["das_source"] == "unscoreable_frame").all(), (
        "add_das must self-degrade (RAN and honestly NaN'd), not be skipped -- the real non-vacuity"
    )


def test_duplicate_family_names_are_deduped_and_still_conserve():
    # family_status is keyed by name, so a duplicated `families` entry must be de-duped or the report's
    # conservation invariant (n_run + n_skipped == n_families_in) would break. n_families_in counts the
    # DISTINCT families.
    actions, frames = _sb360_fixture()
    _out, report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER, families=["add_xt_gk", "add_xt_gk"]
    )
    assert report.n_families_in == 1  # de-duped, not 2
    assert report.n_families_run + report.n_families_skipped == report.n_families_in
