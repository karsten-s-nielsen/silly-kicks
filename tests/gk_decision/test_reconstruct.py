"""ReconstructedOptionSet (TF-62 Phase 2): positional option sets scored in action-LTR.

Uses a deterministic distance-decay xPass stand-in so the expected completion/reachability are exact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gk_decision import GkDecisionParams, ReconstructedOptionSet, option_value
from silly_kicks.gk_decision._columns import OPTION_ROW_COLUMNS


class _FakeXPass:
    """Deterministic stand-in: completion decays with pass distance (positional, no velocity)."""

    def predict_completion(self, ox, oy, tx, ty):
        ox, oy, tx, ty = (np.asarray(a, dtype=float) for a in (ox, oy, tx, ty))
        d = np.hypot(tx - ox, ty - oy)
        return np.clip(1.0 - d / 120.0, 0.0, 1.0)


def _one_decision():
    # SB360-style: frames already action-LTR (keeper team 7 attacks x=105).
    # keeper 99 at (10,34); teammates 11 (near, at the pass end), 12 (far upfield, low xPass),
    # 13 (backward). Opponent 21 sits between keeper and teammate 11 so packing>0 for the chosen pass.
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                type_id=22,  # goalkick (gk_distribution)
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=30.0,
                end_y=34.0,  # chosen pass ~ teammate 11's location
            )
        ]
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=99,
                team_id=7,
                x=10.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                is_actor=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=11,
                team_id=7,
                x=30.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=12,
                team_id=7,
                x=95.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=13,
                team_id=7,
                x=5.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=21,
                team_id=8,
                x=20.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=1,
                team_id=None,
                x=10.0,
                y=34.0,
                is_ball=True,
                is_goalkeeper=False,
                is_actor=False,
            ),
        ]
    )
    return actions, frames


def test_reconstructed_uniform_schema_and_chosen_is_actual_pass():
    actions, frames = _one_decision()
    os_ = ReconstructedOptionSet(
        actions,
        frames,
        xpass=_FakeXPass(),
        params=GkDecisionParams(),
        keeper_ids=[99],
        frame_convention="per_action_ltr",
    )
    rows = os_.option_rows()
    assert list(rows.columns) == list(OPTION_ROW_COLUMNS)
    assert (rows["option_set_source"] == "reconstructed").all()
    assert rows["is_chosen"].sum() == 1  # exactly one chosen (the actual pass)
    # chosen target is the actual pass END (30,34), valued by xPass(keeper(10,34)->end); never snapped
    chosen = rows[rows["is_chosen"]].iloc[0]
    assert chosen["completion"] == pytest.approx(1.0 - 20.0 / 120.0)  # dist keeper(10,34)->end(30,34)=20
    assert (rows["opponents_bypassed"] >= 0).all()


def test_reachability_filter_prunes_unreachable_alternatives():
    actions, frames = _one_decision()
    keep = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=GkDecisionParams(reachability_min_xpass=0.2), keeper_ids=[99]
    ).option_rows()
    prune = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=GkDecisionParams(reachability_min_xpass=0.5), keeper_ids=[99]
    ).option_rows()
    # teammate 12 at (95,34): xPass ~= 1 - 85/120 = 0.292 -> kept at 0.2, pruned at 0.5
    assert (keep["completion"].to_numpy() >= 0.2).all()
    assert (prune["completion"].to_numpy() >= 0.5).all()  # chosen (0.833) is above 0.5, so it survives too
    assert len(prune) < len(keep)  # a higher floor prunes the low-xPass alternative


def test_packing_boundaries_match_glossaried_definition():
    # keeper (10,34) -> chosen end (50,34). Opponents at x = 5 (behind passer), 30 (interior),
    # 50 (on receiver), 70 (outside). packing_made counts (passer.x < dx <= receiver.x] = {30,50} = 2.
    # A backward teammate at (5,34) is an alternative: passer(10)->receiver(5) -> made 0 -> EV == completion.
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=50.0,
                end_y=34.0,
            )
        ]
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=99,
                team_id=7,
                x=10.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                is_actor=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=13,
                team_id=7,
                x=5.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),  # backward teammate
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=21,
                team_id=8,
                x=5.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),  # behind passer
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=22,
                team_id=8,
                x=30.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),  # interior
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=23,
                team_id=8,
                x=50.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),  # on receiver
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=24,
                team_id=8,
                x=70.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),  # outside
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=1,
                team_id=None,
                x=10.0,
                y=34.0,
                is_ball=True,
                is_goalkeeper=False,
                is_actor=False,
            ),
        ]
    )
    rows = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=GkDecisionParams(), keeper_ids=[99]
    ).option_rows()
    chosen = rows[rows["is_chosen"]].iloc[0]
    assert chosen["opponents_bypassed"] == 2.0  # {30, 50}: on-receiver counted (<=), behind/outside not
    back = rows[~rows["is_chosen"]]
    assert len(back) == 1 and back.iloc[0]["opponents_bypassed"] == 0.0  # backward -> made 0
    ev = option_value(back, params=GkDecisionParams())
    assert ev.iloc[0] == pytest.approx(float(back.iloc[0]["completion"]))  # bypassed 0 -> EV == completion


def _mirror_scene():
    # action-LTR scene: keeper 99 (10,34) team7 GK; teammates 11 (30,34) [= pass end] + 12 (50,34);
    # opp 21 (20,34); opp GK 88 (105,34); ball. action end (30,34).
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                time_seconds=0.0,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=30.0,
                end_y=34.0,
            )
        ]
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=99,
                team_id=7,
                x=10.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                is_actor=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=11,
                team_id=7,
                x=30.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=12,
                team_id=7,
                x=50.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=21,
                team_id=8,
                x=20.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=88,
                team_id=8,
                x=105.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=1,
                team_id=None,
                x=10.0,
                y=34.0,
                is_ball=True,
                is_goalkeeper=False,
                is_actor=False,
            ),
        ]
    )
    # match_ltr uses resolve_defended_goals, which needs the fuller frame schema
    frames["time_seconds"] = 0.0
    frames["source_provider"] = "test"
    return actions, frames


def test_orientation_mirror_invariance_away_possession():
    actions, frames = _mirror_scene()
    p = GkDecisionParams(reachability_min_xpass=0.0)  # keep every option so the mirror is compared in full
    leg_a = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=p, keeper_ids=[99], frame_convention="per_action_ltr"
    ).option_rows()
    # Leg B: reflect ALL frame positions to match-LTR (team7 now attacks x=0); the action end stays action-LTR.
    fb = frames.copy()
    fb["x"] = 105.0 - fb["x"]
    fb["y"] = 68.0 - fb["y"]
    leg_b = ReconstructedOptionSet(
        actions, fb, xpass=_FakeXPass(), params=p, keeper_ids=[99], frame_convention="match_ltr"
    ).option_rows()
    cols = ["is_chosen", "completion", "opponents_bypassed"]
    a = leg_a[cols].sort_values(cols).reset_index(drop=True)
    b = leg_b[cols].sort_values(cols).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)  # orientation-invariant
    # NON-VACUITY: scoring leg B's (match-LTR) frame AS per_action_ltr (no reflection) MUST differ --
    # otherwise the reflection is doing nothing and the invariance above is vacuous.
    wrong = ReconstructedOptionSet(
        actions, fb, xpass=_FakeXPass(), params=p, keeper_ids=[99], frame_convention="per_action_ltr"
    ).option_rows()
    w = wrong[cols].sort_values(cols).reset_index(drop=True)
    assert not a.equals(w)
    # CONTRACT PIN (answers the IMPL-04 review): the action anchors are action-LTR (ADR-028) and are NOT
    # reflected -- only the FRAME is. Feeding a match-LTR action too (reflecting end_x/end_y) pairs the
    # reflected-back keeper with a match-LTR end -> a wrong distance, so it MUST differ from leg_a. This is
    # exactly why reflecting the action would DOUBLE-reflect the real (action-LTR) input and break the tier.
    a_mltr = actions.copy()
    a_mltr["start_x"] = 105.0 - a_mltr["start_x"]
    a_mltr["end_x"] = 105.0 - a_mltr["end_x"]
    a_mltr["start_y"] = 68.0 - a_mltr["start_y"]
    a_mltr["end_y"] = 68.0 - a_mltr["end_y"]
    reflected_action = ReconstructedOptionSet(
        a_mltr, fb, xpass=_FakeXPass(), params=p, keeper_ids=[99], frame_convention="match_ltr"
    ).option_rows()
    ra = reflected_action[cols].sort_values(cols).reset_index(drop=True)
    assert not a.equals(ra)  # reflecting the action too (a match-LTR action) breaks orientation-invariance


def test_chosen_into_space_no_receiver_exclusion():
    # pass end (60,34) lies in space: nearest teammate (~33 m) is beyond receiver_exclusion_m, so NO
    # teammate is excluded -> both are alternatives; the chosen target stays the pass end (never snapped).
    actions = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                action_id=1,
                type_id=22,
                player_id=99,
                team_id=7,
                start_x=10.0,
                start_y=34.0,
                end_x=60.0,
                end_y=34.0,
            )
        ]
    )
    frames = pd.DataFrame(
        [
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=99,
                team_id=7,
                x=10.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=True,
                is_actor=True,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=11,
                team_id=7,
                x=30.0,
                y=20.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=12,
                team_id=7,
                x=30.0,
                y=48.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=21,
                team_id=8,
                x=40.0,
                y=34.0,
                is_ball=False,
                is_goalkeeper=False,
                is_actor=False,
            ),
            dict(
                game_id="g",
                period_id=1,
                frame_id=1,
                player_id=1,
                team_id=None,
                x=10.0,
                y=34.0,
                is_ball=True,
                is_goalkeeper=False,
                is_actor=False,
            ),
        ]
    )
    rows = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=GkDecisionParams(reachability_min_xpass=0.0), keeper_ids=[99]
    ).option_rows()
    assert rows["is_chosen"].sum() == 1
    chosen = rows[rows["is_chosen"]].iloc[0]
    assert chosen["completion"] == pytest.approx(1.0 - 50.0 / 120.0)  # keeper(10,34)->end(60,34)=50, never snapped
    assert (~rows["is_chosen"]).sum() == 2  # BOTH teammates kept as alternatives (no wrongful exclusion)


def test_keeper_frame_id_dtype_mismatch_still_scores():
    # CONSIDER-10 / ADR-019: action player_id is int 99 while the frame keeper row is str "99" --
    # ids_match bridges the cross-source dtype so the keeper still resolves and the decision scores.
    actions, frames = _one_decision()
    frames = frames.copy()
    frames["player_id"] = frames["player_id"].astype(object)
    frames.loc[frames["player_id"] == 99, "player_id"] = "99"  # keeper row id now a STRING
    rows = ReconstructedOptionSet(
        actions, frames, xpass=_FakeXPass(), params=GkDecisionParams(reachability_min_xpass=0.0), keeper_ids=[99]
    ).option_rows()
    assert not rows.empty and rows["is_chosen"].sum() == 1  # keeper matched despite int-vs-str
