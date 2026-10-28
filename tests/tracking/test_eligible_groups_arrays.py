"""`_model_eval._eligible_groups` over arrays equals its `2474b73` per-group body (HFU plan R3-8 S3 item 8 (a)).

On HF the per-frame ``groupby`` walk took 390 of the 456 s of the xCross substitution probe (S1f: 1.64 M
``DataFrame.__getitem__`` calls). No-rebaseline kind (c): `_eligible_groups_oracle` below is the `2474b73` body
VERBATIM. Each case compares every returned tuple with the oracle's: the group frame (`assert_frame_equal`, exact),
the defending team, the goal x and the carrier id, values and Python types, for both arms (`xcross`, `xs`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._model_eval import _eligible_groups
from tests.tracking._probe_fixtures import planted_model, probe_frames


# The `2474b73` `_eligible_groups`, VERBATIM (the D20 oracle; only the name differs).
def _eligible_groups_oracle(frames: pd.DataFrame, model, arm: str, advance_m: float) -> list[tuple]:
    """Collect eligible (resolvable carrier + GK row + in-domain) frame groups deterministically.

    Moved line-for-line from ``_xcross_eval.gk_substitution_probe``; ``arm`` selects ONLY the
    ball-position domain gate (arm='xcross': wide-area; arm='xs': attacking third). Returns
    (grp, gk_team, goal_x, cpid) tuples with grp reset-indexed."""
    from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._xcross_attempt import _in_wide_area

    cp = dict(getattr(model, "carrier_params", None) or {})
    carrier = infer_ball_carrier(frames, **cp) if cp else infer_ball_carrier(frames)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = resolve_defended_goals(frames)

    groups_list = []
    for (gid, pid, _fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        # Defending team(s) = non-ball player rows of the OTHER team. Filter by is_ball (not the
        # string "ball") so a provider/fixture that encodes the ball's team_id differently can't be
        # mistaken for a defending team (it would then have no GK row -> the frame would be dropped).
        non_ball = grp[~grp["is_ball"].astype(bool)]
        # .dropna() guards a non-ball player row with NA team_id (unresolved GS jersey): `pd.NA !=
        # poss_team` is NA -> `if` raises "boolean value of NA is ambiguous" (mirrors prepare/compute).
        defending = [t for t in non_ball["team_id"].dropna().unique() if t != poss_team]
        if not defending:
            continue
        goal_x = goal_map.get(gid, pid, defending[0], allow_guess=True)
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if arm == "xcross":
            if not _in_wide_area(bx, by, goal_x, advance_m):
                continue
        else:  # arm == "xs": the `_ball_in_attacking_third` predicate from _xshot_occurrence.py
            if not (abs(bx - goal_x) <= advance_m):  # NaN bx -> False -> skipped, like _in_wide_area
                continue
        cpid = grp["ball_carrier_player_id"].dropna()
        cpid = cpid.iloc[0] if not cpid.empty else None
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == defending[0])
        if cpid is None or not gk_mask.any():
            continue
        groups_list.append((grp.reset_index(drop=True), defending[0], goal_x, cpid))
    return groups_list


def _random_frames(seed: int, n_frames: int = 90, *, team_dtype: str = "object") -> pd.DataFrame:
    """Frames over two games and two periods: possession switches, the ball near either byline or mid-pitch,
    frames without a ball row, without a defending GK, with a null-team player row, and with no defender."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for k in range(n_frames):
        game = "g1" if k < n_frames // 2 else "g2"
        period = 1 if k % 30 < 15 else 2
        frame = 1000 * period + k
        t = 10.0 + 0.4 * k
        attack = "A" if rng.random() < 0.6 else "B"
        defend = "B" if attack == "A" else "A"
        bx = float(rng.choice([3.0, 8.0, 52.0, 97.0, 101.0]))
        by = float(rng.choice([4.0, 10.0, 34.0, 60.0, 64.0]))
        base = dict(game_id=game, period_id=period, frame_id=frame, time_seconds=t, vx=0.0, vy=0.0,
                    ball_state="alive")  # fmt: skip
        # the carrier sits on the ball; three teammates; three defenders; the defending GK at its own goal
        rows.append(base | dict(team_id=attack, player_id=f"{attack}1", x=bx, y=by, is_ball=False, is_goalkeeper=False))
        for j in range(2, 5):
            rows.append(base | dict(team_id=attack, player_id=f"{attack}{j}", x=float(rng.uniform(20, 85)),
                                    y=float(rng.uniform(5, 63)), is_ball=False, is_goalkeeper=False))  # fmt: skip
        if rng.random() > 0.05:  # sometimes no defender at all
            for j in range(1, 4):
                rows.append(base | dict(team_id=defend, player_id=f"{defend}{j}", x=float(rng.uniform(30, 100)),
                                        y=float(rng.uniform(5, 63)), is_ball=False, is_goalkeeper=False))  # fmt: skip
            if rng.random() > 0.1:  # sometimes no GK row
                gx = 104.0 if (period == 1) == (defend == "B") else 1.0
                rows.append(base | dict(team_id=defend, player_id=f"{defend}gk", x=gx, y=34.0, is_ball=False,
                                        is_goalkeeper=True))  # fmt: skip
        if rng.random() < 0.05:  # a player row with no team
            rows.append(base | dict(team_id=None, player_id="X9", x=50.0, y=30.0, is_ball=False, is_goalkeeper=False))
        if rng.random() > 0.05:  # sometimes no ball row
            rows.append(base | dict(team_id="ball", player_id=None, x=bx, y=by, is_ball=True, is_goalkeeper=False))
    frames = pd.DataFrame(rows)
    if team_dtype == "category":
        frames["team_id"] = frames["team_id"].astype("category")  # the tracking-frame storage dtype (ADR-106)
    return frames


def _assert_same(frames: pd.DataFrame, arm: str, advance_m: float) -> int:
    model = planted_model("mixed")
    want = _eligible_groups_oracle(frames, model, arm, advance_m)
    got = _eligible_groups(frames, model, arm, advance_m)
    assert len(got) == len(want)
    for (g_grp, g_team, g_goal, g_cpid), (w_grp, w_team, w_goal, w_cpid) in zip(got, want, strict=True):
        pd.testing.assert_frame_equal(g_grp, w_grp, check_exact=True)
        assert (g_team, g_goal, g_cpid) == (w_team, w_goal, w_cpid)
        assert (type(g_team), type(g_goal), type(g_cpid)) == (type(w_team), type(w_goal), type(w_cpid))
    for i in range(len(want)):  # indexed access, as the panel sampler uses it
        pd.testing.assert_frame_equal(got[i][0], want[i][0], check_exact=True)
    return len(want)


@pytest.mark.parametrize("arm, advance_m", [("xcross", 25.0), ("xcross", 40.0), ("xs", 35.0), ("xs", 60.0)])
@pytest.mark.parametrize("team_dtype", ["object", "category"])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_frames_match_the_oracle(seed, team_dtype, arm, advance_m):
    frames = _random_frames(seed, team_dtype=team_dtype)
    n_eligible = _assert_same(frames, arm, advance_m)
    n_frames = frames[["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    assert n_eligible < n_frames  # the filters bite (non-vacuity)


def test_some_frames_are_eligible_in_each_arm():
    frames = _random_frames(3, n_frames=120)
    assert _assert_same(frames, "xcross", 40.0) > 0
    assert _assert_same(frames, "xs", 60.0) > 0


def test_the_probe_fixtures_match_the_oracle():
    assert _assert_same(probe_frames(), "xcross", 25.0) == 2
    assert _assert_same(probe_frames(), "xs", 35.0) == 2


def test_no_per_frame_groupby_walk(monkeypatch):
    """The array body never walks the per-frame groups (the walk is what cost 390 s on HF). `ordered_group_rows`
    iterates a few groups for their key types (a handful, never one per frame), so the guard counts the per-frame
    groups iterated and refuses a walk of a quarter of the frames or more."""
    from pandas.core.groupby.generic import DataFrameGroupBy

    frames = _random_frames(4)
    n_frames = frames[["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    real_iter = DataFrameGroupBy.__iter__
    seen = {"groups": 0}

    def counting(self):
        keys = self.keys if isinstance(self.keys, list) else [self.keys]
        for item in real_iter(self):
            if "frame_id" in keys:
                seen["groups"] += 1
            yield item

    monkeypatch.setattr(DataFrameGroupBy, "__iter__", counting)
    _eligible_groups(frames, planted_model("mixed"), "xcross", 40.0)
    assert seen["groups"] < n_frames // 4, f"walked {seen['groups']} per-frame groups of {n_frames}"


def test_the_walk_guard_catches_the_per_frame_walk(monkeypatch):
    """Anti-rot: the same counting guard, around the verbatim walk, sees one group per frame."""
    from pandas.core.groupby.generic import DataFrameGroupBy

    frames = _random_frames(4)
    n_frames = frames[["game_id", "period_id", "frame_id"]].drop_duplicates().shape[0]
    real_iter = DataFrameGroupBy.__iter__
    seen = {"groups": 0}

    def counting(self):
        keys = self.keys if isinstance(self.keys, list) else [self.keys]
        for item in real_iter(self):
            if "frame_id" in keys:
                seen["groups"] += 1
            yield item

    monkeypatch.setattr(DataFrameGroupBy, "__iter__", counting)
    _eligible_groups_oracle(frames, planted_model("mixed"), "xcross", 40.0)
    assert seen["groups"] >= n_frames // 4
