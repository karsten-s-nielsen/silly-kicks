"""Put ``scripts/`` on sys.path for this directory.

pyproject sets ``pythonpath = [".", "tests"]``; the script modules (``_corpus``, ``_paired``,
``_cache``, ``_loader_pining``) are not importable without this. Scoped to ``tests/scripts/`` so
the global config is untouched.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

_GK = 22  # goalkick type_id


def _synthetic_skillcorner_match():
    """A tiny SkillCorner-shaped match exercising the GK-distribution domain (A1/A2, M3).

    Under the SHIPPED per-provider distrust (SkillCorner goal-kicks distrusted), every native
    goal-kick origin is IMPUTED, so all six resolve in-box -- the ADR-024 "~=100% own-box"
    acceptance -- while the RAW diagnostic still sees a3's broadcast-ball artifact:

    ==  =====  =============  =====================  ==========================  =========================
    a   team   native start   keeper @ frame         RESOLVED (shipped distrust)  raw_native_out_of_region
    ==  =====  =============  =====================  ==========================  =========================
    0   1      (5, 34)        p10 x=4, visible       tracking_gk, in box         False (native in box)
    1   1      NaN            p10 x=4, visible       tracking_gk, in box         False (imputed)
    2   1      (-1, 34)       p10 x=4, visible       tracking_gk, in box         False (gr_x=-1 <= 16.5)
    3   1      (60, 34)       p10 x=4, NOT visible   goalkick_prior, in box      True  (native far downfield)
    4   1      NaN            p10 x=40 (off-pos)     goalkick_prior, in box      False (imputed)
    5   2      (5, 34)        p20 x=101, visible     tracking_gk, in box         False (AWAY team)
    ==  =====  =============  =====================  ==========================  =========================

    a5 is the load-bearing row: an AWAY-team goal-kick. Team 2 DEFENDS x=105 in the frame
    (``team_attacking_direction="rtl"``), so ``resolve_defended_goals`` maps its goal to x=105 -- but
    ``resolve_gk_geometry`` emits the origin in ACTION-LTR (own goal at x=0), where keeper p20 @ frame
    x=101 reprojects to x~=4. A driver that computes ``gr_x`` from the frame goal map (the ADR-028
    defect) would place a5 at gr_x~=101 and call it out-of-box; the correct ``gr_x = origin_x`` keeps
    it in-box. With all six actions on team 1 the two frames agree and the defect is INVISIBLE -- which
    is exactly how it reached real SkillCorner data uncaught (28.6% own-box vs the correct 100%).

    So: origin_source mix tracking_gk/goalkick_prior, in_own_box ~=100%, gated out_of_region ~=0,
    and the raw diagnostic > 0 (a3). Team 1 defends x=0 (keeper mean-x ~11); team 2 defends x=105.
    """
    times = [5.0, 50.0, 100.0, 150.0, 200.0, 250.0]
    actions = pd.DataFrame(
        {
            "game_id": [9] * 6,
            "action_id": [0, 1, 2, 3, 4, 5],
            "team_id": [1, 1, 1, 1, 1, 2],  # a5 AWAY (frame-defends x=105) -> exercises ADR-028
            "player_id": [10, 10, 10, 10, 10, 20],
            "period_id": [1] * 6,
            "time_seconds": times,
            "type_id": [_GK] * 6,
            "start_x": [5.0, np.nan, -1.0, 60.0, np.nan, 5.0],
            "start_y": [34.0, np.nan, 34.0, 34.0, np.nan, 34.0],
            "end_x": [55.0, 60.0, 58.0, 62.0, 57.0, 55.0],
            "end_y": [34.0, 30.0, 34.0, 30.0, 34.0, 34.0],
        }
    )

    keeper_x = [4.0, 4.0, 4.0, 4.0, 40.0, 4.0]  # team-1 keeper; a4 off-position -> goalkick_prior
    keeper_seen = [True, True, True, False, True, True]  # a3 team-1 keeper not detected
    frame_rows = []
    for i, t in enumerate(times):
        fid = 1000 + i
        # team 1 goalkeeper (player 10)
        frame_rows.append(
            dict(
                game_id=9,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                team_id=1,
                player_id=10,
                is_goalkeeper=True,
                is_ball=False,
                team_attacking_direction="ltr",
                x=keeper_x[i],
                y=33.0,
                source_provider="skillcorner",
                visibility=keeper_seen[i],
            )
        )
        # team 2 goalkeeper (player 20) -- defends x=105
        frame_rows.append(
            dict(
                game_id=9,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                team_id=2,
                player_id=20,
                is_goalkeeper=True,
                is_ball=False,
                team_attacking_direction="rtl",
                x=101.0,
                y=34.0,
                source_provider="skillcorner",
                visibility=True,
            )
        )
        # ball
        frame_rows.append(
            dict(
                game_id=9,
                period_id=1,
                frame_id=fid,
                time_seconds=t,
                team_id=np.nan,
                player_id=np.nan,
                is_goalkeeper=False,
                is_ball=True,
                team_attacking_direction=np.nan,
                x=keeper_x[i],
                y=34.0,
                source_provider="skillcorner",
                visibility=True,
            )
        )
    frames = pd.DataFrame(frame_rows)
    return "skillcorner", "9", actions, frames, 1


@pytest.fixture
def slim_skillcorner_match():
    """(provider, match_id, actions, frames, home_team_id) for a synthetic GK-distribution match."""
    return _synthetic_skillcorner_match()


def _fixture_with_all_rows_offpitch():
    """A measure_match-shaped frame whose every row is gross-off-pitch (for the A2 failing side)."""
    from validate_skillcorner_keeper_origin import EXPECTED_COLS

    n = 3
    row: dict[str, list] = {c: [np.nan] * n for c in EXPECTED_COLS}
    row["is_gross_offpitch"] = [True] * n
    row["xt_gk_origin_source"] = ["native"] * n
    row["in_own_box"] = [False] * n
    return pd.DataFrame(row, columns=EXPECTED_COLS)
