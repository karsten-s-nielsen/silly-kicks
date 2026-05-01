"""Direction-of-play helpers shared between PFF events + tracking adapters.

Extracted from ``silly_kicks/spadl/pff.py`` (PR-S18) into the tracking
package so events PFF, tracking PFF, and tracking Sportec can share one
implementation. ``home_attacks_right_per_period`` is the load-bearing
helper; ``compute_attacking_direction`` is a higher-level wrapper used
by tracking adapters to populate the per-row ``team_attacking_direction``
column.

Pure refactor: zero behaviour change in events.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def home_attacks_right_per_period(
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> dict[int, bool]:
    """Per-period flag: True iff the home team attacks RIGHT (LTR) in that period.

    Mirrors the original PFF events convention: in period 1, the home team
    attacks right when ``home_team_start_left=True`` (its goal is on the
    left, so it shoots toward the right goal). Period 2 inverts period 1.
    Period 3/4 (extra time) follow ``home_team_start_left_extratime``,
    falling back to True for period 4 when ET is absent. Period 5 (PSO)
    is a single-end shootout where direction is moot --- conventionally
    True.

    Parameters
    ----------
    home_team_start_left : bool
        From PFF metadata ``homeTeamStartLeft`` / DFL match-info equivalent.
    home_team_start_left_extratime : bool | None
        From PFF metadata ``homeTeamStartLeftExtraTime`` / DFL equivalent.
        Only required when ET periods (3/4) are present in the data.

    Returns
    -------
    dict[int, bool]
        ``{1: ..., 2: ..., 3: ..., 4: ..., 5: True}``.

    Examples
    --------
    Map a per-period flip lookup for the home team::

        from silly_kicks.tracking._direction import home_attacks_right_per_period
        flips = home_attacks_right_per_period(
            home_team_start_left=True, home_team_start_left_extratime=False,
        )
        assert flips[1] is True and flips[2] is False
    """
    return {
        1: bool(home_team_start_left),
        2: not bool(home_team_start_left),
        3: bool(home_team_start_left_extratime),
        4: (not bool(home_team_start_left_extratime) if home_team_start_left_extratime is not None else True),
        5: True,
    }


def compute_attacking_direction(
    *,
    team_id: pd.Series,
    period_id: pd.Series,
    is_ball: pd.Series,
    home_team_id: Any,
    home_team_start_left: bool,
    home_team_start_left_extratime: bool | None = None,
) -> pd.Series:
    """Per-row pre-flip attacking direction (``"ltr"`` / ``"rtl"``).

    Returns ``"ltr"`` for the team attacking left-to-right in this period
    (i.e., the side whose ``home_attacks_right_per_period`` flag is True
    iff that side is the home team), ``"rtl"`` otherwise. Ball rows always
    get ``None``. Period 5 (PSO) leaves direction undefined (``None``).

    This is the per-row analogue of ``home_attacks_right_per_period``. The
    tracking adapters use this to populate the
    ``team_attacking_direction`` schema column for the unflipped raw input;
    the adapter then per-period flips x/y so the final output is in
    home-team-attacks-LTR coordinates.
    """
    out = pd.Series([None] * len(team_id), dtype="object", index=team_id.index)
    flags = home_attacks_right_per_period(
        home_team_start_left=home_team_start_left,
        home_team_start_left_extratime=home_team_start_left_extratime,
    )
    for p in (1, 2, 3, 4):
        period_mask = (period_id == p) & ~is_ball
        if not period_mask.any():
            continue
        home_attacks_right = flags[p]
        is_home = team_id == home_team_id
        out.loc[period_mask & is_home] = "ltr" if home_attacks_right else "rtl"
        out.loc[period_mask & ~is_home] = "rtl" if home_attacks_right else "ltr"
    return out
