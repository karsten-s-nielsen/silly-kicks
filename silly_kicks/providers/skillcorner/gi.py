"""SkillCorner Game-Intelligence ``passing_option`` parse port (TF-62; shaping only).

Extracts the pre-computed option set the GK-decision metric's native tier consumes, from the 294-column
SkillCorner GI possession model. Raw loading (pining/parquet) stays scripts-side; this is pure shaping.

Imports ``id_compat`` (none needed yet) + pandas ONLY -- never ``silly_kicks.tracking`` (pinned by
``tests/providers/test_appearances_import_allowlist.py``, which sweeps ``providers/skillcorner/*.py``).
"""

from __future__ import annotations

import pandas as pd

_OUT_COLS = [
    "game_id",
    "period_id",
    "decision_id",
    "possessor_id",
    "team_id",
    "target_player_id",
    "target_x",
    "target_y",
    "is_chosen",
    "completion",
    "opponents_bypassed",
]


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)


def parse_passing_options(gi_events: pd.DataFrame, *, game_id) -> pd.DataFrame:
    """Shape the ``passing_option`` rows of a SkillCorner GI events table into canonical option rows.

    One row per ``passing_option``: ``decision_id`` is the possession it belongs to
    (``associated_player_possession_event_id``), ``is_chosen`` is the ``targeted`` flag, and
    ``completion`` / ``opponents_bypassed`` are the native ``xpass_completion`` / ``n_opponents_bypassed``.
    An empty / no-``passing_option`` input yields the empty declared-column frame.

    Examples
    --------
    Shape a SkillCorner Game-Intelligence events table into canonical option rows::

        from silly_kicks.providers.skillcorner import parse_passing_options
        options = parse_passing_options(gi_events, game_id="1886347")
        options[["decision_id", "is_chosen", "completion", "opponents_bypassed"]]
    """
    po = gi_events[gi_events["event_type"] == "passing_option"].copy()
    if po.empty:
        return pd.DataFrame(columns=_OUT_COLS)
    out = pd.DataFrame(
        {
            "game_id": str(game_id),
            "period_id": po["period"].astype("Int64"),
            "decision_id": po["associated_player_possession_event_id"].astype(str),
            "possessor_id": po["player_in_possession_id"],
            "team_id": po["team_id"],
            "target_player_id": po["player_id"],
            "target_x": pd.to_numeric(po["x_start"], errors="coerce"),
            "target_y": pd.to_numeric(po["y_start"], errors="coerce"),
            "is_chosen": _to_bool(po["targeted"]),
            "completion": pd.to_numeric(po["xpass_completion"], errors="coerce"),
            "opponents_bypassed": pd.to_numeric(po["n_opponents_bypassed"], errors="coerce"),
        }
    )
    return out.reset_index(drop=True)[_OUT_COLS]
