"""SkillCorner Game-Intelligence ``passing_option`` parse port (TF-62; shaping only).

Extracts the pre-computed option set the GK-decision metric's native tier consumes, from the 294-column
SkillCorner GI possession model. Raw loading (pining/parquet) stays scripts-side; this is pure shaping.

Imports pandas + the SPADL GI period-label repair (``silly_kicks.spadl._skillcorner_periods``: numpy, pandas and the
numpy + pandas ``silly_kicks.id_compat`` only; the ``providers -> spadl`` direction has a precedent in
``providers/statsbomb/parse.py``) -- never
``silly_kicks.tracking``. The option ``period_id`` is the repaired period, single-sourced with the SPADL converter
(TF-46 Part 2 spec section 26.5), so the two GI readers never disagree on a period. (This module's import boundary is
NOT yet gated: ``tests/providers/test_appearances_import_allowlist.py`` sweeps ``providers/*/appearances.py``; the
repair helper's own imports are pinned by ``tests/spadl/test_skillcorner_periods.py``.)
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.spadl._skillcorner_periods import repair_gi_periods

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
    "period_source",  # native / frame: the GI period-label repair provenance (TF-46 Part 2 spec section 26.5)
]


def _to_bool(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(v).strip().lower() in ("true", "1", "1.0")).astype(bool)


def parse_passing_options(gi_events: pd.DataFrame, *, game_id) -> pd.DataFrame:
    """Shape the ``passing_option`` rows of a SkillCorner GI events table into canonical option rows.

    One row per ``passing_option``: ``decision_id`` is the possession it belongs to
    (``associated_player_possession_event_id``), ``is_chosen`` is the ``targeted`` flag, and
    ``completion`` / ``opponents_bypassed`` are the native ``xpass_completion`` / ``n_opponents_bypassed``.
    ``period_id`` is the GI period after the period-label repair the SPADL converter applies (the same helper), and
    ``period_source`` says whether it is the GI label (``native``) or comes from the GI frame (``frame``). Pass the
    FULL GI events table (every event type): the repair recovers each period's start frame from the
    ``player_possession`` rows, so a table filtered to ``passing_option`` cannot be repaired (guard G1 skips it with a
    ``GiPeriodRepairWarning`` and the option periods stay the GI labels).
    An empty / no-``passing_option`` input yields the empty declared-column frame.

    Examples
    --------
    Shape a SkillCorner Game-Intelligence events table into canonical option rows::

        from silly_kicks.providers.skillcorner import parse_passing_options
        options = parse_passing_options(gi_events, game_id="1886347")
        options[["decision_id", "is_chosen", "completion", "opponents_bypassed"]]
    """
    gi_events = gi_events.reset_index(drop=True)  # a unique index keeps the repair's Series aligned with the subset
    po = gi_events[gi_events["event_type"] == "passing_option"].copy()
    if po.empty:
        return pd.DataFrame(columns=_OUT_COLS)
    repair = repair_gi_periods(gi_events)  # GI facts only, as the converter (single-sourced; spec section 26.5)
    out = pd.DataFrame(
        {
            "game_id": str(game_id),
            "period_id": repair.period.loc[po.index].astype("Int64"),
            "decision_id": po["associated_player_possession_event_id"].astype(str),
            "possessor_id": po["player_in_possession_id"],
            "team_id": po["team_id"],
            "target_player_id": po["player_id"],
            "target_x": pd.to_numeric(po["x_start"], errors="coerce"),
            "target_y": pd.to_numeric(po["y_start"], errors="coerce"),
            "is_chosen": _to_bool(po["targeted"]),
            "completion": pd.to_numeric(po["xpass_completion"], errors="coerce"),
            "opponents_bypassed": pd.to_numeric(po["n_opponents_bypassed"], errors="coerce"),
            "period_source": repair.period_source.loc[po.index].astype("object"),
        }
    )
    return out.reset_index(drop=True)[_OUT_COLS]
