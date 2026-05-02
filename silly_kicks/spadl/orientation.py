"""Direction-of-play handling: convert provider-native input conventions to SPADL LTR.

Provider event streams arrive in three different coordinate conventions:

- ``POSSESSION_PERSPECTIVE``: each team's events are recorded in the team's own
  attacking frame (every team's events at high-x in their own frame). StatsBomb
  and Wyscout open data ship this way.
- ``ABSOLUTE_FRAME_HOME_RIGHT``: both teams in the same coordinate system, with
  the home team attacking right (high-x) and the away team attacking left
  (low-x) consistently across all periods. Sportec, Metrica, Opta (loader-pre-
  normalised), and the kloppy gateway use this convention.
- ``PER_PERIOD_ABSOLUTE``: same as ``ABSOLUTE_FRAME_HOME_RIGHT`` in P1, but
  teams switch ends after halftime. PFF uses an explicit ``homeTeamStartLeft``
  flag per period; raw Opta f24 also ships this way.

The canonical SPADL output convention is "all teams attack left-to-right" --
every team's actions at high-x in their own frame. :func:`to_spadl_ltr` is the
single canonical normalizer; each converter calls it exactly once, declaring
its input convention.

:func:`validate_input_convention` is a heuristic sanity check that surfaces
declared/detected mismatches via warning (default) or raise (under the
``SILLY_KICKS_ASSERT_INVARIANTS=1`` environment variable). It never overrides
the declared convention -- the converter's declared ``input_convention`` is
the load-bearing contract.

Decision: ADR-006 (Direction-of-play handling per converter, silly-kicks 3.0.0).
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

import numpy as np
import pandas as pd

from . import config as spadlconfig

__all__ = [
    "ABSOLUTE_FRAME_HOME_RIGHT",
    "PER_PERIOD_ABSOLUTE",
    "POSSESSION_PERSPECTIVE",
    "DetectionResult",
    "InputConvention",
    "detect_input_convention",
    "to_spadl_ltr",
    "validate_input_convention",
]


class InputConvention(str, Enum):
    """The coordinate convention a provider's raw events ship in.

    Each silly-kicks SPADL converter declares one of these. :func:`to_spadl_ltr`
    dispatches on the value to produce canonical SPADL "all teams attack
    left-to-right" output.
    """

    POSSESSION_PERSPECTIVE = "possession_perspective"
    """Each team's events recorded in the team's own attacking frame
    (StatsBomb, Wyscout). No coordinate flip needed -- already SPADL LTR
    after rescale to (105, 68)."""

    ABSOLUTE_FRAME_HOME_RIGHT = "absolute_frame_home_right"
    """Both teams in the same absolute coordinate system, home team attacks
    right (high-x) consistently across all periods (Sportec, Metrica,
    Opta loader-pre-normalised, kloppy gateway). Mirror away team rows
    to produce SPADL LTR."""

    PER_PERIOD_ABSOLUTE = "per_period_absolute"
    """Both teams in absolute coordinate system, but teams switch ends each
    period (PFF, raw Opta f24). Requires per-period direction lookup
    (e.g., from PFF's ``homeTeamStartLeft`` flag) to mirror the correct
    rows per period."""


# Re-exports for ergonomic call sites: ``to_spadl_ltr(... POSSESSION_PERSPECTIVE)``
# instead of ``to_spadl_ltr(... InputConvention.POSSESSION_PERSPECTIVE)``.
POSSESSION_PERSPECTIVE = InputConvention.POSSESSION_PERSPECTIVE
ABSOLUTE_FRAME_HOME_RIGHT = InputConvention.ABSOLUTE_FRAME_HOME_RIGHT
PER_PERIOD_ABSOLUTE = InputConvention.PER_PERIOD_ABSOLUTE


_ASSERT_ENV_VAR = "SILLY_KICKS_ASSERT_INVARIANTS"


def _strict_mode_enabled() -> bool:
    """Return True iff ``SILLY_KICKS_ASSERT_INVARIANTS=1`` is set."""
    return os.environ.get(_ASSERT_ENV_VAR) == "1"


# ---------------------------------------------------------------------------
# Canonical normalizer
# ---------------------------------------------------------------------------


def to_spadl_ltr(
    actions: pd.DataFrame,
    *,
    input_convention: InputConvention,
    home_team_id: int | str,
    home_attacks_right_per_period: Mapping[int, bool] | None = None,
) -> pd.DataFrame:
    """Normalize action coordinates to the SPADL LTR convention.

    Single canonical entry point for direction-of-play handling. Each silly-kicks
    SPADL converter calls this exactly once, after building its raw action
    DataFrame, declaring the input convention via ``input_convention``.

    SPADL canonical convention: every team's actions are oriented as if the team
    plays from left to right -- shots cluster at high-x for both teams, GK
    actions cluster at low-x for both teams.

    Parameters
    ----------
    actions : pd.DataFrame
        SPADL actions with at least ``team_id``, ``period_id``, ``start_x``,
        ``start_y``, ``end_x``, ``end_y`` columns. Other columns are preserved.
    input_convention : InputConvention
        The convention the input coordinates are in. See :class:`InputConvention`.
    home_team_id : int | str
        Identifier of the home team. Must match the dtype of
        ``actions["team_id"]`` (int for SPADL_COLUMNS providers, string for
        DFL-style providers like Sportec).
    home_attacks_right_per_period : Mapping[int, bool] | None
        Required for ``PER_PERIOD_ABSOLUTE``; ignored for other conventions.
        Maps period_id to True iff the home team attacks right in that period.
        Use :func:`silly_kicks.tracking._direction.home_attacks_right_per_period`
        to derive from a provider's ``homeTeamStartLeft`` flag.

    Returns
    -------
    pd.DataFrame
        A copy of ``actions`` with coordinates normalized to SPADL LTR.

    Raises
    ------
    ValueError
        If ``input_convention`` is ``PER_PERIOD_ABSOLUTE`` but
        ``home_attacks_right_per_period`` is None, or if a period_id present
        in ``actions`` has no entry in ``home_attacks_right_per_period``.

    Examples
    --------
    StatsBomb / Wyscout (possession-perspective input -- no flip needed)::

        from silly_kicks.spadl.orientation import to_spadl_ltr, POSSESSION_PERSPECTIVE
        actions = to_spadl_ltr(
            actions, input_convention=POSSESSION_PERSPECTIVE, home_team_id=100,
        )

    Sportec / Metrica / Opta / kloppy (absolute-frame, home-right -- mirror away)::

        from silly_kicks.spadl.orientation import to_spadl_ltr, ABSOLUTE_FRAME_HOME_RIGHT
        actions = to_spadl_ltr(
            actions, input_convention=ABSOLUTE_FRAME_HOME_RIGHT,
            home_team_id="DFL-CLU-XXX",
        )

    PFF (per-period absolute -- consult homeTeamStartLeft)::

        from silly_kicks.spadl.orientation import to_spadl_ltr, PER_PERIOD_ABSOLUTE
        from silly_kicks.tracking._direction import home_attacks_right_per_period

        flips = home_attacks_right_per_period(
            home_team_start_left=True, home_team_start_left_extratime=False,
        )
        actions = to_spadl_ltr(
            actions, input_convention=PER_PERIOD_ABSOLUTE,
            home_team_id="HOME", home_attacks_right_per_period=flips,
        )
    """
    if input_convention is InputConvention.POSSESSION_PERSPECTIVE:
        # Already SPADL LTR per-team; nothing to do. Copy for caller-mutation safety.
        return actions.copy()

    if input_convention is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT:
        return _mirror_absolute_frame(actions, home_team_id=home_team_id)

    if input_convention is InputConvention.PER_PERIOD_ABSOLUTE:
        if home_attacks_right_per_period is None:
            raise ValueError(
                "to_spadl_ltr: input_convention=PER_PERIOD_ABSOLUTE requires "
                "home_attacks_right_per_period (Mapping[int, bool])."
            )
        return _mirror_per_period(
            actions,
            home_team_id=home_team_id,
            home_attacks_right_per_period=home_attacks_right_per_period,
        )

    # Defensive: enum is closed, but guards future additions.
    raise ValueError(f"to_spadl_ltr: unknown input_convention {input_convention!r}")


def _mirror_absolute_frame(actions: pd.DataFrame, *, home_team_id: int | str) -> pd.DataFrame:
    """Mirror away-team rows by (field_length - x, field_width - y)."""
    out = actions.copy()
    away_idx = (out["team_id"] != home_team_id).to_numpy()
    if not away_idx.any():
        return out
    for col in ("start_x", "end_x"):
        out.loc[away_idx, col] = spadlconfig.field_length - out.loc[away_idx, col].to_numpy()
    for col in ("start_y", "end_y"):
        out.loc[away_idx, col] = spadlconfig.field_width - out.loc[away_idx, col].to_numpy()
    return out


def _mirror_per_period(
    actions: pd.DataFrame,
    *,
    home_team_id: int | str,
    home_attacks_right_per_period: Mapping[int, bool],
) -> pd.DataFrame:
    """Per-period mirror: flip rows whose actual attacking direction is RTL.

    For each row, the team's attacking direction at that period is:
      - home team:   home_attacks_right_per_period[period]
      - away team:   not home_attacks_right_per_period[period]

    Rows where attacking direction is RTL (False) get mirrored to LTR.
    """
    out = actions.copy()
    is_home = (out["team_id"] == home_team_id).to_numpy()

    periods = out["period_id"].unique()
    missing = [p for p in periods if p not in home_attacks_right_per_period]
    if missing:
        raise ValueError(
            f"to_spadl_ltr: home_attacks_right_per_period missing entries for periods {sorted(missing)}; "
            f"got keys {sorted(home_attacks_right_per_period.keys())}"
        )

    # Build per-row mirror mask vectorized via period -> bool lookup
    period_attacks_right = out["period_id"].map(home_attacks_right_per_period).to_numpy()
    # mirror = (home AND home-attacks-left) OR (away AND home-attacks-right)
    mirror_idx = (is_home & ~period_attacks_right) | (~is_home & period_attacks_right)

    if not mirror_idx.any():
        return out
    for col in ("start_x", "end_x"):
        out.loc[mirror_idx, col] = spadlconfig.field_length - out.loc[mirror_idx, col].to_numpy()
    for col in ("start_y", "end_y"):
        out.loc[mirror_idx, col] = spadlconfig.field_width - out.loc[mirror_idx, col].to_numpy()
    return out


# ---------------------------------------------------------------------------
# Detector + validator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectionResult:
    """Result of a heuristic input-convention detection.

    Attributes
    ----------
    convention : InputConvention | None
        The detected convention, or None when the signal is too weak to classify.
    confidence : Literal["high", "medium", "low", "ambiguous"]
        Confidence tier. ``high`` requires >= 10 shots per (match, team, period);
        ``medium`` requires 5-9; otherwise ``ambiguous`` (and ``convention`` is None).
        ``low`` is reserved for fewer than 2 reliable groups (still ambiguous).
    diagnostics : dict
        Per-group means and reasoning, useful for debugging false positives.
    """

    convention: InputConvention | None
    confidence: Literal["high", "medium", "low", "ambiguous"]
    diagnostics: dict[str, object] = field(default_factory=dict)


def detect_input_convention(
    events: pd.DataFrame,
    *,
    match_col: str,
    x_max: float,
    x_col: str = "start_x",
    team_col: str = "team_id",
    period_col: str = "period_id",
    is_shot_col: str | None = None,
    min_shots_per_group_high: int = 10,
    min_shots_per_group_medium: int = 5,
) -> DetectionResult:
    """Heuristic detection of input convention from per-(match, team, period) shot distribution.

    Confidence tiers:
      - >= ``min_shots_per_group_high`` shots per group, all classified clean:
        ``confidence="high"``
      - 5-9 shots per group: ``confidence="medium"`` (still classifies)
      - fewer than ``min_shots_per_group_medium`` per group OR fewer than 2 reliable
        groups: ``convention=None``, ``confidence="ambiguous"`` -- defer to the
        caller's declared convention.

    Classification rules (applied per-match; first match wins):
      1. Every reliable (team, period) group has mean ``x_col`` > ``x_max/2``
         -> ``POSSESSION_PERSPECTIVE``.
      2. Within each match, each team is consistently on the same side across all
         periods, with the two teams on opposite sides -> ``ABSOLUTE_FRAME_HOME_RIGHT``.
      3. Within each match, each team alternates side between periods
         -> ``PER_PERIOD_ABSOLUTE``.
      4. Otherwise -> ``None`` (ambiguous).

    The detector requires ``match_col`` because PFF's ``homeTeamStartLeft`` varies
    per match in multi-match feeds; per-period detection only makes sense when
    grouped by match.

    The detector strongly prefers shot events (``is_shot_col``) when available --
    all-events averages are dominated by team possession share / pitch dominance,
    not attacking direction (verified empirically: StatsBomb 7298 all-events
    misclassified, shots-only correctly classifies with n=34 shots).

    Parameters
    ----------
    events : pd.DataFrame
        Raw events with ``match_col``, ``team_col``, ``period_col``, ``x_col``
        columns.
    match_col : str
        Column identifying the match (REQUIRED). All grouping is done within
        each match.
    x_max : float
        Source coordinate-system x maximum (120 for StatsBomb, 100 for
        Opta/Wyscout, 105 for Sportec/Metrica). Used to compute the high-x/low-x
        midpoint.
    x_col, team_col, period_col : str
        Column names. Defaults match the SPADL schema.
    is_shot_col : str | None
        Boolean column flagging shot events. When provided, only shots are used
        for detection (much sharper signal). When None, all events are used --
        a degraded signal that may misclassify defensive teams.
    min_shots_per_group_high : int
        Minimum shots per (match, team, period) group to call ``confidence="high"``.
        Default 10.
    min_shots_per_group_medium : int
        Minimum shots per group to include in classification at ``confidence="medium"``.
        Default 5.

    Returns
    -------
    DetectionResult
        See :class:`DetectionResult`.

    Examples
    --------
    Detect convention on a real StatsBomb fixture::

        from silly_kicks.spadl.orientation import detect_input_convention

        # events has team_id, period_id, start_x, type_name (with "Shot" rows)
        events["is_shot"] = events["type_name"] == "Shot"
        result = detect_input_convention(
            events, match_col="game_id", x_max=120, is_shot_col="is_shot",
        )
        assert result.convention.value == "possession_perspective"
    """
    df = events
    if is_shot_col is not None:
        df = df[df[is_shot_col].fillna(False).astype(bool)]
    # Coerce x_col to numeric -- some provider bronze schemas (e.g. Sportec via
    # kloppy raw_event passthrough) ship coords as XML-string attributes; the
    # downstream groupby.mean chokes on object dtype. Coerce-then-drop NaN.
    df = df.assign(**{x_col: pd.to_numeric(df[x_col], errors="coerce")})
    df = df.dropna(subset=[match_col, team_col, period_col, x_col])

    if df.empty:
        return DetectionResult(
            convention=None,
            confidence="ambiguous",
            diagnostics={"reason": "no events with required columns"},
        )

    x_mid = x_max / 2.0

    grp = df.groupby([match_col, team_col, period_col])[x_col].agg(["count", "mean"]).reset_index()
    grp = grp.rename(columns={"count": "n", "mean": "mean_x"})
    grp["side"] = np.where(grp["mean_x"].to_numpy() > x_mid, "high", "low")
    grp["reliable_high"] = grp["n"] >= min_shots_per_group_high
    grp["reliable_medium"] = grp["n"] >= min_shots_per_group_medium

    reliable = grp[grp["reliable_medium"]].copy()
    if len(reliable) < 2:
        return DetectionResult(
            convention=None,
            confidence="low",
            diagnostics={
                "reason": f"fewer than 2 (match, team, period) groups with >= {min_shots_per_group_medium} shots",
                "groups": grp.to_dict("records"),
            },
        )

    confidence: Literal["high", "medium"] = "high" if reliable["reliable_high"].all() else "medium"

    # Rule 1: every reliable group attacks high-x → POSSESSION_PERSPECTIVE
    if (reliable["side"] == "high").all():
        return DetectionResult(
            convention=InputConvention.POSSESSION_PERSPECTIVE,
            confidence=confidence,
            diagnostics={
                "reason": "every (match, team, period) group attacks high-x",
                "groups": reliable.to_dict("records"),
            },
        )

    # Per-match analysis for absolute vs per-period
    match_verdicts: list[InputConvention | None] = []
    for _match_id, match_grp in reliable.groupby(match_col, sort=False):
        n_periods = match_grp[period_col].nunique()
        if n_periods < 2:
            # Single-period match: cannot distinguish absolute_no_switch from per_period
            continue
        per_team_sides = match_grp.groupby(team_col)["side"].nunique()
        if (per_team_sides == 1).all():
            # Each team on same side across periods → absolute_no_switch
            team_means = match_grp.groupby(team_col)["mean_x"].mean()
            if team_means.nunique() == 2 and (team_means > x_mid).any() and (team_means < x_mid).any():
                match_verdicts.append(InputConvention.ABSOLUTE_FRAME_HOME_RIGHT)
                continue
        if (per_team_sides == 2).all():
            # Each team alternates side between periods → per_period_absolute
            match_verdicts.append(InputConvention.PER_PERIOD_ABSOLUTE)
            continue
        match_verdicts.append(None)

    if match_verdicts and all(v is InputConvention.ABSOLUTE_FRAME_HOME_RIGHT for v in match_verdicts):
        return DetectionResult(
            convention=InputConvention.ABSOLUTE_FRAME_HOME_RIGHT,
            confidence=confidence,
            diagnostics={
                "reason": "per-team sides consistent across periods within every match",
                "groups": reliable.to_dict("records"),
            },
        )
    if match_verdicts and all(v is InputConvention.PER_PERIOD_ABSOLUTE for v in match_verdicts):
        return DetectionResult(
            convention=InputConvention.PER_PERIOD_ABSOLUTE,
            confidence=confidence,
            diagnostics={
                "reason": "per-team sides alternate between periods within every match",
                "groups": reliable.to_dict("records"),
            },
        )

    return DetectionResult(
        convention=None,
        confidence="ambiguous",
        diagnostics={
            "reason": "no clean classification (mixed signals across matches or single-period data)",
            "groups": reliable.to_dict("records"),
            "match_verdicts": [v.value if v else None for v in match_verdicts],
        },
    )


def validate_input_convention(
    events: pd.DataFrame,
    declared: InputConvention,
    *,
    on_mismatch: Literal["warn", "raise", "silent"] | None = None,
    **detector_kwargs: object,
) -> DetectionResult:
    """Run the convention detector and surface mismatches with the declared contract.

    The declared ``input_convention`` is the load-bearing contract -- the
    detector NEVER overrides it. This function only surfaces mismatches via
    warning / raise / silent per ``on_mismatch``.

    When the detector returns ``convention=None`` (signal too weak), no warning
    fires; the call defers to the declared convention silently.

    Parameters
    ----------
    events : pd.DataFrame
        Raw events. Forwarded to :func:`detect_input_convention`.
    declared : InputConvention
        The convention the converter declares. Compared against the detector
        verdict.
    on_mismatch : {"warn", "raise", "silent"} | None, default None
        How to surface a confident detector / declared mismatch. ``None``
        resolves to ``"raise"`` when the ``SILLY_KICKS_ASSERT_INVARIANTS=1``
        environment variable is set (CI mode), otherwise ``"warn"``.
    **detector_kwargs
        Forwarded to :func:`detect_input_convention`. ``match_col`` and ``x_max``
        are required.

    Returns
    -------
    DetectionResult
        The detector's verdict (for caller introspection / logging).

    Raises
    ------
    ValueError
        When ``on_mismatch="raise"`` AND the detector confidently disagrees with
        ``declared``.

    Examples
    --------
    Wired into a converter (warn default, raise under env-var)::

        from silly_kicks.spadl.orientation import (
            validate_input_convention, ABSOLUTE_FRAME_HOME_RIGHT,
        )

        events["is_shot"] = events["event_type"] == "ShotAtGoal"
        validate_input_convention(
            events,
            declared=ABSOLUTE_FRAME_HOME_RIGHT,
            match_col="game_id",
            is_shot_col="is_shot",
            x_max=105.0,
        )
    """
    if on_mismatch is None:
        on_mismatch = "raise" if _strict_mode_enabled() else "warn"

    # Detector is given the declared.value as a hint? No -- detector is independent.
    # detector_kwargs may include match_col, x_max, is_shot_col, etc.
    # The caller is responsible for passing the right kwargs.
    if "match_col" not in detector_kwargs or "x_max" not in detector_kwargs:
        raise ValueError("validate_input_convention: 'match_col' and 'x_max' must be passed via detector_kwargs")

    # mypy/pyright: detector_kwargs is dict[str, object]; cast at call site.
    result = detect_input_convention(events, **detector_kwargs)  # type: ignore[arg-type]

    if result.convention is None or result.convention == declared:
        # Either signal too weak (defer silently), or detector agrees.
        return result

    msg = (
        f"validate_input_convention: declared={declared.value} but detector inferred "
        f"{result.convention.value} (confidence={result.confidence}). "
        f"Diagnostics: {result.diagnostics.get('reason', 'n/a')}. "
        f"Either fix the upstream loader to ship {declared.value} data, or update "
        f"the converter's declared convention to match what the loader actually produces."
    )
    if on_mismatch == "raise":
        raise ValueError(msg)
    if on_mismatch == "warn":
        warnings.warn(msg, stacklevel=2)
    # on_mismatch == "silent": suppress
    return result
