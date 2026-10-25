"""add_corner_windows -- per-corner window timing + first-contact winner (TF-46, spec §6.2).

Additive enricher on the identification table. For a tracking corner, the target snapshot is
``min(t_firstcontact + target_contact_lead_s, t_kick + target_cap_s)`` with the first contact (and its
winner) from the shipped TF-79 ``first_contact_in_window``; when TF-79 resolves no contact the
``t_kick + cap`` branch is the fallback (winner ``<NA>`` -- NO GI fallback). A freeze-frame corner
(``corner_source == "sb360_event"``) has no continuous tracking, so winner / first_contact are ``<NA>``
and the target-snapshot source is ``sb360_event`` (the first-on-ball-event freeze frame; its time is
supplied by the driver). ``first_possession_time`` is a GI diagnostic (NaN unless ``gi`` is given).

The contact row (TF-79) carries no team, so ``winner_team_id`` is resolved from ``frames`` at the
contact frame. PURE (ADR-033); NaN-tolerant (ADR-003); ``frames`` positional-or-keyword, kwargs
keyword-only (ADR-078).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import canonical_id, ids_match, same_id
from silly_kicks.tracking._restart_validation import GATE_TABLE_VERSION, RATIFIED, DataProfile, GateEntry, GateKey, gate

from ._columns import CORNER_WINDOW_COLUMNS
from ._config import CornerWindowParams

_DEFAULT_WINDOW_PARAMS = CornerWindowParams()
#: Corner-flag positions in the tracking frame (the four pitch corners; the R* >10 m-from-every-flag rule).
_FLAGS = ((0.0, 0.0), (105.0, 0.0), (0.0, 68.0), (105.0, 68.0))
_FLAG_EXCLUDE_M = 10.0


def first_contact_cell_state(margin: float, taker_known: bool) -> str:
    """The first-contact gate cell state (spec sections 8, 9.2): ``contested`` / ``uncontested`` x
    ``taker_known`` / ``taker_unknown``. Contested = ``attribution_margin <= contested_threshold_m`` (0.75 m). A
    NaN margin (no other-team player visible) cannot show the contact is uncontested, so it fails closed to
    ``contested`` (the cell withheld by construction); contested and uncontested never pool (section 15.3).

    Examples
    --------
    >>> from silly_kicks.corners._windows import first_contact_cell_state
    >>> first_contact_cell_state(2.0, True)
    'uncontested_taker_known'
    >>> first_contact_cell_state(float("nan"), False)  # no visible opponent: fails closed
    'contested_taker_unknown'
    """
    contested = not np.isfinite(margin) or margin <= RATIFIED["contested_threshold_m"]
    return f"{'contested' if contested else 'uncontested'}_{'taker_known' if taker_known else 'taker_unknown'}"


def _first_contact_gate(data_profile, corners, i, gate_table, cell_state):
    """The first-contact ``GateEntry`` for corner row ``i``, or ``None`` (withheld). Keyed on
    ``(data_profile, restart_type, "first_contact", cell_state)`` (spec section 9.2). ``None`` data_profile
    withholds (the commit-2 default)."""
    if data_profile is None:
        return None
    kind = str(corners["corner_kind"].iloc[i]) if "corner_kind" in corners.columns else "direct"
    restart_type = "corner_short" if kind == "short" else "corner_crossed"
    return gate(data_profile, restart_type, "first_contact", cell_state, table=gate_table)


def _contacts_by_period(contacts) -> dict[int, tuple[pd.DataFrame, np.ndarray]]:
    """Each period's RESOLVED contacts (``player_id`` set), time-sorted, built ONCE per call (ADR-068): the per-corner
    R* window is then a ``searchsorted`` slice, never a re-filter of the whole contact table."""
    out: dict[int, tuple[pd.DataFrame, np.ndarray]] = {}
    if contacts is None or contacts.empty:
        return out
    resolved = contacts[contacts["player_id"].notna().to_numpy()]
    for period, pos in resolved.groupby("period_id", sort=True).indices.items():
        g = resolved.iloc[pos].sort_values("time_seconds", kind="stable")
        out[int(period)] = (g, g["time_seconds"].to_numpy(dtype=np.float64))
    return out


def _frame_groups(frames: pd.DataFrame):
    """The frame rows grouped by ``frame_id`` ONCE (``group_rows``, ADR-068); ``None`` without a ``frame_id``."""
    if frames is None or frames.empty or "frame_id" not in frames.columns:
        return None
    return group_rows(frames, "frame_id")


def _rstar_first_contact(contacts_by_period, frame_groups, period, t_kick, t_hi, taker_id):
    """The section 9.1 R* first contact in ``(t_kick, t_hi]`` (``t0`` strictly after the kick, ``t1`` included by
    an explicit check). Only RESOLVED contacts count (``player_id`` set). Returns the contact row, or ``None``.

    - Known taker: the earliest contact excluding the taker and any contact within 10 m of a flag.
    - Unknown taker (the geographic rule): every contact inside the restart region (within 10 m of a flag) is
      excluded, and the window STARTS AFTER the last in-region contact -- an out-of-region touch before a later
      in-region one is still part of the kick sequence, not the first contact.

    ``contacts_by_period`` / ``frame_groups`` are built once per call (``_contacts_by_period`` / ``_frame_groups``)."""
    hit = contacts_by_period.get(int(period))
    if hit is None:
        return None
    g, times = hit
    lo = int(np.searchsorted(times, t_kick, side="right"))
    hi = int(np.searchsorted(times, t_hi, side="right"))
    c = g.iloc[lo:hi]
    if c.empty:
        return None
    in_region = np.array(
        [
            _within_flag_zone(frame_groups, f, p)
            for f, p in zip(c["frame_id"].to_numpy(dtype=object), c["player_id"].to_numpy(dtype=object), strict=True)
        ],
        dtype=bool,
    )
    taker = canonical_id(taker_id) if taker_id is not None and not pd.isna(taker_id) else None
    if taker is None:
        if in_region.any():
            last_in = float(c["time_seconds"].to_numpy()[in_region].max())
            keep = (~in_region) & (c["time_seconds"].to_numpy() > last_in)
        else:
            keep = ~in_region
        cand = c[keep]
        return cand.iloc[0] if len(cand) else None
    for j, (_, row) in enumerate(c.iterrows()):
        if same_id(row["player_id"], taker):
            continue  # exclude the taker
        if in_region[j]:
            continue  # exclude a contact within 10 m of any flag (a retriever at the flag)
        return row
    return None


def _within_flag_zone(frame_groups, frame_id, player_id) -> bool:
    """True when ``player_id`` at ``frame_id`` lies within ``_FLAG_EXCLUDE_M`` of any corner flag
    (``frame_groups`` = the call's ``_frame_groups``, an O(1) frame lookup)."""
    if frame_groups is None or player_id is None or pd.isna(player_id) or frame_id is None or pd.isna(frame_id):
        return False
    sub = frame_groups.get(frame_id)
    if "x" not in sub.columns or "y" not in sub.columns:
        return False  # no positions to test against (a minimal frame) -> cannot exclude
    sub = sub[ids_match(sub["player_id"], player_id).to_numpy()]
    if sub.empty:
        return False
    px, py = float(sub["x"].iloc[0]), float(sub["y"].iloc[0])
    return any(np.hypot(px - fx, py - fy) <= _FLAG_EXCLUDE_M for fx, fy in _FLAGS)


def _team_of(frame_groups, frame_id, player_id) -> object:
    """Team of ``player_id`` at ``frame_id`` (``<NA>`` if unresolved); ``frame_groups`` = the call's
    ``_frame_groups``."""
    if frame_groups is None or player_id is None or pd.isna(player_id) or frame_id is None or pd.isna(frame_id):
        return pd.NA
    sub = frame_groups.get(frame_id)
    if sub.empty or "team_id" not in sub.columns:
        return pd.NA
    # NA-safe id compare (ADR-019): the ball row carries a <NA> player_id, so a raw ``==`` yields an
    # NA-bearing mask that crashes under boolean indexing. ids_match maps <NA> -> False.
    match = ids_match(sub["player_id"].reset_index(drop=True), player_id).to_numpy()
    team = sub["team_id"].to_numpy()[match]
    return team[0] if len(team) else pd.NA


@nan_safe_enrichment
def add_corner_windows(
    corners: pd.DataFrame,
    frames: pd.DataFrame,
    contacts: pd.DataFrame,
    *,
    params: CornerWindowParams = _DEFAULT_WINDOW_PARAMS,
    gi: pd.DataFrame | None = None,
    data_profile: DataProfile | None = None,
    gate_table: Mapping[GateKey, GateEntry] | None = None,
) -> pd.DataFrame:
    """Add the window + winner + timing-diagnostic columns to the corners table (new frame).

    The first-contact WINNER is a GATED signal (spec section 9.2): it ships only when the first-contact gate
    passes for the row's ``(data_profile, restart_type, cell_state)``. The candidate is the earliest contact
    in ``(t_kick, t_kick + 6]`` excluding the taker and any contact within 10 m of a flag (section 9.1 R*).
    When the gate is absent (the commit-2 empty ``RESTART_GATE_TABLE``, or no ``data_profile``), the cell is
    WITHHELD: ``winner_team_id`` / ``winner_player_id`` / ``first_contact_time`` are NaN, ``winner_source`` is
    ``"withheld"``, and the target snapshot falls back to ``t_kick + cap``. ``gate_table`` injects a table for
    tests (both sides).

    Examples
    --------
    Enrich an identified corner table with TF-79 timing (every winner withheld under the empty gate table)::

        from silly_kicks.corners import add_corner_windows
        windowed = add_corner_windows(corners, frames, contacts)
    """
    if gi is not None:
        warnings.warn(
            "add_corner_windows(gi=) is deprecated and ignored: first_possession_time now comes from the "
            "resolve_kicks gi_next_possession_time output (spec section 9.4), not an exact-float t_kick join. "
            "The parameter is removed at the next major version.",
            DeprecationWarning,
            stacklevel=2,
        )
    out = corners.copy()
    n = len(out)
    t_init = np.full(n, np.nan)
    t_target = np.full(n, np.nan)
    t_outcome = np.full(n, np.nan)
    first_contact_time = np.full(n, np.nan)
    winner_team: list[object] = [pd.NA] * n
    winner_player: list[object] = [pd.NA] * n
    source: list[object] = [None] * n
    winner_source: list[object] = ["none"] * n
    fc_evidence: list[object] = [pd.NA] * n
    fc_gate_version: list[object] = [pd.NA] * n
    fc_cell: list[object] = [pd.NA] * n
    fc_margin = np.full(n, np.nan)

    # first_possession_time = the GI next-possession time (section 9.4); never from kick_bracket_hi / t_kick.
    gi_next = (
        out["gi_next_possession_time"].astype("float64").to_numpy()
        if "gi_next_possession_time" in out.columns
        else np.full(n, np.nan)
    )
    first_possession_time = gi_next.copy()
    src_col = out["corner_source"].astype(object)
    fp_status: list[object] = [None] * n
    # Built ONCE per call (ADR-068): the per-corner R* window and the per-contact frame lookups slice these.
    contacts_by_period = _contacts_by_period(contacts)
    frame_groups = _frame_groups(frames)
    for i in range(n):
        # first_possession_status (section 9.4): native rows -> native_anchor; SkillCorner-derived rows ->
        # gi when a next possession time exists, else chain (no possession between the restarts).
        if src_col.iloc[i] == "skillcorner_marker":
            fp_status[i] = "gi" if np.isfinite(gi_next[i]) else "chain"
        else:
            fp_status[i] = "native_anchor"

        t_kick = float(out["t_kick"].iloc[i])
        if np.isnan(t_kick):
            continue  # a derived corner with no resolved kick time: NaN window outputs (section 9.4)
        period = int(out["period_id"].iloc[i])
        t_init[i] = t_kick - params.init_lead_s
        t_outcome[i] = t_kick + params.outcome_window_s

        if src_col.iloc[i] == "sb360_event":
            source[i] = "sb360_event"
            t_target[i] = t_kick + params.target_cap_s
            continue

        # Section 9.3: a profile whose EVENTS code the first touch (GS, the section 5.6 PFF fields carried by
        # identify_corners as event_first_touch_*) is NATIVE PASSTHROUGH -- not gated, evidence "reference".
        ev_t = float(out["event_first_touch_time"].iloc[i]) if "event_first_touch_time" in out.columns else np.nan
        if np.isfinite(ev_t):
            first_contact_time[i] = ev_t
            t_target[i] = min(ev_t + params.target_contact_lead_s, t_kick + params.target_cap_s)
            source[i] = "first_contact"
            winner_player[i] = out["event_first_touch_player_id"].iloc[i]
            winner_team[i] = out["event_first_touch_team_id"].iloc[i]
            winner_source[i] = "event_coded"
            fc_evidence[i] = "reference"
            continue

        # R* candidate (section 9.1): known taker -> exclude it + the flag zones; unknown -> the geographic rule.
        taker = out["taker_player_id"].iloc[i] if "taker_player_id" in out.columns else pd.NA
        tk_raw = out["taker_known"].iloc[i] if "taker_known" in out.columns else pd.NA
        taker_known = (
            bool(tk_raw) if pd.notna(tk_raw) else False
        )  # <NA> = unknown -> the taker_unknown rule (fail closed)
        contact = _rstar_first_contact(
            contacts_by_period, frame_groups, period, t_kick, t_kick + params.outcome_window_s, taker
        )
        margin = (
            float(contact["attribution_margin"])
            if contact is not None and "attribution_margin" in contact.index and pd.notna(contact["attribution_margin"])
            else np.nan
        )
        if contact is not None:
            fc_margin[i] = margin
        # The WINNER is gated (section 9.2) on its cell (contested x taker_known, from the R* contact's margin):
        # it ships only when that cell passes. In commit 2 the table is empty, so every winner is withheld.
        entry = _first_contact_gate(data_profile, out, i, gate_table, first_contact_cell_state(margin, taker_known))
        if contact is not None and entry is not None:
            t_fc = float(contact["time_seconds"])
            first_contact_time[i] = t_fc
            t_target[i] = min(t_fc + params.target_contact_lead_s, t_kick + params.target_cap_s)
            source[i] = "first_contact"
            winner_player[i] = contact["player_id"]
            winner_team[i] = _team_of(frame_groups, contact["frame_id"], contact["player_id"])
            winner_source[i] = "tf79_contact"
            fc_evidence[i] = entry.tier
            fc_gate_version[i] = GATE_TABLE_VERSION
            fc_cell[i] = entry.cell_resolution
        else:
            # withheld (no gate entry) or no valid R* contact: NaN winner + the kick+cap snapshot fallback.
            t_target[i] = t_kick + params.target_cap_s
            source[i] = "kick_plus_2s"
            winner_source[i] = "withheld" if entry is None else "none"

    out["t_init"] = t_init
    out["t_target"] = t_target
    out["t_outcome"] = t_outcome
    out["first_contact_time"] = first_contact_time
    out["first_possession_time"] = first_possession_time
    out["winner_team_id"] = pd.array(winner_team, dtype="object")
    out["winner_player_id"] = pd.array(winner_player, dtype="object")
    out["target_snapshot_source"] = pd.Series(source, index=out.index, dtype="object").astype("category")
    out["winner_source"] = pd.Series(winner_source, index=out.index, dtype="object")
    out["first_contact_evidence"] = pd.array(fc_evidence, dtype="object")
    out["first_contact_gate_version"] = pd.array(fc_gate_version, dtype="object")
    out["first_contact_cell_resolution"] = pd.array(fc_cell, dtype="object")
    out["first_contact_attribution_margin"] = fc_margin
    out["first_possession_status"] = pd.Series(fp_status, index=out.index, dtype="object")
    return out.astype({c: t for c, t in CORNER_WINDOW_COLUMNS.items() if c in out.columns})
