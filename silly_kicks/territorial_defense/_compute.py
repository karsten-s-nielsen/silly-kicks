"""``compute_territorial_defense`` -- the TF-54b orchestrator (spec §3, §7).

Ties the engine + both arms together into a per-``(defender, match)`` samples table. Arm A
(identity-exact) scores D's own defensive-action frames; Arm B (attribution-approximate) scores
opponent passes whose target reflects into D's trimmed territory hull. Grouped on the CANONICAL
player id (ADR-019), raw id emitted; a conserving :class:`TerritorialDefenseReport` (ADR-042).

HONEST LIMIT -- this metric is validated as an INSTRUMENT, NOT as player-attributable. The
marginal-removal delta is team-conditioned by construction, and on a single-tournament /
national-team corpus the defender-vs-team confound is unidentifiable, so per-defender numbers are
NOT a defender ranking (see NOTICE / CLAUDE.md; a crossed defender+team ICC over a multi-club
transfer corpus is the future gate).

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks._frame_index import group_rows
from silly_kicks.id_compat import canonical_id, ids_match
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.territory import build_trimmed_hull
from silly_kicks.tracking import GoalEndUnresolvedError, resolve_defended_goals
from silly_kicks.xthreat import require_fitted_xt

from ._arms import arm_a_threat_suppressed, arm_b_threat_suppressed, contesting_defender_is_d
from ._columns import TD_SAMPLE_COLUMNS
from ._config import _DEFAULT_PARAMS, TerritorialDefenseParams
from ._engine import (
    UNRESOLVED_GEOMETRY,
    action_ltr_goal_map,
    build_report,
    classify_arm_a_domain,
    removal_leaves_enough_defenders,
    remove_player_row,
    select_arm_a_domain,
)
from ._report import TerritorialDefenseReport

_FRAME_KEYS = ("game_id", "period_id", "frame_id")
_PASS_TYPE_ID = 0  # SPADL pass
#: The two frame conventions ``compute_territorial_defense`` supports (spec §5.1). ``per_action_ltr`` is
#: SB360 (each freeze-frame in its action's acting-team LTR, ADR-028); ``match_ltr`` is
#: continuous-tracking-derived (home-attacks-right), served by the per-match ``resolve_defended_goals``.
FRAME_CONVENTIONS = ("per_action_ltr", "match_ltr")


def compute_territorial_defense(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt,
    links: pd.DataFrame | None = None,
    visible_area: pd.DataFrame | None = None,
    frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr",
    params: TerritorialDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, TerritorialDefenseReport]:
    """Per-``(game_id, player_id)`` territorial-defense samples + a conserving report.

    Parameters
    ----------
    actions, frames :
        SPADL actions + tracking frames.
    xt :
        A fitted :class:`silly_kicks.xthreat.ExpectedThreat` (injected; refused unfitted).
    visible_area :
        Optional per-action ``visible_area`` polygons (SPEC-02 FOV local-completeness gate).
    frame_convention :
        The coordinate convention of ``frames`` (spec §5.1). ``"per_action_ltr"`` (DEFAULT, SB360
        freeze-frames): each frame is in ITS action's acting-team LTR (``frame_id == action_id``; the
        actor row carries the real id via
        :func:`silly_kicks.keeper_identity.apply_actor_identities_to_frames`); the goal is resolved
        PER FRAME from the acting team (:func:`action_ltr_goal_map`). ``"match_ltr"``
        (continuous-tracking-derived, home-attacks-right): a single per-match
        :func:`silly_kicks.tracking.resolve_defended_goals` map -- the historical behaviour. The compute
        OWNS the convention; classify + the arms are convention-agnostic (they call the factory).

    Returns
    -------
    (samples, report) : the per-defender table (``TD_SAMPLE_COLUMNS``) + conservation over the
        Arm-A domain. **Not a ranking** (see the module honest-limit).

    Examples
    --------
    Runs on SB360-shaped ``actions``/``frames`` with a fitted ``xt`` (default per-action convention).
    DEMOTED to experimental (ADR-090) -- imported from the private ``._compute`` path::

        from silly_kicks.territorial_defense._compute import compute_territorial_defense
        samples, report = compute_territorial_defense(actions, frames, xt=xt)
        # samples: one row per (game_id, player_id); report conserves over the Arm-A domain.
    """
    require_fitted_xt(xt, caller="compute_territorial_defense")
    if frame_convention not in FRAME_CONVENTIONS:
        raise ValueError(f"frame_convention={frame_convention!r} not in {FRAME_CONVENTIONS}.")
    # L7: build the per-(game, period, frame) row-group index ONCE and thread it into every stage.
    groups = group_rows(frames, _FRAME_KEYS)
    goal_map_for = _make_goal_map_for(frames, frame_convention)

    domain = select_arm_a_domain(actions, params=params)
    classified = classify_arm_a_domain(
        domain, frames, params=params, visible_area=visible_area, goal_map_for=goal_map_for, groups=groups
    )
    arm_a, classified = _score_arm_a(classified, frames, xt=xt, goal_map_for=goal_map_for, params=params, groups=groups)
    arm_b, arm_b_census = _score_arm_b(
        actions, frames, domain, xt=xt, goal_map_for=goal_map_for, params=params, groups=groups
    )

    samples = _assemble_samples(classified, arm_a, arm_b)
    report = build_report(classified["td_source"], params=params, n_frames_in=len(domain), arm_b=arm_b_census)
    return samples, report


def _make_goal_map_for(frames, frame_convention):
    """The ONE place the frame convention lives (spec §5.3 / GOAL-SPEC-02): a factory
    ``goal_map_for(game_id, period_id, acting_team_id, opponent_team_id) -> GoalMap`` that classify + the
    arms call, staying convention-agnostic.

    ``per_action_ltr`` builds a PER-FRAME map from the action's acting team; ``match_ltr`` closes over ONE
    per-match ``resolve_defended_goals`` map (acting/opponent unused) -- byte-identical to the historical
    per-match behaviour.
    """
    if frame_convention == "per_action_ltr":

        def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id):
            return action_ltr_goal_map(
                game_id, period_id, acting_team_id=acting_team_id, opponent_team_id=opponent_team_id
            )
    else:  # match_ltr
        match_map = resolve_defended_goals(frames)

        def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id):
            return match_map

    return goal_map_for


def _score_arm_a(classified, frames, *, xt, goal_map_for, params, groups=None):
    """Per Arm-A candidate: the delta on scored frames (NaN otherwise).

    Returns ``(tidy_frame, classified)`` where ``classified``'s ``td_source`` is DOWNGRADED to
    ``unresolved_geometry`` for any frame on which the arm raises ``GoalEndUnresolvedError`` (ADR-055:
    an unresolvable attacked-goal is honest-NaN caught at the edge, never a crash) -- so the report
    conserves against what actually scored. ``goal_map_for`` is the compute-owned convention factory
    (spec §5.3); ``groups`` is a precomputed ``group_rows`` threaded from the orchestrator (built locally
    when ``None``).
    """
    if groups is None:
        groups = group_rows(frames, _FRAME_KEYS)
    classified = classified.copy().reset_index(drop=True)
    td = classified["td_source"].astype(object).tolist()
    rows = []
    for i, cand in enumerate(classified.itertuples()):
        delta = np.nan
        scored = cand.td_source == "scored"
        if scored:
            fr = groups.get(cand.game_id, cand.period_id, cand.frame_id)
            cf = remove_player_row(fr, player_pos=int(cand.defender_pos))
            gm = goal_map_for(cand.game_id, cand.period_id, cand.defending_team_id, cand.attacking_team_id)
            try:
                delta = arm_a_threat_suppressed(
                    fr, cf, attacking_team_id=cand.attacking_team_id, xt=xt, goal_map=gm, params=params
                )
            except GoalEndUnresolvedError:
                td[i] = UNRESOLVED_GEOMETRY
                delta, scored = np.nan, False
            else:
                if not np.isfinite(delta):  # a non-finite threat is an unresolved computation (mirror Arm B)
                    td[i] = UNRESOLVED_GEOMETRY
                    delta, scored = np.nan, False
        rows.append({"game_id": cand.game_id, "defender_id": cand.defender_id, "a_delta": delta, "a_scored": scored})
    classified["td_source"] = td
    return pd.DataFrame(rows), classified


def _score_arm_b(actions, frames, domain, *, xt, goal_map_for, params, groups=None):
    """Per (defender, in-hull opponent pass): the Arm-B delta + contesting-is-D flag + a conserving
    census. Returns ``(rows_df, census)`` where ``census = {"n_in", "n_scored", "drop_reasons"}`` over
    qualifying ``(defender, in-hull-pass)`` pairs (ADR-042): a pass NOT in a defender's hull is OUT OF
    DOMAIN (not counted); a missing frame / depleted removal / unresolvable geometry / non-finite delta
    is dropped-AND-counted. ``goal_map_for`` is the compute-owned convention factory (spec §5.3);
    ``groups`` is a precomputed ``group_rows`` threaded from the orchestrator (built locally when
    ``None``).
    """
    fl = float(spadlconfig.field_length)
    fw = float(spadlconfig.field_width)
    if groups is None:
        groups = group_rows(frames, _FRAME_KEYS)
    passes = actions[actions["type_id"] == _PASS_TYPE_ID]
    # ADR-068: group opponent-pass candidates by game ONCE. A per-defender full-table filter of
    # `passes` would rescan every match's passes for each defender -> O(defenders * passes), quadratic
    # in a multi-match batch; grouping by game bounds each defender to its own game's passes.
    passes_by_game = group_rows(passes, ("game_id",))

    rows: list[dict] = []
    n_in = 0
    n_scored = 0
    drops: dict[str, int] = {}

    def _drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    # one hull per (game, defender): own-half defensive-action locations (D's action-LTR frame).
    for (game_id, defender_id, defending_team_id), grp in _distinct_defenders(domain, actions, params):
        # own-half cut via the configurable threshold (default 52.5), matching the sibling
        # ``territory`` package's ``own_half_max_x`` -- not a hardcoded ``fl / 2`` (L8).
        own_half_xy = grp[grp["start_x"] < params.own_half_max_x][["start_x", "start_y"]].to_numpy(dtype=float)
        hull = build_trimmed_hull(own_half_xy, trim_fraction=params.trim_fraction)
        if hull is None:
            continue  # degenerate hull -> this defender contributes NO pairs (out of domain, not a drop)
        game_passes = passes_by_game.get(game_id)
        if game_passes is None or game_passes.empty:
            continue
        # opponent passes (this game, not D's team, non-NA team). ids_match is fresh-RangeIndex, so
        # .to_numpy() both sides before combining (id_compat/ADR-019); the .notna() drops NaN-team rows
        # (aligning with territory/_compute -- a NaN team is not an opponent).
        team = game_passes["team_id"]
        not_d_team = team.notna().to_numpy(dtype=bool) & ~ids_match(team, defending_team_id).to_numpy(dtype=bool)
        opp = game_passes[not_d_team]
        if opp.empty:
            continue
        # Hull membership: reflect the opponent end into D's action-LTR frame (ADR-028; opposing teams).
        end_x = opp["end_x"].to_numpy(dtype=float)
        end_y = opp["end_y"].to_numpy(dtype=float)
        refl = np.column_stack([fl - end_x, fw - end_y])
        in_hull = hull.contains(refl)
        for i, (opp_row, member) in enumerate(zip(opp.itertuples(), in_hull, strict=True)):
            if not member:
                continue  # out of domain (the pass does not land in D's territory)
            n_in += 1
            fr = groups.get(opp_row.game_id, opp_row.period_id, opp_row.action_id)
            if len(fr) == 0:
                _drop("missing_frame")
                continue
            # PLAN-01 for Arm B: removing the nearest defender must leave >= min_defenders_after_removal,
            # else the counterfactual degrades to attacker-controls-all (an upward-biased outlier).
            if not removal_leaves_enough_defenders(
                fr, defending_team_id=defending_team_id, min_after=params.min_defenders_after_removal
            ):
                _drop("removal_undersupported")
                continue
            # The frame's action is the OPPONENT's pass -> acting team = opp_row.team_id (the passer),
            # its opponent = D's team. Convention factory resolves the goal for this frame (spec §5.3).
            gm = goal_map_for(opp_row.game_id, opp_row.period_id, opp_row.team_id, defending_team_id)
            attacked = gm.attacked_goal(opp_row.game_id, opp_row.period_id, opp_row.team_id, allow_guess=True)
            if attacked is None:
                _drop("unresolved_geometry")
                continue
            # Threat target in FRAME coords: reflect only when the opponent attacks x=0 (away) in the
            # frame. end_x/end_y are pre-extracted float arrays (numpy float -> float, not a pandas Scalar).
            ex, ey = float(end_x[i]), float(end_y[i])
            target = (fl - ex, fw - ey) if float(attacked) == 0.0 else (ex, ey)
            try:
                delta, _pos = arm_b_threat_suppressed(
                    fr,
                    target_xy=target,
                    defending_team_id=defending_team_id,
                    attacking_team_id=opp_row.team_id,
                    xt=xt,
                    goal_map=gm,
                    params=params,
                )
            except GoalEndUnresolvedError:
                _drop("unresolved_geometry")  # ADR-055: caught at the edge -> counted, never a crash
                continue
            if not np.isfinite(delta):
                _drop("non_finite_delta")
                continue
            n_scored += 1
            rows.append(
                {
                    "game_id": game_id,
                    "defender_id": defender_id,
                    "b_delta": delta,
                    "is_d": contesting_defender_is_d(
                        fr, target, defending_team_id=defending_team_id, d_player_id=defender_id
                    ),
                }
            )
    census = {"n_in": n_in, "n_scored": n_scored, "drop_reasons": drops}
    return pd.DataFrame(rows, columns=["game_id", "defender_id", "b_delta", "is_d"]), census


def _distinct_defenders(domain, actions, params):
    """Yield ((game_id, raw defender_id, defending_team_id), that defender's defensive actions).

    ADR-068: group the defensive actions by ``(game_id, player_id)`` ONCE, then O(1) ``.get`` per
    distinct defender -- never a per-defender full-``actions`` mask rescan (which is O(defenders *
    actions), quadratic when both scale). ``group_rows`` keys canonically (ADR-019).
    """
    defensive = actions[actions["type_id"].isin(list(params.defensive_action_type_ids))]
    by_defender = group_rows(defensive, ("game_id", "player_id"))
    seen: dict = {}
    for cand in domain.itertuples():
        key = (canonical_id(cand.game_id), canonical_id(cand.defender_id))
        if key in seen:
            continue
        seen[key] = True
        yield (cand.game_id, cand.defender_id, cand.defending_team_id), by_defender.get(cand.game_id, cand.defender_id)


def _assemble_samples(classified, arm_a, arm_b) -> pd.DataFrame:
    """One row per (game_id, canonical defender id): a_/b_ aggregates + slippage + td_source."""
    a = arm_a.copy()
    a["_ck"] = list(zip(a["game_id"].map(canonical_id), a["defender_id"].map(canonical_id), strict=True))
    # Build the per-defender lookups ONCE (ADR-068), not a full-table rescan per group.
    b = arm_b.copy()
    b_by_ck: dict = {}
    if not b.empty:
        b["_ck"] = list(zip(b["game_id"].map(canonical_id), b["defender_id"].map(canonical_id), strict=True))
        b_by_ck = {ck: g for ck, g in b.groupby("_ck", sort=False)}
    empty_b = b.iloc[0:0]
    cl = classified.copy()
    cl["_ck"] = list(zip(cl["game_id"].map(canonical_id), cl["defender_id"].map(canonical_id), strict=True))
    drop_by_ck = {ck: str(g["td_source"].iloc[0]) for ck, g in cl.groupby("_ck", sort=False)}
    rows = []
    for ck, ag in a.groupby("_ck", sort=False):
        a_scored = ag[ag["a_scored"]]
        a_n = len(a_scored)
        a_sum = float(a_scored["a_delta"].sum()) if a_n else np.nan
        bg = b_by_ck.get(ck, empty_b)
        b_n = len(bg)
        b_sum = float(bg["b_delta"].sum()) if b_n else np.nan
        # b_attribution_slippage = the rate the position-chosen contesting defender is NOT the
        # hull-owner D (attribution ERROR; lower = tighter, 0.0 = perfect). Complement of the is-D
        # rate ``contesting_defender_is_d`` returns; ``.mean()`` skips the honest-NaN anonymous rows,
        # so an all-anonymous (SB360) hull -> NaN, never a fabricated 0 (IMPL-01/IMPL-02, ADR-027).
        slip = (1.0 - float(bg["is_d"].mean())) if b_n else np.nan
        scored = a_n + b_n > 0
        # td_source (per-defender summary): "scored" if any contribution, else this defender's
        # first Arm-A drop reason (the frame-level conservation lives in the report).
        td_source = "scored" if scored else drop_by_ck.get(ck, "no_actor")
        rows.append(
            {
                "game_id": ag["game_id"].iloc[0],
                "player_id": ag["defender_id"].iloc[0],  # raw id (ADR-019 .first())
                "a_threat_suppressed": a_sum,
                "a_frames_scored": a_n,
                "b_threat_suppressed": b_sum,
                "b_frames_scored": b_n,
                "b_attribution_slippage": slip,
                "td_source": td_source,
            }
        )
    out = pd.DataFrame(rows, columns=["game_id", "player_id", *TD_SAMPLE_COLUMNS])
    out["a_frames_scored"] = out["a_frames_scored"].astype("Int64")
    out["b_frames_scored"] = out["b_frames_scored"].astype("Int64")
    return out
