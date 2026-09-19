"""ReconstructedOptionSet -- positional option sets for SB360 + full-tracking (TF-62 Phase 2, spec §5).

Scores every option in ACTION-LTR (the acting keeper's team attacks x=105) so the injected
``PassCompletionModel`` is served in its trained convention. Match-LTR frames (full-tracking) are
reflected to action-LTR via :mod:`silly_kicks.reflection` (ADR-045 -- the ONE reflection seam); SB360
freeze-frames are already action-LTR (verified: actor position == action start), so no reflection.
The action anchors (``start_x/y``, ``end_x/y``) are ALWAYS action-LTR (standard SPADL, ADR-028) and are
NEVER reflected -- only the match-LTR FRAME is. Reflecting the action too would DOUBLE-reflect every
away-possession decision (pinned by ``test_orientation_mirror_invariance_away_possession``'s contract leg).
``opponents_bypassed`` reuses :func:`silly_kicks.tracking.compute_packing_metrics_batch` ``["packing_made"]``
(ADR-039 single definition), fed the promoted public :func:`silly_kicks.tracking.action_ltr_goal_map`.
The chosen option is the ACTUAL pass end (never snapped to a teammate). The reachability filter
(``GkDecisionParams.reachability_min_xpass``) prunes unreachable ALTERNATIVES -- the spike's load-bearing
finding. Drops (``no_frame`` / ``fov_cropped``, ADR-042/077) are COUNTED, never fabricated 0s.

Imports only ``silly_kicks.tracking`` PUBLIC seams + ``keeper_identity``/``id_compat``/``reflection``/
``spadl`` config -- never a ``silly_kicks.tracking._*`` private (pinned by the import-allowlist).
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks import reflection
from silly_kicks.id_compat import ids_isin, ids_match, same_id
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.tracking import (
    action_ltr_goal_map,
    compute_packing_metrics_batch,
    link_actions_to_frames,
    region_observed_fraction,
    resolve_defended_goals,
)

from ._columns import OPTION_ROW_COLUMNS
from ._config import GkDecisionParams

_FL = float(spadlconfig.field_length)
_FW = float(spadlconfig.field_width)


def _disk(cx: float, cy: float, r: float, n: int = 24) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])


def _ball_mask(frame: pd.DataFrame) -> np.ndarray:
    return frame["is_ball"].astype("boolean").fillna(False).to_numpy(dtype=bool)


class ReconstructedOptionSet:
    """Positional (SB360 / full-tracking) tier of the GK-decision :class:`OptionSet` port.

    Yields the uniform option-rows table (:data:`OPTION_ROW_COLUMNS`, ``option_set_source="reconstructed"``)
    the tier-agnostic engine consumes. ``frame_convention="per_action_ltr"`` (SB360 freeze-frames, the
    default) or ``"match_ltr"`` (continuous / home-attacks-right tracking). The reachability floor
    (``GkDecisionParams.reachability_min_xpass``, default 0.85) prunes unreachable ALTERNATIVES; on SB360
    the metric INVERTS below ~0.85 (the bundled WC2022 xPass runs generous there), so the default is the
    responsive value, not 0.5. ``drop_counts()`` returns the
    adapter's own ``no_frame`` / ``fov_cropped`` counts -- pass them to ``compute_gk_decision_value(...,
    extra_drops=option_set.drop_counts())`` so the Report census stays the TRUE decision population.

    Examples
    --------
    Reconstruct SB360 option sets and score them (frames carry the real keeper id via the actor bridge)::

        from silly_kicks.gk_decision import ReconstructedOptionSet, compute_gk_decision_value
        from silly_kicks.expected_passing import PassCompletionModel
        option_set = ReconstructedOptionSet(gk_actions, frames, xpass=PassCompletionModel.bundled(),
                                            params=params, keeper_ids=roster_gks, visible_area=va)
        samples, report = compute_gk_decision_value(option_set, extra_drops=option_set.drop_counts())
    """

    def __init__(
        self,
        actions: pd.DataFrame,
        frames: pd.DataFrame,
        *,
        xpass,
        params: GkDecisionParams,
        keeper_ids,
        frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr",
        links: pd.DataFrame | None = None,
        visible_area: pd.DataFrame | None = None,
    ) -> None:
        self._actions = actions
        self._frames = frames
        self._xpass = xpass
        self._params = params
        self._keeper_ids = list(keeper_ids)
        self._convention = frame_convention
        self._links = links
        self._visible_area = visible_area
        self._drops = {"no_frame": 0, "fov_cropped": 0}
        self._link_cache: pd.DataFrame | None = None

    def drop_counts(self) -> dict[str, int]:
        """The adapter's own dropped-and-counted decisions (``no_frame`` / ``fov_cropped``).

        Examples
        --------
        Thread the adapter drops into the conserving Report::

            samples, report = compute_gk_decision_value(os_, extra_drops=os_.drop_counts())
        """
        return dict(self._drops)

    # -- frame resolution -------------------------------------------------------------------------

    def _frame_id_for(self, gk_acts: pd.DataFrame, action_id):
        if self._links is not None:
            hit = self._links[ids_match(self._links["action_id"], action_id).to_numpy()]
            return hit["frame_id"].iloc[0] if len(hit) else None
        if self._convention == "per_action_ltr":
            return action_id  # SB360 snapshot: frame_id == action_id (snapshot_to_tracking_frames)
        if self._link_cache is None:
            self._link_cache = link_actions_to_frames(gk_acts, self._frames)[0]  # (pointers, LinkReport)
        hit = self._link_cache[ids_match(self._link_cache["action_id"], action_id).to_numpy()]
        return hit["frame_id"].iloc[0] if len(hit) else None

    def _polygon_for(self, action_id):
        if self._visible_area is None:
            return None
        hit = self._visible_area[ids_match(self._visible_area["action_id"], action_id).to_numpy()]
        return hit["polygon"].iloc[0] if len(hit) else None

    # -- steps 1-3: link -> orient to action-LTR -> FOV gate (the testable framing helper) --------

    def _frame_and_anchors_action_ltr(self, a: pd.Series, gk_acts: pd.DataFrame, match_map):
        """Return ``(frame_ltr, keeper_xy, keeper_team, opponent_team)`` or ``None`` (drop counted).

        The frame is put in action-LTR (reflected iff match-LTR and the keeper attacks x=0). The keeper
        + candidates come from THIS frame; the pass end (an action anchor) is already action-LTR.
        """
        action_id, keeper_team = a["action_id"], a["team_id"]
        fid = self._frame_id_for(gk_acts, action_id)
        if fid is None:
            self._drops["no_frame"] += 1
            return None
        frame = self._frames[ids_match(self._frames["frame_id"], fid).to_numpy()].copy()
        if frame.empty:
            self._drops["no_frame"] += 1
            return None
        nonball = frame[~_ball_mask(frame)]
        opp = [t for t in pd.unique(nonball["team_id"].dropna()) if not same_id(t, keeper_team)]
        if len(opp) != 1:  # a malformed frame (not exactly one opponent team) is not scoreable
            self._drops["no_frame"] += 1
            return None
        opp_team = opp[0]
        if self._convention == "match_ltr":
            attacked = match_map.attacked_goal(a["game_id"], a["period_id"], keeper_team, allow_guess=True)
            if attacked is None:  # unresolvable direction -> honest drop, never a guessed orientation
                self._drops["no_frame"] += 1
                return None
            if float(attacked) == 0.0:  # keeper attacks x=0 in match-LTR -> reflect the FRAME ONLY to
                # action-LTR (the action anchors are already action-LTR per ADR-028 -- never reflected).
                frame = reflection.reflect_columns(
                    frame,
                    np.ones(len(frame), dtype=bool),
                    point_x=["x"],
                    point_y=["y"],
                    field_length=_FL,
                    field_width=_FW,
                )
        krow = frame[ids_match(frame["player_id"], a["player_id"]).to_numpy() & ~_ball_mask(frame)]
        if krow.empty and "is_actor" in frame.columns:  # SB360 fallback: the actor row is the keeper
            krow = frame[frame["is_actor"].astype("boolean").fillna(False).to_numpy(dtype=bool)]
        if krow.empty:
            self._drops["no_frame"] += 1
            return None
        keeper_xy = (float(krow.iloc[0]["x"]), float(krow.iloc[0]["y"]))
        if self._visible_area is not None:  # ADR-077 FOV completeness around the keeper neighbourhood
            poly = self._polygon_for(action_id)
            frac = (
                region_observed_fraction(poly, _disk(keeper_xy[0], keeper_xy[1], self._params.fov_radius_m))
                if poly is not None and len(poly) >= 3
                else np.nan
            )
            if not (np.isfinite(frac) and frac >= self._params.fov_min_observed_fraction):
                self._drops["fov_cropped"] += 1
                return None
        return frame, keeper_xy, keeper_team, opp_team

    # -- steps 4-8: value every option -> reachability -> assemble (the valuation core) -----------

    def _rows_for_decision(self, a: pd.Series, gk_acts: pd.DataFrame, match_map):
        got = self._frame_and_anchors_action_ltr(a, gk_acts, match_map)
        if got is None:
            return None
        frame, keeper_xy, keeper_team, opp_team = got
        is_team = ids_match(frame["team_id"], keeper_team).to_numpy(dtype=bool)
        is_keeper = ids_match(frame["player_id"], a["player_id"]).to_numpy(dtype=bool)
        cand = frame[is_team & ~_ball_mask(frame) & ~is_keeper].reset_index(drop=True)
        pass_end = (float(a["end_x"]), float(a["end_y"]))
        # receiver-exclusion: the nearest teammate within receiver_exclusion_m of the pass end is the
        # presumed receiver, dropped from the ALTERNATIVES to avoid double-counting the chosen pass.
        if len(cand):
            d = np.hypot(cand["x"].to_numpy(dtype=float) - pass_end[0], cand["y"].to_numpy(dtype=float) - pass_end[1])
            nearest = int(np.argmin(d))
            if d[nearest] <= self._params.receiver_exclusion_m:
                cand = cand.drop(index=nearest).reset_index(drop=True)
        gm = action_ltr_goal_map(a["game_id"], a["period_id"], acting_team_id=keeper_team, opponent_team_id=opp_team)
        # chosen = the ACTUAL pass end (never snapped) at index 0; then the surviving alternatives in
        # `cand` order (a reset_index'd frame, so .to_numpy preserves the old iterrows order).
        cand_x = cand["x"].to_numpy(dtype=float)
        cand_y = cand["y"].to_numpy(dtype=float)
        tx = np.concatenate([[pass_end[0]], cand_x])
        ty = np.concatenate([[pass_end[1]], cand_y])
        chosen = np.zeros(len(tx), dtype=bool)
        chosen[0] = True
        comp = np.asarray(
            self._xpass.predict_completion(np.full(len(tx), keeper_xy[0]), np.full(len(tx), keeper_xy[1]), tx, ty),
            dtype=float,
        )
        # opponents_bypassed for ALL targets in ONE packing call: the defender extraction + back-line
        # selection + goal_map lookups are invariant across targets (only the receiver varies), so the
        # batch hoists them out of the per-target loop (opt-audit #1). Byte-identical to the old
        # per-target compute_packing_metrics loop (tests/tracking/test_packing_batch).
        byp = np.asarray(
            compute_packing_metrics_batch(
                frame,
                attacking_team_id=keeper_team,
                goal_map=gm,
                passer_xy=keeper_xy,
                receivers=np.column_stack([tx, ty]),
            )["packing_made"],
            dtype=float,
        )
        # reachability: drop unreachable ALTERNATIVES; the chosen option is never dropped (it was played).
        thr = self._params.reachability_min_xpass
        keep = chosen | (np.isfinite(comp) & (comp >= thr))
        return pd.DataFrame(
            {
                "game_id": a["game_id"],
                "period_id": a["period_id"],
                "decision_id": a["action_id"],
                "keeper_id": a["player_id"],
                "team_id": keeper_team,
                "is_chosen": chosen[keep],
                "completion": comp[keep],
                "opponents_bypassed": byp[keep],
                "option_set_source": "reconstructed",
            }
        )

    def option_rows(self) -> pd.DataFrame:
        """The uniform option-rows table over the keeper-possession decisions.

        Examples
        --------
        One row per option; exactly one ``is_chosen`` per ``decision_id`` (the actual pass)::

            rows = reconstructed_option_set.option_rows()
        """
        acts = self._actions[ids_isin(self._actions["player_id"], self._keeper_ids).to_numpy()]
        self._drops = {"no_frame": 0, "fov_cropped": 0}
        self._link_cache = None
        match_map = resolve_defended_goals(self._frames) if self._convention == "match_ltr" else None
        out = [rows for _, a in acts.iterrows() if (rows := self._rows_for_decision(a, acts, match_map)) is not None]
        if not out:
            return pd.DataFrame(columns=list(OPTION_ROW_COLUMNS))
        return pd.concat(out, ignore_index=True)[list(OPTION_ROW_COLUMNS)]
