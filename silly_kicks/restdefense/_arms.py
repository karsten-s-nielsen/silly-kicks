"""restdefense Layer-3 deterrent ARMS (TF-60, ADR-089).

Prices how much the in-possession team's ACTUAL rearguard (or keeper) suppresses the opponent's
counter-danger vs a league-average ghost, in attacker-value units so **negative = deterrent**. Reuses
gkdv's generic delta seams (``delta_threat_suppression_batch`` = xt-weighted threat; ``delta_das_batch``
= dangerous accessible space) unchanged -- the differencing already carries the DAS direction-pin,
the identity-cache-trap avoidance, the ``min_count=1`` honest-NaN, and the sign convention. Only the
ghost-frame builder (:mod:`._counterfactual`) is new. The grain bridge is
:func:`._windows.select_rest_defense_samples` (per-action samples carrying ``frame_id``); the per-frame
deltas map back to the per-action arm table by frame.

The outfield arm **isolates the rearguard by construction**: the outfield ghost repositions only A's
deepest-``n_rearguard`` field defenders and NEVER moves A's keeper, so the arm attributes no keeper
repositioning to the rearguard. It is NOT exactly keeper-*invariant* in VALUE, though: the threat leg
carries ``lambda_gk`` (A's keeper as a TTI control agent) and pitch control is nonlinear, so the fixed
keeper interacts differently with the DIFFERING rearguard across the two legs (measured: moving A's
keeper perturbs the threat arm a few percent; the DAS/space arm is keeper-blind-generic and near-
invariant). The whole-team delta is therefore rearguard-DOMINATED, not keeper-free. (An earlier
docstring claimed the keeper "cancels"; that was an over-claim that only held under a since-fixed
opponent-selection bug -- see ADR-089 / the arm tests.) "Behind the line" is satisfied by construction
-- xT toward A's own goal concentrates the threat arm there, and DAS is *dangerous* accessible space
(already goal-concentrated), so ghosting only A's rearguard leaves a rearguard-dominated delta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gkdv import GkdvParams, delta_das_batch, delta_threat_suppression_batch
from silly_kicks.id_compat import align_join_keys, canonical_id, ids_differ
from silly_kicks.tracking import GoalEndUnresolvedError

from ._columns import (
    RD_FRAME_KEYS,
    RD_GK_DETER_SPACE,
    RD_GK_DETER_THREAT,
    RD_GK_SOURCE,
    RD_OUTFIELD_DETER_SPACE,
    RD_OUTFIELD_DETER_THREAT,
    RD_OUTFIELD_SOURCE,
    RD_SAMPLE_KEYS,
)
from ._config import RestDefenseParams
from ._counterfactual import build_restdefense_ghost_frames
from ._ghost_report import RestDefenseGhostReport
from ._windows import select_rest_defense_samples

_DEFAULT_PARAMS = RestDefenseParams()
_GKDV_PARAMS = GkdvParams()


def rest_defense_outfield_deterrent(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt,
    ghost_outfield_model,
    home_team_id: int | str,
    goal_map=None,
    links: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    visible_area: pd.DataFrame | None = None,
    params: RestDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, RestDefenseGhostReport]:
    """Outfield-rearguard deterrent arm (spec 7.3). Returns ``(arm_table, RestDefenseGhostReport)``.

    ``arm_table`` keys ``RD_SAMPLE_KEYS`` and carries ``rd_outfield_deter_threat`` /
    ``rd_outfield_deter_space`` (attacker-value units, negative = deterrent) + a shared
    ``rd_outfield_source``. ``xt`` is a required fitted ``ExpectedThreat`` (fail-closed).

    Examples
    --------
    Price the in-possession rearguard's counter-danger suppression vs a league-average ghost, then
    join onto the Layer-1/2 samples (negative == more deterrent)::

        arm, report = rest_defense_outfield_deterrent(
            actions, frames, xt=xt, ghost_outfield_model=ghost_outfield_model, home_team_id=1
        )
        merged = merge_rest_defense(samples, arm)
    """
    return _deterrent(
        actions,
        frames,
        which="rearguard",
        model=ghost_outfield_model,
        xt=xt,
        home_team_id=home_team_id,
        goal_map=goal_map,
        links=links,
        carrier=carrier,
        visible_area=visible_area,
        params=params,
        threat_col=RD_OUTFIELD_DETER_THREAT,
        space_col=RD_OUTFIELD_DETER_SPACE,
        source_col=RD_OUTFIELD_SOURCE,
    )


def rest_defense_gk_deterrent(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    xt,
    ghost_gk_model,
    home_team_id: int | str,
    goal_map=None,
    links: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    visible_area: pd.DataFrame | None = None,
    params: RestDefenseParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, RestDefenseGhostReport]:
    """Keeper deterrent arm (spec 7.3). ``ghost_gk_model`` must be a ``sweeper`` variant (parent 9).

    Examples
    --------
    Price the in-possession keeper's counter-danger suppression vs a league-average sweeper ghost::

        arm, report = rest_defense_gk_deterrent(
            actions, frames, xt=xt, ghost_gk_model=GhostGkModel.from_variant("sweeper"), home_team_id=1
        )
    """
    return _deterrent(
        actions,
        frames,
        which="keeper",
        model=ghost_gk_model,
        xt=xt,
        home_team_id=home_team_id,
        goal_map=goal_map,
        links=links,
        carrier=carrier,
        visible_area=None,  # the GK serve has no visible_area kwarg; keeper FOV is not modelled in v1
        params=params,
        threat_col=RD_GK_DETER_THREAT,
        space_col=RD_GK_DETER_SPACE,
        source_col=RD_GK_SOURCE,
    )


def merge_rest_defense(samples: pd.DataFrame, *arms: pd.DataFrame) -> pd.DataFrame:
    """Left-join arm tables onto the Layer-1/2 samples (honest-NaN on arm-dropped rows; spec 14).

    Each arm's keys MUST be a subset of the sample keys (the arm drop-domain is a declared subset of
    the Layer-1 gate) -- an arm key absent from the samples RAISES, so a grain mismatch is loud, not a
    silent fan-out.

    Examples
    --------
    Left-join both deterrent arms onto the Layer-1/2 samples, keeping every sample (honest-NaN where an
    arm dropped the frame)::

        merged = merge_rest_defense(samples, outfield_arm, gk_arm)
        assert len(merged) == len(samples)
    """
    out = samples
    sample_keys = set(zip(*[list(map(canonical_id, samples[k])) for k in RD_SAMPLE_KEYS], strict=True))
    for arm in arms:
        if not len(arm):
            continue
        arm_key_tuples = set(zip(*[list(map(canonical_id, arm[k])) for k in RD_SAMPLE_KEYS], strict=True))
        extra = arm_key_tuples - sample_keys
        if extra:
            raise ValueError(
                f"merge_rest_defense: {len(extra)} arm key(s) are not in the samples -- the arm "
                "drop-domain must be a SUBSET of the Layer-1 gate (spec 14). Example missing key: "
                f"{next(iter(extra))!r}."
            )
        left, right = align_join_keys(out, arm, list(RD_SAMPLE_KEYS))
        out = left.merge(right, on=RD_SAMPLE_KEYS, how="left")
    return out.reset_index(drop=True)


def _deterrent(
    actions,
    frames,
    *,
    which,
    model,
    xt,
    home_team_id,
    goal_map,
    links,
    carrier,
    visible_area,
    params,
    threat_col,
    space_col,
    source_col,
) -> tuple[pd.DataFrame, RestDefenseGhostReport]:
    from silly_kicks.tracking import resolve_defended_goals

    if goal_map is None:
        goal_map = resolve_defended_goals(frames)

    samples = select_rest_defense_samples(actions, frames, goal_map=goal_map, params=params, links=links)
    scored = samples[samples["gate_drop_reason"].isna()].reset_index(drop=True)

    # The ghost must be built for the SAME in-possession team A the action-grid samples identify --
    # otherwise infer_ball_carrier could pick a different team and the deltas would not align with the
    # samples. Derive the per-frame carrier from the scored samples (their team_id IS A).
    car = carrier if carrier is not None else _sample_carrier(scored)

    cf, prov, report = build_restdefense_ghost_frames(
        frames,
        which=which,
        model=model,
        home_team_id=home_team_id,
        carrier=car,
        visible_area=visible_area if which == "rearguard" else None,
        params=params,
    )
    scored_prov = prov[prov["drop_reason"].isna()]

    empty = _empty_arm(threat_col, space_col, source_col)
    if not len(scored):
        return empty, report

    threat, space = _frame_deltas(frames, cf, scored_prov, goal_map, xt)

    arm = scored[[*RD_SAMPLE_KEYS, "frame_id"]].copy()
    delta = _delta_frame(threat, space, threat_col, space_col)
    if len(delta):
        left, right = align_join_keys(arm, delta, RD_FRAME_KEYS)
        arm = left.merge(right, on=RD_FRAME_KEYS, how="left")
    else:
        arm[threat_col] = np.nan
        arm[space_col] = np.nan

    computed_keys = _frame_key_set(scored_prov)
    arm_keys = list(
        zip(
            (canonical_id(v) for v in arm["game_id"]),
            (canonical_id(v) for v in arm["period_id"]),
            (canonical_id(v) for v in arm["frame_id"]),
            strict=True,
        )
    )
    arm[source_col] = ["computed" if k in computed_keys else "ghost_missing" for k in arm_keys]
    arm[threat_col] = pd.to_numeric(arm[threat_col], errors="coerce").astype("float64")
    arm[space_col] = pd.to_numeric(arm[space_col], errors="coerce").astype("float64")
    return arm[[*RD_SAMPLE_KEYS, threat_col, space_col, source_col]].reset_index(drop=True), report


def _sample_carrier(scored: pd.DataFrame) -> pd.DataFrame:
    """Per-frame carrier (ball_carrier_team_id) from the scored samples' in-possession team A."""
    if not len(scored):
        return pd.DataFrame(columns=[*RD_FRAME_KEYS, "ball_carrier_team_id"])
    return (
        scored[[*RD_FRAME_KEYS, "team_id"]]
        .dropna(subset=["frame_id"])
        .drop_duplicates(RD_FRAME_KEYS)
        .rename(columns={"team_id": "ball_carrier_team_id"})
        .reset_index(drop=True)
    )


def _frame_deltas(frames, cf, scored_prov, goal_map, xt):
    """Per-scored-frame (threat, space) deltas via the gkdv batch seams (attacking team = B)."""
    if not len(scored_prov):
        empty = pd.Series(dtype=float)
        return empty, empty
    fk = set(_frame_key_set(scored_prov))
    mask = np.array(
        [
            (canonical_id(g), canonical_id(p), canonical_id(f)) in fk
            for g, p, f in zip(frames["game_id"], frames["period_id"], frames["frame_id"], strict=True)
        ],
        dtype=bool,
    )
    b_by_frame = _opponent_by_frame(frames, scored_prov)
    # DAS measures B's dangerous accessible space (B = the counter-attacker whose threat behind A's
    # line we price), so team_in_possession = B on BOTH legs -- the gkdv caller-sets-team_in_possession
    # contract (tests/gkdv/test_arms.py). The threat seam takes attacking_team_id explicitly and
    # ignores team_in_possession, so this does not perturb it.
    actual = _with_team_in_possession(frames[mask], b_by_frame)
    ghost = _with_team_in_possession(cf[mask], b_by_frame)
    # ADR-055: the threat leg (compute_threat_pc) REFUSES when goal_map cannot resolve the attacked
    # goal -- e.g. a keeperless frame set (the SB360 gk_absent audit roster, or a real freeze frame
    # with no visible/rostered keeper). Refuse -> honest-NaN, never a crash, exactly as
    # compute_rest_defense (_compute.py) and the gkdv/keeper-arm siblings do. Real full-tracking data
    # always resolves the goal (keepers present), so this never fires there. The DAS leg is
    # goal-blind (it infers direction, takes no goal_map) so it proceeds independently.
    try:
        threat = delta_threat_suppression_batch(
            actual, ghost, attacking_team_id_by_frame=b_by_frame, xt=xt, goal_map=goal_map, params=_GKDV_PARAMS
        )
    except GoalEndUnresolvedError:
        threat = pd.Series(dtype=float)
    space = delta_das_batch(actual, ghost, attacking_team_id_by_frame=b_by_frame, params=_GKDV_PARAMS)
    return threat, space


def _with_team_in_possession(frames_subset: pd.DataFrame, b_by_frame: pd.Series) -> pd.DataFrame:
    """Stamp team_in_possession = B per frame (row-order-preserving; drops any prior column)."""
    f = frames_subset.drop(columns=["team_in_possession"], errors="ignore")
    bdf = b_by_frame.rename("team_in_possession").reset_index()
    bdf.columns = [*RD_FRAME_KEYS, "team_in_possession"]
    left, right = align_join_keys(f, bdf, list(RD_FRAME_KEYS))
    return left.merge(right, on=RD_FRAME_KEYS, how="left")


def _opponent_by_frame(frames, scored_prov) -> pd.Series:
    """Series indexed by (game, period, frame) -> the opponent B of the in-possession team A.

    A is the rearguard team in ``scored_prov``; B is the other team present on that frame.
    """
    non_ball = frames[~frames["is_ball"].astype(bool)]
    fteams = non_ball[[*RD_FRAME_KEYS, "team_id"]].dropna(subset=["team_id"]).drop_duplicates()
    a_per = scored_prov[[*RD_FRAME_KEYS, "team_id"]].drop_duplicates().rename(columns={"team_id": "a_team"})
    left, right = align_join_keys(a_per, fteams, list(RD_FRAME_KEYS))
    cand = left.merge(right, on=RD_FRAME_KEYS, how="left")
    # B = the team that is NOT A. ids_differ is column-vs-column; ids_match is Series-vs-SCALAR, so
    # `ids_match(cand["team_id"], cand["a_team"])` compared every row against the whole a_team Series
    # object -> always False -> `~` all True -> B silently resolved to the FIRST team per frame (== A).
    b = cand[ids_differ(cand["team_id"], cand["a_team"]).fillna(False).to_numpy(dtype=bool)].drop_duplicates(
        RD_FRAME_KEYS
    )
    idx = pd.MultiIndex.from_frame(b[RD_FRAME_KEYS])
    return pd.Series(b["team_id"].to_numpy(), index=idx)


def _delta_frame(threat: pd.Series, space: pd.Series, threat_col, space_col) -> pd.DataFrame:
    if not len(threat):
        return pd.DataFrame(columns=[*RD_FRAME_KEYS, threat_col, space_col])
    df = pd.DataFrame({threat_col: threat, space_col: space}).reset_index()
    df.columns = [*RD_FRAME_KEYS, threat_col, space_col]
    return df


def _frame_key_set(df: pd.DataFrame) -> set:
    if not len(df):
        return set()
    return set(
        zip(
            (canonical_id(v) for v in df["game_id"]),
            (canonical_id(v) for v in df["period_id"]),
            (canonical_id(v) for v in df["frame_id"]),
            strict=True,
        )
    )


def _empty_arm(threat_col, space_col, source_col) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype=object),
            "period_id": pd.Series(dtype=object),
            "team_id": pd.Series(dtype=object),
            "action_id": pd.Series(dtype=object),
            threat_col: pd.Series(dtype="float64"),
            space_col: pd.Series(dtype="float64"),
            source_col: pd.Series(dtype=object),
        }
    )
