"""Task 4: failure-mode classification + the trajectory-weak-labelled failed-pass validation set (H1).

The receiver model trains on completed passes but is DEPLOYED on failed passes; completed-pass
accuracy cannot bound failed-pass error (H1). For **intercepted** failures whose ball travelled a
usable distance toward a teammate before being cut out, the release->intercept trajectory weak-labels
the intended lane -- a DIRECT, if partial, failed-pass accuracy estimate. Two honesty limits ride on
every number:

- **R1** -- the covered subset is the EASY TAIL (clear-trajectory interceptions), so the accuracy is an
  UPPER BOUND, not an estimate; the uncovered failures (foot-blocked, immediately-out, ambiguous) are
  the hard cases the model is still applied to.
- **R4** -- the label uses a FORWARD-PROJECTED meeting point (not the endpoint), so through-balls
  conflate model error with (reduced) label error.

The trajectory is used for VALIDATION only; the model never sees it (the leakage ban is a
training-feature constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, ids_match, same_id
from silly_kicks.spadl.config import actiontypes, results
from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.tracking._receiver import NoReleaseDirectionError, geometric_proxy_receiver

_T = {n: i for i, n in enumerate(actiontypes)}
_R = {n: i for i, n in enumerate(results)}
_PASS_TYPES = {_T["pass"], _T["cross"]}
_RESTART_TYPES = {_T["throw_in"], _T["goalkick"], _T["corner_crossed"], _T["corner_short"]}

R1_CAVEAT = "accuracy is an UPPER BOUND on the easy tail (clear-trajectory interceptions), not an estimate"
R4_CAVEAT = "weak label uses a forward-projected meeting point; through-balls conflate model + label error"


def _isna(v) -> bool:
    return v is pd.NA or (isinstance(v, float) and np.isnan(v))


def classify_failure_mode(actions: pd.DataFrame) -> pd.Series:
    """Per-action failure mode from the chronological NEXT TOUCH, indexed by ``action_id``.

    ``intercepted`` (opponent won it in open play) / ``out`` (dead-ball restart) / ``other`` (incl. every
    non-failed pass). Routes through the 4.89.0 chronological sort so a non-chronological mart is robust.

    The "next" is the next TOUCH within the failed pass's ``(game_id, period_id)`` group -- ``non_action``
    and ``foul`` rows are SKIPPED (GS emits non-touch ``non_action`` rows; neither is a ball touch, M1),
    and the group-scoped shift never crosses a period/game boundary (M2). This mirrors
    ``_resolve_next_touch_positions``' "next touch" notion, so an interspersed noise row cannot false-tag
    an interception, nor a period-end pass borrow the next period's kickoff.
    """
    from silly_kicks.spadl.utils import _sort_actions_chronological_or_action_id

    non_touch = {_T["non_action"], _T["foul"]}
    a = _sort_actions_chronological_or_action_id(actions).reset_index(drop=True)
    touch = a[~a["type_id"].isin(non_touch)]  # a failed pass is itself a touch, so it is retained
    g = touch.groupby(["game_id", "period_id"])
    next_type = g["type_id"].shift(-1).reindex(a.index)  # NaN on non-touch rows + period-last touches
    next_team = g["team_id"].shift(-1).reindex(a.index)
    is_failed_pass = a["type_id"].isin(_PASS_TYPES) & (a["result_id"] == _R["fail"])
    out_mask = next_type.isin(_RESTART_TYPES)
    opp_mask = pd.Series(
        [(not _isna(nt)) and (not same_id(t, nt)) for t, nt in zip(a["team_id"], next_team, strict=True)],
        index=a.index,
    )
    mode = np.where(out_mask, "out", np.where(opp_mask, "intercepted", "other"))
    labelled = pd.Series(np.where(is_failed_pass, mode, "other"), index=a["action_id"].to_numpy())
    return labelled.reindex(actions["action_id"].to_numpy())


def _teammates(frame: pd.DataFrame, team, passer_id) -> pd.DataFrame:
    non_ball = frame[~frame["is_ball"].to_numpy(dtype=bool)]  # to_numpy NOT astype on a possibly-object col (ADR-019)
    return non_ball[ids_match(non_ball["team_id"], team) & ~ids_match(non_ball["player_id"], passer_id)]


def trajectory_weak_labels(
    actions: pd.DataFrame, frames: pd.DataFrame, *, links: pd.DataFrame | None = None, min_travel_m: float = 5.0
) -> pd.DataFrame:
    """One row per intercepted failed pass: the weak-labelled intended receiver + a ``covered`` flag.

    ``covered`` iff the ball travelled >= ``min_travel_m`` toward an unambiguous teammate on the
    forward-projected release->intercept ray (R4). Uncovered rows are the R1 hard tail.
    """
    from scripts._rq_corpus import _acting_attacks_rtl, to_frame_coords

    if links is None:
        links, _ = link_actions_to_frames(actions, frames)
    fmode = dict(zip(actions["action_id"].to_numpy(), classify_failure_mode(actions).to_numpy(), strict=True))
    frame_of = dict(zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=True))
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}
    rows = []
    for _, act in actions.iterrows():
        aid = act["action_id"]
        if fmode.get(aid) != "intercepted":
            continue
        start = np.array([float(act["start_x"]), float(act["start_y"])], dtype=np.float64)
        end = np.array([float(act["end_x"]), float(act["end_y"])], dtype=np.float64)
        travel = float(np.linalg.norm(end - start))  # magnitude is reflection-invariant -> gate in either frame
        weak_id, covered = pd.NA, False
        fr = by_frame.get(canonical_id(frame_of.get(aid)))
        if travel >= min_travel_m and fr is not None:
            # Reproject the action-LTR release+end into the FRAME convention (ADR-028) so the ray and the
            # frame teammates share one coordinate system; otherwise an away action mixes an action-LTR
            # release with frame teammates and weak-labels the wrong lane (C-H1).
            attacks_rtl = _acting_attacks_rtl(fr, act["team_id"])
            release = np.array(to_frame_coords(start[0], start[1], attacks_rtl), dtype=np.float64)
            end_f = np.array(to_frame_coords(end[0], end[1], attacks_rtl), dtype=np.float64)
            u = (end_f - release) / travel
            best_perp, best_id = np.inf, pd.NA
            for _, tm in _teammates(fr, act["team_id"], act["player_id"]).iterrows():
                rel = np.array([float(tm["x"]) - release[0], float(tm["y"]) - release[1]], dtype=np.float64)
                proj = float(rel @ u)
                if proj <= 0.0:  # forward of the passer only (projected meeting point, R4)
                    continue
                perp = float(np.linalg.norm(rel - proj * u))
                if perp < best_perp:
                    best_perp, best_id = perp, canonical_id(tm["player_id"])
            if best_id is not pd.NA and best_perp <= min_travel_m:  # unambiguous: within a lane width
                weak_id, covered = best_id, True
        rows.append({"action_id": aid, "weak_receiver_id": weak_id, "covered": covered})
    return pd.DataFrame(rows, columns=["action_id", "weak_receiver_id", "covered"])


def receiver_failed_pass_accuracy(model, actions: pd.DataFrame, frames: pd.DataFrame, *, links=None) -> dict:
    """Model vs geometric-proxy top-1 on the trajectory-validated failed subset (H1) -- an UPPER BOUND (R1)."""
    if links is None:
        links, _ = link_actions_to_frames(actions, frames)
    labels = trajectory_weak_labels(actions, frames, links=links)
    covered = labels[labels["covered"]]
    frame_of = dict(zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=True))
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}
    act_of = {a["action_id"]: a for _, a in actions.iterrows()}
    n = model_hits = proxy_hits = 0
    for _, lab in covered.iterrows():
        fr = by_frame.get(canonical_id(frame_of.get(lab["action_id"])))
        act = act_of.get(lab["action_id"])
        if fr is None or act is None:
            continue
        ranked = model.rank(act, fr)
        if ranked.empty:
            continue
        n += 1
        truth = str(lab["weak_receiver_id"])
        if str(ranked.index[0]) == truth:
            model_hits += 1
        try:
            proxy = geometric_proxy_receiver(act, fr)
        except NoReleaseDirectionError:
            proxy = None  # ball-less frame -> the velocity-based proxy can't compute (Q5); model still scored
        if proxy is not None and str(proxy) == truth:
            proxy_hits += 1
    n_intercepted = int((classify_failure_mode(actions) == "intercepted").sum())
    return {
        "top1": model_hits / n if n else float("nan"),
        "top1_proxy": proxy_hits / n if n else float("nan"),
        "n_scored": n,
        "n_covered": int(covered.shape[0]),
        "n_intercepted": n_intercepted,
        "coverage": (covered.shape[0] / n_intercepted) if n_intercepted else float("nan"),
        "r1_caveat": R1_CAVEAT,
        "r4_caveat": R4_CAVEAT,
    }
