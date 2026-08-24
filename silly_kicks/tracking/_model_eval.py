"""Model-agnostic GK-substitution probe core + registered xS rule + re-gate verdict (ADR-037).

Layering (M4): substitution_deltas() is the ONLY function here that touches a model; it
consumes ghost TARGETS AS DATA (a DataFrame) so tracking/ never imports gkdv/. Pure
evaluators (evaluate_xs_probe, regate_verdict) operate on the tidy deltas frame.
`_xcross_eval.py` remains the frozen xCross wrapper's home (byte-equivalent; golden-pinned).
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, canonical_id_series, ids_match
from silly_kicks.spadl import config as _spadlconfig

# --- Pitch rectangle (never hardcode 105/68): the registered off-pitch policy (spec §3.1(2),
# ADR-037 item 15) reports the fraction of displaced controls that leave this rectangle. ---
_FIELD_LENGTH: float = _spadlconfig.field_length  # 105.0
_FIELD_WIDTH: float = _spadlconfig.field_width  # 68.0

# --- Pre-registered TF-19 xCross viability threshold (moved from _xcross_eval; frozen 4.18.0) ---
TF19_PROBE_RATIO = 2.0
TF19_PROBE_ABS_FLOOR = 0.01

# --- Registered xS probe rule (spec §3.1; locked BEFORE any owner run) ---------------------
XS_PROBE_RATIO = 2.0
XS_PROBE_DOSE_M = 2.0  # gated band: |ghost - actual| >= 2 m, trusted stratum only
XS_PROBE_DOSE_LADDER = (2.0, 3.0, 4.0)  # reported, never gated beyond DOSE_M
XS_PROBE_MIN_BAND_N = 100  # min frames in the gated band
XS_PROBE_MIN_STRATUM_N = 50  # min frames in the trusted (unclamped, in-box) stratum
XS_PROBE_PLACEBO_REPLICATES = 20  # R placebo replicates (paired-vector)
XS_PROBE_PLACEBO_BAND_PCT = 95.0
# NOTE: there is deliberately NO gated-band zero-fraction ceiling (spec §3.1(5) as
# amended): past the placebo gate, an all-zero GK band is a CLEAN FAIL, not
# unmeasurable. The fraction is reported, never gated.
XS_PROBE_MAX_PLACEBO_ZERO_FRACTION = 0.95  # non-degeneracy guard (prong 2)
XS_PROBE_DOSE_RESPONSE_ALPHA = 0.05  # prong 4: game-level sign-flip permutation p
XS_PROBE_DOSE_RESPONSE_PERMUTATIONS = 999
XS_PROBE_MIN_GAME_N = 10  # frames a game needs to contribute a per-game rho
XS_PROBE_MIN_GAMES = 8  # games needed for the dose test to be POWERED
#                         (fixtures carry 10-12 games; real GS corpus 64)

_TIDY_COLUMNS = [
    "game_id",
    "period_id",
    "frame_id",
    "actor_role",
    "replicate",
    "displacement_m",
    "delta_p",
    "ghost_clamped",
    "ghost_out_of_box",
    "moved_off_pitch",
]


def _displacement_panel(goal_x: float) -> list[tuple[str, float, float]]:
    """Geometrically-matched (dx, dy) panel applied identically to GK / nearest-def / random
    outfielders so 'same-magnitude' is comparable (A1). 'depth' is signed toward the attacked goal."""
    toward = 1.0 if goal_x >= 105.0 / 2 else -1.0
    return [
        ("lat+2", 0.0, 2.0),
        ("lat-2", 0.0, -2.0),
        ("lat+4", 0.0, 4.0),
        ("lat-4", 0.0, -4.0),
        ("depth+2", toward * 2.0, 0.0),
        ("depth-2", -toward * 2.0, 0.0),
    ]


def _delta_for_move(
    model, grp, row_mask, moves, extract_fn, extract_kwargs, base_p: float | None = None
) -> list[float]:
    """Baseline predict vs each move-perturbed predict for the single player row(s) in row_mask.

    Moved line-for-line from ``_xcross_eval._abs_delta_for_player``; the fixed panel is
    generalized to ``moves`` = list of (dx, dy) and the extractor is injected (arm-selected).
    ``base_p`` optionally supplies the precomputed UNPERTURBED prediction: it depends only on
    (grp, extract_kwargs), never on row_mask/moves, so targets-mode callers compute it once per
    frame and reuse it across the GK, nearest-def, and every placebo replicate (numerics-identical;
    the panel path passes None and keeps its legacy per-call baseline)."""
    mask = np.asarray(row_mask, dtype=bool)
    if base_p is None:
        base_feats = extract_fn(grp, **extract_kwargs)
        base_p = float(model.predict_proba(base_feats)[0])
    deltas = []
    for dx, dy in moves:
        pert = grp.copy()
        pert.loc[mask, "x"] = pert.loc[mask, "x"].to_numpy(float) + dx
        pert.loc[mask, "y"] = pert.loc[mask, "y"].to_numpy(float) + dy
        feats = extract_fn(pert, **extract_kwargs)
        deltas.append(abs(float(model.predict_proba(feats)[0]) - base_p))
    return deltas


def _moved_off_pitch(grp: pd.DataFrame, row_mask, moves: list[tuple[float, float]]) -> list[bool]:
    """Per-move flag: does the displaced actor land OUTSIDE the pitch rectangle
    [0, _FIELD_LENGTH] x [0, _FIELD_WIDTH]? Parallel to ``_delta_for_move``'s deltas.

    REGISTERED off-pitch policy (spec §3.1(2), ADR-037 item 15): a control pushed off-pitch
    by the paired vector is SCORED anyway, never clamped (clamping would break the
    equal-magnitude paired-vector guarantee). This helper only REPORTS the flag; it feeds no
    verdict. The base position is ``grp.loc[mask]`` -- the exact rows ``_delta_for_move``
    displaces -- so moved = base + (dx, dy) matches its perturbation. A multi-row mask (e.g.
    the rare same-team-double-GK case) is off-pitch iff ANY moved row leaves the rectangle."""
    mask = np.asarray(row_mask, dtype=bool)
    bx = grp.loc[mask, "x"].to_numpy(float)
    by = grp.loc[mask, "y"].to_numpy(float)
    flags: list[bool] = []
    for dx, dy in moves:
        mx = bx + dx
        my = by + dy
        flags.append(bool(((mx < 0.0) | (mx > _FIELD_LENGTH) | (my < 0.0) | (my > _FIELD_WIDTH)).any()))
    return flags


def _eligible_groups(frames: pd.DataFrame, model, arm: str, advance_m: float) -> list[tuple]:
    """Collect eligible (resolvable carrier + GK row + in-domain) frame groups deterministically.

    Moved line-for-line from ``_xcross_eval.gk_substitution_probe``; ``arm`` selects ONLY the
    ball-position domain gate (arm='xcross': wide-area; arm='xs': attacking third). Returns
    (grp, gk_team, goal_x, cpid) tuples with grp reset-indexed."""
    from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._xcross_attempt import _in_wide_area

    cp = dict(getattr(model, "carrier_params", None) or {})
    carrier = infer_ball_carrier(frames, **cp) if cp else infer_ball_carrier(frames)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = resolve_defended_goals(frames)

    groups_list = []
    for (gid, pid, _fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        # Defending team(s) = non-ball player rows of the OTHER team. Filter by is_ball (not the
        # string "ball") so a provider/fixture that encodes the ball's team_id differently can't be
        # mistaken for a defending team (it would then have no GK row -> the frame would be dropped).
        non_ball = grp[~grp["is_ball"].astype(bool)]
        # .dropna() guards a non-ball player row with NA team_id (unresolved GS jersey): `pd.NA !=
        # poss_team` is NA -> `if` raises "boolean value of NA is ambiguous" (mirrors prepare/compute).
        defending = [t for t in non_ball["team_id"].dropna().unique() if t != poss_team]
        if not defending:
            continue
        goal_x = goal_map.get(gid, pid, defending[0], allow_guess=True)
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if arm == "xcross":
            if not _in_wide_area(bx, by, goal_x, advance_m):
                continue
        else:  # arm == "xs": the `_ball_in_attacking_third` predicate from _xshot_occurrence.py
            if not (abs(bx - goal_x) <= advance_m):  # NaN bx -> False -> skipped, like _in_wide_area
                continue
        cpid = grp["ball_carrier_player_id"].dropna()
        cpid = cpid.iloc[0] if not cpid.empty else None
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == defending[0])
        if cpid is None or not gk_mask.any():
            continue
        groups_list.append((grp.reset_index(drop=True), defending[0], goal_x, cpid))
    return groups_list


def _resolve_extractor(arm: str):
    """Arm-selected feature extractor (heavy imports stay function-local)."""
    if arm == "xcross":
        from silly_kicks.tracking._xcross_attempt import extract_xcross_features

        return extract_xcross_features
    from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

    return extract_xshot_features


def _extract_kwargs(arm: str, gk_team, goal_x: float, cpid, *, feature_set: str = "faithful") -> dict:
    """Arm-specific extractor kwargs. The probe measures positional sensitivity; score held at
    NaN for arm='xcross' exactly as the legacy probe did (xs features carry no score input).

    ``feature_set`` (default "faithful") threads the SCORED model's variant to the extractor so a
    velocity-less position_only model is probed on its own 15/26-col vector -- a faithful model gets
    the identical extract as before (byte-equivalent; the golden pin is on faithful output)."""
    if arm == "xcross":
        return {
            "gk_team_id": gk_team,
            "goal_x": goal_x,
            "carrier_player_id": cpid,
            "score_differential": float("nan"),
            "feature_set": feature_set,
        }
    return {"gk_team_id": gk_team, "goal_x": goal_x, "feature_set": feature_set}


def _nearest_def_mask(grp: pd.DataFrame, gk_team, cpid) -> np.ndarray | None:
    """Row mask of the defender nearest the carrier (control a); None when unresolvable.

    Moved line-for-line from the ``gk_substitution_probe`` sampling loop."""
    carr = grp[grp["player_id"].astype(str) == str(cpid)]
    defenders = grp[(grp["team_id"] == gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
    if not (len(carr) and len(defenders)):
        return None
    cx, cy = float(carr["x"].iloc[0]), float(carr["y"].iloc[0])
    d2 = (defenders["x"].to_numpy(float) - cx) ** 2 + (defenders["y"].to_numpy(float) - cy) ** 2
    nd_id = defenders["player_id"].to_numpy()[int(np.argmin(d2))]
    return grp["player_id"].to_numpy() == nd_id


def _model_relevant_def_pool(grp: pd.DataFrame, gk_team, cpid, *, k: int = 5) -> np.ndarray:
    """v2 placebo pool (spec §3): the ball-nearest ``k`` DEFENDERS of ``gk_team``, minus the
    ``nearest_def`` (carrier-nearest, v1's control a) by player_id. Mirrors the xS extractor's
    model reference (5 nearest defenders to the BALL; ``def_xy`` is already GK-free, so the
    'minus GK' is a no-op). Returns an array of 0-5 player_ids (4 when nearest_def is among the
    ball-nearest-k, 5 when it is not, fewer on a sparse-defender frame)."""
    ball = grp[grp["is_ball"].astype(bool)]
    if not len(ball):
        return np.empty(0, dtype=object)
    bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
    defenders = grp[
        ids_match(grp["team_id"], gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)
    ]
    if not len(defenders):
        return np.empty(0, dtype=object)
    d2 = (defenders["x"].to_numpy(float) - bx) ** 2 + (defenders["y"].to_numpy(float) - by) ** 2
    pool = defenders["player_id"].to_numpy()[np.argsort(d2, kind="stable")[:k]]
    nd_mask = _nearest_def_mask(grp, gk_team, cpid)
    if nd_mask is not None:
        nd_id = grp["player_id"].to_numpy()[nd_mask][0]
        # nd_id and pool are both from grp["player_id"] (same-source column) -> a raw != is
        # dtype-safe here (ADR-019 both-column same-source); no cross-source scalar involved.
        pool = pool[pool != nd_id]
    return pool


def _attacker_diag_pool(grp: pd.DataFrame, gk_team, cpid, *, k: int = 5) -> np.ndarray:
    """Non-gating diagnostic pool (spec §3): up to ``k`` nearest ATTACKERS (the ~gk_team team,
    non-GK) to the ball, with the carrier (``cpid``) excluded by id. Reported only (actor_role
    'attacker_diag'); NEVER banded by evaluate_xs_probe."""
    ball = grp[grp["is_ball"].astype(bool)]
    if not len(ball):
        return np.empty(0, dtype=object)
    bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
    attackers = grp[
        ~ids_match(grp["team_id"], gk_team) & ~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)
    ]
    # carrier id crosses columns (ball_carrier_player_id vs player_id) -> ADR-019 ids_match.
    attackers = attackers[~ids_match(attackers["player_id"], cpid)]
    if not len(attackers):
        return np.empty(0, dtype=object)
    d2 = (attackers["x"].to_numpy(float) - bx) ** 2 + (attackers["y"].to_numpy(float) - by) ** 2
    return attackers["player_id"].to_numpy()[np.argsort(d2, kind="stable")[:k]]


def _tidy_rows(gid, pid, fid, role, replicate, moves, deltas, off_pitch, ghost_clamped, ghost_out_of_box) -> list[dict]:
    """One tidy row per (move, |delta P|) for a single actor in a single frame. ``off_pitch``
    is the ``_moved_off_pitch`` flag list, parallel to ``moves``/``deltas``."""
    return [
        {
            "game_id": gid,
            "period_id": pid,
            "frame_id": fid,
            "actor_role": role,
            "replicate": replicate,
            "displacement_m": math.hypot(dx, dy),
            "delta_p": d,
            "ghost_clamped": ghost_clamped,
            "ghost_out_of_box": ghost_out_of_box,
            "moved_off_pitch": op,
        }
        for (dx, dy), d, op in zip(moves, deltas, off_pitch, strict=True)
    ]


def _panel_deltas(model, frames, *, arm, n_frames, n_random, seed, advance_m) -> pd.DataFrame:
    """Legacy displacement-panel sampling loop, moved from ``gk_substitution_probe``.

    RNG discipline (byte-equivalence critical): ONE ``default_rng(seed)`` consumed in the LEGACY
    order -- frame subsample first (only when eligible > n_frames), then per-frame outfielder
    picks of size ``min(n_random, len(out_ids))``."""
    extract_fn = _resolve_extractor(arm)
    rng = np.random.default_rng(seed)
    groups_list = _eligible_groups(frames, model, arm, advance_m)

    # Deterministic sample of up to n_frames.
    idx = np.arange(len(groups_list))
    if len(idx) > n_frames:
        idx = np.sort(rng.choice(idx, size=n_frames, replace=False))

    rows: list[dict] = []
    for i in idx:
        grp, gk_team, goal_x, cpid = groups_list[i]
        gid = grp["game_id"].iloc[0]
        pid = grp["period_id"].iloc[0]
        fid = grp["frame_id"].iloc[0]
        moves = [(dx, dy) for _name, dx, dy in _displacement_panel(goal_x)]
        kw = _extract_kwargs(arm, gk_team, goal_x, cpid, feature_set=getattr(model, "feature_set", "faithful"))
        # GK
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == gk_team)
        gk_deltas = _delta_for_move(model, grp, gk_mask, moves, extract_fn, kw)
        gk_off = _moved_off_pitch(grp, gk_mask, moves)
        rows += _tidy_rows(gid, pid, fid, "gk", 0, moves, gk_deltas, gk_off, False, False)
        # Nearest defender to the carrier (control a)
        nd_mask = _nearest_def_mask(grp, gk_team, cpid)
        if nd_mask is not None:
            nd_deltas = _delta_for_move(model, grp, nd_mask, moves, extract_fn, kw)
            nd_off = _moved_off_pitch(grp, nd_mask, moves)
            rows += _tidy_rows(gid, pid, fid, "nearest_def", 0, moves, nd_deltas, nd_off, False, False)
        # Averaged random-outfielder band (control b)
        outs = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
        out_ids = outs["player_id"].to_numpy()
        if len(out_ids):
            pick = rng.choice(out_ids, size=min(n_random, len(out_ids)), replace=False)
            for pick_i, rid in enumerate(pick):
                rb_mask = grp["player_id"].to_numpy() == rid
                rb_deltas = _delta_for_move(model, grp, rb_mask, moves, extract_fn, kw)
                rb_off = _moved_off_pitch(grp, rb_mask, moves)
                rows += _tidy_rows(gid, pid, fid, "placebo_out", pick_i, moves, rb_deltas, rb_off, False, False)
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)


_TARGET_COLUMNS = ("game_id", "period_id", "frame_id", "target_x", "target_y", "ghost_clamped", "ghost_out_of_box")


def _validate_targets(targets: pd.DataFrame) -> None:
    """Loud-fail contract checks on the caller-supplied ghost-target frame. Targets cross a
    package boundary (gkdv-side producers), so malformed input must never degrade silently."""
    missing = [c for c in _TARGET_COLUMNS if c not in targets.columns]
    if missing:
        raise ValueError(f"targets is missing required column(s): {missing}")
    for c in ("target_x", "target_y"):
        if not np.isfinite(targets[c].to_numpy(float)).all():
            raise ValueError(f"targets[{c!r}] must be finite (found NaN/inf)")
    for c in ("ghost_clamped", "ghost_out_of_box"):
        if targets[c].isna().any():
            raise ValueError(
                f"targets[{c!r}] must be non-null: bool(NaN) coerces to True, which would "
                "silently shrink the trusted (unclamped, in-box) stratum"
            )


def _targets_deltas(
    model, frames, *, arm, targets, n_placebo_replicates, seed, advance_m, placebo="random"
) -> pd.DataFrame:
    """Ghost-target substitution: move the GK to the supplied per-frame target; displace each
    control by the SAME per-frame vector (paired-vector, spec §3.1(2)). The targets DataFrame
    defines the evaluation set (no n_frames subsample); eligible frames without a target are
    skipped. Placebo replicate r draws with ``default_rng(seed + r)``."""
    _validate_targets(targets)
    extract_fn = _resolve_extractor(arm)
    key_cols = ["game_id", "period_id", "frame_id"]
    # ADR-019: targets cross a package boundary, so the join keys are canonicalized on BOTH
    # sides before the lookup (Int64(366) / 366.0 / "366" all match; targets-mode only).
    tkey = targets.copy()
    for c in key_cols:
        tkey[c] = canonical_id_series(tkey[c])
    tkey = tkey.set_index(key_cols)
    if not tkey.index.is_unique:
        raise ValueError("targets must carry exactly one row per (game_id, period_id, frame_id)")
    groups_list = _eligible_groups(frames, model, arm, advance_m)

    # First pass: GK + paired-vector nearest-defender rows, plus the per-frame contexts the
    # placebo replicates re-visit (replicates re-draw the OUTFIELDER, never the vector).
    rows: list[dict] = []
    contexts: list[tuple] = []
    for grp, gk_team, goal_x, cpid in groups_list:
        gid = grp["game_id"].iloc[0]
        pid = grp["period_id"].iloc[0]
        fid = grp["frame_id"].iloc[0]
        if (canonical_id(gid), canonical_id(pid), canonical_id(fid)) not in tkey.index:
            continue
        trow = tkey.loc[(canonical_id(gid), canonical_id(pid), canonical_id(fid))]
        ghost_clamped = bool(trow["ghost_clamped"])
        ghost_oob = bool(trow["ghost_out_of_box"])
        kw = _extract_kwargs(arm, gk_team, goal_x, cpid, feature_set=getattr(model, "feature_set", "faithful"))
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == gk_team)
        if int(gk_mask.sum()) > 1:
            warnings.warn(
                f"frame {(gid, pid, fid)!r}: {int(gk_mask.sum())} goalkeeper rows for the defending "
                "team; the substitution vector derives from the FIRST GK row (see the 4.12.x "
                "compute_ghost_gk same-team-GK de-dup for the upstream condition)",
                stacklevel=2,
            )
        gk_x = float(grp.loc[gk_mask, "x"].iloc[0])
        gk_y = float(grp.loc[gk_mask, "y"].iloc[0])
        # The single per-frame paired vector: GK -> target; controls get the SAME (dx, dy).
        # REGISTERED off-pitch policy: a control pushed off-pitch is scored anyway, never clamped.
        moves = [(float(trow["target_x"]) - gk_x, float(trow["target_y"]) - gk_y)]
        # Per-frame baseline cache: the unperturbed prediction is identical for the GK, the
        # nearest defender, and every placebo replicate of this frame -- compute it ONCE
        # (numerics-identical: same (grp, kw) inputs as the per-call baseline it replaces).
        base_p = float(model.predict_proba(extract_fn(grp, **kw))[0])
        gk_deltas = _delta_for_move(model, grp, gk_mask, moves, extract_fn, kw, base_p=base_p)
        gk_off = _moved_off_pitch(grp, gk_mask, moves)
        rows += _tidy_rows(gid, pid, fid, "gk", 0, moves, gk_deltas, gk_off, ghost_clamped, ghost_oob)
        nd_mask = _nearest_def_mask(grp, gk_team, cpid)
        if nd_mask is not None:
            nd_deltas = _delta_for_move(model, grp, nd_mask, moves, extract_fn, kw, base_p=base_p)
            nd_off = _moved_off_pitch(grp, nd_mask, moves)
            rows += _tidy_rows(gid, pid, fid, "nearest_def", 0, moves, nd_deltas, nd_off, ghost_clamped, ghost_oob)
        if placebo == "model_relevant_def":
            out_ids = _model_relevant_def_pool(grp, gk_team, cpid)
            attacker_ids = _attacker_diag_pool(grp, gk_team, cpid)
        else:  # "random" -- the frozen v1 pool: all non-ball, non-GK players of BOTH teams
            out_ids = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]["player_id"].to_numpy()
            attacker_ids = None
        contexts.append((grp, gid, pid, fid, moves, kw, out_ids, attacker_ids, ghost_clamped, ghost_oob, base_p))

    # I1: fail LOUD on zero overlap -- a silent empty result would read as "no signal" when the
    # real cause is a disjoint key set (targets built from a different frames feed, or a
    # key/dtype mismatch beyond the ADR-019 canonicalization above, e.g. renamed/re-keyed ids).
    if groups_list and len(tkey) and not contexts:
        eg = groups_list[0][0]
        eligible_example = (eg["game_id"].iloc[0], eg["period_id"].iloc[0], eg["frame_id"].iloc[0])
        target_example = tuple(targets.iloc[0][key_cols])
        raise ValueError(
            "targets matched ZERO eligible frames on (game_id, period_id, frame_id), even after "
            "ADR-019 key canonicalization ('366' == 366.0 handled) -- the key sets are disjoint. "
            "targets cross a package boundary: check they were built from the SAME frames feed. "
            f"Example eligible key: {eligible_example!r}; example target key: {target_example!r}."
        )

    # Placebo replicates: replicate r uses default_rng(seed + r); ONE outfielder per frame,
    # drawn without replacement, displaced by the frame's paired vector.
    for r in range(n_placebo_replicates):
        rng_r = np.random.default_rng(seed + r)
        for grp, gid, pid, fid, moves, kw, out_ids, attacker_ids, ghost_clamped, ghost_oob, base_p in contexts:
            if len(out_ids):  # placebo draw FIRST -> v1 rng stream is byte-identical
                rid = rng_r.choice(out_ids, size=1, replace=False)[0]
                pl_mask = grp["player_id"].to_numpy() == rid
                pl_deltas = _delta_for_move(model, grp, pl_mask, moves, extract_fn, kw, base_p=base_p)
                pl_off = _moved_off_pitch(grp, pl_mask, moves)
                rows += _tidy_rows(gid, pid, fid, "placebo_out", r, moves, pl_deltas, pl_off, ghost_clamped, ghost_oob)
            if attacker_ids is not None and len(attacker_ids):  # v2 only: reported, never banded
                aid = rng_r.choice(attacker_ids, size=1, replace=False)[0]
                a_mask = grp["player_id"].to_numpy() == aid
                a_deltas = _delta_for_move(model, grp, a_mask, moves, extract_fn, kw, base_p=base_p)
                a_off = _moved_off_pitch(grp, a_mask, moves)
                rows += _tidy_rows(gid, pid, fid, "attacker_diag", r, moves, a_deltas, a_off, ghost_clamped, ghost_oob)
    return pd.DataFrame(rows, columns=_TIDY_COLUMNS)


def substitution_deltas(
    model,
    frames: pd.DataFrame,
    *,
    arm: str,  # "xcross" | "xs" -- selects extractor + domain gate
    mode: str,  # "panel" (legacy displacement panel) | "targets"
    targets: pd.DataFrame | None = None,
    n_frames: int = 200,
    n_random: int = 3,  # panel-mode outfielder pick count (legacy n_random)
    n_placebo_replicates: int = XS_PROBE_PLACEBO_REPLICATES,
    seed: int = 42,
    advance_m: float = 35.0,
    placebo: str = "random",  # "random" (frozen v1 pool) | "model_relevant_def" (v2, spec §3)
) -> pd.DataFrame:
    """Tidy per-(frame, actor, move) |delta P|: columns game_id, period_id, frame_id,
    actor_role ('gk'|'nearest_def'|'placebo_out'|'attacker_diag'), replicate, displacement_m,
    delta_p, ghost_clamped, ghost_out_of_box, moved_off_pitch. mode='targets' moves the GK to the
    SUPPLIED target and displaces each control by the SAME per-frame vector (paired-vector
    controls, spec §3.1(2)); placebo replicates re-draw the outfielder, never the vector.
    ``placebo`` selects the targets-mode placebo pool: 'random' (frozen v1: any outfielder) or
    'model_relevant_def' (v2, spec §3: the ball-nearest defenders, PLUS non-gating 'attacker_diag'
    rows the evaluator ignores). Panel mode ignores ``placebo`` (no placebo replicates).
    REGISTERED off-pitch policy: a control pushed off-pitch is scored anyway, never
    clamped (clamping would break the paired-vector guarantee); ``moved_off_pitch`` flags
    it per row and ``evaluate_xs_probe`` reports the control fraction (report-only).

    ``replicate`` semantics differ by mode: panel mode tags placebo rows with the outfielder
    PICK index (0..n_random-1, single legacy rng); targets mode tags them with the placebo
    REPLICATE id (0..n_placebo_replicates-1, rng ``seed + r``); gk/nearest_def rows carry 0 in
    both. Task-3's evaluator (evaluate_xs_probe) consumes TARGETS-mode frames only.
    ``advance_m`` is the wide-area advance gate for arm='xcross' and the attacking-third depth
    for arm='xs' -- numerically both 35.0 today; the xs meaning mirrors
    ``_xshot_occurrence._ATTACKING_THIRD_M``, so a move of that constant must be reflected here."""
    if arm not in ("xcross", "xs"):
        raise ValueError(f"unknown arm: {arm!r} (expected 'xcross' or 'xs')")
    if mode not in ("panel", "targets"):
        raise ValueError(f"unknown mode: {mode!r} (expected 'panel' or 'targets')")
    if mode == "panel" and targets is not None:
        raise ValueError("targets supplied but ignored in panel mode; use mode='targets' to consume them")
    if placebo not in ("random", "model_relevant_def"):
        raise ValueError(f"unknown placebo: {placebo!r} (expected 'random' or 'model_relevant_def')")
    if mode == "targets":
        if targets is None:
            raise ValueError("mode='targets' requires a targets DataFrame (one row per frame triple)")
        return _targets_deltas(
            model,
            frames,
            arm=arm,
            targets=targets,
            n_placebo_replicates=n_placebo_replicates,
            seed=seed,
            advance_m=advance_m,
            placebo=placebo,
        )
    return _panel_deltas(model, frames, arm=arm, n_frames=n_frames, n_random=n_random, seed=seed, advance_m=advance_m)


def _dose_response_clustered(gk: pd.DataFrame, *, seed: int = 42) -> tuple[float, float, int]:
    """Cluster-EXACT dose-response (review N1: replaces the equal-block subsample,
    whose min-truncation silently degenerated to a row permutation at m=1 and whose
    power collapse could manufacture a flat-dose veto): per-game Spearman rho, then a
    sign-flip permutation test on the GAME-level rhos. Raggedness is native -- a
    400-frame game contributes a well-estimated rho, a 12-frame game a noisy one,
    NOTHING is truncated; the permutation unit IS the game. Returns (mean_rho, p,
    n_games_used).

    Conventions (registered): a game with constant delta_p gets rho = 0.0 -- zero
    response variance is a MEASURED flat response, not a missing measurement; games
    with < XS_PROBE_MIN_GAME_N frames or constant displacement are skipped (cannot
    measure). Games iterate in sorted order so the seeded sign matrix pairs
    deterministically regardless of incoming row order."""
    from scipy.stats import spearmanr

    rhos = []
    gid_str = gk["game_id"].astype(str)  # hoisted: one astype for the whole loop, numerics-identical
    for gid in sorted(gid_str.unique()):
        g = gk[gid_str == gid]
        if len(g) < XS_PROBE_MIN_GAME_N or g["displacement_m"].nunique() < 2:
            continue
        if g["delta_p"].nunique() < 2:
            rhos.append(0.0)  # constant response == measured FLAT, not unmeasured
            continue
        # spearmanr returns a SignificanceResult whose `.statistic` is the correlation; the scipy
        # stub types the result as an opaque class that omits the attribute, so ignore the stub gap
        # (house idiom -- mirrors scripts/calibrate_xt_bandwidth.py).
        r = spearmanr(g["displacement_m"], g["delta_p"]).statistic  # type: ignore[reportAttributeAccessIssue]
        if np.isfinite(r):
            rhos.append(float(r))
    arr = np.asarray(rhos, dtype=float)
    if len(arr) < XS_PROBE_MIN_GAMES:
        return float("nan"), 1.0, len(arr)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(XS_PROBE_DOSE_RESPONSE_PERMUTATIONS, len(arr)))
    null = (signs * arr).mean(axis=1)
    p = float((np.sum(null >= obs) + 1) / (XS_PROBE_DOSE_RESPONSE_PERMUTATIONS + 1))
    return obs, p, len(arr)


def evaluate_xs_probe(deltas: pd.DataFrame) -> dict:
    """PURE registered xS verdict over a substitution_deltas() frame (spec §3.1).
    Verdicts: 'pass' | 'fail' | 'unmeasurable_at_dose' | 'no_valid_placebo' |
    'band_pass_flat_dose_response'. Every prong is a registered constant; the
    ladder/unbanded/OOD/zero-fraction numbers are report-only."""
    if deltas["delta_p"].isna().any():
        raise ValueError(
            "evaluate_xs_probe: NaN delta_p present -- input integrity failure (a NaN-poisoned "
            "GK band would otherwise fail OPEN into the pre-registered expected 'fail'; cf. the "
            "4.18.0 all-NaN canonical-id class)"
        )
    gk_all = deltas[deltas["actor_role"] == "gk"]
    trusted = gk_all[~gk_all["ghost_clamped"].astype(bool) & ~gk_all["ghost_out_of_box"].astype(bool)]
    band = trusted[trusted["displacement_m"] >= XS_PROBE_DOSE_M]
    # Semi-join keys: dedup so the banded-control merge can never fan out; a GK band with
    # duplicate key triples is an input-contract violation (this evaluator consumes
    # TARGETS-mode deltas -- one GK row per frame; panel-mode-shaped input, with one row
    # per displacement move, is the likely cause) and must raise, not silently re-weight.
    # This check precedes the floor exits below: duplicate keys make len(band) count
    # rows-not-frames, corrupting the very unit the floors test -- input-integrity checks
    # (like the NaN guard above) come before semantic exits. (Empty band: 0 == 0, no raise.)
    frame_keys = band[["game_id", "period_id", "frame_id"]].drop_duplicates()
    if len(frame_keys) != len(band):
        raise ValueError(
            "evaluate_xs_probe: duplicate (game_id, period_id, frame_id) GK rows in the gated "
            "band -- the evaluator consumes TARGETS-mode deltas (one GK row per frame); "
            "panel-mode-shaped input (multiple displacement rows per frame) is the likely cause"
        )
    # Report-only (spec §3.1(2), ADR-037 item 15): the fraction of CONTROL displacements that
    # land off-pitch. The registered paired-vector policy SCORES off-pitch controls (never
    # clamps -- that would break the equal-magnitude guarantee); this only reports the
    # fraction, it feeds NO verdict branch. Defensive: a pre-fix deltas frame (no column) -> NaN.
    if "moved_off_pitch" in deltas.columns:
        controls = deltas[deltas["actor_role"].isin(("nearest_def", "placebo_out"))]
        off_pitch_control_fraction = (
            float(controls["moved_off_pitch"].astype(bool).mean()) if len(controls) else float("nan")
        )
    else:
        off_pitch_control_fraction = float("nan")
    report: dict = {
        "rule": "xs-dose-banded-v1",
        "dose_ladder": {
            float(d): float(trusted.loc[trusted["displacement_m"] >= d, "delta_p"].median())
            if (trusted["displacement_m"] >= d).any()
            else float("nan")
            for d in XS_PROBE_DOSE_LADDER
        },
        "unbanded_median": float(gk_all["delta_p"].median()) if len(gk_all) else float("nan"),
        "trusted_stratum": len(trusted),
        "ood_stratum": int(len(gk_all) - len(trusted)),
        "gated_band_n": len(band),
        "gated_band_zero_fraction": float((band["delta_p"] == 0).mean()) if len(band) else float("nan"),
        "off_pitch_control_fraction": off_pitch_control_fraction,
    }
    if len(trusted) < XS_PROBE_MIN_STRATUM_N or len(band) < XS_PROBE_MIN_BAND_N:
        report["dose_state"] = "not_run"  # stable discriminator for verdict consumers
        report["verdict"] = "unmeasurable_at_dose"
        return report

    def _banded(role: str) -> pd.DataFrame:
        sub = deltas[deltas["actor_role"] == role]
        return sub.merge(frame_keys, on=["game_id", "period_id", "frame_id"])

    nd = _banded("nearest_def")
    placebo = _banded("placebo_out")
    # prong 2: placebo replicates of the SAME functional + non-degeneracy (fail-closed)
    rep_medians = placebo.groupby("replicate")["delta_p"].median()
    placebo_zero_fraction = float((placebo["delta_p"] == 0).mean()) if len(placebo) else 1.0
    report["placebo_replicate_medians"] = [float(v) for v in rep_medians]
    report["placebo_p95"] = (
        float(np.percentile(rep_medians, XS_PROBE_PLACEBO_BAND_PCT)) if len(rep_medians) else float("nan")
    )
    report["placebo_zero_fraction"] = placebo_zero_fraction
    nd_med = float(nd["delta_p"].median()) if len(nd) else float("nan")
    report["nearest_def_median"] = nd_med
    if (
        not np.isfinite(report["placebo_p95"])
        or report["placebo_p95"] <= 0.0
        or placebo_zero_fraction > XS_PROBE_MAX_PLACEBO_ZERO_FRACTION
        or not (np.isfinite(nd_med) and nd_med > 0.0)  # M2 analog
    ):
        report["dose_state"] = "not_run"  # stable discriminator for verdict consumers
        report["verdict"] = "no_valid_placebo"
        return report

    gk_med = float(band["delta_p"].median())
    report["gated_band_median"] = gk_med
    # Zero-inflation is a REPORTED DIAGNOSTIC, never an early return (review B1): zeros
    # have two causes and only the CONTROLS disambiguate — dead controls were already
    # caught fail-closed above as no_valid_placebo, so an all-zero GK band here can
    # only mean the keeper does not move the surface: a CLEAN FAIL (gk_med = 0 ->
    # band_pass False below), the cycle's expected, publishable outcome. A ceiling was
    # also outcome-inert for passes (zero-fraction > 0.5 forces median 0).

    # Cluster-exact dose-response over the trusted stratum (review B4 + N1): per-game
    # rho, sign-flip permutation across games. Same population, all data, no truncation.
    rho_obs, dose_p, n_games = _dose_response_clustered(trusted)
    report["dose_response_rho"] = rho_obs
    report["dose_response_p"] = dose_p
    report["dose_response_n_games"] = n_games
    # Three dose states (N1 point 3, generalized): 'ok' | 'flat' (test RAN, no positive
    # monotone response) | 'underpowered' (too few measurable games). Low power must not
    # manufacture the flat verdict — but it must not let a band pass stand alone either:
    # underpowered + band pass routes to the SUPPORT verdict, unmeasurable_at_dose.
    if n_games < XS_PROBE_MIN_GAMES:
        dose_state = "underpowered"
    elif np.isfinite(rho_obs) and rho_obs > 0 and dose_p < XS_PROBE_DOSE_RESPONSE_ALPHA:
        dose_state = "ok"
    else:
        dose_state = "flat"
    report["dose_state"] = dose_state

    # ratio vs max(control, placebo band): a deliberate strengthening over the spec's
    # nearest-defender-only prong (recorded in ADR-037); an explicit gk_med > p95
    # conjunct is redundant given ratio >= 2 and the p95 > 0 guard above.
    band_pass = gk_med >= XS_PROBE_RATIO * max(nd_med, report["placebo_p95"])
    if band_pass and dose_state == "ok":
        report["verdict"] = "pass"
    elif band_pass and dose_state == "flat":
        report["verdict"] = "band_pass_flat_dose_response"
    elif band_pass and dose_state == "underpowered":  # support verdict, never a manufactured flat
        report["verdict"] = "unmeasurable_at_dose"
    elif not band_pass:
        report["verdict"] = "fail"
    else:  # unreachable today: every dose_state is enumerated above
        raise ValueError(f"evaluate_xs_probe: unknown dose_state {dose_state!r}")
    return report


def xs_substitution_probe(model, frames, targets, *, seed: int = 42) -> dict:
    """The registered xS probe: produce deltas in targets-mode, evaluate the registered rule."""
    deltas = substitution_deltas(model, frames, arm="xs", mode="targets", targets=targets, seed=seed)
    out = evaluate_xs_probe(deltas)
    # Full triple, not bare frame_id: production frame ids restart per game/period, and
    # nunique() on frame_id alone would undercount to ~max-frames-per-game (exec review).
    gk = deltas[deltas["actor_role"] == "gk"]
    out["n_frames_used"] = len(gk[["game_id", "period_id", "frame_id"]].drop_duplicates())
    return out


def xs_substitution_probe_v2(model, frames, targets, *, seed: int = 42) -> dict:
    """The v2 xS probe (ADR-037 amendment): same rule as v1, but the placebo pool is the
    model-relevant defenders (``placebo='model_relevant_def'``) instead of random outfielders.
    Reuses ``evaluate_xs_probe`` verbatim; relabels the report ``rule`` (the pure evaluator is
    unchanged). See docs/superpowers/specs/2026-07-23-tf19-xs-placebo-v2-design.md."""
    deltas = substitution_deltas(
        model, frames, arm="xs", mode="targets", targets=targets, seed=seed, placebo="model_relevant_def"
    )
    out = evaluate_xs_probe(deltas)
    out["rule"] = "xs-dose-banded-v2"  # wrapper-level relabel; evaluator emits v1's constant
    out["placebo_pool"] = "model_relevant_def"
    gk = deltas[deltas["actor_role"] == "gk"]
    out["n_frames_used"] = len(gk[["game_id", "period_id", "frame_id"]].drop_duplicates())
    return out


# --- Re-gate verdict (spec §3.5) -----------------------------------------------------------
_PROBE_VERDICTS = frozenset(
    {
        "pass",
        "fail",
        "band_pass_flat_dose_response",
        "unmeasurable_at_dose",
        "no_valid_placebo",
        "instrument_invalid",
    }
)
# 'degenerate' (S6): the causal harness can genuinely return no-positivity/empty-overlap
# (it already reports claim_supported) -- a real-world outcome, not a caller error.
_ENTANGLEMENT = frozenset({"clears", "inside_band", "degenerate"})


def regate_verdict(*, arm: str, probe_verdict: str, entanglement: str) -> str:
    """Spec §3.5 as a pure function. `entanglement` = the §3.3 GK-confounder-entanglement
    result (supportive context, NOT a causal deterrence estimate). Arms are independent;
    GKDV v1 (physics arms) ships regardless of every outcome here. An all-zero GK band
    with live controls arrives as probe_verdict='fail' (clean, publishable), never as
    an unmeasurable state (review B1)."""
    if arm not in ("shot", "cross") or probe_verdict not in _PROBE_VERDICTS or entanglement not in _ENTANGLEMENT:
        raise ValueError(f"regate_verdict: unknown input {(arm, probe_verdict, entanglement)!r}")
    if probe_verdict == "instrument_invalid":
        return "verdict_void"
    if probe_verdict in ("unmeasurable_at_dose", "no_valid_placebo"):
        return "unmeasurable_at_dose"
    if probe_verdict == "band_pass_flat_dose_response":
        return "gated_flat_dose_response"
    if probe_verdict == "fail":
        return "gated_clean_fail"
    return "joins" if entanglement == "clears" else "joins_with_caveat"


#: Closed routing vocabulary (the ``DAS_SOURCE_VALUES`` pattern): a consumer CASE/enum pins to this
#: set rather than to free-text prose in an ADR.
REGATE_ROUTING_VALUES: tuple[str, ...] = (
    "pending_layer2",
    "gk_feature_engineering",
    "fix_the_instrument",
    "corpus_or_sampling",
    "joins_the_metric",
)

_ROUTING: dict[str, str] = {
    # ADR-037's routing rule is AMENDED here, against its OWN pre-registered disclosure (TF-19 spec
    # §6.4 Registration disclosures: "`regate_verdict`'s routing needs amending ... since that
    # hard-codes H1"). It previously routed `gated_clean_fail` UNCONDITIONALLY to GK feature
    # engineering, which made H2 unreachable by construction -- the function's whole signature has no
    # input through which H2 could be expressed. H2 remains reachable ONLY through row 7 of
    # `gkdv_discrimination_verdict` (PR-3b); this opens the channel without pre-empting the decider.
    "gated_clean_fail": "pending_layer2",
    "gated_flat_dose_response": "pending_layer2",
    "unmeasurable_at_dose": "corpus_or_sampling",
    "verdict_void": "fix_the_instrument",
    "joins": "joins_the_metric",
    "joins_with_caveat": "joins_the_metric",
}


def regate_routing(verdict: str) -> str:
    """What to DO about a :func:`regate_verdict` result -- deliberately a SEPARATE function.

    The verdict answers "what did the probe say"; the routing answers "what should we do about it".
    Only the second may legitimately depend on Layer 2, and fusing them is what hard-coded H1.
    `regate_verdict` is unchanged and stays byte-identical -- every recorded verdict still stands.

    Examples
    --------
    >>> regate_routing("gated_clean_fail")
    'pending_layer2'
    >>> regate_routing("joins_with_caveat")
    'joins_the_metric'
    """
    try:
        return _ROUTING[verdict]
    except KeyError:
        raise ValueError(f"regate_routing: unknown verdict {verdict!r}") from None


# --- Probe-wrapper registry (spec §7): every arm registers its wrapper + rule constants ----
PROBE_WRAPPERS: dict[str, dict] = {}


def _register_wrapper(name: str, wrapper, rule_constants: dict) -> None:
    PROBE_WRAPPERS[name] = {"wrapper": wrapper, "rule_constants": dict(rule_constants)}


def _xcross_wrapper(*args, **kwargs):
    from silly_kicks.tracking._xcross_eval import gk_substitution_probe  # lazy: no top-level cycle

    return gk_substitution_probe(*args, **kwargs)


_register_wrapper("xcross", _xcross_wrapper, {"ratio": TF19_PROBE_RATIO, "abs_floor": TF19_PROBE_ABS_FLOOR})
_register_wrapper(
    "xs",
    xs_substitution_probe,
    {
        "ratio": XS_PROBE_RATIO,
        "dose_m": XS_PROBE_DOSE_M,
        "min_band_n": XS_PROBE_MIN_BAND_N,
        "min_stratum_n": XS_PROBE_MIN_STRATUM_N,
        "placebo_replicates": XS_PROBE_PLACEBO_REPLICATES,
        "placebo_band_pct": XS_PROBE_PLACEBO_BAND_PCT,
        "max_placebo_zero_fraction": XS_PROBE_MAX_PLACEBO_ZERO_FRACTION,
        "dose_response_alpha": XS_PROBE_DOSE_RESPONSE_ALPHA,
        # Exec review: every constant that shapes the verdict belongs in the registry so an
        # introspected manifest is complete.
        "dose_ladder": XS_PROBE_DOSE_LADDER,
        "min_game_n": XS_PROBE_MIN_GAME_N,
        "min_games": XS_PROBE_MIN_GAMES,
        "dose_response_permutations": XS_PROBE_DOSE_RESPONSE_PERMUTATIONS,
    },
)
_register_wrapper(
    "xs_v2",
    xs_substitution_probe_v2,
    {
        **PROBE_WRAPPERS["xs"]["rule_constants"],  # identical numeric rule (spec §2 D3)
        "placebo_pool": "model_relevant_def",  # the ONE difference, self-documented
    },
)
