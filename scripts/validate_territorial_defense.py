"""TF-54b territorial-defense construct-validity battery (owner-run, reported-not-gated).

Runs the SB360 probe battery over a corpus and writes a report -- it does NOT change any library
default (promotion is a separate ADR-009 decision after the owner reads the report). The corpus map
is ``for_each`` (ADR-052: per-match shards, resumable, conserving); the pooled Layer-0/1 verdicts are
CORPUS statistics computed in the REDUCE over ALL shards, NEVER per shard. Clean-tree guard runs
FIRST (ADR-037); the input contract declares which symbols the numbers depend on (ADR-056).

**THE PROBE BATTERY (spec S9), mirroring ``build_tf19_instrument_responsiveness``:** per SCORED
Arm-A frame the driver imposes a positional DOSE on the defender D and measures the RESPONSE as the
change in ``compute_threat_pc`` (NOT the removal arm), so the pooled reduce can emit the
instrument-validity (Layer 0) and responsiveness (Layer 1) verdicts:

* Layer 0 (instrument validity): ``realistic_abs`` (a 2 m dose) vs ``saturating_abs`` (a 10 m = 5x
  dose in the SAME direction) -- a live instrument moves the saturating dose >> the realistic one,
  or clears the placebo band.
* Layer 1 (responsiveness): the dosed-defender median vs ``nd_abs`` (the nearest defending-team
  outfielder displaced by the SAME realistic vector) and the R single-player placebo band.

**HONEST LIMIT (recorded in the report):** the metric is validated as an INSTRUMENT, not as
player-attributable -- the marginal-removal delta is team-conditioned by construction, and on a
single-tournament / national-team corpus the defender-vs-team confound is unidentifiable, so the
elite-defender prior is a FACE-VALIDITY check (elite-defender / elite-team collinear), NOT a
defender ranking. A crossed defender+team ICC over a multi-club transfer corpus is the future gate.

Usage (owner):
    python scripts/validate_territorial_defense.py --out docs/research/territorial_defense_construct_validity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs
from silly_kicks._frame_index import group_rows
from silly_kicks.keeper_identity import apply_actor_identities_to_frames

# territorial_defense was DEMOTED to experimental (ADR-090); the metric is retained on the private
# ``._compute`` path -- this reported-not-gated battery consumes it from there for the redesign work.
from silly_kicks.territorial_defense._arms import arm_a_threat_suppressed
from silly_kicks.territorial_defense._compute import compute_territorial_defense
from silly_kicks.territorial_defense._config import TerritorialDefenseParams
from silly_kicks.territorial_defense._engine import (
    action_ltr_goal_map,
    classify_arm_a_domain,
    remove_player_row,
    select_arm_a_domain,
)
from silly_kicks.territorial_defense._probe import (
    MIN_DOMAIN_FRAMES,
    SATURATING_MULTIPLE,
    TD_PROBE_RATIO,
    expected_direction_for_arm,
    impose_defender_dose,
    layer0_instrument_verdict,
    layer1_responsiveness_verdict,
    paired_vector_controls,
)
from silly_kicks.tracking import (
    GoalEndUnresolvedError,
    SpearmanParams,
    compute_threat_pc,
)
from silly_kicks.xthreat import ExpectedThreat

_ARM_COLUMNS = ("a_threat_suppressed", "b_threat_suppressed")
_FRAME_KEYS = ("game_id", "period_id", "frame_id")

#: Dose battery constants, LOCKED (referenced-not-inlined, mirroring the probe constants). The
#: RESPONSE is the change in ``compute_threat_pc`` under a positional displacement of the defender D.
#: The realistic dose is 2 m, the saturating is 10 m = ``SATURATING_MULTIPLE`` (5x) in the SAME
#: direction (D chases the ball, abandoning its zone); the paired-vector controls use the realistic
#: vector.
REALISTIC_DISP_M: float = 2.0
SATURATING_DISP_M: float = 10.0  # 5x realistic (aligns with SATURATING_MULTIPLE)
N_PLACEBO: int = 3
PROBE_RNG_SEED: int = 0

#: The R single-outfielder placebo-band columns (one per replicate; parent idiom). :func:`pool_shards`
#: FLATTENS these for the placebo 95th percentile, distinct from the nearest-defender ``nd_abs``.
_PLACEBO_COLS = [f"placebo_abs_{k}" for k in range(N_PLACEBO)]

#: PRE-REGISTERED elite-defender ("Van Dijk") face-validity prior, LOCKED before the owner run so it
#: is a confirmatory pre-registration, not a post-hoc read. `name -> expected suppression sign`
#: (positive == the defender's positioning suppresses the attacking team's threat). Matched
#: case-insensitively as a substring of the injected `{player_id: name}` map. This locks the EXPECTED
#: prior only; the OBSERVED sign is what the run tests. NOT a ranking (team-confound; see the module
#: honest-limit).
ELITE_DEFENDER_PRIOR: dict[str, str] = {"Van Dijk": "positive", "Gvardiol": "positive", "Otamendi": "positive"}

#: The date the prior above was locked (stamped into the artifact as the pre-registration record).
ELITE_DEFENDER_PRIOR_LOCKED = "2026-09-09 (before the owner run)"

#: The columns the per-match work function emits + the shard-generation schema token they are pinned
#: to. These MUST move together (ADR-052 / the 4.77.1 stale-shard rule): the for_each fingerprint
#: digests token_inputs only, so renaming a column while leaving the token reuses stale shards.
#:
#: The shard carries TWO row kinds under a ``row_kind`` discriminator (bumped to -3 for the new
#: schema): ``"battery"`` rows -- one per SCORED Arm-A frame (the dose columns feeding the pooled
#: Layer-0/1 verdicts) -- and ``"sample"`` rows -- one per LIBRARY-samples defender (the per-defender
#: aggregate feeding ``named_defender_signs`` + the elite prior). Emitting the library samples verbatim
#: (rather than re-deriving from the battery rows) means a defender with Arm-B contributions but NO
#: scored Arm-A frame is NO LONGER dropped (IMPL-04): the battery rows cover only scored frames, but the
#: sample rows cover every domain defender ``compute_territorial_defense`` returns.
_SHARD_SCHEMA_VERSION = "td-construct-validity-3"
_EMITTED_SHARD_COLUMNS = [
    "row_kind",  # "battery" (per scored frame) | "sample" (per library-samples defender)
    "game_id",
    "period_id",  # battery only (NaN on sample rows)
    "frame_id",  # battery only
    "player_id",
    "player_name",
    # --- battery (per-scored-frame) dose columns; NaN on sample rows ---
    "a_delta",
    "realistic_abs",
    "saturating_abs",
    "realistic_signed",
    "nd_abs",
    *_PLACEBO_COLS,
    # --- sample (per-defender) aggregate columns; NaN on battery rows ---
    "a_threat_suppressed",
    "a_frames_scored",
    "b_threat_suppressed",
    "b_frames_scored",
    "b_attribution_slippage",
    "td_source",
]


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056)."""
    from dataclasses import asdict

    return declare_inputs(
        driver="validate_territorial_defense",
        params={
            "territorial_defense": asdict(TerritorialDefenseParams()),
            "min_domain_frames": MIN_DOMAIN_FRAMES,
            "saturating_multiple": SATURATING_MULTIPLE,
            "td_probe_ratio": TD_PROBE_RATIO,
            "realistic_disp_m": REALISTIC_DISP_M,
            "saturating_disp_m": SATURATING_DISP_M,
            "n_placebo": N_PLACEBO,
            "elite_defender_prior": ELITE_DEFENDER_PRIOR,
            "elite_defender_prior_locked": ELITE_DEFENDER_PRIOR_LOCKED,
        },
        extractors=["silly_kicks.territorial_defense._arms", "silly_kicks.territorial_defense._engine"],
        models=["silly_kicks.territorial_defense._probe"],
    )


def _fit_xt(actions: pd.DataFrame) -> ExpectedThreat:
    """Fit an exogenous xT on the match's own actions (the injected value model)."""
    return ExpectedThreat().fit(actions)


def _threat(frame, *, attacking_team_id, xt, goal_map, params) -> float:
    """``compute_threat_pc`` on one frame (spearman, keeper control-agent term); NaN on an unresolvable
    attacked-goal end (ADR-055 -- caught at the edge, never a crash)."""
    try:
        return compute_threat_pc(
            frame,
            attacking_team_id=attacking_team_id,
            xt=xt,
            goal_map=goal_map,
            method=params.pitch_control_method,
            params=SpearmanParams(lambda_gk=params.lambda_gk),
        )
    except GoalEndUnresolvedError:
        return float("nan")


def _response(frame, factual: float, *, attacking_team_id, xt, goal_map, params) -> float:
    """Signed change in threat under a frame perturbation: ``threat(perturbed) - factual``."""
    return _threat(frame, attacking_team_id=attacking_team_id, xt=xt, goal_map=goal_map, params=params) - factual


def _control_abs(controls: dict, key: str, factual: float, resp_kw: dict) -> float:
    """``|threat(control) - factual|`` for one paired-vector control; NaN when the control is absent
    (D was the only defending outfielder, so :func:`paired_vector_controls` returned no such frame)."""
    return abs(_response(controls[key], factual, **resp_kw)) if key in controls else float("nan")


def _dose_unit_vector(fr, *, defender_pos, game_id, period_id, attacking_team_id, goal_map) -> tuple[float, float]:
    """Unit dose direction for D: toward the ball (D chases the ball, abandoning its zone -> threat
    rises). FALLBACK when D is within 1e-6 m of the ball (or no ball row is present): toward the
    attacked goal end (``goal_map.attacked_goal`` x, y=34.0). A degenerate zero-vector defaults to
    ``(1.0, 0.0)`` so the dose is never NaN."""
    dx0 = float(fr.iloc[defender_pos]["x"])
    dy0 = float(fr.iloc[defender_pos]["y"])
    is_ball = fr["is_ball"].astype("boolean").fillna(False).to_numpy(dtype=bool)
    target: tuple[float, float] | None = None
    if is_ball.any():
        bpos = int(np.flatnonzero(is_ball)[0])
        bx, by = float(fr.iloc[bpos]["x"]), float(fr.iloc[bpos]["y"])
        if float(np.hypot(bx - dx0, by - dy0)) >= 1e-6:  # D is not on the ball
            target = (bx, by)
    if target is None:  # fallback: toward the attacked goal end
        gx = goal_map.attacked_goal(game_id, period_id, attacking_team_id, allow_guess=True)
        target = ((float(gx) if gx is not None else 105.0), 34.0)
    vx, vy = target[0] - dx0, target[1] - dy0
    norm = float(np.hypot(vx, vy))
    if norm < 1e-9:
        return 1.0, 0.0
    return vx / norm, vy / norm


def _measure_match(item, *, xt=None) -> pd.DataFrame:
    """One SB360 match -> ONE per-scored-Arm-A-frame probe-battery shard (a tidy shard).

    Applies the actor bridge (real ids onto the anonymous freeze-frame actor rows), fits an xT
    per-match (or uses the injected ``xt``), runs ``compute_territorial_defense`` (the per-defender
    samples), then runs the spec-S9 dose battery on every SCORED Arm-A frame: the factual threat, the
    realistic/saturating doses on D, and the nearest/placebo paired-vector controls, all as changes in
    ``compute_threat_pc``. Returns EMPTY (columns present) when the match yields no scored Arm-A frame OR
    is too sparse to fit a usable xT -- an empty shard means "ran, produced nothing", distinct from an
    absent one (ADR-052), never a crash that loses the pass. ``xt`` is injectable so a corpus-fit /
    bundled model can be threaded in.

    The shard carries BOTH row kinds (``row_kind``): one ``"battery"`` row per scored Arm-A frame (the
    dose columns) AND one ``"sample"`` row per LIBRARY-samples defender (the per-defender aggregate,
    emitted verbatim from ``compute_territorial_defense``). The sample rows are what
    ``named_defender_signs`` reads, so a defender with Arm-B contributions but NO scored Arm-A frame is
    covered too (IMPL-04) -- the battery rows are Arm-A-scored-only, the sample rows are every domain
    defender.
    """
    from sklearn.exceptions import NotFittedError

    _provider, _game_id, actions, frames, _home, visible_area = item
    if frames is None or len(frames) == 0:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    frames = apply_actor_identities_to_frames(frames, actions)
    params = TerritorialDefenseParams()
    try:
        model = xt if xt is not None else _fit_xt(actions)
        # Thread the SB360 visible_area so the SPEC-02 FOV local-completeness gate is LIVE in the
        # owner run (else FOV-cropped frames score with an upward-biased delta).
        samples, _report = compute_territorial_defense(actions, frames, xt=model, visible_area=visible_area)
    except NotFittedError:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    if samples.empty:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)

    names: dict = {}
    if "player_name" in actions.columns:
        names = dict(zip(actions["player_id"], actions["player_name"], strict=False))

    # The dose battery over every SCORED Arm-A frame (spec S9). SB360 freeze-frames are per-action-LTR,
    # so the goal is resolved PER FRAME from the action's acting team (action_ltr_goal_map) -- a per-match
    # resolve_defended_goals is bimodal on per-action frames and scores nothing (spec S3). frames grouped
    # ONCE (ADR-068). The RNG is seeded so the paired-vector controls are reproducible across a resume.
    def goal_map_for(game_id, period_id, acting_team_id, opponent_team_id):
        return action_ltr_goal_map(game_id, period_id, acting_team_id=acting_team_id, opponent_team_id=opponent_team_id)

    domain = select_arm_a_domain(actions, params=params)
    classified = classify_arm_a_domain(
        domain, frames, params=params, visible_area=visible_area, goal_map_for=goal_map_for
    )
    groups = group_rows(frames, _FRAME_KEYS)
    rng = np.random.default_rng(PROBE_RNG_SEED)

    rows: list[dict] = []
    for cand in classified.itertuples():
        if cand.td_source != "scored":
            continue
        fr = groups.get(cand.game_id, cand.period_id, cand.frame_id)
        if len(fr) == 0:
            continue
        dpos = int(cast(int, cand.defender_pos))  # the int positional index (itertuples types it Scalar)
        # Per-frame goal map: the acting team of this frame's action is D's team (a defensive action).
        gm = goal_map_for(cand.game_id, cand.period_id, cand.defending_team_id, cand.attacking_team_id)
        resp_kw = {"attacking_team_id": cand.attacking_team_id, "xt": model, "goal_map": gm, "params": params}

        f = _threat(fr, **resp_kw)
        ux, uy = _dose_unit_vector(
            fr,
            defender_pos=dpos,
            game_id=cand.game_id,
            period_id=cand.period_id,
            attacking_team_id=cand.attacking_team_id,
            goal_map=gm,
        )
        rdx, rdy = ux * REALISTIC_DISP_M, uy * REALISTIC_DISP_M
        sdx, sdy = ux * SATURATING_DISP_M, uy * SATURATING_DISP_M

        dosed_realistic = impose_defender_dose(fr, defender_pos=dpos, dx=rdx, dy=rdy)
        dosed_saturating = impose_defender_dose(fr, defender_pos=dpos, dx=sdx, dy=sdy)
        realistic_signed = _response(dosed_realistic, f, **resp_kw)
        saturating_signed = _response(dosed_saturating, f, **resp_kw)

        # Paired-vector controls: ONE other defending outfielder displaced by the SAME realistic vector
        # (the nearest to D + R single-player placebos). Empty {} when D is the only defender -> NaN.
        controls = paired_vector_controls(
            fr,
            defender_pos=dpos,
            defending_team_id=cand.defending_team_id,
            dx=rdx,
            dy=rdy,
            r=N_PLACEBO,
            rng=rng,
        )

        # The identity-exact removal delta (a_delta); ADR-055 honest-NaN on an unresolvable end.
        try:
            a_delta = arm_a_threat_suppressed(fr, remove_player_row(fr, player_pos=dpos), **resp_kw)
        except GoalEndUnresolvedError:
            a_delta = float("nan")

        row = {
            "row_kind": "battery",
            "game_id": cand.game_id,
            "period_id": cand.period_id,
            "frame_id": cand.frame_id,
            "player_id": cand.defender_id,
            "player_name": names.get(cand.defender_id),
            "a_delta": float(a_delta),
            "realistic_abs": abs(realistic_signed),
            "saturating_abs": abs(saturating_signed),
            "realistic_signed": realistic_signed,
            "nd_abs": _control_abs(controls, "nearest", f, resp_kw),
        }
        for k in range(N_PLACEBO):
            row[_PLACEBO_COLS[k]] = _control_abs(controls, f"placebo_{k}", f, resp_kw)
        rows.append(row)

    # One "sample" row per LIBRARY-samples defender (EVERY domain defender, not just Arm-A-scored) --
    # emitted verbatim so named_defender_signs never drops an Arm-B-only defender (IMPL-04).
    for s in samples.itertuples():
        rows.append(
            {
                "row_kind": "sample",
                "game_id": s.game_id,
                "player_id": s.player_id,
                "player_name": names.get(s.player_id),
                "a_threat_suppressed": s.a_threat_suppressed,
                "a_frames_scored": s.a_frames_scored,
                "b_threat_suppressed": s.b_threat_suppressed,
                "b_frames_scored": s.b_frames_scored,
                "b_attribution_slippage": s.b_attribution_slippage,
                "td_source": s.td_source,
            }
        )

    if not rows:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    return pd.DataFrame(rows).reindex(columns=_EMITTED_SHARD_COLUMNS)


def _nanmed(a) -> float:
    """Median skipping NaN; NaN (no RuntimeWarning) on an empty or all-NaN array."""
    a = np.asarray(a, dtype=float)
    return float(np.nanmedian(a)) if a.size and bool(np.isfinite(a).any()) else float("nan")


def pool_shards(shards: list[pd.DataFrame]) -> dict:
    """Concatenate the per-frame battery shards and pool the dose magnitudes -- the POOLED-corpus
    statistics the Layer-0/1 verdicts consume (NEVER per shard; mirrors ``build_tf19``).

    ``nd_abs`` (the nearest-defender control) supplies ``nd_med``; the R single-outfielder placebo
    columns (:data:`_PLACEBO_COLS`) are FLATTENED to supply the placebo 95th percentile -- so ``nd_med``
    and ``placebo_p95`` are DISTINCT single-player quantities and the Layer-1 ``max(nd_med, placebo_p95)``
    is meaningful. ``defender_med`` (the dosed-defender responsiveness median) and ``real_med``/``sat_med``
    are RECORDED so the verdict is auditable. ``n_domain`` = count of FINITE ``realistic_abs``.
    """
    if not shards:
        return {}
    df = pd.concat(shards, ignore_index=True)
    # Pool only the per-scored-frame "battery" rows -- the "sample" rows carry NaN dose columns (they
    # are the per-defender aggregate), so nan-aware pooling already excludes them; the explicit filter
    # states the intent and is robust to a future non-NaN sample default.
    if "row_kind" in df.columns:
        df = df[df["row_kind"] == "battery"]
    real = df["realistic_abs"].to_numpy(dtype=float)
    sat = df["saturating_abs"].to_numpy(dtype=float)
    nd = df["nd_abs"].to_numpy(dtype=float)
    placebo_cols = [c for c in _PLACEBO_COLS if c in df.columns]
    placebo = df[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
    return {
        "realistic_abs": real,
        "saturating_abs": sat,
        "real_med": _nanmed(real),
        "sat_med": _nanmed(sat),
        "defender_med": _nanmed(real),
        "nd_med": _nanmed(nd),
        "placebo_p95": float(np.nanpercentile(placebo, 95)) if np.isfinite(placebo).any() else float("nan"),
        "n_domain": int(np.isfinite(real).sum()),
        "n_placebo": int(np.isfinite(placebo).sum()),
    }


def reduce_layer_verdicts(pooled: dict) -> dict:
    """Pooled-corpus Layer-0/1 verdicts, WITH the medians they rest on (auditability).

    ``pooled`` carries the already-pooled ``realistic_abs``/``saturating_abs`` arrays + the
    ``real_med``/``sat_med``/``defender_med``/``nd_med``/``placebo_p95`` scalars + ``n_domain`` (as
    :func:`pool_shards` emits). Layer 1 uses ``defender_med=`` (NOT ``gk_med``). Empty ``pooled`` -> {}.
    """
    if not pooled:
        return {}
    return {
        "layer0": layer0_instrument_verdict(
            realistic_abs=pooled["realistic_abs"],
            saturating_abs=pooled["saturating_abs"],
            placebo_p95=pooled["placebo_p95"],
            n_domain=pooled["n_domain"],
        ),
        "layer1": layer1_responsiveness_verdict(
            defender_med=pooled["defender_med"],
            nd_med=pooled["nd_med"],
            placebo_p95=pooled["placebo_p95"],
            n_domain=pooled["n_domain"],
        ),
        "n_domain": pooled["n_domain"],
        "medians": {
            "real_med": pooled.get("real_med"),
            "sat_med": pooled.get("sat_med"),
            "defender_med": pooled.get("defender_med"),
            "nd_med": pooled.get("nd_med"),
            "placebo_p95": pooled.get("placebo_p95"),
        },
        "n_placebo": pooled.get("n_placebo"),
    }


def _per_defender_from_shards(combined: pd.DataFrame) -> pd.DataFrame:
    """The per-``(game_id, player_id)`` defender table -- the shard's ``"sample"`` rows verbatim.

    These ARE ``compute_territorial_defense``'s per-defender samples (emitted into the shard), so this
    covers EVERY domain defender -- including one with Arm-B contributions but no scored Arm-A frame,
    which the old battery-re-derivation dropped (IMPL-04). One "sample" row per defender per match, so
    no aggregation is needed; a corpus-combined shard already holds one row per ``(game, player)``.
    """
    cols = [
        "game_id",
        "player_id",
        "player_name",
        "a_threat_suppressed",
        "a_frames_scored",
        "b_threat_suppressed",
        "b_frames_scored",
        "b_attribution_slippage",
        "td_source",
    ]
    if combined.empty or "row_kind" not in combined.columns:
        return pd.DataFrame(columns=cols)
    return combined[combined["row_kind"] == "sample"][cols].reset_index(drop=True)


def _elite_prior_report(per_defender: pd.DataFrame) -> dict:
    """Face-validity: the observed suppression sign for each pre-registered elite defender.

    Reads ``player_name`` + ``a_threat_suppressed`` (works on either the per-defender aggregate or the
    raw per-frame shard). Reported WITH the collinearity caveat: on a single-tournament / national-team
    corpus this is NOT attribution -- an elite defender and a strong defensive side are collinear.
    """
    rows = []
    names = per_defender["player_name"].astype("string").fillna("")
    for elite, expected in ELITE_DEFENDER_PRIOR.items():
        mask = names.str.contains(elite, case=False, regex=False)
        sub = per_defender[mask.to_numpy()]
        vals = pd.to_numeric(sub["a_threat_suppressed"], errors="coerce").dropna()
        observed = "positive" if float(vals.sum()) > 0 else ("negative" if len(vals) else "no_data")
        rows.append(
            {
                "defender": elite,
                "expected": expected,
                "observed": observed,
                "meets_prior": observed == expected,
                "n_matches": len(vals),
                "mean_a_threat_suppressed": float(vals.mean()) if len(vals) else float("nan"),
            }
        )
    return {"prior_locked": ELITE_DEFENDER_PRIOR_LOCKED, "checks": rows}


def _cross_team_replication(per_defender: pd.DataFrame) -> dict:
    """Does the corpus contain ANY defender observed on >=2 distinct teams? (the identifying power for
    the defender-vs-team confound). WC2022 alone adds none (one player = one national team)."""
    # One consistent key schema in BOTH branches (a consumer keyed on one must not break on the other).
    n_multi = (
        int((per_defender.groupby("player_name")["game_id"].nunique() >= 2).sum()) if not per_defender.empty else 0
    )
    return {
        "n_defenders_multi_game": n_multi,
        "identifies_confound": False,
        "note": (
            "A per-team decomposition needs a team column + cross-team appearances; WC2022 is "
            "one-player-one-national-team, so it has ZERO identifying power for the defender-vs-team "
            "confound. Ranking is a future ADR-009 gated on a crossed defender+team ICC over a "
            "multi-club transfer corpus."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--token", default=None, help="pining token (else resolved from the environment)")
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON ["3857276", ...] pinning WHICH matches this process handles (parallel split).',
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # Clean-tree guard FIRST, before any corpus work (ADR-037). --list-matches writes no artifact.
    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    from scripts._driver import for_each
    from scripts._loader_pining import load_statsbomb_matches

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None

    if args.list_matches:
        ids = [mid for _p, mid, *_ in load_statsbomb_matches(token=args.token, cache_dir=args.cache_dir)]
        print(json.dumps(ids, indent=2))
        return

    dest = Path(args.out)

    def _matches():
        yield from load_statsbomb_matches(
            match_ids=match_ids, token=args.token, max_matches=args.max_matches, cache_dir=args.cache_dir
        )

    # Guard against a stale-shard reuse (4.77.1): the emitted columns MUST match the declaration.
    res = for_each(
        _matches(),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_measure_match,
        shard_root=dest / "shards",
        token_inputs={
            "metric": "territorial_defense",
            "schema": _SHARD_SCHEMA_VERSION,
            "xt": "per_match_fit",
            "pitch_control_method": "spearman",
        },
        label="match",
    )

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    shards = [pd.read_parquet(s) for s in shard_files]
    combined = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)

    # Per-defender table for the face-validity prior + the "A" deliverable (named_defender_signs) --
    # the shard's "sample" rows (every domain defender, IMPL-04), not a battery re-derivation.
    per_defender = _per_defender_from_shards(combined)
    a_vals = pd.to_numeric(per_defender.get("a_threat_suppressed", pd.Series(dtype=float)), errors="coerce").dropna()
    b_vals = pd.to_numeric(per_defender.get("b_threat_suppressed", pd.Series(dtype=float)), errors="coerce").dropna()
    slip_vals = pd.to_numeric(
        per_defender.get("b_attribution_slippage", pd.Series(dtype=float)), errors="coerce"
    ).dropna()

    # Pooled-corpus Layer-0/1 verdicts (the reduce over ALL shards, never per shard).
    verdicts = reduce_layer_verdicts(pool_shards(shards))

    out = {
        "n_defenders": len(per_defender),
        "n_frames_scored": len(combined),
        "n_arm_a_scored": len(a_vals),
        "n_arm_b_scored": len(b_vals),
        "expected_direction": {c: expected_direction_for_arm(c) for c in _ARM_COLUMNS},
        "verdicts": verdicts,
        "arm_b_slippage": {
            "mean_attribution_slippage": float(slip_vals.mean()) if len(slip_vals) else float("nan"),
            "n": len(slip_vals),
        },
        "elite_defender_prior": _elite_prior_report(per_defender),
        "cross_team_replication": _cross_team_replication(per_defender),
        "honest_limit": (
            "Validated as an INSTRUMENT, NOT player-attributable: the marginal-removal delta is "
            "team-conditioned by construction; on a single-tournament / national-team corpus the "
            "defender-vs-team confound is unidentifiable, so per-defender numbers are NOT a ranking."
        ),
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "input_contract": input_contract(),
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if not per_defender.empty:
        per_defender.to_parquet(dest / "named_defender_signs.parquet", index=False)
    print(json.dumps({k: v for k, v in out.items() if k != "input_contract"}, indent=2, default=str))


if __name__ == "__main__":
    main()
