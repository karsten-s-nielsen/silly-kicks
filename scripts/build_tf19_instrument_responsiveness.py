"""Maintainer driver: TF-19 A+2 physics-arm instrument-validity + responsiveness (reported-not-gated).

Owner-run corpus pass. Per SCORED counterfactual frame it measures, for the Delta-DAS arm, the dose
magnitudes the Layer-0 / Layer-1 verdicts pool over the whole corpus:

* Layer 0 (instrument validity): ``realistic_abs`` (the shipped ghost-dose |Delta-DAS|, Regime O) vs
  ``saturating_abs`` (keeper forced onto the defended goal line) -- a live instrument moves the
  saturating dose >> the realistic one, or clears the placebo band.
* Layer 1 (responsiveness): ``gk_abs`` (the imposed 2 m ladder dose, Regime I) vs ``nd_abs`` (the
  paired-vector control -- the nearest defending-team outfielder displaced by the SAME vector).

Each scored frame is attributed to the REAL defending keeper via ``tracking.resolve_keeper_identities``
(ADR-078 single-source): the native path returns the frame keeper id (velocity-bearing providers), the
roster path resolves SB360's anonymous keeper. The map is threaded DRIVER-side (ADR-037: gkdv reaches
``tracking`` only through ``_das_port``; scripts may import ``tracking`` freely).

TWO library-forced scope facts, both honest (never a fabricated 0/NaN passed as a measurement):

* **The threat arm (``delta_threat_suppression``) is ``arm_unscoreable`` here.** It needs a fitted
  ``ExpectedThreat`` and the package ships no loader (``ExpectedThreat`` exposes only
  fit/interpolator/rate; ``FrozenXt`` wraps an in-memory model). Fitting one in-process is a leakage
  decision that belongs in its own registered cycle -- exactly the constraint
  ``scripts/build_gkdv_arm_values.py`` records. So the reduce reports it ``arm_unscoreable``, a
  first-class verdict, not a skip.
* **Delta-DAS is velocity-constitutive**, so it is NaN on velocity-less SB360 freeze frames regardless of
  keeper identity (ADR-063). The runnable pass is the velocity-bearing corpus (Gradient Sports).

The corpus map is ``for_each`` (ADR-052: per-match shards, resumable, conserving); the pooled Layer-0/1
verdicts, the per-arm ``gate_eligible`` census (spec S6.1), the S6.2 named-keeper sign table and the
Layer-4 ``behavioural_anchoring_verdict`` are computed in a REDUCE over all shards, never per shard.

Frame-level correctness is validated by the OWNER RUN; the unit tests pin the pooled reduce, the
verdict discrimination and the census (``tests/scripts/test_tf19_instrument_responsiveness_driver.py``).

The Layer-0 saturating dose is ``saturating_goalline`` (keeper on the defended goal line -- maximal
displacement); ``impose_defending_keeper_dose`` also supports ``saturating_x30`` (goal-relative x=30 m)
as an alternative saturating landmark, not used by this pass.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/build_tf19_instrument_responsiveness.py --out <DIR> [--providers gradientsports] \
      [--max-per-provider N] [--tracking-limit N] [--match-ids-json FILE | --list-matches]

For a PARALLEL run, split ``--match-ids-json`` across N workers sharing one ``--out`` (each writes
shards + a per-worker manifest, NO verdict), then run ONCE more UNPARTITIONED over the same ``--out``
to produce the authoritative ``metrics.json`` + ``named_keeper_signs.parquet`` (it resumes existing
shards and reduces over ALL). ``--out docs/research/tf19_instrument_responsiveness/`` puts the cited
``metrics.json`` where the ADR-056 staleness detector reads it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs
from silly_kicks.gkdv._probe import (
    MIN_DOMAIN_FRAMES,
    PHYSICS_ARM_PROBE_RATIO,
    REALISTIC_MIN_DISP_M,
    REGIME_I_LADDER_M,
    SATURATING_MULTIPLE,
    SATURATING_X30_GR,
    R,
    layer0_instrument_verdict,
    layer1_responsiveness_verdict,
)
from silly_kicks.gkdv._validate import expected_direction_for_arm

_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

#: The threat arm's OUTPUT column (reported arm_unscoreable here -- see the module docstring).
_THREAT_ARM = "delta_threat_suppression"
_DAS_ARM = "delta_das"

#: The R single-outfielder placebo-band columns (one per replicate; parent-idiom controls). `nd_abs`
#: is the nearest-defender control (distinct single-player quantity). :func:`pool_shards` reads
#: `nd_abs` for `nd_med` and FLATTENS `_PLACEBO_COLS` for the placebo p95.
_PLACEBO_COLS = [f"placebo_abs_{k}" for k in range(R)]

#: The columns a per-frame shard carries. The dose ``*_abs`` feed :func:`pool_shards`; ``realistic_signed``
#: feeds the S6.2 sign table + S6.1 census -- the Regime-O realistic dose SIGNED, computed over the
#: ``|displacement| >= REALISTIC_MIN_DISP_M`` (2 m) subset (spec S4.1): the floor SHARPENS, does not
#: invert, the deterrent sign, and is NOT the no-floor ``build_gkdv_arm_values`` population; ``keeper_gr_depth``
#: feeds the Layer-4 anchoring verdict; ``keeper_key`` is the resolved defending-keeper id (NA where the
#: keeper is unresolved -- dropped-AND-counted in the reduce, never fabricated).
_SHARD_COLUMNS = [
    "keeper_key",
    *_FRAME_KEYS,
    "arm",
    "realistic_abs",
    "saturating_abs",
    "gk_abs",
    "nd_abs",
    *_PLACEBO_COLS,
    "realistic_signed",
    "keeper_gr_depth",
]

#: PRE-REGISTERED named-keeper face-validity prior (spec S4.4), LOCKED 2026-08-29 -- committed BEFORE the
#: owner corpus run so the Alisson/Neuer eye-test is a confirmatory pre-registration, not a post-hoc read.
#: `name -> expected Delta-DAS sign` (negative == deterrent). These are the well-regarded sweeper/elite keepers
#: S4.4 names; a name is matched case-insensitively as a substring of the injected `{player_id: name}` map
#: (so "Neuer" matches "Manuel Neuer"). This locks the EXPECTED prior only -- the OBSERVED sign is what the
#: run tests.
NAMED_KEEPER_PRIOR: dict[str, str] = {"Alisson": "negative", "Neuer": "negative"}

#: Named keepers S4.4 states honestly as EXCLUDED from the locked pass/fail (insufficient / descriptive-only),
#: recorded so the exclusion is explicit rather than silent.
NAMED_KEEPER_CAVEATED: dict[str, str] = {"Ter Stegen": "0_min", "Onana": "descriptive_only"}

#: The date the named prior above was locked (stamped into the artifact as the pre-registration record).
NAMED_KEEPER_PRIOR_LOCKED = "2026-08-29 (before the owner run)"


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056).

    ``GkdvParams`` carries the arm configuration; the probe constants fix the doses + verdict
    thresholds; the ghost model is declared by name (its own chirality/feature-contract stamps pin it,
    ADR-050).
    """
    from dataclasses import asdict

    from silly_kicks.gkdv import GkdvParams

    return declare_inputs(
        driver="build_tf19_instrument_responsiveness",
        # Declare every PRE-REGISTERED threshold the verdicts are judged against, so a change to any of
        # them moves the digest and the staleness detector flags the artifact (ADR-056).
        params={
            "gkdv": asdict(GkdvParams()),
            "min_domain_frames": MIN_DOMAIN_FRAMES,
            "regime_i_ladder_m": REGIME_I_LADDER_M,
            "saturating_multiple": SATURATING_MULTIPLE,
            "physics_arm_probe_ratio": PHYSICS_ARM_PROBE_RATIO,
            "r": R,
            "realistic_min_disp_m": REALISTIC_MIN_DISP_M,
            "saturating_x30_gr": SATURATING_X30_GR,
        },
        extractors=("silly_kicks.gkdv._probe", "silly_kicks.gkdv._arms", "silly_kicks.gkdv._engine"),
        models=("silly_kicks.tracking._ghost_gk.GhostGkModel",),
    )


def _nanmed(a: np.ndarray) -> float:
    """Median skipping NaN; NaN on an empty or all-NaN array (no RuntimeWarning). Applied symmetrically
    to gk_med / nd_med (an earlier gk_med guarded only `len(g)`, so an all-NaN gk leg warned + returned
    NaN inconsistently)."""
    return float(np.nanmedian(a)) if a.size and bool(np.isfinite(a).any()) else float("nan")


def pool_shards(shards: list[pd.DataFrame]) -> dict:
    """Concatenate per-frame shards and pool per arm -- the POOLED-corpus statistics the Layer-0/1
    verdicts consume (never per shard).

    ``nd_abs`` (the nearest-defender control) supplies ``nd_med``; the R single-outfielder placebo
    columns (:data:`_PLACEBO_COLS`) are FLATTENED to supply the placebo 95th percentile -- so ``nd_med``
    and ``placebo_p95`` are DISTINCT single-player quantities and the Layer-1 ``max(nd_med, placebo_p95)``
    is meaningful (a combined multi-player control would collapse them onto one array). The pooled
    medians are RECORDED (``real_med``/``sat_med``/``gk_med``/``nd_med``) so the verdict is auditable --
    e.g. a ``real_med == 0`` (which the Layer-0 guard treats specially) is visible in the artifact.

    Returns ``{arm: {realistic_abs, saturating_abs, real_med, sat_med, placebo_p95, gk_med, nd_med,
    n_domain, n_placebo}}``.
    """
    if not shards:
        return {}
    df = pd.concat(shards, ignore_index=True)
    out: dict = {}
    for arm, g in df.groupby("arm", dropna=False):
        real = g["realistic_abs"].to_numpy(dtype=float)
        sat = g["saturating_abs"].to_numpy(dtype=float)
        gk = g["gk_abs"].to_numpy(dtype=float)
        nd = g["nd_abs"].to_numpy(dtype=float)
        placebo_cols = [c for c in _PLACEBO_COLS if c in g.columns]
        placebo = g[placebo_cols].to_numpy(dtype=float).ravel() if placebo_cols else np.array([np.nan])
        out[str(arm)] = {
            "realistic_abs": real,
            "saturating_abs": sat,
            "real_med": _nanmed(real),
            "sat_med": _nanmed(sat),
            "placebo_p95": float(np.nanpercentile(placebo, 95)) if np.isfinite(placebo).any() else float("nan"),
            "gk_med": _nanmed(gk),
            "nd_med": _nanmed(nd),
            "n_domain": len(g),
            "n_placebo": int(np.isfinite(placebo).sum()),
        }
    return out


def reduce_layer_verdicts(per_arm: dict) -> dict:
    """Pooled-corpus Layer-0/1 verdicts per arm, WITH the medians they rest on (auditability).

    ``per_arm[arm]`` carries the already-pooled ``realistic_abs``/``saturating_abs`` arrays +
    ``real_med``/``sat_med``/``placebo_p95``/``gk_med``/``nd_med`` scalars + ``n_domain`` (as
    :func:`pool_shards` emits). Recording the medians is what makes a Layer-0 verdict auditable: a dead
    instrument that (correctly) reads ``instrument_void`` because ``real_med == 0`` is otherwise invisible.
    """
    out: dict = {}
    for arm, s in per_arm.items():
        out[arm] = {
            "layer0": layer0_instrument_verdict(
                realistic_abs=s["realistic_abs"],
                saturating_abs=s["saturating_abs"],
                placebo_p95=s["placebo_p95"],
                n_domain=s["n_domain"],
            ),
            "layer1": layer1_responsiveness_verdict(
                gk_med=s["gk_med"],
                nd_med=s["nd_med"],
                placebo_p95=s["placebo_p95"],
                n_domain=s["n_domain"],
            ),
            "n_domain": s["n_domain"],
            "medians": {
                "real_med": s.get("real_med"),
                "sat_med": s.get("sat_med"),
                "gk_med": s["gk_med"],
                "nd_med": s["nd_med"],
                "placebo_p95": s["placebo_p95"],
            },
            "n_placebo": s.get("n_placebo"),
        }
    return out


def _named_keeper_signs(combined: pd.DataFrame, *, min_nonzero: int, min_games: int) -> tuple[dict, pd.DataFrame]:
    """S6.1 gate_eligible census + S6.2 named-keeper sign TABLE + Layer-4 anchoring, over the Delta-DAS arm.

    Returns ``(summary, per_keeper)``. ``per_keeper`` is THE "A" DELIVERABLE -- one row per resolved
    keeper (``player_id``, counts, ``mean``, ``observed_sign``, ``sign_matches_expected``,
    ``gate_eligible``) so the owner can do the Alisson/Neuer face-validity read (join ``player_id`` ->
    name via the provider roster). The census aggregates the SIGNED shipped metric (``realistic_signed``,
    Regime O); ``min_nonzero`` binds because Delta-DAS is zero-dominated (spec E2). Anchoring runs on the
    GATE-ELIGIBLE surface and asks whether keepers actually VARY in goal-relative depth.
    """
    from silly_kicks.gkdv import aggregate_by_keeper, behavioural_anchoring_verdict

    das = combined[combined["arm"] == _DAS_ARM]
    # DROPPED-AND-COUNTED (S4.3): aggregate_by_keeper's `dropna=True` silently drops NA-keeper rows, so
    # count them HERE -- an unresolved defending keeper is a reported denominator, never lost.
    n_unresolved_keeper_frames = int(das["keeper_key"].isna().sum())

    census = aggregate_by_keeper(
        das.rename(columns={"keeper_key": "player_id", "realistic_signed": _DAS_ARM}),
        value_col=_DAS_ARM,
        min_nonzero=min_nonzero,
        min_games=min_games,
    )
    expected = expected_direction_for_arm(_DAS_ARM)

    # THE per-KEEPER sign table (the "A" deliverable). observed_sign + whether it matches the arm's
    # expected direction (negative == deterrent), applied PER keeper.
    per_keeper = census.copy()
    per_keeper.insert(1, "arm", _DAS_ARM)
    per_keeper["expected_direction"] = expected
    per_keeper["observed_sign"] = np.where(
        per_keeper["mean"] < 0, "negative", np.where(per_keeper["mean"] > 0, "positive", "zero")
    )
    per_keeper["sign_matches_expected"] = per_keeper["observed_sign"] == expected

    # Anchoring runs on the GATE-ELIGIBLE surface: `behavioural_anchoring_verdict` documents that the
    # S6.1 clustering floors "thin the surface, and they run first", but `aggregate_by_keeper` only
    # FLAGS `gate_eligible` (it does not drop keepers), so feeding the full census would rank
    # single-match / sub-`min_nonzero` keepers -- exactly the population the anchoring's LIMITATION warns of.
    eligible = per_keeper[per_keeper["gate_eligible"]]
    depth = das.groupby("keeper_key")["keeper_gr_depth"].mean().rename("keeper_gr_depth")
    anchoring = behavioural_anchoring_verdict(
        eligible.merge(depth, left_on="player_id", right_index=True, how="left"),
        value_col="mean",
        depth_col="keeper_gr_depth",
    )

    summary = {
        "arm": _DAS_ARM,
        "expected_direction": expected,
        "n_keepers": len(census),
        "n_gate_eligible": int(census["gate_eligible"].sum()),
        "n_eligible_sign_matches_expected": int(eligible["sign_matches_expected"].sum()),
        "n_unresolved_keeper_frames": n_unresolved_keeper_frames,
        "behavioural_anchoring": anchoring,
        "min_nonzero": min_nonzero,
        "min_games": min_games,
    }
    return summary, per_keeper


#: SB360 freeze-frames are velocity-less by design (ADR-063), so the velocity-constitutive Delta-DAS arm
#: cannot score them; the threat arm needs a fitted ExpectedThreat the package cannot load.
_VELOCITY_LESS_PROVIDERS = frozenset({"statsbomb"})


def _provider_support_matrix(providers: list[str]) -> dict:
    """Which ``(provider, arm)`` combos THIS pass can score (S4.5) -- descriptive, from the
    library-forced constraints, not a per-run measurement. Delta-DAS needs velocity (NaN on velocity-less
    SB360); the threat arm needs a fitted ``ExpectedThreat`` (none loadable) -> ``arm_unscoreable`` everywhere.
    """
    return {
        p.strip(): {
            "delta_das": ("unscoreable_velocity_less" if p.strip() in _VELOCITY_LESS_PROVIDERS else "scoreable"),
            "delta_threat_suppression": "arm_unscoreable_no_xt",
        }
        for p in providers
    }


def _named_keeper_check(per_keeper: pd.DataFrame, names_map: dict) -> tuple[pd.DataFrame, dict]:
    """Join injected names onto the per-keeper table + check the LOCKED :data:`NAMED_KEEPER_PRIOR` (S4.4).

    Dependency-inverted: ``names_map`` is an injected ``{player_id: name}`` (the driver never parses a
    roster). Returns ``(per_keeper_with_keeper_name, check)`` where ``check[name]`` records, per
    pre-registered keeper, the matched keepers' observed signs and whether they meet the LOCKED
    expectation -- the confirmatory face-validity read. Only the observed side is measured; the
    expectation was locked before the run.
    """
    from silly_kicks.id_compat import canonical_id

    canon_names = {canonical_id(k): str(v) for k, v in names_map.items()}
    pk = per_keeper.copy()
    pk["keeper_name"] = [canon_names.get(canonical_id(pid)) for pid in pk["player_id"]]

    named = pk["keeper_name"].fillna("")
    check: dict[str, dict] = {}
    for name, expected in NAMED_KEEPER_PRIOR.items():
        matched = pk[named.str.contains(name, case=False, regex=False)]
        check[name] = {
            "expected_sign": expected,
            "n_matched_keepers": len(matched),
            "observed_signs": matched["observed_sign"].tolist(),
            "gate_eligible": [bool(x) for x in matched["gate_eligible"]],
            # confirmatory: TRUE only if the name resolved to >=1 keeper AND every match meets the prior.
            "meets_prior": bool(len(matched)) and bool((matched["observed_sign"] == expected).all()),
        }
    return pk, check


# ---------------------------------------------------------------------------------------------------
# Per-match measurement (owner-run; frame-level correctness validated by the owner pass, not a unit test)


def _keeper_for(keeper_map: dict, game, period, team):
    """Resolved defending keeper id for a ``(game, period, team)`` -- map keys are (canonical game,
    period AS-IS, canonical team) per ADR-078; NA when the team's keeper is unresolved."""
    from silly_kicks.id_compat import canonical_id

    ident = keeper_map.get((canonical_id(game), period, canonical_id(team)))
    return ident.gk_id if ident is not None else pd.NA


def _measure_match(item, *, rng_seed: int) -> tuple[pd.DataFrame, dict]:
    """One per-frame Delta-DAS dose-measurement shard for a match + the keeper-identity counts.

    ``load_matches`` yields ``(provider, match_id, actions, frames, home_team_id)`` -- actions first.
    The dose imposer passes ``model=None`` to :func:`build_ghost_frames`, which resolves the BUNDLED
    default ghost (the ``build_gkdv_arm_values`` convention); no explicit model load is needed. Returns
    ``(shard, keeper_counts)`` where ``keeper_counts`` carries the KeeperIdentityReport totals so the
    driver conserves the resolution (dropped-AND-counted, S4.3).
    """
    from silly_kicks.gkdv import delta_das_batch
    from silly_kicks.gkdv._probe import impose_defending_keeper_dose, paired_vector_controls
    from silly_kicks.tracking import (
        derive_team_in_possession,
        infer_ball_carrier,
        resolve_keeper_identities,
    )

    _provider, _match_id, actions, frames, home_team_id = item

    # Delta-DAS routes through accessible space, which REQUIRES team_in_possession; raw loader frames do
    # not carry it. The carrier is pinned ONCE (spec S4.1) and shared by the possession derivation.
    carrier = infer_ball_carrier(frames)
    frames = derive_team_in_possession(frames, carrier)

    # Resolve the REAL defending keeper per (game, period, team) ONCE (ADR-078 single-source). Native
    # path: the frame keeper id (velocity-bearing GS). Roster path (SB360) is the same seam with a
    # roster injected -- but Delta-DAS is NaN on velocity-less SB360, so this pass runs native. The
    # KeeperIdentityReport is CARRIED OUT (S4.3 dropped-and-counted): an unresolved keeper -> NA
    # keeper_key -> counted, never silently dropped.
    keeper_map, keeper_report = resolve_keeper_identities(actions, frames, identity="native")
    keeper_counts = {
        "n_keeper_teams": keeper_report.n_teams_in,
        "n_keeper_teams_resolved": keeper_report.n_resolved,
        "n_keeper_teams_unresolved": keeper_report.n_unresolved,
        "n_matches": 1,
    }

    # Impose each dose (the imposer runs build_ghost_frames internally with the bundled default ghost,
    # so the scored domain matches build_gkdv_arm_values').
    realistic_imp, realistic_tgt = impose_defending_keeper_dose(frames, home_team_id=home_team_id, dose="realistic")
    sat_imp, sat_tgt = impose_defending_keeper_dose(frames, home_team_id=home_team_id, dose="saturating_goalline")
    ladder_imp, ladder_tgt = impose_defending_keeper_dose(
        frames, home_team_id=home_team_id, dose="ladder", displacement=REGIME_I_LADDER_M
    )
    if not len(ladder_tgt):
        return pd.DataFrame(columns=_SHARD_COLUMNS), keeper_counts
    # Single-player controls (parent idiom): nearest defender ALONE + R single-outfielder placebos.
    controls = paired_vector_controls(frames, ladder_tgt, r=R, rng=np.random.default_rng(rng_seed))

    # Attacking team per full-domain scored frame = the in-possession team (build_ghost_frames's domain
    # filters on it). A Series keyed by (game, period, frame); the arm batch RAISES on a missing key.
    full_keys = ladder_tgt[_FRAME_KEYS].drop_duplicates()
    dom = frames.merge(full_keys, on=_FRAME_KEYS)
    atk_full = dom.drop_duplicates(_FRAME_KEYS).set_index(_FRAME_KEYS)["team_in_possession"]

    def _delta(imposed: pd.DataFrame, tgt: pd.DataFrame, *, signed: bool) -> pd.Series:
        keys = tgt[_FRAME_KEYS].drop_duplicates()
        actual_s = frames.merge(keys, on=_FRAME_KEYS)
        imposed_s = imposed.merge(keys, on=_FRAME_KEYS)
        atk = atk_full.reindex(pd.MultiIndex.from_frame(keys))
        d = delta_das_batch(actual_s, imposed_s, attacking_team_id_by_frame=atk)
        return d if signed else d.abs()

    realistic_signed = _delta(realistic_imp, realistic_tgt, signed=True).rename("realistic_signed")
    realistic_abs = realistic_signed.abs().rename("realistic_abs")
    saturating_abs = _delta(sat_imp, sat_tgt, signed=False).rename("saturating_abs")
    gk_abs = _delta(ladder_imp, ladder_tgt, signed=False).rename("gk_abs")
    nd_abs = _delta(controls["nearest"], ladder_tgt, signed=False).rename("nd_abs")
    placebo = [_delta(controls[f"placebo_{k}"], ladder_tgt, signed=False).rename(_PLACEBO_COLS[k]) for k in range(R)]

    # Per full-domain scored frame: the resolved keeper id + its goal-relative depth (distance from the
    # defended goal; a S6.4 depth proxy). `ladder_tgt` carries defending_team_id / actual_x / defended_goal_x.
    base = ladder_tgt[[*_FRAME_KEYS, "defending_team_id", "actual_x", "defended_goal_x"]].drop_duplicates(_FRAME_KEYS)
    # np.array(..., dtype=object) (not a bare list): keeps the positional assignment while giving a
    # concrete object-ndarray -- a `list[object | NAType]` misses the pandas `__setitem__` overloads.
    base["keeper_key"] = np.array(
        [
            _keeper_for(keeper_map, g, p, t)
            for g, p, t in zip(base["game_id"], base["period_id"], base["defending_team_id"], strict=True)
        ],
        dtype=object,
    )
    defended = base["defended_goal_x"].to_numpy(dtype=float)
    actual_x = base["actual_x"].to_numpy(dtype=float)
    base["keeper_gr_depth"] = np.where(defended == 0.0, actual_x, 105.0 - actual_x)
    base = base.set_index(_FRAME_KEYS)

    rows = base.join([realistic_abs, saturating_abs, gk_abs, nd_abs, *placebo, realistic_signed]).reset_index()
    rows["arm"] = _DAS_ARM
    return rows[_SHARD_COLUMNS], keeper_counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--min-nonzero", type=int, default=20, help="S6.1 gate_eligible floor (registered)")
    ap.add_argument("--min-games", type=int, default=2, help="S6.1 gate_eligible floor (registered)")
    ap.add_argument(
        "--keeper-names-json",
        default=None,
        help='optional JSON {"<player_id>": "<name>"} to resolve keeper names + run the S4.4 '
        "named-keeper face-validity check (the locked prior is recorded regardless).",
    )
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"gradientsports": ["10502", ...]} pinning WHICH matches this process handles (parallel split).',
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--list-matches", action="store_true", help="print available match ids as JSON and exit")
    args = ap.parse_args()

    from scripts._provenance import git_provenance, require_clean_tree

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    # Clean-tree guard FIRST, before any corpus work. --list-matches writes no artifact, so it is exempt.
    prov = (
        {"commit": "n/a", "dirty": False, "tree_state": "clean", "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.list_matches:
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    from scripts._driver import for_each
    from scripts._loader_pining import load_matches
    from scripts._partition import providers_for_slice, worker_tag

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)
    worker = worker_tag(args.match_ids_json)

    _last_keeper: dict = {}

    def _work(item):
        # A stable per-match seed keeps the paired-vector control REPRODUCIBLE across a resume without
        # a wall-clock/global RNG (which would break the ADR-052 skip-existing-shard determinism).
        seed = int.from_bytes(f"{item[0]}:{item[1]}".encode(), "little") % (2**32)
        shard, keeper_counts = _measure_match(item, rng_seed=seed)
        # for_each calls work(item) then counters(item, frame) for the SAME item in that order, so
        # stashing the keeper-resolution counts here (they are not in the shard) is race-free.
        _last_keeper.clear()
        _last_keeper.update(keeper_counts)
        return shard

    res = for_each(
        load_matches(
            providers=providers_for_slice(args.providers.split(","), match_ids),
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        ),
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        counters=lambda _item, _frame: dict(_last_keeper),  # keeper-identity totals, summed by for_each (S4.3)
        shard_root=dest / "shards",
        token_inputs={
            "ghost_model": "default",
            "pitch_control_method": "spearman",
            "arm": _DAS_ARM,
            "doses": ["realistic", "saturating_goalline", "ladder"],
            "regime_i_ladder_m": REGIME_I_LADDER_M,
            # --tracking-limit truncates the frames every downstream computation sees, so a capped
            # smoke run and a full run are DIFFERENT corpora and must never share a generation.
            "tracking_limit": args.tracking_limit,
        },
        tag=worker,
        label="match",
    )

    # A --match-ids-json worker sees a RACE-DEPENDENT PARTIAL shard set (other workers share --out), so
    # it must NOT emit a corpus verdict -- that would present a partial reduce as the whole. It writes
    # shards (already done by for_each) + a per-worker manifest ONLY; the authoritative verdict comes
    # from a final UNPARTITIONED pass (which resumes, skips existing shards, and reduces over ALL of
    # them). Mirrors build_gkdv_arm_values' per-worker-manifest + final-aggregate split.
    if args.match_ids_json is not None:
        worker_manifest = {
            **res.manifest(),
            "run_commit": prov["commit"],
            "run_tree_dirty": prov["dirty"],
            "run_tree_state": prov.get("tree_state"),
            "partition": worker,
            "note": "partition worker: shards written; run an UNPARTITIONED pass for the authoritative verdicts.",
        }
        (dest / f"manifest_{worker}.json").write_text(
            json.dumps(worker_manifest, indent=2, default=str), encoding="utf-8"
        )
        print(json.dumps(worker_manifest, indent=2, default=str))
        return

    # AUTHORITATIVE reduce over ALL shards -- pooled Layer-0/1 verdicts are corpus statistics, never
    # per shard, so they are computed only on the unpartitioned pass (which sees the full shard set).
    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    shards = [pd.read_parquet(s) for s in shard_files]
    combined = pd.concat(shards, ignore_index=True) if shards else pd.DataFrame(columns=_SHARD_COLUMNS)

    verdicts = reduce_layer_verdicts(pool_shards(shards))
    # The threat arm is arm_unscoreable here (no loadable ExpectedThreat) -- a first-class verdict,
    # reported rather than silently omitted.
    verdicts[_THREAT_ARM] = {
        "layer0": "arm_unscoreable",
        "layer1": "arm_unscoreable",
        "n_domain": 0,
        "reason": "needs a fitted ExpectedThreat; the package ships no loader (see build_gkdv_arm_values)",
    }
    named, per_keeper = (
        _named_keeper_signs(combined, min_nonzero=args.min_nonzero, min_games=args.min_games)
        if len(combined)
        else ({}, pd.DataFrame())
    )

    # S4.4 pre-registered named-keeper prior: the LOCKED expectation is recorded regardless; when an
    # (owner-injected) {player_id: name} map is supplied, the confirmatory face-validity CHECK runs and
    # the per-keeper table gains a `keeper_name` column. Name resolution is injected, never parsed here.
    names_map = json.loads(Path(args.keeper_names_json).read_text(encoding="utf-8")) if args.keeper_names_json else None
    named_check = None
    if names_map is not None and len(per_keeper):
        per_keeper, named_check = _named_keeper_check(per_keeper, names_map)

    out = {
        "verdicts": verdicts,
        "named_keeper": named,
        "named_keeper_prior": {
            "expected_deterrent": NAMED_KEEPER_PRIOR,
            "caveated": NAMED_KEEPER_CAVEATED,
            "locked": NAMED_KEEPER_PRIOR_LOCKED,
            "check": named_check,  # null unless --keeper-names-json supplied (join is owner-injected)
        },
        # S4.3 dropped-AND-counted: corpus keeper-resolution totals, so an unresolved keeper is a known
        # denominator, never a silent drop.
        "keeper_identity": {
            "n_keeper_teams": res.counters.get("n_keeper_teams"),
            "n_keeper_teams_resolved": res.counters.get("n_keeper_teams_resolved"),
            "n_keeper_teams_unresolved": res.counters.get("n_keeper_teams_unresolved"),
            "n_unresolved_keeper_frames": named.get("n_unresolved_keeper_frames") if named else 0,
        },
        # S4.5: the PRE-REGISTERED thresholds/positions these verdicts were judged against, in the artifact
        # so a reader never has to reconstruct them from the code.
        "registered_constants": {
            "SATURATING_MULTIPLE": SATURATING_MULTIPLE,
            "PHYSICS_ARM_PROBE_RATIO": PHYSICS_ARM_PROBE_RATIO,
            "R": R,
            "MIN_DOMAIN_FRAMES": MIN_DOMAIN_FRAMES,
            "REGIME_I_LADDER_M": REGIME_I_LADDER_M,
            "REALISTIC_MIN_DISP_M": REALISTIC_MIN_DISP_M,
            "saturating_positions": {"goalline": "defended goal-line centre", "x30_gr_m": SATURATING_X30_GR},
        },
        "provider_support": _provider_support_matrix(args.providers.split(",")),  # S4.5
        "n_frames_scored": len(combined),
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "partition": worker,
        "input_contract": input_contract(),
    }
    # `metrics.json` (not `verdicts_*.json`): the ADR-056 staleness detector globs `metrics.json` keyed
    # on `input_contract.driver`, so this is what wires the declared contract to a live gate. Only the
    # authoritative UNPARTITIONED pass reaches here, so there is exactly one.
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # The "A" deliverable: the per-KEEPER sign table (one row per resolved keeper), not the raw frames.
    if len(per_keeper):
        per_keeper.to_parquet(dest / "named_keeper_signs.parquet", index=False)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
