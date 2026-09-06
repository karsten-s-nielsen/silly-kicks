"""Owner-run construct-validity pass for the TF-54b counterfactual territorial-dominance metric.

Runs the PRE-REGISTERED battery (spec section 7): component-level completion AUC/ECE/Brier-skill;
the synthetic-interception target-recovery of the failed-pass target model versus the "death =
intercept" and origin-zone-centroid baselines; the composed mechanism-versus-v1 discriminant; the
locked elite-defender ("Van Dijk") prior; plus reliability / discriminant / outcome-lens numbers
that are REPORTED, never gating.

Leakage discipline (spec 7.1): the injected ``ExpectedThreat`` and ``PassCompletionModel`` are fit on
a corpus DISJOINT from the scored matches -- the pass K-folds the corpus by match, fitting on the
train folds and scoring the held-out fold, so no scored match is in its own fit set. The validation
fits its OWN models (never the bundled weights), so ``run_commit`` fully describes it.

The expensive per-match corpus load is sharded with ``for_each`` (ADR-052): one shard per match
holding that match's actions, resumable on a crash. The K-fold fit + score + the whole battery run
in the reduce over all shards, off the network. The pre-registered constants and locked prior are
committed BEFORE the owner run (the TF-19 NAMED_KEEPER_PRIOR idiom) and stamped into the artifact, so
the gate cannot be moved after the numbers are seen; ``decide_promotion`` reads ONLY those constants.

The frame-level correctness (the counterfactual valuation itself) is validated by the OWNER RUN and
the territory unit tests; the offline tests here pin the pure pieces -- the synthetic-interception
perturbation, the locked constants, ``decide_promotion``, and the provenance wiring.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/validate_territory_counterfactual.py --out <DIR> [--providers statsbomb] \
      [--max-per-provider N] [--tracking-limit N] [--n-folds K] [--match-ids-json FILE]
"""

from __future__ import annotations

import argparse
import json
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._input_contract import declare_inputs
from scripts._synthetic_interception import perturb_interception

# ---------------------------------------------------------------------------------------------------
# PRE-REGISTERED, LOCKED constants (spec CONSIDER-1). Committed BEFORE the owner run and stamped into
# the artifact, mirroring TF-19's NAMED_KEEPER_PRIOR: the gate cannot be moved after seeing the run.

#: The locked elite-defender ("Van Dijk") prior: acknowledged elite WC2022 centre-backs, matched by
#: an ASCII-folded surname substring against the roster ``player_name`` (so an accented roster name
#: like "Antonio Rudiger" still matches). A non-empty frozen constant; str members per SPEC-04.
ELITE_DEFENDER_PRIOR: frozenset[int | str] = frozenset(
    {
        "van Dijk",  # Virgil van Dijk (Netherlands)
        "Gvardiol",  # Josko Gvardiol (Croatia)
        "Rudiger",  # Antonio Rudiger (Germany)
        "Marquinhos",  # Marquinhos (Brazil)
        "Otamendi",  # Nicolas Otamendi (Argentina)
    }
)

#: Completion-model floors (mirroring GkRetentionModel's ece <= 0.10 style gate). The Brier floor is a
#: SKILL score (BSS = 1 - brier / brier_noskill against the held-out base rate), NOT a fixed ceiling:
#: a fixed 0.22 ceiling is looser than no-skill at ~75% completion and would gate nothing.
COMPLETION_AUC_FLOOR: float = 0.65
COMPLETION_ECE_CEILING: float = 0.10
COMPLETION_BRIER_SKILL_FLOOR: float = 0.10
#: The elite-defender prior clears iff the matched elites land at or above this rank-quantile of the
#: territorial-dominance value at meaningful pass-faced volume.
ELITE_DEFENDER_TOP_QUANTILE: float = 0.75

#: The date the constants above were locked (stamped into the artifact as the pre-registration record).
PRE_REGISTRATION_LOCKED = "2026-09-05 (before the owner run)"


def brier_skill_score(brier: float, base_rate: float) -> float:
    """BSS = 1 - brier / brier_noskill, where brier_noskill = base_rate * (1 - base_rate) (PLAN-09).

    A no-skill model predicting the base rate everywhere scores ``brier_noskill``; BSS above 0 beats
    it. NaN when the base rate is degenerate (all-complete or all-fail), where no-skill is trivial.
    """
    noskill = base_rate * (1 - base_rate)
    return 1 - brier / noskill if noskill > 0 else float("nan")


def decide_promotion(metrics: dict) -> dict:
    """The promotion verdict -- reads ONLY the locked pre-registered constants (spec section 7.2).

    Promote iff the completion model clears its three floors AND the composed counterfactual beats
    both naive target-recovery baselines AND the elite-defender prior clears its top-quantile. The
    Brier check is a SKILL score against the held-out no-skill Brier, never a raw fixed ceiling. No
    threshold is inlined here: every comparison references a module-level constant.

    The gate itself covers only the COMPUTABLE legs above -- spec section 7.2's Primary-1 real-data
    mechanism leg is infeasible event-only (a real failed pass's intended target is unobservable; see
    ``_REAL_DATA_LEG_STATUS``) and is never gated on. That status lives in
    ``metrics["mechanism"]["real_data_leg"]``, but a bare ``promote: True`` invites over-reading it as
    covering that leg too -- so this also surfaces two TRANSPARENCY fields (``real_data_leg_uncomputed``
    / ``promote_scope``) that make the omission visible on the decision itself, without changing the
    promote/no-promote logic above.
    """
    comp = metrics["completion"]
    bss = brier_skill_score(float(comp["brier"]), float(comp["base_rate"]))
    completion_ok = (
        float(comp["auc"]) >= COMPLETION_AUC_FLOOR
        and float(comp["ece"]) <= COMPLETION_ECE_CEILING
        and bss >= COMPLETION_BRIER_SKILL_FLOOR
    )

    mech = metrics["mechanism"]
    cf_err = float(mech["counterfactual_error"])
    mechanism_ok = cf_err < float(mech["baseline_death_error"]) and cf_err < float(mech["baseline_centroid_error"])

    prior_ok = float(metrics["elite_prior"]["elite_quantile"]) >= ELITE_DEFENDER_TOP_QUANTILE

    real_data_leg = mech.get("real_data_leg", {})
    real_data_leg_uncomputed = real_data_leg.get("status") == "not_computed_requires_owner_decision"

    return {
        "promote": bool(completion_ok and mechanism_ok and prior_ok),
        "completion_ok": bool(completion_ok),
        "mechanism_ok": bool(mechanism_ok),
        "prior_ok": bool(prior_ok),
        "brier_skill_score": bss,
        "real_data_leg_uncomputed": bool(real_data_leg_uncomputed),
        "promote_scope": (
            "promote reflects the synthetic mechanism + prior only; Primary-1 real-data leg is "
            "infeasible event-only (see mechanism.real_data_leg)"
        ),
    }


def input_contract() -> dict:
    """Declare WHICH SYMBOLS these numbers depend on (ADR-056).

    The territory / counterfactual params, the locked pre-registered floors + prior, and the extractor
    modules the valuation runs through -- a change to any moves the digest and the staleness detector
    flags a committed artifact.
    """
    from dataclasses import asdict

    from silly_kicks.territory import CounterfactualParams, TerritoryParams

    return declare_inputs(
        driver="validate_territory_counterfactual",
        params={
            "territory": asdict(TerritoryParams()),
            "counterfactual": asdict(CounterfactualParams()),
            "completion_auc_floor": COMPLETION_AUC_FLOOR,
            "completion_ece_ceiling": COMPLETION_ECE_CEILING,
            "completion_brier_skill_floor": COMPLETION_BRIER_SKILL_FLOOR,
            "elite_defender_top_quantile": ELITE_DEFENDER_TOP_QUANTILE,
            "elite_defender_prior": sorted(str(x) for x in ELITE_DEFENDER_PRIOR),
        },
        extractors=(
            "silly_kicks.territory._counterfactual",
            "silly_kicks.territory._compute",
            "silly_kicks.expected_passing._model",
        ),
        models=(
            "silly_kicks.expected_passing.PassCompletionModel",
            "silly_kicks.xthreat.ExpectedThreat",
        ),
    )


# ---------------------------------------------------------------------------------------------------
# Pure reduce helpers (owner-run; frame-level numerics validated by the owner pass, not a unit test).


def _ascii_fold(s: object) -> str:
    """Accent-fold to lowercase ASCII, so a locked ASCII surname matches an accented roster name."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii").lower()


def _is_elite(player_name: object) -> bool:
    folded = _ascii_fold(player_name)
    return any(_ascii_fold(name) in folded for name in ELITE_DEFENDER_PRIOR)


def completion_metrics(y: np.ndarray, oof: np.ndarray) -> dict:
    """Held-out AUC / ECE / Brier + the base rate over the finite-prediction rows."""
    from sklearn.metrics import roc_auc_score

    from silly_kicks._calibration_metrics import ece

    keep = np.isfinite(oof)
    yk = y[keep].astype(float)
    ok = oof[keep].astype(float)
    if not len(ok):
        return {"auc": float("nan"), "ece": float("nan"), "brier": float("nan"), "base_rate": float("nan"), "n": 0}
    auc = float(roc_auc_score(yk, ok)) if len(np.unique(yk)) > 1 else float("nan")
    return {
        "auc": auc,
        "ece": float(ece(yk, ok)),
        "brier": float(np.mean((ok - yk) ** 2)),
        "base_rate": float(yk.mean()),
        "n": len(ok),
    }


def _recover_target(xt, completion_model, origin, death, *, cone_deg: float, min_support: float, conditioned: bool):
    """The q*c-weighted mean cone zone-centre (the counterfactual's recovered target).

    ``conditioned=True`` restricts to the death-direction cone (the metric's estimator);
    ``conditioned=False`` uses the whole unconditioned destination distribution (the origin-zone
    centroid baseline). Returns ``(x, y)`` or ``None`` when the selected support is below the floor.
    """
    from silly_kicks.territory._counterfactual import _within_cone
    from silly_kicks.xthreat import destination_profiles

    ox, oy = float(origin[0]), float(origin[1])
    prof = destination_profiles(xt, np.array([ox]), np.array([oy]))
    centres = prof.zone_centres
    probs = prof.probabilities[0]
    if conditioned:
        sel = _within_cone((death[0] - ox, death[1] - oy), centres, (ox, oy), cone_deg)
    else:
        sel = np.ones(len(centres), dtype=bool)
    support = float(probs[sel].sum())
    if support < min_support or not sel.any():
        return None
    q = probs[sel] / support
    c = np.asarray(
        completion_model.predict_completion(
            np.full(int(sel.sum()), ox), np.full(int(sel.sum()), oy), centres[sel, 0], centres[sel, 1]
        ),
        dtype=float,
    )
    w = q * c
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0:
        return None
    w = w / total
    return float((w * centres[sel, 0]).sum()), float((w * centres[sel, 1]).sum())


def target_recovery_battery(
    completed_passes: pd.DataFrame, xt, completion_model, *, cone_deg: float, min_support: float, seed: int
) -> dict:
    """Synthetic-interception recovery (SPEC-02 / SPEC-09) over held-out completed passes.

    For each completed pass (true end = ground truth), synthesize an interception at a random
    flight-fraction AND angular offset, hide the true end, and recover the target three ways:
    counterfactual (cone-conditioned), "death = the synthetic intercept", and the origin-zone
    centroid (unconditioned). Reports the mean recovery error of each -- the counterfactual should
    beat both baselines.
    """
    rng = np.random.default_rng(seed)
    cf_err: list[float] = []
    death_err: list[float] = []
    centroid_err: list[float] = []
    sx = completed_passes["start_x"].to_numpy(dtype=float)
    sy = completed_passes["start_y"].to_numpy(dtype=float)
    ex = completed_passes["end_x"].to_numpy(dtype=float)
    ey = completed_passes["end_y"].to_numpy(dtype=float)
    for i in range(len(sx)):
        origin = (float(sx[i]), float(sy[i]))
        true_end = (float(ex[i]), float(ey[i]))
        if not all(np.isfinite([*origin, *true_end])):
            continue
        f = float(rng.uniform(0.3, 0.8))
        delta = float(rng.uniform(-0.6, 0.6))
        dx, dy = perturb_interception(origin, true_end, fraction=f, angle_offset_rad=delta)
        death = (float(dx), float(dy))
        recovered = _recover_target(
            xt, completion_model, origin, death, cone_deg=cone_deg, min_support=min_support, conditioned=True
        )
        if recovered is None:
            continue
        centroid = _recover_target(
            xt, completion_model, origin, death, cone_deg=cone_deg, min_support=min_support, conditioned=False
        )
        if centroid is None:
            continue
        cf_err.append(float(np.hypot(recovered[0] - true_end[0], recovered[1] - true_end[1])))
        death_err.append(float(np.hypot(death[0] - true_end[0], death[1] - true_end[1])))
        centroid_err.append(float(np.hypot(centroid[0] - true_end[0], centroid[1] - true_end[1])))
    return {
        "counterfactual_error": float(np.mean(cf_err)) if cf_err else float("nan"),
        "baseline_death_error": float(np.mean(death_err)) if death_err else float("nan"),
        "baseline_centroid_error": float(np.mean(centroid_err)) if centroid_err else float("nan"),
        "n": len(cf_err),
    }


def _elite_name_match_counts(defender_table: pd.DataFrame) -> dict:
    """Per locked-name resolution census (spec IMPORTANT-4): how many DISTINCT players each locked
    ELITE_DEFENDER_PRIOR name resolves to. A name matching 0 (absent) or >1 (ambiguous) players must
    be visible in the artifact, never silently under- or over-counted."""
    from silly_kicks.id_compat import canonical_id

    names = sorted(str(x) for x in ELITE_DEFENDER_PRIOR)
    if not len(defender_table):
        return {n: {"n_players": 0, "player_ids": []} for n in names}
    folded = [_ascii_fold(n) for n in defender_table["player_name"]]
    out: dict = {}
    for name in names:
        key = _ascii_fold(name)
        ids = [pid for pid, fn in zip(defender_table["player_id"], folded, strict=False) if key in fn]
        distinct = sorted({str(canonical_id(i)) for i in ids})
        out[name] = {"n_players": len(distinct), "player_ids": distinct}
    return out


def elite_prior_verdict(defender_table: pd.DataFrame, *, min_volume: int) -> dict:
    """Where the locked elite defenders land on the territorial-dominance value (spec 7.2 Primary-2).

    Ranks defenders at meaningful pass-faced volume and reports the median rank-quantile of the
    matched elites (fraction of defenders they meet or exceed), plus the per-locked-name resolution
    census so a name matching 0 or >1 players is visible (IMPORTANT-4).
    """
    from silly_kicks.territory._columns import TR_PASSES_AIMED_INTO_HULL, TR_XT_PREVENTED

    match_counts = _elite_name_match_counts(defender_table)
    ambiguous = sorted(n for n, v in match_counts.items() if v["n_players"] != 1)
    base = {"name_match_counts": match_counts, "names_not_uniquely_resolved": ambiguous}
    if not len(defender_table) or TR_PASSES_AIMED_INTO_HULL not in defender_table.columns:
        return {"elite_quantile": float("nan"), "n_elite_matched": 0, "n_defenders": 0, **base}

    at_volume = defender_table[defender_table[TR_PASSES_AIMED_INTO_HULL] >= min_volume].copy()
    if not len(at_volume):
        return {"elite_quantile": float("nan"), "n_elite_matched": 0, "n_defenders": 0, **base}
    # Rank on total threat PREVENTED in the defender's territory (the "Van Dijk" prior value).
    at_volume["_quantile"] = at_volume[TR_XT_PREVENTED].rank(pct=True, method="average")
    elites = at_volume[at_volume["is_elite"]]
    quantiles = {str(n): float(q) for n, q in zip(elites["player_name"], elites["_quantile"], strict=False)}
    return {
        "elite_quantile": float(elites["_quantile"].median()) if len(elites) else float("nan"),
        "n_elite_matched": len(elites),
        "n_defenders": len(at_volume),
        "elite_quantiles": quantiles,
        **base,
    }


def _build_name_map(actions: pd.DataFrame) -> dict:
    """Canonical ``player_id -> player_name`` from the source actions (the roster identity the loader
    attaches). territory samples never carry ``player_name``, so the elite prior JOINS it from here
    (C2 -- the territory output schema was never going to carry a name)."""
    from silly_kicks.id_compat import canonical_id_series

    if not len(actions) or "player_name" not in actions.columns:
        return {}
    sub = actions.dropna(subset=["player_id"])
    pid = canonical_id_series(sub["player_id"])
    out: dict = {}
    for p, n in zip(pid, sub["player_name"], strict=False):
        if pd.notna(n) and p not in out:
            out[p] = str(n)
    return out


def _build_team_map(actions: pd.DataFrame) -> dict:
    """Canonical ``player_id -> canonical team_id`` (a player has one team; last write wins)."""
    from silly_kicks.id_compat import canonical_id_series

    if not len(actions):
        return {}
    sub = actions.dropna(subset=["player_id", "team_id"])
    pid = canonical_id_series(sub["player_id"])
    tid = canonical_id_series(sub["team_id"])
    return dict(zip(pid, tid, strict=False))


def _team_shots_conceded(actions: pd.DataFrame) -> dict:
    """Canonical ``team_id -> shots faced`` (a defensive-strength proxy): per game, a team concedes
    every shot taken by any OTHER team in that game."""
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.id_compat import canonical_id_series

    shot_id = spadlconfig.actiontype_id["shot"]
    if not len(actions):
        return {}
    df = pd.DataFrame(
        {
            "g": canonical_id_series(actions["game_id"]),
            "t": canonical_id_series(actions["team_id"]),
            "is_shot": (actions["type_id"].to_numpy() == shot_id).astype(int),
        }
    ).dropna(subset=["g", "t"])
    per_gt = df.groupby(["g", "t"], sort=False)["is_shot"].sum().reset_index()  # columns: g, t, is_shot
    per_gt["conceded"] = per_gt.groupby("g")["is_shot"].transform("sum") - per_gt["is_shot"]
    conceded: dict = {}
    for team, val in zip(per_gt["t"], per_gt["conceded"], strict=False):
        conceded[team] = conceded.get(team, 0.0) + float(val)
    return conceded


def _shuffle_pass_outcomes(actions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """A shuffled-outcome placebo: permute the completed/failed ``result_id`` AMONG pass rows only, so
    the geometry-outcome link is destroyed while the marginal completion rate is preserved."""
    import silly_kicks.spadl.config as spadlconfig

    pass_id = spadlconfig.actiontype_id["pass"]
    out = actions.copy()
    idx = np.where(out["type_id"].to_numpy() == pass_id)[0]
    if len(idx) > 1:
        vals = out["result_id"].to_numpy().copy()
        vals[idx] = vals[rng.permutation(idx)]
        out["result_id"] = vals
    return out


def _aggregate_defenders(samples: pd.DataFrame, *, name_map: dict) -> pd.DataFrame:
    """Pool per-match samples into one row per defender -- METHOD/COLUMNS-AWARE (C1).

    Sums only the metric columns actually present, so this works on BOTH the counterfactual table and
    the v1 ``completed_failed`` table (which lacks the counterfactual-only columns). ``player_name`` is
    JOINED from the source-actions ``name_map`` by canonical id (C2), never read off the territory
    output (which never carries it)."""
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.territory._columns import (
        TR_EXPECTED_THREAT_FACED,
        TR_PASSES_AIMED_INTO_HULL,
        TR_PASSES_INTO_HULL,
        TR_XT_CONCEDED,
        TR_XT_NET,
        TR_XT_PREVENTED,
        TR_XT_PREVENTED_ABOVE_EXPECTATION,
    )

    if not len(samples):
        return pd.DataFrame()
    scored = samples[samples["territory_hull_source"] == "resolved"].copy()
    if not len(scored):
        return pd.DataFrame()
    candidate = (
        TR_XT_CONCEDED,
        TR_XT_PREVENTED,
        TR_XT_NET,
        TR_XT_PREVENTED_ABOVE_EXPECTATION,
        TR_EXPECTED_THREAT_FACED,
        TR_PASSES_INTO_HULL,
        TR_PASSES_AIMED_INTO_HULL,
    )
    sum_cols = [c for c in candidate if c in scored.columns]
    agg = scored.groupby("player_id", dropna=True)[sum_cols].sum().reset_index()
    agg["player_name"] = [name_map.get(canonical_id(pid), str(pid)) for pid in agg["player_id"]]
    agg["is_elite"] = [bool(_is_elite(n)) for n in agg["player_name"]]
    return agg


def _corr_with_team_strength(table: pd.DataFrame, team_map: dict, team_conceded: dict) -> float | None:
    """Spearman of per-defender threat-prevented vs their team's shots-conceded (defensive strength)."""
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.territory._columns import TR_XT_PREVENTED

    if not len(table) or TR_XT_PREVENTED not in table.columns:
        return None
    prevented: list[float] = []
    conceded: list[float] = []
    for pid, prev in zip(table["player_id"], table[TR_XT_PREVENTED], strict=False):
        team = team_map.get(canonical_id(pid))
        if team is None or team not in team_conceded:
            continue
        prevented.append(float(prev))
        conceded.append(float(team_conceded[team]))
    if len(prevented) < 3:
        return None
    r = pd.Series(prevented).corr(pd.Series(conceded), method="spearman")
    return float(r) if pd.notna(r) else None


def _reliability(cf_all: pd.DataFrame, *, name_map: dict) -> dict:
    """Split-half stability: correlate per-defender threat-prevented across two disjoint match halves
    (spec 7.2 Secondary Reliability). REPORTED, never gating."""
    from silly_kicks.id_compat import canonical_id_series
    from silly_kicks.territory._columns import TR_XT_PREVENTED

    if not len(cf_all):
        return {"split_half_spearman": None, "n_common": 0}
    cg = canonical_id_series(cf_all["game_id"])
    games = sorted(cg.dropna().unique())
    if len(games) < 2:
        return {"split_half_spearman": None, "n_common": 0, "note": "need >=2 games for split-half"}
    half = max(1, len(games) // 2)
    ga, gb = set(games[:half]), set(games[half:])
    a = _aggregate_defenders(cf_all[cg.isin(ga).to_numpy()], name_map=name_map)
    b = _aggregate_defenders(cf_all[cg.isin(gb).to_numpy()], name_map=name_map)
    if not len(a) or not len(b):
        return {"split_half_spearman": None, "n_common": 0}
    cols = ["player_id", TR_XT_PREVENTED]
    m = a[cols].merge(b[cols], on="player_id", suffixes=("_a", "_b"))
    if len(m) < 3:
        return {"split_half_spearman": None, "n_common": len(m)}
    r = m[f"{TR_XT_PREVENTED}_a"].corr(m[f"{TR_XT_PREVENTED}_b"], method="spearman")
    return {"split_half_spearman": float(r) if pd.notna(r) else None, "n_common": len(m)}


def _discriminant(
    cf_table: pd.DataFrame, v1_table: pd.DataFrame, placebo_table: pd.DataFrame, *, team_map: dict, team_conceded: dict
) -> dict:
    """Discriminant sub-checks (spec 7.2 Secondary), REPORTED never gating: vs v1, vs volume, vs team
    defensive strength, and beats-a-shuffled-outcome-placebo, plus SPEC-06 non-degeneracy."""
    from silly_kicks.territory._columns import (
        TR_PASSES_AIMED_INTO_HULL,
        TR_XT_CONCEDED,
        TR_XT_NET,
        TR_XT_PREVENTED,
    )

    out: dict = {}
    if not len(cf_table):
        return out
    if len(v1_table) and TR_XT_PREVENTED in v1_table.columns:
        cols = ["player_id", TR_XT_PREVENTED]
        m = cf_table[cols].merge(v1_table[cols], on="player_id", suffixes=("_cf", "_v1"))
        if len(m) >= 3:
            r = m[f"{TR_XT_PREVENTED}_cf"].corr(m[f"{TR_XT_PREVENTED}_v1"], method="spearman")
            out["vs_v1_spearman"] = float(r) if pd.notna(r) else None
    if len(cf_table) >= 3 and TR_PASSES_AIMED_INTO_HULL in cf_table.columns:
        r = cf_table[TR_XT_PREVENTED].corr(cf_table[TR_PASSES_AIMED_INTO_HULL], method="spearman")
        out["vs_volume_spearman"] = float(r) if pd.notna(r) else None
    if len(cf_table) >= 3 and TR_XT_NET in cf_table.columns and TR_XT_CONCEDED in cf_table.columns:
        r = cf_table[TR_XT_NET].corr(-cf_table[TR_XT_CONCEDED], method="spearman")
        out["non_degeneracy_net_vs_neg_conceded_spearman"] = float(r) if pd.notna(r) else None
    real_r = _corr_with_team_strength(cf_table, team_map, team_conceded)
    placebo_r = _corr_with_team_strength(placebo_table, team_map, team_conceded)
    out["vs_team_strength_spearman"] = real_r
    out["placebo_vs_team_strength_spearman"] = placebo_r
    out["beats_shuffled_placebo"] = bool(real_r is not None and placebo_r is not None and abs(real_r) > abs(placebo_r))
    return out


#: The Primary-1 real-data mechanism leg (spec 7.2) is SURFACED, never silently stubbed. A real failed
#: pass's INTENDED target is unobservable event-only; the only leakage-free weak label (the ADR-066
#: trajectory-to-next-touch proxy) needs simultaneous teammate positions, i.e. tracking, absent from
#: this event-only corpus. Which event-only weak label to adopt materially changes the validity read,
#: so it is a research-design decision for the owner -- recorded here rather than approximated.
_REAL_DATA_LEG_STATUS = {
    "status": "not_computed_requires_owner_decision",
    "reason": (
        "a real failed pass's intended target is unobservable event-only; the leakage-free weak label "
        "(ADR-066 trajectory-to-next-touch) needs tracking, absent from this event-only corpus. "
        "Choosing an event-only weak label is an owner research-design decision, not a silent stub."
    ),
}


def run_battery(
    shard_frames: list[pd.DataFrame], *, n_folds: int, cone_deg: float, min_support: float, seed: int
) -> tuple[dict, pd.DataFrame]:
    """The whole reduce: leakage-disjoint K-fold fit + score + the pre-registered battery (spec 7.2).

    ``shard_frames`` is one actions frame per match (each carrying ``game_id`` set to the match id).
    For each fold, fit ``ExpectedThreat`` + ``PassCompletionModel`` on the OTHER folds' actions and
    score the held-out fold's matches under ``method="counterfactual"``, ``method="completed_failed"``
    (v1) and a shuffled-outcome placebo. Accumulates the samples + held-out completion predictions +
    the synthetic-interception recovery, then assembles: completion metrics; the synthetic mechanism
    leg (+ the surfaced real-data leg status); the elite-defender prior; and the reported secondary
    reliability / discriminant legs.
    """
    import silly_kicks.spadl.config as spadlconfig
    from silly_kicks.expected_passing import PassCompletionModel
    from silly_kicks.territory import CounterfactualParams, compute_territorial_dominance
    from silly_kicks.xthreat import ExpectedThreat

    by_match = {str(f["game_id"].iloc[0]): f for f in shard_frames if len(f)}
    match_ids = sorted(by_match)
    n_folds = max(2, min(n_folds, len(match_ids)))
    folds = [list(a) for a in np.array_split(np.array(match_ids, dtype=object), n_folds)]

    pooled = pd.concat(shard_frames, ignore_index=True) if shard_frames else pd.DataFrame()
    name_map = _build_name_map(pooled)
    team_map = _build_team_map(pooled)
    team_conceded = _team_shots_conceded(pooled)

    pass_id = spadlconfig.actiontype_id["pass"]
    success = spadlconfig.result_id["success"]
    cf_params = CounterfactualParams(direction_cone_degrees=cone_deg, min_transition_support=min_support)

    cf_samples: list[pd.DataFrame] = []
    v1_samples: list[pd.DataFrame] = []
    placebo_samples: list[pd.DataFrame] = []
    comp_y: list[np.ndarray] = []
    comp_oof: list[np.ndarray] = []
    recovery_keys = ("counterfactual_error", "baseline_death_error", "baseline_centroid_error")
    rec_sums: dict[str, list[float]] = {k: [] for k in recovery_keys}
    rec_n = 0

    for fi, fold in enumerate(folds):
        held = set(fold)
        train = pd.concat([by_match[m] for m in match_ids if m not in held], ignore_index=True)
        if not len(train):
            continue
        xt = ExpectedThreat(method="singh_counts").fit(train)
        cm = PassCompletionModel().fit(train)
        for m in fold:
            scored_actions = by_match[m]
            cf, _rep = compute_territorial_dominance(
                scored_actions, xt=xt, method="counterfactual", completion_model=cm, cf_params=cf_params
            )
            v1, _rep2 = compute_territorial_dominance(scored_actions, xt=xt)
            placebo_actions = _shuffle_pass_outcomes(scored_actions, np.random.default_rng(seed + fi + 101))
            cf_pl, _rep3 = compute_territorial_dominance(
                placebo_actions, xt=xt, method="counterfactual", completion_model=cm, cf_params=cf_params
            )
            cf_samples.append(cf)
            v1_samples.append(v1)
            placebo_samples.append(cf_pl)
            # Held-out completion metrics for THIS match's passes.
            p = scored_actions[scored_actions["type_id"] == pass_id].dropna(
                subset=["start_x", "start_y", "end_x", "end_y"]
            )
            if len(p):
                comp_y.append((p["result_id"].to_numpy() == success).astype(int))
                comp_oof.append(
                    np.asarray(
                        cm.predict_completion(
                            p["start_x"].to_numpy(),
                            p["start_y"].to_numpy(),
                            p["end_x"].to_numpy(),
                            p["end_y"].to_numpy(),
                        ),
                        dtype=float,
                    )
                )
                completed = p[p["result_id"] == success]
                rec = target_recovery_battery(
                    completed, xt, cm, cone_deg=cone_deg, min_support=min_support, seed=seed + fi
                )
                for k in recovery_keys:
                    if rec["n"] and np.isfinite(rec[k]):
                        rec_sums[k].append(rec[k] * rec["n"])  # volume-weighted running sum
                rec_n += rec["n"]

    cf_all = pd.concat(cf_samples, ignore_index=True) if cf_samples else pd.DataFrame()
    v1_all = pd.concat(v1_samples, ignore_index=True) if v1_samples else pd.DataFrame()
    placebo_all = pd.concat(placebo_samples, ignore_index=True) if placebo_samples else pd.DataFrame()
    defender_table = _aggregate_defenders(cf_all, name_map=name_map)
    v1_table = _aggregate_defenders(v1_all, name_map=name_map)
    placebo_table = _aggregate_defenders(placebo_all, name_map=name_map)

    y = np.concatenate(comp_y) if comp_y else np.array([])
    oof = np.concatenate(comp_oof) if comp_oof else np.array([])

    # Synthetic mechanism leg (top-level keys decide_promotion reads) + the surfaced real-data leg.
    mechanism: dict = {k: (float(np.sum(rec_sums[k]) / rec_n) if rec_sums[k] else float("nan")) for k in recovery_keys}
    mechanism["n"] = int(rec_n)
    mechanism["real_data_leg"] = dict(_REAL_DATA_LEG_STATUS)

    empty_prior = {"elite_quantile": float("nan"), "n_elite_matched": 0, "n_defenders": 0}
    lens_note = "reported-not-gating; possession-reaches-shot AUC not computed offline"
    metrics: dict = {
        "completion": completion_metrics(y, oof),
        "mechanism": mechanism,
        "elite_prior": elite_prior_verdict(defender_table, min_volume=1) if len(defender_table) else empty_prior,
        "secondary": {
            "outcome_lens": {"note": lens_note, "auc": None},
            "reliability": _reliability(cf_all, name_map=name_map),
            "discriminant": _discriminant(
                defender_table, v1_table, placebo_table, team_map=team_map, team_conceded=team_conceded
            ),
        },
        "n_folds": n_folds,
        "n_matches": len(match_ids),
    }
    return metrics, defender_table


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="output dir OUTSIDE the repo (shards + metrics.json + parquet)")
    ap.add_argument(
        "--source",
        choices=("open-data", "pining"),
        default="open-data",
        help="'open-data' = PUBLIC StatsBomb open data (the reproducible default; the locked elite prior "
        "matches men's WC2022) via statsbombpy; 'pining' = the pining providers (owner-tier cross-check).",
    )
    ap.add_argument("--competition-id", type=int, default=43, help="open-data competition (default 43 = World Cup)")
    ap.add_argument("--season-id", type=int, default=106, help="open-data season (default 106 = 2022)")
    ap.add_argument("--providers", default="statsbomb", help="comma-separated pining providers (--source pining only)")
    ap.add_argument("--max-per-provider", type=int, default=None, help="cap the number of matches (both sources)")
    ap.add_argument("--tracking-limit", type=int, default=None, help="cap frames parsed per match (pining only)")
    ap.add_argument("--n-folds", type=int, default=5, help="leakage-disjoint K-fold count (fit-train, score-held-out)")
    ap.add_argument("--min-volume", type=int, default=20, help="min passes-aimed-into-hull for the elite-prior ranking")
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help='JSON {"statsbomb": ["3869685", ...]} pinning WHICH matches this process handles.',
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    args = ap.parse_args()

    # Clean-tree guard FIRST, before any corpus work: a cited construct-validity artifact must record
    # the code that produced it (ADR-037 / ADR-052).
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    from scripts._driver import for_each

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    dest = Path(args.out)

    # open-data (default) = PUBLIC StatsBomb open data so the report is reproducible AND the locked
    # men's-WC2022 elite prior matches; pining = owner-tier cross-check. Both yield the same 5-tuple.
    if args.source == "open-data":
        from scripts._sb_open_data import load_open_data_matches

        matches_iter = load_open_data_matches(
            competition_id=args.competition_id,
            season_id=args.season_id,
            match_ids=(match_ids or {}).get("statsbomb"),
            max_matches=args.max_per_provider,
        )
        source_token = {"source": "open-data", "competition_id": args.competition_id, "season_id": args.season_id}
        corpus_label = f"statsbomb-open (competition {args.competition_id}, season {args.season_id})"
    else:
        from scripts._loader_pining import load_matches
        from scripts._partition import providers_for_slice

        matches_iter = load_matches(
            providers=providers_for_slice(args.providers.split(","), match_ids),
            match_ids=match_ids,
            max_per_provider=args.max_per_provider,
            tracking_limit=args.tracking_limit,
        )
        source_token = {"source": "pining", "providers": args.providers, "tracking_limit": args.tracking_limit}
        # IMPL-05 footgun guard: the pre-registered elite-defender prior is men's-WC2022-specific and is
        # meaningless against other corpora (e.g. the pining 'statsbomb' provider is a PRIVATE women's corpus).
        warnings.warn(
            "Validating from --source pining: the men's-WC2022 elite-defender face-validity prior is "
            "meaningless against other corpora (e.g. the pining 'statsbomb' provider is a PRIVATE women's "
            "corpus) -- pass --source open-data for the pre-registered battery.",
            stacklevel=2,
        )
        corpus_label = f"pining:{args.providers}"

    def _work(item):
        _provider, match_id, actions, _frames, _home = item
        out = actions.copy()
        out["game_id"] = str(match_id)  # stable K-fold grouping key == the match id
        return out

    res = for_each(
        matches_iter,
        key=lambda item: (str(item[0]), str(item[1])),
        work=_work,
        shard_root=dest / "shards",
        token_inputs={
            "metric": "territory_counterfactual",
            "xt_method": "singh_counts",
            "n_folds": args.n_folds,
            **source_token,
        },
        label="match",
    )

    shard_files = sorted(res.shard_dir.glob("*.parquet"))
    shard_frames = [pd.read_parquet(s) for s in shard_files]
    if not shard_frames:
        raise SystemExit("no match shards collected; nothing to validate")

    from silly_kicks.territory import CounterfactualParams

    cf = CounterfactualParams()  # the metric's own cone / support defaults (not gating thresholds)
    metrics, defender_table = run_battery(
        shard_frames,
        n_folds=args.n_folds,
        cone_deg=cf.direction_cone_degrees,
        min_support=cf.min_transition_support,
        seed=20260905,
    )
    if len(defender_table):
        metrics["elite_prior"] = elite_prior_verdict(defender_table, min_volume=args.min_volume)

    decision = decide_promotion(metrics)

    out = {
        **metrics,
        "decision": decision,
        "pre_registration": {
            "elite_defender_prior": sorted(str(x) for x in ELITE_DEFENDER_PRIOR),
            "completion_auc_floor": COMPLETION_AUC_FLOOR,
            "completion_ece_ceiling": COMPLETION_ECE_CEILING,
            "completion_brier_skill_floor": COMPLETION_BRIER_SKILL_FLOOR,
            "elite_defender_top_quantile": ELITE_DEFENDER_TOP_QUANTILE,
            "locked": PRE_REGISTRATION_LOCKED,
        },
        "providers": corpus_label,
        **res.manifest(),
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov.get("tree_state"),
        "input_contract": input_contract(),
    }
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    if len(defender_table):
        defender_table.to_parquet(dest / "named_defender_signs.parquet", index=False)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
