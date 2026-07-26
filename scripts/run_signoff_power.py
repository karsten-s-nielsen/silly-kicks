"""Maintainer driver: the TF-19 sign-off power curves (ICC + ATT).

Produces the two curves §6.4 needs to become signable:

* the **ICC** power curve at all three ``ICC_ANCHORS`` -- discharging §6.1's registered precondition
  ("the gate is registered only if detection at the anchor ... is >= 0.8"), which shipped in PR-3 as
  a docstring promise no code could keep;
* the **ATT** power curves at all three ``ATT_RELATIVE_ANCHORS`` for BOTH Layer 2 outcomes, from
  which ``N_MIN_MATCHED`` is taken as the MAXIMUM of the two (``Y_close_attempt`` has the lower base
  rate, so an ``N_min`` derived on ``Y_attempt`` alone would be anti-conservative for the outcome
  the decider's row 7 fires on).

FIREWALL (spec §5.1): this script NEVER estimates an ATT on the observed outcome. `att_power_curve`
accepts no outcome vector at all -- only an `InjectionSpec` recipe it draws from itself -- so Layer
2's real contrast stays unread until PR-3b. Do not add an "and also report the observed ATT" flag.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/run_signoff_power.py --out <DIR> [--providers gradientsports] \
      [--max-per-provider N] [--tracking-limit N] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def build_design_matrix(spells, confounders):
    """Assemble ``X``, failing LOUD on an unusable column rather than inside sklearn.

    MEASURED: ``fit_propensity`` on an X carrying one all-NaN column raises
    ``ValueError: Input X contains NaN. LogisticRegression does not accept missing value`` -- which
    names no column and surfaces deep in a corpus run. Silently DROPPING the column would be worse
    still: it would weaken the registered design with no record.
    """
    import numpy as np

    missing = [c for c in confounders if c not in spells.columns]
    if missing:
        raise ValueError(f"design matrix: confounders absent from spells: {missing}")
    dead = [c for c in confounders if not np.isfinite(spells[c].to_numpy(dtype=float)).any()]
    if dead:
        raise ValueError(
            f"design matrix: confounders are entirely non-finite: {dead}. "
            "fit_propensity would raise 'Input X contains NaN' without naming them."
        )
    return spells.loc[:, list(confounders)].to_numpy(dtype=float)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git from PATH is the house pattern
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover -- provenance is best-effort, never fatal
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lock-commit", default=None, help="the commit the run was registered against")
    ap.add_argument(
        "--arm-values",
        default=None,
        help="parquet of per-action gkdv arm values (arm_value, keeper_key, game_id) for the ICC leg",
    )
    args = ap.parse_args()

    import pandas as pd

    from scripts._loader_pining import load_matches
    from silly_kicks._group_metrics import icc_power_curve
    from silly_kicks.causal import LAYER2_CONFOUNDERS, build_opportunities, layer2_config
    from silly_kicks.causal._confounders import join_layer2_confounders
    from silly_kicks.causal.power import InjectionSpec, att_power_curve
    from silly_kicks.gkdv._validate import ATT_RELATIVE_ANCHORS, ICC_ANCHORS

    spells_all = []
    # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
    for _provider, _match_id, actions, frames, home_team_id in load_matches(
        providers=args.providers.split(","),
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        sp = build_opportunities(
            frames, actions, home_team_id=home_team_id, model_metadata={}, config=layer2_config({})
        )
        if not len(sp):
            continue
        sp = join_layer2_confounders(sp, frames=frames, actions=actions, home_team_id=home_team_id)
        spells_all.append(sp)

    spells = pd.concat(spells_all, ignore_index=True)
    X = build_design_matrix(spells, LAYER2_CONFOUNDERS)
    Z = spells["Z"].to_numpy(dtype=int)
    clusters = spells["game_id"].to_numpy()
    sizes = tuple(int(s) for s in (500, 1000, 2000, 4000, 8000) if s <= len(spells))

    att: dict[str, dict] = {}
    for outcome in ("Y_attempt", "Y_close_attempt"):
        base_rate = float(spells[outcome].mean())
        att[outcome] = {"base_rate": base_rate, "anchors": {}}
        for anchor in ATT_RELATIVE_ANCHORS:
            att[outcome]["anchors"][str(anchor)] = att_power_curve(
                Z=Z,
                injection=InjectionSpec(base_rate=base_rate, relative_effect=float(anchor)),
                X=X,
                clusters=clusters,
                sizes=sizes,
                n_replicates=200,
                rng_seed=args.seed,
            )

    # N_MIN_MATCHED: smallest matched-n bin reaching 0.80 at the 0.15 anchor, MAX over both outcomes.
    def _n_min(outcome: str) -> int | None:
        curve = att[outcome]["anchors"]["0.15"]
        ok = [curve["matched_n_by_size"][s] for s in sizes if curve["power_by_size"][s] >= 0.80]
        return min(ok) if ok else None

    per_outcome = {o: _n_min(o) for o in ("Y_attempt", "Y_close_attempt")}
    resolved = [v for v in per_outcome.values() if v is not None]
    n_min = max(resolved) if len(resolved) == len(per_outcome) else None

    # ICC leg. Its input is PER-ACTION ARM VALUES (delta_das / delta_threat_suppression) grouped by
    # keeper and blocked by match -- i.e. the output of a gkdv arm pass over the corpus, which is
    # the expensive leg (accessible-space + Spearman pitch control on every domain frame, and
    # neither arm may use a PitchControlCache: it keys on frame IDENTITY, so a ghost frame would be
    # served its twin's surface and every delta would collapse to zero). That pass is therefore run
    # ONCE and PERSISTED, and this script consumes the table rather than recomputing it per anchor.
    icc = None
    if args.arm_values:
        av = pd.read_parquet(args.arm_values)
        for col in ("arm_value", "keeper_key", "game_id"):
            if col not in av.columns:
                raise ValueError(f"--arm-values table missing required column {col!r}")
        av = av.dropna(subset=["arm_value", "keeper_key", "game_id"])
        icc = icc_power_curve(
            av["arm_value"].to_numpy(dtype=float),
            av["keeper_key"].to_numpy(),
            av["game_id"].to_numpy(),
            anchors=ICC_ANCHORS,
            n_replicates=200,
            rng_seed=args.seed,
        )
        # A keeper appearing in exactly one match makes the block permutation a pure relabelling,
        # so the null equals the observed statistic and nothing is detectable. Report it rather than
        # letting a structurally-zero power read as "no signal".
        spanning = av.groupby("keeper_key")["game_id"].nunique()
        icc["n_keepers"] = len(spanning)
        icc["n_single_match_keepers"] = int((spanning <= 1).sum())

    out = {
        "lock_commit": args.lock_commit,
        "run_commit": _git_commit(),
        "n_spells": len(spells),
        "sizes": list(sizes),
        "att": att,
        "n_min_per_outcome": per_outcome,
        "N_MIN_MATCHED": n_min,
        "icc": icc,
        "note": (
            "FIREWALL: no observed-outcome ATT is computed here. att_power_curve accepts no outcome "
            "vector -- only an InjectionSpec it draws from."
        ),
    }
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "metrics.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_spells", "n_min_per_outcome", "N_MIN_MATCHED")}, indent=2))


if __name__ == "__main__":
    main()
