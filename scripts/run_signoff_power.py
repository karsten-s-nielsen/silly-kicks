"""Maintainer driver: the TF-19 sign-off power curves (ICC + ATT).

Produces the two curves S6.4 needs to become signable:

* the **ICC** power curve at all three ``ICC_ANCHORS`` -- discharging S6.1's registered precondition
  ("the gate is registered only if detection at the anchor ... is >= 0.8"), which shipped in PR-3 as
  a docstring promise no code could keep;
* the **ATT** power curves at all three ``ATT_RELATIVE_ANCHORS`` for BOTH Layer 2 outcomes, from
  which ``N_MIN_MATCHED`` is taken as the MAXIMUM of the two (``Y_close_attempt`` has the lower base
  rate, so an ``N_min`` derived on ``Y_attempt`` alone would be anti-conservative for the outcome
  the decider's row 7 fires on).

FIREWALL (spec S5.1): this script NEVER estimates an ATT on the observed outcome. `att_power_curve`
accepts no outcome vector at all -- only an `InjectionSpec` recipe it draws from itself -- so Layer
2's real contrast stays unread until PR-3b. Do not add an "and also report the observed ATT" flag.

Usage (on the box, scripts/ on sys.path, pining token in env). Build the spells table FIRST with
`scripts/build_layer2_spells.py` -- it is shardable, resumable and partitionable, whereas the inline
fallback below is a single serial corpus walk that must start over if anything downstream raises:

  python scripts/build_layer2_spells.py --out <SPELLDIR> --match-ids-json <SLICE.json>
  python scripts/run_signoff_power.py --out <DIR> --spells <SPELLDIR>/layer2_spells.parquet \
      --arm-values <ARMDIR>/arm_values_delta_das.parquet [--seed 0]

Both input tables must carry CLEAN provenance: their manifests are read and a dirty, unprovenanced
or mixed-commit input is refused, because this run's own clean SHA would otherwise describe numbers
derived from code it does not name.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_design_matrix(spells, confounders):
    """Assemble ``X``, failing LOUD on an unusable column rather than inside sklearn.

    MEASURED: ``fit_propensity`` on an X carrying one all-NaN column raises
    ``ValueError: Input X contains NaN. LogisticRegression does not accept missing value`` -- which
    names no column and surfaces deep in a corpus run. Silently DROPPING the column would be worse
    still: it would weaken the registered design with no record.

    The check covers PARTIAL non-finiteness too, not merely all-NaN columns. sklearn rejects a
    single NaN cell, and every row belongs to some cluster, so a resample will eventually include
    it -- meaning a partially-NaN confounder fails the run just as surely as a dead one, only later
    and with a message that names nothing. Dropping the offending ROWS is equally rejected: it would
    silently redefine the estimation sample, which is a design change, not error handling.
    """
    import numpy as np

    missing = [c for c in confounders if c not in spells.columns]
    if missing:
        raise ValueError(f"design matrix: confounders absent from spells: {missing}")
    counts = {
        c: int((~np.isfinite(spells[c].to_numpy(dtype=float))).sum())
        for c in confounders
        if not np.isfinite(spells[c].to_numpy(dtype=float)).all()
    }
    if counts:
        n = len(spells)
        detail = ", ".join(f"{c}: {k}/{n} non-finite" for c, k in sorted(counts.items()))
        raise ValueError(
            f"design matrix: confounders carry non-finite values ({detail}). fit_propensity would "
            "raise 'Input X contains NaN' without naming them, part-way through the run. Fix the "
            "confounder join upstream -- do not drop rows here, that would change the design."
        )
    return spells.loc[:, list(confounders)].to_numpy(dtype=float)


def _upstream_provenance(table_path: str, *, allow_dirty: bool) -> dict:
    """Read the provenance of an INPUT table produced by another driver.

    CLAUDE.md states the rule this implements: an artifact whose inputs came from another driver
    needs provenance on BOTH, or the clean SHA on the downstream metrics launders the dirty
    upstream input. The producers write a `*_manifest.json` beside their table carrying
    `run_commit` / `run_tree_dirty`; a table with NO manifest is reported as unknown rather than
    assumed clean, on the same fail-closed principle as a missing git checkout.

    A dirty UPSTREAM is refused exactly like a dirty tree -- a clean run over a dirty input is not
    a clean result.
    """
    p = Path(table_path)
    # EVERY manifest in the directory, not `sorted(...)[0]`. Picking one relies on the corpus
    # manifest happening to sort before the per-worker files -- true for both current producers
    # purely by naming, and silently false the moment a partition is named differently, at which
    # point this would certify ONE partition while describing a whole corpus. That is the same
    # "artifact misdescribes its own scope" defect as the last-writer-wins manifest bug.
    #
    # Consistency is also DERIVED from the commits actually recorded rather than read from a
    # self-reported `commit_consistent` field, so a pre-4.64.0 manifest that predates the field is
    # validated on evidence instead of fail-opening on a missing key.
    found = sorted(p.parent.glob("*manifest*.json"))
    commits: set[str] = set()
    dirty = not found  # no manifest at all => unprovenanced => treated exactly like dirty
    for f in found:
        m = json.loads(f.read_text(encoding="utf-8"))
        rc = m.get("run_commit")
        if isinstance(rc, list):
            commits.update(str(x) for x in rc)
        elif rc:
            commits.add(str(rc))
        dirty = dirty or bool(m.get("run_tree_dirty", True))

    prov = {
        "path": str(p),
        "run_commit": (next(iter(commits)) if len(commits) == 1 else sorted(commits)) or "unknown",
        "run_tree_dirty": dirty,
        "commit_consistent": len(commits) <= 1,
        "manifests": [str(f) for f in found],
    }
    if dirty and not allow_dirty:
        where = f"{len(found)} manifest(s)" if found else "NO manifest found beside the table"
        raise SystemExit(
            f"upstream input {p} is not certified clean ({where}, commit {prov['run_commit']}). "
            "Stamping this run's clean SHA on numbers derived from it would record a provenance "
            "that is verifiable-looking and false. Rebuild the input from a clean tree, or pass "
            "--allow-dirty to mark this artifact dirty too."
        )
    if not prov["commit_consistent"]:
        raise SystemExit(
            f"upstream input {p} was built by workers at DIFFERENT commits "
            f"({prov['run_commit']}) -- it is a blend of code versions, not one run."
        )
    return prov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; artifact is marked)")
    ap.add_argument("--lock-commit", default=None, help="the commit the run was registered against")
    ap.add_argument(
        "--arm-values",
        default=None,
        help="parquet of per-action gkdv arm values (arm_value, keeper_key, game_id) for the ICC leg",
    )
    ap.add_argument(
        "--spells",
        default=None,
        help=(
            "parquet of prebuilt Layer 2 spells from scripts/build_layer2_spells.py. STRONGLY "
            "preferred: without it this script walks the whole corpus inline, which took 8.7h and "
            "was then lost entirely when the analysis step that follows it raised."
        ),
    )
    args = ap.parse_args()

    # Provenance FIRST: refuse a dirty tree before any corpus work is paid for.
    from scripts._provenance import git_provenance, require_clean_tree

    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)

    import pandas as pd

    from silly_kicks._group_metrics import icc_power_curve
    from silly_kicks.causal import LAYER2_CONFOUNDERS
    from silly_kicks.causal.power import InjectionSpec, att_power_curve
    from silly_kicks.gkdv._validate import ATT_RELATIVE_ANCHORS, ICC_ANCHORS

    # Provenance of every UPSTREAM artifact, not just this run's. A clean SHA here would otherwise
    # launder a dirty input: the spells table and the arm-values table are what the ATT and ICC
    # numbers are computed FROM, and they are produced by different drivers at different times.
    upstream = {}
    for label, path in (("spells", args.spells), ("arm_values", args.arm_values)):
        if path:
            upstream[label] = _upstream_provenance(path, allow_dirty=args.allow_dirty)

    if args.spells:
        # The corpus pass is a SEPARATE, shardable, resumable, partitionable driver. Consuming its
        # output here means a crash in the analysis below costs seconds to retry, not another
        # corpus walk -- which is exactly what was lost the first time this ran.
        spells = pd.read_parquet(args.spells)
    else:
        # Imported HERE, not at the top: the `--spells` path must not require the pining loader (or
        # its credentials) merely to re-run the cheap analysis on an already-built table.
        from scripts._driver import for_each, shard_path
        from scripts._loader_pining import load_matches
        from silly_kicks.causal import build_opportunities, layer2_config
        from silly_kicks.causal._confounders import join_layer2_confounders

        # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
        def _work(item):
            _provider, _match_id, actions, frames, home_team_id = item
            sp = build_opportunities(
                frames, actions, home_team_id=home_team_id, model_metadata={}, config=layer2_config({})
            )
            if not len(sp):
                return None  # still writes an EMPTY shard: "ran, produced no spell"
            return join_layer2_confounders(sp, frames=frames, actions=actions, home_team_id=home_team_id)

        # This is the 8.7-hour path the module docstring warns about: it walked 64 matches, raised in
        # the cheap analysis below, and lost every one of them. Sharding it means the SAME crash now
        # costs a re-read. `--spells` remains the preferred route -- that driver is also
        # partitionable -- but the fallback must not stay the trap it was.
        res = for_each(
            load_matches(
                providers=args.providers.split(","),
                max_per_provider=args.max_per_provider,
                tracking_limit=args.tracking_limit,
            ),
            key=lambda item: (str(item[0]), str(item[1])),
            work=_work,
            shard_root=Path(args.out) / "shards",
            # Mirrors `build_layer2_spells`' declaration, because it is the same computation: the
            # opportunity builder, its Layer-2 config, and the confounder join determine a spell.
            # `matching.py` is NOT declared -- the analysis below re-reads these shards on every
            # invocation, so it consumes the content rather than determining it.
            token_inputs={
                "layer2_config": "v1",
                "build_opportunities": "v1",
                "join_layer2_confounders": "v1",
                "tracking_limit": args.tracking_limit,
            },
            tag="signoff_spells",
            label="match",
        )
        if res.failures:
            raise RuntimeError(f"{len(res.failures)} match(es) failed: {res.failures}. Re-run to retry only them.")
        # Combined from THIS PASS'S keys, not `_driver.reconcile`: this driver has no partition
        # surface (no --match-ids-json, no worker tag), so a whole-generation read would fold in
        # matches from a wider earlier run over the same --out. See `reconcile`'s docstring.
        parts = [f for f in (pd.read_parquet(shard_path(res.shard_dir, k)) for k in res.keys) if len(f)]
        spells = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    X = build_design_matrix(spells, LAYER2_CONFOUNDERS)
    Z = spells["Z"].to_numpy(dtype=int)
    clusters = spells["game_id"].to_numpy()
    sizes = tuple(int(s) for s in (500, 1000, 2000, 4000, 8000) if s <= len(spells))

    # Treated prevalence decides whether ANY of this is estimable, so it is reported next to the
    # curves rather than left to be inferred from them. A resample carrying one treatment class is
    # counted as `n_degenerate_by_size`; a size where that approaches `n_replicates` is reporting an
    # INESTIMABLE design at that n, which is a different statement from a weak effect and must not
    # be read as one.
    n_treated = int(Z.sum())
    prevalence = float(Z.mean()) if len(Z) else 0.0
    print(f"spells={len(spells)} treated={n_treated} prevalence={prevalence:.4f}", flush=True)

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
        "run_commit": prov["commit"],
        "run_tree_dirty": prov["dirty"],
        "run_tree_state": prov["tree_state"],
        "n_spells": len(spells),
        "n_treated": n_treated,
        "treated_prevalence": prevalence,
        "spells_source": args.spells or "inline corpus pass",
        "upstream_provenance": upstream,
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
