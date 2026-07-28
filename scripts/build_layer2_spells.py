"""Maintainer driver: persist TF-19 Layer 2 spells for the sign-off ATT power leg.

`scripts/run_signoff_power.py --spells <parquet>` consumes the table this writes. Splitting the
corpus pass out of the power script is not a convenience -- it is the fix for a MEASURED failure:
the power script built spells inline, spent 8.7h walking 64 matches, then died in the cheap
analysis step that followed and lost every one of them, because nothing had been written to disk.

Two properties follow from that, and both are load-bearing:

* **Per-match shards, written on completion.** A crash resumes from the shard directory instead of
  restarting the corpus. An existing shard is skipped.
* **Partitionable.** `--match-ids-json` pins which matches THIS process handles, so N processes
  share one `--out` and the pass is wall-clock bounded by the slowest partition rather than the
  sum of all matches.

FIREWALL (spec S5.1): this driver builds the Layer 2 DESIGN and never estimates its ATT. The
persisted table carries `Z` and the outcome columns because power simulation needs the treatment
assignment and the base rate -- `att_power_curve` draws its own outcomes from an `InjectionSpec`
and cannot be handed an observed one. Do not add a "report the observed ATT" flag here either.

Usage (on the box, scripts/ on sys.path, pining token in env):
  python scripts/build_layer2_spells.py --list-matches
  python scripts/build_layer2_spells.py --out <DIR> --match-ids-json <SLICE.json>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _aggregate_manifests(dest) -> dict:
    """Corpus-wide totals plus the two counts that decide whether the ATT design is estimable.

    `n_treated` is reported at corpus scope for a reason: a rare treatment is invisible per match
    (one match can legitimately contain zero) but decides the whole power curve, because a cluster
    resample drawn from a corpus with almost no treated units yields replicates that carry a single
    treatment class and cannot be estimated at all.
    """
    from scripts._partition import aggregate_manifests

    corpus = aggregate_manifests(dest, defaults=("n_matches", "n_spells", "n_treated"))
    n, k = corpus["n_spells"], corpus["n_treated"]
    corpus["treated_prevalence"] = (k / n) if n else None
    return corpus


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="output dir (not needed with --list-matches)")
    ap.add_argument("--providers", default="gradientsports")
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--tracking-limit", type=int, default=None)
    ap.add_argument(
        "--match-ids-json",
        default=None,
        help=(
            'JSON {"gradientsports": ["10502", ...]} pinning WHICH matches this process handles. '
            "This is how the corpus pass is PARALLELISED: split the id list N ways and launch N "
            "processes, each with its own slice and a SHARED --out."
        ),
    )
    ap.add_argument("--allow-dirty", action="store_true", help="permit a dirty tree (dev only; manifest is marked)")
    ap.add_argument(
        "--list-matches",
        action="store_true",
        help="print the available match ids as JSON and exit (build the parallel split from this)",
    )
    args = ap.parse_args()

    if not args.list_matches and not args.out:
        raise SystemExit("--out is required unless --list-matches is given")

    from scripts._provenance import git_provenance, require_clean_tree

    # --list-matches writes no artifact, so it is exempt from the clean-tree requirement.
    prov = (
        {"commit": "n/a", "dirty": False, "dirty_files": []}
        if args.list_matches
        else require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)
    )

    if args.list_matches:
        from scripts._partition import list_match_ids

        print(json.dumps(list_match_ids(args.providers.split(",")), indent=2))
        return

    import pandas as pd

    from scripts._loader_pining import load_matches
    from scripts._partition import providers_for_slice, worker_tag, write_table_atomically
    from silly_kicks.causal import build_opportunities, layer2_config
    from silly_kicks.causal._confounders import join_layer2_confounders

    match_ids = json.loads(Path(args.match_ids_json).read_text(encoding="utf-8")) if args.match_ids_json else None
    # A provider absent from THIS slice belongs to another worker; the loader would otherwise read
    # that as "the whole manifest" (see providers_for_slice for the measured behaviour).
    providers = providers_for_slice(args.providers.split(","), match_ids)
    tag = worker_tag(args.match_ids_json)  # BEFORE the loop: the shard write uses it per match
    dest = Path(args.out)
    shard_dir = dest / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    totals = {"n_matches": 0, "n_spells": 0, "n_treated": 0}

    # load_matches yields (provider, match_id, ACTIONS, FRAMES, home_team_id) -- actions FIRST.
    for provider, match_id, actions, frames, home_team_id in load_matches(
        providers=providers,
        match_ids=match_ids,
        max_per_provider=args.max_per_provider,
        tracking_limit=args.tracking_limit,
    ):
        shard = shard_dir / f"{provider}__{match_id}.parquet"
        if shard.is_file():
            print(f"  skip {match_id}: shard exists", flush=True)
            continue

        sp = build_opportunities(
            frames, actions, home_team_id=home_team_id, model_metadata={}, config=layer2_config({})
        )
        if len(sp):
            sp = join_layer2_confounders(sp, frames=frames, actions=actions, home_team_id=home_team_id)
            sp = sp.copy()
            sp["provider"] = str(provider)
            sp["match_id"] = str(match_id)
            totals["n_treated"] += int(sp["Z"].sum())
        totals["n_matches"] += 1
        totals["n_spells"] += len(sp)

        # Written even when EMPTY: an absent shard means "not yet run", a present empty one means
        # "run, produced no spell". Conflating them would make a resume silently recompute.
        write_table_atomically(sp, shard, tag=tag)
        print(f"  {match_id}: {len(sp)} spells -> {shard.name}", flush=True)

    shards = sorted(shard_dir.glob("*.parquet"))
    frames_in = [pd.read_parquet(s) for s in shards]
    combined = pd.concat([f for f in frames_in if len(f)], ignore_index=True) if frames_in else pd.DataFrame()
    if len(combined):
        write_table_atomically(combined, dest / "layer2_spells.parquet", tag=tag)
    (dest / f"manifest_{tag}.json").write_text(
        json.dumps(
            {**totals, "run_commit": prov["commit"], "run_tree_dirty": prov["dirty"], "partition": tag},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    corpus = _aggregate_manifests(dest)
    corpus["spells_path"] = str(dest / "layer2_spells.parquet") if len(combined) else None
    corpus["n_rows_written"] = len(combined)
    (dest / "layer2_spells_manifest.json").write_text(json.dumps(corpus, indent=2, default=str), encoding="utf-8")
    print(json.dumps(corpus, indent=2, default=str))


if __name__ == "__main__":
    main()
