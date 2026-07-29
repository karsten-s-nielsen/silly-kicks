# TF-19 §3.3 GK-confounder entanglement — the F6 correction

**Run:** 2026-07-28, `run_commit d1fc18d`, `run_tree_dirty: false`.
**Driver:** `scripts/validate_xshot_causal.py --out …` (analysis pass over 179 persisted shards).
**Corpus:** 179 matches — skillcorner 108, idsse 7, gradientsports 64 — 98,789 opportunities,
carrier coverage 1.0 on all three providers, so all three are included.

## Why this run exists (ADR-037 finding F6)

4.60.0 recorded the xS arm's re-gate verdict as `joins_with_caveat`. `regate_verdict` reads
`entanglement` **only** on a `pass`, v2 returned `pass` — and the driver had hard-coded that input
as a **registered default**, annotated *"inert unless the probe surprises with `pass`"*. It
surprised. So a parameter documented as inert decided the verdict, and
`scripts/validate_xshot_causal.py`, which measures it, had never been run.

Nothing was overclaimed — `joins_with_caveat` is the conservative branch — but the attribution was
false. This is the measurement.

## Result

| | |
|---|---|
| ATT without GK | 0.0952 (SE 0.0070) |
| ATT with GK | 0.0882 (SE 0.0074) |
| GK ablation shift | **−0.006999533** |
| Placebo band p95 (cluster) | 0.004689687 |
| `GK_ABLATION_MIN_SHIFT` | 0.01 |
| **entanglement** | **`inside_band`** (refused: `False`) |
| control conversions | 730 (floor 30 — R10 never fired) |
| PS overlap / GK NaN | 1.0 / 0.0 |

The GK block moves the ATT **more than the permutation null** (0.0070 > 0.0047) but **below** the
registered absolute floor of 0.01, so `clears = False` → `inside_band`.

`regate_verdict(arm="shot", probe_verdict="pass", entanglement="inside_band")` → **`joins_with_caveat`**,
routing to `joins_the_metric`.

**The measured value equals the default that was assumed.** F6 is therefore closed by confirmation,
not reversal: the verdict is unchanged but now earned. That is a weaker claim than "we found the
default was wrong" and a stronger one than "we assumed it was right."

## Two properties of this artifact worth reading

**`commit_consistent: false` in this file is a FALSE POSITIVE, fixed in the same release (4.68.0).**
The value is left exactly as the run produced it — a research artifact records what was computed,
and editing the number afterwards is the falsification this cycle exists to prevent.

What it actually reported: the directory held eight worker manifests **unanimously at `6b242cf`**,
each having built 21–23 matches (179 total), plus one analysis manifest at `d1fc18d` carrying
**`n_matches: 0`** — it built nothing, because every shard already existed. A pass that contributed
no data was voting on the data's lineage. **The corpus was single-commit throughout.**

`_partition.aggregate_manifests` now lets only manifests that CONTRIBUTED data vote, and reports
`commits_seen` alongside so a non-contributing pass stays visible rather than silently absorbed.
Re-aggregating these same manifests yields `commit_consistent: true`, `run_commit: 6b242cf`. The
reason for narrowing it: a guard that cries wolf is worse than no guard — it teaches readers to skim
past the one field built to be un-skippable.

Independently of the flag, the two-commit *execution* (shards at `6b242cf`, analysis at `d1fc18d`)
is checkably benign: 4.66.0 changed only `silly_kicks/causal/matching.py` and `__init__.py` —
`git diff 6b242cf d1fc18d` over `causal/opportunities.py`, `causal/_confounders.py` and `tracking/`
is **empty**, so every input to shard construction is byte-identical across the two commits.

**This run required 4.66.0 to exist at all.** On the first attempt the analysis died with
`'<' not supported between instances of 'int' and 'str'`: `game_id` is `int` for gradientsports and
`str` for the other two, and `_cluster_reassign` sorted its cluster ids. The deeper defect the crash
exposed was that `game_id` alone is not a valid cluster key for an arm that **pools** providers — a
stringifying repair would have merged gradientsports `123` with skillcorner `"123"` into one
cluster, corrupting the very cluster-exchangeable null the placebo band is drawn from. The crash was
the lucky failure mode. See the ADR-037 amendment (4.66.0).
