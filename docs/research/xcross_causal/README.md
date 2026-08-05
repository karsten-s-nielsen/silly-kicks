# xCross causal harness — 4.74.0 re-run

Re-run on the corrected geometry (ADR-051 PR 5), and **the first provenanced version of this
artifact**.

    run_commit  35a66679c7bd   run_tree_dirty  false   status  ok

## Verdict: unchanged

    gk_clears_placebo_band  False        (was False)
    causal_claim_supported  True

## Do NOT read the magnitudes as a before/after of the geometry fix

The corpus changed substantially between the two runs, so the numbers are **not** comparable:

| | previous (4.18.0) | this run (4.74.0) |
|---|---|---|
| `n_opportunities` | 23,966 | **52,978** |
| `n_treated` | 669 | **4,193** |
| `gk_ablation_shift` | +0.017937 | −0.005962 |
| `placebo_band_p95` | — | 0.009063 |
| `gk_clears_placebo_band` | False | **False** |

The opportunity count more than doubled and the treated count grew six-fold. That is a **corpus and
pipeline difference**, not an effect of the chiral-transform correction — 4.66.0 alone changed how the
pooled arm keys clusters, after finding that `game_id` is `int` for Gradient Sports and `str` for the
other providers and that `game_id` on its own is not a valid cluster key for a pooled arm (the crash
was the lucky failure mode; silent cross-provider cluster fusion was the other).

Writing "the shift moved from +0.0179 to −0.0060" would attribute several releases of pipeline change
to this PR. The defensible statement is narrower and is the one above: **the verdict is stable.**

## Provenance

The previous artifact carried **no `run_commit` and no `run_tree_dirty`** — the third recorded
instance of that class, and one the provenance gate could not see, because `ARTIFACT_DRIVERS` is
hand-maintained and its only anti-rot assertion is a floor (`>= 6` against 14 entries). The driver was
enrolled and wired in 4.74.0, and the gate was landed **red** first — observed failing 3 of its 5
per-driver assertions — before the wiring went in.

## Precision caveat

See spec K10. `estimate_att` is 1:1 nearest-neighbour matching, so every ablation shift and every
placebo draw is an exact multiple of `1/n_treated`. Here that spacing is `1/4193 = 0.000239`. The
printed digits overstate the resolution; do not quote these figures to more than ~4 significant
figures, and do not read two runs landing on the same value as anything other than the same lattice
point.
