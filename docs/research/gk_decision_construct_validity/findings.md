# TF-62 GK build-up decision-quality construct-validity battery (findings)

**Reported-not-gated.** This battery promotes no default and flips no gate (ADR-092); it records whether
the `silly_kicks.gk_decision` chosen-vs-available metric is construct-valid **as an instrument**. The
verdict is **positive on the native tier** — the metric is responsive, discriminating, and carries a real
net-of-team keeper signal at owner-tier scale — with the standing honest limit that it is validated **as an
instrument, NOT as a per-keeper ranking**. The reconstruction (SB360 + full-tracking) tier is **moderately
fidelity-faithful** to native at the shipped reachability default.

- **run_commit:** `22575debf4e36baa063db49de8d914d9c4dea87d` (tree clean, `run_tree_dirty: false` — the TF-62
  library commit; the report is the ADR-092 §12.7 clean-tree second commit). Input-contract digest
  `c345986c6dbfba87905ecfda698ef7633b2c4a1d95469a118ddd327f2638ba69`.
- **Corpus (owner-tier, aggregate-only per the reversibility-not-provenance rule; raw data never committed):**
  - **Native tier:** 909 SkillCorner Game-Intelligence matches (906 contributing decisions) →
    **22,934 GK build-up decisions / 153 keepers / 92 teams**.
  - **Reconstruction fidelity (SkillCorner Rosetta Stone):** the native-GI vs reconstructed-from-tracking
    comparison ran over the **875 S1-geometry-passing** SkillCorner matches of the 889 fully-cached set
    (**14 excluded = 1.6%** by the loader's spec-4.4 ball-off-pitch rate-gate — normal data-quality
    attrition, no clustering), yielding **1,764 shared `(keeper, game_id)` pairs**.
  - **SB360 reachability sweep:** 60 StatsBomb-360 matches → 2,366 reconstructed GK decisions at reachability 0.
- **Metric + registered params (frozen `GkDecisionParams`):** option value `EV(o) = xpass_completion(o) ×
  (1 + max(0, opponents_bypassed(o)))` (`option_value = "completion_progression"`); `min_options = 3`;
  `reachability_min_xpass = 0.85` (shipped default); `min_per_keeper = 5`. Metrics: `decision_value`
  (chosen − mean available), `sel_efficiency` (chosen / max), `decision_pct` (fraction of alternatives beaten,
  ties split half → random = 0.5).

## Headline — native tier VALIDATED as an instrument

| Leg | Result | Verdict |
|---|---|---|
| Responsiveness | `decision_pct` 0.565 vs 0.5 (**t = +26.96**); `decision_value` 0.274 vs 0 (**t = +27.26**) | **responsive** |
| Discrimination (one-way keeper ICC vs a keeper-label permutation null) | `decision_value` **0.039**, `sel_efficiency` **0.037**, `decision_pct` **0.028** — all vs null p95 ~0.0012, **p = 0.000** (n_keepers 149, n 22,921) | **discriminating** |
| Net of team (club-adjusted leave-one-keeper-out keeper ICC) | `decision_value` **0.061**, `sel_efficiency` **0.073**, `decision_pct` **0.053** — all **p = 0.000** (n_keepers 113 multi-keeper-team, n 17,879) | **real net of team** |
| Transfer (crossing-keeper residual sign-agreement / correlation) | 11 usable crossing keepers; sign-agreement 0.64/0.55/0.45, resid-corr 0.05/−0.10/−0.18 | **inconclusive → ranking not licensed** |

**Interpretation.** The metric moves the right way (keepers beat a random-choice baseline by ~27 σ), and
*which* keeper is making the decision explains variance beyond a permutation null (one-way ICC ~0.03–0.04,
p=0). The team component is of similar magnitude (team one-way ICC ~0.026–0.036), so the raw signal is
**largely-but-not-entirely team** — but a **leave-one-keeper-out club-adjusted** ICC of **0.05–0.07 (p=0)**
shows a genuine within-club keeper signal survives team removal. (The naive team-fixed-effect residual ICC
collapses to ~0.002–0.003 because subtracting a team mean that *includes* the keeper absorbs a
single-dominant-keeper team's own signal into its baseline — which is exactly why the leave-one-keeper-out
club-adjusted estimator is the correct net-of-team measure, following Eyestone.) This reproduces the Eyestone
collaboration's "real, modest, largely-but-not-entirely-team" verdict at 906-match scale. The **transfer leg
is underpowered** (only 11 keepers change clubs in this corpus) and is **inconclusive**, so a per-keeper
**ranking is NOT licensed**.

## Reconstruction fidelity — moderate, significant (definitive uncapped run)

Per-`(keeper, game_id)` Spearman ρ of native-GI vs reconstructed-from-tracking, at the shipped reachability
default (0.85), over **n = 1,764** shared pairs:

| metric | ρ | p | n |
|---|---|---|---|
| `decision_value` | **0.247** | 6.7e-26 | 1,764 |
| `sel_efficiency` | **0.238** | 3.9e-24 | 1,764 |

**Both p < 0.001 by ~20 orders of magnitude** — the reconstruction tier is a moderately faithful rank-proxy
for native decision quality. (An earlier 60-match-capped run reported sel_eff ρ 0.334 / decision_value ρ 0.232
at n = 121; the full-corpus values converge at **ρ ≈ 0.24 for both metrics**, confirming the capped sel_eff
was a small-sample over-estimate and firming the decision_value significance from a borderline p = 0.010 to
p ≈ 1e-26. This uncapped run is the definitive fidelity number.)

## SB360 reachability sweep — why the shipped default is 0.85

Threshold grid over the reconstructed SB360 option rows (built at reachability 0 = all options), re-filtering
alternatives by `completion >= threshold` and recomputing `decision_pct`:

| reachability threshold | decision_pct mean | t vs 0.5 | sel_efficiency mean | n decisions |
|---|---|---|---|---|
| 0.00 | 0.398 | **−14.77** | 0.567 | 2,366 |
| 0.50 | 0.402 | **−14.03** | 0.569 | 2,359 |
| 0.70 | 0.429 | −9.77 | 0.592 | 2,293 |
| **0.85** | **0.547** | **+5.29** | 0.724 | 1,880 |
| 0.90 | 0.615 | +10.30 | 0.803 | 1,330 |
| 0.95 | 0.715 | +10.97 | 0.892 | 395 |

**recommended_threshold = 0.85 = shipped_default.** On zero-velocity SB360 the "all visible teammates" option
set (threshold 0/0.5) **inverts** `decision_pct` below 0.5 (t ≈ −14): the bundled WC2022 xPass runs generous on
SB360, so ~95% of reconstructed options clear 0.5 and the 0.5 filter can't prune the unrealistic
`opponents_bypassed`-heavy upfield options the keeper correctly declines. The **0.85** filter matches the
provider's xPass distribution, prunes them, and restores responsiveness (t = +5.29). A deeper per-provider
xPass recalibration for SB360 is a future ADR-009 refinement (not required for responsiveness — 0.85 handles it).

## Honest limit (load-bearing)

Validated as an **instrument**, **NOT** player-attributable. The net-of-team signal is real *within*
multi-keeper teams, but the transfer-robust keeper-intrinsic component is underpowered/unconfirmed (11 crossing
keepers), so **per-keeper numbers are NOT a ranking**. Ranking is a future ADR-009 gated on a crossed
keeper+team ICC over a larger multi-club transfer corpus (same posture as ADR-090). The metric is also
**value-function dependent**: `completion × progression` penalises deliberately-direct keepers even net of
team; the pluggable typed value-fn seam is the hook for xT-/retention-based variants (reserved-typed-not-built).

## Caveats
- **Reported-not-gated:** these numbers flip no gate and trigger no retrain / re-materialize.
- **Aggregate-only:** per the reversibility-not-provenance rule, only corpus statistics (t / ICC / ρ) are
  emitted; no per-keeper or per-decision rows are committed, and the report is not reconstructable to the raw
  owner-tier data.
- **Reconstruction-fidelity corpus bound:** the ρ is over the **875 S1-passing** SkillCorner matches (14/889
  = 1.6% dropped by the loader geometry gate) — the *right* conservatism for a fidelity comparison (a
  geometrically-broken match's reconstructed tracking would corrupt the comparison), recorded here so the
  number is not misread as "all 889".
- **Native leg is bit-reproducible:** the native verdicts reduce over the same 909 for_each shards under a
  fixed RNG seed, so this run's native block is byte-identical to the earlier capped run's — only the
  reconstruction corpus differs.
