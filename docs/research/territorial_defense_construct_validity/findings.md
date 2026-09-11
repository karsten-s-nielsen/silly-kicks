# TF-54b territorial-defense construct-validity battery (findings)

**Reported-not-gated.** This battery promotes no default and flips no gate (ADR-090); it records
whether the SB360 territorial-defense removal counterfactual is construct-valid **as an instrument**.
The verdict is a clear **negative**: on the full StatsBomb-360 corpus the threat arm is
`instrument_void` / `not_responsive` under the dose battery. This confirms — and strengthens — the
honest limit the metric already ships with ("validated as an INSTRUMENT, NOT player-attributable /
NOT a ranking").

- **run_commit:** `ce0401a9f6d38f99333ffcefa047b0211f317f7d` (tree clean; the TF-54b library commit —
  the battery ran against a clean checkout, the two commit-2 artifacts shelved for its duration).
- **Corpus:** **321 matches** — the full StatsBomb-360 corpus reachable via the pining loader
  (`load_statsbomb_matches`: WC2022 + the multi-competition open-data 360 set + the licensed 360
  cohort). `n_failed=0`, `n_counters_unrecorded=0`.
- **Populations:** 6,852 domain defenders; Arm A scored 6,359 `(defender, match)` rows, Arm B scored
  3,271; the dose battery pooled **23,366 finite domain frames** (70,098 single-player placebo draws).
- **Registered thresholds (locked, from the input contract):** `MIN_DOMAIN_FRAMES=200`,
  `realistic_disp_m=2.0`, `saturating_disp_m=10.0`, `SATURATING_MULTIPLE=5.0`, `TD_PROBE_RATIO=2.0`,
  `n_placebo=3`. Arm params: `pitch_control_method="spearman"`, `defensive_action_type_ids={9,10,18}`,
  `trim_fraction=0.70`, `lambda_gk=3.0`, `local_radius_m=10.0`, `min_local_observed_fraction=0.70`,
  `min_defenders_after_removal=1`, `arm_b_rule="nearest_to_target"`, `own_half_max_x=52.5`.

## Headline

**The SB360 territorial-defense threat arm is a VOID / non-responsive instrument under the dose probe.**

| Layer | Verdict |
|---|---|
| Layer 0 — instrument validity | `instrument_void` |
| Layer 1 — responsiveness | `not_responsive` |

Pooled dose |Δthreat| statistics (attacker-value units; `positive = threat suppressed`):

| quantity | value | meaning |
|---|---|---|
| realistic dose median (`real_med`, ±2 m) | **0.0** | the contesting defender moved a realistic 2 m |
| saturating dose median (`sat_med`, ±10 m) | **0.0** | the same defender moved a saturating 10 m |
| dosed-defender median (`defender_med`) | **0.0** | Layer-1 responsiveness quantity |
| nearest-defender control median (`nd_med`) | **0.0** | one defending outfielder moved by the same vector |
| single-outfielder placebo p95 (`placebo_p95`) | 0.1367 | R single-player placebos (95th pct) |

**Interpretation.** `n_domain=23,366` is far above the 200-frame floor, so this is **not** a coverage
artefact (`arm_unscoreable` was ruled out). The dose response is *genuinely* zero at the median:
moving the contesting defender by a realistic 2 m — or even a saturating 10 m — leaves the
zero-velocity positional pitch-control threat unchanged in **more than half** of scored freeze-frames
(median exactly 0.0), while a random single-player placebo move reaches a p95 of 0.137. The instrument
fails both Layer-0 legs (the `>= 5x realistic` leg is vacuous with `real_med=0`, so the placebo-band
backstop decides it: `sat_med 0.0 < placebo_p95 0.137`) and the Layer-1 responsiveness test
(`defender_med 0.0 < TD_PROBE_RATIO x max(nd_med, placebo_p95)`). On crowded SB360 penalty-box
freeze-frames a single defender's small displacement is absorbed by the remaining players, so the
marginal-removal / marginal-displacement threat delta is a very weak signal. This is the probe working
as designed — detecting that the arm is not a responsive instrument — **not** a null "no-effect"
claim, and it mirrors the TF-60 Layer-3 and xtgk_v2 construct-validity outcomes (a counterfactual
metric that does not validate stays as experimental infrastructure and promotes nothing).

## Named-defender face validity — 2 of 3 (PRE-REGISTERED prior, locked 2026-09-09)

Prior (elite defender -> expected `positive` = threat-suppressing), locked before the run:
`{"Van Dijk": "positive", "Gvardiol": "positive", "Otamendi": "positive"}`.

| defender | expected | observed (mean `a_threat_suppressed`) | n_matches | meets prior |
|---|---|---|---|---|
| Van Dijk | positive | positive (+7.4e-05) | 10 | YES |
| Gvardiol | positive | positive (+1.9e-03) | 12 | YES |
| Otamendi | positive | negative (0.0) | 7 | NO |

**2 of 3**, but the magnitudes are noise-level (10^-5 to 10^-3), consistent with the weak-instrument
headline — this is face-validity corroboration only, never a measurement or a ranking. The full
per-`(defender, match)` sign table is the machine-readable deliverable `named_defender_signs.parquet`
(6,852 defenders; `player_name` populated where StatsBomb carried it).

## Arm-B attribution slippage

`mean_attribution_slippage = NaN` over `n=0` scored defenders — the slippage is **honest-NaN**, by
construction, on anonymous SB360 freeze-frames (identity of the position-chosen contesting defender is
un-measurable; ADR-027 forbids a fabricated 0). This is the designed behaviour, not a defect: Arm B's
attribution-error diagnostic is only defined where defender identity is observable.

## Cross-team replication / the confound

`n_defenders_multi_game=1477`, `identifies_confound=false`. The marginal-removal delta is
team-conditioned by construction (vacated space re-partitions to teammates), so per-defender numbers
are **NOT** a defender ranking; the elite-defender ("Van Dijk") prior is elite-defender / elite-team
**collinear**. A crossed defender+team ICC over a **multi-club transfer** corpus (with a team column
and cross-team appearances) is the future ADR-009 gate — this single-tournament / national-team-heavy
corpus has no identifying power for the defender-vs-team confound even at 1,477 multi-game defenders.

## Caveats
- **Reported-not-gated:** these numbers flip no gate and trigger no retrain / re-materialize.
- **Threat arm only:** the dose battery probes `a_threat_suppressed` (`compute_threat_pc` on the
  removal counterfactual); threat suppression is a Tier-1 dimensionless lift at zero velocity (ADR-063),
  so SB360 freeze-frames are in-scope (unlike ΔDAS, which is NaN on velocity-less providers).
- **Instrument, not attribution:** validated as an instrument only; per-defender numbers are not a
  ranking (the honest limit already in the code + ADR-090).
