# TF-63 xImpact — in-game win-probability construct-validity report

**Reported, not gated** (spec §9.3). Promotes no default. The numbers below are the bundled
`default` `WinProbabilityModel` (`silly_kicks/win_probability/weights/`) and its `metrics.json`.

## Corpus + provenance

- **3,961** public StatsBomb open-data matches across **80** (competition, season) releases;
  **7,974,436** action rows.
- `training_commit = ae10f8a75c84ef7f8b55297c8ac1d96bbfae8a32`, clean tree (open-data mode, no
  credentials). GroupKFold-by-match out-of-fold for the calibration metrics.
- Model: interval-hazard logistic on `(score_diff, minutes_remaining, base_strength, home,
  man_advantage)` → forward Markov chain on `score_diff`. `base_strength` is leakage-free by date
  (mean xG-supremacy over each team's strictly-earlier matches). v1 ships the base GLM+chain; the
  isotonic recalibration layer is reserved and unused (base calibration is already well inside the
  gates — below).

## Calibration (out-of-fold)

| metric | value | reference |
|---|---|---|
| ECE | **0.0168** | `ece_max = 0.10` (well within) |
| reliability slope | **0.991** | `|slope − 1| ≤ 0.25` |
| Brier | 0.1385 | — |

The 30-match dev smoke had ECE 0.1037 (borderline); the full-corpus fit resolves it — calibration is
the property that governs leverage, and it is excellent.

## Expected-goals sanity gate

The fitted hazard's expected total goals (0-0, full match, neutral) is **2.465** vs the empirical
corpus rate **2.958** goals/match; `|Δ| = 0.492 < tol 0.50` — **passes, but tight** (0.008 margin).
The model slightly **under-predicts total goals**. This is a coarse scale sanity check on the hazard,
not a calibration gate; the win/draw/loss calibration (ECE 0.017) is what governs the leverage
weighting and is unaffected. Disclosed as a known characteristic.

## Face validity — `goal_leverage = ΔP(win | goal) = P(win | score+1) − P(win | score)`

Leverage of scoring the next goal, from the acting team's perspective, at representative game states
(neutral strength, home, even man-advantage), computed from the bundled model:

| game state | P(win) | leverage ΔP(win\|goal) |
|---|---|---|
| kickoff, 0-0 (90 min remaining) | 0.4392 | +0.278 |
| 0-0 at 45' (45 rem) | 0.3608 | +0.388 |
| 0-0 at 85' (5 rem) | 0.0859 | +0.849 |
| **0-0 at 89' (1 rem) — late go-ahead** | 0.0191 | **+0.966** |
| leading +1 at 89' | 0.9853 | +0.015 |
| **leading +3 at 89' — garbage time** | 1.0000 | **−0.000** |
| leading +4 at 89' — garbage time | 1.0000 | +0.000 |
| trailing −1 at 85' | 0.0031 | +0.083 |
| trailing −1 at 89' — late equalizer | 0.0000 | +0.019 |
| trailing −3 at 89' | 0.0000 | +0.000 |

**Reads correctly on the win/garbage axis.** A late **go-ahead** goal that breaks a 0-0 in the 89th
minute is maximal (+0.966 — it nearly converts a coin-flip-losing state to a win); an early goal is
moderate; a **garbage-time** goal while already winning by 3–4 is ≈0. `xImpact = VAEP_adjusted ×
leverage` therefore near-zeroes the 4-0 stoppage-time goal and amplifies the late winner, which is the
metric's whole point.

**Honest limitation — the trailing late *equalizer* scores LOW, by construction.** Because leverage is
ΔP(**win**), a goal that converts a near-certain loss into a likely *draw* (−1 → 0 with a minute left)
moves P(win) only from 0.0000 to 0.0191 — low win-leverage, even though it is a high-*points* swing
(loss → draw = +1 point). A win-probability weighting values *winning*, not *avoiding defeat*; an
expected-points weighting would rank the late equalizer far higher. This is a design property of the
ΔP(win) definition (spec §9.3), disclosed here rather than smoothed over: the spec's shorthand "late
equalizers score high" holds for a late *go-ahead* goal but not for a *trailing* equalizer. An
expected-points variant is a possible future extension (not built; ADR-009).

## Bottom line

The in-game win-probability model is well-calibrated out-of-fold and its `goal_leverage` produces a
coherent, face-valid re-weighting of action value on the win/garbage-time axis. The two disclosed
characteristics — a slight total-goals under-prediction (expected-goals gate margin 0.008) and the
low win-leverage of a trailing equalizer under the ΔP(win) definition — are reported, not gated.
