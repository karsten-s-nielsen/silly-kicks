# TF-46 Arm 1 corner-offence — construct-validity memo

**Corpus:** owner-tier SkillCorner (909 matches, 3 seasons) + IDSSE/Sportec (7) + StatsBomb SB360 (327).
**Provenance:** `metrics.json` in this directory, `run_commit` `6ee7e62`, `run_tree_dirty: false`
(ADR-037). The numbers below are the reading of that file; the file is the source of truth.

The encoder (`RunZoneModel`, GMM) and the routine basis (`CornerRoutineModel`, NMF) are UNSUPERVISED.
TF-46 names the construct by its mechanism, so there is no accuracy claim and no accuracy gate. This is an
instrument, not a validated team-quality ranking.

## Headline

Corner **shot generation (`n_shots`) separates teams beyond chance and survives netting out the opponent
faced and the routine mix** (ICC 0.068 vs permutation null 0.038, p = 0.000; opponent-adjusted p = 0.043;
routine-fixed-effect p = 0.000; n = 6082 corners, 201 teams). This is a real team-level instrument of
corner shot creation.

## What it does NOT measure

- **Causal routine value.** Per-routine chance quality is an outcome ASSOCIATION, not a causal valuation
  of the routine (that is Arm 3). It does not control for finishing skill.
- **Goal / xG ranking.** The team signal holds for `n_shots` only. `n_goals` and `sum_xg` do not clear
  the permutation null at this corpus (rare events: 0.023 goals per corner; xG defined on 3089 corners).
- **Season-to-season skill.** Transfer is not established (see gate 3).
- **Defender position beyond the opponent mean.** The net-of-opponent leg removes defending-team strength
  only as a leave-one-attacker-out mean, not per setup.
- **SkillCorner xG.** SkillCorner ships no shot xG, so `sum_xg` is honest-NaN there, never imputed.

## The three gates (`docs/howto/construct-validity.md`)

### 1. Responsiveness — PASS
The per-corner chance quality moves off its neutral baseline on real data: `n_shots` mean 0.265,
`n_goals` mean 0.023, `sum_xg` mean 0.032 per corner.

### 2. Discrimination, net of team — PASS for `n_shots`; not for `n_goals` / `sum_xg`
One-way team ICC vs a 1000-permutation label null, then net of the opponent faced (leave-one-attacker-out
defending-team residual) and net of the routine mix (dominant-routine fixed effect).

| metric | one-way ICC | null p95 | p | opponent-adjusted p | routine-FE p |
|---|---|---|---|---|---|
| `n_shots` | 0.068 | 0.038 | 0.000 | 0.043 | 0.000 |
| `n_goals` | 0.034 | 0.044 | 0.404 | 0.199 | 0.410 |
| `sum_xg`  | 0.043 | 0.051 | 0.174 | 0.236 | 0.160 |

`n_shots` clears the null and survives both adjustments — the team shot-generation signal is not merely
"faced weak defences" nor an artefact of which routine a team favours. `n_goals` and `sum_xg` stay inside
the null throughout: underpowered, not a team signal at this corpus.

### 3. Transfer (season to season) — NOT ESTABLISHED
A team's season residual (team-season mean net of the leave-the-team-out season-league baseline) in its
two most-sampled seasons, correlated across teams. `n_shots`: 59 teams, sign-agreement 0.46, residual
correlation 0.08. `n_goals`: 0.47 / -0.11. `sum_xg`: 36 teams, 0.50 / -0.12. Sign-agreement near 0.5 and
residual correlation near 0 — season-to-season persistence is not demonstrated. This is a stated
limitation, not a pass; a longer per-team history is needed to settle it.

## Reported structural + reliability checks (spec section 12)

- **12.5 cross-provider stability.** Per-provider `n_shots` means differ by provider shot definition
  (SkillCorner 0.19, IDSSE 0.18, StatsBomb SB360 0.34), so the metric is comparable WITHIN a provider,
  not across. Goal rates agree more closely (0.021 / 0.035 / 0.024).
- **12.6 convergent timing.** TF-79 first-contact vs the SkillCorner GI first player-possession time
  agree strongly: Spearman rho 0.9997 (p = 0), n = 2965 corners, median offset 1.5 s (GI possession lands
  ~1.5 s after first contact; the 8.3 s mean is pulled by a second-phase right tail). The two independent
  onset measures track the same event — the timing primitive is convergent-valid.
- **12.7 SB360 init divergence.** The -2 s (training) vs kick-frame (SB360 serve) initial positions, on
  IDSSE (continuous tracking at both), diverge by 1.10 m mean / 3.35 m p95 over 577 players, flipping the
  argmax initial zone for 5.2% of players. This bounds the SB360 kick-frame serve error at ~5% initial-zone
  reassignment versus the -2 s-trained zones.
- **Non-vacuity, orientation invariance, synthetic recovery, NMF reconstruction curve.** Covered CI
  test-side (`tests/corners/`, `tests/scripts/test_validate_corners.py`) and in the fit summary
  (`../../../` fit `metrics.json`: target GMM k = 7, NMF reconstruction 52.9 -> 1.86 at the rank-15 elbow,
  42 run columns).
- **Fetch integrity.** `n_manifest_listing_failed` = 0 and `n_matches_gi_fetch_failed` = 0: no provider
  listing or GI fetch was silently swallowed; every dark leg would be attributed.

## Reading (GO / NO-GO)

GO as a corner **shot-generation** instrument, net of opponent and routine. NO-GO for a goal- or
xG-based team ranking, and NO-GO for a season-predictive claim, at this corpus. Use within-provider.
Arm 3 (causal routine valuation) remains a later cycle.
