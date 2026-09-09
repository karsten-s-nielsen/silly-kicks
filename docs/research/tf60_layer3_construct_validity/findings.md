# TF-60 Layer-3 deterrent arms — out-of-sample validity study (ADR-089)

## Verdict

The naive counterfactual rest-defense **deterrent arms do not survive out-of-sample
validation as deterrent metrics**, so they are **demoted to experimental** (removed from
`silly_kicks.restdefense`'s public surface; code retained privately) pending a redesign.

- **Convergent validity** against the team's own Layer-1 rest-defense structure is weak/absent
  (§2) — and that anchor is the wrong criterion anyway (a deterrent's construct is the
  *opponent's* suppressed counter-danger, not the team's own shape).
- **Predictive validity** against the opponent's realized counter-danger (§3) is the decisive
  test. The two **outfield** arms are **confounded by attacking commitment**: their apparent
  deterrent signal is *where the turnover happens*, not the rearguard's suppression — controlling
  for commitment kills it, and a mediation decomposition attributes ~76 % of the outfield-space
  arm's effect to turnover position.
- Only the **keeper-threat** arm shows a small, control-robust deterrent (ATT −0.0013…−0.0023
  xT, significant in **all four** control specifications, not mediated) — but it is
  pooled/SkillCorner-driven, marginal on Gradient Sports, and underpowered on full tracking, so
  it is **not** an independently-validated shippable metric either.

The ghost-GK both-axes convention correctness fix + re-fit weights (the rest of the ADR-089
cycle) ship regardless of this result. The redesign that this study motivates — a
**within-turnover counterfactual counter** — is registered as the next program
(`docs/superpowers/specs/2026-09-08-tf60-counterfactual-counter-redesign-design.md`).

## 1. Data and provenance

- **Corpus:** 179 matches — 64 Gradient Sports, 108 SkillCorner, 7 IDSSE/Sportec.
- **Arm table:** `arm179/layer3_arm_values.parquet`, **`run_commit a867f9b` (clean tree, 30
  shards/manifests)** — the re-fit ghost-GK weights commit (both-axes convention). 101,598
  committed-forward arm samples across 116 keepers. Recorded in `construct_findings.json`
  (`provenance.arm_values_run_commit`); `predval_findings.json` derives from the same table.
- **Arm sign convention:** `actual − ghost` in attacker-value units, so **negative = deterrent**
  (a valid deterrent reduces the opponent's counter-danger).

## 2. Convergent validity (construct anchor) — weak, and the wrong criterion

`construct_validity_anchor.py` correlates each outfield arm against the Layer-1 rest-defense
structure KPIs (numerical superiority behind the ball; −compactness) on all 101,598 samples.
Pre-registered criterion: `rho < 0 ∧ p < 0.05 ∧ |rho| ≥ 0.10`.

| Arm × KPI | Spearman ρ | criterion met |
|---|---|---|
| threat × num_superiority | −0.048 (p≈1e-52) | ✗ (\|ρ\|<0.10) |
| threat × −compactness | +0.132 | ✗ (wrong sign) |
| space × num_superiority | +0.006 | ✗ |
| space × −compactness | +0.027 | ✗ |

**None clears the bar.** But convergent validity against the team's *own* structure was always a
weak proxy: a better rearguard need not correlate with a specific shape KPI. This anchor is
reported, not decisive.

**Named-keeper prior (pre-registered, LOCKED):** Alisson and Neuer → net deterrent. **Not
confirmed** on the arm: Alisson median 0.0 (n=1553), Neuer median 0.0 (n=1343), corpus median
0.0; Mann-Whitney one-sided p=0.355; ADR-060 effect/SE = −1.67 (n.s.). The keeper arm is
zero-dominated (a missing ghost or no counter-threat → exactly 0), so the median is uninformative
for elite keepers — another sign the naive arm is not a clean keeper signal.

## 3. Predictive validity — the decisive test

`predictive_validity.py`. A **turnover** = two consecutive possessions with a team flip and a
live-ball gap ≤ 5 s (**n=15,113** across 179 games; GS 7,796 / SC 6,449 / IDSSE 868). The
**predictor** is the mean arm over team A's committed-forward possession; the **outcome** is team
B's realized counter-danger (peak xT of B's next possession; corpus mean 0.0196, shot rate
0.030). Four estimators per arm: Spearman, and **propensity-matched ATT** (Abadie-Imbens,
treatment = arm below median = the *more-deterrent* half) under four control sets — **none**,
**turnover-x**, **commitment** (A's possession max end-x), **both**. A valid deterrent has
**ATT < 0, significant**.

### 3.1 Outfield arms — confounded by commitment

Pooled (n≈14,800):

| Arm | ATT ctrl_none | ctrl_turnoverx | ctrl_commitment | ctrl_both |
|---|---|---|---|---|
| `rd_outfield_deter_threat` | −0.0008 (t −1.36) | **+0.0013 (t +2.36)** | +0.0005 (t +0.96) | **+0.0014 (t +2.62)** |
| `rd_outfield_deter_space` | **−0.0035 (t −5.21)** | −0.00004 (t −0.06) | −0.0005 (t −0.82) | +0.0004 (t +0.64) |

The **outfield-space** arm is the cleanest illustration of the confound: ctrl_none looks like a
strong deterrent (ATT −0.0035, t −5.21), but it **collapses to ≈0 once commitment is controlled**
(t −0.82). Mediation confirms it — the mediator being *where the turnover happens* (turnover-x):

| Arm | total effect | NDE (direct) | NIE (indirect, Sobel z) | prop. mediated |
|---|---|---|---|---|
| `rd_outfield_deter_space` | +0.00406 | +0.00098 (t **0.92**) | +0.00308 (z 11.7) | **0.758** |
| `rd_outfield_deter_threat` | +0.00012 | −0.00094 (t −1.95) | +0.00107 (z 9.24) | ≈0 total¹ |

**a-path** arm→turnover-x = +8.72 (t 12.96); **b-path** turnover-x→counter = +0.00035 (t 27.2).
A committed team has both a more-positive ("worse") arm **and** concedes turnovers higher up the
pitch, and turnover height drives counter-danger. **~76 % of the outfield-space arm's effect runs
through turnover position, and its direct effect is ≈0 (t 0.92).** The arm is a **commitment
proxy**, not a clean deterrent.

¹ The outfield-threat arm's raw total effect is ≈0 (a small negative direct + a positive
indirect that nearly cancel), so its `prop_mediated` is numerically degenerate; the ATT table
above is the readable summary — it flips positive under every control set that includes
commitment.

### 3.2 Keeper-threat arm — the one control-robust signal

| Cohort | ATT ctrl_none | ctrl_turnoverx | ctrl_commitment | ctrl_both |
|---|---|---|---|---|
| **Pooled** | −0.0013 (t −2.38) | −0.0014 (t −2.60) | −0.0023 (t −4.06) | −0.0018 (t −3.23) |
| SkillCorner | −0.0022 (t −2.32) | −0.0011 (t −1.50) | −0.0018 (t −2.32) | −0.0022 (t −2.62) |
| Gradient Sports | −0.0014 (t −1.91) | −0.0009 (t −1.15) | −0.0013 (t −1.87) | −0.0014 (t −1.76) |
| Full tracking (n=781) | −0.0036 (t −0.91) | −0.0023 (t −0.64) | −0.0046 (t −1.15) | −0.0070 (t −1.55) |

Pooled: **ATT < 0 significant in all four control sets** (t −2.4 to −4.1) and **not mediated**
(prop_mediated −0.03, NIE Sobel z −0.16). This is a genuine — if small (~−0.0014 to −0.0023 xT
off B's counter-danger) — keeper deterrent. **But** it is **SkillCorner-driven** (SC significant,
GS only marginal, full tracking underpowered at n=781), so it is not independently confirmed
across providers.

The keeper-**space** arm is null everywhere (pooled ATT_both −0.0010, t −1.79; no significant
cell across cohorts).

## 4. Decision

Applying the standing rule — *a metric that failed its validation must not ship as that metric*:

- **All four arm metrics are demoted to experimental.** None is a validated deterrent: the two
  outfield arms are commitment-confounded; the keeper-space arm is null; the keeper-threat arm is
  control-robust only when pooled/on SkillCorner. They are removed from the public
  `silly_kicks.restdefense` surface (glossary, `__all__`, SB360 boundary audit); the code stays
  in the private `restdefense._arms` / `_counterfactual` modules for the redesign.
- **What ships this cycle:** the ghost-GK both-axes goal-relative convention correctness fix +
  the five re-fit ghost-GK variants + the mixed-provider validation-driver fix + the gkdv
  re-materialize / TF-19 sign-off re-run against the new weights — plus this study.
- **Next cycle (registered):** a **within-turnover counterfactual counter** — price B's realized
  counter-danger under the **actual** vs a league-average **ghost** rearguard while **holding the
  turnover event fixed**. Fixing the turnover removes the commitment confound (the mediator this
  study isolates), which the possession-aggregate arm conflates. Because the mediation shows the
  outfield arm's direct effect is ≈0, the redesign may confirm the outfield deterrent is small
  and the keeper is the real lever — an outcome this study already leans toward.

## 5. Reproduce

Inputs: `arm179/layer3_arm_values.parquet` (run_commit `a867f9b`), the corpus-fitted xT
(`xt.npz`), the tc3 `_actions` SPADL cache. Harness in `harness/`:

- `harness/construct_validity_anchor.py` → `construct_findings.json` (§2; stamps the arm-values
  `run_commit` it read).
- `harness/predictive_validity.py` → `predval_findings.json` (§3; turnover extraction +
  Spearman + matched ATT via `silly_kicks.causal.matching` + natural direct/indirect mediation).

Both derive from the single clean arm table (`run_commit a867f9b`), so this study's numbers carry
that provenance.
