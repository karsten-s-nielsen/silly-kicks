# TF-60 rest-defense deterrent — counterfactual-counter redesign (design)

| Field | Value |
|---|---|
| **Date** | 2026-09-08 |
| **Status** | **Proposed** (next cycle) — registered by ADR-089's 2026-09-08 amendment |
| **Supersedes** | The Layer-3 counterfactual deterrent *arms* (demoted to experimental; `docs/research/tf60_layer3_construct_validity/`) |

## 1. Why

The Layer-3 arms priced the rearguard's counterfactual suppression at the **possession-aggregate**
grain: predictor = mean arm over team A's committed-forward possession, outcome = team B's counter
after any subsequent loss. The out-of-sample validity study
(`docs/research/tf60_layer3_construct_validity/findings.md`) showed this grain **confounds the
deterrent with attacking commitment**: a committed team both has a "worse" arm and concedes
turnovers higher up the pitch, and turnover height drives counter-danger. Controlling for
commitment kills the outfield arms' apparent signal (outfield-space matched ATT −0.0035 → −0.0005;
~76 % mediated by turnover position; direct effect t 0.92). Only the keeper-threat arm was
control-robust, and only pooled/on SkillCorner.

The confound is **structural to the grain**, not a tuning problem: the arm never conditions on the
specific loss event, so the covariance between "how committed A was" and "where/whether A lost the
ball" leaks straight into the metric.

## 2. The design — a within-turnover counterfactual counter

Condition on the **actual turnover event** and difference the opponent's realized counter under two
rearguards **holding that event fixed**:

- Anchor on each realized turnover `t` (the same live-ball, ≤5 s, team-flip definition the study
  used): the ball state, the winning-team B's positions, and A's rearguard at the moment of loss are
  all **given**.
- **Factual leg:** B's counter-danger given A's **actual** rearguard geometry at `t`.
- **Counterfactual leg:** the same, with A's rearguard replaced by the **league-average ghost**
  (`build_restdefense_ghost_frames` / `GhostOutfieldModel`, `GhostGkModel(sweeper)`) — *and nothing
  else about the turnover changed*.
- **Deterrent(t) = counter_ghost(t) − counter_actual(t)** — how much B's realized counter-danger the
  actual rearguard suppressed **at a fixed loss**, so commitment (which fixes *where* the turnover
  is) can no longer drive the difference.

Aggregate per keeper / per team over their turnovers. Because the mediator (turnover position) is
now inside the conditioning set rather than a free predictor, the estimand is the direct effect the
possession-aggregate arm could not isolate.

## 3. Reuse

- The ghost engine + serves are shipped as private infrastructure this cycle
  (`restdefense._counterfactual.build_restdefense_ghost_frames`; both models serve frame-ready
  coords). The redesign consumes them unchanged.
- The turnover extractor + the counter-danger outcome (peak next-possession xT via
  `xthreat._physical.values_at_points`) exist in the study harness
  (`docs/research/tf60_layer3_construct_validity/harness/predictive_validity.py`) and become the
  library seam.
- Validation reuses the matched-ATT / mediation harness as the acceptance gate: the redesigned metric
  must remain significant **under commitment + turnover-x controls** (the specs the arms failed), and
  its mediation `prop_mediated` must be materially below the arms' ~0.76.

## 4. Open questions (for the next cycle's spec)

- **Grain of the counterfactual outcome:** the whole next possession vs a fixed post-turnover
  horizon (e.g. the first N seconds / first shot) — the study used peak next-possession xT.
- **Keeper vs outfield split:** the study already leans "keeper is the real lever, outfield deterrent
  is small"; decide whether the outfield counter ships at all or only the keeper counter.
- **Public surface:** whether the redesigned metric ships as a VAEP feature, a per-keeper/per-team
  summary, or both — and its glossary + SB360-audit registration (deferred until it validates).
- **Provider coverage:** the keeper signal was SkillCorner-driven; a redesign gate should require
  cross-provider robustness rather than a pooled pass.

## 5. Non-goals

- Not this cycle. This document *registers* the program; the next cycle writes the full spec + plan.
- No change to the shipped Layer-1 KPIs or Layer-2 danger valuation.
