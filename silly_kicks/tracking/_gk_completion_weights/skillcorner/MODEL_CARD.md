# GK-completion model — `skillcorner` variant

**What it is.** The pass-completion probability `P(success | geometry)` that xT-GK's RAV term
consumes, fit for **SkillCorner** data. Logistic regression; sklearn at fit, pure-numpy
`sigmoid(Xβ)` at serve (no runtime sklearn). Loaded via `GkCompletionModel.from_variant("skillcorner")`,
or auto-selected by `compute_xt_gk` when `frames["source_provider"] == "skillcorner"`.

**Label construct.** SPADL `result_id == success` = *the pass reached a teammate*, sourced from
SkillCorner's **native** `pass_outcome` outcome (the converter's corrected `result_id`, ADR-024
amendment 4.21.0). Trained on the **`native` (`pass_outcome`) tier only** — the `inferred`
(`received==True`) and `stopgap` (`same_team_next` residual) tiers are excluded from training
(positive-only / proxy → would bias the multiplicatively-consumed calibration). Construct-comparable
to the GS `default` (both native completion).

**Why a distinct variant (not the GS default).** Measured on the corrected native label (10
SkillCorner matches, N=542): the GS `default` does **not** transfer to SkillCorner (GK-pass AUC
**0.412**, worse than chance) — SkillCorner's tracking-derived geometry differs. A SkillCorner-fit
model is required.

**Performance (out-of-fold, match-grouped).**
- **GK-passes: AUC 0.739, ECE 0.036** (n=461) — clears the 0.70 floor, well-calibrated. This is the
  intended domain.
- **Goal-kicks: AUC 0.433 (chance)** (n=81) — **KNOWN LIMITATION:** goal-kick completion is not
  predictable from geometry even on the native label (short-goal-kick tactics, aerial regime).
  Goal-kicks are still *model-scored* (their `xt_gk` is on-scale — see the comparability gate) but
  carry **low discrimination**; treat goal-kick `xt_gk` as ~base-rate. A per-type base-rate serve
  switch is a tracked follow-up (TODO.md).

**Pooling.** SkillCorner `xt_gk` is **within tolerance** of GS `xt_gk` on matched distance bands
(`scripts/_xtgk_comparability.py`, owner-run) → poolable directly. Do **not** pool `xt_gk` across
`xt_gk_completion_variant` values without a validated comparability check (ADR-024, H1/D-S9).

**Provenance + reproduction.** See `metrics.json` (decision, per-sub-domain AUC/ECE, sample sizes).
Trained via `python scripts/train_gk_completion.py --variant skillcorner`. Attribution: xT-GK is
Jeffrey Eyestone's (Pitch to the Pros 1), public-with-attribution — see NOTICE / ADR-024.
