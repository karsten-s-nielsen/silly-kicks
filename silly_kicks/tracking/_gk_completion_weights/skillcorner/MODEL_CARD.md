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
- **Goal-kicks: AUC 0.433 (chance)** (n=81) — goal-kick completion is not predictable from geometry
  even on the native label (short-goal-kick tactics, aerial regime). **As of 4.21.4 (SK-91) the
  per-type serve gate routes SkillCorner goal-kicks to the calibrated `base_rate`** (tagged
  `xt_gk_completion_source = "base_rate"`); the gate (`_type_serve_mode`, artifact version 1.1.0)
  is baked into `model.json` and serves the model only where the held-out AUC LCB > 0.5 (GK-passes:
  AUC 0.737, LCB 0.674 → `model`). Throw-ins (degenerate, n≈2) base-rate by construction.

**Pooling.** SkillCorner `xt_gk` is **within tolerance** of GS `xt_gk` on matched distance bands
(`scripts/_xtgk_comparability.py`, owner-run) → poolable directly. Do **not** pool `xt_gk` across
`xt_gk_completion_variant` values without a validated comparability check (ADR-024, H1/D-S9).

**Provenance + reproduction.** See `metrics.json` (decision, per-sub-domain AUC/ECE, sample sizes).
Reproduce with `python scripts/train_gk_completion.py --variant skillcorner` — the script now defaults
to **full-match frames** (`--tracking-limit None`, SK-91). A small `--tracking-limit` does **not**
reproduce these weights: ~20 s of frames starves the SkillCorner derived-GK, which then over-flags
goalkeepers in some matches and inflates the frame-derived GK-pass domain (full frames reproduce
N=542 exactly). The 4.21.4 re-bundle attaches the per-type gate onto the committed coefficients
without changing them. Attribution: xT-GK is Jeffrey Eyestone's (Pitch to the Pros 1),
public-with-attribution — see NOTICE / ADR-024.
