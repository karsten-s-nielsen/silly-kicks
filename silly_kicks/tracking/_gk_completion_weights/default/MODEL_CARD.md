# GK-completion model — `default` variant (Gradient Sports)

**What it is.** The pass-completion probability `P(success | geometry)` that xT-GK's RAV term
consumes. Logistic regression; sklearn at fit, pure-numpy `sigmoid(Xβ)` at serve (no runtime
sklearn). Loaded via `GkCompletionModel.from_variant("default")` and auto-selected by `compute_xt_gk`
for all **native-completion** providers — Gradient Sports, Sportec/IDSSE, and `snapshot` event
sources (`variant_key_for_provider` maps everything except `skillcorner` here). SkillCorner uses its
own variant (GS does not transfer; see that variant's card).

**Label construct.** SPADL `result_id == success` = *the pass reached a teammate*, from the provider's
native completion outcome (Gradient Sports event `result`; Sportec/IDSSE via the kloppy gateway's DFL
`play_evaluation`).

**Training corpus + gate.** 64 WC2022 Gradient Sports matches (the full pining manifest).
**Green gate = native-origin pooled out-of-fold calibration:** AUC **0.855**, CI95 [0.838, 0.872],
n_native 2953 of 3491 rows, Brier 0.119 < base 0.176; density-feature finite 99%. See `metrics.json`.

**Retrained 4.73.0 (PR-S141, ADR-051).** These weights replace a set fitted against pre-ADR-028
geometry. RC2/RC5 (4.71.0) corrected `_gk_geometry`'s frame-coordinate reprojection and the
cross-team next-event borrow, both of which reach these features through
`prepare_gk_completion_training_data → resolve_gk_geometry`. The same run widened the corpus from 30
matches (n_rows 1666, AUC 0.838) to 64, so **the change is geometry AND corpus, not geometry alone** —
the two are not separable after the fact. Superseded coefficients are recorded in `metrics.json`.

**Per-type serve gate (4.21.4, SK-91).** A per-type `_type_serve_mode` (artifact version 1.1.0) is
baked into `model.json`: serve the model where the held-out AUC LCB > 0.5, else the calibrated
`base_rate`. For GS, **goal-kicks stay `model`-scored** (held-out AUC **0.835**, LCB 0.809, n=936 — unlike
SkillCorner, GS goal-kick completion *is* predictable from geometry); GK-passes `model` (AUC 0.855,
LCB 0.836, n=2553); the near-empty throw-in sub-domain (n=2) base-rates by construction. `load()` fail-opens (a pre-gate
artifact serves all types `model`).

**Missing-value policy.** Per-feature density NaN → training-mean impute (neutral after
standardization). Whole-row geometry-unscoreable → per-type base rate (in the standalone
`compute_gk_completion`; the RAV path NaNs unresolvable-destination rows honestly).

**Provenance + reproduction.** Reproduce with `python scripts/train_gk_completion.py --providers
gradientsports --max-per-provider 64 --mode retrain --feature-space moved --probe-old <matrix>.parquet
--reason "<what moved>"`. `metrics.json` records `run_commit` and `run_tree_state` (this bundle was
produced from a clean tree). The `--probe-old` matrix is the SAME corpus extracted at the commit just
BEFORE the change under test — it is compared element-wise, so it must be row-aligned; see the ADR-052
amendment. The script defaults to **full-match frames**
(`--tracking-limit None`, SK-91). A small `--tracking-limit` does **not** reproduce these weights —
GS matches are ~162 k frames and the density feature needs the full match. (The 4.21.4 re-bundle
attaches the per-type gate onto the committed coefficients within a tolerance that absorbs the
unrecorded original `tracking_limit`'s density float noise, without changing them.) Pickle-free JSON
\+ SHA256 envelope (`model.json` + `SHA256SUMS`). Caller `completion=` override supported.
Attribution: xT-GK is Jeffrey Eyestone's (Pitch to the Pros 1), public-with-attribution — see
NOTICE / ADR-024.
