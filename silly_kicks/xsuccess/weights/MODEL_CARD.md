# xSuccess model -- `default` variant (TF-61)

**What it is.** An event-only, END-BLIND action-completion probability `P(success | pre-action
context)` over ALL on-ball action types, from Paul/Klemp/Memmert 2025 ("Beyond Outcome Bias").
Calibrated XGBoost; xgboost + isotonic at fit, xgboost + pure-numpy isotonic at serve (no runtime
sklearn). Loaded via `silly_kicks.xsuccess.XSuccessModel.bundled()`. Consumed by
`VAEP.rate_adjusted` to remove outcome bias.

**END-BLIND (correctness, not a caveat).** Features use only start-anchored geometry + type +
bodypart + time -- NEVER the realized action end. For a failed action the SPADL end IS the outcome
(the interception/death point), so an end-using completion model would be a postdiction that
re-injects the very bias VAEP_adjusted removes. Guarded by an end-location invariance test.

**Label construct.** SPADL `result_id == success`. All on-ball action types retained (some are
structurally one-sided, e.g. clearance/bad_touch; the model learns their base rate).

**Training corpus + metrics.** 3961 match(es) from `statsbomb-open (comp 2, season 27), (comp 2, season 44), (comp 7, season 27), (comp 7, season 108), (comp 7, season 235), (comp 9, season 27), (comp 9, season 281), (comp 11, season 1), (comp 11, season 2), (comp 11, season 4), (comp 11, season 21), (comp 11, season 22), (comp 11, season 23), (comp 11, season 24), (comp 11, season 25), (comp 11, season 26), (comp 11, season 27), (comp 11, season 37), (comp 11, season 38), (comp 11, season 39), (comp 11, season 40), (comp 11, season 41), (comp 11, season 42), (comp 11, season 90), (comp 11, season 278), (comp 12, season 27), (comp 12, season 86), (comp 16, season 1), (comp 16, season 2), (comp 16, season 4), (comp 16, season 21), (comp 16, season 22), (comp 16, season 23), (comp 16, season 24), (comp 16, season 25), (comp 16, season 26), (comp 16, season 27), (comp 16, season 37), (comp 16, season 39), (comp 16, season 41), (comp 16, season 44), (comp 16, season 71), (comp 16, season 76), (comp 16, season 276), (comp 16, season 277), (comp 35, season 75), (comp 37, season 4), (comp 37, season 42), (comp 37, season 90), (comp 37, season 281), (comp 43, season 3), (comp 43, season 51), (comp 43, season 54), (comp 43, season 55), (comp 43, season 106), (comp 43, season 269), (comp 43, season 270), (comp 43, season 272), (comp 44, season 107), (comp 49, season 3), (comp 49, season 107), (comp 53, season 106), (comp 53, season 315), (comp 55, season 43), (comp 55, season 282), (comp 72, season 30), (comp 72, season 107), (comp 81, season 48), (comp 81, season 275), (comp 87, season 84), (comp 87, season 268), (comp 87, season 279), (comp 116, season 68), (comp 131, season 281), (comp 135, season 281), (comp 182, season 281), (comp 223, season 282), (comp 1238, season 108), (comp 1267, season 107), (comp 1470, season 274)` (7974436 on-ball rows).
GroupKFold-by-match out-of-fold: AUC 0.895, Brier 0.084 vs base rate 0.835. Per-type
reliability + calibration-in-the-large in `metrics.json`.

**Missing-value policy.** A non-finite feature yields a NaN probability (never fabricated).

**Provenance.** `metrics.json` records `training_commit` (12e5677e7301a7201c6c5c18160ef6a655bb5f23) and the tree state (clean for the
bundle). Pickle-free booster-JSON + metadata + SHA256 envelope with a feature contract + chirality
probe; `load()` is fail-closed (ADR-011/016/040/050). Every bundled model carries a card (ADR-088).
Attribution: Paul/Klemp/Memmert 2025; XGBoost (Chen & Guestrin 2016) -- see NOTICE.
