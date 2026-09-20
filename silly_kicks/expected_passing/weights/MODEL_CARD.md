# Pass-completion model -- `default` variant (TF-54b)

**What it is.** An event-only pass-completion probability `P(complete | origin -> target geometry)`,
shipped as a reusable seam. Logistic regression; sklearn at fit, pure-numpy `sigmoid(Xb)` at serve
(no runtime sklearn). Loaded via `silly_kicks.expected_passing.PassCompletionModel.bundled()`. Carried
from the unmerged `ab9001c` alongside `xthreat.destination_profiles`; the event-only "threat prevented"
counterfactual cone it was built to weight was not carried into the TF-54b cycle, so no shipped feature
consumes it yet -- it is library infrastructure for a future consumer.

**Label construct.** SPADL `result_id == success` = the pass reached a teammate. Completed passes are
labelled at their real end; failed passes at their SPADL death/recovery location (the field-standard
expected-passing label).

**Features (event-only, 10).** distance, angle-to-goal, forward and lateral components, origin/target
x and y, origin/target pitch-third. No tracking, no teammate positions. A consumer scoring a FAILED
pass evaluates the model at a HYPOTHESISED target (e.g. an xT-grid destination), within the geometry
range completed passes already cover.

**Training corpus + metrics.** 3961 match(es) from `statsbomb-open (all 80 open-data competitions)` (3337914 finite-coordinate
pass rows). This bundle was fit on the full public open-data corpus (80 (competition, season) releases in the StatsBomb open-data manifest). GroupKFold-by-match out-of-fold: AUC 0.794, ECE 0.006, Brier
0.129 vs base rate 0.799 (Brier skill score 0.194 vs the base-rate baseline). See `metrics.json`.

**Missing-value policy.** A non-finite coordinate yields an all-NaN feature row and a NaN probability
(never a fabricated value); a consumer drops-and-counts such a target.

**Provenance + reproduction.** Reproduce with `python scripts/train_pass_completion.py --out <DIR>
--all-competitions`. `metrics.json` records `training_commit` (6c75619eabd17935718245dcc02f092e2f450d42) and the tree state (this
bundle was produced from a clean tree). Pickle-free JSON + SHA256 envelope (`model.json` +
`SHA256SUMS`) with a feature contract + chirality probe; `load()` is fail-closed
(ADR-011/016/040/044/050). Every bundled model carries a card (ADR-088). Attribution:
expected-passing / pass-completion modelling -- see NOTICE.
