# xT-GK v2 construct-validity with the real ρ (SP5 sign-off) — Design

**Status:** DRAFT (for review). NOT committed by this session.
**Date:** 2026-07-11 · **For:** silly-kicks session (implementation).
**Origin:** the PR-S111 deferred SP5 follow-up — wire a runnable real-ρ path into `validate_xtgk_v2.py`
and run it locally so the retrained production ρ's metric can be certified against baselines. Governing:
ADR-036 (xT-GK v2, §Part 5 validation), ADR-009 (auditable-manifest reporting).

## 0. Context — what exists, what's missing

`scripts/validate_xtgk_v2.py::construct_validity_scores` fits V on a possession-parity train split, computes
`compute_xt_gk_v2` on the test split, and reports **AUC lift** of v2 over baselines (the informative quantity —
V is by construction ~expected first-shot xG, so absolute AUC vs a possession→shot target is partly circular).
Today it hardwires a **constant-ρ stub** (`_ConstRho`, ρ≡0.75) + an **empty** retention-feature frame — the CI
smoke path. There is **no runnable entry point** that loads a real cohort and injects the bundled ρ.

Two facts settle the design:
- **`compute_xt_gk_v2` does NOT gate to the GK-distribution domain** — it computes the metric for every action
  passed in (`_metric.py:66-82`; the code comment notes it's "fine for the GK-distribution slice"). Since the
  bundled ρ is trained **only on the GK-distribution domain**, evaluating it on all actions is OOD → an unfair
  test. **The harness must restrict evaluation to the GK-distribution domain.**
- `load_xtgk_cohort` already returns actions with `xg`, `possession_id`, `pressure_on_actor__bekkers_pi`,
  `start/end x/y`, `type_id`, `result_id` — everything the harness needs **except** a GK-distribution marker
  (`is_gk_distribution`) and the **stored v1 composite** (`xt_gk`) — both live on `fct_action_context` (F1 +
  v1), each one additive column away.

## 1. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Keep v1 as a `v1_stored` baseline (read, don't recompute)** | Include v1 via the **stored `fct_action_context.xt_gk`** column — the strongest, most decision-relevant baseline. Drop the OLD frames-recompute path (`_v1_composite` + its `frames`/`xt`/`home_team_id` params), NOT the baseline. | **Verified live:** v1's composite is materialized marts-native, covering **GS 3457/3873 (89.3%) / SC 5487/5487 (100%)** of the GK-distribution domain — one additive column, **no frames**. The earlier "needs frames" rationale conflated *recomputing* v1 with *reading its stored output*. v1 (an xT-derived metric) is plausibly the strongest baseline, so it MUST be in the lift's `max(...)` — omitting it would flatter v2 (integrity). The construct-validity report is *the* place to show v2 beats the metric it replaces; "v1 is retiring" argues *for* the last-chance comparison. |
| **Evaluation domain** | The **GK-distribution domain** (`is_gk_distribution`) — fit V on the full train stream, evaluate v2 on the GK-distribution **test** subset. | ρ is GK-domain-trained; `compute_xt_gk_v2` doesn't self-gate. Matches the training domain (goal-kicks + acting-GK passes), maximizing in-domain signal. |
| **Domain source** | Add `c.is_gk_distribution` to `load_xtgk_cohort`'s SELECT (additive; the deep-zone gate ignores it). | The marts have it (F1); one additive column beats a duplicate loader or the narrower frames-free goalkicks-only `gk_distribution_mask`. |
| **Reuse the gate prep** | `load_xtgk_cohort` → `prepare_cohort` (pressure=`bekkers_pi`, xg=`xg`), then restrict test to `is_gk_distribution`. | Single-sources the cohort prep with the deep-zone gate. |
| **Run scope** | Per-provider: **gradientsports + skillcorner** (both now have ρ variants), each with its own `from_variant(variant_key_for_provider(...))`. | Both variants shipped in PR-S111; validate both. |
| **Output** | Print the AUC table + lift, AND commit a report under `docs/research/xtgk_v2_construct_validity/`. | Auditable, mirrors the deep-zone gate's `GATE_FINDINGS.md` (ADR-009). |
| **Version** | Patch bump **4.44.0 → 4.44.1**. | Scripts + tests only (+ one additive loader column); no wheel/library behaviour change (the `4.21.2` scripts-only-bump precedent). |

## 2. Architecture

```
scripts/validate_xtgk_v2.py
  construct_validity_scores(actions, *, xg_column, pressure_column, retention=None)   [PARAMETERIZED]
     - fit V on the FULL train (possession-parity) split
     - restrict test -> GK-distribution subset (actions["is_gk_distribution"])
     - y = possession_reaches_shot(FULL test) then indexed to the GK subset (forward-scan needs full seq)
     - retention=None -> _ConstRho() + empty feats (CI smoke, unchanged)
       retention=<model> -> feats = extract_retention_features(gk_test, pressure_column=...)
     - AUC lift of xt_gk_v2 over {raw_completion, destination_xt, v1_stored}
  main()  [NEW runnable]
     - argparse --provider {gradientsports, skillcorner}
     - load_xtgk_cohort -> prepare_cohort -> construct_validity_scores(retention=from_variant(...))
     - print AUC table + lift; write docs/research/xtgk_v2_construct_validity/<provider>.md

scripts/_loader_databricks.py
  load_xtgk_cohort: SELECT ... , c.is_gk_distribution, c.xt_gk   [two additive columns]
```

## 3. `construct_validity_scores` changes

- **Signature:** `construct_validity_scores(actions, *, xg_column, pressure_column, retention: RetentionModel |
  None = None) -> dict`. Drop `frames`. (CI smoke calls it with no `retention` → stub, unchanged.)
- **Domain restriction:** after the train/test split and V fit, compute `y_full =
  _possession_reaches_shot(test)` on the **full** test (the forward scan needs the intact possession
  sequence), then `gk = test["is_gk_distribution"].fillna(False).to_numpy()`, `test_gk = test[gk]`,
  `y = y_full[gk]`. If `is_gk_distribution` is absent (synthetic CI fixtures), fall back to all-test (the CI
  smoke has no such column and doesn't need the restriction — the stub is domain-agnostic).
- **Real features:** `feats = extract_retention_features(test_gk, pressure_column=pressure_column)` when a real
  `retention` is passed (its `predict_proba(feats)` consumes the 8 `RETENTION_FEATURE_NAMES`); else the empty
  frame for `_ConstRho`.
- **Baselines (on `test_gk`):** `raw_completion` (`result_id == success`), `destination_xt`
  (`_destination_only_v`), and **`v1_stored`** = the stored `test_gk["xt_gk"]` column (v1's composite),
  scored on its **non-null** rows only. **Drop the OLD frames-recompute** `_v1_composite` helper + the
  `frames`/`xt`/`home_team_id` params (v1 is now *read*, not recomputed).
- **Output:** `{"xt_gk_v2": {auc}, "raw_completion": {auc}, "destination_xt": {auc}, "v1_stored": {auc, n},
  "lift": v2_auc − max(raw, dest, v1_stored), "n_test_gk": …, "_note": …}`. The lift's `max(...)` **includes
  v1_stored** (fair comparison — v1 is plausibly the strongest baseline). Report `n_v1` (v1 coverage:
  ~89% GS / 100% SC) alongside `n_test_gk`; where v1 is null it's simply out of its own denominator (report,
  don't hide). Note updated: frames-free (v1 read from `c.xt_gk`), GK-domain-restricted, **V out-of-sample /
  ρ in-sample (production model)** + V∝first-shot-xG circularity; drop the WC2018/Neuer TODO.

## 4. `main()` (runnable owner-run)

- argparse `--provider` (choices from `_ALLOWED_PROVIDERS`; default `gradientsports`).
- `raw, _ = load_xtgk_cohort(provider)` → `actions = prepare_cohort(raw, pressure_column=bekkers_pi,
  frame_present_column=…)` (reuse the gate's constants/prep).
- `variant = variant_key_for_provider(provider)`; `rho = GkRetentionModel.from_variant(variant)`.
- `scores = construct_validity_scores(actions, xg_column="xg",
  pressure_column="pressure_on_actor__bekkers_pi", retention=rho)`.
- Print the AUC table + lift; write `docs/research/xtgk_v2_construct_validity/<provider>.md` (provider, n,
  variant, AUC table, lift, the split/circularity note).
- **I run it locally** (Databricks) for GS + SC and commit both reports with the real numbers.

## 5. `load_xtgk_cohort` change

Add `c.is_gk_distribution` **and `c.xt_gk`** to `_XTGK_ACTIONS_SQL`'s SELECT list (the `fct_action_context c`
join already exists). Coerce `is_gk_distribution` (`astype("boolean").fillna(False).astype(bool)` — warning-free
form) and `xt_gk` (`pd.to_numeric(..., errors="coerce")`, keeping NaN where v1 is null — it's a baseline, not
a domain gate). The deep-zone gate ignores both extra columns (additive, no behaviour change); note them in the
loader docstring.

## 6. Testing

- **CI smoke (unchanged):** `test_validate_v2_smoke.py` still calls `construct_validity_scores` with no
  `retention` → stub path → all baselines present + finite v2 AUC. (v1_composite assertion removed.)
- **New CI test — real-features branch (NON-VACUOUS fixture):** a fixture that contains **BOTH
  `is_gk_distribution` True AND False rows** (so the restriction is real, not a no-op/vacuous) + an `xt_gk`
  column (some non-null, ≥1 null to exercise the v1-coverage path) + a **fake real** retention model
  (`class _Fake: def predict_proba(self, f): return np.full(len(f), 0.6)`, exercised via `retention=` so
  `extract_retention_features` runs on the GK subset). Assert: `0 < n_test_gk < n_test` (restriction applied),
  features built (no crash on the 8-column extract), v2 + `v1_stored` AUCs finite, `v1_stored["n"]` < n_test_gk
  (the null rows dropped from v1's denominator), lift present and computed over `max(raw, dest, v1_stored)`.
- **Domain-fallback test:** `construct_validity_scores` on a fixture WITHOUT `is_gk_distribution` (the CI
  smoke shape) → falls back to all-test, still returns the baselines (guards the absent-column branch).
- Full non-e2e suite green; ruff clean; **bare pyright 0**.

## 7. Deferred / not in scope

- **Recomputing v1 from frames** is NOT done (the old `_v1_composite` path) — v1 is *read* from the stored
  `c.xt_gk` column instead. (v1 IS in the report as the `v1_stored` baseline; only the frames-recompute is out.)
- **Batch/vectorized `compute_xt_gk_v2`** (the per-action Python loop is fine for the GK-distribution slice;
  `_metric.py:63` already flags a batch path as a separate follow-up if the lakehouse needs full-stream).
- **Lakehouse xt_gk_v2 re-materialization** on the 4.44.0 pin (their action, relayed).

## 8. Release

Patch bump **4.44.0 → 4.44.1** + tag. Standard lockstep (pyproject / `__init__` / CHANGELOG / TODO / uv.lock).
CHANGELOG records the construct-validity result (per-provider lift). ADR-036 amendment (SP5 real-ρ
construct-validity wired + run; v1_composite dropped; result summary). `/final-review` (C4 count stays 28 — a
script harness + a report + an additive loader column are not architectural). The committed
`docs/research/xtgk_v2_construct_validity/` reports carry the real numbers.

## 9. Open items for the implementation session

1. **The lift result is not pre-decidable** — if v2 does NOT beat the baselines out-of-sample, that's a
   reportable finding (surface it; do not massage). The harness reports honestly; a null/negative lift is a
   real datapoint about the metric, not a bug to fix.
2. **`prepare_cohort` passes both new columns through** — `is_gk_distribution` (bool) AND `xt_gk` (float, NaN
   preserved). `prepare_cohort` operates on pressure/frame_present; extra columns should ride through untouched
   — verify (hard gate). Do NOT let it drop `xt_gk`'s NaNs as if they were unscoreable rows (they're just out
   of v1's baseline denominator).
3. **`extract_retention_features` on `test_gk`** — the pressure column is `pressure_on_actor__bekkers_pi`
   (passed as `pressure_column`); confirm the extractor reads it (it reads `actions[pressure_column]`).
