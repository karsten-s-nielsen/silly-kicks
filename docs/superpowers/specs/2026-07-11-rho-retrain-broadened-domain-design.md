# ρ retrain on the broadened `is_gk_distribution` domain + loader collapse + GK-resolver dtype hardening — Design

**Status:** DRAFT (for review). NOT committed by this session.
**Date:** 2026-07-11 · **For:** silly-kicks session (implementation).
**Origin:** the two PR-S110 deferred follow-ups (both blocked on the lakehouse materializing
`fct_action_context.is_gk_distribution` — now **live**) + one PR-S110-deferred hardening the owner asked to
fold in. Governing: ADR-036 (xT-GK v2), ADR-024 precedent (trained-light ρ class), ADR-019 (dtype-safe ids),
ADR-009 (calibration/gate discipline).

## 0. Context — why now, and the live-data grounding

PR-S110 shipped the public `gk_distribution_mask` and moved the ρ retention loader/trainer onto a
self-adapting `is_gk_distribution` domain, deferring two items until the lakehouse materialized the column.
**F1 is done** — a read-only probe of `soccer_analytics.dev_gold` (2026-07-11) confirms:

- `fct_action_context.is_gk_distribution` is **live** (BOOLEAN, materialized **nullable** — see §6; the
  loader's `COALESCE(is_gk_distribution, FALSE)` already defends against the NULLs).
- The broadened domain (goal-kicks → full GK-distribution) per provider:

  | Provider | Matches | Goal-kicks | Full domain | GK-pass broadening | Growth |
  |---|---|---|---|---|---|
  | gradientsports | 64 | 1,001 | 3,873 | 2,885 | 3.9× |
  | skillcorner | 108 | 1,189 | 5,487 | 4,306 | 4.6× |

The 4–5× growth — and crucially the inclusion of GK **open-play passes** (real geometric variation vs the
narrow fixed-spot goal-kick) — is what makes a **SkillCorner variant** worth re-attempting (it was near-chance
AUC 0.54 / calibration-failing on 1,189 goal-kicks-only).

This PR bundles three coupled changes (all in the ρ / GK-distribution neighborhood):
- **Part A** — retrain the ρ `default` + re-attempt the SkillCorner variant on the broadened domain.
- **Part B** — collapse the transitional self-adapting loader probe (the column is permanently live).
- **Part C** — the PR-S110-deferred `acting_gk_from_frames` team-join dtype hardening.

## 1. Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Scope** | One PR, three coupled parts (A retrain, B loader-collapse, C dtype hardening). | B retires A's own transitional scaffolding (National Park); C is a one-line hardening in the same resolver family the domain relies on. |
| **SkillCorner variant** | Re-attempt; **bundle only if it passes the SAME calibration gate** (`ece≤0.10 AND \|slope−1\|≤0.25`). No lowered bar. | Data-driven (5,487 rows now). If it still fails, keep `_PROVIDER_VARIANT={}` + document — exactly the PR-S109 discipline. |
| **Compute** | Local. Load cohort read-only from Databricks, fit locally. | ρ is a logistic on ~4–5k rows; DGX is tuning-only (not needed). |
| **Public API** | **No signature change.** Library changes = re-bundled `default` weights + (conditionally) a new `skillcorner` variant dir + `_PROVIDER_VARIANT` map update + the `_gk_resolve.py` one-line hardening. | Additive weights + registry + a dtype-safe internal compare. |
| **Version** | Minor bump **4.43.0 → 4.44.0**. | New `default` weights change `compute_xt_gk_v2` serve output → **xT-GK v2 retrain trigger** (opt-in; not in any default xfn list → NOT a forced VAEP retrain). Hyrum-flag the lakehouse. |

## 2. Part A — ρ retrain on the broadened domain

- **No domain code change needed.** `load_retention_cohort` already reads `is_gk_distribution`;
  `prepare_retention_training_data` already ORs it into the domain (`goal-kicks ∪ COALESCE(is_gk_distribution,
  FALSE)`). The retrain is: run `scripts/train_gk_retention.py --variant default --data-source gradientsports`
  (and `--variant skillcorner --data-source skillcorner`) against the live marts, per-variant fit + OOF CV +
  calibration gate, re-bundle the passing variants under `silly_kicks/xtgk/_retention_weights/`.
- **`default` (GS, 3,873 rows):** expected to pass (it passed on 1,001 goal-kicks-only at AUC 0.776 / ECE
  0.090 / slope 1.01; more rows + richer geometry should hold or improve). Re-bundle `default/model.json` +
  `SHA256SUMS`. **If it regresses below the gate, STOP and report** — do not bundle a failing default.
- **`skillcorner` (5,487 rows):** fit + gate. **Bundle iff it passes.** If it passes → new
  `_retention_weights/skillcorner/` dir + set `_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}` in
  `silly_kicks/xtgk/_retention.py` (so `variant_key_for_provider("skillcorner")` routes to it). If it fails →
  leave `_PROVIDER_VARIANT={}` (SC falls back to `default`), record the failing metrics in the CHANGELOG/ADR.
- **Metrics recorded** per variant (AUC, ECE, reliability_slope, n, gate pass/fail) in the CHANGELOG + the ADR
  amendment — the auditable manifest (ADR-009).
- **Serve output changes** for `compute_xt_gk_v2` (new `default` ρ) → lakehouse re-materializes its xt_gk_v2
  columns when it re-pins. Hyrum-flagged.

### 2.1 Gate-scope clarification (coordination — what this PR does and does NOT move)

Retraining ρ does **NOT** move the make-or-break **deep-zone gate**. `run_deep_zone_gate` reads **V's**
per-cell surfaces + support (`_occupied` counts extended *move* actions) — **not ρ, not
`is_gk_distribution`**. The GO-leaning deep-zone result (WC2022 relative 0.86 / RM 1.05, 4.42.0) is settled by
V and is unchanged here. What the new production ρ changes is the **metric-level construct-validity**
(`compute_xt_gk_v2` lift over baselines) — owner-run with the real ρ (CI uses a `_ConstRho` stub). So
"closing the gate" in this context means **re-running the owner-run metric / construct-validity with the
production ρ**, NOT re-running the deep-zone diagnostic — a reader must not infer ρ is what flips
GO-leaning → GO (it isn't). **If** the deep-zone gate is ever re-run on the broadened cohort, the Q4
pre-registered numbers stay **LOCKED** (`effect_floor 0.005 / relative 0.25 / n_min 30 / decreasing`) — a
retrain is never a licence to re-tune the gate. **Owner follow-up (post-bundle, tie to §6):** re-run the
real-ρ construct-validity (lift vs raw-completion / destination-V / v1 baselines) before xT-GK v2 is called
done.

## 3. Part B — collapse the transitional loader probe

- Remove `should_select_is_gk_distribution` + `_build_retention_sql` + `_IS_GK_DISTRIBUTION_PROBE` from
  `scripts/_loader_databricks.py`. Replace `_RETENTION_SQL_TEMPLATE` with a single `_RETENTION_SQL` constant
  that unconditionally `SELECT`s `c.is_gk_distribution` (the column is permanently live).
- `load_retention_cohort` drops the probe round-trip; keeps the defensive
  `df["is_gk_distribution"].fillna(False).astype(bool)` (the column is nullable — §6).
- **Intentional hard-coupling + fail-loud (m4):** the unconditional read makes `is_gk_distribution` a HARD
  dependency (correct — it's permanently materialized as of F1). A missing column surfaces as a
  self-identifying Databricks `[UNRESOLVED_COLUMN]/[COLUMN_NOT_FOUND] ... is_gk_distribution` error (names
  the column — not a cryptic fault). Document the hard dependency in the loader docstring (silly-kicks ≥4.44.0
  requires the F1 column); no bespoke pre-check (re-adding a probe defeats Part B, and the SQL error already
  identifies the column).
- Update `tests/xtgk/test_retention_loader_domain.py`: the `should_select_*` / `_build_retention_sql` unit
  tests are **removed** (the functions are gone); the trainer-domain tests (present/NULL/absent behavior via
  `prepare_retention_training_data`) **stay** (that logic is unchanged). Add a guard that the retention SQL
  now contains `c.is_gk_distribution` unconditionally and no longer references the probe helpers.

## 4. Part C — `acting_gk_from_frames` team-join dtype hardening

- **`silly_kicks/tracking/_gk_resolve.py:165`** — replace `match_team = gk_in_frame["gk_team_id"] ==
  gk_in_frame["team_id"]` with `match_team = ids_equal(gk_in_frame["gk_team_id"],
  gk_in_frame["team_id"]).to_numpy()` (`ids_equal` already imported). Line 166 (`match_team if same_team else
  ~match_team`) is unchanged — `~` on the non-nullable bool array preserves the opposing-team NA semantics
  byte-for-byte (`ids_equal` NA→False, `~`→True, identical to the raw path's `~(NA→False)`).
- **Byte-identity is the hard requirement.** `_gk_from_frames_linked` is the SHARED body for BOTH
  `acting_gk_from_frames` (same_team=True) AND `defending_gk_from_frames` (same_team=False, "defending
  byte-identical" per CLAUDE.md). On matched/same-kind dtypes with no NA, `ids_equal` fast-paths to `(==) &
  notna & notna` ≡ the raw `==` result → the existing green gates
  (`test_invariant_gk_resolve.py`, `test_gk_resolve.py`, `test_acting_gk_from_frames.py`,
  `test_gk_fallback_integration.py`) MUST stay byte-identical.
- **The fix changes behavior ONLY on:** (a) mismatched action-team vs frame-team dtypes (was silently no-match
  → now resolves), and (b) a nullable-`Int64` NA team in the join (raw `==` yields `<NA>` → pandas boolean
  masking is fragile; `ids_equal` yields deterministic False). Both are strict correctness improvements.
- **Positional vs index-aligned (m2):** `ids_equal` is POSITIONAL (RangeIndex-returning); raw `==` is
  index-aligned. They coincide here ONLY because `gk_in_frame` is a fresh inner-`merge` result (default
  `RangeIndex`). The `.to_numpy()` collapses to a plain positional bool array that masks `gk_in_frame`
  positionally — robust to that assumption. Add a one-line code comment at the site so a future upstream
  reindex of `gk_in_frame` can't silently break the byte-identity guarantee.

## 5. Testing

- **Part A (retrain):** the retrain is owner-run (`train_gk_retention.py __main__`, `@slow`/not-CI). CI keeps
  the existing pure `prepare/gate/CV` unit tests. Add: a test that `GkRetentionModel.from_variant("default")`
  loads + serves in [0,1] (already exists — keep green after re-bundle); if SC is bundled, an analogous
  `from_variant("skillcorner")` load test + a `variant_key_for_provider("skillcorner") == "skillcorner"`
  assertion (else assert it falls back to `"gs"`/default and `_PROVIDER_VARIANT=={}`). The
  `SHA256SUMS`-integrity + tamper tests already cover the re-bundled artifacts.
- **Part A — CI-enforce the calibration gate on the bundled metrics (the "no lowered bar" invariant).** The
  integrity/tamper tests verify the bundle isn't *altered*, NOT that the model *meets the bar* — so today
  "bundle only if it passes" rests on owner discipline, and a future re-bundle of a regressed model would
  sail through. New CI test (`tests/xtgk/test_retention_bundle_calibration.py`): enumerate **every present**
  `_retention_weights/*/` dir, read its `metrics.json`, and assert `ece ≤ ece_max AND abs(reliability_slope −
  1) ≤ slope_tol AND auc ≥ 0.5` — where the bar is the **canonical `_ECE_MAX`/`_SLOPE_TOL` imported
  from `scripts.train_gk_retention`** (0.10 / 0.25), NOT the recorded fields. **Defense-in-depth (nicety):**
  ALSO assert `metrics["ece_max"] == _ECE_MAX AND metrics["slope_tol"] == _SLOPE_TOL` so a hand-loosened
  `metrics.json` (e.g. edit `ece_max`→0.5 to self-certify a bad model) can't pass; accidental drift is already
  impossible (the trainer writes them straight from those constants), so this guards manual tampering only.
  The trainer already persists `metrics.json` with exactly these keys (lines 107-110) — no new persistence,
  just enforcement. CI can't re-run the fit, but it *can* certify the metrics you bundled clear the bar. This is
  the CI-side complement to the owner-run fit and guards all future re-bundles.
- **Part B (loader):** SQL-shape guard (unconditional `c.is_gk_distribution`, no probe helpers); the
  present/NULL/absent domain tests stay green (trainer logic unchanged).
- **Part C (hardening):** (i) **byte-identity — NON-VACUOUS.** A matched-dtype fixture alone would pass
  without ever exercising the `~`/NaN path that IS the hardening — the recurring vacuous-fixture trap. The
  fixture MUST include a **float-NaN team row on the defending (opposing) path** (and a normal matched row),
  and assert BOTH `acting_gk_from_frames` AND `defending_gk_from_frames` outputs are byte-identical pre/post
  (the `~match_team` opposing branch on a NaN team is the exact case that proves the shared
  `defending_gk_from_frames` body is untouched); (ii) **mismatched-dtype now resolves** — string action
  `team_id` vs Int64 frame `team_id`, assert the acting GK is now correctly resolved (was NaN); (iii)
  **nullable-`Int64` NA team → deterministic not-matched** (raw `==` yields `<NA>` → pandas boolean-masking
  raises; `ids_equal` → deterministic False, no raise).
- Full non-e2e suite green; ruff clean; **bare pyright 0 errors** (whole repo, per
  [[feedback_pyright_full_package_scope]]).

## 6. Deferred / heads-up (not this PR)

- **Nullable column heads-up to the lakehouse:** F1's plan intended `is_gk_distribution` **non-nullable**; it
  shipped **nullable** (899 GS / 557 SC NULLs — LEFT-JOIN misses + gaps). silly-kicks is defended
  (`COALESCE`/`fillna(False)`), so this is **non-blocking** — but relay it so they can decide whether to
  enforce non-nullable (their earlier SB360 gap fix would make it fully populated on the tracking arms).
- **Event-only "half (b)"** (acting-GK passes on statsbomb/wyscout via a lineup-GK join) stays deferred at
  both layers (D3) — event providers get goal-kicks-only via `frames=None`.
- **Owner re-run of the real-ρ construct-validity (m3, ties to §2.1):** after the new `default` is bundled,
  the owner re-runs `scripts/validate_xtgk_v2.py` construct-validity with the production ρ (CI uses the
  `_ConstRho` stub) — confirm the metric's lift over the raw-completion / destination-V / v1 baselines holds
  before xT-GK v2 is declared done. This is validation, not a gate re-tune (Q4 stays locked).
- **`acting_gk_from_frames` broader dtype audit:** Part C fixes the one team-join seam; no evidence of others,
  but the ADR-019 boundary lint / NaN-safety gates remain the backstop.

## 7. Release

Minor bump **4.43.0 → 4.44.0** + tag. Standard lockstep (pyproject / `__init__` / CHANGELOG / TODO /
uv.lock). ADR-036 amendment (records the retrain metrics manifest, the SC bundle/no-bundle decision, Part B
loader collapse, Part C hardening + its byte-identity guarantee). `/final-review` (C4 count stays 28 — no new
aggregator/container; a re-bundled model + a hardened compare are not architectural). Remove the two
now-done TODO follow-ups; the deferred heads-ups (§6) stay tracked.

## 8. Open items for the implementation session

1. **`default` gate outcome is not pre-decidable** — if the GS retrain drops below the gate on the broadened
   domain, STOP and surface it (do not bundle a failing default; the old default stays until resolved).
2. **SkillCorner bundle is conditional on the gate** — the PR's shape (whether `_retention_weights/skillcorner/`
   + the `_PROVIDER_VARIANT` edit exist) depends on the live fit result. The plan must handle both branches.
3. **Re-bundle provenance** — record `sklearn_version` / `training_commit` / n / metrics in the model metadata
   + CHANGELOG (ADR-009 auditable manifest); the corpus-identity guard (if any) tolerances may need the
   broadened-domain row counts.
