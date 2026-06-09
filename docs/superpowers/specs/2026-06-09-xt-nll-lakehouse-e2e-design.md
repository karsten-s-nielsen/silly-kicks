# Design: Committed owner-gated lakehouse-mart NLL cross-check

**Date:** 2026-06-09
**Status:** Draft v3 (brainstorming) — incorporates external review rounds 1+2 (2026-06-09) → pending user review → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** TODO.md "SK-xT-1 follow-ups" → "Committed owner-gated lakehouse-mart NLL cross-check"
(2026-06-07 SK-xT-1). Builds on `xthreat/` (ADR-021).

## Context

SK-xT-1 (4.17.0, ADR-021) shipped a pluggable xT framework with a held-out **transition-model
NLL** evaluator (`compute_holdout_nll` — the negative log-likelihood of a pass's destination zone
given its source zone under a fitted transition matrix) and two transition families: the classic
`singh_counts` and the new `kde_smoothed`.

During 4.17.0 the KDE-beats-Singh claim was triangulated **once, as a non-committed one-off**,
against the production mart `soccer_analytics.dev_gold.fct_action_values` (~8.8M actions / 5,404
matches): it confirmed a **~4% relative held-out-NLL win for KDE** at a production-tuned bandwidth.
That one-off was deliberately kept out of the default suite (SK-xT-1 spec review #1) to avoid
baking a product→consumer-infra dependency into silly-kicks' tests.

This work makes that triangulation **permanent and reproducible** as an `@pytest.mark.e2e`,
owner-gated test — a durable regression tripwire that catches any future change which silently
tanks KDE's advantage on real data. It is the lakehouse-mart **owner-gated sibling** of the existing
open-data `tests/test_xthreat_statsbomb_e2e.py` (which asserts the same KDE-beats-Singh held-out-NLL
on StatsBomb open data), and follows the gating discipline of the owner-gated GradientSports e2e
(`tests/spadl/test_gradientsports_scoreline_e2e.py`).

### Empirical grounding (carried from SK-xT-1)

| Reference | Resolution | Scored set | Singh NLL | KDE NLL | Rel. win |
|-----------|-----------|-----------|-----------|---------|----------|
| Lakehouse Phase-0/1 (committed) | 12×8 (96 zones) | **passes** | 3.78924 | ~3.748 | ~1.08% |
| 4.17.0 one-off (tuned bandwidth, full mart) | — | passes | — | — | **~4%** |

Two facts drive the design and were both flagged by the external review:

1. **Scored set = successful passes only.** `compute_holdout_nll` internally filters to
   `_get_successful_move_actions` (pass + dribble + cross, success-only; `_grid.py:105-128`). The
   StatsBomb sibling additionally **pre-filters the holdout to passes** before scoring
   (`holdout_passes`, `test_xthreat_statsbomb_e2e.py:90`) — and the published lakehouse reference is
   explicitly **"Held-out NLL (passes)"**. To triangulate against 3.789 and to be apples-to-apples
   with the sibling, this test **fits on the full train but scores a passes-only holdout**, and
   states the scored set explicitly (the internal filter makes it invisible at the call site).
2. **Absolute NLL is resolution- and corpus-dependent** (16×12 = 192 zones ⇒ higher absolute NLL
   than 12×8; the mart grows over time). Only the **relative KDE-over-Singh win** is a stable
   invariant, so only the relative win is asserted; absolutes are logged for human triangulation.

The shipped `KDEParams.bandwidth` default is **1.0** (pure Silverman). The held-out-optimal
multiplier is **corpus-size-dependent**: ~1.0 on a 64-match sample, **≥4 on the 8.9M-action mart**
(adaptive Silverman shrinks per-zone bandwidth ~`n^(-1/6)`, so larger corpora need a larger
multiplier). The one-off's ~4% was at the tuned (≥4) bandwidth. **Consequence:** `bandwidth=4.0` is
only well-tuned for the *full* mart — on a small subsample it is over-smoothed and may not beat
Singh. The hard assertions therefore fire **only on the full corpus** (see Decisions).

## Decisions (confirmed with user + external review 2026-06-09)

1. **Scored set = passes-only.** Fit Singh/KDE on the full `train`; score `compute_holdout_nll` on a
   passes-only holdout (`holdout[holdout.type_id == actiontype_id["pass"]]`). Matches the StatsBomb
   sibling and the published 3.789 reference. The scored set is named in the logged block.
2. **Assertion = tuned bandwidth + conservative floor, full-corpus-only.** Fit KDE at a
   production-representative `bandwidth=4.0`; on the **full mart** hard-assert KDE strictly beats
   Singh AND the relative win clears a conservative floor (~1.5%, well below the observed ~4%). When
   `XT_NLL_E2E_MAX_MATCHES` is set (subsample), **all assertions downgrade to log-only** (bandwidth
   4.0 is mis-tuned for small corpora — asserting would risk a spurious red).
3. **Shipped-default check is a hard STRICT-BEAT assert (full corpus).** Also fit
   `KDEParams(bandwidth=1.0)` (the shipped default) and hard-assert it **strictly beats Singh** at
   16×12 — `kde1 < singh`, no relative floor. The first owner run confirmed it (+3.03% on the full
   9.6M-action mart), so it is asserted, not deferred. **Why strict-beat, not a floor:** the
   default's margin *erodes as the mart grows* (the held-out-optimal bandwidth multiplier rises with
   corpus size, so a fixed 1.0 under-smooths progressively more — measured ~+8.7% on a 300-match
   smoke → ~+3.0% on the full mart). A floor would risk tripping on benign corpus growth; strict-beat
   catches only a real "the shipped default stopped beating Singh" regression. The tuned KDE(4.0)
   keeps the sensitivity *floor* (Decision 2); the default gets the strict-beat. No coverage gap: the
   shipped default is protected.
4. **Resolution = 16×12 (assert) + 12×8 (log).** Hard-assert at the silly-kicks library default
   16×12; additionally compute+log 12×8 for direct triangulation against the published 3.789→3.748.
   Resolution + scored set are stated in the logged block (SK-xT-1 convention).
5. **Live-path coverage guard + NA-free masking.** After shaping, before splitting: (a) assert the
   string→id mapping covered the corpus (`type_id`/`result_id` mapped fraction > 0.95) — fails loud
   on mart-vocabulary drift (the `"successful"` vs `"success"` / casing class of bug that has bitten
   this codebase) instead of silently dropping every move → `nan` → a confusing "KDE regression";
   then (b) **drop the ≤5% unmapped rows** (`actions.dropna(subset=["type_id","result_id"])`). Step
   (b) is load-bearing: the `Int64` ids carry `<NA>` for unmapped rows, and a nullable-boolean
   `<NA>` mask (`holdout[holdout.type_id == pass_id]`, and the internal `_get_move_actions` OR-mask)
   **can raise `ValueError` on older pandas** (the owner's `<2.3.0` range — the connector forces
   `pandas<2.3.0`; CI's 2.3.3 tolerates it, exact lower bound unmeasured). The masking only ever
   runs in the owner env, so CI is structurally blind to it; dropping unmapped rows in the
   orchestrator makes every downstream `==`-mask NA-free and version-independent regardless of the
   precise raising boundary. The drop lives in the orchestrator (not the pure shaper, which keeps its
   deliberately NaN-tolerant contract that the unit test exercises).
6. **Adapter placement.** The mart read **and** the mart→SPADL shaping live together in
   `scripts/_loader_databricks.py` (it already imports `silly_kicks.spadl` and converts to SPADL in
   `_convert`). The e2e imports them via `import scripts._loader_databricks as L` (PEP-420 namespace
   package — already used by `tests/calibration/test_loader_databricks.py:3`). No `importlib`
   file-loading, no cross-import from an e2e module.
7. **Pure verdict logic is unit-tested in CI.** Both the relative-win arithmetic
   (`(singh − kde)/singh`) **and the tripwire predicate** (strict-beat AND `rel >= floor`) are
   extracted to pure helpers in the existing shared `tests/_xthreat_helpers.py` —
   `nll_relative_win(baseline_nll, candidate_nll)` and
   `kde_clears_tripwire(singh_nll, kde_nll, *, floor)` — and unit-tested with synthetic NLLs
   (KDE-wins, KDE-loses, just-below/exactly-at floor, NaN-from-empty-corpus). The owner gate then
   only supplies data; the *entire* decision (not just the ratio) is shift-left TDD'd, so a flipped
   comparison or wrong-direction floor is caught in CI, not owner-only. (Long-term, if a second
   consumer appears — e.g. `scripts/calibrate_xt_bandwidth.py`, which already reports the Singh
   baseline — `nll_relative_win` graduates to `xthreat/_eval.py` as a public companion to
   `compute_holdout_nll`; not promoted speculatively, YAGNI.)
8. **Corpus = full mart + env knob.** Default to the full mart; honor `XT_NLL_E2E_MAX_MATCHES` to
   subsample for a faster smoke run (log-only, per Decision 2). The subsample is `ORDER BY match_id
   LIMIT n` — deterministic but **lexicographically biased** (first-N ids, not a representative
   sample); documented as a smoke aid only.

## Components

### 1. Loader: read + shape — `scripts/_loader_databricks.py`

**`fetch_action_values(*, max_matches: int | None = None) -> pd.DataFrame`**
- Reuses the module's `_connect()` + `_query_param()` (batched fetch).
- Table name is a **fixed module constant** `_ACTION_VALUES_TABLE = "soccer_analytics.dev_gold.fct_action_values"`
  — never interpolated from caller input (mirrors the `_ALLOWED_PROVIDERS` discipline).
- Selects only the columns the passes-NLL path needs — **no `period`/`action_id`** (unused; dropped
  to keep the ~8.8M-row pull lean):
  `match_id, start_x, start_y, end_x, end_y, action_type, action_result`.
- `max_matches` (when set): `SELECT DISTINCT match_id ... ORDER BY match_id LIMIT %(n)s`
  (parameterized) → `WHERE match_id IN (...)` (parameterized id list). `None` ⇒ full table.

**`shape_action_values(df: pd.DataFrame) -> pd.DataFrame`** (pure, no network)
- `type_id = action_type.map(spadlconfig.actiontype_id).astype("Int64")`,
  `result_id = action_result.map(spadlconfig.result_id).astype("Int64")` — string → nullable-int
  code (unmapped → `<NA>`; `Int64` is deliberate per ADR-019, avoiding a float id column; the
  orchestrator drops the `<NA>` rows after the coverage guard, so the ids reach the masks NA-free).
- `game_id = match_id` (the `holdout_split` + per-group default key).
- Pure + NaN-tolerant (it does **not** itself guard coverage or drop rows — that is the
  orchestrator's job, so the unit test can exercise the NaN-passthrough path).
- The `import silly_kicks.spadl.config as spadlconfig` stays **function-local** (the module's
  established lazy-import discipline — top-level imports are `os`/`Iterator`/`pandas` only).

### 2. The e2e test — `tests/test_xthreat_nll_lakehouse_e2e.py`

Flat `test_xthreat_*` naming. A **thin orchestrator** — fetch → shape → guard → split → fit → score
→ verdict — over proven, unit-tested seams:

- **Gating:** `pytestmark = [pytest.mark.e2e, skipif(missing any DATABRICKS_* env), skipif(databricks-sql-connector not importable)]`.
- **Source:** `import scripts._loader_databricks as L`; `raw = L.fetch_action_values(max_matches=...)`;
  `actions = L.shape_action_values(raw)`.
- **Coverage guard + NA-drop (Decision 5):** `assert actions.type_id.notna().mean() > 0.95` and
  same for `result_id`, with an actionable "mart vocab drift" message; then
  `actions = actions.dropna(subset=["type_id", "result_id"])` so every downstream `==`-mask is
  NA-free (mask-safe on the owner's `pandas<2.3.0`).
- **Split + passes-only scoring (Decision 1):** `train, holdout = holdout_split(actions, holdout_fraction=0.15)`;
  `holdout_passes = holdout[holdout.type_id == spadlconfig.actiontype_id["pass"]]`.
- **Fits + scoring** at `GridSpec(16, 12)` (assert) and `GridSpec(12, 8)` (log): Singh,
  KDE(`bandwidth=4.0`), KDE(`bandwidth=1.0`); `compute_holdout_nll(matrix, holdout_passes, grid=grid)`
  for each. Relative wins via `nll_relative_win` (Decision 7).
- **Hard assertions — full corpus only (Decisions 2–3):**
  - `kde_clears_tripwire(singh_nll, kde4_nll, floor=_MIN_RELATIVE_WIN)` at 16×12 (tested predicate:
    strict-beat AND `rel >= 0.015`; the tuned-bandwidth sensitivity tripwire).
  - `kde_clears_tripwire(singh_nll, kde1_nll, floor=0.0)` at 16×12 (strict-beat: the shipped default
    must beat Singh; no eroding floor — see Decision 3).
  - When `XT_NLL_E2E_MAX_MATCHES` is set: **skip both hard asserts** (subsample is log-only).
- **Triangulation log (always):** a labeled block — scored set (`passes,success`), n_actions,
  n_train/holdout matches, both resolutions' Singh/KDE@1.0/KDE@4.0 NLLs + relative wins, and the
  published 12×8 reference (3.78924 → ~3.748).

### Unit-testable seams (regular suite, no network)

- **Shaping** — extend the existing `tests/calibration/test_loader_databricks.py`
  (`import scripts._loader_databricks as L`) with `L.shape_action_values(...)` cases on a tiny
  synthetic mart-shaped DataFrame: correct `type_id`/`result_id` mapping; `game_id` aliasing;
  `Int64` dtype; unmapped/NaN passthrough.
- **Verdict logic** — a new regular test `tests/test_xthreat_nll_relative_win.py` (note the
  `test_xthreat_*` prefix) for both helpers from `tests/_xthreat_helpers.py`: `nll_relative_win`
  (KDE-wins/loses, exactly-at-floor, `nan` when baseline is `nan`/0 — the empty-corpus case) and
  `kde_clears_tripwire` (clears well-above-floor, fails just-below-floor, True exactly-at-floor,
  False on KDE-loss, False on `nan`).

The live-mart assertions are inherently owner-gated and cannot be unit-tested without the mart — but
shaping, the coverage threshold, **and the full tripwire predicate** now all run in CI.

## Out of scope

- Changing any library default (bandwidth stays 1.0; tuning lives in `scripts/calibrate_xt_bandwidth.py`).
- Wiring the mart pull into `_loader_databricks.load_matches`'s converter path (that is the TF-24
  calibration harness path; this helper is a standalone read of the already-valued gold mart).
- Any VAEP/xT retrain trigger — read-only validation. **Additive, no behavior change.**

## Housekeeping (one feature branch, one commit)

- **Single commit per branch** (`pr-s<NN>`-style or descriptive feature branch); **explicit
  per-commit approval + the git-commit sentinel** before committing (the standing policy + the
  sentinel hook). No per-task commits.
- Run the mandatory **`/final-review`** gate before committing.
- Remove the **sole** item under "SK-xT-1 follow-ups" in `TODO.md` — remove the whole now-empty
  subsection (header + intro + bullet), leaving the `---` separators intact.
- **Version: 4.21.2** (the parallel session is holding, so the number is clear). Bump across the
  version sites + `uv lock` + a dated CHANGELOG entry matching the file's `## [x.y.z] — DATE` /
  `### <Type> — <desc>` style. **The wheel ships only `silly_kicks/`** (`packages = ["silly_kicks"]`)
  and every artifact here is in `scripts/`+`tests/`, so the 4.21.2 wheel is byte-identical to 4.21.1
  except `__version__` — per the per-PR-bump convention we still bump+tag+publish, but the CHANGELOG
  entry says **"no shipped-library change — test + dev-tooling helpers only"** so a consumer diffing
  wheels isn't misled (Hyrum).
- No new ADR/NOTICE (tests existing ADR-021 functionality; no new methodology). C4-free.
- Spec + plan bundle into the single feature commit (no standalone doc commit).

## Verification

1. **Default suite green** (shaping + verdict unit tests added; nothing else changes):
   `python -m pytest tests/ -m "not e2e and not slow" -q`.
2. **e2e collects-but-skips without env:** confirm the e2e test is skipped (not errored) when
   `DATABRICKS_*` / connector are absent.
3. **Full CI lint locally:** `ruff check silly_kicks/ tests/ scripts/ && ruff format --check
   silly_kicks/ tests/ scripts/ && pyright silly_kicks/` (whole-tree format per the 4.21.0 lesson).
4. **Owner run** (isolated env with the connector — NEVER the main `.venv`, whose `pandas<2.3.0`
   conflict the connector triggers):
   `DATABRICKS_HOST=... DATABRICKS_HTTP_PATH=... DATABRICKS_TOKEN=... pytest tests/test_xthreat_nll_lakehouse_e2e.py -m e2e -s`
   → expect KDE(4.0) win ≳4% at 16×12 on passes, the 12×8 block near 3.789→3.748, and the logged
   `kde1<singh` result (to decide promotion). Optional faster smoke: `XT_NLL_E2E_MAX_MATCHES=300`
   (log-only).
