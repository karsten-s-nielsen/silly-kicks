# Design: xT-GK per-type base-rate serve switch (goal-kicks)

**Date:** 2026-06-09
**Status:** Draft v3 — incorporates sibling-session review rounds 1+2 (2026-06-09) → pending user review → implementation plan
**Author:** silly-kicks session (Karsten)
**Origin:** TODO.md "xT-GK per-type base-rate serve switch for goal-kicks (4.21.0 follow-up)".
Implements the deferred slice of `2026-06-09-xt-gk-multiprovider-completion-design.md` §2.3/m3.

## Context

xT-GK's RAV term consumes `P(success | geometry)` from a fitted `GkCompletionModel`. 4.21.0 ships
goal-kicks **always model-scored** — `compute_xt_gk` hard-wires `xt_gk_completion_source = "model"`
for every geometry-resolved in-scope row (`silly_kicks/tracking/_xt_gk.py:452`). But goal-kick
completion is poorly predicted from geometry: the bundled `skillcorner` variant measured **goal-kick
AUC 0.433** (chance; GK-pass AUC 0.739), 2026-06-09. So on goal-kicks the RAV `p` is a chance-level
number consumed multiplicatively — on-scale (the SC-vs-GS comparability gate passed *with* goal-kicks
included), but low-discrimination.

The fix (spec §2.3): when a variant's goal-kick sub-domain can't beat chance **with confidence**,
serve the **per-type calibrated base rate** for goal-kicks (tagged `xt_gk_completion_source =
"base_rate"`) instead of the geometric prediction — honest about geometry's limit there, never a
fabricated prediction. Deferred from 4.21.0 because it is a `compute_xt_gk` change that **also
affects the GS `default`** (whose per-type goal-kick behavior was never separately gated) — better
measured and gated on its own than bundled into the SkillCorner release.

### What already exists (verified 2026-06-09)

- `GkCompletionModel` stores per-type `_base_rates` (`{goalkick, throw_in, other}`, fit from the
  in-sample label means; `_gk_completion.py:103-106`) and serializes them (`:152,166`).
- `predict_proba` already serves the per-type base rate for **geometry-missing** rows
  (`_gk_completion.py:124-129`) via `_base_rate_for_type` (`:132-137`).
- The train script `scripts/train_gk_completion.py` already computes per-type held-out AUC + a
  **bootstrap LCB** (`_bootstrap_auc_ci` `:96`; `_report("goalkick", …)` `:191`) — but uses only the
  **GK-pass** point floor (`_GKPASS_AUC_FLOOR = 0.70` `:36`) for *variant selection*. The goal-kick
  measurement is reported, never wired to a serve gate.
- `compute_xt_gk`'s RAV path scores **only geometry-resolved** in-scope rows (geometry-missing →
  NaN, not base-rated — m2; `_xt_gk.py:448-452`). So inside `compute_xt_gk` the *only* base-rate
  trigger is the new type-gate.

So the missing pieces are exactly: (a) a per-type **serve-gate decision** stored with the model, and
(b) the **serve switch** in `compute_xt_gk` that reads it and tags `base_rate`. **Coefficients do not
change** — only goal-kick rows of a *failing* variant flip model→base_rate.

## Decisions (confirmed with user 2026-06-09)

1. **Gate lives in the model artifact.** A new serialized per-type serve-mode field on
   `GkCompletionModel`; an owner-run re-measure re-serializes the bundled `default` + `skillcorner`
   (coefficients unchanged). Self-contained, travels with the model, consistent with `_base_rates`.
   Hub `full` re-upload is a follow-up (fail-open covers the gap).
2. **Gate floor = bootstrap-LCB(per-type AUC) > 0.5** (+ a minimum-n guard; degenerate/undefined AUC
   → base_rate). Serve the geometric model only if that type beats chance with 95% confidence; else
   base_rate. Conservative against the small/noisy goal-kick sample (§3.2/m3 "lower-confidence-bound +
   report n"). skillcorner goal-kicks (0.433) fail → base_rate; the GS goal-kick LCB is measured
   during the owner run (the TODO's "potentially GS").
   - **Why LCB here while the *bundling* gate uses point-AUC ≥ 0.70 (deliberate, `train_gk_completion.py:197-198`).**
     They answer different questions. Bundling asks "is this variant's GK-pass model good enough to
     *ship the variant*?" — a high point bar, where the LCB would over-reject a clearly-good model on a
     few-hundred-row sample. The serve gate asks, per type, "is geometry better than chance *with
     confidence* for THIS type, or should I serve the calibrated mean?" — a low bar (0.5) where the
     conservative LCB is the honest choice for a noisy per-type sample. Not a contradiction; different
     bar, different purpose.
   - **STOP-safeguard for `other`/GK-pass (review H2).** The uniform rule (Decision 4) applies LCB>0.5
     to every type, including the dominant `other`/GK-pass signal. In practice GK-pass (point ≥ 0.70,
     decent n) clears 0.5 comfortably — but a small per-type sample could in principle dip its LCB
     below 0.5. A GK-pass flip-to-base_rate is a **far larger** change than the goal-kick fix and a
     red flag, so the owner run **measures `other`/GK-pass LCB explicitly and treats a flip as a
     STOP/design-review condition, never a silent ship**.
3. **Backward-compat = fail-open.** A 4.21.0 model JSON without the gate field loads with all types
   `"model"` (= current behavior). Purely additive; no custom/unbundled model breaks.
4. **(decided) Uniform per-type gate**, driven by the stored decision — not goal-kick-special-cased.
   The same rule (`LCB(AUC) > 0.5 → model`) is applied to every type; a type whose positive class is
   degenerate/insufficient so AUC is undefined (e.g. `throw_in`, near-empty positive class — §1 of
   the multiprovider spec; m3) **fails → `base_rate`**. `other`/GK-pass clears it (it's the
   bundling-gated signal, AUC ≥ 0.70); `goalkick` is the measured case. No hard-coded per-type
   special path — the gate is one rule over the stored per-type LCB.
5. **(decided) The switch lives in `compute_xt_gk`** reading a model-exposed gate (per the TODO),
   not inside `predict_proba` — keeps `predict_proba` a pure scorer and the per-row source tagging at
   the orchestration layer.

## Components

### 1. `GkCompletionModel` — per-type serve-gate (`silly_kicks/tracking/_gk_completion.py`)

- **New field** `_type_serve_mode: dict[str, str]` over `{goalkick, throw_in, other}` →
  `"model"` | `"base_rate"`. It is a **train-time CV product** (held-out AUC, unseen by in-sample
  `fit()`), so the train script computes and assigns it before `save()`; `fit()` leaves it `{}`.
- **New metadata** `_type_gate_metrics: dict[str, dict]` (`{type: {auc, lcb, n}}`) — model-card /
  "report n" transparency; not read at serve.
- **New pure module-level gate-decision function (review M2)** —
  `serve_mode_from_lcb(lcb: float | None, n: int, *, lcb_floor: float = 0.5, n_min: int = _GATE_N_MIN) -> str`:
  returns `"base_rate"` if `lcb` is `None`/NaN (undefined/degenerate AUC), `n < n_min`, or
  `lcb <= lcb_floor`; else `"model"`. `_GATE_N_MIN` is a small constant (≈ 50 — below which a per-type
  AUC is too unstable to trust; pinned in the plan). This is the one place the rule lives — **unit-tested at the
  boundaries** (LCB just above/below 0.5; `None`/degenerate → base_rate; n-too-small → base_rate), so
  a flipped comparison is caught in CI, not owner-only. Both the train script and the model derive
  the per-type mode through it.
- **New pure methods:**
  - `serve_mode_for_types(type_ids: np.ndarray) -> np.ndarray` — per-row `"model"`/`"base_rate"`,
    defaulting to `"model"` for any type absent from `_type_serve_mode` (the fail-open default).
  - `base_rate_for_types(type_ids: np.ndarray) -> np.ndarray` — vectorized per-type base rate
    (reuses the existing `_base_rates` / `_base_rate_for_type` logic).
- **Serialization:** `VERSION` bump; `save()` writes `type_serve_mode` + `type_gate_metrics`;
  `load()` reads them with `.get(..., {})` — **absent → `{}` → all `"model"`** (fail-open).
  Inference stays pure-numpy + sklearn-free.

### 2. `compute_xt_gk` — the serve switch (`silly_kicks/tracking/_xt_gk.py`)

After `_completion_p` returns the model `pc` for the geometry-resolved `mask` (the only base-rate
trigger here is the type-gate, since geometry-missing rows are unscored — m2):

```python
tids   = actions.loc[mask, "type_id"].to_numpy()
modes  = completion_model.serve_mode_for_types(tids)
is_base = modes == "base_rate"
pc[is_base] = completion_model.base_rate_for_types(tids[is_base])
out.loc[mask, "xt_gk_completion_variant"] = completion_key
out.loc[mask, "xt_gk_completion_source"]  = np.where(is_base, "base_rate", "model")
```

The base-rated `pc` flows into `_rav` exactly as the model `pc` did, so `xt_gk` reflects the
base-rate `p`. The gate *knowledge* stays in the model (single source); `compute_xt_gk` orchestrates
+ tags. `completion_model` is already resolved (`_resolve_completion_for_frames`, `:303-333`).

**Atomic mirror** (`atomic/tracking/...`) mirrors the same switch via the shared path.

### 3. Train script (`scripts/train_gk_completion.py`) — the bulk of the work (review H1)

The current per-type measurement is **under-scoped for this gate** and must be reworked:
- `_report` (per-type AUC + bootstrap-LCB) lives **only** in `_train_skillcorner` (`:157,190-192`);
  the GS `main()` path (`:256-327`) measures only a *pooled* `native_auc` (`:299`) — **no per-type
  AUC at all**. The gate "applies to both fits" (the GS goal-kick LCB is the TODO's "potentially
  GS"), so the per-type measurement must be added to the GS path too.
- The existing partition is `{goalkick, gk_pass = ~goalkick}` (`:191-192`) — `gk_pass` lumps
  `throw_in` into the non-goalkick bucket. But `_base_rates` / the serve gate key on the **3-way**
  `{goalkick, throw_in, other}` (`_gk_completion.py:106`). So `gk_pass ≠ other`, and there is **no
  separate `throw_in` AUC**. The gate keys cannot be computed from today's report.

So: **hoist the per-type measurement into one shared helper** (match-grouped OOF CV → per-bucket
AUC + bootstrap-LCB + n) that **both** `main()` (gs) and `_train_skillcorner` call, over the **3-way
`{goalkick, throw_in, other}`** partition that matches `_base_rates`. Derive each bucket's serve mode
via the shared `serve_mode_from_lcb` (Component 1), record `{auc, lcb, n}` per bucket (m3 "report n"),
and assign `_type_serve_mode` + `_type_gate_metrics` to the model before `save()`. Expected outcome:
`other`/GK-pass clears (point ≥ 0.70), `throw_in` is degenerate → base_rate, `goalkick` is measured
— with the `other`-flip STOP-safeguard (Decision 2) enforced in the owner run.

### 4. Provenance / report (`silly_kicks/tracking/_xt_gk.py`)

- `xt_gk_completion_source` now actually emits `"base_rate"` (was always `"model"`).
- `XtGkReport` gains `completion_source_counts: dict[str, int]` (mirrors the column `value_counts`;
  §2.4 wanted it — moot while always-model). `from_frame` populates it.

### 5. Owner-run re-measure + re-bundle (during implementation)

Run the extended train script on the owner corpora (GS WC2022 + SkillCorner pining via the existing
`_loader_*`, read-only). The re-bundle keeps the **committed coefficients** and attaches only the
freshly-measured gate — but two things must be proven to agree, not assumed (review-2 MEDIUM):

1. **Served coefficients** = the committed `coef`/`intercept`/`mean`/`std`/`base_rates` (kept).
2. **The gate** = OOF AUC freshly CV-measured on the **re-extracted** corpus.

The OOF gate only validly describes the *served* model if the re-extracted corpus is identical to the
one the committed coefficients were fit on (same matches, same `extract_gk_completion_features`
version, same seed, same GS stream). So the re-bundle adds a **corpus-identity guard**: re-fit a
full-data model on the re-extracted corpus and `np.testing.assert_allclose` it against the committed
`coef`/`intercept`/`mean`/`std` at `atol=1e-9` (the script's own idiom, `train_gk_completion.py:227`)
**before** attaching the OOF gate. **Match →** corpus/procedure are identical, the gate provably
describes the served model, attach it + re-save (coefficients byte-unchanged vs HEAD). **Mismatch →**
the corpus/extract/seed/sklearn drifted → **abort and investigate**, never ship a gate measured on the
wrong data. The reviewed diff confirms the only delta is `type_serve_mode` + `type_gate_metrics` +
`version`. Non-goal-kick serve output is therefore byte-identical; only a failing variant's goal-kick
(and degenerate throw-in) rows flip. **The GS goal-kick LCB + the `other`/GK-pass LCB STOP-check are
measured here** and reported to the user (like the xT-NLL real-data run) before finalizing — including
the **per-type `n`** (review-2 L-C: if a type's `n` is near `_GATE_N_MIN`, the n-guard rather than the
AUC decides its flip; acceptable, and visible in `_type_gate_metrics`). Hub `full` re-upload is a
follow-up; fail-open serves it as model until then.

## Testing (TDD, CI everywhere — the gate is committed, so no owner gate is needed)

- **Pure gate decision (CI, review M2 — lead with this):** `serve_mode_from_lcb` at the boundaries —
  LCB just above/below `0.5` → `model`/`base_rate`; `None`/NaN (degenerate AUC) → `base_rate`;
  `n < n_min` → `base_rate`. The rule lives in one tested place; a flipped comparison fails in CI.
- **Model (CI):** `serve_mode_for_types` maps per the stored gate (absent types → `"model"`);
  `base_rate_for_types` returns the per-type `_base_rates`; `save`/`load` round-trips
  `type_serve_mode` + `type_gate_metrics`; **load fail-open** — a JSON without the field → all
  `"model"`.
- **`compute_xt_gk` switch (CI):** a synthetic `GkCompletionModel` with `goalkick → "base_rate"`
  yields, on goal-kick rows, `xt_gk_completion_source == "base_rate"` **and** the RAV `p` equal to the
  goal-kick base rate (so `xt_gk` visibly differs from the model-scored value); GK-pass rows stay
  `"model"`. **A `throw_in → "base_rate"` row is also covered (review M1)**, not just goal-kick. A
  model with all types `"model"` is **byte-identical to today** (regression lock).
- **Real-artifact gate lock (CI everywhere, review M3 — stronger than an owner e2e):** load the
  **bundled `skillcorner`** model and assert `_type_serve_mode["goalkick"] == "base_rate"`
  (post-rebundle), and that `compute_xt_gk` tags `skillcorner` goal-kick rows `base_rate`. This
  skillcorner lock is **authorable red-first** (its 0.433<0.5 outcome is known a-priori). The GS
  `default` artifact test is a **measured-value golden lock** (review-2 L-A): its expected goal-kick
  mode is **filled in from the owner-run report** (not known a-priori, unlike skillcorner), then it is
  a permanent regression lock — flagged as such so it is not mistaken for a guess. Because the gate is
  committed (not in remote data), CI fully locks the real serve behavior — **no owner-gated e2e is
  needed for this PR.**
- **Atomic mirror (CI, review L3):** atomic `add_xt_gk` (`atomic/tracking/features.py:182`)
  **delegates** to `tracking.features.add_xt_gk` (no own `compute_xt_gk`/tagging), so the switch is
  inherited — the atomic test is a **parity lock over the shared path**, not a second implementation.
- **Report (CI):** `XtGkReport.completion_source_counts` == the column `value_counts`.
- **Train-script gate wiring (CI smoke):** the shared per-type measurement + `serve_mode_from_lcb`
  assign + serialize `_type_serve_mode` on a synthetic fit (does-it-run; the *real* AUC values are
  owner-run, but the decision logic is already locked by the pure-function test above).
- **Inherited xT-GK gates:** construct-invariant, nan-safety, id-dtype, dup-action_id,
  provenance-skip idempotence (already enumerated for xT-GK).

## Out of scope

- The other (already-shipped) §2 components — variant selection, the SkillCorner `result_id` fix,
  the comparability gate — all landed in 4.21.0. This PR is **only** the per-type serve switch.
- Re-fitting coefficients or changing features — the gate is additive metadata; coefficients are
  unchanged.
- Hub `full` re-upload — a follow-up (fail-open serves it as model meanwhile); noted in the PR.
- Calibration / AUC bundling gates — unchanged (this adds a *serve* gate, not a *bundling* gate).

## Hyrum / housekeeping (one feature branch, one commit)

- **Not a VAEP retrain** (xt_gk is opt-in — in no default xfn list). **But an `xt_gk` serve-output
  change for any base_rate-flipped type** — goal-kicks of a failing variant (skillcorner certainly; GS
  if its LCB ≤ 0.5) **and degenerate throw-ins** (review M1) → **lakehouse re-materializes `xt_gk` for
  those rows** (CHANGELOG-flagged Hyrum surface; name throw-in, not just goal-kicks). **Quantify the
  blast radius (review-2 L-B):** the owner run already produces the per-type counts, so the CHANGELOG
  states the magnitude — "≈ N% of `xt_gk` rows re-materialized (goal-kicks + throw-ins for variants
  X)" — so the lakehouse can scope its re-materialization rather than discover it.
- **Hub-`full` window (review L2):** until the Hub `full` artifact is re-uploaded, a `full` user
  fail-opens to **model-scored goal-kicks** (the old chance-level `p`) while the bundled `default`
  serves the gate — a transient bundled-vs-Hub difference; **CHANGELOG-flag it so it isn't read as a
  bug.**
- **ADR-024 amendment** (refines xT-GK completion serving; no new methodology). C4-free (a serve
  gate on an existing model — no new model/aggregator/backend/enumeration). The `VERSION` bump is
  **informational only** (review L4) — `load()` fail-opens on missing fields and never version-gates.
- Remove the TODO item; version-bump to **4.21.4** (4.21.3 was taken by the sportec play_evaluation
  PR #121; reconcile vs `origin/main` at commit time) + `uv lock` + dated CHANGELOG; re-serialized
  `default`/`skillcorner` artifacts committed (reviewed diff, not blind).
- Single feature branch + one commit, explicit per-commit approval + the git-commit sentinel;
  `/final-review` before committing.

## Verification

1. Default suite green: `python -m pytest tests/ -m "not e2e and not slow" -q` (new model + switch +
   atomic + report tests + the byte-identical regression lock).
2. Full CI lint locally: `ruff check` + `ruff format --check` (whole tree) + `pyright silly_kicks/`.
3. Owner run (during impl, reported): the extended train script on GS + SkillCorner → the per-type
   `{goalkick, throw_in, other}` LCBs + serve decisions (incl. the `other`/GK-pass STOP-check);
   re-serialized artifacts; confirm `coef`/`intercept`/`mean`/`std`/`base_rates` byte-unchanged vs
   HEAD and that only the base_rate-flipped types (goal-kicks of a failing variant + degenerate
   throw-ins) change serve output.
