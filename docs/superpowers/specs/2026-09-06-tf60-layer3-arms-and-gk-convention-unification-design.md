# TF-60 — Complete Layer-3 Counterfactual Arms + Ghost-GK Convention Unification — Design

- **Status:** DRAFT — for owner review + independent `/review-spec`. Supersedes the split PR4/PR6
  decomposition of the parent spec (`2026-08-30-tf60-rest-defense-structure-and-gk-design.md` §17):
  the owner directed (2026-09-06) a **gold-standard, long-term** reshape where *scope and breaking
  changes are not a concern* — fold PR4 (GK arms) and PR6 (outfield arm) into **one** complete
  Layer-3 cycle, and **unify the ghost-GK goal-relative convention now** (retrain), rather than
  leave the two ghost models on inconsistent conventions.
- **Date:** 2026-09-06
- **Feature:** TF-60 (rest defense), Layer 3 — the counterfactual deterrent arms.
- **Decision record:** ADR-089 (to be written in this cycle).
- **Touches:** `silly_kicks/tracking/` (ghost-GK convention + both serves), `silly_kicks/gkdv/`
  (engine write-back simplification), `silly_kicks/restdefense/` (new Layer-3 modules), `scripts/`
  (ghost-GK re-train + a new Layer-3 corpus driver), `docs/research/` (applied construct-validity).
- **Depends on (all shipped):** PR1 Layer-1 (ADR-080), PR2 Layer-2 (ADR-081), PR3 ghost-GK sweeper
  re-fit (ADR-083), PR5 ghost-outfield model (ADR-087); gkdv delta seams + probe verdicts (ADR-043,
  TF-19 A+2/ADR-082); ADR-005/011/016/040/044/050/076 (trained-model discipline), ADR-019
  (id_compat), ADR-028/051-D3/055 (orientation + `GoalMap`), ADR-037 (restdefense import allowlist),
  ADR-052 (corpus driver seam), ADR-053/054 (SB360 audit + velocity provenance), ADR-062/063/067/077
  (SB360 FOV / velocity tiers / velocity-keyed variants), ADR-068/073 (no-rescan + sub-quadratic).

---

## 1. Executive summary

TF-60 Layer 3 prices how much an in-possession team's *actual* rearguard positioning suppresses the
opponent's counter-danger, versus a **league-average "ghost"** in the same frame state. It is a
**counterfactual deterrent** metric in attacker-value units, so **negative = deterrent** (the gkdv
sign convention). Two interventions, each with a threat sub-arm (xt-weighted pitch control) and a
space sub-arm (ΔDAS):

| Arm | Intervention | Ghost source |
|---|---|---|
| `rd_gk_deter_threat` / `rd_gk_deter_space` | ghost A's **keeper** | `serve_ghost_gk_positions` (**sweeper** variant, PR3) |
| `rd_outfield_deter_threat` / `rd_outfield_deter_space` | ghost A's deepest-`n_rearguard` **field defenders** | `serve_ghost_outfield_positions` (PR5) |

This cycle ships the **complete** Layer 3 (both interventions) as one coherent subsystem, plus three
supporting changes the gold-standard end-state requires:

1. **Serve-returns-frame-coords** — both `serve_ghost_gk_positions` and `serve_ghost_outfield_positions`
   return frame-ready `ghost_x` / `ghost_y` (keeping the goal-relative `ghost_gr_x` / `ghost_gr_y` as
   audit). The goal-relative→frame conversion moves **into the serve** (the one place that owns both
   the model's transform functions *and* the orientation via `GoalMap`), so no consumer ever
   re-derives orientation. gkdv's engine is simplified to consume `ghost_x/_y` directly.
2. **Ghost-GK both-axes convention unification (retrain)** — the ghost-GK feature extractor currently
   uses an **x-only** goal-relative convention (y absolute), whereas the newer ghost-outfield model
   uses the correct **full 180° point reflection** (both axes). This cycle changes ghost-GK's
   convention to both-axes and re-fits **all five variants** (`default` / `position_only` / `full` /
   `sweeper` / `sweeper_position_only`), making the model fully orientation-invariant and both ghost
   models share **one** convention. This is a coupled, not optional, change: once the serve returns
   frame coords **and** the model predicts both-axes gr_y, gkdv's legacy x-only write-back would be
   *wrong*, so the retrain and the serve-frame-coords change are one design.
3. **Instrument-validity probe + applied construct-validity report** — a TF-19-A+2-rigor probe
   (`restdefense/_probe.py`) reusing gkdv's generic Layer-0/Layer-1 verdict functions, plus a
   DGX-corpus applied report (`docs/research/`, provenance-stamped, reported-never-gated) establishing
   each arm's instrument validity and expected-sign behaviour against a **pre-registered anchor**.

**Additive to VAEP** — the arms enter no default xfn list, change no existing VAEP feature, force no
VAEP retrain. The ghost-GK retrain **does** trigger a **gkdv re-materialize + TF-19 sign-off re-run**
and re-materialization of any consumer persisting `ghost_gk_x/_y` — a downstream cost surfaced in §7.
**No new C4 container** (restdefense exists; the arms are `compute_*` functions, no `add_*`
aggregator → C4 aggregator count stays 33).

---

## 2. Scope and non-goals

### In scope

- Serve-frame-coords on both ghost serves + gkdv engine simplification (§4).
- Ghost-GK both-axes convention unification + re-fit of all 5 variants + HF re-publish (§5).
- The complete Layer-3 counterfactual engine `restdefense/_counterfactual.py` (`which ∈ {keeper,
  rearguard}`) + the arms `restdefense/_arms.py` (4 arm columns) + `merge_rest_defense` +
  `RestDefenseGhostReport` (§6).
- The instrument-validity probe `restdefense/_probe.py` (§7) + a corpus driver
  `scripts/build_tf60_layer3_arm_values.py` + the applied construct-validity report
  `docs/research/tf60_layer3_construct_validity/` (§8).
- Full CI method-gate registration + `@e2e` method gate (§10); the complete artifact set (§11).

### Non-goals

- **No new VAEP feature, no VAEP retrain.** The arms are coach-facing counterfactual outputs; opting
  any into a default xfn list would be a self-triggered VAEP retrain and is out of scope.
- **No composites / archetypes / rankings in the library** (raw-primitives-ship discipline, ADR-055 /
  NOTICE TF-45).
- **No new C4 container or `add_*` aggregator.** The arms are `compute_*`.
- **No change to Layer-1/2** (`compute_rest_defense` / `summarize_rest_defense` Layer-1/2 columns are
  byte-identical), except that `summarize_rest_defense` learns to *also* mean the arm columns when
  they are present in a merged table (§6.4) — a purely additive rollup that does not touch Layer-1/2
  values.
- **No predictive success/counter classifier** (Forcher AUC-0.60 and the xBG/EPV classifiers stay
  framing only).

---

## 3. The design decisions (owner-directed) and their rationale

| # | Decision | Rationale |
|---|---|---|
| D1 | **Fold PR4 + PR6 → one complete Layer-3 cycle.** | Layer 3 is one subsystem; splitting it was scope-minimization. The shared `which`-engine + gkdv delta seams make the keeper arm nearly free once the outfield arm exists. |
| D2 | **Ship the keeper ΔDAS arm despite it being a weak instrument, with an *honest* verdict.** | The house pattern is to ship the honest lens + the validity caveat (xtgk-v2 "NOT construct-validated"). The probe (§7) reports `arm_unscoreable` / `instrument_void` per arm; hiding a weak arm is less honest than shipping it flagged. |
| D3 | **Move the goal-relative→frame conversion into the serve seam(s).** | The write-back must use the model's *own* `_gr_x`/`_gr_y` (self-inverse) so it can never drift from the training convention. The only place with both the transform functions and the orientation (`GoalMap`) is the serve. This kills the "which convention / did I flip both axes" bug class at the source, for every future consumer. |
| D4 | **Unify the ghost-GK convention to both-axes now (retrain).** | The two ghost models use inconsistent conventions; ghost-GK's x-only leaves it not y-chirality-invariant (§5.1). The full point reflection is correct (ADR-051 §8b). Once the serve returns frame coords and the arms consume them, a lingering x-only GK model is a latent trap. |
| D5 | **Upgrade construct-validity to TF-19-A+2 rigor** (dose + placebo controls + pooled instrument/responsiveness verdicts + pre-registered anchor). | The outfield intervention (rearguard displacement) is distinct from TF-19's keeper displacement, so its instrument validity deserves independent establishment; a bare sign test is not the gold standard. |
| D6 | **One branch, minimal commits** — the *only* multi-commit split is the provenance-mandated one (code clean → DGX train/run → weights+report). | Owner directive + the ADR-063 `_provenance` clean-tree rule: bundled weights and a provenance-stamped research artifact must be produced *from* an already-committed clean code state. |

---

## 4. Piece 1 — Serve-returns-frame-coords + gkdv engine simplification

### 4.1 Both serves gain frame coordinates

`serve_ghost_gk_positions` (`_ghost_gk.py:2795`) currently returns one row per
`(game_id, period_id, frame_id, gk_team_id)` with **goal-relative** `ghost_gr_x` / `ghost_gr_y` +
`ghost_clamped` / `ghost_out_of_box`. `serve_ghost_outfield_positions` (`_ghost_outfield.py:1038`)
returns one row per `(game_id, period_id, frame_id, team_id, slot_index)` with goal-relative
`ghost_gr_x` / `ghost_gr_y` + `player_id` + `ghost_outfield_source`.

**Change (both):** add frame-coordinate columns `ghost_x` / `ghost_y`, computed from the goal-relative
values via **the model's own transform functions** (`_ghost_gk`'s `to_gr_x`/`to_gr_vx` family;
`_ghost_outfield`'s `_gr_x`/`_gr_y`), which are self-inverse, using the defended-goal end the serve
already resolves internally (it *must*, to extract goal-relative features — the GK serve threads
`goal_x` through `_serve_positions_core`; the outfield serve builds a `GoalMap` in its extractor). The
existing `ghost_gr_x/_y` columns are **retained** as audit. This is the single source of truth for the
conversion: because it uses the same transform the model was trained with, it cannot drift.

- ghost-GK: the extractor's transforms are **currently local closures** inside
  `extract_ghost_gk_features` (`_ghost_gk.py:737-741`), unreachable by the serve. §5.2 **hoists**
  `to_gr_x` / `to_gr_y` / `to_gr_vx` / `to_gr_vy` to module scope (taking an explicit `flip`, mirroring
  `_ghost_outfield._gr_x`/`_gr_y`/`_gr_vel`), so the extractor **and** the serve call the *same*
  module-level function — the transform *is* the convention (§4.3), automatically both-axes after §5.
  The GK serve builds its output columns inline (no `_SERVE_OUTPUT_COLS` constant, unlike outfield's
  `_ghost_outfield.py:938`), so `ghost_x` / `ghost_y` are appended at the serve's output-frame
  construction (`_ghost_gk.py:2900-2914`); the plan names the exact site.
- ghost-outfield: `ghost_x = _gr_x(ghost_gr_x, flip)`, `ghost_y = _gr_y(ghost_gr_y, flip)` (both axes
  — the model already trains both-axes; `_gr_x`/`_gr_y` are already module-level).

The serve output-column lists (`_SERVE_OUTPUT_COLS` and the GK equivalent) grow by `ghost_x`,
`ghost_y`. A NaN / `variant_unavailable` / `fov_cropped` ghost yields NaN frame coords (never a
fabricated coordinate).

### 4.2 gkdv engine consumes frame coords

gkdv's `_engine._build_provenance` (`_engine.py:390-392`) currently converts goal-relative→frame
itself (x-only: `ghost_x = where(defended==0.0, gr_x, 105-gr_x)`; `ghost_y = ghost_gr_y`). After §5
the ghost-GK model predicts **both-axes** gr_y, so this x-only conversion would place the keeper at
the flipped gr_y *as if it were absolute* — a silent y-error for high-x-defending teams. The fix is
to **consume the serve's `ghost_x/_y` directly** and delete the local conversion:

- `build_ghost_frames` (`_engine.py:554`) already calls `serve_ghost_gk_positions`; it now reads
  `ghost_x/_y` from the serve output.
- `_build_provenance` drops the local goal-relative→frame math; `ghost_x/_y` flow through from the
  serve. `_PROVENANCE_COLUMNS` and `provenance_to_targets` are unchanged in shape (they still expose
  `ghost_x/_y` + the audit `ghost_gr_x/_y`).
- `_write_back` (`_engine.py:438`) is unchanged (it already writes `ghost_x/_y`).

This is a **coordinated, required** change: the ghost-GK convention flip (§5) and the serve-frame-coords
move (§4.1) together make gkdv correct without gkdv re-deriving any orientation.

**Correctness note:** with the serve-frame-coords change *alone* (no retrain), the served physical
positions are **byte-identical** — the serve just also reports the frame coords it (or gkdv) would
have computed. The gkdv keeper-arm value change in this cycle is caused by the §5 **retrain**, not the
§4 refactor. The two are separated in the tests (§10): a frame-coords-parity test (physical position
unchanged for a fixed model) and the retrain golden regeneration.

### 4.3 The transform is the single source of truth (no convention flag needed)

The serve conversion uses the model's **own module-level** `_gr_x` / `_gr_y` transform helpers (they
are self-inverse), so the code that computes the frame coord and the code that computed the
goal-relative feature are literally the same function — the conversion cannot drift from the training
convention, and there is **no separate convention flag** to keep in sync. (For ghost-outfield these
helpers are already module-level; for ghost-GK, §5.2 hoists them from their current closure form so
this "same function" guarantee holds — without the hoist the serve would re-derive `flip` + the
formula as a second copy, reintroducing the exact drift class this eliminates.) Cross-convention loading is prevented by
the existing fail-closed load guard, not by a flag: a pre-retrain (x-only) ghost-GK artifact loaded
against the post-retrain (both-axes) code recomputes a **different** feature-contract fingerprint on
the fixed probe → `IntegrityError` (ADR-011/050). So the serve never sees a convention-mismatched
model — a stale artifact fails loud at load, and an external consumer holding old weights must
re-download the re-published both-axes variant (the correct fail-loud outcome). This is why the
two-commit structure (§9) is load-bearing: Commit-1 code cannot load the old bundled weights, so the
DGX re-fit from Commit-1 produces the matching weights that Commit 2 bundles.

---

## 5. Piece 2 — Ghost-GK both-axes convention unification (retrain)

### 5.1 Chesterton's Fence — why x-only exists and why both-axes is correct

The ghost-GK extractor (`extract_ghost_gk_features`, `_ghost_gk.py:734`) sets `flip = goal_x > 50.0`
and applies `to_gr_x` (x flip) + `to_gr_vx` (vx negate), but leaves **y absolute**:
`ball_y = by_raw` (`:754`), `ball_vy = bvy_raw` (`:756`), `atk_cy = mean(attacking y)` (`:800`),
`ball_to_goal_angle = arctan2(ball_y - GOAL_Y, ball_x)` (`:816`), and the **target** `gr_y` is
absolute. Audited feature-by-feature:

- **Invariant under a y-mirror** (`y → 68 - y`), so unaffected either way: `ball_dist`
  (`(ball_y-34)²` symmetric), `defensive_line_width` (a y-range), `attackers_in_box`
  (box predicate symmetric around y=34), `ball_to_nearest_atk` (a y-difference), `compactness`
  (hull area), the velocity-derivative features (y-differences).
- **Signed-y quantities that a mirror *should* negate/flip** (ADR-051 §8b: "adding a signed-y quantity
  … every BEARING negated"): `ball_y`, `ball_vy`, `atk_cy`, `ball_to_goal_angle`, and the target `gr_y`.

Under the current x-only convention these signed-y features stay absolute, so a mirror-image scene at
the **other** goal scores **differently** — the model is **not y-chirality-invariant**. No feature
genuinely *requires* absolute y (the invariant ones don't care; the signed ones should flip). x-only
is a legacy pre-ADR-051-§8b choice, self-consistent (train + serve both absolute) but suboptimal (it
cannot share y-geometry signal between the two goalmouths, and it violates orientation invariance).
**The both-axes retrain is the correct fix.**

### 5.2 The change

- **Hoist the GK goal-relative transforms to module scope.** `to_gr_x` / `to_gr_vx` are currently
  local closures inside `extract_ghost_gk_features` (`_ghost_gk.py:737-741`); lift them (and the new
  `to_gr_y(y, flip) = (_FIELD_WIDTH - y) if flip else y` and `to_gr_vy(vy, flip) = -vy if flip else vy`)
  to module-level functions taking an explicit `flip`, mirroring `_ghost_outfield._gr_x`/`_gr_y`/`_gr_vel`.
  Both the extractor and the serve (§4) then call the *same* module-level function (the D3 anti-drift
  guarantee), and the two extractors share one convention.
- Route `ball_y`, `ball_vy`, `atk_cy`, `ball_to_goal_angle`, and the target `gr_y` through the hoisted
  `to_gr_y` / `to_gr_vy` (the signed-y features from §5.1).
- This changes the feature vector's numeric output for high-x-defending frames → the model's
  **feature-contract fingerprint** and **chirality fingerprint** change → a re-fit is mandatory
  (a mixed-convention bundle is incoherent, so **all 5 variants** re-fit together).
- **Strengthen the chirality gate to full x+y.** The ghost-GK chirality probe today verifies the
  x-goal-flip. Post-retrain the model is fully orientation-invariant, so the probe's
  `canonical_probe_frame` and its verification are extended to include a y-mirror leg (a scene and its
  180° point reflection must score consistently). This is a load-bearing upgrade: it is what proves
  the both-axes convention actually took.
- `GEOMETRY_VERSION` is **not** bumped — the ghost-GK `to_gr_*` helpers are **confined to
  `_ghost_gk.py`** (module-level after the §5.2 hoist), not the shared `_geometry.to_goal_relative_*`
  that `GEOMETRY_VERSION` versions. The model's own feature-contract + chirality fingerprints carry the
  change. (If review finds any shared-`_geometry` path is touched, `GEOMETRY_VERSION` bumps per
  ADR-051 §8b — but the current reading is that the change is confined to `_ghost_gk.py`.)

### 5.3 Re-fit + publish (Phase B, DGX)

- Re-train all 5 logical variants (`default` / `position_only` / `full` / `sweeper` /
  `sweeper_position_only`) via the existing `scripts/train_ghost_gk.py`, unchanged except that it now
  trains against the both-axes extractor. (`position_only` is a `--feature-set`, not a `--variant`
  (`train_ghost_gk.py:303`), so the 5 are produced by variant × feature-set combos.) **Same public
  179-match corpus** (ADR-038 fail-closed public-only), **clean `training_commit` = Commit-1 SHA**
  (§9), two-commit provenance.
- **Bundling: 4 in the wheel, 1 on the Hub.** `default` / `position_only` / `sweeper` /
  `sweeper_position_only` are bundled (ride Commit 2); **`full` is Hub-hosted** (`from_variant("full")`
  → `from_hub`, `_ghost_gk.py:2461`), not in the wheel — it is re-fit and **re-published to HF**, not
  bundled. All 5 re-published to HF through the ADR-088 card-required seam.
- Regenerate the golden / chirality / feature-contract artifacts + `SHA256SUMS`; HF re-publish all 5
  variants through the ADR-088 card-required seam (`scripts/_hub_publish.publish_model_with_card`).
- The `sweeper` variant re-fit here is the one the keeper arm consumes (§6.2) — so the keeper arm
  automatically uses the both-axes sweeper.

### 5.4 Downstream — the gkdv re-materialize + TF-19 sign-off re-run are IN this cycle (owner-directed 2026-09-06)

The keeper ghost position changes (both-axes gr_y), so gkdv's keeper-arm deltas change → the gkdv
arm-values table + the ICC-gated TF-19 sign-off (completed 4.68.0) must be re-produced from the new
weights. The owner directed (2026-09-06) that this re-run is **pulled into this cycle**, not left as a
follow-up — so Phase B (§9) additionally:

- **Re-runs `scripts/build_gkdv_arm_values.py`** on the public corpus with the new both-axes ghost-GK
  weights → a re-materialized gkdv arm-values table (its own ADR-052 shard generation + ADR-037
  provenance, `run_commit` = Commit-1 SHA).
- **Re-runs the TF-19 sign-off** (`scripts/build_tf19_instrument_responsiveness.py`: the §6.1 ICC
  gate, §6.2 per-keeper sign table vs the locked `NAMED_KEEPER_PRIOR`, Layer-4 anchoring) from those
  re-materialized values, and **verifies the sign-off gates still hold** — the ICC gate still fires
  (power 1.0), the named-keeper prior (Alisson/Neuer → deterrent) is still met. A gate that *flips*
  under the new weights is a finding surfaced to the owner, not silently re-baselined.
- Both re-run artifacts + the sign-off verification ride **Commit 2** (they derive from Commit-1 code +
  the new weights). This is a **coordination point** with the parallel TF-19 workstream (per the owner
  ruling that another session's inflight work is not a constraint — the artifacts are re-produced here
  from the authoritative new weights; the plan records the coordination).

Also downstream (unchanged): **`add_ghost_gk` / `compute_ghost_gk` / `ghost_gk_xfns` outputs change** →
any consumer persisting `ghost_gk_x/_y` re-materializes; `ghost_gk_xfns` is in **no** default xfn list
(CLAUDE.md), so **no VAEP retrain**.

**Honest magnitude note:** keepers cluster near goal-center (y≈34), so the practical change to GK
y-predictions is likely *small* — the blast radius (re-mat, TF-19 re-run, re-publish) may exceed the
measurable effect. The retrain is justified on correctness/consistency (orientation invariance), not
on an expected accuracy jump; the re-fit report records the measured CV delta, and the TF-19 re-run
confirms the sign-off is robust to it.

---

## 6. Piece 3 — Complete Layer-3 counterfactual engine + arms

### 6.1 `restdefense/_counterfactual.py` — a gkdv sibling

gkdv's `build_ghost_frames` ghosts the **defending** keeper near the **attacked** goal (verified
`_engine.py`), the opposite geometry to rest defense (which ghosts the **in-possession** team A's own
rearguard near **A's own** defended goal). So this is a **sibling engine**, not a gkdv call; only
gkdv's generic **delta** seams and **verdict** functions are reused.

```python
build_restdefense_ghost_frames(
    frames, *, which: Literal["keeper", "rearguard"], model, home_team_id,
    carrier=None, params: RestDefenseParams = _DEFAULT,
) -> tuple[cf_frames, provenance, RestDefenseGhostReport]
```

- **PURE** — never mutates `frames`; returns the full input with only the substituted rows' `x/y`
  overwritten, preserving row order + `player_id` (so gkdv's `_assert_legs_aligned` on
  `(game,period,frame,player_id)` passes).
- **Domain** (drop-and-counted; conservation `n_frames_scored + Σ drop_reasons == n_frames_in`, the
  `RestDefenseGhostReport` field names mirroring `GkdvReport`): alive ball; in-possession team A
  resolved (a shared carrier pin — inferred once, shared by the domain filter and the serve, the gkdv
  §4.2 discipline); **committed-forward** gate (ball ≥ `params.min_ball_advance_m` from A's own goal
  toward B's goal); the required substitution set present with finite coords (A's keeper for
  `which="keeper"`; ≥`n_rearguard` A field defenders for `which="rearguard"`); `GoalMap` resolvable.
- **Substitution:**
  - `which="rearguard"` → `serve_ghost_outfield_positions(frames, model=model, home_team_id=…,
    carrier=…, n_rearguard=params.n_rearguard, visible_area=…)`, moving A's deepest-`n_rearguard`
    field defenders. Matched on `(game,period,frame,team_id=A,player_id)` via the serve's `player_id`.
  - `which="keeper"` → `serve_ghost_gk_positions(frames, model=model, home_team_id=…, carrier=…)`,
    moving A's keeper. `model` must be a **sweeper** variant (`GhostGkModel.from_variant("sweeper"…)`,
    per CLAUDE.md — never the frozen default), because the in-possession keeper can sit at the 30–45 m
    an advanced sweeper occupies, out of the default model's 30 m label ceiling (§ parent spec §9).
  - The write-back reads the serve's **frame** `ghost_x/_y` (§4) — no reflection math in restdefense,
    no convention-drift risk.
- **Missing/NaN ghost** (incl. serve `variant_unavailable` / `fov_cropped`) → **dropped-and-counted**,
  never Δ=0. A **non-finite** served ghost on a scored frame → **raise** (pitch control silently drops
  NaN-coordinate rows, so a NaN ghost would make the player vanish rather than error — the exact gkdv
  guard, `_engine.py:574`).

### 6.2 `restdefense/_arms.py` — the arms

Two public functions, each parameterized by its ghost model, returning a **separate keyed arm table**
(key = the samples key `(game_id, period_id, team_id, action_id)`) so a caller left-joins onto the
Layer-1/2 samples:

```python
rest_defense_gk_deterrent(
    actions, frames, *, xt, ghost_gk_model, home_team_id,
    goal_map=None, links=None, carrier=None, visible_area=None, params=_DEFAULT,
) -> tuple[pd.DataFrame, RestDefenseGhostReport]        # rd_gk_deter_threat/_space (+ source)

rest_defense_outfield_deterrent(
    actions, frames, *, xt, ghost_outfield_model, home_team_id,
    goal_map=None, links=None, carrier=None, visible_area=None, params=_DEFAULT,
) -> tuple[pd.DataFrame, RestDefenseGhostReport]        # rd_outfield_deter_threat/_space (+ source)
```

- **Grain bridge.** The arm reuses restdefense's own window-selection seam (the one
  `compute_rest_defense` uses, in `_windows.py`), so its key set is a declared **subset** of the
  samples keys. It builds ghost frames for the linked frames, calls the gkdv batch seams with
  `attacking_team_id_by_frame = B` (a per-frame Series mapping each scored frame to A's opponent),
  and maps the per-`(game,period,frame)` delta Series back to the per-action arm table via the
  action↔frame link. (A frame carries one in-possession team, so B is unambiguous per frame; two
  sampled actions never map to one frame with different in-possession teams.)
- **Reuse, not reimplement** (parent spec §19): `delta_threat_suppression_batch` (threat) +
  `delta_das_batch` (space), both `negative = deterrent`. A `GkdvParams()` is constructed to supply
  the pitch-control method / `lambda_gk` (both legs identical → no drift).
- **The outfield arm STRUCTURALLY isolates the rearguard, but is NOT exactly keeper-*value*-invariant**
  (corrected 2026-09-07, owner-ratified; ADR-089 FINDING-2). The outfield ghost repositions only A's
  deepest-`n_rearguard` field defenders and **never moves A's keeper**, so the arm attributes no keeper
  repositioning to the rearguard — that structural isolation is exact. It does NOT follow that the
  keeper "cancels in the delta": `delta_threat_suppression` carries `lambda_gk` (A's keeper as a TTI
  control agent) and pitch control is **nonlinear**, so the fixed keeper interacts differently with the
  DIFFERING rearguard across the two legs and its net contribution does not subtract out. Measured
  (toy fixture, after the opponent-selection fix): moving A's keeper shifts the outfield **threat** arm
  ~3.6%; the DAS/space arm is keeper-blind-generic and near-invariant (~1%). The whole-team delta is
  therefore **rearguard-dominated**, not keeper-free. *(The revision-2 draft here claimed "the keeper's
  contribution cancels — a property, not a coincidence"; that was an over-claim, proven false by
  execution — its mandated exact-equality guard test only passed under a since-fixed opponent-selection
  bug. The shipped guard is the exact STRUCTURAL isolation — the ghost never touches the keeper — plus
  the honest space-is-near-invariant / threat-is-keeper-sensitive directional test.)*
- **"Behind the line" is satisfied by construction, not by a zone restriction.** The threat arm uses
  `compute_threat_pc` with xT toward A's own goal `G_A`, which concentrates in the zone behind A's
  line for free. The space arm uses `delta_das_batch`, whose DAS is *dangerous* accessible space
  (already goal-concentrated toward A's goal); and because only A's rearguard (or keeper) moved, the
  whole-team ΔDAS is rearguard-dominated. This reconciles the parent spec §7.3 "behind the line"
  wording with the seam-reuse mandate — no separate zone-restricted DAS (which §19 rejects).

### 6.3 Provenance columns

Each arm table carries **one shared** `<arm>_source` (`rd_gk_source` / `rd_outfield_source`) over the
closed vocabulary `{computed, ghost_missing, unlinked, unresolved, fov_cropped}` — the *domain*
provenance, shared by both sub-arms of that intervention (they share the same ghost/domain/link
state). The space arm's **velocity-NaN is not a per-row token**: within a match the DAS-scoreability
is near-constant (all rows scoreable, or all NaN on a velocity-less provider), which is the shape
`schema.py` rejects as a constant column. Per ADR-054 (value-changes → column, interpretation-changes
→ diagnostic), the space arm's velocity-NaN is self-describing (the value is NaN) + `validate_velocity_regime`;
the source column stays the domain provenance that genuinely varies per row.

### 6.4 `merge_rest_defense` + rollup + reports

- `merge_rest_defense(samples, *arms) -> pd.DataFrame` — left-joins the arm tables onto the Layer-1/2
  samples (honest-NaN on arm-dropped rows), asserts each arm's keys ⊆ sample keys, and reconciles
  drop-conservation across `RestDefenseReport` (samples) and each `RestDefenseGhostReport` (arm) — an
  arm-dropped row is counted once in the ghost report, never double-counted.
- `summarize_rest_defense(samples, by=…)` gains the ability to **mean the arm columns** when they are
  present in the (merged) table — the coach-facing per-`(team, match)` deterrent — a purely additive
  rollup extension (Layer-1/2 aggregation byte-identical).
- `RestDefenseGhostReport` (new): `params`, `n_frames_in`, `n_frames_scored`, `drop_reasons`
  (field names mirroring `GkdvReport` / `RestDefenseReport`). Conservation is a **CI gate**, not a
  dataclass property (as in gkdv / the existing `RestDefenseReport`).

---

## 7. Piece 4 — Instrument-validity probe (`restdefense/_probe.py`)

Mirrors `gkdv/_probe.py` (TF-19 A+2, ADR-082), **reusing gkdv's generic public verdict functions** and
adding the restdefense-specific dose imposers (the two interventions' geometry differs from gkdv's
defending-keeper geometry, so the imposers are new; the verdicts are generic and imported):

- **Reused from `silly_kicks.gkdv` (public `__all__` — the two verdict FUNCTIONS only):**
  `layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain)` and
  `layer1_responsiveness_verdict(*, gk_med, nd_med, placebo_p95, n_domain)` — both operate on
  already-**pooled** corpus statistics (verified generic + public: `_probe.py:195,270`;
  `gkdv/__init__.py __all__:56-57`). The thresholds `SATURATING_MULTIPLE` / `MIN_DOMAIN_FRAMES` /
  `PHYSICS_ARM_PROBE_RATIO` are **private module constants inside `gkdv/_probe.py`** (NOT in gkdv's
  public `__all__`), and the verdict functions read them **internally** — so restdefense reuses the
  functions and thereby inherits the TF-19-established physics-arm thresholds **without importing any
  private constant** (which would trip the §10 import-allowlist gate). restdefense does not reference
  those constants at all; if a restdefense-specific threshold is ever needed it would declare its own
  and reimplement the tiny verdict, but v1 reuses the functions as-is.
- **New in restdefense:** `impose_rearguard_dose(frames, *, which, home_team_id, dose, displacement=…,
  model=…, params=…)` — substitutes only A's rearguard (or keeper) at an imposed position, reusing the
  §6.1 engine's domain/provenance so the scored set matches `build_restdefense_ghost_frames` exactly
  (the gkdv `_probe._build_dose_targets` discipline). `paired_vector_controls` for the rearguard
  intervention displaces A's other outfielders by the same per-frame vector (a rearguard-slot analog of
  the gkdv single-player control); the outfield intervention displaces multiple slots, so the dose /
  control structure is defined per §7.1.
- **restdefense-local `EXPECTED_DIRECTION`** for its 4 arm columns (the gkdv `EXPECTED_DIRECTION` maps
  gkdv arm names; restdefense declares its own, reusing `expected_direction_for_arm`'s form).

### 7.1 Pooled verdicts + pre-registered anchors (owner ratifies in `/review-spec`)

- **Instrument validity (Layer 0):** for each of the 4 arms, `instrument_valid` / `instrument_void` /
  `arm_unscoreable`, computed as a **pooled-corpus** statistic (never per shard — the TF-19 lesson).
  Expectation (to be confirmed by the run, not asserted): outfield ΔDAS + both threat arms
  `instrument_valid`; keeper ΔDAS likely `arm_unscoreable` / weak (the TF-19 finding), shipped with the
  honest verdict (D2).
- **Responsiveness (Layer 1):** does the arm move under an imposed rearguard/keeper dose more than
  under single-player placebo controls (`gk_med ≥ ratio · max(nd_med, placebo_p95)`).
- **Pre-registered expected-sign anchor** (mirroring TF-19's locked `NAMED_KEEPER_PRIOR`; **the
  DIRECTIONS below are owner-ratified 2026-09-06 and LOCKED before any run** — the exact statistical
  test + effect-size/noise thresholds are finalized in the plan and ratified at `/review-plan`, but
  the directions cannot change after the run):
  - **Outfield arm (LOCKED):** possessions with a **stronger Layer-1 rearguard** (higher
    `rd_num_superiority` and/or a more-compact rearguard) score a **more-negative** (more deterrent)
    `rd_outfield_deter_*`.
  - **Keeper arm (LOCKED):** a **named high-line/sweeper prior** — aggressive sweeper-keepers score
    **more deterrent** — analogous to TF-19's Alisson/Neuer prior; the concrete keeper name list is
    locked in the plan (owner-injected, stamped into the artifact) before the run.
  The anchor is stamped into the artifact and checked confirmatorily; a non-decisive result reads
  "unvalidatable where the arm would matter," never "no value."

---

## 8. Piece 5 — Applied construct-validity report (Phase B)

- **Driver** `scripts/build_tf60_layer3_arm_values.py` — ADR-052 `for_each` sharding + ADR-037
  provenance (`require_clean_tree(git_provenance(), …)` in `main()`, `--allow-dirty`, `run_commit` +
  `run_tree_dirty` stamped; a declared `_EMITTED_SHARD_COLUMNS` + `_SHARD_SCHEMA_VERSION` pair pinned
  together per the 4.77.1 stale-shard rule). It runs both arms over the public corpus → a per-`(team,
  match)` arm-values table (mirroring `scripts/build_gkdv_arm_values.py` /
  `scripts/build_tf19_instrument_responsiveness.py`). ASCII-only source (the `test_driver_source_is_ascii`
  rule). Registered in `tests/scripts/test_provenance_wiring.ARTIFACT_DRIVERS` and the ADR-056
  `_script_population` gate.
- **Report** `docs/research/tf60_layer3_construct_validity/findings.md` — per-arm Layer-0/Layer-1
  verdicts + the pre-registered expected-sign check + an honest construct-validity note (a deterrent
  metric is a descriptive lens, not a validated predictor — the xtgk-v2 pattern). **Reported, never
  gated.** Produced from the Commit-1 clean SHA using the freshly-trained both-axes ghost-GK weights +
  the bundled ghost-outfield weights (its provenance chain is Commit-1 all the way down).

---

## 9. Cycle structure (one branch, provenance-mandated two commits)

One feature branch off `main`. Minimal commits; the **only** split is the one clean provenance forces
(the PR3/PR5 precedent):

- **Commit 1 (code, `training_commit`-clean):** everything except weights + the applied report — the
  serve / gkdv / ghost-GK-convention changes, the restdefense Layer-3 modules + probe, the corpus
  driver, all CI gates, and **all release docs that do not depend on a run** (this spec, the plan,
  ADR-089, CHANGELOG entry, version bump, CLAUDE.md, glossary, NOTICE, C4 if needed, the parent-spec
  §17 arc update). Design + release docs ride Commit 1 (no standalone doc commit). **Commit 1 is a
  clean *training* SHA, not an independently-green CI state:** the both-axes code changes both
  fingerprints, so the 4 bundled ghost-GK variants are transiently *red on load* between Commit 1 and
  Commit 2. That is harmless under branch-tip `--merge` gating — GitHub's required checks gate on the
  **head SHA (Commit 2)**, which is green once the re-fit weights are bundled; CI never sees Commit 1
  in isolation when both commits are pushed together. Do not read "Commit 1" as an independent gate.
- **DGX (Phase B), from Commit-1 SHA:** (a) re-train all 5 ghost-GK variants (both-axes) → weights;
  (b) run the Layer-3 arm-values corpus pass + the probe → the construct-validity artifacts; (c)
  **re-run `build_gkdv_arm_values.py` + the TF-19 sign-off** from the new weights and verify the
  sign-off gates hold (§5.4). All `training_commit` / `run_commit` = Commit-1 SHA (clean).
- **Commit 2 (weights + artifacts):** the **4 bundled** re-fit ghost-GK weight files (`full` is
  HF-published, not committed to the wheel) + regenerated goldens / chirality / feature-contract /
  `SHA256SUMS`, the HF cards, the construct-validity report, and any release-doc lines that cite
  measured numbers.
- **Merge with `--merge`** (never squash) to preserve the two-commit `training_commit` (squash orphans
  Commit 1 → a dangling `training_commit`). One version bump, one tag, HF re-publish of the 5 variants.

Each step — Commit 1 / DGX run / Commit 2 / push / merge / tag / HF publish — is a **separate
owner go-ahead**. The author does **not** run `/review-*` on their own work; the owner starts the
independent `/review-spec`, `/review-plan`, `/review-impl` sessions.

---

## 10. Validation (CI gates + e2e)

Methods CI-gated; applied results reported-not-gated (repo convention).

- **Frame-coords parity** (§4.2): for a fixed model, the serve's `ghost_x/_y` equals the physical
  position the old path produced — the refactor alone is byte-identical.
- **Ghost-GK retrain gates:** regenerated golden / chirality (now x+y) / feature-contract; the
  integrity-on-load gate covers the **4 bundled** variants (offline CI has no network — `full` cannot
  load offline; its integrity is covered at publish / a network-gated test, and the existing offline
  `full` path is an ImportError mock, `test_ghost_gk.py:447`); `position_only` round-trips; the
  both-axes convention is proven by the strengthened chirality y-leg (§5.2) — the load-bearing
  non-vacuity assertion that the flip took (a scene and its 180° reflection score consistently; a
  mutation that should break it does).
- **TF-19 sign-off robustness (Phase B, reported-not-gated):** re-run the ICC gate + per-keeper sign
  table + Layer-4 anchoring from the re-materialized gkdv arm-values (§5.4); confirm the ICC gate still
  fires (power 1.0) and the locked `NAMED_KEEPER_PRIOR` still holds. A gate that flips under the new
  weights is surfaced to the owner, never silently re-baselined.
- **Counterfactual non-vacuity + mirror-invariance** (the CLAUDE.md "both-sides / non-vacuity"
  discipline): the ghost leg *measurably differs* from its factual twin (mirroring
  `tests/gkdv/test_arms.py::test_unpinned_implementation_would_measurably_differ`); a two-sided band
  test; and — the write-back landmine — **mirror the frames, the arm deltas are invariant** (catches a
  wrong y-flip in the serve→frame conversion).
- **restdefense method gates** (extend the existing suites): liveness (every emitted arm column
  non-NaN + non-constant on a multi-domain fixture); purity (ADR-033: arms never mutate inputs; a 2nd
  variant for the `visible_area` branch); id-dtype invariance (ADR-019); D3 / orientation (ADR-051:
  direction never from team identity; mirror-invariant action-LTR geometry); FOV completeness
  (ADR-077); SB360 audit (ADR-053: the arms are boundary entries → per-column verdict +
  `verdict_provenance`; velocity-less-degrading columns `honest_nan`/`differs_by_design`, never
  `silent_degrade`); glossary (ADR-048: +4 arm columns, companions glossary-exempt); import allowlist
  (restdefense imports gkdv/tracking public seams only — the probe reuses gkdv **public** verdicts, no
  private import); sub-quadratic growth (ADR-073) for any new looping primitive.
- **Probe gates** (mirror `tests/gkdv/test_probe_*`): the Layer-0 discrimination non-vacuity (the void
  `and` is load-bearing), the paired-control distinctness, the pooled-not-per-shard property.
- **`@e2e` method gate** (owner/fixture-gated): run both arms on ≥1 real linked-tracking match
  (native keeper identity, full-coverage FOV) and ≥1 real SB360 match (roster keeper identity, FOV
  companions with `<1.0` observed fraction on a cropped advanced-ball frame). Asserts the **method**
  (non-empty, conservation reconciles, FOV companions populated) — not metric values.

---

## 11. Complete artifact set (enumerated up front — the recurring lesson)

**Code:** `_ghost_gk.py` (both-axes extractor + convention metadata), `_ghost_outfield.py`
(serve `ghost_x/_y`), both serves; `gkdv/_engine.py` (consume frame coords); `restdefense/`:
`_counterfactual.py`, `_arms.py`, `_probe.py`, `__init__.py` exports (+ `build_restdefense_ghost_frames`,
`rest_defense_gk_deterrent`, `rest_defense_outfield_deterrent`, `merge_rest_defense`,
`RestDefenseGhostReport`), `_columns.py` (+4 arm columns + source vocab), `_compute.py`
(`summarize_rest_defense` arm rollup). **Scripts:** `scripts/train_ghost_gk.py` (retrain — likely
unchanged beyond the extractor), `scripts/build_tf60_layer3_arm_values.py` (new driver),
`scripts/build_gkdv_arm_values.py` + `scripts/build_tf19_instrument_responsiveness.py` (re-run from
the new weights, §5.4 — likely unchanged, just re-executed), `scripts/_hub_publish` re-publish path. **Bundled weights (Commit 2):** the **4** bundled re-fit ghost-GK variants (`default` /
`position_only` / `sweeper` / `sweeper_position_only`) + regenerated
goldens/chirality/feature-contract/`SHA256SUMS`. **HF:** re-publish **all 5** GK variant repos with
cards (incl. the Hub-only `full`) through the ADR-088 card-required seam.
**Tests:** the §10 gates + registry edits (glossary, SB360 audit registry, id-scalar registry if a
new public id-scalar function is added, provenance-wiring `ARTIFACT_DRIVERS`, ADR-056 script
population, C4 completeness if touched — expected untouched). **Docs:** ADR-089, CHANGELOG, version
bump (one file, ADR-079), CLAUDE.md contract updates (restdefense Layer-3 + the ghost-GK convention
unification + the serve-frame-coords contract), NOTICE (Kim 2026 DEFCON, Le 2017, Bischofberger & Baca
2026), the parent TF-60 spec §17 arc update, this spec + the plan, the `docs/research/` construct-
validity report (Commit 2). **Re-run artifacts (Commit 2, §5.4):** the re-materialized gkdv
arm-values table + the re-run TF-19 sign-off artifacts (ICC gate, per-keeper sign table, Layer-4
anchoring) + a short note recording the sign-off is robust to the retrain (or flagging any gate that
moved). **C4:** unchanged (no new container/aggregator) — verify the completeness gate stays green.

---

## 12. Provider / velocity / FOV handling

- **Velocity tiers (ADR-063):** threat arms are T1 (positional, computed on velocity-less SB360 via
  the zero-velocity model); space (ΔDAS) arms are T-vel (honest-NaN on SB360). Handled by the gkdv
  delta seams (`delta_das_batch` → NaN on velocity-less; `delta_threat_suppression_batch` always
  scores).
- **FOV (ADR-077):** the serve already returns `fov_cropped` NaN when A's rearguard region is under-
  observed → dropped-and-counted. Count/region companions on the samples table are unchanged (Layer-1/2);
  the arm tables carry the shared source vocab incl. `fov_cropped`.
- **Keeper identity (ADR-078):** `native` for continuous tracking; `roster` +
  `apply_keeper_identities_to_frames` for anonymous SB360 (the keeper arm needs the resolved keeper on
  the numbered freeze-frame rows).

---

## 13. Rejected alternatives

- **Leave the two ghost conventions inconsistent (serve owns the conversion, no retrain).** Rejected by
  the owner (gold-standard): a lingering x-only GK model that is not y-chirality-invariant is a latent
  trap once every consumer routes through the serve. (This *was* offered as an option; the owner chose
  unify-now.)
- **Zone-restricted DAS for the space arm.** Rejected: reimplements gkdv's DAS direction-pin/identity-
  cache differencing (a documented silent-bug seam, parent §19); and DAS is already danger-weighted so
  the whole-team ΔDAS is behind-the-line-dominated.
- **A per-row velocity token on the space arm.** Rejected (ADR-054): near-constant per match → the
  shape `schema.py` rejects; `validate_velocity_regime` is the diagnostic.
- **Stub `which="keeper"` (keep PR4 separate).** Rejected by the owner (fold PR4 in).
- **Bare expected-sign construct-validity.** Rejected (D5): the TF-19-A+2 instrument/responsiveness
  rigor is the gold standard for a counterfactual metric.
- **Squash-merge.** Rejected: orphans the `training_commit` (must `--merge`).

---

## 14. Open questions / owner-ratification items

1. **Pre-registered expected-sign anchors** (§7.1) — **RESOLVED (owner-ratified 2026-09-06):** the two
   DIRECTIONS are locked (outfield: stronger rearguard → more deterrent; keeper: sweeper-keeper → more
   deterrent). The exact statistical test + thresholds + the concrete keeper name list are finalized in
   the plan and ratified at `/review-plan`, stamped into the artifact before the run.
2. **Committed-forward gate default** (`min_ball_advance_m = 52.5`) — **status-quo, not new scope**:
   this gate + default already shipped in Layers 1-2 (`_config.py:51`, `_windows.py:151`), with
   calibration deferred (empty `for_provider`). The arms inherit it unchanged; sensitivity reported.
3. **TF-19 sign-off re-run ownership/sequencing** — **RESOLVED (owner-directed 2026-09-06):** the
   gkdv re-materialize + TF-19 sign-off re-run are **pulled into this cycle's Phase B** (§5.4/§9);
   the plan details the tasks + the coordination point with the parallel TF-19 workstream.
4. **`GEOMETRY_VERSION`** — current reading is the ghost-GK `to_gr_*` helpers are confined to
   `_ghost_gk.py` (module-level after the §5.2 hoist), not the shared `_geometry` module, so no bump;
   review confirms no shared-`_geometry` path is touched (§5.2).
5. **Construct-validity magnitude** — the keeper y-change may be small (§5.4); the re-fit report records
   the measured CV delta so the cost/benefit is visible.
