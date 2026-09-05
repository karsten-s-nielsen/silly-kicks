# TF-60 PR5 — Ghost-Outfield Model (Rest-Defense Rearguard Positioning) — Design

**Status:** Draft for independent review (2026-09-03).
**Version / PR / ADR:** *unassigned.* The version bump, `PR-Snnn`, and `ADR-nnn` are taken from `main`
only at the commit gate at the end of the cycle — no number is claimed before then.
**Parent spec:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md`
(the ghost-outfield primitive; §12 API, §16 sequencing).
**Author-review note:** this spec is written by the implementing session; per standing rule the
**independent `/review-spec` is run by a separate session the owner starts** — the author does not
review their own work.

---

## 1. Executive summary

PR5 ships **one new trained-model primitive** — `silly_kicks/tracking/_ghost_outfield.py`, a
**league-average rearguard-positioning model** that mirrors `GhostGkModel` (TF-18; trained-model
discipline **ADR-011 / 016 / 040 / 044 / 050 / 076** — *not* ADR-083, which is only the ghost-GK
sweeper re-fit and whose extended grid §4 explicitly scopes out) but predicts
**individual rearguard outfield players** instead of the goalkeeper. It exists to feed **PR6's outfield
counterfactual arm** (`which="rearguard"`): substitute a team's actual rearguard with league-average
"ghost" positions and difference the pitch-control / accessible-space fields (`ΔDAS`, `Δthreat`) to
value the rearguard's *positioning*.

It is **additive** — no existing feature changes, **no VAEP retrain**, in **no** default xfn list — and
it ships the model + its served seam + bundled weights + HF publish, **not** any counterfactual arm
(that is PR6).

**Sequencing — PR5 before PR4 (owner-directed; amends the parent arc).** The parent spec §17 orders the
remaining arc **PR4 (GK arms) → PR5 (ghost-outfield model) → PR6 (outfield arm)** and states each cycle
"depends only on the prior cycle's public surface." This sub-spec runs **PR5 before PR4**, on the
owner's explicit direction: after asking whether PR5/PR6 could precede PR4 "if more clearly defined,"
the owner directed *"We will do PR5 in this cycle."* That decision is the authorization; sequencing an
approved arc is the owner's call, and this records it.

- **Dependency-safe.** The real dependency DAG is `PR3 → PR4` (GK arms consume the shipped re-fit) and
  `PR5 → PR6` (the outfield arm consumes this model), with **PR4 ⊥ PR5** (independent). The parent's
  "each depends only on the prior cycle" (§17 line 631) was a simplifying statement of a *linear*
  order, not a true dependency constraint; PR5-before-PR4 violates no real dependency (the independent
  review concurred).
- **Technical motivation (not the authorization).** The TF-19 A+2 corpus run (4.104.0) measured `ΔDAS`
  as a *weak* keeper-deterrence instrument — *"accessible space is outfield-dominated"* — which is
  exactly why the **outfield** `ΔDAS` arm is the instrumentally-**soundest** part of the arc, and the
  Gradient Sports 27.5 m keeper-clamp (ADR-083, `docs/research/gs_keeper_clamp/`) is **keeper-only**, so
  the outfield corpus is clean. **The TODO TF-60 row records this analysis but explicitly tags it
  "informs the brainstorm, NOT decided scope; validate by measuring"** — so the TODO is the *evidence*
  behind the motivation, **not** the sequencing authorization. (An earlier draft mis-cited the TODO as
  authorizing the reorder; it does not — the owner's direction does.)
- **Recording obligation.** As part of this cycle's docs, the parent spec §17 arc table and the TODO
  TF-60 row are updated to reflect the PR4↔PR5 swap, so the durable record matches the owner's decision
  at the commit gate. Until that lands, this section is the record.
- **Owner confirmation (2026-09-04).** In response to the independent spec review that flagged the
  reorder, the owner **explicitly confirmed** both (1) PR5-before-PR4 is the approved sequencing and
  (2) reconciling parent §17 + the TODO TF-60 row at this cycle's Phase-B doc commit (not a standalone
  commit) is the correct handling.

---

## 2. Scope — the complete artifact set

PR5 delivers **all** of the following as one cycle. Phase B lands in **two owner-approved commits**
(code → then weights), exactly like PR3 — honest weight provenance requires training from an
already-committed clean code state, so a single commit is impossible (see §14):

1. `silly_kicks/tracking/_ghost_outfield.py` — the `GhostOutfieldModel` class (mirrors `GhostGkModel`).
2. `serve_ghost_outfield_positions(...)` — the public `tracking` seam PR6 consumes (mirrors
   `serve_ghost_gk_positions`).
3. `scripts/train_ghost_outfield.py` — the trainer (mirrors `scripts/train_ghost_gk.py`).
4. `scripts/publish_ghost_outfield.py` — the HF publisher (mirrors `scripts/publish_ghost_gk.py`;
   reuses `_hub_publish.upload_model_only`'s allowlist leak-guard).
5. **Bundled weights** `silly_kicks/tracking/_ghost_outfield_weights/{default,position_only}/` (npz +
   metadata.json + SHA256SUMS), trained on the DGX from a clean, CI-green commit.
6. **HF Hub repos** `silly-kicks/ghost-outfield-v1` + `silly-kicks/ghost-outfield-position-only-v1`,
   **with model cards** — and **the in-repo model cards ship in the release commit** (the PR3 lesson:
   a model card documents a shipped artifact and is never a post-merge afterthought).
7. Tests (§13).
8. Docs: **a new ADR** (next-free number, taken at ship), CHANGELOG, CLAUDE.md (a GhostOutfield
   contract bullet), `NOTICE` (attribution — §12). **Plus the §1 recording obligation: reconcile the
   parent spec §17 arc table to the PR5-before-PR4 order, and swap the TODO TF-60 row's PR4↔PR5
   entries** (in addition to marking PR5 shipped). Both land in this cycle's Phase-B doc commit.
9. C4: **+0 action-coupled aggregator** (no `add_ghost_outfield`; see §10). The `tracking` container's
   DSL description may need a one-line model-list touch — verify the C4 completeness gate.

---

## 3. The target — individual rearguard players, lateral-slot-keyed

**Target granularity: individual players (decided).** The model predicts a **single rearguard player's
league-average `(x, y)`**, not a line/shape aggregate. Rationale: the novel contribution is
positioning *attribution* (which defender is out of position); the counterfactual physics (`ΔDAS` /
`Δthreat`) is built from *individual* positions regardless, so a line/shape ghost would have to
synthesize individuals anyway; and it mirrors the established defensive-ghosting framing (Le 2017;
DEFCON-GNN) and the per-agent `GhostGkModel`.

**Slot assignment: deterministic lateral rank (decided).** Per frame:
- The rearguard = A's deepest `n_rearguard` (default **4**, tunable) outfield players, selected by
  **`select_back_line_players`** (TF-14), oriented by **`GoalMap` / `resolve_defended_goals`** (ADR-055
  — *never team identity*; ADR-051-D3 direction is a value, not a default).
- Sort them by lateral coordinate in the oriented (goal-relative / action-LTR) frame → **slot `1..n`**
  (goal-relative-left to -right). The slot index is a **numeric** model feature (like `GhostGkModel`
  trains `phase` numerically, so the pickle-free boosted traversal matches sklearn — ADR-016).

Why lateral rank, not detected roles: TF-39 `infer_positions` is documented-unstable over short
windows, and the true role-alignment gold standard (template + Hungarian, the **TF-51 Track-B**
machinery) is **not shipped** — depending on it would make PR5 non-self-contained. Lateral rank is
deterministic, per-frame (no cross-frame identity needed), and formation-agnostic (the deepest-`n`
*is* the rest-defense rearguard). A detected-role **feature** is deferred (§5), never a dependency.

**Rearguard size: fixed `n_rearguard` (default 4, tunable serve parameter) — decided.** `n` is **fixed
at serve, not adapted per frame** (this resolves the parent §17 open item on variable rearguard size).
Consequences, recorded and accepted:
- **back-4:** the deepest-4 is the back line exactly.
- **back-3:** the deepest-4 is the 3 centre-backs + the deepest covering midfielder — which *is* the
  rest-defense rearguard (the players actually held back), so this is correct by design, not a defect.
- **back-5:** the deepest-4 is the 3 central defenders + the deeper wing-back; the higher wing-back
  (5th-deepest) is excluded, which is acceptable — the higher wing-back is the least central to
  counter-cover and is typically part of the attack, not the rest defense.

Fixed `n` is chosen over adaptive `n` for three reasons: (1) **PR6 needs the same `n` on both legs** —
the actual and ghost rearguards must have identical slot cardinality for the per-slot `actual − ghost`
match, and an adaptive `n` could differ between the factual frame and its ghost; (2) formation
detection (TF-39) is unstable over the windows a batched serve sees; (3) "deepest-`n`" is
formation-agnostic by construction. A caller studying back-5 systems can raise `n_rearguard` to 5.

**Counterfactual matching (how PR6 will consume this).** The slot is deterministic on **both** legs,
so PR6 substitutes *actual slot-K → ghost slot-K* — per-player, unambiguous, no assignment step at
differencing time. (PR6 itself is out of scope; this only records the contract the model must satisfy.)

**Orientation contract.** The model is trained and predicts in the **oriented goal-relative frame**
(A's defended goal fixed); `serve_ghost_outfield_positions` re-projects to raw frame coordinates for
the caller (mirroring how `serve_ghost_gk_positions` handles orientation — ADR-028/051-D3/055). No raw
team-identity orientation anywhere.

---

## 4. Model architecture — mirror `GhostGkModel`

`GhostOutfieldModel` reuses the `GhostGkModel` machinery **verbatim in spirit** (share code where the
class boundary allows; do not fork the numba kernel or the load-guards):
- **Estimator:** two `HistGradientBoostingRegressor` ensembles (x and y), served as the **exact
  pickle-free boosted mean** (`baseline + Σ_trees leaf_value`), no sklearn at inference. Default
  hyperparameters mirror ghost-GK: 500 trees, max depth 8, 5-fold CV. **numba leaf walk** reused
  (ADR-076) — bit-identical with the numpy fallback.
- **Serialization:** parameters-only npz + JSON metadata + SHA256SUMS; **pickle-free**;
  `stores_training_data = false` (ADR-044). **Chirality** + **feature-contract** load-guards
  (ADR-011/016/040/050): `load()` re-runs the model's own outputs on a canonical probe frame and
  raises on mismatch, and verifies the extractor's feature fingerprint + declared geometry constants.
- **No grid / no KDE density.** Unlike `GhostGkModel`, PR5 needs **only the mean serve** — there is no
  `predict_density` and no `GhostGridSpec` (the ghost-GK grid existed for the KDE density and the
  keeper's 30 m label cap). The rearguard has **no restrictive label-domain filter**: a rearguard
  player's honest position anywhere is a valid label (contrast the ghost-GK's `[0, 30] m` box). This
  is a *simplification* relative to ghost-GK, recorded so a reviewer does not expect a grid.

---

## 5. Features — the frozen, leakage-safe vector

> **AMENDMENT (2026-09-04, owner-approved after the impl review).** The model is now
> **possession-conditioned**, not in-possession-only. The `team_in_possession` feature (below) was
> constant `1.0` under the original in-possession-only domain — a dead feature. Rather than delete it,
> the domain was widened to the gold-standard general design (matching ghost-GK, which conditions the
> keeper on possession): **training models BOTH teams' deepest-n per frame** — the ball-carrier's
> rest-defense rearguard (`team_in_possession=1`) AND the other team's defensive line facing the attack
> (`team_in_possession=0`) — so `team_in_possession` is a **live discriminator**, every feature carries
> signal, and the model is a reusable positioning primitive. **Rest defense (this PR's driver + PR6)
> serves the in-possession slice**, so the serve/PR6 API is unchanged. The extractor gains a
> `both_teams` flag (`True` for training, `False`/default for the in-possession serve). §6's corpus and
> the trainer report a **per-possession MAE breakdown** so the rest-defense (in-possession) quality is
> judged separately from the out-of-possession line. See decision **D11**.

Parent §17 delegated the leakage-safe feature list to this sub-spec, and the plan's leakage review needs
a concrete vector to bind to. The feature **set** (which columns exist) is **frozen here**; only the
numeric parameters inside a feature (zone boundaries, the `k` in a "`k`-nearest counter-threats"
centroid) are pinned when the extractor is written. The extractor is a **new `_ghost_outfield`
extractor**, not a re-use of the GK extractor — the rest-defense frame is *the opposite* of ghost-GK's:
team **A is in possession and attacking**, and the model predicts **A's own deepest defenders** (the
rearguard it keeps back), conditioned on where the ball is and where **B's** forward players (the
counter-threats) sit relative to A's defended goal.

**The leakage rule (the crux).** The target is slot-`K`'s `(x, y)`. **No feature may encode A's
rearguard coordinates or geometry** — the target set — because any *positional* rearguard summary
(line-x, deepest-defender, rearguard compactness/width) contains slot-`K` and would leak the label.
(The **`slot_index` is exempt**: it is the rearguard's lateral *rank* — an alignment/ordering key, not
a positional summary of anyone's coordinates — so it carries no coordinate information about slot-`K`.)
This is the one place the
outfield model *cannot* copy ghost-GK, which freely feeds the defending line's geometry because the
keeper is not part of that line. The model therefore conditions **only** on the ball, B's counter-threat
geometry, game context, and the slot index. It also does **not** feed the slot player's own
velocity — a "league-average given the situation" ghost must not be told which way *this* player is
moving (velocity is a position derivative that would bias the ghost toward the actual player). Velocity
features are strictly **situational** (ball + opponent-mass), exactly as ghost-GK's are.

**Faithful feature set (20 columns).** All geometry is goal-relative to **A's defended goal** (the
counter-attack target), oriented by `GoalMap`/ADR-055.

| Family | Feature | Velocity? |
|---|---|---|
| Ball state | `ball_x`, `ball_y` | — |
| Ball state | `ball_vx`, `ball_vy`, `ball_speed` | **vel** |
| Ball state | `ball_distance_to_own_goal`, `ball_to_own_goal_angle`, `ball_in_own_half` | — |
| Opponent counter-threat (B; leakage-safe, B ≠ target) | `opp_in_def_third_count` (B players in A's defensive third), `opp_deepest_x` (B's nearest-to-A's-goal player's x), `opp_forward_centroid_x`, `opp_forward_centroid_y`, `ball_to_deepest_opp_dist` | — |
| Opponent counter-threat | `opp_forward_centroid_vx` (counter-mass closing speed) | **vel** |
| Game context | `phase`, `team_in_possession`, `score_diff`, `time_seconds`, `period_id` | — |
| Slot | `slot_index` (lateral rank `1..n`, the multi-agent feature) | — |

**`position_only` (16 columns)** drops the 4 velocity features (`ball_vx`, `ball_vy`, `ball_speed`,
`opp_forward_centroid_vx`) — dropped, **not** NaN-filled (ADR-067; the feature contract raises on
non-finite). This is the SB360 freeze-frame serve variant.

**Escalation, recorded but not in v1.** If a *rearguard-shape* conditioning feature ("given my
teammates are here, where should I be") is later shown to lower held-out MAE, it MUST be computed
**leave-one-out** (excluding slot-`K`) to stay leakage-safe — and it is a feature-contract change
(a new probe fingerprint), so it is a deliberate v2 revision, never a silent add. A **detected-role
feature** (TF-39) is likewise deferred: it is position-derived and would have to be leakage-safe (from a
*prior* window, not the current frame's rearguard) — out of the frozen v1 vector.

`slot_index` and every id comparison use `id_compat` (ADR-019); NA-safe throughout (ADR-027).

---

## 6. Corpus & training

- **Corpus:** the same **179-match public corpus** the ghost-GK `default`/`full` used
  (`sk_stageB_448/ghost_cache`: Gradient Sports + IDSSE/Sportec + SkillCorner). **Gradient Sports is
  fully usable here** — the 27.5 m clamp is *keeper-only*; GS outfielders are tracked across the full
  pitch (verified, `docs/research/gs_keeper_clamp/`). So, unlike the sweeper (GS gave 0 % high-sweeper
  signal), the outfield model gets GS's full signal.
- **Training rows:** for each frame, **`n` rows** (one per rearguard slot); label = that slot's
  player's oriented `(x, y)`; feature = frame state + `slot`.
- **CV:** StratifiedGroupKFold by (match + provider), mirroring the ghost-GK trainer. Report overall +
  per-provider euclidean MAE; per-**slot** MAE (each slot's fit quality). Acceptance bars mirror
  ghost-GK (overall MAE, per-provider MAE, cross-fold std) — set with the corpus evidence, not
  pre-committed for the slot-MAE.
- **Provenance:** `training_commit` from a **clean, CI-green** tree (the ADR-063/`_provenance`
  discipline — no `--allow-dirty` for shipped weights); the corpus-driver seam (`scripts/_driver.py`)
  if the extraction is sharded.

---

## 7. Variants & velocity (ADR-067)

- **`default`** (faithful, velocity-bearing) + **`position_only`** (velocity features dropped, **not
  NaN-filled** — the feature contract raises on non-finite) for velocity-less SB360 freeze-frames.
- **Both variants train on the same 179-match continuous-tracking corpus** (§6); `position_only` simply
  drops the velocity columns from the extractor (the ADR-067 pattern), so it is **trained** on
  continuous tracking and **served** on SB360. SB360 freeze-frames are **not** in the training corpus —
  they are a serve-time target only.
- Velocity-keyed **auto-select at the serve seam** via the declared marker (`variant_key_for_velocity`
  + a `_resolve_*_model_for_frames` resolver), mirroring ghost-GK. A missing `position_only` → honest
  NaN, never the invalid faithful default (the ADR-067 asymmetry). A **mixed-availability** frame set
  RAISES.

---

## 8. SB360 / FOV honesty (ADR-077, ADR-063)

- **The deep rearguard is exactly what a ball-advanced broadcast FOV crops** (the parent spec's honest
  ceiling) — the moment rest defense matters most. So:
  - A rearguard slot whose player is **outside the visible area** on SB360 is **honest-NaN**, never a
    fabricated position (ADR-077); the served ghost for a missing actual slot is dropped-and-counted by
    the (PR6) consumer, never Δ=0.
  - `serve_ghost_outfield_positions` carries the **ADR-077 FOV companions** where a region/count is
    involved, and `validate_fov` remains the frame-set diagnostic.
  - `position_only` on velocity-less SB360; velocity-constitutive columns stay honest-NaN (ADR-063
    tiers).
- **SB360 boundary audit (ADR-053):** the model's serve seam registers an SB360 verdict + rationale
  like every other; velocity-invariant → `works`, cropped-rearguard → `honest_nan`.

---

## 9. Coherence — the conditional escalation

The one risk in independent per-slot prediction is an **incoherent ghost rearguard** (slots out of
lateral order, or overlapping). PR5:
1. **Builds the independent per-slot model** (no shape constraint by default).
2. **Measures ghost-rearguard coherence** on held-out frames: (a) the fraction of frames where the
   ghost slots preserve lateral ordering `1..n`; (b) the minimum pairwise ghost-slot distance
   distribution. Reported in the trainer `metrics.json`.
3. **Adds a lightweight shape constraint only if the measurement shows material incoherence** — e.g. a
   monotone re-sort of the ghost slots, or a minimum-separation prior. This is **YAGNI-gated**: the
   spec does *not* pre-build the constraint; it commits to *measuring* and to a named remedy if needed.
   The remedy, if triggered, is documented in the ADR before it ships.

A **non-vacuity gate** on the coherence metric: assert the metric can register incoherence (feed a
deliberately shuffled ghost set and confirm the ordering-fraction drops), so a green coherence reading
is not vacuous (the codebase's "test both sides of a band" discipline).

---

## 10. Surface & C4

- **Ship `serve_ghost_outfield_positions` only** — the `tracking` public seam PR6 consumes (the same
  role `serve_ghost_gk_positions` plays for gkdv). Plus `GhostOutfieldModel` (`from_variant`,
  `from_hub`, fail-closed `load`).
- **No `add_ghost_outfield` action-coupled aggregator** (+0 to the C4 aggregator count, stays 33). The
  model exists to feed PR6's counterfactual via the seam, not to emit a VAEP feature column; adding an
  aggregator would be a new default-surface + a VAEP-feature question with no consumer this cycle
  (ADR-009 — the library ships raw primitives, not composites without a consumer).
- **Cascade traps to check** (the PR2/PR3 lesson): any new **public** symbol → the **id-scalar
  registry** (`conftest_id_scalar.py`) if it takes an id scalar, **public-API-examples**
  (`_PUBLIC_MODULE_FILES` + example-or-`_EXAMPLES_DEBT` for every non-`_` def), and the **scoped
  pyright** gate (`test_pyright_clean_tracking_namespace` is stricter than bare pyright — verify
  against it). No new glossary columns (no aggregator) → no C4 feature-column-count bump.

---

## 11. Bundling & HF publish

- Bundle `default` + `position_only` under `_ghost_outfield_weights/`; `.gitattributes` `** binary`
  (the CRLF-vs-SHA trap).
- Publish to **per-variant HF repos** `silly-kicks/ghost-outfield-v1` +
  `silly-kicks/ghost-outfield-position-only-v1` via `publish_ghost_outfield.py` (round-trip verify;
  `upload_model_only` allowlist; **verify the Hub file count ~4–5**, never the whole-folder leak).
- **Model cards ship in the release commit** (in-repo `docs/huggingface/model-cards/`), uploaded to the
  Hub as READMEs. This is called out because it is exactly the artifact PR3 dropped.

---

## 12. Trained-model discipline & attribution (the contracts)

- **ADR-011/016/040/044/050:** parameters-only, pickle-free, fail-closed load (chirality +
  feature-contract), declared geometry constants, `base_score`-safe loader if xgboost is involved
  (it is not — HGBR).
- **Byte-identity discipline (the PR3 lesson, [[reference_trained_model_metadata_sha_never_roundtrips]]):**
  a full save→load→save metadata-SHA round-trip is **unachievable** (`save()` recomputes the
  feature-contract probe). Any byte-identity claim is proven field-level, not by a full-SHA gate.
- **ADR-076** numba leaf walk reused bit-identically (numpy fallback; lazy import).
- **ADR-019** id_compat for every id compare/key; **ADR-055** GoalMap orientation; **ADR-027**
  NA-never-a-0; **ADR-063** velocity tiers; **ADR-067** velocity-keyed variants; **ADR-053/054** SB360
  audit + velocity provenance; **ADR-068/073** no rescan-in-loop / sub-quadratic guard if the trainer
  loops over frames/slots.
- **ADR-005 attribution:** a `NOTICE` entry for the ghost-outfield model (Le et al. 2017 ghosting; the
  ghost-GK NOTICE entry covers the *concept* but the outfield model is a distinct artifact — add an
  entry). DEFCON-GNN (Kim 2026, arXiv 2512.10355) is a **comparator, not implemented**.

---

## 13. Testing

Mirror the ghost-GK test surface, plus the multi-agent specifics:
- **Trained-model gates:** golden / chirality / feature-contract load tests on the bundled weights;
  the parity gate (`predict_mean == sklearn` ≤ 1e-6) on a fresh fit; the three-way byte-identity
  discipline for any "unchanged" claim.
- **Load-guard non-vacuity (negative test):** the load guards must be shown to *fire*, not merely to
  pass — a positive-only load test is vacuous the same way an untested band is. Assert `load()` **raises
  the model's `IntegrityError`** when (a) a serialized weight is perturbed (chirality: the re-run
  outputs no longer match the recorded probe), (b) a declared geometry constant is changed
  (feature-contract), and (c) a `SHA256SUMS` entry is tampered. This mirrors the §9 coherence
  non-vacuity assertion and the codebase's "test both sides" discipline.
- **Slot/target:** a per-slot serve test (running the model per slot yields `n` distinct ghost
  positions); orientation invariance (the ghost is action-LTR; a mirrored frame + held `home_team_id`
  yields the mirrored ghost — the ADR-051-D3 direction-invariance idiom).
- **Coherence gate (§9)** with its non-vacuity assertion.
- **Velocity-keyed variant** resolution tests (faithful ↔ position_only; missing-position_only → NaN
  not default; mixed → raise).
- **SB360 boundary audit** entry (ADR-053) + a cropped-rearguard honest-NaN test.
- **Aggregator/liveness/purity/id-dtype-invariance** gates only insofar as a *public* symbol is added
  (no `add_*`, so no aggregator-liveness entry; but `serve_ghost_outfield_positions` + the model class
  need the public-API-examples + id-scalar registry accounting).
- **`@slow` trainer `main()` smoke** on a tiny committed corpus (skips on sklearn < 1.9; runs on CI's
  primary leg), asserting the extended fit + metrics blocks — the ADR-052 shard-token discipline if the
  extraction is sharded.

---

## 14. Phasing (mirrors PR3)

- **Phase A (local):** all code + tests against a **locally-fit toy** ghost-outfield; `default`
  byte-identity is N/A (new model), but the load-guards + parity + coherence gates pass on the toy;
  full suite + lint + scoped pyright green. **No commit** — no real weights yet.
- **Phase B (DGX) — TWO owner-approved commits (the PR3 provenance structure):**
  - **Commit 1 (code, clean, CI-green):** all Phase-A code + toy-validated tests, **no weights**. This is
    the exact tree the DGX trains from, so `training_commit` = Commit 1's SHA (honest; no `--allow-dirty`).
    A single commit cannot do this — the weights must be trained from an already-committed clean state, or
    `training_commit` stamps a dirty/dangling SHA (ADR-063 `_provenance`).
  - **DGX train** from Commit 1: fit `default` + `position_only` on the 179-match corpus.
  - **Commit 2 (weights + release):** bundle; regenerate per-artifact goldens; re-point the toy-based
    gates at the real weights (direction-not-magnitude); **write the model cards**; bump version + the new
    ADR + CHANGELOG + TODO + CLAUDE.md (incl. the §1 recording obligation).
  - **Merge with `--merge` (MANDATORY, never squash)** — a squash orphans Commit 1 and dangles the
    `training_commit` SHA the weights cite (the PR3 owner ruling).
  - Then tag / publish / HF-publish — **each a separate owner go-ahead**.

---

## 15. Out of scope (PR5)

- **PR6** the outfield counterfactual arm (`restdefense/_arms.py` `which="rearguard"`,
  `build_restdefense_ghost_frames`, `merge_rest_defense`) — a separate cycle that *consumes* this model.
- The **coherence shape-regularizer** unless §9's measurement triggers it.
- Any **role-detection dependency** (TF-39 as a hard input) or the TF-51 role-alignment.
- A **KDE density** read-out / grid (mean-serve only).
- Any **PR4** GK-arm work.

---

## 16. Resolved decisions (for the reviewer's checklist)

| # | Decision | Choice |
|---|---|---|
| D0 | Arc sequencing | **PR5 before PR4**, owner-directed; amends parent §17; dependency-safe (PR4 ⊥ PR5) (§1) |
| D1 | Target granularity | **Individual players**, not line/shape (§3) |
| D2 | Slot assignment | **Deterministic lateral rank** within the deepest-`n` rearguard (§3) |
| D3 | Model architecture | **Mirror `GhostGkModel`** (HGBR boosted mean, pickle-free, load-guards, numba); **no grid / no KDE / no label-domain cap** (§4) |
| D4 | Corpus | Same 179-match public corpus; **GS fully usable** (clamp is keeper-only) (§6) |
| D5 | Variants | `default` + `position_only`, velocity-keyed (ADR-067) (§7) |
| D6 | Surface | `serve_ghost_outfield_positions` seam; **no `add_ghost_outfield`** (+0 C4) (§10) |
| D7 | Coherence | **Measure**, escalate to a shape constraint only if incoherence is measured (§9) |
| D8 | Cards | **In the release commit** (§11) |
| D9 | Rearguard size | **Fixed `n_rearguard`** (default 4, tunable); not adapted per frame (§3) |
| D10 | Feature vector | **Frozen, leakage-safe** — 20 faithful / 16 position_only; no A-rearguard-derived feature; situational velocity only (§5) |
| D11 | Possession conditioning | **Possession-conditioned** (owner-approved post-review): training models BOTH teams' deepest-n (`team_in_possession` live, 1=carrier/rearguard, 0=defending line); rest defense serves the in-possession slice (serve/PR6 unchanged); per-possession MAE reported (§5 amendment) |

---

## 17. Remaining open points (plan-/measurement-time, not spec blockers)

The review's two must-fix items are now closed: the feature vector is frozen (§5, D10) and the
rearguard-size question is resolved (§3, D9). What remains genuinely needs data or is an
implementation-time detail, not a spec decision:

1. **Coherence trigger threshold** — the metric pair (ordering-fraction + min-pairwise-distance) is
   fixed (§9); the *incoherence level* that triggers the §9 shape-constraint remedy is set from the
   held-out measurement, so it is decided at training time, reported in the ADR. (§9)
2. **Code sharing with `GhostGkModel`** — how much of the HGBR/load-guard/numba machinery is shared vs
   duplicated is an implementation-time call, bounded by Chesterton's Fence on the frozen ghost-GK
   artifact surface (do not refactor the shipped ghost-GK to share code). The plan pins the exact
   shared seams. (§4)
3. **`serve_ghost_outfield_positions` return shape** — pinned against the verified
   `serve_ghost_gk_positions` contract: a ghost-rows DataFrame keyed by
   `(game_id, period_id, frame_id, team_id, slot_index)` carrying ghost **goal-relative** coordinates
   `ghost_gr_x`/`ghost_gr_y` (**the caller does the write-back to frame coords**, exactly as the GK seam
   does — an earlier draft said "raw frame coords," which mis-stated the GK contract) + a
   `ghost_outfield_source` provenance column. Exact column spellings are finalized in the plan. (§10)
