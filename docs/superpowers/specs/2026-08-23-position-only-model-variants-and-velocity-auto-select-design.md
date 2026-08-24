# Position-only model variants + velocity-keyed auto-select — design

> **Status:** DRAFT for review (uncommitted). Version / ADR / PR-Snnn numbers are assigned at
> commit time, never in this document.

## Executive summary (for reviewers)

Three fitted models — `XShotOccurrenceModel`, `XCrossAttemptModel`, `GhostGkModel` — carry a
minority of velocity-derived features. As of 4.90.0 they **honest-NaN** on velocity-less StatsBomb-360
freeze-frames: the ADR-054 velocity contract stopped them *fabricating* a value from a structurally
absent feature, but left them producing *nothing*. This cycle makes them **produce a real value on
SB360** by shipping **position-only variants** (the same models re-fit with the velocity features
dropped) and **auto-selecting** the position-only variant at the serve seam when the frames declare
velocity is unavailable.

The design is deliberately conservative in surface area and consistent with an existing, proven
pattern: `GkCompletionModel` resolves a *provider*-keyed variant at serve time
(`variant_key_for_provider`, `_gk_completion.py:275`, + `_resolve_completion_for_frames`,
`_xt_gk.py:353`) with an instance-override escape hatch (`GkRetentionModel` has its own analogous
resolver in `xtgk/_retention.py`). We reuse that shape, keyed on **velocity-availability** instead of
provider — with one deliberate asymmetry: a missing variant falls back to NaN, not to the default,
because the default velocity model is invalid on velocity-less frames (§D4). The models stay pure
scorers; variant selection is policy at the edge; and every served value on the `add_*` path is
stamped with a provenance column so auto-select is auditable, not magic.

The cycle also folds in the already-measured **#4 space_creation fix** (its licensed-corpus coverage
figure was measured on a pre-4.90.0 artifact and is stale): the cycle's final coverage run captures
both that fix and the new position-only model values in one refreshed artifact.

**Net observable change:** on velocity-less (declared) frames, `xshot_occurrence` / `xcross_attempt` /
`ghost_gk_x`/`_y` move from NaN to a real position-only value, and gkdv's ghost arm — which currently
drops SB360 frames — begins to work there. On velocity-bearing frames every output is byte-identical
to today. This is a disclosed retrain / Hyrum-law trigger.

## Global constraints (bind every task)

- **Engineering bar:** SOLID, hexagonal (pure engine, policy at the edge), TDD (red-first), clean
  code, no shortcuts on type safety / security / test coverage.
- **No pickle, parameters-only, fail-closed load** — every bundled artifact is npz / booster-JSON +
  JSON metadata + `SHA256SUMS`, guarded by chirality + feature-contract at load (ADR-011, ADR-016,
  ADR-040, ADR-044, ADR-050).
- **Trained-artifact provenance** — training runs on a **clean, pushed** tree; `training_commit` /
  `training_platform` recorded in metadata (ADR-037). Corpus passes adopt the `scripts/_driver.py`
  seam (ADR-052) and stamp `run_commit` (ADR-037/ADR-056).
- **No default-list change** — none of the three models is in `tracking_default_xfns`; that stays
  true. `xshot`/`xcross` remain in `pre_shot_gk_full_default_xfns` only; ghost stays opt-in-only.
- **Numbers at commit time** — version, ADR, PR-Snnn are decided only when the single commit(s) are
  made, never reserved here.
- **Minimal commits** — add a commit only where clean provenance *requires* one (see §9).

## Background & motivation

### The three models and their velocity dependence (verified)

| Model | Module | Feature constant | Total | Velocity-derived features |
|---|---|---|---|---|
| xShot | `tracking/_xshot_occurrence.py` | `XSHOT_FEATURE_NAMES_FAITHFUL` | 27 | `speed` (1) — `math.hypot(bvx,bvy)` |
| xCross | `tracking/_xcross_attempt.py` | `XCROSS_FEATURE_NAMES_FAITHFUL` | 16 | `ball_speed` (1) |
| ghost | `tracking/_ghost_gk.py` | `GHOST_GK_FEATURE_NAMES` | 26 | `ball_vx`, `ball_vy`, `ball_speed`, `defensive_line_speed`, `defending_centroid_vx` (5) |

xShot/xCross drop a *single* single-frame speed feature; ghost drops five, two of which
(`defensive_line_speed`, `defending_centroid_vx`) are **cross-frame temporal derivatives** requiring
`prev_state` in `_extract_all_ghost_gk_features`. All remaining features are purely positional
(polar coordinates, occlusion geometry, nearest-k distances/bearings, convex-hull compactness,
context flags). This is why position-only variants are viable: the great majority of each model's
signal is positional.

### Why NaN today

`compute_xshot_occurrence` / `compute_xcross_attempt` / ghost's `_serve_positions_core` each carry the
ADR-054 two-prong velocity guard: **declared** velocity-unavailable → honest NaN; **undeclared**
missing `vx`/`vy` → loud raise. The feature contract (`_feature_contract.py`) **raises on any
non-finite value** in the recorded fingerprint, and the contract probe frames supply concrete
`vx`/`vy` on every row — so a fitted model cannot simply be handed a NaN `speed`; there is no honest
in-band degrade. The 2026-08-21 spec recorded this explicitly:

> "the position-only re-fit is the real unlock — deliberately deferred to a separate cycle. So this
> cycle takes SB360 xShot to no value, not a partial one."
> "No position-only re-fit of xShot, xCross, or ghost … a separate future cycle with DGX training."

That deferred cycle is this one. The 2026-08-21 spec sketched **no** corpus, feature-drop strategy,
or variant scheme — the design below is new.

### The #4 finding this cycle folds in

The committed licensed-corpus coverage artifact (`docs/research/sb360_licensed_coverage/`) was
rendered from a **pre-4.90.0** parquet (commit `cf2f155`). A fresh 30-match measurement at 4.90.1
(commit `5c70fc4`, on the DGX) shows the 4.90.0 `add_space_creation` softening working on real data:
`space_created_m2` / `space_denied_m2_opponent` / `obso_epv_source` move out of the fully-NaN set,
and the "23 matches raised" line disappears (0 raises). The committed artifact is therefore stale.
Rather than commit that interim refresh now (which would churn the other session's in-flight
provenance work), this cycle regenerates the coverage once at the end — capturing the space_creation
fix **and** the new position-only model values together.

## Goals / non-goals

**Goals**
1. Position-only variants of xShot, xCross, ghost-GK, trained on the full-tracking corpus with
   velocity features dropped, bundled as parameters-only fail-closed artifacts.
2. Velocity-keyed auto-select at the three existing serve seams, with a provenance column and the
   `model=` override retained.
3. A reported velocity-vs-position-only skill-delta comparability artifact.
4. Refresh the licensed-corpus coverage (folds in #4), capturing both fixes.
5. Full docs: ADR, CHANGELOG, TODO, feature_glossary, CLAUDE.md, PRIVATE_CONSUMERS.

**Non-goals** (nothing is deferred *within* the goal; these are simply out of frame)
- No change to the existing velocity (`default`) variants' weights — the position-only variants are
  *new artifacts beside* them.
- No change to any default xfn list.
- No new methodology (same modelling; NOTICE unchanged).
- No SB360-native training (the 30-match licensed corpus is too small and goalkick-sparse at 0.444
  frame-existence; position-only variants train on the rich full-tracking corpus and serve on SB360).

## Detailed design

### D1. Position-only feature sets — EXTEND the existing `feature_set` param

The `feature_set` parameter **already exists**, keyed differently, and is **already keyword-only**:

- `XShotFeatureSet = Literal["faithful", "extended"]` (`_xshot_occurrence.py:125`);
  `extract_xshot_features(..., *, feature_set="faithful")` (`:154-160`).
- `XCrossFeatureSet = Literal["faithful", "extended"]` (`_xcross_attempt.py:39`).

`"extended"` is a **reserved roadmap slot, not a dead stub** — the novel separately-droppable GK
block and the crosser-position confounder are documented as its future direction
(`_xcross_attempt.py:48,59`; `_xshot_occurrence.py:166-168` "not implemented in PR-S75 … see the
TF-16 spec"). This cycle therefore **extends** the literal to
`Literal["faithful", "extended", "position_only"]` and implements **only** the `position_only`
branch; `"extended"` keeps its `NotImplementedError` intact (Chesterton's Fence — we do not delete a
reserved slot we were not asked to touch).

New name constants: `XSHOT_FEATURE_NAMES_POSITION_ONLY` (26 = faithful minus `speed`),
`XCROSS_FEATURE_NAMES_POSITION_ONLY` (15 = faithful minus `ball_speed`),
`GHOST_GK_FEATURE_NAMES_POSITION_ONLY` (21 = the 26 minus the five velocity features). Velocity
features are **dropped** (shorter vector), never NaN-filled — forced by the feature contract, which
raises on any non-finite fingerprint value (`_feature_contract.py:188-189`). A loaded model carries
`feature_set` in metadata; the serve path calls the extractor with `model.feature_set`.

**Three guard sites per model hard-raise on non-faithful today; lift ONLY the `position_only` path at
each, leaving `extended` raising:**

- `extract_*_features` → `if feature_set != "faithful": raise NotImplementedError` (xShot `:177-182`).
- `*Model.__init__` → same raise (xShot `:413-414`).
- `prepare_*_training_data` → same raise (xShot `:734-735`).

Each becomes "raise on `extended`; accept `faithful` and `position_only`". A TDD rung asserts
`extended` **still** raises (National Park Principle: do not widen a guard beyond the one value asked
for).

**xShot / xCross are genuine single-column drops; ghost is the ASYMMETRIC case — a different
extraction path, and that is a feature, not DRY sugar.** Ghost's two cross-frame derivatives
(`defensive_line_speed`, `defending_centroid_vx`) come from a real per-frame accumulation loop
(`prev_state` / `prev_timestamps`, `_ghost_gk.py:876,928,984`). The position-only ghost extractor is a
genuinely **single-frame-capable** path with no predecessor state — which is exactly what a lone
SB360 freeze-frame provides. So ghost's position-only path is a *distinct single-frame extraction*,
not feature-set-driven final assembly of one computation. This carries a dedicated correctness
obligation: **position-only ghost must be proven correct on a single freeze frame with no prior
frame** (TDD, §D6).

### D2. Training & corpus

Position-only variants train on the **same full-tracking pining corpus** as the velocity variants
via the existing `--providers` path in `scripts/train_xshot_occurrence.py`,
`scripts/train_xcross_attempt.py`, and `scripts/train_ghost_gk.py`. Each trainer gains
`--feature-set {faithful,position_only}` (default `faithful`, preserving current behavior). When
`position_only`, the trainer passes the feature set through `prepare_*_training_data` → the extractor,
and CV / `_gates` operate on the position-only vector unchanged.

The **bundled** position-only default is public-reproducible (idsse+skillcorner,
`shipped_variant="public"`) **if it clears the model's acceptance set**; a full-corpus position-only
variant is available via the HF Hub, mirroring the existing public/full split. Because the public
position-only *ghost* may fail acceptance (Risk 1, and ghost drops the largest feature share), **which
corpus ships as the bundled default is a DECISION GATE inside commit 2, not a mechanical bundling
step** — and it is recorded machine-checkably, not in prose: `metadata.json` carries `shipped_variant`
plus a `reproducibility ∈ {"public", "restricted"}` field with a reason, and a test asserts a
`restricted` bundled default carries that documented caveat. This reconciles the D2 intent
(public-reproducible) with the Risk-1 fallback (full-corpus) — the metadata records which actually
shipped. See §Risks and §9.

### D3. Serialization & bundled variants

New bundled directories, same format as the existing `default/`:

- `silly_kicks/tracking/_xshot_weights/position_only/`
- `silly_kicks/tracking/_xcross_weights/position_only/`
- `silly_kicks/tracking/_ghost_gk_weights/position_only/`

Each holds the model file (`model.json` booster / `rfcde_weights.npz`), `metadata.json`,
`SHA256SUMS`, `metrics.json`. `metadata.json` additionally records `feature_set="position_only"`, the
position-only `feature_names`, a **position-only chirality block** (the probe re-run through the
position-only extractor + model), and a **position-only feature-contract block** (same declared
geometry constants as the velocity variant — `goal_width` for xShot; `penalty_area_half_width` /
`penalty_area_depth` / `goal_width` for xCross; `penalty_area_half_width` / `penalty_area_depth` for
ghost — with the position-only fingerprint). The **shared** chirality / feature-contract / SHA256
machinery is generic (it fingerprints a caller callable), **but the per-model `_chirality_block` and
`_feature_contract_block` currently call the extractor WITHOUT a feature set, defaulting to faithful**
(`_xshot_occurrence.py:342,367`). Both must become feature-set-aware, and they differ (r2):
`_chirality_block(model)` takes the model, so it reads `model.feature_set`; `_feature_contract_block()`
is **model-independent by design** (`_xshot_occurrence.py:348`, "takes no model"), so it gains a **new
`feature_set` parameter** that the `save()` call site (`:504`) passes. This **is** new (small,
per-model) code — correcting the earlier draft's "no new load-guard code" claim.

**Resolving the variant — a directory, NOT an alias (r3).** `from_variant("position_only")` resolves
the new bundled directory via the `SHA256SUMS`-exists branch (`_xshot_occurrence.py:606`). **Do NOT
add `"position_only"` to `_VARIANT_ALIASES`** — that dict maps a name *onto the default bundle*
(`_VARIANT_ALIASES.get(variant, variant)`, `:602`; `{"public": "default"}`, `:326`), so aliasing
`position_only` would silently load the *default* weights. Per model: xShot / xCross's
`from_variant(variant: str)` needs **only the new directory** (no alias); ghost's
`from_variant(variant: GhostGkVariant)` (`_ghost_gk.py:2167`) takes a `Literal`, so `GhostGkVariant`
**must be extended** to include `"position_only"` or the call is a type error.

### D4. Velocity-keyed auto-select seam (core)

**Two layers, cleanly separated** (correcting the earlier draft, which conflated a pure key function
with a resolver that owns fallback/raise):

**Layer A — a PURE 2-way key.** `variant_key_for_velocity(frames) -> str`, in
`tracking/_velocity_availability.py` (ADR-063): returns `"position_only"` when
`velocity_unavailable_by_design(frames)`, else `"default"`. No IO, no fallback, no raise — the shape
of `variant_key_for_provider` (`_gk_completion.py:275`). **Placement note (r4):** that cited analogue
lives in its *model* module; we place ours in the *detector* module because it is shared by all three
models (their one common velocity seam). This deliberately inverts the precedent's layering — a
pragmatic SoC trade, documented rather than claimed as identical structure.

**Layer B — a per-model `(model, variant_key)` resolver** `_resolve_*_model_for_frames(frames, model)`,
mirroring `_resolve_completion_for_frames` (`_xt_gk.py:353`, the **completion** helper on
`GkCompletionModel`), which owns the bundled-check and fallback:
1. explicit `model=` instance/string → wins; `variant_key = "custom"` unconditionally (per D5's
   closed set — do NOT read `shipped_variant`, which would leak open values like `"public"`).
2. else `key = variant_key_for_velocity(frames)`; `from_variant(key)` → `(model, key)`.
3. else if `key == "position_only"` and it is **not bundled** → `FileNotFoundError` → warn → return a
   NaN-fallback sentinel (the seam emits NaN rows).

**The fallback DIRECTION is the load-bearing asymmetry with the completion template.**
`_resolve_completion_for_frames` falls back to `"default"` on a missing variant, because its default
is a valid scorer for any provider. Here the **default is INVALID on velocity-less frames** — running
it hands `speed=NaN` to XGBoost-as-missing, the exact ADR-054 fabrication — so a missing
`position_only` variant must fall back to **NaN, never to default**. Stated and tested explicitly.

**The RAISE stays where ADR-054 already puts it — in `compute_*`, not in the resolver.** The serve
seam keeps the ADR-054 guard verbatim: **undeclared** missing `vx`/`vy` → `raise`
(`_xshot_occurrence.py:863`) **before** the resolver is consulted, so the resolver only ever sees a
declared-unavailable or a velocity-bearing set. Auto-select fires only on the *declared* marker.

**Mixed-availability sets must RAISE — this closes a real fabrication hole.**
`velocity_unavailable_by_design` requires the marker on EVERY row and returns `False` on a
partially-marked set (`_velocity_availability.py:24-30`). Without a guard, a mixed set (some rows
freeze-frame, some velocity-bearing) resolves to the **default** velocity variant, and the
velocity-less rows get `speed=NaN` fabricated — exactly what ADR-054 killed, reappearing on mixed
frames. The completion template guards its analogue by raising on >1 real provider (`_xt_gk.py:358`);
we add the velocity analogue as a new single-sourced predicate in `_velocity_availability.py`
(detects a partially-marked set) and the serve seam **raises** on it (a mixed-availability set is a
caller error). Empty frames stay `False` → resolve to `default`, byte-identical — the documented VAEP
column-discovery caller passes an empty frame (`_velocity_availability.py:69-72`). Both are TDD rungs.

**Ghost specifics.** `_serve_positions_core` currently raises `_GhostVelocityUnavailableError` on the
declared marker, caught by `compute_ghost_gk` (→ NaN rows) and `serve_ghost_gk_positions` (→ zero
rows). Under auto-select it resolves and serves the position-only ghost instead; both entry points
emit real positions — the mechanism by which **gkdv's ghost arm starts working on SB360** (it consumes
`serve_ghost_gk_positions`, which currently yields zero rows there).

**Concrete seam restructure (r1) — the one place real branching control flow changes, replicated
across all three seams.** Verified in `compute_xshot_occurrence` (`_xshot_occurrence.py:854-912`);
`compute_xcross_attempt` and `_serve_positions_core` mirror it. Three concrete changes at each seam:

1. **Invert the declared → NaN early return.** Today `:854` resolves the model first
   (`m = _resolve_model(model)`) and `:863` unconditionally `return out` (all-NaN) on the declared
   marker — *that line is the behavior being inverted*. The declared branch instead consults Layer B
   (`_resolve_*_model_for_frames`) → serve `position_only`; the NaN return survives only as the
   unbundled fallback (§D4 branch 3). The ADR-054 raise prong (`:869`) is left untouched.
2. **Thread `feature_set` at the serve-time extract — the FOURTH feature_set site.** The per-frame
   `extract_*_features(grp, gk_team_id=..., goal_x=...)` call (`:912`) carries no `feature_set`; it
   must pass `feature_set=m.feature_set`. This is beyond the three guard sites (D1) and the two load
   blocks (D3) — six feature_set touch-points total per model, all enumerated so none is missed.
3. **Insert the mixed-availability raise BEFORE both existing prongs, keyed on the marker
   row-distribution — not column presence.** The two existing prongs are effectively column-level
   (`:863` all-marked via `velocity_unavailable_by_design`; `:869` `"vx" not in frames.columns`) and
   **structurally cannot catch a mixed set**: a partially-marked frame still carries a `vx` column
   (NaN on the freeze rows), so `:869` passes and the marked rows reach `speed = hypot(bvx, bvy)`
   (`:213`) as NaN → the exact fabrication. The new predicate keys off the `speed_source` marker
   distribution (`0 < n_marked < len(frames)`), not column presence, and the raise is inserted ahead
   of both existing prongs.

### D5. Provenance columns

Each `add_*` aggregator path emits a provenance column naming HOW the variant was selected, over a
**closed** vocabulary `{default, position_only, custom}`: `default` / `position_only` when auto-select
fired (Layer A returns exactly those literals), and `custom` for **any** explicit `model=` override.
The column answers "auto-default / auto-position-only / caller-override" — *auditable auto-select, not
variant lineage* — so the override branch maps to `custom` **unconditionally** and does **not** read
the model's `shipped_variant` (reading it would leak open values like `"public"` outside the closed
set). This follows the *stamped-provenance-column* shape of `das_source`, **not** the OPEN
`xt_gk_completion_variant` (which deliberately carries `shipped_variant` lineage) — an earlier draft
conflated the two (V1):

- `add_xshot_occurrence` → `xshot_occurrence_variant`
- `add_xcross_attempt` → `xcross_attempt_variant`
- `add_ghost_gk` → `ghost_gk_variant`

Because the seam is `compute_*` (§D4), the `*_xfns` path also serves position-only *values* on
velocity-less frames — it simply does **not** emit the string provenance column (VAEP matrices stay
numeric — the exact `das_xfns` vs `add_das` split). Each new column gets a `feature_glossary` entry
and is registered in the ADR-033 `add_*` purity gate with ≥2 variants (present/absent branches).

**Auditability limit on the numeric path (documented, m1).** On the `*_xfns` / VAEP path a downstream
consumer cannot tell from a value alone which variant produced a given SB360 cell — acceptable by the
`das_xfns` precedent, but this cycle is a NaN→value *behavior* change, so it is not free. The rule we
document (PRIVATE_CONSUMERS): traceability on the numeric path keys off the **frame-level** velocity
marker (`velocity_unavailable_by_design`) or the match provider, not the per-cell value; the per-row
`*_variant` column exists on the `add_*` path for consumers who need per-action provenance.

### D6. Validation, acceptance, TDD

**Training gates (reused, fail-closed):** each position-only training run passes the existing
`_gates` (`enough_usable_folds >= 2`, `pr_auc > positive_rate`, `brier < base_rate_brier`,
`log_loss < ln 2` for xShot/xCross; ghost's blocking `predict_mean` vs sklearn parity `<= 1e-6` +
zero categorical splits + the MAE acceptance set).

**New comparability artifact (reported, not a hard gate):** a script trains/loads both the velocity
and position-only variants and reports the held-out AUC / Brier (xShot/xCross) or MAE (ghost) **delta**
— how much skill dropping velocity costs — written to `docs/research/position_only_variants/`. This is
a research artifact (ADR-037 provenance), not a CI gate, because the position-only variant is
*expected* to be weaker; the point is to quantify and disclose the cost, and to inform the
public-vs-full bundled-ghost decision (§Risks).

**TDD ladder (red-first):**
1. `variant_key_for_velocity` (Layer A) — declared-unavailable → `position_only`; velocity-bearing →
   `default`; pure, no IO, no raise.
2. Mixed / empty availability — the new partially-marked predicate is True on a mixed set and the
   seam **RAISES** on it (closes the M3 fabrication hole); an **empty** frame set → `default`,
   byte-identical, no raise (the VAEP column-discovery caller, `_velocity_availability.py:69-72`).
3. Per-model resolver `_resolve_*_model_for_frames` (Layer B) — override wins (variant = shipped /
   `custom`); declared + bundled → `position_only`; declared + **unbundled** → NaN sentinel + warn,
   falling back to **NaN, NOT default** (the load-bearing asymmetry with the completion template);
   velocity-bearing → `default`.
4. ADR-054 raise preserved — undeclared missing `vx`/`vy` still RAISES at `compute_*` (the raise lives
   in the seam, not the resolver), unchanged from 4.90.0.
5. `extended` still raises — at all three guard sites per model (extract / init / prepare):
   `position_only` accepted, `extended`'s `NotImplementedError` intact (B2).
6. Extractor position-only feature set — correct length (26 / 15 / 21), no velocity name present,
   finite on the contract probe frame; **ghost proven correct on a single freeze frame with no
   predecessor** (the asymmetric single-frame path, D1 / M1).
7. Load guards — bundled position-only artifacts pass chirality + feature contract (with the
   per-model blocks threaded by `feature_set`, m3); a corrupted position-only fingerprint raises.
8. Behavioral — `add_*` on a declared velocity-less frame emits a **value** (not NaN) + provenance
   `= "position_only"`; on a velocity-bearing frame byte-identical to today + provenance `= "default"`.
9. Non-vacuity — a velocity-bearing frame never resolves to `position_only`.
10. gkdv unlock — `serve_ghost_gk_positions` on a declared velocity-less frame now yields rows (was
    zero), and the gkdv ghost arm produces a delta there.
11. Metadata gate (M4) — a `restricted` bundled default carries the documented reproducibility caveat
    (machine-checkable, not prose).
12. `@pytest.mark.slow` train-smoke for `--feature-set position_only` on each trainer (does-it-run,
    acceptance-all-true), mirroring the existing smokes.

Golden/version-sensitive tests stay on all matrix legs; slow smokes on the primary leg only
(ADR-023).

### D7. Coverage refresh (#4 folded in)

After the position-only variants are bundled, re-run
`scripts/validate_sb360_licensed_corpus.py` on the DGX (owner token, clean tree) → refreshed
`coverage.parquet` + `manifest_all.json`, then `scripts/render_sb360_licensed_coverage.py` →
`coverage.md`. The refreshed artifact records the space_creation fix (already measured) **and** the
new fully-NaN → populated transitions for `add_xshot_occurrence` / `add_xcross_attempt` /
`add_ghost_gk`. The narrative ("40 → 31 fully-NaN lift") updates to the new count.

## Testing summary

- Unit: `variant_key_for_velocity` (Layer A, pure 2-way) + the Layer-B resolver (3 branches, with the
  NaN-not-default fallback) + the ADR-054 raise pinned in `compute_*` (not the resolver) + the
  mixed-availability raise + extractor feature-set.
- Load-guard: chirality + feature-contract for each position-only artifact.
- Behavioral: auto-select value/provenance both directions; non-vacuity; gkdv unlock.
- Slow: `--feature-set position_only` train-smoke ×3.
- Purity (ADR-033): the three new provenance columns, ≥2 variants each.
- Full suite + `.venv312` (pandas-3 span, ADR-057) + ruff + whole-branch pyright green before the
  final commit (the net, per the 4.89.0 lesson).

## Commits & provenance (§9)

Clean provenance forces a dependency chain; the minimum is **3 commits**, each *required* by a
provenance dependency (not tidiness):

1. **Library code** — extractors (feature-set param + new name constants), `variant_key_for_velocity`,
   the serve-seam resolver + provenance columns, feature_glossary, ADR, guard-independent tests.
   Pushed so the DGX can train against it.
2. **Position-only weights — a DECISION GATE, not a mechanical drop** — trained on the DGX against the
   *pushed* commit-1 tree (clean; `training_commit` = commit 1). This commit makes the Risk-1
   public-vs-full bundled-*ghost* ship decision (recorded machine-checkably in `metadata.json` per D2),
   produces the comparability artifact, bundles the chosen weights, and adds the load-guard +
   behavioral tests that need the bundled artifacts. Named a gate so a future reader does not treat it
   as mechanical (n3).
3. **Refreshed coverage** — run against the *committed* commit-2 weights (clean; `run_commit` =
   commit 2) + the coverage narrative update + CHANGELOG / version bump.

Squashing to one commit would leave the weights' `training_commit` and the coverage's `run_commit`
citing dead SHAs — dirty provenance — which is exactly the failure "add a commit only when clean
provenance requires it" guards against. Each artifact records a **live** parent SHA.

## Docs / ADR / coordination

- **New ADR** — velocity-keyed variant auto-select + the position-only model family (a new convention
  with downstream consumers and a NaN → value behavior change). Number at commit time.
- **CHANGELOG** — the retrain / Hyrum trigger (SB360 values change from NaN to a number; velocity
  frames byte-identical), the new provenance columns, the gkdv unlock.
- **TODO** — release line replaced.
- **feature_glossary** — three provenance columns.
- **CLAUDE.md** — durable contract: velocity-keyed auto-select, position-only variants, provenance
  columns.
- **PRIVATE_CONSUMERS.md** — the behavior change + new columns for the lakehouse.
- **Coordination** — the position-only variants are *new* artifacts beside the untouched `default`
  weights, so there is no conflict with the other session's ghost re-fit (4.81.0) or its in-flight
  provenance work; a single merge at the end minimizes churn for it.

## Risks & open decisions

1. **Public-vs-full bundled ghost (a measured commit-2 gate, not a guess).** The public corpus
   (idsse+skillcorner) may be too thin for a skillful position-only *ghost* specifically — ghost drops
   5/26 features, the largest cut. The comparability artifact (§D6) measures the public position-only
   ghost's MAE against the velocity ghost and against the full-corpus position-only ghost; if the
   public variant fails the ghost acceptance set (`overall_mae_lt_2m`, `per_provider_mae_lt_3m`), the
   bundled default becomes the full-corpus position-only ghost, recorded machine-checkably as
   `reproducibility="restricted"` in `metadata.json` with a reason (D2), asserted by a test. This is
   the commit-2 decision gate (§9), decided on the measurement, not assumed here.
2. **Behavior change disclosure.** SB360 xShot/xCross/ghost go NaN → value; gkdv's ghost arm begins
   producing on SB360. Disclosed via CHANGELOG + the provenance columns + PRIVATE_CONSUMERS. The NaN
   is only 4.90.0-fresh, so the installed base depending on it is minimal; velocity-bearing consumers
   are byte-identical.
3. **DGX dependency floor.** The `scripts/_driver.py` corpus seam requires `ruthless-efficiency>=0.4.0`
   (`fingerprint`); the DGX training venv must satisfy the pyproject floor before the runs (observed
   below-floor once this cycle and upgraded).

## Attribution

No new methodology — position-only variants re-fit the same models on the same corpus with a reduced
feature set. NOTICE is unchanged. Existing per-feature citations stand.

## Revision log

**Review round 1** (independent parallel critic, checked against `main` @ `5c70fc4`). All findings
verified against the tree and accepted; changes:

- **B1** — D1 rewritten: `feature_set` already exists as `Literal["faithful","extended"]`; we
  *extend* to add `"position_only"` and keep `"extended"`'s reservation (Chesterton's Fence), not
  redefine the literal.
- **B2** — D1 enumerates the three per-model guard sites (extract / init / prepare) and lifts only
  `position_only`; TDD rung 5 asserts `extended` still raises.
- **M1** — D1 reframes ghost as the asymmetric *single-frame* extraction path (a feature, not DRY
  sugar); TDD rung 6 adds the single-freeze-frame-no-predecessor correctness obligation.
- **M2** — D4 split into Layer A (pure 2-way key) and Layer B (`(model, key)` resolver owning the
  bundled-check); the ADR-054 raise pinned to `compute_*`, not the resolver.
- **M2 (asymmetry)** — D4 + exec summary: missing `position_only` falls back to **NaN, not default**
  (the default is invalid on velocity-less frames), unlike the completion template.
- **M3** — D4 adds a partially-marked-availability predicate + a serve-seam **raise** on mixed sets
  (closes the fabrication hole); TDD rung 2.
- **M4** — D2 + Risk 1 + §9: bundled public-vs-full choice is a commit-2 decision gate with a
  machine-checkable `reproducibility` field in `metadata.json`; TDD rung 11.
- **m1** — D5 documents the numeric-path auditability limit + the frame-level traceability rule.
- **m2** — D4 + TDD rung 2: empty frames → `default`, byte-identical, no raise.
- **m3** — D3 corrected: the per-model `_chirality_block` / `_feature_contract_block` need
  `feature_set` threading — this *is* new (small) code, not a free reuse.
- **n1** — D1 notes `feature_set` is already keyword-only.
- **n2** — citation fixed: `_resolve_completion_for_frames` is in `_xt_gk.py:353` (GkCompletionModel);
  retention has its own resolver in `xtgk/_retention.py`.
- **n3** — §9 names commit 2 a decision gate.

**Review round 2** (same critic, re-verified every round-1 fix against source; verdict: ready to
plan, five residual Minor items). All verified against the tree and folded in:

- **r1** (implementation trap) — D4 gains a "Concrete seam restructure" subsection: the declared→NaN
  `return out` (`:863`) is inverted into the Layer-B decision; the serve-time `extract_*_features`
  call (`:912`) is enumerated as the FOURTH `feature_set` site (six total per model); the
  mixed-availability raise keys off the `speed_source` row-distribution (`0 < n_marked < len`), not
  column presence, inserted before both existing prongs (the column-level `:869` prong cannot catch a
  mixed set).
- **r2** — D3 corrected: `_feature_contract_block()` is model-independent, so it gains a new
  `feature_set` parameter (passed from `save()`); only `_chirality_block(model)` reads
  `model.feature_set`.
- **r3** (implementation trap) — D3 corrected: `position_only` is a **directory**, resolved via the
  `SHA256SUMS`-exists branch; do NOT add it to `_VARIANT_ALIASES` (which maps onto default → wrong
  weights). xShot/xCross need only the directory; ghost extends the `GhostGkVariant` Literal.
- **r4** — D4 Layer A documents that placing `variant_key_for_velocity` in the detector module
  inverts the cited precedent's layering, justified by shared-across-3-models.
- **r5** — Testing-summary bullet updated to the Layer-A / Layer-B (3 branches) / raise-in-`compute_*`
  / mixed-raise shape.

**Plan review round 2** (V1 surfaced during plan review, but the fix lives in the spec): D5 declared a
**closed** vocabulary `{default, position_only, custom}` while also saying "mirroring
`xt_gk_completion_variant`" — but that column is **open** (carries `shipped_variant` lineage). The two
are mutually exclusive. Resolved by keeping the set closed: the override branch maps to `"custom"`
**unconditionally** (never `shipped_variant`), following the `das_source` stamped-column shape, not the
open `xt_gk_completion_variant`. Fixed in D5 and D4 Layer-B item 1.
