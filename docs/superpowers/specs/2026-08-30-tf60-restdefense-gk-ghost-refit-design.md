# TF-60 PR3 — Rest-Defense GK-Ghost Model Re-fit (extended-grid variant) — Sub-spec

- **Status:** DRAFT (author self-reviewed; awaiting owner review, then an independent `/review-impl`).
- **Date:** 2026-08-30
- **Feature:** TF-60 PR3 (reshaped — see §1 and the parent spec's arc amendment).
- **Parent spec:** `docs/superpowers/specs/2026-08-30-tf60-rest-defense-structure-and-gk-design.md` (§9,
  §10, §17, §20.1 amended by this cycle).
- **Evidence:** `docs/research/tf60_ghost_gk_in_possession_validity/` (the §9 finding this cycle exists to
  address).
- **Decision record:** ADR-083 (to be written in the implementation commit). *(Was ADR-082, but the
  parallel TF-19 A+2 session shipped ADR-082 / PR-S175 / silly-kicks 4.104.0 on 2026-08-31; PR3 renumbered
  to the next free after merging that release into the branch.)*
- **Version:** 4.105.0 (next free from `main` as of 2026-08-31 — main is now 4.104.0; confirm at ship;
  PR-S176). The Layer-3 GK arms move to the
  next cycle.
- **Precedent:** this is a **trained-model cycle** and follows the TF-18 / TF-16 / TF-17 pattern — a
  dedicated sub-spec covering target/grid parametrisation, feature set, training corpus, validation
  harness, and HF publishing, shipping as **one cycle** (code + pipeline + bundled weights). The
  "grid becomes a first-class per-model variable" refactor mirrors xthreat's `GridSpec` (ADR-021) **for the
  grid-as-dataclass aspect only** — xthreat's `ExpectedThreat` has no `save`/`load`, so the
  **save-serialize + load-restore + byte-preserving-metadata** aspect here (§4.2) is **novel** and is where
  the care goes (review P3-03).
- **Depends on / touches:** `silly_kicks/tracking/_ghost_gk.py` (the `GhostGkModel` machinery), the
  bundled-weights layout `_ghost_gk_weights/`, `scripts/train_ghost_gk.py`, `scripts/publish_ghost_gk.py`.
  ADR-011/016/040/044/050 (trained-model discipline: parameters-only, pickle-free, fail-closed load,
  chirality + feature-contract), ADR-037 (gkdv consumes tracking public seams only), ADR-063/067
  (velocity tiers / velocity-keyed `position_only`), ADR-076 (numba leaf traversal), ADR-038
  (fail-closed public corpus).

---

## 1. Why this cycle exists (the reshaped arc)

The parent spec's §9 flagged, as **a gate not an assumption**, that the shipped `GhostGkModel` — exercised
by GKDV only in the defending-near-attacked-goal regime — might not be valid for the **in-possession
high-sweeper** geometry rest defense needs. PR3's brainstorm measured it
(`docs/research/tf60_ghost_gk_in_possession_validity/`): the shipped `default` model **hard-saturates at
its trained-label ceiling `GRID_X_MAX = 30 m`** and cannot represent a keeper standing 30–45 m off its own
goal. The cause is structural — `prepare_ghost_gk_training_data` (`_ghost_gk.py:1204`) **drops every
training label with `gk_x > 30 m` as "sweeper rushes"** — and the built-in `ghost_out_of_box` flag is blind
to it (the model clips its own output to ≤30 m, then reports "in box").

So the shipped model **cannot** back the Layer-3 GK arms without systematically compressing the deterrent
for exactly the aggressive sweeper-keepers the metric exists to reward (parent §16.1). Per the owner ruling
(2026-08-30, "insert a GK-ghost model cycle before the arms"), the arc is reshaped so a **rest-defense GK-
ghost re-fit** precedes the GK-arm cycle — mirroring the existing model→arm shape (PR4 ghost-outfield
model → PR5 outfield arm):

| Cycle | Content | Model? |
|---|---|---|
| PR1 ✅ | Layer 1 structure KPIs | — |
| PR2 ✅ | Layer 2 danger valuation | — |
| **PR3 (this)** | **Rest-defense GK-ghost re-fit** — extended-grid additive variant + bundled weights + HF publish | **yes** |
| PR4 (was PR3) | Layer-3 **GK arms** (consume this variant) | no |
| PR5 (was PR4) | ghost-outfield model | yes |
| PR6 (was PR5) | Layer-3 **outfield arm** | no |

---

## 2. Objective

Ship an **additive** `GhostGkModel` variant whose trained label domain and grid are **extended upfield**,
so it can represent the in-possession high-sweeper regime, **without touching** the frozen `default` /
`position_only` / `full` variants (GKDV keeps `default`; **no GKDV retrain, no GKDV behaviour change**).
The rest-defense GK-arm cycle (PR4) will select this variant explicitly.

The load-bearing insight: **the training data already exists.** `prepare_ghost_gk_training_data` extracts
**both** keepers across the **whole** match, then discards the labels above 30 m. Lifting that cap on a
re-extraction retains the high-sweeper labels that were being thrown away — so this is a **re-fit on the
same corpus**, not new data collection.

---

## 3. Scope and non-goals

### In scope
- A "grid becomes a first-class per-model variable" refactor of `GhostGkModel` (see §4), **byte-identical**
  for `default`.
- A new bundled variant pair — `sweeper` (faithful, 26 feat) and `sweeper_position_only`
  (21 feat, for velocity-less SB360) — with an **extended grid** and a widened label domain.
- Trainer + publisher support for the extended grid / new variant, on the **existing** DGX corpus.
- The full trained-model gate suite for the new variant (golden / chirality / feature-contract /
  integrity-on-load / save-load round-trip / position_only round-trip), **plus** a two-sided
  saturation-vs-tracking regression gate (§9).
- A committed, reproducible §9 finding + probes (already written this cycle).

### Explicit non-goals
- **The Layer-3 GK arms** (`build_restdefense_ghost_frames`, `_arms.py`, `merge_rest_defense`,
  `rest_defense_gk_deterrent`) — PR4.
- **Any change to `default` / `position_only` / `full`.** They stay byte-identical (frozen). **No GKDV
  retrain**, no VAEP retrain, no re-materialize; this variant enters no default xfn list and changes no
  existing column.
- **New data collection.** Re-fit on the existing corpus with the cap lifted.
- **The ghost-outfield model** (PR5) and **new geometry constants** beyond the grid bounds.

---

## 4. Architecture — the grid becomes a first-class per-model variable

### 4.1 The problem in the current code
`GRID_X_MIN/MAX`, `GRID_Y_MIN/MAX`, `GRID_RESOLUTION` are **module-level constants** (`_ghost_gk.py:91-97`)
consumed at five sites: the training **label-domain filter** (`prepare_ghost_gk_training_data:1204-1220`);
the KDE density grid (`_GRID_X/_GRID_Y/GRID_NX/GRID_NY`); `save`'s `grid_spec` metadata (`:2146-2153`); the
serve-time **`ghost_out_of_box`** flag (`:2802`, `positions[:,0] > GRID_X_MAX`); and the density spread /
KDE kernels. **`load` records but does NOT restore `grid_spec`** (verified: `:2231-2340` reads
n_estimators/feature_set/carrier_params/pitch dims/chirality/contract, never the grid). So today the grid
is global and a differently-gridded artifact would be silently served on the 30 m grid.

### 4.2 The refactor
Introduce a frozen `GhostGridSpec(x_min, x_max, y_min, y_max, resolution)` with
`DEFAULT_GHOST_GRID` = the current constants, and give `GhostGkModel` a `self.grid_spec` (constructor
default `DEFAULT_GHOST_GRID`). Thread it through **every** consuming site:

| Site | Change |
|---|---|
| `__init__` | new `grid_spec: GhostGridSpec = DEFAULT_GHOST_GRID` |
| `save` (`:2146`) | write `self.grid_spec` (not the module constants) — but **byte-preserving**: see the serialization pin below |
| `load` (`:2301`-ish) | **restore** `grid_spec` from metadata into the instance (back-compat: absent → `DEFAULT_GHOST_GRID`) |
| label filter (`prepare_ghost_gk_training_data:1204`) | take grid bounds as a param; the trainer passes the target variant's grid |
| `ghost_out_of_box` (serve, `:2802`) | `positions[:,0] > self.grid_spec.x_max` (variant-relative ceiling) |
| KDE density path (`predict_density`) | **unchanged**; a guard at the top of `predict_density` **raises** on `self.grid_spec != DEFAULT_GHOST_GRID` (see the resolved density decision below) — no KDE-kernel change, so the KDE goldens are byte-identical |

**Serialization pin (byte-identity, review P3-02).** `save`'s current `grid_spec` metadata is a **7-key**
dict — `{x_min, x_max, y_min, y_max, nx, ny, resolution}` (`:2146-2153`) — while `GhostGridSpec` is **5
fields** (`nx`/`ny` are DERIVED: `nx = round((x_max − x_min) / resolution)`, `ny = round((y_max − y_min) /
resolution)`; verified against `default`'s metadata: `(30−0)/0.5 = 60`, `(50−18)/0.5 = 64`). So a naive
`dataclasses.asdict(self.grid_spec)` would emit **5** keys, change `default`'s `metadata.json`, and break
its committed SHA. The pin: `GhostGridSpec` serializes via a dedicated `to_metadata_dict()` that emits the
exact 7 keys WITH derived `nx`/`ny`, in the current key order — so `default`'s metadata is byte-identical.

**Chesterton's Fence / byte-identity:** for `default` this is a pure no-op — `DEFAULT_GHOST_GRID` equals the
current constants, so the label filter, KDE grid, `out_of_box`, and the `grid_spec` metadata are unchanged.
**MEASURED CORRECTION (impl review P3I-02; three-way gate OWNER-RATIFIED 2026-08-31):** a full
`save`→`load`→`save` `SHA256SUMS` round-trip is **not achievable and never was** — `save()` recomputes `feature_contract.probe_sha256` from the *current*
extractor, which has evolved since the bundled `default` was trained on the DGX, so `default`'s
`metadata.json` SHA differs on re-save **independent of this refactor**. Byte-identity is therefore gated
three ways, none of which is the full-SHA round-trip: **(1)** the existing golden / chirality /
feature-contract / KDE-density behavioural tests pass **unchanged** (277 ghost tests, the real proof that
`default`'s behaviour is untouched); **(2)** the `grid_spec` FIELD serialises byte-identically —
`DEFAULT_GHOST_GRID.to_metadata_dict()` equals `default`'s committed `grid_spec` (this is what catches a
mis-serialised 5-key grid_spec — review P3-02); **(3)** a re-save of `default` changes **only**
`feature_contract` and nothing else (a pinned per-field diff, so a refactor that moves any *other* field
for `default` fails). The module constants remain (now consumed by `DEFAULT_GHOST_GRID` and any legacy
reference) — not deleted (a genuine reader may still import them).

**Density-grid scope — RESOLVED (fail-loud, not thread-through).** The KDE density kernels
(`_bin_ngp`/`_bin_cic`/`_kde_density_*`) read the module-global grid (`GRID_NX`/`GRID_NY`/`_GRID_X`/
`GRID_RESOLUTION`) for their `.reshape` and binning; `grid_points` itself is already a passed-in argument.
Threading a per-model grid through that path is **speculative capability with no consumer**: rest defense
uses `serve_ghost_gk_positions` → `predict_mean`, never `predict_density`; `predict_density` is only
available on a **locally-fit** model (the per-sample arrays are not bundled), so the bundled `sweeper`
variant cannot call it at all; and no cycle in the arc needs an extended-grid density surface. So this
cycle **scopes the density path to the default grid and fails LOUD**: **`predict_density` ONLY** raises a
clear error on a non-default-grid model rather than silently returning a density computed on the wrong 30 m
grid. **`compute_ghost_gk` / `add_ghost_gk` are NOT guarded and WORK on the extended grid** — their
density read-out was retired 2026-07-20 (docstring `:2444-2453`; §3.1), so they use the grid-independent
`predict_mean` leaf-value traversal (the `ghost_out_of_box` flag reads `self.grid_spec.x_max` post-refactor,
so it is variant-relative), exactly as `serve_ghost_gk_positions("sweeper")` does. Guarding them would
wrongly refuse a legitimate mean-path positioning use of the variant. This is the gold-standard choice here
— it
does the minimal correct thing (a loud refusal, NOT the silent-wrong-grid trap the §9 finding itself
warns about), it is **byte-identical for `default`** (zero KDE-path changes, so the KDE goldens pass
untouched), and it defers the KDE grid-threading to a well-scoped follow-up **with a real validating
consumer** instead of building it speculatively now (YAGNI). The grid is fully first-class for the
mean/serve path (save / load / label filter / `ghost_out_of_box`) — the path that matters — and the
density path's limitation is explicit and enforced, not latent.

---

## 5. The extended grid and label domain

- **`x_max`**: extend from **30 m → 52.5 m** (halfway). A keeper past halfway is essentially always an
  artifact, so halfway is a principled hard ceiling that safely covers the high-sweeper envelope. The
  re-fit cycle **measures and reports** the empirical committed-forward keeper-depth distribution on the
  DGX corpus (the "fraction >30 m" the owner deferred lands here) and confirms 52.5 m is not clipping real
  positions; if the corpus argues for a tighter data-driven ceiling (e.g. p99.5), the sub-spec value is
  revised in the implementation with that evidence.
- **`y_min/y_max`**: unchanged (18–50, i.e. ±16 m). A high sweeper stays roughly central; the corpus check
  confirms the y-envelope, and widens only with evidence.
- **`resolution`**: unchanged (0.5 m). `nx` grows 60 → 105; grid cost is trivial and does not affect the
  bundled artifact size (bundled trees are grid-independent; density arrays are not bundled).
- **Label filter refinement**: the current filter conflates "sweeper rush" (a legitimate high position) and
  "off-pitch artifact." The re-fit **keeps** `gk_x ∈ [0, x_max]` (retaining legitimate high sweepers) while
  still **dropping genuinely off-pitch / impossible** coordinates (x<0, x>105, y off-pitch). The
  `n_out`-dropped warning stays, now counting only true artifacts.

---

## 6. Training corpus and the honest coverage caveat

- **Corpus**: the **same** DGX corpus the `default`/`full` variants used (providers: sportec,
  gradientsports, skillcorner; 179 games — source: the bundled `default` `_ghost_gk_weights/default/
  metadata.json` `corpus_provenance`, not the trainer script). No new collection.
- **Where the high-sweeper signal lives**: the previously-dropped >30 m labels come **predominantly from
  the full-tracking providers (sportec + GS)**. SkillCorner's FOV/detection filter (`keeper_detection_mask`,
  detected-only) thins its high-sweeper contribution — the exact selection bias the trainer already
  documents (`detection_selection_bias`: detected frames "under-sample the deep sweeper regime"). This is
  **acceptable and must be reported**: the served high-sweeper predictions are grounded in full-tracking
  data, and the metrics section (§7) reports the **per-provider coverage of the >30 m stratum** so a
  consumer knows the provenance of the high-sweeper regime.
- **Data-visibility (ADR-038)**: the **bundled public** variant trains on a **public-only** corpus, exactly
  as `default` does; any owner-tier variant is not bundled. The trainer's existing fail-closed public-corpus
  discipline is reused unchanged.

---

## 7. Validation

Methods are CI-gated; applied metrics are reported (repo convention).

### 7.1 The two-sided saturation-vs-tracking gate (the cycle's signature test)
Promote the §9 controlled-extrapolation probe into a committed regression gate on the sportec slim fixture
(no DGX needed at CI time). It asserts **both sides** (CLAUDE.md "every band needs a test from both sides"):
1. **The old `default` variant SATURATES** — its predicted keeper gr_x max caps at ~30 m while the actual
   keeper (scene translated upfield) reaches well above it. This pins the defect so a future "fix" that
   silently reverts is caught.
2. **The new `sweeper` variant TRACKS** — its predicted keeper gr_x rises past 30 m with the advancing
   scene (a mutation that fails to track fails the gate). Non-vacuity: the two variants **measurably
   differ** on the same frames.

Because the real `sweeper` weights come from the DGX, the gate is authored against a **toy** extended-
grid model locally (fit on a fixture with the cap lifted), then re-pointed at the bundled weights when they
land; the assertion is qualitative (tracks vs saturates), not a magnitude tied to the toy.

### 7.2 Standard trained-model gates (per variant)
Golden `predict_mean` parity (numpy reconstruction == sklearn to ≤1e-6 on the fresh fit); chirality
fingerprint; feature-contract fingerprint + declared constants; integrity-on-load (SHA256SUMS, pitch-dim
guard, serve-estimator guard); the three-way `default` byte-identity of §4.2 (behavioural goldens
unchanged + grid_spec-field identity + re-save-differs-only-in-feature_contract — **not** a full-SHA
round-trip); `position_only` round-trip. The bundled `default` artifact's committed goldens must pass
**unchanged** (the behavioural half of the §4.2 byte-identity proof).

### 7.3 Reported metrics (not gated)
CV MAE overall + **per-provider** + a dedicated **>30 m high-sweeper-stratum MAE** (does the variant predict
the high regime accurately, not merely lower the overall error?); the DGX committed-forward keeper-depth
distribution + fraction >30 m (the deferred §9 population number); the per-provider >30 m coverage (§6).

---

## 8. HF publishing and bundling

- `scripts/publish_ghost_gk.py` is extended to publish the new variant(s) with the same discipline it
  already enforces: refuse to publish an artifact whose `metadata.json` lacks a feature contract, and
  assert the round-tripped model loads without `MissingFeatureContractWarning`.
- **Bundling**: `sweeper` and `sweeper_position_only` are **bundled in the wheel** (like
  `default`/`position_only`) under `_ghost_gk_weights/sweeper/` and `_ghost_gk_weights/
  sweeper_position_only/`. The grid extension does not materially change the bundled size (trees are
  grid-independent; per-sample density arrays are not bundled). `GhostGkVariant` gains the two literals;
  `from_variant("sweeper")` loads from the bundled dir (Hub fallback available if a size concern
  emerges, mirroring `full`).

---

## 9. Public API surface

```python
from silly_kicks.tracking import GhostGkModel
GhostGkModel.from_variant("sweeper")             # extended-grid, faithful (velocity-bearing)
GhostGkModel.from_variant("sweeper_position_only")  # extended-grid, position_only (SB360)
```

- New `GhostGkVariant` literals: `"sweeper"`, `"sweeper_position_only"` (added to the existing
  `Literal["default", "full", "position_only"]`).
- **Naming (resolved, owner pick invited):** the variant is named for its **capability**, not its
  consumer, following the existing convention (`default`/`full`/`position_only` describe the artifact, not
  who uses it). `sweeper` = "the variant that can place a high sweeper" (the extended label domain). It is
  deliberately **not** `rest_defense` — a consumer-tied name would break the convention and the variant is
  reusable by any high-sweeper positioning need, not just rest defense. The cross-layer asymmetry (variant
  `sweeper` in `tracking`, arms functions `rest_defense_*` in `restdefense`) is expected: each layer keeps
  its own naming convention. `extended` / `full_depth` are viable alternatives; the final pick is invited
  in review (it enters a public `Literal` + metadata, so pin it before publish — but it is trivial to
  change now, no back-compat).
- `GhostGridSpec` / `DEFAULT_GHOST_GRID` are **internal** (`_ghost_gk`) — variants are selected by name, not
  by constructing a grid. `grid_spec` remains a metadata field (now also restored on `load`).
- `serve_ghost_gk_positions` / `compute_ghost_gk` are **unchanged** in signature; they already accept a
  `model` (instance or variant name), so PR4 passes `from_variant("sweeper")`. The velocity-keyed
  auto-select (`_resolve_ghost_model_for_frames`, ADR-067) resolves the `position_only` sub-variant on
  declared-velocity-less frames — this must be extended so a caller who supplies `sweeper` gets
  `sweeper_position_only` on SB360 (see §10).

---

## 10. Velocity-keyed variant resolution (ADR-067) for the new pair

The existing `_resolve_ghost_model_for_frames` maps a declared-velocity-less frame set to the
`position_only` variant. With a second variant pair, the resolution must become **pair-aware**: a caller
supplying `sweeper` on velocity-less SB360 frames resolves to `sweeper_position_only`; the
missing-`position_only` fallback stays **NaN, never `default`** (ADR-067's load-bearing asymmetry — the
default-velocity model is invalid on velocity-less frames). This is a small, additive change to the
resolver's variant-key logic (mirroring `variant_key_for_velocity`), gated by the velocity-availability
tests extended to the new pair.

---

## 11. Error handling / degradation

- Loading a `sweeper` artifact goes through the **same** fail-closed guards as `default` (SHA256SUMS,
  pitch-dim, serve-estimator, chirality, feature-contract). A grid-restore back-compat path: an artifact
  with no `grid_spec` in metadata loads with `DEFAULT_GHOST_GRID` (pre-refactor `default` behaviour).
- `predict_density` on a non-default-grid model **raises** a clear "extended-grid density not supported this
  cycle" error (the resolved density decision, §4.2) — never a silently-wrong density on the 30 m grid.
  `compute_ghost_gk` / `add_ghost_gk` are **not** guarded and **work** on the extended grid (they use the
  grid-independent `predict_mean` path; their density read-out was retired 2026-07-20).
- Velocity contract unchanged — the serving seam's mixed/undeclared-missing-velocity raises are inherited.

---

## 12. Decomposition (one cycle; local now, DGX weights later)

Per the owner ruling "build + test locally now, DGX weights later," a single squash-merged branch:

1. **Local (this session, after review + go-ahead):** the grid-first-class refactor (byte-identical
   `default`), the new variant plumbing (literals, label-domain param, `from_variant` wiring, pair-aware
   velocity resolution), trainer + publisher support, the §9 finding + probes (done), and **all gates
   authored + passing against a locally-fit toy extended-grid model** + the byte-identity `default` proof.
   Lint (CI scope) + pyright + full suite green.
2. **DGX (coordinated when the machine frees up):** re-extract features with the cap lifted + re-fit →
   bundled `sweeper` + `sweeper_position_only` weights; measure + record the deferred §9
   population metrics; HF publish.
3. **Finalize:** swap the toy for the real bundled weights, regenerate the per-artifact goldens (chirality /
   feature-contract fingerprints come from the shipped artifact), re-run the full suite, then — **only with
   the real weights in place** — the version bump / ADR-083 / commit / merge / tag / publish, each a
   separate owner go-ahead.

**No tag ships on a toy model.** The wheel that ships carries the real DGX-trained weights.

---

## 13. CI gates (summary)
Two-sided saturation-vs-tracking (§7.1); per-variant golden/chirality/feature-contract/integrity/round-trip
(§7.2); `default` byte-identity (§4.2); velocity-availability for the new pair (§10); publisher
contract-required assertion (§8); public-corpus fail-closed on the bundled variant (§6); numba/numpy leaf
parity inherited (ADR-076). Lint at CI scope; pyright bare; `tests/tracking/` with `--benchmark-skip`.

---

## 14. Attribution (ADR-005 / NOTICE)
No new methodology beyond TF-18's (ghosting; Le et al. 2017). The variant re-fits an existing model class on
an extended label domain — no new citation required; the parent spec's TF-60 NOTICE block already covers the
rest-defense lineage. (If the KDE density grid is threaded per-model, no new attribution either.)

---

## 15. Rejected alternatives
- **Reuse the shipped `default` model with an honest note.** Rejected — the §9 finding shows a hard 30 m
  ceiling that compresses the aggressive-sweeper signal the metric exists to reward; a note cannot fix a
  structural label filter.
- **A parametric / percentile league-average sweeper shape.** Rejected — the parent spec §19 already
  rejected a parametric ghost (ignores game state; a corpus-percentile shape is a consumer-side
  frozen-exogenous artifact). The trained model conditioned on frame state is the gold-standard
  counterfactual.
- **Extend `default`'s own grid (retrain in place).** Rejected — that changes GKDV's served model (a GKDV
  retrain + behaviour change). The variant must be **additive**; `default` stays frozen.
- **Collect new high-sweeper data.** Rejected — the labels already exist in the corpus, discarded by the
  cap; a re-fit recovers them.

---

## 16. Resolved decisions + open review items

**Resolved this cycle (2026-08-30):**
1. **Density-grid scope (§4.2) — FAIL-LOUD, not thread-through.** `predict_density` raises on a non-default
   grid; no KDE-kernel change; byte-identical `default`; the grid is first-class for the mean/serve path.
   Rationale: the bundled variant has no density path, no arm-cycle consumer needs extended-grid density,
   and a loud refusal beats speculative capability at higher risk to the frozen `default` (YAGNI +
   minimal-correct).
2. **`x_max` = 52.5 m (§5)** — halfway, y unchanged; corpus-validated in the DGX step (revise down only if
   the empirical envelope argues for a tighter ceiling — a grid-size nicety, not a correctness change).
3. **High-sweeper provider coverage (§6) — reported, not gated.** Per the repo's report-don't-gate
   convention. A per-provider >30 m coverage figure + a >30 m-stratum MAE go into `metrics.json`; the
   stratum-MAE acceptance bar is set with the DGX evidence, not pre-committed to an arbitrary number. A
   hard minimum-per-provider gate is rejected (arbitrary, and against the convention).
4. **Variant naming (§9) — `sweeper` / `sweeper_position_only`** (capability-descriptive, reusable; final
   pick invited in review — trivial to change, no back-compat).

**Open review items:**
- **Toy-model gate fidelity (§7.1):** the saturation gate is authored against a locally-fit toy extended-
  grid model; confirm the qualitative tracks-vs-saturates assertion is robust to the toy→real weights swap
  (assert *direction*, never a toy magnitude).
- **The load-bearing additivity claim (§2/§3):** the "`default`/`position_only`/`full` frozen → no GKDV
  retrain" property — the reviewer flagged this for explicit scrutiny. It rests on (a) `DEFAULT_GHOST_GRID`
  equalling the current constants so the refactor is a `default` no-op, gated by the three-way byte-identity
  of §4.2 (unchanged behavioural goldens + grid_spec-field identity + re-save-differs-only-in-feature_contract
  — NOT a full-SHA round-trip, which is unachievable), and (b) the new variant being an additive artifact GKDV
  never selects (GKDV keeps `default`).
- **Final variant-name pick (§9).**
