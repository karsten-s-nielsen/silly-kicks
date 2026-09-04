# TF-54 + TF-55 Player-Quality Bundle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:executing-plans (inline) to implement
> this plan task-by-task. Steps use checkbox (`- [ ]`) syntax. **This plan deliberately overrides the
> skill's per-step commit cadence** — see Global Constraints.

**Goal:** Ship two per-player descriptive quality metrics — TF-54 territorial dominance and TF-55
Glicko-2 duel ratings — as two sibling event-only `compute_*` packages in one PR (two commits).

**Architecture:** Two independent packages (`silly_kicks/territory/`, `silly_kicks/duels/`) each
mirroring the `shot_stopping/` / `restdefense/` idiom (frozen params, `_columns`/`_compute`/`_report`,
flat `__all__`). TF-54 = trimmed defensive hull × injected xT (a `method=` valuation family, default
`completed_failed`, `counterfactual` reserved); TF-55 = a pure `update_glicko` primitive + a resumable
`compute_duel_ratings` orchestrator. Both are windowed-API-over-atoms (NEW pattern; per-`(player,
match)` atom + optional `window=`). Neither is an `add_*` aggregator (C4 aggregator count stays 33).

**Tech Stack:** Python, pandas, numpy, scipy (`ConvexHull`, point-in-hull — already a dep),
`silly_kicks.xthreat.ExpectedThreat` (TF-54, injected), `silly_kicks.id_compat`, `silly_kicks.spadl`.

**Spec:** `docs/superpowers/specs/2026-09-03-tf54-tf55-player-quality-bundle-design.md` — the plan
argues from it; executors read both.

## Global Constraints

Every task's requirements implicitly include this section (values copied verbatim from the spec §3).

- **No version / PR-S / ADR number until commit-prep.** Assign at Task B7 after `git fetch && git
  merge origin/main`; NOBODY reserves. Placeholders `<VERSION>` / `PR-S<N>` / `ADR-NNN` everywhere.
- **Additive — no VAEP/tracking retrain, no re-materialize.** Both are new `compute_*` modules.
- **Event-only.** Each package imports `spadl` / `id_compat` / (TF-54 only) `xthreat` + pandas /
  numpy / scipy ONLY — NEVER `silly_kicks.tracking`. Pinned by a per-package AST import-allowlist
  test (the `tests/shot_stopping/test_import_allowlist.py` idiom, with planted-violation meta-tests).
- **`compute_*`, not `add_*`.** C4 aggregator count stays **33**; two new containers added, C4
  re-rendered via Graphviz **dot** (`C:/Users/Karsten/.claude/tools/graphviz/dot.exe`), never Smetana.
- **ids via `id_compat` (ADR-019); output ids RAW.** Group/join/compare via `ids_equal`/`same_id`/
  `canonical_id_series`; emit raw ids (the TF-59 precedent). Any grouping on an id uses a canonical
  key + emits the raw id via `.first()` (the ADR-085 IMPL-01 lesson).
- **Order-insensitive (ADR-065).** Any positional/`.shift()`/trajectory-sequential logic sorts
  chronologically first via the robust `(game_id, period_id, time_seconds, action_id)` key.
- **Honest coverage, never fabricate (ADR-042 / ADR-027).** Unscoreable player/duel/pass →
  dropped-and-**counted** in a conserving report; never scored 0 or silently NaN.
- **Frozen params + `for_provider`** per package (override map EMPTY until an ADR-009 apply-gate).
- **Purity (ADR-033).** Every public entry returns NEW objects; never mutates a caller input.
- **COMMIT CADENCE (override of the skill default):** do **NOT** commit per step. TDD each step
  (write failing test → run red → implement → run green) but commit ONLY at the two feature gates
  (Task A7, Task B7). Each commit is a fully-tested coherent state. **No commit without explicit
  per-commit owner approval.**
- **Owner-confirmed decisions (TF-54/55 brainstorm; confirmed 2026-09-04; formal sink = commit-prep
  ADR):** (1) indeterminate duels → **exclude** (not draw=0.5), counted; (2) **complete real-data
  e2e green before each commit**; (3) **windowed-API-over-atoms grain** for both (TF-55 extends the
  TODO "rating period = match" by a thin trajectory-slice helper).
- **Source real e2e data from a public runtime API** (pining, or StatsBomb open data via statsbombpy
  — the sibling restdefense-e2e pattern; importorskip-guarded), never a downloaded/committed data
  folder. IMPL-10: TF-54's e2e uses statsbombpy; a native-duel provider for TF-55 may still use pining.
- **Feature branch, one PR, no worktrees.** Branch off the merged default branch.

## File Structure

**Create — TF-54 (`silly_kicks/territory/`):** `__init__.py`, `_columns.py`, `_config.py`,
`_hull.py`, `_compute.py`, `_report.py`.
**Create — TF-55 (`silly_kicks/duels/`):** `__init__.py`, `_columns.py`, `_config.py`,
`_extract.py`, `_compute.py`, `_report.py`.
**Create — tests:** `tests/territory/{__init__,test_hull,test_compute,test_reflection_invariance,
test_columns,test_config,test_import_allowlist,test_purity,test_e2e}.py`;
`tests/duels/{__init__,test_glicko_primitive,test_extract,test_compute,test_columns,test_config,
test_import_allowlist,test_purity,test_e2e}.py`.
**Modify:** `silly_kicks/feature_glossary.py` (2 new `compute_*` legs), `NOTICE`,
`tests/invariants/glossary_emitted_columns.py` (2 new legs), `tests/test_public_api_examples.py`
(register the 8 new public modules), `docs/c4/architecture.dsl` (+2 containers), `docs/c4/
architecture.html` (re-render). **At Task B7 (commit-prep) only:** `silly_kicks/_version.py`,
`CHANGELOG.md`, `TODO.md`, new `docs/superpowers/adrs/ADR-NNN-*.md`.

---

# GROUP A — TF-54 Territorial Dominance (→ Commit 1)

### Task A1: `_columns.py` + `_config.py`

**Files:** Create `silly_kicks/territory/_columns.py`, `silly_kicks/territory/_config.py`; Test
`tests/territory/test_columns.py`, `tests/territory/test_config.py`.

**Interfaces — Produces:**
- `TERRITORY_COLUMNS: dict[str, str]` (schema) + per-column name constants + `TERRITORY_METHODS:
  frozenset[str]` (`{"completed_failed", "counterfactual"}`).
- `TerritoryParams` (frozen dataclass): `trim_fraction: float = 0.70`, `forward_threshold_m: float
  = 0.0`, `defensive_action_types: tuple[str,...] = ("tackle","interception","clearance")`,
  `own_half_max_x: float = 52.5`, `_is_universal_default` + `.default`/`.for_provider`/`.is_default`.
  `_PROVIDER_TERRITORY_PARAMS: dict = {}` (empty, ADR-009).

**Steps:**
- [ ] **A1.1** Write `tests/territory/test_config.py`: `TerritoryParams.default().is_default()` True;
  `default(force_universal=True).is_default()` False; `for_provider("statsbomb") == TerritoryParams()`.
  (Mirror `tests/shot_stopping/test_config.py`.)
- [ ] **A1.2** Write `tests/territory/test_columns.py`: `set(TERRITORY_COLUMNS)` == the pinned column
  set (§5.6 of the spec); dtypes (`game_id`/`player_id`/`hull_source` `object`; `passes_into_hull`/
  `defensive_actions_in_hull` `Int64`; xT/area/coords/rates `float64`).
- [ ] **A1.3** Run both → FAIL (modules absent).
- [ ] **A1.4** Implement `_columns.py` + `_config.py` (copy the `shot_stopping` `_config.py` method
  bodies verbatim, retyped for `TerritoryParams`; each method carries the `>>>` doctest examples the
  public-api gate requires).
- [ ] **A1.5** Run both → PASS.

### Task A2: `_hull.py` — trimmed defensive hull

**Files:** Create `silly_kicks/territory/_hull.py`; Test `tests/territory/test_hull.py`.

**Interfaces — Produces:**
- `build_trimmed_hull(defensive_actions_xy: np.ndarray, *, trim_fraction: float) -> Hull | None` —
  keep the `trim_fraction` fraction of points nearest the centroid, `scipy.spatial.ConvexHull` over
  survivors; `None` if < 3 non-collinear survivors (degenerate).
- `Hull` value object: `.contains(xy: np.ndarray) -> np.ndarray[bool]` (vectorized point-in-hull via
  half-plane test or `Delaunay.find_simplex >= 0`), `.area: float`, `.centroid: tuple[float,float]`.

**Steps:**
- [ ] **A2.1** Write `tests/territory/test_hull.py`: (a) a square of 4 points → hull area exact
  (e.g. 100.0 for a 10×10), centroid exact, a point inside → True, outside → False, on-edge behavior
  pinned; (b) trim: 8 points where 2 are far outliers, `trim_fraction=0.75` → hull excludes the 2
  outliers (area shrinks to the inlier hull); (c) degenerate: 2 points → `None`; 3 collinear → `None`.
- [ ] **A2.2** Run → FAIL.
- [ ] **A2.3** Implement `_hull.py`. Trim = sort by squared distance to centroid, keep the nearest
  `ceil(n * trim_fraction)`; `ConvexHull`; contains via `Delaunay(hull_points).find_simplex(pts) >= 0`.
  Guard `< 3` survivors AND a `QhullError`/collinear → return `None`.
- [ ] **A2.4** Run → PASS.

### Task A3: `_report.py`

**Files:** Create `silly_kicks/territory/_report.py`; tested via A4's compute tests.

**Interfaces — Produces:** `TerritoryReport` (frozen): `n_players_in`, `n_scored`,
`n_degenerate_hull`, `n_no_actions`, `n_passes_considered`, `n_passes_into_hull`, and a documented
conservation invariant `n_scored + n_degenerate_hull + n_no_actions == n_players_in`. Carries a
`>>>` conservation doctest (public-api gate).

**Steps:**
- [ ] **A3.1** Implement `_report.py` (mirror `ShotStoppingReport`). Conservation asserted by A4.

### Task A4: `_compute.py` — the core (the load-bearing task)

**Files:** Create `silly_kicks/territory/_compute.py`; Test `tests/territory/test_compute.py`,
`tests/territory/test_reflection_invariance.py`.

**Interfaces — Consumes:** A1 (`TERRITORY_COLUMNS`, `TerritoryParams`), A2 (`build_trimmed_hull`),
A3 (`TerritoryReport`), `xthreat.require_fitted_xt` + **`ExpectedThreat.rate`** (the PUBLIC scorer —
NOT the private `_physical.values_at_points`), the inline point reflection `(105 − x, 68 − y)` using
`spadl.config` pitch dims (or `silly_kicks.reflection`; there is **no** `_geometry` module — PLAN-08),
`id_compat`.
**Produces:**
```python
def compute_territorial_dominance(
    actions: pd.DataFrame, *, xt: "ExpectedThreat",
    method: str = "completed_failed", window: "Collection | None" = None,
    params: TerritoryParams = _DEFAULT,
) -> tuple[pd.DataFrame, TerritoryReport]:
```
(`ExpectedThreat` imported `TYPE_CHECKING`-only, duck-typed at runtime — the ADR-022 idiom.)

**Key contracts to implement (spec §5.2–§5.5):**
- `require_fitted_xt(xt, caller="compute_territorial_dominance")` FIRST; unknown `method` raises;
  `method="counterfactual"` raises `NotImplementedError("... construct-validated follow-on; spec §2")`.
- Per defender: build the hull from THEIR own-half (`start_x < own_half_max_x` in the defender's
  action-LTR frame) defensive actions (`defensive_action_types`), trimmed. Degenerate → drop-and-count.
- For each OPPONENT pass: reflect its `end` into the defender's frame `(105 − end_x, 68 − end_y)` for
  the point-in-hull membership; compute `xt` on the pass `end` in the OPPONENT's own frame. Completed
  → `xt_conceded`; failed → `xt_prevented` (**both use the recorded `end`; a failed pass's `end` is
  the death location — spec §5.3 SHOULD-FIX 1**). `xt_net = conceded − prevented`.
- Forward-flag companions (`_forward` = destination x − origin x > `forward_threshold_m`); volume
  (`passes_into_hull`) + rates (`_rate` = total / volume, NaN on zero volume); hull geometry
  (`area`, `centroid_x/y`); `defensive_actions_in_hull` count; `hull_source` provenance.
- Grouping on `(canonical game, canonical player)` (ADR-019), raw ids emitted via `.first()`.
- `window=None` → per-`(player, game)` atoms; a `window` (a `Collection` of `game_id`s) → additive
  aggregation, hull re-derived over the window's pooled actions (spec §5.5).

**Steps:**
- [ ] **A4.1** Write `tests/territory/test_compute.py` (analytic, exact):
  - a hand-placed defender action set → known hull; two opponent completed passes (one into, one out
    of the hull) with a toy xT → exact `xt_conceded`; two failed passes similarly → exact
    `xt_prevented`; `xt_net`.
  - **SHOULD-FIX 1:** a FAILED pass whose SPADL `end` (death location) ≠ its intended target →
    membership + xT both use the `end`; assert the documented under-count (a pass "aimed in" but with
    death `end` OUTSIDE the hull is NOT counted as prevented).
  - forward-flag split; `passes_into_hull` + rate NaN on zero volume; degenerate hull → row dropped,
    `TerritoryReport.n_degenerate_hull == 1`, conservation holds.
  - `require_fitted_xt` raises on `None`/`"grid"`/unfitted; `method="counterfactual"` raises
    `NotImplementedError`; unknown method raises.
  - RAW-id output; mixed-dtype id grouping does not fragment (int 7 vs "7" → one row) (ADR-019).
  - **`window=` aggregation (PLAN-06):** a 2-game window (a `Collection` of `game_id`s) → additive
    `xt_conceded`/`xt_prevented` over the pooled games + the season hull **re-derived** over the
    pooled actions (spec §5.5); `window=None` returns the per-game atoms unchanged.
- [ ] **A4.2** Write `tests/territory/test_reflection_invariance.py`: build one physical scene; score
  it with the defender as home-attacking then as away-attacking (reflect the whole frame). Assert
  `xt_conceded`/`xt_prevented` **identical per row** (ADR-028/D3). Non-vacuity: a version that skips
  the reflection scores them DIFFERENTLY (assert the delta is non-zero without the reflection).
- [ ] **A4.3** Run both → FAIL.
- [ ] **A4.4** Implement `_compute.py` (+ finalize `_report.py`).
- [ ] **A4.5** Run both → PASS. **Red-green:** flip the completed/failed assignment → conceded/
  prevented swap → tests fail → restore.

### Task A5: `__init__.py` + import-allowlist + purity

**Files:** Create `silly_kicks/territory/__init__.py`; Test `tests/territory/test_import_allowlist.py`,
`tests/territory/test_purity.py`.

**Steps:**
- [ ] **A5.1** `__init__.py`: flat `__all__` (`TERRITORY_COLUMNS`, `TerritoryParams`,
  `TerritoryReport`, `compute_territorial_dominance`, + any public column constants).
- [ ] **A5.2** `test_import_allowlist.py` (copy `tests/shot_stopping/test_import_allowlist.py`; ban
  `silly_kicks.tracking`; keep its planted-violation meta-tests). Run → PASS.
- [ ] **A5.3** `test_purity.py`: `compute_territorial_dominance` never mutates `actions` (snapshot
  every array arg; value-equality; `out is not actions`). Register a 2nd variant if a branch adds a
  conditional column (ADR-033). Run → PASS.

### Task A6: glossary + NOTICE + public-api + C4 (territory)

**Files:** Modify `silly_kicks/feature_glossary.py`, `NOTICE`,
`tests/invariants/glossary_emitted_columns.py`, `tests/test_public_api_examples.py`,
`docs/c4/architecture.dsl`, `docs/c4/architecture.html`.

**Steps:**
- [ ] **A6.1** Add a `_territory_columns()` leg to `glossary_emitted_columns.py` (mirror the
  `_shot_stopping_columns()` leg) + wire into `emitted_columns()`.
- [ ] **A6.2** Add a `FeatureColumn` per TERRITORY metric column to `feature_glossary.py` with
  `emitting_module="silly_kicks.territory._compute"`, units (`xG` for xT cols, `m^2` for area,
  `count`), `higher_is_better` (conceded no; prevented yes; net context — set per column) + attribution.
- [ ] **A6.3** Add NOTICE entry (Sumpter / Twelve, §11) verbatim; docstrings cross-link "See NOTICE".
- [ ] **A6.4** Register the 3 discovered territory public modules (`_columns`/`_compute`/`_report`)
  in `tests/test_public_api_examples.py::_PUBLIC_MODULE_FILES` (the ADR-085 PR2 lesson — a new public
  module fails `test_derived_surface_is_fully_accounted_for` until registered); each public symbol
  carries a real Examples section (methods `>>>`; `compute_*` a usage-sketch literal block).
- [ ] **A6.5** Add a `territory` container + relationships to `architecture.dsl`; re-render
  `architecture.html` via `dot` (per the CLAUDE.md render pipeline); verify clean (aggregator count
  still 33). Run the C4 completeness gate → PASS.
- [ ] **A6.6** Run the full non-e2e suite for the touched surface (glossary coverage, public-api,
  id-scalar/id-dtype registries, purity, C4) → PASS.

### Task A7: TF-54 real-data e2e + **Commit-1 gate**

**Files:** Create `tests/territory/test_e2e.py` (`@pytest.mark.e2e`).

**Steps:**
- [ ] **A7.1** `test_e2e.py`: load real matches via **statsbombpy** (StatsBomb open data — the sibling
  restdefense-e2e pattern; importorskip-guarded); **PINNED xt recipe (PLAN-07 / IMPL-10):** fit
  `xt = ExpectedThreat(method="singh_counts").fit(corpus_actions)` — the classic deterministic grid —
  where `corpus_actions` is **all-but-the-scored-match, scored match EXCLUDED** from the fit (no bundled
  artifact); actions stay in the raw per-acting-team-LTR frame (NO `play_left_to_right`, ADR-028). If
  statsbombpy / the network is unavailable the `@e2e` **skips cleanly**. Run
  `compute_territorial_dominance`. Assert: columns/dtypes conform, values finite
  + plausible, `TerritoryReport` conserves, reflection-invariance holds on real away-team defenders.
- [ ] **A7.2** Run the TF-54 e2e on real data → GREEN (statsbombpy + network). **DONE this cycle:** ran
  green on StatsBomb WC2022 open data (it caught the real `type_id`-vs-`type_name` bug, since fixed).
- [ ] **A7.3** Run the FULL CI-faithful non-e2e suite (`pytest -m "not e2e" -p no:randomly`) + ruff
  (CI scope) + pyright (bare) → all green.
- [ ] **A7.4** **COMMIT-1 GATE (owner-approval required):** present `git status` + the diff + the
  TF-54 real-data e2e evidence; **STOP** for explicit owner approval. On approval: `git add -A &&
  git commit` (message: `feat(territory): TF-54 territorial dominance ...`, trailers). **No version
  bump in this commit** (release bookkeeping is Task B7). No push yet.

---

# GROUP B — TF-55 Glicko-2 Duel Ratings (→ Commit 2)

### Task B1: `_columns.py` + `_config.py`

**Files:** Create `silly_kicks/duels/_columns.py`, `_config.py`; Test `tests/duels/test_columns.py`,
`test_config.py`.

**Interfaces — Produces:**
- `DUEL_COLUMNS: dict[str,str]` (§5b.5) + name constants.
- `DuelRatingParams` (frozen): `initial_rating: float = 1500.0`, `initial_rd: float = 350.0`,
  `initial_volatility: float = 0.06`, `tau: float = 0.5`, `apply_inactivity_rd_growth: bool = True`,
  `+ .default`/`.for_provider`/`.is_default`; `_PROVIDER_DUEL_PARAMS = {}`.
- `GlickoState` (frozen): `rating`, `rd`, `volatility`.

**Steps:** mirror A1 (config doctests; column set + dtypes: ids/source `object`, counts `Int64`,
rating/rd/volatility `float64`). Write tests → red → implement → green.

### Task B2: `_extract.py` — duel extraction

**Files:** Create `silly_kicks/duels/_extract.py`; Test `tests/duels/test_extract.py`.

**Interfaces — Produces:** `extract_duels(actions) -> tuple[list[DuelGame], DuelExtractReport]` where
`DuelGame = (game_id, period_id, time_seconds, winner_player, winner_team, loser_player, loser_team,
source)`; `source ∈ {"native","derived"}`. Strategy chosen at frame-set granularity
(`labeling_strategy` = native if `tackle_winner_player_id` present & populated, else derive).

**Contracts:** native = sportec `tackle_winner_*`/`tackle_loser_*` (ADR-001, NaN elsewhere); derived
= `tackle`/`take_on` result adjacency (a successful tackle beats the take_on actor and vice-versa —
pin the exact adjacency rule from the SPADL result); ground duels only; **indeterminate excluded +
counted** (no clear winner). All ids via `id_compat`.

**Steps:**
- [ ] **B2.1** `test_extract.py`: (a) sportec-shaped actions with native winner/loser → clean
  `DuelGame`, `source="native"`; (b) non-native actions with a `tackle`/`take_on` adjacency →
  derived winner/loser, `source="derived"`; (c) an indeterminate contest → excluded, report count +1;
  (d) `DuelExtractReport` conserves (`n_native + n_derived + n_excluded == n_candidate`).
- [ ] **B2.2** Run → FAIL. **B2.3** Implement. **B2.4** Run → PASS.

### Task B3: `update_glicko` primitive

**Files:** Add to `silly_kicks/duels/_compute.py`; Test `tests/duels/test_glicko_primitive.py`.

**Interfaces — Produces:** `update_glicko(ratings: Mapping[player, GlickoState], period_games:
Sequence[tuple], *, params: DuelRatingParams) -> dict[player, GlickoState]` — pure Glicko-2 for ONE
rating period (the standard τ / scale-480.../μ,φ transform, volatility Illinois-algorithm iteration,
inactivity RD growth for a player with no game this period).

**Steps:**
- [ ] **B3.1** `test_glicko_primitive.py`: reproduce **Glickman's published worked example** to the
  documented tolerance (the canonical numeric oracle: the player at 1500/200 vs opponents
  1400/30, 1550/100, 1700/300 with results 1/0/0 → published new rating ≈1464.06, RD ≈151.52,
  σ ≈0.05999). + inactivity-only period → rating unchanged, RD grows.
- [ ] **B3.2** Run → FAIL. **B3.3** Implement the Glicko-2 math. **B3.4** Run → PASS. **Red-green:**
  perturb a constant → the worked-example value moves off → fails → restore.

### Task B4: `compute_duel_ratings` orchestrator + `_report.py`

**Files:** `silly_kicks/duels/_compute.py`, `_report.py`; Test `tests/duels/test_compute.py`.

**Interfaces — Produces:**
```python
def compute_duel_ratings(
    actions: pd.DataFrame, *, initial_ratings=None, window: "Collection | None" = None, params=_DEFAULT,
) -> tuple[pd.DataFrame, DuelRatingReport]:
```
Walks matches chronologically (ADR-065 sort), threads state via `update_glicko`, emits per-`(player,
game)` snapshots (`duel_rating`/`_deviation`/`_volatility` + `duels_contested/won/lost` +
`duel_winner_source`) + final ratings; `initial_ratings` resumes. `window=` = trajectory slice
(§5b.4). `DuelRatingReport` conserves native/derived/excluded + `n_players`.

**Steps:**
- [ ] **B4.1** `test_compute.py`: (a) a 2-match sequence → per-match snapshots match a hand-threaded
  `update_glicko`; (b) **resume-equivalence:** `compute_duel_ratings(all_matches)` final ratings ==
  two calls threaded via `initial_ratings` (byte-equal); (c) chronological-order-insensitivity
  (permuted input rows → identical result, ADR-065); (d) `window=` slice returns rating-as-of-end;
  (e) RAW ids; report conserves.
- [ ] **B4.2** Run → FAIL. **B4.3** Implement orchestrator + `_report.py`. **B4.4** Run → PASS.

### Task B5: `__init__.py` + import-allowlist + purity

Mirror A5 for `silly_kicks/duels/` (`__all__`: `DUEL_COLUMNS`, `DuelRatingParams`, `GlickoState`,
`DuelRatingReport`, `update_glicko`, `compute_duel_ratings`, `extract_duels`). Run → PASS.

### Task B6: glossary + NOTICE + public-api + C4 (duels)

Mirror A6 for duels: `_duel_columns()` glossary leg; `FeatureColumn` per column
(`emitting_module="silly_kicks.duels._compute"`; `duel_rating` higher-is-better yes); NOTICE
(Glickman, §11); register the discovered duels public modules in `_PUBLIC_MODULE_FILES` (each symbol
an Examples section — `update_glicko` a real `>>>`; `compute_duel_ratings` a usage-sketch block); add
a `duels` container to `architecture.dsl` + re-render `architecture.html` via `dot` (count still 33).
Run the touched-surface gates → PASS.

### Task B7: TF-55 real-data e2e + release commit-prep + **Commit-2 gate**

**Files:** Create `tests/duels/test_e2e.py` (`@e2e`). At commit-prep: `silly_kicks/_version.py`,
`CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-NNN-*.md`.

**Steps:**
- [ ] **B7.1** `test_e2e.py`: load a **sportec** match (native duels) + a **derive-provider** match
  (e.g. gradientsports/skillcorner via pining); run `compute_duel_ratings`. Assert ratings evolve,
  `DuelRatingReport` native/derived/excluded conserve, resume-equivalence holds on the real trajectory,
  both winner paths exercised.
- [ ] **B7.2** Run the TF-55 e2e on real data → GREEN (owner-run; pining access).
- [ ] **B7.3** **Commit-prep (assign numbers NOW):** `git fetch && git merge origin/main` (BOM/CRLF
  care); assign `<VERSION>`/`PR-S<N>`/`ADR-NNN` from the merged state; bump `silly_kicks/_version.py`
  (single source); `uv lock` if needed; write the `ADR-NNN` (records the bundle + the 3 owner-confirmed
  decisions + rejected-alts B/C from spec §6); add the CHANGELOG `[<VERSION>]` entry (both features,
  additive/no-retrain); groom `TODO.md` (remove the shipped TF-54/TF-55 On-Deck rows; keep the
  `counterfactual`-method follow-on as a new backlog row — **owner-approve any new backlog row**);
  final C4 re-render (both containers) via `dot`.
- [ ] **B7.4** Run the FULL non-e2e suite + ruff + pyright + all C4/glossary/public-api gates → green.
- [ ] **B7.5** **COMMIT-2 GATE (owner-approval required):** present `git status` + diff + the TF-55
  real-data e2e evidence + the version/ADR numbers; **STOP** for explicit owner approval. On approval:
  `git commit` (message: `feat(duels): TF-55 Glicko-2 duel ratings + release <VERSION> ...`, trailers).
- [ ] **B7.6** On owner go-ahead (separate): push the branch, open ONE PR (both commits), watch CI to
  green, then the owner tag+publish flow.

---

## Self-Review (author checklist, run against the spec)

1. **Spec coverage:** §5 (TF-54 hull/method-family/reconciliation/enrichments/windowing) → A2/A4;
   §5b (TF-55 primitive/orchestrator/extraction/params/windowing) → B2/B3/B4; §6 rejected-alts →
   ADR (B7); §7 packaging/glossary/NOTICE/C4 → A6/B6; §8 two-tier testing incl. real-data e2e →
   A4/A7/B3/B7; §9 two-commit gates → A7/B7. No section unmapped.
2. **Placeholder scan:** version/PR/ADR are intentional `<...>` placeholders (Global Constraints);
   no TBD/"add error handling"/"similar to". Code steps carry real signatures + real test assertions.
3. **Type consistency:** `TerritoryParams`/`DuelRatingParams`/`GlickoState`/`TerritoryReport`/
   `DuelRatingReport` names consistent A1↔A3↔A4 and B1↔B3↔B4↔B7; `compute_*` signatures match the
   spec verbatim. `method` names match `TERRITORY_METHODS`.

## References (NOTICE, spec §11)
- TF-54: Sumpter *Soccermatics*; Twelve.football "Earpiece" (module 10.2).
- TF-55: Glickman, *Glicko-2*; StatsBomb HOPS (module 10.1).
