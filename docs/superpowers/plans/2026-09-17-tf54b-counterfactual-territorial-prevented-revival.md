# TF-54b Path B (revival) — counterfactual territorial "threat prevented" Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended)
> or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Revive the event-only `method="counterfactual"` territorial "threat prevented" valuation onto
current `main`, re-fit + re-bundle `PassCompletionModel` on the full public open-data corpus, and add a
crossed defender+team-cell census that gates whether a defender ranking is licensed.

**Architecture:** `silly_kicks.territory.compute_territorial_dominance` gains back a
`method="counterfactual"` door (removed on `main` — ADR-090 gave the name to the tracking sibling). The
cone valuation `Σ q·c·xT` is ported verbatim from branch `ab9001c` onto main's v1-only territory files.
The reusable seams it needs (`xthreat.destination_profiles`, `expected_passing.PassCompletionModel`,
`territory.build_trimmed_hull`, `scripts/_sb_open_data`) are already on `main`. A new owner-run census
driver + a pure-numpy crossed variance-components ICC (scripts-side, not a library API) decide the
ranking gate. Two commits: code (clean tree), then bundled weights + validation/census artifacts (both
stamping Commit 1).

**Tech Stack:** Python, numpy, pandas, scipy (hull), pure-numpy logistic serve; `statsbombpy` (scripts-only
network dep, `importorskip`-guarded); ADR-052 `for_each` sharding; ADR-011 bundled-artifact discipline.

**Spec:** `docs/superpowers/specs/2026-09-17-tf54b-counterfactual-territorial-prevented-revival-design.md`
(R2 APPROVE). Read it alongside this plan; the plan argues the spec.

**Port source:** branch `origin/feat/tf54b-counterfactual-territorial-prevented` @ `ab9001c`. Extract any
file with `git show ab9001c:<path>`. The branch's territory files are v1+cone **supersets** of main's
v1-only files; the port re-applies the cone additions.

---

## Global Constraints (copied verbatim from the spec §3)

- **Event-only import graph** — `territory` / `expected_passing` import `spadl` / `id_compat` /
  (`xthreat` for territory) / numpy / scipy ONLY, **never** `tracking`. AST allowlist gates.
- **Injected fitted ports** — `xt: ExpectedThreat` and `completion_model: PassCompletionModel` injected,
  `TYPE_CHECKING`-only import + duck-typed (ADR-022). Value lookups via `values_at_points` /
  `destination_profiles`, **never** raw `.xT` / `.transition_matrix` / `rate`.
- **Bundled-artifact discipline (ADR-011/016/040/044/050)** for `PassCompletionModel`: pickle-free JSON +
  `SHA256SUMS`, feature contract, chirality probe, fail-closed load, inference imports no sklearn.
- **Canonical-id grouping** (ADR-019); **drop-and-count conservation** (ADR-042); **ADR-028** 180° point
  reflection `(x,y)→(fl−x, fw−y)`; **purity** (ADR-033, ≥2 variants); **artifact provenance**
  (`scripts/_provenance.require_clean_tree` + `run_commit`/`training_commit`; ADR-052 `for_each`; ADR-056
  `_input_contract.declare_inputs`).
- **`for_provider` ships EMPTY** (ADR-009) for `TerritoryParams`, `CounterfactualParams`, and the census.
- **`completed_failed` is byte-identical** — same shape and values with the cone code present.
- **Census/gate thresholds + the elite-defender prior are LOCKED constants committed in Commit 1** (the
  TF-19 `NAMED_KEEPER_PRIOR` idiom); `decide_promotion` / the census gate read ONLY those constants.
- **Version / PR-Snnn NOT locked until Commit 2** — re-derive from `main` at commit-prep (`git fetch &&
  git merge origin/main`). **ADR number NOT locked** — placeholder `ADR-099`, assigned at Commit 2 / release-prep (may not release first).
- **ADR-009 ranking posture** — the library ships only the per-`(defender, match)` primitive; a ranking is
  a reported artifact + an ADR-009 apply, never a library function.
- **Test invocation (repo landmine, applies to every "Run" step)** — tests use
  `./.venv/Scripts/python.exe` (py3.10.19 + pytest 9.0.3; the uv base interpreter has no pytest), with
  `SILLY_KICKS_ASSERT_INVARIANTS=1` (CI sets it) and `-p no:randomly`. ruff/pyright run in the SYSTEM
  interpreter (`python -m ruff` / `python -m pyright`), scoped to `silly_kicks/ tests/ scripts/`, never `.`.
  Never `pip install` into `.venv`. Whole-suite baseline: **9240** collected (`not e2e`) on branch HEAD.

**Verified port facts (already true on `main`):** `PassCompletionModel.predict_completion(origin_x,
origin_y, target_x, target_y) -> np.ndarray`, `.bundled()`, `.fit(actions)`, `.save/.load`;
`xthreat.destination_profiles(xt, ox, oy) -> DestinationProfile(zone_centres, zone_values, probabilities)`;
`territory.build_trimmed_hull(xy, trim_fraction=) -> Hull|None` with `.contains/.area/.centroid`;
`causal.power` exposes `att_power_curve` + `InjectionSpec` + `_resample_clusters` — **no ICC curve**;
`scripts/_sb_open_data`: `assert_statsbomb_open_data_mode()`, `all_open_competitions() -> list[(comp,season)]`,
`load_open_data_matches(competition_id, season_id) -> yields (provider, match_id, actions, frames, home)`.
main's `territory/_config.py`,`_report.py`,`_columns.py` are v1-ONLY (`CounterfactualParams` / cf census
fields / `columns_for_method` all ABSENT — verified by execution).

---

## Task 1: `_columns.py` — cone columns, method set, schema resolver

**Files:**
- Modify: `silly_kicks/territory/_columns.py`
- Test: `tests/territory/test_columns_counterfactual.py` (new)

**Interfaces:**
- Consumes: nothing new.
- Produces: `TERRITORY_METHODS = frozenset({"completed_failed", "counterfactual"})`;
  `columns_for_method(method) -> dict[str,str]`; `_COUNTERFACTUAL_ONLY_COLUMNS`;
  `TERRITORY_TARGET_SOURCE_VALUES = frozenset({"observed","modeled","unresolved"})`; the five
  `TR_*` cone constants (`TR_EXPECTED_THREAT_FACED`, `TR_XT_PREVENTED_ABOVE_EXPECTATION`,
  `TR_PASSES_AIMED_INTO_HULL`, `TR_MEAN_COMPLETION_FACED`, `TR_TARGET_SOURCE`).

- [ ] **Step 1: Write failing test** — `tests/territory/test_columns_counterfactual.py`:

```python
from silly_kicks.territory import _columns as C
def test_completed_failed_schema_is_v1_verbatim():
    assert C.columns_for_method("completed_failed") == dict(C.TERRITORY_COLUMNS)
def test_counterfactual_adds_exactly_five_columns():
    extra = set(C.columns_for_method("counterfactual")) - set(C.TERRITORY_COLUMNS)
    assert extra == {"territory_expected_threat_faced", "territory_xt_prevented_above_expectation",
                     "territory_passes_aimed_into_hull", "territory_mean_completion_faced",
                     "territory_target_source"}
def test_method_set_has_both():
    assert C.TERRITORY_METHODS == frozenset({"completed_failed", "counterfactual"})
def test_unknown_method_raises():
    import pytest
    with pytest.raises(ValueError): C.columns_for_method("nope")
```

- [ ] **Step 2: Run — expect FAIL** (`columns_for_method` absent). `pytest tests/territory/test_columns_counterfactual.py -v`
- [ ] **Step 3: Port the cone additions** from `git show ab9001c:silly_kicks/territory/_columns.py`.
      Apply lines 12–16 (docstring + `TERRITORY_METHODS` with `"counterfactual"`), 75–128 (the five
      `TR_*` constants, `TERRITORY_TARGET_SOURCE_VALUES`, `_COUNTERFACTUAL_ONLY_COLUMNS`,
      `columns_for_method`) onto main's `_columns.py`. Keep main's v1 constant block unchanged.
- [ ] **Step 4: Run — expect PASS.** Also run `pytest tests/territory/ -k "not counterfactual" -q` to
      confirm no v1 column test regressed.
- [ ] **Step 5:** No commit (batched into Commit 1 at the end).

## Task 2: `_config.py` — `CounterfactualParams`

**Files:**
- Modify: `silly_kicks/territory/_config.py`
- Test: `tests/territory/test_counterfactual_params.py` (port from `ab9001c`)

**Interfaces:**
- Produces: `CounterfactualParams(direction_cone_degrees=45.0, min_transition_support=1e-6)` with
  `.default(force_universal=)`, `.for_provider(provider)`, `.is_default()`; `_PROVIDER_COUNTERFACTUAL_PARAMS = {}`.

- [ ] **Step 1: Write failing test** — port `git show ab9001c:tests/territory/test_counterfactual_params.py`.
- [ ] **Step 2: Run — expect FAIL** (`CounterfactualParams` absent). 
- [ ] **Step 3: Port** the `CounterfactualParams` dataclass + `_PROVIDER_COUNTERFACTUAL_PARAMS` from
      `git show ab9001c:silly_kicks/territory/_config.py` (lines 85–152) onto main's `_config.py` (append
      after `TerritoryParams`; the file's header/imports already match).
- [ ] **Step 4: Run — expect PASS.** Doctests too: `pytest --doctest-modules silly_kicks/territory/_config.py -q`.
- [ ] **Step 5:** No commit.

## Task 3: `_report.py` — counterfactual census fields

**Files:**
- Modify: `silly_kicks/territory/_report.py`
- Test: `tests/territory/test_report_counterfactual.py` (port from `ab9001c`)

**Interfaces:**
- Produces: `TerritoryReport` gains `n_target_modeled: int = 0`, `n_target_unresolved: int = 0` (defaults
  preserve every v1 construction).

- [ ] **Step 1: Write failing test** — port `git show ab9001c:tests/territory/test_report_counterfactual.py`.
- [ ] **Step 2: Run — expect FAIL** (fields absent).
- [ ] **Step 3: Port** the two trailing fields + the docstring Notes block from
      `git show ab9001c:silly_kicks/territory/_report.py` (lines 28–52) onto main's `_report.py`.
- [ ] **Step 4: Run — expect PASS.** Confirm a v1 6-arg `TerritoryReport(...)` still constructs.
- [ ] **Step 5:** No commit.

## Task 4: `_counterfactual.py` — the joint `q·c·xT` valuation (NEW, verbatim port)

**Files:**
- Create: `silly_kicks/territory/_counterfactual.py`
- Test: `tests/territory/test_counterfactual_compute.py` (port from `ab9001c`, extended)

**Interfaces:**
- Consumes: `_columns` TR_* constants; `xthreat.destination_profiles`; injected `completion_model`,
  `xt`; `_hull.Hull`; `CounterfactualParams`.
- Produces: `counterfactual_rows(defs_grouped, passes_by_game, *, xt, completion_model, params, fl, fw,
  window) -> (rows, census)`; the `DefenderGroup` type alias.

- [ ] **Step 1: Write failing test** — port `git show ab9001c:tests/territory/test_counterfactual_compute.py`.
      It MUST include the §5.7 worked-example golden (assert to the last digit): `conceded==0.15`,
      `prevented==0.078`, `expected_threat_faced==0.168`, `xt_prevented_above_expectation==0.018`,
      `mean_completion_faced==0.6`, `passes_aimed_into_hull==2`; and the uniform-xT invariant
      `prevented==0.06` exactly. Use a toy `xt` + a toy completion model exposing
      `predict_completion(ox,oy,tx,ty)->const 0.6`.
- [ ] **Step 2: Run — expect FAIL** (module absent).
- [ ] **Step 3: Create** `silly_kicks/territory/_counterfactual.py` from
      `git show ab9001c:silly_kicks/territory/_counterfactual.py` **verbatim** (267 lines). It imports
      `destination_profiles` from `silly_kicks.xthreat` (present on main) and the TR_* from `._columns`
      (Task 1). No edits needed — the branch file targets exactly these seams.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5:** No commit.

## Task 5: `_compute.py` — re-add the door, params, and dispatch

**Files:**
- Modify: `silly_kicks/territory/_compute.py`
- Test: `tests/territory/test_compute.py` (existing, extended) + `tests/territory/test_counterfactual_e2e.py`
  (port from `ab9001c`)

**Interfaces:**
- Consumes: `_counterfactual.counterfactual_rows`, `columns_for_method`, `CounterfactualParams`.
- Produces: `compute_territorial_dominance(actions, *, xt, method="completed_failed", window=None,
  params=_DEFAULT, completion_model=None, cf_params=None)`; internal `_counterfactual_dispatch(...)`.

- [ ] **Step 1: Write failing test** — add a `completed_failed`-byte-identity guard + a
      counterfactual-dispatch smoke to `tests/territory/test_compute.py`, and port
      `tests/territory/test_counterfactual_e2e.py` from `ab9001c`:

```python
def test_completed_failed_byte_identical_to_v1(sample_actions, fitted_xt):
    # snapshot BEFORE the cone code changes _compute (captured on main @ 2ca0ef4), compared after.
    out, rep = compute_territorial_dominance(sample_actions, xt=fitted_xt)  # default method
    assert list(out.columns) == list(TERRITORY_COLUMNS)
    pd.testing.assert_frame_equal(out, EXPECTED_V1_SNAPSHOT)  # committed golden parquet
def test_counterfactual_requires_completion_model(sample_actions, fitted_xt):
    with pytest.raises(ValueError, match="requires a fitted completion_model"):
        compute_territorial_dominance(sample_actions, xt=fitted_xt, method="counterfactual")
def test_counterfactual_dispatch_emits_cf_columns(sample_actions, fitted_xt, toy_completion):
    out, rep = compute_territorial_dominance(sample_actions, xt=fitted_xt, method="counterfactual",
                                             completion_model=toy_completion)
    assert "territory_xt_prevented_above_expectation" in out.columns
    assert rep.n_target_modeled + rep.n_target_unresolved >= 0
def test_two_passes_count_columns_do_not_coincide(aimed_but_not_into_fixture, fitted_xt, toy_completion):
    # PLAN-05 / spec §5.3: passes_into_hull (v1: observed end in hull) and passes_aimed_into_hull
    # (cf denominator: death-cone ∩ hull) are DISTINCT quantities. Fixture has a failed pass aimed into
    # the hull that died SHORT (end outside hull) → aimed>into on that defender.
    out, _ = compute_territorial_dominance(aimed_but_not_into_fixture, xt=fitted_xt,
                                           method="counterfactual", completion_model=toy_completion)
    row = out.iloc[0]
    assert row["territory_passes_aimed_into_hull"] != row["territory_passes_into_hull"]
```

- [ ] **Step 2: Capture the v1 snapshot FIRST** (before editing `_compute.py`): run current main's
      `compute_territorial_dominance(sample_actions, xt=fitted_xt)` and write
      `tests/territory/data/territory_v1_snapshot.parquet` as `EXPECTED_V1_SNAPSHOT`. Run the byte-identity
      test — expect PASS now (unchanged code), so it becomes a real regression guard. Run the two
      counterfactual tests — expect FAIL.
- [ ] **Step 3: Apply the cone dispatch** to `_compute.py`. Diff `git show ab9001c:silly_kicks/territory/_compute.py`
      against main and apply the additions: the `completion_model`/`cf_params` params (lines 74–75), the
      param-resolution block (103–117), `columns_for_method(method)` for `out_columns` (147), the
      `method=="counterfactual"` branch (158–172), and the `_counterfactual_dispatch` function (246–297).
      Keep main's v1 grouping loop body unchanged. Update the module docstring (11–15) to state the door is
      re-added (not "removed"). Update imports: add `TERRITORY_METHODS` stays, add `columns_for_method`,
      `CounterfactualParams`, `counterfactual_rows`, and the cone TR_* to the import block.
- [ ] **Step 4: Run — expect PASS** for all three, and **byte-identity still PASS**. Run full
      `pytest tests/territory/ -q`.
- [ ] **Step 5:** No commit.

## Task 6: `__init__.py` — export the new public surface

**Files:**
- Modify: `silly_kicks/territory/__init__.py`
- Test: `tests/territory/test_public_surface.py` (new, tiny)

**Interfaces:**
- Produces: `CounterfactualParams`, `columns_for_method`, `TERRITORY_TARGET_SOURCE_VALUES` importable
  from `silly_kicks.territory`.

- [ ] **Step 1: Write failing test:**

```python
def test_counterfactual_surface_exported():
    from silly_kicks.territory import CounterfactualParams, columns_for_method, TERRITORY_TARGET_SOURCE_VALUES  # noqa
```

- [ ] **Step 2: Run — expect FAIL.**
- [ ] **Step 3: Add** `CounterfactualParams` (from `._config`), `columns_for_method` +
      `TERRITORY_TARGET_SOURCE_VALUES` (from `._columns`) to the imports and `__all__`.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5:** No commit.

## Task 7: `scripts/_synthetic_interception.py` (NEW, verbatim port) + tests

**Ordering (PLAN-01):** this task precedes Task 8 because Task 8's mechanism-recovery test imports
`perturb_interception` from this module.

**Files:**
- Create: `scripts/_synthetic_interception.py`
- Test: `tests/scripts/test_synthetic_interception.py` (new)

- [ ] **Step 1: Write failing test** — assert the two ground-truth geometry consequences from the module
      docstring: `angle_offset_rad=0 → death == origin + f·(end−origin)` (zero perpendicular distance);
      `angle_offset_rad=δ≠0 → perpendicular distance == f·|v|·|sin δ|` exactly; broadcast shape preserved.
- [ ] **Step 2: Run — expect FAIL** (module absent).
- [ ] **Step 3: Create** from `git show ab9001c:scripts/_synthetic_interception.py` **verbatim** (71 lines,
      pure numpy).
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5:** No commit.

## Task 8: Import-allowlist + purity + reflection-invariance + mechanism-recovery tests

**Files:**
- Modify: `tests/territory/test_import_allowlist.py` (extend for `_counterfactual.py`)
- Modify: `tests/test_add_star_purity.py` — N/A (territory ships no `add_*`); instead the counterfactual
  path's purity is covered by a `_counterfactual` no-mutation test.
- Test (new): `tests/territory/test_counterfactual_reflection_invariance.py`,
  `tests/territory/test_counterfactual_mechanism_recovery.py`

**Interfaces:** consumes Tasks 4–5 and Task 7's `perturb_interception`.

- [ ] **Step 1: Write reflection-invariance test** — one scene scored from either team's perspective
      (reflect all coords `(fl−x, fw−y)` incl. a failed pass whose death ≠ intended target) yields
      per-row-identical cf metrics.
- [ ] **Step 2: Write mechanism-recovery test** — using `scripts/_synthetic_interception.perturb_interception`
      (Task 7, already created), on a fixture the cone-conditioned target estimator recovers the hidden true
      end better than (a) "death = synthetic intercept" and (b) origin-zone centroid. Assert BOTH sides (the
      from-both-sides rule): a wildly-wrong cone (`direction_cone_degrees≈1`) does NOT beat the baselines.
- [ ] **Step 3: Write no-mutation test** — `counterfactual_rows` does not mutate `defs_grouped` /
      `passes_by_game` / input arrays (snapshot + compare).
- [ ] **Step 4: Extend import-allowlist** — add `_counterfactual.py` to the event-only allowlist set with a
      planted-violation meta-test (a `silly_kicks.tracking` import fails the gate).
- [ ] **Step 5: Run all — expect PASS.** No commit.

## Task 9: `scripts/train_pass_completion.py` — add an all-competitions re-fit mode

**Files:**
- Modify: `scripts/train_pass_completion.py`
- Test: `tests/scripts/test_train_pass_completion.py` (extend the existing test if present, else new smoke)

**Interfaces:**
- Consumes: `_sb_open_data.all_open_competitions()`, `load_open_data_matches`.
- Produces: `--all-competitions` flag (mutually exclusive with `--competition-id/--season-id`) that
  iterates `all_open_competitions()` and `for_each`-shards over every `(comp, season)`'s matches; the
  `for_each` shard token includes an `all_competitions` marker (the 4.77.1 stale-shard rule — a different
  corpus MUST invalidate the generation).

- [ ] **Step 1: Write failing test** — with `all_open_competitions` monkeypatched to a 2-tuple stub and
      `load_open_data_matches` to a tiny fixture, `main(["--all-competitions", "--out", tmp, "--allow-dirty"])`
      trains and writes `model.json` + `MODEL_CARD.md` + `metrics.json` whose `n_competitions == 2`; and
      the shard-generation token differs from the single-competition run's token.
- [ ] **Step 2: Run — expect FAIL** (flag absent).
- [ ] **Step 3: Implement** the flag: when set, `competitions = all_open_competitions()`; the corpus loop
      iterates each and `for_each`-shards its matches; `metrics.json` records `n_competitions` and the
      competition list; the card notes "full public open-data corpus". Keep the single-competition default
      path byte-identical. Call `assert_statsbomb_open_data_mode()` before any load.
- [ ] **Step 4: Run — expect PASS.** `-m "not e2e"` for the smoke.
- [ ] **Step 5:** No commit.

## Task 10: `scripts/_crossed_icc.py` — pure-numpy crossed variance-components + bootstrap power (NEW)

**Files:**
- Create: `scripts/_crossed_icc.py`
- Test: `tests/scripts/test_crossed_icc.py` (new)

**Rationale:** `causal.power` has NO ICC curve (verified). The crossed defender+team ICC estimator and its
power leg are NEW work (spec §8 SPEC-03). Scripts-side, not a library API (ADR-009 keeps rankings/analysis
consumer-side).

**Interfaces:**
- Produces:
  - `crossed_variance_components(y, defender_codes, team_codes) -> dict{"var_defender","var_team",
    "var_resid","icc_defender"}` — a two-way crossed random-effects ANOVA / Henderson-III moment estimator
    (fitting-constants reductions via `np.linalg.lstsq` on the defender and team indicator designs;
    negative variance estimates truncated to 0 with a recorded flag). `icc_defender = var_defender /
    (var_defender + var_team + var_resid)`.
  - `bootstrap_icc_ci(y, defender_codes, team_codes, *, n_boot, alpha, rng_seed) -> dict{"icc","lo","hi"}`
    — nonparametric bootstrap resampling **whole defenders** (cluster resample, mirroring
    `causal.power._resample_clusters`'s style — not a call), refit per replicate, percentile CI.

- [ ] **Step 1: Write failing test** — a balanced synthetic crossed design with KNOWN
      `σ²_defender, σ²_team, σ²_resid` (seeded RNG, large n): `crossed_variance_components` recovers each
      within a stated tolerance and `icc_defender` within ±0.03; a defender-null design (`σ²_defender=0`)
      returns `icc_defender≈0` with a bootstrap `lo` that includes 0; a strong-defender design returns
      `lo>0`. Assert the negative-variance truncation flag on a tiny/degenerate design.
- [ ] **Step 2: Run — expect FAIL** (module absent).
- [ ] **Step 3: Implement** both functions in pure numpy. Use Henderson Method III: reductions in the error
      SS from OLS fits of (a) mean only, (b) mean+defender, (c) mean+defender+team, equate expected
      mean-squares to solve for the components; truncate negatives to 0 (record `had_negative_estimate`).
      Bootstrap over unique defender codes.
- [ ] **Step 4: Run — expect PASS.**
- [ ] **Step 5:** No commit.

## Task 11: `scripts/build_territory_ranking_census.py` — census + gate + ranking (NEW)

**Files:**
- Create: `scripts/build_territory_ranking_census.py`
- Test: `tests/scripts/test_build_territory_ranking_census.py` (new)

**Interfaces:**
- Consumes: `_sb_open_data`, `_crossed_icc`, `_driver.for_each`, `_provenance.require_clean_tree`,
  `_input_contract.declare_inputs`, `compute_territorial_dominance(method="counterfactual")`, public
  lineups (via `statsbombpy`, `importorskip`-guarded).
- Produces: a `census.json` (Tier-1 counts always; Tier-2 ICC/power iff Tier-1 clears) + a `ranking.parquet`
  **iff the gate clears**; the LOCKED constants `MIN_MULTI_TEAM_DEFENDERS`, `MIN_PASSES_FACED`,
  `ICC_LOWER_FLOOR`, `POWER_FLOOR`, `ICC_EFFECT_SIZE` (committed here, before the run); a pure
  `census_gate(census) -> {"ranking_licensed": bool, "reason": str}` reading ONLY those constants.

- [ ] **Step 1: Write failing test** (offline, no network): feed a hand-built per-`(defender, game, team)`
      metric table + lineup map into the census's pure functions:
  - Tier-1 counting: distinct defenders; `n_multi_team_defenders` (defenders on ≥2 distinct teams);
    defender×team cells clearing `MIN_PASSES_FACED`; conservation of counts.
  - Gating from BOTH sides: a design with `n_multi_team_defenders < MIN_MULTI_TEAM_DEFENDERS` →
    `ranking_licensed False` (Tier-2 not run); a design clearing Tier-1 with a strong defender signal →
    Tier-2 runs and `ranking_licensed True`; a design clearing Tier-1 but ICC `lo < ICC_LOWER_FLOOR` →
    `ranking_licensed False`.
  - `ranking.parquet` is produced only when licensed.
- [ ] **Step 2: Run — expect FAIL** (module absent).
- [ ] **Step 3: Implement.** Structure the pure core (`tier1_counts`, `tier2_icc`, `census_gate`,
      `build_ranking`) separately from the `main()` I/O (so the tests need no network). `main()`:
      `require_clean_tree` FIRST; `assert_statsbomb_open_data_mode`; `declare_inputs`; `for_each`-shard the
      per-match counterfactual compute (needs a fitted `xt` + `completion_model` fit inline, self-contained,
      on a leakage-disjoint split — **do NOT import from Task 12**, no forward dependency; if a shared helper
      is later wanted, place it in a neutral `scripts/_territory_cf_fit.py` both drivers import, never
      census→validator); build the
      defender×team table from public lineups; run Tier-1 → gate → Tier-2 → gate; write `census.json` +
      (iff licensed) `ranking.parquet`; stamp `run_commit`. `--allow-dirty` records `dirty:true`. Register
      the driver in the ADR-052 population + ADR-056 input-contract gates (`tests/scripts/`).
- [ ] **Step 4: Run — expect PASS** (`-m "not e2e"`).
- [ ] **Step 5:** No commit.

## Task 12: `scripts/validate_territory_counterfactual.py` — port + repoint to broad corpus (NEW)

**Files:**
- Create: `scripts/validate_territory_counterfactual.py`
- Test: `tests/scripts/test_tf54b_drivers.py` (port from `ab9001c`, extended)

**Interfaces:**
- Consumes: `_sb_open_data`, `_synthetic_interception`, `_driver.for_each`, `_provenance`, `_input_contract`,
  `compute_territorial_dominance`, `PassCompletionModel`.
- Produces: `ELITE_DEFENDER_PRIOR` (LOCKED constant, re-drawn for the broad corpus), `decide_promotion`,
  `completion_metrics`, `target_recovery_battery`, `elite_prior_verdict`, `run_battery`, `input_contract`,
  `main()`.

- [ ] **Step 1: Write failing test** — port `git show ab9001c:tests/scripts/test_tf54b_drivers.py`; add a
      test that `main` iterates `all_open_competitions()` (monkeypatched stub) rather than a single
      competition, and that `decide_promotion` reads only the locked constants.
- [ ] **Step 2: Run — expect FAIL** (module absent).
- [ ] **Step 3: Create** from `git show ab9001c:scripts/validate_territory_counterfactual.py`, then apply
      the corpus delta: replace the single `--competition-id/--season-id` corpus with iteration over
      `all_open_competitions()` (keep a `--competitions-json` override, mirroring
      `train_match_outcome_dependence.py`); re-draw `ELITE_DEFENDER_PRIOR` for the broad corpus (locked
      before the run); `require_clean_tree` FIRST; `declare_inputs`; `for_each`-shard; leakage-disjoint
      model fits. Keep the pre-registered battery + `decide_promotion` (gate reads only locked constants).
- [ ] **Step 4: Run — expect PASS** (`-m "not e2e"`; the full corpus run is `@e2e`, owner-run).
- [ ] **Step 5:** No commit.

## Task 13: Glossary + NOTICE + ADR-099

**Files:**
- Modify: `silly_kicks/feature_glossary.py`, `tests/invariants/glossary_emitted_columns.py`, `NOTICE`
- Create: `docs/superpowers/adrs/ADR-099-tf54b-counterfactual-territorial-prevented-revival.md`

- [ ] **Step 1: Write failing test** — the glossary-coverage harness
      (`tests/invariants/glossary_emitted_columns.py`) must require the 4 cone METRIC columns
      (`territory_expected_threat_faced`, `territory_xt_prevented_above_expectation`,
      `territory_passes_aimed_into_hull`, `territory_mean_completion_faced`) under `method="counterfactual"`;
      `territory_target_source` is PROVENANCE and stays EXCLUDED (mirrors `territory_hull_source`). Run —
      expect FAIL (columns undocumented).
- [ ] **Step 2: Add** the 4 `FeatureColumn` records to `feature_glossary.py` (emitting_module
      `silly_kicks.territory._counterfactual`, unit/`higher_is_better` per the spec §5.3 table:
      `xt_prevented_above_expectation` higher-is-better True; document the rate/faced companions), and wire
      the counterfactual-method column set into `glossary_emitted_columns.py`.
- [ ] **Step 3: Run — expect PASS.** Confirm `test_no_stale_entries` still passes (no over-documentation).
- [ ] **Step 4: NOTICE** — add the counterfactual/expected-passing/GSAA/ICC references (spec §12); ADR-085
      (TF-59 GSAA) confirmed as the analog. **ADR-099** — create from
      `git show ab9001c:docs/superpowers/adrs/ADR-089-tf54b-counterfactual-territorial-prevented.md`,
      renumber to the release-prep-assigned free number (placeholder `ADR-099` until then), and add the revival deltas: re-add-after-removal (Chesterton: dead
      `NotImplementedError`, not rejection; coexists with ADR-090 tracking sibling), broad open-data corpus,
      `PassCompletionModel` re-fit+re-bundle, the crossed-cell census + ICC gate, and the "ship ranking
      in-cycle iff gate clears (ADR-009 apply, not a library API)" decision.
- [ ] **Step 5:** No commit.

## Task 14: Whole-suite green + lint + types + C4 check (Commit 1 gate)

**Files:** none new.

- [ ] **Step 1:** `SILLY_KICKS_ASSERT_INVARIANTS=1 ./.venv/Scripts/python.exe -m pytest tests/ -m "not e2e"
      -q --benchmark-skip -p no:randomly` — all green. **The env var is load-bearing** (CI sets it, ci.yml:76/212;
      without it the coordinate-invariant asserts are off = false-green risk at this gate). **Baseline:** `9240`
      tests collected on the branch HEAD before Task 1 (measured 2026-09-17; `--co -q` reports it). Re-measure
      at execution start; Task 14 asserts baseline + the tasks' new tests all pass, 0 failed, 0 errors.
- [ ] **Step 2:** ruff/pyright run in the SYSTEM interpreter, not `.venv` (memory: build/ruff/pyright in
      system py3.14). `python -m ruff check silly_kicks/ tests/ scripts/` and
      `python -m ruff format --check silly_kicks/ tests/ scripts/` — clean (CI scope, never `.`).
- [ ] **Step 3:** `python -m pyright` (bare, config-driven) — clean.
- [ ] **Step 4:** Confirm C4 unchanged — `expected_passing` container already on `main`; no new
      container/aggregator/backend/model, so no `dot` re-render needed. Verify the C4 completeness gate
      still passes.
- [ ] **Step 5:** No commit — hand the fully-green tree to the owner for review + the Commit-1 approval gate.

## Commit 1 (owner-approved only)

Version bump `silly_kicks/_version.py` + `CHANGELOG.md` entry + the resolved **ADR number** are ALL locked
at **Commit 2 / release-prep** (owner ruling 2026-09-17 — may not release first, so version and ADR number
could collide; `git fetch && git merge origin/main` first, then take the free number and replace the
`ADR-099` placeholder + rename the ADR file). Commit 1 carries the `ADR-099` placeholder, no version bump,
no CHANGELOG entry. Commit 1 =
Tasks 1–13 code + tests + spec + ADR + glossary + NOTICE, docs included (provenance counts untracked as
dirty). `completed_failed` byte-identical; default unchanged. **No commit without an explicit owner yes.**

## Owner-run at Commit 1 (clean tree, `--out` OUTSIDE the repo)

- [ ] Re-fit `PassCompletionModel`: `python scripts/train_pass_completion.py --all-competitions --out <ext>`
      → `model.json` + `MODEL_CARD.md` + `metrics.json` (`training_commit` = Commit 1). Held-out AUC/ECE/
      Brier must clear the locked floors, else keep existing weights (record the fallback).
- [ ] `python scripts/validate_territory_counterfactual.py --out <ext>` → construct-validity artifact.
- [ ] `python scripts/build_territory_ranking_census.py --out <ext>` → `census.json` + (iff gate clears)
      `ranking.parquet`.

## Commit 2 (owner-approved only)

- [ ] Copy the re-bundled `expected_passing/weights/{model.json,SHA256SUMS,MODEL_CARD.md}` into the repo.
- [ ] Add `docs/research/territory_counterfactual_construct_validity/` + `docs/research/territory_ranking_census/`.
- [ ] All artifacts stamp Commit 1 (`run_commit`/`training_commit`); load-bearing **non-squash** merge so
      the stamps resolve. Then owner-driven: push → PR → CI green → admin-merge → tag → PyPI, explicit yes
      at every gate.

---

## Self-Review

**Spec coverage:** §5 cone mechanism → Tasks 1–6 (verbatim port) + Task 8 (reflection/mechanism/no-mutation
tests). §5.2 synthetic substrate → Task 7. §5b re-fit → Task 9. §7 broad-corpus validation → Task 12.
§8 census+gate+ranking → Tasks 10–11. §2 non-goals (byte-identity, no library ranking API, event-only) →
Task 5 byte-identity guard, Task 11 (ranking is a driver artifact), Task 8 import-allowlist. §9
glossary/NOTICE/ADR → Task 13. §10 2-commit provenance → Commit 1/2 sections. All spec sections mapped.

**Placeholder scan:** no TBD/TODO; every port names an exact `git show ab9001c:<path>`; the two genuinely-new
modules (`_crossed_icc.py`, `build_territory_ranking_census.py`) carry concrete algorithms + from-both-sides
gate tests. The one deliberately deferred detail — the exact Henderson-III mean-square coefficients — is
pinned by the known-variance recovery fixture (Task 10 Step 1), which is the correct place to lock a numeric
estimator.

**Type consistency:** `predict_completion(origin_x, origin_y, target_x, target_y)` matches the cone call in
Task 4; `destination_profiles(xt, ox, oy)` fields (`zone_centres/zone_values/probabilities`) match
`_counterfactual.py`; `TerritoryReport` cone fields (Task 3) match the report construction in
`_counterfactual_dispatch` (Task 5); `columns_for_method` (Task 1) is consumed in Task 5.

**Scope:** one feature branch, two commits (code, then artifacts) — the single justified second commit is the
provenance stamp, per spec §10. No unrelated refactoring.
