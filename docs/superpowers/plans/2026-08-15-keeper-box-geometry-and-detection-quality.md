# Keeper-box geometry & detection-quality cycle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the shipped SkillCorner keeper-origin resolver on real pining data and land its CI rate-gates; decide the `gr_x` behind-the-line predicate on a pre-registered materiality rule and execute the clamp if warranted; refresh the TF-24 Stage-2 recommendation on corrected geometry — all in one PR.

**Architecture:** Three independent pieces on one pining-sourced corpus (no strict ordering; verified decoupled). New/extended `scripts/` drivers adopt the `scripts/_driver.py` shard seam and `scripts/_provenance.py` clean-tree discipline. The only library change is a pure geometry predicate reading a new `spadlconfig` constant, landed **only if** the measurement warrants it. Design spec: `docs/superpowers/specs/2026-08-15-keeper-box-geometry-and-detection-quality-design.md`.

**Tech Stack:** Python, pandas/numpy, pytest, xgboost (ghost/xCross re-fit), `ruthless` (calibration + `for_each` fingerprint), the pining-for-the-data API, the DGX (aarch64) for corpus passes.

## Global Constraints

- **Data source:** every driver and probe pulls via `pining-for-the-data` (`scripts/_loader_pining.py`); the downloaded provider folders are NOT an input. Temporary within-run caching (a materialized `tc3` frame cache) is allowed.
- **Corpus drivers adopt `scripts/_driver.py`** (`for_each` / `CorpusPassResult`) and stamp provenance via `scripts/_provenance.py` (`require_clean_tree(git_provenance(), allow_dirty=...)` in `main()`, `run_commit` + `run_tree_dirty` in every artifact).
- **One PR, minimal commits.** The only commits are the DGX-provenance boundaries: commit-1 (drivers + gates + measurement), commit-2 (the `gr_x` clamp, only if warranted), commit-3 (release). Commit granularity is NOT a plan concern; **no task below contains a commit step.** Merge is `--merge` (non-squash) so stamped SHAs survive.
- **Verification is shown, not claimed.** Before any commit is requested, the CI-faithful gate is run and its real exit-coded output pasted: `python -m pytest tests/ -m "not e2e" --benchmark-skip`, `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`.
- **Two pre-commit gates for every data-touching driver:** a committed-fixture test AND a real-data pining probe (a match or two, on DGX for SkillCorner/full-corpus drivers).
- **New gates land red-first** — observed failing before the fix exists.
- **DGX-run tasks require a clean committed tree** (spec §8); that commit is taken through the owner-approval flow at the checkpoint, never scripted here.
- **No version/PR/ADR numbers are assigned until commit-prep**, after merging `origin/main`.
- **`penalty_area_depth = 16.5`, `penalty_area_half_width = 20.16`** (`silly_kicks/spadl/config.py:39-40`). The new lower-bound constant is `penalty_area_min_gr_x` (non-strict; `gr_x == 0` on the goal line counts as in-box).
- **τ = 3.62e-5** (materiality floor). **φ retired as a clamp gate** (only a possible D-data ingestion trigger).

---

# Phase A — SkillCorner keeper-origin validation + CI rate-gates

Prior art: S1–S4 shipped 4.37.0/ADR-024. This phase validates on real data, characterizes off-pitch/behind-line rows (which feed Phase B), and lands the standing rate-gates.

## Task A1: SkillCorner keeper-origin validation driver

**Files:**
- Create: `scripts/validate_skillcorner_keeper_origin.py`
- Create: `tests/scripts/test_validate_skillcorner_keeper_origin.py`
- Reference (read, do not modify): `scripts/_driver.py` (`for_each`, `CorpusPassResult`, `reconcile`), `scripts/_provenance.py` (`git_provenance`, `require_clean_tree`), `scripts/_loader_pining.py` (`load_matches`, `select_match_ids`), `silly_kicks/tracking/_gk_resolve.py` + the `resolve_gk_geometry` surface (emits `xt_gk_origin_source ∈ {tracking_gk, goalkick_prior, unresolved, native}` + resolved `xt_gk_origin_x/_y`).

**Interfaces:**
- Consumes: `load_matches(providers=["skillcorner"], match_ids=…, token=…) -> Iterator[(provider, match_id, actions, frames, home_team_id)]`.
- Produces: a per-match tidy frame (one row per GK-distribution action) with columns `provider, match_id, action_id, xt_gk_origin_source, origin_x, origin_y, in_own_box, gr_x, y, in_y_band, is_behind_line, is_gross_offpitch, is_visible`; a combined table `skillcorner_keeper_origin.parquet`; and a `manifest_all.json` carrying `run_commit`/`run_tree_dirty`.
- `gr_x` is distance from the **defended** goal line, resolved per row via `resolve_defended_goals(frames).get(game_id, period_id, team_id)` (ADR-055 `GoalMap`, NOT `home_team_id`); `is_behind_line = gr_x < 0`. `is_gross_offpitch` is the pitch-rectangle mask (M1 — there is NO reusable loader off-pitch boolean; this is a stated, cycle-owned predicate). `is_visible` is the genuine per-row detection bit (`is_detected`, `_loader_pining.py:464`), meaningful for SkillCorner (broadcast gaps) and typically all-true on dense providers.

- [ ] **Step 1: Write the failing driver-shape test.** In `tests/scripts/test_validate_skillcorner_keeper_origin.py`, assert the module exposes `measure_match(provider, match_id, actions, frames, home_team_id) -> pd.DataFrame` and `main()`, and that `measure_match` on the committed slim SkillCorner fixture (`tests/datasets/…` slim frames + actions — reuse the fixture the existing keeper-origin per-tier golden uses) returns a non-empty frame with exactly the columns listed in Interfaces/Produces.

```python
import pandas as pd
import scripts.validate_skillcorner_keeper_origin as drv

EXPECTED_COLS = {
    "provider", "match_id", "action_id", "xt_gk_origin_source", "origin_x", "origin_y",
    "in_own_box", "gr_x", "y", "in_y_band", "is_behind_line", "is_gross_offpitch", "is_visible",
}

def test_measure_match_shape(slim_skillcorner_match):
    provider, match_id, actions, frames, home = slim_skillcorner_match
    out = drv.measure_match(provider, match_id, actions, frames, home)
    assert set(out.columns) == EXPECTED_COLS
    assert len(out) > 0
    assert out["xt_gk_origin_source"].isin({"tracking_gk", "goalkick_prior", "unresolved", "native"}).all()
```

**M3 — the fixture must exercise the GK-distribution domain.** The `slim_skillcorner_match` fixture MUST contain goal-kick / keeper-origin actions, or A1's shape test and A2's rate-gates are vacuous. Confirm `tests/datasets/tracking/action_context_slim/skillcorner_slim.parquet` (or the per-tier keeper-origin golden fixture from 4.37.0) has ≥1 native goal-kick row (for the out-of-region rate) and non-empty `xt_gk_origin_source`. If no committed fixture has GK-distribution rows, create a slim one that does (a handful of goal-kick + open-play-GK-pass actions with frames) — this is a prerequisite for A2 running "all legs, not @e2e". **N-3: a newly-created SkillCorner slim fixture MUST be cut from a redistributable PUBLIC match** (visibility is keyed per-match via `match_visibility`, never on the provider name; only public-corpus matches may be committed — see `scripts/_corpus.py` / `PUBLIC_CORPUS`).

- [ ] **Step 2: Run it, expect ImportError/AttributeError (module absent).** Run: `python -m pytest tests/scripts/test_validate_skillcorner_keeper_origin.py -v` → FAIL.

- [ ] **Step 3a: Create the shared off-pitch mask** `scripts/_offpitch.py::off_pitch_mask(x, y, *, margin_m=OFF_PITCH_MARGIN_M) -> np.ndarray` returning `(x < -m) | (x > 105 + m) | (y < -m) | (y > 68 + m)`, with `OFF_PITCH_MARGIN_M` a stated cycle-owned constant (propose 2.0 m — the "few-metre tolerance for legitimately off-pitch keepers" ADR-024 S1 allows). Both this driver and `measure_box_constant_delta.py` import it (one implementation). Add a unit test (a point 3 m past the touchline is off-pitch; a keeper 1 m behind the goal line is not, at margin 2.0). **N-2: confirm 2.0 m against ADR-024 S1's actual gross-off-pitch fail-loud bound** (defined in the SkillCorner converter / resolver, NOT the loader — there is no loader constant); align `OFF_PITCH_MARGIN_M` to it, or state the deliberate difference in the constant's docstring, so the CI rate-gate and S1 are measuring the same "gross off-pitch."

- [ ] **Step 3b: Implement `measure_match`.** Resolve keeper-origin geometry on the match via the shipped `resolve_gk_geometry` path (do NOT re-implement resolution) and restrict to the GK-distribution domain. Build the goal map ONCE per match: `gmap = resolve_defended_goals(frames)`. For each action row, resolve its defended goal end `goal_x = gmap.get(game_id, period_id, team_id)` (skip/`unresolved` when `None`), then `gr_x = to_goal_relative_x(origin_x, goal_x=goal_x)` (apply per row, or vectorize the `(105 - x) if goal_x > 50 else x` rule). Emit `in_y_band = |y − 34| <= penalty_area_half_width`, `is_behind_line = gr_x < 0`, `in_own_box` from the resolved origin, `is_gross_offpitch = off_pitch_mask(origin_x, origin_y)`, and `is_visible` from the frame's detection bit at the action frame.

- [ ] **Step 4: Run the shape test, expect PASS.**

- [ ] **Step 5: Implement `main()` with the `for_each` + provenance skeleton.**

```python
def main() -> None:
    ap = argparse.ArgumentParser(description="Validate SkillCorner keeper-origin resolution on pining data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--match-ids-json", default=None)
    ap.add_argument("--max-per-provider", type=int, default=None)
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)  # in main(), before any work
    out = pathlib.Path(args.out)
    match_ids = {"skillcorner": json.load(open(args.match_ids_json))} if args.match_ids_json else None
    items = load_matches(providers=["skillcorner"], match_ids=match_ids, max_per_provider=args.max_per_provider)
    res = for_each(
        items,
        key=lambda m: ("skillcorner", str(m[1])),           # (provider, match_id) — providers share game_ids
        work=lambda m: measure_match(*m),
        shard_root=out / "shards",
        token_inputs={"schema": _SHARD_SCHEMA_VERSION, "driver": "skillcorner-keeper-origin-1"},
    )
    combined = reconcile(res.shard_dir, out / "skillcorner_keeper_origin.parquet", tag="all")
    manifest = {**res.manifest(), "run_commit": prov["commit"], "run_tree_dirty": prov["dirty"],
                "platform": prov["platform"], "machine": prov["machine"]}
    (out / "manifest_all.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
```

- [ ] **Step 6: Add the ASCII-only + argparse guard tests** the repo requires of every driver (mirror `tests/scripts/test_validate_*` for the ASCII gate and the `--help` parse test).

- [ ] **Step 7: Run the full test file, expect PASS.** Run: `python -m pytest tests/scripts/test_validate_skillcorner_keeper_origin.py -v`.

- [ ] **Step 8: Real-data pining probe (DGX or local via token).** Run on 1–2 real SkillCorner matches: `python scripts/validate_skillcorner_keeper_origin.py --out /tmp/skc_probe --max-per-provider 2 --allow-dirty`. Confirm a well-formed combined parquet with non-empty rows and a populated `xt_gk_origin_source`. Paste the row counts + source distribution. (This is a `--allow-dirty` dev smoke; the artifact records `dirty: true`.)

## Task A2: Structural rate-gates on the slim fixture (all legs)

**Files:**
- Create: `tests/scripts/test_skillcorner_rate_gates_structural.py`
- Reference: the combined-frame shape from A1.

**Interfaces:**
- Consumes: `measure_match(...)` output frame from A1.
- Produces: the two structural gate functions `offpitch_rate(frame) -> float` and `out_of_region_goalkick_rate(frame) -> float` (put these small pure helpers in the driver module so both the gate and the `@e2e` contract call one implementation).

- [ ] **Step 1: Write the structural gate test with a mandatory both-sides mutation.**

```python
import numpy as np
import scripts.validate_skillcorner_keeper_origin as drv

_LOOSE_OFFPITCH_CEILING = 0.25   # deliberately generous; the TIGHT baseline lives in the @e2e contract

def test_offpitch_rate_computed_finite_and_under_loose_ceiling(slim_skillcorner_match):
    frame = drv.measure_match(*slim_skillcorner_match)
    rate = drv.offpitch_rate(frame)
    assert np.isfinite(rate)
    assert 0.0 <= rate <= _LOOSE_OFFPITCH_CEILING

def test_offpitch_gate_fails_when_breached():  # the mandatory failing-side assertion
    frame = _fixture_with_all_rows_offpitch()   # every row is_gross_offpitch=True -> rate 1.0
    assert drv.offpitch_rate(frame) > _LOOSE_OFFPITCH_CEILING
```

- [ ] **Step 2: Run, expect FAIL** (helpers not defined). Run: `python -m pytest tests/scripts/test_skillcorner_rate_gates_structural.py -v`.

- [ ] **Step 3: Implement `offpitch_rate` and `out_of_region_goalkick_rate`** as pure functions over the frame (`frame["is_gross_offpitch"].mean()`; for the goal-kick rate: native goal-kick rows with `in_own_box == False` divided by native goal-kick rows). Add the mirror both-sides test for the goal-kick rate.

- [ ] **Step 4: Run, expect PASS.**

## Task A3: Tight corpus-baseline `@e2e` data-contract

**Files:**
- Create: `tests/scripts/test_skillcorner_rate_gates_e2e.py` (marked `@pytest.mark.e2e`)
- Modify (post-Phase-A-run): the pinned thresholds, from the measured baseline.

**Interfaces:**
- Consumes: the full-corpus `skillcorner_keeper_origin.parquet` from A1's DGX run + its `manifest_all.json`.

**L1 — this file is AUTHORED in commit-3, with the baselines already pinned**, not scaffolded in Phase A. A committed `@e2e` test with `BASELINE_OFFPITCH = None` would `TypeError` on `rate <= None + MARGIN` the moment `-m e2e` collects it. Write it once, at commit-3, from the measured artifact — and belt-and-suspenders, guard it so an unpinned copy self-skips rather than errors:

```python
import pytest

BASELINE_OFFPITCH = 0.0XX   # measured from the Phase-A DGX artifact (pinned at authoring)
BASELINE_OOR = 0.0XX
MARGIN = 0.02
pytestmark = [pytest.mark.e2e,
              pytest.mark.skipif(BASELINE_OFFPITCH is None, reason="rate baselines not yet pinned")]
```

- [ ] **Step 1: Write the `@e2e` contract** (at commit-3) that loads the full-corpus artifact + `manifest_all.json`, refuses a missing/dirty manifest (unprovenanced == fail), and asserts `offpitch_rate ≤ BASELINE_OFFPITCH + MARGIN`, `out_of_region_goalkick_rate ≤ BASELINE_OOR + MARGIN`, goal-kick origins ≈100% own-box, and the scatter-SD collapse (ADR-024 acceptance), with the baselines pinned from the artifact.
- [ ] **Step 2: Confirm the contract passes** on the committed artifact. (Lands with the artifact in commit-3.)

## Task A4: Own-half bound (validate-then-maybe) + ±window (measure-before-optimize)

**Files:**
- Modify (conditional): the SkillCorner keeper-origin resolver (`silly_kicks/tracking/_gk_resolve.py` / `_gk_geometry.py`) — ONLY if the Phase-A measurement warrants it.
- Create: `docs/research/skillcorner_keeper_origin/README.md` + `metrics.json`.

- [ ] **Step 1: From A1's DGX artifact, measure** whether open-play pass origins still land in the attacking half. Record in the research artifact.
- [ ] **Step 2: (conditional) add the generous own-half bound** ONLY if the measurement shows pass origins staying attacking-half; beyond it → `unresolved`, never clamped. If the measurement does not support it, record "not added" with the number.
- [ ] **Step 3: measure-before-optimize the `_tracking_gk_xy_detected` ±window loop** — record its current cost/behavior in the artifact before any change; optimize only if the measurement warrants, with a before/after.

---

# Phase B — the `gr_x` decision (measurement + conditional clamp)

## Task B1: Extend `measure_box_constant_delta.py` with the training-feature-delta measurement

**Files:**
- Modify: `scripts/measure_box_constant_delta.py`
- Modify: `tests/tracking/test_measure_box_constant_delta.py` (or the driver's existing test)
- Reference: `silly_kicks/tracking/_ghost_gk.py::prepare_ghost_gk_training_data` (`attackers_in_box`), `silly_kicks/tracking/_xcross_attempt.py::prepare_xcross_training_data` (`box_off_def_ratio`), `silly_kicks/tracking/_geometry.py::in_penalty_area_goal_relative_array`.

**Interfaces:**
- Produces: new artifact fields per model — `{ghost,xcross}_changed_fraction`, `{ghost,xcross}_real_near_line_fraction`, `{ghost,xcross}_offpitch_fraction` (at margins 1/2/5 m), `{ghost,xcross}_train_behind_line_base_rate`, and a `{ghost,xcross}_dist_to_goal_hist` — all under `run_commit`/`run_tree_dirty`.

**Prerequisites (L3):**
- `--data-dir` is the **already-materialized 4.81.0 `tc3` 179-match frame cache on the DGX** (pining-sourced; no task re-materializes it — it exists from the prior cycle). If absent, materialize it via `scripts/_loader_pining_to_cache.py` first.
- The `behind_line_ghost_frames` fixture (B1 Step 1) MUST contain ≥1 attacker with `gr_x < 0` inside the y-band, or the `(clamped < base).any()` non-vacuity assertion is itself vacuous — build it explicitly with a behind-goal-line attacker.

- [ ] **Step 1: Write the scoped-clamp context manager + its test (N1).**

```python
# in scripts/measure_box_constant_delta.py
import contextlib
import silly_kicks.tracking._geometry as _geo

@contextlib.contextmanager
def _scoped_gr_x_clamp():
    """Measurement-only: patch the MODULE ATTRIBUTE so both consumers (attribute-access) see it."""
    original = _geo.in_penalty_area_goal_relative_array
    def clamped(gr_x, y):
        import numpy as np
        return original(gr_x, y) & (np.asarray(gr_x) >= 0.0)  # N2: 0.0 == future penalty_area_min_gr_x
    _geo.in_penalty_area_goal_relative_array = clamped
    try:
        yield
    finally:
        _geo.in_penalty_area_goal_relative_array = original
```

```python
# test: the clamp actually changes attackers_in_box on a behind-line fixture, and reverts
def test_scoped_clamp_changes_box_feature_and_reverts(behind_line_ghost_frames, home_id):
    base = prepare_ghost_gk_training_data(behind_line_ghost_frames, home_team_id=home_id)[0]["attackers_in_box"]
    with _scoped_gr_x_clamp():
        clamped = prepare_ghost_gk_training_data(behind_line_ghost_frames, home_team_id=home_id)[0]["attackers_in_box"]
    assert (clamped <= base).all() and (clamped < base).any()   # clamp can only remove behind-line attackers
    after = prepare_ghost_gk_training_data(behind_line_ghost_frames, home_team_id=home_id)[0]["attackers_in_box"]
    assert (after == base).all()   # attribute restored
```

- [ ] **Step 2: Run, expect FAIL** (context manager not defined). Run: `python -m pytest tests/tracking/test_measure_box_constant_delta.py -k scoped_clamp -v`.

- [ ] **Step 3: Implement `measure_training_flip(frames, actions, home_team_id, *, model) -> dict`.** Two measurements, computed differently ON PURPOSE:
  - **`changed_fraction`** (the τ input, C4-clean) comes from the SEAM: call the model's `prepare_*_training_data` twice (baseline, then inside `_scoped_gr_x_clamp()`) and diff the box column (`attackers_in_box` / `box_off_def_ratio`) — the fraction of training rows whose value differs.
  - **`dist_to_goal_hist`, `real_near_line_fraction`, `offpitch_fraction`** are a DESCRIPTIVE characterization of the behind-line points, computed from the frames' player positions — NOT the seam, which returns an aggregated count with no per-attacker position. This is correct: they are evidence about the positions, not the training-row selection (which stays seam-sourced). Per training frame, take the attacker (ghost) / attacker+defender (xCross) positions in the attacked box, resolve the attacked goal via `resolve_defended_goals(frames).attacked_goal(...)`, compute their `gr_x`, and over the `gr_x < 0 ∧ in y-band` subset report the histogram of `gr_x`, `real_near_line_fraction` = share with `-margin <= gr_x < 0`, and `offpitch_fraction` = share with `gr_x < -margin`, at margins {1, 2, 5} m. This is NOT Phase A's row set — Phase A characterizes keeper origins behind the DEFENDED goal; here it is attackers/defenders behind the ATTACKED goal (different population, different goal end).
  - **`train_behind_line_base_rate`** = share of training rows carrying any behind-line-in-band point (the C3 base rate that says whether τ can bind).
  Reuse the driver's existing `for_each`/provenance wiring; add all fields to the emitted metrics.

- [ ] **Step 3c: Population-coherence assertion (N-1) — tie the two decision legs to the SAME points.** The B2 gate ANDs `changed_fraction` (seam) with `real_near_line_fraction` (frame reconstruction), so the two MUST describe the same population or the gate is incoherent. (1) The reconstruction uses `in_penalty_area_goal_relative_array` and mirrors the seam's attacker/defender selection — not a re-derived box test. (2) Assert per model the **row-set identity**: `{training rows where the seam box feature changed under the scoped clamp} == {training rows where the reconstruction found ≥1 behind-line-in-band point}`. A row-set identity (not a count identity) is used deliberately because it holds for BOTH the ghost count *and* the xCross ratio — a count-delta does not decompose for a ratio. For ghost, additionally assert the stronger count identity: `sum(attackers_in_box_baseline − attackers_in_box_clamped) == n_reconstructed_behind_line_in_band_attackers`. If either disagrees, the reconstruction has drifted from what the seam counts — fix it, do not proceed; a divergent pair makes "material" and "real" measure different populations.

- [ ] **Step 4: Anti-drift seam test (C4).** R_M is identical by construction *because* the measurement calls the library seam; guard that a future "optimization" cannot silently re-implement extraction by spying that the seam is actually called.

```python
def test_measurement_routes_through_library_seam(monkeypatch, slim_ghost_match, home_id):
    import silly_kicks.tracking as T
    frames, actions = slim_ghost_match
    calls = []
    real = T.prepare_ghost_gk_training_data
    monkeypatch.setattr(T, "prepare_ghost_gk_training_data",
                        lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    drv.measure_training_flip(frames, actions, home_id, model="ghost")
    assert calls, "measurement must route through prepare_ghost_gk_training_data, not re-implement extraction"
```

- [ ] **Step 5: Run the driver's full test file, expect PASS.**

- [ ] **Step 6: Real-data pining probe (DGX or local).** Run on 1–2 matches: `python scripts/measure_box_constant_delta.py --data-dir <tc3-cache> --out /tmp/box_probe --allow-dirty`. Confirm the new fields are populated and finite. Paste them.

- [ ] **Step 7: (DGX, clean tree) full-corpus run** producing the decision artifact `docs/research/box_constant_delta/metrics.json` (extended). Requires commit-1 on a clean tree first (owner-approval checkpoint).

## Task B2: (conditional — only if D-geom warranted) the declared-constant clamp

Gate on the pre-registered rule applied to B1's artifact: **material** (`changed_fraction ≥ 3.62e-5` for ghost OR xCross) **AND real-positions-exist** (`real_near_line_fraction` non-negligible — a genuine near-goal population the geometry is wrong for). If not warranted → skip B2/B3 entirely (record the decision in the artifact + the gr_x ADR). Independently (D-data): if `offpitch_fraction` is material, file the ingestion-seam TODO whether or not B2 runs.

**Files:**
- Modify: `silly_kicks/spadl/config.py` (add `penalty_area_min_gr_x`)
- Modify: `silly_kicks/tracking/_geometry.py` (`in_penalty_area_goal_relative_array` body + `GEOMETRY_VERSION` if numeric output changes)
- Modify: `scripts/train_ghost_gk.py::cache_token()` (include the new constant)
- Modify: `silly_kicks/tracking/_ghost_gk.py` + `silly_kicks/tracking/_xcross_attempt.py` feature-contract blocks (declare `penalty_area_min_gr_x`)
- Modify/Create tests: `tests/tracking/test_geometry_box_predicate_parity.py`, `tests/tracking/test_geometry_constant_enumeration.py`, `tests/tracking/test_cache_token_moves_on_clamp.py` (new)

- [ ] **Step 1: Red-first cache_token test — the DERIVATION property, never a substring (H1).** A substring assertion (`"min" not in cache_token()`) is vacuous: the natural wiring `…-{penalty_area_min_gr_x:.4f}` renders `"…-0.0000"`, which contains no `"min"`, so the assertion stays green after wiring and guards nothing — the exact "a guard that certifies the failure it catches" trap, on the one mechanism the spec calls load-bearing. Test that the token MOVES with the constant instead:

```python
def test_cache_token_derives_from_min_gr_x(monkeypatch):
    from scripts.train_ghost_gk import cache_token
    import silly_kicks.spadl.config as spc
    t0 = cache_token()
    monkeypatch.setattr(spc, "penalty_area_min_gr_x", 5.0)  # perturb the constant
    assert cache_token() != t0                              # the token MUST move with it
```

  This is red-first in BOTH pre-states and green only when fully wired: before the constant exists, `monkeypatch.setattr` raises `AttributeError` (the attribute is absent); after the constant exists but before `cache_token()` reads it, the token does not move and the assertion fails. It is correct in one direction — **no Step-5 invert dance** — and it proves the actual guarantee: any future value change to the bound invalidates the feature cache. Also assert ghost's and xCross's `_feature_contract_block()` declared-constant dict gains `penalty_area_min_gr_x`.

- [ ] **Step 2: Red-first predicate parity.** Add cases to `test_geometry_box_predicate_parity.py` asserting a point at `gr_x = -0.01, y = 34` is IN-box under the current predicate and OUT under the clamped one; run against the un-clamped body → the "OUT" case FAILS.

- [ ] **Step 3: Add `penalty_area_min_gr_x = 0.0` to `spadlconfig`** (with the non-strict / `gr_x==0`-in-box docstring) and wire it into `in_penalty_area_goal_relative_array`: `return (gr_x <= _spc.penalty_area_depth) & (gr_x >= _spc.penalty_area_min_gr_x) & (np.abs(y - GOAL_Y) <= _spc.penalty_area_half_width)`. Bump `GEOMETRY_VERSION` only if a numeric feature output changes (it does — do bump it).

- [ ] **Step 4: Wire `cache_token()`** to include `penalty_area_min_gr_x` in its format string, and add `penalty_area_min_gr_x` to ghost's and xCross's `_feature_contract_block()` declared-constants (following the existing `penalty_area_depth` entry).

- [ ] **Step 5: Invert the predicate-parity red-first case** (the `gr_x=-0.01` IN→OUT case) to its post-wiring "OUT" expectation. The cache_token test from Step 1 is NOT inverted — it is already correct in one direction. Run: `python -m pytest tests/tracking/test_geometry_box_predicate_parity.py tests/tracking/test_geometry_constant_enumeration.py tests/tracking/test_cache_token_moves_on_clamp.py -v` → PASS.

- [ ] **Step 6: Run the geometry-constant enumeration + full geometry suite** to confirm the new constant is declared-or-exempt and nothing else regressed.

- [ ] **Step 7: N2 consistency guard (L2).** Add a test that the measurement's scoped clamp (`scripts.measure_box_constant_delta._scoped_gr_x_clamp`) produces the SAME box feature as the real predicate now reading `penalty_area_min_gr_x` — so a future non-zero constant cannot silently diverge what the measurement predicts from what ships. Run on the behind-line fixture; assert `attackers_in_box` is identical under both.

## Task B3: (conditional) re-fit ghost + xCross, re-stamp, republish

- [ ] **Step 1: (DGX, clean commit-2 tree) re-fit ghost** — `python scripts/train_ghost_gk.py --providers skillcorner idsse gradientsports --output-dir models/ …` (pining-sourced). The `cache_token` change forces a re-extract; confirm it does not reuse stale features (the token differs).
- [ ] **Step 2: (DGX) re-fit xCross** — `python scripts/train_xcross_attempt.py --providers skillcorner idsse gradientsports …`. Preserve the paired data-effect + fail-closed acceptance gates.
- [ ] **Step 3: Re-stamp both feature contracts on x86** via `scripts/stamp_feature_contracts.py`.
- [ ] **Step 4: Confirm `load()` on both artifacts passes** (chirality + feature contract, incl. the new declared constant) and republish to the Hub.

---

# Phase C — TF-24 Stage-2 refresh

## Task C1: carrier-refusal red-first test + Stage-2 sweep

**Files:**
- Create/confirm: a red-first test that `_load_carrier_selection` refuses a dirty/unprovenanced selection (`scripts/calibrate_tracking_defaults.py:253` already enforces this — add the test if one does not exist).

- [ ] **Step 1: Write the refusal test.**

```python
import pytest, json
from scripts.calibrate_tracking_defaults import _load_carrier_selection

def test_stage2_refuses_dirty_carrier(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"beta": 0.0, "gamma": 0.25, "run_commit": "abc", "run_tree_dirty": True}))
    with pytest.raises(ValueError, match="dirty"):
        _load_carrier_selection(str(p))

def test_stage2_refuses_unprovenanced_carrier(tmp_path):
    p = tmp_path / "sel.json"
    p.write_text(json.dumps({"beta": 0.0, "gamma": 0.25}))   # no run_commit
    with pytest.raises(ValueError, match="provenance"):
        _load_carrier_selection(str(p))
```

- [ ] **Step 2: Run, expect PASS** (the guard is already coded) — this locks the contract. Run: `python -m pytest tests/scripts/ -k carrier -v`.

- [ ] **Step 3: (DGX, clean commit-1 tree) run Stage-2** — `python scripts/calibrate_tracking_defaults.py --stage 2 --source pining --providers skillcorner idsse gradientsports --n-trials 60 --store tc3_stage2.db --xt-artifact <frozen> --carrier-best <4.81.0 carrier_selected.json>`. Produces the recommendation manifest under `docs/research/…` with provenance.

- [ ] **Step 4: Record the recommendation** (k3 / pre_seconds / min_displacement_m + Brier delta vs incumbent) in the artifact. **Do NOT change any library default** (ADR-009). Surface whether any recommendation would be worth a future adopt-PR.

---

# Phase D — release

## Task D1: ADRs, CHANGELOG, TODO, version, ADR-code sweep

**Files:**
- Create: `docs/superpowers/adrs/ADR-0NN-gr-x-behind-line-decision.md` (new ADR — the basis-A rule, the measured outcome, the declared-constant/cache-token mechanism)
- Modify: `docs/superpowers/adrs/ADR-024-*.md` (amendment — the two standing rate-gates)
- Modify: `CHANGELOG.md`, `TODO.md` (retire the two picked-up rows; add the D-data ingestion TODO if warranted), `pyproject.toml` + `silly_kicks/__init__.py` + `uv.lock` (version bump), `CLAUDE.md` (only if a durable contract changed — e.g. the new constant / gr_x decision)

- [ ] **Step 1: Merge `origin/main`** into the branch and take the next-free version/PR/ADR numbers (reconcile the shared release-mechanics files with the parallel SB360 cycle — only the Release header line collides).
- [ ] **Step 2: Write the new gr_x ADR + the ADR-024 amendment** reflecting the measured outcome (clamp shipped / deferred / immaterial).
- [ ] **Step 3: ADR-code reconciliation sweep** — verify documented ADRs still match the codebase; fix drift.
- [ ] **Step 4: Update CHANGELOG + TODO + version files + CLAUDE.md** (durable-contract change only).
- [ ] **Step 5: Run the full CI-faithful gate** (Global Constraints) and paste the output. Confirm the C4 count is unchanged (no new action-coupled aggregator).

---

## Self-review — spec coverage

- Phase A validation + rate-gates (structural + `@e2e`) → A1/A2/A3; own-half + ±window → A4. ✅ (spec §4)
- gr_x measurement (R_M via seams, scoped clamp, base rate, distance-to-goal histogram, N1/N2) → B1. ✅ (spec §5.2, §5.3 D-geom lens)
- gr_x pre-registered rule + declared-constant clamp + cache_token/contract red-first → B2 (conditional). ✅ (spec §5.3/§5.4/§5.5, C5)
- Two-model re-fit + re-stamp + republish → B3 (conditional). ✅ (spec §5.5)
- TF-24 Stage-2 + carrier-refusal contract → C1. ✅ (spec §6, C6)
- ADRs (new gr_x + ADR-024 amendment), release, ADR-code sweep → D1. ✅ (spec §7, §11)
- Commit & verification protocol, pining sourcing, `_driver.py`/provenance adoption → Global Constraints. ✅ (spec §8)
