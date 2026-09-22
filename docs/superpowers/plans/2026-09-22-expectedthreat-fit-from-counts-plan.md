# SK-XT-COUNTS — `ExpectedThreat.fit_from_counts` — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-09-22-expectedthreat-fit-from-counts-design.md` (r2 APPROVE).
**Proposed version:** `<next-free version>` / PR-Snnn / **ADR-NNN** (re-derive at commit-prep). **Base:** branch `feat/expectedthreat-fit-from-counts` off `origin/main` (silly-kicks **4.121.0**) — `git fetch && git checkout main && git pull` FIRST.

**Goal:** Add a distributed-friendly counts-based fit (`fit_from_counts`) + a public zone-binning contract to `silly_kicks.xthreat.ExpectedThreat`, single-sourcing the count→matrix cores so `fit(actions)` stays byte-identical. Purely additive; no VAEP/tracking retrain, no re-materialize, C4-free.

**Architecture:** `_grid._scoring_prob` / `_action_prob` and `_transitions.singh_transition_matrix` are refactored into a pure `*_from_counts` CORE + a thin `from-actions` wrapper (count → call core). `fit(actions)` calls the wrapper (byte-identical); `fit_from_counts` calls the core. `zones_of`/`flat_indexes_of` are instance methods over `_get_cell_indexes`/`_get_flat_indexes`; membership constants are class attributes sourced from `spadlconfig`.

## Global constraints
- **TDD red-green:** every Task-1 test written FAILING first, observed red, then Task-2 turns it green.
- **No commit/push/tag without explicit owner approval** for that specific commit (STOP at Task 4). Plan approval is NOT commit authority.
- **Additive only.** `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles`/`to_dict`/`from_dict` behaviour + the SK-xT-1 frozen-oracle parity (`tests/test_xthreat.py::test_singh_path_byte_identical_to_legacy` + `test_singh_path_byte_identical_on_worldcup`) stay byte-identical — asserted green.
- **CI scope, run BEFORE push (the SK-XT-SER CI-miss lesson):** `python -m ruff check silly_kicks/ tests/` + `ruff format --check` + **BARE `pyright` (whole config scope, INCL `tests/`)** + `python -m pytest tests/ -m "not e2e"`. Bare pyright over the new test file is mandatory — SK-XT-SER shipped a `method: str → Literal` test-file error that `pyright silly_kicks/` missed.

## File structure
- `silly_kicks/xthreat/_grid.py` — add `_scoring_prob_from_counts(goal_counts, shot_counts)` + `_action_prob_from_counts(shot_counts, move_counts)` cores; `_scoring_prob`/`_action_prob` call them (byte-identical).
- `silly_kicks/xthreat/_transitions.py` — add `_singh_from_counts(transition_counts, start_counts)` core; `singh_transition_matrix` calls it (byte-identical).
- `silly_kicks/xthreat/_model.py` — `fit_from_counts` + `zones_of` + `flat_indexes_of` methods on `ExpectedThreat`; `MOVE_TYPE_NAMES`/`SHOT_TYPE_NAME` class attributes. **Imports to ADD (PLAN-02):** `_get_flat_indexes` (for `flat_indexes_of` — `_model.py` currently imports `_get_cell_indexes` but NOT `_get_flat_indexes`) + the three new cores `_scoring_prob_from_counts`/`_action_prob_from_counts` (from `_grid`) + `_singh_from_counts` (from `_transitions`).
- `tests/xthreat/test_fit_from_counts.py` — NEW (the sub-dir convention, next to `test_serialize.py`/`test_counterfactual_seam.py`).
- `CHANGELOG.md`, `docs/superpowers/adrs/ADR-NNN-*.md`, `silly_kicks/_version.py` (→ next-free), `uv.lock` (`uv lock`).
- **No** `xthreat/__init__.py __all__` change for the methods (reachable via the exported `ExpectedThreat` class — the SK-XT-SER SHOULD-FIX). **No** `NOTICE`/C4/glossary change.

---

### Task 1: Red — the failing tests

**File:** `tests/xthreat/test_fit_from_counts.py`. Build a small SPADL `actions` fixture `A` in-code (not a committed parquet) that DELIBERATELY contains the §6.1 boundary cases. A helper `_aggregate(A, l, w)` computes the 5 count arrays from `A` using the SAME filters sk uses (shot=`type_id==shot`, goal=+`result==success`; move=`type_id ∈ {pass,dribble,cross}`; `move_counts` = valid-start `_count`; `transition_start_counts` = valid-start+end `_count`; `transition_counts` = successful valid-start+end flat(from)→flat(to)).

- [ ] **Step 1: Write all 7 tests (spec §6):**
  1. `test_functional_equivalence_matrices_byte_identical` — `xt_c = ExpectedThreat(l,w).fit_from_counts(**_aggregate(A))`; `xt_a = ExpectedThreat(l,w).fit(A)`; `np.array_equal` on all 4 matrices; `np.allclose(xt_c.xT, xt_a.xT)`.
     - Fixture `A` MUST include: **(a)** a `pass`/`cross` with a valid start + `end_x/end_y = NaN` (the D3 trigger — asserts `array_equal` on `move_prob_matrix` AND `transition_matrix`); **(b)** a zone with shots but 0 goals + a zone with 0 actions (`_safe_divide` 0-branch); **(c)** a move with `end_x > 105` (clamp to `l-1`).
  2. `test_additivity` — `_aggregate(A) + _aggregate(B)` (element-wise per array) fit == `ExpectedThreat(l,w).fit(pd.concat([A,B]))` (matrices `array_equal`, xT `allclose`).
  3. `test_round_trip` — `ExpectedThreat.from_dict(xt_c.to_dict())` reconstructs (reuse the ADR-100 harness).
  4. `test_zone_binning_parity` — `xt.zones_of(xs,ys)` == `_get_cell_indexes(xs,ys,l,w)` and `xt.flat_indexes_of(xs,ys)` == `_get_flat_indexes(...)` on a probe grid incl. edges `x∈{0,105,120}`, `y∈{0,68,-5}` and the `l-1`/`w-1` clamp.
  5. `test_kde_params_raises` — `fit_from_counts(**counts, params=KDEParams())` → `pytest.raises(ValueError, match="KDE")`.
  6. `test_shape_guard_raises` — a `(w, l+1)` `shot_counts` → `ValueError` (non-vacuity).
  7. `test_fit_parity_oracle_unaffected` — assert `tests/test_xthreat.py::test_singh_path_byte_identical_to_legacy` still passes after the core extraction (or run it in-process on the fixture); the refactor is behaviour-preserving.
- [ ] **Step 2: Run → RED.** `python -m pytest tests/xthreat/test_fit_from_counts.py -q` → all fail (`AttributeError: fit_from_counts`). Capture the red.

---

### Task 2: Green — refactor cores + implement

- [ ] **Step 1: Extract the count cores (behaviour-preserving).**
  - `_grid.py`: `def _scoring_prob_from_counts(goal_counts, shot_counts): return _safe_divide(goal_counts, shot_counts)`. `_scoring_prob(actions,l,w)` → count then `return _scoring_prob_from_counts(goalmatrix, shotmatrix)`.
  - `_grid.py`: `def _action_prob_from_counts(shot_counts, move_counts): total = move_counts + shot_counts; return _safe_divide(shot_counts, total), _safe_divide(move_counts, total)`. `_action_prob` → count then call it.
  - `_transitions.py`: `def _singh_from_counts(transition_counts, start_counts): n = start_counts.shape[0]; T = np.zeros((n,n)); nz = start_counts > 0; T[nz] = transition_counts[nz] / start_counts[nz, None]; return T`. `singh_transition_matrix` → build `start_counts` (all valid-start+end moves) + `counts` (successful) via `np.add.at`, then `return _singh_from_counts(counts, start_counts)`.
  - **Verify byte-identity immediately:** run `tests/test_xthreat.py -q` → green (the extraction moved no float op).

- [ ] **Step 2: `fit_from_counts`** on `ExpectedThreat` (mirrors `fit` at `_model.py:112`):
  - Validate: `if self.method == "kde_smoothed" or isinstance(params, KDEParams): raise ValueError("KDE transition is not a pure count aggregate; use fit(actions) for KDE, or fit_from_counts with singh/default params.")`. Shape-check each count against `(self.w, self.l)` / `(self.w*self.l,)²` → `ValueError`.
  - **`np.asarray(..., dtype=float64)` ALL FIVE counts uniformly first** (PLAN-01 — the shape-guard already rejects list inputs, but casting all five removes the read-as-intentional asymmetry): `sc, gc, mc, tsc, tc = (np.asarray(x, dtype=np.float64) for x in (shot_counts, goal_counts, move_counts, transition_start_counts, transition_counts))`. Then `self.scoring_prob_matrix = _scoring_prob_from_counts(gc, sc)`; `self.shot_prob_matrix, self.move_prob_matrix = _action_prob_from_counts(sc, mc)`; `self.transition_matrix = _singh_from_counts(tc, tsc.ravel())`; `self.xT, self.heatmaps = value_iteration(self.scoring_prob_matrix, self.shot_prob_matrix, self.move_prob_matrix, self.transition_matrix, eps=self.eps)`; `return self`. (Same `value_iteration` call as `fit`.)

- [ ] **Step 3: `zones_of` / `flat_indexes_of` + membership constants.**
  - `def zones_of(self, xs, ys) -> tuple[NDArray[int], NDArray[int]]: xi, yj = _get_cell_indexes(pd.Series(xs), pd.Series(ys), self.l, self.w); return xi.to_numpy(), yj.to_numpy()`.
  - `def flat_indexes_of(self, xs, ys) -> NDArray[int]: return _get_flat_indexes(pd.Series(xs), pd.Series(ys), self.l, self.w).to_numpy()`.
  - Class attributes: `MOVE_TYPE_NAMES = ("pass", "dribble", "cross")`, `SHOT_TYPE_NAME = "shot"` (documented as the `spadlconfig` sets the internal filters use; a test asserts they match `spadlconfig.actiontype_id` keys). Reachable via the exported class → **no `__all__` change**.

- [ ] **Step 4: Examples blocks (the SK-XT-SER meta-gate lesson).** `fit_from_counts`, `zones_of`, `flat_indexes_of` are new PUBLIC methods → each needs an Examples section or `test_public_api_examples` goes red. Add an indented-RST literal-block Example to each (not an executable `>>>` — counts need a real aggregate). Cross-ref ADR-NNN + ADR-041 (orientation) in `fit_from_counts`.

- [ ] **Step 5: Run → GREEN.** `python -m pytest tests/xthreat/test_fit_from_counts.py tests/test_xthreat.py tests/xthreat/ -q` → all pass (incl. the parity oracle).

---

### Task 3: Full gates + docs

- [ ] **Step 1: Lint/type/full suite (CI-faithful).** `python -m ruff check silly_kicks/ tests/` + `ruff format --check` + **BARE `pyright`** (whole scope incl `tests/` — run it, do NOT scope to `silly_kicks/`; if a numpy-stub-version discrepancy floods noise, isolate the new files' errors) + `python -m pytest tests/ -m "not e2e"`. Also run `python -m pytest tests/test_public_api_examples.py -q` (the 3 new methods' Examples). Confirm SK-xT-1 parity green.
- [ ] **Step 2: Docs.** `CHANGELOG.md` `Added` (fit_from_counts + zones_of/flat_indexes_of + the 5-count contract + singh-only). **ADR-NNN** from `ADR-TEMPLATE.md` recording spec D1–D6 (esp. D1 single-source cores, D3 three move aggregates + the start-vs-start+end filter distinction). `_version.py` → next-free; `uv lock`. **No** `NOTICE`/C4/`__all__`/glossary change (verify each untouched).
- [ ] **Step 3: /final-review** (C4 count 33 unchanged; version single-source gate; TODO grooming).

---

### Task 4: Commit gate

- [ ] **Step 1:** Re-run the full gate set green; capture the pytest exit code (not via `| tail`).
- [ ] **Step 2:** `git status` shows ONLY: `_grid.py`, `_transitions.py`, `_model.py`, `tests/xthreat/test_fit_from_counts.py`, `CHANGELOG.md`, `ADR-NNN-*.md`, `_version.py`, `uv.lock`, the spec+plan docs. Nothing else (`uv.lock` unchanged if the dynamic version makes `uv lock` a no-op — then it is not staged).
- [ ] **Step 3: STOP — request explicit owner approval to commit.** Show the diff / file list. On approval: single commit on `feat/expectedthreat-fit-from-counts`, subject `feat(xthreat): ExpectedThreat.fit_from_counts + zone-binning contract (SK-XT-COUNTS, ADR-NNN) — silly-kicks <next-free>`, `Co-Authored-By` trailer. Then owner-gated push → PR → CI green → admin merge (non-squash) → tag → PyPI (each its own approval).

---

## Self-review
1. **Spec coverage:** §2 API → Task 2 Steps 2-3; §3 internals → Task 2 Step 1 cores; §4 5-count contract → Task 2 Step 2 + Task 1 `_aggregate`; §5 constraints (KDE-raise, orientation, raw-counts, additivity) → Task 2 Step 2 + tests 2/5; §6 TDD → Task 1; §7 D1–D6 → ADR-NNN; §8 ship → Task 3-4. ✓
2. **`fit` byte-identity:** the core extraction moves no float op; parity oracle asserted green (Task 2 Step 1 + test 7). ✓
3. **Meta-gates (SK-XT-SER lessons):** Examples on the 3 new public methods (Task 2 Step 4); bare pyright over `tests/` (Task 3 Step 1); no `__all__` change (methods reachable via class). ✓
4. **Correctness crux:** `transition_start_counts` (valid start+end) is the Singh denominator, NOT `move_counts` (valid-start) — pinned by test 1's boundary fixture (a). ✓
