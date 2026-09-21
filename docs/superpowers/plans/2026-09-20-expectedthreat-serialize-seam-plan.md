# SK-XT-SER — `ExpectedThreat` serialize seam — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-09-20-expectedthreat-serialize-seam-design.md` (APPROVE'd by lakehouse; SHOULD-FIX + 3 CONSIDER applied).
**Proposed version:** <next-free version> / PR-Snnn / **ADR-NNN** (re-derive at commit-prep). **Base:** branch `feat/expectedthreat-serialize-seam` off `origin/main` (silly-kicks **4.120.0**) — the working-tree 4.119.1 checkout is stale; `git fetch && git checkout main && git pull` FIRST.

**Goal:** Add `to_dict`/`from_dict`/`save`/`load` to `silly_kicks.xthreat.ExpectedThreat` — a JSON, pickle-free, fail-closed, raw-orientation-verbatim serialize seam for cross-process transport of a fitted model. Purely additive; no existing path changes; no VAEP/tracking retrain, no re-materialize, C4-free.

**Architecture:** `ExpectedThreat` (`xthreat/_model.py`) gains four methods. `to_dict`/`from_dict` are the single serialization source; `save`/`load` are thin JSON-file wrappers over them (spec D1). No `__all__` change — they are methods on the already-exported class (spec §2 SHOULD-FIX).

## Global constraints
- **TDD red-green**: every test in Task 1 is written FAILING first, observed red, then Task 2 turns it green. No implementation before a red test.
- **No commit/push/tag without explicit owner approval** for that specific commit (STOP at Task 4 Step 3). Plan approval is NOT commit authority.
- **Additive only.** `fit`/`rate`/`interpolator`/`values_at_points`/`destination_profiles` behaviour and the SK-xT-1 frozen-oracle parity (`tests/test_xthreat.py::test_singh_path_byte_identical_to_legacy` + `_on_worldcup`) stay byte-identical — asserted green in the full suite.
- **CI scope**: `python -m ruff check silly_kicks/ tests/` + `ruff format --check` + bare `pyright` + `python -m pytest tests/ -m "not e2e"`.

## File structure
- `silly_kicks/xthreat/_model.py` — add the 4 methods + imports (`json`, `pathlib.Path`, `dataclasses.asdict`, `os` for the `os.PathLike` type, `Any`/`Mapping` from `typing`/`collections.abc`, and `_METHOD_TO_PARAMS_TYPE` from `xthreat._params`). `NotFittedError` is ALREADY imported (`_model.py:9`); `validate_params_for_method`/`GridSpec`/`Method`/`XtParams` already imported (`:20`).
- `tests/xthreat/test_serialize.py` — NEW (the `tests/xthreat/` sub-dir already holds `test_counterfactual_seam.py`; the newest xthreat-test convention). The 7 red-green tests.
- `CHANGELOG.md`, `docs/superpowers/adrs/ADR-NNN-*.md`, `silly_kicks/_version.py` (→ `<next-free version>`), `uv.lock` (`uv lock` follows).
- **No** `xthreat/__init__.py __all__` change (spec §2 SHOULD-FIX). **No** `NOTICE` change (Singh xT already cited). **No** C4 change.

---

### Task 1: Red — the 7 failing tests

**File:** `tests/xthreat/test_serialize.py`. Reuse the committed `spadl_actions` fixture (`tests/conftest.py:32`) for a real fit (regular suite, not `@e2e`). Fit both a `singh_counts` and a `kde_smoothed` model. `destination_profiles` imported from `silly_kicks.xthreat`.

**PLAN-02 (fixture non-degeneracy — verify at the red-green run):** the fidelity/functional tests are vacuous if `xT` is degenerate (all-zero / near-constant) on the small `spadl.json` fixture. In the fitted-model setup assert `np.count_nonzero(xt.xT) > 1` for BOTH methods before round-tripping; if `kde_smoothed` degenerates on `spadl.json` at the impl run, fit that leg on a larger committed fixture (e.g. the WC2018 HDF the `_on_worldcup` parity test uses) rather than weakening the assertion. This is an impl-time confirmation, recorded so the executor checks it rather than discovering a green-but-vacuous test.

- [ ] **Step 1: Write all 7 tests (spec §6), each mapped:**
  1. `test_round_trip_fidelity[singh_counts|kde_smoothed]` — `xt2 = ExpectedThreat.from_dict(xt.to_dict())`; `np.array_equal` on `xT` + `scoring_prob_matrix` + `shot_prob_matrix` + `move_prob_matrix` + `transition_matrix` + each `heatmaps` entry (same `len`); `==` on `l`/`w`/`eps`/`method`/`params`; `xt2.grid == xt.grid`; assert each array's `.dtype` and `.shape` equal.
  2. `test_functional_equivalence[...]` — `xs, ys = spadl_actions["start_x"].to_numpy(), spadl_actions["start_y"].to_numpy()`; `np.allclose(xt.rate(spadl_actions), xt2.rate(spadl_actions))` AND `destination_profiles(xt2, xs, ys)` equals `destination_profiles(xt, xs, ys)` on `zone_centres`/`zone_values`/`probabilities` (allclose). (Catches a silent orientation flip.) *(PLAN-01: xs/ys named so the test is copy-runnable.)*
  3. `test_unfitted_to_dict_raises` — `ExpectedThreat().to_dict()` → `pytest.raises(NotFittedError)` (`sklearn.exceptions.NotFittedError`).
  4. Both methods parametrized across 1/2 (proves `params` incl. `KDEParams.kernel` round-trips).
  5. `test_json_boundary[...]` — `ExpectedThreat.from_dict(json.loads(json.dumps(xt.to_dict())))` passes 1+2 (proves no non-JSON type leaks; a stray `np.float64`/`np.ndarray` would raise in `json.dumps`).
  6. `test_from_dict_fail_closed` — (a) `to_dict()` minus `transition_matrix` → `pytest.raises((KeyError, ValueError))`; (b) `format_version` set to `2` → `pytest.raises(ValueError)`; (c) `xT` overwritten all-zero → `pytest.raises(NotFittedError)`. NON-VACUITY: also assert the *unmutated* dict still reconstructs (so the test proves the guard fires on corruption, not on everything).
  7. `test_save_load_file_round_trip` — `xt.save(tmp_path/"xt.json"); ExpectedThreat.load(...)` passes 1+2; a file whose JSON has `format_version: 2` → `load` raises `ValueError` (wrapper inherits fail-closed).

- [ ] **Step 2: Run → RED.** `python -m pytest tests/xthreat/test_serialize.py -q` → all fail (methods absent: `AttributeError: to_dict`). Capture the red.

---

### Task 2: Green — implement the four methods

**File:** `silly_kicks/xthreat/_model.py`. Add imports; add methods on `ExpectedThreat`.

- [ ] **Step 1: Imports.** `import json`, `import os`, `from pathlib import Path`, `from dataclasses import asdict`, `from typing import Any`, `from collections.abc import Mapping`; extend the `_params` import to include `_METHOD_TO_PARAMS_TYPE`.

- [ ] **Step 2: `to_dict`.**
  - Guard: `if not np.any(self.xT): raise NotFittedError("ExpectedThreat.to_dict on an unfitted model (fit first).")` — same predicate as `require_fitted_xt`/`rate`.
  - Return `{"format_version": 1, "l": int(self.l), "w": int(self.w), "eps": float(self.eps), "method": self.method, "params": (asdict(self.params) if self.params is not None else None), "xT": self.xT.tolist(), "scoring_prob_matrix": self.scoring_prob_matrix.tolist(), "shot_prob_matrix": self.shot_prob_matrix.tolist(), "move_prob_matrix": self.move_prob_matrix.tolist(), "transition_matrix": self.transition_matrix.tolist(), "heatmaps": [h.tolist() for h in self.heatmaps]}`. (`.tolist()` yields pure Python scalars → JSON-safe; the 4 matrices are non-`None` on a fitted model, guaranteed by the guard.)

- [ ] **Step 3: `from_dict` (classmethod), fail-closed in spec-§5 order.**
  - `fv = d.get("format_version"); if fv != 1: raise ValueError(f"unsupported ExpectedThreat format_version {fv!r}; this reader supports 1")` — FIRST, catches missing AND newer.
  - Rebuild `params`: `pd = d.get("params"); params = (None if pd is None else _METHOD_TO_PARAMS_TYPE[d["method"]](**pd))`. Direct-index `d["method"]`/`d["l"]`/`d["w"]`/`d["eps"]` → `KeyError` on a missing required key (structural completeness). `_METHOD_TO_PARAMS_TYPE[...]` raises `KeyError` on an unknown method (belt-and-braces before the ctor's `validate_params_for_method`).
  - `model = cls(l=d["l"], w=d["w"], eps=d["eps"], method=d["method"], params=params)` — the ctor runs `validate_params_for_method` (raises `TypeError` on a method/params mismatch).
  - Set fitted state (direct-index → `KeyError` on any missing array; explicit `float64`): `model.xT = np.asarray(d["xT"], dtype=np.float64)`; the same for `scoring_prob_matrix`/`shot_prob_matrix`/`move_prob_matrix`/`transition_matrix`; `model.heatmaps = [np.asarray(h, dtype=np.float64) for h in d["heatmaps"]]`.
  - `require_fitted_xt(model, caller="from_dict")` — raises `NotFittedError` on an all-zero `xT` (non-vacuity, test 6c). Import `require_fitted_xt` from `xthreat._physical` (already a public seam) — **guard against a circular import**: `_physical` imports `ExpectedThreat` under `TYPE_CHECKING` only, so a top-level `from silly_kicks.xthreat._physical import require_fitted_xt` in `_model.py` is safe; if a cycle appears at import, fall back to a function-local import inside `from_dict`. Verify with `python -c "import silly_kicks.xthreat"`.
  - `return model`.

- [ ] **Step 4: `save` / `load` (thin wrappers).**
  - `def save(self, path: str | os.PathLike[str]) -> None: Path(path).write_text(json.dumps(self.to_dict()), encoding="utf-8")`.
  - `@classmethod def load(cls, path: str | os.PathLike[str]) -> "ExpectedThreat": return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))`.

- [ ] **Step 5: Docstrings + a doctest-safe Examples block.** Each method gets a numpydoc docstring; `to_dict`/`from_dict` carry an **indented RST literal block** Example (NOT an executable `>>>` — a real fit needs `actions`; the literal-block form satisfies `test_public_api_examples._has_real_example` without a doctest that CI would execute — per the doctest policy). Cross-ref ADR-NNN + ADR-041 (raw orientation).

- [ ] **Step 6: Run → GREEN.** `python -m pytest tests/xthreat/test_serialize.py -q` → 7 (parametrized ~11) pass. Then the xthreat regression set: `python -m pytest tests/test_xthreat.py tests/test_xthreat_physical.py tests/xthreat/ -q` → green (parity oracle untouched).

---

### Task 3: Full gates + docs + version

- [ ] **Step 1: Lint/type/full suite.** `python -m ruff check silly_kicks/ tests/` + `ruff format --check silly_kicks/ tests/` + bare `pyright` (0 errors) + `python -m pytest tests/ -m "not e2e"` (full green; confirm the SK-xT-1 parity tests pass unchanged).
- [ ] **Step 2: Docs.** `CHANGELOG.md` `Added` (the 4 methods; JSON/format_version/raw-orientation/fail-closed). **ADR-NNN** from `ADR-TEMPLATE.md` recording spec D1–D6 (esp. D3 raw-orientation, D4 format_version fail-closed, D5 no-SHA-sidecar with the bundled-vs-consumer-fitted reasoning, D6 from_dict-bypasses-fit). `_version.py` → `<next-free version>`; then `uv lock` (never hand-edit). **No** `NOTICE`/C4/`__all__`/glossary change (verify each is untouched).
- [ ] **Step 3: /final-review** (C4 drift-check confirms count 33 unchanged; version single-source gate; TODO grooming if applicable).

---

### Task 4: Commit gate

- [ ] **Step 1:** Re-run the full gate set green; capture the pytest exit code (not via `| tail`).
- [ ] **Step 2:** `git status` shows ONLY: `silly_kicks/xthreat/_model.py`, `tests/xthreat/test_serialize.py`, `scripts/build_gkdv_arm_values.py` + `tests/scripts/test_build_gkdv_arm_values.py` (the ADR-100 blast-radius co-change — the gkdv threat-arm refusal rested on "ExpectedThreat has no save/load"; adding the seam falsifies that, so its pre-authorised revisit is folded in here), `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `ADR-NNN-*.md`, `_version.py`, the spec+plan docs. (`uv.lock` is UNCHANGED — the project version is dynamic/editable per ADR-079, so `uv lock` is a no-op; not staged. The untracked `it` session-debris file is excluded, not mine.) Nothing else.
- [ ] **Step 3: STOP — request explicit owner approval to commit.** Show the diff / file list. On approval: single commit on `feat/expectedthreat-serialize-seam`, subject `feat(xthreat): ExpectedThreat serialize seam — to_dict/from_dict/save/load (SK-XT-SER, ADR-NNN) — silly-kicks <next-free version>`, `Co-Authored-By` trailer. Then owner-gated push → PR → CI green → admin merge (non-squash) → tag `v<next-free version>` → PyPI (each its own approval).

---

## Self-review
1. **Spec coverage:** §2 API → Task 2; §3 state → Task 2 Steps 2-3; §4 raw-orientation → test 2 + Task 2 (verbatim `.tolist()`/`np.asarray`, no transpose); §5 fail-closed → Task 2 Step 3 + test 6; §6 TDD → Task 1; §7 D1–D6 → ADR-NNN; §8 ship → Task 3-4. ✓
2. **SHOULD-FIX applied:** no `__all__` change; methods reached via the already-exported class. ✓
3. **No leakage/retrain:** additive methods only; `fit`/`rate` untouched; SK-xT-1 parity asserted green. C4 count 33 unchanged (no aggregator). ✓
4. **Circular-import risk** (`from_dict` → `require_fitted_xt`) explicitly guarded + verified (Task 2 Step 3). ✓
