# Ghost-GK Parameters-Only Artifacts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make distributed `GhostGkModel` artifacts parameters-only — `save()` never persists the three per-sample arrays — retiring the `ghost_gk_density_spread` column, adding a bundled-weights CI allowlist and corpus provenance, correcting two false model-card claims, and fixing the `from_variant("public")` alias that serves a restricted artifact.

**Architecture:** The served position (`predict_mean`) reads only tree nodes + baselines, so dropping `training_gk_x/y` + `training_leaves` from `save()` is byte-identical for every consumer except the density pass. `predict_density` is kept for locally-fit models; only its *serving* from a distributed artifact is removed. The strip is a pure re-save (`load(old).save(new)`), no retrain. Everything else in the plan is either (a) making the test suite green again after the column retires, (b) a durable guard so this can't recur, or (c) the independent alias fix folded in by owner decision.

**Tech Stack:** Python, numpy, pandas, pytest, sklearn (fit-time only), HistGradientBoostingRegressor, npz+JSON+SHA256 serialization.

---

## Reference facts (verified against the tree 2026-07-20)

Do not re-derive these; they are measured. If a line number has drifted, re-grep the symbol.

- **`save()`** — fitted-guard `_ghost_gk.py:1785-1794` (7 attrs); `save_dict` build `:1796-1812`; npz write `:1815`; metadata dict `:1820-1840` (stamps `sklearn.__version__` at `:1834`); SHA256 write follows.
- **`load()`** — pre-Option-A guard `:1922-1928`; array reads `:1935-1937`; attr assigns `:1947-1949`; sklearn-version warning `:1962-1972` (already reads recorded value); chirality verify `:1974-1982`.
- **`predict_density` guard** — `:1675-1682` (raises `"Model not fitted..."`); leaf-match weights recomputed per query at `:1697`.
- **`fit()`** — sets the three arrays at `:1587-1589` (UNCHANGED by this plan).
- **`compute_ghost_gk`** — signature `:2035-2044` (has `kde_backend`); density pass `:2115-2159` (inits 3 cols, calls `predict_density` at `:2132`, emits `[d.spread ...]` at `:2143`).
- **`add_ghost_gk`** — `features.py:4353`; `kde_backend` param + doc `:4392-4398`; `compute_ghost_gk` call `:4431-4439`; column slices `:4446` and `:4471`.
- **`ghost_gk_xfns`** — `features.py:4488`; `col_names` hard-coded `:4519`; `range(3)` loop `:4525`; `compute_ghost_gk` call `:4547-4554`.
- **Atomic mirror** — `atomic/tracking/features.py:32,38,69,96` is a pure re-export; **no code edit needed**, but separately test-covered.
- **`from_variant` defect** — `_xshot_occurrence.py:534-558` (`elif` at `:550`); `_xcross_attempt.py:565-584` (`elif` at `:576`). Caches: `_VARIANT_CACHE = {}` at `_xshot_occurrence.py:307` / `_xcross_attempt.py:79`. Weights roots: `_XSHOT_WEIGHTS_ROOT` `:306`, `_XCROSS_WEIGHTS_ROOT` `:78`. Both bundled `default/metadata.json` record `shipped_variant: "public"` (verified). `GhostGkModel.from_variant` `:1986` is unaffected.
- **Trainer** — `scripts/train_ghost_gk.py`; `save()` call `:610`; **round-trip verify `:616` iterates `_training_gk_x/y/leaves`** (BREAKS post-strip); metrics dict `:647-651` already computes `providers`/`n_games`/`n_samples` but writes them to `metrics.json`, which is **not** bundled.
- **Bundled weights dirs (7 total):** `_ghost_gk_weights/default`, `_gk_completion_weights/{default,skillcorner}`, `_xcross_weights/default`, `_xshot_weights/default`, `xtgk/_retention_weights/{default,skillcorner}`. Only ghost carries per-sample arrays.
- **Bundled `default` size:** `rfcde_weights.npz` = 7,376,181 bytes; the 3 arrays = 6,611,763 (89.6%). Post-strip ≈ 764,418 bytes.
- **Test surface** (§9 of spec, grep-derived): 7 test modules carrying the column string + 3 binary fixtures + 2 generators; 9 `kde_backend` call sites incl. the signature guard at `test_ghost_gk_kde_vectorized.py:970`; 3 literal-`"1.2.0"` sites; 2 perf-spy sites in `test_ghost_gk_frame_restriction.py`.

## Working conventions for this plan

- **No worktree, no incremental doc commits.** Per project convention: work in the current tree; the spec, this plan, the research note, CHANGELOG, and ADR all ride the SINGLE final feature commit (Task 15). Code tasks commit normally as they land.
- **Branch first.** Current branch is `main`; create a feature branch before Task 1 (Task 0).
- **Run the specific tests per task.** The full suite will be RED from Task 1 until Task 11 lands (the strip breaks goldens until they are regenerated). Each task runs its own targeted tests; Task 14 is the full-suite-green gate.
- **`uv run pyright` is BARE (full tree), never path-scoped** — CI gates the whole tree.
- **Regenerate goldens, never hand-edit binaries.** Golden `.npz`/`.parquet` fixtures are produced by their generator scripts.
- **`allow_pickle=True` for repo-owned fixtures** — several store object arrays; `allow_pickle=False` raises on them.

---

## File structure

**Modified — library:**
- `silly_kicks/tracking/_ghost_gk.py` — `save()`, `load()`, `predict_density` message, `compute_ghost_gk`, version constant, provenance in metadata (Tasks 1, 2, 8)
- `silly_kicks/tracking/features.py` — `add_ghost_gk`, `ghost_gk_xfns` (Task 3)
- `silly_kicks/tracking/_xshot_occurrence.py`, `silly_kicks/tracking/_xcross_attempt.py` — alias fix (Task 10)

**Modified — tests:**
- `tests/tracking/test_ghost_gk.py`, `test_ghost_gk_integration.py`, `test_action_ltr_mirror_invariance.py`, `test_ghost_gk_frame_restriction.py`, `test_ghost_gk_serve_mean.py`, `test_ghost_gk_refactor_equivalence.py`, `test_ghost_gk_r3.py`, `test_ghost_gk_kde_vectorized.py`, `test_train_ghost_gk_cli.py` — column/backend/version/perf-spy updates (Tasks 4, 5, 6, 7)
- `tests/test_add_star_purity.py` — the `_frames_with_ghost` fixture + registry (Task 4)
- `tests/invariants/conftest_id_scalar.py`, `tests/invariants/conftest_id_dtype.py` — `kde_backend` kwargs in registered calls (Task 5)

**Created — tests:**
- `tests/tracking/test_ghost_gk_parameters_only.py` — the strip contract (Task 1)
- `tests/tracking/test_bundled_weights_allowlist.py` — the allowlist gate (Task 9)
- `tests/scripts/test_ghost_gk_provenance.py` — corpus-provenance metadata contract (Task 8)
- `tests/tracking/test_from_variant_serve_identity.py` — serve-time variant-identity gate (Task 10)

**Modified — scripts / trainer:**
- `scripts/train_ghost_gk.py` — round-trip verify fix + provenance plumbing (Tasks 8, 13)
- delete `scripts/measure_ghost_gk_estimators.py`, `scripts/validate_ghost_gk_refit.py`, `scripts/measure_ghost_gk_serve_delta.py` (Task 13)

**Modified — artifact + docs:**
- `silly_kicks/tracking/_ghost_gk_weights/default/{rfcde_weights.npz,metadata.json,SHA256SUMS}` — regenerated (Task 11)
- `docs/huggingface/model-cards/ghost-gk-v1-model-card.md`, `docs/research/tf19_pr2/hf_upload_instructions.md`, `TODO.md`, `pyproject.toml`, `CLAUDE.md` (Task 12)
- `CHANGELOG.md`, `docs/superpowers/adrs/ADR-0XX-*.md`, version-bump sites (Task 15)

---

## Task 0: Branch

- [ ] **Step 1: Create the feature branch**

Run:
```bash
git checkout -b feat/ghost-gk-parameters-only
```
Expected: `Switched to a new branch 'feat/ghost-gk-parameters-only'`

- [ ] **Step 2: Confirm clean tree except the uncommitted design docs**

Run: `git status --short`
Expected: only `??` entries for `docs/superpowers/specs/2026-07-20-ghost-gk-parameters-only-artifacts-design.md`, `docs/superpowers/plans/2026-07-20-ghost-gk-parameters-only-artifacts.md`, and `docs/research/ghost_gk_spread_aggregates/`. No tracked-file modifications.

---

## Task 1: `save()` stops persisting arrays; `load()` tolerates absence; version 1.3.0; sklearn_version preserved

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `save()` `:1785-1840`, `load()` `:1922-1949`, version constant, `predict_density` guard `:1675-1682`
- Test: `tests/tracking/test_ghost_gk_parameters_only.py` (create)

- [ ] **Step 1: Write the failing contract test**

Create `tests/tracking/test_ghost_gk_parameters_only.py`:
```python
"""The parameters-only artifact contract (spec 2026-07-20, §2 + §4)."""

import json

import numpy as np
import pytest

from silly_kicks.tracking._ghost_gk import GhostGkModel


@pytest.fixture(scope="module")
def bundled() -> GhostGkModel:
    return GhostGkModel.from_variant("default")


def test_save_omits_the_three_arrays(tmp_path, bundled):
    """save() writes a 1.3.0 npz that contains none of the per-sample arrays."""
    bundled.save(tmp_path / "m")
    with np.load(tmp_path / "m" / "rfcde_weights.npz", allow_pickle=True) as z:
        files = set(z.files)
    assert "training_gk_x" not in files
    assert "training_gk_y" not in files
    assert "training_leaves" not in files
    # tree ensembles + baselines are still present
    assert "n_trees" in files and "n_trees_y" in files
    assert "baseline_x" in files and "baseline_y" in files


def test_metadata_marks_parameters_only(tmp_path, bundled):
    bundled.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["version"] == "1.3.0"
    assert meta["stores_training_data"] is False


def test_predict_mean_byte_identical_after_strip(tmp_path, bundled):
    """The served position is unchanged by dropping the arrays."""
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    golden = np.load(
        "tests/tracking/fixtures/ghost_gk_kde_golden.npz", allow_pickle=True
    )
    cols = [str(c) for c in golden["feature_cols"]]
    import pandas as pd

    X = pd.DataFrame(golden["features"][:20], columns=cols)[GHOST_GK_FEATURE_NAMES]
    before = bundled.predict_mean(X)

    bundled.save(tmp_path / "m")
    reloaded = GhostGkModel.load(tmp_path / "m")
    after = reloaded.predict_mean(X)
    assert np.array_equal(before, after)  # byte-identical, not approx


def test_sklearn_version_preserved_not_restamped(tmp_path, bundled, monkeypatch):
    """Migration must NOT rewrite the recorded training-time sklearn version."""
    recorded = bundled._sklearn_version
    assert recorded is not None
    # Simulate a different runtime sklearn than the one the model was fit under.
    import sklearn

    monkeypatch.setattr(sklearn, "__version__", recorded + "-different")
    bundled.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["sklearn_version"] == recorded  # preserved, not the runtime value


def test_predict_density_message_names_the_cause(tmp_path, bundled):
    """A loaded parameters-only model gives a density-specific error, not 'not fitted'."""
    bundled.save(tmp_path / "m")
    reloaded = GhostGkModel.load(tmp_path / "m")
    import pandas as pd

    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES

    X = pd.DataFrame(
        np.zeros((1, len(GHOST_GK_FEATURE_NAMES))), columns=GHOST_GK_FEATURE_NAMES
    )
    with pytest.raises(RuntimeError, match="parameters-only|density.*not.*available|fit.*locally"):
        reloaded.predict_density(X)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py -x -q`
Expected: FAIL — `test_save_omits_the_three_arrays` finds the arrays present; version is `"1.2.0"`; `_sklearn_version` attr may not exist.

- [ ] **Step 3: Store the recorded sklearn_version on load, add the attr default**

In `load()`, after `model.training_platform = metadata.get("training_platform")` (`:1955`), add:
```python
        model._sklearn_version = metadata.get("sklearn_version")
```
In `__init__` (near the other `self._…` initializers around `:1501-1503`), add:
```python
        self._sklearn_version: str | None = None
```

- [ ] **Step 4: `save()` drops the arrays and preserves the recorded sklearn version**

Replace the `save_dict` literal (`:1797-1805`, the six leading keys through `baseline_y`) — remove the three array entries so it begins:
```python
        save_dict: dict[str, np.ndarray] = {
            "n_trees": np.array([len(self._tree_nodes)]),
            "n_trees_y": np.array([len(self._tree_nodes_y)]),
            "baseline_x": np.array([self._baseline_x], dtype=np.float64),
            "baseline_y": np.array([self._baseline_y], dtype=np.float64),
        }
```
Change the fitted-guard (`:1785-1794`) to require only the four parameter attrs:
```python
        if (
            self._tree_nodes is None
            or self._tree_nodes_y is None
            or self._baseline_x is None
            or self._baseline_y is None
        ):
            msg = "Model not fitted. Call .fit() first."
            raise RuntimeError(msg)
```
In the metadata dict (`:1834`), replace the sklearn stamp:
```python
            "sklearn_version": self._sklearn_version or sklearn.__version__,
```
Add two keys to the metadata dict (after `"version"`):
```python
            "version": "1.3.0",
            "stores_training_data": False,
```

- [ ] **Step 5: `load()` tolerates missing arrays**

Replace the unconditional array reads (`:1935-1937`) with:
```python
            training_gk_x = np.array(data["training_gk_x"]) if "training_gk_x" in data.files else None
            training_gk_y = np.array(data["training_gk_y"]) if "training_gk_y" in data.files else None
            training_leaves = np.array(data["training_leaves"]) if "training_leaves" in data.files else None
```
The pre-Option-A guard at `:1922-1928` (requiring `n_trees_y` + `baseline_x`) stays — it correctly rejects genuinely-old artifacts; the three arrays are no longer required for a valid load.

- [ ] **Step 6: `predict_density` gives a density-specific message**

Replace the guard body at `:1675-1682`:
```python
        if self._tree_nodes is None or self._tree_nodes_y is None:
            msg = "Model not fitted. Call .fit() or .load() first."
            raise RuntimeError(msg)
        if self._training_leaves is None or self._training_gk_x is None or self._training_gk_y is None:
            msg = (
                "predict_density is not available on a parameters-only artifact "
                "(distributed artifacts store learned parameters only, not per-sample "
                "training data; spec 2026-07-20). Fit the model locally with .fit() to "
                "use the density path."
            )
            raise RuntimeError(msg)
```

- [ ] **Step 7: Run the contract test to green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py -x -q`
Expected: PASS (5 tests).

- [ ] **Step 8: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_parameters_only.py
git commit -m "feat(tracking): GhostGkModel save() is parameters-only (v1.3.0 artifact)"
```

---

## Task 2: `compute_ghost_gk` drops the density pass, `kde_backend`, and the spread column

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `compute_ghost_gk` `:2035-2159`
- Test: reuse `tests/tracking/test_ghost_gk_parameters_only.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/tracking/test_ghost_gk_parameters_only.py`:
```python
def test_compute_ghost_gk_emits_two_columns_no_kde_backend(bundled):
    """compute_ghost_gk serves positions only; no density column, no kde_backend kwarg."""
    import inspect

    from silly_kicks.tracking._ghost_gk import compute_ghost_gk

    sig = inspect.signature(compute_ghost_gk)
    assert "kde_backend" not in sig.parameters
    doc = compute_ghost_gk.__doc__ or ""
    assert "ghost_gk_density_spread" not in doc
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py::test_compute_ghost_gk_emits_two_columns_no_kde_backend -x -q`
Expected: FAIL — `kde_backend` is still a parameter.

- [ ] **Step 3: Rewrite the density pass to positions-only**

In `compute_ghost_gk`: remove `kde_backend: str = "vectorized",` from the signature (`:2043`). Replace the body `:2115-2159` with:
```python
    out = frames.copy()
    out["ghost_gk_x"] = np.nan
    out["ghost_gk_y"] = np.nan

    resolved, meta, _batch_features, positions, _clamped = _serve_positions_core(
        frames,
        model=model,
        home_team_id=home_team_id,
        actions=actions,
        carrier=carrier,
        link_frame_ids=link_frame_ids,
    )

    if len(positions) == 0:
        return out

    result_df = pd.DataFrame(
        {
            "game_id": meta["game_id"].values,
            "period_id": meta["period_id"].values,
            "frame_id": meta["frame_id"].values,
            "team_id": meta["gk_team_id"].values,
            "ghost_gk_x": positions[:, 0],
            "ghost_gk_y": positions[:, 1],
        }
    )

    gk_mask = out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)
    gk_rows_df = out.loc[gk_mask, ["game_id", "period_id", "frame_id", "team_id"]].copy()
    gk_rows_df = gk_rows_df.merge(
        result_df,
        on=["game_id", "period_id", "frame_id", "team_id"],
        how="left",
    )
    out.loc[gk_mask, "ghost_gk_x"] = gk_rows_df["ghost_gk_x"].values
    out.loc[gk_mask, "ghost_gk_y"] = gk_rows_df["ghost_gk_y"].values

    return out
```
Update the docstring line `:2047` from `Adds ghost_gk_x, ghost_gk_y, ghost_gk_density_spread columns.` to `Adds ghost_gk_x, ghost_gk_y columns.`, and delete the `ghost_gk_density_spread`-describing paragraph (`:2055-2063`) plus the `kde_backend` parameter doc block.

- [ ] **Step 4: Run to green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py -x -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py tests/tracking/test_ghost_gk_parameters_only.py
git commit -m "feat(tracking): compute_ghost_gk serves positions only (retire density pass)"
```

---

## Task 3: `add_ghost_gk` + `ghost_gk_xfns` drop the spread column and `kde_backend`

**Files:**
- Modify: `silly_kicks/tracking/features.py` — `add_ghost_gk` `:4353-4485`, `ghost_gk_xfns` `:4488-4571`

- [ ] **Step 1: Add the failing test**

Append to `tests/tracking/test_ghost_gk_parameters_only.py`:
```python
def test_add_ghost_gk_and_xfns_have_no_kde_backend():
    import inspect

    from silly_kicks.tracking.features import add_ghost_gk, ghost_gk_xfns

    assert "kde_backend" not in inspect.signature(add_ghost_gk).parameters
    assert "kde_backend" not in inspect.signature(ghost_gk_xfns).parameters


def test_ghost_gk_xfns_emits_six_columns_not_nine():
    """2 metric columns x 3 gamestate slots (spread retired) = 6, via the frames=None contract.

    frames=None is the ADR-005 no-frames path: the xfn early-returns named NaN columns WITHOUT
    needing a GK-bearing frame (verified on the pre-strip xfn: this exact call returns 9 named
    columns today, 3 metrics x 3 slots). It therefore exercises the column-NAME contract this task
    changes (col_names 3->2, slot loop stays 3), not the empty-data path. The frames-PRESENT
    emission path is covered by the Task 4 integration tests + the liveness gate — do not duplicate
    it here."""
    import pandas as pd

    from silly_kicks.tracking.features import ghost_gk_xfns

    (xfn,) = ghost_gk_xfns(home_team_id=1)
    out = xfn([pd.DataFrame(index=range(2))] * 3, None)
    assert list(out.columns) == [
        "ghost_gk_x_a0", "ghost_gk_y_a0",
        "ghost_gk_x_a1", "ghost_gk_y_a1",
        "ghost_gk_x_a2", "ghost_gk_y_a2",
    ]
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py -k "kde_backend or six_columns" -x -q`
Expected: FAIL — `kde_backend` present; xfns emits 9 columns.

- [ ] **Step 3: Edit `add_ghost_gk`**

Remove `kde_backend: str = "vectorized",` from the signature (`:4362`) and its parameter-doc block (`:4392-4398`). In the `compute_ghost_gk` call (`:4431-4439`), remove the `kde_backend=kde_backend,` line. In the two column lists, drop `"ghost_gk_density_spread"`:
- `:4446` → `[["game_id", "period_id", "frame_id", "team_id", "ghost_gk_x", "ghost_gk_y"]]`
- `:4471` → `[["ghost_gk_x", "ghost_gk_y"]]`

Update the docstring line `:4366` to `Adds ghost_gk_x, ghost_gk_y per action (defending GK's ghost position at the linked frame).`

- [ ] **Step 4: Edit `ghost_gk_xfns`**

Remove `kde_backend` from its signature. Change `col_names` (`:4519`) to `["ghost_gk_x", "ghost_gk_y"]`, and remove `kde_backend=kde_backend,` from the inner `compute_ghost_gk` call (`:4553`).

**DO NOT touch the `range(3)` at `:4525`** — it is the gamestate-SLOT loop (a0/a1/a2), matching `states[:3]` and the `range(3)` slot loop, NOT the column count. Output columns = `len(col_names)` × 3 slots = 2 × 3 = 6. Changing it to `range(2)` would silently drop slot `a2`. The `for slot in states[:3]` (`:4541`) and the enumerate at `:4556` are the frames-present slot loops and are also unchanged.

Verify after editing:
```bash
grep -n "range(3)\|states\[:3\]\|col_names = " silly_kicks/tracking/features.py | sed -n '1,6p'
```
Expected: `col_names` has 2 entries; `range(3)` and both `states[:3]` still present.

- [ ] **Step 5: Run to green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_parameters_only.py -k "kde_backend or six_columns" -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/features.py tests/tracking/test_ghost_gk_parameters_only.py
git commit -m "feat(tracking): add_ghost_gk / ghost_gk_xfns drop density_spread + kde_backend"
```

---

## Task 4: Update the column-carrying test surface (§9.1, §9.2) + purity fixture

**Files:**
- Modify: `tests/tracking/test_ghost_gk.py`, `test_ghost_gk_integration.py`, `test_action_ltr_mirror_invariance.py`, `test_ghost_gk_serve_mean.py`, `test_ghost_gk_refactor_equivalence.py`, `test_ghost_gk_frame_restriction.py` (the `_GHOST_COLS` list only; the perf spies are Task 7)
- Modify: `tests/test_add_star_purity.py` — `_frames_with_ghost` fixture `:186-194`
- Modify: `tests/tracking/test_aggregator_column_liveness.py:362` if it asserts a column count

- [ ] **Step 1: Inventory the exact assertions to change**

Run:
```bash
grep -rn "ghost_gk_density_spread" tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_action_ltr_mirror_invariance.py tests/tracking/test_ghost_gk_serve_mean.py tests/tracking/test_ghost_gk_refactor_equivalence.py tests/tracking/test_ghost_gk_frame_restriction.py tests/test_add_star_purity.py
```
Expected: the ~19 hits from the spec inventory. Read each in context.

- [ ] **Step 2: Remove the column from each assertion**

For each hit: if it is a column-presence assertion (`assert "ghost_gk_density_spread" in out.columns`), delete it. If it is a value/allclose assertion on the column, delete that assertion. If it is a member of a `_GHOST_COLS`/expected-columns list, remove the element. Do NOT touch `ghost_gk_x`/`ghost_gk_y` assertions. In `tests/test_add_star_purity.py` `_frames_with_ghost` (`:192-194`), delete the line `f["ghost_gk_density_spread"] = np.where(gk, 1.0, np.nan)`.

The `default_model_features`-bound density tests (§9.2, e.g. `test_golden_discrete_mode`, `test_golden_continuous`, `test_golden_fft_scalars`) are handled in Task 6, not here — leave them for now.

- [ ] **Step 3: Bump the three literal-`"1.2.0"` version-pin sites to `"1.3.0"` (§9.6)**

These assert the artifact version and break on the 1.3.0 bump. Two are in this task's modules; the third (`test_train_ghost_gk_cli.py`) is not column-carrying, so it is handled here explicitly rather than left to Task 14 discovery:
```bash
grep -rn '"1.2.0"' tests/tracking/test_ghost_gk_r3.py tests/tracking/test_ghost_gk_serve_mean.py tests/tracking/test_train_ghost_gk_cli.py
```
At `test_ghost_gk_r3.py:50`, `test_ghost_gk_serve_mean.py:219`, `test_train_ghost_gk_cli.py:65`, change `== "1.2.0"` to `== "1.3.0"` (keep the Option-A comment). Do NOT touch `tests/spadl/test_add_possessions.py` (that `1.2.0` is a library-version docstring, out of scope).

- [ ] **Step 4: Run each touched module (density-golden tests will still fail — expected)**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_action_ltr_mirror_invariance.py tests/tracking/test_ghost_gk_serve_mean.py tests/tracking/test_ghost_gk_refactor_equivalence.py tests/tracking/test_ghost_gk_r3.py tests/tracking/test_train_ghost_gk_cli.py tests/test_add_star_purity.py -q`
Expected: the column-carrying + version-pin tests PASS; any remaining failures are `default_model_features`-bound density goldens (Task 6) or perf spies (Task 7) — note which, do not fix here.

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(tracking): drop ghost_gk_density_spread from column assertions + bump version pins"
```

---

## Task 5: `kde_backend` removal across call sites + retire the signature guard (§9.3)

**Files:**
- Modify: `tests/tracking/test_ghost_gk_kde_vectorized.py:970` (the `inspect.signature` guard)
- Modify: `tests/invariants/conftest_id_scalar.py:877,1083`, `tests/invariants/conftest_id_dtype.py:175`, and any other `kde_backend=` keyword call site

- [ ] **Step 1: Find every remaining `kde_backend` keyword call site**

Run: `grep -rn "kde_backend" tests/ silly_kicks/ scripts/`
Expected: after Tasks 2–3, the remaining hits are test-side call sites + the signature guard + `predict_density`'s own `kde_backend` param (KEPT — `predict_density` still takes it) + its docstrings.

- [ ] **Step 2: Remove `kde_backend=` from `add_ghost_gk`/`compute_ghost_gk`/`ghost_gk_xfns` call sites**

For each call to those three functions passing `kde_backend=...`, delete the kwarg. **Do NOT** touch `model.predict_density(..., kde_backend=...)` calls — that method keeps the parameter.

- [ ] **Step 3: Retire the signature guard with a recorded reason**

At `tests/tracking/test_ghost_gk_kde_vectorized.py:966-970`, replace the assertion `assert "kde_backend" in inspect.signature(add_ghost_gk).parameters` and its test with:
```python
def test_add_ghost_gk_has_no_kde_backend_after_strip():
    """kde_backend was retired from the aggregator surface when the density column
    retired (spec 2026-07-20 §3.1 / §9.3). predict_density still accepts it for
    locally-fit models; the aggregators no longer serve density."""
    import inspect

    from silly_kicks.atomic.tracking.features import add_ghost_gk

    assert "kde_backend" not in inspect.signature(add_ghost_gk).parameters
```

- [ ] **Step 4: Run the affected suites**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py::test_add_ghost_gk_has_no_kde_backend_after_strip tests/invariants/ -q`
Expected: the guard test + invariants registries PASS (density goldens in the kde_vectorized module remain for Task 6).

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(tracking): retire kde_backend from aggregator call sites + signature guard"
```

---

## Task 6: Backend-parity fixture swap (kept) + retire real-model fft fidelity (§9.4)

**Files:**
- Modify: `tests/tracking/test_ghost_gk_kde_vectorized.py` — `default_model_features` fixture `:60-64`, `test_golden_fft_scalars` `:122+`, `test_golden_continuous`, `test_golden_discrete_mode`, and any other `default_model_features`-bound density test
- Modify: the generator `scripts/gen_ghost_gk_kde_golden.py`

- [ ] **Step 1: Replace `default_model_features` with a locally-fit 4000-sample model**

At `:60-64`, replace:
```python
@pytest.fixture(scope="module")
def fitted_density_model():
    """A locally-fit model (4000 samples) that CAN run predict_density.

    The bundled default is parameters-only post-2026-07-20 and cannot serve density.
    4000 samples clears the 1e-2 fft bound at ~2.3x margin (spec §9.4 measurement);
    400 samples does not (2/4 breach)."""
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    rng = np.random.default_rng(20260720)
    X = pd.DataFrame(rng.standard_normal((4000, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 4000).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 4000), "gk_y": rng.uniform(25, 45, 4000)})
    model = GhostGkModel(n_estimators=60)
    model.fit(X, labels)
    return model, X.iloc[:_N_GOLDEN]
```

- [ ] **Step 2: Rebind the KEPT backend-parity tests to the fitted model, regenerate their golden**

Backend-*vs-backend* parity (scipy vs vectorized/cpu-numba/fft/fft-cic) stays. Repoint `test_golden_continuous` and `test_golden_discrete_mode` at `fitted_density_model`, and regenerate the golden from the same fitted model:
```bash
.venv/Scripts/python.exe scripts/gen_ghost_gk_kde_golden.py --model fitted --seed 20260720 --n-samples 4000 --out tests/tracking/fixtures/ghost_gk_kde_golden.npz
```
(If the generator has no such flags, edit it to fit the identical 4000-sample seed-20260720 model instead of loading `from_variant("default")`, then run it.)

- [ ] **Step 3: Delete `test_golden_fft_scalars` with a recorded reason**

Replace the `test_golden_fft_scalars` body (`:122+`) with a skip-marked stub documenting the retirement:
```python
@pytest.mark.skip(
    reason="Real-model fft fidelity retired (spec 2026-07-20 §9.4). The property is "
    "unmeasurable once artifacts are parameters-only: kernel width scales with n_train "
    "(neff**(-1/6)), and no practical fitted fixture reaches the 36k-sample regime "
    "(4000 samples is 1.40x broader). Backend-vs-backend parity is retained on the fitted "
    "fixture above. Coverage lost is recorded in the ADR."
)
def test_golden_fft_scalars_RETIRED():
    pass
```

- [ ] **Step 4: Run the module**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_kde_vectorized.py -q`
Expected: PASS (backend-parity green on the fitted fixture; fft-scalars skipped).

- [ ] **Step 5: Commit**

```bash
git add tests/tracking/test_ghost_gk_kde_vectorized.py tests/tracking/fixtures/ghost_gk_kde_golden.npz scripts/gen_ghost_gk_kde_golden.py
git commit -m "test(tracking): backend-parity on fitted fixture; retire real-model fft fidelity"
```

---

## Task 7: Re-anchor the structural perf guards to the feature extractor (§9.5)

**Files:**
- Modify: `tests/tracking/test_ghost_gk_frame_restriction.py` — `:126-145` and `:367-390`

- [ ] **Step 1: Read both current spies**

Run: `sed -n '124,146p;365,392p' tests/tracking/test_ghost_gk_frame_restriction.py`
Both spy `model.predict_density` and read `captured[-1]`.

- [ ] **Step 2: Re-anchor to `_extract_all_ghost_gk_features`**

In each test, replace the `predict_density` spy with an extractor spy. Example for `test_restriction_shrinks_predict_set`:
```python
    def test_restriction_shrinks_predict_set(self, monkeypatch):
        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()

        import silly_kicks.tracking._ghost_gk as gg

        captured: list[int] = []
        orig = gg._extract_all_ghost_gk_features

        def spy(frames_arg, **kwargs):
            feats, meta = orig(frames_arg, **kwargs)
            captured.append(len(feats))
            return feats, meta

        monkeypatch.setattr(gg, "_extract_all_ghost_gk_features", spy)
        compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)
        restricted_n = captured[-1]
        captured.clear()
        compute_ghost_gk(frames, model=model, home_team_id=1)  # full
        full_n = captured[-1]

        assert restricted_n < full_n
        assert restricted_n == 2 * len(linked)  # 2 GKs per linked frame — behaviour unchanged
```
Apply the analogous change to `test_predict_set_equals_linked_count`. Update the docstring to say the guard now spies the feature extractor (the dominant remaining cost, ~18x predict_mean; spec §9.5). Confirm `compute_ghost_gk` internally routes extraction through the module-level `_extract_all_ghost_gk_features` name so the monkeypatch is observed; if it imports the symbol locally, patch at the import site instead (`silly_kicks.tracking._ghost_gk._extract_all_ghost_gk_features`).

- [ ] **Step 3: Run the module**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/tracking/test_ghost_gk_frame_restriction.py
git commit -m "test(tracking): re-anchor frame-restriction perf guards to the feature extractor"
```

---

## Task 8: Corpus provenance in metadata + trainer round-trip fix (§6)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` — `save()` metadata + a `corpus_provenance` attr
- Modify: `scripts/train_ghost_gk.py` — thread provenance into the model + fix the `:616` round-trip verify
- Test: `tests/scripts/test_ghost_gk_provenance.py` (create)

> **OWNER DECISION FLAGGED (see handoff):** the *migrated bundled* `default` (Task 11) cannot gain real provenance without a retrain — the in-memory 1.2.0 model carries no corpus info. This task makes `save()` record provenance when supplied and honest-null when not. Whether the bundled artifact ships `corpus_provenance: null` (honest) or an operator-supplied known block is the owner's call; the plan defaults to null.

> **The spec §6 "join-liveness gate" is deliberately NOT implemented, and this is a decision, not an omission.** §6's gate assumed provenance would perform the `match_id → is_public_row` classification join and record a public/restricted split. The owner decided provenance records **providers + counts only, no split** (design session 2026-07-20). With no split, there is no classification join: the trainer derives `providers` from the per-file `source_provider` column (`train_ghost_gk.py:307`) and `n_games` from `groups` — neither joins against a registered id set. A `assert_match_join_lives` guard would therefore protect a join that never runs — a *tested-but-uncalled gate*, the exact dead-guard anti-pattern this repo warns about. It is dropped. If a future change adds the visibility split to provenance, it adds the split **and** its liveness guard together. Recorded in the ADR (Task 15).

- [ ] **Step 1: Write the failing gate**

Create `tests/scripts/test_ghost_gk_provenance.py`:
```python
"""Corpus-provenance metadata block (spec 2026-07-20 §6). Providers + counts only, no split."""

import json

import numpy as np
import pandas as pd

from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel


def _fit_small() -> GhostGkModel:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(rng.standard_normal((300, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, 300).astype(float)
    labels = pd.DataFrame({"gk_x": rng.uniform(2, 20, 300), "gk_y": rng.uniform(25, 45, 300)})
    m = GhostGkModel(n_estimators=20)
    m.fit(X, labels)
    return m


def test_provenance_recorded_when_supplied(tmp_path):
    m = _fit_small()
    m.corpus_provenance = {"providers": ["gradientsports", "sportec"], "n_games": 71, "n_rows": 300}
    m.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["corpus_provenance"]["n_games"] == 71
    assert meta["corpus_provenance"]["providers"] == ["gradientsports", "sportec"]
    # NEVER a per-match id list and NEVER a public/restricted split
    assert "match_ids" not in meta["corpus_provenance"]
    assert "visibility" not in meta["corpus_provenance"]


def test_provenance_null_when_absent(tmp_path):
    m = _fit_small()
    m.save(tmp_path / "m")
    meta = json.loads((tmp_path / "m" / "metadata.json").read_text())
    assert meta["corpus_provenance"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_ghost_gk_provenance.py -x -q`
Expected: FAIL — `corpus_provenance` attr/key missing.

- [ ] **Step 3: `save()` records the provenance block**

In `__init__`, add `self.corpus_provenance: dict | None = None`. In `load()`, add `model.corpus_provenance = metadata.get("corpus_provenance")`. In the metadata dict (`:1820+`), add:
```python
            "corpus_provenance": self.corpus_provenance,
```

- [ ] **Step 4: Fix the trainer round-trip verify (breaks post-strip)**

At `scripts/train_ghost_gk.py:616`, the attr loop iterates `_training_gk_x/y/leaves`, which are `None` after load on a parameters-only artifact. Replace the loop to verify only the persisted parameters:
```python
    for attr in ("_tree_nodes", "_tree_nodes_y"):
        orig = getattr(final_model, attr)
        back = getattr(loaded, attr)
        for i, (a, b) in enumerate(zip(orig, back, strict=True)):
            np.testing.assert_array_equal(a, b, err_msg=f"{attr}[{i}]")
    assert loaded._baseline_x == final_model._baseline_x  # noqa: S101
    assert loaded._baseline_y == final_model._baseline_y  # noqa: S101
```
Also thread provenance into the saved model, before `final_model.save(...)` at `:610`:
```python
    final_model.corpus_provenance = {
        "providers": sorted({str(p) for p in provider_labels.tolist()}),
        "n_games": len(set(groups.tolist())),
        "n_rows": len(features),
    }
```
(Providers come from the per-file `source_provider` column already read at `:307`; `n_games` from `groups`. No match→registered classification join is performed — see the task preamble on why the §6 join-liveness gate is deliberately absent.)

- [ ] **Step 5: Run to green**

Run: `.venv/Scripts/python.exe -m pytest tests/scripts/test_ghost_gk_provenance.py -x -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk.py scripts/train_ghost_gk.py tests/scripts/test_ghost_gk_provenance.py
git commit -m "feat(tracking): corpus-provenance metadata block + trainer round-trip fix"
```

---

## Task 9: Bundled-weights allowlist CI gate (§5) + shipped-artifact contract

**Files:**
- Test: `tests/tracking/test_bundled_weights_allowlist.py` (create)

**Design note (resolves review C1 — denylist → allowlist).** An earlier draft of this task coded a
`_FORBIDDEN` *denylist* (`set(z.files) & {training_gk_x, training_gk_y, training_leaves}`), which is
**fail-open**: it rejects exactly today's three names and permits everything else, so a future
`training_gk_z` / `sample_weights` / renamed `gk_positions` passes silently — the exact unforeseen
case the control exists to stop, and a direct contradiction of spec §5's own words ("*a NAME
allowlist — each bundled artifact declares the array names it may contain; anything else fails*").
This task implements the **allowlist**: the ghost npz may contain **only** the parameter arrays;
any unrecognized name fails. That is fail-closed against a rename or a new array, without the
`max_leaf_nodes` shape formula §5 correctly rejects as unwritable.

A bare **size cap** (`max(shape) <= 4096`) was considered and NOT adopted as the primary guard: it
is fail-open for a small-subsample artifact (per-sample arrays < 4096 pass) AND fail-*closed* for a
legitimately larger tree (a future `max_leaf_nodes` bump pushes `tree_nodes_*` above 4096 and trips
a false positive). The name allowlist has neither failure mode. It stays name-scoped by necessity
(the artifact records no corpus-size hyperparameter to bound against), so it remains a guard against
inadvertence rather than a determined author — but it is now fail-closed against the inadvertent
case, which is §5's stated goal.

- [ ] **Step 1: Write the gate (allowlist + shipped-artifact contract + non-vacuity)**

Create `tests/tracking/test_bundled_weights_allowlist.py`:
```python
"""Bundled weights artifacts carry learned parameters only (spec 2026-07-20 §5).

Allowlist, not denylist: the ghost npz may contain ONLY the parameter arrays; any unrecognized
name FAILS (fail-closed against a rename or a new per-sample array). Only ghost ships an npz today
(the other six bundled dirs are model.json boosters/logistic — no arrays to inspect), so the gate
is scoped to the ghost npz by enumeration, with a non-vacuity meta-test."""

import json
import re
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2] / "silly_kicks"
_GHOST_NPZ = _ROOT / "tracking" / "_ghost_gk_weights" / "default" / "rfcde_weights.npz"

# The ONLY array names a parameters-only ghost artifact may contain. Anything else fails.
_ALLOWED_EXACT = {"n_trees", "n_trees_y", "baseline_x", "baseline_y"}
_ALLOWED_PREFIX = ("tree_nodes_", "tree_dtype_")  # covers both x and y ensembles (…_y_ starts with these)


def _is_allowed(name: str) -> bool:
    return name in _ALLOWED_EXACT or name.startswith(_ALLOWED_PREFIX)


def test_the_ghost_npz_exists_and_is_enumerated():
    """Non-vacuity: the gate must actually be pointed at a real bundled artifact."""
    assert _GHOST_NPZ.exists(), f"ghost artifact not found at {_GHOST_NPZ}"


@pytest.mark.xfail(
    strict=True,
    reason="Bundled artifact is stripped in Task 11; this xfail is REMOVED there. "
    "strict=True means it FAILS loudly if the migration is forgotten (test passes but marker remains).",
)
def test_ghost_npz_contains_only_allowed_parameter_arrays():
    with np.load(_GHOST_NPZ, allow_pickle=True) as z:
        unexpected = sorted(n for n in z.files if not _is_allowed(n))
    assert not unexpected, f"ghost npz carries non-parameter arrays {unexpected} — parameters-only violated"


@pytest.mark.xfail(
    strict=True,
    reason="Shipped contract (C3): version/flag are set by the Task 11 migration; xfail REMOVED there.",
)
def test_shipped_ghost_default_is_parameters_only_v130():
    """C3: lock the SHIPPED artifact contract, not just the re-save round-trip.

    The round-trip contract test re-saves a copy and checks that; it always passes regardless of
    the committed artifact. This asserts the file that actually ships."""
    meta = json.loads((_GHOST_NPZ.parent / "metadata.json").read_text())
    assert meta["version"] == "1.3.0"
    assert meta["stores_training_data"] is False


def test_allowlist_is_not_vacuous(tmp_path):
    """A synthetic artifact WITH an unrecognized array must fail the allowlist rule."""
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, gk_positions=np.zeros((3, 2)), n_trees=np.array([1]))
    with np.load(bad, allow_pickle=True) as z:
        unexpected = sorted(n for n in z.files if not _is_allowed(n))
    assert unexpected == ["gk_positions"]  # a rename is caught — this is the fail-closed property
```

- [ ] **Step 2: Run — the two xfail-strict tests xfail (green), the rest pass**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_bundled_weights_allowlist.py -q`
Expected: `2 xfailed, 2 passed`. The gate is committed-before-fix (TDD), but CI is GREEN across commits 9→10 — the ghost xfails flip to expected-pass in Task 11 (which removes both markers). If Task 11 forgets to strip, the xfail stays correct (still fails); if it strips but forgets to remove the marker, `strict=True` turns the now-passing test into a loud failure — a forcing function.

- [ ] **Step 3: Commit**

```bash
git add tests/tracking/test_bundled_weights_allowlist.py
git commit -m "test(tracking): bundled-weights parameter-array allowlist + shipped-artifact contract"
```

---

## Task 10: `from_variant("public")` alias fix + serve-time identity gate (§8)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` `:534-558`, `silly_kicks/tracking/_xcross_attempt.py` `:565-584`
- Test: `tests/tracking/test_from_variant_serve_identity.py` (create)

- [ ] **Step 1: Write the failing gate**

Create `tests/tracking/test_from_variant_serve_identity.py`:
```python
"""from_variant('public') must NOT serve the restricted sc_extended artifact (spec §8)."""

import json

from silly_kicks.tracking._xshot_occurrence import (
    _HUB_VARIANTS,
    _VARIANT_ALIASES,
    _XSHOT_WEIGHTS_ROOT,
)


def test_public_resolves_to_bundled_default():
    assert _VARIANT_ALIASES["public"] == "default"


def test_hub_variants_do_not_include_public():
    """A name presented as reproducible must resolve inside the wheel."""
    assert _HUB_VARIANTS.isdisjoint({"public", "default"})


def test_bundled_default_declares_public_shipped_variant():
    """The alias is the literal truth, not a shim: default's metadata says shipped_variant=public."""
    meta = json.loads((_XSHOT_WEIGHTS_ROOT / "default" / "metadata.json").read_text())
    assert meta["shipped_variant"] == "public"


def test_public_and_default_return_the_same_bundled_object():
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    a = XShotOccurrenceModel.from_variant("public")
    b = XShotOccurrenceModel.from_variant("default")
    assert a is b  # same cached bundled instance; NOT a Hub download
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_from_variant_serve_identity.py -x -q`
Expected: FAIL — `_VARIANT_ALIASES` / `_HUB_VARIANTS` undefined.

- [ ] **Step 3: Add the alias table + resolve before the cache, in both models**

In `_xshot_occurrence.py` near `:306-307` add:
```python
_VARIANT_ALIASES = {"public": "default"}  # bundled default IS the public arm (metadata-verified)
_HUB_VARIANTS = frozenset({"sc_extended"})
```
Rewrite `from_variant` (`:534-558`) so the alias resolves FIRST:
```python
    @classmethod
    def from_variant(cls, variant: str = "default") -> XShotOccurrenceModel:
        variant = _VARIANT_ALIASES.get(variant, variant)
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        weights_dir = _XSHOT_WEIGHTS_ROOT / variant
        if (weights_dir / "SHA256SUMS").exists():
            model = cls.load(weights_dir)
        elif variant in _HUB_VARIANTS:
            model = cls.from_hub(_HF_REPO_ID)
        else:
            raise FileNotFoundError(
                f"No bundled xShotOccurrence weights for variant {variant!r} at {weights_dir}. "
                "Train via scripts/train_xshot_occurrence.py, or use from_hub()."
            )
        _VARIANT_CACHE[variant] = model
        return model
```
Apply the byte-equivalent change to `_xcross_attempt.py` (`_VARIANT_ALIASES`/`_HUB_VARIANTS` near `:78-79`, `from_variant` at `:565-584`).

- [ ] **Step 4: Run to green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_from_variant_serve_identity.py -x -q`
Expected: PASS. (`test_public_and_default_return_the_same_bundled_object` requires no network — both hit the bundled `default`.)

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_xshot_occurrence.py silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_from_variant_serve_identity.py
git commit -m "fix(tracking): from_variant('public') serves bundled default, not restricted sc_extended"
```

---

## Task 11: Regenerate the bundled `default` artifact (migration, §4)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk_weights/default/{rfcde_weights.npz,metadata.json,SHA256SUMS}`
- Modify: `tests/tracking/test_bundled_weights_allowlist.py` (remove the two Task 9 xfail markers)

- [ ] **Step 1: Migrate via load(old).save(new) into a temp dir and verify byte-identity**

Run:
```bash
.venv/Scripts/python.exe - <<'PY'
import numpy as np, pandas as pd, tempfile, pathlib, json
from silly_kicks.tracking._ghost_gk import GhostGkModel, GHOST_GK_FEATURE_NAMES
src = pathlib.Path("silly_kicks/tracking/_ghost_gk_weights/default")
m = GhostGkModel.load(src)
g = np.load("tests/tracking/fixtures/ghost_gk_kde_golden.npz", allow_pickle=True)
X = pd.DataFrame(g["features"][:50], columns=[str(c) for c in g["feature_cols"]])[GHOST_GK_FEATURE_NAMES]
before = m.predict_mean(X)
tmp = pathlib.Path(tempfile.mkdtemp()) / "m"
m.save(tmp)
back = GhostGkModel.load(tmp)
assert np.array_equal(before, back.predict_mean(X)), "predict_mean drifted"
meta = json.loads((tmp/"metadata.json").read_text())
assert meta["version"] == "1.3.0" and meta["stores_training_data"] is False
assert meta["sklearn_version"] == json.loads((src/"metadata.json").read_text())["sklearn_version"], "sklearn_version re-stamped"
with np.load(tmp/"rfcde_weights.npz", allow_pickle=True) as z:
    assert not ({"training_gk_x","training_gk_y","training_leaves"} & set(z.files))
print("MIGRATION OK; new npz bytes:", (tmp/"rfcde_weights.npz").stat().st_size)
print("TMP:", tmp)
PY
```
Expected: `MIGRATION OK; new npz bytes: ~764418` and the `TMP` path.

- [ ] **Step 2: Copy the three migrated files over the bundled artifact**

Run (substitute the printed `TMP`):
```bash
cp "<TMP>/rfcde_weights.npz" "<TMP>/metadata.json" "<TMP>/SHA256SUMS" silly_kicks/tracking/_ghost_gk_weights/default/
```

- [ ] **Step 3: Remove the two Task 9 xfail markers and verify the gate passes for real**

In `tests/tracking/test_bundled_weights_allowlist.py`, delete the `@pytest.mark.xfail(...)` decorator above **both** `test_ghost_npz_contains_only_allowed_parameter_arrays` and `test_shipped_ghost_default_is_parameters_only_v130` (the migrated artifact now satisfies them for real).

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_bundled_weights_allowlist.py -q`
Expected: `4 passed` (no xfail). If either still xfails, the migration did not take — re-check Step 2. If either now ERRORs as `XPASS`, the marker removal was missed.

- [ ] **Step 4: Verify from_variant + chirality load the migrated artifact**

Run: `.venv/Scripts/python.exe -c "from silly_kicks.tracking._ghost_gk import GhostGkModel; m=GhostGkModel.from_variant('default'); print('loaded, version ok')"`
Expected: `loaded, version ok` with no `IntegrityError` (chirality fingerprint is unchanged — it reads `predict_mean`).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/tracking/_ghost_gk_weights/default/ tests/tracking/test_bundled_weights_allowlist.py
git commit -m "chore(weights): migrate bundled ghost-GK default to parameters-only v1.3.0 (7.4MB->764KB)"
```

---

## Task 12: Model card + companion documents (§3.3, §7, §12)

**Files:**
- Modify: `docs/huggingface/model-cards/ghost-gk-v1-model-card.md` `:82`, `:84`, the `predict_density` example
- Modify: `docs/research/tf19_pr2/hf_upload_instructions.md` `:174-186`
- Modify: `TODO.md` `:167-181`
- Modify: `pyproject.toml` `:135`
- Modify: `CLAUDE.md` `:13`, `:19`

- [ ] **Step 1: Replace card `:84` wholesale, add the `predict_density` precondition**

Replace the `:84` sentence with:
```
Only the learned model parameters are published: the two gradient-boosted tree ensembles (split thresholds, feature indices, leaf values) and their additive baselines. No per-sample training data and no raw provider tracking data is redistributed.
```
`:82` (Gradient Sports row) needs no rewording — it becomes true of a 1.3.0 artifact. Confirm the exact post-edit line carries no residual per-sample / "leaf-aggregated" phrasing (review C5). It reads, verbatim and unchanged:

```
| Gradient Sports | FIFA World Cup 2022 | Owner-tier source — only the trained model weights are distributed here; the underlying raw tracking data is **not** redistributed |
```

"only the trained model weights are distributed" is true once the arrays are stripped; there is no "leaf-aggregated" or per-sample wording to remove. Leave it unchanged, but verify by grep in Step 5.

Add a one-line precondition above the `predict_density` runnable example: *"`predict_density` requires a locally-fit model; the distributed artifact is parameters-only and serves `predict_mean` positions."*

- [ ] **Step 2: Runbook — mirror the corrected claim, state the ordering, add the pre-upload assertion**

In `docs/research/tf19_pr2/hf_upload_instructions.md:174-186`: update the licensing-note quote to the corrected `:84` wording; add the constraint *"The Hub `full` upload MUST be produced by the post-strip `save()` (v1.3.0, parameters-only) and MUST NOT precede the parameters-only PR."*; and add the pre-upload assertion:
```bash
python -c "import numpy as np,sys; z=np.load(sys.argv[1],allow_pickle=True); bad={'training_gk_x','training_gk_y','training_leaves'}&set(z.files); assert not bad, f'ABORT: staged artifact carries {bad}'; print('OK: parameters-only')" <staged>/rfcde_weights.npz
```

- [ ] **Step 3: TODO.md — reflect the ordering constraint**

Edit `TODO.md:167-181` so the ghost Hub upload entry states it is gated on the parameters-only PR + produced by the post-strip `save()`; remove the "a live breakage, not a nice-to-have" framing that reads as an instruction to proceed.

- [ ] **Step 4: pyproject + CLAUDE.md figure corrections**

`pyproject.toml:135`: change `The "default" weights (~12 MB) ship bundled in the package.` to `The "default" weights (~0.76 MB, parameters-only since v1.3.0) ship bundled in the package.`
`CLAUDE.md:13`: correct the bundled-variant figures (`default` is now ~0.76 MB / parameters-only; drop the stale "9 MB, 36k samples" / "91 MB, 537k samples").
`CLAUDE.md:19`: change `predict_density retained for ghost_gk_density_spread + the mode` to note the column is retired and `predict_density` serves locally-fit models only.

- [ ] **Step 5: Verify no lingering false claim in the two public docs**

Run: `grep -n "leaf-aggregated\|no raw provider tracking data\|~12 MB\|ghost_gk_density_spread" docs/huggingface/model-cards/ghost-gk-v1-model-card.md pyproject.toml CLAUDE.md`
Expected: the only remaining `ghost_gk_density_spread` hits are historical CHANGELOG-style references (none in the card as a live claim); no `leaf-aggregated`, no `~12 MB`.

- [ ] **Step 6: Commit**

```bash
git add docs/huggingface/model-cards/ghost-gk-v1-model-card.md docs/research/tf19_pr2/hf_upload_instructions.md TODO.md pyproject.toml CLAUDE.md
git commit -m "docs: correct ghost-GK card + runbook + companion docs for parameters-only artifacts"
```

---

## Task 13: Retire the three stale scripts (§9.8)

**Files:**
- Delete: `scripts/measure_ghost_gk_estimators.py`, `scripts/validate_ghost_gk_refit.py`, `scripts/measure_ghost_gk_serve_delta.py`

- [ ] **Step 1: Confirm nothing imports them**

Run: `grep -rn "measure_ghost_gk_estimators\|validate_ghost_gk_refit\|measure_ghost_gk_serve_delta" silly_kicks/ tests/ scripts/ docs/`
Expected: hits only inside the three files themselves and in `docs/PRIVATE_CONSUMERS.md` / CLAUDE.md prose if any. If a test or `docs/PRIVATE_CONSUMERS.md` pins one, stop and reconcile before deleting.

- [ ] **Step 2: Delete the three files**

Run:
```bash
git rm scripts/measure_ghost_gk_estimators.py scripts/validate_ghost_gk_refit.py scripts/measure_ghost_gk_serve_delta.py
```

- [ ] **Step 3: Commit**

```bash
git commit -m "chore(scripts): retire 3 ghost-GK scripts that score the KDE mode (dead since ADR-016)"
```

---

## Task 14: Full suite green + full-tree pyright

- [ ] **Step 1: Run the full non-e2e suite, benchmark-skipped, to a log**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --benchmark-skip -q > pytest_full.txt 2>&1; echo "EXIT=$?"`
Expected: `EXIT=0`. If not, open `pytest_full.txt`, find the failures, and fix them in the owning task's spirit (do NOT loosen a golden bound or skip a real test to force green). Common expected touch-ups: a `ghost_gk_refactor_golden.npz` / `ghost_gk_backward_compat.parquet` regeneration (their generators are `scripts/make_ghost_gk_golden.py` and the refactor-equivalence fixture builder).

- [ ] **Step 2: Regenerate the remaining two binary goldens if they fail**

If `test_ghost_gk_refactor_equivalence.py` or the backward-compat test fails on a stored-column mismatch:
```bash
.venv/Scripts/python.exe scripts/make_ghost_gk_golden.py   # regenerates ghost_gk_refactor_golden.npz sans density col
```
Regenerate `ghost_gk_backward_compat.parquet` from its builder (grep `ghost_gk_backward_compat` for the generator). Re-run Step 1.

- [ ] **Step 3: Full-tree pyright (BARE — never path-scoped)**

Run: `uv run pyright`
Expected: no NEW errors introduced by this change. Compare against `main`'s baseline for pre-existing stub noise (a bare local run reports more than CI; the real question is whether an error is in a file this PR touched).

- [ ] **Step 4: Commit any green-up fixes**

```bash
git add tests/ scripts/
git commit -m "test(tracking): regenerate remaining ghost goldens; full suite green"
```
Then remove the scratch log: `rm pytest_full.txt` (never commit it).

---

## Task 15: CHANGELOG, ADR, version bump — the single docs+release commit

**Files:**
- Modify: `CHANGELOG.md`
- Create: `docs/superpowers/adrs/ADR-0XX-ghost-gk-parameters-only-artifacts.md`
- Modify: the 5 version-bump sites (see `reference_version_bump_checklist`)
- Stage (already on disk, uncommitted): the spec, this plan, the research note

- [ ] **Step 1: Write the ADR, Decision section carrying the §10 dispositions**

Create the ADR. Its Decision section MUST record, each as a decision with reasoning:
1. **Forward-only** — the disposition for already-published wheels (`default` ships in wheel + sdist; PyPI immutable; a yanked `==` still resolves) and Hub revisions.
2. **No retraining** — and its consequence for provenance: the migrated bundled `default` ships `corpus_provenance: null` (honest — the 1.2.0 model carries no corpus info; real provenance requires a retrain, which is out of scope), while the trainer plumbing ensures every *future* trained artifact carries it. (This is the owner-decision point flagged in Task 8 — record whichever way the owner ruled.)
3. **Hygiene framing** on a public repo.
4. **The §6 join-liveness gate is intentionally absent** — the owner's providers+counts-only (no split) provenance decision removed the classification join it would guard; a tested-but-uncalled guard is the dead-guard anti-pattern (review C2). If a future change adds the split, it adds the guard with it.
5. **The Hub `ghost-gk-v1` repo stays functionally broken for `from_hub` after this PR** — it has no chirality block, so 4.51.0+ `load()` rejects it, and this PR does not upload the stripped `full`. Task 12 gates that upload behind this PR + a post-strip `save()`; the ADR OWNS this as a stated, deferred disposition so it does not read as fixed. Restoring `from_hub` is a separate scheduled follow-up (produce stripped `full` via `save()` → run the runbook's pre-upload assertion → upload).

Plus: the owner sign-off for retiring `ghost_gk_density_spread` (design session 2026-07-20); the alias fix's own line; the retired real-model fft coverage (§9.4); and the §5 allowlist's name-scoped limitation (fail-closed against inadvertence, not a determined author; the bare size cap was considered and rejected for its two-sided fragility — review C1).

**State the allowlist as the generalizable control, not a ghost-specific patch** (review durable-quality note): the anti-rot property — a new array name in *any* bundled weights directory fails CI until a human classifies it as parameter-or-per-sample — is the durable long-term win, more than this one strip. It is what prevents the *next* inadvertent per-sample array in any future bundled model (a new trained head, a re-bundled variant), which is the recurrence class §5 exists to close. Frame the ADR decision around "distributed artifacts carry learned parameters only, enforced by a per-artifact name allowlist" rather than "ghost stopped shipping three arrays."

Cross-reference: amends ADR-016, ADR-038.

- [ ] **Step 2: CHANGELOG — two entries under a new version heading**

Add the version section with: the parameters-only artifact change (breaking artifact format 1.2.0→1.3.0, forward-incompatible; column `ghost_gk_density_spread` + `kde_backend` retired; bundled `default` 7.4MB→764KB; no retrain; lakehouse re-materializes the retired passthrough column) AND a **separate, standalone** line for the `from_variant("public")` alias fix (user-actionable: `from_variant("public")` now serves the bundled reproducible model, not the Hub-hosted restricted `sc_extended`).

- [ ] **Step 3: Bump the version at all 5 sites**

Follow `reference_version_bump_checklist` (do not guess the number — it is the next free release; confirm with the owner). **Cross-workstream coordination:** the other session's parked ADR-044 also wants the next release slot — the owner assigns and merges the number; do not model the other session's timing or claim a number. Update all five sites consistently once the owner confirms.

- [ ] **Step 4: Stage docs + everything, single final commit**

```bash
git add docs/superpowers/specs/2026-07-20-ghost-gk-parameters-only-artifacts-design.md \
        docs/superpowers/plans/2026-07-20-ghost-gk-parameters-only-artifacts.md \
        docs/research/ghost_gk_spread_aggregates/ \
        docs/superpowers/adrs/ CHANGELOG.md
git add -A
git commit -m "docs: spec + plan + research note + ADR + CHANGELOG + version bump for ghost-GK parameters-only"
```
(Note: per project convention the docs ride this final commit, not standalone doc commits. If the owner prefers one squashed commit for the whole PR, `git rebase` is a separate owner-driven step — do not squash unprompted.)

- [ ] **Step 5: Final verification before PR**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" --benchmark-skip -q > /tmp/final.txt 2>&1; echo "EXIT=$?"`
Expected: `EXIT=0`. Then `uv run pyright` clean of new errors. Only then open the PR.

---

## Self-review notes (author, for the reviewer)

- **Spec coverage:** §2→T1, §3.1→T2/T3, §3.3→T12, §4→T1/T11, §5→T9/T11, §6→T8, §7→T12, §8→T10, §9.1→T4, §9.2→T6, §9.3→T5, §9.4→T6, §9.5→T7, §9.6→T4 Step 3 (all three version-pin sites, explicit), §9.7→T1/T8/T9/T10, §9.8→T13, §10→T15 (ADR), §12→T12, §13→no code (open items).
- **Two gaps the spec's grep-inventory missed, both now covered:** the trainer round-trip verify at `train_ghost_gk.py:616` (T8 Step 4) and the fact that `metrics.json` does not ship so provenance must go in `metadata.json` (T8 Step 3).
- **One owner decision surfaced, not decided (T8 preamble, recorded in the ADR):** migrated bundled `default` provenance = honest-null vs operator-supplied.
- **Second external review (2026-07-20) applied:** C1 denylist→allowlist (T9, fail-closed against renames; size cap considered + rejected with reasons); C2 dead join-liveness guard dropped (T8, tied to the owner's no-split decision, recorded in the ADR); C3 shipped-artifact version/flag contract added (T9); C4 the two ghost gate cases are `xfail(strict=True)` in T9 and flipped in T11 → CI green across commits; C5 exact post-edit card `:82` quoted (T12); C6 surfaced a real plan bug — `range(3)` in `ghost_gk_xfns` is the gamestate-SLOT loop, not the column count, so it MUST stay `range(3)` (T3 Step 4 corrected; changing it would drop slot a2).
- **Red-CI window now bounded to targeted tests only:** with the T9 xfail approach the *committed* tree stays green from T9 onward; the only red is within a task's own targeted run before its implementation step. T14 is the full-suite green gate.
- **Hub `from_hub` disposition:** the public Hub `ghost-gk-v1` stays broken (no chirality block) after this PR until the stripped `full` is uploaded; T12 gates that upload and T15's ADR owns it as a stated deferred disposition (review cross-note).
- **Third review (2026-07-20) — approved to execute.** C1–C5 confirmed resolved; C6 demonstrated a non-issue (the `frames=None` xfns test reaches the column-name path — verified on the pre-strip xfn, returns 9 named columns; test docstring records this). Folded in the reviewer's durable-quality note: the ADR frames the name allowlist as the generalizable anti-rot control (T15), not a ghost-specific patch.
