# Owner-gated lakehouse-mart NLL cross-check — Implementation Plan (v3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SK-xT-1 KDE-vs-Singh held-out-NLL triangulation a permanent, owner-gated `@pytest.mark.e2e` regression tripwire against the real `dev_gold.fct_action_values` mart — with the shaping, the verdict arithmetic, and the loader import all on proven, unit-tested, non-fragile paths.

**Architecture:** `scripts/_loader_databricks.py` gains a mart **read** (`fetch_action_values`) and a pure mart→SPADL **shape** (`shape_action_values`) next to each other (both adapter logic). A shared pure verdict helper (`nll_relative_win`) lands in `tests/_xthreat_helpers.py`. The e2e test is a thin orchestrator — fetch → shape → coverage-guard → split → **passes-only** score → verdict — mirroring `tests/test_xthreat_statsbomb_e2e.py`. Shaping is unit-tested in the existing `tests/calibration/test_loader_databricks.py`; the verdict is unit-tested in a new regular test. Hard assertions fire **only on the full corpus**.

**Tech Stack:** pytest, pandas, numpy, `silly_kicks.xthreat`, `silly_kicks.spadl.config`, `scripts._loader_databricks` (PEP-420 namespace pkg), `databricks-sql-connector` (lazy, owner-only).

**Spec:** `docs/superpowers/specs/2026-06-09-xt-nll-lakehouse-e2e-design.md` (v2).

**Incorporates external review round 1 (2026-06-09):** C1 passes-only scored set; C2 live coverage guard; C3 full-corpus-only asserts + log-only `kde1`; H1 shaping-in-loader + `import scripts._loader_databricks`; H2 unit-tested verdict; process: single commit + `/final-review` + version-reconcile-at-commit.

**Incorporates external review round 2 (2026-06-09):** HIGH — drop `<NA>` rows after the coverage guard so the `Int64` `==`-masks are NA-free (a nullable-`<NA>` mask raises on the owner's `pandas<2.3.0`; CI's 2.3.3 is blind); M2 — fold the strict-beat+floor predicate into the CI-tested `kde_clears_tripwire`; M3 — `spadlconfig` import stays function-local; M1 — keep the per-PR bump but flag the wheel as byte-identical (CHANGELOG "no shipped-library change"); L2 — `test_xthreat_*` filename prefix.

**Execution note (2026-06-09):** the real-data validation runs (isolated env, owner mart) closed the round-1 `kde1` log-only deferral. The full run confirmed the shipped-default KDE(1.0) beats Singh at 16×12 full-mart (+3.03%), so `kde1` was **promoted from log-only to a hard STRICT-BEAT assert** (`floor=0.0`, full-corpus only). No floor on it — the default's margin erodes as the mart grows (smoke +8.71% → full +3.03%), so a floor would false-trip on benign growth; strict-beat catches only a real regression. Verdict unit tests: 10 → 11.

---

## Commit policy (read first)

This branch produces **one commit at the end** (Task 6), gated on **explicit user approval + the
git-commit sentinel** (standing policy + the sentinel hook). Tasks 1–5 stage changes and run
tests/lint but **do NOT commit**. Do all work on a feature branch created in Task 0.

## File Structure

The repo's xthreat tests are **flat** under `tests/` (`test_xthreat_*`) with a shared
`tests/_xthreat_helpers.py`; `pyproject.toml` sets `pythonpath = [".", "tests"]`, so `scripts` and
`tests` are both importable (`import scripts._loader_databricks as L`, `from tests._xthreat_helpers
import ...`).

- **Modify** `scripts/_loader_databricks.py` — add the `_ACTION_VALUES_TABLE`/`_ACTION_VALUES_COLUMNS` constants, `fetch_action_values()`, and the pure `shape_action_values()` (with a **function-local** `spadlconfig` import — the module's lazy-import discipline).
- **Modify** `tests/calibration/test_loader_databricks.py` — add unit tests for `shape_action_values`.
- **Modify** `tests/_xthreat_helpers.py` — add the pure `nll_relative_win()` + `kde_clears_tripwire()`.
- **Create** `tests/test_xthreat_nll_relative_win.py` — regular-suite unit test for both verdict helpers.
- **Create** `tests/test_xthreat_nll_lakehouse_e2e.py` — the owner-gated e2e orchestrator.
- **Modify** `TODO.md`, version sites (`pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `uv.lock`).

---

## Task 0: Orientation, branch, baseline

**Files:** none modified yet.

- [ ] **Step 1: Create the feature branch** (never a worktree, per standing policy)

Run: `git switch -c sk-xt-nll-lakehouse-e2e`
Expected: switched to a new branch off `main` (HEAD `2e913f0` / 4.21.1).

- [ ] **Step 2: Skim the two proven seams to mirror**

Run: `sed -n '85,101p' tests/test_xthreat_statsbomb_e2e.py; echo '---'; sed -n '1,10p' tests/calibration/test_loader_databricks.py`
Confirm: the sibling fits Singh/KDE on full `train` but **scores `holdout_passes`** (passes-only); the calibration test imports the loader as `import scripts._loader_databricks as L`. Reuse both shapes.

- [ ] **Step 3: Capture the green baseline**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q 2>&1 | tail -5`
Expected: all pass (record the count). Pre-change baseline.

---

## Task 1: Loader read + shape (`fetch_action_values`, `shape_action_values`)

**Files:**
- Modify: `scripts/_loader_databricks.py`
- Test: `tests/calibration/test_loader_databricks.py`

- [ ] **Step 1: Write the failing shaping unit tests**

Append to `tests/calibration/test_loader_databricks.py` (it already imports `pandas as pd` and `scripts._loader_databricks as L` at the top — add only the `spadlconfig` import):

```python
import silly_kicks.spadl.config as spadlconfig


def _mart_row(match_id, action_type, action_result, **kw):
    base = dict(
        match_id=match_id, start_x=10.0, start_y=20.0, end_x=30.0, end_y=40.0,
        action_type=action_type, action_result=action_result,
    )
    base.update(kw)
    return base


def test_shape_action_values_maps_strings_to_int_codes():
    df = pd.DataFrame([_mart_row(101, "pass", "success"), _mart_row(101, "dribble", "fail")])
    out = L.shape_action_values(df)
    assert out.loc[0, "type_id"] == spadlconfig.actiontype_id["pass"]
    assert out.loc[0, "result_id"] == spadlconfig.result_id["success"]
    assert out.loc[1, "type_id"] == spadlconfig.actiontype_id["dribble"]
    assert out.loc[1, "result_id"] == spadlconfig.result_id["fail"]


def test_shape_action_values_uses_nullable_int_dtype():
    out = L.shape_action_values(pd.DataFrame([_mart_row(1, "pass", "success")]))
    assert str(out["type_id"].dtype) == "Int64"
    assert str(out["result_id"].dtype) == "Int64"


def test_shape_action_values_aliases_match_id_to_game_id():
    out = L.shape_action_values(pd.DataFrame([_mart_row(777, "pass", "success")]))
    assert (out["game_id"] == 777).all()


def test_shape_action_values_tolerates_unmapped_vocab():
    # Unknown action_type/result -> <NA> (the move filter drops it; must not raise).
    df = pd.DataFrame([_mart_row(1, "teleport", "success"), _mart_row(1, "pass", "quantum")])
    out = L.shape_action_values(df)
    assert pd.isna(out.loc[0, "type_id"])
    assert pd.isna(out.loc[1, "result_id"])


def test_shape_action_values_preserves_coordinates():
    out = L.shape_action_values(pd.DataFrame([_mart_row(1, "pass", "success", start_x=5.5, end_y=63.0)]))
    assert out.loc[0, "start_x"] == 5.5
    assert out.loc[0, "end_y"] == 63.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/calibration/test_loader_databricks.py -q -k shape_action_values`
Expected: FAIL with `AttributeError: module ... has no attribute 'shape_action_values'`.

- [ ] **Step 3: Implement read + shape**

In `scripts/_loader_databricks.py`, add the constants beside `_ALLOWED_PROVIDERS` / `_FETCH_BATCH` (do **not** add a module-top `silly_kicks` import — the module keeps `silly_kicks` imports function-local; `shape_action_values` imports `spadlconfig` inside the function):

```python
# Fixed, fully-qualified gold mart for the owner-gated xT NLL cross-check (read-only).
# A module constant — never interpolated from caller input (mirrors the _ALLOWED_PROVIDERS discipline).
_ACTION_VALUES_TABLE = "soccer_analytics.dev_gold.fct_action_values"
# Only the passes-NLL columns; period/action_id deliberately omitted (unused — keeps the ~8.8M-row pull lean).
_ACTION_VALUES_COLUMNS = "match_id, start_x, start_y, end_x, end_y, action_type, action_result"
```

Add the two functions (after `load_matches`, before `_convert`):

```python
def fetch_action_values(*, max_matches: int | None = None) -> pd.DataFrame:
    """Read the gold action-values mart for the owner-gated xT held-out-NLL cross-check.

    Pulls only the columns the passes transition-NLL path needs. SPADL-id shaping is
    ``shape_action_values``. Read-only.

    Parameters
    ----------
    max_matches : int | None
        When set, restrict to the first ``max_matches`` distinct ``match_id`` (deterministic
        ``ORDER BY match_id`` — lexicographically biased, a smoke aid only). ``None`` reads all.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        if max_matches is not None:
            ids = [
                r[0]
                for r in _query_param(
                    cur,
                    f"SELECT DISTINCT match_id FROM {_ACTION_VALUES_TABLE} ORDER BY match_id LIMIT %(n)s",  # noqa: S608
                    {"n": int(max_matches)},
                ).itertuples(index=False)
            ]
            if not ids:
                return pd.DataFrame(columns=_ACTION_VALUES_COLUMNS.replace(" ", "").split(","))
            placeholders = ", ".join(f"%(m{i})s" for i in range(len(ids)))
            params = {f"m{i}": v for i, v in enumerate(ids)}
            sql = f"SELECT {_ACTION_VALUES_COLUMNS} FROM {_ACTION_VALUES_TABLE} WHERE match_id IN ({placeholders})"  # noqa: S608
            return _query_param(cur, sql, params)
        return _query_param(cur, f"SELECT {_ACTION_VALUES_COLUMNS} FROM {_ACTION_VALUES_TABLE}")  # noqa: S608
    finally:
        conn.close()


def shape_action_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map the gold action-values mart to the SPADL-id columns the xthreat NLL path expects.

    Pure, NaN-tolerant: ``action_type`` / ``action_result`` strings -> nullable-int ``type_id`` /
    ``result_id`` codes (unmapped -> <NA>; ``Int64`` is deliberate per ADR-019, avoiding a float id
    column — the caller drops the <NA> rows after its coverage guard so the ids reach the masks
    NA-free); ``match_id`` -> ``game_id`` (the holdout_split key). Coverage + NA-drop are the
    caller's job, not this function's.
    """
    import silly_kicks.spadl.config as spadlconfig  # function-local: keep module import cheap

    out = df.copy()
    out["type_id"] = out["action_type"].map(spadlconfig.actiontype_id).astype("Int64")
    out["result_id"] = out["action_result"].map(spadlconfig.result_id).astype("Int64")
    out["game_id"] = out["match_id"]
    return out
```

- [ ] **Step 4: Run shaping tests to verify they pass**

Run: `python -m pytest tests/calibration/test_loader_databricks.py -q -k shape_action_values`
Expected: 5 passed.

- [ ] **Step 5: Lint the changed files**

Run: `ruff check scripts/_loader_databricks.py tests/calibration/test_loader_databricks.py && ruff format --check scripts/_loader_databricks.py tests/calibration/test_loader_databricks.py`
Expected: no errors. (The `# noqa: S608` mirrors the file's existing parameterized-query pattern — table is a fixed constant, only `match_id` values are parameterized.)

---

## Task 2: Pure verdict helpers `nll_relative_win` + `kde_clears_tripwire`

**Files:**
- Modify: `tests/_xthreat_helpers.py`
- Test: `tests/test_xthreat_nll_relative_win.py` (create)

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_xthreat_nll_relative_win.py`:

```python
"""Regular-suite unit tests for the pure NLL verdict helpers used by the owner-gated e2e."""

import math

from tests._xthreat_helpers import kde_clears_tripwire, nll_relative_win


def test_relative_win_positive_when_candidate_lower():
    assert math.isclose(nll_relative_win(4.0, 3.8), 0.05, rel_tol=1e-9)  # 5% improvement (float-safe)


def test_relative_win_negative_when_candidate_higher():
    assert nll_relative_win(4.0, 4.2) < 0


def test_relative_win_exactly_at_floor():
    assert math.isclose(nll_relative_win(4.0, 3.94), 0.015, rel_tol=1e-9)


def test_relative_win_nan_when_baseline_nonfinite_or_zero():
    assert math.isnan(nll_relative_win(float("nan"), 3.8))
    assert math.isnan(nll_relative_win(0.0, 3.8))


def test_relative_win_nan_when_candidate_nan():
    # empty-corpus holdout -> compute_holdout_nll returns nan for the candidate
    assert math.isnan(nll_relative_win(4.0, float("nan")))


def test_tripwire_clears_well_above_floor():
    assert kde_clears_tripwire(4.0, 3.8, floor=0.015) is True  # 5% >> 1.5%


def test_tripwire_fails_just_below_floor():
    assert kde_clears_tripwire(4.0, 3.95, floor=0.015) is False  # 1.25% < 1.5%


def test_tripwire_true_exactly_at_floor():
    assert kde_clears_tripwire(4.0, 3.94, floor=0.015) is True  # 1.5% == floor


def test_tripwire_false_when_kde_loses():
    assert kde_clears_tripwire(4.0, 4.2, floor=0.015) is False


def test_tripwire_false_on_nan():
    assert kde_clears_tripwire(float("nan"), 3.8, floor=0.015) is False
    assert kde_clears_tripwire(4.0, float("nan"), floor=0.015) is False


def test_tripwire_strict_beat_with_zero_floor():
    # floor=0.0 == strict-beat (the shipped-default KDE(1.0) contract): any win clears, tie/loss fails.
    assert kde_clears_tripwire(4.0, 3.99, floor=0.0) is True
    assert kde_clears_tripwire(4.0, 4.0, floor=0.0) is False
    assert kde_clears_tripwire(4.0, 4.1, floor=0.0) is False
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_xthreat_nll_relative_win.py -q`
Expected: FAIL with `ImportError: cannot import name 'kde_clears_tripwire'` (or `nll_relative_win`).

- [ ] **Step 3: Implement both helpers**

Append to `tests/_xthreat_helpers.py` (the file already imports `numpy as np` at the top — reuse it, no new import):

```python
def nll_relative_win(baseline_nll: float, candidate_nll: float) -> float:
    """Relative held-out-NLL improvement of ``candidate`` over ``baseline``: ``(b - c) / b``.

    Positive == candidate is better (lower NLL). Returns ``nan`` if ``baseline`` is non-finite or
    zero (e.g. an empty-corpus ``compute_holdout_nll``), or if ``candidate`` is ``nan``.
    """
    if not np.isfinite(baseline_nll) or baseline_nll == 0:
        return float("nan")
    return (baseline_nll - candidate_nll) / baseline_nll


def kde_clears_tripwire(singh_nll: float, kde_nll: float, *, floor: float) -> bool:
    """The owner-gated tripwire predicate: KDE strictly beats Singh AND clears the relative floor.

    Pure + NaN-safe (a non-finite relative win -> ``False``). Unit-tested so a flipped comparison or
    wrong-direction floor is caught in CI, not only on the owner's mart.
    """
    rel = nll_relative_win(singh_nll, kde_nll)
    if not np.isfinite(rel):
        return False
    return bool(kde_nll < singh_nll and rel >= floor)
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/test_xthreat_nll_relative_win.py -q`
Expected: 11 passed.

- [ ] **Step 5: Lint**

Run: `ruff check tests/_xthreat_helpers.py tests/test_xthreat_nll_relative_win.py && ruff format --check tests/_xthreat_helpers.py tests/test_xthreat_nll_relative_win.py`
Expected: no errors.

---

## Task 3: The owner-gated e2e orchestrator

**Files:**
- Create: `tests/test_xthreat_nll_lakehouse_e2e.py`

- [ ] **Step 1: Write the e2e module**

Create `tests/test_xthreat_nll_lakehouse_e2e.py`:

```python
"""Owner-gated e2e: KDE-vs-Singh held-out transition-NLL on the real gold action-values mart.

Permanent, reproducible triangulation of the SK-xT-1 ~4% KDE win (4.17.0 ran it as a non-committed
one-off). Runs only where the owner Databricks credentials + databricks-sql-connector are reachable
(public CI skips). Thin orchestrator over unit-tested seams: scripts._loader_databricks (read+shape),
tests._xthreat_helpers.nll_relative_win (verdict). Scores PASSES-ONLY (matches the StatsBomb sibling
+ the lakehouse's published "Held-out NLL (passes)" 3.789->3.748). Hard asserts on the FULL corpus
only. See ADR-021 and docs/superpowers/specs/2026-06-09-xt-nll-lakehouse-e2e-design.md.
"""

import importlib.util
import os

import pytest

import scripts._loader_databricks as L
import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat import (
    GridSpec,
    KDEParams,
    compute_holdout_nll,
    holdout_split,
    kde_smoothed_transition_matrix,
    singh_transition_matrix,
)
from tests._xthreat_helpers import kde_clears_tripwire, nll_relative_win

_DBX_ENV = ("DATABRICKS_HOST", "DATABRICKS_HTTP_PATH", "DATABRICKS_TOKEN")


def _connector_available() -> bool:
    # find_spec("databricks.sql") imports the parent `databricks` to read its __path__, so it RAISES
    # ModuleNotFoundError (not returns None) when the connector is absent — guard it.
    try:
        return importlib.util.find_spec("databricks.sql") is not None
    except ModuleNotFoundError:
        return False


# Conservative floor for the KDE(4.0)-over-Singh relative held-out-NLL win. The 4.17.0 one-off
# measured ~4% at bandwidth>=4 on the full mart; this floor sits well below that so the tripwire
# tracks a real regression without flaking as the mart grows. See spec / ADR-021.
_MIN_RELATIVE_WIN = 0.015
_PROD_BANDWIDTH = 4.0  # held-out-optimal multiplier is >=4 on ~8.9M actions (full mart only)
_MIN_MAPPED = 0.95  # live-path coverage guard against mart-vocabulary drift

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in _DBX_ENV),
        reason="owner-tier Databricks credentials (DATABRICKS_HOST/HTTP_PATH/TOKEN)",
    ),
    pytest.mark.skipif(
        not _connector_available(),
        reason="databricks-sql-connector not importable (install in an isolated env, NOT the main .venv)",
    ),
]


def _nlls(train, holdout_passes, grid):
    singh = singh_transition_matrix(train, grid)
    kde4 = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=_PROD_BANDWIDTH))
    kde1 = kde_smoothed_transition_matrix(train, grid, KDEParams(bandwidth=1.0))
    return {
        "singh": compute_holdout_nll(singh, holdout_passes, grid=grid),
        "kde4": compute_holdout_nll(kde4, holdout_passes, grid=grid),
        "kde1": compute_holdout_nll(kde1, holdout_passes, grid=grid),
    }


def test_kde_beats_singh_on_holdout_nll_real_mart(capsys):
    max_matches = os.environ.get("XT_NLL_E2E_MAX_MATCHES")
    subsampled = max_matches is not None
    raw = L.fetch_action_values(max_matches=int(max_matches) if subsampled else None)
    assert len(raw) > 0, "gold mart returned no rows"

    actions = L.shape_action_values(raw)
    # Coverage guard: fail loud on mart-vocabulary drift instead of silently dropping every move.
    type_cov = actions["type_id"].notna().mean()
    result_cov = actions["result_id"].notna().mean()
    assert type_cov > _MIN_MAPPED, f"action_type vocab drift: only {type_cov:.1%} mapped to a SPADL type_id"
    assert result_cov > _MIN_MAPPED, f"action_result vocab drift: only {result_cov:.1%} mapped to a SPADL result_id"

    # Drop the <=5% unmapped rows so every downstream ==-mask is NA-free: a nullable-boolean <NA>
    # mask can raise ValueError on older pandas (the owner's <2.3.0 env); CI's 2.3.3 tolerates it,
    # so CI is blind to this path. Dropping here is correct regardless of the exact raising boundary.
    n_raw = len(actions)
    actions = actions.dropna(subset=["type_id", "result_id"])
    n_dropped = n_raw - len(actions)

    train, holdout = holdout_split(actions, holdout_fraction=0.15)
    # Score PASSES-ONLY (sibling + published-reference parity); fit on the full train.
    pass_id = spadlconfig.actiontype_id["pass"]
    holdout_passes = holdout[holdout["type_id"] == pass_id]
    assert len(train) > 0 and len(holdout_passes) > 0

    grid_default, grid_lakehouse = GridSpec(16, 12), GridSpec(12, 8)
    nll_d = _nlls(train, holdout_passes, grid_default)
    nll_l = _nlls(train, holdout_passes, grid_lakehouse)

    with capsys.disabled():
        print("\n=== xT held-out transition-NLL cross-check (gold mart; scored=passes,success) ===")
        print(f"n_actions={len(actions)} (dropped {n_dropped} unmapped)  "
              f"n_train_matches={train['game_id'].nunique()}  "
              f"n_holdout_pass={len(holdout_passes)}  subsampled={subsampled}")
        for label, d in (("16x12 (default)", nll_d), ("12x8 (lakehouse)", nll_l)):
            print(f"[{label}] singh={d['singh']:.5f}  "
                  f"kde@1.0={d['kde1']:.5f} ({nll_relative_win(d['singh'], d['kde1'])*100:+.2f}%)  "
                  f"kde@4.0={d['kde4']:.5f} ({nll_relative_win(d['singh'], d['kde4'])*100:+.2f}%)")
        print("Lakehouse published reference (12x8, passes): singh 3.78924 -> kde ~3.748")

    if subsampled:
        pytest.skip("XT_NLL_E2E_MAX_MATCHES set: bandwidth=4.0 is tuned for the full mart; "
                    "subsampled run is log-only (see printed block above).")

    # Hard tripwire — FULL corpus, 16x12, passes-only. (1) tuned KDE(4.0) clears the sensitivity
    # floor; (2) shipped-default KDE(1.0) STRICTLY beats Singh (no floor — its margin erodes with
    # corpus growth, so a floor would trip on benign growth; strict-beat catches a real regression).
    assert kde_clears_tripwire(nll_d["singh"], nll_d["kde4"], floor=_MIN_RELATIVE_WIN), (
        f"KDE(4.0) failed the tripwire at 16x12 (need strict-beat AND rel>={_MIN_RELATIVE_WIN}): {nll_d}"
    )
    assert kde_clears_tripwire(nll_d["singh"], nll_d["kde1"], floor=0.0), (
        f"shipped-default KDE(1.0) no longer strictly beats Singh at 16x12 (full mart): {nll_d}"
    )
```

> Note: `pytest.skip()` inside the test body (after logging) yields a clean "skipped" for subsampled smoke runs while still printing the numbers under `-s`. The hard asserts never run subsampled (C3).

- [ ] **Step 2: Lint**

Run: `ruff check tests/test_xthreat_nll_lakehouse_e2e.py && ruff format --check tests/test_xthreat_nll_lakehouse_e2e.py`
Expected: no errors (run `ruff format` on the file if the multi-line `print`s are flagged).

- [ ] **Step 3: Confirm collect-but-skip locally + import safety**

Run: `python -m pytest tests/test_xthreat_nll_lakehouse_e2e.py -q`
Expected: `1 skipped` (no DATABRICKS_* env) — NOT an error. Proves the module imports cleanly (incl. `import scripts._loader_databricks`, which must NOT pull the connector) and the gate works.

---

## Task 4: Housekeeping (staged, not committed)

**Files:** `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `uv.lock`

- [ ] **Step 1: Remove the completed TODO item**

The "Committed owner-gated lakehouse-mart NLL cross-check." bullet is the **sole** item under `### SK-xT-1 follow-ups (unblocked, ready)`. Verify boundaries with `sed -n '/SK-xT-1 follow-ups/,/^---/p' TODO.md`, then remove the whole subsection (header + the `Build directly on the 4.17.0 pluggable xT (...)` intro + the bullet), leaving the surrounding `---` separators intact.

- [ ] **Step 2: Bump the version to 4.21.2**

4.21.1 is tagged; 4.21.2 is free (the parallel session is holding). **Verify the number is still
free** before baking it into the bump/CHANGELOG/tag (PR-S88 collision precedent — cheap insurance):

Run: `git fetch --tags && git tag --list 'v4.21.2'`
Expected: empty output (still free). If non-empty, bump to the next free patch and substitute it
everywhere in this task + Task 6.

Then set `pyproject.toml` + `silly_kicks/__init__.py` (`__version__`) to `4.21.2`
(`grep -n version pyproject.toml` to confirm the current value before editing).

- [ ] **Step 3: Add the CHANGELOG entry**

Add above `## [4.21.1] — 2026-06-09`, matching the dated `### <Type> — <desc>` style:

```markdown
## [4.21.2] — 2026-06-09

### Added — owner-gated lakehouse-mart xT held-out-NLL cross-check

A permanent `@pytest.mark.e2e`, owner-gated regression tripwire
(`tests/test_xthreat_nll_lakehouse_e2e.py`) triangulating KDE-vs-Singh held-out transition-NLL on
**passes** against `soccer_analytics.dev_gold.fct_action_values` (the 4.17.0 work ran this as a
non-committed one-off; ~4% relative KDE win on ~8.9M actions). Fits on the full train, scores a
passes-only holdout (parity with the StatsBomb sibling + the published "Held-out NLL (passes)"
3.789→3.748), hard-asserts KDE(4.0) beats Singh at 16×12 with a conservative 1.5% relative-win
floor **on the full corpus only**, logs 12×8 + the shipped-default KDE(1.0). Adds the
`fetch_action_values` + pure `shape_action_values` mart helpers to `scripts/_loader_databricks.py`
(unit-tested) and pure `nll_relative_win` / `kde_clears_tripwire` verdict helpers (unit-tested).
Skips wherever the owner Databricks credentials + `databricks-sql-connector` are absent (public CI).
**No shipped-library change** — every artifact is in `scripts/` + `tests/`; the `silly_kicks/` wheel
is unchanged except `__version__`. Additive — no behavior change, no retrain trigger. (TODO SK-xT-1
follow-up; ADR-021.)
```

- [ ] **Step 4: Re-lock + verify the bump**

Run: `uv lock && grep -rn "4\.21\.2" pyproject.toml silly_kicks/__init__.py CHANGELOG.md && python -c "import silly_kicks; print(silly_kicks.__version__)"`
Expected: `4.21.2` everywhere; the import prints `4.21.2`; `uv.lock` updates the silly-kicks pin only.

---

## Task 5: Full-suite verification + final-review

**Files:** none (verification only).

- [ ] **Step 1: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e and not slow" -q 2>&1 | tail -5; echo "EXIT: ${PIPESTATUS[0]}"`
Expected: baseline count + 16 new passing (5 shaping + 11 verdict); 0 failures; EXIT 0. (Read the real summary line — don't narrate from memory.)

- [ ] **Step 2: Replicate the full CI lint job (whole tree)**

Run: `ruff check silly_kicks/ tests/ scripts/ && ruff format --check silly_kicks/ tests/ scripts/ && pyright silly_kicks/`
Expected: all clean. (Whole-tree `ruff format --check` per the 4.21.0 lesson.)

- [ ] **Step 3: Confirm e2e gating once more**

Run: `python -m pytest tests/test_xthreat_nll_lakehouse_e2e.py -q`
Expected: `1 skipped` — the tripwire never runs/breaks in the public path.

- [ ] **Step 4: Run `/final-review`**

Invoke the `/final-review` skill (mandatory pre-commit gate). Address any findings. Confirm C4 is unaffected (no new container/enumeration/backend/aggregator — this is a test + two helpers).

---

## Task 6: Single commit + PR (gated on explicit approval)

**Files:** all of the above.

- [ ] **Step 1: Present the diff + commit command and HOLD for explicit approval**

Run: `git status && git --no-pager diff --stat`
Present the staged set + the exact commit command. **Do not create the sentinel or commit without an explicit per-commit "yes"** (standing policy + the sentinel hook). Once approved (user creates/authorizes the sentinel at `C:\Users\Karsten\.claude-git-approval`):

- [ ] **Step 2: Stage + commit (single commit)**

Write the commit message to a temp file (apostrophe-safe per the multiline-commit lesson) and:

```bash
git add scripts/_loader_databricks.py tests/calibration/test_loader_databricks.py \
  tests/_xthreat_helpers.py tests/test_xthreat_nll_relative_win.py tests/test_xthreat_nll_lakehouse_e2e.py \
  TODO.md pyproject.toml silly_kicks/__init__.py CHANGELOG.md uv.lock \
  docs/superpowers/specs/2026-06-09-xt-nll-lakehouse-e2e-design.md \
  docs/superpowers/plans/2026-06-09-xt-nll-lakehouse-e2e.md
git commit -F .git/COMMIT_XT_NLL.txt
```

Commit subject (suggested): `test(xthreat): owner-gated lakehouse-mart KDE-vs-Singh held-out-NLL tripwire -- silly-kicks 4.21.2 (ADR-021)`. Body: summarize the helpers + the passes-only/full-corpus-only design + the review points addressed. End with the `Co-Authored-By` trailer.

- [ ] **Step 3: Push + open the PR** (bare commands, no pipes — per the sentinel-chaining gotcha)

Run: `git push origin sk-xt-nll-lakehouse-e2e`
Then `gh pr create --title "..." --body-file .git/PR_XT_NLL.md` (body to a file per the multiline lesson). Remove the temp files after.

- [ ] **Step 4: Merge + tag + publish** (per `reference_version_bump_checklist`)

Squash `--admin` merge; push the annotated `v4.21.2` tag (triggers `publish.yml` → PyPI). Confirm PyPI shows 4.21.2 (cache-bust the JSON API).

---

## Self-Review notes (v3)

- **Review round 1 coverage:** C1 passes-only (Task 3 + spec). C2 coverage guard (Task 3). C3 full-corpus-only asserts + subsample skip + log-only kde1 (Task 3). H1 shaping-in-loader + `import scripts._loader_databricks` + shaping unit-tested in the calibration test, no importlib/cross-import (Tasks 1, 3). H2 verdict unit-tested incl. NaN-empty (Task 2). Process: single approved commit (Task 6), `/final-review` (Task 5 Step 4). L items: period/action_id dropped (Task 1), Int64 ids (Task 1), subsample-bias documented (docstring + spec).
- **Review round 2 coverage:** HIGH `<NA>`-drop after the guard, before any `==`-mask (Task 3 Step 1 — `dropna`). M2 `kde_clears_tripwire` unit-tested (Task 2) + used in the lone hard assert (Task 3). M3 `spadlconfig` function-local (Task 1 Step 3). M1 wheel-identical CHANGELOG note (Task 4 Step 3). L2 `test_xthreat_nll_relative_win.py` rename (Tasks 2, 6).
- **Type/name consistency:** `fetch_action_values` / `shape_action_values` / `nll_relative_win` / `kde_clears_tripwire` / `_MIN_RELATIVE_WIN` / `_PROD_BANDWIDTH` / `_MIN_MAPPED` used identically across tasks. `compute_holdout_nll(matrix, holdout_passes, grid=grid)` matches `_eval.py`; the `holdout_passes` filter + NA-free ids match the sibling and the owner-pandas constraint.
- **No placeholders:** every code + command step is concrete (the only deliberate deferral is the owner-run promotion of the log-only kde1 check, explicitly flagged).
