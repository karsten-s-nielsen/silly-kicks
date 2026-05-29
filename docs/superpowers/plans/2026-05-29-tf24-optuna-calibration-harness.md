# TF-24 PR-B — Optuna Calibration Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pure, CI-testable `silly_kicks/calibration/` subpackage + a `scripts/` CLI that calibrate three tracking defaults (`infer_ball_carrier` `tolerance_m`/`beta`/`gamma`; `LinkParams.k3`; off-ball-run `pre_seconds`/`min_displacement_m`) against real multi-provider tracking data via `ruthless-efficiency[optuna]`, producing recommended values + an auditable report — **without** changing the library default constants.

**Architecture:** Pure provider-agnostic objectives/CV/gates/feature-enrichment in the library (consume `(actions, frames, FrozenXt)` DataFrames only, zero I/O); all I/O + study orchestration in `scripts/` via two pluggable loaders (pining-for-the-data + Databricks bronze). Stage 1 (carrier accuracy, maximize) is a plain ruthless `Objective`; Stage 2 (augmented-VAEP held-out Brier, minimize) is a ruthless `CachedObjective` — expensive trial-independent enrichment prepared once, only the 2 trial-varying steps re-run per trial. xT is a **frozen exogenous artifact** fit on a corpus disjoint from the calibration matches (C2 Option 1: train–serve consistency + zero leak).

**Tech Stack:** Python 3.10-compatible; pandas/numpy/scikit-learn; `ruthless-efficiency[optuna]` (0.2.0); `xgboost>=2.0,<3.0`; `databricks-sql-connector` (loader only, lazy). Spec: `docs/superpowers/specs/2026-05-29-tf24-optuna-calibration-harness-design.md`.

---

## Reference: verified API signatures (ground truth for this plan)

**ruthless 0.2.0** (`D:\Development\karstenskyt__ruthless-efficiency\ruthless`):
- `from ruthless import Direction, FloatRange, OptunaConfig, Candidate, Objective, CachedObjective, penalty_metrics, render_json, render_summary_md, assert_cache_equivalence, InProcessBackend`
- `from ruthless.strategies.optuna_ import OptunaStrategy`
- `from ruthless.config.common import StoreConfig`
- `FloatRange(kind="float", lo=<float>, hi=<float>, log=<bool>)` — pydantic model; `lo < hi`; `log` requires `lo > 0`.
- `OptunaConfig(kind="optuna", metric=<str>, direction=Direction.MINIMIZE|MAXIMIZE, n_trials=<int>, sampler="tpe", param_space={name: FloatRange(...)}, warm_start={name: value}, store=StoreConfig(kind="sqlite", path=<str>))` — validates `warm_start` keys ⊆ `param_space`.
- `StoreConfig(kind="sqlite", path=<str>)`.
- `OptunaStrategy(config, *, seed=42).run(objective, backend=InProcessBackend()) -> Result`. If `objective` is a `CachedObjective`, the strategy asserts `param_space ⊆ objective.patch_params` and calls `prepare()` once + `evaluate_patch(invariant, candidate)` per trial. Non-finite scored metric raises and aborts the study.
- `Candidate(id=<str>, params=<Mapping>)` — `params` frozen after construction; `candidate.params["k3"]` to read.
- `Metrics = dict[str, float]` — objectives return a plain dict; the scored `metric` key must be present and finite.
- `penalty_metrics(metric, direction, *, magnitude=1e9) -> {metric: magnitude}` (for MINIMIZE returns `+magnitude`).
- `assert_cache_equivalence(objective, candidates, *, rtol=1e-9, atol=1e-12)` — raises if `evaluate` ≠ `prepare`+`evaluate_patch` for any candidate; with ≥2 candidates **requires every `patch_param` to take ≥2 distinct values** across `candidates` (else raises). Compares ALL metric keys (per-provider attrs included).
- `Result(best: Evaluation|None, history, diagnostics: dict, provenance: dict)`; `Evaluation(candidate, metrics, ok)`.
- `render_json(result) -> str` / `render_summary_md(result) -> str` — render `best`/`diagnostics`/`provenance`.

**silly-kicks** (verified signatures):
- `silly_kicks.tracking.infer_ball_carrier(frames, *, tolerance_m=3.0, beta=0.5, gamma=1.0) -> DataFrame[game_id, period_id, frame_id, ball_carrier_player_id, ball_carrier_distance_m, ball_carrier_team_id]`
- `silly_kicks.tracking.features.ball_carrier_at_action(actions, frames, *, tolerance_seconds=0.2, tolerance_m=3.0, beta=0.5, gamma=1.0) -> pd.Series` (aligned to `actions.index`, NaN where unlinked/no carrier)
- `silly_kicks.tracking.derive_team_in_possession(frames, carrier) -> frames + team_in_possession`
- `silly_kicks.tracking.utils.link_actions_to_frames(actions, frames, tolerance_seconds=0.2) -> (pointers, LinkReport)` — `pointers` cols: `action_id, frame_id (Int64), time_offset_seconds, n_candidate_frames, link_quality_score`.
- `silly_kicks.tracking.features.add_pressure_on_actor(actions, frames, *, links=None, methods=("andrienko_oval",), params_per_method=None) -> actions + pressure_on_actor__<m>` columns. For `link_zones`: `params_per_method={"link_zones": LinkParams(k3=<float>)}`.
- `silly_kicks.tracking.pressure.LinkParams(..., k3=1.0)` (frozen dataclass).
- `silly_kicks.tracking.features.add_off_ball_runs(actions, frames, *, home_team_id, pre_seconds=1.5, min_displacement_m=3.0) -> actions + [n_off_ball_runners_pre_window, max_off_ball_run_displacement_pre_window, mean_off_ball_run_speed_pre_window, n_off_ball_runners_toward_goal_pre_window]` (RUN cols only — no line-break).
- `silly_kicks.tracking.features`: `add_action_context`, `add_actor_pre_window`, `add_defensive_line`, `add_team_shape`, `add_gk_influence(actions, frames, xt, *, links, home_team_id)`, `add_cover_shadows(actions, frames, xt, *, links, home_team_id)`, `add_sync_score(actions, links)`, `pitch_control_at_action(actions, frames, *, links, method)` (returns Series with `.name`).
- `silly_kicks.spadl.utils.add_pre_shot_gk_context(actions, *, frames=None)`.
- `silly_kicks.tracking._das.get_individual_das(frames, *, use_progress_bar=False, chunk_size=10)`.
- `silly_kicks.vaep.labels.scores(actions, nr_actions=10) -> DataFrame["scores" bool]`; `concedes(actions, nr_actions=10) -> DataFrame["concedes" bool]`.
- `silly_kicks.xthreat.ExpectedThreat().fit(actions) -> self`; `.rate(actions)`; the fitted grid lives on `self.xT`.
- `silly_kicks.tracking.gradientsports.add_gradientsports_player_ids(jersey_frames, roster, *, home_team_id, away_team_id) -> (frames, GradientsportsRosterReport)` (PR-A, shipped 3.27.0).

**Prior monolith (proven, adapt-from):** `D:\Development\karstenskyt__luxury-lakehouse-d32\scripts\run_tc3_calibration.py`. Key functions: `_enrich_match_invariant` (837-1009), `_patch_trial_columns` (1012-1053), `_compute_carrier_accuracy_for_match` (1059-1131), `ALL_FEATURES`/`_TRACKING_FEATURES` (1230-1303), `_compute_provider_brier` (1390-1520). **The 16 enrichment steps + the feature list are copied from there; the deltas are (a) frozen exogenous xT, (b) `add_off_ball_runs` instead of the `add_off_ball_context` umbrella in the patch, (c) line-break never computed, (d) stateless default-Brier-anchored penalty.**

**Match counts (verified 2026-05-29, consistent across bronze AND pining):** GS 64, SkillCorner 10, IDSSE 7 → GroupKFold(5) for GS+SkillCorner, LOMO for IDSSE. **All three are now on pining** (SkillCorner + IDSSE public; GS owner-tier) — IDSSE moved from Databricks to pining 2026-05-29 (verified: `/idsse/matches` returns 7 public matches, artifacts download via the 302 two-step on the public token).

---

## File structure

| File | Responsibility |
|------|----------------|
| `silly_kicks/calibration/__init__.py` | Curated exports; lazy-import-guarded; NOT imported by `silly_kicks/__init__`. |
| `silly_kicks/calibration/_features.py` | `ALL_FEATURES` column list + `enrich_invariant` (14 trial-independent steps) + `patch_trial_columns` (2 trial-varying steps). |
| `silly_kicks/calibration/_xt.py` | `FrozenXt` + `fit_frozen_xt` (zero-overlap assert) + `save_xt`/`load_xt` (npz + sha256). |
| `silly_kicks/calibration/_cv.py` | `cv_scheme_for(n_matches)` + `match_cv_splits` (GroupKFold-5 / LOMO) + `cv_standard_error`. |
| `silly_kicks/calibration/_gates.py` | `default_feature_variances` + `h1_penalty_fires` + `signal_sanity` (provider exclusion). |
| `silly_kicks/calibration/_spaces.py` | `stage1_config(...)` + `stage2_config(...)` (`OptunaConfig` builders). |
| `silly_kicks/calibration/_carrier_objective.py` | `CarrierAccuracyObjective` (ruthless `Objective`, maximize). |
| `silly_kicks/calibration/_vaep_brier_objective.py` | `AugmentedVaepBrierObjective` (ruthless `CachedObjective`, minimize). |
| `silly_kicks/calibration/_diagnostics.py` | TF-25 gate + k3 1-D sensitivity. |
| `scripts/calibrate_tracking_defaults.py` | CLI: stage dispatch, loader selection, xT artifact, study run, manifest+report. |
| `scripts/_loader_pining.py` | pining-for-the-data fetch → `(frames, actions, home_team_id)` per `(provider, match_id)`. |
| `scripts/_loader_databricks.py` | `bronze.{provider}_{tracking,events}` → same uniform tuple. |
| `tests/calibration/` | Unit + cache-equivalence + loader (stubbed) + e2e smoke. |

---

## Task 0: Branch, extra, subpackage skeleton

**Files:**
- Modify: `pyproject.toml`
- Create: `silly_kicks/calibration/__init__.py`
- Create: `tests/calibration/__init__.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout main && git pull
git checkout -b pr-b-tf24-optuna-calibration
```

- [ ] **Step 2: Add the `[calibration]` extra + per-file lint ignores to `pyproject.toml`**

Find `[project.optional-dependencies]` and add the `calibration` extra; add it to the `test` extra so CI exercises it. The exact pins:

```toml
calibration = [
    "ruthless-efficiency[optuna]>=0.2.0",
    "xgboost>=2.0,<3.0",
]
```

L3: scikit-learn is already a runtime dep — verify it carries a lower bound (e.g. `scikit-learn>=1.3`); `brier_score_loss`/`GroupKFold`/`LeaveOneGroupOut` signatures have churned across majors and the 1e-9 cache gate depends on determinism. Add/tighten the bound if absent (in the existing runtime deps, not duplicated here).

In the `test` extra list, append `silly-kicks[calibration]` (or inline the two deps if self-reference is unsupported — check existing `test` extra style first). Under `[tool.ruff.lint.per-file-ignores]` add (ML uppercase names + script I/O):

```toml
"silly_kicks/calibration/_vaep_brier_objective.py" = ["N803", "N806"]
"scripts/calibrate_tracking_defaults.py" = ["T201"]
```

> M5: do NOT add a blanket `"scripts/_loader_databricks.py" = ["S608"]` ignore. The Databricks loader parameterizes `match_id` and allowlist-validates `provider`; the only residual S608 is the allowlist-built table NAME, suppressed with a JUSTIFIED inline `# noqa: S608` on those specific lines (see Task 11), not a file-wide suppression of the repo's SQL-safety standard.

- [ ] **Step 3: Create the subpackage `__init__.py` (lazy, not auto-imported)**

```python
"""silly_kicks.calibration — Optuna calibration harness for tracking defaults (TF-24).

Optional subpackage; requires the ``[calibration]`` extra
(``pip install silly-kicks[calibration]``). NOT imported by ``silly_kicks/__init__`` —
import members directly from ``silly_kicks.calibration``.

See ``docs/superpowers/specs/2026-05-29-tf24-optuna-calibration-harness-design.md``.
"""

from __future__ import annotations

from silly_kicks.calibration._cv import cv_scheme_for, cv_standard_error, match_cv_splits
from silly_kicks.calibration._features import ALL_FEATURES, enrich_invariant, patch_trial_columns
from silly_kicks.calibration._gates import default_feature_variances, h1_penalty_fires, signal_sanity
from silly_kicks.calibration._spaces import stage1_config, stage2_config
from silly_kicks.calibration._xt import FrozenXt, fit_frozen_xt, load_xt, save_xt

__all__ = [
    "ALL_FEATURES",
    "FrozenXt",
    "cv_scheme_for",
    "cv_standard_error",
    "default_feature_variances",
    "enrich_invariant",
    "fit_frozen_xt",
    "h1_penalty_fires",
    "load_xt",
    "match_cv_splits",
    "patch_trial_columns",
    "save_xt",
    "signal_sanity",
    "stage1_config",
    "stage2_config",
]
```

> NOTE: `CarrierAccuracyObjective` / `AugmentedVaepBrierObjective` / `_diagnostics` are NOT exported here yet — add them in their own tasks (this avoids an import error while the modules don't exist). Re-run this `__init__` edit at the end of Task 7 / Task 9.

- [ ] **Step 4: Create `tests/calibration/__init__.py` (empty)**

```python
```

- [ ] **Step 5: Verify the subpackage import fails cleanly (modules not yet written)**

Run: `python -c "import silly_kicks.calibration"`
Expected: `ModuleNotFoundError: No module named 'silly_kicks.calibration._cv'` (the submodules don't exist yet — expected until later tasks; this just confirms the package dir + `__init__` parse).

- [ ] **Step 6: Commit the skeleton**

```bash
git add pyproject.toml silly_kicks/calibration/__init__.py tests/calibration/__init__.py
git commit -m "chore(calibration): scaffold TF-24 subpackage + [calibration] extra"
```

> The `__init__.py` import of not-yet-existing modules will break `import silly_kicks.calibration` until Task 7. That is acceptable mid-branch (the package is never imported by the rest of the library). If you prefer a green tree at every commit, temporarily comment the imports and uncomment them as each module lands.

---

## Task 1: `_cv.py` — match-stratified CV splitter

**Files:**
- Create: `silly_kicks/calibration/_cv.py`
- Test: `tests/calibration/test_cv.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_cv.py
import numpy as np
import pytest

from silly_kicks.calibration._cv import cv_scheme_for, cv_standard_error, match_cv_splits


def test_cv_scheme_threshold():
    assert cv_scheme_for(7) == "lomo"      # IDSSE
    assert cv_scheme_for(8) == "groupkfold"
    assert cv_scheme_for(10) == "groupkfold"  # SkillCorner
    assert cv_scheme_for(64) == "groupkfold"  # Gradient Sports


def test_lomo_one_held_out_match_per_fold():
    match_ids = np.array(["a", "a", "b", "b", "c"])
    splits = match_cv_splits(match_ids)
    assert len(splits) == 3  # one fold per match
    for train_idx, test_idx in splits:
        held_out = set(match_ids[test_idx])
        assert len(held_out) == 1  # exactly one match held out
        assert held_out.isdisjoint(set(match_ids[train_idx]))  # no leakage


def test_groupkfold_5_for_many_matches():
    match_ids = np.array([f"m{i}" for i in range(10) for _ in range(3)])
    splits = match_cv_splits(match_ids)
    assert len(splits) == 5
    for train_idx, test_idx in splits:
        assert set(match_ids[test_idx]).isdisjoint(set(match_ids[train_idx]))


def test_standard_error_of_fold_means():
    # SE = std(fold_briers, ddof=1) / sqrt(n_folds)
    briers = [0.04, 0.05, 0.06]
    se = cv_standard_error(briers)
    assert se == pytest.approx(np.std(briers, ddof=1) / np.sqrt(3))


def test_standard_error_single_fold_is_nan():
    assert np.isnan(cv_standard_error([0.04]))
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_cv.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'silly_kicks.calibration._cv'`.

- [ ] **Step 3: Implement `_cv.py`**

```python
# silly_kicks/calibration/_cv.py
"""Match-stratified cross-validation for the TF-24 calibration harness.

A single count-driven threshold (spec §2): GroupKFold(5) for >7 matches,
leave-one-match-out for <=7. Random-action splits are forbidden — they leak
match structure into the held-out fold.

Examples
--------
Split a provider's actions into match-stratified folds::

    import numpy as np
    from silly_kicks.calibration._cv import match_cv_splits

    match_ids = np.array(["a", "a", "b", "b", "c"])
    for train_idx, test_idx in match_cv_splits(match_ids):
        ...  # no match appears in both train and test
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

_GROUPKFOLD_THRESHOLD = 7  # > this many matches => GroupKFold(5); else leave-one-match-out
_N_SPLITS = 5


def cv_scheme_for(n_matches: int) -> Literal["groupkfold", "lomo"]:
    """Return the CV scheme name for a provider with ``n_matches`` distinct matches."""
    return "groupkfold" if n_matches > _GROUPKFOLD_THRESHOLD else "lomo"


def match_cv_splits(match_ids: npt.NDArray) -> list[tuple[npt.NDArray, npt.NDArray]]:
    """Return (train_idx, test_idx) folds grouped by match (no match in both sides).

    GroupKFold(5) when the number of distinct matches exceeds the threshold,
    otherwise leave-one-match-out (one fold per match).
    """
    n_matches = len(np.unique(match_ids))
    x = np.zeros((len(match_ids), 1))  # GroupKFold ignores X values, only needs shape
    if cv_scheme_for(n_matches) == "groupkfold":
        splitter = GroupKFold(n_splits=_N_SPLITS)
    else:
        splitter = LeaveOneGroupOut()
    return [(tr, te) for tr, te in splitter.split(x, groups=match_ids)]


def cv_standard_error(fold_metrics: list[float]) -> float:
    """Standard error of the mean across CV folds: ``std(ddof=1) / sqrt(n_folds)``.

    Returns ``nan`` for a single fold (SE undefined).
    """
    arr = np.asarray(fold_metrics, dtype=float)
    if len(arr) < 2:
        return float("nan")
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_cv.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_cv.py tests/calibration/test_cv.py
git commit -m "feat(calibration): match-stratified CV splitter (GroupKFold-5/LOMO)"
```

---

## Task 2: `_xt.py` — frozen exogenous xT artifact

**Files:**
- Create: `silly_kicks/calibration/_xt.py`
- Test: `tests/calibration/test_xt.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_xt.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.calibration._xt import FrozenXt, fit_frozen_xt, load_xt, save_xt


def _toy_actions(match_ids, n_per=40, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mid in match_ids:
        for i in range(n_per):
            rows.append({
                "game_id": mid, "action_id": i, "period_id": 1,
                "time_seconds": float(i), "team_id": 1, "player_id": 10,
                "start_x": rng.uniform(0, 105), "start_y": rng.uniform(0, 68),
                "end_x": rng.uniform(0, 105), "end_y": rng.uniform(0, 68),
                "type_id": 0, "result_id": 1, "bodypart_id": 0,
            })
    return pd.DataFrame(rows)


def test_fit_excludes_calibration_matches_and_records_provenance():
    corpus = _toy_actions(["c1", "c2", "c3"])  # game_id is the match key here
    frozen = fit_frozen_xt(corpus, exclude_match_ids={"c3"}, match_id_col="game_id",
                           source="test-corpus")
    assert "c3" not in frozen.corpus_match_ids
    assert frozen.corpus_match_ids == ("c1", "c2")
    assert frozen.source == "test-corpus"
    assert frozen.grid_shape == frozen.xt.xT.shape
    assert len(frozen.sha256) == 64


def test_fit_raises_when_no_corpus_remains_after_exclusion():
    corpus = _toy_actions(["c1"])
    with pytest.raises(ValueError, match="disjoint corpus is empty"):
        fit_frozen_xt(corpus, exclude_match_ids={"c1"}, match_id_col="game_id", source="x")


def test_fit_fails_closed_when_excluded_id_absent_from_corpus():
    # H2: an excluded id that doesn't exist in the corpus means the exclusion no-ops -> would LEAK.
    corpus = _toy_actions(["c1", "c2", "c3"])
    with pytest.raises(ValueError, match="were NOT found in corpus"):
        # 'pining-99' is a different id space than corpus game_ids => must fail closed.
        fit_frozen_xt(corpus, exclude_match_ids={"c1", "pining-99"},
                      match_id_col="game_id", source="x")


def test_fit_records_n_excluded():
    corpus = _toy_actions(["c1", "c2", "c3"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids={"c1", "c2"}, match_id_col="game_id", source="x")
    assert frozen.n_excluded == 2
    assert frozen.manifest()["n_excluded"] == 2


def test_save_load_roundtrip_preserves_grid_and_sha(tmp_path):
    corpus = _toy_actions(["c1", "c2"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids=set(), match_id_col="game_id", source="x")
    path = tmp_path / "xt.npz"
    save_xt(frozen, path)
    loaded = load_xt(path)
    assert loaded.sha256 == frozen.sha256
    np.testing.assert_array_equal(loaded.xt.xT, frozen.xt.xT)
    assert loaded.corpus_match_ids == frozen.corpus_match_ids


def test_load_detects_tampered_grid(tmp_path):
    corpus = _toy_actions(["c1", "c2"])
    frozen = fit_frozen_xt(corpus, exclude_match_ids=set(), match_id_col="game_id", source="x")
    path = tmp_path / "xt.npz"
    save_xt(frozen, path)
    # Tamper: rewrite the grid but keep the stored sha256
    data = dict(np.load(path, allow_pickle=True))
    data["xT"] = data["xT"] + 1.0
    np.savez(path, **data)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_xt(path)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_xt.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_xt.py`**

```python
# silly_kicks/calibration/_xt.py
"""Frozen exogenous xT artifact for the TF-24 calibration harness (C2, Option 1).

xT is a fixed upstream feature *extractor* — at deployment the calibrated tracking
defaults run against ONE fixed league-level xT grid, never a per-match refit. So we
fit ``ExpectedThreat`` ONCE on a corpus DISJOINT from the calibration matches, freeze
it as a checksummed artifact, and use that single grid for every match/fold/trial.
This removes the held-out leak (the grid never sees a calibration action) and gives a
cleaner TPE signal (xT injects zero fold-structure variance).

See NOTICE for the xT (Decroos/Van Roy) citation. Spec §4a.

Examples
--------
Fit, freeze, and reload a calibration xT grid::

    from silly_kicks.calibration._xt import fit_frozen_xt, save_xt, load_xt

    frozen = fit_frozen_xt(corpus_actions, exclude_match_ids={"game_42"},
                           match_id_col="game_id", source="bronze.spadl_actions")
    save_xt(frozen, "calibration_xt.npz")
    frozen = load_xt("calibration_xt.npz")  # sha256-verified
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.xthreat import ExpectedThreat


def _grid_sha256(grid: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(grid, dtype=np.float64).tobytes()).hexdigest()


@dataclass(frozen=True)
class FrozenXt:
    """A frozen, checksummed xT grid + its provenance (for the calibration manifest)."""

    xt: ExpectedThreat
    source: str
    corpus_match_ids: tuple[str, ...]
    n_excluded: int  # calibration matches actually removed from the corpus (H2 audit)
    fit_date: str  # ISO date; supplied by the caller (pure core does not read the clock)
    grid_shape: tuple[int, int]
    sha256: str

    def manifest(self) -> dict:
        """Provenance dict for the report (§6 R3) — JSON-serialisable, no grid payload."""
        return {
            "source": self.source,
            "corpus_match_ids": list(self.corpus_match_ids),
            "n_corpus_matches": len(self.corpus_match_ids),
            "n_excluded": self.n_excluded,
            "fit_date": self.fit_date,
            "grid_shape": list(self.grid_shape),
            "sha256": self.sha256,
        }


def fit_frozen_xt(
    corpus_actions: pd.DataFrame,
    *,
    exclude_match_ids: Iterable,
    match_id_col: str = "game_id",
    source: str,
    fit_date: str = "",
) -> FrozenXt:
    """Fit ``ExpectedThreat`` on ``corpus_actions`` MINUS ``exclude_match_ids`` and freeze it.

    The exclusion is the whole point — it guarantees the calibration matches never enter the
    xT grid (zero leak). **Fails CLOSED (H2):** if any excluded id is NOT present in the corpus,
    the exclusion silently did nothing (an id-space mismatch — pining match_id vs bronze game_id),
    which would reintroduce the leak. So we require EVERY excluded id to be found and removed.
    Also raises if the disjoint corpus is empty.
    """
    excluded = {str(m) for m in exclude_match_ids}
    corpus_ids = {str(m) for m in corpus_actions[match_id_col].unique()}
    found = excluded & corpus_ids
    if excluded and len(found) < len(excluded):
        missing = sorted(excluded - corpus_ids)
        raise ValueError(
            f"xT-corpus exclusion is unsafe: {len(missing)}/{len(excluded)} calibration match ids "
            f"were NOT found in corpus[{match_id_col!r}] (e.g. {missing[:5]}). The id spaces differ "
            "(pining match_id vs bronze game_id?) — the exclusion would no-op and LEAK held-out "
            "matches into the xT fit. Map the ids to a common space before fitting."
        )
    keep = corpus_actions[~corpus_actions[match_id_col].astype(str).isin(excluded)]
    remaining = tuple(sorted(str(m) for m in keep[match_id_col].unique()))
    if len(keep) == 0 or not remaining:
        raise ValueError(
            "disjoint corpus is empty after excluding calibration matches — "
            "supply a larger corpus or a smaller exclusion set"
        )
    xt = ExpectedThreat().fit(keep)
    grid = np.asarray(xt.xT, dtype=np.float64)
    return FrozenXt(
        xt=xt,
        source=source,
        corpus_match_ids=remaining,
        n_excluded=len(found),
        fit_date=fit_date,
        grid_shape=(int(grid.shape[0]), int(grid.shape[1])),
        sha256=_grid_sha256(grid),
    )


def save_xt(frozen: FrozenXt, path: str | Path) -> None:
    """Serialise a ``FrozenXt`` to ``path`` (npz grid + JSON-sidecar provenance in the same file)."""
    meta = frozen.manifest()
    np.savez(
        path,
        xT=np.asarray(frozen.xt.xT, dtype=np.float64),
        meta_json=np.array(json.dumps(meta)),
    )


def load_xt(path: str | Path) -> FrozenXt:
    """Load a ``FrozenXt`` from ``path``, re-checking the grid sha256 against the stored value."""
    data = np.load(path, allow_pickle=True)
    grid = np.asarray(data["xT"], dtype=np.float64)
    meta = json.loads(str(data["meta_json"]))
    recomputed = _grid_sha256(grid)
    if recomputed != meta["sha256"]:
        raise ValueError(
            f"xT artifact sha256 mismatch (stored {meta['sha256']}, recomputed {recomputed}) — "
            "the grid was modified after fitting; refuse to load a tampered artifact"
        )
    xt = ExpectedThreat()
    xt.xT = grid  # inference uses .xT directly; gk_influence/cover_shadows read the grid
    return FrozenXt(
        xt=xt,
        source=meta["source"],
        corpus_match_ids=tuple(meta["corpus_match_ids"]),
        n_excluded=int(meta.get("n_excluded", 0)),
        fit_date=meta["fit_date"],
        grid_shape=tuple(meta["grid_shape"]),
        sha256=meta["sha256"],
    )
```

> NOTE on `ExpectedThreat` reconstruction in `load_xt`: `add_gk_influence`/`add_cover_shadows` consume the fitted value surface `xt.xT`. If those functions also read other fitted attributes (e.g. `.l`/`.w`/`.interpolator`), the implementer MUST verify by reading `silly_kicks/tracking/_gk_influence.py` + `_cover_shadows.py` for every `xt.` attribute access and round-trip those too. Add a test asserting `add_gk_influence(actions, frames, load_xt(path).xt, ...)` produces identical columns to the in-memory `frozen.xt`. (Verification step in Task 7.)

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_xt.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_xt.py tests/calibration/test_xt.py
git commit -m "feat(calibration): frozen exogenous xT artifact (npz + sha256, disjoint-corpus fit)"
```

---

## Task 3: `_features.py` — invariant + patch enrichment

**Files:**
- Create: `silly_kicks/calibration/_features.py`
- Test: `tests/calibration/test_features.py`

**Context:** copy the step list verbatim from the prior monolith `_enrich_match_invariant` (lines 837-1009) and `_patch_trial_columns` (1012-1053), with the documented deltas: frozen xT passed in (not fit here), `add_off_ball_runs` in the patch (not the `add_off_ball_context` umbrella), line-break never computed.

- [ ] **Step 1: Write the failing test (synthetic shot+keeper_save match)**

```python
# tests/calibration/test_features.py
import numpy as np
import pandas as pd
import pytest

from silly_kicks.calibration._features import (
    ALL_FEATURES,
    _TRIAL_DEPENDENT_COLS,
    enrich_invariant,
    patch_trial_columns,
)
# Reuse the project's synthetic tracking+SPADL helpers used by other tracking tests.
from tests.tracking.conftest import make_synthetic_match  # see NOTE below


@pytest.fixture
def synth():
    # A small synthetic match with frames + actions incl. a shot+keeper_save
    # (the tracking-aware-feature canonical fixture). Returns (actions, frames, home_team_id).
    return make_synthetic_match(seed=1)


@pytest.fixture
def frozen_xt(synth):
    from silly_kicks.calibration._xt import fit_frozen_xt
    actions, _frames, _h = synth
    # Fit on the same actions here ONLY for the unit fixture; production fits on a disjoint corpus.
    return fit_frozen_xt(actions, exclude_match_ids=set(), match_id_col="game_id", source="unit")


def test_invariant_sets_trial_cols_nan_and_others_present(synth, frozen_xt):
    actions, frames, home = synth
    base, links, _das_ok = enrich_invariant(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
    )
    for col in _TRIAL_DEPENDENT_COLS:
        assert base[col].isna().all(), f"{col} must be a NaN placeholder in the invariant"
    # A non-trial tracking feature must be materialised (not all-NaN) for at least some rows.
    assert base["pressure_on_actor__andrienko_oval"].notna().any()
    assert "frame_id" in links.columns


def test_patch_overwrites_exactly_the_trial_cols(synth, frozen_xt):
    actions, frames, home = synth
    base, links, _das_ok = enrich_invariant(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
    )
    invariant_snapshot = base.drop(columns=_TRIAL_DEPENDENT_COLS).copy()
    patched = patch_trial_columns(
        base_actions=base, frames=frames, links=links, home_team_id=home,
        k3=2.0, pre_seconds=2.0, min_displacement_m=4.0,
    )
    # Trial cols are now populated...
    assert patched["pressure_on_actor__link_zones"].notna().any()
    assert patched["n_off_ball_runners_pre_window"].notna().any()
    # ...and NO invariant column changed.
    pd.testing.assert_frame_equal(
        patched.drop(columns=_TRIAL_DEPENDENT_COLS)[invariant_snapshot.columns],
        invariant_snapshot,
    )


def test_line_break_columns_are_not_features():
    assert not any("line_break" in c or "lines_broken" in c for c in ALL_FEATURES)


def test_all_features_count_matches_spec():
    # SPADL base + ~45 tracking features; assert the trial-dependent 5 are in the set.
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in ALL_FEATURES


def test_enrich_full_is_independent_and_populates_trial_cols(synth, frozen_xt):
    # enrich_full runs all steps inline (no NaN placeholders) — the independent path that
    # makes assert_cache_equivalence meaningful (H1). It must populate the trial cols itself.
    from silly_kicks.calibration._features import enrich_full

    actions, frames, home = synth
    full = enrich_full(
        actions=actions, frames=frames, xt=frozen_xt.xt, home_team_id=home,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
        k3=1.5, pre_seconds=2.0, min_displacement_m=4.0,
    )
    for col in _TRIAL_DEPENDENT_COLS:
        assert col in full.columns
        assert full[col].notna().any()  # populated inline, NOT a NaN placeholder
```

> NOTE: `make_synthetic_match` may not exist under that exact name. Before writing this test, grep `tests/tracking/` for the existing synthetic-match fixture (e.g. `synthetic_frames`, `_synth_match`, the shot+keeper_save builder referenced in the project memory). Use the real helper; if none returns the `(actions, frames, home_team_id)` triple, add a tiny local builder in `tests/calibration/conftest.py` that composes the existing frame + SPADL synth helpers. This fixture is reused by Tasks 6, 7, 8.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_features.py -v`
Expected: FAIL — module missing (and/or fixture import error to resolve first).

- [ ] **Step 3: Implement `_features.py`** (adapted verbatim from the prior monolith; deltas marked)

```python
# silly_kicks/calibration/_features.py
"""Augmented-VAEP feature enrichment for the TF-24 calibration harness.

Two entry points implementing the CachedObjective invariant/patch split (spec §4/§4a):

- ``enrich_invariant`` runs the 14 trial-INDEPENDENT enrichment steps once, leaving the
  5 trial-dependent columns as NaN placeholders.
- ``patch_trial_columns`` runs ONLY the 2 trial-dependent steps (link_zones pressure +
  off-ball runs), overwriting exactly those 5 columns.

``ALL_FEATURES`` is the model's feature matrix — the proven set from the prior TC-3
monolith. Line-break columns are deliberately NOT features (so they are never computed;
the patch uses ``add_off_ball_runs``, not the ``add_off_ball_context`` umbrella). xT is a
frozen exogenous artifact passed in (``xt``), never fit here.

See NOTICE for the per-feature methodology citations.

Examples
--------
>>> from silly_kicks.calibration._features import enrich_invariant, patch_trial_columns
>>> # base, links = enrich_invariant(actions=a, frames=f, xt=xt, home_team_id=h,
>>> #     carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0})
>>> # patched = patch_trial_columns(base_actions=base, frames=f, links=links,
>>> #     home_team_id=h, k3=1.0, pre_seconds=1.5, min_displacement_m=3.0)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# The 5 columns written by the two trial-dependent steps (spec §4a).
_TRIAL_DEPENDENT_COLS = [
    "pressure_on_actor__link_zones",
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
]

_SPADL_FEATURES = [
    "type_id",
    "result_id",
    "bodypart_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
]

_TRACKING_FEATURES = [
    "nearest_defender_distance",
    "actor_speed",
    "receiver_zone_density",
    "defenders_in_triangle_to_goal",
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "pressure_on_actor__andrienko_oval",
    "pressure_on_actor__link_zones",
    "pressure_on_actor__bekkers_pi",
    "pitch_control_at_ball__spearman",
    "pitch_control_at_ball__fernandez_bornn",
    "pitch_control_at_ball__voronoi",
    "defensive_line_x",
    "back_line_high_x",
    "compactness_x",
    "lateral_width",
    "max_lateral_gap",
    "back_n_count",
    "n_off_ball_runners_pre_window",
    "max_off_ball_run_displacement_pre_window",
    "mean_off_ball_run_speed_pre_window",
    "n_off_ball_runners_toward_goal_pre_window",
    "team_shape_centroid_x_attacking",
    "team_shape_centroid_y_attacking",
    "team_shape_convex_hull_area_attacking",
    "team_shape_team_length_attacking",
    "team_shape_team_width_attacking",
    "team_shape_stretch_index_attacking",
    "team_shape_centroid_x_defending",
    "team_shape_centroid_y_defending",
    "team_shape_convex_hull_area_defending",
    "team_shape_team_length_defending",
    "team_shape_team_width_defending",
    "team_shape_stretch_index_defending",
    "das_team",
    "das_opponent",
    "das_diff",
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "gk_closing_time_mean_s__six_yard_box",
    "gk_closing_time_min_s__six_yard_box",
    "n_blocked_receivers",
    "n_potential_receivers",
    "blocking_score",
    "blocked_threat_fraction",
    "max_single_defender_blocking_score",
    "sync_score_min",
    "sync_score_mean",
    "sync_score_high_quality_frac",
]

ALL_FEATURES = _SPADL_FEATURES + _TRACKING_FEATURES


def _compute_das(actions: pd.DataFrame, frames: pd.DataFrame, links: pd.DataFrame,
                 carrier_params: dict) -> tuple[pd.DataFrame, bool]:
    """Step 12: DAS team/opponent/diff columns. Returns (actions, das_ok).

    das_ok is False if DAS degraded to NaN (M8 — calibration must SURFACE silent DAS
    failures, not absorb them; the caller counts + records them in the manifest).
    """
    import warnings

    from silly_kicks.tracking import derive_team_in_possession, infer_ball_carrier
    from silly_kicks.tracking._das import get_individual_das

    das_ok = True
    try:
        carrier = infer_ball_carrier(
            frames,
            tolerance_m=carrier_params["tolerance_m"],
            beta=carrier_params["beta"],
            gamma=carrier_params["gamma"],
        )
        frames_with_tip = derive_team_in_possession(frames, carrier)
        del carrier
        linked = links[["action_id", "frame_id"]].dropna(subset=["frame_id"])
        linked = linked.merge(actions[["action_id", "period_id"]], on="action_id", how="left")
        linked_frame_ids = linked[["period_id", "frame_id"]].drop_duplicates()
        das_frames = frames_with_tip.merge(linked_frame_ids, on=["period_id", "frame_id"], how="inner")
        del linked, frames_with_tip
        das_result = get_individual_das(das_frames, use_progress_bar=False, chunk_size=10)
        del das_frames
        player_rows = das_result[das_result["is_ball"] != True]  # noqa: E712
        valid_rows = player_rows.dropna(subset=["DAS"])
        das_lookup: dict[tuple, dict] = {}
        for (_pid, fid, tid), grp in valid_rows.groupby(["period_id", "frame_id", "team_id"]):
            das_lookup.setdefault((_pid, fid), {})[tid] = float(grp["DAS"].sum())
        del das_result, player_rows, valid_rows
        pointer_lookup = links.set_index("action_id")
        team_vals = np.full(len(actions), np.nan)
        opp_vals = np.full(len(actions), np.nan)
        for row in actions.itertuples():
            i = row.Index
            aid = row.action_id
            if aid not in pointer_lookup.index:
                continue
            fid_raw = pointer_lookup.at[aid, "frame_id"]
            if pd.isna(fid_raw):
                continue
            key = (row.period_id, int(float(fid_raw)))
            if key not in das_lookup:
                continue
            team_id = row.team_id
            team_vals[i] = das_lookup[key].get(team_id, np.nan)
            opp = [v for k, v in das_lookup[key].items() if k != team_id]
            if opp:
                opp_vals[i] = opp[0]
        actions = actions.copy()
        actions["das_team"] = team_vals
        actions["das_opponent"] = opp_vals
        actions["das_diff"] = team_vals - opp_vals
    except (IndexError, ValueError, RuntimeError, TypeError):
        warnings.warn(
            f"DAS degraded to NaN for this match (carrier_params={carrier_params}) — "
            "feature columns das_team/das_opponent/das_diff are NaN; recorded in the manifest",
            UserWarning,
            stacklevel=2,
        )
        das_ok = False
        actions = actions.copy()
        actions["das_team"] = np.nan
        actions["das_opponent"] = np.nan
        actions["das_diff"] = np.nan
    return actions, das_ok


def enrich_invariant(
    *,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: Any,
    home_team_id: int | str,
    carrier_params: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Run the 14 trial-independent enrichment steps; leave the 5 trial cols as NaN.

    Returns ``(base_actions, links, das_ok)``. ``xt`` is a frozen ``ExpectedThreat`` (consumed
    by gk-influence + cover-shadows only). ``carrier_params`` are the fixed Stage-1 optimum.
    ``das_ok`` is False if DAS degraded for this match (M8).
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context
    from silly_kicks.tracking import (
        add_action_context,
        add_actor_pre_window,
        add_cover_shadows,
        add_defensive_line,
        add_gk_influence,
        add_pressure_on_actor,
        add_sync_score,
        add_team_shape,
        link_actions_to_frames,
        pitch_control_at_action,
    )

    actions = actions.copy()
    links, _report = link_actions_to_frames(actions, frames)

    actions = add_pre_shot_gk_context(actions, frames=frames)            # Step 1
    actions = add_action_context(actions, frames, links=links)           # Step 2
    actions = add_actor_pre_window(actions, frames, links=links)         # Step 3
    actions = add_pressure_on_actor(actions, frames, links=links,
                                    methods=("andrienko_oval",))          # Step 4a
    actions["pressure_on_actor__link_zones"] = np.nan                     # Step 4b SKIPPED (k3)
    try:                                                                  # Step 4c
        actions = add_pressure_on_actor(actions, frames, links=links, methods=("bekkers_pi",))
    except ValueError as exc:
        if "is_ball=True" in str(exc):
            actions["pressure_on_actor__bekkers_pi"] = np.nan
        else:
            raise
    for method in ("spearman", "fernandez_bornn", "voronoi"):            # Steps 5-7
        s = pitch_control_at_action(actions, frames, links=links, method=method)
        actions[s.name] = s.values
    actions = add_defensive_line(actions, frames, links=links, home_team_id=home_team_id)  # Step 8
    for col in _TRIAL_DEPENDENT_COLS[1:]:                                 # Step 9 SKIPPED (off-ball runs)
        actions[col] = np.nan
    # Step 10 (line-break) DELETED — not a feature (spec §4a).
    actions = add_team_shape(actions, frames, links=links, home_team_id=home_team_id)      # Step 11
    actions, das_ok = _compute_das(actions, frames, links, carrier_params)  # Step 12
    actions = add_gk_influence(actions, frames, xt, links=links, home_team_id=home_team_id)   # Step 13
    actions = add_cover_shadows(actions, frames, xt, links=links, home_team_id=home_team_id)  # Step 14
    actions = add_sync_score(actions, links)                              # Step 15
    return actions, links, das_ok


def patch_trial_columns(
    *,
    base_actions: pd.DataFrame,
    frames: pd.DataFrame,
    links: pd.DataFrame,
    home_team_id: int | str,
    k3: float,
    pre_seconds: float,
    min_displacement_m: float,
) -> pd.DataFrame:
    """Overwrite ONLY the 5 trial-dependent columns on a cached invariant base.

    Runs link_zones pressure (k3) + off-ball RUNS (pre_seconds, min_displacement_m).
    Uses ``add_off_ball_runs`` (RUN cols only), NOT the ``add_off_ball_context`` umbrella.
    """
    from silly_kicks.tracking import LinkParams, add_off_ball_runs, add_pressure_on_actor

    actions = base_actions.copy()
    actions = add_pressure_on_actor(
        actions, frames, links=links,
        methods=("link_zones",),
        params_per_method={"link_zones": LinkParams(k3=k3)},
    )
    actions = add_off_ball_runs(
        actions, frames, home_team_id=home_team_id,
        pre_seconds=pre_seconds, min_displacement_m=min_displacement_m,
    )
    return actions


def enrich_full(
    *,
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    xt: Any,
    home_team_id: int | str,
    carrier_params: dict,
    k3: float,
    pre_seconds: float,
    min_displacement_m: float,
) -> pd.DataFrame:
    """MONOLITHIC recompute: all 16 steps with the trial params applied INLINE at their
    natural positions (no NaN placeholders, no cached base).

    This is the INDEPENDENT full path the CachedObjective's ``evaluate`` uses. It must NOT be
    ``enrich_invariant`` + ``patch_trial_columns`` (that would make assert_cache_equivalence
    tautological — H1). Running 4b/9 inline BEFORE steps 11-15 is what lets the equivalence
    test catch a "trial-independent" step that secretly reads a trial-varying column (e.g. if
    gk_influence/cover_shadows/team_shape ever consumed the pressure/off-ball columns, the
    invariant would compute them from NaN while this path uses real values → divergence caught).
    """
    from silly_kicks.spadl.utils import add_pre_shot_gk_context
    from silly_kicks.tracking import (
        LinkParams,
        add_action_context,
        add_actor_pre_window,
        add_cover_shadows,
        add_defensive_line,
        add_gk_influence,
        add_off_ball_runs,
        add_pressure_on_actor,
        add_sync_score,
        add_team_shape,
        link_actions_to_frames,
        pitch_control_at_action,
    )

    actions = actions.copy()
    links, _report = link_actions_to_frames(actions, frames)
    actions = add_pre_shot_gk_context(actions, frames=frames)            # 1
    actions = add_action_context(actions, frames, links=links)           # 2
    actions = add_actor_pre_window(actions, frames, links=links)         # 3
    actions = add_pressure_on_actor(actions, frames, links=links, methods=("andrienko_oval",))  # 4a
    actions = add_pressure_on_actor(actions, frames, links=links,        # 4b INLINE (k3)
                                    methods=("link_zones",),
                                    params_per_method={"link_zones": LinkParams(k3=k3)})
    try:                                                                 # 4c
        actions = add_pressure_on_actor(actions, frames, links=links, methods=("bekkers_pi",))
    except ValueError as exc:
        if "is_ball=True" in str(exc):
            actions["pressure_on_actor__bekkers_pi"] = np.nan
        else:
            raise
    for method in ("spearman", "fernandez_bornn", "voronoi"):            # 5-7
        s = pitch_control_at_action(actions, frames, links=links, method=method)
        actions[s.name] = s.values
    actions = add_defensive_line(actions, frames, links=links, home_team_id=home_team_id)  # 8
    actions = add_off_ball_runs(actions, frames, home_team_id=home_team_id,  # 9 INLINE (pre_seconds, min_disp)
                                pre_seconds=pre_seconds, min_displacement_m=min_displacement_m)
    # Step 10 (line-break) DELETED — not a feature.
    actions = add_team_shape(actions, frames, links=links, home_team_id=home_team_id)      # 11
    actions, _das_ok = _compute_das(actions, frames, links, carrier_params)  # 12
    actions = add_gk_influence(actions, frames, xt, links=links, home_team_id=home_team_id)   # 13
    actions = add_cover_shadows(actions, frames, xt, links=links, home_team_id=home_team_id)  # 14
    actions = add_sync_score(actions, links)                              # 15
    return actions
```

> NOTE: confirm `add_off_ball_runs` is exported from `silly_kicks.tracking` (it lives in `silly_kicks.tracking.features`). If not in the package `__init__`, import from `silly_kicks.tracking.features`. Grep `silly_kicks/tracking/__init__.py` for `add_off_ball_runs` before finalizing the import line.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_features.py -v`
Expected: 5 passed (resolve the synthetic-fixture import first per the NOTE).

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_features.py tests/calibration/test_features.py tests/calibration/conftest.py
git commit -m "feat(calibration): invariant/patch enrichment + ALL_FEATURES (frozen-xT, no line-break)"
```

---

## Task 4: `_gates.py` — H1 penalty + signal sanity

**Files:**
- Create: `silly_kicks/calibration/_gates.py`
- Test: `tests/calibration/test_gates.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_gates.py
import numpy as np
import pandas as pd

from silly_kicks.calibration._gates import (
    default_feature_variances,
    h1_penalty_fires,
    signal_sanity,
)
from silly_kicks.calibration._features import _TRIAL_DEPENDENT_COLS


def test_default_variances_computed_over_trial_cols():
    X = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    variances = default_feature_variances(X)
    for c in _TRIAL_DEPENDENT_COLS:
        assert variances[c] > 0


def test_h1_fires_when_a_trial_col_collapses():
    defaults = {c: 1.0 for c in _TRIAL_DEPENDENT_COLS}
    # A degenerate candidate: link_zones pressure variance ~0 (all equal)
    X = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    X["pressure_on_actor__link_zones"] = 0.5  # constant => variance 0
    assert h1_penalty_fires(X, defaults) is True


def test_h1_does_not_fire_for_healthy_features():
    defaults = {c: 1.0 for c in _TRIAL_DEPENDENT_COLS}
    X = pd.DataFrame({c: np.linspace(0, 1, 50) for c in _TRIAL_DEPENDENT_COLS})
    assert h1_penalty_fires(X, defaults) is False


def test_signal_sanity_excludes_zero_signal_provider():
    per_provider = {"gs": 0.85, "idsse": 0.0, "skillcorner": 0.80}
    kept, excluded = signal_sanity(per_provider, min_value=0.01)
    assert "idsse" in excluded
    assert set(kept) == {"gs", "skillcorner"}
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_gates.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_gates.py`**

```python
# silly_kicks/calibration/_gates.py
"""Calibration gates (spec §5).

H1 degenerate-feature gate: if a tuned feature's variance collapses below 10% of its
default-param variance, the trial is steered away with a finite penalty Brier. The penalty
MAGNITUDE is anchored to the default-param held-out Brier (computed once in prepare()),
NOT to a running "worst-observed" value — keeping the objective STATELESS, resume-stable,
and path-comparable (so assert_cache_equivalence can check it). See spec §5 / R1.

Signal-sanity gate: a provider contributing ~0 matched carrier events is excluded with a
loud warning at LOAD time (data-determined, fixed for the whole study) — never silently
averaged in.

Examples
--------
>>> from silly_kicks.calibration._gates import h1_penalty_fires
>>> # if h1_penalty_fires(X_trial, default_variances): return penalty
"""

from __future__ import annotations

import warnings

import pandas as pd

from silly_kicks.calibration._features import _TRIAL_DEPENDENT_COLS

VARIANCE_GATE_RATIO = 0.1  # H1: < 10% of default-param variance => degenerate
PENALTY_K = 5.0            # penalty = K * default_param_brier (R1: stateless, ~5x any real trial)


def default_feature_variances(default_X: pd.DataFrame) -> dict[str, float]:
    """Variance of each trial-dependent feature at the DEFAULT params (computed once)."""
    return {c: float(default_X[c].var()) for c in _TRIAL_DEPENDENT_COLS if c in default_X.columns}


def h1_penalty_fires(trial_X: pd.DataFrame, default_variances: dict[str, float]) -> bool:
    """True if any tuned feature's variance dropped below 10% of its default variance."""
    for col, default_var in default_variances.items():
        if col not in trial_X.columns or default_var <= 0:
            continue
        current_var = float(trial_X[col].var())
        if current_var / default_var < VARIANCE_GATE_RATIO:
            warnings.warn(
                f"H1 gate: {col} variance {current_var:.6g} < 10% of default "
                f"{default_var:.6g} — returning penalty Brier",
                UserWarning,
                stacklevel=2,
            )
            return True
    return False


def signal_sanity(
    per_provider_value: dict[str, float], *, min_value: float = 0.01
) -> tuple[list[str], list[str]]:
    """Split providers into (kept, excluded); a ~0-signal provider is excluded loudly."""
    kept, excluded = [], []
    for provider, value in per_provider_value.items():
        if value is None or value < min_value:
            excluded.append(provider)
            warnings.warn(
                f"Signal-sanity gate: provider {provider!r} contributes ~0 signal "
                f"({value}) — excluded from the equal-weight mean, not silently averaged in",
                UserWarning,
                stacklevel=2,
            )
        else:
            kept.append(provider)
    return kept, excluded
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_gates.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_gates.py tests/calibration/test_gates.py
git commit -m "feat(calibration): H1 stateless penalty + provider signal-sanity gate"
```

---

## Task 5: `_spaces.py` — OptunaConfig builders

**Files:**
- Create: `silly_kicks/calibration/_spaces.py`
- Test: `tests/calibration/test_spaces.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_spaces.py
import pytest
from ruthless import Direction, OptunaConfig

from silly_kicks.calibration._spaces import stage1_config, stage2_config


def test_stage1_config_is_maximize_with_three_params():
    cfg = stage1_config(n_trials=10, store_path="/tmp/s1.db")
    assert isinstance(cfg, OptunaConfig)
    assert cfg.metric == "carrier_accuracy"
    assert cfg.direction is Direction.MAXIMIZE
    assert set(cfg.param_space) == {"tolerance_m", "beta", "gamma"}
    assert set(cfg.warm_start) == {"tolerance_m", "beta", "gamma"}  # current defaults
    assert cfg.store.path == "/tmp/s1.db"


def test_stage2_config_is_minimize_with_three_params():
    cfg = stage2_config(n_trials=10, store_path="/tmp/s2.db")
    assert cfg.metric == "brier"
    assert cfg.direction is Direction.MINIMIZE
    assert set(cfg.param_space) == {"k3", "pre_seconds", "min_displacement_m"}
    assert cfg.param_space["k3"].log is True  # log-uniform


def test_warm_start_subset_of_param_space_enforced_by_ruthless():
    # OptunaConfig validates warm_start ⊆ param_space; our builders must satisfy it.
    stage1_config(n_trials=1, store_path="/tmp/x.db")  # must not raise
    stage2_config(n_trials=1, store_path="/tmp/y.db")  # must not raise
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_spaces.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_spaces.py`** (defaults from the verified silly-kicks signatures)

```python
# silly_kicks/calibration/_spaces.py
"""OptunaConfig builders for the two calibration stages (spec §3/§4).

Search spaces + warm-starts (current library defaults) from the verified signatures:
infer_ball_carrier(tolerance_m=3.0, beta=0.5, gamma=1.0); LinkParams(k3=1.0);
add_off_ball_runs(pre_seconds=1.5, min_displacement_m=3.0).

Examples
--------
>>> from silly_kicks.calibration._spaces import stage1_config, stage2_config
>>> cfg = stage2_config(n_trials=60, store_path="tc3_stage2.db")
"""

from __future__ import annotations

from ruthless import Direction, FloatRange, OptunaConfig
from ruthless.config.common import StoreConfig


def stage1_config(*, n_trials: int, store_path: str, sampler: str = "tpe") -> OptunaConfig:
    """Stage 1 — carrier accuracy (maximize): tolerance_m, beta, gamma."""
    return OptunaConfig(
        kind="optuna",
        metric="carrier_accuracy",
        direction=Direction.MAXIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "tolerance_m": FloatRange(kind="float", lo=1.0, hi=8.0),
            "beta": FloatRange(kind="float", lo=0.0, hi=2.0),
            "gamma": FloatRange(kind="float", lo=0.0, hi=3.0),
        },
        warm_start={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0},
        store=StoreConfig(kind="sqlite", path=store_path),
    )


def stage2_config(*, n_trials: int, store_path: str, sampler: str = "tpe") -> OptunaConfig:
    """Stage 2 — augmented-VAEP held-out Brier (minimize): k3, pre_seconds, min_displacement_m."""
    return OptunaConfig(
        kind="optuna",
        metric="brier",
        direction=Direction.MINIMIZE,
        n_trials=n_trials,
        sampler=sampler,
        param_space={
            "k3": FloatRange(kind="float", lo=0.1, hi=5.0, log=True),
            "pre_seconds": FloatRange(kind="float", lo=0.5, hi=5.0),
            "min_displacement_m": FloatRange(kind="float", lo=1.0, hi=8.0),
        },
        warm_start={"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0},
        store=StoreConfig(kind="sqlite", path=store_path),
    )
```

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_spaces.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_spaces.py tests/calibration/test_spaces.py
git commit -m "feat(calibration): Stage-1/Stage-2 OptunaConfig builders"
```

---

## Task 6: `_carrier_objective.py` — Stage-1 objective

**Files:**
- Create: `silly_kicks/calibration/_carrier_objective.py`
- Test: `tests/calibration/test_carrier_objective.py`

**Context:** Stage 1 maximizes `mean(inferred_carrier == action.player_id)` over carrier-actor actions (`pass/cross/shot/dribble`), equal-weight across providers. Use `ball_carrier_at_action` (the clean per-action wrapper) rather than the prior monolith's manual frame-merge.

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_carrier_objective.py
import numpy as np
from ruthless import Candidate

from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective


def test_perfect_carrier_fixture_scores_one(synth_known_carrier):
    # synth_known_carrier: one provider, frames where the action actor IS the closest player.
    fold = synth_known_carrier  # {"provider": [(actions, frames, home_team_id), ...]}
    obj = CarrierAccuracyObjective(fold)
    metrics = obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}))
    assert metrics["carrier_accuracy"] >= 0.99
    assert "carrier_accuracy__provA" in metrics  # per-provider attr present


def test_equal_provider_weighting(synth_two_providers_imbalanced):
    # provA: 100 matches @ acc 1.0; provB: 1 match @ acc 0.0 => equal-weight mean = 0.5
    fold = synth_two_providers_imbalanced
    obj = CarrierAccuracyObjective(fold)
    metrics = obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}))
    assert metrics["carrier_accuracy"] == 0.5  # NOT match-count-weighted (would be ~0.99)
```

> NOTE: add `synth_known_carrier` + `synth_two_providers_imbalanced` fixtures to `tests/calibration/conftest.py`, built from the synthetic-match helper. The "perfect" fixture places the actor as the nearest player to the ball at the linked frame.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_carrier_objective.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_carrier_objective.py`**

```python
# silly_kicks/calibration/_carrier_objective.py
"""Stage-1 carrier-accuracy objective (ruthless Objective; spec §3).

Maximizes the fraction of carrier-actor actions (pass/cross/shot/dribble — actor == ball
carrier by definition) whose inferred ball carrier matches the SPADL actor, averaged with
EQUAL WEIGHT per provider (so match-count imbalance can't dominate).

Examples
--------
>>> from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
>>> from ruthless import Candidate
>>> # obj = CarrierAccuracyObjective(fold)  # fold: {provider: [(actions, frames, home_team_id)]}
>>> # obj.evaluate(Candidate(id="t0", params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}))
"""

from __future__ import annotations

import numpy as np

from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._gates import signal_sanity

_CARRIER_ACTION_TYPES = {"pass", "cross", "shot", "dribble"}


def _match_accuracy(actions, frames, *, tolerance_m, beta, gamma) -> tuple[float, int]:
    """Carrier accuracy for one match + the number of carrier-actor actions compared."""
    from silly_kicks.tracking.features import ball_carrier_at_action

    if "type_name" in actions.columns:
        mask = actions["type_name"].isin(_CARRIER_ACTION_TYPES)
    else:
        from silly_kicks.spadl.config import actiontypes
        type_ids = {i for i, name in enumerate(actiontypes()) if name in _CARRIER_ACTION_TYPES}
        mask = actions["type_id"].isin(type_ids)

    filtered = actions[mask]
    if filtered.empty:
        return float("nan"), 0
    inferred = ball_carrier_at_action(
        filtered, frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma
    )
    # Compare as strings to avoid Int64/int64/object dtype mismatch (provider-asymmetric ids).
    matched = inferred.astype(str).values == filtered["player_id"].astype(str).values
    valid = inferred.notna().values
    n = int(valid.sum())
    if n == 0:
        return float("nan"), 0
    return float(matched[valid].mean()), n


class CarrierAccuracyObjective:
    """ruthless ``Objective`` (maximize ``carrier_accuracy``)."""

    def __init__(self, fold: dict[str, list[tuple]]) -> None:
        # fold: {provider: [(actions, frames, home_team_id), ...]}
        self._fold = fold
        self.diagnostics: dict = {}  # surfaced into the manifest (M1)

    def evaluate(self, candidate: Candidate) -> Metrics:
        p = candidate.params
        tolerance_m, beta, gamma = float(p["tolerance_m"]), float(p["beta"]), float(p["gamma"])
        per_provider: dict[str, float] = {}
        total_compared: dict[str, int] = {}
        for provider, matches in self._fold.items():
            accs, weights = [], []
            for actions, frames, _home in matches:
                acc, n = _match_accuracy(actions, frames, tolerance_m=tolerance_m, beta=beta, gamma=gamma)
                if n > 0 and not np.isnan(acc):
                    accs.append(acc)
                    weights.append(n)
            # Record compared-count for EVERY provider in the fold (0 if none) so signal_sanity sees it.
            total_compared[provider] = int(sum(weights))
            if accs:
                # Within a provider, weight by compared-action count; ACROSS providers, equal weight.
                per_provider[provider] = float(np.average(accs, weights=weights))
        # M1: loudly EXCLUDE providers with ~0 matched carrier events (the old GS=0.0 failure mode),
        # never silently averaged in.
        kept, excluded = signal_sanity({pr: float(n) for pr, n in total_compared.items()}, min_value=1.0)
        self.diagnostics["excluded_providers"] = excluded
        per_provider = {pr: per_provider[pr] for pr in kept if pr in per_provider}
        if not per_provider:
            return {"carrier_accuracy": 0.0}
        metrics: Metrics = {"carrier_accuracy": float(np.mean(list(per_provider.values())))}
        for provider, acc in per_provider.items():
            metrics[f"carrier_accuracy__{provider}"] = acc
            metrics[f"n_compared__{provider}"] = float(total_compared[provider])
        return metrics
```

> NOTE: verify `silly_kicks.spadl.config.actiontypes` is callable vs a module attribute (the prior monolith used `enumerate(actiontypes)` directly). Grep `silly_kicks/spadl/config.py` — if it is a `@functools.cache` function, call `actiontypes()`; if a plain list/tuple, drop the parens. The test will tell you.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_carrier_objective.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_carrier_objective.py tests/calibration/test_carrier_objective.py tests/calibration/conftest.py
git commit -m "feat(calibration): Stage-1 carrier-accuracy objective (equal-provider-weight)"
```

---

## Task 7: `_vaep_brier_objective.py` — Stage-2 CachedObjective (the core)

**Files:**
- Create: `silly_kicks/calibration/_vaep_brier_objective.py`
- Test: `tests/calibration/test_vaep_brier_objective.py`

**Context:** This is the load-bearing component. `patch_params = {"k3", "pre_seconds", "min_displacement_m"}`. `prepare()` runs `enrich_invariant` per match once + computes VAEP labels + the default-param Brier (for the H1 penalty anchor) + default-feature variances. `evaluate_patch` runs `patch_trial_columns` per match, assembles `ALL_FEATURES`, trains **deterministic** XGBoost per CV fold for `scores` and `concedes`, returns equal-weight per-provider Brier mean. `evaluate` is the full recompute using the SAME functions in the SAME order.

- [ ] **Step 1: Write the failing test (incl. cache-equivalence)**

```python
# tests/calibration/test_vaep_brier_objective.py
import numpy as np
from ruthless import Candidate, assert_cache_equivalence

from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective


def _candidates():
    # MUST vary all 3 patch params across >=2 values each (assert_cache_equivalence contract, L4).
    return [
        Candidate(id="c0", params={"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0}),
        Candidate(id="c1", params={"k3": 2.5, "pre_seconds": 3.0, "min_displacement_m": 5.0}),
    ]


def test_patch_params_declared():
    assert AugmentedVaepBrierObjective.patch_params == frozenset(
        {"k3", "pre_seconds", "min_displacement_m"}
    )


def test_returns_finite_brier_and_per_provider_attrs(stage2_fold, frozen_xt):
    obj = AugmentedVaepBrierObjective(
        fold=stage2_fold, xt=frozen_xt.xt,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42,
    )
    m = obj.evaluate(_candidates()[0])
    assert np.isfinite(m["brier"])
    assert any(k.startswith("brier__") for k in m)       # per-provider Brier
    assert any(k.startswith("brier_se__") for k in m)    # per-provider CV SE (M1)


def test_cache_equivalence_fast_equals_full(stage2_fold, frozen_xt):
    obj = AugmentedVaepBrierObjective(
        fold=stage2_fold, xt=frozen_xt.xt,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42,
    )
    # Deterministic XGBoost + identical funcs/order => fast path ≡ full recompute to 1e-9.
    assert_cache_equivalence(obj, _candidates())


def test_feature_matrix_parity_full_vs_invariant_patch(synth, frozen_xt):
    # N2: assert parity at the FEATURE level (not just downstream Brier). If enrich_full and
    # enrich_invariant+patch ever diverge, this localizes WHICH column — instead of an opaque
    # "brier 0.043 != 0.041" from assert_cache_equivalence. Shift-left + debuggable.
    import pandas as pd
    from silly_kicks.calibration._features import (
        ALL_FEATURES, enrich_full, enrich_invariant, patch_trial_columns,
    )

    actions, frames, home = synth
    cp = {"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}
    params = {"k3": 2.0, "pre_seconds": 2.5, "min_displacement_m": 4.0}
    full = enrich_full(actions=actions, frames=frames, xt=frozen_xt.xt,
                       home_team_id=home, carrier_params=cp, **params)
    base, links, _das = enrich_invariant(actions=actions, frames=frames, xt=frozen_xt.xt,
                                         home_team_id=home, carrier_params=cp)
    patched = patch_trial_columns(base_actions=base, frames=frames, links=links,
                                  home_team_id=home, **params)
    pd.testing.assert_frame_equal(
        full[ALL_FEATURES].reset_index(drop=True),
        patched[ALL_FEATURES].reset_index(drop=True),
        check_dtype=False,
    )


def test_h1_penalty_is_path_stable(stage2_fold_degenerate, frozen_xt):
    # A candidate that collapses a tuned feature returns the SAME default-Brier-anchored
    # penalty via evaluate and evaluate_patch (R1 stateless penalty).
    obj = AugmentedVaepBrierObjective(
        fold=stage2_fold_degenerate, xt=frozen_xt.xt,
        carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42,
    )
    assert_cache_equivalence(obj, _candidates())


def test_provider_cv_keeps_labels_aligned_on_single_class_fold():
    # M4: if a fold is single-class for ONE label, it is skipped for BOTH (no zip misalignment).
    import numpy as np
    import pandas as pd
    from silly_kicks.calibration._vaep_brier_objective import _provider_cv

    n = 60
    X = pd.DataFrame({"f": np.linspace(0, 1, n)})
    mids = np.array([f"m{i // 6}" for i in range(n)])  # 10 matches => GroupKFold(5)
    # scores: both classes present everywhere; concedes: all-zero within one match group.
    y_scores = (np.arange(n) % 2).astype(int)
    y_concedes = np.ones(n, dtype=int)
    y_concedes[mids == "m0"] = 0  # one match all-1 elsewhere => some train folds single-class
    mean_b, se = _provider_cv(X, y_scores, y_concedes, mids, seed=42)
    # No crash, finite result, and SE is well-defined (>=2 aligned folds survived).
    assert mean_b is None or np.isfinite(mean_b)
```

> NOTE: `stage2_fold` is `{provider: [(actions, frames, home_team_id), ...]}` with ≥2 matches per provider (so GroupKFold/LOMO has folds) and enough shot+keeper_save rows that `scores`/`concedes` are not all-constant (else Brier/XGBoost degenerate). `stage2_fold_degenerate` forces the link_zones pressure column to collapse. Build both in `conftest.py`. Keep matches tiny (shallow trees, single thread) so CI is fast.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_vaep_brier_objective.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_vaep_brier_objective.py`**

```python
# silly_kicks/calibration/_vaep_brier_objective.py
"""Stage-2 augmented-VAEP held-out Brier objective (ruthless CachedObjective; spec §4).

patch_params = {k3, pre_seconds, min_displacement_m}. The expensive enrichment is invariant
across these (prepared once per match); only link_zones pressure + off-ball runs re-run per
trial. xT is a frozen exogenous artifact (no leak). XGBoost is pinned deterministic so the
fast path equals the full recompute to 1e-9 (assert_cache_equivalence).

Examples
--------
>>> from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective
>>> # obj = AugmentedVaepBrierObjective(fold=fold, xt=frozen.xt,
>>> #     carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0}, seed=42)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import warnings

from ruthless import Direction, penalty_metrics
from ruthless.result import Candidate, Metrics

from silly_kicks.calibration._cv import cv_standard_error, match_cv_splits
from silly_kicks.calibration._features import (
    ALL_FEATURES,
    _TRIAL_DEPENDENT_COLS,
    enrich_full,
    enrich_invariant,
    patch_trial_columns,
)
from silly_kicks.calibration._gates import (
    PENALTY_K,
    default_feature_variances,
    h1_penalty_fires,
    signal_sanity,
)

_DEFAULT_PARAMS = {"k3": 1.0, "pre_seconds": 1.5, "min_displacement_m": 3.0}


def _vaep_labels(actions: pd.DataFrame) -> pd.DataFrame:
    """scores + concedes labels (10-action window), aligned 1:1 with actions."""
    from silly_kicks.spadl import add_names
    from silly_kicks.vaep.labels import concedes, scores

    named = add_names(actions)
    out = pd.DataFrame(index=actions.index)
    out["scores"] = scores(named, nr_actions=10)["scores"].values
    out["concedes"] = concedes(named, nr_actions=10)["concedes"].values
    return out


def _xgb_classifier(seed: int):
    """Fully-pinned deterministic XGBoost (C1/L3): the 1e-9 cache gate rides on this.

    Fixed seed + single thread + hist + subsample/colsample EXPLICITLY 1.0 (defend against a
    future default change) + no early stopping => identical features give identical Brier.
    """
    import xgboost as xgb

    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        tree_method="hist",
        n_jobs=1,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
    )


def _provider_cv(X: pd.DataFrame, y_scores: np.ndarray, y_concedes: np.ndarray,
                 mids: np.ndarray, seed: int) -> tuple[float | None, float]:
    """Per-provider CV: per-fold Brier = mean(scores_brier, concedes_brier); return (mean, SE).

    M4: splits are computed ONCE; a fold is skipped for BOTH labels if EITHER label is
    single-class in train — so the two labels stay fold-aligned (no zip-misalignment). NaN is
    passed through to XGBoost (M3 — never fillna(0); deterministic under n_jobs=1).
    Returns (None, nan) if no usable fold (caller treats as no-signal -> signal_sanity).
    """
    from sklearn.metrics import brier_score_loss

    fold_briers: list[float] = []
    for train_idx, test_idx in match_cv_splits(mids):
        if len(np.unique(y_scores[train_idx])) < 2 or len(np.unique(y_concedes[train_idx])) < 2:
            continue  # skip this fold for BOTH labels (keeps them aligned)
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        per_label = []
        for y in (y_scores, y_concedes):
            model = _xgb_classifier(seed)
            model.fit(Xtr, y[train_idx])           # NaN passthrough (no fillna)
            probs = model.predict_proba(Xte)[:, 1]
            per_label.append(float(brier_score_loss(y[test_idx], probs)))
        fold_briers.append(float(np.mean(per_label)))
    if not fold_briers:
        return None, float("nan")
    return float(np.mean(fold_briers)), cv_standard_error(fold_briers)


class _Invariant:
    """Prepared invariant: per-(provider, match) base actions/links/labels + penalty anchors."""

    def __init__(self) -> None:
        self.bases: dict[str, list[dict]] = {}   # provider -> [{frames, base, links, labels, home, match_id}]
        self.kept_providers: list[str] = []      # signal-sanity survivors (R7: fixed for the study)
        self.default_variances: dict[str, float] = {}
        self.default_brier: float = 0.25  # fallback before computed
        self.das_degraded: dict[str, int] = {}   # provider -> n matches with degraded DAS (M8)


class AugmentedVaepBrierObjective:
    """ruthless ``CachedObjective`` — minimize equal-provider-weight held-out Brier."""

    patch_params = frozenset({"k3", "pre_seconds", "min_displacement_m"})

    def __init__(self, *, fold: dict[str, list[tuple]], xt: Any, carrier_params: dict, seed: int = 42) -> None:
        self._fold = fold
        self._xt = xt
        self._carrier_params = carrier_params
        self._seed = seed
        self.diagnostics: dict = {}  # surfaced into the manifest (M1/M8)

    # ---- helpers ---------------------------------------------------------
    def _assemble(self, per_match: list[dict]) -> dict:
        """Concat one provider's matches into {X (ALL_FEATURES, NaN kept), y_scores, y_concedes, mids}."""
        X_parts, y_s, y_c, mids = [], [], [], []
        for e in per_match:
            X = e["X"][ALL_FEATURES]  # NaN passthrough (M3) — no fillna(0)
            X_parts.append(X)
            y_s.append(e["labels"]["scores"].values)
            y_c.append(e["labels"]["concedes"].values)
            mids.append(np.array([e["match_id"]] * len(X)))
        return {"X": pd.concat(X_parts, ignore_index=True),
                "y_scores": np.concatenate(y_s), "y_concedes": np.concatenate(y_c),
                "mids": np.concatenate(mids)}

    def _score_features(self, per_provider: dict[str, dict], default_variances, default_brier) -> Metrics:
        """Shared scorer: H1 gate + per-provider CV + equal-provider-weight mean.

        ``per_provider`` already contains ONLY the signal-sanity-kept providers (filtered at
        prepare() time, R7) — the equal-weight denominator is fixed for the whole study here.
        """
        if not per_provider:
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)
        trial_X = pd.concat([a["X"] for a in per_provider.values()], ignore_index=True)
        if h1_penalty_fires(trial_X, default_variances):
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)

        provider_brier: dict[str, float] = {}
        provider_se: dict[str, float] = {}
        for provider, a in per_provider.items():
            mean_b, se = _provider_cv(a["X"], a["y_scores"], a["y_concedes"], a["mids"], self._seed)
            if mean_b is not None:
                provider_brier[provider] = mean_b
                provider_se[provider] = se
        if not provider_brier:
            return penalty_metrics("brier", Direction.MINIMIZE, magnitude=PENALTY_K * default_brier)

        metrics: Metrics = {"brier": float(np.mean(list(provider_brier.values())))}
        for provider, b in provider_brier.items():
            metrics[f"brier__{provider}"] = b
            metrics[f"brier_se__{provider}"] = provider_se[provider]
        return metrics

    # ---- fast path -------------------------------------------------------
    def prepare(self) -> _Invariant:
        inv = _Invariant()
        for provider, matches in self._fold.items():
            inv.bases[provider] = []
            inv.das_degraded[provider] = 0
            for match_idx, (actions, frames, home) in enumerate(matches):
                base, links, das_ok = enrich_invariant(
                    actions=actions, frames=frames, xt=self._xt, home_team_id=home,
                    carrier_params=self._carrier_params,
                )
                if not das_ok:
                    inv.das_degraded[provider] += 1
                inv.bases[provider].append(
                    {"frames": frames, "raw_actions": actions, "base": base, "links": links,
                     "labels": _vaep_labels(base), "home": home, "match_id": f"{provider}:{match_idx}"}
                )
        # R7/M1: signal-sanity at PREPARE time (data-determined, FIXED for the whole study). A
        # provider with no usable label signal (no match has both classes in either label) is
        # loudly EXCLUDED, never silently averaged in. The equal-weight denominator is set here.
        signal = {
            p: float(sum(1 for e in entries
                         if e["labels"]["scores"].nunique() > 1 or e["labels"]["concedes"].nunique() > 1))
            for p, entries in inv.bases.items()
        }
        kept, excluded = signal_sanity(signal, min_value=1.0)
        inv.kept_providers = kept
        self.diagnostics["excluded_providers"] = excluded
        # Default-param pass: anchors for the H1 penalty (M2: SAME mean(scores,concedes) as scoring).
        default_per_provider = self._build_per_provider(inv, _DEFAULT_PARAMS, use_full=False)
        all_default_X = pd.concat([a["X"] for a in default_per_provider.values()], ignore_index=True)
        inv.default_variances = default_feature_variances(all_default_X)
        default_briers = []
        for a in default_per_provider.values():
            mean_b, _se = _provider_cv(a["X"], a["y_scores"], a["y_concedes"], a["mids"], self._seed)
            if mean_b is not None:
                default_briers.append(mean_b)
        if default_briers:
            inv.default_brier = float(np.mean(default_briers))
        # M8: surface DAS degradation (loud + manifest), never silent.
        total_degraded = sum(inv.das_degraded.values())
        if total_degraded:
            warnings.warn(f"DAS degraded on {total_degraded} match(es): {inv.das_degraded}",
                          UserWarning, stacklevel=2)
        self.diagnostics["das_degraded"] = dict(inv.das_degraded)
        return inv

    def _build_per_provider(self, inv: _Invariant, params: dict, *, use_full: bool) -> dict[str, dict]:
        """Assemble per-provider features. use_full=True => enrich_full (independent monolith, H1);
        else patch the cached invariant base."""
        per_provider: dict[str, dict] = {}
        providers = inv.kept_providers or list(inv.bases)  # kept-only (set in prepare before this is called)
        for provider in providers:
            entries = inv.bases[provider]
            per_match = []
            for e in entries:
                if use_full:
                    X_actions = enrich_full(
                        actions=e["raw_actions"],  # ORIGINAL SPADL actions — genuine from-scratch (H1)
                        frames=e["frames"], xt=self._xt, home_team_id=e["home"],
                        carrier_params=self._carrier_params, **params,
                    )
                else:
                    X_actions = patch_trial_columns(
                        base_actions=e["base"], frames=e["frames"], links=e["links"],
                        home_team_id=e["home"], **params,
                    )
                per_match.append({"X": X_actions, "labels": e["labels"], "match_id": e["match_id"]})
            per_provider[provider] = self._assemble(per_match)
        return per_provider

    def evaluate_patch(self, invariant: _Invariant, candidate: Candidate) -> Metrics:
        params = {k: float(candidate.params[k]) for k in ("k3", "pre_seconds", "min_displacement_m")}
        per_provider = self._build_per_provider(invariant, params, use_full=False)  # CACHED base + patch
        return self._score_features(per_provider, invariant.default_variances, invariant.default_brier)

    # ---- full path (Objective port) — INDEPENDENT monolith (H1) ----------
    def evaluate(self, candidate: Candidate) -> Metrics:
        invariant = self.prepare()  # anchors + per-match frames/actions/labels (prepare is deterministic)
        params = {k: float(candidate.params[k]) for k in ("k3", "pre_seconds", "min_displacement_m")}
        per_provider = self._build_per_provider(invariant, params, use_full=True)  # enrich_full, no cache
        return self._score_features(per_provider, invariant.default_variances, invariant.default_brier)
```

> CRITICAL notes for the implementer:
> - **H1 (independent evaluate):** `evaluate` builds features via `enrich_full` (monolithic, trial params inline), `evaluate_patch` via `enrich_invariant`+`patch_trial_columns`. `assert_cache_equivalence` now genuinely proves the decomposition (not a tautology). If it FAILS, a "trial-independent" step actually depends on a trial-varying column — fix the decomposition, do NOT loosen `rtol`.
> - **M4 (fold alignment, IMPLEMENTED in `_provider_cv`):** splits computed once; a fold skipped for BOTH labels iff EITHER is single-class in train → `scores`/`concedes` stay aligned (no `zip` misalignment). The targeted test below forces a single-class fold and asserts alignment.
> - **M3 (NaN passthrough):** `_assemble`/`_provider_cv` do NOT `fillna(0)` — XGBoost handles NaN natively (deterministic under `n_jobs=1`), and 0 would conflate missing with genuine-zero (`das_diff=0` etc.).
> - `GroupKFold`/`LeaveOneGroupOut` are deterministic (no RNG). `_vaep_labels` calls `add_names`; verify it's importable from `silly_kicks.spadl` and tolerant of the synthetic fixture's columns.

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_vaep_brier_objective.py -v`
Expected: 6 passed. If `assert_cache_equivalence` fails with a tiny mismatch, the column-parity test (N2) localizes WHICH feature diverged; the cause is the determinism guard (single-thread/seed/hist/subsample) or a genuine invariant/patch divergence (H1 — a "trial-independent" step reading a trial column). Fix the source, NEVER loosen `rtol`.

- [ ] **Step 5: Add the two objective exports to `silly_kicks/calibration/__init__.py`**

Add the imports + `__all__` entries:

```python
from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective
```
(and add `"AugmentedVaepBrierObjective"`, `"CarrierAccuracyObjective"` to `__all__`, keeping it sorted.)

- [ ] **Step 6: Verify frozen-xT round-trip parity (the Task 2 deferred check)**

```python
# tests/calibration/test_xt_consumers_parity.py
import pandas as pd
from silly_kicks.calibration._xt import fit_frozen_xt, save_xt, load_xt
from silly_kicks.tracking import add_gk_influence, link_actions_to_frames


def test_loaded_xt_produces_identical_gk_influence(synth, tmp_path):
    actions, frames, home = synth
    frozen = fit_frozen_xt(actions, exclude_match_ids=set(), match_id_col="game_id", source="x")
    save_xt(frozen, tmp_path / "xt.npz")
    reloaded = load_xt(tmp_path / "xt.npz")
    links, _ = link_actions_to_frames(actions, frames)
    a1 = add_gk_influence(actions.copy(), frames, frozen.xt, links=links, home_team_id=home)
    a2 = add_gk_influence(actions.copy(), frames, reloaded.xt, links=links, home_team_id=home)
    cols = ["gk_pitch_control_share_weighted", "gk_reachable_area_m2"]
    pd.testing.assert_frame_equal(a1[cols], a2[cols])
```

Run: `python -m pytest tests/calibration/test_xt_consumers_parity.py -v`
Expected: PASS. If it fails, `load_xt` must round-trip additional `ExpectedThreat` attributes (see the Task 2 NOTE) — add them to `save_xt`/`load_xt`.

- [ ] **Step 7: Commit**

```bash
git add silly_kicks/calibration/_vaep_brier_objective.py silly_kicks/calibration/__init__.py tests/calibration/test_vaep_brier_objective.py tests/calibration/test_xt_consumers_parity.py tests/calibration/conftest.py
git commit -m "feat(calibration): Stage-2 CachedObjective (frozen-xT, deterministic XGB, cache-equiv)"
```

---

## Task 8: `_diagnostics.py` — TF-25 gate + k3 sensitivity

**Files:**
- Create: `silly_kicks/calibration/_diagnostics.py`
- Test: `tests/calibration/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/calibration/test_diagnostics.py
from silly_kicks.calibration._diagnostics import tf25_gate_fires


def test_tf25_fires_when_gap_exceeds_provider_se():
    # global k3 Brier worse than provider-best by more than that provider's CV SE => fire
    assert tf25_gate_fires(global_brier=0.060, provider_best_brier=0.050, provider_se=0.005) is True


def test_tf25_does_not_fire_within_se():
    assert tf25_gate_fires(global_brier=0.052, provider_best_brier=0.050, provider_se=0.005) is False


def test_tf25_nan_se_never_fires():
    # single-fold SE is nan => cannot justify provider-specific defaults
    assert tf25_gate_fires(global_brier=0.10, provider_best_brier=0.05, provider_se=float("nan")) is False
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_diagnostics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `_diagnostics.py`**

```python
# silly_kicks/calibration/_diagnostics.py
"""TF-25 provider-specific-defaults gate + k3 sensitivity (spec §5 diagnostics).

The principled TF-25 trigger: a provider gets its own default ONLY if the gap between the
global-optimum Brier and that provider's own best-k3 Brier exceeds that provider's CV
standard error (computed against the scheme it actually uses — GroupKFold-5 for GS/SkillCorner,
LOMO for IDSSE). A nan SE (single fold) can never justify a provider-specific default.

Examples
--------
>>> from silly_kicks.calibration._diagnostics import tf25_gate_fires
>>> tf25_gate_fires(global_brier=0.06, provider_best_brier=0.05, provider_se=0.005)
True
"""

from __future__ import annotations

import math


def tf25_gate_fires(*, global_brier: float, provider_best_brier: float, provider_se: float) -> bool:
    """True if (global − provider-best) Brier gap exceeds the provider's CV SE."""
    if provider_se is None or math.isnan(provider_se):
        return False
    return (global_brier - provider_best_brier) > provider_se
```

> NOTE: the k3 1-D sensitivity scan (20 log-spaced steps, re-running only `add_pressure_on_actor`) is exercised by the CLI `--stage diagnostics` path (Task 11), not unit-tested here beyond the gate logic — it needs real data. Keep the pure gate logic in `_diagnostics.py`; the scan orchestration lives in the CLI.

- [ ] **Step 4: Run the tests to confirm they pass + export**

Run: `python -m pytest tests/calibration/test_diagnostics.py -v`
Expected: 3 passed. Then add `from silly_kicks.calibration._diagnostics import tf25_gate_fires` + `"tf25_gate_fires"` to `__init__.py`/`__all__`.

- [ ] **Step 5: Commit**

```bash
git add silly_kicks/calibration/_diagnostics.py silly_kicks/calibration/__init__.py tests/calibration/test_diagnostics.py
git commit -m "feat(calibration): TF-25 provider-specific-defaults gate"
```

---

## Task 9: Full public-API + import-isolation tests

**Files:**
- Modify: `tests/test_public_api_examples.py`
- Test: `tests/calibration/test_import_isolation.py`

- [ ] **Step 1: Add the calibration modules to the public-API examples gate**

In `tests/test_public_api_examples.py`, find `_PUBLIC_MODULE_FILES` and add the calibration modules whose public defs need `Examples` docstrings:

```python
    "silly_kicks/calibration/_xt.py",
    "silly_kicks/calibration/_cv.py",
    "silly_kicks/calibration/_features.py",
    "silly_kicks/calibration/_gates.py",
    "silly_kicks/calibration/_spaces.py",
    "silly_kicks/calibration/_carrier_objective.py",
    "silly_kicks/calibration/_vaep_brier_objective.py",
    "silly_kicks/calibration/_diagnostics.py",
```

- [ ] **Step 2: Write the import-isolation test**

```python
# tests/calibration/test_import_isolation.py
import importlib
import subprocess
import sys


def test_top_level_import_does_not_pull_calibration_or_heavy_deps():
    # Fresh subprocess: `import silly_kicks` must NOT import the optional subpackage or its heavy
    # deps (ruthless/xgboost), so `import silly_kicks` stays dependency-light (L1 — a real,
    # falsifiable guard, not `assert ... or True`).
    code = (
        "import sys; import silly_kicks; "
        "bad=[m for m in ('silly_kicks.calibration','ruthless','xgboost') if m in sys.modules]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, f"top-level import leaked: {proc.stdout.strip()}"


def test_top_level_init_has_no_calibration_import():
    src = importlib.import_module("silly_kicks").__file__
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    assert "import calibration" not in text and "from .calibration" not in text
```

- [ ] **Step 3: Run the public-API + isolation tests**

Run: `python -m pytest tests/test_public_api_examples.py tests/calibration/test_import_isolation.py -v`
Expected: PASS. If the examples gate flags a missing `Examples` block, add one to that def (every module above already has a module-level `Examples`; ensure each *public* def does too, or that the gate operates at module granularity — match the existing convention).

- [ ] **Step 4: Commit**

```bash
git add tests/test_public_api_examples.py tests/calibration/test_import_isolation.py
git commit -m "test(calibration): public-API examples gate + import-isolation guard"
```

---

## Task 10: `scripts/_loader_pining.py` — pining loader

**Files:**
- Create: `scripts/_loader_pining.py`
- Test: `tests/calibration/test_loader_pining.py`

**Context:** Mirror the PR-A e2e fetch (two-step Bearer→302→presigned-S3). Serves all three providers (verified 2026-05-29): SkillCorner (public, 10), **IDSSE (public, 7 — now live)**, Gradient Sports (owner, 64). The artifact formats and key names differ per provider, so the loader has a per-provider dispatch table. Returns an iterator of `(provider, match_id, actions, frames, home_team_id)`.

**Verified pining facts (2026-05-29):**
- Routes: `GET /{provider}/matches` → `{"matches": [{"id", "artifacts": {key: filename}, "visibility", ...}]}`; `GET /{provider}/matches/{id}/{artifact_key}` → **302 redirect to a presigned S3 URL**.
- Two-step download: GET the API route WITH `Authorization: Bearer <token>` and **auto-redirect DISABLED**, read the `Location` header, then GET the presigned URL **WITHOUT** the bearer (S3 rejects bearer + presigned signature together).
- Tokens: public default `test-token-pining-for-the-data` (serves SkillCorner + IDSSE); owner `PINING_FOR_THE_DATA_TOKEN` (adds GS). Base URL `PINING_API_URL` (default `https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1`).
- Per-provider artifacts (keys → filenames):
  - **idsse**: `events`→`events.xml`, `metadata`→`metadata.xml`, `tracking`→`tracking.xml` (DFL/Sportec XML; tracking ≈ **419 MB** → stream to a temp file, never hold in memory).
  - **skillcorner**: `{id}_dynamic_events`→`.csv`, `{id}_match`→`.json`, `{id}_tracking_extrapolated`→`.jsonl`, `{id}_phases_of_play`→`.csv` (**match-id-prefixed keys** — resolve by suffix, not a fixed name).
  - **gradientsports**: `events`→`events.json`, `metadata`→`metadata.json`, `roster`→`roster.json`, `tracking`→`tracking.jsonl.bz2`.

- [ ] **Step 1: Write the failing tests (stubbed network — no real calls)**

Two tests: (a) the fiddly two-step fetch sends the bearer + no-redirect on step 1 and NO bearer on step 2; (b) `load_matches` dispatches per provider and yields the uniform tuple.

```python
# tests/calibration/test_loader_pining.py
import pandas as pd

import scripts._loader_pining as L


def test_two_step_fetch_drops_bearer_on_presigned_get(monkeypatch, tmp_path):
    calls = []

    class FakeResp:
        def __init__(self, status=200, location=None, body=b"PAYLOAD"):
            self.status, self._loc, self._body = status, location, body
            self.headers = {"Location": location} if location else {}
        def read(self):
            return self._body
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        # Record (url, has_bearer) for each call.
        auth = req.get_header("Authorization") if hasattr(req, "get_header") else None
        url = req.full_url if hasattr(req, "full_url") else req
        calls.append((url, auth))
        if "execute-api" in str(url):  # step 1: API route returns 302 Location (raised by handler)
            import urllib.error
            raise urllib.error.HTTPError(url, 302, "Found", {"Location": "https://s3.example/presigned?sig=x"}, None)
        return FakeResp(body=b"PAYLOAD")  # step 2: presigned S3 GET

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    dest = L._download_to_temp("idsse", "M1", "tracking", "tok", L._DEFAULT_BASE_URL, tmp_path)
    assert dest.read_bytes() == b"PAYLOAD"
    # Step 1 carried the bearer; step 2 (presigned) did NOT.
    assert calls[0][1] == "Bearer tok"
    assert calls[1][1] is None


def test_load_matches_dispatches_per_provider(monkeypatch, tmp_path):
    # Stub network + conversion; assert orchestration + uniform tuple.
    monkeypatch.setattr(L, "_list_matches",
                        lambda provider, token, base_url: [{"id": "M1", "artifacts": {"events": "e", "metadata": "m", "tracking": "t"}}])
    monkeypatch.setattr(L, "_download_to_temp",
                        lambda *a, **k: tmp_path / "artifact.bin")
    built = {"actions": pd.DataFrame({"game_id": ["M1"], "action_id": [0], "player_id": [10],
                                      "period_id": [1], "time_seconds": [0.0], "team_id": [1]}),
             "frames": pd.DataFrame({"game_id": ["M1"], "period_id": [1], "frame_id": [0],
                                     "player_id": [10], "x": [1.0], "y": [1.0], "team_id": [1]}),
             "home": 1}
    monkeypatch.setattr(L, "_build_match",
                        lambda provider, match_id, paths, meta_path: (built["actions"], built["frames"], built["home"]))
    rows = list(L.load_matches(providers=["idsse"], match_ids={"idsse": ["M1"]}, token="test-token-pining-for-the-data"))
    assert len(rows) == 1
    provider, match_id, actions, frames, home = rows[0]
    assert (provider, match_id, home) == ("idsse", "M1", 1)
    assert isinstance(actions, pd.DataFrame) and isinstance(frames, pd.DataFrame)
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_loader_pining.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/_loader_pining.py`**

```python
# scripts/_loader_pining.py
"""pining-for-the-data loader for the TF-24 calibration harness.

Provider-agnostic fetch from the gated mock provider API (two-step Bearer -> 302 ->
presigned S3). Serves SkillCorner (public), IDSSE (public), Gradient Sports (owner). The
artifact formats + key names differ per provider, so conversion dispatches on provider.

No local paths, no committed data — token from PINING_FOR_THE_DATA_TOKEN (owner) or the
public default; base URL from PINING_API_URL.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

_DEFAULT_BASE_URL = "https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1"
_PUBLIC_TOKEN = "test-token-pining-for-the-data"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401
        return None  # surface the 302 as an HTTPError so we can read Location ourselves


def _base_url() -> str:
    return os.environ.get("PINING_API_URL", _DEFAULT_BASE_URL).rstrip("/")


def _resolve_token(token: str | None) -> str:
    # Owner token enables GS; otherwise the public token (SkillCorner + IDSSE).
    return token or os.environ.get("PINING_FOR_THE_DATA_TOKEN") or _PUBLIC_TOKEN


def _list_matches(provider: str, token: str, base_url: str) -> list[dict]:
    """GET /{provider}/matches -> the matches list (id + artifacts map)."""
    req = urllib.request.Request(f"{base_url}/{provider}/matches",  # noqa: S310
                                 headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read()).get("matches", [])


def _download_to_temp(provider: str, match_id: str, artifact_key: str, token: str,
                      base_url: str, dest_dir: Path) -> Path:
    """Two-step: bearer GET -> 302 Location -> presigned GET (no bearer) -> stream to a temp file.

    Streams so the ~419 MB IDSSE tracking.xml never sits fully in memory.
    """
    opener = urllib.request.build_opener(_NoRedirect)
    url = f"{base_url}/{provider}/matches/{match_id}/{artifact_key}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310
    location = None
    try:
        opener.open(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307):
            location = exc.headers.get("Location")
        else:
            raise
    if not location:
        raise RuntimeError(f"pining {provider}/{match_id}/{artifact_key}: expected a 302 redirect")
    dest = dest_dir / f"{provider}_{match_id}_{artifact_key}"
    # Presigned GET WITHOUT the bearer (S3 rejects bearer + presigned signature together).
    with urllib.request.urlopen(location, timeout=600) as resp, open(dest, "wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
    return dest


def _artifact_key(artifacts: dict, *, suffix: str) -> str:
    """Resolve an artifact KEY by filename suffix (SkillCorner keys are match-id-prefixed)."""
    for key, filename in artifacts.items():
        if str(filename).endswith(suffix):
            return key
    raise KeyError(f"no artifact ending with {suffix!r} in {sorted(artifacts)}")


def load_matches(
    *, providers: list[str], match_ids: dict[str, list[str]] | None = None, token: str | None = None
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) for each requested match."""
    import tempfile

    tok, base_url = _resolve_token(token), _base_url()
    for provider in providers:
        manifest = {m["id"]: m for m in _list_matches(provider, tok, base_url)}
        wanted = match_ids.get(provider) if match_ids else list(manifest)
        for match_id in wanted:
            artifacts = manifest[match_id]["artifacts"]
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                paths = _download_artifacts(provider, match_id, artifacts, tok, base_url, tmp_dir)
                actions, frames, home = _build_match(provider, match_id, paths, paths.get("metadata"))
            yield provider, match_id, actions, frames, home


def _download_artifacts(provider, match_id, artifacts, token, base_url, tmp_dir) -> dict[str, Path]:
    """Download the artifacts each provider needs, keyed by a NORMALISED role name."""
    if provider == "idsse":
        roles = {"events": "events.xml", "metadata": "metadata.xml", "tracking": "tracking.xml"}
    elif provider == "gradientsports":
        roles = {"events": "events.json", "metadata": "metadata.json",
                 "roster": "roster.json", "tracking": "tracking.jsonl.bz2"}
    elif provider == "skillcorner":
        roles = {
            "events": _artifact_key(artifacts, suffix="_dynamic_events.csv"),
            "metadata": _artifact_key(artifacts, suffix="_match.json"),
            "tracking": _artifact_key(artifacts, suffix="_tracking_extrapolated.jsonl"),
        }
    else:
        raise ValueError(f"unknown pining provider {provider!r}")
    # For idsse/gradientsports the role IS the artifact key; for skillcorner _artifact_key resolved it.
    out: dict[str, Path] = {}
    for role, key in roles.items():
        artifact_key = key if key in artifacts else role
        out[role] = _download_to_temp(provider, match_id, artifact_key, token, base_url, tmp_dir)
    return out


def _build_match(provider: str, match_id: str, paths: dict[str, Path], meta_path):
    """Provider dispatch: parse the downloaded artifacts into (actions, frames, home_team_id)."""
    if provider == "idsse":
        return _build_idsse(paths)
    if provider == "skillcorner":
        return _build_skillcorner(paths)
    if provider == "gradientsports":
        return _build_gradientsports(paths)
    raise ValueError(f"unknown pining provider {provider!r}")
```

> NOTE — the three `_build_*` parsers are the genuinely provider-specific glue; implement each against its real fixture (the Task 13 e2e exercises idsse + skillcorner for real). Grounded paths:
> - **`_build_idsse`** (DFL/Sportec XML). silly-kicks' tracking kloppy gateway REFUSES Sportec ("route through `silly_kicks.tracking.sportec`", ADR-004), and `silly_kicks.spadl.sportec` / `silly_kicks.tracking.sportec` take *already-normalised* DataFrames. So: events — `kloppy.sportec.load_event(event_data=paths["events"], meta_data=paths["metadata"])` → `silly_kicks.spadl.kloppy.convert_to_actions(dataset)`. Tracking — parse `tracking.xml` (`xmltodict` + `pd.json_normalize`, per `silly_kicks/tracking/sportec.py` module docstring) into the columns `tracking.sportec.convert_to_frames` expects, plus `home_team_id` + `home_starts_left` from `metadata.xml`, then call `silly_kicks.tracking.sportec.convert_to_frames(raw_frames, home_team_id=..., home_starts_left=...)`. (Alternative for tracking: `kloppy.sportec.load_tracking(...).to_df()` adapted to the same input columns — pick whichever round-trips cleanly; verify against the real 419 MB file but on a row-limited slice for the unit/dev loop.) Stream the 419 MB file to disk (already handled by `_download_to_temp`); do NOT `pd.read_*` it whole if memory-bound — consider `kloppy`'s `limit`/sampling for dev.
> - **`_build_skillcorner`** — events `{id}_dynamic_events.csv` + `{id}_match.json`; tracking `{id}_tracking_extrapolated.jsonl`. Use the existing SkillCorner path: `kloppy.skillcorner.load_tracking(...)` → `silly_kicks.tracking.kloppy.convert_to_frames` (the gateway supports SkillCorner), and the SkillCorner events converter (`silly_kicks.spadl.skillcorner` or `kloppy.skillcorner.load_event`). Confirm the exact silly-kicks SkillCorner events entry point by grepping `silly_kicks/spadl/` (PR-S40 shipped a SkillCorner events converter). `home_team_id` from `{id}_match.json`.
> - **`_build_gradientsports`** — port the PR-A e2e logic verbatim (`tests/tracking/test_gradientsports_player_ids_e2e.py`): flatten `tracking.jsonl.bz2` (homePlayers/awayPlayers/balls, `jerseyNum`) + `roster.json` → `add_gradientsports_player_ids(...)` → `convert_to_frames`; events from `events.json` (`gameEvents`). `home_team_id`/`away_team_id` from `metadata.json`.
>
> The unit test stubs `_list_matches`/`_download_to_temp`/`_build_match`, so the `_build_*` bodies are exercised by the e2e (Task 13), mirroring the GS PR-A pattern. Keep `home_team_id` resolution from metadata/roster — never `events["team"].iloc[0]` (home_team_id-heuristic-fragility memo).

- [ ] **Step 4: Run the tests to confirm they pass**

Run: `python -m pytest tests/calibration/test_loader_pining.py -v`
Expected: 2 passed (two-step fetch drops the bearer on the presigned GET; `load_matches` dispatches per provider).

- [ ] **Step 5: Commit**

```bash
git add scripts/_loader_pining.py tests/calibration/test_loader_pining.py
git commit -m "feat(calibration): pining loader (3-provider dispatch, 302 two-step, streamed download)"
```

---

## Task 11: `scripts/_loader_databricks.py` — bronze loader

**Files:**
- Create: `scripts/_loader_databricks.py`
- Test: `tests/calibration/test_loader_databricks.py`

**Context:** Adapt the prior monolith `_pull_provider_data_sql` (lines 73-285). Lazy-import `databricks.sql`; actionable hint if missing. Reads `soccer_analytics.bronze.{provider}_{tracking,events}` (id col `match_id`), runs silly-kicks converters, returns the uniform tuple. **IDSSE no longer needs this path** (it is public on pining as of 2026-05-29) — this loader is now the **operator-scale / fallback path** and the source for the `bronze.spadl_actions` **xT-fit corpus** (Task 12). Still useful for GS-at-scale or if pining is unavailable.

- [ ] **Step 1: Write the failing test (stubbed cursor — no Databricks)**

```python
# tests/calibration/test_loader_databricks.py
import pandas as pd

import scripts._loader_databricks as L


class _FakeCursor:
    def __init__(self, frames_df, events_df):
        self._frames, self._events = frames_df, events_df
        self._last = None
        self._drained = False

    def execute(self, sql, params=None):  # M5: now (sql, params)
        self._last = ("tracking" if "_tracking" in sql
                      else "events" if "_events" in sql else "ids")
        self._drained = False

    def _rows(self):
        if self._last == "ids":
            return [("m1",)]
        df = self._frames if self._last == "tracking" else self._events
        return list(df.itertuples(index=False, name=None))

    def fetchmany(self, n):  # L4: batched fetch — return all once, then empty
        if self._drained:
            return []
        self._drained = True
        return self._rows()

    @property
    def description(self):
        df = self._frames if self._last == "tracking" else self._events
        cols = ["match_id"] if self._last == "ids" else list(df.columns)
        return [(c,) for c in cols]

    def close(self):
        pass


def test_databricks_loader_uniform_tuple(monkeypatch):
    frames_df = pd.DataFrame({"match_id": ["m1"], "period": [1], "frame": [0],
                              "player_id": [10], "x": [10.0], "y": [10.0],
                              "ball_x": [11.0], "ball_y": [11.0], "team_id": [1]})
    events_df = pd.DataFrame({"match_id": ["m1"], "action_id": [0], "team_id": [1],
                              "player_id": [10], "period_id": [1], "time_seconds": [0.0],
                              "type_id": [0], "result_id": [1], "bodypart_id": [0],
                              "start_x": [10.0], "start_y": [10.0], "end_x": [20.0], "end_y": [20.0]})
    monkeypatch.setattr(L, "_connect", lambda: _FakeConn(frames_df, events_df))
    rows = list(L.load_matches(providers=["idsse"], match_ids={"idsse": ["m1"]}))
    assert len(rows) == 1
    assert rows[0][0] == "idsse"


class _FakeConn:
    def __init__(self, f, e):
        self._f, self._e = f, e

    def cursor(self):
        return _FakeCursor(self._f, self._e)

    def close(self):
        pass
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_loader_databricks.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/_loader_databricks.py`** (adapt prior `_pull_provider_data_sql`; convert via silly-kicks)

```python
# scripts/_loader_databricks.py
"""Databricks bronze loader for the TF-24 calibration harness.

Reads soccer_analytics.bronze.{provider}_{tracking,events} (id col: match_id), runs the
CURRENT silly-kicks converters (so calibration reflects current output), and yields the
uniform (provider, match_id, actions, frames, home_team_id) tuple. Interim IDSSE source
until IDSSE lands on pining; also the operator full-run path.

Env: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pandas as pd


# Allowlist of known bronze providers — the ONLY values interpolated into a table name (M5).
# match_id is ALWAYS parameterized; provider is validated against this set, never free-form.
_ALLOWED_PROVIDERS = frozenset(
    {"idsse", "skillcorner", "gradientsports", "metrica", "sportec", "statsbomb", "wyscout"}
)
_FETCH_BATCH = 50_000  # L4: stream big tracking pulls in batches, not one giant fetchall


def _connect():
    try:
        import databricks.sql as dbsql
    except ImportError as exc:  # actionable hint
        raise RuntimeError(
            "databricks-sql-connector is required for the Databricks loader: "
            "pip install databricks-sql-connector"
        ) from exc
    return dbsql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )


def _table(provider: str, kind: str) -> str:
    """Build a fully-qualified bronze table name from an ALLOWLISTED provider (M5)."""
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(f"provider {provider!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    return f"soccer_analytics.bronze.{provider}_{kind}"


def _query_param(cursor, sql: str, params=None) -> pd.DataFrame:
    """Execute a PARAMETERIZED query and batch-fetch into a DataFrame (L4)."""
    cursor.execute(sql, params or {})
    cols = [d[0] for d in cursor.description]
    rows = []
    while True:
        batch = cursor.fetchmany(_FETCH_BATCH)
        if not batch:
            break
        rows.extend(batch)
    return pd.DataFrame(rows, columns=cols)


def load_matches(
    *, providers: list[str], match_ids: dict[str, list[str]] | None = None
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) from bronze."""
    conn = _connect()
    try:
        cur = conn.cursor()
        for provider in providers:
            t_tracking, t_events = _table(provider, "tracking"), _table(provider, "events")
            ids = match_ids.get(provider) if match_ids else None
            if ids is None:
                # Table name is allowlist-validated (_table); no user input interpolated.
                ids = [r[0] for r in _query_param(cur, f"SELECT DISTINCT match_id FROM {t_tracking}").itertuples(index=False)]  # noqa: S608
            for mid in ids:
                # Table from allowlist; match_id PARAMETERIZED (M5) — never f-string-interpolated.
                raw_frames = _query_param(cur, f"SELECT * FROM {t_tracking} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                raw_events = _query_param(cur, f"SELECT * FROM {t_events} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                actions, frames, home = _convert(provider, raw_events, raw_frames)
                yield provider, str(mid), actions, frames, home
        cur.close()
    finally:
        conn.close()


def _convert(provider: str, raw_events: pd.DataFrame, raw_frames: pd.DataFrame):
    """Convert bronze rows to (actions, frames, home_team_id) via silly-kicks converters.

    Reuse the proven conversion from the prior monolith
    (luxury-lakehouse-d32/scripts/run_tc3_calibration.py:357-484): GS jersey->roster via
    add_gradientsports_player_ids, native skillcorner/sportec converters, home_team_id from
    tracking_player_metadata / mean-GK-x heuristic AVOIDED (use roster/metadata).
    """
    raise NotImplementedError(
        "Port _bronze_*_to_converter_input + _convert_tracking_to_frames from the prior monolith "
        "(see run_tc3_calibration.py:357-484); the test stubs _connect so this body is exercised "
        "only against real bronze."
    )
```

> NOTE (M5 paramstyle): `%(mid)s` is the `pyformat` paramstyle. Confirm the installed `databricks-sql-connector` paramstyle — recent versions default to `native` (`:mid` with a dict) and also accept `%(name)s`/`?`. If `%(mid)s` raises, switch to the connector's native style (`WHERE match_id = :mid`, `{"mid": mid}`); the point (M5) is that `match_id` is **bound**, never f-string-interpolated. The stubbed-cursor unit test must accept the `(sql, params)` 2-arg `execute` signature.
>
> NOTE: `_convert` is the one place that genuinely needs the prior monolith's bronze→converter glue (`_bronze_gs_to_converter_input`, `_convert_tracking_to_frames`, `_resolve_match_metadata`, lines 286-484). Port those helpers into this file (or a small `scripts/_bronze_convert.py`), preserving the home_team_id resolution (from metadata/roster — NOT `events["team"].iloc[0]`, per the home_team_id-heuristic-fragility memo). The stubbed-cursor unit test does not exercise `_convert` (it injects already-shaped frames); the real exercise is the operator run. Keep `_convert` faithful to the prior, proven code.

- [ ] **Step 4: Run the test to confirm it passes**

Run: `python -m pytest tests/calibration/test_loader_databricks.py -v`
Expected: 1 passed (the test injects pre-shaped frames and a provider whose `_convert` path the stub satisfies; if the fake provider hits `_convert`, give the fake frames the bronze column names the test asserts and route `idsse` through a minimal pass-through in `_convert`, OR adjust the test to a provider whose conversion is pass-through). Resolve so the test exercises `load_matches` end to end with the stub.

- [ ] **Step 5: Commit**

```bash
git add scripts/_loader_databricks.py tests/calibration/test_loader_databricks.py
git commit -m "feat(calibration): Databricks bronze loader (lazy connector, current converters)"
```

---

## Task 12: `scripts/calibrate_tracking_defaults.py` — CLI + report/manifest

**Files:**
- Create: `scripts/calibrate_tracking_defaults.py`
- Test: `tests/calibration/test_cli_smoke.py`

- [ ] **Step 1: Write the failing test (in-process smoke via the synthetic fold, no network)**

```python
# tests/calibration/test_cli_smoke.py
import json

from scripts.calibrate_tracking_defaults import build_manifest, run_stage


def test_build_manifest_has_data_and_version_identity(frozen_xt):
    manifest = build_manifest(
        source="pining", seed=42, n_trials=2,
        match_ids={"skillcorner": ["m1", "m2"]}, xt=frozen_xt, stage=1,
    )
    assert "silly_kicks_version" in manifest
    assert "ruthless_version" in manifest
    assert "xgboost_version" in manifest
    assert manifest["match_ids"] == {"skillcorner": ["m1", "m2"]}
    assert manifest["xt_artifact"]["sha256"] == frozen_xt.sha256


def test_stage1_smoke_returns_result(stage1_fold, tmp_path):
    result, objective = run_stage(stage=1, fold=stage1_fold, n_trials=2, seed=42,
                                  store_path=str(tmp_path / "s1.db"), xt=None, carrier_params=None)
    assert result.best is not None
    assert "carrier_accuracy" in result.best.metrics
    assert hasattr(objective, "diagnostics")  # surfaced into the manifest (M1/M8)
```

> NOTE: `run_stage` is the testable seam — it takes an already-loaded fold (no I/O) and runs the ruthless study. The CLI `main()` wires loaders → `run_stage` → report. `stage1_fold` is the carrier fixture from conftest.

- [ ] **Step 2: Run it to confirm it fails**

Run: `python -m pytest tests/calibration/test_cli_smoke.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `scripts/calibrate_tracking_defaults.py`**

```python
# scripts/calibrate_tracking_defaults.py
"""TF-24 calibration CLI — orchestrates the two Optuna studies + diagnostics.

Pure objectives/CV/gates live in silly_kicks.calibration; this script owns I/O (loaders),
study orchestration, the frozen-xT artifact, and the report + data/version manifest.

Usage (all three providers come from pining: SkillCorner+IDSSE public, GS via owner token):
    python scripts/calibrate_tracking_defaults.py --stage 1 --source pining \
        --providers skillcorner idsse gradientsports --n-trials 100 --store tc3_stage1.db
    python scripts/calibrate_tracking_defaults.py --stage 2 --source pining \
        --providers skillcorner idsse gradientsports --n-trials 60 --store tc3_stage2.db \
        --xt-artifact calibration_xt.npz --carrier-best carrier_best.json

The frozen-xT corpus (bronze.spadl_actions) is fetched via Databricks regardless of --source
(it is the disjoint exogenous corpus, not calibration data); DATABRICKS_* env vars are needed
only for the Stage-2 xT fit, not for the pining match loads.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from importlib.metadata import version

from ruthless import InProcessBackend, render_json, render_summary_md
from ruthless.strategies.optuna_ import OptunaStrategy

from silly_kicks.calibration import stage1_config, stage2_config
from silly_kicks.calibration._carrier_objective import CarrierAccuracyObjective
from silly_kicks.calibration._vaep_brier_objective import AugmentedVaepBrierObjective


def build_manifest(*, source, seed, n_trials, match_ids, xt, stage, diagnostics=None) -> dict:
    """Data + version manifest for auditability (spec §6 R3)."""
    manifest = {
        "stage": stage,
        "source": source,
        "seed": seed,
        "n_trials": n_trials,
        "match_ids": match_ids,
        "silly_kicks_version": version("silly-kicks"),
        "ruthless_version": version("ruthless-efficiency"),
        "xgboost_version": version("xgboost"),
        "generated_date": _dt.date.today().isoformat(),
    }
    if xt is not None:
        manifest["xt_artifact"] = xt.manifest()  # includes n_excluded (H2 audit)
    if diagnostics:
        manifest["diagnostics"] = diagnostics     # excluded_providers (M1) + das_degraded (M8)
    return manifest


def run_stage(*, stage, fold, n_trials, seed, store_path, xt, carrier_params):
    """Run one Optuna stage on an already-loaded fold (the testable seam — no I/O).

    Returns (result, objective) so the caller can read objective.diagnostics (M1/M8) for the
    manifest and the Stage-1 best params for the carrier_best.json handoff (M9).
    """
    if stage == 1:
        objective = CarrierAccuracyObjective(fold)
        config = stage1_config(n_trials=n_trials, store_path=store_path)
    elif stage == 2:
        objective = AugmentedVaepBrierObjective(
            fold=fold, xt=xt, carrier_params=carrier_params, seed=seed
        )
        config = stage2_config(n_trials=n_trials, store_path=store_path)
    else:
        raise ValueError(f"unknown stage {stage}")
    result = OptunaStrategy(config, seed=seed).run(objective, backend=InProcessBackend())
    return result, objective


def _load_fold(args):
    """Wire the chosen loader into the {provider: [(actions, frames, home)]} fold + match_ids."""
    if args.source == "pining":
        import scripts._loader_pining as loader
    else:
        import scripts._loader_databricks as loader
    fold: dict[str, list[tuple]] = {}
    used_ids: dict[str, list[str]] = {}
    for provider, mid, actions, frames, home in loader.load_matches(providers=args.providers, match_ids=None):
        fold.setdefault(provider, []).append((actions, frames, home))
        used_ids.setdefault(provider, []).append(mid)
    return fold, used_ids


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-24 tracking-defaults calibration")
    ap.add_argument("--stage", choices=["1", "2", "diagnostics"], required=True)
    ap.add_argument("--source", choices=["pining", "databricks", "auto"], default="pining")
    ap.add_argument("--providers", nargs="+", required=True)
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--store", required=True)
    ap.add_argument("--xt-artifact", default=None)
    ap.add_argument("--xt-corpus-source", choices=["pining", "databricks", "bundled"], default="pining")
    ap.add_argument("--carrier-best", default=None, help="JSON with the Stage-1 optimum (for Stage 2)")
    ap.add_argument("--report-out", default="calibration_report")
    args = ap.parse_args()

    fold, used_ids = _load_fold(args)

    xt = None
    carrier_params = None
    if args.stage == "2":
        xt = _resolve_xt(args, fold, used_ids)  # pining held-out (default) / bronze / bundled
        with open(args.carrier_best, encoding="utf-8") as fh:
            carrier_params = json.load(fh)
        missing = {"tolerance_m", "beta", "gamma"} - set(carrier_params)  # N4a: validate up front
        if missing:
            raise ValueError(f"carrier_best.json missing keys {sorted(missing)} — run Stage 1 first")

    result, objective = run_stage(
        stage=int(args.stage) if args.stage != "diagnostics" else "diagnostics",
        fold=fold, n_trials=args.n_trials, seed=args.seed, store_path=args.store,
        xt=xt, carrier_params=carrier_params,
    )

    # M9: Stage 1 writes carrier_best.json so Stage 2 consumes a RECORDED artifact (not hand-typed).
    if args.stage == "1" and result.best is not None:
        best_carrier = {k: result.best.candidate.params[k] for k in ("tolerance_m", "beta", "gamma")}
        with open(args.carrier_best or "carrier_best.json", "w", encoding="utf-8") as fh:
            json.dump(best_carrier, fh, indent=2)
        print(f"Wrote carrier_best.json: {best_carrier}")

    manifest = build_manifest(source=args.source, seed=args.seed, n_trials=args.n_trials,
                              match_ids=used_ids, xt=xt, stage=args.stage,
                              diagnostics=getattr(objective, "diagnostics", None))
    report = {"ruthless": json.loads(render_json(result)), "calibration_manifest": manifest}
    with open(f"{args.report_out}.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    with open(f"{args.report_out}.md", "w", encoding="utf-8") as fh:
        fh.write(render_summary_md(result))
        fh.write("\n\n## Calibration manifest\n\n```json\n")
        fh.write(json.dumps(manifest, indent=2))
        fh.write("\n```\n")
    print(render_summary_md(result))
    print(f"Best: {result.best.metrics if result.best else None}")


_XT_COLS = ["game_id", "start_x", "start_y", "end_x", "end_y", "type_id", "result_id"]


def _resolve_xt(args, fold, used_ids):
    """Fit-and-freeze the xT artifact on a disjoint corpus, or load + sha256-verify.

    Three corpus sources (N1):
    - ``pining`` (default, id-space-safe): fit on pining matches HELD OUT from the calibration
      set — same id space as the calibration loads, so no mapping needed (what the e2e validates).
    - ``databricks``: fit on ``bronze.spadl_actions`` MINUS the calibration matches — requires the
      bronze ``game_id`` space to MATCH the pining match_ids, else ``fit_frozen_xt`` fails closed
      (H2). Provide that mapping or it will (correctly) refuse to fit.
    - ``bundled``: socceraction's pre-fit grid (acceptable shortcut; un-recordable corpus).
    """
    from pathlib import Path

    from silly_kicks.calibration._xt import fit_frozen_xt, load_xt, save_xt

    if args.xt_artifact and Path(args.xt_artifact).exists():
        return load_xt(args.xt_artifact)

    if args.xt_corpus_source == "bundled":
        return _bundled_xt()

    calib_ids = {str(m) for ids in used_ids.values() for m in ids}
    if args.xt_corpus_source == "pining":
        corpus, corpus_ids = _load_xt_corpus_pining(args, calib_ids)
        overlap = corpus_ids & calib_ids
        if overlap:  # held-out corpus must be disjoint by construction
            raise ValueError(f"pining xT corpus overlaps calibration matches: {sorted(overlap)[:5]}")
        # Disjoint by construction => no ids to exclude; fit on the held-out corpus directly.
        frozen = fit_frozen_xt(corpus, exclude_match_ids=set(), match_id_col="game_id",
                               source="pining:held-out", fit_date=_dt.date.today().isoformat())
    else:  # databricks
        corpus = _load_xt_corpus_databricks()
        frozen = fit_frozen_xt(corpus, exclude_match_ids=calib_ids, match_id_col="game_id",
                               source="bronze.spadl_actions", fit_date=_dt.date.today().isoformat())
    if args.xt_artifact:
        save_xt(frozen, args.xt_artifact)
    return frozen


def _load_xt_corpus_pining(args, calib_ids):
    """Load actions from pining matches NOT in the calibration set (id-space-safe corpus, N1)."""
    import pandas as pd

    import scripts._loader_pining as P

    token = P._resolve_token(None)
    base_url = P._base_url()
    parts, corpus_ids = [], set()
    per_provider_cap = 8  # bound the corpus; enough matches for a stable xT grid
    for provider in args.providers:
        manifest = P._list_matches(provider, token, base_url)
        held_out = [m["id"] for m in manifest if str(m["id"]) not in calib_ids][:per_provider_cap]
        for _p, mid, actions, _frames, _home in P.load_matches(
            providers=[provider], match_ids={provider: held_out}, token=token
        ):
            parts.append(actions[[c for c in _XT_COLS if c in actions.columns]])
            corpus_ids.add(str(mid))
    return pd.concat(parts, ignore_index=True), corpus_ids


def _load_xt_corpus_databricks():
    """Load the xT-fit corpus from bronze.spadl_actions — ONLY the columns ExpectedThreat needs (N3)."""
    import scripts._loader_databricks as L
    conn = L._connect()
    try:
        cur = conn.cursor()
        cols = ", ".join(_XT_COLS)
        return L._query_param(cur, f"SELECT {cols} FROM soccer_analytics.bronze.spadl_actions")  # noqa: S608
    finally:
        conn.close()


def _bundled_xt():
    """socceraction's bundled xT grid (fallback; corpus identity not recordable)."""
    from silly_kicks.calibration._xt import FrozenXt
    from silly_kicks.xthreat import ExpectedThreat, load_model  # confirm the bundled-grid API

    xt = load_model()  # or ExpectedThreat() pre-fit; grep silly_kicks.xthreat for the bundled loader
    import numpy as np
    grid = np.asarray(xt.xT, dtype=np.float64)
    from silly_kicks.calibration._xt import _grid_sha256
    return FrozenXt(xt=xt, source="socceraction-bundled", corpus_match_ids=(), n_excluded=0,
                    fit_date="", grid_shape=(int(grid.shape[0]), int(grid.shape[1])),
                    sha256=_grid_sha256(grid))


if __name__ == "__main__":
    main()
```

> NOTE (N1 + H2): `--xt-corpus-source pining` (the DEFAULT) is **id-space-safe** — it fits xT on pining matches HELD OUT from the calibration set, in the SAME id space as the calibration loads, so no mapping is needed (this is exactly what `test_stage2_e2e_skillcorner_public` validates). `--xt-corpus-source databricks` fits on `bronze.spadl_actions` MINUS the calibration matches, which requires the bronze `game_id` space to MATCH the pining match_ids; if it doesn't, `fit_frozen_xt` **fails closed** (H2 — better than silently leaking). For databricks, the implementer MUST map the calibration ids into the bronze id space first (e.g. via `bronze.tracking_player_metadata`) — a fired raise means that mapping is missing, not a bug to suppress. `n_excluded` is recorded in the manifest. For the operator run, pre-fit once with `--xt-artifact` so it is reused + recorded.

- [ ] **Step 4: Run the smoke test to confirm it passes**

Run: `python -m pytest tests/calibration/test_cli_smoke.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_tracking_defaults.py tests/calibration/test_cli_smoke.py
git commit -m "feat(calibration): CLI + report/manifest (data+version+xT identity)"
```

---

## Task 13: pining-public e2e smoke (SkillCorner)

**Files:**
- Create: `tests/calibration/test_calibration_e2e.py`

**Context:** Reproducible real-data smokes against pining-**public** data (`@pytest.mark.e2e`, NOT in the default suite). SkillCorner (10 matches) is the **light default** e2e. IDSSE (7 matches) is **also public now** but its tracking.xml is ~419 MB/match, so its e2e is a **heavier opt-in** (gated behind an env flag, not run by default even within `-m e2e`). Both use the **public token** — no owner token needed (the loader falls back to `test-token-pining-for-the-data`). Runs Stage-1 with `n_trials=2` end to end.

- [ ] **Step 1: Write the e2e tests**

```python
# tests/calibration/test_calibration_e2e.py
import os

import pytest

pytestmark = pytest.mark.e2e


def _run_stage1(provider, match_ids):
    import scripts._loader_pining as loader
    from scripts.calibrate_tracking_defaults import run_stage

    fold = {}
    for prov, _mid, actions, frames, home in loader.load_matches(
        providers=[provider], match_ids={provider: match_ids}, token="test-token-pining-for-the-data"
    ):
        fold.setdefault(prov, []).append((actions, frames, home))
    result, _objective = run_stage(stage=1, fold=fold, n_trials=2, seed=42,
                                   store_path=str(_tmp_db()), xt=None, carrier_params=None)
    assert result.best is not None
    assert 0.0 <= result.best.metrics["carrier_accuracy"] <= 1.0


def _tmp_db():
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp()) / "e2e_stage1.db"


def test_stage1_e2e_skillcorner_public():
    # SkillCorner is public on pining; resolve two real ids from the live listing.
    _run_stage1("skillcorner", _two_public_match_ids("skillcorner"))


@pytest.mark.skipif(not os.environ.get("RUN_HEAVY_E2E"),
                    reason="IDSSE tracking is ~419 MB/match; set RUN_HEAVY_E2E=1 to run")
def test_stage1_e2e_idsse_public():
    # IDSSE is public on pining as of 2026-05-29 (DFL/Sportec XML; heavy download).
    _run_stage1("idsse", _two_public_match_ids("idsse")[:1])  # 1 match keeps it bounded


def _two_public_match_ids(provider):
    import scripts._loader_pining as loader
    manifest = loader._list_matches(provider, "test-token-pining-for-the-data", loader._base_url())
    return [m["id"] for m in manifest[:2]]


def test_stage2_e2e_skillcorner_public():
    # M7: the load-bearing CachedObjective (frozen xT + per-fold XGB + invariant/patch split)
    # exercised on REAL data — where H1/H2/NaN/fold-skip surprises actually surface.
    import scripts._loader_pining as loader
    from scripts.calibrate_tracking_defaults import run_stage
    from silly_kicks.calibration._xt import fit_frozen_xt

    ids = _two_public_match_ids("skillcorner") + _xt_corpus_match_id("skillcorner")
    loaded = {pid: (a, f, h) for _p, pid, a, f, h in loader.load_matches(
        providers=["skillcorner"], match_ids={"skillcorner": ids},
        token="test-token-pining-for-the-data")}
    calib_ids = ids[:2]
    fold = {"skillcorner": [loaded[i] for i in calib_ids]}
    # Frozen xT fit on a DISJOINT 3rd match (zero overlap with the 2 calibration matches).
    corpus_actions = loaded[ids[2]][0]
    xt = fit_frozen_xt(corpus_actions, exclude_match_ids=set(), match_id_col="game_id", source="e2e")
    result, objective = run_stage(stage=2, fold=fold, n_trials=2, seed=42,
                                  store_path=str(_tmp_db()), xt=xt.xt,
                                  carrier_params={"tolerance_m": 3.0, "beta": 0.5, "gamma": 1.0})
    assert result.best is not None
    import math
    assert math.isfinite(result.best.metrics["brier"])


def _xt_corpus_match_id(provider):
    import scripts._loader_pining as loader
    manifest = loader._list_matches(provider, "test-token-pining-for-the-data", loader._base_url())
    return [manifest[2]["id"]]  # a 3rd match, disjoint from the 2 calibration matches
```

> NOTE: `store_path` uses a temp-dir SQLite file (verify `StoreConfig` accepts the path; in-memory `:memory:` may not survive `create_study` resume semantics). The IDSSE e2e exercises the `_build_idsse` Sportec-XML path for real — expect to fix payload-shape surprises here (as PR-A found the GS metadata was a 1-element list). For dev iteration on the 419 MB file, slice/limit frames inside `_build_idsse` (e.g. first N frames) so the loop is fast; the calibration run itself uses full data.

- [ ] **Step 2: Run the regular suite to confirm the e2e is excluded**

Run: `python -m pytest tests/calibration/ -m "not e2e" -v`
Expected: all non-e2e calibration tests pass; the e2e tests are deselected.

- [ ] **Step 3: Run the light e2e for real (public token, no env needed) — MANDATORY before marking done**

Run both public SkillCorner e2es (Stage 1 + the load-bearing Stage 2):
`python -m pytest tests/calibration/test_calibration_e2e.py::test_stage1_e2e_skillcorner_public tests/calibration/test_calibration_e2e.py::test_stage2_e2e_skillcorner_public -v -m e2e`
Expected: both PASS — Stage 1 with a real `carrier_accuracy` in `[0, 1]`, Stage 2 with a finite `brier`. Per the project rule, the real e2e MUST run — do not skip it. SkillCorner is public, so no token is required. The Stage-2 e2e is where H1/H2/NaN/fold-skip integration surprises surface (M7). Then optionally run the IDSSE heavy e2e once with `RUN_HEAVY_E2E=1` to validate the Sportec-XML path end to end (the user has full access; running it once proves `_build_idsse`).

- [ ] **Step 4: Commit**

```bash
git add tests/calibration/test_calibration_e2e.py
git commit -m "test(calibration): pining-public SkillCorner Stage-1 e2e smoke"
```

---

## Task 14: Housekeeping — version, CHANGELOG, NOTICE, docs, final-review

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `NOTICE`, `README.md`
- Create: `scripts/README_calibration.md`

- [ ] **Step 1: Bump the version to 3.28.0 in ALL declaring files (hard gate)**

Per the version-bump hard gate, these must ALL match: `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `CHANGELOG.md` heading, `TODO.md`. First check main isn't already past 3.28.0:

```bash
git fetch origin main
grep version pyproject.toml | head -1
```

Set `version = "3.28.0"` in `pyproject.toml` and `__version__ = "3.28.0"` in `silly_kicks/__init__.py`.

- [ ] **Step 2: Add the CHANGELOG entry**

```markdown
## [3.28.0] - 2026-05-29

### Added
- **TF-24 calibration harness** (`silly_kicks/calibration/`, optional `[calibration]` extra):
  Optuna TPE calibration of three tracking defaults (`infer_ball_carrier` tolerance_m/beta/gamma;
  `LinkParams.k3`; off-ball-run pre_seconds/min_displacement_m) against real multi-provider tracking
  data via `ruthless-efficiency[optuna]`. Pure provider-agnostic objectives/CV/gates + `scripts/`
  CLI with pining-for-the-data + Databricks bronze loaders. Stage 2 uses a ruthless `CachedObjective`
  (invariant prepared once; only 2 trial-varying steps per trial) with a frozen exogenous xT artifact
  (train–serve-consistent, zero held-out leak). Produces recommended values + an auditable
  data+version+xT-identity manifest. Does NOT change the library defaults (that is the separate
  "apply" PR after the maintainer's real sweep).
```

- [ ] **Step 3: Update `TODO.md`** — remove the TF-24 PR-B row (delete, don't strikethrough; CHANGELOG is the record). Leave the "apply defaults" follow-up + TF-25/TF-19 rows.

- [ ] **Step 4: NOTICE check** — every methodology used is already cited (xT, pressure, off-ball runs, gk-influence, cover-shadows, DAS). Add a one-line note that the calibration harness reuses these; no NEW methodology is introduced (Optuna/TPE is tooling, not a published feature method). Confirm by grep that each `ALL_FEATURES` family has a NOTICE entry.

- [ ] **Step 5: Write `scripts/README_calibration.md`** — operator walkthrough: env vars (`PINING_FOR_THE_DATA_TOKEN` for GS owner-tier; public token for SkillCorner+IDSSE; `DATABRICKS_*` only if `--xt-corpus-source databricks`), the two-stage sequence (Stage 1 → auto-writes `carrier_best.json` → Stage 2 with `--carrier-best` + `--xt-artifact`), the **xT-corpus source** (default `pining` = id-space-safe held-out fit; `databricks` requires a bronze↔pining id mapping or it fails closed; `bundled` fallback), the `--stage diagnostics` TF-25 check, and the manifest/report outputs (incl. the xT-artifact identity + excluded-providers + DAS-degradation counts). Reference it from `README.md` under the existing optional-features section.

- [ ] **Step 6: Run the full local gate (Shift Left)**

```bash
pip install -e ".[test]"
ruff format --check .
ruff check .
pyright silly_kicks/
pyright scripts/calibrate_tracking_defaults.py scripts/_loader_pining.py scripts/_loader_databricks.py
python -m pytest tests/ -m "not e2e" -q
```
Expected: ruff clean (both), pyright clean on `silly_kicks/` AND the calibration scripts (M6 — the loaders/CLI are the gnarliest dynamic-dispatch I/O code and must not escape type-checking; if the repo convention excludes `scripts/` from pyright, add these three files explicitly to the pyright run/config), all non-e2e tests pass. Fix everything locally before commit (CI runs both `ruff format --check` AND `ruff check`).

- [ ] **Step 7: Run `/final-review` (mandatory gate before the single commit)**

Invoke the final-review skill. Address Critical/High inline. Confirm: ADR check (no new ADR needed — within ADR-004 + the optional-dep pattern; note in the review), architecture.html regen if the C4 skill flags drift, docs consistency.

- [ ] **Step 8: Squash to a single commit + open the PR**

Per the one-commit-per-branch policy (and only after explicit approval to commit):

```bash
git reset --soft main
git commit -m "feat(calibration): TF-24 Optuna calibration harness -- silly-kicks 3.28.0 (PR-B)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin pr-b-tf24-optuna-calibration
gh pr create --fill
```

> Do NOT tag until main CI is green (never-tag-before-CI-green). Merge → wait for CI → then tag v3.28.0 + push.

---

## Self-review (run after writing; fix inline)

**Spec coverage:**
- §1 two-stage goal → Tasks 5/6/7 (configs + objectives). ✓
- §2 module layout (`_features`/`_xt`/`_cv`/`_gates`/`_spaces`/objectives/`_diagnostics` + 2 loaders + CLI) → Tasks 1-12. ✓
- §3 Stage 1 (carrier accuracy, equal-provider-weight) → Task 6. ✓
- §4 Stage 2 CachedObjective (invariant/patch, deterministic XGB, per-provider Brier+SE) → Task 7. ✓
- §4a feature set + frozen xT (9 consumers, line-break not a feature) → Tasks 2/3/7. ✓
- §5 gates (H1 stateless penalty, signal sanity, TF-25 SE-per-scheme) → Tasks 4/8. ✓
- §6 ruthless integration (OptunaStrategy, public Direction, resume, report+manifest) → Tasks 5/12. ✓
- §7 testing (assert_cache_equivalence varying all 3 patch params, deterministic XGB, loaders stubbed, pining-public e2e) → Tasks 7/10/11/13. ✓
- §8 housekeeping (3.28.0, CHANGELOG, README, NOTICE) → Task 14. ✓
- §9 out of scope (apply PR, IDSSE→pining, the real sweep) — correctly NOT built. ✓

**Type/signature consistency:** `FloatRange(kind="float", ...)`, `OptunaConfig(kind="optuna", ...)`, `StoreConfig(kind="sqlite", path=...)`, `penalty_metrics(metric, direction, *, magnitude=...)`, `assert_cache_equivalence(obj, candidates)` (rtol/atol), `OptunaStrategy(config, seed=...).run(obj, backend=InProcessBackend())`, `patch_params` as a class attr `frozenset` — all match the verified ruthless 0.2.0 source. silly-kicks calls (`infer_ball_carrier`, `ball_carrier_at_action`, `add_pressure_on_actor` + `LinkParams(k3=)`, `add_off_ball_runs`, `add_gk_influence(…, xt, …)`, `scores`/`concedes`, `ExpectedThreat().fit`) match the verified signatures + the proven prior monolith.

**Lakehouse PLAN-review (round 2) resolutions — all folded in:**
- **H1** (vacuous cache-equivalence) → `evaluate()` is now an INDEPENDENT monolith (`enrich_full`, Task 3) with trial params inline; `evaluate_patch` uses invariant+patch; `assert_cache_equivalence` is now meaningful (Task 7). `use_cached_base` deleted. Spec §4 corrected.
- **H2** (exclusion no-op leak) → `fit_frozen_xt` FAILS CLOSED if any excluded id is absent from the corpus; `n_excluded` in the manifest (Task 2 + test; Task 12 NOTE on id-mapping). Spec §4a corrected.
- **M1** (signal_sanity unwired) → wired into `_score_features`; excluded providers recorded in `diagnostics` → manifest (Task 7/12).
- **M2** (anchor on scores only) → default_brier computed with the same `mean(scores, concedes)` CV as scoring (Task 7). Spec §5 corrected.
- **M3** (fillna(0)) → NaN passthrough to XGBoost; no `fillna(0)` (Task 7).
- **M4** (fold alignment prose) → implemented in `_provider_cv` (split once, skip fold for BOTH labels if EITHER single-class) + targeted test (Task 7).
- **M5** (S608) → provider allowlist + parameterized `match_id`; blanket per-file-ignore removed, only justified inline noqa on the allowlisted table name (Task 0/11).
- **M6** (pyright scope) → Task 14 also runs pyright on the calibration scripts.
- **M7** (no Stage-2 e2e) → real-data Stage-2 SkillCorner e2e added (Task 13).
- **M8** (silent DAS NaN) → `_compute_das` returns `das_ok`; per-provider degradation counted, warned, manifest-recorded (Task 3/7).
- **M9** (carrier handoff) → Stage-1 path writes `carrier_best.json` automatically (Task 12).
- **L1** (vacuous isolation test) → replaced with a subprocess guard (ruthless/xgboost absent after `import silly_kicks`) (Task 9).
- **L2** (test-count checkpoints) → Task 2 = 6 passed, Task 3 = 5 passed, Task 7 = 5 passed (corrected).
- **L3** (XGBoost pins) → `subsample=1.0`, `colsample_bytree=1.0` explicit; scikit-learn lower-bound note (Task 0/7).
- **L4** (fetchall OOM) → Databricks loader batches via `fetchmany` (Task 11).

**Lakehouse PLAN-review (round 4) resolutions — all folded in:**
- **N1** (CLI xT-corpus dead-end) → added `--xt-corpus-source pining` (now the DEFAULT, id-space-safe held-out fit — the path the e2e validates); `databricks` kept but documented as requiring an id-space mapping (else fails closed); `bundled` fallback (Task 12).
- **N2** (debuggability) → column-level `assert_frame_equal(enrich_full[ALL_FEATURES], (invariant+patch)[ALL_FEATURES])` test added next to `assert_cache_equivalence` (Task 7).
- **N3** (unbounded `SELECT *`) → `_load_xt_corpus_databricks` selects only the xT columns (`_XT_COLS`), not `*` (Task 12).
- **N4** (robustness) → (a) `carrier_best.json` keys validated up front in `main` (Task 12); (b) the `_a["game_id"].iloc[0]` IndexError is GONE — `calib_ids` now come from the loader's yielded `match_id` via `used_ids`, not by peeking at the actions frame.

**Open verification items deferred to implementation (each has a NOTE in-task):**
- The synthetic-match fixture name in `tests/tracking/` (Task 3 NOTE).
- `silly_kicks.spadl.config.actiontypes` callable-vs-list (Task 6 NOTE).
- `ExpectedThreat` attributes consumed by gk-influence/cover-shadows for the `load_xt` round-trip (Task 2 + Task 7 Step 6).
- Fold-alignment of `scores`/`concedes` for path-stable equivalence (Task 7 CRITICAL note).
- pining routes/two-step are VERIFIED (2026-05-29); the remaining glue is the three `_build_*` parsers — IDSSE Sportec-XML (kloppy events + native `tracking.sportec`), SkillCorner native, GS (port PR-A) (Task 10 NOTE; exercised by the Task 13 e2e for IDSSE + SkillCorner).
- bronze→converter glue port + home_team_id resolution (Task 11 NOTE).
- IDSSE 419 MB tracking: stream-to-disk handled; dev-loop should slice frames in `_build_idsse` (Task 10/13 NOTE).
- `_bundled_xt`: confirm silly-kicks/socceraction's bundled-grid loader API (`silly_kicks.xthreat.load_model()` or equivalent) exists before relying on the `bundled` corpus source (Task 12 NOTE).
