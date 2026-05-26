# TF-18 Ghost-GK Training Data Assembly + Hub Publish

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this spec's plan task-by-task.

**Goal:** Ship `prepare_ghost_gk_training_data()` as a composable building block for training Ghost-GK models, refactor `compute_ghost_gk` to share iteration logic and accept match context, flesh out the training script, and add a publish script for HuggingFace Hub.

**Context:** TF-18 model code shipped in 3.19.0 (PR-S54). The `GhostGkModel` class, `extract_ghost_gk_features()`, serialization (npz+JSON+SHA256), and `from_hub()` lazy download are all implemented. What's missing: (a) a public function to assemble training data from tracking frames, (b) real match context flowing through to features (score_diff, phase, ball_carrier), (c) a working training script, (d) Hub publish tooling.

**Architecture:** Shared internal batch helper extracts iteration logic from `compute_ghost_gk` into `_extract_all_ghost_gk_features()`. Both the inference path (`compute_ghost_gk`) and training path (`prepare_ghost_gk_training_data`) call the shared helper. Match context (score, phase, carrier) is optional --- the API works without events, but produces better features when SPADL actions are provided. The training script is a reference CLI that works with any parquet data source. No lakehouse dependency.

---

## 1. Scope

### In scope

1. **Shared batch helper** (`_extract_all_ghost_gk_features`) --- internal function extracting the frame-iteration + velocity-tracking + feature-extraction loop from `compute_ghost_gk`.
2. **Match context resolution** --- helper functions to derive `score_diff`, `phase`, and `ball_carrier_team_id` from SPADL actions + tracking frames.
3. **`prepare_ghost_gk_training_data()`** --- new public function in `_ghost_gk.py`, re-exported through `silly_kicks.tracking.__init__.py`. Per-game. Takes TRACKING_FRAMES_COLUMNS + optional SPADL actions, returns `(features, labels)` ready for `GhostGkModel.fit()`.
4. **`compute_ghost_gk()` refactoring** --- switch from inline loop to shared helper. Add optional `actions` parameter for match context. Backward-compatible (actions=None preserves defaults).
5. **Training script** (`scripts/train_ghost_gk.py`) --- reference CLI: load parquets, per-game feature extraction, GroupKFold CV, metrics, final model save.
6. **Publish script** (`scripts/publish_ghost_gk.py`) --- round-trip verify + upload to HuggingFace Hub.
7. **Bug fix:** `"timestamp"` references in `_ghost_gk.py` lines 342 and 837 should be `"time_seconds"` (matching TRACKING_FRAMES_COLUMNS schema).
8. **Tests** for all new code + backward compatibility.

### Out of scope

- Actual training on real data (requires multi-provider tracking data).
- Lakehouse integration script (lakehouse repo consumes the silly-kicks API).
- MLflow / UC Volume delivery (lakehouse-specific).
- Model card template authoring (lakehouse publishes model cards via its own `upload_hf_readme` infrastructure).
- TF-19 GKDV Layer 3 composition.
- ADR-005 amendment for Ghost-GK serialization (separate follow-up).

---

## 2. Shared Batch Helper

### `_extract_all_ghost_gk_features()`

Internal function. Single source of truth for the frame-iteration loop that both inference and training use.

```python
def _extract_all_ghost_gk_features(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    carrier: pd.DataFrame | None = None,
    score_at_time: Callable[[Any, float], float] | None = None,
    phase_at_time: Callable[[Any, float], int] | None = None,
    subsample_fps: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

**Parameters:**

| Param | Type | Purpose |
|-------|------|---------|
| `frames` | DataFrame | TRACKING_FRAMES_COLUMNS, LTR-normalized, vx/vy present |
| `home_team_id` | str or int | Home team (attacks right, GK at x=0) |
| `carrier` | DataFrame or None | Per-frame `(game_id, period_id, frame_id, ball_carrier_team_id)` |
| `score_at_time` | Callable or None | `(game_id, time_seconds) -> score_diff` (GK team minus opponent) |
| `phase_at_time` | Callable or None | `(game_id, time_seconds) -> int` (0=open, 1=set_piece, 2=goal_kick) |
| `subsample_fps` | float or None | Thin frames to target fps before extraction |

**Returns:**

- `features`: DataFrame shape `(n_samples, len(GHOST_GK_FEATURE_NAMES))` with `GHOST_GK_FEATURE_NAMES` columns.
- `meta`: DataFrame shape `(n_samples, 6)`: `game_id`, `period_id`, `frame_id`, `gk_team_id`, `gk_x_gr`, `gk_y_gr`. The `gk_x_gr`/`gk_y_gr` columns are the actual GK position in goal-relative coordinates (training labels).

**Logic:**

1. **Normalize `home_team_id`** to match `frames["team_id"].dtype` at entry (see Team ID Normalization below).
2. **Subsample** (if `subsample_fps` is not None): compute `step = round(frame_rate / subsample_fps)`, keep every `step`-th frame_id per (game_id, period_id).
3. **Group** by `(game_id, period_id, frame_id)`, sorted.
4. **Per frame, per GK row:**
   - Determine `goal_x`: `0.0` if `gk_team == home_team_id` else `105.0` (both already normalized to same type).
   - Look up `score_diff` via `score_at_time(game_id, time_s)` (default 0.0). This returns home-team perspective. **If `gk_team != home_team_id`, negate `score_diff`** so each GK sees the diff from their own team's perspective.
   - Look up `phase` via `phase_at_time(game_id, time_s)` (default 0).
   - Look up `ball_carrier_team_id` from pre-indexed `carrier` (default None). Before the loop: `carrier_idx = carrier.set_index(["game_id", "period_id", "frame_id"])["ball_carrier_team_id"]` for O(1) per-frame lookup.
   - Retrieve velocity state `(prev_defensive_line_x, prev_defending_centroid_x, dt)` from per-`(game_id, gk_team)` tracking dict.
   - Call `extract_ghost_gk_features()` with all context.
   - Extract actual GK (x, y) and transform to goal-relative coords for meta.
   - Update velocity state.
5. **Concatenate** all feature rows + meta rows.

---

## 3. Match Context Resolution

Three internal helper functions, all in `_ghost_gk.py`:

### `_build_score_lookup(actions, home_team_id) -> Callable`

Builds a `(game_id, time_seconds) -> score_diff` callback from SPADL actions.

1. Filter goals: `type_name == "shot"` and `result_name in ("success", "owngoal")` (using `spadlconfig.actiontypes_df()` / `spadlconfig.results_df()` for ID lookup to support both named and ID-based DataFrames).
2. For own goals: attribute to the **opponent** of the acting team (the shooting team "scored" against themselves). Flip `team_id` before accumulation.
3. Sort by `(game_id, time_seconds)`.
4. Per game: cumulative home goals and away goals via running sum.
5. Score diff = `home_score - away_score` (always home perspective).
6. Return callback that uses `np.searchsorted` on sorted timestamps for O(log n) lookup. The callback always returns home-team perspective; the **caller** (shared helper) negates for away-team GKs.

### `_build_phase_lookup(actions) -> Callable`

Builds a `(game_id, time_seconds) -> int` callback.

1. Identify set-piece action types: `freekick`, `corner`, `goalkick` (via `spadlconfig.actiontypes_df()`). `throw_in` excluded --- throw-ins don't materially alter GK positioning expectations (GK rarely adjusts for a throw-in the way they do for a corner or free kick).
2. Sort actions by `(game_id, time_seconds)`.
3. For any query timestamp: find the most recent prior action. If it's a set-piece type and within `_SET_PIECE_DECAY_SECONDS = 10.0` seconds, return 1 (set_piece) or 2 (goal_kick). Otherwise 0 (open play). The constant is module-level in `_ghost_gk.py`.
4. Return callback using `np.searchsorted`.

### Ball carrier

Not a new helper --- reuses existing `infer_ball_carrier(frames)` + `derive_team_in_possession(frames, carrier)` from `_ball_carrier.py`. Returns per-frame `ball_carrier_team_id`.

---

## 4. `prepare_ghost_gk_training_data()` Public API

```python
def prepare_ghost_gk_training_data(
    frames: pd.DataFrame,
    *,
    home_team_id: str | int,
    actions: pd.DataFrame | None = None,
    subsample_fps: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble training features + labels from one game's tracking frames.

    Parameters
    ----------
    frames : pd.DataFrame
        TRACKING_FRAMES_COLUMNS schema, LTR-normalized, with vx/vy
        columns (from smooth_frames + derive_velocities).
    home_team_id : str | int
        Home team ID (attacks right in LTR convention).
    actions : pd.DataFrame | None
        SPADL actions for the same game. Provides score_diff and phase
        context. If None, both default to 0 (valid but less informative).
    subsample_fps : float
        Target frame rate for training (default 1.0 Hz). At 25fps
        source data, 1fps reduces training set ~25x while preserving
        positional diversity (GK moves ~0.5m/s average).

    Returns
    -------
    features : pd.DataFrame
        (n_samples, len(GHOST_GK_FEATURE_NAMES)) with GHOST_GK_FEATURE_NAMES columns.
    labels : pd.DataFrame
        (n_samples, 2) with columns "gk_x", "gk_y" in goal-relative
        coordinates matching the GhostGkModel training domain
        ([0, 30] x [18, 50]).

    Examples
    --------
    >>> features, labels = prepare_ghost_gk_training_data(
    ...     frames, home_team_id=1, actions=actions, subsample_fps=1.0
    ... )
    >>> model = GhostGkModel()
    >>> model.fit(features, labels)
    """
```

**Internal flow:**

1. Build context callbacks from `actions` (if provided):
   - `score_fn = _build_score_lookup(actions, home_team_id)`
   - `phase_fn = _build_phase_lookup(actions)`
2. Infer ball carrier: `carrier = infer_ball_carrier(frames)` then `derive_team_in_possession(frames, carrier)` to get per-frame `ball_carrier_team_id`.
3. Call `_extract_all_ghost_gk_features(frames, home_team_id=home_team_id, carrier=carrier_df, score_at_time=score_fn, phase_at_time=phase_fn, subsample_fps=subsample_fps)`.
4. Extract labels: `labels = meta[["gk_x_gr", "gk_y_gr"]].rename(columns={"gk_x_gr": "gk_x", "gk_y_gr": "gk_y"})`.
5. Drop rows where `gk_x` or `gk_y` is NaN (GK not visible in frame).
6. Validate feature width: assert `features.shape[1] == len(GHOST_GK_FEATURE_NAMES)`.
7. Filter label domain: drop rows where `gk_x` or `gk_y` falls outside `[0, 30] x [18, 50]` (sweeper-keeper rushes, off-pitch artifacts). If any rows are dropped, emit `warnings.warn(f"Dropped {n} of {total} rows with GK outside goal-relative domain (sweeper rushes/artifacts)", stacklevel=2)`.
8. Return `(features, labels)`.

---

## 5. `compute_ghost_gk()` Refactoring

### API change

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,   # NEW
) -> pd.DataFrame:
```

`actions=None` is backward-compatible (produces identical output to 3.19.0 behavior).

### Internal change

Replace the inline iteration loop (lines 816-901) with:

```python
# Build context (same as prepare_ghost_gk_training_data)
score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
phase_fn = _build_phase_lookup(actions) if actions is not None else None
carrier_df = infer_ball_carrier(frames)
carrier = derive_team_in_possession(frames, carrier_df)  # always — only needs frames

features, meta = _extract_all_ghost_gk_features(
    frames, home_team_id=home_team_id,
    carrier=carrier, score_at_time=score_fn, phase_at_time=phase_fn,
)

# Batch predict
densities = resolved.predict_density(features)

# Merge back (same merge logic as current)
```

Ball carrier is always computed via `infer_ball_carrier(frames)` regardless of whether `actions` is provided. This is critical: carrier only needs tracking frames (not events), and gating it on `actions` would create a train/inference mismatch (training always has carrier, inference wouldn't). The O(n) cost of `infer_ball_carrier` is small relative to `extract_ghost_gk_features` which dominates. The `actions` parameter only gates score_diff and phase context.

### Propagation to `add_ghost_gk`

The action-coupled aggregator `add_ghost_gk(actions, frames, ...)` already has access to `actions`. Thread it through to `compute_ghost_gk`:

```python
def add_ghost_gk(actions, frames, *, model=None, home_team_id, links=None, **kwargs):
    # ... existing linkage logic ...
    result = compute_ghost_gk(linked_frames, model=model, home_team_id=home_team_id, actions=actions)
    # ...
```

This means the full action-coupled path automatically gets score/phase/carrier context. Zero API change for `add_ghost_gk` callers.

---

## 6. Bug Fix: `"timestamp"` -> `"time_seconds"`

`_ghost_gk.py` line 342:
```python
# BEFORE
time_s = float(frame_data["timestamp"].iloc[0]) if "timestamp" in frame_data.columns else 0.0
# AFTER
time_s = float(frame_data["time_seconds"].iloc[0]) if "time_seconds" in frame_data.columns else 0.0
```

`_ghost_gk.py` line 837 (in `compute_ghost_gk`, replaced by shared helper):
```python
# Same fix in _extract_all_ghost_gk_features
```

The tracking schema (`TRACKING_FRAMES_COLUMNS`) defines `time_seconds`, not `timestamp`. The current code silently falls back to 0.0 for all frames, making the `time_seconds` feature and velocity `dt` computation incorrect.

---

## 7. Team ID Normalization

`home_team_id` arrives as `str | int` from the caller. Tracking frames store `team_id` with a provider-specific dtype (int64 for StatsBomb/Opta, object/string for Sportec DFL-OBJ-* IDs). The `gk_team == home_team_id` comparison in goal-end assignment is **silent on type mismatch** --- `1 == "1"` is `False` in Python, causing all GKs to be assigned to the wrong goal end.

**Contract:** `_extract_all_ghost_gk_features` normalizes `home_team_id` at entry to match `frames["team_id"].dtype`:

```python
frame_team_dtype = frames["team_id"].dtype
if frame_team_dtype == object:
    home_team_id = str(home_team_id)
else:
    try:
        home_team_id = type(frames["team_id"].iloc[0])(home_team_id)
    except (ValueError, TypeError) as exc:
        raise TypeError(
            f"home_team_id={home_team_id!r} cannot be coerced to "
            f"frames['team_id'] dtype {frame_team_dtype}"
        ) from exc
```

This normalization happens once at the shared helper entry point. Downstream code (score lookup, goal-end assignment) uses the normalized value.

---

## 8. NaN Policy

**Features:** `HistGradientBoostingRegressor` handles NaN features natively (routes NaN to a dedicated bin at each split). No imputation needed. Feature columns may contain NaN when upstream data is incomplete (e.g., no defenders visible for `n_defenders_in_zone`).

**Labels:** Rows with NaN in `gk_x` or `gk_y` (GK not visible) are dropped in `prepare_ghost_gk_training_data()` before returning. `GhostGkModel.fit()` receives no NaN labels.

**Match context:** When `actions=None`, `score_diff` defaults to 0.0 and `phase` defaults to 0 (open play). These are valid feature values, not NaN --- the model learns "no context available" as a distinct state.

---

## 9. Training Script

`scripts/train_ghost_gk.py` --- reference CLI, no lakehouse dependency.

### Arguments

```
--data-dir PATH          Directory of tracking parquets (TRACKING_FRAMES_COLUMNS)
--output-dir PATH        Where to save model artifact (default: models/)
--actions-dir PATH       Optional: directory of SPADL actions parquets
--home-teams PATH        JSON file: {"game_id": "home_team_id", ...}
--subsample-fps FLOAT    Training frame rate (default: 1.0)
--n-estimators INT       HistGBR trees (default: 500)
--max-depth INT          Tree depth (default: 8)
--cv-folds INT           GroupKFold folds (default: 5)
```

### Pipeline

```
1. Load tracking parquets from --data-dir
   - Validate TRACKING_FRAMES_COLUMNS schema
   - Validate vx/vy columns present (raise if missing: "Run smooth_frames + derive_velocities first")
   - Report: n_games, n_frames, providers (from source_provider)

2. Load actions parquets from --actions-dir (if provided)
   - Validate SPADL columns present (type_id or type_name, result_id or result_name)

3. Load home team mapping from --home-teams JSON

4. Per-game feature extraction (pre-built groupby dicts to avoid O(k×n) boolean masks):
   frames_by_game = dict(list(frames.groupby("game_id")))
   actions_by_game = dict(list(actions.groupby("game_id"))) if actions is not None else {}
   for game_id in sorted(frames_by_game):
       game_frames = frames_by_game[game_id]
       game_actions = actions_by_game.get(game_id) if actions is not None else None
       home = home_team_map[str(game_id)]
       feats, labs = prepare_ghost_gk_training_data(
           game_frames, home_team_id=home, actions=game_actions,
           subsample_fps=args.subsample_fps,
       )
       all_features.append(feats)
       all_labels.append(labs)
       all_game_ids.extend([game_id] * len(feats))  # for GroupKFold grouping
       all_providers.extend([game_frames["source_provider"].iloc[0]] * len(feats))
   features = pd.concat(all_features, ignore_index=True)
   labels = pd.concat(all_labels, ignore_index=True)
   groups = np.array(all_game_ids)  # match-level CV splits
   providers = np.array(all_providers)  # for StratifiedGroupKFold

5. StratifiedGroupKFold CV (match-level, provider-stratified):
   - groups = game_ids_per_sample (carried through from step 4)
   - stratification labels = provider per sample (carried through from step 4)
   - Uses `sklearn.model_selection.StratifiedGroupKFold` to ensure each fold
     has proportional representation of each tracking data provider.
     Prevents folds where all Sportec data is in test and all Metrica in train.
   - Per fold: fit, predict, compute MAE (x, y, euclidean), per-provider MAE
   - Report: mean +/- std across folds

6. Feature importance:
   - Train on all data
   - Permutation importance (sklearn.inspection.permutation_importance)
   - Report: top features, any <1% contributors

7. Final model:
   - model.save(output_dir / "ghost_gk_v1")
   - Round-trip verify: GhostGkModel.load() -> predict on sample -> assert matches

8. Metrics summary:
   - Print acceptance criteria pass/fail:
     - Overall mode MAE < 2.0m
     - Per-provider MAE < 3.0m
     - Cross-fold MAE std < 0.5m
     - Artifact size < 15MB
   - Save metrics.json alongside model with schema:
     ```json
     {
       "n_games": 120,
       "n_samples": 54000,
       "n_providers": 3,
       "providers": ["sportec", "metrica", "skillcorner"],
       "cv_folds": 5,
       "subsample_fps": 1.0,
       "hyperparameters": {
         "n_estimators": 500,
         "max_depth": 8
       },
       "cv_mae_x_mean": 1.42,
       "cv_mae_x_std": 0.08,
       "cv_mae_y_mean": 1.31,
       "cv_mae_y_std": 0.07,
       "cv_mae_euclidean_mean": 1.85,
       "cv_mae_euclidean_std": 0.10,
       "per_provider_mae_euclidean": {
         "sportec": 1.78,
         "metrica": 1.92,
         "skillcorner": 1.89
       },
       "acceptance": {
         "overall_mae_lt_2m": true,
         "per_provider_mae_lt_3m": true,
         "cross_fold_std_lt_05m": true,
         "artifact_size_lt_15mb": true
       },
       "artifact_size_bytes": 4200000
     }
     ```
```

### Data format

The script accepts any parquet files in TRACKING_FRAMES_COLUMNS schema. Data sources:

- **Lakehouse:** Export from Databricks `fct_tracking_frames` to parquet.
- **Local kloppy:** `convert_to_frames()` + `smooth_frames()` + `derive_velocities()` on local provider data.
- **Open data:** Metrica sample games, SkillCorner open data.

The `--home-teams` JSON is a simple `{game_id: home_team_id}` mapping. Any data source can produce this.

---

## 10. Publish Script

`scripts/publish_ghost_gk.py` --- upload trained artifact to HuggingFace Hub.

```
--artifact-dir PATH      Model artifact directory (from train script)
--repo-id STR            HF Hub repo (default: karsten-s-nielsen/ghost-gk-v1)
--verify-only            Dry run: verify artifact integrity without uploading
```

### Pipeline

```
1. Load artifact: GhostGkModel.load(artifact_dir)
   - SHA-256 verification runs automatically
   - Predict on a small synthetic sample (sanity check)

2. Upload to HF Hub:
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path=str(artifact_dir),
       repo_id=repo_id,
       repo_type="model",
   )

3. Verify download:
   model = GhostGkModel.from_hub(repo_id)
   # Predict on same synthetic sample, assert identical results
```

The lakehouse can wrap this in its own three-legged delivery (HF Hub + MLflow + UC Volume + model card) using `upload_hf_readme()` and `artifact_deploy`. silly-kicks doesn't need to know about that.

---

## 11. Testing Strategy

### New unit tests (in `tests/tracking/test_ghost_gk.py` or new files)

| Test | Validates |
|------|-----------|
| `test_extract_all_features_shape` | Shared helper returns (n, len(GHOST_GK_FEATURE_NAMES)) features + (n, 6) meta |
| `test_extract_all_features_velocity_state` | Velocity features are non-NaN after first frame |
| `test_extract_all_features_subsample` | Subsampling reduces row count proportionally |
| `test_extract_all_features_goal_relative_coords` | GK positions in meta are in [0, 30] x [18, 50] domain |
| `test_build_score_lookup` | Running score correct at 0-0, after goal, after multiple goals |
| `test_build_score_lookup_no_goals` | Returns 0 for entire game when no goals scored |
| `test_build_phase_lookup` | Returns 1 after freekick, 2 after goalkick, 0 otherwise |
| `test_build_phase_lookup_decay` | Returns 0 when >10s after set piece |
| `test_prepare_training_data_basic` | Returns (features, labels), correct shapes, no NaN labels |
| `test_prepare_training_data_with_actions` | score_diff != 0 when goals present in actions |
| `test_prepare_training_data_without_actions` | Defaults work, score_diff=0, phase=0 |
| `test_prepare_training_data_subsample` | 1fps on 25fps data reduces ~25x |
| `test_compute_ghost_gk_with_actions` | Context flows through, features differ from no-actions case |
| `test_compute_ghost_gk_backward_compat` | actions=None produces identical output to 3.19.0. **Baseline strategy:** The plan's first task computes expected output from the pre-refactor code on the synthetic fixture and saves it as a golden-file `.parquet` test artifact. The post-refactor test loads the golden file and asserts exact equality. |
| `test_time_seconds_not_timestamp` | Bug fix: time_seconds column used, not timestamp |
| `test_add_ghost_gk_threads_actions` | Aggregator passes actions to compute_ghost_gk |
| `test_home_team_id_normalization_int_str` | `home_team_id=1` works when `frames["team_id"]` is string dtype |
| `test_home_team_id_normalization_str_int` | `home_team_id="1"` works when `frames["team_id"]` is int64 dtype |
| `test_build_score_lookup_own_goals` | Own goals attributed to opponent, score_diff correct |
| `test_prepare_training_data_sweeper_rush_filtered` | GK outside [0,30]x[18,50] filtered + warning emitted |
| `test_phase_lookup_excludes_throw_in` | throw_in does not trigger set-piece phase |

### Integration tests

| Test | Validates |
|------|-----------|
| `test_round_trip_train_predict` | prepare_training_data -> fit -> predict round-trip on synthetic data |
| `test_train_script_smoke` | Script runs on small synthetic parquets, produces artifact directory with expected files |

### e2e tests (require trained model / Hub access)

| Test | Validates |
|------|-----------|
| `test_publish_round_trip` | save -> upload -> from_hub -> predict consistency |

---

## 12. File Changes Summary

| File | Change |
|------|--------|
| `silly_kicks/tracking/_ghost_gk.py` | Add `_extract_all_ghost_gk_features`, `_build_score_lookup`, `_build_phase_lookup`, `prepare_ghost_gk_training_data`, `_SET_PIECE_DECAY_SECONDS`. Refactor `compute_ghost_gk` to use shared helper + accept `actions`. Fix `timestamp` -> `time_seconds`. Update `add_ghost_gk` to thread `actions`. Team ID normalization at shared helper entry. |
| `silly_kicks/tracking/__init__.py` | Re-export `prepare_ghost_gk_training_data` (public API). |
| `scripts/train_ghost_gk.py` | Full implementation: data loading, per-game extraction via groupby dicts, StratifiedGroupKFold CV, metrics.json, save. |
| `scripts/publish_ghost_gk.py` | New: artifact verify + HF Hub upload + download verify. |
| `tests/tracking/test_ghost_gk.py` | Add tests for shared helper, context resolution, prepare_training_data, team ID normalization, own-goal score tracking, sweeper-keeper label filtering. |
| `tests/tracking/test_ghost_gk_integration.py` | Add round-trip train/predict test, train script smoke test. |
| `CHANGELOG.md` | Document new public API + bug fix. |
| `TODO.md` | Update TF-18 status. |

---

## 13. Consumer Integration Patterns

### Lakehouse (downstream consumer)

```python
# In lakehouse training script (PEP 723 / HF Jobs):
from silly_kicks.tracking import (
    GhostGkModel, prepare_ghost_gk_training_data,
)

# 1. Query tracking frames from Databricks -> TRACKING_FRAMES_COLUMNS parquet
# 2. Query SPADL actions
# 3. Per-game:
features, labels = prepare_ghost_gk_training_data(
    game_frames, home_team_id=home, actions=game_actions,
)
# 4. CV + fit (lakehouse handles Optuna, cost tracking)
# 5. model.save() -> three-legged publish (HF + MLflow + UC Volume)
```

### Local user (kloppy data)

```python
import silly_kicks.tracking as tracking
from silly_kicks.tracking import (
    GhostGkModel, prepare_ghost_gk_training_data,
)

# 1. Load via kloppy
frames = tracking.convert_to_frames(kloppy_tracking_dataset)
frames = tracking.smooth_frames(frames)
frames = tracking.derive_velocities(frames)
frames = tracking.play_left_to_right(frames, home_team_id=home)

# 2. Prepare training data
features, labels = prepare_ghost_gk_training_data(
    frames, home_team_id=home, subsample_fps=1.0,
)

# 3. Train
model = GhostGkModel()
model.fit(features, labels)
model.save(Path("my_ghost_gk"))
```

### Inference with context (upgraded from 3.19.0)

```python
# Before (3.19.0): score_diff=0, phase=0, carrier=None for all frames
result = compute_ghost_gk(frames, home_team_id=home)

# After: real context when actions available
result = compute_ghost_gk(frames, home_team_id=home, actions=spadl_actions)
```

---

## 14. Acceptance Criteria

- [ ] `prepare_ghost_gk_training_data()` returns correct shapes and types.
- [ ] `compute_ghost_gk(actions=None)` produces identical output to 3.19.0.
- [ ] `compute_ghost_gk(actions=spadl_df)` produces different (better-informed) features.
- [ ] `timestamp` bug fixed --- `time_seconds` used throughout.
- [ ] Training script runs on synthetic parquets end-to-end.
- [ ] Publish script verifies artifact integrity + round-trip.
- [ ] All existing tests pass unchanged.
- [ ] New tests cover shared helper, context resolution, training data assembly.
- [ ] ruff check + ruff format + pyright clean.
