# SkillCorner Keeper-Origin Resolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop trusting the SkillCorner broadcast ball-event coordinate as a keeper-distribution origin; resolve keeper origins via a provider-aware, detection-aware fallback ladder, keeping full-tracking providers byte-identical.

**Architecture:** A new opt-in `distrust_native_origin` flag on the geometry engine (`resolve_restart_geometry`/`resolve_gk_geometry`, default-off → byte-identical). `compute_xt_gk` decides the policy via a fail-safe provider allowlist and passes the GK-distribution scope down. A detection-aware GK-position helper (±1 s window, `visibility` gate, per-type goal-area clamp) drives the ladder. S1 (transform invariant) and S4 (out-of-region guard) are warn-and-flag + countable, never crash.

**Tech Stack:** Python, pandas, numpy. Tests via pytest. Project conventions: `np.select`/vectorized dispatch, `warnings.warn(stacklevel=2)`, ADR-019 dtype-safe ids, no pandera.

---

> **⚑ EXECUTED + REVISED POST-VALIDATION (4.37.0).** Real-data validation (Databricks bronze + pining)
> revised two things from the tasks below before ship: (1) **distrust scope narrowed to GOAL-KICKS ONLY** —
> open-play GK passes' native origin IS the keeper (validated 0.4 m), so they keep native; the `gk_distribution_mask`
> param and the open-play/`unresolved` branches in Task 3 were removed (distrust operates on `is_gk`). (2) **an
> ADR-028 re-projection** of the detected keeper to action-LTR was added to `_tracking_gk_xy_detected` (away-team
> origins were landing at the wrong end of the pitch; tests now cover home AND away every tier). Everything else
> shipped as planned. Canonical record: ADR-024's 4.37.0 amendment + the design doc's post-validation banner.

---

## ⚠️ Commit policy (overrides the skill's "frequent commits")

Per repo policy: **ONE commit per branch, at the very end, after `/final-review`, and ONLY with explicit owner approval.** Do **NOT** commit per task. Each task ends green (tests pass); the single final commit is Task 11 and is currently **HELD** ("no commits" in effect). Work on a feature branch `pr-s<NN>-skillcorner-keeper-origin` off `main` (PR-S number assigned by the owner; do not guess).

---

## File structure

- `silly_kicks/tracking/_gk_geometry.py` — **modify**: add `native_origin_is_trusted` allowlist; `_tracking_gk_xy_detected` helper; `distrust_native_origin` + `gk_distribution_mask` params on `resolve_restart_geometry`/`resolve_gk_geometry`; the broadcast origin ladder; `flag_native_goalkick_out_of_region` (S4 guard).
- `silly_kicks/tracking/_xt_gk.py` — **modify**: provider→distrust decision; pass flag + mask to `resolve_gk_geometry`; emit `xt_gk_native_goalkick_out_of_region` provenance column; `XtGkReport.n_native_goalkick_out_of_region`.
- `silly_kicks/tracking/skillcorner.py` — **modify**: S1 within-pitch per-row warn-and-flag.
- `silly_kicks/tracking/schema.py` — **modify**: `TrackingConversionReport.n_gross_off_pitch`.
- `tests/tracking/test_gk_geometry_distrust_native.py` — **create**: ladder tiers + allowlist + detected-helper + S4 guard + regression discrimination.
- `tests/tracking/test_xt_gk_distrust_native.py` — **create**: compute_xt_gk provider wiring + coherence (origin feeds pressure/RAV).
- `tests/tracking/test_skillcorner_within_pitch_invariant.py` — **create**: S1 invariant + S2 visibility-survives guard.
- `docs/superpowers/adrs/ADR-024-xt-gk.md` — **modify**: amendment.
- `pyproject.toml` / `silly_kicks/__init__.py` / `TODO.md` / `CHANGELOG.md` — **modify**: version bump (all four must match).

---

## Task 1: Fail-safe provider allowlist (`native_origin_is_trusted`)

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_gk_geometry_distrust_native.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_gk_geometry_distrust_native.py
import pytest
from silly_kicks.tracking._gk_geometry import native_origin_is_trusted


@pytest.mark.parametrize(
    "provider,trusted",
    [
        ("gradientsports", True),
        ("idsse", True),
        ("metrica", True),
        ("sportec", True),
        ("statsbomb", True),
        ("wyscout", True),
        ("GradientSports", True),  # case-insensitive
        ("skillcorner", False),  # broadcast -> distrust
        ("SkillCorner", False),
        (None, False),  # fail-safe: unknown -> distrust
        ("some_future_broadcast", False),  # fail-safe default
    ],
)
def test_native_origin_trust_allowlist(provider, trusted):
    assert native_origin_is_trusted(provider) is trusted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py::test_native_origin_trust_allowlist -v`
Expected: FAIL with `ImportError: cannot import name 'native_origin_is_trusted'`

- [ ] **Step 3: Write minimal implementation**

Add near the top of `_gk_geometry.py` (after the existing module constants, ~line 53):

```python
# Provider-aware native-origin trust (CR 2026-06-30 H1). FAIL-SAFE allowlist: unknown / None /
# future providers default to DISTRUST (route the GK-distribution origin through the detection-aware
# ladder); only KNOWN full-tracking providers are trusted to carry a real keeper origin in start_x.
# A denylist would let a new broadcast source silently corrupt origins (mirrors the access_tier
# privacy default). The regression gate is preserved: every currently-tested full-tracking provider
# is named here -> frozen native-first path -> byte-identical.
_NATIVE_ORIGIN_TRUSTED = frozenset({"gradientsports", "idsse", "metrica", "sportec", "statsbomb", "wyscout"})


def native_origin_is_trusted(provider: str | None) -> bool:
    """True iff ``provider`` is a known full-tracking source whose SPADL ``start_x`` is a real
    keeper position (not a broadcast ball-detection artifact). Unknown / None -> False (distrust)."""
    return provider is not None and str(provider).lower() in _NATIVE_ORIGIN_TRUSTED
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py::test_native_origin_trust_allowlist -v`
Expected: PASS

---

## Task 2: Detection-aware GK-position helper (`_tracking_gk_xy_detected`)

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_gk_geometry_distrust_native.py`

**Context:** mirrors the existing single-frame `_tracking_gk_xy` (L104-129) but (a) searches a ±`window_s` frame window, (b) gates on the keeper's own-row `visibility` being truthy (real detection), (c) picks nearest-in-time with ties → at-or-before, (d) applies the goal-area clamp only when `clamp_goal_area=True`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from silly_kicks.tracking._gk_geometry import _tracking_gk_xy_detected


def _frames_with_gk(detections):
    """detections: list of (frame_id, time_seconds, gk_x, gk_y, visible). One GK (team '1')
    per frame + a ball row. frame_rate 10 Hz."""
    rows = []
    for fid, t, gx, gy, vis in detections:
        rows.append(dict(game_id="g", period_id=1, frame_id=fid, time_seconds=t, player_id="gk1",
                         team_id="1", is_goalkeeper=True, is_ball=False, x=gx, y=gy, visibility=vis,
                         frame_rate=10.0))
        rows.append(dict(game_id="g", period_id=1, frame_id=fid, time_seconds=t, player_id=None,
                         team_id=None, is_goalkeeper=False, is_ball=True, x=50.0, y=34.0, visibility=None,
                         frame_rate=10.0))
    return pd.DataFrame(rows)


def _action(frame_t):
    return pd.DataFrame([dict(game_id="g", period_id=1, action_id=0, team_id="1", player_id="gk1",
                             type_id=0, start_x=25.0, start_y=40.0, end_x=45.0, end_y=30.0,
                             time_seconds=frame_t)])


def test_detected_helper_picks_nearest_visible_in_window():
    # action at t=10.0; detected GK at t=9.9 (x=5.0) visible and t=10.5 (x=8.0) visible.
    # nearest-in-time = 9.9 (|0.1| < |0.5|) -> x=5.0. No clamp (open-play).
    frames = _frames_with_gk([(98, 9.8, 99.0, 1.0, False),   # invisible -> skipped
                              (99, 9.9, 5.0, 33.0, True),     # visible, nearest
                              (105, 10.5, 8.0, 33.0, True)])  # visible, farther
    actions = _action(10.0)
    xy = _tracking_gk_xy_detected(actions, frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.allclose(xy[0], [5.0, 33.0])


def test_detected_helper_ties_break_at_or_before():
    # equidistant visible detections at t=9.5 (x=4) and t=10.5 (x=9); tie -> at-or-before (9.5 -> x=4).
    frames = _frames_with_gk([(95, 9.5, 4.0, 30.0, True), (105, 10.5, 9.0, 30.0, True)])
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.allclose(xy[0], [4.0, 30.0])


def test_detected_helper_clamp_drops_off_position_keeper():
    # goal-kick clamp: a detected keeper at x=20 (> 16.5) -> NaN (falls through to rule point).
    frames = _frames_with_gk([(100, 10.0, 20.0, 34.0, True)])
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=True)
    assert np.isnan(xy[0, 0])


def test_detected_helper_no_visible_detection_returns_nan():
    frames = _frames_with_gk([(100, 10.0, 5.0, 34.0, False)])  # present but not visible
    xy = _tracking_gk_xy_detected(_action(10.0), frames, links=None, window_s=1.0, clamp_goal_area=False)
    assert np.isnan(xy[0, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k detected_helper -v`
Expected: FAIL with `ImportError: cannot import name '_tracking_gk_xy_detected'`

- [ ] **Step 3: Write minimal implementation**

Add to `_gk_geometry.py` (after `_tracking_gk_xy`, ~L129). Reuses `resolve_frame_ids_by_position` for the anchor frame, then expands to a per-period window:

```python
def _tracking_gk_xy_detected(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    links: pd.DataFrame | None,
    *,
    window_s: float = 1.0,
    clamp_goal_area: bool,
) -> np.ndarray:
    """Acting-team GK position resolved from a real broadcast detection within +/- ``window_s`` of
    each action's linked frame. 'Detected' = the GK's OWN frame row has ``visibility`` truthy
    (coerced via :func:`_truthy_bool`; never interpolation). Picks nearest-in-time, ties ->
    at-or-before (origin wants the pre-release keeper position). ``clamp_goal_area`` requires the
    resolved x <= ``_GOAL_AREA_DEPTH`` (goal-kick semantics) else NaN; open-play passes no clamp.
    NaN where no visible detection in the window. NEVER mutates inputs."""
    from ._id_compat import ids_match
    from ._kernels import resolve_frame_ids_by_position

    n = len(actions)
    res = np.full((n, 2), np.nan, dtype=float)
    anchor_fid = resolve_frame_ids_by_position(actions, frames, links=links)

    # Restrict to detected (visible) acting-team GK rows, indexed for per-(game,period) window search.
    gk = frames[
        frames["is_goalkeeper"].astype(bool)
        & (~frames["is_ball"].astype(bool))
        & _truthy_bool(frames["visibility"])
    ].copy()
    if gk.empty:
        return res

    # Map anchor frame_id -> its time_seconds + period, to centre the window in seconds.
    frame_meta = frames.drop_duplicates("frame_id").set_index("frame_id")
    team_ids = actions["team_id"].to_numpy()
    period_ids = actions["period_id"].to_numpy() if "period_id" in actions.columns else np.zeros(n)

    for i in range(n):
        if not np.isfinite(anchor_fid[i]):
            continue
        try:
            t0 = float(frame_meta.at[int(anchor_fid[i]), "time_seconds"])
        except KeyError:
            continue
        cand = gk[
            ids_match(gk["team_id"], team_ids[i])
            & (gk["period_id"].to_numpy() == period_ids[i])
            & (np.abs(gk["time_seconds"].to_numpy(float) - t0) <= window_s)
        ]
        if cand.empty:
            continue
        dt = cand["time_seconds"].to_numpy(float) - t0
        # nearest-in-time; ties -> at-or-before (dt <= 0 preferred). Sort key: (|dt|, dt > 0).
        order = np.lexsort((dt > 0, np.abs(dt)))
        row = cand.iloc[order[0]]
        gx, gy = float(row["x"]), float(row["y"])
        if clamp_goal_area and gx > _GOAL_AREA_DEPTH:
            continue  # off-position -> NaN -> falls to rule point
        res[i] = (gx, gy)
    return res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k detected_helper -v`
Expected: PASS (4 tests)

---

## Task 3: `distrust_native_origin` ladder in the engine

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py` (`resolve_restart_geometry`, `resolve_gk_geometry`)
- Test: `tests/tracking/test_gk_geometry_distrust_native.py`

**Context:** when `distrust_native_origin=True`, GK-distribution rows (identified by `gk_distribution_mask`, an optional bool array aligned to `actions`) skip the native origin tier and run the detection-aware ladder: goal-kick → detected(clamp) → rule-point; open-play GK pass/throw → detected(no-clamp) → unresolved. The frozen path (flag off) is untouched. Destination logic is unchanged for all modes.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
from silly_kicks.tracking._gk_geometry import resolve_gk_geometry

_GOALKICK = 22  # spadlconfig.actiontype_id["goalkick"]
_PASS = 0


def _frames(dets):
    rows = []
    for fid, t, gx, gy, vis in dets:
        rows += [
            dict(game_id="g", period_id=1, frame_id=fid, time_seconds=t, player_id="gk1", team_id="1",
                 is_goalkeeper=True, is_ball=False, x=gx, y=gy, visibility=vis, frame_rate=10.0),
            dict(game_id="g", period_id=1, frame_id=fid, time_seconds=t, player_id=None, team_id=None,
                 is_goalkeeper=False, is_ball=True, x=50.0, y=34.0, visibility=None, frame_rate=10.0),
        ]
    return pd.DataFrame(rows)


def _acts(type_id):
    # native start at x=25 (the scattered broadcast ball-event origin we must NOT trust)
    return pd.DataFrame([dict(game_id="g", period_id=1, action_id=0, team_id="1", player_id="gk1",
                             type_id=type_id, start_x=25.0, start_y=40.0, end_x=45.0, end_y=30.0,
                             time_seconds=10.0)])


def test_distrust_goalkick_detected_uses_tracked_in_box():
    frames = _frames([(100, 10.0, 5.5, 34.0, True)])  # keeper detected in box
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True,
                              gk_distribution_mask=np.array([True]))
    assert out["origin_source"].iloc[0] == "tracking_gk"
    assert np.allclose([out["origin_x"].iloc[0], out["origin_y"].iloc[0]], [5.5, 34.0])


def test_distrust_goalkick_no_detection_falls_to_rule_point():
    frames = _frames([(100, 10.0, 5.5, 34.0, False)])  # not visible
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True,
                              gk_distribution_mask=np.array([True]))
    assert out["origin_source"].iloc[0] == "goalkick_prior"
    assert np.allclose([out["origin_x"].iloc[0], out["origin_y"].iloc[0]], [5.5, 34.0])


def test_distrust_openplay_gk_pass_detected_no_clamp_accepts_sweeper():
    frames = _frames([(100, 10.0, 45.0, 34.0, True)])  # sweeper-keeper at halfway, detected
    out = resolve_gk_geometry(_acts(_PASS), frames=frames, distrust_native_origin=True,
                              gk_distribution_mask=np.array([True]))
    assert out["origin_source"].iloc[0] == "tracking_gk"
    assert np.allclose([out["origin_x"].iloc[0], out["origin_y"].iloc[0]], [45.0, 34.0])


def test_distrust_openplay_gk_pass_no_detection_is_unresolved():
    frames = _frames([(100, 10.0, 45.0, 34.0, False)])  # not visible
    out = resolve_gk_geometry(_acts(_PASS), frames=frames, distrust_native_origin=True,
                              gk_distribution_mask=np.array([True]))
    assert out["origin_source"].iloc[0] == "unresolved"
    assert np.isnan(out["origin_x"].iloc[0])


def test_distrust_never_trusts_native_for_goalkick():
    # keeper detected -> tracked; native x=25 must never be the origin.
    frames = _frames([(100, 10.0, 5.5, 34.0, True)])
    out = resolve_gk_geometry(_acts(_GOALKICK), frames=frames, distrust_native_origin=True,
                              gk_distribution_mask=np.array([True]))
    assert out["origin_x"].iloc[0] != 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k distrust -v`
Expected: FAIL with `TypeError: resolve_gk_geometry() got an unexpected keyword argument 'distrust_native_origin'`

- [ ] **Step 3: Write minimal implementation**

In `resolve_gk_geometry` (L56), thread the two new params through to the engine:

```python
def resolve_gk_geometry(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None,
    links: pd.DataFrame | None = None,
    distrust_native_origin: bool = False,
    gk_distribution_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    g = resolve_restart_geometry(
        actions, frames=frames, links=links, impute_types=(_GOALKICK,),
        distrust_native_origin=distrust_native_origin, gk_distribution_mask=gk_distribution_mask,
    )
    # ... (unchanged label-map + column-rename body below)
```

In `resolve_restart_geometry` (L191), add the params and the broadcast branch. Add to the signature:

```python
def resolve_restart_geometry(
    actions: pd.DataFrame,
    *,
    frames: pd.DataFrame | None = None,
    links: pd.DataFrame | None = None,
    impute_types: tuple[int, ...] | None = None,
    distrust_native_origin: bool = False,
    gk_distribution_mask: np.ndarray | None = None,
) -> pd.DataFrame:
```

Then, immediately AFTER the origin native/unresolved seed (after L237 `oconf = ...`), insert the broadcast override BEFORE the existing `need = ...` tier cascade.

**IMPORTANT (correctness):** the broadcast block resolves the ORIGIN only and must NOT disturb the DESTINATION cascade (distrust is origin-only — the broadcast ball *endpoint* is correct). The existing destination cascade (L284/L292) reads the same `eligible` variable, so we must NOT mutate `eligible`; instead introduce a dedicated `origin_eligible` for the origin tiers.

```python
    # --- Broadcast distrust (CR 2026-06-30 S3): the native origin of a GK-distribution action from a
    # broadcast provider is a ball-detection artifact, NOT the keeper. Resolve those rows via the
    # detection-aware ladder and lock them out of the native-first ORIGIN cascade below (DESTINATION
    # is unchanged -- distrust is origin-only). Goal-kicks: detected(clamp) -> rule point. Open-play
    # GK pass/throw: detected(no-clamp) -> unresolved (no impute). ---
    origin_eligible = eligible  # the origin cascade uses THIS; destination keeps `eligible` (origin-only)
    if distrust_native_origin:
        gk_dist = (
            np.asarray(gk_distribution_mask, dtype=bool)
            if gk_distribution_mask is not None
            else is_gk  # default scope: goal-kicks only
        )
        if frames is not None and gk_dist.any():
            gk_det_clamp = _tracking_gk_xy_detected(actions, frames, links, clamp_goal_area=True)
            gk_det_free = _tracking_gk_xy_detected(actions, frames, links, clamp_goal_area=False)
            for i in np.where(gk_dist)[0]:
                det = gk_det_clamp[i] if is_gk[i] else gk_det_free[i]
                if np.isfinite(det[0]):
                    ox[i], oy[i] = det
                    osrc[i], oconf[i] = "tracking_gk", _CONF_TRACKING_GK
                elif is_gk[i]:
                    ox[i], oy[i] = _RULE_POINT
                    osrc[i], oconf[i] = "restart_prior", _PRIOR_CONF[_GOALKICK]
                else:
                    ox[i], oy[i] = np.nan, np.nan
                    osrc[i], oconf[i] = "unresolved", 0.0
        else:
            # no frames: goal-kicks -> rule point; open-play GK passes -> unresolved.
            for i in np.where(gk_dist)[0]:
                if is_gk[i]:
                    ox[i], oy[i] = _RULE_POINT
                    osrc[i], oconf[i] = "restart_prior", _PRIOR_CONF[_GOALKICK]
                else:
                    ox[i], oy[i] = np.nan, np.nan
                    osrc[i], oconf[i] = "unresolved", 0.0
        # lock broadcast rows out of the ORIGIN cascade only (destination still uses `eligible`).
        origin_eligible = eligible & ~gk_dist
```

Then change the **three** origin-cascade `need` computations to read `origin_eligible` instead of `eligible` (currently L239, L246, L254 — `need = (osrc == "unresolved") & eligible` → `need = (osrc == "unresolved") & origin_eligible`). **Leave the destination cascade's `eligible` references (L284, L292) untouched.** In non-distrust mode `origin_eligible is eligible`, so the path is byte-identical (regression gate preserved).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k distrust -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the whole geometry test file**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -v`
Expected: PASS (all tasks 1-3 tests)

---

## Task 4: S4 out-of-region native goal-kick guard

**Files:**
- Modify: `silly_kicks/tracking/_gk_geometry.py`
- Test: `tests/tracking/test_gk_geometry_distrust_native.py`

**Context:** a `native` goal-kick origin beyond the penalty area (`x > 16.5` LTR own-half) is physically implausible — warn (`stacklevel=2`) and return a per-row bool flag (countable). Never revert/crash. Fires mainly for full-tracking providers / unknown future providers (SkillCorner goalkicks no longer use native).

- [ ] **Step 1: Write the failing test**

```python
import warnings
import numpy as np
import pandas as pd
from silly_kicks.tracking._gk_geometry import flag_native_goalkick_out_of_region

_GOALKICK = 22


def test_s4_flags_and_warns_out_of_region_native_goalkick():
    actions = pd.DataFrame([
        dict(type_id=_GOALKICK), dict(type_id=_GOALKICK), dict(type_id=0),
    ])
    geom = pd.DataFrame({
        "origin_x": [40.0, 5.5, 90.0],          # row0 native far from goal; row2 not a goalkick
        "origin_source": ["native", "native", "native"],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flags = flag_native_goalkick_out_of_region(actions, geom)
    assert list(flags) == [True, False, False]
    assert any("goal-kick" in str(x.message) for x in w)


def test_s4_ignores_imputed_origins():
    actions = pd.DataFrame([dict(type_id=_GOALKICK)])
    geom = pd.DataFrame({"origin_x": [40.0], "origin_source": ["tracking_gk"]})  # not native
    flags = flag_native_goalkick_out_of_region(actions, geom)
    assert list(flags) == [False]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k s4 -v`
Expected: FAIL with `ImportError: cannot import name 'flag_native_goalkick_out_of_region'`

- [ ] **Step 3: Write minimal implementation**

Add to `_gk_geometry.py`:

```python
def flag_native_goalkick_out_of_region(actions: pd.DataFrame, geom: pd.DataFrame) -> np.ndarray:
    """S4 (CR 2026-06-30): per-row bool flag for a NATIVE goal-kick origin beyond the penalty area
    (x > _GOAL_AREA_DEPTH in LTR own-half coords) -- physically implausible (a goal kick is taken
    from the goal area). Warns (data-quality signal pointing upstream); NEVER reverts/crashes (one
    bad row must not fail a match). Countable: caller sums it into an observable report field."""
    tid = actions["type_id"].to_numpy()
    ox = geom["origin_x"].to_numpy(float)
    src = geom["origin_source"].to_numpy().astype(object)
    flags = (tid == _GOALKICK) & (src == "native") & np.isfinite(ox) & (ox > _GOAL_AREA_DEPTH)
    n = int(flags.sum())
    if n:
        warnings.warn(
            f"resolve_gk_geometry: {n} native goal-kick origin(s) beyond the penalty area "
            f"(x > {_GOAL_AREA_DEPTH}); data-quality signal (provider may feed ball-location as "
            "origin -- route through native_origin_is_trusted/the distrust ladder). Not reverted.",
            stacklevel=2,
        )
    return flags
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k s4 -v`
Expected: PASS (2 tests)

---

## Task 5: Wire provider-distrust + S4 flag into `compute_xt_gk`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py`
- Test: `tests/tracking/test_xt_gk_distrust_native.py`

**Context:** `compute_xt_gk` already computes `in_scope = _gk_distribution_mask(...)` (L445) and resolves the provider for the completion variant (`_resolve_completion_for_frames`, L431). Add: resolve the provider string, compute `distrust`, pass `distrust_native_origin=distrust` + `gk_distribution_mask=in_scope` to `resolve_gk_geometry` (L454), and emit the S4 flag as a provenance column.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_xt_gk_distrust_native.py — builds a minimal fitted xT + SkillCorner frames +
# a goalkick whose NATIVE origin is the scattered ball location; asserts the resolved origin is the
# tracked keeper, and that the SAME resolved origin feeds pressure (xt_gk_pressure non-NaN at that
# location). Reuse the existing xt_gk fixture helpers in tests/tracking/ (see test_xt_gk*.py).
import numpy as np
from silly_kicks.tracking._xt_gk import compute_xt_gk
from tests.tracking._xt_gk_fixtures import make_fitted_xt, make_skillcorner_case  # see Step 3 note


def test_skillcorner_goalkick_origin_is_tracked_not_native():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt())
    gk = out[actions["type_id"].to_numpy() == 22].iloc[0]
    assert gk["xt_gk_origin_source"] == "tracking_gk"
    assert abs(gk["xt_gk_origin_x"] - 5.5) < 1e-6
    assert gk["xt_gk_origin_x"] != 25.0


def test_native_goalkick_out_of_region_column_emitted():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt())
    assert "xt_gk_native_goalkick_out_of_region" in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -v`
Expected: FAIL (`KeyError`/`AssertionError` — origin still native; new column absent). If the shared fixture helper does not yet exist, first add `tests/tracking/_xt_gk_fixtures.py` with `make_fitted_xt`/`make_skillcorner_case` modeled on the existing `tests/tracking/test_xt_gk*.py` builders (center-origin SkillCorner frames already in SPADL coords, `source_provider="skillcorner"`, a `visibility=True` GK row near the goal, a goalkick action whose `start_x=native_origin_x`).

- [ ] **Step 3: Write minimal implementation**

In `_xt_gk.py`, add the provider import (top, with other `_gk_geometry` imports inside `compute_xt_gk` — L413):

```python
    from ._gk_geometry import flag_native_goalkick_out_of_region, native_origin_is_trusted, resolve_gk_geometry
```

**L1 (DRY):** the "single real provider, exclude `snapshot`, raise on >1" rule is currently inline in
`_resolve_completion_for_frames` (L366-373). Factor it into one helper so it doesn't drift. Add at module scope
(near `_resolve_completion_for_frames`):

```python
def _resolve_single_provider(frames: pd.DataFrame) -> str | None:
    """The single REAL tracking provider for a one-match frame set (``snapshot`` excluded, C3).
    Raises on >1 (one call = one match = one provider). Returns None when no provider tag is present."""
    provs = []
    if "source_provider" in frames.columns:
        provs = [p for p in pd.unique(frames["source_provider"].dropna()) if str(p).lower() != "snapshot"]
    if len(provs) > 1:
        raise ValueError(
            f"xT-GK: frames span multiple real providers {sorted(map(str, provs))}; one call = one "
            "match = one provider. Pass an explicit completion= model for a mixed/cross-provider stack."
        )
    return str(provs[0]) if provs else None
```

Refactor `_resolve_completion_for_frames` to use it (replace its inline L366-373 block):

```python
    prov = _resolve_single_provider(frames)
    key = variant_key_for_provider(prov)
```

Then in `compute_xt_gk`, resolve the provider once for the distrust decision (after L431):

```python
    resolved_provider = _resolve_single_provider(frames)
    distrust = not native_origin_is_trusted(resolved_provider)
```

**C1 — Chesterton's fence on the >1-provider escape hatch (resolved before implementing).** The old
`_resolve_completion_for_frames` returned early on a `completion=` override, *deliberately skipping* the >1-provider
raise — a documented escape hatch ("Pass an explicit completion= model for a mixed/cross-provider stack"). Calling
`_resolve_single_provider` unconditionally in `compute_xt_gk` for the distrust decision **removes that capability**
(a Hyrum-surface change), not merely "strictly-safer". **Verified dead** (2026-06-30): grep of all `completion=`
call sites — every one passes single-provider (or snapshot) frames; `add_xt_gk` does not thread `completion=`; the
cross-provider `_xtgk_comparability.py` calls `add_xt_gk` **per-match** (single provider). No caller relies on the
escape hatch → **tighten + test + honest note**:
- **(a) Pin the new behavior** with a test (Task 5 Step 5 below): `compute_xt_gk` raises on >1 provider **even with**
  `completion=`; and single-provider + `completion=` still works.
- **(b) Honest CHANGELOG/ADR wording** (Task 11): *"Removed the mixed-provider `completion=` escape hatch; enforced
  one-call-one-match uniformly across the completion AND geometry paths"* — name it as a capability removal, not a
  soft "strictly-safer".

- [ ] **Step 5 (C1): Pin the tightened contract**

```python
import pytest
import pandas as pd
from silly_kicks.tracking._xt_gk import compute_xt_gk
from tests.tracking._xt_gk_fixtures import make_fitted_xt, make_skillcorner_case


def test_multi_provider_raises_even_with_completion_override():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    frames = frames.copy()
    frames.loc[frames.index[: len(frames) // 2], "source_provider"] = "gradientsports"  # 2 real providers
    from silly_kicks.tracking._gk_completion import GkCompletionModel
    model = GkCompletionModel.from_variant("default")
    with pytest.raises(ValueError, match="multiple real providers"):
        compute_xt_gk(actions, frames, xt=make_fitted_xt(), completion=model)


def test_single_provider_with_completion_override_still_works():
    actions, frames = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5)
    from silly_kicks.tracking._gk_completion import GkCompletionModel
    model = GkCompletionModel.from_variant("default")
    out = compute_xt_gk(actions, frames, xt=make_fitted_xt(), completion=model)  # no raise
    assert "xt_gk" in out.columns
```

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -k "multi_provider_raises or single_provider_with_completion" -v`
Expected: PASS (2 tests)

Add the new provenance column to the `out` scaffold (after L443):

```python
    out["xt_gk_native_goalkick_out_of_region"] = np.zeros(len(actions), dtype=bool)
```

Change the `resolve_gk_geometry` call (L454) to:

```python
    geom = resolve_gk_geometry(
        actions, frames=frames, links=pointers,
        distrust_native_origin=distrust, gk_distribution_mask=in_scope,
    )
```

After the provenance assignments (after L464), emit the S4 flag:

```python
    s4_flags = flag_native_goalkick_out_of_region(actions, geom)
    out["xt_gk_native_goalkick_out_of_region"] = s4_flags
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -v`
Expected: PASS

---

## Task 6: Coherence test — resolved origin feeds pressure AND completion

**Files:**
- Test: `tests/tracking/test_xt_gk_distrust_native.py`

**Context:** review low-1. `compute_xt_gk` already feeds `sx/sy` (resolved origin) to `pressure_on_actor` (L511-514) and to `_completion_p` via `geom`/`mask` (L518). Pin it: moving the resolved origin must move `xt_gk_pressure` (and the RAV path), not only `base`/`dzv`.

- [ ] **Step 1: Write the failing test (it should PASS immediately — it pins existing behavior)**

```python
def test_resolved_origin_feeds_pressure_and_rav_not_only_base():
    # two SkillCorner cases identical except the tracked keeper position; EVERY origin-derived term
    # must respond to the resolved origin -- pressure AND the RAV/base/dzv values -- not just base.
    a1, f1 = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=5.5, defender_near=(5.5, 34.0))
    a2, f2 = make_skillcorner_case(native_origin_x=25.0, tracked_gk_x=12.0, defender_near=(5.5, 34.0))
    o1 = compute_xt_gk(a1, f1, xt=make_fitted_xt())
    o2 = compute_xt_gk(a2, f2, xt=make_fitted_xt())
    r1, r2 = o1[o1["xt_gk"].notna()].iloc[0], o2[o2["xt_gk"].notna()].iloc[0]
    assert r1["xt_gk_pressure"] != r2["xt_gk_pressure"]  # origin -> pressure
    assert r1["xt_gk_base"] != r2["xt_gk_base"]          # origin -> base
    # RAV depends on the origin via the completion-probability feature (computed at the resolved
    # origin); a moved origin must move the RAV path too (the whole coherence claim).
    assert r1["xt_gk_rav"] != r2["xt_gk_rav"]            # origin -> RAV/completion
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -k resolved_origin_feeds_pressure_and_rav -v`
Expected: PASS (pins existing wiring — `compute_xt_gk` feeds the resolved origin to pressure (L511-514) and to
`_completion_p`/RAV via `geom`+`mask` (L518)). If it FAILS, the resolved origin is NOT reaching that term —
investigate before proceeding (do not "fix" the test). NOTE on fixture design: ensure the two `tracked_gk_x` values
fall in grid cells / completion-feature regions that actually differ, so the inequality is meaningful (a too-small
delta can land in the same xT cell — choose 5.5 vs 12.0 m to cross a cell boundary).

---

## Task 7: `XtGkReport.n_native_goalkick_out_of_region`

**Files:**
- Modify: `silly_kicks/tracking/_xt_gk.py` (`XtGkReport`)
- Test: `tests/tracking/test_xt_gk_distrust_native.py`

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd
from silly_kicks.tracking._xt_gk import XtGkReport


def test_report_counts_out_of_region_flags():
    df = pd.DataFrame({
        "xt_gk_origin_source": ["native", "tracking_gk"],
        "xt_gk_dest_source": ["native", "next_event"],
        "xt_gk": [0.1, 0.2],
        "xt_gk_native_goalkick_out_of_region": [True, False],
    })
    rep = XtGkReport.from_frame(df)
    assert rep.n_native_goalkick_out_of_region == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -k report_counts -v`
Expected: FAIL with `AttributeError`/`TypeError` (field missing)

- [ ] **Step 3: Write minimal implementation**

Add the field to the `XtGkReport` dataclass (after `spans_multiple_variants`, L76):

```python
    n_native_goalkick_out_of_region: int = 0
```

In `from_frame` (L95 `return cls(...)`), add:

```python
        n_native_goalkick_out_of_region=int(
            df["xt_gk_native_goalkick_out_of_region"].sum()
            if "xt_gk_native_goalkick_out_of_region" in df.columns
            else 0
        ),
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_xt_gk_distrust_native.py -k report_counts -v`
Expected: PASS

---

## Task 8: S1 within-pitch invariant + `TrackingConversionReport.n_gross_off_pitch`

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (`TrackingConversionReport`)
- Modify: `silly_kicks/tracking/skillcorner.py` (`convert_to_frames`)
- Test: `tests/tracking/test_skillcorner_within_pitch_invariant.py`

**Context:** review H2. Per-row gross-off-pitch → `warnings.warn` + count, NEVER crash/clamp. `TOL_XY` provisional (15 m player; ball wider) — flagged for empirical re-calibration on the pining corpus (DGX; §11 of the spec). Hard fail is a batch CI rate-gate (Task 9-adjacent; thresholds deferred), not a per-row assertion.

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_skillcorner_within_pitch_invariant.py
import warnings
import numpy as np
import pandas as pd
from silly_kicks.tracking import skillcorner


def _bronze(extra_player_native_x=None):
    """Minimal SkillCorner bronze (center-origin) with one home + one away player + ball, period 1."""
    base = []
    for pid, tid, gk, x, y in [("p1", "31", True, -50.0, 0.0), ("p2", "40", False, 10.0, 5.0)]:
        base.append(dict(match_id="m", period=1, frame=1, timestamp=0.0, player_id=pid, team_id=tid,
                         is_goalkeeper=gk, x=x, y=y, ball_x=0.0, ball_y=0.0, ball_z=0.0,
                         is_visible=True, frame_rate=10.0))
    if extra_player_native_x is not None:
        base.append(dict(match_id="m", period=1, frame=1, timestamp=0.0, player_id="p3", team_id="40",
                         is_goalkeeper=False, x=extra_player_native_x, y=0.0, ball_x=0.0, ball_y=0.0,
                         ball_z=0.0, is_visible=True, frame_rate=10.0))
    return pd.DataFrame(base)


def test_behind_goal_keeper_within_tolerance_does_not_warn():
    # native x=-60 -> SPADL x=-7.5 (behind goal line) -> within TOL -> no warning, no crash.
    bronze = _bronze(extra_player_native_x=-60.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert not any("off-pitch" in str(x.message) for x in w)
    assert report.n_gross_off_pitch == 0


def test_gross_off_pitch_warns_and_counts_but_does_not_crash():
    # native x=-200 -> SPADL x=-147.5 -> gross off-pitch -> warn + count, NO exception.
    bronze = _bronze(extra_player_native_x=-200.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        frames, report = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert any("off-pitch" in str(x.message) for x in w)
    assert report.n_gross_off_pitch >= 1
    assert len(frames) > 0  # did not crash; row retained (never clamped/dropped)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_skillcorner_within_pitch_invariant.py -k pitch -v`
Expected: FAIL with `AttributeError: 'TrackingConversionReport' object has no attribute 'n_gross_off_pitch'`

- [ ] **Step 3a: Add the report field**

In `schema.py`, add to `TrackingConversionReport` (with a default so existing constructors stay valid):

```python
    n_gross_off_pitch: int = 0
```

- [ ] **Step 3b: Add the invariant in `convert_to_frames`**

In `skillcorner.py`, after the coordinate shift + concat (after L156 `df["time_seconds"] = ...`, before `df.sort_values`), insert:

```python
    # S1 within-pitch invariant (CR 2026-06-30 H2): a correct centre-origin -> SPADL transform keeps
    # players within the pitch except a tolerance for legitimately off-pitch bodies (keepers behind
    # the goal line; out-of-play ball). Per-row GROSS off-pitch -> warn + count; NEVER clamp/crash
    # (one noisy row must not fail a match). A SYSTEMATIC fraction off-pitch is caught by the CI
    # rate-gate (test_skillcorner_within_pitch_invariant). TOL provisional -- re-calibrate from the
    # measured bronze range on the pining corpus (spec section 11).
    _TOL_XY = 15.0  # provisional player tolerance (m); ball gets a wider bound below
    _TOL_BALL = 30.0  # provisional ball tolerance (m); out-of-play balls fly farther
    px, py = df["x"].to_numpy(float), df["y"].to_numpy(float)
    is_ball_arr = df["is_ball"].to_numpy(bool)
    tol = np.where(is_ball_arr, _TOL_BALL, _TOL_XY)
    off = (px < -tol) | (px > 105.0 + tol) | (py < -tol) | (py > 68.0 + tol)
    n_gross_off_pitch = int(np.nansum(off))
    if n_gross_off_pitch:
        warnings.warn(
            f"skillcorner.convert_to_frames: {n_gross_off_pitch} row(s) gross off-pitch beyond "
            f"tolerance (player {_TOL_XY} m / ball {_TOL_BALL} m) -- likely a coordinate-transform "
            "or ingestion bug upstream. Not clamped.",
            stacklevel=2,
        )
```

Add `import warnings` at the top of `skillcorner.py` if absent. Thread the count into the report constructor (the `TrackingConversionReport(...)` call ~L209):

```python
        n_gross_off_pitch=n_gross_off_pitch,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_skillcorner_within_pitch_invariant.py -k pitch -v`
Expected: PASS (2 tests)

---

## Task 9: S2 visibility-survives guard + regression discrimination

**Files:**
- Test: `tests/tracking/test_skillcorner_within_pitch_invariant.py` (S2 guard)
- Test: `tests/tracking/test_gk_geometry_distrust_native.py` (regression discrimination)

- [ ] **Step 1: Write the S2 guard test**

```python
def test_visibility_survives_convert_and_preprocess():
    from silly_kicks.tracking.preprocess import PreprocessConfig
    bronze = _bronze()
    frames, _ = skillcorner.convert_to_frames(bronze, home_team_id="31")
    assert "visibility" in frames.columns
    # the home GK row was is_visible=True in bronze -> truthy visibility after convert
    gk = frames[(frames["player_id"] == "p1")]
    assert bool(gk["visibility"].iloc[0]) is True
    # preprocess (smoothing) must not drop/blank visibility
    frames_pp, _ = skillcorner.convert_to_frames(
        bronze, home_team_id="31", preprocess=PreprocessConfig(smoothing_method="ema"),
    )
    assert "visibility" in frames_pp.columns
    assert frames_pp["visibility"].notna().any()
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/tracking/test_skillcorner_within_pitch_invariant.py -k visibility -v`
Expected: PASS (visibility already carried per `skillcorner.py` L109; this pins it). If preprocess blanks it, narrow the preprocess column set to exclude `visibility` and re-run.

- [ ] **Step 3: Write the regression discrimination test**

```python
def test_full_tracking_path_byte_identical_to_no_flag():
    # GS-style frames (trusted provider): distrust must be OFF -> identical to the frozen call.
    frames = _frames([(100, 10.0, 5.5, 34.0, True)])  # any frames
    frames["source_provider"] = "gradientsports"
    acts = _acts(_GOALKICK)
    acts["start_x"] = 5.5  # native (trusted) origin present
    baseline = resolve_gk_geometry(acts, frames=frames)  # default flag off
    trusted = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=False,
                                  gk_distribution_mask=np.array([True]))
    pd.testing.assert_frame_equal(baseline, trusted)
    assert trusted["origin_source"].iloc[0] == "native"  # trusted provider keeps native


def test_distrust_changes_origin_only_destination_byte_identical():
    # M1: the riskiest edit (origin_eligible vs eligible split) makes "distrust is origin-only" true.
    # Pin it: same broadcast goal-kick + frames, distrust True vs False -> ONLY origin columns differ;
    # destination columns (dest_x/_y/dest_source) byte-identical. Catches a future wrong-`eligible` edit.
    frames = _frames([(100, 10.0, 5.5, 34.0, True)])
    acts = _acts(_GOALKICK)
    acts["end_x"], acts["end_y"] = np.nan, np.nan  # force destination onto the next_event/ladder path
    nxt = _acts(_PASS); nxt["action_id"] = 1; nxt["start_x"], nxt["start_y"] = 60.0, 30.0
    acts = pd.concat([acts, nxt], ignore_index=True)
    mask = np.array([True, False])
    off = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=False, gk_distribution_mask=mask)
    on = resolve_gk_geometry(acts, frames=frames, distrust_native_origin=True, gk_distribution_mask=mask)
    dest_cols = ["dest_x", "dest_y", "dest_source"]
    pd.testing.assert_frame_equal(off[dest_cols], on[dest_cols])  # destination untouched by distrust
    assert not off["origin_source"].equals(on["origin_source"])   # origin DID change (sanity)
```

- [ ] **Step 4: Run it**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py -k "byte_identical or origin_only_destination" -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full new test suite + the existing xt_gk + geometry regression suites**

Run: `python -m pytest tests/tracking/test_gk_geometry_distrust_native.py tests/tracking/test_xt_gk_distrust_native.py tests/tracking/test_skillcorner_within_pitch_invariant.py -v`
Then the existing parity suites (must stay green):
Run: `python -m pytest tests/tracking/ -k "xt_gk or gk_geometry or restart or skillcorner" -m "not e2e" -v`
Expected: ALL PASS (frozen GS/idsse/metrica behavior unchanged)

---

## Task 10: Full quality gate (Shift Left)

**Files:** none (verification only)

- [ ] **Step 1: Full test suite (no e2e)**

Run: `python -m pytest tests/ -m "not e2e" --tb=short`
Expected: PASS

- [ ] **Step 2: Lint + format + types (match CI exactly)**

Run: `ruff format --check . && ruff check . && pyright silly_kicks/`
Expected: clean (pyright over the FULL package, not just changed files — per repo memory)

- [ ] **Step 3: NaN-safety + liveness + dup-action_id auto-enumerating gates**

Run: `python -m pytest tests/test_enrichment_nan_safety.py tests/tracking/test_aggregator_column_liveness.py tests/tracking/test_frame_aware_xfns_dup_action_id.py tests/test_add_star_purity.py -v`
Expected: PASS (no new `add_*` surface added here, but these auto-discover — confirm green)

---

## Task 11: Version bump, ADR amendment, CHANGELOG, TODO — then HELD commit

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md`, `docs/superpowers/adrs/ADR-024-xt-gk.md`

**Context:** minor version bump (xt_gk serve-output change for SkillCorner; not a forced VAEP retrain — xt_gk is opt-in). All four version locations must match (hard gate). ADR-024 amendment documents the provider-aware origin-trust contract.

- [ ] **Step 1: Bump the version in all four locations to the agreed `X.Y.0`** (owner assigns the exact number alongside the PR-S number). Confirm they match:

Run: `grep -R "version" pyproject.toml | head -1; grep "__version__" silly_kicks/__init__.py`
Expected: identical `X.Y.0`

- [ ] **Step 2: Add a CHANGELOG entry + a TODO On-Deck/shipped update** (per repo grooming rules: delete shipped rows, no strikethrough; CHANGELOG is the record). Add an ADR-024 amendment paragraph describing: provider-aware native-origin trust (fail-safe allowlist), the detection-aware ladder by action type, S4 warn-and-count, S1 warn-and-count, regression-gate guarantee, lakehouse re-materialize trigger (SkillCorner GK distributions). **C1 (Hyrum-surface — name it explicitly):** "Removed the mixed-provider `completion=` escape hatch; `compute_xt_gk` now enforces one-call-one-match uniformly across the completion AND geometry paths (a >1-provider frame set raises even with `completion=`). Verified no caller relied on it."

- [ ] **Step 2b (M2): record the deferred rate-gates as explicit TODO.md follow-up items** (so they LAND, not drift): (i) "S1 SkillCorner off-pitch rate-gate — measure bronze off-pitch rate on the pining corpus, set `TOL_XY`/ball bound + CI threshold, wire batch gate"; (ii) "S4 out-of-region native-goalkick rate-gate — measure corpus rate, wire CI gate". Both reference this plan + ADR-024.

- [ ] **Step 3: Run `/final-review`** (mandatory before the single commit).

- [ ] **Step 4: HELD — do NOT commit.** "No commits" is currently in effect. When the owner approves, make the SINGLE commit:

```bash
git add -A
git commit -m "feat(tracking): SkillCorner keeper-origin resolution (broadcast distrust) -- silly-kicks X.Y.0 (ADR-024, PR-S<NN>)"
```

(Commit message per repo convention; the harness appends the Co-Authored-By + session trailers.)

---

## Deferred (empirical, NOT pre-built — see spec §11)

> **M2 (review):** the per-row warns are the *alarm*; the standing rate-gates are the *smoke detector*. A count that
> nothing routinely checks is the silent-guard failure mode that produced the stale grid. So the rate-gate
> THRESHOLDS are deferred (they need the measured corpus rate) but the gates themselves are a **tracked follow-up
> that MUST land** — recorded as an explicit TODO.md item in Task 11, not an open-ended "later". Sequence: measure
> corpus rate (DGX) → set threshold with margin → wire the CI/batch gate that hard-fails a systematic break.

- **S1 `TOL_XY` (player) + ball bound + rate-gate:** re-measure the real bronze player+ball off-pitch range on the
  pining corpus (DGX); replace the provisional 15 m / 30 m; then wire the batch/CI gate hard-failing when the
  off-pitch rate exceeds the measured baseline + margin (the 123-type transform break). **Tracked follow-up.**
- **S4 out-of-region rate-gate:** measure the out-of-region native-goalkick rate on the corpus; wire a CI gate
  asserting no provider exceeds a small % (catches a provider silently feeding ball-as-origin). **Tracked follow-up.**
- **Open-play own-half misdetection bound (spec §3.4):** validate-then-maybe — after the fix, check whether SkillCorner pass origins localize to the own half; only add a generous own-half bound if attacking-half origins persist (beyond → `unresolved`/flagged, never clamped).
- **`window_s` / at-or-before bias:** tunable; defaults `1.0` / ties-earlier ship as-is.
- **Perf (review low-2):** the `_tracking_gk_xy_detected` per-row window loop mirrors the existing `_tracking_gk_xy` cost profile (~100 keeper actions/match). Measure-before-optimize: if it shows as a hotspot on the full pining corpus, vectorize the window search (groupby per `(game,period)` + `searchsorted` on sorted `time_seconds`). Do NOT pre-optimize.
- **Acceptance validation on real data:** goal-kick origins ≈ 100% own-box; pass origins localize; scatter SD collapses (DGX, against the pining SkillCorner matches).
```
