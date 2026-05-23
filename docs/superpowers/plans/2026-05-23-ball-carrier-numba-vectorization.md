# Ball Carrier Numba Vectorization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python `iterrows()` inner loop in `infer_ball_carrier` with a numba `@njit` kernel over pre-indexed dense numpy arrays, achieving ~30-50× speedup (31s→<2s per GS match) while preserving bit-identical output.

**Architecture:** Three-phase pipeline: (1) pre-index pandas → dense numpy arrays with player_id→integer slot mapping, (2) numba `@njit` sequential loop with Python fallback, (3) post-process numpy → pandas DataFrame. The public API is unchanged.

**Tech Stack:** numpy, pandas, numba (optional, already in `[numba]` + `[test]` extras), pytest, pytest-benchmark.

**Spec:** `docs/superpowers/specs/2026-05-23-ball-carrier-numba-vectorization-design.md`

---

### Task 1: Verify baseline — existing tests pass

Before changing anything, confirm the 28 existing tests pass.

**Files:**
- Read: `tests/tracking/test_ball_carrier.py`
- Read: `tests/invariants/test_invariant_ball_carrier.py`

- [ ] **Step 1: Run existing ball carrier tests**

Run: `python -m pytest tests/tracking/test_ball_carrier.py tests/invariants/test_invariant_ball_carrier.py -v --tb=short`
Expected: 28 passed (25 unit + 3 invariant)

---

### Task 2: Write the Python fallback kernel + pre-indexing (TDD red)

Write a test that calls the new `_carrier_loop_numpy` on hand-built dense arrays, then implement the kernel and pre-indexing. This task builds the full pipeline WITHOUT numba.

**Files:**
- Create: `tests/tracking/test_ball_carrier_numba_parity.py`
- Modify: `silly_kicks/tracking/_ball_carrier.py`

- [ ] **Step 1: Write parity test scaffolding and Python kernel test**

Create `tests/tracking/test_ball_carrier_numba_parity.py`:

```python
"""Parity tests: numba kernel produces identical output to Python fallback.

numba is a test dependency (pyproject.toml [test] extra) — these always run in CI.
"""

from __future__ import annotations

import numpy as np
import pytest


def _build_dense_fixture(
    *,
    n_frames: int = 5,
    max_players: int = 3,
    has_velocity: bool = True,
    include_dead: bool = False,
    include_nan_velocity: bool = False,
    n_segments: int = 1,
) -> dict:
    """Build dense pre-indexed arrays for kernel tests.

    Returns a dict of all arrays needed by _carrier_loop_numpy / _carrier_loop_numba.
    """
    bx = np.array([50.0] * n_frames)
    by = np.array([34.0] * n_frames)
    ball_dead = np.zeros(n_frames, dtype=np.bool_)
    if include_dead and n_frames >= 3:
        ball_dead[1] = True  # frame 1 is dead

    px = np.full((n_frames, max_players), np.nan)
    py = np.full((n_frames, max_players), np.nan)
    pvx = np.full((n_frames, max_players), np.nan)
    pvy = np.full((n_frames, max_players), np.nan)
    player_slots = np.full((n_frames, max_players), -1, dtype=np.int64)
    team_slots = np.full((n_frames, max_players), -1, dtype=np.int64)
    n_valid = np.zeros(n_frames, dtype=np.int64)

    for f in range(n_frames):
        # Player 0: at (51, 34) — 1m from ball
        px[f, 0] = 51.0
        py[f, 0] = 34.0
        pvx[f, 0] = 0.0
        pvy[f, 0] = 0.0
        player_slots[f, 0] = 0
        team_slots[f, 0] = 0
        # Player 1: at (52, 34) — 2m from ball
        px[f, 1] = 52.0
        py[f, 1] = 34.0
        pvx[f, 1] = 0.0
        pvy[f, 1] = 0.0
        player_slots[f, 1] = 1
        team_slots[f, 1] = 1
        n_valid[f] = 2

    if include_nan_velocity and n_frames >= 4:
        pvx[3, 0] = np.nan
        pvy[3, 0] = np.nan

    seg_starts = np.array([0], dtype=np.int64)
    seg_ends = np.array([n_frames], dtype=np.int64)
    if n_segments == 2 and n_frames >= 4:
        mid = n_frames // 2
        seg_starts = np.array([0, mid], dtype=np.int64)
        seg_ends = np.array([mid, n_frames], dtype=np.int64)

    return dict(
        bx=bx, by=by, ball_dead=ball_dead,
        px=px, py=py, pvx=pvx, pvy=pvy,
        player_slots=player_slots,
        n_valid=n_valid,
        seg_starts=seg_starts, seg_ends=seg_ends,
        tolerance_m=3.0, beta=0.5, gamma=1.0,
        has_velocity=has_velocity,
    )


class TestPythonKernelBasic:
    def test_basic_winner_selection(self):
        """Closest player wins in distance-only mode."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=1, has_velocity=False)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # player 0 at 1m beats player 1 at 2m
        np.testing.assert_allclose(winner_dist[0], 1.0, atol=1e-9)

    def test_dead_ball_produces_minus_one(self):
        """Dead ball frames produce winner_slot=-1."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=3, include_dead=True, has_velocity=False)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        assert winner_slot[1] == -1
        assert np.isnan(winner_dist[1])

    def test_hysteresis_retains_incumbent(self):
        """Incumbent keeps carrier even when slightly farther."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=2, has_velocity=False)
        # Frame 0: player 0 at 1m wins.
        # Frame 1: swap — player 0 at 2m, player 1 at 1.8m.
        # Difference (0.2m) < gamma (1.0m) → incumbent (0) retained.
        arrays["px"][1, 0] = 52.0  # player 0 now at 2m
        arrays["py"][1, 0] = 34.0
        arrays["px"][1, 1] = 51.8  # player 1 now at 1.8m
        arrays["py"][1, 1] = 34.0
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0
        assert winner_slot[1] == 0  # incumbent retained

    def test_segment_boundary_resets_incumbent(self):
        """New segment resets incumbent — no carry-over across periods."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=4, n_segments=2, has_velocity=False)
        # Segment 0 frames [0,1]: player 0 wins both (closer).
        # Segment 1 frames [2,3]: player 1 closer, should win (no incumbent carry-over).
        arrays["px"][2, 0] = 52.5
        arrays["px"][2, 1] = 50.5
        arrays["px"][3, 0] = 52.5
        arrays["px"][3, 1] = 50.5
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # seg 0
        assert winner_slot[2] == 1  # seg 1 — no incumbent

    def test_nan_velocity_treated_as_zero(self):
        """NaN velocity → 0.0 velocity-toward-ball, not NaN propagation."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=5, include_nan_velocity=True, has_velocity=True)
        winner_slot, winner_dist = _carrier_loop_numpy(**arrays)
        # Frame 3 has NaN velocity on player 0 — should still produce a valid winner
        assert winner_slot[3] >= 0
        assert not np.isnan(winner_dist[3])

    def test_tiebreak_lowest_slot(self):
        """Equal scores → lowest player_slots value wins."""
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy

        arrays = _build_dense_fixture(n_frames=1, has_velocity=False)
        # Both players at same distance
        arrays["px"][0, 0] = 51.0
        arrays["px"][0, 1] = 51.0
        arrays["py"][0, 0] = 34.0
        arrays["py"][0, 1] = 34.0
        winner_slot, _ = _carrier_loop_numpy(**arrays)
        assert winner_slot[0] == 0  # slot 0 < slot 1


class TestPreIndexRoundTrip:
    def test_int_player_ids(self):
        """Integer pid → slot → pid round-trip is identity."""
        from silly_kicks.tracking._ball_carrier import _pre_index_frames
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=52.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = _pre_index_frames(frames)
        slot_to_pid = result["slot_to_pid"]
        pid_to_slot = result["pid_to_slot"]
        for pid, slot in pid_to_slot.items():
            assert slot_to_pid[slot] == pid

    def test_string_player_ids(self):
        """String pid (Sportec DFL-OBJ-*) → slot → pid round-trip is identity."""
        from silly_kicks.tracking._ball_carrier import _pre_index_frames
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            players=[
                dict(pid="DFL-OBJ-0001", tid="T1", x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid="DFL-OBJ-0002", tid="T2", x=52.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = _pre_index_frames(frames)
        slot_to_pid = result["slot_to_pid"]
        pid_to_slot = result["pid_to_slot"]
        for pid, slot in pid_to_slot.items():
            assert slot_to_pid[slot] == pid
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py -v --tb=short`
Expected: FAIL — `_carrier_loop_numpy` and `_pre_index_frames` not defined.

- [ ] **Step 3: Implement `_pre_index_frames` in `_ball_carrier.py`**

Add the following function before `infer_ball_carrier` in `silly_kicks/tracking/_ball_carrier.py`:

```python
def _pre_index_frames(
    frames: pd.DataFrame,
) -> dict:
    """Convert long-form tracking frames to dense numpy arrays for kernel consumption.

    Returns a dict with keys:
        bx, by, ball_dead: (n_frames,) arrays
        px, py, pvx, pvy: (n_frames, max_players) arrays, NaN-padded
        player_slots: (n_frames, max_players) int64, -1 for empty
        n_valid: (n_frames,) int64
        seg_starts, seg_ends: (n_segments,) int64
        frame_meta: (n_frames, 3) — game_id, period_id, frame_id per frame
        slot_to_pid: list — inverse mapping (slot → player_id)
        slot_to_team_id: list — player_slot → team_id direct lookup (O(1) post-process)
        pid_to_slot: dict — forward mapping
        pid_dtype, tid_dtype: dtype — for output casting
        has_velocity: bool
    """
    has_velocity = "vx" in frames.columns and "vy" in frames.columns

    ball_mask = frames["is_ball"] == True  # noqa: E712
    ball_rows = frames[ball_mask]
    player_rows = frames[~ball_mask & frames["x"].notna()]

    # Player/team ID ↔ slot mappings (sorted for deterministic tiebreak)
    unique_pids = sorted(player_rows["player_id"].unique())
    pid_to_slot = {pid: i for i, pid in enumerate(unique_pids)}
    slot_to_pid = list(unique_pids)

    # Direct player_slot → team_id lookup (O(1) in post-process, replaces
    # O(n×m) scan of player_slots/team_slots arrays).
    _pid_tid = (
        player_rows[["player_id", "team_id"]]
        .drop_duplicates(subset=["player_id"])
        .set_index("player_id")["team_id"]
        .to_dict()
    )
    slot_to_team_id = [_pid_tid.get(pid) for pid in unique_pids]

    # Ball position per frame
    ball_pos = (
        ball_rows.groupby(["game_id", "period_id", "frame_id"], dropna=False)
        .agg(bx=("x", "mean"), by=("y", "mean"), bs=("ball_state", "first"))
        .reset_index()
    )

    # Unique frames sorted for stable ordering
    unique_frames = (
        frames[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .sort_values(["game_id", "period_id", "frame_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    n_frames = len(unique_frames)

    frame_ball = unique_frames.merge(
        ball_pos, on=["game_id", "period_id", "frame_id"], how="left"
    )

    # Build frame index for O(1) lookup: (game_id, period_id, frame_id) → row index
    frame_to_idx: dict[tuple, int] = {}
    for i, row in enumerate(unique_frames.itertuples(index=False)):
        frame_to_idx[(row.game_id, row.period_id, row.frame_id)] = i

    # Per-frame ball arrays
    bx_arr = frame_ball["bx"].to_numpy(dtype=np.float64)
    by_arr = frame_ball["by"].to_numpy(dtype=np.float64)
    bs_arr = frame_ball["bs"].to_numpy()
    ball_dead = np.array(
        [(bs == "dead") or np.isnan(bx_arr[i]) or np.isnan(by_arr[i])
         for i, bs in enumerate(bs_arr)],
        dtype=np.bool_,
    )

    # Player groups
    player_groups = dict(
        iter(player_rows.groupby(["game_id", "period_id", "frame_id"]))
    )
    max_players = max((len(g) for g in player_groups.values()), default=0)
    if max_players == 0:
        max_players = 1  # avoid zero-width arrays

    # Dense player arrays
    px = np.full((n_frames, max_players), np.nan)
    py = np.full((n_frames, max_players), np.nan)
    pvx = np.full((n_frames, max_players), np.nan)
    pvy = np.full((n_frames, max_players), np.nan)
    player_slot_arr = np.full((n_frames, max_players), -1, dtype=np.int64)
    n_valid = np.zeros(n_frames, dtype=np.int64)

    for key, group in player_groups.items():
        f_idx = frame_to_idx.get(key)
        if f_idx is None:
            continue
        n = min(len(group), max_players)
        n_valid[f_idx] = n
        px[f_idx, :n] = group["x"].to_numpy(dtype=np.float64)[:n]
        py[f_idx, :n] = group["y"].to_numpy(dtype=np.float64)[:n]
        if has_velocity:
            pvx[f_idx, :n] = group["vx"].to_numpy(dtype=np.float64)[:n]
            pvy[f_idx, :n] = group["vy"].to_numpy(dtype=np.float64)[:n]
        else:
            pvx[f_idx, :n] = 0.0
            pvy[f_idx, :n] = 0.0
        pids = group["player_id"].to_numpy()
        for j in range(n):
            player_slot_arr[f_idx, j] = pid_to_slot.get(pids[j], -1)

    # Segment boundaries: contiguous ranges per (game_id, period_id)
    seg_groups = unique_frames.groupby(
        ["game_id", "period_id"], dropna=False, sort=True
    )
    seg_starts_list = []
    seg_ends_list = []
    for _, seg_idx in seg_groups.groups.items():
        idx_arr = np.asarray(sorted(seg_idx), dtype=np.int64)
        seg_starts_list.append(int(idx_arr[0]))
        seg_ends_list.append(int(idx_arr[-1]) + 1)
    seg_starts = np.array(seg_starts_list, dtype=np.int64)
    seg_ends = np.array(seg_ends_list, dtype=np.int64)

    # Frame metadata for post-processing
    frame_meta_gid = unique_frames["game_id"].to_numpy()
    frame_meta_pid = unique_frames["period_id"].to_numpy()
    frame_meta_fid = unique_frames["frame_id"].to_numpy()

    return dict(
        bx=bx_arr, by=by_arr, ball_dead=ball_dead,
        px=px, py=py, pvx=pvx, pvy=pvy,
        player_slots=player_slot_arr,
        n_valid=n_valid,
        seg_starts=seg_starts, seg_ends=seg_ends,
        frame_meta_gid=frame_meta_gid,
        frame_meta_pid=frame_meta_pid,
        frame_meta_fid=frame_meta_fid,
        slot_to_pid=slot_to_pid,
        slot_to_team_id=slot_to_team_id,
        pid_to_slot=pid_to_slot,
        pid_dtype=frames["player_id"].dtype,
        tid_dtype=frames["team_id"].dtype,
        has_velocity=has_velocity,
    )
```

- [ ] **Step 4: Implement `_carrier_loop_numpy` in `_ball_carrier.py`**

Add the Python fallback kernel:

```python
def _carrier_loop_numpy(
    bx: np.ndarray,
    by: np.ndarray,
    ball_dead: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pvx: np.ndarray,
    pvy: np.ndarray,
    player_slots: np.ndarray,
    n_valid: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    tolerance_m: float,
    beta: float,
    gamma: float,
    has_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Python fallback for carrier loop — identical logic to numba kernel."""
    n_frames = len(bx)
    winner_slot = np.full(n_frames, -1, dtype=np.int64)
    winner_dist = np.full(n_frames, np.nan)
    n_segments = len(seg_starts)

    for s in range(n_segments):
        incumbent = -1
        for f in range(seg_starts[s], seg_ends[s]):
            if ball_dead[f]:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            nv = n_valid[f]
            if nv == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Compute distances for valid players
            dists = np.empty(nv)
            for i in range(nv):
                dx = px[f, i] - bx[f]
                dy = py[f, i] - by[f]
                dists[i] = np.sqrt(dx * dx + dy * dy)

            # Filter to tolerance
            within_mask = dists <= tolerance_m
            if not within_mask.any():
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Build candidate arrays
            cand_indices = np.flatnonzero(within_mask)
            cand_dists = dists[cand_indices]
            scores = cand_dists.copy()

            if has_velocity:
                for ci, i in enumerate(cand_indices):
                    dx = px[f, i] - bx[f]
                    dy = py[f, i] - by[f]
                    d = dists[i]
                    if d > 0:
                        ux = -dx / d
                        uy = -dy / d
                    else:
                        ux = 0.0
                        uy = 0.0
                    vx_val = pvx[f, i]
                    vy_val = pvy[f, i]
                    if np.isnan(vx_val) or np.isnan(vy_val):
                        v_toward = 0.0
                    else:
                        v_toward = vx_val * ux + vy_val * uy
                        if v_toward < 0:
                            v_toward = 0.0
                    scores[ci] = cand_dists[ci] - beta * v_toward

            # Hysteresis
            if incumbent >= 0 and gamma > 0:
                for ci, i in enumerate(cand_indices):
                    if player_slots[f, i] == incumbent:
                        scores[ci] -= gamma
                        break

            # Select best: lowest score, tiebreak by lowest slot
            min_score = scores[0]
            best_ci = 0
            for ci in range(1, len(scores)):
                if scores[ci] < min_score - 1e-12:
                    min_score = scores[ci]
                    best_ci = ci
                elif abs(scores[ci] - min_score) < 1e-12:
                    if player_slots[f, cand_indices[ci]] < player_slots[f, cand_indices[best_ci]]:
                        best_ci = ci
                        min_score = scores[ci]

            best_i = cand_indices[best_ci]
            winner_slot[f] = player_slots[f, best_i]
            winner_dist[f] = cand_dists[best_ci]
            incumbent = player_slots[f, best_i]

    return winner_slot, winner_dist
```

- [ ] **Step 5: Run the parity tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py -v --tb=short`
Expected: 8 passed (6 kernel + 2 round-trip)

---

### Task 3: Rewrite `infer_ball_carrier` to use the three-phase pipeline

Replace the `iterrows` body with calls to `_pre_index_frames`, the kernel, and post-processing.

**Files:**
- Modify: `silly_kicks/tracking/_ball_carrier.py`

- [ ] **Step 1: Add numba import pattern and `_post_process` helper**

Add at the top of `_ball_carrier.py`, after existing imports:

```python
try:
    from ._ball_carrier_numba import _carrier_loop_numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False
```

Add the post-processing helper before `infer_ball_carrier`:

```python
def _post_process(
    winner_slot: np.ndarray,
    winner_dist: np.ndarray,
    pre: dict,
) -> pd.DataFrame:
    """Map kernel output back to a DataFrame with player_id/team_id."""
    result_cols = [
        "game_id", "period_id", "frame_id",
        "ball_carrier_player_id", "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]
    n = len(winner_slot)
    slot_to_pid = pre["slot_to_pid"]
    slot_to_team_id = pre["slot_to_team_id"]
    pid_dtype = pre["pid_dtype"]
    tid_dtype = pre["tid_dtype"]

    carrier_pids = np.empty(n, dtype=object)
    carrier_tids = np.empty(n, dtype=object)
    for i in range(n):
        ws = winner_slot[i]
        if ws < 0:
            carrier_pids[i] = np.nan
            carrier_tids[i] = np.nan
        else:
            carrier_pids[i] = slot_to_pid[ws]
            tid = slot_to_team_id[ws]
            carrier_tids[i] = tid if tid is not None else np.nan

    out = pd.DataFrame({
        "game_id": pre["frame_meta_gid"],
        "period_id": pre["frame_meta_pid"],
        "frame_id": pre["frame_meta_fid"],
        "ball_carrier_player_id": carrier_pids,
        "ball_carrier_distance_m": winner_dist,
        "ball_carrier_team_id": carrier_tids,
    }, columns=result_cols)

    if str(pid_dtype) == "Int64":
        out["ball_carrier_player_id"] = pd.to_numeric(
            out["ball_carrier_player_id"], errors="coerce"
        ).astype("Int64")
    if str(tid_dtype) == "Int64":
        out["ball_carrier_team_id"] = pd.to_numeric(
            out["ball_carrier_team_id"], errors="coerce"
        ).astype("Int64")

    return out
```

- [ ] **Step 2: Rewrite `infer_ball_carrier` body**

Replace the body of `infer_ball_carrier` (everything after the docstring) with the three-phase pipeline. Keep the function signature, docstring, and `result_cols` unchanged:

```python
    result_cols = [
        "game_id", "period_id", "frame_id",
        "ball_carrier_player_id", "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]

    if len(frames) == 0:
        return pd.DataFrame(columns=result_cols)

    has_velocity = "vx" in frames.columns and "vy" in frames.columns
    if not has_velocity:
        warnings.warn(
            "vx/vy columns not found; falling back to distance-only carrier "
            "inference. Call derive_velocities() first for velocity-aware scoring.",
            UserWarning,
            stacklevel=2,
        )

    # Phase 1: pre-index
    pre = _pre_index_frames(frames)

    # Phase 2: kernel
    kernel_args = dict(
        bx=pre["bx"], by=pre["by"], ball_dead=pre["ball_dead"],
        px=pre["px"], py=pre["py"], pvx=pre["pvx"], pvy=pre["pvy"],
        player_slots=pre["player_slots"],
        n_valid=pre["n_valid"],
        seg_starts=pre["seg_starts"], seg_ends=pre["seg_ends"],
        tolerance_m=tolerance_m, beta=beta, gamma=gamma,
        has_velocity=has_velocity,
    )
    if _HAS_NUMBA:
        winner_slot, winner_dist = _carrier_loop_numba(**kernel_args)
    else:
        winner_slot, winner_dist = _carrier_loop_numpy(**kernel_args)

    # Phase 3: post-process
    return _post_process(winner_slot, winner_dist, pre)
```

- [ ] **Step 3: Delete `_nan_row` and `_select_best`**

Remove the `_nan_row` function (lines 242-250 in current file) and `_select_best` function (lines 253-272 in current file). They are no longer called.

- [ ] **Step 4: Run all 28 existing tests**

Run: `python -m pytest tests/tracking/test_ball_carrier.py tests/invariants/test_invariant_ball_carrier.py -v --tb=short`
Expected: 28 passed — output is bit-identical to the old implementation.

- [ ] **Step 5: Run the new parity tests**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py -v --tb=short`
Expected: 8 passed

---

### Task 4: Create the numba `@njit` kernel

Implement the numba kernel in a new file, following the `pitch_control/_numba_kernels.py` pattern.

**Files:**
- Create: `silly_kicks/tracking/_ball_carrier_numba.py`

- [ ] **Step 1: Create `_ball_carrier_numba.py`**

Create `silly_kicks/tracking/_ball_carrier_numba.py`:

```python
"""Optional numba-accelerated kernel for ball-carrier inference.

Mirrors the Python _carrier_loop_numpy in _ball_carrier.py but uses
@numba.njit for ~30-50x speedup on large tracking datasets.

Import pattern:
    try:
        from ._ball_carrier_numba import _carrier_loop_numba
        _HAS_NUMBA = True
    except ImportError:
        _HAS_NUMBA = False
"""

from __future__ import annotations

import numpy as np

try:
    from numba import njit  # type: ignore[import-not-found]
except ImportError as e:
    raise ImportError(
        "numba is required for _ball_carrier_numba. "
        "Install with: pip install silly-kicks[numba]"
    ) from e


@njit(cache=True)
def _carrier_loop_numba(
    bx: np.ndarray,
    by: np.ndarray,
    ball_dead: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pvx: np.ndarray,
    pvy: np.ndarray,
    player_slots: np.ndarray,
    n_valid: np.ndarray,
    seg_starts: np.ndarray,
    seg_ends: np.ndarray,
    tolerance_m: float,
    beta: float,
    gamma: float,
    has_velocity: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-accelerated carrier loop — identical logic to _carrier_loop_numpy.

    Parameters
    ----------
    bx, by : (n_frames,) float64 — ball position per frame
    ball_dead : (n_frames,) bool — True if dead ball or NaN ball position
    px, py : (n_frames, max_players) float64 — player positions, NaN-padded
    pvx, pvy : (n_frames, max_players) float64 — player velocities, NaN-padded
    player_slots : (n_frames, max_players) int64 — player slot indices, -1 empty
    n_valid : (n_frames,) int64 — valid player count per frame
    seg_starts, seg_ends : (n_segments,) int64 — half-open segment ranges
    tolerance_m, beta, gamma : float64 — algorithm parameters
    has_velocity : bool — whether to use velocity scoring

    Returns
    -------
    winner_slot : (n_frames,) int64 — winning player slot (-1 = no carrier)
    winner_dist : (n_frames,) float64 — distance to ball (NaN = no carrier)
    """
    n_frames = len(bx)
    winner_slot = np.full(n_frames, -1, dtype=np.int64)
    winner_dist = np.full(n_frames, np.nan)
    n_segments = len(seg_starts)

    for s in range(n_segments):
        incumbent = -1
        for f in range(seg_starts[s], seg_ends[s]):
            if ball_dead[f]:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            nv = n_valid[f]
            if nv == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
                continue

            # Compute distances
            best_ci = -1
            best_score = 1e30
            best_dist = np.nan
            best_slot = -1

            # First pass: find candidates within tolerance and compute scores
            n_within = 0
            for i in range(nv):
                dx = px[f, i] - bx[f]
                dy = py[f, i] - by[f]
                d = np.sqrt(dx * dx + dy * dy)
                if d > tolerance_m:
                    continue

                score = d
                if has_velocity:
                    if d > 0:
                        ux = -dx / d
                        uy = -dy / d
                    else:
                        ux = 0.0
                        uy = 0.0
                    vx_val = pvx[f, i]
                    vy_val = pvy[f, i]
                    if np.isnan(vx_val) or np.isnan(vy_val):
                        v_toward = 0.0
                    else:
                        v_toward = vx_val * ux + vy_val * uy
                        if v_toward < 0.0:
                            v_toward = 0.0
                    score = d - beta * v_toward

                # Hysteresis
                if incumbent >= 0 and gamma > 0.0 and player_slots[f, i] == incumbent:
                    score -= gamma

                # Select best: lowest score, tiebreak by lowest slot
                slot_i = player_slots[f, i]
                if n_within == 0:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i
                    best_ci = i
                elif score < best_score - 1e-12:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i
                    best_ci = i
                elif abs(score - best_score) < 1e-12 and slot_i < best_slot:
                    best_score = score
                    best_dist = d
                    best_slot = slot_i
                    best_ci = i

                n_within += 1

            if n_within == 0:
                winner_slot[f] = -1
                winner_dist[f] = np.nan
                incumbent = -1
            else:
                winner_slot[f] = best_slot
                winner_dist[f] = best_dist
                incumbent = best_slot

    return winner_slot, winner_dist
```

- [ ] **Step 2: Run numba parity test**

Add the numba parity test class to `tests/tracking/test_ball_carrier_numba_parity.py`:

```python
class TestNumbaParity:
    """Numba kernel must produce bit-identical output to Python fallback."""

    @pytest.mark.parametrize("scenario", [
        dict(n_frames=5, has_velocity=False, include_dead=False, include_nan_velocity=False, n_segments=1),
        dict(n_frames=5, has_velocity=True, include_dead=False, include_nan_velocity=False, n_segments=1),
        dict(n_frames=5, has_velocity=False, include_dead=True, include_nan_velocity=False, n_segments=1),
        dict(n_frames=5, has_velocity=True, include_dead=False, include_nan_velocity=True, n_segments=1),
        dict(n_frames=6, has_velocity=True, include_dead=False, include_nan_velocity=False, n_segments=2),
    ])
    def test_parity(self, scenario):
        from silly_kicks.tracking._ball_carrier import _carrier_loop_numpy
        from silly_kicks.tracking._ball_carrier_numba import _carrier_loop_numba

        arrays = _build_dense_fixture(**scenario)
        numpy_slot, numpy_dist = _carrier_loop_numpy(**arrays)
        numba_slot, numba_dist = _carrier_loop_numba(**arrays)
        np.testing.assert_array_equal(numpy_slot, numba_slot)
        # NaN == NaN comparison for dist
        for i in range(len(numpy_dist)):
            if np.isnan(numpy_dist[i]):
                assert np.isnan(numba_dist[i]), f"Frame {i}: numpy=NaN, numba={numba_dist[i]}"
            else:
                np.testing.assert_allclose(numpy_dist[i], numba_dist[i], rtol=1e-12)
```

- [ ] **Step 3: Run all parity tests**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py -v --tb=short`
Expected: 13 passed (6 Python kernel + 2 round-trip + 5 numba parity)

- [ ] **Step 4: Run all 28 existing tests with numba active**

Run: `python -m pytest tests/tracking/test_ball_carrier.py tests/invariants/test_invariant_ball_carrier.py -v --tb=short`
Expected: 28 passed

---

### Task 5: Fallback path test

Verify the Python fallback produces correct results when numba is unavailable.

**Files:**
- Modify: `tests/tracking/test_ball_carrier_numba_parity.py`

- [ ] **Step 1: Add fallback test class**

Append to `tests/tracking/test_ball_carrier_numba_parity.py`:

```python
from unittest.mock import patch


class TestFallbackPath:
    """Verify Python fallback produces correct results when _HAS_NUMBA=False."""

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_basic_carrier_without_numba(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            ball_x=50.0,
            ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        result = infer_ball_carrier(frames)
        assert result["ball_carrier_player_id"].iloc[0] == 10
        assert result["ball_carrier_team_id"].iloc[0] == 1

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_hysteresis_without_numba(self):
        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _concat_frames, _make_carrier_frame

        f1 = _make_carrier_frame(
            frame_id=1, ball_x=50.0, ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=53.0, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        f2 = _make_carrier_frame(
            frame_id=2, ball_x=50.0, ball_y=34.0,
            players=[
                dict(pid=10, tid=1, x=52.0, y=34.0, vx=0.0, vy=0.0),
                dict(pid=20, tid=1, x=51.8, y=34.0, vx=0.0, vy=0.0),
            ],
        )
        frames = _concat_frames(f1, f2)
        result = infer_ball_carrier(frames, gamma=1.0)
        carriers = result.sort_values("frame_id")["ball_carrier_player_id"].tolist()
        assert carriers == [10, 10]

    @patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False)
    def test_dead_ball_without_numba(self):
        import pandas as pd

        from silly_kicks.tracking._ball_carrier import infer_ball_carrier
        from tests.tracking.test_ball_carrier import _make_carrier_frame

        frames = _make_carrier_frame(
            ball_state="dead",
            players=[dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0)],
        )
        result = infer_ball_carrier(frames)
        assert pd.isna(result["ball_carrier_player_id"].iloc[0])
```

- [ ] **Step 2: Run fallback tests**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py::TestFallbackPath -v --tb=short`
Expected: 3 passed

---

### Task 6: E2e benchmark test

Add the performance benchmark test using pytest-benchmark, marked e2e.

**Files:**
- Modify: `tests/tracking/test_ball_carrier_numba_parity.py`

- [ ] **Step 1: Add e2e benchmark test**

Append to `tests/tracking/test_ball_carrier_numba_parity.py`:

```python
@pytest.mark.e2e
def test_bench_infer_ball_carrier_gs_match(benchmark):
    """Carrier inference for a full GS match should complete in <2s end-to-end.

    Uses pytest-benchmark for statistical rigor. Performance assertion on
    benchmark.stats.stats.mean, matching all existing perf budget tests.
    """
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames("gradientsports")
    result = benchmark(infer_ball_carrier, frames)
    assert len(result) > 0
    assert list(result.columns) == [
        "game_id", "period_id", "frame_id",
        "ball_carrier_player_id", "ball_carrier_distance_m",
        "ball_carrier_team_id",
    ]
    # End-to-end target: <2s per spec. Using 3s budget for CI variance.
    assert benchmark.stats.stats.mean < 3.0
```

- [ ] **Step 2: Add e2e numba-vs-numpy parity test on real GS data**

Append to `tests/tracking/test_ball_carrier_numba_parity.py`:

```python
@pytest.mark.e2e
def test_numba_numpy_parity_real_gs_match():
    """Full GS match: numba and numpy fallback produce identical DataFrames.

    This is the production-scale parity gate — synthetic fixtures cannot catch
    data-shape-dependent divergence between the single-pass numba kernel and
    the two-pass numpy fallback.
    """
    import pandas as pd

    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    from tests.tracking._provider_inputs import load_provider_frames

    frames = load_provider_frames("gradientsports")

    # Run with numba (default when available)
    result_numba = infer_ball_carrier(frames)

    # Run with numpy fallback
    with patch("silly_kicks.tracking._ball_carrier._HAS_NUMBA", False):
        result_numpy = infer_ball_carrier(frames)

    pd.testing.assert_frame_equal(result_numba, result_numpy)
```

- [ ] **Step 3: Verify tests are collected but skipped in non-e2e runs**

Run: `python -m pytest tests/tracking/test_ball_carrier_numba_parity.py --collect-only -q`
Expected: both e2e tests are collected. When run without `-m e2e`, they are skipped/deselected.

---

### Task 7: Lint, type-check, and full test suite

Final verification across all gates.

**Files:**
- All modified files

- [ ] **Step 1: ruff lint**

Run: `python -m ruff check silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/_ball_carrier_numba.py tests/tracking/test_ball_carrier_numba_parity.py`
Expected: All checks passed

- [ ] **Step 2: ruff format**

Run: `python -m ruff format --check silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/_ball_carrier_numba.py tests/tracking/test_ball_carrier_numba_parity.py`
Expected: All already formatted (fix if not)

- [ ] **Step 3: pyright**

Run: `uv run pyright silly_kicks/tracking/_ball_carrier.py tests/tracking/test_ball_carrier_numba_parity.py`
Expected: 0 errors

Note: `_ball_carrier_numba.py` uses `numba` type stubs which pyright may not resolve — this is expected and matches `_numba_kernels.py` (which has `# type: ignore[import-not-found]`).

- [ ] **Step 4: Full ball carrier test suite**

Run: `python -m pytest tests/tracking/test_ball_carrier.py tests/invariants/test_invariant_ball_carrier.py tests/tracking/test_ball_carrier_numba_parity.py -v --tb=short`
Expected: 28 existing + ~16 new = ~44 passed (e2e tests deselected)

---

### Task 8: Commit

Single commit with all changes.

**Files:**
- All modified and new files

- [ ] **Step 1: Stage files**

```bash
git add silly_kicks/tracking/_ball_carrier.py \
       silly_kicks/tracking/_ball_carrier_numba.py \
       tests/tracking/test_ball_carrier_numba_parity.py
```

- [ ] **Step 2: Verify diff looks right**

Run: `git diff --cached --stat`
Expected: 3 files changed (1 modified, 2 new)

- [ ] **Step 3: Commit**

```bash
git commit -m "$(cat <<'EOF'
perf(tracking): vectorize infer_ball_carrier via numba @njit kernel (PR-S50)

Replace iterrows() inner loop with dense-numpy pre-indexing + numba @njit
sequential kernel. Python fallback when numba unavailable. ~30-50x speedup
(31s → <2s per GS match). Enables TC-3 Optuna calibration at practical speed.

Public API unchanged. Output bit-identical to previous implementation.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

**Do NOT commit until /final-review has been run.**
