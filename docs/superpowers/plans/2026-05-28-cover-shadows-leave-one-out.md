# Cover Shadows leave-one-out perf refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `max_single_defender_blocking_score` (the `detailed=False` cover-shadow feature) dramatically faster by hoisting the redundant per-defender man-marking re-classification, precomputing per-player interception probabilities once per receiver, and computing the leave-one-out as a single vectorized re-scan — **bit-identical to current behavior within `rtol 1e-10`**.

**Architecture:** The current lightweight branch runs a full `lane_control` (man-marker classify + 3-lane survival scan) per `(lane-blocker, receiver)` pair — `O(blockers × receivers)`. Because removing a lane-blocker provably never changes the greedy man-marker set (spec §2.1, no-ripple), the racer set is fixed; we classify once, precompute per-player `p_int` once per receiver, and re-run only the clamped survival recurrence with each blocker's row masked, vectorized over the blocker axis. Validated against an independent test-vendored frozen oracle.

**Tech Stack:** Python, numpy, pandas, pytest, Hypothesis. Spec: `docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md`.

---

## File Structure

- `silly_kicks/tracking/_cover_shadows.py` — **modify**. Refactor `_compute_lane_probabilities` into `_lane_int_probs` + `_lane_received_survival`; add `_lane_received_batched`; rewrite the `detailed=False` branch of `_compute_cover_shadow_dict`.
- `tests/tracking/_cover_shadows_reference.py` — **create**. Test-vendored frozen oracle (independent of the new helpers): a frozen copy of the pre-refactor survival scan + a reference `max_single` computation.
- `tests/tracking/test_cover_shadows.py` — **modify**. No-ripple guard (already present); add the exactness test against the oracle, perf-budget guard, atomic parity.
- `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md` — **modify** (Task 8).

**Invariants the implementation must hold (from the spec):**
- **INV-1:** never subtract a contribution from the *clamped accumulation*. The leave-one-out re-runs the clamped recurrence with the blocker's `p_int` row masked. (Adjusting the per-step *player sum* by excluding a row is fine; subtracting from the post-clamp running total is not.)
- All five `_CS_COL_NAMES` columns stay bit-stable (within `rtol 1e-10`); `n_blocked_receivers` especially, since it flows through `_compute_lane_probabilities` via `lane_control`.

---

## Task 1: Confirm the no-ripple invariant guard is green (already implemented)

The load-bearing property test + adversarial fixtures already exist in `tests/tracking/test_cover_shadows.py` (`TestManMarkerInvariantUnderLaneBlockerRemoval`). This task only verifies them — do not rewrite them.

**Files:**
- Test: `tests/tracking/test_cover_shadows.py::TestManMarkerInvariantUnderLaneBlockerRemoval`

- [ ] **Step 1: Run the no-ripple guard and confirm green**

Run: `python -m pytest tests/tracking/test_cover_shadows.py::TestManMarkerInvariantUnderLaneBlockerRemoval -q`
Expected: `5 passed` (3 adversarial fixtures + Hypothesis property test + guard-the-guard).

This is the foundation: the entire refactor is bit-identical *because* removing a lane-blocker never changes the man-marker set. If this is red, stop — the premise is broken.

---

## Task 2: Capture the performance baseline (before any change)

Measure-before-optimize: record the current timing so the Task 7 budget is grounded in a real number, captured on un-changed code.

**Files:**
- None modified (measurement only; record the number in the PR description and in Task 7's `_BUDGET`).

- [ ] **Step 1: Measure current `_compute_cover_shadow_dict(detailed=False)` timing**

Create a throwaway script `_perf_baseline.py` at repo root:

```python
import time
import numpy as np
from tests.tracking._gk_test_helpers import _make_two_team_frame
from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict
from silly_kicks.xthreat import ExpectedThreat

xt = ExpectedThreat()
xt.fit(  # minimal fit so the interpolator is usable; mirrors tests' fitted_xt
    np.array([[10.0, 34.0], [50.0, 34.0], [90.0, 34.0], [70.0, 30.0], [75.0, 40.0]]),
    np.array([0.01, 0.05, 0.4, 0.3, 0.35]),
)
frame = _make_two_team_frame(
    home_positions=[(55.0, 30.0), (58.0, 35.0), (52.0, 28.0), (62.0, 40.0), (57.0, 32.0),
                    (60.0, 38.0), (65.0, 25.0), (20.0, 15.0), (25.0, 55.0), (30.0, 10.0)],
    away_positions=[(50.0, 34.0), (75.0, 34.0), (80.0, 25.0), (85.0, 45.0), (70.0, 20.0),
                    (70.0, 48.0), (90.0, 30.0), (95.0, 40.0), (45.0, 15.0), (45.0, 55.0)],
)
N = 50
t0 = time.perf_counter()
for _ in range(N):
    _compute_cover_shadow_dict(frame, (50.0, 34.0), 2, xt, home_team_id=1, detailed=False)
dt = (time.perf_counter() - t0) / N
print(f"baseline per-call: {dt*1000:.2f} ms")
```

Run: `python _perf_baseline.py`
Expected: prints a per-call millisecond figure. **Record it** (the PR will report speedup; Task 7's budget is derived from the post-change figure × 1.5).

> If `ExpectedThreat.fit` signature differs, mirror the `fitted_xt` fixture in `tests/tracking/test_cover_shadows.py` (search for `def fitted_xt`) — use the same construction. Do not block on a perfect fit; any usable interpolator gives a representative timing.

- [ ] **Step 2: Capture PRE-change real-match `max_single` values (for the Task 7 diff)**

The real-match bit-identicality confirmation must compare pre- vs post-change values, so the
**pre-change** values must be captured **now**, before any code change (capturing them at Task 7
would require git stash/checkout gymnastics on already-refactored code).

If a real match is available locally (e.g. a WC2018 / Gradient Sports fixture per the repo's
datasets), run `add_cover_shadows(actions, frames, xt, home_team_id=..., detailed=False)` over it
and persist the `max_single_defender_blocking_score` column to disk:

```python
# in a throwaway _prematch_capture.py — adapt loader to the available local match
enriched = add_cover_shadows(actions, frames, xt, home_team_id=HOME, detailed=False)
enriched[["action_id", "max_single_defender_blocking_score"]].to_parquet("_prematch_maxsingle.parquet")
```

Keep `_prematch_maxsingle.parquet` (gitignored / not committed) for the Task 7 diff. **If no real
match is available locally, record that explicitly** — Task 7 Step 3 will then rely on the
synthetic exactness test + no-ripple proof, and the PR must say so (no silent skip).

- [ ] **Step 3: Delete the throwaway scripts**

Run: `rm _perf_baseline.py` (keep `_prematch_maxsingle.parquet` if created).
(The committed timing guard lands in Task 7; these scripts are investigation-only.)

---

## Task 3: Vendored frozen reference oracle + value-level exactness test

Build an oracle that shares **none** of the new helpers (spec §6.1): a frozen copy of the current survival scan plus a reference `max_single`. Asserting current production `== reference` here proves the oracle faithfully reproduces today's values (and confirms value-level no-ripple) *before* any production change.

**Files:**
- Create: `tests/tracking/_cover_shadows_reference.py`
- Test: `tests/tracking/test_cover_shadows.py` (new class `TestLeaveOneOutExactness`)

- [ ] **Step 1: Create the frozen oracle module**

Create `tests/tracking/_cover_shadows_reference.py`:

```python
"""Frozen, independent oracle for the cover-shadows leave-one-out (PR-S65).

!!! FROZEN at pre-PR-S65 behavior. Do NOT update this file to track production
!!! changes. Its entire value is being an INDEPENDENT reimplementation: keeping it
!!! "in sync" with production would make the exactness test circular and worthless.
!!! If production ever legitimately changes the value, that is a NEW spec and a NEW
!!! oracle — never an edit here.

`_reference_lane_probabilities` is a verbatim copy of the PRE-refactor
`_cover_shadows._compute_lane_probabilities` (the sequential survival scan).
`_reference_max_single` reproduces the `detailed=False` max_single computation
using that frozen scan with the man-marker set classified ONCE on the full frame
(fixed cast). It shares none of the new production helpers, so asserting
production == reference validates the helper extraction AND the vectorization
against independent code. See spec §6.1.
"""
from __future__ import annotations

import numpy as np

from silly_kicks.tracking._cover_shadows import (
    CoverShadowParams,
    _classify_man_markers,
    ball_drag_time,
    player_tti,
)


def _reference_lane_probabilities(targets, defender_pos, defender_vel, attacker_pos, attacker_vel, *, params):
    """Frozen verbatim copy of pre-refactor _compute_lane_probabilities."""
    n_points = targets.shape[0]
    d_from_passer = np.linalg.norm(targets - targets[0:1], axis=1)
    t_ball = ball_drag_time(d_from_passer, params)
    tti_def = player_tti(defender_pos, defender_vel, targets, is_defender=True, params=params)
    tti_att = player_tti(attacker_pos, attacker_vel, targets, is_defender=False, params=params)
    s = np.sqrt(3.0) * params.sigma / np.pi

    def _p_int(tti_matrix):
        dt = t_ball[np.newaxis, :] - tti_matrix
        return 1.0 / (1.0 + np.exp(-dt / s))

    p_int_def = _p_int(tti_def)
    p_int_att = _p_int(tti_att)
    p_blocked = 0.0
    p_received = 0.0
    p_anyone_prior = 0.0
    for k in range(1, n_points):
        dt_k = t_ball[k] - t_ball[k - 1]
        if dt_k <= 0:
            continue
        p_ctrl = 1.0 - np.exp(-params.lambda_ctrl * dt_k)
        total_contrib_k = 0.0
        for j in range(len(defender_pos)):
            contrib = float(p_int_def[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_blocked += contrib
            total_contrib_k += contrib
        for j in range(len(attacker_pos)):
            contrib = float(p_int_att[j, k]) * p_ctrl * (1.0 - p_anyone_prior)
            p_received += contrib
            total_contrib_k += contrib
        p_anyone_prior = min(p_anyone_prior + total_contrib_k, 1.0)
    return p_blocked, p_received


def _reference_max_single(frame_data, passer_xy, attacking_team_id, xt, *, home_team_id):
    """Reference max_single_defender_blocking_score via frozen scan + fixed cast."""
    p = CoverShadowParams()
    players = frame_data[~frame_data["is_ball"].astype(bool)]
    attackers = players[players["team_id"] == attacking_team_id]
    attackers_outfield = attackers[~attackers["is_goalkeeper"].astype(bool)]
    ball_rows = frame_data[frame_data["is_ball"].astype(bool)]
    ball_x = float(ball_rows.iloc[0]["x"])
    attacking_high = str(attacking_team_id) == str(home_team_id)
    dangerous = (
        attackers_outfield[attackers_outfield["x"] > ball_x]
        if attacking_high
        else attackers_outfield[attackers_outfield["x"] < ball_x]
    )
    if len(dangerous) == 0:
        return 0.0

    defenders_outfield = players[
        (players["team_id"] != attacking_team_id) & (~players["is_goalkeeper"].astype(bool))
    ]
    goal_x_own = 105.0 if attacking_high else 0.0
    man_markers = _classify_man_markers(defenders_outfield, attackers, goal_x_own=goal_x_own, params=p)
    lane_blocker_ids = [pid for pid in defenders_outfield["player_id"] if pid not in man_markers]
    if not lane_blocker_ids:
        return 0.0

    xt_interp = xt.interpolator()
    passer = np.array(passer_xy, dtype=np.float64)
    att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
    att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)

    max_def = 0.0
    for d_pid in lane_blocker_ids:
        kept = defenders_outfield[defenders_outfield["player_id"].isin(lane_blocker_ids)]
        full_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
        full_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
        keep_mask = (kept["player_id"] != d_pid).to_numpy()
        score_d = 0.0
        for _, recv in dangerous.iterrows():
            recv_x = float(recv["x"]); recv_y = float(recv["y"])
            recv_xt = float(xt_interp(np.array([recv_x]), np.array([recv_y]))[0, 0])
            receiver = np.array([recv_x, recv_y], dtype=np.float64)
            pass_vec = receiver - passer
            pass_dist = np.linalg.norm(pass_vec)
            if pass_dist < 1e-6:
                continue
            u = pass_vec / pass_dist
            u_perp = np.array([-u[1], u[0]])
            half_width = p.cone_width_factor * pass_dist / 2.0
            t = np.linspace(0.0, 1.0, p.n_sample_points)
            center = passer[np.newaxis, :] + t[:, np.newaxis] * pass_vec[np.newaxis, :]
            left = center + t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            right = center - t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            old_recv = 0.0
            new_recv = 0.0
            for lane in (center, left, right):
                _, base_rec = _reference_lane_probabilities(lane, full_pos, full_vel, att_pos, att_vel, params=p)
                _, loo_rec = _reference_lane_probabilities(
                    lane, full_pos[keep_mask], full_vel[keep_mask], att_pos, att_vel, params=p
                )
                old_recv += base_rec
                new_recv += loo_rec
            score_d += recv_xt * max(new_recv - old_recv, 0.0)
        max_def = max(max_def, score_d)
    return max_def
```

- [ ] **Step 2: Write the exactness test (current production == reference)**

Add to `tests/tracking/test_cover_shadows.py`:

```python
class TestLeaveOneOutExactness:
    """Production max_single == independent frozen oracle, within rtol 1e-10 (spec §6.1)."""

    def _fixture(self):
        return _make_two_team_frame(
            home_positions=[(55.0, 30.0), (58.0, 35.0), (52.0, 28.0), (62.0, 40.0), (57.0, 32.0),
                            (60.0, 38.0), (65.0, 25.0), (20.0, 15.0), (25.0, 55.0), (30.0, 10.0)],
            away_positions=[(50.0, 34.0), (75.0, 34.0), (80.0, 25.0), (85.0, 45.0), (70.0, 20.0),
                            (70.0, 48.0), (90.0, 30.0), (95.0, 40.0), (45.0, 15.0), (45.0, 55.0)],
        )

    def test_production_matches_frozen_oracle(self, fitted_xt):
        from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict
        from tests.tracking._cover_shadows_reference import _reference_max_single

        frame = self._fixture()
        prod = _compute_cover_shadow_dict(frame, (50.0, 34.0), 2, fitted_xt, home_team_id=1, detailed=False)
        ref = _reference_max_single(frame, (50.0, 34.0), 2, home_team_id=1, xt=fitted_xt)
        assert prod is not None
        np.testing.assert_allclose(
            prod["max_single_defender_blocking_score"], ref, rtol=1e-10,
            err_msg="production max_single diverged from the frozen leave-one-out oracle",
        )
```

> Use the existing `fitted_xt` fixture (search `def fitted_xt` in this file). If it is module-scoped/class-local, move/duplicate its construction so `TestLeaveOneOutExactness` can request it.

- [ ] **Step 3: Run the exactness test against CURRENT (un-refactored) code — THE LYNCHPIN**

Run: `python -m pytest tests/tracking/test_cover_shadows.py::TestLeaveOneOutExactness -q`
Expected: PASS. (Proves the oracle reproduces current behavior and that the fixed-cast/decoupled computation already equals current — value-level no-ripple confirmation before touching production.)

**This step is the lynchpin of the entire validation.** Passing here on *un-refactored* production is what certifies the reference's fidelity to the original — its attacker race set (all attacking players incl. GK, matching `lane_control`'s `attackers_all`), its defender/lane-blocker set, lane geometry, and params. **If this step fails, the REFERENCE is wrong — fix the reference. NEVER adjust the reference to make a later (post-refactor) mismatch pass.** Once this step is green, the reference is frozen: any subsequent mismatch (Task 4/5) is a *production* bug, not an oracle bug. Editing the oracle to pass after the refactor would make the golden-master circular and worthless.

- [ ] **Step 4: Commit checkpoint** — *(no commit; per project flow all work rides one end-of-PR commit. Treat "commit" steps in this plan as "stage + verify green", and do the single commit in Task 8.)*

---

## Task 4: Extract `_lane_int_probs` + `_lane_received_survival` (bit-identical refactor)

Factor the precompute and the sequential survival scan out of `_compute_lane_probabilities`, recomposing it as their composition. This must be **exactly** bit-identical (it is consumed by `lane_control`, hence by `n_blocked` and the baseline).

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py` (`_compute_lane_probabilities` at `:356-435`)
- Test: existing suite + `TestLeaveOneOutExactness`

- [ ] **Step 1: Run the full cover_shadows suite to capture the green baseline**

Run: `python -m pytest tests/tracking/test_cover_shadows.py -q`
Expected: all pass. Note the count.

- [ ] **Step 2: Add `_lane_int_probs` and `_lane_received_survival`, recompose `_compute_lane_probabilities`**

In `silly_kicks/tracking/_cover_shadows.py`, replace the body of `_compute_lane_probabilities` (`:356-435`) with the composition, and add the two helpers immediately above it:

```python
def _lane_int_probs(targets, defender_pos, defender_vel, attacker_pos, attacker_vel, *, params):
    """Per-lane clamp-independent precompute: interception probs + per-step control.

    Returns (p_int_def, p_int_att, t_ball, p_ctrl). p_ctrl[k] = 1 - exp(-lambda * dt_k)
    for dt_k > 0 else 0.0 (p_ctrl[0] = 0.0); a 0.0 entry reproduces the original
    ``dt_k <= 0`` skip exactly (zero contribution, prior unchanged).
    """
    n_points = targets.shape[0]
    d_from_passer = np.linalg.norm(targets - targets[0:1], axis=1)
    t_ball = ball_drag_time(d_from_passer, params)
    tti_def = player_tti(defender_pos, defender_vel, targets, is_defender=True, params=params)
    tti_att = player_tti(attacker_pos, attacker_vel, targets, is_defender=False, params=params)
    s = np.sqrt(3.0) * params.sigma / np.pi

    def _p_int(tti_matrix):
        dt = t_ball[np.newaxis, :] - tti_matrix
        return 1.0 / (1.0 + np.exp(-dt / s))

    p_int_def = _p_int(tti_def)
    p_int_att = _p_int(tti_att)
    dt = np.empty(n_points)
    dt[0] = 0.0
    dt[1:] = t_ball[1:] - t_ball[:-1]
    p_ctrl = np.where(dt > 0, 1.0 - np.exp(-params.lambda_ctrl * dt), 0.0)
    return p_int_def, p_int_att, t_ball, p_ctrl


def _lane_received_survival(p_int_def, p_int_att, p_ctrl):
    """Sequential clamped survival scan (verbatim arithmetic of the original loop).

    Bit-identical to the pre-refactor _compute_lane_probabilities inner loop.
    """
    n_points = p_ctrl.shape[0]
    n_def = p_int_def.shape[0]
    n_att = p_int_att.shape[0]
    p_blocked = 0.0
    p_received = 0.0
    p_anyone_prior = 0.0
    for k in range(1, n_points):
        pc = p_ctrl[k]
        if pc <= 0.0:
            continue
        total_contrib_k = 0.0
        for j in range(n_def):
            contrib = float(p_int_def[j, k]) * pc * (1.0 - p_anyone_prior)
            p_blocked += contrib
            total_contrib_k += contrib
        for j in range(n_att):
            contrib = float(p_int_att[j, k]) * pc * (1.0 - p_anyone_prior)
            p_received += contrib
            total_contrib_k += contrib
        p_anyone_prior = min(p_anyone_prior + total_contrib_k, 1.0)
    return p_blocked, p_received


def _compute_lane_probabilities(targets, defender_pos, defender_vel, attacker_pos, attacker_vel, *, params):
    """Compute P(blocked) and P(received) for one lane (composition of the two helpers)."""
    p_int_def, p_int_att, _t_ball, p_ctrl = _lane_int_probs(
        targets, defender_pos, defender_vel, attacker_pos, attacker_vel, params=params
    )
    return _lane_received_survival(p_int_def, p_int_att, p_ctrl)
```

- [ ] **Step 3: Run the full suite + exactness test**

Run: `python -m pytest tests/tracking/test_cover_shadows.py -q`
Expected: same pass count as Step 1. `TestLeaveOneOutExactness` still passes (production still uses the old `detailed=False` branch, now backed by the recomposed function).

- [ ] **Step 4: Verify `n_blocked` bit-stability explicitly (load-bearing)**

Run: `python -m pytest tests/tracking/test_cover_shadows.py -q -k "blocked or correlation or detailed"`
Expected: PASS. `n_blocked_receivers` flows through `lane_control → _compute_lane_probabilities`; its stability proves the refactor is bit-identical for the function's *other* consumer (the baseline / main path), not just the leave-one-out.

---

## Task 5: Rewrite the `detailed=False` branch — precompute + vectorized masked re-scan

Replace the `O(blockers × receivers)` `lane_control` loop with: classify-once (reuse `lane_blocker_ids`), per-receiver `_lane_int_probs`, and a single vectorized leave-one-out via a new `_lane_received_batched`.

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py` (add `_lane_received_batched`; rewrite the `else` branch at `:895-932`)
- Test: `TestLeaveOneOutExactness`, full suite

- [ ] **Step 1: Add `_lane_received_batched` (vectorized over the blocker axis)**

Add near the other helpers in `silly_kicks/tracking/_cover_shadows.py`:

```python
def _lane_received_batched(p_int_def, p_int_att, p_ctrl):
    """Baseline + leave-one-out p_received for one lane, vectorized over blockers.

    Returns (p_blocked_full, p_received_full, p_received_loo) where p_received_loo[m]
    is p_received with lane-blocker row m excluded.

    INV-1: the clamped recurrence is RE-RUN per variant (variant 0 = full set,
    variant m+1 = exclude blocker m), tracked by an independent ``prior`` per variant.
    Excluding a blocker adjusts only the per-step PLAYER sum (``full_def - def_col``);
    it never subtracts a contribution from the post-clamp accumulation.
    """
    n_points = p_ctrl.shape[0]
    nb = p_int_def.shape[0]
    nv = nb + 1  # 0 = full, 1..nb = leave-one-out
    prior = np.zeros(nv)
    p_blocked = np.zeros(nv)
    p_received = np.zeros(nv)
    att_sum_all = p_int_att.sum(axis=0)  # (n_points,)
    for k in range(1, n_points):
        pc = p_ctrl[k]
        if pc <= 0.0:
            continue
        def_col = p_int_def[:, k]  # (nb,)
        full_def = def_col.sum()
        def_sum = np.empty(nv)
        def_sum[0] = full_def
        def_sum[1:] = full_def - def_col  # exclude each blocker (per-step masked sum)
        att_sum = att_sum_all[k]
        one_minus_prior = 1.0 - prior
        blk = def_sum * pc * one_minus_prior
        rec = att_sum * pc * one_minus_prior
        p_blocked += blk
        p_received += rec
        prior = np.minimum(prior + blk + rec, 1.0)
    return p_blocked[0], p_received[0], p_received[1:]
```

- [ ] **Step 2: Rewrite the `detailed=False` branch**

In `_compute_cover_shadow_dict`, replace the entire `else:` block at `:895-932` (the lightweight approximation) with:

```python
    else:
        # Lightweight: classify once (lane_blocker_ids), precompute p_int per receiver,
        # then a single vectorized leave-one-out (re-run the clamped survival per excluded
        # lane-blocker). Bit-identical to the prior per-(d, r) lane_control loop within
        # rtol 1e-10 (man-marking is invariant under lane-blocker removal; see spec §2.1).
        xt_interp = xt.interpolator()  # type: ignore[union-attr]
        kept = defenders_outfield[defenders_outfield["player_id"].isin(lane_blocker_ids)]
        lb_pos = kept[["x", "y"]].to_numpy(dtype=np.float64)
        lb_vel = kept[["vx", "vy"]].to_numpy(dtype=np.float64)
        att_pos = attackers[["x", "y"]].to_numpy(dtype=np.float64)
        att_vel = attackers[["vx", "vy"]].to_numpy(dtype=np.float64)
        n_lb = lb_pos.shape[0]
        passer = np.array(passer_xy, dtype=np.float64)

        score_per_blocker = np.zeros(n_lb)
        for _, recv_row in dangerous.iterrows():
            recv_x = float(recv_row["x"])
            recv_y = float(recv_row["y"])
            recv_xt = float(xt_interp(np.array([recv_x]), np.array([recv_y]))[0, 0])

            receiver = np.array([recv_x, recv_y], dtype=np.float64)
            pass_vec = receiver - passer
            pass_dist = np.linalg.norm(pass_vec)
            if pass_dist < 1e-6:
                continue
            u = pass_vec / pass_dist
            u_perp = np.array([-u[1], u[0]])
            half_width = cs_params.cone_width_factor * pass_dist / 2.0
            t = np.linspace(0.0, 1.0, cs_params.n_sample_points)
            center = passer[np.newaxis, :] + t[:, np.newaxis] * pass_vec[np.newaxis, :]
            left = center + t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]
            right = center - t[:, np.newaxis] * half_width * u_perp[np.newaxis, :]

            old_recv = 0.0
            new_recv = np.zeros(n_lb)
            for lane in (center, left, right):
                p_int_def, p_int_att, _t_ball, p_ctrl = _lane_int_probs(
                    lane, lb_pos, lb_vel, att_pos, att_vel, params=cs_params
                )
                _pb, base_rec, loo_rec = _lane_received_batched(p_int_def, p_int_att, p_ctrl)
                old_recv += base_rec
                new_recv += loo_rec

            delta = np.maximum(new_recv - old_recv, 0.0)
            score_per_blocker += recv_xt * delta

        max_def = float(score_per_blocker.max()) if n_lb > 0 else 0.0
```

> **Deliberate deviation from spec §5.** The spec proposed merging the baseline into the precompute
> to remove the first-loop `lc_orig` pass. This plan instead **keeps the first loop unchanged**
> (`:833-845`) — it still computes `n_blocked` via `lane_control` on the full frame — so
> `n_blocked_receivers` stays provably bit-identical (it flows through untouched code). The new
> branch recomputes the baseline `old_recv` (variant 0 of `_lane_received_batched`), so `old_recv`
> is computed twice per receiver. This residual `O(receivers)` double-compute is **accepted**: it is
> cheap, and the dominant `O(blockers × receivers)` cost is still eliminated. Merging the baseline
> (per spec §5.2) is a possible future cleanup, deferred to keep `n_blocked` bit-stability airtight.
> The spec is amended to record this choice.
>
> The `receiver_records` construction (`:904-913`) is now obsolete and is part of the replaced
> block. The att race set is `attackers` (all attacking players incl. GK), matching `lane_control`'s
> `attackers_all` — and Task 3 Step 3 is what certifies this set construction is correct.

- [ ] **Step 3: Run the exactness test (production == frozen oracle within rtol 1e-10)**

Run: `python -m pytest tests/tracking/test_cover_shadows.py::TestLeaveOneOutExactness -q`
Expected: PASS at `rtol=1e-10`. This proves the precompute + vectorized re-scan equals the independent frozen leave-one-out, and guards INV-1.

- [ ] **Step 4: Run the full cover_shadows suite + correlation + atomic**

Run: `python -m pytest tests/tracking/test_cover_shadows.py tests/atomic/tracking/test_cover_shadows_atomic.py -q`
Expected: all pass. The ρ≥0.7 correlation test passes with the same ρ (values unchanged within tolerance). All five columns bit-stable.

---

## Task 6: Atomic parity test

Pin the "atomic inherits by pure delegation" claim so a future atomic fork cannot silently drift (spec §6.5).

**Files:**
- Test: `tests/atomic/tracking/test_cover_shadows_atomic.py`

- [ ] **Step 1: Write the parity test**

Add to `tests/atomic/tracking/test_cover_shadows_atomic.py` (mirror the existing fixtures/imports in that file; it already uses `_make_two_team_frame` and a fitted xt):

```python
def test_atomic_max_single_equals_standard(fitted_xt):
    """atomic.add_cover_shadows max_single == standard, on a shared frame (pure delegation)."""
    import numpy as np

    from silly_kicks.tracking.features import add_cover_shadows as std_cs
    from silly_kicks.atomic.tracking.features import add_cover_shadows as atomic_cs
    # ... build a shared multi-blocker `frames` + standard `actions` and atomic `atomic_actions`
    # matching the existing tests in this file (same passer/receiver geometry) ...
    std = std_cs(actions, frames, fitted_xt, home_team_id=1, detailed=False)
    atom = atomic_cs(atomic_actions, frames, fitted_xt, home_team_id=1, detailed=False)
    np.testing.assert_allclose(
        atom["max_single_defender_blocking_score"].to_numpy(),
        std["max_single_defender_blocking_score"].to_numpy(),
        rtol=1e-10, equal_nan=True,
    )
```

> Reuse the action/frame construction already present in `test_cover_shadows_atomic.py` (search for an existing `add_cover_shadows` call there and copy its inputs). The atomic and standard action rows must reference the same frame so the linked frame is identical.

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/atomic/tracking/test_cover_shadows_atomic.py::test_atomic_max_single_equals_standard -q`
Expected: PASS.

---

## Task 7: CI perf-budget guard + real-match confirmation

**Files:**
- Test: `tests/tracking/test_cover_shadows.py` (new `TestCoverShadowPerfBudget`)
- (Optional, PR-description only) real-match bit-identicality probe.

- [ ] **Step 1: Add a flat perf-budget timing guard**

Add to `tests/tracking/test_cover_shadows.py`:

```python
class TestCoverShadowPerfBudget:
    """Guard against silent regression of the leave-one-out optimization (spec §7)."""

    # Flat budget = worst observed post-change CI timing x 1.5 headroom.
    # Fill <BUDGET_S> after Step 2 (measure on this machine, pad for slow Windows CI).
    _BUDGET_S = 0.0  # TODO replace with measured value in Step 2

    def test_detailed_false_under_budget(self, fitted_xt):
        import time

        from silly_kicks.tracking._cover_shadows import _compute_cover_shadow_dict

        frame = _make_two_team_frame(
            home_positions=[(55.0, 30.0), (58.0, 35.0), (52.0, 28.0), (62.0, 40.0), (57.0, 32.0),
                            (60.0, 38.0), (65.0, 25.0), (20.0, 15.0), (25.0, 55.0), (30.0, 10.0)],
            away_positions=[(50.0, 34.0), (75.0, 34.0), (80.0, 25.0), (85.0, 45.0), (70.0, 20.0),
                            (70.0, 48.0), (90.0, 30.0), (95.0, 40.0), (45.0, 15.0), (45.0, 55.0)],
        )
        N = 20
        t0 = time.perf_counter()
        for _ in range(N):
            _compute_cover_shadow_dict(frame, (50.0, 34.0), 2, fitted_xt, home_team_id=1, detailed=False)
        per_call = (time.perf_counter() - t0) / N
        assert per_call < self._BUDGET_S, f"per-call {per_call*1000:.2f} ms exceeds budget {self._BUDGET_S*1000:.2f} ms"
```

- [ ] **Step 2: Measure post-change timing and set `_BUDGET_S`**

Run: `python -m pytest "tests/tracking/test_cover_shadows.py::TestCoverShadowPerfBudget::test_detailed_false_under_budget" -q -s`
With `_BUDGET_S = 999.0` temporarily, capture the printed/observed per-call time (add a `print(per_call)` if needed). Then set `_BUDGET_S` = `max(observed_per_call, <baseline-from-Task-2-if-known>) * 1.5`, padded generously for slower Windows CI (per the repo's perf-budget convention). Re-run; expected PASS comfortably under budget.

> **Do not skip this step.** The committed default `_BUDGET_S = 0.0` fails-closed (any per-call time
> exceeds 0.0), so a skipped step surfaces as a red test rather than a silent pass — but the budget
> must be set to a real, padded value before the suite is green. In subagent execution, treat this
> as a required step, not optional.

- [ ] **Step 3: Real-match bit-identicality confirmation (diff against Task 2 capture)**

If `_prematch_maxsingle.parquet` was captured in Task 2 Step 2, recompute
`max_single_defender_blocking_score` over the **same** real match with the now-post-change code and
diff against the persisted pre-change values:

```python
pre = pd.read_parquet("_prematch_maxsingle.parquet").set_index("action_id")["max_single_defender_blocking_score"]
post = add_cover_shadows(actions, frames, xt, home_team_id=HOME, detailed=False).set_index("action_id")["max_single_defender_blocking_score"]
max_abs = (post.reindex(pre.index) - pre).abs().max()
print(f"real-match max|Δ| = {max_abs:.3e}")   # expect < 1e-10
```

Confirm `max |Δ| < 1e-10`. Report this figure (and the Task 2 baseline vs post-change speedup) in
the PR description; then `rm _prematch_maxsingle.parquet`. This is empirical evidence beyond the
synthetic golden (not a committed test — real-match fixtures aren't in-repo).

> If Task 2 Step 2 recorded "no real match available", state that explicitly in the PR and rely on
> the synthetic exactness test (§6.1) + the no-ripple proof/property test — **do not silently skip
> the claim.**

---

## Task 8: Docs, version bump, TODO grooming, final-review, single commit

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py` (docstrings), `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md`

- [ ] **Step 1: Update docstrings**

In `_cover_shadows.py`, update the `_compute_cover_shadow_dict` docstring (and add docstrings to `_lane_int_probs` / `_lane_received_survival` / `_lane_received_batched`) to state: the no-ripple property (spec §2.1), the precompute + masked re-scan approach, and INV-1. Add `See docs/superpowers/specs/2026-05-28-cover-shadows-leave-one-out-decouple-design.md`.

- [ ] **Step 2: Version bump (3.25.1 patch)**

Set version to `3.25.1` in `pyproject.toml` and `silly_kicks/__init__.py` (`__version__`). (Confirm 3.25.1 vs 3.26.0 with the maintainer at this point per spec §9; patch is the recommended SemVer-honest choice for a bit-identical change.)

- [ ] **Step 3: CHANGELOG entry**

Add under a new `## 3.25.1` heading:

```
### Performance
- cover_shadows: `max_single_defender_blocking_score` (`detailed=False`) now computed via a
  single vectorized leave-one-out (hoisted the redundant per-defender man-marking
  re-classification, which is provably a no-op for lane-blocker removals). Bit-identical
  within `rtol 1e-10` — **no value or API change; no golden/model regeneration required.**
```

- [ ] **Step 4: TODO grooming**

Delete the "TF-7 cover_shadows nested-loop optimization (deferred, 3.25.0 review)" row from `TODO.md` (CHANGELOG is the record; do not strikethrough).

- [ ] **Step 5: TF-30 spec amendment**

Add a one-line note at the top of `docs/superpowers/specs/2026-05-10-tf30-cover-shadows-design.md` pointing to the 2026-05-28 leave-one-out spec for the lightweight-path perf refactor.

- [ ] **Step 6: Run the full non-e2e suite + lint + types**

Run: `python -m pytest tests/ -m "not e2e" -q`
Run: `ruff format --check . ; ruff check . ; pyright silly_kicks/`
Expected: all green. Fix any issues.

- [ ] **Step 7: Run `/final-review` (mandatory gate)**

Invoke the `final-review` skill. Address findings.

- [ ] **Step 8: Single commit on a `pr-s65-...` branch**

Create branch `pr-s65-cover-shadows-leave-one-out` off `main`; stage all changes; make ONE commit (per project commit policy — explicit maintainer approval required before committing). Do not push/tag until requested and CI is green.

---

## Self-Review notes (author)

- **Spec coverage:** §2.1 no-ripple → Task 1 guard (done) + Task 3 value-level confirmation. §5 algorithm → Tasks 4–5. §6.1 oracle → Task 3. §6.2 ρ → Task 5 Step 4. §6.3 real-match → Task 7 Step 3. §6.4 property test → Task 1 (done). §6.5 atomic + n_blocked → Task 6 + Task 4 Step 4. §6.6 three steps → Tasks 3/4/5. §7 perf → Tasks 2 + 7. §9 docs/version → Task 8.
- **INV-1** is enforced structurally (`_lane_received_batched` re-runs the clamped recurrence per variant) and guarded by the Task 3/5 exactness test.
- **Bit-stability of `n_blocked`** is preserved by leaving the first loop / `lane_control` untouched and proven in Task 4 Step 4.
