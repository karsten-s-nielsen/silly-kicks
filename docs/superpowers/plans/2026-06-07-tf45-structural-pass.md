# TF-45 `structural_pass` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the three raw per-pass structural primitives — Line Bypass Score, Space Gain Metric, Structural Disruption Index (`structural_lbs` / `structural_sgm` / `structural_sdi`) — as a pure silly-kicks tracking feature (arXiv:2603.28916). **Also fold in (per maintainer) a systemic fix** to the dup-`action_id` crash that affects ~8 shipped frame-aware xfns families when composed into VAEP gamestates.

**Architecture:** Pure pandas/numpy ADR-005 tracking feature. A pandas-free hexagonal core `_structural_pass_core` holds all the math; a DataFrame primitive `compute_structural_pass_metrics` wraps it (defender selection + per-team coordinate mirror); a shared `_kernels._structural_pass_at_actions` batch kernel is called by BOTH the `add_structural_pass` aggregator and the `structural_pass_xfns` VAEP factory (DRY + the 3×-not-9× call-count budget). Atomic mirror re-exports `compute_*` and wraps `add_*`/`xfns` with `end = x+dx` endpoint synthesis. TIV/archetypes/rankings are out of scope (consumer-side). **Systemic fix:** a shared `_kernels.resolve_frame_ids_by_position` resolves linked frame_ids by position (dup-`action_id`-safe), a red-first behavioral gate enumerates every `*_xfns` factory, and the ~8 broken families are retrofitted to the resolver — all in this single commit.

**Tech Stack:** Python 3.10, pandas 2.3, numpy 2.2, pytest. Spec: `docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md`.

**Commit policy (per maintainer):** This is ONE feature branch → ONE squash commit/PR at the end (docs bundled, no standalone doc commits). Tasks below end at a **green test checkpoint**, NOT a commit. The single commit (Task 11) is gated on explicit per-commit sentinel approval — do not create the `~/.claude-git-approval` sentinel without it.

**Venv/commands:** use the repo venv. Test command prefix:
`D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest`

---

## File Structure

- **Create** `silly_kicks/tracking/_structural_pass.py` — `StructuralPassParams`, `_structural_pass_core` (pure), `compute_structural_pass_metrics` (DataFrame primitive). One responsibility: the structural-pass math + per-frame wrapper.
- **Modify** `silly_kicks/tracking/_kernels.py` — add `_structural_pass_at_actions` (per-action batch).
- **Modify** `silly_kicks/tracking/features.py` — re-export the primitive; add `add_structural_pass` + `structural_pass_xfns`.
- **Modify** `silly_kicks/tracking/__init__.py` — export the public surface.
- **Modify** `silly_kicks/atomic/tracking/features.py` — atomic mirror (re-export `compute_*`; endpoint-synth wrappers for `add_*`/`xfns`).
- **Create** `tests/tracking/test_structural_pass.py` — core + DataFrame + aggregator + xfns + atomic-parity oracle tests.
- **Create** `tests/tracking/test_structural_pass_perf_budget.py` — 3×-call-count structural guard.
- **Create** `tests/tracking/test_structural_pass_e2e.py` — owner-gated WC2022 e2e.
- **Create** `tests/tracking/test_frame_aware_xfns_dup_action_id.py` — behavioral gate over ALL `*_xfns` (systemic fix).
- **Modify** `silly_kicks/tracking/_kernels.py` — add shared `resolve_frame_ids_by_position`; retrofit ~8 broken xfns transformers in `silly_kicks/tracking/features.py` (`pitch_control`, `obso`, `pausa`, `space_creation`, `pressure`, `cover_shadow`, `gk_influence`, `player_influence`).
- **Create** `scripts/tune_structural_pass_sigma.py` — owner-gated σ-tuning reproducibility script.
- **Modify** `NOTICE`, `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `docs/c4/architecture.dsl` (+ regen `architecture.html`).

---

## Task 0: Feature branch + module skeleton

**Files:**
- Modify: (git) create branch
- Create: `silly_kicks/tracking/_structural_pass.py`

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b feat/tf45-structural-pass
```

- [ ] **Step 2: Create the module with params + import stubs (so later imports resolve)**

Create `silly_kicks/tracking/_structural_pass.py`:

```python
"""Per-pass structural primitives (TF-45): Line Bypass Score, Space Gain Metric,
Structural Disruption Index.

Quantifies how a pass deforms the opponent's defensive structure, independent of
outcome value. Library ships RAW primitives only; the TIV z-norm composite,
K-means archetypes, and passer/receiver rankings are corpus-level and live with
consumers (mirrors the frozen-exogenous-xT decision, ADR-009).

INVARIANT: post-normalization SPADL action coords (start_x/start_y, end_x/end_y;
acting team attacks +x) and LTR tracking coords (home attacks +x) share the
[0,105]x[0,68] pitch frame. Defenders are mirrored (105-x, 68-y) into the action's
attack-positive frame iff the acting team is the AWAY team. We mirror DEFENDERS
(not the action coords as _line_breaking.py:243-252 does) on purpose: LBS is only
clean in attack-positive coords (otherwise the inequality flips sign per team);
SGM/SDI are isometry-invariant so the direction does not matter for them.

CAVEAT (see NOTICE): LBS is purely 1-D along the attacking axis — a defender whose
d_x in (start_x, end_x] is counted even if he is on the opposite touchline.
Receiver location x_r is the pass DESTINATION (end_x/end_y); SPADL has no
receiver_player_id.

See spec docs/superpowers/specs/2026-06-07-tf45-structural-pass-design.md.
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._id_compat import ids_match, same_id


@dataclass(frozen=True)
class StructuralPassParams:
    """Tunable parameters for structural-pass metrics.

    sigma: defender spatial-influence radius (m) for the SGM Gaussian density.
    Default 15.0 — empirically tuned (2,466 real WC2022 passes): smallest sigma at
    which the faithful 1/rho is intrinsically bounded by pitch geometry (no
    eps-floor). See scripts/tune_structural_pass_sigma.py + spec D1. No is_default()
    (matches CoverShadowParams / LineBreakingParams).
    """

    sigma: float = 15.0
```

- [ ] **Step 3: Verify the module imports**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -c "from silly_kicks.tracking._structural_pass import StructuralPassParams; print(StructuralPassParams().sigma)"`
Expected: `15.0`

---

## Task 1: Pure core `_structural_pass_core` (TDD)

**Files:**
- Modify: `silly_kicks/tracking/_structural_pass.py`
- Test: `tests/tracking/test_structural_pass.py`

- [ ] **Step 1: Write the failing core oracle tests**

Create `tests/tracking/test_structural_pass.py`:

```python
"""Tests for per-pass structural primitives (TF-45)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._structural_pass import _structural_pass_core


def _sgm_ref(defs, p, sigma):
    d2 = ((defs - np.asarray(p)) ** 2).sum(axis=1)
    rho = np.exp(-d2 / (2.0 * sigma * sigma)).sum()
    return 1.0 / rho


class TestStructuralPassCore:
    def test_lbs_counts_defenders_in_band(self):
        # passer at x=40, receiver at x=70; defenders at x=50,60 (in band), 30,80 (out)
        defs = np.array([[50.0, 34.0], [60.0, 20.0], [30.0, 34.0], [80.0, 34.0]])
        lbs, _sgm, _sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 2

    def test_lbs_boundary_strict_lower_inclusive_upper(self):
        # start_x < d_x <= end_x : d at start_x excluded, d at end_x included
        defs = np.array([[40.0, 34.0], [70.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 1  # x=40 excluded (strict <), x=70 included (<=)

    def test_lbs_backward_pass_is_zero(self):
        defs = np.array([[50.0, 34.0], [60.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (70.0, 34.0), (40.0, 34.0), 15.0)
        assert lbs == 0  # receiver behind passer -> empty band

    def test_lbs_zero_with_defenders_present_is_not_nan(self):
        # forward pass, defenders present but none in band -> structural_lbs == 0 (NOT nan)
        defs = np.array([[10.0, 34.0], [90.0, 34.0]])
        lbs, _, _ = _structural_pass_core(defs, (40.0, 34.0), (50.0, 34.0), 15.0)
        assert lbs == 0

    def test_zero_defenders_all_nan(self):
        defs = np.empty((0, 2))
        lbs, sgm, sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert np.isnan(lbs) and np.isnan(sgm) and np.isnan(sdi)

    def test_single_defender_is_numeric(self):
        defs = np.array([[55.0, 34.0]])
        lbs, sgm, sdi = _structural_pass_core(defs, (40.0, 34.0), (70.0, 34.0), 15.0)
        assert lbs == 1
        assert np.isfinite(sgm) and np.isfinite(sdi)

    def test_sgm_matches_reference(self):
        defs = np.array([[55.0, 30.0], [60.0, 40.0]])
        p, r, sigma = (40.0, 34.0), (70.0, 34.0), 15.0
        _, sgm, _ = _structural_pass_core(defs, p, r, sigma)
        expected = _sgm_ref(defs, r, sigma) - _sgm_ref(defs, p, sigma)
        assert sgm == pytest.approx(expected, abs=1e-9)

    def test_sdi_matches_centroid_reference(self):
        defs = np.array([[50.0, 30.0], [60.0, 38.0]])
        p, r = (40.0, 34.0), (70.0, 34.0)
        _, _, sdi = _structural_pass_core(defs, p, r, 15.0)
        c = defs.mean(axis=0)
        expected = np.hypot(r[0] - c[0], r[1] - c[1]) - np.hypot(p[0] - c[0], p[1] - c[1])
        assert sdi == pytest.approx(expected, abs=1e-9)
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py -q`
Expected: FAIL — `ImportError: cannot import name '_structural_pass_core'`

- [ ] **Step 3: Implement the pure core**

Append to `silly_kicks/tracking/_structural_pass.py`:

```python
def _structural_pass_core(
    defenders_xy: np.ndarray,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    sigma: float,
) -> tuple[float, float, float]:
    """Pure structural-pass math. defenders_xy is (n,2) in the acting-attack-positive
    frame; passer_xy=(start_x,start_y), receiver_xy=(end_x,end_y) in the same frame.

    Returns (structural_lbs, structural_sgm, structural_sdi).
    0 defenders -> (nan, nan, nan) (only degenerate case: rho=0 / centroid undefined).
    >=1 defender -> all numeric. structural_lbs is an int-valued float (count).
    """
    d = np.asarray(defenders_xy, dtype="float64")
    if d.ndim != 2 or d.shape[0] == 0:
        return (np.nan, np.nan, np.nan)

    p = np.asarray(passer_xy, dtype="float64")
    r = np.asarray(receiver_xy, dtype="float64")

    # LBS: defenders with start_x < d_x <= end_x (forward-only by construction)
    lbs = float(np.count_nonzero((d[:, 0] > p[0]) & (d[:, 0] <= r[0])))

    # SGM: inverse Gaussian density (available space), receiver minus passer
    two_s2 = 2.0 * sigma * sigma
    rho_p = np.exp(-((d - p) ** 2).sum(axis=1) / two_s2).sum()
    rho_r = np.exp(-((d - r) ** 2).sum(axis=1) / two_s2).sum()
    sgm = (1.0 / rho_r) - (1.0 / rho_p)

    # SDI: distance-from-defensive-centroid, receiver minus passer
    c = d.mean(axis=0)
    sdi = float(np.hypot(r[0] - c[0], r[1] - c[1]) - np.hypot(p[0] - c[0], p[1] - c[1]))

    return (lbs, float(sgm), sdi)
```

- [ ] **Step 4: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py -q`
Expected: PASS (8 tests)

---

## Task 2: DataFrame primitive `compute_structural_pass_metrics` (TDD)

**Files:**
- Modify: `silly_kicks/tracking/_structural_pass.py`
- Test: `tests/tracking/test_structural_pass.py`

- [ ] **Step 1: Write failing tests (use the shared frame-fixture builder)**

Append to `tests/tracking/test_structural_pass.py`:

```python
from tests.tracking.test_defensive_line import _make_frame_rows
from silly_kicks.tracking._structural_pass import compute_structural_pass_metrics


class TestComputePrimitive:
    def _frame(self):
        # away outfield acts as defenders for a HOME pass: x=50,60 in band
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_home_pass_keys_and_lbs(self):
        out = compute_structural_pass_metrics(
            self._frame(),
            attacking_team_id=1,
            home_team_id=1,
            passer_xy=(40.0, 34.0),
            receiver_xy=(70.0, 34.0),
        )
        assert set(out) == {"structural_lbs", "structural_sgm", "structural_sdi"}
        assert out["structural_lbs"] == 2.0  # away defenders at x=50,60

    def test_gk_excluded_ball_excluded(self):
        # away GK at (102,34) is NOT a defender; ball excluded; lbs unchanged
        out = compute_structural_pass_metrics(
            self._frame(), attacking_team_id=1, home_team_id=1,
            passer_xy=(40.0, 34.0), receiver_xy=(70.0, 34.0),
        )
        assert out["structural_lbs"] == 2.0

    def test_away_pass_mirror_matches_home(self):
        # Build a mirror-symmetric scenario: an away pass with mirrored coords must
        # yield the same metrics as the equivalent home pass.
        frame = _make_frame_rows(
            home_outfield_xs=[55.0, 45.0, 75.0, 25.0],  # these become defenders for an AWAY pass
            home_outfield_ys=[34.0, 48.0, 34.0, 34.0],
            away_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            away_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        )
        # away pass: action coords already attack-positive (away attacks +x in SPADL)
        out = compute_structural_pass_metrics(
            frame, attacking_team_id=2, home_team_id=1,
            passer_xy=(40.0, 34.0), receiver_xy=(70.0, 34.0),
        )
        # home defenders mirrored: 105-55=50, 105-45=60 in band (40,70] -> lbs 2
        assert out["structural_lbs"] == 2.0
        assert np.isfinite(out["structural_sgm"]) and np.isfinite(out["structural_sdi"])

    def test_zero_defenders_nan(self):
        # frame with only the acting team + GKs -> 0 opponent outfield
        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[], away_outfield_ys=[],
        )
        out = compute_structural_pass_metrics(
            frame, attacking_team_id=1, home_team_id=1,
            passer_xy=(40.0, 34.0), receiver_xy=(70.0, 34.0),
        )
        assert all(np.isnan(v) for v in out.values())
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestComputePrimitive -q`
Expected: FAIL — `cannot import name 'compute_structural_pass_metrics'`

- [ ] **Step 3: Implement the DataFrame primitive**

Append to `silly_kicks/tracking/_structural_pass.py`:

```python
def compute_structural_pass_metrics(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    home_team_id: int | str,
    passer_xy: tuple[float, float],
    receiver_xy: tuple[float, float],
    params: StructuralPassParams | None = None,
) -> dict[str, float]:
    """Per-frame structural-pass metrics for ONE linked frame.

    Schema-agnostic: passer_xy/receiver_xy passed explicitly (re-exportable to atomic).
    Defenders = opponent outfield (~is_ball FIRST, then ~ids_match(team, attacking)
    & ~is_goalkeeper). Defenders mirrored into the acting-attack-positive frame iff
    the acting team is AWAY, then handed to _structural_pass_core.
    """
    if params is None:
        params = StructuralPassParams()

    nan_out = {"structural_lbs": np.nan, "structural_sgm": np.nan, "structural_sdi": np.nan}
    if frame is None or len(frame) == 0:
        return dict(nan_out)
    if not (np.isfinite(passer_xy[0]) and np.isfinite(passer_xy[1])
            and np.isfinite(receiver_xy[0]) and np.isfinite(receiver_xy[1])):
        return dict(nan_out)

    players = frame[~frame["is_ball"].astype(bool)]
    opp = players[
        ~ids_match(players["team_id"], attacking_team_id).to_numpy()
        & ~players["is_goalkeeper"].astype(bool).to_numpy()
    ]
    dx = opp["x"].to_numpy(dtype="float64")
    dy = opp["y"].to_numpy(dtype="float64")
    ok = np.isfinite(dx) & np.isfinite(dy)
    dx, dy = dx[ok], dy[ok]
    if dx.size == 0:
        return dict(nan_out)

    # Mirror defenders into the acting team's attack-positive frame iff AWAY.
    if not same_id(attacking_team_id, home_team_id):
        dx, dy = 105.0 - dx, 68.0 - dy
    defenders_xy = np.column_stack([dx, dy])

    lbs, sgm, sdi = _structural_pass_core(defenders_xy, passer_xy, receiver_xy, params.sigma)
    return {"structural_lbs": lbs, "structural_sgm": sgm, "structural_sdi": sdi}
```

- [ ] **Step 4: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py -q`
Expected: PASS (12 tests)

---

## Task 3: Shared batch kernel `_structural_pass_at_actions` (TDD)

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py`
- Test: `tests/tracking/test_structural_pass.py`

- [ ] **Step 1: Write the failing kernel test**

Append to `tests/tracking/test_structural_pass.py`:

```python
def _actions(team_id=1):
    return pd.DataFrame({
        "game_id": [1, 1],
        "action_id": [1, 2],
        "period_id": [1, 1],
        "time_seconds": [1.0, 1.0],
        "team_id": [team_id, team_id],
        "player_id": [50, 51],
        "start_x": [40.0, 0.0],
        "start_y": [34.0, 34.0],
        "end_x": [70.0, 5.0],
        "end_y": [34.0, 34.0],
        "type_id": [0, 0],
    })


class TestKernel:
    def test_batch_aligns_and_computes(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        out = _structural_pass_at_actions(_actions(), frame, home_team_id=1)
        assert list(out.columns) == ["structural_lbs", "structural_sgm", "structural_sdi"]
        assert out["structural_lbs"].iloc[0] == 2
        assert len(out) == 2

    def test_unlinked_action_nan(self):
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
            time_seconds=500.0,
        )
        out = _structural_pass_at_actions(_actions(), frame, home_team_id=1)
        assert pd.isna(out["structural_lbs"].iloc[0])

    def test_duplicate_action_id_in_slot_does_not_raise(self):
        # VAEP shifted gamestate slots repeat the boundary action -> non-unique
        # action_id. The kernel must resolve frame_id by position, not via
        # set_index("action_id").at[dup] (which returns a Series and raises).
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        slot = _actions()
        slot["action_id"] = [1, 1]  # duplicate, as in a shifted boundary slot
        out = _structural_pass_at_actions(slot, frame, home_team_id=1)
        assert len(out) == 2
        assert out["structural_lbs"].iloc[0] == 2  # forward pass row resolved correctly

    def test_string_team_id_dtype_safe(self):
        # opponent selection (ids_match) + mirror (same_id) must tolerate object team ids.
        from silly_kicks.tracking._kernels import _structural_pass_at_actions

        frame = _make_frame_rows(
            home_team_id="H", away_team_id="A",
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        slot = _actions(team_id="H")
        out = _structural_pass_at_actions(slot, frame, home_team_id="H")
        assert out["structural_lbs"].iloc[0] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestKernel -q`
Expected: FAIL — `cannot import name '_structural_pass_at_actions'`

- [ ] **Step 3: Implement the kernel (per-action loop, dup-action_id-SAFE)**

> ⚠️ **Critical (review #3-A):** VAEP shifted gamestate slots (`states[1]`, `states[2]`) repeat the period-boundary action, so the `action_id` column is **non-unique** (empirically `[10.0, 10.0, 11.0, 12.0]`). `pointers.set_index("action_id").at[dup_id]` returns a *Series* → `int(float(...))` raises `TypeError`. We therefore resolve each row's frame_id **by position** via a unique surrogate id + vectorized `reindex` (never `.at` on a possibly-non-unique index). `period_id` is cast to `int` before `get_group` because the shift promotes it to float. Frame lookup assumes numeric `(period_id, frame_id)` — true for every provider (frame_id is a frame number; only team/player/match ids are string). The per-pass `iterrows` loop is inherent to a per-pass metric (each row has distinct passer/receiver coords); `get_group` is an O(1) indexed lookup, not the O(n·m) mask antipattern.

Append to `silly_kicks/tracking/_kernels.py` (end of file):

```python
def _structural_pass_at_actions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    params=None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """structural_lbs / structural_sgm / structural_sdi for each pass/cross action at
    its linked frame. Returns a DataFrame aligned with actions.index (3 columns).

    Shared by add_structural_pass and structural_pass_xfns (DRY + 3x-not-9x budget).
    Non-pass/cross rows and unlinked/degenerate rows -> NaN. Robust to non-unique
    action_id in shifted VAEP gamestate slots (frame_id resolved by position).
    """
    from ._structural_pass import StructuralPassParams, compute_structural_pass_metrics
    from .utils import link_actions_to_frames

    if params is None:
        params = StructuralPassParams()

    n = len(actions)
    out = pd.DataFrame(
        {
            "structural_lbs": np.full(n, np.nan),
            "structural_sgm": np.full(n, np.nan),
            "structural_sdi": np.full(n, np.nan),
        },
        index=actions.index,
    )
    if n == 0 or len(frames) == 0:
        return out

    # Resolve each row's frame_id BY POSITION (dup-action_id-safe). add_ supplies
    # links keyed by unique action_id; the xfns path links a unique-surrogate copy.
    if links is not None:
        fid_lookup = links.drop_duplicates("action_id").set_index("action_id")["frame_id"]
        fid_by_pos = fid_lookup.reindex(actions["action_id"].to_numpy()).to_numpy(dtype="float64")
    else:
        link_input = actions.copy()
        link_input["action_id"] = np.arange(n)  # unique positional surrogate
        pointers, _ = link_actions_to_frames(link_input, frames)
        fid_by_pos = (
            pointers.set_index("action_id")["frame_id"].reindex(np.arange(n)).to_numpy(dtype="float64")
        )

    col_lbs = np.full(n, np.nan)
    col_sgm = np.full(n, np.nan)
    col_sdi = np.full(n, np.nan)
    frame_groups = frames.groupby(["period_id", "frame_id"])

    PASS, CROSS = 0, 1
    for j, (_idx, row) in enumerate(actions.iterrows()):
        if row.get("type_id") not in (PASS, CROSS):
            continue
        tid = row["team_id"]
        if pd.isna(tid) or np.isnan(fid_by_pos[j]):
            continue
        pid = int(row["period_id"])  # shift may promote period_id to float
        fid = int(fid_by_pos[j])
        try:
            frame_data = frame_groups.get_group((pid, fid))
        except KeyError:
            continue

        m = compute_structural_pass_metrics(
            frame_data,
            attacking_team_id=tid,
            home_team_id=home_team_id,
            passer_xy=(float(row["start_x"]), float(row["start_y"])),
            receiver_xy=(float(row["end_x"]), float(row["end_y"])),
            params=params,
        )
        col_lbs[j] = m["structural_lbs"]
        col_sgm[j] = m["structural_sgm"]
        col_sdi[j] = m["structural_sdi"]

    out["structural_lbs"] = col_lbs
    out["structural_sgm"] = col_sgm
    out["structural_sdi"] = col_sdi
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestKernel -q`
Expected: PASS (2 tests)

---

## Task 4: `add_structural_pass` aggregator + exports (TDD)

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_structural_pass.py`

- [ ] **Step 1: Write failing aggregator tests**

Append to `tests/tracking/test_structural_pass.py`:

```python
class TestAggregator:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_appends_three_namespaced_columns(self):
        from silly_kicks.tracking.features import add_structural_pass

        res = add_structural_pass(_actions(), self._frame(), home_team_id=1)
        for c in ("structural_lbs", "structural_sgm", "structural_sdi"):
            assert c in res.columns
        assert res["structural_lbs"].iloc[0] == 2
        # provenance present
        assert "frame_id" in res.columns

    def test_non_pass_cross_is_na(self):
        from silly_kicks.tracking.features import add_structural_pass

        acts = _actions()
        acts.loc[0, "type_id"] = 11  # shot
        res = add_structural_pass(acts, self._frame(), home_team_id=1)
        assert pd.isna(res["structural_lbs"].iloc[0])

    def test_links_path_equals_internal(self):
        from silly_kicks.tracking.features import add_structural_pass
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        a = add_structural_pass(acts, frame, home_team_id=1)
        links, _ = link_actions_to_frames(acts, frame)
        b = add_structural_pass(acts, frame, home_team_id=1, links=links)
        pd.testing.assert_series_equal(a["structural_sgm"], b["structural_sgm"])

    def test_provenance_present_and_unsuffixed_on_rechain(self):
        # Re-running must not produce frame_id_x/frame_id_y merge suffixes (the real
        # provenance-skip hazard), and provenance must remain present.
        from silly_kicks.tracking.features import add_structural_pass

        acts, frame = _actions(), self._frame()
        once = add_structural_pass(acts, frame, home_team_id=1)
        twice = add_structural_pass(once, frame, home_team_id=1)
        assert "frame_id" in twice.columns
        assert "frame_id_x" not in twice.columns and "frame_id_y" not in twice.columns
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestAggregator -q`
Expected: FAIL — `cannot import name 'add_structural_pass'`

- [ ] **Step 3: Implement `add_structural_pass` in features.py**

Add to `silly_kicks/tracking/features.py` (near the other `add_*` aggregators, e.g. after `add_defensive_line`). First ensure the re-export import exists near the top-of-module imports (after line 67 block):

```python
from ._structural_pass import (  # noqa: E402  (grouped with peer feature re-exports)
    StructuralPassParams,
    compute_structural_pass_metrics,
)
```

Then add the aggregator:

```python
@nan_safe_enrichment
def add_structural_pass(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    links: pd.DataFrame | None = None,
    params: StructuralPassParams | None = None,
) -> pd.DataFrame:
    """Append structural_lbs / structural_sgm / structural_sdi (NA for non-pass/cross).

    Per-pass structural primitives (TF-45, arXiv:2603.28916). Idempotent provenance
    columns. Accepts caller-supplied ``links`` (skips internal link_actions_to_frames).
    The body implements the NaN-safety contract; @nan_safe_enrichment + the CI gate
    only verify it.

    See NOTICE for full bibliographic citations.
    """
    batch = _kernels._structural_pass_at_actions(
        actions, frames, home_team_id=home_team_id, params=params, links=links
    )
    out = actions.copy()
    # House Int64 idiom (features.py:1154): float Series w/ NaN -> Int64 (<NA>) preserves
    # the LBS=0-vs-NaN distinction. NOT pd.array(..., dtype="Int64") (review #3-C).
    out["structural_lbs"] = batch["structural_lbs"].astype("Int64")
    out["structural_sgm"] = batch["structural_sgm"].to_numpy()
    out["structural_sdi"] = batch["structural_sdi"].to_numpy()

    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    existing = [c for c in provenance_cols if c in out.columns]
    if not existing:
        pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
        if len(pointers) > 0:
            ptr_cols = pointers.set_index("action_id")[provenance_cols]
            out = out.merge(ptr_cols, left_on="action_id", right_index=True, how="left")
    return out
```

Note: `Series.astype("Int64")` from a float Series with NaN coerces NaN→`pd.NA` and finite counts→ints (LBS values are integral). This preserves the LBS=0-vs-NaN distinction.

- [ ] **Step 4: Export in `__init__.py`**

In `silly_kicks/tracking/__init__.py`, add `"add_structural_pass"`, `"compute_structural_pass_metrics"`, `"StructuralPassParams"` to `__all__` (alphabetically near the others) and add to the `from .features import (...)` block:

```python
    add_structural_pass,
    compute_structural_pass_metrics,
    StructuralPassParams,
```

(Confirm `StructuralPassParams` + `compute_structural_pass_metrics` are re-exported by `features.py` — they are, via the Step 3 import.)

- [ ] **Step 5: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestAggregator -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the NaN-safety gate (auto-discovers the decorated helper)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/test_enrichment_nan_safety.py -q`
Expected: PASS

---

## Task 5: `structural_pass_xfns` VAEP factory + perf guard (TDD)

**Files:**
- Modify: `silly_kicks/tracking/features.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_structural_pass.py`, `tests/tracking/test_structural_pass_perf_budget.py`

- [ ] **Step 1: Write failing xfns + perf tests**

Append to `tests/tracking/test_structural_pass.py`:

```python
class TestXfns:
    def test_emits_namespaced_per_slot_columns(self):
        from silly_kicks.tracking.features import structural_pass_xfns

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        xfns = structural_pass_xfns(home_team_id=1)
        assert len(xfns) == 1
        transformer = xfns[0]
        assert getattr(transformer, "_frame_aware", False) is True
        states = [_actions(), _actions(), _actions()]
        cols = transformer(states, frame)
        assert "structural_lbs_a0" in cols.columns
        assert "structural_sdi_a2" in cols.columns

    def test_frames_none_guard(self):
        from silly_kicks.tracking.features import structural_pass_xfns

        transformer = structural_pass_xfns(home_team_id=1)[0]
        states = [_actions(), _actions(), _actions()]
        cols = transformer(states, None)
        assert cols["structural_lbs_a0"].isna().all()

    def test_real_gamestates_with_duplicate_action_ids(self):
        # End-to-end guard: real gamestates() produce shifted slots with NON-UNIQUE
        # action_id at period boundaries. The transformer must not raise.
        from silly_kicks.tracking.features import structural_pass_xfns
        from silly_kicks.vaep.feature_framework import gamestates

        frame = _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )
        states = gamestates(_actions(), nb_prev_actions=3)
        assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
        transformer = structural_pass_xfns(home_team_id=1)[0]
        cols = transformer(states, frame)  # must not raise
        assert "structural_lbs_a0" in cols.columns
```

Create `tests/tracking/test_structural_pass_perf_budget.py`:

```python
"""Structural perf guard: the VAEP factory must call the shared kernel 3x (once per
gamestate slot), NOT 9x. Deterministic call-count, not a wall-clock ceiling."""

from __future__ import annotations

import pandas as pd

import silly_kicks.tracking._kernels as _kernels
from silly_kicks.tracking.features import structural_pass_xfns
from tests.tracking.test_defensive_line import _make_frame_rows
from tests.tracking.test_structural_pass import _actions


def test_kernel_called_once_per_slot(monkeypatch):
    calls = {"n": 0}
    orig = _kernels._structural_pass_at_actions

    def spy(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(_kernels, "_structural_pass_at_actions", spy)
    frame = _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
        home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
        away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
    )
    transformer = structural_pass_xfns(home_team_id=1)[0]
    transformer([_actions(), _actions(), _actions()], frame)
    assert calls["n"] == 3
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestXfns tests/tracking/test_structural_pass_perf_budget.py -q`
Expected: FAIL — `cannot import name 'structural_pass_xfns'`

- [ ] **Step 3: Implement `structural_pass_xfns`**

Add to `silly_kicks/tracking/features.py` (after `add_structural_pass`). The transformer references `_kernels._structural_pass_at_actions` via the MODULE attribute (so the perf-spy monkeypatch is observed):

```python
def structural_pass_xfns(
    *,
    home_team_id: int | str,
    params: StructuralPassParams | None = None,
) -> list:
    """VAEP xfn factory: ONE FrameAwareTransformer emitting structural_lbs/sgm/sdi x 3
    gamestate slots = 9 columns (structural_lbs_a0 .. structural_sdi_a2). Calls the
    SHARED _kernels._structural_pass_at_actions once per slot (3x, not 9x).

    See NOTICE for full bibliographic citations.
    """
    col_names = ["structural_lbs", "structural_sgm", "structural_sdi"]

    def _structural_pass_transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for i in range(min(3, len(states))):
                for col in col_names:
                    out[f"{col}_a{i}"] = np.nan
            return out
        for i, slot in enumerate(states[:3]):
            batch = _kernels._structural_pass_at_actions(
                slot, frames, home_team_id=home_team_id, params=params
            )
            for col in col_names:
                out[f"{col}_a{i}"] = batch[col].to_numpy()
        return out

    _structural_pass_transformer._frame_aware = True  # type: ignore[attr-defined]
    _structural_pass_transformer.__name__ = "structural_pass"
    return [_structural_pass_transformer]
```

Add `"structural_pass_xfns"` to `__init__.py` `__all__` and the `from .features import (...)` block.

- [ ] **Step 4: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestXfns tests/tracking/test_structural_pass_perf_budget.py -q`
Expected: PASS (3 tests)

---

## Task 5A: Shared dup-safe frame-id resolver + DRY refactor

> **Context (folded into this PR per maintainer):** the per-slot `pointers.set_index("action_id").at[aid, "frame_id"]` pattern is a **systemic latent bug**: VAEP shifted gamestate slots repeat the boundary action, so `action_id` is non-unique and `.at` returns a Series → the transformer raises. **Empirically confirmed broken** through real `gamestates()`: `pitch_control_xfns`, `obso_xfns`, `pausa_xfns`, `space_creation_xfns`, `pressure_default_xfns`, `cover_shadow_xfns`, `gk_influence_xfns`, `player_influence_xfns`. (Safe families use `_resolve_action_frame_context` merges or dedup'd lookups.) A grep of `.at[...]` does NOT capture all of them (e.g. `pausa` breaks via a different mechanism), so the fix is **driven by a behavioral gate** (Task 5B), not a hand-listed seam inventory.

**Files:**
- Modify: `silly_kicks/tracking/_kernels.py`
- Test: `tests/tracking/test_structural_pass.py` (kernel tests already cover this path)

- [ ] **Step 1: Add the shared resolver to `_kernels.py`**

```python
def resolve_frame_ids_by_position(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    links: pd.DataFrame | None = None,
) -> np.ndarray:
    """Float64 array (len == len(actions)): linked frame_id per action ROW, NaN if
    unlinked. Robust to NON-UNIQUE action_id (VAEP shifted gamestate slots repeat the
    boundary action). NEVER uses .at on a possibly-non-unique index.

    add_*-supplied `links` are keyed by the unique action_id; the internal-link path
    re-keys to a unique positional surrogate before linking.
    """
    from .utils import link_actions_to_frames

    n = len(actions)
    if n == 0:
        return np.full(0, np.nan)
    if links is not None:
        # links action_id and actions action_id are both the post-link int64 SPADL id
        # (link_actions_to_frames casts to int64); reindex aligns by exact id. The
        # internal-link path below uses an int64 surrogate, so both paths are
        # dtype-consistent — locked by test_resolve_frame_ids_by_position.
        fid = links.drop_duplicates("action_id").set_index("action_id")["frame_id"]
        return fid.reindex(actions["action_id"].to_numpy()).to_numpy(dtype="float64")
    link_input = actions.copy()
    link_input["action_id"] = np.arange(n)  # unique positional surrogate
    pointers, _ = link_actions_to_frames(link_input, frames)
    return pointers.set_index("action_id")["frame_id"].reindex(np.arange(n)).to_numpy(dtype="float64")
```

- [ ] **Step 2: Write the resolver-equivalence test (the linchpin for the whole sweep)**

This is the single highest-leverage test: every retrofitted family routes frame-id resolution through this one function, so it must be **byte-equivalent** to the old `set_index("action_id").at[aid, "frame_id"]` on unique action_ids. Append to `tests/tracking/test_structural_pass.py`:

```python
class TestResolveFrameIdsByPosition:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0], home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0], away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_unique_links_equals_old_at_lookup(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        links, _ = link_actions_to_frames(acts, frame)
        got = resolve_frame_ids_by_position(acts, frame, links=links)
        # old path, value-for-value:
        pl = links.set_index("action_id")
        old = np.array([
            float(pl.at[a, "frame_id"]) if (a in pl.index and pd.notna(pl.at[a, "frame_id"])) else np.nan
            for a in acts["action_id"]
        ])
        np.testing.assert_array_equal(np.isnan(got), np.isnan(old))
        np.testing.assert_array_equal(got[~np.isnan(got)], old[~np.isnan(old)])

    def test_links_path_equals_internal_link_path(self):
        # the equivalence the safe aggregator path relies on
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position
        from silly_kicks.tracking.utils import link_actions_to_frames

        acts, frame = _actions(), self._frame()
        links, _ = link_actions_to_frames(acts, frame)
        a = resolve_frame_ids_by_position(acts, frame, links=links)
        b = resolve_frame_ids_by_position(acts, frame)  # internal surrogate link
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(a[~np.isnan(a)], b[~np.isnan(b)])

    def test_duplicate_action_id_position_aligned_no_raise(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

        acts, frame = _actions(), self._frame()
        acts["action_id"] = [1, 1]  # duplicate
        out = resolve_frame_ids_by_position(acts, frame)  # must not raise
        assert len(out) == 2

    def test_unlinked_row_nan(self):
        from silly_kicks.tracking._kernels import resolve_frame_ids_by_position

        frame = self._frame()
        frame["time_seconds"] = 500.0  # far from action times -> no link
        out = resolve_frame_ids_by_position(_actions(), frame)
        assert np.isnan(out).all()
```

- [ ] **Step 3: Run the resolver-equivalence test**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestResolveFrameIdsByPosition -q`
Expected: PASS (4 tests). **Do not proceed to the 8-family retrofit until this is green** — it is the contract every retrofit depends on.

- [ ] **Step 4: Refactor `_structural_pass_at_actions` to use it (DRY)**

Replace the inline `if links is not None: ... else: ...` frame-id resolution block in `_structural_pass_at_actions` (Task 3) with:

```python
    fid_by_pos = resolve_frame_ids_by_position(actions, frames, links=links)
```

- [ ] **Step 5: Re-run the structural_pass kernel + xfns tests (no regression)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py -q`
Expected: PASS (all).

---

## Task 5B: Red-first behavioral gate over ALL frame-aware xfns

**Files:**
- Create: `tests/tracking/test_frame_aware_xfns_dup_action_id.py`

- [ ] **Step 1: Write the gate (enumerates every `*_xfns`, runs through real gamestates, meta-asserts coverage)**

```python
"""Behavioral gate (ADR-019-style): NO frame-aware xfns may raise on the non-unique
action_id that real VAEP gamestate slots carry at period boundaries. Enumerates the
registered surface so future xfns are auto-covered; a meta-assertion proves the gate
sees every *_xfns factory."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import silly_kicks.tracking.features as F
from silly_kicks.vaep.feature_framework import gamestates
from tests.tracking.test_defensive_line import _make_frame_rows


def _xt():
    from silly_kicks.xthreat import ExpectedThreat

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt


def _actions():
    return pd.DataFrame({
        "game_id": [1] * 4, "period_id": [1] * 4, "action_id": [10, 11, 12, 13],
        "time_seconds": [1.0, 2.0, 3.0, 4.0], "team_id": [1] * 4, "player_id": [5, 6, 7, 8],
        "start_x": [40.0, 45.0, 50.0, 55.0], "start_y": [34.0] * 4,
        "end_x": [70.0, 75.0, 60.0, 65.0], "end_y": [34.0] * 4,
        "type_id": [0] * 4, "result_id": [1] * 4, "bodypart_id": [0] * 4,
    })


# Complete frame fixture: enough columns that the ONLY failure mode is the dup-action_id
# bug (a missing column would be a fixture gap, not the bug — see _run_family).
def _frame():
    fr = _make_frame_rows(
        home_outfield_xs=[20.0, 22.0, 24.0, 26.0], home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
        away_outfield_xs=[50.0, 60.0, 30.0, 80.0], away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
    )
    fr["vx"] = 0.0
    fr["vy"] = 0.0
    fr["z"] = 0.0
    fr["speed"] = 0.0
    fr["ball_state"] = "alive"
    return fr


# Factory name -> built list of transformers. Covers the entire *_xfns surface.
# Construction MUST succeed (no silent skip) — an unconstructable factory is a gate
# FAILURE, not a skip, so a family can never go unprobed (no-silent-caps discipline).
_CONSTRUCT_ALLOWLIST: set[str] = set()  # factories that genuinely cannot construct (none today)


def _build(name):
    fac = getattr(F, name)
    if isinstance(fac, list):
        return fac
    xt = _xt()
    for args, kw in (((), {"home_team_id": 1}), ((xt,), {"home_team_id": 1})):
        try:
            return fac(*args, **kw)
        except TypeError:
            continue
    raise AssertionError(
        f"{name}: no known construction signature — extend _build (do NOT skip; an "
        f"unprobed family re-opens the hole this gate closes)."
    )


_DUP_SIGNATURE = "truth value of a Series is ambiguous"

_XFNS_NAMES = sorted(n for n in dir(F) if n.endswith("_xfns"))


def test_meta_gate_covers_every_xfns_factory():
    # The gate parametrization must equal the registered surface (no silent gaps).
    assert set(_XFNS_NAMES) == {n for n in dir(F) if n.endswith("_xfns")}
    assert len(_XFNS_NAMES) >= 20
    assert not _CONSTRUCT_ALLOWLIST, "no construct-skips are expected today"


def _run_family(name):
    """Run every frame-aware transformer of `name` through a dup-action_id gamestate.
    Discriminates the target bug from a fixture gap so 5C fixes the bug, not the fixture."""
    states = gamestates(_actions(), nb_prev_actions=3)
    assert states[1]["action_id"].duplicated().any()  # precondition: dup exists
    frame = _frame()
    for t in _build(name):
        if not getattr(t, "_frame_aware", False):
            continue
        try:
            t(states, frame)
        except Exception as exc:  # noqa: BLE001
            if _DUP_SIGNATURE in str(exc):
                raise AssertionError(
                    f"{name}: DUP-ACTION_ID BUG — retrofit to resolve_frame_ids_by_position (Task 5C)."
                ) from exc
            raise AssertionError(
                f"{name}: non-dup error ({type(exc).__name__}: {exc}). This is a FIXTURE GAP — "
                f"extend _frame(), do NOT alter the family's logic."
            ) from exc


@pytest.mark.parametrize("name", _XFNS_NAMES)
def test_xfns_survives_duplicate_action_id_gamestate(name):
    _run_family(name)  # MUST NOT raise on the non-unique action_id
```

- [ ] **Step 2: Run the gate RED — it enumerates the broken families**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_frame_aware_xfns_dup_action_id.py -q`
Expected: the meta + structural_pass + the known-safe families PASS; the broken families FAIL with the explicit **"DUP-ACTION_ID BUG"** message (the discriminated `_DUP_SIGNATURE`). If any family fails with **"FIXTURE GAP"** instead, extend `_frame()` first — that family is not necessarily a dup-bug. Record the DUP-ACTION_ID set — that is the retrofit checklist for Task 5C. (Known at authoring time: `pitch_control_xfns`, `pitch_control_default_xfns`, `obso_xfns`, `pausa_xfns`, `space_creation_xfns`, `pressure_default_xfns`, `cover_shadow_xfns`, `gk_influence_xfns`, `player_influence_xfns`.)

---

## Task 5C: Retrofit each broken family to the shared resolver (gate → green)

**Files:**
- Modify: `silly_kicks/tracking/features.py` (the broken transformers), possibly `silly_kicks/tracking/_kernels.py`

For EACH family the gate reported failing, replace its per-slot
`pointer_lookup = pointers.set_index("action_id")` + per-row `fid_raw = pointer_lookup.at[aid, "frame_id"]`
with the dup-safe positional resolver. The retrofit pattern (apply per transformer):

- [ ] **Step 1: Apply the retrofit pattern to each failing family**

```python
# BEFORE (inside the per-slot loop body of a *_xfns transformer):
    pointers, _ = link_actions_to_frames(slot, frames)
    pointer_lookup = pointers.set_index("action_id")
    for j, (_idx, row) in enumerate(slot.iterrows()):
        aid = row["action_id"]
        if aid not in pointer_lookup.index:
            continue
        fid_raw = pointer_lookup.at[aid, "frame_id"]   # <-- raises on non-unique aid
        if pd.isna(fid_raw):
            continue
        fid = int(float(fid_raw))
        ...

# AFTER:
    fid_by_pos = _kernels.resolve_frame_ids_by_position(slot, frames)   # one call per slot
    for j, (_idx, row) in enumerate(slot.iterrows()):
        if np.isnan(fid_by_pos[j]):
            continue
        fid = int(fid_by_pos[j])
        ...
```

Notes for the executor:
- The `j` index is the enumerate position — `fid_by_pos` is positionally aligned to `slot`, so `fid_by_pos[j]` is the right row. Drop the `aid`/`pointer_lookup` lines entirely.
- If a transformer caches per `(period_id, frame_id, ...)`, keep that cache; only the frame-id *resolution* changes.
- A few families (e.g. `pausa`, `pressure_default_xfns`) reach the bug via a lifted per-Series helper or a different internal path, not a literal `.at` in the transformer — fix wherever that family's slot-keyed `set_index("action_id")` lives (the gate is the authority: keep going until that family is green). Cast `period_id` to `int` before any `get_group` if the family uses one (shift promotes it to float).
- Where the bug lives in a **helper shared** by the safe `add_*` aggregator and the broken xfns (likely `pitch_control`/`obso`), the resolver swap **will** touch the aggregator path too — that is fine and expected. The contract is: the **aggregator's observable behavior must not change**, guaranteed by `TestResolveFrameIdsByPosition` (resolver ≡ old `.at` on unique ids) + the aggregator's own existing tests. Do NOT alter the safe families (`das`, `defensive_line`, `ghost_gk`, `xshot`, etc.) — they resolve via merges/dedup and the gate already passes them.

- [ ] **Step 2: Run the gate GREEN**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_frame_aware_xfns_dup_action_id.py -q`
Expected: ALL PASS (every `*_xfns` survives the dup-action_id slot).

- [ ] **Step 3: Run each retrofitted family's existing tests (no behavior regression)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_cover_shadows.py tests/tracking/test_gk_influence.py tests/tracking/test_player_influence.py tests/tracking/test_pitch_control.py tests/tracking/test_obso.py tests/tracking/test_pausa.py tests/tracking/test_space_creation.py tests/tracking/test_pressure*.py -q`
Expected: PASS (the retrofit changes only the dup-unsafe frame-id resolution; aggregator/unique-id paths are byte-identical).

---

## Task 6: Atomic mirror + parity test (TDD)

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/tracking/test_structural_pass.py`

- [ ] **Step 1: Write the failing atomic-parity test (REAL dx/dy)**

Append to `tests/tracking/test_structural_pass.py`:

```python
class TestAtomicMirror:
    def _frame(self):
        return _make_frame_rows(
            home_outfield_xs=[20.0, 22.0, 24.0, 26.0],
            home_outfield_ys=[20.0, 30.0, 40.0, 50.0],
            away_outfield_xs=[50.0, 60.0, 30.0, 80.0],
            away_outfield_ys=[34.0, 20.0, 34.0, 34.0],
        )

    def test_atomic_add_matches_standard_with_real_dxdy(self):
        from silly_kicks.tracking.features import add_structural_pass as std
        from silly_kicks.atomic.tracking.features import add_structural_pass as atom

        std_acts = _actions()  # start=(40,34) end=(70,34)
        std_res = std(std_acts, self._frame(), home_team_id=1)

        atom_acts = pd.DataFrame({
            "game_id": [1, 1], "action_id": [1, 2], "period_id": [1, 1],
            "time_seconds": [1.0, 1.0], "team_id": [1, 1], "player_id": [50, 51],
            "x": [40.0, 0.0], "y": [34.0, 34.0],
            "dx": [30.0, 5.0], "dy": [0.0, 0.0],  # end = x+dx, y+dy == standard
            "type_id": [0, 0],
        })
        atom_res = atom(atom_acts, self._frame(), home_team_id=1)
        assert atom_res["structural_lbs"].iloc[0] == std_res["structural_lbs"].iloc[0] == 2

    def test_atomic_xfns_synthesizes_endpoints(self):
        from silly_kicks.atomic.tracking.features import structural_pass_xfns

        atom_state = pd.DataFrame({
            "game_id": [1], "action_id": [1], "period_id": [1], "time_seconds": [1.0],
            "team_id": [1], "player_id": [50], "x": [40.0], "y": [34.0],
            "dx": [30.0], "dy": [0.0], "type_id": [0],
        })
        t = structural_pass_xfns(home_team_id=1)[0]
        cols = t([atom_state, atom_state, atom_state], self._frame())
        assert cols["structural_lbs_a0"].iloc[0] == 2
```

- [ ] **Step 2: Run to verify failure**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py::TestAtomicMirror -q`
Expected: FAIL — `cannot import name 'add_structural_pass'` from atomic

- [ ] **Step 3: Implement the atomic mirror**

In `silly_kicks/atomic/tracking/features.py`: re-export `compute_structural_pass_metrics` + `StructuralPassParams` directly (add to the existing `from silly_kicks.tracking.features import (...)` block and `__all__`). Then add the endpoint-synthesizing wrappers:

```python
def _atomic_to_endpoints(actions):
    """Synthesize start_x/start_y/end_x/end_y from atomic x,y,dx,dy. structural_pass
    needs the RECEIVER (end), so a passer-only x->start_x rename is insufficient."""
    adapted = actions.copy()
    adapted["start_x"] = adapted["x"]
    adapted["start_y"] = adapted["y"]
    adapted["end_x"] = adapted["x"] + adapted["dx"]
    adapted["end_y"] = adapted["y"] + adapted["dy"]
    return adapted


def add_structural_pass(actions, frames, *, home_team_id, links=None, params=None):
    """Atomic-SPADL aggregator for structural-pass primitives. Synthesizes end_x/end_y
    from x+dx / y+dy (atomic has no end_*), delegates to the standard aggregator."""
    from silly_kicks.tracking.features import add_structural_pass as _std

    adapted = _atomic_to_endpoints(actions)
    result = _std(adapted, frames, home_team_id=home_team_id, links=links, params=params)
    return result.drop(columns=["start_x", "start_y", "end_x", "end_y"])


def structural_pass_xfns(*, home_team_id, params=None):
    """Atomic VAEP factory: same as the standard one but each gamestate slot has its
    end_x/end_y synthesized from x,y,dx,dy before the shared kernel runs."""
    from silly_kicks.tracking.features import structural_pass_xfns as _std_xfns

    inner = _std_xfns(home_team_id=home_team_id, params=params)[0]

    def _atomic_transformer(states, frames):
        adapted_states = [_atomic_to_endpoints(s) for s in states]
        return inner(adapted_states, frames)

    _atomic_transformer._frame_aware = True
    _atomic_transformer.__name__ = "structural_pass"
    return [_atomic_transformer]
```

Add `"add_structural_pass"`, `"structural_pass_xfns"`, `"compute_structural_pass_metrics"`, `"StructuralPassParams"` to the atomic module `__all__`.

- [ ] **Step 4: Run to verify pass**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py -q`
Expected: PASS (all)

---

## Task 7: σ-tuning reproducibility script

**Files:**
- Create: `scripts/tune_structural_pass_sigma.py`

- [ ] **Step 1: Write the owner-gated tuning script**

Create `scripts/tune_structural_pass_sigma.py` — a cleaned-up, repo-resident version of the investigation run that reproduces the D1 σ-sweep table (the source of the frozen σ=15 and the e2e SGM ceiling). It pulls GS matches via `scripts/_loader_pining.load_matches` (owner-gated; mirrors `scripts/train_*.py`), computes `_structural_pass_core` over a σ grid for open-play successful passes, and prints the AUC / conditioning / silhouette table.

```python
"""Reproduce the TF-45 SGM sigma choice (frozen default 15.0) on real WC2022 GS data.

Owner-gated (needs PINING_FOR_THE_DATA_TOKEN). Emits the D1 sigma-sweep table used to
justify sigma=15 and the e2e SGM-conditioning ceiling (max|SGM|<=200, p99<=20).

Usage: python scripts/tune_structural_pass_sigma.py [n_matches]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from scripts._loader_pining import load_matches
from silly_kicks.tracking._structural_pass import _structural_pass_core
from silly_kicks.tracking.utils import link_actions_to_frames

SIGMAS = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
FINAL_THIRD_X = 70.0


def main(n_matches: int) -> None:
    records: list[dict] = []
    for _prov, mid, actions, frames, home in load_matches(
        providers=["gradientsports"], max_per_provider=n_matches, tracking_limit=None
    ):
        home_id = int(home)
        passes = actions[(actions["type_id"] == 0) & (actions["result_id"] == 1)].copy()
        if passes.empty:
            continue
        pointers, _ = link_actions_to_frames(passes, frames, on_low_coverage="ignore")
        ptr = pointers.set_index("action_id")["frame_id"]
        outf = frames[(~frames["is_ball"].astype(bool)) & (~frames["is_goalkeeper"].astype(bool))]
        fg = outf.groupby(["period_id", "frame_id"])
        for _, row in passes.iterrows():
            fid = ptr.get(row["action_id"])
            if pd.isna(fid):
                continue
            try:
                fr = fg.get_group((int(row["period_id"]), int(fid)))
            except KeyError:
                continue
            opp = fr[fr["team_id"].astype(str) != str(row["team_id"])]
            dx = opp["x"].to_numpy(float)
            dy = opp["y"].to_numpy(float)
            ok = np.isfinite(dx) & np.isfinite(dy)
            dx, dy = dx[ok], dy[ok]
            if dx.size == 0:
                continue
            if int(row["team_id"]) != home_id:
                dx, dy = 105.0 - dx, 68.0 - dy
            d = np.column_stack([dx, dy])
            sx, sy, ex, ey = (float(row[c]) for c in ("start_x", "start_y", "end_x", "end_y"))
            rec = {"enters_third": (sx < FINAL_THIRD_X) and (ex >= FINAL_THIRD_X)}
            for sig in SIGMAS:
                _, sgm, _ = _structural_pass_core(d, (sx, sy), (ex, ey), sig)
                rec[f"sgm_{sig}"] = sgm
            records.append(rec)

    df = pd.DataFrame(records)
    label = df["enters_third"].to_numpy(bool)
    from sklearn.metrics import roc_auc_score

    print(f"passes={len(df)} base_rate_enters_third={label.mean():.3f}")
    print(f"{'sigma':>6} {'sgmAUC':>7} {'p99abs':>9} {'maxabs':>9}")
    for sig in SIGMAS:
        s = df[f"sgm_{sig}"].to_numpy(float)
        m = np.isfinite(s)
        auc = roc_auc_score(label[m], s[m]) if label[m].any() and not label[m].all() else float("nan")
        print(f"{sig:>6.1f} {auc:>7.4f} {np.percentile(np.abs(s[m]), 99):>9.3f} {np.abs(s[m]).max():>9.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
```

- [ ] **Step 2: Smoke-check it imports (no token run required here)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -c "import ast; ast.parse(open(r'scripts/tune_structural_pass_sigma.py', encoding='utf-8').read()); print('OK')"`
Expected: `OK`

(Optional owner-gated full run, if a token is present: `python scripts/tune_structural_pass_sigma.py 3` → reproduces the D1 table.)

---

## Task 8: Owner-gated e2e

**Files:**
- Create: `tests/tracking/test_structural_pass_e2e.py`

- [ ] **Step 1: Write the e2e (predicate-selected forward pass; concrete SGM ceiling)**

Create `tests/tracking/test_structural_pass_e2e.py`:

```python
"""Owner-gated WC2022 GS e2e for structural_pass (TF-45). Needs PINING_FOR_THE_DATA_TOKEN.

The LBS-AUC assertion is a correctness/regression guard, NOT a reproduction of the
paper's progression finding (structural_lbs > 0 <=> forward pass, so it is partly
tautological). All validation metrics use open-play successful `pass` rows only.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")


@pytest.mark.skipif(not _TOKEN, reason="needs PINING_FOR_THE_DATA_TOKEN (owner-tier GS)")
def test_structural_pass_real_wc2022():
    from scripts._loader_pining import load_matches
    from silly_kicks.tracking.features import add_structural_pass

    parts = []
    for _p, _mid, actions, frames, home in load_matches(
        providers=["gradientsports"], max_per_provider=2, tracking_limit=None
    ):
        # Enrich on the FULL action stream, THEN filter — pre-filtering before the
        # link hides dropped actions from the ADR-017 coverage guard (utils.py:230-233),
        # producing spurious low-coverage warnings (review #3-D).
        enriched = add_structural_pass(actions, frames, home_team_id=int(home))
        passes = enriched[(enriched["type_id"] == 0) & (enriched["result_id"] == 1)].copy()
        passes["enters_third"] = (passes["start_x"] < 70.0) & (passes["end_x"] >= 70.0)
        parts.append(passes)
    df = pd.concat(parts, ignore_index=True)
    valid = df[df["structural_lbs"].notna()].copy()
    assert len(valid) > 500

    # 1. base-rate band (paper-consistent territorial-progression frequency)
    base = valid["enters_third"].mean()
    assert 0.07 <= base <= 0.13, base

    # 2. LBS regression guard (tautological, NOT a paper reproduction)
    from sklearn.metrics import roc_auc_score

    lab = valid["enters_third"].to_numpy(bool)
    auc = roc_auc_score(lab, valid["structural_lbs"].to_numpy(float))
    assert auc >= 0.70, auc

    # 3. targeted coordinate-frame invariant — predicate-selected at runtime (no frozen id)
    fwd = valid[(valid["end_x"] - valid["start_x"] > 25.0) & (valid["structural_lbs"] >= 1)]
    assert len(fwd) > 0, "expected forward passes with >=1 bypassed defender"

    # 4. SGM conditioning at sigma=15 (concrete ceilings; a drift to sigma=12 would trip)
    sgm = valid["structural_sgm"].to_numpy(float)
    sgm = sgm[np.isfinite(sgm)]
    assert np.abs(sgm).max() <= 200.0, np.abs(sgm).max()
    assert np.percentile(np.abs(sgm), 99) <= 20.0
```

- [ ] **Step 2: Verify it collects + skips cleanly without a token**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass_e2e.py -q`
Expected: 1 skipped (no token) — confirms no collection/import errors.

(If a token is present in the env, the test runs and must PASS.)

---

## Task 9: Full suite + lint + type gate (Shift Left)

- [ ] **Step 1: Run the structural-pass tests + the cross-cutting gates**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/tracking/test_structural_pass.py tests/tracking/test_structural_pass_perf_budget.py tests/test_enrichment_nan_safety.py tests/tracking/test_provenance_skip_guard.py -q`
Expected: PASS

- [ ] **Step 2: Run the full non-e2e suite (no regressions)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip`
Expected: PASS (3623+ passed; no new failures)

- [ ] **Step 3: Lint + format + type (replicate the full CI lint job)**

Run:
```bash
ruff check silly_kicks/ tests/ scripts/
ruff format --check silly_kicks/ tests/ scripts/
pyright silly_kicks/
```
Expected: all clean. (Fix any `# type: ignore[arg-type]` needs on `float(row[col])`/`pd.array` per the codebase idiom; run `ruff check --fix` for I001 import-sort.)

---

## Task 10: Docs — NOTICE, CLAUDE.md, TODO.md, CHANGELOG, C4

**Files:**
- Modify: `NOTICE`, `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `docs/c4/architecture.dsl`

- [ ] **Step 1: NOTICE — new reference block**

In `NOTICE`, after the TF-32 "Through the Gaps" block, add:

```
The per-pass structural primitives in silly_kicks/tracking/_structural_pass.py
(PR-S<NN>, TF-45) implement the methodology described in:

- Karakus, O., & Arkadas, H. (2026). "Structural Pass Analysis in Football:
  Learning Pass Archetypes and Tactical Impact from Spatio-Temporal Tracking
  Data." arXiv:2603.28916. (Line Bypass Score, Space Gain Metric, Structural
  Disruption Index.)
  Faithfulness caveats: (a) receiver location x_r is the pass DESTINATION
  (end_x/end_y) — SPADL has no receiver_player_id; (b) LBS is purely 1-D along
  the attacking axis (a far-touchline defender in the x-band is counted). The
  library ships RAW primitives only; the TIV z-norm composite, K-means
  archetypes, and passer/receiver rankings are corpus-level (consumer-side). The
  owner-gated e2e LBS-AUC is a regression guard, not a reproduction of the paper's
  progression result.
```

- [ ] **Step 2: CLAUDE.md — one PR-S line in the tracking architecture paragraph**

Add (sentence form, after the most recent `PR-S##` line in the tracking section):
`PR-S<NN> ships TF-45 structural-pass primitives (_structural_pass.py: _structural_pass_core pure math + compute_structural_pass_metrics per-frame + StructuralPassParams[sigma=15.0 empirically tuned]; add_structural_pass aggregator emitting raw structural_lbs/structural_sgm/structural_sdi via the shared _kernels._structural_pass_at_actions + structural_pass_xfns VAEP factory; atomic mirror with end=x+dx synthesis; arXiv:2603.28916). Library = raw primitives only (TIV/archetypes/rankings consumer-side). Also fixes a systemic dup-action_id crash in ~8 frame-aware xfns families (pitch_control/obso/pausa/space_creation/pressure/cover_shadow/gk_influence/player_influence) via shared _kernels.resolve_frame_ids_by_position + a behavioral gate over all *_xfns.`

- [ ] **Step 3: TODO.md — close the TF-45 row**

Mark the Tier-5 TF-45 row shipped, and fix the two errors the investigation found: metric NAMES are **Line Bypass Score / Space Gain Metric / Structural Disruption Index** (not "Line-Breaking/Spatial Gain"); the reuse claims (`compute_defensive_line` centroid; `_cover_shadows` Gaussian) were both wrong — SDI uses a fresh full-team 2-D centroid, SGM is a direct KDE. Record σ=15 (empirically tuned).

- [ ] **Step 4: CHANGELOG.md — feature entry + systemic-fix entry + version bump**

Add a new version section (next free minor after `origin/main` per the version-bump checklist — reconcile at commit time, do NOT pre-reserve). Two entries: (1) the TF-45 feature; (2) **a `### Fixed` entry for the systemic dup-`action_id` xfns crash** — list the ~8 affected families and note it is a **VAEP-feature-path behavior change** (those `*_xfns` previously raised when composed into gamestates; they now produce values → Hyrum/retrain note for any consumer using the xfns path). Bump all version sites: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md` header, and re-lock with `uv lock`.

- [ ] **Step 5: ADR — frame-aware xfns frame-id resolution invariant**

The systemic fix establishes a new cross-cutting invariant + conformance test, exactly the codebase's "shared control + conformance gate" pattern (cf. ADR-003 nan-safety + its CI gate, ADR-019 id-dtype contract + its gate). Create `docs/superpowers/adrs/ADR-0NN-frame-aware-xfns-frame-id-resolution.md` (next free number — 015 or 020+; reconcile at commit time, do NOT pre-reserve). State the decision: **"Frame-aware xfns MUST resolve a linked frame_id by position via `_kernels.resolve_frame_ids_by_position`, NEVER via `set_index('action_id').at[...]` — VAEP shifted gamestate slots carry non-unique action_id at period boundaries. Enforced by `tests/tracking/test_frame_aware_xfns_dup_action_id.py` (auto-enumerates every `*_xfns` + meta-asserts coverage)."** Context = the empirically-confirmed ~8-family blast radius; Consequences = the VAEP-path behavior change (crash→values) + the auto-coverage gate for future xfns. Cross-link from `resolve_frame_ids_by_position`'s docstring. (ADR-005 still covers structural_pass itself; this ADR is for the dup-safety invariant only.)

- [ ] **Step 6: C4 — aggregator count bump + regen**

In `docs/c4/architecture.dsl`, find the `tracking` container description and increment the enumerated aggregator **count** (a new `add_*` aggregator was added). Then regen the HTML via the `mad-scientist-skills:c4` pipeline (structurizr.war → plantuml.jar → c4_assemble.py; Java 21 + jars in `~/.claude/tools/`). Confirm only the count token changed (no new KDE backend / trained model).

Run (verify the DSL still parses / regen succeeds): per the c4 skill.

- [ ] **Step 7: Re-run lint after doc/code edits**

Run: `ruff check silly_kicks/ tests/ scripts/ && ruff format --check silly_kicks/ tests/ scripts/`
Expected: clean.

---

## Task 11: Single commit + PR (GATED on explicit approval)

- [ ] **Step 1: Final verification (evidence before claiming done)**

Run: `D:/Development/karstenskyt__silly-kicks_part-deux/.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip ; echo "EXIT: ${PIPESTATUS[0]}"`
Expected: PASS, EXIT 0.

- [ ] **Step 2: Present command + full diff to the maintainer and HOLD**

Do NOT create the `~/.claude-git-approval` sentinel. Present `git status`, the full `git diff`, and the proposed commit message. Wait for explicit per-commit approval (the maintainer creates the sentinel or authorizes it).

- [ ] **Step 3: On approval — single commit (message via temp file, NOT inline heredoc)**

Write the message to `.git/COMMIT_TF45.txt` (apostrophes safe), then:
```bash
git add -A
git commit -F .git/COMMIT_TF45.txt
```
Message ends with:
`Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`

- [ ] **Step 4: Push + open PR (body via file)**

```bash
git push -u origin feat/tf45-structural-pass
gh pr create --title "feat(tracking): TF-45 structural-pass primitives (LBS/SGM/SDI) -- silly-kicks <ver>" --body-file .git/PR_TF45.md
```
Remove the temp message/PR files afterward.

---

## Self-Review (spec coverage)

- **D1 σ=15** → Task 0 (param default) + Task 7 (reproducibility script) + Task 8 (e2e ceiling). ✓
- **D2 receiver=end_x/end_y** → Task 2 (`receiver_xy` from end coords); Task 6 atomic synthesizes `end=x+dx`. ✓
- **D3 all pass+cross, NA otherwise** → Task 3 kernel `type_id in (0,1)` gate; Task 4 non-pass test. ✓
- **D4 primitives only** → no TIV/archetype/ranking code anywhere. ✓
- **Pure hexagonal core (M4)** → Task 1. **Shared kernel + 3× budget (🟠2)** → Tasks 3, 5. ✓
- **Namespaced columns (S2)** → all tasks assert `structural_*`; exact key-set in Task 2. ✓
- **0-defender NaN vs LBS=0 (S1)** → Task 1 core tests. ✓
- **`~ids_match` + `~is_ball` first (🟡4)** → Task 2 impl + GK/ball test. ✓
- **Atomic endpoint synthesis (🔴)** → Task 6, real-dx/dy parity. ✓
- **NaN-safety marker + gate (🟠1)** → Task 4 Steps 3/6. ✓
- **Coordinate invariant + targeted e2e (🟠3)** → module docstring (Task 0) + Task 8 forward-pass predicate. ✓
- **e2e PA-only + honesty caveat (M2/M3)** → Task 8. **High-NA slots (M5)** → expected, no NaN-crash (Task 5 covers transformer). ✓
- **is_default dropped (M1)** → Task 0 param has none. ✓
- **NOTICE/CLAUDE/TODO/CHANGELOG/C4** → Task 10. ✓
- **Systemic dup-action_id xfns fix (folded in)** → shared resolver (Task 5A), red-first behavioral gate over all `*_xfns` + meta-assert (Task 5B), retrofit ~8 families to green (Task 5C), CHANGELOG `### Fixed` + Hyrum note (Task 10 Step 4). ✓

---

## Cross-session plan review resolutions (2026-06-07)

All verified against real code; the 🟠A finding was confirmed empirically (`gamestates(_actions())[1].action_id == [10.0, 10.0, 11.0, 12.0]`; `set_index("action_id").at[10]` → Series → `int(float(...))` raises `TypeError`).

- 🟠 **A — non-unique `action_id` in shifted gamestate slots would crash the xfns path** (unit/perf tests false-greened on 3 identical slots). **Fixed:** kernel resolves frame_id **by position** via a unique surrogate + vectorized `reindex` (never `.at` on a possibly-non-unique index); `period_id` cast to `int` before `get_group` (shift promotes it to float). New regression tests: `test_duplicate_action_id_in_slot_does_not_raise` (Task 3) + `test_real_gamestates_with_duplicate_action_ids` (Task 5, end-to-end via real `gamestates()`).
- 🟡 **B — frame-id dtype safety.** Kept the per-action `get_group` idiom (consistent with the per-action-loop peers `add_cover_shadows`/`add_player_influence`, not the precompute-merge `_defensive_line_at_actions`), with a documented **numeric-`(period_id, frame_id)`** assumption (true for every provider; only team/player/match ids are string). Added `test_string_team_id_dtype_safe` to confirm the actual dtype hazard (object `team_id`) is handled by `ids_match`/`same_id`.
- 🟡 **C — Int64 idiom.** Use `batch["structural_lbs"].astype("Int64")` (features.py:1154), not `pd.array(..., dtype="Int64")`.
- 🟡 **D — e2e pre-filter coverage warning.** Enrich the FULL action stream then filter to open-play `pass`, so the ADR-017 link-coverage guard sees the real population (no spurious low-coverage warnings).
- 🟢 **Minors.** Strengthened the provenance test to assert `frame_id` present + `frame_id_x/_y` absent (real merge-suffix hazard); added a one-line note that the per-pass `iterrows`/`get_group` is per-pass-inherent (O(1) lookup, not the O(n·m) mask antipattern); version-bump footgun is N/A (library wheel).

### Systemic dup-action_id xfns bug — FOLDED INTO THIS PR (maintainer decision)

The flagged `pointers.set_index("action_id").at[aid]` pattern turned out to be **systemic**, not 2-3 functions. Empirically confirmed broken through real `gamestates()`: **`pitch_control_xfns`, `obso_xfns`, `pausa_xfns`, `space_creation_xfns`, `pressure_default_xfns`, `cover_shadow_xfns`, `gk_influence_xfns`, `player_influence_xfns`** (~8 families). A `.at[...]` grep does NOT capture all of them (`pausa` breaks via a different path), so the fix is driven by the **behavioral gate** (Task 5B), not a seam inventory — per the "behavioral gate over manual seam inventory" discipline. Safe families (merge-based `_resolve_action_frame_context` / dedup'd lookups): `das`, `defensive_line`, `elastic_sync`, `line_breaking_ward`, `off_ball_context`, `pre_shot_gk_default`, `pre_shot_gk_angle`, `shape_graph`, `team_shape`, `tracking_default`, `actor_pre_window`, `ghost_gk`, `xshot_occurrence`, `pre_shot_gk_full`. The maintainer chose to fold the fix into the single TF-45 commit (Tasks 5A–5C). The defensive `_defensive_line_at_actions` kernel (`_kernels.py:824`) already uses a positional pattern, confirming the hazard was known.

### Plan review round 2 (the systemic-sweep revision) — incorporated

- 🟠 **#1 Resolver-equivalence is the linchpin** — added `TestResolveFrameIdsByPosition` (Task 5A Step 2): unique-links ≡ old `.at` value-for-value; links-path ≡ internal-link path; dup → position-aligned no-raise; unlinked → NaN. Gated: do not retrofit until green. This is the contract all 8 retrofits depend on.
- 🟠 **#2 Gate must not silent-skip** — `_build` now **fails** (not `pytest.skip`) on an unknown construction signature; empty `_CONSTRUCT_ALLOWLIST` + meta-assert it stays empty, so no family goes unprobed (no-silent-caps).
- 🟠 **#3 Gate must discriminate bug vs fixture gap** — `_run_family` checks the `_DUP_SIGNATURE` ("truth value of a Series is ambiguous") and fails with an explicit **"DUP-ACTION_ID BUG → retrofit"** vs **"FIXTURE GAP → extend `_frame()`, do NOT alter the family"** message. `_frame()` made complete (vx/vy/z/speed/ball_state). 5C "keep going until green" can no longer drive the executor to mangle a family for a fixture gap.
- 🟠 **#4 Document the systemic change** — added the ADR task (Task 10 Step 5: frame-aware xfns frame-id-resolution invariant + the gate as its conformance test) and the separate CHANGELOG `### Fixed` entry with the Hyrum/VAEP-retrain note.
- 🟡 **5C aggregator wording** — reworded: a shared-helper retrofit legitimately touches the aggregator path; the contract is "aggregator behavior unchanged," guaranteed by the equivalence test + the aggregator's own tests (not "don't touch").
- 🟡 **Resolver dtype** — added a comment that both paths use the int64 SPADL/surrogate id (no int-vs-float reindex miss), locked by the equivalence test.
- 🟡 **Stale `pd.array` note** — corrected to `Series.astype("Int64")`.
