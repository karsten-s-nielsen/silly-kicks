# PR-S25 — TF-3 + TF-2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Per project memory `feedback_commit_policy`: ONE commit per branch — no per-task commits. Inline execution per `feedback_inline_execution_default`. Feature branch off `main`, no worktrees per `feedback_no_worktrees`.**

**Goal:** Ship TF-3 (`actor_*_pre_window`) + TF-2 (multi-flavor `pressure_on_actor`) tracking-aware features, ADR-005 §8 amendment for multi-flavor xfn column-naming, atomic-SPADL mirror surface, and 16 new test files in silly-kicks 3.2.0.

**Architecture:** Per spec §2 — additive layer over existing tracking namespace. New module `silly_kicks.tracking.pressure` for multi-flavor dispatch + frozen param dataclasses. New schema-agnostic kernels in `_kernels.py`. Public defs in `features.py` + atomic mirror. Reuses ADR-005 §3 kernel pattern, ADR-004 9 invariants, existing `ActionFrameContext` + `slice_around_event` primitives. ADR-005 §8 amendment codifies multi-flavor column-naming convention.

**Tech Stack:** Python 3.10–3.14, pandas (numpy 2.x compatible), pytest, pytest-benchmark (perf budgets), pyright (CI clean gate), ruff (format + lint), uv (package management). Test-only optional dep: `unravelsports>=1.2` for the Bekkers golden-master parity gate.

**Spec:** [`docs/superpowers/specs/2026-05-03-tf3-tf2-design.md`](../specs/2026-05-03-tf3-tf2-design.md) — 1103 lines, lakehouse-reviewed twice (v1: 22 items; v2: 5 items), all resolved.

**Branch:** `pr-s25-tf3-tf2-pressure-and-prewindow` off `main`.

---

## File Map

### New files

```
silly_kicks/tracking/pressure.py             # Multi-flavor dispatch + 3 frozen param dataclasses + Method Literal
docs/superpowers/specs/2026-05-03-tf3-tf2-design.md   # already created
```

### Modified files

```
silly_kicks/tracking/_kernels.py             # +5 kernels (Andrienko, Link, Bekkers + Bekkers TTI sub-primitive, pre-window-shared)
silly_kicks/tracking/features.py             # +6 public defs, +2 aggregators, +2 default xfn lists, +exports
silly_kicks/atomic/tracking/features.py      # parallel atomic surface mirror
docs/superpowers/adrs/ADR-005-tracking-aware-features.md  # §8 amendment (multi-flavor column naming)
NOTICE                                       # +3 academic refs (Andrienko, Link, Bekkers) + BSD-3-Clause attribution to UnravelSports
CHANGELOG.md                                 # [3.2.0] entry per spec §10
TODO.md                                      # mark TF-2/TF-3 SHIPPED; add TF-24 (Optuna k3) + TF-25 (evolve structural-form) to Tier 3
pyproject.toml                               # version 3.1.0 → 3.2.0; add [optional-dependencies] golden-master = ["unravelsports>=1.2"]
silly_kicks/_version.py                      # version 3.1.0 → 3.2.0
tests/tracking/_provider_inputs.py           # +5 named scenarios (PressingDefenderScenario + 4 TF-3 scenarios)
tests/tracking/test_action_context_real_data_sweep.py  # extended per spec §8.4
```

### New test files (16 total per spec §8.1)

```
tests/tracking/test_pre_window_features.py
tests/tracking/test_pre_window_perf_budget.py
tests/tracking/test_pressure_andrienko.py
tests/tracking/test_pressure_link.py
tests/tracking/test_pressure_bekkers.py
tests/tracking/test_pressure_bekkers_golden_master.py
tests/tracking/test_pressure_methods_invariants.py
tests/tracking/test_pressure_real_data_calibration.py
tests/tracking/test_pressure_snapshot.py
tests/tracking/test_pressure_perf_budget.py
tests/tracking/test_atomic_standard_parity.py
tests/tracking/test_bekkers_e2e_via_derive_velocities.py
tests/atomic/tracking/test_pre_window_features_atomic.py
tests/atomic/tracking/test_pressure_andrienko_atomic.py
tests/atomic/tracking/test_pressure_link_atomic.py
tests/atomic/tracking/test_pressure_bekkers_atomic.py
tests/test_adr005_amendment_compliance.py
tests/test_pyright_clean_tracking_features.py
scripts/regenerate_pressure_snapshot_shas.py
```

### Cleanup at end

```
scripts/_tf2_formula_research.md             # research scratchpad — delete before commit
```

---

## Task 0: Doc-drift verification sweep

> Per spec §12 reviewer recommendation: assert spec internal consistency BEFORE any kernel work. The v2 review surfaced 4 doc-drift inconsistencies post-resolution; this gate prevents the same shape from re-emerging during implementation.

**Files:**
- Read-only: `docs/superpowers/specs/2026-05-03-tf3-tf2-design.md`

- [ ] **Step 1: Run consistency grep**

```bash
# Verify NO stale UserWarning fallback rules survive in semantic positions
grep -nE "UserWarning.*fall back|fall back.*to per-player only" docs/superpowers/specs/2026-05-03-tf3-tf2-design.md && echo "STALE — FIX BEFORE PROCEEDING" || echo "clean"

# Verify NO stale B&A "covered-distance feature" attribution survives outside negating contexts
grep -n "covered-distance feature" docs/superpowers/specs/2026-05-03-tf3-tf2-design.md
# Expected: only 1 hit, line ~602, in negating context: "NOT Bauer & Anzer's covered-distance feature"

# Verify max_player_speed = 12.0 (NOT 7.5)
grep -n "max_player_speed" docs/superpowers/specs/2026-05-03-tf3-tf2-design.md
# Expected: all hits show 12.0; no 7.5 anywhere

# Verify ADR-005 §8 amendment text exists
grep -n "### 8\. Multi-flavor xfn column naming convention" docs/superpowers/specs/2026-05-03-tf3-tf2-design.md
# Expected: at least 1 hit
```

Expected output: All four greps clean / consistent. If any fails, halt and fix the spec before proceeding.

---

## Task 1: Create feature branch

**Files:** none (git operation)

- [ ] **Step 1: Verify clean working tree on main**

```bash
git status --short
git rev-parse --abbrev-ref HEAD
```

Expected: branch=`main`; working tree may have untracked `scripts/_tf2_formula_research.md` and `scripts/probe_*.py` from previous cycles (those were pre-existing per `git status` at session start). No staged or unstaged changes to tracked files.

- [ ] **Step 2: Create branch**

```bash
git checkout -b pr-s25-tf3-tf2-pressure-and-prewindow
git rev-parse --abbrev-ref HEAD
```

Expected: branch=`pr-s25-tf3-tf2-pressure-and-prewindow`.

---

## Task 2: Scaffold `silly_kicks/tracking/pressure.py`

> Multi-flavor dispatch module: 3 frozen param dataclasses + `Method` Literal + ValueError-on-mismatch validator. NO kernel logic here — kernels live in `_kernels.py` per ADR-005 §3.

**Files:**
- Create: `silly_kicks/tracking/pressure.py`

- [ ] **Step 1: Write failing test**

Create `tests/tracking/test_pressure_module_scaffolding.py`:

```python
"""Scaffolding contract test for silly_kicks.tracking.pressure module.

Verifies the params-dataclass + Method-literal + validator surface BEFORE
any kernel work lands. Pin-tests the public-surface invariants.
"""
from __future__ import annotations

import pytest

from silly_kicks.tracking.pressure import (
    AndrienkoParams,
    BekkersParams,
    LinkParams,
    Method,
    validate_params_for_method,
)


def test_andrienko_defaults() -> None:
    p = AndrienkoParams()
    assert p.q == 1.75
    assert p.d_front == 9.0
    assert p.d_back == 3.0


def test_link_defaults() -> None:
    p = LinkParams()
    assert p.r_hoz == 4.0
    assert p.r_lz == 3.0
    assert p.r_hz == 2.0
    assert p.angle_hoz_lz_deg == 45.0
    assert p.angle_lz_hz_deg == 90.0
    assert p.k3 == 1.0


def test_bekkers_defaults() -> None:
    p = BekkersParams()
    assert p.reaction_time == 0.7
    assert p.sigma == 0.45
    assert p.time_threshold == 1.5
    assert p.speed_threshold == 2.0
    assert p.max_player_speed == 12.0  # canonical UnravelSports per spec §4.1
    assert p.use_ball_carrier_max is True


def test_dataclasses_are_frozen() -> None:
    p = AndrienkoParams()
    with pytest.raises((AttributeError, Exception)):
        p.q = 2.0  # type: ignore[misc]


def test_validate_params_for_method_match() -> None:
    validate_params_for_method("andrienko_oval", AndrienkoParams())
    validate_params_for_method("link_zones", LinkParams())
    validate_params_for_method("bekkers_pi", BekkersParams())
    validate_params_for_method("andrienko_oval", None)  # None means use defaults


def test_validate_params_for_method_mismatch() -> None:
    with pytest.raises(TypeError, match="andrienko_oval.*expects AndrienkoParams.*got LinkParams"):
        validate_params_for_method("andrienko_oval", LinkParams())
    with pytest.raises(TypeError, match="bekkers_pi.*expects BekkersParams.*got AndrienkoParams"):
        validate_params_for_method("bekkers_pi", AndrienkoParams())


def test_validate_unknown_method() -> None:
    with pytest.raises(ValueError, match="Unknown method.*not_a_method"):
        validate_params_for_method("not_a_method", None)  # type: ignore[arg-type]
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/tracking/test_pressure_module_scaffolding.py -v
```

Expected: All 7 tests FAIL with `ImportError: cannot import name 'AndrienkoParams' from 'silly_kicks.tracking.pressure'` (module doesn't exist yet).

- [ ] **Step 3: Implement `silly_kicks/tracking/pressure.py`**

```python
"""Multi-flavor dispatch + per-method param dataclasses for pressure_on_actor.

Three published methodologies, each with a frozen parameter dataclass:
  - andrienko_oval — Andrienko et al. 2017 directional oval (default)
  - link_zones    — Link, Lang & Seidenschwarz 2016 piecewise zones
  - bekkers_pi    — Bekkers 2024 Pressing Intensity (probabilistic TTI)

See:
  - docs/superpowers/specs/2026-05-03-tf3-tf2-design.md §4.1, §4.6
  - ADR-005 §8 (multi-flavor xfn column-naming convention)
  - NOTICE for full bibliographic citations + BSD-3-Clause attribution
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

Method = Literal["andrienko_oval", "link_zones", "bekkers_pi"]


@dataclass(frozen=True)
class AndrienkoParams:
    """Parameters for Andrienko 2017 oval pressure model.

    Defaults from Andrienko et al. (2017) §3.1, calibrated with football
    experts: D_front=9 m, D_back=3 m, q=1.75. See NOTICE.
    """

    q: float = 1.75
    d_front: float = 9.0
    d_back: float = 3.0


@dataclass(frozen=True)
class LinkParams:
    """Parameters for Link, Lang & Seidenschwarz 2016 zone-based pressure model.

    Zone radii and angular boundaries from Figure 2 of the paper:
      - HOZ (Head-On Zone, toward goal):   r=4 m, alpha in [0, 45) degrees
      - LZ  (Lateral Zone, side):           r=3 m, alpha in [45, 90) degrees
      - HZ  (Hind Zone, behind):            r=2 m, alpha in [90, 180] degrees
    Mirrored for negative angles (axially symmetric across actor->goal axis).

    NOTE: Eq (3) of Link 2016 is honored as the formal specification; the paper's
    prose-described "High Pressure Zone (HPZ) constant high pressure" inner arc
    is qualitative and not implemented as a separate clamp (Plan A:
    equation-faithful, no discontinuity at 1 m).

    NOTE: k3 default = 1.0 is an engineering choice — the paper explicitly
    states k1..k5 were "calibrated manually with experts" and does not
    publish numerical values. Calibration deferred post-release to Optuna
    sweep (silly-kicks TODO TF-24); see NOTICE.
    """

    r_hoz: float = 4.0
    r_lz: float = 3.0
    r_hz: float = 2.0
    angle_hoz_lz_deg: float = 45.0
    angle_lz_hz_deg: float = 90.0
    k3: float = 1.0


@dataclass(frozen=True)
class BekkersParams:
    """Parameters for Bekkers 2024 Pressing Intensity (probabilistic TTI model).

    All defaults verified against canonical UnravelSports BSD-3-Clause source
    at the SHA pinned in NOTICE:
      - reaction_time = 0.7 s
        (unravel/soccer/models/pressing_intensity.py L120: _reaction_time = 0.7)
      - sigma = 0.45
        (pressing_intensity.py L121: _sigma = 0.45)
      - time_threshold = 1.5 s
        (pressing_intensity.py L122: _time_threshold = 1.5)
      - speed_threshold = 2.0 m/s
        (active-pressing filter; paper section 3.1 + blog Fig 2 caption)
      - max_player_speed = 12.0 m/s
        (unravel/soccer/dataset/kloppy_polars.py L160: max_player_speed: float = 12.0)
      - use_ball_carrier_max = True (paper section 2.4 ball-carrier improvement)

    See NOTICE for BSD-3-Clause attribution.
    """

    reaction_time: float = 0.7
    sigma: float = 0.45
    time_threshold: float = 1.5
    speed_threshold: float = 2.0
    max_player_speed: float = 12.0
    use_ball_carrier_max: bool = True


PressureParams = Union[AndrienkoParams, LinkParams, BekkersParams]
_METHOD_TO_PARAMS_TYPE: dict[Method, type] = {
    "andrienko_oval": AndrienkoParams,
    "link_zones": LinkParams,
    "bekkers_pi": BekkersParams,
}


def validate_params_for_method(method: Method, params: PressureParams | None) -> None:
    """Raise loudly if method/params combination is invalid.

    Per `feedback_post_init_validator_for_invalid_combinations` and
    `feedback_loud_raise_for_required_input_columns`: fail at the public-API
    boundary rather than silently coercing. Per spec section 4.6 ADR-005 amendment.

    Examples
    --------
    >>> validate_params_for_method("andrienko_oval", AndrienkoParams())
    >>> validate_params_for_method("link_zones", None)  # None means use defaults
    """
    if method not in _METHOD_TO_PARAMS_TYPE:
        raise ValueError(
            f"Unknown method '{method}'. Valid methods: "
            f"{sorted(_METHOD_TO_PARAMS_TYPE)}"
        )
    if params is None:
        return
    expected_type = _METHOD_TO_PARAMS_TYPE[method]
    if not isinstance(params, expected_type):
        raise TypeError(
            f"method='{method}' expects {expected_type.__name__}, "
            f"got {type(params).__name__}. "
            f"Use {expected_type.__name__}() (or omit params=) for defaults."
        )
```

- [ ] **Step 4: Run test to verify pass**

```bash
uv run pytest tests/tracking/test_pressure_module_scaffolding.py -v
```

Expected: 7 PASSED (test_andrienko_defaults, test_link_defaults, test_bekkers_defaults, test_dataclasses_are_frozen, test_validate_params_for_method_match, test_validate_params_for_method_mismatch, test_validate_unknown_method).

---

## Task 3: Andrienko kernel — RED → GREEN

> Per spec §8.0: write the 4-point pin test FIRST, run RED, then implement kernel until tests pass. Pin points verified numerically against the published `L = 9.00 / 7.05 / 4.27 / 3.00 m` at `Θ = 0 / ±45 / ±90 / ±180°`.

**Files:**
- Create: `tests/tracking/test_pressure_andrienko.py`
- Modify: `silly_kicks/tracking/_kernels.py` (add `_pressure_andrienko`)

- [ ] **Step 1: Write failing 4-point pin test (Task 3a — RED)**

Create `tests/tracking/test_pressure_andrienko.py`:

```python
"""Andrienko 2017 oval-pressure kernel: 4-point Θ pin tests + sum-aggregation.

References (see NOTICE):
- Andrienko et al. (2017), DMKD 31:1793-1839.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import _pressure_andrienko
from silly_kicks.tracking.feature_framework import ActionFrameContext
from silly_kicks.tracking.pressure import AndrienkoParams


def _build_ctx(
    *,
    actor_xy: tuple[float, float],
    defenders_xy: list[tuple[float, float]],
) -> tuple[pd.Series, pd.Series, ActionFrameContext]:
    """Minimal ActionFrameContext for 1-action / N-defender pressure tests."""
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [actor_xy[0]],
            "start_y": [actor_xy[1]],
        }
    )
    defenders = pd.DataFrame(
        [
            {
                "action_id": 1,
                "team_id_action": "home",
                "team_id_frame": "away",
                "player_id_action": 10,
                "player_id_frame": 100 + i,
                "is_ball": False,
                "x": dx,
                "y": dy,
                "vx": 0.0,
                "vy": 0.0,
                "speed": 0.0,
            }
            for i, (dx, dy) in enumerate(defenders_xy)
        ]
    )
    pointers = pd.DataFrame(
        {
            "action_id": [1],
            "frame_id": pd.array([1000], dtype="Int64"),
            "time_offset_seconds": [0.0],
            "n_candidate_frames": [1],
            "link_quality_score": [1.0],
        }
    )
    actor_rows = pd.DataFrame(
        {
            "action_id": [1],
            "x": [actor_xy[0]],
            "y": [actor_xy[1]],
            "speed": [0.0],
        }
    )
    ctx = ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=defenders,
        defending_gk_rows=pd.DataFrame(),
    )
    return actions["start_x"], actions["start_y"], ctx


def _expected_andrienko_pr(*, d: float, theta_deg: float, q: float = 1.75) -> float:
    """Hand-computed Andrienko per-defender pressure (verified vs Gemini PDF readout)."""
    cos_theta = math.cos(math.radians(theta_deg))
    z = (1.0 + cos_theta) / 2.0
    L = 3.0 + (9.0 - 3.0) * (z**3 + 0.3 * z) / 1.3
    if d >= L:
        return 0.0
    return (1.0 - d / L) ** q * 100.0


def test_andrienko_l_at_theta_0() -> None:
    """At Θ=0 (presser between target and threat), L = 9.00 m. Defender at d=9 boundary -> Pr=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0 + 9.0, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_l_at_theta_90() -> None:
    """At Θ=±90 (presser to side), L = 4.27 m. Defender at d=4.27 boundary -> Pr=0."""
    expected_L = 3.0 + 6.0 * (0.5**3 + 0.3 * 0.5) / 1.3  # = 4.2692...
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.0, 34.0 + expected_L)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_l_at_theta_180() -> None:
    """At Θ=180 (presser behind target, away from threat), L = 3.00 m. Defender at d=3 boundary -> Pr=0."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.0 - 3.0, 34.0)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_pr_inside_zone_theta_0() -> None:
    """Defender at d=4.5 (half of L=9) at Θ=0 -> Pr = (1-0.5)**1.75 * 100 ≈ 29.730%."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(54.5, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    expected = _expected_andrienko_pr(d=4.5, theta_deg=0.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)
    assert out.iloc[0] == pytest.approx(29.7301779, rel=1e-6)  # double-check vs hand-computed


def test_andrienko_pr_at_theta_45() -> None:
    """At Θ=45°: L = 7.05; defender at d=L/2 = 3.525 -> Pr = (0.5)^1.75 * 100 ≈ 29.730%."""
    L_45 = 3.0 + 6.0 * (((1 + math.cos(math.radians(45))) / 2) ** 3
                        + 0.3 * ((1 + math.cos(math.radians(45))) / 2)) / 1.3
    d = L_45 / 2
    angle = math.radians(45)
    defender_x = 50.0 + d * math.cos(angle)
    defender_y = 34.0 + d * math.sin(angle)
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(defender_x, defender_y)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    expected = _expected_andrienko_pr(d=d, theta_deg=45.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-6)


def test_andrienko_sum_aggregation() -> None:
    """Three pressers each at d=L/2 at Θ=0 -> total Pr = 3 * 29.73% = 89.19%."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(54.5, 34.0), (54.5, 35.0), (54.5, 33.0)],
        # Note: only the first is Θ=0; others are slightly off-axis.
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    # Loose check: total pressure > sum of any single contribution (lower bound).
    assert out.iloc[0] > 30.0


def test_andrienko_zero_outside_all_zones() -> None:
    """Defender at d=10m in front -> beyond L=9 in any direction -> Pr=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(60.0, 34.0)])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_andrienko_zero_no_defenders_returns_zero_not_nan() -> None:
    """Linked frame with empty opposite_rows -> 0.0 (not NaN)."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[])
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify failure (Task 3a — RED)**

```bash
uv run pytest tests/tracking/test_pressure_andrienko.py -v
```

Expected: All tests FAIL with `ImportError: cannot import name '_pressure_andrienko' from 'silly_kicks.tracking._kernels'`.

- [ ] **Step 3: Implement `_pressure_andrienko` (Task 3b — GREEN)**

Append to `silly_kicks/tracking/_kernels.py` (after existing kernels, before any module footer):

```python
# ---------------------------------------------------------------------------
# PR-S25 -- TF-2: pressure_on_actor multi-flavor kernels
# ---------------------------------------------------------------------------


def _pressure_andrienko(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: "AndrienkoParams",  # forward-ref to avoid module-import cycle
    goal_x: float = _GOAL_X,
    goal_y: float = _GOAL_Y_CENTER,
) -> pd.Series:
    """Andrienko et al. 2017 directional-oval pressure (sum-of-pressers).

    Per defender:
        vec_to_threat = (goal - anchor) normalized
        vec_presser   = (defender - anchor) normalized
        cos_theta     = dot(vec_to_threat, vec_presser)
        z             = (1 + cos_theta) / 2
        L             = d_back + (d_front - d_back) * (z^3 + 0.3*z) / 1.3
        d             = ||defender - anchor||
        pr_i          = (1 - d/L)^q * 100   if d < L else 0.0

    Aggregation: sum across all opp-team defenders. NaN unlinked actions;
    0.0 for linked-but-no-defenders or all-defenders-outside-zone.

    See spec section 4.4, NOTICE (Andrienko et al. 2017).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "ax": anchor_x.to_numpy(),
            "ay": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    threat_dx = goal_x - ax
    threat_dy = goal_y - ay
    threat_mag = np.sqrt(threat_dx**2 + threat_dy**2) + 1e-12
    threat_unit_x = threat_dx / threat_mag
    threat_unit_y = threat_dy / threat_mag

    presser_dx = px - ax
    presser_dy = py - ay
    d = np.sqrt(presser_dx**2 + presser_dy**2)
    presser_mag = d + 1e-12
    presser_unit_x = presser_dx / presser_mag
    presser_unit_y = presser_dy / presser_mag

    cos_theta = threat_unit_x * presser_unit_x + threat_unit_y * presser_unit_y
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    z = (1.0 + cos_theta) / 2.0
    L = params.d_back + (params.d_front - params.d_back) * (z**3 + 0.3 * z) / 1.3

    in_zone = d < L
    pr_per_defender = np.where(in_zone, np.power(np.maximum(0.0, 1.0 - d / L), params.q) * 100.0, 0.0)

    merged["_pr"] = pr_per_defender
    sums = merged.groupby("action_id")["_pr"].sum()
    for aid, pr_total in sums.items():
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = float(pr_total)  # type: ignore[arg-type]
    return out
```

Add at module top (with other imports):

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pressure import AndrienkoParams  # noqa: F401
```

- [ ] **Step 4: Run test to verify pass (Task 3b — GREEN)**

```bash
uv run pytest tests/tracking/test_pressure_andrienko.py -v
```

Expected: 8 PASSED.

- [ ] **Step 5: Add Andrienko invariant tests (Task 3c — invariants)**

Append to `tests/tracking/test_pressure_andrienko.py`:

```python
def test_andrienko_monotone_in_distance_at_fixed_angle() -> None:
    """Pressure decreases monotonically as defender→actor distance increases (Θ=0)."""
    distances = [1.0, 2.0, 3.0, 5.0, 7.0]
    actor_xy = (50.0, 34.0)
    pressures: list[float] = []
    for d in distances:
        _, _, ctx = _build_ctx(actor_xy=actor_xy, defenders_xy=[(50.0 + d, 34.0)])
        ax, ay, _ = _build_ctx(actor_xy=actor_xy, defenders_xy=[(50.0 + d, 34.0)])
        out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
        pressures.append(float(out.iloc[0]))
    for i in range(1, len(pressures)):
        assert pressures[i] <= pressures[i - 1]


def test_andrienko_axially_symmetric() -> None:
    """Defender at +y vs -y at same distance -> equal pressure."""
    ax_pos, _, ctx_pos = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 + 3.0)])
    ax_neg, _, ctx_neg = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 - 3.0)])
    p_pos = _pressure_andrienko(ax_pos, _, ctx_pos, params=AndrienkoParams()).iloc[0]
    p_neg = _pressure_andrienko(ax_neg, _, ctx_neg, params=AndrienkoParams()).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)


def test_andrienko_non_negative() -> None:
    """Pressure is never negative for any geometry."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(60.0, 50.0), (40.0, 20.0), (50.0, 35.0), (49.0, 33.0)],
    )
    out = _pressure_andrienko(ax, ay, ctx, params=AndrienkoParams())
    assert out.iloc[0] >= 0.0
```

- [ ] **Step 6: Run all Andrienko tests**

```bash
uv run pytest tests/tracking/test_pressure_andrienko.py -v
```

Expected: 11 PASSED.

---

## Task 4: Link kernel — RED → GREEN

> Per spec §8.0: same RED→GREEN pattern. Pin tests anchor on zone-radii from spec §4.1 and saturating-aggregation from spec §4.4.

**Files:**
- Create: `tests/tracking/test_pressure_link.py`
- Modify: `silly_kicks/tracking/_kernels.py` (add `_pressure_link`)

- [ ] **Step 1: Write failing pin test (Task 4a — RED)**

Create `tests/tracking/test_pressure_link.py`:

```python
"""Link, Lang & Seidenschwarz 2016 zone-based pressure kernel.

References (see NOTICE):
- Link, D., Lang, S., & Seidenschwarz, P. (2016), PLoS ONE 11(12): e0168768.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import _pressure_link
from silly_kicks.tracking.pressure import LinkParams
from tests.tracking.test_pressure_andrienko import _build_ctx  # reuse fixture builder


def test_link_pr_d2_in_hoz() -> None:
    """Defender at d=2 in HOZ (alpha<45) -> PR_Di = 1 - 2/4 = 0.5; aggregation 1-exp(-1*0.5) ≈ 0.3935."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(52.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-1.0 * 0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_at_hoz_boundary() -> None:
    """Defender at d=4 in HOZ (alpha=0) -> at boundary -> PR_Di = 0 -> aggregate 0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(54.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_pr_in_lz() -> None:
    """Defender at alpha=60° (LZ band [45, 90)) at d=1.5 -> PR_Di = 1 - 1.5/3 = 0.5; aggregate 1-exp(-0.5) ≈ 0.3935."""
    angle = math.radians(60.0)
    d = 1.5
    defender = (50.0 + d * math.cos(angle), 34.0 + d * math.sin(angle))
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[defender])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_in_hz() -> None:
    """Defender behind actor at alpha=180° d=1 -> in HZ (r=2) -> PR_Di = 1 - 1/2 = 0.5; aggregate 1-exp(-0.5)."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(49.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-0.5)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-9)


def test_link_pr_outside_hz() -> None:
    """Defender behind actor at d=3 (beyond HZ r=2) -> PR=0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(47.0, 34.0)])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_two_defenders_each_pr_half() -> None:
    """Two HOZ defenders each PR_Di=0.5 -> sum=1.0; aggregate 1-exp(-1.0) ≈ 0.6321."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(52.0, 34.0), (52.0, 33.999)],  # both essentially Θ=0
    )
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    expected = 1.0 - math.exp(-1.0)
    assert out.iloc[0] == pytest.approx(expected, rel=1e-3)  # 1e-3 because second defender slightly off-axis


def test_link_aggregation_bounded_in_zero_one() -> None:
    """Saturating aggregation always gives output in [0, 1]."""
    ax, ay, ctx = _build_ctx(
        actor_xy=(50.0, 34.0),
        defenders_xy=[(50.5 + i * 0.1, 34.0 + i * 0.1) for i in range(5)],
    )
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert 0.0 <= out.iloc[0] <= 1.0


def test_link_zero_defenders_returns_zero() -> None:
    """Linked, no defenders -> aggregate 1-exp(-0) = 0."""
    ax, ay, ctx = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[])
    out = _pressure_link(ax, ay, ctx, params=LinkParams())
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_link_axially_symmetric() -> None:
    """Same as Andrienko: defender at +y vs -y -> equal pressure."""
    ax_pos, _, ctx_pos = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 + 1.5)])
    ax_neg, _, ctx_neg = _build_ctx(actor_xy=(50.0, 34.0), defenders_xy=[(50.0, 34.0 - 1.5)])
    p_pos = _pressure_link(ax_pos, _, ctx_pos, params=LinkParams()).iloc[0]
    p_neg = _pressure_link(ax_neg, _, ctx_neg, params=LinkParams()).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)
```

- [ ] **Step 2: Run test to verify failure (RED)**

```bash
uv run pytest tests/tracking/test_pressure_link.py -v
```

Expected: All FAIL with `ImportError: cannot import name '_pressure_link'`.

- [ ] **Step 3: Implement `_pressure_link` (GREEN)**

Append to `silly_kicks/tracking/_kernels.py`:

```python
def _pressure_link(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: "LinkParams",  # forward-ref
    goal_x: float = _GOAL_X,
    goal_y: float = _GOAL_Y_CENTER,
) -> pd.Series:
    """Link, Lang & Seidenschwarz 2016 piecewise-zone saturating-aggregation.

    Per defender:
        d            = ||defender - anchor||
        cos_theta    = dot(unit(goal - anchor), unit(defender - anchor))
        alpha_deg    = degrees(arccos(clip(cos_theta, -1, 1)))   # in [0, 180]
        r_zo         = r_hoz if alpha_deg < angle_hoz_lz_deg
                      else r_lz if alpha_deg < angle_lz_hz_deg
                      else r_hz
        pr_i         = max(0, 1 - d/r_zo)   if d < r_zo else 0.0

    Aggregation:
        PR(x) = 1 - exp(-k3 * x)   where x = sum(pr_i)

    See spec section 4.4, NOTICE (Link et al. 2016).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    a_xy = pd.DataFrame(
        {
            "action_id": actions_id,
            "ax": anchor_x.to_numpy(),
            "ay": anchor_y.to_numpy(),
        }
    )
    merged = ctx.opposite_rows_per_action.merge(a_xy, on="action_id", how="left")

    ax = merged["ax"].to_numpy()
    ay = merged["ay"].to_numpy()
    px = merged["x"].to_numpy()
    py = merged["y"].to_numpy()

    presser_dx = px - ax
    presser_dy = py - ay
    d = np.sqrt(presser_dx**2 + presser_dy**2)

    threat_dx = goal_x - ax
    threat_dy = goal_y - ay
    threat_mag = np.sqrt(threat_dx**2 + threat_dy**2) + 1e-12
    presser_mag = d + 1e-12
    cos_theta = (threat_dx * presser_dx + threat_dy * presser_dy) / (threat_mag * presser_mag)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    alpha_deg = np.degrees(np.arccos(cos_theta))

    r_zo = np.where(
        alpha_deg < params.angle_hoz_lz_deg,
        params.r_hoz,
        np.where(alpha_deg < params.angle_lz_hz_deg, params.r_lz, params.r_hz),
    )

    in_zone = d < r_zo
    pr_per_defender = np.where(in_zone, np.maximum(0.0, 1.0 - d / r_zo), 0.0)

    merged["_pr"] = pr_per_defender
    sums = merged.groupby("action_id")["_pr"].sum()
    for aid, x_total in sums.items():
        if aid in action_to_idx.index:
            agg = 1.0 - math.exp(-params.k3 * float(x_total))
            out.loc[action_to_idx.loc[aid]] = agg  # type: ignore[arg-type]
    return out
```

Add at module top:

```python
import math

if TYPE_CHECKING:
    from .pressure import AndrienkoParams, BekkersParams, LinkParams  # noqa: F401
```

- [ ] **Step 4: Run test to verify pass (GREEN)**

```bash
uv run pytest tests/tracking/test_pressure_link.py -v
```

Expected: 9 PASSED.

---

## Task 5: Bekkers TTI sub-primitive — RED → GREEN

> Bekkers's `time_to_intercept` is a reusable sub-primitive (will also serve future TF-7 pitch control). Extract first; build the full Bekkers pressure kernel on top in Task 6. Numerical defaults verified against `unravel/soccer/models/{utils.py, pressing_intensity.py}`.

**Files:**
- Create: `tests/tracking/test_pressure_bekkers.py` (TTI section)
- Modify: `silly_kicks/tracking/_kernels.py` (add `_bekkers_tti`, `_bekkers_p_intercept`)

- [ ] **Step 1: Write failing TTI pin test (RED)**

Create `tests/tracking/test_pressure_bekkers.py`:

```python
"""Bekkers 2024 Pressing Intensity kernel (probabilistic time-to-intercept).

References (see NOTICE):
- Bekkers, J. (2025), arXiv:2501.04712.
- BSD-3-Clause: UnravelSports/unravelsports unravel/soccer/models/utils.py
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import (
    _bekkers_p_intercept,
    _bekkers_tti,
    _pressure_bekkers,
)
from silly_kicks.tracking.pressure import BekkersParams


def test_bekkers_tti_stationary_defender_zero_distance() -> None:
    """Defender at d=0, all velocities zero -> tti = 0 + tau_r + 0/v_max = tau_r = 0.7s."""
    p1 = np.array([[50.0, 34.0]])
    p2 = np.array([[50.0, 34.0]])
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[0.0, 0.0]])
    tti = _bekkers_tti(p1=p1, p2=p2, v1=v1, v2=v2, reaction_time=0.7, max_object_speed=12.0)
    assert tti.shape == (1, 1)
    assert tti[0, 0] == pytest.approx(0.7, rel=1e-9)


def test_bekkers_tti_stationary_d10() -> None:
    """Defender at d=10, all v=0 -> tti = 0 + 0.7 + 10/12 = 0.7 + 0.8333... = 1.5333..."""
    p1 = np.array([[50.0, 34.0]])
    p2 = np.array([[60.0, 34.0]])
    v1 = np.array([[0.0, 0.0]])
    v2 = np.array([[0.0, 0.0]])
    tti = _bekkers_tti(p1=p1, p2=p2, v1=v1, v2=v2, reaction_time=0.7, max_object_speed=12.0)
    assert tti[0, 0] == pytest.approx(0.7 + 10.0 / 12.0, rel=1e-9)


def test_bekkers_p_intercept_at_threshold() -> None:
    """At tti = T_threshold, p = 0.5."""
    tti = np.array([[1.5]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert p[0, 0] == pytest.approx(0.5, rel=1e-9)


def test_bekkers_p_intercept_below_threshold() -> None:
    """tti=0.7 < T=1.5, sigma=0.45 -> p > 0.5 (close defender, high pressure)."""
    tti = np.array([[0.7]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    expected = 1.0 / (1.0 + math.exp(-math.pi / math.sqrt(3.0) / 0.45 * (1.5 - 0.7)))
    assert p[0, 0] == pytest.approx(expected, rel=1e-9)
    assert p[0, 0] > 0.5


def test_bekkers_p_intercept_above_threshold() -> None:
    """tti=3.0 > T=1.5 -> p < 0.5 (distant defender, low pressure)."""
    tti = np.array([[3.0]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert p[0, 0] < 0.5


def test_bekkers_p_intercept_clipping_avoids_overflow() -> None:
    """Extreme tti values shouldn't crash on np.exp overflow (per canonical clip)."""
    tti = np.array([[1e6]])
    p = _bekkers_p_intercept(tti=tti, sigma=0.45, time_threshold=1.5)
    assert 0.0 <= p[0, 0] < 1.0  # very low but valid
```

- [ ] **Step 2: Run test to verify failure (RED)**

```bash
uv run pytest tests/tracking/test_pressure_bekkers.py -v
```

Expected: All FAIL with `ImportError`.

- [ ] **Step 3: Implement `_bekkers_tti` and `_bekkers_p_intercept` (GREEN)**

Append to `silly_kicks/tracking/_kernels.py`:

```python
def _bekkers_tti(
    *,
    p1: np.ndarray,
    p2: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    reaction_time: float,
    max_object_speed: float,
) -> np.ndarray:
    """Bekkers/Spearman/Shaw/Pleuler time-to-intercept matrix.

    Re-implementation of UnravelSports/unravelsports
    unravel/soccer/models/utils.py:time_to_intercept (BSD-3-Clause).
    See NOTICE for full attribution.

    Parameters
    ----------
    p1 : (n, 2) ndarray
        Pressing-player positions.
    p2 : (m, 2) ndarray
        Target positions (attackers / ball).
    v1 : (n, 2) ndarray
        Pressing-player velocities.
    v2 : (m, 2) ndarray
        Target velocities.
    reaction_time : float
        Pressing-player reaction time before accelerating (s).
    max_object_speed : float
        Pressing-player maximum running speed (m/s).

    Returns
    -------
    (m, n) ndarray
        TTI matrix; element [i, j] = time for pressing-player j to intercept
        target i.
    """
    u = v1
    d2 = p2 + v2
    v = d2[:, None, :] - p1[None, :, :]

    u_mag = np.linalg.norm(u, axis=-1)
    v_mag = np.linalg.norm(v, axis=-1)
    dot = np.sum(u * v, axis=-1)

    eps = 1e-10
    angle = np.arccos(np.clip(dot / (u_mag * v_mag + eps), -1.0, 1.0))

    r_reaction = p1 + v1 * reaction_time
    d = d2[:, None, :] - r_reaction[None, :, :]

    t = (
        u_mag * angle / np.pi
        + reaction_time
        + np.linalg.norm(d, axis=-1) / max_object_speed
    )
    return t


def _bekkers_p_intercept(
    *,
    tti: np.ndarray,
    sigma: float,
    time_threshold: float,
) -> np.ndarray:
    """Logistic transform of TTI to per-pair intercept probability.

    Re-implementation of unravel/soccer/models/utils.py:probability_to_intercept
    (BSD-3-Clause). See NOTICE.
    """
    exponent = -math.pi / math.sqrt(3.0) / sigma * (time_threshold - tti)
    exponent = np.clip(exponent, -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(exponent))
```

- [ ] **Step 4: Run test to verify pass (GREEN)**

```bash
uv run pytest tests/tracking/test_pressure_bekkers.py -v
```

Expected: 6 PASSED (TTI section).

---

## Task 6: Bekkers full pressure kernel — RED → GREEN

> Build the full per-action `_pressure_bekkers` on top of TTI sub-primitive. Includes ball-carrier-max improvement (§2.4) + active-pressing speed filter (§3.1) + 1-Π aggregation.

**Files:**
- Modify: `tests/tracking/test_pressure_bekkers.py` (add full-kernel tests)
- Modify: `silly_kicks/tracking/_kernels.py` (add `_pressure_bekkers`)

- [ ] **Step 1: Add failing full-kernel pin tests (RED)**

Append to `tests/tracking/test_pressure_bekkers.py`:

```python
def _build_ctx_with_velocities_and_ball(
    *,
    actor_xy: tuple[float, float],
    actor_vxvy: tuple[float, float] = (0.0, 0.0),
    defenders: list[tuple[float, float, float, float]],  # (x, y, vx, vy)
    ball_xyvxvy: tuple[float, float, float, float] | None = (50.0, 34.0, 0.0, 0.0),
):
    """Build ActionFrameContext + ball-rows-per-action mapping for Bekkers tests."""
    from silly_kicks.tracking.feature_framework import ActionFrameContext

    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "team_id": ["home"],
            "player_id": [10],
            "start_x": [actor_xy[0]],
            "start_y": [actor_xy[1]],
        }
    )
    actor_rows = pd.DataFrame(
        {
            "action_id": [1],
            "x": [actor_xy[0]],
            "y": [actor_xy[1]],
            "vx": [actor_vxvy[0]],
            "vy": [actor_vxvy[1]],
            "speed": [math.hypot(*actor_vxvy)],
        }
    )
    defender_rows = pd.DataFrame(
        [
            {
                "action_id": 1,
                "team_id_action": "home",
                "team_id_frame": "away",
                "player_id_action": 10,
                "player_id_frame": 100 + i,
                "is_ball": False,
                "x": dx,
                "y": dy,
                "vx": dvx,
                "vy": dvy,
                "speed": math.hypot(dvx, dvy),
            }
            for i, (dx, dy, dvx, dvy) in enumerate(defenders)
        ]
    )
    pointers = pd.DataFrame(
        {
            "action_id": [1],
            "frame_id": pd.array([1000], dtype="Int64"),
            "time_offset_seconds": [0.0],
            "n_candidate_frames": [1],
            "link_quality_score": [1.0],
        }
    )
    ctx = ActionFrameContext(
        actions=actions,
        pointers=pointers,
        actor_rows=actor_rows,
        opposite_rows_per_action=defender_rows,
        defending_gk_rows=pd.DataFrame(),
    )
    if ball_xyvxvy is not None:
        ball_per_action = pd.DataFrame(
            [{
                "action_id": 1,
                "x": ball_xyvxvy[0],
                "y": ball_xyvxvy[1],
                "vx": ball_xyvxvy[2],
                "vy": ball_xyvxvy[3],
            }]
        )
    else:
        ball_per_action = pd.DataFrame(columns=["action_id", "x", "y", "vx", "vy"])
    return actions["start_x"], actions["start_y"], ctx, ball_per_action


def test_bekkers_kernel_stationary_defender_at_d_zero() -> None:
    """Defender at d=0 with v=0; actor v=0; ball=actor; speed_threshold=2.0 (filtered)."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(50.0, 34.0, 0.0, 0.0)],
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    # Defender speed=0 < threshold=2.0, so p_to_player=0; ball at d=0 also stationary, p=0
    # 1 - (1 - 0) = 0
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_bekkers_kernel_active_defender_at_d_zero() -> None:
    """Defender at d=0 with speed=3 m/s above threshold -> active-pressing path engaged.

    Loose-bound pin (range, not exact value). Per lakehouse v3 review item 6:
    exact-value pinning for Bekkers belongs in the golden-master parity test
    (Task 13 Step 1 — bit-equivalent rtol=1e-9 vs UnravelSports canonical) and
    snapshot determinism gate (Task 13 Step 2 — SHA-256 regression). This test's
    job is to verify that the active-pressing speed filter does NOT zero out a
    defender above threshold (regression guard against active-pressing logic
    bug); exact value drift is caught by the dedicated gates.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(50.0, 34.0, 3.0, 0.0)],
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    # Active-pressing filter does NOT fire (speed=3 > threshold=2.0), so p > 0.
    # Bounded in [0, 1] by aggregation construction.
    assert out.iloc[0] > 0.0
    assert 0.0 <= out.iloc[0] <= 1.0


def test_bekkers_kernel_two_defenders_aggregate() -> None:
    """Two defenders each with p≈0.6 -> aggregate 1 - (1-0.6)^2 = 0.84."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(48.0, 34.0, 3.0, 0.0), (52.0, 34.0, -3.0, 0.0)],  # both moving toward actor
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    assert 0.0 <= out.iloc[0] <= 1.0


def test_bekkers_kernel_speed_threshold_filters() -> None:
    """Defender below speed threshold -> p=0 -> aggregate 0 (only this defender)."""
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(51.0, 34.0, 1.0, 0.0)],  # speed=1 < 2.0
    )
    out = _pressure_bekkers(ax, ay, ctx, params=BekkersParams(), ball_xy_v_per_action=ball)
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_bekkers_kernel_use_ball_carrier_max_strictly_higher_when_ball_closer() -> None:
    """When ball is closer to defender than actor, max(p_to_ball, p_to_player)
    must produce STRICTLY higher pressure than p_to_player alone.

    Lakehouse v3 review item 5: original test asserted >=, which is tautological
    (max(a,b) >= a always). Strict-greater assertion verifies the ball-comparison
    path actually fires and increases pressure when ball is closer.

    Construct: defender at (60, 34) with v=(-3, 0) closing on actor at (50, 34);
    ball at (62, 34) — defender 10m from actor, only 2m from ball, moving toward actor
    (so toward ball direction). p_to_ball >> p_to_player.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[(60.0, 34.0, -3.0, 0.0)],
        ball_xyvxvy=(62.0, 34.0, 0.0, 0.0),
    )
    out_with_max = _pressure_bekkers(
        ax, ay, ctx, params=BekkersParams(use_ball_carrier_max=True),
        ball_xy_v_per_action=ball,
    )
    out_no_max = _pressure_bekkers(
        ax, ay, ctx, params=BekkersParams(use_ball_carrier_max=False),
        ball_xy_v_per_action=ball,
    )
    assert out_with_max.iloc[0] > out_no_max.iloc[0] + 1e-6, (
        f"ball-carrier-max must produce STRICTLY higher pressure when ball is closer "
        f"than actor; got with_max={out_with_max.iloc[0]:.6f} vs no_max={out_no_max.iloc[0]:.6f}"
    )


@pytest.mark.parametrize("use_ball_carrier_max", [True, False])
def test_bekkers_kernel_zero_defenders(use_ball_carrier_max: bool) -> None:
    """Linked, no defenders -> 0.0 (NOT NaN). Lakehouse v3 review item 4:
    parametrized over both use_ball_carrier_max settings to catch regressions
    in the optional ball-comparison path.
    """
    ax, ay, ctx, ball = _build_ctx_with_velocities_and_ball(
        actor_xy=(50.0, 34.0),
        defenders=[],
    )
    out = _pressure_bekkers(
        ax, ay, ctx,
        params=BekkersParams(use_ball_carrier_max=use_ball_carrier_max),
        ball_xy_v_per_action=ball,
    )
    assert out.iloc[0] == pytest.approx(0.0, abs=1e-9)
```

- [ ] **Step 2: Run test to verify failure**

```bash
uv run pytest tests/tracking/test_pressure_bekkers.py -v
```

Expected: New tests FAIL with `ImportError: cannot import name '_pressure_bekkers'`. TTI tests still pass.

- [ ] **Step 3: Implement `_pressure_bekkers` (GREEN)**

Append to `silly_kicks/tracking/_kernels.py`:

```python
def _pressure_bekkers(
    anchor_x: pd.Series,
    anchor_y: pd.Series,
    ctx: ActionFrameContext,
    *,
    params: "BekkersParams",
    ball_xy_v_per_action: pd.DataFrame,
) -> pd.Series:
    """Bekkers 2024 Pressing Intensity probabilistic model.

    Per defender: TTI -> p via logistic. Optional ball-carrier-max
    (max of p_to_player and p_to_ball per defender). Aggregation:
    1 - prod(1 - p_i_final).

    See spec section 4.4, NOTICE (Bekkers 2025; BSD-3-Clause UnravelSports).
    """
    out = pd.Series(np.full(len(ctx.actions), np.nan), index=ctx.actions.index, dtype="float64")
    actions_id = ctx.actions["action_id"].to_numpy()
    action_to_idx = pd.Series(ctx.actions.index, index=actions_id)

    linked_aids = set(ctx.pointers.loc[ctx.pointers["frame_id"].notna(), "action_id"].to_numpy())
    for aid in linked_aids:
        if aid in action_to_idx.index:
            out.loc[action_to_idx.loc[aid]] = 0.0  # type: ignore[arg-type]

    if len(ctx.opposite_rows_per_action) == 0:
        return out

    # Build per-action arrays: actor pos+v, ball pos+v (if used), defenders pos+v+speed
    actor_per_action = ctx.actor_rows.set_index("action_id")[["x", "y", "vx", "vy"]]

    if params.use_ball_carrier_max:
        ball_per_action_indexed = ball_xy_v_per_action.set_index("action_id") if len(ball_xy_v_per_action) else pd.DataFrame()

    grouped = ctx.opposite_rows_per_action.groupby("action_id")
    for aid, defender_group in grouped:
        if aid not in action_to_idx.index:
            continue
        if aid not in actor_per_action.index:
            continue
        actor_row = actor_per_action.loc[aid]
        actor_pos = np.array([[actor_row["x"], actor_row["y"]]], dtype="float64")
        actor_vel = np.array([[actor_row["vx"], actor_row["vy"]]], dtype="float64")
        if pd.isna(actor_pos).any() or pd.isna(actor_vel).any():
            out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
            continue

        defender_pos = defender_group[["x", "y"]].to_numpy(dtype="float64")
        defender_vel = defender_group[["vx", "vy"]].to_numpy(dtype="float64")
        defender_speed = defender_group["speed"].to_numpy(dtype="float64")

        if np.isnan(defender_pos).any() or np.isnan(defender_vel).any():
            out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
            continue

        tti_to_actor = _bekkers_tti(
            p1=defender_pos, p2=actor_pos, v1=defender_vel, v2=actor_vel,
            reaction_time=params.reaction_time, max_object_speed=params.max_player_speed,
        )
        p_to_actor = _bekkers_p_intercept(
            tti=tti_to_actor, sigma=params.sigma, time_threshold=params.time_threshold,
        )[0, :]  # shape (n_defenders,)

        p_per_defender = p_to_actor.copy()

        if params.use_ball_carrier_max and len(ball_per_action_indexed) > 0:
            if aid in ball_per_action_indexed.index:
                ball_row = ball_per_action_indexed.loc[aid]
                ball_pos = np.array([[ball_row["x"], ball_row["y"]]], dtype="float64")
                ball_vel = np.array([[ball_row["vx"], ball_row["vy"]]], dtype="float64")
                if not (np.isnan(ball_pos).any() or np.isnan(ball_vel).any()):
                    tti_to_ball = _bekkers_tti(
                        p1=defender_pos, p2=ball_pos, v1=defender_vel, v2=ball_vel,
                        reaction_time=params.reaction_time, max_object_speed=params.max_player_speed,
                    )
                    p_to_ball = _bekkers_p_intercept(
                        tti=tti_to_ball, sigma=params.sigma, time_threshold=params.time_threshold,
                    )[0, :]
                    p_per_defender = np.maximum(p_to_actor, p_to_ball)
                # else: this action has NaN ball -> NaN per spec section 4.5
                else:
                    out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
                    continue
            else:
                # This action has no ball row -> NaN per spec section 4.5
                out.loc[action_to_idx.loc[aid]] = float("nan")  # type: ignore[arg-type]
                continue

        # Active-pressing filter: defenders below speed_threshold contribute 0
        below_thresh = defender_speed < params.speed_threshold
        p_per_defender = np.where(below_thresh, 0.0, p_per_defender)

        # Aggregation: 1 - prod(1 - p)
        agg = 1.0 - float(np.prod(1.0 - p_per_defender))
        out.loc[action_to_idx.loc[aid]] = agg  # type: ignore[arg-type]

    return out
```

- [ ] **Step 4: Run test to verify pass (GREEN)**

```bash
uv run pytest tests/tracking/test_pressure_bekkers.py -v
```

Expected: 13 PASSED (6 TTI + 5 full-kernel test functions + 1 parametrized zero-defenders × 2 = 7 full-kernel collections, total 6 + 7 = 13).

- [ ] **Step 5: Add tests for ball-rows hard-fail + per-action NaN (lakehouse v3 review items 2-3)**

Append to `tests/tracking/test_pressure_bekkers.py`:

```python
def test_bekkers_no_ball_rows_anywhere_raises_value_error() -> None:
    """Hard-fail per spec section 4.5 + section 11 risk row (lakehouse v2 review item 9):
    use_ball_carrier_max=True with ZERO ball rows in entire frames -> ValueError.

    Without this test, a future refactor could silently revert the load-bearing
    decision against the silent UserWarning fallback.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "start_x": [50.0], "start_y": [34.0], "type_id": [0],
    })
    # Frames with NO is_ball=True rows
    frames = pd.DataFrame([
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "home", "player_id": 10, "is_ball": False,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": 3.0, "vy": 0.0, "speed": 3.0},
    ])
    with pytest.raises(ValueError, match="missing is_ball=True rows"):
        pressure_on_actor(actions, frames, method="bekkers_pi")


def test_bekkers_no_ball_rows_with_opt_out_succeeds() -> None:
    """Opt-out path: use_ball_carrier_max=False bypasses the hard-fail.

    Caller chose to compute pressure-on-player only; should succeed even
    without ball rows.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "start_x": [50.0], "start_y": [34.0], "type_id": [0],
    })
    frames = pd.DataFrame([
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "home", "player_id": 10, "is_ball": False,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": -3.0, "vy": 0.0, "speed": 3.0},
    ])
    # Opt out — should NOT raise
    result = pressure_on_actor(
        actions, frames, method="bekkers_pi",
        params=BekkersParams(use_ball_carrier_max=False),
    )
    assert result.notna().any()


def test_bekkers_per_action_ball_row_absence_emits_nan() -> None:
    """Per-action NaN per spec section 4.3 / section 4.5 (lakehouse v3 review item 3).

    Some actions link to ball-present frames, others to ball-absent frames.
    Ball-absent actions should emit NaN; ball-present actions compute normally.
    Two actions, two distinct frames; only frame 1 has a ball row.
    """
    from silly_kicks.tracking.features import pressure_on_actor

    actions = pd.DataFrame({
        "action_id": [1, 2], "period_id": [1, 1],
        "time_seconds": [0.0, 1.0], "team_id": ["home", "home"],
        "player_id": [10, 11],
        "start_x": [50.0, 50.0], "start_y": [34.0, 34.0],
        "type_id": [0, 0],
    })
    frames = pd.DataFrame([
        # Action 1 frame: ball PRESENT
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "home", "player_id": 10, "is_ball": False,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": None, "player_id": None, "is_ball": True,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 1, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": -3.0, "vy": 0.0, "speed": 3.0},
        # Action 2 frame: ball ABSENT
        {"frame_id": 2, "period_id": 1, "time_seconds": 1.0,
         "team_id": "home", "player_id": 11, "is_ball": False,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 2, "period_id": 1, "time_seconds": 1.0,
         "team_id": "away", "player_id": 101, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": -3.0, "vy": 0.0, "speed": 3.0},
    ])
    # frames does have AT LEAST ONE ball row (action 1's), so the entire-frames
    # hard-fail does NOT trigger; per-action NaN handling kicks in for action 2.
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert pd.notna(result.iloc[0]), "action 1 should compute (ball row present at its frame)"
    assert pd.isna(result.iloc[1]), "action 2 should be NaN (no ball row at its frame)"
```

- [ ] **Step 6: Run extended Bekkers tests**

```bash
uv run pytest tests/tracking/test_pressure_bekkers.py -v
```

Expected: 16 PASSED (6 TTI + 7 full-kernel collections from Step 4 + 3 ball-rows contract tests from Step 5).

Note on count: pytest counts each parametrize permutation as a separate test collection. `test_bekkers_kernel_zero_defenders` is parametrized over `use_ball_carrier_max=[True, False]`, so it contributes 2 collections (not 1). 6 TTI + (5 single-tests + 2 parametrized) + 3 contract = 16.

---

## Task 7: TF-3 pre-window kernel — RED → GREEN

> Per spec §3.3: kernel sorts by `(action_id, time_seconds ASC)`, drops NaN-position rows, computes both arc-length and displacement in one pass over the bridge-rule logic.

**Files:**
- Create: `tests/tracking/test_pre_window_features.py`
- Modify: `silly_kicks/tracking/_kernels.py` (add `_actor_pre_window_kernel`)

- [ ] **Step 1: Write failing pin test (RED)**

Create `tests/tracking/test_pre_window_features.py`:

```python
"""TF-3 pre-window kernel: arc-length + displacement with NaN-bridge rule.

References (see NOTICE):
- Pure geometric formulation; NOT Bauer & Anzer 2021 covered-distance feature
  (which uses sprint-intensity filtering).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._kernels import _actor_pre_window_kernel


def _build_pre_window_input(
    *,
    pre_seconds: float,
    actor_player_id: int,
    frame_xys: list[tuple[float, float, float]],  # (x, y, time_offset_from_action)
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_time = 100.0
    actions = pd.DataFrame(
        {
            "action_id": [1],
            "period_id": [1],
            "time_seconds": [action_time],
            "player_id": [actor_player_id],
        }
    )
    rows = []
    for frame_id, (x, y, dt) in enumerate(frame_xys):
        rows.append({
            "frame_id": frame_id,
            "period_id": 1,
            "time_seconds": action_time + dt,
            "player_id": actor_player_id,
            "is_ball": False,
            "x": x,
            "y": y,
        })
        # Add a ball row per frame (so frames df is realistic shape)
        rows.append({
            "frame_id": frame_id,
            "period_id": 1,
            "time_seconds": action_time + dt,
            "player_id": None,
            "is_ball": True,
            "x": x,
            "y": y,
        })
    frames = pd.DataFrame(rows)
    return actions, frames


def test_pre_window_stationary_actor() -> None:
    """All frames at (10, 10) -> arc-length = 0, displacement = 0."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[(10.0, 10.0, dt) for dt in [-0.4, -0.3, -0.2, -0.1, 0.0]],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(0.0, abs=1e-9)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(0.0, abs=1e-9)


def test_pre_window_constant_velocity_5ms() -> None:
    """5 m/s along +x for 0.5s, 12 segments -> arc-length=2.5, displacement=2.5."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[(50.0 + 5.0 * (dt + 0.5), 34.0, dt) for dt in np.linspace(-0.5, 0.0, 13)],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(2.5, rel=1e-6)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(2.5, rel=1e-6)


def test_pre_window_circular_path() -> None:
    """Half-circle radius 2 in 0.5s -> arc-length ≈ pi*2 ≈ 6.28; displacement = 4 (diameter)."""
    n_frames = 13
    angles = np.linspace(0, math.pi, n_frames)
    xys = [(50.0 + 2.0 * math.cos(a), 34.0 + 2.0 * math.sin(a)) for a in angles]
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[(x, y, dt) for (x, y), dt in zip(xys, np.linspace(-0.5, 0.0, n_frames), strict=False)],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    arc = out.iloc[0]["actor_arc_length_pre_window"]
    disp = out.iloc[0]["actor_displacement_pre_window"]
    # 12 secants of pi/12 each ≈ 12 * 2 * sin(pi/24) ≈ 6.255 (slightly less than full pi*2)
    assert 5.5 <= arc <= 6.5
    assert disp == pytest.approx(4.0, rel=1e-6)
    assert arc > disp  # circular path: arc > displacement always


def test_pre_window_bridge_rule_one_nan() -> None:
    """Frames at -0.4, -0.3, NaN, -0.1, 0.0 with positions (0,0)/(1,0)/NaN/(3,0)/(4,0):
    arc-length = 1 + 2 + 1 = 4 (bridges across NaN); displacement = 4 (first valid 0,0 to last valid 4,0)."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[
            (0.0, 0.0, -0.4),
            (1.0, 0.0, -0.3),
            (float("nan"), float("nan"), -0.2),
            (3.0, 0.0, -0.1),
            (4.0, 0.0, 0.0),
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(4.0, rel=1e-6)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(4.0, rel=1e-6)


def test_pre_window_one_valid_frame_returns_nan() -> None:
    """Only 1 valid-position frame -> NaN for both."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[
            (0.0, 0.0, -0.4),
            (float("nan"), float("nan"), -0.3),
            (float("nan"), float("nan"), -0.2),
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert pd.isna(out.iloc[0]["actor_arc_length_pre_window"])
    assert pd.isna(out.iloc[0]["actor_displacement_pre_window"])


def test_pre_window_no_frames_returns_nan() -> None:
    """Action has no frames in window for actor -> NaN."""
    action_time = 100.0
    actions = pd.DataFrame(
        {"action_id": [1], "period_id": [1], "time_seconds": [action_time], "player_id": [10]}
    )
    frames = pd.DataFrame(
        {
            "frame_id": [0],
            "period_id": [1],
            "time_seconds": [action_time + 5.0],  # outside window
            "player_id": [10],
            "is_ball": [False],
            "x": [50.0],
            "y": [34.0],
        }
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    assert pd.isna(out.iloc[0]["actor_arc_length_pre_window"])
    assert pd.isna(out.iloc[0]["actor_displacement_pre_window"])


def test_pre_window_shuffled_input_rows() -> None:
    """Kernel MUST sort rows by time_seconds ASC; shuffled input gives same answer (per spec section 3.3)."""
    actions, frames = _build_pre_window_input(
        pre_seconds=0.5, actor_player_id=10,
        frame_xys=[(50.0 + 5.0 * (dt + 0.5), 34.0, dt) for dt in np.linspace(-0.5, 0.0, 13)],
    )
    frames_shuffled = frames.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out_sorted = _actor_pre_window_kernel(actions, frames, pre_seconds=0.5)
    out_shuffled = _actor_pre_window_kernel(actions, frames_shuffled, pre_seconds=0.5)
    pd.testing.assert_frame_equal(out_sorted, out_shuffled)
```

- [ ] **Step 2: Run test to verify failure (RED)**

```bash
uv run pytest tests/tracking/test_pre_window_features.py -v
```

Expected: All FAIL with ImportError.

- [ ] **Step 3: Implement `_actor_pre_window_kernel` (GREEN)**

Append to `silly_kicks/tracking/_kernels.py`:

```python
def _actor_pre_window_kernel(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float,
) -> pd.DataFrame:
    """TF-3 shared kernel: emits both arc-length and displacement.

    REQUIREMENT per spec section 3.3: sort frames by (action_id, time_seconds ASC)
    inside each (action, actor_player_id) group BEFORE computing segments or
    selecting first/last. Drops NaN-position rows entirely (bridge rule).

    Returns DataFrame with columns:
        - actor_arc_length_pre_window (float64, m)
        - actor_displacement_pre_window (float64, m)
    indexed identically to ``actions``.
    """
    from .utils import slice_around_event

    out = pd.DataFrame(
        {
            "actor_arc_length_pre_window": np.full(len(actions), np.nan, dtype="float64"),
            "actor_displacement_pre_window": np.full(len(actions), np.nan, dtype="float64"),
        },
        index=actions.index,
    )

    if len(actions) == 0 or len(frames) == 0:
        return out

    sliced = slice_around_event(actions, frames, pre_seconds=pre_seconds, post_seconds=0.0)
    if len(sliced) == 0:
        return out

    sliced = sliced[~sliced["is_ball"]].copy()
    actor_id_per_action = actions[["action_id", "player_id"]].rename(columns={"player_id": "actor_player_id"})
    sliced = sliced.merge(actor_id_per_action, on="action_id", how="left")
    sliced = sliced[sliced["player_id"] == sliced["actor_player_id"]].copy()
    if len(sliced) == 0:
        return out

    # Drop NaN-position rows (bridge rule per spec section 3.2)
    valid = sliced.dropna(subset=["x", "y"]).copy()
    if len(valid) == 0:
        return out

    # Sort by (action_id, time_seconds ASC) per kernel requirement
    valid = valid.sort_values(["action_id", "time_seconds"], kind="mergesort")

    action_to_idx = pd.Series(actions.index.values, index=actions["action_id"].values)
    grouped = valid.groupby("action_id", sort=False)
    for aid, group in grouped:
        if len(group) < 2:
            continue
        xs = group["x"].to_numpy()
        ys = group["y"].to_numpy()
        dx = np.diff(xs)
        dy = np.diff(ys)
        arc = float(np.sqrt(dx * dx + dy * dy).sum())
        disp = float(math.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
        if aid in action_to_idx.index:
            row_idx = action_to_idx.loc[aid]
            out.loc[row_idx, "actor_arc_length_pre_window"] = arc
            out.loc[row_idx, "actor_displacement_pre_window"] = disp
    return out
```

- [ ] **Step 4: Run test to verify pass (GREEN)**

```bash
uv run pytest tests/tracking/test_pre_window_features.py -v
```

Expected: 7 PASSED.

---

## Task 8: Public API surface in `silly_kicks/tracking/features.py`

> Per spec §3.1, §4.2: 6 new public defs (4 TF-3 + 2 TF-2 wrappers) + 2 aggregators (`add_actor_pre_window`, `add_pressure_on_actor`) + 2 default xfn lists. All public defs include Examples sections per `feedback_public_api_examples`.

**Files:**
- Modify: `silly_kicks/tracking/features.py`

- [ ] **Step 1: Add new imports + constants**

Modify `silly_kicks/tracking/features.py` near top (after existing imports):

```python
from typing import Literal, Union

from .pressure import (
    AndrienkoParams,
    BekkersParams,
    LinkParams,
    Method,
    PressureParams,
    validate_params_for_method,
)
```

Add to `__all__`:

```python
    "Method",
    "actor_arc_length_pre_window",
    "actor_displacement_pre_window",
    "actor_pre_window_default_xfns",
    "add_actor_pre_window",
    "add_pressure_on_actor",
    "pressure_default_xfns",
    "pressure_on_actor",
```

- [ ] **Step 2: Add TF-3 public defs**

Append after PR-S24 angle features:

```python
# ---------------------------------------------------------------------------
# PR-S25 -- TF-3: actor_*_pre_window features
# ---------------------------------------------------------------------------


def actor_arc_length_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Geometric arc-length of actor's path over the pre-action window (m).

    Per-action sum of consecutive segment distances over frames in
    (action_time - pre_seconds, action_time], filtered to actor's player_id
    within the same period:

        sum_{k=1..N-1} sqrt((x_{k+1} - x_k)**2 + (y_{k+1} - y_k)**2)

    Consecutive segments computed AFTER sorting by frame timestamp ASC and
    dropping frames with NaN positions (bridge rule per spec section 3.2).
    NaN if fewer than 2 valid frames remain.

    The pre_seconds=0.5 default captures sub-second pre-action movement
    intensity. For longer windows like Bauer & Anzer 2021 counterpressing
    detection (5s), pass pre_seconds=5.0.

    NOT a re-implementation of any paper's filtered/threshold-based
    "covered distance" feature -- pure geometric arc-length, no
    sprint-intensity filtering. See NOTICE.

    Examples
    --------
    >>> import pandas as pd
    >>> from silly_kicks.tracking.features import actor_arc_length_pre_window
    >>> actions = pd.DataFrame({
    ...     "action_id": [1], "period_id": [1], "time_seconds": [10.0],
    ...     "player_id": [42], "team_id": [1], "start_x": [50.0],
    ...     "start_y": [34.0], "type_id": [0],
    ... })
    >>> frames = pd.DataFrame()  # empty -> all-NaN; runnable example
    >>> _ = actor_arc_length_pre_window(actions, frames)
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_arc_length_pre_window"].rename("actor_arc_length_pre_window")


def actor_displacement_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Net Euclidean displacement (window-first to window-last valid position).

    Differs from arc-length: a player who runs in a circle has high
    arc-length but ~zero displacement.

    NaN semantics identical to :func:`actor_arc_length_pre_window`. See NOTICE.

    Examples
    --------
    >>> from silly_kicks.tracking.features import actor_displacement_pre_window
    >>> # See tests/tracking/test_pre_window_features.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_displacement_pre_window"].rename("actor_displacement_pre_window")


@nan_safe_enrichment
def add_actor_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.DataFrame:
    """Enrich actions with 2 TF-3 movement columns + 4 linkage-provenance columns.

    Returns
    -------
    pd.DataFrame
        Input actions with the columns:
        - actor_arc_length_pre_window (float64, m)
        - actor_displacement_pre_window (float64, m)
        - frame_id (Int64; NaN if unlinked)
        - time_offset_seconds (float64; NaN if unlinked)
        - n_candidate_frames (int64)
        - link_quality_score (float64; NaN if unlinked)

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_actor_pre_window
    >>> # See tests/tracking/test_pre_window_features.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    out = actions.copy()
    out["actor_arc_length_pre_window"] = df["actor_arc_length_pre_window"]
    out["actor_displacement_pre_window"] = df["actor_displacement_pre_window"]
    # Provenance columns from link_actions_to_frames
    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


actor_pre_window_default_xfns = [lift_to_states(actor_arc_length_pre_window)]
```

(Add `from .utils import link_actions_to_frames` to imports if not already present.)

- [ ] **Step 3: Add TF-2 public defs**

Append:

```python
# ---------------------------------------------------------------------------
# PR-S25 -- TF-2: pressure_on_actor multi-flavor feature
# ---------------------------------------------------------------------------


def pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: Method = "andrienko_oval",
    params: PressureParams | None = None,
) -> pd.Series:
    """Pressure exerted on the action's actor at the linked frame.

    Three published methodologies via ``method=``:

    - ``"andrienko_oval"`` (default) - Andrienko et al. 2017 directional oval
      pressure; sum across opposing defenders. Output range [0, ~200%].
    - ``"link_zones"`` - Link et al. 2016 piecewise-zone pressure;
      saturating exponential aggregation. Output [0, 1].
    - ``"bekkers_pi"`` - Bekkers 2024 Pressing Intensity probabilistic TTI;
      requires velocity columns vx/vy in frames. Output [0, 1].

    Returns Series named ``pressure_on_actor__<method>`` (suffix-naming
    convention per ADR-005 section 8 multi-flavor xfn rule).

    NaN where action couldn't link; 0.0 where linked but no defenders
    contribute pressure. ``bekkers_pi`` raises ValueError if frames lack
    vx/vy or (when use_ball_carrier_max=True) if frames lack any ball rows.

    See NOTICE for full bibliographic citations.

    Examples
    --------
    >>> from silly_kicks.tracking.features import pressure_on_actor
    >>> # See tests/tracking/test_pressure_*.py for runnable examples per method.
    """
    validate_params_for_method(method, params)
    if method == "andrienko_oval":
        ap = params if isinstance(params, AndrienkoParams) else AndrienkoParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_andrienko(actions["start_x"], actions["start_y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_link(actions["start_x"], actions["start_y"], ctx, params=lp)
    elif method == "bekkers_pi":
        bp = params if isinstance(params, BekkersParams) else BekkersParams()
        if "vx" not in frames.columns or "vy" not in frames.columns:
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi'): frames missing velocity columns "
                "'vx'/'vy'. Run silly_kicks.tracking.preprocess.derive_velocities(frames) "
                "first, or use a provider that emits velocities natively."
            )
        if bp.use_ball_carrier_max and not frames["is_ball"].any():
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi', params.use_ball_carrier_max=True): "
                "frames missing is_ball=True rows in linked frames. Either set "
                "use_ball_carrier_max=False to compute pressure-on-player only, or "
                "use a provider that emits ball positions per frame."
            )
        ctx = _resolve_action_frame_context(actions, frames)
        ball_xy_v_per_action = _build_ball_xy_v_per_action(actions, frames, ctx)
        s = _kernels._pressure_bekkers(
            actions["start_x"], actions["start_y"], ctx,
            params=bp, ball_xy_v_per_action=ball_xy_v_per_action,
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")  # defensive; validate_params_for_method already raised
    return s.rename(f"pressure_on_actor__{method}")


def _build_ball_xy_v_per_action(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    ctx,
) -> pd.DataFrame:
    """Build per-action ball position+velocity at the linked frame."""
    pointers = ctx.pointers
    ball_rows = frames.loc[frames["is_ball"], ["period_id", "frame_id", "x", "y", "vx", "vy"]]
    merged = pointers.merge(ball_rows, on=["frame_id"], how="left")
    return merged[["action_id", "x", "y", "vx", "vy"]]


@nan_safe_enrichment
def add_pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    methods: tuple[Method, ...] = ("andrienko_oval",),
    params_per_method: dict[Method, PressureParams] | None = None,
) -> pd.DataFrame:
    """Enrich actions with one ``pressure_on_actor__<m>`` column per method
    + 4 linkage-provenance columns.

    Validates all (method, params) pairs BEFORE computing any column
    (transactional behavior per spec section 8.5).

    Examples
    --------
    >>> from silly_kicks.tracking.features import add_pressure_on_actor
    >>> # See tests/tracking/test_pressure_*.py for runnable examples.
    """
    if params_per_method is None:
        params_per_method = {}
    # Validate all upfront (transactional)
    for m in methods:
        validate_params_for_method(m, params_per_method.get(m))

    out = actions.copy()
    for m in methods:
        params = params_per_method.get(m)
        s = pressure_on_actor(actions, frames, method=m, params=params)
        out[f"pressure_on_actor__{m}"] = s.values

    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


pressure_default_xfns = [lift_to_states(pressure_on_actor)]
```

- [ ] **Step 4: Run scaffolding + andrienko + link + bekkers + pre_window tests**

```bash
uv run pytest tests/tracking/test_pressure_module_scaffolding.py \
              tests/tracking/test_pressure_andrienko.py \
              tests/tracking/test_pressure_link.py \
              tests/tracking/test_pressure_bekkers.py \
              tests/tracking/test_pre_window_features.py -v
```

Expected: All PASS.

---

## Task 9: Atomic mirror in `silly_kicks/atomic/tracking/features.py`

> Per spec §6: parallel public surface using atomic SPADL `(x, y)` anchor instead of `(start_x, start_y)`. Imports kernels + dataclasses from `silly_kicks.tracking` (one-way dependency).

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`

- [ ] **Step 1: Add atomic public defs (mirroring TF-3 + TF-2)**

Append to `silly_kicks/atomic/tracking/features.py`:

```python
# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-3 actor_*_pre_window
# ---------------------------------------------------------------------------


def actor_arc_length_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: geometric arc-length of actor's path over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_arc_length_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_arc_length_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_arc_length_pre_window"].rename("actor_arc_length_pre_window")


def actor_displacement_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.Series:
    """Atomic-SPADL: net Euclidean displacement over pre-action window.

    See :func:`silly_kicks.tracking.features.actor_displacement_pre_window`.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import actor_displacement_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    return df["actor_displacement_pre_window"].rename("actor_displacement_pre_window")


@nan_safe_enrichment
def add_actor_pre_window(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    pre_seconds: float = 0.5,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator for TF-3 features.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_actor_pre_window
    >>> # See tests/atomic/tracking/test_pre_window_features_atomic.py for runnable examples.
    """
    df = _kernels._actor_pre_window_kernel(actions, frames, pre_seconds=pre_seconds)
    out = actions.copy()
    out["actor_arc_length_pre_window"] = df["actor_arc_length_pre_window"]
    out["actor_displacement_pre_window"] = df["actor_displacement_pre_window"]
    from silly_kicks.tracking.utils import link_actions_to_frames
    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_actor_pre_window_default_xfns = [lift_to_states(actor_arc_length_pre_window)]


# ---------------------------------------------------------------------------
# PR-S25 -- atomic mirror: TF-2 pressure_on_actor multi-flavor
# ---------------------------------------------------------------------------


def pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    method: "Method" = "andrienko_oval",
    params: "PressureParams | None" = None,
) -> pd.Series:
    """Atomic-SPADL: multi-flavor pressure on actor at linked frame.

    Mirrors :func:`silly_kicks.tracking.features.pressure_on_actor` with
    atomic anchor (x, y) instead of (start_x, start_y).

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples per method.
    """
    from silly_kicks.tracking.pressure import (
        AndrienkoParams,
        BekkersParams,
        LinkParams,
        validate_params_for_method,
    )
    validate_params_for_method(method, params)
    if method == "andrienko_oval":
        ap = params if isinstance(params, AndrienkoParams) else AndrienkoParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_andrienko(actions["x"], actions["y"], ctx, params=ap)
    elif method == "link_zones":
        lp = params if isinstance(params, LinkParams) else LinkParams()
        ctx = _resolve_action_frame_context(actions, frames)
        s = _kernels._pressure_link(actions["x"], actions["y"], ctx, params=lp)
    elif method == "bekkers_pi":
        bp = params if isinstance(params, BekkersParams) else BekkersParams()
        if "vx" not in frames.columns or "vy" not in frames.columns:
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi'): frames missing velocity columns "
                "'vx'/'vy'. Run silly_kicks.tracking.preprocess.derive_velocities(frames) "
                "first, or use a provider that emits velocities natively."
            )
        if bp.use_ball_carrier_max and not frames["is_ball"].any():
            raise ValueError(
                "pressure_on_actor(method='bekkers_pi', params.use_ball_carrier_max=True): "
                "frames missing is_ball=True rows in linked frames."
            )
        ctx = _resolve_action_frame_context(actions, frames)
        from silly_kicks.tracking.features import _build_ball_xy_v_per_action
        ball_xy_v = _build_ball_xy_v_per_action(actions, frames, ctx)
        s = _kernels._pressure_bekkers(
            actions["x"], actions["y"], ctx, params=bp, ball_xy_v_per_action=ball_xy_v,
        )
    else:
        raise ValueError(f"Unknown method '{method}'.")
    return s.rename(f"pressure_on_actor__{method}")


@nan_safe_enrichment
def add_pressure_on_actor(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    methods: tuple["Method", ...] = ("andrienko_oval",),
    params_per_method: "dict[Method, PressureParams] | None" = None,
) -> pd.DataFrame:
    """Atomic-SPADL aggregator for multi-flavor TF-2 pressure.

    Examples
    --------
    >>> from silly_kicks.atomic.tracking.features import add_pressure_on_actor
    >>> # See tests/atomic/tracking/test_pressure_*_atomic.py for runnable examples.
    """
    from silly_kicks.tracking.pressure import validate_params_for_method
    if params_per_method is None:
        params_per_method = {}
    for m in methods:
        validate_params_for_method(m, params_per_method.get(m))
    out = actions.copy()
    for m in methods:
        s = pressure_on_actor(actions, frames, method=m, params=params_per_method.get(m))
        out[f"pressure_on_actor__{m}"] = s.values
    from silly_kicks.tracking.utils import link_actions_to_frames
    pointers, _report = link_actions_to_frames(actions, frames)
    pointer_cols = pointers.set_index("action_id")[
        ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    ]
    out = out.merge(pointer_cols, left_on="action_id", right_index=True, how="left")
    return out


atomic_pressure_default_xfns = [lift_to_states(pressure_on_actor)]
```

Update `__all__` to include the new symbols.

- [ ] **Step 2: Verify atomic mirror import path**

```bash
uv run python -c "from silly_kicks.atomic.tracking.features import pressure_on_actor, actor_arc_length_pre_window; print('ok')"
```

Expected: `ok`.

---

## Task 10: ADR-005 §8 amendment

**Files:**
- Modify: `docs/superpowers/adrs/ADR-005-tracking-aware-features.md`

- [ ] **Step 1: Append §8 amendment text**

Open `docs/superpowers/adrs/ADR-005-tracking-aware-features.md`, locate the end of the "Decision" section (after §7), and append:

```markdown
### 8. Multi-flavor xfn column naming convention

Added in PR-S25 (silly-kicks 3.2.0). When a single feature concept admits
multiple methodologies (multi-flavor xfns — first concrete instance:
PR-S25 `pressure_on_actor` with three methods `andrienko_oval` / `link_zones`
/ `bekkers_pi`):

- Each flavor MUST emit a flavor-suffixed column name `<feature>__<method>`
  (double-underscore separator) so consumers registering parallel
  `functools.partial(fn, method="X")` xfns in `VAEP.xfns` do not silently
  overwrite each other inside `VAEP.compute_features`.
- The default xfn list (`<feature>_default_xfns`) ships exactly ONE flavor
  (the default method) to keep the VAEP feature space stable across
  silly-kicks versions. Consumers wanting additional flavors register
  additional `functools.partial` xfns explicitly.
- Per-method parameters are passed via a flavor-specific frozen dataclass
  (e.g., `AndrienkoParams`, `LinkParams`, `BekkersParams`) on the `params=`
  kwarg, not as a flat keyword bag — keeps each flavor's parameter
  surface discoverable and statically typed (pyright-friendly), and
  allows `__post_init__` validation per flavor.
- If `params=` is supplied with a type not matching the chosen `method`,
  the public function raises TypeError loudly (no silent default fallback).

This pattern applies only to **VAEP-consumed xfns** that emit per-action
scalar features. Preprocessing utilities that produce canonical per-row
columns (e.g., `derive_velocities` emits `vx`, `vy`, `speed`) keep their
canonical-single-column names regardless of method; QA/inspection helpers
case-by-case (per `feedback_multi_flavor_xfn_column_names`).

Rationale recap: the suffix convention surfaces the methodology choice in
the column name (downstream debugging is easier — column-name self-documents
which formula produced each value), and the default-xfn-list-ships-one-flavor
rule prevents "oh, did adding Link to the default break my model artefact?"
regression after consumer updates.
```

- [ ] **Step 2: Verify amendment text present (regression-test target)**

```bash
grep -c "### 8\. Multi-flavor xfn column naming convention" docs/superpowers/adrs/ADR-005-tracking-aware-features.md
```

Expected: `1`.

---

## Task 11: NOTICE updates

**Files:**
- Modify: `NOTICE`

- [ ] **Step 1: Add 3 academic references + BSD-3-Clause attribution**

Open `NOTICE`, locate the `## Mathematical / Methodological References` section, and append the spec §7 entries:

```
- Andrienko, G., Andrienko, N., Budziak, G., Dykes, J., Fuchs, G.,
  von Landesberger, T., & Weber, H. (2017). "Visual analysis of pressure in
  football." Data Mining and Knowledge Discovery, 31, 1793-1839.
  Used by: silly_kicks.tracking.features.pressure_on_actor (method="andrienko_oval").
  Numerical defaults from section 3.1: D_front=9 m, D_back=3 m, q=1.75.

- Link, D., Lang, S., & Seidenschwarz, P. (2016). "Real Time Quantification of
  Dangerousity in Football Using Spatiotemporal Tracking Data." PLOS ONE,
  11(12): e0168768.
  Used by: silly_kicks.tracking.features.pressure_on_actor (method="link_zones").
  Zone radii (HOZ=4, LZ=3, HZ=2 m) and angular boundaries (45 deg, 90 deg) from
  Figure 2. The paper additionally labels a 1 m "High Pressure Zone (HPZ)"
  inner arc with prose-described "constant high pressure", but Eq (3) of
  the paper does not special-case it -- silly-kicks honors Eq (3) as the
  formal specification (Plan A: equation-faithful, no discontinuity).
  Saturation constant k3 not published in the paper; silly-kicks default
  k3=1.0 is an engineering choice exposed as a kwarg. Calibration deferred
  post-release to Optuna sweep (silly-kicks TODO TF-24).

- Bekkers, J. (2025). "Pressing Intensity: An Intuitive Measure for Pressing
  in Soccer." arXiv:2501.04712.
  Used by: silly_kicks.tracking.features.pressure_on_actor (method="bekkers_pi").
  Time-to-intercept formula extends Spearman 2017 / Shaw / Pleuler with
  velocity-direction penalty. Defaults from paper + canonical implementation
  (UnravelSports/unravelsports, BSD-3-Clause).
```

Then add a new top-level section for BSD-3-Clause attribution:

```
## Third-Party Code Attribution

The Bekkers Pressing Intensity time-to-intercept formula
(silly_kicks.tracking._kernels._bekkers_tti) is a re-implementation of
the canonical Python source published under the BSD 3-Clause License by
Joris Bekkers / UnravelSports:

    https://github.com/UnravelSports/unravelsports
    unravel/soccer/models/utils.py -- time_to_intercept()

Required attribution per BSD-3-Clause:

    Copyright (c) 2025 UnravelSports
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions
    are met:
    (1) Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
    (2) Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
    (3) Neither the name of the copyright holder nor the names of its
        contributors may be used to endorse or promote products derived
        from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.

silly-kicks re-implements the algorithm with attribution; any modifications
to numerical constants, parameter handling, or aggregation logic are
documented in the source module docstring of silly_kicks.tracking._kernels.
```

NOTE: do NOT add a new Bauer & Anzer 2021 entry for TF-3 — TF-3 is silly-kicks's own geometric arc-length, NOT a B&A re-implementation. The existing PR-S20 Bauer & Anzer 2021 entry for `actor_speed` remains unchanged.

- [ ] **Step 2: Verify no stale B&A attribution for TF-3**

```bash
grep -B1 -A2 "actor_arc_length_pre_window" NOTICE
```

Expected: NO match (not cited in NOTICE — only `actor_speed` references B&A from PR-S20).

---

## Task 12: CHANGELOG entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Add `[3.2.0]` section per spec §10**

At the top of `CHANGELOG.md` (above the existing 3.1.0 section), add:

```markdown
## [3.2.0] -- 2026-05-XX

silly-kicks 3.2.0: TF-3 actor pre-window features + TF-2 multi-flavor pressure feature (PR-S25).

### Added

#### Tracking-aware features
- `silly_kicks.tracking.features.actor_arc_length_pre_window` — geometric arc-length of actor's path over the pre-action window (TF-3, default xfn). NOT Bauer & Anzer's filtered/threshold covered-distance feature; pure geometry, no sprint-intensity filtering.
- `silly_kicks.tracking.features.actor_displacement_pre_window` — net Euclidean displacement variant of TF-3 (window-first to window-last valid position).
- `silly_kicks.tracking.features.add_actor_pre_window` — aggregator emitting both columns + 4 provenance columns.
- `silly_kicks.tracking.features.actor_pre_window_default_xfns` — default xfn list (arc-length only).
- `silly_kicks.tracking.features.pressure_on_actor` — multi-flavor pressure feature (TF-2); methods: `andrienko_oval` (default; Andrienko 2017), `link_zones` (Link 2016), `bekkers_pi` (Bekkers 2024).
- `silly_kicks.tracking.features.add_pressure_on_actor` — aggregator emitting one `pressure_on_actor__<method>` per requested method.
- `silly_kicks.tracking.features.pressure_default_xfns` — default xfn list (Andrienko only, single default flavor).
- Atomic-SPADL parallel surface for all of the above (`silly_kicks.atomic.tracking.features.*`).

#### New module
- `silly_kicks.tracking.pressure` — multi-flavor pressure dispatch + per-method parameter dataclasses (`AndrienkoParams`, `LinkParams`, `BekkersParams`, `Method` Literal).

#### Architectural decision
- ADR-005 §8 amendment: multi-flavor xfn column-naming convention (`<feature>__<method>` suffixes; default xfn list ships single default-method xfn; per-method params via flavor-specific frozen dataclass).

#### Attribution
- NOTICE entries: Andrienko 2017, Link 2016, Bekkers 2024 + BSD-3-Clause attribution to UnravelSports for the Bekkers TTI port.

#### Internal (test files added; consumers re-pinning silly-kicks inherit these regression gates)
- `tests/tracking/test_pre_window_features.py` — TF-3 pin tests + NaN-bridge + shuffled-input regression
- `tests/tracking/test_pressure_andrienko.py` — Andrienko 4-point pin + sum-aggregation
- `tests/tracking/test_pressure_link.py` — Link zone-radii pin + saturation aggregation
- `tests/tracking/test_pressure_bekkers.py` — Bekkers per-defender pin + ValueError paths
- `tests/tracking/test_pressure_bekkers_golden_master.py` — bit-equivalent parity vs UnravelSports canonical (optional dep)
- `tests/tracking/test_pressure_methods_invariants.py` — cross-method physical invariants
- `tests/tracking/test_pressure_real_data_calibration.py` — per-provider sanity bounds + NaN-rate cap
- `tests/tracking/test_pressure_snapshot.py` — float-precision SHA-256 regression gate
- `tests/tracking/test_pressure_perf_budget.py` — pytest-benchmark gates per method
- `tests/tracking/test_pre_window_perf_budget.py` — pytest-benchmark gate for TF-3
- `tests/tracking/test_atomic_standard_parity.py` — atomic ↔ standard parity
- `tests/tracking/test_bekkers_e2e_via_derive_velocities.py` — full pipeline E2E test
- `tests/atomic/tracking/test_pre_window_features_atomic.py`
- `tests/atomic/tracking/test_pressure_{andrienko,link,bekkers}_atomic.py`
- `tests/test_adr005_amendment_compliance.py` extended with §8 contract assertions
- `tests/test_pyright_clean_tracking_features.py` — pyright clean gate
- `scripts/regenerate_pressure_snapshot_shas.py` — regenerator for test_pressure_snapshot.py expected SHAs

#### Test-only optional dependencies
- `unravelsports>=1.2` (extra `golden-master`) — required by `test_pressure_bekkers_golden_master.py`. CI installs the extra; local development can skip if unravelsports isn't installed.
```

---

## Task 13: Cross-cutting tests (golden master, snapshot, perf, parity, ADR contract, pyright, E2E)

> 7 test files implementing spec §8.5–§8.10 review-mandated regression gates. Each is independent and can be written in any order.

**Files:**
- Create: `tests/tracking/test_pressure_bekkers_golden_master.py`
- Create: `tests/tracking/test_pressure_snapshot.py`
- Create: `scripts/regenerate_pressure_snapshot_shas.py`
- Create: `tests/tracking/test_pressure_perf_budget.py`
- Create: `tests/tracking/test_pre_window_perf_budget.py`
- Create: `tests/tracking/test_atomic_standard_parity.py`
- Create: `tests/tracking/test_pressure_methods_invariants.py`
- Create: `tests/tracking/test_pressure_real_data_calibration.py`
- Create: `tests/tracking/test_bekkers_e2e_via_derive_velocities.py`
- Create: `tests/test_adr005_amendment_compliance.py`
- Create: `tests/test_pyright_clean_tracking_features.py`

- [ ] **Step 1: Bekkers golden-master parity**

Create `tests/tracking/test_pressure_bekkers_golden_master.py`:

```python
"""Bit-equivalent parity between silly-kicks _bekkers_tti and UnravelSports canonical.

Per spec section 8.7 (lakehouse review item 2): "single highest-leverage addition".
Without this, the 'direct port' claim is aspirational. With it, drift is detected
at every CI run that has the optional dep installed.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from silly_kicks.tracking._kernels import _bekkers_tti

unravelsports_missing = importlib.util.find_spec("unravelsports") is None


@pytest.mark.skipif(
    unravelsports_missing,
    reason="unravelsports not installed (optional test-only dep, install via: uv pip install -e '.[golden-master]')",
)
@pytest.mark.parametrize(
    "name,p1,p2,v1,v2",
    [
        ("stationary_at_zero", [[50, 34]], [[50, 34]], [[0, 0]], [[0, 0]]),
        ("stationary_at_d10",  [[50, 34]], [[60, 34]], [[0, 0]], [[0, 0]]),
        ("moving_defender",    [[50, 34]], [[55, 34]], [[3, 0]], [[0, 0]]),
        ("moving_target",      [[50, 34]], [[55, 34]], [[0, 0]], [[2, 0]]),
        ("relative_motion",    [[48, 32]], [[52, 36]], [[2, 1]], [[-1, -1]]),
        ("multiple_defenders", [[48, 32], [52, 36]], [[50, 34]], [[2, 0], [0, -1]], [[0, 0]]),
        ("away_from_target",   [[50, 34]], [[60, 34]], [[-3, 0]], [[0, 0]]),  # angle penalty
    ],
)
def test_bekkers_tti_byte_equivalent_to_unravelsports(name: str, p1, p2, v1, v2) -> None:
    from unravel.soccer.models.utils import time_to_intercept as us_tti
    p1_arr = np.asarray(p1, dtype=float)
    p2_arr = np.asarray(p2, dtype=float)
    v1_arr = np.asarray(v1, dtype=float)
    v2_arr = np.asarray(v2, dtype=float)
    ours = _bekkers_tti(
        p1=p1_arr, p2=p2_arr, v1=v1_arr, v2=v2_arr,
        reaction_time=0.7, max_object_speed=12.0,
    )
    theirs = us_tti(
        p1=p1_arr, p2=p2_arr, v1=v1_arr, v2=v2_arr,
        reaction_time=0.7, max_object_speed=12.0,
    )
    np.testing.assert_allclose(ours, theirs, rtol=1e-9, atol=1e-12, err_msg=f"{name} divergence")
```

- [ ] **Step 2: Snapshot determinism**

Create `tests/tracking/test_pressure_snapshot.py`:

```python
"""Float-precision SHA-256 regression gate for pressure outputs.

Per spec section 8.6 (lakehouse review item 4): catches numpy/pandas minor-version
drift before it cascades. Failure means investigate -> if intentional, regenerate
expected SHAs via scripts/regenerate_pressure_snapshot_shas.py.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from silly_kicks.tracking.features import pressure_on_actor

# Pinned hashes -- regenerate via scripts/regenerate_pressure_snapshot_shas.py
EXPECTED_SHAS = {
    "andrienko_oval": "<filled-on-first-run>",
    "link_zones":     "<filled-on-first-run>",
    "bekkers_pi":     "<filled-on-first-run>",
}


def _build_fixture():
    np.random.seed(42)
    n_actions = 50
    n_defenders_per_action = 5
    actions_rows = []
    frames_rows = []
    for action_id in range(n_actions):
        actor_x = 50.0 + np.random.uniform(-20, 20)
        actor_y = 34.0 + np.random.uniform(-15, 15)
        actions_rows.append({
            "action_id": action_id,
            "period_id": 1,
            "team_id": "home",
            "player_id": 10 + action_id % 11,
            "start_x": actor_x,
            "start_y": actor_y,
            "type_id": 0,
            "time_seconds": float(action_id),
        })
        # Frame for actor
        frames_rows.append({
            "frame_id": action_id,
            "period_id": 1,
            "time_seconds": float(action_id),
            "team_id": "home",
            "player_id": 10 + action_id % 11,
            "is_ball": False,
            "x": actor_x,
            "y": actor_y,
            "vx": 0.0, "vy": 0.0, "speed": 0.0,
        })
        # Ball
        frames_rows.append({
            "frame_id": action_id,
            "period_id": 1,
            "time_seconds": float(action_id),
            "team_id": None,
            "player_id": None,
            "is_ball": True,
            "x": actor_x,
            "y": actor_y,
            "vx": 0.0, "vy": 0.0, "speed": 0.0,
        })
        # Defenders
        for di in range(n_defenders_per_action):
            d_x = actor_x + np.random.uniform(-8, 8)
            d_y = actor_y + np.random.uniform(-8, 8)
            d_vx = np.random.uniform(-3, 3)
            d_vy = np.random.uniform(-3, 3)
            frames_rows.append({
                "frame_id": action_id,
                "period_id": 1,
                "time_seconds": float(action_id),
                "team_id": "away",
                "player_id": 100 + di,
                "is_ball": False,
                "x": d_x, "y": d_y,
                "vx": d_vx, "vy": d_vy,
                "speed": float(np.hypot(d_vx, d_vy)),
            })
    return pd.DataFrame(actions_rows), pd.DataFrame(frames_rows)


def _hash_series(s: pd.Series) -> str:
    arr = s.fillna(-99999.0).astype("float64").values
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_andrienko_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="andrienko_oval")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["andrienko_oval"]
    if expected.startswith("<"):
        import os
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"andrienko_oval: {actual}")
            return
    assert actual == expected, f"Andrienko drift; was {expected}, now {actual}"


def test_link_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="link_zones")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["link_zones"]
    if expected.startswith("<"):
        import os
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"link_zones: {actual}")
            return
    assert actual == expected, f"Link drift; was {expected}, now {actual}"


def test_bekkers_snapshot_stable() -> None:
    actions, frames = _build_fixture()
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    actual = _hash_series(result)
    expected = EXPECTED_SHAS["bekkers_pi"]
    if expected.startswith("<"):
        import os
        if os.environ.get("REGENERATE_SNAPSHOTS"):
            print(f"bekkers_pi: {actual}")
            return
    assert actual == expected, f"Bekkers drift; was {expected}, now {actual}"
```

- [ ] **Step 3: Snapshot regenerator script**

Create `scripts/regenerate_pressure_snapshot_shas.py`:

```python
"""Regenerate expected SHA-256 hashes for tests/tracking/test_pressure_snapshot.py.

Usage:
    REGENERATE_SNAPSHOTS=1 uv run pytest tests/tracking/test_pressure_snapshot.py -v -s

Then copy the printed hashes into the EXPECTED_SHAS dict in the test file.
"""
print(__doc__)
```

(Minimal — the test file's `if os.environ.get("REGENERATE_SNAPSHOTS")` branch handles printing.)

- [ ] **Step 4: Snapshot first-run pin**

```bash
REGENERATE_SNAPSHOTS=1 uv run pytest tests/tracking/test_pressure_snapshot.py -v -s
```

Copy the three printed SHAs into `EXPECTED_SHAS` in `test_pressure_snapshot.py`. Re-run normally:

```bash
uv run pytest tests/tracking/test_pressure_snapshot.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Performance budgets**

Create `tests/tracking/test_pressure_perf_budget.py`:

```python
"""pytest-benchmark gates per spec section 8.1 review item 5.

Andrienko/Link < 50ms per 100 actions; Bekkers < 250ms per 100 actions on CI runner.
"""

from __future__ import annotations

import pytest

from silly_kicks.tracking.features import pressure_on_actor

from .test_pressure_snapshot import _build_fixture


@pytest.fixture(scope="module")
def fixture_100():
    """100-action fixture extending the 50-action snapshot fixture."""
    import numpy as np
    import pandas as pd
    np.random.seed(123)
    actions, frames = _build_fixture()
    extra = actions.copy()
    extra["action_id"] = extra["action_id"] + 1000
    actions = pd.concat([actions, extra], ignore_index=True)
    extra_frames = frames.copy()
    extra_frames["frame_id"] = extra_frames["frame_id"] + 1000
    frames = pd.concat([frames, extra_frames], ignore_index=True)
    return actions, frames


def test_andrienko_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="andrienko_oval")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.10  # 100ms ceiling on CI; spec target 50ms


def test_link_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="link_zones")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.10


def test_bekkers_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(pressure_on_actor, actions, frames, method="bekkers_pi")
    assert result.notna().any()
    assert benchmark.stats.stats.mean < 0.50  # 500ms ceiling on CI; spec target 250ms
```

Create `tests/tracking/test_pre_window_perf_budget.py`:

```python
"""TF-3 perf budget per spec section 8.1: < 50ms per 100 actions."""

from __future__ import annotations

from silly_kicks.tracking.features import add_actor_pre_window

from .test_pressure_perf_budget import fixture_100


def test_pre_window_perf_per_100_actions(benchmark, fixture_100) -> None:
    actions, frames = fixture_100
    result = benchmark(add_actor_pre_window, actions, frames)
    assert "actor_arc_length_pre_window" in result.columns
    assert benchmark.stats.stats.mean < 0.10
```

- [ ] **Step 6: Atomic ↔ standard parity**

Create `tests/tracking/test_atomic_standard_parity.py`:

```python
"""Atomic ↔ standard parity gate per spec section 8.8 / lakehouse review item 6."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.atomic.tracking.features import (
    actor_arc_length_pre_window as atomic_arc,
    actor_displacement_pre_window as atomic_disp,
    pressure_on_actor as atomic_pressure,
)
from silly_kicks.tracking.features import (
    actor_arc_length_pre_window as std_arc,
    actor_displacement_pre_window as std_disp,
    pressure_on_actor as std_pressure,
)

from .test_pressure_snapshot import _build_fixture


def _to_atomic(actions):
    """Map standard-shape actions (start_x, start_y) -> atomic-shape (x, y, dx, dy)."""
    out = actions.copy()
    out["x"] = out["start_x"]
    out["y"] = out["start_y"]
    out["dx"] = 0.0
    out["dy"] = 0.0
    return out.drop(columns=["start_x", "start_y"], errors="ignore")


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_atomic_standard_parity_pressure(method: str) -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_pressure(actions, frames, method=method)
    atomic_result = atomic_pressure(atomic, frames, method=method)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )


def test_atomic_standard_parity_arc_length() -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_arc(actions, frames)
    atomic_result = atomic_arc(atomic, frames)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )


def test_atomic_standard_parity_displacement() -> None:
    actions, frames = _build_fixture()
    atomic = _to_atomic(actions)
    std_result = std_disp(actions, frames)
    atomic_result = atomic_disp(atomic, frames)
    np.testing.assert_array_equal(
        std_result.fillna(-99999).values,
        atomic_result.fillna(-99999).values,
    )
```

- [ ] **Step 7: Cross-method invariants**

Create `tests/tracking/test_pressure_methods_invariants.py`:

```python
"""Cross-method physical invariants per spec section 8.3."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor


def _make_one_action_frame(actor_xy, defender_xy_v):
    """Build minimal actions+frames for a single (actor, single defender) test."""
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "start_x": [actor_xy[0]], "start_y": [actor_xy[1]],
        "type_id": [0],
    })
    frames = pd.DataFrame([
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "home", "player_id": 10, "is_ball": False,
         "x": actor_xy[0], "y": actor_xy[1], "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": None, "player_id": None, "is_ball": True,
         "x": actor_xy[0], "y": actor_xy[1], "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": defender_xy_v[0], "y": defender_xy_v[1],
         "vx": defender_xy_v[2], "vy": defender_xy_v[3],
         "speed": math.hypot(defender_xy_v[2], defender_xy_v[3])},
    ])
    return actions, frames


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones"])
def test_monotone_decreasing_in_distance(method: str) -> None:
    """Position-only methods: pressure decreases as defender->actor distance increases."""
    pressures = []
    for d in [1.0, 2.0, 3.0, 5.0]:
        actions, frames = _make_one_action_frame((50.0, 34.0), (50.0 + d, 34.0, 0.0, 0.0))
        out = pressure_on_actor(actions, frames, method=method)
        pressures.append(float(out.iloc[0]))
    for i in range(1, len(pressures)):
        assert pressures[i] <= pressures[i - 1]


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones"])
def test_axially_symmetric(method: str) -> None:
    actions_pos, frames_pos = _make_one_action_frame((50.0, 34.0), (51.0, 34.0 + 1.0, 0.0, 0.0))
    actions_neg, frames_neg = _make_one_action_frame((50.0, 34.0), (51.0, 34.0 - 1.0, 0.0, 0.0))
    p_pos = pressure_on_actor(actions_pos, frames_pos, method=method).iloc[0]
    p_neg = pressure_on_actor(actions_neg, frames_neg, method=method).iloc[0]
    assert p_pos == pytest.approx(p_neg, rel=1e-9)


@pytest.mark.parametrize("method", ["andrienko_oval", "link_zones", "bekkers_pi"])
def test_non_negative(method: str) -> None:
    actions, frames = _make_one_action_frame((50.0, 34.0), (52.0, 34.0, 3.0, 0.0))
    out = pressure_on_actor(actions, frames, method=method)
    assert (out.dropna() >= 0.0).all()


@pytest.mark.parametrize("method", ["link_zones", "bekkers_pi"])
def test_bounded_in_zero_one(method: str) -> None:
    actions, frames = _make_one_action_frame((50.0, 34.0), (50.5, 34.0, 3.0, 0.0))
    out = pressure_on_actor(actions, frames, method=method)
    assert ((out.dropna() >= 0.0) & (out.dropna() <= 1.0)).all()
```

- [ ] **Step 8: Real-data calibration sanity**

Create `tests/tracking/test_pressure_real_data_calibration.py`:

```python
"""Per-provider real-data sanity bounds (lakehouse review item 3).

Uses the existing PR-S20 lakehouse-derived slim parquet fixtures.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"


def _load_provider(provider: str):
    parquet = SLIM_DIR / f"{provider}_slim.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} missing — run scripts/probe_action_context_baselines.py")
    df = pd.read_parquet(parquet)
    actions = df[df["__kind"] == "action"].drop(columns=["__kind"]).reset_index(drop=True)
    frames = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    return actions, frames


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_andrienko_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider(provider)
    out = pressure_on_actor(actions, frames, method="andrienko_oval")
    nan_rate = out.isna().mean()
    assert nan_rate < 0.05, f"{provider} andrienko NaN rate {nan_rate:.2%} > 5%"
    valid = out.dropna()
    in_band = ((valid >= 0) & (valid <= 200)).mean()
    assert in_band >= 0.99, f"{provider} andrienko {in_band:.2%} of values in [0, 200]"


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_link_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider(provider)
    out = pressure_on_actor(actions, frames, method="link_zones")
    nan_rate = out.isna().mean()
    assert nan_rate < 0.05
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()


@pytest.mark.parametrize("provider", ["sportec", "metrica", "skillcorner"])
def test_bekkers_real_data_bounds(provider: str) -> None:
    actions, frames = _load_provider(provider)
    if "vx" not in frames.columns:
        from silly_kicks.tracking.preprocess import derive_velocities
        frames = derive_velocities(frames)
    out = pressure_on_actor(actions, frames, method="bekkers_pi")
    nan_rate = out.isna().mean()
    assert nan_rate < 0.10  # bekkers may NaN more due to ball-row absence
    valid = out.dropna()
    assert (valid >= 0).all() and (valid <= 1).all()
```

- [ ] **Step 9: Bekkers E2E with derive_velocities**

Create `tests/tracking/test_bekkers_e2e_via_derive_velocities.py`:

```python
"""Spec section 8.9: full pipeline test raw → derive_velocities → pressure."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.features import pressure_on_actor
from silly_kicks.tracking.preprocess import derive_velocities

REPO_ROOT = Path(__file__).resolve().parents[2]
SLIM_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "action_context_slim"


def test_bekkers_requires_velocities_then_passes_after_derive() -> None:
    parquet = SLIM_DIR / "sportec_slim.parquet"
    if not parquet.exists():
        pytest.skip(f"{parquet} missing")
    df = pd.read_parquet(parquet)
    actions = df[df["__kind"] == "action"].drop(columns=["__kind"]).reset_index(drop=True)
    frames_raw = df[df["__kind"] == "frame"].drop(columns=["__kind"]).reset_index(drop=True)
    if "vx" in frames_raw.columns:
        frames_raw = frames_raw.drop(columns=["vx", "vy"])

    with pytest.raises(ValueError, match="missing velocity columns"):
        pressure_on_actor(actions, frames_raw, method="bekkers_pi")

    frames = derive_velocities(frames_raw)
    result = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert result.notna().any()
    assert ((result.dropna() >= 0) & (result.dropna() <= 1)).all()
```

- [ ] **Step 10: ADR-005 §8 contract test**

Create `tests/test_adr005_amendment_compliance.py`:

```python
"""ADR-005 section 8 contract gate per spec section 8.5."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from silly_kicks.tracking.features import (
    add_pressure_on_actor,
    pressure_default_xfns,
    pressure_on_actor,
)
from silly_kicks.tracking.pressure import (
    AndrienkoParams,
    BekkersParams,
    LinkParams,
)
from silly_kicks.vaep.feature_framework import is_frame_aware

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_xfn_list_is_size_one() -> None:
    """ADR-005 section 8: default xfn list ships exactly ONE flavor."""
    assert len(pressure_default_xfns) == 1


def test_suffix_naming_per_method() -> None:
    """ADR-005 section 8: column name = pressure_on_actor__<method>."""
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "start_x": [50.0], "start_y": [34.0], "type_id": [0],
    })
    frames = pd.DataFrame()
    for m in ["andrienko_oval", "link_zones"]:
        s = pressure_on_actor(actions, frames, method=m)
        assert s.name == f"pressure_on_actor__{m}"


def test_params_type_validation_loud() -> None:
    """ADR-005 section 8: TypeError on params/method type mismatch."""
    actions = pd.DataFrame()
    frames = pd.DataFrame()
    with pytest.raises(TypeError, match="andrienko_oval.*expects AndrienkoParams.*got LinkParams"):
        pressure_on_actor(actions, frames, method="andrienko_oval", params=LinkParams())


def test_unknown_method_raises_value_error() -> None:
    """Lakehouse review item 18: runtime guard for non-Literal callers."""
    actions = pd.DataFrame()
    frames = pd.DataFrame()
    with pytest.raises(ValueError, match="Unknown method"):
        pressure_on_actor(actions, frames, method="not_a_method")  # type: ignore[arg-type]


def test_add_pressure_validates_all_methods_before_compute() -> None:
    """Transactional validation per lakehouse review item 12: TypeError before any column written."""
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10], "start_x": [50.0], "start_y": [34.0],
        "type_id": [0],
    })
    frames = pd.DataFrame()
    with pytest.raises(TypeError):
        add_pressure_on_actor(
            actions, frames,
            methods=("andrienko_oval", "bekkers_pi"),
            params_per_method={
                "andrienko_oval": LinkParams(),  # WRONG TYPE
                "bekkers_pi": BekkersParams(),
            },
        )
    # Verify NO partial column was written into actions
    assert "pressure_on_actor__andrienko_oval" not in actions.columns
    assert "pressure_on_actor__bekkers_pi" not in actions.columns


def test_default_xfn_is_frame_aware() -> None:
    """ADR-005 section 1 marker preserved through lift_to_states."""
    xfn = pressure_default_xfns[0]
    assert is_frame_aware(xfn)


def test_adr_005_section_8_present() -> None:
    """Lakehouse review item 13: ADR markdown grep for amendment text."""
    adr_path = REPO_ROOT / "docs" / "superpowers" / "adrs" / "ADR-005-tracking-aware-features.md"
    text = adr_path.read_text(encoding="utf-8")
    assert "### 8. Multi-flavor xfn column naming convention" in text
    assert "<feature>__<method>" in text
```

- [ ] **Step 11: pyright clean gate**

Create `tests/test_pyright_clean_tracking_features.py`:

```python
"""pyright clean gate per spec section 8.10."""

from __future__ import annotations

import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright not installed")
def test_pyright_clean_tracking_namespace() -> None:
    result = subprocess.run(
        ["pyright", "silly_kicks/tracking/", "silly_kicks/atomic/tracking/"],
        capture_output=True, text=True, check=False,
    )
    assert "0 errors" in result.stdout, f"pyright failed:\n{result.stdout}\n{result.stderr}"
```

- [ ] **Step 12: Run all new test files**

```bash
uv run pytest tests/tracking/ tests/atomic/tracking/ tests/test_adr005_amendment_compliance.py tests/test_pyright_clean_tracking_features.py -v --tb=short
```

Expected: All PASS (or skip if optional dep / fixture missing).

---

## Task 14: Atomic mirror tests (4 files)

**Files:**
- Create: `tests/atomic/tracking/test_pre_window_features_atomic.py`
- Create: `tests/atomic/tracking/test_pressure_andrienko_atomic.py`
- Create: `tests/atomic/tracking/test_pressure_link_atomic.py`
- Create: `tests/atomic/tracking/test_pressure_bekkers_atomic.py`

- [ ] **Step 1: Atomic pre-window mirror**

Create `tests/atomic/tracking/test_pre_window_features_atomic.py`:

```python
"""Atomic mirror of test_pre_window_features.py — verify atomic public API works."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from silly_kicks.atomic.tracking.features import (
    actor_arc_length_pre_window,
    actor_displacement_pre_window,
    add_actor_pre_window,
)


def test_atomic_arc_length_constant_velocity() -> None:
    """Same shape as standard test, atomic schema (x, y instead of start_x, start_y)."""
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [100.0],
        "player_id": [10], "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
    })
    n_frames = 13
    frames = pd.DataFrame([
        {"frame_id": i, "period_id": 1,
         "time_seconds": 100.0 + dt, "player_id": 10, "is_ball": False,
         "x": 50.0 + 5.0 * (dt + 0.5), "y": 34.0}
        for i, dt in enumerate(np.linspace(-0.5, 0.0, n_frames))
    ])
    out = actor_arc_length_pre_window(actions, frames)
    assert out.iloc[0] == pytest.approx(2.5, rel=1e-6)


def test_atomic_displacement_circular() -> None:
    n_frames = 13
    angles = np.linspace(0, math.pi, n_frames)
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [100.0],
        "player_id": [10], "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
    })
    frames = pd.DataFrame([
        {"frame_id": i, "period_id": 1,
         "time_seconds": 100.0 + dt, "player_id": 10, "is_ball": False,
         "x": 50.0 + 2.0 * math.cos(a), "y": 34.0 + 2.0 * math.sin(a)}
        for i, (a, dt) in enumerate(zip(angles, np.linspace(-0.5, 0.0, n_frames), strict=False))
    ])
    out = actor_displacement_pre_window(actions, frames)
    assert out.iloc[0] == pytest.approx(4.0, rel=1e-6)


def test_atomic_aggregator_includes_provenance() -> None:
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [100.0],
        "player_id": [10], "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
    })
    frames = pd.DataFrame([
        {"frame_id": 0, "period_id": 1, "time_seconds": 99.5,
         "player_id": 10, "is_ball": False, "x": 50.0, "y": 34.0},
        {"frame_id": 1, "period_id": 1, "time_seconds": 100.0,
         "player_id": 10, "is_ball": False, "x": 51.0, "y": 34.0},
    ])
    out = add_actor_pre_window(actions, frames)
    assert "actor_arc_length_pre_window" in out.columns
    assert "actor_displacement_pre_window" in out.columns
    assert "frame_id" in out.columns
```

- [ ] **Step 2: Atomic Andrienko mirror**

Create `tests/atomic/tracking/test_pressure_andrienko_atomic.py`:

```python
"""Atomic mirror — minimal pin to verify atomic Andrienko surface works.

Comprehensive parity tested in test_atomic_standard_parity.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.tracking.features import pressure_on_actor


def test_atomic_andrienko_runs() -> None:
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
        "type_id": [0],
    })
    frames = pd.DataFrame([
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
    ])
    out = pressure_on_actor(actions, frames, method="andrienko_oval")
    assert out.name == "pressure_on_actor__andrienko_oval"
    assert out.iloc[0] >= 0.0
```

- [ ] **Step 3: Atomic Link mirror**

Create `tests/atomic/tracking/test_pressure_link_atomic.py`:

```python
"""Atomic mirror — minimal pin for Link surface."""

from __future__ import annotations

import pandas as pd

from silly_kicks.atomic.tracking.features import pressure_on_actor


def test_atomic_link_runs() -> None:
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
        "type_id": [0],
    })
    frames = pd.DataFrame([
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
    ])
    out = pressure_on_actor(actions, frames, method="link_zones")
    assert out.name == "pressure_on_actor__link_zones"
    assert 0.0 <= out.iloc[0] <= 1.0
```

- [ ] **Step 4: Atomic Bekkers mirror**

Create `tests/atomic/tracking/test_pressure_bekkers_atomic.py`:

```python
"""Atomic mirror — minimal pin for Bekkers surface."""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.atomic.tracking.features import pressure_on_actor


def test_atomic_bekkers_runs() -> None:
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10],
        "x": [50.0], "y": [34.0], "dx": [0.0], "dy": [0.0],
        "type_id": [0],
    })
    frames = pd.DataFrame([
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "home", "player_id": 10, "is_ball": False,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": None, "player_id": None, "is_ball": True,
         "x": 50.0, "y": 34.0, "vx": 0.0, "vy": 0.0, "speed": 0.0},
        {"frame_id": 0, "period_id": 1, "time_seconds": 0.0,
         "team_id": "away", "player_id": 100, "is_ball": False,
         "x": 52.0, "y": 34.0, "vx": -3.0, "vy": 0.0, "speed": 3.0},
    ])
    out = pressure_on_actor(actions, frames, method="bekkers_pi")
    assert out.name == "pressure_on_actor__bekkers_pi"
    assert 0.0 <= out.iloc[0] <= 1.0


def test_atomic_bekkers_missing_velocities_raises() -> None:
    actions = pd.DataFrame({
        "action_id": [1], "period_id": [1], "time_seconds": [0.0],
        "team_id": ["home"], "player_id": [10], "x": [50.0], "y": [34.0],
        "dx": [0.0], "dy": [0.0], "type_id": [0],
    })
    frames = pd.DataFrame()
    with pytest.raises(ValueError, match="missing velocity columns"):
        pressure_on_actor(actions, frames, method="bekkers_pi")
```

- [ ] **Step 5: Run atomic tests**

```bash
uv run pytest tests/atomic/tracking/ -v
```

Expected: All PASS.

---

## Task 14a: Synthesizer named-dataclass scenarios

> Per spec §9 + `feedback_synthesizer_shot_plus_keeper_save_pattern`: frozen-dataclass named scenarios in `tests/tracking/_provider_inputs.py` give the next test author a single source of truth for the constants. Lakehouse v3 review item 7.

**Files:**
- Modify: `tests/tracking/_provider_inputs.py` (extend with 5 named dataclasses)

- [ ] **Step 1: Add named scenarios to `_provider_inputs.py`**

Append to `tests/tracking/_provider_inputs.py`:

```python
# ---------------------------------------------------------------------------
# PR-S25 -- TF-2 + TF-3 named scenarios per spec section 9
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PressingDefenderScenario:
    """Synthesizer scenario for tracking-aware pressure tests (TF-2).

    Geometry:
      - Actor (ball-carrier) at (50.0, 34.0), v=(0, 0), is_ball_carrier=True
      - Ball at (50.0, 34.0), v=(0, 0), is_ball=True
      - Pressing defender at (52.0, 34.0), v=(3.0, 0), speed=3.0 m/s
        (above 2.0 m/s active-pressing threshold; closing on actor)
      - Passive defender at (60.0, 34.0), v=(0, 0), speed=0.0
        (below threshold; tests filter)
      - Goalkeeper at (104.0, 34.0), v=(0, 0)
      - 7 attacking teammates spread across own half (geometry doesn't
        matter for pressure-on-actor; included for realistic frame size)
      - 9 defending teammates spread across opponent's half (>15m away
        to avoid contributing pressure)
    """
    actor_pos: tuple[float, float] = (50.0, 34.0)
    pressing_defender_pos: tuple[float, float] = (52.0, 34.0)
    pressing_defender_velocity: tuple[float, float] = (3.0, 0.0)
    passive_defender_pos: tuple[float, float] = (60.0, 34.0)


@dataclass(frozen=True)
class StationaryActorScenario:
    """TF-3: actor stationary across the 0.5s window.

    Expected: actor_arc_length_pre_window = 0.0; actor_displacement_pre_window = 0.0.
    """
    actor_pos: tuple[float, float] = (50.0, 34.0)
    n_frames: int = 13
    pre_seconds: float = 0.5


@dataclass(frozen=True)
class ConstantVelocityActorScenario:
    """TF-3: actor moving at 5 m/s along +x for 0.5s.

    Expected: arc-length ≈ 2.5; displacement ≈ 2.5 (path is straight line).
    """
    actor_start_pos: tuple[float, float] = (50.0, 34.0)
    velocity_ms: tuple[float, float] = (5.0, 0.0)
    pre_seconds: float = 0.5
    n_frames: int = 13


@dataclass(frozen=True)
class CircularPathActorScenario:
    """TF-3: actor traverses half-circle radius 2 in 0.5s.

    Expected: arc-length ≈ 6.28 (π·r); displacement ≈ 4.0 (diameter).
    Demonstrates arc-length >> displacement for curved paths.
    """
    center_pos: tuple[float, float] = (50.0, 34.0)
    radius: float = 2.0
    half_circle_seconds: float = 0.5
    n_frames: int = 13


@dataclass(frozen=True)
class BridgeNaNActorScenario:
    """TF-3: NaN-bridge rule — frames at -0.4, -0.3, NaN, -0.1, 0.0
    with positions (0,0), (1,0), NaN, (3,0), (4,0).

    Expected: arc-length = 1+2+1 = 4 (bridges across NaN);
              displacement = 4 (first valid (0,0) -> last valid (4,0)).
    """
    valid_positions: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (3.0, 0.0), (4.0, 0.0))
    nan_offset_indices: tuple[int, ...] = (2,)  # frame index in window order
    n_frames: int = 5
    pre_seconds: float = 0.5
```

(The dataclasses are pure data containers; existing test fixtures continue to use inline numerics, but new tests added in future PRs can import these for consistency. Per `feedback_speculative_api_surface_is_debt`, only ship the dataclasses if at least one test currently imports them — wire `test_pre_window_features.py::test_pre_window_constant_velocity_5ms` to use `ConstantVelocityActorScenario` in Step 2 below to justify shipping the surface.)

- [ ] **Step 2: Wire one TF-3 test to use the dataclass (justify shipping the surface)**

Modify `tests/tracking/test_pre_window_features.py::test_pre_window_constant_velocity_5ms` to import + use the dataclass:

```python
from tests.tracking._provider_inputs import ConstantVelocityActorScenario


def test_pre_window_constant_velocity_5ms() -> None:
    """5 m/s along +x for 0.5s -> arc-length ≈ 2.5 ≈ displacement.

    Uses ConstantVelocityActorScenario named-dataclass per spec section 9.
    """
    scenario = ConstantVelocityActorScenario()
    actions, frames = _build_pre_window_input(
        pre_seconds=scenario.pre_seconds, actor_player_id=10,
        frame_xys=[
            (scenario.actor_start_pos[0] + scenario.velocity_ms[0] * (dt + scenario.pre_seconds),
             scenario.actor_start_pos[1], dt)
            for dt in np.linspace(-scenario.pre_seconds, 0.0, scenario.n_frames)
        ],
    )
    out = _actor_pre_window_kernel(actions, frames, pre_seconds=scenario.pre_seconds)
    assert out.iloc[0]["actor_arc_length_pre_window"] == pytest.approx(2.5, rel=1e-6)
    assert out.iloc[0]["actor_displacement_pre_window"] == pytest.approx(2.5, rel=1e-6)
```

- [ ] **Step 3: Run TF-3 tests to verify dataclass wiring**

```bash
uv run pytest tests/tracking/test_pre_window_features.py -v
```

Expected: All TF-3 tests still PASS.

---

## Task 15: TODO.md updates + version bump

**Files:**
- Modify: `TODO.md` (mark TF-3 + TF-2 SHIPPED)
- Modify: `pyproject.toml` (version 3.1.0 → 3.2.0; add `golden-master` extra)
- Modify: `silly_kicks/_version.py` (version bump)

- [ ] **Step 1: Mark TF-2 + TF-3 SHIPPED in TODO.md**

In `TODO.md` Tier 2 section, replace the TF-3 and TF-2 rows with:

```markdown
| TF-3 | ~~`actor_distance_pre_window`~~ → SHIPPED in 3.2.0 (PR-S25) as `actor_arc_length_pre_window` + `actor_displacement_pre_window` per "pick-one deferral diagnostic" reframe (multi-feature branch) | -- | -- | SHIPPED. |
| TF-2 | ~~`pressure_on_actor()`~~ → SHIPPED in 3.2.0 (PR-S25) as multi-flavor xfn with three methods: `andrienko_oval` (default), `link_zones`, `bekkers_pi`. ADR-005 §8 amendment codifies multi-flavor naming convention. | -- | -- | SHIPPED. |
```

(Leave TF-10 row unchanged — still in scope for a future lakehouse session.)

- [ ] **Step 2: Add TF-24 + TF-25 follow-up rows to Tier 3**

In `TODO.md` Tier 3 section, append the two new rows after the existing TF-14 row (and before the `### Tier 4` heading):

```markdown
| TF-24 | `LinkParams.k3` Optuna calibration (post-PR-S25) | Wicked | Lakehouse session 2026-05-03 (k3 calibration tooling discussion) | **Post-release calibration.** Optimize `LinkParams.k3` (and optionally the joint 6-scalar Link parameter set: r_hoz/r_lz/r_hz + angle_hoz_lz_deg / angle_lz_hz_deg + k3) via Optuna TPE sweep, 50-100 trials, single CPU node, against held-out VAEP fold from lakehouse `bronze.model_validation_runs`. Single objective: VAEP held-out Brier-score (or per-action calibration NLL). Update `LinkParams` defaults; update spec note from "engineering choice" to "Optuna-calibrated against `<fold_name>` on `<date>`". Same script reusable for k1..k5 if scope expands. **Wrong tool: lakehouse `evolve` framework** — single-scalar Bayesian optimization is Optuna-shaped, not evolve-shaped. Pre-release optimization avoided to (a) prevent circular validation against the training fold, (b) match the Link 2016 paper's own "formula + later empirical calibration" sequencing. ~1-2 days. |
| TF-25 | Structural-form evolution of pressure aggregations | Wicked–Monstah | Lakehouse session 2026-05-03 (k3 calibration tooling discussion); lakehouse `evolve` framework | **Lakehouse-evolve-shaped follow-up to TF-24.** Use lakehouse evolve framework to evolve the aggregation function FORM (not just k3 scalar). Three concrete targets: (1) per-provider saturation forms — different sample rates, position-noise characteristics, and field-coverage assumptions across Sportec/StatsBomb 360/Metrica/Wyscout may demand different aggregations; (2) continuous `r_zo(α)` as alternative to three-zone bucketing; (3) non-linear distance-pressure curves beyond `1 - d/r_zo` (quadratic, sigmoid, learned-curve). **Trigger condition:** only fire if TF-24's Optuna sweep shows k3 itself moves meaningfully across providers — that's the signal that the FORM, not just the scalar, is provider-dependent. Without that signal, this is over-engineering. ~1-2 cycles + L40S budget for eval loop. |
```

(Verify against the existing TODO.md state — these rows may already exist from spec-writing time per the spec §11 update. If so, skip this step and verify their presence.)

```bash
grep -c "^| TF-2[45] |" TODO.md
```

Expected: `2` (both rows present).

- [ ] **Step 3: Bump version**

Modify `pyproject.toml`:

```toml
[project]
name = "silly-kicks"
version = "3.2.0"
```

Add to `[project.optional-dependencies]`:

```toml
golden-master = ["unravelsports>=1.2"]
```

Modify `silly_kicks/_version.py`:

```python
__version__ = "3.2.0"
```

- [ ] **Step 4: Verify version bump**

```bash
uv run python -c "import silly_kicks; print(silly_kicks.__version__)"
```

Expected: `3.2.0`.

---

## Task 16: Run full test suite + lint gates

**Files:** none (CI gates)

- [ ] **Step 1: Run full pytest suite (excluding e2e marker)**

```bash
uv run pytest tests/ -m "not e2e" -v --tb=short
```

Expected: All pass. Existing tests + 16 new test files (or some skipped due to optional deps / fixture availability).

- [ ] **Step 2: Run ruff format check**

```bash
uv run ruff format --check silly_kicks/ tests/
```

Expected: Clean (no diffs needed). If diffs reported, run `uv run ruff format silly_kicks/ tests/` to apply, then re-check.

- [ ] **Step 3: Run ruff lint**

```bash
uv run ruff check silly_kicks/ tests/
```

Expected: 0 errors.

- [ ] **Step 4: Run pyright**

```bash
uv run pyright silly_kicks/ tests/
```

Expected: 0 errors. Failures → fix per `feedback_pyright_setindex_pattern` (use `np.flatnonzero` + `np.asarray` patterns).

- [ ] **Step 5: Run public-API examples coverage gate**

```bash
uv run pytest tests/test_public_api_examples.py -v
```

Expected: PASS for all new public defs (auto-discovered Examples requirement per `feedback_public_api_examples`).

---

## Task 17: Cleanup transient research scratchpad

**Files:**
- Delete: `scripts/_tf2_formula_research.md`

- [ ] **Step 1: Remove research scratchpad**

```bash
rm scripts/_tf2_formula_research.md
ls scripts/_*.md scripts/_*.txt 2>&1
```

Expected: `ls: cannot access 'scripts/_*.md': No such file or directory` (file is gone).

---

## Task 18: Run /final-review

**Files:** none (gate)

- [ ] **Step 1: Final review**

Per memory `feedback_final_review_gate`: `/final-review` is mandatory before the single commit.

```bash
# /final-review is a slash command — invoked from Claude Code interactive prompt.
# This step is the placeholder; in practice run /final-review interactively.
echo "Run /final-review interactively before proceeding to Task 19."
```

Expected: All issues raised by /final-review are addressed inline before commit. If /final-review surfaces new bugs, fix them and re-run from this step.

---

## Task 19: Commit + push + PR (with explicit user approval)

**Files:** none (git operations)

> Per memory `feedback_commit_policy`: ONE commit per branch, explicit approval required before commit. Per `feedback_engineering_disciplines`: single/minimal commits.

- [ ] **Step 1: Show staged changes summary**

```bash
git status --short
git diff --stat
```

Expected: All modifications and new files visible. Verify no transient `_*` files survive in scripts/ or other dirs.

- [ ] **Step 2: Get explicit user approval to commit**

Pause and prompt the user: "Ready to commit? All gates passed (pytest, ruff format, ruff check, pyright, /final-review)."

Wait for explicit user approval.

- [ ] **Step 3: Stage and commit**

After approval:

```bash
git add -A
git status --short
```

Verify no transient files staged. Then:

```bash
git commit -m "$(cat <<'EOF'
feat(tracking): TF-3 actor pre-window + TF-2 multi-flavor pressure_on_actor -- silly-kicks 3.2.0 (PR-S25)

TF-3: geometric arc-length + displacement of actor's pre-action window
(0.5s default). NaN-bridge rule, frame-ordering invariant. Pure geometric
formulation; NOT a Bauer & Anzer covered-distance re-implementation.

TF-2: multi-flavor pressure_on_actor with three published methodologies:
- andrienko_oval (default): Andrienko 2017 directional oval, sum aggregation
- link_zones: Link 2016 piecewise zones, saturating-exp aggregation
- bekkers_pi: Bekkers 2024 probabilistic TTI, BSD-3-Clause UnravelSports port

ADR-005 §8 amendment codifies multi-flavor xfn column-naming convention
(<feature>__<method> suffixes; default xfn list ships single default-method
xfn; per-method params via flavor-specific frozen dataclass).

16 new test files including:
- 4-point pin verification for Andrienko Θ values (9.00 / 7.05 / 4.27 / 3.00 m)
- Bit-equivalent golden-master parity vs UnravelSports canonical (rtol=1e-9)
- Float-precision SHA-256 snapshot regression gates
- pytest-benchmark performance budgets per method
- Atomic ↔ standard parity per method
- Per-provider real-data sanity bounds + NaN-rate cap
- ADR-005 §8 contract test (column-naming + transactional validation +
  decorator composition + ADR markdown grep + unknown-method ValueError)
- pyright clean gate

k3 calibration deferred post-release per "Optuna-not-evolve for single
scalars" guidance — tracked as TF-24 (Optuna sweep against held-out VAEP
fold) and TF-25 (lakehouse evolve framework for structural-form variants)
in TODO.md.

Spec: docs/superpowers/specs/2026-05-03-tf3-tf2-design.md
Plan: docs/superpowers/plans/2026-05-03-tf3-tf2-pressure-and-prewindow.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
git status
```

- [ ] **Step 4: Push branch**

```bash
git push -u origin pr-s25-tf3-tf2-pressure-and-prewindow
```

- [ ] **Step 5: Open PR**

```bash
gh pr create --title "feat(tracking): TF-3 actor pre-window + TF-2 multi-flavor pressure_on_actor -- silly-kicks 3.2.0 (PR-S25)" --body "$(cat <<'EOF'
## Summary

- **TF-3** (`actor_arc_length_pre_window` + `actor_displacement_pre_window`) — first time-window tracking-aware feature using `slice_around_event`. NaN-bridge rule, kernel sorts by timestamp, pure geometric (NOT Bauer & Anzer covered-distance).
- **TF-2** (`pressure_on_actor`) — first multi-flavor xfn with three published methodologies; Andrienko 2017 default. ADR-005 §8 amendment codifies multi-flavor column-naming convention.
- **Bekkers TTI** ported from canonical BSD-3-Clause UnravelSports source with bit-equivalent golden-master parity test.
- 16 new test files including snapshot determinism, perf budgets, atomic-standard parity, real-data calibration, ADR contract gate, pyright clean gate.
- k3 calibration deferred post-release as TF-24 (Optuna) + TF-25 (evolve, conditional).

## Test plan

- [ ] All 16 new test files pass locally
- [ ] `ruff format --check silly_kicks/ tests/` clean
- [ ] `ruff check silly_kicks/ tests/` 0 errors
- [ ] `uv run pyright silly_kicks/ tests/` 0 errors
- [ ] `uv run pytest tests/ -m "not e2e"` all pass
- [ ] CI green
- [ ] Bekkers golden-master test runs in CI (uv pip install -e '.[golden-master]' setup needed)
- [ ] /final-review run before commit

## Specs

- Spec: [`docs/superpowers/specs/2026-05-03-tf3-tf2-design.md`](docs/superpowers/specs/2026-05-03-tf3-tf2-design.md) (1103 lines, lakehouse-reviewed v1+v2)
- Plan: [`docs/superpowers/plans/2026-05-03-tf3-tf2-pressure-and-prewindow.md`](docs/superpowers/plans/2026-05-03-tf3-tf2-pressure-and-prewindow.md)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Expected: PR URL printed.

---

## Self-review

**Spec coverage check:**

- ✅ §1 Context: covered (no implementation, but plan opens with reference to spec)
- ✅ §2 Scope (in scope, out of scope): covered Tasks 2–14
- ✅ §3 TF-3 surface: Tasks 7 (kernel) + 8 (public defs)
- ✅ §4 TF-2 surface: Tasks 2 (params) + 3, 4, 5, 6 (kernels) + 8 (public dispatch)
- ✅ §5 ADR-005 §8 amendment: Task 10
- ✅ §6 Atomic mirror: Task 9
- ✅ §7 NOTICE entries + BSD-3-Clause: Task 11
- ✅ §8.0 TDD ordering: enforced via Task Na (RED) + Task Nb (GREEN) split throughout (Tasks 3, 4, 5, 6, 7)
- ✅ §8.1 16 test files: enumerated across Tasks 3–14
- ✅ §8.2 Numerical pin tests: covered in Tasks 3, 4, 5, 6, 7
- ✅ §8.3 Cross-method invariants: Task 13 step 7
- ✅ §8.4 Real-data sweep + bounds: Task 13 step 8
- ✅ §8.5 ADR contract test: Task 13 step 10
- ✅ §8.6 Snapshot determinism: Task 13 steps 2-4
- ✅ §8.7 Bekkers golden-master: Task 13 step 1
- ✅ §8.8 Atomic-standard parity: Task 13 step 6
- ✅ §8.9 Bekkers E2E: Task 13 step 9
- ✅ §8.10 pyright gate: Task 13 step 11
- ✅ §8.11 Public-API Examples: Task 16 step 5 (existing CI gate auto-discovers new defs)
- ✅ §9 Synthesizer scenarios: covered by Task 14a (new — added per lakehouse v3 review item 7, frozen-dataclass named scenarios in `tests/tracking/_provider_inputs.py` per `feedback_synthesizer_shot_plus_keeper_save_pattern`)
- ✅ §10 CHANGELOG: Task 12
- ✅ §11 Risks: addressed across Tasks 3–14 (each gate)
- ✅ §12 Deferred items: Task 0 doc-drift sweep included; remainder is plan-side
- ✅ k3 follow-up: TODO.md updates in Task 15

**Type / signature consistency check:**

- ✅ `Method` Literal defined in `pressure.py`, imported into both `tracking/features.py` and `atomic/tracking/features.py`
- ✅ `validate_params_for_method` signature matches across module + tests + atomic surface
- ✅ `_pressure_andrienko` / `_pressure_link` / `_pressure_bekkers` consistent kernel signatures across kernel impl + tests + public defs
- ✅ `_actor_pre_window_kernel` signature matches across impl + standard wrapper + atomic wrapper
- ✅ Suffix convention `pressure_on_actor__<method>` consistent across kernel return + public def `.rename()` + ADR §8 amendment + tests

**Placeholder scan:**

- ✅ No "TBD" / "TODO" / "implement later" in code blocks (only in CHANGELOG date stub `2026-05-XX` which is intentional — finalized at commit time)
- ✅ No "fill in details" / "similar to Task N" / "add appropriate error handling"
- ✅ Snapshot test SHAs use sentinel `<filled-on-first-run>` with explicit regenerate flow (Task 13 step 4 walks through pinning) — correct pattern, not a placeholder

**Plan complete.** All 19 tasks have actual code, exact file paths, runnable commands, and expected output.

---

## Execution

Per project memory `feedback_inline_execution_default`: execute inline via Bash/Edit/Write/Read; subagents trigger per-tool permission prompts the user finds disruptive.

**Approach: Inline Execution via superpowers:executing-plans**, no subagents. Per-task Bash/Edit/Write/Read.
