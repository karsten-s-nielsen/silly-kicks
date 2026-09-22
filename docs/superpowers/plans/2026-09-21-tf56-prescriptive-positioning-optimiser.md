# TF-56 Prescriptive defensive-positioning optimiser — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `silly_kicks/positioning/` — a pure simulated-annealing solver for the best *reachable* defensive shape + a measured frame/team `positioning_gap` metric (threat the realized shape failed to suppress vs the reachable optimum).

**Architecture:** A tracking-consuming sibling package (gkdv/territorial_defense shape). Objective-agnostic SA solver (`optimise_positions`) behind composable `Objective`/`Constraint`/`Optimizer` protocols; the shipped metric column freezes to one canonical `ThreatObjective` (via `compute_threat_pc`). Pure; `xt` injected; no bundled weights.

**Tech Stack:** Python, numpy, pandas; reuses `silly_kicks.tracking` public seams (`compute_threat_pc`, `compute_tti`, `get_das`, `resolve_defended_goals`) + `id_compat`/`reflection`. SA is hand-rolled numpy (no new dep).

**Spec:** `docs/superpowers/specs/2026-09-21-tf56-prescriptive-positioning-optimiser-design.md` — executors read both.

## Global Constraints

- **Commit discipline OVERRIDES the writing-plans per-task "Commit" template.** NO per-task commits, NO micro-commits. Work accumulates on ONE feature branch — **`feat/tf56-positioning`**, off `main` (branch name carries no version; safe to fix now). Exactly **two** commits, each a fully-tested coherent state behind an explicit **human-approval gate** (Task 14 = Commit-1, Task 16 = Commit-2). Never `git commit`/`push`/merge/tag without explicit per-action approval. The per-task "Verify" steps below run tests but DO NOT commit.
- **No pre-claimed numbers.** Version / `PR-Snnn` / ADR numbers are next-free, resolved at commit-prep against `main`; no task writes a specific number into code/docs (TF-63 triple-collision lesson). The ADR filename uses `ADR-NNN` until commit-prep.
- **Hexagonal boundaries.** `positioning` imports `silly_kicks.tracking` PUBLIC seams + `silly_kicks.id_compat` / `silly_kicks.reflection` ONLY — never a `tracking._*` private. `xt` is INJECTED (never `import silly_kicks.xthreat` for weights). Nothing imports `positioning`.
- **A `compute_*`, not an `add_*`** → action-coupled aggregator count stays **33**. In no default xfn list; no `*_xfns`.
- **Frozen params, ADR-009.** `PositioningParams`/`ReachabilityParams`/`SAParams` frozen dataclasses, `for_provider` empty.
- **Determinism.** `positioning_gap` is a pure function of inputs — SA RNG seeded from `(game_id, period_id, frame_id)`. A stochastic column is unacceptable.
- **Reachability defaults:** `max_reach_seconds=0.7`, `reaction_time=0.1`, `max_acceleration=SpearmanParams.max_acceleration` (7.0).
- **Velocity REQUIRED** (Tier-3, ADR-063): declared velocity-less → excluded-and-counted; undeclared-missing `vx`/`vy` → RAISE.
- **`compute_threat_pc` computes the surface directly, NEVER `PitchControlCache`** (ADR-043 identity-key landmine).
- **NO bundled weights** (pure solver). Commit-2 is an owner-run construct-validity REPORT with a clean `training_commit`.

---

## Task 0: Package scaffold + frozen config + import-allowlist

**Files:**
- Create: `silly_kicks/positioning/__init__.py`, `silly_kicks/positioning/_config.py`
- Test: `tests/positioning/__init__.py` (empty; metric-test-dir convention), `tests/positioning/test_config.py`, `tests/positioning/test_import_allowlist.py`

**Interfaces:**
- Produces: `PositioningParams`, `ReachabilityParams`, `SAParams` (frozen; `.default()` / `.for_provider(p)` returning the same frozen default, ADR-009).

- [ ] **Step 1: Write failing test** (`tests/positioning/test_config.py`)

```python
import dataclasses
from silly_kicks.positioning import PositioningParams, ReachabilityParams, SAParams

def test_params_are_frozen_with_documented_defaults():
    p = PositioningParams.default()
    assert dataclasses.is_dataclass(p) and p.__dataclass_params__.frozen
    assert p.domain_ball_to_goal_m > 0 and p.sample_fps == 1.0
    r = ReachabilityParams.default()
    assert (r.max_reach_seconds, r.reaction_time) == (0.7, 0.1)
    assert r.max_acceleration == 7.0  # SpearmanParams.max_acceleration
    s = SAParams.default()
    assert (s.num_iterations, s.patience) == (2000, 200)

def test_for_provider_is_empty_adr009():
    # No per-provider tuning ships; for_provider returns the frozen default for any provider.
    assert PositioningParams.for_provider("skillcorner") == PositioningParams.default()
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError`). Run: `python -m pytest tests/positioning/test_config.py -v`

- [ ] **Step 3: Implement** `_config.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from silly_kicks.tracking.pitch_control import SpearmanParams  # public seam; for max_acceleration=7.0

@dataclass(frozen=True)
class ReachabilityParams:
    max_reach_seconds: float = 0.7
    reaction_time: float = 0.1
    max_acceleration: float = SpearmanParams().max_acceleration  # 7.0
    @classmethod
    def default(cls) -> "ReachabilityParams": return cls()
    @classmethod
    def for_provider(cls, provider: str) -> "ReachabilityParams": return cls()  # ADR-009: no per-provider tune

@dataclass(frozen=True)
class SAParams:
    num_iterations: int = 2000
    patience: int = 200
    init_sigma_m: float = 2.0     # initial Gaussian perturbation stddev — matched to the ~2 m reachable radius; spike-verified SEED-INVARIANT optimum (§8 amendment). 5.0 left the optimum seed-noisy (gap std 0.324 vs 0.000).
    cooling: float = 0.995        # geometric temperature/sigma decay per iter
    @classmethod
    def default(cls) -> "SAParams": return cls()
    @classmethod
    def for_provider(cls, provider: str) -> "SAParams": return cls()

@dataclass(frozen=True)
class PositioningParams:
    domain_ball_to_goal_m: float = 35.0   # matches gkdv's _DOMAIN_BALL_TO_GOAL_M (gkdv/_engine.py); verify unchanged at impl
    sample_fps: float = 1.0
    reachability: ReachabilityParams = ReachabilityParams()
    sa: SAParams = SAParams()
    @classmethod
    def default(cls) -> "PositioningParams": return cls()
    @classmethod
    def for_provider(cls, provider: str) -> "PositioningParams": return cls()
```

> **Implementation note:** `35.0` is gkdv's `_DOMAIN_BALL_TO_GOAL_M` (verified `gkdv/_engine.py`); re-`grep -rn "_DOMAIN_BALL_TO_GOAL_M" silly_kicks/gkdv/` at impl to confirm it has not changed, then keep it verbatim (spec §6).

- [ ] **Step 4: Implement** `__init__.py` (surface grows per task; start with config)

```python
from ._config import PositioningParams, ReachabilityParams, SAParams
__all__ = ["PositioningParams", "ReachabilityParams", "SAParams"]
```

- [ ] **Step 5: Write the import-allowlist gate** (`tests/positioning/test_import_allowlist.py`) — AST over `silly_kicks/positioning/**`: every module-level import is in `{silly_kicks.tracking, silly_kicks.tracking.*(public), silly_kicks.id_compat, silly_kicks.reflection, silly_kicks.spadl.config, numpy, pandas, dataclasses, ...stdlib}`; assert NO `silly_kicks.tracking._*` private and NO `silly_kicks.xthreat`. Reverse direction: AST over the rest of `silly_kicks/` asserts nothing imports `silly_kicks.positioning`. (Copy the shape of `tests/gkdv/test_import_allowlist.py` / `tests/territorial_defense/test_import_allowlist.py`.)

- [ ] **Step 5b: Create shared test fixtures** (`tests/positioning/conftest.py`) — **REVIEW-FIX (plan-r1 #08):** no later task owned the frame fixtures the tests reference (`one_frame`, `two_team_velocity_frame`, `movable_ids`, `mixed_domain_frames`, `actions_frames_fixture`), and `one_frame` is shared across Tasks 1/5/6/8 so it MUST be one canonical source (ad-hoc per-task frames risk cross-task inconsistency). `fitted_xt` is FREE from the root `tests/conftest.py:44` — do NOT redefine it. Build these by REUSING an existing tracking frame builder (start from the frame in `tests/tracking/test_compute_threat_pc.py` / the `tests/tracking` conftest helpers), not hand-rolled columns:

```python
import pandas as pd, pytest
# Reuse the tracking test frame builder; do NOT hand-roll TRACKING_FRAMES_COLUMNS.

@pytest.fixture
def one_frame() -> pd.DataFrame:
    """One tracking frame: defending + attacking team + ball row; vx/vy + is_goalkeeper present;
    team_in_possession = attacking; built so repositioning a defender measurably lowers
    compute_threat_pc (a real, non-degenerate gap exists). Reuses the tracking test frame."""
    ...  # build via the tracking helper; assert two teams + a ball row + finite vx/vy

@pytest.fixture
def two_team_velocity_frame(one_frame): return one_frame

@pytest.fixture
def movable_ids(one_frame) -> list:
    """The defending outfielders' player_ids in one_frame (GK excluded)."""
    ...

@pytest.fixture
def mixed_domain_frames() -> pd.DataFrame:
    """Concat spanning: in-domain, ball-far (out_of_domain), one-team, velocity-less (declared marker),
    unresolved-goal, no-movable — for the Task 8 conservation census."""
    ...

@pytest.fixture
def actions_frames_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(actions, frames) reused for any pressure regression; reuse tests/tracking pressure fixtures."""
    ...
```

Each `...` is filled at impl by reusing the named tracking test frame builder + asserting the stated properties (a real gap; declared velocity marker on the velocity-less slice) — not fresh column construction.

- [ ] **Step 6: Run — expect PASS.** `python -m pytest tests/positioning/ -v`

**Verify:** config tests + allowlist green; conftest fixtures import + build. (No commit — Global Constraints.)

---

## Task 1: `Objective` protocol + `ThreatObjective`

**Files:**
- Create: `silly_kicks/positioning/_objectives.py`
- Test: `tests/positioning/test_objectives.py`

**Interfaces:**
- Consumes: `compute_threat_pc(frame, *, attacking_team_id, xt, goal_map, method="spearman", params=None, field_weight=None) -> float` (`silly_kicks.tracking`); `GoalMap`.
- Produces: `Objective` (Protocol, `score(frame) -> float`), `ThreatObjective(*, xt, goal_map, attacking_team_id, params=None)`.

- [ ] **Step 1: Write failing test**

```python
import numpy as np, pandas as pd, pytest
from silly_kicks.positioning import ThreatObjective
from silly_kicks.tracking import resolve_defended_goals

def test_threat_objective_is_frame_to_float_and_moving_a_defender_changes_it(two_team_velocity_frame, fitted_xt):
    frames = two_team_velocity_frame                 # fixture: one frame, two teams, ball, vx/vy present
    gmap = resolve_defended_goals(frames)
    obj = ThreatObjective(xt=fitted_xt, goal_map=gmap, attacking_team_id=<opp id>)
    s0 = obj.score(frames)
    moved = frames.copy()
    # shove one defender far off the ball line
    di = moved.index[<a defender row>]
    moved.loc[di, "x"] += 20.0
    s1 = obj.score(moved)
    assert isinstance(s0, float)
    assert s1 != s0, "moving a defender must change the threat surface (PitchControlCache NOT served)"
```

- [ ] **Step 2: Run — expect FAIL** (import error).

- [ ] **Step 3: Implement** `_objectives.py` (protocol + ThreatObjective)

```python
from __future__ import annotations
from typing import Protocol, runtime_checkable
import pandas as pd
from silly_kicks.tracking import compute_threat_pc  # public seam

@runtime_checkable
class Objective(Protocol):
    def score(self, frame: pd.DataFrame) -> float: ...   # lower = defensively safer

class ThreatObjective:
    """xT-weighted pitch-control threat the DEFENCE concedes (compute_threat_pc, spearman, GK-aware)."""
    def __init__(self, *, xt, goal_map, attacking_team_id, params=None):
        self._xt, self._gmap, self._att, self._params = xt, goal_map, attacking_team_id, params
    def score(self, frame: pd.DataFrame) -> float:
        # method="spearman" (GK-aware via lambda_gk); computes the surface DIRECTLY (no PitchControlCache).
        return float(compute_threat_pc(frame, attacking_team_id=self._att, xt=self._xt,
                                       goal_map=self._gmap, method="spearman", params=self._params))
```

- [ ] **Step 4: Run — expect PASS.** Add `ThreatObjective`, `Objective` to `__init__` `__all__`.

**Verify:** the non-vacuity test proves the cache is not served (a moved defender moves the score).

---

## Task 2: `WeightedSum` + `CappedContribution` composables

**Files:** Modify `silly_kicks/positioning/_objectives.py`; Test: `tests/positioning/test_objectives.py`

**Interfaces:** Produces `WeightedSum(list[tuple[Objective, float]])`, `CappedContribution(objective, *, cap)` — both `Objective`s.

- [ ] **Step 1: Write failing test**

```python
from silly_kicks.positioning import WeightedSum, CappedContribution, Objective

class _Const:
    def __init__(self, v): self.v = v
    def score(self, frame): return self.v

def test_weighted_sum_is_objective_and_combines():
    ws = WeightedSum([(_Const(2.0), 1.0), (_Const(4.0), 0.5)])
    assert isinstance(ws, Objective)
    assert ws.score(None) == 2.0 * 1.0 + 4.0 * 0.5

def test_capped_contribution_caps_a_single_agents_marginal_score():
    # cap limits how much any one agent can push the score (the "abandons a man" fix).
    capped = CappedContribution(_Const(100.0), cap=1.0)
    assert isinstance(capped, Objective)
    assert capped.score(None) <= <documented cap semantics>   # see impl docstring
```

- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** `WeightedSum` (sum of `w*obj.score(frame)`) + `CappedContribution` (wraps an objective; caps the per-agent marginal contribution — the implementation computes each movable agent's leave-one-out marginal and clips it at `cap`; the docstring states the exact semantics + why it fixes the aggregate-averaging artifact). — [ ] **Step 4: PASS.** Add both to `__all__`.

**Verify:** protocol conformance (`isinstance(..., Objective)`) + composition math + the cap unit-test.

---

## Task 3: `pressure_on_target` frame-level primitive + `PressureObjective`

**REVIEW-FIX (plan-r1 #07):** the earlier plan proposed a carrier-only `pressure_on_carrier(frame)` primitive that `add_pressure_on_actor` (per-ACTOR, vectorized) would delegate to under a byte-identical golden — incompatible (a per-actor aggregator can't delegate to a carrier-only, target-less fn) and refactor-risky. Fixed two ways: the primitive takes a **target player_id**, and it **REUSES the existing `pressure_on_actor` computer via a synthesized single-action call** rather than refactoring the vectorized aggregator — so `add_pressure_on_actor` is UNTOUCHED (no parity golden needed; the existing `tests/tracking/test_pressure_*.py` remain the guard) and the pressure math is not duplicated.

**Files:**
- Create: `silly_kicks/tracking/_pressure_target.py` (or add to `tracking/pressure.py`) — `pressure_on_target`; re-export from `tracking/__init__` public surface.
- Modify: `silly_kicks/positioning/_objectives.py` (`PressureObjective`).
- Test: `tests/positioning/test_objectives.py`, `tests/tracking/test_pressure_target.py`.

**Interfaces:**
- Consumes: `pressure_on_actor(actions, frames, *, method, params, links=None) -> pd.Series` (the single-method per-actor computer that `add_pressure_on_actor` dispatches to, `features.py:1060`; a per-action Series → the 1-row synthesized call yields a 1-element Series → `.iloc[0]` scalar).
- Produces (tracking, public): `pressure_on_target(frame, player_id, *, method="bekkers_pi", params=None) -> float` — pressing intensity on `player_id` in `frame`, computed by synthesizing a one-row `actions` frame for that player at the frame's `(game_id, period_id, frame_id)` + calling `pressure_on_actor` on the `{frame}` set and extracting the scalar (REUSE, not re-implementation).
- Produces (positioning): `PressureObjective(*, method="bekkers_pi", params=None)` — `score(frame)` locates the ball-carrier (in-possession actor) via the frame's possession flag / nearest-to-ball, calls `pressure_on_target` on it. **Sign:** higher pressure = defensively better, so `score` returns `-pressure` (documented) → `WeightedSum` with `ThreatObjective` composes under the shared "lower = better" contract.

- [ ] **Step 1: Write failing tests**

```python
def test_pressure_on_target_reuses_pressure_on_actor_scalar(one_frame):
    from silly_kicks.tracking import pressure_on_target
    pid = <a player id under pressure in one_frame>
    p = pressure_on_target(one_frame, pid, method="bekkers_pi")
    assert isinstance(p, float) and p >= 0.0

def test_pressure_objective_is_lower_is_better_frame_to_float(one_frame):
    from silly_kicks.positioning import PressureObjective, Objective
    obj = PressureObjective(method="bekkers_pi")
    assert isinstance(obj, Objective)
    s = obj.score(one_frame)
    assert isinstance(s, float)   # == -pressure_on_target(carrier); higher pressure -> lower (safer) score
```

- [ ] **Step 2: Run — expect FAIL** (import errors).

- [ ] **Step 3: Implement** `pressure_on_target` (synthesize the one-row actions frame for `player_id`; call `pressure_on_actor(synth_actions, frame_set, method=, params=)`; return the `pressure_on_actor__<method>` scalar). Implement `PressureObjective.score` (carrier lookup + `-pressure_on_target`). `add_pressure_on_actor` is NOT modified.

- [ ] **Step 4: Run — expect PASS.** Add `pressure_on_target` to `tracking.__all__`; `PressureObjective` to `positioning.__all__`.

- [ ] **Step 5: Regression** — run the EXISTING `tests/tracking/test_pressure_*.py` (must stay green; `add_pressure_on_actor`/`pressure_on_actor` unchanged, so no new golden is introduced).

**Verify:** `PressureObjective` is a `frame -> float` `Objective`; pressure math REUSED (not duplicated, not refactored); `add_pressure_on_actor` byte-identical by being untouched. 33-aggregator count unaffected (no new `add_*`; `pressure_on_target` is a `compute_*`-style primitive).

---

## Task 4: `DasObjective` adapter

**Files:** Modify `_objectives.py`; Test: `tests/positioning/test_objectives.py`

**Interfaces:** Consumes `get_das(frames) -> DataFrame` (team-level AS/DAS). Produces `DasObjective()` — `score(frame) -> float` = the defending team's conceded-DAS scalar for the single frame.

- [ ] **Step 1: Write failing test** — `DasObjective().score(one_frame)` returns a float = the attacking team's DAS (the space the defence conceded) reduced from a single-frame `get_das` call.
- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** — wrap `get_das` on a one-frame DataFrame, extract the DAS scalar (attacking team's dangerous accessible space). — [ ] **Step 4: PASS.** Add to `__all__`.

**Verify:** frame->float; velocity-dependent (documented — DasObjective is exploratory, not the shipped column; velocity gating handled at the compute layer, Task 8).

---

## Task 5: `Constraint` protocol + `ReachabilityConstraint`

**Files:** Create `silly_kicks/positioning/_constraints.py`; Test: `tests/positioning/test_constraints.py`

**Interfaces:** Consumes `compute_tti(pos, vel, targets, reaction_time, max_acceleration) -> ndarray` (`silly_kicks.tracking.pitch_control`). Produces `Constraint` (Protocol, `is_feasible(player_id, candidate_xy, frame) -> bool`), `ReachabilityConstraint(params: ReachabilityParams)`.

- [ ] **Step 1: Write the RED-GREEN failing test (both sides of the horizon)**

```python
from silly_kicks.positioning import ReachabilityConstraint, ReachabilityParams

def test_reachability_rejects_beyond_horizon_accepts_within(one_frame):
    c = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=0.7))
    pid = <a player id in one_frame>
    near = <candidate ~1 m from the player>
    far  = <candidate ~40 m from the player>
    assert c.is_feasible(pid, near, one_frame) is True
    assert c.is_feasible(pid, far, one_frame) is False   # unreachable in 0.7 s from rest+velocity

def test_tighter_horizon_shrinks_feasible_set(one_frame):
    pid, cand = <player>, <candidate ~8 m away>
    loose = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=1.5))
    tight = ReachabilityConstraint(ReachabilityParams(max_reach_seconds=0.3))
    assert loose.is_feasible(pid, cand, one_frame) and not tight.is_feasible(pid, cand, one_frame)
```

- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** — `is_feasible` reads the player's REAL `(x,y)` + `(vx,vy)` from `frame` (id via `id_compat`), calls `compute_tti(pos, vel, [candidate], reaction_time, max_acceleration)`, returns `tti <= max_reach_seconds`. — [ ] **Step 4: PASS.** Add to `__all__`.

**Verify:** both sides of the horizon; feasibility is vs the REAL player state (not a mid-search intermediate).

---

## Task 6: `Optimizer` protocol + `SimulatedAnnealing` + `OptimizeResult`

**Files:** Create `silly_kicks/positioning/_optimizer.py`; Test: `tests/positioning/test_optimizer.py`

**Interfaces:** Produces `OptimizeResult(best_frame, best_score, actual_score, n_iter, n_feasible_proposals, converged)`; `Optimizer` (Protocol, `optimize(frame, *, movable, objective, constraints, rng) -> OptimizeResult`); `SimulatedAnnealing(*, params: SAParams=SAParams.default())`.

- [ ] **Step 1: Write failing tests**

```python
import numpy as np
from silly_kicks.positioning import SimulatedAnnealing, SAParams

def test_actual_score_is_incumbent0_and_best_le_actual(one_frame, const_objective, movable_ids):
    sa = SimulatedAnnealing(SAParams(num_iterations=200, patience=50))
    res = sa.optimize(one_frame, movable=movable_ids, objective=<threat obj>,
                      constraints=[<reachability>], rng=np.random.default_rng(0))
    assert res.actual_score == <threat obj>.score(one_frame)   # incumbent-0, same path
    assert res.best_score <= res.actual_score                   # gap >= 0 by construction

def test_determinism_same_seed_same_result(one_frame, movable_ids):
    a = SimulatedAnnealing().optimize(one_frame, movable=movable_ids, objective=<obj>,
                                      constraints=[<c>], rng=np.random.default_rng(42))
    b = SimulatedAnnealing().optimize(one_frame, movable=movable_ids, objective=<obj>,
                                      constraints=[<c>], rng=np.random.default_rng(42))
    assert a.best_score == b.best_score and a.best_frame.equals(b.best_frame)
```

- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** SA:
  - `actual_score = objective.score(frame)` ONCE as the incumbent-0 (store on the result).
  - Loop `num_iterations`: pick a random movable player, propose `candidate = pos + rng.normal(0, sigma, 2)`; reject if any constraint `is_feasible` is False (count feasible proposals); else score the modified frame; Metropolis accept `exp(-(new-cur)/T)`; decay `T *= cooling`, `sigma *= cooling`; track `best`; patience early-stop when no `best` improvement for `patience` iters (`converged=True` if it early-stopped, else False).
  - Return `OptimizeResult(best_frame, best_score, actual_score, n_iter, n_feasible_proposals, converged)`. NEVER mutate the input `frame` (operate on a copy).
- [ ] **Step 4: PASS.** Add `SimulatedAnnealing`, `OptimizeResult` (and `Optimizer`) to `__all__`.

**Verify:** `best_score <= actual_score` (path identity), determinism under a fixed rng.

---

## Task 7: `optimise_positions` (pure solver entry)

**Files:** Create `silly_kicks/positioning/_solve.py`; Test: `tests/positioning/test_solve.py`

**Interfaces:** Produces `optimise_positions(frame, *, movable, objective, constraints, optimizer=None, seed=None) -> OptimizeResult`.

- [ ] **Step 1: Write failing tests** — (a) purity: `frame` unchanged after the call (snapshot equality); (b) default optimizer = `SimulatedAnnealing()`; (c) deterministic when `seed` supplied; (d) `gap = actual_score - best_score >= 0`.
- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** — validate `movable` non-empty; default `optimizer=SimulatedAnnealing()`; build `rng=np.random.default_rng(seed)`; return `optimizer.optimize(frame.copy(), movable=movable, objective=objective, constraints=constraints, rng=rng)`. — [ ] **Step 4: PASS.** Add to `__all__`.

**Verify:** purity (ADR-033), determinism, gap≥0.

---

## Task 8: `compute_positioning_gap` (domain gate + orientation + census)

**Files:** Create `silly_kicks/positioning/_compute.py`, `silly_kicks/positioning/_report.py`; Test: `tests/positioning/test_compute.py`

**Interfaces:**
- Consumes: `resolve_defended_goals(frames) -> GoalMap`, `GoalEndUnresolvedError` (ADR-055); `id_compat` (`canonical_id_series`, `ids_match`); `optimise_positions`; `ThreatObjective`; `ReachabilityConstraint`.
- Produces: `compute_positioning_gap(frames, *, xt, movable=None, objective=None, params=PositioningParams.default()) -> (samples, PositioningReport)`; `PositioningReport` (frozen; fields `n_frames_in`, `n_frames_scored`, per-reason drop counts) with a conservation invariant method.

- [ ] **Step 1: Write failing tests** (mixed fixture):
  - `test_conserves` — `report.n_frames_scored + sum(report.drops.values()) == report.n_frames_in` on a fixture mixing: in-domain, out-of-domain (ball far from goal), non-two-team, velocity-less (declared), unresolved-goal, no-movable.
  - `test_source_tokens` — every samples row's `positioning_gap_source` ∈ the closed set; a non-two-team frame → `excluded_not_two_teams` (NOT `excluded_out_of_domain`).
  - `test_gap_ge_zero_and_honest_nan` — scored rows have `positioning_gap >= 0`; every excluded/degenerate row has `positioning_gap` NaN (never 0).
  - `test_undeclared_missing_velocity_raises` — frames missing `vx`/`vy` WITHOUT the velocity-unavailable marker → raises (ADR-063); declared-unavailable → `velocity_unavailable` counted.
  - `test_orientation_goal_map_not_team_identity` — an away-defending / mirrored-frame fixture scores coherently (uses `goal_map`, not team id).
  - `test_determinism` — same frames → byte-identical `positioning_gap` column (seed from `(game_id, period_id, frame_id)`).
- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** the data flow (spec §6): build `goal_map` ONCE; per-condition domain gate assigning the specific `positioning_gap_source` token; velocity check (declared→count, undeclared-missing→raise); 1-fps downsample within domain; for each scored frame: derive `attacking_team_id` (opponent) from `goal_map`, run `optimise_positions(frame, movable=<defending outfielders>, objective=ThreatObjective(...), constraints=[ReachabilityConstraint(...)], seed=hash of (game,period,frame))`, set `threat_actual=res.actual_score`, `threat_optimum=res.best_score`, `positioning_gap=threat_actual-threat_optimum`; catch `GoalEndUnresolvedError` at the edge → `unresolved_geometry` NaN row; canonical-group team id, emit raw. Build `PositioningReport`. — [ ] **Step 4: PASS.**

**Verify:** conservation, census tokens, honest-NaN, velocity tiering, orientation, determinism.

---

## Task 9: `summarize_positioning_gap`

**Files:** Modify `_compute.py`; Test: `tests/positioning/test_compute.py`

**Interfaces:** Produces `summarize_positioning_gap(samples) -> DataFrame` (per `(game_id, team_id)`: `mean_positioning_gap`, `n_scored`, drop counts).

- [ ] **Step 1: Write failing test** — grain `(game_id, team_id)`; `mean_positioning_gap` = mean over `scored` rows only; `n_scored` matches. — [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** (group on canonical id, emit raw). — [ ] **Step 4: PASS.** Add `compute_positioning_gap`, `summarize_positioning_gap`, `PositioningReport` to `__all__`.

**Verify:** grain + mean-over-scored-only.

---

## Task 10: `_probe.py` — validation battery machinery (fixture-tested)

**Files:** Create `silly_kicks/positioning/_probe.py`; Test: `tests/positioning/test_probe.py`

> **AMENDED 2026-09-22 (owner-ratified; tracks the §8 spec amendment).** The dose/responsiveness machinery below is RETIRED — a commit-1 spike MEASURED `positioning_gap` non-monotonic in any dose (it re-anchors to each defender's reachable set) and orthogonal to concurrent shape-badness, so "dose → gap rises" is invalid as an instrument. The battery is now **optimizer-stability + discrimination** (fixture instrument-validity) and **predictive** `corr(gap_t, conceded_threat_{t+Δ})` (corpus construct-validity). The dose is retired as a GATE; it is NOT re-added even as a reported diagnostic (the spike numbers in §8 already characterize the non-monotonicity).

**Interfaces:** May reuse the gkdv `_probe` pooled-reduce SHAPE for reference (READ `silly_kicks/gkdv/_probe.py`), but the verdicts are TF-56-specific. Produces: `optimizer_stability_verdict(frame, *, movable, objective, constraints, seeds, params)` (per-frame gap std across `seeds` ≈ 0 ∧ `num_iterations`-doubling leaves the optimum unchanged → `stable` / `seed_unstable`); `discrimination_verdict(gaps)` (`gaps` std > 0 across a shape spectrum → `discriminating` / `degenerate`); `predictive_verdict(paired)` (pooled `corr(positioning_gap_t, conceded_threat_{t+Δ})` with effect size + significance → `predictive` iff positive ∧ significant, else `not_predictive`; `arm_unscoreable` for a thin/velocity-less domain, a distinct token); `averaging_artifact_demo(...)` (reported); `horizon_sensitivity(...)` (reported). The predictive verdict is a pooled REDUCE over the corpus shards (NOT per shard), fed a `paired` frame of `(positioning_gap_t, conceded_threat_{t+Δ})`; conceded threat is derived corpus-side (the attacking team's realized action-threat over the next window — the driver's job, Task 15).

- [ ] **Step 1: Write failing tests** (fixture-scale, non-vacuous):
  - stability — with `init_sigma_m=2.0` the per-frame gap std across a seed set is ≈ 0 (assert < a tight tol) AND doubling `num_iterations` leaves the optimum unchanged; a companion asserts the verdict FLIPS to `seed_unstable` at a deliberately-large sigma (both sides).
  - discrimination — `discrimination_verdict` returns `discriminating` on a spread of shapes and `degenerate` on an all-identical-gap input (both sides).
  - predictive — `predictive_verdict` returns `predictive` on a synthetic paired frame with a planted positive correlation and `not_predictive` on a zero-correlation one (both sides); `arm_unscoreable` on an empty/velocity-less domain (distinct token).
  - averaging-artifact demo — WITHOUT `CappedContribution`, a threat+pressure `WeightedSum` optimum abandons a marked man; WITH the cap it does not (assert both sides).
  - horizon-sensitivity — returns the gap distribution at `max_reach_seconds ∈ {0.5, 0.7, 1.0}`.
- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** the probe helpers (pooled verdicts computed in a reduce — NOT per shard). — [ ] **Step 4: PASS.**

**Verify:** each verdict has a both-sides test; the stability test is non-vacuous (it FLIPS at a large sigma); the averaging-artifact demo is non-vacuous.

---

## Task 11: Commit-1 perf benchmark (MEASURES per-frame cost)

**Files:** Create `tests/positioning/test_positioning_perf_budget.py`

**Interfaces:** Consumes `tests/_perf_structural.py` (the shared harness, ADR-073) — `assert_subquadratic_growth(measure_work, *, sizes, max_exponent=1.5)` + `call_counter(monkeypatch, module, name)` to spy the per-iteration `compute_threat_pc` / feasibility-check call counts. (This module EXISTS — do not invent a new harness.)

- [ ] **Step 1: Write the benchmark** — use `call_counter` to count `ThreatObjective.score` (i.e. `compute_threat_pc`) invocations across one `optimise_positions` call and assert it equals the SA loop's `n_iter` + 1 (the incumbent-0 evaluation) — no hidden quadratic re-scoring. Use `assert_subquadratic_growth` over increasing `num_iterations` to confirm the loop is linear in iterations. Print (informational) the wall-clock of one scored-frame `compute_positioning_gap` call — the MEASURED per-frame cost feeding the feasibility decision.
- [ ] **Step 2: Run — capture the MEASURED per-frame cost.**
- [ ] **Step 3: FEASIBILITY GATE** — if the measured cost makes a bounded corpus (order dozens of matches × domain frames) infeasible on the DGX, implement the **coarse-grid pitch-control search surface** lever NOW (spec §8b): add `search_grid_resolution` to `SAParams`, run the SA on a coarser `compute_threat_pc` grid, and score the final `actual_score`/`best_score` at full resolution. This lands in Commit-1, BEFORE Commit-2. If cost is acceptable, skip the lever (record the decision + the number in the ADR).
- [ ] **Step 4: Run — PASS** (structural guard green; feasibility decision recorded).

**Verify:** per-frame cost is MEASURED (not an estimate) before any corpus run; the coarse-grid lever is landed iff needed.

---

## Task 12: Cross-cutting registrations (glossary / metric_contracts / C4 / NOTICE / gates)

**Files:**
- Modify: `silly_kicks/feature_glossary.py` (+ `tests/test_feature_glossary_coverage.py` passes), `silly_kicks/metric_contracts.py` (+ `tests/test_metric_contracts.py`), `docs/c4/architecture.dsl` (+ regen `architecture.html`; `tests/test_c4_aggregator_count.py`), `NOTICE`, `pyproject.toml` (ruff per-file-ignore), `tests/test_add_star_purity.py` (register the 2nd purity variant only if any `add_*` is added — none here), `tests/tracking/`… (see below)
- Test: `tests/positioning/test_velocity_tiering.py`, `tests/positioning/test_id_dtype_invariance.py`, `tests/positioning/test_orientation.py`

- [ ] **Step 1: Glossary** — add `positioning_gap`/`threat_actual`/`threat_optimum` `FeatureColumn` records (`emitting_module="_compute"`, `higher_is_better=False`); run `tests/test_feature_glossary_coverage.py` (must pass — a new metric column must be documented).
- [ ] **Step 2: metric_contracts (ADR-098)** — export `POSITIONING_METRIC_COLUMNS` + grain `("game_id","team_id")` from `silly_kicks/positioning/__init__.py`; register in `METRIC_CONTRACTS`; run `tests/test_metric_contracts.py` (a new column-emitting family MUST register or it fails).
- [ ] **Step 3: C4** — add a `positioning` container to `architecture.dsl` (analyst→positioning; positioning→tracking public seams; `xt` injected) + relationships; regen `architecture.html` via the `mad-scientist-skills:c4` skill; run `tests/test_c4_aggregator_count.py` — aggregator count STAYS 33 (positioning is a container, not an `add_*`).
- [ ] **Step 4: NOTICE** — add Oonk & Shah (databallpy `optimization`, MIT), Le et al. 2017, Spearman, Bekkers, Pleuler TTI to the References section; cross-link from module docstrings.
- [ ] **Step 5: pyproject** — add a `silly_kicks/positioning/*` ruff per-file-ignore for `N803`/`N806` IF uppercase math naming is used (`X`, etc. in `_optimizer`); else skip.
- [ ] **Step 6: Velocity-tiering + id-dtype + orientation tests** — the ADR-063 declared/undeclared behaviour (Task 8 covers; add a dedicated test), ADR-019 id-dtype invariance (numeric-actions × string-frames), orientation goal_map-not-identity. Run all.
- [ ] **Step 7: Run the FULL positioning suite + the touched cross-cutting gates — PASS.**

**Verify:** glossary coverage, metric_contracts completeness, C4 count 33 + container present, purity/id-dtype/orientation/velocity gates green.

---

## Task 13: ADR draft + spec/plan in tree

**Files:** Create `docs/superpowers/adrs/ADR-NNN-tf56-prescriptive-positioning-optimiser.md` (number assigned at commit-prep). Spec + this plan already in tree.

- [ ] **Step 1: Draft the ADR** from `docs/superpowers/adrs/ADR-TEMPLATE.md` — records: prescriptive SA solver + measured `positioning_gap`; objective **A** (threat) is the shipped column, composite/pressure/DAS via the protocol (consumer-side, CLAUDE.md raw-primitives convention); frozen intent-set params (ADR-009); velocity-required (FOV + reachability, SB360 deferred); per-defender ranking deferred (ADR-009, multi-club ICC); the perf feasibility gate + coarse-grid lever decision (record the measured number from Task 11); demote-if-fail (commit-2 battery); 2-commit clean-provenance; the additive `pressure_on_carrier` extraction. NO pre-claimed version/PR number in the body.
- [ ] **Step 2:** Verify the ADR references match the code (cite by reading).

**Verify:** ADR present, decisions recorded, no pre-claimed numbers.

---

## Task 14: Full-suite green + **COMMIT 1 (human-approval gate)**

- [ ] **Step 1:** `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/` — clean.
- [ ] **Step 2:** `python -m pyright` (bare) — clean.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e" -p no:randomly --benchmark-skip` — green.
- [ ] **Step 4:** Resolve version/PR/ADR numbers next-free against `main`; rename the ADR file; ensure `_version.py` bump is NOT done here (Commit-1 carries no version claim per the TF-63 lesson — the bump lands in the release commit if this is a single-cycle release, OR at Commit-2 per the 2-phase pattern; confirm with the owner at the gate).
- [ ] **Step 5: STOP — present the diff/file list to the owner. Do NOT commit.** On explicit approval: `git add` the positioning package + tests + docs (spec/plan/ADR) + C4 + glossary + metric_contracts + NOTICE + pyproject + the pressure-primitive refactor + its golden; ONE commit. Message: `feat(positioning): TF-56 prescriptive defensive-positioning optimiser + measured positioning_gap -- <version> (<PR>, <ADR>) -- Commit 1/2`. Trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` (NO Claude-Session trailer).

**Verify:** lint + pyright + full suite green BEFORE the gate; commit only on explicit approval.

---

## Task 15: Commit-2 corpus driver (owner-run)

**Files:** Create `scripts/build_tf56_positioning_validity.py`; Test: `tests/scripts/test_build_tf56_positioning_validity.py` (ASCII-help + smoke), extend `tests/scripts/test_provenance_wiring.py` (add the driver to `ARTIFACT_DRIVERS`).

**Interfaces:** Consumes `scripts/_driver.py` (`for_each`, ADR-052 shards), `scripts/_provenance.py` (`require_clean_tree`, `git_provenance`), `scripts/_loader_pining` (velocity-bearing tracking loaders), `silly_kicks.positioning._probe`.

- [ ] **Step 1: Write the driver test** — module imports without the tracking loaders (function-local imports); `main()` argparse-guarded; `require_clean_tree` called from `main()`; source is ASCII (Windows `--help`); registered in `ARTIFACT_DRIVERS`; the shard schema token + emitted columns pinned (4.77.1 stale-shard trap). Add a LOCAL reduce-path test on a tiny synthetic pooled corpus that runs the pooled verdicts (the [[feedback_test_trainer_locally_before_dgx]] gate — exercise the reduce before any DGX run).
- [ ] **Step 2: FAIL.** — [ ] **Step 3: Implement** — `for_each` over a BOUNDED velocity-bearing corpus (dozens of matches; SkillCorner/Sportec/GS), per-match shard = per-frame `positioning_gap` samples PAIRED with the corpus-derived **conceded threat over the next window** (the attacking team's realized action-threat / shots in the `t+Δ` window — computed shard-side from the same match actions, no external xG; `Δ` a driver constant, reported at a couple of values); a reduce computes the pooled **predictive verdict** (`corr(gap_t, conceded_threat_{t+Δ})` + effect size + significance) + **discrimination** (real-data gap distribution across `(game, team)`) + the **optimizer-stability** verdict + horizon-sensitivity + the averaging-artifact demo; writes `docs/research/tf56_positioning/` (report + `verdict.json`), stamping `run_commit` + `run_tree_dirty`. **Shard schema carries the paired conceded-threat column → bump the shard schema token (4.77.1 stale-shard trap).** — [ ] **Step 4: PASS (local reduce test).**

**Verify:** driver adopts `_driver`/`_provenance` seams; local reduce path tested BEFORE the DGX run.

---

## Task 16: Owner-run corpus battery + **COMMIT 2 (human-approval gate)**

- [ ] **Step 1:** On the DGX, from a CLEAN tree at the Commit-1 SHA, run `scripts/build_tf56_positioning_validity.py` on the bounded corpus → `docs/research/tf56_positioning/`.
- [ ] **Step 2:** Read the verdict. **GO** (predictive-positive-and-significant ∧ discriminating ∧ optimizer-stable/non-degenerate) → the glossaried column STAYS. **NO-GO** → **demote in this commit**: remove `positioning_gap`/`threat_actual`/`threat_optimum` from `feature_glossary` + `metric_contracts` + `positioning.__all__`'s metric surface (retain the solver + metric code as private modules), update the ADR + CHANGELOG to record the demotion (territorial_defense precedent). The solver + protocols stay public either way.
- [ ] **Step 3:** Bump `_version.py` to the resolved next-free version + CHANGELOG entry + TODO update (the release commit; single-source SSOT).
- [ ] **Step 4: STOP — present the diff to the owner. Do NOT commit/tag.** On explicit approval: ONE commit (`training_commit` = clean Commit-1 SHA), message `... -- Commit 2/2`. Then tag/publish only on further explicit go-ahead.

**Verify:** report has clean provenance; column ship-status matches the verdict; `.gitattributes` check N/A (no SHA-checksummed weights — pure solver, no `model.json`).

---

## Self-review (author checklist)

- **Spec coverage:** §4 module → Task 0; §5 protocols/solver → Tasks 1–7; §6 compute → Tasks 8–9; §7 edges → Tasks 8 (+ velocity/orientation Task 12); §8 probes → Task 10; §8b perf → Task 11; §9 testing → per-task + Task 12; §10 surface/registrations → Task 12; §11 release → Tasks 14/16; the pressure-primitive extraction (§5) → Task 3. All covered.
- **No pre-claimed numbers:** ADR is `ADR-NNN`; version bump only at the release/Commit-2 gate; no `PR-Snnn` literal in code/docs.
- **Commit discipline:** exactly 2 commits (Task 14, Task 16), each behind an explicit approval gate; no per-task commits.
- **Type consistency:** `OptimizeResult` fields (Tasks 6/7/8), `Objective.score(frame)->float` (Tasks 1–4), `positioning_gap_source` closed set (Task 8), `POSITIONING_METRIC_COLUMNS` grain `(game_id, team_id)` (Tasks 9/12) — consistent across tasks.
- **KNOWN implementation-time reads (not placeholders):** `domain_ball_to_goal_m` value (Task 0 reads gkdv), the gkdv `_probe` verdict helper reuse-vs-reimplement decision (Task 10 reads gkdv), the exact fixture player ids in the `<...>` test slots (filled from the committed fixture at implementation).
```
