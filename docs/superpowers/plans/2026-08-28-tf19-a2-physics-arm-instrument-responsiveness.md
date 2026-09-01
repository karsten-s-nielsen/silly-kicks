# TF-19 A+2 — physics-arm instrument validity + responsiveness + named-keeper validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the physics-arm Layer-0 (instrument validity) + Layer-1 (responsiveness) probes, a caller-supplied keeper-identity resolver, the arm-direction-key normalization, and an owner-run corpus driver, so the shipped GKDV physics arms (`delta_das`, `delta_threat_suppression`) get a demonstrable, reported-not-gated named-keeper validation on WC2022 GS — without foreclosing SB360.

**Architecture:** New pure functions in `gkdv/_probe.py` (dose imposer + Layer-0/Layer-1 verdicts + paired-vector controls) reusing the shipped ghost engine and arms; identity + arm-key seams in `gkdv/`; a two-stage `scripts/` driver (ADR-052 `for_each` map of per-frame values + a reduce computing pooled verdicts, the per-arm `gate_eligible` census, and keeper aggregation). Reported-not-gated; no gate flip, no retrain, C4-free.

**Tech Stack:** numpy, pandas; the shipped `silly_kicks.gkdv` engine/arms/metric/validate; `scripts/_driver.py` (`for_each`), `scripts/_provenance.py`, `scripts/_sb_raw.py` (`parse_roster`).

**Spec:** `docs/superpowers/specs/2026-08-28-tf19-a2-physics-arm-instrument-responsiveness-design.md` (APPROVED, two lakehouse reviews). Read it first — this plan implements it.

## Global Constraints

- **Physics arms only** (`delta_das`, `delta_threat_suppression`). Attempt arms (xS/xCross) out of scope.
- **Reported-not-gated.** No gate flip: `TF19_PROBE_ABS_FLOOR` untouched, `regate_verdict` routing untouched, `EXPECTED_DIRECTION` untouched. The composing `gkdv_discrimination_verdict` + Layers 2/3 stay deferred.
- **No retrain, no re-materialize, no artifact change, C4-free** (no new action-coupled `add_*` aggregator).
- **ADR-037 import allowlist:** `gkdv/` imports `tracking._das` ONLY (via `_das_port.py`); `tracking/` never imports `gkdv/`. Cross-package assembly (roster parsing, model loading) lives in the `scripts/` driver, never in `gkdv/`.
- **Pre-registered constants locked in code before the owner run.** Copied from the parent spec: `SATURATING_MULTIPLE = 5`, the 2 m realistic floor, the `{2, 3, 4}` ladder. New physics-arm registrations (parent Layer-1 idiom fixes the form, not the value): `PHYSICS_ARM_PROBE_RATIO = 2.0`, `MIN_DOMAIN_FRAMES`, `R = 3` (paired-vector control count), the two saturating positions (goal-line centre; goal-relative `x = 30 m`). No `abs_floor` (Layer 1 is comparable-not-decisive).
- **Arm-column → direction-key mapping is a required seam:** `delta_threat_suppression` (arm column) ≠ `delta_threat` (`EXPECTED_DIRECTION` key); pin `_ARM_DIRECTION_KEY` and test that every arm column resolves.
- **`arm_unscoreable` is a first-class verdict** — velocity-less ΔDAS OR a domain below `MIN_DOMAIN_FRAMES` — distinct from `instrument_void`. Never a fabricated 0/NaN passed off as a measurement.
- **Layer-0/1 verdicts are POOLED-corpus statistics computed in a REDUCE over all shards**, never per shard.
- **Reach is measured on the binding census:** per-arm `gate_eligible` (both `min_nonzero ≥ 20` AND `min_games ≥ 2`), ΔDAS and ΔThreat separately.
- Lint at CI scope (`ruff check/format silly_kicks/ tests/ scripts/`); bare `pyright`; full `-m "not e2e"`. Tools via `python -m`.
- **NOBODY CLAIMS VERSION NUMBERS UNTIL COMMIT-PREP.** ADR is `ADR-082`; do NOT touch `pyproject.toml`/`__init__`/`uv.lock`/CHANGELOG/TODO until commit-prep after `git fetch && git merge origin/main`.
- **Single feature branch, single commit, single PR. No `git commit` in any step. No commit without explicit owner approval.**

---

## Task 1: Dose imposer (`gkdv/_probe.py`)

**Files:**
- Create: `silly_kicks/gkdv/_probe.py`
- Test: `tests/gkdv/test_probe_dose.py` (create)

**Interfaces:**
- Consumes: `build_ghost_frames`, `_goal_lookup`, `_same_team` (`gkdv/_engine.py`, within-package); `GkdvParams`; `resolve_defended_goals`/`GoalMap` (`tracking._gk_resolve`, already used by the engine). NOT `provenance_to_targets` — its 7-col `_TARGET_COLUMNS` contract can't supply the columns needed here (see P1 / `_build_dose_targets`).
- Produces:
  - `Dose` (enum/`Literal["realistic","ladder","saturating_goalline","saturating_x30"]` — the discrete doses; the ladder distances live in `LADDER_M`).
  - `LADDER_M: tuple[float, float, float] = (2.0, 3.0, 4.0)`; `REALISTIC_MIN_DISP_M: float = 2.0`; `SATURATING_X30_GR: float = 30.0`.
  - `impose_defending_keeper_dose(frames, *, home_team_id, dose, displacement=None, params=_DEFAULT_PARAMS) -> tuple[pd.DataFrame, pd.DataFrame]` returning `(imposed_frames, targets)`. `imposed_frames` is a NEW frame (never mutates `frames`) with ONLY the defending keeper substituted to the dose position; `targets` is the per-scored-frame defending-keeper table from `_build_dose_targets` (carries `game_id/period_id/frame_id`, `defending_team_id`, `actual_x/actual_y`, `defended_goal_x`, `ghost_x/ghost_y`, `displacement_m`) with `imp_x`/`imp_y` (the applied dose position) attached, so downstream (`paired_vector_controls`) has the displacement vector. For `dose="realistic"` the imposed position is the ghost model's own position filtered to `displacement_m >= REALISTIC_MIN_DISP_M` (so `model=` must be supplied for the realistic dose); for `"ladder"` it is the actual position displaced toward the defended goal by `displacement` m; for the two `saturating_*` it is the goal-relative landmark reprojected into frame coords via the defended-goal end.

- [ ] **Step 1: Write the failing test — saturating dose puts the defending keeper on the defended goal line**

```python
# tests/gkdv/test_probe_dose.py
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gkdv._probe import impose_defending_keeper_dose


def _two_keeper_frame(defended_end_home=0.0):
    """One alive-ball frame, ball near the home-defended goal, both keepers present.

    home team (id 1) defends x=0; away (id 2) attacks toward x=0. Ball at x=12 (within the
    default domain_ball_to_goal_m of the x=0 goal). Away is in possession/attacking.
    """
    rows = [
        # game, period, frame, team, player, is_ball, is_gk, x, y, vx, vy, speed, attacking_dir
        (1, 1, 10, 2, 200, False, False, 12.0, 34.0, 0.0, 0.0, 0.0, "rtl"),  # ball carrier (away)
        (1, 1, 10, 1, 100, False, True, 4.0, 30.0, 0.0, 0.0, 0.0, "rtl"),    # HOME keeper (defending)
        (1, 1, 10, 2, 201, False, True, 100.0, 34.0, 0.0, 0.0, 0.0, "rtl"),  # away keeper (far)
        (1, 1, 10, None, None, True, False, 12.0, 34.0, 0.0, 0.0, 0.0, "rtl"),  # ball
    ]
    cols = ["game_id", "period_id", "frame_id", "team_id", "player_id", "is_ball",
            "is_goalkeeper", "x", "y", "vx", "vy", "speed", "team_attacking_direction"]
    return pd.DataFrame(rows, columns=cols)


def test_saturating_goalline_puts_home_keeper_on_x0_centre():
    frames = _two_keeper_frame()
    imposed, targets = impose_defending_keeper_dose(
        frames, home_team_id=1, dose="saturating_goalline"
    )
    # only the defending (home) keeper moved; it now sits at the defended goal line centre.
    gk = imposed[(imposed["team_id"] == 1) & imposed["is_goalkeeper"]]
    assert float(gk["x"].iloc[0]) == 0.0
    assert float(gk["y"].iloc[0]) == 34.0
    # nothing else moved (away keeper unchanged) and frames is not mutated
    away_gk = imposed[(imposed["team_id"] == 2) & imposed["is_goalkeeper"]]
    assert float(away_gk["x"].iloc[0]) == 100.0
    assert float(frames[(frames["team_id"] == 1) & frames["is_goalkeeper"]]["x"].iloc[0]) == 4.0
```

- [ ] **Step 2: Run — expect FAIL** (`ModuleNotFoundError: silly_kicks.gkdv._probe`).

Run: `python -m pytest tests/gkdv/test_probe_dose.py -q`

- [ ] **Step 3: Implement `impose_defending_keeper_dose`**

```python
# silly_kicks/gkdv/_probe.py
"""TF-19 physics-arm instrument-validity (Layer 0) + responsiveness (Layer 1) probes.

Reported-not-gated. Depends on the shipped gkdv engine/arms ONLY (ADR-037: gkdv may import
tracking._das solely via _das_port; this module imports neither tracking nor providers).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks.id_compat import align_join_keys, same_id
from silly_kicks.spadl import config as _spadlconfig

from ._engine import GkdvParams, _goal_lookup, _same_team, build_ghost_frames  # within-package reuse

_DEFAULT_PARAMS = GkdvParams()
_FIELD_LENGTH = _spadlconfig.field_length  # 105.0
_GOAL_Y = _spadlconfig.field_width / 2.0   # 34.0
_FRAME_KEYS = ["game_id", "period_id", "frame_id"]

Dose = Literal["realistic", "ladder", "saturating_goalline", "saturating_x30"]

LADDER_M: tuple[float, float, float] = (2.0, 3.0, 4.0)
REALISTIC_MIN_DISP_M: float = 2.0
SATURATING_X30_GR: float = 30.0


def impose_defending_keeper_dose(
    frames: pd.DataFrame,
    *,
    home_team_id: int | str,
    dose: Dose,
    displacement: float | None = None,
    model=None,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Substitute ONLY the defending keeper at the dose position. PURE (new frame)."""
    _, provenance, _ = build_ghost_frames(frames, model=model, home_team_id=home_team_id, params=params)
    targets = _build_dose_targets(frames, provenance)
    if not len(targets):
        return frames.copy(), targets

    defended = targets["defended_goal_x"].to_numpy(dtype=float)  # 0.0 or 105.0 per scored frame
    actual_x = targets["actual_x"].to_numpy(dtype=float)
    actual_y = targets["actual_y"].to_numpy(dtype=float)

    if dose == "saturating_goalline":
        imp_x = defended
        imp_y = np.full(len(targets), _GOAL_Y)
    elif dose == "saturating_x30":
        # goal-relative x = 30 m: 30 m from the defended goal, toward the pitch.
        imp_x = np.where(defended == 0.0, SATURATING_X30_GR, _FIELD_LENGTH - SATURATING_X30_GR)
        imp_y = actual_y
    elif dose == "ladder":
        if displacement is None:
            raise ValueError("dose='ladder' requires displacement=")
        # move the keeper `displacement` m toward the defended goal (goal-ward = honest extreme).
        sign = np.where(defended == 0.0, -1.0, 1.0)
        imp_x = actual_x + sign * float(displacement)
        imp_y = actual_y
    elif dose == "realistic":
        # the ghost model's own position, filtered to |displacement| >= REALISTIC_MIN_DISP_M.
        imp_x = targets["ghost_x"].to_numpy(dtype=float)
        imp_y = targets["ghost_y"].to_numpy(dtype=float)
        keep = targets["displacement_m"].to_numpy(dtype=float) >= REALISTIC_MIN_DISP_M
        targets = targets.loc[keep].reset_index(drop=True)
        imp_x, imp_y = imp_x[keep], imp_y[keep]
    else:
        raise ValueError(f"unknown dose: {dose!r}")

    imposed = _substitute_defending_keeper(frames, targets, imp_x, imp_y, params=params)
    targets = targets.copy()
    targets["imp_x"] = imp_x  # the applied dose position, for the paired-vector control's vector
    targets["imp_y"] = imp_y
    return imposed, targets


def _substitute_defending_keeper(frames, targets, imp_x, imp_y, *, params):
    """Mirror of _engine._write_back but for an IMPOSED (dose) position."""
    out = frames.copy()
    move = targets[[*_FRAME_KEYS, "defending_team_id"]].copy()
    move["imp_x"] = imp_x
    move["imp_y"] = imp_y
    gk_mask = (out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)).to_numpy()
    gk_side = out.loc[gk_mask, [*_FRAME_KEYS, "team_id"]].rename(columns={"team_id": "defending_team_id"})
    left, right = align_join_keys(gk_side, move, [*_FRAME_KEYS, "defending_team_id"])
    joined = left.merge(right, on=[*_FRAME_KEYS, "defending_team_id"], how="left")
    joined.index = gk_side.index
    hit = joined["imp_x"].notna().to_numpy() & joined["imp_y"].notna().to_numpy()
    idx = joined.index[hit]
    if len(idx):
        out.loc[idx, "x"] = joined.loc[idx, "imp_x"].to_numpy(dtype=float)
        out.loc[idx, "y"] = joined.loc[idx, "imp_y"].to_numpy(dtype=float)
        if not params.ghost_keeps_actual_velocity:
            for c in ("vx", "vy", "speed"):
                if c in out.columns:
                    out.loc[idx, c] = 0.0
    return out


def _build_dose_targets(frames: pd.DataFrame, provenance: pd.DataFrame) -> pd.DataFrame:
    """Per-scored-DEFENDING-frame table for the dose imposer + paired-vector controls.

    Sourced correctly (do NOT route through provenance_to_targets, whose 7-col _TARGET_COLUMNS
    contract renames ghost_x/y->target_x/y and drops actual_*/defended_goal_x):
      * ghost_x/ghost_y/displacement_m/defending_team_id  <- `provenance` (_PROVENANCE_COLUMNS)
      * actual_x/actual_y                                 <- `frames`' defending-GK rows
      * defended_goal_x (0.0/105.0)                       <- resolve_defended_goals goal map
    Carries every column the imposer AND paired_vector_controls read.
    """
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals

    scored = provenance[provenance["drop_reason"].isna()]
    keep = _same_team(scored["gk_team_id"], scored["defending_team_id"]).to_numpy()  # ADR-019 safe mask
    defending = scored.loc[keep, [*_FRAME_KEYS, "defending_team_id", "ghost_x", "ghost_y", "displacement_m"]]

    players = frames[~frames["is_ball"].astype(bool)]
    gk = players[players["is_goalkeeper"].astype(bool)][[*_FRAME_KEYS, "team_id", "x", "y"]]
    gk = gk.rename(columns={"team_id": "defending_team_id", "x": "actual_x", "y": "actual_y"})
    left, right = align_join_keys(defending, gk, [*_FRAME_KEYS, "defending_team_id"])
    t = left.merge(right, on=[*_FRAME_KEYS, "defending_team_id"], how="left")

    goal_map = resolve_defended_goals(frames)
    t["defended_goal_x"] = [
        _goal_lookup(goal_map, g, p, tm)
        for g, p, tm in zip(t["game_id"], t["period_id"], t["defending_team_id"], strict=True)
    ]
    return t.reset_index(drop=True)
```

> The defending-row filter reuses the engine's `_same_team` (column-vs-column, ADR-019-safe;
> `.to_numpy()` because `scored` is a non-contiguous slice — the `ids_equal` fresh-RangeIndex trap).
> **Verify `_goal_lookup` and `_same_team` are importable from `_engine`** (both are used internally —
> `_goal_lookup` at `:385`, `_same_team` at `:442`); if either is defined below its use / not
> module-level, promote it or replicate its short body (`_goal_lookup` returns `0.0`/`105.0` from the
> goal map). Do NOT add columns to `provenance_to_targets` — its `out[list(_TARGET_COLUMNS)]` projection
> is a cross-package-pinned contract (`_engine.py:61-64`). Drop the unused `same_id` import if the final
> module doesn't reference it (ruff F401).

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Add ladder + away-team (rtl) correctness tests**

```python
def test_ladder_moves_toward_defended_goal_home():
    frames = _two_keeper_frame()
    imposed, _ = impose_defending_keeper_dose(frames, home_team_id=1, dose="ladder", displacement=3.0)
    gk = imposed[(imposed["team_id"] == 1) & imposed["is_goalkeeper"]]
    assert float(gk["x"].iloc[0]) == 1.0  # 4.0 - 3.0, toward x=0


def test_saturating_x30_home_is_30m_from_defended_goal():
    frames = _two_keeper_frame()
    imposed, _ = impose_defending_keeper_dose(frames, home_team_id=1, dose="saturating_x30")
    gk = imposed[(imposed["team_id"] == 1) & imposed["is_goalkeeper"]]
    assert float(gk["x"].iloc[0]) == 30.0
```

- [ ] **Step 6: Run — expect PASS.**

---

## Task 2: Layer-0 instrument-validity verdict (`gkdv/_probe.py`)

**Files:**
- Modify: `silly_kicks/gkdv/_probe.py`
- Test: `tests/gkdv/test_probe_layer0.py` (create)

**Interfaces:**
- Consumes: Task 1's imposer; `delta_das_batch`, `delta_threat_suppression_batch` (`gkdv/_arms.py`).
- Produces:
  - `SATURATING_MULTIPLE: float = 5.0`, `MIN_DOMAIN_FRAMES: int` (registered; the value + its derivation recorded in the artifact — start at a stated default, e.g. 200, and record the corpus-measured basis).
  - `LAYER0_VERDICTS = ("instrument_valid", "instrument_void", "arm_unscoreable")`.
  - `layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str` — a PURE function over already-pooled scalars/arrays: applies the two `arm_unscoreable` short-circuits FIRST (any leg all-NaN/empty → provider can't score; `n_domain < MIN_DOMAIN_FRAMES` → insufficient support), then the parent's void condition.

- [ ] **Step 1: Write the failing tests — dead→void, live→valid, unscoreable legs (all non-vacuous)**

```python
# tests/gkdv/test_probe_layer0.py
import numpy as np
import pytest

from silly_kicks.gkdv._probe import MIN_DOMAIN_FRAMES, layer0_instrument_verdict


def test_live_instrument_is_valid():
    # saturating median 0.5 >= 5x realistic 0.05 => valid (either leg suffices).
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, 0.05), saturating_abs=np.full(300, 0.5),
        placebo_p95=0.02, n_domain=300,
    )
    assert v == "instrument_valid"


def test_dead_instrument_is_void():
    # saturating flat (0.04): NOT >= 5x realistic (0.05->0.25) AND NOT > placebo p95 (0.10) => void.
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, 0.05), saturating_abs=np.full(300, 0.04),
        placebo_p95=0.10, n_domain=300,
    )
    assert v == "instrument_void"


def test_velocity_less_arm_is_unscoreable_not_void():
    v = layer0_instrument_verdict(
        realistic_abs=np.full(300, np.nan), saturating_abs=np.full(300, np.nan),
        placebo_p95=np.nan, n_domain=300,
    )
    assert v == "arm_unscoreable"  # asserted DISTINCT from instrument_void


def test_thin_domain_is_unscoreable_not_void():
    v = layer0_instrument_verdict(
        realistic_abs=np.full(3, 0.05), saturating_abs=np.full(3, 0.04),
        placebo_p95=0.10, n_domain=3,
    )
    assert v == "arm_unscoreable"  # a thin domain must NOT read as "broken"
    assert MIN_DOMAIN_FRAMES > 3
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError`).

- [ ] **Step 3: Implement the verdict**

```python
# silly_kicks/gkdv/_probe.py (append)
SATURATING_MULTIPLE: float = 5.0
MIN_DOMAIN_FRAMES: int = 200  # registered; derivation recorded in the artifact
LAYER0_VERDICTS = ("instrument_valid", "instrument_void", "arm_unscoreable")


def layer0_instrument_verdict(*, realistic_abs, saturating_abs, placebo_p95, n_domain) -> str:
    real = np.asarray(realistic_abs, dtype=float)
    sat = np.asarray(saturating_abs, dtype=float)
    # (a) provider can't score this arm; (b) insufficient support -> arm_unscoreable, NOT void.
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    real_med = np.nanmedian(real) if real.size and np.isfinite(real).any() else np.nan
    sat_med = np.nanmedian(sat) if sat.size and np.isfinite(sat).any() else np.nan
    if not (np.isfinite(real_med) and np.isfinite(sat_med) and np.isfinite(placebo_p95)):
        return "arm_unscoreable"
    passes_multiple = sat_med >= SATURATING_MULTIPLE * real_med
    passes_placebo = sat_med > float(placebo_p95)
    # Parent VOID condition (copied): void iff NOT multiple AND NOT placebo. Valid = either.
    return "instrument_void" if (not passes_multiple and not passes_placebo) else "instrument_valid"
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Discrimination proof (documented in the module docstring; not committed):** flip the parent void condition to `and` → `test_live_instrument_is_valid` still passes but `test_dead_instrument_is_void` flips — confirming the boolean is load-bearing. Revert.

---

## Task 3: Layer-1 responsiveness + paired-vector controls (`gkdv/_probe.py`)

> **CORRECTED (as-built, 2026-08-29 — see spec §4.2 + ADR).** The `paired_vector_controls` design below
> (one COMBINED control moving the nearest + R random outfielders together, returning a single
> `pd.DataFrame`) was a flaw: it makes `nd_med` and `placebo_p95` the same array (the Layer-1 `max` is
> decorative) and compares a 1-player keeper move against an R+1-player control. As-built, it follows
> the parent `_model_eval` idiom — returns `dict[str, pd.DataFrame]` `{"nearest", "placebo_0..{R-1}"}`,
> ONE defending outfielder displaced per control (nearest ALONE + R single-outfielder placebos) — so
> `nd_med` and `placebo_p95` are DISTINCT. Also as-built: the Layer-0 multiple leg requires
> `real_med > 0` (else `sat ≥ 5·0` vacuously validates a dead instrument).

**Files:**
- Modify: `silly_kicks/gkdv/_probe.py`
- Test: `tests/gkdv/test_probe_layer1.py` (create)

**Interfaces:**
- Produces:
  - `PHYSICS_ARM_PROBE_RATIO: float = 2.0` (new registration; NOT `TF19_PROBE_RATIO`/`XS_PROBE_RATIO`, which are model-specific — see spec §4.2), `R: int = 3`.
  - **Pinned regimes** (so `_measure_match` and the controls are unambiguous): `REGIME_O_DOSE = "realistic"` (observed ghost — the shipped metric) and `REGIME_I_DOSE = "ladder"` with `REGIME_I_LADDER_M = 2.0` (the imposed, discriminating dose — the registered 2 m convention). `layer1_responsiveness_verdict` is computed **once per regime per arm**; the paired-vector control for a regime displaces by **that regime's own per-frame `(imp − actual)` vector**.
  - `LAYER1_VERDICTS = ("responsive", "not_responsive", "arm_unscoreable")`.
  - `paired_vector_controls(frames, targets, *, r, rng) -> pd.DataFrame` — displaces the nearest defender + `R` random outfielders by the SAME per-frame vector the keeper moved, returning control frames for the placebo band.
  - `layer1_responsiveness_verdict(*, gk_med, nd_med, placebo_p95, n_domain) -> str` — the shipped idiom `gk_med >= RATIO * max(nd_med, placebo_p95)` with the same `n_domain`/NaN `arm_unscoreable` short-circuits as Layer 0.

- [ ] **Step 1: Write the failing tests — both sides of the ratio + identical-vector control**

```python
# tests/gkdv/test_probe_layer1.py
import numpy as np

from silly_kicks.gkdv._probe import PHYSICS_ARM_PROBE_RATIO, layer1_responsiveness_verdict


def test_responsive_when_gk_beats_ratio_times_controls():
    assert PHYSICS_ARM_PROBE_RATIO == 2.0
    v = layer1_responsiveness_verdict(gk_med=0.30, nd_med=0.10, placebo_p95=0.12, n_domain=300)
    assert v == "responsive"  # 0.30 >= 2.0 * max(0.10, 0.12)


def test_not_responsive_when_flat():
    v = layer1_responsiveness_verdict(gk_med=0.20, nd_med=0.15, placebo_p95=0.12, n_domain=300)
    assert v == "not_responsive"  # 0.20 < 2.0 * 0.15
```

```python
def test_paired_vector_control_applies_the_identical_vector():
    # A keeper displaced by (dx,dy) must displace the control player by the SAME (dx,dy).
    import pandas as pd
    from silly_kicks.gkdv._probe import paired_vector_controls
    frames = pd.DataFrame({
        "game_id": [1, 1, 1], "period_id": [1, 1, 1], "frame_id": [10, 10, 10],
        "team_id": [1, 1, 2], "player_id": [100, 101, 200],
        "is_ball": [False, False, False], "is_goalkeeper": [True, False, False],
        "x": [4.0, 20.0, 50.0], "y": [30.0, 40.0, 34.0],
    })
    targets = pd.DataFrame({
        "game_id": [1], "period_id": [1], "frame_id": [10], "defending_team_id": [1],
        "actual_x": [4.0], "actual_y": [30.0], "imp_x": [0.0], "imp_y": [34.0],  # vector (-4, +4)
    })
    ctrl = paired_vector_controls(frames, targets, r=1, rng=np.random.default_rng(0))
    moved = ctrl[(ctrl["player_id"] == 101)]  # nearest defender (same team, outfield)
    assert float(moved["x"].iloc[0]) == 16.0 and float(moved["y"].iloc[0]) == 44.0  # 20-4, 40+4
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the verdict + controls** (the control picks the nearest defending-team outfielder to the keeper, plus `r` random defending-team outfielders, and adds the keeper's `(imp - actual)` vector to each; `layer1_responsiveness_verdict` mirrors Layer 0's short-circuits then applies `gk_med >= PHYSICS_ARM_PROBE_RATIO * max(nd_med, placebo_p95)`).

```python
# silly_kicks/gkdv/_probe.py (append)
PHYSICS_ARM_PROBE_RATIO: float = 2.0
R: int = 3
LAYER1_VERDICTS = ("responsive", "not_responsive", "arm_unscoreable")


def layer1_responsiveness_verdict(*, gk_med, nd_med, placebo_p95, n_domain) -> str:
    if n_domain < MIN_DOMAIN_FRAMES:
        return "arm_unscoreable"
    if not (np.isfinite(gk_med) and np.isfinite(nd_med) and np.isfinite(placebo_p95)):
        return "arm_unscoreable"
    thresh = PHYSICS_ARM_PROBE_RATIO * max(float(nd_med), float(placebo_p95))
    return "responsive" if gk_med >= thresh else "not_responsive"


def paired_vector_controls(frames, targets, *, r, rng):
    """Placebo band: displace the nearest defending-team outfielder + r random ones by the SAME
    per-frame vector the keeper moved. PURE (new frame). ADR-068: the outfield rows are grouped
    ONCE via group_rows and looked up per target -- never a full-table filter inside the loop."""
    from silly_kicks._frame_index import group_rows  # ADR-068 grouping seam

    out = frames.copy()
    outfield_mask = (~out["is_ball"].astype(bool) & ~out["is_goalkeeper"].astype(bool)).to_numpy()
    outfield = out.loc[outfield_mask]
    groups = group_rows(outfield, [*_FRAME_KEYS, "team_id"])  # key -> positional rows (empty on miss)
    for tgt in targets.itertuples(index=False):
        rows = groups.get((tgt.game_id, tgt.period_id, tgt.frame_id, tgt.defending_team_id))
        if rows is None or not len(rows):
            continue
        sub = outfield.iloc[rows]  # candidate defending-team outfielders in this frame
        ax, ay = float(tgt.actual_x), float(tgt.actual_y)
        dx, dy = float(tgt.imp_x) - ax, float(tgt.imp_y) - ay
        dist = np.hypot(sub["x"].to_numpy(float) - ax, sub["y"].to_numpy(float) - ay)
        order = np.argsort(dist, kind="stable")           # nearest first
        extras = order[1:]
        pick_local = [order[0], *(rng.choice(extras, size=min(r, len(extras)), replace=False) if len(extras) else [])]
        pick_idx = sub.index[list(pick_local)]
        out.loc[pick_idx, "x"] = out.loc[pick_idx, "x"].to_numpy(float) + dx
        out.loc[pick_idx, "y"] = out.loc[pick_idx, "y"].to_numpy(float) + dy
    return out
```

> `group_rows(df, by).get(key)` returns the positional row indices for that key (empty on miss) — confirm the exact return shape against `silly_kicks/_frame_index.py` before implementing (ADR-068 seam). `paired_vector_controls` consumes a `targets` table carrying `imp_x`/`imp_y` (the applied dose) plus `actual_x`/`actual_y` and `defending_team_id` — Task 1's `impose_defending_keeper_dose` attaches `imp_x`/`imp_y` to the `targets` it returns (see its interface).

- [ ] **Step 4: Run — expect PASS.**

---

## Task 4: Consumer seams — keeper-identity resolver + arm-direction-key normalization

> **SUPERSEDED (as-built, 2026-08-29 — see ADR).** The gkdv-local `resolve_defending_keeper_id` /
> `silly_kicks/gkdv/_identity.py` half below was **DROPPED**. The SB360 first-class-provider cycle
> shipped ONE keeper-identity resolver `tracking.resolve_keeper_identities` (ADR-078, ADR-055
> single-source), so a second gkdv-local identity path would violate ADR-055 and ADR-037 (gkdv reaches
> `tracking` only through `_das_port`). The Task-5 driver consumes `tracking.resolve_keeper_identities`
> DRIVER-side instead — no `gkdv/_identity.py`, no `resolve_defending_keeper_id` export, no
> `test_identity_resolver.py`. **Only the arm-direction-key half (`_ARM_DIRECTION_KEY` /
> `expected_direction_for_arm`) was built as written.**

**Files:**
- Create: `silly_kicks/gkdv/_identity.py`
- Modify: `silly_kicks/gkdv/_validate.py` (add `_ARM_DIRECTION_KEY` + `expected_direction_for_arm`)
- Modify: `silly_kicks/gkdv/__init__.py` (export the new public names)
- Test: `tests/gkdv/test_identity_resolver.py`, `tests/gkdv/test_arm_direction_key.py` (create)

**Interfaces:**
- Produces:
  - `resolve_defending_keeper_id(frames, *, identity="native", roster=None, on_unresolved="drop") -> pd.Series` — per defending-GK frame row, the keeper id. **The discriminator is the explicit `identity` kwarg, NEVER a guess** (the driver sets it per provider): `identity="native"` returns the frame `player_id` (identity-bearing tracking); `identity="roster"` maps the resolved `team_id` → `roster[team_id]` (a `dict[team_id → gk_player_id]` the caller injects). Unresolvable (roster missing the team, or a non-native id with no roster) → NA (dropped-and-counted by the caller, never fabricated). PURE. **Roster contract:** `roster` is `{team_id: gk_player_id}`; the driver builds it from `scripts/_sb_raw.py::parse_roster` (whose shape is `{player_id: {name, jersey, team, position}}`) by keeping `position == "Goalkeeper"` and keying by `team` — the team→GK reduction is a driver-side transform, stated in Task 5.
  - `_ARM_DIRECTION_KEY = {"delta_das": "delta_das", "delta_threat_suppression": "delta_threat"}`; `expected_direction_for_arm(arm_column) -> str` (looks up `_ARM_DIRECTION_KEY` then `EXPECTED_DIRECTION`; raises `KeyError` on an unmapped arm).

- [ ] **Step 1: Write the failing arm-key test (F1 guard)**

```python
# tests/gkdv/test_arm_direction_key.py
import pytest

from silly_kicks.gkdv._validate import EXPECTED_DIRECTION, expected_direction_for_arm


def test_every_arm_column_resolves_to_expected_direction():
    for arm_col in ("delta_das", "delta_threat_suppression"):
        assert expected_direction_for_arm(arm_col) == "negative"


def test_unmapped_arm_raises_not_silent():
    with pytest.raises(KeyError):
        expected_direction_for_arm("delta_not_an_arm")
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: expected_direction_for_arm`).

- [ ] **Step 3: Implement in `_validate.py`**

```python
# silly_kicks/gkdv/_validate.py (append, beside EXPECTED_DIRECTION)
#: Arm OUTPUT column -> EXPECTED_DIRECTION key. The threat arm's column is
#: `delta_threat_suppression` (_arms.py) but its direction key is `delta_threat` — this is the
#: canonical bridge, so a new arm cannot silently skip its sign check.
_ARM_DIRECTION_KEY: dict[str, str] = {
    "delta_das": "delta_das",
    "delta_threat_suppression": "delta_threat",
}


def expected_direction_for_arm(arm_column: str) -> str:
    """The expected sign for an arm's OUTPUT column (negative == deterrent). Raises on unmapped."""
    return EXPECTED_DIRECTION[_ARM_DIRECTION_KEY[arm_column]]
```

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Write the failing identity-resolver tests**

```python
# tests/gkdv/test_identity_resolver.py
import numpy as np
import pandas as pd

from silly_kicks.gkdv._identity import resolve_defending_keeper_id


def _gk_frames(player_id):
    return pd.DataFrame({
        "game_id": [1], "period_id": [1], "frame_id": [10], "team_id": [1],
        "player_id": [player_id], "is_ball": [False], "is_goalkeeper": [True],
    })


def test_native_player_id_passes_through():
    frames = _gk_frames(player_id=77)
    out = resolve_defending_keeper_id(frames, identity="native")
    assert int(out.iloc[0]) == 77


def test_roster_identity_maps_team_to_gk_id():
    frames = _gk_frames(player_id=0)          # SB360: row-numbered, not identity
    out = resolve_defending_keeper_id(frames, identity="roster", roster={1: 999})  # team_id -> gk_id
    assert int(out.iloc[0]) == 999


def test_roster_missing_team_is_dropped_not_fabricated():
    frames = _gk_frames(player_id=0)
    out = resolve_defending_keeper_id(frames, identity="roster", roster={2: 999})  # team 1 absent
    assert out.isna().iloc[0]                  # NA -> caller drops-and-counts, never fabricated
```

> The discriminator is the **explicit `identity` kwarg** — the resolver never guesses from roster
> presence (spec §4.3). `roster` is `{team_id: gk_player_id}`; the driver's `parse_roster` →
> `{team_id: gk_id}` reduction (keep `position == "Goalkeeper"`, key by `team`) is specified in Task 5.

- [ ] **Step 6: Implement `_identity.py`, run — expect PASS.**

- [ ] **Step 7: Export** `resolve_defending_keeper_id`, `expected_direction_for_arm` (and the Task 1-3 public probe names) in `gkdv/__init__.py` `__all__`.

---

## Task 5: Owner-run corpus driver (map + reduce)

**Files:**
- Create: `scripts/build_tf19_instrument_responsiveness.py`
- Modify: `tests/scripts/_script_population.py` (enroll in the ADR-052/ADR-056 driver populations)
- Test: `tests/scripts/test_tf19_instrument_responsiveness_driver.py` (create)

**Interfaces:**
- Consumes: `for_each` (`scripts/_driver.py`), `require_clean_tree`/`git_provenance` (`scripts/_provenance.py`), the Task 1-4 probe/identity seams, `aggregate_by_keeper`/`behavioural_anchoring_verdict` (`gkdv`), `parse_roster` (`scripts/_sb_raw.py`, SB360 only), the pining loaders.
- Produces: `docs/research/tf19_instrument_responsiveness/` (per-frame shards + a reduced `verdicts.json` + `named_keeper_signs.parquet` + provenance).

- [ ] **Step 1: Write the failing test — the shard→POOL→verdict path pools, and a per-shard impl would fail**

```python
# tests/scripts/test_tf19_instrument_responsiveness_driver.py
import numpy as np
import pandas as pd

from silly_kicks.gkdv._probe import MIN_DOMAIN_FRAMES
from scripts.build_tf19_instrument_responsiveness import pool_shards, reduce_layer_verdicts


def _shard(nrows):
    """A per-frame shard as the for_each map emits it: one row per scored frame, per arm."""
    return pd.DataFrame({
        "arm": ["delta_das"] * nrows,
        "realistic_abs": np.full(nrows, 0.05),
        "saturating_abs": np.full(nrows, 0.5),   # live instrument: sat >> 5x realistic
        "gk_abs": np.full(nrows, 0.30),          # Regime-I keeper delta
        "nd_abs": np.full(nrows, 0.10),          # paired-vector control delta (nd_med + placebo p95)
    })


def test_two_thin_shards_pool_to_valid_but_one_alone_is_unscoreable():
    half = MIN_DOMAIN_FRAMES // 2 + 1            # each shard < MIN_DOMAIN_FRAMES; two shards >= it
    pooled = pool_shards([_shard(half), _shard(half)])
    assert pooled["delta_das"]["n_domain"] >= MIN_DOMAIN_FRAMES
    assert reduce_layer_verdicts(pooled)["delta_das"]["layer0"] == "instrument_valid"
    # A per-shard implementation would see only `half` (< floor) -> arm_unscoreable. Non-vacuous:
    one = pool_shards([_shard(half)])
    assert one["delta_das"]["n_domain"] < MIN_DOMAIN_FRAMES
    assert reduce_layer_verdicts(one)["delta_das"]["layer0"] == "arm_unscoreable"
```

> Red-first discrimination: stub `pool_shards` to return per-shard (not concatenated) and watch the
> two-shard assert fall to `arm_unscoreable` — proving the test detects the failure it is named for.

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement the reduce (novel logic) + wire the driver.**

The novel pieces the test pins are the POOL (concatenate shards → per-arm pooled scalars) and the
reduce (pooled scalars → verdicts) — real code:

```python
# scripts/build_tf19_instrument_responsiveness.py
import numpy as np
import pandas as pd

from silly_kicks.gkdv._probe import layer0_instrument_verdict, layer1_responsiveness_verdict


def pool_shards(shards: list[pd.DataFrame]) -> dict:
    """Concatenate per-frame shards and pool per arm — the POOLED-corpus statistics the
    Layer-0/1 verdicts consume (never per shard). `nd_abs` is the paired-vector control delta:
    it supplies BOTH `nd_med` and the placebo p95."""
    if not shards:
        return {}
    df = pd.concat(shards, ignore_index=True)
    out = {}
    for arm, g in df.groupby("arm", dropna=False):
        nd = g["nd_abs"].to_numpy(float)
        out[str(arm)] = {
            "realistic_abs": g["realistic_abs"].to_numpy(float),
            "saturating_abs": g["saturating_abs"].to_numpy(float),
            "placebo_p95": float(np.nanpercentile(nd, 95)) if np.isfinite(nd).any() else np.nan,
            "gk_med": float(np.nanmedian(g["gk_abs"].to_numpy(float))),
            "nd_med": float(np.nanmedian(nd)),
            "n_domain": int(len(g)),
        }
    return out


def reduce_layer_verdicts(per_arm: dict) -> dict:
    """Pooled-corpus Layer-0/1 verdicts per arm. `per_arm[arm]` carries the already-pooled
    realistic_abs/saturating_abs arrays + placebo_p95/gk_med/nd_med scalars + n_domain."""
    out = {}
    for arm, s in per_arm.items():
        out[arm] = {
            "layer0": layer0_instrument_verdict(
                realistic_abs=s["realistic_abs"], saturating_abs=s["saturating_abs"],
                placebo_p95=s["placebo_p95"], n_domain=s["n_domain"],
            ),
            "layer1": layer1_responsiveness_verdict(
                gk_med=s["gk_med"], nd_med=s["nd_med"], placebo_p95=s["placebo_p95"],
                n_domain=s["n_domain"],
            ),
        }
    return out
```

Then the driver wiring (orchestration of already-specified seams — no new logic):
- `main()` calls `require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)` **FIRST**.
- **Roster transform (per provider):** for a native-identity provider (GS/tracking) `identity="native"`,
  `roster=None`. For SB360 the driver reduces `scripts/_sb_raw.py::parse_roster`
  (`{player_id: {name, jersey, team, position}}`) → `{team_id: gk_player_id}` by keeping
  `position == "Goalkeeper"` and keying by `team`, and passes `identity="roster", roster=<that dict>`.
- **Map:** `for_each(match_ids, key=..., work=_measure_match, shard_root=..., token_inputs={"schema": ...})`
  writes one per-frame shard per match. `_measure_match` computes, per arm, **both regimes** — Regime O
  at the `"realistic"` dose and Regime I at the `"ladder"` dose (`REGIME_I_LADDER_M = 2.0`) — running the
  arm batch on `(actual, imposed)` for the gk leg (`gk_abs`) and on `(actual, paired_vector_controls)` for
  the placebo/nearest-defender leg (`nd_abs`), plus the `"saturating_*"` doses for Layer 0
  (`saturating_abs`); it stamps each row's defending-keeper id via `resolve_defending_keeper_id` and the
  arm column via `expected_direction_for_arm`.
- **Reduce:** `pool_shards(all_shards)` → `reduce_layer_verdicts` (pooled Layer-0/1 per arm per regime);
  the per-arm `gate_eligible` census via `aggregate_by_keeper(obs, value_col=arm_col, min_nonzero=20,
  min_games=2)` for ΔDAS **and** ΔThreat separately; the named-keeper sign table +
  `behavioural_anchoring_verdict`. Stamp `run_commit`/`run_tree_dirty` into `verdicts.json`.

- [ ] **Step 4: Run — expect PASS.**

- [ ] **Step 5: Provenance + population tests**

```python
def test_driver_refuses_dirty_tree_and_offers_allow_dirty():
    import scripts.build_tf19_instrument_responsiveness as d
    assert hasattr(d, "main")
    # AST/behaviour checks mirror tests/scripts/test_provenance_wiring.py: main() calls
    # require_clean_tree, offers --allow-dirty, never shells out to `git rev-parse`.


def test_gate_eligible_census_binds_on_min_nonzero_for_das():
    # E2: a keeper in >=2 matches but <20 nonzero dDAS obs is NOT gate_eligible (min_nonzero binds).
    import pandas as pd
    from silly_kicks.gkdv import aggregate_by_keeper
    obs = pd.DataFrame({
        "player_id": [7] * 25,
        "game_id": [1] * 13 + [2] * 12,        # 2 distinct games -> clears min_games
        "delta_das": [0.0] * 20 + [-0.1] * 5,  # only 5 nonzero -> fails min_nonzero=20
    })
    row = aggregate_by_keeper(obs, value_col="delta_das", min_nonzero=20, min_games=2)
    row = row[row["player_id"] == 7].iloc[0]
    assert int(row["n_games"]) == 2 and int(row["n_nonzero"]) == 5
    assert not bool(row["gate_eligible"])       # min_nonzero binds, not min_games
```

Enroll the driver in `tests/scripts/_script_population.py` so the ADR-052 (`for_each`) and ADR-056 (artifact-provenance) population gates see it; run `python -m pytest tests/scripts/ -q` → PASS.

- [ ] **Step 6: Run the population + provenance suite — expect PASS.**

---

## Task 6: Docs, ADR (placeholder), CLAUDE.md, C4, gates

- [ ] **Step 1: Docstrings** — every new public function (`impose_defending_keeper_dose`, `layer0_instrument_verdict`, `layer1_responsiveness_verdict`, `paired_vector_controls`, `resolve_defending_keeper_id`, `expected_direction_for_arm`) gets a real Examples section (the `test_public_api_examples` gate; the ghost-numba cycle's `_FlatTrees` lesson — a public non-underscore name needs an Examples section or it fails the gate).

- [ ] **Step 2: ADR (PLACEHOLDER number)** — create `docs/superpowers/adrs/ADR-082-tf19-physics-arm-instrument-responsiveness.md` from the template: physics-arms-only scope; `arm_unscoreable` as a Chesterton's-Fence clearance (parent S1 not ported); `PHYSICS_ARM_PROBE_RATIO` a new registration; the caller-supplied identity seam (SB360 not foreclosed); reported-not-gated; the pooled-reduce convention. Leave the number `ADR-082` — renamed at commit-prep.

- [ ] **Step 3: CLAUDE.md** — extend the GKDV bullet: the physics arms now have Layer-0/Layer-1 probes (`gkdv/_probe.py`, reported-not-gated), `arm_unscoreable` verdict, `PHYSICS_ARM_PROBE_RATIO`, the identity-resolver seam, `_ARM_DIRECTION_KEY`; no gate flip, no retrain.

- [ ] **Step 4: NOTICE** unchanged (no new methodology). **C4** — no new action-coupled aggregator; run `python -m pytest tests/ -k c4 -q` (count unchanged); re-render only if flagged.

- [ ] **Step 5: DO NOT bump versions or write CHANGELOG.** Commit-prep only.

- [ ] **Step 6: Full CI-faithful gate**

```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest tests/ -m "not e2e" -v --tb=short
```

- [ ] **Step 7: /final-review + /c4**, then STOP and report. Do NOT commit. At commit-prep (owner-approved): `git fetch && git merge origin/main`, take the real NEXT-FREE version/PR-S/ADR, fill the version strings + rename the ADR, add the CHANGELOG entry, single commit, single PR.

---

## Self-review notes (author)

- **Spec coverage:** §4.1 imposer+Layer0 → Tasks 1-2; §4.2 Layer1+controls → Task 3; §4.3 identity resolver → Task 4; §4.4 arm-key + reach + named-keeper → Tasks 4-5; §4.5 driver map+reduce → Task 5; §5 tests → per-task (red-first, non-vacuous); §6 constraints → Global Constraints; §9 data-support matrix → the `arm_unscoreable` + resolver seams (Tasks 2/4).
- **Type consistency:** `impose_defending_keeper_dose`, `layer0_instrument_verdict`, `layer1_responsiveness_verdict`, `paired_vector_controls`, `resolve_defending_keeper_id`, `expected_direction_for_arm`, `_ARM_DIRECTION_KEY`, `PHYSICS_ARM_PROBE_RATIO`, `MIN_DOMAIN_FRAMES`, `SATURATING_MULTIPLE`, `R` named identically across tasks.
- **Two implementation-time seam checks the round-2 review flagged (carry into execution):** the `for_each`→pooled-verdict *reduce* (Task 5 Step 1 pins it) and the `_ARM_DIRECTION_KEY` resolution (Task 4 Step 1 pins it) — the two producer/consumer conventions a spec can't fully pin.
- **Open implementer notes (P1 data-sourcing):** Task 1 does NOT route through `provenance_to_targets`
  (its 7-col `_TARGET_COLUMNS` renames `ghost_x/y`→`target_x/y` and drops `actual_*`/`defended_goal_x`).
  `_build_dose_targets` sources `ghost_x/ghost_y/displacement_m/defending_team_id` from the `provenance`
  frame (`_PROVENANCE_COLUMNS`, `_engine.py:435`), `actual_x/actual_y` from `frames`' defending-GK rows,
  and `defended_goal_x` from `resolve_defended_goals` via `_engine._goal_lookup`. Before implementing:
  confirm `_goal_lookup` and `_same_team` are importable module-level from `_engine` (used at `:385`,
  `:442`) — promote or replicate if nested. Confirm `GkdvParams.ghost_keeps_actual_velocity` (`:112`,
  default True). Do NOT extend `provenance_to_targets`/`_TARGET_COLUMNS` (cross-package-pinned).
- **The returned `targets` contract is the Task 1 → Task 3 producer/consumer pair** (the round-3 seam
  the reviewer flagged for a targeted re-check): it carries `_FRAME_KEYS`, `defending_team_id`,
  `actual_x/actual_y`, `defended_goal_x`, `ghost_x/ghost_y`, `displacement_m`, and (post-dose)
  `imp_x/imp_y` — every column both the imposer and `paired_vector_controls` read.
- **No commit steps.** Single commit at commit-prep, owner-approved.
