# SB360 boundary-audit closeout (xtgk + gkdv) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** register `xtgk.compute_xt_gk_v2` and the three `gkdv` boundary entry points in the SB360 audit, empty `UNAUDITABLE_BOUNDARY`, and enforce the substantive-vs-structural verdict distinction WHERE IT IS MACHINE-CHECKABLE (`works`⇒structural, locking the frame-blind case), author-asserting the observationally-ambiguous `honest_nan` case (gkdv) with a mandatory rationale. The gate locks half the distinction; naming that ceiling is the durability contribution (see Task 2's Known limit).

**Architecture:** All work is in `tests/sb360/` + docs. Each boundary entry gets an INLINE adapter in `tests/sb360/_entries/_boundary.py` (exactly as `_call_restart_coordinates` already is), synthesizing its non-frame inputs (xtgk: deterministic velocity-blind port doubles; gkdv: an adapter-local possession `carrier` + per-action projection via the harness's own `links`). Observations are TRANSCRIBED FROM EXECUTION via the lock test, never guessed.

**Tech Stack:** pytest, pandas, numpy; `silly_kicks.xtgk`, `silly_kicks.gkdv`, `silly_kicks.tracking`; the `[das]` extra (accessible-space) already in `[test]`.

**Design doc:** `docs/superpowers/specs/2026-08-19-sb360-boundary-audit-xtgk-gkdv-design.md` (approved, 2 reviews).

## Global Constraints

- **Test-registry + docs ONLY.** No public library signature change; no `silly_kicks/` source edit except the version bump.
- **Observations are TRANSCRIBED FROM EXECUTION.** Register each entry with the predicted `AxisVerdict.observation`, run `tests/sb360/test_axis_locks.py::test_observations_match_the_registry` for that entry, and if its failure message reports a different observation, update the entry to the OBSERVED value. Never leave a guessed observation.
- **Boundary adapters are INLINE in `tests/sb360/_entries/_boundary.py`** (like `_call_restart_coordinates`). They are NOT part of `scripts/_sb_battery.py`'s `ADAPTER_MAP` (which serves the `tracking.__all__` `add_*` surface the regenerator loops). Shared helpers (`audit_xt`) are imported from `scripts._sb_battery` — the `tests -> scripts` layering `tests/sb360/_calls.py` already uses.
- **No perturbation of shared-fixture player positions.** gkdv possession comes from reshaping the fixture's OWN `team_in_possession` column into a `carrier`; xtgk augments only `actions` (adds a `pressure` column), deterministically from `actions`.
- **No new library dependency.** `[das]` (accessible-space) is already declared in `[test]` (`pyproject.toml`), and `add_das` is already an audited entry, so `gkdv.delta_das` produces a real verdict in CI.
- **Each registration task removes its entry from `UNAUDITABLE_BOUNDARY` in the SAME change** (keeps `test_uncovered_boundary_points_each_carry_a_reason` green). The task registering the LAST boundary entry (Task 5) also removes the strict xfail and bumps `NOT_EXERCISED_BUDGET`, so the suite ends green at every task boundary.
- **Match each file's existing ASCII/comment style.**
- **Version number is NOT claimed up front** — set at completion to the next-available from `main` (4.88.0 as of 4.87.0). Five sites: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`. Hand-edit `uv.lock`'s self-version line — do NOT run `uv lock` (fails on `main`: scikit-learn 1.9.0 vs requires-python>=3.10).
- **C4-free** (no new action-coupled `add_*`, backend, or model) — no `architecture.html` change.
- **Verification per task:** `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -v`. Final: full `python -m pytest tests/ -m "not e2e" --benchmark-skip` + `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check ...` + `python -m pyright`.

---

### Task 1: `verdict_provenance` scaffolding (vocabulary + field + `add_restart_coordinates`)

**Files:**
- Modify: `tests/sb360/_vocabulary.py`
- Modify: `tests/sb360/_registry.py` (`Sb360Entry`, `_entry`)
- Modify: `tests/sb360/_entries/_boundary.py` (declare on the existing entry)
- Test: `tests/sb360/test_registry_surface.py`

**Interfaces:**
- Produces: `VERDICT_PROVENANCE: frozenset[str]`; `Sb360Entry.verdict_provenance: str | None`, `Sb360Entry.provenance_rationale: str | None`; `_entry(..., verdict_provenance=None, provenance_rationale=None)`.

- [ ] **Step 1: Write the failing test** (append to `tests/sb360/test_registry_surface.py`)

```python
def test_verdict_provenance_vocabulary_and_restart_declaration():
    from tests.sb360._vocabulary import VERDICT_PROVENANCE

    assert VERDICT_PROVENANCE == frozenset({"substantive", "structural"})
    entry = SB360_ENTRIES["spadl.add_restart_coordinates"]
    assert entry.verdict_provenance == "structural"
    assert entry.provenance_rationale, "a structural boundary entry needs a stated reason"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/sb360/test_registry_surface.py::test_verdict_provenance_vocabulary_and_restart_declaration -q`
Expected: FAIL (`ImportError: cannot import name 'VERDICT_PROVENANCE'`).

- [ ] **Step 3: Add the vocabulary** (`tests/sb360/_vocabulary.py`, after the `ADJUDICATIONS`/`ADMISSIBLE_FROM` block)

```python
# --- verdict_provenance.* -------------------------------------------------------------
#: Whether a BOUNDARY entry's verdict is SUBSTANTIVE (a velocity-consuming function whose own
#: handling moved the value) or STRUCTURAL (a function the axes cannot substantively reach:
#: frame-blind -> `identical`; downstream-of-a-refusing-seam -> `honest_nan`). Scoped to
#: BOUNDARY_ENTRY_POINTS so an empty UNAUDITABLE_BOUNDARY is not misread as end-to-end coverage
#: (ADR-053 amendment / spec Part 4).
VERDICT_PROVENANCE: frozenset[str] = frozenset({"substantive", "structural"})
```

- [ ] **Step 4: Add the fields** (`tests/sb360/_registry.py`)

On `Sb360Entry` (after `structurally_impossible`):

```python
    #: Boundary-entry provenance (substantive/structural). None on the add_* surface. See Part 4.
    verdict_provenance: str | None = None
    provenance_rationale: str | None = None
```

On `_entry(...)`: add `verdict_provenance=None, provenance_rationale=None` to the signature and pass both into `Sb360Entry(...)`.

- [ ] **Step 5: Declare on `add_restart_coordinates`** (`tests/sb360/_entries/_boundary.py`, inside the existing `_entry(...)` call)

```python
    verdict_provenance="structural",
    provenance_rationale=(
        "Reads no velocity-sensitive input -- ADR-025 imputes restart coordinates from the "
        "action's own geometry -- so both legs observe `identical`. A frame-coupling regression "
        "tripwire, not degradation coverage."
    ),
```

- [ ] **Step 6: Run the test + the sb360 suite, verify green**

Run: `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -q`
Expected: PASS (no gate yet, so no unregistered-boundary failure).

---

### Task 2: The provenance meta-gate

**Files:**
- Test: `tests/sb360/test_registry_surface.py`

**Interfaces:**
- Consumes: `VERDICT_PROVENANCE`, `Sb360Entry.verdict_provenance/provenance_rationale`, `iter_verdicts`, `BOUNDARY_ENTRY_POINTS`.

- [ ] **Step 1: Write the gate** (append to `tests/sb360/test_registry_surface.py`)

```python
def test_boundary_entries_declare_admissible_provenance():
    """Every REGISTERED boundary entry declares substantive/structural, admissibly from its
    observation, so an empty UNAUDITABLE_BOUNDARY cannot be misread as end-to-end degradation
    coverage (ADR-053 Part 4). Population derived from BOUNDARY_ENTRY_POINTS -- a new boundary
    entry without a provenance fails here.

    KNOWN LIMIT (spec Part 4): this gate locks HALF the distinction. `works`=>`structural` is tight
    (a value that cannot move was not substantively handled). `differs_by_design`/`silent_degrade`
    =>`substantive` is enforceable but inert this cycle. But `honest_nan` is OBSERVATIONALLY
    AMBIGUOUS -- self-refusal (substantive) and inherited-refusal (structural, gkdv) both produce
    `all_nan`, so the gate CANNOT check gkdv's `structural` choice; it is author-asserted, forced
    only to carry a rationale. This is the machine-checkability ceiling, named deliberately.

    Cannot be landed red against the correct Task-1 state (add_restart_coordinates is already
    `structural`+`works`), so it was MUTATION-VERIFIED (ADR-051), both admissibility branches: see
    Step 2.
    """
    from tests.sb360._registry import BOUNDARY_ENTRY_POINTS
    from tests.sb360._vocabulary import VERDICT_PROVENANCE

    for name in sorted(set(BOUNDARY_ENTRY_POINTS) & set(SB360_ENTRIES)):
        entry = SB360_ENTRIES[name]
        prov = entry.verdict_provenance
        assert prov in VERDICT_PROVENANCE, (
            f"{name}: registered boundary entry carries verdict_provenance {prov!r}, not in "
            f"{sorted(VERDICT_PROVENANCE)}. Declare substantive/structural (spec Part 4)."
        )
        for _axis, _roster, col, v in iter_verdicts(entry):
            if v.adjudication == "works":
                assert prov == "structural", (
                    f"{name}.{col}: `works` (from `identical`) forces `structural` -- a value that "
                    f"cannot move across the velocity legs was not substantively handled. Got {prov!r}."
                )
            if v.adjudication in {"differs_by_design", "silent_degrade"}:
                assert prov == "substantive", (
                    f"{name}.{col}: {v.adjudication!r} forces `substantive` -- the value moved "
                    f"because of the function. Got {prov!r}."
                )
        if prov == "structural":
            assert entry.provenance_rationale, (
                f"{name}: `structural` requires a non-empty provenance_rationale naming WHY "
                f"(frame-blind / inherited-from-refusal)."
            )
```

- [ ] **Step 2: Mutation-verify RED — BOTH admissibility branches**

Branch A (`works`⇒`structural`): temporarily change `add_restart_coordinates`'s `verdict_provenance`
to `"substantive"`, run the gate, confirm it FAILS on `works` => `structural`, then RESTORE
`"structural"`.

Branch B (`differs`/`silent_degrade`⇒`substantive`) is forward-looking scaffolding with no live case
this cycle, so pin it by mutation too, or it could silently break before the first substantive
boundary entry arrives: temporarily set ONE verdict of a `structural` entry to
`AxisVerdict("differs", "differs_by_design")` (e.g. edit the `add_restart_coordinates` velocity block
in-place), run the gate, confirm it FAILS on `differs_by_design` => `substantive` (the entry is still
declared `structural`), then RESTORE. (Do this AFTER Task 1's `add_restart_coordinates` declaration
exists; it needs no other boundary entry.)

Run for each: `python -m pytest tests/sb360/test_registry_surface.py::test_boundary_entries_declare_admissible_provenance -q`

- [ ] **Step 3: Run green**

Run: `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -q`
Expected: PASS.

---

### Task 3: `xtgk.compute_xt_gk_v2` adapter + registration

**Files:**
- Modify: `tests/sb360/_entries/_boundary.py` (doubles + adapter + entry)
- Modify: `tests/sb360/test_registry_surface.py` (remove from `UNAUDITABLE_BOUNDARY`)
- Test: `tests/sb360/test_boundary_adapters.py` (new — liveness)

**Interfaces:**
- Consumes: `silly_kicks.xtgk.compute_xt_gk_v2`, `PressureLevels`, `State`, `DeltaV`.
- Produces: entry `xtgk.compute_xt_gk_v2` emitting `xt_gk_v2_position`, `xt_gk_v2_pev`, `xt_gk_v2_retention_loss`, `xt_gk_v2_dzv`, `xt_gk_v2`.

- [ ] **Step 1: Write the liveness test FIRST** (`tests/sb360/test_boundary_adapters.py`, new file)

```python
"""Non-vacuity guards for the boundary-entry adapters (spec Part 1/Part 2 + Non-vacuity section)."""
from __future__ import annotations

import numpy as np

from tests.sb360 import _fixture as F
from tests.sb360._registry import SB360_ENTRIES


def test_xt_gk_v2_columns_are_live_so_identical_is_a_real_comparison():
    """The `identical`->`works` verdict must be a real number comparison, not NaN==NaN. Every
    column is finite; the four terms the doubles drive are non-constant. `xt_gk_v2_pev` is EXEMPT:
    it is 0 by construction in the base metric (p'=p), documented in _metric.py."""
    entry = SB360_ENTRIES["xtgk.compute_xt_gk_v2"]
    actions, frames, links = F.build_leg_b()  # velocity-bearing leg; frame-blind fn ignores it
    out = entry.call(actions, frames, links, F.HOME_TEAM_ID)
    for col in ("xt_gk_v2_position", "xt_gk_v2_pev", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"):
        vals = out[col].to_numpy(dtype=float)
        assert np.isfinite(vals).all(), f"{col} has non-finite values: {vals}"
    for col in ("xt_gk_v2_position", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"):
        assert np.unique(out[col].to_numpy(dtype=float)).size > 1, f"{col} is constant -- doubles not live"
    assert np.allclose(out["xt_gk_v2_pev"].to_numpy(dtype=float), 0.0), "pev is 0 by construction (p'=p)"
```

- [ ] **Step 2: Run it, verify it fails**

Run: `python -m pytest tests/sb360/test_boundary_adapters.py -q`
Expected: FAIL (`KeyError: 'xtgk.compute_xt_gk_v2'` — entry not registered yet).

- [ ] **Step 3: Add the port doubles + adapter** (`tests/sb360/_entries/_boundary.py`)

```python
import numpy as np
import pandas as pd

from silly_kicks.xtgk import DeltaV, PressureLevels, State, compute_xt_gk_v2


def _xt_gk_pressure_levels() -> PressureLevels:
    # Cutpoints (1.0, 2.0) split the synthetic pressure {0.5, 1.5, 2.5} into levels {1, 2, 3}.
    return PressureLevels.from_cutpoints((1.0, 2.0))


class _XtGkPossessionValueDouble:
    """Deterministic, velocity-blind PossessionValue double. Monotone in zone so position/dzv are
    LIVE; reads only (zone, pressure), never a frame, so it cannot mask a velocity dependence."""

    def __init__(self, pressure_levels: PressureLevels) -> None:
        self.pressure_levels = pressure_levels

    def value(self, zone: int, p: int) -> float:
        return 0.001 * (int(zone) + 1) * (1.0 + 0.1 * int(p))

    def surface(self, p: int):  # part of the Protocol; unused by the per-action loop
        return np.zeros((1, 1), dtype=float)

    def delta_v(self, s: State, s_next: State) -> DeltaV:
        pos = self.value(s_next.zone, s.pressure_level) - self.value(s.zone, s.pressure_level)
        return DeltaV(delta=pos, pressure_component=0.0, position_component=pos)  # p'=p -> pev 0


class _XtGkRetentionDouble:
    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        f = features["feat"].to_numpy(dtype=float)
        return 0.35 + 0.30 * (np.sin(f) * 0.5 + 0.5)  # deterministic, in (0.35, 0.65), non-constant


class _XtGkTurnoverDouble:
    def value(self, zone: int, p: int) -> float:
        return 0.0005 * (int(zone) + 1) * (1.0 + 0.05 * int(p))


def _call_xt_gk_v2(actions, frames, links, home_team_id):
    """Frame-blind: `frames`/`links`/`home_team_id` are ignored (compute_xt_gk_v2 reads only
    `actions` + injected ports). Both legs share identical `actions`, so the output is identical
    by construction -- a frame-coupling tripwire (spec Part 1)."""
    a = actions.copy()
    # Positional index, NOT action_id-cast: dtype-agnostic (ADR-019 consistency with the gkdv
    # sibling; immune if the harness ever grows an id_dtype axis).
    idx = np.arange(len(a), dtype=float)
    a["pressure"] = 0.5 + (idx % 3)  # 0.5/1.5/2.5 -> levels 1/2/3
    pl = _xt_gk_pressure_levels()
    rf = pd.DataFrame({"feat": idx}, index=a.index)
    return compute_xt_gk_v2(
        a,
        possession_value=_XtGkPossessionValueDouble(pl),
        retention=_XtGkRetentionDouble(),
        turnover_cost=_XtGkTurnoverDouble(),
        pressure_levels=pl,
        retention_features=rf,
    )
```

- [ ] **Step 4: Register the entry** (`tests/sb360/_entries/_boundary.py`)

```python
_XTGK_V2_COLS = (
    "xt_gk_v2_position",
    "xt_gk_v2_pev",
    "xt_gk_v2_retention_loss",
    "xt_gk_v2_dzv",
    "xt_gk_v2",
)

_entry(
    "xtgk.compute_xt_gk_v2",
    _call_xt_gk_v2,
    columns=_XTGK_V2_COLS,
    velocity={c: AxisVerdict("identical", "works") for c in _XTGK_V2_COLS},
    visibility={
        r: {c: AxisVerdict("identical", "works") for c in _XTGK_V2_COLS}
        for r in ("gk_absent", "defender_absent", "gk_one_end")
    },
    applicability={c: "no_support" for c in _XTGK_V2_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _XTGK_V2_COLS},
    verdict_provenance="structural",
    provenance_rationale=(
        "Frame-blind: compute_xt_gk_v2 reads `actions` + injected ports, never a frame, so both "
        "legs observe `identical`. `works` means it fabricates nothing through a frame it never "
        "reads -- NOT that xt_gk_v2 is velocity-robust or SB360-computable (its velocity-dependence "
        "lives in its inputs: pressure/is_gk_distribution/retention_features, computed upstream from "
        "tracking, unavailable on real SB360). A frame-coupling regression tripwire. ADR-053 Part 4."
    ),
)
```

- [ ] **Step 5: Remove `xtgk.compute_xt_gk_v2` from `UNAUDITABLE_BOUNDARY`** (`tests/sb360/test_registry_surface.py`) — delete its dict entry.

- [ ] **Step 6: Transcribe observations + run the gates**

Run: `python -m pytest "tests/sb360/test_axis_locks.py::test_observations_match_the_registry[xtgk.compute_xt_gk_v2-velocity-full]" tests/sb360/test_axis_locks.py -k xtgk -q`
If any observation disagrees, update that `AxisVerdict` to the OBSERVED value (the failure message prints it) and re-run.
Then run the full sb360 suite: `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -q`
Expected: PASS. The strict xfail `test_every_boundary_entry_point_is_registered` still xfails (3 gkdv names remain unregistered).

---

### Task 4: `gkdv.build_ghost_frames` adapter helper + registration

**Files:**
- Modify: `tests/sb360/_entries/_boundary.py` (gkdv helpers + entry)
- Modify: `tests/sb360/test_registry_surface.py` (remove from `UNAUDITABLE_BOUNDARY`)
- Modify: `tests/sb360/_registry.py` (`NOT_EXERCISED_BUDGET` incremental bump — see Step 6)
- Test: `tests/sb360/test_boundary_adapters.py` (gkdv asymmetry)

**Interfaces:**
- Consumes: `silly_kicks.gkdv.build_ghost_frames`, `GkdvParams`; `silly_kicks.id_compat.{ids_equal, canonical_id}`.
- Produces: `_gkdv_scored(frames, home_team_id) -> (cf, prov)`; `_gkdv_per_action(frames, cf, prov, links, actions, arm_fn) -> pd.Series`; entry `gkdv.build_ghost_frames` emitting `ghost_x`, `ghost_y`, `displacement_m`.

- [ ] **Step 1: Write the asymmetry test FIRST** (`tests/sb360/test_boundary_adapters.py`)

```python
def test_gkdv_build_ghost_frames_is_live_asymmetric_across_legs():
    """Leg A (freeze-frame) serves no ghost (ADR-054 refusal) -> all NaN; Leg B scores the
    in-domain actions -> finite. That asymmetry is the honest_nan signal (spec Part 2)."""
    entry = SB360_ENTRIES["gkdv.build_ghost_frames"]
    a_out = entry.call(*F.build_leg_a(), F.HOME_TEAM_ID)
    b_out = entry.call(*F.build_leg_b(), F.HOME_TEAM_ID)
    assert not np.isfinite(a_out["displacement_m"].to_numpy(dtype=float)).any(), "Leg A must be all-NaN"
    assert np.isfinite(b_out["displacement_m"].to_numpy(dtype=float)).any(), "Leg B must score >=1 action"
```

(Note: `F.build_leg_a()` returns `(actions, frames, links)`; the `*` unpacks them, then `F.HOME_TEAM_ID` is the 4th arg.)

- [ ] **Step 2: Run it, verify it fails** — `KeyError: 'gkdv.build_ghost_frames'`.

- [ ] **Step 3: Add the gkdv helpers** (`tests/sb360/_entries/_boundary.py`)

```python
from silly_kicks.gkdv import GkdvParams, build_ghost_frames
from silly_kicks.id_compat import canonical_id, ids_equal, same_id

_GK_FK = ["game_id", "period_id", "frame_id"]


def _gkdv_scored(frames, home_team_id):
    """Run build_ghost_frames with the fixture's OWN possession (no shared-position change) and
    possession_stride=1 (score every eligible frame so anchor frames are never stride-dropped)."""
    players = frames[~frames["is_ball"].astype(bool)]
    carrier = (
        players[[*_GK_FK, "team_in_possession"]]
        .drop_duplicates(subset=_GK_FK)
        .rename(columns={"team_in_possession": "ball_carrier_team_id"})
        .reset_index(drop=True)
    )
    cf, prov, _ = build_ghost_frames(
        frames, home_team_id=home_team_id, carrier=carrier, params=GkdvParams(possession_stride=1)
    )
    return cf, prov


def _gkdv_per_action(frames, cf, prov, links, actions, arm_fn):
    """Project a per-frame arm to per-action: the arm value at the action's ANCHOR frame if that
    frame was scored, else NaN. Frame ids matched canonically (ADR-019)."""
    scored = prov[prov["drop_reason"].isna()]
    scored_ids = {canonical_id(f) for f in scored["frame_id"]}
    anchor = {canonical_id(aid): fid for aid, fid in zip(links["action_id"], links["frame_id"])}
    team = {canonical_id(aid): tid for aid, tid in zip(actions["action_id"], actions["team_id"])}
    vals = []
    for aid in actions["action_id"]:
        key = canonical_id(aid)
        fid = anchor.get(key)
        if fid is None or pd.isna(fid) or canonical_id(fid) not in scored_ids:
            vals.append(np.nan)
            continue
        actual = frames[ids_equal(frames["frame_id"], pd.Series(fid, index=frames.index))]
        ghost = cf[ids_equal(cf["frame_id"], pd.Series(fid, index=cf.index))]
        vals.append(float(arm_fn(actual, ghost, team[key])))
    return pd.Series(vals, index=actions.index)
```

- [ ] **Step 4: Add the `build_ghost_frames` adapter + entry** (`tests/sb360/_entries/_boundary.py`)

```python
_GKDV_STRUCTURAL_RATIONALE = (
    "Inherited from serve_ghost_gk_positions, which REFUSES velocity-less freeze-frames "
    "(ADR-054): Leg A scores zero frames -> NaN, Leg B scores the in-domain actions. Same "
    "`honest_nan` class as add_ghost_gk. The arms are never reached on Leg A, so their intrinsic "
    "zero-velocity behaviour is out of scope and contingent on that refusal (cf. ADR-063). ADR-053 Part 4."
)


def _call_gkdv_build_ghost_frames(actions, frames, links, home_team_id):
    cf, prov = _gkdv_scored(frames, home_team_id)
    scored = prov[prov["drop_reason"].isna()]
    defending = scored[ids_equal(scored["gk_team_id"], scored["defending_team_id"])]

    def _arm(actual, ghost, _team):  # arm_fn shape; reads the defending keeper's provenance row
        target = actual["frame_id"].iloc[0]
        row = defending[[bool(same_id(f, target)) for f in defending["frame_id"]]]
        if not len(row):
            return np.nan
        return float(row["displacement_m"].iloc[0])

    disp = _gkdv_per_action(frames, cf, prov, links, actions, _arm)
    # ghost_x/ghost_y follow the same projection.
    def _xy(colname):
        def arm(actual, ghost, _team):
            target = actual["frame_id"].iloc[0]
            row = defending[[bool(same_id(f, target)) for f in defending["frame_id"]]]
            return float(row[colname].iloc[0]) if len(row) else np.nan
        return _gkdv_per_action(frames, cf, prov, links, actions, arm)

    return actions.assign(
        ghost_x=_xy("ghost_x").to_numpy(),
        ghost_y=_xy("ghost_y").to_numpy(),
        displacement_m=disp.to_numpy(),
    )


_GKDV_BGF_COLS = ("ghost_x", "ghost_y", "displacement_m")
_entry(
    "gkdv.build_ghost_frames",
    _call_gkdv_build_ghost_frames,
    columns=_GKDV_BGF_COLS,
    velocity={c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
    visibility={
        "gk_absent": {c: AxisVerdict("no_signal", "not_exercised", rationale=_GKDV_STRUCTURAL_RATIONALE) for c in _GKDV_BGF_COLS},
        "defender_absent": {c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
        "gk_one_end": {c: AxisVerdict("all_nan", "honest_nan") for c in _GKDV_BGF_COLS},
    },
    applicability={c: "no_support" for c in _GKDV_BGF_COLS},
    applicability_deltas={c: {"extreme": 0.0, "near": 0.0} for c in _GKDV_BGF_COLS},
    verdict_provenance="structural",
    provenance_rationale=_GKDV_STRUCTURAL_RATIONALE,
)
```

(`not_exercised` requires a per-verdict rationale — `RATIONALE_ALWAYS` — hence the `rationale=` on the gk_absent verdicts.)

- [ ] **Step 5: Remove `gkdv.build_ghost_frames` from `UNAUDITABLE_BOUNDARY`.**

- [ ] **Step 6: Transcribe observations, bump the budget, run**

Run: `python -m pytest tests/sb360/test_axis_locks.py -k build_ghost_frames tests/sb360/test_boundary_adapters.py -m "not e2e" --benchmark-skip -q`
Transcribe any disagreeing observation (velocity/full, defender_absent, gk_one_end, gk_absent) from the failure message and update. Re-run.

**Bump `NOT_EXERCISED_BUDGET` (`tests/sb360/_registry.py`) — REQUIRED to end green.** This task adds `not_exercised` verdicts (build_ghost_frames's 3 columns on the `gk_absent` roster), so `test_not_exercised_count_is_within_its_locked_budget` fails until the budget is updated. Run `python -m pytest tests/sb360/test_registry_surface.py::test_not_exercised_count_is_within_its_locked_budget -m "not e2e" --benchmark-skip -q`; it reports the ACTUAL count. Set `NOT_EXERCISED_BUDGET` to that value (predicted 49 -> 52) and extend its explanatory comment with the gkdv reason (gk_absent removes both keepers; gkdv requires a defending GK, so both legs score zero -> no_signal on these columns). Task 5 bumps it once more for delta_das + delta_threat.

Then: `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -q`
Expected: PASS. The strict xfail still xfails (`delta_das`, `delta_threat_suppression` remain).

---

### Task 5: `gkdv.delta_das` + `gkdv.delta_threat_suppression`, then retire the category

**Files:**
- Modify: `tests/sb360/_entries/_boundary.py` (two adapters + entries)
- Modify: `tests/sb360/test_registry_surface.py` (empty `UNAUDITABLE_BOUNDARY`; remove the xfail)
- Modify: `tests/sb360/_registry.py` (`NOT_EXERCISED_BUDGET`)
- Test: `tests/sb360/test_boundary_adapters.py`

**Interfaces:**
- Consumes: `silly_kicks.gkdv.{delta_das, delta_threat_suppression}`, `silly_kicks.tracking.resolve_defended_goals`, `scripts._sb_battery.audit_xt`.

- [ ] **Step 1: Write the arm asymmetry test FIRST** (`tests/sb360/test_boundary_adapters.py`)

```python
def test_gkdv_arms_are_live_asymmetric_across_legs():
    for name, col in (("gkdv.delta_das", "delta_das"),
                      ("gkdv.delta_threat_suppression", "delta_threat_suppression")):
        entry = SB360_ENTRIES[name]
        a = entry.call(*F.build_leg_a(), F.HOME_TEAM_ID)[col].to_numpy(dtype=float)
        b = entry.call(*F.build_leg_b(), F.HOME_TEAM_ID)[col].to_numpy(dtype=float)
        assert not np.isfinite(a).any(), f"{name}: Leg A must be all-NaN (ghost refusal)"
        assert np.isfinite(b).any(), f"{name}: Leg B must score >=1 action"
```

- [ ] **Step 2: Run, verify it fails** — `KeyError`.

- [ ] **Step 3: Add the two adapters + entries** (`tests/sb360/_entries/_boundary.py`)

```python
def _call_gkdv_delta_das(actions, frames, links, home_team_id):
    from silly_kicks.gkdv import delta_das

    cf, prov = _gkdv_scored(frames, home_team_id)
    vals = _gkdv_per_action(
        frames, cf, prov, links, actions,
        lambda actual, ghost, tid: delta_das(actual, ghost, attacking_team_id=tid),
    )
    return actions.assign(delta_das=vals.to_numpy())


def _call_gkdv_delta_threat(actions, frames, links, home_team_id):
    from silly_kicks.gkdv import delta_threat_suppression
    from silly_kicks.tracking import resolve_defended_goals

    from scripts._sb_battery import audit_xt

    cf, prov = _gkdv_scored(frames, home_team_id)
    goal_map = resolve_defended_goals(frames)  # byte-identical to the engine's _pin_defended_goal
    xt = audit_xt()
    vals = _gkdv_per_action(
        frames, cf, prov, links, actions,
        lambda actual, ghost, tid: delta_threat_suppression(
            actual, ghost, attacking_team_id=tid, xt=xt, goal_map=goal_map
        ),
    )
    return actions.assign(delta_threat_suppression=vals.to_numpy())


for _name, _fn, _col in (
    ("gkdv.delta_das", _call_gkdv_delta_das, "delta_das"),
    ("gkdv.delta_threat_suppression", _call_gkdv_delta_threat, "delta_threat_suppression"),
):
    _entry(
        _name,
        _fn,
        columns=(_col,),
        velocity={_col: AxisVerdict("all_nan", "honest_nan")},
        visibility={
            "gk_absent": {_col: AxisVerdict("no_signal", "not_exercised", rationale=_GKDV_STRUCTURAL_RATIONALE)},
            "defender_absent": {_col: AxisVerdict("all_nan", "honest_nan")},
            "gk_one_end": {_col: AxisVerdict("all_nan", "honest_nan")},
        },
        applicability={_col: "no_support"},
        applicability_deltas={_col: {"extreme": 0.0, "near": 0.0}},
        verdict_provenance="structural",
        provenance_rationale=_GKDV_STRUCTURAL_RATIONALE,
    )
```

- [ ] **Step 4: Empty `UNAUDITABLE_BOUNDARY`** — remove `delta_das` + `delta_threat_suppression`; the dict is now `{}`. Keep the symbol (an empty dict), so `test_uncovered_boundary_points_each_carry_a_reason` still guards future additions.

- [ ] **Step 5: Retire the strict xfail** (`tests/sb360/test_registry_surface.py`)

Remove `@pytest.mark.xfail(reason=..., strict=True)` from `test_every_boundary_entry_point_is_registered`; it now passes as a plain completeness assertion. Update its docstring/message (drop "structurally out of reach"; state that every boundary entry is now registered and a new one must register or CI fails).

- [ ] **Step 6: Transcribe observations, then bump `NOT_EXERCISED_BUDGET`**

Run: `python -m pytest tests/sb360/test_axis_locks.py -k "delta_das or delta_threat" tests/sb360/test_boundary_adapters.py -m "not e2e" --benchmark-skip -q` and transcribe any disagreeing observations.
Then run `python -m pytest tests/sb360/test_registry_surface.py::test_not_exercised_count_is_within_its_locked_budget -m "not e2e" --benchmark-skip -q`; it reports the ACTUAL not_exercised count. Set `NOT_EXERCISED_BUDGET` in `_registry.py` to that value — this is the SECOND, FINAL bump (Task 4 already took it 49 -> 52 for build_ghost_frames; this task adds delta_das + delta_threat's 2 columns on `gk_absent`, predicted **52 -> 54**). The gkdv reason comment was added in Task 4; extend it if needed to cover all 5 gkdv columns.

- [ ] **Step 7: Confirm the dark-column set is unchanged**

Run: `python -m pytest tests/sb360/test_registry_surface.py::test_no_column_is_unexercised_on_every_roster_except_the_recorded_ones -q`
Expected: PASS unchanged — gkdv columns are `honest_nan` (exercised) on `defender_absent` + `gk_one_end`, `not_exercised` only on `gk_absent`, so none is dark-on-every-roster and `_EXPECTED_DARK_COLUMNS` needs no edit. If it fails, STOP and re-examine (a gkdv column dark everywhere would mean Leg B scored nothing on a keeper-present roster).

- [ ] **Step 8: Full sb360 suite green**

Run: `python -m pytest tests/sb360/ -m "not e2e" --benchmark-skip -q`
Expected: PASS, including `test_every_boundary_entry_point_is_registered` (now a plain pass) and `test_boundary_entries_declare_admissible_provenance` covering all 5 boundary entries.

---

### Task 6: Documentation + version

**Files:**
- Modify: `docs/superpowers/adrs/ADR-053-*.md` (the SB360 audit ADR — add the amendment)
- Modify: `CHANGELOG.md`, `TODO.md`
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`

- [ ] **Step 1: ADR-053 amendment.** Append the boundary-entry policy from the spec's "ADR-053 amendment" section: the substantive-vs-structural/inherited distinction; frame-blind injected-port orchestrators audited via synthesize-and-inject (citing `audit_xt`/`visible_area_coverage`); `works` on a frame-blind orchestrator means "fabricates nothing through a frame it never reads", not velocity-robust/SB360-computable; downstream-of-refusal inherits (contingent); the distinction is TEST-LOCKED via `verdict_provenance` + the meta-gate.

- [ ] **Step 2: Delete the stale TODO item.** Remove the "Four boundary entry points are unaudited ..." bullet from `TODO.md`'s "From the SB360 coverage audit (ADR-053)" section (it is now false — all are registered). Update the `TODO.md` Release line to this cycle.

- [ ] **Step 3: CHANGELOG entry.** Add a `## [4.88.0]` entry (keyed `PR-Snnn`, next in sequence): boundary-audit closeout — `xtgk.compute_xt_gk_v2` + the 3 gkdv boundary points registered; `UNAUDITABLE_BOUNDARY` emptied; strict xfail retired; `verdict_provenance` field + meta-gate; ADR-053 amendment. Test-registry + docs only; **no library change, no retrain, C4-free**.

- [ ] **Step 4: Version bump (LAST).** Set the next-available version from `main` (4.88.0 as of 4.87.0 — re-confirm `main` has not advanced) in `pyproject.toml` and `silly_kicks/__init__.py`, and hand-edit the self-version line in `uv.lock` (do NOT run `uv lock`). The CHANGELOG + TODO already carry it from Steps 2-3.

- [ ] **Step 5: Full verification.**

Run: `python -m pytest tests/ -m "not e2e" --benchmark-skip` (expect all green; run on the pandas-2 and pandas-3 venvs if available — the observations are classifications and should be version-stable, but confirm).
Run: `python -m ruff check silly_kicks/ tests/ scripts/` and `python -m ruff format --check silly_kicks/ tests/ scripts/` and `python -m pyright`.
Expected: clean. Then STOP — the user reviews and commits once.

---

## Self-review notes (author)

- **Spec coverage:** Part 1 (xtgk) -> Task 3; Part 2 (gkdv) -> Tasks 4-5; Part 3 (retire category) -> Task 5 steps 4-6; Part 4 (verdict_provenance) -> Tasks 1-2 + declarations threaded through 3-5; ADR amendment + TODO + version -> Task 6.
- **Green at every task boundary:** each registration task removes its entry from `UNAUDITABLE_BOUNDARY` (keeps `test_uncovered...` green); the xfail flip + budget bump are in Task 5, the task that registers the last entry (before which the xfail still xfails).
- **Transcribe-from-execution:** every observation is registered as a prediction and reconciled against the lock test's reported value; the predicted tables are hypotheses, not asserted truth.
- **No commit steps** anywhere; the user commits once at the end.
