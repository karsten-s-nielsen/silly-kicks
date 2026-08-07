# SB360 honest degradation + StatsBomb parse port — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax
> for tracking.

**Goal:** The ghost-GK path refuses honestly on freeze-frames instead of fabricating, and
`providers/statsbomb` turns StatsBomb 360 payloads into the `snapshot_to_tracking_frames` contract.

**Architecture:** Commit 1 puts one refusal at the shared serving seam `_serve_positions_core`, so all
three public ghost entry points inherit it, and adds a `validate_velocity_regime` diagnostic for the
five aggregators whose interpretation (not value) changes without velocity. Commit 2 EXTRACTS the port
from `scripts/build_sb360_coverage.py`, which already implements most of the parse half, and re-points
the script at it.

**Tech Stack:** Python 3.10–3.14, pandas, numpy, pytest. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-08-05-sb360-degradation-and-statsbomb-port-design.md` (rev 4)

## Global Constraints

* **Branch `sb360-degradation-and-port`. Two commits. Merge with `--merge`, NEVER squash.**
* **ADR number and version are read off `main` at commit-prep, never assumed.** `main` was 4.75.0 /
  ADR-053 when this plan was written, so expect 4.76.0 / ADR-054 — confirm, do not assume.
* **Version, `CHANGELOG.md` and `TODO.md` are written at merge time**, not during the cycle. They are
  the entire conflict surface with the other session's concurrent Cycle B.
* **`CLAUDE.md:140` is NOT deferrable** — it is standing instruction the harness loads, and it names
  `add_ghost_gk` as the running example of an unfixed fabrication. Task 5.
* **Lint at CI scope, never `.`**: `python -m ruff check silly_kicks/ tests/ scripts/`,
  `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`. Neither tool is
  on PATH — always `python -m`.
* **`scripts/` is ASCII-only** (ruff RUF001/2/3 + the cp1252 `--help` gate). No em-dashes, `≥`, `→`.
* **Never invoke a `scripts/*.py` that lacks argparse** — `--help` is ignored and `main()` runs.
* **A driver cannot run in the commit that introduces it** — `scripts/_provenance.py:73-76` counts
  untracked files as dirty. **Commit the spec and this plan FIRST**, then run anything.
* **Test the failing side too.** Every guard needs a planted case proving it can fail.
* Full suite to a UNIQUE log path per run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip`.
* No commits without explicit owner approval. Approval to commit includes approval to push.

---

## File Structure

| file | responsibility |
|---|---|
| `silly_kicks/tracking/_ghost_gk.py` | modify. The refusal in `_serve_positions_core`; `GHOST_GK_SOURCE_VALUES`. |
| `silly_kicks/tracking/features.py` | modify. `add_ghost_gk` emits the column; the `:4533` short-circuit must not bypass the guard. |
| `silly_kicks/tracking/schema.py` | modify. `VelocityRegimeDiagnosis` + the regime tokens, beside `TimeBaseDiagnosis` (:262) and `IdDtypeDiagnosis` (:293). |
| `silly_kicks/tracking/utils.py` | modify. `validate_velocity_regime`, beside `validate_time_base` (:598) and `validate_id_dtypes` (:685). |
| `silly_kicks/tracking/__init__.py` | modify. Export the diagnostic + the source constants. |
| `silly_kicks/spadl/_sb_coordinates.py` | **new.** The scalar affine, extracted from `_convert_locations`. |
| `silly_kicks/spadl/statsbomb.py` | modify. `_convert_locations` becomes a thin wrapper. |
| `silly_kicks/providers/statsbomb/__init__.py` | **new.** The port's public surface. |
| `silly_kicks/providers/statsbomb/parse.py` | **new.** Extracted from `build_sb360_coverage.py`. |
| `scripts/build_sb360_coverage.py` | modify. Re-pointed at the port. |
| `tests/datasets/statsbomb/three-sixty/` | **new.** Committed slice + `SOURCE_SHA`. |

---

## PHASE 0 — the premise

### Task 0: Measure the no-op claim BEFORE changing anything

The whole cycle is retrain-free only if ghost positions on velocity-bearing frames are unchanged.
Risk 1 says measure early; this task is that measurement, and it is a **characterization test that
must survive the rest of the plan**.

**Files:**
- Create: `tests/tracking/test_ghost_gk_velocity_path_unchanged.py`

**Interfaces:**
- Produces: `tests/tracking/data/ghost_velocity_path_baseline.npz` — the pre-change oracle.

- [ ] **Step 1: Capture the baseline on the UNMODIFIED tree**

```python
# tests/tracking/test_ghost_gk_velocity_path_unchanged.py
"""The cycle's premise: ghost positions on velocity-bearing frames do not move.

Captured on the unmodified tree BEFORE the refusal lands. If this test ever fails, the
degradation cycle has become a retrain trigger and the scope decision must be revisited
rather than absorbed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_b

_BASELINE = pathlib.Path(__file__).parent / "data" / "ghost_velocity_path_baseline.npz"


def _serve():
    actions, frames, _links = build_leg_b()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    return out[["ghost_gk_x", "ghost_gk_y"]].to_numpy(dtype=float)


@pytest.mark.skipif(not _BASELINE.is_file(), reason="baseline not captured yet")
def test_velocity_path_positions_are_unchanged():
    ref = np.load(_BASELINE)["positions"]
    got = _serve()
    assert got.shape == ref.shape, f"row count changed: {got.shape} vs {ref.shape}"
    # Tolerance CHOSEN, not inherited (CLAUDE.md/ADR-050). atol=1e-6, rtol=0 is the feature-contract
    # value, and the feature contract is the matching precedent: a COMMITTED artifact compared in CI
    # across environments. The sibling ghost golden's rtol=1e-9/atol=0.0 is NOT the precedent -- its
    # docstring scopes it to a SAME-ENVIRONMENT, one-machine-one-cycle equivalence gate.
    #
    # rtol=0 deliberately: rtol scales by |desired|, so a positive rtol with atol=0.0 degenerates to
    # EXACT at a zero value -- and both columns can reach exactly 0.0, because the 4.22.1 pitch clamp
    # bounds gx/gy and features.py:4586-4587 map a clamped far-end position onto 0.0.
    #
    # equal_nan defaults True; this baseline is all-finite (the non-vacuity test asserts it), so it
    # does not bite here -- recorded because CLAUDE.md flags equal_nan as load-bearing.
    #
    # The err_msg carries the max delta because the STATED harm is a MISREADING, not a flake: this
    # test's failure says the premise broke and the cycle is a retrain trigger. A ~1 ULP drift must
    # read as platform noise; a real move must read as a retrain trigger. Tolerance alone does not
    # solve that.
    delta = float(np.nanmax(np.abs(got - ref)))
    np.testing.assert_allclose(
        got,
        ref,
        rtol=0,
        atol=1e-6,
        err_msg=(
            f"ghost positions moved on the velocity path: max |delta| = {delta:.3e}. "
            f"Below ~1e-12 this is platform float noise; at 1e-6 or above the cycle is a "
            f"RETRAIN TRIGGER and the scope decision must be re-taken."
        ),
    )


def test_the_baseline_is_not_vacuous():
    """A baseline of all-NaN would make the assertion above pass while proving nothing."""
    got = _serve()
    assert len(got) > 0, "fixture produced no ghost rows"
    assert np.isfinite(got).all(), "velocity-bearing leg must produce finite ghosts"
```

- [ ] **Step 2: Generate the baseline**

Run:
```bash
python -c "
import numpy as np, pathlib, warnings
warnings.simplefilter('ignore')
import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_b
a, f, _ = build_leg_b()
out = T.add_ghost_gk(a, f, home_team_id=1)
p = pathlib.Path('tests/tracking/data/ghost_velocity_path_baseline.npz')
np.savez_compressed(p, positions=out[['ghost_gk_x','ghost_gk_y']].to_numpy(dtype=float))
print('wrote', p)
"
```

- [ ] **Step 3: Verify both tests pass on the unmodified tree**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_path_unchanged.py -v`
Expected: 2 passed.

- [ ] **Step 4: DELETE the `skipif` once the baseline is committed**

The `skipif` exists only for the write-before-generate ordering in Steps 1-2. Left in, the cycle's
single load-bearing assertion silently PASSES wherever the npz is absent — a fresh clone, a partial
checkout, a worker who reran Step 1 without Step 2. Replace it with a hard assertion so a missing
oracle is RED:

```python
def test_velocity_path_positions_are_unchanged():
    assert _BASELINE.is_file(), f"baseline oracle missing at {_BASELINE} -- regenerate it"
    ref = np.load(_BASELINE)["positions"]
```

---

## PHASE 1 — Commit 1, honest degradation

### Task 1: The refusal at the shared serving seam

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`_serve_positions_core`, ~line 2290)
- Test: `tests/tracking/test_ghost_gk_velocity_refusal.py`

**Interfaces:**
- Consumes: `velocity_unavailable_by_design(frames: pd.DataFrame) -> bool` from
  `silly_kicks/tracking/_velocity_availability.py:15`.
- Produces: `GHOST_GK_COMPUTED`, `GHOST_GK_VELOCITY_UNAVAILABLE`, `GHOST_GK_NO_KEEPER`,
  `GHOST_GK_UNLINKED`, `GHOST_GK_SOURCE_VALUES` in `silly_kicks/tracking/_ghost_gk.py`. Tasks 2 and 4
  consume these.

- [ ] **Step 1: Write the failing tests — BOTH directions**

```python
# tests/tracking/test_ghost_gk_velocity_refusal.py
"""The ghost path must obey the speed_source contract in BOTH directions.

CLAUDE.md: "An UNMARKED or PARTIALLY-marked frame set missing vx/vy still RAISES: fail-loud
wins on a mixed frame set." Measured before this cycle, the ghost path fabricated in both.
"""

from __future__ import annotations

import numpy as np
import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_a


def _marked():
    """Freeze-frame leg: every row already carries speed_source='unavailable'."""
    return build_leg_a()


def _unmarked():
    actions, frames, links = build_leg_a()
    frames = frames.copy()
    frames["speed_source"] = None
    return actions, frames, links


def test_marked_frames_degrade_to_nan_not_a_coordinate():
    actions, frames, _ = _marked()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    assert out["ghost_gk_x"].isna().all(), "marked frames must not produce a coordinate"
    assert out["ghost_gk_y"].isna().all()


def test_unmarked_velocity_less_frames_RAISE():
    """The failing side. Fail-loud wins on a mixed frame set."""
    actions, frames, _ = _unmarked()
    with pytest.raises(ValueError, match="speed_source"):
        T.add_ghost_gk(actions, frames, home_team_id=1)


def test_the_marked_case_is_not_vacuous():
    """Non-vacuity: the fixture must actually have rows to refuse on."""
    actions, frames, _ = _marked()
    assert len(actions) > 0 and len(frames) > 0
```

- [ ] **Step 2: Run to verify BOTH fail**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_refusal.py -v`
Expected: `test_marked_frames_degrade_to_nan_not_a_coordinate` FAILS (real coordinates ~99.3),
`test_unmarked_velocity_less_frames_RAISE` FAILS (`DID NOT RAISE`). Paste both into the commit message.

- [ ] **Step 3: Add the vocabulary to `_ghost_gk.py`**

Place beside `GHOST_GK_FEATURE_NAMES`:

```python
#: Closed vocabulary for ``ghost_gk_source`` (the DAS_SOURCE_VALUES / PRESS_COMMITMENT_SOURCE_VALUES
#: pattern). Each token is exported so a consumer enum pins to this set, not to string literals.
GHOST_GK_COMPUTED = "computed"
GHOST_GK_VELOCITY_UNAVAILABLE = "velocity_unavailable"
GHOST_GK_NO_KEEPER = "no_keeper"
GHOST_GK_UNLINKED = "unlinked"
GHOST_GK_SOURCE_VALUES: tuple[str, ...] = (
    GHOST_GK_COMPUTED,
    GHOST_GK_VELOCITY_UNAVAILABLE,
    GHOST_GK_NO_KEEPER,
    GHOST_GK_UNLINKED,
)
```

- [ ] **Step 4: Put the refusal in `_serve_positions_core`**

At the TOP of `_serve_positions_core`, before model resolution:

```python
    # Velocity contract (ADR-053 follow-up). This is the SHARED serving seam -- add_ghost_gk,
    # compute_ghost_gk and serve_ghost_gk_positions all funnel through here, so the guard placed
    # at any one of them would leave the other two fabricating. The 4.22.1 physical-pitch clamp
    # lives here for the same reason: policy at the edge, and this function IS the edge.
    #
    # The model is an HGBR: absent velocity features are NOT zero-filled, they are routed down
    # each split's LEARNED missing-value direction, fitted where NaN meant an occasional dropped
    # measurement. On a freeze-frame 5 of 26 features are absent on 100% of rows, so the output
    # is a plausible coordinate with no basis. Refuse instead.
    if _velocity_unavailable_by_design(frames):
        raise _GhostVelocityUnavailable
    if "vx" not in frames.columns or "vy" not in frames.columns:
        raise ValueError(
            "compute_ghost_gk requires vx/vy on frames (call derive_velocities() first), or "
            "declare speed_source unavailable. See the velocity-availability contract."
        )
```

with, at module scope:

```python
from ._velocity_availability import velocity_unavailable_by_design as _velocity_unavailable_by_design


class _GhostVelocityUnavailable(Exception):
    """Internal signal: frames declare velocity structurally unavailable.

    Never escapes the module. Each public seam catches it and degrades in the shape its own
    output allows -- NaN rows with provenance for the two column-emitting seams, NO rows for
    serve_ghost_gk_positions (gkdv RAISES on a non-finite ghost on a scored row, so NaN rows
    there would break TF-19 rather than degrade it -- _engine.py:557-562).
    """
```

- [ ] **Step 5: Run — the RAISE test passes, the NaN test still fails**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_refusal.py -v`
Expected: `test_unmarked_velocity_less_frames_RAISE` PASSES;
`test_marked_frames_degrade_to_nan_not_a_coordinate` still FAILS (the internal exception is not yet
caught). That split is the point — Task 2 catches it.

- [ ] **Step 6: Verify Task 0's baseline still holds**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_path_unchanged.py -v`
Expected: 2 passed. **If this fails, STOP** — the cycle is a retrain trigger and the scope decision
must be re-taken.

---

### Task 2: Per-seam degradation and the `ghost_gk_source` column

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`compute_ghost_gk`, `serve_ghost_gk_positions`)
- Modify: `silly_kicks/tracking/features.py:4533` (the short-circuit) and the `add_ghost_gk` body
- Test: `tests/tracking/test_ghost_gk_velocity_refusal.py` (extend)

**Interfaces:**
- Consumes: `_GhostVelocityUnavailable`, `GHOST_GK_*` from Task 1.
- Produces: `ghost_gk_source` on `add_ghost_gk` (actions) and `compute_ghost_gk` (frames);
  `serve_ghost_gk_positions` returns an EMPTY frame with its normal columns.

- [ ] **Step 1: Write the failing tests for all three seams**

Append:

```python
def test_compute_ghost_gk_emits_nan_and_provenance_on_frames():
    _actions, frames, _ = _marked()
    out = T.compute_ghost_gk(frames, home_team_id=1)
    assert out["ghost_gk_x"].isna().all()
    assert (out["ghost_gk_source"] == T.GHOST_GK_VELOCITY_UNAVAILABLE).all()


def test_serve_ghost_gk_positions_returns_NO_ROWS_not_nan_rows():
    """gkdv/_engine.py:557-562 RAISES on a non-finite ghost on a SCORED frame, so NaN rows here
    would break TF-19 rather than degrade it. Returning nothing routes into the existing counted
    drop (_DROP_NO_GHOST)."""
    _actions, frames, _ = _marked()
    out = T.serve_ghost_gk_positions(frames, home_team_id=1)
    assert len(out) == 0, "marked frames must yield NO served rows"
    assert "ghost_gr_x" in out.columns, "empty frame must keep its schema"


def test_add_ghost_gk_emits_the_source_column_on_the_velocity_path_too():
    """The column is UNCONDITIONAL (the _press_commitment._OUTPUT_COLS pattern). A conditionally
    present provenance column cannot be told from an older version that lacked it."""
    from tests.sb360._fixture import build_leg_b

    actions, frames, _ = build_leg_b()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    assert "ghost_gk_source" in out.columns
    assert (out["ghost_gk_source"] == T.GHOST_GK_COMPUTED).all()


def test_the_short_circuit_cannot_bypass_the_guard():
    """features.py:4533 skips compute when frames already carry NON-NaN ghost columns.

    The bypass condition is non-NaN ghost columns PLUS a marked frame set. Feeding
    compute_ghost_gk's own output would NOT exercise it -- after the refusal that output is
    all-NaN, so `notna().any()` is False, the short-circuit is never taken, and the test would
    travel the ordinary recompute path while claiming to prove the bypass closed. It would pass
    against a build with no guard at all.

    So use the PRE-POPULATED fixture the purity gate already maintains for this exact branch
    (tests/test_add_star_purity.py:187, `_frames_with_ghost()` -- "precomputed ghost columns on
    GK rows -> exercises add_ghost_gk's precompute short-circuit"), and mark it unavailable.
    """
    from silly_kicks.tracking.schema import SPEED_SOURCE_UNAVAILABLE
    from tests.test_add_star_purity import _frames_with_ghost

    frames = _frames_with_ghost().copy()
    assert frames["ghost_gk_x"].notna().any(), "fixture must PRE-POPULATE ghosts or this is vacuous"
    frames["speed_source"] = SPEED_SOURCE_UNAVAILABLE

    actions, _f, _l = _marked()
    out = T.add_ghost_gk(actions, frames, home_team_id=1)
    assert out["ghost_gk_x"].isna().all()
    assert (out["ghost_gk_source"] == T.GHOST_GK_VELOCITY_UNAVAILABLE).all()


def test_precomputed_ghosts_UNMARKED_and_velocity_less_still_RAISE():
    """The hole the marker check alone leaves open, and the third instance of this shape.

    `velocity_unavailable_by_design` is an ALL-rows predicate, so an UNMARKED or PARTIALLY-marked
    frame set returns False, the early return does not fire, the precompute short-circuit is taken,
    compute_ghost_gk never runs, and the serving-seam guard is never reached.

    MEASURED before the fix: `add_ghost_gk` returned `ghost_gk_x = 52.5`.

    The plan's other unmarked test cannot catch this -- it builds from `build_leg_a`, which carries
    NO precomputed ghost columns, so `notna().any()` is False, the short-circuit is skipped and the
    seam guard fires. It passes while the hole stays open.
    """
    import numpy as np

    from tests.test_add_star_purity import _frames_with_ghost, make_actions

    frames = _frames_with_ghost().copy()
    frames = frames.drop(columns=[c for c in ("vx", "vy") if c in frames.columns])
    frames["speed_source"] = None
    assert frames["ghost_gk_x"].notna().any(), "fixture must pre-populate ghosts or this is vacuous"

    with pytest.raises(ValueError, match="vx/vy"):
        T.add_ghost_gk(make_actions(), frames, home_team_id=5)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_refusal.py -v`
Expected: the four new tests FAIL (`AttributeError` on the constants, or the internal exception
escaping).

- [ ] **Step 3: Catch and degrade in `compute_ghost_gk`**

Wrap its call to `_serve_positions_core`:

```python
    try:
        resolved, meta, batch_features, positions, clamped = _serve_positions_core(...)
    except _GhostVelocityUnavailable:
        out = frames.copy()
        out["ghost_gk_x"] = np.nan
        out["ghost_gk_y"] = np.nan
        out["ghost_gk_source"] = GHOST_GK_VELOCITY_UNAVAILABLE
        return out
```

and on the normal path set `out["ghost_gk_source"] = GHOST_GK_COMPUTED`.

- [ ] **Step 4: Route the refusal through the EXISTING empty-frame branch — do NOT write a new one**

`serve_ghost_gk_positions` already has this path at `_ghost_gk.py:2446-2487`, guarded by
`if len(positions) == 0:`. It carries ~20 lines of comment recording that the four join-key dtypes
are **DERIVED FROM THE INPUT, not hard-coded**, because a hard-coded pair "was measured wrong on
both" provider families and silently degrades `period_id`/`frame_id` from int64 to object under a
`pd.concat` across a per-match loop — an ADR-019 class defect. It also goes through one real value
rather than a zero-row slice, for a documented pandas-3 inference reason.

**A new `_empty_served_frame()` helper would duplicate that and reintroduce the exact defect the
comment exists to record.** Instead, make the refusal reach that branch:

```python
    try:
        resolved, meta, batch_features, positions, clamped = _serve_positions_core(...)
    except _GhostVelocityUnavailable:
        # Reuse the EXISTING len(positions) == 0 branch below rather than building a second
        # empty frame -- its join-key dtypes are derived from the input for a measured reason
        # (_ghost_gk.py:2452-2460). NO rows, not NaN rows: gkdv/_engine.py:557-562 RAISES on a
        # non-finite ghost on a scored frame, so NaN here would break TF-19 rather than degrade it.
        # Only `positions` is read by the branch below (it reads `frames` for its dtypes and never
        # touches meta/clamped) -- the other two are set for shape consistency and are inert.
        meta = frames.iloc[:0]
        positions = np.empty((0, 2), dtype=float)
        clamped = np.zeros(0, dtype=bool)
```

then fall through to the existing branch unchanged. If the local structure makes that awkward,
extract `:2446-2487` verbatim into a helper and call it from BOTH places — but do not write a second
implementation.

This seam emits NO `ghost_gk_source`: it has no row to carry it, gkdv's `_build_provenance` projects
to `_PROVENANCE_COLUMNS` (`_engine.py:59-74`) and would drop it anyway, and its siblings use the
`ghost_` prefix rather than `ghost_gk_`.

- [ ] **Step 5: Emit the column from `add_ghost_gk`, unconditionally — on a path that BYPASSES the
`notna()` filter**

**The obvious implementation is wrong.** `add_ghost_gk` builds its GK lookup by filtering
`ghost_frames` on `ghost_gk_x.notna()` (`features.py:4549`). After the refusal every `ghost_gk_x` is
NaN, so that filter yields an EMPTY frame, nothing merges, and every action would fall through to
`GHOST_GK_UNLINKED` — contradicting the spec's contract and two of this task's own assertions.

So project the source column separately, from the frames, WITHOUT the `notna()` predicate:

```python
    # The source projection must not go through the notna() filter above: on a refused frame set
    # every ghost_gk_x is NaN, so that filter is empty and every action would read "unlinked"
    # instead of "velocity_unavailable" -- the reason being reported would be wrong.
    src_by_frame = (
        ghost_frames.loc[
            ghost_frames["is_goalkeeper"].astype(bool) & ~ghost_frames["is_ball"].astype(bool),
            ["game_id", "period_id", "frame_id", "team_id", "ghost_gk_source"],
        ]
        if "ghost_gk_source" in ghost_frames.columns
        else None
    )
```

then merge it on the same keys as the position lookup, **through its OWN `align_join_keys` call**.
Signature: `align_join_keys(left, right, keys)` (`silly_kicks/id_compat.py:479`) — it RETURNS a tuple
and both sides must be reassigned; `keys` entries are a `str` or a `(left_key, right_key)` pair.

Two points a worker will otherwise stop to derive:

* **`src_by_frame` needs its own call.** The existing call at `features.py:4560` reassigns only
  `linked` and `gk_ghost`; a third frame is not covered by it. Omitting it walks the first string-id
  consumer of the new port into the ADR-019 defect class already fixed two lines above, where a
  comment records that it replaced an ad-hoc `game_id.astype(str)` hand-patch.
* **Run it against the ALREADY-aligned `linked`**, or `linked` is realigned a second time. That is
  harmless — but say so, because working out *that* it is harmless costs more than reading it.

**The gate that turns "named as required" into "verified" is
`tests/tracking/test_ghost_gk_id_dtype.py:144-146`**, which parametrizes `team_dtype` over
`{int64, Int64, float64, string}` and `home_scalar` over `{1, "1", 1.0}`. Running it is the evidence
this step should produce; a source merge without alignment fails there.

Default to `GHOST_GK_UNLINKED` only where the action genuinely reached no frame.

**Key the short-circuit guard on the MARKER, not on the source column.** An earlier draft added a
`ghost_gk_source != velocity_unavailable` clause to the `:4533` condition. That is a proxy, and it
fails in the case that matters most: frames enriched by a PRIOR release carry real ghost values and a
`speed_source` marker but **no** `ghost_gk_source` column at all, so `.all()` runs on an EMPTY Series
and returns **vacuously True** (measured). All three clauses hold, the short-circuit is taken, and the
guard is bypassed on exactly the cached/persisted frames a real consumer has lying around.

So refuse on the invariant, before the short-circuit is even considered:

```python
    # Refuse on the MARKER, ahead of the precompute short-circuit. Keying this on ghost_gk_source
    # would be a proxy for the marker, and frames enriched by an earlier release carry ghost values
    # and a marker but no provenance column -- `.all()` on the absent column is vacuously True, so
    # the proxy silently admits precisely the legacy frames it was meant to catch.
    if velocity_unavailable_by_design(frames):
        out = actions.copy()
        out["ghost_gk_x"] = np.nan
        out["ghost_gk_y"] = np.nan
        out["ghost_gk_source"] = GHOST_GK_VELOCITY_UNAVAILABLE
        return out

    # BOTH guards, in the _press_commitment.py:103-107 order. The marker predicate above is an
    # ALL-rows test ("True iff EVERY row declares kinematics structurally unavailable",
    # _velocity_availability.py:16), so on an UNMARKED or PARTIALLY-marked set it returns False and
    # the early return does NOT fire. Without this second check the short-circuit below is then
    # reached, compute_ghost_gk never runs, and Task 1's guard inside _serve_positions_core is never
    # reached either. MEASURED on precomputed-ghost frames with vx/vy dropped and no marker:
    # add_ghost_gk returned ghost_gk_x = 52.5 -- a fabricated coordinate on frames CLAUDE.md says
    # must RAISE.
    if "vx" not in frames.columns or "vy" not in frames.columns:
        raise ValueError(
            "add_ghost_gk requires vx/vy on frames (call derive_velocities() first), or "
            "declare speed_source unavailable. See the velocity-availability contract."
        )

    if "ghost_gk_x" in frames.columns and frames["ghost_gk_x"].notna().any():
        ghost_frames = frames          # legitimate precompute alias; velocity is present
```

With both guards ahead of it, the short-circuit is reachable only by frames that are legitimately
scoreable, so it no longer matters that it bypasses the serving seam.

**The legitimately-aliased path emits `GHOST_GK_COMPUTED`, not `GHOST_GK_UNLINKED`.** When the
short-circuit IS taken on velocity-bearing pre-enriched frames — the case `_frames_with_ghost` exists
to cover — those frames carry no `ghost_gk_source`, so `src_by_frame` is `None` and every action would
fall to the `UNLINKED` default while `ghost_gk_x` holds a real number. **A provenance token that
contradicts the value beside it is worse than no column**: "unlinked" next to a coordinate is a claim
the data refutes, and a consumer enum pinned to the vocabulary would have to special-case it. So when
`src_by_frame is None` and positions merged, fill `GHOST_GK_COMPUTED`; reserve `GHOST_GK_UNLINKED` for
actions that genuinely reached no frame.

The mechanism is one line, placed either side of the ADR-028 reprojection at
`features.py:4578-4587` — order is free, because NaN is invariant under both `105 - gx` and
`np.where(flip, 68 - gy, gy)`. **Say which side you put it on**, so a later reader does not have to
re-derive that it does not matter:

```python
    out["ghost_gk_source"] = np.where(
        out["ghost_gk_x"].notna(), GHOST_GK_COMPUTED, GHOST_GK_UNLINKED
    )
```

**Gap A — `GHOST_GK_NO_KEEPER` has no producer in any specified path.** The vocabulary declares four
tokens; the plan specifies three. An action that links to a frame carrying no DEFENDING keeper falls
out at the `ids_differ` filter (`features.py:4571`), arrives with `ghost_gk_x` NaN, and the line above
labels it `"unlinked"` — which is FALSE, it did reach a frame.

**DECIDED: emit `NO_KEEPER`.** The distinction is exactly what a GK consumer needs — "the frame had
no defending keeper" and "the action reached no frame" are different facts with different remedies,
and collapsing them into `unlinked` states something the data refutes. The two are separable at the
merge: `linked` carries `frame_id` per action, so an action WITH a frame_id but no ghost value is
`NO_KEEPER`, and one with no frame_id is `UNLINKED`:

```python
    has_frame = out["action_id"].isin(linked["action_id"])
    out["ghost_gk_source"] = np.where(
        out["ghost_gk_x"].notna(),
        GHOST_GK_COMPUTED,
        np.where(has_frame, GHOST_GK_NO_KEEPER, GHOST_GK_UNLINKED),
    )
```

Rejected alternative: document that the token is unreachable here, per `features.py:2745`'s
*"``unscoreable_call`` is never produced here"*. That is the right move when a token genuinely
belongs to another layer; here it belongs to THIS one and is cheap to produce.

**Gap B — DECIDED: add the closed-set guard.** `_das.py:103-106` validates against
`DAS_SOURCE_VALUES` and raises on an out-of-vocabulary token. A vocabulary described as closed but
unenforced is not closed — it is a comment. `das` puts its check in an exception constructor because
its token travels on an exception; ours travels on a column, so the analogue is a post-condition at
the emit site:

```python
    _emitted = set(pd.unique(out["ghost_gk_source"].dropna()))
    if not _emitted <= set(GHOST_GK_SOURCE_VALUES):
        raise ValueError(
            f"ghost_gk_source emitted values outside its closed vocabulary: "
            f"{sorted(_emitted - set(GHOST_GK_SOURCE_VALUES))}"
        )
```

Cheap, and it is what makes the exported `GHOST_GK_SOURCE_VALUES` a contract a consumer enum can pin
to rather than a suggestion.

This is the same branch Task 4 Step 3 registers a purity variant for — fix them together.

- [ ] **Step 6: Run all refusal tests**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_refusal.py -v`
Expected: all 7 pass.

- [ ] **Step 7: Re-verify the baseline and lint**

Run: `python -m pytest tests/tracking/test_ghost_gk_velocity_path_unchanged.py -v`
Expected: 2 passed — positions unchanged, schema wider.

Run: `python -m ruff check silly_kicks/ && python -m ruff format --check silly_kicks/`

---

### Task 3: `validate_velocity_regime`

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (the dataclass + tokens)
- Modify: `silly_kicks/tracking/utils.py` (the validator)
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_velocity_regime.py`

**Interfaces:**
- Produces: `validate_velocity_regime(frames, *, on_mismatch="raise") -> VelocityRegimeDiagnosis`
  with fields `regime`, `speed_source_counts`, `has_velocity_columns`, `message`.

- [ ] **Step 1: Write the failing tests — all three regimes**

```python
# tests/tracking/test_velocity_regime.py
from __future__ import annotations

import pytest

import silly_kicks.tracking as T
from tests.sb360._fixture import build_leg_a, build_leg_b


def test_positional_only_regime():
    _a, frames, _l = build_leg_a()
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "positional_only"
    assert d.has_velocity_columns is False


def test_velocity_informed_regime():
    _a, frames, _l = build_leg_b()
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_informed"


def test_mixed_regime_is_the_one_fail_loud_exists_for():
    _a, frames, _l = build_leg_a()
    frames = frames.copy()
    frames.loc[frames.index[: len(frames) // 2], "speed_source"] = None
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "mixed"
    with pytest.raises(ValueError, match="mixed"):
        T.validate_velocity_regime(frames, on_mismatch="raise")


def test_forgot_derive_velocities_is_NOT_labelled_mixed():
    """Distinct from MIXED, and it is the case a user is most likely to hit. Labelling it 'mixed'
    would raise with 'some rows can carry velocity and others structurally cannot', which is false
    for these frames."""
    _a, frames, _l = build_leg_b()
    frames = frames.drop(columns=["vx", "vy"]).copy()
    frames["speed_source"] = "native"
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_missing"
    assert "derive_velocities()" in d.message


def test_frames_without_a_speed_source_column_do_not_CRASH():
    """Measured on the first draft: frames.get('speed_source') returns None, None == marker is a
    Python bool, and False.sum() raises AttributeError. A row-count guard does not prevent it."""
    frames = pd.DataFrame({"x": [1.0, 2.0], "y": [3.0, 4.0]})
    d = T.validate_velocity_regime(frames, on_mismatch="ignore")
    assert d.regime == "velocity_missing"
    assert d.has_velocity_columns is False
```

The test module needs `import pandas as pd` alongside its existing imports.

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_velocity_regime.py -v`
Expected: FAIL, `AttributeError: module 'silly_kicks.tracking' has no attribute
'validate_velocity_regime'`.

- [ ] **Step 3: Implement**

**Split across two modules, because the family it joins is split.** An earlier draft invented one
new module holding both halves, which breaks the very precedent the design argument rests on:
`TimeBaseDiagnosis` and `IdDtypeDiagnosis` live in `schema.py:262,293`, while `validate_time_base`
and `validate_id_dtypes` live in `utils.py:598,685`. Matching that also inherits their
debt-registration idiom and avoids adding a module to `_PUBLIC_MODULE_FILES`.

Put the dataclass and tokens in `schema.py`; put the validator in `utils.py`. Import paths are
Hyrum surface — free to choose now, breaking to change later.

```python
# --- silly_kicks/tracking/schema.py (beside TimeBaseDiagnosis / IdDtypeDiagnosis) ---
"""Pre-flight velocity-regime diagnostic (the ADR-017 / ADR-019 pattern).

Third member of the validate_time_base / validate_id_dtypes family. Five aggregators produce
values whose INTERPRETATION changes without velocity while the value stays honest and usable --
pitch control at zero velocity is a well-defined positional model. That is a property of the
frame SET, not of any row, so it is a diagnostic rather than fifteen per-row provenance columns
each carrying a constant.

Takes `frames` only. Its two siblings take (actions, frames); velocity regime is a property of
frames alone, and an unread parameter is the dead-`home_team_id` defect recorded against
space_creation.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .schema import SPEED_SOURCE_UNAVAILABLE

VELOCITY_INFORMED = "velocity_informed"
POSITIONAL_ONLY = "positional_only"
MIXED = "mixed"
#: Fourth regime, and NOT a variant of MIXED. Frames that declare velocity available (or declare
#: nothing) but carry no vx/vy are the "forgot derive_velocities()" case: recoverable by calling it,
#: with nothing structurally missing. Labelling that "mixed" would attach the message "some rows can
#: carry velocity and others structurally cannot", which is FALSE for those frames -- and it is the
#: case a user is most likely to hit, so the default raise would explain itself wrongly.
VELOCITY_MISSING = "velocity_missing"
#: An EMPTY frame set is not a velocity problem. It does NOT raise, and that FOLLOWS the siblings
#: rather than departing from them -- measured on a schema-shaped zero-row frame (the realistic
#: case a per-match loop produces), validate_time_base AND validate_id_dtypes both return a
#: diagnosis. (An earlier draft claimed they disagreed. That was measured on a BARE pd.DataFrame()
#: with no columns at all, where validate_time_base raises KeyError from a two-column lookup --
#: a degenerate shape, not a policy, and the wrong case to generalise from.)
#:
#: The token exists because a zero-row set is genuinely more informative than velocity_missing,
#: and because it is what suppresses the default raise -- which is the real point.
EMPTY = "empty"
#: Exported for the same reason GHOST_GK_SOURCE_VALUES is: a regime string that can RAISE by
#: default is a consumer-facing contract, and consumers must pin an enum to this set rather than
#: to string literals.
VELOCITY_REGIME_VALUES: tuple[str, ...] = (
    VELOCITY_INFORMED,
    POSITIONAL_ONLY,
    MIXED,
    VELOCITY_MISSING,
    EMPTY,
)


@dataclass(frozen=True)
class VelocityRegimeDiagnosis:
    regime: str
    speed_source_counts: dict[str, int]
    has_velocity_columns: bool
    message: str


def validate_velocity_regime(
    frames: pd.DataFrame,
    *,
    on_mismatch: Literal["warn", "raise", "ignore"] = "raise",
) -> VelocityRegimeDiagnosis:
    """Report whether `frames` carry usable kinematics, before anything is computed."""
    # Guard on COLUMN PRESENCE, not row count. `frames.get("speed_source")` returns None on a
    # missing key; `None == "unavailable"` is the Python bool False; `False.sum()` raises
    # AttributeError. A diagnostic whose job is to report on frames must not crash on the frames
    # most likely to need diagnosing -- third-party builders, derived frames, anything upstream
    # of the schema. Measured: an `if n else 0` guard does NOT prevent this.
    has_marker = "speed_source" in frames.columns
    counts: dict[str, int] = {}
    if has_marker:
        counts = {str(k): int(v) for k, v in frames["speed_source"].value_counts(dropna=False).items()}
    has_cols = "vx" in frames.columns and "vy" in frames.columns

    n = len(frames)
    if n == 0:
        return VelocityRegimeDiagnosis(EMPTY, counts, has_cols, "velocity regime: empty frame set.")
    n_unavailable = int((frames["speed_source"] == SPEED_SOURCE_UNAVAILABLE).sum()) if has_marker else 0

    if n_unavailable == n:
        regime = POSITIONAL_ONLY
    elif n_unavailable == 0 and has_cols:
        regime = VELOCITY_INFORMED
    elif n_unavailable == 0:
        regime = VELOCITY_MISSING
    else:
        regime = MIXED

    message = (
        f"velocity regime: {regime} ({n_unavailable}/{n} rows declare "
        f"speed_source={SPEED_SOURCE_UNAVAILABLE!r}; vx/vy columns "
        f"{'present' if has_cols else 'absent'})."
    )
    if regime == MIXED:
        message += (
            " A mixed frame set cannot be scored coherently: some rows can carry velocity and "
            "others structurally cannot. Fail-loud wins here."
        )
    elif regime == VELOCITY_MISSING:
        message += (
            " Nothing is structurally missing -- call derive_velocities() first, or declare "
            "speed_source unavailable if this source has no per-player temporal history."
        )
    if regime in (MIXED, VELOCITY_MISSING):
        if on_mismatch == "raise":
            raise ValueError(message)
        if on_mismatch == "warn":
            warnings.warn(message, stacklevel=2)
    return VelocityRegimeDiagnosis(regime, counts, has_cols, message)
```

- [ ] **Step 4: Export it**

Add `validate_velocity_regime`, `VelocityRegimeDiagnosis`, the five regime tokens plus
`VELOCITY_REGIME_VALUES`, and the four `GHOST_GK_*` constants plus `GHOST_GK_SOURCE_VALUES` to
`silly_kicks/tracking/__init__.py`'s imports and `__all__`. Exporting the ghost vocabulary while
leaving the regime vocabulary as bare literals would apply the same house rule unevenly in one commit.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/tracking/test_velocity_regime.py -v`
Expected: 5 passed.

- [ ] **Step 6: Satisfy `tests/test_public_api_examples.py` — three distinct obligations**

**(a) The module-level constants drag nothing.** Measured: `_walk_public_definitions` (:1182-1199)
enumerates `_DocstringEligibleNode` (:1161) = `AsyncFunctionDef | FunctionDef | ClassDef`. A
module-level assignment is `ast.Assign` and is not eligible, so `VELOCITY_REGIME_VALUES` and the five
tokens are free — `DAS_SOURCE_VALUES` is the live precedent, exported with no Examples section.

**(b) `VelocityRegimeDiagnosis` IS a `ClassDef` and needs one.** Both siblings pass only because they
are registered as debt — `_EXAMPLES_DEBT` at `:771` (`TimeBaseDiagnosis`) and `:777`
(`IdDtypeDiagnosis`), each with a one-line reason. Write a real Examples section, or mirror the
siblings' debt entry.

**(c) No new module registration is needed** now that the code lands in `schema.py` and `utils.py`,
both already in `_PUBLIC_MODULE_FILES` (:137) — the hand-maintained tuple pinned to the derived
surface by the meta-assertion at `:1409`. Had this stayed in a new module, that registration would
have been mandatory and `:1469` requires `_EXAMPLES_DEBT` keys to reference registered files, so
registration would have come first regardless.

`validate_velocity_regime` still needs a real Examples section. A `>>> ...  # doctest: +SKIP` stub is rejected by
`_demonstrates_something`. Use an indented RST literal block (accepted as a REAL example by
`_has_real_example`) since a meaningful call needs a real `frames`.

Run: `python -m pytest tests/test_public_api_examples.py -q`

---

### Task 4: Gate surface and SB360 re-adjudication

**Files:**
- Modify: `silly_kicks/feature_glossary.py`
- Modify: `tests/test_add_star_purity.py`
- Regenerate: `tests/sb360/_entries/_gk.py`, `docs/research/sb360_coverage/behaviour_matrix.md`

- [ ] **Step 1: Run the gates and record what breaks**

Run: `python -m pytest tests/test_feature_glossary_coverage.py tests/test_add_star_purity.py tests/tracking/test_aggregator_column_liveness.py tests/tracking/test_mirror_registry.py tests/tracking/test_id_dtype_invariance.py tests/invariants/ -q`

Expected: **purity and glossary FAIL; liveness PASSES.** Two corrections to an earlier draft of this
step. The glossary gate lives in `tests/test_feature_glossary_coverage.py`
(`test_no_undocumented_columns`, and `test_emitting_module_importable_and_not_lazily_features` — the
gate that catches a bare-string `emitting_module`), which was ABSENT from the command, so the step
predicted a failure the invocation could not produce. And liveness passes for the reason Step 3 gives.

- [ ] **Step 2: Add the glossary entry**

Two gate constraints, both violated by the obvious draft. `emitting_module` is gate-checked for
importability and must be the **dotted path via the module constant** — `_M_GHOST_GK =
"silly_kicks.tracking._ghost_gk"` already exists at `feature_glossary.py:123`, and the `das_source`
entry uses exactly that. And `unit` is a closed `Literal` (`feature_glossary.py:23-37`) that does
**not** admit `"category"`; `das_source`, the direct precedent for a string provenance column, uses
`dimensionless`. pyright flags the Literal.

```python
    FeatureColumn(
        name="ghost_gk_source",
        definition=(
            "Which path produced the ghost-GK position: 'computed' from a velocity-bearing "
            "frame, 'velocity_unavailable' when the frame source structurally cannot carry "
            "kinematics (freeze-frames), 'no_keeper' when no goalkeeper was resolvable, "
            "'unlinked' when the action reached no frame."
        ),
        unit="dimensionless",
        emitting_module=_M_GHOST_GK,
        attribution=None,
        higher_is_better=None,
    ),
```

- [ ] **Step 3: Register liveness and purity entries**

Liveness: **nothing to register — this half is a no-op.** `test_aggregator_column_liveness.py`
derives added columns from each entry's output automatically, and its non-constant prong is
float-dtype-gated, so a new STRING column is picked up with no registration and correctly exempted.
The non-null prong passes because it reads `"computed"` on the velocity-bearing liveness fixture.
Confirm by running the gate; do not go hunting for a registration hook that does not exist.

Purity: **ADD to the two variants that already exist** — do not replace them.
`tests/test_add_star_purity.py:360-362` already registers `("compute", ...)` and `("precomputed",
..., _frames_with_ghost())` under `"tracking:add_ghost_gk"`, and the comment at `:358` says they
cover the precompute short-circuit branch. The NaN-vs-computed split is a DIFFERENT branch axis, so
this adds a third (and, if the refused-precomputed path differs, a fourth) — replacing the existing
two would silently drop the short-circuit branch's coverage.

- [ ] **Step 4: Regenerate the SB360 registry**

**Do NOT run this concurrently with pytest** — it rewrites `_entries/` under a live collection.

```bash
python tests/sb360/_regenerate.py && python tests/sb360/_adjudicate.py
```

Then hand-write rationales for the NEW `ghost_gk_source` column on all three axes.
`_regenerate.py:120` derives columns from actual output, so it WILL appear. Precedent to copy:
`press_commitment_source` in `tests/sb360/_entries/_offball.py` — verdicts at `:332`, `:362`, `:391`,
`applicability="no_support"` at `:405`, deltas at `:410`.

- [ ] **Step 5: Confirm the four ghost verdicts moved**

Run: `python -m pytest tests/sb360/ -q`
Expected: pass, with `add_ghost_gk` reading `honest_nan` on both velocity-axis columns.

- [ ] **Step 6: Re-render the behaviour matrix**

`behaviour_matrix.md` is renderer-generated and its summary counts change by construction
(`silent_degrade` 4 -> 0):

```bash
python scripts/render_sb360_matrix.py --out docs/research/sb360_coverage/behaviour_matrix.md
```

---

### Task 5: Execute the CLAUDE.md:146 caller sweep

Every consumer classified **with evidence on both sides**, plus the four things the rule says a
symbol sweep cannot see.

- [ ] **Step 1: Classify each consumer and record the evidence**

| consumer | expected classification | evidence to record |
|---|---|---|
| `docs/research/ghost_gk_spread_aggregates/harness/01_build_queries.py` | **clause (a) AND (e), and the ONLY caller of the seam outside `silly_kicks/`.** Imports `_serve_positions_core` and `GHOST_GK_FEATURE_NAMES` directly at `:9`, calls the seam at `:40`; its sibling `02_ground_truth_36k.py:6` records that its queries are "real feature vectors extracted by `_serve_positions_core`". **Classify with evidence on both sides**: read the harness and determine whether it passes frames carrying `vx`/`vy`, or raw frames it expects the extractor to tolerate. The new presence check fires BEFORE model resolution, so an unmarked velocity-less input that used to work now raises. If it does, that is a real break in a research harness and needs a decision, not a footnote. |
| `gkdv/_engine.py:537` | affected — inherits refusal via no rows | run `tests/gkdv/ -q`; `_DROP_NO_GHOST` count rises, `_engine.py:557` raise not reached |
| `atomic/tracking/features.py` | affected — mirrors all three | `tests/atomic/ -q` |
| `features.py:4536, 4628` | affected — xfns inherit, no string column | `tests/tracking/test_frame_aware_xfns_dup_action_id.py` |
| `scripts/make_ghost_gk_golden.py` | unaffected | Task 0 baseline holds |
| `scripts/gen_ghost_gk_kde_golden.py` | unaffected | same |
| `tests/tracking/data/ghost_gk_refactor_golden.npz` | clause (c) — unaffected | the byte-identity assertion IS the evidence |
| `tests/tracking/fixtures/ghost_gk_kde_golden.npz` | clause (c) — unaffected | same |
| `docs/huggingface/model-cards/ghost-gk-v1-model-card.md:102` | update if it describes the fabricating path | read it |
| `CLAUDE.md:140` | **clause (d) — MUST update** | its ADR-053 bullet names `add_ghost_gk` as the running example of an unfixed fabrication |
| `docs/research/sb360_coverage/{README,coverage}.md` | annotate as historical | do not rewrite the measurement |
| `docs/PRIVATE_CONSUMERS.md` | check for `_ghost_gk` pins | grep |

- [ ] **Step 2: Update `CLAUDE.md:140`**

Its ADR-053 bullet currently reads "...a fitted model silently imputing the features it was trained
on is not." Add that the fabrication was REPAIRED in this cycle and the ghost path now refuses, so
the bullet stops describing a live defect.

- [ ] **Step 3: Full suite + lint**

Run: `python -m pytest tests/ -m "not e2e" -q --benchmark-skip > "$(mktemp -t sb360_c1_XXXX).log" 2>&1; echo EXIT=$?`
Expected: 0 failed.

- [ ] **Step 4: Propose Commit 1** (owner approval required; do not commit unprompted).

---

## PHASE 2 — Commit 2, the parse port

### Task 6: Extract the scalar affine, with a characterization test built first

**There is NO existing golden-parity gate on the StatsBomb events converter.**
`tests/spadl/test_statsbomb.py`'s only two `assert_frame_equal` calls (`:393`, `:398`) compare
conversions to EACH OTHER. So this task BUILDS the gate, then refactors under it.

**Files:**
- Create: `tests/spadl/test_statsbomb_coordinate_characterization.py`
- Create: `silly_kicks/spadl/_sb_coordinates.py`
- Modify: `silly_kicks/spadl/statsbomb.py:393-427`

**Interfaces:**
- Produces: `sb_xy_to_spadl(xy: np.ndarray, *, fidelity_version: int, y_offset: np.ndarray | float) -> np.ndarray`
  taking an `(N, 2)` array and returning `(N, 2)` in SPADL coordinates, **without clipping**.

- [ ] **Step 1: Build the characterization test on the UNMODIFIED tree**

```python
# tests/spadl/test_statsbomb_coordinate_characterization.py
"""Pins the events converter's coordinate output. NONE existed before this cycle -- the two
assert_frame_equal calls in test_statsbomb.py compare conversions to each other, pinning no values.

Captured on the unmodified tree so the scalar-affine extraction can be proven byte-identical.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.spadl.statsbomb import _convert_locations

_CASES = [[10.0, 20.0], [60.5, 40.5], [119.0, 79.0], [1.0, 1.0], [60.0, 40.0, 2.5]]


def test_fidelity_1_coordinates_are_pinned():
    got = _convert_locations(pd.Series(_CASES), 1)
    np.testing.assert_allclose(
        got,
        np.array(
            [
                [8.3125, 51.4250],
                [52.5000, 34.0000],
                [103.6875, 1.2750],
                [0.4375, 67.5750],
                [52.0625, 34.0425],
            ]
        ),
        rtol=0,
        atol=1e-9,
    )
```

The block above was **generated by running `_convert_locations` on `main` @ `5b1a0a1`**, not
hand-computed. An earlier draft of this plan hand-derived it and got 4 of the 5 rows wrong — in a
task whose entire purpose is pinning coordinate values. If you change `_CASES`, regenerate by running;
do not derive.

Note row 4 (`[60.0, 40.0, 2.5]`): the 3-element form takes `y_offset = 0.05` rather than `crc`, which
is why its y differs from row 1's despite the same input y. That branch is EVENT semantics (a shot's
z-height) and must not reach the polygon path in Task 8.

- [ ] **Step 2: Verify it passes before any refactor**

Run: `python -m pytest tests/spadl/test_statsbomb_coordinate_characterization.py -v`

- [ ] **Step 3: Extract the scalar affine**

```python
# silly_kicks/spadl/_sb_coordinates.py
"""StatsBomb 120x80 -> SPADL affine, WITHOUT the clip.

Split for the reason ADR-038 already split SkillCorner's: `_scale_to_spadl` is affine only and
`_transform_coords` is scale + clamp, because a clamp that is safe for events (on-pitch by
construction) is destructive for anything else. A `visible_area` polygon legitimately extends past
the touchline -- the camera sees beyond it -- so clipping would silently shrink the observed
region, which is the whole quantity.

`_convert_locations` remains the per-ROW wrapper over this. Do NOT call it on a flat polygon:
`[x1,y1,x2,y2,...]` satisfies its `len >= 2` guard and yields only the FIRST vertex, with no error
and no NaN. Reshape to (N, 2) first.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from . import config as spadlconfig

SB_FIELD_LENGTH = 120.0
SB_FIELD_WIDTH = 80.0


def cell_side(fidelity_version: int) -> float:
    return 0.1 if fidelity_version == 2 else 1.0


def sb_xy_to_spadl(
    xy: npt.NDArray[np.float64],
    *,
    fidelity_version: int,
    y_offset: npt.NDArray[np.float64] | float,
) -> npt.NDArray[np.float64]:
    """(N, 2) StatsBomb cell coordinates -> (N, 2) SPADL. No clipping."""
    crc = cell_side(fidelity_version) / 2
    out = np.empty_like(xy, dtype=float)
    out[:, 0] = (xy[:, 0] - crc) / SB_FIELD_LENGTH * spadlconfig.field_length
    out[:, 1] = spadlconfig.field_width - (xy[:, 1] - y_offset) / SB_FIELD_WIDTH * spadlconfig.field_width
    return out
```

- [ ] **Step 4: Re-point `_convert_locations` at it**, keeping its clip and its 3-element
`y_offset = 0.05` branch (both are EVENT semantics and must not reach the polygon path).

- [ ] **Step 5: Prove byte-identity**

Run: `python -m pytest tests/spadl/test_statsbomb_coordinate_characterization.py tests/spadl/test_statsbomb.py -v`
Expected: all pass, unchanged.

---

### Task 7: Extract the port from `build_sb360_coverage.py`

**Files:**
- Create: `silly_kicks/providers/statsbomb/__init__.py`, `silly_kicks/providers/statsbomb/parse.py`
- Modify: `scripts/build_sb360_coverage.py`, `silly_kicks/providers/__init__.py`
- Test: `tests/providers/statsbomb/test_parse.py`

**Interfaces:**
- Produces:
  `shape_snapshots(frames_raw: list[dict], actions: pd.DataFrame, *, fidelity_version: int = 1) -> tuple[pd.DataFrame, pd.DataFrame, JoinReport]`
  returning `(snapshots, visible_area, report)`.
  `defending_gk_visible(players) -> bool`, `acting_side_gk_visible(players) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/providers/statsbomb/test_parse.py
from __future__ import annotations

import pandas as pd

from silly_kicks.providers.statsbomb import shape_snapshots

_FF = [
    {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
    {"teammate": False, "actor": False, "keeper": True, "location": [118.0, 40.0]},
]


def _actions():
    return pd.DataFrame(
        {"action_id": [0], "original_event_id": ["uuid-1"], "game_id": [1],
         "period_id": [1], "time_seconds": [10.0], "start_x": [60.0], "start_y": [34.0]}
    )


def test_snapshots_carry_the_contract_columns():
    raw = [{"event_uuid": "uuid-1", "freeze_frame": _FF, "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0]}]
    snaps, _va, _r = shape_snapshots(raw, _actions())
    assert set(snaps.columns) >= {"action_id", "team_id", "player_id", "is_goalkeeper", "x", "y"}
    assert len(snaps) == 2


def test_is_goalkeeper_comes_from_the_keeper_flag():
    raw = [{"event_uuid": "uuid-1", "freeze_frame": _FF, "visible_area": []}]
    snaps, _va, _r = shape_snapshots(raw, _actions())
    assert snaps["is_goalkeeper"].tolist() == [False, True]


def test_zero_overlap_is_COUNTED_not_silently_empty():
    """3 of 22 open matches ship a 360 file with zero event_uuid overlap. The script already
    picked WARN + match_join_rate=0.0; the port adopts it."""
    raw = [{"event_uuid": "no-such-uuid", "freeze_frame": _FF, "visible_area": []}]
    with pd.option_context("mode.chained_assignment", None):
        snaps, _va, report = shape_snapshots(raw, _actions())
    assert report.join_rate == 0.0
    assert report.n_frames == 1 and report.n_mapped == 0
```

- [ ] **Step 2: Run to verify failure** (`ModuleNotFoundError: silly_kicks.providers.statsbomb`).

- [ ] **Step 3: Move `_defending_gk_visible`, `_acting_side_gk_visible` and the join logic**
into `parse.py`, dropping the leading underscore on the two that become public. Keep their
docstrings verbatim — they carry the measured reasoning about which keeper is "the" keeper.

**⚠ This edits a file the spec claimed the cycle would not touch.**
`tests/scripts/test_build_sb360_coverage.py` asserts directly on `mod._defending_gk_visible`
(`:68, :71, :84, :90`) and `mod._acting_side_gk_visible` (`:83, :91`), so moving and renaming them
breaks it. Spec §3 names `tests/scripts/` as the concurrent Cycle B's territory and states "this cycle
touches none of those" — **that claim is now false**, and the other session is executing there right
now.

Choose BEFORE starting Phase 2, and record the choice:

* **Edit the test** as part of this task and correct spec §3 — cleaner end state, but it puts this
  cycle into Cycle B's file while Cycle B is live.
* **Leave thin underscore-prefixed aliases** in the script (`_defending_gk_visible =
  defending_gk_visible`) so the existing test is untouched — zero conflict surface, one line of
  indirection, and the aliases can be retired in a later cycle.

Default to the second unless the owner says otherwise; the conflict is real and the cost of the alias
is one line.

- [ ] **Step 4: Re-point the script**

`scripts/build_sb360_coverage.py` imports from the port instead of defining them. **Its numbers must
not move** — `docs/research/sb360_coverage/coverage.md` was produced by it.

- [ ] **Step 5: Fix the `providers/__init__.py` docstring**

It currently ends "Behind the ``[parse-dfl]`` extra." That becomes false. Reword so the extra is
attributed to the sportec port specifically.

- [ ] **Step 6: Run**

Run: `python -m pytest tests/providers/ tests/scripts/test_build_sb360_coverage.py -v`

---

### Task 8: `visible_area` — scaled, inverted, NOT clipped

- [ ] **Step 1: Write the failing tests**

```python
def test_a_beyond_touchline_vertex_SURVIVES():
    """The clip is EVENT semantics. A camera legitimately sees past the touchline, so clipping
    would silently shrink the observed region -- ADR-038's defect class."""
    raw = [{"event_uuid": "uuid-1", "freeze_frame": _FF,
            "visible_area": [-5.0, -5.0, 125.0, -5.0, 125.0, 85.0, -5.0, 85.0]}]
    _s, va, _r = shape_snapshots(raw, _actions())
    xs = va.iloc[0]["polygon"][:, 0]
    assert xs.min() < 0.0, "a vertex outside the pitch was clipped away"


def test_a_flat_polygon_round_trips_to_N_vertices():
    """_convert_locations returns (1, 2) on a flat polygon -- first vertex only, silently."""
    raw = [{"event_uuid": "uuid-1", "freeze_frame": _FF,
            "visible_area": [0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0]}]
    _s, va, _r = shape_snapshots(raw, _actions())
    assert va.iloc[0]["polygon"].shape == (4, 2)
```

- [ ] **Step 2: Implement** — reshape flat to `(N, 2)`, call `sb_xy_to_spadl` with
`y_offset=cell_side(fidelity)/2` (the 3-element shot-height branch must NOT apply), no clip.

- [ ] **Step 3: MEASURE the `crc` question and record the answer**

Run on the committed slice: compute polygon area with and without `crc`, and check player positions
against the polygon both ways. `scripts/build_sb360_coverage.py:163` (`_visible_fraction`) applies
**no** cell-centre correction, no y-inversion and no clip, so if the port applies `crc` two readers
of the same polygon in the same repo disagree. Record the decision and its measurement in the port's
module docstring.

---

### Task 9: The committed slice, licensing, and Commit 2's caller sweep

- [ ] **Step 1: Extend `tests/datasets/statsbomb/`** with a `three-sixty/` sibling and a
`SOURCE_SHA`. Do NOT create a parallel directory — the existing one carries the license note.

- [ ] **Step 2: The golden test reads the slice with stdlib `json`.** `statsbombpy` is a SCRIPT
dependency, lazily imported in `build_sb360_coverage.py` and declared nowhere in `pyproject.toml`;
an `importorskip`-guarded golden gate is vacuously green wherever it is absent.

- [ ] **Step 3: Add a `NOTICE` entry** naming the **StatsBomb Public Data License
(non-commercial)**. `NOTICE` currently has zero StatsBomb entries while carrying kloppy's BSD-3-Clause.

- [ ] **Step 4: Commit 2's caller sweep** — §1.8 covered Commit 1 only.
`tests/datasets/statsbomb/spadl-WorldCup-2018.h5` is a **clause-(c) committed artifact derived from
the changed function** (generated by `scripts/build_worldcup_fixture.py`, consumed by
`tests/conftest.py:18`, `test_enrichment_provider_e2e.py:50`, `vaep/test_labels_windowing_e2e.py:12`,
`spadl/test_add_possessions.py:878`). Task 6's characterization test is its evidence.

- [ ] **Step 5: Clause (e), the second hop.** `build_sb360_coverage.py` takes `require_clean_tree`.
Decide and record: does `coverage.md` need re-running after the re-point, or an explicit declaration
that it remains valid? Its numbers must be unchanged by construction — Step 4 of Task 7 asserts that.

- [ ] **Step 6: Register the port's modules AND add real Examples sections**

**Registration is separate from documentation and is not optional.** `_PUBLIC_MODULE_FILES`
(`tests/test_public_api_examples.py:137`) is a hand-maintained tuple pinned to the derived surface by
the meta-assertion at `:1409` — a module in the derived surface but absent from the tuple FAILS CI.
Both `silly_kicks/providers/statsbomb/__init__.py` and `parse.py` land in it under rule P2.

Spec §2.5: `tests/test_public_api_examples.py` rule **P2** is "no underscore-prefixed path component",
so `silly_kicks/providers/statsbomb/*` lands in its derived surface and every public symbol needs a
real Examples section. Do this HERE rather than discovering it at the Step 7 full-suite run, by which
point the port has several public functions.

A `>>> f(x)  # doctest: +SKIP` stub is rejected by `_demonstrates_something`. Use an indented RST
literal block — `_has_real_example` accepts a >=4-space-indented non-`>>>` line in the Examples
section — since a meaningful call needs real payloads.

Run: `python -m pytest tests/test_public_api_examples.py -q`

- [ ] **Step 7: Full suite + lint + pyright.**

- [ ] **Step 8: Propose Commit 2** (owner approval required).

---

## PHASE 3 — merge

### Task 10: Version, CHANGELOG, TODO, ADR

- [ ] **Step 1:** Read the version off `main` (expect 4.76.0) and write **five sites**:
`pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, the CHANGELOG heading, TODO "Current release".
- [ ] **Step 2:** Write the ADR (expect ADR-054).
- [ ] **Step 3:** Queue `_defending_goal` explicitly — a `TODO.md` row, or add `_ghost_gk.py` to the
D3 unit in `tests/tracking/test_mirror_registry.py:294-311` and let the membership assertion trip
deliberately. **Do not leave it merely "declared"** — it appears zero times in `CLAUDE.md`, `TODO.md`
and every ADR, so a bare mention drops it.
- [ ] **Step 4:** Merge with `--merge`, never squash.
