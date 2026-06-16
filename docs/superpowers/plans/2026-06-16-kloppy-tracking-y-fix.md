# kloppy-tracking-y Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Fix the kloppy tracking gateway's inverted y-axis (CS-pin) and single-source the IDSSE/Sportec
parser, eliminating the train/serve drift the bug exposed.

**Architecture:** Spec at `docs/superpowers/specs/2026-06-15-kloppy-tracking-y-fix-design.md` (rev-4-final).
Sequenced PRs: **PR-S94 (T1)** CS-pins the gateway (calibration/external-consumer fix) — fully detailed
below; **PR-S95 (T3)** the `silly-kicks[parse-dfl]` parse port + IDSSE re-route — outlined (executable
detail gated on the lakehouse cross-repo decisions, §PR-S95); **T2** is a measured no-op (Gate D clean),
documented in ADR-031.

**Tech stack:** Python 3.10–3.14, pandas/numpy, kloppy ≥3.18, pytest, ruff, pyright. DGX
(`ssh karsten@192.168.68.73`, venv `~/sk-s93-venv`, `source ~/.pining_env`,
`export PINING_CACHE_DIR=~/Development/silly-kicks/xt_bandwidth_run/artifact_cache`) for all
data-bearing gates.

**Owner-policy adaptations (override the skill's per-task commits):** ONE commit per branch, on the
feature branch `pr-s94-kloppy-tracking-y` (no worktree). There are **NO intermediate commits**: every
task below ends in a **local verify checkpoint** (`pytest`/`ruff`/`pyright`) and **staging** of its
files; the single commit happens only in Task 5.4, **after `/final-review` and explicit owner
approval**. Never tag before CI is green.

**RED-first under single-commit (M1):** "RED-first" here means *run the new test, observe the failure,
and capture the failing output as the sensitivity proof* (paste into the PR description / a plan note)
— it is **NOT** a git commit. Every test authored RED ships **green** in the Task 5.4 commit (made
green by the fix). "Commit it RED" would be unsatisfiable under the single-commit policy — do not do it.

---

# PR-S94 — CS-pin the kloppy tracking gateway (T1)

**Branch:** `pr-s94-kloppy-tracking-y` off `main`.
**Scope reminder (Gate C):** this fixes the silly-kicks **calibration/pining** path + external
kloppy-gateway consumers — NOT lakehouse production (which builds SC/Metrica via its own `convert.py`).
Retrain = calibration recommendations only; Metrica retrain conditional on Gate A.

## Phase 0 — Gate 0: pin the `transform()` signature (DGX, measurement-first)

The exact gateway change depends on whether `to_coordinate_system` coexists with `to_pitch_dimensions`
(R1). Measure before coding.

### Task 0.1: Confirm CS-pin equivalence + pin the signature

- [ ] **Step 1: Write the DGX probe** `scripts/_tf48_cspin_equiv.py` (or reuse if present): load one
      SkillCorner match via kloppy; build frames two ways and compare to the *current* gateway output.

```python
# scripts/_tf48_cspin_equiv.py  (PERSISTED DGX probe — S3: Gate 0's verdict is load-bearing for the
# Task 3.1 signature, so keep it reproducible alongside the other scripts/_tf48_*.py, per spec §5)
import os
import numpy as np
from kloppy import skillcorner
from kloppy.domain import Dimension, MetricPitchDimensions, Orientation
from silly_kicks.spadl._kloppy_coordinates import socceraction_coordinate_system  # after Task 1.1
# ... load a cached SkillCorner TrackingDataset `ds` (see _loader_pining._build_skillcorner) ...
dims = MetricPitchDimensions(x_dim=Dimension(0,105.0), y_dim=Dimension(0,68.0),
                             standardized=False, pitch_length=105.0, pitch_width=68.0)
cur = ds.transform(to_pitch_dimensions=dims, to_orientation=Orientation.HOME_AWAY)          # current (buggy)
cs  = socceraction_coordinate_system(ds.metadata)
A = ds.transform(to_pitch_dimensions=dims, to_orientation=Orientation.HOME_AWAY, to_coordinate_system=cs)  # both kwargs
B = ds.transform(to_orientation=Orientation.HOME_AWAY, to_coordinate_system=cs)                            # CS only
# For each candidate, gather (x,y) per player-frame and compare to cur: expect x==cur.x, y==68-cur.y.
# Report max|cand.x - cur.x| and max|cand.y - (68-cur.y)| for A and B.
```

- [ ] **Step 2: Run on DGX**

Run: `ssh karsten@192.168.68.73 'cd ~/Development/silly-kicks && source ~/.pining_env; export PINING_CACHE_DIR=~/Development/silly-kicks/xt_bandwidth_run/artifact_cache; ~/sk-s93-venv/bin/python scripts/_tf48_cspin_equiv.py skillcorner'`
Expected: exactly one of A/B gives `max|x diff|≈0 AND max|y-(68-cur.y)|≈0` (the report's max-dev-0.000).

- [ ] **Step 3: Record the verdict** in the spec §4.2 / a plan note: the winning signature is the one
      used in Task 3.1. (Prior evidence: adding `to_coordinate_system` to the existing call — candidate
      A — reproduced `(x, 68−y)`; confirm.)

> NOTE: Task 0.1 imports `socceraction_coordinate_system`, so it runs *after* Task 1.1 lands on the DGX
> tree (scp the new module first). It is gating for Task 3.1 only — Phase 1/2 do not depend on it.

## Phase 1 — Shared `_kloppy_coordinates` extraction (byte-equivalent event path)

### Task 1.1: Create the shared module

**Files:** Create `silly_kicks/spadl/_kloppy_coordinates.py`; Modify `silly_kicks/spadl/kloppy.py:293-326`.

- [ ] **Step 1: Create `_kloppy_coordinates.py`** — move `_SoccerActionCoordinateSystem` verbatim
      (currently `spadl/kloppy.py:293-326`) + add the helper.

```python
"""Shared kloppy→SPADL coordinate system, used by BOTH the event gateway (spadl/kloppy.py) and the
tracking gateway (tracking/kloppy.py) so events and frames cannot drift. See ADR-031."""
from __future__ import annotations

from kloppy.domain import (  # type: ignore[reportMissingImports]
    CoordinateSystem, Dimension, MetricPitchDimensions, Origin, PitchDimensions, Provider,
    VerticalOrientation,
)

from . import config as spadlconfig


class _SoccerActionCoordinateSystem(CoordinateSystem):
    def __init__(self, *, pitch_length: float, pitch_width: float) -> None:
        self._pitch_length = pitch_length
        self._pitch_width = pitch_width

    @property
    def provider(self) -> Provider:
        return "SoccerAction"  # type: ignore[reportReturnType]

    @property
    def origin(self) -> Origin:
        return Origin.BOTTOM_LEFT

    @property
    def vertical_orientation(self) -> VerticalOrientation:
        return VerticalOrientation.BOTTOM_TO_TOP

    @property
    def pitch_length(self) -> float:  # type: ignore[override]
        return self._pitch_length

    @property
    def pitch_width(self) -> float:  # type: ignore[override]
        return self._pitch_width

    @property
    def pitch_dimensions(self) -> PitchDimensions:
        return MetricPitchDimensions(
            x_dim=Dimension(0, spadlconfig.field_length),
            y_dim=Dimension(0, spadlconfig.field_width),
            pitch_length=self._pitch_length,
            pitch_width=self._pitch_width,
            standardized=True,
        )


def socceraction_coordinate_system(metadata) -> _SoccerActionCoordinateSystem:
    """Build the canonical SPADL coordinate system from a kloppy dataset's metadata."""
    cs = metadata.coordinate_system
    return _SoccerActionCoordinateSystem(pitch_length=cs.pitch_length, pitch_width=cs.pitch_width)
```

- [ ] **Step 2: Re-export from `spadl/kloppy.py`** — delete the class body (`:293-326`), add at the top
      (near the other local imports, ~`:46-50`):

```python
from ._kloppy_coordinates import _SoccerActionCoordinateSystem, socceraction_coordinate_system
```

      and replace the inline construction at `:198-201` with the helper:

```python
    new_dataset = dataset.transform(
        to_orientation=Orientation.HOME_AWAY,
        to_coordinate_system=socceraction_coordinate_system(dataset.metadata),
    )
```

      Remove now-unused kloppy-domain imports from `spadl/kloppy.py` only if they are unused elsewhere
      (check `Origin`, `VerticalOrientation`, `PitchDimensions`, `CoordinateSystem` — `pyright` + `ruff`
      will flag; keep any still referenced).

- [ ] **Step 2b: Confirm the call-site refactor is byte-equivalent (D — it's a move PLUS a call-site
      change, not a pure move).** The helper reads `metadata.coordinate_system.pitch_length/pitch_width`;
      diff that against the ORIGINAL inline construction (`spadl/kloppy.py:198-201`) before deleting it.
      **Verified 2026-06-16:** the original inline already passed
      `pitch_length=dataset.metadata.coordinate_system.pitch_length`,
      `pitch_width=dataset.metadata.coordinate_system.pitch_width` — identical source, so the helper is
      exact. Record this; if a future reader finds the original sourced dims differently, the helper must
      mirror it.

- [ ] **Step 3: Verify the event path is byte-identical**

Run: `python -m pytest tests/spadl/test_kloppy.py -v --tb=short`
Expected: PASS unchanged (the extraction is a pure move + helper; event output identical).

- [ ] **Step 4: Lint/type checkpoint**

Run: `ruff format --check silly_kicks/spadl/ && ruff check silly_kicks/spadl/ && pyright silly_kicks/`
Expected: clean.

### Task 1.2: Event-path byte-equivalence golden (guards the extraction durably)

**Files:** Create `tests/spadl/test_kloppy_coordinates_extraction.py`.

- [ ] **Step 1: Write the test** — assert the event conversion output is byte-identical on a committed
      kloppy fixture. Prefer a data assertion over import-identity (N3). Use the committed
      `sportec_dataset` / `metrica_dataset` event fixtures `test_kloppy.py` already loads. **The fixture
      must have NON-DEFAULT pitch dimensions (D)** — i.e. `metadata.coordinate_system.pitch_length/width`
      ≠ 105×68 — so the golden actually exercises the helper's metadata-sourcing and can't pass
      vacuously; if both committed fixtures are 105×68, add a second fixture (or override the
      coordinate_system) with e.g. 105.3×68.5 and assert that flows through unchanged.

```python
# tests/spadl/test_kloppy_coordinates_extraction.py
import pandas as pd
from silly_kicks.spadl import kloppy as spk
# Reuse the fixture loader test_kloppy.py uses (a committed StatsBomb/Metrica kloppy sample).
def test_event_conversion_byte_identical_after_extraction(kloppy_sample_dataset):
    actions, _ = spk.convert_to_actions(kloppy_sample_dataset)
    golden = pd.read_parquet("tests/spadl/_golden_kloppy_actions.parquet")
    pd.testing.assert_frame_equal(actions.reset_index(drop=True), golden, check_dtype=True)
```

- [ ] **Step 2: Capture the golden** from the *pre-change* output (run on `main` before Task 1.1, or via
      `git stash`): `actions.to_parquet("tests/spadl/_golden_kloppy_actions.parquet")`. **Stage** it
      (ships in the Task 5.4 commit — no intermediate commit).
- [ ] **Step 3: Run** `python -m pytest tests/spadl/test_kloppy_coordinates_extraction.py -v` → PASS.

## Phase 2 — RED-first guards (run failing to prove sensitivity; ship green in Task 5.4)

### Task 2.0: Capture minimal committed fixtures (DGX)

**Files:** Create `tests/datasets/tracking/ytest/{skillcorner,metrica}_yident_slice.parquet` (actions +
shooter-window frames; minimal, off-centre).

- [ ] **Step 1: Write the capture script** `scripts/_capture_yident_fixtures.py` (DGX): for 1 SC + 1
      Metrica match, select HOME off-centre shots (`|start_y−34|>8`) + the shooter's frames within ±0.5 s,
      write a tiny parquet (actions slice + frame slice). Follow the `tests/regressions/extratime/
      capture_goldens.py` precedent.
- [ ] **Step 2: Run on DGX**, scp the resulting parquets into `tests/datasets/tracking/ytest/`.
      Verify each is < ~200 KB (R5: stay in-repo).
- [ ] **Step 3: Stage the fixtures** (ship in the Task 5.4 commit — no intermediate commit).

### Task 2.1: Gateway parity contract test (RED)

**Files:** Create `tests/tracking/test_kloppy_tracking_y_parity.py`.

- [ ] **Step 0: Bind the fixture (B — it is NOT Task 2.0's parquets).** This test runs BOTH
      `spk.convert_to_actions(ds)` and `tracking_kloppy.convert_to_frames(ds)`, so it needs a live kloppy
      **TrackingDataset that carries events + tracking** — a dataset *object*, not the post-conversion
      parquet slices Task 2.0 captures (those feed Task 2.2 only). Resolve in this order:
      (a) build a **Metrica** kloppy tracking dataset from committed raw files — check
      `tests/datasets/tracking/metrica/` for raw tracking + `tests/datasets/kloppy/metrica_events.json`
      + `epts_metrica_metadata.xml` (the loader pattern is in `tests/regressions/extratime/_builders.py`);
      (b) if no committed raw Metrica tracking exists, **build a minimal synthetic kloppy TrackingDataset
      in-test** (a few frames + matching events with known off-centre coords) — **shared with Task 3.2**
      via a common `conftest` fixture. Name the chosen fixture explicitly; do NOT leave
      `kloppy_sample_dataset` unbound.
- [ ] **Step 1: Write the test** — on that kloppy fixture, assert event-gateway canonical-y ==
      tracking-gateway canonical-y for the same player/instant (precedent: lakehouse `test_convert_drift.py`).
      **Must isolate the y-mirror from orientation (S1):** restrict to off-centre y (`|start_y−34|>8`),
      compare action↔**shooter** (same player + nearest instant, gap<0.05 s), and reproject via the
      library transform (`tracking._action_orientation.acting_team_attacks_rtl` +
      `reproject_to_action_ltr`, §7.6) so an away-team 180° orientation difference can't masquerade as the
      bug. **Verify the `output_convention` literal against the converter** (S2): `tracking/kloppy.py`
      declares `Literal["absolute_frame", "ltr"]` — `"absolute_frame"` is valid; do not pass `"absolute"`.

```python
# FAILS today (tracking y ≈ 68-2y off vs canonical), PASSES after Task 3.1
def test_event_and_tracking_gateways_agree_on_y(kloppy_sample_dataset):
    actions, _ = spk.convert_to_actions(kloppy_sample_dataset)
    frames, _ = tracking_kloppy.convert_to_frames(kloppy_sample_dataset, output_convention="absolute_frame")
    # off-centre HOME (ltr) shots; nearest-instant shooter frame; library reproject; action↔shooter
    # (NOT action↔ball). For each: assert abs(action_start_y - shooter_frame_y) < 1.0
    ...  # join glue finalized once the committed fixture columns (Task 2.0) are known
```

- [ ] **Step 2: Run → confirm RED** (`python -m pytest tests/tracking/test_kloppy_tracking_y_parity.py -v`);
      capture the failing output (≈ `68−2y` gap) as the sensitivity proof in the PR description / a plan
      note — **not a commit** (M1). It ships green in Task 5.4.

### Task 2.2: Cross-provider y-identity golden (RED — SkillCorner + Metrica)

**Files:** Create `tests/tracking/test_kloppy_y_identity_golden.py`.

- [ ] **Step 1: Write the test** — parametrized over `{skillcorner, metrica}` fixtures (Task 2.0):
      assert acting-player frame-y ≈ action `start_y` (off-centre, `|start_y−34|>8`), using the library
      orientation transform; compare action↔shooter, never ball (diagnosis discipline). This is the
      silly-kicks half of Gate E; the native-IDSSE entry is added in PR-S95 (green-from-start).

```python
import pytest
@pytest.mark.parametrize("provider", ["skillcorner", "metrica"])
def test_acting_player_frame_y_matches_action_off_centre(provider):
    actions, frames = _load_yident_slice(provider)  # from tests/datasets/tracking/ytest/
    ... # off-centre HOME shots; nearest-time shooter frame; assert |action_start_y - frame_y| < 1.5
```

- [ ] **Step 2: Run → confirm RED** for both providers; capture the failing output as the sensitivity
      proof (PR description / plan note) — **not a commit** (M1). Ships green in Task 5.4.

> Metrica caveat: Gate A may show Metrica was already canonical (CS-pin no-op). If the Metrica slice is
> *not* RED on `main`, mark it `xfail(reason="Gate A pending")` until Task 4.1 resolves whether Metrica
> moved — do NOT assert a fix for a provider that was already correct (N4).

## Phase 3 — CS-pin the gateway (RED → GREEN)

### Task 3.1: Add `to_coordinate_system` to the tracking gateway

**Files:** Modify `silly_kicks/tracking/kloppy.py:104-113`.

- [ ] **Step 1: Apply the Gate-0-pinned signature — candidate B (CS-ONLY; Gate 0 proved A is a
      silent non-fix).** REMOVE `to_pitch_dimensions` and rely on the CS's own standardized dims
      (matches the event gateway). The `MetricPitchDimensions`/`Dimension` imports in `tracking/kloppy.py`
      become unused → prune them (pyright/ruff will flag):

```python
    from silly_kicks.spadl._kloppy_coordinates import socceraction_coordinate_system
    transformed = dataset.transform(
        to_orientation=Orientation.HOME_AWAY,
        to_coordinate_system=socceraction_coordinate_system(dataset.metadata),  # NEW (ADR-031, Gate-0 cand. B)
    )
```

- [ ] **Step 2: Run the RED guards → GREEN**

Run: `python -m pytest tests/tracking/test_kloppy_tracking_y_parity.py tests/tracking/test_kloppy_y_identity_golden.py -v`
Expected: PASS (SkillCorner certainly; Metrica per Gate A / its xfail status).

- [ ] **Step 3: Regression — existing tracking tests still pass**

Run: `python -m pytest tests/tracking/ -m "not e2e and not slow" -q`
Expected: PASS (orientation tests, adapter tests unaffected — they use sportec/GS native or
already-labeled frames).

- [ ] **Step 4: Lint/type** `ruff format --check . && ruff check . && pyright silly_kicks/` → clean.

### Task 3.2: Committed "no-op on an already-canonical provider" unit test (S4)

**Files:** Create `tests/tracking/test_kloppy_cs_pin_noop_canonical.py`.

- [ ] **Step 1: Write the test** — durably guard the "never double-invert a clean provider" property the
      spec leans on (why a blanket flip is wrong). Build a synthetic kloppy `TrackingDataset` already in
      `BOTTOM_TO_TOP` (canonical) and assert the CS-pinned `convert_to_frames` is **identity in y**
      (y unchanged, not `68−y`). Green-from-start (it's a safety guard, not RED-first).

```python
# A synthetic already-canonical dataset must come out y-unchanged (no double-invert).
def test_cs_pin_is_y_identity_on_already_canonical_dataset(synthetic_bottom_to_top_dataset):
    frames, _ = tracking_kloppy.convert_to_frames(synthetic_bottom_to_top_dataset,
                                                  output_convention="absolute_frame")
    # for each player-frame, frame_y == the input canonical y (within fp tol), NOT 68 - y
    ...
```

- [ ] **Step 2: Run** → PASS. Stage (ships in Task 5.4).

## Phase 4 — DGX gates A/B + blast-radius/sensitivity e2e

> **Coverage split (E2E note) — state it so no reader over-claims:** committed automated coverage =
> gateway parity (Task 2.1) + coordinate identity (Task 2.2) + no-op safety (Task 3.2) + blast-radius
> sensitivity (Task 4.3). **Feature-value fix-correctness** (action↔shooter lands at identity post-fix)
> rides the **manual DGX Gates A/B** — measurements recorded in ADR-031, not committed tests. Task 4.3
> proves *which columns depend on y*, NOT that the post-fix values are right.

### Task 4.1: Gate A — Metrica (HARD pre-merge gate)

- [ ] **Step 1:** On DGX, run the off-centre action↔shooter localization on ≥2 Metrica matches with vs
      without the CS-pin (adapt `scripts/_tf48_clean_localize.py`). Record: does the CS-pin land Metrica
      at identity (was inverted), or is Metrica already canonical (no-op)?
- [ ] **Step 2: Decide the Metrica retrain trigger (N4):** trigger ONLY if values moved. Update §8 +
      CHANGELOG accordingly. If no-op, un-xfail Task 2.2's Metrica case only if it now asserts the
      pre-existing-correct invariant; otherwise keep it as a plain green guard.

### Task 4.2: Gate B — SkillCorner re-verify

- [ ] **Step 1:** On DGX, post-fix SC action↔shooter at identity (off-centre); confirm residual ≈0.2 m
      and decide whether `spadl/skillcorner.py` events need their own flip (evidence says NO; confirm).
- [ ] **Step 2:** Record the SC residual + events-converter decision in the spec/ADR.

### Task 4.3: Blast-radius / sensitivity A/B e2e (committed)

**Files:** Create `tests/tracking/test_y_blast_radius_ab.py`.

- [ ] **Step 1: Scope to OFFLINE-computable features (C — must be CI-green, no weight fetch / no
      network).** Restrict the A/B set to the geometric features that compute fully from committed inputs:
      `add_action_context` (`nearest_defender_distance`, `defenders_in_triangle_to_goal`,
      `receiver_zone_density`), `add_pre_shot_gk_*` distances/angles, `add_pressure_on_actor`. **Exclude**
      the model/cache-dependent §6 members (`obso`, `pitch_control`, `xt_gk`, `gk_influence`,
      `player_influence`, `space_creation`, `das`) — they need ghost-GK weights / xT surfaces / pinning
      caches and would make the test fetch artifacts or skip; they are covered by the manual Gate B only.
      Add a one-line comment in the test naming the excluded set + why.
- [ ] **Step 2:** On a committed SC slice, run the offline subset twice (frames as-is vs `y=68−y`) and
      assert the CORRUPTED geometric columns change and the IMMUNE columns do not. **Sensitivity** e2e
      (which columns depend on y) — NOT fix-correctness (manual Gates A/B). Stage as a durable guard.
- [ ] **Step 3: Run offline** (no network) → PASS.

## Phase 5 — Cross-repo handoff + ADR-031 + release + commit

### Task 5.0: Issue the Gate-C lakehouse handoff (M2 — the actual prod gap)

PR-S94 fixes the gateway (calibration); **lakehouse-prod SkillCorner/Metrica y-correctness is built by
the lakehouse's own `convert.py` and is NOT touched by any PR here** — it must not fall through the
cross-repo crack. This task *issues* the handoff (the lakehouse runs it); track it as an open cross-repo
TODO until they report back.

- [ ] **Step 1:** Add a `TODO.md` cross-repo item: "Lakehouse: verify `_bronze_metrica_to_frames`
      (`convert.py:279`, flips y `(1−y)*68`) + `_bronze_skillcorner_to_frames` (`convert.py:390`, `y+34`
      NO flip — the asymmetry is the red flag) for a y-mirror; add a lakehouse y-identity golden. Their
      `test_frame_orientation_golden.py` is x-based (blind to a y-mirror) and `correct_frames_to_home_ltr`
      is a 180° net (can't fix a single-axis flip) → prod SC/Metrica is currently y-unguarded."
- [ ] **Step 2:** Relay the Gate-C handoff block (off-centre action↔shooter diagnostic on both builders
      + a distinct lakehouse y-identity golden) for the lakehouse session to run. NOT asserted broken
      without their measurement.

### Task 5.1: Write ADR-031

**Files:** Create `docs/superpowers/adrs/ADR-031-kloppy-tracking-y.md`.

- [ ] Record: root cause (gateway asymmetry), CS-pin (not blanket flip), §6 blast radius + the A/B
      verdict, Gate-C resolution (PR-S94 = calibration not lakehouse-prod), the Gate-D verdict (native
      clean per period; ET unverified), the parse-port decision + C4 release-coupling (forward-ref to
      PR-S95). Repo-qualify ADR numbers (N2).

### Task 5.2: Version bump (hard gate — all must match)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`.

- [ ] Bump `4.28.0 → 4.29.0` in `pyproject.toml` + `silly_kicks/__init__.py`; add the CHANGELOG entry
      (note the **calibration** retrain trigger + Metrica-conditional; state that **lakehouse adoption of
      4.29.0 is NOT correctness-urgent** — Gate C: lakehouse prod doesn't route through the fixed
      gateway, so it can ride a later bump — S5).
- [ ] **TRIM, do NOT delete (M3),** the "### Confirmed bugs" kloppy-y row in `TODO.md`: PR-S94 fixes
      only the gateway. Leave it as "remaining: PR-S95 parse port + lakehouse SC/Metrica builder
      y-verification (Task 5.0) + ET native handedness (C2, unverified)". Deleting it would claim closure
      the codebase hasn't reached (Hyrum/honesty on a consumed artifact).
- [ ] `uv lock` to refresh `uv.lock`. Verify all five (pyproject + __init__ + CHANGELOG + TODO + uv.lock)
      agree (the version-bump gate).

### Task 5.3: `/final-review`

- [ ] Run the `/final-review` gate. Address findings. Re-run
      `python -m pytest tests/ -m "not e2e" -q && ruff format --check . && ruff check . && pyright silly_kicks/`.

### Task 5.4: Single commit (ONLY after explicit owner approval)

- [ ] Stage all PR-S94 changes; one commit message ending with the Co-Authored-By trailer. Do NOT tag.
      Wait for CI green (owner monitors), then tag per the never-tag-before-green policy.

---

# PR-S95 — `silly-kicks[parse-dfl]` parse port + IDSSE re-route (T3) — OUTLINE

Detailed plan deferred until the lakehouse responds to the §4.4/§4.5 handoffs (which canonical
smoother+velocity callable; port-adoption commitment). Executable shape:

1. **New `silly_kicks/providers/sportec/parse.py`** behind a `[parse-dfl]` extra (pyproject); typed
   returns `MatchInfo`/`EventRow`/`TrackingFrames`; lift the lakehouse pure parse fns (raw DFL X/Y, no
   flip — confirmed). Parse-port **semantic-equality** parity test vs the committed DFL slice (S2),
   RED-first.
2. **Re-route `_loader_pining.py` IDSSE** (`:300-315`) to port-parse → native converters; retire
   `_kloppy_tracking_to_frames` (`:318-359`) + the kloppy IDSSE events path (`:304`). Update the
   `game_id`-None workaround + `test_calibrate_cli.py:36` (R4 — grep all consumers first).
3. **Numeric single-source gate (B4):** end-to-end `convert(velocity(smooth(parse)))==production`
   parity test (C1 order); or a bounded+tracked accepted residual (C3).
4. **native-IDSSE y-identity golden** added to the §7.1 cross-provider golden — green-from-start.
5. **Lakehouse adoption (separate lakehouse PR):** S3 checklist (pyproject hard-dep, terraform `==`
   pin/ADR-046, PEP-723 footgun). **Not correctness-urgent (S5)** — the lakehouse can adopt on its own
   cadence; the genuinely urgent lakehouse item is the independent Task-5.0 builder y-check, not this
   version bump. ADR-031 amended.
6. Version bump, `/final-review`, single commit, CI-green-then-tag.

# T2 — native sportec handedness — DONE (no code)

Gate D measured clean per period (§4.3). No fix, no retrain. Documented in ADR-031; the native-IDSSE
golden in PR-S95 is the durable guard. Optional thin PR only if you want a standalone native-handedness
invariant isolated from PR-S95.

---

## Self-review notes (rev-5 + rev-6 applied)
- **rev-6 readiness fixes:** (A) Phase 2 header no longer says "commit failing"; (B) Task 2.1 binds an
  explicit live kloppy TrackingDataset fixture (Metrica-committed or synthetic, shared with 3.2 — NOT
  Task 2.0's parquets); (C) Task 4.3 scoped to offline geometric features (model/cache-dependent §6
  members excluded, covered by manual Gate B); (D) Task 1.1 diffs the call-site refactor (verified the
  original inline already read `metadata.coordinate_system` dims) + Task 1.2 uses a non-default-pitch
  fixture so the byte-equivalence golden isn't vacuous.
- **Spec coverage:** T1 fully tasked (extraction, CS-pin, parity, cross-provider golden, no-op safety
  test, Gates 0/A/B, sensitivity e2e, **Gate-C lakehouse handoff (Task 5.0 — M2)**, ADR, release);
  Gate-C reframe in scope + retrain; Gate D = no-op tasked as doc; PR-S95/T2 outlined with gating stated.
- **RED-first under single-commit (M1):** "RED" = run + observe + capture failing output as evidence,
  NOT a commit; tests ship green in Task 5.4. Tasks 2.1/2.2 reworded accordingly; Task 1.2
  (byte-equivalence), Task 3.2 (no-op safety), and the native-IDSSE golden are green-from-start guards.
- **RED tests isolate the y-mirror from orientation (S1):** both 2.1 and 2.2 use the library reproject +
  action↔shooter + off-centre filter, so an away-team 180° difference can't be mistaken for the bug.
- **Honest closure (M3):** Task 5.2 TRIMS the TODO row (gateway fixed; builders + PR-S95 + ET remain),
  never deletes it. Task 4.3 named a sensitivity e2e, not fix-correctness (manual Gates A/B own that).
- **No silent caps:** Metrica `xfail`-gated on Gate A, not assumed broken (N4); ET native handedness
  flagged unverified (C2); lakehouse builder y-correctness explicitly handed off (M2), not assumed.
- **Owner policy:** single commit per PR, `/final-review` before commit, version-bump 5-file gate,
  no-tag-before-green — encoded in Phase 5.
