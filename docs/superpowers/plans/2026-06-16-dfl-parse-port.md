# DFL parse+shape port (PR-S95 / T3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Single-source the IDSSE/Sportec DFL **parse** layer by upstreaming the lakehouse's pure parse
functions into a new `silly_kicks/providers/sportec/` parse+shape port (behind `[parse-dfl]`), and
re-route the dev harness onto it — eliminating the parser-layer dev/prod drift and correcting the
harness's y-inverted IDSSE frames.

**Architecture:** Two composables split at the **bronze seam** (review A) — `parse_dfl_*(xml) ->
SportecTrackingBronze/SportecEventBronze` (faithful bytes→bronze) + `shape_*_to_native(bronze) ->
NativeTrackingInput/NativeEventsInput` (bronze→converter input). Production: `parse → bronze → persist →
shape → convert`; harness: `parse → shape → convert`. Drift killed by a **committed golden** parity
test pinned to lakehouse `0efac60`. Data-quality (smooth/velocity) stays consumer-side. Lakehouse
adoption is a separate lakehouse PR (its choice: delete-and-depend vs keep-both-and-parity).

**Tech stack:** Python 3.10–3.14, stdlib `xml.etree`, pandas, pytest, ruff, pyright. DGX
(`ssh karsten@192.168.68.73`, venv `~/sk-s93-venv`, `source ~/.pining_env`,
`PINING_CACHE_DIR=~/Development/silly-kicks/xt_bandwidth_run/artifact_cache`) for the DFL fixture +
golden capture; the pinned lakehouse clone is `~/Development/luxury-lakehouse` @ `0efac60`.

**Spec:** `docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md` (rev ×2). **Decision:** ADR-031
(amend). **Base:** `main` @ v4.29.0 → **target 4.30.0**.

**Owner-policy adaptations (override the skill's per-task commits):** feature branch
`pr-s95-dfl-parse-port` off `main`, **no worktree**. **NO intermediate commits** — every task ends in a
local verify checkpoint + `git add` (staging); the **single commit** is Task 6.4, after `/final-review`
+ explicit owner approval. RED-first = run + observe failure + capture output, NOT a commit. Never tag
before CI green.

---

## File structure

| Path | Responsibility | Action |
|------|----------------|--------|
| `silly_kicks/providers/__init__.py` | new top-level `providers` namespace (the generalizable per-provider parse-port home) | Create |
| `silly_kicks/providers/sportec/__init__.py` | export the public port surface | Create |
| `silly_kicks/providers/sportec/parse.py` | the parse+shape port: typed bronze dataclasses + `parse_dfl_*` (bytes→bronze) + `shape_*_to_native` (bronze→native) + lifted helpers | Create |
| `pyproject.toml` | add `[parse-dfl]` optional extra | Modify |
| `scripts/_reduce_idsse_dfl_slice.py` | DGX-only, pure stdlib `ET` (either env): build the schema-valid reduced DFL XML slice; assert acceptance by BOTH the lakehouse ET parser AND kloppy (G3) | Create (DGX) |
| `scripts/_capture_lakehouse_golden.py` | DGX-only, run in the **lakehouse clone venv**: import the real `ingestion.idsse` + `analytics.action_context.convert` → capture the `0efac60` bronze/native goldens (G2) | Create (DGX) |
| `scripts/_capture_oldpath_golden.py` | DGX-only, run in **`~/sk-s93-venv`**: run the kloppy `_kloppy_tracking_to_frames` old-path on the shared slice → freeze `idsse_oldpath_harness_golden.parquet` BEFORE Task 3.1 retires it (G2) | Create (DGX) |
| `tests/datasets/sportec/idsse_slice/` | committed reduced DFL XML (positions/events/metadata) + `*_golden.parquet` references | Create (fixtures) |
| `tests/providers/sportec/test_parse_port_parity.py` | RED-first parity vs the committed goldens (bronze + native) | Create |
| `scripts/_loader_pining.py` | re-route IDSSE → port + native converters; retire `_kloppy_tracking_to_frames` + kloppy IDSSE events path + `game_id`-None stamp | Modify |
| `tests/calibration/test_calibrate_cli.py` | update the `game_id`-None expectation (`:36`) | Modify |
| `tests/tracking/test_kloppy_y_identity_golden.py` | add the native-IDSSE entry (green per Gate D) | Modify |
| `tests/calibration/test_calibration_invariance_e2e.py` | harness IDSSE pre/post-re-route invariance (y moves, non-y ~0) | Create |
| `docs/superpowers/adrs/ADR-031-kloppy-tracking-y-inversion.md` | amend (T3 shipped: parse port) | Modify |
| `pyproject.toml`/`__init__.py`/`CHANGELOG.md`/`TODO.md`/`uv.lock` | version 4.30.0 | Modify |

---

## Phase 0 — DGX: schema-valid reduced DFL fixture + committed goldens (review D + N2)

**The capture spans TWO mutually-exclusive environments (G2):** the lakehouse goldens import
`ingestion.idsse` → only importable in the **lakehouse clone venv**; the old-path harness golden runs the
kloppy `_kloppy_tracking_to_frames` from the **silly-kicks repo** → only runnable in `~/sk-s93-venv`. So
the reduced XML is built ONCE by a pure-stdlib script (either env), shared on disk, and consumed by two
env-gated capture scripts. Three scripts, run in sequence.

#### Task 0.1: Build the reduced DFL slice — assert acceptance by BOTH parsers (G3)

**Files:** Create `scripts/_reduce_idsse_dfl_slice.py`; outputs → `tests/datasets/sportec/idsse_slice/{positions,events,info}.xml`.

- [ ] **Step 0: Prep the lakehouse clone's own venv (F1).** The "can't import the lakehouse module" premise
      is only true in silly-kicks' env; in the **lakehouse clone's own venv** the deps
      (`ingestion.guards`/`utils`/`workflows`) are present. On the DGX:
      `cd ~/Development/luxury-lakehouse && uv sync` (its `0efac60` env).
- [ ] **Step 1: Write the pure-stdlib reduction** (`xml.etree` only — runs in either env). STRUCTURE-AWARE
      (review D): parse a real pining IDSSE match's positions XML, KEEP a few whole `<FrameSet>`s (1 ball +
      the GK + ~3 outfielders per team), TRIM each kept FrameSet's `<Frame>` children to a contiguous window
      holding an off-centre action; re-serialise → valid DFL. Same for events XML + metadata. Write the 3
      XML files into `tests/datasets/sportec/idsse_slice/`.
- [ ] **Step 2: Assert acceptance by BOTH parsers (G3 — a hard reduction constraint, not an afterthought).**
      kloppy's DFL parser is materially stricter than the lakehouse's hand-rolled `xml.etree` parser (more
      metadata/element validation — part of why the lakehouse hand-rolled its own). A slice trimmed only to
      satisfy the lenient lakehouse parser may be REJECTED by kloppy → Task 0.3's old-path golden can't be
      captured → Task 4.2 has nothing to compare against. So the reduction must satisfy the **intersection**
      of both parsers' requirements. The reduction script asserts BOTH (run each in its own env, or import
      kloppy here since it lives in `~/sk-s93-venv`):

```python
# scripts/_reduce_idsse_dfl_slice.py -- pure stdlib reduction + dual-parser acceptance gate (DGX).
import xml.etree.ElementTree as ET
# ... reduce -> tests/datasets/sportec/idsse_slice/{positions,events,info}.xml ...
# Gate A -- kloppy (the STRICTER parser; run in ~/sk-s93-venv):
from kloppy import sportec
ds = sportec.load_tracking(meta_data=info_path, raw_data=positions_path, coordinates="secondspectrum")
assert len(ds.records) > 0, "reduced DFL rejected by kloppy -- grow the slice to kloppy's schema floor"
# Gate B -- the lakehouse ET parser (run in the lakehouse clone venv; see Task 0.2):
#   asserted there as `rows and all(rows.values())`.
```

      If kloppy needs a fuller slice, grow the fixture to the intersection (size-check in Step 3 still
      applies). Budget this as a possible kloppy-DFL-schema rabbit hole, not a 1-hour fixture.
- [ ] **Step 3:** Verify the 3 XML files total < ~400 KB (leave headroom under the ~500 KB R1/R5 ceiling for
      the parquet goldens; trim the window if larger). **Stage** (ship in Task 6.4).
- [ ] **Step 4: Record the pin** — write the lifted lakehouse SHA (`0efac60`) into
      `tests/datasets/sportec/idsse_slice/SOURCE_SHA` so re-pinning is a one-line auditable change.

#### Task 0.2: Capture the lakehouse goldens — real module, lakehouse clone venv (F1, G2)

**Files:** Create `scripts/_capture_lakehouse_golden.py` (run in `~/Development/luxury-lakehouse`'s venv);
outputs → `tests/datasets/sportec/idsse_slice/*_golden.parquet`.

- [ ] **Step 1:** Import and run the **REAL** lakehouse functions on the shared reduced slice — NOT a
      hand-copy — so the Task 2.1 parity test is a genuine "port reproduces production" check (F1):

```python
# scripts/_capture_lakehouse_golden.py -- run in ~/Development/luxury-lakehouse (its venv), DGX-only.
from ingestion.idsse import (_parse_teams, _parse_positions_xml, _parse_events_xml,  # the REAL module
                             _IDSSE_TRACKING_BRONZE_COLS, _IDSSE_EVENTS_BRONZE_COLS)
from analytics.action_context.convert import _bronze_idsse_to_sportec_input          # the REAL shaper
_FIX = "<sk-repo>/tests/datasets/sportec/idsse_slice"
# Gate B (G3): the lakehouse ET parser must accept the reduced slice.
rows = _parse_positions_xml(f"{_FIX}/positions.xml", player_team_map, match_id, logger)
assert rows and all(rows.values()), "reduced DFL rejected by the lakehouse parser"
# Capture the GENUINE lakehouse@0efac60 goldens (the frozen reference the port must reproduce):
bronze_trk = _bronze_tracking_df(rows)                  # _IDSSE_TRACKING_BRONZE_COLS
native_trk = _bronze_idsse_to_sportec_input(bronze_trk) # EXPECTED_INPUT_COLUMNS
bronze_evt, native_evt = ...                            # the real events parse + shaper
for df, name in [(bronze_trk,"idsse_parse_bronze_golden"), (native_trk,"idsse_shape_native_golden"),
                 (bronze_evt,"idsse_events_bronze_golden"), (native_evt,"idsse_events_native_golden")]:
    df.to_parquet(f"{_FIX}/{name}.parquet")
```

- [ ] **Step 2:** Run in the lakehouse clone venv. Expected: Gate B passes; 4 golden parquets written;
      print sizes. **Stage.**

#### Task 0.3: Freeze the OLD-PATH harness golden — silly-kicks venv, BEFORE retirement (F2, G2)

**Files:** Create `scripts/_capture_oldpath_golden.py` (run in `~/sk-s93-venv`); output →
`tests/datasets/sportec/idsse_slice/idsse_oldpath_harness_golden.parquet`.

- [ ] **Step 1:** While the kloppy `_kloppy_tracking_to_frames` path STILL EXISTS (before Task 3.1 retires
      it), run it on the SAME shared reduced slice and freeze its output. Task 4.2 asserts
      new-path-vs-this-frozen-golden (so the e2e never reconstructs deleted code, and the N6 inversion
      delta is a committed auditable artifact):

```python
# scripts/_capture_oldpath_golden.py -- run in ~/sk-s93-venv, DGX-only, BEFORE Task 3.1.
from scripts._loader_pining import _kloppy_tracking_to_frames   # still present pre-Task-3.1
_FIX = "tests/datasets/sportec/idsse_slice"
frames_old = _kloppy_tracking_to_frames(meta=f"{_FIX}/info.xml", raw=f"{_FIX}/positions.xml", ...)
frames_old.to_parquet(f"{_FIX}/idsse_oldpath_harness_golden.parquet")
```

- [ ] **Step 2:** Run in `~/sk-s93-venv`. Expected: the old-path golden parquet is written (relies on
      Gate A from Task 0.1 — kloppy accepted the slice); print size. `scp`/stage all parquets into
      `tests/datasets/sportec/idsse_slice/`; verify the slice + all goldens total < ~500 KB (R1/R5 —
      trim the window if larger). **Stage** (ship in Task 6.4).

---

## Phase 1 — the parse+shape port

### Task 1.1: `[parse-dfl]` extra + the `providers` namespace skeleton

**Files:** Modify `pyproject.toml`; Create `silly_kicks/providers/__init__.py`,
`silly_kicks/providers/sportec/__init__.py`.

- [ ] **Step 1: Add the extra** in `pyproject.toml` under `[project.optional-dependencies]`:

```toml
parse-dfl = []  # the DFL parse port is stdlib-xml.etree + pandas (already core deps); the extra is the
                # opt-in handle + the lakehouse's hard-dep marker on adoption. Add deps here only if the
                # port later needs one.
```

- [ ] **Step 2: Create the namespace** `silly_kicks/providers/__init__.py`:

```python
"""Per-provider raw-data parse ports (bytes -> provider-canonical bronze rows).

A parse port is the faithful `bytes -> bronze` boundary for a provider's native files; a separate
`shape_*` composable maps bronze -> a silly-kicks converter's input. See ADR-031 (T3) +
docs/superpowers/specs/2026-06-16-dfl-parse-port-design.md. Behind the `[parse-dfl]` extra.
"""
```

- [ ] **Step 3: Create** `silly_kicks/providers/sportec/__init__.py` re-exporting the public surface:
      `parse_dfl_match_info`, `parse_dfl_events`, `parse_dfl_tracking`, `shape_tracking_to_native`,
      `shape_events_to_native`, `MatchInfo`, `SportecTrackingBronze`, `SportecEventBronze`. In `parse.py`,
      declare each `parse_dfl_*`/`shape_*` as a signature-complete `raise NotImplementedError("Task 1.3/1.4")`
      **stub** for now (bodies filled in 1.3/1.4) — this is what makes Task 2.1's RED natural (F4).
- [ ] **Step 4:** `python -c "import silly_kicks.providers.sportec"` → no error. Stage.

### Task 1.2: Typed return types

**Files:** `silly_kicks/providers/sportec/parse.py` (start the module).

- [ ] **Step 1: Define the typed returns** (Hyrum-pinned shapes; N1 — silly-kicks' own domain names).
      `MatchInfo` is a frozen dataclass; the bronze frames are typed as DataFrames with a pinned column
      tuple constant (mirrors how `tracking/schema.py` pins `EXPECTED_INPUT_COLUMNS`):

```python
from dataclasses import dataclass

# Provenance: upstreamed from luxury-lakehouse src/ingestion/idsse.py @ 0efac60 (owner-owned; MIT).
# SportecTrackingBronze is silly-kicks' OWN domain name, field-identical today to the lakehouse
# _IDSSE_TRACKING_BRONZE_COLS / bronze.idsse_tracking (a versioned cross-repo contract -- ADR-031 N1).
SPORTEC_TRACKING_BRONZE_COLS: tuple[str, ...] = (  # copy from idsse.py:846 verbatim
    "match_id", "period", "frame", "timestamp", "player_id", "team", "x", "y",
    "ball_x", "ball_y", "ball_z", "ball_s", "ball_status", "is_goalkeeper", "frame_rate", ...
)
SPORTEC_EVENT_BRONZE_COLS: tuple[str, ...] = (...)  # copy from idsse.py:354 (_IDSSE_EVENTS_BRONZE_COLS)

@dataclass(frozen=True)
class MatchInfo:
    home_team_id: str
    away_team_id: str
    player_team_map: dict[str, str]
    gk_player_ids: frozenset[str]
    home_team_start_left: bool
    home_team_start_left_extratime: bool | None

# SportecTrackingBronze / SportecEventBronze are pd.DataFrames whose columns == the *_BRONZE_COLS
# tuples; a thin validator asserts the column set on return (TypedDict-style runtime contract).
```

- [ ] **Step 2:** Copy `SPORTEC_TRACKING_BRONZE_COLS`/`SPORTEC_EVENT_BRONZE_COLS` **exactly** from the
      pinned clone's `_IDSSE_TRACKING_BRONZE_COLS:846` / `_IDSSE_EVENTS_BRONZE_COLS:354` (the latter is
      computed at import — materialise its concrete tuple). `pyright silly_kicks/providers/` clean. Stage.

### Task 1.3: Lift the parse functions (bytes → bronze)

**Files:** `silly_kicks/providers/sportec/parse.py`.

- [ ] **Step 1: Copy verbatim** from `~/Development/luxury-lakehouse/src/ingestion/idsse.py` @ `0efac60`
      (the cloned pin): `_parse_float_or_none`, `_parse_bool_or_none`, `_SECTION_TO_PERIOD`, `_parse_teams`
      (:437), `_parse_match_metadata` (:517), `_parse_positions_xml` (:620, two-pass), `_build_event_row`
      (:1147), `_parse_events_xml` (:1356), `derive_idsse_home_team_start_left` (`spadl_adapter.py:438`).
      Adapt ONLY: drop the lakehouse `logger` arg (use the module logger or a no-op); keep the bodies
      otherwise byte-for-byte (the parity test enforces this). NO lakehouse-internal imports.
- [ ] **Step 2: Wrap as the public parse API** returning the typed bronze:

```python
def parse_dfl_match_info(info_xml) -> MatchInfo: ...        # wraps _parse_teams + _parse_match_metadata + derive_*
def parse_dfl_tracking(positions_xml, *, player_team_map, match_id) -> SportecTrackingBronze:
    rows_by_period = _parse_positions_xml(positions_xml, player_team_map, match_id)
    df = _bronze_tracking_df(rows_by_period)               # -> SPORTEC_TRACKING_BRONZE_COLS, column-validated
    return df
def parse_dfl_events(events_xml, *, player_team_map, match_id, metadata=None) -> SportecEventBronze: ...
```

- [ ] **Step 2b: Confirm purity (D/Chesterton):** grep the copied bodies for `self.`/`timed_check`/
      `FilterResult`/`workflow`/`spark` → none (verified on `0efac60`; re-confirm post-copy).
- [ ] **Step 3:** `pyright silly_kicks/providers/`; `ruff check`/`format`; import smoke. Stage.

### Task 1.4: Lift the shapers (bronze → native converter input)

**Files:** `silly_kicks/providers/sportec/parse.py`.

- [ ] **Step 1: Copy the shaper** `_bronze_idsse_to_sportec_input` (from `action_context/convert.py` /
      `tracking_context.py` @ `0efac60`) as `shape_tracking_to_native(bronze: SportecTrackingBronze) ->
      NativeTrackingInput` — emits `tracking.sportec.EXPECTED_INPUT_COLUMNS` (`x→x_centered`, `y→y_centered`,
      `ball_s→speed_native`, `ball_z→z`, `ball_status→ball_state`, explode ball rows, team-label→`team_id`,
      …). Lift the **IDSSE function only** (the lakehouse file is shared with other-provider builders).
      Validate the output column set == `tracking.sportec.EXPECTED_INPUT_COLUMNS`.
- [ ] **Step 2: Lift the events shaper** `shape_events_to_native(bronze: list[SportecEventBronze]) ->
      NativeEventsInput` from the lakehouse `spadl_adapter.py` IDSSE-events adapter → emits the
      `spadl.sportec.convert_to_actions` input contract.
- [ ] **Step 3:** `pyright`/`ruff`; import smoke. Stage.

---

## Phase 2 — RED-first parity vs the committed golden (review N2/E/F)

> **Sequencing (F4):** author Task 2.1 **before** filling the parse/shape bodies (Tasks 1.3/1.4). After
> Task 1.1 (the `providers.sportec` skeleton with the public names declared but unimplemented) +
> Task 1.2 (typed returns), the public functions exist as raise-`NotImplementedError` stubs — so running
> Task 2.1 here yields a **natural RED** (no temporary `return empty` hack needed), and Tasks 1.3/1.4
> then drive it green. Execute order: 1.1 → 1.2 → **2.1 (author + observe RED)** → 1.3 → 1.4 → 2.1 (green).

### Task 2.1: Parse(+shape)-port parity test

**Files:** Create `tests/providers/sportec/test_parse_port_parity.py`.

- [ ] **Step 1: Write the test** — runs the port on the committed reduced DFL slice and asserts it
      reproduces the committed `0efac60` goldens (bronze AND native; tracking AND events) by semantic
      equality:

```python
import pandas as pd, pytest
from pathlib import Path
from silly_kicks.providers.sportec import (
    parse_dfl_match_info, parse_dfl_tracking, parse_dfl_events,
    shape_tracking_to_native, shape_events_to_native,
)
_FIX = Path(__file__).resolve().parents[2] / "datasets" / "sportec" / "idsse_slice"

def _assert_semantic_equal(got, golden, *, exact_cols):
    # canonical ordering + float tolerance; byte-exact only for ids / ball_state enums (S2)
    g = got.sort_values(sorted(got.columns)).reset_index(drop=True)
    e = pd.read_parquet(golden).sort_values(... ).reset_index(drop=True)
    pd.testing.assert_frame_equal(g[exact_cols], e[exact_cols], check_dtype=True)        # exact
    pd.testing.assert_frame_equal(g.drop(columns=exact_cols), e.drop(columns=exact_cols),
                                  check_exact=False, atol=1e-6)                            # tolerant

def test_tracking_bronze_matches_lakehouse_golden():
    mi = parse_dfl_match_info(_FIX / "info.xml")
    bronze = parse_dfl_tracking(_FIX / "positions.xml", player_team_map=mi.player_team_map, match_id="slice")
    _assert_semantic_equal(bronze, _FIX / "idsse_parse_bronze_golden.parquet", exact_cols=["player_id","ball_status"])

def test_tracking_native_matches_lakehouse_golden():
    ... shape_tracking_to_native(bronze) vs idsse_shape_native_golden.parquet ...

def test_events_bronze_and_native_match_golden(): ...   # the events equivalents
```

- [ ] **Step 2: Prove RED FIRST (E + F4).** Authored before Tasks 1.3/1.4, so the public functions are
      still the Task-1.1 `raise NotImplementedError` stubs — run:
      `python -m pytest tests/providers/sportec/test_parse_port_parity.py -v` → **FAIL** naturally (no
      `return empty` hack). Capture the failing output as the sensitivity proof (PR description / plan
      note) — **not a commit**.
- [ ] **Step 3:** With the real port (Tasks 1.3/1.4), run → **PASS**. Stage.

---

## Phase 3 — re-route the dev harness + retire the kloppy IDSSE path

### Task 3.1: Re-route `_loader_pining.py` IDSSE to the port + native converters

**Files:** Modify `scripts/_loader_pining.py` (the `_build_idsse` branch `:300-315`).

- [ ] **Step 0 (G4 — verify the native converter accepts `"absolute_frame"`, don't assume from the
      kloppy gateway):** CONFIRMED against source — `silly_kicks/tracking/sportec.py:60-69`
      `convert_to_frames(..., *, output_convention: Literal["absolute_frame", "ltr"] | None = None)`, and
      `"absolute_frame"` is the documented historical default (the docstring example at ~:115 passes it
      explicitly). This is the **native** sportec converter (distinct from the kloppy gateway the SC/Metrica
      golden entries use, which the lakehouse calls with `"ltr"`). Both Task 3.1 and Task 4.1 use
      `"absolute_frame"` → valid. (Re-grep at execution time to guard against drift since 4.29.0.)
- [ ] **Step 1: Replace** `_build_idsse` to parse via the port → native converters (no kloppy parser):

```python
def _build_idsse(paths, tracking_limit):
    from silly_kicks.providers.sportec import (
        parse_dfl_match_info, parse_dfl_events, parse_dfl_tracking,
        shape_tracking_to_native, shape_events_to_native,
    )
    from silly_kicks.spadl import sportec as sportec_spadl
    from silly_kicks.tracking import sportec as sportec_tracking

    mi = parse_dfl_match_info(paths["metadata"])
    bronze_trk = parse_dfl_tracking(paths["tracking"], player_team_map=mi.player_team_map, match_id=...)
    native_trk = shape_tracking_to_native(bronze_trk)
    frames, _ = sportec_tracking.convert_to_frames(
        native_trk, home_team_id=mi.home_team_id, home_team_start_left=mi.home_team_start_left,
        home_team_start_left_extratime=mi.home_team_start_left_extratime, output_convention="absolute_frame",
    )
    bronze_evt = parse_dfl_events(paths["events"], player_team_map=mi.player_team_map, match_id=...)
    native_evt = shape_events_to_native(bronze_evt)
    actions, _ = sportec_spadl.convert_to_actions(
        native_evt, home_team_id=mi.home_team_id, home_team_start_left=mi.home_team_start_left)
    return actions, _preprocess(frames), mi.home_team_id   # _preprocess = silly-kicks smooth+velocity (consumer-side)
```

- [ ] **Step 2: Retire** `_kloppy_tracking_to_frames` (`:318-359`) + the kloppy IDSSE events path + the
      `game_id`-None stamp (`:313-314`) — `git grep _kloppy_tracking_to_frames` confirms the single
      caller is `_build_idsse` (PR-S94 verified). Remove the now-unused kloppy `sportec.load_*` imports
      in that branch.
- [ ] **Step 3: Run** `python -m pytest tests/calibration/test_loader_pining.py -v` (mock-driven) → adjust
      mocks if they assumed the kloppy path; the loader must still yield `(actions, frames, home)`.

### Task 3.2: Update the `game_id`-None expectation (R4)

**Files:** Modify `tests/calibration/test_calibrate_cli.py:36`.

- [ ] **Step 1: grep ALL consumers** of the `game_id`-None behaviour
      (`git grep -n "game_id.*None" tests/ scripts/`). The native converter sets `game_id` from
      `match_id`, so the IDSSE actions now carry a real `game_id`.
- [ ] **Step 2: Update** the `test_calibrate_cli.py:36` parametrization (the `(None, "DFL-MAT-1")` IDSSE
      case → the native `game_id`). Run → PASS. Stage.

---

## Phase 4 — golden coverage + invariance e2e

### Task 4.1: Add native-IDSSE to the cross-provider y-identity golden

**Files:** Modify `tests/tracking/test_kloppy_y_identity_golden.py`; reuse the Phase-0 reduced DFL slice.

- [ ] **Step 1:** Add an `idsse` entry to the parametrized loader — load via the port → native
      `convert_to_frames(..., output_convention="absolute_frame")` (F3/G4: confirmed accepted by the
      **native** sportec converter — Task 3.1 Step 0, `tracking/sportec.py:67`; matches the harness's
      convention in Task 3.1. The existing SC/Metrica entries also use `absolute_frame` but go through the
      kloppy gateway — a different converter — so the native acceptance is verified independently. The
      `"ltr"` convention is the lakehouse pipeline's separate downstream choice, NOT this y-identity
      guard's), assert acting-player frame-y ≈ action `start_y` off-centre.
      **Green-from-start** (Gate D: native sportec is y-correct) — NOT RED-first (it's a regression
      guard, nothing to drive out). Run → PASS. Stage.

### Task 4.2: Calibration-invariance e2e (review minor + N6 tolerance)

**Files:** Create `tests/calibration/test_calibration_invariance_e2e.py`.

- [ ] **Step 1 (G1 — load the frozen golden, do NOT reconstruct deleted code):** `_kloppy_tracking_to_frames`
      is deleted by Task 3.1, so the OLD-way frames come from the **Task 0.3 frozen artifact**
      `idsse_oldpath_harness_golden.parquet` — NOT reconstructed in-test. Build the NEW-way frames on the
      committed IDSSE slice (port → native `convert_to_frames`), then assert vs the frozen old-golden:
      y-anchored feature columns **move** (the `|68−2y|` un-inversion — the correctness fix), while
      x-only / y-symmetric / frame-integrated columns are **~unchanged** (tolerance for parse-engine
      numeric noise, NOT for a flip). Documents N6. Run → PASS. Stage.

---

## Phase 5 — full-suite + lint/type gate

### Task 5.1: Full local verification (the PR-S94 CI-miss lesson)

- [ ] **Step 1:** `ruff format --check . && ruff check .` → clean.
- [ ] **Step 2:** `pyright silly_kicks/` (full package) → 0 errors.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e and not slow" --benchmark-skip -q` (the **whole**
      `tests/` dir, run in the background per the >30s rule) → all pass. (Don't scope to a subset.)

---

## Phase 6 — ADR + release + handoff + commit

### Task 6.0: Issue the lakehouse-adoption handoff (B)

- [ ] **Step 1:** Relay (copy/paste) the lakehouse-adoption handoff: the port ships at silly-kicks
      4.30.0; the lakehouse picks **B-i** (delete-and-depend: import `silly_kicks.providers.sportec`,
      delete its IDSSE parse+shape functions, pin `silly-kicks[parse-dfl]>=4.30.0`, S3 checklist incl.
      dropping `test_idsse_converter_no_drift`) or **B-ii** (keep-both + parity, bump-time sync). Sequence
      after the lakehouse parser churn settles. NOT a silly-kicks TODO item.
- [ ] **Step 2 — forward note (review):** make the handoff explicit that the cross-repo **bronze
      contract** (`SPORTEC_*_BRONZE_COLS` ≡ lakehouse `bronze.idsse_*`) is only kept LIVE by a
      **lakehouse-SIDE parity test**: under B-i, a test asserting the imported port reproduces a
      lakehouse-captured golden on the lakehouse's own corpus; under B-ii, the bump-time keep-both parity.
      silly-kicks' committed golden pins the port to `0efac60` from THIS side only — without the
      lakehouse-side guard, "single-sourced" silently decays the next time the lakehouse re-pins past
      `0efac60`. State this as the adoption contract, not an optional nicety.

### Task 6.1: Amend ADR-031

**Files:** Modify `docs/superpowers/adrs/ADR-031-kloppy-tracking-y-inversion.md`.

- [ ] Record T3 shipped: the parse+shape port (bronze-split), the committed-golden parity pinned to
      `0efac60`, the N1 bronze cross-repo-contract, the N6 correctness-remediation retrain, and the
      lakehouse's B-i/B-ii choice. Status → "T1 + T3 shipped; T2 = no-op".

### Task 6.2: Version bump 4.30.0 (5-file gate)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock`.

- [ ] Bump `4.29.0 → 4.30.0` in pyproject + `__init__`; CHANGELOG entry (the port + the **correctness**
      retrain for IDSSE calibration consumers, N6); **TODO** — remove the now-shipped "PR-S95 / T3"
      bullet from "### Confirmed bugs" (completed items are deleted, not annotated), leaving only the
      IDSSE-ET-unverified follow-up; `uv lock`. Verify all five agree.

### Task 6.3: `/final-review`

- [ ] Run `/final-review` (code + docs + C4 drift). The new `providers/` container is a C4 change —
      regenerate `docs/c4/architecture.html` (add the `providers.sportec` parse port; note it feeds
      both `spadl`/`tracking` converters). Address findings.

### Task 6.4: Single commit (ONLY after explicit owner approval)

- [ ] `git add -A`; one commit (subject:
      `feat(providers)!: DFL parse+shape port single-sources the IDSSE/Sportec parser -- silly-kicks 4.30.0 (ADR-031, PR-S95)`)
      ending with the `Co-Authored-By` trailer. Do NOT tag. Wait for CI green (owner monitors), then
      tag `v4.30.0`.

---

## Self-review

- **Spec coverage:** §2.2 port (Tasks 1.2–1.4) · §2.3 lift/SHA-pin (1.3/1.4, 0.1 Step 4) · §2.4 data-quality
  consumer-side (3.1 `_preprocess`) · §3 re-route/retire (3.1/3.2) · §4.1 parity committed-golden RED-first
  (0.2 + 2.1) · §4.4 e2e (4.2) · §4.5 schema-valid fixture (0.1) · §5 adoption B-i/B-ii (6.0) · N1 naming/
  contract (1.2) · N4 events bronze (0.2/1.2) · N5 test_convert_drift (6.0) · N6 correctness retrain (4.2/6.2).
- **Placeholder scan:** lifted bodies are "copy verbatim from `0efac60` <file:line>" (precise, not vague —
  the source is the authority; reproducing 100s of lines inline would be noise). All assertions/diffs/
  signatures are concrete.
- **Type consistency:** `SportecTrackingBronze`/`SportecEventBronze`/`MatchInfo`/`NativeTrackingInput` used
  consistently across Tasks 1.2→1.4→2.1→3.1.
- **RED-first under single-commit:** Task 2.1 parity is authored before the parse bodies (F4) so RED is
  natural off the Task-1.1 `NotImplementedError` stubs, then driven green (evidence, not a commit);
  Task 4.1 native-IDSSE golden + Task 4.2 e2e are green-from-start guards.
- **Plan-review fixes folded (rev 2 — F1–F4):** F1 — Phase 0 captures goldens by running the REAL lakehouse
  functions in the lakehouse clone's OWN venv, not a hand-copy. F2 — the OLD-path harness golden is frozen
  BEFORE Task 3.1 retires `_kloppy_tracking_to_frames`. F3 — native convert pinned to `"absolute_frame"`.
  F4 — Task 2.1 authored before 1.3/1.4. Forward note — Task 6.0 Step 2 makes the lakehouse-side parity
  test the live cross-repo contract.
- **Plan-review fixes folded (rev 3 — G1–G4):** **G1** — Task 4.2 Step 1 now LOADS the Task-0.3 frozen
  `idsse_oldpath_harness_golden.parquet` (no in-test reconstruction of the deleted kloppy path; removes the
  rev-2 self-contradiction). **G2** — Phase 0 split into THREE scripts across the two mutually-exclusive
  envs: `_reduce_idsse_dfl_slice.py` (pure stdlib, Task 0.1) → `_capture_lakehouse_golden.py` (lakehouse
  clone venv, Task 0.2) + `_capture_oldpath_golden.py` (`~/sk-s93-venv`, Task 0.3), sharing the reduced XML
  on disk. **G3** — Task 0.1 Step 2 asserts the reduced slice is accepted by BOTH parsers (kloppy Gate A —
  the stricter — and the lakehouse ET parser Gate B), flagged as a hard reduction constraint (kloppy-DFL
  schema rabbit-hole risk). **G4** — Task 3.1 Step 0 records the source-confirmed native acceptance of
  `"absolute_frame"` (`tracking/sportec.py:67`), independent of the kloppy-gateway SC/Metrica entries.
