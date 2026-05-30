# ET-direction converter consistency (silly-kicks 4.0.0) — Implementation Plan

> **v4 (2026-05-30): APPROVED — "ready to execute"** (lakehouse plan-review round 2). Folds in round-2 polish a–g (GS wrapper sketch, period-3-only verify, synthetic-vs-real assertion-strength note, pitch-dims, Task 4→9 dependency, home_team_id cast, Task 0 wording). v3 incorporated round 1 (A–I) **and the lakehouse data delivery + §8 audit**. The GS real-data ET fixture is delivered (`tests/regressions/extratime/gs_et/`, meta carries `home_team_start_left_extratime=True`). The **§8 historical audit returned a clean zero** (IDSSE 0 / Metrica 0 ET matches ever processed; GS 5 but GS already raises + carries the flag) → **no remediation owed, the §9-item-6 ship-gate is SATISFIED**, 4.0.0 may publish to PyPI without waiting. IDSSE/Metrica ET fixtures are **not deliverable** (zero such data in bronze) → synthesize them in silly-kicks (Task 8). Spec v3 + ADR-010 are the design records.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.
>
> **Commit policy override:** this project is **ONE commit per branch** (user rule, overrides the skill's "frequent commits"). Do **NOT** `git commit` per task. Work accumulates uncommitted on the branch `pr-s70-et-direction-consistency`; the single commit happens after `/final-review` + explicit user approval (Task 12). Each task ends by **running the tests green**, not committing.

**Goal:** Make all per-period-absolute converters (Sportec + Metrica + Gradient Sports, tracking **and** events) **raise** on extra-time-without-`home_team_start_left_extratime`, via one public shared guard — never silently ship wrong ET geometry.

**Architecture:** Full rename `tracking/_direction.py → tracking/direction.py`; add public `require_et_direction` guard there; call it from all 5 per-period-absolute converters (GS refactors its 2 inline raises to it; Sportec ×2 + Metrica ×1 gain it). Public calibration-labelled `filter_extratime_frames` helper DRYs the loaders. **SemVer 4.0.0** (breaking behaviour change). Companion **ADR-010** + spec (both already on-branch).

**Tech stack:** pandas, pytest, ruff, pyright, uv. Reference: `docs/superpowers/specs/2026-05-30-et-direction-converter-consistency-design.md`, `docs/superpowers/adrs/ADR-010-extratime-direction-fail-loud.md`.

**Ship gate (spec §9 item 6): ✅ SATISFIED.** The lakehouse §8 historical-data audit completed (2026-05-30) and returned a **clean zero** — no ET matches were ever processed through the silent-default Sportec/Metrica path (IDSSE 0, Metrica 0; GS already raised + carries the flag), so **no historical data is mis-oriented and no remediation is owed**. The audit *is* the report-back. 4.0.0 may tag + publish to PyPI normally. (Audit memo: lakehouse `memory/project_et_direction_section_8_audit.md`.)

---

## File structure (what changes)

| File | Responsibility | Action |
|---|---|---|
| `silly_kicks/tracking/direction.py` | per-period direction + public `require_et_direction` | **rename** from `_direction.py` |
| `silly_kicks/tracking/__init__.py` | re-export `require_et_direction` | modify |
| `silly_kicks/spadl/__init__.py` | re-export `require_et_direction` (events) | modify |
| `silly_kicks/tracking/utils.py` | public `filter_extratime_frames` (calibration-labelled) | modify |
| `silly_kicks/tracking/sportec.py` | tracking guard | modify |
| `silly_kicks/tracking/gradientsports.py` | refactor inline raise → shared guard | modify |
| `silly_kicks/spadl/sportec.py` | events guard (NEW raise) | modify |
| `silly_kicks/spadl/metrica.py` | events guard (NEW raise) | modify |
| `silly_kicks/spadl/gradientsports.py` | refactor inline raise → shared guard | modify |
| `scripts/_loader_pining.py` | collapse `_apply_et_direction` → `filter_extratime_frames` | modify |
| `scripts/_loader_databricks.py` | use `filter_extratime_frames` on sportec frames | modify |
| `tests/tracking/test_direction.py` | guard unit + parity + structural | create |
| `tests/regressions/extratime/` | frozen RT-only goldens + real ET fixtures | create |
| pyproject / `__init__` / TODO / CHANGELOG / uv.lock | 4.0.0 bump | modify |

---

## Task 0: Capture RT-only goldens against CURRENT (3.30) behaviour — ONE-WAY DOOR, DO FIRST

**Files:**
- Create: `tests/regressions/extratime/capture_goldens.py` (capture script — committed in the single branch commit, kept for reproducibility / future golden regen)
- Create: `tests/regressions/extratime/golden_{sportec_tracking,gs_tracking,sportec_events,gs_events,metrica_events}_rt.parquet`

RT-only output is unaffected by the ET change (the guard never fires without ET periods), so these goldens pin "no RT regression". They MUST be captured **before** any converter edit so the baseline is provably the pre-change behaviour.

> **Baseline premise (review I):** this is a **baseline-pinning** exercise, not a correctness-verification one — the goldens encode whatever silly-kicks 3.30 currently does on the RT path, **bugs included**. The spec's premise ("RT-only output is unaffected by the ET change") makes that the correct reference *for this change*. If a 3.30 RT-path bug is later discovered, the affected golden must be regenerated separately.

- [ ] **Step 1: Confirm on a clean tree at the pre-change converters** (the loader fix from this branch does not touch converters, so it is fine to be present).

Run: `git -C . status --short` — expect only the loader/spec/ADR files modified, no converter files.

- [ ] **Step 2: Write the capture script** — build one minimal **regular-time-only** fixture per converter family (2 periods, 2 teams, ball; for events: SPADL-input shape per each converter's `EXPECTED_INPUT_COLUMNS`), call each converter with `home_team_start_left=True` (NO ET), and write the output to the golden parquet. Use deterministic synthetic data (fixed values, no RNG).

```python
# tests/regressions/extratime/capture_goldens.py
"""Capture RT-only converter goldens against the CURRENT (pre-4.0.0) behaviour.
RT-only output is unaffected by the ET guard; these pin no-regression. Re-runnable."""
from pathlib import Path
import pandas as pd
OUT = Path(__file__).parent
# ... build per-converter RT-only inputs (see each converter's EXPECTED_INPUT_COLUMNS) ...
# frames_sportec = sportec_tracking.convert_to_frames(rt_input, home_team_id="H", home_team_start_left=True)
# frames_sportec.to_parquet(OUT / "golden_sportec_tracking_rt.parquet")
# ... repeat for gs tracking, sportec/gs/metrica events ...
```

> Implementation note: reuse the input-builder helpers from the existing converter tests (`tests/tracking/test_sportec_tracking*.py`, `tests/spadl/test_*`) so the fixtures match real `EXPECTED_INPUT_COLUMNS`. Each builder is ~15 lines of literal DataFrame construction — inline them, no RNG.

- [ ] **Step 3: Run the capture, commit the parquet outputs to the working tree (uncommitted; part of the single branch commit).**

Run: `python tests/regressions/extratime/capture_goldens.py`
Expected: 5 `golden_*_rt.parquet` files written under `tests/regressions/extratime/`.

- [ ] **Step 4: Verify the goldens are non-empty + load.**

Run: `python -c "import pandas as pd, glob; [print(f, pd.read_parquet(f).shape) for f in glob.glob('tests/regressions/extratime/golden_*_rt.parquet')]"`
Expected: 5 files, each non-empty.

---

## Task 1: Rename `_direction.py → direction.py` + structural safety tests (T-A / T-B)

**Files:**
- Rename: `silly_kicks/tracking/_direction.py` → `silly_kicks/tracking/direction.py`
- Modify: every importer of `_direction`
- Test: `tests/tracking/test_direction.py` (create)

- [ ] **Step 1: Enumerate ALL importers (none may be missed).**

Run: `grep -rn "_direction" silly_kicks/ tests/ scripts/`
Record every hit. Known: `tracking/sportec.py`, `tracking/gradientsports.py` (if it imports it), `spadl/sportec.py`, `spadl/gradientsports.py`, `spadl/metrica.py`, plus any `tracking/_direction` test. Note: match the **module** token `_direction` (not the substring `direction` which would catch `home_attacks_right_per_period`-unrelated hits) — verify each hit is an import of the module.

- [ ] **Step 2: Write T-A (symbol-preservation safety net) BEFORE the rename.**

```python
# tests/tracking/test_direction.py
def test_direction_public_symbols_present():
    from silly_kicks.tracking import direction
    assert hasattr(direction, "home_attacks_right_per_period")
    assert hasattr(direction, "require_et_direction")  # added in Task 2

def test_require_et_direction_reexported():
    import silly_kicks.tracking as t
    import silly_kicks.spadl as s
    assert hasattr(t, "require_et_direction")
    assert hasattr(s, "require_et_direction")
```

- [ ] **Step 3: Run T-A to verify it FAILS** (module not yet renamed / symbol absent).

Run: `pytest tests/tracking/test_direction.py -q`
Expected: FAIL (`No module named 'silly_kicks.tracking.direction'` or missing attr).

- [ ] **Step 4: `git mv` the module + update every importer + private→public name.**

```bash
git mv silly_kicks/tracking/_direction.py silly_kicks/tracking/direction.py
```
Then in each importer from Step 1, replace `from . import _direction` / `from .. tracking import _direction` / `_direction.` with the `direction` equivalents. (`home_attacks_right_per_period` keeps its name; only the module name changes.)

- [ ] **Step 5: Add re-exports.** In `silly_kicks/tracking/__init__.py` and `silly_kicks/spadl/__init__.py`, add `require_et_direction` to the imports + `__all__` (the function lands in Task 2; add the import now and it will resolve once Task 2 defines it — or sequence Step 5 after Task 2 Step 3).

- [ ] **Step 6: Run the full tracking + spadl suites to confirm the rename broke nothing.**

Run: `pytest tests/tracking/ tests/spadl/ -m "not e2e" -q`
Expected: PASS (same count as before the rename; T-A's `require_et_direction` assertions still fail until Task 2 — run `-k "not require_et_direction"` here, or fold Step 5/Task 2 first).

---

## Task 2: Public `require_et_direction` guard

**Files:**
- Modify: `silly_kicks/tracking/direction.py`
- Test: `tests/tracking/test_direction.py`

- [ ] **Step 1: Write the failing tests.**

```python
import pandas as pd, pytest
from silly_kicks.tracking.direction import require_et_direction

def test_require_et_raises_when_et_present_and_flag_none():
    with pytest.raises(ValueError, match="ET periods"):
        require_et_direction(pd.Series([1, 1, 3, 4]), None, source="sportec convert_to_frames")

def test_require_et_noop_when_flag_provided():
    require_et_direction(pd.Series([1, 3]), True, source="x")  # no raise

def test_require_et_noop_when_no_et_periods():
    require_et_direction(pd.Series([1, 1, 2, 2]), None, source="x")  # no raise

def test_require_et_message_names_source():
    with pytest.raises(ValueError, match="metrica convert_to_actions"):
        require_et_direction(pd.Series([3]), None, source="metrica convert_to_actions")
```

- [ ] **Step 2: Run — verify FAIL** (`ImportError: require_et_direction`).

Run: `pytest tests/tracking/test_direction.py -k require_et -q` → FAIL.

- [ ] **Step 3: Implement the guard in `direction.py`.**

```python
from collections.abc import Sequence
import numpy as np  # (module already imports pd; add np if not present)

def require_et_direction(
    period_ids: "pd.Series | np.ndarray | Sequence[int]",
    home_team_start_left_extratime: bool | None,
    *,
    source: str,
) -> None:
    """Raise ValueError if extra-time periods (3/4) are present but the ET start
    direction is unset. Per-period-absolute converters (Sportec, Metrica, Gradient
    Sports) need home_team_start_left_extratime to orient ET; guessing it silently
    flips ET coordinates. See ADR-010.
    """
    if home_team_start_left_extratime is None and pd.Series(period_ids).isin([3, 4]).any():
        raise ValueError(
            f"{source}: data contains ET periods (period_id in {{3, 4}}) but "
            "home_team_start_left_extratime was not provided. Set it from the match "
            "metadata (e.g. homeTeamStartLeftExtraTime), or filter ET out before converting."
        )
```
(`import pandas as pd` already present in the module.)

- [ ] **Step 4: Run — verify PASS** (guard tests + T-A re-export tests).

Run: `pytest tests/tracking/test_direction.py -q` → PASS.

---

## Task 3: Refactor the Gradient Sports inline raises to the shared guard (no behaviour change)

**Files:**
- Modify: `silly_kicks/tracking/gradientsports.py` (the `isin([3, 4])` raise, ~line 114)
- Modify: `silly_kicks/spadl/gradientsports.py` (the `isin([3, 4])` raise, ~line 326)

- [ ] **Step 1: Confirm the existing GS ET raise tests pass (baseline).**

Run: `pytest tests/ -k "gradientsports and (et or extratime or extra_time)" -q` (and the converter tests). Expected: PASS.

- [ ] **Step 2: Replace each inline raise with the guard.** In both files, replace the `if ...isin([3, 4]).any() and home_team_start_left_extratime is None: raise ValueError(...)` block with:

```python
from .direction import require_et_direction  # tracking; for spadl use: from ..tracking.direction import require_et_direction
require_et_direction(frames["period_id"], home_team_start_left_extratime, source="gradientsports convert_to_frames")
# events: require_et_direction(events["period_id"], home_team_start_left_extratime, source="gradientsports convert_to_actions")
```

- [ ] **Step 3: Run the GS tests — verify still PASS** (same raise, now via the shared guard).

> **Contract clarification (review C):** GS has raised on ET-without-flag for some time; this refactor **standardizes the message wording** to the shared-guard format. The exception **type stays `ValueError`** and the trigger condition is unchanged — only the message text changes. Internal silly-kicks tests are updated; any external consumer parsing the GS message *text* (unlikely, but possible) must update. **Document in the CHANGELOG [4.0.0] migration section** (Task 11).

Run: `pytest tests/ -k gradientsports -m "not e2e" -q` → PASS.

---

## Task 4: Sportec converters — add the guard (NEW raise; was silent default)

**Files:**
- Create: `tests/tracking/conftest.py` (shared RT+ET input builders — review F)
- Modify: `silly_kicks/tracking/sportec.py` (before the `home_attacks_right_per_period` call, ~line 132)
- Modify: `silly_kicks/spadl/sportec.py` (before its `home_attacks_right_per_period` call)
- Test: `tests/tracking/test_direction.py` (or the existing sportec test files)

- [ ] **Step 0: Create the shared fixture builders (review F).** Tasks 4, 5, and 6 all need the same RT+ET input shapes per converter family. Put them in one place so they don't drift:

```python
# tests/tracking/conftest.py
import pytest, pandas as pd
# Minimal, deterministic (no RNG) input builders, parametrizable by ET presence.
# One per converter family, matching that converter's EXPECTED_INPUT_COLUMNS.
def sportec_tracking_input(*, et: bool) -> pd.DataFrame: ...   # 2 RT periods + (period 3 if et)
def gs_tracking_input(*, et: bool) -> pd.DataFrame: ...
def sportec_events_input(*, et: bool) -> pd.DataFrame: ...      # SPADL-input shape
def gs_events_input(*, et: bool) -> pd.DataFrame: ...
def metrica_events_input(*, et: bool) -> pd.DataFrame: ...
# Export them (plain functions or fixtures) for tests/tracking + tests/spadl reuse.
```
> Reuse existing per-converter test builders where they exist (`tests/tracking/test_sportec_tracking*.py`, `tests/spadl/test_*`) rather than inventing new shapes — copy their literal DataFrame construction here and add the `et` branch (append period-3 rows).

- [ ] **Step 1: Write the failing tests (NEW behaviour — Sportec must now raise).**

```python
def test_sportec_tracking_raises_on_et_without_flag():
    import pandas as pd
    from silly_kicks.tracking import sportec as sp
    rt_et = _sportec_rt_input_with_et()  # build minimal sportec frames incl. period 3
    with pytest.raises(ValueError, match="ET periods"):
        sp.convert_to_frames(rt_et, home_team_id="H", home_team_start_left=True)  # no ET flag

def test_sportec_tracking_ok_with_et_flag():
    from silly_kicks.tracking import sportec as sp
    rt_et = _sportec_rt_input_with_et()
    out = sp.convert_to_frames(rt_et, home_team_id="H", home_team_start_left=True,
                               home_team_start_left_extratime=False)
    assert out["period_id"].isin([3]).any()  # ET kept, no raise
```
(Mirror for `spadl/sportec.py` events with a SPADL-input fixture incl. period 3.)

- [ ] **Step 2: Run — verify FAIL** (currently silently defaults, no raise).

Run: `pytest tests/tracking/test_direction.py -k sportec -q` → FAIL (no exception raised).

- [ ] **Step 3: Insert the guard** before the `home_attacks_right_per_period(...)` call in each file:

```python
require_et_direction(out["period_id"], home_team_start_left_extratime, source="sportec convert_to_frames")
# events: require_et_direction(actions["period_id"], home_team_start_left_extratime, source="sportec convert_to_actions")
```
(Import `require_et_direction` from `.direction` / `..tracking.direction`. Place it AFTER the existing `home_attacks_right_per_period`-vs-`home_team_start_left` mutual-exclusion check, BEFORE the flip.)

- [ ] **Step 4: Run — verify PASS.**

Run: `pytest tests/tracking/test_direction.py -k sportec -q` → PASS.

---

## Task 5: Metrica events converter — add the guard (NEW raise)

**Files:**
- Modify: `silly_kicks/spadl/metrica.py` (before its `home_attacks_right_per_period` call)
- Test: `tests/spadl/test_metrica*.py` or `tests/tracking/test_direction.py`

- [ ] **Step 1: Write the failing test** (Metrica events must now raise on ET-without-flag) — mirror Task 4 Step 1 with a Metrica SPADL-input fixture incl. period 3. Run → FAIL.
- [ ] **Step 2: Insert the guard** `require_et_direction(actions["period_id"], home_team_start_left_extratime, source="metrica convert_to_actions")` before the `home_attacks_right_per_period` call.
- [ ] **Step 3: Run → PASS.**

---

## Task 6: Cross-provider parity test (message equality + happy-path equivalence)

**Files:** Test: `tests/tracking/test_direction.py`

- [ ] **Step 1: Write the parity test.**

```python
import re, pytest
_CONVERTERS = [
    ("sportec convert_to_frames", lambda f: _call_sportec_tracking(f)),
    ("gradientsports convert_to_frames", lambda f: _call_gs_tracking(f)),
    ("sportec convert_to_actions", lambda f: _call_sportec_events(f)),
    ("gradientsports convert_to_actions", lambda f: _call_gs_events(f)),
    ("metrica convert_to_actions", lambda f: _call_metrica_events(f)),
]

@pytest.mark.parametrize("source,call", _CONVERTERS)
def test_all_converters_raise_same_message_shape_on_et_without_flag(source, call):
    with pytest.raises(ValueError) as exc:
        call(et=True, flag=None)
    msg = str(exc.value)
    assert msg.startswith(f"{source}: data contains ET periods")
    assert "home_team_start_left_extratime" in msg

@pytest.mark.parametrize("source,call", _CONVERTERS)
def test_all_converters_et_orientation_reflects_with_flag(source, call):
    # review B: assert the flag actually orients ET, not just "ET present + no raise".
    # The SAME ET input under flag=True vs flag=False must reflect a home-player ET row's x
    # across the pitch (LTR output in [0,105] => x_left + x_right == 105). Catches
    # cross-provider drift where the guard fires identically but the flip math diverges.
    out_left = call(et=True, flag=True)
    out_right = call(et=True, flag=False)
    et_l = out_left[out_left["period_id"].isin([3, 4])].sort_values("player_id")["x"].iloc[0]
    et_r = out_right[out_right["period_id"].isin([3, 4])].sort_values("player_id")["x"].iloc[0]
    assert abs(et_l + et_r - 105.0) < 1e-6, f"{source}: ET not reflected by flag: {et_l} vs {et_r}"
```

- [ ] **Step 2: Run → PASS.** Implement the `_call_*` thin wrappers using the shared `conftest.py` builders (Task 4 Step 0). Each wrapper builds the same RT+ET input, varies only the ET flag, returns the converter output. (For events converters whose output column is not literally `x`/`player_id`, adapt to the SPADL action coordinate column — assert the analogous start_x reflection.)

---

## Task 7: RT-only golden no-regress (uses Task 0 goldens)

**Files:** Test: `tests/regressions/extratime/test_rt_no_regress.py` (create)

- [ ] **Step 1: Write the gate.**

```python
import pandas as pd, pytest
from pathlib import Path
GOLD = Path(__file__).parent
_CASES = ["sportec_tracking", "gs_tracking", "sportec_events", "gs_events", "metrica_events"]

@pytest.mark.parametrize("case", _CASES)
def test_rt_only_output_bit_identical_to_3_30_golden(case):
    golden = pd.read_parquet(GOLD / f"golden_{case}_rt.parquet")
    current = _run_converter_rt_only(case)   # SAME RT-only input as capture_goldens.py
    # review G: golden is parquet-roundtripped, current is in-memory -> assert VALUE equality,
    # not backend-dtype identity (pyarrow-vs-numpy variance across capture/test runs).
    # Converter dtype contracts are already covered by each converter's _finalize_output tests.
    pd.testing.assert_frame_equal(
        current.reset_index(drop=True), golden.reset_index(drop=True), check_dtype=False
    )
```

- [ ] **Step 2: Run → PASS** (RT-only is unaffected by the guard, so current == 3.30 golden). If it fails, the change leaked into the RT path — investigate before proceeding.

---

## Task 8: ET round-trip fixtures — GS real-data (delivered) + IDSSE/Metrica synthetic

**Files:**
- Use (delivered): `tests/regressions/extratime/gs_et/{frames,actions,meta}.parquet` + `README.md`
- Create: `tests/regressions/extratime/{sportec_idsse_et,metrica_et}/{frames_or_actions}.parquet` (synthetic)
- Create: `tests/regressions/extratime/test_real_et_roundtrip.py`

**Context (lakehouse delivery + §8 audit):** the lakehouse §8 inventory found **zero** IDSSE/Metrica ET matches in bronze (Bundesliga regular season has no ET; cup-with-ET not ingested) and 5 GS ET matches. So only GS has real ET data — delivered as match **10517** (A-League, full 4-period; 459 distinct period-3 frames; `meta.home_team_start_left_extratime=True`). IDSSE/Metrica get **synthetic** ET fixtures (honest unit+parity+golden already cover them; the synthetic fixture closes the e2e ET round-trip).

> **GS fixture shape (from `gs_et/README.md`):** `frames.parquet` is **raw bronze** (`match_id`, `period`, `frame_num`, `period_elapsed_time`, `team_side`, `is_ball`, `jersey_num`, `x`, `y`, `z`) — **period 3 only**, 459 frames × (22 players + ball). **No roster delivered.** The ET-direction logic is **roster-independent** (keys on team membership + period + coords), so flatten the raw frames to the GS converter input the loader-way, with a **synthesized** roster. Reference helper (review a):
>
> ```python
> def _convert_gs_tracking(raw, *, home_team_id, away_team_id, home_team_start_left,
>                          home_team_start_left_extratime):
>     from silly_kicks.tracking.gradientsports import add_gradientsports_player_ids, convert_to_frames
>     hid, aid = int(home_team_id), int(away_team_id)              # review f: explicit int cast
>     jf = raw.rename(columns={"frame_num": "frame_id", "period": "period_id"}).copy()
>     jf["game_id"] = int(raw["match_id"].iloc[0])
>     jf["team_id"] = jf["team_side"].map({"home": hid, "away": aid})
>     jf["jersey_number"] = jf["jersey_num"].astype(str)
>     jf["x_centered"], jf["y_centered"] = jf["x"], jf["y"]        # bronze GS = centered metres
>     # synthesized roster: one row per (team_side, jersey) -> a stable synthetic player_id
>     ros = (jf[~jf["is_ball"]][["team_id", "jersey_number"]].drop_duplicates()
>            .assign(shirt_number=lambda d: d["jersey_number"],
>                    player_id=lambda d: range(1, len(d) + 1),
>                    position_group_type="MF"))
>     resolved, _ = add_gradientsports_player_ids(jf, ros, home_team_id=hid, away_team_id=aid)
>     frames, _ = convert_to_frames(resolved, home_team_id=hid, home_team_start_left=home_team_start_left,
>                                   home_team_start_left_extratime=home_team_start_left_extratime)
>     return frames
> ```
> (Adjust column names to the actual `EXPECTED_INPUT_COLUMNS` when implementing.) `meta.home_team_id` may read as int `364` or str `"364"` — the cast handles it. **away_team_id** is not in `meta`; derive it from the non-home `team_side`/roster or extend the helper.
>
> **Period-3-only input (review b):** the fixture has periods 3 only — verify `convert_to_frames` accepts period-3-only frames (PER_PERIOD_ABSOLUTE flips per period from the flag map, so it *should* not require period 1/2 present, but **confirm against the converter's input validation**; if it rejects, prepend a couple of minimal synthetic period-1 rows). **Pitch dims (review d):** silly-kicks tracking output is the SPADL **0–105 × 0–68** frame (`TRACKING_CONSTRAINTS`), so the `between(0,105)`/`between(0,68)` bounds assertions are contract-correct.

- [ ] **Step 1: GS real-data round-trip test** using the delivered fixture. The `meta` carries the **true** ET flag, so assert **orientation correctness**, not just no-raise:

```python
import pandas as pd, pytest
from pathlib import Path
GS = Path(__file__).parent / "gs_et"

def test_gs_real_et_roundtrip_correct_orientation():
    frames = pd.read_parquet(GS / "frames.parquet")
    meta = pd.read_parquet(GS / "meta.parquet").iloc[0]
    # reuse the loader's GS flatten+convert, OR call convert_to_frames directly on the
    # already-resolved frames. With the TRUE flag -> no raise + ET frames present + in-bounds.
    out = _convert_gs_tracking(frames, home_team_id=meta["home_team_id"],
                               home_team_start_left=bool(meta["home_start_left"]),
                               home_team_start_left_extratime=bool(meta["home_team_start_left_extratime"]))
    et = out[out["period_id"].isin([3, 4])]
    assert len(et) > 0
    assert et["x"].between(0, 105).all() and et["y"].between(0, 68).all()

def test_gs_real_et_raises_without_flag():
    frames = pd.read_parquet(GS / "frames.parquet")
    meta = pd.read_parquet(GS / "meta.parquet").iloc[0]
    with pytest.raises(ValueError, match="ET periods"):
        _convert_gs_tracking(frames, home_team_id=meta["home_team_id"],
                             home_team_start_left=bool(meta["home_start_left"]),
                             home_team_start_left_extratime=None)
```

- [ ] **Step 2: Synthesize IDSSE/Metrica ET fixtures** — minimal Sportec-shaped / Metrica-shaped frames/actions with `period_id IN (3,4)` rows, mirroring the existing `tests/tracking/test_sportec_tracking*.py` / `tests/spadl/test_metrica*.py` input builders (reuse the Task 4 Step 0 `conftest.py` builders with `et=True`). Commit the parquet for an e2e round-trip; with a chosen ET flag → converts + ET present + in-bounds; without → raises.

> **Assertion-strength asymmetry (review c):** synthetic IDSSE/Metrica fixtures have **no ground-truth ET orientation** (they're synthesized), so this step proves **e2e parquet-roundtrip shape + no-crash + presence + bounds — NOT orientation truth**. That is intentional and sufficient: **orientation reflection is already asserted for all 5 converters by Task 6** (`x_left + x_right == 105`, flag=True vs False on the same synthetic input). Only GS has a real-data orientation-truth test (Step 1), because only GS has real ET data + a true flag.

- [ ] **Step 3: Run → PASS.**

Run: `pytest tests/regressions/extratime/test_real_et_roundtrip.py -q` → PASS.

---

## Task 9: `filter_extratime_frames` public helper + loader collapse

**Files:**
- Modify: `silly_kicks/tracking/utils.py`
- Modify: `scripts/_loader_pining.py` (collapse `_apply_et_direction` → `filter_extratime_frames`)
- Modify: `scripts/_loader_databricks.py` (apply on sportec frames)
- Test: `tests/tracking/test_utils.py` (or existing) + the existing loader tests stay green

- [ ] **Step 1: Write the failing test for `filter_extratime_frames`.**

```python
import pandas as pd, pytest
from silly_kicks.tracking.utils import filter_extratime_frames

def test_filter_extratime_drops_et_with_warning():
    f = pd.DataFrame({"period_id": [1, 1, 3, 4]})
    with pytest.warns(UserWarning, match="ET"):
        out = filter_extratime_frames(f, label="gs 1")
    assert set(out["period_id"]) == {1}

def test_filter_extratime_noop_without_et():
    f = pd.DataFrame({"period_id": [1, 2]})
    out = filter_extratime_frames(f, label="x")
    assert len(out) == 2
```

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `filter_extratime_frames(frames, *, label)` in `tracking/utils.py` with the calibration-labelled docstring (spec §7): drop `period_id in {3,4}` with a `UserWarning`; no-op otherwise. Export from `silly_kicks.tracking`.
- [ ] **Step 4: Collapse the loader.** **Depends on: Task 4 (Sportec guard) (review e)** — the `_loader_databricks.py` filter is only *necessary* once the Sportec silent default is removed (it RAISES in 4.0.0). Do not land this step against a pre-Task-4 tree, or it would silently drop ET that the old code would still (mis-)process. **Scope (review A):** both `scripts/_loader_pining.py` and `scripts/_loader_databricks.py` are the **TF-24 calibration** harness's loaders (sample-based; `scripts/`, per CLAUDE.md "I/O lives in scripts/…"). They are **NOT** AC-1 production — AC-1 production is `luxury-lakehouse src/analytics/action_context/pipeline.py`, which (per spec §7 + Phase A) must source the real ET flag from `MatchMeta`, **never** filter. Applying `filter_extratime_frames` here is therefore correct (calibration ET-skip is acceptable). In `scripts/_loader_pining.py`, replace the local `_apply_et_direction` (which returns `(frames, et_param)`): resolve `et_param` inline from metadata, and where it dropped ET, call `filter_extratime_frames`. Update the existing loader unit tests to target `filter_extratime_frames`. Add `filter_extratime_frames(frames, label="sportec ...")` in `_loader_databricks.py` before its `convert_to_frames` — required now, because the Sportec silent default is gone (it RAISES in 4.0.0), so the calibration databricks path must explicitly drop ET. Add a code comment in both loaders: *"calibration only — AC-1 production sources home_team_start_left_extratime via MatchMeta (lakehouse Phase A), never filters."*
- [ ] **Step 5: Run loader tests → PASS.**

Run: `pytest tests/calibration/test_loader_pining.py tests/calibration/test_loader_databricks.py tests/tracking/test_utils.py -q` → PASS.

---

## Task 9b: TF-24 sweep memory bound — CLI match-subset / limit flags

**Why (in scope per user):** the original goal — running the local TF-24 sweep — is blocked by **two** things: the ET crash (Tasks 1-9) and an **OOM**. The pining loader already supports subsetting (`load_matches(match_ids=…, tracking_limit=…)`; the xT corpus uses `tracking_limit=50`), but the main fold load hardcodes `match_ids=None` + no `tracking_limit` (`calibrate_tracking_defaults.py:84`), and the CLI exposes no knob, so it loads **all 64 GS matches at full depth → ~7.8 GiB OOM** on a local machine.

**Files:**
- Modify: `scripts/calibrate_tracking_defaults.py` (argparse + `_load_fold`)
- Modify: `scripts/_loader_pining.py` (`load_matches` gains `max_per_provider`)
- Test: `tests/calibration/test_loader_pining.py` + a CLI-arg-threading test

- [ ] **Step 1: Failing test — `load_matches` caps per provider.**

```python
def test_load_matches_max_per_provider_truncates(monkeypatch):
    import scripts._loader_pining as L
    monkeypatch.setattr(L, "_list_matches", lambda p, t, b: [{"id": str(i), "artifacts": {}} for i in range(5)])
    monkeypatch.setattr(L, "_download_artifacts", lambda *a, **k: {})
    monkeypatch.setattr(L, "_build_match", lambda *a, **k: (None, None, None))
    got = [mid for _p, mid, *_ in L.load_matches(providers=["gradientsports"], max_per_provider=2)]
    assert got == ["0", "1"]
```

- [ ] **Step 2: Run → FAIL** (`load_matches` has no `max_per_provider`).

- [ ] **Step 3: Implement.** In `_loader_pining.py` `load_matches`, add `max_per_provider: int | None = None`; after computing `wanted`, `if max_per_provider: wanted = wanted[:max_per_provider]`. In `calibrate_tracking_defaults.py`: add args `--match-ids` (repeatable `provider:id1,id2`), `--max-matches-per-provider` (int), `--tracking-limit` (int); parse `--match-ids` into the `{provider: [ids]}` dict; thread all three into `_load_fold`'s `load_matches(..., match_ids=parsed, tracking_limit=args.tracking_limit, max_per_provider=args.max_matches_per_provider)`. (Databricks loader: thread `tracking_limit` where applicable; `max_per_provider`/`match_ids` are pining-shaped — guard if unsupported there.)

- [ ] **Step 4: Run → PASS.** Manual smoke (optional, network): `python scripts/calibrate_tracking_defaults.py --stage 1 --source pining --providers gradientsports --max-matches-per-provider 3 --tracking-limit 200 --n-trials 2 --store /tmp/s.db` — bounded memory, completes.

---

## Task 10: Atomic-mirror verification

**Files:** (likely none) — verification only.

- [ ] **Step 1: Confirm atomic SPADL does not re-do per-period-absolute conversion.**

Run: `grep -rn "home_attacks_right_per_period\|home_team_start_left_extratime\|to_spadl_ltr" silly_kicks/atomic/`
Expected: no hits (atomic operates on already-LTR-normalised SPADL actions). If hits exist, add the guard there mirroring Tasks 4-5 and note it. Record the result either way.

---

## Task 11: Version 4.0.0 + docs (hard gate)

**Files:** `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `CHANGELOG.md`, `uv.lock`

- [ ] **Step 1: Bump all five to 4.0.0** (`version`/`__version__`/Current-release/new `## [4.0.0]` heading), then `uv lock`.
- [ ] **Step 2: CHANGELOG `[4.0.0]` BREAKING entry** — the symmetric ET guard, the 5 converters affected, the public `require_et_direction` + `filter_extratime_frames`, the `_direction → direction` rename, the **standardized GS message wording** (review C), and the migration (pass `home_team_start_left_extratime`); reference ADR-010. **Cross-repo ordering (review H):** include — *"Lakehouse / consumers with ET matches: upgrade to the lakehouse Phase-A PR first (adds `MatchMeta.home_team_start_left_extratime` and plumbs it to `convert_to_frames`/`convert_to_actions`) BEFORE bumping the silly-kicks pin to 4.0.0. A pin bump without that plumbing will raise on any in-scope ET match."*
- [ ] **Step 3: Verify version consistency.**

Run: `grep -rn "4\.0\.0" pyproject.toml silly_kicks/__init__.py TODO.md` + `grep -m1 'silly-kicks' -A1 uv.lock` → all 4.0.0; no stale 3.30.x in authoritative files.

---

## Task 12: Final gates + single commit

- [ ] **Step 1:** `python -m ruff format silly_kicks/ scripts/ tests/` + `python -m ruff check silly_kicks/ scripts/ tests/` → clean.
- [ ] **Step 2:** `python -m pyright silly_kicks/` → 0 errors (full package, per project rule).
- [ ] **Step 3:** Full non-e2e suite: `pytest tests/ -m "not e2e" -q` → all pass. Then e2e where feasible: `pytest tests/ -m e2e -k "direction or sportec or gradientsports or metrica" -q`.
- [ ] **Step 4:** Run `/final-review` (mandatory). Update C4 only if a container/relationship changed (it should not — converters are existing components).
- [ ] **Step 5:** Present the diff and request approval for the single commit. On approval: ONE commit (spec + ADR + plan + impl + GS+synthetic fixtures + version), push, open PR. **Ship-gate (§9 item 6) is SATISFIED** (the §8 audit returned a clean zero — see header), so the normal release flow applies: merge → main CI green → tag `v4.0.0` → PyPI publish. Coordinate the lakehouse Phase-A PR + pin bump ordering (CHANGELOG migration note) but silly-kicks 4.0.0 is **not** held.

---

## Self-review (against spec v3)

- §2 symmetric guard + public API → Tasks 2-5, re-exports Task 1/2. ✓
- §3 full rename → Task 1. ✓
- §6 parity (msg + happy-path) → Task 6; RT-only golden → Tasks 0+7; events round-trip → Tasks 4-5; real-data ET → Task 8. ✓
- §7 `filter_extratime_frames` labelled-public + loader collapse → Task 9. ✓
- §5 4.0.0 → Task 11. ✓ ADR-010 + spec on-branch → Task 12 commit. ✓
- §9 item 6 ship-gate → Task 12 Step 5 (publish held). ✓
- Atomic mirror → Task 10 (verify). ✓
- **Sequencing hazard pinned:** goldens captured in Task 0 BEFORE any converter edit. ✓
