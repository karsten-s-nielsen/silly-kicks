# Kloppy Sportec + Metrica + `_SoccerActionCoordinateSystem` Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Whitelist `Provider.SPORTEC` and `Provider.METRICA` in the kloppy SPADL converter, fix the latent `_SoccerActionCoordinateSystem.__init__` bug that breaks all kloppy-based conversion on real datasets, and align kloppy output coords with the silly-kicks `[0, field_length] × [0, field_width]` clamping convention.

**Architecture:** Single PR, single squash commit on branch `feat/kloppy-sportec-metrica`. TDD: RED tests first per phase, then minimal implementation. Real-fixture tests vendored from kloppy's BSD-3 test files. No new event-type handlers needed (empirical probe confirmed both providers' event types are already covered by existing `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES`).

**Tech Stack:** Python 3.10+, kloppy>=3.15.0, pandas>=2.1.1, numpy>=1.26.0, pytest, ruff, pyright. Existing silly-kicks SPADL converter pipeline (`silly_kicks/spadl/kloppy.py`).

**Spec:** `docs/superpowers/specs/2026-04-28-kloppy-sportec-metrica-design.md`

**Operating rules:**
- No commits without explicit user approval, but the WIP commits below are safe to make freely (they get squashed at the end before user-approval-gated push).
- All commands are run from `D:\Development\karstenskyt__silly-kicks`.
- Bash tool: any command potentially exceeding 30s must use `run_in_background=true`. Default `timeout=15000`.
- All `git push`, `gh pr ...`, `git tag --push` commands need explicit user approval (per standing rule). The plan calls these out as `[USER APPROVAL GATE]`.

---

## Phase 1: Setup

### Task 1: Create branch and stage spec doc

**Files:**
- Currently untracked on `main`: `docs/superpowers/specs/2026-04-28-kloppy-sportec-metrica-design.md`
- Currently untracked on `main`: `docs/superpowers/plans/2026-04-28-kloppy-sportec-metrica.md` (this file)

- [ ] **Step 1: Confirm clean tree state**

Run:
```bash
git status --short
```
Expected output (the spec, this plan, and the two pre-existing untracked items):
```
?? README.md.backup
?? docs/superpowers/
?? uv.lock
```

- [ ] **Step 2: Verify on main at the expected commit**

Run:
```bash
git rev-parse --abbrev-ref HEAD && git log --oneline -1
```
Expected:
```
main
07211c6 feat(atomic): SPADL parity for the 1.1.0 → 1.4.0 helper family — silly-kicks 1.5.0 (#10)
```

- [ ] **Step 3: Create the feat branch**

Run:
```bash
git checkout -b feat/kloppy-sportec-metrica
```
Expected: `Switched to a new branch 'feat/kloppy-sportec-metrica'`

- [ ] **Step 4: Stage the spec and plan docs onto the branch**

Run:
```bash
git add docs/superpowers/specs/2026-04-28-kloppy-sportec-metrica-design.md docs/superpowers/plans/2026-04-28-kloppy-sportec-metrica.md
```

- [ ] **Step 5: Commit (WIP — gets squashed later)**

Run:
```bash
git commit -m "WIP: spec + plan for kloppy Sportec/Metrica + _SoccerActionCoordinateSystem fix"
```

---

### Task 2: Create `tests/datasets/kloppy/` directory and vendor fixtures

**Files:**
- Create: `tests/datasets/kloppy/sportec_events.xml` (15 KB, from kloppy `tests/files/sportec_events.xml`)
- Create: `tests/datasets/kloppy/sportec_meta.xml` (12 KB, from kloppy `tests/files/sportec_meta.xml`)
- Create: `tests/datasets/kloppy/metrica_events.json` (1.7 MB, from kloppy `tests/files/metrica_events.json`)
- Create: `tests/datasets/kloppy/epts_metrica_metadata.xml` (34 KB, from kloppy `tests/files/epts_metrica_metadata.xml`)

- [ ] **Step 1: Create the directory**

Run:
```bash
mkdir -p tests/datasets/kloppy
```

- [ ] **Step 2: Download all four fixtures from kloppy GitHub master**

Run:
```bash
curl -sL -o tests/datasets/kloppy/sportec_events.xml \
  "https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/tests/files/sportec_events.xml"
curl -sL -o tests/datasets/kloppy/sportec_meta.xml \
  "https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/tests/files/sportec_meta.xml"
curl -sL -o tests/datasets/kloppy/metrica_events.json \
  "https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/tests/files/metrica_events.json"
curl -sL -o tests/datasets/kloppy/epts_metrica_metadata.xml \
  "https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/tests/files/epts_metrica_metadata.xml"
```

- [ ] **Step 3: Verify file sizes match expected**

Run:
```bash
ls -la tests/datasets/kloppy/
```
Expected (sizes ±1KB):
```
sportec_events.xml         ~15 KB
sportec_meta.xml           ~12 KB
metrica_events.json        ~1.7 MB
epts_metrica_metadata.xml  ~34 KB
```

If any file is < 100 bytes, the curl failed silently (likely 404 due to upstream rename). Re-fetch from the right path before proceeding — listing is at `https://github.com/PySport/kloppy/tree/master/kloppy/tests/files`.

- [ ] **Step 4: Smoke-test that all four files load via kloppy**

Run:
```bash
uv run --with "kloppy>=3.15" python -c "
from kloppy import sportec, metrica
ds_sp = sportec.load_event(
    event_data='tests/datasets/kloppy/sportec_events.xml',
    meta_data='tests/datasets/kloppy/sportec_meta.xml',
)
print('sportec OK:', ds_sp.metadata.provider, len(ds_sp.events), 'events')
ds_me = metrica.load_event(
    event_data='tests/datasets/kloppy/metrica_events.json',
    meta_data='tests/datasets/kloppy/epts_metrica_metadata.xml',
)
print('metrica OK:', ds_me.metadata.provider, len(ds_me.events), 'events')
"
```
Expected:
```
sportec OK: sportec 29 events
metrica OK: metrica 3594 events
```

- [ ] **Step 5: Stage and commit (WIP)**

Run:
```bash
git add tests/datasets/kloppy/
git commit -m "WIP: vendor kloppy test fixtures (sportec + metrica)"
```

---

### Task 3: Create `tests/datasets/kloppy/README.md` with provenance + licenses

**Files:**
- Create: `tests/datasets/kloppy/README.md`

- [ ] **Step 1: Write the README**

Create `tests/datasets/kloppy/README.md` with this exact content:

```markdown
# Kloppy test fixtures

These files are vendored from the [kloppy](https://github.com/PySport/kloppy) test suite for use as test fixtures by `tests/spadl/test_kloppy.py`.

## Files

| File | Origin | License |
|---|---|---|
| `sportec_events.xml` | [PySport/kloppy `kloppy/tests/files/sportec_events.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/sportec_events.xml) | BSD-3-Clause (kloppy synthetic test data) |
| `sportec_meta.xml` | [PySport/kloppy `kloppy/tests/files/sportec_meta.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/sportec_meta.xml) | BSD-3-Clause (kloppy synthetic test data) |
| `metrica_events.json` | [PySport/kloppy `kloppy/tests/files/metrica_events.json`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/metrica_events.json) | Originally from [`metrica-sports/sample-data`](https://github.com/metrica-sports/sample-data) Sample Game 2 — **CC-BY-NC-4.0** (test-data-only; not redistributed in the wheel) |
| `epts_metrica_metadata.xml` | [PySport/kloppy `kloppy/tests/files/epts_metrica_metadata.xml`](https://github.com/PySport/kloppy/blob/master/kloppy/tests/files/epts_metrica_metadata.xml) | BSD-3-Clause (kloppy synthetic test data) |

## Notes

- The Metrica events JSON and the EPTS metadata are from **different matches**. This pairing matches kloppy's own test fixture pairing (see `kloppy/tests/test_metrica_events.py`'s FIXME) and is sufficient for converter contract testing — it exercises the parsing pipeline and event-type mapping without depending on perfect alignment between event coordinates and player metadata.
- These fixtures are **test-data only**. They are excluded from the published `silly-kicks` wheel via `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` in `pyproject.toml`, which packages only the `silly_kicks/` source tree.
- License compliance: kloppy is BSD-3-Clause (attribution preserved by linking back). The Metrica file's CC-BY-NC license restricts commercial **redistribution**; using it for testing in a non-commercial open-source library is permitted, and we do not redistribute it in the published wheel.
```

- [ ] **Step 2: Stage and commit (WIP)**

Run:
```bash
git add tests/datasets/kloppy/README.md
git commit -m "WIP: document kloppy fixture provenance + licenses"
```

---

## Phase 2: Bug fix (`_SoccerActionCoordinateSystem`) — RED → GREEN

### Task 4: Write `TestKloppyCoordinateSystemFix::test_real_dataset_transform_does_not_typeerror` (RED)

**Files:**
- Modify: `tests/spadl/test_kloppy.py`

**Background:** This test loads the Sportec fixture, monkey-patches `_SUPPORTED_PROVIDERS` to suppress the unrecognized-provider warning (the bug fix is independent of the whitelist), and asserts `convert_to_actions()` does not raise `TypeError`. RED today because line 168 of `silly_kicks/spadl/kloppy.py` calls `_SoccerActionCoordinateSystem(pitch_length=..., pitch_width=...)` and the class has no `__init__`.

- [ ] **Step 1: Add the new test class to `tests/spadl/test_kloppy.py`**

Open `tests/spadl/test_kloppy.py`. After the existing `TestKloppyPreserveNative` class (around line 67) and before the `# E2E tests` comment (around line 70), insert:

```python
# ---------------------------------------------------------------------------
# Real-fixture tests (vendored kloppy fixtures under tests/datasets/kloppy/)
# ---------------------------------------------------------------------------

from pathlib import Path

import pandas as pd

from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.schema import KLOPPY_SPADL_COLUMNS


_KLOPPY_FIXTURES_DIR = Path(__file__).parent.parent / "datasets" / "kloppy"


class TestKloppyCoordinateSystemFix:
    """Regression test for the latent _SoccerActionCoordinateSystem bug.

    Pre-fix, ``convert_to_actions`` raised ``TypeError: _SoccerActionCoordinateSystem()
    takes no arguments`` on line 168 the moment a real EventDataset reached
    ``dataset.transform(...)``. The pre-existing mock-based tests never exercised
    this path. This test exists to ensure the bug stays fixed.
    """

    def test_real_dataset_transform_does_not_typeerror(self):
        """convert_to_actions on a real Sportec dataset must not raise TypeError."""
        from kloppy import sportec
        from kloppy.domain import Provider
        from packaging import version

        from silly_kicks.spadl import kloppy as spkloppy

        ds = sportec.load_event(
            event_data=str(_KLOPPY_FIXTURES_DIR / "sportec_events.xml"),
            meta_data=str(_KLOPPY_FIXTURES_DIR / "sportec_meta.xml"),
        )

        # Whitelist Sportec for this test only (the whitelist add lands in a later task);
        # this isolates the bug-fix coverage from the whitelist add.
        original = dict(spkloppy._SUPPORTED_PROVIDERS)
        spkloppy._SUPPORTED_PROVIDERS[Provider.SPORTEC] = version.parse("3.15.0")
        try:
            # Must not raise TypeError. The conversion may emit warnings; that's fine.
            actions, _report = spkloppy.convert_to_actions(ds, game_id="bug_fix_smoke")
        finally:
            spkloppy._SUPPORTED_PROVIDERS.clear()
            spkloppy._SUPPORTED_PROVIDERS.update(original)

        assert isinstance(actions, pd.DataFrame)
        assert len(actions) > 0
```

- [ ] **Step 2: Run the new test and confirm RED**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyCoordinateSystemFix::test_real_dataset_transform_does_not_typeerror -v
```
Expected: `FAILED` with `TypeError: _SoccerActionCoordinateSystem() takes no arguments`.

- [ ] **Step 3: Commit RED test (WIP)**

Run:
```bash
git add tests/spadl/test_kloppy.py
git commit -m "WIP: RED test for _SoccerActionCoordinateSystem bug"
```

---

### Task 5: Implement `_SoccerActionCoordinateSystem.__init__` fix (GREEN)

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py:241-262`

- [ ] **Step 1: Replace the broken class definition**

Open `silly_kicks/spadl/kloppy.py`. Locate the existing `_SoccerActionCoordinateSystem` class (lines 241-262):

```python
class _SoccerActionCoordinateSystem(CoordinateSystem):
    @property
    def provider(self) -> Provider:
        return "SoccerAction"  # type: ignore[reportReturnType]  # kloppy API varies by version

    @property
    def origin(self) -> Origin:
        return Origin.BOTTOM_LEFT

    @property
    def vertical_orientation(self) -> VerticalOrientation:
        return VerticalOrientation.BOTTOM_TO_TOP

    @property
    def pitch_dimensions(self) -> PitchDimensions:
        return MetricPitchDimensions(
            x_dim=Dimension(0, spadlconfig.field_length),
            y_dim=Dimension(0, spadlconfig.field_width),
            pitch_length=self.pitch_length,
            pitch_width=self.pitch_width,
            standardized=True,
        )
```

Replace with (note the new `__init__` and explicit `pitch_length`/`pitch_width` property overrides reading from instance state):

```python
class _SoccerActionCoordinateSystem(CoordinateSystem):
    def __init__(self, *, pitch_length: float, pitch_width: float) -> None:
        self._pitch_length = pitch_length
        self._pitch_width = pitch_width

    @property
    def provider(self) -> Provider:
        return "SoccerAction"  # type: ignore[reportReturnType]  # kloppy API varies by version

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
```

- [ ] **Step 2: Run the regression test and confirm GREEN**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyCoordinateSystemFix::test_real_dataset_transform_does_not_typeerror -v
```
Expected: `PASSED`.

- [ ] **Step 3: Run all kloppy tests to confirm no regression of existing mock tests**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py -v
```
Expected: all tests pass (the existing 4 mock-based `TestKloppyPreserveNative` tests + the new regression test + the 1 e2e skip stub).

- [ ] **Step 4: Commit (WIP)**

Run:
```bash
git add silly_kicks/spadl/kloppy.py
git commit -m "WIP: fix _SoccerActionCoordinateSystem __init__ for kloppy 3.15+"
```

---

## Phase 3: Provider whitelist (Sportec + Metrica) — RED → GREEN

### Task 6: Add session-scoped fixture loaders + `TestKloppySportec` (RED, 5 tests)

**Files:**
- Modify: `tests/spadl/test_kloppy.py`

**Background:** The dataset loading is the slow part (~1.7MB JSON parse for Metrica). Module-scoped fixtures load each provider once. Each test then runs `convert_to_actions` on the parsed dataset.

- [ ] **Step 1: Add module-scoped fixtures + TestKloppySportec class**

Open `tests/spadl/test_kloppy.py`. After the `TestKloppyCoordinateSystemFix` class (added in Task 4), insert:

```python
@pytest.fixture(scope="module")
def sportec_dataset():
    """Module-scoped: parse the Sportec fixture once per test module."""
    from kloppy import sportec
    return sportec.load_event(
        event_data=str(_KLOPPY_FIXTURES_DIR / "sportec_events.xml"),
        meta_data=str(_KLOPPY_FIXTURES_DIR / "sportec_meta.xml"),
    )


@pytest.fixture(scope="module")
def metrica_dataset():
    """Module-scoped: parse the Metrica fixture once per test module."""
    from kloppy import metrica
    return metrica.load_event(
        event_data=str(_KLOPPY_FIXTURES_DIR / "metrica_events.json"),
        meta_data=str(_KLOPPY_FIXTURES_DIR / "epts_metrica_metadata.xml"),
    )


class TestKloppySportec:
    """End-to-end conversion tests for the Sportec (IDSSE) provider."""

    def test_convert_to_actions_basic(self, sportec_dataset):
        """convert_to_actions returns (DataFrame, ConversionReport) with the canonical
        SPADL schema and at least one row from the Sportec fixture (29 events)."""
        from silly_kicks.spadl import kloppy as spkloppy

        actions, report = spkloppy.convert_to_actions(sportec_dataset, game_id="sportec_smoke")

        assert isinstance(actions, pd.DataFrame)
        # Schema check: every canonical column present, in declared order.
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())
        # Dtype check per schema.
        for col, expected_dtype in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, f"{col}: got {actions[col].dtype}, want {expected_dtype}"
        # Non-empty: the Sportec fixture has 29 events; some are excluded (CARD, GENERIC,
        # BALL_OUT, RECOVERY) so action count is < 29 but > 0.
        assert len(actions) > 0
        # Report shape.
        assert report.provider == "Kloppy"
        assert report.total_events > 0
        assert report.total_actions == len(actions)

    def test_no_unrecognized_event_types(self, sportec_dataset):
        """All Sportec event types are covered by _MAPPED_EVENT_TYPES ∪ _EXCLUDED_EVENT_TYPES.
        Catches future kloppy versions that introduce new event types we don't handle."""
        import warnings as warnings_mod

        from silly_kicks.spadl import kloppy as spkloppy

        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, report = spkloppy.convert_to_actions(sportec_dataset, game_id="sportec_unrec")

        assert report.unrecognized_counts == {}
        unrec_warnings = [str(w.message) for w in caught if "unrecognized event types" in str(w.message)]
        assert unrec_warnings == [], f"Got unrecognized-event warnings: {unrec_warnings}"

    def test_action_id_unique_and_zero_indexed(self, sportec_dataset):
        """action_id must be range(len(actions)) per game_id."""
        from silly_kicks.spadl import kloppy as spkloppy

        actions, _ = spkloppy.convert_to_actions(sportec_dataset, game_id="sportec_aid")
        assert actions["action_id"].tolist() == list(range(len(actions)))

    def test_preserve_native_with_dict_raw_event(self, sportec_dataset):
        """preserve_native surfaces a raw_event field as an extra column (Sportec raw_event is dict)."""
        from silly_kicks.spadl import kloppy as spkloppy

        first_event = next(iter(sportec_dataset.events))
        raw_keys = list(first_event.raw_event.keys())
        # Pick a key that doesn't collide with KLOPPY_SPADL_COLUMNS.
        candidate = next((k for k in raw_keys if k not in KLOPPY_SPADL_COLUMNS), None)
        assert candidate is not None, f"Sportec raw_event has no non-schema-overlapping key. Keys: {raw_keys}"

        actions, _ = spkloppy.convert_to_actions(
            sportec_dataset, game_id="sportec_pn", preserve_native=[candidate]
        )
        assert candidate in actions.columns
        # At least the first row should have a non-NaN value for the preserved column
        # (synthetic dribble rows get NaN; the very first action originates from a real event).
        assert actions[candidate].notna().any()

    def test_input_dataset_not_mutated(self, sportec_dataset):
        """convert_to_actions must not mutate the input dataset's metadata or event count."""
        from silly_kicks.spadl import kloppy as spkloppy

        before_provider = sportec_dataset.metadata.provider
        before_coord_sys_id = id(sportec_dataset.metadata.coordinate_system)
        before_event_count = len(sportec_dataset.events)

        _, _ = spkloppy.convert_to_actions(sportec_dataset, game_id="sportec_mut")

        assert sportec_dataset.metadata.provider == before_provider
        assert id(sportec_dataset.metadata.coordinate_system) == before_coord_sys_id
        assert len(sportec_dataset.events) == before_event_count
```

- [ ] **Step 2: Run the new tests and confirm RED**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppySportec -v
```
Expected: all 5 tests fail. The exact failure modes vary:
- `test_convert_to_actions_basic`: emits the unrecognized-provider warning then likely passes (because the bug is already fixed in Task 5). Actually it depends — if `Provider.SPORTEC not in _SUPPORTED_PROVIDERS`, only a warning fires; conversion continues. So this may already be GREEN after Task 5.
- `test_no_unrecognized_event_types`: should still pass (the unrec-event check is on `EventType`, not on provider whitelisting).
- `test_action_id_unique_and_zero_indexed`: should pass.
- `test_preserve_native_with_dict_raw_event`: should pass.
- `test_input_dataset_not_mutated`: should pass.

**Reality check:** after Task 5's bug fix, the Sportec tests may all pass even without Task 7's whitelist add. That's fine — the whitelist add silences the unrecognized-provider warning, which we'll assert separately in Task 8 via the `ConversionReport` test. Document the actual GREEN/RED state per test in the commit message.

- [ ] **Step 3: Commit (WIP)**

Run:
```bash
git add tests/spadl/test_kloppy.py
git commit -m "WIP: TestKloppySportec real-fixture suite (5 tests)"
```

---

### Task 7: Add `TestKloppyMetrica` (RED, 5 tests, mirror of Sportec)

**Files:**
- Modify: `tests/spadl/test_kloppy.py`

- [ ] **Step 1: Add the TestKloppyMetrica class**

Open `tests/spadl/test_kloppy.py`. After the `TestKloppySportec` class (added in Task 6), insert:

```python
class TestKloppyMetrica:
    """End-to-end conversion tests for the Metrica Sports provider."""

    def test_convert_to_actions_basic(self, metrica_dataset):
        """convert_to_actions returns (DataFrame, ConversionReport) with the canonical
        SPADL schema for the Metrica fixture (3594 events)."""
        from silly_kicks.spadl import kloppy as spkloppy

        actions, report = spkloppy.convert_to_actions(metrica_dataset, game_id="metrica_smoke")

        assert isinstance(actions, pd.DataFrame)
        assert list(actions.columns) == list(KLOPPY_SPADL_COLUMNS.keys())
        for col, expected_dtype in KLOPPY_SPADL_COLUMNS.items():
            assert str(actions[col].dtype) == expected_dtype, f"{col}: got {actions[col].dtype}, want {expected_dtype}"
        assert len(actions) > 0
        assert report.provider == "Kloppy"
        assert report.total_events > 0
        assert report.total_actions == len(actions)

    def test_no_unrecognized_event_types(self, metrica_dataset):
        """All Metrica event types are covered by _MAPPED_EVENT_TYPES ∪ _EXCLUDED_EVENT_TYPES."""
        import warnings as warnings_mod

        from silly_kicks.spadl import kloppy as spkloppy

        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, report = spkloppy.convert_to_actions(metrica_dataset, game_id="metrica_unrec")

        assert report.unrecognized_counts == {}
        unrec_warnings = [str(w.message) for w in caught if "unrecognized event types" in str(w.message)]
        assert unrec_warnings == [], f"Got unrecognized-event warnings: {unrec_warnings}"

    def test_action_id_unique_and_zero_indexed(self, metrica_dataset):
        """action_id must be range(len(actions)) per game_id."""
        from silly_kicks.spadl import kloppy as spkloppy

        actions, _ = spkloppy.convert_to_actions(metrica_dataset, game_id="metrica_aid")
        assert actions["action_id"].tolist() == list(range(len(actions)))

    def test_preserve_native_with_dict_raw_event(self, metrica_dataset):
        """preserve_native surfaces a raw_event field as an extra column (Metrica raw_event is dict)."""
        from silly_kicks.spadl import kloppy as spkloppy

        first_event = next(iter(metrica_dataset.events))
        raw_keys = list(first_event.raw_event.keys())
        candidate = next((k for k in raw_keys if k not in KLOPPY_SPADL_COLUMNS), None)
        assert candidate is not None, f"Metrica raw_event has no non-schema-overlapping key. Keys: {raw_keys}"

        actions, _ = spkloppy.convert_to_actions(
            metrica_dataset, game_id="metrica_pn", preserve_native=[candidate]
        )
        assert candidate in actions.columns
        assert actions[candidate].notna().any()

    def test_input_dataset_not_mutated(self, metrica_dataset):
        """convert_to_actions must not mutate the input dataset's metadata or event count."""
        from silly_kicks.spadl import kloppy as spkloppy

        before_provider = metrica_dataset.metadata.provider
        before_coord_sys_id = id(metrica_dataset.metadata.coordinate_system)
        before_event_count = len(metrica_dataset.events)

        _, _ = spkloppy.convert_to_actions(metrica_dataset, game_id="metrica_mut")

        assert metrica_dataset.metadata.provider == before_provider
        assert id(metrica_dataset.metadata.coordinate_system) == before_coord_sys_id
        assert len(metrica_dataset.events) == before_event_count
```

- [ ] **Step 2: Run the new tests and observe RED/GREEN state**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyMetrica -v
```
Expected: same situation as Sportec — most/all may already pass after Task 5's bug fix. Document actual state.

- [ ] **Step 3: Commit (WIP)**

Run:
```bash
git add tests/spadl/test_kloppy.py
git commit -m "WIP: TestKloppyMetrica real-fixture suite (5 tests)"
```

---

### Task 8: Add `TestKloppyConversionReport` parametrized over both providers (RED)

**Files:**
- Modify: `tests/spadl/test_kloppy.py`

**Background:** This test class asserts (a) no `unrecognized provider` warning fires when converting Sportec/Metrica — RED today because they're not whitelisted, and (b) the `ConversionReport`'s mapped/excluded/unrecognized count dicts are pairwise disjoint (a useful invariant catching future logic errors in the bucketing code).

**Deviation note:** the spec §5.1 listed test #8 as a single function parametrized over 3 providers (StatsBomb, Sportec, Metrica). The plan instead has two functions parametrized over 2 providers each (Sportec, Metrica). StatsBomb is dropped because no raw-events StatsBomb fixture exists in `tests/datasets/`; adding one would be scope creep. The disjointness test is added as a best-practice contract check beyond spec — net function count goes from spec's 13 to 14, parametrized cases from 16 to 17. Spec §10 verification checklist will be updated accordingly during implementation (or just expect 14/17 in the verification step).

- [ ] **Step 1: Add the TestKloppyConversionReport class**

Open `tests/spadl/test_kloppy.py`. After the `TestKloppyMetrica` class (added in Task 7), insert:

```python
class TestKloppyConversionReport:
    """ConversionReport shape and provider-whitelist warning behaviour."""

    @pytest.mark.parametrize("provider_name,fixture_name", [
        ("sportec", "sportec_dataset"),
        ("metrica", "metrica_dataset"),
    ])
    def test_no_unrecognized_provider_warning(self, request, provider_name, fixture_name):
        """A whitelisted provider must NOT trigger the 'not yet supported' warning."""
        import warnings as warnings_mod

        from silly_kicks.spadl import kloppy as spkloppy

        dataset = request.getfixturevalue(fixture_name)
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always")
            _, _ = spkloppy.convert_to_actions(dataset, game_id=f"{provider_name}_warn")

        provider_warnings = [
            str(w.message) for w in caught
            if "not yet supported" in str(w.message)
        ]
        assert provider_warnings == [], (
            f"{provider_name} should be whitelisted; got warning(s): {provider_warnings}"
        )

    @pytest.mark.parametrize("fixture_name", ["sportec_dataset", "metrica_dataset"])
    def test_report_count_dicts_disjoint(self, request, fixture_name):
        """ConversionReport mapped/excluded/unrecognized count dicts must be pairwise disjoint."""
        from silly_kicks.spadl import kloppy as spkloppy

        dataset = request.getfixturevalue(fixture_name)
        _, report = spkloppy.convert_to_actions(dataset, game_id="report_disjoint")

        mapped = set(report.mapped_counts.keys())
        excluded = set(report.excluded_counts.keys())
        unrecognized = set(report.unrecognized_counts.keys())
        assert mapped & excluded == set()
        assert mapped & unrecognized == set()
        assert excluded & unrecognized == set()
```

- [ ] **Step 2: Run the new tests and confirm RED on `test_no_unrecognized_provider_warning`**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyConversionReport -v
```
Expected:
- `test_no_unrecognized_provider_warning[sportec-sportec_dataset]`: **FAIL** (Sportec not whitelisted → warning fires)
- `test_no_unrecognized_provider_warning[metrica-metrica_dataset]`: **FAIL** (Metrica not whitelisted → warning fires)
- `test_report_count_dicts_disjoint[sportec_dataset]`: **PASS**
- `test_report_count_dicts_disjoint[metrica_dataset]`: **PASS**

- [ ] **Step 3: Commit (WIP)**

Run:
```bash
git add tests/spadl/test_kloppy.py
git commit -m "WIP: TestKloppyConversionReport parametrized over both providers"
```

---

### Task 9: Implement provider whitelist (GREEN)

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py:54-57`

- [ ] **Step 1: Add Sportec + Metrica to `_SUPPORTED_PROVIDERS`**

Open `silly_kicks/spadl/kloppy.py`. Locate the existing `_SUPPORTED_PROVIDERS` dict (lines 54-57):

```python
_SUPPORTED_PROVIDERS = {
    Provider.STATSBOMB: version.parse("3.15.0"),
    # Provider.OPTA: version.parse("3.15.0"),
}
```

Replace with:

```python
_SUPPORTED_PROVIDERS = {
    Provider.STATSBOMB: version.parse("3.15.0"),
    Provider.SPORTEC: version.parse("3.15.0"),
    Provider.METRICA: version.parse("3.15.0"),
    # Provider.OPTA: version.parse("3.15.0"),  # has its own dedicated converter in spadl/opta.py
}
```

- [ ] **Step 2: Run the previously-RED tests and confirm GREEN**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyConversionReport -v
```
Expected: all 4 parametrized cases PASS.

- [ ] **Step 3: Run the full kloppy suite to confirm no regression**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py -v
```
Expected: 19 PASS + 1 SKIP — `TestKloppyPreserveNative` (4 tests), `TestKloppyCoordinateSystemFix` (1), `TestKloppySportec` (5), `TestKloppyMetrica` (5), `TestKloppyConversionReport` (2 functions × 2 parametrized cases = 4), `TestKloppyE2E::test_placeholder` (1 skip).

- [ ] **Step 4: Commit (WIP)**

Run:
```bash
git add silly_kicks/spadl/kloppy.py
git commit -m "WIP: whitelist Sportec + Metrica in _SUPPORTED_PROVIDERS"
```

---

## Phase 4: Coordinate clamping — RED → GREEN

### Task 10: Write `TestKloppyCoordinateClamping::test_clamps_to_pitch_bounds` (RED)

**Files:**
- Modify: `tests/spadl/test_kloppy.py`

**Background:** Empirical probe showed Metrica produces `start_x ∈ [-1.62, 104.63]`, `start_y ∈ [-2.03, 70.31]` — slightly off-pitch in source. The kloppy converter is the lone outlier among silly-kicks converters that does not clamp output coords.

- [ ] **Step 1: Add the TestKloppyCoordinateClamping class**

Open `tests/spadl/test_kloppy.py`. After the `TestKloppyConversionReport` class (added in Task 8), insert:

```python
class TestKloppyCoordinateClamping:
    """Output coords must be clamped to [0, field_length] x [0, field_width]
    (105 x 68 m), matching the convention established by StatsBomb / Wyscout / Opta
    converters."""

    @pytest.mark.parametrize("fixture_name", ["sportec_dataset", "metrica_dataset"])
    def test_clamps_to_pitch_bounds(self, request, fixture_name):
        """All start_x, start_y, end_x, end_y values must lie in the SPADL frame."""
        from silly_kicks.spadl import kloppy as spkloppy

        dataset = request.getfixturevalue(fixture_name)
        actions, _ = spkloppy.convert_to_actions(dataset, game_id="clamp_test")

        L = spadlconfig.field_length  # 105.0
        W = spadlconfig.field_width  # 68.0

        for col, upper in [("start_x", L), ("end_x", L), ("start_y", W), ("end_y", W)]:
            col_min = actions[col].min()
            col_max = actions[col].max()
            assert col_min >= 0, f"{col}.min()={col_min} (must be >= 0)"
            assert col_max <= upper, f"{col}.max()={col_max} (must be <= {upper})"
```

- [ ] **Step 2: Run the new test and confirm RED on Metrica**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyCoordinateClamping -v
```
Expected:
- `test_clamps_to_pitch_bounds[sportec_dataset]`: PASS or FAIL — empirical probe showed Sportec extends to x=107.31 due to source pitch dimension scaling. If Sportec FAILs too, that's expected.
- `test_clamps_to_pitch_bounds[metrica_dataset]`: **FAIL** (`start_x.min()=-1.62 ...`)

- [ ] **Step 3: Commit (WIP)**

Run:
```bash
git add tests/spadl/test_kloppy.py
git commit -m "WIP: RED test for kloppy converter coord clamping"
```

---

### Task 11: Implement coordinate clamping in `convert_to_actions` (GREEN)

**Files:**
- Modify: `silly_kicks/spadl/kloppy.py:208-211`

- [ ] **Step 1: Add clamping immediately before `_finalize_output`**

Open `silly_kicks/spadl/kloppy.py`. Locate the section around lines 208-211 in `convert_to_actions`:

```python
    df_actions["action_id"] = range(len(df_actions))
    df_actions = _add_dribbles(df_actions)

    df_actions = _finalize_output(df_actions, KLOPPY_SPADL_COLUMNS, extra_columns=preserve_native)
```

Insert four `.clip()` lines between `_add_dribbles` and `_finalize_output`:

```python
    df_actions["action_id"] = range(len(df_actions))
    df_actions = _add_dribbles(df_actions)

    # Clamp output coords to the SPADL pitch frame, matching the convention
    # established by the StatsBomb, Wyscout, and Opta converters. Source data
    # may emit slightly off-pitch coordinates (recording-noise tolerance);
    # downstream silly-kicks consumers (VAEP, xT, possession, GK enrichments)
    # assume bounded coords.
    df_actions["start_x"] = df_actions["start_x"].clip(0, spadlconfig.field_length)
    df_actions["start_y"] = df_actions["start_y"].clip(0, spadlconfig.field_width)
    df_actions["end_x"] = df_actions["end_x"].clip(0, spadlconfig.field_length)
    df_actions["end_y"] = df_actions["end_y"].clip(0, spadlconfig.field_width)

    df_actions = _finalize_output(df_actions, KLOPPY_SPADL_COLUMNS, extra_columns=preserve_native)
```

- [ ] **Step 2: Run the previously-RED clamping test and confirm GREEN**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py::TestKloppyCoordinateClamping -v
```
Expected: both parametrized cases PASS.

- [ ] **Step 3: Run the full kloppy suite to confirm no regression**

Run:
```bash
uv run pytest tests/spadl/test_kloppy.py -v
```
Expected: all tests PASS.

- [ ] **Step 4: Run the full silly-kicks test suite (excl. e2e) to confirm no cross-converter regression**

Run with `run_in_background=true` because full suite may take >30s:
```bash
uv run pytest tests/ -m "not e2e" -v --tb=short
```
Expected: all tests PASS, including StatsBomb / Wyscout / Opta / VAEP / xT / atomic suites.

- [ ] **Step 5: Commit (WIP)**

Run:
```bash
git add silly_kicks/spadl/kloppy.py
git commit -m "WIP: clamp kloppy output coords to SPADL pitch frame"
```

---

## Phase 5: Documentation + version bump

### Task 12: Update `CHANGELOG.md` with the 1.6.0 entry

**Files:**
- Modify: `CHANGELOG.md`

- [ ] **Step 1: Insert the 1.6.0 section above the existing 1.5.0 section**

Open `CHANGELOG.md`. Find the line:
```
## [1.5.0] — 2026-04-27
```

Insert the following block immediately above that line (preserving everything below):

```markdown
## [1.6.0] — 2026-04-28

### Added
- **Kloppy converter: Sportec and Metrica support.** `Provider.SPORTEC`
  (Sportec Solutions / IDSSE Bundesliga event format) and `Provider.METRICA`
  (Metrica Sports) are now first-class whitelisted providers in
  `silly_kicks.spadl.kloppy.convert_to_actions`. Empirical verification on
  real fixture data confirms zero new event-type mappings are required —
  both providers' kloppy serializers emit only event types already covered
  by the existing `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES` sets.
  `preserve_native` works transparently for both (their `raw_event` is a
  `dict`).
- Real-fixture end-to-end test suites for Sportec and Metrica under
  `tests/spadl/test_kloppy.py`, plus a parametrized coordinate-clamping
  test and a per-provider `ConversionReport` shape test. Test fixtures
  vendored from kloppy's BSD-3-Clause-licensed test files into
  `tests/datasets/kloppy/`.

### Fixed
- **`_SoccerActionCoordinateSystem` was unusable on real datasets.** The
  class definition omitted `__init__`, but `convert_to_actions()`
  instantiated it with `pitch_length=` / `pitch_width=` kwargs. On any
  kloppy version with the current `CoordinateSystem` ABC signature
  (kloppy 3.15+), this raised `TypeError` the moment a real
  `EventDataset` reached `dataset.transform()`. Latent since 1.0.0
  because pre-existing `tests/spadl/test_kloppy.py` was pure mocks
  that never reached the transform call. Affected **all** kloppy-based
  conversion including the previously-whitelisted StatsBomb path.

### Changed
- **Kloppy converter now clamps output coordinates to
  `[0, field_length] × [0, field_width]` (105 × 68 m).** This aligns the
  kloppy converter with the established silly-kicks convention — StatsBomb
  / Wyscout / Opta converters all clamp; kloppy was the lone outlier.
  Empirically Metrica events emit slight off-pitch coords (observed
  `x ∈ [-1.62, 104.63]` on the sample game) within source-recording-noise
  tolerance. Downstream consumers depending on raw off-pitch coordinates
  from the kloppy path specifically should re-verify (no such consumer
  documented).

```

- [ ] **Step 2: Commit (WIP)**

Run:
```bash
git add CHANGELOG.md
git commit -m "WIP: CHANGELOG 1.6.0 entry"
```

---

### Task 13: Update `README.md` provider-coverage list

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Locate the provider-coverage section**

Open `README.md`. Search for the section listing supported providers (likely under a header like "Supported providers", "Converters", or in the Features list).

- [ ] **Step 2: Update Kloppy mention to enumerate providers**

Find any text resembling:
```
- Kloppy
```
or
```
- StatsBomb / Opta / Wyscout / Kloppy converters
```

Update so the Kloppy line specifies which providers route through it. Example replacement:
```
- **Kloppy gateway**: StatsBomb, Sportec (IDSSE Bundesliga), and Metrica Sports event data via the kloppy parsing layer
```

If the README doesn't have a provider list at all, **skip this task** and document the skip in the commit message — the converter is discoverable via docs/code, no README change is mandatory. Re-confirm with the user.

- [ ] **Step 3: Commit (WIP)**

Run (only if a README change was made):
```bash
git add README.md
git commit -m "WIP: README provider-coverage update for Sportec/Metrica"
```

---

### Task 14: Bump version in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:7`

- [ ] **Step 1: Bump version 1.5.0 → 1.6.0**

Open `pyproject.toml`. Line 7:
```
version = "1.5.0"
```

Replace with:
```
version = "1.6.0"
```

- [ ] **Step 2: Commit (WIP)**

Run:
```bash
git add pyproject.toml
git commit -m "WIP: bump version 1.5.0 -> 1.6.0"
```

---

## Phase 6: Pre-PR gates

### Task 15: Run ruff check

- [ ] **Step 1: Run lint**

Run:
```bash
uv run ruff check .
```
Expected: `All checks passed!`. If violations appear, fix them (typically import ordering or unused imports in the new test file). Re-run until green.

- [ ] **Step 2: If fixes were needed, commit (WIP)**

Run only if ruff produced edits:
```bash
git add -u
git commit -m "WIP: ruff fixes"
```

---

### Task 16: Run ruff format check

- [ ] **Step 1: Verify formatting**

Run:
```bash
uv run ruff format --check .
```
Expected: `X files already formatted`. If files need reformatting, run `uv run ruff format .` to fix, then commit.

- [ ] **Step 2: If reformat was needed, commit (WIP)**

Run only if ruff produced edits:
```bash
git add -u
git commit -m "WIP: ruff format"
```

---

### Task 17: Run pyright

- [ ] **Step 1: Verify exact pandas-stubs pin matches CI**

Per the `feedback_ci_cross_version` memory, local Python 3.14 vs CI Python 3.10-3.12 can produce divergent pyright output. Before running pyright, confirm `pandas-stubs>=2.2.0` is installed in the dev env:

```bash
uv pip list | grep -i pandas-stubs
```
Expected: `pandas-stubs    2.2.x` or higher.

If absent, install with:
```bash
uv sync --extra dev
```

- [ ] **Step 2: Run pyright**

Run:
```bash
uv run pyright
```
Expected: `0 errors, 0 warnings, 0 informations`. If errors surface, fix at the source (likely in the new test code's import paths or type annotations).

- [ ] **Step 3: If fixes were needed, commit (WIP)**

Run only if fixes:
```bash
git add -u
git commit -m "WIP: pyright fixes"
```

---

### Task 18: Run full test suite (non-e2e)

- [ ] **Step 1: Run all tests (background — may exceed 30s)**

Run with `run_in_background=true`:
```bash
uv run pytest tests/ -m "not e2e" -v --tb=short
```

While running, you'll be notified on completion. Do NOT poll. When complete, read the output file to verify:
- All previously-passing tests still PASS
- All new kloppy tests PASS
- No new SKIPs (other than the existing e2e skip stub)

Expected end-of-output: `XXX passed, Y skipped in NN.NNs`.

- [ ] **Step 2: If any test fails, diagnose and fix**

Use the systematic-debugging skill. Do NOT proceed to PR until all green.

---

## Phase 7: Squash and PR

### Task 19: Squash all WIP commits to a single release commit

**Files:**
- Local git history (no file changes)

- [ ] **Step 1: Confirm current commit count on the branch**

Run:
```bash
git log --oneline main..HEAD
```
Expected: ~12–17 WIP commits (one per Task 1-14 with potential extras from Tasks 15/16/17 if ruff/pyright needed fixes; Task 13 README commit is conditional).

- [ ] **Step 2: Soft reset to main and re-stage everything as one commit**

Run:
```bash
git reset --soft main
git status --short
```
Expected: all changes appear staged (`M` or `A` markers, no `??`).

- [ ] **Step 3: Create the single squash commit**

Run via heredoc to preserve formatting:
```bash
git commit -m "$(cat <<'EOF'
feat(spadl): kloppy Sportec + Metrica + _SoccerActionCoordinateSystem fix — silly-kicks 1.6.0

Adds Provider.SPORTEC and Provider.METRICA to the kloppy converter's
_SUPPORTED_PROVIDERS whitelist. Empirical verification on real kloppy
fixture data confirms zero new event-type mappings are required — both
providers' serializers emit only event types already covered by
_MAPPED_EVENT_TYPES ∪ _EXCLUDED_EVENT_TYPES.

Fixes a latent bug in _SoccerActionCoordinateSystem (no __init__, but
convert_to_actions instantiates it with pitch_length=/pitch_width=
kwargs) that has prevented the kloppy converter from running end-to-end
on any real EventDataset since silly-kicks 1.0.0. The pre-existing
mock-based tests never exercised dataset.transform().

Aligns the kloppy converter with the established silly-kicks convention
of clamping output coords to [0, field_length] × [0, field_width],
matching StatsBomb / Wyscout / Opta. Metrica events emit slight off-pitch
coords (x ∈ [-1.62, 104.63] empirically) within source-recording-noise
tolerance.

13 new test functions (16 parametrized cases) including a regression
test for the _SoccerActionCoordinateSystem bug, real-fixture end-to-end
suites for Sportec and Metrica vendored from kloppy's BSD-3-licensed
test files, and a per-provider coordinate-clamping test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Verify single commit on the branch**

Run:
```bash
git log --oneline main..HEAD
```
Expected: exactly one commit on the branch, with the long message above.

- [ ] **Step 5: Final dry-run of the test suite from the squashed state**

Run with `run_in_background=true`:
```bash
uv run pytest tests/ -m "not e2e" --tb=short
```
Expected: same green output as Task 18 step 1.

---

### Task 20: Push branch and open PR `[USER APPROVAL GATE]`

**Files:**
- Remote: `origin/feat/kloppy-sportec-metrica`
- GitHub PR

- [ ] **Step 1: Show the user the pending push**

Stop and present:
- Branch: `feat/kloppy-sportec-metrica`
- Commit count: 1 (squashed)
- Diff summary: `git diff --stat main..HEAD`
- Asking for approval to push.

**Wait for explicit user approval before proceeding.**

- [ ] **Step 2: Push branch (only after approval)**

Run:
```bash
git push -u origin feat/kloppy-sportec-metrica
```

- [ ] **Step 3: Open PR (only after Step 2 succeeds)**

Run via heredoc:
```bash
gh pr create --title "feat(spadl): kloppy Sportec + Metrica + _SoccerActionCoordinateSystem fix — silly-kicks 1.6.0" --body "$(cat <<'EOF'
## Summary

- Whitelist `Provider.SPORTEC` (IDSSE Bundesliga) and `Provider.METRICA` in the kloppy SPADL converter
- Fix latent `TypeError` in `_SoccerActionCoordinateSystem` that has prevented all kloppy-based conversion from running end-to-end on real datasets since silly-kicks 1.0.0 (latent because pre-existing tests were pure mocks)
- Align kloppy output coordinate clamping with the StatsBomb / Wyscout / Opta convention

## Spec & plan

- `docs/superpowers/specs/2026-04-28-kloppy-sportec-metrica-design.md`
- `docs/superpowers/plans/2026-04-28-kloppy-sportec-metrica.md`

## Test plan

- [x] 13 new test functions added (16 parametrized cases) under `tests/spadl/test_kloppy.py`
- [x] Real-fixture end-to-end coverage for Sportec + Metrica via vendored kloppy test files (BSD-3-Clause)
- [x] Regression test for the `_SoccerActionCoordinateSystem` bug — RED-fails on `main`
- [x] Coordinate clamping test parametrized over both providers — RED-fails on `main` for Metrica
- [x] All previously-passing tests in `tests/` still GREEN
- [x] `ruff check`, `ruff format --check`, `pyright` all green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```
Expected: PR URL printed.

- [ ] **Step 4: Report PR URL to user and wait**

---

### Task 21: Merge after CI green `[USER APPROVAL GATE]`

- [ ] **Step 1: Wait for CI**

Use `gh pr checks <PR_NUMBER>` to poll. Or use `gh pr view <PR_NUMBER> --json statusCheckRollup` for structured status.

- [ ] **Step 2: Present final status to user**

Stop and present:
- CI status (all green)
- Asking for approval to admin-squash-merge.

**Wait for explicit user approval before proceeding.**

- [ ] **Step 3: Merge (only after approval)**

Run:
```bash
gh pr merge --admin --squash --delete-branch
```

---

### Task 22: Tag and trigger PyPI publish `[USER APPROVAL GATE]`

- [ ] **Step 1: Sync local main**

Run:
```bash
git checkout main && git pull
```
Expected: `Updating ...` ending at the new squash commit on main.

- [ ] **Step 2: Verify the squash commit landed**

Run:
```bash
git log --oneline -2
```
Expected:
```
<new-sha> feat(spadl): kloppy Sportec + Metrica + _SoccerActionCoordinateSystem fix — silly-kicks 1.6.0 (#NN)
07211c6 feat(atomic): SPADL parity for the 1.1.0 → 1.4.0 helper family — silly-kicks 1.5.0 (#10)
```

- [ ] **Step 3: Stop and ask user for approval to tag + push**

The tag push triggers PyPI auto-publish — non-reversible. Stop and present:
- Tag: `v1.6.0`
- Will trigger PyPI workflow on push.
- Asking for explicit approval.

**Wait for explicit user approval before proceeding.**

- [ ] **Step 4: Create and push tag (only after approval)**

Run:
```bash
git tag v1.6.0 && git push origin v1.6.0
```
Expected: `* [new tag] v1.6.0 -> v1.6.0`.

- [ ] **Step 5: Verify PyPI workflow fired**

Run:
```bash
gh run list --workflow="publish.yml" --limit 3
```
(Workflow name may differ; check `.github/workflows/` for the publish workflow filename.)

Wait for the workflow to complete (typically 1-3 min). Verify success via:
```bash
gh run view <RUN_ID> --log
```

- [ ] **Step 6: Confirm PyPI release**

Run:
```bash
uv pip index versions silly-kicks
```
Expected: `silly-kicks==1.6.0` listed.

---

## Done

- [ ] Final verification:
  - [ ] PR squash-merged
  - [ ] Branch deleted
  - [ ] Tag `v1.6.0` pushed
  - [ ] PyPI shows `silly-kicks==1.6.0`
  - [ ] CHANGELOG, README, pyproject all reflect 1.6.0
