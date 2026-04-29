# Kloppy converter: Sportec + Metrica support and `_SoccerActionCoordinateSystem` fix

**Status:** Approved (design)
**Target release:** silly-kicks 1.6.0
**Author:** Claude Opus 4.7 (1M)
**Date:** 2026-04-28

---

## 1. Problem

`silly_kicks/spadl/kloppy.py` whitelists only `Provider.STATSBOMB`. Downstream
consumers cannot run kloppy-based SPADL conversion against any other provider,
even though kloppy upstream supports many more — relevant near-term: Sportec
(IDSSE Bundesliga event format) and Metrica Sports.

Empirical probe (kloppy 3.18, with real fixture data fetched from kloppy's
public test files) confirms two facts that scope this work:

1. **Both providers' event types are already covered** by the existing
   `_MAPPED_EVENT_TYPES` ∪ `_EXCLUDED_EVENT_TYPES` sets — no new event-handler
   functions or mapping changes are required.
2. **A pre-existing bug blocks the kloppy converter from running end-to-end on
   any provider, including the already-whitelisted StatsBomb path:**
   `_SoccerActionCoordinateSystem` (lines 241–262) is defined as a class with
   no `__init__`, but `convert_to_actions()` instantiates it as
   `_SoccerActionCoordinateSystem(pitch_length=..., pitch_width=...)` (line 168).
   Kloppy 3.15+ `CoordinateSystem` is an ABC with no kwarg-accepting init, so
   this raises `TypeError: _SoccerActionCoordinateSystem() takes no arguments`.
   The bug has been latent since 1.0.0 because `tests/spadl/test_kloppy.py`
   exclusively uses `MagicMock` datasets that never reach `dataset.transform()`.

## 2. Goals

1. Whitelist `Provider.SPORTEC` and `Provider.METRICA` in
   `_SUPPORTED_PROVIDERS`.
2. Fix `_SoccerActionCoordinateSystem` so it accepts and exposes `pitch_length`
   / `pitch_width`.
3. Add real-fixture end-to-end tests for **all three** kloppy providers
   (StatsBomb, Sportec, Metrica) so the bug cannot regress and so future
   kloppy version drift is caught early.
4. Align kloppy converter with the established silly-kicks convention of
   clamping output coordinates to `[0, field_length] × [0, field_width]`.

## 3. Non-goals

1. No new event-type handlers or mapping changes — the empirical probe shows
   none are needed for either new provider.
2. No `clip_to_pitch()` post-conversion utility — clamping happens inline in
   the kloppy converter, matching how StatsBomb / Wyscout / Opta handle it.
   A standalone helper could be added in a future PR if a use case emerges.
3. No changes to `convert_to_actions()` signature, `preserve_native` plumbing,
   or any of the `_parse_*` event handlers.
4. No bump to the `kloppy>=3.15.0` floor in `pyproject.toml`. The probe
   covered kloppy 3.18 but the API surface we depend on (`Provider`,
   `EventType`, `CoordinateSystem` ABC) is stable across 3.15 → 3.18.
5. No luxury-lakehouse-side changes (out of scope per brief).

## 4. Architecture

### 4.1 Bug fix — `_SoccerActionCoordinateSystem`

Replace the current property-only class with one that explicitly accepts
`pitch_length` / `pitch_width` via `__init__` and overrides the inherited
`pitch_length` / `pitch_width` properties to return the stored values. The
existing `pitch_dimensions` property continues to construct a
`MetricPitchDimensions` from these values plus `spadlconfig.field_length` /
`field_width`.

```python
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
    def pitch_length(self) -> float:
        return self._pitch_length

    @property
    def pitch_width(self) -> float:
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

### 4.2 Whitelist

```python
_SUPPORTED_PROVIDERS = {
    Provider.STATSBOMB: version.parse("3.15.0"),
    Provider.SPORTEC:   version.parse("3.15.0"),
    Provider.METRICA:   version.parse("3.15.0"),
}
```

The `Provider.OPTA` commented-out entry stays commented — Opta has its own
dedicated `silly_kicks/spadl/opta.py` converter; routing it through kloppy
would be a separate decision.

### 4.3 Coordinate clamping (cross-converter consistency)

Insert clamping immediately before `_finalize_output()` in
`convert_to_actions()`:

```python
df_actions["start_x"] = df_actions["start_x"].clip(0, spadlconfig.field_length)
df_actions["start_y"] = df_actions["start_y"].clip(0, spadlconfig.field_width)
df_actions["end_x"]   = df_actions["end_x"].clip(0, spadlconfig.field_length)
df_actions["end_y"]   = df_actions["end_y"].clip(0, spadlconfig.field_width)
```

**Rationale:** All three sibling converters clamp output coordinates to the
SPADL frame:
- `silly_kicks/spadl/statsbomb.py:380-381` — `np.clip(coords, 0, field_length)`
- `silly_kicks/spadl/_wyscout_events.py:426,430,431,435` — `.clip(0, field_*)`
- `silly_kicks/spadl/opta.py:142,144` — `clip(0, 100)` in source-coord space

The kloppy converter is the lone outlier. Empirically, Metrica events emit
slight off-pitch coords (observed range `x ∈ [-1.62, 104.63]`,
`y ∈ [-2.03, 70.31]` on the sample game) — within source-recording-noise
tolerance, but enough to violate the SPADL bounding contract that downstream
silly-kicks consumers (VAEP, xT, possession reconstruction, GK role detection)
implicitly assume. Aligning the kloppy converter removes a footgun.

### 4.4 Public API surface

Unchanged. `convert_to_actions(dataset, game_id=None, *, preserve_native=None)`
keeps the same signature, return type, and `ConversionReport` semantics.
`preserve_native` was already verified to work for both new providers
(`raw_event` is a `dict` for Sportec and Metrica events).

## 5. Test plan

### 5.1 TDD order — RED tests written first

Each test must fail against `main` before any production-code change is made.
After all RED tests are committed (in a temporary commit, then squashed), the
production code changes are added incrementally and each test moves to GREEN
in the listed order.

| # | Test | RED-failure mode on `main` |
|---|---|---|
| 1 | `TestKloppyCoordinateSystemFix::test_real_dataset_transform_does_not_typeerror` (regression) | `TypeError: _SoccerActionCoordinateSystem() takes no arguments` |
| 2 | `TestKloppySportec::test_convert_to_actions_basic` (rows, schema, non-empty) | Provider not whitelisted (warning) + bug #1 |
| 3 | `TestKloppySportec::test_no_unrecognized_event_types` | depends on production-step 4 (bug fix) + 5 (whitelist) |
| 4 | `TestKloppySportec::test_action_id_unique_and_zero_indexed` | depends on steps 4 + 5 |
| 5 | `TestKloppySportec::test_preserve_native_with_dict_raw_event` | depends on steps 4 + 5 |
| 6 | `TestKloppySportec::test_input_dataset_not_mutated` | depends on steps 4 + 5 |
| 7 | `TestKloppyMetrica::*` (mirror of 2–6 for Metrica, 5 tests) | depends on steps 4 + 5 |
| 8 | `TestKloppyConversionReport::test_report_shape_per_provider` (parametrized over 3 providers) | depends on steps 4 + 5 |
| 9 | `TestKloppyCoordinateClamping::test_clamps_to_pitch_bounds` (parametrized over Sportec + Metrica) | Metrica produces `start_x = -1.62` on `main`; turns GREEN at production step 6 |

Total: **13 new test functions** (1 + 5 + 5 + 1 + 1), **16 parametrized cases** when expanded.

### 5.2 Test-class structure

```
tests/spadl/test_kloppy.py
├── TestKloppyPreserveNative           (existing — kept as-is, mock-based)
├── TestKloppyCoordinateSystemFix      (NEW — regression test #1, mock dataset)
├── TestKloppySportec                  (NEW — real-fixture suite)
│   ├── test_convert_to_actions_basic
│   ├── test_no_unrecognized_event_types
│   ├── test_action_id_unique_and_zero_indexed
│   ├── test_preserve_native_with_dict_raw_event
│   └── test_input_dataset_not_mutated
├── TestKloppyMetrica                  (NEW — real-fixture suite, mirrors Sportec)
│   └── (same five tests)
├── TestKloppyConversionReport         (NEW — parametrized over all three providers)
│   └── test_report_shape_per_provider
├── TestKloppyCoordinateClamping       (NEW — parametrized over Sportec, Metrica)
│   └── test_clamps_to_pitch_bounds
└── TestKloppyE2E                      (existing — placeholder skip stub stays for now;
                                        new real-fixture suites supersede the need)
```

### 5.3 Test fixtures

Vendor four files into `tests/datasets/kloppy/` with a `README.md` documenting
provenance and licenses:

| File | Size | Origin |
|---|---|---|
| `sportec_events.xml` | 15 KB | kloppy `tests/files/sportec_events.xml` (BSD-3-Clause) |
| `sportec_meta.xml` | 12 KB | kloppy `tests/files/sportec_meta.xml` (BSD-3-Clause) |
| `metrica_events.json` | 1.7 MB | kloppy `tests/files/metrica_events.json` (originally from `metrica-sports/sample-data` Sample_Game_2, CC-BY-NC-4.0) |
| `epts_metrica_metadata.xml` | 34 KB | kloppy `tests/files/epts_metrica_metadata.xml` (BSD-3-Clause) |

**Total tests/ growth:** ~1.7 MB (final ~3.2 MB). Acceptable for a Python
library — comparable projects (kloppy itself, pandas test fixtures) routinely
ship multi-MB committed test data. The full Metrica file was chosen over a
subset because empirical analysis showed first-100 misses SHOT events
entirely (0 of 20) and first-200 catches only 1 of 20 — a subset trades
short-term repo size for long-term "missing-event-in-subset" debugging cost.

`tests/datasets/kloppy/README.md` will document:
- Each fixture's origin URL (`https://github.com/PySport/kloppy/blob/master/...`)
- Each fixture's license (BSD-3 for kloppy synthetic test data;
  CC-BY-NC-4.0 for Metrica Sample_Game_2)
- Why the EPTS metadata is paired with the events JSON despite being from a
  different match (matches kloppy's own fixture pairing — preserves the
  upstream test contract; documented as a known limitation in kloppy's own
  test file)

### 5.4 Test fixture loader pattern

Add a session-scoped pytest fixture in `tests/spadl/test_kloppy.py`:

```python
@pytest.fixture(scope="module")
def kloppy_fixtures_dir() -> Path:
    return Path(__file__).parent.parent / "datasets" / "kloppy"

@pytest.fixture(scope="module")
def sportec_dataset(kloppy_fixtures_dir):
    from kloppy import sportec
    return sportec.load_event(
        event_data=kloppy_fixtures_dir / "sportec_events.xml",
        meta_data=kloppy_fixtures_dir / "sportec_meta.xml",
    )

@pytest.fixture(scope="module")
def metrica_dataset(kloppy_fixtures_dir):
    from kloppy import metrica
    return metrica.load_event(
        event_data=kloppy_fixtures_dir / "metrica_events.json",
        meta_data=kloppy_fixtures_dir / "epts_metrica_metadata.xml",
    )
```

The dataset fixtures are `scope="module"` so kloppy parsing (the slow part)
happens once per provider, not per test. Each test then runs
`convert_to_actions()` on the parsed dataset (fast).

### 5.5 Specific assertion details per test

- **`test_convert_to_actions_basic`** — assert `(df, report)` shape:
  `df.columns` matches `KLOPPY_SPADL_COLUMNS.keys()` exactly; `len(df) > 0`;
  `report.total_events > 0`; `report.unrecognized_counts == {}`.
- **`test_no_unrecognized_event_types`** — assert `report.unrecognized_counts == {}`
  AND no `UserWarning` matching `"unrecognized event types"` is emitted.
  Catches future kloppy versions that introduce new event types.
- **`test_action_id_unique_and_zero_indexed`** — assert
  `df["action_id"].tolist() == list(range(len(df)))` per `game_id`.
- **`test_preserve_native_with_dict_raw_event`** — pick a raw_event key that
  exists on the first event of the loaded dataset (probe at fixture-load time
  via `next(iter(ds.events)).raw_event.keys()`) and assert it surfaces as a
  column with non-null values for the first row.
- **`test_input_dataset_not_mutated`** — capture
  `id(dataset.metadata.coordinate_system)` before the call, assert unchanged
  after. (kloppy's `dataset.transform()` returns a new dataset, but verify.)
- **`test_report_shape_per_provider`** — parametrized over all three providers
  (StatsBomb fixture loaded via existing test infrastructure if available, or
  marked skipped if not); assert `report.provider == "Kloppy"` and the three
  count dicts have non-overlapping keys.
- **`test_clamps_to_pitch_bounds`** — parametrized over Sportec and Metrica;
  assert `0 <= df["start_x"].min()` and `df["start_x"].max() <= field_length`
  and same for start_y / end_x / end_y. RED-fails on Metrica today
  (`start_x.min() ≈ -1.62`).

### 5.6 Cross-cutting validation

Run `uv run pytest tests/spadl/test_kloppy.py -v` after each TDD step and
verify the expected red→green transition.

Run the full unaffected test suite at the end to confirm no regression in
StatsBomb / Wyscout / Opta paths or in the GK / possession enrichment helpers
(which all assume bounded coordinates):
```bash
uv run pytest tests/ -m "not e2e" -v
```

## 6. Implementation order

1. **Branch:** `feat/kloppy-sportec-metrica` from `main` at commit `07211c6`.
2. **Vendor fixtures:** copy 4 files into `tests/datasets/kloppy/`, write
   `README.md` with provenance + licenses. Verify they load via
   `kloppy.sportec.load_event` / `kloppy.metrica.load_event`.
3. **Write all RED tests** in `tests/spadl/test_kloppy.py`. Run
   `uv run pytest tests/spadl/test_kloppy.py -v` and confirm the expected
   RED failures (each test fails with the documented mode in §5.1).
4. **Production change 1: bug fix.** Replace `_SoccerActionCoordinateSystem`
   per §4.1. Re-run tests — test #1 turns GREEN.
5. **Production change 2: whitelist.** Add Sportec + Metrica per §4.2.
   Re-run — tests #2 through #8 turn GREEN. (Some Sportec / Metrica tests
   may pass incidentally after step 4 alone; verify all do after step 5.)
6. **Production change 3: clamping.** Add the four `clip()` calls per §4.3.
   Re-run — test #9 turns GREEN. Confirm the existing StatsBomb / Wyscout /
   Opta tests still pass (no regression).
7. **Lint / type / format gates:**
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   uv run pyright
   uv run pytest tests/ -m "not e2e"
   ```
   All must pass green before the next step.
8. **README + CHANGELOG.** Update README provider-coverage list. Add CHANGELOG
   entry under `## [1.6.0]` per §7.
9. **Version bump:** `pyproject.toml` `version = "1.5.0"` → `"1.6.0"`.
10. **Squash to single commit.** Verify `git log main..HEAD --oneline` shows
    one commit. Push and open PR. Wait for explicit user approval before
    each subsequent step (push, PR open, merge, tag-push).

## 7. CHANGELOG entry

```markdown
## [1.6.0] — 2026-04-XX

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
  from the kloppy path specifically should re-verify (none documented).
```

## 8. Release ritual (post-merge)

User-approval-gated at each step:

1. Open PR with `gh pr create`.
2. Wait for CI green.
3. After user approval: `gh pr merge --admin --squash --delete-branch`.
4. Locally: `git checkout main && git pull`.
5. After user approval: `git tag v1.6.0 && git push origin v1.6.0` —
   PyPI publish workflow auto-fires on tag push.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Vendored Metrica fixture license (CC-BY-NC) restricts commercial use | Document explicitly in `tests/datasets/kloppy/README.md`; the fixture is **test-data only**, not redistributed in the wheel (excluded by `[tool.hatch.build.targets.wheel] packages = ["silly_kicks"]` which already excludes `tests/`). Verify wheel build excludes the fixture. |
| Coord clamping could mask a real upstream bug (e.g. kloppy producing wildly off-pitch coords because of a coordinate-system misconfiguration) | The empirical magnitude is small (≤2m). For larger off-pitch coords, the conversion still proceeds (clamped to the boundary) but operators can check raw kloppy `dataset.events[i].coordinates` for diagnosis. |
| Future kloppy version adds new event types not in `_MAPPED_EVENT_TYPES` / `_EXCLUDED_EVENT_TYPES` | `test_no_unrecognized_event_types` will RED-fail on the new fixture — catches the drift. The existing `_event_type_counts` warning at conversion time also surfaces it to users. |
| Kloppy upstream changes the `CoordinateSystem` ABC signature again | Test #1 (`test_kloppy_real_dataset_transform_does_not_typeerror`) is a regression test that exercises the full transform. Any breaking ABC change surfaces immediately on `pip install kloppy` upgrades in CI. |
| Repo size growth (~1.7 MB) | One-time cost; git delta-compression, infrequent fixture changes. Comparable to pandas / numpy test data sizes. |

## 10. Verification checklist (post-implementation)

- [ ] All 13 new test functions GREEN (16 parametrized cases when expanded)
- [ ] All existing `tests/` tests still GREEN (no regression in StatsBomb /
  Wyscout / Opta / VAEP / xT / atomic / utils paths)
- [ ] `uv run ruff check .` → 0 violations
- [ ] `uv run ruff format --check .` → no diff
- [ ] `uv run pyright` → 0 errors (per `feedback_ci_cross_version` memory:
  pin `pandas-stubs>=2.2.0` matches CI; verify locally)
- [ ] `pyproject.toml` version is `1.6.0`
- [ ] `CHANGELOG.md` has the `## [1.6.0]` entry per §7
- [ ] `README.md` provider-coverage list mentions Sportec and Metrica
- [ ] `tests/datasets/kloppy/README.md` exists with provenance + license info
- [ ] Single commit on the branch (squashed) with message
  `feat(spadl): kloppy Sportec + Metrica + _SoccerActionCoordinateSystem fix — silly-kicks 1.6.0`
- [ ] Wheel build excludes `tests/datasets/kloppy/` (`uv build` and inspect
  `.whl` contents — `tests/` should not appear)
