# PFF to Gradient Sports Rename — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename all `"pff"` references to `"gradientsports"` across the codebase to reflect the PFF FC → Gradient Sports rebrand.

**Architecture:** The rename is a mechanical search-and-replace across three layers: (1) runtime Python modules and schema constants, (2) test files and fixtures, (3) documentation. The internal provider key becomes `"gradientsports"` (lowercase, collapsed — matching `"metrica"`, `"skillcorner"`, `"statsbomb"` convention). This is a breaking change (approved by the sole consumer).

**Tech Stack:** Python, pandas, pytest, ruff, pyright

---

## File Structure

### Files renamed (7 moves)

| Old Path | New Path |
|----------|----------|
| `silly_kicks/spadl/pff.py` | `silly_kicks/spadl/gradientsports.py` |
| `silly_kicks/tracking/pff.py` | `silly_kicks/tracking/gradientsports.py` |
| `tests/spadl/test_pff.py` | `tests/spadl/test_gradientsports.py` |
| `tests/test_tracking_pff.py` | `tests/test_tracking_gradientsports.py` |
| `tests/datasets/pff/` (directory) | `tests/datasets/gradientsports/` |
| `tests/datasets/tracking/pff/` (directory) | `tests/datasets/tracking/gradientsports/` |
| `docs/examples/pff_wc2022_walkthrough.py` | `docs/examples/gradientsports_wc2022_walkthrough.py` |

### Files modified (source — ~15 files)

| File | Change Summary |
|------|---------------|
| `silly_kicks/spadl/schema.py` | `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS`; docstrings |
| `silly_kicks/spadl/__init__.py` | Re-export `gradientsports` module + `GRADIENTSPORTS_SPADL_COLUMNS` |
| `silly_kicks/spadl/gradientsports.py` (post-rename) | Docstring; self-import ref in Examples |
| `silly_kicks/tracking/schema.py` | `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`; `TRACKING_CATEGORICAL_DOMAINS` frozenset `"pff"` → `"gradientsports"`; docstrings |
| `silly_kicks/tracking/__init__.py` | Re-export `gradientsports` module + `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` |
| `silly_kicks/tracking/gradientsports.py` (post-rename) | `_PROVIDER_NAME`; schema import; docstring |
| `silly_kicks/tracking/_direction.py` | Docstring-only (PFF → Gradient Sports references) |
| `tests/fixtures/baselines/preprocess_baseline.json` | `"pff"` key → `"gradientsports"` |
| `tests/fixtures/baselines/preprocess_sweep_log.json` | `"pff"` key → `"gradientsports"` |
| `silly_kicks/tracking/preprocess/_provider_defaults_generated.py` | Regenerated via script (key becomes `"gradientsports"`) |
| `scripts/regenerate_provider_defaults.py` | Provider list: `"pff"` → `"gradientsports"` |
| `NOTICE` | Update brand reference |

### Files modified (tests — ~20 files)

| File | Change Summary |
|------|---------------|
| `tests/invariants/_loaders.py` | `load_pff_synthetic` → `load_gradientsports_synthetic`; imports |
| `tests/tracking/_provider_inputs.py` | `PFF_DIR` → `GRADIENTSPORTS_DIR`; `"pff"` → `"gradientsports"` |
| `tests/test_preprocess_baseline_integrity.py` | `"pff"` → `"gradientsports"` in parametrize lists |
| `tests/test_tracking_cross_provider_parity.py` | Imports + dict key |
| `tests/test_tracking_converter_preprocess_kwarg.py` | Imports |
| `tests/test_tracking_realistic_fixtures.py` | Imports + `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` |
| `tests/test_tracking_real_data_sweep.py` | Imports + env var `PFF_TRACKING_DIR` → `GRADIENTSPORTS_TRACKING_DIR` |
| `tests/test_tracking_schema.py` | `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` |
| `tests/spadl/test_cross_provider_parity.py` | Imports |
| `tests/tracking/test_action_context_real_data_sweep.py` | Imports + env var `PFF_TRACKING_DIR` → `GRADIENTSPORTS_TRACKING_DIR` + `_skip_if_no_pff_dir` → `_skip_if_no_gs_dir` |
| `tests/tracking/test_das_e2e.py` | `PFF_DIR` import → `GRADIENTSPORTS_DIR`; `"pff"` → `"gradientsports"` |
| `tests/tracking/test_gk_influence_e2e.py` | `PFF_DIR` import → `GRADIENTSPORTS_DIR`; `"pff"` → `"gradientsports"` |
| `tests/tracking/test_pressure_real_data_calibration.py` | `PFF_DIR` import → `GRADIENTSPORTS_DIR` |
| `tests/tracking/test_gk_identification.py` | `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` |
| `tests/datasets/tracking/empirical_action_context_baselines.json` | `"pff"` key + `"synthetic_pff_medium_halftime"` |
| `tests/datasets/tracking/empirical_probe_baselines.json` | `"probe_run_source_pff_path_marker"` key + top-level `"pff"` provider key |
| `scripts/probe_tracking_baselines.py` | `PFF_LOCAL_DIR` → `GRADIENTSPORTS_LOCAL_DIR`; `_list_pff_tracking_files` → `_list_gs_tracking_files` |
| `scripts/probe_preprocess_baseline.py` | `PFF_LOCAL_DIR` → `GRADIENTSPORTS_LOCAL_DIR`; `_enrich_from_pff_block` → `_enrich_from_gs_block` |
| All 14 tracking provider test files (grep list above) | `"pff"` → `"gradientsports"` in parametrize/dict keys |

### Files modified (docs — ~5 files, NOT historical specs/plans)

| File | Change Summary |
|------|---------------|
| `CHANGELOG.md` | Add breaking-change note to upcoming release + one-liner disclaimer at top; historical entries UNTOUCHED |
| `CLAUDE.md` | `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS`; `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`; module paths |
| `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md` | Runtime identifiers + "(formerly PFF)" on first mention |
| `docs/superpowers/adrs/ADR-006-direction-of-play-handling.md` | Runtime identifiers + "(formerly PFF)" on first mention |
| `docs/superpowers/adrs/ADR-007-gk-identification-algorithm.md` | Runtime identifiers + "(formerly PFF)" on first mention |

### Historical specs/plans — UNTOUCHED

All files under `docs/superpowers/specs/` and `docs/superpowers/plans/` are historical records and are **not edited or renamed**. They document decisions made when the brand was "PFF" and remain accurate records of that time. The spec filename `2026-04-30-pff-fc-events-converter-design.md` stays as-is — renaming it would break cross-references and `git log --follow` is opt-in.

### Checked and found clean

- `tests/tracking/conftest.py` — grepped, zero PFF hits

---

## Task 1: Rename SPADL module + update schema constant

**Files:**
- Rename: `silly_kicks/spadl/pff.py` → `silly_kicks/spadl/gradientsports.py`
- Modify: `silly_kicks/spadl/schema.py`
- Modify: `silly_kicks/spadl/__init__.py`
- Modify: `silly_kicks/spadl/gradientsports.py` (post-rename)

- [ ] **Step 1: Rename the module file**

```bash
cd D:/Development/karstenskyt__silly-kicks
git mv silly_kicks/spadl/pff.py silly_kicks/spadl/gradientsports.py
```

- [ ] **Step 2: Update schema constant name in `silly_kicks/spadl/schema.py`**

Change `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS` (line 61). Update docstrings to say "Gradient Sports" instead of "PFF".

```python
GRADIENTSPORTS_SPADL_COLUMNS: dict[str, str] = {
    **SPADL_COLUMNS,
    "tackle_winner_player_id": "Int64",
    "tackle_winner_team_id": "Int64",
    "tackle_loser_player_id": "Int64",
    "tackle_loser_team_id": "Int64",
}
"""Gradient Sports SPADL output schema: SPADL_COLUMNS + 4 nullable Int64 tackle-actor
passthrough columns. NaN on rows where no challenge winner/loser is
identifiable (i.e., everywhere except CH events).

Identifier-conventions rationale (ADR-001) shared with SPORTEC_SPADL_COLUMNS.

Dtype departure from SPORTEC_SPADL_COLUMNS (which uses ``object`` strings):
Gradient Sports native player/team identifiers are integers, whereas kloppy hands sportec
strings. Using ``Int64`` (pandas nullable) preserves int-ness while allowing
NaN on non-tackle rows. Long-term unification of the two extended schemas
under a common name is a follow-up TODO."""
```

- [ ] **Step 3: Update `__init__.py` re-exports**

In `silly_kicks/spadl/__init__.py`:

- `__all__`: replace `"PFF_SPADL_COLUMNS"` → `"GRADIENTSPORTS_SPADL_COLUMNS"` and `"pff"` → `"gradientsports"`
- Import line: `from . import config, opta, pff, statsbomb, wyscout` → `from . import config, gradientsports, opta, statsbomb, wyscout`
- Schema import: `from .schema import PFF_SPADL_COLUMNS, ...` → `from .schema import GRADIENTSPORTS_SPADL_COLUMNS, ...`

- [ ] **Step 4: Update the renamed module's docstring and internal references**

In `silly_kicks/spadl/gradientsports.py`:

- Docstring line 1: `"""PFF FC / Gradient Sports DataFrame SPADL converter."""` → `"""Gradient Sports DataFrame SPADL converter."""`
- Line ~288 (Examples): `from silly_kicks.spadl import pff` → `from silly_kicks.spadl import gradientsports`
- All PFF references in the docstring body that refer to the brand (not the data format): update to "Gradient Sports". Keep "PFF" where it refers to the original data format vocabulary (e.g., "PFF events have a hierarchical shape" — this describes the data, not the company).

- [ ] **Step 5: Verify module imports**

```bash
uv run python -c "from silly_kicks.spadl import gradientsports; print(gradientsports.__name__)"
```

Expected: `silly_kicks.spadl.gradientsports`

---

## Task 2: Rename tracking module + update schema constants

**Files:**
- Rename: `silly_kicks/tracking/pff.py` → `silly_kicks/tracking/gradientsports.py`
- Modify: `silly_kicks/tracking/schema.py`
- Modify: `silly_kicks/tracking/__init__.py`
- Modify: `silly_kicks/tracking/gradientsports.py` (post-rename)
- Modify: `silly_kicks/tracking/_direction.py`

- [ ] **Step 1: Rename the module file**

```bash
git mv silly_kicks/tracking/pff.py silly_kicks/tracking/gradientsports.py
```

- [ ] **Step 2: Update schema constants in `silly_kicks/tracking/schema.py`**

Line 44: `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`. Update docstring.

Line 67: In `TRACKING_CATEGORICAL_DOMAINS["source_provider"]` frozenset, replace `"pff"` → `"gradientsports"`:

```python
GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS: dict[str, str] = {
    **TRACKING_FRAMES_COLUMNS,
    "player_id": "Int64",
    "team_id": "Int64",
}
"""Gradient Sports native output: nullable Int64 identifiers (matches
GRADIENTSPORTS_SPADL_COLUMNS convention from PR-S18; allows NaN on ball rows).
game_id stays int64."""
```

```python
TRACKING_CATEGORICAL_DOMAINS: dict[str, frozenset[str]] = {
    "ball_state": frozenset({"alive", "dead"}),
    "team_attacking_direction": frozenset({"ltr", "rtl"}),
    "speed_source": frozenset({"native", "derived"}),
    "source_provider": frozenset({"gradientsports", "sportec", "metrica", "skillcorner"}),
    "is_goalkeeper_source": frozenset({"native", "derived"}),
}
```

Line 77 docstring in `TrackingConversionReport`: `"pff" | "sportec"` → `"gradientsports" | "sportec"`.

- [ ] **Step 3: Update `__init__.py` re-exports**

In `silly_kicks/tracking/__init__.py`:

- `__all__`: replace `"PFF_TRACKING_FRAMES_COLUMNS"` → `"GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS"` and `"pff"` → `"gradientsports"`
- Import line 134: `from . import feature_framework, features, pff, ...` → `from . import feature_framework, features, gradientsports, ...`
- Schema import: `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`

- [ ] **Step 4: Update the renamed module's internals**

In `silly_kicks/tracking/gradientsports.py`:

- Docstring: `"""PFF FC tracking DataFrame converter."""` → `"""Gradient Sports tracking DataFrame converter."""`
- Line 23: `from .schema import PFF_TRACKING_FRAMES_COLUMNS` → `from .schema import GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
- Line 29: `_PROVIDER_NAME = "pff"` → `_PROVIDER_NAME = "gradientsports"`
- Any internal reference to `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
- Line ~96 (Examples): `from silly_kicks.tracking.pff import convert_to_frames` → `from silly_kicks.tracking.gradientsports import convert_to_frames`
- Runtime value `out["source_provider"] = "pff"` → `out["source_provider"] = "gradientsports"` (this should already be handled by `_PROVIDER_NAME` if it uses the constant; verify and fix if hardcoded)

- [ ] **Step 5: Update `_direction.py` docstring**

In `silly_kicks/tracking/_direction.py`, update docstring references (lines 1, 3-4, 26, 37, 39):

- `"Direction-of-play helpers shared between PFF events + tracking adapters"` → `"Direction-of-play helpers shared between Gradient Sports events + tracking adapters"`
- `"Extracted from silly_kicks/spadl/pff.py"` → `"Extracted from silly_kicks/spadl/gradientsports.py"`
- Keep "PFF metadata" references where they describe the raw data format fields (`homeTeamStartLeft`, `homeTeamStartLeftExtraTime`) — these are the actual JSON field names from the data provider.

- [ ] **Step 6: Verify module imports**

```bash
uv run python -c "from silly_kicks.tracking import gradientsports; print(gradientsports._PROVIDER_NAME)"
```

Expected: `gradientsports`

---

## Task 3: Update preprocess codegen pipeline

**Files:**
- Modify: `tests/fixtures/baselines/preprocess_baseline.json`
- Modify: `tests/fixtures/baselines/preprocess_sweep_log.json`
- Modify: `scripts/regenerate_provider_defaults.py`
- Regenerate: `silly_kicks/tracking/preprocess/_provider_defaults_generated.py`

- [ ] **Step 1: Rename the key in `preprocess_baseline.json`**

In `tests/fixtures/baselines/preprocess_baseline.json`:

- Line 8: `"pff"` in `providers_probed` array → `"gradientsports"`
- Line 14: `"pff_local:10502.jsonl.bz2"` → `"gradientsports_local:10502.jsonl.bz2"`
- Line 56: Top-level `"pff": {` key → `"gradientsports": {`

- [ ] **Step 2: Rename the key in `preprocess_sweep_log.json`**

Same changes:
- `providers_probed` array: `"pff"` → `"gradientsports"`
- `probe_sources`: `"pff_local:..."` → `"gradientsports_local:..."`
- Top-level provider key: `"pff"` → `"gradientsports"`

- [ ] **Step 3: Update `scripts/regenerate_provider_defaults.py`**

Line 61: change provider iteration order from `("sportec", "pff", "metrica", "skillcorner")` → `("sportec", "gradientsports", "metrica", "skillcorner")`.

- [ ] **Step 4: Regenerate the provider defaults**

```bash
uv run python scripts/regenerate_provider_defaults.py
```

Expected output: `[regen] wrote silly_kicks/tracking/preprocess/_provider_defaults_generated.py`

Verify the generated file now has `"gradientsports": PreprocessConfig(...)` instead of `"pff"`.

---

## Task 4: Rename test files and fixture directories

**Files:**
- Rename: `tests/spadl/test_pff.py` → `tests/spadl/test_gradientsports.py`
- Rename: `tests/test_tracking_pff.py` → `tests/test_tracking_gradientsports.py`
- Rename: `tests/datasets/pff/` → `tests/datasets/gradientsports/`
- Rename: `tests/datasets/tracking/pff/` → `tests/datasets/tracking/gradientsports/`
- Rename: `docs/examples/pff_wc2022_walkthrough.py` → `docs/examples/gradientsports_wc2022_walkthrough.py`

- [ ] **Step 1: Rename SPADL test file**

```bash
git mv tests/spadl/test_pff.py tests/spadl/test_gradientsports.py
```

- [ ] **Step 2: Rename tracking test file**

```bash
git mv tests/test_tracking_pff.py tests/test_tracking_gradientsports.py
```

- [ ] **Step 3: Rename SPADL fixture directory**

```bash
git mv tests/datasets/pff tests/datasets/gradientsports
```

- [ ] **Step 4: Rename tracking fixture directory**

```bash
git mv tests/datasets/tracking/pff tests/datasets/tracking/gradientsports
```

- [ ] **Step 5: Rename example walkthrough**

```bash
git mv docs/examples/pff_wc2022_walkthrough.py docs/examples/gradientsports_wc2022_walkthrough.py
```

---

## Task 5: Update all test file internals

**Files:**
- Modify: `tests/spadl/test_gradientsports.py` (post-rename)
- Modify: `tests/test_tracking_gradientsports.py` (post-rename)
- Modify: `tests/invariants/_loaders.py`
- Modify: `tests/tracking/_provider_inputs.py`
- Modify: `tests/test_preprocess_baseline_integrity.py`
- Modify: `tests/test_tracking_cross_provider_parity.py`
- Modify: `tests/test_tracking_converter_preprocess_kwarg.py`
- Modify: `tests/test_tracking_realistic_fixtures.py`
- Modify: `tests/test_tracking_real_data_sweep.py`
- Modify: `tests/spadl/test_cross_provider_parity.py`
- Modify: `tests/tracking/test_action_context_real_data_sweep.py`
- Modify: `tests/invariants/test_direction_of_play.py`
- Modify: `tests/invariants/test_vaep_geometric_sanity.py`
- Modify: `tests/datasets/tracking/empirical_action_context_baselines.json`
- Modify: `tests/datasets/tracking/empirical_probe_baselines.json`

- [ ] **Step 1: Update `tests/spadl/test_gradientsports.py`**

- Line 10: `from silly_kicks.spadl import pff as pff_mod` → `from silly_kicks.spadl import gradientsports as gs_mod`
- All references to `pff_mod` → `gs_mod`
- Fixture path references to `tests/datasets/pff/` → `tests/datasets/gradientsports/`
- Docstring: `"PFF"` → `"Gradient Sports"` where referring to the brand

- [ ] **Step 2: Update `tests/test_tracking_gradientsports.py`**

- Line 1: docstring `"silly_kicks.tracking.pff"` → `"silly_kicks.tracking.gradientsports"`
- Line 8: `from silly_kicks.tracking.pff import convert_to_frames` → `from silly_kicks.tracking.gradientsports import convert_to_frames`
- Line 10: `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
- Line 15: `FIXTURE_DIR` path from `"pff"` → `"gradientsports"`
- Line 22: `set(PFF_TRACKING_FRAMES_COLUMNS)` → `set(GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS)`
- Line 87: `assert report.provider == "pff"` → `assert report.provider == "gradientsports"`

- [ ] **Step 3: Update `tests/invariants/_loaders.py`**

- Line 5-6 in docstring: `"PFF synthetic"` → `"Gradient Sports synthetic"`
- Line 352: `def load_pff_synthetic()` → `def load_gradientsports_synthetic()`
- Line 353: `from silly_kicks.spadl import pff` → `from silly_kicks.spadl import gradientsports`
- Line 354: `from tests.spadl.test_pff import _load_synthetic_events` → `from tests.spadl.test_gradientsports import _load_synthetic_events`
- Line 358: `pff.convert_to_actions(` → `gradientsports.convert_to_actions(`

Update the 2 call sites in other invariant test files:

- `tests/invariants/test_direction_of_play.py:44` — `_loaders.load_pff_synthetic` → `_loaders.load_gradientsports_synthetic`; parametrize ID `"pff_synthetic"` → `"gradientsports_synthetic"`
- `tests/invariants/test_vaep_geometric_sanity.py:56` — `_loaders.load_pff_synthetic` → `_loaders.load_gradientsports_synthetic`; parametrize ID `"pff_synthetic"` → `"gradientsports_synthetic"`

- [ ] **Step 4: Update `tests/tracking/_provider_inputs.py`**

- Line 19: `PFF_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "pff"` → `GRADIENTSPORTS_DIR = REPO_ROOT / "tests" / "datasets" / "tracking" / "gradientsports"`
- Line 48: `if provider == "pff":` → `if provider == "gradientsports":`
- Line 49: `PFF_DIR / "medium_halftime.parquet"` → `GRADIENTSPORTS_DIR / "medium_halftime.parquet"`
- Line 74: `"source_provider": "pff"` → `"source_provider": "gradientsports"`

- [ ] **Step 5: Update `tests/test_preprocess_baseline_integrity.py`**

- Line 40, 51, 68: `"pff"` → `"gradientsports"` in all `@pytest.mark.parametrize` lists
- Line 87: `"pff"` → `"gradientsports"` in provider tuple

- [ ] **Step 6: Update `tests/test_tracking_cross_provider_parity.py`**

- Line 17: `from silly_kicks.tracking import pff, sportec` → `from silly_kicks.tracking import gradientsports, sportec`
- Line 34: path `"pff"` → `"gradientsports"`
- Line 35: `pff.convert_to_frames` → `gradientsports.convert_to_frames`
- Line 58: dict key `"pff"` → `"gradientsports"`

- [ ] **Step 7: Update `tests/test_tracking_converter_preprocess_kwarg.py`**

All 5 import sites (lines 75, 91, 115, 140, 172):
`from silly_kicks.tracking.pff import convert_to_frames` → `from silly_kicks.tracking.gradientsports import convert_to_frames`

Also update any fixture paths referencing `"pff"` → `"gradientsports"`.

- [ ] **Step 8: Update `tests/test_tracking_realistic_fixtures.py`**

Lines 79, 97, 120: `from silly_kicks.tracking.pff import convert_to_frames` → `from silly_kicks.tracking.gradientsports import convert_to_frames`

Update fixture directory references `"pff"` → `"gradientsports"`.

- [ ] **Step 9: Update `tests/test_tracking_real_data_sweep.py`**

Line 173: `from silly_kicks.tracking.pff import convert_to_frames` → `from silly_kicks.tracking.gradientsports import convert_to_frames`

- [ ] **Step 10: Update `tests/spadl/test_cross_provider_parity.py`**

- Line 173: `from silly_kicks.spadl import pff` → `from silly_kicks.spadl import gradientsports`
- Line 176: `from tests.spadl.test_pff import _load_synthetic_events` → `from tests.spadl.test_gradientsports import _load_synthetic_events`

- [ ] **Step 11: Update `tests/tracking/test_action_context_real_data_sweep.py`**

Lines 184, 478, 823: `from silly_kicks.tracking.pff import convert_to_frames as pff_convert` → `from silly_kicks.tracking.gradientsports import convert_to_frames as gs_convert`

Update all references from `pff_convert` → `gs_convert`.

- [ ] **Step 12: Update `tests/test_tracking_schema.py`**

- `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS` (import + all assertion references)

- [ ] **Step 13: Update `tests/test_tracking_real_data_sweep.py` env var**

- Line 93: `os.environ.get("PFF_TRACKING_DIR")` → `os.environ.get("GRADIENTSPORTS_TRACKING_DIR")`
- All skip messages and local variables: `pff_dir` → `gs_dir`, `"PFF_TRACKING_DIR"` → `"GRADIENTSPORTS_TRACKING_DIR"`

**Developer impact:** Any local `.env` file or CI secret that sets `PFF_TRACKING_DIR` must be updated to `GRADIENTSPORTS_TRACKING_DIR`. Without this, real-data sweep tests will silently skip (not fail).

- [ ] **Step 14: Update `tests/tracking/test_action_context_real_data_sweep.py` env var**

- `_skip_if_no_pff_dir` → `_skip_if_no_gs_dir`
- Lines 106-114: `"PFF_TRACKING_DIR"` → `"GRADIENTSPORTS_TRACKING_DIR"`, `pff_dir` → `gs_dir`
- All call sites of the helper + docstrings

- [ ] **Step 15: Update `tests/tracking/test_das_e2e.py`, `test_gk_influence_e2e.py`, `test_pressure_real_data_calibration.py`**

- Import: `PFF_DIR` → `GRADIENTSPORTS_DIR`
- Parametrize/conditional: `"pff"` → `"gradientsports"`, `PFF_DIR.exists()` → `GRADIENTSPORTS_DIR.exists()`

- [ ] **Step 16: Update `tests/tracking/test_gk_identification.py`**

- Import: `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`
- Assertions referencing it

- [ ] **Step 17: Update probe scripts**

In `scripts/probe_tracking_baselines.py`:
- Line 40: `PFF_LOCAL_DIR = Path(r"D:\...\Tracking Data")` → `GRADIENTSPORTS_LOCAL_DIR = Path(r"D:\...\Tracking Data")` (same path, new variable name)
- `_list_pff_tracking_files` → `_list_gs_tracking_files`
- All references to `PFF_LOCAL_DIR` → `GRADIENTSPORTS_LOCAL_DIR`

In `scripts/probe_preprocess_baseline.py`:
- `PFF_LOCAL_DIR` → `GRADIENTSPORTS_LOCAL_DIR`
- `_enrich_from_pff_block` → `_enrich_from_gs_block`
- All references updated

- [ ] **Step 18: Update tracking provider test parametrize values**

In each of these 14 files (from the grep), find `"pff"` in parametrize lists, dict keys, or conditional branches and replace with `"gradientsports"`:

- `tests/tracking/test_cover_shadows_providers.py`
- `tests/tracking/test_gk_influence_e2e.py`
- `tests/tracking/pitch_control/test_lakehouse_parity.py`
- `tests/tracking/test_team_shape_providers.py`
- `tests/tracking/test_line_breaking_providers.py`
- `tests/tracking/test_das_e2e.py`
- `tests/tracking/test_action_context_real_data_sweep.py` (already covered in Step 11)
- `tests/tracking/test_off_ball_runs_providers.py`
- `tests/tracking/test_pressure_real_data_calibration.py`
- `tests/tracking/test_synthesizer_fixture_density.py`
- `tests/tracking/test_empirical_action_context_baselines.py`
- `tests/tracking/test_action_context_cross_provider.py`
- `tests/tracking/test_action_context_expected_output.py`

- [ ] **Step 19: Update JSON baseline files**

In `tests/datasets/tracking/empirical_action_context_baselines.json`:
- Line 150: `"pff": {` → `"gradientsports": {`
- Line 151: `"synthetic_pff_medium_halftime"` → `"synthetic_gradientsports_medium_halftime"`

In `tests/datasets/tracking/empirical_probe_baselines.json`:
- Line 4: `"probe_run_source_pff_path_marker"` → `"probe_run_source_gradientsports_path_marker"`
- Line 69: top-level `"pff": {` key → `"gradientsports": {`

---

## Task 6: Update example walkthrough

**Files:**
- Modify: `docs/examples/gradientsports_wc2022_walkthrough.py` (post-rename)

- [ ] **Step 1: Update all imports and references**

- Line 1: `"PFF FC WC 2022"` → `"Gradient Sports WC 2022"`
- Line 4: `"PFF directory"` → `"Gradient Sports directory"`
- Line 7: `pff_wc2022_walkthrough.py` → `gradientsports_wc2022_walkthrough.py`
- Line 9: `"PFF directory"` → `"Gradient Sports directory"`
- Line 27: `from silly_kicks.spadl import add_names, boundary_metrics, coverage_metrics, pff` → `... gradientsports`
- Line 36: docstring references to `silly_kicks.spadl.pff` → `silly_kicks.spadl.gradientsports`
- Line 42: `pff.EXPECTED_INPUT_COLUMNS` → `gradientsports.EXPECTED_INPUT_COLUMNS`
- Line 61: `"Real PFF data"` → `"Real Gradient Sports data"` (data format clarification)
- Line 137: `"pff_dir"` CLI arg → `"gs_dir"` or keep as-is for backward compat with user scripts. **Decision:** rename to `"gs_dir"` (breaking change approved; sole consumer).
- Line 141: `args.pff_dir` → `args.gs_dir`

Update all call sites from `pff.` → `gradientsports.`.

---

## Task 7: Update documentation (non-historical)

**Files:**
- Modify: `NOTICE`
- Modify: `CHANGELOG.md`
- Modify: `CLAUDE.md`
- Modify: `docs/superpowers/adrs/ADR-004-tracking-namespace-charter.md`
- Modify: `docs/superpowers/adrs/ADR-006-direction-of-play-handling.md`
- Modify: `docs/superpowers/adrs/ADR-007-gk-identification-algorithm.md`

- [ ] **Step 1: Update `NOTICE`**

Line 45: `"PFF, Sportec"` → `"Gradient Sports (formerly PFF FC), Sportec"`. The data was licensed from the legal entity known as PFF at the time; preserving the former name in parentheses maintains attribution accuracy.

- [ ] **Step 2: Update `CHANGELOG.md`**

**Do NOT rewrite historical release entries.** Historical entries like "Added `silly_kicks.spadl.pff`" were accurate at the time of release. Rewriting them to say `gradientsports` would be factually wrong — that version did NOT ship a gradientsports module.

Add ONLY the following to the upcoming release section:

```markdown
### Breaking

- **Provider rename: PFF → Gradient Sports.** All runtime identifiers changed:
  `"pff"` → `"gradientsports"` (source_provider column values, module names,
  schema constants). `PFF_SPADL_COLUMNS` → `GRADIENTSPORTS_SPADL_COLUMNS`;
  `PFF_TRACKING_FRAMES_COLUMNS` → `GRADIENTSPORTS_TRACKING_FRAMES_COLUMNS`.
  Reflects the PFF FC → Gradient Sports corporate rebrand.
```

Add a one-liner at the top of CHANGELOG.md (after the title):

```markdown
> **Note:** PFF FC rebranded to Gradient Sports in 3.12.0. Historical entries below use the name current at time of release.
```

- [ ] **Step 3: Update `CLAUDE.md`**

Search for `PFF_SPADL_COLUMNS`, `PFF_TRACKING_FRAMES_COLUMNS`, `pff`, `PFF` and update:
- Schema constant names
- Module references (`spadl/pff.py` → `spadl/gradientsports.py`, `tracking/pff.py` → `tracking/gradientsports.py`)
- Brand references

- [ ] **Step 4: Update ADR files**

In each ADR, replace runtime identifiers (`"pff"` as provider key, `PFF_*` constants, module paths) with new names. On **first mention** in each ADR, write "Gradient Sports (formerly PFF)" to preserve navigability — future readers will see "Gradient Sports" in ADRs but "PFF" in all git history from that era. Subsequent mentions in the same ADR use just "Gradient Sports".

---

## Task 8: Version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `silly_kicks/__init__.py`

- [ ] **Step 1: Bump version to 3.12.0**

In `pyproject.toml` line 7: `version = "3.11.3"` → `version = "3.12.0"`

In `silly_kicks/__init__.py` line 7: `__version__ = "3.11.3"` → `__version__ = "3.12.0"`

---

## Task 9: Run verification suite

- [ ] **Step 1: Verify zero remaining `pff` hits in runtime code**

Use the Grep tool: pattern `\bpff\b` in `silly_kicks/**/*.py` — expected: zero hits.

- [ ] **Step 2: Ruff lint**

```bash
uv run ruff check silly_kicks/ tests/ scripts/
```

Expected: no new errors (uses project-configured rule set from pyproject.toml).

- [ ] **Step 3: Ruff format**

```bash
uv run ruff format --check silly_kicks/ tests/ scripts/
```

Expected: clean.

- [ ] **Step 4: Pyright**

```bash
uv run pyright silly_kicks/
```

Expected: no new type errors.

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/ -m "not e2e" -v --tb=short
```

Expected: all tests pass. Watch specifically for:
- `test_preprocess_baseline_integrity` — confirms codegen pipeline consistency
- `test_tracking_cross_provider_parity` — confirms the renamed module still works
- `tests/spadl/test_gradientsports.py` — SPADL converter tests
- `tests/test_tracking_gradientsports.py` — tracking converter tests
- All tracking provider parametrized tests

- [ ] **Step 6: Final grep audit**

```bash
grep -rn "pff" --include="*.py" --include="*.json" silly_kicks/ tests/ scripts/
```

Filter results: any remaining `pff` hits should ONLY be in:
- Historical comments that describe the original data format field names (acceptable)
- `__pycache__` directories (will be regenerated)

Zero hits expected in:
- Import paths
- Runtime string values (`source_provider`, `_PROVIDER_NAME`, dict keys)
- Schema constant names
- Parametrize values

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "refactor: rename PFF provider to Gradient Sports (gradientsports)

Reflects the PFF FC -> Gradient Sports corporate rebrand. Breaking change:
all runtime identifiers changed ('pff' -> 'gradientsports' in module names,
source_provider column values, schema constants, test fixtures).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```
