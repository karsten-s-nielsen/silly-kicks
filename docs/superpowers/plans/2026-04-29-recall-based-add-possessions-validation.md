# Recall-based `add_possessions` validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship silly-kicks 1.8.0 — a documentation-and-test-scaffolding release that publishes empirical recall/precision/F1 baselines for `add_possessions`, promotes a public `boundary_metrics` utility (TypedDict return, keyword-only args), and adds a 3-fixture parametrized CI regression gate (recall ≥ 0.85 AND precision ≥ 0.30 per match) — replacing a previously-skipping single-fixture F1 e2e test that produced zero coverage for ~6 release cycles.

**Architecture:** Hexagonal — pure pandas-in/pandas-out helpers in `silly_kicks/spadl/utils.py` next to the existing `add_possessions`; re-exported via `silly_kicks/spadl/__init__.py` symmetric with the rest of the public post-conversion family. Test scaffolding lives in `tests/spadl/test_add_possessions.py` (existing file). Three raw StatsBomb open-data fixtures vendored under `tests/datasets/statsbomb/raw/events/` with license attribution.

**Tech Stack:** Python 3.10+ (stdlib `typing.TypedDict`), pandas, numpy, pytest, pytest's `parametrize`. No new dependencies.

**Important — silly-kicks commit discipline (per `feedback_commit_policy` memory):**
- **Literally ONE commit per branch.** No per-task commits. All changes accumulate locally; the final task commits everything as a single squash-ready commit.
- The `git commit` step is at the end of the plan, not per-task. Do not run `git add` / `git commit` between tasks.
- Branch name: `feat/recall-based-add-possessions-validation`.
- User approval gates apply at: (a) the single commit, (b) `git push`, (c) PR open, (d) PR merge, (e) tag push. Stop and ask before each.

**Cross-version pin (per `feedback_ci_cross_version` memory):** before running any verification gate, install the EXACT CI pin so local pyright matches CI bit-for-bit:
```
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `silly_kicks/spadl/utils.py` | Modify | Insert `BoundaryMetrics` TypedDict + `boundary_metrics()` after `add_possessions` (line 693, before `add_names` at 696). Replace the misleading "boundary-F1 ~0.90" passage in `add_possessions` docstring (lines 580-582) with empirical baselines. |
| `silly_kicks/spadl/__init__.py` | Modify | Re-export `boundary_metrics` and `BoundaryMetrics` (alphabetical insertion in `__all__` + import block). |
| `tests/spadl/test_add_possessions.py` | Modify | Remove local `_boundary_f1` helper. Import `boundary_metrics` from utils. Add `TestBoundaryMetrics` unit-test class with 4 sub-classes (Contract / Correctness / Degenerate / Errors). Rewrite e2e class as `TestBoundaryAgainstStatsBombNative` — no `e2e` marker, parametrize over 3 matches, recall + precision gates. |
| `tests/datasets/statsbomb/raw/events/7298.json` | Create | StatsBomb open-data Women's World Cup match. |
| `tests/datasets/statsbomb/raw/events/7584.json` | Create | StatsBomb open-data Champions League match. |
| `tests/datasets/statsbomb/raw/events/3754058.json` | Create | StatsBomb open-data Premier League match. |
| `tests/datasets/statsbomb/README.md` | Create | StatsBomb open-data attribution + license link. |
| `pyproject.toml` | Modify | Version `1.7.0` → `1.8.0` (line 7). |
| `CHANGELOG.md` | Modify | New `## [1.8.0] — 2026-04-29` entry at the top. |
| `TODO.md` | Modify | Add `## Open PRs` section with PR-S9 + PR-S10 entries. |
| `CLAUDE.md` | Modify (optional) | Tighten "Testing" wording: committed-fixture tests should not be marked e2e. |
| `docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md` | Already created | Design doc — bundled into the single commit. |
| `docs/superpowers/plans/2026-04-29-recall-based-add-possessions-validation.md` | This file | Implementation plan — bundled into the single commit. |

---

## Task 0: Pre-flight — verify clean baseline

**Files:** None (verification only).

- [ ] **Step 1: Verify clean working tree on branch `main`**

```bash
git status
git log --oneline -1
grep "^version" pyproject.toml
```

Expected output:
- `git status`: only untracked files allowed are `README.md.backup` and `uv.lock` (per session-start gitStatus).
- `git log`: `45ef2f8 feat(spadl): dedicated DataFrame converters for Sportec + Metrica + kloppy direction-of-play unification — silly-kicks 1.7.0 (#12)`.
- `pyproject.toml`: `version = "1.7.0"`.

If any of these don't match, STOP and surface to the user.

- [ ] **Step 2: Create feature branch**

```bash
git checkout -b feat/recall-based-add-possessions-validation
```

- [ ] **Step 3: Install exact CI pin**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

Expected: clean install, no version conflicts. This must run in the same uv-managed environment as the test runner.

- [ ] **Step 4: Run baseline tests**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -40
```

Expected: all tests either pass or skip (the 5 `test_predict*` cases skip because no WorldCup HDF5; the existing `TestBoundaryF1AgainstStatsBombNative::test_boundary_f1_against_native_possession_id` skips because no fixture). No failures, no errors. If anything fails, STOP — fix the baseline before doing PR-S8 work.

- [ ] **Step 5: Run baseline lint + pyright**

```bash
uv run ruff check silly_kicks/ tests/
uv run ruff format --check silly_kicks/ tests/
uv run pyright silly_kicks/
```

Expected: zero errors from each. If any fail, STOP — fix the baseline first.

---

## Task 1: Write `TestBoundaryMetrics` unit tests (failing — function doesn't exist yet)

**Files:**
- Modify: `tests/spadl/test_add_possessions.py` (append `TestBoundaryMetrics` family at end of file).

- [ ] **Step 1: Add a temporary import that will fail**

At the top of `tests/spadl/test_add_possessions.py`, find this import block (line 22-23):

```python
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions
```

Replace with:

```python
from silly_kicks.spadl import config as spadlconfig
from silly_kicks.spadl.utils import add_possessions, boundary_metrics
```

This will cause every test in the file to fail at collection time with `ImportError: cannot import name 'boundary_metrics' from 'silly_kicks.spadl.utils'`. Intentional — confirms the test wiring exercises the new symbol.

- [ ] **Step 2: Append the four `TestBoundaryMetrics` classes at the end of `test_add_possessions.py`**

After the existing class `TestBoundaryF1AgainstStatsBombNative` (the very last class — line 525-end), append the following:

```python


# ---------------------------------------------------------------------------
# Unit tests for the public boundary_metrics utility (added in 1.8.0)
# ---------------------------------------------------------------------------


class TestBoundaryMetricsContract:
    def test_returns_dict_with_required_keys(self):
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 1, 1]),
            native=pd.Series([0, 0, 1, 1]),
        )
        assert set(m.keys()) == {"precision", "recall", "f1"}

    def test_all_metric_values_are_floats(self):
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 1, 1]),
            native=pd.Series([0, 0, 1, 1]),
        )
        assert isinstance(m["precision"], float)
        assert isinstance(m["recall"], float)
        assert isinstance(m["f1"], float)

    def test_keyword_only_args_required(self):
        # Positional invocation must raise TypeError. The args are asymmetric
        # (swapping inputs swaps precision and recall), so positional usage is
        # a silent footgun we eliminate at the API surface.
        with pytest.raises(TypeError):
            boundary_metrics(pd.Series([0, 0, 1]), pd.Series([0, 0, 1]))  # type: ignore[misc]


class TestBoundaryMetricsCorrectness:
    def test_identical_sequences_all_metrics_one(self):
        s = pd.Series([0, 0, 1, 1, 2, 2])
        m = boundary_metrics(heuristic=s, native=s)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_relabeled_identical_sequences_all_metrics_one(self):
        # Boundaries are invariant under counter-relabeling: [0,0,1,1] and
        # [5,5,7,7] emit boundaries at the same row. Both should report
        # perfect agreement.
        h = pd.Series([0, 0, 1, 1, 2, 2])
        n = pd.Series([5, 5, 7, 7, 9, 9])
        m = boundary_metrics(heuristic=h, native=n)
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_completely_disjoint_boundaries_all_zero(self):
        # Heuristic emits boundary at idx 2; native emits at idx 1.
        # No overlap → TP=0, FP=1, FN=1 → precision=recall=f1=0.
        h = pd.Series([0, 0, 1, 1])
        n = pd.Series([0, 1, 1, 1])
        m = boundary_metrics(heuristic=h, native=n)
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_partial_overlap_hand_computed(self):
        # Heuristic boundaries at idx 1, 2, 4 (3 total)
        # Native boundaries at idx 1, 4 (2 total)
        # Shared boundaries at idx 1, 4 → TP=2, FP=1 (idx 2), FN=0
        # precision = 2/(2+1) = 2/3 ≈ 0.6667
        # recall = 2/(2+0) = 1.0
        # f1 = 2 * 0.6667 * 1.0 / (0.6667 + 1.0) = 0.8
        h = pd.Series([0, 1, 2, 2, 3])
        n = pd.Series([0, 1, 1, 1, 2])
        m = boundary_metrics(heuristic=h, native=n)
        assert abs(m["precision"] - 2 / 3) < 1e-9
        assert m["recall"] == 1.0
        assert abs(m["f1"] - 0.8) < 1e-9


class TestBoundaryMetricsDegenerate:
    def test_empty_sequences_returns_zeros(self):
        m = boundary_metrics(
            heuristic=pd.Series([], dtype=np.int64),
            native=pd.Series([], dtype=np.int64),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_single_row_returns_zeros(self):
        # Single-row sequences have no consecutive pairs → no boundaries
        # detectable in either series → degenerate → all zeros.
        m = boundary_metrics(
            heuristic=pd.Series([0]),
            native=pd.Series([0]),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0

    def test_constant_sequences_returns_zeros(self):
        # All-constant sequences have zero boundaries → degenerate → all zeros.
        m = boundary_metrics(
            heuristic=pd.Series([0, 0, 0, 0, 0]),
            native=pd.Series([7, 7, 7, 7, 7]),
        )
        assert m["precision"] == 0.0
        assert m["recall"] == 0.0
        assert m["f1"] == 0.0


class TestBoundaryMetricsErrors:
    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match=r"length"):
            boundary_metrics(
                heuristic=pd.Series([0, 0, 1, 1]),
                native=pd.Series([0, 0, 1]),
            )
```

- [ ] **Step 3: Run the new tests to verify they fail at ImportError**

```bash
uv run pytest tests/spadl/test_add_possessions.py -v --tb=short 2>&1 | tail -20
```

Expected: collection-time `ImportError` for `boundary_metrics`. Every test in the file errors. This confirms test wiring works and the symbol is genuinely not present.

---

## Task 2: Implement `BoundaryMetrics` + `boundary_metrics()` in utils.py

**Files:**
- Modify: `silly_kicks/spadl/utils.py` (insert directly after `add_possessions` at line 693, before `add_names` at line 696).

- [ ] **Step 1: Add `TypedDict` import**

At the top of `silly_kicks/spadl/utils.py` find:

```python
import warnings
from typing import Final
```

Replace with:

```python
import warnings
from typing import Final, TypedDict
```

- [ ] **Step 2: Insert `BoundaryMetrics` TypedDict + `boundary_metrics()` function**

In `silly_kicks/spadl/utils.py`, find the end of `add_possessions` (the closing `return sorted_actions` and blank line before `def add_names`):

```python
    sorted_actions = sorted_actions.drop(columns=["_new_possession"])

    return sorted_actions


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
```

Replace with:

```python
    sorted_actions = sorted_actions.drop(columns=["_new_possession"])

    return sorted_actions


class BoundaryMetrics(TypedDict):
    """Boundary-detection metrics returned by :func:`boundary_metrics`.

    All three values are floats in ``[0.0, 1.0]``. When a denominator is
    zero (no boundaries in either input, no boundaries in the heuristic,
    or no boundaries in the native), the corresponding metric is reported
    as ``0.0`` rather than raising — callers can compute on degenerate
    sequences (empty, single-row, all-constant) without guarding.
    """

    precision: float
    recall: float
    f1: float


def boundary_metrics(
    *,
    heuristic: pd.Series,
    native: pd.Series,
) -> BoundaryMetrics:
    """Boundary precision / recall / F1 between two possession-id sequences.

    Compares two integer possession-id sequences over identical row order
    (typically: :func:`add_possessions`'s heuristic output and a provider's
    native possession_id, both on the same SPADL action stream). Reports
    where the two sequences emit possession boundaries — invariant under
    counter relabeling, since boundaries are detected as consecutive-row
    inequality regardless of the absolute id values.

    Empirical baselines on StatsBomb open-data for silly-kicks's
    :func:`add_possessions` (3 matches across Women's World Cup,
    Champions League, Premier League):

    - Recall ≈ 0.93 — every real possession boundary is detected.
    - Precision ≈ 0.42 — the heuristic emits ~2× more boundaries than
      StatsBomb's native annotation (the precision gap reflects the
      team-change-with-carve-outs algorithm class, not a defect).
    - F1 ≈ 0.58.

    Recall is the meaningful regression signal. F1 conflates two
    independent signals with very different magnitudes; consumers
    should report F1 alongside but should not treat it as a primary
    metric.

    Parameters
    ----------
    heuristic : pd.Series
        Possession-id sequence from ``add_possessions`` (or any other
        heuristic). Integer-typed.
    native : pd.Series
        Provider-native possession-id sequence (e.g. StatsBomb
        ``possession``). Same length and row order as ``heuristic``.

    Returns
    -------
    BoundaryMetrics
        ``{"precision": ..., "recall": ..., "f1": ...}``. Returns ``0.0``
        for any metric whose denominator is zero (empty / single-row /
        constant sequences, or no boundaries on one side).

    Raises
    ------
    ValueError
        If ``len(heuristic) != len(native)``.

    Examples
    --------
    Validate :func:`add_possessions` against StatsBomb native::

        actions, _ = statsbomb.convert_to_actions(
            events, home_team_id=100, preserve_native=["possession"]
        )
        actions = add_possessions(actions)
        m = boundary_metrics(
            heuristic=actions["possession_id"],
            native=actions["possession"].astype("int64"),
        )
        # m["recall"] ~0.93, m["precision"] ~0.42, m["f1"] ~0.58 on
        # typical StatsBomb open-data matches.
    """
    if len(heuristic) != len(native):
        raise ValueError(
            f"boundary_metrics: heuristic and native must have the same length. "
            f"Got len(heuristic)={len(heuristic)} vs len(native)={len(native)}."
        )

    # Degenerate sizes: 0 or 1 → no consecutive pairs → no boundaries.
    n = len(heuristic)
    if n < 2:
        return BoundaryMetrics(precision=0.0, recall=0.0, f1=0.0)

    h_changes = heuristic.ne(heuristic.shift(1)).iloc[1:].to_numpy()
    n_changes = native.ne(native.shift(1)).iloc[1:].to_numpy()

    tp = int((h_changes & n_changes).sum())
    fp = int((h_changes & ~n_changes).sum())
    fn = int((~h_changes & n_changes).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return BoundaryMetrics(precision=precision, recall=recall, f1=f1)


def add_names(actions: pd.DataFrame) -> pd.DataFrame:
```

- [ ] **Step 3: Run the unit tests**

```bash
uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryMetricsContract tests/spadl/test_add_possessions.py::TestBoundaryMetricsCorrectness tests/spadl/test_add_possessions.py::TestBoundaryMetricsDegenerate tests/spadl/test_add_possessions.py::TestBoundaryMetricsErrors -v --tb=short
```

Expected: all 11 unit tests pass.

- [ ] **Step 4: Run the full `test_add_possessions.py` to confirm no other breakage**

```bash
uv run pytest tests/spadl/test_add_possessions.py -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass + 11 new pass + 1 e2e test still skip (`test_boundary_f1_against_native_possession_id`).

---

## Task 3: Re-export `boundary_metrics` and `BoundaryMetrics` from `silly_kicks.spadl`

**Files:**
- Modify: `silly_kicks/spadl/__init__.py`.

- [ ] **Step 1: Add to `__all__` and import block**

Find `silly_kicks/spadl/__init__.py`:

```python
"""Implementation of the SPADL language."""

__all__ = [
    "SPADL_COLUMNS",
    "ConversionReport",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "config",
    "kloppy",
    "opta",
    "play_left_to_right",
    "results_df",
    "statsbomb",
    "validate_spadl",
    "wyscout",
]

from . import config, opta, statsbomb, wyscout
from .config import actiontypes_df, bodyparts_df, results_df
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import (
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    play_left_to_right,
    validate_spadl,
)
```

Replace with:

```python
"""Implementation of the SPADL language."""

__all__ = [
    "SPADL_COLUMNS",
    "BoundaryMetrics",
    "ConversionReport",
    "actiontypes_df",
    "add_gk_distribution_metrics",
    "add_gk_role",
    "add_names",
    "add_possessions",
    "add_pre_shot_gk_context",
    "bodyparts_df",
    "boundary_metrics",
    "config",
    "kloppy",
    "opta",
    "play_left_to_right",
    "results_df",
    "statsbomb",
    "validate_spadl",
    "wyscout",
]

from . import config, opta, statsbomb, wyscout
from .config import actiontypes_df, bodyparts_df, results_df
from .schema import SPADL_COLUMNS, ConversionReport
from .utils import (
    BoundaryMetrics,
    add_gk_distribution_metrics,
    add_gk_role,
    add_names,
    add_possessions,
    add_pre_shot_gk_context,
    boundary_metrics,
    play_left_to_right,
    validate_spadl,
)
```

- [ ] **Step 2: Verify the public import works**

```bash
uv run python -c "from silly_kicks.spadl import add_possessions, boundary_metrics, BoundaryMetrics; print('OK')"
```

Expected: `OK`. If `ImportError` or `AttributeError`, fix and re-run.

- [ ] **Step 3: Verify ruff still clean**

```bash
uv run ruff check silly_kicks/
```

Expected: zero errors.

---

## Task 4: Vendor 3 raw StatsBomb open-data fixtures + attribution README

**Files:**
- Create: `tests/datasets/statsbomb/raw/events/7298.json`
- Create: `tests/datasets/statsbomb/raw/events/7584.json`
- Create: `tests/datasets/statsbomb/raw/events/3754058.json`
- Create: `tests/datasets/statsbomb/README.md`

- [ ] **Step 1: Create the directory + download the 3 files**

```bash
mkdir -p tests/datasets/statsbomb/raw/events
curl -fsSL https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/7298.json -o tests/datasets/statsbomb/raw/events/7298.json
curl -fsSL https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/7584.json -o tests/datasets/statsbomb/raw/events/7584.json
curl -fsSL https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/3754058.json -o tests/datasets/statsbomb/raw/events/3754058.json
```

Expected: each file ~600 KB. If `curl` fails (network/404), STOP and surface — match IDs may have moved upstream.

- [ ] **Step 2: Sanity-check the downloads**

```bash
ls -lh tests/datasets/statsbomb/raw/events/
uv run python -c "import json; [print(f'{m}: {len(json.load(open(f\"tests/datasets/statsbomb/raw/events/{m}.json\")))} events') for m in [7298, 7584, 3754058]]"
```

Expected: 3 JSON files, each containing 3000-4500 events (typical match volume). If event count is suspiciously low (< 1000), the file is corrupt — re-download.

- [ ] **Step 3: Create attribution README**

Create `tests/datasets/statsbomb/README.md` with this content:

```markdown
# StatsBomb open-data fixtures

Vendored from https://github.com/statsbomb/open-data under the StatsBomb
Public Data License (non-commercial). See
https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf for the
full license text.

Used for offline e2e validation of silly-kicks's SPADL converters and
post-conversion enrichments. The three matches under `raw/events/` are
the same ones measured during the luxury-lakehouse PR-LL2 boundary-
metrics campaign that produced the empirical baselines published in
`silly_kicks.spadl.add_possessions`'s docstring:

- `7298.json` — Women's World Cup
- `7584.json` — Champions League
- `3754058.json` — Premier League

License compliance: this is non-commercial use; redistribution is
permitted under the same license.
```

- [ ] **Step 4: Confirm pytest collection still works**

```bash
uv run pytest tests/ --collect-only -q 2>&1 | tail -10
```

Expected: collection succeeds with the existing test count plus the 11 new `TestBoundaryMetrics` cases. No collection errors from the new files (they're under `datasets/`, not picked up as tests).

---

## Task 5: Rewrite the e2e test class — drop `e2e` marker, parametrize, recall + precision gates

**Files:**
- Modify: `tests/spadl/test_add_possessions.py` (replace the existing `_boundary_f1` helper + `TestBoundaryF1AgainstStatsBombNative` class).

- [ ] **Step 1: Remove the local `_boundary_f1` helper**

In `tests/spadl/test_add_possessions.py`, find the helper section starting at line 491:

```python
# ---------------------------------------------------------------------------
# Helpers — boundary-F1 metric for end-to-end validation
# ---------------------------------------------------------------------------


def _boundary_f1(heuristic: pd.Series, native: pd.Series) -> float:
    """F1 score on possession boundaries (where the id changes between consecutive rows).

    Boundaries are invariant under counter relabeling, so the heuristic's
    possession_id (0-indexed) and the provider's native possession_id
    (whatever offset) compare directly on where they emit a boundary.

    Returns 0.0 when no boundaries exist in either series (degenerate input).
    """
    h_changes = heuristic.ne(heuristic.shift(1)).iloc[1:].to_numpy()
    n_changes = native.ne(native.shift(1)).iloc[1:].to_numpy()

    tp = int((h_changes & n_changes).sum())
    fp = int((h_changes & ~n_changes).sum())
    fn = int((~h_changes & n_changes).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

Delete the entire block (the section comment + the `_boundary_f1` function, ~26 lines). Helper is now public via `boundary_metrics` imported at the top of the file.

- [ ] **Step 2: Replace the e2e test class**

In `tests/spadl/test_add_possessions.py`, find the existing class (line 519-end of file):

```python
# ---------------------------------------------------------------------------
# End-to-end: heuristic vs StatsBomb native possession_id (requires fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.e2e
class TestBoundaryF1AgainstStatsBombNative:
    """Validate the heuristic against StatsBomb's native possession_id.

    Requires raw StatsBomb event JSON committed at
    ``tests/datasets/statsbomb/raw/events/<MATCH_ID>.json`` — these are
    e2e fixtures not committed to the repo. Skips when absent.

    Procedure (also useful as documentation for downstream consumers
    running this validation against their own StatsBomb data):

    1. Load raw StatsBomb events with the top-level ``possession`` field.
    2. Convert via ``statsbomb.convert_to_actions(events, ...,
       preserve_native=['possession'])``.
    3. Run ``add_possessions(actions)`` to produce the heuristic
       ``possession_id`` column alongside the native ``possession``.
    4. Compute boundary-F1 between the two on the same action stream.

    Published heuristic baselines for similar team-change-with-carve-outs
    approaches sit in the 0.85-0.95 boundary-F1 range vs StatsBomb's
    proprietary possession assignment.
    """

    def test_boundary_f1_against_native_possession_id(self):
        import json
        import os

        fixture_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "statsbomb", "raw", "events", "7584.json"
        )
        if not os.path.exists(fixture_path):
            pytest.skip(f"Raw StatsBomb fixture not found at {fixture_path}")

        from silly_kicks.spadl import statsbomb

        with open(fixture_path, encoding="utf-8") as f:
            events_raw = json.load(f)

        # Adapter: StatsBomb open data top-level keys → silly-kicks EXPECTED_INPUT_COLUMNS.
        _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
        adapted = pd.DataFrame(
            [
                {
                    "game_id": 7584,
                    "event_id": e.get("id"),
                    "period_id": e.get("period"),
                    "timestamp": e.get("timestamp"),
                    "team_id": (e.get("team") or {}).get("id"),
                    "player_id": (e.get("player") or {}).get("id"),
                    "type_name": (e.get("type") or {}).get("name"),
                    "location": e.get("location"),
                    "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
                    # preserve_native target: top-level possession sequence number.
                    "possession": e.get("possession"),
                }
                for e in events_raw
            ]
        )

        actions, _report = statsbomb.convert_to_actions(
            adapted,
            home_team_id=int(adapted["team_id"].iloc[0]),
            preserve_native=["possession"],
        )

        # Drop synthetic dribbles (NaN original_event_id → no native possession to compare against).
        non_synthetic = actions[actions["possession"].notna()].copy()
        non_synthetic = add_possessions(non_synthetic)

        f1 = _boundary_f1(non_synthetic["possession_id"], non_synthetic["possession"].astype(np.int64))
        # Defensible baseline from published team-change-with-carve-outs literature.
        # Refine threshold to (observed - 0.02) once measured locally on this fixture.
        assert f1 >= 0.80, f"boundary-F1 {f1:.4f} below 0.80 baseline"
```

Replace the entire block above with:

```python
# ---------------------------------------------------------------------------
# End-to-end: add_possessions vs StatsBomb native possession_id
# ---------------------------------------------------------------------------


class TestBoundaryAgainstStatsBombNative:
    """Validate add_possessions against StatsBomb's native possession_id.

    Empirically against StatsBomb open-data, this heuristic achieves
    boundary recall ~0.93 and boundary F1 ~0.58. The precision gap is
    intrinsic to the team-change-with-carve-outs algorithm class. The
    CI gate below tests recall AND precision because both are observable
    behaviors that downstream consumers can develop dependencies on —
    F1 conflates two signals with very different magnitudes and is
    recorded for diagnostics only.

    Fixtures (committed under ``tests/datasets/statsbomb/raw/events/``)
    are 3 diverse matches measured during the luxury-lakehouse PR-LL2
    boundary-metrics campaign:

    - 7298  — Women's World Cup
    - 7584  — Champions League
    - 3754058 — Premier League

    Per-match independent gates: any single match falling below
    ``recall >= 0.85 AND precision >= 0.30`` fires the regression.
    """

    @pytest.mark.parametrize("match_id", [7298, 7584, 3754058])
    def test_boundary_metrics_against_native_possession_id(self, match_id: int):
        import json
        import os

        fixture_path = os.path.join(
            os.path.dirname(__file__), "..", "datasets", "statsbomb", "raw", "events", f"{match_id}.json"
        )
        if not os.path.exists(fixture_path):
            pytest.fail(
                f"StatsBomb fixture not found at {fixture_path}. "
                f"This test requires committed fixtures under tests/datasets/statsbomb/raw/events/ — "
                f"see tests/datasets/statsbomb/README.md for vendoring details."
            )

        from silly_kicks.spadl import statsbomb

        with open(fixture_path, encoding="utf-8") as f:
            events_raw = json.load(f)

        # Adapter: StatsBomb open data top-level keys → silly-kicks EXPECTED_INPUT_COLUMNS.
        _top_level_keys = {"id", "period", "timestamp", "team", "player", "type", "location"}
        adapted = pd.DataFrame(
            [
                {
                    "game_id": match_id,
                    "event_id": e.get("id"),
                    "period_id": e.get("period"),
                    "timestamp": e.get("timestamp"),
                    "team_id": (e.get("team") or {}).get("id"),
                    "player_id": (e.get("player") or {}).get("id"),
                    "type_name": (e.get("type") or {}).get("name"),
                    "location": e.get("location"),
                    "extra": {k: v for k, v in e.items() if k not in _top_level_keys},
                    # preserve_native target: top-level possession sequence number.
                    "possession": e.get("possession"),
                }
                for e in events_raw
            ]
        )

        actions, _report = statsbomb.convert_to_actions(
            adapted,
            home_team_id=int(adapted["team_id"].dropna().iloc[0]),
            preserve_native=["possession"],
        )

        # Keep only non-synthetic rows. Synthetic dribbles inserted by
        # _add_dribbles have possession=NaN (no source event to inherit
        # from); we want to compare heuristic vs native only where both
        # are defined. .copy() avoids SettingWithCopyWarning on the
        # subsequent add_possessions call.
        non_synthetic = actions[actions["possession"].notna()].copy()
        non_synthetic = add_possessions(non_synthetic)

        m = boundary_metrics(
            heuristic=non_synthetic["possession_id"],
            native=non_synthetic["possession"].astype(np.int64),
        )

        # Per-match independent gates.
        # Recall floor: 0.85 — 8pp below the worst observed (~0.93).
        # Precision floor: 0.30 — 9pp below the worst observed (~0.39 on match 3754058).
        # F1 in message only — F1 conflates two signals; gating on it
        # would re-introduce the misrepresentation problem the docstring
        # rewrite is fixing.
        assert m["recall"] >= 0.85 and m["precision"] >= 0.30, (
            f"match {match_id}: recall={m['recall']:.4f} "
            f"precision={m['precision']:.4f} f1={m['f1']:.4f}; "
            f"thresholds: recall>=0.85 precision>=0.30"
        )
```

Note the change from `pytest.skip` (old behavior when fixture absent) to `pytest.fail` — fixtures are now committed; if missing, that's a packaging bug worth surfacing, not a silently-skipping test.

- [ ] **Step 3: Run the rewritten e2e test**

```bash
uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative -v --tb=short
```

Expected: 3 parametrized tests pass with comfortable margin. The full assert message will print the actual recall/precision/F1 on test failure; on success the test stays silent. If you want to see the values, run with `-s`:

```bash
uv run pytest tests/spadl/test_add_possessions.py::TestBoundaryAgainstStatsBombNative -v -s --tb=short
```

If any match comes in tighter than expected (precision < 0.35 or recall < 0.90), STOP and report the values — discuss with user before locking thresholds.

- [ ] **Step 4: Run full test suite to confirm no regressions**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass + 11 new TestBoundaryMetrics + 3 new TestBoundaryAgainstStatsBombNative pass + 5 `test_predict*` skip (PR-S9 territory). Zero failures.

---

## Task 6: Update `add_possessions` docstring with empirical baselines

**Files:**
- Modify: `silly_kicks/spadl/utils.py:580-582`.

- [ ] **Step 1: Replace the docstring passage**

In `silly_kicks/spadl/utils.py`, find this passage (within the `add_possessions` docstring, currently at lines 580-582):

```python
    The carve-out is approximate (StatsBomb's proprietary possession rules
    capture additional context), but matches typical published heuristics
    at boundary-F1 ~0.90 against StatsBomb's native possession_id.
```

Replace with:

```python
    The carve-out is approximate (StatsBomb's proprietary possession rules
    capture additional context like merging brief opposing-team actions
    back into the containing possession). Empirically against StatsBomb
    open-data the heuristic achieves:

        - Boundary recall: ~0.93 — every real possession boundary is detected.
        - Boundary precision: ~0.42 — the heuristic emits ~2× more boundaries
          than StatsBomb's native annotation, since it can't replicate the
          "merge brief opposing actions" rule structurally.
        - Boundary F1: ~0.58 (peak ~0.605 at max_gap_seconds=10.0).

    Recall is the meaningful metric for downstream consumers — possessions
    detected by the heuristic correspond to real possession changes. The
    precision gap reflects the algorithm class, not a defect. Consumers
    needing strict StatsBomb-equivalent semantics should use the native
    possession_id where available; the heuristic is a possession proxy
    for sources without one (Wyscout, Sportec, Metrica, etc.).

    Published "0.85-0.95 F1" baselines exist for related heuristic methods
    in the literature, but use looser boundary-matching criteria or
    different ground-truth annotations than StatsBomb's open-data
    possession_id. See :func:`boundary_metrics` for downstream
    measurement.
```

- [ ] **Step 2: Verify the docstring renders correctly**

```bash
uv run python -c "import silly_kicks.spadl.utils as u; print(u.add_possessions.__doc__)"
```

Expected: prints the full updated docstring. No syntax errors.

- [ ] **Step 3: Run the unit tests on `add_possessions` to confirm no functional regression**

```bash
uv run pytest tests/spadl/test_add_possessions.py::TestAddPossessionsContract tests/spadl/test_add_possessions.py::TestSameTeamSamePossession tests/spadl/test_add_possessions.py::TestTeamChangeNewPossession tests/spadl/test_add_possessions.py::TestPeriodChange tests/spadl/test_add_possessions.py::TestGameChange tests/spadl/test_add_possessions.py::TestMaxGapTimeout tests/spadl/test_add_possessions.py::TestSetPieceCarveOut tests/spadl/test_add_possessions.py::TestMonotonic tests/spadl/test_add_possessions.py::TestCombinedBoundaries tests/spadl/test_add_possessions.py::TestSortOrder tests/spadl/test_add_possessions.py::TestErrors -v --tb=short 2>&1 | tail -10
```

Expected: all pass (none of these touch the docstring).

---

## Task 7: Update CHANGELOG, TODO.md, optionally CLAUDE.md, version bump

**Files:**
- Modify: `pyproject.toml`
- Modify: `CHANGELOG.md`
- Modify: `TODO.md`
- Modify (optional): `CLAUDE.md`

- [ ] **Step 1: Bump version in pyproject.toml**

In `pyproject.toml`, find:

```toml
version = "1.7.0"
```

Replace with:

```toml
version = "1.8.0"
```

- [ ] **Step 2: Add CHANGELOG entry**

In `CHANGELOG.md`, find the existing entry header:

```markdown
## [1.7.0] — 2026-04-29
```

Insert this new entry directly above it (so the newest entry stays at the top):

```markdown
## [1.8.0] — 2026-04-29

### Added
- **Public `silly_kicks.spadl.boundary_metrics(*, heuristic, native)` utility**
  for computing precision / recall / F1 between two possession-id sequences.
  Returns a `BoundaryMetrics` TypedDict (also re-exported from
  `silly_kicks.spadl`). Keyword-only arguments — the metric is asymmetric
  (precision and recall swap when inputs swap), so positional usage is a
  silent footgun the API surface eliminates. Returns `0.0` for any metric
  whose denominator is zero (empty / single-row / constant sequences).
  Length-mismatched inputs raise `ValueError`.
- 3 vendored StatsBomb open-data fixtures under
  `tests/datasets/statsbomb/raw/events/` (matches 7298, 7584, 3754058
  — Women's World Cup, Champions League, Premier League). License
  attribution in `tests/datasets/statsbomb/README.md`. Used by the new
  parametrized regression gate.

### Changed
- **`add_possessions` docstring is now honest about empirical performance.**
  The previous "boundary-F1 ~0.90" claim was 30+ percentage points above
  the actual measurement on StatsBomb open-data. New text reports
  recall ~0.93, precision ~0.42, F1 ~0.58 (peak ~0.605 at
  `max_gap_seconds=10.0`) and explains why precision is the way it is
  (intrinsic to the team-change-with-carve-outs algorithm class, not a
  defect — StatsBomb's proprietary annotation merges brief opposing-
  team actions back into the containing possession; the heuristic
  cannot replicate that structurally).
- **e2e validation gate replaces F1 ≥ 0.80 with recall ≥ 0.85 AND
  precision ≥ 0.30 per match.** Recall enforces the helper's primary
  contract (catching every real boundary). Precision floor catches the
  "boundary cardinality halved or doubled" regression class that affects
  per-possession aggregation downstream. F1 stays in the assert message
  for diagnostics only — gating on F1 would re-introduce the
  misrepresentation problem this PR is fixing.
- **Test class renamed** `TestBoundaryF1AgainstStatsBombNative` →
  `TestBoundaryAgainstStatsBombNative`. Parametrized over the 3 vendored
  fixtures with per-match independent gates.

### Fixed
- **e2e regression coverage now actually runs in CI.** The previous
  `TestBoundaryF1AgainstStatsBombNative::test_boundary_f1_against_native_possession_id`
  was `@pytest.mark.e2e` and silently skipped on every CI run since
  1.2.0 because the fixture wasn't committed. Plus it was also skipping
  locally (the fixture was never on the user's only development
  machine). Net: ~6 release cycles of zero coverage on this test. PR-S8
  vendors the fixtures and drops the marker so the test runs on every
  PR + push.

### Notes
- The 5 `test_predict*` cases in `tests/vaep/`, `tests/test_xthreat.py`,
  and `tests/atomic/` continue to skip in CI (and locally) because they
  depend on the un-committed `tests/datasets/statsbomb/spadl-WorldCup-2018.h5`
  fixture. Closing that gap is queued as PR-S9 (generate the HDF5 from
  open-data raw events; commit + drop e2e markers). Tracked in
  `TODO.md`.
- Algorithmic precision improvement for `add_possessions` is queued as
  PR-S10 (look-ahead merge rules for brief opposing-team actions;
  re-measure `max_gap_seconds` defaults using the new
  `boundary_metrics` utility).

```

- [ ] **Step 3: Add `## Open PRs` section to TODO.md**

In `TODO.md`, append after the existing `## Architecture` section:

```markdown

## Open PRs

| # | Size | Item | Context |
|---|------|------|---------|
| PR-S9 | Medium | e2e prediction tests in CI via WorldCup HDF5 generation | Generate `tests/datasets/statsbomb/spadl-WorldCup-2018.h5` from open-data raw events (~64 matches × ~600 KB). Output structure: `games` table + `actions/game_<id>` per match (see `tests/vaep/test_vaep.py:48` for shape contract). Drop `@pytest.mark.e2e` on the 5 `test_predict*` cases. Conversion script committed at `scripts/build_worldcup_fixture.py`. Estimated 1-2 hours. See `docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md` § 10 for design notes. |
| PR-S10 | Medium-Large | `add_possessions` algorithmic precision improvement | Close the precision gap from ~42% toward 60-70% via brief-opposing-action merge rule, defensive-action class, and/or spatial continuity check. Plus re-measure `max_gap_seconds` parameter sweep using the new `boundary_metrics` utility before changing the default. New parameters likely: `merge_brief_opposing_actions`, `brief_action_window_seconds`. Atomic-SPADL counterpart must mirror any semantic change. Best done AFTER PR-S9 — 64-match WorldCup fixture is more reliable than PR-S8's 3-match set for parameter-tuning. |
```

- [ ] **Step 4: (Optional) Tighten CLAUDE.md "Testing" wording**

In `CLAUDE.md` (project root), find:

```
## Testing

```bash
python -m pytest tests/ -m "not e2e" -v --tb=short
```

e2e tests require dataset fixtures not committed to the repo.
```

Replace with:

```
## Testing

```bash
python -m pytest tests/ -m "not e2e" -v --tb=short
```

e2e tests require dataset fixtures not committed to the repo. Tests with
fixtures committed to the repo should not be marked e2e — they run in
the regular suite.
```

This is optional. Skip if you'd rather not bundle into PR-S8.

---

## Task 8: Verification gates — full local CI parity

**Files:** None (verification only).

- [ ] **Step 1: Re-confirm exact CI pin (in case shells were restarted)**

```bash
uv pip install --upgrade ruff==0.15.7 pyright==1.1.395 pandas-stubs==2.3.3.260113
```

- [ ] **Step 2: Ruff lint**

```bash
uv run ruff check silly_kicks/ tests/
```

Expected: zero errors. If any, fix and re-run.

- [ ] **Step 3: Ruff format check**

```bash
uv run ruff format --check silly_kicks/ tests/
```

Expected: zero formatting issues. If any, run `uv run ruff format silly_kicks/ tests/` and re-check.

- [ ] **Step 4: Pyright**

```bash
uv run pyright silly_kicks/
```

Expected: zero errors. Common gotchas to watch for in this PR:
- `BoundaryMetrics(precision=..., recall=..., f1=...)` — pyright should accept the construction since `BoundaryMetrics` is a `TypedDict`.
- `pd.Series.shift(1)` typing — pandas-stubs generally returns `Series` from `shift`, so `.iloc[1:].to_numpy()` chain should resolve cleanly.

If errors surface, fix the source (do NOT add `# type: ignore` comments unless absolutely necessary; if unavoidable, comment with `[reportXxx]` specificity per existing style in `silly_kicks/spadl/utils.py:719,752`).

- [ ] **Step 5: Full pytest suite**

```bash
uv run pytest tests/ -v --tb=short 2>&1 | tail -50
```

Expected:
- All existing tests pass.
- 11 new TestBoundaryMetrics unit tests pass.
- 3 new TestBoundaryAgainstStatsBombNative parametrized tests pass.
- 5 `test_predict*` cases skip (PR-S9 territory).
- Zero failures, zero errors.

If any pass/skip count looks off, STOP and investigate before commit.

---

## Task 9: `/final-review` skill + single commit + user-gated push/PR/merge/tag

**Files:** All accumulated changes commit together.

- [ ] **Step 1: Run /final-review skill**

Invoke the `mad-scientist-skills:final-review` skill (per `feedback_commit_policy` memory: mandatory before commit, not just pre-PR).

```
/mad-scientist-skills:final-review
```

Expected: passes with no critical issues. Address any flagged issues inline before proceeding to commit. The C4 architecture diagram regen, if triggered, is part of this step.

- [ ] **Step 2: Stage all changes — explicit user approval gate**

STOP HERE. Surface to the user:
> "All implementation tasks complete + verification gates green + /final-review passed. Ready to stage + commit. Approve?"

Wait for explicit user approval. Do NOT proceed without it.

- [ ] **Step 3: Stage + commit (single commit, on user approval)**

```bash
git add silly_kicks/spadl/utils.py silly_kicks/spadl/__init__.py tests/spadl/test_add_possessions.py tests/datasets/statsbomb/raw/events/7298.json tests/datasets/statsbomb/raw/events/7584.json tests/datasets/statsbomb/raw/events/3754058.json tests/datasets/statsbomb/README.md pyproject.toml CHANGELOG.md TODO.md docs/superpowers/specs/2026-04-29-recall-based-add-possessions-validation-design.md docs/superpowers/plans/2026-04-29-recall-based-add-possessions-validation.md
```

If `CLAUDE.md` was also touched in Task 7 step 4, add it to the `git add` list.

```bash
git commit -m "$(cat <<'EOF'
feat(spadl): public boundary_metrics utility + recall-based add_possessions CI gate — silly-kicks 1.8.0

- New public silly_kicks.spadl.boundary_metrics(*, heuristic, native) — BoundaryMetrics TypedDict return, keyword-only args, length-mismatch validation, degenerate-input safe. Re-exported from silly_kicks.spadl.
- add_possessions docstring now honest about empirical baselines (recall ~0.93, precision ~0.42, F1 ~0.58); replaces the prior "boundary-F1 ~0.90" claim that was 30+pp above measured.
- 3-fixture parametrized regression gate (matches 7298/7584/3754058 from StatsBomb open-data, vendored under tests/datasets/statsbomb/raw/events/) — recall >= 0.85 AND precision >= 0.30 per match. F1 in assert message only.
- Drop @pytest.mark.e2e on the rewritten test (fixtures are committed). The previous test was silently skipping in CI for ~6 release cycles.
- Algorithm itself unchanged. PR-S9 (e2e prediction tests in CI) and PR-S10 (algorithmic precision improvement) tracked in TODO.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: commit succeeds, no pre-commit hook failure. If a hook fails, fix the underlying issue and create a NEW commit (per `feedback_commit_policy` — never amend; new commit covers the fix on top of the original work, since the failed commit didn't actually create a commit).

- [ ] **Step 4: Verify commit**

```bash
git log --oneline -1
git show --stat HEAD
```

Expected: single commit with the title above + the file list (~12 files including the 3 fixture JSONs).

- [ ] **Step 5: Push — explicit user approval gate**

STOP HERE. Surface to the user:
> "Commit created: `<hash> feat(spadl): ...`. Ready to push to origin? `git push -u origin feat/recall-based-add-possessions-validation`"

Wait for explicit user approval.

- [ ] **Step 6: Push (on user approval)**

```bash
git push -u origin feat/recall-based-add-possessions-validation
```

- [ ] **Step 7: Open PR — explicit user approval gate**

STOP HERE. Surface to the user:
> "Branch pushed. Ready to open PR? Title: `feat(spadl): public boundary_metrics utility + recall-based add_possessions CI gate — silly-kicks 1.8.0`"

Wait for explicit user approval.

- [ ] **Step 8: Create PR (on user approval)**

```bash
gh pr create --title "feat(spadl): public boundary_metrics utility + recall-based add_possessions CI gate — silly-kicks 1.8.0" --body "$(cat <<'EOF'
## Summary

- New public `silly_kicks.spadl.boundary_metrics(*, heuristic, native) -> BoundaryMetrics` for computing precision/recall/F1 between two possession-id sequences. TypedDict return, keyword-only args, length-mismatch validation, degenerate-input safe.
- `add_possessions` docstring rewritten to publish honest empirical baselines (recall ~0.93, precision ~0.42, F1 ~0.58) measured across 3 StatsBomb open-data matches. Replaces the prior "boundary-F1 ~0.90" claim that was 30+ percentage points above actual.
- 3-fixture parametrized regression gate (`recall ≥ 0.85 AND precision ≥ 0.30` per match) replaces a single-fixture F1 ≥ 0.80 e2e test that was silently skipping in CI for ~6 release cycles. Fixtures vendored under `tests/datasets/statsbomb/raw/events/` (~1.8 MB total, StatsBomb open-data license).
- `add_possessions` algorithm itself is unchanged. Two follow-ups queued in `TODO.md` (PR-S9: e2e prediction tests in CI via WorldCup HDF5 generation; PR-S10: algorithmic precision improvement using the new `boundary_metrics` utility).

Design + plan committed under `docs/superpowers/{specs,plans}/2026-04-29-recall-based-add-possessions-validation*`.

## Test plan

- [x] 11 new `TestBoundaryMetrics` unit tests (Contract / Correctness / Degenerate / Errors)
- [x] 3 new parametrized `TestBoundaryAgainstStatsBombNative` tests pass with comfortable margin
- [x] All existing tests pass (5 `test_predict*` continue to skip — PR-S9)
- [x] `uv run ruff check silly_kicks/ tests/` — zero errors
- [x] `uv run ruff format --check silly_kicks/ tests/` — zero formatting issues
- [x] `uv run pyright silly_kicks/` — zero errors (pin: pandas-stubs==2.3.3.260113 to match CI)
- [x] `/mad-scientist-skills:final-review` — passed

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 9: Wait for CI green, then merge — explicit user approval gate**

After the PR is open, CI runs. STOP HERE and surface to the user:
> "PR opened: <URL>. Waiting for CI. Once green, ready to squash-merge with `--admin --delete-branch`?"

Wait for CI green + explicit user approval. If CI fails, fix and push a new commit on the branch (per silly-kicks single-commit policy: amend the local commit, force-push the branch — only because the branch is private to this PR. NEVER force-push main.)

- [ ] **Step 10: Squash-merge (on user approval after CI green)**

```bash
gh pr merge --admin --squash --delete-branch
```

Expected: merge succeeds, branch deleted, local branch should be cleaned up:

```bash
git checkout main
git pull
git branch -d feat/recall-based-add-possessions-validation 2>/dev/null || true
```

- [ ] **Step 11: Tag + push tag — explicit user approval gate**

STOP HERE. Surface to the user:
> "Merge complete on main. Ready to tag v1.8.0 and push? This will auto-fire the PyPI publish workflow."

Wait for explicit user approval.

- [ ] **Step 12: Tag + push (on user approval) — PyPI auto-publish**

```bash
git tag v1.8.0
git push origin v1.8.0
```

Expected: tag push triggers the `publish.yml` GitHub Actions workflow, which builds the wheel and publishes to PyPI. Verify by checking the Actions tab and `https://pypi.org/project/silly-kicks/1.8.0/` after a few minutes.

- [ ] **Step 13: Update auto-memory — release state**

Update `C:\Users\Karsten\.claude\projects\D--Development-karstenskyt--silly-kicks\memory\project_release_state.md` to reflect 1.8.0 shipped (current version, latest commit hash on main, etc.). Memory should match observable repo state.

---

## Self-Review Checklist

Run after writing this plan, before handoff to executing-plans.

**1. Spec coverage:**
- ✅ § 4.1 file structure → Task 0 + each subsequent task touches the listed files
- ✅ § 4.2 boundary_metrics API → Task 2 (TypedDict + keyword-only + degenerate handling)
- ✅ § 4.3 add_possessions docstring update → Task 6
- ✅ § 4.4 e2e test rewrite → Task 5
- ✅ § 4.5 TestBoundaryMetrics class → Task 1 + 2
- ✅ § 5 pipeline → Task 5 (the parametrized test body matches the spec's pipeline)
- ✅ § 6 fixture vendoring → Task 4
- ✅ § 7 TDD ordering → Tasks 1-7 follow the spec's exact order
- ✅ § 8 verification gates → Task 8
- ✅ § 9 commit cycle → Task 9
- ✅ § 10 PR-S9/PR-S10 deferred → Task 7 step 3 (TODO.md update)
- ✅ § 11 risks → Task 5 step 3 surfaces tight-precision-margin warning
- ✅ § 12 acceptance criteria → mapped across Tasks 1-9

**2. Placeholder scan:** zero "TBD", "TODO" within the plan body. Each task has explicit code blocks. The "(Optional)" CLAUDE.md tightening in Task 7 step 4 is rejectable but explicit.

**3. Type consistency:** `BoundaryMetrics` is the TypedDict name throughout (Tasks 1, 2, 3, 7). `boundary_metrics` is the function name throughout. `TestBoundaryAgainstStatsBombNative` is the renamed test class throughout. No naming inconsistencies.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-29-recall-based-add-possessions-validation.md`. Per silly-kicks pattern (per `feedback_engineering_disciplines` memory: user finds subagent approval friction excessive), this plan is for **inline execution** via `superpowers:executing-plans` — not subagent-driven.
