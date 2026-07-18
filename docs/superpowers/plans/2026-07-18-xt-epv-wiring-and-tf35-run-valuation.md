# Real-xT EPV wiring + TF-35 run valuation — Implementation Plan (4.52.0)

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline execution
> is the house default — [[feedback_inline_execution_default]]; no worktrees, branch
> `pr-s119-xt-epv-and-run-values` off main @ ec543cc). Steps use checkbox (`- [ ]`) syntax.
>
> **Revision 3 (2026-07-18)** — plan-review rounds 1 and 2 applied. R2 blockers N1 (commit
> sequencing), N3 (tolerance), N4 (guard-collapse fork) fixed; N2 was retracted by the
> reviewer after verification, with its two surviving notes kept. S11–S16 applied. Every
> claim re-verified against the tree before applying (§Review log) — including two
> re-measurements that corrected numbers I had asserted.

**Goal:** Wire fitted `ExpectedThreat` models into the OBSO/space-creation/PAUSA EPV seam
(repairing two verified orientation defects en route) and ship TF-35 v1 off-ball-run
valuation — per the APPROVED spec
`docs/superpowers/specs/2026-07-18-xt-epv-wiring-and-tf35-run-valuation-design.md`.

**Architecture:** The xT storage y-inversion is neutralized ONCE, inside `xthreat` (whose
convention it is — review Fork A), and consumed from there. The obso family gains an opt-in
`xt=` kwarg plus a per-row `obso_epv_source` provenance column at the aggregator edge;
OBSO's missing orientation handling, player_influence's latent y-mirror, and
space_creation's axis-1-only opponent mirror are repaired as separately-committed fixes.
Part B adds `tracking/_run_values.py` sharing the TF-4 candidacy predicate.

**Tech stack:** pandas/numpy, scipy `RectBivariateSpline` (already a HARD base dep,
`pyproject.toml:35` + `xthreat/_model.py:8` — nothing new is added), house gates, pytest.

### Commit plan — ONE commit on ONE branch (owner policy, 2026-07-18)

**Single feature branch → ONE commit → PR.** This is the standing house rule
([[feedback_commit_policy]]: one commit per branch + explicit approval) and it OVERRIDES
review round-1's S9 split-commits recommendation and round-2's N1 sequencing note, both of
which assumed a multi-commit PR. Revisions 2 and 3 wrongly adopted that split; corrected
here. The commit is **approval-gated**: complete all tasks, run the gates, present the
diff, and commit only on explicit owner approval.

Task ORDER still matters even though the commit is single — not for bisectability but for
TDD correctness: the space-creation golden (Task 0 Step 3) MUST be captured on the
unmodified tree before Task 6 changes the mirror axis, and the orientation goldens must be
observed RED before Tasks 7/8 fix them.

| Tasks | Content | Value change (all lands in the one commit) |
|---|---|---|
| 0–5 | Fork A + guard collapse, cache `__len__`, `xt=` threading, warning categories, provenance column, PAUSA | **None** (additive) |
| 6 | space_creation opponent mirror → point reflection | None measurable (≤6e-16) |
| 7 | OBSO orientation repair (DEFECT A) | **Away-team obso/pausa/space-creation rows** |
| 8 | player_influence threat-grid fix (DEFECT B) | **Small, all rows** |
| 9–13 | Part B (TF-35) + TF-4 `toward_goal` re-key | New columns + TF-4 divergence rows |
| R1 | Release mechanics (version/CHANGELOG/TODO/ADRs/C4/e2e) | — |

Because it is one commit, the CHANGELOG and PR body carry the full Hyrum inventory (R1
Step 6) — that is where a downstream investigator looks, in place of `git bisect`.

**Gates:** `python -m pytest tests/ -m "not e2e" --tb=short` (capture the exit code — never
`| tail`, [[feedback_piped_test_runs_mask_exit_codes]]); `python -m ruff check .` and
`python -m ruff format --check .` separately; bare `python -m pyright` (whole repo).

**Mutation checks (review S7) — four required**, each "remove the transform → this named
test goes RED", with the RED output pasted into the PR notes: Task 1 (`flipud`), Task 6
(`axis=(0,1)`), Task 7 (EPV x-flip AND target reprojection = two checks), Task 8 (adapter
swap).

### Verified anchors (re-verified @ ec543cc across review rounds 1–2)

`ExpectedThreat.xT` shape (w=12, l=16), storage row 0 = top of pitch (`xthreat/_grid.py:16-26`;
`rate()` indexes `grid[w-1-yj]` at `_model.py:254`) · cell centres from `spadl.config`
(`_model.py:174-178`) · the fitted-guard triple exists TWICE: a named
`_require_fitted_xt` (`vaep/features/expected_threat.py:27-51`, called at `:81`, referenced
in a docstring at `:70`, **imported by nothing else — verified repo-wide grep**) and an
**INLINED** copy with byte-identical messages (`atomic/vaep/features.py:479-486`) ·
`_grid_value` lazy-import idiom (`_xt_gk.py:189-211`) · `ObsoParams(grid_nx=104, grid_ny=68)`
(`_obso.py:51-56`) · aggregator signatures (`features.py:4795, 4889, 5032, 5159, 5216, 5275`)
— **none accept `params=`** · `compute_pass_obso` DOES accept `params=` (`_obso.py:285-296`) ·
`acting_team_attacks_rtl` / `reproject_to_action_ltr` (`_action_orientation.py:31-114`;
**unresolvable direction → False, no flip** — `:41-47`) · TF-4 candidacy (`_off_ball_runs.py:117-137`),
toward-goal sign test (`:172-197`), `_validate_ltr` (`:37-59`) · `PitchControlCache` =
`__init__`/`surface`/`_key` + `self._store` ONLY (`pitch_control/_cache.py:30-125`) ·
`PitchControlSurface.player_surface` AND `player_share` both use raw
`np.where(self.player_ids == player_id)` (`_surface.py:140,167`) ·
`resolve_next_touch_receiver` (`spadl/utils.py:1292`) · `_OBSO_COLUMNS` pre-seed
`out[col] = np.nan` (`features.py:4862-4863`) and `_SPACE_CREATION_COLUMNS` pre-seed
(`:5114-5115`) · `tracking_default_xfns` = the 4 action_context lifts only (`features.py:460-465`).

**Measurements taken during review (numbers I had asserted, corrected):**

| Quantity | Asserted | **Measured** |
|---|---|---|
| `_make_synthetic_epv_grid` y-symmetry | exact | **exact** ✓ (tiled x-ramp) |
| `_make_synthetic_reachability_grid` y-symmetry | exact | **4.440892098500626e-16**, 3752/6400 cells |
| `_interpolate_grid` symmetry loss | 8.7e-15 | **3.331e-16** at the real (32,50) target; 5.551e-16 at (68,104) |
| Symmetrizing the source helps? | assumed yes | **No** — still 3.331e-16 post-resample |

**Gate reality (review B3, extended per the round-2 process rule).** Registration answers
*whether* a helper is exercised; the **"asserts what"** column answers *what is checked* —
the column whose absence produced the reviewer's (withdrawn) N2. Assertion bodies quoted
from source, not inferred:

| Gate | Auto-enumerates? | **Asserts what** (quoted) | Consequence for us |
|---|---|---|---|
| `test_add_star_purity.py` | **YES** — `__all__ ∪ features.__all__`, 2 meta-asserts (`:520-555`) | inputs unmutated by value + `out is not input`, per registered variant (`:502-506`) | new `add_*` forced; ≥2 variants for a new input-mode branch |
| `test_aggregator_column_liveness.py` | **YES** — pins `tracking.__all__` (`:397-402`) | `assert added` (≥1 new column, `:415`); `dead = [c for c in added if not out[c].notna().any()]` → **100%-null only** (`:416-420`); non-constant check gated on `pd.api.types.is_float_dtype(out[c])` **and** `notna().sum() >= 2` **and** `nunique() == 1` (`:443-456`), with a hardcoded `provenance` exempt-name set (`:432-441`) | new `add_*` forced. **`obso_epv_source` passes**: partial `pd.NA` is fine, and a `string` dtype never reaches the float-gated constant check — no exempt-set entry needed |
| `test_id_dtype_invariance.py` | **YES** — `dir(F)` (`:74-81`) | `assert_frame_equal(..., check_dtype=False)` across the 4 id-dtype permutations | new `add_*` forced. A call-level-constant string label is invariant by construction |
| `test_frame_aware_xfns_dup_action_id.py` | **YES** — `dir(F)` `*_xfns` + `>= 21` count floor + empty allowlist | each factory produces values (does not raise) on duplicate `action_id` gamestate slots | new factory forced |
| `test_enrichment_nan_safety.py` | **YES** — `@nan_safe_enrichment`-decorated helpers | return type + row count preserved on NaN-id input (the decorator itself is a pure marker, `_nan_safety.py`, no dtype logic) | forced |
| **`test_public_api_examples.py`** | **NO** — hardcoded `_PUBLIC_MODULE_FILES` (`:18-72`, 54 entries) | `_has_examples_section` = **substring** match on `"Examples\n---"` or `">>> "` (`:91-97`); **nothing is executed** (`:129`) | **add the 3 new module paths manually**, then write Examples sections |
| **`test_action_ltr_mirror_invariance.py`** | **NO** — 4 hand-written fns, no registry, no `__all__` meta-assert | per-column `abs(b - m) < tol` (default `tol=1e-6`) between base and mirrored runs (`:165`) | **write a 5th function** for obso |
| `_EXHAUSTIVE_EMITTED` (`test_add_star_purity.py:616-648`) | **NO** — opt-in dict, 3 keys, **none of ours** | `emitted == expected` (added-minus-`_PROVENANCE`) **and** every expected column named in the docstring (`:651-666`) | register `add_off_ball_run_values` deliberately |

**String-column note (surviving from the retracted N2).** `obso_epv_source` is the **first
non-numeric column any tracking aggregator emits** (verified: `grep 'dtype="string"'` and
`grep 'out\[.*\] = "'` over `features.py` both return zero). All five auto-enumerating gates
were verified clean for it: liveness fails only on 100%-null columns (`:414-418`) and its
non-constant check is float-gated (`:448`); `_EXHAUSTIVE_EMITTED` doesn't register our
aggregators; `nan_safe_enrichment` is a pure marker; id-dtype compares with
`check_dtype=False` and the label is call-level constant. **Two things must not be
"simplified" later** — see the code comments required in Task 3 Step 4.

---

## Task 0: Branch, baseline, and capture the mirror golden

- [ ] **Step 1:** `git checkout -b pr-s119-xt-epv-and-run-values` (from main @ ec543cc).
- [ ] **Step 2:** Baseline green:
  `python -m pytest tests/ -m "not e2e" -q --tb=short; echo EXIT=$?` → `EXIT=0`;
  `python -m ruff check .` clean; `python -m pyright` 0 errors.
- [ ] **Step 3 (S16 — capture on the UNMODIFIED tree, before any edit):** create
  `scripts/gen_space_creation_mirror_golden.py`, which builds the Task-6 fixture, runs
  `add_space_creation(actions, frames, home_team_id=1)`, and writes
  `tests/tracking/fixtures/space_creation_mirror_golden.npz` with arrays
  `space_created_m2`, `space_denied_m2_opponent` plus a `source_commit` string
  (`git rev-parse HEAD`). Run it and commit the npz with C1. **Rationale:** a wall of
  pasted floats in the test body is opaque and unmaintainable; an npz + generator records
  provenance and mirrors the house pattern (`scripts/gen_ghost_gk_kde_golden.py`). The
  fixture builder must live in a shared module both the script and the test import, so the
  golden and the assertion cannot drift.

---

## Part 1 — additive (Tasks 1–5): no value change

### Task 1: Fork A — `xthreat/_physical.py`; collapse BOTH guard copies

**Rationale (review Fork A, verified):** the y-inversion is *xthreat's* storage convention,
so the anti-corruption layer belongs on xthreat's boundary. The guard already exists twice —
a named function in `vaep` and a byte-identical **inlined** triple in `atomic.vaep` — so the
previous design would have shipped a third copy plus a test asserting the copies agree. No
obstacles: scipy is already a hard base dep; `silly_kicks/__init__.py` is inert; no test
pins the full message strings (only substrings `"fitted ExpectedThreat"` / `"bundled"` at
`tests/vaep/test_xt_feature.py:43,48`); no runtime `vaep→xthreat` edge to invert; `xtgk`
(×7 modules) and `calibration` already import xthreat internals at module level; the SK-xT-1
oracle pins VALUES, not module inventory. **Hard constraint:** the new module must NOT
import `vaep.feature_framework` (cycle).

**N4 resolution — no fork, no text change.** Revision 2 left "pick one of two binding
shapes at implementation time" (a placeholder) and proposed `caller="atomic xt_xfns"`
(which would alter shipped message text for no benefit). Both are fixed: **delete the
private `_require_fitted_xt` name entirely** and call
`require_fitted_xt(model, caller="xt_xfns")` directly at both call sites. Verified safe —
repo-wide grep shows `_require_fitted_xt` is referenced only inside `expected_threat.py`
itself. `caller="xt_xfns"` at BOTH sites reproduces today's messages **byte-identically**.
The test then asserts the *property* ("no second implementation exists"), not a binding.

**Files:** Create `silly_kicks/xthreat/_physical.py`; Modify `silly_kicks/xthreat/__init__.py`,
`silly_kicks/vaep/features/expected_threat.py`, `silly_kicks/atomic/vaep/features.py`,
`tests/test_public_api_examples.py`; Test: `tests/xthreat/test_physical.py`.

- [ ] **Step 1: Write the failing tests** (`tests/xthreat/test_physical.py`):

```python
"""Orientation goldens for the physical-coordinate adapters (ADR-041).

The fixture value EQUALS the physical y-centre of each storage band, so ANY y-mirror bug
returns 68 - y. A y-symmetric fixture would pass under the very bug these adapters exist
to prevent (feedback_symmetry_test_insufficient_pin_ground_truth).
"""

import ast
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from silly_kicks.xthreat import ExpectedThreat, physical_grid, require_fitted_xt, values_at_points


def _asymmetric_model() -> ExpectedThreat:
    m = ExpectedThreat()
    w, _l = m.xT.shape  # (12, 16)
    cell_w = 68.0 / w
    for i in range(w):
        # storage row 0 = TOP of pitch -> physical y-centre (w-1-i+0.5)*cell_w
        m.xT[i, :] = (w - 1 - i + 0.5) * cell_w
    return m


class TestPhysicalGrid:
    def test_orientation_golden_uniform(self):
        m = _asymmetric_model()
        gx = (np.arange(104) + 0.5) * (105.0 / 104)
        gy = (np.arange(68) + 0.5) * (68.0 / 68)
        out = physical_grid(m, gx, gy)
        assert out.shape == (68, 104)
        interior = (gy > 68.0 / 12) & (gy < 68.0 - 68.0 / 12)
        np.testing.assert_allclose(out[interior, 0], gy[interior], atol=1e-9)

    def test_orientation_golden_nonuniform_grid_y(self):
        """No symmetry precondition: the data-level flipud works on ANY ascending grid."""
        m = _asymmetric_model()
        out = physical_grid(m, np.array([10.0, 52.5, 95.0]), np.array([10.0, 20.0, 55.0]))
        np.testing.assert_allclose(out[:, 0], np.array([10.0, 20.0, 55.0]), atol=1e-9)

    def test_corner_cells_pinned(self):
        m = ExpectedThreat()
        m.xT[:] = 0.0
        m.xT[0, 15] = 0.9   # storage top-right   -> physical (x~101.7, y~65.2)
        m.xT[11, 0] = 0.2   # storage bottom-left -> physical (x~3.3,  y~2.83)
        gx = np.array([105.0 / 16 * 0.5, 105.0 / 16 * 15.5])
        gy = np.array([68.0 / 12 * 0.5, 68.0 / 12 * 11.5])
        out = physical_grid(m, gx, gy)
        assert out[1, 1] == pytest.approx(0.9)  # high y, high x
        assert out[0, 0] == pytest.approx(0.2)  # low y,  low x

    def test_rejects_non_ascending_grid(self):
        m = _asymmetric_model()
        with pytest.raises(ValueError, match="ascending"):
            physical_grid(m, np.array([50.0, 10.0]), np.array([10.0, 50.0]))


class TestValuesAtPoints:
    def test_matches_rate_exactly(self):
        m = _asymmetric_model()
        actions = pd.DataFrame(
            {
                "type_id": [0, 0],
                "result_id": [1, 1],
                "start_x": [10.0, 60.0],
                "start_y": [10.0, 50.0],
                "end_x": [30.0, 80.0],
                "end_y": [40.0, 20.0],
            }
        )
        expected = values_at_points(m, actions["end_x"], actions["end_y"]) - values_at_points(
            m, actions["start_x"], actions["start_y"]
        )
        np.testing.assert_allclose(m.rate(actions), expected, atol=1e-12)

    def test_nan_coords_are_nan(self):
        m = _asymmetric_model()
        out = values_at_points(m, np.array([np.nan, 10.0]), np.array([34.0, np.nan]))
        assert np.isnan(out).all()


class TestRequireFittedXt:
    @pytest.mark.parametrize(
        ("bad", "exc"),
        [("default", NotImplementedError), (None, ValueError), ("UNFITTED", NotFittedError)],
    )
    def test_triple(self, bad, exc):
        model = ExpectedThreat() if bad == "UNFITTED" else bad
        with pytest.raises(exc):
            require_fitted_xt(model, caller="probe")

    def test_messages_are_byte_identical_to_the_shipped_text(self):
        """caller="xt_xfns" must reproduce today's messages EXACTLY (N4)."""
        with pytest.raises(NotImplementedError) as e1:
            require_fitted_xt("default", caller="xt_xfns")
        assert str(e1.value) == (
            "xt_xfns: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat."
        )
        with pytest.raises(ValueError) as e2:
            require_fitted_xt(None, caller="xt_xfns")
        assert str(e2.value) == "xt_xfns requires a fitted ExpectedThreat (model=...)."
        with pytest.raises(NotFittedError) as e3:
            require_fitted_xt(ExpectedThreat(), caller="xt_xfns")
        assert str(e3.value) == (
            "xt_xfns requires a fitted ExpectedThreat; call model.fit(actions) first."
        )


class TestNoSecondImplementation:
    """Property, not binding (N4): the guard logic exists in exactly ONE module."""

    _FRAGMENT = "bundled xT grid variants are not shipped yet"

    def test_only_physical_module_contains_the_guard_text(self):
        root = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks"
        owners = sorted(
            p.relative_to(root).as_posix()
            for p in root.rglob("*.py")
            if self._FRAGMENT in p.read_text(encoding="utf-8")
        )
        assert owners == ["xthreat/_physical.py"], f"guard duplicated into: {owners}"

    def test_call_sites_delegate(self):
        """Both xt_xfns factories must route through the shared guard."""
        for mod in ("silly_kicks/vaep/features/expected_threat.py",
                    "silly_kicks/atomic/vaep/features.py"):
            src = (pathlib.Path(__file__).resolve().parents[2] / mod).read_text(encoding="utf-8")
            assert "require_fitted_xt(model, caller=\"xt_xfns\")" in src
            assert "_require_fitted_xt" not in src, f"{mod}: private guard name survived"
```

- [ ] **Step 2:** Run: `python -m pytest tests/xthreat/test_physical.py -v` → FAIL
  (`ImportError: cannot import name 'physical_grid'`).
- [ ] **Step 3: Implement** `silly_kicks/xthreat/_physical.py` — module docstring, then:

```python
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
from sklearn.exceptions import NotFittedError

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xthreat._grid import _get_cell_indexes

if TYPE_CHECKING:
    from silly_kicks.xthreat._model import ExpectedThreat

__all__ = ["physical_grid", "require_fitted_xt", "values_at_points"]


def require_fitted_xt(model: "ExpectedThreat | str | None", *, caller: str) -> None:
    """Fail closed unless ``model`` is a fitted ExpectedThreat.

    The single source for this guard: ``vaep.features.xt_xfns``, its atomic mirror and the
    tracking adapters all call it (ADR-041 -- two copies collapsed).

    Parameters
    ----------
    model : ExpectedThreat | str | None
        The candidate xT model.
    caller : str
        Public name used in the messages (e.g. ``"xt_xfns"``, ``"add_obso"``).

    Raises
    ------
    NotImplementedError
        If ``model`` is a ``str`` (a future bundled-variant name; not shipped yet).
    ValueError
        If ``model`` is ``None``.
    NotFittedError
        If ``model`` is an unfitted ExpectedThreat (all-zero ``.xT``).

    Examples
    --------
    Guard a caller-supplied model::

        from silly_kicks.xthreat import require_fitted_xt

        require_fitted_xt(model, caller="my_feature")
    """
    if isinstance(model, str):
        raise NotImplementedError(
            f"{caller}: bundled xT grid variants are not shipped yet; pass a fitted ExpectedThreat."
        )
    if model is None:
        raise ValueError(f"{caller} requires a fitted ExpectedThreat (model=...).")
    if not np.any(model.xT):  # same fitted-check ExpectedThreat.rate() uses
        raise NotFittedError(f"{caller} requires a fitted ExpectedThreat; call model.fit(actions) first.")
```

  plus `physical_grid(model, grid_x, grid_y)` — validate 1-D strictly-ascending grids
  (`ValueError` naming "ascending"), derive cell centres from `spadlconfig` exactly as
  `_model.py:174-178`, then:

```python
    phys = np.flipud(xT)  # row 0 = y=0: THE inversion-neutralization point
    spline = RectBivariateSpline(x_centres, y_centres, phys.T, kx=1, ky=1)
    return np.asarray(spline(gx, gy)).T  # (ny, nx), ascending-y rows
```

  and `values_at_points(model, x, y)` — nearest-cell lookup via `_get_cell_indexes` with the
  `(n_rows-1)-yj` inversion, NaN→NaN (the `_xt_gk._grid_value` body, now natively inside
  xthreat so no lazy private cross-package import is needed). Both call
  `require_fitted_xt(..., caller="physical_grid" | "values_at_points")`. Every public symbol
  carries an `Examples` section.

  **Do NOT copy `_model.py:171`'s `if RectBivariateSpline is None:` guard** — unreachable in
  production (scipy is a hard dep), exercised only by a monkeypatch; shipping an unexercised
  copy violates [[feedback_guard_expectation_must_be_independent_of_guarded_input]].

- [ ] **Step 4: Export** — `xthreat/__init__.py`: `from silly_kicks.xthreat._physical import
  physical_grid, require_fitted_xt, values_at_points`; add the three names to `__all__`
  (RUF022: snake_case block, alphabetical).
- [ ] **Step 5: Collapse copy #1** — `vaep/features/expected_threat.py`: delete the
  `_require_fitted_xt` def (`:27-51`), replace the call at `:81` with
  `require_fitted_xt(model, caller="xt_xfns")` (module-level
  `from silly_kicks.xthreat import require_fitted_xt` — verified cycle-free and matching
  `xtgk`'s shipped practice), and update the docstring cross-reference at `:70` to
  `:func:`silly_kicks.xthreat.require_fitted_xt``.
- [ ] **Step 6: Collapse copy #2** — `atomic/vaep/features.py`: replace the **inlined**
  triple at `:479-486` with the same single call, `caller="xt_xfns"` (messages byte-identical
  to what it raises today).
- [ ] **Step 7: Examples gate (B3)** — add `"silly_kicks/xthreat/_physical.py"` to
  `_PUBLIC_MODULE_FILES` (`tests/test_public_api_examples.py:18-72`, the xthreat block at
  `:34-38`).
- [ ] **Step 8:** Run:
  `python -m pytest tests/xthreat/ tests/vaep/test_xt_feature.py tests/atomic/test_atomic_xt_feature.py tests/test_public_api_examples.py -v`
  → PASS.
- [ ] **Step 9 (MUTATION CHECK — observe RED, revert, do not commit):** remove `np.flipud`
  → `test_orientation_golden_uniform` FAILS returning `68 - y`. Paste RED into PR notes.

### Task 2: `PitchControlCache.__len__` (review B2)

**Files:** Modify `pitch_control/_cache.py`; Test: `tests/tracking/test_pitch_control_cache.py`.

- [ ] **Step 1: Failing test:**

```python
def test_cache_exposes_size(single_frame):
    cache = PitchControlCache()
    assert len(cache) == 0
    cache.surface(single_frame, attacking_team_id=1, decompose=True)
    assert len(cache) == 1
    cache.surface(single_frame, attacking_team_id=1, decompose=True)  # same key
    assert len(cache) == 1
```

- [ ] **Step 2:** Run → FAIL (`TypeError: object of type 'PitchControlCache' has no len()`).
- [ ] **Step 3: Implement** (this PR is the FOURTH family threading the cache; a public size
  is the honest observable — `_store` pins a private, a spy on `compute_pitch_control` tests
  that PC ran rather than that the cache was shared):

```python
    def __len__(self) -> int:
        """Number of memoized surfaces (canonical frames only).

        Examples
        --------
        >>> cache = PitchControlCache()
        >>> len(cache)
        0
        """
        return len(self._store)
```

- [ ] **Step 4:** Run → PASS.

### Task 3: `xt=` threading, warning categories, `obso_epv_source` provenance

**Files:** Create `silly_kicks/tracking/_warnings.py`; Modify `tracking/features.py`,
`tracking/_obso.py`, `tracking/__init__.py`, `pyproject.toml`,
`tests/test_public_api_examples.py`; Test: `tests/tracking/test_obso_xt_wiring.py`.

- [ ] **Step 1: Failing tests** — mutual exclusion, category-typed synthetic warning, no
  warning when `xt=` supplied, real-xT ≠ synthetic (non-vacuous), and the provenance column
  incl. the `links=`-supplied case:

```python
def test_epv_source_provenance_column(obso_actions, obso_frames):
    """The DATA records which surface was used -- a warning cannot survive into a mart (S3)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntheticEPVWarning)
        syn = F.add_obso(obso_actions, obso_frames)
    xtd = F.add_obso(obso_actions, obso_frames, xt=_fitted_xt())
    inj = F.add_obso(obso_actions, obso_frames, epv_grid=np.full((68, 104), 0.2))
    rows = syn["obso_actual"].notna()
    assert (syn.loc[rows, "obso_epv_source"] == "synthetic").all()
    assert (xtd.loc[rows, "obso_epv_source"] == "xt").all()
    assert (inj.loc[rows, "obso_epv_source"] == "injected").all()
    assert str(syn["obso_epv_source"].dtype) == "string"   # extension dtype, NOT object


def test_epv_source_present_even_with_supplied_links(obso_actions, obso_frames, obso_links):
    """Must NOT live inside the `links is None` provenance branch (verified trap)."""
    xtd = F.add_obso(obso_actions, obso_frames, xt=_fitted_xt(), links=obso_links)
    assert (xtd.loc[xtd["obso_actual"].notna(), "obso_epv_source"] == "xt").all()


def test_space_creation_shares_the_same_provenance_name(sc_actions, sc_frames):
    """S14: ONE name across both aggregators -- a consumer joining them gets no conflict."""
    out = F.add_space_creation(sc_actions, sc_frames, home_team_id=1, xt=_fitted_xt())
    assert (out.loc[out["space_created_m2"].notna(), "obso_epv_source"] == "xt").all()
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `tracking/_warnings.py` with `SyntheticEPVWarning(UserWarning)`
  and `IgnoredSurfaceInputsWarning(UserWarning)` (review S2: separate categories, so
  silencing the routine synthetic notice cannot also silence a genuine misuse signal), each
  with an `Examples` section showing a `filterwarnings` line. These are the package's first
  public Warning subclasses (verified: only two module-private `IntegrityError` classes
  existed) — the module docstring states that this is the new convention. Then in
  `features.py`:

```python
def _resolve_epv_grid(xt, epv_grid, *, caller: str, params=None) -> tuple["np.ndarray | None", str]:
    """Resolve xt=/epv_grid= to (grid, source_label). The CALLER emits the warning (S4).

    ``caller`` is the public function name and appears in user-facing messages (S12).
    ``params`` selects the build geometry; None -> default ObsoParams. The aggregators
    expose no ``params`` kwarg (verified), so they pass None; ``compute_pass_obso`` passes
    its OWN resolved params so a non-default geometry is honoured (Fork-B narrow fix).
    """
    from silly_kicks.xthreat import physical_grid, require_fitted_xt

    from ._obso import ObsoParams

    if xt is not None and epv_grid is not None:
        raise ValueError(f"{caller}: pass either xt= or epv_grid=, not both")
    if xt is not None:
        require_fitted_xt(xt, caller=caller)
        p = params or ObsoParams()
        gx = (np.arange(p.grid_nx) + 0.5) * (p.pitch_length / p.grid_nx)
        gy = (np.arange(p.grid_ny) + 0.5) * (p.pitch_width / p.grid_ny)
        return physical_grid(xt, gx, gy), "xt"
    if epv_grid is not None:
        return epv_grid, "injected"
    return None, "synthetic"
```

  Each of `add_obso`, `obso_xfns`, `add_space_creation`, `space_creation_xfns`, `add_pausa`,
  `pausa_xfns` gains `xt: "ExpectedThreat | None" = None` and, as its first body statement,
  resolves with **its own name** as `caller=` (S12) and warns with `stacklevel=2` (S4):

```python
    epv_grid, _epv_source = _resolve_epv_grid(xt, epv_grid, caller="add_obso")
    if _epv_source == "synthetic":
        warnings.warn(
            "add_obso: OBSO EPV is the synthetic linspace(0.01, 0.3) placeholder ramp -- "
            "pass xt= (fitted ExpectedThreat) or epv_grid= for production surfaces.",
            SyntheticEPVWarning,
            stacklevel=2,
        )
```

  `compute_pass_obso` gains `xt=` resolved with `params=params or ObsoParams()` (the narrow
  Fork-B fix: **build at the EFFECTIVE geometry, never a hardcoded default**) and emits NO
  warning (engines stay silent; policy is the aggregator's).
- [ ] **Step 4: Provenance column** — in `add_obso` AND `add_space_creation`, seed
  `obso_epv_source` **separately from** the `np.nan` pre-seed loops (`features.py:4862-4863`
  and `:5114-5115`) and fill via `.loc`, with this comment verbatim (the retracted-N2
  survivor — a later "simplification" would silently produce an object column):

```python
    # NOTE: seeded SEPARATELY from the _OBSO_COLUMNS float pre-seed on purpose. Folding this
    # into that `out[col] = np.nan` loop and writing a str via .at upcasts to OBJECT dtype,
    # not pandas "string". Keep the pd.array(..., dtype="string") seed + .loc fill.
    # First non-numeric column emitted by a tracking aggregator (ADR-041).
    out["obso_epv_source"] = pd.array([pd.NA] * len(out), dtype="string")
```

  placed OUTSIDE the `if not has_provenance and links is None:` block at `:4872` (verified
  trap: a caller passing `links=` would otherwise never receive it). **S14 decision (no
  fork): ONE name, `obso_epv_source`, in both aggregators** — space-creation is
  OBSO-derived and both read the same injected grid, so a consumer joining the two gets no
  conflicting columns. Not added to any `*_xfns` output (string provenance is not a VAEP
  feature — the `_COORD_COLS` / `n_valued_*` precedent).
- [ ] **Step 5 (S11 — measure the blast radius, then scope):** run
  `python -m pytest tests/ -m "not e2e" -q -W error::silly_kicks.tracking.SyntheticEPVWarning --tb=no -q > /tmp/sweep.txt; echo EXIT=$?`
  and count the failing test MODULES. Then add to `pyproject.toml`'s
  `[tool.pytest.ini_options]` (verified: no `filterwarnings` key today, `:228-235`):

```toml
filterwarnings = [
    "error::silly_kicks.tracking.SyntheticEPVWarning",
    "error::silly_kicks.tracking.IgnoredSurfaceInputsWarning",
]
```

  and add `pytestmark = pytest.mark.filterwarnings("ignore::silly_kicks.tracking.SyntheticEPVWarning")`
  at MODULE level to exactly the modules the sweep named (module-level, not per-test, keeps
  the diff small). **Record the module list in the PR notes** — it is the audit of every
  call site that uses the synthetic surface, which is the point of S3. If the sweep names
  more than ~10 modules, keep the global entries but note in the PR that the opt-out list is
  the real inventory.
- [ ] **Step 6:** Iterate to `EXIT=0` on the full suite.
- [ ] **Step 7: Exports** — `tracking/__init__.py`: import both warnings from `._warnings`
  and add them to `__all__` in the **CamelCase class block** (RUF022 — not the tail). Add
  `silly_kicks/tracking/_warnings.py` to `_PUBLIC_MODULE_FILES`.
- [ ] **Step 8:** Run `python -m pytest tests/tracking/test_obso_xt_wiring.py tests/test_public_api_examples.py -v` → PASS.

### Task 4: PAUSA — cache threading + ignored-inputs warning

**Files:** Modify `features.py` (`add_pausa:5216-5258`, `pausa_xfns:5275`); Test: append to
`tests/tracking/test_obso_xt_wiring.py`.

- [ ] **Step 1: Failing tests:**

```python
def test_pausa_warns_when_surface_inputs_ignored(obso_actions, obso_frames):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntheticEPVWarning)
        enriched = F.add_obso(obso_actions, obso_frames)
    with pytest.warns(IgnoredSurfaceInputsWarning, match="ignored"):
        F.add_pausa(enriched, obso_frames, xt=_fitted_xt())


def test_pausa_threads_cache(obso_actions, obso_frames):
    from silly_kicks.tracking.pitch_control import PitchControlCache

    cache = PitchControlCache()
    F.add_pausa(obso_actions, obso_frames, xt=_fitted_xt(), pitch_control_cache=cache)
    assert len(cache) > 0        # Task 2's public observable
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** — `add_pausa` gains `pitch_control_cache=None`, forwards it and
  `xt=` into its internal `add_obso` call (`:5250-5258`), and warns
  `IgnoredSurfaceInputsWarning` before the columns-present short-circuit (`:5249`) when
  surface inputs were supplied but the obso columns already exist. Gate the SYNTHETIC
  warning so it fires only when PAUSA actually computes OBSO (no spurious double warning on
  the reuse path).
- [ ] **Step 4:** Run → PASS.

### Task 5: Part-1 closeout

- [ ] **Step 1:** Purity variants — add an `xt=`-supplied second variant for
  `tracking:add_obso`, `tracking:add_space_creation`, `tracking:add_pausa` in
  `tests/test_add_star_purity.py` (ADR-033 ≥2-variant contract for the new input-mode
  branch), via a module-level `_purity_fitted_xt()` helper.
- [ ] **Step 2:** Full gates → green.
- [ ] **Step 3:** gates green. Do NOT commit here -- this PR is ONE commit, taken after
  R1 with explicit owner approval. Continue to Task 6.

---

## Part 2 — Task 6: space_creation opponent mirror

**Review B1 (both defects confirmed; measurement reproduced exactly):** revision 1 asserted
symmetry of the RAW synthetic grids — the wrong object (`_space_creation.py:223-224` flips
the `_interpolate_grid`-RESAMPLED grids) — and the assertion is false today (measured
4.440892098500626e-16 over 3752/6400 cells; `np.linspace` ULP amplified by `exp()`).

**Resolution (deviates from the reviewer's recommendation — no production mutation):** the
measurements show the spec's gate is achievable *without* touching production numerics.
Post-resample asymmetry at the real (32,50) target is **3.331e-16**, and symmetrizing the
source does **not** reduce it (still 3.331e-16) — so the mutation would change 2997/6400
cells of a placeholder grid, perturbing every existing obso/space-creation golden's inputs,
for **zero** measured benefit. The reviewer's underlying demands (gate the OUTPUT, not the
raw grids; never pre-authorize an unmeasured mutation) are fully honored.

**N3 — tolerance corrected.** `space_created_m2` is an **area in m²** (cell area 4.4625 m²,
pitch 7140 m²), so an absolute `atol=1e-12` is far tighter than the computation's relative
float noise at those magnitudes — the reviewer's naive bound ≈4.2e-12 exceeds it. Use
**`rtol=1e-9, atol=0`**: it scales with magnitude, sits ~6–7 orders above the measured
3.331e-16 relative noise, and still separates a structural O(1) regression by ~9 orders.

**Files:** Modify `_space_creation.py:223-224`; Test: `tests/tracking/test_space_creation_mirror.py`.

- [ ] **Step 1: Write the gate BEFORE the change** (spec §A6 TDD ordering), reading the
  Task-0 npz — no pasted float walls (S16):

```python
"""Opponent-mirror upgrade gate (ADR-041 / spec A6).

Measured 2026-07-18 @ ec543cc: the synthetic EPV grid is EXACTLY y-symmetric (tiled
x-ramp); the synthetic reachability grid is y-symmetric to 4.44e-16 (linspace ULP,
3752/6400 cells); _interpolate_grid carries that to 3.331e-16 at the real (32,50) target
(symmetrizing the source does NOT improve it). So axis=1 -> axis=(0,1) is a no-op on the
synthetic path at the 1e-16 level, while a structural regression is O(1). space_created_m2
is an AREA in m^2, so the tolerance is RELATIVE (N3): rtol=1e-9, atol=0.
"""

GOLDEN = np.load(Path(__file__).parent / "fixtures" / "space_creation_mirror_golden.npz")


def test_opponent_mirror_output_unchanged_on_synthetic_path():
    """THE gate: output pinned to values captured on the PRE-change tree (Task 0)."""
    actions, frames = build_mirror_fixture()          # shared builder, also used by the generator
    out = F.add_space_creation(actions, frames, home_team_id=1)
    np.testing.assert_allclose(
        out["space_created_m2"].to_numpy(dtype=float), GOLDEN["space_created_m2"],
        rtol=1e-9, atol=0,
    )
    np.testing.assert_allclose(
        out["space_denied_m2_opponent"].to_numpy(dtype=float), GOLDEN["space_denied_m2_opponent"],
        rtol=1e-9, atol=0,
    )


def test_synthetic_grid_symmetry_noise_floor_is_documented():
    """Pins the measured floor: a STRUCTURAL symmetry break fails here, not in the gate above."""
    epv = _make_synthetic_epv_grid()
    reach = _make_synthetic_reachability_grid()
    np.testing.assert_array_equal(epv, np.flipud(epv))            # exactly symmetric
    assert np.abs(reach - np.flipud(reach)).max() < 1e-14
    resampled = _interpolate_grid(reach, (32, 50))
    assert np.abs(resampled - np.flipud(resampled)).max() < 1e-14
```

- [ ] **Step 2:** Run → PASS on the unmodified tree (it pins current behaviour by construction).
- [ ] **Step 3: Implement** `_space_creation.py:223-224`:

```python
        # Point reflection (ADR-041): the opponent attacks the other goal AND the y-axis
        # mirrors with it. Equivalent to the previous axis=1 flip for the y-symmetric
        # synthetic grids (gated at rtol=1e-9 in test_space_creation_mirror.py); CORRECT
        # for an injected, y-asymmetric xT-derived surface. NOTE distance_weight is
        # deliberately NOT mirrored (ball-anchored, 4.24.0) -- do not "fix" it.
        transition_opp = np.flip(transition_interp, axis=(0, 1))
        epv_opp = np.flip(epv_interp, axis=(0, 1))
```

- [ ] **Step 4:** Re-run the file → still PASS. Then
  `python -m pytest tests/tracking/test_space_creation.py tests/tracking/test_pausa.py tests/tracking/test_obso.py -q --tb=short` → PASS.
- [ ] **Step 5: Add the asymmetric-injection discriminator** (proves the change is not
  vacuous): `epv = np.zeros((68,104)); epv[10, 10] = 1.0` (low-y, low-x); call
  `compute_space_created(..., epv_grid=epv, include_opponent_perspective=True)` and assert
  the opponent multiplier's mass sits at HIGH-y/HIGH-x, explicitly asserting it is NOT at
  low-y/high-x (where axis=1-only would put it).
- [ ] **Step 6 (MUTATION CHECK):** revert to `axis=1` → Step 5 goes RED (Step 1's gate will
  NOT — that is exactly why Step 5 exists). Restore.
- [ ] **Step (no commit):** gates green -> continue to Task 7. The single approval-gated commit happens after R1.

**Chesterton note (carried into the code comment):** `distance_weight` (`:205-209`) is
deliberately un-mirrored (ball-anchored) and the two branches normalize by their own maxima
(`max_trans` vs `max_trans_opp`) — the 4.24.0 fix stopping the opponent LOO from being the
pointwise negation of the team LOO. Do not touch either.

---

## Part 3 — Task 7: DEFECT-A repair, OBSO orientation

**Files:** Modify `features.py:4661-4792` (`_precompute_obso_lookup`), `_obso.py`
(`compute_pass_obso` docstring contract); Test: `tests/tracking/test_obso_orientation.py`
+ a **fifth hand-written function** in `tests/tracking/test_action_ltr_mirror_invariance.py`
(no registry — helpers `_mirror` at `:149`, `_assert_invariant(base, mir, aid, cols, *, tol=1e-6)`
at `:165`).

- [ ] **Step 1: Failing ground-truth tests.** Fixture: team 1 home (`team_attacking_direction`
  "ltr"), team 2 away ("rtl"). Two AWAY passes with mirrored action-LTR targets — action 10
  forward (`end_x=90`), action 11 backward (`end_x=15`), identical otherwise. Pre-fix the +x
  ramp rewards frame-high-x (the away team's OWN goal), so backward scores higher:

```python
def test_away_forward_pass_beats_backward_pass(away_fixture):
    actions, frames = away_fixture
    out = F.add_obso(actions, frames, home_team_id=1)
    o = out.set_index(actions["action_id"])["obso_actual"]
    assert o.loc[10] > o.loc[11]     # PRE-FIX THIS INVERTS


def test_away_target_is_sampled_at_the_reprojected_point(away_target_fixture):
    """The only well-controlled cluster sits at FRAME (15, 48) = action-LTR (90, 20)."""
    actions, frames = away_target_fixture
    out = F.add_obso(actions, frames, home_team_id=1)
    assert out.loc[0, "obso_actual"] > out.loc[1, "obso_actual"]
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** in `_precompute_obso_lookup`, after `fid_by_pos` (`:4697`):

```python
    # ADR-028 (ADR-041 amendment): frames are home-attacks-right; actions are
    # acting-team-LTR. For an RTL-acting team both (a) the action-LTR target must be
    # point-reflected into frame coords and (b) the attack-LTR EPV grid must be x-flipped
    # into frame orientation. transition_grid is orientation-neutral (ball-anchored).
    from ._action_orientation import acting_team_attacks_rtl
    from ._obso import _get_default_grids

    flip_rtl = acting_team_attacks_rtl(actions, frames).to_numpy(dtype=bool)
    transition_grid, epv_grid = _get_default_grids(transition_grid, epv_grid)
    epv_grid_rtl = np.ascontiguousarray(epv_grid[:, ::-1])
```

  and inside the loop, AFTER the existing NaN guard at `:4728`:

```python
        if flip_rtl[i]:
            target_x = 105.0 - float(target_x)
            target_y = 68.0 - float(target_y)
            epv_for_action = epv_grid_rtl
        else:
            epv_for_action = epv_grid
```

  passing `epv_grid=epv_for_action` at `:4778`. `compute_pass_obso`'s docstring gains the
  explicit contract: **grids and `target_position` are in the FRAMES' coordinate
  convention**; per-action orientation is the aggregator's job. `add_obso`'s `home_team_id`
  docstring: orientation is keyed on `team_attacking_direction`; the parameter is retained
  for signature stability.
- [ ] **Step 4:** Run
  `python -m pytest tests/tracking/test_obso_orientation.py tests/tracking/test_obso.py tests/tracking/test_pausa.py tests/tracking/test_space_creation.py -v`
  → PASS. **Home-team obso values must be UNCHANGED** (home rows never flip) — if an
  existing home-row assertion moves, stop and diagnose.
- [ ] **Step 5: Write the fifth mirror-invariance function** in
  `test_action_ltr_mirror_invariance.py`, copying `test_defensive_line_mirror_invariant`'s
  shape with `_mirror` + `_assert_invariant(base, mir, aid, ["obso_actual", "obso_peak",
  "obso_optimal"], tol=1e-6)`. Run the file → PASS.
- [ ] **Step 6 (MUTATION CHECKS ×2):** (a) drop the EPV flip → Step 1's first test RED;
  (b) drop the target reprojection → Step 1's second test RED. Restore; paste RED output.
- [ ] **Step (no commit):** gates green -> continue to Task 8. The single approval-gated commit happens after R1.

---

## Part 4 — Task 8: DEFECT-B repair, player_influence

**Files:** Modify `_player_influence.py:116-122`; Test: `tests/tracking/test_player_influence_orientation.py`.

- [ ] **Step 1: Failing golden** — asymmetric model (value ≡ physical y); two same-team
  outfielders, one high-y one low-y, otherwise identical; home team attacking (no x-flip, so
  the y axis is isolated):

```python
def test_high_y_player_receives_high_y_threat(pi_fixture):
    frame, xt, HIGH_Y_PID, LOW_Y_PID = pi_fixture
    out = compute_player_influence(frame, xt, attacking_team_id=1, home_team_id=1)
    assert out[HIGH_Y_PID].off_ball_xt > out[LOW_Y_PID].off_ball_xt   # PRE-FIX: inverted
```

- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** — `_player_influence.py:117-118`:

```python
    # Physically-oriented (ascending-y) threat grid -- ADR-041. The raw xt.interpolator()
    # output preserves xT's INVERTED row storage, which silently y-mirrored this fusion
    # (invisible only because a fitted xT surface is near-y-symmetric).
    threat_grid = physical_grid(xt, pc.grid_x, pc.grid_y)  # (ny, nx)
```

  (module-level `from silly_kicks.xthreat import physical_grid`; the away x-flip at
  `:121-122` UNCHANGED). Docstring gains (a) the frames-are-home-attacks-right contract
  sentence (the x-flip key `same_id(attacking_team_id, home_team_id)` is only correct under
  it) and (b) that unfitted `xt` now raises `NotFittedError` via the shared guard.
- [ ] **Step 4:** Run
  `python -m pytest tests/tracking/test_player_influence_orientation.py tests/tracking/test_player_influence.py tests/tracking/test_gk_influence.py -q --tb=short`
  → PASS. Existing numeric assertions may shift slightly (near-symmetric fixtures); each
  update is evidence of the fixed defect — record them for the CHANGELOG.
- [ ] **Step 5 (MUTATION CHECK):** revert to `xt.interpolator(...)` → Step 1 RED. Restore.
- [ ] **Step (no commit):** gates green -> continue to Task 9. The single approval-gated commit happens after R1.

---

## Part 5 — TF-35 v1 (Tasks 9–13)

### Task 9: shared candidacy predicate + TF-4 `toward_goal` re-key

**S13 — corrected: the divergence IS reachable, so this is NOT a pure no-op.** Revision 2
claimed re-keying was a no-op because `_validate_ltr` rejects unoriented frames and
home ⟺ "ltr". But `acting_team_attacks_rtl` returns **False (no flip) when the acting team's
direction is unresolvable** in that period (`_action_orientation.py:41-47` — team absent from
the frames, or all-null direction), whereas home-keying *always* resolves and would flip an
away team. So the two authorities disagree exactly on actions whose acting team has no
non-ball frame row carrying a direction. Those rows exist in real data (a team briefly absent
from a broadcast frame window). **Disposition:** adopt the direction authority anyway (one
authority is the goal, and its fallback is the conservative "don't flip"), but treat it as a
**behaviour change with a test**, not a no-op, and say so in the CHANGELOG. Verified NOT a
retrain trigger: `off_ball_context_xfns` is absent from `tracking_default_xfns`
(`features.py:460-465`, the 4 action_context lifts only — the reviewer confirmed this).

**Files:** Modify `_off_ball_runs.py`; Test: `tests/tracking/test_off_ball_runs_orientation.py`.

- [ ] **Step 1: Failing/agreement tests** — both the agreement case and the divergence case:

```python
def test_toward_goal_agrees_across_both_modules_on_canonical_frames(gt_fixture):
    """One authority (S1): TF-4's count must equal TF-35's per-action toward_goal sum."""
    actions, frames = gt_fixture
    tf4 = F.add_off_ball_runs(actions, frames, home_team_id=1)
    runs = detect_off_ball_runs(actions, frames, home_team_id=1,
                                params=RunValuationParams(min_peak_speed_ms=0.0))
    counts = runs.groupby("action_id")["toward_goal"].sum()
    idx = tf4.set_index(actions["action_id"])
    for aid, n in counts.items():
        assert idx.loc[aid, "n_off_ball_runners_toward_goal_pre_window"] == n


def test_unresolvable_direction_does_not_flip(unresolvable_direction_fixture):
    """S13: the acting team has no direction-carrying frame row -> conservative no-flip.

    This is the documented BEHAVIOUR CHANGE vs the old home_team_id keying, which would
    have flipped an away team here.
    """
    actions, frames = unresolvable_direction_fixture
    out = F.add_off_ball_runs(actions, frames, home_team_id=1)
    assert out.loc[0, "n_off_ball_runners_toward_goal_pre_window"] == EXPECTED_NO_FLIP
```

- [ ] **Step 2: Extract the shared predicate** (module level, replacing `:126-137`):

```python
def _prepare_run_candidates(sliced: pd.DataFrame) -> pd.DataFrame:
    """Shared TF-4 / TF-35 run-candidacy (ADR-042; 3rd-consumer extraction).

    Keeps same-team-as-actor, non-actor, non-goalkeeper rows; drops NaN positions and
    dead-ball frames. Expects ``sliced`` to already carry ``action_team_id`` /
    ``actor_player_id`` and to have ball rows removed. Action-level dead-ball tagging and
    the <2-frame skip stay loop-local in each consumer.
    """
    teammates = sliced[
        ids_equal(sliced["team_id"], sliced["action_team_id"]).to_numpy()
        & ids_differ(sliced["player_id"], sliced["actor_player_id"]).to_numpy()
        & (~sliced["is_goalkeeper"].astype(bool)).to_numpy()
    ].copy()
    teammates = teammates.dropna(subset=["x", "y"])
    if "ball_state" in teammates.columns:
        teammates = teammates[teammates["ball_state"] != "dead"]
    return teammates
```

- [ ] **Step 3: Re-key TF-4's `toward_goal`** (`:172-197`): compute
  `flip = acting_team_attacks_rtl(actions, frames)` once per game, map by `action_id`, and
  count `dx_ltr > 0` where `dx_ltr = -dx if flip else dx`. Comment: single orientation
  authority (ADR-028/ADR-041); unresolvable direction → no flip (documented in the docstring).
- [ ] **Step 4:** Run the WHOLE tracking suite (review: "output-identical by construction" is
  a claim the diff must earn — do not narrow to `-k off_ball`):
  `python -m pytest tests/tracking/ -q --tb=short; echo EXIT=$?` → `EXIT=0`.

### Task 10: `_run_values.py` — params, coverage warning, `detect_off_ball_runs`

**Files:** Create `silly_kicks/tracking/_run_values.py`; Modify `tracking/__init__.py`,
`tests/test_public_api_examples.py`; Test: `tests/tracking/test_run_values_detect.py`.

- [ ] **Step 1: Write the ground-truth fixture + tests** (expectation BY CONSTRUCTION — spec
  R2-2; no test-side reimplementation of the rules):

```python
"""Hand-written ground truth. One game, team 1 home (ltr), team 2 away (rtl).

Action 10 (home pass by P1, t=30.0) -- runners designed by hand:
  A pid 101: disp 6.0 m, peak 7.0 m/s       -> qualifies sprint-ON and sprint-OFF
  B pid 102: disp 4.0 m, peak 3.5 m/s       -> qualifies sprint-OFF only
  C pid 103: disp 2.0 m, peak 8.0 m/s       -> fails displacement, never qualifies
  GK pid 100 (is_goalkeeper): disp 6.0 fast -> excluded by candidacy
  Opp pid 201: disp 8.0 fast                -> excluded (other team)
  D pid 104: single frame in window         -> excluded (<2 frames)
Action 11 (home pass, t=60.0): ball_state 'dead' at the action frame -> NO rows.
Action 12 (AWAY pass by team 2, t=90.0): runner E pid 202, disp 6.0 at 7.0 m/s toward
  frame x=0 (the away team's attacked goal) -> emitted positions must be ACTION-LTR.
"""

EXPECTED_SPRINT_OFF = {(10, 101), (10, 102), (12, 202)}
EXPECTED_SPRINT_ON = {(10, 101), (12, 202)}
```

  with tests: detect == `EXPECTED_SPRINT_OFF` at `min_peak_speed_ms=0`; the TF-4 kernel's
  counts match the SAME hand-written truth (2 / NA / 1) — both implementations pinned to one
  external oracle, neither to the other; sprint-on discriminator; away positions action-LTR
  (`run_end_x > run_start_x`, `toward_goal` True); NaN-speed fallback sets
  `peak_speed_source == "displacement_rate"` vs `"measured"`; and the floor resolution:

```python
def test_floor_resolution_fail_loud():
    from silly_kicks.tracking._run_values import _FLOOR_BY_METHOD   # S15: no literal pin

    assert RunValuationParams().resolved_region_floor() == _FLOOR_BY_METHOD["spearman"]
    with pytest.raises(ValueError, match="no calibrated floor"):
        RunValuationParams(pitch_control_method="voronoi").resolved_region_floor()
    assert RunValuationParams(pitch_control_method="voronoi",
                              region_influence_floor=0.5).resolved_region_floor() == 0.5
```

- [ ] **Step 2:** Run → FAIL (module absent).
- [ ] **Step 3: Implement.** `RunValuationParams` (frozen): `pre_seconds=1.5`,
  `min_displacement_m=3.0`, `min_peak_speed_ms=5.56`, `region_influence_floor: float|None=None`,
  `pitch_control_method="spearman"`, `__post_init__` validation, `resolved_region_floor()`
  reading `_FLOOR_BY_METHOD = {"spearman": 0.1}` with a fail-loud message naming the voronoi
  binary-{0,1} hazard. **Docstring states honestly (S5): 0.1 is a spec-time starting value,
  NOT calibrated — the R1 sensitivity probe is its calibration; and the whole floor apparatus
  exists only because `run_value` is a MAX over a thresholded region (an influence-weighted
  mean would delete it — recorded v2 fork).** `RunValueCoverageWarning(UserWarning)` joins
  `tracking/_warnings.py` (one module, one import path).
  `detect_off_ball_runs(actions, frames, *, home_team_id, params=None)` per Task 9's shared
  candidacy + the window/dead-ball handling of `_off_ball_runs_kernel:105-123`; per
  `(action_id, player_id)` group: skip <2 rows; first-vs-last → `displacement_m`,
  `duration_s`, `mean_speed_ms`, `peak_speed_ms` (max finite `speed`); qualify iff
  `disp >= min_displacement_m` AND (`peak >= min_peak_speed_ms` OR all-NaN-speed AND
  `disp/duration >= min_peak_speed_ms`); emit `peak_speed_source ∈ {"measured",
  "displacement_rate"}` (S3 — the V9 bias becomes data, not a docstring note). Then
  point-reflect the four position columns where `acting_team_attacks_rtl`, and set
  `toward_goal = run_end_x > run_start_x`. Ids passthrough source dtype (ADR-019).
- [ ] **Step 4:** Run → PASS. Add `silly_kicks/tracking/_run_values.py` to
  `_PUBLIC_MODULE_FILES`; every public symbol carries an `Examples` section.

### Task 11: `value_off_ball_runs`

**Files:** Modify `_run_values.py`; Test: `tests/tracking/test_run_values_value.py`.

- [ ] **Step 1: Failing tests — materialized, no ellipsis (S8):** roles from the receiver
  (`target` / `disruptive`, `is_receiver` True); `run_value` == a hand-computed
  `EXPECTED_MAX` built into the fixture; failed pass → all-NaN roles/values; unresolved
  receiver → off-domain; **absent-at-pass-frame → `run_value` NaN, row SURVIVES, role still
  assigned, `RunValueCoverageWarning` raised**; `enabled_pass_credit ==
  max(0, Δ values_at_points)`.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement** `value_off_ball_runs(runs, actions, frames, xt, *, links=None,
  pitch_control_cache=None, params=None)`: domain = `type_id ∈ {pass, cross}` AND
  `result_id == success`; receiver via `resolve_next_touch_receiver` (unresolved ⇒ whole
  action off-domain); one decomposed PC surface per action at the linked frame;
  `threat = physical_grid(xt, pc.grid_x, pc.grid_y)` x-flipped when the acting team attacks
  RTL (PC queried in FRAME coords; only emitted positions are action-LTR); then **per runner,
  resolve the index ONCE with the safe compare (review B4)** — verified that
  `player_surface` AND `player_share` both use a raw
  `np.where(self.player_ids == player_id)` (`_surface.py:140,167`), which can disagree with
  an ADR-019-safe compare on exactly the mixed-dtype ids these helpers exist for, so a
  check-then-call would raise inside the loop:

```python
        idx = _safe_index_of(pc.player_ids, pid)   # canonical-id compare; None if absent
        if idx is None:
            n_unvalued += 1
            run_value = np.nan
        else:
            ps = pc.per_player_influence[idx]      # NOT pc.player_surface(pid): one authority
            region = ps >= params.resolved_region_floor()
            run_value = float((pc.surface * threat)[region].max()) if region.any() else 0.0
```

  disruptive rows get `enabled_pass_credit = max(0.0, values_at_points(end) −
  values_at_points(start))` in action-LTR coords; if `n_unvalued`, ONE
  `warnings.warn(..., RunValueCoverageWarning, stacklevel=2)`.
- [ ] **Step 4:** Run → PASS.
- [ ] **Step 5: Record the latent gap** (do NOT fix — scope): `player_surface`/`player_share`
  raw id compare is an ADR-019 gap that also makes `_player_influence.py:139`'s
  `except ValueError: → 0.0` **silently zero players** on dtype-mismatched frames. Goes into
  TODO Technical Debt in Task R1 with this evidence.

### Task 12: aggregator + xfns + atomic mirror

- [ ] **Step 1: Failing tests** — the 5 wide columns, the 0-vs-NA rule, xfns shape (4 numeric
  × 3 slots with `n_valued_disruptive_runs` EXCLUDED), and the auto-discovering absence guard
  copied from `tests/tracking/test_packing_xfns_leakage_guard.py:31-75` (forbidden substring
  `"run_value"`, `__name__` pin on both factories, non-vacuity floor).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3: Implement `add_off_ball_run_values`** (`@nan_safe_enrichment`) emitting
  `run_value_target` (float64), `n_disruptive_runs` (Int64), `run_value_disruptive_sum`
  (float64, skipna), `n_valued_disruptive_runs` (Int64), `run_value_enabled_pass` (float64,
  once per action). Off-domain ⇒ all five NA; on-domain-with-nothing ⇒ 0. Docstring carries
  the R2-3 mean-bias rule (**divide by `n_valued_disruptive_runs`, never
  `n_disruptive_runs`**), the F4 leakage `.. warning::`, and the provider-bias note pointing
  at `peak_speed_source`.
- [ ] **Step 4: `off_ball_run_value_xfns`** — the `off_ball_context_xfns` template
  (`features.py:1645-1694`); Int64 → `astype("float64")` BEFORE `.to_numpy()`;
  `_frame_aware = True`; `__name__ = "off_ball_run_values"`; guarded by `require_fitted_xt`
  at factory time.
- [ ] **Step 5: Atomic mirror** following atomic `add_packing` exactly (`end = x+dx`,
  synthesized `result_id` = success iff the next atom is `receival`, numeric-only).
- [ ] **Step 6:** Run both test files → PASS.

### Task 13: gate registrations + fixture extension

- [ ] **Step 1 (explicit precondition — a step, not a hope):** `Grep` `tests/` for imports of
  `test_aggregator_column_liveness` and for reuse of its `_frames()`/`_actions()` builders →
  expect zero hits outside the file. **Record the grep output in the PR notes.**
- [ ] **Step 2: Extend the fixture** (`test_aggregator_column_liveness.py:120-170`): in
  windows 0 and 4 give one non-receiver teammate stored `vx = 7.0 + 0.5*w_idx` with a
  matching positional advance (≈6.4–7.0 m/s over the 1.4 s window — today's fastest off-ball
  mover is ≈4.95 m/s, so sprint-gated columns would be born dead); set those windows'
  pass/cross `result_id = success` and append a same-team follow-up touch ≈1 s later by a
  sprinting runner so `resolve_next_touch_receiver` resolves and `run_value_target` is live
  and non-constant.
- [ ] **Step 3:** Register liveness entries for `add_off_ball_run_values` (+ atomic) — the
  gate meta-asserts `set(ENTRIES) == {add_* in tracking.__all__}`, so this is forced. Run the
  WHOLE liveness file (every other aggregator must still pass under the extended fixture).
- [ ] **Step 4:** Purity entries (≥2 variants: internal-link vs supplied
  `links` + `pitch_control_cache`) for both packages; **register `add_off_ball_run_values`
  in `_EXHAUSTIVE_EMITTED`** with its 5 columns (its docstring enumerates them exhaustively).
- [ ] **Step 5:** NOTICE entries — Sumpter/Twelve (Soccermatics Pro module 16.2, a
  practitioner anchor, stated as such) + Esposito et al. 2026 (framing only, DOI
  10.1177/17479541261427153); docstrings cross-link per ADR-005.
- [ ] **Step 6:** Full gates → green.
- [ ] **Step (no commit):** gates green -> continue to R1. The single approval-gated commit happens after R1.

---

## Task R1: release mechanics (before opening the PR)

- [ ] **Step 1: PRE-SPECIFY the Part A e2e band, THEN probe (S6).** Write
  `tests/tracking/test_obso_orientation_e2e.py` (`@pytest.mark.e2e`) with the band committed
  **before** the probe runs: post-repair away/home mean-`obso_actual` ratio ∈ **[0.7, 1.4]**
  (the spec §A4 a-priori prediction — a pure orientation error means the two distributions
  must be comparable up to team strength). Then run on real WC2022/GS data. **Outside the
  band ⇒ a FINDING: stop and report; do not widen the band to fit.** Record the pre-repair
  ratio as evidence the oracle discriminates.
- [ ] **Step 2: Part B probe-then-pin** (legitimate — no prior): `n_disruptive_runs` and
  `run_value_*` magnitude bands, INCLUDING the 3-point `region_influence_floor` sensitivity
  check (0.05 / 0.1 / 0.2). Pin bands with the probe date; if 0.1 is not a defensible knee,
  change the default and say why.
- [ ] **Step 3:** ADR-041 (Fork A single-sourcing incl. the **cost/benefit** — not
  impossibility — record of the `_xt_gk._grid_value` non-migration; the orientation repairs;
  the ADR-028 amendment; the `interpolator()` docstring fork; the Fork B deferral + its
  narrow fix; the first-non-numeric-column convention) and ADR-042 (TF-35, incl. the S5
  max-vs-weighted-mean fork and the S13 TF-4 behaviour change).
- [ ] **Step 4:** CLAUDE.md — correct the ADR-028 "self-reconciling" list (obso →
  reprojected-at-query-seam); 4.52.0 paragraph; C4 count 29→30; **add scipy to the
  Dependencies line** (verified stale: scipy is a hard base dep at `pyproject.toml:35` but
  CLAUDE.md lists only pandas/numpy/scikit-learn).
- [ ] **Step 5:** Version lockstep 4.52.0 (`pyproject.toml`, `silly_kicks/__init__.py`,
  `uv.lock`, CHANGELOG, TODO). TODO gains the Task-11 Technical Debt entry.
- [ ] **Step 5b: ADR-041 must record the fixture-mislabelling finding** (§Execution findings
  #6): every synthetic tracking fixture encoded two teams attacking the same way; the guard
  re-baselines the repo's core shared fixture (4 auto-enumerating gates) at zero measured
  value change; and the acting-team-only narrowing was considered and rejected. Also record
  the promotion follow-up: `validate_period_directions` lives in `_action_orientation.py`
  but is called only from `_off_ball_runs.py`; promoting it into `acting_team_attacks_rtl`
  itself turns silently-wrong into a hard raise across its **7** call sites
  (`features.py:1720/2126/4123/4765`, `utils.py:854`, `_gk_geometry.py:196`,
  `_kernels.py:877`) and is gated on the consumer-side uniform-direction pre-check.
- [ ] **Step 6: Downstream impact (S10)** — in the CHANGELOG and PR body name explicitly:
  every away-team `obso_actual/peak/optimal`, `pausa_*`, `space_created_m2` /
  `space_denied_m2_opponent` changes (DEFECT A, applies even without `xt=`); all
  `player_influence_*` shift slightly (DEFECT B); TF-4's
  `n_off_ball_runners_toward_goal_pre_window` changes on unresolvable-direction rows (S13);
  five new `fct_action_context` candidate columns plus `obso_epv_source` and
  `peak_speed_source`; the lakehouse recompute sequence and cost; batch it with the
  already-queued 4.49–4.51 triggers.
- [ ] **Step 7:** `/final-review` (incl. C4 Phase 4) → fix findings → gates green → owner
  approval → push, PR, CI green, admin squash merge, tag after main CI green, PyPI.

---

## Execution findings (Tasks 0-5, 2026-07-18)

Two plan/review claims did NOT survive contact with the code. Both are the failure mode the
round-2 process rule names — a claim about executable behaviour asserted from adjacent
evidence — so both are recorded here with the executed evidence.

1. **The `vaep → xthreat` module-level import is NOT cycle-free (Task 1 Step 5).** Review
   round 1 verified "no runtime vaep→xthreat edge to invert … a new xthreat module stays
   cycle-free PROVIDED it does not import `vaep.feature_framework`", and the plan adopted a
   module-level import on that basis. It fails immediately:
   `xthreat/__init__ → _eval → _grid → spadl.config → spadl/__init__:45 →
   tracking.direction → tracking/__init__:239 → tracking.feature_framework →
   vaep.feature_framework → vaep/__init__:3 → vaep.features/__init__:21 →
   expected_threat → xthreat (partially initialized)`. The verification checked whether the
   NEW module imports vaep; it did not check whether xthreat TRANSITIVELY reaches vaep — it
   does, via `spadl/__init__`. **Resolution:** function-local import at both call sites (the
   sanctioned `_xt_gk.py:198` idiom). Single-sourcing is preserved; only the binding moved.
   This is also why the original SK-xT-2 design was TYPE_CHECKING-only.
2. **The mirror-golden fixture was vacuous twice before it was valid (Task 0 Step 3).**
   `space_created_m2` is the ACTOR's own leave-one-out, which is identically `0.0` unless
   the actor is a MARGINAL controller. First draft (actor behind the ball, near-zero EPV):
   `[0., 0.]`. Second draft (actor ahead but with the defence parked deep, so the attacking
   team already held ~100% control): `[0., 0.]` again. A golden of zeros would have passed
   under ANY change to the mirror axis. Caught only because the generator refuses to write
   an all-zero golden — a guard added mid-task after the first draft slipped through an
   `isfinite`-only check. Final geometry (actor ahead, nearest opponent ~11-13 m beyond)
   yields `created=[0.254, 0.581]`, `denied=[0.180, 0.137]` — non-zero AND distinct per row.
   **The generator's non-vacuity guard is now part of the deliverable**, not a scaffold.

3. **Self-caught S4 defect: funnelling the warning through a helper added a frame.** With
   all six sites calling `_warn_synthetic_epv`, `stacklevel=2` blames `features.py` —
   library internals — which is precisely the "warnings become noise" failure S4 named.
   Corrected to `stacklevel=3` (verified `@nan_safe_enrichment` returns `fn` UNWRAPPED, so
   it contributes no frame; all six sites call the helper at identical depth, so one
   constant is right everywhere) and pinned by five parametrized tests asserting the
   warning's recorded `filename` is the caller's file.
4. **S11 blast radius measured: 9 modules** (`test_add_star_purity`,
   `test_enrichment_nan_safety`, `test_aggregator_column_liveness`,
   `test_frame_aware_xfns_dup_action_id`, `test_id_dtype_invariance`,
   `test_provenance_skip_guard`, `test_obso`, `test_pausa`, `test_space_creation`). Small
   and structurally stable: five are the auto-enumerating gates (which sweep every
   aggregator on defaults BY DESIGN, so their opt-out is permanent and correct) and four
   are the OBSO-family feature tests whose subject IS the default surface. Global
   `filterwarnings = ["error::...", ...]` adopted with per-module `pytestmark` opt-outs
   carrying explicit reasons; **that opt-out list is the synthetic-EPV call-site
   inventory**, which is what S3 actually asked for.
5. **Tooling note (process):** the first attempt to add those opt-outs inserted them by
   "last line matching an import regex", which lands INSIDE a multi-line
   `from x import (...)` and broke 4 of 9 files. Re-done with `ast.parse` +
   `max(node.end_lineno)` and an `ast.parse` round-trip before writing. A regex over
   Python source is not a parser; the syntax check caught it immediately, but the lesson
   is the same class as the round-2 process rule.

6. **SCOPE CHECK (owner-flagged): the Q1 guard re-baselines the repo's core test fixture.**
   Q1's decision (harden the validator, fix the fixtures, re-key TF-4) was reasoned about as
   "fix two fixtures." The actual reach is larger and must not be discovered at review time:

   **RESOLVED (cross-session round 2): the guard was NARROWED to self-contradiction only.**
   The first draft raised on (a) all teams sharing a direction and (b) a team with no
   direction. An exhaustive 3-agent audit then falsified the premise both rules rested on:

   | Shape | Produced by | Is it an error? |
   |---|---|---|
   | Uniform `"ltr"` for both teams | `_snapshot.py:92,118` (public `snapshot_to_tracking_frames`) | **No** -- snapshot frames are ALREADY action-LTR, so "never flip" is correct |
   | All-null direction | `skillcorner.py:282` / `metrica.py:180` under `output_convention="absolute_frame"`; documented at `skillcorner.py:180`; fed to the training corpora by `scripts/_loader_pining.py:439,529` | **No** -- unoriented, i.e. no orientation asserted |
   | Period-5 unlabelled | `direction.py:29` `_LTR_KNOWN_PERIODS=(1,2,3,4)`; `:267` "PSO: orientation undefined --- never flip" | **No** -- undefined by nature |
   | One team carrying BOTH directions | (nothing legitimate) | **Yes** -- the only impossible state |

   The distinction is **mislabelled vs unoriented**, and only the former is impossible. The
   first draft conflated them and therefore REGRESSED all three production shapes -- the old
   `_validate_ltr` accepted an all-null column, so `add_off_ball_runs` had always worked on
   them. `acting_team_attacks_rtl` returning False on an unresolvable direction is the
   CONTRACT ("no orientation asserted -> no flip"), not a gap; treating it as a gap was a
   Chesterton's Fence walked past.

   Shipped rule: raise iff one team resolves to both directions in a `(game_id, period_id)`.
   Period 5 is excluded anyway as a documented invariant (redundant under the narrow rule --
   an all-null period cannot self-contradict -- but it stops a future widening from silently
   re-breaking shootouts, which is exactly how the first draft failed). All three shapes are
   pinned as ACCEPTED by `TestUnorientedFramesAreAccepted`.

   **Promotion into `acting_team_attacks_rtl` (7 call sites) is REJECTED ON EVIDENCE, not
   deferred** -- the three shapes are produced by the library itself, so no consumer-side
   data could ever establish the precondition. Recorded as rejected, with the shapes named,
   so it is not retried later.

   **Fixture scope, correctly sized.** The audit found ~78 uniform-`"ltr"` sites across 39
   test files, but most are not defects (single-team or orientation-irrelevant fixtures are
   fine under the narrow rule). Only fixtures asserting TWO-TEAM orientation semantics were
   wrong. Four were corrected -- `test_off_ball_runs.py` (required by the re-key),
   `test_defensive_line.py`, `test_aggregator_column_liveness.py` (shared by liveness /
   purity / nan-safety) and `conftest_id_dtype.py` -- each verified value-neutral. The
   remaining sites are recorded as a SURVEY in ADR-041, not queued as a cleanup.

   **The re-key's honest framing:** aligning TF-4 with `acting_team_attacks_rtl` buys
   CONSISTENCY (it was the last of 8 modules keyed on home/away identity), NOT correctness --
   the two authorities already agree on correctly-labelled frames and are both arbitrary on
   unoriented ones. CHANGELOG must say "aligns TF-4 with the ADR-028 orientation authority",
   never "fixes wrong values". Behaviour on unoriented frames is pinned by
   `test_unoriented_behaviour_is_pinned_no_flip`.

   **Measured consequence: no values moved.** The liveness gate passes 33/33 after the
   correction, and the reason is structural, not luck: every action in that fixture belongs
   to team 5 (home, `"ltr"`), so `acting_team_attacks_rtl` returns False for all of them
   exactly as before; only the *opponent's* label changed, and no aggregator reads the
   opponent's direction. The correction is therefore behaviour-neutral by construction, and
   the gate that would have caught structural drift is green.

   **The underlying finding is the durable one:** EVERY synthetic tracking fixture in the
   repo encoded the physically-impossible state (two teams attacking the same way). No
   existing test had ever exercised a correctly-labelled two-team frame -- which is exactly
   why the mislabelling survived long enough for TF-4 to be built on identity-keying, and
   why re-keying it *looked* unsafe when it was not. Belongs in ADR-041.

   **Alternative considered and rejected:** narrowing the guard to the ACTING team only
   (raise when the acting team's own direction is unresolvable/contradicted) would have cut
   the fixture blast radius. Rejected once the measurement showed the broad guard costs
   nothing: it would be strictly less correct, and would have been chosen only to avoid work
   that turned out to be free.

Findings 1-3 do not change the design; all are recorded because the next reviewer should
not re-derive them, and #1 in particular contradicts a claim in the round-1 verification
table.

**Deviation from plan TDD ordering (disclosed):** Tasks 2, 3 and 4 were written
implementation-first, then tests — so no RED was observed for them. Their tests are
non-vacuous (Task 3 includes the discriminating real-xT-vs-synthetic check and the five
attribution checks above), but they lack the "proved it can fail" evidence Task 1 has via
its verified `flipud` mutation check. Retrofitting a mutation check per task is the cheap
honest remedy if required.

## Review log

**Round 2 (plan) — 2026-07-18, lakehouse session. Verdict: approve after N1, N3, N4; N2
withdrawn by the reviewer after they read the gate bodies.** All claims re-verified before
applying; two of my own asserted numbers were corrected by re-measurement.

| Item | Disposition |
|---|---|
| **N1** commit split not achievable as sequenced | **Applied** — tasks renumbered 0–13 and made contiguous per commit; each commit fires immediately after its own last task, none deferred |
| **N2** string column breaks gates | **Retracted by reviewer**; the two surviving notes kept — the first-non-numeric-column convention is documented, and the separate `pd.array(dtype="string")` seed + `.loc` fill carries a "do not fold into the `np.nan` pre-seed" comment at BOTH `_OBSO_COLUMNS` and `_SPACE_CREATION_COLUMNS` |
| **N3** `atol=1e-12` wrong for an m² area | **Applied** — `rtol=1e-9, atol=0`; and re-measured: `_interpolate_grid` loses only **3.331e-16** at the real (32,50) target (not the 8.7e-15 I had quoted from a broad shape sweep), and symmetrizing the source does **not** improve it — which further confirms not mutating production |
| **N4** unresolved binding fork + changed shipped text | **Applied** — the private `_require_fitted_xt` name is DELETED (verified imported by nothing else); both sites call `require_fitted_xt(model, caller="xt_xfns")`, reproducing today's messages byte-identically (asserted verbatim in a test); the property "no second implementation" is asserted by source scan, not by a binding |
| S11 filterwarnings blast radius | Applied — a measurement step produces the exact opt-out module list, added as module-level `pytestmark`; the list is recorded in the PR notes as the synthetic-call-site inventory |
| S12 `caller=` string | Applied — each aggregator passes its own public name; no `"add_obso/xt="` composites |
| S13 TF-4 divergence is reachable | **Applied, and my "no-op" claim withdrawn** — `acting_team_attacks_rtl` returns False on unresolvable direction while home-keying always flips, so re-keying changes those rows; now carries its own test and a CHANGELOG line |
| S14 provenance naming | Applied — decided, no fork: ONE name `obso_epv_source` in both aggregators |
| S15 floor-literal pin | Applied — the test imports `_FLOOR_BY_METHOD` instead of hardcoding 0.1 |
| S16 pasted constants | Applied — golden captured at Task 0 into a committed npz via `scripts/gen_space_creation_mirror_golden.py`, with the source commit recorded and a shared fixture builder so golden and assertion cannot drift |
| **Process rule (round-2 addendum)** | Extracted from errors on BOTH sides: the plan's unmeasured "≤ 1 ulp" symmetry claim and the reviewer's unread "fails the liveness gate" claim. Rule added to **CLAUDE.md › Testing**: *claims about a gate's behaviour must quote the assertion body, not its registration; numeric claims carry a pasted measurement; measure at the scale you assert at.* Deliberately **NOT** an ADR — verification discipline, not an architecture decision (precedent: the structural-guard and ADR-023 slow-gating paragraphs live in CLAUDE.md › Testing). Retroactive sweep done: (1) the `8.7e-15` figure is corrected everywhere it appears and now carries its target shape — the only two occurrences are explicitly labelled as the superseded assertion beside the measurement; (2) the gate-reality table above gained the **"asserts what"** column, quoted from source. |

**Round 1 (plan) — 2026-07-18.** 4 blockers + 10 should-fixes + 2 forks; all claims
confirmed on re-verification, none refuted, and the B1 float measurement reproduced exactly
(4.440892098500626e-16, 3752/6400). Fork A **taken** (verification found the guard already
duplicated twice, so the old design would have shipped copy #3 plus a sentinel); Fork B
**deferred** with its narrow bug fixed (build at the EFFECTIVE `params` geometry). B1's
production mutation **not** taken — measurement showed it unnecessary. B2 `__len__`, B3
gate-reality table + manual Examples registration + "write a 5th mirror function", B4
resolve-index-once. S1–S10 applied (5 split commits, provenance columns, separate warning
categories, pre-specified e2e band, 4 mutation checks, set-based identity gate via a
hand-built fixture, downstream-impact section).

**Spec correction to fold in when ADR-041 is written:** §A0.5's "half-cell registration
offset" mischaracterizes `_interpolate_grid`, which is an endpoint-preserving NODE-registration
bilinear resample (verified) — the accurate statement is that a cell-centre-sampled source
resampled onto node registration drifts by up to half a cell at the edges.

---

## Execution findings (Tasks 9-13 + R1, 2026-07-18)

Four more claims/assumptions did not survive contact with the code. Recorded with executed
evidence, per the same rule as the Tasks 0-5 section.

7. **DEFECT C — the ADR-028 reflection was applied on ONE axis (Tasks 7/8 rework).** Both
   the OBSO repair and the `player_influence` repair first flipped the threat grid as
   `[:, ::-1]`. ADR-028's relation is a 180-degree POINT reflection (`x -> 105-x` AND
   `y -> 68-y`), so the correct transform is `[::-1, ::-1]`. An x-only mirror is exact only
   for a y-symmetric grid — which the synthetic ramp IS and a fitted xT very nearly is — so
   the incomplete repair passed every x-axis test written for it. Found only by writing a
   deliberately y-ASYMMETRIC oracle per site; both went RED with the preference exactly
   inverted (OBSO: 0 vs 0.729) before the fix. `space_creation` had already been given
   `axis=(0, 1)` in Task 6, which is what made the inconsistency visible.

8. **Task 13 Step 1's expected "zero hits" was wrong.** The precondition grep found TWO real
   consumers of the shared liveness fixture outside its own file:
   `tests/test_add_star_purity.py:51` (imports `make_actions`/`make_frames`) and
   `tests/tracking/_space_creation_mirror_fixture.py:8`. So extending the fixture touches the
   purity gate too. Both were re-run and pass; the fixture change is additive (two windows
   gain sprinters, one action is appended).

9. **The liveness fixture's first extension produced a CONSTANT metric column.** With one
   disruptive sprinter per on-domain window, `run_value_disruptive_sum` came out 1.0 in both
   — live but informationally dead, exactly what the non-constant check exists to catch.
   Window 4 now carries a second disruptive sprinter. Also: `run_value_target` is 0.0 in
   window 0 (the receiver controls nothing above the floor) and 1.0 in window 4, which is an
   honest zero, not a gap.

10. **The atomic mirror was silently all-`<NA>` and no auto-enumerating gate could see it.**
    `_packing_atomic_adapter` maps every non-domain atom to standard `non_action`, which
    `resolve_next_touch_receiver` deliberately SKIPS (a non-action is not a touch). That
    erases the `receival` atom — the row carrying receiver identity — so no receiver ever
    resolved, every action fell off-domain, and all five columns returned `<NA>` for every
    row. Purity, id-dtype, and nan-safety all PASS on an all-`<NA>` mirror. Fixed with a
    local `_restore_reception_touches` (reception atoms -> standard `bad_touch`: a real
    touch the resolver sees, off-domain for TF-35 so it is never valued itself), and pinned
    by three new tests including a non-vacuity one.

**Two process notes, both self-inflicted:**

- A `pd.NA` boolean-ambiguity bug shipped into the aggregator (`role == "target"` on a
  nullable string column with `<NA>` -> `.to_numpy(dtype=bool)` raises) and was reachable
  only when a detected run belongs to an off-domain action. No standard fixture produced
  that shape; the atomic purity variant did. Now regression-tested directly.
- `git stash` was run to capture a pyright baseline WHILE the full suite was executing,
  which invalidated that run and left conflict markers from an unrelated May stash in
  `_ghost_gk.py` / `train_ghost_gk.py`. The old stash was recovered by SHA
  (`git stash store`), the two files were reset to HEAD, and the suite was re-run clean.
