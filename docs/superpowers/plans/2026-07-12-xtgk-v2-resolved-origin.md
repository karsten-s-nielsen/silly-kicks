# xT-GK v2 Resolved-Origin Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE for this repo:** the owner's standing policy is **inline execution, no subagents**
> (`feedback_inline_execution_default`) and **ONE commit per branch**
> (`feedback_commit_policy`). The per-task `git commit` steps below are therefore
> **`git add` only** — the single squash commit happens at the end (Task 14). Do not deviate.

**Goal:** Stop xT-GK v2 scoring ~24% of its GK-distribution domain at a fabricated grid zone, by
feeding it the resolved keeper origins that already exist in gold, and make the omission
impossible to repeat silently.

**Architecture:** A new pure library helper (`apply_resolved_gk_geometry`) overrides raw
coordinates with gold's resolved ones on the GK-distribution domain and stamps per-row provenance;
`compute_xt_gk_v2` gains a NaN guard (never fabricate a zone), a coordinate-coherence check
(actions↔ρ-features can't diverge), and a warn-once attestation. Both Databricks loaders pipe
through the helper, so all three consuming scripts inherit the fix. ρ is retrained on the corrected
cohort, and SP5 is re-run under both ρ vintages.

**Tech Stack:** Python 3.10–3.12, pandas, numpy, pytest, ruff, pyright, databricks-sql-connector
(owner-run only).

**Spec:** `docs/superpowers/specs/2026-07-12-xtgk-v2-resolved-origin-design.md` (rev 3)

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `silly_kicks/xtgk/_possession_value.py` | grid binning + **new** `finite_coord_mask`; corrected `flat_zones` docstring | Modify |
| `silly_kicks/xtgk/_resolved_geometry.py` | **NEW** — `apply_resolved_gk_geometry`, `GK_GEOMETRY_SOURCE_COLUMN`, stamp values | Create |
| `silly_kicks/xtgk/_retention_features.py` | extract `_coord_derived` (single-sourced), stamp passthrough | Modify |
| `silly_kicks/xtgk/_metric.py` | NaN guard + coherence check + warn-once attestation | Modify |
| `silly_kicks/xtgk/__init__.py` | export the new public surface | Modify |
| `scripts/_loader_databricks.py` | SELECT resolved coords; apply helper in **both** loaders | Modify |
| `scripts/validate_xtgk_v2.py` | `--retention-weights` | Modify |
| `scripts/xtgk_v2_keeper_discrimination.py` | `--retention-weights` | Modify |
| `scripts/xtgk_v2_kappa_sweep.py` | `--retention-weights` | Modify |
| `tests/xtgk/test_flat_zones_contract.py` | **NEW** — T3 | Create |
| `tests/xtgk/test_apply_resolved_gk_geometry.py` | **NEW** — T2 | Create |
| `tests/xtgk/test_metric_nan_coord_guard.py` | **NEW** — T1 | Create |
| `tests/xtgk/test_metric_retention_coherence.py` | **NEW** — T5 | Create |
| `tests/xtgk/test_deep_zone_gate_nan_invariance.py` | **NEW** — T4 | Create |
| `tests/xtgk/test_resolved_origin_changes_score_e2e.py` | **NEW** — T6 | Create |

---

## Task 0: Branch + baseline

**Files:** none (verification only)

- [ ] **Step 1: Cut the branch off main**

```bash
git checkout main
git pull
git checkout -b pr-s113-xtgk-v2-resolved-origin
```

- [ ] **Step 2: Confirm a green baseline before touching anything**

Run: `python -m pytest tests/xtgk -q -m "not e2e"`
Expected: all pass, 0 failures. If anything is already red, STOP and report — do not build on red.

---

## Task 1: `finite_coord_mask` + corrected `flat_zones` docstring (T3)

**Files:**
- Modify: `silly_kicks/xtgk/_possession_value.py` (after `flat_zones`, ~line 52)
- Test: `tests/xtgk/test_flat_zones_contract.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_flat_zones_contract.py`:

```python
"""T3 — flat_zones' NaN->0 behaviour is PINNED (the six fit seams depend on it), and
finite_coord_mask is the blessed alternative for SCORING callers. ADR-036 amendment."""

import numpy as np
import pandas as pd

from silly_kicks.xtgk._possession_value import finite_coord_mask, flat_zones, zone_of


def test_flat_zones_nan_still_maps_to_zone_176_pinned():
    """PINNED, deliberately. _markov.py:65 / _empirical.py:83 / _diagnostics.py:123 call
    flat_zones WITH NaN rows to assign pressure terciles, then drop them before solving.
    Changing this would silently move every fitted surface."""
    z = flat_zones(pd.Series([float("nan")]), pd.Series([float("nan")]))
    assert int(z[0]) == 176
    assert int(zone_of(0.0, 0.0)) == 176  # NaN -> (0,0) -> the own-corner cell


def test_finite_coord_mask_flags_every_non_finite_coordinate():
    actions = pd.DataFrame(
        {
            "start_x": [5.0, np.nan, 5.0, 5.0, 5.0],
            "start_y": [34.0, 34.0, np.nan, 34.0, 34.0],
            "end_x": [40.0, 40.0, 40.0, np.nan, 40.0],
            "end_y": [34.0, 34.0, 34.0, 34.0, np.inf],
        }
    )
    mask = finite_coord_mask(actions)
    assert mask.tolist() == [True, False, False, False, False]


def test_finite_coord_mask_is_all_true_on_clean_input():
    actions = pd.DataFrame(
        {"start_x": [5.0, 6.0], "start_y": [34.0, 30.0], "end_x": [40.0, 41.0], "end_y": [34.0, 20.0]}
    )
    assert finite_coord_mask(actions).all()
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/xtgk/test_flat_zones_contract.py -q`
Expected: FAIL — `ImportError: cannot import name 'finite_coord_mask'`.

- [ ] **Step 3: Implement `finite_coord_mask` and fix the docstring**

In `silly_kicks/xtgk/_possession_value.py`, replace the `flat_zones` docstring body and append the
new function directly beneath it:

```python
COORD_COLUMNS = ("start_x", "start_y", "end_x", "end_y")


def flat_zones(x, y, l: int = N, w: int = M) -> np.ndarray:
    """Vectorized flat grid indices for coordinate Series. NaN coords map to zone 0-bin -> **176**.

    .. warning::
       The NaN -> ``(0.0, 0.0)`` fallback is a **FIT-PATH contract**, NOT a general one. It is safe
       only because every fitting seam drops NaN-coord rows before a surface is solved --
       ``_moves.py``, ``_xg_reward.py``, ``_markov.py:106``, ``_empirical.py:87``,
       ``_turnover.py:131`` drop *before* calling in; ``_markov.py:65``, ``_empirical.py:83`` and
       ``_diagnostics.py:123`` pass NaN rows *through* here to assign pressure terciles and drop
       them afterwards. Either way no NaN row reaches a fitted surface.

       **SCORING callers MUST mask with** :func:`finite_coord_mask` **first.** A scoring caller that
       does not will silently fabricate a real value at zone 176 (the own-corner cell) for every
       NaN-coord row. That defect shipped in 4.40.0-4.45.0 and corrupted ~24% of the xT-GK v2
       GK-distribution domain; see ADR-036 and
       ``docs/superpowers/specs/2026-07-12-xtgk-v2-resolved-origin-design.md``.
    """
    xs = pd.to_numeric(pd.Series(x), errors="coerce").fillna(0.0)
    ys = pd.to_numeric(pd.Series(y), errors="coerce").fillna(0.0)
    return _get_flat_indexes(xs, ys, l, w).to_numpy()


def finite_coord_mask(actions: pd.DataFrame) -> npt.NDArray[np.bool_]:
    """True where ALL of ``start_x``/``start_y``/``end_x``/``end_y`` are finite.

    The blessed pre-filter for any caller that SCORES (as opposed to fits) on the grid -- it is
    what stops :func:`flat_zones` fabricating zone 176 from a NaN coordinate. See ADR-036.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"start_x": [5.0, float("nan")], "start_y": [34.0, 34.0],
    ...      "end_x": [40.0, 40.0], "end_y": [34.0, 34.0]}
    ... )
    >>> finite_coord_mask(a).tolist()
    [True, False]
    """
    mask = np.ones(len(actions), dtype=bool)
    for col in COORD_COLUMNS:
        mask &= np.isfinite(pd.to_numeric(actions[col], errors="coerce").to_numpy(dtype=float))
    return mask
```

- [ ] **Step 4: Run the test — it must pass**

Run: `python -m pytest tests/xtgk/test_flat_zones_contract.py -q`
Expected: 3 passed.

- [ ] **Step 5: Stage (no commit — one commit per branch)**

```bash
git add silly_kicks/xtgk/_possession_value.py tests/xtgk/test_flat_zones_contract.py
```

---

## Task 2: `apply_resolved_gk_geometry` + the `gk_geometry_source` stamp (T2)

**Files:**
- Create: `silly_kicks/xtgk/_resolved_geometry.py`
- Test: `tests/xtgk/test_apply_resolved_gk_geometry.py` (create)

**Stamp contract (spec §2.1):** seven values. `unresolved` **wins** whenever any of the four
coordinates is still non-finite after resolution (R3). Missing resolved-coordinate columns →
warn + full no-op + `unattested`, **never** `native` (R2).

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_apply_resolved_gk_geometry.py`:

```python
"""T2 -- apply_resolved_gk_geometry: OVERRIDE (not coalesce), stamp semantics, purity."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.xtgk import apply_resolved_gk_geometry
from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN


def _frame():
    """Eight rows covering every stamp value reachable with the resolved columns present."""
    return pd.DataFrame(
        {
            "is_gk_distribution": [True, True, True, True, True, False, True, True],
            # row0 native (raw == resolved)     | row1 GS NaN origin rescued
            # row2 SC present-but-WRONG         | row3 dest-only override
            # row4 mixed: resolved origin + NaN dest -> `unresolved` wins (R3)
            # row5 off-domain
            # row6 (S3) GS unresolved-origin/good-dest: raw NaN origin AND resolved-NULL origin
            # row7 (S2) never attested: finite raw coords, ALL resolved coords NULL -> `unattested`
            "start_x": [5.5, np.nan, 25.0, 5.5, np.nan, 60.0, np.nan, 5.5],
            "start_y": [34.0, np.nan, 40.0, 34.0, np.nan, 20.0, np.nan, 34.0],
            "end_x": [40.0, 40.0, 40.0, 33.0, np.nan, 70.0, 40.0, 40.0],
            "end_y": [34.0, 34.0, 34.0, 30.0, np.nan, 20.0, 34.0, 34.0],
            "xt_gk_origin_x": [5.5, 5.5, 4.29, 5.5, 7.08, np.nan, np.nan, np.nan],
            "xt_gk_origin_y": [34.0, 34.0, 34.01, 34.0, 34.44, np.nan, np.nan, np.nan],
            "xt_gk_dest_x": [40.0, 40.0, 40.0, 35.0, np.nan, np.nan, 40.0, np.nan],
            "xt_gk_dest_y": [34.0, 34.0, 34.0, 31.0, np.nan, np.nan, 34.0, np.nan],
        }
    )


def test_s3_unresolved_origin_with_good_dest_is_stamped_unresolved():
    """S3: raw NaN origin + resolved-NULL origin + finite dest (the GS `unresolved` shape). Today
    this only lands on `unresolved` via the FINAL precedence np.where -- a reorder would silently
    regress it to `native`/`resolved_dest` with no failing test. This is that test."""
    out = apply_resolved_gk_geometry(_frame())
    assert np.isnan(out.loc[6, "start_x"])
    assert out.loc[6, GK_GEOMETRY_SOURCE_COLUMN] == "unresolved"


def test_s2_never_attested_row_is_unattested_not_native():
    """S2: finite raw coords but ALL resolved coords NULL. Nothing attested this row, so stamping
    it `native` ("raw already equalled resolved") would be a lie AND would suppress the metric's
    warn-once."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[7, "start_x"] == pytest.approx(5.5)  # untouched
    assert out.loc[7, GK_GEOMETRY_SOURCE_COLUMN] == "unattested"


def test_override_not_coalesce_replaces_present_but_wrong_skillcorner_origin():
    """THE load-bearing case. Row 2's raw origin is PRESENT (25.0) and WRONG (broadcast ball);
    a coalesce would leave it. It must be REPLACED by the resolved keeper origin."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[2, "start_x"] == pytest.approx(4.29)
    assert out.loc[2, "start_y"] == pytest.approx(34.01)
    assert out.loc[2, GK_GEOMETRY_SOURCE_COLUMN] == "resolved_origin"


def test_nan_origin_is_filled_from_resolved():
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[1, "start_x"] == pytest.approx(5.5)
    assert out.loc[1, GK_GEOMETRY_SOURCE_COLUMN] == "resolved_origin"


def test_native_row_unchanged_and_stamped_native():
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[0, "start_x"] == pytest.approx(5.5)
    assert out.loc[0, GK_GEOMETRY_SOURCE_COLUMN] == "native"


def test_dest_override_path_synthetic_real_data_is_a_noop():
    """Real cohorts never exercise this (measured: 0 rows differ) -- so it is tested synthetically."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[3, "end_x"] == pytest.approx(35.0)
    assert out.loc[3, "end_y"] == pytest.approx(31.0)
    assert out.loc[3, GK_GEOMETRY_SOURCE_COLUMN] == "resolved_dest"


def test_unresolved_wins_precedence_on_mixed_row():
    """R3: row 4 has a RESOLVED origin but a still-NaN dest. The stamp answers 'will this row
    score?', so `unresolved` must win over `resolved_origin`."""
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[4, "start_x"] == pytest.approx(7.08)  # origin WAS applied
    assert np.isnan(out.loc[4, "end_x"])  # dest stays NaN
    assert out.loc[4, GK_GEOMETRY_SOURCE_COLUMN] == "unresolved"


def test_off_domain_row_untouched():
    out = apply_resolved_gk_geometry(_frame())
    assert out.loc[5, "start_x"] == pytest.approx(60.0)
    assert out.loc[5, GK_GEOMETRY_SOURCE_COLUMN] == "off_domain"


def test_resolved_both_when_origin_and_dest_change():
    df = _frame()
    df.loc[0, "xt_gk_origin_x"] = 6.0
    df.loc[0, "xt_gk_dest_x"] = 41.0
    out = apply_resolved_gk_geometry(df)
    assert out.loc[0, GK_GEOMETRY_SOURCE_COLUMN] == "resolved_both"


def test_purity_input_never_mutated_and_new_object_returned():
    df = _frame()
    before = df.copy(deep=True)
    out = apply_resolved_gk_geometry(df)
    pd.testing.assert_frame_equal(df, before)
    assert out is not df


def test_missing_domain_column_raises():
    df = _frame().drop(columns=["is_gk_distribution"])
    with pytest.raises(ValueError, match="is_gk_distribution"):
        apply_resolved_gk_geometry(df)


def test_missing_resolved_columns_warns_noops_and_stamps_unattested_never_native():
    """R2: stamping `native` here would suppress the metric's warn-once while origins are still
    raw -- exactly the SkillCorner present-and-wrong hole the stamp exists to close."""
    df = _frame().drop(columns=["xt_gk_origin_x", "xt_gk_origin_y"])
    with pytest.warns(UserWarning, match="xt_gk_origin_x"):
        out = apply_resolved_gk_geometry(df)
    assert out.loc[2, "start_x"] == pytest.approx(25.0)  # no-op: raw retained
    in_domain = out["is_gk_distribution"].to_numpy(dtype=bool)
    assert set(out.loc[in_domain, GK_GEOMETRY_SOURCE_COLUMN]) == {"unattested"}
    assert out.loc[5, GK_GEOMETRY_SOURCE_COLUMN] == "off_domain"
```

- [ ] **Step 2: Run it and watch it fail**

Run: `python -m pytest tests/xtgk/test_apply_resolved_gk_geometry.py -q`
Expected: FAIL — `ImportError: cannot import name 'apply_resolved_gk_geometry'`.

- [ ] **Step 3: Implement the helper**

Create `silly_kicks/xtgk/_resolved_geometry.py`:

```python
"""Resolved GK-distribution geometry (ADR-036 amendment, 4.46.0).

The GK-distribution domain's canonical SPADL coords are NOT trustworthy:
  * Gradient Sports -- ~60% of goal-kicks carry a NaN origin (the taker is not in the raw event).
  * SkillCorner     -- the native goal-kick origin is the broadcast BALL detection, not the keeper
                       (ADR-024 / PR-S104): PRESENT, finite, and ~10-20 m wrong.

v1 (``tracking/_xt_gk.py``) resolves both via ``resolve_gk_geometry`` and the lakehouse persists the
result as ``fct_action_context.xt_gk_origin_x/_y`` + ``xt_gk_dest_x/_y`` (PR-S101). v2 never read
them. This module is the ONE callable that injects them, so the rule lives in one place instead of
as a prose contract two consumers must each re-derive -- which is how the bug happened.

Policy lives at the EDGE: the metric engine stays provenance-free and reads exactly
``start_x``/``end_x``. See ADR-025 (this is a transient SCORING-TIME view; canonical coords are
never written back) and ``feedback_policy_at_edge_not_shared_engine``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

GK_GEOMETRY_SOURCE_COLUMN = "gk_geometry_source"

#: The seven stamp values. ``unresolved`` WINS over any ``resolved_*`` when a coordinate is still
#: non-finite (it answers "will this row score?", pairing with the metric's NaN guard).
GK_GEOMETRY_SOURCES = (
    "off_domain",
    "native",
    "resolved_origin",
    "resolved_dest",
    "resolved_both",
    "unresolved",
    "unattested",
)


def _changed(raw: np.ndarray, res: np.ndarray) -> np.ndarray:
    """True where the resolved value EXISTS and differs from the raw one (NaN raw counts as a
    difference, so a rescue is a change)."""
    return np.isfinite(res) & ~np.isclose(raw, res, atol=1e-9, rtol=0.0, equal_nan=True)


def apply_resolved_gk_geometry(
    actions: pd.DataFrame,
    *,
    domain_column: str = "is_gk_distribution",
    origin_columns: tuple[str, str] = ("xt_gk_origin_x", "xt_gk_origin_y"),
    dest_columns: tuple[str, str] = ("xt_gk_dest_x", "xt_gk_dest_y"),
) -> pd.DataFrame:
    """OVERRIDE the GK-distribution rows' coords with gold's resolved keeper geometry; stamp provenance.

    PURE: returns a NEW frame, never mutates ``actions``.

    **Override, not coalesce.** A ``fillna`` would rescue Gradient Sports' NaN origins and silently
    leave SkillCorner's *present-and-wrong* broadcast-ball origins in place.

    Parameters
    ----------
    actions : pd.DataFrame
        Attack-LTR SPADL with ``start_x``/``start_y``/``end_x``/``end_y`` and ``domain_column``.
    domain_column : str
        The GK-distribution domain flag. **Required** -- absent raises (treating every row as
        in-domain would overwrite open-play coords with keeper geometry).
    origin_columns, dest_columns : tuple[str, str]
        The resolved coords. If either PAIR is absent this is an observable no-op (warn) and every
        in-domain row is stamped ``unattested`` -- never ``native``, which would suppress the
        metric's warn-once while the origins are still raw.

    Returns
    -------
    pd.DataFrame
        A copy with the overridden coords plus a ``gk_geometry_source`` column.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"is_gk_distribution": [True], "start_x": [25.0], "start_y": [40.0],
    ...      "end_x": [40.0], "end_y": [34.0], "xt_gk_origin_x": [4.29],
    ...      "xt_gk_origin_y": [34.0], "xt_gk_dest_x": [40.0], "xt_gk_dest_y": [34.0]}
    ... )
    >>> out = apply_resolved_gk_geometry(a)
    >>> float(out.loc[0, "start_x"]), out.loc[0, "gk_geometry_source"]
    (4.29, 'resolved_origin')
    """
    if domain_column not in actions.columns:
        raise ValueError(
            f"apply_resolved_gk_geometry requires the domain column {domain_column!r}. Without it "
            "every row would be treated as a GK distribution and open-play coordinates would be "
            "overwritten with keeper geometry. Supply fct_action_context.is_gk_distribution."
        )

    out = actions.copy()
    domain = out[domain_column].fillna(False).to_numpy(dtype=bool)
    source = np.where(domain, "unattested", "off_domain").astype(object)

    ox, oy = origin_columns
    dx_c, dy_c = dest_columns
    missing = [c for c in (ox, oy, dx_c, dy_c) if c not in out.columns]
    if missing:
        warnings.warn(
            f"apply_resolved_gk_geometry: resolved-coordinate columns {missing} are absent -- "
            "no-op; GK-distribution origins remain RAW (Gradient Sports NaN / SkillCorner "
            "broadcast-ball). Rows stamped 'unattested'. Select xt_gk_origin_x/_y + "
            "xt_gk_dest_x/_y from fct_action_context (silly-kicks >= 4.36.0).",
            stacklevel=2,
        )
        out[GK_GEOMETRY_SOURCE_COLUMN] = source
        return out

    def _num(col: str) -> np.ndarray:
        return pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)

    sx, sy, ex, ey = _num("start_x"), _num("start_y"), _num("end_x"), _num("end_y")
    rox, roy, rdx, rdy = _num(ox), _num(oy), _num(dx_c), _num(dy_c)

    origin_changed = domain & (_changed(sx, rox) | _changed(sy, roy))
    dest_changed = domain & (_changed(ex, rdx) | _changed(ey, rdy))

    # Apply wherever a resolved value exists (not only where it differs) -- idempotent.
    apply_o = domain & np.isfinite(rox) & np.isfinite(roy)
    apply_d = domain & np.isfinite(rdx) & np.isfinite(rdy)
    sx = np.where(apply_o, rox, sx)
    sy = np.where(apply_o, roy, sy)
    ex = np.where(apply_d, rdx, ex)
    ey = np.where(apply_d, rdy, ey)
    out["start_x"], out["start_y"], out["end_x"], out["end_y"] = sx, sy, ex, ey

    finite = np.isfinite(sx) & np.isfinite(sy) & np.isfinite(ex) & np.isfinite(ey)
    both = origin_changed & dest_changed
    # S2: a row whose resolved coords are ALL null was never attested by the mart. Stamping it
    # `native` would assert "raw already equalled resolved" -- which is FALSE, nothing attested it --
    # and would suppress the metric's warn-once. Such rows stay `unattested` (the initial value).
    attested = np.isfinite(rox) | np.isfinite(roy) | np.isfinite(rdx) | np.isfinite(rdy)
    source = np.where(domain & attested & origin_changed & ~dest_changed, "resolved_origin", source)
    source = np.where(domain & attested & dest_changed & ~origin_changed, "resolved_dest", source)
    source = np.where(domain & attested & both, "resolved_both", source)
    source = np.where(domain & attested & ~origin_changed & ~dest_changed, "native", source)
    # R3 precedence: `unresolved` wins over every resolved_*/native when a coord is still non-finite.
    source = np.where(domain & ~finite, "unresolved", source)
    out[GK_GEOMETRY_SOURCE_COLUMN] = source
    return out
```

- [ ] **Step 4: Export it, then run the test**

In `silly_kicks/xtgk/__init__.py` add the import and the `__all__` entry (keep `__all__`
alphabetical — it is a static list; pyright rejects dynamic construction):

```python
from silly_kicks.xtgk._resolved_geometry import (
    GK_GEOMETRY_SOURCE_COLUMN,
    GK_GEOMETRY_SOURCES,
    apply_resolved_gk_geometry,
)
```

and into `__all__`, in alphabetical position:

```python
    "GK_GEOMETRY_SOURCES",
    "GK_GEOMETRY_SOURCE_COLUMN",
    ...
    "apply_resolved_gk_geometry",
```

Also export `finite_coord_mask` from Task 1:

```python
from silly_kicks.xtgk._possession_value import (
    DeltaV,
    PossessionValue,
    PressureLevel,
    State,
    finite_coord_mask,
    mirror_zone,
    zone_of,
)
```
and add `"finite_coord_mask",` to `__all__`.

Run: `python -m pytest tests/xtgk/test_apply_resolved_gk_geometry.py -q`
Expected: **12 passed** (the 10 rev-1 cases plus the S2 `unattested`-not-`native` and S3
unresolved-origin/good-dest precedence tests).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/xtgk/_resolved_geometry.py silly_kicks/xtgk/__init__.py tests/xtgk/test_apply_resolved_gk_geometry.py
```

---

## Task 3: Single-source the coordinate-derived ρ features

**Why first:** the metric's coherence check (Task 5) must recompute *exactly* what
`extract_retention_features` computes. Extracting the shared helper now means the two can never
drift — recomputing them independently in `_metric.py` would be the same class of bug we are fixing.

**Files:**
- Modify: `silly_kicks/xtgk/_retention_features.py`
- Test: `tests/xtgk/test_retention_features.py` (existing — extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/xtgk/test_retention_features.py`:

```python
def test_coord_derived_is_the_single_source_used_by_extract():
    """The metric's coherence check recomputes these; they MUST be the same code path."""
    import pandas as pd

    from silly_kicks.xtgk._retention_features import (
        COORD_DERIVED_NAMES,
        _coord_derived,
        extract_retention_features,
    )

    a = pd.DataFrame(
        {
            "start_x": [5.5, 30.0],
            "start_y": [34.0, 20.0],
            "end_x": [40.0, 55.0],
            "end_y": [34.0, 44.0],
            "type_id": [12, 0],
            "pressure": [0.1, 0.4],
        }
    )
    full = extract_retention_features(a)
    derived = _coord_derived(a)
    for c in COORD_DERIVED_NAMES:
        pd.testing.assert_series_equal(full[c], derived[c], check_names=False)


def test_gk_geometry_source_passes_through_when_present():
    import pandas as pd

    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN
    from silly_kicks.xtgk._retention_features import extract_retention_features

    a = pd.DataFrame(
        {
            "start_x": [5.5],
            "start_y": [34.0],
            "end_x": [40.0],
            "end_y": [34.0],
            "type_id": [12],
            "pressure": [0.1],
            GK_GEOMETRY_SOURCE_COLUMN: ["resolved_origin"],
        }
    )
    out = extract_retention_features(a)
    assert out[GK_GEOMETRY_SOURCE_COLUMN].tolist() == ["resolved_origin"]
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/xtgk/test_retention_features.py -q`
Expected: FAIL — `ImportError: cannot import name 'COORD_DERIVED_NAMES'`.

- [ ] **Step 3: Refactor `_retention_features.py`**

**Import placement (ruff E402).** `GK_GEOMETRY_SOURCE_COLUMN` must be imported at the **top of the
file**, alongside the existing `import silly_kicks.spadl.config as spadlconfig` — NOT below
`RETENTION_FEATURE_NAMES`. `E` is selected in ruff config and the `xtgk` per-file-ignores are only
`N803`/`N806`/`E741`, so a module-level import after a statement is an E402 failure in Task 10.

Add to the import block at the top:

```python
from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN
```

Then replace the body of `silly_kicks/xtgk/_retention_features.py` below `RETENTION_FEATURE_NAMES`:

```python
#: The subset of RETENTION_FEATURE_NAMES that is pure arithmetic on start/end coords. The metric's
#: coordinate-coherence check recomputes EXACTLY these (ADR-036 amendment) -- single-sourced here so
#: the check and the trainer can never drift.
COORD_DERIVED_NAMES = ["length", "forwardness", "dy_abs", "dest_x", "dest_y_off"]


def _coord_derived(actions: pd.DataFrame) -> pd.DataFrame:
    """The five coordinate-derived retention features. RAW arithmetic -- standardisation happens
    inside ``GkRetentionModel``, which is what makes them directly comparable to the coords."""
    ox = pd.to_numeric(actions["start_x"], errors="coerce").to_numpy(float)
    oy = pd.to_numeric(actions["start_y"], errors="coerce").to_numpy(float)
    dxx = pd.to_numeric(actions["end_x"], errors="coerce").to_numpy(float)
    dyy = pd.to_numeric(actions["end_y"], errors="coerce").to_numpy(float)
    dx, dy = dxx - ox, dyy - oy
    length = np.hypot(dx, dy)
    return pd.DataFrame(
        {
            "length": length,
            "forwardness": np.divide(dx, length, out=np.zeros_like(dx), where=length > 0),
            "dy_abs": np.abs(dy),
            "dest_x": dxx,
            "dest_y_off": np.abs(dyy - spadlconfig.field_width / 2),
        },
        index=actions.index,
    )


def extract_retention_features(actions: pd.DataFrame, *, pressure_column: str = "pressure") -> pd.DataFrame:
    """8 mart-derived features from an attack-LTR SPADL action frame carrying start/end coords,
    ``type_id``, and a pressure column. NaN-coord-tolerant (geometry rows drop downstream).

    When ``actions`` carries the :data:`GK_GEOMETRY_SOURCE_COLUMN` stamp (i.e. it came through
    :func:`~silly_kicks.xtgk.apply_resolved_gk_geometry`) the stamp is **passed through** as a
    non-feature column. This is inert to the model -- ``GkRetentionModel.fit``/``predict_proba``
    both select ``features[self.feature_names]`` -- and lets ``compute_xt_gk_v2`` attest that the
    features and the actions came from the same resolved frame.

    Examples
    --------
    >>> import pandas as pd
    >>> a = pd.DataFrame(
    ...     {"start_x": [5.5], "start_y": [34.0], "end_x": [40.0], "end_y": [34.0],
    ...      "type_id": [12], "pressure": [0.1]}
    ... )
    >>> sorted(extract_retention_features(a).columns) == sorted(RETENTION_FEATURE_NAMES)
    True
    """
    out = _coord_derived(actions)
    tid = actions["type_id"].to_numpy()
    out["release_pressure"] = (
        pd.to_numeric(actions[pressure_column], errors="coerce").to_numpy(float)
        if pressure_column in actions.columns
        else np.full(len(actions), np.nan)
    )
    out["is_goalkick"] = (tid == _GOALKICK).astype(float)
    out["is_throw_in"] = (tid == _THROW_IN).astype(float)
    if GK_GEOMETRY_SOURCE_COLUMN in actions.columns:
        out[GK_GEOMETRY_SOURCE_COLUMN] = actions[GK_GEOMETRY_SOURCE_COLUMN].to_numpy()
    return out
```

Also export `COORD_DERIVED_NAMES` — add `"COORD_DERIVED_NAMES",` to `silly_kicks/xtgk/__init__.py`'s
`__all__` and to the `_retention_features` import line.

- [ ] **Step 4: Run the whole retention-feature suite (regression guard)**

Run: `python -m pytest tests/xtgk/test_retention_features.py tests/xtgk/test_retention.py -q`
Expected: all pass — the refactor must be behaviour-preserving for the 8 named features.

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/xtgk/_retention_features.py silly_kicks/xtgk/__init__.py tests/xtgk/test_retention_features.py
```

---

## Task 4: Metric NaN guard (T1)

**Files:**
- Modify: `silly_kicks/xtgk/_metric.py`
- Test: `tests/xtgk/test_metric_nan_coord_guard.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_metric_nan_coord_guard.py`:

```python
"""T1 -- compute_xt_gk_v2 NEVER fabricates a zone from a NaN coordinate (ADR-036 amendment)."""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV

GOALKICK = spadlconfig.actiontype_id["goalkick"]
_OUT = ["xt_gk_v2_position", "xt_gk_v2_pev", "xt_gk_v2_retention_loss", "xt_gk_v2_dzv", "xt_gk_v2"]


class _StubV:
    pressure_levels = None

    def value(self, zone, p):
        return 0.02

    def surface(self, p):
        return np.full((12, 16), 0.02)

    def delta_v(self, s, s_next):
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _CountingRho:
    """Records how many rows it was asked to score -- the NaN rows must never reach it."""

    def __init__(self):
        self.seen = 0

    def predict_proba(self, features):
        self.seen += len(features)
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.05

    def surface(self, p):
        return np.full((12, 16), 0.05)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _actions(start_x):
    n = len(start_x)
    return pd.DataFrame(
        {
            "game_id": [1] * n,
            "period_id": [1] * n,
            "action_id": list(range(n)),
            "type_id": [GOALKICK] * n,
            "start_x": start_x,
            "start_y": [34.0] * n,
            "end_x": [40.0] * n,
            "end_y": [34.0] * n,
            "pressure": [0.1] * n,
        }
    )


def _levels(actions):
    return PressureLevels().fit(actions["pressure"])


def _call(actions, rho):
    from silly_kicks.xtgk._retention_features import extract_retention_features

    return compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=rho,
        turnover_cost=_StubTurnover(),
        pressure_levels=_levels(actions),
        retention_features=extract_retention_features(actions),
    )


def test_nan_coord_row_emits_nan_not_zone_176_value():
    """The defect: NaN -> flat_zones -> zone 176 -> a REAL number. Now it must be NaN."""
    actions = _actions([5.5, np.nan])
    out = _call(actions, _CountingRho())
    assert out.loc[0, "xt_gk_v2"] == pytest.approx(out.loc[0, "xt_gk_v2"])  # finite row scores
    for c in _OUT:
        assert np.isnan(out.loc[1, c]), f"{c} was fabricated for a NaN-coord row"


def test_finite_rows_are_byte_identical_to_the_all_finite_run():
    """The guard must not perturb a single finite row. S4: the claim is BYTE-identity, so this
    asserts exact `==`, not pytest.approx -- the code path supports it."""
    clean = _actions([5.5, 6.0])
    mixed = _actions([5.5, np.nan])
    out_clean = _call(clean, _CountingRho())
    out_mixed = _call(mixed, _CountingRho())
    for c in _OUT:
        assert out_mixed.loc[0, c] == out_clean.loc[0, c]


def test_rho_is_never_called_on_non_finite_rows():
    """Closes the silent mean-imputation path (_retention.py:81) without touching predict_proba."""
    rho = _CountingRho()
    _call(_actions([5.5, np.nan, np.nan]), rho)
    assert rho.seen == 1, f"rho scored {rho.seen} rows; only the 1 finite row should reach it"


def test_warns_with_a_count_of_dropped_rows():
    with pytest.warns(UserWarning, match="2 of 3"):
        _call(_actions([5.5, np.nan, np.nan]), _CountingRho())
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/xtgk/test_metric_nan_coord_guard.py -q`
Expected: FAIL — the NaN row currently gets a real value (zone 176), `rho.seen == 3`, no warning.

- [ ] **Step 3: Implement the guard in `_metric.py`**

Replace the body of `compute_xt_gk_v2` from the `zones_o = ...` line through the `return` with:

```python
    finite = finite_coord_mask(actions)
    n = len(actions)
    n_bad = int((~finite).sum())
    if n_bad:
        warnings.warn(
            f"compute_xt_gk_v2: {n_bad} of {n} actions have non-finite coordinates; their "
            "xt_gk_v2_* are emitted as NaN (never fabricated to a grid zone). For the "
            "GK-distribution domain, route actions through apply_resolved_gk_geometry first -- "
            "unresolved rows are honest NaN, not zone 176. See ADR-036.",
            stacklevel=2,
        )

    zones_o = flat_zones(actions["start_x"], actions["start_y"], l, w)
    zones_d = flat_zones(actions["end_x"], actions["end_y"], l, w)
    zones_arg = zones_o if pl.mode == "zone_conditional" else None
    levels = pl.apply(actions[pressure_column], zones=zones_arg)  # p' = p (base metric)

    position = np.full(n, np.nan)
    pev = np.full(n, np.nan)
    ret_loss = np.full(n, np.nan)
    dzv = np.full(n, np.nan)

    idx = np.flatnonzero(finite)
    if len(idx):
        # rho is scored ONLY on finite rows -- a NaN-coord row would otherwise be silently
        # mean-imputed by GkRetentionModel.predict_proba and multiply every term.
        rho = np.asarray(retention.predict_proba(retention_features.iloc[idx]), dtype=float)
        for k, i in enumerate(idx):
            p = int(levels[i])
            s = State(int(zones_o[i]), p)  # type: ignore[arg-type]
            s_next = State(int(zones_d[i]), p)  # type: ignore[arg-type]
            dv = possession_value.delta_v(s, s_next)
            v_s = float(possession_value.value(int(zones_o[i]), p))
            v_opp = float(turnover_cost.value(int(zones_o[i]), p))
            position[i] = rho[k] * dv.position_component
            pev[i] = rho[k] * dv.pressure_component
            ret_loss[i] = -(1.0 - rho[k]) * v_s
            dzv[i] = -(1.0 - rho[k]) * kappa * v_opp
    total = position + pev + ret_loss + dzv
    return pd.DataFrame(
        {
            "xt_gk_v2_position": position,
            "xt_gk_v2_pev": pev,
            "xt_gk_v2_retention_loss": ret_loss,
            "xt_gk_v2_dzv": dzv,
            "xt_gk_v2": total,
        },
        index=actions.index,
    )
```

Imports: `_metric.py:13` **already** reads
`from silly_kicks.xtgk._possession_value import State, flat_zones`. **REPLACE** that line — do not
add a second import of the same names, which is a ruff `F811` redefinition failure in Task 10:

```python
import warnings  # add to the stdlib import block at the top

from silly_kicks.xtgk._possession_value import State, finite_coord_mask, flat_zones  # REPLACES line 13
```

Also **keep** the existing `# NOTE (scale):` comment above the per-action loop — it documents a
lakehouse batch-path follow-up and is not obsoleted by this change.

- [ ] **Step 4: Run — new test green, existing metric tests still green**

Run: `python -m pytest tests/xtgk/test_metric_nan_coord_guard.py tests/xtgk/test_metric.py -q`
Expected: all pass.

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/xtgk/_metric.py tests/xtgk/test_metric_nan_coord_guard.py
```

---

## Task 5: Coordinate-coherence check + warn-once attestation (T5)

**Contract (spec §2.1(3), R1+R4):** the check compares **coordinates**, not provenance —
recompute the coordinate-derived ρ features from `actions` and compare to what the caller supplied.
It catches F1 (resolved actions + raw features) and R1's mirror (raw actions + resolved features)
**symmetrically**, plus mart-vintage divergence, with no case table.

**Files:**
- Modify: `silly_kicks/xtgk/_metric.py`
- Test: `tests/xtgk/test_metric_retention_coherence.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/xtgk/test_metric_retention_coherence.py`:

```python
"""T5 -- actions and retention_features MUST describe the same coordinates (ADR-036 amendment).

Closes F1 (resolved actions + raw features) AND R1's mirror (raw actions + resolved features).
"""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, apply_resolved_gk_geometry, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV
from silly_kicks.xtgk._retention_features import extract_retention_features

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _StubV:
    def value(self, zone, p):
        return 0.02

    def surface(self, p):
        return np.full((12, 16), 0.02)

    def delta_v(self, s, s_next):
        return DeltaV(delta=0.03, pressure_component=0.0, position_component=0.03)


class _StubRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.05

    def surface(self, p):
        return np.full((12, 16), 0.05)

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _raw():
    """A SkillCorner-shaped row: raw origin PRESENT and WRONG; identical end_x in both frames
    (the dest override is a measured no-op) -- so ONLY the origin diverges."""
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "type_id": [GOALKICK],
            "is_gk_distribution": [True],
            "start_x": [25.0],
            "start_y": [40.0],
            "end_x": [40.0],
            "end_y": [34.0],
            "xt_gk_origin_x": [4.29],
            "xt_gk_origin_y": [34.0],
            "xt_gk_dest_x": [40.0],
            "xt_gk_dest_y": [34.0],
            "pressure": [0.1],
        }
    )


def _call(actions, feats):
    return compute_xt_gk_v2(
        actions,
        possession_value=_StubV(),
        retention=_StubRho(),
        turnover_cost=_StubTurnover(),
        pressure_levels=PressureLevels().fit(actions["pressure"]),
        retention_features=feats,
    )


def test_f1_resolved_actions_with_raw_features_raises():
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    with pytest.raises(ValueError, match="coordinate"):
        _call(resolved, extract_retention_features(raw))


def test_r1_mirror_raw_actions_with_resolved_features_raises():
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    with pytest.raises(ValueError, match="coordinate"):
        _call(raw, extract_retention_features(resolved))


def test_origin_only_divergence_is_caught_end_x_is_identical():
    """Proves the check spans length/forwardness/dy_abs and is NOT dest_x-only. The dest override
    is a measured no-op on real data, so a dest_x-only check would miss every real divergence."""
    raw = _raw()
    resolved = apply_resolved_gk_geometry(raw)
    assert float(raw.loc[0, "end_x"]) == float(resolved.loc[0, "end_x"])  # dest identical
    assert float(raw.loc[0, "start_x"]) != float(resolved.loc[0, "start_x"])  # origin differs
    with pytest.raises(ValueError, match="coordinate"):
        _call(resolved, extract_retention_features(raw))


def test_coherent_pair_scores_normally():
    resolved = apply_resolved_gk_geometry(_raw())
    out = _call(resolved, extract_retention_features(resolved))
    assert np.isfinite(out.loc[0, "xt_gk_v2"])


def test_unstamped_actions_with_a_gk_domain_warn_once_and_still_score():
    raw = _raw()  # never passed through the helper -> no stamp column at all
    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        out = _call(raw, extract_retention_features(raw))
    assert np.isfinite(out.loc[0, "xt_gk_v2"])


def test_unattested_STAMPED_actions_also_warn():
    """S1: the R2 semantics ("`unattested` is treated as unstamped for warning purposes") shipped
    UNTESTED. A frame that WENT THROUGH the helper but found no resolved columns is stamped
    `unattested`, and must still warn -- otherwise it scores raw origins in silence."""
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    raw = _raw().drop(columns=["xt_gk_origin_x", "xt_gk_origin_y", "xt_gk_dest_x", "xt_gk_dest_y"])
    with pytest.warns(UserWarning):  # the helper itself warns about the missing columns
        stamped = apply_resolved_gk_geometry(raw)
    assert stamped.loc[0, GK_GEOMETRY_SOURCE_COLUMN] == "unattested"

    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        out = _call(stamped, extract_retention_features(stamped))
    assert np.isfinite(out.loc[0, "xt_gk_v2"])


def test_mixed_vintage_frame_warns_when_only_SOME_rows_are_unattested():
    """S1 again, the case `.all()` would miss: a concatenated frame where one row is attested and
    one is not. `.any()` must fire."""
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    resolved = apply_resolved_gk_geometry(_raw())
    unattested = resolved.copy()
    unattested[GK_GEOMETRY_SOURCE_COLUMN] = "unattested"
    unattested["action_id"] = [1]
    mixed = pd.concat([resolved, unattested], ignore_index=True)

    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        _call(mixed, extract_retention_features(mixed))
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/xtgk/test_metric_retention_coherence.py -q`
Expected: FAIL — no `ValueError` is raised; no warning.

- [ ] **Step 3: Implement in `_metric.py`**

Add these two private helpers above `compute_xt_gk_v2`:

```python
def _check_coordinate_coherence(actions: pd.DataFrame, retention_features: pd.DataFrame) -> None:
    """actions and retention_features MUST describe the SAME coordinates (ADR-036 amendment).

    Compares COORDINATES, not provenance: recomputes the coordinate-derived retention features from
    ``actions`` and compares. Catches, symmetrically and with no case table:
      * resolved ``actions`` + ρ-features built from the RAW frame (F1);
      * RAW ``actions`` + ρ-features built from the resolved frame (R1's mirror);
      * two frames resolved against different mart vintages (equal stamps, different coords).
    A stamp-equality check would miss the third and needs a 4-row case table for the first two.
    """
    from silly_kicks.xtgk._retention_features import COORD_DERIVED_NAMES, _coord_derived

    if not set(COORD_DERIVED_NAMES).issubset(retention_features.columns):
        return  # not a retention-feature frame (e.g. a test stub) -- nothing to attest
    expected = _coord_derived(actions)
    for col in COORD_DERIVED_NAMES:
        got = pd.to_numeric(retention_features[col], errors="coerce").to_numpy(dtype=float)
        exp = expected[col].to_numpy(dtype=float)
        if got.shape != exp.shape or not np.allclose(got, exp, atol=1e-6, rtol=0.0, equal_nan=True):
            raise ValueError(
                f"compute_xt_gk_v2: retention_features[{col!r}] does not match the coordinates in "
                "`actions`. The rho features were built from a DIFFERENT frame -- typically one "
                "side went through apply_resolved_gk_geometry and the other did not, so the grid "
                "zones and rho would disagree. Build retention_features from the SAME (resolved) "
                "frame you pass as `actions`. See ADR-036."
            )


def _warn_if_unattested(actions: pd.DataFrame, domain_column: str) -> None:
    """Warn once when a GK-distribution domain is present but resolution was never attested."""
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    if domain_column not in actions.columns:
        return
    domain = actions[domain_column].fillna(False).to_numpy(dtype=bool)
    if not domain.any():
        return
    n_unattested = int(domain.sum())
    if GK_GEOMETRY_SOURCE_COLUMN in actions.columns:
        stamps = actions.loc[domain, GK_GEOMETRY_SOURCE_COLUMN].to_numpy()
        # S1: ANY unattested row, not ALL. A concatenated mixed-vintage frame (realistic for the
        # lakehouse) would otherwise score its unattested rows on RAW origins in silence.
        n_unattested = int((stamps == "unattested").sum())
        if n_unattested == 0:
            return
    warnings.warn(
        f"compute_xt_gk_v2: {n_unattested} of {int(domain.sum())} {domain_column} rows carry no "
        "attested resolved geometry. Their origins are RAW -- Gradient Sports goal-kicks are ~60% "
        "NaN and SkillCorner's are the broadcast BALL, not the keeper. Route actions through "
        "apply_resolved_gk_geometry first. See ADR-036.",
        stacklevel=2,
    )
```

**Ordering.** Call `_warn_if_unattested` **before** `_check_coordinate_coherence`, so a caller who
made both mistakes still sees the actionable warning before the raise.

and call them at the top of `compute_xt_gk_v2`, immediately after the existing
`retention_features is None` guard:

```python
    _warn_if_unattested(actions, domain_column)
    _check_coordinate_coherence(actions, retention_features)
```

Add the `domain_column` parameter to the signature (after `pressure_column`):

```python
    domain_column: str = "is_gk_distribution",
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/xtgk/test_metric_retention_coherence.py tests/xtgk/test_metric.py tests/xtgk/test_metric_nan_coord_guard.py -q`
Expected: all pass.

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/xtgk/_metric.py tests/xtgk/test_metric_retention_coherence.py
```

---

## Task 6: Deep-zone gate NaN-invariance regression (T4)

**This test is the evidence for non-goal #1** ("do not re-run the deep-zone gate"). It must exist,
and it must pass, or the claim is unsupported.

**Files:**
- Test: `tests/xtgk/test_deep_zone_gate_nan_invariance.py` (create)

- [ ] **Step 1: Write the test**

Create `tests/xtgk/test_deep_zone_gate_nan_invariance.py`:

**THREE fixture faults were found by EXECUTION in review and are corrected below. Do not
"simplify" them back:**
1. **Shots must sit in the attacking half with margin.** The ADR-028 orientation guard
   (`_validate.py:47-55`) counts a NaN-`start_x` shot as own-half (`NaN > 52.5` is `False`), so a
   naive fixture makes `fit(contaminated)` raise `ValueError: only 43% of shots are in the attacking
   half` — the escalation clause would misfire on a fixture artifact.
2. **Every shot must carry an `xg`.** Drawing `xg` independently of `type_id` gave **0 of 18** shots
   a reward → all three surfaces were identically **zero** → the assertion was `allclose(0, 0)`. A
   vacuous test cited as *the evidence* for non-goal #1 is worse than no test. A non-vacuity
   meta-assertion now guards this.
3. **ONE shared `PressureLevels` across both legs.** Refitting per leg moves the tercile cutpoints
   (240 vs 300 rows → `(0.294, 0.594)` → `(0.303, 0.610)`, flipping **8 of 240** clean rows), so the
   surfaces genuinely differ and the property is FALSE as designed. The property under test is
   *NaN-row-drop invariance of the fit seams*, **not** quantile stability.

```python
"""T4 -- the FIT path is invariant to NaN-coord rows, which is why the 4.42.0 deep-zone gate
verdict does NOT need re-running under the resolved-origin fix (ADR-036 amendment, non-goal #1).

If this ever fails, the gate MUST be re-run.
"""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import MarkovPossessionValue, PressureLevels

PASS = spadlconfig.actiontype_id["pass"]
SHOT = spadlconfig.actiontype_id["shot"]
GOALKICK = spadlconfig.actiontype_id["goalkick"]
SUCCESS = spadlconfig.result_id["success"]


def _cohort(n=240, seed=0):
    rng = np.random.default_rng(seed)
    is_shot = rng.random(n) < 0.12
    # FIXTURE FAULT 1: shots MUST be in the attacking half with margin, or a NaN-start_x shot in the
    # contaminated leg trips the ADR-028 orientation guard (NaN > 52.5 is False -> counted own-half).
    start_x = np.where(is_shot, rng.uniform(70.0, 100.0, n), rng.uniform(0.0, 105.0, n))
    return pd.DataFrame(
        {
            "game_id": rng.integers(1, 5, n),
            "period_id": np.ones(n, dtype=int),
            "action_id": np.arange(n),
            "time_seconds": np.arange(n, dtype=float),
            "team_id": rng.integers(1, 3, n),
            "player_id": rng.integers(1, 23, n),
            "type_id": np.where(is_shot, SHOT, PASS),
            "result_id": np.full(n, SUCCESS),
            "possession_id": rng.integers(1, 40, n),
            "start_x": start_x,
            "start_y": rng.uniform(0, 68, n),
            "end_x": rng.uniform(0, 105, n),
            "end_y": rng.uniform(0, 68, n),
            "pressure": rng.uniform(0, 1, n),
            # FIXTURE FAULT 2: EVERY shot carries an xg. Drawing xg independently of type_id gave
            # 0/18 shots a reward -> every surface was identically ZERO -> allclose(0, 0).
            "xg": np.where(is_shot, rng.uniform(0.05, 0.4, n), np.nan),
        }
    )


def _nan_rows(n=60, seed=1):
    """The real defect's shape: goal-kicks / passes whose ORIGIN is NaN."""
    rng = np.random.default_rng(seed)
    bad = _cohort(n, seed=seed)
    bad["type_id"] = np.where(rng.random(n) < 0.5, GOALKICK, PASS)
    bad["xg"] = np.nan
    bad["start_x"] = np.nan
    bad["start_y"] = np.nan
    bad["action_id"] = np.arange(1000, 1000 + n)
    return bad


def test_fitted_surfaces_and_support_are_invariant_to_added_nan_coord_rows():
    clean = _cohort()
    contaminated = pd.concat([clean, _nan_rows()], ignore_index=True)

    # FIXTURE FAULT 3: ONE shared PressureLevels. Refitting per leg moves the tercile cutpoints
    # (8/240 clean rows flip), so the surfaces would differ for a reason that has nothing to do
    # with the property under test.
    pl = PressureLevels().fit(clean["pressure"])

    def _fit(actions):
        return MarkovPossessionValue().fit(
            actions, xg_column="xg", pressure_column="pressure", pressure_levels=pl
        )

    v_clean = _fit(clean)
    v_contaminated = _fit(contaminated)

    # META-ASSERTION (non-vacuity): an all-zero surface would make the invariance check below
    # `allclose(0, 0)` -- it would pass while proving nothing. That exact defect shipped in the
    # first draft of this test. Guard it.
    for p in (1, 2, 3):
        surface = v_clean.surface(p)  # type: ignore[arg-type]
        assert (surface != 0).sum() > 0, (
            f"pressure tercile {p}: the fitted V surface is ALL ZERO, so the invariance assertion "
            "below would be vacuous. Fix the fixture (shots need xg), do not weaken the test."
        )

    for p in (1, 2, 3):
        np.testing.assert_allclose(
            v_clean.surface(p),  # type: ignore[arg-type]
            v_contaminated.surface(p),  # type: ignore[arg-type]
            atol=1e-12,
            err_msg=(
                f"pressure tercile {p}: the fitted V surface MOVED when NaN-coord rows were added. "
                "The deep-zone gate would then be contaminated and MUST be re-run -- ADR-036 "
                "non-goal #1 no longer holds."
            ),
        )
        np.testing.assert_array_equal(
            v_clean.support(p),  # type: ignore[arg-type]
            v_contaminated.support(p),  # type: ignore[arg-type]
            err_msg=f"pressure tercile {p}: the fitted support counts moved (spec: surfaces AND support).",
        )
```

- [ ] **Step 2: Run it — it must pass FIRST TIME**

Run: `python -m pytest tests/xtgk/test_deep_zone_gate_nan_invariance.py -q`
Expected: PASS immediately (no production change needed — the fit seams already `dropna`).

**Verified by execution before this plan was written** (with the corrected fixture): surfaces carry
10 / 17 / 12 non-zero cells at `max|V| ~ 0.34`; surface **and** support are identical across both
legs in all three terciles; and — the check that gives the test teeth — **fabricating the NaN rows
to `(0, 0)` (exactly what `flat_zones` does) MOVES the surfaces**, so the test would genuinely
catch the regression it claims to guard.

If it FAILS: **stop and escalate.** It means non-goal #1 in the spec is wrong and the deep-zone
gate is contaminated too — a materially bigger PR. Do not "fix" the test.

- [ ] **Step 3: Stage**

```bash
git add tests/xtgk/test_deep_zone_gate_nan_invariance.py
```

---

## Task 7: End-to-end A/B — resolution must CHANGE the score (T6)

House lesson: an A/B must exercise the path that can change the value, or a fix can be inert on the
very case it exists for.

**Files:**
- Test: `tests/xtgk/test_resolved_origin_changes_score_e2e.py` (create)

- [ ] **Step 1: Write the test**

Create `tests/xtgk/test_resolved_origin_changes_score_e2e.py`:

```python
"""T6 -- the fix must actually MOVE the number on the case it exists for.

A SkillCorner-shaped goal-kick (raw origin = broadcast ball at x=25, resolved = keeper at x=4.29)
must score DIFFERENTLY once resolved. A fix that is inert here would be worthless.
"""

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.xtgk import PressureLevels, apply_resolved_gk_geometry, compute_xt_gk_v2
from silly_kicks.xtgk._possession_value import DeltaV, zone_of
from silly_kicks.xtgk._retention_features import extract_retention_features

GOALKICK = spadlconfig.actiontype_id["goalkick"]


class _ZoneSensitiveV:
    """V depends on the ORIGIN zone -- so a moved origin must move the score."""

    def value(self, zone, p):
        return 0.001 * (zone % 17)

    def surface(self, p):
        return np.zeros((12, 16))

    def delta_v(self, s, s_next):
        return DeltaV(
            delta=0.0,
            pressure_component=0.0,
            position_component=0.001 * (s_next.zone % 17) - 0.001 * (s.zone % 17),
        )


class _StubRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.8)


class _StubTurnover:
    def value(self, zone, p):
        return 0.002 * (zone % 13)

    def surface(self, p):
        return np.zeros((12, 16))

    def support(self, p):
        return np.full((12, 16), 100, dtype=int)


def _skillcorner_goalkick():
    return pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "action_id": [0],
            "type_id": [GOALKICK],
            "is_gk_distribution": [True],
            "start_x": [25.0],  # broadcast BALL detection -- present, finite, WRONG
            "start_y": [40.0],
            "end_x": [55.0],
            "end_y": [34.0],
            "xt_gk_origin_x": [4.29],  # the actual keeper
            "xt_gk_origin_y": [34.0],
            "xt_gk_dest_x": [55.0],
            "xt_gk_dest_y": [34.0],
            "pressure": [0.1],
        }
    )


def _score(actions):
    return compute_xt_gk_v2(
        actions,
        possession_value=_ZoneSensitiveV(),
        retention=_StubRho(),
        turnover_cost=_StubTurnover(),
        pressure_levels=PressureLevels().fit(pd.Series([0.0, 0.1, 1.0])),
        retention_features=extract_retention_features(actions),
    )


def test_resolving_a_skillcorner_goalkick_changes_its_score():
    raw = _skillcorner_goalkick()
    resolved = apply_resolved_gk_geometry(raw)

    # The origin genuinely moves to a different grid cell -- otherwise this test proves nothing.
    assert zone_of(25.0, 40.0) != zone_of(4.29, 34.0)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the raw leg intentionally warns (unattested)
        raw_score = float(_score(raw).loc[0, "xt_gk_v2"])
    resolved_score = float(_score(resolved).loc[0, "xt_gk_v2"])

    assert np.isfinite(raw_score) and np.isfinite(resolved_score)
    assert raw_score != resolved_score, (
        "resolving the origin did not change the score -- the fix is inert on the exact case "
        "(SkillCorner present-but-wrong broadcast-ball origin) it exists for"
    )
```

- [ ] **Step 2: Run it**

Run: `python -m pytest tests/xtgk/test_resolved_origin_changes_score_e2e.py -q`
Expected: PASS.

- [ ] **Step 3: Stage**

```bash
git add tests/xtgk/test_resolved_origin_changes_score_e2e.py
```

---

## Task 8: Loaders — SELECT the resolved coords and apply the helper

**Files:**
- Modify: `scripts/_loader_databricks.py` (`_XTGK_ACTIONS_SQL` ~line 201; `load_xtgk_cohort` ~line 224; `_RETENTION_SQL` ~line 272; `load_retention_cohort` ~line 287)

Both SQL blocks already `LEFT JOIN ... fct_action_context c` — only the SELECT list changes.

- [ ] **Step 1: Add the columns to `_XTGK_ACTIONS_SQL`**

In `_XTGK_ACTIONS_SQL`, change the line:

```sql
  c.is_gk_distribution, c.xt_gk, c.player_key,
```
to:
```sql
  c.is_gk_distribution, c.xt_gk, c.player_key,
  c.xt_gk_origin_x, c.xt_gk_origin_y, c.xt_gk_dest_x, c.xt_gk_dest_y,
  c.xt_gk_origin_source, c.xt_gk_dest_source,
```

- [ ] **Step 2: Coerce + apply in `load_xtgk_cohort`**

Extend the `numeric` tuple with the four new coordinate columns:

```python
    numeric = (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "xt_gk_origin_x",
        "xt_gk_origin_y",
        "xt_gk_dest_x",
        "xt_gk_dest_y",
        "pressure_on_actor__bekkers_pi",
        "pressure_on_actor__andrienko_oval",
        "xg",
        "time_seconds",
        "xt_gk",
    )
```

and immediately before `return actions, shot_xg`:

```python
    # ADR-036 amendment (4.46.0): the GK-distribution domain's canonical coords are NOT trustworthy
    # (GS ~60% NaN goal-kick origins; SkillCorner's native origin is the broadcast BALL). Inject the
    # resolved keeper geometry v1 already computed and the lakehouse already persists. Doing it HERE
    # means all three consumers (validate_xtgk_v2 / keeper_discrimination / kappa_sweep) inherit it,
    # and rho features are necessarily built from the resolved frame.
    from silly_kicks.xtgk import apply_resolved_gk_geometry

    actions = apply_resolved_gk_geometry(actions)
    return actions, shot_xg
```

- [ ] **Step 3: Same for `_RETENTION_SQL` / `load_retention_cohort`**

In `_RETENTION_SQL`, change:

```sql
  c.is_gk_distribution
```
to:
```sql
  c.is_gk_distribution,
  c.xt_gk_origin_x, c.xt_gk_origin_y, c.xt_gk_dest_x, c.xt_gk_dest_y,
  c.xt_gk_origin_source, c.xt_gk_dest_source
```

In `load_retention_cohort`, extend the numeric coercion loop:

```python
    for col in (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "xt_gk_origin_x",
        "xt_gk_origin_y",
        "xt_gk_dest_x",
        "xt_gk_dest_y",
        "pressure",
        "time_seconds",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
```

and apply the helper immediately after the `is_gk_distribution` coercion, BEFORE the time-sort:

```python
    from silly_kicks.xtgk import apply_resolved_gk_geometry

    df = apply_resolved_gk_geometry(df)
```

- [ ] **Step 4: Verify the loaders still import cleanly (no Databricks needed)**

Run: `python -c "import scripts._loader_databricks as m; print(sorted(c for c in m._XTGK_ACTIONS_SQL.split() if 'xt_gk_origin' in c or 'xt_gk_dest' in c))"`
Expected: prints the four resolved-coordinate columns (plus the two `_source` columns).

Run: `python -m pytest tests/xtgk/test_retention_loader_domain.py -q`
Expected: PASS (existing loader-contract guard).

- [ ] **Step 5: Stage**

```bash
git add scripts/_loader_databricks.py
```

---

## Task 9: `--retention-weights` on all THREE consuming scripts

The two-leg SP5 (spec §2.5) needs to score the **corrected cohort** under both the **pre-fix** and
the **retrained** ρ. All three scripts currently hardcode `GkRetentionModel.from_variant(...)`.

**Files:**
- Modify: `scripts/validate_xtgk_v2.py`
- Modify: `scripts/xtgk_v2_keeper_discrimination.py` (~line 114)
- Modify: `scripts/xtgk_v2_kappa_sweep.py` (~line 31)

- [ ] **Step 1: Add the shared resolver to `scripts/_loader_databricks.py`**

Append:

```python
def resolve_retention_model(provider: str, weights_dir: str | None = None):
    """Load the rho model for ``provider``, or from an explicit artifact dir.

    ``weights_dir`` exists for the ADR-036 two-leg SP5 re-run: leg 1 scores the CORRECTED cohort
    under the PRE-FIX rho, leg 2 under the retrained one, so the delta is attributable.
    """
    from pathlib import Path

    from silly_kicks.xtgk._retention import GkRetentionModel, variant_key_for_provider

    if weights_dir:
        return GkRetentionModel.load(Path(weights_dir))
    return GkRetentionModel.from_variant(variant_key_for_provider(provider))
```

`GkRetentionModel.load(path)` is **confirmed** (`_retention.py:127`) to take a **directory**
containing `model.json` + `SHA256SUMS` — exactly the shape of `_retention_weights/<variant>/`. No
new API is needed.

- [ ] **Step 2: Wire the flag into each script**

All three scripts use `ap` for the parser and `a = ap.parse_args()` for the namespace
(`validate_xtgk_v2.py:263`, `xtgk_v2_keeper_discrimination.py:92`, `xtgk_v2_kappa_sweep.py:16`), so
the same two edits apply verbatim to each.

Add to each `argparse` block:

```python
    ap.add_argument(
        "--retention-weights",
        default=None,
        help="Path to a rho artifact dir (model.json + SHA256SUMS). Overrides the provider "
        "variant. Used by the ADR-036 two-leg SP5 re-run: leg 1 = corrected coords + PRE-FIX "
        "rho; leg 2 = corrected coords + retrained rho.",
    )
```

Replace each `GkRetentionModel.from_variant(...)` call site with:

```python
    from _loader_databricks import resolve_retention_model  # type: ignore[import-not-found]

    rho = resolve_retention_model(a.provider, a.retention_weights)
```

Concretely: `xtgk_v2_kappa_sweep.py:31` currently reads
`rho = GkRetentionModel.from_variant(variant_key_for_provider(a.provider))`, and
`xtgk_v2_keeper_discrimination.py:114` reads `rho = GkRetentionModel.from_variant(variant)` — both
become the two lines above. Do the same at `validate_xtgk_v2.py`'s own `from_variant` call site.
Drop the now-unused `GkRetentionModel` import where it becomes dead (ruff F401 will tell you).
**Keep `variant_key_for_provider`** — Step 3's `rho_label` still calls it to name the default-variant
leg, so it stays live in every script that adopts the label. Ruff will not flag it; that is expected.

- [ ] **Step 3: Echo the ρ provenance into EVERY report header (B4)**

Without this the two SP5 legs are **indistinguishable in their own artifacts** —
`xtgk_v2_keeper_discrimination.py` writes a block labelled `rho variant: {variant}`, and that label
is **identical** for both legs because `--retention-weights` is never echoed anywhere. Two lines:

In each script, wherever the report header/label is composed, add the weights path:

```python
    rho_label = a.retention_weights or f"variant:{variant_key_for_provider(a.provider)}"
```

and use `rho_label` in place of the bare `variant` string in the emitted header (e.g. the
keeper-discrimination `- rho variant: ...` line and the `validate_xtgk_v2` report header). Every
artifact must state which ρ produced it.

- [ ] **Step 4: Verify each script still parses its args**

Run: `python scripts/validate_xtgk_v2.py --help`
Run: `python scripts/xtgk_v2_keeper_discrimination.py --help`
Run: `python scripts/xtgk_v2_kappa_sweep.py --help`
Expected: each prints usage including `--retention-weights`, exit 0.

- [ ] **Step 5: Stage**

```bash
git add scripts/_loader_databricks.py scripts/validate_xtgk_v2.py scripts/xtgk_v2_keeper_discrimination.py scripts/xtgk_v2_kappa_sweep.py
```

---

## Task 10: Full local gate — lint, types, tests

- [ ] **Step 1: Ruff**

Run: `python -m ruff check silly_kicks tests scripts`
Run: `python -m ruff format --check silly_kicks tests scripts`
Expected: both clean. (`ruff format --check` is a SEPARATE gate from `ruff check` — CI runs both.)

- [ ] **Step 2: Pyright over the WHOLE repo**

Run: `python -m pyright`
Expected: 0 errors. Bare `pyright` — never scope it to one package; that has masked errors before.

- [ ] **Step 3: Full non-e2e suite**

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: all pass, 0 failures.

- [ ] **Step 4: Stage nothing new; fix any failure and re-run**

---

## Task 11 (OWNER-RUN): retrain ρ on the corrected cohort

Requires `DATABRICKS_HOST` / `DATABRICKS_HTTP_PATH` / `DATABRICKS_TOKEN`. Read-only queries.

- [ ] **Step 1: Preserve the pre-fix ρ artifacts (needed for SP5 leg 1)**

```bash
mkdir -p /tmp/rho_prefix
cp -r silly_kicks/xtgk/_retention_weights/default /tmp/rho_prefix/default
cp -r silly_kicks/xtgk/_retention_weights/skillcorner /tmp/rho_prefix/skillcorner
```

- [ ] **Step 2: Retrain both variants**

Both flags are **required** and confirmed (`scripts/train_gk_retention.py:82-83`):

```bash
python scripts/train_gk_retention.py --provider gradientsports --variant default
python scripts/train_gk_retention.py --provider skillcorner   --variant skillcorner
```

- [ ] **Step 3: Read the gate result for each**

Open `silly_kicks/xtgk/_retention_weights/<variant>/metrics.json`. The gate is
`ece <= 0.10 AND |reliability_slope - 1| <= 0.25` (`scripts/train_gk_retention.py:20-21`).

- **Both pass** → both ship; leave `_PROVIDER_VARIANT = {"skillcorner": "skillcorner"}`.
- **SkillCorner fails** → do **not** ship it. Delete `_retention_weights/skillcorner/`, set
  `_PROVIDER_VARIANT: dict[str, str] = {}` in `silly_kicks/xtgk/_retention.py:29` (so every
  provider falls back to `default`), and record the failing numbers in the CHANGELOG. This is the
  4.42.0 precedent — do not ship weights that do not calibrate.
- **gradientsports fails** → there is no fallback. **STOP and escalate to the owner.** That is a
  finding, not something to work around.

- [ ] **Step 4: Regenerate `SHA256SUMS` and re-run the bundle guard**

Run: `python -m pytest tests/xtgk/test_retention_bundle_calibration.py -q`
Expected: PASS — this certifies whatever shipped against the canonical `_ECE_MAX`/`_SLOPE_TOL`.

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/xtgk/_retention_weights silly_kicks/xtgk/_retention.py
```

---

## Task 12 (OWNER-RUN): two-leg SP5 re-run + reports

**Pre-frozen, do not deviate:** same metrics (outcome-AUC lift over `max(baselines)`; action-level
keeper ICC by `player_key`), same baselines, same κ=1 headline, same a-priori parameters. Only
**coordinates** and **ρ** move. Report whatever it shows.

- [ ] **Step 1: Preserve the 4.45.0 (contaminated) reports as the baseline record**

```bash
cd docs/research/xtgk_v2_construct_validity
for f in gradientsports skillcorner keeper_discrimination faithfulness_audit; do
  git mv "$f.md" "$f.4.45.0-raw-origins.md"
done
cd -
```

- [ ] **Step 2: Leg 1 — corrected coords + PRE-FIX ρ, THEN PRESERVE THE OUTPUTS**

**B4 — `validate_xtgk_v2._write_report` OVERWRITES `{provider}.md` (`:251-253`).** Leg 2 runs the
same scripts and would destroy leg 1's artifacts before Step 4 can read them. Copying the outputs
between legs is a **scripted step, not a memory**:

```bash
python scripts/validate_xtgk_v2.py --provider gradientsports --retention-weights /tmp/rho_prefix/default
python scripts/validate_xtgk_v2.py --provider skillcorner   --retention-weights /tmp/rho_prefix/skillcorner
python scripts/xtgk_v2_keeper_discrimination.py --provider gradientsports --retention-weights /tmp/rho_prefix/default
python scripts/xtgk_v2_keeper_discrimination.py --provider skillcorner   --retention-weights /tmp/rho_prefix/skillcorner

# PRESERVE leg 1 before leg 2 overwrites it.
cd docs/research/xtgk_v2_construct_validity
for f in gradientsports skillcorner keeper_discrimination; do cp "$f.md" "$f.leg1-prefix-rho.md"; done
cd -
```

- [ ] **Step 3: Leg 2 — corrected coords + RETRAINED ρ (bundled; no flag)**

```bash
python scripts/validate_xtgk_v2.py --provider gradientsports
python scripts/validate_xtgk_v2.py --provider skillcorner
python scripts/xtgk_v2_keeper_discrimination.py --provider gradientsports
python scripts/xtgk_v2_keeper_discrimination.py --provider skillcorner
python scripts/xtgk_v2_kappa_sweep.py --provider gradientsports
python scripts/xtgk_v2_kappa_sweep.py --provider skillcorner
```

The κ sweep runs on **leg 2 only, deliberately** — κ=1 is the pre-frozen headline (spec §2.5) and
the sweep is a sensitivity report, not part of the leg-vs-leg attribution.

- [ ] **Step 3b: Emit the stamp / NaN-out census (B5)**

The NaN-out count the delta table needs **has no scripted source** — it exists only as a stderr
warning. Add a census print to `xtgk_v2_keeper_discrimination.py` (which scores the **full**
GK-distribution domain, unlike `validate_xtgk_v2`, which scores only the odd-possession TEST half),
right after `compute_xt_gk_v2`:

```python
    from silly_kicks.xtgk import finite_coord_mask
    from silly_kicks.xtgk._resolved_geometry import GK_GEOMETRY_SOURCE_COLUMN

    n_nan = int((~finite_coord_mask(gk)).sum())
    print(f"- census: {len(gk)} GK-distribution actions (POST-prepare_cohort); {n_nan} NaN-coord -> xt_gk_v2 = NaN")
    print(f"- census: stamps = {gk[GK_GEOMETRY_SOURCE_COLUMN].value_counts().to_dict()}")
```

**State the denominator in the delta table**: the cohort is **post-`prepare_cohort`** (frame-absent
null-pressure rows already dropped), so this count is `<=` the spec §1.4 figure of ≤464 for GS.

- [ ] **Step 4: Write the delta table into `docs/research/xtgk_v2_construct_validity/README.md`**

It must carry, for each provider: outcome-AUC lift and keeper ICC at **4.45.0 (raw)**, **leg 1
(coords only)** and **leg 2 (coords + ρ)**; plus this caveat **verbatim** (spec §2.5 / F8):

> **Leg-1 attribution caveat.** Leg 1 (corrected coords + pre-fix ρ) also shifts the pre-fix ρ's
> *input distribution*, because its features derive from the now-overridden coordinates. Leg 1
> therefore isolates *"origin effect **including** ρ-input shift"*, **not** pure zone relabeling.

Also report the count of rows the metric now NaN-outs per provider (the union of unresolved-origin
and unresolved-dest rows; GS ≤ 464 per spec §1.4).

- [ ] **Step 5: Stage**

```bash
git add docs/research/xtgk_v2_construct_validity
```

---

## Task 13: Docs — ADR-036 amendment, CHANGELOG, TODO, version bump

**Files:**
- Modify: `docs/superpowers/adrs/ADR-036-*.md`
- Modify: `CHANGELOG.md`, `TODO.md`, `CLAUDE.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`

- [ ] **Step 1: ADR-036 amendment**

Add a `4.46.0 amendment — resolved GK-distribution geometry` section covering: the root cause
(`flat_zones`' fit-path contract violated at the one scoring seam); the measured blast radius
(§1.4 of the spec, both providers, origin **and** dest); `apply_resolved_gk_geometry` (override,
not coalesce) + the `gk_geometry_source` stamp; the NaN guard; the coordinate-coherence check; and
— **required** — the **ADR-025 interplay** paragraph (spec §2.3): this is a *transient
scoring-time view*; canonical coordinates are **never** written back, so ADR-025's
never-mutate-canonical fence is intact, and the side-band idiom was rejected because it would push
provenance policy into the metric engine.

Three further items that **must** appear:

1. **The fourth consumer (S5).** `scripts/validate_xtgk_possession_value.py:101` — the deep-zone
   **gate** script — also imports `load_xtgk_cohort`, so the in-loader override changes **its**
   input too. That is fine this cycle (the gate is not re-run, and the fit seams drop NaN rows
   either way — T4). But state plainly: **a future gate re-run would fit `V` on different
   coordinates than the 4.42.0 run did**, so its numbers are not directly comparable to the
   recorded GO-leaning result.
2. **Non-goal #2 gets a home.** Spec §3 says ρ's mean-imputation is "documented, not altered" but
   nothing documents it. Add one line to `GkRetentionModel.predict_proba`'s docstring: non-finite
   features are imputed to the **training mean** (neutral post-standardisation), which is why
   `compute_xt_gk_v2` masks non-finite rows **upstream** rather than relying on this behaviour.
3. **Hyrum note for the lakehouse.** On pandas 3.0 the `gk_geometry_source` column materialises as
   `str` dtype rather than `object`. Anything asserting `dtype == object` on it will break.

- [ ] **Step 2: Version bump — all five must agree**

- `pyproject.toml`: `version = "4.46.0"`
- `silly_kicks/__init__.py`: `__version__ = "4.46.0"`
- `CHANGELOG.md`: new `## [4.46.0] — 2026-07-12` section
- `TODO.md`: header `**Current release**: silly-kicks 4.46.0 ...`
- `uv.lock`: run `uv lock`

- [ ] **Step 3: TODO.md — the National Park line + the F7 deferral**

Fix the doc-vs-reality defect at `TODO.md:28`: it claims TF-19's arms are "**UNBLOCKED**" while the
shipped weights record `"tf19_ready": false`
(`silly_kicks/tracking/_xcross_weights/default/metrics.json:134`). Correct it to state that the
xCross arm's pre-registered TF-19 viability gate **failed** and TF-19 consumption is gated on GK
feature-engineering.

Add the deferred F7 item under Technical Debt:

```markdown
- **`flat_zones` `nan_ok` hardening (deferred from 4.46.0).** The NaN->zone-176 trap is currently
  closed by a corrected docstring + `finite_coord_mask` at the one scoring seam. A
  `nan_ok: bool = False` parameter (default raises; the three NaN-tolerant fit seams `_markov.py:65`,
  `_empirical.py:83`, `_diagnostics.py:123` pass `True`) would make the pit-of-failure structurally
  hard to enter rather than merely documented. Deferred because it perturbs the exact fit seams whose
  byte-identity licenses "the deep-zone gate need not be re-run" (ADR-036 non-goal #1).
```

- [ ] **Step 4: CLAUDE.md**

Update the `xT-GK v2 possession value (xtgk/)` bullet: the new public surface
(`apply_resolved_gk_geometry`, `finite_coord_mask`, `gk_geometry_source`), the NaN guard, the
coherence check, and the **re-materialize trigger** for the lakehouse (with the §5 handoff caveat:
the lakehouse must keep `is_gk_distribution` — or the stamp — on the frame it passes, or the
attestation cannot protect it).

- [ ] **Step 5: Stage**

```bash
git add docs/superpowers/adrs CHANGELOG.md TODO.md CLAUDE.md pyproject.toml silly_kicks/__init__.py uv.lock
```

---

## Task 14: `/final-review`, C4, single commit, PR

- [ ] **Step 1: Run `/final-review`**

This is **mandatory** (owner policy) and includes the C4 Phase-4 architecture check. No new
action-coupled aggregator ships here, so the C4 count **stays 28** — verify `docs/c4/architecture.dsl`
still says 28 and does not need a regen.

- [ ] **Step 2: Re-run the full gate one last time**

```bash
python -m ruff check silly_kicks tests scripts
python -m ruff format --check silly_kicks tests scripts
python -m pyright
python -m pytest tests/ -m "not e2e" -q
```
Expected: all clean.

- [ ] **Step 3: ONE commit for the whole branch**

```bash
git add -A
git commit -m "$(cat <<'EOF'
fix(xtgk): resolved GK-distribution origins -- v2 no longer scores 24% of its domain at a fabricated zone (ADR-036, PR-S113)

flat_zones' NaN->0 mapping is a FIT-PATH contract (every fit seam dropna's), but the
scoring seam _metric.py:56-57 dropped nothing -- so a NaN coordinate was fabricated into
grid zone 176 (the own-corner cell) and scored as a real number. Measured on live gold:
24.4% of the Gradient Sports GK-distribution domain (60.2% of its goal-kicks) and, via a
second route, 21.5% of SkillCorner's (its native goal-kick origin is the broadcast BALL,
not the keeper -- ADR-024/PR-S104). The resolved keeper origins were already materialized
in fct_action_context (xt_gk_origin_x/_y, PR-S101); the v2 loader simply never SELECTed
them, while the v1 comparator in the 4.45.0 head-to-head did -- so that comparison was
never apples-to-apples.

- apply_resolved_gk_geometry: pure, OVERRIDE-not-coalesce (SkillCorner's raw origin is
  present AND wrong, so a fillna would silently miss it), + a gk_geometry_source stamp.
- compute_xt_gk_v2: NaN guard (emit NaN, never fabricate a zone; rho is no longer scored
  on non-finite rows, closing _retention.py:81's silent mean-imputation), a
  coordinate-coherence check (actions and retention_features cannot diverge), and a
  warn-once attestation.
- Both Databricks loaders apply the helper, so all three consuming scripts inherit it.
- rho retrained on the corrected cohort; SP5 re-run under both rho vintages.

The deep-zone gate is NOT re-run: every fit seam drops NaN coords, regression-gated by
tests/xtgk/test_deep_zone_gate_nan_invariance.py.

xT-GK v2 re-materialize trigger (opt-in; NOT a forced VAEP retrain).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push and open the PR**

```bash
git push -u origin pr-s113-xtgk-v2-resolved-origin
gh pr create --title "fix(xtgk): resolved GK-distribution origins -- xT-GK v2 (ADR-036, PR-S113)" --body "$(cat <<'EOF'
## What

xT-GK v2 was scoring ~24% of its GK-distribution domain at a **fabricated grid zone**.

`flat_zones` maps a NaN coordinate to `(0.0, 0.0)` -> **zone 176** (the own-corner cell). That is
safe at every *fitting* seam, because each one drops NaN coords. It is unsafe at the one *scoring*
seam, `_metric.py:56-57`, which dropped nothing and emitted a real number for every row.

Meanwhile the **resolved keeper origins already existed** in `fct_action_context`
(`xt_gk_origin_x/_y`, shipped by PR-S101/4.36.0) -- in the very table the v2 loader already JOINs.
It simply never SELECTed them. The v1 comparator in the 4.45.0 head-to-head *did*, so that
comparison was never apples-to-apples.

## Measured on live gold

| provider | domain | affected | how |
|---|---|---|---|
| gradientsports | 3874 | **946 (24.4%)** -- incl. **595/988 goal-kicks (60.2%)** | NaN origin -> fabricated zone 176 |
| skillcorner | 5487 | **1181 (21.5%)** -- every goal-kick | native origin is the broadcast **BALL**, not the keeper (ADR-024/PR-S104): present, finite, ~10-20 m wrong |

595 of the 946 GS rows have a resolved origin sitting unused in gold; 351 are genuinely
unresolvable and are now honest NaN rather than a fabricated number.

## Changes

- **`apply_resolved_gk_geometry`** (new, pure, public): **OVERRIDE, not coalesce** -- a `fillna`
  would rescue GS's NaN origins and *silently* leave SkillCorner's present-and-wrong ones. Emits a
  per-row `gk_geometry_source` stamp.
- **`compute_xt_gk_v2`**: NaN guard (emit NaN, never fabricate a zone); rho is no longer scored on
  non-finite rows, closing `_retention.py:81`'s silent mean-imputation; a **coordinate-coherence
  check** so `actions` and `retention_features` cannot describe different frames; and a warn-once
  attestation.
- **Both Databricks loaders** apply the helper, so all three consuming scripts inherit it.
- **rho retrained** on the corrected cohort (SkillCorner's variant had been trained on 1181
  goal-kicks with the wrong geometry).
- **SP5 re-run** under both rho vintages so the delta is attributable.

## Not re-run: the deep-zone gate

Every fit seam drops NaN coords, so the fitted `V` surface is clean and the GO-leaning gate verdict
stands. That is **regression-gated**, not asserted:
`tests/xtgk/test_deep_zone_gate_nan_invariance.py`.

## Consumer impact

**xT-GK v2 re-materialize trigger** (opt-in; **not** a forced VAEP retrain). The lakehouse must call
`apply_resolved_gk_geometry` before `compute_xt_gk_v2`, and must keep `is_gk_distribution` (or the
stamp) on the frame it passes -- otherwise the attestation cannot protect it.

Spec: `docs/superpowers/specs/2026-07-12-xtgk-v2-resolved-origin-design.md`
Plan: `docs/superpowers/plans/2026-07-12-xtgk-v2-resolved-origin.md`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Do NOT tag before main CI is green.** Wait for the merge + green main run.

---

## Self-review notes

- **Spec coverage:** §2.1 → Tasks 1–5; §2.2 → Task 8; §2.3 → Task 13 Step 1; §2.4 → Task 11;
  §2.5 → Tasks 9 + 12; §3 non-goal #1 → Task 6 (T4); §4 T1–T7 → Tasks 4, 2, 1, 6, 5, 7, 11;
  §5 handoff caveat → Task 13 Step 4; §6 → Task 13; §7/§8 → no code.
- **Deliberate ordering:** Task 3 (single-source `_coord_derived`) precedes Task 5 because the
  coherence check must recompute *exactly* what the trainer computes — recomputing it independently
  would reintroduce the very drift class this PR fixes.
- **Task 6 must pass first try.** If it fails, non-goal #1 is false and the deep-zone gate is
  contaminated too — escalate, do not adjust the test.
- **Cross-session review (round 1, by execution).** Task 6's fixture had a **triple fault**, all
  three reproduced in a real venv and all three now fixed in-place: (1) it **errored** on the ADR-028
  orientation guard, because a NaN-`start_x` shot counts as own-half, so the escalation clause would
  have misfired on a fixture artifact; (2) it was **vacuous** — `xg` was drawn independently of
  `type_id`, so **0 of 18** shots carried a reward and all three V surfaces were identically zero,
  making the invariance assertion `allclose(0, 0)`; (3) the property was **false as designed**,
  because `PressureLevels` was refit per leg and the moved cutpoints flipped 8/240 rows' terciles.
  The corrected fixture is verified by execution: non-vacuous (10/17/12 non-zero cells), invariant
  in surface **and** support, and — the check the original lacked entirely — **it moves when the
  NaN rows are fabricated to zone 176**, proving it can catch the regression it guards.
  Task 12 also destroyed its own leg-1 artifacts (`_write_report` overwrites) and had no scripted
  source for the NaN-out census; both are now scripted steps. Two ruff traps (E402 import placement
  in Task 3, F811 duplicate import in Task 4) would have failed Task 10.
