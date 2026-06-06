# Tracking id-dtype safety contract — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every tracking-feature id comparison and id-valued merge key dtype-safe, so a caller that supplies string ids (or a string `home_team_id`) gets correct features instead of silently-wrong ones — plus an opt-in loud validator.

**Architecture:** One shared private primitive (`silly_kicks/tracking/_id_compat.py`) defines "id identity" once: a single `_canonical` truth (scalar + vectorized), comparison helpers (`ids_equal`/`ids_differ`/`ids_match`/`same_id`), and a pre-merge `align_join_keys`. All have a same-dtype fast path (zero cost when dtypes already match). A red-first **asymmetric** behavioral gate (numeric actions × string frames) drives the seam fixes: it enumerates which aggregators still fail. A standalone public `validate_id_dtypes` mirrors ADR-017's `validate_time_base`.

**Tech Stack:** Python 3.10, pandas 2.3.x, numpy ≥2.0, pytest, ruff 0.15.7, pyright 1.1.409.

**Spec:** `docs/superpowers/specs/2026-06-06-tracking-id-dtype-contract-design.md`

---

## Workflow notes (READ FIRST — repo overrides the skill default)

- **No per-task git commits.** This repo's solo-maintainer workflow (and the commit-sentinel hook) requires a **single feature commit at the end, gated on explicit owner approval**. Each task below ends with a **Checkpoint (stage only — do NOT commit)**. The final commit is Task 14, and it must NOT run until the owner explicitly approves the diff. Never create the commit sentinel (`~/.claude-git-approval`) without an explicit per-commit "yes".
- **Branch:** create `feat/tracking-id-dtype-contract` off `main` before Task 1 (`git switch -c feat/tracking-id-dtype-contract`). Branch creation is not gated.
- **Venv:** use `.venv` (uv CPython 3.10.19). Run pytest as `.venv/Scripts/python.exe -m pytest ...`.
- **Read exit codes honestly.** Never pipe pytest to `tail`/`head` and trust the exit code. Run unpiped, or append `; echo "EXIT: $?"`.
- **Lint parity before the final commit (Task 13):** `ruff check`, `ruff format --check`, `pyright` over the WHOLE `silly_kicks/` package, not just edited files.

## File structure

| File | Responsibility | Action |
|---|---|---|
| `silly_kicks/tracking/_id_compat.py` | The shared id-identity primitive (canonicalization + comparison + join-key alignment) | **Create** |
| `silly_kicks/tracking/schema.py` | Add `IdDtypeDiagnosis` frozen dataclass | Modify |
| `silly_kicks/tracking/utils.py` | `_diagnose_id_dtypes` + `validate_id_dtypes`; apply helpers in `_resolve_action_frame_context` + `play_left_to_right` + linker merge | Modify |
| `silly_kicks/tracking/__init__.py` | Export `validate_id_dtypes`, `IdDtypeDiagnosis` | Modify |
| `silly_kicks/tracking/_defensive_line.py`, `_kernels.py`, `_off_ball_runs.py`, `_gk_influence.py`, `_line_breaking.py`, `_player_influence.py`, `_ghost_gk.py`, `features.py`, `direction.py`, `_ball_carrier.py` | Apply the helpers at each id seam the gate flags; replace ad-hoc `astype(str)` patches with `align_join_keys` | Modify |
| `tests/tracking/test_id_compat.py` | Unit tests for the primitive | Create |
| `tests/tracking/test_id_dtype_invariance.py` | Red-first **asymmetric** behavioral gate (primary CI gate) | Create |
| `tests/tracking/test_id_compat_lint.py` | AST lint backstop | Create |
| `tests/tracking/test_validate_id_dtypes.py` | Validator raise/warn/ignore + diagnosis shape | Create |
| `docs/superpowers/adrs/ADR-019-tracking-id-dtype-contract.md`, `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `uv.lock` | Release artifacts | Modify |

## Canonical transform patterns (applied at the seams in Tasks 6–10)

These are the exact rewrites. The site lists are the **known** seams; the gate (Task 5) + AST lint (Task 10) are the completeness backstop for any missed one.

- **Vectorized Series scalar-match** `series == home_team_id` → `ids_match(series, home_team_id)`
  (e.g. `utils.py:156`, `_defensive_line.py:62`).
- **Numpy-array scalar-match** `arr == team_id` → `ids_match(arr, team_id)` (`ids_match` accepts
  `pd.Series | np.ndarray`) (e.g. `_gk_influence.py:337`, `_player_influence.py` surface masks).
- **Scalar Python equality** `if action_team == home_team_id:` → `if same_id(action_team, home_team_id):`
  (e.g. `_defensive_line.py:69`, `_off_ball_runs.py:331,353`, `_gk_influence.py:307,350`,
  `_line_breaking.py:241`, `_player_influence.py:120`, `_ghost_gk.py:759`).
- **Merged opponent mask** `merged[merged["team_id_dl"] != merged["team_id_action"]]` →
  `merged[ids_differ(merged["team_id_dl"], merged["team_id_action"])]`
  (e.g. `_kernels.py:861`, `_off_ball_runs.py:291`, `features.py:3786`).
- **`_resolve` masks** (`utils.py:600,610,622`) → `ids_equal` / `ids_differ`, with the columns
  coerced **once** into locals (A1 de-dup).
- **Pre-merge join-key alignment** — replace each ad-hoc `df["game_id"] = df["game_id"].astype(str)`
  block (`features.py:3775-3776`, `_kernels.py:850`, `_off_ball_runs.py:308-315`) with
  `linked, other = align_join_keys(linked, other, ["game_id", "period_id", "frame_id"])` immediately
  before the merge.

---

## Task 1: `_canonical` truth + scalar/vectorized canonicalization

**Files:**
- Create: `silly_kicks/tracking/_id_compat.py`
- Test: `tests/tracking/test_id_compat.py`

- [ ] **Step 1: Write the failing test** (canonicalization table, both entry points)

```python
# tests/tracking/test_id_compat.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking import _id_compat as idc

# (input, expected canonical) — the single truth, asserted at BOTH entry points
CANON_CASES = [
    (366, "366"),
    (366.0, "366"),
    (np.int64(366), "366"),
    (pd.array([366], dtype="Int64")[0], "366"),
    ("366", "366"),
    ("DFL-CLU-A", "DFL-CLU-A"),
    (366.5, "366.5"),
]

@pytest.mark.parametrize("raw,expected", CANON_CASES)
def test_canonical_id_scalar(raw, expected):
    assert idc.canonical_id(raw) == expected

def test_canonical_id_scalar_na():
    assert idc.canonical_id(None) is pd.NA
    assert idc.canonical_id(np.nan) is pd.NA
    assert idc.canonical_id(pd.NA) is pd.NA

def test_canonical_id_series_matches_scalar():
    # single-truth: vectorized output == scalar `_canonical`, fed per NATURAL dtype (real id
    # columns are homogeneous, not a mixed object dump — an integral float in an object column
    # is the one documented pathological divergence, excluded here).
    for dtype, vals, exp in [
        ("int64", [366, 7], ["366", "7"]),
        ("Int64", [366, 7], ["366", "7"]),
        ("float64", [366.0, 366.5], ["366", "366.5"]),
        ("object", ["366", "DFL-CLU-A"], ["366", "DFL-CLU-A"]),
    ]:
        out = idc.canonical_id_series(pd.Series(vals, dtype=dtype))
        assert out.tolist() == exp
        assert [idc.canonical_id(v) for v in vals] == exp  # scalar entry agrees

def test_canonical_id_series_integral_float_collapse():
    # the str(366.0)->"366.0" trap: must collapse
    out = idc.canonical_id_series(pd.Series([366.0, 366.5, np.nan], dtype="float64"))
    assert out.tolist()[:2] == ["366", "366.5"]
    assert out.tolist()[2] is pd.NA

def test_canonical_id_series_int64_and_Int64():
    assert idc.canonical_id_series(pd.Series([366], dtype="int64")).tolist() == ["366"]
    assert idc.canonical_id_series(pd.Series([366], dtype="Int64")).tolist() == ["366"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -v ; echo "EXIT: $?"`
Expected: FAIL — `ModuleNotFoundError` / `AttributeError: module ... has no attribute 'canonical_id'`.

- [ ] **Step 3: Write minimal implementation**

```python
# silly_kicks/tracking/_id_compat.py
"""Dtype-safe id identity for tracking-feature seams (ADR-019).

Defines "id identity" once: a single canonicalization truth (scalar + vectorized),
comparison helpers, and pre-merge join-key alignment. All comparison/align helpers
have a same-dtype fast path (zero cost when both sides already share a numpy kind).

See docs/superpowers/specs/2026-06-06-tracking-id-dtype-contract-design.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _canonical(x) -> str | type(pd.NA):
    """Single-element canonical id key. NA/None/NaN -> pd.NA (never matches).

    Integral numerics collapse: 366, 366.0, np.int64(366), Int64(366), "366" -> "366".
    Genuine strings pass through. Non-integral floats stringify as-is (won't match).
    """
    if x is None or x is pd.NA:
        return pd.NA
    # floats / numpy floats: NaN -> NA; integral -> collapse to int string
    if isinstance(x, (float, np.floating)):
        if np.isnan(x):
            return pd.NA
        if float(x).is_integer():
            return str(int(x))
        return str(x)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x)


def canonical_id(x) -> str | type(pd.NA):
    """Scalar entry point — delegates to the single `_canonical` truth."""
    return _canonical(x)


def canonical_id_series(s: pd.Series) -> pd.Series:
    """Vectorized canonicalization, NaN-safe, object dtype. Matches `canonical_id`
    element-wise. Avoids naive `.astype(str)` (which yields '366.0', not '366')."""
    s = pd.Series(s)
    out = pd.Series(pd.NA, index=s.index, dtype="object")
    notna = s.notna()
    if not notna.any():
        return out
    vals = s[notna]
    kind = vals.dtype.kind
    if kind == "f":
        integral = vals == vals.round()
        out.loc[vals.index[integral]] = vals[integral].astype("Int64").astype("string").astype("object")
        out.loc[vals.index[~integral]] = vals[~integral].astype("string").astype("object")
    elif kind in ("i", "u") or str(vals.dtype) in ("Int64", "Int32", "Int16", "Int8"):
        out.loc[vals.index] = vals.astype("Int64").astype("string").astype("object")
    else:
        # object/string: VECTORIZED stringify (A1 — no Python loop). Real object id columns are
        # homogeneous strings (genuine ids or stringified ints); astype("string") matches
        # `_canonical` for str/object-int/non-integral-float elements. The ONLY divergence is an
        # *integral float stored in an object column* (366.0 -> "366.0" vs _canonical's "366") —
        # a doubly-pathological case (float ids in object dtype) that does not occur for real id
        # columns; documented, not Python-looped.
        out.loc[vals.index] = vals.astype("string").astype("object")
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -v ; echo "EXIT: $?"`
Expected: PASS (all canonicalization tests).

- [ ] **Step 5: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/_id_compat.py tests/tracking/test_id_compat.py
```

---

## Task 2: comparison helpers (`ids_equal`, `ids_differ`, `ids_match`, `same_id`)

**Files:**
- Modify: `silly_kicks/tracking/_id_compat.py`
- Test: `tests/tracking/test_id_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_id_compat.py

def test_ids_equal_cross_dtype_and_na():
    a = pd.Series([366, 7], dtype="int64")
    b = pd.Series(["366", "9"], dtype="object")   # asymmetric
    out = idc.ids_equal(a, b)
    assert out.tolist() == [True, False]
    assert out.dtype == np.bool_                    # C1: non-nullable
    # NA never equals anything
    an = pd.Series([366, pd.NA], dtype="Int64")
    bn = pd.Series(["366", "366"], dtype="object")
    assert idc.ids_equal(an, bn).tolist() == [True, False]

def test_ids_differ_both_present_and_left_join_miss():
    # opponent mask: object NaN (unmatched left-join) must NOT count as "differ"
    a = pd.Series(["5", pd.NA, "9"], dtype="object")   # team_id_dl (NaN = unmatched)
    b = pd.Series(["5", "5", "5"], dtype="object")      # team_id_action
    out = idc.ids_differ(a, b)
    assert out.tolist() == [False, False, True]         # row1 NA -> excluded (N1)
    assert out.dtype == np.bool_

def test_ids_match_series_and_array():
    s = pd.Series([366, 7], dtype="Int64")
    assert idc.ids_match(s, "366").tolist() == [True, False]      # string scalar
    arr = np.array(["366", "7"], dtype=object)
    assert idc.ids_match(arr, 366).tolist() == [True, False]      # numpy array side

def test_ids_match_same_kind_fast_path():
    s = pd.Series([366, 7], dtype="int64")
    assert idc.ids_match(s, 366).tolist() == [True, False]

def test_same_id_scalar():
    assert idc.same_id(366, "366") is True
    assert idc.same_id("366", 366) is True
    assert idc.same_id(366, 7) is False
    assert idc.same_id(pd.NA, 366) is False
    assert idc.same_id(366, np.nan) is False

def test_object_object_takes_raw_fast_path_no_canonicalize(monkeypatch):
    # A1: genuine-string providers (sportec/kloppy) must NOT Python-loop through
    # canonical_id_series — object x object is directly comparable.
    calls = {"n": 0}
    real = idc.canonical_id_series
    monkeypatch.setattr(idc, "canonical_id_series",
                        lambda s: (calls.__setitem__("n", calls["n"] + 1), real(s))[1])
    a = pd.Series(["DFL-CLU-A", "DFL-CLU-B"], dtype="object")
    b = pd.Series(["DFL-CLU-A", "DFL-CLU-Z"], dtype="object")
    assert idc.ids_equal(a, b).tolist() == [True, False]
    assert idc.ids_differ(a, b).tolist() == [False, True]
    assert calls["n"] == 0  # zero canonicalization on the object x object path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -k "ids_ or same_id" -v ; echo "EXIT: $?"`
Expected: FAIL — `AttributeError: module ... has no attribute 'ids_equal'`.

- [ ] **Step 3: Write minimal implementation** (append to `_id_compat.py`)

```python
def _directly_comparable(a_dtype, b_dtype) -> bool:
    """True when two columns can be raw-compared/merged WITHOUT canonicalization:
    same numpy kind (Int64 nullable and int64 both report 'i'), OR both object.
    Both-object is safe under the same-provider invariant (C2): two same-provider
    object id columns are both genuine strings, so raw ==/!= is correct — and it
    avoids the object Python-loop regression (A1) for sportec/kloppy."""
    ak, bk = a_dtype.kind, b_dtype.kind
    if ak == "O" and bk == "O":
        return True
    return ak == bk and ak in ("i", "u", "f", "b")


def _as_bool(series: pd.Series) -> pd.Series:
    """Resolve any NA to False and pin non-nullable np.bool_ (C1)."""
    return series.fillna(False).astype(bool)


def _positional(a: pd.Series, b: pd.Series):
    """Positional (not label) alignment — seam columns share a frame; stay consistent (N-a)."""
    return a.reset_index(drop=True), b.reset_index(drop=True)


def ids_equal(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise id equality, dtype-safe, POSITIONAL. NA never equals anything -> False.
    Returns a non-nullable np.bool_ Series."""
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _directly_comparable(a.dtype, b.dtype):
        eq = (pa == pb) & pa.notna() & pb.notna()
        return _as_bool(eq)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    eq = (ca == cb) & ca.notna() & cb.notna()
    return _as_bool(eq)


def ids_differ(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise id inequality requiring BOTH present (opponent mask), POSITIONAL.
    A NA on either side (e.g. an unmatched how='left' row) is NOT "differ" (N1).
    Returns a non-nullable np.bool_ Series."""
    a, b = pd.Series(a), pd.Series(b)
    pa, pb = _positional(a, b)
    if _directly_comparable(a.dtype, b.dtype):
        differ = pa.notna() & pb.notna() & (pa != pb)
        return _as_bool(differ)
    ca, cb = canonical_id_series(pa), canonical_id_series(pb)
    differ = ca.notna() & cb.notna() & (ca != cb)
    return _as_bool(differ)


def ids_match(series, scalar) -> pd.Series:
    """Vectorized `series == scalar`, dtype-safe. `series` may be a pd.Series or
    np.ndarray. Returns a non-nullable np.bool_ Series."""
    s = pd.Series(series)
    key = canonical_id(scalar)
    if key is pd.NA:
        return pd.Series(np.zeros(len(s), dtype=bool), index=s.index)
    # fast path: directly comparable to a 1-element scalar series (incl. object x object)
    scal_s = pd.Series([scalar])
    if _directly_comparable(s.dtype, scal_s.dtype):
        return _as_bool(s == scalar)
    return _as_bool(canonical_id_series(s) == key)


def same_id(a, b) -> bool:
    """Scalar-vs-scalar id equality (groupby-loop comparisons). False if either is NA."""
    ca, cb = _canonical(a), _canonical(b)
    if ca is pd.NA or cb is pd.NA:
        return False
    return bool(ca == cb)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -v ; echo "EXIT: $?"`
Expected: PASS (all).

- [ ] **Step 5: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/_id_compat.py tests/tracking/test_id_compat.py
```

---

## Task 3: `align_join_keys` (pre-merge join-key alignment, M1)

**Files:**
- Modify: `silly_kicks/tracking/_id_compat.py`
- Test: `tests/tracking/test_id_compat.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/tracking/test_id_compat.py

def test_align_join_keys_prevents_merge_raise():
    left = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64"), "v": [10, 20]})
    right = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object"), "w": [100, 200]})
    # raw merge would raise ValueError on int64 x object key
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    out = l2.merge(r2, on="game_id", how="left")
    assert out["w"].tolist() == [100, 200]

def test_align_join_keys_same_kind_noop():
    left = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64")})
    right = pd.DataFrame({"game_id": pd.Series([1, 2], dtype="int64")})
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    # fast path: dtype unchanged (no needless stringify)
    assert l2["game_id"].dtype == np.dtype("int64")
    assert r2["game_id"].dtype == np.dtype("int64")

def test_align_join_keys_object_object_noop():  # N-c: object x object is mergeable as-is
    left = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object")})
    right = pd.DataFrame({"game_id": pd.Series(["1", "2"], dtype="object")})
    l2, r2 = idc.align_join_keys(left, right, ["game_id"])
    assert l2["game_id"].dtype == object and r2["game_id"].dtype == object

def test_align_join_keys_pair_names():  # A3: differently-named keys (frame_id_int vs frame_id)
    left = pd.DataFrame({"frame_id_int": pd.Series([1, 2], dtype="int64"), "v": [10, 20]})
    right = pd.DataFrame({"frame_id": pd.Series(["1", "2"], dtype="object"), "w": [100, 200]})
    l2, r2 = idc.align_join_keys(left, right, [("frame_id_int", "frame_id")])
    out = l2.merge(r2, left_on="frame_id_int", right_on="frame_id", how="left")
    assert out["w"].tolist() == [100, 200]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -k align -v ; echo "EXIT: $?"`
Expected: FAIL — `AttributeError: ... has no attribute 'align_join_keys'`.

- [ ] **Step 3: Write minimal implementation** (append to `_id_compat.py`)

```python
def align_join_keys(left: pd.DataFrame, right: pd.DataFrame,
                    keys: list):
    """Canonicalize id-valued join keys on both sides to a common dtype BEFORE merge.

    `keys` entries are either a `str` (same column name on both sides) or a
    `(left_key, right_key)` tuple for differently-named keys (A3 — e.g. the
    `frame_id_int` (left) vs `frame_id` (right) merge in `_kernels.py`, which
    `pd.merge(left_on=..., right_on=...)` allows but `align_join_keys` could not
    bridge without pair support).

    Per key: if both sides are directly comparable (same numpy kind, OR both object —
    N-b) -> no-op (fast path, zero cost); else coerce both to canonical string.
    Returns (left, right) copies ready for the merge. Prevents the ValueError pandas
    raises on a mixed-dtype merge key (ADR-019 / M1)."""
    left, right = left.copy(), right.copy()
    for k in keys:
        lk, rk = (k, k) if isinstance(k, str) else (k[0], k[1])
        if lk not in left.columns or rk not in right.columns:
            continue
        if _directly_comparable(left[lk].dtype, right[rk].dtype):
            continue
        left[lk] = canonical_id_series(left[lk])
        right[rk] = canonical_id_series(right[rk])
    return left, right
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat.py -v ; echo "EXIT: $?"`
Expected: PASS (all).

- [ ] **Step 5: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/_id_compat.py tests/tracking/test_id_compat.py
```

---

## Task 4: `IdDtypeDiagnosis` + `validate_id_dtypes` + exports

**Files:**
- Modify: `silly_kicks/tracking/schema.py` (add dataclass after `TimeBaseDiagnosis`)
- Modify: `silly_kicks/tracking/utils.py` (add `_diagnose_id_dtypes` + `validate_id_dtypes`)
- Modify: `silly_kicks/tracking/__init__.py` (export both)
- Test: `tests/tracking/test_validate_id_dtypes.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_validate_id_dtypes.py
import warnings
import numpy as np
import pandas as pd
import pytest
from silly_kicks.tracking import validate_id_dtypes, IdDtypeDiagnosis


def _actions(team_dtype, player_dtype):
    return pd.DataFrame({
        "action_id": [0, 1], "period_id": [1, 1],
        "team_id": pd.Series([5, 5], dtype=team_dtype),
        "player_id": pd.Series([10, 11], dtype=player_dtype),
    })

def _frames(team_dtype, player_dtype):
    return pd.DataFrame({
        "period_id": [1, 1], "frame_id": [0, 0],
        "team_id": pd.Series([5, 6], dtype=team_dtype),
        "player_id": pd.Series([10, 20], dtype=player_dtype),
        "is_ball": [False, False],
    })

def test_matched_dtypes_no_mismatch():
    diag = validate_id_dtypes(_actions("int64", "int64"), _frames("int64", "int64"),
                              on_mismatch="raise")
    assert isinstance(diag, IdDtypeDiagnosis)
    assert not diag.has_mismatch

def test_mismatch_raises_by_default():
    with pytest.raises(ValueError, match="id dtype"):
        validate_id_dtypes(_actions("int64", "int64"), _frames("object", "object"))

def test_mismatch_warn_returns_diag():
    with pytest.warns(UserWarning, match="id dtype"):
        diag = validate_id_dtypes(_actions("int64", "int64"), _frames("object", "object"),
                                  on_mismatch="warn")
    assert diag.has_mismatch
    assert "team_id" in diag.coercion_required_columns

def test_home_team_id_axis_flagged():
    diag = validate_id_dtypes(_actions("int64", "int64"), _frames("int64", "int64"),
                              home_team_id="5", on_mismatch="ignore")
    assert diag.home_team_id_requires_coercion
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_validate_id_dtypes.py -v ; echo "EXIT: $?"`
Expected: FAIL — `ImportError: cannot import name 'validate_id_dtypes'`.

- [ ] **Step 3a: Add `IdDtypeDiagnosis` to `schema.py`** (after `TimeBaseDiagnosis`, ~line 215)

```python
@dataclasses.dataclass(frozen=True)
class IdDtypeDiagnosis:
    """Action-vs-frame id-dtype compatibility diagnosis (ADR-019).

    Produced by ``silly_kicks.tracking.utils._diagnose_id_dtypes`` and surfaced by
    ``validate_id_dtypes``. The tracking-feature seams coerce id dtypes transparently;
    this is the opt-in loud guard for a dtype-sensitive consumer (e.g. the lakehouse).

    Attributes:
        per_column: id col -> (action_dtype_str, frame_dtype_str).
        coercion_required_columns: cols whose action/frame numpy kinds differ
            (would silently mis-compare / raise on merge without coercion).
        home_team_id_dtype: dtype/kind of the scalar arg, if supplied (else None).
        home_team_id_requires_coercion: scalar kind vs frame team_id kind differ.
        message: human-readable summary.
    """

    per_column: dict[str, tuple[str, str]]
    coercion_required_columns: tuple[str, ...]
    home_team_id_dtype: str | None
    home_team_id_requires_coercion: bool
    message: str

    @property
    def has_mismatch(self) -> bool:
        return len(self.coercion_required_columns) > 0 or self.home_team_id_requires_coercion
```

- [ ] **Step 3b: Add diagnosis + validator to `utils.py`** (near `validate_time_base`, ~line 476). Add `IdDtypeDiagnosis` to the existing `from .schema import ...` line.

```python
_ID_COLUMNS = ("player_id", "team_id", "defending_gk_player_id")


def _diagnose_id_dtypes(actions, frames, home_team_id=None):
    from .schema import IdDtypeDiagnosis

    per_column: dict[str, tuple[str, str]] = {}
    coercion: list[str] = []
    for col in _ID_COLUMNS:
        if col in actions.columns and col in frames.columns:
            ad, fd = actions[col].dtype, frames[col].dtype
            per_column[col] = (str(ad), str(fd))
            if ad.kind != fd.kind:
                coercion.append(col)
    ht_dtype = None
    ht_coerce = False
    if home_team_id is not None and "team_id" in frames.columns:
        ht_dtype = type(home_team_id).__name__
        scal_kind = pd.Series([home_team_id]).dtype.kind
        ht_coerce = scal_kind != frames["team_id"].dtype.kind
    bits = [f"{c}: action={per_column[c][0]} vs frame={per_column[c][1]}" for c in coercion]
    if ht_coerce:
        bits.append(f"home_team_id={ht_dtype} vs frame team_id={frames['team_id'].dtype}")
    message = (
        "id dtype mismatch (coercion applied at seams): " + "; ".join(bits)
        if bits else "id dtypes compatible"
    )
    return IdDtypeDiagnosis(per_column, tuple(coercion), ht_dtype, ht_coerce, message)


def validate_id_dtypes(actions, frames, *, home_team_id=None, on_mismatch="raise"):
    """Pre-flight guard that actions + frames share comparable id dtypes (ADR-019).

    The tracking-feature seams coerce id dtypes transparently, so this is an OPT-IN
    loud guard, not a required call. Mirrors :func:`validate_time_base`: ``on_mismatch``
    defaults to ``"raise"`` (an explicitly-invoked assertion fails loud); ``"warn"`` /
    ``"ignore"`` available. The diagnosis is returned under all policies.
    """
    diag = _diagnose_id_dtypes(actions, frames, home_team_id=home_team_id)
    if diag.has_mismatch:
        if on_mismatch == "raise":
            raise ValueError(f"validate_id_dtypes: {diag.message}")
        if on_mismatch == "warn":
            warnings.warn(f"validate_id_dtypes: {diag.message}", UserWarning, stacklevel=2)
    return diag
```

- [ ] **Step 3c: Export from `__init__.py`** — add `"IdDtypeDiagnosis"` and `"validate_id_dtypes"` to `__all__` and the corresponding imports (alongside `TimeBaseDiagnosis` / `validate_time_base`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_validate_id_dtypes.py -v ; echo "EXIT: $?"`
Expected: PASS (all 4).

- [ ] **Step 5: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/schema.py silly_kicks/tracking/utils.py silly_kicks/tracking/__init__.py tests/tracking/test_validate_id_dtypes.py
```

---

## Task 5: Red-first ASYMMETRIC behavioral gate (the driver, M2)

**Files:**
- Create: `tests/tracking/test_id_dtype_invariance.py`

This test is written to **fail against current code** and drives Tasks 6–9. It runs each public aggregator on an all-numeric baseline and on asymmetric-dtype permutations, asserting identical output.

- [ ] **Step 1: Write the gate**

```python
# tests/tracking/test_id_dtype_invariance.py
"""ADR-019 primary gate: feature outputs must not depend on id dtype.

The production failure is ASYMMETRIC (numeric actions x string frames). Casting BOTH
sides to string would be homogeneous and pass on broken code (object==object works) —
so we vary the two sides INDEPENDENTLY and assert every permutation equals the
all-numeric baseline. home_team_id dtype is a SEPARATE axis.
"""
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import features as F
from tests.tracking.conftest_id_dtype import (  # tiny shared fixture (Step 1b)
    make_actions, make_frames, AGGREGATORS, REGISTERED_AGGREGATORS, NON_LINKED_AGGREGATORS,
)

# N-e: aggregators return rows deterministically ordered by action_id (the SPADL contract),
# so baseline and variant align positionally after reset_index — a same-rows-different-order
# variant would (correctly) be treated as a regression. Assumption stated, not silently relied on.

# Entity-id columns we CAST when building the asymmetric variants (input side).
STRINGIFY_COLS = ["team_id", "player_id"]


def _is_id_col(name: str) -> bool:
    """B1: any id-valued OUTPUT column must be excluded from the value comparison —
    a numeric baseline (99) vs a string variant ("99") legitimately differ, and
    assert_frame_equal(check_dtype=False) still compares VALUES. Excluding only
    team_id/player_id is too narrow: aggregators may surface defending_gk_player_id,
    *_id provenance (frame_id), or other ids. Generic id-name rule instead."""
    return "team_id" in name or "player_id" in name or name.endswith("_id")


def _stringify(df, cols):
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64").astype("string").astype("object")
    return df


# (actions_to_string, frames_to_string, home_team_id_to_string)
PERMUTATIONS = [
    (False, True, False),   # numeric actions x STRING frames  <- the lakehouse bug
    (True, False, False),   # STRING actions x numeric frames  <- the reverse
    (False, False, True),   # string home_team_id only         <- scalar-arg axis
    (True, True, True),     # all string                       <- homogeneous sanity
]


@pytest.mark.parametrize("agg", AGGREGATORS, ids=lambda a: a.__name__)
@pytest.mark.parametrize("act_str,frm_str,ht_str", PERMUTATIONS)
def test_aggregator_id_dtype_invariant(agg, act_str, frm_str, ht_str):
    base_actions, base_frames, home = make_actions(), make_frames(), 5
    baseline = agg(base_actions.copy(), base_frames.copy(), home)

    a = _stringify(base_actions, STRINGIFY_COLS) if act_str else base_actions.copy()
    f = _stringify(base_frames, STRINGIFY_COLS) if frm_str else base_frames.copy()
    h = "5" if ht_str else home
    variant = agg(a, f, h)

    # compare FEATURE columns only — every id-valued column legitimately differs
    # numeric-vs-string and must be excluded (B1, generic id-name rule).
    feat_cols = [c for c in baseline.columns if not _is_id_col(c)]
    pd.testing.assert_frame_equal(
        baseline[feat_cols].reset_index(drop=True),
        variant[feat_cols].reset_index(drop=True),
        check_dtype=False, check_like=True,
    )


def test_enumerated_surface_equals_registered():  # B3 meta-assertion
    enumerated = {a.__name__ for a in AGGREGATORS}
    covered = enumerated | set(NON_LINKED_AGGREGATORS)
    assert covered == REGISTERED_AGGREGATORS, (
        "id-dtype gate must cover every registered public aggregator (in AGGREGATORS or, "
        "with a justification, NON_LINKED_AGGREGATORS); "
        f"uncovered: {REGISTERED_AGGREGATORS - covered}"
    )
```

- [ ] **Step 1b: Create the shared fixture** `tests/tracking/conftest_id_dtype.py`

```python
# tests/tracking/conftest_id_dtype.py
"""Tiny shared fixture + the enumerated aggregator surface for the id-dtype gate.

`AGGREGATORS` is the list under test. `REGISTERED_AGGREGATORS` is derived from the
library's public `add_*` exports so the meta-assertion (B3) catches an aggregator
added but not wired into the gate.
"""
import numpy as np
import pandas as pd
from silly_kicks.tracking import features as F

# Build the smallest actions+frames that exercise actor/opponent/GK/defensive-line/
# possession resolution: 2 actions, 2 teams (5,6), a GK, a couple outfielders, ball.
def make_actions() -> pd.DataFrame:
    return pd.DataFrame({
        "game_id": [1, 1], "action_id": [0, 1], "period_id": [1, 1],
        "time_seconds": [10.0, 20.0],
        "team_id": pd.Series([5, 5], dtype="int64"),
        "player_id": pd.Series([10, 11], dtype="int64"),
        "start_x": [50.0, 60.0], "start_y": [34.0, 30.0],
        "end_x": [60.0, 80.0], "end_y": [30.0, 40.0],
        "type_id": [0, 11], "result_id": [1, 1], "bodypart_id": [0, 0],
    })

def make_frames() -> pd.DataFrame:
    # one frame per action time, both teams present + GK + ball
    rows = []
    for t in (10.0, 20.0):
        for (pid, team, gk, x) in [(10,5,False,55),(11,5,False,40),(20,6,False,70),
                                    (1,5,True,5),(2,6,True,100)]:
            rows.append(dict(game_id=1, period_id=1, frame_id=int(t), time_seconds=t,
                             frame_rate=25.0, player_id=pid, team_id=team, is_ball=False,
                             is_goalkeeper=gk, x=float(x), y=34.0, z=0.0, speed=1.0,
                             speed_source="native", ball_state="alive",
                             team_attacking_direction="ltr", confidence=None,
                             visibility=None, source_provider="gradientsports",
                             is_goalkeeper_source="native"))
        rows.append(dict(game_id=1, period_id=1, frame_id=int(t), time_seconds=t,
                         frame_rate=25.0, player_id=pd.NA, team_id=pd.NA, is_ball=True,
                         is_goalkeeper=False, x=58.0, y=32.0, z=0.0, speed=5.0,
                         speed_source="native", ball_state="alive",
                         team_attacking_direction="ltr", confidence=None, visibility=None,
                         source_provider="gradientsports", is_goalkeeper_source="native"))
    f = pd.DataFrame(rows)
    f["player_id"] = f["player_id"].astype("Int64")
    f["team_id"] = f["team_id"].astype("Int64")
    return f

# Aggregators under test. Real signatures DIFFER (verified): add_action_context takes NO
# home_team_id (resolves teams from the action/frame id columns); add_defensive_line /
# add_off_ball_context take keyword-only home_team_id (+ n); add_gk_influence /
# add_player_influence take `xt: ExpectedThreat` as a required POSITIONAL 3rd arg + home_team_id.
# So each entry is an explicit per-aggregator ADAPTER with the uniform gate signature
# (actions, frames, home_team_id) -> DataFrame. `_named` preserves __name__ for the
# meta-assertion + test ids.

import functools

@functools.cache
def _xt():
    # Any VALID fitted xT suffices — the gate asserts dtype-invariance with a shared artifact,
    # so this is NOT required to track tests/conftest.py::fitted_xt (N2). Built LAZILY (N1) so an
    # xT-API problem breaks only the two influence tests, not collection of the whole gate.
    import numpy as np
    from silly_kicks.xthreat import ExpectedThreat
    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    return xt

def _named(fn, name):
    fn.__name__ = name
    return fn

AGGREGATORS = [
    _named(lambda a, f, home_team_id: F.add_action_context(a, f), "add_action_context"),
    _named(lambda a, f, home_team_id: F.add_defensive_line(a, f, home_team_id=home_team_id, n=4),
           "add_defensive_line"),
    _named(lambda a, f, home_team_id: F.add_off_ball_context(a, f, home_team_id=home_team_id),
           "add_off_ball_context"),
    # added in Tasks 7-9 (xT built lazily via _xt()):
    # _named(lambda a, f, home_team_id: F.add_ghost_gk(a, f), "add_ghost_gk"),
    # _named(lambda a, f, home_team_id: F.add_gk_influence(a, f, _xt(), home_team_id=home_team_id),
    #        "add_gk_influence"),
    # _named(lambda a, f, home_team_id: F.add_player_influence(a, f, _xt(), home_team_id=home_team_id),
    #        "add_player_influence"),
]

# Public add_* surface — the meta-assertion (B3) checks AGGREGATORS covers all LINKED ones.
REGISTERED_AGGREGATORS = {
    name for name in dir(F)
    if name.startswith("add_") and callable(getattr(F, name))
}

# Aggregators that legitimately compare NO ids (e.g. frames-only / pure-geometry). Each entry
# MUST carry a one-line "compares no ids" justification (N-d). The AST lint (Task 10) is the
# cross-check: an allowlisted aggregator whose module the lint flags is a contradiction → fail.
NON_LINKED_AGGREGATORS: dict[str, str] = {
    # "add_xxx": "reason it compares no action/frame/home_team ids",
}
```

> NOTE: `REGISTERED_AGGREGATORS` will initially be a SUPERSET of `AGGREGATORS` (the meta-test goes red). That is intentional — it is the running TODO of seams still to wire in. Tasks 6–9 grow `AGGREGATORS` until the meta-assertion passes. Some `add_*` may legitimately not take `(actions, frames, home_team_id)` (e.g. frames-only); for those, add to an explicit `NON_LINKED_AGGREGATORS` allowlist in this file with a one-line reason, and subtract it in the meta-assertion. Document each exclusion.

- [ ] **Step 2: Run against CURRENT code — confirm RED**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py -v ; echo "EXIT: $?"`
Expected: **FAIL** — asymmetric variants mis-resolve (feature columns differ) and/or merges raise `ValueError`, and the meta-assertion reports uncovered aggregators. **Record which aggregators fail and how** (mis-compare vs merge-raise) — this is the worklist for Tasks 6–9. If it does NOT go red, the gate is wrong (re-check the permutation asymmetry) before proceeding.

- [ ] **Step 3: Checkpoint (stage only — do NOT commit)**

```bash
git add tests/tracking/test_id_dtype_invariance.py tests/tracking/conftest_id_dtype.py
```

---

## Task 6: Fix `_resolve_action_frame_context` masks + linker merge (A1 de-dup + align)

**Files:**
- Modify: `silly_kicks/tracking/utils.py:585-625` (the `long` build + 3 masks) and the linker merge
- Modify: `silly_kicks/tracking/utils.py:156` (`play_left_to_right`)

- [ ] **Step 1: Apply the transforms.** Add `from ._id_compat import ids_equal, ids_differ, ids_match, align_join_keys` to `utils.py` imports. Then:

`play_left_to_right` (line 156):
```python
# before:  home_player_mask = (~is_ball) & (out["team_id"] == home_team_id)
home_player_mask = (~is_ball) & ids_match(out["team_id"], home_team_id)
```

`_resolve_action_frame_context` — align the merge keys, then coerce each suffixed id column **once** and reuse across masks (A1):
```python
# before the inner merge on ["period_id","frame_id"] (~line 591):
pointer_with_period, frames = align_join_keys(pointer_with_period, frames, ["period_id", "frame_id"])
long = pointer_with_period.merge(frames, on=["period_id", "frame_id"], how="inner",
                                 suffixes=("_action", "_frame"))

# replace the three masks (lines 600/610/622) with helper calls:
if "player_id_frame" in long.columns:
    actor_mask = ids_equal(long["player_id_frame"], long["player_id_action"]) & (~long["is_ball"])
    actor_long = long.loc[actor_mask].copy()
else:
    actor_long = long.iloc[0:0].copy()
...
if "team_id_frame" in long.columns:
    opp_mask = ids_differ(long["team_id_frame"], long["team_id_action"]) & (~long["is_ball"])
    opposite = long.loc[opp_mask].copy()
...
if "defending_gk_player_id" in long.columns and "player_id_frame" in long.columns:
    gk_mask = ids_equal(long["player_id_frame"], long["defending_gk_player_id"]) & (~long["is_ball"])
    defending_gk_rows = long.loc[gk_mask].copy()
```
(`ids_equal`/`ids_differ` already encode the `.notna()` guard, so the explicit `gk_id_action.notna()` is dropped.)

- [ ] **Step 2: Grow the gate** — `add_action_context` is already in `AGGREGATORS`. Run the gate:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py -k action_context -v ; echo "EXIT: $?"`
Expected: the `add_action_context` permutations now **PASS** (previously red).

- [ ] **Step 3: Regression** — the existing action_context tests still pass:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "action_context or resolve" -v ; echo "EXIT: $?"`
Expected: PASS.

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/utils.py
```

---

## Task 7: Fix ghost-GK merge + opponent seam (replace hand-patch with `align_join_keys`)

**Files:**
- Modify: `silly_kicks/tracking/features.py:3775-3786`

- [ ] **Step 1: Apply.** Add `from ._id_compat import ids_differ, align_join_keys` (or extend an existing `_id_compat` import). Replace the `game_id.astype(str)` block (3775-3776) and the merge + opponent mask:

```python
# remove the conditional `linked["game_id"]=...astype(str)` / `gk_ghost["game_id"]=...` block.
linked, gk_ghost = align_join_keys(linked, gk_ghost, ["game_id", "period_id", "frame_id"])
merged = linked.merge(gk_ghost, on=["game_id", "period_id", "frame_id"],
                      how="left", suffixes=("_action", "_gk"))
# before: defending = merged[merged["team_id_action"] != merged["team_id_gk"]]
defending = merged[ids_differ(merged["team_id_action"], merged["team_id_gk"])]
```

- [ ] **Step 2: Add `F.add_ghost_gk` to `AGGREGATORS`** in `conftest_id_dtype.py`, then run:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py -k ghost -v ; echo "EXIT: $?"`
Expected: ghost-GK permutations PASS (including `string frames × numeric actions`, which previously raised `ValueError` at the merge — proving M1).

- [ ] **Step 3: Regression** — existing ghost-GK tests:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk.py -v ; echo "EXIT: $?"`
Expected: PASS.

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/features.py silly_kicks/tracking/conftest_id_dtype.py tests/tracking/conftest_id_dtype.py
```

---

## Task 8: Fix defensive-line + off-ball merges + all scalar `same_id` / array `ids_match` seams

**Files:**
- Modify: `_kernels.py:850-861`, `_off_ball_runs.py:283-356`, `_defensive_line.py:62,69,206`, `_gk_influence.py:307,337,350`, `_line_breaking.py:241`, `_player_influence.py:120`, `_ghost_gk.py:759`

- [ ] **Step 1: Apply the canonical transforms** (see "Canonical transform patterns" above) at each site. For each file, add the needed `from ._id_compat import ...` names. Examples (apply the same shape at every listed line):

`_kernels.py` (A3 — replace the `:844-850` `game_id` astype(str) hand-patch; align ALL three
merge keys including the differently-named `frame_id_int`↔`frame_id` pair, then the mask). The
existing `frame_id_int = frame_id.astype("int64")` line stays; `align_join_keys` makes the right
side match it regardless of caller dtype:
```python
# keep: linked["frame_id_int"] = linked["frame_id"].astype("int64")
# remove the `if linked["game_id"].dtype != dl["game_id"].dtype: ...astype(str)...` block.
linked, dl = align_join_keys(
    linked, dl, ["game_id", "period_id", ("frame_id_int", "frame_id")]
)
merged = linked.merge(dl, left_on=["game_id", "period_id", "frame_id_int"],
                      right_on=["game_id", "period_id", "frame_id"], how="left",
                      suffixes=("_action", "_dl"))
opposing = merged[ids_differ(merged["team_id_dl"], merged["team_id_action"])]
```

`_off_ball_runs.py` — three edits:
1. **The `:283-288` merge** has the same A3 `frame_id_int`↔`frame_id` pattern as `_kernels.py`:
   align with `align_join_keys(linked, dl, ["game_id", "period_id", ("frame_id_int", "frame_id")])`
   before it; then the `:291` mask → `ids_differ(merged["team_id_dl"], merged["team_id_action"])`.
2. **The `:308-315` isinstance/astype(str) groupby-key block** — replace with canonicalizing
   `game_id` on both `opposing` and the `frame_groups` keys via `canonical_id_series` (or build
   `frame_groups` from a `non_ball_non_gk` whose `game_id` is pre-aligned to `opposing`), so the
   `(game_id, period_id, frame_id, team_id)` lookup key matches regardless of caller dtype.
3. **The scalar `:331`/`:353`** `if action_team == home_team_id:` → `if same_id(action_team, home_team_id):`.

`_defensive_line.py`:
```python
# :62   & (frames["team_id"] == team_id)            -> & ids_match(frames["team_id"], team_id)
# :69   defends_x0 = team_id == home_team_id          -> defends_x0 = same_id(team_id, home_team_id)
# :206  defends_x0 = team_id == home_team_id          -> defends_x0 = same_id(team_id, home_team_id)
```

`_gk_influence.py`:
```python
# :307  if defending_team_id == home_team_id:        -> if same_id(defending_team_id, home_team_id):
# :337  team_mask_arr = surface.player_team_ids == defending_team_id
#                                                     -> ids_match(surface.player_team_ids, defending_team_id).to_numpy()
# :350  if attacking_team_id != home_team_id:        -> if not same_id(attacking_team_id, home_team_id):
```

`_line_breaking.py:241`, `_player_influence.py:120`, `_ghost_gk.py:759` — the scalar
`if a == home_team_id:` / `!=` form → `same_id` / `not same_id`. Check `_player_influence` for a
surface `player_team_ids ==` array seam (same as `_gk_influence:337`) and apply `ids_match`.

- [ ] **Step 2: Grow the gate** — add `F.add_defensive_line`, `F.add_off_ball_context`,
`F.add_gk_influence`, `F.add_line_break`, `F.add_player_influence`, `F.add_ghost_gk` (already),
and any other registered linked aggregator to `AGGREGATORS`. Run the FULL gate:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py -v ; echo "EXIT: $?"`
Expected: all permutations PASS for every wired aggregator.

- [ ] **Step 3: Regression** — the affected feature suites:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/ -k "defensive_line or off_ball or gk_influence or line_break or player_influence" -v ; echo "EXIT: $?"`
Expected: PASS.

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/_kernels.py silly_kicks/tracking/_off_ball_runs.py silly_kicks/tracking/_defensive_line.py silly_kicks/tracking/_gk_influence.py silly_kicks/tracking/_line_breaking.py silly_kicks/tracking/_player_influence.py silly_kicks/tracking/_ghost_gk.py tests/tracking/conftest_id_dtype.py
```

---

## Task 9: Fix possession / `_ball_carrier` + `direction.py` seams; close the meta-assertion

**Files:**
- Modify: `silly_kicks/tracking/_ball_carrier.py`, `silly_kicks/tracking/direction.py` (and `derive_team_in_possession`)

- [ ] **Step 1: Locate + apply.** Grep the two files for id comparisons:

Run: `.venv/Scripts/python.exe -m pytest -q ; echo "scan:"` then use Grep for `team_id|player_id|home_team_id|in_possession|==|!=` in `_ball_carrier.py` / `direction.py`. Apply `ids_match` (Series/array scalar), `same_id` (scalar), `ids_equal`/`ids_differ` (column-vs-column) per the canonical patterns. `derive_team_in_possession` resolves the carrier's team from frames → coerce the carrier team against any compared team id.

- [ ] **Step 2: Close the meta-assertion** — add every remaining registered linked `add_*` to `AGGREGATORS` (or to the documented `NON_LINKED_AGGREGATORS` allowlist with a reason). Run:

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py::test_enumerated_surface_equals_registered -v ; echo "EXIT: $?"`
Expected: PASS (enumerated == registered).

- [ ] **Step 3: Full gate green**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_dtype_invariance.py -v ; echo "EXIT: $?"`
Expected: PASS (all aggregators × all permutations).

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)**

```bash
git add silly_kicks/tracking/_ball_carrier.py silly_kicks/tracking/direction.py tests/tracking/conftest_id_dtype.py
```

---

## Task 10: AST lint backstop

**Files:**
- Create: `tests/tracking/test_id_compat_lint.py`

- [ ] **Step 1: Write the lint.** Scan `silly_kicks/tracking/*.py` for raw `==`/`!=` `Compare`
nodes where an operand Name/Subscript references a known id (`home_team_id`, any `*team_id*` or
`*player_id*` column subscript) and the comparison is not already wrapped in an `_id_compat`
helper. Allowlist legitimate non-id comparisons (e.g. `== 0`, `.notna()`, sentinel string
literals like `== "native"`, period/frame_id where dtype is contract-guaranteed int64).

```python
# tests/tracking/test_id_compat_lint.py
import ast, pathlib, pytest

TRACKING = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks" / "tracking"
ID_HINTS = ("team_id", "player_id", "home_team_id")
ALLOW = {"_id_compat.py"}  # the helpers themselves

def _id_operand(node):
    if isinstance(node, ast.Name) and any(h in node.id for h in ID_HINTS):
        return True
    if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
        return isinstance(node.slice.value, str) and any(h in node.slice.value for h in ID_HINTS)
    return False

def _raw_id_compares(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Compare) and isinstance(n.ops[0], (ast.Eq, ast.NotEq)):
            operands = [n.left, *n.comparators]
            id_side = any(_id_operand(o) for o in operands)
            literal_side = any(isinstance(o, ast.Constant) for o in operands)
            if id_side and not literal_side:  # id == id / id == var, not id == "native"/0
                hits.append(n.lineno)
    return hits

@pytest.mark.parametrize("path", sorted(TRACKING.glob("*.py")), ids=lambda p: p.name)
def test_no_raw_id_comparisons(path):
    if path.name in ALLOW:
        pytest.skip("helper module")
    hits = _raw_id_compares(path)
    assert not hits, f"{path.name}: raw id comparisons at lines {hits}; use _id_compat helpers"
```

- [ ] **Step 2: Run + reconcile.** Run the lint; for each remaining hit, either route it through a
helper or add a precise allowlist entry (file+line) with a one-line justification comment.

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat_lint.py -v ; echo "EXIT: $?"`
Expected: PASS after the Tasks 6–9 rewrites (any residual is allowlisted with reason).

- [ ] **Step 3: Checkpoint (stage only — do NOT commit)**

```bash
git add tests/tracking/test_id_compat_lint.py
```

---

## Task 11: Structural perf guard + informational benchmark (B1/N2)

**Files:**
- Create: `tests/tracking/test_id_compat_perf.py`

- [ ] **Step 1: Write the structural guard** (call-count, NOT wall-clock).

```python
# tests/tracking/test_id_compat_perf.py
import numpy as np, pandas as pd
from unittest import mock
from silly_kicks.tracking import _id_compat as idc
from silly_kicks.tracking.utils import _resolve_action_frame_context
from tests.tracking.conftest_id_dtype import make_actions, make_frames

def test_resolve_coerces_each_hot_column_at_most_once():
    # cross-dtype (string frames x numeric actions): the always-slow lakehouse path
    actions = make_actions()
    frames = make_frames()
    frames["team_id"] = frames["team_id"].astype("Int64").astype("string").astype("object")
    frames["player_id"] = frames["player_id"].astype("Int64").astype("string").astype("object")
    actions = actions.assign(defending_gk_player_id=pd.Series([1, 2], dtype="int64"))

    real = idc.canonical_id_series
    seen = {}
    def spy(s):
        # count coercions per (id, dtype) of the column object
        seen[id(s)] = seen.get(id(s), 0) + 1
        return real(s)
    with mock.patch.object(idc, "canonical_id_series", side_effect=spy):
        # patch the name as imported into utils too
        import silly_kicks.tracking.utils as U
        with mock.patch.object(U, "canonical_id_series", side_effect=spy, create=True):
            _resolve_action_frame_context(actions, frames)
    # no single column object canonicalized more than once (A1 de-dup held)
    assert all(v <= 1 for v in seen.values()), seen
```

> If the de-dup is implemented by coercing into locals (recommended), assert instead that the
> number of `canonical_id_series` calls within `_resolve_action_frame_context` is ≤ the number of
> distinct id columns (2 frame + 2 action + gk). Adjust the assertion to the chosen impl; the
> invariant is "each column coerced once, not per-mask."

- [ ] **Step 2: Informational benchmark (runs in CI, non-gating).**

```python
def test_resolve_cross_dtype_benchmark(benchmark):
    actions = make_actions()
    frames = pd.concat([make_frames()] * 20000, ignore_index=True)  # ~ scale up rows
    frames["team_id"] = frames["team_id"].astype("Int64").astype("string").astype("object")
    frames["player_id"] = frames["player_id"].astype("Int64").astype("string").astype("object")
    benchmark(lambda: _resolve_action_frame_context(actions.copy(), frames.copy()))


def test_object_object_no_canonicalize_spy(monkeypatch):
    # A1: object x object (genuine-string providers) must take the raw fast path —
    # this is the regression the behavioral gate (correctness-only) cannot see.
    calls = {"n": 0}
    real = idc.canonical_id_series
    monkeypatch.setattr(idc, "canonical_id_series",
                        lambda s: (calls.__setitem__("n", calls["n"] + 1), real(s))[1])
    a = pd.Series(["DFL-A", "DFL-B"] * 50000, dtype="object")
    b = pd.Series(["DFL-A", "DFL-Z"] * 50000, dtype="object")
    idc.ids_equal(a, b)
    idc.ids_differ(a, b)
    assert calls["n"] == 0


def test_resolve_object_object_benchmark(benchmark):
    # genuine-string provider path (sportec/kloppy): both sides object.
    actions = make_actions()
    actions["team_id"] = actions["team_id"].astype(str)
    actions["player_id"] = actions["player_id"].astype(str)
    frames = pd.concat([make_frames()] * 20000, ignore_index=True)
    frames["team_id"] = frames["team_id"].astype("Int64").astype("string").astype("object")
    frames["player_id"] = frames["player_id"].astype("Int64").astype("string").astype("object")
    benchmark(lambda: _resolve_action_frame_context(actions.copy(), frames.copy()))
```

(`pytest-benchmark` is already a test dep; if not, gate this test behind an importorskip and note it in CHANGELOG. It is informational — failure of the benchmark assertion is not a gate; the timing is reported.)

- [ ] **Step 3: Run**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_id_compat_perf.py -v ; echo "EXIT: $?"`
Expected: PASS (structural guard); benchmark reports a time.

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)**

```bash
git add tests/tracking/test_id_compat_perf.py
```

---

## Task 12: Atomic mirror verification

**Files:**
- Inspect: `silly_kicks/atomic/tracking/features.py` (and any atomic id-comparison seams)

- [ ] **Step 1: Verify.** The atomic tracking features compose the same `tracking` aggregators
(ADR-005 mirror). Confirm atomic does not carry its OWN raw id comparisons:

Run: Grep `silly_kicks/atomic/tracking/` for `team_id|player_id|home_team_id` with `==`/`!=`.
If any exist, apply the same canonical transforms (Task 8 patterns) and add the atomic aggregators
to a parallel section in the gate (or assert atomic re-exports the fixed tracking helpers).
If atomic purely re-exports/composes `tracking`, record "no atomic-specific seams" and move on.

- [ ] **Step 2: Run the atomic tracking suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -k "atomic and tracking" -v ; echo "EXIT: $?"`
Expected: PASS.

- [ ] **Step 3: Checkpoint (stage only — do NOT commit)** — `git add` any atomic file touched.

---

## Task 13: Full suite + lint/type parity (shift-left before commit)

- [ ] **Step 1: Full non-e2e suite**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q ; echo "EXIT: $?"`
Expected: PASS (no regressions). Read the summary line, not a piped tail.

- [ ] **Step 2: Dependency-light import guard still holds** (bare `import silly_kicks` must not pull xgboost/numba — `_id_compat` is numpy/pandas only):

Run: `.venv/Scripts/python.exe -c "import silly_kicks; print('ok')" ; echo "EXIT: $?"`
Expected: `ok`.

- [ ] **Step 3: Lint + format + types (WHOLE package)**

```bash
.venv/Scripts/ruff.exe check silly_kicks/ tests/ ; echo "EXIT: $?"
.venv/Scripts/ruff.exe format --check silly_kicks/ tests/ ; echo "EXIT: $?"
.venv/Scripts/pyright.exe silly_kicks/ ; echo "EXIT: $?"
```
Expected: all clean. Fix `I001` import-sort with `ruff check --fix`; resolve any `N806`/`E402`/pyright `Scalar`→numeric with the codebase `# type: ignore[arg-type]` idiom.

- [ ] **Step 4: Checkpoint (stage only — do NOT commit)** — `git add` any lint fixups.

---

## Task 14: Release artifacts + SINGLE final commit (APPROVAL-GATED)

**Files:** `docs/superpowers/adrs/ADR-019-tracking-id-dtype-contract.md` (new), `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md`, `uv.lock`

- [ ] **Step 1: Reconcile version + ADR number against `origin/main`.** `git fetch origin`; confirm the next free minor after the tagged version and the next free ADR number (expected **4.15.0** / **ADR-019**, but reconcile — another session may have shipped). Use the version-bump checklist (5 sites: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `uv.lock` via `uv lock`).

- [ ] **Step 2: Write ADR-019** from the spec's Decision + Review-disposition sections (two layers: comparison + join-key; coerce-at-seam + opt-in validator; rejected alternatives; Hyrum note: string-id callers' features change from silently-wrong to correct).

- [ ] **Step 3: CHANGELOG entry** under the new version — new public `validate_id_dtypes` + `IdDtypeDiagnosis`; the seam fix; the lakehouse handshake (it may drop its string-coercion workaround); Hyrum flag (feature values change for string-id callers). Mark the `ghost_gk`/DL/off-ball ad-hoc `astype(str)` patches as removed/replaced.

- [ ] **Step 4: TODO.md** — mark the "GS tracking-frames id dtype inconsistency" item done (reframed + fixed via ADR-019).

- [ ] **Step 5: `uv lock`** to refresh `uv.lock` (no new runtime deps; pytest-benchmark only if newly added to `[test]`).

- [ ] **Step 6: C4 check** — no new KDE backend / trained model / `add_*` aggregator → **C4-free** (confirm DSL tokens/aggregator count unchanged; skip regen).

- [ ] **Step 7: Final verification before commit**

Run: `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q ; echo "EXIT: $?"`
Expected: PASS.

- [ ] **Step 8: PRESENT diff + command, HOLD for explicit approval.** Show `git status` + `git diff --stat` + the proposed commit message. **Do NOT create the sentinel or commit without an explicit per-commit "yes" from the owner.** On approval (owner creates the sentinel or says proceed), single commit:

```bash
git commit -F .git/COMMIT_id_dtype.txt   # message authored to a temp file, removed after
```
Commit message (subject): `feat(tracking): dtype-safe id contract at feature seams + validate_id_dtypes -- silly-kicks 4.15.0 (ADR-019)` with body summarizing the two layers + Hyrum note, ending with the `Co-Authored-By` trailer.

- [ ] **Step 9: Push + PR** (not gated beyond chat approval): `git push -u origin feat/tracking-id-dtype-contract`; `gh pr create --body-file .git/PR_id_dtype.md`. Squash-merge with `--admin` per solo-maintainer flow; annotated `v4.15.0` tag triggers publish.

---

## Plan review disposition (cross-session, 2026-06-06)

| Item | Disposition |
|---|---|
| **A1** — object×object never fast-paths → `canonical_id_series` Python-loop regresses genuine-string providers (sportec/kloppy) + the lakehouse object side | **Adopted both fixes.** `_directly_comparable` (same-kind OR both-object) gives object×object a raw fast path (safe under C2); `canonical_id_series` object branch vectorized (`astype("string")`, no Python loop). Spy test (object×object → 0 canonicalize calls) + object×object benchmark (Tasks 2, 11). |
| **A2** — influence aggregators need `xt` positional + `add_action_context` takes no `home_team_id`; uniform `_adapt` is wrong, leaving the `_gk_influence:337` array seam gate-uncovered | **Adopted.** Per-aggregator adapters with verified real signatures; concrete `ExpectedThreat(l=16,w=12)` xT (mirrors `tests/conftest.py::fitted_xt`). gk/player influence wired into the gate (Tasks 7-9). |
| **A3** — `_kernels.py`/`_off_ball_runs.py` merge on `frame_id_int`(left)↔`frame_id`(right); `align_join_keys` matched same-name only → mixed-dtype merge still raises | **Adopted.** `align_join_keys` extended to accept `(left_key, right_key)` pairs; the two merges align `("frame_id_int","frame_id")`. Pair test added (Task 3). |
| **N-a** positional consistency in `ids_equal` | Adopted (`_positional`). |
| **N-b** `align_join_keys` object×object needless coerce | Adopted (folded into `_directly_comparable`). |
| **N-c** `align_join_keys` test only int/int | Adopted (object-noop + pair tests). |
| **N-d** meta-assertion allowlist is the leak point | Adopted: `NON_LINKED_AGGREGATORS` requires a per-entry justification; AST lint is the cross-check. |
| **N-e** gate row-order sensitivity | Adopted: deterministic `action_id` ordering assumption stated in the gate. |

### Plan review round 3 (2026-06-06)

| Item | Disposition |
|---|---|
| **B1** — gate's `feat_cols` excludes only `team_id`/`player_id`; any other id-valued output column (`defending_gk_player_id`, `*_id` provenance, surfaced GK/actor id) false-fails the asymmetric variant (`check_dtype=False` still compares values; `99` ≠ `"99"`) | **Adopted.** Replaced the hardcoded 2-element exclusion with a generic `_is_id_col` predicate (`"team_id" in c or "player_id" in c or c.endswith("_id")`); kept a separate `STRINGIFY_COLS` for the input-cast side. |
| **N1** — `_XT` built at module import breaks collection of the whole gate on xT-API drift | **Adopted.** Lazy `@functools.cache _xt()`; influence adapters call `_xt()`. |
| **N2** — "mirrors fitted_xt exactly" implies a false sync obligation | **Adopted.** Comment now: any valid fitted xT; not required to track `fitted_xt`. |
| **N3** — full 2³ permutations optional | **Declined (YAGNI).** Reviewer agrees the 4 axis-independent + homogeneous permutations are the load-bearing set. |

## Self-review (completed by author)

- **Spec coverage:** §1 primitive → Tasks 1–3; §1 join-key layer (M1) → Task 3 + applied 6/7/8; §2 validator → Task 4; §3 seam application → Tasks 6–9; §3 CI gate (asymmetric, M2; meta-assert B3) → Tasks 5/9/10; testing strategy (both-entry-point B2 → Task 1; np.bool_ C1 → Task 2; N1 → Task 2; perf B1/N2 → Task 11) → covered; atomic mirror → Task 12; packaging/ADR/version → Task 14.
- **Placeholder scan:** site lists are concrete (file:line); the one deliberate open set (any seam the audits missed) is closed by the gate (Task 5) + AST lint (Task 10), which is the design's completeness mechanism, not a placeholder.
- **Type consistency:** helper names (`canonical_id`, `canonical_id_series`, `ids_equal`, `ids_differ`, `ids_match`, `same_id`, `align_join_keys`, `validate_id_dtypes`, `IdDtypeDiagnosis`) used identically across tasks.
