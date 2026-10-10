# Agent-Support Phase 3 (MCP `coords` aspect + howto link-check) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a read-only coordinate-integrity lib seam (`silly_kicks.spadl.diagnose_coordinates`), bind it as `diagnose_provider`'s `coords` aspect, add a `docs/howto`+`docs/context` reference link-check guard, and fix the ADR-032→056 doc mis-cite.

**Architecture:** Pure lib core (pandas in, frozen dataclass out, zero I/O / zero mutation) + adapter-only MCP binding (the server owns no analysis). The link-check is a `test_*_wired` structural guard (runs under `pytest tests/`, no `ci.yml` edit).

**Tech Stack:** Python, pandas, numpy, pytest, FastMCP (`mcp[cli]>=1,<2` optional extra — already installed in `[test]`).

**Spec:** `docs/superpowers/specs/2026-10-09-agent-support-phase3-mcp-coords-and-howto-linkcheck-design.md` (APPROVED; owner-ratified §7 decisions). Executors read the spec alongside this plan.

## Global Constraints

- **Read-only / pure:** `diagnose_coordinates` mutates nothing (compare input pre/post in tests), does no I/O, raises on `actions is None and frames is None`. The MCP server adds no analysis logic beyond the bind + `_json_safe`.
- **Frame coords are float32 STORAGE (ADR-106):** upcast each slice to float64 at the read boundary before stats.
- **Frozen flag tokens (Hyrum):** exactly `coords_scale_suspect`, `actions_out_of_pitch`, `coords_gross_out_of_range`, `coords_all_nan`, `actions_start_nan`.
- **Pitch constants from the canonical config:** `from silly_kicks.spadl import config as spadlconfig` → `spadlconfig.field_length` (105.0), `spadlconfig.field_width` (68.0). Never hard-code 105/68.
- **New public module MUST register** in `tests/test_public_api_examples.py` `_PUBLIC_MODULE_FILES` (+ doctest or `_EXAMPLES_DEBT`). Run the surface gates (`test_public_api_examples`, `c4`, `metric_contracts`) locally before the commit.
- **Docs travel with the ONE approval-gated final commit** — no separate/early doc commit. Feature branch, no worktree. Independent `/review-plan` before coding and `/review-impl` before the commit; author never reviews own; reports to `D:\Development\_reviews\`.

---

## File Structure

- **Create** `silly_kicks/spadl/_coordinate_diagnosis.py` — the seam + dataclasses + scale classifier (single responsibility: coordinate integrity).
- **Modify** `silly_kicks/spadl/__init__.py` — export `diagnose_coordinates`, `CoordinateDiagnosis`, `CoordinateTableDiagnosis`, `CoordinateAxisStats`, `CoordinateDiagnosisParams` in `__all__` + imports.
- **Modify** `silly_kicks/mcp/server.py` — `_diagnose`/`_flags`/docstring gain `coords`.
- **Create** `tests/spadl/test_diagnose_coordinates.py` — the seam's unit tests.
- **Modify** `tests/mcp/test_*.py` (or add `tests/mcp/test_coords_aspect.py`) — the MCP `coords` binding test.
- **Create** `tests/test_howto_links_wired.py` — the link-check guard + its own precondition test.
- **Modify** `tests/test_public_api_examples.py` — register the new spadl module file.
- **Modify** `AGENTS.md:123`, `docs/context/conventions-core.md:58` — ADR-032→056 (byte-neutral). *(Already applied in the working tree during the spec cycle; this task verifies they are present + correct, since they travel with this commit.)*
- **Modify** `docs/howto/mcp.md`, `CHANGELOG.md` — document the new aspect + the release note.

---

### Task 1: `diagnose_coordinates` seam + dataclasses + scale classifier

**Files:**
- Create: `silly_kicks/spadl/_coordinate_diagnosis.py`
- Test: `tests/spadl/test_diagnose_coordinates.py`

**Interfaces:**
- Produces: `diagnose_coordinates(actions: pd.DataFrame | None, frames: pd.DataFrame | None, *, params: CoordinateDiagnosisParams = CoordinateDiagnosisParams()) -> CoordinateDiagnosis`; the four frozen dataclasses; `CoordinateDiagnosisParams.for_provider(provider: str) -> CoordinateDiagnosisParams`.

**Pre-registered scale-classification (DECISION OQ1 — fixed BEFORE coding, not tuned to a corpus):**

```
MIN_N = 20                               # below this many finite rows -> "undetermined"
METERS_ASPECT_LO, METERS_ASPECT_HI = 1.30, 1.80   # SPADL x/y span ratio ≈ 105/68 = 1.544
# per table, over finite coords: xr = p99(x) - p01(x); yr = p99(y) - p01(y)
#   mag    = max(p99(x), p99(y))
#   aspect = xr / yr          (yr <= 1e-9 -> inf)
# classify:
#   n_finite < MIN_N                                              -> "undetermined"
#   mag <= 1.5                                                    -> "normalized_0_1"
#   METERS_ASPECT_LO <= aspect <= METERS_ASPECT_HI and 40 <= mag <= 150 -> "spadl_meters"
#   mag <= 110                                                    -> "scale_0_100"
#   else                                                          -> "suspect"
```

Fixtures table (each exercises one branch; `tests/spadl/test_diagnose_coordinates.py`): spadl `x~U(0,105) y~U(0,68) n=100`→`spadl_meters`; `U(0,1)`→`normalized_0_1`; `U(0,100)` square→`scale_0_100`; `U(0,1050)`→`suspect`; `n=5`→`undetermined`.

- [ ] **Step 1: Write the failing tests** (`tests/spadl/test_diagnose_coordinates.py`)

```python
import numpy as np
import pandas as pd
import pytest
from silly_kicks.spadl import (
    diagnose_coordinates, CoordinateDiagnosis, CoordinateDiagnosisParams,
)

def _actions(x, y, n):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "game_id": 1, "period_id": 1,
        "start_x": rng.uniform(0, x, n), "start_y": rng.uniform(0, y, n),
        "end_x": rng.uniform(0, x, n), "end_y": rng.uniform(0, y, n),
    })

def _frames(x, y, n):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "game_id": 1, "period_id": 1, "is_ball": False,
        "x": rng.uniform(0, x, n).astype("float32"), "y": rng.uniform(0, y, n).astype("float32"),
    })

def test_spadl_meters_clean():
    d = diagnose_coordinates(_actions(105, 68, 100), _frames(105, 68, 100))
    assert d.actions.inferred_scale == "spadl_meters"
    assert d.frames.inferred_scale == "spadl_meters"
    assert d.flags == []

def test_normalized_0_1_flags_scale():
    d = diagnose_coordinates(_actions(1, 1, 100), None)
    assert d.actions.inferred_scale == "normalized_0_1"
    assert "coords_scale_suspect" in d.flags

def test_scale_0_100_flags_scale():
    d = diagnose_coordinates(_actions(100, 100, 100), None)
    assert d.actions.inferred_scale == "scale_0_100"
    assert "coords_scale_suspect" in d.flags

def test_undetermined_below_min_n():
    d = diagnose_coordinates(_actions(105, 68, 5), None)
    assert d.actions.inferred_scale == "undetermined"

def test_actions_out_of_pitch_flags_but_frames_off_pitch_is_info():
    a = _actions(105, 68, 100); a.loc[0, "start_x"] = 130.0   # 25 m off, actions are clipped by contract
    f = _frames(105, 68, 100); f.loc[0, "x"] = np.float32(130.0)  # legit off-pitch for tracking
    d = diagnose_coordinates(a, f)
    assert "actions_out_of_pitch" in d.flags
    assert d.frames.out_of_pitch_fraction > 0      # reported...
    assert "coords_gross_out_of_range" not in d.flags  # ...but NOT a defect flag for frames

def test_gross_out_of_range_both_tables():
    d = diagnose_coordinates(_actions(1050, 680, 100), _frames(1050, 680, 100))
    assert "coords_gross_out_of_range" in d.flags

def test_all_nan_and_actions_start_nan():
    a = _actions(105, 68, 100); a[["start_x", "start_y"]] = np.nan
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" in d.flags
    f = _frames(105, 68, 100); f[["x", "y"]] = np.nan
    d2 = diagnose_coordinates(None, f)
    assert "coords_all_nan" in d2.flags

def test_end_only_nan_does_not_flag_actions_start_nan():   # discriminating: start-NaN vs end-NaN (frozen token)
    a = _actions(105, 68, 100); a[["end_x", "end_y"]] = np.nan   # END only
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" not in d.flags   # start coords intact -> must NOT fire

def test_start_only_nan_does_not_overclaim_coords_all_nan():   # coords_all_nan = TRUE all-NaN, not any-per-row
    a = _actions(105, 68, 100); a[["start_x", "start_y"]] = np.nan   # start NaN, end finite
    d = diagnose_coordinates(a, None)
    assert "actions_start_nan" in d.flags
    assert "coords_all_nan" not in d.flags   # end coords finite -> NOT all-coords-NaN; name must not over-claim

def test_both_none_raises():
    with pytest.raises(ValueError):
        diagnose_coordinates(None, None)

def test_one_none_diagnoses_the_present_table():
    d = diagnose_coordinates(_actions(105, 68, 100), None)
    assert d.actions is not None and d.frames is None

def test_purity_and_determinism():
    a, f = _actions(105, 68, 100), _frames(105, 68, 100)
    a2, f2 = a.copy(), f.copy()
    d1 = diagnose_coordinates(a, f); d2 = diagnose_coordinates(a, f)
    pd.testing.assert_frame_equal(a, a2); pd.testing.assert_frame_equal(f, f2)  # unmutated
    assert d1 == d2  # deterministic

def test_notes_state_non_claims():
    d = diagnose_coordinates(_actions(105, 68, 100), _frames(105, 68, 100))
    joined = " ".join(d.notes).lower()
    assert "orientation" in joined and "off-pitch" in joined
```

- [ ] **Step 2: Run to verify they fail** — `.venv/Scripts/python.exe -m pytest tests/spadl/test_diagnose_coordinates.py -q` → FAIL (ImportError: cannot import `diagnose_coordinates`).

- [ ] **Step 3: Implement the seam** (`silly_kicks/spadl/_coordinate_diagnosis.py`)

```python
"""Read-only coordinate-integrity diagnosis (SPADL frame). Pure: pandas in, frozen dataclass out.

Does NOT measure orientation/direction (see silly_kicks.tracking check_orientation / the MCP tool) and
does NOT flag legitimate off-pitch TRACKING positions (SkillCorner/SB tracking is legitimately off-pitch;
only ACTIONS are clipped to the pitch). A scale/units HEURISTIC + bounds/NaN tripwire, not a proof.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as spadlconfig

_MIN_N = 20
_METERS_ASPECT_LO, _METERS_ASPECT_HI = 1.30, 1.80
_ACTION_COLS = ("start_x", "start_y", "end_x", "end_y")
_FRAME_COLS = ("x", "y")


@dataclass(frozen=True)
class CoordinateDiagnosisParams:
    field_length: float = float(spadlconfig.field_length)
    field_width: float = float(spadlconfig.field_width)
    off_pitch_tol_m: float = 1.0
    gross_range_factor: float = 2.0

    @classmethod
    def for_provider(cls, provider: str) -> "CoordinateDiagnosisParams":  # noqa: ARG003 - neutral v1
        """v1 returns the neutral default for EVERY provider; promotable later with no API break."""
        return cls()


@dataclass(frozen=True)
class CoordinateAxisStats:
    min: float
    p01: float
    p50: float
    p99: float
    max: float


@dataclass(frozen=True)
class CoordinateTableDiagnosis:
    table: str
    n_rows: int
    x: CoordinateAxisStats
    y: CoordinateAxisStats
    inferred_scale: str
    out_of_pitch_fraction: float
    gross_out_of_range_fraction: float
    coord_nan_fraction: float       # fraction of rows with ANY NaN coord (INFO)
    all_coords_nan: bool            # every diagnosed coord cell is NaN (the coords_all_nan predicate)


@dataclass(frozen=True)
class CoordinateDiagnosis:
    actions: CoordinateTableDiagnosis | None
    frames: CoordinateTableDiagnosis | None
    flags: list[str]
    notes: list[str]


def _axis_stats(a: np.ndarray) -> CoordinateAxisStats:
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        nan = float("nan")
        return CoordinateAxisStats(nan, nan, nan, nan, nan)
    p01, p50, p99 = (float(v) for v in np.percentile(finite, [1, 50, 99]))
    return CoordinateAxisStats(float(finite.min()), p01, p50, p99, float(finite.max()))


def _classify(xs: CoordinateAxisStats, ys: CoordinateAxisStats, n_finite: int) -> str:
    if n_finite < _MIN_N or not np.isfinite(xs.p99) or not np.isfinite(ys.p99):
        return "undetermined"
    xr, yr = xs.p99 - xs.p01, ys.p99 - ys.p01
    mag = max(xs.p99, ys.p99)
    aspect = xr / yr if yr > 1e-9 else float("inf")
    if mag <= 1.5:
        return "normalized_0_1"
    if _METERS_ASPECT_LO <= aspect <= _METERS_ASPECT_HI and 40.0 <= mag <= 150.0:
        return "spadl_meters"
    if mag <= 110.0:
        return "scale_0_100"
    return "suspect"


def _table_diag(df: pd.DataFrame, cols: tuple[str, ...], table: str, p: CoordinateDiagnosisParams) -> CoordinateTableDiagnosis:
    present = [c for c in cols if c in df.columns]
    xcols = [c for c in present if c.endswith("x")]
    ycols = [c for c in present if c.endswith("y")]
    xv = df[xcols].to_numpy(dtype="float64", copy=True).ravel() if xcols else np.array([])
    yv = df[ycols].to_numpy(dtype="float64", copy=True).ravel() if ycols else np.array([])
    xs, ys = _axis_stats(xv), _axis_stats(yv)
    n_finite = int(min(np.isfinite(xv).sum(), np.isfinite(yv).sum())) if xcols and ycols else 0
    # per-row any-NaN over the present coord columns
    coord = df[present].to_numpy(dtype="float64", copy=True)
    nan_rows = np.isnan(coord).any(axis=1)
    nan_frac = float(nan_rows.mean()) if len(df) else 0.0
    all_nan = bool(coord.size and np.isnan(coord).all())   # true all-coords-NaN (NOT any-per-row)
    # per-row out-of-pitch (beyond tol) and gross out-of-range, over x-cols and y-cols jointly
    def _oob(frac_only_gross: bool) -> float:
        if not (xcols and ycols) or not len(df):
            return 0.0
        xo = df[xcols].to_numpy(dtype="float64", copy=True)
        yo = df[ycols].to_numpy(dtype="float64", copy=True)
        if frac_only_gross:
            bad = (np.abs(xo) > p.gross_range_factor * p.field_length).any(axis=1) | (
                np.abs(yo) > p.gross_range_factor * p.field_width).any(axis=1)
        else:
            bad = ((xo < -p.off_pitch_tol_m) | (xo > p.field_length + p.off_pitch_tol_m)).any(axis=1) | (
                (yo < -p.off_pitch_tol_m) | (yo > p.field_width + p.off_pitch_tol_m)).any(axis=1)
        return float(np.nanmean(bad.astype("float64")))
    return CoordinateTableDiagnosis(
        table=table, n_rows=int(len(df)), x=xs, y=ys,
        inferred_scale=_classify(xs, ys, n_finite),
        out_of_pitch_fraction=_oob(False), gross_out_of_range_fraction=_oob(True),
        coord_nan_fraction=nan_frac, all_coords_nan=all_nan,
    )


def diagnose_coordinates(actions, frames, *, params: CoordinateDiagnosisParams = CoordinateDiagnosisParams()) -> CoordinateDiagnosis:
    if actions is None and frames is None:
        raise ValueError("diagnose_coordinates needs at least one of actions/frames")
    a = _table_diag(actions, _ACTION_COLS, "actions", params) if actions is not None else None
    f = _table_diag(frames, _FRAME_COLS, "frames", params) if frames is not None else None
    flags: list[str] = []
    _NEUTRAL_SCALE = ("spadl_meters", "undetermined")  # everything else = a scale/units bug (§3.3 tripwire)
    if (a and a.inferred_scale not in _NEUTRAL_SCALE) or (f and f.inferred_scale not in _NEUTRAL_SCALE):
        flags.append("coords_scale_suspect")
    if a and a.out_of_pitch_fraction > 0.0:
        flags.append("actions_out_of_pitch")
    if (a and a.gross_out_of_range_fraction > 0.0) or (f and f.gross_out_of_range_fraction > 0.0):
        flags.append("coords_gross_out_of_range")
    if (a and a.all_coords_nan) or (f and f.all_coords_nan):   # true all-NaN, not any-per-row
        flags.append("coords_all_nan")
    if actions is not None and len(actions):
        _sx = actions[["start_x", "start_y"]].to_numpy(dtype="float64", copy=True)
        if float(np.isnan(_sx).any(axis=1).mean()) > 0.0:  # START coords ONLY (spec §3.3; frozen token) — NOT end
            flags.append("actions_start_nan")
    notes = [
        "Does NOT measure orientation/direction of play (see check_orientation).",
        "Frames out_of_pitch_fraction is INFO: tracking is legitimately off-pitch; only actions-off-pitch flags.",
        "Scale/units is a heuristic tripwire (aspect-ratio + magnitude), not a proof.",
    ]
    return CoordinateDiagnosis(actions=a, frames=f, flags=flags, notes=notes)
```

- [ ] **Step 4: Run to verify pass** — `.venv/Scripts/python.exe -m pytest tests/spadl/test_diagnose_coordinates.py -q` → PASS. Predicates are pinned to spec §3.3/§3.4, each with a DISCRIMINATING test: `actions_start_nan` = START coords only (end-only-NaN test must NOT fire it), `coords_all_nan` = true all-coords-NaN (start-only-NaN test must NOT fire it), `coords_scale_suspect` = any non-SPADL scale (normalized_0_1 + scale_0_100 tests DO fire it). No undecided predicate.

- [ ] **Step 5: ruff** — `py -3.14 -m ruff format silly_kicks/spadl/_coordinate_diagnosis.py tests/spadl/test_diagnose_coordinates.py` + `ruff check` both. No commit yet (docs/code travel with the final commit).

---

### Task 2: Export + public-API registration

**Files:**
- Modify: `silly_kicks/spadl/__init__.py` (add to `__all__` + imports)
- Modify: `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`)

**Interfaces:** Consumes Task 1's public names.

- [ ] **Step 1:** add to `silly_kicks/spadl/__init__.py` `__all__`: `"diagnose_coordinates", "CoordinateDiagnosis", "CoordinateTableDiagnosis", "CoordinateAxisStats", "CoordinateDiagnosisParams"`; and `from ._coordinate_diagnosis import (diagnose_coordinates, CoordinateDiagnosis, CoordinateTableDiagnosis, CoordinateAxisStats, CoordinateDiagnosisParams)`.
- [ ] **Step 2:** add `"silly_kicks/spadl/_coordinate_diagnosis.py"` to `_PUBLIC_MODULE_FILES`. Add a runnable doctest to `diagnose_coordinates` (a 3-row synthetic frame → `.flags`) OR, if a doctest is awkward, add an `_EXAMPLES_DEBT` entry `"silly_kicks/spadl/_coordinate_diagnosis.py::diagnose_coordinates"` with a note. *(Prefer the doctest.)*
- [ ] **Step 3: Run the surface gate** — `.venv/Scripts/python.exe -m pytest tests/test_public_api_examples.py -q` → PASS (`test_derived_surface_is_fully_accounted_for` green). Also `-m pytest tests/ -k "c4 or metric_contracts" ... ` to confirm no container/metric registration regressed.

---

### Task 3: MCP `coords` aspect (adapter only)

**Files:**
- Modify: `silly_kicks/mcp/server.py:162` (`_diagnose`), `:177` (`_flags`), `:189` (docstring)
- Test: `tests/mcp/test_coords_aspect.py`

**Interfaces:** Consumes `silly_kicks.spadl.diagnose_coordinates`.

- [ ] **Step 1: Write the failing MCP test** (`tests/mcp/test_coords_aspect.py`) — follow the existing `tests/mcp/` fixtures (a synthetic `LoadedMatch` or the established mock); assert `diagnose_provider(provider, match_ref, aspect="coords")` returns a dict with `aspect == "coords"`, JSON-safe `findings` (nested `CoordinateDiagnosis`), and `flags` equal to the lib's `diag.flags`. Assert an unknown aspect still raises `ValueError`.
- [ ] **Step 2: Run → FAIL** (coords raises the unknown-aspect ValueError).
- [ ] **Step 3: Implement** — in `_diagnose`, before the `raise`: `if aspect == "coords": from silly_kicks.spadl import diagnose_coordinates; return diagnose_coordinates(getattr(loaded, "actions", None), getattr(loaded, "frames", None))`. In `_flags`: `if aspect == "coords": return list(getattr(diag, "flags", []))`. Update the `diagnose_provider` docstring aspect set to `{keeper, convention, id_dtype, coords}` and the `_diagnose` ValueError message `known: keeper|convention|id_dtype|coords`.
- [ ] **Step 4: Run → PASS**; also `tests/mcp/test_registration.py` stays green (tool set unchanged — still 3 tools; `coords` is an aspect, not a new tool).
- [ ] **Step 5: ruff.**

---

### Task 4: `docs/howto` + `docs/context` link-check guard

**Files:**
- Create: `tests/test_howto_links_wired.py`

- [ ] **Step 1: Write the guard + its OWN precondition test** (TDD: the precondition fixture first). The guard resolves, across `docs/howto/*.md` + `docs/context/*.md`:

```python
import re, pathlib
_REPO = pathlib.Path(__file__).resolve().parent.parent
_DIRS = [_REPO / "docs/howto", _REPO / "docs/context"]
_MD_LINK = re.compile(r"\]\((?!https?://)([^)\s#]+)(?:#([^)\s]+))?\)")
_ADR = re.compile(r"\bADR-(\d{3})\b")
_CODE = re.compile(r"\b((?:silly_kicks|tests|scripts)/[\w/]+\.py)(?::(\d+))?")
_DOCP = re.compile(r"\b(docs/[\w./-]+\.md)\b")   # bare doc-path prose mentions (the dominant ref kind here)

def _slug(h):  # GitHub-style heading slug
    return re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")

def _headings(md: str):
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*)$", md, re.M)}

def _collect():
    md_files = [p for d in _DIRS for p in d.glob("*.md")]
    hard_errs, soft_warns, counts = [], [], {"md": 0, "adr": 0, "code": 0, "docp": 0}
    for p in md_files:
        txt = p.read_text(encoding="utf-8")
        for m in _MD_LINK.finditer(txt):
            counts["md"] += 1
            target = (p.parent / m.group(1)).resolve()
            if not target.exists():
                target = (_REPO / m.group(1)).resolve()
            if not target.exists():
                hard_errs.append((str(p), m.group(0), "md-link missing")); continue
            if m.group(2) and target.suffix == ".md":  # anchor
                if _slug(m.group(2)) not in _headings(target.read_text(encoding="utf-8")):
                    hard_errs.append((str(p), m.group(0), "md anchor missing"))
        for m in _ADR.finditer(txt):
            counts["adr"] += 1
            if not list((_REPO / "docs/superpowers/adrs").glob(f"ADR-{m.group(1)}-*.md")):
                hard_errs.append((str(p), m.group(0), "ADR file missing"))
        for m in _CODE.finditer(txt):
            counts["code"] += 1
            f = _REPO / m.group(1)
            if not f.exists():
                hard_errs.append((str(p), m.group(0), "code path missing"))
            elif m.group(2) and len(f.read_text(encoding="utf-8").splitlines()) < int(m.group(2)):
                soft_warns.append((str(p), m.group(0), "line beyond EOF (SOFT)"))
        for m in _DOCP.finditer(txt):
            counts["docp"] += 1
            if not (_REPO / m.group(1)).exists():
                hard_errs.append((str(p), m.group(0), "doc path missing"))
    return hard_errs, soft_warns, counts

def test_howto_context_references_resolve():
    hard, soft, counts = _collect()
    assert not hard, "unresolved references:\n" + "\n".join(map(str, hard))
    # SOFT line-suffix warnings are surfaced, non-failing:
    if soft:
        import warnings; warnings.warn("line-suffix refs beyond EOF: " + "; ".join(str(s) for s in soft))

def test_linkcheck_non_vacuity_floor():
    _, _, counts = _collect()
    assert counts["adr"] >= 200 and counts["code"] >= 60 and counts["docp"] >= 20, counts  # measured live: ADR 465 / code 120 / docp 39 (md-links 0 -> no md floor)

def test_linkcheck_precondition_catches_a_broken_ref(tmp_path, monkeypatch):
    # a fixture md with one broken md-link, one bad ADR, one missing code path -> all three reported
    bad = tmp_path / "howto"; bad.mkdir()
    (bad / "x.md").write_text("[a](does_not_exist.md) ADR-999 silly_kicks/nope_xyz.py docs/nope_abc.md", encoding="utf-8")
    monkeypatch.setattr("tests.test_howto_links_wired._DIRS", [bad])
    hard, _, _ = _collect()
    assert len(hard) == 4   # md-link + ADR + code-path + doc-path all reported
```

- [ ] **Step 2: Run → confirm** the non-vacuity floor + precondition pass on the real tree; FIX any REAL broken ref the guard surfaces in `docs/howto`/`docs/context` (that is the guard doing its job — resolve each, do not weaken the guard). Expect it to re-surface nothing after the ADR-032→056 fix lands (Task 5).
- [ ] **Step 3: ruff.**

*(Measured live @50fab4c: ADR-refs 465, code-paths 120, doc-paths 39, md-links 0. Floors pinned ~50% below: adr≥200, code≥60, docp≥20; NO md-link floor — none present today, the resolver is kept for future md links. Re-measure + keep the margin if the docs grow/shrink materially. The dominant ref kind is the bare doc-path prose mention, NOT markdown `](...)` links.)*

---

### Task 5: doc-rot fix + docs + the single final commit

**Files:** `AGENTS.md:123`, `docs/context/conventions-core.md:58` (ADR-032→056, byte-neutral — verify present), `docs/howto/mcp.md` (the `coords` aspect), `CHANGELOG.md`.

- [ ] **Step 1:** verify `AGENTS.md:123` + `conventions-core.md:58` read `ADR-056` (applied during the spec cycle; they travel with THIS commit). Confirm `test_agents_md_budget.py` green (byte-neutral change).
- [ ] **Step 2:** `docs/howto/mcp.md` — add `coords` to the `diagnose_provider` aspect list + one line on what it diagnoses (scale/bounds/NaN tripwire; NOT orientation). CHANGELOG entry (CI-infra/tooling; version bump only if the surface gates demand — a new additive public fn is minor; confirm the repo's bump convention with the owner at commit-prep).
- [ ] **Step 3: Full local gate** — `.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q` green; the surface gates (`test_public_api_examples`, `c4`, `test_agents_md_budget`) green; ruff check + format --check at CI scope (`silly_kicks/ tests/ scripts/`).
- [ ] **Step 4: Independent `/review-impl`** (separate session; author does not review own) → address findings.
- [ ] **Step 5: ONE approval-gated commit** on the feature branch (code + tests + spec + plan + docs together — docs are NOT a separate commit). Present the diff + file list; wait for explicit owner yes; then commit + push + PR. Merge is a separate owner go after CI green.

---

## Self-Review

**Spec coverage:** §3 coords seam → T1; §3.2 dataclasses/`for_provider` → T1; §3.3 checks (scale/out-of-pitch/gross/NaN) → T1; §3.4 frozen flags → T1 (predicate); §3.5 MCP binding → T3; §4 link-check (md+anchor+ADR+code HARD, `:NNN` SOFT, non-vacuity, ceiling) → T4; §5 tests (both-sides/purity/determinism/MCP/precondition) → T1,T3,T4; §5 surface gates → T2,T5; §6 workflow → T5; §7 decisions (OQ1 bands pre-registered → T1; OQ2 seam → T1; OQ3 resolution → T4; OQ4 path → T1; OQ5 tokens → T1; OQ6 doc-rot → T5). No gap.

**Placeholder scan:** none. Both plan-review rounds' SHOULD-FIXes are resolved IN the plan: `actions_start_nan` is START-only (§3.3) + discriminating end-only test; `coords_scale_suspect` = `not in {spadl_meters, undetermined}`; `coords_all_nan` = true all-coords-NaN (field `all_coords_nan`) + a start-only-NaN non-overclaim test; non-vacuity floors pinned from MEASURED live counts (ADR 465 / code 120 / docp 39, md-links 0). No TBD/TODO.

**Type consistency:** `diagnose_coordinates` signature, the four dataclasses + field names, and the flag tokens are identical across T1/T2/T3 and match the spec §3. `for_provider` signature consistent.
