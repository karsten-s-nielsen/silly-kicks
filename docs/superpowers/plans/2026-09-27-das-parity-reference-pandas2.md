# DAS corpus-parity: pandas-2 reference subprocess — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` (inline). Steps use
> checkbox (`- [ ]`) syntax for tracking. **This project bans subagents and micro-commits** (see Global
> Constraints) — do NOT dispatch subagents and do NOT commit per task.

**Goal:** Make the native-DAS §7.2 corpus-parity leg compare the native engine against a FAITHFUL
`accessible-space==2.0.15` reference by running the library in a pinned pandas-2 subprocess (it is
silently broken under the driver's pandas 3), and remove the divergence-exclusion machinery that was
chasing that environment artifact.

**Architecture:** A new sk-free module `scripts/_das_reference_leg.py` holds the reference recipe
(`_REFERENCE_COMMON`, `_reference_lib_frames`), a collision-free unique-frame key, fail-loud env guards,
and a `__main__` that reads a frames parquet and writes team/player parquet. The driver marshals each
match's scored frames to a temp parquet outside the repo tree, invokes a documented pandas-2 interpreter
(`--reference-python` / `SK_DAS_REFERENCE_PYTHON`) on that module, and reads the arrays back. The reduce
grades and finite-mask-checks over `reason == Reason.OK` rows; `reason_counts` is the per-class §7.2
accounting; the D-OFF / D-KEY exclusion machinery is deleted.

**Tech Stack:** Python 3.12; production process pandas 3 / numpy 2; reference subprocess pandas 2.x +
`accessible-space==2.0.15` (venv at `~/das-parity/py2ref`); pytest; parquet (pyarrow) for marshalling.

**Spec:** `docs/superpowers/specs/2026-09-27-das-parity-reference-pandas2-design.md` (read it alongside
this plan — the plan argues from the spec).

## Global Constraints

- **Branch:** `feat/combined-provenance-dgx` (PR #261), shared with the F1b/T10 session. Work directly on
  it (no worktrees). Coordinate push/merge with the owner.
- **Commit discipline:** ONE commit for the whole change, only after the full suite + lint + pyright are
  green AND the owner has explicitly approved that specific commit. NO micro-commits, NO per-task commits,
  NO commit/push/tag without explicit per-commit owner approval. Tasks below end at "tests green", never
  at a commit.
- **No subagents.** Execute inline. Any `/review-*` runs in a separate owner-launched session, never from
  here.
- **No version bump, no PyPI publish, no CHANGELOG entry this cycle.** A CHANGELOG line lands at release.
- **Reference-env pins (exact):** `accessible-space==2.0.15`, `pandas<3`. The accessible-space version is
  read via `importlib.metadata.version("accessible-space")` — `accessible_space.__version__` does NOT
  exist.
- **Temp files:** written to a system tempdir (`tempfile.mkdtemp`) OUTSIDE the repo tree — never inside
  it (the driver's `require_clean_tree` guard, ADR-037, would trip).
- **Lint scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`; `pyright`
  bare. Never lint `.`.
- **The `reference_leg=` injection seam stays** — the offline reduce-path test injects a golden stub and
  must never spawn the subprocess or require `accessible-space`.

---

## File structure

- `scripts/_das_reference_leg.py` (**create**) — sk-free reference module. Top-level imports: `pandas`,
  `numpy`, `os`, `sys`, `importlib.metadata`, `warnings` ONLY. `accessible_space` imported LAZILY inside
  the leg function. Responsibility: run accessible-space 2.0.15 faithfully under pandas 2 and marshal
  results.
- `scripts/validate_das_native_parity.py` (**modify**) — reference leg becomes a subprocess call; reduce
  re-anchors to `reason == OK`; divergence machinery deleted; provenance stamped.
- `tests/scripts/test_das_native_parity_driver.py` (**modify**) — 6 existing tests EDIT/DELETE/RE-ANCHOR;
  new reduce/subprocess tests.
- `tests/scripts/test_das_reference_leg.py` (**create**) — unit tests for the sk-free module (offline, no
  accessible-space).
- `pyproject.toml` (**modify**) — `das-reference` extra declares `pandas<3` alongside
  `accessible-space==2.0.15`.
- `docs/superpowers/adrs/ADR-107-*.md`, `ADR-108-*.md` (**modify**) — Consequences amendment.
- `docs/context/tracking-features.md` (**modify**) — a DAS-parity note.

---

### Task 1: sk-free reference module `scripts/_das_reference_leg.py`

**Files:**
- Create: `scripts/_das_reference_leg.py`
- Create: `tests/scripts/test_das_reference_leg.py`

**Interfaces:**
- Consumes: nothing from other tasks. (Reads the frozen recipe currently inline in the driver.)
- Produces (imported by Task 2 and the tests):
  - `_X_OFFSET: float = 52.5`, `_Y_OFFSET: float = 34.0`
  - `_REFERENCE_COMMON: dict[str, Any]` — verbatim the current driver value.
  - `_reference_lib_frames(frames: pd.DataFrame) -> pd.DataFrame` — verbatim from the driver.
  - `_add_unique_frame_col(lib: pd.DataFrame, col: str = "_uframe") -> pd.DataFrame` — adds an int column
    that is the dense group code of `(game_id, period_id, frame_id)` (shared by a frame's rows, distinct
    across periods).
  - `_check_reference_env(pandas_version: str, asp_version: str) -> None` — raises `RuntimeError` unless
    pandas major `< 3` and `asp_version == "2.0.15"`.
  - `reference_leg_arrays(frames: pd.DataFrame) -> dict[str, np.ndarray]` — keys
    `team_keys (N,3) object`, `team_as (N,)`, `team_das (N,)`, `player_keys (M,4) object`,
    `player_as (M,)`, `player_das (M,)`; sets the fail-loud warning filter, checks env, applies offside
    via accessible-space with the unique frame key.
  - `__main__`: `python _das_reference_leg.py <in_parquet> <out_dir>` writes `team.parquet` and
    `player.parquet` into `out_dir`. (No `env.parquet` — PLAN-03: the driver's pre-pass `_probe_reference_env`
    captures the full reference environment for provenance, and the subprocess-side `_check_reference_env`
    guards it per match, so a per-match env table would be dead output. `_reference_env` is therefore not
    part of the module.)

**TDD execution order (PLAN-02):** write the Step 2 tests FIRST and run them (Step 3 — they fail:
`ModuleNotFoundError: scripts._das_reference_leg`), THEN add the module code (Step 1, then Step 4),
re-running the tests after each until green. The code blocks are listed implementation-first only for
reading; execute tests-first.

- [ ] **Step 1: Create the module skeleton (constants + pure helpers, lazy accessible-space).**

```python
"""Faithful accessible-space 2.0.15 DAS reference leg, run in a pinned pandas-2 subprocess.

sk-free by construction: top-level imports are pandas/numpy/stdlib only, and ``accessible_space`` is
imported LAZILY inside :func:`reference_leg_arrays` so the driver and CI can import this module for the
constants / gate / keying helper without the library present. See
docs/superpowers/specs/2026-09-27-das-parity-reference-pandas2-design.md.
"""
from __future__ import annotations

import importlib.metadata
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_X_OFFSET = 52.5
_Y_OFFSET = 34.0

# Byte-for-byte the golden generator's ``_COMMON`` (tests/tracking/_fixtures/das_golden/_generate.py).
# ``attacking_direction_col`` is the driver's shared per-frame direction column name.
_REFERENCE_COMMON: dict[str, Any] = dict(
    frame_col="frame_id",
    player_col="player_id",
    team_col="team_id",
    x_col="x",
    y_col="y",
    vx_col="vx",
    vy_col="vy",
    team_in_possession_col="team_in_possession",
    ball_player_id="ball",
    period_col="period_id",
    attacking_direction_col="_das_parity_dir",
    infer_attacking_direction=False,
    use_progress_bar=False,
)


def _reference_lib_frames(frames: pd.DataFrame) -> pd.DataFrame:
    """Centre coords to the accessible-space origin, float64, ball id/team, string ``p<id>``/``t<id>``
    ids, numpy-object string columns (pandas-2 has no arrow default, but keep the coercion so the recipe
    is identical to the frozen golden generator)."""
    out = frames.copy()
    out["x"] = out["x"].astype("float64") - _X_OFFSET
    out["y"] = out["y"].astype("float64") - _Y_OFFSET
    out["vx"] = out["vx"].astype("float64")
    out["vy"] = out["vy"].astype("float64")
    is_ball = out["is_ball"].to_numpy(dtype=bool)
    pid = out["player_id"].astype("object")
    pid[is_ball] = "ball"
    pid[~is_ball] = ["p" + str(int(v)) for v in out.loc[~is_ball, "player_id"]]
    out["player_id"] = pid
    tid = out["team_id"].astype("object")
    tid[is_ball] = None
    tid[~is_ball] = ["t" + str(int(v)) for v in out.loc[~is_ball, "team_id"]]
    out["team_id"] = tid
    out["team_in_possession"] = ["t" + str(int(v)) if pd.notna(v) else None for v in out["team_in_possession"]]
    for c in ("player_id", "team_id", "team_in_possession"):
        out[c] = out[c].astype(object)
    return out


def _add_unique_frame_col(lib: pd.DataFrame, col: str = "_uframe") -> pd.DataFrame:
    """Dense group code over (game_id, period_id, frame_id): one code per frame, shared by its rows,
    distinct across periods. Feeds accessible-space's ``frame_col`` so its frame-id-only pivot cannot
    merge frames that reuse ``frame_id`` between periods."""
    out = lib.copy()
    out[col] = out.groupby(["game_id", "period_id", "frame_id"], sort=True, observed=True).ngroup()
    return out


def _check_reference_env(pandas_version: str, asp_version: str) -> None:
    if int(pandas_version.split(".")[0]) >= 3:
        raise RuntimeError(f"reference leg requires pandas<3 (accessible-space 2.0.15 CoW offside bug); got {pandas_version}")
    if asp_version != "2.0.15":
        raise RuntimeError(f"reference leg requires accessible-space==2.0.15 (frozen oracle); got {asp_version}")
```

(`importlib.metadata` remains imported — `reference_leg_arrays` reads the accessible-space version through
it for `_check_reference_env`. No `_reference_env` / `platform` in the module: provenance comes from the
driver's `_probe_reference_env`.)

- [ ] **Step 2: Write failing tests for the pure helpers + guards.**

```python
# tests/scripts/test_das_reference_leg.py
import importlib
import warnings
import numpy as np
import pandas as pd
import pytest

from scripts import _das_reference_leg as R


def test_module_imports_without_accessible_space(monkeypatch):
    # The lazy-import contract: importing the module and using its pure helpers must NOT require
    # accessible_space (absent in CI). Simulate absence and re-import.
    import builtins
    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name == "accessible_space" or name.startswith("accessible_space."):
            raise ImportError("accessible_space blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    mod = importlib.reload(R)
    assert callable(mod._reference_lib_frames)
    assert mod._REFERENCE_COMMON["frame_col"] == "frame_id"
    importlib.reload(R)  # restore


def test_reference_common_matches_golden_generator():
    # NEW gate (no such test existed): the recipe must equal the frozen golden generator's _COMMON,
    # modulo attacking_direction_col (the generator uses "dir"; the parity pins the shared dir column).
    gen = importlib.import_module("tests.tracking._fixtures.das_golden._generate")
    common = dict(R._REFERENCE_COMMON)
    expected = dict(gen._COMMON)
    common.pop("attacking_direction_col")
    expected.pop("attacking_direction_col", None)
    assert common == expected


def test_unique_frame_col_distinguishes_reused_frame_id_across_periods():
    lib = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1],
            "period_id": [1, 1, 2, 2],
            "frame_id": [5, 5, 5, 5],  # reused across periods
            "player_id": ["a", "b", "a", "b"],
        }
    )
    out = R._add_unique_frame_col(lib)
    # rows within one (game,period,frame) share a code; the two periods differ.
    assert out.loc[0, "_uframe"] == out.loc[1, "_uframe"]
    assert out.loc[2, "_uframe"] == out.loc[3, "_uframe"]
    assert out.loc[0, "_uframe"] != out.loc[2, "_uframe"]


@pytest.mark.parametrize(
    "pv,av,ok",
    [("2.3.3", "2.0.15", True), ("3.0.6", "2.0.15", False), ("2.3.3", "2.1.0", False)],
)
def test_check_reference_env(pv, av, ok):
    if ok:
        R._check_reference_env(pv, av)
    else:
        with pytest.raises(RuntimeError):
            R._check_reference_env(pv, av)


def test_offside_warning_is_an_error_under_the_filter():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Offside not properly detectable")
        with pytest.raises(UserWarning):
            warnings.warn("Offside not properly detectable, maybe too few defenders. Ignoring offside.")
```

- [ ] **Step 3: Run the tests to verify they fail (module not yet created).**

Run: `python -m pytest tests/scripts/test_das_reference_leg.py -v`
Expected: FAIL at collection — `ModuleNotFoundError: No module named 'scripts._das_reference_leg'`
(the module is created in Step 1, which — per the TDD note above — is executed AFTER this step).

- [ ] **Step 4: Implement `reference_leg_arrays` + `__main__` (lazy accessible-space).**

```python
def reference_leg_arrays(frames: pd.DataFrame) -> dict[str, np.ndarray]:
    """Faithful accessible-space DAS reference for one match (all periods). Fail-loud on a bad env or a
    silent offside skip."""
    warnings.filterwarnings("error", message="Offside not properly detectable")
    _check_reference_env(pd.__version__, importlib.metadata.version("accessible-space"))
    import accessible_space as asp  # lazy: only when actually running the leg

    lib = _reference_lib_frames(frames).reset_index(drop=True)
    lib = _add_unique_frame_col(lib)  # collision-free frame key
    common = {**_REFERENCE_COMMON, "frame_col": "_uframe"}
    team = asp.get_dangerous_accessible_space(lib.copy(), **common)
    ind = asp.get_individual_dangerous_accessible_space(lib.copy(), **common)
    as_t = np.asarray(team.acc_space, dtype=float)
    das_t = np.asarray(team.das, dtype=float)
    as_p = np.asarray(ind.player_acc_space, dtype=float)
    das_p = np.asarray(ind.player_das, dtype=float)

    kept = lib["team_in_possession"].notna().to_numpy()
    per_frame: dict[tuple, tuple[float, float]] = {}
    player_rows: list[tuple] = []
    ti = 0
    for i in range(len(lib)):
        if not kept[i]:
            continue
        r = lib.iloc[i]
        key = (r["game_id"], r["period_id"], r["frame_id"])
        per_frame.setdefault(key, (as_t[ti], das_t[ti]))
        if not bool(r["is_ball"]):
            player_rows.append((*key, int(str(r["player_id"])[1:]), as_p[ti], das_p[ti]))
        ti += 1

    team_keys = sorted(per_frame)
    pid_sorted = sorted(player_rows, key=lambda t: (t[0], t[1], t[2], t[3]))
    return {
        "team_keys": np.array(team_keys, dtype=object),
        "team_as": np.array([per_frame[k][0] for k in team_keys], dtype=float),
        "team_das": np.array([per_frame[k][1] for k in team_keys], dtype=float),
        "player_keys": np.array([r[:4] for r in pid_sorted], dtype=object),
        "player_as": np.array([r[4] for r in pid_sorted], dtype=float),
        "player_das": np.array([r[5] for r in pid_sorted], dtype=float),
    }


def _main(in_parquet: str, out_dir: str) -> None:
    frames = pd.read_parquet(in_parquet)
    arrays = reference_leg_arrays(frames)
    out = Path(out_dir)
    tk = arrays["team_keys"]
    pd.DataFrame(
        {"game_id": tk[:, 0], "period_id": tk[:, 1], "frame_id": tk[:, 2], "as": arrays["team_as"], "das": arrays["team_das"]}
    ).to_parquet(out / "team.parquet")
    pk = arrays["player_keys"]
    pd.DataFrame(
        {"game_id": pk[:, 0], "period_id": pk[:, 1], "frame_id": pk[:, 2], "player_id": pk[:, 3], "as": arrays["player_as"], "das": arrays["player_das"]}
    ).to_parquet(out / "player.parquet")


if __name__ == "__main__":
    _main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 5: Run the module tests to verify they pass.**

Run: `python -m pytest tests/scripts/test_das_reference_leg.py -v`
Expected: PASS. (`reference_leg_arrays` / `_main` are not exercised offline — no accessible-space
locally; they are covered by the owner-run DGX smoke, spec §8.)

---

### Task 2: driver — reference-python locate, env probe, subprocess reference leg

**Files:**
- Modify: `scripts/validate_das_native_parity.py` (delete the in-process `_reference_leg` +
  `_reference_lib_frames` + `_REFERENCE_COMMON`; add `_resolve_reference_python`, `_probe_reference_env`,
  `_reference_leg_subprocess`; import `_REFERENCE_COMMON` / `_reference_lib_frames` from Task 1's module;
  **drop the def-time `reference_leg=_reference_leg` default from BOTH `_measure_match` (:426) AND
  `run_corpus` (:710)** — leaving either referencing the deleted name is a `NameError` at import that
  fails the whole module (and every test) to collect; wire `main`; fix the stale `_REFERENCE_COMMON`
  comment).
- Test: `tests/scripts/test_das_native_parity_driver.py` (add locate/probe/subprocess tests; update the
  7th affected test — `test_measure_match_empty_frames_returns_columns`, which calls `_measure_match`
  bare).

**Interfaces:**
- Consumes (Task 1): `scripts._das_reference_leg._REFERENCE_COMMON`, `_reference_lib_frames`,
  `_check_reference_env`, module file path.
- Produces:
  - `_resolve_reference_python(arg: str | None) -> str`
  - `_probe_reference_env(reference_python: str) -> dict[str, str]`
  - `_reference_leg_subprocess(frames: pd.DataFrame, *, reference_python: str) -> dict[str, np.ndarray]`
  - `_measure_match(item, *, reference_leg, direction_col=None)` and `run_corpus(..., reference_leg, ...)`
    — `reference_leg` becomes a REQUIRED keyword on BOTH (drop the `=_reference_leg` default); `main`
    passes the subprocess closure, `run_corpus` forwards it to `_measure_match`, and the tests inject the
    stub (the seam is unchanged for callers, which already pass `reference_leg=`).

- [ ] **Step 1: Write failing tests (monkeypatched subprocess — no real interpreter).**

```python
def test_resolve_reference_python_missing_is_fatal(monkeypatch, tmp_path):
    monkeypatch.delenv("SK_DAS_REFERENCE_PYTHON", raising=False)
    with pytest.raises(SystemExit) as ei:
        D._resolve_reference_python(None)
    assert "accessible-space==2.0.15" in str(ei.value) and "pandas<3" in str(ei.value)


def test_resolve_reference_python_accepts_existing_executable(tmp_path, monkeypatch):
    fake = tmp_path / "python"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("SK_DAS_REFERENCE_PYTHON", str(fake))
    assert D._resolve_reference_python(None) == str(fake)


def test_probe_reference_env_rejects_pandas3(monkeypatch):
    class R:
        stdout = "3.0.6\n2.5.3\n2.0.15\n3.12.0\n"  # pandas / numpy / accessible-space / python
    monkeypatch.setattr(D.subprocess, "run", lambda *a, **k: R())
    with pytest.raises(RuntimeError):
        D._probe_reference_env("pyx")


def test_reference_leg_subprocess_round_trips(monkeypatch, tmp_path):
    frames = pd.DataFrame({"game_id": [1], "period_id": [1], "frame_id": [7], "is_ball": [True],
                           "player_id": ["ball"], "team_id": [None], "x": [0.0], "y": [0.0], "vx": [0.0],
                           "vy": [0.0], "team_in_possession": ["t1"], "_das_parity_dir": [1.0]})

    def fake_run(cmd, **kw):
        out = Path(cmd[-1])  # out_dir is the last arg
        pd.DataFrame({"game_id": [1], "period_id": [1], "frame_id": [7], "as": [3.0], "das": [1.5]}).to_parquet(out / "team.parquet")
        pd.DataFrame({"game_id": [1], "period_id": [1], "frame_id": [7], "player_id": [9], "as": [2.0], "das": [1.0]}).to_parquet(out / "player.parquet")
        class R: returncode = 0
        return R()

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    got = D._reference_leg_subprocess(frames, reference_python="pyx")
    assert got["team_das"].tolist() == [1.5]
    assert got["player_keys"][0].tolist() == [1, 1, 7, 9]
```

- [ ] **Step 2: Run to verify failure.**

Run: `python -m pytest tests/scripts/test_das_native_parity_driver.py -k "reference_python or probe or subprocess" -v`
Expected: FAIL (`_resolve_reference_python` / `_probe_reference_env` / `_reference_leg_subprocess` not
defined).

- [ ] **Step 3: Implement the driver functions.**

```python
import functools
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from scripts._das_reference_leg import _REFERENCE_COMMON, _check_reference_env, _reference_lib_frames  # noqa: F401

_REFERENCE_MODULE = Path(__file__).with_name("_das_reference_leg.py")
_PROVISION_HINT = (
    "python3.12 -m venv ~/das-parity/py2ref && "
    "~/das-parity/py2ref/bin/pip install 'accessible-space==2.0.15' 'pandas<3'"
)
_PROBE_CODE = (
    "import importlib.metadata as m, pandas, numpy, platform, sys;"
    "sys.stdout.write('\\n'.join([pandas.__version__, numpy.__version__, "
    "m.version('accessible-space'), platform.python_version()]) + '\\n')"
)


def _resolve_reference_python(arg: str | None) -> str:
    p = arg or os.environ.get("SK_DAS_REFERENCE_PYTHON")
    if not p or not Path(p).is_file():
        raise SystemExit(
            "reference python not found; pass --reference-python or set SK_DAS_REFERENCE_PYTHON to a "
            f"pandas<3 + accessible-space==2.0.15 interpreter. Provision it with:\n    {_PROVISION_HINT}"
        )
    return p


def _probe_reference_env(reference_python: str) -> dict[str, str]:
    out = subprocess.run([reference_python, "-c", _PROBE_CODE], capture_output=True, text=True, check=True)
    pv, nv, av, pyv = [s for s in out.stdout.splitlines() if s][:4]
    _check_reference_env(pv, av)  # raises on pandas>=3 or asp!=2.0.15
    return {"pandas": pv, "numpy": nv, "accessible_space": av, "python": pyv}


def _reference_leg_subprocess(frames: pd.DataFrame, *, reference_python: str) -> dict[str, np.ndarray]:
    d = Path(tempfile.mkdtemp(prefix="das_ref_"))  # OUTSIDE the repo tree (clean-tree guard, ADR-037)
    try:
        in_pq = d / "in.parquet"
        frames.to_parquet(in_pq)
        subprocess.run([reference_python, str(_REFERENCE_MODULE), str(in_pq), str(d)], check=True)
        team = pd.read_parquet(d / "team.parquet")
        player = pd.read_parquet(d / "player.parquet")
        return {
            "team_keys": team[list(_FRAME_KEYS)].to_numpy(),
            "team_as": team["as"].to_numpy(dtype=float),
            "team_das": team["das"].to_numpy(dtype=float),
            "player_keys": player[[*_FRAME_KEYS, "player_id"]].to_numpy(),
            "player_as": player["as"].to_numpy(dtype=float),
            "player_das": player["das"].to_numpy(dtype=float),
        }
    finally:
        shutil.rmtree(d, ignore_errors=True)
```

Delete the in-process `_reference_leg`, the inline `_reference_lib_frames`, and the inline
`_REFERENCE_COMMON` (now imported from the module). In `main`, after resolving the token, resolve +
probe the reference python and pass the closure:

```python
reference_python = _resolve_reference_python(args.reference_python)
ref_env = _probe_reference_env(reference_python)  # fail-loud before any corpus work
out = run_corpus(refs, _load, dest, prov=prov, shard_root=..., reference_leg=functools.partial(
    _reference_leg_subprocess, reference_python=reference_python), reference_env=ref_env)
```

Add `ap.add_argument("--reference-python", default=None, help="pandas<3 + accessible-space==2.0.15 interpreter")`.
Drop the `=_reference_leg` default from BOTH `_measure_match` (:426) and `run_corpus` (:710), making
`reference_leg` a required keyword on each (`run_corpus` already forwards it at :722). Add a
`reference_env: dict | None = None` keyword to `run_corpus` that it stamps into the artifact (Task 4).

- [ ] **Step 3b: Update the 7th affected test (`test_measure_match_empty_frames_returns_columns`).**

It calls `_measure_match` bare (`D._measure_match(("skillcorner", "1", None, pd.DataFrame()))`); with
`reference_leg` now required that is a `TypeError`. Pass a stub (unused — empty frames return early before
the leg is called):

```python
def test_measure_match_empty_frames_returns_columns():
    shard = D._measure_match(("skillcorner", "1", None, pd.DataFrame()), reference_leg=lambda f: {})
    assert list(shard.columns) == D._EMITTED_SHARD_COLUMNS
    assert shard.empty
```

- [ ] **Step 4: Run to verify pass.**

Run: `python -m pytest tests/scripts/test_das_native_parity_driver.py -k "reference_python or probe or subprocess or measure_match_empty" -v`
Expected: PASS (including the updated `test_measure_match_empty_frames_returns_columns`).

---

### Task 3: driver reduce — re-anchor to `reason == OK`, delete divergence machinery, schema bump

**Files:**
- Modify: `scripts/validate_das_native_parity.py` (`_EMITTED_SHARD_COLUMNS`, `_d_off_per_frame`,
  `_frame_flags`, `_measure_match`, `_team_rows`, `_player_rows`, `_mark_divergences`,
  `_divergence_block`, `reduce_parity`, `_SHARD_SCHEMA_VERSION`).
- Test: `tests/scripts/test_das_native_parity_driver.py` (the 6 reduce/divergence tests
  EDIT/DELETE/RE-ANCHOR below; the 7th affected test — `test_measure_match_empty_frames_returns_columns`,
  broken by the Task 2 signature change, not the reduce change — is updated in Task 2 Step 3b).

**Interfaces:**
- Consumes: the shard columns from `_measure_match`.
- Produces: `reduce_parity(shards)` output WITHOUT a `divergences` key; grade + `finite_mask_mismatches`
  computed over `reason == Reason.OK` rows; `reason_counts` unchanged (per-class §7.2 accounting).

- [ ] **Step 1: Delete the divergence machinery.**
  - `_EMITTED_SHARD_COLUMNS`: remove `"is_d_off"`; KEEP `"reason"`. Bump `_SHARD_SCHEMA_VERSION` to
    `"das-native-parity-3"`.
  - Delete `_d_off_per_frame`. Simplify `_frame_flags` to emit `(game_id, period_id, frame_id, reason)`
    only (drop the `is_d_off` column).
  - `_measure_match`: drop `match_row["is_d_off"] = False` and `shard["is_d_off"] = ...`; KEEP
    `match_row["reason"] = int(Reason.OK)` and `shard["reason"] = shard["reason"].astype("uint8")`.
  - `_team_rows` / `_player_rows`: drop the `"is_d_off"` field from each emitted row (KEEP `"reason"`).
  - Delete `_mark_divergences` and `_divergence_block` entirely.

- [ ] **Step 2: Re-anchor `reduce_parity` to `reason == OK`.**

```python
def _clean_ok(g: pd.DataFrame) -> pd.DataFrame:
    """Rows whose frame is scoreable (reason == OK). On a grain-mixed concat the reason column arrives
    object-typed with NaN, so cast through Int64 first (a plain object .fillna downcast is deprecated in
    pandas 3)."""
    if g.empty:
        return g
    reason = g["reason"].astype("Int64").fillna(int(Reason.OK)).astype("int64")
    return g[reason == int(Reason.OK)]

def reduce_parity(shards: list[pd.DataFrame]) -> dict:
    if not shards:
        return {}
    combined = pd.concat(shards, ignore_index=True)
    if combined.empty:
        return {}
    out: dict = {}
    for provider, sub in combined.groupby("provider"):
        team = sub[sub["grain"] == "team"].copy()
        player = sub[sub["grain"] == "player"].copy()
        match = sub[sub["grain"] == "match"]
        team_ok = _clean_ok(team)
        player_ok = _clean_ok(player)

        def _mask_mismatch(g: pd.DataFrame) -> int:
            return int((g["finite_ref"].astype("boolean") != g["finite_native"].astype("boolean")).sum())

        quad = team["quad_shift_das"].abs()
        nbmax = team["numba_minus_numpy_das"].abs()
        out[str(provider)] = {
            "team": _grade_grain(team_ok) if not team_ok.empty else {},
            "player": _grade_grain(player_ok) if not player_ok.empty else {},
            "finite_mask_mismatches": {"team": _mask_mismatch(team_ok), "player": _mask_mismatch(player_ok)},
            "quadrature_shift_das": {"median": _pct(quad, 50), "p90": _pct(quad, 90), "max": _pct(quad, 100)},
            "numba_vs_numpy_das_max_abs": _pct(nbmax, 100),
            "timings_ms_per_frame": {leg: _pct(match[f"ms_frame_{leg}"], 50) for leg in ("ref", "numpy", "numba", "periodic")},
            "reason_counts": {col: int(match[col].fillna(0).sum()) for col in _REASON_COLS.values()},
            "n_scored_frames": int(match["n_scored_frames"].fillna(0).sum()),
            "n_matches_scored": int(match["game_id"].nunique()),
        }
    return out
```

- [ ] **Step 3: EDIT / DELETE / RE-ANCHOR the 6 tests.**
  - `test_full_reduce_path_is_schema_complete_and_counts_reconcile`: delete the `div = prov["divergences"]`
    block and its assertions (lines currently :110–121). Keep everything else.
  - `test_native_reproduces_the_reference_leg_within_parity`: delete the three
    `prov["divergences"][...]` assertions (:138–140). Keep the `< 1e-6`, `finite_mask == {0,0}`, finite
    `quadrature_shift`.
  - `test_d_off_per_frame_flags_frames_with_fewer_than_two_defenders`: **delete** (and delete the
    `_off_frame` helper, used only here).
  - `test_reduce_excludes_d_off_rows_from_headline_and_counts_them`: **delete**.
  - `test_reduce_counts_and_excludes_d_key_frames_colliding_across_periods`: **delete** (its intent is
    re-homed to `test_unique_frame_col_distinguishes_reused_frame_id_across_periods` in Task 1).
  - `test_reduce_headline_finite_mask_excludes_degrade_reason_frames`: **re-anchor** — replace the
    `is_d_off=` kwargs (now an invalid column) by dropping them; replace
    `assert prov["divergences"]["d_ballnan_frames"] == 1` with the count on the MATCH row and read it from
    `reason_counts`:

```python
def test_reduce_headline_finite_mask_excludes_degrade_reason_frames():
    rows = [
        _mk_row("team", period_id=1, frame_id=1, abs_das=1e-9, rel_das=1e-9, abs_as=1e-9, rel_as=1e-9,
                finite_ref=True, finite_native=True, quad_shift_das=0.0, reason=int(Reason.OK)),
        _mk_row("team", period_id=1, frame_id=2, abs_das=np.nan, rel_das=np.nan, abs_as=np.nan, rel_as=np.nan,
                finite_ref=True, finite_native=False, quad_shift_das=np.nan, reason=int(Reason.BALL_NAN)),
        _mk_row("match", n_scored_frames=2, **{**{c: 0 for c in D._REASON_COLS.values()}, "reason_ball_nan": 1}),
    ]
    prov = D.reduce_parity([_shard(rows)])["skillcorner"]
    assert prov["finite_mask_mismatches"]["team"] == 0, "BALL_NAN row is excluded by the reason==OK filter"
    assert prov["reason_counts"]["reason_ball_nan"] == 1, "the NaN-degrade class is counted"
```

  Also update `_match_row()` if needed (it already sets all `_REASON_COLS` to 0). Update the section
  comment at :218–220 to "reason==OK clean filter + reason_counts accounting" (drop "Divergence
  exclusion").

- [ ] **Step 4: Run the reduce tests to verify pass.**

Run: `python -m pytest tests/scripts/test_das_native_parity_driver.py -v`
Expected: PASS (all remaining tests; the reduce grades over reason==OK, no `divergences` key).

---

### Task 4: provenance stamping + `pyproject` reference-env pin

**Files:**
- Modify: `scripts/validate_das_native_parity.py` (`run_corpus` stamps `reference_env`; `input_contract`
  folds the reference-env pins into the digest).
- Modify: `pyproject.toml` (`das-reference` extra).
- Test: `tests/scripts/test_das_native_parity_driver.py`.

**Interfaces:**
- Consumes: `reference_env` dict from `_probe_reference_env` (Task 2).
- Produces: `metrics.json["reference_env"]`; `input_contract()["params"]["reference_env_pins"]`.

- [ ] **Step 1: Failing test.**

```python
def test_input_contract_declares_reference_env_pins():
    ic = D.input_contract()
    assert ic["params"]["reference_env_pins"] == {"accessible-space": "2.0.15", "pandas": "<3"}

def test_run_corpus_stamps_reference_env(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_numba_available", lambda: False)
    refs, load, stub_ref = _corpus()  # the existing module-level helper (same one _run uses)
    out = D.run_corpus(refs, load, tmp_path / "out", prov=_CLEAN_PROV, direction_col="dir",
                       reference_leg=stub_ref, reference_env={"pandas": "2.3.3", "accessible_space": "2.0.15"})
    assert out["reference_env"] == {"pandas": "2.3.3", "accessible_space": "2.0.15"}
```

(The existing `_run` helper passes no `reference_env`; give `run_corpus` a `reference_env=None` default
so the existing tests keep working, and stamp `{}` when None.)

- [ ] **Step 2: Run to verify failure.** `python -m pytest tests/scripts/test_das_native_parity_driver.py -k "reference_env" -v` → FAIL.

- [ ] **Step 3: Implement.**
  - `run_corpus(..., reference_env: dict | None = None)`: add `"reference_env": reference_env or {}` to
    the `out` dict written to `metrics.json`.
  - `input_contract`: add `"reference_env_pins": {"accessible-space": "2.0.15", "pandas": "<3"}` to
    `params`.
  - `pyproject.toml`: in the `das-reference` optional-dependency group, add `pandas<3` alongside
    `accessible-space==2.0.15` (verify the exact current group name; keep formatting).

- [ ] **Step 4: Run to verify pass.** `python -m pytest tests/scripts/test_das_native_parity_driver.py -k "reference_env" -v` → PASS.

---

### Task 5: docs — ADR-107/108 amendment + `docs/context` note + stale-comment fix

**Files:**
- Modify: `docs/superpowers/adrs/ADR-107-*.md`, `docs/superpowers/adrs/ADR-108-*.md` (Consequences).
- Modify: `docs/context/tracking-features.md`.
- (The stale `_REFERENCE_COMMON` driver comment is fixed in Task 2 Step 3 when the constant moves.)

**Interfaces:** none (docs only).

- [ ] **Step 1: Amend ADR-107 and ADR-108 Consequences** with a paragraph: `accessible-space==2.0.15`
  silently disables offside under pandas-3 Copy-on-Write (read-only `PLAYER_POS`, a caught
  `ValueError`) and conflates frames that reuse `frame_id` across periods (`frame_id`-only pivot); the
  native engine is immune to both, which is why the §7.2 parity leg runs the reference in a pinned
  pandas-2 subprocess. A future maintainer must not "simplify" the subprocess away.

- [ ] **Step 2: Add a `docs/context/tracking-features.md` note** (near the DAS section) with the same
  rule + a pointer to this plan/spec and the root-cause record.

- [ ] **Step 3: Verify the docs render / links resolve** (`python -m pytest tests/test_agents_md_budget.py -q`
  if the note touches any budgeted surface; otherwise a manual read). Expected: no doc-gate failures.

---

### Task 6: integration gate (no commit)

**Files:** none new.

- [ ] **Step 1: Full suite.** `python -m pytest tests/ -m "not e2e" -q` → all pass.
- [ ] **Step 2: Lint (CI scope).** `python -m ruff check silly_kicks/ tests/ scripts/` and
  `python -m ruff format --check silly_kicks/ tests/ scripts/` → clean.
- [ ] **Step 3: Types.** `pyright` (bare) → no new errors.
- [ ] **Step 4: Present the full diff + file list to the owner** and STOP. Do NOT commit. Await explicit
  per-commit approval; then (owner-gated) make ONE commit on `feat/combined-provenance-dgx`, message
  scoped to the whole change, coordinated with the F1b/T10 session. The owner runs the DGX corpus smoke
  (spec §8, including a multi-period reused-`frame_id` match) as final validation.

---

## Self-review

**Spec coverage:** §3.1 subprocess → Task 2; §3.2 sk-free module → Task 1; §3.3 unique-frame keying →
Task 1 (`_add_unique_frame_col`); §3.4 locate → Task 2; §3.5 fail-loud guards → Task 1 (`_check_reference_env`
+ warning filter) & Task 2 (probe); §3.6 reduce re-anchor + deletes + schema bump → Task 3; §3.7 tests →
Tasks 1–4; §3.8 provenance → Task 4; §6 spec amendment → covered by the reduce change; §7 ADR/docs →
Task 5; §8 rollout → Task 6. All covered.

**Placeholder scan:** no TBD/TODO; every code/test step carries real code. The only untested-offline
paths (`reference_leg_arrays`/`_main` end-to-end) are explicitly delegated to the owner DGX smoke, not
hidden.

**Type consistency:** `_reference_leg_subprocess` returns the same dict keys the old `_reference_leg`
did (`team_keys`,`team_as`,`team_das`,`player_keys`,`player_as`,`player_das`), so `_join_on_keys` /
`_team_rows` / `_player_rows` consume it unchanged. `reference_leg` closure signature `(frames) -> dict`
matches the stub the tests inject. `_check_reference_env(pandas_version, asp_version)` is called with the
same argument order in Task 1 tests and Task 2 probe. `_FRAME_KEYS` is the existing
`("game_id","period_id","frame_id")`.
