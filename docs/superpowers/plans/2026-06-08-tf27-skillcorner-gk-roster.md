# TF-27 — SkillCorner external-roster GK verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade `derive_goalkeepers` SkillCorner validation from Tier-2 (self-consistency) to Tier-1 (external roster ground truth) via an e2e gate, plus a CI-runnable synthetic guard — no `silly_kicks/` behaviour change.

**Architecture:** All new code is test + `scripts/` support (SkillCorner `match.json` parsing is loader/consumer territory per ADR-001). A shared pure comparator (`compare_gk_picks`) is the single CI-tested seam used by both the e2e and the synthetic unit test. Frames are built through the reused `_loader_pining` SkillCorner seam (one frame-construction path). The gate is per-match (team_ids recur across the A-League sample) and exact-set-equality (catches over-identification), with an in-comparator sweeper allowlist.

**Tech Stack:** Python, pandas, pytest (`pytest.mark.e2e`), kloppy (SkillCorner parser), pining-for-the-data public token.

**Spec:** `docs/superpowers/specs/2026-06-08-tf27-skillcorner-gk-roster-verification-design.md`

**COMMIT POLICY (overrides the skill's per-task commits):** This branch gets **ONE commit**, created only in the final task **after `/final-review` and explicit user approval**. Individual tasks end at "tests green" — do **not** `git commit` mid-plan.

---

## File Structure

- **Create** `tests/_skillcorner_sample.py` — shared support: `SAMPLE_DIR`, `MATCH_IDS`, `find_artifact`, `available_matches`, `build_skillcorner_gk_truth`, `compare_gk_picks`, `Mismatch`, `AgreementResult`.
- **Create** `tests/tracking/test_gk_skillcorner_roster.py` — synthetic, regular-suite unit tests (truth builder, comparator, derive→compare planted-GK frame).
- **Create** `scripts/download_skillcorner_sample.py` — idempotent sample downloader.
- **Create** `tests/tracking/test_gk_skillcorner_roster_e2e.py` — the Tier-1 e2e gate.
- **Modify** `scripts/_loader_pining.py` — extract `build_skillcorner_frames(paths, tracking_limit)`; `_build_skillcorner` delegates to it.
- **Modify** `tests/spadl/test_skillcorner_e2e.py` — import shared `SAMPLE_DIR`/`MATCH_IDS`/`find_artifact`/`available_matches` (DRY).
- **Modify (doc closure)** `docs/superpowers/adrs/ADR-007*.md`, `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`.

---

## Task 1: Shared support module — truth builder + comparator (TDD)

**Files:**
- Create: `tests/_skillcorner_sample.py`
- Test: `tests/tracking/test_gk_skillcorner_roster.py`

- [ ] **Step 1: Write the failing unit tests**

Create `tests/tracking/test_gk_skillcorner_roster.py`:

```python
"""TF-27: synthetic, CI-runnable guards for the SkillCorner GK-roster verification harness.

No network. Exercises the SAME pure functions the e2e gate uses
(tests/_skillcorner_sample.py), so the comparator is CI-covered (the e2e itself is
-m "not e2e" and does not run in CI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tests._skillcorner_sample import (
    AgreementResult,
    build_skillcorner_gk_truth,
    compare_gk_picks,
)


def _meta(players):
    return {"players": players}


def _p(pid, team, acronym, short_name="X. Y"):
    return {"id": pid, "team_id": team, "short_name": short_name,
            "player_role": {"acronym": acronym}}


# --- build_skillcorner_gk_truth ---

def test_truth_one_gk_per_team_str_keyed():
    meta = _meta([_p(10, 100, "GK"), _p(11, 100, "CB"), _p(20, 200, "GK"), _p(21, 200, "SUB")])
    assert build_skillcorner_gk_truth(meta) == {"100": ["10"], "200": ["20"]}


def test_truth_two_rostered_gks_returns_both_no_raise():
    meta = _meta([_p(10, 100, "GK"), _p(12, 100, "GK")])
    assert build_skillcorner_gk_truth(meta) == {"100": ["10", "12"]}


def test_truth_zero_gk_team_omitted():
    meta = _meta([_p(11, 100, "CB"), _p(20, 200, "GK")])
    truth = build_skillcorner_gk_truth(meta)
    assert "100" not in truth and truth["200"] == ["20"]


# --- compare_gk_picks ---

def test_compare_exact_match_is_perfect():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10"], (999, 200): ["20"]}  # int team key, int->str cast required
    r = compare_gk_picks(truth, picks, match_id=999)
    assert r.is_perfect and len(r.matched) == 2 and not r.mismatched


def test_compare_over_identification_fails_by_default():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10", "11"], (999, 200): ["20"]}  # starter + outfielder
    r = compare_gk_picks(truth, picks, match_id=999)
    assert not r.is_perfect and len(r.mismatched) == 1
    assert r.mismatched[0].team_id == "100"


def test_compare_allowlisted_over_identification_passes_via_subset():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10", "11"], (999, 200): ["20"]}
    r = compare_gk_picks(truth, picks, match_id=999, subset_allowlist=frozenset({("999", "100")}))
    assert r.is_perfect


def test_compare_wrong_pick_fails():
    truth = {"100": ["10"]}
    picks = {(999, 100): ["11"]}
    assert not compare_gk_picks(truth, picks, match_id=999).is_perfect


def test_compare_no_roster_gk_reported_not_failed():
    truth = {"100": ["10"]}  # team 200 has no rostered GK
    picks = {(999, 100): ["10"], (999, 200): ["20"]}
    r = compare_gk_picks(truth, picks, match_id=999)
    assert r.is_perfect and ("999", "200") in r.no_roster_gk


def test_compare_truth_team_missing_from_derived_fails():
    truth = {"100": ["10"], "200": ["20"]}
    picks = {(999, 100): ["10"]}  # derive found no GK for 200
    r = compare_gk_picks(truth, picks, match_id=999)
    assert not r.is_perfect and any(m.team_id == "200" for m in r.mismatched)


def test_cross_match_same_team_id_no_contamination():
    # Two matches share team 100 with DIFFERENT GKs. Per-match compare must pass;
    # a merged-truth implementation (last-match-win) would cross-validate and fail.
    rA = compare_gk_picks({"100": ["10"]}, {(1, 100): ["10"]}, match_id=1)
    rB = compare_gk_picks({"100": ["99"]}, {(2, 100): ["99"]}, match_id=2)
    agg = rA + rB
    assert agg.is_perfect and len(agg.matched) == 2


def test_agreement_result_empty_identity():
    e = AgreementResult.empty()
    assert e.is_perfect and (e + e).is_perfect
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster.py -q`
Expected: FAIL — `ModuleNotFoundError: tests._skillcorner_sample`.

- [ ] **Step 3: Implement the shared module**

Create `tests/_skillcorner_sample.py`:

```python
"""Shared SkillCorner pining-sample test support (TF-27 + the SPADL e2e).

Pure, network-free helpers for the GK-roster verification harness live here so the
e2e gate and the CI synthetic guard call the SAME comparator (no drifting second
path). Filesystem/sample constants are shared with tests/spadl/test_skillcorner_e2e.py.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- sample-dir convention (mirrors the maintainer's pre-download layout) ---

SAMPLE_DIR = Path(
    os.environ.get("SKILLCORNER_SAMPLE_DIR", r"C:\Users\Karsten\AppData\Local\Temp\skillcorner_sample")
)

MATCH_IDS = [
    "1886347", "1899585", "1925299", "1953632", "1996435",
    "2006229", "2011166", "2013725", "2015213", "2017461",
]


def find_artifact(match_dir: Path, suffix: str) -> Path | None:
    """Find an artifact by filename suffix (bare or {id}-prefixed)."""
    candidates = list(match_dir.glob(f"*{suffix}"))
    return candidates[0] if candidates else None


def available_matches(required_suffixes: tuple[str, ...]) -> list[str]:
    """Match ids whose dir contains every required artifact suffix."""
    out = []
    for mid in MATCH_IDS:
        d = SAMPLE_DIR / mid
        if all(find_artifact(d, s) is not None for s in required_suffixes):
            out.append(mid)
    return out


# --- ground-truth extraction (pure) ---

def build_skillcorner_gk_truth(meta: dict) -> dict[str, list[str]]:
    """Map {str(team_id): [str(gk_player_id), ...]} from a SkillCorner match.json.

    Ground truth is players whose ``player_role.acronym == "GK"`` (the starter; subs
    carry "SUB"). Teams with zero GK-acronym players are omitted (no anchor). Never
    raises on cardinality.
    """
    truth: dict[str, list[str]] = {}
    for p in meta.get("players", []):
        role = (p.get("player_role") or {}).get("acronym")
        if role == "GK":
            truth.setdefault(str(p["team_id"]), []).append(str(p["id"]))
    return truth


# --- comparison (pure; the single CI-tested gate) ---

@dataclass(frozen=True)
class Mismatch:
    match_id: str
    team_id: str
    expected: tuple[str, ...]
    got: tuple[str, ...]
    rule: str
    names: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgreementResult:
    matched: tuple[tuple[str, str], ...] = ()
    mismatched: tuple[Mismatch, ...] = ()
    no_roster_gk: tuple[tuple[str, str], ...] = ()

    @classmethod
    def empty(cls) -> "AgreementResult":
        return cls()

    @property
    def is_perfect(self) -> bool:
        return len(self.mismatched) == 0

    def __add__(self, other: "AgreementResult") -> "AgreementResult":
        return AgreementResult(
            self.matched + other.matched,
            self.mismatched + other.mismatched,
            self.no_roster_gk + other.no_roster_gk,
        )

    def summary(self) -> str:
        lines = [
            f"matched={len(self.matched)} mismatched={len(self.mismatched)} "
            f"no_roster_gk={len(self.no_roster_gk)}"
        ]
        for m in self.mismatched:
            names = f" ({', '.join(m.names)})" if m.names else ""
            lines.append(
                f"  MISMATCH match={m.match_id} team={m.team_id} "
                f"expected={list(m.expected)} got={list(m.got)} rule={m.rule}{names}"
            )
        return "\n".join(lines)


def compare_gk_picks(
    truth: dict[str, list[str]],
    derived_picks: dict[tuple, list[str]],
    *,
    match_id: str | int,
    subset_allowlist: frozenset[tuple[str, str]] = frozenset(),
    name_map: dict[str, str] | None = None,
) -> AgreementResult:
    """Compare one match's derived GK picks against its roster truth.

    Default rule: exact set equality per team (catches over-identification). For a
    team in ``subset_allowlist`` (set of ``(str(match_id), str(team_id))``) the rule
    relaxes to ``truth[team] <= picks[team]``. Teams with no roster GK -> no_roster_gk
    (not a failure). String-casts the team key on both sides (derived keys carry int
    team_id). ``name_map`` (id -> short_name) is used only for diagnostics.
    """
    mid = str(match_id)
    names = name_map or {}
    derived: dict[str, list[str]] = {str(tid): [str(p) for p in pids] for (_g, tid), pids in derived_picks.items()}
    matched: list[tuple[str, str]] = []
    mismatched: list[Mismatch] = []
    no_roster: list[tuple[str, str]] = []
    for team in sorted(set(truth) | set(derived)):
        if team not in truth:
            no_roster.append((mid, team))
            continue
        expected = set(truth[team])
        got = set(derived.get(team, []))
        allow = (mid, team) in subset_allowlist
        ok = expected <= got if allow else expected == got
        if ok:
            matched.append((mid, team))
        else:
            ids = sorted(expected | got)
            mismatched.append(
                Mismatch(
                    mid, team, tuple(sorted(expected)), tuple(sorted(got)),
                    "subset" if allow else "exact",
                    tuple(names.get(i, i) for i in ids),
                )
            )
    return AgreementResult(tuple(matched), tuple(mismatched), tuple(no_roster))
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster.py -q`
Expected: PASS (all tests in the file).

---

## Task 2: Synthetic derive→compare end-to-end (no network)

Exercises `derive_goalkeepers` → `compare_gk_picks` once on a planted-GK frame set, so the real algorithm-to-comparator wiring is CI-covered.

**Files:**
- Test: `tests/tracking/test_gk_skillcorner_roster.py` (append)

- [ ] **Step 1: Write the failing test (append to the file from Task 1)**

```python
from silly_kicks.tracking._gk_identification import derive_goalkeepers


def _planted_frames(n_frames: int = 12) -> pd.DataFrame:
    """Two teams; one planted GK each dwelling in its penalty area near goal.

    GK criterion (derive_goalkeepers): in PA (x<16.5 or x>88.5, 13.84<=y<=54.16) for
    >=40% of frames AND mean dist-to-nearest-goal-line <20m. Team-1 GK at x~3, team-2
    GK at x~102; outfielders at midfield (x~52) so they fail both criteria.
    """
    rows = []
    for f in range(1, n_frames + 1):
        base = dict(game_id="g1", period_id=1, frame_id=f, time_seconds=float(f),
                    frame_rate=10.0, is_ball=False, z=0.0)
        # team 1: GK pid=1 near x=3; two outfielders at midfield
        rows.append({**base, "team_id": "100", "player_id": "10", "x": 3.0, "y": 34.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "100", "player_id": "11", "x": 52.0, "y": 30.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "100", "player_id": "12", "x": 60.0, "y": 40.0, "is_goalkeeper": False})
        # team 2: GK pid=20 near x=102; two outfielders at midfield
        rows.append({**base, "team_id": "200", "player_id": "20", "x": 102.0, "y": 34.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "200", "player_id": "21", "x": 53.0, "y": 38.0, "is_goalkeeper": False})
        rows.append({**base, "team_id": "200", "player_id": "22", "x": 45.0, "y": 30.0, "is_goalkeeper": False})
        # ball
        rows.append({**base, "team_id": np.nan, "player_id": np.nan, "x": 50.0, "y": 34.0, "is_ball": True, "is_goalkeeper": False})
    return pd.DataFrame(rows)


def test_derive_then_compare_perfect_on_planted_frames():
    frames = _planted_frames()
    _out, picks = derive_goalkeepers(frames)
    truth = {"100": ["10"], "200": ["20"]}
    result = compare_gk_picks(truth, picks, match_id="g1")
    assert result.is_perfect, result.summary()
```

- [ ] **Step 2: Run to verify it passes**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster.py::test_derive_then_compare_perfect_on_planted_frames -q`
Expected: PASS. (If FAIL: the planted geometry is wrong — confirm GK x within 20m of a goal line and in PA-y band; the test, not the algorithm, is at fault.)

- [ ] **Step 3: Run the whole file**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster.py -q`
Expected: PASS.

---

## Task 3: DRY the existing SPADL e2e onto the shared module

**Files:**
- Modify: `tests/spadl/test_skillcorner_e2e.py:18-48`

- [ ] **Step 1: Replace the local constants/finder with shared imports**

In `tests/spadl/test_skillcorner_e2e.py`, delete the local `_SAMPLE_DIR`, `_MATCH_IDS`, `_find_artifact`, `_has_data` definitions and replace with:

```python
from tests._skillcorner_sample import SAMPLE_DIR as _SAMPLE_DIR
from tests._skillcorner_sample import MATCH_IDS as _MATCH_IDS
from tests._skillcorner_sample import available_matches, find_artifact as _find_artifact


def _has_data():
    return bool(available_matches(("_dynamic_events.csv", "_match.json")))
```

Use the underscore-prefixed suffixes everywhere (that is what the downloader writes, e.g. `1886347_match.json`) — one convention, no foot-gun.

Leave the rest of the file (fixtures, test bodies, `pytestmark = pytest.mark.e2e`) unchanged — they already reference `_SAMPLE_DIR` / `_MATCH_IDS` / `_find_artifact`.

- [ ] **Step 2: Verify collection + no-data skip still works**

Run: `python -m pytest tests/spadl/test_skillcorner_e2e.py -q`
Expected: PASS or all-SKIPPED (sample dir empty on this machine) — NOT a collection/import error.

---

## Task 4: Extract `build_skillcorner_frames` in the loader (single frame path)

**Files:**
- Modify: `scripts/_loader_pining.py:223-244`

- [ ] **Step 1: Extract the frame sub-expression; `_build_skillcorner` delegates**

Replace `_build_skillcorner` (currently lines 223-244) with:

```python
def build_skillcorner_frames(paths, tracking_limit):
    """Preprocessed silly-kicks frames from SkillCorner artifacts (tracking ds path).

    The single SkillCorner frame-construction path: kloppy load -> convert_to_frames
    -> _preprocess (smooth + velocities), yielding SPADL-bounds (0-105/0-68) frames.
    Reused by both _build_skillcorner (calibration) and the TF-27 GK-roster e2e.
    """
    from kloppy import skillcorner

    from silly_kicks.tracking import kloppy as tracking_kloppy

    ds = skillcorner.load(
        meta_data=str(paths["metadata"]),
        raw_data=str(paths["tracking"]),
        limit=tracking_limit,
        include_empty_frames=False,
    )
    frames, _report = tracking_kloppy.convert_to_frames(ds)
    return _preprocess(frames)


def _build_skillcorner(paths, match_id, tracking_limit):
    """SkillCorner: kloppy tracking + silly-kicks SkillCorner events converter."""
    frames = build_skillcorner_frames(paths, tracking_limit)
    # Events: SkillCorner dynamic-events CSV + match.json -> silly-kicks SkillCorner SPADL converter.
    from silly_kicks.spadl import skillcorner as sk_spadl

    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    home_team_id = str(meta["home_team"]["id"])  # authoritative; matches kloppy tracking team ids
    raw_events = pd.read_csv(paths["events"], low_memory=False)
    actions, _evt_report = sk_spadl.convert_to_actions(raw_events, meta)
    return actions, frames, home_team_id
```

Note: `frames` (tracking ds) and `actions` (events CSV) come from independent sources, so this extraction cannot perturb `actions`. It is a **verbatim relocation** of the `_preprocess(frames)` sub-expression with delegation — one frame path, no divergence. **Regression safety** rests on: (a) the verbatim relocation; (b) the **new chain test in Step 2** (CI, monkeypatched — proves `build_skillcorner_frames` does `load → convert_to_frames → _preprocess` and passes `limit`/`include_empty_frames` through); and (c) the Task 7 e2e, which runs `build_skillcorner_frames` on real SkillCorner data and gates on GK correctness. (The existing `test_loader_pining.py` stubs `_build_match` and does **not** exercise this path, so it is NOT the safety net — Step 4 only confirms the module still imports and the unrelated loader functions stay green.)

- [ ] **Step 2: Write the CI chain test for the new seam**

Append to `tests/calibration/test_loader_pining.py` (which already does `import scripts._loader_pining as L`):

```python
def test_build_skillcorner_frames_chains_load_convert_preprocess(monkeypatch):
    # TF-27: the extracted seam must do skillcorner.load -> convert_to_frames -> _preprocess,
    # passing limit + include_empty_frames through. Monkeypatched: no kloppy parse, no network.
    # (function-local imports -> patch the SOURCE module attrs, not L's namespace.)
    sentinel = pd.DataFrame({"x": [1.0, 2.0]})
    seen = {}

    def _fake_load(**kwargs):
        seen["load_kwargs"] = kwargs
        return "DS"

    def _fake_convert(ds):
        seen["ds"] = ds
        return sentinel, None

    monkeypatch.setattr("kloppy.skillcorner.load", _fake_load)
    monkeypatch.setattr("silly_kicks.tracking.kloppy.convert_to_frames", _fake_convert)
    monkeypatch.setattr(L, "_preprocess", lambda f: f.assign(_preprocessed=True))

    out = L.build_skillcorner_frames({"metadata": "m.json", "tracking": "t.jsonl"}, 123)

    assert seen["ds"] == "DS"  # convert_to_frames received the load() result
    assert seen["load_kwargs"]["limit"] == 123
    assert seen["load_kwargs"]["include_empty_frames"] is False
    assert out["_preprocessed"].all() and list(out["x"]) == [1.0, 2.0]  # preprocess applied to convert output
```

- [ ] **Step 3: Run the chain test**

Run: `python -m pytest tests/calibration/test_loader_pining.py::test_build_skillcorner_frames_chains_load_convert_preprocess -q`
Expected: PASS. (If it FAILs before the Step-1 edit: `AttributeError: module ... has no attribute 'build_skillcorner_frames'` — confirms the test targets the new seam.)

- [ ] **Step 4: Import smoke + unrelated loader tests still green**

Run: `python -m pytest tests/calibration/test_loader_pining.py -q -m "not e2e"`
Expected: PASS or SKIPPED (network-gated) — NOT an import/collection error. (Confirms the module still imports and the dedup/retry/fetch/ET/cap tests are unaffected — this is import + unrelated-coverage, NOT regression proof of the extraction, which Steps 2-3 + the e2e provide.)

---

## Task 5: Idempotent sample downloader

**Files:**
- Create: `scripts/download_skillcorner_sample.py`

- [ ] **Step 1: Write the downloader**

Create `scripts/download_skillcorner_sample.py`:

```python
#!/usr/bin/env python
"""Idempotently download the 10 public SkillCorner matches into SKILLCORNER_SAMPLE_DIR.

Populates SAMPLE_DIR/<match_id>/<original_filename> for the match.json, tracking
(extrapolated jsonl), and dynamic-events CSV artifacts — the layout both
tests/spadl/test_skillcorner_e2e.py and tests/tracking/test_gk_skillcorner_roster_e2e.py
read. Uses the pining PUBLIC token (no owner tier needed). Skips files already on disk.

Run: python scripts/download_skillcorner_sample.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run as a bare script (`python scripts/download_skillcorner_sample.py`): put the REPO
# ROOT on sys.path so the namespace-package imports below resolve to the SAME module
# objects pytest uses (no dual-module footgun; scripts/ has no __init__.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._loader_pining import (  # noqa: E402
    _artifact_key,
    _base_url,
    _download_to_temp,
    _list_matches,
    _resolve_token,
)
from tests._skillcorner_sample import MATCH_IDS, SAMPLE_DIR  # noqa: E402

_SUFFIXES = ("_match.json", "_tracking_extrapolated.jsonl", "_dynamic_events.csv")


def main() -> int:
    tok, base = _resolve_token(None), _base_url()
    manifest = {m["id"]: m for m in _list_matches("skillcorner", tok, base)}
    wanted = [mid for mid in MATCH_IDS if mid in manifest] or list(manifest)
    for mid in wanted:
        artifacts = manifest[mid]["artifacts"]
        dest = SAMPLE_DIR / mid
        dest.mkdir(parents=True, exist_ok=True)
        for stale in dest.glob("skillcorner_*"):  # sweep interrupted-download temps
            stale.unlink()
        for suffix in _SUFFIXES:
            key = _artifact_key(artifacts, suffix=suffix)
            target = dest / str(artifacts[key])  # original filename, ends with suffix
            if target.exists():
                print(f"  skip {mid}/{target.name} (present)")
                continue
            tmp = _download_to_temp("skillcorner", mid, key, tok, base, dest)
            tmp.replace(target)  # same-dir atomic move
            print(f"  saved {mid}/{target.name}")
    print(f"done -> {SAMPLE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-check it imports (no download in CI)**

Run: `python -c "import ast; ast.parse(open('scripts/download_skillcorner_sample.py').read())"`
Expected: no output, exit 0. (Actual download happens in Task 7, maintainer-run.)

---

## Task 6: The Tier-1 e2e gate

**Files:**
- Create: `tests/tracking/test_gk_skillcorner_roster_e2e.py`

- [ ] **Step 1: Write the e2e test**

Create `tests/tracking/test_gk_skillcorner_roster_e2e.py`:

```python
"""TF-27: SkillCorner derived-GK vs external-roster Tier-1 validation (e2e).

Requires the 10 public A-League matches pre-downloaded into SKILLCORNER_SAMPLE_DIR
(run scripts/download_skillcorner_sample.py). Anchors derive_goalkeepers against the
match.json roster GK per team; exact-set-equality gate with an in-comparator sweeper
allowlist. Per match (team_ids recur across the sample — never merge truth dicts).
"""

from __future__ import annotations

import json

import pytest

from scripts._loader_pining import build_skillcorner_frames
from silly_kicks.tracking._gk_identification import derive_goalkeepers
from tests._skillcorner_sample import (
    SAMPLE_DIR,
    AgreementResult,
    available_matches,
    build_skillcorner_gk_truth,
    compare_gk_picks,
    find_artifact,
)

_FRAME_CAP = 8000  # probe-confirmed correct picks; anchors the starting GK; bounds runtime

# Genuine sweeper-keeper exceptions, (match_id, team_id) as strings, with justification.
# Empty until the e2e surfaces a real, investigated multi-pick. DO NOT add without one.
_SUBSET_ALLOWLIST: frozenset[tuple[str, str]] = frozenset()

_REQUIRED = ("_match.json", "_tracking_extrapolated.jsonl")

pytestmark = pytest.mark.e2e


def _matches():
    return available_matches(_REQUIRED)


@pytest.mark.skipif(not _matches(), reason="SkillCorner sample (match.json + tracking) not available")
def test_skillcorner_derived_gk_matches_roster():
    overall = AgreementResult.empty()
    for mid in _matches():
        match_dir = SAMPLE_DIR / mid
        meta_path = find_artifact(match_dir, "_match.json")
        trk_path = find_artifact(match_dir, "_tracking_extrapolated.jsonl")
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)

        truth = build_skillcorner_gk_truth(meta)
        name_map = {str(p["id"]): p.get("short_name", str(p["id"])) for p in meta.get("players", [])}

        paths = {"metadata": str(meta_path), "tracking": str(trk_path)}
        frames = build_skillcorner_frames(paths, _FRAME_CAP)

        # Join-key guard (loud, not skip): every rostered GK id must appear in frames,
        # and the overall id overlap must be substantial — a drift on an unprobed match
        # is a structural failure, never a silent "no data" or a confusing GK mismatch.
        frame_ids = {str(x) for x in frames.loc[~frames["is_ball"], "player_id"].dropna().unique()}
        roster_ids = {str(p["id"]) for p in meta.get("players", [])}
        for team, gks in truth.items():
            for gk in gks:
                assert gk in frame_ids, (
                    f"match {mid} team {team}: rostered GK {gk} "
                    f"({name_map.get(gk)}) absent from frame player_ids — id-scheme drift"
                )
        overlap = len(frame_ids & roster_ids)
        assert overlap >= min(20, len(frame_ids)), (
            f"match {mid}: only {overlap} frame ids match the roster — id-scheme drift"
        )

        _out, picks = derive_goalkeepers(frames)
        overall = overall + compare_gk_picks(
            truth, picks, match_id=mid, subset_allowlist=_SUBSET_ALLOWLIST, name_map=name_map
        )

    assert not overall.no_roster_gk, f"teams without a roster GK (unexpected):\n{overall.summary()}"
    assert overall.is_perfect, f"derived GK != roster GK:\n{overall.summary()}"
    assert len(overall.matched) >= 2, "no teams validated — sample present but empty?"
```

- [ ] **Step 2: Verify it collects and skips cleanly without the sample**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster_e2e.py -q`
Expected: 1 SKIPPED (sample not present) — NOT a collection/import error.

---

## Task 7: Run the real Tier-1 validation (maintainer, network)

**Files:** none (execution + possible allowlist/`derive_goalkeepers` fix)

- [ ] **Step 1: Download the sample**

Run: `python scripts/download_skillcorner_sample.py`
Expected: per-artifact "saved …/…"/"skip …" lines, ending `done -> <SAMPLE_DIR>`.

- [ ] **Step 2: Run the e2e gate**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster_e2e.py -q -m e2e`
Expected: PASS (`overall.is_perfect`, ≥20 teams validated across 10 matches).

- [ ] **Step 3: If RED — branch on cause (in scope, not deferred)**

- A **genuine `derive_goalkeepers` gap** (derived a non-GK, or missed a GK present in frames): STOP and use superpowers:systematic-debugging; fix the algorithm in `silly_kicks/_gk_identification.py` with a new red→green unit test in `tests/tracking/test_gk_identification.py`. (This would flip the version bump from patch to minor — note it.)
- A **genuine sweeper multi-pick** (derive returns the true starter + a real sweeper-keeper): add that `(match_id, team_id)` to `_SUBSET_ALLOWLIST` **with a one-line `# justification:` comment**, and add a covering assertion to the synthetic unit test.
- A **join-key/structural assertion**: investigate the loader/id path; do not weaken the guard.

- [ ] **Step 4: Confirm green**

Run: `python -m pytest tests/tracking/test_gk_skillcorner_roster_e2e.py -q -m e2e`
Expected: PASS.

---

## Task 8: Documentation closure + version bump

**Files:**
- Modify: `docs/superpowers/adrs/ADR-007*.md`, `CLAUDE.md`, `TODO.md`, `CHANGELOG.md`, `pyproject.toml`, `silly_kicks/__init__.py`

- [ ] **Step 1: Read the current version + ADR-007 + the CLAUDE.md PR-S26 note**

Run: `grep -n "version" pyproject.toml | head -3; grep -n "__version__" silly_kicks/__init__.py`
Run: `grep -rn "Tier-2\|extrapolated-SkillCorner\|SkillCorner" docs/superpowers/adrs/ADR-007*.md`
Expected: current version `4.18.0`; ADR-007 SkillCorner Tier-2 wording located.

- [ ] **Step 2: ADR-007 — record SkillCorner Tier-1**

In `docs/superpowers/adrs/ADR-007*.md`, update the validation-tier text so SkillCorner is **Tier-1 (external roster ground truth, pining `match.json player_role`, PR-S86)**; Metrica remains the documented permanent limitation (anonymized → no roster). Add a one-line pointer to `tests/tracking/test_gk_skillcorner_roster_e2e.py`.

- [ ] **Step 3: CLAUDE.md — update the PR-S26 note**

In `CLAUDE.md`, change "only Tier-2 algorithm-self-consistency for Metrica + extrapolated-SkillCorner" to record **SkillCorner Tier-1 (external roster, PR-S86)**; Metrica stays Tier-2/blocked.

- [ ] **Step 4: TODO.md — remove the closed TF-27 SkillCorner row**

Delete the `**TF-27 (Wicked):**` row from `## Research & Future Work` (validated/closed — per the "don't leave closed items in TODO sections" rule). Update the `## On Deck` "Most recently shipped" parenthetical if it references TF-27.

- [ ] **Step 5: Version bump (hard gate — all four must match)**

Set `4.18.1` in BOTH `pyproject.toml` (`version = "4.18.1"`) and `silly_kicks/__init__.py` (`__version__ = "4.18.1"`). (If Task 7 forced a `derive_goalkeepers` code fix, use `4.19.0` instead.)

Prepend a `CHANGELOG.md` entry:

```markdown
## 4.18.1 — 2026-06-08

### Added
- TF-27: SkillCorner derived-GK Tier-1 validation against the external pining
  `match.json` roster (`tests/tracking/test_gk_skillcorner_roster_e2e.py`, e2e) +
  a CI-runnable synthetic guard sharing the same pure comparator
  (`tests/_skillcorner_sample.py`). `scripts/download_skillcorner_sample.py`
  populates the sample dir (also unblocks the existing SkillCorner SPADL e2e).

### Changed
- Refactored `scripts/_loader_pining._build_skillcorner` to delegate frame
  construction to a new `build_skillcorner_frames` seam (single frame path; no
  behaviour change — calibration unaffected). Breadcrumb for future calibration work.
- ADR-007 / CLAUDE.md: SkillCorner derived-GK identification upgraded Tier-2 → Tier-1.
```

Also update the `**Last updated**` / `**Current release**` line in `TODO.md` to `4.18.1`.

- [ ] **Step 6: Verify the version-bump gate**

Run: `grep -rn "4.18.1" pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md`
Expected: a hit in all four files.

---

## Task 9: Final verification + single commit (approval-gated)

**Files:** none (verification + the one commit)

- [ ] **Step 1: Lint + types + full non-e2e suite (Shift Left)**

Run: `ruff format --check . ; ruff check . ; pyright silly_kicks/`
Expected: all clean (no `silly_kicks/` change unless Task 7 forced a fix). This mirrors CI (repo-wide), which is green on `main`, so any failure should be in files this PR touched — if pre-existing debt surfaces in untouched files, do NOT fix it here (out of scope); report it.

Run: `python -m pytest tests/ -m "not e2e" -q`
Expected: PASS (includes the new synthetic `tests/tracking/test_gk_skillcorner_roster.py`).

- [ ] **Step 2: Run `/final-review` (mandatory)**

Invoke the `/final-review` skill; address any findings before committing.

- [ ] **Step 3: Get explicit user approval to commit, then ONE commit**

Ask the user to approve the commit. On approval:

```bash
git add -A
git commit -m "$(cat <<'EOF'
test(tracking): TF-27 SkillCorner derived-GK Tier-1 roster validation -- silly-kicks 4.18.1 (PR-S86, ADR-007)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

Expected: one commit on `pr-s86-tf27-skillcorner-gk-roster`. Do NOT tag (await CI green per policy).

---

## Self-Review (completed by plan author)

**Spec coverage:** §3.1.1 shared module (Task 1) ✓; §3.1.2 downloader (Task 5) ✓; §3.1.3 e2e gate, per-match, join-key guard, allowlist (Task 6) ✓; §3.1.4 synthetic CI guard incl. cross-match collision + subset path (Tasks 1-2) ✓; §3.3 frame-seam reuse (Task 4) ✓; §4 exact-equality + allowlist + N=8000 + id dtypes (Tasks 1,6) ✓; §5 TDD order ✓; §6 doc closure + version gate (Task 8) ✓; §7 out-of-scope honoured (no `silly_kicks/` parser).

**Deviation from spec (noted):** §3.3's "equality guard test" (asserting `_build_skillcorner`'s frames equal the seam's) is replaced by a **chain test on `build_skillcorner_frames` itself** (Task 4 Step 2) — because `_build_skillcorner` *delegates* to the seam, asserting their equality would be tautological, whereas the chain test proves the seam does `load → convert → _preprocess` with the right args. Regression safety is honestly: verbatim relocation + the chain test (CI) + the Task 7 e2e on real data. The existing `test_loader_pining.py` is NOT cited as coverage (it stubs `_build_match` and never runs this path).

**Placeholder scan:** none — every code step is complete; the only intentionally-empty value is `_SUBSET_ALLOWLIST = frozenset()` (populated only on a real, justified finding).

**Type consistency:** `AgreementResult`/`Mismatch`/`compare_gk_picks`/`build_skillcorner_gk_truth`/`build_skillcorner_frames`/`available_matches`/`find_artifact` names + signatures are identical across Tasks 1, 2, 3, 4, 6.
