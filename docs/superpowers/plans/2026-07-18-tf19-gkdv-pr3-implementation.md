# TF-19 GKDV PR-3 (gkdv package) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `silly_kicks/gkdv/` — the ghost-substitution engine, the two gate-independent physics arms (ΔDAS, ΔGK-threat-suppression), per-keeper aggregation, and the §6.1–6.3 validation harness plus §6.4 Layer 4 — as ONE feature branch, ONE commit, ONE PR.

**Architecture:** A new hexagonal package mirroring `xtgk/`, importing `tracking/` **public seams only, never the reverse** (pinned by an AST allowlist test). The engine emits a **targets frame as DATA** that the already-shipped probe (`tracking/_model_eval.py`) consumes — the two are decoupled by a typed contract, not by function calls. Three additive serving seams land in `_ghost_gk.py` first, because the probe's contract requires per-row provenance the current serve discards.

**Tech Stack:** pandas / numpy / scipy, optional `accessible-space` (`[das]` extra, lazily imported), pytest. No new runtime dependency.

**Source spec:** `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md` (amended 2026-07-18, two review rounds). **PR-3 scope = §9 item 4 as amended.** §6.4 Layers 0–3 + `gkdv_discrimination_verdict` are **PR-3b**, after owner sign-off — do not build them here.

---

## Pre-flight decisions (resolve BEFORE the tasks they gate)

| # | Decision | Gates | Recommendation |
|---|---|---|---|
| D1 | `_group_metrics.py` **public or private** (spec §6.1, review S8) | Task 11 | **SETTLED — (a) PRIVATE `silly_kicks/_group_metrics.py`, and the RATIONALE CHANGES.** Confirmed with the lakehouse session (2026-07-18): they have **zero** matches for `icc` / `intraclass` / `keeper_spread` / `group_spread` anywhere in code, dbt or docs, and **no foreseeable intent** — per-keeper aggregates are dbt models, and an ICC is a *model-validation* statistic they consume as a verdict, not as a computation. Their words: **"don't mint a surface for us."** ⇒ **Drop the downstream-consumption justification entirely — it is factually wrong for this module.** Use the internal reason, which is sufficient on its own: *gkdv/ is library code and cannot import from `scripts/`.* |
| D2 | **Merge order** vs the parallel session's PR-S119 (spec §9.1, review B3) | Tasks 11, 14 | **SETTLED — do not model the other session's timing.** The owner notifies early if another session merges to `main`, and merges their changes in when that happens; until final-review time this PR worries only about its own work. The durable rules stand regardless of order: **re-derive the aggregator count from `tracking.__all__` at the moment you edit** (never pre-write it), **do not let the `tracking` C4 description grow**, and **whoever lands second regenerates `architecture.html` post-merge** rather than resolving a 294 KB generated-artifact conflict by hand. |
| D3 | **Version + ADR numbers** | Task 15 | **SETTLED — no session owns the next release number, so NEVER claim one early.** Do not write a version or ADR number into any file until the owner steers it at commit time; the owner asks whether the version has been updated *everywhere* when a session is ready to commit. Task 15 Step 1 resolves both mechanically at that moment. |
| D4 | **`[das]` is installed on ZERO CI legs — MEASURED, not open** | Tasks 9, 14 | **SETTLED — option (a): ACTIVATE, in this PR.** Owner rule: *when possible we always do things immediately.* Add `das` to the primary leg **and** keep the structural guard library-free (both — they fix different failures). The "owner-verified only" fallback is **withdrawn**; the ADR records that this PR activated the suites. If any of the 71 fail: **STOP and surface it to the owner** — actual error, file:line, and PROOF it is pre-existing (stash the changes, re-run, show the same failure). **Do NOT classify it as out of scope yourself, and do NOT silently fix or skip it.** Scope is the owner's call, always. |

### D4 measurement (run 2026-07-18 — do not re-derive, act on it)

```
pyproject.toml:51            das = ["accessible-space>=2.0,<3"]     <- the extra EXISTS
.github/workflows/ci.yml:25  pip install ruff==0.15.7 pyright==1.1.409 pandas-stubs==2.3.3.260113
.github/workflows/ci.yml:28  pip install -e ".[test]"
.github/workflows/ci.yml:64  pip install -e ".[kloppy,xgboost,test]"   <- no das
grep -rn "das]" .github/workflows/*.yml  ->  NO MATCH
```

The extra is defined and installable; it is simply never installed. **The CI fix is a one-token edit to line 64** (Task 9, Step 5).

`accessible_space` is imported lazily (`_das.py::_import_accessible_space`), so nothing errors — **it all silently skips**:

| Suite | Tests | Status in CI |
|---|---|---|
| `tests/tracking/test_das.py` | 51 | skipped |
| `tests/tracking/test_das_offside.py` | 7 | skipped |
| `tests/invariants/test_das_invariants.py` | 7 | skipped |
| `tests/tracking/test_das_e2e.py` | 6 | skipped |
| `tests/gkdv/test_arms.py -k das` (this PR) | — | would skip |

**71 tests have had zero automated coverage since TF-28 shipped.** That is a pre-existing gap this plan did not create — but this PR is the moment it becomes load-bearing, because we are adding a second consumer of an unverified subsystem whose correctness depends on a direction-inference subtlety inside it. **A guard that never runs is not a guard.**

**Second measured fact that forces the design (Task 9):** `_pin_attacking_direction` calls `_import_accessible_space()` in its own body and imports `infer_playing_direction` directly. So stubbing `get_individual_das` alone is **NOT sufficient** to make the structural guard library-free — gkdv's contact with accessible-space must be narrowed to a single port that tests can stub. That is the better design regardless: **gkdv depends on a seam, not on `_das`'s internals.**

**Not a blocker:** §6.4 sign-off. Layers 0–3 are PR-3b; only Layer 4 ships here, and it carries no new registered threshold beyond the 0.5 m separation.

---

## Two code facts discovered during planning that constrain the arms

Both were verified in source while writing this plan and are **not** in the spec. Fold them into the ADR.

1. **`lambda_gk` exists ONLY on `SpearmanParams`** (`pitch_control/_params.py:20-46`, default `3.0`). `FernandezBornnParams` and `VoronoiParams` have no GK term, so those methods are **GK-weight-blind** and a ghost-GK substitution through them loses the keeper's 3× influence. **The ΔGK-threat-suppression arm MUST pin `method="spearman"`.** Also note `lambda_gk` is applied *after* the influence field is computed (`_spearman.py:229-235`), so it is a **gain on the GK's contribution, not the mechanism by which position enters** — position enters through TTI. The spec's S9 sensitivity leg is still worth running; describe it as a gain sweep.
2. **A player row with NaN `x`/`y` is SILENTLY DROPPED by pitch control**, not raised on (`_spearman.py:145-146` does `.dropna(subset=["x", "y"])`). So a ghost write-back that produces NaN makes the keeper *vanish* from the surface rather than erroring — another silent-null of exactly the shape this cycle exists to eliminate. **The engine must assert finite ghost coordinates before returning** (Task 5, Step 5).

---

## File structure

**Created**

| File | Responsibility |
|---|---|
| `silly_kicks/gkdv/__init__.py` | Public surface: `build_ghost_frames`, `provenance_to_targets`, `GkdvParams`, `GkdvReport`, `delta_das`, `delta_threat_suppression`, `aggregate_by_keeper`, validation constants |
| `silly_kicks/gkdv/_engine.py` | Ghost-substitution counterfactual engine + the provenance→targets adapter |
| `silly_kicks/gkdv/_arms.py` | The two physics arms + their silent-zero guards |
| `silly_kicks/gkdv/_metric.py` | Observation-level → per-keeper aggregation (frames-resolved GK `player_id`) |
| `silly_kicks/gkdv/_validate.py` | §6.1–6.3 registered constants + Layer 4 behavioural anchoring |
| `silly_kicks/_group_metrics.py` | Domain-free grouped statistics lifted from `scripts/xtgk_v2_keeper_discrimination.py` (see D1) |
| `tests/gkdv/test_engine.py` | Engine domain/drop-accounting/write-back/purity |
| `tests/gkdv/test_provenance_to_targets.py` | The four-point adapter contract + red-first both-teams raise |
| `tests/gkdv/test_arms.py` | Planted-polarity, shared-cache guard, NaN-drop guard, method pin |
| `tests/gkdv/test_metric.py` | Aggregation grain + keeper resolution |
| `tests/gkdv/test_validate.py` | Layer 4 anchoring + both-sides band tests |
| `tests/gkdv/test_import_allowlist.py` | gkdv→tracking public-seams-only AST lint + 2 meta-tests |
| `tests/test_group_metrics.py` | Lifted-statistics unit tests |
| `docs/superpowers/adrs/ADR-043-tf19-gkdv-v1.md` | Decision record (number resolved in Task 15) |
| `docs/PRIVATE_CONSUMERS.md` | Register of downstream code coupling to silly-kicks privates (module → consumer → reason → exit condition) |

**Modified**

| File | Change |
|---|---|
| `silly_kicks/tracking/_ghost_gk.py` | New positions-only seam emitting per-row `ghost_clamped` + `ghost_out_of_box`; id-compat hardening |
| `silly_kicks/tracking/_cover_shadows.py` | New `compute_threat_pc()` facade over `_voronoi_threat` |
| `silly_kicks/tracking/__init__.py` | Export the new serving seam + facade |
| `scripts/xtgk_v2_keeper_discrimination.py` | Re-point at `_group_metrics` (delete-and-depend) |
| `tests/xtgk/test_keeper_discrimination.py` | Re-point |
| `tests/test_public_api_examples.py` | Add gkdv modules (causal/ precedent) |
| `TODO.md`, `CLAUDE.md`, `NOTICE`, `CHANGELOG.md`, `docs/c4/architecture.dsl` | Docs (Task 14) |
| `pyproject.toml`, `silly_kicks/__init__.py` | Version bump (Task 15) |

---

## Test fixture conventions (READ BEFORE WRITING ANY TEST)

**Verified against `tests/tracking/test_ghost_gk.py` at `ec543cc`. Do not substitute anything else — an unverified fixture anchor is exactly the failure this repo's process rule targets.**

| Need | Use | Location |
|---|---|---|
| Ghost-GK-shaped frames (ball, GK, defenders, attackers) | `_make_ghost_gk_frames(*, home_team_id=1, away_team_id=2, period_id=1, frame_id=1, game_id="100", timestamp=1.0)` | `tests/tracking/test_ghost_gk.py:58` |
| Multi-frame variant | `_make_multi_frame_fixture(*, n_frames=5, home_team_id=1, away_team_id=2, game_id="100", fps=25.0)` | `:728` |
| SPADL actions for score/phase context | `_make_spadl_actions(...)` | `:611` |
| **A model** | `_fitted_model(*, n_estimators=10, n_samples=100)` → returns a **`(model, X, labels)` tuple**, so pass `model=_fitted_model()[0]`. Cached and shared — **treat as READ-ONLY.** | `:24` |

**NEVER `model="default"` in any gkdv test.** The spec's parallelism rule (§9 item 4) was deliberately RETAINED as a design principle at the 2026-07-18 amendment, for three reasons that all still hold: `sc_extended` is HF-only and its Hub repos do not exist yet; PR-2's `load()` chirality enforcement is fail-closed, so touching bundled weights couples gkdv's CI to artifact-metadata integrity; and the bundled weights will move again if the attempt track routes to GK feature engineering. Synthetic/fixture models keep gkdv's suite independent of all three.

- [ ] **Make the rule self-enforcing** — add to `tests/gkdv/test_import_allowlist.py`:

```python
def test_no_gkdv_test_pins_the_bundled_default_weights():
    """The spec's retained parallelism rule (§9 item 4), enforced rather than trusted.

    gkdv numerics must come from synthetic/fixture models only. A test that loads the
    bundled `default` variant couples this suite to artifact-metadata integrity (PR-2's
    load() chirality enforcement is fail-closed) and will move when the weights move.
    """
    offenders = []
    for path in sorted((pathlib.Path(__file__).resolve().parent).glob("test_*.py")):
        text = path.read_text(encoding="utf-8")
        if 'model="default"' in text or "model='default'" in text or "from_variant(" in text:
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders}: gkdv tests must use _fitted_model()[0], never the bundled weights "
        "(spec §9 item 4 parallelism rule)."
    )
```

---

## Task 1: Branch and pre-flight

**Files:** none (verification only)

- [ ] **Step 1: Confirm a clean tree on main at the expected commit**

```bash
cd "D:/Development/karstenskyt__silly-kicks_part-deux"
git status --porcelain          # expect EXACTLY these two, and nothing else:
                                #   M docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md
                                #  ?? docs/superpowers/plans/2026-07-18-tf19-gkdv-pr3-implementation.md
git log --oneline -1            # expect: ec543cc ... 4.51.0 ... (or later if PR-S119 landed)
```

**P9:** the spec amendment and this plan are expected to be **present and uncommitted** — they ride in THIS PR's single commit, because the house rule forbids standalone doc commits. Anything *else* in the output is unexpected: stash or resolve it before starting. (An earlier draft said "expect: EMPTY" and then contradicted itself one line later.)

- [ ] **Step 2: Create the feature branch (NEVER a worktree — house rule)**

```bash
git switch -c feat/tf19-gkdv-pr3
```

- [ ] **Step 3: Record the current aggregator count and C4 headroom (do not hard-code them later)**

```bash
.venv/Scripts/python.exe -c "import silly_kicks.tracking as t; n=[x for x in t.__all__ if x.startswith('add_')]; print('raw add_*:', len(n), '| counted:', len(n)-1)"
.venv/Scripts/python.exe -c "
import re; s=open('docs/c4/architecture.dsl',encoding='utf-8').read()
m=[x for x in re.findall(r'\"([^\"]{120,})\"', s) if 'aggregator' in x][0]
print(len(m), '/200 chars'); print(m)"
```

Expected today: counted 29; the tracking description 191/200. **If PR-S119 landed first these differ — use the measured values in Task 14.**

- [ ] **Step 4: Confirm the probe contract we must satisfy has not moved**

```bash
.venv/Scripts/python.exe -c "from silly_kicks.tracking._model_eval import _TARGET_COLUMNS; print(_TARGET_COLUMNS)"
```

Expected exactly:
`('game_id', 'period_id', 'frame_id', 'target_x', 'target_y', 'ghost_clamped', 'ghost_out_of_box')`

---

## ⚠ MANDATORY for every gkdv function taking `home_team_id` (Tasks 5, 8, 9) — id-safety

**[ADDED 2026-07-18 during execution, from the id-dtype root-fix investigation.]** `home_team_id` is a **caller-supplied scalar of uncontrolled dtype**, and a raw `==`/`!=` against an id COLUMN is the single most damaging bug shape in this codebase. It was measured live at `spadl/utils.py:1531` (`away_idx = actions.team_id != home_team_id`): with object-string `team_id` and an int `home_team_id`, `!=` is True for **every** row, so **all HOME rows get mirrored 180°** as well — not merely away rows missed. Five sibling sites carry the same defect today.

`gkdv` is a **brand-new consumer threading exactly that scalar**, and it sits outside the AST lint's non-recursive `tracking/*.py` glob — so nothing would catch a regression here.

**Therefore, non-negotiable in Tasks 5, 8 and 9:**
- Any comparison of an id COLUMN against `home_team_id`, `attacking_team_id`, `gk_team_id` or any other caller-supplied scalar **MUST** go through `silly_kicks.tracking._id_compat` (`ids_match` for column-vs-scalar; `same_id` for scalar-vs-scalar). **Never a raw `==`/`!=`.**
- Add `"silly_kicks.tracking._id_compat"` to `ALLOW_PRIVATE` in `tests/gkdv/test_import_allowlist.py` when first needed — it is a **repo-wide mandated seam** (ADR-019), NOT a confined exemption; see the two-kinds-of-exemption comment in Task 9.
- Each such function gets a regression test passing a **dtype-mismatched-but-value-equal** scalar (e.g. `"1"` where the frame carries `1`) and asserting **identical output**. That test is what makes the requirement stick — the lint provably cannot.

> Note the wider fix (routing the 6 live sites, promoting `_id_compat` to top-level, replacing the lint with an enumeration registry) is **out of PR-3's scope** and tracked separately. What is in scope here is not minting instance number seven.

---

## ⚠ Constraint on `_ghost_gk.py` — its MODULE PATH is load-bearing downstream (read before Tasks 2 and 3)

The lakehouse pins **four silly-kicks private module paths as hardcoded strings** in
`src/ingestion/exec_visibility.py:467-472`, feeding their ADR-044 executor-env drift guard:
`silly_kicks.tracking._ghost_gk`, `._xt_gk`, `._gk_completion`, `._gk_geometry`.

This is a nastier coupling class than an import: **a rename or relocation would not fail at
import time — it would silently degrade their guard.**

**For this PR: we are fine.** Tasks 2 and 3 *modify* `_ghost_gk.py` (adding a serving seam,
per-row provenance, and id-compat hardening) but **do not rename or move it**, and add no new
private module the lakehouse would need to know about.

**Therefore, a hard constraint on execution: do NOT rename, split, or relocate
`silly_kicks/tracking/_ghost_gk.py` in this PR**, however tempting a tidy-up looks while
refactoring its body. If a future cycle wants to, that is a cross-repo coordination item, not a
local decision. Record this in the ADR so the next person refactoring these modules learns it
from us rather than from a broken guard.

---

## Task 2: `_ghost_gk` positions-only serving seam with per-row provenance

The probe requires `ghost_clamped` and `ghost_out_of_box` **per row and non-null** (`_validate_targets` raises otherwise, because `bool(NaN)` is `True` and would silently shrink the trusted stratum). `compute_ghost_gk` currently applies the clamp as a whole-array `np.clip` behind one batch warning and runs an unconditional KDE density pass (~91% of its cost) we do not need.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (body of `compute_ghost_gk` is lines 2056-2162; the shareable region is 2056-2134)
- Modify: `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_ghost_gk_serve_positions.py` (create)

- [ ] **Step 0: Capture a pre-refactor OUTPUT GOLDEN — BEFORE editing anything**

> **Why this step exists.** An earlier draft of this plan claimed "the golden bundle test passing is the proof the refactor is byte-identical." **That was false.** `tests/tracking/test_weights_bundle_golden.py` references `_ghost_gk` only as `("silly_kicks.tracking._ghost_gk", "GhostGkModel")` — an import/loadability check. It never calls `compute_ghost_gk`. The five modules that do call it (`test_ghost_gk`, `_frame_restriction`, `_integration`, `_r3`, `_serve_mean`) assert **structure and behaviour** — columns added, LTR required, two-GK handling — not values. **No output golden exists anywhere**, so extracting a 79-line body would otherwise ship a numeric shift green.

Create `scripts/make_ghost_gk_golden.py`:

```python
"""Capture a compute_ghost_gk output golden for the TF-19 PR-3 refactor equivalence gate.

Run on the UNMODIFIED tree BEFORE Task 2's edit; the npz is the pre-refactor oracle.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np

sys.path.insert(0, ".")
from silly_kicks.tracking import compute_ghost_gk  # noqa: E402
from tests.tracking.test_ghost_gk import _fitted_model, _make_multi_frame_fixture  # noqa: E402

frames = _make_multi_frame_fixture(n_frames=5)
out = compute_ghost_gk(frames, model=_fitted_model()[0], home_team_id=1)
gk = out[out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)]
gk = gk.sort_values(["game_id", "period_id", "frame_id", "team_id"])

np.savez(
    "tests/tracking/data/ghost_gk_refactor_golden.npz",
    ghost_gk_x=gk["ghost_gk_x"].to_numpy(dtype=float),
    ghost_gk_y=gk["ghost_gk_y"].to_numpy(dtype=float),
    ghost_gk_density_spread=gk["ghost_gk_density_spread"].to_numpy(dtype=float),
    source_commit=np.array(
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    ),
)
print("wrote golden for", len(gk), "GK rows")
```

```bash
mkdir -p tests/tracking/data
.venv/Scripts/python.exe scripts/make_ghost_gk_golden.py
```
Expected: `wrote golden for N GK rows` with N > 0. **If N == 0 the golden is vacuous — stop and fix the fixture before refactoring.**

Then add the equivalence gate (it passes trivially now and must KEEP passing after Step 3):

```python
# tests/tracking/test_ghost_gk_refactor_equivalence.py
"""compute_ghost_gk output must not move when its body is extracted (TF-19 PR-3 Task 2)."""
from __future__ import annotations

import pathlib

import numpy as np

from silly_kicks.tracking import compute_ghost_gk
from tests.tracking.test_ghost_gk import _fitted_model, _make_multi_frame_fixture

GOLDEN = pathlib.Path(__file__).parent / "data" / "ghost_gk_refactor_golden.npz"


def test_compute_ghost_gk_output_matches_the_pre_refactor_golden():
    ref = np.load(GOLDEN, allow_pickle=False)
    frames = _make_multi_frame_fixture(n_frames=5)
    out = compute_ghost_gk(frames, model=_fitted_model()[0], home_team_id=1)
    gk = out[out["is_goalkeeper"].astype(bool) & ~out["is_ball"].astype(bool)]
    gk = gk.sort_values(["game_id", "period_id", "frame_id", "team_id"])

    assert len(gk) == len(ref["ghost_gk_x"]) > 0, "golden is vacuous or row count moved"
    for col in ("ghost_gk_x", "ghost_gk_y", "ghost_gk_density_spread"):
        np.testing.assert_allclose(
            gk[col].to_numpy(dtype=float), ref[col], rtol=1e-9, atol=0.0,
            err_msg=f"{col} moved: the body extraction was NOT behaviour-preserving",
        )
```

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_refactor_equivalence.py -v
```
Expected: PASS on the unmodified tree — that is the baseline the refactor must preserve.

- [ ] **Step 1: Write the failing tests**

```python
# tests/tracking/test_ghost_gk_serve_positions.py
"""serve_ghost_gk_positions: positions-only serve + per-row clamp/OOD provenance (TF-19 PR-3)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import serve_ghost_gk_positions
from silly_kicks.tracking._ghost_gk import GRID_X_MAX


def _frames() -> pd.DataFrame:
    """Two frames, one GK per team, ball present. Reuses the shared ghost fixture shape."""
    from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames

    return _make_ghost_gk_frames()


def test_returns_one_row_per_frame_and_gk_team():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    assert set(out.columns) >= {
        "game_id", "period_id", "frame_id", "gk_team_id",
        "ghost_gr_x", "ghost_gr_y", "ghost_clamped", "ghost_out_of_box",
    }
    assert not out.duplicated(subset=["game_id", "period_id", "frame_id", "gk_team_id"]).any()


def test_flags_are_non_null_booleans():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    for col in ("ghost_clamped", "ghost_out_of_box"):
        assert out[col].notna().all(), f"{col} must be non-null (bool(NaN) is True)"
        assert out[col].dtype == bool


def test_positions_are_finite():
    out = serve_ghost_gk_positions(_frames(), model=_fitted_model()[0], home_team_id=1)
    assert np.isfinite(out["ghost_gr_x"]).all()
    assert np.isfinite(out["ghost_gr_y"]).all()


# --- BOTH FLAGS MUST FIRE (spec §7) -------------------------------------------------
# These two are the load-bearing tests in this module. The flags define the probe's
# TRUSTED STRATUM (§3.1(3)): if either is structurally always-False, the dose-banded
# gate silently evaluates on everything and PR-3b inherits a dead stratification.
# Every OTHER test here is satisfiable with both flags all-False.

def test_out_of_box_flag_FIRES_on_a_planted_beyond_hull_ghost(monkeypatch):
    """Plant a ghost past GRID_X_MAX (30 m goal-relative); the flag must fire."""
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model), "predict_mean",
        lambda self, X: np.column_stack([np.full(len(X), 45.0), np.full(len(X), 34.0)]),
    )
    out = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)
    assert out["ghost_out_of_box"].any(), "out-of-box flag never fires -> stratum is dead"
    assert not out["ghost_clamped"].any(), "45 m is on-pitch; the clamp must NOT fire here"


def test_clamped_flag_FIRES_on_a_planted_off_pitch_ghost(monkeypatch):
    """Plant a ghost outside the physical pitch; the clamp flag must fire."""
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model), "predict_mean",
        lambda self, X: np.column_stack([np.full(len(X), -12.0), np.full(len(X), 34.0)]),
    )
    with pytest.warns(UserWarning, match="outside the physical pitch"):
        out = serve_ghost_gk_positions(_frames(), model=model, home_team_id=1)
    assert out["ghost_clamped"].all(), "clamp flag never fires -> per-row provenance is dead"
    assert (out["ghost_gr_x"] >= 0.0).all(), "a clamped position must land back on the pitch"
```

> **P5 — a tautology was REMOVED here.** An earlier draft asserted
> `out["ghost_out_of_box"] == (out["ghost_gr_x"] > GRID_X_MAX)`, which merely restates the
> implementation (`ghost_out_of_box = positions[:,0] > GRID_X_MAX` and
> `ghost_gr_x = positions[:,0]`): it asserts `a > k == a > k` and **cannot fail**. The
> property its name claimed — "before write-back" — is also **not testable in this module**,
> because `serve_ghost_gk_positions` performs no write-back. The real property (*the flag
> keys on the goal-relative value and SURVIVES write-back*) is tested in Task 5.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_serve_positions.py -v`
Expected: FAIL — `ImportError: cannot import name 'serve_ghost_gk_positions'`

- [ ] **Step 3: Implement the seam**

Refactor so `compute_ghost_gk` and the new function share one private body. First declare the warning category near the module's other constants (P8) — and check whether PR-S119 has landed a `_warnings.py` convention to reuse rather than duplicating one:

```bash
ls silly_kicks/tracking/_warnings.py 2>/dev/null && grep -n "class .*Warning" silly_kicks/tracking/_warnings.py
```

```python
class GhostClampWarning(UserWarning):
    """The served ghost position fell outside the physical pitch and was clamped.

    A dedicated category so a consumer can silence the batch-clamp notice without
    silencing every ``UserWarning`` from ``tracking`` -- it is emitted from two public
    entry points (``compute_ghost_gk`` and ``serve_ghost_gk_positions``).
    """
```

Then in `silly_kicks/tracking/_ghost_gk.py`, add after `compute_ghost_gk`:

```python
def _serve_positions_core(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None,
    home_team_id: int | str,
    actions: pd.DataFrame | None,
    carrier: pd.DataFrame | None,
    link_frame_ids: set[int] | None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Shared serve: (meta, positions, clamped_mask).

    Single-sources model resolution, context callbacks, feature extraction, the 4.12.1
    duplicate-(frame, gk_team) collapse, ``predict_mean`` and the 4.22.1 physical-pitch
    clamp. Returns the goal-relative positions AND the per-row clamp mask captured
    BEFORE ``np.clip`` -- the information ``compute_ghost_gk`` discards.
    """
    resolved = _resolve_model(model)
    score_fn = _build_score_lookup(actions, home_team_id) if actions is not None else None
    phase_fn = _build_phase_lookup(actions) if actions is not None else None
    if carrier is None:
        carrier_raw = infer_ball_carrier(frames, **resolved.carrier_params)
        carrier = carrier_raw[["game_id", "period_id", "frame_id", "ball_carrier_team_id"]]

    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        carrier=carrier,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
        link_frame_ids=link_frame_ids,
    )
    if len(batch_features) == 0:
        return meta, np.empty((0, 2), dtype=float), np.empty(0, dtype=bool)

    _key_cols = ["game_id", "period_id", "frame_id", "gk_team_id"]
    _keep = ~meta.duplicated(subset=_key_cols, keep="first")
    if not _keep.all():
        keep_mask = _keep.to_numpy()
        meta = meta[keep_mask].reset_index(drop=True)
        batch_features = batch_features[keep_mask].reset_index(drop=True)

    positions = resolved.predict_mean(batch_features)

    _lo = np.array([0.0, 0.0])
    _hi = np.array([_FIELD_LENGTH, _FIELD_WIDTH])
    clamped = ((positions < _lo) | (positions > _hi)).any(axis=1)
    if bool(clamped.any()):
        # P7: stacklevel is 3, NOT the original 2. The warning has moved one frame deeper
        # (user -> compute_ghost_gk -> _serve_positions_core -> warn), so stacklevel=2 would
        # now point at library internals instead of the caller. 3 is correct for BOTH public
        # entry points, since serve_ghost_gk_positions sits at the same depth.
        # P8: category added while re-homing. The message is UNCHANGED (Chesterton), but the
        # warning is now emitted from a SECOND public entry point, and a consumer wanting to
        # silence the batch-clamp notice should not have to silence every UserWarning from
        # tracking. If this is declined, record the omission as a decision in the ADR.
        warnings.warn(
            "ghost-GK: one or more served positions fell outside the physical pitch and "
            "were clamped; suspect upstream tracking quality (e.g. a mis-flagged "
            "is_goalkeeper).",
            GhostClampWarning,
            stacklevel=3,
        )
        positions = np.clip(positions, _lo, _hi)
    return meta, positions, clamped


def serve_ghost_gk_positions(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    carrier: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Serve ghost-GK positions ONLY, with per-row clamp / out-of-training-box provenance.

    Positions-only sibling of :func:`compute_ghost_gk`: it skips the KDE density pass
    (the entire cost driver) and, unlike ``compute_ghost_gk``, returns the per-row
    ``ghost_clamped`` mask instead of collapsing it into one batch warning. Coordinates
    are GOAL-RELATIVE (``x`` = distance from the defended goal line) -- the caller does
    the write-back to frame coordinates.

    ``ghost_out_of_box`` marks positions beyond the ghost's trained label hull
    (``GRID_X_MAX`` = 30 m) and is evaluated on goal-relative ``x`` BEFORE any flip.

    Returns
    -------
    pd.DataFrame
        One row per ``(game_id, period_id, frame_id, gk_team_id)`` with ``ghost_gr_x``,
        ``ghost_gr_y``, ``ghost_clamped``, ``ghost_out_of_box``.

    Examples
    --------
    >>> out = serve_ghost_gk_positions(frames, model=_fitted_model()[0], home_team_id=1)  # doctest: +SKIP
    >>> bool(out["ghost_clamped"].notna().all())  # doctest: +SKIP
    True
    """
    meta, positions, clamped = _serve_positions_core(
        frames,
        model=model,
        home_team_id=home_team_id,
        actions=actions,
        carrier=carrier,
        link_frame_ids=link_frame_ids,
    )
    if len(positions) == 0:
        return pd.DataFrame(
            columns=["game_id", "period_id", "frame_id", "gk_team_id",
                     "ghost_gr_x", "ghost_gr_y", "ghost_clamped", "ghost_out_of_box"]
        )
    return pd.DataFrame(
        {
            "game_id": meta["game_id"].to_numpy(),
            "period_id": meta["period_id"].to_numpy(),
            "frame_id": meta["frame_id"].to_numpy(),
            "gk_team_id": meta["gk_team_id"].to_numpy(),
            "ghost_gr_x": positions[:, 0],
            "ghost_gr_y": positions[:, 1],
            "ghost_clamped": clamped.astype(bool),
            "ghost_out_of_box": (positions[:, 0] > GRID_X_MAX),
        }
    )
```

Then rewrite `compute_ghost_gk`'s body lines 2056-2134 to call `_serve_positions_core`, keeping everything from `predict_density` onward unchanged.

- [ ] **Step 4: Export it**

In `silly_kicks/tracking/__init__.py`, add `serve_ghost_gk_positions` to the `_ghost_gk` import block and to `__all__`, adjacent to `compute_ghost_gk`.

- [ ] **Step 5: Run the new tests AND the refactor equivalence gate**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_serve_positions.py -v
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_refactor_equivalence.py -v
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk.py tests/tracking/test_ghost_gk_serve_mean.py \
    tests/tracking/test_ghost_gk_frame_restriction.py tests/tracking/test_ghost_gk_r3.py -v
```
Expected: all PASS. **The Step-0 npz equivalence gate is the proof the body extraction did not move `compute_ghost_gk`'s output** — the behavioural suites above assert structure (columns added, LTR required, two-GK handling), not values, so they alone would let a numeric shift ship green. `test_weights_bundle_golden.py` is NOT a guard here: its only ghost reference is an import/loadability check on `GhostGkModel`.

---

## Task 3: Harden the ghost extractor's raw id comparisons (ADR-019)

`extract_ghost_gk_features` still uses raw `==`/`!=` on team ids. On Gradient Sports frames `team_id` is nullable `Int64` while other providers carry object strings, so a raw compare yields **empty defending/attacking splits and a corrupt feature row rather than an error** — and Task 5 routes every ghost position through this extractor.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (compares near lines 521-523, 584, 771 — **re-derive, do not trust these numbers**)
- Test: `tests/tracking/test_ghost_gk_id_dtype.py` (create)

- [ ] **Step 1: Locate the compares**

```bash
grep -n 'team_id"\] ==\|team_id"\] !=\|== gk_team\|!= gk_team' silly_kicks/tracking/_ghost_gk.py
```

- [ ] **Step 2: Write the failing test**

```python
# tests/tracking/test_ghost_gk_id_dtype.py
"""ADR-019: ghost-GK feature extraction must be dtype-safe on team ids."""
from __future__ import annotations

import pandas as pd

from silly_kicks.tracking import serve_ghost_gk_positions
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def test_string_team_ids_give_same_positions_as_numeric():
    numeric = _make_ghost_gk_frames()
    stringy = numeric.copy()
    stringy["team_id"] = stringy["team_id"].map(lambda v: None if pd.isna(v) else str(v))

    a = serve_ghost_gk_positions(numeric, model=_fitted_model()[0], home_team_id=1)
    b = serve_ghost_gk_positions(stringy, model=_fitted_model()[0], home_team_id="1")

    assert len(a) == len(b) > 0, "string-id path produced a different row count"
    pd.testing.assert_series_equal(
        a["ghost_gr_x"].reset_index(drop=True),
        b["ghost_gr_x"].reset_index(drop=True),
        check_names=False,
    )
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_id_dtype.py -v`
Expected: FAIL — row-count mismatch or differing positions.

- [ ] **Step 4: Route the compares through `_id_compat`**

Replace each raw compare using the helpers already imported elsewhere in `tracking/`:

```python
from ._id_compat import ids_match, ids_differ, same_id

# was:  defending = players[players["team_id"] == gk_team]
defending = players[ids_match(players["team_id"], gk_team)]

# was:  attacking = players[players["team_id"] != gk_team]
attacking = players[~ids_match(players["team_id"], gk_team)]
```

- [ ] **Step 5: Run the new test and the REAL byte-identity gate**

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_ghost_gk_id_dtype.py tests/tracking/test_ghost_gk_refactor_equivalence.py -v
```
Expected: PASS (matched-dtype output unchanged; only the cross-dtype path is fixed).

> **[CORRECTED during execution]** an earlier revision paired this with
> `tests/tracking/test_weights_bundle_golden.py`. That file only import-checks
> `GhostGkModel` — it would NOT detect a numeric shift. Task 2's npz equivalence golden is
> the actual byte-identity proof. Same mistake, same file, third occurrence: **before citing
> a test as a gate, open it and read the assertion body.**

**[EXECUTION FINDING] One of the three hardened sites has NO gate, by construction — and that is the correct outcome, not a gap to paper over.** The velocity-state defending split takes its `gk_team` scalar from `frame_data` itself, so `ids_match` and a raw `==` agree for *every* real dtype: no input discriminates them, and any "test" would pass before and after the fix. It is also not lintable — `tests/tracking/test_id_compat_lint.py` matches only two shapes (`== home_team_id`, and cross-source suffixed subscripts like `team_id_action` vs `team_id_frame`), and `frame_data["team_id"] == gk_team` is neither. **Do NOT extend the lint to cover it**: comparing a column to a scalar drawn from that same column is SAFE and syntactically identical to the unsafe cross-source case, so a broader rule would flag correct code and breed `noqa` exemptions — precisely why ADR-027 concluded the behavioural gate, not the lint, is the backstop. The change stays because it is load-bearing *as consistency*: `TestExtractionRestriction`'s golden requires this block's identity rule to match the extractor's, and Task 3 changes the extractor's. **Record this in the ADR as a reasoned no-gate decision** (Task 14).

---

## Task 4: `gkdv/` skeleton, params, and the import-allowlist gate

**Files:**
- Create: `silly_kicks/gkdv/__init__.py`, `silly_kicks/gkdv/_engine.py` (stub), `silly_kicks/gkdv/_arms.py` (stub), `silly_kicks/gkdv/_metric.py` (stub), `silly_kicks/gkdv/_validate.py` (stub)
- Create: `tests/gkdv/__init__.py`, `tests/gkdv/test_import_allowlist.py`

- [ ] **Step 1: Write the failing allowlist test (imitates `tests/tracking/test_id_compat_lint.py`, including BOTH meta-tests)**

```python
# tests/gkdv/test_import_allowlist.py
"""gkdv -> tracking: PUBLIC SEAMS ONLY, and never the reverse (ADR-037)."""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2] / "silly_kicks"
GKDV = ROOT / "gkdv"
TRACKING = ROOT / "tracking"

#: Private tracking modules gkdv is permitted to import, each with a recorded reason.
#: Additive-only; every entry is a deliberate decision.
ALLOW_PRIVATE = {
    # (none today -- gkdv consumes tracking's PUBLIC surface only)
}


def _imported_tracking_symbols(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "tracking" in node.module:
            tail = node.module.split(".")[-1]
            if tail.startswith("_") and tail not in ALLOW_PRIVATE:
                hits.append(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if "tracking._" in a.name:
                    hits.append(a.name)
    return hits


@pytest.mark.parametrize("path", sorted(GKDV.glob("*.py")), ids=lambda p: p.name)
def test_gkdv_imports_only_public_tracking_seams(path):
    hits = _imported_tracking_symbols(path)
    assert not hits, (
        f"{path.name}: imports PRIVATE tracking module(s) {hits}. Import the public seam "
        "(silly_kicks.tracking.<name>) or add an ALLOW_PRIVATE entry with a reason."
    )


@pytest.mark.parametrize("path", sorted(TRACKING.glob("*.py")), ids=lambda p: p.name)
def test_tracking_never_imports_gkdv(path):
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    for node in ast.walk(tree):
        mod = getattr(node, "module", None) or ""
        names = [a.name for a in getattr(node, "names", [])]
        assert "gkdv" not in mod and not any("gkdv" in n for n in names), (
            f"{path.name}: tracking/ must NEVER import gkdv/ -- the probe consumes ghost "
            "positions as DATA (a targets DataFrame) precisely to keep this direction closed."
        )


def test_detector_fires_on_a_planted_private_import(tmp_path):
    """META: the detector must actually detect. Without this the lint can silently pass."""
    planted = tmp_path / "_planted.py"
    planted.write_text("from silly_kicks.tracking._ghost_gk import GRID_X_MAX\n", encoding="utf-8")
    assert _imported_tracking_symbols(planted), "detector failed to flag a private import"


def test_gkdv_package_is_non_empty():
    """META: pins the gate's surface -- an empty package would make the lint vacuous."""
    modules = sorted(p.name for p in GKDV.glob("*.py"))
    assert len(modules) >= 5, f"expected the full gkdv module set, found {modules}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_import_allowlist.py -v`
Expected: FAIL — `GKDV.glob` yields nothing, so `test_gkdv_package_is_non_empty` fails.

- [ ] **Step 3: Create the package with the frozen params**

```python
# silly_kicks/gkdv/__init__.py
"""GKDV -- GK Deterrent Value (TF-19, ADR-043).

Counterfactual valuation of goalkeeper positioning: how much does the ACTUAL keeper's
position change the attacking team's accessible space and threat, relative to a
league-average "ghost" keeper in the same frame state? Arms are defined in
attacker-value units as ``actual - ghost``, so **negative = deterrent** uniformly.

Depends on ``silly_kicks.tracking`` PUBLIC seams only, never the reverse
(pinned by ``tests/gkdv/test_import_allowlist.py``).

See NOTICE for full bibliographic citations.
"""

from ._arms import delta_das, delta_threat_suppression
from ._engine import GkdvParams, GkdvReport, build_ghost_frames, provenance_to_targets
from ._metric import aggregate_by_keeper

__all__ = [
    "GkdvParams",
    "GkdvReport",
    "aggregate_by_keeper",
    "build_ghost_frames",
    "delta_das",
    "delta_threat_suppression",
    "provenance_to_targets",
]
```

```python
# silly_kicks/gkdv/_engine.py  (params + report; engine body lands in Task 5)
"""Ghost-substitution counterfactual engine (spec §4)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Attacking-third predicate: ball within this distance of the attacked goal (spec §4.1).
_DOMAIN_BALL_TO_GOAL_M = 35.0


@dataclass(frozen=True)
class GkdvParams:
    """Registered knobs for the GKDV v1 counterfactual. Frozen; echoed into GkdvReport.

    Examples
    --------
    >>> GkdvParams().possession_stride
    5
    """

    #: Sample every Nth eligible frame per possession (cost control, spec §5).
    possession_stride: int = 5
    #: Ball-to-attacked-goal distance bounding the domain, metres.
    domain_ball_to_goal_m: float = _DOMAIN_BALL_TO_GOAL_M
    #: Pitch-control method for the threat arm. HARD CONSTRAINT, not guidance: lambda_gk
    #: exists ONLY on SpearmanParams, so any other method silently produces a GK-BLIND arm.
    #: Validated fail-loud in __post_init__ -- the field is kept (rather than hard-wired) so
    #: the constraint is self-documenting and a future GK-aware method can join the allowlist.
    pitch_control_method: str = "spearman"
    #: GK control-rate multiplier, surfaced here because it governs the threat arm's gain.
    lambda_gk: float = 3.0
    #: Ghost keeps the factual keeper's velocity (minimal-intervention counterfactual).
    ghost_keeps_actual_velocity: bool = True
    #: Deterministic seed threaded through both legs so identity assertions stay exact.
    seed: int = 42

    #: Methods that carry a GK term. Only these may score the threat arm.
    _GK_AWARE_METHODS = ("spearman",)

    def __post_init__(self) -> None:
        """Fail at CONSTRUCTION, not at call time, on a GK-blind pitch-control method."""
        if self.pitch_control_method not in self._GK_AWARE_METHODS:
            raise ValueError(
                f"pitch_control_method={self.pitch_control_method!r} is GK-BLIND: lambda_gk "
                f"exists only on SpearmanParams, so a ghost-GK substitution through it loses "
                f"the keeper's control-rate multiplier entirely and the threat arm would "
                f"measure nothing about the keeper. Allowed: {self._GK_AWARE_METHODS}."
            )


@dataclass(frozen=True)
class GkdvReport:
    """Run-level audit. Echoes the params actually used -- registration without
    traceability is not registration.

    Examples
    --------
    >>> GkdvReport(params=GkdvParams(), n_frames_in=0, n_frames_scored=0,
    ...            drop_reasons={}, n_clamped=0, n_out_of_box=0).n_frames_scored
    0
    """

    params: GkdvParams
    n_frames_in: int
    n_frames_scored: int
    drop_reasons: dict
    n_clamped: int
    n_out_of_box: int
```

Create `_arms.py`, `_metric.py`, `_validate.py` with module docstrings and the function stubs their `__init__` imports (raise `NotImplementedError` for now — they are filled in Tasks 8-12).

- [ ] **Step 4: Run the allowlist test**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_import_allowlist.py -v`
Expected: all 4 test groups PASS.

---

## Task 5: `build_ghost_frames` — the engine

**Files:**
- Modify: `silly_kicks/gkdv/_engine.py`
- Test: `tests/gkdv/test_engine.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/gkdv/test_engine.py
"""build_ghost_frames: domain, drop accounting, write-back, purity (spec §4)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv import GkdvParams, build_ghost_frames
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def test_returns_counterfactual_provenance_and_report():
    frames = _make_ghost_gk_frames()
    cf, prov, report = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    assert isinstance(cf, pd.DataFrame) and isinstance(prov, pd.DataFrame)
    assert report.n_frames_in >= report.n_frames_scored
    assert set(prov.columns) >= {
        "game_id", "period_id", "frame_id", "gk_team_id", "player_id",
        "ghost_x", "ghost_y", "displacement_m", "ghost_clamped",
        "ghost_out_of_box", "drop_reason",
    }


def test_input_frames_are_never_mutated():
    frames = _make_ghost_gk_frames()
    before = frames.copy(deep=True)
    build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    pd.testing.assert_frame_equal(frames, before)


def test_drop_reasons_conserve():
    frames = _make_ghost_gk_frames()
    _cf, prov, report = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    dropped = int(prov["drop_reason"].notna().sum())
    assert report.n_frames_scored + dropped == report.n_frames_in
    assert sum(report.drop_reasons.values()) == dropped


def test_scored_ghost_coordinates_are_finite():
    """A NaN ghost is SILENTLY DROPPED by pitch control -- never emit one."""
    frames = _make_ghost_gk_frames()
    _cf, prov, _r = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    scored = prov[prov["drop_reason"].isna()]
    assert np.isfinite(scored["ghost_x"]).all()
    assert np.isfinite(scored["ghost_y"]).all()


def test_writeback_places_ghost_at_both_goal_ends():
    """x = gr_x if the defended goal is at x=0 else 105 - gr_x; y unchanged."""
    frames = _make_ghost_gk_frames()
    cf, prov, _r = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    gk_rows = cf[cf["is_goalkeeper"].astype(bool) & ~cf["is_ball"].astype(bool)]
    assert (gk_rows["x"].between(0.0, 105.0)).all()
    assert (gk_rows["y"].between(0.0, 68.0)).all()


def test_out_of_box_flag_keys_on_GOAL_RELATIVE_x_and_survives_writeback(monkeypatch):
    """P5: the real property the serve-side tautology could not express.

    Plant a ghost 45 m goal-relative while the DEFENDED goal is at x=105, so write-back
    gives a frame-coordinate x of 105-45 = 60 -- nowhere near 30. The flag must still be
    True, proving it keys on the goal-relative value and is not recomputed post-flip.
    """
    model = _fitted_model()[0]
    monkeypatch.setattr(
        type(model), "predict_mean",
        lambda self, X: np.column_stack([np.full(len(X), 45.0), np.full(len(X), 34.0)]),
    )
    frames = _make_ghost_gk_frames()
    _cf, prov, _r = build_ghost_frames(frames, model=model, home_team_id=1)
    scored = prov[prov["drop_reason"].isna()]
    assert scored["ghost_out_of_box"].all(), "flag lost across write-back"
    assert (scored["ghost_x"] < 105.0).all() and (scored["ghost_x"] > 0.0).all()
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_engine.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement `build_ghost_frames`**

```python
def build_ghost_frames(
    frames: pd.DataFrame,
    *,
    model=None,
    home_team_id: int | str,
    carrier: pd.DataFrame | None = None,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, pd.DataFrame, GkdvReport]:
    """Build the ghost-keeper counterfactual frames plus per-frame provenance.

    PURE: never mutates ``frames``.

    Domain (spec §4.1): alive ball, in-possession team attacking, ball within
    ``params.domain_ball_to_goal_m`` of the attacked goal, defending-GK row present.
    Frames with a missing/NaN GK block are **dropped-and-counted, never scored as
    Delta = 0** -- a zero delta from a missing keeper reads as "no deterrence" and biases
    keeper aggregates toward the null.

    Returns
    -------
    (counterfactual_frames, provenance, report)

    Examples
    --------
    >>> cf, prov, rep = build_ghost_frames(frames, home_team_id=1)  # doctest: +SKIP
    >>> rep.n_frames_scored <= rep.n_frames_in  # doctest: +SKIP
    True
    """
    from silly_kicks.tracking import serve_ghost_gk_positions

    src = frames  # never mutated
    eligible, drops = _apply_domain(src, params)          # -> (frame keys, {reason: n})
    served = serve_ghost_gk_positions(
        src, model=model, home_team_id=home_team_id, carrier=carrier,
        link_frame_ids=set(eligible["frame_id"].unique()),
    )
    goal_map = _pin_defended_goal(src, home_team_id)       # computed ONCE (spec §4.2)
    prov = _build_provenance(src, served, goal_map, drops)
    cf = _write_back(src, prov, params)
    report = GkdvReport(
        params=params,
        n_frames_in=int(len(eligible) + sum(drops.values())),
        n_frames_scored=int(prov["drop_reason"].isna().sum()),
        drop_reasons=drops,
        n_clamped=int(prov["ghost_clamped"].fillna(False).sum()),
        n_out_of_box=int(prov["ghost_out_of_box"].fillna(False).sum()),
    )
    if not np.isfinite(prov.loc[prov["drop_reason"].isna(), ["ghost_x", "ghost_y"]]).all().all():
        raise ValueError(
            "build_ghost_frames produced a non-finite ghost coordinate on a SCORED frame. "
            "Pitch control silently DROPS NaN-coordinate rows (_spearman.py dropna), so a "
            "NaN ghost would make the keeper vanish rather than error."
        )
    return cf, prov, report
```

Implement the four private helpers in the same module: `_apply_domain`, `_pin_defended_goal` (reuse `silly_kicks.tracking.defended_goal_x`), `_build_provenance` (write-back rule `x = gr_x` if the defended goal is at x=0 else `105 - gr_x`, `y` unchanged; `displacement_m = hypot(ghost - actual)`), and `_write_back` (copy frames, replace the defending-GK row's `x`/`y`, keep `vx`/`vy` per `params.ghost_keeps_actual_velocity`).

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_engine.py -v`
Expected: all PASS.

---

## Task 6: `provenance_to_targets` — the adapter the probe consumes

The provenance frame is **not** the targets frame. Four mismatches (spec §4.6): names, dropped rows must be excluded (`_validate_targets` requires finite coords on every row), keying is 3 keys with **one row per frame** while `compute_ghost_gk` writes **both teams' keepers**, and both flags must be non-null.

**Files:**
- Modify: `silly_kicks/gkdv/_engine.py`
- Test: `tests/gkdv/test_provenance_to_targets.py` (create)

- [ ] **Step 1: Write the failing contract tests**

```python
# tests/gkdv/test_provenance_to_targets.py
"""The provenance -> targets adapter contract (spec §4.6, review S7)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv import build_ghost_frames, provenance_to_targets
from silly_kicks.tracking._model_eval import _TARGET_COLUMNS, _validate_targets
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames


def _prov():
    frames = _make_ghost_gk_frames()
    _cf, prov, _r = build_ghost_frames(frames, model=_fitted_model()[0], home_team_id=1)
    return frames, prov


def test_emits_exactly_the_probe_contract_columns():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert set(_TARGET_COLUMNS) <= set(t.columns)


def test_passes_the_shipped_validator():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    _validate_targets(t)  # must not raise


def test_target_coords_finite_on_every_row():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert np.isfinite(t["target_x"]).all() and np.isfinite(t["target_y"]).all()


def test_exactly_one_row_per_frame_triple():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert not t.duplicated(subset=["game_id", "period_id", "frame_id"]).any()


def test_selects_the_DEFENDING_keeper_not_both():
    frames, prov = _prov()
    t = provenance_to_targets(prov, frames=frames, home_team_id=1)
    assert len(t) <= prov["frame_id"].nunique()


def test_naive_both_teams_passthrough_RAISES():
    """RED-FIRST: the dangerous mistake must fail loudly, not silently pick a keeper."""
    frames, prov = _prov()
    naive = prov.rename(columns={"ghost_x": "target_x", "ghost_y": "target_y"})
    naive = naive[list(_TARGET_COLUMNS)]
    if naive.duplicated(subset=["game_id", "period_id", "frame_id"]).any():
        with pytest.raises(ValueError, match="exactly one row per"):
            _validate_targets(naive)
    else:
        pytest.skip("fixture has a single GK team; uniqueness hazard not exercised here")
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_provenance_to_targets.py -v`
Expected: FAIL — `ImportError` / `NotImplementedError`.

- [ ] **Step 3: Implement the adapter**

```python
def provenance_to_targets(
    provenance: pd.DataFrame,
    *,
    frames: pd.DataFrame,
    home_team_id: int | str,
) -> pd.DataFrame:
    """Project the engine's provenance frame onto the probe's ``_TARGET_COLUMNS`` contract.

    The provenance frame and the targets frame are DIFFERENT views and this adapter is the
    only supported bridge (spec §4.6):

    * renames ``ghost_x``/``ghost_y`` -> ``target_x``/``target_y``;
    * DROPS dropped frames (the probe requires finite coordinates on every row);
    * selects the **defending-team** keeper via the pinned goal map, so the result carries
      exactly one row per ``(game_id, period_id, frame_id)`` -- ``compute_ghost_gk`` serves
      BOTH teams' keepers and a naive pass-through would either trip the probe's uniqueness
      check or silently select the wrong keeper;
    * guarantees both flags are non-null (``bool(NaN)`` is ``True``, which would silently
      shrink the probe's trusted stratum).

    Examples
    --------
    >>> targets = provenance_to_targets(prov, frames=frames, home_team_id=1)  # doctest: +SKIP
    """
    scored = provenance[provenance["drop_reason"].isna()].copy()
    defending = _select_defending_keeper(scored, frames=frames, home_team_id=home_team_id)
    out = defending.rename(columns={"ghost_x": "target_x", "ghost_y": "target_y"})
    out["ghost_clamped"] = out["ghost_clamped"].fillna(False).astype(bool)
    out["ghost_out_of_box"] = out["ghost_out_of_box"].fillna(False).astype(bool)
    out = out[list(_TARGET_COLUMNS)]
    if out.duplicated(subset=["game_id", "period_id", "frame_id"]).any():
        raise ValueError(
            "provenance_to_targets produced >1 row per (game_id, period_id, frame_id) -- "
            "the defending-keeper selection failed. Do NOT pass both teams' keepers."
        )
    return out.reset_index(drop=True)
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_provenance_to_targets.py -v`
Expected: all PASS.

---

## Task 7: `compute_threat_pc()` facade in `_cover_shadows.py`

The arm needs the xT-weighted Voronoi threat integral on a full frame — **not** `compute_blocking_score`, whose removal legs cancel. gkdv must not import the private `_voronoi_threat`.

**Files:**
- Modify: `silly_kicks/tracking/_cover_shadows.py`, `silly_kicks/tracking/__init__.py`
- Test: `tests/tracking/test_compute_threat_pc.py` (create)

- [ ] **Step 1: Write the failing test**

```python
# tests/tracking/test_compute_threat_pc.py
"""compute_threat_pc: the public facade over the xT-weighted Voronoi threat integral."""
from __future__ import annotations

import pytest

from silly_kicks.tracking import compute_threat_pc
from tests.tracking.test_cover_shadows import _frame, _fitted_xt  # existing helpers


def test_returns_a_finite_scalar():
    v = compute_threat_pc(_frame(), attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    assert isinstance(v, float)


def test_moving_the_keeper_changes_the_value():
    """NON-VACUITY: if this cannot move, the whole arm is dead."""
    base = _frame()
    moved = base.copy()
    gk = moved["is_goalkeeper"].astype(bool) & ~moved["is_ball"].astype(bool)
    moved.loc[gk, "x"] = moved.loc[gk, "x"] + 8.0
    a = compute_threat_pc(base, attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    b = compute_threat_pc(moved, attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    assert a != b, "threat_pc is insensitive to keeper position -- arm would be vacuous"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_compute_threat_pc.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the facade**

```python
def compute_threat_pc(
    frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    xt: ExpectedThreat,
    home_team_id: int | str,
    method: Literal["spearman", "fernandez_bornn", "voronoi"] = "spearman",
    params: PitchControlParams | None = None,
) -> float:
    """xT-weighted Voronoi pitch-control threat integral for ``frame``.

    The GK-sensitive term inside :func:`compute_blocking_score`. Its keeper sensitivity is
    inherited entirely from the pitch-control surface (``lambda_gk`` lives on
    ``SpearmanParams``), so ``method`` must stay ``"spearman"`` for a keeper-aware value --
    the other methods carry no GK term.

    Computes the surface DIRECTLY (never via ``PitchControlCache``): callers pass modified
    frames, and the cache key excludes player positions, so a cached lookup would silently
    return the canonical frame's surface.

    Examples
    --------
    >>> compute_threat_pc(frame, attacking_team_id=2, xt=xt, home_team_id=1)  # doctest: +SKIP
    0.0123
    """
    _validate_ltr(frame, caller="compute_threat_pc")
    surface = compute_pitch_control(frame, attacking_team_id, method=method, params=params)
    threat, _per_receiver = _voronoi_threat(
        surface, xt, frame, attacking_team_id=attacking_team_id, home_team_id=home_team_id
    )
    return float(threat)
```

Export it from `silly_kicks/tracking/__init__.py`.

- [ ] **Step 4: Run the test**

Run: `.venv/Scripts/python.exe -m pytest tests/tracking/test_compute_threat_pc.py tests/tracking/test_cover_shadows.py -v`
Expected: PASS (existing cover-shadow tests unchanged — the facade adds, it does not alter).

---

## Task 8: `_arms.py` — ΔGK-threat-suppression

**Files:**
- Modify: `silly_kicks/gkdv/_arms.py`
- Test: `tests/gkdv/test_arms.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/gkdv/test_arms.py
"""Physics arms: polarity, silent-zero guards, method pin (spec §5)."""
from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.gkdv import GkdvParams, delta_threat_suppression
from tests.tracking.test_cover_shadows import _frame, _fitted_xt


def test_deterrent_keeper_gives_a_NEGATIVE_delta():
    """Attacker-value units: actual - ghost. A deterrent actual keeper suppresses threat."""
    actual = _frame()
    ghost = actual.copy()
    gk = ghost["is_goalkeeper"].astype(bool) & ~ghost["is_ball"].astype(bool)
    ghost.loc[gk, "x"] = ghost.loc[gk, "x"] - 10.0  # ghost sits deeper = less suppression
    d = delta_threat_suppression(
        actual, ghost, attacking_team_id=2, xt=_fitted_xt(), home_team_id=1
    )
    assert d < 0, "a better-positioned actual keeper must score negative (= deterrent)"


def test_arm_refuses_a_pitch_control_cache():
    """The cache key excludes positions -> a shared cache would silently return Delta == 0."""
    import inspect

    sig = inspect.signature(delta_threat_suppression)
    assert "pitch_control_cache" not in sig.parameters


def test_non_spearman_method_is_rejected_AT_CONSTRUCTION():
    """A GK-blind method must be unrepresentable, not merely rejected at call time.

    lambda_gk exists ONLY on SpearmanParams, so any other method silently yields an arm
    that measures nothing about the keeper. GkdvParams.__post_init__ raises, so the bad
    configuration cannot be built and then passed around.
    """
    with pytest.raises(ValueError, match="GK-BLIND"):
        GkdvParams(pitch_control_method="voronoi")
    with pytest.raises(ValueError, match="GK-BLIND"):
        GkdvParams(pitch_control_method="fernandez_bornn")


def test_identical_frames_give_exactly_zero():
    f = _frame()
    d = delta_threat_suppression(f, f.copy(), attacking_team_id=2, xt=_fitted_xt(), home_team_id=1)
    assert d == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_arms.py -v`
Expected: FAIL — `NotImplementedError`.

- [ ] **Step 3: Implement**

```python
def delta_threat_suppression(
    actual_frame: pd.DataFrame,
    ghost_frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    xt,
    home_team_id: int | str,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> float:
    """Delta-GK-threat-suppression: ``threat_pc(actual) - threat_pc(ghost)``.

    Attacker-value units, so **negative = deterrent**: a deterrent actual keeper suppresses
    attacker threat, hence ``threat_pc(actual) < threat_pc(ghost)``.

    Deliberately accepts NO ``pitch_control_cache``: the cache key is
    ``(game_id, period_id, frame_id, team, method, params, ball_position, decompose)`` and
    excludes player positions, so a shared cache would serve the ghost frame the ACTUAL
    frame's surface and the delta would be silently zero.

    Examples
    --------
    >>> delta_threat_suppression(actual, ghost, attacking_team_id=2, xt=xt, home_team_id=1)  # doctest: +SKIP
    -0.0042
    """
    from silly_kicks.tracking import compute_threat_pc

    if params.pitch_control_method != "spearman":
        raise ValueError(
            f"pitch_control_method must be 'spearman' (got {params.pitch_control_method!r}): "
            "lambda_gk exists only on SpearmanParams, so other methods are GK-weight-blind "
            "and the ghost substitution would lose the keeper's control-rate multiplier."
        )
    kw = dict(attacking_team_id=attacking_team_id, xt=xt, home_team_id=home_team_id,
              method=params.pitch_control_method)
    return float(compute_threat_pc(actual_frame, **kw) - compute_threat_pc(ghost_frame, **kw))
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_arms.py -v`
Expected: all PASS.

---

## Task 9: `_arms.py` — ΔDAS with pinned attacking direction

`accessible-space` infers playing direction per period from an argmin over each team's mean x. **Moving the keeper perturbs that mean**, so the two legs can infer different directions and the delta stops being a counterfactual. `get_das` cannot pin direction (it hardcodes `infer_attacking_direction=True`); `get_individual_das` can.

**Files:**
- Modify: `silly_kicks/gkdv/_arms.py`
- Test: `tests/gkdv/test_arms.py` (extend)

- [ ] **Step 1: Write the failing test**

```python
def _port_frames() -> pd.DataFrame:
    """Minimal frames for the STUBBED path. Content is irrelevant -- the port is stubbed --
    so this is deliberately local rather than imported from a sibling test module."""
    return pd.DataFrame(
        {
            "game_id": ["100"] * 3,
            "period_id": [1] * 3,
            "frame_id": [1] * 3,
            "player_id": ["p1", "p2", None],
            "team_id": [1, 2, None],
            "x": [50.0, 55.0, 52.0],
            "y": [34.0, 30.0, 34.0],
            "is_ball": [False, False, True],
            "is_goalkeeper": [True, False, False],
        }
    )


def test_das_arm_passes_ONE_pinned_direction_to_BOTH_legs(monkeypatch):
    """STRUCTURAL primary -- and it runs on EVERY CI leg, with NO accessible-space.

    Every asserted fact here is about the CALLS, not the returns, so the stub returns a
    synthetic scalar instead of delegating to the real library. Delegation was the ONLY
    reason this guard previously needed `importorskip` -- and since `[das]` is installed on
    ZERO CI legs (D4), that made a guard for a declared live hazard skip everywhere.

    The property -- "one direction, computed on the FACTUAL frames, passed identically to
    both legs" -- is gkdv's own code, so testing it must not require the optional extra.
    Detects a revert to `get_das(infer_attacking_direction=True)`.
    """
    import silly_kicks.gkdv._das_port as port
    from silly_kicks.gkdv import delta_das

    pinned = pd.Series([1.0, 1.0, 1.0])
    monkeypatch.setattr(port, "pin_direction", lambda frames: pinned)

    calls: list[dict] = []

    def _stub_team_das(frames, *, attacking_team_id, direction_col):
        calls.append(
            {
                "col": direction_col,
                "values": tuple(frames[direction_col]) if direction_col in frames else None,
            }
        )
        return float(len(calls))  # synthetic -- NO delegation, NO library

    monkeypatch.setattr(port, "team_das", _stub_team_das)

    actual = _port_frames()
    ghost = actual.copy()
    gk = ghost["is_goalkeeper"].astype(bool) & ~ghost["is_ball"].astype(bool)
    ghost.loc[gk, "x"] = ghost.loc[gk, "x"] - 6.0
    delta_das(actual, ghost, attacking_team_id=2)

    assert len(calls) == 2, f"expected exactly two DAS legs, saw {len(calls)}"
    assert all(c["col"] == "attacking_direction" for c in calls), (
        "a leg ran WITHOUT a pinned direction column -- accessible-space would re-infer "
        "direction from team mean-x, which the ghost displacement perturbs"
    )
    assert calls[0]["values"] == calls[1]["values"], (
        "the two legs used DIFFERENT direction vectors -- the delta is not a counterfactual"
    )
    assert calls[0]["values"] == tuple(pinned), "the pinned FACTUAL direction was not used"


def test_unpinned_implementation_would_measurably_differ():
    """VALUE discriminator: prove the pin is not a no-op on this fixture.

    Deliberately EXTREME roster: a 4 m keeper move shifts an 11-player mean by only
    ~0.36 m, so a realistic fixture cannot flip the direction argmin. This fixture uses a
    small roster and a large displacement so an UNPINNED implementation demonstrably
    infers a different direction for the ghost leg. If this test ever goes green with the
    pin removed, the fixture has stopped discriminating and must be made more extreme.
    """
    pytest.importorskip("accessible_space")
    from accessible_space.interface import infer_playing_direction

    from tests.tracking.test_das import _frames as _das_frames

    actual = _das_frames()
    ghost = actual.copy()
    gk = ghost["is_goalkeeper"].astype(bool) & ~ghost["is_ball"].astype(bool)
    ghost.loc[gk, "x"] = ghost.loc[gk, "x"] - 60.0  # deliberately extreme

    def _infer(f):
        g = f.copy()
        # Mirrors _pin_attacking_direction's own ball-masking step. House idiom rather than
        # `== True`: equivalent on a plain bool column, safe on a nullable one.
        g.loc[g["is_ball"].astype(bool), "team_id"] = None
        return tuple(infer_playing_direction(
            g, team_col="team_id", period_col="period_id",
            team_in_possession_col="team_in_possession", x_col="x",
            ball_team=None, frame_col="frame_id",
        ).to_numpy())

    assert _infer(actual) != _infer(ghost), (
        "fixture no longer discriminates: an unpinned implementation infers the SAME "
        "direction for both legs here, so the pinning guard above proves nothing"
    )
```

> **P4 — a tautology was REPLACED here.** The earlier draft called
> `delta_das(actual, ghost, attacking_team_id=2)` **twice with identical arguments** and
> asserted equality. That asserts determinism, not direction pinning; nothing was mirrored
> despite the comment saying so, and it could not fail or detect a revert to
> `get_das(infer_attacking_direction=True)`.
>
> **`importorskip` now appears on the VALUE discriminator ONLY.** The structural guard above
> is library-free by construction (D4).

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_arms.py -k das -v`
Expected: FAIL — `ImportError` / `NotImplementedError`.

- [ ] **Step 3: Implement**

**First create the port — gkdv's ONLY contact with accessible-space.** This is what makes the structural guard library-free, and it is the correct hexagonal boundary regardless: gkdv depends on a seam, not on `_das`'s internals.

```python
# silly_kicks/gkdv/_das_port.py
"""The ONLY seam through which gkdv touches accessible-space.

Two functions, both resolved through this module so a test can stub them without the
optional ``[das]`` extra installed. Keeping the contact surface here (rather than importing
``_das`` privates from ``_arms.py``) means:

* the structural direction-pinning guard runs on EVERY CI leg, not only where
  ``accessible-space`` happens to be installed -- a guard for a live hazard that silently
  skips is not a guard;
* the gkdv -> tracking private-import allowlist has exactly ONE entry to justify, in one
  file, instead of exemptions scattered across the arms.

``_pin_attacking_direction`` itself calls ``_import_accessible_space()``, so stubbing only
the DAS scorer would NOT be sufficient -- both functions must sit behind this port.
"""
from __future__ import annotations

import pandas as pd

from silly_kicks.tracking._id_compat import ids_match


def pin_direction(frames: pd.DataFrame) -> pd.Series:
    """Per-row attacking direction inferred ONCE from the FACTUAL frames."""
    from silly_kicks.tracking._das import _pin_attacking_direction

    return _pin_attacking_direction(frames)["attacking_direction"]


def team_das(frames: pd.DataFrame, *, attacking_team_id, direction_col: str) -> float:
    """Sum per-player DAS for the attacking team under a PINNED direction column."""
    from silly_kicks.tracking._das import get_individual_das

    out = get_individual_das(frames, attacking_direction_col=direction_col)
    # House idiom (~df["is_ball"].astype(bool)), NOT `!= True`: on a nullable BooleanDtype or
    # object column `pd.NA != True` yields pd.NA, and a mask carrying pd.NA behaves differently
    # from a plain bool mask. A null-propagating compare behind a lint exemption is exactly the
    # defect class this PR exists to remove -- so no `# noqa: E712` here either.
    rows = out[~out["is_ball"].astype(bool) & ids_match(out["team_id"], attacking_team_id)]
    return float(rows["DAS"].dropna().sum())
```

Register the single allowlist exemption in `tests/gkdv/test_import_allowlist.py`:

```python
ALLOW_PRIVATE = {
    # TWO DIFFERENT KINDS OF EXEMPTION -- do not merge these comments.
    #
    # (1) CONFINED to _das_port.py. Both _pin_attacking_direction and get_individual_das are
    #     private tracking seams with no public equivalent, and confining them to that single
    #     module is what lets the structural direction guard run library-free. If this import
    #     appears in any OTHER gkdv module, that is a real violation -- route it via the port.
    "silly_kicks.tracking._das",
    #
    # (2) REPO-WIDE MANDATED SEAM, not confined. ADR-019 requires EVERY id comparison to go
    #     through _id_compat, so _engine.py, _arms.py and _metric.py all import it legitimately.
    #     Do NOT "fix" that by routing id comparisons through _das_port -- the mandate is the
    #     opposite of confinement.
    "silly_kicks.tracking._id_compat",
}
```

Then the arm — note it calls **through the port module** (`_das_port.pin_direction`), never by importing the names, so `monkeypatch.setattr(_das_port, ...)` intercepts:

```python
def delta_das(
    actual_frame: pd.DataFrame,
    ghost_frame: pd.DataFrame,
    *,
    attacking_team_id: int | str,
    params: GkdvParams = _DEFAULT_PARAMS,
) -> float:
    """Delta-DAS: attacking team's dangerous accessible space, ``actual - ghost``.

    Direction is pinned ONCE on the FACTUAL frames and the SAME pinned column is passed to
    BOTH legs -- accessible-space otherwise infers playing direction per period from
    ``groupby(team)[x].mean().idxmin()``, and moving the keeper shifts an 11-player mean by
    ~0.36 m per 4 m of displacement, so the legs could infer OPPOSITE directions and the
    delta would not be a counterfactual at all.

    Routed through ``get_individual_das`` (summed per team) because ``get_das`` hardcodes
    ``infer_attacking_direction=True`` and cannot accept a pin.

    NOTE (interpretation limit, spec §5): accessible-space receives no keeper flag
    (``_COLUMN_MAP`` has no ``is_goalkeeper``), so this arm measures the accessible-space
    consequence of relocating a GENERIC player. Keeper-specific physics are not modelled
    here -- unlike the threat arm, which weights the keeper by ``lambda_gk``.

    Examples
    --------
    >>> delta_das(actual, ghost, attacking_team_id=2)  # doctest: +SKIP
    -1.87
    """
    from . import _das_port  # module-attribute access at CALL time -> stubbable

    # ONE direction, inferred from the FACTUAL frames, applied to BOTH legs. Neither leg
    # may infer: accessible-space derives direction per period from
    # groupby(team)[x].mean().idxmin(), and the ghost displacement perturbs that mean.
    direction = _das_port.pin_direction(actual_frame).to_numpy()
    actual_pinned = actual_frame.copy()
    actual_pinned["attacking_direction"] = direction
    ghost_pinned = ghost_frame.copy()
    ghost_pinned["attacking_direction"] = direction

    kw = dict(attacking_team_id=attacking_team_id, direction_col="attacking_direction")
    return float(_das_port.team_das(actual_pinned, **kw) - _das_port.team_das(ghost_pinned, **kw))
```

> **Seam-resolution pin (do not "tidy" this).** `delta_das` must reach the port through the
> **module** (`from . import _das_port`, then `_das_port.team_das(...)`). If someone hoists
> this to `from ._das_port import team_das` at gkdv-import time, `monkeypatch.setattr(port,
> ...)` stops intercepting and the structural guard silently stops guarding. It would fail
> loudly here (`len(calls) == 2` breaks), so this is a correctness note rather than a
> silent-null — but pin it anyway.

> **ALLOWLIST NOTE:** this imports two private `tracking` symbols. Either promote
> `_pin_attacking_direction` to a public seam in Task 7's export block (**preferred** —
> it is exactly the kind of seam gkdv is supposed to consume), or add both to
> `ALLOW_PRIVATE` in `tests/gkdv/test_import_allowlist.py` **with the reason recorded**.
> Prefer promotion; the allowlist is for genuine exceptions.

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_arms.py -v`
Expected: all PASS (DAS tests skip cleanly without the `[das]` extra).

- [ ] **Step 5: Activate the DAS suites in CI (D4, Fix 2) — a one-token edit**

`pyproject.toml:51` already defines `das = ["accessible-space>=2.0,<3"]`; it is simply never installed. Edit `.github/workflows/ci.yml:64`:

```yaml
# before
      - run: pip install -e ".[kloppy,xgboost,test]"
# after
      - run: pip install -e ".[kloppy,xgboost,das,test]"
```

This is the ADR-023 **primary leg** (`ubuntu-3.12`), which already carries the primary-only pattern for expensive extras. One dependency activates **71 tests that have been dark since TF-28 shipped** — `test_das.py` (51), `test_das_offside.py` (7), `test_das_invariants.py` (7), `test_das_e2e.py` (6) — plus this PR's ΔDAS value discriminator.

```bash
.venv/Scripts/python.exe -m pytest tests/tracking/test_das.py tests/tracking/test_das_offside.py \
    tests/invariants/test_das_invariants.py -q
```
Expected: they RUN (not skip) locally, where `[das]` is installed. **If any of the 71 fail: STOP and surface it** — the actual error, file:line, and PROOF that it predates this change (stash, re-run, show the identical failure). **Do NOT decide yourself whether it belongs in this PR** — neither by fixing it silently nor by deferring it. Scope is the owner's decision, without exception. (D4 is settled as ACTIVATE; there is no decline branch. The structural guard needs no extra and runs everywhere regardless.)

---

## Task 10: `_metric.py` — per-keeper aggregation

**Files:**
- Modify: `silly_kicks/gkdv/_metric.py`
- Test: `tests/gkdv/test_metric.py` (create)

- [ ] **Step 1: Write the failing tests**

```python
# tests/gkdv/test_metric.py
"""Per-keeper aggregation: grain-agnostic, keyed on the frames-resolved GK player_id."""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.gkdv import aggregate_by_keeper


def _obs():
    return pd.DataFrame(
        {
            "player_id": [10, 10, 10, 20, 20],
            "game_id": ["g1", "g1", "g2", "g1", "g2"],
            "value": [-0.02, -0.01, 0.0, 0.03, 0.0],
        }
    )


def test_reports_mean_AND_median_and_nonzero_counts():
    out = aggregate_by_keeper(_obs(), value_col="value", min_nonzero=1)
    assert {"player_id", "mean", "median", "n", "n_nonzero", "n_games"} <= set(out.columns)
    row = out[out["player_id"] == 10].iloc[0]
    assert row["n"] == 3 and row["n_nonzero"] == 2


def test_min_nonzero_excludes_a_keeper_from_the_gate_surface():
    out = aggregate_by_keeper(_obs(), value_col="value", min_nonzero=2)
    assert 20 not in set(out.loc[out["gate_eligible"], "player_id"])


def test_input_is_not_mutated():
    df = _obs()
    before = df.copy(deep=True)
    aggregate_by_keeper(df, value_col="value", min_nonzero=1)
    pd.testing.assert_frame_equal(df, before)
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_metric.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def aggregate_by_keeper(
    observations: pd.DataFrame,
    *,
    value_col: str,
    min_nonzero: int = 20,
    min_games: int = 2,
) -> pd.DataFrame:
    """Aggregate observation-level arm values to per-keeper rows.

    Keyed on the frames-resolved GK ``player_id`` (spec §5): every frame row carries one,
    and the gold-mart ``player_key`` is deliberately NOT used -- it is an actions-grain
    lakehouse column, and a pure library module must not depend on a gold join.

    Grain-agnostic: any observation-level table with ``player_id``, ``game_id`` and
    ``value_col`` aggregates, so a future window-grain arm reuses this unchanged.

    Reports mean AND median (the registered gate reads the mean), plus per-keeper nonzero
    counts -- Delta-DAS is exactly 0 whenever the displacement moves no accessible-space
    boundary, and small displacements dominate.

    Examples
    --------
    >>> aggregate_by_keeper(obs, value_col="delta_das")  # doctest: +SKIP
    """
    src = observations
    grp = src.groupby("player_id", dropna=True)
    out = grp.agg(
        mean=(value_col, "mean"),
        median=(value_col, "median"),
        n=(value_col, "size"),
        n_games=("game_id", "nunique"),
    ).reset_index()
    nz = grp[value_col].apply(lambda s: int((s != 0).sum())).rename("n_nonzero").reset_index()
    out = out.merge(nz, on="player_id", how="left")
    out["gate_eligible"] = (out["n_nonzero"] >= min_nonzero) & (out["n_games"] >= min_games)
    return out
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_metric.py -v`
Expected: all PASS.

---

## Task 11: `_group_metrics.py` lift + re-point (**GATED on D1 and D2**)

> **STOP:** confirm decision **D1** (private vs public) and **D2** (merge order) before starting. If PR-S119 has not landed, rebase first — this task edits a file the parallel session's QA-bundle row also targets.

**Files:**
- Create: `silly_kicks/_group_metrics.py`, `tests/test_group_metrics.py`
- Modify: `scripts/xtgk_v2_keeper_discrimination.py`, `tests/xtgk/test_keeper_discrimination.py`

- [ ] **Step 1: Read the source of truth before moving anything**

```bash
grep -n "def icc_one_way" -A 40 scripts/xtgk_v2_keeper_discrimination.py
grep -n "def keeper_spread" -A 25 scripts/xtgk_v2_keeper_discrimination.py
grep -rn "icc_one_way\|keeper_spread" scripts/ tests/ silly_kicks/
```

- [ ] **Step 2: Create the module — move the two functions VERBATIM, renaming only `keeper_spread`**

`keeper_spread` becomes group-neutral (`group_spread`) at lift time, per the spec. Head the module with the precedent's docstring convention:

```python
# silly_kicks/_group_metrics.py
"""Domain-free grouped statistics (ICC, spread, permutation band, power simulation).

Lifted from ``scripts/xtgk_v2_keeper_discrimination.py`` at TF-19 PR-3 so the library --
not a script -- is the single source. **The reason is internal**: ``gkdv/`` is library code
and cannot import from ``scripts/``, and the published wheel ships only ``silly_kicks/``.
Mirrors the ``silly_kicks/_calibration_metrics.py`` precedent, whose docstring likewise
records an internal-consumers-only lift.

PRIVATE (decision D1). It carries no stability promise and **has no downstream consumer**:
the lakehouse confirmed on 2026-07-18 that it neither imports nor plans to import these
statistics -- per-keeper aggregates are dbt models there, and an ICC is a model-validation
statistic they consume as a verdict rather than compute. Crucially this is an OBSERVED
PATTERN, not a prediction: their statistical gates already live lakehouse-side in
``src/analytics/xg_calibration.py`` (per-provider discrimination gate, n-aware calibration
test), which is where an ICC would land if they ever wanted one. If that ever changes they
will say so; promoting this to ``silly_kicks/group_metrics.py`` + ``_PUBLIC_MODULE_FILES``
is then a deliberate, requested step -- not something to pre-empt.

TERMINOLOGY (a real cross-repo false-friend): the "discrimination" this module measures is
GROUP-VARIANCE discrimination (ICC -- does the metric separate keepers?). It is NOT the
CLASSIFIER discrimination (ROC-AUC) that the same word denotes ~25 times in the lakehouse
and in our own xG/PSxG gates. Do not conflate them when grepping either repo.
"""
```

> **Rationale correction, worth understanding rather than pattern-matching.** An earlier draft
> justified the library home with *"anything the lakehouse imports cannot live in `scripts/`"* —
> a **downstream-consumption** argument. The lakehouse checked and that premise is **false for
> this module**. Using it would have paired a consumption rationale with a private module, which
> is incoherent (a private module a consumer imports is a Hyrum contract with no guarantee).
> The internal reason is both true and sufficient; state only that one.

- [ ] **Step 3: Re-point both consumers (delete-and-depend — no copies left behind)**

In `scripts/xtgk_v2_keeper_discrimination.py`, delete the two function bodies and import them:

```python
from silly_kicks._group_metrics import group_spread, icc_one_way
```

Update `tests/xtgk/test_keeper_discrimination.py` imports identically.

- [ ] **Step 4: Run both suites — the lift must be behaviour-preserving**

```bash
.venv/Scripts/python.exe -m pytest tests/test_group_metrics.py tests/xtgk/test_keeper_discrimination.py -v
grep -rn "def icc_one_way\|def keeper_spread" scripts/    # expect: NO hits (no duplicate left)
```
Expected: PASS, and no duplicated definition remains.

---

## Task 12: `_validate.py` — §6.1–6.3 constants + Layer 4 behavioural anchoring

Layer 4 ships **here, in PR-3**, because it gates §6.1's ICC, which ships here (review N3(a)). §6.4 Layers 0–3 are PR-3b.

**Files:**
- Modify: `silly_kicks/gkdv/_validate.py`
- Test: `tests/gkdv/test_validate.py` (create)

- [ ] **Step 1: Write the failing tests — INCLUDING the both-sides band test**

```python
# tests/gkdv/test_validate.py
"""Registered validation constants + Layer 4 behavioural anchoring (spec §6)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.gkdv._validate import (
    ICC_ANCHORS,
    TERCILE_SEPARATION_M,
    behavioural_anchoring_verdict,
)


def test_registered_constants_are_frozen_values():
    assert ICC_ANCHORS == (0.015, 0.020, 0.026)
    assert TERCILE_SEPARATION_M == 0.5


def test_anchored_arm_passes():
    """Top and bottom terciles differ in mean signed depth by more than the threshold."""
    df = pd.DataFrame({"player_id": range(9),
                       "value": np.linspace(-0.05, 0.05, 9),
                       "signed_dx": np.linspace(-3.0, 3.0, 9)})
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "anchored"


def test_unanchored_arm_is_uninterpretable_not_evidence():
    """The PEV lesson: an arm not tracking a behaviour keepers vary is NOT evidence."""
    df = pd.DataFrame({"player_id": range(9),
                       "value": np.linspace(-0.05, 0.05, 9),
                       "signed_dx": np.zeros(9)})
    assert behavioural_anchoring_verdict(df, value_col="value", depth_col="signed_dx") == "uninterpretable"


def test_band_is_correctly_SIZED_not_just_detecting():
    """BOTH SIDES: a no-effect fixture must land INSIDE the permutation band.

    Without this, an anti-conservative band ships undetected -- the exact hazard the
    match-block null design exists to prevent.
    """
    from silly_kicks._group_metrics import icc_one_way

    rng = np.random.default_rng(42)
    df = pd.DataFrame({"player_id": rng.integers(0, 8, 400), "value": rng.normal(size=400)})
    icc = icc_one_way(df["value"].to_numpy(), df["player_id"].to_numpy())
    assert abs(icc) < 0.05, f"no-effect fixture produced ICC={icc:.4f} -- band is anti-conservative"
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_validate.py -v`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement**

```python
#: Pre-registered ICC anchor band (spec §1.3, measured 0.015-0.026 across cohorts).
ICC_ANCHORS: tuple[float, float, float] = (0.015, 0.020, 0.026)
#: Layer 4: minimum mean signed goal-relative depth separation between outer terciles.
TERCILE_SEPARATION_M: float = 0.5
#: Per-arm expected direction: both arms are attacker-value, so deterrent = negative.
EXPECTED_DIRECTION: dict[str, str] = {"delta_das": "negative", "delta_threat": "negative"}


def behavioural_anchoring_verdict(
    per_keeper: pd.DataFrame, *, value_col: str, depth_col: str
) -> str:
    """Layer 4: is the arm tracking a behaviour keepers actually VARY?

    Splits keepers into terciles by arm value; the top and bottom terciles must differ in
    mean signed goal-relative depth by at least ``TERCILE_SEPARATION_M``. If they do not,
    the arm's ICC is reported ``"uninterpretable"`` rather than as evidence.

    This is the guard the sibling possession-value metric's failure teaches: a metric can
    read flat because it rewards a behaviour good keepers do not perform, in which case an
    ICC near zero says nothing about keepers.

    Examples
    --------
    >>> behavioural_anchoring_verdict(per_keeper, value_col="delta_das", depth_col="signed_dx")  # doctest: +SKIP
    'anchored'
    """
    ranked = per_keeper.sort_values(value_col)
    k = max(1, len(ranked) // 3)
    lo = ranked.head(k)[depth_col].mean()
    hi = ranked.tail(k)[depth_col].mean()
    return "anchored" if abs(hi - lo) >= TERCILE_SEPARATION_M else "uninterpretable"
```

- [ ] **Step 4: Run the tests**

Run: `.venv/Scripts/python.exe -m pytest tests/gkdv/test_validate.py -v`
Expected: all PASS.

---

## Task 13: Cross-cutting CI gate registration

**Files:**
- Modify: `tests/test_public_api_examples.py`
- Verify: `tests/test_add_star_purity.py`, `tests/tracking/test_aggregator_column_liveness.py`

- [ ] **Step 1: Register gkdv in the Examples gate (causal/ precedent)**

Append to `_PUBLIC_MODULE_FILES` in `tests/test_public_api_examples.py`:

```python
    "silly_kicks/gkdv/__init__.py",
```

**What this gate does and does NOT do — decided explicitly, not assumed.** `_has_examples_section` (`tests/test_public_api_examples.py:91-97`) is a **substring test** for `"Examples\n---"` or `">>> "`, and **there is no doctest runner anywhere in this repo** (verified: `grep -rn doctest pyproject.toml setup.cfg pytest.ini tox.ini .github/workflows/*.yml` returns empty). So **Examples sections are illustrative-only; nothing executes them.**

That matters here because several gkdv docstrings state *contracts*: `serve_ghost_gk_positions` declares its output is goal-relative, `build_ghost_frames` declares the write-back rule, `delta_threat_suppression` declares the negative-equals-deterrent polarity, and `delta_das` declares the direction-pinning rule. **Per this gate's own contract those are prose, not verification.** We take the "point at the executable contract" option rather than adding a doctest runner:

- the goal-relative/write-back contract is executably gated by `tests/gkdv/test_engine.py::test_writeback_places_ghost_at_both_goal_ends` and `tests/gkdv/test_engine.py::test_out_of_box_flag_keys_on_GOAL_RELATIVE_x_and_survives_writeback`. **[PLAN SELF-INCONSISTENCY, fixed during Task 2 review]** an earlier revision also cited `tests/tracking/test_ghost_gk_serve_positions.py::test_out_of_box_is_computed_on_goal_relative_x_before_writeback` — but Task 2's own P5 note **deleted that test as a tautology**, so the citation pointed at a test that by design does not exist. A docstring citing a non-existent test id fails **silently as documentation**, which is the same class of defect this whole cycle keeps closing. Verify every cited test id resolves before writing it into a docstring:
  ```bash
  .venv/Scripts/python.exe -m pytest "<the exact node id>" --collect-only -q
  ```
- the polarity contract by `tests/gkdv/test_arms.py::test_deterrent_keeper_gives_a_NEGATIVE_delta`;
- the direction-pinning contract by `tests/gkdv/test_arms.py::test_das_arm_pins_direction_on_the_factual_frames_for_BOTH_legs`.

**Add a one-line pointer to the owning test in each of those four docstrings** rather than restating the contract in prose that no gate runs. (This programme has already shipped one wrong orientation contract in a docstring; do not repeat the shape.)

Only non-underscore modules are listed; `_engine`/`_arms`/`_metric`/`_validate` are private. If a frozen dataclass trips the gate, add it to `_SKIP_SYMBOLS` with a reason comment (the `OpportunityConfig` precedent):

```python
        "GkdvParams",  # frozen dataclass — fields are the documentation (OpportunityConfig precedent)
        "GkdvReport",  # frozen dataclass — fields are the documentation
```

- [ ] **Step 2: Confirm gkdv needs NO purity/liveness entry**

```bash
.venv/Scripts/python.exe -c "import silly_kicks.gkdv as g; print([n for n in g.__all__ if n.startswith('add_')])"
```
Expected: `[]` — gkdv ships **no `add_*` action-coupled aggregator**, so `PURITY_ENTRIES` and the liveness registry are untouched and the C4 count is unchanged by this cycle. Record that in the ADR.

- [ ] **Step 3: Run the gate suite**

```bash
.venv/Scripts/python.exe -m pytest tests/test_public_api_examples.py tests/test_add_star_purity.py -v
```
Expected: PASS.

---

## Task 14: Documentation (**C4 step gated on D2**)

**Files:** `NOTICE`, `TODO.md`, `CLAUDE.md`, `docs/superpowers/adrs/ADR-043-tf19-gkdv-v1.md`, `docs/c4/architecture.dsl`

- [ ] **Step 1: NOTICE — add the attribution entries**

Under "Mathematical / Methodological References" add Le et al. 2017 (ghosting) and the DEFCON-GNN comparator (arXiv:2512.10355), matching the surrounding entry format.

- [ ] **Step 2: Write the ADR**

Copy `docs/superpowers/adrs/ADR-TEMPLATE.md`. Record: the gkdv→tracking dependency rule; the two arms and their **asymmetric keeper modelling** (ΔDAS is keeper-blind, Δthreat carries `lambda_gk = 3.0`); the NaN-ghost silent-drop guard; the cache-refusal rule; D1's visibility decision; and that PR-3 ships **without** §6.4 Layers 0–3.

Four items the review rounds established that must reach the ADR as **decisions, not notes**:

1. **`method="spearman"` is a HARD API CONSTRAINT**, enforced in `GkdvParams.__post_init__` — a GK-blind configuration is unrepresentable, not merely discouraged. (`lambda_gk` exists only on `SpearmanParams`; and it is a *gain* applied after the influence field, so keeper position enters via TTI — which is why the S9 sensitivity leg is a gain sweep, not a mechanism probe.)
2. **The `_das.py` route choice**: the spec §5 amendment left two options open; we take `get_individual_das(..., attacking_direction_col=…)` summed per team, which needs **no `_das.py` edit** — the smaller blast radius. Record that as the reason.
3. **`[das]` was installed on ZERO CI legs** before this PR, leaving 71 TF-28 tests dark since it shipped. **This PR activated them on the ADR-023 primary leg** (D4, settled). Record the measurement, the one-token fix, and any pre-existing failures the activation surfaced — reported, not fixed here.
4. **`gkdv/_das_port.py` is the single narrow port** onto accessible-space, and the reason is testability of a live hazard, not tidiness: it lets the direction-pinning guard run on every leg rather than only where an optional extra happens to be installed.

5. **A reasoned NO-GATE decision (Task 3, velocity-state defending split).** Two of the three
   hardened id-compare sites are mutation-killable; the third is not, *by construction* — its
   `gk_team` scalar comes from the same frame it is compared against, so no input distinguishes
   `ids_match` from `==`. It is equally not lintable without flagging safe code (the
   same-source column-vs-scalar shape is syntactically identical to the unsafe cross-source
   one). The change is kept for CONSISTENCY — `TestExtractionRestriction`'s golden requires
   this block's identity rule to match the extractor's, which Task 3 changed. Record that it
   ships deliberately ungated, with the reason, rather than leaving a future reader to
   discover an untested line and "fix" it. Cross-reference ADR-027's finding that the
   name-heuristic lint is incomplete and the behavioural gate is the real backstop.
6. **`_ghost_gk.py`'s module PATH is pinned downstream** (lakehouse `exec_visibility.py:467-472`,
   ADR-044 drift guard), alongside `_xt_gk`, `_gk_completion` and `_gk_geometry`. This PR modifies
   `_ghost_gk.py` but deliberately does **not** rename or relocate it. Record the constraint so a
   future refactor of these four modules is treated as cross-repo coordination — a rename degrades
   their guard **silently**, with no import error.

6. **The private-consumer register ships in this PR** — see Task 14 Step 6. Record in the ADR that
   `docs/PRIVATE_CONSUMERS.md` was created here, why (the `_ghost_gk` path pin was found by asking
   a question, not by any standing mechanism), and that it is silly-kicks-side bookkeeping the
   lakehouse can correct rather than a negotiated contract.

**ADR follow-up to register (found by accident this cycle — worth making loud):** the
`importorskip` idiom silently converts *"optional dependency"* into *"optional testing"*, and
nothing in the suite reports it. That is how 71 DAS tests sat dark in CI from TF-28 until this
plan's D4 measurement. Register a cheap standing guard as a follow-up: either a test asserting
that **every `importorskip`'d module is installed on at least one CI leg**, or simply a CI step
that **prints the skip count** so a silent mass-skip becomes visible. Neither belongs in PR-3's
scope, but the gap should not go back to being invisible now that it has been found.

- [ ] **Step 3: TODO.md — the five updates the spec §9.1 registers**

1. Line ~50: replace the stale pre-retrain numbers (`0.00107`, "2.59×", "~10×") with `0.009697`, ratio ≈2.21×, and a ~10% relative miss; keep "the xS arm has never been measured" but re-scope to **PR-3b-gated**.
2. Lines ~58-68: **remove** the `test_xshot_gradientsports_e2e` known-failure entry — it passed on 2026-07-18 under 4.51.0. Record in the CHANGELOG that the pass was obtained under **local xgboost 2.1.4** while the artifact was produced under 3.2.0.
3. Header: `Current release` → the new version.
4. The "98 owner-tier SkillCorner matches" entry → "weights landed 4.51.0; `sc_extended` is HF-only and the Hub upload is the remaining owner action".
5. TF-19 entry: PR-3 shipped; PR-3b + owner run remain.

- [ ] **Step 4: CLAUDE.md — add the gkdv architecture bullet**

Add a `gkdv/` bullet to the Architecture list mirroring the `xtgk/` entry's density, and add the both-sides rule to Key conventions:

> **Every band needs a test from BOTH sides, and every counterfactual needs a non-vacuity assertion that it actually moved something.** Four silent-null defects in this codebase share one shape (a y-inversion, a fabricated grid origin, an identity-keyed pitch-control cache, and a mirrored external-provider event frame).

- [ ] **Step 5: Create `docs/PRIVATE_CONSUMERS.md` — the private-consumer register**

This cycle learned that `_ghost_gk`'s module path is load-bearing for a downstream guard **only because we happened to ask**. Nothing in either repo would have told us. Record what we now know so the next refactor has a known blast radius instead of a guess.

```markdown
# Private-module consumers

Downstream code that knowingly imports silly-kicks **private** (underscore-prefixed) modules or
pins their paths. Underscore modules carry **no stability promise** — this file exists so a
refactor can see its blast radius, not to turn these into supported API.

**Contributor rule:** before renaming, splitting or relocating any `silly_kicks/**/_*.py` listed
here, treat it as cross-repo coordination. A path pin in particular fails **silently** — no
`ImportError`, just a degraded consumer.

Verified 2026-07-18 with the luxury-lakehouse session. Line numbers are theirs and will drift;
the module/consumer pairing is the durable part.

| silly-kicks private | What is used | Consumer (luxury-lakehouse) | Why | Exit condition |
|---|---|---|---|---|
| `tracking/_xt_gk.py` | `XtGkParams`, **private fn** `_resolve_completion_for_frames` | `src/analytics/action_context/enrich.py:490` | No public seam exposes per-frame completion resolution | **Lakehouse migrates to xT-GK v2** (`xtgk.compute_xt_gk_v2`). v1 is already frozen and is removed ≥1 release after that migration. |
| `tracking/_xt_gk.py` | `XtGkReport` | `src/analytics/action_context/pipeline.py:98` | Aggregate QA type is not re-exported publicly | Same v2 migration; or promote the report type if v2 keeps an equivalent |
| `tracking/_ghost_gk.py`, `_xt_gk.py`, `_gk_completion.py`, `_gk_geometry.py` | **module PATHS as hardcoded strings** | `src/ingestion/exec_visibility.py:467-472` (their ADR-044 executor-env drift guard) | Needs stable module identities to detect executor-env drift | A public introspection surface for shipped-module identity, **or** an accepted standing pin coordinated on rename. **Highest-risk entry: degrades silently.** |
| `tracking/_id_compat.py` | `ids_match` | `src/tests/action_context/test_frame_orientation_golden.py:49` (test only) | ADR-019 id-comparison semantics have no public equivalent | Promote the `_id_compat` helpers to a public surface, or the consumer inlines the semantics |
| `tracking/_das.py` | `get_das` | Recorded in their `docs/superpowers/specs/2026-05-14-tracking-context-oom-bekkers-fix-design.md:225` | Needed `chunk_size`, absent from `add_das` at the time | **Stated by the consumer:** switch back once `add_das` exposes `chunk_size`. **VERIFY whether this already shipped and the coupling is retired** — if so, delete this row. |

**Not consumers** (checked, recorded so the question is not re-asked): `_calibration_metrics.py`
and `_group_metrics.py` have **no** downstream consumer — the lakehouse computes its statistical
gates lakehouse-side (`src/analytics/xg_calibration.py`) and consumes model-validation results as
verdicts, not as computations.
```

- [ ] **Verify the one uncertain row before committing.** The `_das.get_das` coupling carries a stated exit condition; check whether it has already been met, and delete the row if so rather than recording a stale entry:

```bash
grep -n "chunk_size" silly_kicks/tracking/features.py | grep -i "add_das" || \
  .venv/Scripts/python.exe -c "import inspect, silly_kicks.tracking as t; print('chunk_size' in inspect.signature(t.add_das).parameters)"
```

- [ ] **Add the pointer to `CLAUDE.md`** so a refactorer meets it without knowing to look:

> **Private modules can have downstream consumers.** Before renaming, splitting or relocating any
> `silly_kicks/**/_*.py`, check `docs/PRIVATE_CONSUMERS.md`. Path pins fail **silently** — a
> renamed module degrades a consumer's guard with no `ImportError`.

- [ ] **Step 6: C4 — re-derive, then TRIM (never append)**

```bash
.venv/Scripts/python.exe -c "import silly_kicks.tracking as t; print(len([x for x in t.__all__ if x.startswith('add_')])-1)"
```

Edit `docs/c4/architecture.dsl`: add the `gkdv` container (mirroring `xtgk`) and update the `tracking` description to the **measured** count. The tracking box has ~9 chars of headroom against the 200-char cap, so **delete prose to make room**. Then regenerate and verify:

```bash
.venv/Scripts/python.exe -m pytest tests/test_c4_dsl_description_cap.py -v
```
Expected: PASS.

**Cross-session merge protocol for the C4 box (agreed with the PR-S119 session; mirrors spec §9.1 so both sides hold the same agreement):**

- The parallel PR-S119 adds `add_off_ball_run_values` and takes the count 29→30 (net-zero characters). This PR adds a `gkdv` reference into the **9 characters of headroom** on a box measured at **191/200**; two other boxes are at exactly 200.
- **Re-derive the count from `tracking.__all__` — never trust a number written in either plan.**
- **Do not let the `tracking` description GROW.** Even a well-meant "improvement" to that prose consumes the headroom the other PR needs, and the second lander then has to delete the first lander's wording under a red gate.
- **`docs/c4/architecture.html` is a ~294 KB GENERATED artifact.** If both PRs regenerate it the conflict is **not textually resolvable**, and the house rule forbids hand-editing generated HTML. **Whoever lands second regenerates it AFTER the merge**, rather than resolving the conflict by hand.

Regenerate the HTML via the `mad-scientist-skills:c4` skill (Java 21 + `~/.claude/tools/`) — subject to the post-merge rule above.

---

## Task 15: Release — full verification, version bump, ONE commit

- [ ] **Step 1: Resolve the version and ADR number — AT THIS MOMENT, not earlier (D3)**

> **No session owns the next release number.** Do not write a version or ADR number into any
> file before this step, and do not infer one from what another session might take. The owner
> steers the number at commit time and asks whether the version has been updated
> **everywhere** — so treat Step 2's site list as a checklist to confirm, not a guess to make.

```bash
git fetch origin && git log --oneline origin/main -3   # has main moved since branching?
grep -n 'version = ' pyproject.toml
ls docs/superpowers/adrs/ | sort -V | tail -3
```

If `main` has moved, merge it in first (`git stash` → `git merge --ff-only origin/main` → `git stash pop`) and re-derive. Then rename the ADR file to the resolved number and fix its internal references.

- [ ] **Step 2: Bump BOTH version sites**

```bash
# pyproject.toml:7   version = "4.53.0"
# silly_kicks/__init__.py:7   __version__ = "4.53.0"
grep -rn '4\.51\.0' pyproject.toml silly_kicks/__init__.py   # confirm both found, then edit
```

- [ ] **Step 3: CHANGELOG entry**

Add a top entry matching the existing format, covering: the gkdv package; the three `_ghost_gk` serving-seam additions; the `compute_threat_pc` facade; the `_group_metrics` lift; the ghost-extractor ADR-019 hardening; **and the explicit statement that no default xfn list changed, so there is NO retrain trigger.**

- [ ] **Step 4: Full local verification — replicate the CI lint job, not just the venv**

```bash
.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q --benchmark-skip
.venv/Scripts/python.exe -m ruff check silly_kicks/ tests/ scripts/
.venv/Scripts/python.exe -m ruff format --check silly_kicks/ tests/ scripts/
uv run pyright silly_kicks/          # MUST be 0 errors
uv run pyright                       # full tree -- compare to BASELINE, not to zero (below)
```

**ruff scope**: use `silly_kicks/ tests/ scripts/` — CI's exact scope (`ci.yml:26`). An earlier
revision omitted `scripts/`, which let a lint break in a generator script through undetected.

**[CORRECTED 2026-07-18 — measured, do not re-derive] pyright expectations:**
- `uv run pyright --version` resolves **1.1.409, exactly CI's pin**. The old "uv is 1.1.410 and
  stricter" reasoning is **stale** — do not use a version difference to explain a mismatch.
- `uv run pyright silly_kicks/` → **0 errors**. This is the hard gate; source must be clean.
- Bare `uv run pyright` (config scope = `silly_kicks` + `tests` + `scripts`) reports **28 errors
  on a clean `ec543cc` tree, ALL inside `tests/`** (13 in `test_xt_gk.py`, rest in
  player-influence / gk-influence / nan-safety tests). **This is NOT main being red** —
  `ec543cc` shipped as 4.51.0 with green CI. They come from test files importing optional deps
  (kloppy / accessible-space / xgboost / databricks) that resolve differently under uv than
  under CI's `[test]` extra.
- **Therefore: compare the full-tree count to the BASELINE, never to zero.** To prove this PR
  contributes zero, stash the changes, re-run, and compare counts:
  ```bash
  git stash -u && uv run pyright 2>&1 | tail -1 && git stash pop
  ```
- The in-suite gate `test_pyright_clean_tracking_namespace` only scopes
  `silly_kicks/tracking/` + `atomic/tracking/`, so it cannot see test-file errors. It passing is
  necessary, not sufficient.

- [ ] **Step 5: Single commit (SENTINEL-GATED — request explicit approval first)**

Write the message to a temp file (never `-m` with apostrophes):

```bash
git add -A
git status --short          # review EVERY file before committing
git commit -F .git/COMMIT_GKDV.txt
```

Message shape (mirroring recent commits):

```
feat(gkdv): TF-19 PR-3 -- ghost-substitution engine + physics arms v1 -- silly-kicks 4.53.0 (ADR-043, PR-S120)
```

- [ ] **Step 6: Push, PR, tag, publish**

```bash
git push -u origin feat/tf19-gkdv-pr3
gh pr create --title "<same as commit subject>" --body-file .git/PR_GKDV.md
```
After CI is green: squash-merge with `--admin`, then tag `v4.53.0` and let `publish.yml` push to PyPI.

---

## Self-review

**Spec coverage.** §2.1 package layout → Task 4. §2.2 probe-core/causal promotions → already shipped (verified, no task). §2.3 C4/NOTICE → Task 14. §4.1-4.6 engine incl. the adapter → Tasks 5-6. §4.3 serving seams → Task 2. §4 ADR-019 hardening → Task 3. §5 both arms + silent-zero guards + `lambda_gk` registration → Tasks 7-9. §5 aggregation → Task 10. §6.1 ICC lift → Task 11. §6.1-6.3 constants + Layer 4 → Task 12. §7 gates incl. both-sides bands → Tasks 12-13. §8 degrade verdict → **covered in Task 5's drop accounting**; the run-level arm degrade verdict is stubbed but its Layer-0 companion is PR-3b, as §8's amendment requires. §9 sequencing/TODO → Task 14. §6.4 Layers 0-3 → **deliberately absent (PR-3b)**.

**Placeholders.** None: every code step carries real code; every command has an expected result; the three `0XX`/`4.53.0` markers are resolved mechanically in Task 15 Step 1 under the no-reservation policy.

**Type consistency.** `GkdvParams`/`GkdvReport` (Task 4) are used unchanged in Tasks 5, 8, 9. `serve_ghost_gk_positions` returns `ghost_gr_x`/`ghost_gr_y` (Task 2), consumed as goal-relative and written back in Task 5, renamed to `target_x`/`target_y` only in Task 6. `compute_threat_pc` (Task 7) is keyword-only and called that way in Task 8. `aggregate_by_keeper` emits `player_id`/`mean`/`median`/`n`/`n_nonzero`/`n_games`/`gate_eligible` (Task 10), and Task 12 consumes `player_id` + a value column.

**Known deviations from the writing-plans skill, both deliberate house rules:** no worktree (`git switch -c` in the main checkout, per the no-worktrees rule), and **no per-task commits** — the repo requires one commit per PR behind the approval sentinel, so tasks end at test-green checkpoints and Task 15 makes the single commit.
