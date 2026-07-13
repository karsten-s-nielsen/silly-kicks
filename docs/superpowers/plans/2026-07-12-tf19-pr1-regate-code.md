# TF-19 PR-1 (Re-gate Code) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **NOTE for this repo:** merges are squash-`--admin`; commits are USER-GATED (chat
> approval; sentinel discipline — never create/offer the sentinel). Per-task steps
> therefore `git add` only; ONE commit at the end (Task 12) presented to the user for
> approval. Do not deviate.

**Goal:** Ship the re-gate code of the TF-19 cycle: the generalized GK-substitution
probe with the registered xS rule, the public `causal/` promotion with a builder that
can express the shot arm, the xS extractor ADR-019 hardening, chirality-fingerprint
emission, the §3.5 verdict function, and the cycle's docs — per the triple-reviewed
spec `docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md` (§9 PR-1).

**Architecture:** New `tracking/_model_eval.py` holds the model-agnostic probe core
(pure evaluator over pre-substituted inputs — the gkdv engine is PR-3 and `tracking/`
never imports `gkdv/`), the xS wrapper + registered constants, the `PROBE_WRAPPERS`
registry, and the re-gate verdict function; `_xcross_eval.py` keeps its public names as
a byte-equivalent home for the frozen xCross wrapper. `silly_kicks/_causal/` moves to
public `silly_kicks/causal/` with the FULL builder surface (incl. the
result-conditioned outcome axis) and cluster-aware placebo. A new
`tracking/_chirality.py` produces the behavioral chirality fingerprint that all three
model `save()` paths emit (load-enforcement is PR-2).

**Tech Stack:** Python 3.10–3.12, pandas, numpy, xgboost (via `[train]`), pytest,
ruff==0.15.7, pyright==1.1.409.

**Boundary — ruthless-efficiency:** shared training/tuning logic lives upstream in
`ruthless-efficiency` (floor `>=0.2.1`; API: `Candidate(id=, params=)`,
`result.best.candidate.params`, NO `IntRange`). This PR adds **zero** tuning logic to
silly-kicks: the train scripts are touched ONLY for chirality-fingerprint emission and
probe-sample provenance. Model-EVALUATION machinery (probes/verdicts) is
domain-specific and stays here (the `_xcross_eval.py` docstring records this split);
grouped validation statistics land in `silly_kicks/_group_metrics.py` in **PR-3**, not
here, and also do NOT go to ruthless (validation ≠ training/tuning). If the owner-run
paired-test re-run ever needs HPO-harness changes, they go upstream to ruthless first —
never forked here.

**Version/PR numbers:** next-free at release time (expected 4.47.0 / PR-S114; verify
against `origin/main` tags before bumping — no reservations).

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `tests/tracking/_probe_fixtures.py` | shared synthetic probe frames + planted models (extracted from `test_xcross_eval.py`) | Create |
| `tests/tracking/goldens/xcross_probe_golden.json` | PRE-refactor probe report on the committed fixture | Create (Task 1, from CURRENT code) |
| `silly_kicks/tracking/_model_eval.py` | generic substitution-delta producer, pure xS evaluator + registered constants, `PROBE_WRAPPERS`, `regate_verdict` | Create |
| `silly_kicks/tracking/_xcross_eval.py` | UNCHANGED public surface; internals re-pointed to `_model_eval` core (byte-equivalent) | Modify |
| `silly_kicks/tracking/_chirality.py` | canonical asymmetric probe frame + behavioral fingerprint | Create |
| `silly_kicks/tracking/_xcross_attempt.py` | `save()` metadata gains `chirality` block (~:466-481) | Modify |
| `silly_kicks/tracking/_xshot_occurrence.py` | canonical-id hardening (~:202-235) + `save()` chirality block (~:424-440) | Modify |
| `silly_kicks/tracking/_ghost_gk.py` | `save()`/`to_metadata` gains chirality block | Modify |
| `silly_kicks/causal/` (from `silly_kicks/_causal/`) | public promotion; full builder surface; result-conditioned outcome; cluster placebo | Move+Modify |
| `scripts/validate_xcross_causal.py` | import re-point `_causal`→`causal` | Modify |
| `scripts/train_xcross_attempt.py` | probe-sample provenance (provider+match ids) into `_probe_sample/meta.json` + `metrics.json` | Modify |
| `tests/tracking/test_model_eval.py` | core + xS evaluator + registry + verdict tests (T-groups below) | Create |
| `tests/tracking/test_probe_discriminating_power.py` | instrument meta-test (mixed planted model) | Create |
| `tests/tracking/test_xshot_id_dtype.py` | extractor dtype-invariance | Create |
| `tests/tracking/test_chirality_fingerprint.py` | emission + round-trip | Create |
| `tests/causal/test_builder_surface.py` | expressibility + result-axis + positive-control + cluster placebo | Create |
| `tests/causal/test_owner_run_refusals.py` | fail-loud refusal branches (held-out probe; control-conversion floor) | Create |
| `tests/causal/_fixtures.py` | NEW `simple_actions(specs)` helper (`actions()` untouched) | Modify |
| `scripts/validate_xshot_causal.py` | shot-arm runner (Task 6b): cluster placebo + control-Y floor | Create |
| `pyproject.toml` | per-file-ignores glob `_causal` → `causal` | Modify |
| `tests/causal/*` (existing) | import re-points only (`silly_kicks._causal` → `silly_kicks.causal`) | Modify |
| `tests/test_public_api_examples.py` | add `silly_kicks/causal/*` to `_PUBLIC_MODULE_FILES` | Modify |
| docs: ADR-037 (new), ADR-015, TODO.md, CLAUDE.md, NOTICE, CHANGELOG.md, `docs/c4/architecture.dsl` | Task 11 | Modify/Create |

Registered-constant summary (all in `_model_eval.py`, locked in this PR, before any
owner run — exact values below in Task 3):
`XS_PROBE_RATIO=2.0`, `XS_PROBE_DOSE_M=2.0`, `XS_PROBE_DOSE_LADDER=(2.0,3.0,4.0)`,
`XS_PROBE_MIN_BAND_N=100`, `XS_PROBE_MIN_STRATUM_N=50`, `XS_PROBE_PLACEBO_REPLICATES=20`,
`XS_PROBE_PLACEBO_BAND_PCT=95.0`,
`XS_PROBE_MAX_PLACEBO_ZERO_FRACTION=0.95`, `XS_PROBE_DOSE_RESPONSE_ALPHA=0.05`,
`XS_PROBE_DOSE_RESPONSE_PERMUTATIONS=999`, `XS_PROBE_MIN_GAME_N=10`,
`XS_PROBE_MIN_GAMES=8`.

---

## Task 0: Branch + green baseline

**Files:** none (verification only)

- [ ] **Step 1: Branch off current main**

```bash
git checkout main
git pull
git checkout -b pr-s114-tf19-regate-code
```

- [ ] **Step 2: Green baseline**

Run: `python -m pytest tests/tracking/test_xcross_eval.py tests/causal tests/tracking/test_xcross_attempt.py tests/tracking/test_xshot_occurrence.py tests/tracking/test_xshot_occurrence_integration.py tests/tracking/test_ghost_gk_integration.py tests/tracking/test_xcross_attempt_integration.py tests/test_public_api_examples.py -q -m "not e2e"`
Expected: all pass (every later oracle is in this list). The `-m "not e2e"` filter is
LOAD-BEARING on the owner box: the token + local pining data are present there, so
unfiltered runs execute the owner-gated real-data e2es (~40 min) — those are not part
of the standard gate (repo convention; Task 11 uses the same filter). If anything is
red, STOP and report — do not build on red.

---

## Task 1: Golden capture — BEFORE any refactor

The byte-equivalence oracle must come from PRE-refactor code, else the test proves the
new code equals itself (second-session review, Minors).

**Files:**
- Create: `tests/tracking/_probe_fixtures.py`
- Create: `tests/tracking/goldens/xcross_probe_golden.json`

- [ ] **Step 1: Extract the shared fixture module**

`tests/tracking/test_xcross_eval.py` already builds synthetic wide-area probe frames
(`_probe_frames`, ~line 42) and a trained toy model (`_fit_probe_model`, :127). Create
`tests/tracking/_probe_fixtures.py` by MOVING `_probe_frames` (and any helper it uses)
there, exporting `probe_frames()` (rename, drop the underscore), **with two ADDED
outfielders** (P4 — this must happen in THIS task, before the golden is captured: with
exactly 3 outfielders `min(n_random, k)` saturates and the golden is blind to the pick
count and draw order): add per frame an attacker `A3` at (92.0, 26.0, vx=0, vy=0) and a
defender `B2` at (98.0, 12.0, vx=0, vy=0) — off-axis, non-symmetric positions. The
existing `test_xcross_eval.py` probe tests run against the enriched fixture; if any of
them pins fixture-composition-sensitive values, update those pins in the same step
(they are pre-golden and fixture-relative, not frozen constants). Then add:

```python
"""Shared synthetic fixtures for the GK-substitution probe family (PR-1, ADR-037).

probe_frames()      -- two wide-area frames: carrier, defenders, GK, ball (goal at x=105).
planted_model(kind) -- deterministic 'models' exposing predict_proba(feats_df) -> np.ndarray.
    All kinds carry a WEAK DENSE term over every Def/Off distance slot so ANY outfielder
    move yields a small nonzero delta -- without it, moving an attacker changes only
    OffDist_* (which a gk_r+DefDist_0-only model ignores), every placebo replicate
    median is exactly 0, placebo_p95 = 0, and the fail-closed no_valid_placebo guard
    fires for a fixture reason (self-verify + 4.46.0-session review, convergent).
    'mixed'    dense + 3.0*(30-GK_r)/30      # GK per-meter sensitivity DECISIVELY > controls
               # (1.5/30 vs 0.8/20 was only 1.25x -- a pass would have ridden a
               #  displacement-projection accident, not the planted property)
    'gk_blind' dense only                    # zero GK dependence: every GK move is a no-op
    'chiral'   dense + 0.9*GK_theta          # SIGNED term -> negates under the y-mirror.
               # ONLY this kind can detect chirality: GK_r/DefDist are MAGNITUDES,
               # y-mirror-INVARIANT (GOAL_Y=34 sits ON the mirror axis). A fingerprint
               # test built on 'mixed' would pass while proving nothing.
"""

from typing import ClassVar

import numpy as np


class _Planted:
    carrier_params: ClassVar[dict] = {}  # RUF012: mutable class default needs ClassVar

    def __init__(self, kind: str):
        self.kind = kind

    def predict_proba(self, feats):
        gk_r = feats["GK_r"].to_numpy(float) if "GK_r" in feats.columns else feats["gk_r"].to_numpy(float)
        dense_cols = [c for c in feats.columns if c.startswith(("DefDist_", "OffDist_"))] or (
            ["dist_nearest_def", "dist_nearest_teammate"] if "dist_nearest_def" in feats.columns else []
        )
        dense = np.nansum([(20.0 - feats[c].to_numpy(float)) / 20.0 for c in dense_cols], axis=0)
        z = 0.05 + 0.1 * dense  # weak, dense: any outfielder move registers
        if self.kind == "mixed":
            z = z + 3.0 * (30.0 - gk_r) / 30.0
        elif self.kind == "chiral":
            th_col = "GK_theta" if "GK_theta" in feats.columns else "gk_theta"
            z = z + 0.9 * feats[th_col].to_numpy(float)
        return 1.0 / (1.0 + np.exp(-z))


def planted_model(kind: str) -> _Planted:
    return _Planted(kind)
```

Update `test_xcross_eval.py` to import `probe_frames` from the new module (delete its
local copy). Run: `python -m pytest tests/tracking/test_xcross_eval.py -q` — all pass.

- [ ] **Step 2: Generate the golden from CURRENT (pre-refactor) code**

The golden model is `planted_model("mixed")` — PURE NUMPY, bit-stable across every CI
leg (it exposes `carrier_params={}` so the probe accepts it). Do NOT use the trained-
XGBoost helper (`_fit_probe_model`, `test_xcross_eval.py:127` — note: there is NO
`_toy_model`): a per-leg re-fit compared at 1e-12 rides on xgboost cross-platform
reproducibility, the known-fragile golden class (goldens run on ALL legs, ADR-023).
Write and run a one-off (do not commit the script; commit its OUTPUT):

```bash
mkdir -p tests/tracking/goldens
python - <<'EOF'
import json
from tests.tracking._probe_fixtures import planted_model, probe_frames
from silly_kicks.tracking import _xcross_eval as ev
report = ev.gk_substitution_probe(planted_model("mixed"), probe_frames(), home_team_id="A")
pinned = {k: report[k] for k in (
    "gk_median_abs_delta","gk_mean_abs_delta","gk_p90_abs_delta",
    "nearest_def_median_abs_delta","random_band_median_abs_delta",
    "tf19_ready","tf19_reason","n_frames_used")}
with open("tests/tracking/goldens/xcross_probe_golden.json","w") as f:
    json.dump(pinned, f, indent=1, sort_keys=True)
print(pinned)
EOF
```

Float policy: compared with `pytest.approx(rel=1e-12)` — safe because the model is pure
numpy. NOTE: `tf19_reason` is a FREE-TEXT string built from branch logic
(`_xcross_eval.py:266-270`) — the Task 2 re-point must reproduce it byte-identically.

- [ ] **Step 3: Add the golden regression test to `test_xcross_eval.py`**

```python
def test_probe_report_matches_pre_refactor_golden():
    import json, pathlib

    from tests.tracking._probe_fixtures import planted_model, probe_frames

    golden = json.loads((pathlib.Path(__file__).parent / "goldens" / "xcross_probe_golden.json").read_text())
    report = ev.gk_substitution_probe(planted_model("mixed"), probe_frames(), home_team_id="A")
    for k, v in golden.items():
        if isinstance(v, float):
            assert report[k] == pytest.approx(v, rel=1e-12), k
        else:
            assert report[k] == v, k
```

Run: `python -m pytest tests/tracking/test_xcross_eval.py -q` — all pass (it trivially
passes NOW; its job is to stay green through Task 2).

- [ ] **Step 4: Stage**

```bash
git add tests/tracking/_probe_fixtures.py tests/tracking/goldens/xcross_probe_golden.json tests/tracking/test_xcross_eval.py
```

---

## Task 2: `_model_eval.py` core — generalize the substitution machinery

**Files:**
- Create: `silly_kicks/tracking/_model_eval.py`
- Modify: `silly_kicks/tracking/_xcross_eval.py` (internals only; public names/outputs byte-equivalent)
- Test: `tests/tracking/test_model_eval.py` (create)

Design (spec §2.2 + M4): the core is split into (a) a **delta producer**
`substitution_deltas(...)` that perturbs/substitutes actor rows and re-predicts — the
only piece that touches a model — and (b) **pure evaluators** over the resulting tidy
DataFrame. The xCross wrapper keeps its exact legacy behavior by calling the core with
the legacy panel; the golden from Task 1 pins byte-equivalence. `tracking/` never
imports `gkdv/` — ghost targets arrive as DATA (a per-frame targets DataFrame).

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_model_eval.py`:

```python
"""PR-1: generic substitution core + registry (ADR-037). xS evaluator tests are added in Task 3."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _model_eval as me
from tests.tracking._probe_fixtures import planted_model, probe_frames


def test_probe_wrappers_registry_lists_both_arms():
    assert set(me.PROBE_WRAPPERS) == {"xcross", "xs"}
    for name, entry in me.PROBE_WRAPPERS.items():
        assert callable(entry["wrapper"]), name
        assert isinstance(entry["rule_constants"], dict) and entry["rule_constants"], name


def test_registry_meta_every_wrapper_has_a_pinned_rule_test():
    # Meta-assertion (spec §7): each registry key must appear in a PINNED_RULES map here.
    PINNED_RULES = {
        "xcross": {"ratio": 2.0, "abs_floor": 0.01},
        "xs": {"ratio": 2.0, "dose_m": 2.0, "placebo_band_pct": 95.0},
    }
    assert set(PINNED_RULES) == set(me.PROBE_WRAPPERS)
    for name, pins in PINNED_RULES.items():
        rc = me.PROBE_WRAPPERS[name]["rule_constants"]
        for k, v in pins.items():
            assert rc[k] == v, (name, k)


def test_substitution_deltas_panel_mode_produces_tidy_rows():
    frames = probe_frames()
    out = me.substitution_deltas(
        planted_model("mixed"), frames, arm="xcross", mode="panel", seed=42
    )
    assert set(out.columns) >= {
        "game_id", "period_id", "frame_id", "actor_role", "displacement_m", "delta_p"
    }
    assert set(out["actor_role"].unique()) <= {"gk", "nearest_def", "placebo_out"}
    assert (out["delta_p"] >= 0).all()


def test_substitution_deltas_target_mode_moves_gk_to_supplied_position():
    frames = probe_frames()
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    targets = (
        gk[["game_id", "period_id", "frame_id"]]
        .drop_duplicates()
        .assign(target_x=90.0, target_y=34.0, ghost_clamped=False, ghost_out_of_box=False)
    )
    out = me.substitution_deltas(
        planted_model("mixed"), frames, arm="xs", mode="targets", targets=targets,
        n_placebo_replicates=3, seed=42,
    )
    gk_rows = out[out["actor_role"] == "gk"]
    assert len(gk_rows) == len(targets)
    assert (gk_rows["displacement_m"] > 0).all()  # the fixture GK is not at (90,34)
    # paired-vector controls: every control row's displacement equals its frame's GK displacement
    per_frame = out.pivot_table(index="frame_id", columns="actor_role", values="displacement_m", aggfunc="first")
    assert np.allclose(per_frame["nearest_def"], per_frame["gk"])
    reps = out[out["actor_role"] == "placebo_out"]["replicate"].nunique()
    assert reps == 3
```

- [ ] **Step 2: Run and watch it fail**

Run: `python -m pytest tests/tracking/test_model_eval.py -q`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.tracking._model_eval`.

- [ ] **Step 3: Implement `_model_eval.py`**

Create `silly_kicks/tracking/_model_eval.py`. The frame-eligibility walk, goal map,
carrier resolution, and per-actor perturbation are MOVED from `_xcross_eval.py`
(`gk_substitution_probe` :171-259 and `_abs_delta_for_player` :132-150) and
generalized; keep the moved logic line-for-line where possible:

```python
"""Model-agnostic GK-substitution probe core + registered xS rule + re-gate verdict (ADR-037).

Layering (M4): substitution_deltas() is the ONLY function here that touches a model; it
consumes ghost TARGETS AS DATA (a DataFrame) so tracking/ never imports gkdv/. Pure
evaluators (evaluate_xs_probe, regate_verdict) operate on the tidy deltas frame.
`_xcross_eval.py` remains the frozen xCross wrapper's home (byte-equivalent; golden-pinned).
See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# --- Registered xS probe rule (spec §3.1; locked BEFORE any owner run) ---------------------
XS_PROBE_RATIO = 2.0
XS_PROBE_DOSE_M = 2.0                      # gated band: |ghost - actual| >= 2 m, trusted stratum only
XS_PROBE_DOSE_LADDER = (2.0, 3.0, 4.0)     # reported, never gated beyond DOSE_M
XS_PROBE_MIN_BAND_N = 100                  # min frames in the gated band
XS_PROBE_MIN_STRATUM_N = 50                # min frames in the trusted (unclamped, in-box) stratum
XS_PROBE_PLACEBO_REPLICATES = 20           # R placebo replicates (paired-vector)
XS_PROBE_PLACEBO_BAND_PCT = 95.0
# NOTE: there is deliberately NO gated-band zero-fraction ceiling (spec §3.1(5) as
# amended): past the placebo gate, an all-zero GK band is a CLEAN FAIL, not
# unmeasurable. The fraction is reported, never gated.
XS_PROBE_MAX_PLACEBO_ZERO_FRACTION = 0.95  # non-degeneracy guard (prong 2)
XS_PROBE_DOSE_RESPONSE_ALPHA = 0.05        # prong 4: game-level sign-flip permutation p
XS_PROBE_DOSE_RESPONSE_PERMUTATIONS = 999
XS_PROBE_MIN_GAME_N = 10                   # frames a game needs to contribute a per-game rho
XS_PROBE_MIN_GAMES = 8                     # games needed for the dose test to be POWERED
                                           # (fixtures carry 10-12 games; real GS corpus 64)


def substitution_deltas(
    model,
    frames: pd.DataFrame,
    *,
    arm: str,                        # "xcross" | "xs" -- selects extractor + domain gate
    mode: str,                       # "panel" (legacy displacement panel) | "targets"
    targets: pd.DataFrame | None = None,  # mode="targets": one row per (game_id, period_id,
                                     # frame_id): target_x, target_y, ghost_clamped, ghost_out_of_box
    n_frames: int = 200,
    n_placebo_replicates: int = XS_PROBE_PLACEBO_REPLICATES,
    seed: int = 42,
    advance_m: float = 35.0,
) -> pd.DataFrame:
    """Tidy per-(frame, actor, move) |delta P|: columns game_id, period_id, frame_id,
    actor_role ('gk'|'nearest_def'|'placebo_out'), replicate, displacement_m, delta_p,
    ghost_clamped, ghost_out_of_box. mode='targets' moves the GK to the SUPPLIED target
    and displaces each control by the SAME per-frame vector (paired-vector controls,
    spec §3.1(2)); placebo replicates re-draw the outfielder, never the vector."""
    ...  # moved-and-generalized body; see below
```

Implementation notes for the body (the executor writes it by MOVING code):

1. Copy `gk_substitution_probe`'s eligibility walk (`_xcross_eval.py:171-206`) into a
   private `_eligible_groups(frames, model, arm, advance_m)`. For `arm="xcross"` it is
   verbatim (wide-area gate). For `arm="xs"` replace the `_in_wide_area(...)` call with
   the attacking-third gate: `abs(bx - goal_x) <= advance_m` (this is
   `_xshot_occurrence.py`'s `_ball_in_attacking_third` predicate — import it if public
   in that module, else re-state the one-line comparison with a comment naming it).
2. Copy `_abs_delta_for_player` (`:132-150`) into `_delta_for_move(model, grp, row_mask,
   moves, extract_fn, extract_kwargs)` where `moves` is a list of `(dx, dy)` — the
   panel becomes `moves`, and mode="targets" computes the GK's move as
   `(target_x - gk_x, target_y - gk_y)` and uses THAT single vector for the GK, the
   nearest defender, and each placebo outfielder (paired-vector; REGISTERED off-pitch
   policy: a control pushed off-pitch is scored anyway, never clamped — clamping would
   break the paired-vector guarantee; the off-pitch control fraction is reported).
   `extract_fn` is
   `extract_xcross_features` for arm="xcross" and `extract_xshot_features`
   (signature: `extract_xshot_features(grp, gk_team_id=..., goal_x=...)` — check the
   real signature at `_xshot_occurrence.py` and pass exactly its parameters) for arm="xs".
3. Placebo replicates: `rng.choice` outfielders WITHOUT replacement per replicate,
   seeded `default_rng(seed + replicate)`; `replicate` column 0..R-1 (`gk` and
   `nearest_def` rows carry replicate 0).
4. `displacement_m = math.hypot(dx, dy)`; `ghost_clamped`/`ghost_out_of_box` copied
   from `targets` (False for mode="panel").
5. Return the tidy frame; NO medians here — evaluators do statistics.

Then re-point `_xcross_eval.py`: replace the bodies of `_abs_delta_for_player` and the
sampling loop inside `gk_substitution_probe` with calls into
`_model_eval.substitution_deltas(model, frames, arm="xcross", mode="panel", ...)` +
median aggregation reproducing the EXACT legacy output dict (keys :271-284 unchanged).
`_tf19_ready` STAYS in `_xcross_eval.py` and reads the constants as MODULE attributes
of `_xcross_eval` (which RE-EXPORTS them from `_model_eval` per the registry fix
below): the monkeypatch-binding test at `test_xcross_eval.py:157-166` patches
`_xcross_eval`'s globals, so a from-import local inside `_tf19_ready` would stop
binding. The pins at :21-22 pass via the re-export.

**Registry + constants ownership (import-order trap, found by executed self-verify):**
registering "xcross" from `_xcross_eval.py`'s module bottom does NOT work — nothing in
`silly_kicks/` imports `_xcross_eval` (tests/scripts only), so a test importing only
`_model_eval` sees `PROBE_WRAPPERS == {"xs"}`; and importing `_xcross_eval` from
`_model_eval`'s bottom creates a two-way cycle that ImportErrors when `_xcross_eval`
loads first. Therefore: (a) `TF19_PROBE_RATIO` / `TF19_PROBE_ABS_FLOOR` MOVE to
`_model_eval.py` and `_xcross_eval.py` RE-EXPORTS them (`from
silly_kicks.tracking._model_eval import TF19_PROBE_ABS_FLOOR, TF19_PROBE_RATIO` — the
pins at `test_xcross_eval.py:21-22` still pass); (b) both registrations live in
`_model_eval.py`'s bottom, xcross via a LAZY shim:

```python
PROBE_WRAPPERS: dict[str, dict] = {}


def _register_wrapper(name: str, wrapper, rule_constants: dict) -> None:
    PROBE_WRAPPERS[name] = {"wrapper": wrapper, "rule_constants": dict(rule_constants)}


def _xcross_wrapper(*args, **kwargs):
    from silly_kicks.tracking._xcross_eval import gk_substitution_probe  # lazy: no top-level cycle

    return gk_substitution_probe(*args, **kwargs)


_register_wrapper("xcross", _xcross_wrapper, {"ratio": TF19_PROBE_RATIO, "abs_floor": TF19_PROBE_ABS_FLOOR})
```

Byte-equivalence constraints on the re-point (golden-caught, but know the knobs):
(i) `gk_substitution_probe` keeps its FULL legacy signature including the
accepted-and-unused `actions=None` (the trainer passes `actions=pa` at
`train_xcross_attempt.py:384`); (ii) panel mode must preserve the legacy SINGLE-rng
draw order (`default_rng(seed)`: frame subsample :221 then per-frame outfielder picks
:248) — the per-replicate `default_rng(seed + replicate)` scheme is TARGETS-MODE-ONLY;
(iii) **`substitution_deltas` gains `n_random: int = 3` (panel-mode pick count) with
explicit wrapper forwarding** — and BEFORE the Task-1 golden is captured, the fixture
gains **two more outfielders** (a second A attacker + a second B defender, off-axis
positions): with exactly 3 outfielders, `min(n_random, k)` saturates and the golden is
set-invariant — blind to a wrong pick count AND to a changed draw order, the two knobs
it exists to pin (P4). The golden also pins `gk_mean_abs_delta` + `gk_p90_abs_delta`
(medians alone are robust to exactly the row-duplication/drop drift a re-point can
introduce).

- [ ] **Step 4: Run — new tests green (except the xs-registry key until Task 3), golden + legacy green**

Run: `python -m pytest tests/tracking/test_model_eval.py::test_substitution_deltas_panel_mode_produces_tidy_rows tests/tracking/test_model_eval.py::test_substitution_deltas_target_mode_moves_gk_to_supplied_position tests/tracking/test_xcross_eval.py -q`
Expected: PASS, including `test_probe_report_matches_pre_refactor_golden` (byte-equivalence through the refactor).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/tracking/_model_eval.py silly_kicks/tracking/_xcross_eval.py tests/tracking/test_model_eval.py
```

---

## Task 3: The registered xS evaluator (pure) + xS wrapper

**Files:**
- Modify: `silly_kicks/tracking/_model_eval.py`
- Test: `tests/tracking/test_model_eval.py` (extend)

- [ ] **Step 1: Write the failing tests** (append to `test_model_eval.py`)

```python
def _deltas(n=300, dose=3.0, gk_scale=0.02, seed=0, zero_frac=0.0, trusted=True):
    """Synthetic tidy deltas: n frames, gk deltas ~ gk_scale, controls ~ gk_scale/4."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        d = dose * (0.5 + rng.random())
        gk_dp = 0.0 if rng.random() < zero_frac else gk_scale * d * (0.5 + rng.random())
        rows.append(dict(game_id=f"m{i % 10}", period_id=1, frame_id=i, actor_role="gk",
                         replicate=0, displacement_m=d, delta_p=gk_dp,
                         ghost_clamped=not trusted, ghost_out_of_box=False))
        rows.append(dict(game_id=f"m{i % 10}", period_id=1, frame_id=i, actor_role="nearest_def",
                         replicate=0, displacement_m=d, delta_p=gk_scale * d / 4,
                         ghost_clamped=False, ghost_out_of_box=False))
        for r in range(me.XS_PROBE_PLACEBO_REPLICATES):
            rows.append(dict(game_id=f"m{i % 10}", period_id=1, frame_id=i, actor_role="placebo_out",
                             replicate=r, displacement_m=d, delta_p=gk_scale * d / 8 * rng.random(),
                             ghost_clamped=False, ghost_out_of_box=False))
    return pd.DataFrame(rows)


def test_xs_evaluator_passes_on_strong_dose_responsive_signal():
    out = me.evaluate_xs_probe(_deltas(gk_scale=0.02))
    assert out["verdict"] == "pass"
    assert out["gated_band_n"] >= me.XS_PROBE_MIN_BAND_N
    assert out["dose_response_rho"] > 0


def test_xs_evaluator_unmeasurable_when_band_too_small():
    """n=60: the STRATUM floor (50) passes, so the BAND floor (100) is the SOLE trigger
    (P7 — n=20 tripped the stratum floor first and the band floor could regress to 0
    all-green)."""
    out = me.evaluate_xs_probe(_deltas(n=60, dose=1.0))  # dose<2 keeps banded n under 100
    assert out["trusted_stratum"] >= me.XS_PROBE_MIN_STRATUM_N
    assert out["gated_band_n"] < me.XS_PROBE_MIN_BAND_N
    assert out["verdict"] == "unmeasurable_at_dose"


def test_xs_evaluator_unmeasurable_when_trusted_stratum_empty():
    out = me.evaluate_xs_probe(_deltas(trusted=False))
    assert out["verdict"] == "unmeasurable_at_dose"


def test_xs_evaluator_no_valid_placebo_is_fail_closed():
    d = _deltas()
    d.loc[d["actor_role"] == "placebo_out", "delta_p"] = 0.0
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "no_valid_placebo"


def test_placebo_zero_concentration_prong_has_its_own_trigger():
    """P7: the MAX_PLACEBO_ZERO_FRACTION prong only decides when zeros CONCENTRATE in
    some replicates while others stay live — the all-zero test above hits p95<=0 first
    and this ceiling could regress unnoticed. 19 dead replicates + replicate-0 zeroed
    in 40% of frames: fraction 0.97 STRICTLY above the 0.95 ceiling (execution found
    the original 19-dead-only fixture lands EXACTLY on 19/20 == 0.95 against the
    strict > rule — a boundary fixture that could never fire), while replicate-0's
    median stays positive so p95 > 0 (the prong's distinguishing condition)."""
    d = _deltas()
    dead = d["actor_role"].eq("placebo_out") & d["replicate"].ne(0)
    d.loc[dead, "delta_p"] = 0.0
    rep0 = d.index[d["actor_role"].eq("placebo_out") & d["replicate"].eq(0)]
    d.loc[rep0[: int(0.4 * len(rep0))], "delta_p"] = 0.0  # 40% of the live replicate
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "no_valid_placebo"
    assert out["placebo_p95"] > 0
    assert out["placebo_zero_fraction"] > me.XS_PROBE_MAX_PLACEBO_ZERO_FRACTION


def test_xs_evaluator_flat_dose_response_overrides_band_pass():
    d = _deltas()
    # constant gk delta regardless of dose -> band median can pass, dose-response is flat
    d.loc[d["actor_role"] == "gk", "delta_p"] = 0.05
    out = me.evaluate_xs_probe(d)
    assert out["verdict"] == "band_pass_flat_dose_response"


def test_xs_evaluator_all_zero_gk_band_with_live_controls_is_a_clean_fail():
    """Review B1: zeros + LIVE controls = the keeper does not matter — a publishable
    FAIL, never 'unmeasurable'. The fraction stays reported (it makes the fail
    interpretable)."""
    out = me.evaluate_xs_probe(_deltas(zero_frac=0.9))
    assert out["verdict"] == "fail"
    assert out["gated_band_zero_fraction"] > 0.8
    assert out["placebo_p95"] > 0


def test_dose_response_null_is_centred_on_zero_under_no_signal():
    """Review B4 positive control for the NULL itself: with delta_p carrying no dose
    signal the p-value must not be small — else it is not a p-value."""
    d = _deltas()
    rng = np.random.default_rng(0)
    d.loc[d["actor_role"] == "gk", "delta_p"] = rng.random(int((d["actor_role"] == "gk").sum()))
    out = me.evaluate_xs_probe(d)
    assert out["dose_response_p"] > 0.05
    assert out["dose_response_n_games"] >= me.XS_PROBE_MIN_GAMES  # the test RAN — this is 'flat', not underpowered


def test_underpowered_dose_test_routes_band_pass_to_unmeasurable_never_flat():
    """Review N1(3): low power must not manufacture band_pass_flat_dose_response — but a
    band pass with an unrunnable dose test must not stand alone either. 3 games clears
    the band-n floor while staying under XS_PROBE_MIN_GAMES."""
    rows = _deltas(n=300)
    rows["game_id"] = "m" + (rows["frame_id"] % 3).astype(str)  # 3 games, ~100 gk frames each
    out = me.evaluate_xs_probe(rows)
    assert out["dose_state"] == "underpowered"
    assert out["verdict"] == "unmeasurable_at_dose"
    assert out["verdict"] != "band_pass_flat_dose_response"


def test_xs_evaluator_reports_ladder_and_ood_strata():
    out = me.evaluate_xs_probe(_deltas())
    assert set(out["dose_ladder"]) == set(me.XS_PROBE_DOSE_LADDER)
    assert "ood_stratum" in out and "trusted_stratum" in out
```

- [ ] **Step 2: Run and watch fail** — `python -m pytest tests/tracking/test_model_eval.py -q` → the new tests FAIL (`evaluate_xs_probe` missing).

- [ ] **Step 3: Implement `evaluate_xs_probe` in `_model_eval.py`**

```python
def _dose_response_clustered(gk: pd.DataFrame, *, seed: int = 42) -> tuple[float, float, int]:
    """Cluster-EXACT dose-response (review N1: replaces the equal-block subsample,
    whose min-truncation silently degenerated to a row permutation at m=1 and whose
    power collapse could manufacture a flat-dose veto): per-game Spearman rho, then a
    sign-flip permutation test on the GAME-level rhos. Raggedness is native — a
    400-frame game contributes a well-estimated rho, a 12-frame game a noisy one,
    NOTHING is truncated; the permutation unit IS the game. Returns (mean_rho, p,
    n_games_used).

    Conventions (registered): a game with constant delta_p gets rho = 0.0 — zero
    response variance is a MEASURED flat response, not a missing measurement; games
    with < XS_PROBE_MIN_GAME_N frames or constant displacement are skipped (cannot
    measure). Games iterate in sorted order so the seeded sign matrix pairs
    deterministically regardless of incoming row order."""
    from scipy.stats import spearmanr

    rhos = []
    for gid in sorted(gk["game_id"].astype(str).unique()):
        g = gk[gk["game_id"].astype(str) == gid]
        if len(g) < XS_PROBE_MIN_GAME_N or g["displacement_m"].nunique() < 2:
            continue
        if g["delta_p"].nunique() < 2:
            rhos.append(0.0)  # constant response == measured FLAT, not unmeasured
            continue
        r, _ = spearmanr(g["displacement_m"], g["delta_p"])
        if np.isfinite(r):
            rhos.append(float(r))
    arr = np.asarray(rhos, dtype=float)
    if len(arr) < XS_PROBE_MIN_GAMES:
        return float("nan"), 1.0, len(arr)
    obs = float(arr.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(XS_PROBE_DOSE_RESPONSE_PERMUTATIONS, len(arr)))
    null = (signs * arr).mean(axis=1)
    p = float((np.sum(null >= obs) + 1) / (XS_PROBE_DOSE_RESPONSE_PERMUTATIONS + 1))
    return obs, p, len(arr)


def evaluate_xs_probe(deltas: pd.DataFrame) -> dict:
    """PURE registered xS verdict over a substitution_deltas() frame (spec §3.1).
    Verdicts: 'pass' | 'fail' | 'unmeasurable_at_dose' | 'no_valid_placebo' |
    'band_pass_flat_dose_response'. Every prong is a registered constant; the
    ladder/unbanded/OOD/zero-fraction numbers are report-only."""
    gk_all = deltas[deltas["actor_role"] == "gk"]
    trusted = gk_all[~gk_all["ghost_clamped"].astype(bool) & ~gk_all["ghost_out_of_box"].astype(bool)]
    band = trusted[trusted["displacement_m"] >= XS_PROBE_DOSE_M]
    report: dict = {
        "rule": "xs-dose-banded-v1",
        "dose_ladder": {
            float(d): float(trusted.loc[trusted["displacement_m"] >= d, "delta_p"].median())
            if (trusted["displacement_m"] >= d).any() else float("nan")
            for d in XS_PROBE_DOSE_LADDER
        },
        "unbanded_median": float(gk_all["delta_p"].median()) if len(gk_all) else float("nan"),
        "trusted_stratum": int(len(trusted)),
        "ood_stratum": int(len(gk_all) - len(trusted)),
        "gated_band_n": int(len(band)),
        "gated_band_zero_fraction": float((band["delta_p"] == 0).mean()) if len(band) else float("nan"),
    }
    if len(trusted) < XS_PROBE_MIN_STRATUM_N or len(band) < XS_PROBE_MIN_BAND_N:
        report["verdict"] = "unmeasurable_at_dose"
        return report

    frame_keys = band[["game_id", "period_id", "frame_id"]]
    def _banded(role: str) -> pd.DataFrame:
        sub = deltas[deltas["actor_role"] == role]
        return sub.merge(frame_keys, on=["game_id", "period_id", "frame_id"])

    nd = _banded("nearest_def")
    placebo = _banded("placebo_out")
    # prong 2: placebo replicates of the SAME functional + non-degeneracy (fail-closed)
    rep_medians = placebo.groupby("replicate")["delta_p"].median()
    placebo_zero_fraction = float((placebo["delta_p"] == 0).mean()) if len(placebo) else 1.0
    report["placebo_replicate_medians"] = [float(v) for v in rep_medians]
    report["placebo_p95"] = float(np.percentile(rep_medians, XS_PROBE_PLACEBO_BAND_PCT)) if len(rep_medians) else float("nan")
    report["placebo_zero_fraction"] = placebo_zero_fraction
    nd_med = float(nd["delta_p"].median()) if len(nd) else float("nan")
    report["nearest_def_median"] = nd_med
    if (
        not np.isfinite(report["placebo_p95"]) or report["placebo_p95"] <= 0.0
        or placebo_zero_fraction > XS_PROBE_MAX_PLACEBO_ZERO_FRACTION
        or not (np.isfinite(nd_med) and nd_med > 0.0)  # M2 analog
    ):
        report["verdict"] = "no_valid_placebo"
        return report

    gk_med = float(band["delta_p"].median())
    report["gated_band_median"] = gk_med
    # Zero-inflation is a REPORTED DIAGNOSTIC, never an early return (review B1): zeros
    # have two causes and only the CONTROLS disambiguate — dead controls were already
    # caught fail-closed above as no_valid_placebo, so an all-zero GK band here can
    # only mean the keeper does not move the surface: a CLEAN FAIL (gk_med = 0 ->
    # band_pass False below), the cycle's expected, publishable outcome. A ceiling was
    # also outcome-inert for passes (zero-fraction > 0.5 forces median 0).

    # Cluster-exact dose-response over the trusted stratum (review B4 + N1): per-game
    # rho, sign-flip permutation across games. Same population, all data, no truncation.
    rho_obs, dose_p, n_games = _dose_response_clustered(trusted)
    report["dose_response_rho"] = rho_obs
    report["dose_response_p"] = dose_p
    report["dose_response_n_games"] = n_games
    # Three dose states (N1 point 3, generalized): 'ok' | 'flat' (test RAN, no positive
    # monotone response) | 'underpowered' (too few measurable games). Low power must not
    # manufacture the flat verdict — but it must not let a band pass stand alone either:
    # underpowered + band pass routes to the SUPPORT verdict, unmeasurable_at_dose.
    if n_games < XS_PROBE_MIN_GAMES:
        dose_state = "underpowered"
    elif np.isfinite(rho_obs) and rho_obs > 0 and dose_p < XS_PROBE_DOSE_RESPONSE_ALPHA:
        dose_state = "ok"
    else:
        dose_state = "flat"
    report["dose_state"] = dose_state

    # ratio vs max(control, placebo band): a deliberate strengthening over the spec's
    # nearest-defender-only prong (recorded in ADR-037); an explicit gk_med > p95
    # conjunct is redundant given ratio >= 2 and the p95 > 0 guard above.
    band_pass = gk_med >= XS_PROBE_RATIO * max(nd_med, report["placebo_p95"])
    if band_pass and dose_state == "ok":
        report["verdict"] = "pass"
    elif band_pass and dose_state == "flat":
        report["verdict"] = "band_pass_flat_dose_response"
    elif band_pass:  # dose_state == "underpowered": support verdict, never a manufactured flat
        report["verdict"] = "unmeasurable_at_dose"
    else:
        report["verdict"] = "fail"
    return report
```

*Execution amendment (2026-07-13, Task-10 review): the Task-2-registered off-pitch policy's reporting half is implemented as tidy column `moved_off_pitch` + report-only `off_pitch_control_fraction` in `evaluate_xs_probe` (control rows; NaN if column absent). Verdict logic untouched.*

NOTE on scipy: `scipy` is already a transitive dependency (sklearn) — import it inside
the functions (house style: heavy imports function-local). Register the wrapper:

```python
def xs_substitution_probe(model, frames, targets, *, seed: int = 42) -> dict:
    """The registered xS probe: produce deltas in targets-mode, evaluate the registered rule."""
    deltas = substitution_deltas(model, frames, arm="xs", mode="targets", targets=targets, seed=seed)
    out = evaluate_xs_probe(deltas)
    # Full triple, not bare frame_id: production frame ids restart per game/period, and
    # nunique() on frame_id alone would undercount to ~max-frames-per-game (exec review).
    gk = deltas[deltas["actor_role"] == "gk"]
    out["n_frames_used"] = len(gk[["game_id", "period_id", "frame_id"]].drop_duplicates())
    return out


_register_wrapper("xs", xs_substitution_probe, {
    "ratio": XS_PROBE_RATIO, "dose_m": XS_PROBE_DOSE_M,
    "min_band_n": XS_PROBE_MIN_BAND_N, "min_stratum_n": XS_PROBE_MIN_STRATUM_N,
    "placebo_replicates": XS_PROBE_PLACEBO_REPLICATES,
    "placebo_band_pct": XS_PROBE_PLACEBO_BAND_PCT,
    "max_placebo_zero_fraction": XS_PROBE_MAX_PLACEBO_ZERO_FRACTION,
    "dose_response_alpha": XS_PROBE_DOSE_RESPONSE_ALPHA,
    # Exec review: every constant that shapes the verdict belongs in the registry so an
    # introspected manifest is complete.
    "dose_ladder": XS_PROBE_DOSE_LADDER,
    "min_game_n": XS_PROBE_MIN_GAME_N,
    "min_games": XS_PROBE_MIN_GAMES,
    "dose_response_permutations": XS_PROBE_DOSE_RESPONSE_PERMUTATIONS,
})
```

Exec-review hardening (all non-registered; landed with Task 3's fix loop): loud raise on
any NaN `delta_p` at the top of `evaluate_xs_probe` (a NaN-poisoned GK band otherwise
fails OPEN into the pre-registered expected "fail" — the manufactured-outcome class);
`frame_keys.drop_duplicates()` semi-join in `_banded` + loud raise on duplicate GK band
keys (panel-mode-shaped input fan-out); `dose_state="not_run"` emitted on early-exit
reports (stable discriminator for the Task-5 verdict function and runners); explicit
`elif band_pass and dose_state == "underpowered"` + unreachable-else guard; a
merge-key/banding adversarial regression test (colliding frame ids across games +
out-of-band decoy deltas); per-game `astype(str)` hoisted; targets-mode per-frame
baseline cache in `_targets_deltas` (halves model calls; panel path untouched).

- [ ] **Step 4: Run** — `python -m pytest tests/tracking/test_model_eval.py -q` → all pass (registry test now finds both arms).

- [ ] **Step 5: Stage**

```bash
git add silly_kicks/tracking/_model_eval.py tests/tracking/test_model_eval.py
```

---

## Task 4: Instrument meta-test (discriminating power)

**Files:**
- Test: `tests/tracking/test_probe_discriminating_power.py` (create)

- [ ] **Step 1: Write the test** (it should PASS against Tasks 2-3 if the instrument works; if it fails, the INSTRUMENT is wrong — fix `_model_eval`, never the test):

```python
"""Spec §3.1 instrument validation: the probe must PASS a planted mixed-dependence
GK-responsive model and FAIL a GK-blind one — under the actual control construction.
A null from an instrument that has never detected a planted signal is uninterpretable."""

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _model_eval as me
from tests.tracking._probe_fixtures import planted_model, probe_frames


def _targets(frames, rng, spread=6.0):
    gk = frames[frames["is_goalkeeper"].astype(bool)]
    t = gk[["game_id", "period_id", "frame_id", "x", "y"]].drop_duplicates(
        subset=["game_id", "period_id", "frame_id"]
    )
    return t.assign(
        target_x=t["x"] - spread * (0.5 + rng.random(len(t))),
        target_y=t["y"] + rng.normal(scale=2.0, size=len(t)),
        ghost_clamped=False,
        ghost_out_of_box=False,
    ).drop(columns=["x", "y"])


def _run(kind, n_frames=150, seed=7):
    # replicate the fixture frames enough times (distinct game_ids) to clear MIN_BAND_N
    base = probe_frames()
    reps = []
    for i in range(n_frames):
        r = base.copy()
        r["game_id"] = f"m{i % 12}"
        r["frame_id"] = r["frame_id"] + 10 * i
        r["time_seconds"] = r["time_seconds"] + 10.0 * i  # carrier hysteresis: no duplicate clocks per game
        reps.append(r)
    frames = pd.concat(reps, ignore_index=True)
    rng = np.random.default_rng(seed)
    targets = _targets(frames, rng)
    out = me.xs_substitution_probe(planted_model(kind), frames, targets, seed=seed)
    # fixture-validity preconditions (M1): the verdict is meaningless if these fail
    assert out["gated_band_n"] >= me.XS_PROBE_MIN_BAND_N, "fixture too small for the registered rule"
    assert out.get("placebo_p95", 0) != 0 or out["verdict"] == "no_valid_placebo"
    return out


def test_mixed_dependence_planted_model_passes():
    out = _run("mixed")
    assert out["gated_band_zero_fraction"] < 1.0
    assert out["verdict"] == "pass", out


def test_gk_blind_model_is_a_clean_interpretable_fail():
    out = _run("gk_blind")
    assert out["verdict"] == "fail", out
    assert out["gated_band_zero_fraction"] == 1.0  # every GK move is a no-op...
    assert out["placebo_p95"] > 0                  # ...but the CONTROLS are live
    # together: a clean GK-insensitivity finding, NOT a degenerate measurement (B1/B2)
```

- [ ] **Step 2: Run** — `python -m pytest tests/tracking/test_probe_discriminating_power.py -q`
Expected: PASS. If `test_mixed_dependence_planted_model_passes` fails, debug the
instrument (paired-vector construction, banding, placebo) — this test is the evidence
the whole re-gate rests on; do NOT weaken it. Mark it `@pytest.mark.slow` ONLY if
runtime exceeds ~60 s (it is platform-invariant, ADR-023-eligible).

- [ ] **Step 3: Stage**

```bash
git add tests/tracking/test_probe_discriminating_power.py
```

---

## Task 5: Re-gate verdict function (§3.5 as code)

**Files:**
- Modify: `silly_kicks/tracking/_model_eval.py`
- Test: `tests/tracking/test_model_eval.py` (extend)

- [ ] **Step 1: Failing tests** (append; parametrized over EVERY §3.5 row):

```python
@pytest.mark.parametrize(
    "arm,probe,entangle,expected",
    [
        ("shot", "pass", "clears", "joins"),
        ("shot", "pass", "inside_band", "joins_with_caveat"),
        ("shot", "band_pass_flat_dose_response", "clears", "gated_flat_dose_response"),
        ("shot", "unmeasurable_at_dose", "clears", "unmeasurable_at_dose"),
        ("shot", "no_valid_placebo", "clears", "unmeasurable_at_dose"),
        ("shot", "pass", "degenerate", "joins_with_caveat"),
        ("shot", "fail", "degenerate", "gated_clean_fail"),
        ("shot", "instrument_invalid", "clears", "verdict_void"),
        ("shot", "fail", "clears", "gated_clean_fail"),
        ("cross", "pass", "clears", "joins"),
        ("cross", "pass", "inside_band", "joins_with_caveat"),
        ("cross", "fail", "inside_band", "gated_clean_fail"),
    ],
)
def test_regate_verdict_table(arm, probe, entangle, expected):
    assert me.regate_verdict(arm=arm, probe_verdict=probe, entanglement=entangle) == expected


def test_regate_verdict_rejects_unknown_inputs():
    with pytest.raises(ValueError):
        me.regate_verdict(arm="shot", probe_verdict="maybe", entanglement="clears")
```

- [ ] **Step 2: Run → FAIL** (`regate_verdict` missing).

- [ ] **Step 3: Implement** (in `_model_eval.py`):

```python
_PROBE_VERDICTS = frozenset({
    "pass", "fail", "band_pass_flat_dose_response", "unmeasurable_at_dose",
    "no_valid_placebo", "instrument_invalid",
})
# 'degenerate' (S6): the causal harness can genuinely return no-positivity/empty-overlap
# (it already reports claim_supported) — a real-world outcome, not a caller error.
_ENTANGLEMENT = frozenset({"clears", "inside_band", "degenerate"})


def regate_verdict(*, arm: str, probe_verdict: str, entanglement: str) -> str:
    """Spec §3.5 as a pure function. `entanglement` = the §3.3 GK-confounder-entanglement
    result (supportive context, NOT a causal deterrence estimate). Arms are independent;
    GKDV v1 (physics arms) ships regardless of every outcome here. An all-zero GK band
    with live controls arrives as probe_verdict='fail' (clean, publishable), never as
    an unmeasurable state (review B1)."""
    if arm not in ("shot", "cross") or probe_verdict not in _PROBE_VERDICTS or entanglement not in _ENTANGLEMENT:
        raise ValueError(f"regate_verdict: unknown input {(arm, probe_verdict, entanglement)!r}")
    if probe_verdict == "instrument_invalid":
        return "verdict_void"
    if probe_verdict in ("unmeasurable_at_dose", "no_valid_placebo"):
        return "unmeasurable_at_dose"
    if probe_verdict == "band_pass_flat_dose_response":
        return "gated_flat_dose_response"
    if probe_verdict == "fail":
        return "gated_clean_fail"
    return "joins" if entanglement == "clears" else "joins_with_caveat"
```

- [ ] **Step 4: Run** — `python -m pytest tests/tracking/test_model_eval.py -q` → all pass.

- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_model_eval.py tests/tracking/test_model_eval.py`

---

## Task 6: `causal/` promotion — full builder surface + result axis + cluster placebo

**Files:**
- Move: `silly_kicks/_causal/` → `silly_kicks/causal/` (`git mv`)
- Modify: `silly_kicks/causal/__init__.py`, `opportunities.py`, `matching.py`
- Modify: `scripts/validate_xcross_causal.py`, `tests/causal/*` (imports only)
- Modify: `tests/test_public_api_examples.py` (`_PUBLIC_MODULE_FILES`)
- Test: `tests/causal/test_builder_surface.py` (create)

- [ ] **Step 1: `git mv` + import re-points, keep everything green FIRST**

```bash
git mv silly_kicks/_causal silly_kicks/causal
grep -rl "silly_kicks._causal\|silly_kicks/_causal" silly_kicks scripts tests pyproject.toml docs/superpowers/adrs/ADR-015-causal-validation-port.md
```

Re-point every hit to `silly_kicks.causal` — INCLUDING `pyproject.toml`'s
per-file-ignores glob `"silly_kicks/_causal/**/*.py" = ["N803","N806"]` (~:192), the
only thing suppressing the X_base/Y/Z naming in `matching.py`; miss it and ruff floods
at Task 10 (P6a). Rewrite `causal/__init__.py`:

```python
"""Public causal-validation toolkit (ADR-015, promoted by TF-19/ADR-037 as the second
consumer). Pure numpy/sklearn propensity matching + a parameterized opportunity-row
builder. The xCross harness configuration is the default-constants path; the TF-19
shot arm is expressible purely as builder arguments (tested)."""

from silly_kicks.causal.matching import (
    GK_ABLATION_MIN_SHIFT,
    PLACEBO_BAND_PERCENTILE,
    CausalEstimate,
    abadie_imbens_se,
    estimate_att,
    estimate_atnt,
    fit_propensity,
    placebo_shift,
    propensity_match,
    smd_balance,
)
from silly_kicks.causal.opportunities import (
    SHOT_ARM_CONFOUNDERS,
    OpportunityConfig,
    build_opportunities,
    shot_arm_config,
    xcross_config,
)

__all__ = [
    "GK_ABLATION_MIN_SHIFT",
    "PLACEBO_BAND_PERCENTILE",
    "SHOT_ARM_CONFOUNDERS",
    "CausalEstimate",
    "OpportunityConfig",
    "abadie_imbens_se",
    "build_opportunities",
    "estimate_att",
    "estimate_atnt",
    "fit_propensity",
    "placebo_shift",
    "propensity_match",
    "shot_arm_config",
    "smd_balance",
    "xcross_config",
]
```

Run: `python -m pytest tests/causal -q` → all pass (move-only, no behavior change yet).

- [ ] **Step 2: Failing tests for the new surface**

Create `tests/causal/test_builder_surface.py`:

```python
"""PR-1 (ADR-037): the builder must express the §3.3 shot arm purely as arguments,
the outcome axis must be result-conditionable, and the placebo must be cluster-aware."""

import numpy as np
import pandas as pd
import pytest

import silly_kicks.spadl.config as spadlconfig
from silly_kicks.causal.matching import placebo_shift
from silly_kicks.causal.opportunities import (
    EXPOSURE_WINDOW_SECONDS,
    build_opportunities,
    shot_arm_config,
    xcross_config,
)
from tests.causal._fixtures import META, WIDE, actions, frames  # existing fixture helpers

SHOT = spadlconfig.actiontype_id["shot"]
SUCCESS = spadlconfig.result_id["success"]
FAIL = spadlconfig.result_id["fail"]


def test_xcross_default_config_reproduces_legacy_byte_identically():
    f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})
    a = actions([("cross", 10.1)])
    legacy = build_opportunities(f, a, home_team_id=5, model_metadata=META)
    explicit = build_opportunities(f, a, home_team_id=5, model_metadata=META, config=xcross_config(META))
    pd.testing.assert_frame_equal(legacy, explicit)


def test_shot_arm_outcome_is_the_anchor_inclusive_success_window():
    """P1 second re-registration: own-result-only made control Y ≡ 0 (controls have no
    anchor action) — the ATT was confounder-invariant and the entanglement gate dead."""
    cfg = shot_arm_config(META)
    assert cfg.outcome_result_ids == (SUCCESS,)
    assert cfg.outcome_window_seconds == 6.0  # == OUTCOME_WINDOW_SECONDS, the registered value
    assert cfg.outcome_window_anchor_inclusive is True
    assert cfg.extractor == "xs"


def test_saved_shot_yields_zero_scored_shot_yields_one():
    f = frames({10.0: 5, 10.2: 5}, {10.0: WIDE, 10.2: WIDE})
    cfg = shot_arm_config(META)
    saved = actions([("shot", 10.1, FAIL)])
    scored = actions([("shot", 10.1, SUCCESS)])
    y_saved = build_opportunities(f, saved, home_team_id=5, model_metadata=META, config=cfg)["Y"]
    y_scored = build_opportunities(f, scored, home_team_id=5, model_metadata=META, config=cfg)["Y"]
    assert int(y_saved.iloc[0]) == 0
    assert int(y_scored.iloc[0]) == 1


def test_cluster_placebo_reassigns_whole_clusters_under_unequal_sizes():
    """P5 property test: cluster-CONSTANT X_gk + UNEQUAL sizes — every destination
    cluster must receive exactly ONE source cluster's constant (a row-permuting
    implementation stamping 'cluster' FAILS here)."""
    rng = np.random.default_rng(0)
    sizes = [7, 13, 4, 21, 9]  # unequal, contiguous
    clusters = np.concatenate([np.full(s, i) for i, s in enumerate(sizes)])
    n = len(clusters)
    xb = rng.normal(size=(n, 2))
    xg = np.column_stack([clusters.astype(float), clusters.astype(float)])  # cluster-constant
    z = (rng.uniform(size=n) < 0.4).astype(int)
    y = rng.normal(size=n)
    out = placebo_shift(xb, xg, y, z, n_seeds=3, rng_seed=0, cluster_ids=clusters)
    assert out["permutation_unit"] == "cluster"
    # Re-run ONE permutation step exactly as the implementation does and assert the
    # property on the permuted xg it produced (expose the permuted block via a
    # returned sample or a small pure helper `_cluster_reassign(xg, clusters, rng)`
    # that placebo_shift uses and the test calls directly):
    from silly_kicks.causal.matching import _cluster_reassign

    permuted = _cluster_reassign(xg, clusters, np.random.default_rng(1))
    for d in np.unique(clusters):
        vals = np.unique(permuted[clusters == d, 0])
        assert len(vals) == 1  # exactly one source cluster's constant


def test_positive_control_ablation_detects_planted_gk_confounding():
    """Instrument validation (spec §3.3): a PLANTED GK->Z,Y confounder must produce a
    gk shift ABOVE the cluster placebo band. Only the null was pinned before."""
    rng = np.random.default_rng(1)
    n = 800
    xb = rng.normal(size=(n, 2))
    gk_signal = rng.normal(size=n)
    xg = np.column_stack([gk_signal, rng.normal(size=n)])
    z = (1 / (1 + np.exp(-(1.5 * gk_signal))) > rng.uniform(size=n)).astype(int)
    y = 0.8 * gk_signal + 0.3 * z + rng.normal(scale=0.5, size=n)
    clusters = np.repeat(np.arange(40), n // 40)
    out = placebo_shift(xb, xg, y, z, n_seeds=30, rng_seed=0, cluster_ids=clusters)
    from silly_kicks.causal.matching import _att_with_block
    real = abs(_att_with_block(xb, xg, y, z, seed=0) - out["base_att"])
    assert real > out["band_p95"], (real, out["band_p95"])
```

Run → FAIL (`shot_arm_config`, `xcross_config`, `config=`, `cluster_ids=` missing).
(P10: the existing `actions(rows)` fixture takes full 11-column positional rows — do
NOT shape-sniff it. Add a NEW helper to `tests/causal/_fixtures.py`:
`simple_actions(specs)` building the full 11-column frame from
`(type_name, t, result_id=success)` tuples; the tests above call `simple_actions`,
`actions()` and its callers stay untouched.)

- [ ] **Step 3: Implement the builder surface**

In `causal/opportunities.py`, add a frozen config + two constructors, and thread it:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class OpportunityConfig:
    """Full builder surface (ADR-037/M8): everything a consumer arm needs, as arguments."""
    treatment_type_names: tuple[str, ...]
    outcome_type_names: tuple[str, ...]
    outcome_result_ids: tuple[int, ...] | None = None   # None = type-only (legacy xCross)
    outcome_window_seconds: float = OUTCOME_WINDOW_SECONDS  # ALWAYS a window (R8: the
    # own-result 'None' form was structurally degenerate for controls and is banned)
    outcome_window_anchor_inclusive: bool = False        # False = legacy strict-post (xCross)
    exposure_window_seconds: float = EXPOSURE_WINDOW_SECONDS
    max_spell_seconds: float = MAX_SPELL_SECONDS         # THREADED into the spell loop
    confounders: tuple[str, ...] = tuple(PAPER_CONFOUNDERS)
    gk_block: tuple[str, ...] = tuple(GK_BLOCK)
    domain: str = "wide_area"       # "wide_area" | "attacking_third"
    extractor: str = "xcross"       # "xcross" | "xs" — threaded via _extract_row adapters


def xcross_config(model_metadata: dict) -> OpportunityConfig:
    return OpportunityConfig(
        treatment_type_names=tuple(model_metadata.get("cross_types", ("cross",))),
        outcome_type_names=("shot", "shot_freekick", "shot_penalty"),
    )


#: The §3.3 xS-side confounder list — a FRESH registered decision (xS has no _CONFOUNDERS
#: constant to reuse): the ball-geometry trio is the xS surface's core positional state.
SHOT_ARM_CONFOUNDERS = ("r", "theta", "speed", "openGoal", "DefDist_0", "DefDist_1")


def shot_arm_config(model_metadata: dict) -> OpportunityConfig:
    # Outcome (P1 re-registration): ANCHOR-INCLUSIVE success window — ts >= anchor,
    # result_id == success, 6 s. Y = the anchor shot's own goal OR a rebound goal for
    # treated spells, and a within-window goal for CONTROLS (anchor = entry). The
    # earlier own-result-only registration made control Y ≡ 0 by construction (controls
    # have no anchor action), which made the ATT confounder-INVARIANT and the
    # entanglement gate structurally dead. Anchor-inclusion also moots the
    # np.isclose time-scan concern (the window catches the anchor action).
    return OpportunityConfig(
        treatment_type_names=("shot", "shot_freekick", "shot_penalty"),
        outcome_type_names=("shot", "shot_freekick", "shot_penalty"),
        outcome_result_ids=(_spc.result_id["success"],),
        outcome_window_seconds=OUTCOME_WINDOW_SECONDS,
        outcome_window_anchor_inclusive=True,   # NEW config field: (ts >= anchor) vs legacy (ts > anchor)
        domain="attacking_third",
        extractor="xs",                          # P2: the extractor AXIS (see below)
        confounders=SHOT_ARM_CONFOUNDERS,
        gk_block=("GK_r", "GK_theta"),           # P2: xS GK names — xcross gk_* don't exist in xS features
    )
```

**P2 — the extractor axis.** `OpportunityConfig` gains `extractor:
Literal["xcross", "xs"] = "xcross"` and `outcome_window_anchor_inclusive: bool = False`
(legacy default preserves the byte-identical xCross path: strictly-post window).
`_row()` threads the extractor via per-extractor ADAPTER CLOSURES defined next to the
configs (the two real signatures differ — xcross takes `carrier_player_id` +
`score_differential`, xS does not):

```python
def _extract_row(cfg, grp, *, gk_team_id, goal_x, carrier_pid, sd):
    if cfg.extractor == "xs":
        from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

        return extract_xshot_features(grp, gk_team_id=gk_team_id, goal_x=goal_x).iloc[0]
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features

    return extract_xcross_features(
        grp, gk_team_id=gk_team_id, goal_x=goal_x, carrier_player_id=carrier_pid, score_differential=sd
    ).iloc[0]
```

`_label_outcome` reads the window bound from `cfg`: `lo = anchor if
cfg.outcome_window_anchor_inclusive else np.nextafter(anchor, np.inf)` and selects
`(ts >= lo) & (ts <= anchor + cfg.outcome_window_seconds)` (the `None`-window branch is
DELETED — the P1 registration replaced it).

**P1(b) — builder-level positive control (the wrong-layer lesson):** add to
`test_builder_surface.py` a synthetic-spell e2e that constructs frames+actions where
some CONTROL spells convert within their window, asserts the fixture-validity
precondition `opp.loc[opp["Z"] == 0, "Y"].var() > 0` (control Y must VARY or the ATT is
confounder-invariant and the instrument is dead), then runs the full
`build_opportunities(config=shot_arm_config(META))` → propensity → `placebo_shift`
chain on the BUILDER's output with a planted GK→(Z,Y) confounder and asserts the real
shift clears the band — the positive control at the layer the guard defends.
**Fixture mechanism (R9, load-bearing)**: a CONTROL spell can only convert via a
success-shot in `(spell_end, entry + 6]` — in-spell shots become treatment by
construction, and Y is deliberately not possession-clamped, which is the only door.
Construct control conversions as spell-ends-early + late goal: ball exits the domain at
entry + 1 s, the possessing team scores at entry + 4 s. A naive in-spell scoring shot
yields Z=1 and a fixture that cannot satisfy the `Y.var() > 0 among Z==0` precondition.

`build_opportunities` gains `config: OpportunityConfig | None = None`; when None it
constructs `xcross_config(model_metadata)` (legacy path byte-identical — the existing
positional flow simply reads from the config). `_label_outcome` becomes:

```python
def _label_outcome(actions, gid, per, team, anchor, cfg) -> int:
    type_ids = {_spc.actiontype_id[n] for n in cfg.outcome_type_names}
    sel = (
        ids_match(actions["game_id"], gid)
        & (actions["period_id"] == per)
        & ids_match(actions["team_id"], team)
        & actions["type_id"].isin(type_ids)
    )
    if cfg.outcome_result_ids is not None:
        sel &= actions["result_id"].isin(cfg.outcome_result_ids)
    ts = actions.loc[sel, "time_seconds"].to_numpy(dtype=float)
    # Anchor-inclusive (shot arm) vs legacy strictly-post (xCross) — P1: an own-result
    # 'None' window is banned (control Y would be structurally 0).
    in_window = (ts >= anchor) if cfg.outcome_window_anchor_inclusive else (ts > anchor)
    return int(bool((in_window & (ts <= anchor + cfg.outcome_window_seconds)).any()))
```

`_frame_domain_state` reads `cfg.domain`: `"attacking_third"` replaces the
`_in_wide_area(bx, by, goal_x, advance_m)` call with `abs(bx - goal_x) <= advance_m`.
`_row`/`_label_treatment` read windows + type names from `cfg`. Keep every existing
constant/name exported (the known-truth tests pin them).

In `causal/matching.py`, extend `placebo_shift` (additive kwarg, default preserves
legacy row-permutation exactly):

```python
def placebo_shift(X_base, X_gk, Y, Z, *, n_seeds: int, rng_seed: int, cluster_ids=None) -> dict:
```

with, inside the loop, when `cluster_ids is not None`, a PER-DESTINATION-CLUSTER
mapping (P5: concatenating permuted variable-size blocks straddles destination
boundaries and drifts the null back toward the row-i.i.d. permutation B3 rejected),
implemented as a PURE helper the test exercises directly:
`_cluster_reassign(X_gk, cluster_ids, rng) -> np.ndarray` — draw σ = a permutation of
the unique clusters; destination cluster d receives source cluster σ(d)'s X_gk rows,
recycled to d's size via `np.resize` — each destination cluster gets EXACTLY ONE
source cluster's values; `placebo_shift` calls it per seed. Document in the docstring that under
unequal sizes this is whole-cluster REASSIGNMENT-with-recycling, no longer a strict row
permutation (the null it draws is the cluster-exchangeable one, which is the point).
Add `"permutation_unit": "cluster" if cluster_ids is not None else "row"` to the
return. The `test_cluster_placebo_permutes_whole_clusters` test is REPLACED by the
property test P5 demands: cluster-CONSTANT X_gk values with UNEQUAL cluster sizes,
asserting post-permutation that every destination cluster carries exactly one source
cluster's constant (an implementation that row-permutes and stamps "cluster" fails).

- [ ] **Step 4: Run** — `python -m pytest tests/causal -q` → ALL pass (legacy known-truth
tests unmodified except imports; new surface tests green).

- [ ] **Step 5: Examples gate** — in `tests/test_public_api_examples.py`, add the three
`silly_kicks/causal/*.py` files to `_PUBLIC_MODULE_FILES` (same tuple style as existing
entries). The REAL work (P6b — the "matching functions already do" claim was wrong):
`abadie_imbens_se` GAINS an `Examples` section; `CausalEstimate` (no docstring today)
and the new `OpportunityConfig` are ClassDefs the gate walks — cover via field
docstrings or add both to the gate's `_SKIP_SYMBOLS` (DetectionResult precedent);
`build_opportunities`' `config=` param and both config constructors get Examples.
Run: `python -m pytest tests/test_public_api_examples.py -q` → pass.

- [ ] **Step 6: Stage**

```bash
git add silly_kicks/causal tests/causal scripts/validate_xcross_causal.py tests/test_public_api_examples.py
git rm -r --cached silly_kicks/_causal 2>/dev/null || true
```

---

## Task 6b: Owner-run measurement layer (P3 — the spec's M5/M6 had no task)

**Files:** Modify `scripts/train_xcross_attempt.py`, `scripts/validate_xcross_causal.py`;
Create `scripts/validate_xshot_causal.py`.

- [ ] **Step 1: Provider-controlled probe capture (spec M5).** `train_xcross_attempt.py`
gains `--probe-providers` (default `gradientsports` = the GATED cohort) applied INSIDE
the capture loop (:70-76): only matches whose provider is in the list are eligible for
`probe_keep`. A second capture list `--probe-comparison-providers` (default
`skillcorner`) persists to `_probe_sample_comparison/` — the reported-not-gated
same-population leg. Both `meta.json`s carry `probe_matches` + the provider filter used.
- [ ] **Step 2: Held-out gated statistic (spec M6).** Record each probe match's
training-fold membership into `meta.json` (`"in_training_folds": bool` per match,
computable from the fold assignment already built in `_cv_metrics`/paired-test code);
the gate-assembly block (:364-400) computes the gated probe on held-out matches only
when the paired test admits the probe provider to training, and FAILS LOUD (refuse to
emit `tf19_ready`) if no held-out probe match exists.
- [ ] **Step 3: Cluster placebo actually invoked.** `scripts/validate_xcross_causal.py`
passes match-level ids: `cluster_ids=opp["game_id"].to_numpy()` into `placebo_shift`,
reporting BOTH bands (`band_p95_row` legacy + `band_p95_cluster`) so the frozen 4.18.0
record stays comparable; the GATE reads the cluster band.
- [ ] **Step 4: Shot-arm runner.** Create `scripts/validate_xshot_causal.py` as a thin
clone of `validate_xcross_causal.py` over `build_opportunities(config=
shot_arm_config(META))` with the xS confounder/GK-block columns, cluster placebo, and
the same claim-gate structure; `--help` must run dep-free (args parsed before loader
imports, house pattern). **R10 — the control-conversion door is narrow in real data**
(a control converts only via a success-shot in `(spell_end, entry+6]`): the runner
reports the control-Y rate and count, and REFUSES the entanglement verdict below
`SHOT_ARM_MIN_CONTROL_CONVERSIONS = 30` (registered in the runner next to the other
claim-gate constants) — near-zero control conversions re-create P1's degeneracy as a
DATA condition.
- [ ] **Step 4b: Red-first tests for the refusal branches (R11).** The fail-loud paths
are exactly the branches that must demonstrably fire: extract the gate-assembly /
refusal logic into pure helpers (`_gated_probe_matches(meta: dict, admitted: bool) ->
list | Refusal`, `_entanglement_gate(opp: pd.DataFrame) -> verdict | Refusal`) and add
`tests/causal/test_owner_run_refusals.py` with planted `meta.json` dicts / opportunity
frames driving: (a) no held-out probe match when the provider is admitted → refusal
fires; (b) control conversions below the floor → refusal fires; (c) healthy fixtures →
verdicts emitted. Layer discipline: the tests construct each condition at the layer the
guard defends.
- [ ] **Step 5: Stage** — `git add scripts/train_xcross_attempt.py scripts/validate_xcross_causal.py scripts/validate_xshot_causal.py tests/causal/test_owner_run_refusals.py`

---

## Task 7: xS extractor canonical-id hardening (ADR-019)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (~:202-204, :229)
- Test: `tests/tracking/test_xshot_id_dtype.py` (create)

- [ ] **Step 1: Failing test**

```python
"""ADR-019: extract_xshot_features must be invariant to id dtype (Int64 vs str team ids).
The xCross extractor was canonicalized in 4.18.0; xS still used raw == (spec §4)."""

import pandas as pd
import pandas.testing as pdt

from silly_kicks.tracking._xshot_occurrence import extract_xshot_features
from tests.tracking._probe_fixtures import probe_frames


def test_extract_xshot_features_invariant_to_team_id_dtype():
    frames = probe_frames()
    grp = frames[frames["frame_id"] == frames["frame_id"].iloc[0]].reset_index(drop=True)
    grp_str = grp.copy()
    # numeric-vs-string worst case: '366.0' != '366' was the shipped GS bug class
    grp_num = grp.copy()
    grp_num["team_id"] = pd.array([1 if t == "A" else 2 for t in grp["team_id"]], dtype="Int64")
    f_str = extract_xshot_features(grp_str, gk_team_id="B", goal_x=105.0)
    f_num = extract_xshot_features(grp_num, gk_team_id=2, goal_x=105.0)
    pdt.assert_frame_equal(f_str, f_num)
    # cross-dtype: numeric frames, string gk_team_id — must not silently empty the GK mask
    f_cross = extract_xshot_features(grp_num, gk_team_id="2", goal_x=105.0)
    pdt.assert_frame_equal(f_cross, f_num)
```

(Adjust the call signature to `extract_xshot_features`'s REAL parameters — open
`_xshot_occurrence.py` and match exactly; the test intent is fixed: same physical frame,
three id-dtype presentations, byte-identical features.)

Run → FAIL on the cross-dtype case (raw `==` yields empty masks → NaN GK features).

- [ ] **Step 2: Implement** — in `_xshot_occurrence.py` replace the three raw compares
(the region read at :202-204 and :229 in review):

```python
from silly_kicks.tracking._id_compat import ids_match

is_gk_team = ids_match(players["team_id"], gk_team_id)
defending = players[is_gk_team & (~players_is_gk)]
attacking = players[~is_gk_team]
gk_rows = players[is_gk_team & players_is_gk]
```

(One computed mask, three consumers. NA semantics are DETERMINATE — verified:
`ids_match` returns a NON-nullable bool Series via `_as_bool`'s `fillna(False)`
(`_id_compat.py:95-97, 132-143`), so NaN-team rows are False in `is_gk_team` and the
plain complement `attacking = players[~is_gk_team]` puts them in `attacking` —
byte-matching the legacy `!=` behavior. No extra `fillna` needed.)

- [ ] **Step 3: Run** — the new test + `tests/tracking/test_xshot_occurrence.py` +
`tests/tracking/test_xshot_occurrence_integration.py` -q → ALL pass (matched-dtype
behavior byte-identical; only the previously-broken cross-dtype path changes).

- [ ] **Step 4: Stage** — `git add silly_kicks/tracking/_xshot_occurrence.py tests/tracking/test_xshot_id_dtype.py`

---

## Task 8: Chirality fingerprint emission + probe-sample provenance

**Files:**
- Create: `silly_kicks/tracking/_chirality.py`
- Modify: `silly_kicks/tracking/_xcross_attempt.py` (save() metadata dict, ~:466-481)
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (save() metadata dict, ~:424-440)
- Modify: `silly_kicks/tracking/_ghost_gk.py` (metadata.json writer)
- Modify: `scripts/train_xcross_attempt.py` (probe-sample provenance, ~:70-76, :241-247)
- Test: `tests/tracking/test_chirality_fingerprint.py` (create)

- [ ] **Step 1: Failing test**

```python
"""Spec §3.4/M2: chirality fingerprints are BEHAVIORAL (model outputs on a canonical
asymmetric probe frame), never self-declared strings; emitted at save() time (PR-1).
load()-enforcement is PR-2 — here we pin emission + round-trip determinism."""

import numpy as np

from silly_kicks.tracking._chirality import canonical_probe_frame, chirality_fingerprint
from tests.tracking._probe_fixtures import planted_model


def test_canonical_probe_frame_is_y_asymmetric_and_deterministic():
    f1, f2 = canonical_probe_frame(), canonical_probe_frame()
    import pandas.testing as pdt
    pdt.assert_frame_equal(f1, f2)
    mirrored = f1.copy()
    mirrored["y"] = 68.0 - mirrored["y"]
    assert not np.allclose(np.sort(f1["y"]), np.sort(mirrored["y"]))  # genuinely asymmetric


def _predict_on(model):
    def predict(frame):
        from silly_kicks.tracking._xshot_occurrence import extract_xshot_features
        feats = extract_xshot_features(frame, gk_team_id="B", goal_x=105.0)
        return model.predict_proba(feats)
    return predict


def test_fingerprint_changes_under_a_y_mirror_for_a_chiral_model():
    """MUST use the 'chiral' planted kind: 'mixed'/'gk_blind' consume only MAGNITUDES
    (GK_r, Def/OffDist), which are invariant under y->68-y (GOAL_Y=34 is ON the mirror
    axis) — a fingerprint test built on them passes while proving nothing (review B3)."""
    predict = _predict_on(planted_model("chiral"))
    fp = chirality_fingerprint(predict)
    mirrored_fp = chirality_fingerprint(lambda f: predict(f.assign(y=68.0 - f["y"])))
    assert fp["frame_sha256"] == mirrored_fp["frame_sha256"]  # same canonical frame
    assert not np.allclose(fp["outputs"], mirrored_fp["outputs"])  # chirality DETECTABLE


def test_fingerprint_is_blind_to_a_mirror_for_a_magnitude_only_model():
    """Guard-the-guard: if this ever FAILS, the fixture drifted (a signed term leaked
    into 'mixed') and the chirality test above has become vacuous."""
    predict = _predict_on(planted_model("mixed"))
    fp = chirality_fingerprint(predict)
    mirrored_fp = chirality_fingerprint(lambda f: predict(f.assign(y=68.0 - f["y"])))
    assert np.allclose(fp["outputs"], mirrored_fp["outputs"])
```

Run → FAIL (`_chirality` missing).

- [ ] **Step 2: Implement `_chirality.py`**

```python
"""Behavioral chirality fingerprint (ADR-037; enforcement in load() lands in PR-2).

A y-mirrored model serves inverted signed features silently — the 4.18.0-weights class
of bug. The fingerprint is the model's OUTPUTS on a fixed, deliberately y-ASYMMETRIC
synthetic frame: derived from behavior, so a mislabeled artifact cannot satisfy it.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

_CHIRALITY_VERSION = "chirality-probe-1"


def canonical_probe_frame() -> pd.DataFrame:
    """One synthetic frame, goal at x=105, all rows deliberately OFF the y=34 mirror axis."""
    rows = [
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="A",
             player_id="A1", x=80.0, y=20.0, vx=1.0, vy=0.5, is_ball=False, is_goalkeeper=False),
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="A",
             player_id="A2", x=88.0, y=45.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False),
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="B",
             player_id="B1", x=92.0, y=25.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False),
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="B",
             player_id="B2", x=95.0, y=50.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False),
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="B",
             player_id="BGK", x=103.0, y=30.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=True),
        dict(game_id="chir", period_id=1, frame_id=1, time_seconds=10.0, team_id="A",
             player_id="ball", x=82.0, y=21.0, vx=2.0, vy=1.0, is_ball=True, is_goalkeeper=False),
    ]
    return pd.DataFrame(rows)


def chirality_fingerprint(predict_on_frame) -> dict:
    """predict_on_frame: Callable[[pd.DataFrame], np.ndarray] — the model's own feature
    extraction + predict on the canonical frame. Returns a JSON-serializable dict."""
    frame = canonical_probe_frame()
    frame_sha = hashlib.sha256(
        json.dumps(frame.to_dict("records"), sort_keys=True, default=str).encode()
    ).hexdigest()
    outputs = np.asarray(predict_on_frame(frame), dtype=float).ravel()
    return {
        "version": _CHIRALITY_VERSION,
        "frame_sha256": frame_sha,
        "outputs": [round(float(v), 10) for v in outputs],
    }
```

- [ ] **Step 3: Wire emission into the three save() paths**

In `_xcross_attempt.py`'s `save()` metadata dict (the block review-verified at
~:466-481) add one key (build the predict callable from the model's OWN extractor on
the canonical frame — goal_x=105.0, gk_team_id="B", carrier "A1", score NaN):

```python
"chirality": _chirality_block(self),
```

with a module-level helper in the same file:

```python
def _chirality_block(model) -> dict:
    from silly_kicks.tracking._chirality import canonical_probe_frame, chirality_fingerprint

    def _predict(frame):
        feats = extract_xcross_features(
            frame, gk_team_id="B", goal_x=105.0, carrier_player_id="A1", score_differential=float("nan")
        )
        return model.predict_proba(feats)

    return chirality_fingerprint(_predict)
```

Mirror in `_xshot_occurrence.py` (its extractor + parameters) and in `_ghost_gk.py` —
DECIDED chain (there is NO `to_metadata`; the metadata dict is the inline literal in
`GhostGkModel.save()` at `_ghost_gk.py:1652-1671`, json.dump :1672-1674, self-hashing
:1677-1685): `feats = extract_ghost_gk_features(canonical_probe_frame(),
gk_team_id="B", goal_x=105.0, score_diff=0.0, phase=0, ball_carrier_team_id="A")` →
`outputs = model.predict_mean(feats)` (the predicted (x, y); sparse-frame NaN features
are fine — the booster treats NaN as missing, and determinism is all the fingerprint
needs). Adjacent latent gap to RECORD in ADR-037 (not fixed here — scope):
`extract_ghost_gk_features` carries the same raw `==`/`!=` team-id compares
(`_ghost_gk.py:488-490`) that Task 7 fixes in xS; the canonical frame uses matched
string dtypes so the fingerprint is unaffected. Extend
`test_chirality_fingerprint.py` with a round-trip test per model: fit the smallest
possible toy (reuse each model's existing smoke-test fit fixture), `save()` to tmp_path,
read `metadata.json`, assert the `chirality` block exists and recomputing it on the
loaded model reproduces `outputs` exactly.

- [ ] **Step 4: Probe-sample provenance in the trainer**

In `scripts/train_xcross_attempt.py`: the probe bundle is a 3-TUPLE at THREE sites that
must all grow together (self-verify) — `_extract`'s return annotation (:50) and return
value (:82-88), AND the cache-hit default `probe_bundle = ([], [], None)` (:221) —
otherwise the unpack at :241 raises ValueError. Where the sample is captured (:70-76),
also collect `(provider, match_id)` per kept match (the loader generator yields the
provider + match id — confirmed); where `_probe_sample/` is persisted (:241-247), write
`"probe_matches": [[provider, match_id], ...]` into `meta.json`, and at the metrics
gate block (~:364-400) read it back with `.get("probe_matches", [])` — a cache-hit run
against a PRE-plan `meta.json` must not KeyError — emitting `"probe_sample_matches"`.
Add a schema test to `test_chirality_fingerprint.py` asserting
a synthetic meta.json written by the (extracted, if needed) helper contains the key —
if the write is inline, extract `_write_probe_sample(dir, frames, actions, meta)` so it
is unit-testable without Databricks.

- [ ] **Step 5: Run** — `python -m pytest tests/tracking/test_chirality_fingerprint.py tests/tracking/test_xcross_attempt.py tests/tracking/test_xshot_occurrence.py tests/tracking/test_ghost_gk_integration.py -q` → all pass.

- [ ] **Step 6: Stage**

```bash
git add silly_kicks/tracking/_chirality.py silly_kicks/tracking/_xcross_attempt.py silly_kicks/tracking/_xshot_occurrence.py silly_kicks/tracking/_ghost_gk.py scripts/train_xcross_attempt.py tests/tracking/test_chirality_fingerprint.py
```

---

## Task 9: xCross re-run report enrichment (report-only diagnostics)

**Files:**
- Modify: `silly_kicks/tracking/_xcross_eval.py` (output dict only)
- Test: `tests/tracking/test_xcross_eval.py` (extend)

- [ ] **Step 1: Failing test** (append to `test_xcross_eval.py`):

```python
def test_probe_report_carries_report_only_diagnostics():
    from tests.tracking._probe_fixtures import planted_model, probe_frames

    report = ev.gk_substitution_probe(planted_model("mixed"), probe_frames(), home_team_id="A")
    assert "gk_zero_fraction" in report            # report-only; NOT part of _tf19_ready
    assert "random_band_zero_fraction" in report   # S5: post-B1 THE diagnostic separating
    assert "gk_median_abs_delta_at_2m" in report   # 'unmeasurable' from 'clean fail'
    assert "gk_median_abs_delta_at_4m" in report   # P9: REAL dose diagnostics, not prose
    # the FROZEN verdict fields are untouched — golden still green
```

- [ ] **Step 2: Implement** — in the wrapper's return dict add (computed from the Task-2
deltas frame it already has): `"gk_zero_fraction": float((gk_deltas == 0).mean())`,
`"random_band_zero_fraction": float((rb_deltas == 0).mean())` (S5 — both arms carry the
zeros-vs-live-controls diagnostic even though only xS gates on it), and the P9
report-only dose diagnostics from the frozen panel's two dose levels:
`"gk_median_abs_delta_at_2m"` / `"gk_median_abs_delta_at_4m"` (groupby the GK deltas'
`displacement_m`) and `"gk_dose_ratio_4m_over_2m"` (NaN-safe division). No prose-note
constant. `_tf19_ready` inputs unchanged.

- [ ] **Step 3: Run** — `python -m pytest tests/tracking/test_xcross_eval.py -q` → all pass
INCLUDING the golden (new keys are additive; the golden compares only pinned fields).

- [ ] **Step 4: Stage** — `git add silly_kicks/tracking/_xcross_eval.py tests/tracking/test_xcross_eval.py`

---

## Task 10: Docs — ADR-037, ADR-015, TODO, CLAUDE.md, NOTICE, C4

**Files:** `docs/superpowers/adrs/ADR-037-tf19-gkdv-regate-and-v1.md` (create),
`ADR-015-causal-validation-port.md`, `TODO.md`, `CLAUDE.md`, `NOTICE`,
`docs/c4/architecture.dsl` (+ regen), `CHANGELOG.md`.

- [ ] **Step 1: ADR-037.** Required content (each item one subsection; source = the spec,
which the ADR cites rather than duplicates): (1) the two-track decision + owner scope
choices; (2) the §1.2 corrected findings INCLUDING the retraction record and the
production chirality mis-serve; (3) the registered xS rule constants (copy the values
from `_model_eval.py`) + the instrument-validation discipline sentence: "no gate this
cycle is an instrument until it detects a planted signal under the actual clustering
structure and the actual control construction"; (4) the §3.5 verdict table (paste from
spec) + pointer to `regate_verdict`; (5) the §3.3 re-registration (anchor-INCLUSIVE 6 s
success window — SECOND re-registration, after the own-result form proved structurally
degenerate for controls; instrument relabeled GK-confounder entanglement); (6) the
gkdv→tracking dependency rule (public seams only; allowlist test lands PR-3); (7) the
B1 worked sign example — numeric: `threat_pc(actual)=0.30, threat_pc(ghost)=0.42 ⇒
Δ = −0.12 < 0 = deterrent; the Δ_blocking reduction equals +0.12 (defense-positive) —
the arm reports the NEGATION`; (8) ADR-025 interplay note for the future engine
(scoring-time views never persist); (9) chirality fingerprint decision (behavioral,
fail-closed in PR-2, legacy override); (10) ruthless boundary (this plan's header
paragraph, condensed); (11) the zero-inflation-prong REVERSAL (spec §3.1(5) as amended:
reported diagnostic, never a gate — controls disambiguate zeros; an all-zero GK band
with live controls is the clean, publishable fail), noting it reverses the earlier
ANDed-prong registration from the spec-review round, with reasoning; (12) recorded
latent gap: `extract_ghost_gk_features` raw `==`/`!=` team-id compares
(`_ghost_gk.py:488-490`) — the same ADR-019 class Task 7 fixes in xS; out of PR-1
scope, canonical-frame fingerprint unaffected; (13) the ratio prong's strengthening to
`max(nd_med, placebo_p95)` — deliberate amendment, with the redundant
`gk_med > placebo_p95` conjunct dropped (implied by ratio ≥ 2 with p95 > 0); (14)
`regate_verdict`'s `no_valid_placebo → unmeasurable_at_dose` conflation is DELIBERATE
(the probe report string preserves the control-construction-vs-support distinction;
follow-ups differ, both stay gated); (15) the paired-vector off-pitch policy (score,
never clamp; fraction reported).
- [ ] **Step 2: ADR-015** — Status: `implemented (private port, 4.18.0) → PROMOTED to
public silly_kicks/causal/ (PR-1, TF-19/ADR-037)`; one paragraph on the widened builder
surface + result-conditioned outcome axis.
- [ ] **Step 3: TODO.md** — TF-19 entry: replace the FULL three-name span "(Alisson,
Ter Stegen, Neuer should score strongly negative" with "(Alisson and Neuer should score
strongly negative — Ter Stegen played 0 WC2022 minutes; Onana is descriptive-only under
the ≥2-match rule —" (a single-token replacement produces duplicated-name nonsense);
update the Layer-3 status prose to point at
this cycle: "re-gate in flight per ADR-037: correctness retrain + first xS measurement;
frozen cross gate EXPECTED to hold". Do not touch the 4.46.0-corrected gate-record text.
Also add under Technical Debt (found during PR-1 Task 0, 2026-07-12, owner box):

```markdown
- **Known pre-existing failure: `test_xshot_gradientsports_e2e` (owner-gated real-data
  e2e) fails the Brier gate by 1.2e-4 on main/4.46.0** (brier 0.167758 vs base-rate
  0.167637; PR-AUC and log-loss gates PASS; loaders clean). Hypothesis: the bundled xS
  hyperparameters were HPO'd on y-defective geometry (ADR-031) and the e2e now feeds
  them corrected frames — a marginal calibration shortfall, the fresh-fit cousin of the
  PR-S114 chirality mis-serve. Expected fix: the TF-19 PR-2 retrain (fresh HPO on
  corrected geometry, ADR-037); re-run this e2e after PR-2 and remove this entry.
```
- [ ] **Step 4: CLAUDE.md** — add a PR-S114 bullet to the tracking section: `_model_eval`
(registered xS rule + `PROBE_WRAPPERS` + `regate_verdict`), `causal/` promotion (public,
full builder surface, cluster placebo), xS extractor ADR-019 hardening (VAEP-invariant
for matched dtypes), chirality fingerprint emission (enforcement PR-2). Note the
gkdv→tracking rule.
- [ ] **Step 5: NOTICE** — add "Le et al. 2017, Data-Driven Ghosting (ghosting
counterfactuals)" and "Kim et al. 2025, DEFCON-GNN, arXiv:2512.10355 (comparator)" under
Mathematical/Methodological References.
- [ ] **Step 6: C4** — edit `docs/c4/architecture.dsl`: append ADR-037 to the tracking
container's consumer-contracts clause. Aggregator COUNT stays 28 BY THE HOUSE
DEFINITION: `len([n for n in tracking.__all__ if n.startswith("add_")]) - 1`, excluding
the roster helper `add_gradientsports_player_ids` — the raw count is 29; assert the
DEFINED count, not the raw one (P8a).
Regen via the `mad-scientist-skills:c4` pipeline (Java 21 + jars in `~/.claude/tools/`).
- [ ] **Step 7: CHANGELOG.md** — new `## [4.47.0]` section (verify next-free with
`git ls-remote --tags origin` — the LOCAL clone does not carry release tags; local
`git tag` tops out far below origin): summarize Tasks 1-9 with the spec + ADR-037
pointers; flag NO retrain trigger from this PR alone (code only; weights land PR-2).
- [ ] **Step 8: Stage** — `git add docs TODO.md CLAUDE.md NOTICE CHANGELOG.md`

---

## Task 11: Full local gate

- [ ] **Step 1:** `python -m ruff check silly_kicks tests scripts` AND
`python -m ruff format --check silly_kicks tests scripts` → both clean.
- [ ] **Step 2:** `python -m pyright` (bare, whole repo) → 0 errors.
- [ ] **Step 3:** `python -m pytest tests/ -m "not e2e" -q --benchmark-skip` → all pass.
- [ ] **Step 4:** Version bump to the verified next-free (expected 4.47.0):
`pyproject.toml`, `silly_kicks/__init__.py`, `TODO.md` header, `CHANGELOG.md` header,
then `uv lock`. Stage: `git add pyproject.toml silly_kicks/__init__.py TODO.md CHANGELOG.md uv.lock`

---

## Task 12: /final-review + single gated commit + PR

- [ ] **Step 1:** Run `/final-review` (mandatory; includes the C4 check — count stays 28).
- [ ] **Step 2:** Re-run the Task 11 gate one last time — all clean.
- [ ] **Step 3:** Write the commit message to a temp file (NEVER `-m` with apostrophes)
and PRESENT the commit to the user for chat approval (sentinel discipline — do not
create/offer the sentinel; wait):

```
feat(tracking,causal): TF-19 re-gate code -- registered xS probe + public causal/ + chirality fingerprints -- silly-kicks 4.47.0 (ADR-037, PR-S114)

PR-1 of the TF-19 GKDV cycle (spec: docs/superpowers/specs/2026-07-12-tf19-gkdv-regate-and-v1-design.md,
triple-reviewed). No weights change here (retrains + load()-enforcement land in PR-2);
no retrain trigger from this PR alone.

- tracking/_model_eval.py: model-agnostic GK-substitution core (pure evaluator over
  pre-substituted inputs; tracking/ never imports gkdv/), the REGISTERED xS rule
  (dose-banded on the trusted stratum, paired-vector placebo over R replicates with a
  fail-closed non-degeneracy guard, cluster-exact dose-response prong ANDed into the
  verdict; the GK-band zero fraction is report-only — only the placebo-side
  non-degeneracy gates), PROBE_WRAPPERS registry, and regate_verdict (the ADR-037 decision
  table as a pure, parametrized-tested function). xCross wrapper byte-equivalent
  (pre-refactor golden) + report-only diagnostics.
- Instrument validation: a mixed-dependence planted model must PASS the probe and a
  GK-blind model must FAIL it, with fixture-validity preconditions asserted in-test.
- causal/ promoted public (ADR-015 'one move'): full builder surface (treatment/outcome
  type_ids + RESULT_IDS + windows + domain + confounders as arguments; the shot arm is
  expressible purely as builder args, tested), outcome re-registered as the anchor
  shot's OWN result (the windowed 'goal' was inexpressible and measured rebound goals),
  cluster-aware placebo_shift + a positive-control test (planted GK-confounding is
  DETECTED, not just the null pinned), Examples-gate registration.
- xS extractor ADR-019 hardening: canonical-id team compares (matched-dtype behavior
  byte-identical; the cross-dtype silent-empty-GK-mask path is fixed).
- Chirality fingerprints (behavioral, on a canonical y-asymmetric probe frame) emitted
  by xS/xCross/ghost-GK save(); probe-sample provenance (provider+match ids) recorded.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01TqpgWYzVXkRNY8tfTMooAQ
```

- [ ] **Step 4:** After user approval: commit, push `pr-s114-tf19-regate-code`, open the
PR with `--body-file` (body = the commit message + the standard footer), squash-`--admin`
merge on green CI, tag `v4.47.0` after main CI is green, confirm PyPI publish.

---

## Self-review notes

- **Spec coverage:** §2.2 → Tasks 2, 6; §3.1 → Tasks 3, 4; §3.2 → Task 9; §3.3 → Task 6;
  §3.4 (fingerprint emission + provenance) → Task 8; §3.5 → Task 5; §4 hardening → Task 7;
  §9 PR-1 docs → Task 10. NOT in PR-1 by design: gkdv package, `_group_metrics` lift,
  power sim, `serve_ghost_gk_positions()`/`compute_threat_pc()` seams (all PR-3);
  `load()` fail-closed enforcement + retrained weights + model-card fix (PR-2); owner
  runs (split per §9: retrains + xCross probe + causal after this PR; xS probe after PR-3).
- **M4 honored:** the xS probe's data-producer takes ghost TARGETS as a DataFrame; no
  gkdv import exists anywhere in `tracking/`.
- **Ruthless boundary:** train-script edits are provenance/fingerprint only (Task 8);
  zero tuning logic added; `_cv_score`/HPO surfaces untouched.
- **Rev 2 (post self-verify-by-execution + the 4.46.0 session's plan review):** the
  original known-risks (a)-(c) are RESOLVED — verified names/signatures are now inline:
  `_fit_probe_model` exists but the golden deliberately uses `planted_model("mixed")`
  (pure numpy, cross-leg stable); `extract_xshot_features(frame_data, *, gk_team_id,
  goal_x, feature_set="faithful")` and `extract_xcross_features(frame_data, *,
  gk_team_id, goal_x, carrier_player_id, feature_set="faithful",
  score_differential=nan)` verified; ragged blocks handled by the registered
  `_equal_block_subsample` (same population for rho_obs AND its null). Review-round
  fixes folded: B1 zero-inflation early-return DELETED (clean fail reportable;
  spec §3.1(5)/§3.5 amended); B2 gk_blind assertions tightened + planted models gained
  the weak dense term (without it, placebo replicates were all-zero and
  no_valid_placebo fired for a fixture reason); B3 'chiral' planted kind + a
  guard-the-guard mirror-blindness test; B4 permutation aligned (block-order, same
  population, null-centering positive control); registry import-cycle fixed (constants
  own-home in `_model_eval`, lazy xcross shim); trainer probe-bundle 3-tuple sites +
  `.get("probe_matches", [])` cache-compat; ghost fingerprint chain decided (inline
  save() dict :1652-1671, `extract_ghost_gk_features` + `predict_mean`); 'mixed' GK
  dominance made decisive (3.0/30 — was a projection accident at 1.5/30);
  targets-mode-only per-replicate RNG (panel keeps the legacy single-rng draw order);
  `actions=None` kept in the wrapper signature; S2 verified (placebo_shift returns
  base_att/band_p95/shifts, `_att_with_block` at matching.py:209); S6 'degenerate'
  entanglement state added.
- **Rev 5 (analysis-session re-audit, R1–R11) — SINGLE-SOURCING PASS COMPLETE: every
  accepted finding now lives in the task that produces its artifact; the list below is
  CHANGELOG ONLY (nothing here instructs the executor — if a bullet conflicts with a
  task body, the task body wins and the bullet is a stale record).** Rev-5 deltas:
  R1 Task-6 registration test rewritten to the anchor-inclusive window (the red-first
  path can no longer resurrect the dead outcome); R2 ADR-037 item (5) records the
  SECOND re-registration; R3 was already in the Task-1 capture tuple (pre-edit
  snapshot); R4 `_label_outcome` snippet is single-form (anchor_inclusive, no None
  branch); R5 `OpportunityConfig` snippet carries `extractor` +
  `outcome_window_anchor_inclusive`, `outcome_window_seconds` no longer admits None
  (R8), `max_spell_seconds` threaded; R6 vacuous fixtures replaced in-place
  (band-floor-only trigger at n=60; 19-dead/1-live placebo-concentration test; unused
  imports dropped; redundant conjunct dropped with the ADR note inline; cluster test
  replaced by the `_cluster_reassign` property test); R7 old text corrected in-place
  (Task 0 baseline widened; Examples-gate paragraph names its real work; Task 9 ships
  2m/4m dose diagnostics; TODO three-name span; C4 defined count; `git ls-remote
  --tags`; commit-message prong description); R9 control-conversion fixture mechanism
  stated at the test; R10 `SHOT_ARM_MIN_CONTROL_CONVERSIONS = 30` refusal in the
  shot-arm runner; R11 Task 6b Step 4b refusal tests + File Structure rows.
- **Rev 4 (analysis-session plan review, P1–P10 + minors) — changelog only, all
  entries since single-sourced into task bodies (rev 5):**
  - P6(a): Task 6's re-point sweep INCLUDES `pyproject.toml` — the per-file-ignores glob
    `"silly_kicks/_causal/**/*.py" = ["N803","N806"]` (~:192) re-points to
    `silly_kicks/causal/**`, or ruff floods at Task 11. P6(b): the Examples-gate step
    names its real work: `abadie_imbens_se` GETS an Examples section; `CausalEstimate`
    and `OpportunityConfig` are classes — cover via field docstrings or the gate's
    `_SKIP_SYMBOLS` (DetectionResult precedent); the claim "matching functions already
    do" is retracted.
  - P7: `test_xs_evaluator_unmeasurable_when_band_too_small` uses `_deltas(n=60)` (60
    clears the stratum floor of 50 so the BAND floor is the sole trigger); add a
    19-dead-replicates + 1-live placebo fixture asserting `no_valid_placebo` with
    `placebo_p95 > 0` (the zero-concentration prong's only reachable trigger).
  - P8(a): the C4 count is DEFINITIONAL — `len([n for n in tracking.__all__ if
    n.startswith("add_")]) - 1`, excluding the roster helper
    `add_gradientsports_player_ids`; raw 29 = counted 28. Task 10/12 use THIS check;
    the spec's §2.3 line gains the same parenthetical. P8(b): Task 0's baseline adds
    `test_xshot_occurrence_integration.py`, `test_ghost_gk_integration.py`,
    `test_xcross_attempt_integration.py`, `test_public_api_examples.py`.
  - P9: Task 9 replaces the constant `dose_response_note` string with real report-only
    dose diagnostics from the wrapper's deltas frame:
    `gk_median_abs_delta_at_2m`/`_at_4m` (groupby displacement level) + their ratio.
  - P10: the causal fixture tests use a NEW `simple_actions(specs)` helper in
    `tests/causal/_fixtures.py` building the full 11-column frame from
    `(type_name, t, result_id=success)` tuples; `actions()` is untouched (no
    shape-sniffing).
  - Minors, all binding: drop the unused `spearmanr` import from `evaluate_xs_probe`
    and the unused `field` import from the dataclass snippet (F401); `_Planted`
    declares `carrier_params: ClassVar[dict] = {}` (RUF012); `regate_verdict`'s
    `no_valid_placebo → unmeasurable_at_dose` conflation is recorded as DELIBERATE in
    ADR-037 (the probe report string preserves the distinction; follow-ups differ but
    both stay gated); TODO.md edit anchors on the full three-name span "(Alisson, Ter
    Stegen, Neuer should score strongly negative"; version check uses `git ls-remote
    --tags origin` (local clones lack release tags); `causal/__init__` additionally
    exports `OpportunityConfig`, `xcross_config`, `shot_arm_config`,
    `SHOT_ARM_CONFOUNDERS`, `smd_balance`, `abadie_imbens_se`, `CausalEstimate`,
    `GK_ABLATION_MIN_SHIFT`, `PLACEBO_BAND_PERCENTILE`; `cfg.max_spell_seconds` is
    THREADED into the spell loop (the module constant becomes the default only); the
    Task 12 commit message's "dose-response + zero-inflation prongs ANDed" is corrected
    to "dose-response prong ANDed; GK-band zero fraction report-only (placebo-side
    non-degeneracy gates)"; ADR-037 additionally records: the ratio prong's
    strengthening to `max(nd_med, placebo_p95)` as a deliberate amendment (and the
    redundant `gk_med > placebo_p95` conjunct is DROPPED — implied by ratio ≥ 2 with
    p95 > 0), and the paired-vector off-pitch policy (score, never clamp, report the
    fraction).
- **Rev 3 (round-2 review, N1):** `_equal_block_subsample` REPLACED by
  `_dose_response_clustered` — per-game Spearman ρ + sign-flip permutation across
  game-level ρ's (cluster-exact, ragged-native, nothing truncated; min-block surgery
  degenerated to a row permutation at m=1 and its power collapse could manufacture the
  flat-dose veto). Adaptations: constant-delta games count ρ=0 (measured flat, keeps
  the flat-override fixture meaningful); three dose states with underpowered → the
  SUPPORT verdict, never flat and never a lone band pass; sorted game iteration for
  row-order-independent determinism (M1 moot, M2 moot — no groupby.apply);
  `XS_PROBE_MIN_GAME_N=10` / `XS_PROBE_MIN_GAMES=8` registered (fixtures carry 10-12
  games); `dose_response_n_games` + `dose_state` reported. Also settled: the
  known-truth causal tests call only public `build_opportunities` (grep-verified) —
  the `_label_outcome(cfg)` signature change breaks nothing.
