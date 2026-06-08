# TF-17 xCrossAttempt — PR-B (weights + GK validation + TF-19 wiring) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (owner runs inline — no subagents per `feedback_inline_execution_default`). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the untrained xCross code path into a shipped, weighted feature — train on the clean-4.13.0-GS pining corpus, produce the headline GK-validation evidence (ablation + substitution probe + permutation importance), wire it into the GK-union xfn list, and bundle/publish weights.

**Architecture:** Hexagonal. The three maintainer-run validations live in a new **private** module `silly_kicks/tracking/_xcross_eval.py` (clean test imports — `scripts/` is not a package). `train_xcross_attempt.py` calls into it and writes everything to `metrics.json`. The bundled `default` booster (tiny) is the production model; the Hub `full` is opt-in reproducibility only, published iff the two-candidate paired test ships two. TDD throughout.

**Tech Stack:** pandas, numpy, scikit-learn (runtime); xgboost (inference + HPO, lazy); ruthless-efficiency[optuna] (HPO); huggingface_hub (`[xcross]` extra, lazy). No new runtime deps.

**Spec:** `docs/superpowers/specs/2026-06-06-tf17-xcross-pr-b-design.md` (settled, 2 review rounds).

**Reference modules (read before starting — PR-B mirrors them):**
- `silly_kicks/tracking/_xcross_attempt.py` — extractor (`extract_xcross_features`, sig: `(frame_data, *, gk_team_id, goal_x, carrier_player_id, feature_set="faithful", score_differential=np.nan)`), `XCROSS_FEATURE_NAMES_FAITHFUL` (16), `XCROSS_GK_BLOCK` (6, contiguous tail), `_build_goal_map` (`:241`), `_in_wide_area` (`:260`), `_pinned_params` (`:369`), `XCrossAttemptModel` (`from_variant` `:527`, `from_hub` `:548`), `_DEFAULT_CARRIER_PARAMS`.
- `scripts/train_xcross_attempt.py` — `_cv_metrics`, `_gates`, `_paired_data_effect`, `main` ship logic.
- `scripts/train_xshot_occurrence.py`, `scripts/publish_xshot_occurrence.py` — the weights-cycle template (PR-S80).
- `silly_kicks/tracking/_xshot_occurrence.py` — `from_hub` working pattern is in `_ghost_gk.py` (xS's still raises); `subsample_negatives` (re-used verbatim).
- `tests/tracking/test_xshot_occurrence_integration.py:152-179` — the liveness-tripwire / from_variant / metadata-intent tests to mirror.
- `tests/datasets/tracking/xshot_directional/frozen_rows.parquet` — directional fixture shape to mirror.

**Conventions (verified):**
- `warnings.warn(..., stacklevel=2)` on every warning.
- Run suite: `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- Lint trio (CI parity, run all three): `uv run ruff check silly_kicks/ tests/ scripts/` && `uv run ruff format --check silly_kicks/ tests/ scripts/` && `uv run pyright silly_kicks/` (pyright pinned 1.1.409).
- **Commit policy (owner rule):** ONE feature branch, work staged per task, **squashed into a single commit at the END (Task 16) with explicit approval**. The per-task "Stage" steps below `git add` only — do NOT commit mid-plan. Branch first (never commit on `main`).
- **No subagents** (owner rule); inline execution.

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `silly_kicks/tracking/_xcross_eval.py` | **Create** | private maintainer-eval module: `TF19_PROBE_RATIO`/`TF19_PROBE_ABS_FLOOR`, the shared `_cv_score` (used by the gate + ablation — M4), `gk_block_ablation`, `gk_substitution_probe`, `permutation_importance_report` |
| `scripts/train_xcross_attempt.py` | **Modify** | `_cv_metrics` delegates to `ev._cv_score` (M4, no drift); call the eval module on the shipped candidate; save a deterministic probe-sample during extraction (M1/M3); score-range probe; write all validations to `metrics.json` |
| `silly_kicks/tracking/_xcross_attempt.py` | **Modify** | real `from_hub` body (mirror `_ghost_gk.py`) |
| `silly_kicks/tracking/features.py` | **Modify** | append `xcross_attempt_xfns()` to `pre_shot_gk_full_default_xfns` (`:742`) |
| `silly_kicks/atomic/tracking/features.py` | **Modify** | append `xcross_attempt_xfns()` to `atomic_pre_shot_gk_full_default_xfns` (`:489`) + import |
| `scripts/publish_xcross_attempt.py` | **Create** | HF upload + round-trip verify (mirror `publish_xshot_occurrence.py`) |
| `silly_kicks/tracking/_xcross_weights/default/` | **Create (artifact)** | bundled booster (from the box run) — model.json/metadata.json/metrics.json/SHA256SUMS |
| `pyproject.toml` | **Modify** | `[xcross]` extra; hatch exclude `_xcross_weights/full` on wheel + sdist; version 4.15.0; ruff per-file-ignore for the trainer if needed |
| `docs/huggingface/model-cards/xcross-attempt-v1-model-card.md` | **Create (iff full ships)** | model card |
| `tests/tracking/test_xcross_eval.py` | **Create** | unit tests for the eval module (fixture-fit model) |
| `tests/tracking/test_xcross_attempt_integration.py` | **Modify** | tripwire / from_variant / from_hub-shape / xfn-membership / metadata-intent |
| `tests/tracking/test_xcross_attempt_e2e.py` | **Create** | token-gated e2e (acceptance gates, cross-provider, ablation-runs, probe-runs) |
| `tests/datasets/tracking/xcross_directional/frozen_rows.parquet` | **Create (fixture)** | directional CI tripwire data |
| `NOTICE`, `CHANGELOG.md`, `TODO.md`, `silly_kicks/__init__.py`, `docs/superpowers/adrs/ADR-011-*.md`, `uv.lock` | **Modify** | ship hygiene |

**Two-phase execution:** Tasks 0–8 + 13–16 run on **Windows** (this workstation). Tasks 9–12 run on the **DGX Spark box** (`ssh karsten@192.168.68.73`). Code (0–8) must land before the box run (11) computes the validations; bundling (13) consumes the box artifacts.

---

## Task 0: Branch + `[xcross]` extra

**Files:** Modify `pyproject.toml`

- [ ] **Step 1: Verify next PR-S number + create the branch** (off current `main` @ 4.14.0)

```bash
git log --oneline -1            # expect 576b867 (4.14.0)
git checkout -b pr-s84-tf17-xcross-weights
```

- [ ] **Step 2: Add the `[xcross]` optional extra** to `pyproject.toml` `[project.optional-dependencies]` (mirror `[ghost-gk]`)

```toml
xcross = ["huggingface_hub>=0.20.0"]
```

- [ ] **Step 3: Verify it parses** — `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb')); print('ok')"` → `ok`.
- [ ] **Step 4: Stage** — `git add pyproject.toml`

---

## Task 1: `_xcross_eval.py` — TF-19 constants + `gk_block_ablation` + CV helper

**Files:** Create `silly_kicks/tracking/_xcross_eval.py`; Test `tests/tracking/test_xcross_eval.py`

- [ ] **Step 1: Write the failing test** in `tests/tracking/test_xcross_eval.py`

```python
import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _xcross_eval as ev
from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCROSS_GK_BLOCK


def _synth(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    # make gk_r genuinely informative so ablation can show a non-trivial (here ~0+) delta deterministically
    y = ((X["gk_r"] + X["dist_endline"] * 0.5 + rng.normal(scale=0.5, size=n)) > 0).astype(int).to_numpy()
    groups = np.array((["g1"] * (n // 2)) + (["g2"] * (n - n // 2)))
    return X, y, groups


def test_tf19_constants_present_and_typed():
    assert ev.TF19_PROBE_RATIO == 2.0
    assert ev.TF19_PROBE_ABS_FLOOR == 0.01


def test_gk_block_ablation_emits_with_without_and_deltas():
    X, y, groups = _synth()
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.gk_block_ablation(X, y, groups, params, seed=42)
    for k in ("with_gk_pr_auc", "without_gk_pr_auc", "with_gk_log_loss",
              "without_gk_log_loss", "delta_pr_auc", "delta_log_loss"):
        assert k in out, k
    # deltas are the difference (with - without)
    assert out["delta_pr_auc"] == pytest.approx(out["with_gk_pr_auc"] - out["without_gk_pr_auc"], abs=1e-9)
    # dropping the GK block leaves the 10 non-GK features
    assert len([c for c in X.columns if c not in XCROSS_GK_BLOCK]) == 10
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_eval.py -v` → FAIL (`ModuleNotFoundError`).

- [ ] **Step 3: Implement** `silly_kicks/tracking/_xcross_eval.py` (constants + CV helper + ablation)

```python
"""TF-17 xCrossAttempt maintainer-eval helpers (PR-B). PRIVATE, single-repo.

Three shipped-surface GK validations, all REPORTED (never assert "GK wins"):
- gk_block_ablation        -> marginal predictive value (reported context)
- gk_substitution_probe    -> does the surface move when the GK moves? (THE TF-19 gate)
- permutation_importance_report -> CV-held-out feature weights incl. score_differential (context)

Not promoted to ruthless-efficiency (an optimisation/search substrate, not model-evaluation);
promote to a public model-eval home only if/when a 2nd consumer (TF-19 / retro-xS) lands.
See docs/superpowers/specs/2026-06-06-tf17-xcross-pr-b-design.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from silly_kicks.tracking._xcross_attempt import (
    XCROSS_FEATURE_NAMES_FAITHFUL,
    XCROSS_GK_BLOCK,
    _pinned_params,
)

# --- Pre-registered TF-19 viability threshold (C1; frozen before the run) -----------------
TF19_PROBE_RATIO = 2.0       # GK median |Δ| must be >= 2x the stronger positional control
TF19_PROBE_ABS_FLOOR = 0.01  # AND GK median |Δ| >= 0.01 (1 pp of P(cross)) in absolute terms


def _cv_score(X: pd.DataFrame, y, groups, params: dict, *, seed: int = 42,
              negative_subsample: float | None = None) -> dict:
    """SINGLE shared CV scorer for BOTH the acceptance gate (trainer `_cv_metrics`) AND the
    ablation (M4 — one implementation, so they cannot drift on seed / negative_subsample). The
    splitter is ALWAYS ``random_state=42`` (the gate's fold construction); ``seed`` drives ONLY
    ``negative_subsample`` (``seed + fold_i``), exactly as the trainer's original `_cv_metrics`
    did. Returns the per-fold means both callers need (the trainer adds positive_rate /
    base_rate_brier on top)."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
    from sklearn.model_selection import StratifiedGroupKFold

    from silly_kicks.tracking._xshot_occurrence import subsample_negatives

    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)  # gate's fold construction
    prs, brs, lls = [], [], []
    for fold_i, (tr, te) in enumerate(skf.split(X, y, groups)):
        if len(np.unique(y[tr])) < 2:
            continue
        Xtr, ytr = X.iloc[tr], y[tr]
        if negative_subsample:  # TRAIN fold only; the eval fold keeps the true balance
            Xtr, ytr, _ = subsample_negatives(Xtr, ytr, ytr, fraction=negative_subsample, seed=seed + fold_i)
            if len(np.unique(ytr)) < 2:
                continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(ytr.mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(Xtr.to_numpy(float), ytr)
        p = clf.predict_proba(X.iloc[te].to_numpy(float))[:, 1]
        lls.append(log_loss(y[te], p, labels=[0, 1]))
        brs.append(brier_score_loss(y[te], p))
        if len(np.unique(y[te])) == 2:
            prs.append(average_precision_score(y[te], p))
    return {
        "pr_auc": float(np.mean(prs)) if prs else float("nan"),
        "brier": float(np.mean(brs)) if brs else float("nan"),
        "log_loss": float(np.mean(lls)) if lls else float("inf"),
        "pr_auc_std": float(np.std(prs)) if prs else float("nan"),
        "n_usable_folds": len(lls),
    }


def gk_block_ablation(X: pd.DataFrame, y, groups, params: dict, *, seed: int = 42,
                      negative_subsample: float | None = None) -> dict:
    """Reported context: held-out PR-AUC + log-loss WITH vs WITHOUT XCROSS_GK_BLOCK, scored via
    the SAME `_cv_score` the acceptance gate uses (so deltas are gate-comparable for any seed/ns)."""
    with_ = _cv_score(X[XCROSS_FEATURE_NAMES_FAITHFUL], y, groups, params,
                      seed=seed, negative_subsample=negative_subsample)
    base_cols = [c for c in XCROSS_FEATURE_NAMES_FAITHFUL if c not in XCROSS_GK_BLOCK]
    wo_ = _cv_score(X[base_cols], y, groups, params, seed=seed, negative_subsample=negative_subsample)
    return {
        "with_gk_pr_auc": with_["pr_auc"], "without_gk_pr_auc": wo_["pr_auc"],
        "with_gk_log_loss": with_["log_loss"], "without_gk_log_loss": wo_["log_loss"],
        "delta_pr_auc": with_["pr_auc"] - wo_["pr_auc"],
        "delta_log_loss": with_["log_loss"] - wo_["log_loss"],
        "note": "reported context (marginal predictive value); NOT the tf19_ready gate",
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_eval.py -v` → PASS (3 tests). (Requires `[xgboost]`.)
- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xcross_eval.py tests/tracking/test_xcross_eval.py`

---

## Task 2: `_xcross_eval.py` — `gk_substitution_probe` (the TF-19 gate)

**Files:** Modify `silly_kicks/tracking/_xcross_eval.py`; Test `tests/tracking/test_xcross_eval.py`

- [ ] **Step 1: Write failing tests** (append to `tests/tracking/test_xcross_eval.py`)

```python
def _probe_frames():
    """Two wide-area frames, ball near the left byline, carrier A1, one defender, a GK, ball row.
    Attacked goal at x=105 (GK near x~104)."""
    rows = []
    for fr, t in [(1, 40.0), (2, 40.4)]:
        rows += [
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="A",
                 player_id="A1", x=96.0, y=8.0, vx=1.0, vy=0.0, is_ball=False, is_goalkeeper=False, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="A",
                 player_id="A2", x=99.0, y=34.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="B",
                 player_id="B1", x=100.0, y=20.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="B",
                 player_id="Bgk", x=104.0, y=34.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=True, ball_state="alive"),
            dict(game_id="g", period_id=1, frame_id=fr, time_seconds=t, team_id="ball",
                 player_id=None, x=96.0, y=8.0, vx=1.0, vy=0.0, is_ball=True, is_goalkeeper=False, ball_state="alive"),
        ]
    return pd.DataFrame(rows)


def _fit_probe_model():
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel
    X, y, groups = _synth()
    return XCrossAttemptModel().fit(X, pd.Series(y))


def test_gk_substitution_probe_emits_gk_and_two_controls():
    m = _fit_probe_model()
    out = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=42)
    for k in ("gk_median_abs_delta", "nearest_def_median_abs_delta",
              "random_band_median_abs_delta", "tf19_ready", "n_frames_used"):
        assert k in out, k
    assert isinstance(out["tf19_ready"], bool)
    assert out["n_frames_used"] >= 1


def test_gk_substitution_probe_is_deterministic():
    m = _fit_probe_model()
    a = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    b = ev.gk_substitution_probe(m, _probe_frames(), actions=None, home_team_id="A", n_frames=2, seed=7)
    assert a["gk_median_abs_delta"] == b["gk_median_abs_delta"]
    assert a["random_band_median_abs_delta"] == b["random_band_median_abs_delta"]


def test_tf19_ready_reads_pinned_constants(monkeypatch):
    """C1: the gate uses TF19_PROBE_RATIO/TF19_PROBE_ABS_FLOOR from the module, not an inline literal."""
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is True           # 0.05 >= 2*0.02 and >= 0.01
    assert ev._tf19_ready(gk=0.03, nearest_def=0.02, rand=0.01) is False          # 0.03 < 2*0.02 -> ratio fails
    assert ev._tf19_ready(gk=0.008, nearest_def=0.02, rand=0.01) is False         # below abs floor
    assert ev._tf19_ready(gk=0.05, nearest_def=0.0, rand=0.0) is False            # M2: no control band (nd==0)
    assert ev._tf19_ready(gk=0.05, nearest_def=float("nan"), rand=float("nan")) is False  # M2: no control band
    monkeypatch.setattr(ev, "TF19_PROBE_RATIO", 10.0)
    assert ev._tf19_ready(gk=0.05, nearest_def=0.02, rand=0.01) is False          # respects the constant
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_eval.py -k probe -v` → FAIL.

- [ ] **Step 3: Implement** the probe + `_tf19_ready` + panel (append to `_xcross_eval.py`)

```python
def _tf19_ready(gk: float, nearest_def: float, rand: float) -> bool:
    """Pre-registered numeric gate (C1 + M2): GK median |Δ| must (a) be >= RATIO x the stronger
    positional control AND (b) clear the absolute floor (a big ratio over a negligible band is
    still negligible). M2: a real control band is REQUIRED — the nearest-defender control must be
    finite and > 0, else there was no placebo comparison at all and we must NOT pass on the
    abs-floor alone (that would re-open the A1 placebo hole at the gate)."""
    if not np.isfinite(gk):
        return False
    nd_ok = np.isfinite(nearest_def) and nearest_def > 0.0
    if not nd_ok:  # M2: no control band -> cannot declare tf19_ready
        return False
    control = max(float(nearest_def), float(rand) if np.isfinite(rand) else 0.0)
    return bool(gk >= TF19_PROBE_RATIO * control and gk >= TF19_PROBE_ABS_FLOOR)


def _displacement_panel(goal_x: float) -> list[tuple[str, float, float]]:
    """Geometrically-matched (dx, dy) panel applied identically to GK / nearest-def / random
    outfielders so 'same-magnitude' is comparable (A1). 'depth' is signed toward the attacked goal."""
    toward = 1.0 if goal_x >= 105.0 / 2 else -1.0
    return [
        ("lat+2", 0.0, 2.0), ("lat-2", 0.0, -2.0),
        ("lat+4", 0.0, 4.0), ("lat-4", 0.0, -4.0),
        ("depth+2", toward * 2.0, 0.0), ("depth-2", -toward * 2.0, 0.0),
    ]


def _abs_delta_for_player(model, grp, *, row_mask, panel, gk_team_id, goal_x, carrier_pid, sd) -> list[float]:
    """Baseline predict vs each panel-perturbed predict for the single player row(s) in row_mask."""
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features

    base_feats = extract_xcross_features(grp, gk_team_id=gk_team_id, goal_x=goal_x,
                                         carrier_player_id=carrier_pid, score_differential=sd)
    base_p = float(model.predict_proba(base_feats)[0])
    deltas = []
    for _name, dx, dy in panel:
        pert = grp.copy()
        pert.loc[row_mask, "x"] = pert.loc[row_mask, "x"].to_numpy(float) + dx
        pert.loc[row_mask, "y"] = pert.loc[row_mask, "y"].to_numpy(float) + dy
        feats = extract_xcross_features(pert, gk_team_id=gk_team_id, goal_x=goal_x,
                                        carrier_player_id=carrier_pid, score_differential=sd)
        deltas.append(abs(float(model.predict_proba(feats)[0]) - base_p))
    return deltas


def gk_substitution_probe(model, frames: pd.DataFrame, actions=None, *, home_team_id,
                          n_frames: int = 200, n_random: int = 3, seed: int = 42,
                          advance_m: float = 35.0) -> dict:
    """THE TF-19 viability gate (deterministic). For a fixed sample of wide-area frames, measure
    |P(cross|actual) - P(cross|shifted)| for the GK vs a nearest-defender control vs an averaged
    random-outfielder band, over a geometrically-matched displacement panel. Establishes the
    surface is GK-RESPONSIVE (necessary for TF-19); NOT causal GK importance (that is PR-C)."""
    from silly_kicks.tracking._xcross_attempt import _build_goal_map, _in_wide_area
    from silly_kicks.tracking._ball_carrier import derive_team_in_possession, infer_ball_carrier

    rng = np.random.default_rng(seed)
    cp = dict(getattr(model, "carrier_params", None) or {})
    carrier = infer_ball_carrier(frames, **cp) if cp else infer_ball_carrier(frames)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _build_goal_map(frames)

    # Collect eligible (resolvable carrier + GK row + wide-area) frame groups deterministically.
    groups_list = []
    for (gid, pid, fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], sort=False):
        in_poss = grp["team_in_possession"].dropna()
        if in_poss.empty:
            continue
        poss_team = in_poss.iloc[0]
        defending = [t for t in grp["team_id"].unique() if t not in (poss_team, "ball")]
        if not defending:
            continue
        goal_x = goal_map.get((gid, pid, defending[0]))
        if goal_x is None:
            continue
        ball = grp[grp["is_ball"]]
        bx = float(ball["x"].iloc[0]) if len(ball) else np.nan
        by = float(ball["y"].iloc[0]) if len(ball) else np.nan
        if not _in_wide_area(bx, by, goal_x, advance_m):
            continue
        cpid = grp["ball_carrier_player_id"].dropna()
        cpid = cpid.iloc[0] if not cpid.empty else None
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == defending[0])
        if cpid is None or not gk_mask.any():
            continue
        groups_list.append((grp.reset_index(drop=True), defending[0], goal_x, cpid))

    if not groups_list:
        return {"gk_median_abs_delta": float("nan"), "nearest_def_median_abs_delta": float("nan"),
                "random_band_median_abs_delta": float("nan"), "tf19_ready": False, "n_frames_used": 0,
                "note": "no eligible wide-area frames with a resolvable carrier + GK"}

    # Deterministic sample of up to n_frames.
    idx = np.arange(len(groups_list))
    if len(idx) > n_frames:
        idx = np.sort(rng.choice(idx, size=n_frames, replace=False))

    gk_d, nd_d, rb_d = [], [], []
    for i in idx:
        grp, gk_team, goal_x, cpid = groups_list[i]
        panel = _displacement_panel(goal_x)
        sd = float("nan")  # probe measures positional sensitivity; score held at NaN
        # GK
        gk_mask = grp["is_goalkeeper"].astype(bool) & (grp["team_id"] == gk_team)
        gk_d += _abs_delta_for_player(model, grp, row_mask=gk_mask, panel=panel,
                                      gk_team_id=gk_team, goal_x=goal_x, carrier_pid=cpid, sd=sd)
        # Nearest defender to the carrier (control a)
        carr = grp[grp["player_id"].astype(str) == str(cpid)]
        defenders = grp[(grp["team_id"] == gk_team) & ~grp["is_ball"].astype(bool)
                        & ~grp["is_goalkeeper"].astype(bool)]
        if len(carr) and len(defenders):
            cx, cy = float(carr["x"].iloc[0]), float(carr["y"].iloc[0])
            d2 = (defenders["x"].to_numpy(float) - cx) ** 2 + (defenders["y"].to_numpy(float) - cy) ** 2
            nd_id = defenders["player_id"].to_numpy()[int(np.argmin(d2))]
            nd_mask = grp["player_id"].to_numpy() == nd_id
            nd_d += _abs_delta_for_player(model, grp, row_mask=nd_mask, panel=panel,
                                          gk_team_id=gk_team, goal_x=goal_x, carrier_pid=cpid, sd=sd)
        # Averaged random-outfielder band (control b)
        outs = grp[~grp["is_ball"].astype(bool) & ~grp["is_goalkeeper"].astype(bool)]
        out_ids = outs["player_id"].to_numpy()
        if len(out_ids):
            pick = rng.choice(out_ids, size=min(n_random, len(out_ids)), replace=False)
            for rid in pick:
                rb_d += _abs_delta_for_player(model, grp, row_mask=grp["player_id"].to_numpy() == rid,
                                              panel=panel, gk_team_id=gk_team, goal_x=goal_x,
                                              carrier_pid=cpid, sd=sd)

    gk_med = float(np.median(gk_d)) if gk_d else float("nan")
    nd_med = float(np.median(nd_d)) if nd_d else float("nan")
    rb_med = float(np.median(rb_d)) if rb_d else float("nan")
    ready = _tf19_ready(gk_med, nd_med, rb_med)
    if not (np.isfinite(nd_med) and nd_med > 0.0):
        reason = "no control band (nearest-defender |Δ| absent/zero) — cannot compare; False (M2)"
    elif not ready:
        reason = "GK |Δ| did not clear ratio>=2.0 x control AND abs-floor>=0.01"
    else:
        reason = "GK |Δ| cleared both controls and the absolute floor"
    return {
        "gk_median_abs_delta": gk_med, "gk_mean_abs_delta": float(np.mean(gk_d)) if gk_d else float("nan"),
        "gk_p90_abs_delta": float(np.percentile(gk_d, 90)) if gk_d else float("nan"),
        "nearest_def_median_abs_delta": nd_med, "random_band_median_abs_delta": rb_med,
        "tf19_ready": ready, "tf19_reason": reason,
        "tf19_probe_ratio": TF19_PROBE_RATIO, "tf19_probe_abs_floor": TF19_PROBE_ABS_FLOOR,
        "n_frames_used": int(len(idx)),
        "note": "responsiveness (necessary for TF-19), NOT causal GK primacy (PR-C); "
                "nearest-def control is partially self-limiting (floating identity)",
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_eval.py -k "probe or tf19" -v` → PASS. **N2 (execution note):** `test_gk_substitution_probe_emits_gk_and_two_controls` asserts `n_frames_used >= 1`, which requires `_probe_frames()` to clear the live `_in_wide_area` + carrier + GK-resolution path. If the fixture lands inert (0 frames), nudge the ball/carrier coords (ball `y=8` is inside the `y<14` wide corridor and `x=96` within 35 m of `goal_x=105` — keep it there) until a frame qualifies; the test is the guard, this is the one spot that may need fixture-tuning rather than first-try green.
- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xcross_eval.py tests/tracking/test_xcross_eval.py`

---

## Task 3: `_xcross_eval.py` — `permutation_importance_report` (CV-held-out + coverage)

**Files:** Modify `silly_kicks/tracking/_xcross_eval.py`; Test `tests/tracking/test_xcross_eval.py`

- [ ] **Step 1: Write failing test** (append)

```python
def test_permutation_importance_cv_held_out_and_reports_coverage():
    X, y, groups = _synth()
    X = X.copy()
    X["score_differential"] = 0.0           # fully covered -> coverage 1.0
    X.loc[:9, "score_differential"] = np.nan  # a few missing -> coverage < 1.0
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.permutation_importance_report(X, y, groups, params, n_repeats=5, seed=42)
    assert "importances" in out and "score_differential" in out["importances"]
    assert "score_differential_importance" in out
    assert out["score_differential_coverage"] == pytest.approx(1 - 10 / len(X), abs=1e-9)
    assert out["held_out"] is True
    assert set(out["importances"]) == set(XCROSS_FEATURE_NAMES_FAITHFUL)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_eval.py -k importance -v` → FAIL.

- [ ] **Step 3: Implement** (append to `_xcross_eval.py`)

```python
def permutation_importance_report(X: pd.DataFrame, y, groups, params: dict, *,
                                  n_repeats: int = 10, seed: int = 42) -> dict:
    """Reported context: CV-HELD-OUT permutation importance (C2 — never permuted on the all-data
    shipped model's own training data). For each fold: fit on K-1, permute+score on fold K with
    scoring='average_precision' (B3); average importances across folds. Also report
    score_differential coverage (non-NaN fraction over the full matrix, B2)."""
    import xgboost as xgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedGroupKFold

    X = X[XCROSS_FEATURE_NAMES_FAITHFUL]
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups).astype(str)
    n_splits = max(2, min(5, len(np.unique(groups))))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    per_fold: list[np.ndarray] = []
    for tr, te in skf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p_ = dict(_pinned_params(params))
        p_["base_score"] = float(y[tr].mean())
        clf = xgb.XGBClassifier(**p_)
        clf.fit(X.iloc[tr].to_numpy(float), y[tr])
        r = permutation_importance(clf, X.iloc[te].to_numpy(float), y[te],
                                   scoring="average_precision", n_repeats=n_repeats, random_state=seed)
        per_fold.append(r.importances_mean)

    if per_fold:
        mean_imp = np.mean(np.vstack(per_fold), axis=0)
        importances = {f: float(v) for f, v in zip(XCROSS_FEATURE_NAMES_FAITHFUL, mean_imp)}
    else:
        importances = {f: float("nan") for f in XCROSS_FEATURE_NAMES_FAITHFUL}

    coverage = float(X["score_differential"].notna().mean())
    return {
        "importances": importances,
        "score_differential_importance": importances["score_differential"],
        "score_differential_coverage": coverage,
        "scoring": "average_precision", "n_repeats": n_repeats, "n_folds_used": len(per_fold),
        "held_out": True,
        "note": "CV-held-out, architecture-representative; NOT measured on the production "
                "weights' own training data. score_differential importance is qualified by coverage.",
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_eval.py -v` → PASS (all eval tests).
- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xcross_eval.py tests/tracking/test_xcross_eval.py`

---

## Task 4: Trainer integration — validations into `metrics.json` + probe sample + score-range probe

**Files:** Modify `scripts/train_xcross_attempt.py`; Test `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Read** `scripts/train_xcross_attempt.py` `main` (the ship block `:302-347`) + `_extract` (`:48-72`).

- [ ] **Step 2: Write a failing smoke assertion** — extend the existing trainer smoke test (or add) in `tests/tracking/test_xcross_attempt_integration.py` to assert the new `metrics.json` keys after a synthetic `--data-dir` run

```python
def test_train_script_emits_gk_validations(tmp_path):
    import json, os, subprocess, sys
    import numpy as np, pandas as pd
    from silly_kicks.spadl import config as spc

    def _match(dirp, seed):
        rng = np.random.default_rng(seed)
        rows, acts = [], []
        for fr, t in enumerate(np.linspace(0, 30, 60), start=1):
            wide_y = 8.0 if fr % 2 else 60.0
            rows += [
                dict(game_id=dirp.name, period_id=1, frame_id=fr, time_seconds=float(t), team_id="A",
                     player_id="A1", x=96.0, y=wide_y, vx=1.0, vy=0.0, is_ball=False, is_goalkeeper=False, ball_state="alive"),
                dict(game_id=dirp.name, period_id=1, frame_id=fr, time_seconds=float(t), team_id="B",
                     player_id="Bgk", x=104.0, y=34.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=True, ball_state="alive"),
                dict(game_id=dirp.name, period_id=1, frame_id=fr, time_seconds=float(t), team_id="B",
                     player_id="B1", x=100.0, y=30.0, vx=0.0, vy=0.0, is_ball=False, is_goalkeeper=False, ball_state="alive"),
                dict(game_id=dirp.name, period_id=1, frame_id=fr, time_seconds=float(t), team_id="ball",
                     player_id=None, x=96.0, y=wide_y, vx=1.0, vy=0.0, is_ball=True, is_goalkeeper=False, ball_state="alive"),
            ]
            if fr % 5 == 0:  # periodic crosses -> both label classes
                acts.append(dict(game_id=dirp.name, period_id=1, team_id="A", time_seconds=float(t),
                                 type_id=spc.actiontype_id["cross"], result_id=spc.result_id["success"]))
        dirp.mkdir(parents=True)
        pd.DataFrame(rows).to_parquet(dirp / "frames.parquet")
        pd.DataFrame(acts).to_parquet(dirp / "actions.parquet")

    data = tmp_path / "data"
    _match(data / "m1", 1); _match(data / "m2", 2)
    out = tmp_path / "out"
    r = subprocess.run([sys.executable, "scripts/train_xcross_attempt.py", "--data-dir", str(data),
                        "--output-dir", str(out), "--n-trials", "3"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=os.getcwd()))
    assert r.returncode == 0, r.stderr
    metrics = json.load(open(out / "xcross_attempt_v1" / "metrics.json"))
    assert "gk_block_ablation" in metrics
    assert "gk_substitution_probe" in metrics and "tf19_ready" in metrics["gk_substitution_probe"]
    assert "permutation_importance" in metrics
    assert "score_differential_coverage" in metrics["permutation_importance"]
    assert "score_differential_range_probe" in metrics
```

- [ ] **Step 3: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py::test_train_script_emits_gk_validations -v` → FAIL (keys absent).

- [ ] **Step 4: Implement — `_extract` saves a probe sample (M1 + M3: final, non-contradictory code).** Replace `_extract` (`scripts/train_xcross_attempt.py:48-72`) in full with the 5-tuple version below — it captures the first `probe_keep` shipped matches' frames+actions **before** `del frames`, under a guard:

```python
def _extract(source, horizon_seconds, *, probe_keep=2):
    from silly_kicks.tracking._ball_carrier import DEFAULT_CARRIER_PARAMS
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data

    parts_x, parts_y, parts_g, parts_p = [], [], [], []
    probe_frames, probe_actions, probe_home = [], [], None
    for prov, mid, actions, frames, home in source:
        X, y, groups = prepare_xcross_training_data(
            frames, actions, home_team_id=home, horizon_seconds=horizon_seconds,
            wide_area_only=True, carrier_params=DEFAULT_CARRIER_PARAMS,  # 4.7.0 shared constant (anti-drift)
        )
        if len(X):
            parts_x.append(X)
            parts_y.append(np.asarray(y, int))
            parts_g.append(np.asarray(groups))
            parts_p.append(np.array([prov] * len(X)))
            if len(probe_frames) < probe_keep:          # M3: capture a COPY before del frames
                # N3 (memory): keeps up to `probe_keep` matches' frames+actions resident for the whole
                # loop (deliberate, bounded — vs the original's immediate del). Fine at tracking scale on
                # the box; probe_keep=2 caps it. The per-match `del frames` still frees all others.
                probe_frames.append(frames.copy())
                probe_actions.append(actions.copy())
                probe_home = home
            print(f"  {prov}/{mid}: {len(X)} rows, {int(np.asarray(y).sum())} positives")
        del frames
    if not parts_x:
        raise SystemExit("No usable training data.")
    X = pd.concat(parts_x, ignore_index=True)[XCROSS_FEATURE_NAMES_FAITHFUL]
    return (X, np.concatenate(parts_y), np.concatenate(parts_g), np.concatenate(parts_p),
            (probe_frames, probe_actions, probe_home))
```

Update the **call site in `main`** — `probe_bundle` MUST be defined on **both** branches (M1: the cache-hit path never calls `_extract`):

```python
    # --- Phase 1: stream + extract + cache ---
    probe_bundle = ([], [], None)   # M1: bound on BOTH branches (cache-hit never calls _extract)
    if (cache / "features.parquet").exists():
        print(f"Loading cached features from {cache}")
        X = pd.read_parquet(cache / "features.parquet")
        y = np.load(cache / "labels.npy")
        groups = np.load(cache / "groups.npy", allow_pickle=True)
        providers = np.load(cache / "providers.npy", allow_pickle=True)
    else:
        if args.providers:
            source = _iter_matches_from_pining(args.providers.split(","), args.max_per_provider)
        else:
            source = _iter_matches_from_dir(Path(args.data_dir))
        t0 = time.time()
        X, y, groups, providers, probe_bundle = _extract(source, args.horizon_seconds)
        print(f"Extracted {len(X)} rows ({int(y.sum())} positives) in {time.time() - t0:.0f}s")
        cache.mkdir(parents=True, exist_ok=True)
        X.to_parquet(cache / "features.parquet")
        np.save(cache / "labels.npy", y)
        np.save(cache / "groups.npy", groups)
        np.save(cache / "providers.npy", providers)
        pf, pa, ph = probe_bundle                       # persist the probe sample (fresh-extract only)
        if pf:
            ps = cache.parent / "_probe_sample"
            ps.mkdir(parents=True, exist_ok=True)
            pd.concat(pf, ignore_index=True).to_parquet(ps / "frames.parquet")
            pd.concat(pa, ignore_index=True).to_parquet(ps / "actions.parquet")
            json.dump({"home_team_id": str(ph)}, open(ps / "meta.json", "w"))
```

(M1: if a feature cache exists WITHOUT a `_probe_sample/`, the probe load in Step 6 **raises loudly** rather than defaulting to a spurious `tf19_ready:False`. Box flow Task 9 deletes both caches, so a clean run always extracts the sample.)

- [ ] **Step 5: Implement — score-range probe** (B6). After loading/【extracting `X`, in `main` before HPO:

```python
    sd = X["score_differential"].to_numpy(dtype=float)
    sd_fin = sd[np.isfinite(sd)]
    sd_probe = {
        "coverage": float(np.isfinite(sd).mean()),
        "min": float(sd_fin.min()) if sd_fin.size else float("nan"),
        "max": float(sd_fin.max()) if sd_fin.size else float("nan"),
        "abs_ge_12_count": int((np.abs(sd_fin) >= 12).sum()),
    }
    if sd_probe["abs_ge_12_count"] > 0:   # B6 HARD-FAIL: phantom-owngoal signature (the old +-18 bug)
        raise SystemExit(f"score_differential range probe FAILED (impossible |sd|>=12): {sd_probe}. "
                         "Rebuild the feature cache on clean 4.13.0 GS events.")
    if sd_fin.size and np.abs(sd_fin).max() > 6:  # B6 SOFT-WARN: a real rout is legitimate
        print(f"WARN score_differential |max|>6 (legit blowout possible): {sd_probe}", file=sys.stderr)
```

- [ ] **Step 5b: M4 — refactor the trainer's `_cv_metrics` to DELEGATE to `ev._cv_score`** (one CV implementation; eliminates the seed/`negative_subsample` drift class entirely — the gate and the ablation now physically share folds). Replace `_cv_metrics` (`scripts/train_xcross_attempt.py:105-143`) in full:

```python
def _cv_metrics(X, y, groups, params, *, negative_subsample=None, seed=42) -> dict:
    """Label-stratified, match-grouped CV at FIXED params -> gate metrics on the TRUE balance.
    M4: scoring is delegated to silly_kicks.tracking._xcross_eval._cv_score so the acceptance gate
    and the GK-block ablation cannot drift (same folds for any seed/negative_subsample)."""
    from silly_kicks.tracking import _xcross_eval as ev

    s = ev._cv_score(X, y, groups, params, seed=seed, negative_subsample=negative_subsample)
    base = float(np.asarray(y, dtype=int).mean())
    return {**s, "positive_rate": base, "base_rate_brier": base * (1 - base)}
```

This is behaviour-preserving: `_cv_score` is the verbatim extraction of the original loop (splitter `random_state=42`, `seed` → `subsample_negatives(seed+fold_i)` only), and `_cv_metrics` re-adds its only two extra keys (`positive_rate`, `base_rate_brier`). The original `subsample_negatives`/`_pinned_params` imports move into `_cv_score`; remove them from `_cv_metrics` if now unused there (keep them where `_paired_data_effect` / the final fit still use them — `_paired_data_effect` is a DIFFERENT protocol and is intentionally left untouched).

- [ ] **Step 6: Implement — run the three validations on the shipped candidate** and fold into `metrics`. **Exact placement (N3):** in `main`'s ship block the two lines `metrics = {...}` (the existing literal) and `json.dump(metrics, open(art / "metrics.json", "w"), indent=2)` are adjacent — insert the computation **after** the final-fit `model` is built + the existing `metrics = {...}` literal, and **before** the `json.dump`. The `model` variable is the final-fit `XCrossAttemptModel` already built in the ship block; `ship_mask`/`candidates`/`shipped`/`sd_probe` are all in scope.

```python
    from silly_kicks.tracking import _xcross_eval as ev
    shipped_params = candidates[shipped]["params"]
    gk_ablation = ev.gk_block_ablation(X[ship_mask], y[ship_mask], groups[ship_mask], shipped_params, seed=seed)
    perm_imp = ev.permutation_importance_report(X[ship_mask], y[ship_mask], groups[ship_mask],
                                                shipped_params, n_repeats=10, seed=seed)
    # M1: the probe sample is REQUIRED — refuse to ship a spurious tf19_ready=False on a missing sample.
    ps = cache.parent / "_probe_sample"
    if not (ps / "frames.parquet").exists():
        raise SystemExit(
            "Feature cache present but _probe_sample/ absent -> cannot run the TF-19 substitution "
            "probe (the headline deliverable). Delete the feature cache and re-extract (box Task 9), "
            "or restore the probe sample. Refusing to ship a spurious tf19_ready=False.")
    pf = pd.read_parquet(ps / "frames.parquet")
    pa = pd.read_parquet(ps / "actions.parquet")
    phome = json.load(open(ps / "meta.json"))["home_team_id"]
    probe = ev.gk_substitution_probe(model, pf, actions=pa, home_team_id=phome, n_frames=200, seed=seed)

    metrics.update({
        "gk_block_ablation": gk_ablation,
        "gk_substitution_probe": probe,
        "permutation_importance": perm_imp,
        "score_differential_range_probe": sd_probe,
        "tf19_ready": probe.get("tf19_ready", False),
    })
    if not probe.get("tf19_ready", False):
        print(f"NOTE: tf19_ready=False ({probe.get('tf19_reason')}) — surface ships, but flagged "
              "NOT TF-19-ready (loud, not silent).", file=sys.stderr)
```

- [ ] **Step 7: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py::test_train_script_emits_gk_validations -v` → PASS.

- [ ] **Step 7b: M4 — no-divergence test (the gate IS `_cv_score`).** After Step 5b the trainer's `_cv_metrics` *delegates* to `ev._cv_score`, so there is only one CV implementation and drift is structurally impossible. This test proves the delegation holds **across the divergence axes the old parity test was blind to** — `seed` (the splitter) and `negative_subsample` — so a future re-introduction of a separate loop is caught:

```python
import pytest


@pytest.mark.parametrize("seed,ns", [(42, None), (7, None), (7, 0.5)])
def test_cv_metrics_delegates_to_eval_cv_score(seed, ns):
    """M4 (closed by extraction): the acceptance gate (_cv_metrics) and the ablation share ONE
    _cv_score, so they use identical folds for ANY seed / negative_subsample. Exact equality (same
    call), exercised on the seed + ns axes the old parity test could not see."""
    import importlib.util
    from pathlib import Path
    import numpy as np, pandas as pd
    from silly_kicks.tracking import _xcross_eval as ev
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL

    spec = importlib.util.spec_from_file_location("_train_xcross", Path("scripts/train_xcross_attempt.py"))
    trainer = importlib.util.module_from_spec(spec); spec.loader.exec_module(trainer)

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(300, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    y = (rng.random(300) > 0.6).astype(int)
    groups = np.array((["g1"] * 150) + (["g2"] * 150))
    params = {"n_estimators": 40, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}

    m = trainer._cv_metrics(X, y, groups, params, seed=seed, negative_subsample=ns)
    s = ev._cv_score(X, y, groups, params, seed=seed, negative_subsample=ns)
    assert m["pr_auc"] == s["pr_auc"]            # exact: _cv_metrics literally calls _cv_score
    assert m["log_loss"] == s["log_loss"]
    assert {"positive_rate", "base_rate_brier"} <= set(m)   # the gate keys still present
```

Run: `python -m pytest tests/tracking/test_xcross_attempt_integration.py::test_cv_metrics_delegates_to_eval_cv_score -v` → PASS (3 params). Exact equality (no tolerance — same call, addresses N1). **If it fails, someone re-introduced a second CV loop in `_cv_metrics` — restore the delegation.**

- [ ] **Step 8: Stage** — `git add scripts/train_xcross_attempt.py tests/tracking/test_xcross_attempt_integration.py`

---

## Task 5: Real `from_hub` body

**Files:** Modify `silly_kicks/tracking/_xcross_attempt.py:548`; Test `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Write failing tests**

```python
def test_from_hub_shape_mocked(monkeypatch, tmp_path):
    """from_hub downloads then loads; mock snapshot_download to a local saved artifact."""
    import numpy as np, pandas as pd
    from silly_kicks.tracking import _xcross_attempt as xc
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(120, 16)), columns=xc.XCROSS_FEATURE_NAMES_FAITHFUL)
    m = xc.XCrossAttemptModel().fit(X, pd.Series((rng.random(120) > 0.6).astype(int)))
    d = tmp_path / "art"; m.save(d)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id: str(d))
    back = xc.XCrossAttemptModel.from_hub("silly-kicks/xcross-attempt-v1")
    np.testing.assert_allclose(m.predict_proba(X), back.predict_proba(X), rtol=1e-9)


def test_from_variant_default_does_not_cascade_to_hub(monkeypatch):
    """B4: default loads bundled-or-raises; it must NOT call snapshot_download."""
    from silly_kicks.tracking import _xcross_attempt as xc
    xc._VARIANT_CACHE.clear()
    called = {"hub": False}
    monkeypatch.setattr(xc.XCrossAttemptModel, "from_hub",
                        classmethod(lambda cls, repo_id=xc._HF_REPO_ID: called.__setitem__("hub", True)))
    # with no bundled default dir, default must raise FileNotFoundError WITHOUT touching the hub
    if not (xc._XCROSS_WEIGHTS_ROOT / "default" / "SHA256SUMS").exists():
        with pytest.raises(FileNotFoundError):
            xc.XCrossAttemptModel.from_variant("default")
        assert called["hub"] is False
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py -k from_hub -v` → FAIL (current `from_hub` raises unconditionally).

- [ ] **Step 3: Implement** — replace `from_hub` body (`_xcross_attempt.py:548-552`) with the working ghost-GK pattern

```python
    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> XCrossAttemptModel:
        """Download published weights from HuggingFace Hub and load.

        Requires ``pip install silly-kicks[xcross]``.

        Examples
        --------
        >>> # model = XCrossAttemptModel.from_hub()
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("xCrossAttempt Hub weights require: pip install silly-kicks[xcross]") from None
        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py -k "from_hub or from_variant" -v` → PASS.
- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/_xcross_attempt.py tests/tracking/test_xcross_attempt_integration.py`

---

## Task 6: xfn-union wiring (TF-19 wiring)

**Files:** Modify `silly_kicks/tracking/features.py:742`, `silly_kicks/atomic/tracking/features.py:489`; Test `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Write failing tests**

```python
def test_xcross_xfns_in_pre_shot_gk_full_default():
    from silly_kicks.tracking import features as tf
    from silly_kicks.tracking._xcross_attempt import xcross_attempt_xfns
    names = {getattr(fn, "__name__", "") for fn in tf.pre_shot_gk_full_default_xfns}
    assert any("xcross" in n for n in names), names


def test_atomic_xcross_xfns_in_pre_shot_gk_full_default():
    from silly_kicks.atomic.tracking import features as af
    names = {getattr(fn, "__name__", "") for fn in af.atomic_pre_shot_gk_full_default_xfns}
    assert any("xcross" in n for n in names), names


def test_import_silly_kicks_does_not_import_xgboost():
    import subprocess, sys, os
    code = "import sys, silly_kicks; assert 'xgboost' not in sys.modules, sorted(m for m in sys.modules if 'xgb' in m)"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONPATH=os.getcwd()))
    assert r.returncode == 0, r.stderr
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py -k "pre_shot_gk_full" -v` → FAIL.

- [ ] **Step 3: Implement** — `silly_kicks/tracking/features.py`: add the import near the xS one (`:56`) and extend the union (`:742`)

```python
from ._xcross_attempt import xcross_attempt_xfns   # near line 56, beside xshot_occurrence_xfns
```
```python
# line 742 — append xcross to the GK-union list (NOT the general default; Hyrum/PR-S80 P3)
pre_shot_gk_full_default_xfns = (
    pre_shot_gk_default_xfns + pre_shot_gk_angle_default_xfns
    + xshot_occurrence_xfns() + xcross_attempt_xfns()
)
```

`silly_kicks/atomic/tracking/features.py`: import (`:19` area) + extend (`:489`)

```python
from silly_kicks.tracking._xcross_attempt import add_xcross_attempt, xcross_attempt_xfns
```
```python
atomic_pre_shot_gk_full_default_xfns = (
    atomic_pre_shot_gk_default_xfns + atomic_pre_shot_gk_angle_default_xfns
    + xshot_occurrence_xfns() + xcross_attempt_xfns()
)
```

(If `add_xcross_attempt` isn't already re-exported in the atomic `__all__`, add it — see PR-A Task 12.)

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py -k "pre_shot_gk_full or does_not_import_xgboost" -v` → PASS.
- [ ] **Step 5: Stage** — `git add silly_kicks/tracking/features.py silly_kicks/atomic/tracking/features.py tests/tracking/test_xcross_attempt_integration.py`

---

## Task 7: Directional fixture + bundled-model integration tests

> These tests need the **bundled `default/`** to exist — they will FAIL until Task 13 commits the box artifact. Author them now (TDD), but mark the bundled-model ones to **xfail with a clear reason** until weights land, then flip in Task 13.

**Files:** Create `tests/datasets/tracking/xcross_directional/frozen_rows.parquet` (built in Task 13 from the trained model's feature space); Modify `tests/tracking/test_xcross_attempt_integration.py`

- [ ] **Step 1: Write the fixture-schema test** (runs without weights once the parquet exists)

```python
def test_xcross_directional_fixture_schema():
    import pandas as pd
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL
    df = pd.read_parquet("tests/datasets/tracking/xcross_directional/frozen_rows.parquet")
    assert set(XCROSS_FEATURE_NAMES_FAITHFUL).issubset(df.columns)
    assert "label" in df.columns and df["label"].nunique() == 2
    assert df["label"].sum() >= 3 and (df["label"] == 0).sum() >= 3
```

- [ ] **Step 2: Write the bundled-model tripwire + in-bounds + metadata-intent tests** (xfail until Task 13)

```python
import pytest

_NO_WEIGHTS = not __import__("pathlib").Path(
    "silly_kicks/tracking/_xcross_weights/default/SHA256SUMS").exists()


@pytest.mark.skipif(_NO_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_bundled_model_is_live_not_degenerate():
    from sklearn.metrics import roc_auc_score
    import pandas as pd
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCrossAttemptModel
    df = pd.read_parquet("tests/datasets/tracking/xcross_directional/frozen_rows.parquet")
    m = XCrossAttemptModel.from_variant("default")
    p = m.predict_proba(df[XCROSS_FEATURE_NAMES_FAITHFUL])
    assert roc_auc_score(df["label"].to_numpy(), p) >= 0.9


@pytest.mark.skipif(_NO_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_from_variant_default_in_bounds():
    import numpy as np, pandas as pd
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCrossAttemptModel
    m = XCrossAttemptModel.from_variant("default")
    p = m.predict_proba(pd.DataFrame(np.zeros((4, 16)), columns=XCROSS_FEATURE_NAMES_FAITHFUL))
    assert np.all((p >= 0) & (p <= 1))


@pytest.mark.skipif(_NO_WEIGHTS, reason="bundled xcross default weights land in Task 13")
def test_xcross_bundled_metadata_matches_training_intent():
    import json
    from pathlib import Path
    md = json.load(open(Path("silly_kicks/tracking/_xcross_weights/default/metadata.json")))
    assert md["carrier_params"] == {"tolerance_m": 3.0, "beta": 0.0, "gamma": 0.25}
    assert md["pitch_length"] == 105.0 and md["pitch_width"] == 68.0
    assert "geometry_version" in md and "xgboost_version" in md
```

- [ ] **Step 3: Run** — `python -m pytest tests/tracking/test_xcross_attempt_integration.py -k "directional or bundled or from_variant_default_in_bounds" -v` → the schema test ERRORS (no parquet yet — acceptable placeholder; it goes green in Task 13) and the bundled ones SKIP. Confirm skip reasons print.
- [ ] **Step 4: Stage** — `git add tests/tracking/test_xcross_attempt_integration.py`

---

## Task 8: Token-gated e2e tests

**Files:** Create `tests/tracking/test_xcross_attempt_e2e.py`

- [ ] **Step 1: Read** `tests/tracking/test_das_e2e.py` (or the pining-using e2e) for the `PINING_FOR_THE_DATA_TOKEN` skip pattern + loader import.

- [ ] **Step 2: Write the e2e tests** (skip when token unset; import eval fns directly — the B1 payoff)

```python
import os
import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.e2e

_NO_TOKEN = not os.environ.get("PINING_FOR_THE_DATA_TOKEN")


def _one_public_match():
    import sys
    sys.path.insert(0, "scripts")
    from _loader_pining import load_matches
    for prov, mid, actions, frames, home in load_matches(providers=["skillcorner"], max_per_provider=1):
        return prov, mid, actions, frames, home
    pytest.skip("no skillcorner match available")


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_xcross_cross_provider_extract_runs():
    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, prepare_xcross_training_data
    _, _, actions, frames, home = _one_public_match()
    X, y, groups = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X):
        assert list(X.columns) == XCROSS_FEATURE_NAMES_FAITHFUL
        assert set(np.unique(y)).issubset({0, 1})


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_surface_gk_block_ablation_runs():
    from silly_kicks.tracking import _xcross_eval as ev
    from silly_kicks.tracking._xcross_attempt import prepare_xcross_training_data
    _, _, actions, frames, home = _one_public_match()
    X, y, groups = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if len(X) < 20 or len(np.unique(groups)) < 2:
        pytest.skip("insufficient single-match data for a 2-group CV ablation")
    params = {"n_estimators": 30, "max_depth": 3, "learning_rate": 0.1, "min_child_weight": 1, "reg_lambda": 1.0}
    out = ev.gk_block_ablation(X, y, groups, params)
    assert "delta_pr_auc" in out and "delta_log_loss" in out  # EMITS both, regardless of sign


@pytest.mark.skipif(_NO_TOKEN, reason="PINING_FOR_THE_DATA_TOKEN unset")
def test_gk_substitution_sensitivity_runs():
    from silly_kicks.tracking import _xcross_eval as ev
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel, prepare_xcross_training_data
    _, _, actions, frames, home = _one_public_match()
    X, y, _ = prepare_xcross_training_data(frames, actions, home_team_id=home)
    if not len(X):
        pytest.skip("no wide-area rows in this match")
    m = XCrossAttemptModel().fit(X, pd.Series(y))
    out = ev.gk_substitution_probe(m, frames, actions=actions, home_team_id=home, n_frames=50)
    # EMITS the distributions + flag; does NOT assert GK wins (inert -> reported, not a failure)
    for k in ("gk_median_abs_delta", "nearest_def_median_abs_delta",
              "random_band_median_abs_delta", "tf19_ready"):
        assert k in out
```

- [ ] **Step 3: Run** (token set) — `python -m pytest tests/tracking/test_xcross_attempt_e2e.py -m e2e -v`. Expected: PASS or principled `skip` (insufficient data), never a crash. (Token IS available — must actually run, not skip on token.)
- [ ] **Step 4: Stage** — `git add tests/tracking/test_xcross_attempt_e2e.py`

---

## Task 9: [BOX] Env sync + repo ≥4.13.0 + stale-cache delete

> Run on `ssh karsten@192.168.68.73`. Owner token passed inline (stays out of logs). Use `nohup … >log 2>&1 &` for the long run (Task 11); poll `tail`, do not block.

- [ ] **Step 1: Sync the box repo to this branch** (so the GS converter is clean 4.13.0+ and the new eval code is present)

```bash
ssh karsten@192.168.68.73 'cd ~/Development/silly-kicks && git fetch origin && git checkout pr-s84-tf17-xcross-weights || git checkout main'
```

(If the branch isn't pushed yet — it won't be, single-commit-at-end — instead `git fetch && git checkout main && git pull` to get ≥4.13.0, then `git apply` a patch of the working-tree changes per the PR-S80 deploy note. Generate the patch on Windows: `git diff main > /tmp/pr-s84.patch`, scp it, `git apply`.)

**N1 — verify the patch carries the new files BEFORE scp.** `git diff main` includes **staged** new files but skips **untracked** ones; Tasks 1–8 each `git add`, so this works only if staging happened. Confirm on Windows: `git diff main --stat | grep -E "_xcross_eval|train_xcross_attempt"` — both must appear, or the box would `git apply` cleanly yet import a missing `_xcross_eval` at Task 11. (If a file is missing, `git add` it first.) After `git apply` on the box: `ssh … 'cd ~/Development/silly-kicks && python -c "import silly_kicks.tracking._xcross_eval; print(\"eval ok\")"'`.

- [ ] **Step 2: Confirm version ≥4.13.0** — `ssh … 'cd ~/Development/silly-kicks && python -c "import silly_kicks; print(silly_kicks.__version__)"'` → ≥4.13.0.
- [ ] **Step 3: Reuse/refresh the venv + install** — `pip install -e ".[train,xgboost,kloppy]"` (add `pyarrow` if missing — the PR-S80 pilot caught it).
- [ ] **Step 4: DELETE any stale feature cache** (§6 — a stale cache auto-loads and silently poisons the run)

```bash
ssh … 'rm -rf ~/Development/xcross_refit/xcross_attempt_v1/_feature_cache ~/Development/xcross_refit/xcross_attempt_v1/_probe_sample'
```

---

## Task 10: [BOX] Fork B — choose the corpus source

- [ ] **Step 1: Inspect the clean_cache layout** — `ssh … 'ls ~/Development/ghost_gk_refit/clean_cache | head; ls ~/Development/ghost_gk_refit/clean_cache/* 2>/dev/null | head'`.
- [ ] **Step 2: Decide** —
  - If it is `DIR/<game>/{frames,actions}.parquet` → use `--data-dir ~/Development/ghost_gk_refit/clean_cache` (reuse).
  - Otherwise (it's a ghost-GK *feature* cache, not raw frames/actions) → use `--providers skillcorner,idsse,gradientsports` (fresh pull; the 4.13.0 converter produces clean GS either way).
- [ ] **Step 3: Record the choice** in the run log (it goes into the PR description + memory).

---

## Task 11: [BOX] Training run + paired test + validations

- [ ] **Step 1: Launch** (owner token inline; `nohup`; full corpus — no `--max-per-provider`, no `--negative-subsample`)

```bash
ssh … 'cd ~/Development/silly-kicks && PINING_FOR_THE_DATA_TOKEN=<token> nohup \
  python scripts/train_xcross_attempt.py \
    --providers skillcorner,idsse,gradientsports \
    --output-dir ~/Development/xcross_refit \
    --n-trials 60 --horizon-seconds 1.0 \
  > ~/Development/xcross_refit/run.log 2>&1 &'
```

(Or `--data-dir <clean_cache>` per Task 10. `--n-trials` is the HPO budget — pick the same as the xS/ghost-GK runs; the owner sets it.)

- [ ] **Step 2: Poll the log** (do NOT block; the owner signals progress)

```bash
ssh … 'tail -40 ~/Development/xcross_refit/run.log'
```

Watch for: the **score-range probe** line (must NOT hard-fail; `abs_ge_12_count==0`), per-match positive counts, the paired-test `ship_two` decision, the acceptance gates, and `tf19_ready`.

- [ ] **Step 3: On completion, read `metrics.json`** — `ssh … 'cat ~/Development/xcross_refit/xcross_attempt_v1/metrics.json'`. Confirm: `acceptance` all true; `shipped_variant`; `gk_block_ablation`/`gk_substitution_probe`/`permutation_importance`/`score_differential_range_probe` present; `score_differential_coverage`; `tf19_ready`.
- [ ] **Step 4: If acceptance gates FAILED** — the trainer wrote `metrics_FAILED.json` and refused to bundle. Diagnose from the log (positives too few / folds < 2 / corpus issue); fix and re-run. Do NOT hand-edit gates.

---

## Task 12: [BOX] Publish `full` to Hub (iff `ship_two`) + fetch `default` back

- [ ] **Step 1: If `shipped_variant == "full"`** — create the repo + publish (the booster is tiny; Hub is the opt-in reproducibility mirror)

```bash
ssh … 'cd ~/Development/silly-kicks && python -c "from huggingface_hub import HfApi; HfApi().create_repo(\"silly-kicks/xcross-attempt-v1\", repo_type=\"model\", exist_ok=True)"'
ssh … 'cd ~/Development/silly-kicks && python scripts/publish_xcross_attempt.py --artifact-dir ~/Development/xcross_refit/xcross_attempt_v1 --repo-id silly-kicks/xcross-attempt-v1'
```

(`publish_xcross_attempt.py` is created in Task 14 — push it to the box first, or run the upload inline via `HfApi().upload_folder`. The script does the round-trip `from_hub` verify.)

- [ ] **Step 2: If `shipped_variant == "public"`** — mirror xS exactly: **no Hub repo**, bundle only. Skip publish.
- [ ] **Step 3: Fetch the artifact back to Windows** for bundling — `scp -r karsten@192.168.68.73:~/Development/xcross_refit/xcross_attempt_v1 /tmp/xcross_artifact` (or via the run output). Keep model.json/metadata.json/metrics.json/SHA256SUMS.

---

## Task 13: [WINDOWS] Bundle `default` + hatch exclude + build & size-check + flip Task-7 tests

**Files:** Create `silly_kicks/tracking/_xcross_weights/default/`; build `tests/datasets/tracking/xcross_directional/frozen_rows.parquet`; Modify `pyproject.toml`

- [ ] **Step 1: Copy the artifact into the bundle path**

```bash
mkdir -p silly_kicks/tracking/_xcross_weights/default
cp /tmp/xcross_artifact/{model.json,metadata.json,metrics.json,SHA256SUMS} silly_kicks/tracking/_xcross_weights/default/
```

- [ ] **Step 2: Verify it loads** — `python -c "from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel as M; M._VARIANT_CACHE.clear(); print(M.from_variant('default').feature_set)"` → `faithful` (SHA-256 verified).

- [ ] **Step 3: Hatch exclude `_xcross_weights/full` on BOTH targets** — `pyproject.toml` (`:114` wheel, `:123` sdist), append to each `exclude` list:

```toml
exclude = ["silly_kicks/tracking/_ghost_gk_weights/full", "silly_kicks/tracking/_xcross_weights/full"]
```

- [ ] **Step 4: Build BOTH artifacts + size-check < 100 MB** (the 4.10.0 sdist lesson)

```bash
python -m build
ls -lh dist/*.whl dist/*.tar.gz
python -c "import glob,os; [print(f, os.path.getsize(f)/1e6,'MB') for f in glob.glob('dist/*')]; assert all(os.path.getsize(f) < 100e6 for f in glob.glob('dist/*')), 'artifact >=100MB'"
```

- [ ] **Step 5: Build the directional fixture** (`tests/datasets/tracking/xcross_directional/frozen_rows.parquet`). Write a one-off snippet: take a handful of cherry-picked maximally-separable feature rows (a near-byline wide cross-imminent state → label 1; a deep central quiet state → label 0; ≥3 each), in the 16-col `XCROSS_FEATURE_NAMES_FAITHFUL` schema + a `label` column, verify the bundled model ranks them (roc_auc ≥ 0.9), then `to_parquet`. Confirm with the Task-7 tripwire.

```bash
python -m pytest tests/tracking/test_xcross_attempt_integration.py -k "directional or bundled or from_variant_default_in_bounds or metadata_matches" -v
```
Expected: all PASS now (skips flip to pass; schema test green).

- [ ] **Step 6: Stage** — `git add silly_kicks/tracking/_xcross_weights/default pyproject.toml tests/datasets/tracking/xcross_directional/frozen_rows.parquet`

---

## Task 14: [WINDOWS] Publish script + model card (iff `full` shipped)

**Files:** Create `scripts/publish_xcross_attempt.py`; Create (iff full) `docs/huggingface/model-cards/xcross-attempt-v1-model-card.md`

- [ ] **Step 1: Create `scripts/publish_xcross_attempt.py`** (mirror `publish_xshot_occurrence.py` verbatim with substitutions)

```python
#!/usr/bin/env python
"""Publish a trained xCrossAttempt artifact to HuggingFace Hub (TF-17 weights, PR-B).

Verifies SHA-256 + a sanity prediction, uploads the folder, then re-downloads via from_hub and
asserts identical predictions. ``--verify-only`` stops before upload. Requires silly-kicks[xgboost,xcross].
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-dir", required=True)
    ap.add_argument("--repo-id", default="silly-kicks/xcross-attempt-v1")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    from silly_kicks.tracking._xcross_attempt import XCROSS_FEATURE_NAMES_FAITHFUL, XCrossAttemptModel

    art = Path(args.artifact_dir)
    model = XCrossAttemptModel.load(art)  # SHA-256 verified
    sample = pd.DataFrame(np.zeros((3, len(XCROSS_FEATURE_NAMES_FAITHFUL))), columns=XCROSS_FEATURE_NAMES_FAITHFUL)
    local_pred = model.predict_proba(sample)
    print(f"Loaded + verified {art}; sample preds {local_pred.tolist()}")
    if args.verify_only:
        print("verify-only: not uploading.")
        return

    from huggingface_hub import HfApi

    HfApi().upload_folder(folder_path=str(art), repo_id=args.repo_id, repo_type="model")
    back = XCrossAttemptModel.from_hub(args.repo_id)
    np.testing.assert_allclose(local_pred, back.predict_proba(sample), rtol=0, atol=0)
    print(f"Published to {args.repo_id} + round-trip verified.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify** — `python scripts/publish_xcross_attempt.py --artifact-dir silly_kicks/tracking/_xcross_weights/default --verify-only` → loads + prints sample preds.

- [ ] **Step 3: Model card (ONLY if `shipped_variant=="full"`)** — create `docs/huggingface/model-cards/xcross-attempt-v1-model-card.md` mirroring `ghost-gk-v1-model-card.md` (YAML frontmatter: license/tags/pipeline_tag/library_name). Record: state-anchored cross-attempt propensity; 7 paper confounders + GK block; trained providers; the GK-ablation + tf19_ready + score_differential-importance(+coverage) headline numbers from `metrics.json`; carrier params; **no `~` (strikethrough) — use "approx."**. If `public` shipped, **skip** (no Hub repo, mirror xS).

- [ ] **Step 4: Stage** — `git add scripts/publish_xcross_attempt.py` (+ the model card iff created).

---

## Task 15: [WINDOWS] Ship hygiene — NOTICE / version 4.15.0 / CHANGELOG / ADR-011 / uv.lock / TODO

**Files:** `NOTICE`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-011-*.md`, `uv.lock`

- [ ] **Step 1: NOTICE** — verify the PR-A Cao et al. entry is present (it is — PR-A Task 14). Extend with one sentence on the GK-confounder finding only if `tf19_ready` warrants. No new entry needed.

- [ ] **Step 2: Version bump 4.14.0 → 4.15.0 (4-file hard-gate)** — `pyproject.toml` `version`, `silly_kicks/__init__.py` `__version__`, `TODO.md` "Current release", `CHANGELOG.md` new dated section:

```markdown
## [4.15.0] - 2026-06-06
### Added
- TF-17 xCrossAttempt (xCross) TRAINED weights (PR-B). Bundled `<shipped_variant>` model trained on
  the clean-4.13.0-GS pining corpus (<N> matches) against the 4.7.0 carrier defaults; two-candidate
  public/full paired test shipped `<shipped_variant>`. `from_variant("default")` + `from_hub` live.
  `xcross_attempt_xfns` wired into `pre_shot_gk_full_default_xfns` (+ atomic) only.
### Validation (reported, in metrics.json)
- GK-block ablation Δ PR-AUC <x> / Δ log-loss <y> (reported context).
- GK substitution-sensitivity probe: GK median |Δ| <g> vs nearest-def <d> / random-band <r>;
  tf19_ready=<bool> (pre-registered: ratio>=2.0 AND abs-floor>=0.01).
- Permutation importance (CV-held-out): score_differential importance <s> at coverage <c>%.
### Note
- A future TF-24 carrier-default change is an xCross retrain trigger (carrier params in metadata).
```

- [ ] **Step 3: ADR-011 note** — append "Update — TF-17 PR-B (xCross weights)": 2nd trained-model weights cycle after xS; same staged code→weights pattern; paired-test outcome + tf19_ready recorded; `_xcross_eval.py` private model-eval home.

- [ ] **Step 4: `uv lock`** — `uv lock` (sync the version bump + `[xcross]` extra). Confirm `uv.lock` changed.

- [ ] **Step 5: TODO groom** (delete shipped, don't strikethrough) — remove the PR-B residual from the TF-17 row (leave the PR-C residual); in the "xS / xCross re-fit on clean GS events" row, delete the xCross half (xS half stays if still open).

- [ ] **Step 6: Stage** — `git add NOTICE pyproject.toml silly_kicks/__init__.py CHANGELOG.md TODO.md docs/superpowers/adrs/ uv.lock`

---

## Task 16: [WINDOWS] Full verification + /final-review + single commit + merge + tag

- [ ] **Step 1: Full non-e2e suite** — `python -m pytest tests/ -m "not e2e" -v --tb=short`. Expected: all green (eval unit + integration tripwire/from_hub/xfn + bundled-model + existing suite). Capture the pass count.
- [ ] **Step 2: e2e (token set — must run, not skip)** — `python -m pytest tests/tracking/test_xcross_attempt_e2e.py -m e2e -v`. Expected: PASS / principled skip, no crash. **N2 awareness:** the public e2e only asserts the validations *run + emit keys* (one skillcorner match may legitimately give `n_frames_used:0`); the **`tf19_ready` gate's true value is exercised only by the box full-corpus run (Task 11)** — a green public e2e is NOT headline validation. Read `tf19_ready` from the box `metrics.json`, not from the e2e.
- [ ] **Step 3: Lint trio** — `uv run ruff check silly_kicks/ tests/ scripts/` && `uv run ruff format --check silly_kicks/ tests/ scripts/` && `uv run pyright silly_kicks/`. Expected: clean. (Add a `pyproject.toml` ruff per-file-ignore for `scripts/train_xcross_attempt.py` if it trips `X`/`Y` naming — mirror the xS script entry.)
- [ ] **Step 4: Dependency-light import guard** — `python -c "import sys, silly_kicks; assert 'xgboost' not in sys.modules"`.
- [ ] **Step 5: Version hard-gate check** — confirm `pyproject.toml` / `silly_kicks/__init__.py` / `TODO.md` / `CHANGELOG.md` all read 4.15.0 and `uv.lock` is synced.
- [ ] **Step 6: `/final-review`** (mandatory, owner rule) — run it; fold findings; regenerate C4 if the review flags drift.
- [ ] **Step 7: Present the full diff + proposed commit message; HOLD for explicit approval** (commit sentinel). Proposed message (`git commit -F`):

```
feat(tracking): TF-17 xCrossAttempt trained weights + GK validation + TF-19 wiring -- silly-kicks 4.15.0 (PR-S84)

Trains xCross on the clean-4.13.0-GS pining corpus vs the 4.7.0 carrier defaults; ships the
<shipped_variant> bundled model. Headline GK validations (private _xcross_eval.py, all reported
to metrics.json): GK-block ablation, GK substitution-sensitivity probe (the TF-19 gate;
pre-registered ratio>=2.0 + abs-floor>=0.01, with nearest-def + averaged-random controls),
CV-held-out permutation importance incl. score_differential (+coverage). from_hub live;
xcross_attempt_xfns wired into pre_shot_gk_full_default_xfns (+atomic) only.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
```

- [ ] **Step 8: After approval** — `git commit -F <msgfile>`; push `git push -u origin pr-s84-tf17-xcross-weights`; open PR `gh pr create`.
- [ ] **Step 9: Merge** (branch protection REVIEW_REQUIRED, solo maintainer) — `gh pr merge --admin --squash --delete-branch`. **Wait for main CI green (owner signals — do not poll), THEN tag** `git tag -a v4.15.0 -m "silly-kicks 4.15.0 (PR-S84)"` && `git push origin v4.15.0`.

---

## Self-Review (completed against the spec)

- **Spec coverage:** §2 Fork A (private `_xcross_eval.py`) → Tasks 1–3; §3.1 ablation → Task 1; §3.2 probe (panel + 2 controls + pinned `TF19_PROBE_RATIO`/`ABS_FLOOR` + responsiveness-not-causal note) → Task 2; §3.3 importance (CV-held-out + coverage + pinned scoring/n_repeats) → Task 3; §4 `from_hub` real body + `[xcross]` extra + bundle + both-artifact size check → Tasks 0,5,13; §5.1 xfn wiring + Hyrum note → Task 6; §5.2 directional tripwire → Tasks 7,13; §5.3 e2e → Task 8; §6 clean-cache prereq (repo ≥4.13.0, stale-cache delete, score-range hard/soft probe) → Tasks 4,9,11; §7 box run + horizon → Tasks 9–11; §8 model card / NOTICE / version 4.15.0 / ADR-011 / uv.lock / TODO → Tasks 14,15; §11 DoD → Task 16.
- **Placeholder scan:** box-run values left intentionally owner-set are `--n-trials` (HPO budget) + the Fork-B source choice (Task 10) + the CHANGELOG `<x>/<y>/…` numbers (filled from `metrics.json` after the run) — these are run-outputs, not design gaps.
- **Type consistency:** `_cv_score(X, y, groups, params, *, seed, negative_subsample)` (shared by the trainer's `_cv_metrics` AND `gk_block_ablation` — M4), `gk_block_ablation(X, y, groups, params, *, seed, negative_subsample)`, `gk_substitution_probe(model, frames, actions=None, *, home_team_id, n_frames, n_random, seed, advance_m)`, `permutation_importance_report(X, y, groups, params, *, n_repeats, seed)`, `_tf19_ready(gk, nearest_def, rand)`, `TF19_PROBE_RATIO`/`TF19_PROBE_ABS_FLOOR`, `XCROSS_FEATURE_NAMES_FAITHFUL` (16) / `XCROSS_GK_BLOCK` (6) used consistently across Tasks 1–8 + the trainer (Task 4) + tests.
- **Commit discipline:** per-task steps `git add` only; the single squash commit is Task 16 Step 7–8 after `/final-review` + explicit approval (owner rule).
- **Two-phase boundary:** code (0–8) lands before the box run (11) which computes the validations; bundling (13) + tripwire-flip + ship (14–16) consume the box artifacts. Task 7's bundled-model tests `skipif` until Task 13 supplies the weights, then go green.
