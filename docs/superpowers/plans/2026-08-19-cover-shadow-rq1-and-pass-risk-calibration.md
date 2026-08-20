# Cover-shadow RQ1 + pass-risk calibration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended)
> or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Measure two shipped predictors (`lane_control` cover-shadow blocking; `pitch_control_at_target`)
against real GS WC2022 pass outcomes, emitting two reported-not-gated research artifacts.

**Architecture:** One expensive corpus-pass driver (`build_rq_pass_scores.py`) shards a per-pass score table
over GS WC2022 via `for_each`; two thin consumers (`validate_cover_shadow_rq1.py`,
`validate_pass_risk_calibration.py`) read that persisted table and each compute + write their artifact.
Shared helpers: `_rq_corpus.py` (played-pass extraction + orientation) and `_rq_metrics.py` (pure metrics).
No library code changes; no retrain; C4-free (all under `scripts/`).

**Tech Stack:** Python, pandas/numpy, sklearn (`roc_auc_score`), existing `silly_kicks` + `scripts/` seams.

> **Deviation from spec §3 (deliberate, flagged for review):** the spec says "two `validate_*` drivers
> sharing a loader," but two independent `for_each` drivers each re-load GS WC2022 (separate generation
> fingerprints). To honor the spec's own §1/§2 "ONE corpus pass → two artifacts," this plan splits the
> expensive pass into `build_rq_pass_scores.py` (the shardable corpus driver) and makes the two `validate_*`
> scripts persisted-table consumers — the codebase's established pattern ("an expensive corpus pass is its
> OWN shardable driver; the consumer takes the persisted table"; layer2-spells / arm-values precedent). It
> also lets the metric framing iterate without re-loading GS.

## Global Constraints

- **Reported-not-gated:** no CI assertion on the artifact numbers. Non-vacuity is asserted IN each driver
  (fails the run), never in CI.
- **No retrain, no library behaviour change:** consume existing seams only. C4 aggregator count unchanged.
- **Corpus:** GS WC2022 via `scripts/_loader_pining.load_matches(providers=["gradientsports"], cache_dir=…)`
  — owner-tier, full 25 Hz tracking.
- **Metric hierarchy (spec §2, load-bearing):** the leakage-free **completed-pass** rate leads
  (Driver A: `P(is_blocked_majority | completed)`; Driver B: `P(control < τ | completed)`); every
  failed-pass-reading metric (AUC, slope, recall, balanced-accuracy, all-passes low-control band) is
  labelled optimistic. This cycle measures **over-prediction, not detection** — state it in both artifacts.
- **Driver discipline:** `git_provenance()` + `require_clean_tree(prov, allow_dirty=…)` called from `main()`
  BEFORE work; `--allow-dirty` offered (records `dirty:true`); never shell out to `git rev-parse`.
  `for_each` sharding with a declared `_SHARD_SCHEMA_VERSION` referenced from `token_inputs`.
  `declare_inputs(...)` digests σ/λ + pitch-control method + `GEOMETRY_VERSION`. All three drivers enrolled
  in `ARTIFACT_DRIVERS` (`tests/scripts/test_provenance_wiring.py`). Private-seam use recorded in
  `docs/PRIVATE_CONSUMERS.md`.
- **Consumers refuse a dirty/missing/mismatched-commit upstream** (`build_rq_pass_scores`'s manifest) —
  the ADR-037 "every artifact this number derives from" rule.
- **ASCII-only `scripts/` sources** (the driver ASCII gate); every `scripts/` driver's `main()` parses
  `--help` (argparse) and exits 0.
- **NEVER a commit step.** The user commits once, at the end, on explicit approval.

## File structure

- Create `scripts/_rq_metrics.py` — pure metric helpers (Task 1).
- Create `scripts/_rq_corpus.py` — `extract_played_passes` + orientation helper (Task 2).
- Create `scripts/build_rq_pass_scores.py` — the corpus-pass driver: `score_match` (Task 3) + `main` (Task 4).
- Create `scripts/validate_cover_shadow_rq1.py` — consumer + cover-shadow artifact (Task 5).
- Create `scripts/validate_pass_risk_calibration.py` — consumer + pass-risk artifact (Task 6).
- Modify `tests/scripts/test_provenance_wiring.py` — add 3 drivers to `ARTIFACT_DRIVERS` (Tasks 4/5/6).
- Modify `docs/PRIVATE_CONSUMERS.md` — record `lane_control` / `LaneControlResult` /
  `resolve_next_touch_receiver` consumption (Task 3).
- Create `docs/superpowers/adrs/ADR-064-cover-shadow-rq1-and-pass-risk-validation.md` + CHANGELOG/TODO (Task 7).
- Tests: `tests/scripts/test_rq_metrics.py`, `test_rq_corpus.py`, `test_build_rq_pass_scores.py`,
  `test_validate_cover_shadow_rq1.py`, `test_validate_pass_risk_calibration.py`.

---

### Task 1: `_rq_metrics.py` — pure metric helpers

**Files:**
- Create: `scripts/_rq_metrics.py`
- Test: `tests/scripts/test_rq_metrics.py`

**Interfaces:**
- Consumes: `silly_kicks._calibration_metrics.ece`, `.reliability_slope`; `sklearn.metrics.roc_auc_score`.
- Produces:
  - `false_positive_rate(is_blocked: np.ndarray, is_completed: np.ndarray) -> float` — `P(blocked | completed)`.
  - `false_alarm_rate(control: np.ndarray, is_completed: np.ndarray, tau: float) -> float` — `P(control < tau | completed)`.
  - `auc(y: np.ndarray, score: np.ndarray) -> float` — NaN-safe wrapper over `roc_auc_score` (returns
    `float("nan")` when `y` has one class).
  - `reliability_curve(y, score, n_bins=10) -> dict` — `{"bin_mid": [...], "mean_pred": [...], "emp_rate": [...], "n": [...]}`.
  - `confusion(pred: np.ndarray, actual_pos: np.ndarray) -> dict` — `{"tp","fp","tn","fn","precision","recall","specificity","balanced_accuracy"}`.
  - `low_control_completion_band(control, is_success, taus=(0.1,0.2,0.3)) -> dict` — `{tau: P(success | control < tau)}` over ALL passes.
  - re-exports `ece`, `reliability_slope` for driver use.

- [ ] **Step 1: Write the failing test**

```python
# tests/scripts/test_rq_metrics.py
import numpy as np
from scripts import _rq_metrics as M

def test_false_positive_rate_conditions_on_completed_only():
    # 4 passes: blocked/completed, blocked/completed, blocked/failed, open/completed
    is_blocked   = np.array([True, True, True, False])
    is_completed = np.array([True, True, False, True])
    # P(blocked | completed): completed = idx 0,1,3 -> blocked at 0,1 -> 2/3
    assert abs(M.false_positive_rate(is_blocked, is_completed) - 2/3) < 1e-9

def test_false_alarm_rate_completed_only():
    control      = np.array([0.05, 0.15, 0.5, 0.05])
    is_completed = np.array([True, True, True, False])   # last is failed -> excluded
    # tau=0.1: completed = 0,1,2 -> control<0.1 at idx0 -> 1/3
    assert abs(M.false_alarm_rate(control, is_completed, tau=0.1) - 1/3) < 1e-9

def test_auc_nan_safe_on_single_class():
    assert np.isnan(M.auc(np.array([1,1,1]), np.array([0.1,0.2,0.3])))
    assert M.auc(np.array([0,0,1,1]), np.array([0.1,0.2,0.8,0.9])) == 1.0

def test_confusion_balanced_accuracy():
    pred       = np.array([True, True, False, False])
    actual_pos = np.array([True, False, False, True])
    c = M.confusion(pred, actual_pos)
    assert c["tp"] == 1 and c["fp"] == 1 and c["tn"] == 1 and c["fn"] == 1
    assert abs(c["balanced_accuracy"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run, expect FAIL** — `python -m pytest tests/scripts/test_rq_metrics.py -v` (module missing).

- [ ] **Step 3: Implement `scripts/_rq_metrics.py`**

```python
"""Pure metric helpers for the RQ validation cycle. No corpus, no I/O."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score
from silly_kicks._calibration_metrics import ece, reliability_slope  # re-exported for drivers

def false_positive_rate(is_blocked, is_completed) -> float:
    is_blocked, is_completed = np.asarray(is_blocked, bool), np.asarray(is_completed, bool)
    denom = int(is_completed.sum())
    return float(is_blocked[is_completed].mean()) if denom else float("nan")

def false_alarm_rate(control, is_completed, tau: float) -> float:
    control, is_completed = np.asarray(control, float), np.asarray(is_completed, bool)
    denom = int(is_completed.sum())
    return float((control[is_completed] < tau).mean()) if denom else float("nan")

def auc(y, score) -> float:
    y, score = np.asarray(y, float), np.asarray(score, float)
    m = np.isfinite(score)
    if len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], score[m]))

def reliability_curve(y, score, n_bins: int = 10) -> dict:
    y, score = np.asarray(y, float), np.asarray(score, float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(score, edges) - 1, 0, n_bins - 1)
    out = {"bin_mid": [], "mean_pred": [], "emp_rate": [], "n": []}
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out["bin_mid"].append(float((edges[b] + edges[b + 1]) / 2))
            out["mean_pred"].append(float(score[mask].mean()))
            out["emp_rate"].append(float(y[mask].mean()))
            out["n"].append(int(mask.sum()))
    return out

def confusion(pred, actual_pos) -> dict:
    pred, actual_pos = np.asarray(pred, bool), np.asarray(actual_pos, bool)
    tp = int((pred & actual_pos).sum()); fp = int((pred & ~actual_pos).sum())
    tn = int((~pred & ~actual_pos).sum()); fn = int((~pred & actual_pos).sum())
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ba = np.nanmean([rec, spec])
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": prec,
            "recall": rec, "specificity": spec, "balanced_accuracy": float(ba)}

def low_control_completion_band(control, is_success, taus=(0.1, 0.2, 0.3)) -> dict:
    control, is_success = np.asarray(control, float), np.asarray(is_success, bool)
    band = {}
    for tau in taus:
        m = control < tau
        band[tau] = float(is_success[m].mean()) if m.any() else float("nan")
    return band
```

- [ ] **Step 4: Run, expect PASS.**

---

### Task 2: `_rq_corpus.py` — played-pass extraction + orientation

**Files:**
- Create: `scripts/_rq_corpus.py`
- Test: `tests/scripts/test_rq_corpus.py`

**Interfaces:**
- Consumes: `silly_kicks.spadl.utils.resolve_next_touch_receiver(actions, *, positions=None) -> pd.Series`;
  `silly_kicks.tracking.link_actions_to_frames`; `silly_kicks.spadl.config` for the pass/cross type ids and
  the success/fail `result_id`.
- Produces: `extract_played_passes(actions, frames) -> pd.DataFrame` with ONE row per played pass:
  columns `game_id, period_id, action_id, frame_id, attacking_team_id, passer_x, passer_y,
  target_x, target_y, target_source ("receiver"|"end_xy"), is_completed (bool), is_fail (bool)`. Targets are
  in the FRAME coordinate convention (reprojected for away-team actions — see Step 3). Also
  `to_frame_coords(x, y, attacks_rtl: bool) -> tuple[float, float]`.

- [ ] **Step 1: Write the failing test** (uses a hand-built actions+frames — no corpus)

```python
# tests/scripts/test_rq_corpus.py
import numpy as np, pandas as pd
from scripts import _rq_corpus as C

def test_to_frame_coords_reflects_away_only():
    assert C.to_frame_coords(30.0, 20.0, attacks_rtl=False) == (30.0, 20.0)
    # away attacks x=105 in action-LTR -> point-reflect into home-attacks-right frame
    assert C.to_frame_coords(30.0, 20.0, attacks_rtl=True) == (75.0, 48.0)  # (105-30, 68-20)

def test_completed_pass_target_is_receiver_frame_position(mini_actions, mini_frames):
    out = C.extract_played_passes(mini_actions, mini_frames)
    row = out[out["action_id"] == 0].iloc[0]         # a completed home pass to a known teammate
    assert row["target_source"] == "receiver" and bool(row["is_completed"])
    assert not bool(row["is_fail"])

def test_failed_pass_target_is_end_xy(mini_actions, mini_frames):
    out = C.extract_played_passes(mini_actions, mini_frames)
    row = out[out["action_id"] == 1].iloc[0]         # a failed pass
    assert row["target_source"] == "end_xy" and bool(row["is_fail"])
```

(Fixtures `mini_actions`/`mini_frames`: a 2-pass, 2-team synthetic set — one completed home pass whose next
same-team touch is a known teammate, one failed pass — built inline in a `conftest`/fixture; frames carry
`team_attacking_direction`, `team_id`, `is_ball`, `x`, `y`, `frame_id` matching the action `action_id`s.)

- [ ] **Step 2: Run, expect FAIL** (module missing).

- [ ] **Step 3: Implement `scripts/_rq_corpus.py`**

```python
"""Shared corpus helpers for the RQ validation cycle: played-pass extraction + orientation."""
from __future__ import annotations
import pandas as pd
from silly_kicks.spadl import config as spc
from silly_kicks.spadl.utils import resolve_next_touch_receiver
from silly_kicks.tracking import link_actions_to_frames
from silly_kicks.id_compat import ids_match, canonical_id

_PASS_TYPES = {spc.actiontype_id[t] for t in ("pass", "cross")}   # spec §4: pass/cross ONLY (F4)
_CROSS = spc.actiontype_id["cross"]                               # crosses are aerial -> Driver A headline is pass-only
_SUCCESS = spc.result_id["success"]

def to_frame_coords(x: float, y: float, attacks_rtl: bool) -> tuple[float, float]:
    # Action-LTR (acting team attacks x=105) -> frame convention (home-attacks-right). Away-team actions
    # are a 180-degree point reflection (ADR-028); home actions are already frame-aligned.
    return (105.0 - x, 68.0 - y) if attacks_rtl else (x, y)

def _acting_attacks_rtl(fr: pd.DataFrame, team_id) -> bool:
    # GS frames are home-attacks-right; a team's own player rows carry team_attacking_direction in
    # {"ltr","rtl"}. NB is_ball via to_numpy(dtype=bool) NOT .astype(bool) on an object column (ADR-019).
    prow = fr[ids_match(fr["team_id"], team_id) & ~fr["is_ball"].to_numpy(dtype=bool)]
    return (not prow.empty) and str(prow["team_attacking_direction"].iloc[0]) == "rtl"

def _player_frame_xy(fr: pd.DataFrame, pid) -> tuple[float, float] | None:
    row = fr[ids_match(fr["player_id"], pid)]
    return None if row.empty else (float(row["x"].iloc[0]), float(row["y"].iloc[0]))

def extract_played_passes(actions: pd.DataFrame, frames: pd.DataFrame, *,
                          links: pd.DataFrame | None = None) -> pd.DataFrame:
    passes = actions[actions["type_id"].isin(_PASS_TYPES)].copy()
    links = links if links is not None else link_actions_to_frames(actions, frames)  # SHARE the link (consistency)
    passes["frame_id"] = passes["action_id"].map(links.set_index("action_id")["frame_id"])
    passes = passes[passes["frame_id"].notna()]
    receiver_id = resolve_next_touch_receiver(actions).reindex(passes.index)
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}   # index ONCE per match (F1)
    rows = []
    for idx, a in passes.iterrows():
        fr = by_frame.get(canonical_id(a["frame_id"]))
        if fr is None:
            continue
        attacks_rtl = _acting_attacks_rtl(fr, a["team_id"])
        is_completed = int(a["result_id"]) == _SUCCESS
        rid = receiver_id.get(idx)
        rec_xy = _player_frame_xy(fr, rid) if (is_completed and pd.notna(rid)) else None
        if rec_xy is not None:                                    # receiver read from frame -> NOT reflected
            tx, ty, src = rec_xy[0], rec_xy[1], "receiver"
        else:                                                     # end_xy is action-LTR -> reflect for away
            tx, ty = to_frame_coords(float(a["end_x"]), float(a["end_y"]), attacks_rtl)
            src = "end_xy"
        px, py = to_frame_coords(float(a["start_x"]), float(a["start_y"]), attacks_rtl)
        rows.append({"game_id": a["game_id"], "period_id": a["period_id"], "action_id": a["action_id"],
                     "frame_id": a["frame_id"], "attacking_team_id": a["team_id"],
                     "passer_x": px, "passer_y": py, "target_x": tx, "target_y": ty,
                     "target_source": src, "is_cross": int(a["type_id"]) == _CROSS,
                     "is_completed": is_completed, "is_fail": not is_completed})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run, expect PASS.**

---

### Task 3: `build_rq_pass_scores.py::score_match` — lane + control per pass (with the orientation gate)

**Files:**
- Create: `scripts/build_rq_pass_scores.py` (the `score_match` work function + module constants)
- Modify: `docs/PRIVATE_CONSUMERS.md`
- Test: `tests/scripts/test_build_rq_pass_scores.py`

**Interfaces:**
- Consumes: `_rq_corpus.extract_played_passes`; `silly_kicks.tracking._cover_shadows.lane_control` +
  `LaneControlResult`; `silly_kicks.tracking.resolve_defended_goals`;
  `silly_kicks.tracking.pitch_control_at_target`.
- Produces: `score_match(actions, frames) -> pd.DataFrame` — the per-pass shard: the `extract_played_passes`
  columns PLUS `p_blocked_center, p_blocked_mean, p_blocked_max, is_blocked_majority, control`.
  `_SHARD_SCHEMA_VERSION = "rq-scores-1"` and `_EMITTED_SHARD_COLUMNS` (the exact list).

- [ ] **Step 1: Write the failing test** — including the ADR-028 both-sides orientation gate

```python
# tests/scripts/test_build_rq_pass_scores.py
import numpy as np, pandas as pd
from scripts import build_rq_pass_scores as B

def test_shard_has_declared_columns(mini_actions, mini_frames):
    out = B.score_match(mini_actions, mini_frames)
    assert set(out.columns) == set(B._EMITTED_SHARD_COLUMNS)  # keys the rows ACTUALLY carry, not selected
    assert out["p_blocked_mean"].between(0.0, 1.0).all()
    assert out["control"].between(0.0, 1.0).all()

def test_lane_scoring_is_orientation_invariant(mini_actions, mini_frames):
    """ADR-028 physical mirror: a home pass and its full-pitch mirror (frame + action + team flipped)
    must produce the SAME p_blocked. If it does not, the driver's to_frame_coords reprojection is wrong."""
    base = B.score_match(mini_actions, mini_frames)
    m_actions, m_frames = _mirror_match(mini_actions, mini_frames)  # x->105-x, y->68-y, swap home/away
    mirrored = B.score_match(m_actions, m_frames)
    np.testing.assert_allclose(sorted(base["p_blocked_mean"]), sorted(mirrored["p_blocked_mean"]), atol=1e-9)
```

(`_mirror_match`: point-reflect every `start/end/x/y`, negate direction labels, swap the two `team_id`s and
`home` — the ADR-051 Gate-A physical mirror helper; build it in the test module.)

- [ ] **Step 2: Run, expect FAIL** (module missing).

- [ ] **Step 3: Implement `score_match`**

```python
"""Corpus-pass driver: shard the per-pass cover-shadow + pitch-control scores over GS WC2022."""
from __future__ import annotations
import numpy as np, pandas as pd
from silly_kicks.tracking import resolve_defended_goals, pitch_control_at_target, link_actions_to_frames
from silly_kicks.tracking._cover_shadows import lane_control
from silly_kicks.id_compat import canonical_id
from scripts import _rq_corpus as C

_SHARD_SCHEMA_VERSION = "rq-scores-1"
_EMITTED_SHARD_COLUMNS = [
    "game_id", "period_id", "action_id", "frame_id", "attacking_team_id",
    "passer_x", "passer_y", "target_x", "target_y", "target_source", "is_cross",
    "is_completed", "is_fail", "p_blocked_center", "p_blocked_mean", "p_blocked_max",
    "is_blocked_majority", "control",
]

def score_match(actions: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    links = link_actions_to_frames(actions, frames)   # ONE link -> same release frame for lane AND control
    passes = C.extract_played_passes(actions, frames, links=links)
    if passes.empty:
        return pd.DataFrame(columns=_EMITTED_SHARD_COLUMNS)
    gm = resolve_defended_goals(frames)
    control = pitch_control_at_target(actions, frames, links=links, method="spearman")  # SHARED link
    ctrl_by_aid = pd.Series(control.values, index=actions["action_id"].values)
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}   # index ONCE per match (F1)
    recs = []
    for _, p in passes.iterrows():
        fr = by_frame.get(canonical_id(p["frame_id"]))
        if fr is None:                     # symmetry with extract_played_passes (Rev-2 residual)
            continue
        r = lane_control(fr, (p["passer_x"], p["passer_y"]), (p["target_x"], p["target_y"]),
                         goal_map=gm, attacking_team_id=p["attacking_team_id"])
        lanes = (r.p_blocked_center, r.p_blocked_left, r.p_blocked_right)
        recs.append({**p.to_dict(),
                     "p_blocked_center": r.p_blocked_center,
                     "p_blocked_mean": float(np.mean(lanes)),
                     "p_blocked_max": float(np.max(lanes)),
                     "is_blocked_majority": bool(r.is_blocked_majority),
                     "control": float(ctrl_by_aid.get(p["action_id"], np.nan))})
    return pd.DataFrame(recs)[_EMITTED_SHARD_COLUMNS]
```

> **Frame/point consistency (Rev-2 residual):** the SHARED `links` guarantees `pitch_control_at_target` and
> `lane_control` resolve to the SAME release frame per pass. They sample DIFFERENT target POINTS by design —
> pitch control at the SPADL destination `(end_x, end_y)` (the shipped seam's natural output, which is what we
> are validating), the lane at the receiver's frame position (completed) or `end_xy` (failed). For a completed
> pass these nearly coincide (`end_xy` ≈ receiver); the difference is deliberate (each metric's natural target)
> and recorded in ADR-064, not silently assumed.

- [ ] **Step 4: Run, expect PASS.** Then add a `docs/PRIVATE_CONSUMERS.md` entry recording that
  `build_rq_pass_scores.py` imports `_cover_shadows.lane_control` / `LaneControlResult` and
  `spadl.utils.resolve_next_touch_receiver` (path pins fail silently — this is the guard).

---

### Task 4: `build_rq_pass_scores.py::main` — the sharded corpus pass

**Files:**
- Modify: `scripts/build_rq_pass_scores.py` (add `main`)
- Modify: `tests/scripts/test_provenance_wiring.py` (add `"build_rq_pass_scores"` to `ARTIFACT_DRIVERS`)
- Test: `tests/scripts/test_build_rq_pass_scores.py` (extend)

**Interfaces:**
- Consumes: `scripts._driver.for_each`, `.reconcile`; `scripts._provenance.git_provenance`,
  `.require_clean_tree`; `scripts._input_contract.declare_inputs`; `scripts._loader_pining.load_matches`;
  `silly_kicks.tracking._cover_shadows.CoverShadowParams`; `silly_kicks.tracking._geometry.GEOMETRY_VERSION`.
- Produces: writes `<out>/pass_scores.parquet` + `<out>/manifest.json` (with `run_commit`, `run_tree_dirty`,
  `schema=_SHARD_SCHEMA_VERSION`, `n_matches`, `n_passes`).

- [ ] **Step 1: Write the failing test** — `main()` on a tiny monkeypatched corpus + `--help`

```python
def test_main_writes_pass_scores_and_stamps_provenance(tmp_path, monkeypatch):
    import sys, json
    from scripts import build_rq_pass_scores as B
    def fake_load(**kw):  # yields ONE (provider, match_id, actions, frames, home)
        yield ("gradientsports", "m1", _mini_actions(), _mini_frames(), 1)
    monkeypatch.setattr(B, "load_matches", fake_load)
    monkeypatch.setattr(sys, "argv", ["build_rq_pass_scores.py",
                                      "--out", str(tmp_path/"out"), "--shard-root", str(tmp_path/"sh"),
                                      "--allow-dirty", "--min-passes", "1", "--min-completed", "1"])  # F2
    B.main()
    man = json.loads((tmp_path/"out"/"manifest.json").read_text())
    assert man["schema"] == "rq-scores-1" and man["n_passes"] > 0
    assert isinstance(man["run_tree_dirty"], bool) and man["run_commit"]

def test_help_parses_and_exits_zero():
    import sys, pytest
    from scripts import build_rq_pass_scores as B
    with pytest.raises(SystemExit) as e:
        old = sys.argv; sys.argv = ["build_rq_pass_scores.py", "--help"]
        try: B.main()
        finally: sys.argv = old
    assert e.value.code == 0
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement `main`** (argparse; provenance FIRST; `for_each`; reconcile; non-vacuity)

```python
def main() -> None:
    import argparse
    from scripts._driver import for_each, reconcile
    from scripts._provenance import git_provenance, require_clean_tree
    from scripts._input_contract import declare_inputs
    from silly_kicks.tracking._cover_shadows import CoverShadowParams
    from silly_kicks.tracking._geometry import GEOMETRY_VERSION
    ap = argparse.ArgumentParser(description="Shard per-pass cover-shadow + pitch-control scores (GS WC2022).")
    ap.add_argument("--out", required=True); ap.add_argument("--shard-root", required=True)
    ap.add_argument("--cache-dir", default=None); ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--min-passes", type=int, default=_MIN_PASSES)            # injectable floors (F2)
    ap.add_argument("--min-completed", type=int, default=_MIN_COMPLETED)
    args = ap.parse_args()
    prov = require_clean_tree(git_provenance(), allow_dirty=args.allow_dirty)  # FIRST, before any load
    cs = CoverShadowParams()
    token_inputs = declare_inputs(schema=_SHARD_SCHEMA_VERSION, sigma=cs.sigma, lambda_ctrl=cs.lambda_ctrl,
                                  pc_method="spearman", geometry_version=GEOMETRY_VERSION)
    items = load_matches(providers=["gradientsports"], cache_dir=args.cache_dir)
    res = for_each(items, key=lambda t: t[1], shard_root=args.shard_root, token_inputs=token_inputs,
                   work=lambda t: score_match(t[2], t[3]), label="match")
    reconcile(res.generation, f"{args.out}/pass_scores.parquet", tag="all")
    df = pd.read_parquet(f"{args.out}/pass_scores.parquet")
    n_completed = int(df["is_completed"].sum())
    assert len(df) >= args.min_passes and df["is_fail"].any() and df["is_completed"].any(), "vacuous pass set"
    assert n_completed >= args.min_completed, "too few completed passes for the leakage-free headline"
    _write_manifest(f"{args.out}/manifest.json", prov, schema=_SHARD_SCHEMA_VERSION,
                    n_matches=res.n_keys, n_passes=len(df))
```

`_MIN_PASSES` / `_MIN_COMPLETED`: module constants set relative to GS WC2022's known volume (64 matches ×
~900 passes ⇒ `_MIN_PASSES = 20_000`, `_MIN_COMPLETED = 12_000`) so a half-empty run trips them; the CLI
args let the `main()` unit test lower them to 1 (F2).

> **Licensing (F5, spec §7):** `--shard-root` and `--out` are **owner-run, gitignored** locations.
> `pass_scores.parquet` carries per-pass GS player positions (`passer_x/y`, `target_x/y`) derived from
> owner-tier GS tracking, so it is NEVER committed — only the two consumers' aggregate `metrics.json`
> (rates/AUC, no positions) + `README.md` land under `docs/research/`, ship-mask-labeled via
> `scripts/_corpus.py` (Tasks 5/6). The safe default for `--shard-root`/`--out` is a path OUTSIDE the repo;
> if a repo-relative default is offered, add a matching `.gitignore` entry so raw positions can never be
> committed.

- [ ] **Step 4: Run, expect PASS**, and run `python -m pytest tests/scripts/test_provenance_wiring.py -v`
  after adding `"build_rq_pass_scores"` to `ARTIFACT_DRIVERS`.

---

### Task 5: `validate_cover_shadow_rq1.py` — consumer + cover-shadow artifact

**Files:**
- Create: `scripts/validate_cover_shadow_rq1.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (add `"validate_cover_shadow_rq1"`)
- Test: `tests/scripts/test_validate_cover_shadow_rq1.py`

**Interfaces:**
- Consumes: `_rq_metrics` (`false_positive_rate`, `auc`, `ece`, `reliability_slope`, `reliability_curve`,
  `confusion`); the persisted `pass_scores.parquet` + its `manifest.json`.
- Produces: writes `docs/research/cover_shadow_rq1/metrics.json` + `README.md`; a
  `compute_cover_shadow_metrics(df) -> dict` pure function (tested without I/O).

- [ ] **Step 1: Write the failing test** (pure metrics + upstream-dirty refusal)

```python
def test_headline_fp_rate_is_pass_only_completed():
    import pandas as pd
    from scripts.validate_cover_shadow_rq1 import compute_cover_shadow_metrics
    df = pd.DataFrame({"is_blocked_majority":[True,True,True,False,True],
                       "is_completed":[True,True,False,True,True], "is_fail":[False,False,True,False,False],
                       "is_cross":[False,False,False,False,True],   # idx4 is a CROSS -> excluded from headline
                       "p_blocked_center":[.9,.8,.7,.1,.9], "p_blocked_mean":[.8,.7,.6,.1,.9],
                       "p_blocked_max":[.95,.9,.8,.2,.95]})
    m = compute_cover_shadow_metrics(df)
    # PASS-ONLY completed = idx 0,1,3 (cross idx4 dropped) -> blocked-majority at 0,1 -> 2/3
    assert abs(m["headline_fp_rate"]["majority"] - 2/3) < 1e-9
    assert "pass_plus_cross_secondary" in m                       # paper-comparable cut retains the cross
    assert m["paper_reconciliation"]["required_sentence"]         # Q1

def test_consumer_refuses_dirty_upstream(tmp_path, monkeypatch):
    import sys, json, pandas as pd, pytest
    from scripts import validate_cover_shadow_rq1 as V
    scores = tmp_path / "pass_scores.parquet"
    pd.DataFrame({"is_blocked_majority": [True], "is_completed": [True], "is_fail": [False],
                  "p_blocked_center": [.9], "p_blocked_mean": [.8], "p_blocked_max": [.95]}).to_parquet(scores)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"schema": "rq-scores-1", "run_tree_dirty": True, "run_commit": "abc", "n_passes": 1}))
    monkeypatch.setattr(sys, "argv", ["validate_cover_shadow_rq1.py",
        "--pass-scores", str(scores), "--out", str(tmp_path / "art"), "--allow-dirty"])
    with pytest.raises(SystemExit) as e:
        V.main()
    assert e.value.code != 0   # refuses a dirty upstream artifact (ADR-037)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `compute_cover_shadow_metrics` + `main`

`compute_cover_shadow_metrics(df)` returns, per §2 hierarchy:
- `headline_fp_rate`: on **PASS-ONLY** rows (`df[~df["is_cross"]]`) — `false_positive_rate` conditioned on
  `is_completed`, keyed `"majority"` (the `is_blocked_majority` verdict) + `"center"/"mean"/"max"` (thresholded
  `p_blocked_* > 0.5`). Pass-only is the headline because `lane_control` models GROUND-lane screening and
  crosses are aerial (the ball clears ground defenders).
- `pass_plus_cross_secondary`: the same FP rates on ALL rows (pass+cross) — the paper-comparable cut
  (Cascioli scored pass lanes), tagged secondary so it is not read as the headline.
- `optimistic`: `auc(df["is_fail"], df[f"p_blocked_{agg}"])` for the three aggs; `confusion(df["is_blocked_majority"], df["is_fail"])` → recall/specificity/balanced-accuracy; `reliability_slope(df["is_fail"], df["p_blocked_mean"])` — all tagged `"leakage_inflated": True`.
- `recalibration_baseline`: `ece(df["is_fail"], df["p_blocked_mean"])` + `reliability_curve(df["is_fail"], df["p_blocked_mean"])`, tagged as the σ/λ objective + the §9 selection-bias caveat string.
- `paper_reconciliation`: `{"cascioli_majority_recall":0.369,"cascioli_majority_precision":0.220, "required_sentence": "our recall {recall} vs paper 0.369; ..."}`.
- `scope_note`: the "measures OVER-PREDICTION, not DETECTION" sentence (§2).

`main`: argparse (`--pass-scores`, `--out`, `--allow-dirty`); `require_clean_tree` on THIS tree; read the
upstream `manifest.json`, REFUSE if `run_tree_dirty` is True, missing, or `run_commit` differs from a
`--expect-commit` when supplied; compute; write `metrics.json` + a `README.md` that embeds the spec §6
limitations verbatim + the required paper-reconciliation sentence + the scope note. **Both committed outputs
carry the owner-tier GS ship-mask label via `scripts/_corpus.py` (spec §7, F5), and hold only aggregate
rates/AUC — never per-pass positions** (those stay in the gitignored `pass_scores.parquet`, Task 4).

- [ ] **Step 4: Run, expect PASS** + provenance-wiring test green with the new driver enrolled.

---

### Task 6: `validate_pass_risk_calibration.py` — consumer + pass-risk artifact

**Files:**
- Create: `scripts/validate_pass_risk_calibration.py`
- Modify: `tests/scripts/test_provenance_wiring.py` (add `"validate_pass_risk_calibration"`)
- Test: `tests/scripts/test_validate_pass_risk_calibration.py`

**Interfaces:**
- Consumes: `_rq_metrics` (`false_alarm_rate`, `auc`, `ece`, `reliability_slope`, `reliability_curve`,
  `low_control_completion_band`); the persisted `pass_scores.parquet` + manifest.
- Produces: `compute_pass_risk_metrics(df) -> dict`; writes `docs/research/pass_risk_calibration/`.

- [ ] **Step 1: Write the failing test**

```python
def test_headline_is_completed_only_false_alarm_rate():
    import pandas as pd
    from scripts.validate_pass_risk_calibration import compute_pass_risk_metrics
    df = pd.DataFrame({"control":[0.05,0.15,0.5,0.05], "is_completed":[True,True,True,False],
                       "is_success":[True,True,True,False]})
    m = compute_pass_risk_metrics(df)
    assert abs(m["headline_false_alarm_rate"]["0.1"] - 1/3) < 1e-9   # P(control<0.1 | completed)
    assert m["low_control_completion_band"]["contaminated"] is True  # split, all-passes = caveated
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `compute_pass_risk_metrics` (mirror Task 5's hierarchy):
- `headline_false_alarm_rate`: `{str(tau): false_alarm_rate(df["control"], df["is_completed"], tau)}` for
  τ ∈ {0.1,0.2,0.3} — completed-only, the clean headline.
- `optimistic`: `auc(df["is_success"], df["control"])`, `ece`, `reliability_slope`, `reliability_curve` — tagged leakage-inflated.
- `low_control_completion_band`: `low_control_completion_band(df["control"], df["is_success"])` tagged `{"contaminated": True}` + the "distinct from the clean false-alarm headline" note.
- `scope_note`: the same over-prediction-not-detection sentence.
`main`: same shape as Task 5 (upstream-manifest refusal; `require_clean_tree`; write artifact + README).

- [ ] **Step 4: Run, expect PASS** + provenance-wiring green.

---

### Task 7: ADR + docs + full-suite verification

**Files:**
- Create: `docs/superpowers/adrs/ADR-064-cover-shadow-rq1-and-pass-risk-validation.md`
- Modify: `CHANGELOG.md`, `TODO.md` (mark TF-30(b) validation harness as landed; the σ/λ recalibration row
  stays), `CLAUDE.md` (a one-line durable bullet: the two artifacts, the leakage-free-headline rule, the
  build-driver + persisted-table + two-consumers structure).

- [ ] **Step 1: Write ADR-064** recording: the reported-not-gated decision; the leakage-aware metric
  hierarchy (completed-pass anchor leads; failed-pass metrics optimistic; over-prediction-not-detection
  scope); the build-driver/two-consumer structure and why (one corpus pass); the corpus scope = SPADL
  `pass`/`cross` only (spec §4; short set-pieces excluded, F4) with a comparability note that crosses are
  aerial while the lane model is ground-pass-oriented, so the Cascioli reconciliation records whether the
  paper scored ground passes only (and, if it matters, reports a pass-only cut alongside pass+cross); the
  owner-tier ship-mask + gitignored-raw-positions posture (F5); and the two deferrals with homes (σ/λ →
  TF-24 with the M5 selection-bias handoff caveat; Power-2017 receiver → the On-Deck item).
- [ ] **Step 2: CHANGELOG + CLAUDE.md + TODO** entries as above.
- [ ] **Step 3: Run the full CI-faithful gate and read the real exit code:**
  `python -m pytest tests/ -m "not e2e" --benchmark-skip` ; `python -m ruff check silly_kicks/ tests/ scripts/`
  ; `python -m ruff format --check silly_kicks/ tests/ scripts/` ; `python -m pyright`. All green.
- [ ] **Step 4 (owner-run, `@e2e`, not CI):** `python scripts/build_rq_pass_scores.py --out … --shard-root …
  --cache-dir …` on the real GS WC2022 corpus (clean tree), then each `validate_*` consumer against the
  persisted table, producing the two `docs/research/` artifacts with clean-tree provenance.

---

## Self-review

- **Spec coverage:** §2 hierarchy → Tasks 1/5/6 (headline FP/false-alarm, optimistic AUC/slope/recall, ECE
  baseline, scope note). §3 architecture → build-driver + two consumers (deviation flagged). §4 Driver A →
  Tasks 2/3/5. §5 Driver B → Tasks 2/3/6. §6 limitations → embedded verbatim in both READMEs (Tasks 5/6).
  §7 discipline → Tasks 4/5/6 (provenance, `for_each`, `declare_inputs`, ARTIFACT_DRIVERS, PRIVATE_CONSUMERS,
  **and the corpus-visibility ship-mask on committed artifacts + gitignored raw `pass_scores.parquet` — F5**).
  §8 testing → per-task unit tests + non-vacuity in `main` + e2e Task 7. §9 deferrals → ADR-064 (Task 7).
  §10/§11 → Task 7 (ADR, C4-free, no retrain). Q1 (paper-reconciliation sentence) → Task 5. Q2 (3-point band,
  corpus-relative floors) → Tasks 1/6/4.
- **Placeholder scan:** the `_MIN_PASSES`/`_MIN_COMPLETED` values are concrete (20_000 / 12_000, corpus-
  derived); `test_consumer_refuses_dirty_upstream` now has a full body; the three test helpers `_mirror_match`
  / `_acting_attacks_rtl` / `_player_frame_xy` are named with their exact behaviour specified in the Step-3
  implementation notes (mirror = point-reflect + swap teams; attacks_rtl = away team on GS home-attacks-right
  frames; player_frame_xy = the frame row's x/y, fall back to `end_xy` if absent) — concrete, not deferred.
- **Type consistency:** `is_blocked_majority` (bool), `p_blocked_{center,mean,max}` (float), `control`
  (float), `is_completed`/`is_fail`/`is_cross` (bool), `target_source` (str) are used identically across
  Tasks 2–6; `_EMITTED_SHARD_COLUMNS` is the single source the shard, the reconcile, and both consumers key on.
- **Resolved (was an open item):** the `lane_control` coordinate contract is **verified frame-convention** —
  it calls `_validate_ltr` and uses `goal_map`/`attacking_team_id` for lane DIRECTION, not coordinate
  reprojection — so the plan's `to_frame_coords` approach is correct. Task 3's
  `test_lane_scoring_is_orientation_invariant` stays as the backstop (it also catches the stale-docstring risk
  that an internal reflection could key off `attacking_team_id`).

## As-built deviations (recorded at commit-prep, 4.87.0)

The plan is the design of record; four seam details differed in implementation and are pinned here so the
committed plan matches the code:

1. **`link_actions_to_frames` returns `(pointers, LinkReport)` (ADR-004), not a bare DataFrame** — unpacked
   as `links, _ = link_actions_to_frames(...)` in `_rq_corpus`/`build_rq_pass_scores`.
2. **`for_each` result uses `res.shard_dir` + `res.manifest()`** (not `res.generation`/`_write_manifest`);
   `git_provenance()` returns a **dict**, so provenance is stamped as `prov["commit"]`/`prov["dirty"]`.
3. **The Task-3 physical-mirror orientation test was replaced** — reflecting the frame *and* the action
   double-reflects against an asymmetric fixture. The valid checks are a direct **away-pass reprojection**
   test in `test_rq_corpus.py` (`start_x=30 → frame x=75`) plus in-range / non-degenerate `p_blocked`
   assertions in `test_build_rq_pass_scores.py`; the `to_frame_coords` reflection is unit-tested directly.
4. **Module aliases are lowercase `rqm`/`rqc`** (not `M`/`C`) to satisfy `N812`.
