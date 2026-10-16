# Trained-artifact integrity gate + xShot label-filter defect — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Steps use checkbox (`- [ ]`) syntax. **House overrides (non-negotiable):** NO worktrees
> (feature branch `feat/hf-jobs-training-harness` directly); NO subagents; NO per-step / micro-commits —
> the plan ends in **owner-gated commits** of fully-tested coherent states — **two, forced by ADR-037**:
> Commit 1 = code; Commit 2 = the audit verdicts + re-fit weights (producible only on Commit 1's clean
> tree). Not micro-commits (GS-dedup precedent). Do not commit, push, tag,
> bump the version, or spend on HF without Karsten's explicit per-action go.

**Goal:** Fix the xShot-occurrence label-filter defect (silent `type_name`-guard bypass → label counted
*any* action, not just shots), add a layered trained-artifact integrity gate (reproduction tripwire +
construct-validity base-rate band) that the vacuous campaign floor lacked, audit every campaign artifact
against it, and re-fit the one defective model family.

**Architecture:** The fix single-sources the shot-type filter by `type_id` **inside** `build_xshot_labels`
(mirroring `build_xcross_labels`), so no caller can bypass it. The gate is a pure base-rate predicate
(`silly_kicks`) + a corpus-driver verdict harness (`scripts/`, on `scripts/_driver.py`). The audit and the
re-fit are gated DGX/HF runs that consume the gate.

**Tech Stack:** Python 3.10.19 (`.venv`), pandas 2.x/3.x, numpy, xgboost; `scripts/_driver.py` (ADR-052
resume/shard/provenance); `silly_kicks.spadl.config` (`actiontype_id`).

**Spec:** `docs/superpowers/specs/2026-10-15-trained-artifact-integrity-xshot-label-defect-design.md`
(ADR-126, both reviewers APPROVE r2). The plan argues from the spec; executors read both.

## Global Constraints

- **Branch:** `feat/hf-jobs-training-harness` (PR #284). No new branch, no worktree.
- **Owner-gated commits — two, forced by ADR-037 (not micro-commits).** Commit 1 = code (fix + gate +
  driver + tests + ADR + CHANGELOG + docs); all docs travel WITH Commit 1, no separate/early doc commit.
  Commit 2 = audit verdicts + re-fit weights, produced on Commit 1's clean tree (valid `run_tree_dirty:
  false`). Both are fully-tested coherent states (GS-dedup precedent). No per-step/micro-commits.
- **Version `4.138.0` (provisional)** single-sourced in `silly_kicks/_version.py`; **ADR-126 (provisional)**.
  Ladder 2026-10-15: 4.136.0/ADR-124 = ball-contact (merged main), 4.137.0/ADR-125 = gs-dedup (PR #286).
  **Re-confirm next-free ADR + version at the commit gate** (they ride PR #286's merge order).
- **Ids:** route every id comparison/join/dict-key through `silly_kicks.id_compat`; never raw `==`/`str()`
  (ADR-019). The label dict-key tuple in `_build_occurrence_labels` is pre-existing and out of scope here
  (the fix adds no new id comparison).
- **No pandera; schemas are plain dicts.** Vectorize; `warnings.warn(..., stacklevel=2)`.
- **Lint/type at CI scope:** `python -m ruff check silly_kicks/ tests/ scripts/` + `--format --check`;
  `pyright` (local gate = scoped to changed files; CI authoritative).
- **No HF spend > $25; no owner-tier/pining/raw data to HF.** Re-fit uses the staged neutral corpus.
- **Freeze → independent impl review (author does NOT run it) → owner commit-go.**

---

## Review resolution (r1: A APPROVE + B APPROVE; r2: A APPROVE + 1 SHOULD FIX — all addressed)

- **A r2 TAI-PLAN-02 (band floor probe self-contradiction).** r2's Task 2 test asserted
  `base_rate_plausible(0.0005, …) is False` while the band is inclusive and `lo` defaults to `0.0005`
  (`0.0005 ≤ 0.0005` → True) — the probe equalled the floor, so Step 4 "verify PASS" was unreachable. Fixed:
  probe now `0.0001` (strictly below the floor). Band stays inclusive.
- **B r2 observation (test tightness, non-finding).** Added a `0.0450` probe in the `(0.0375, 0.05)` window
  that the NEW derived bound rejects but the OLD fixed `0.05` would have accepted — pins the exact closed
  gap, not just a gross inflation.


- **A TAI-PLAN-01 (commit vs clean-tree provenance).** Correct: ADR-037 drivers refuse a dirty tree, so the
  audit verdict + re-fit weights can only be produced AFTER the code is committed. Replaced "one commit"
  with the forced **two-commit sequence** (code → produce-on-clean-tree → artifacts); GS-dedup precedent,
  not micro-commits. See Global Constraints, Task 6 header, Final Steps 3-6.
- **B TAI-PLAN-01 (band value).** Correct plan↔spec drift: r1 hard-coded `(0.003, 0.05)`; `0.05` was ~2× the
  physical `~0.025`, so a 2–3× inflation would slip. Task 2 now **derives** the upper bound physically
  (`physical_upper_bound = tol · n_events · horizon · fps / n_rows`, `tol=1.5`), per spec §4.2; the test
  asserts a ~2.4× inflation (0.06) FAILS. `tol` justified in ADR-126 (Task 4).

---

## File Structure

| File | Responsibility |
|---|---|
| `silly_kicks/tracking/_xshot_occurrence.py` | **Modify** — `build_xshot_labels` gains the `type_id` shot filter; `prepare_xshot_training_data` loses the `type_name` guard. |
| `silly_kicks/tracking/_artifact_integrity.py` | **Create** — pure base-rate-band predicate + per-model bands (I2). |
| `scripts/validate_trained_artifact_integrity.py` | **Create** — verdict corpus-driver: reproduction tripwire (I3) + band + acceptance, per model family; audit entry (I4). |
| `tests/tracking/test_xshot_occurrence.py` | **Modify** — add I1 detection test; add `type_id` to 5 `test_label_*` fixtures + the R2-L2 golden-master. |
| `tests/tracking/test_artifact_integrity.py` | **Create** — I2 band unit. |
| `tests/scripts/test_validate_trained_artifact_integrity.py` | **Create** — I3 tripwire unit (CI-safe slice). |
| `docs/superpowers/adrs/ADR-126-trained-artifact-integrity-xshot-label-filter.md` | **Create** — the ADR. |
| `CHANGELOG.md`, `AGENTS.md`, `docs/PRIVATE_CONSUMERS.md` | **Modify** — release entry, convention/public-surface note, downstream consumers. |

---

## Task 1: xShot label-filter fix + caller fixtures (the defect)

**Files:**
- Modify: `silly_kicks/tracking/_xshot_occurrence.py` (`build_xshot_labels` ~289-322; `_DEFAULT_SHOT_TYPES`
  :326; `prepare_xshot_training_data` filter :917-922; imports)
- Modify: `tests/tracking/test_xshot_occurrence.py` (fixtures :233/253/263/272/282 + golden-master :843;
  new detection test)

**Interfaces:**
- Produces: `build_xshot_labels(frames_index, shots, *, horizon_seconds=1.0, shot_types=_DEFAULT_SHOT_TYPES)`
  — now filters `shots` to `type_id ∈ {actiontype_id[t] for t in shot_types}` internally. Requires `shots`
  to carry `type_id` (SPADL always does; `shots.parquet` does).

- [ ] **Step 1: Write the failing detection test (I1).** Add to `test_xshot_occurrence.py`:

```python
def test_build_xshot_labels_filters_to_shots_on_typeid_only_actions():
    """I1 (detection): fed a full actions table with type_id but NO type_name,
    build_xshot_labels must count ONLY shots, not every action."""
    import pandas as pd
    from silly_kicks.spadl import config as _spc
    shot_id = _spc.actiontype_id["shot"]
    pass_id = _spc.actiontype_id["pass"]
    fidx = pd.DataFrame({"game_id": [1, 1], "period_id": [1, 1],
                         "time_seconds": [9.5, 9.5], "team_in_possession": [2, 2]})
    # Two same-team events at t=10.0 within the 1s window: one shot, one pass. Only the shot counts.
    actions = pd.DataFrame({
        "game_id": [1, 1], "period_id": [1, 1], "team_id": [2, 2],
        "time_seconds": [10.0, 10.0], "type_id": [shot_id, pass_id],
    })  # NO type_name column
    y = xs.build_xshot_labels(fidx, actions, horizon_seconds=1.0)
    assert list(y) == [1, 1]            # both fidx rows see the shot
    # and the pass-only counterfactual labels nothing:
    passes = actions[actions["type_id"] == pass_id]
    y2 = xs.build_xshot_labels(fidx, passes, horizon_seconds=1.0)
    assert list(y2) == [0, 0]
```

- [ ] **Step 2: Run it — verify it FAILS on `ac875c4`.**
  `python -m pytest tests/tracking/test_xshot_occurrence.py::test_build_xshot_labels_filters_to_shots_on_typeid_only_actions -v`
  Expected: FAIL — current `build_xshot_labels` does not filter (the pass counts → `y2 == [1,1]`, or a
  `KeyError` once Step 4 lands). This is the detection-before-fix evidence.

- [ ] **Step 3: Add the spadl-config import + move the constant.** In `_xshot_occurrence.py`: add
  `from silly_kicks.spadl import config as _spc` to the import block (mirror `_xcross_attempt.py:23`); if an
  equivalent import already exists, reuse it. Move `_DEFAULT_SHOT_TYPES = ("shot", "shot_penalty",
  "shot_freekick")` from its current line (~326) to **above** `build_xshot_labels` (so it is a valid default
  arg).

- [ ] **Step 4: Implement the `type_id` filter inside `build_xshot_labels`.** Replace the body:

```python
def build_xshot_labels(
    frames_index: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    horizon_seconds: float = 1.0,
    shot_types: tuple[str, ...] = _DEFAULT_SHOT_TYPES,
) -> pd.Series:
    # ... docstring: note `shots` must carry `type_id`; filter mirrors build_xcross_labels ...
    type_ids = {_spc.actiontype_id[t] for t in shot_types}
    shots_f = shots[shots["type_id"].isin(type_ids)][
        ["game_id", "period_id", "team_id", "time_seconds"]
    ]
    y = _build_occurrence_labels(
        frames_index, shots_f, horizon=horizon_seconds, frame_team_col="team_in_possession"
    )
    return pd.Series(y, index=frames_index.index)
```

  Update the docstring `shots` param to state the `type_id` requirement.

- [ ] **Step 5: Delete the fragile guard in `prepare_xshot_training_data`.** Replace the `:917-921` block:

```python
    shots_f = shots
    if "type_name" in shots.columns:
        shots_f = shots[shots["type_name"].isin(types)]
    elif not types:
        shots_f = shots.iloc[0:0]
    labels = build_xshot_labels(fidx, shots_f, horizon_seconds=horizon_seconds).to_numpy()
```

  with (the filter now lives in `build_xshot_labels`; pass the raw table + `shot_types`):

```python
    labels = build_xshot_labels(
        fidx, shots, horizon_seconds=horizon_seconds, shot_types=types
    ).to_numpy()
```

  (`types` is already `_DEFAULT_SHOT_TYPES if shot_types is None else tuple(shot_types)` at
  `prepare_xshot_training_data:842` — keep that line.)

- [ ] **Step 6: Update the 5 `test_label_*` fixtures (§4.1.1).** In each of the `shots = pd.DataFrame({...})`
  at lines ~233, 253, 263, 272, 282, add `"type_id": [11]` (= `actiontype_id["shot"]`; one row each). Do not
  change the `assert list(y) == …` lines — all fixture rows are shots, so the filter keeps them and the
  expectations hold. (Prefer `"type_id": [_spc.actiontype_id["shot"]]` with a top-of-test import to avoid a
  magic 11; a literal `11` is acceptable if the module already hard-codes it elsewhere.)

- [ ] **Step 7: Update the R2-L2 golden-master fixture.** In
  `test_build_xshot_labels_bit_identical_after_refactor` (shots fixture at :843
  `{"game_id": ["g"], "period_id": [1], "team_id": ["A"], "time_seconds": [1.2]}`), add
  `"type_id": [11]`. The frozen golden labels are unchanged (the single row is a shot → kept).

- [ ] **Step 8: Run the xShot label tests + the detection test.**
  `python -m pytest tests/tracking/test_xshot_occurrence.py -v`
  Expected: ALL pass, including the Step-1 detection test (now PASS) and the golden-master (bit-identical).

---

## Task 2: Base-rate plausibility band (I2)

**Files:**
- Create: `silly_kicks/tracking/_artifact_integrity.py`
- Create: `tests/tracking/test_artifact_integrity.py`

**Resolves (plan-r2) B TAI-PLAN-01:** the band is **physically DERIVED per artifact** from its own
metadata, not a loose hard-coded constant. Spec §4.2 says "an upper bound derived from
`n_events · horizon · fps / n_rows`"; r1 hard-coded `(0.003, 0.05)` whose `0.05` was ~2× the physical
`~0.025` for the xShot corpus, so a 2–3× inflation would slip. The derived bound closes that.

**Interfaces:**
- Produces: `physical_upper_bound(n_events, horizon_s, fps, n_rows, *, tol=1.5) -> float`
  (`= tol * n_events * horizon_s * fps / n_rows`) and
  `base_rate_plausible(positive_rate, *, n_events, horizon_s, fps, n_rows, lo=0.0005, tol=1.5) -> bool`.
  The physical maximum positive rate for an occurrence label is `n_events · (fps·horizon) / n_rows`
  (max positive frames over extracted rows); `tol` (default 1.5) is the only slack. For a mixed-fps corpus
  the harness passes the effective `fps = n_rows / total_attacking_third_possession_seconds` (equivalently
  `hi = tol · n_events · horizon / duration_s`), so the bound is fps-robust. `MODEL_INTEGRITY_TOL:
  dict[str, float]` holds the per-family `tol` (default 1.5), the only tunable — justified in ADR-126.

- [ ] **Step 1: Write the failing test** (uses the real xShot numbers: 21 shots, 10fps, 1s, 8394 rows →
  physical max ≈ 0.025, derived `hi = 1.5 × 0.025 ≈ 0.0375`):

```python
def test_base_rate_plausible_derived_bound():
    from silly_kicks.tracking._artifact_integrity import base_rate_plausible, physical_upper_bound
    kw = dict(n_events=21, horizon_s=1.0, fps=10.0, n_rows=8394)   # skillcorner-1925299
    assert abs(physical_upper_bound(**kw, tol=1.5) - 0.0375) < 1e-3
    assert base_rate_plausible(0.0154, **kw) is True     # correct xShot (below physical max)
    assert base_rate_plausible(0.2220, **kw) is False    # the 13x defect
    assert base_rate_plausible(0.0600, **kw) is False    # a ~2.4x inflation still fails (B's gap closed)
    assert base_rate_plausible(0.0450, **kw) is False    # in (0.0375, 0.05): DERIVED bound catches it; the
                                                         # old fixed 0.05 would have PASSED -> pins the gap
    assert base_rate_plausible(0.0001, **kw) is False    # strictly below the lo=0.0005 floor (inclusive band)
```

- [ ] **Step 2: Run — verify FAIL** (`ModuleNotFoundError`).
  `python -m pytest tests/tracking/test_artifact_integrity.py -v`
- [ ] **Step 3: Implement** `_artifact_integrity.py`:

```python
"""Trained-artifact integrity predicates (ADR-126). Pure; no I/O."""
from __future__ import annotations

#: Only tunable: slack over the PHYSICAL maximum positive rate. Justified in ADR-126.
MODEL_INTEGRITY_TOL: dict[str, float] = {"xshot_occurrence": 1.5, "xcross_attempt": 1.5}

def physical_upper_bound(n_events, horizon_s, fps, n_rows, *, tol=1.5) -> float:
    """Max plausible per-row positive rate for an occurrence label, with slack.

    Physical max = n_events * (fps * horizon_s) / n_rows (max positive frames / rows).
    Pass fps = n_rows / total_duration_s for a mixed-fps corpus (then it reads
    tol * n_events * horizon_s / duration_s, fps-independent)."""
    if n_rows <= 0:
        return float("inf")
    return tol * n_events * horizon_s * fps / n_rows

def base_rate_plausible(positive_rate, *, n_events, horizon_s, fps, n_rows,
                        lo=0.0005, tol=1.5) -> bool:
    """True iff lo <= positive_rate <= physical_upper_bound(...) (NaN-safe: NaN -> False)."""
    hi = physical_upper_bound(n_events, horizon_s, fps, n_rows, tol=tol)
    return bool(positive_rate == positive_rate and lo <= positive_rate <= hi)
```

- [ ] **Step 4: Run — verify PASS.** Same command. The verdict harness (Task 3) supplies `n_events`
  (shots/crosses count), `fps`, `n_rows` from the artifact's own extraction/metadata, so the bound tracks
  the corpus instead of a fixed constant.

---

## Task 3: Reproduction tripwire + verdict harness (I3) + audit entry (I4)

**Files:**
- Create: `scripts/validate_trained_artifact_integrity.py`
- Create: `tests/scripts/test_validate_trained_artifact_integrity.py`

**Interfaces:**
- Consumes: Task 2's `base_rate_plausible` / `physical_upper_bound` / `MODEL_INTEGRITY_TOL`; `scripts/_driver.py`
  (`for_each`/provenance); each family's shipped `metrics.json`.
- Produces: `reproduces(shipped: dict, fresh: dict, *, rel_tol: float = 0.05) -> bool` and a
  `verdict(family, shipped, fresh) -> dict` recording `reproduced`, `base_rate_ok`, `lift_ok`,
  `calibration_ok`, `enough_folds`, and provenance (`run_commit`, `run_tree_dirty: False`, `input_contract`).

- [ ] **Step 1: Write the failing tripwire unit.**

```python
def test_reproduces_passes_on_match_fails_on_perturbation():
    from scripts.validate_trained_artifact_integrity import reproduces
    shipped = {"n_rows": 156106, "n_positive": 2405,
               "candidates": {"public": {"metrics": {"pr_auc": 0.0824, "positive_rate": 0.0154}}}}
    fresh_ok = {"n_rows": 156106, "n_positive": 2405,
                "candidates": {"public": {"metrics": {"pr_auc": 0.0831, "positive_rate": 0.0154}}}}
    fresh_bad = {"n_rows": 156106, "n_positive": 34649,
                 "candidates": {"public": {"metrics": {"pr_auc": 0.3418, "positive_rate": 0.2220}}}}
    assert reproduces(shipped, fresh_ok, rel_tol=0.05) is True
    assert reproduces(shipped, fresh_bad, rel_tol=0.05) is False   # the committed defect vs correct re-fit
```

- [ ] **Step 2: Run — verify FAIL** (module/function absent).
  `python -m pytest tests/scripts/test_validate_trained_artifact_integrity.py -v`
- [ ] **Step 3: Implement `reproduces` + `verdict`** in `scripts/validate_trained_artifact_integrity.py`:
  compare `n_rows` (exact), `n_positive` (exact) and the shipped candidate `pr_auc`/`positive_rate` within
  `rel_tol`; `verdict` composes `reproduced` with `base_rate_plausible(positive_rate, n_events=, fps=,
  n_rows=, horizon_s=)` (n_events/fps/n_rows read from the artifact's extraction + metadata),
  `pr_auc > positive_rate` (lift),
  `brier < base_rate_brier`, `n_usable_folds >= min`. Emit the provenance block
  (`require_clean_tree(git_provenance())`, ADR-037; `declare_inputs`/`input_contract` digest, ADR-056).
- [ ] **Step 4: Run — verify PASS.** Same command.
- [ ] **Step 5: Add the full-corpus audit entry point** (`main(argv)`) over the campaign families, built on
  `scripts/_driver.py` `for_each` (resume/shards). Mark the real-corpus pass `@pytest.mark.slow` / opt-in
  `e2e` (ADR-023 — one dedicated job, never inline). A CI-safe unit slice (Steps 1-4) is the gate's teeth
  in the fast suite.

---

## Task 4: ADR-126 + CHANGELOG + public-surface notes

**Files:**
- Create: `docs/superpowers/adrs/ADR-126-trained-artifact-integrity-xshot-label-filter.md`
- Modify: `CHANGELOG.md`, `AGENTS.md`, `NOTICE` (if a published method is cited), `docs/context/*` pointer

- [ ] **Step 1: Write ADR-126** documenting: the defect (silent `type_name`-guard bypass, reproduced to the
  integer 1449 vs 109; physical-bound proof), the fix (single-sourced `type_id` filter in
  `build_xshot_labels`), the layered gate (reproduction tripwire + base-rate band — why
  `pr_auc_gt_base_rate` alone missed it), the audit (I4), the Hyrum impact (xShot output 22%→1.5%), and the
  base-rate-band justification — specifically **why the bound is physically DERIVED** (`tol · n_events ·
  horizon · fps / n_rows`) rather than a fixed constant, and why `tol=1.5` (slack over the physical max; a
  2–3× inflation still fails). Also record the **forced two-commit sequence** (ADR-037 clean-tree
  provenance: code commit → produce audit/weights on the clean tree → artifact commit). Header:
  **provisional number — re-confirm at commit**.
- [ ] **Step 2: CHANGELOG `[Unreleased]`** — a `fix(tracking)` entry (defect + corrected label) plus the
  gate/audit, keyed per house style. Note version is provisional (`4.138.0`, re-confirm).
- [ ] **Step 3: AGENTS.md** — if `base_rate_plausible`/`physical_upper_bound` or the integrity gate is a new
  durable contract, add a Key-conventions bullet (≤600 chars, `ADR-126` ref) and respect the budget test
  (`tests/test_agents_md_budget.py`). Add `_artifact_integrity.py` to the tracking surface map if public.
- [ ] **Step 4: Register any new public module** in `tests/test_public_api_examples.py` (house rule: a new
  public module MUST register) and ensure doctests (`--doctest-modules`) pass.

---

## Task 5: Downstream consumer audit (§5, library-only)

**Files:**
- Modify: `docs/PRIVATE_CONSUMERS.md`

- [ ] **Step 1: Enumerate** in-repo consumers of the xShot output (`xshot_occurrence` column /
  `add_xshot_occurrence` / `compute_xshot_occurrence` / `XShotOccurrenceModel`), incl. GKDV Layer-2, via
  repo grep. Record each + the behavioural change (output distribution 22%→1.5%) so re-fit consumers are
  on notice. **Lakehouse re-materialization is explicitly out of scope** (owner 2026-10-15).

---

## Task 6 (GATED RUN — owner go + ≤ $25): audit all campaign artifacts + re-fit xShot

> Not code. A DGX/HF run that consumes Tasks 1-3. Gated on: all code tasks green, independent review passed,
> owner go, and the $25 ceiling. Do NOT start without the explicit go.
>
> **Clean-tree prerequisite (resolves A TAI-PLAN-01).** The audit driver and the re-fit trainer both call
> `require_clean_tree(git_provenance())` (ADR-037) and refuse / stamp `run_tree_dirty: true` on an
> uncommitted tree. Their outputs (verdict `metrics.json`, re-fit weights) are therefore produced **after
> Commit 1 (code)** against its clean tree, and land in **Commit 2 (artifacts)**. This is a forced
> two-commit sequence (GS-dedup precedent), NOT micro-commits — both are fully-tested coherent states. It
> feeds the Final-Step-4 owner structure decision.

- [ ] **Step 1:** Run `validate_trained_artifact_integrity` (Task 3) against each committed campaign
  artifact: `xshot_occurrence` (default+position_only), `xcross_attempt`, `receiver_model`, `ghost_gk` ×4,
  `ghost_outfield`, the 4 public-wave trainers. Record GO/NO-GO per `docs/howto/construct-validity.md`
  (verdict `metrics.json` + provenance). Expected: xShot NO-GO (pre-fix), xCross GO; others per result.
- [ ] **Step 2:** Re-fit xShot (default + position_only) on the staged neutral corpus via
  `--data-dir`/`--arms` (correct — `shots.parquet` carries `type_id`; and now `build_xshot_labels` is
  robust regardless). Certify with the gate (reproduces the *corrected* ~1.5%/≈5× lift; base-rate band
  PASS). Weights land in **Commit 2 (artifacts)**, produced against Commit 1's clean tree.
- [ ] **Step 3:** For any other NO-GO the audit surfaces, STOP and surface to the owner (scope to extend is
  the owner's) — do not silently fix or defer.

---

## Final: review handoff + owner-gated commits (two, forced by ADR-037)

- [ ] **Step 1:** Full suite green (`-m "not e2e"`), `ruff`/`pyright` clean at CI scope.
- [ ] **Step 2:** FREEZE the tree. Hand to an independent impl review (author does NOT run it).
- [ ] **Step 3 — Commit 1 (code), owner-gated:** on review pass + owner go, re-confirm next-free ADR +
  version at HEAD; bump `_version.py` → the confirmed number (`4.138.0` provisional); assemble ONE commit
  of the code change (Tasks 1-5: fix + gate module + driver + tests + ADR-126 + CHANGELOG + docs +
  PRIVATE_CONSUMERS). Present the diff/file list; **wait for the explicit per-commit yes.** Trailer:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`; no `Claude-Session:` trailer.
- [ ] **Step 4 — Task 6 run (owner go + ≤$25):** on Commit 1's clean tree, run the audit + re-fit (Task 6).
  Outputs carry `run_tree_dirty: false` because the code is committed. Any NO-GO → STOP + surface.
- [ ] **Step 5 — Commit 2 (artifacts), owner-gated:** assemble the audit verdict `metrics.json` + re-fit
  xShot weights (+ a CHANGELOG note of the audit outcomes). Present the diff/file list; **wait for the
  explicit per-commit yes.** This is the second fully-tested coherent state (GS-dedup precedent), NOT a
  micro-commit.
- [ ] **Step 6 — DEFERRED owner structure decision:** confirm at the Commit-1 gate: (a) whether the
  xShot-integrity code bundles with the visibility-blind harness work or is its own commit, and (b) which
  commit carries the `4.138.0` version bump (code vs artifacts). The two-commit code→artifact split is
  forced regardless; this decision only fixes the code-side grouping + version placement.

---

## Self-Review (checklist run against the spec)

- **Spec coverage:** §4.1 fix → Task 1; §4.1.1 caller sweep → Task 1 Steps 6-7; I2 band → Task 2; I3
  tripwire → Task 3; §4.2 verdict + §4.3 audit → Tasks 3/6; §4.4 re-fit → Task 6; §5 downstream → Task 5;
  ADR/CHANGELOG/version → Task 4 + Final; detection-before-fix → Task 1 Steps 1-2, Task 2/3 failing-first.
- **Placeholder scan:** every code step shows actual code or an exact command; no TBD/"handle edge cases".
- **Type/name consistency:** `build_xshot_labels(..., shot_types=_DEFAULT_SHOT_TYPES)`, `_spc.actiontype_id`,
  `base_rate_plausible`, `physical_upper_bound`, `MODEL_INTEGRITY_TOL`, `reproduces`, `verdict` used
  consistently across tasks.
- **House rules:** two owner-gated commits forced by ADR-037 (code, then artifacts — GS-dedup precedent,
  not micro-commits); no worktrees/subagents; provisional
  ADR/version re-confirmed at commit; ids via `id_compat`; `@pytest.mark.slow` for the real-corpus audit.
