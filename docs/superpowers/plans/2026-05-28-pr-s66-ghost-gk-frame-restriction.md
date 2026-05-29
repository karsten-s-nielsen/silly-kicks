# Ghost-GK Linked-Frame Restriction (PR-S66) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restrict the expensive per-sample ghost-GK KDE (`predict_density`) to action-linked frames while keeping feature extraction over the full frames, making `add_ghost_gk` / `ghost_gk_xfns` materially faster with byte-identical output.

**Architecture:** `_extract_all_ghost_gk_features` stays full-frame (preserving the two cross-frame deps: per-period defending-goal mean-x, and the cross-period one-step velocity state). A new keyword-only `link_frame_ids` on `compute_ghost_gk` restricts only `batch_features`/`meta` before the per-sample (cross-sample-independent) KDE. `add_ghost_gk` derives the set from its pointers; `ghost_gk_xfns` derives the union over its three gamestate slots.

**Tech Stack:** Python 3.10+ (CI 3.10–3.12, local 3.14), pandas, numpy, scikit-learn; pytest. Spec: `docs/superpowers/specs/2026-05-28-pr-s66-ghost-gk-frame-restriction-design.md`.

---

## Project conventions (read before starting)

- **ONE commit per branch, explicit approval first.** The per-task TDD steps below say "verify" — they do **not** commit. All staging + the single commit happen only in the final task, after `/final-review` and explicit user approval. Branch `pr-s66-ghost-gk-frame-restriction` is already created off `main`.
- **Shift-left gates** (run before declaring done, Task 7): `ruff format --check .`, `ruff check .`, `pyright silly_kicks/` (full package, never just changed files), and `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- **Function-local imports:** `add_ghost_gk` and `ghost_gk_xfns` import `compute_ghost_gk` via `from ._ghost_gk import …` *inside* the function. To spy/patch it in tests, patch the **source** `silly_kicks.tracking._ghost_gk.compute_ghost_gk`, not a `features` attribute.
- **No new columns, no new ADR, no NOTICE entry** (no new methodology). Version → **3.26.0** (minor: adds the public `link_frame_ids` kwarg).

## File structure

- Modify: `silly_kicks/tracking/_ghost_gk.py` — `compute_ghost_gk` gains `link_frame_ids` + the post-extraction restriction (Task 1).
- Modify: `silly_kicks/tracking/features.py` — `add_ghost_gk` derives+passes `link_frame_ids` (Task 2); `ghost_gk_xfns` union-of-slots restriction (Task 3).
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py` — NEW file for all PR-S66 unit + golden + e2e tests (Tasks 1–4). Keeps the restriction suite cohesive and discoverable; imports shared helpers from `tests/tracking/test_ghost_gk.py`.
- Modify (Task 8): `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`.
- No change needed: `silly_kicks/atomic/tracking/features.py` (re-exports `add_ghost_gk`/`ghost_gk_xfns` from the main module — inherits the fix). Task 4 includes a guard test asserting the re-export, not a duplicate impl.

---

### Task 1: `compute_ghost_gk` — `link_frame_ids` restriction (primitive)

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`compute_ghost_gk`, currently ~lines 1367–1454)
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py` (new)

- [ ] **Step 1: Write the failing tests**

Create `tests/tracking/test_ghost_gk_frame_restriction.py`:

```python
"""PR-S66 — ghost-GK linked-frame restriction (KDE-only, bit-identical)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking._ghost_gk import compute_ghost_gk
from tests.tracking.test_ghost_gk import _fitted_model, _make_ghost_gk_frames

_GHOST_COLS = ["ghost_gk_x", "ghost_gk_y", "ghost_gk_spread"]


def _make_goal_flip_velocity_fixture(home_team_id: int = 1, away_team_id: int = 2):
    """5 frames, 1 period, engineered to exercise BOTH cross-frame deps.

    - Goal-flip dep: home GK sits at x=5 in frames 1-4 (full-period mean stays
      < 52.5 -> defends x=0), but is camped at x=60 in the lone linked frame 5
      (frame-5-alone mean >= 52.5 would flip the inferred goal to x=105).
    - Velocity dep: the home defensive line shifts every frame, so frame 5's
      one-step velocity (vs the real frame-4 predecessor) differs from a
      no-predecessor compute.

    Returns (frames, linked_frame_ids).
    """
    parts = []
    for fid in range(1, 6):
        f = _make_ghost_gk_frames(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            frame_id=fid,
            timestamp=float(fid) * 0.04,
        )
        defmask = (
            (f["team_id"] == home_team_id)
            & ~f["is_goalkeeper"].astype(bool)
            & ~f["is_ball"].astype(bool)
        )
        f.loc[defmask, "x"] = f.loc[defmask, "x"] + fid * 2.0  # moving back line
        if fid == 5:
            gkmask = (f["team_id"] == home_team_id) & f["is_goalkeeper"].astype(bool)
            f.loc[gkmask, "x"] = 60.0  # camp high only in the linked frame
        parts.append(f)
    return pd.concat(parts, ignore_index=True), {5}


def _linked_gk_rows(result: pd.DataFrame, link_frame_ids: set) -> pd.DataFrame:
    mask = (
        result["is_goalkeeper"].astype(bool)
        & ~result["is_ball"].astype(bool)
        & result["frame_id"].astype(int).isin(link_frame_ids)
    )
    return result.loc[mask].sort_values(["frame_id", "team_id"]).reset_index(drop=True)


class TestComputeGhostGkRestriction:
    def test_full_vs_restricted_bit_identical(self):
        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()

        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        restricted = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)

        f_rows = _linked_gk_rows(full, linked)
        r_rows = _linked_gk_rows(restricted, linked)
        assert len(f_rows) == len(r_rows) > 0
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(f_rows[col].to_numpy(), r_rows[col].to_numpy())

    def test_naive_prefilter_discriminates(self):
        """Proves the fixture actually triggers the cross-frame deps: dropping
        unlinked frames BEFORE extraction must change BOTH the extracted features
        (the deps' direct effect) and — through the model — the ghost output.
        The FEATURE-level assertion (M2) pins the fixture to the mechanism
        regardless of how feature-sensitive _fitted_model() is, so the check can't
        pass/fail vacuously on model insensitivity."""
        from silly_kicks.tracking._ghost_gk import _extract_all_ghost_gk_features

        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()
        fid = next(iter(linked))

        # --- Feature-level: the linked frame's extracted features MUST differ ---
        feat_full, meta_full = _extract_all_ghost_gk_features(frames, home_team_id=1)
        feat_naive, meta_naive = _extract_all_ghost_gk_features(
            frames[frames["frame_id"].astype(int).isin(linked)], home_team_id=1
        )
        row_full = feat_full[meta_full["frame_id"].astype(int) == fid].reset_index(drop=True)
        row_naive = feat_naive[meta_naive["frame_id"].astype(int) == fid].reset_index(drop=True)
        assert len(row_full) == len(row_naive) > 0
        assert not np.allclose(
            row_full.to_numpy(dtype=float), row_naive.to_numpy(dtype=float), equal_nan=True
        ), "fixture does not change extracted features; cross-frame deps not exercised"

        # --- Output-level: the difference propagates through the model ---
        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        naive = compute_ghost_gk(
            frames[frames["frame_id"].astype(int).isin(linked)],
            model=model,
            home_team_id=1,
        )
        f_rows = _linked_gk_rows(full, linked)
        n_rows = _linked_gk_rows(naive, linked)
        differs = any(
            not np.allclose(f_rows[col].to_numpy(), n_rows[col].to_numpy(), equal_nan=True)
            for col in _GHOST_COLS
        )
        assert differs

    def test_restriction_shrinks_predict_set(self, monkeypatch):
        model, _, _ = _fitted_model()
        frames, linked = _make_goal_flip_velocity_fixture()

        captured: list[int] = []
        orig = model.predict_density

        def spy(features):
            captured.append(len(features))
            return orig(features)

        monkeypatch.setattr(model, "predict_density", spy)
        compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)
        restricted_n = captured[-1]
        captured.clear()
        compute_ghost_gk(frames, model=model, home_team_id=1)  # full
        full_n = captured[-1]

        assert restricted_n < full_n
        assert restricted_n == 2 * len(linked)  # 2 GKs (home + away) per linked frame

    def test_link_frame_ids_none_unchanged(self):
        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        a = compute_ghost_gk(frames, model=model, home_team_id=1)
        b = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=None)
        gk = a["is_goalkeeper"].astype(bool) & ~a["is_ball"].astype(bool)
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(a.loc[gk, col].to_numpy(), b.loc[gk, col].to_numpy())
```

**Signature note (C1, verified — read before running).**
`_extract_all_ghost_gk_features(frames, *, home_team_id, carrier=None,
score_at_time=None, phase_at_time=None, subsample_fps=None)` — `score_at_time` and
`phase_at_time` **default to `None`**, so the 2-arg call in
`test_naive_prefilter_discriminates` does NOT `TypeError`. It is apples-to-apples
with that test's `compute_ghost_gk(..., actions=None)` path, which also passes
`score_fn=None, phase_fn=None`; so the only full-vs-naive feature differences are
the goal-relative-coord flip + velocity (score/phase are frame-independent zeros
in both). `row_*.to_numpy(dtype=float)` is safe — all 26 `GHOST_GK_FEATURE_NAMES`
are numeric (they feed the KDE). Re-confirm with
`grep -n "def _extract_all_ghost_gk_features" silly_kicks/tracking/_ghost_gk.py`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -v`
Expected: FAIL — `compute_ghost_gk() got an unexpected keyword argument 'link_frame_ids'`.

(Note: the fixture has 2 GKs/frame × 5 frames = 10 full samples; restricted to frame 5 = 2 samples. Both the Task 1 and Task 4 predict-count asserts use `2 * len(linked)` — consistent.)

- [ ] **Step 3: Implement the restriction in `compute_ghost_gk`**

In `silly_kicks/tracking/_ghost_gk.py`, change the signature and add the restriction after extraction. Replace:

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

with (add the kwarg + a Parameters line in the docstring):

```python
def compute_ghost_gk(
    frames: pd.DataFrame,
    *,
    model: GhostGkModel | GhostGkVariant | None = None,
    home_team_id: int | str,
    actions: pd.DataFrame | None = None,
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
```

Add to the docstring Parameters block (above Returns):

```
    link_frame_ids : set[int] | None, default None
        When provided, restrict the per-sample KDE (`predict_density`) to GK
        samples whose ``frame_id`` is in this set. Feature extraction still runs
        over the FULL frames, so the per-period defending-goal mean-x and the
        cross-period one-step velocity state are preserved exactly — the KDE is
        per-sample independent, so the restricted result is byte-identical to the
        unrestricted one for the kept frames. When None, every sample is predicted
        (backward-compatible). See PR-S66 spec §2-§3.
```

Then locate the extraction + predict block:

```python
    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
    )

    if len(batch_features) == 0:
        return out

    # Batch predict
    densities = resolved.predict_density(batch_features)
```

and replace with (C3 breadcrumb comment on the extraction call + the restriction):

```python
    # NOTE (PR-S66): extraction runs over the FULL frames by design — the
    # per-period defending-goal mean-x and the cross-period one-step velocity
    # state are cross-frame dependencies, so pre-filtering frames here would NOT
    # be bit-identical. Only the per-sample KDE below is restricted. If a future
    # measurement shows this full-frame extraction is itself a bottleneck, see
    # spec §5 for the precompute-and-inject variant (TODO in TODO.md).
    batch_features, meta = _extract_all_ghost_gk_features(
        frames,
        home_team_id=home_team_id,
        score_at_time=score_fn,
        phase_at_time=phase_fn,
    )

    if len(batch_features) == 0:
        return out

    # Linked-frame restriction: keep only samples on action-linked frames before
    # the expensive per-sample KDE. predict_density has zero cross-sample coupling,
    # so this is byte-identical to predicting all samples and dropping the rest.
    if link_frame_ids is not None:
        keep = meta["frame_id"].astype(int).isin(link_frame_ids).to_numpy()
        batch_features = batch_features[keep].reset_index(drop=True)
        meta = meta[keep].reset_index(drop=True)
        if len(batch_features) == 0:
            return out

    # Batch predict
    densities = resolved.predict_density(batch_features)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -v`
Expected: PASS (4 tests).

**Merge confirm (verified):** the post-predict merge is already
`gk_rows_df.merge(result_df, on=[...], how="left")` and `out` initializes the
three ghost columns to NaN, so a restricted `result_df` (linked frames only) adds
values to the linked GK rows and leaves the rest NaN — no rows dropped. The merge
block is unchanged by this task; confirm `how="left"` is intact during Step 3.

---

### Task 2: `add_ghost_gk` — derive `link_frame_ids` from pointers

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`add_ghost_gk`, currently ~lines 3518–3616)
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py`

- [ ] **Step 1: Write the failing tests**

Append to the test file:

```python
class TestAddGhostGkRestriction:
    def _make_actions(self, frame_ids=(5,)):
        # One shot action per linked frame; defending GK is the AWAY team.
        rows = []
        for k, fid in enumerate(frame_ids):
            rows.append(
                {
                    "action_id": k,
                    "game_id": "100",
                    "period_id": 1,
                    "time_seconds": float(fid) * 0.04,
                    "team_id": 1,  # home attacks -> away GK is the ghost target
                    "player_id": "p99",
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "type_name": "shot",
                    "result_name": "fail",
                    "bodypart_name": "foot",
                }
            )
        return pd.DataFrame(rows)

    def test_add_ghost_gk_passes_pointers_frame_ids(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        actions = self._make_actions(frame_ids=(5,))

        captured: dict = {}
        real = ghost_mod.compute_ghost_gk

        def spy(frames_arg, **kwargs):
            captured["link_frame_ids"] = kwargs.get("link_frame_ids")
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", spy)
        add_ghost_gk(actions, frames, model=model, home_team_id=1)

        # Internally-computed pointers (no links kwarg) still drive restriction.
        assert captured["link_frame_ids"] is not None
        assert captured["link_frame_ids"] <= {1, 2, 3, 4, 5}

    def test_add_ghost_gk_output_unchanged_by_restriction(self, monkeypatch):
        """Action-coupled columns identical whether the internal compute is
        restricted (real path) or not (forced full via patched kwarg-strip)."""
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import add_ghost_gk

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        actions = self._make_actions(frame_ids=(5,))

        restricted = add_ghost_gk(actions, frames, model=model, home_team_id=1)

        real = ghost_mod.compute_ghost_gk

        def force_full(frames_arg, **kwargs):
            kwargs["link_frame_ids"] = None
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", force_full)
        full = add_ghost_gk(actions, frames, model=model, home_team_id=1)

        for col in _GHOST_COLS:
            np.testing.assert_array_equal(
                restricted[col].to_numpy(), full[col].to_numpy()
            )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py::TestAddGhostGkRestriction -v`
Expected: FAIL — `captured["link_frame_ids"]` is `None` (add_ghost_gk does not yet derive/pass it).

- [ ] **Step 3: Implement in `add_ghost_gk`**

In `silly_kicks/tracking/features.py`, find:

```python
    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)

    # Short-circuit: skip compute if frames already have ghost columns
    if "ghost_gk_x" in frames.columns and frames["ghost_gk_x"].notna().any():
        ghost_frames = frames
    else:
        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved_model,
            home_team_id=home_team_id,
            actions=actions_for_context,
        )
```

Replace with:

```python
    # Link actions to frames
    if links is not None:
        pointers = links
    else:
        pointers, _ = link_actions_to_frames(actions, frames)

    # PR-S66: restrict the per-frame KDE to the frames these actions link to.
    # add_ghost_gk always has pointers (supplied or internally computed), so the
    # restriction applies regardless of source; the per-frame ghost is internal
    # and the action mapping reads only linked frames, so unrestricted frames
    # staying NaN changes no consumed value. Bit-identical (see compute_ghost_gk).
    link_frame_ids: set[int] | None = None
    if "frame_id" in pointers.columns:
        link_frame_ids = set(pointers["frame_id"].dropna().astype(int).tolist())

    # Short-circuit: skip compute if frames already have ghost columns
    if "ghost_gk_x" in frames.columns and frames["ghost_gk_x"].notna().any():
        ghost_frames = frames
    else:
        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved_model,
            home_team_id=home_team_id,
            actions=actions_for_context,
            link_frame_ids=link_frame_ids,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py::TestAddGhostGkRestriction -v`
Expected: PASS (2 tests).

---

### Task 3: `ghost_gk_xfns` — restrict via union of gamestate slots

**Files:**
- Modify: `silly_kicks/tracking/features.py` (`ghost_gk_xfns` / `_ghost_gk_transformer`, currently ~lines 3619–3663)
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
class TestGhostGkXfnsRestriction:
    def _states(self):
        # 3 gamestate slots (a0/a1/a2); each a single action on a distinct frame.
        def one(action_id, fid):
            return pd.DataFrame(
                [{
                    "action_id": action_id,
                    "game_id": "100",
                    "period_id": 1,
                    "time_seconds": float(fid) * 0.04,
                    "team_id": 1,
                    "player_id": "p99",
                    "start_x": 80.0,
                    "start_y": 34.0,
                    "type_name": "shot",
                    "result_name": "fail",
                    "bodypart_name": "foot",
                }]
            )
        return [one(0, 5), one(0, 4), one(0, 3)]

    def test_xfns_union_passed_to_compute(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import ghost_gk_xfns

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()

        captured: dict = {}
        real = ghost_mod.compute_ghost_gk

        def spy(frames_arg, **kwargs):
            captured.setdefault("calls", 0)
            captured["calls"] += 1
            captured["link_frame_ids"] = kwargs.get("link_frame_ids")
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", spy)
        (xfn,) = ghost_gk_xfns(model=model, home_team_id=1)
        xfn(self._states(), frames)

        # Single compute call, restricted to the UNION of the slots' linked frames.
        assert captured["calls"] == 1
        assert captured["link_frame_ids"] is not None
        assert {3, 4, 5} <= ({1, 2, 3, 4, 5} | captured["link_frame_ids"])
        assert captured["link_frame_ids"] <= {1, 2, 3, 4, 5}

    def test_xfns_output_unchanged_by_restriction(self, monkeypatch):
        import silly_kicks.tracking._ghost_gk as ghost_mod
        from silly_kicks.tracking.features import ghost_gk_xfns

        model, _, _ = _fitted_model()
        frames, _ = _make_goal_flip_velocity_fixture()
        states = self._states()

        (xfn,) = ghost_gk_xfns(model=model, home_team_id=1)
        restricted = xfn(states, frames)

        real = ghost_mod.compute_ghost_gk

        def force_full(frames_arg, **kwargs):
            kwargs["link_frame_ids"] = None
            return real(frames_arg, **kwargs)

        monkeypatch.setattr(ghost_mod, "compute_ghost_gk", force_full)
        (xfn2,) = ghost_gk_xfns(model=model, home_team_id=1)
        full = xfn2(states, frames)

        pd.testing.assert_frame_equal(restricted, full)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py::TestGhostGkXfnsRestriction -v`
Expected: FAIL — `link_frame_ids` is `None` (transformer computes over full frames).

- [ ] **Step 3: Implement in `ghost_gk_xfns`**

In `silly_kicks/tracking/features.py`, find the transformer body:

```python
        from ._ghost_gk import _resolve_model, compute_ghost_gk

        resolved = _resolve_model(model)

        # Compute once — add_ghost_gk short-circuits on pre-computed frames
        ghost_frames = compute_ghost_gk(frames, model=resolved, home_team_id=home_team_id)

        for i, slot in enumerate(states[:3]):
            enriched = add_ghost_gk(
                slot,
                ghost_frames,
                model=resolved,
                home_team_id=home_team_id,
            )
            for col in col_names:
                out[f"{col}_a{i}"] = enriched[col].values if col in enriched.columns else np.nan

        return out
```

Replace with:

```python
        from ._ghost_gk import _resolve_model, compute_ghost_gk

        resolved = _resolve_model(model)

        # PR-S66: link each gamestate slot once and restrict the single
        # compute_ghost_gk to the UNION of their linked frames. The union ⊇ every
        # slot's linked set, the KDE is byte-identical per sample, and each
        # per-slot add_ghost_gk reads only its own linked frames (union extras
        # stay NaN, unread). Reusing pointers as `links` avoids re-linking.
        slot_pointers: list[pd.DataFrame] = []
        link_frame_ids: set[int] = set()
        for slot in states[:3]:
            pointers, _ = link_actions_to_frames(slot, frames)
            slot_pointers.append(pointers)
            if "frame_id" in pointers.columns:
                link_frame_ids |= set(pointers["frame_id"].dropna().astype(int).tolist())

        ghost_frames = compute_ghost_gk(
            frames,
            model=resolved,
            home_team_id=home_team_id,
            link_frame_ids=link_frame_ids,
        )

        for i, (slot, pointers) in enumerate(zip(states[:3], slot_pointers)):
            enriched = add_ghost_gk(
                slot,
                ghost_frames,
                model=resolved,
                home_team_id=home_team_id,
                links=pointers,
            )
            for col in col_names:
                out[f"{col}_a{i}"] = enriched[col].values if col in enriched.columns else np.nan

        return out
```

(`link_actions_to_frames` is already imported at module level in `features.py`; confirm with `grep -n "def link_actions_to_frames\|link_actions_to_frames" silly_kicks/tracking/features.py` and add the import only if the transformer can't see it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py::TestGhostGkXfnsRestriction -v`
Expected: PASS (2 tests).

---

### Task 4: e2e realistic-scale golden + perf guard + atomic re-export guard

**Files:**
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py`

- [ ] **Step 1: Write the tests**

Append (e2e marker keeps the heavy fixture out of the default suite):

```python
def _make_dense_match(n_frames: int = 250, home_team_id: int = 1, away_team_id: int = 2):
    """Multi-frame fixture for the structural call-count guard + a bit-identical
    check at modest size. NOTE: this is the lightweight CI guard, NOT the §5 scale
    measurement — the real full-match timing (n≈3000 rows≈70k, bundled model) is
    Task 5, run manually against the per-half budget."""
    parts = []
    for fid in range(1, n_frames + 1):
        f = _make_ghost_gk_frames(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            frame_id=fid,
            timestamp=float(fid) * 0.04,
        )
        defmask = (
            (f["team_id"] == home_team_id)
            & ~f["is_goalkeeper"].astype(bool)
            & ~f["is_ball"].astype(bool)
        )
        f.loc[defmask, "x"] = f.loc[defmask, "x"] + (fid % 7)  # mild movement
        parts.append(f)
    # link every 25th frame (≈ one action/sec)
    linked = set(range(1, n_frames + 1, 25))
    return pd.concat(parts, ignore_index=True), linked


@pytest.mark.e2e
class TestGhostGkRestrictionStructuralGuard:
    def test_bit_identical_at_guard_scale(self):
        model, _, _ = _fitted_model()
        frames, linked = _make_dense_match()

        full = compute_ghost_gk(frames, model=model, home_team_id=1)
        restricted = compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)

        f_rows = _linked_gk_rows(full, linked)
        r_rows = _linked_gk_rows(restricted, linked)
        assert len(f_rows) == len(r_rows) > 0
        for col in _GHOST_COLS:
            np.testing.assert_array_equal(f_rows[col].to_numpy(), r_rows[col].to_numpy())

    def test_predict_set_equals_linked_count(self, monkeypatch):
        """Structural perf guard (CI-robust): the restricted KDE runs on exactly
        the linked GK-sample count, far below the full-frame sample count."""
        model, _, _ = _fitted_model()
        frames, linked = _make_dense_match()

        captured: list[int] = []
        orig = model.predict_density

        def spy(features):
            captured.append(len(features))
            return orig(features)

        monkeypatch.setattr(model, "predict_density", spy)
        compute_ghost_gk(frames, model=model, home_team_id=1, link_frame_ids=linked)
        restricted_n = captured[-1]
        captured.clear()
        compute_ghost_gk(frames, model=model, home_team_id=1)
        full_n = captured[-1]

        # 2 GKs per linked frame; full = 2 GKs × 250 frames.
        assert restricted_n == 2 * len(linked)
        assert full_n == 2 * 250
        assert restricted_n < full_n / 5  # large reduction


def test_atomic_reexports_add_ghost_gk():
    """Atomic mirror re-exports (no duplicate impl) — inherits the fix."""
    from silly_kicks.atomic.tracking import features as atomic_feat
    from silly_kicks.tracking import features as main_feat

    assert atomic_feat.add_ghost_gk is main_feat.add_ghost_gk
    assert atomic_feat.ghost_gk_xfns is main_feat.ghost_gk_xfns
```

- [ ] **Step 2: Run the e2e + regular tests**

Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -v` (regular: atomic re-export + Tasks 1–3)
Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -v -m e2e` (scale tests)
Expected: PASS. If `test_atomic_reexports_add_ghost_gk` fails, the atomic module has a duplicate impl — STOP and surface it (the spec assumed a re-export).

---

### Task 5: §5 empirical full-match validation — the extraction-scope gate

**Files:** none committed (measurement + decision). Use a real match if a tracking dataset is available in this environment; otherwise the `_make_dense_match` fixture at full half scale (`n_frames≈70_000`-row equivalent: ~250 frames already ≈ 23 rows/frame; scale `n_frames` so total rows ≈ 70k, i.e. `n_frames≈3000`).

- [ ] **Step 1: Measure both paths at full-match scale**

Run a throwaway script (do NOT commit) timing `add_ghost_gk` with the **bundled default model** (not the tiny test model — the KDE cost scales with training-set size, so the test model would understate it):

```python
import time, pandas as pd
from silly_kicks.tracking.features import add_ghost_gk, link_actions_to_frames
# build/load a full-half frames df (~70k rows, both GKs, 2 periods) + ~1500 actions
# ... (real match preferred; else scale _make_dense_match) ...
t0 = time.perf_counter(); enr = add_ghost_gk(actions, frames, home_team_id=HOME); t_restricted = time.perf_counter() - t0
# force-full reference via monkeypatch-equivalent: call compute_ghost_gk(..., link_frame_ids=None) + manual map, time it
print("restricted (per-half):", t_restricted)
print("approx per-250-batch:", t_restricted / (len(frames_unique_frame_ids) / 250))
```

Capture: (i) per-half wall-clock full vs restricted, (ii) per-250-batch figure, (iii) KDE sample-count reduction. Verify byte-identical action-coupled columns full vs restricted on this data.

- [ ] **Step 2: Apply the numeric decision gate**

- Restricted per-half post-fix wall-clock **< 15 min (50% of the 30-min budget)** → extraction restriction is **deferred**. Proceed to Task 7 (Task 6 is skipped). Record the C3 breadcrumb in Task 8.
- Restricted per-half post-fix wall-clock **≥ 15 min** → extraction restriction is **in scope**. Do Task 6 before Task 7.

Record all measured numbers (both scales) — they go in the PR description (Task 9).

---

### Task 6 (CONDITIONAL — only if Task 5 gate fires): restrict feature extraction

**Skip entirely unless Task 5 step 2 selected "in scope."** Do not build speculatively.

**Files:**
- Modify: `silly_kicks/tracking/_ghost_gk.py` (`_extract_all_ghost_gk_features`, `compute_ghost_gk`)
- Test: `tests/tracking/test_ghost_gk_frame_restriction.py`

- [ ] **Step 1: Write the failing golden** — a SEPARATE bit-identical test (full vs extraction-restricted) on `_make_goal_flip_velocity_fixture` AND `_make_dense_match`, asserting the linked-frame ghost columns are byte-identical when extraction is restricted via the precompute-and-inject variant. (Distinct from Task 1's KDE-only golden.)

- [ ] **Step 2: Implement the precompute-and-inject variant** per spec §5:
  1. Cheap full-frame pass computing only `defensive_line_x` + `defending_centroid_x` per processed frame in `(game_id, period_id, frame_id)` sort order, reproducing the exact `(game_id, gk_team)` **cross-period** `prev_state` chain + `dt` fallback (§2.2), to build a per-`(linked frame, gk_team)` lookup of `(prev_defensive_line_x, prev_defending_centroid_x, dt)`.
  2. Compute the per-period `_defending_goal` dict over the full frames (already full-frame; hoist it).
  3. Run the heavy `extract_ghost_gk_features` only on linked frames, injecting (1)+(2) via its existing `prev_*`/`dt`/`goal_x` params.
  Gate this behind the same `link_frame_ids` (extend, don't add a second kwarg).

- [ ] **Step 3: Run the new golden + the full restriction suite** — all PASS, byte-identical. Re-measure Task 5 numbers; confirm under budget.

---

### Task 7: Shift-left gate sweep (lint, types, full suite)

**Files:** none (verification).

- [ ] **Step 1: Format check** — Run: `ruff format --check .` → Expected: no files would be reformatted. (If it flags, run `ruff format .` then re-check.)
- [ ] **Step 2: Lint** — Run: `ruff check .` → Expected: `All checks passed!`
- [ ] **Step 3: Types (full package)** — Run: `pyright silly_kicks/` → Expected: 0 errors. (Full package per project rule, not just changed files.)
- [ ] **Step 4: Full non-e2e suite** — Run: `python -m pytest tests/ -m "not e2e" -v --tb=short` → Expected: all pass (the existing ghost_gk suite + the new restriction file).
- [ ] **Step 5: e2e restriction tests** — Run: `python -m pytest tests/tracking/test_ghost_gk_frame_restriction.py -m e2e -v` → Expected: pass.

---

### Task 8: Version bump (3.26.0) + CHANGELOG + TODO + C3 breadcrumb

**Files:**
- Modify: `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`

- [ ] **Step 1: Bump version (both files must match)**

**Version-collision check (do this FIRST).** Target 3.26.0 **assumes PR-S65
(cover_shadows) shipped as the patch `3.25.1`** — confirmed in the CHANGELOG head
(`## [3.25.1] — … (PR-S65)`). If any other parallel silly-kicks PR has already
taken `3.26.0` (re-check `pyproject.toml` on `main` + open PRs at execution time),
bump this PR to `3.27.0` instead and update the CHANGELOG heading + commit message
accordingly. Re-read the live `version =` line before editing rather than trusting
the literal below.

`pyproject.toml`: `version = "3.25.1"` → `version = "3.26.0"`.
`silly_kicks/__init__.py`: `__version__ = "3.25.1"` → `__version__ = "3.26.0"`.

- [ ] **Step 2: CHANGELOG entry** — insert above `## [3.25.1]`:

```markdown
## [3.26.0] — 2026-05-28

### Performance
- **Ghost-GK linked-frame restriction (`add_ghost_gk`, `ghost_gk_xfns`, TF-18).**
  `compute_ghost_gk` gains an optional `link_frame_ids` kwarg that restricts the
  expensive per-sample density KDE (`predict_density`) to action-linked frames.
  Feature extraction still runs over the full frames, so the two cross-frame
  dependencies — per-period defending-goal mean-x inference and the cross-period
  one-step velocity state — are preserved exactly; the KDE is per-sample
  independent, so the output is **byte-identical** to the unrestricted compute.
  `add_ghost_gk` derives the set from its link pointers (supplied or internally
  computed); `ghost_gk_xfns` restricts to the union of its three gamestate slots.
  No new columns, no API break (additive kwarg). (PR-S66)
```

(If Task 6 landed, add a second bullet noting feature-extraction is also restricted via the precompute-and-inject variant, still byte-identical.)

- [ ] **Step 3: TODO.md grooming** — delete the PR-S66 candidate row if one exists. **C3 breadcrumb:** only if Task 6 was deferred, add one row under the tracking/perf section:

```markdown
| TF-18-ext | ghost_gk: restrict `_extract_all_ghost_gk_features` (full-frame velocity-state/goal-mean precompute + linked-only heavy extraction) | — | Only if measured a co-bottleneck — PR-S66 §5 gate found full-half post-KDE-fix time < 15 min so deferred. See spec §5 for the variant. |
```

(Verify the column shape matches the existing TODO table before inserting — `grep -n "^| TF-" TODO.md | head`.)

- [ ] **Step 4: Verify the four-file gate matches** — Run: `grep -n "3.26.0" pyproject.toml silly_kicks/__init__.py CHANGELOG.md` and confirm TODO.md no longer has a stale PR-S66 row. All version references must read `3.26.0`.

---

### Task 9: Final review + single commit

**Files:** all changed.

- [ ] **Step 1: Run `/final-review`** (mandatory project gate) and address any findings.
- [ ] **Step 2: Confirm clean tree state** — Run: `git status` and `git diff --stat`. Expected changed files: `silly_kicks/tracking/_ghost_gk.py`, `silly_kicks/tracking/features.py`, `tests/tracking/test_ghost_gk_frame_restriction.py`, `pyproject.toml`, `silly_kicks/__init__.py`, `CHANGELOG.md`, `TODO.md`, plus the spec + this plan under `docs/superpowers/`. The `.hypothesis/` dir must NOT be staged.
- [ ] **Step 3: Get explicit user approval to commit** (per commit policy). Then stage + ONE commit:

```bash
git add silly_kicks/tracking/_ghost_gk.py silly_kicks/tracking/features.py \
  tests/tracking/test_ghost_gk_frame_restriction.py pyproject.toml \
  silly_kicks/__init__.py CHANGELOG.md TODO.md \
  docs/superpowers/specs/2026-05-28-pr-s66-ghost-gk-frame-restriction-design.md \
  docs/superpowers/plans/2026-05-28-pr-s66-ghost-gk-frame-restriction.md
git commit -m "$(cat <<'EOF'
perf(tracking): restrict ghost_gk KDE to linked frames -- silly-kicks 3.26.0 (PR-S66)

compute_ghost_gk gains link_frame_ids; extraction stays full-frame (preserves
per-period goal-mean + cross-period velocity deps), only the per-sample KDE is
restricted -> byte-identical. add_ghost_gk derives the set from its pointers;
ghost_gk_xfns from the union of its 3 gamestate slots.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 4: Push + open PR** only when asked; wait for main CI green before tagging (never tag before CI green).

---

## Self-review (completed by plan author)

**Spec coverage:** §3.1 → Task 1; §3.2 → Task 2; §3.3 (VAEP union) → Task 3; §3.4 (atomic re-export, training untouched) → Task 4 guard; §4.1 golden + discrimination + unit tests → Tasks 1–3; §4.2 e2e structural guard + scale → Task 4; §5 full-scale gate + numeric trigger → Task 5; §5 conditional variant → Task 6; §6 housekeeping/version gate → Tasks 7–8; C3 breadcrumb → Task 8 step 3 + Task 1 code comment. All covered.

**Placeholder scan:** Task 5 contains intentionally-sketched measurement code (a throwaway, uncommitted script whose exact data source depends on the environment) — flagged as such, not a shipped artifact. All shipped code/tests are complete.

**Type/name consistency:** `link_frame_ids` (set | None) used identically across `compute_ghost_gk`, `add_ghost_gk`, `ghost_gk_xfns`; `_GHOST_COLS` / `_make_goal_flip_velocity_fixture` / `_linked_gk_rows` / `_make_dense_match` defined once and reused; `predict_density` spy signature matches `GhostGkModel.predict_density(features)`.
