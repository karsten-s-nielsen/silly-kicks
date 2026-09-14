# TF-62 Phase 2 — GK build-up Decision-quality: the reconstruction tiers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the TF-62 cycle by adding the **reconstruction tiers** to `silly_kicks/gk_decision/` — `ReconstructedOptionSet` (SB360 freeze-frame + full-tracking) with an injected xPass, the reused packing progression kernel, and the reachability filter — plus the promoted public `tracking.action_ltr_goal_map` constructor (with `territorial_defense` migrated onto it) and the battery's reconstruction validation legs (fidelity, SB360 responsiveness, reachability sweep). This is **Phase 2 of one cycle**; Phase 1 (the native anchor) is already built and independently reviewed.

**Architecture:** The tier-agnostic engine (`compute_gk_decision_value`) and the `OptionSet` port are unchanged from Phase 1. Phase 2 adds ONE new adapter, `ReconstructedOptionSet`, that yields the SAME uniform option-rows table (`OPTION_ROW_COLUMNS`: `game_id, period_id, decision_id, keeper_id, team_id, is_chosen, completion, opponents_bypassed, option_set_source`). It **scores every option in the action-LTR frame** (the acting keeper's team attacks x=105): keeper + candidate teammate positions come from the linked freeze-frame (reflected to action-LTR when the frames are match-LTR), the chosen target is the **actual pass end** (never snapped), `completion` is the injected `PassCompletionModel` served in its trained action-LTR convention, and `opponents_bypassed` is `tracking.compute_packing_metrics(...)["packing_made"]` fed the promoted `tracking.action_ltr_goal_map`. The reachability filter (`GkDecisionParams.reachability_min_xpass`) prunes unreachable candidates — the load-bearing spike finding, made a parameter.

**Tech Stack:** Python, pandas, numpy; `silly_kicks.tracking` (public seams: `compute_packing_metrics`, `action_ltr_goal_map`, `resolve_defended_goals`, `gk_distribution_mask`, `snapshot_to_tracking_frames`, `link_actions_to_frames`, `GoalMap`), `silly_kicks.keeper_identity.apply_actor_identities_to_frames`, `silly_kicks.expected_passing.PassCompletionModel`, `silly_kicks.reflection` (ADR-045), `silly_kicks.id_compat`. No new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-09-11-tf62-gk-buildup-decision-value-design.md` (this plan implements its **PR2**, §0/§5/§7/§9/§11/§12.6). Read it. **Phase 1** (the native anchor this builds on) was built first per spec §0 — native-first is a BUILD ORDER, not a separate commit; the Phase-1 planning notes were folded into this plan and the spec rather than kept as a separate superseded file.

## Global Constraints

- **ONE cycle, ONE commit (OVERRIDES the writing-plans template).** Phase 1 was **not** committed separately (there is no provenance chain forcing a split — the metric bundles no owner-run weights; it reuses the already-bundled `PassCompletionModel`). Phase 1 + Phase 2 land as **ONE coherent, fully-tested commit**, made only after the whole cycle is green **AND** the reconstruction validation probes have actually run **AND** Karsten explicitly approves the shown diff. **NO per-task commits, NO micro-commits.** Each task ends at green tests. Feature branch `feat/tf62-gk-buildup-decision-value` (already carries Phase 1's uncommitted work); never a worktree.
- **Validation before "done" (the owner's hard bar).** Phase 2 is not complete until its probes are **run on real data**, not inherited from the spike: the reconstruction-fidelity leg (reconstructed-from-tracking vs native-GI on the SkillCorner Rosetta Stone), the SB360-responsiveness leg, and the reachability two-sided gate. Owner-run (owner-tier corpus); aggregate-only artifacts (reversibility-not-provenance).
- **`compute_*`, not `add_*`** — Phase 2 registers NO action-coupled aggregator (C4 aggregator count unchanged; the `gk_decision` C4 container already exists from Phase 1; count stays 399). In **no** default xfn list. Additive — no VAEP/tracking retrain, no re-materialize. `ReconstructedOptionSet` reads no `result_id`/post-contact outcome (value = pre-outcome xPass + geometry + the actual pass end as the chosen anchor), so the no-`*_xfns` / no-leakage posture holds.
- **Import-allowlist RELAXES (this is the one intended widening).** Phase 2 makes `gk_decision` a tracking-CONSUMING sibling (like `gkdv`/`restdefense`/`territorial_defense`): it MAY import `silly_kicks.tracking` **public** seams and `silly_kicks.keeper_identity`, but NEVER a `silly_kicks.tracking._*` private module, and **nothing** imports `gk_decision`. Task 5 rewrites the Phase-1 allowlist test (`test_gk_decision_never_imports_tracking_in_pr1`) to the sibling shape with a planted-violation meta-test for BOTH the private-tracking ban and the reverse.
- **ONE goal-end implementation (ADR-055, TF62-SPEC-09).** `action_ltr_goal_map` is PROMOTED to a public `tracking` seam and `territorial_defense`'s private copy is DELETED and re-imported from `tracking` — never a second copy of the orientation logic. Task 1.
- **Score in action-LTR; reflect frames via the ONE public seam.** All valuation happens in action-LTR (keeper attacks x=105) so `PassCompletionModel` is served in its trained convention. Match-LTR frames (full-tracking) are reflected to action-LTR via `silly_kicks.reflection` (ADR-045 — the ONE reflection seam, which negates vectors/point-reflects positions correctly), never a hand-rolled `105-x`. SB360 freeze-frames are already action-LTR (verified: actor position == action start, |dx|=|dy|=0.00 m) → no reflection. The correctness backstop is the Task-3 mirror-invariance gate on away-possession decisions.
- **ids** via `id_compat` (ADR-019); keepers grouped on `canonical_id_series`, raw id emitted; the keeper↔frame match via `ids_equal`/`ids_match`; never raw `==`/`astype(str)` on ids.
- **Conservation** (ADR-042/043): every GK decision dropped-and-counted, never a fabricated 0. Phase 2 adds the reconstruction drop reasons `no_frame` (no linked freeze-frame / no positions) and `fov_cropped` (SB360 under-observed local region, ADR-077) to `GK_DECISION_DROP_REASONS`, and `GkDecisionReport` grows the matching counters — conservation stays `n_decisions_in == n_scored + Σ drop_reasons`. `too_few_options` / `no_unique_chosen` / `chosen_unvalued` are the engine's existing drops (Phase 1); the adapter's own drops are counted by the adapter and folded into the same Report.

---

## File Structure

| File | Responsibility |
|---|---|
| `silly_kicks/tracking/_gk_resolve.py` | MODIFY: add `action_ltr_goal_map` (moved from `territorial_defense/_engine.py`), beside `resolve_defended_goals` |
| `silly_kicks/tracking/__init__.py` | MODIFY: export `action_ltr_goal_map` in `__all__` |
| `silly_kicks/territorial_defense/_engine.py` | MODIFY: DELETE the local `action_ltr_goal_map`; import it from `silly_kicks.tracking` |
| `silly_kicks/territorial_defense/_compute.py` | MODIFY: import `action_ltr_goal_map` from `tracking` (the `_make_goal_map_for` factory unchanged otherwise) |
| `silly_kicks/gk_decision/_columns.py` | MODIFY: extend `GK_DECISION_DROP_REASONS` with `no_frame`, `fov_cropped` |
| `silly_kicks/gk_decision/_report.py` | MODIFY: add `n_no_frame`, `n_fov_cropped` counters; conservation includes them |
| `silly_kicks/gk_decision/_reconstruct.py` | CREATE: `ReconstructedOptionSet` + the action-LTR framing helpers |
| `silly_kicks/gk_decision/_compute.py` | MODIFY: thread the adapter's own drop counts into `GkDecisionReport` (see Task 2 note) |
| `silly_kicks/gk_decision/__init__.py` | MODIFY: export `ReconstructedOptionSet` |
| `silly_kicks/feature_glossary.py` | NO new columns (reconstruction emits the same 5 metric columns); Task 7 verifies count stays 399 |
| `tests/gk_decision/test_reconstruct.py` | CREATE: `ReconstructedOptionSet` unit + boundary + orientation + reachability + chosen-into-space |
| `tests/gk_decision/test_reconstruct_sb360_e2e.py` | CREATE: end-to-end on a committed SB360 fixture (actions + snapshot frames) |
| `tests/gk_decision/test_import_allowlist.py` | REWRITE: sibling shape (tracking public allowed, `tracking._*` banned, reverse ban) |
| `tests/gk_decision/test_report_conservation_reconstruct.py` | CREATE: conservation incl. `no_frame`/`fov_cropped` |
| `tests/tracking/test_action_ltr_goal_map_public.py` | CREATE: public seam + `territorial_defense`-migration parity |
| `tests/invariants/conftest_id_scalar.py` | MODIFY (if required): register/justify `action_ltr_goal_map` in `PUBLIC_ID_SCALAR_ENTRIES` |
| `tests/test_public_api_examples.py` | MODIFY (if required): `action_ltr_goal_map` Examples (the docstring already carries doctests) |
| `scripts/validate_gk_decision.py` | MODIFY: add the reconstruction legs (fidelity / SB360-responsiveness / reachability sweep) |
| `tests/scripts/test_gk_decision_battery_kernels.py` | MODIFY: known-truth kernels for the new legs (fidelity Spearman, reachability-sweep contrast) |
| `NOTICE`, `CLAUDE.md`, `CHANGELOG.md`, `TODO.md`, `docs/superpowers/adrs/ADR-092-*.md`, `docs/superpowers/specs/2026-09-11-tf62-*-design.md` | MODIFY: rewrite the PR1-scoped release artifacts + reconcile the SPEC (§0/§12.2/§12.7) to the WHOLE cycle (Task 7) |

**Uniform option-rows schema (unchanged):** `game_id, period_id, decision_id, keeper_id, team_id, is_chosen (bool), completion (float64), opponents_bypassed (float64), option_set_source (str)`. `ReconstructedOptionSet` emits `option_set_source == "reconstructed"`; exactly one `is_chosen=True` per `decision_id` (the actual pass).

---

### Task 1: Promote `action_ltr_goal_map` to a public `tracking` seam; migrate `territorial_defense` onto it (ADR-055 / TF62-SPEC-09)

**Files:**
- Modify: `silly_kicks/tracking/_gk_resolve.py` (add `action_ltr_goal_map`), `silly_kicks/tracking/__init__.py` (export it)
- Modify: `silly_kicks/territorial_defense/_engine.py` (delete local def; import from tracking), `silly_kicks/territorial_defense/_compute.py` (import from tracking)
- Create: `tests/tracking/test_action_ltr_goal_map_public.py`
- Modify (if the gate requires): `tests/invariants/conftest_id_scalar.py`, `tests/test_public_api_examples.py`

**Interfaces:**
- Produces (public): `tracking.action_ltr_goal_map(game_id, period_id, *, acting_team_id, opponent_team_id) -> GoalMap` — the per-frame action-LTR map (acting team attacks x=105, defends 0; opponent defends 105). Body is a VERBATIM move of the current `territorial_defense/_engine.py:40` definition (it already imports `GoalMap`, `canonical_id`, `spadlconfig`, `MappingProxyType` — all available in `_gk_resolve.py`).
- Consumes: unchanged in `territorial_defense` (the `_make_goal_map_for` factory calls the same function, now imported from `tracking`).

- [ ] **Step 1: Write the failing public-seam + migration-parity test**

```python
# tests/tracking/test_action_ltr_goal_map_public.py
import inspect
from silly_kicks.tracking import action_ltr_goal_map

def test_action_ltr_goal_map_is_public_and_action_ltr():
    gm = action_ltr_goal_map(7, 1, acting_team_id=1, opponent_team_id=2)
    assert gm.attacked_goal(7, 1, 1, allow_guess=True) == 105.0  # acting team attacks opponent's end
    assert gm.attacked_goal(7, 1, 2, allow_guess=True) == 0.0    # opponent attacks acting team's end

def test_territorial_defense_uses_the_promoted_public_seam():
    # territorial_defense must NOT define its own action_ltr_goal_map -- it imports the tracking one.
    import silly_kicks.territorial_defense._engine as eng
    from silly_kicks.tracking import action_ltr_goal_map as public
    assert eng.action_ltr_goal_map is public  # same object, not a re-implementation
    # and the source file no longer DEFINES it
    src = inspect.getsource(eng)
    assert "def action_ltr_goal_map(" not in src
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/tracking/test_action_ltr_goal_map_public.py -v`
Expected: FAIL — `ImportError: cannot import name 'action_ltr_goal_map' from silly_kicks.tracking`.

- [ ] **Step 3: Move the function**

Cut the `action_ltr_goal_map` definition (and its docstring/doctests) VERBATIM from `silly_kicks/territorial_defense/_engine.py` into `silly_kicks/tracking/_gk_resolve.py`, placed immediately after `resolve_defended_goals`. Confirm `_gk_resolve.py` already imports `GoalMap`, `canonical_id`, `spadlconfig`, and `from types import MappingProxyType` (add any missing import). Add `"action_ltr_goal_map"` to `silly_kicks/tracking/__init__.py`'s `__all__` and its import block.

- [ ] **Step 4: Re-point `territorial_defense`**

In `silly_kicks/territorial_defense/_engine.py`, DELETE the local definition and add `from silly_kicks.tracking import action_ltr_goal_map` (keep it re-exported at module scope so `_engine.action_ltr_goal_map` still resolves — `_compute.py` imports it `from ._engine import action_ltr_goal_map`; leave that import path working, OR re-point `_compute.py` to import from `tracking` directly and drop the `_engine` re-export). Prefer re-pointing `_compute.py` to `from silly_kicks.tracking import action_ltr_goal_map` and removing the `_engine` symbol entirely, so there is exactly one definition and one import path.

- [ ] **Step 5: Wire the public-surface gates**

Run `python -m pytest tests/test_public_api_examples.py -k action_ltr_goal_map -v` and `python -m pytest tests/invariants/ -k id_scalar -v`. If `test_public_api_examples` flags the new export, its docstring already carries runnable `>>>` doctests (self-contained) — confirm it is discovered (the `_gk_resolve.py` module is already public via `resolve_defended_goals`). If the id-scalar registry meta-assertion fails (the function takes `acting_team_id`/`opponent_team_id` scalars), register it in `PUBLIC_ID_SCALAR_ENTRIES` exercised on a matched id, a value-equal cross-dtype id (`1` vs `"1"`), and a float id (`1.0`) — OR add a justified entry (it CONSTRUCTS a map by canonicalising ids rather than comparing them; the comparison lives in `GoalMap.attacked_goal`, already covered). Pick registration if the meta-assertion demands coverage; record the reason inline either way.

- [ ] **Step 6: Run the ADR-055 population gate + territorial_defense suite**

Run: `python -m pytest tests/tracking/test_goal_map_population.py tests/tracking/test_action_ltr_goal_map_public.py tests/territorial_defense/ -v`
Expected: PASS (the semantic AST goal-map-population gate must stay green — the move does not add a hand-rolled ternary; `territorial_defense` behaviour is byte-identical, only the import path changed).

---

### Task 2: `ReconstructedOptionSet` — the positional adapter (action-LTR valuation + reachability)

**Files:**
- Create: `silly_kicks/gk_decision/_reconstruct.py`
- Modify: `silly_kicks/gk_decision/_columns.py` (drop reasons), `silly_kicks/gk_decision/_report.py` (counters), `silly_kicks/gk_decision/_compute.py` (fold adapter drops into the Report — see note), `silly_kicks/gk_decision/__init__.py` (export)
- Test: `tests/gk_decision/test_reconstruct.py`

**Interfaces:**
- Consumes: `actions` (SPADL, with `game_id, period_id, action_id, type_id, player_id, team_id, start_x, start_y, end_x, end_y`), `frames` (long-form `TRACKING_FRAMES_COLUMNS`, carrying real `player_id` after `apply_actor_identities_to_frames` on SB360, or native ids on full-tracking), `xpass: PassCompletionModel`, `params: GkDecisionParams`, `keeper_ids`, `frame_convention: Literal["per_action_ltr","match_ltr"]="per_action_ltr"`, `links: pd.DataFrame | None = None` (optional pre-computed `link_actions_to_frames` pointers), `visible_area: pd.DataFrame | None = None` (SB360 FOV polygons, ADR-077).
- Produces: `ReconstructedOptionSet(...)` implementing the `OptionSet` port — `option_rows() -> pd.DataFrame` (uniform `OPTION_ROW_COLUMNS`, `option_set_source="reconstructed"`), PLUS `drop_counts() -> dict[str,int]` returning `{"no_frame": …, "fov_cropped": …}` (the adapter's own drops; the engine adds `too_few_options`/`no_unique_chosen`/`chosen_unvalued`).

**Note on Report wiring:** the Phase-1 engine builds `GkDecisionReport` from `n_in` = number of grouped decisions in `option_rows()`. A decision the adapter drops as `no_frame`/`fov_cropped` produces NO option rows, so the engine would never count it. Thread the adapter's `drop_counts()` into the Report so `n_decisions_in` is the TRUE decision count (domain size), not the survivors: `compute_gk_decision_value(option_set, *, params, extra_drops=None)` accepts an optional `extra_drops: dict[str,int]` (default `None`), adds them to the matching counters, and includes them in `n_decisions_in`. `ReconstructedOptionSet` exposes `drop_counts()`; the SB360/full-tracking driver passes it as `extra_drops`. (Native/`SkillCornerGIOptionSet` has no such drops → `extra_drops=None` → Phase-1 behaviour byte-identical.)

- [ ] **Step 1: Write the failing unit test** (a hand-built 1-decision frame with known geometry → known option rows)

```python
# tests/gk_decision/test_reconstruct.py
import numpy as np, pandas as pd, pytest
from silly_kicks.gk_decision import ReconstructedOptionSet, GkDecisionParams
from silly_kicks.gk_decision._columns import OPTION_ROW_COLUMNS

class _FakeXPass:
    """Deterministic stand-in: completion decays with pass distance (positional, no velocity)."""
    def predict_completion(self, ox, oy, tx, ty):
        ox, oy, tx, ty = map(lambda a: np.asarray(a, dtype=float), (ox, oy, tx, ty))
        d = np.hypot(tx - ox, ty - oy)
        return np.clip(1.0 - d / 120.0, 0.0, 1.0)

def _one_decision():
    # SB360-style: frames already action-LTR (keeper team 7 attacks x=105).
    # keeper 99 at (10,34); teammates 11 (short, near, high xPass), 12 (far upfield, low xPass),
    # 13 (backward). Opponent 21 sits between keeper and teammate 11 to make packing>0.
    actions = pd.DataFrame([dict(
        game_id="g", period_id=1, action_id=1, type_id=22,  # goalkick (gk_distribution)
        player_id=99, team_id=7, start_x=10.0, start_y=34.0, end_x=30.0, end_y=34.0,  # chosen -> ~teammate 11
    )])
    frames = pd.DataFrame([
        dict(game_id="g", period_id=1, frame_id=1, player_id=99, team_id=7, x=10.0, y=34.0, is_ball=False, is_goalkeeper=True,  is_actor=True),
        dict(game_id="g", period_id=1, frame_id=1, player_id=11, team_id=7, x=30.0, y=34.0, is_ball=False, is_goalkeeper=False, is_actor=False),
        dict(game_id="g", period_id=1, frame_id=1, player_id=12, team_id=7, x=95.0, y=34.0, is_ball=False, is_goalkeeper=False, is_actor=False),
        dict(game_id="g", period_id=1, frame_id=1, player_id=13, team_id=7, x=5.0,  y=34.0, is_ball=False, is_goalkeeper=False, is_actor=False),
        dict(game_id="g", period_id=1, frame_id=1, player_id=21, team_id=8, x=20.0, y=34.0, is_ball=False, is_goalkeeper=False, is_actor=False),
        dict(game_id="g", period_id=1, frame_id=1, player_id=1,  team_id=None, x=10.0, y=34.0, is_ball=True, is_goalkeeper=False, is_actor=False),
    ])
    return actions, frames

def test_reconstructed_uniform_schema_and_chosen_is_actual_pass():
    actions, frames = _one_decision()
    os_ = ReconstructedOptionSet(actions, frames, xpass=_FakeXPass(),
                                 params=GkDecisionParams(), keeper_ids=[99],
                                 frame_convention="per_action_ltr")
    rows = os_.option_rows()
    assert list(rows.columns) == list(OPTION_ROW_COLUMNS)
    assert (rows["option_set_source"] == "reconstructed").all()
    assert rows["is_chosen"].sum() == 1                       # exactly one chosen (the actual pass)
    # chosen target is the actual pass END, valued by xPass(keeper->end); never snapped to a teammate id
    chosen = rows[rows["is_chosen"]].iloc[0]
    assert chosen["completion"] == pytest.approx(1.0 - 20.0 / 120.0)   # dist keeper(10,34)->end(30,34)=20
    # teammate 12 upfield is very far (low xPass) but present pre-filter; opponents_bypassed is a count >=0
    assert (rows["opponents_bypassed"] >= 0).all()

def test_reachability_filter_prunes_unreachable_alternatives():
    actions, frames = _one_decision()
    # threshold below 12's xPass keeps it; above prunes it. 12 at (95,34): dist ~85 -> xPass ~0.29.
    keep = ReconstructedOptionSet(actions, frames, xpass=_FakeXPass(),
                                  params=GkDecisionParams(reachability_min_xpass=0.2),
                                  keeper_ids=[99]).option_rows()
    prune = ReconstructedOptionSet(actions, frames, xpass=_FakeXPass(),
                                   params=GkDecisionParams(reachability_min_xpass=0.5),
                                   keeper_ids=[99]).option_rows()
    tgt_ids_keep = set(prune["decision_id"])  # both still have the decision if >= min_options survive
    assert (keep["completion"] >= 0.2).all() or keep.empty
    assert (prune["completion"] >= 0.5).all() or prune.empty
    assert len(prune) <= len(keep)             # a higher floor prunes at least as much
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/gk_decision/test_reconstruct.py -v`
Expected: FAIL — `ReconstructedOptionSet` not defined.

- [ ] **Step 3: Implement `_reconstruct.py`**

Design (spec §5): per GK decision, score every option in action-LTR.

```python
# silly_kicks/gk_decision/_reconstruct.py
"""ReconstructedOptionSet -- positional option sets for SB360 + full-tracking (spec §5).

Scores every option in ACTION-LTR (the acting keeper's team attacks x=105) so the injected
PassCompletionModel is served in its trained convention. Match-LTR frames are reflected to action-LTR
via silly_kicks.reflection (ADR-045 -- the ONE reflection seam). opponents_bypassed reuses
tracking.compute_packing_metrics["packing_made"] (ADR-039 single definition), fed the promoted public
tracking.action_ltr_goal_map. The chosen option is the ACTUAL pass end (never snapped to a teammate).
Reachability (GkDecisionParams.reachability_min_xpass) prunes unreachable candidates -- the spike's
load-bearing finding. Drops (no_frame / fov_cropped, ADR-042/077) are COUNTED, never fabricated 0s.
"""
from __future__ import annotations
from typing import Literal
import numpy as np, pandas as pd

from silly_kicks import reflection
from silly_kicks.id_compat import canonical_id, ids_equal, ids_isin, ids_match
from silly_kicks.tracking import (
    action_ltr_goal_map, compute_packing_metrics, link_actions_to_frames, resolve_defended_goals,
)
from ._columns import OPTION_ROW_COLUMNS
from ._config import GkDecisionParams

_GK_DISTRIBUTION_TYPES = None  # domain resolved by the caller via gk_distribution_mask; the adapter
                               # receives already-filtered GK-decision actions (one row per decision).

class ReconstructedOptionSet:
    def __init__(self, actions, frames, *, xpass, params: GkDecisionParams, keeper_ids,
                 frame_convention: Literal["per_action_ltr", "match_ltr"] = "per_action_ltr",
                 links=None, visible_area=None) -> None:
        self._actions = actions
        self._frames = frames
        self._xpass = xpass
        self._params = params
        self._keeper_ids = list(keeper_ids)
        self._convention = frame_convention
        self._links = links
        self._visible_area = visible_area
        self._drops = {"no_frame": 0, "fov_cropped": 0}

    def drop_counts(self) -> dict:
        # option_rows() must have been called (it accumulates); call it if not yet done.
        return dict(self._drops)

    def option_rows(self) -> pd.DataFrame:
        acts = self._actions
        # keeper-possession decisions only (the caller passes gk_distribution_mask-filtered actions,
        # but re-guard on keeper_ids so a mis-scoped caller cannot leak outfield possessions).
        acts = acts[ids_isin(acts["player_id"], self._keeper_ids).to_numpy()]
        # link each action to its freeze-frame; SB360 snapshot sets frame_id == action_id.
        links = self._links if self._links is not None else link_actions_to_frames(acts, self._frames)
        match_map = resolve_defended_goals(self._frames) if self._convention == "match_ltr" else None
        out_rows: list[pd.DataFrame] = []
        self._drops = {"no_frame": 0, "fov_cropped": 0}
        for _, a in acts.iterrows():
            rows = self._rows_for_decision(a, links, match_map)
            if rows is None:               # no_frame / fov_cropped already counted
                continue
            out_rows.append(rows)
        if not out_rows:
            return pd.DataFrame(columns=list(OPTION_ROW_COLUMNS))
        return pd.concat(out_rows, ignore_index=True)[list(OPTION_ROW_COLUMNS)]
```

`_rows_for_decision(action, links, match_map)` — the per-decision core:

1. Resolve the linked frame: `fid = links.loc[links["action_id"]==action_id, "frame_id"]`; if none, `self._drops["no_frame"] += 1; return None`. Slice `frame = self._frames[ids_equal(self._frames["frame_id"], fid)]` (a copy).
2. **Orient to action-LTR.** `keeper_team = action["team_id"]`. If `self._convention == "match_ltr"`: `attacked = match_map.attacked_goal(game, period, keeper_team, allow_guess=True)`; if `attacked == 0.0` the keeper attacks x=0 → reflect the frame's geometry to action-LTR with `reflection.reflect_columns(frame, ...)` (positions point-reflected, `is_ball`/ids untouched) AND reflect the action's `(start_x,start_y,end_x,end_y)` the same way; if `attacked == 105.0` no reflection; if unresolved (`None`) count `no_frame` and `return None` (honest-NaN, never a guessed direction). If `per_action_ltr`: no reflection (SB360 frames are already action-LTR; the action anchors already are too).
3. **FOV completeness (SB360, ADR-077).** If `self._visible_area` is supplied, compute the observed fraction of a convex disk of radius `params.fov_radius_m` around the keeper (the build-up neighbourhood) with the PUBLIC `silly_kicks.tracking.region_observed_fraction(polygon, region)` (CONSIDER-7 — the named seam, not "the ADR-077 seam"): a fraction below `params.fov_min_observed_fraction` (PLAN-11 — a named param, not an unspecified "floor"), or a missing/degenerate polygon (→ NaN), counts `fov_cropped` and returns None (honest-NaN, never a fabricated fraction; pass the convex disk as the region so a concave `visible_area` is the clipped SUBJECT, never over-reported). Full-tracking passes `visible_area=None` → no FOV gate.
4. **Keeper + candidate positions (action-LTR).** keeper row = frame row whose `player_id` `ids_equal` the action's `player_id` (fallback: the `is_actor` row for SB360). `keeper_xy = (x, y)` of that row. Candidates = frame rows with `ids_match(team_id, keeper_team)`, not `is_ball`, not the keeper row → `candidate_xy` each.
5. **Chosen option = the ACTUAL pass end** (`end_x, end_y`, action-LTR) — a synthetic option row with `is_chosen=True`; NEVER snapped to a teammate. Exclude the single nearest visible teammate to the pass end from the *alternatives* (the presumed receiver, to avoid double-counting) — a documented exclusion; if the pass end lies in space (nearest teammate farther than a threshold) the exclusion is a no-op (Task-3 test).
6. **Value every option** (chosen + surviving alternatives):
   - `completion = self._xpass.predict_completion(keeper_x, keeper_y, target_x, target_y)` (action-LTR).
   - `opponents_bypassed = compute_packing_metrics(frame_ltr, attacking_team_id=keeper_team, goal_map=action_ltr_goal_map(game, period, acting_team_id=keeper_team, opponent_team_id=<the other team in the frame>), passer_xy=keeper_xy, receiver_xy=target_xy)["packing_made"]`.
7. **Reachability filter (alternatives only):** drop alternatives with `completion < params.reachability_min_xpass`. The chosen option is NEVER dropped by reachability (the keeper actually played it). If fewer than `min_options` survive, the ENGINE counts it `too_few_options` (do not pre-count here — return the rows and let the engine drop, so there is ONE conservation authority).
8. Assemble the uniform table (`keeper_id=action["player_id"]`, `team_id=keeper_team`, `decision_id=action_id`, `option_set_source="reconstructed"`).

Return the assembled `pd.DataFrame`. Determine `<the other team in the frame>` as the unique non-keeper team id among non-ball rows (NaN-safe via `dropna().unique()`); if not exactly one opponent team, count `no_frame` and return None (a malformed frame is not scoreable).

**Factor for independent red-green (CONSIDER-9):** put steps 1–3 (link → orient/reflect to action-LTR → FOV gate) in a testable helper `_frame_and_anchors_action_ltr(action, links, match_map) -> tuple | None` returning `(frame_ltr, keeper_xy, pass_end_xy, keeper_team, opponent_team)` or `None` (with the drop counted) — it is exercised on its own by Task 3's orientation mirror-invariance gate. Steps 4–8 (value every option → reachability → assemble) are the valuation core, exercised by Task 2 Step 1 + Task 3's magnitude/reachability gates. The two seams red-green separately.

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest tests/gk_decision/test_reconstruct.py -v`
Expected: PASS.

- [ ] **Step 5: Extend `_columns.py` / `_report.py` / `_compute.py`**

`GK_DECISION_DROP_REASONS = ("too_few_options", "no_unique_chosen", "chosen_unvalued", "no_frame", "fov_cropped")`. `GkDecisionReport` gains `n_no_frame: int = 0`, `n_fov_cropped: int = 0`; its conservation identity becomes `n_scored + n_too_few_options + n_no_unique_chosen + n_chosen_unvalued + n_no_frame + n_fov_cropped == n_decisions_in`. `compute_gk_decision_value(option_set, *, params=..., extra_drops: dict | None = None)` adds `extra_drops` counters and includes their sum in `n_decisions_in`. Export `ReconstructedOptionSet` in `__init__.py` `__all__`.

**`GkDecisionParams` gains the two FOV fields (PLAN-11):** `fov_radius_m: float = 10.0` and `fov_min_observed_fraction: float = 0.7` — the disk radius + observed-fraction floor the Step-3 pt3 gate uses (they DECIDE the `fov_cropped` drop, so they are named params, not buried constants; mirror the shape of `territorial_defense`'s `local_completeness_ok(radius_m, min_fraction)`). `for_provider`-tunable, LEFT UN-TUNED for the battery sweep to recommend (ADR-009 — same posture as `reachability_min_xpass`). Extend the Phase-1 `tests/gk_decision/test_config_report.py` to assert both defaults.

- [ ] **Step 6: Run the Phase-1 engine tests (regression — `extra_drops=None` byte-identical)**

Run: `python -m pytest tests/gk_decision/ -v`
Expected: PASS (Phase-1 `test_compute.py` unchanged behaviour; new reconstruction tests green).

---

### Task 3: Reconstruction correctness gates (magnitude · orientation · reachability two-sided · chosen-into-space)

**Files:**
- Modify: `tests/gk_decision/test_reconstruct.py` (add the gates)
- Create: `tests/gk_decision/test_report_conservation_reconstruct.py`

**Interfaces:** consumes Task 2's `ReconstructedOptionSet`; the `_FakeXPass` from Task 2.

- [ ] **Step 1: Magnitude / packing-boundary gate (spec §11 — the geometric-boundary half deferred from PR1).** Assert the ABSOLUTE `opponents_bypassed` (= `packing_made`) on hand-built geometry, pinning the bypass boundaries the reused packing kernel defines: opponent exactly on the pass segment, opponent behind the keeper (origin), opponent outside the passer→receiver x-interval, and a **backward/lateral option ⇒ `packing_made == 0` ⇒ `EV == completion`** (the §4 boundary). This pins that the reconstruction's bypass count IS the glossaried `packing_made` (TF62-SPEC-01), not a divergent reinvention.

```python
def test_packing_boundaries_match_glossaried_definition():
    # keeper (10,34) -> receiver (50,34); opponents at x = 5(behind), 30(on-segment interior),
    # 50(on receiver), 70(outside interval). packing_made counts (p.x < d.x <= r.x] -> {30, 50} = 2.
    ...
    assert chosen_bypassed == 2.0
    # backward option: receiver at (5,34) (x < keeper) -> made 0 -> EV == completion
    ...
```

- [ ] **Step 2: Orientation mirror-invariance gate (spec §11).** Build ONE physical decision, score it (a) as `per_action_ltr` (keeper attacks 105) and (b) as `match_ltr` with the frame + action reflected so the keeper attacks x=0 (an away-possession period). The resulting option rows (`completion`, `opponents_bypassed`, `decision_value` downstream) must be EQUAL — the metric is orientation-invariant. This is the non-vacuity backstop for the action-LTR reflection: a one-sided assertion would pass if BOTH legs were wrong the same way, so assert equality across the two conventions AND that the match_ltr leg actually reflected (a probe row lands where the reflection predicts).

- [ ] **Step 3: Reachability TWO-SIDED gate (spec §11 / §3C non-vacuity).** With the filter (`reachability_min_xpass=0.5`) an upfield-only alternative set that would invert the signal is pruned; without it (`=0.0`) the same set is kept. Assert the two option sets DIFFER (the filter changes something — a filter that changed nothing would be vacuous) AND that every surviving alternative clears the floor.

- [ ] **Step 4: Chosen-into-space gate (spec §5/§11).** A decision whose pass end lies in space (no visible teammate within the receiver-exclusion threshold): the chosen row's target is the pass END (unchanged), the receiver-exclusion is a documented no-op (no alternative wrongly dropped), and the decision still scores if ≥ `min_options` alternatives survive.

- [ ] **Step 5: Conservation gate (`no_frame`/`fov_cropped`).** In `test_report_conservation_reconstruct.py`, drive the SB360/full-tracking path over a mixed batch: one decision with a linked frame (scored), one with no linked frame (`no_frame`), one FOV-cropped (`fov_cropped`), one with < `min_options` reachable (`too_few_options`). Assert `n_decisions_in == n_scored + Σ drop_reasons` with `extra_drops` threaded from `ReconstructedOptionSet.drop_counts()`.

- [ ] **Step 6: keeper↔frame id-dtype mismatch gate (CONSIDER-10; ADR-019 hardening of the NEW actions↔frames match surface).** A decision whose action `player_id` is `int` while the linked frame's keeper row `player_id` is `str` (or vice-versa) must still match the keeper row (via `ids_equal`) and score — reconstruction adds a new cross-source id-match seam, so pin it against a cross-dtype mismatch (low risk; `ids_equal` is itself tested — this exercises the specific new surface, not `ids_equal`).

- [ ] **Step 7: Run**

Run: `python -m pytest tests/gk_decision/test_reconstruct.py tests/gk_decision/test_report_conservation_reconstruct.py -v`
Expected: PASS.

---

### Task 4: SB360 end-to-end on a committed fixture (actions + snapshot frames + actor bridge)

**Files:**
- Create: `tests/gk_decision/test_reconstruct_sb360_e2e.py`
- Create (if not reusable): a tiny committed SB360-shaped fixture under `tests/gk_decision/` (a handful of GK build-up actions + their `snapshot_to_tracking_frames` frames + `visible_area`), OR reuse an existing SB360 fixture from `tests/sb360/`/`tests/tracking/` if one carries GK goal-kicks (grep first; prefer reuse).

**Interfaces:** consumes `tracking.gk_distribution_mask`, `keeper_identity.apply_actor_identities_to_frames`, `tracking.snapshot_to_tracking_frames`, `expected_passing.PassCompletionModel.bundled()`, `ReconstructedOptionSet`, `compute_gk_decision_value`.

- [ ] **Step 1: Write the failing e2e test.** On the fixture: (1) build frames via `snapshot_to_tracking_frames`, (2) stamp keeper/actor identity via `apply_actor_identities_to_frames`, (3) domain via `gk_distribution_mask(actions, frames, resolve_gk="robust")`, (4) `ReconstructedOptionSet(gk_actions, frames, xpass=PassCompletionModel.bundled(), params=GkDecisionParams(), keeper_ids=roster_gks, frame_convention="per_action_ltr", visible_area=va)`, (5) `compute_gk_decision_value(option_set, extra_drops=option_set.drop_counts())`. Assert: samples non-empty (≥1 scored decision), `option_set_source=="reconstructed"`, conservation holds, keeper id is the REAL stamped id (not the synthetic `{0,1}`/row number), and every `sel_efficiency ∈ [0,1]`.

- [ ] **Step 2–4: Fixture + implement to green.** If no reusable fixture exists, build a minimal one (document its provenance in a header; keep it committed + small). Mark this test `@pytest.mark.e2e` ONLY if it needs an un-committed dataset; if the fixture is committed, it runs in the regular suite (per CLAUDE.md: committed-fixture tests are NOT `e2e`). Verify green.

Run: `python -m pytest tests/gk_decision/test_reconstruct_sb360_e2e.py -v`
Expected: PASS.

---

### Task 5: Rewrite the import-allowlist to the tracking-CONSUMING sibling shape

**Files:**
- Rewrite: `tests/gk_decision/test_import_allowlist.py`

- [ ] **Step 1: Rewrite the gate.** Replace `test_gk_decision_never_imports_tracking_in_pr1` with the sibling shape (mirror `tests/gkdv/test_import_allowlist.py` / `tests/territorial_defense/test_import_allowlist.py`): `gk_decision` MAY import `silly_kicks.tracking` (public) and `silly_kicks.keeper_identity`, but MUST NOT import any `silly_kicks.tracking._*` PRIVATE module; and nothing in `silly_kicks` outside `gk_decision` imports `gk_decision`. Keep both planted-violation meta-tests (a planted `from silly_kicks.tracking._das import get_das` FIRES the private-ban; a planted public `from silly_kicks.tracking import compute_packing_metrics` does NOT).

```python
# the private-tracking ban (replaces the blanket tracking ban)
_BANNED_PRIVATE_PREFIX = "silly_kicks.tracking._"
def _is_banned(m: str) -> bool:
    return m.startswith(_BANNED_PRIVATE_PREFIX)
```

- [ ] **Step 2: Run.**

Run: `python -m pytest tests/gk_decision/test_import_allowlist.py -v`
Expected: PASS (with `_reconstruct.py` importing `silly_kicks.tracking` public seams + `silly_kicks.keeper_identity` — allowed; no `tracking._*` import).

---

### Task 6: Battery reconstruction legs — fidelity · SB360 responsiveness · reachability sweep

**Files:**
- Modify: `scripts/validate_gk_decision.py`
- Modify: `tests/scripts/test_gk_decision_battery_kernels.py`

**Interfaces:**
- Produces: pure kernels `fidelity_spearman(recon_samples, native_samples, *, on) -> dict` (rank correlation of `decision_value`/`sel_efficiency` on the shared `(game_id, decision_id)` keys — the Rosetta-Stone fidelity), `reachability_sweep(...)` (naive vs filtered responsiveness contrast) added to the existing `reduce_samples`. The corpus orchestration remains owner-run (not CI).

- [ ] **Step 1: Write the failing kernel tests** (known-truth):
  - `fidelity_spearman`: two rankings that agree perfectly → ρ==1.0; reversed → ρ==-1.0; on the shared-key join only (a decision missing from one side is dropped, counted).
  - `reachability_sweep`: a synthetic option corpus where the naive set (all teammates) yields a non-responsive/negative `sel_efficiency` contrast and the filtered set yields a positive one → the kernel returns both, and they DIFFER (the §3C two-sided property at the battery level).

- [ ] **Step 2: Run to verify it fails.** `python -m pytest tests/scripts/test_gk_decision_battery_kernels.py -v` → FAIL (new kernels undefined).

- [ ] **Step 3: Implement the kernels + wire the driver legs.** Add `fidelity_spearman` + `reachability_sweep` to `validate_gk_decision.py` (test-pinned). Wire the owner-run orchestration: on SkillCorner Rosetta-Stone matches (native GI + tracking), build BOTH `SkillCornerGIOptionSet` (native) and `ReconstructedOptionSet` (from the SkillCorner tracking, `frame_convention="match_ltr"`) THROUGH THE SAME ENGINE, join on `(game_id, decision_id)`, report fidelity ρ; on SB360, run the reachability sweep (naive vs `reachability_min_xpass=0.5`) + the responsiveness leg (`sel_efficiency`/`decision_pct` vs the random-choice placebo). These are the probes the owner runs before the cycle is "done"; keep them aggregate-only. `require_clean_tree` + `declare_inputs` already present from Phase 1 — extend `declare_inputs` to name the reconstruction symbols (`PassCompletionModel`, `compute_packing_metrics`, `action_ltr_goal_map`, `GkDecisionParams.reachability_min_xpass`).

**Per-provider xPass recalibration — MEASUREMENT present, APPLY deferred (CONSIDER-8; spec §9 leg 7).** The fidelity leg MEASURES the §3B calibration offset (it reports the mean completion offset alongside ρ), so that measurement is in-cycle, not dropped. The per-provider recalibration **apply** (a `PassCompletionModel.from_variant` / `for_provider` change to the bundled xPass) is a **future ADR-009 gated apply** (spec §8.4/§12.4), intentionally NOT this cycle: the cycle ships the measurement + the default; tuning is a separate gated PR.

- [ ] **Step 4: Run the kernel tests.** `python -m pytest tests/scripts/test_gk_decision_battery_kernels.py tests/scripts/test_provenance_wiring.py -v` → PASS. (The full corpus run is owner-run; the completion gate below requires it be RUN and reported.)

---

### Task 7: Whole-cycle docs — rewrite the PR1-scoped release artifacts to the whole cycle

The Phase-1 commit-prep (done in a prior session) wrote CHANGELOG/TODO/CLAUDE.md/ADR-092 framing PR1 as a shipped release with PR2 "a separate later cycle / NOT in this release." That framing is WRONG for a one-cycle/one-commit deliverable and MUST be corrected here.

**Files:** `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `docs/superpowers/adrs/ADR-092-tf62-gk-buildup-decision-value.md`, `NOTICE`, `silly_kicks/feature_glossary.py` (verify only).

- [ ] **Step 0: SPEC reconciliation (SHOULD-FIX TF62-PLAN-06 — the design-of-record still says the opposite).** The spec's §0 ("Two fully-validated PRs, each one coherent tested commit ... owner approval gate before each commit") and §12.2 ("ship all three tiers, native-first, in two fully-validated PRs") are UNCHANGED and labelled owner-approved — they contradict the one-cycle/one-commit decision this plan executes, so a later "match the spec" could wrongly re-split. Rewrite the two-PR/two-commit language in §0 and §12.2 to the one-cycle/one-commit reality (**native-first is preserved as a BUILD ORDER, not two commits**), and record the decision + rationale (no provenance split — the metric bundles no owner-run weights) as a new resolved-decision `§12.7`. The spec must record the shipped history, not contradict it. (This does NOT change the decision — only makes the design-of-record consistent with it.)
- [ ] **Step 1: ADR-092** — retitle from "(native anchor, PR1)" to the whole metric; the Decision section describes the full metric (native + reconstruction tiers) shipped in ONE commit; remove the "PR1 / PR2 later" split language; keep the honest-limit + reported-not-gated posture; add the reconstruction consequences (`action_ltr_goal_map` promotion, `territorial_defense` migration, the reachability filter, the fidelity/SB360 validation legs actually run).
- [ ] **Step 2: CHANGELOG `[4.113.0]`** — rewrite to the whole cycle: native + SB360 + full-tracking reconstruction, the reachability filter, `tracking.action_ltr_goal_map` promoted (ADR-055) with `territorial_defense` migrated, and the reconstruction-fidelity/SB360-responsiveness validation results (fill in the RUN numbers). Delete the "NOT in this release (PR2)" bullet.
- [ ] **Step 3: CLAUDE.md `gk_decision` bullet** — update to describe the shipped reconstruction tiers (drop the "PR2 (NOT in 4.113.0)" clause); note the `action_ltr_goal_map` public promotion + `territorial_defense` migration; keep the honest-limit verbatim in spirit; note the import-allowlist is now the tracking-consuming-sibling shape.
- [ ] **Step 4: TODO.md** — the TF-62 On-Deck row is now FULLY shipped → REMOVE it entirely (no breadcrumbs, per grooming). Update the top summary to the whole-cycle description. (If a TF-62 follow-on exists — e.g. a per-provider xPass recalibration apply, ADR-009 — add it ONLY with explicit owner approval; do not invent backlog.)
- [ ] **Step 5: NOTICE** — extend the TF-62 entry to name the reconstruction seams (packing progression, `PassCompletionModel` as injected xPass, the action-LTR goal map).
- [ ] **Step 6: feature_glossary** — VERIFY no new columns (reconstruction emits the same 5 metric columns); `len(FEATURE_GLOSSARY)` stays 399; the C4 count stays 399. Run `python -m pytest tests/test_c4_feature_column_count.py tests/invariants/test_glossary_emitted_columns.py -v`.

---

## Whole-cycle completion gate (NOT a per-task commit)

- [ ] Full non-e2e suite green at CI scope — **run streamed, not piped through `tail`** (visible progress): `python -m pytest tests/ -m "not e2e" -p no:randomly` (tee to a file), then `python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check silly_kicks/ tests/ scripts/`, `python -m pyright`.
- [ ] C4 diagram regenerated via Graphviz `dot` (per the CLAUDE.md pipeline); confirm the `gk_decision` container is present and the aggregator/feature-column counts are unchanged (33 / 399).
- [ ] **Owner-run the battery on the owner-tier corpus — ALL legs — and confirm the reconstruction probes actually pass:** reconstruction fidelity ρ (reconstructed-vs-native on the SkillCorner Rosetta Stone), SB360 responsiveness (chosen vs random placebo), reachability two-sided sweep (naive fails, filtered responds), plus the Phase-1 legs (discrimination, net-of-team, transfer). Fill the RUN numbers into the CHANGELOG/ADR (Task 7). **The cycle is not done until these are green + reported.**
- [ ] **Present the full whole-cycle diff + test/lint output + the battery report to Karsten and request explicit approval. On approval: ONE coherent commit for the ENTIRE cycle (Phase 1 + Phase 2).** (No commit before this gate; no micro-commits.)

---

## Self-Review

**Spec coverage (PR2 scope, spec §0/§5/§7/§9/§11/§12.6):** `ReconstructedOptionSet` SB360 + full-tracking (Task 2/4) ✓; reachability filter as a first-class `GkDecisionParams` field consumed by the adapter (Task 2) ✓; packing reuse via `compute_packing_metrics["packing_made"]` fed an action-LTR `GoalMap` (Task 2) ✓; promoted public `tracking.action_ltr_goal_map` + `territorial_defense` migration (Task 1) ✓; chosen = actual pass end never snapped + receiver-exclusion (Task 2/3) ✓; orientation mirror-invariance + magnitude/packing-boundary + reachability two-sided + chosen-into-space gates (Task 3) ✓; conservation incl. `no_frame`/`fov_cropped` (Task 2/3) ✓; battery fidelity + SB360-responsiveness + reachability-sweep legs (Task 6) ✓; import-allowlist relaxed to tracking-consuming shape (Task 5) ✓; whole-cycle release artifacts corrected (Task 7) ✓.

**Placeholder scan:** Task 2 Step 3's `_rows_for_decision` is specified as an 8-point algorithm with the exact seams named (not "TBD"); its behaviour is pinned by Task 2 Step 1 + all of Task 3. Task 6 Step 3's corpus orchestration is owner-run (not CI) — its testable kernels carry known-truth code (Step 1). Task 4's fixture is "reuse if one exists, else build minimal + document" (a real instruction, checked by grep first). No "handle edge cases"/"add validation" placeholders.

**Type consistency:** `ReconstructedOptionSet.option_rows()` yields `OPTION_ROW_COLUMNS` (Task 2) — the SAME schema `SkillCornerGIOptionSet` (Phase 1) yields and `compute_gk_decision_value` (Phase 1) consumes; `option_value` reads `completion`/`opponents_bypassed` from it (Phase 1). `drop_counts()` → `compute_gk_decision_value(..., extra_drops=)` → `GkDecisionReport` counters (Task 2 Step 5) align. `action_ltr_goal_map` signature is byte-identical across the move (Task 1). `compute_packing_metrics(..., goal_map=, passer_xy=, receiver_xy=)["packing_made"]` matches the verified live signature. `PassCompletionModel.predict_completion(ox,oy,tx,ty)->np.ndarray` matches spec §13.

**Scope check:** one cycle, one commit; native (Phase 1) + reconstruction (Phase 2) as a single deliverable; nothing deferred (the value-fn xT/retention variants stay reserved typed doors per the approved spec §12.5; the per-provider xPass recalibration is an explicit future ADR-009 apply, not silently parked). Full-tracking reconstruction is UNVALIDATED in the spike but IS validated in-cycle by the Rosetta-Stone fidelity leg (Task 6) — the native-first ordering exists precisely to make that comparison possible.

## Review disposition (plan R1 — 2026-09-12)

Independent plan review: REQUEST CHANGES, no blocking, "don't change the single-commit decision, architecture, or native-first ordering." All addressed in-place; architecture/decision/ordering unchanged.

| Finding | Disposition |
|---|---|
| **SHOULD TF62-PLAN-06** — the SPEC (§0, §12.2) still says "two PRs / two commits", contradicting the one-commit decision; Task 7 rewrote the other artifacts but omitted the spec | Task 7 **Step 0** added: rewrite §0/§12.2 to one-cycle/one-commit (native-first preserved as a build order), record the decision + no-provenance-split rationale as spec §12.7. The design-of-record must not contradict the shipped history. |
| CONSIDER-7 — "the ADR-077 seam" unnamed (it decides a drop reason) | Task 2 Step 3 pt3 now names the PUBLIC `silly_kicks.tracking.region_observed_fraction(polygon, region)` + the convex-disk-as-region / honest-NaN contract. |
| CONSIDER-8 — spec §9 leg 7 (per-provider xPass recalibration) absent from Task 6 | Task 6 note added: the MEASUREMENT (mean completion offset) rides the fidelity leg (in-cycle); the recalibration APPLY is a future ADR-009 (spec §8.4/§12.4), deferred by design, not dropped. |
| CONSIDER-9 — Task 2 Step 3 is a dense 8-point algo in one step | Factored: steps 1–3 → the testable helper `_frame_and_anchors_action_ltr` (exercised by Task 3's orientation gate); steps 4–8 → the valuation core (Task 2 Step 1 + Task 3 magnitude/reachability) — the two red-green separately. |
| CONSIDER-10 — new actions↔frames id-match surface unhardened | Task 3 **Step 6** added: a keeper↔frame cross-dtype (`int` vs `str`) match-and-score gate (ADR-019 hardening of the specific new surface). |
| COULD-NOT-VERIFY — C4 counts post-regen / battery numbers (NDA) / full-tracking behaviour | C4 counts are the completion-gate's job (no-new-aggregator sound); battery numbers are owner-run; full-tracking is validated in-cycle by the fidelity leg. |

**R2 (2026-09-12) — APPROVE.** SHOULD-FIX + both actionable CONSIDERs resolved; the two optional CONSIDERs (PLAN-09 split / PLAN-10 id-mismatch test) acceptably declined. Two non-blocking nits closed anyway: **PLAN-11** — the FOV gate's `fov_radius_m`/`fov_min_observed_fraction` are now named `GkDecisionParams` fields (Task 2), not an unspecified floor; and the Task-7 File-Structure row now lists the spec (Step 0 edits it). Ready to execute; the owner-approval commit gate still governs the commit.
