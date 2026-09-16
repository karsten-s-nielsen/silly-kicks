# TF-51 Item 4 — Defensive-credit atomic-SPADL mirror — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development`
> (recommended) or `superpowers:executing-plans` to implement task-by-task. Steps use checkbox
> (`- [ ]`) syntax for tracking.

**Goal:** Ship the atomic-SPADL mirror of the TF-51 defensive-credit family (`compute_defensive_credits`,
`add_defensive_credit`, `compute_bravery`) plus the co-shipped `atomic/spadl/config.py` `interception`
dedup, faithful to the atom representation.

**Architecture:** A pure `_defensive_credit_atomic_adapter` re-lifts the atom stream into a std-shaped
stream (endpoints from `x,y,dx,dy`; std `type_id` for the six rule-anchor types; a per-type
next-atom `result_id`), the existing std sub-package does the credit work unchanged, and the aggregate
is assembled on a copy of the caller's atomic frame via a newly-extracted shared rollup. Bravery
delegates to the std event-only computation and honest-NaNs the atomic-uncomputable set-piece columns.

**Tech Stack:** Python, pandas, numpy. `silly_kicks.atomic.tracking.features`,
`silly_kicks.tracking.defensive_credit`, `silly_kicks.atomic.spadl`.

**Spec:** `docs/superpowers/specs/2026-09-15-tf51-item4-defensive-credit-atomic-mirror-design.md`
(Approved, r2). Executors read both.

---

## Global Constraints

Copied verbatim from the spec + CLAUDE.md. Every task implicitly includes these.

- **Faithful, not std-parity.** Delegate on the atom stream; never assert byte-equality with std.
- **No `*_xfns`** for either representation (F4 result-leakage). The absence guard stays green.
- **Honest-NaN, never a fabricated sentinel** (ADR-027): atomic-uncomputable → `NaN`/`pd.NA`, not `0`.
- **Symbolic ids only** — `spadlconfig.actiontype_id[...]` / `atomicconfig.actiontype_id[...]`, never raw
  ints (a future renumber must not silently break the map).
- **Fail-loud on a missing injected column** (ADR-043) — never an all-NaN credit column.
- **Purity (ADR-033):** any new `add_*` is pure (no caller-input mutation, returns a new object); a
  conditional-column `add_*` registers ≥2 purity variants.
- **Commit discipline (CLAUDE.md — overrides the writing-plans per-step "commit" cadence):** NO
  micro-commits. Tasks end at a green-tested deliverable and **do not commit**. There is exactly ONE
  commit for the whole change, in Task 10, behind an explicit human-approval gate. One feature branch
  off `main`; no worktree.
- **Green bar before proposing the commit:** `python -m pytest tests/ -m "not e2e"` green;
  `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/
  tests/ scripts/` clean; `python -m pyright` clean.
- **Migration owed (Task 1, recorded in CHANGELOG):** atomic `type_id` tail renumber → atomic-SPADL
  re-materialize **paired with an AtomicVAEP retrain** (the default `xfns_default` encodes those ids).

---

## File structure

| File | Responsibility | Tasks |
|---|---|---|
| `silly_kicks/atomic/spadl/config.py` | Remove duplicate `"interception"`; tail renumber | 1 |
| `silly_kicks/atomic/vaep/features.py` | Fix `actiontype_onehot` docstring "33"→32 | 1 |
| `docs/c4/architecture.dsl` + `architecture.html` | "33-type"→"32-type" + regen | 1 |
| `tests/atomic/...` fixtures | Regenerate any carrying tail ids 24–32 | 1 |
| `silly_kicks/tracking/defensive_credit/_orchestration.py` | Extract `_rollup_defending_aggregate` | 3 |
| `silly_kicks/atomic/tracking/features.py` | `_defensive_credit_atomic_adapter`, `compute_defensive_credits`, `_aggregate_defensive_credit`, `add_defensive_credit`, `compute_bravery` | 2,4,5,6 |
| `tests/atomic/test_atomic_defensive_credit.py` | All new behavioral tests | 2,4,5,6,8 |
| `tests/test_add_star_purity.py` | Register `atomic.tracking:add_defensive_credit` (2 variants); header 15→16 | 7 |
| `CHANGELOG.md`, `silly_kicks/_version.py`, `docs/superpowers/adrs/ADR-0NN-*.md`, `TODO.md` | Release bookkeeping | 9,10 |

---

## Task 1: Co-shipped config fix — `interception` dedup + tail renumber

**Files:**
- Modify: `silly_kicks/atomic/spadl/config.py:28-40` (the `actiontypes` list)
- Modify: `silly_kicks/atomic/vaep/features.py:199` (docstring "33 boolean columns" → 32)
- Modify: `docs/c4/architecture.dsl:24` ("33-type" → "32-type"), regen `docs/c4/architecture.html`
- Test: `tests/atomic/test_atomic_spadl.py` (id-map guard)
- Regenerate (if they carry tail ids): `tests/datasets/spadl/atomic_spadl.json`,
  `tests/atomic/_golden_atomic_pre_shot_gk_context_v280.parquet`

**Interfaces:**
- Produces: `atomicconfig.actiontype_id["interception"] == 10`; tail ids `out=24, offside=25, goal=26,
  owngoal=27, yellow_card=28, red_card=29, corner=30, freekick=31`; `len(atomicconfig.actiontypes) == 32`.

- [ ] **Step 1: Write failing guard test.**

```python
# tests/atomic/test_atomic_spadl.py
import silly_kicks.atomic.spadl.config as ac

def test_interception_is_not_duplicated():
    assert ac.actiontypes.count("interception") == 1
    assert ac.actiontype_id["interception"] == 10          # inherited std index, no shadow
    assert len(ac.actiontypes) == 32

def test_tail_ids_after_dedup():
    expected = {"receival": 23, "out": 24, "offside": 25, "goal": 26, "owngoal": 27,
                "yellow_card": 28, "red_card": 29, "corner": 30, "freekick": 31}
    for name, idx in expected.items():
        assert ac.actiontype_id[name] == idx
```

- [ ] **Step 2: Run — verify it fails** (`interception` at 24, len 33).

Run: `python -m pytest tests/atomic/test_atomic_spadl.py::test_interception_is_not_duplicated tests/atomic/test_atomic_spadl.py::test_tail_ids_after_dedup -v`
Expected: FAIL (`actiontype_id["interception"] == 24`, `len == 33`).

- [ ] **Step 3: Remove the duplicate `"interception"`** from `atomic/spadl/config.py::actiontypes`:

```python
actiontypes = [
    *_spadl.actiontypes,
    "receival",
    # "interception" removed -- inherited from _spadl.actiontypes[10]; the appended
    # duplicate made the reverse dict resolve interception->24, shadowing std interception (idx 10).
    "out",
    "offside",
    "goal",
    "owngoal",
    "yellow_card",
    "red_card",
    "corner",
    "freekick",
]
```

- [ ] **Step 4: Run — verify Step-1 tests pass.**

Run: `python -m pytest tests/atomic/test_atomic_spadl.py -v`
Expected: PASS.

- [ ] **Step 5: Fix the stale docstring.** In `atomic/vaep/features.py:199`, change
  `# feats has 33 boolean columns per slot (one per atomic action type).` → `32`.

- [ ] **Step 6: Enumerate + regenerate serialized fixtures.** Find every committed fixture derived
  from `convert_to_atomic` and check for `type_id` in the 24–32 tail:

Run: `python -c "import pandas as pd; df=pd.read_parquet('tests/atomic/_golden_atomic_pre_shot_gk_context_v280.parquet'); print(sorted(df['type_id'].unique()))"`
Run: `python -c "import json; d=json.load(open('tests/datasets/spadl/atomic_spadl.json')); print(sorted({r['type_id'] for r in d}))"` (adapt to the file's actual shape)

For each fixture carrying tail ids, regenerate it via its committed generator (or a minimal
`convert_to_atomic` re-run), and **prove the diff is purely the renumber** (only tail `type_id`s move;
no other column changes). If a fixture carries NO tail id, leave it untouched and record that.

- [ ] **Step 7: Update the C4 prose + regen.** Edit `docs/c4/architecture.dsl:24`
  `continuous 33-type action representation` → `continuous 32-type action representation`; regenerate
  `docs/c4/architecture.html` via the C4 pipeline (Graphviz `dot`, per CLAUDE.md C4 section).

- [ ] **Step 8: Run the atomic + VAEP suites — verify green** (catches any consumer that regressed on
  the renumber).

Run: `python -m pytest tests/atomic/ -m "not e2e" -v`
Expected: PASS. Do NOT commit.

---

## Task 2: `_defensive_credit_atomic_adapter`

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py` (add the adapter beside `_packing_atomic_adapter`)
- Test: `tests/atomic/test_atomic_defensive_credit.py`

**Interfaces:**
- Consumes: `_structural_pass_atomic_endpoints` (`atomic/tracking/features.py:143`),
  `atomicconfig.actiontype_id`, `spadlconfig.actiontype_id`, `spadlconfig.result_id`,
  `ids_equal` (already imported).
- Produces: `_defensive_credit_atomic_adapter(actions: pd.DataFrame, params: DefensiveCreditParams)
  -> pd.DataFrame` — a NEW std-shaped frame with `start_x/y`, `end_x/y`, std `type_id`, synth
  `result_id`; caller frame never mutated.

- [ ] **Step 1: Write failing tests** (endpoints, type map, per-type result table):

```python
# tests/atomic/test_atomic_defensive_credit.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks.atomic.spadl import config as ac
from silly_kicks.spadl import config as sc
from silly_kicks.tracking.defensive_credit import DefensiveCreditParams
from silly_kicks.atomic.tracking.features import _defensive_credit_atomic_adapter as A

def _atom(type_name, team=1, player=10, x=50.0, y=34.0, dx=5.0, dy=0.0, gid=1, pid=1, aid=0, t=0.0):
    return dict(game_id=gid, period_id=pid, action_id=aid, time_seconds=t, team_id=team,
                player_id=player, x=x, y=y, dx=dx, dy=dy,
                type_id=ac.actiontype_id[type_name], bodypart_id=0)

def _frame(rows):
    return pd.DataFrame(rows)

def test_endpoints_synthesized():
    out = A(_frame([_atom("pass", x=10.0, y=20.0, dx=3.0, dy=-4.0)]), DefensiveCreditParams())
    assert out.loc[0, "start_x"] == 10.0 and out.loc[0, "start_y"] == 20.0
    assert out.loc[0, "end_x"] == 13.0 and out.loc[0, "end_y"] == 16.0

def test_domain_type_map_and_nonaction():
    rows = [_atom("pass", aid=0), _atom("receival", aid=1)]
    out = A(_frame(rows), DefensiveCreditParams())
    assert out.loc[0, "type_id"] == sc.actiontype_id["pass"]
    assert out.loc[1, "type_id"] == sc.actiontype_id["non_action"]   # atomic-only atom off-domain

def test_pass_success_iff_next_receival():
    completed = A(_frame([_atom("pass", aid=0), _atom("receival", aid=1)]), DefensiveCreditParams())
    assert completed.loc[0, "result_id"] == sc.result_id["success"]
    failed = A(_frame([_atom("pass", aid=0), _atom("interception", team=2, aid=1)]), DefensiveCreditParams())
    assert failed.loc[0, "result_id"] == sc.result_id["fail"]

def test_pass_success_to_keeper_reception():
    out = A(_frame([_atom("pass", team=1, aid=0), _atom("keeper_pick_up", team=1, aid=1)]), DefensiveCreditParams())
    assert out.loc[0, "result_id"] == sc.result_id["success"]

def test_take_on_success_iff_retained_same_team_not_lost():
    retained = A(_frame([_atom("take_on", team=1, aid=0), _atom("dribble", team=1, aid=1)]), DefensiveCreditParams())
    assert retained.loc[0, "result_id"] == sc.result_id["success"]
    lost = A(_frame([_atom("take_on", team=1, aid=0), _atom("interception", team=2, aid=1)]), DefensiveCreditParams())
    assert lost.loc[0, "result_id"] == sc.result_id["fail"]

def test_shot_success_iff_next_goal():
    scored = A(_frame([_atom("shot", aid=0), _atom("goal", aid=1)]), DefensiveCreditParams())
    assert scored.loc[0, "result_id"] == sc.result_id["success"]
    missed = A(_frame([_atom("shot", aid=0), _atom("out", aid=1)]), DefensiveCreditParams())
    assert missed.loc[0, "result_id"] == sc.result_id["fail"]

def test_period_last_atom_is_fail():
    out = A(_frame([_atom("pass", aid=0)]), DefensiveCreditParams())
    assert out.loc[0, "result_id"] == sc.result_id["fail"]

def test_adapter_does_not_mutate_caller():
    frame = _frame([_atom("pass", aid=0), _atom("receival", aid=1)])
    before = frame.copy(deep=True)
    _ = A(frame, DefensiveCreditParams())
    pd.testing.assert_frame_equal(frame, before)
```

- [ ] **Step 2: Run — verify all fail** (`_defensive_credit_atomic_adapter` not defined).

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -v`
Expected: FAIL (ImportError / not defined).

- [ ] **Step 3: Implement the adapter** in `atomic/tracking/features.py`:

```python
from silly_kicks.tracking.defensive_credit import DefensiveCreditParams  # add to the top imports

# domain types the credit rules anchor on (symbolic; NaN-safe int compares)
_DC_DOMAIN_TYPES = ("pass", "cross", "shot", "shot_penalty", "take_on", "bad_touch")

def _defensive_credit_atomic_adapter(actions: pd.DataFrame, params: DefensiveCreditParams) -> pd.DataFrame:
    """Re-lift the atom stream into a std-shaped stream for the TF-51 credit engine (faithful).

    Endpoints from x,y,dx,dy; std type for the six rule-anchor types (else non_action); a per-type
    next-atom result_id (pass/cross: receival|same-team keeper reception; take_on: retained same-team,
    not interception/out; shot/shot_penalty: next==goal; bad_touch/off-domain: fail). possession_id is
    NOT synthesized here -- with_possessions derives it downstream from time/team, not result_id.
    Pure; the caller frame is never mutated (`_structural_pass_atomic_endpoints` copies)."""
    adapted = _structural_pass_atomic_endpoints(actions)  # start/end + copy
    n = len(actions)
    type_id = actions["type_id"].to_numpy()

    std_ids = np.full(n, spadlconfig.actiontype_id["non_action"], dtype="int64")
    is_domain = np.zeros(n, dtype=bool)
    for name in _DC_DOMAIN_TYPES:
        mask = type_id == atomicconfig.actiontype_id[name]
        std_ids[mask] = spadlconfig.actiontype_id[name]
        is_domain |= mask

    next_type = np.full(n, -1.0)
    same_gp = np.zeros(n, dtype=bool)
    if n > 1:
        next_type[:-1] = type_id[1:]
        game = actions["game_id"].to_numpy()
        period = actions["period_id"].to_numpy()
        same_gp[:-1] = (game[1:] == game[:-1]) & (period[1:] == period[:-1])
    team_s = actions["team_id"].reset_index(drop=True)
    next_team_same = ids_equal(team_s, team_s.shift(-1)).to_numpy()

    recv = atomicconfig.actiontype_id["receival"]
    keeper = [atomicconfig.actiontype_id["keeper_pick_up"], atomicconfig.actiontype_id["keeper_claim"]]
    goal = atomicconfig.actiontype_id["goal"]
    interception = atomicconfig.actiontype_id["interception"]
    out_id = atomicconfig.actiontype_id["out"]

    is_pass_like = np.isin(type_id, [atomicconfig.actiontype_id["pass"], atomicconfig.actiontype_id["cross"]])
    is_take_on = type_id == atomicconfig.actiontype_id["take_on"]
    is_shot_like = np.isin(type_id, [atomicconfig.actiontype_id["shot"], atomicconfig.actiontype_id["shot_penalty"]])

    pass_ok = same_gp & ((next_type == recv) | (np.isin(next_type, keeper) & next_team_same))
    take_on_ok = same_gp & next_team_same & ~np.isin(next_type, [interception, out_id])
    shot_ok = same_gp & (next_type == goal)
    success = (is_pass_like & pass_ok) | (is_take_on & take_on_ok) | (is_shot_like & shot_ok)

    adapted["type_id"] = std_ids
    adapted["result_id"] = np.where(success, spadlconfig.result_id["success"], spadlconfig.result_id["fail"])
    return adapted
```

- [ ] **Step 4: Run — verify all Step-1 tests pass.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -v`
Expected: PASS. Do NOT commit.

---

## Task 3: Extract the shared defending-aggregate rollup (DRY refactor of the std module)

**Scope:** This is the ONE shipped-std-production-code touch in this change — **owner-approved, spec §2
item 9 / §5 option (a) (2026-09-16)** — and it is **parity-gated** (Step 4) so std `add_defensive_credit`
output is byte-identical. A pure DRY refactor, no behavior change.

**Files:**
- Modify: `silly_kicks/tracking/defensive_credit/_orchestration.py` (extract lines 215-246 body)
- Test: existing `tests/tracking/test_defensive_credit*.py` (parity — std output must be byte-identical)

**Interfaces:**
- Produces: `_rollup_defending_aggregate(actions: pd.DataFrame, long: pd.DataFrame, *, params,
  visible_area, links) -> pd.DataFrame` — takes a pre-computed long-form + the caller's actions,
  returns actions + the four aggregate columns (+ the two visible_area companions when supplied). The
  std `_aggregate_defensive_credit` becomes `long = compute_defensive_credits(...);
  return _rollup_defending_aggregate(actions, long, params=params, visible_area=visible_area, links=links)`.

- [ ] **Step 1: Write a failing parity test** proving the refactor is output-preserving:

```python
# tests/tracking/test_defensive_credit_rollup_refactor.py
# Build a small credits fixture (reuse an existing defensive-credit test fixture builder),
# call add_defensive_credit BEFORE and AFTER wiring through _rollup_defending_aggregate,
# assert byte-identical. (Red first: _rollup_defending_aggregate not yet importable.)
from silly_kicks.tracking.defensive_credit._orchestration import _rollup_defending_aggregate  # noqa
def test_rollup_symbol_exists():
    assert callable(_rollup_defending_aggregate)
```

- [ ] **Step 2: Run — verify it fails** (ImportError).

Run: `python -m pytest tests/tracking/test_defensive_credit_rollup_refactor.py -v`
Expected: FAIL.

- [ ] **Step 3: Extract the rollup.** Move `_orchestration.py:215-246` (the `out = actions.copy()` …
  `return out` body, including the `visible_area` companion branch) into:

```python
def _rollup_defending_aggregate(actions, long, *, params, visible_area=None, links=None):
    """Assemble the per-action defending-team aggregate from a pre-computed long-form. Pure.

    Splits the caller's OWN `actions` frame (never a synthesized/adapted one) into net/plus/minus/n
    over the defending credits, and appends the ADR-077 FOV companions when `visible_area` is given.
    Shared verbatim by the standard and the atomic `_aggregate_defensive_credit` -- the ONE rollup."""
    out = actions.copy()
    act_team = actions.set_index("action_id")["team_id"]
    if long.empty:
        defending = long
    else:
        long = long.copy()
        long["_acting_team"] = long["action_id"].map(act_team)
        keep = ids_differ(long["team_id"], long["_acting_team"])
        defending = long[keep.to_numpy()]
    # ... (net/plus/minus/cnt + the visible_area branch, moved verbatim) ...
    return out
```

Then rewrite the std `_aggregate_defensive_credit` to call it:

```python
def _aggregate_defensive_credit(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
                                on_target_column="shot_on_target_derived", links=None, params=None,
                                visible_area=None):
    params = params or DefensiveCreditParams()
    long = compute_defensive_credits(actions, frames, xg_column=xg_column, xt=xt,
                                     blocked_column=blocked_column, on_target_column=on_target_column,
                                     links=links, params=params)
    return _rollup_defending_aggregate(actions, long, params=params, visible_area=visible_area, links=links)
```

- [ ] **Step 4: Run the FULL std defensive-credit suite — verify byte-identical (no behavior change).**

Run: `python -m pytest tests/tracking/ -k defensive_credit -m "not e2e" -v`
Expected: PASS (all existing tests unchanged). Do NOT commit.

---

## Task 4: Atomic `compute_defensive_credits` + preserve_native required-column raise

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/atomic/test_atomic_defensive_credit.py`

**Interfaces:**
- Consumes: `_defensive_credit_atomic_adapter` (Task 2); std `compute_defensive_credits`.
- Produces: `compute_defensive_credits(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
  on_target_column="shot_on_target_derived", links=None, params=None) -> pd.DataFrame` (long-form,
  atomic entry).

- [ ] **Step 1: Write failing tests** (public return + required-column raise):

```python
def test_missing_xg_column_raises(single_atomic_scene, frames_fixture):  # helpers below / reuse
    import silly_kicks.atomic.tracking.features as atf
    with pytest.raises(ValueError, match="xg"):
        atf.compute_defensive_credits(single_atomic_scene.drop(columns=["xg"]),
                                      frames_fixture, xg_column="xg", xt=fitted_xt)

def test_long_form_credits_through_public_entry(pressured_failed_pass_atomic, frames_fixture, fitted_xt):
    import silly_kicks.atomic.tracking.features as atf
    long = atf.compute_defensive_credits(pressured_failed_pass_atomic, frames_fixture,
                                         xg_column="xg", xt=fitted_xt)
    assert set(long["rule"]) & {"pressure_pass_fail"}          # the scripted rule fired
    assert (long[long["rule"] == "pressure_pass_fail"]["signed_value"].abs() > 0).all()
```

(Build `pressured_failed_pass_atomic` as a hand-scripted atomic fixture: a failed pass with a nearby
opponent defender in the linked frame, carrying a `preserve_native` `xg` column and `shot_blocked`.)

- [ ] **Step 2: Run — verify fail.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k "compute or missing_xg" -v`
Expected: FAIL.

- [ ] **Step 3: Implement.**

```python
def _require_columns(actions, required, *, fn):
    missing = [c for c in required if c not in actions.columns]
    if missing:
        raise ValueError(
            f"{fn} (atomic): required column(s) {missing} absent. Injected analytics are not "
            f"atom-derivable -- thread them through the conversion: "
            f"convert_to_atomic(std_actions, preserve_native=['xg', 'shot_blocked', 'cross_blocked'])."
        )

def compute_defensive_credits(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
                              on_target_column="shot_on_target_derived", links=None, params=None):
    """Atomic-SPADL mirror of tracking.defensive_credit.compute_defensive_credits (TF-51, faithful).

    Re-lifts the atom stream (`_defensive_credit_atomic_adapter`) and delegates to the std engine;
    with_possessions runs INSIDE the std entry on the adapted (result_id-bearing) stream. See NOTICE.
    Window params (resulting_shot_max_actions / recovery_max_actions) count ATOM rows here -- denser
    than std actions (faithful, per the spec)."""
    from silly_kicks.tracking.defensive_credit import compute_defensive_credits as _std
    params = params or DefensiveCreditParams()
    _require_columns(actions, [xg_column, blocked_column], fn="compute_defensive_credits")
    adapted = _defensive_credit_atomic_adapter(actions, params)
    return _std(adapted, frames, xg_column=xg_column, xt=xt, blocked_column=blocked_column,
                on_target_column=on_target_column, links=links, params=params)
```

- [ ] **Step 4: Run — verify pass.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k "compute or missing_xg" -v`
Expected: PASS. Do NOT commit.

---

## Task 5: Atomic `_aggregate_defensive_credit` + `add_defensive_credit`

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/atomic/test_atomic_defensive_credit.py`

**Interfaces:**
- Consumes: atomic `compute_defensive_credits` (Task 4); `_rollup_defending_aggregate` (Task 3);
  `link_actions_to_frames`.
- Produces: `add_defensive_credit(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
  on_target_column="shot_on_target_derived", links=None, params=None, visible_area=None) -> pd.DataFrame`
  — atomic actions + `defensive_credit_net/_plus/_minus` + `n_defensive_credits` (+ FOV companions when
  `visible_area`); the 16th atomic.tracking feature mirror.

- [ ] **Step 1: Write failing tests** (aggregate columns; no synth-column leak; purity via public API;
  visible_area companion):

```python
def test_aggregate_columns_and_no_synth_leak(pressured_failed_pass_atomic, frames_fixture, fitted_xt):
    import silly_kicks.atomic.tracking.features as atf
    before = pressured_failed_pass_atomic.copy(deep=True)
    out = atf.add_defensive_credit(pressured_failed_pass_atomic, frames_fixture, xg_column="xg", xt=fitted_xt)
    # aggregate columns present + finite
    for c in ["defensive_credit_net", "defensive_credit_plus", "defensive_credit_minus", "n_defensive_credits"]:
        assert c in out.columns
    # atomic columns intact, NO synth std columns leaked
    assert "start_x" not in out.columns and "result_id" not in out.columns
    assert {"x", "y", "dx", "dy"}.issubset(out.columns)
    # purity: caller not mutated
    pd.testing.assert_frame_equal(pressured_failed_pass_atomic, before)

def test_visible_area_companion_additive(pressured_failed_pass_atomic, frames_fixture, fitted_xt, polygons):
    import silly_kicks.atomic.tracking.features as atf
    base = atf.add_defensive_credit(pressured_failed_pass_atomic, frames_fixture, xg_column="xg", xt=fitted_xt)
    withva = atf.add_defensive_credit(pressured_failed_pass_atomic, frames_fixture, xg_column="xg",
                                      xt=fitted_xt, visible_area=polygons)
    for c in ["defensive_credit_net", "defensive_credit_plus", "defensive_credit_minus", "n_defensive_credits"]:
        pd.testing.assert_series_equal(base[c], withva[c])   # primary columns byte-identical
    assert {"defensive_credit_observed_fraction", "defensive_credit_observed_source"}.issubset(withva.columns)
```

- [ ] **Step 2: Run — verify fail.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k "aggregate or visible_area" -v`
Expected: FAIL.

- [ ] **Step 3: Implement** (assemble on the CALLER's atomic frame via the shared rollup):

```python
def _aggregate_defensive_credit(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
                                on_target_column="shot_on_target_derived", links=None, params=None,
                                visible_area=None):
    from silly_kicks.tracking.defensive_credit._orchestration import _rollup_defending_aggregate
    params = params or DefensiveCreditParams()
    long = compute_defensive_credits(actions, frames, xg_column=xg_column, xt=xt,
                                     blocked_column=blocked_column, on_target_column=on_target_column,
                                     links=links, params=params)
    # `actions` is the CALLER's atomic frame -> rollup assembles on it; no synth column leaks.
    return _rollup_defending_aggregate(actions, long, params=params, visible_area=visible_area, links=links)

def add_defensive_credit(actions, frames, *, xg_column, xt, blocked_column="shot_blocked",
                         on_target_column="shot_on_target_derived", links=None, params=None,
                         visible_area=None):
    """Atomic-SPADL mirror of tracking.add_defensive_credit (TF-51). See NOTICE. Faithful; no *_xfns."""
    from silly_kicks.tracking.utils import link_actions_to_frames
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]
    out = _aggregate_defensive_credit(actions, frames, xg_column=xg_column, xt=xt,
                                      blocked_column=blocked_column, on_target_column=on_target_column,
                                      links=pointers, params=params, visible_area=visible_area)
    provenance_cols = ["frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]
    if not any(c in out.columns for c in provenance_cols) and len(pointers) > 0:
        ptr = pointers.drop_duplicates("action_id").set_index("action_id")[provenance_cols]
        out = out.merge(ptr, left_on="action_id", right_index=True, how="left")
    return out
```

- [ ] **Step 4: Run — verify pass.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k "aggregate or visible_area" -v`
Expected: PASS. Do NOT commit.

---

## Task 6: Atomic `compute_bravery` (honest-NaN set-piece)

**Files:**
- Modify: `silly_kicks/atomic/tracking/features.py`
- Test: `tests/atomic/test_atomic_defensive_credit.py`

**Interfaces:**
- Consumes: std `compute_bravery`.
- Produces: `compute_bravery(actions, *, shot_blocked_column="shot_blocked",
  cross_blocked_column="cross_blocked") -> pd.DataFrame`.

- [ ] **Step 1: Write failing tests:**

```python
def test_bravery_setpiece_is_honest_nan(atomic_bravery_scene):
    # atomic_bravery_scene: shots + open-play crosses (some blocked) + a collapsed set-piece,
    # carrying preserve_native shot_blocked + cross_blocked.
    import silly_kicks.atomic.tracking.features as atf
    b = atf.compute_bravery(atomic_bravery_scene)
    assert b["bravery_set_piece_crosses"].isna().all()
    assert b["n_set_piece_crosses_faced"].isna().all()          # NOT 0 (ADR-027)
    assert b["bravery_shots"].notna().any()                     # shots computed
    assert b["bravery_pct_known_domain"].notna().any()          # headline survives

def test_bravery_missing_cross_blocked_raises(atomic_bravery_scene):
    import silly_kicks.atomic.tracking.features as atf
    with pytest.raises(ValueError, match="cross_blocked"):
        atf.compute_bravery(atomic_bravery_scene.drop(columns=["cross_blocked"]))
```

- [ ] **Step 2: Run — verify fail.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k bravery -v`
Expected: FAIL.

- [ ] **Step 3: Implement:**

```python
def compute_bravery(actions, *, shot_blocked_column="shot_blocked", cross_blocked_column="cross_blocked"):
    """Atomic-SPADL mirror of tracking.defensive_credit.compute_bravery (TF-51, event-only). See NOTICE.

    shot + open-play cross survive atomic intact (shared std ids), so their bravery is exact. Atomic
    `_simplify` collapses corner_crossed/freekick_crossed into corner/freekick, so set-piece crosses are
    UNOBSERVABLE -> `bravery_set_piece_crosses` = NaN and `n_set_piece_crosses_faced` = <NA> (ADR-027;
    never a conflated crossed+short overcount)."""
    from silly_kicks.tracking.defensive_credit import compute_bravery as _std
    _require_columns(actions, [shot_blocked_column, cross_blocked_column], fn="compute_bravery")
    out = _std(actions, shot_blocked_column=shot_blocked_column, cross_blocked_column=cross_blocked_column)
    if not out.empty:
        out = out.copy()
        out["bravery_set_piece_crosses"] = np.nan
        out["n_set_piece_crosses_faced"] = pd.array([pd.NA] * len(out), dtype="Int64")
    return out
```

- [ ] **Step 4: Run — verify pass.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k bravery -v`
Expected: PASS. Do NOT commit.

---

## Task 7: Gate registrations

**Files:**
- Modify: `tests/test_add_star_purity.py` (register `atomic.tracking:add_defensive_credit`; header 15→16)
- Test: the meta-assertions in `test_add_star_purity.py` + the no-xfns absence guard + glossary +
  C4 completeness gates (run, expect green — no edits needed)

**Interfaces:**
- Consumes: `add_defensive_credit` (Task 5).

- [ ] **Step 1: Register the purity entry (2 variants — the conditional-column contract, ADR-033).**

```python
# in PURITY_ENTRIES, atomic.tracking block:
"atomic.tracking:add_defensive_credit": [
    ("defaults", _axg_inputs, lambda i: atf.add_defensive_credit(i[0], i[1], xg_column="xg", xt=i[2])),
    ("with_visible_area", _axg_va_inputs,
     lambda i: atf.add_defensive_credit(i[0], i[1], xg_column="xg", xt=i[2], visible_area=i[3])),
],
```

(Reuse / add the atomic xg+frames+xt input builders next to the existing `_axtf_inputs`; the
`with_visible_area` builder mirrors `_action_context_va_inputs`.)

- [ ] **Step 2: Bump the header comment** `15 feature mirrors below` → `16 feature mirrors below`.

- [ ] **Step 3: Run the purity gate + its meta-assertion + the surface-pinned gates.**

Run: `python -m pytest tests/test_add_star_purity.py -v`
Run: `python -m pytest -k "xfns_absence or feature_glossary_coverage or c4" -m "not e2e" -v`
Expected: PASS (purity green incl. the new 2-variant entry; no-xfns absence guard green — defensive
credit ships none in either representation; glossary green — mirror emits only already-documented
names; C4 completeness green — atomic subpackage already modeled). Do NOT commit.

---

## Task 8: Faithful-representation limitation pins

**Files:**
- Test: `tests/atomic/test_atomic_defensive_credit.py`

- [ ] **Step 1: Write the limitation tests** (documented behaviors, pinned so a future change is
  deliberate):

```python
def test_shot_freekick_not_a_resulting_shot_on_atomic(take_on_then_shot_freekick_atomic, frames_fixture, fitted_xt):
    # a beaten take_on whose only following shot is a shot_freekick (collapsed to atomic `freekick`)
    # -> NOT detected as a resulting shot, so beaten_1v1 does not fire (faithful limitation).
    import silly_kicks.atomic.tracking.features as atf
    long = atf.compute_defensive_credits(take_on_then_shot_freekick_atomic, frames_fixture,
                                         xg_column="xg", xt=fitted_xt)
    assert "beaten_1v1" not in set(long["rule"])

def test_recovery_fires_on_distance1_interception_atom(failed_pass_then_opp_interception_atomic, frames_fixture, fitted_xt):
    import silly_kicks.atomic.tracking.features as atf
    long = atf.compute_defensive_credits(failed_pass_then_opp_interception_atomic, frames_fixture,
                                         xg_column="xg", xt=fitted_xt)
    assert "recovery_double_credit" in set(long["rule"])   # atom-dense recovery
```

- [ ] **Step 2: Run — implement any fixture builders needed; verify pass.**

Run: `python -m pytest tests/atomic/test_atomic_defensive_credit.py -k "freekick or recovery" -v`
Expected: PASS. Do NOT commit.

---

## Task 9: Full green bar + release bookkeeping (NO commit)

**Files:**
- Modify: `CHANGELOG.md`, `silly_kicks/_version.py`, `TODO.md`
- Create: `docs/superpowers/adrs/ADR-0NN-tf51-item4-defensive-credit-atomic-mirror.md`

- [ ] **Step 1: Full non-e2e suite.**

Run: `python -m pytest tests/ -m "not e2e"`
Expected: PASS (all shards). Fix any regression before proceeding.

- [ ] **Step 2: Lint + format + types at CI scope.**

Run: `python -m ruff check silly_kicks/ tests/ scripts/`
Run: `python -m ruff format --check silly_kicks/ tests/ scripts/`
Run: `python -m pyright`
Expected: all clean.

- [ ] **Step 3: Write the ADR** (representation-adapter decision; faithful mechanism; the atomic
  `type_id` renumber migration + **AtomicVAEP retrain owed**; no `*_xfns`, no defensive-credit retrain;
  bravery honest-NaN; C4 "32-type"). Number it the next free `ADR-0NN`.

- [ ] **Step 4: CHANGELOG entry** (new `PR-Snnn`): the atomic mirror (`compute_defensive_credits`,
  `add_defensive_credit`, `compute_bravery`), the co-shipped `interception` dedup + tail renumber
  (**breaking**: atomic `type_id` renumber → atomic-SPADL re-materialize **paired with AtomicVAEP
  retrain**), faithful-representation limitations, "additive to VAEP for the credit family, no
  `*_xfns`, no credit-family retrain". Bump `silly_kicks/_version.py` (next free minor).

- [ ] **Step 5: Update `TODO.md`** — mark TF-51 Item 4 shipped; leave Track B / Item-4-siblings intact.

Do NOT commit.

---

## Task 10: Human-approval commit gate

- [ ] **Step 1: Show the full diff / file list to Karsten** (`git status` + `git diff --stat` + the
  new/regenerated files, incl. the C4 html and any regenerated fixtures).
- [ ] **Step 2: STOP. Wait for Karsten's explicit "yes" to this specific commit.** No spec, plan, or
  green suite authorizes it. (CLAUDE.md commit discipline.)
- [ ] **Step 3: On approval only** — one commit on the feature branch, message ending with the
  `Co-Authored-By: Claude ...` trailer (NO `Claude-Session:` trailer). Then STOP again before push /
  PR / merge (each is a separate go-ahead).

---

## Self-review (author, against the spec)

- **Spec coverage:** §3 adapter → Task 2; §3.4 no-leak assembly → Tasks 3+5; §4 compute → Task 4; §5
  aggregate/add → Task 5; §6 bravery honest-NaN → Task 6; §7 required-column raise → Tasks 4/5/6; §8
  window semantics (atom-dense) → documented in Task 4 docstring; §9 config dedup + renumber + fixture
  regen + AtomicVAEP retrain + DSL → Task 1 + Task 9 ADR/CHANGELOG; §10 gates → Task 7; §11 tests →
  Tasks 2/4/5/6/8; §12 delivery (one commit, human gate) → Task 10. No gap.
- **Placeholder scan:** ADR number is `ADR-0NN` (assigned at write-time, Task 9 Step 3) and the version
  is "next free minor" — both are release-time facts, not code placeholders. All code steps carry real
  signatures.
- **Type consistency:** `_defensive_credit_atomic_adapter(actions, params)` (Task 2) is called with the
  resolved `params` in Task 4; `_rollup_defending_aggregate(actions, long, *, params, visible_area,
  links)` (Task 3) is called identically in Task 5; `_require_columns(actions, required, *, fn)`
  defined in Task 4 is reused in Task 6. Consistent.
