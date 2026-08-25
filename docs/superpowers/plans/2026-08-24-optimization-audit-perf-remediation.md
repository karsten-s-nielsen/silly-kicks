# Optimization-Audit Perf Remediation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the repo-wide *rescan-in-loop* O(n²) defect class and adjacent output-preserving perf debt found by the 2026-08-24 optimization audit, plus the turnover perf rewrite (item B).

**Architecture:** A new leaf seam `silly_kicks/_frame_index.py` (`group_rows`/`RowGroups`) provides an O(1) dtype-safe row-group lookup that replaces `df[df[k]==v]`-in-a-loop across nine sites. Remaining fixes hoist loop-invariant work, vectorize a possession label loop, single-pass a double XML parse, and harden four script drivers. Every library change is byte-identical output, gated by a parity test + a structural call-count guard.

**Tech Stack:** Python, pandas, numpy, scipy; `tests/_perf_structural.py` (`call_counter`, `row_iteration_counter`); `pytest`.

## Global Constraints

- **Output-preserving.** Every library change reproduces current output byte-for-byte. No VAEP/model-retrain trigger. Item B lands only behind an exact-equivalence oracle.
- **No new runtime dependency** (Batches 2–6). Item B route = pure-numpy; `numba` is a documented fallback only.
- **Hexagonal purity.** Core functions stay pure (pandas in/out, zero I/O). Policy at the edge. `_frame_index.py` is a leaf module with no intra-package deps (position of `id_compat`/`reflection`/`_polygon`).
- **ADR-019 dtype-safe keys.** Any id used as a group/lookup key is canonicalized via `id_compat.canonical_id`.
- **TDD, no wall-clock asserts.** Parity test + structural guard per fix. Fixtures MUST have ≥2 groups (a single-group fixture cannot distinguish O(n) from O(n²)). Every parity test asserts the failing side too (both-sides rule).
- **CI-faithful verification** before "done": `pytest -m "not e2e"` (no `--benchmark-skip`), `ruff check/format silly_kicks/ tests/ scripts/`, bare `pyright`.
- **One feature branch, ONE commit — the owner's standing policy.** No step commits; batches/phases are execution + review ordering ONLY, never commit boundaries. The entire cycle lands as a single commit the owner authors on explicit approval.
- **`silly_kicks/_frame_index.py` stays PRIVATE (F7)** — precedent is the private leaf utils `_geometry.py`/`_polygon.py`, NOT `id_compat` (whose underscore was removed because it is a *mandatory* seam). Record it in the **in-repo (first-party) consumers** table of `docs/PRIVATE_CONSUMERS.md` with a YAGNI exit condition ("promote if a cross-package/cross-repo consumer appears"). Public promotion is spec Open Decision §7.1a if the owner prefers it.
- **"Red-first" applies to NEW-code tests only (F3).** Task 1 (new module) is genuinely red-first. For CONSUMER tasks, the **parity test is an INVARIANCE guard** (green before and after; it goes red only if the refactor breaks output — do NOT instruct "confirm it fails first"), and the **structural guard is the anti-regression guard**, demonstrated red-capable by a mutation (move the hoisted call back into the loop), not by a pre-fix AttributeError.
- **Byte-identity is conditional on a clean key column (F1b).** `group_rows` canonicalizes keys (ADR-019); it is byte-identical only where the key column is single-dtype-clean. A consumer whose old raw `==` mis-resolved dtypes is a *behavior change*, not parity — that is a separate owner-approved task.

---

## Phase 0 — Shared seam

### Task 1: `group_rows` / `RowGroups`

**Files:**
- Create: `silly_kicks/_frame_index.py`
- Test: `tests/test_frame_index.py`
- Modify: `docs/PRIVATE_CONSUMERS.md` (add to the **in-repo (first-party) consumers** table, per F7)

**Interfaces:**
- Produces:
  - `group_rows(df: pd.DataFrame, by: str | tuple[str, ...]) -> RowGroups`
  - `RowGroups.get(*key) -> pd.DataFrame` — rows for `key` (O(1)); empty frame (df columns/dtypes) on miss.
  - `RowGroups.__contains__(key) -> bool`
- Consumes: `silly_kicks.id_compat.canonical_id`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_frame_index.py
import numpy as np
import pandas as pd
import pytest
from silly_kicks._frame_index import group_rows


def _frames():
    return pd.DataFrame({
        "game_id": pd.array([1, 1, 1, 2, 2], dtype="Int64"),
        "frame_id": pd.array([10, 10, 11, 10, 10], dtype="Int64"),
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
    })


def test_single_key_lookup_matches_boolean_filter():
    df = _frames()
    g = group_rows(df, "frame_id")
    # dtype-agnostic key: Python int / str both resolve the Int64 group
    for key in (10, "10", np.int64(10)):
        got = g.get(key).sort_index()
        exp = df[df["frame_id"] == 10]
        pd.testing.assert_frame_equal(got, exp)


def test_multi_key_lookup_matches_boolean_filter():
    df = _frames()
    g = group_rows(df, ("game_id", "frame_id"))
    got = g.get(2, 10)
    exp = df[(df["game_id"] == 2) & (df["frame_id"] == 10)]
    pd.testing.assert_frame_equal(got, exp)


def test_missing_key_returns_empty_frame_not_keyerror():
    df = _frames()
    g = group_rows(df, "frame_id")
    out = g.get(999)  # absent
    assert out.empty
    assert list(out.columns) == list(df.columns)
    assert out.dtypes.equals(df.dtypes)


def test_within_group_order_preserved():
    df = _frames()
    g = group_rows(df, "frame_id")
    # game 1 frame 10 has x=[0.0,1.0] in original order
    assert g.get(10)["x"].tolist() == [0.0, 1.0, 3.0, 4.0]  # both games' frame 10, source order


def test_mixed_dtype_key_raises_not_silent_row_loss():
    # F1(a): int 366 and str "366" are distinct groups that canonicalize equal -> refuse loud.
    df = pd.DataFrame({"k": pd.array([366, "366"], dtype="object"), "x": [0.0, 1.0]})
    with pytest.raises(ValueError, match="collapsed under|mixes dtypes"):
        group_rows(df, "k")


def test_contains_single_and_multi_key():
    df = _frames()
    assert 10 in group_rows(df, "frame_id")
    assert 999 not in group_rows(df, "frame_id")
    g = group_rows(df, ("game_id", "frame_id"))
    assert (2, 10) in g          # multi-key membership: pass a tuple
    assert (2, 11) not in g
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_frame_index.py -q`
Expected: FAIL — `ModuleNotFoundError: silly_kicks._frame_index`.

- [ ] **Step 3: Implement the seam** (per spec §3.2 — `groupby().indices` positional lookup, canonicalized keys, empty-on-miss)

```python
# silly_kicks/_frame_index.py
from __future__ import annotations
import pandas as pd
from silly_kicks.id_compat import canonical_id


class RowGroups:
    """O(1) dtype-safe row-group lookup — replaces `df[df[k]==v]`-in-a-loop (ADR-068).

    Backed by ``groupby().indices`` (positional int arrays, no frame copies). Keys are
    canonicalized (ADR-019). A missing key returns an EMPTY frame with df's columns/dtypes
    (never KeyError), matching the boolean-filter semantics it replaces.
    """

    def __init__(self, df: pd.DataFrame, by: str | tuple[str, ...]) -> None:
        self._df = df
        self._by = (by,) if isinstance(by, str) else tuple(by)
        gb = df.groupby(list(self._by), sort=False)
        self._indices = {self._canon(k): v for k, v in gb.indices.items()}
        # F1(a) collision guard: canonicalization collapses 366/366.0/"366" -> "366"; a mixed-dtype
        # key column would silently overwrite a group and lose its rows. Refuse loud instead.
        if len(self._indices) != len(gb.indices):
            raise ValueError(
                f"group_rows: {len(gb.indices) - len(self._indices)} group key(s) collapsed under "
                f"ADR-019 canonicalization on {self._by} -- the key column mixes dtypes. Clean it first."
            )

    def _canon(self, key):
        if len(self._by) == 1:
            return canonical_id(key)
        return tuple(canonical_id(k) for k in key)  # multi-key: `key` is a tuple

    def get(self, *key) -> pd.DataFrame:
        k = key[0] if (len(key) == 1 and len(self._by) == 1) else tuple(key)
        pos = self._indices.get(self._canon(k))
        return self._df.take(pos) if pos is not None else self._df.iloc[:0]

    def __contains__(self, key) -> bool:
        # single-key: pass the scalar; multi-key: pass a tuple, e.g. `(2, 10) in groups`.
        return self._canon(key) in self._indices


def group_rows(df: pd.DataFrame, by: str | tuple[str, ...]) -> RowGroups:
    return RowGroups(df, by)
```

- [ ] **Step 4: Run to verify pass + no regressions**

Run: `python -m pytest tests/test_frame_index.py -q` → PASS.
Then `python -m ruff check silly_kicks/_frame_index.py tests/test_frame_index.py && python -m ruff format --check silly_kicks/_frame_index.py tests/test_frame_index.py && python -m pyright silly_kicks/_frame_index.py`.

**pandas-major note (verified: single-key `groupby([col]).indices` → scalar keys, multi-key → tuples, on 2.3.3).** The `_canon`/`get` single-vs-multi split depends on that key shape, which is pandas-version-sensitive. Do NOT pin the shape with a literal assertion (CLAUDE.md: "assert the behaviour, not the shape across pandas majors"). The **behavioral tests `test_single_key_lookup_matches_boolean_filter` / `test_multi_key_lookup_matches_boolean_filter` run on ALL matrix legs (not `@slow`/`@e2e`) and thus pin the shape across the ADR-057 span (2.3.3 and 3.0.5)** — if a pandas major changes `.indices` key shape, they go red on that leg. That is the cross-version guard.

- [ ] **Step 5: Document the seam**

Add to the **in-repo (first-party) consumers** table of `docs/PRIVATE_CONSUMERS.md` (R4b — that file's cross-repo table is for lakehouse blast radius; this seam is all in-repo): `silly_kicks/_frame_index.py` — private row-group lookup (ADR-068); consumed by `causal/`, `tracking/`, `spadl/`; exit condition "promote to `silly_kicks.__all__` only if a cross-package/cross-repo consumer appears (YAGNI)."

---

## Phase 2 — Batch 2: rescan-in-loop (byte-identical)

> Each task: (1) write the **parity test** (an INVARIANCE guard — capture current output on a ≥2-group fixture; green before AND after; it goes red only if the refactor breaks output — do NOT expect it to fail first, F3); (2) apply `group_rows` (Task 1); (3) confirm parity green; (4) add the **structural guard** — spy `group_rows` **as resolved in the consumer's module**, assert it is constructed once per public call, and demonstrate red-capability with a mutation test (move the `group_rows(...)` call back inside the loop → count rises). The structural guard is a post-fix invariant + mutation proof, NOT red-first (pre-fix the symbol isn't imported → bare AttributeError = red-for-the-wrong-reason). Byte-identity holds only if the key column is single-dtype-clean (F1b); assert the key dtype on the fixture. Fixtures MUST have ≥2 groups or the guard is a false green.

### Task 2: `causal/opportunities.py` — per-frame spell scan

**Files:** Modify `silly_kicks/causal/opportunities.py:259-284`; Test `tests/causal/test_opportunities_perf.py` (+ parity in existing `tests/causal/test_opportunities.py`).
**Interfaces:** Consumes `group_rows` (Task 1).

- [ ] **Step 1: Parity test** — build a 2-period, multi-frame `frames`+`actions` fixture (reuse `tests/causal/_fixtures`); assert `build_opportunities(...)` output equals a captured reference (compute once with current code, hard-code the expected `shape`/columns/spell rows).
- [ ] **Step 2: Structural guard (mutation-proof, NOT red-first — `group_rows` is a new symbol, §3.3/R2)**

```python
# tests/causal/test_opportunities_perf.py
from tests._perf_structural import call_counter
import silly_kicks.causal.opportunities as opp

def test_frame_lookup_built_once(monkeypatch):
    calls = call_counter(monkeypatch, opp, "group_rows")  # after Step 4 the module imports it
    # ... build a multi-frame fixture and call build_opportunities(...)
    assert calls["n"] >= 1  # tighten to == number of (game_id, period_id) groups; NOT per-frame
```
Note (F3): this is the anti-regression guard; pre-import it errors (AttributeError), which is NOT a meaningful red. Demonstrate red-capability post-fix via the mutation test (move `group_rows` inside the `frame_keys` loop → count rises to per-frame).
- [ ] **Step 3: Implement** — `from silly_kicks._frame_index import group_rows`; inside the `for (gid, per), g in poss.groupby(...)` loop, after `g = g.sort_values(...)`, build `frame_groups = group_rows(g, "frame_id")` once; replace line 264 `grp = g[g["frame_id"] == fid]` → `grp = frame_groups.get(fid)`. Iteration order unchanged (still `frame_keys`).
- [ ] **Step 4: Run** parity + structural → PASS; tighten the structural assertion to `== <number of (game_id, period_id) groups in the fixture>` (F11 — the build is once per `poss.groupby(["game_id","period_id"])` iteration, so on a multi-game fixture it is NOT `n_periods`).

### Task 3: `defensive_credit/_resolution.py` + `_orchestration.py` — per-action×per-rule frame rescan

**Files:** Modify `silly_kicks/tracking/defensive_credit/_resolution.py:51`, `.../_orchestration.py:~90-139`, `.../_rules.py` (RuleContext); Test `tests/tracking/test_defensive_credit_perf.py` + parity in existing defensive-credit tests.
**Interfaces:** Consumes `group_rows`. `RuleContext` gains `frame_groups: RowGroups | None`. `resolve_responsible_defenders(..., frame_groups: RowGroups | None = None)` — when provided, use `frame_groups.get(frame_id)` instead of the boolean filter; when `None`, keep the current filter (unit-test path).
**F4 — key on `frame_id` ALONE.** The old filter is `frames[frames["frame_id"] == frame_id]`, so `group_rows(frames, "frame_id")` reproduces it EXACTLY (byte-identical). Do NOT add `period_id` to the key here — that is a separate behavior change (differs iff `frame_id` is not unique across periods) and must not be laundered through this parity task.

- [ ] **Step 1: Parity test** — assert `compute_defensive_credits(...)` long-form output byte-identical on the existing fixtures. (No output change is expected — `frame_id`-alone keying matches the old filter exactly.)
- [ ] **Step 2: Structural guard** — `call_counter(monkeypatch, _orchestration, "group_rows")`; assert `n == 1` per `compute_defensive_credits` call (was: per action×rule). Post-fix invariant + mutation proof, not red-first (F3).
- [ ] **Step 3: Implement** — build `frame_groups = group_rows(frames, "frame_id")` once in `compute_defensive_credits`; store on `RuleContext`; thread into `resolve_responsible_defenders`; `fr = frame_groups.get(frame_id) if frame_groups is not None else frames[frames["frame_id"] == frame_id]`.
- [ ] **Step 4: Run** parity + structural + full defensive-credit suite → PASS.
- [ ] **Step 5 (SEPARATE, do not fold in): raise the cross-period `frame_id` question** — determine whether `frame_id` is per-period or globally unique in these frames (spec Open Decision §7.5). If per-period, the `frame_id`-alone filter is a latent cross-period bug; fixing it (`(period_id, frame_id)` key) is its own owner-approved, separately-tested behavior change with a colliding-`frame_id` fixture — NOT part of this refactor.

### Task 4: `_gk_identification.py::derive_goalkeepers` — per-(game,team) + per-GK full scan

**Files:** Modify `silly_kicks/tracking/_gk_identification.py:105-169`; Test `tests/tracking/test_gk_identification_perf.py` + parity on a 2-game×2-team fixture.
**Interfaces:** Consumes `group_rows`.

- [ ] **Step 1: Parity test on a CLEAN-ID fixture** — 2-game × 2-team fixture; **first assert `game_id`/`team_id`/`player_id` are single-dtype** (F1b — byte-identity holds only if clean; `group_rows`' collision guard RAISES on a mixed-dtype column, so a dirty column fails loud at build, not silently). Assert identical `frames_out["is_goalkeeper"]` and `derived_picks`.
- [ ] **Step 2: Structural guard** — spy `group_rows`; assert built once, not per team/GK. Post-fix invariant + mutation proof (F3).
- [ ] **Step 3: Implement** — `team_groups = group_rows(player_rows, ("game_id","team_id"))`; `team_rows = team_groups.get(game_id, team_id)`. For the write, build `gk_pos = group_rows(frames_out, ("game_id","team_id","player_id"))` once and set `is_goalkeeper=True` via the union of `gk_pos.get(game_id, team_id, pid).index` for each picked pid (replaces the per-GK 3-condition boolean mask). Keys are canonicalized → drops the raw `==` id compares (ADR-019). NOTE: on a clean-id fixture canonical == raw so this is byte-identical; if any id column is mixed-dtype in production, routing through canonicalization is a deliberate ADR-019 correctness change (own owner-approved task, dirty-column fixture), not this parity task.
- [ ] **Step 4: Run** parity + structural → PASS.

### Task 5: `causal/_confounders.py::_pressure_at_entry` — full players rescan per spell

**Files:** Modify `silly_kicks/causal/_confounders.py:61-97`; Test `tests/causal/test_confounders_perf.py` + parity.
**Interfaces:** Consumes `group_rows`; uses `id_compat.ids_match`.

- [ ] **Step 1: Parity test** — multi-spell fixture; assert identical `bekkers_pi` array from `_pressure_at_entry`.
- [ ] **Step 2: Structural guard (mutation-proof, NOT red-first — `group_rows` is a new symbol, §3.3/R2)** — spy `group_rows` in the consumer module; assert built once (mirror sibling `_defending_team_id`); demonstrate red-capability by moving the call back inside the spell loop → count rises.
- [ ] **Step 3: Implement** — `frame_groups = group_rows(players, ("game_id","period_id","frame_id"))` before the spell loop; `grp = frame_groups.get(gid, per, fid)`; replace `[same_id(t, team) for t in grp["team_id"]]` with `grp[ids_match(grp["team_id"], team)]` (ADR-019 vectorized).
- [ ] **Step 4: Run** parity + structural → PASS.

### Task 6: `_off_ball_runs.py:127` — per-game frame refilter

**Files:** Modify `silly_kicks/tracking/_off_ball_runs.py:~120-130`; Test `tests/tracking/test_off_ball_runs_perf.py` + parity on a 2-game fixture.

- [ ] **Step 1: Parity test** — 2-game fixture; assert identical off-ball-run output.
- [ ] **Step 2: Structural guard (mutation-proof, NOT red-first — new symbol, §3.3/R2)** — spy `group_rows`; assert built once; red-capability via the mutation test (move the call inside the per-game loop → count rises).
- [ ] **Step 3: Implement** — `frame_groups = group_rows(frames, "game_id")` before the `for game_id, game_actions in actions.groupby("game_id")` loop; `game_frames = frame_groups.get(game_id)`.
- [ ] **Step 4: Run** → PASS.

### Task 7: `_run_values.py:248` — per-game frame refilter

**Files:** Modify `silly_kicks/tracking/_run_values.py:~245-250`; Test `tests/tracking/test_run_values_perf.py` + parity on a 2-game fixture.

- [ ] **Step 1: Parity test** — 2-game fixture; identical run-value output.
- [ ] **Step 2: Structural guard (mutation-proof, NOT red-first — new symbol, §3.3/R2)** — spy `group_rows`; assert `n == 1`; red-capability via the mutation test (move the call inside the per-game loop → count rises).
- [ ] **Step 3: Implement** — `frame_groups = group_rows(frames, "game_id")` before the loop; `game_frames = frame_groups.get(game_id)` (replaces the `ids_equal(...)` boolean filter).
- [ ] **Step 4: Run** → PASS.

### Task 8: `spadl/_skillcorner_inference.py::infer_defensive_actions` — O(n×m) windowed nearest-after

**Files:** Modify `silly_kicks/spadl/_skillcorner_inference.py:59-94`; Test `tests/spadl/test_skillcorner_inference.py` (extend) — parity on a committed SkillCorner fixture.
**Interfaces:** Uses `pd.merge_asof` (NOT `group_rows` — this is a windowed nearest-after join, not a keyed lookup).

- [ ] **Step 1: Parity test (PRIMARY guard)** — committed SkillCorner fixture (or synthetic `player_possession`+`obe_regains` with ≥2 defensive-start rows and multiple candidate regains); capture current `actions` output; assert equal. This changes `actions` output shape if wrong → hard blocker. **This is the primary guard for this task** (the structural one below is weak — see F2).
- [ ] **Step 2: Structural guard (F2 — `row_iteration_counter` is VACUOUS: the loop is `for idx in pp.index[...]` with `.loc`/mask, not `.iterrows()`)** — instead assert `pd.merge_asof` is called (once) and the per-row `obe_regains` boolean-scan is gone. This is only weakly red-capable ("new primitive runs"), so lean on the golden parity test as the real guard.
- [ ] **Step 3: Implement** — replace the `for idx in pp.index[defensive_mask]` loop's per-row `obe_regains[mask].sort_values().iloc[0]` with a single `pd.merge_asof(defensive_starts.sort_values(t), obe_regains.sort_values(t), by=("period","team_id"), direction=..., tolerance=...)` reproducing the current sort+first-match direction/window. Read the site first to fix exact column names/direction.
- [ ] **Step 4: Run** parity (golden, primary) + structural → PASS.

---

## Phase 3 — Batch 3: possession O(k²) labels

### Task 9: `vaep/labels.py` — vectorize `_scores_possession` / `_concedes_possession`

**Files:** Modify `silly_kicks/vaep/labels.py:288-375`; Test `tests/vaep/test_labels_windowing*.py` (extend for parity both `xg_column` modes) + `tests/vaep/test_labels_possession_perf.py`.

- [ ] **Step 1: Parity test (both modes) — fuller oracle (F6).** On the existing windowing fixtures + a new fixture, capture current output for `xg_column=None` AND `xg_column="xg"`; assert equal. The fixture MUST include: (i) a possession containing **BOTH teams' actions** (carve-out/native possession can hold opposing actions — a single-team possession is a false green for the team-aware split); (ii) the **goal/owngoal-action-scores-itself** second pass (`labels.py:322-328,367-373`); (iii) a possession with **multiple downstream same-team goals of DECREASING xG** (proves reverse-cumulative MAX, not first, on the xG path).
- [ ] **Step 2: Structural guard (F2 — `row_iteration_counter` is VACUOUS here, plain `.loc` loop not iterrows; genuinely red-first)** — spy via the house helper: `call_counter(monkeypatch, pandas.core.indexing._LocIndexer, "__getitem__")` (its module arg accepts a class). **Assert SCALE-INDEPENDENCE, not `== 0` (R3):** score a k=4 possession and a k=12 possession, assert the `.loc` count does NOT scale with k (bounded/constant) — the robust O(k²)→O(k) invariant. Pre-fix the count is O(k²) and scales → FAILS red. (A `== 0` threshold is brittle to incidental `.loc` in the vectorized path and doesn't prove the win.)
- [ ] **Step 3: Implement (team-aware, F6)** — per `(game_id, possession_id)` group: numpy arrays for `goal`/`owngoal`/`team_id`/`xg`. The scoring condition is TEAM-OF-POS-relative, so precompute **two** reverse-cumulative aggregates over the group (one per team present) of the eligible downstream goal (`_scores`: same-team goal / other-team owngoal; `_concedes`: mirror), then index each position by ITS OWN team. Bool path = reverse-cumulative-OR; xG path = reverse-cumulative-MAX. Then apply the self-scoring second pass. Mirror `_scores_time`/`_concedes_time`.
- [ ] **Step 4: Run** parity (both modes, all three fixture cases) + structural + full vaep labels suite → PASS.

---

## Phase 4 — Batch 4: loop-invariant / uncached recompute (byte-identical)

### Task 10: `pitch_control/_surface.py` — cache the interpolator on the frozen surface

**Files:** Modify `silly_kicks/tracking/pitch_control/_surface.py:79-115`; Test `tests/tracking/pitch_control/test_surface_interp_cache.py` + parity via existing pitch-control tests.

- [ ] **Step 1: Parity test** — assert `at_point`/`at_points` values unchanged for a known surface.
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive over-called pre-fix, §3.3/R2)** — `call_counter(monkeypatch, <surface module>, "RegularGridInterpolator")`; drive `_obso.py`-style double query of one `event_surface`; assert `n == 1` (pre-fix: one build per `.at_*` call → the assertion FAILS red pre-fix).
- [ ] **Step 3: Implement** — lazily build + cache the interpolator on first `.at_*` use via `object.__setattr__(self, "_interp", ...)` (frozen dataclass). Do NOT key a module-level cache on `id(self)` (unsafe across GC).
- [ ] **Step 4: Run** → PASS.

### Task 11: `pitch_control/_spearman.py` / `_fernandez_bornn.py` / `_voronoi.py` — memoize grid/meshgrid

**Files:** Modify the three modules (grid construction sites); Test `tests/tracking/pitch_control/test_grid_memoization.py` + parity.

- [ ] **Step 1: Parity test** — identical grids/targets for a given `(grid_cells_x, grid_cells_y)`.
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive, §3.3/R2)** — `call_counter` on `np.meshgrid`; assert bounded (1 per unique grid config across a multi-surface pass); pre-fix it rebuilds per surface → FAILS red.
- [ ] **Step 3: Implement** — extract a `@functools.lru_cache`-wrapped `_grid(grid_cells_x, grid_cells_y)` helper returning read-only arrays (`.setflags(write=False)`); verify no caller mutates the arrays (copy at the mutation site if any).
- [ ] **Step 4: Run** → PASS.

### Task 12: `causal/matching.py::_cluster_reassign` — hoist invariant grouping out of the seed loop

**Files:** Modify `silly_kicks/causal/matching.py:227-266` + `placebo_shift:298-316`; Test `tests/causal/test_matching_placebo_perf.py` + parity (fixed RNG).

- [ ] **Step 1: Parity test** — fixed RNG seed; assert identical placebo distribution before/after.
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive, §3.3/R2)** — `call_counter` on `np.unique`; assert `n == 1` (pre-fix: `n_seeds`=200 → FAILS red).
- [ ] **Step 3: Implement** — factorize/argsort `cluster_ids` once outside the `for s in range(n_seeds)` loop; per seed apply only the permutation to the precomputed grouping.
- [ ] **Step 4: Run** → PASS.

### Task 13: `spadl/utils.py::add_gk_role` (+ atomic dup) — hoist invariant `ids_isin`

**Files:** Modify `silly_kicks/spadl/utils.py:239-270` AND `silly_kicks/atomic/spadl/utils.py:199-230`; Test `tests/spadl/test_add_gk_role_perf.py` + parity (K∈{1,3}).

- [ ] **Step 1: Parity test** — assert identical `gk_role` for `distribution_lookback_actions ∈ {1, 3}` in both SPADL and atomic.
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive, §3.3/R2)** — `call_counter` on `ids_isin`; call with K=3; assert `n == 1` (pre-fix: 3 → FAILS red).
- [ ] **Step 3: Implement** — hoist `cur_is_known_gk = ids_isin(cur_player_arr, goalkeeper_ids)` and the four k-invariant array builds above the `for k in range(1, ...)` loop, in BOTH files.
- [ ] **Step 4: Run** parity + structural (both modules) → PASS.

### Task 14: `_cover_shadows.py` — reuse batched variant-0 for the baseline `n_blocked` loop

**Files:** Modify `silly_kicks/tracking/_cover_shadows.py:1113-1131`; Test `tests/tracking/test_cover_shadows_baseline_parity.py` (bit-identical) + structural.
**Precondition:** land ONLY if variant-0 reproduces the loop's `n_blocked` **bit-identically** on committed cover-shadow fixtures (spec §5 documents the bit-identical constraint). Else defer this task.

- [ ] **Step 1: Parity test** — assert `n_blocked_receivers` bit-identical (the whole point of this task).
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive, §3.3/R2)** — `call_counter` on `lane_control`; assert it is no longer called per-receiver in the baseline path (pre-fix: once per dangerous receiver → FAILS red).
- [ ] **Step 3: Implement** — reuse `_lane_received_batched`'s variant-0 output (already computed at ~line 1257) for the baseline instead of the per-receiver `lane_control()` call.
- [ ] **Step 4: Run** — if bit-identical PASS; else revert and mark deferred.

---

## Phase 5 — Batch 5: parse ports (golden-parity-gated)

### Task 15: `providers/sportec/parse.py::_parse_positions_xml` — single-pass + merge

**Files:** Modify `silly_kicks/providers/sportec/parse.py:481-540`; Test: existing `tests/providers/sportec/test_parse_port_parity.py` (golden) + `tests/providers/sportec/test_single_pass_parse.py` (structural).

- [ ] **Step 1: Parity (golden)** — run the existing golden-parity test against `idsse_slice/` (`SOURCE_SHA`); this is the hard gate — any diff blocks.
- [ ] **Step 2: Structural guard (genuinely red-first — pre-existing primitive, §3.3/R2)** — `call_counter(monkeypatch, <parse module>, "iterparse")` (patch `xml.etree.ElementTree.iterparse` as resolved in the module); assert `n == 1` (pre-fix: 2 → FAILS red).
- [ ] **Step 3: Implement** — collect player rows in ONE `iterparse` pass with `ball_*` blank, accumulate `ball_by_frame` as the stream progresses (ordering guarantee per the docstring), then one vectorized `merge` on `(period, frame)` reproducing the current per-row `dict.get()` join.
- [ ] **Step 4: Run** golden parity + structural → PASS. (Output-identical → retrain-neutral.)

---

## Phase 6 — Batch 6: script driver resilience / RAM

> Not library code (DGX / HF Jobs). "Parity" = identical produced artifact/cache; the change is resilience + memory.

### Task 16: `scripts/_xtgk_comparability.py` — `--cache-dir` + single fetch + resume

**Files:** Modify `scripts/_xtgk_comparability.py:79-133`; Test `tests/scripts/test_xtgk_comparability_cache.py`.

- [ ] **Step 1: Test** — assert a `--cache-dir` arg exists and the tracking artifact is fetched once (spy the loader), not twice.
- [ ] **Step 2: Implement** — add `--cache-dir` pass-through; reuse the fetched artifact across the grid-fit and `_collect` passes; adopt `scripts/_driver.py::for_each` for resume where it fits.
- [ ] **Step 3: Run** → PASS.

### Task 17: `scripts/calibrate_tracking_defaults.py::_load_fold` — safe default cap

**Files:** Modify `scripts/calibrate_tracking_defaults.py:110-145`; Test `tests/scripts/test_calibrate_load_fold_cap.py`.

- [ ] **Step 1: Test** — assert the default no longer loads unbounded (a safe cap applies unless an explicit opt-out flag is passed).
- [ ] **Step 2: Implement** — change `tracking_limit`/`max_per_provider` defaults from `None` to a safe cap; add an explicit `--no-cap` opt-out; document the Hyrum change in CLI help + CHANGELOG.
- [ ] **Step 3: Run** → PASS.

### Task 18: `scripts/_loader_databricks.py` — batch the N+1 SQL

**Files:** Modify `scripts/_loader_databricks.py:150-157`; Test `tests/scripts/test_loader_databricks_batch.py` (mock the connection).

- [ ] **Step 1: Test** — assert one `WHERE match_id IN (...)` query per table (2 total), not 2N.
- [ ] **Step 2: Implement** — batch by IN-list + client-side `groupby`; parity: identical per-match frames.
- [ ] **Step 3: Run** → PASS.

### Task 19: `scripts/_loader_pining_to_cache.py` — skip guard (do LAST; rebase on PR-S163)

**Files:** Modify `scripts/_loader_pining_to_cache.py:44-79`; Test `tests/scripts/test_loader_pining_cache_skip.py`.
**Precondition:** PR-S163 merged; verify not deliberately ADR-052-exempt.

- [ ] **Step 1: Test** — assert an existing per-match cache dir is skipped (no re-fetch); spy the loader.
- [ ] **Step 2: Implement** — walk match IDs, skip if the cache dir exists, thread `--cache-dir`; adopt `for_each` if the cache-write shape fits.
- [ ] **Step 3: Run** → PASS.

---

## Phase B — turnover perf rewrite

### Task 20: `xtgk/_turnover.py::_opp_first_shot_after_turnover` — pure-numpy possession pre-aggregation

**Files:** Modify `silly_kicks/xtgk/_turnover.py:163-191`; Test `tests/xtgk/test_turnover_perf.py` (exact-equivalence oracle) + extend `tests/xtgk/test_turnover_faithful.py`.
**Precondition:** item A (row-order sort) already in place — it is the sort precondition the possession walk relies on.

- [ ] **Step 1: Exact-equivalence oracle (red)**

```python
# tests/xtgk/test_turnover_perf.py — rename the current loop to `_opp_first_shot_after_turnover_ref`
# (kept in-test), assert the new impl is byte-identical on a churny multi-game corpus.
def test_new_impl_equals_reference_loop():
    a = _churny_corpus()  # multi-game, turnovers, consecutive opp possessions, shots, window None & finite
    tv = EmpiricalTurnoverValue(window_seconds=None)
    got = tv._opp_first_shot_after_turnover(a, "xg", window_seconds=None)
    exp = _reference_loop(a, "xg", window_seconds=None)  # the current implementation, verbatim
    np.testing.assert_array_equal(got, exp)
    # and again for window_seconds=10.0
```
The fixture MUST include (i) a possession chain spanning TWO consecutive opponent possessions before a shot (the semantics the naive rewrite gets wrong), and (ii) for the finite-window run, a shot placed JUST BEYOND `window_seconds=10.0` plus a within-window shot for contrast (F5 — the finite-window assertion is vacuous without a beyond-window shot to exclude). **This oracle is the SOLE guard for this task (F2)** — there is no pandas primitive to spy on a numpy double loop, so do NOT add a vacuous `row_iteration_counter` structural guard here.

- [ ] **Step 2: Benchmark the route** — on a realistic synthetic corpus, confirm pure-numpy beats the loop. If it does not, or the vectorization proves semantically fragile, switch to the documented fallback (`numba @njit` the existing loop verbatim, behind the `[numba]` extra) — decide here, record the measurement.
- [ ] **Step 3: Implement (pure-numpy)** — per spec §Batch B: groupby `(game_id, possession_id)` → possessing team, start time, first-shot xG, **AND the first shot's `time_seconds` (F5 — carry it, or a finite window cannot be honored)**; order possessions per game; per turnover walk the ordered opponent-possession run honoring break/continue (consecutive-opp-possession semantics), game bound, and — for finite `window_seconds` — credit only if `first_shot_time - turnover_time <= window_seconds`.
- [ ] **Step 4: Run** oracle (both windows, incl. the beyond-window shot) + full `tests/xtgk/` → PASS. (No structural guard — the oracle is the guard, per Step 1.)

---

## Phase 7 — Decision record

### Task 21: ADR-068

**Files:** Create `docs/superpowers/adrs/ADR-068-rescan-in-loop-and-frame-index-seam.md`; Modify `CLAUDE.md` (Key conventions — one line: rescan-in-loop → `group_rows`), `CHANGELOG.md`.

- [ ] **Step 1** — Write ADR-068: the rescan-in-loop defect class, the `group_rows` seam + why one seam over nine inline `groupby`s, the ADR-019 key-canonicalization, the turnover item-B route decision (pure-numpy, numba fallback), and the structural-guard convention that pins it. Assign the next free ADR number at commit-prep (verify against merged `origin/main`).
- [ ] **Step 2** — CHANGELOG entry (assign version at commit-prep from merged `origin/main`); note: output-preserving, no retrain trigger; `calibrate_tracking_defaults` default-cap is the one Hyrum-note behavior change (operator-facing).

---

## Self-review notes

- **Spec coverage:** every spec §4 site maps to a task (Batch 2 → Tasks 2–8, Batch 3 → 9, Batch 4 → 10–14, Batch 5 → 15, Batch 6 → 16–19, item B → 20, ADR → 21). Low-tier deferred items (spec §2) are intentionally not tasked.
- **Type consistency:** `group_rows`/`RowGroups.get` names are used identically in Tasks 2–7. `RuleContext.frame_groups` (Task 3) and `resolve_responsible_defenders(frame_groups=...)` match.
- **No placeholders:** each task names exact files/lines, the exact transform, and real parity + structural test scaffolding. Sites read only via the audit (Tasks 6–8, 10–19) carry a "read the site first for exact local names" instruction where surrounding variable names were not verified line-by-line — the transform and tests are concrete.
- **Ordering:** Task 1 (seam) precedes all consumers. Task 19 (`_loader_pining_to_cache`) is gated on PR-S163. Task 14 is conditional on bit-identical parity. Task 20 Step 2 gates the route on a measurement.
