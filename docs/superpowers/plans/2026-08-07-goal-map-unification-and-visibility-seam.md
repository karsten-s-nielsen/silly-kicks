# Goal-map unification and the visible-area seam — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Revision:** 4 (reviews 1-3; response logs at the end). Rev 1 wrote call sites against the old seam contract; rev 2 kept a gate the fix makes vacuous; rev 3 left the test HARNESS on the old contract.

**Goal:** Replace **10** hand-rolled goal-end derivations with one `GoalMap` seam that owns construction *and* lookup, re-key **2** direction bools onto it, pin the population so an eleventh fails CI, and ship the `visible_area` consuming seam.

**Architecture:** `resolve_defended_goals(frames)` returns a frozen `GoalMap` built **once per match** from the full frames, partitioning `(game, period, team)` into `resolved` / `guessed` / `unresolved` over canonical keys. It is **threaded** into per-frame functions, where it *replaces* `home_team_id`. Consumers call `get` (own end) or `attacked_goal` (opponent's end) — never a plain dict.

**Tech Stack:** Python 3.10+, pandas, numpy, pytest. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-08-07-goal-map-unification-and-visibility-seam-design.md` (rev 8).

## Global Constraints

- **ONE commit, squash-merged.** Task 13 is the only commit, and only after explicit owner approval.
- **Nothing skipped or deferred** without explicit owner approval.
- Branch from `main` at `12f77f9`. Branch: `goal-map-unification`.
- Lint at CI scope: `python -m ruff check silly_kicks/ tests/ scripts/` and `ruff format --check` on the same three. Never `ruff check .`.
- `python -m pyright` bare. Full suite on `.venv312`: `python -m pytest tests/ -m "not e2e" -v --tb=short`.
- **`GoalMap` keys are canonical STRINGS.** `canonical_id(1) == '1'`, `canonical_id(pd.NA) is pd.NA` (never `None`). **Never hand out `dict(gm.resolved)` or `{**guessed, **resolved}`** — a raw-tuple lookup against string keys misses silently. Always `gm.get(...)` / `gm.attacked_goal(...)`.
- **The map is built ONCE per match from full frames.** Per-frame construction measured **78.8% wrong** (spec §15). Two layers, two rules — they are NOT the same rule:
  - **Per-frame functions** (`lane_control`, `compute_blocking_score`, `compute_gk_influence`, `_voronoi_threat`, `_compute_cover_shadow_dict`): `goal_map: GoalMap` **required, no default.** A default there re-admits the 78.8% path at exactly the call sites that forget to pass it.
  - **`add_*` aggregators**: `goal_map: GoalMap | None = None` — build from their own full `frames` when `None`. That is correct by construction (they HAVE the full frames) and it is the established `links` / `pitch_control_cache` pattern. **It is also what makes Gate C possible**: a gate that varies the map needs a seam to inject one through, and an aggregator that always builds its own is untestable by Gate C.
- **One policy for an unresolvable end, stated by LAYER** (spec §2.9): the `add_*` surface emits **NaN + provenance** and never raises — that is where ADR-003 NaN-tolerance applies. **Never coerce and never fail-open, at any layer.**
- **SEVEN breaking public changes:** two renames; `home_team_id` → `goal_map` on `lane_control` and `compute_blocking_score`; `home_team_id` **removed** from `add_gk_influence`, `add_cover_shadows` and **`gk_influence_xfns`**. (The seventh applies the same rule as 5 and 6: `_get_gi` lives inside that factory, so after the re-key its `home_team_id` is required-and-unread — the dead-parameter shape this cycle deletes. Hyrum-visible to xfns callers.)
- **Policy lives at the EDGE, via a NAMED exception.** Per-frame functions (`lane_control`, `compute_blocking_score`, `compute_gk_influence`, `_voronoi_threat`) require a resolvable map and raise **`GoalEndUnresolvedError`** (a `ValueError` subclass, defined in `_gk_resolve.py`). The `add_*` aggregator catches it **by name** and emits a NaN row + provenance. This is what makes §2.9 expressible — `LaneControlResult` carries three `bool` fields and there is no NaN bool — and it keeps the decision in ONE place: having the aggregator pre-check resolvability would duplicate the exact lookup the per-frame function is about to do, and the two copies could drift on which accessor and which `allow_guess`. A second implementation of the decision, inside the commit that deletes second implementations. Direct callers of the public per-frame functions still fail loud.
- **Gate C replaces Gate B** for the two re-keyed aggregators (Task 5b). Proven able to fail before being written (spec §16.3).
- Version **4.77.0** in five sites: `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock`, `CHANGELOG.md`, `TODO.md`.

---

## File Structure

| file | responsibility | task |
|---|---|---|
| `silly_kicks/tracking/_gk_resolve.py` | `GoalMap` + `resolve_defended_goals`; delete `defended_goal_x` | 2 |
| `tests/tracking/test_goal_map_population.py` | NEW — semantic AST gate | 1 |
| `tests/tracking/test_goal_map_seam.py` | NEW — ladder, accessors, canonical keys | 2 |
| `tests/tracking/test_goal_map_oracle.py` | NEW — golden CAPTURED pre-change | 2 |
| `silly_kicks/tracking/_ghost_gk.py` | map construction, fallback, source token | 3, 7 |
| `silly_kicks/tracking/_xcross_attempt.py` | delete `_build_goal_map`; convert 2 lookups | 3 |
| `silly_kicks/tracking/_model_eval.py`, `silly_kicks/causal/opportunities.py` | wrapper callers; convert lookups | 3 |
| `silly_kicks/tracking/_gk_influence.py` | `goal_map` replaces `home_team_id`; 2 sites | 4 |
| `silly_kicks/tracking/features.py` | 3 sites + build/thread the map + re-export | 4 |
| `silly_kicks/tracking/_cover_shadows.py` | `goal_map` replaces `home_team_id`; 5 sites | 5 |
| `silly_kicks/gkdv/_engine.py`, `_shot_goalmouth.py`, `_xshot_occurrence.py` | seam consumers | 6 |
| `tests/tracking/_mirror_entries/influence_family.py` | 2 × `defect_b` removed; stale prose | 4, 5 |
| `tests/tracking/test_goal_map_consumers.py` | NEW — **consumer** characterization | 8 |
| `silly_kicks/tracking/_snapshot.py` | dtype pin | 9 |
| `silly_kicks/tracking/_visibility.py` | NEW — primitives + `add_visible_area_coverage` | 10 |
| `silly_kicks/providers/statsbomb/parse.py`, `scripts/build_sb360_coverage.py` | `observed_pitch_fraction` + ADR-042 denominator | 11 |

---

## Task 1: The semantic AST gate — landed RED

**Files:** Create `tests/tracking/test_goal_map_population.py`

- [ ] **Step 1: Write the gate**

Use the predicate from spec §4.2 (`IfExp` ∪ `If`/`else`-assigning-one-name ∪ `np.where`; **no dict clause**). Three corrections over the first draft:

```python
SEAM = "silly_kicks/tracking/_gk_resolve.py"

# SB_FIELD_LENGTH is 120.0 (StatsBomb native), NOT a pitch end. Without this exclusion a
# `0.0 / SB_FIELD_LENGTH` binding is a false positive, and the "fix" would be an exemption
# for something that is not the defect.
_NOT_PITCH = {"SB_FIELD_LENGTH", "SB_FIELD_WIDTH"}


def _is_pitch(n: ast.AST) -> bool:
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and float(n.value) == 105.0:
        return True
    name = getattr(n, "id", None) or getattr(n, "attr", None)
    if not isinstance(name, str) or name in _NOT_PITCH:
        return False
    return bool(PITCHY.search(name))


def test_goal_end_derivation_lives_in_exactly_one_place():
    found = _scan()
    assert found, "scanner found nothing -- it is broken, not the tree clean"
    assert SEAM in found, "the seam itself derives no goal end -- the gate would pass vacuously"
    assert len(found[SEAM]) == 1, f"the seam must derive it ONCE, found {found[SEAM]}"
    extra = {k: v for k, v in found.items() if k not in ({SEAM} | set(_EXEMPT))}
    assert not extra, "goal-end derivation outside the seam: " + "; ".join(
        f"{k}:{v}" for k, v in sorted(extra.items())
    )
```

Plus the four non-vacuity tests from rev 1 (if/else plant, ternary, `np.where`, minutes-dict negative) and `test_every_exempt_entry_actually_matches`.

- [ ] **Step 2: Run it and OBSERVE IT RED**

Run: `python -m pytest tests/tracking/test_goal_map_population.py -v`

Expected: `test_goal_end_derivation_lives_in_exactly_one_place` **FAILS** listing **10** sites outside the seam across 5 modules — `_ghost_gk.py:850,876`; `_xcross_attempt.py:291`; `features.py:3108,3337,3525`; `_gk_influence.py:318`; `_cover_shadows.py:611,910,1073`. (Census is 12 = 10 + seam + exempt.) The other five tests PASS.

**Do not proceed until you have seen this failure.**

---

## Task 2: The `GoalMap` seam

**Files:** Modify `silly_kicks/tracking/_gk_resolve.py:323-353`, `silly_kicks/tracking/__init__.py:166,418`. Create `tests/tracking/test_goal_map_seam.py`, `tests/tracking/test_goal_map_oracle.py`.

**Interfaces — Produces:**
- `GoalMap(resolved, guessed, unresolved)`, frozen; `MappingProxyType` mappings; keys are canonical string tuples
- `.get(game_id, period_id, team_id, *, allow_guess=False) -> float | None`
- `.attacked_goal(game_id, period_id, team_id, *, allow_guess=False) -> float | None`
- `.ends_in_period(game_id, period_id, *, allow_guess=False) -> dict`
- `.n_resolved`, `.n_guessed`
- `resolve_defended_goals(frames) -> GoalMap`

- [ ] **Step 1: CAPTURE the golden from the pre-change tree**

Rev 1 transcribed values from the spec, against a *different* fixture, in a file importing a symbol that does not exist yet — it could not run pre-change, and would have passed while proving nothing. Capture instead:

```bash
python -c "
from silly_kicks.tracking import defended_goal_x
from tests.tracking.test_goal_map_oracle import well_formed_frames   # fixture only
print(repr(defended_goal_x(well_formed_frames())))
"
```

Create `tests/tracking/test_goal_map_oracle.py` containing **only** `well_formed_frames()` first (2 periods, ends swapped, every team with a keeper, seed `20260807`), run the command above **from the repo root**, and paste its output directly into `GOLDEN`. `repr()` prints a literal with tuple keys that pastes without hand-editing; a `json.dumps` of `str(list(k))` does not, and hand-transformation at the one step whose point is mechanical capture is how a golden stops being a capture. Record the command and `12f77f9` in the file header. Only then add the assertions.

- [ ] **Step 2: Write the seam tests (RED)**

Take rev 1's suite with three corrections:

```python
def test_ladder_na_team_is_unresolved_and_in_NEITHER_mapping():
    f = well_formed_frames()
    f.loc[f["is_goalkeeper"] & (f["team_id"] == AWAY), "team_id"] = pd.NA
    gm = resolve_defended_goals(f)
    # canonical_id(pd.NA) is pd.NA -- NEVER None. Rev 1 tested `is None` and asserted nothing.
    na_keys = [k for k in gm.unresolved if k[2] is pd.NA]
    assert na_keys
    for k in na_keys:
        assert k not in gm.resolved and k not in gm.guessed


def test_keys_are_canonical_strings_and_lookups_accept_any_id_dtype():
    gm = resolve_defended_goals(well_formed_frames())
    assert all(isinstance(k[2], str) for k in gm.resolved)
    assert gm.get(1, 1, HOME) == gm.get("1", "1", "1") == gm.get(1.0, 1.0, 1.0)


def test_a_RAW_TUPLE_lookup_against_the_mapping_MISSES():
    """Witness for the rule that consumers must never hold the mapping as a plain dict."""
    gm = resolve_defended_goals(well_formed_frames())
    assert dict(gm.resolved).get((1, 1, HOME)) is None      # int tuple: MISSES
    assert gm.get(1, 1, HOME) == 0.0                        # accessor: hits
```

Plus rev 1's tests for `get`, `attacked_goal`, the degeneracy guard, the ladder, all-NaN-x through **both** mappings, string `is_ball`, non-`"ball"` ball team, and frozen-ness.

- [ ] **Step 3: Run to verify RED** — `python -m pytest tests/tracking/test_goal_map_seam.py -v` → import error.

- [ ] **Step 4: Implement the seam**

As spec §2.1, with the pools and the period index **precomputed once** (rev 1 rebuilt a merged dict on every lookup, and `attacked_goal` / `ends_in_period` then linearly scanned it — O(groups) per lookup inside per-frame loops):

```python
@dataclass(frozen=True)
class GoalMap:
    resolved: Mapping[tuple, float]
    guessed: Mapping[tuple, float]
    unresolved: frozenset

    def __post_init__(self) -> None:
        strict = dict(self.resolved)
        loose = {**dict(self.guessed), **strict}
        by_period: dict[tuple, dict] = {}
        for pool_name, pool in (("strict", strict), ("loose", loose)):
            idx: dict[tuple, dict] = {}
            for (g, p, t), v in pool.items():
                idx.setdefault((g, p), {})[t] = v
            by_period[pool_name] = idx
        # frozen=True blocks normal assignment; these are derived caches, not state.
        object.__setattr__(self, "_strict", strict)
        object.__setattr__(self, "_loose", loose)
        object.__setattr__(self, "_by_period", by_period)

    def _pool(self, allow_guess: bool) -> dict:
        return self._loose if allow_guess else self._strict

    def _period(self, g, p, allow_guess: bool) -> dict:
        return self._by_period["loose" if allow_guess else "strict"].get((g, p), {})
```

`get` looks up `self._pool(...)`; `ends_in_period` returns `dict(self._period(...))`; `attacked_goal` reads the period index, requires exactly one opponent, **and refuses when that opponent's end equals this team's own end** (spec §2.1 — in the degenerate case there IS exactly one opponent, so a count-only guard passes and the answer would say a team attacks the goal it defends).

`resolve_defended_goals` is spec §2.1's vectorized builder: `_truthy_bool` for both `is_*` columns, ball excluded by `is_ball`, `dropna=False`, and the ladder applied with `np.isfinite` on **both** the GK mean and the outfield mean.

- [ ] **Step 5: Update exports** — `tracking/__init__.py:166,418` and `features.py:74,138`: `defended_goal_x` → `GoalMap`, `resolve_defended_goals`.

- [ ] **Step 6: Run** — `python -m pytest tests/tracking/test_goal_map_seam.py tests/tracking/test_goal_map_oracle.py -v`

---

## Task 3: The map-constructing forks and their lookups

**Files:** `_ghost_gk.py:841-846, 850, 876`; `_xcross_attempt.py:282-292, 331, 356, 722, 742`; `_model_eval.py:129,134,150`; `causal/opportunities.py:39,251` + `_frame_domain_state`

- [ ] **Step 1: `_ghost_gk` — build from `work`**

Replace `:846-850` with `_goal_map = resolve_defended_goals(work)`. `work` is the post-subsample / post-link-filter set; passing `frames` would change the map whenever `subsample_fps` is set. Delete the comment at `:841-846` claiming teams swap ends on LTR-normalized data — `play_left_to_right` normalizes so the home team attacks left-to-right in *every* period; replace with a note that the end comes from the frames.

- [ ] **Step 2: `_ghost_gk` — delete the identity-keyed fallback**

```python
            goal_x = _goal_map.get(gid, pid, gk_team, allow_guess=True)
            if goal_x is None:
                continue  # unresolvable end -> no ghost; provenance token in Task 7
```

- [ ] **Step 3: `_xcross_attempt` — delete the fork AND convert its lookups**

Delete `_build_goal_map` (`:282-292`). At `:331` and `:722`: `goal_map = resolve_defended_goals(frames)`.

**Do NOT build a merged plain dict.** Rev 1 did, and the surviving raw-tuple lookups at `:356` and `:742` would then miss every time against canonical string keys. Convert both:

```python
-        goal_x = goal_map.get((gid, pid, defending[0]))
+        goal_x = goal_map.get(gid, pid, defending[0], allow_guess=True)
```

`allow_guess=True` reproduces the fork's N1 coverage, so values are unchanged.

- [ ] **Step 4: The two wrapper callers**

`_model_eval.py:129,134` and `causal/opportunities.py:39,251` import `_build_goal_map`. Both switch to `resolve_defended_goals` and convert their lookups the same way — `_model_eval.py:150`, and `opportunities`' `_frame_domain_state(grp, goal_map, gid, per, …)`, whose internal lookup must become `goal_map.get(gid, per, team, allow_guess=True)`.

- [ ] **Step 5: Run**

Run: `python -m pytest tests/tracking/ tests/causal/ -m "not e2e" -q`

Expected: PASS. If xCross or xS tests fail with empty/None results, the cause is almost certainly a **missed raw-tuple lookup** (canonical string keys vs int tuples) — grep for `goal_map.get((` before treating it as a behaviour change.

---

## Task 4: The gk_influence family — `goal_map` replaces `home_team_id`

**Files:** `_gk_influence.py` (signature, `:318`, `:371`, `:401`); `features.py:3108,3337,3525` + the three call sites; `tests/tracking/_mirror_entries/influence_family.py`

- [ ] **Step 1: Change the signature**

```python
 def compute_gk_influence(
     frame: pd.DataFrame,
     attacking_team_id: int | str,
     gk_player_id: int | str,
     xt: ExpectedThreat,
     *,
-    home_team_id: int | str,
+    goal_map: GoalMap,
     ...
```

Required, no default: a default re-admits per-frame construction, measured 78.8% wrong (spec §15). Verified by AST that `home_team_id`'s only reads here are `:318`, `:371` and the `:401` pass-through — nothing else uses it.

- [ ] **Step 2: The own-end site (`:316-321`)**

```python
    defending_team_id = gk_row["team_id"]
    _gid, _pid = frame["game_id"].iloc[0], frame["period_id"].iloc[0]
    goal_x = goal_map.get(_gid, _pid, defending_team_id, allow_guess=True)
    if goal_x is None:
        # PRECONDITION: the caller (add_gk_influence) resolves the end and emits the NaN row
        # itself, so reaching here is a programming error -- same shape as :202's
        # "gk_player_id not found in frame".
        raise ValueError(
            f"compute_gk_influence: goal_map does not resolve team {defending_team_id!r} in "
            f"(game={_gid}, period={_pid}). Callers must check before calling."
        )
```

- [ ] **Step 3: The direction site (`:371-372`)**

```python
    # Direction from the FRAMES. NOTE: acting_team_attacks_rtl does NOT fit -- it is
    # (actions, frames) -> Series, per-ACTION, and this is a per-frame call.
    _attacked = goal_map.attacked_goal(_gid, _pid, attacking_team_id, allow_guess=True)
    if _attacked is None:
        raise ValueError(  # explicit: `if _attacked == 0.0` alone would fail OPEN
            f"compute_gk_influence: goal_map does not resolve the goal attacked by "
            f"{attacking_team_id!r} in (game={_gid}, period={_pid})."
        )
    if _attacked == 0.0:
        threat_grid = threat_grid[::-1, ::-1]
```

The `is None` branch is not defensive padding: `attacked_goal` returned `None` for 34.2% of team-frames in the spec's §15 measurement, and both original guards treated that as "attacking rightward".

- [ ] **Step 4: The three `features.py` sites — build the map ONCE**

In each of `_gk_influence_at_actions`, `_closing_time_per_series` and `gk_influence_xfns`, hoist a single build **above** the per-frame loop:

```python
    goal_map = resolve_defended_goals(frames)   # ONCE per match, from the FULL frames
```

then at `:3108` / `:3337` / `:3525`:

```python
-            goal_x = 0.0 if same_id(gk_team, home_team_id) else 105.0
+            goal_x = goal_map.get(
+                frame_data["game_id"].iloc[0], frame_data["period_id"].iloc[0], gk_team, allow_guess=True
+            )
+            if goal_x is None:
+                continue
```

and pass `goal_map=goal_map` at every `compute_gk_influence(...)` call site.

- [ ] **Step 5: Remove `defect_b`, fix the stale prose**

Delete `defect_b=` from the `add_gk_influence` entry (`influence_family.py:82`) and replace the `:43-50` comment, which claims "GATE B FAILS AND IS NOT XFAILED" — untrue then (the entry carried a strict xfail) and doubly untrue now.

- [ ] **Step 6: Run** — `python -m pytest tests/tracking/test_mirror_registry.py -q -k "gk_influence"`

Expected after Task 5b: Gate A **PASS**, Gate B **SKIP** (`home_team_id` is gone, so `role="unused"`), Gate C **PASS**. Not XFAIL, not XPASS. Until Task 5b lands, Gate C does not exist yet and Gate B will XPASS — which is why 5b is not optional.

---

## Task 5: The cover-shadow family — 5 sites

Measured (spec §14.3): the three goal-end bindings alone move **zero** Gate B columns; with the two direction bools all five go to exactly `0.0`.

**Files:** `_cover_shadows.py` — signatures of `lane_control`, `compute_blocking_score`, `_voronoi_threat`, `_compute_cover_shadow_dict`; sites `:611`, `:704`, `:910`, `:1030`, `:1073`; `influence_family.py`

- [ ] **Step 1: Change the four signatures**

`home_team_id` → `goal_map: GoalMap` on all four, and update every internal pass-through (`:894`, `:940`, `:1062`, `:1101`, `:1119`). `lane_control` and `compute_blocking_score` are **public** — this is breaking change 3 and 4.

- [ ] **Step 2: Edit `_voronoi_threat` FIRST**

Order matters and it bit the executability pass: a caller edited before its callee keeps calling the old one, which presents as an independent defect (spec §14.4). `_voronoi_threat` is called by the others, so it goes first.

At `:704`:

```python
    _attacked = goal_map.attacked_goal(
        frame["game_id"].iloc[0], frame["period_id"].iloc[0], attacking_team_id, allow_guess=True
    )
    if _attacked is None:
        raise ValueError(  # explicit; `== 105.0` alone would fail OPEN
            f"_voronoi_threat: goal_map does not resolve the goal attacked by {attacking_team_id!r}."
        )
    attacking_toward_high_x = _attacked == 105.0
```

- [ ] **Step 3: The three opponent-end bindings (`:611`, `:910`, `:1073`)**

```python
    # The DEFENDERS' own goal = the end the attacking team ATTACKS. attacked_goal is a real
    # lookup of the opponent's entry; `105.0 - get(...)` would be a second implementation and
    # is wrong on a degenerate map.
    goal_x_own = goal_map.attacked_goal(
        frame["game_id"].iloc[0], frame["period_id"].iloc[0], attacking_team_id, allow_guess=True
    )
    if goal_x_own is None:
        raise ValueError(  # precondition; the aggregator emits the NaN row (Task 5 Step 5)
            f"cover shadows: goal_map does not resolve the goal attacked by {attacking_team_id!r}."
        )
```

(frame variable is `frame` at `:611`/`:910`, `frame_data` at `:1073`.)

- [ ] **Step 4: `_compute_cover_shadow_dict`'s two sites, in file order**

`:1030` (direction) precedes `:1073` (own end). Resolve `_attacked` once at `:1030` and reuse it at `:1073` — they are the same quantity.

- [ ] **Step 5: Build the map once at the `features.py` callers**

`features.py:3709` and `:3834` call `_compute_cover_shadow_dict` inside per-action loops. Hoist `goal_map = resolve_defended_goals(frames)` above each loop and pass it down.

- [ ] **Step 6: Remove `defect_b`** from `add_cover_shadows` and update the `:147` comment.

- [ ] **Step 7: Run**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -q`
Expected: `6 xfailed`, `0 failed`, `0 xpassed`.

---

## Task 5b: Gate C — the census the re-key would otherwise delete

Landed in this cycle, not deferred. Without it the three direction-bool sites
(`_gk_influence.py:371`, `_cover_shadows.py:704`, `:1030`) have **no detector**: the AST gate cannot
see a bool binding, and Gate B's variable no longer carries direction (spec §16).

**Files:** Modify `tests/tracking/_mirror_registry.py` (one field), `tests/tracking/test_mirror_registry.py` (the gate), `tests/tracking/_mirror_entries/influence_family.py` (two entries)

- [ ] **Step 1: Add the entry field**

The registry has ONE call field (`_mirror_registry.py:45`, `call: Callable  # (actions, frames, home_team_id) -> pd.DataFrame`), so Gate C needs its own:

```python
    call_with_map: Callable | None = None   # (actions, frames, goal_map) -> pd.DataFrame
```

**A non-`None` `call_with_map` IS the swappable predicate** — do not add a separate
`gate_c_swappable` flag, which is a second field that can disagree with the first.

- [ ] **Step 1b: Update the Gate A lambdas — REQUIRED, or Gate A raises**

Gate A calls `entry.call(...)` **unconditionally** (`test_mirror_registry.py:125-126`); the
`home_team_id_role` only selects *which* id is passed, never whether to pass one. With the parameter
removed (breaking 5 and 6) the existing lambdas raise `TypeError`. Both entries must ignore the
third argument:

```python
-   call=lambda a, f, h: add_gk_influence(a, f, gate_xt(), home_team_id=h),
+   call=lambda a, f, _h: add_gk_influence(a, f, gate_xt()),
+   call_with_map=lambda a, f, gm: add_gk_influence(a, f, gate_xt(), goal_map=gm),
```

The `goal_map=` keyword in the second lambda is the optional aggregator-level parameter from the
Global Constraints (`GoalMap | None = None`, built from the aggregator's own frames when omitted).
Gate C **requires** that seam — an aggregator that always builds its own map cannot be tested by a
gate that varies the map. Task 4 Step 4 and Task 5 Step 5 must add it to `add_gk_influence` and
`add_cover_shadows` respectively.

This is precisely the cost `space_creation` avoided by keeping the parameter unread. Paying it is
the decision (spec §16.2); skipping this step turns Gate A red.

- [ ] **Step 2: Write Gate C**

```python
@pytest.mark.parametrize("name", sorted(MIRROR_ENTRIES))
def test_gate_c_goal_map_is_the_direction_source(name):
    """D1, one variable further out than Gate B.

    Gate B varied ``home_team_id``. Once direction comes from the map that parameter carries
    nothing, so Gate B goes vacuous (or skips). This holds the FRAMES fixed and varies the
    MAP: the invariant columns must MOVE. If they do not, the aggregator is not reading the
    map and the re-key is cosmetic.
    """
    entry = MIRROR_ENTRIES[name]
    if entry.call_with_map is None:
        pytest.skip(f"{name} does not consume a goal map")

    actions, frames = canonical_scene()
    true_map = resolve_defended_goals(frames)
    flipped = GoalMap(
        MappingProxyType({k: (105.0 if v == 0.0 else 0.0) for k, v in true_map.resolved.items()}),
        MappingProxyType({k: (105.0 if v == 0.0 else 0.0) for k, v in true_map.guessed.items()}),
        true_map.unresolved,
    )

    ref = entry.call_with_map(actions.copy(), frames.copy(), true_map)
    alt = entry.call_with_map(actions.copy(), frames.copy(), flipped)

    moved = 0
    for col, cls in entry.columns.items():
        if cls != "invariant" or col in entry.gate_b_exempt:
            continue
        r = pd.to_numeric(ref[col], errors="coerce").to_numpy(dtype=float)
        v = pd.to_numeric(alt[col], errors="coerce").to_numpy(dtype=float)
        both = np.isfinite(r) & np.isfinite(v)
        if both.any() and float(np.abs(r[both] - v[both]).max()) > 1e-12:
            moved += 1
    assert moved > 0, (
        f"{name}: swapping the goal map moved NOTHING. Either the aggregator does not read "
        "the map, or this gate is vacuous -- both are failures."
    )
```

- [ ] **Step 3: Set `home_team_id_role="unused"` on both entries**

Honest: the parameter is gone. Gate B will now **SKIP** for them — so Task 4 Step 6 and Task 5 Step 7 expect SKIP for Gate B and PASS for Gate C, and the suite's skip count rises by 2.

- [ ] **Step 4: Run, and check the magnitudes against the recorded defect**

Run: `python -m pytest tests/tracking/test_mirror_registry.py -q -k "gate_c"`

Expected: PASS for `add_gk_influence` and `add_cover_shadows`.

**Expected column counts after the FULL re-key** — `add_cover_shadows` **5**; `add_gk_influence`
**2** (`gk_pitch_control_share_weighted` *and* `gk_closing_time_min_s__six_yard_box`), **not the 1**
the spec §16.3 probe saw. The probe did not inject the map into `_closing_time_per_series`; here it
is re-keyed, so the closing-time columns must move too. **A 1-column result means
`_closing_time_per_series` was missed** — do not read it as success.

Magnitudes should match what `influence_family.py` recorded for the D3 defect: `share ~0.1085`,
`closing_time_min ~4.38 s`, `blocking_score ~148.83`.

**What Gate C does NOT prove.** `moved > 0` shows the map is *consulted* — not that the *right
accessor* was chosen, since `get` and `attacked_goal` both move when the map is swapped. The
correctness half is Task 8 (consumer characterization). "Gate C replaces Gate B" is true of the
*detection* Gate B provided, not of correctness.


---

## Task 6: The remaining seam consumers

**Files:** `gkdv/_engine.py:188-190, 252, 257, 303`; `_shot_goalmouth.py:746,770,783-788`; `_xshot_occurrence.py:667,744,757,863,885`; `scripts/validate_shot_goalmouth_sb.py:541,663`; `tests/scripts/test_validate_shot_goalmouth_sb_shards.py:130`; `tests/tracking/test_gk_resolve_goal_map.py`; `tests/gkdv/test_import_allowlist.py:23`

- [ ] **Step 1: gkdv — return the `GoalMap`, make the `None → NaN` explicit**

`_pin_defended_goal` returns `resolve_defended_goals(frames)` directly; delete its local `canonical_id` re-keying (the seam canonicalizes now). Then `_goal_lookup` must change arity — rev 1 left it as `float(goal_map.get(key, np.nan))`, which is wrong against `GoalMap.get(game_id, period_id, team_id)` and would raise on `float(None)`:

```python
def _goal_lookup(goal_map, g, p, t) -> float:
    # _DROP_NO_GOAL_MAP keys on NaN, so the None -> NaN conversion is load-bearing.
    v = goal_map.get(g, p, t, allow_guess=True)
    return float("nan") if v is None else float(v)
```

Update both call sites (`:252`, `:257`). `_DROP_NO_GOAL_MAP` (`:303`) becomes reachable — update its `:317` note, which asserts it cannot fire.

- [ ] **Step 2: `_shot_goalmouth` — use `attacked_goal`, drop the raw tuple picking**

```python
        ends = goal_map.ends_in_period(row["game_id"], row["period_id"], allow_guess=True)
        attacked = goal_map.attacked_goal(
            row["game_id"], row["period_id"], row["team_id"], allow_guess=True
        )
        degenerate = len(ends) == 2 and len(set(ends.values())) == 1
        resolved = attacked is not None and not degenerate
```

This removes the cross-DataFrame raw tuple `==` at `:783` that the spec flags as a live ADR-019 hazard.

- [ ] **Step 3: `_xshot_occurrence`** — `:667` import `resolve_defended_goals`; `:744`/`:863` build the map; **convert the lookups at `:757` and `:885`** to `goal_map.get(gid, pid, def_team, allow_guess=True)`.

- [ ] **Step 4: The artifact driver and its monkeypatch**

`scripts/validate_shot_goalmouth_sb.py:541,663` — same substitution. `tests/scripts/test_validate_shot_goalmouth_sb_shards.py:130` patches with `lambda _f: {...}`; it must return a `GoalMap` with **canonical** keys:

```python
        _gk_resolve, "resolve_defended_goals",
        lambda _f: _gk_resolve.GoalMap(
            MappingProxyType({
                (canonical_id("g"), canonical_id(1), canonical_id(_HOME_TEAM)): 0.0,
                (canonical_id("g"), canonical_id(1), canonical_id(_AWAY_TEAM)): 105.0,
            }),
            MappingProxyType({}), frozenset(),
        ),
```

- [ ] **Step 5: `tests/tracking/test_gk_resolve_goal_map.py`** — delete the `_defended_goal_x is defended_goal_x` identity test (the shim is gone); update the behaviour tests.

- [ ] **Step 6: Prose** — `tests/gkdv/test_import_allowlist.py:23` is a comment, not a caller. Update the symbol name.

- [ ] **Step 7: Run** — `python -m pytest tests/ -m "not e2e" -q -x`, then confirm Task 1's gate is now GREEN.

---

## Task 7: Ghost provenance token

**Files:** `_ghost_gk.py:293-312`; `features.py` (`add_ghost_gk`)

- [ ] **Step 1:** Add `GHOST_GK_GOAL_END_UNRESOLVED = "goal_end_unresolved"` to the vocabulary and `GHOST_GK_SOURCE_VALUES`.
- [ ] **Step 2:** Rows skipped in Task 3 Step 2 get that token + NaN positions. `serve_ghost_gk_positions` returns **no row** (ADR-054 D2 — gkdv raises on a non-finite ghost).
- [ ] **Step 3:** Test the column and the asymmetry (both sides — token present *and* `len(positions) == 0`).
- [ ] **Step 4: Run** — `python -m pytest tests/tracking/ -q -k ghost`

---

## Task 8: CONSUMER characterization

Rev 1's version called `resolve_defended_goals` and asserted on the `GoalMap` — it exercised **no consumer**, while being the one task named for consumer behaviour. Every defect review 5 found was of exactly the kind it would have caught.

**Files:** Create `tests/tracking/test_goal_map_consumers.py`

- [ ] **Step 1: Build a fixture where the contracts DIFFER**

`canonical_scene()` has one period and one GK row per team per frame, so per-frame and per-period construction coincide — which is why pass 1 was blind. This fixture must not be:

```python
def sparse_keeper_frames(n_frames=60, gk_detect_rate=0.196, seed=4242):
    """2 periods (ends swapped), keeper detected in 19.6% of frames (the SkillCorner rate,
    ADR-038:123), outfielders pushing toward the goal they ATTACK.

    A fixture where the old and new contracts agree makes a green run evidence of nothing.
    """
```

- [ ] **Step 2: Assert the estimator property**

```python
def test_period_scoped_map_is_correct_where_a_per_frame_map_is_not():
    frames = sparse_keeper_frames()
    truth = {(1, 1, HOME): 0.0, (1, 1, AWAY): 105.0, (1, 2, HOME): 105.0, (1, 2, AWAY): 0.0}
    gm = resolve_defended_goals(frames)
    for key, want in truth.items():
        assert gm.get(*key, allow_guess=True) == want

    # non-vacuity: the per-frame map this cycle rejected really is different here
    wrong = 0
    for (gid, pid, fid), grp in frames.groupby(["game_id", "period_id", "frame_id"]):
        fm = resolve_defended_goals(grp)
        for team in (HOME, AWAY):
            if fm.get(gid, pid, team, allow_guess=True) != truth[(gid, pid, team)]:
                wrong += 1
    assert wrong > 0, "fixture cannot distinguish the two contracts -- it proves nothing"
```

- [ ] **Step 3: Exercise each CONSUMER on each degenerate shape**

One test per (consumer, shape). Consumers: `add_gk_influence`, `add_cover_shadows`, `add_ghost_gk`, `add_xcross_attempt`, `add_xshot_occurrence`, `add_shot_goalmouth`. Shapes: GK-less group, NA-team GK row, all-NaN-x GK group, string `is_ball`, nullable-boolean `pd.NA`.

Each asserts the **documented outcome** — not merely "differs from before":

```python
# NOTE the call shapes: `xt` is POSITIONAL on both aggregators, and `home_team_id` is GONE
# (breaking changes 5 and 6). Sketches that pass `xt=xt` or `home_team_id=` will not run.

_GKI_INVARIANT = [
    "gk_pitch_control_share_weighted",
    "gk_reachable_area_m2",
    "gk_closing_time_min_s__six_yard_box",
    "gk_closing_time_mean_s__six_yard_box",
]


def test_add_gk_influence_nans_ALL_invariant_columns_on_an_unresolvable_end():
    """All four, not just the scalar: GkInfluence.closing_times is a dict, and an EMPTY dict
    would make features.py:3158's `for zn, zct in gi.closing_times.items()` never assign the
    closing-time columns -- absent rather than NaN. Asserting only the scalar would pass while
    those took an unspecified path."""
    out = add_gk_influence(actions, all_nan_x_frames(), gate_xt())
    for col in _GKI_INVARIANT:
        assert out[col].isna().all(), col


_CS_INVARIANT = [
    "blocking_score",
    "n_potential_receivers",
    "max_single_defender_blocking_score",
    "n_blocked_receivers",
    "blocked_threat_fraction",
]


def test_add_cover_shadows_does_not_fail_open_when_direction_is_unresolved():
    """All five, mirroring the gk_influence case. NOTE `is_blocked_any` is a
    LaneControlResult FIELD, not an emitted column -- asserting on it is a KeyError."""
    out = add_cover_shadows(actions, na_team_frames(), gate_xt())
    for col in _CS_INVARIANT:
        assert out[col].isna().all(), col
```

- [ ] **Step 4: Run** — `python -m pytest tests/tracking/test_goal_map_consumers.py -v`

---

## Task 9: `_snapshot` dtype pin

- [ ] **Step 1:** Write the contract test — `frames[col].dtype == TRACKING_FRAMES_COLUMNS[col]` for the five id columns. Version-independent: there is no pandas axis in CI (`ci.yml` is OS × Python only).
- [ ] **Step 2:** Run it; record pass or fail on the local pandas.
- [ ] **Step 3:** After `_snapshot.py:173` (`frames = frames[list(...)]`), cast each column to its declared dtype.
- [ ] **Step 4: Run** — `python -m pytest tests/tracking/test_snapshot.py -v`

---

## Task 10: The visibility seam

**Files:** Create `silly_kicks/tracking/_visibility.py`, `tests/tracking/test_visibility.py`; modify `tracking/__init__.py`

- [ ] **Step 1:** Tests — `point_observed` returns `None` (not `False`) for a missing polygon; `region_observed_fraction` takes an `(M,2)` **polygon** (the half-pitch × triangle case is exactly `0.75` — verified: clipped trapezoid 2677.5 / triangle 3570); coverage is NaN never `1.0` when no polygon exists; the source vocabulary is closed; the aggregator is pure.
- [ ] **Step 2:** Sutherland–Hodgman convex clip (~30 lines, dependency-free) + shoelace.
- [ ] **Step 3:** `point_observed`, `region_observed_fraction`, `add_visible_area_coverage` emitting `visible_area_fraction` (clipped ∈ [0,1]) and `visible_area_source` over `{observed, no_polygon, degenerate_polygon, unlinked}`.
- [ ] **Step 4: Run** — `python -m pytest tests/tracking/test_visibility.py -v`

---

## Task 11: `observed_pitch_fraction` and the artifact denominator

- [ ] **Step 1:** Rename `visible_fraction` → `observed_pitch_fraction`, delete the old name, return the **clipped** share ∈ [0,1] and **NaN** for a degenerate polygon. The polygon stays unclipped — ADR-054 D5 is about the vertices.
- [ ] **Step 2:** Extend the crc witness. `test_crc_is_invisible_to_visible_fraction` uses a polygon at x 10–110, y 10–70 — **entirely interior**, so clipping is a no-op and it would keep passing while the property it witnesses became false. Add a touchline-crossing case and re-state D5's reason as *alignment*.
- [ ] **Step 3:** `build_sb360_coverage.py:255` — accumulate only finite fractions, carry `n_with_polygon` as the denominator, report it alongside the rate (ADR-042: a coverage denominator must never masquerade as a signal). With only 32.6% of goal kicks carrying a freeze-frame, the old `sum_area += NaN` would poison most buckets.
- [ ] **Step 4: Run** — `python -m pytest tests/providers/ tests/scripts/ -q`

---

## Task 12: Registrations, docs, version

- [ ] **Step 1:** Register `add_visible_area_coverage` in the glossary, liveness (fixture must **vary** coverage), `PURITY_ENTRIES` (**two** variants — it branches on polygon presence), the mirror registry (**Gate A only**: it takes no `home_team_id` and no `goal_map`, so `home_team_id_role="unused"` and `gate_c_swappable=False` — Gates B and C both SKIP, which must be stated so the registration is not recorded as an assertion it never makes), id-dtype invariance, SB360 verdict registry.
- [ ] **Step 2:** C4 aggregator count 32 → 33; regenerate via `mad-scientist-skills:c4`.
- [ ] **Step 3:** Prose sites: `CLAUDE.md`; `TODO.md:53-62`; `gkdv/_engine.py:317`; `ADR-030:28`; `ADR-043:326`; `CHANGELOG.md:1605`; `influence_family.py:43-50`.
- [ ] **Step 4:** ADR-055 — the census and how it was derived; the ladder; `attacked_goal`'s degeneracy guard; **why the map is period-scoped and threaded** (the 78.8% measurement); why `home_team_id` was replaced rather than retired by disuse; the two `defect_b` removals; and **Gate C — why Gate B was retired for two aggregators and what replaced it** (the more consequential decision, and the first thing a future reader will ask on seeing two entries skip Gate B), including §16.3's proof-before-spec and its honest qualification.
- [ ] **Step 5:** Version 4.77.0 in five sites; `uv lock`.
- [ ] **Step 6:** `ruff check` / `ruff format --check` / `pyright` / full suite — all clean on `.venv312`.

---

## Task 13: Final review and the single commit

- [ ] **Step 1:** Run `/final-review` — before proposing the commit.
- [ ] **Step 2:** `git grep -n "4\.77\.0"` → five sites.
- [ ] **Step 3:** Confirm §6's three measurements are recorded (M1 NA-team rows in the ghost training corpus; M2 GK-less groups per provider; **M3 the period-flip invariant, which can gate the cycle**).
- [ ] **Step 4: STOP. Request explicit owner approval.**
- [ ] **Step 5:** One commit: `fix(tracking): unify goal-end derivation behind one GoalMap seam -- silly-kicks 4.77.0 (PR-S145, ADR-055)`
- [ ] **Step 6:** Push, PR, watch CI on five legs, squash-merge.

---

## Post-merge follow-ups

- Re-run `scripts/build_sb360_coverage.py` (a driver cannot stamp a SHA that does not yet exist).
- §6 M1 → if non-zero, a `GhostGkModel` retrain as its own weights cycle.
- The six remaining D3 aggregators stay with ADR-051.

---

## Review 1 response log

| # | finding | resolution |
|---|---|---|
| P1 | canonical keys are STRINGS; six raw-tuple lookups miss silently | **Verified** (`canonical_id(1) == '1'`). The merged plain dict is deleted from the plan entirely; all six lookups convert to `gm.get(...)`. Global constraint added; Task 2 Step 2 carries a witness test that a raw-tuple lookup misses |
| P2 | per-frame map is a different estimator, failing OPEN | **Verified, and worse than stated: 78.8% wrong, `attacked_goal` None 34.2%** (spec §15). Map is now built once per match and threaded; `goal_map` **replaces** `home_team_id` on five functions (spec §2.8) |
| P3 | Task 8 exercised no consumer | Rewritten as `test_goal_map_consumers.py`: 6 consumers × 5 shapes, plus a fixture non-vacuity assertion that the old and new contracts actually differ on it |
| P4 | `is None` vs `is pd.NA` | **Verified** — `canonical_id(pd.NA) is pd.NA`. Corrected; one spelling now |
| P5 | four handlings, two breaking the plan's own NaN-tolerance constraint | One policy (spec §2.9): NaN + provenance, never raise, never fail-open. Applied at all four |
| P6 | `_pool()` rebuilt per lookup | Pools and a `(game, period)` index precomputed in `__post_init__` |
| P7 | "11 sites" stated, 10 listed | Corrected to 10; census 12 = 10 + seam + exempt |
| P8 | gate is file-granular; `_is_pitch` matches `SB_FIELD_LENGTH` (120.0) | Added `assert SEAM in found` and `len(found[SEAM]) == 1`; `_NOT_PITCH` excludes the SB constants |
| P9 | the oracle was transcribed, not captured | Task 2 Step 1 is now a capture command against the pre-change tree, recording the command and the SHA |
| P10–P11 | spec §2.7 count drift; a refuted paragraph left standing | Spec rev 7 |
| P12 | plan goal line miscounts | "Replace 10 … re-key 2 … pin so an eleventh fails" |
| P13 | reuse instruction order-inverted | Task 5 Step 4 follows file order: `:1030` then `:1073` |
| P14 | "any failure is a real behaviour change" would misdirect | Task 3 Step 5 now names the missed-lookup cause first |

**The lesson, which is about method and not this plan:** pass 1 executed the **diff**; pass 2 executed the **consumers**, and only the second found P1/P2. The generalization is stronger than "execute the diff" — **execute it on a fixture that can distinguish the new contract from the old**. `canonical_scene` could not, and looked thorough.

---

## Review 2 response log

| # | finding | resolution |
|---|---|---|
| Q1 | replacing `home_team_id` makes Gate B vacuous — the cycle deletes the only detector for the class it fixes | **The sharpest finding of the cycle.** Verified: Gate B skips on `role="unused"` (`test_mirror_registry.py:241-242`), and the AST gate cannot see a bool binding, so the three direction-bool sites would have had NO detector. **Task 5b lands Gate C** (vary the MAP, frames fixed), and it was **proven able to fail before being written** — spec §16.3: `share 0.108532`, `blocking_score 148.83`, the same magnitudes the registry recorded as the D3 defect. Owner decision: `home_team_id` is also **removed** from both `add_*` wrappers (breaking 5 and 6), so Gate B's old claim becomes unrepresentable rather than merely unasserted |
| Q2 | the three empty-result symbols do not exist, and `lane_control` cannot express one | **Verified**: `LaneControlResult` carries 6 floats + **3 bools**, and there is no NaN bool. Resolved by layer instead of by return type — **policy at the edge**: the `add_*` aggregator resolves the end and emits the NaN row; per-frame functions require a resolvable map and RAISE, matching the existing precedent at `_gk_influence.py:202`. Keeps §2.9 uniform where ADR-003 actually applies and avoids a seventh breaking change |
| Q3 | `GkInfluence(nan, nan, {})` drops columns rather than NaN-ing them, and Task 8 would not catch it | Task 8 now asserts **all four** invariant columns, with the reason inline: an empty `closing_times` dict makes `features.py:3158`'s loop never assign the closing-time columns |
| Q4 | the capture command is not pasteable | `repr()` instead of `json.dumps(str(list(k)))`; run from the repo root with `from tests.tracking…`, avoiding the `sys.path` shadow of `tracking` |
| Q5 | Task 8's sketches will not run — `xt` is positional, `home_team_id` required | Fixed, with the call shape called out; `home_team_id` is now gone entirely |
| Q6 | `add_visible_area_coverage` cannot make a Gate B assertion | Registered **Gate A only**, with `home_team_id_role="unused"` and `gate_c_swappable=False` stated so the skip is recorded, not implied |
| Q7 | the breaking count is contingent | Settled at **six**, enumerated in spec §9 |

**Corrected contradiction introduced by this revision:** the Global Constraint had read *"Never raise, never coerce, never fail-open"* two lines above a constraint saying per-frame functions raise. Restated by layer. That is the stranded-edit pattern reviews 3 and 4 both flagged; it appeared again here, in a document being edited to close a review.

---

## Review 3 response log

| # | finding | resolution |
|---|---|---|
| R1 | Gate C calls `entry.call_with_map`, which does not exist | Task 5b Step 1 adds `call_with_map: Callable \| None = None` to the registry entry. Taking the reviewer's better form: **a non-`None` `call_with_map` IS the swappable predicate** — no separate `gate_c_swappable` flag, which would be a second field that can disagree with the first |
| R2 | removing `home_team_id` breaks **Gate A**, which calls `entry.call` unconditionally | **Verified** (`test_mirror_registry.py:125-126` — the role selects which id is passed, never whether). New Task 5b Step 1b updates both lambdas to ignore the third argument. This is exactly the cost `space_creation` avoided by keeping the parameter unread; paying it is the decision |
| R3 | `gk_influence_xfns` keeps a required, now-dead `home_team_id` | **Verified.** Removed — a **seventh** breaking change. Applying the rule already used for 5 and 6 rather than asking a fourth time: a dead required parameter is the shape this cycle deletes. Flagged as Hyrum-visible to xfns callers |
| R4 | Task 8 asserts on `is_blocked_any`, which is a `LaneControlResult` FIELD, not an emitted column | **Verified** — absent from `features.py`; it would `KeyError`. Replaced with `_CS_INVARIANT`, all five emitted columns, mirroring the gk_influence fix |
| R5 | "policy at the edge" makes the aggregator re-derive the per-frame function's decision | **Adopted, and it is better than what I had.** A named `GoalEndUnresolvedError` raised by the per-frame functions and caught by name at the aggregator keeps the decision in ONE place. Pre-checking would have duplicated the exact lookup the callee is about to do — a second implementation of the decision, inside the commit that deletes second implementations |
| R6 | Task 4 Step 6 still expects PASS where Task 5b says SKIP | Fixed: Gate A PASS / Gate B SKIP / Gate C PASS, with a note that Gate B XPASSes until 5b lands — which is why 5b is not optional. **This is the stranded-edit pattern the plan's own closing note names, appearing in the revision that names it** |
| R7 | Gate C proves the map is consulted, not that the right accessor was chosen | Stated in Task 5b Step 4: `get` and `attacked_goal` both move when the map is swapped, so the correctness half is Task 8. "Gate C replaces Gate B" is true of the *detection*, not of correctness |
| R8 | Task 5b's cross-check should expect **2** columns for `add_gk_influence`, not the probe's 1 | Added, with the failure reading spelled out: a 1-column result means `_closing_time_per_series` was missed and must not be read as success |
| R9 | ADR-055 should record Gate C and why Gate B was retired | Added to Task 12 Step 4, including §16.3's proof-before-spec and its qualification |

**One gap this revision exposed in itself.** R1's fix has Gate C inject a map via
`add_gk_influence(..., goal_map=gm)` — but the plan had the aggregator *build* its map internally,
and a Global Constraint read *"No `goal_map=None` default anywhere"*. Gate C would have had nothing
to inject through. The constraint is now stated by layer: **required with no default on the
per-frame functions** (where a default re-admits the 78.8% path), **optional on the `add_*`
aggregators** (which have the full frames, and where the seam is what makes Gate C possible at all).
Found by re-reading the edit rather than by the reviewer — the same "one call site left on the old
contract" shape, at the layer the review had just moved to.

---

## IMPLEMENTATION STATE — handoff (2026-08-07)

Branch `goal-map-unification`, **nothing committed**, tree importable, CI-scope lint and format
clean, **no version number claimed** (all five sites untouched, pending the other session).

### Complete and verified

| task | evidence |
|---|---|
| **1** AST gate | Landed RED and **observed failing** on all 10 forks / 5 modules; now GREEN |
| **2** `GoalMap` seam | `GoalMap`, `resolve_defended_goals`, `GoalEndUnresolvedError`, `_end_from_mean_x`; golden **captured from the pre-change tree**; byte-identity PASSES |
| **3** forks + wrappers | `_ghost_gk` (built from `work`, identity fallback deleted), `_xcross_attempt` fork deleted, `_model_eval` + `causal/opportunities` converted; **all six raw-tuple lookups** now `gm.get(...)` |
| **4** gk_influence sites | 5 of 5 re-keyed (`_gk_influence.py:318`, `:371`; `features.py` ×3, each building the map ONCE above its loop) |
| **5** cover-shadow sites | 5 of 5 re-keyed; 5 signatures + 5 pass-throughs moved to `goal_map` |
| **6** library consumers | `_xshot_occurrence`, `_shot_goalmouth` (raw cross-DataFrame tuple `==` deleted), `gkdv/_engine` (`None -> NaN` explicit), `test_gk_resolve_goal_map.py` |
| — | the §17.1 cascade: `compute_threat_pc`, `gkdv.delta_threat_suppression`, `scripts/build_gkdv_arm_values.py` |

**Population gate is GREEN**: the goal-end rule exists in exactly ONE place in the package.

### Remaining

- **Task 4 finish** — `add_gk_influence` / `add_cover_shadows` / `gk_influence_xfns`: drop
  `home_team_id`, add optional `goal_map: GoalMap | None = None`; remove the two `defect_b`
  markers and fix the stale `influence_family.py:43-50` prose.
- **Task 5b** — Gate C: `call_with_map` on the registry entry, the gate itself, `role="unused"`,
  **and Step 1b's Gate A lambda updates** (Gate A calls `entry.call` unconditionally, so it raises
  `TypeError` the moment `home_team_id` is gone).
- **Task 6 finish** — `tests/scripts/test_validate_shot_goalmouth_sb_shards.py:130` monkeypatch
  (must return a `GoalMap` with **canonical** keys), `scripts/validate_shot_goalmouth_sb.py`,
  `tests/gkdv/test_import_allowlist.py:23` prose.
- **Tasks 7-12** — ghost token; consumer characterization; `_snapshot` dtype pin;
  `tracking/_visibility.py`; `observed_pitch_fraction` + the ADR-042 denominator; registrations,
  C4 32→33, ADR-055, prose sweep, version.

### KNOWN RED -- MEASURED, not estimated

Full sweep of `tests/tracking/ tests/gkdv/ tests/causal/` (`-m "not e2e"`, 7m36s):

```
153 failed, 3292 passed, 20 skipped, 8 xfailed, 3 errors
```

**Every failure sampled is ONE mechanical class**: a test-side caller passing `home_team_id` to
a signature that now takes `goal_map`. Dominant modes:

```
15  _compute_cover_shadow_dict()   got an unexpected keyword argument 'home_team_id'
 8  compute_blocking_score()       got an unexpected keyword argument 'home_team_id'
 7  lane_control()                 got an unexpected keyword argument 'home_team_id'
 7  delta_threat_suppression()     got an unexpected keyword argument 'home_team_id'
 2  _voronoi_threat()              got an unexpected keyword argument 'home_team_id'
 1  compute_threat_pc()            got an unexpected keyword argument 'home_team_id'
```

No library change is implied by any of them. Each call site builds
`goal_map = resolve_defended_goals(frames)` once and passes it. This is **Task 6 finish**, and it
is materially larger than the plan anticipated -- the plan named three test files; the real surface
is every test exercising the five changed public signatures.

**A correction worth keeping.** An earlier handoff draft recorded "23 failed" and asserted
`tests/tracking/` and `tests/causal/` "were last seen clean". Both came from a run that PREDATED
the cover-shadow cascade. The rule this repo already has -- *a claim about a gate's behaviour
carries a pasted measurement, not an adjacent one* -- applies to progress reports too.

### BLOCKING DECISION -- resolve this FIRST

`_gk_influence.py:415` calls `select_back_line_players(frame, team_id=..., home_team_id=...)`.
My pass-through edit rewrote that kwarg to `goal_map=`, which is **wrong** -- that function does
not take one. 29 of the 153 failures are this single miscast call.

It is not a typo to patch, because `compute_gk_influence` no longer HAS a `home_team_id` to pass:

```python
select_back_line_players(frames, team_id, home_team_id, *, n=4, adaptive_max_n=5)
```

It uses `home_team_id` to decide which goal is "own goal" -- i.e. it is **direction-keyed**, and it
lives in `_defensive_line.py`, one of the THREE files pinned by
`test_mirror_registry.py::test_defensive_line_d3_unit_is_enumerated`. The information is available
at the call site (`goal_x` is already resolved there), so the options are:

1. **Thread the resolved end**: give `select_back_line_players` a `goal_x: float` in place of
   `home_team_id`. Correct, consistent with the whole cycle -- but a **tenth** breaking change,
   and it changes the pinned D3 unit's membership, so that gate's assertion must move with it.
   `_packing.py:166` and the module's own docstring example are the other two callers.
2. **Keep `home_team_id` on `add_gk_influence`** purely to forward here -- rejected reasoning
   already (it is the dead-parameter shape), but it would bound the cascade.
3. Something narrower nobody has proposed yet.

**Do not guess.** The cascade has now expanded at every layer it touched (5 -> 7 -> 9 breaking
changes, two packages), and this one reaches into a gate's pinned registry. Get an owner decision
before writing code.

### For whoever resumes

1. **Do not text-sweep for `home_team_id=home_team_id,`** — §17.1: the punctuation-sensitive grep
   missed the call that caused the cascade. Use `ruff`/`pyright` after each signature change; the
   linter sees the call graph, the grep sees a string.
2. **Edit `features.py` and `_cover_shadows.py` by AST line range, never by global replace.** A
   blanket keyword replace rewrote four unrelated families here and had to be reverted.
3. **Callee before caller** within a module (`_voronoi_threat` before its callers).
4. Four edit-mechanics errors occurred in this stretch; **all four were caught by running, none by
   reading.** Run something after every edit.
