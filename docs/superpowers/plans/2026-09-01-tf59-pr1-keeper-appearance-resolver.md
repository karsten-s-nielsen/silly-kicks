# TF-59 PR1 — Keeper-Identity Resolver + Appearance-Interval Port — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote silly-kicks' keeper-identity resolver to shared public infrastructure, give it an event-only path and an injected keeper appearance-interval port with substitution/red-card-granular per-action resolution, and ship four provider extractors that populate the port from raw feeds.

**Architecture:** Move `tracking/_keeper_identity.py` → public `silly_kicks/keeper_identity.py` (breaking, no shim; native/frame path lazy-delegates to `tracking._gk_resolve`). Add a normalized `KeeperAppearances` table (period-relative intervals) and make `add_defending_gk_player_id` interval-aware (byte-identical when no intervals are supplied). Four `providers/<p>/appearances.py` extractors normalize each provider's native substitution encoding into the port.

**Tech Stack:** Python, pandas (nullable dtypes), `silly_kicks.id_compat` for all id comparisons. No new runtime dependency. StatsBomb/DFL/GS raw parsing already exists in `providers/` + `spadl/` + `scripts/_loader_pining.py`.

**Spec:** `docs/superpowers/specs/2026-09-01-tf59-gk-shot-stopping-and-keeper-appearance-resolver-design.md` (the plan argues from this spec; executors read both). This plan is **PR1** of the spec's two-PR arc. PR2 (the `shot_stopping` metric) is a separate plan authored after PR1 merges.

## Global Constraints

Every task's requirements implicitly include this section. Values are copied from the spec / CLAUDE.md verbatim.

- **ONE commit for all of PR1.** No per-task commits, no micro-commits, no "commit after each step." Tasks end at *green tests*; the single commit happens only at the final commit-prep gate (Task 10) **after explicit human approval of the diff**. Never `git commit`/`git push` without that approval.
- **One feature branch off `main`; no worktrees.** Branch first if on `main`.
- **Breaking import move, no shim, fail-loud** — a stale `from silly_kicks.tracking import resolve_keeper_identities` must raise `ImportError`, not silently degrade.
- **All id comparisons via `silly_kicks.id_compat`** (`ids_equal`/`ids_match`/`same_id`/`canonical_id`) — never raw `==`/`!=` on ids, never `astype(str)` on an id used as a key (ADR-019).
- **Nullable dtypes** for ids (`Int64`/`object`), never a non-NaN sentinel (ADR-027). An unresolvable keeper is `pd.NA`, never a fabricated id.
- **Period-relative time** (`time_seconds` resets per period; ADR-017). Appearance intervals are period-relative.
- **Testing at CI scope:** `python -m pytest tests/ -m "not e2e" -v --tb=short`; lint `python -m ruff check silly_kicks/ tests/ scripts/` + `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright` (bare). Real-data tests are `@pytest.mark.e2e` and skip without data/token.
- **Both-sides test discipline:** every band/invariance test asserts the failing side too; every counterfactual asserts it measurably differs from its twin.
- **No version / PR-S / ADR numbers in code or docs until commit-prep** (Task 10), re-derived after `git fetch && git merge origin/main`.
- **Additive to trained models — no retrain, no re-materialize.** The `native`/`roster` outputs stay byte-identical; only the new opt-in `appearances`/event-only paths add behavior.

## File Structure

**Created:**
- `silly_kicks/keeper_identity.py` — the promoted resolver (moved from `tracking/_keeper_identity.py`), + the `KeeperAppearances` schema (`KEEPER_APPEARANCE_COLUMNS`, `validate_keeper_appearances`) + interval-aware `add_defending_gk_player_id`.
- `silly_kicks/providers/statsbomb/appearances.py` — `extract_keeper_appearances` (StatsBomb raw events + lineups → port).
- `silly_kicks/providers/sportec/appearances.py` — `extract_keeper_appearances` (DFL raw → port).
- `silly_kicks/providers/gradientsports/__init__.py`, `silly_kicks/providers/gradientsports/appearances.py` — new package + extractor.
- `silly_kicks/providers/skillcorner/__init__.py`, `silly_kicks/providers/skillcorner/appearances.py` — new package + extractor (handles `meta/` and `matches/` layouts).
- `tests/keeper_identity/` (new dir): `test_appearance_port.py`, `test_interval_resolution.py`, `test_event_only_path.py`, `test_promotion_imports.py`.
- `tests/providers/{statsbomb,sportec,gradientsports,skillcorner}/test_appearances.py`.
- `tests/providers/test_appearances_import_allowlist.py`.
- Committed fixtures: a StatsBomb open-data match JSON **with a real GK substitution** (`tests/datasets/statsbomb/raw/events/<id>.json` + its lineups), a synthetic DFL emergency-keeper fixture, a synthetic GS `SUB` fixture, a SkillCorner public happy-path `match.json` fixture (or reuse the sample dir).
- `docs/superpowers/adrs/ADR-<next>-keeper-appearance-resolution.md` (number assigned at commit-prep).

**Modified:**
- `silly_kicks/tracking/_keeper_identity.py` — **deleted** (moved).
- `silly_kicks/tracking/__init__.py`, `silly_kicks/tracking/features.py`, `silly_kicks/tracking/_run_features.py` — import from the new home; drop the re-exports.
- `scripts/build_tf19_instrument_responsiveness.py`, `scripts/_sb_roster.py`, `scripts/_sb_battery.py` — import from `silly_kicks.keeper_identity`.
- `tests/tracking/test_keeper_identity*.py` — import path + the native monkeypatch seam migration.
- `silly_kicks/providers/__init__.py` — export the two new provider subpackages if the package re-exports (follow existing pattern).
- `docs/PRIVATE_CONSUMERS.md` — remove the `_keeper_identity` private-path pin (module is now public).
- `docs/c4/` model — new provider extractor modules; re-render `architecture.html` via Graphviz `dot`.

---

### Task 1: Promote the resolver to `silly_kicks/keeper_identity.py`

Move the module verbatim (semantics unchanged), rewire every consumer, and migrate the native delegation test seam. This is a mechanical refactor task with a behavioral guard: the full existing keeper-identity suite must pass at the new path.

**Files:**
- Create: `silly_kicks/keeper_identity.py` (content = current `tracking/_keeper_identity.py`, with the native-path import made lazy).
- Delete: `silly_kicks/tracking/_keeper_identity.py`.
- Modify: `silly_kicks/tracking/__init__.py` (drop re-exports of `resolve_keeper_identities`/`add_defending_gk_player_id`/`apply_keeper_identities_to_frames`/`KeeperIdentity`/`KeeperIdentityMap`/`KeeperIdentityReport`/`KEEPER_ID_SOURCE_*`), `silly_kicks/tracking/features.py`, `silly_kicks/tracking/_run_features.py`, `scripts/build_tf19_instrument_responsiveness.py`, `scripts/_sb_roster.py`, `scripts/_sb_battery.py`, `docs/PRIVATE_CONSUMERS.md`.
- Test: `tests/keeper_identity/test_promotion_imports.py`; migrate `tests/tracking/test_keeper_identity.py`, `test_keeper_identity_native.py`, `test_keeper_identity_roster.py`, `test_keeper_identity_contracts.py` (move to `tests/keeper_identity/` or repoint imports — keep them running).

**Interfaces:**
- Produces (unchanged public semantics, new import path): `silly_kicks.keeper_identity.{resolve_keeper_identities, add_defending_gk_player_id, apply_keeper_identities_to_frames, KeeperIdentity, KeeperIdentityMap, KeeperIdentityReport, KEEPER_ID_SOURCE_EVENT, KEEPER_ID_SOURCE_ROSTER, KEEPER_ID_SOURCE_NATIVE, KEEPER_ID_SOURCE_DERIVED, KEEPER_ID_SOURCE_UNRESOLVED, KEEPER_ID_SOURCE_VALUES}`.

- [ ] **Step 1: Write the failing promotion-guard test**

```python
# tests/keeper_identity/test_promotion_imports.py
def test_public_home_exports_the_resolver():
    from silly_kicks.keeper_identity import (
        resolve_keeper_identities, add_defending_gk_player_id,
        apply_keeper_identities_to_frames, KeeperIdentity, KEEPER_ID_SOURCE_VALUES,
    )
    assert callable(resolve_keeper_identities)
    assert set(KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}

def test_old_tracking_path_is_a_clean_break():
    import silly_kicks.tracking as T
    for name in ("resolve_keeper_identities", "add_defending_gk_player_id", "KeeperIdentity"):
        assert not hasattr(T, name), f"{name} must no longer be re-exported from tracking (breaking, no shim)"

def test_importing_keeper_identity_does_not_import_tracking():
    import sys
    for m in [k for k in sys.modules if k.startswith("silly_kicks.tracking")]:
        del sys.modules[m]
    import silly_kicks.keeper_identity  # noqa: F401
    assert "silly_kicks.tracking" not in sys.modules, "keeper_identity must stay tracking-free at import"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python -m pytest tests/keeper_identity/test_promotion_imports.py -v`
Expected: FAIL — `ModuleNotFoundError: silly_kicks.keeper_identity` (module not yet moved). All three tests red before Step 3.

- [ ] **Step 3: `git mv` the module, make the native import lazy, drop tracking re-exports**

```bash
git mv silly_kicks/tracking/_keeper_identity.py silly_kicks/keeper_identity.py
```
In `silly_kicks/keeper_identity.py`, change the top-level `from ._gk_resolve import acting_gk_from_frames, defending_gk_from_frames` into a **lazy import inside `_resolve_from_native`** (function-local), so importing this module does not import `tracking`:

```python
def _resolve_from_native(actions, frames, links=None):
    # Lazy import keeps `import silly_kicks.keeper_identity` light: importing
    # `tracking._gk_resolve` runs tracking/__init__ (numba + ~30 submodules).
    from silly_kicks.tracking._gk_resolve import (
        acting_gk_from_frames,
        defending_gk_from_frames,
    )
    ...
```
Keep the module's `from silly_kicks.id_compat import ...` absolute. In `tracking/__init__.py` delete the re-export lines and the `from .features import ...`/`from ._keeper_identity import ...` for these symbols (**note: the 4.105.0 release — on `origin/main`, which must be merged into the working tree first — modified `tracking/__init__.py`; re-verify the exact keeper-identity re-export lines against the merged file before deleting**); in `tracking/features.py` and `tracking/_run_features.py` replace `from ._keeper_identity import ...` with `from silly_kicks.keeper_identity import ...`.

- [ ] **Step 4: Migrate the native delegation test seam**

In `tests/tracking/test_keeper_identity_native.py` the delegation test patches the module attribute. With the lazy import it must patch the **definition site**:

```python
def test_native_path_delegates_and_does_not_reimplement():
    """Single-source (ADR-055): the native path CALLS the TF-13 resolvers."""
    import silly_kicks.tracking._gk_resolve as GK
    real_def, real_act = GK.defending_gk_from_frames, GK.acting_gk_from_frames
    with (
        mock.patch("silly_kicks.tracking._gk_resolve.defending_gk_from_frames", wraps=real_def) as md,
        mock.patch("silly_kicks.tracking._gk_resolve.acting_gk_from_frames", wraps=real_act) as ma,
    ):
        resolve_keeper_identities(_actions(), _frames(), identity="native")
    assert md.called and ma.called
```
Repoint every `from silly_kicks.tracking._keeper_identity import ...` / `import ... as T; T.resolve_keeper_identities` in the four keeper-identity test files to `silly_kicks.keeper_identity`. The `test_resolver_is_exported_from_tracking` test in `test_keeper_identity_native.py` is now false — replace it with `test_resolver_is_exported_from_keeper_identity` asserting the new home.

- [ ] **Step 5: Rewire scripts + PRIVATE_CONSUMERS**

Replace the resolver imports in `scripts/build_tf19_instrument_responsiveness.py`, `scripts/_sb_roster.py`, `scripts/_sb_battery.py` with `from silly_kicks.keeper_identity import ...`. Remove the `_keeper_identity` entry from `docs/PRIVATE_CONSUMERS.md` (now public).

- [ ] **Step 6: Run the full keeper-identity suite + promotion guard**

Run: `python -m pytest tests/keeper_identity/ tests/tracking/test_keeper_identity*.py -v`
Expected: PASS (all migrated tests green; native delegation proven at the new seam).

- [ ] **Step 7: Grep for stragglers**

Run: `python -m pytest tests/ -m "not e2e" -k "keeper or tf19 or sb_battery or run_features" -q` and `python -m ruff check silly_kicks/ tests/ scripts/`
Expected: no unresolved `tracking._keeper_identity` imports; lint clean.

---

### Task 2: Event-only enumeration path

Make `resolve_keeper_identities` runnable with `frames=None` on the `roster` path by enumerating `(game, period, team)` from `actions`.

**Files:**
- Modify: `silly_kicks/keeper_identity.py` (`resolve_keeper_identities` signature → `frames=None` default; `_resolve_from_roster` enumerates from `actions` when `frames is None`).
- Test: `tests/keeper_identity/test_event_only_path.py`.

**Interfaces:**
- Produces: `resolve_keeper_identities(actions, frames=None, *, identity, roster=None)` — `identity="roster"` works with `frames=None`; `identity="native"` with `frames=None` raises `ValueError` (native needs positions).

- [ ] **Step 1: Write the failing test**

```python
# tests/keeper_identity/test_event_only_path.py
import pandas as pd, pytest
from silly_kicks.keeper_identity import resolve_keeper_identities, KEEPER_ID_SOURCE_ROSTER

def _actions():
    return pd.DataFrame({
        "game_id": [1, 1, 1, 1], "period_id": [1, 1, 2, 2], "team_id": [10, 20, 10, 20],
        "player_id": [901, 902, 901, 902], "type_name": ["pass"]*4, "time_seconds": [1.0, 2.0, 3.0, 4.0],
    })

def test_roster_path_runs_without_frames():
    m, rep = resolve_keeper_identities(_actions(), identity="roster", roster={10: 901, 20: 902})
    assert m[(  # canonical keys per ADR-055 rule 2
        __import__("silly_kicks.id_compat", fromlist=["canonical_id"]).canonical_id(1), 1,
        __import__("silly_kicks.id_compat", fromlist=["canonical_id"]).canonical_id(10))].source == KEEPER_ID_SOURCE_ROSTER
    assert rep.n_resolved >= 1

def test_native_without_frames_raises():
    with pytest.raises(ValueError, match="native.*frames"):
        resolve_keeper_identities(_actions(), identity="native")
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/keeper_identity/test_event_only_path.py -v`
Expected: FAIL (`resolve_keeper_identities` requires `frames`).

- [ ] **Step 3: Implement the event-only enumeration**

In `resolve_keeper_identities`, default `frames=None`. In `_resolve_from_roster`, when `frames is None` build the seed triples from `actions`:

```python
if frames is None:
    seed_df = actions.loc[actions["team_id"].notna(), ["game_id", "period_id", "team_id"]].dropna().drop_duplicates()
    frame_team_values = actions.loc[actions["team_id"].notna(), "team_id"].dropna().unique()
else:
    non_ball = ~frames["is_ball"].astype("boolean").fillna(False)
    frame_team_values = frames.loc[non_ball, "team_id"].dropna().unique()
    seed_df = frames.loc[non_ball, ["game_id", "period_id", "team_id"]].dropna().drop_duplicates()
```
In `_resolve_from_native`, raise `ValueError("native identity requires frames")` if `frames is None`.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/keeper_identity/test_event_only_path.py tests/tracking/test_keeper_identity_roster.py -v`
Expected: PASS (event-only works; existing roster-with-frames unchanged).

---

### Task 3: Keeper appearance-interval port

Add the normalized `KeeperAppearances` schema + a validator.

**Files:**
- Modify: `silly_kicks/keeper_identity.py` (add `KEEPER_APPEARANCE_COLUMNS`, `KEEPER_APPEARANCE_SOURCE_VALUES`, `validate_keeper_appearances`).
- Test: `tests/keeper_identity/test_appearance_port.py`.

**Interfaces:**
- Produces:
  - `KEEPER_APPEARANCE_COLUMNS: dict[str, str]` = `{"game_id": "object", "team_id": "object", "player_id": "object", "period_id": "int64", "start_time_seconds": "float64", "end_time_seconds": "float64", "source": "object"}` (**all three ids `object`** — tolerant of string ids like DFL/SkillCorner *and* numeric ids; comparisons route through `id_compat`, never a raw dtype cast — DFL `MatchInfo.gk_player_ids` is `frozenset[str]`, so `Int64` would drop it).
  - `KEEPER_APPEARANCE_SOURCE_VALUES: tuple[str, ...]` = `("native_intervals", "sub_events", "starting_xi", "emergency_keeper")`.
  - `validate_keeper_appearances(df) -> pd.DataFrame` — raises `ValueError` on missing columns / non-period-relative negatives / `start >= end`; returns the frame unchanged on success.

- [ ] **Step 1: Write the failing test**

```python
# tests/keeper_identity/test_appearance_port.py
import numpy as np, pandas as pd, pytest
from silly_kicks.keeper_identity import (
    KEEPER_APPEARANCE_COLUMNS, KEEPER_APPEARANCE_SOURCE_VALUES, validate_keeper_appearances,
)

def _appearances():
    return pd.DataFrame({
        "game_id": ["g1", "g1"], "team_id": pd.array([10, 20], dtype="Int64"),
        "player_id": pd.array([901, 902], dtype="Int64"), "period_id": [1, 1],
        "start_time_seconds": [0.0, 0.0], "end_time_seconds": [np.inf, np.inf],
        "source": ["starting_xi", "starting_xi"],
    })

def test_valid_appearances_round_trip():
    df = validate_keeper_appearances(_appearances())
    assert list(df.columns) == list(KEEPER_APPEARANCE_COLUMNS)

def test_missing_column_raises():
    with pytest.raises(ValueError, match="missing"):
        validate_keeper_appearances(_appearances().drop(columns=["period_id"]))

def test_start_after_end_raises():
    bad = _appearances(); bad.loc[0, "start_time_seconds"] = 100.0; bad.loc[0, "end_time_seconds"] = 10.0
    with pytest.raises(ValueError, match="start.*end"):
        validate_keeper_appearances(bad)

def test_source_vocab_is_closed():
    assert set(KEEPER_APPEARANCE_SOURCE_VALUES) == {"native_intervals", "sub_events", "starting_xi", "emergency_keeper"}

def test_string_ids_are_tolerated():
    # DFL ids are strings (MatchInfo.gk_player_ids: frozenset[str]); the port must accept them un-coerced.
    df = _appearances().copy()
    df["team_id"] = ["DFL-CLU-00000G", "DFL-CLU-00000P"]
    df["player_id"] = ["DFL-OBJ-0027AX", "DFL-OBJ-0027V2"]
    out = validate_keeper_appearances(df)
    assert list(out["player_id"]) == ["DFL-OBJ-0027AX", "DFL-OBJ-0027V2"]  # no Int64 coercion / no raise
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/keeper_identity/test_appearance_port.py -v`
Expected: FAIL (symbols not defined).

- [ ] **Step 3: Implement the schema + validator**

```python
KEEPER_APPEARANCE_COLUMNS = {
    "game_id": "object", "team_id": "object", "player_id": "object", "period_id": "int64",
    "start_time_seconds": "float64", "end_time_seconds": "float64", "source": "object",
}
KEEPER_APPEARANCE_SOURCE_VALUES = ("native_intervals", "sub_events", "starting_xi", "emergency_keeper")

def validate_keeper_appearances(df):
    missing = [c for c in KEEPER_APPEARANCE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"keeper appearances missing column(s): {missing}")
    s, e = df["start_time_seconds"], df["end_time_seconds"]
    if (s < 0).any():
        raise ValueError("start_time_seconds must be period-relative (>= 0)")
    if (s >= e).any():
        raise ValueError("each appearance needs start < end")
    bad_src = set(df["source"].dropna()) - set(KEEPER_APPEARANCE_SOURCE_VALUES)
    if bad_src:
        raise ValueError(f"unknown appearance source(s): {sorted(bad_src)}")
    return df
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/keeper_identity/test_appearance_port.py -v`
Expected: PASS.

---

### Task 4: Interval-granular resolution in `add_defending_gk_player_id`

Add `appearances=None`; when supplied, resolve each action's defending keeper by the interval covering its time. **Byte-identical when omitted.**

**Files:**
- Modify: `silly_kicks/keeper_identity.py` (`add_defending_gk_player_id` gains `*, appearances=None`).
- Test: `tests/keeper_identity/test_interval_resolution.py`.

**Interfaces:**
- Consumes: `KeeperAppearances` (Task 3), the resolved `keeper_map` (Task 1).
- Produces: `add_defending_gk_player_id(actions, keeper_map, *, appearances=None) -> pd.DataFrame` — new `defending_gk_player_id` column; interval wins over map where an interval covers the action time; gap/absent → map fallback; unresolvable → `pd.NA`. **When `appearances` is supplied, also emits `defending_gk_source`** over the closed vocab `{appearance, map_fallback, appearance_map_conflict, unresolved}` — this realizes spec §5.4's appearance↔map cross-check (a disagreement between the interval-resolved keeper and the coarse-map keeper is a durable per-row signal — the ADR-054 source-column pattern). `defending_gk_source` is emitted **only** on the appearance path, so the omit-`appearances` output stays byte-identical. Also produces the module constant `DEFENDING_GK_SOURCE_VALUES`.

- [ ] **Step 1: Write the failing test (both sides)**

```python
# tests/keeper_identity/test_interval_resolution.py
import numpy as np, pandas as pd
from silly_kicks.id_compat import canonical_id
from silly_kicks.keeper_identity import (
    resolve_keeper_identities, add_defending_gk_player_id, validate_keeper_appearances,
)

def _actions():
    # shots by team 10 in period 1 at t=100 (before sub) and t=3000 (after sub); defending team 20
    return pd.DataFrame({
        "game_id": ["g", "g"], "period_id": [1, 1], "team_id": [10, 10],
        "player_id": [500, 500], "type_name": ["shot", "shot"], "time_seconds": [100.0, 3000.0],
    })

def _appearances_with_gk_sub():
    # team 20 keeper 902 until t=2700, then keeper 999 from 2700
    return validate_keeper_appearances(pd.DataFrame({
        "game_id": ["g", "g", "g"], "team_id": pd.array([10, 20, 20], dtype="Int64"),
        "player_id": pd.array([901, 902, 999], dtype="Int64"), "period_id": [1, 1, 1],
        "start_time_seconds": [0.0, 0.0, 2700.0], "end_time_seconds": [np.inf, 2700.0, np.inf],
        "source": ["starting_xi", "starting_xi", "sub_events"],
    }))

def test_attribution_flips_at_the_sub_minute():
    m, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 901, 20: 902})
    out = add_defending_gk_player_id(_actions(), m, appearances=_appearances_with_gk_sub())
    ids = list(out["defending_gk_player_id"])
    assert canonical_id(ids[0]) == canonical_id(902), "pre-sub shot -> starter keeper"
    assert canonical_id(ids[1]) == canonical_id(999), "post-sub shot -> replacement keeper"

def test_omitting_appearances_is_byte_identical():
    m, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 901, 20: 902})
    base = add_defending_gk_player_id(_actions(), m)
    also = add_defending_gk_player_id(_actions(), m, appearances=None)
    pd.testing.assert_frame_equal(base, also)
    assert "defending_gk_source" not in base.columns  # provenance is appearance-path only
    # both shots attribute to the coarse map keeper 902 (no interval)
    assert all(canonical_id(v) == canonical_id(902) for v in base["defending_gk_player_id"])

def test_conflict_flagged_when_interval_disagrees_with_map():
    # coarse map says team-20 keeper is 902 all period; the interval says 999 after the sub (spec §5.4 cross-check).
    m, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 901, 20: 902})
    out = add_defending_gk_player_id(_actions(), m, appearances=_appearances_with_gk_sub())
    src = list(out["defending_gk_source"])
    assert src[0] == "appearance"               # pre-sub: interval agrees with the map (902)
    assert src[1] == "appearance_map_conflict"  # post-sub: interval 999 disagrees with map 902
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/keeper_identity/test_interval_resolution.py -v`
Expected: FAIL (`appearances` kwarg unknown / no flip).

- [ ] **Step 3: Implement interval resolution**

Extend `add_defending_gk_player_id` (keep the existing `by_gp` opponent-derivation for the fallback). When `appearances` is not None, build a per-`(game, period, team)` sorted interval list and, for each action, pick the defending team's keeper whose `[start, end)` covers `time_seconds`; else fall back to the coarse map value already computed:

```python
def add_defending_gk_player_id(actions, keeper_map, *, appearances=None):
    out = actions.copy()
    fallback = _coarse_defending_gk(actions, keeper_map)  # existing per-action map logic, refactored out
    if appearances is None:
        out["defending_gk_player_id"] = fallback
        return out  # byte-identical omit path: no defending_gk_source column
    intervals = _index_appearances(appearances)  # {(canonical g, period, canonical team): [(start, end, gk), ...]}
    vals, srcs = [], []
    for i, (g, p, team, t) in enumerate(zip(actions["game_id"], actions["period_id"], actions["team_id"], actions["time_seconds"])):
        opp = _defending_team_for(g, p, team, keeper_map)  # opponent within (g, p); None if not two-team
        key = (canonical_id(g), p, canonical_id(opp)) if opp is not None else None
        gk = _keeper_covering(intervals.get(key, ()), t) if key is not None else pd.NA
        coarse = fallback.iloc[i]
        if gk is not pd.NA:                                    # spec §5.4 appearance↔map cross-check
            src = "appearance_map_conflict" if (coarse is not pd.NA and not same_id(gk, coarse)) else "appearance"
            vals.append(gk)
        elif coarse is not pd.NA:
            src = "map_fallback"; vals.append(coarse)
        else:
            src = "unresolved"; vals.append(pd.NA)
        srcs.append(src)
    out["defending_gk_player_id"] = pd.Series(vals, index=out.index, dtype="object")
    out["defending_gk_source"] = pd.Series(srcs, index=out.index, dtype="object")  # closed vocab; appearance-path only
    return out
```
`_keeper_covering(rows, t)` returns the `gk` of the first `(start, end, gk)` with `start <= t < end` (treat `end=inf`/NaN as open), else `pd.NA`. Use `id_compat.canonical_id` for all keys and `id_compat.same_id` for the coarse-vs-interval comparison; never `astype(str)`. Register `DEFENDING_GK_SOURCE_VALUES = ("appearance", "map_fallback", "appearance_map_conflict", "unresolved")` as a module constant.

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest tests/keeper_identity/test_interval_resolution.py -v`
Expected: PASS (flip at sub minute; byte-identical when omitted).

- [ ] **Step 5: Guard the existing coarse callers**

Run: `python -m pytest tests/ -m "not e2e" -k "keeper or defending_gk or run_features or gk_role" -q`
Expected: PASS (coarse path unchanged — refactor is behavior-preserving).

---

### Task 5: StatsBomb / SB360 extractor

`extract_keeper_appearances` from StatsBomb raw events + lineups (`Starting XI` → openers; `Substitution` → boundaries; incoming `position == "Goalkeeper"` → GK sub). Commit a **real open-data GK-sub fixture**.

**Files:**
- Create: `silly_kicks/providers/statsbomb/appearances.py`.
- Create fixtures: a StatsBomb open-data `events/<id>.json` **and** `lineups/<id>.json` for a GK-sub match (Step 1), **plus `lineups/7584.json`** (the full-match test reads it; the repo has **no `lineups/` dir** yet — both must be committed). All redistributable open data.
- Test: `tests/providers/statsbomb/test_appearances.py`.

**Interfaces:**
- Produces: `extract_keeper_appearances(events: list[dict], lineups: list[dict], *, game_id) -> pd.DataFrame` (validated `KeeperAppearances`).

- [ ] **Step 1: Identify + commit a real GK-sub open-data match (with lineups)**

The three committed fixtures (`3754058`, `7298`, `7584`) have no GK sub, and the repo has **no committed `lineups/` dir**. From the StatsBomb open-data index find a match whose events contain a `Substitution` where the outgoing keeper's lineup position is `Goalkeeper` (a keeper red card / injury). Method (redistributable open data): scan `events/*.json` for a `Substitution` whose outgoing `player.position.name == "Goalkeeper"`. Commit that match's `events/<id>.json` **and** `lineups/<id>.json`, **plus `lineups/7584.json`** (the full-match test needs it), under `tests/datasets/statsbomb/raw/`. Record the chosen match id in `GK_SUB_MATCH`. **Fallback if no GK-sub open-data match is found quickly:** hand-author a minimal synthetic StatsBomb-shaped `events`+`lineups` pair mirroring the real `Starting XI`/`Substitution` structure (the real committed fixture is preferred and can replace it later) — the unit test must not be left blocked on discovery.

- [ ] **Step 2: Write the failing test**

```python
# tests/providers/statsbomb/test_appearances.py
import json, pathlib, numpy as np
from silly_kicks.providers.statsbomb.appearances import extract_keeper_appearances

RAW = pathlib.Path(__file__).parents[2] / "datasets/statsbomb/raw"
GK_SUB_MATCH = "<chosen_id>"  # set in Step 1

def _load(kind, mid):
    return json.load(open(RAW / kind / f"{mid}.json", encoding="utf-8"))

def test_starting_keepers_and_gk_sub_interval():
    ev, lu = _load("events", GK_SUB_MATCH), _load("lineups", GK_SUB_MATCH)
    ap = extract_keeper_appearances(ev, lu, game_id=GK_SUB_MATCH)
    # exactly two teams, each with a starting keeper interval from 0
    starters = ap[ap["start_time_seconds"] == 0.0]
    assert starters["team_id"].nunique() == 2
    # the subbed team has TWO keeper intervals in the sub period (starter closes, replacement opens)
    subbed = ap.groupby(["team_id", "period_id"]).size()
    assert (subbed >= 2).any(), "a GK sub yields >=2 keeper intervals for that (team, period)"

def test_full_match_keeper_has_open_interval():
    ev, lu = _load("events", "7584"), _load("lineups", "7584")  # no GK sub -> both keepers full match
    ap = extract_keeper_appearances(ev, lu, game_id="7584")
    assert np.isinf(ap["end_time_seconds"]).sum() == 2  # both starters play to the whistle
```

- [ ] **Step 3: Run to verify it fails**

Run: `python -m pytest tests/providers/statsbomb/test_appearances.py -v`
Expected: FAIL (module missing).

- [ ] **Step 4: Implement the extractor**

Read `Starting XI` events → `tactics.lineup[]`, keeper = `position.name == "Goalkeeper"`, opens `(team, period=1, start=0)`. Walk `Substitution` events in chronological order: convert `minute/second` + `period` to **period-relative** `time_seconds` (subtract the period's start minute); when the outgoing player is the current keeper, close their interval at that time and open the replacement's from that time (source `sub_events`). Emergency `Player Off`/`Player On` without replacement handled likewise if the affected player is the keeper. Emit keeper rows only; `validate_keeper_appearances` before returning. All id handling via `id_compat`.

- [ ] **Step 5: Run to verify it passes**

Run: `python -m pytest tests/providers/statsbomb/test_appearances.py -v`
Expected: PASS.

---

### Task 6: Sportec / DFL extractor

**Files:**
- Create: `silly_kicks/providers/sportec/appearances.py`.
- Test: `tests/providers/sportec/test_appearances.py` (unit builds a synthetic **bronze DataFrame** — no XML fixture; an `@e2e` parses real IDSSE).

**Interfaces:**
- Produces: `extract_keeper_appearances(match_info: MatchInfo, events_bronze: pd.DataFrame, *, game_id) -> pd.DataFrame` — consumes the **parsed DFL bronze DataFrame** (from `parse_dfl_events`, which retains `sub_player_in`/`sub_player_out`/`sub_playing_position`/`sub_team` + `other_action_player_becomes_goalkeeper` + event timing) and `MatchInfo.gk_player_ids`/`player_team_map` (openers). It reads bronze COLUMNS, not raw XML — no re-parse (the parse signature is `parse_dfl_events(events_path, *, match_info, match_id)`, parse.py:1944, so re-parsing text is wrong). **DFL ids are strings** (`gk_player_ids: frozenset[str]`), consumed as-is (the port is `object`).

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/sportec/test_appearances.py
import pandas as pd
from silly_kicks.providers.sportec.appearances import extract_keeper_appearances
from silly_kicks.providers.sportec.parse import MatchInfo

def _match_info():
    # DFL string ids; two starting keepers (GK-H, GK-A) + a bench keeper GK-H2.
    # NOTE: match MatchInfo's real fields (parse.py:1852-1863) — add any required fields at implementation.
    return MatchInfo(
        home_team_id="CLU-H", away_team_id="CLU-A",
        player_team_map={"GK-H": "CLU-H", "GK-H2": "CLU-H", "GK-A": "CLU-A", "OUT-A": "CLU-A"},
        gk_player_ids=frozenset({"GK-H", "GK-H2", "GK-A"}),
    )

def _bronze():
    # one TW substitution (GK-H off, GK-H2 on) + one emergency keeper (OUT-A becomes GK).
    # Column names per spadl/sportec.py:335-341 + parse.py:1664 — confirm exact names at implementation.
    return pd.DataFrame([
        {"event_type": "Substitution", "sub_player_out": "GK-H", "sub_player_in": "GK-H2",
         "sub_playing_position": "TW", "sub_team": "CLU-H", "period_id": 1, "timestamp_seconds": 2700.0,
         "other_action_player_becomes_goalkeeper": None},
        {"event_type": "OtherPlayerAction", "sub_player_out": None, "sub_player_in": None,
         "sub_playing_position": None, "sub_team": "CLU-A", "period_id": 2, "timestamp_seconds": 500.0,
         "other_action_player_becomes_goalkeeper": "OUT-A"},
    ])

def test_gk_sub_and_emergency_keeper():
    ap = extract_keeper_appearances(_match_info(), _bronze(), game_id="SYN")
    assert ap.groupby(["team_id", "period_id"]).size().max() >= 2   # TW sub -> 2 keeper intervals for CLU-H
    assert (ap["source"] == "emergency_keeper").any()               # OUT-A becomes GK
    assert set(ap["player_id"]) >= {"GK-H", "GK-H2", "OUT-A"}       # string ids preserved
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/providers/sportec/test_appearances.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement (bronze → appearances)**

Confirm the bronze column names against `silly_kicks/spadl/sportec.py:335-341` + `silly_kicks/providers/sportec/parse.py:1664` (`sub_player_in`/`sub_player_out`/`sub_playing_position`/`sub_team` + `other_action_player_becomes_goalkeeper`; event time via the bronze timestamp/frame column). Implement: openers from `match_info.gk_player_ids` mapped to team via `match_info.player_team_map` (per team, period 1, start 0); for each bronze row with `sub_playing_position == "TW"`, read its period-relative time, close the outgoing keeper (`sub_player_out`) and open the incoming (`sub_player_in`, source `sub_events`); for each row with a non-null `other_action_player_becomes_goalkeeper`, open an `emergency_keeper` interval for that player and close the prior keeper. String ids consumed as-is; `validate_keeper_appearances` before return.

- [ ] **Step 4: Run to verify it passes + add the IDSSE e2e**

Run: `python -m pytest tests/providers/sportec/test_appearances.py -v`
Add an `@pytest.mark.e2e` test that parses a real IDSSE match via `parse_dfl_match_info(info_path)` + `parse_dfl_events(events_path, match_info=..., match_id=...)` (pulled through `scripts/_loader_pining`, public token) and asserts the extractor validates + returns two starting-keeper intervals (the 7 public IDSSE matches have no GK sub, so this exercises the happy path + the real bronze shape). Skips without the pining token.
Expected: unit PASS; e2e skips locally / runs where pining is reachable.

---

### Task 7: Gradient Sports extractor (new package)

**Files:**
- Create: `silly_kicks/providers/gradientsports/__init__.py`, `silly_kicks/providers/gradientsports/appearances.py`.
- Test: `tests/providers/gradientsports/test_appearances.py` (synthetic `SUB` gameEvents + a roster stub; owner e2e marked `@e2e`).

**Interfaces:**
- Produces: `extract_keeper_appearances(events: list[dict], roster: list[dict], *, game_id) -> pd.DataFrame` — openers from roster `positionGroupType` GK; `SUB` gameEvents (`playerOffId`/`playerOnId` + `startGameClock`/`period`) → boundaries when a keeper is involved.

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/gradientsports/test_appearances.py
from silly_kicks.providers.gradientsports.appearances import extract_keeper_appearances

def _roster():  # two GKs + outfielders
    return [{"player": {"id": 901}, "team": {"id": 10}, "positionGroupType": "GK"},
            {"player": {"id": 902}, "team": {"id": 20}, "positionGroupType": "GK"},
            {"player": {"id": 999}, "team": {"id": 20}, "positionGroupType": "GK"}]

def _events_gk_sub():
    return [{"gameEvents": {"gameEventType": "SUB", "startGameClock": 2830, "period": 1,
                            "playerOffId": 902, "playerOnId": 999, "teamId": 20}}]

def test_gk_sub_creates_two_intervals():
    ap = extract_keeper_appearances(_events_gk_sub(), _roster(), game_id="gs1")
    team10 = set(int(x) for x in ap[ap["team_id"] == 10]["player_id"])
    team20 = set(int(x) for x in ap[ap["team_id"] == 20]["player_id"])
    assert team10 == {901}          # team 10 keeper plays the whole match
    assert team20 == {902, 999}     # team 20: starter 902 then replacement 999

def test_outfielder_sub_creates_no_keeper_interval():
    ev = [{"gameEvents": {"gameEventType": "SUB", "startGameClock": 3000, "period": 2,
                          "playerOffId": 500, "playerOnId": 501, "teamId": 10}}]
    ap = extract_keeper_appearances(ev, _roster(), game_id="gs1")
    assert (ap["source"] == "sub_events").sum() == 0  # non-keeper sub does not touch keeper intervals
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/providers/gradientsports/test_appearances.py -v`
Expected: FAIL (package missing).

- [ ] **Step 3: Implement**

Build GK id set from roster (`positionGroupType` in {"GK","GOALKEEPER"}, per team). Openers per team (period 1, start 0). For each `SUB` gameEvent where `playerOffId` is a GK: `startGameClock` is already period-relative seconds → close outgoing keeper, open incoming (`playerOnId`) from that time. `teamId` may be null on the SUB — derive the team from the outgoing player's roster team. `validate_keeper_appearances` before return. ids via `id_compat`.

- [ ] **Step 4: Run to verify it passes + add the owner e2e**

Run: `python -m pytest tests/providers/gradientsports/test_appearances.py -v`
Add an `@pytest.mark.e2e` test that pulls one WC2022 GS match via `scripts/_loader_pining` (owner token) and asserts the extractor returns ≥2 keeper intervals total and validates. Skips without `PINING_FOR_THE_DATA_TOKEN`.
Expected: unit PASS; e2e skips locally / runs on the owner box.

---

### Task 8: SkillCorner extractor (new package, two layouts)

**Files:**
- Create: `silly_kicks/providers/skillcorner/__init__.py`, `silly_kicks/providers/skillcorner/appearances.py`.
- Test: `tests/providers/skillcorner/test_appearances.py` (public happy-path fixture; peggy44 e2e marked `@e2e`).

**Interfaces:**
- Produces: `extract_keeper_appearances(match_json: dict, *, game_id=None) -> pd.DataFrame` — reads `players[]` (`player_role.acronym=="GK"`, `playing_time.by_period[]` `start_frame`/`end_frame`) + `match_periods` to convert frames → period-relative seconds. `source` is **always `native_intervals`** (the closed vocab has no `red_card` token; a red-carded keeper's interval simply ends at their `end_frame` — the boundary captures it). SkillCorner ids are strings → the `object` port consumes them as-is.

- [ ] **Step 1: Write the failing test**

```python
# tests/providers/skillcorner/test_appearances.py
import json, pathlib, numpy as np
from silly_kicks.providers.skillcorner.appearances import extract_keeper_appearances

FIX = pathlib.Path(__file__).parents[2] / "datasets/skillcorner/public_match.json"

def test_public_match_two_full_keepers():
    ap = extract_keeper_appearances(json.load(open(FIX, encoding="utf-8")))
    starters = ap[ap["start_time_seconds"] == 0.0]
    assert starters["team_id"].nunique() == 2
    # a full-match keeper's last-period interval ends at that period's length (finite, from match_periods)
    assert (ap["end_time_seconds"] > 0).all() and not ap["end_time_seconds"].isna().any()

def test_frames_convert_to_period_relative_seconds():
    ap = extract_keeper_appearances(json.load(open(FIX, encoding="utf-8")))
    # period 1 starter opens at 0.0 (start_frame maps to the period start)
    p1 = ap[(ap["period_id"] == 1) & (ap["start_time_seconds"] == 0.0)]
    assert len(p1) == 2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/providers/skillcorner/test_appearances.py -v`
Expected: FAIL.

- [ ] **Step 3: Commit the public happy-path fixture + implement**

Fetch one public A-League `match.json` (e.g. `1886347`, via `scripts/_loader_pining`, public token) to `tests/datasets/skillcorner/public_match.json` (redistributable open data). **The public `match.json` carries `players[].playing_time.by_period[]` `start_frame`/`end_frame` + `match_periods` — probe-verified 2026-09-01** (`has_playing_time=True` on all 10 public matches). Implement: for each `players[]` with `player_role.acronym=="GK"`, for each `playing_time.by_period[]` entry, map `start_frame`/`end_frame` to seconds relative to that period's `match_periods[start_frame]` at the feed's fps (frames ÷ fps; derive fps from `match_periods` `duration_frames`/`duration_minutes`×60 and assert it against the documented 10 fps). Set `source="native_intervals"` for every interval. Accept **both layouts**: the function takes the parsed `match.json` dict whether it came from `meta/<id>.json` (24/25) or `matches/<id>.json` (25/26) — identical schema. `validate_keeper_appearances` before return.

- [ ] **Step 4: Run to verify it passes + add the peggy44 e2e**

Run: `python -m pytest tests/providers/skillcorner/test_appearances.py -v`
Add an `@pytest.mark.e2e` test over the peggy44 GK-change matches (via HF; skips without HF token) asserting a subbed match yields ≥2 keeper intervals for the subbed team and interval boundaries align to the `end_frame`.
Expected: unit PASS; e2e skips locally / runs where HF access exists.

---

### Task 9: Import-allowlist, ADR, C4, docs

**Files:**
- Create: `tests/providers/test_appearances_import_allowlist.py`.
- Create: `docs/superpowers/adrs/ADR-<next>-keeper-appearance-resolution.md` (number at commit-prep).
- Modify: `silly_kicks/providers/__init__.py` (register new subpackages per existing pattern); `docs/c4/` + re-render `architecture.html`.

**Interfaces:** none (guards + docs).

- [ ] **Step 1: Write the import-allowlist test**

```python
# tests/providers/test_appearances_import_allowlist.py
import ast, pathlib
FILES = list((pathlib.Path("silly_kicks/providers")).rglob("appearances.py"))

def test_extractors_do_not_import_tracking_or_metric():
    for f in FILES:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        mods |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(m and (m.startswith("silly_kicks.tracking") or m.startswith("silly_kicks.shot_stopping")) for m in mods), f"{f} must not import tracking/shot_stopping"
        assert any(m and m.startswith("silly_kicks.keeper_identity") for m in mods), f"{f} must produce the keeper_identity port"
```

- [ ] **Step 2: Run to verify it fails, then satisfy it**

Run: `python -m pytest tests/providers/test_appearances_import_allowlist.py -v`
Expected: FAIL first if any extractor imports tracking; adjust extractors to import only `silly_kicks.keeper_identity` (for `validate_keeper_appearances`) + `id_compat` + `providers.<self>.parse`. Re-run → PASS.

- [ ] **Step 3: Write the ADR (number left as `ADR-<next>`)**

Draft `ADR-<next>-keeper-appearance-resolution.md`: context (ADR-078 foreclosed subs; TF-59 needs gold-standard attribution), decision (promote resolver; event-only path; appearance-interval port; interval resolution; four extractors; placement (i)), consequences (breaking import move; no retrain; amends ADR-078), alternatives (per spec §10). Leave the number token literal `ADR-<next>` until commit-prep.

- [ ] **Step 4: Update C4 + re-render**

Add the new provider extractor modules to the C4 model; re-render per house rule:
Run: `structurizr.war export ... → c4_assemble.py --inject-wrap-width → plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg *.puml → c4_assemble.py --svg-dir`
Verify viewBoxes changed via `dot` (not Smetana) and the assemble did not abort on a 0-entity placeholder.

- [ ] **Step 5: Run the full non-e2e suite + lint + types**

Run: `python -m pytest tests/ -m "not e2e" -q` ; `python -m ruff check silly_kicks/ tests/ scripts/` ; `python -m ruff format --check silly_kicks/ tests/ scripts/` ; `python -m pyright`
Expected: all green.

---

### Task 10: Commit-prep gate (single commit, explicit approval)

No code changes — this is the gate. **Do not commit automatically.**

- [ ] **Step 1: Sync + branch check**

Run: `git fetch && git merge origin/main` (resolve BOM/CRLF traps per the house note). Confirm on a feature branch, not `main`; if on `main`, create one. Re-derive NEXT-FREE version / PR-S / ADR numbers from `origin/main` + memory; fill the `ADR-<next>` token, the CHANGELOG entry, the version bump in the five single-source places (`silly_kicks/_version.py` drives it), and the TODO release line.

- [ ] **Step 2: Full CI-faithful gate**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short` ; ruff check + format --check ; pyright. Owner-run the `@e2e` GS + SkillCorner extractor tests on the boxes that have access.
Expected: green everywhere.

- [ ] **Step 3: Show the diff and STOP for approval**

Run: `git status` + `git --no-pager diff --stat` + the full `git --no-pager diff`. Present the complete change set (spec + TODO CSE split ride this commit) to Karsten. **Wait for explicit "yes, commit" for this specific diff.** Do not proceed without it.

- [ ] **Step 4: Commit (only after approval)**

On explicit approval, one commit for all of PR1 with the standard trailer. Do not push unless separately approved.

---

## Self-Review

- **Spec coverage:** §5.1 (Task 1), §5.2 (Task 2), §5.3 (Task 3), §5.4 (Task 4), §5.5 four extractors (Tasks 5–8, placement (i) — new GS/SkillCorner packages), §5.6 blast radius (Task 1 consumers + Task 9 docs), §5.7 validation (per-task tests + Task 9 allowlist + e2e), §9 bookkeeping (Task 9), §12 delivery/commit discipline (Task 10). PR2 (§6, the metric) is a separate plan — out of scope here, by the two-PR arc.
- **Placeholder scan:** the only intentional token is `ADR-<next>` and the StatsBomb `GK_SUB_MATCH` id, both resolved during execution (Task 5 Step 1 / Task 10 Step 1); no vague "TODO/handle edge cases".
- **Type consistency:** `extract_keeper_appearances` returns a validated `KeeperAppearances` in every extractor; `add_defending_gk_player_id(actions, keeper_map, *, appearances=None)` and `resolve_keeper_identities(actions, frames=None, *, identity, roster=None)` signatures are consistent across Tasks 1/2/4 and the extractor tests.
- **Commit discipline:** no per-task commit steps anywhere; the sole commit is Task 10, gated on explicit approval — matches the Global Constraints.
