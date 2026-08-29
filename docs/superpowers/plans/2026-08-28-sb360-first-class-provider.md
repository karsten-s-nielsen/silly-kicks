# SB360 First-Class Tracking-Feature Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make StatsBomb-360 a first-class, single-sourced tracking-feature provider — a keeper-identity resolver that unlocks the GK aggregators on anonymous freeze-frames, one canonical `add_*` call convention, and a provider-agnostic `run_tracking_features` producer.

**Architecture:** Dependency-inverted and hexagonal. The library consumes *injected* artifacts (a `{team_id: gk_id}` roster dict, a fitted `ExpectedThreat`, an `xg_column`); the driver (`scripts/`) builds them; `providers/statsbomb` stays pure-shaping (ADR-054). Keeper identity has ONE resolver in `tracking/` — its native path *delegates* to the existing TF-13 `*_gk_from_frames` functions (ADR-055 single-source); only the SB360 roster/event ladder is new. The producer orchestrates the already-audited `add_*` family; correctness is proven by equivalence to composing them.

**Tech Stack:** Python 3.10–3.12, pandas 2.3.3 / 3.0.x (ADR-057 span), numpy. Tests via `pytest`. Two venvs: `.venv` (py3.10 / pandas 2, has ruff + pyright) and `.venv312` (py3.12 / pandas 3, CI-repro — use for the pandas-3 behaviour checks).

**Spec:** `docs/superpowers/specs/2026-08-28-sb360-first-class-provider-design.md`. Read it once for the WHY; this plan is the HOW.

## Global Constraints

Every task's requirements implicitly include this section.

- **Injected artifacts, never fetched.** The library never parses raw StatsBomb JSON. The roster (`{team_id: gk_id}`), `xt` (`ExpectedThreat`), and `xg_column` are caller-supplied. `providers/statsbomb` stays pure-shaping (ADR-054): raw-JSON parsing is `scripts/`-side.
- **Single-source keeper identity (ADR-055).** There is exactly ONE keeper-identity resolver, `tracking.resolve_keeper_identities`. Its `identity="native"` path **delegates** to the existing (unchanged) `defending_gk_from_frames` / `acting_gk_from_frames` — it MUST NOT reimplement frame-based identity resolution. Only the `identity="roster"` (SB360) path is new work.
- **gkdv reaches the resolver DRIVER-side only (ADR-037).** `gkdv/` may import only `tracking._das` (via `_das_port`). A gkdv-domain consumer of the resolver goes through a `scripts/` driver, never a gkdv library import. (No code in this plan lives in `gkdv/`.)
- **Honest degradation (ADR-063 / ADR-054).** Velocity-constitutive families stay honest-NaN on velocity-less SB360 frames (naming the keeper does NOT make them scoreable). An unresolved keeper → NA, dropped-and-**counted**, never a fabricated id. An absent injected model → that family's columns are honest-NaN, never a fabricated value.
- **Id comparisons via `id_compat` (ADR-019).** Every id comparison/lookup (roster keys vs frame `team_id`, action↔frame) goes through `silly_kicks.id_compat` (`ids_match` / `ids_equal` / `same_id` / `canonical_id`). Never raw `==`, never `astype(str)` on an id used as a dict key. Map keys are canonical (`canonical_id`).
- **Reports conserve (ADR-052).** `KeeperIdentityReport` and `TrackingFeaturesReport` conserve exactly: resolved + unresolved == total; families run + skipped == families in.
- **Return shapes (decided in spec, do not re-open):**
  - `resolve_keeper_identities(...) -> tuple[KeeperIdentityMap, KeeperIdentityReport]`, where `KeeperIdentityMap` is a pure mapping `{(canonical(game_id), period_id, canonical(team_id)) → KeeperIdentity(gk_id, source, conflict)}`. The resolver mutates NEITHER `actions` NOR `frames`. The map is applied by two pure placement helpers (`add_defending_gk_player_id`, `apply_keeper_identities_to_frames`).
  - `run_tracking_features(...) -> tuple[pd.DataFrame, TrackingFeaturesReport]`, returning the **enriched ACTIONS** (action-grain columns; the family is action-coupled) plus the report.
- **Identity→frame bridge (R1) — load-bearing.** `add_pre_shot_gk_position` matches `frame.player_id == defending_gk_player_id` (`utils.py:1034`); SB360 frames carry synthetic numbered ids, so the resolved roster id must be stamped onto the frames' `is_goalkeeper` rows (`apply_keeper_identities_to_frames`) or the GK-position features are silently NaN. The producer applies the frame bridge ONLY on the roster path (`identity="roster"`); on the native path frames already carry real ids and a bridge would clobber a mid-period sub. Every keystone test carries a `pre_shot_gk_x.notna().any()` non-vacuity assertion.
- **`keeper_id_source` vocabulary:** `{event, roster, native, derived, unresolved}` — the roster path emits `event`/`roster`, the native path emits `native`/`derived` (inherited from the resolved keeper's frame `is_goalkeeper_source`), and `unresolved` everywhere. Mirror the `_das.py` `DAS_SOURCE_*` idiom (module-level constants + a `KEEPER_ID_SOURCE_VALUES` tuple, exported via `tracking/__init__.py`).
- **Canonical `add_*` call shape:** `frames` is NEVER keyword-only (positional-or-keyword, next to `actions`); every OPTIONAL parameter is keyword-only; a single REQUIRED fitted model (`xt: ExpectedThreat`) MAY be the 3rd positional. The ONLY signature change is `add_pre_shot_gk_angle` (its keyword-only `frames` → positional). Do NOT churn the 5 positional-`xt` aggregators (108 call sites, type-guarded, no correctness payoff).
- **No commit, ever.** The user commits once, at the end, on their own explicit approval. Do not run `git commit`.
- **Version / `PR-Sxxx` / `ADR-0xx` are PLACEHOLDERS throughout.** They are assigned ONLY at the single end-of-cycle commit, confirmed next-free against `main` (another session may take the next number first). Five version sites: `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` (via `uv lock`, never hand-edited), `CHANGELOG.md`, `TODO.md`.
- **Component 2 (snapshot id dtype) carries NO task.** It already shipped in 4.79.0 (ADR-057/058); verified during planning (pandas 3.0.3: `player_id` dtype `Int64`, zero FutureWarnings). The plan does not touch `_snapshot.py` dtype handling.
- **Lint at CI scope** (`python -m ruff check silly_kicks/ tests/ scripts/`, `python -m ruff format --check ...`), `pyright` bare, both via `python -m` on `.venv`. Full suite: `python -m pytest tests/ -m "not e2e" -v --tb=short`.

---

### Task 1: Keeper-identity module skeleton — constants, types, report

**Files:**
- Create: `silly_kicks/tracking/_keeper_identity.py`
- Test: `tests/tracking/test_keeper_identity.py`

**Interfaces:**
- Produces (later tasks rely on these exact names):
  - `KEEPER_ID_SOURCE_EVENT = "event"`, `KEEPER_ID_SOURCE_ROSTER = "roster"`, `KEEPER_ID_SOURCE_NATIVE = "native"`, `KEEPER_ID_SOURCE_DERIVED = "derived"`, `KEEPER_ID_SOURCE_UNRESOLVED = "unresolved"`.
  - `KEEPER_ID_SOURCE_VALUES: tuple[str, ...]` — the five in the order above.
  - `class KeeperIdentity(NamedTuple)`: `gk_id: object`, `source: str`, `conflict: bool`.
  - `KeeperIdentityMap = dict[tuple[object, object, object], KeeperIdentity]` (keys `(canonical game_id, period_id, canonical team_id)`).
  - `@dataclasses.dataclass(frozen=True) class KeeperIdentityReport`: `n_teams_in: int`, `n_resolved: int`, `n_unresolved: int`, `n_conflict: int`, `source_counts: dict[str, int]`.
  - `def resolve_keeper_identities(actions, frames, *, identity, roster=None) -> tuple[KeeperIdentityMap, KeeperIdentityReport]` — stub raising `NotImplementedError` in this task.

- [ ] **Step 1: Write the failing test** — `tests/tracking/test_keeper_identity.py`

```python
from __future__ import annotations

import dataclasses

import pandas as pd
import pytest

from silly_kicks.tracking._keeper_identity import (
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_NATIVE,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_UNRESOLVED,
    KEEPER_ID_SOURCE_VALUES,
    KeeperIdentity,
    KeeperIdentityReport,
    resolve_keeper_identities,
)


def test_source_vocabulary_is_exactly_the_five_values():
    assert KEEPER_ID_SOURCE_VALUES == (
        KEEPER_ID_SOURCE_EVENT,
        KEEPER_ID_SOURCE_ROSTER,
        KEEPER_ID_SOURCE_NATIVE,
        KEEPER_ID_SOURCE_DERIVED,
        KEEPER_ID_SOURCE_UNRESOLVED,
    )
    assert set(KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}


def test_keeper_identity_is_a_three_field_named_tuple():
    ki = KeeperIdentity(gk_id=7, source=KEEPER_ID_SOURCE_ROSTER, conflict=False)
    assert (ki.gk_id, ki.source, ki.conflict) == (7, "roster", False)


def test_report_is_frozen_and_conserves():
    rep = KeeperIdentityReport(
        n_teams_in=2, n_resolved=2, n_unresolved=0, n_conflict=0, source_counts={"roster": 2}
    )
    assert rep.n_resolved + rep.n_unresolved == rep.n_teams_in
    with pytest.raises(dataclasses.FrozenInstanceError):
        rep.n_resolved = 1  # type: ignore[misc]


def test_resolve_is_stubbed_for_now():
    with pytest.raises(NotImplementedError):
        resolve_keeper_identities(pd.DataFrame(), pd.DataFrame(), identity="roster")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/tracking/test_keeper_identity.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'silly_kicks.tracking._keeper_identity'`.

- [ ] **Step 3: Write minimal implementation** — `silly_kicks/tracking/_keeper_identity.py`

Mirror the `_das.py` provenance idiom (module-level doc-commented constants + a values tuple). Use a `NamedTuple` for the map value and a frozen dataclass for the report.

```python
"""Keeper-identity resolution for tracking frames (ADR-055 single-source).

The tracking GK families (``add_pre_shot_gk_*`` / ``add_xt_gk`` / ``add_ghost_gk``) need the REAL
keeper identity, which SB360 freeze-frames do not carry (rows are numbered). This module is the ONE
resolver. Its ``identity="native"`` path DELEGATES to ``defending_gk_from_frames`` /
``acting_gk_from_frames`` (which already return the keeper ``player_id`` from the frame); only the
``identity="roster"`` path (SB360's injected-roster + goal-kick-event ladder) is new work.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, NamedTuple, TypeAlias

import pandas as pd

#: The keeper identity was named by a goal-kick actor event (the SB360 acting keeper; most
#: authoritative for that team-period, and beats a stale roster after a substitution).
KEEPER_ID_SOURCE_EVENT = "event"
#: The keeper identity came from the injected ``{team_id: gk_id}`` roster (the SB360 defending
#: keeper, whom no event names).
KEEPER_ID_SOURCE_ROSTER = "roster"
#: The keeper identity came from the frame's ``is_goalkeeper`` row carrying a real provider-assigned
#: ``player_id`` (non-SB360 providers), whose ``is_goalkeeper_source`` was ``"native"``.
KEEPER_ID_SOURCE_NATIVE = "native"
#: As ``native``, but the frame's ``is_goalkeeper`` was set by positional derivation
#: (``is_goalkeeper_source == "derived"``).
KEEPER_ID_SOURCE_DERIVED = "derived"
#: No rung named this team's keeper -> the identity is NA, counted (never fabricated).
KEEPER_ID_SOURCE_UNRESOLVED = "unresolved"

#: Closed vocabulary for the ``source`` field of a resolved keeper identity.
KEEPER_ID_SOURCE_VALUES: tuple[str, ...] = (
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_NATIVE,
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_UNRESOLVED,
)


class KeeperIdentity(NamedTuple):
    """One resolved keeper identity for a ``(game, period, team)``.

    ``conflict`` records a roster-vs-event disagreement (both named a keeper and they differed);
    ``source`` still records the WINNING rung per precedence, so the disagreement is a separate,
    durable signal, never a lost warning.
    """

    gk_id: object
    source: str
    conflict: bool


#: ``{(canonical game_id, period_id, canonical team_id) -> KeeperIdentity}``. Keys are canonical
#: (ADR-055 rule 2); look up via ``canonical_id``, never a raw tuple.
KeeperIdentityMap: TypeAlias = dict[tuple[object, object, object], KeeperIdentity]


@dataclasses.dataclass(frozen=True)
class KeeperIdentityReport:
    """Run-level audit of keeper-identity resolution. Conserves: ``n_resolved + n_unresolved ==
    n_teams_in`` (ADR-052).
    """

    n_teams_in: int
    n_resolved: int
    n_unresolved: int
    n_conflict: int
    source_counts: dict


def resolve_keeper_identities(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    identity: Literal["native", "roster"],
    roster: dict | None = None,
) -> tuple[KeeperIdentityMap, KeeperIdentityReport]:
    """Resolve the real keeper identity per ``(game, period, team)``. See module docstring."""
    raise NotImplementedError  # filled in Tasks 2 (roster) + 3 (native)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/tracking/test_keeper_identity.py -v`
Expected: PASS (4 tests).

---

### Task 2: Roster path — `identity="roster"` (the SB360 new work)

**Files:**
- Modify: `silly_kicks/tracking/_keeper_identity.py` (implement the `roster` branch)
- Test: `tests/tracking/test_keeper_identity_roster.py`

**Interfaces:**
- Consumes: `id_compat.canonical_id`, `id_compat.ids_match`, `id_compat.same_id`; the SPADL action-type name for a goal kick (below).
- Behaviour of the `roster` branch:
  1. Determine the match's teams from `frames`' non-ball `team_id` values (or from `actions`). Under `identity="roster"`, if NONE of the frames' non-ball `team_id`s intersect `roster`'s keys → **raise** `ValueError`. **This is a roster-APPLICABILITY guard, not a `{0,1}` detector** (P3): it fires on the synthetic `{0,1}` fallback pair (its primary purpose) but ALSO on a real-team frame set paired with a wrong-match roster or a dtype `id_compat` can't bridge — all of which are "the roster names none of this match's teams". A NEW guard, NOT a mirror of `shape_snapshots` (which silently emits `{0,1}`, `parse.py:297`). Message: `"roster names none of this match's teams: frame teams {...}, roster keys {...} (the synthetic {0,1} fallback is one instance)"`. A passing guard therefore proves the roster APPLIES, not that the frames are non-synthetic.
  2. For every `(game_id, period_id, team_id)` present, seed `roster[team_id]` → `KeeperIdentity(gk_id, "roster", conflict=False)`. A team with no roster entry → `KeeperIdentity(pd.NA, "unresolved", False)`.
  3. Goal-kick event override (event > roster): for every goal-kick action with a non-null `player_id`, that action's acting team's keeper = `player_id`. If it differs (via `id_compat`) from the roster seed for that `(game, period, team)` → `conflict=True`; the winning `source` becomes `"event"`. If a period has two DIFFERENT goal-kick takers for one team (a mid-period sub) → the later-time one wins and `conflict=True`.
  4. Build the `KeeperIdentityReport` (conserving; `source_counts` over `KEEPER_ID_SOURCE_VALUES`).

- **Goal-kick action type:** resolve the exact SPADL type name during implementation — read `silly_kicks/spadl/config.py` (`actiontypes` / `spadlconfig`) for the goal-kick type id and the `type_name` → id mapping. Use the canonical `type_name == "goalkick"` (confirm the exact spelling in `spadlconfig` before coding; do not guess). Gate the event rung on `type_name == <goalkick>` AND `player_id` not null.

- [ ] **Step 1: Write the failing tests** — `tests/tracking/test_keeper_identity_roster.py`

```python
from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._keeper_identity import (
    KEEPER_ID_SOURCE_EVENT,
    KEEPER_ID_SOURCE_ROSTER,
    KEEPER_ID_SOURCE_UNRESOLVED,
    resolve_keeper_identities,
)

# NOTE: replace GOALKICK with the real SPADL type_name for a goal kick, read from spadlconfig.
GOALKICK = "goalkick"


def _actions(*, goalkick_taker=None, goalkick_team=None):
    """One shot by team 10 (defended by team 20's keeper) + optionally a goal kick by `goalkick_team`."""
    rows = [
        {"action_id": 0, "game_id": 1, "period_id": 1, "time_seconds": 5.0,
         "team_id": 10, "player_id": 101, "type_name": "shot"},
    ]
    if goalkick_taker is not None:
        rows.append(
            {"action_id": 1, "game_id": 1, "period_id": 1, "time_seconds": 60.0,
             "team_id": goalkick_team, "player_id": goalkick_taker, "type_name": GOALKICK}
        )
    return pd.DataFrame(rows)


def _frames(team_ids=(10, 20)):
    """Minimal frames carrying the two real team ids on non-ball rows + a ball row."""
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1],
            "period_id": [1, 1, 1],
            "frame_id": [0, 0, 0],
            "team_id": [team_ids[0], team_ids[1], pd.NA],
            "player_id": [1, 2, pd.NA],
            "is_ball": [False, False, True],
            "is_goalkeeper": [True, True, False],
        }
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_defending_keeper_resolves_from_roster():
    m, rep = resolve_keeper_identities(
        _actions(), _frames(), identity="roster", roster={10: 901, 20: 902}
    )
    assert m[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
    assert m[(canonical_id(1), 1, canonical_id(20))].source == KEEPER_ID_SOURCE_ROSTER
    assert rep.n_resolved == 2 and rep.n_unresolved == 0


def test_goalkick_event_overrides_a_wrong_roster_starter():
    # Roster says team 20's keeper is 902, but a goal kick by 999 (team 20) says otherwise -> event wins, conflict.
    m, rep = resolve_keeper_identities(
        _actions(goalkick_taker=999, goalkick_team=20),
        _frames(),
        identity="roster",
        roster={10: 901, 20: 902},
    )
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert entry.gk_id == 999
    assert entry.source == KEEPER_ID_SOURCE_EVENT
    assert entry.conflict is True
    assert rep.n_conflict == 1


def test_unresolved_team_is_NA_and_counted_not_fabricated():
    m, rep = resolve_keeper_identities(
        _actions(), _frames(), identity="roster", roster={10: 901}  # no entry for team 20
    )
    entry = m[(canonical_id(1), 1, canonical_id(20))]
    assert pd.isna(entry.gk_id)
    assert entry.source == KEEPER_ID_SOURCE_UNRESOLVED
    assert rep.n_unresolved == 1


def test_synthetic_0_1_team_pair_raises_under_roster_identity():
    # Frames carry the synthetic {0,1} fallback pair; roster keys (10,20) intersect none of them.
    synthetic = _frames(team_ids=(0, 1))
    with pytest.raises(ValueError, match="synthetic|roster|team"):
        resolve_keeper_identities(
            _actions(), synthetic, identity="roster", roster={10: 901, 20: 902}
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_keeper_identity_roster.py -v`
Expected: FAIL with `NotImplementedError` (all four).

- [ ] **Step 3: Implement the `roster` branch** in `_keeper_identity.py`

Replace the `raise NotImplementedError` with a dispatch on `identity`, and implement the roster branch per the interface above. Key points the implementation MUST honour:
- Build the team set from `frames` non-ball rows via `frames.loc[~frames["is_ball"].astype("boolean").fillna(False), "team_id"].dropna()`, canonicalised.
- Synthetic-fallback guard: `if not any(ids_match(pd.Series(list(roster.keys())), t).any() for t in frame_team_ids): raise ValueError(...)`.
- Seed from roster; override from goal-kick events (group goal kicks per `(game, period, team)`, take the latest `time_seconds`); set `conflict` when roster-seed and event disagree via `same_id`.
- Keys via `canonical_id` for both `game_id` and `team_id`; `period_id` used as-is.
- Assemble `KeeperIdentityReport` with `source_counts` initialised to `{v: 0 for v in KEEPER_ID_SOURCE_VALUES}`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/tracking/test_keeper_identity_roster.py -v`
Expected: PASS (4 tests).

---

### Task 3: Native path — `identity="native"` (delegates to TF-13; single-source)

**Files:**
- Modify: `silly_kicks/tracking/_keeper_identity.py` (implement the `native` branch)
- Modify: `silly_kicks/tracking/__init__.py` (export the resolver + constants + types)
- Modify: `silly_kicks/tracking/features.py` (re-export `resolve_keeper_identities` if `__init__` sources GK-resolvers via `.features`, matching `defending_gk_from_frames`)
- Test: `tests/tracking/test_keeper_identity_native.py`

**Interfaces:**
- Consumes: `defending_gk_from_frames`, `acting_gk_from_frames` (both `silly_kicks/tracking/_gk_resolve.py`, unchanged). Both return a `pd.Series` aligned to `actions.index` of the keeper `player_id` (opposing / acting team respectively).
- Behaviour of the `native` branch:
  1. Call `defending_gk_from_frames(actions, frames)` and `acting_gk_from_frames(actions, frames)` — **delegate; do not reimplement** (ADR-055).
  2. For each action, `defending_gk_from_frames` gives the OPPONENT team's keeper id, and `acting_gk_from_frames` gives the ACTING team's keeper id. Emit `(game, period, opponent_team) → def_keeper` and `(game, period, acting_team) → act_keeper` candidate entries. The opponent team is the match's other team (2-team match; derive per action).
  3. Reduce candidates per `(game, period, team)`: take the consensus keeper id (mode); if a period holds two distinct keeper ids for one team → later-time / modal wins, `conflict=True`.
  4. `source`: join the resolved keeper `player_id` back to the frame's `is_goalkeeper_source` for that `(game, team)` → `"native"` or `"derived"`. A team with no keeper in any linked frame → `"unresolved"`, NA, counted.
- Export in `tracking/__init__.py`: add `resolve_keeper_identities`, `KeeperIdentity`, `KeeperIdentityMap`, `KeeperIdentityReport`, and the five `KEEPER_ID_SOURCE_*` constants + `KEEPER_ID_SOURCE_VALUES` to `__all__` (alphabetical) and the import block (follow the `_das.py` re-export pattern; if `__init__` sources GK resolvers through `.features`, add the re-export there too).

- [ ] **Step 1: Write the failing tests** — `tests/tracking/test_keeper_identity_native.py`

```python
from __future__ import annotations

from unittest import mock

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._keeper_identity import (
    KEEPER_ID_SOURCE_DERIVED,
    KEEPER_ID_SOURCE_NATIVE,
    resolve_keeper_identities,
)


def _actions():
    return pd.DataFrame(
        {
            "action_id": [0, 1],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [5.0, 6.0],
            "team_id": [10, 20],
            "player_id": [101, 201],
            "type_name": ["pass", "pass"],
        }
    )


def _frames(gk_source_team20="native"):
    # Two teams, each with a keeper carrying a REAL player_id (native-provider shape).
    return pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1],
            "frame_id": [0, 0, 0, 1, 1],
            "time_seconds": [5.0, 5.0, 5.0, 6.0, 6.0],
            "team_id": [10, 20, pd.NA, 10, 20],
            "player_id": [910, 920, pd.NA, 910, 920],
            "is_ball": [False, False, True, False, False],
            "is_goalkeeper": [True, True, False, True, True],
            "is_goalkeeper_source": ["native", gk_source_team20, "native", "native", gk_source_team20],
        }
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_native_path_resolves_keeper_ids_from_the_frame():
    m, rep = resolve_keeper_identities(_actions(), _frames(), identity="native")
    assert m[(canonical_id(1), 1, canonical_id(10))].gk_id == 910
    assert m[(canonical_id(1), 1, canonical_id(20))].gk_id == 920
    assert m[(canonical_id(1), 1, canonical_id(10))].source == KEEPER_ID_SOURCE_NATIVE
    assert rep.n_resolved == 2


def test_native_path_source_reflects_is_goalkeeper_source():
    m, _ = resolve_keeper_identities(_actions(), _frames(gk_source_team20="derived"), identity="native")
    assert m[(canonical_id(1), 1, canonical_id(20))].source == KEEPER_ID_SOURCE_DERIVED


def test_native_path_delegates_and_does_not_reimplement():
    """Single-source (ADR-055): the native path CALLS the TF-13 resolvers."""
    real_def = T.defending_gk_from_frames
    real_act = T.acting_gk_from_frames
    with (
        mock.patch("silly_kicks.tracking._keeper_identity.defending_gk_from_frames", wraps=real_def) as md,
        mock.patch("silly_kicks.tracking._keeper_identity.acting_gk_from_frames", wraps=real_act) as ma,
    ):
        resolve_keeper_identities(_actions(), _frames(), identity="native")
    assert md.called, "native path must delegate to defending_gk_from_frames, not reimplement it"
    assert ma.called, "native path must delegate to acting_gk_from_frames, not reimplement it"


def test_resolver_is_exported_from_tracking():
    assert hasattr(T, "resolve_keeper_identities")
    assert T.resolve_keeper_identities is resolve_keeper_identities
    assert set(T.KEEPER_ID_SOURCE_VALUES) == {"event", "roster", "native", "derived", "unresolved"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/tracking/test_keeper_identity_native.py -v`
Expected: FAIL — `NotImplementedError` (native branch) and `AttributeError` (not yet exported).

- [ ] **Step 3: Implement the `native` branch + wire the imports/exports**

- In `_keeper_identity.py`: `from ._gk_resolve import acting_gk_from_frames, defending_gk_from_frames` at module top (import by name so the test's `mock.patch` on the module attribute works). Implement the reduction per the interface.
- In `tracking/__init__.py`: add the exports (alphabetical in `__all__`, and an import line following the `_das` pattern; if GK resolvers are re-exported via `.features`, mirror that).
- In `tracking/features.py`: if it is the re-export hub for GK resolvers, add `resolve_keeper_identities` (and the constants/types) to its imports + `__all__` so `__init__`'s `from .features import (...)` picks them up. Read `features.py`'s existing `from ._gk_resolve import (...)` block and mirror it.

- [ ] **Step 4: Run tests + the full keeper-identity suite + lint**

Run: `python -m pytest tests/tracking/test_keeper_identity.py tests/tracking/test_keeper_identity_roster.py tests/tracking/test_keeper_identity_native.py -v`
Expected: PASS (all).
Run: `python -m ruff check silly_kicks/tracking/_keeper_identity.py silly_kicks/tracking/__init__.py && python -m pyright silly_kicks/tracking/_keeper_identity.py`
Expected: clean.

---

### Task 4: Placement helpers (actions + frame bridge), purity + ADR-019 guards, driver roster-map helper

**Files:**
- Modify: `silly_kicks/tracking/_keeper_identity.py` (add `add_defending_gk_player_id`, `apply_keeper_identities_to_frames`)
- Modify: `silly_kicks/tracking/__init__.py` + `features.py` (export the two helpers, following Task 3's export path)
- Create: `scripts/_sb_roster.py` (driver-side roster-map helper)
- Test: `tests/tracking/test_keeper_placement_helpers.py` (the two helpers, incl. the R1 bridge)
- Test: `tests/tracking/test_keeper_identity_contracts.py` (purity + id-dtype)
- Test: `tests/scripts/test_sb_roster.py` (helper)

**Interfaces:**
- Produces (the two single-sourced PLACEMENT helpers — the resolver returns a pure map; these apply it, and both the producer (Task 6) AND the gkdv driver (A3) use them so the bridge is single-sourced):
  - `add_defending_gk_player_id(actions: pd.DataFrame, keeper_map: KeeperIdentityMap) -> pd.DataFrame` — returns a COPY of `actions` with `defending_gk_player_id` stamped per action (opponent lookup: the 2-team match's other team, looked up in the map via `canonical_id`). NA where unresolvable. Pure.
  - `apply_keeper_identities_to_frames(frames: pd.DataFrame, keeper_map: KeeperIdentityMap) -> pd.DataFrame` — returns a COPY of `frames` with each non-ball `is_goalkeeper` row's `player_id` set to `keeper_map[(canonical(game), period, canonical(team))].gk_id` (left unchanged where the map has no entry or `gk_id` is NA). **This is the R1 identity→frame bridge.** Pure. Callers apply it only where the frame ids are NOT already real (the roster/SB360 path) — see Task 6 §1.
- Produces: `scripts/_sb_roster.py::build_gk_roster_map(roster: dict[int, dict]) -> dict[object, object]` — turns `parse_roster(...)` output (`{player_id: {name, jersey, team, position}}`) into `{team_id: gk_id}` by filtering `position == "Goalkeeper"` keyed by `team`. Driver-side (ADR-054: library never parses raw JSON). If a team has >1 goalkeeper in the roster (a named substitute), keep the first and record nothing extra — the goal-kick-event rung resolves subs at resolution time; document this.

- [ ] **Step 1: Write the failing tests**

`tests/tracking/test_keeper_placement_helpers.py` (the two helpers + the R1 bridge non-vacuity):

```python
from __future__ import annotations

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.tracking import add_defending_gk_player_id, apply_keeper_identities_to_frames


def _sb360_fixture():
    """A shot by team 10 with team 20's keeper in the freeze-frame; synthetic numbered frame ids."""
    actions = pd.DataFrame(
        {"action_id": [0], "game_id": [1], "period_id": [1], "time_seconds": [5.0],
         "team_id": [10], "player_id": [101], "type_name": ["shot"], "start_x": [90.0], "start_y": [34.0]}
    )
    snapshots = pd.DataFrame(
        {"action_id": [0, 0, 0], "team_id": [10, 10, 20], "x": [90.0, 80.0, 104.0],
         "y": [34.0, 40.0, 34.0], "is_goalkeeper": [False, False, True]}
    )
    frames, _ = T.snapshot_to_tracking_frames(snapshots, actions)
    return actions, frames


def test_add_defending_gk_player_id_stamps_opponent_keeper_and_is_pure():
    actions, frames = _sb360_fixture()
    m, _ = T.resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    snap = actions.copy(deep=True)
    out = add_defending_gk_player_id(actions, m)
    pd.testing.assert_frame_equal(actions, snap)  # pure
    # the shot is by team 10 -> defending keeper is team 20's (902)
    assert out["defending_gk_player_id"].iloc[0] == 902


def test_frame_bridge_stamps_real_id_onto_the_synthetic_keeper_row_and_is_pure():
    actions, frames = _sb360_fixture()
    m, _ = T.resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    snap = frames.copy(deep=True)
    bridged = apply_keeper_identities_to_frames(frames, m)
    pd.testing.assert_frame_equal(frames, snap)  # pure -- caller's frames untouched
    krow = bridged[(bridged["team_id"] == 20) & bridged["is_goalkeeper"].astype("boolean").fillna(False)]
    assert (krow["player_id"] == 902).all(), "the synthetic keeper-row id must be bridged to the roster id"


def test_bridge_unlocks_pre_shot_gk_position_the_R1_deliverable():
    """The whole point: without the bridge, add_pre_shot_gk_position is NaN on SB360 (frame ids are
    synthetic). With the bridge (real keeper id on the frame row + on the action), it produces a real
    position."""
    actions, frames = _sb360_fixture()
    m, _ = T.resolve_keeper_identities(actions, frames, identity="roster", roster={10: 901, 20: 902})
    stamped_actions = add_defending_gk_player_id(actions, m)

    # WITHOUT the bridge: the synthetic keeper id (a small int) != 902 -> NaN.
    unbridged = T.add_pre_shot_gk_position(stamped_actions, frames)
    assert unbridged["pre_shot_gk_x"].isna().all(), "control: unbridged SB360 frames yield NaN GK position"

    # WITH the bridge: the keeper row now carries 902, matching the action stamp -> real position.
    bridged = T.add_pre_shot_gk_position(stamped_actions, apply_keeper_identities_to_frames(frames, m))
    assert bridged["pre_shot_gk_x"].notna().any(), "bridged frames must yield a REAL GK position (R1)"
```

`tests/tracking/test_keeper_identity_contracts.py`:

```python
from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id
from silly_kicks.tracking._keeper_identity import resolve_keeper_identities


def _actions():
    return pd.DataFrame(
        {"action_id": [0], "game_id": [1], "period_id": [1], "time_seconds": [5.0],
         "team_id": [10], "player_id": [101], "type_name": ["shot"]}
    )


def _frames():
    return pd.DataFrame(
        {"game_id": [1, 1, 1], "period_id": [1, 1, 1], "frame_id": [0, 0, 0],
         "team_id": [10, 20, pd.NA], "player_id": [1, 2, pd.NA],
         "is_ball": [False, False, True], "is_goalkeeper": [True, True, False]}
    ).astype({"team_id": "Int64", "player_id": "Int64"})


def test_resolver_does_not_mutate_its_inputs():
    a, f = _actions(), _frames()
    a_snap, f_snap = a.copy(deep=True), f.copy(deep=True)
    resolve_keeper_identities(a, f, identity="roster", roster={10: 901, 20: 902})
    pd.testing.assert_frame_equal(a, a_snap)
    pd.testing.assert_frame_equal(f, f_snap)


def test_roster_keys_match_across_id_dtypes_via_id_compat():
    # Frames carry Int64 team ids; roster keys are python ints AND strings -- both must resolve (ADR-019).
    m_int, _ = resolve_keeper_identities(_actions(), _frames(), identity="roster", roster={10: 901, 20: 902})
    m_str, _ = resolve_keeper_identities(_actions(), _frames(), identity="roster", roster={"10": 901, "20": 902})
    assert m_int[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
    assert m_str[(canonical_id(1), 1, canonical_id(20))].gk_id == 902
```

`tests/scripts/test_sb_roster.py`:

```python
from __future__ import annotations

from scripts._sb_roster import build_gk_roster_map


def test_build_gk_roster_map_filters_goalkeepers_keyed_by_team():
    roster = {
        901: {"name": "A", "jersey": 1, "team": 10, "position": "Goalkeeper"},
        102: {"name": "B", "jersey": 9, "team": 10, "position": "Center Forward"},
        902: {"name": "C", "jersey": 1, "team": 20, "position": "Goalkeeper"},
    }
    assert build_gk_roster_map(roster) == {10: 901, 20: 902}
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_keeper_placement_helpers.py tests/tracking/test_keeper_identity_contracts.py tests/scripts/test_sb_roster.py -v`
Expected: FAIL (`ImportError` for the two helpers; `ModuleNotFoundError: scripts._sb_roster`; contract tests may already pass if Task 2/3 built purely — if they fail, fix the resolver to copy rather than mutate).

- [ ] **Step 3: Implement**

- In `_keeper_identity.py`: `add_defending_gk_player_id` (opponent lookup per action via `canonical_id`, returns a copy) and `apply_keeper_identities_to_frames` (stamp non-ball `is_goalkeeper` rows' `player_id` from the map per `(game, period, team)` via `canonical_id`, returns a copy). Both PURE. Export both from `tracking/__init__.py` (+ `features.py` re-export) following Task 3's path.
- `scripts/_sb_roster.py`: pure function building `{team_id: gk_id}`; no I/O, no `statsbombpy`.
- If `test_resolver_does_not_mutate_its_inputs` fails, ensure the resolver never assigns into `actions`/`frames` (operate on local copies / derived Series only).
- If `test_roster_keys_match_across_id_dtypes_via_id_compat` (or the helper id lookups) fail, route every roster-key / frame-team / map-key comparison through `ids_match` / `canonical_id` (never raw `==` or `astype(str)`).

- [ ] **Step 4: Run to verify pass** (incl. the R1 bridge control-and-treatment)

Run: `python -m pytest tests/tracking/test_keeper_placement_helpers.py tests/tracking/test_keeper_identity_contracts.py tests/scripts/test_sb_roster.py -v`
Expected: PASS — including `test_bridge_unlocks_pre_shot_gk_position_the_R1_deliverable` (unbridged→NaN control, bridged→real position), which is the load-bearing proof that the identity→frame bridge unlocks the SB360 GK feature.

---

### Task 5: Canonical call convention — the gate (red-first) + `add_pre_shot_gk_angle` fix

**Files:**
- Create: `tests/tracking/test_call_convention_registry.py`
- Modify: `silly_kicks/tracking/features.py` (`add_pre_shot_gk_angle` signature at `:846`)
- Modify: any internal caller of `add_pre_shot_gk_angle` that passes `frames` by keyword only if a positional form is now clearer (optional; keyword calls still work).

**Interfaces:**
- The gate derives the frame-consuming `add_*` surface from `tracking.__all__` (names starting with `add_` that take a `frames` parameter, via `inspect.signature`) and asserts, for each, the two canonical rules:
  1. `frames` is NOT keyword-only (it appears as `POSITIONAL_OR_KEYWORD`).
  2. Every parameter after the first fitted-model/`frames` block that is OPTIONAL is `KEYWORD_ONLY`.
- Exemptions in a `_CALL_SHAPE_EXEMPT: dict[str, str]` (name → reason): `add_sync_score` (link-consumer, no `frames`), `add_visible_area_coverage` (no `frames`, requires `visible_area`), `add_gradientsports_player_ids` (jersey helper over different inputs). Each exemption carries a non-empty reason (a `test_exemptions_carry_a_reason`).
- Two meta-assertions pin the registry to the public surface (surface − (conforming ∪ exempt) is empty; exempt ⊆ surface). Model these on `tests/tracking/test_mirror_registry.py::test_every_public_add_is_registered` / `test_registry_has_no_stale_entries`.
- **P5 (confirm at red-first):** the five positional-`xt` aggregators (`add_gk_influence`, `add_cover_shadows`, `add_off_ball_run_values`, `add_player_influence`, `add_xt_gk`) keep `xt` as a REQUIRED positional (NO default) → the `optional-params-after-frames` gate does not flag them (it only flags positional+OPTIONAL). This is correct-by-construction, but the red-first run is the check: when the gate first runs, confirm the ONLY offender is `add_pre_shot_gk_angle` (keyword-only `frames`). If any of the five is flagged, it has an *optional* positional after `frames` that must be reconsidered — do not silently exempt it.

- [ ] **Step 1: Write the failing gate** — `tests/tracking/test_call_convention_registry.py`

```python
from __future__ import annotations

import inspect

import silly_kicks.tracking as T

#: Frame-consuming add_* whose signature legitimately does not fit the canonical (actions, frames, ...) shape.
_CALL_SHAPE_EXEMPT = {
    "add_sync_score": "link-consumer family (TF-6): add_sync_score(actions, links, *, ...), no frames",
    "add_visible_area_coverage": "takes no frames and REQUIRES visible_area",
    "add_gradientsports_player_ids": "jersey/roster helper over different inputs, returns frames",
}


def _public_add_names() -> set[str]:
    return {n for n in T.__all__ if n.startswith("add_")}


def _takes_frames(fn) -> bool:
    return "frames" in inspect.signature(fn).parameters


def _frame_consumers() -> dict[str, inspect.Signature]:
    out = {}
    for name in _public_add_names():
        fn = getattr(T, name)
        sig = inspect.signature(fn)
        if "frames" in sig.parameters:
            out[name] = sig
    return out


def test_frames_is_never_keyword_only():
    offenders = {}
    for name, sig in _frame_consumers().items():
        if name in _CALL_SHAPE_EXEMPT:
            continue
        p = sig.parameters["frames"]
        if p.kind == inspect.Parameter.KEYWORD_ONLY:
            offenders[name] = str(sig)
    assert not offenders, f"frames must be positional-or-keyword (canonical shape): {offenders}"


def test_optional_params_after_frames_are_keyword_only():
    offenders = {}
    for name, sig in _frame_consumers().items():
        if name in _CALL_SHAPE_EXEMPT:
            continue
        params = list(sig.parameters.values())
        # allow: actions, frames, and a single required positional model (e.g. xt)
        for p in params[2:]:
            positional = p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            optional = p.default is not inspect.Parameter.empty
            if positional and optional:
                offenders.setdefault(name, []).append(p.name)
    assert not offenders, f"optional params after frames must be keyword-only: {offenders}"


def test_exemptions_carry_a_reason():
    for name, reason in _CALL_SHAPE_EXEMPT.items():
        assert reason.strip(), f"{name} exemption needs a reason"


def test_exemptions_are_all_real_exports():
    stale = set(_CALL_SHAPE_EXEMPT) - _public_add_names()
    assert not stale, f"exemption names a non-exported add_*: {sorted(stale)}"
```

- [ ] **Step 2: Run to verify it fails RED on `add_pre_shot_gk_angle`**

Run: `python -m pytest tests/tracking/test_call_convention_registry.py::test_frames_is_never_keyword_only -v`
Expected: FAIL — offenders includes `add_pre_shot_gk_angle` (its `frames` is currently `KEYWORD_ONLY`).

- [ ] **Step 3: Fix `add_pre_shot_gk_angle`** — `features.py:846`

Change the signature from keyword-only `frames`:
```python
def add_pre_shot_gk_angle(actions: pd.DataFrame, *, frames: pd.DataFrame, links: pd.DataFrame | None = None) -> pd.DataFrame:
```
to positional-or-keyword `frames`, matching `add_pre_shot_gk_position` (`:726`):
```python
def add_pre_shot_gk_angle(actions: pd.DataFrame, frames: pd.DataFrame, *, links: pd.DataFrame | None = None) -> pd.DataFrame:
```
Existing keyword callers (`add_pre_shot_gk_angle(actions, frames=frames)`) remain valid — this widens, it does not remove the keyword form. Check the atomic mirror (`silly_kicks/atomic/tracking/features.py`) for a mirrored `add_pre_shot_gk_angle`; if present, apply the identical change and its liveness/mirror tests still pass.

- [ ] **Step 4: Run the gate + the pre-shot-gk tests + full tracking gate suite**

Run: `python -m pytest tests/tracking/test_call_convention_registry.py -v`
Expected: PASS (4).
Run: `python -m pytest tests/tracking/ -k "pre_shot_gk or mirror_registry or aggregator_column_liveness" -m "not e2e" -v`
Expected: PASS (no regression from the signature widening).

---

### Task 6: The producer — `run_tracking_features` + Component-1 threading + trap-2 non-vacuity

**Files:**
- Create: `silly_kicks/tracking/_run_features.py`
- Modify: `silly_kicks/tracking/__init__.py` (export `run_tracking_features`, `TrackingFeaturesReport`)
- Test: `tests/tracking/test_run_tracking_features.py`

**Interfaces:**
- Produces:
  - `def run_tracking_features(actions, frames, *, links=None, xt=None, xg_column=None, roster=None, identity="native", visible_area=None, home_team_id=None, families=None, pitch_control_cache=None) -> tuple[pd.DataFrame, TrackingFeaturesReport]`.
  - `@dataclasses.dataclass(frozen=True) class TrackingFeaturesReport`: `n_families_in: int`, `n_families_run: int`, `n_families_skipped: int`, `family_status: dict[str, str]` (name → `"ran"` / `"skipped: <reason>"`), `keeper_report: KeeperIdentityReport | None`. Conserves: `n_families_run + n_families_skipped == n_families_in`.
- Behaviour:
  1. **Resolve keeper identity first, then bridge it onto BOTH grains** (Component 1; R1 — the load-bearing step). Call `resolve_keeper_identities(actions, frames, identity=identity, roster=roster)`. Then apply the map with the two single-sourced placement helpers (Task 4):
     - `actions2 = add_defending_gk_player_id(actions, m)` — stamps `defending_gk_player_id` per action (opponent lookup). Required by `add_pre_shot_gk_*`.
     - **`frames2 = apply_keeper_identities_to_frames(frames, m)` — ONLY on the roster path (`identity == "roster"`).** This is the **identity→frame bridge** WITHOUT which the cycle's headline feature is NaN: `add_pre_shot_gk_position` locates the keeper by matching `frame.player_id == defending_gk_player_id` (`utils.py:1034`), but SB360 frames carry **synthetic numbered** `player_id`s (`_snapshot.py:111`), so the roster id (`902`) matches no frame row → `pre_shot_gk_* = NaN`. The bridge stamps the resolved id onto the frames' `is_goalkeeper` rows so the match succeeds. On the **native path it is NOT applied** — native frames already carry real keeper ids, and stamping the per-`(game,period,team)` consensus would clobber a mid-period sub's correct per-frame id.
     - Run the GK families on `(actions2, frames2)`. Both helpers return COPIES; the caller's `actions`/`frames` are never mutated (the resolver stays pure — F1). `add_ghost_gk` is unaffected by R1 (it finds the keeper by `is_goalkeeper`, `_ghost_gk.py:2534`, not by identity match) but the bridge is harmless to it.
  2. **Pre-link once** (`link_actions_to_frames(actions, frames)` → `links` if not supplied) and build/share one `PitchControlCache` if not supplied; thread `links=` and `pitch_control_cache=` into every family that accepts them.
  3. **Dispatch = model-injection routing + family classification** (F3): a family needing `xt` gets the injected `xt`; `add_defensive_credit` gets `xg_column`; the link-consumer (`add_sync_score`) follows its own path. Where a required model is absent, that family is SKIPPED with `"skipped: <model> not supplied"` and its columns are simply not added (honest absence — the producer does not fabricate).
  4. **Per-family guard — the producer never crashes (P1).** Each family runs under `try/except Exception` (the `run_add_star_battery` precedent): a family that RAISES on the given frames is caught and recorded `family_status[name] = "skipped: <ExcType>: <msg>"`, adding NO columns. This is the accurate contract — the producer does NOT promise every family self-degrades. **Two behaviours coexist and both are correct on velocity-less SB360 frames:** (a) a family that self-degrades emits its NaN value columns WITH a provenance token — `add_das` catches `DasUnscoreableError` internally and returns `das_team`/`das_opponent`/`das_diff` = NaN with `das_source == "unscoreable_frame"` (`features.py:3236-3248`); the ADR-063 four (`add_gk_influence`/`add_cover_shadows`/`add_player_influence`/`add_space_creation`) lift Tier-1 and NaN Tier-2 via `zero_velocity_if_unavailable`; (b) a family that would raise is caught → skipped. Naming the keeper does NOT make any velocity-constitutive metric score (ADR-063) — it stays NaN whether the family self-degrades (a) or is skipped (b).
  5. **`families`** selects all (default) or a named subset.
  6. Returns `(enriched_actions, report)`.
- Single-source note: the per-aggregator routing table this producer owns is what Task 7 re-points `scripts/_sb_battery.py` at. Design the routing as a small, importable structure (e.g. a `FAMILY_MODEL_REQUIREMENTS` dict) that Task 7 can reuse.
- **Scope note (do NOT expand):** the native-path `defending_gk_player_id` stamp uses the resolver MAP, NOT `spadl.utils.add_pre_shot_gk_context`. Reconciling the two native paths (they can differ on the event-primary keeper / at a mid-period sub — see the spec's "Out of scope / follow-ups") is a documented follow-up, not this cycle. Do not wire `add_pre_shot_gk_context` into the producer.

- [ ] **Step 1: Write the failing tests** — `tests/tracking/test_run_tracking_features.py`

**One CONCRETE SB360 fixture serves every test (P2 — no `...` stubs).** `add_pre_shot_gk_position` is
position-only, so a velocity-less SB360 fixture (built by the known-working `snapshot_to_tracking_frames`)
exercises the composition equivalence AND the trap-2 non-vacuity — no invented velocity-bearing fixture
is needed. `ROSTER = {10: 901, 20: 902}` maps the two real match teams to their keeper ids.

```python
from __future__ import annotations

import pandas as pd

import silly_kicks.tracking as T
from silly_kicks.tracking import (
    add_defending_gk_player_id,
    apply_keeper_identities_to_frames,
    run_tracking_features,
)

ROSTER = {10: 901, 20: 902}


def _sb360_fixture():
    """A shot by team 10 with a freeze-frame carrying BOTH teams' keepers (team 20 = the defending
    keeper the shot needs). ``snapshot_to_tracking_frames`` numbers the rows and stamps
    ``speed_source == 'unavailable'`` (velocity-less by construction)."""
    actions = pd.DataFrame(
        {
            "action_id": [0],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [5.0],
            "team_id": [10],
            "player_id": [101],
            "type_name": ["shot"],
            "start_x": [90.0],
            "start_y": [34.0],
        }
    )
    # Freeze-frame players: shooter (team 10), a team-10 field player, and team 20's keeper near its goal.
    snapshots = pd.DataFrame(
        {
            "action_id": [0, 0, 0],
            "team_id": [10, 10, 20],
            "x": [90.0, 80.0, 104.0],
            "y": [34.0, 40.0, 34.0],
            "is_goalkeeper": [False, False, True],
        }
    )
    frames, _links = T.snapshot_to_tracking_frames(snapshots, actions)
    return actions, frames


def test_producer_equals_composing_the_add_star_calls_after_the_same_resolution():
    actions, frames = _sb360_fixture()
    # Baseline: resolve identity -> stamp defending_gk_player_id on ACTIONS -> BRIDGE identity onto the
    # FRAME keeper rows -> add_pre_shot_gk_position. F6 + R1: the baseline INCLUDES the resolver step
    # AND the frame bridge; without the frame bridge, add_pre_shot_gk_position finds no keeper row on
    # the synthetically-numbered SB360 frames and returns NaN -- and the equality would pass VACUOUSLY
    # (NaN == NaN). The single-sourced helpers guarantee baseline == producer.
    m, _ = T.resolve_keeper_identities(actions, frames, identity="roster", roster=ROSTER)
    base = T.add_pre_shot_gk_position(
        add_defending_gk_player_id(actions, m),
        apply_keeper_identities_to_frames(frames, m),
    )
    out, _report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER, families=["add_pre_shot_gk_position"]
    )
    added = base.columns.difference(actions.columns)
    # NON-VACUITY (R1): the bridge must produce a REAL keeper position, not NaN==NaN. A shot with the
    # defending keeper in-frame MUST yield a populated pre_shot_gk_x, or the keystone proves nothing.
    assert out["pre_shot_gk_x"].notna().any(), (
        "identity->frame bridge failed: pre_shot_gk_x is all-NaN, so the SB360 GK feature did not "
        "unlock -- the cycle's headline deliverable. (This assertion is what makes the equality below "
        "non-vacuous.)"
    )
    pd.testing.assert_frame_equal(
        out[added].reset_index(drop=True), base[added].reset_index(drop=True)
    )


def test_report_conserves():
    actions, frames = _sb360_fixture()
    _out, report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER,
        families=["add_pre_shot_gk_position", "add_team_shape"],
    )
    assert report.n_families_run + report.n_families_skipped == report.n_families_in


def test_absent_model_skips_the_family_not_fabricates():
    actions, frames = _sb360_fixture()
    _out, report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER, families=["add_xt_gk"]  # xt not supplied
    )
    assert "add_xt_gk" in report.family_status
    assert report.family_status["add_xt_gk"].startswith("skipped")


def test_naming_the_keeper_does_not_make_velocity_metrics_score_on_sb360():
    """Trap 2 non-vacuity (ADR-063): the keeper IS named (non-vacuity — the metric COULD have moved),
    yet DAS stays NaN on velocity-less frames. `add_das` self-degrades (catches DasUnscoreableError):
    it EMITS the `das_*` columns as NaN with `das_source == 'unscoreable_frame'` — it does NOT raise
    or get skipped, so the witness is a real NaN VALUE, not an absent column. Note the column is
    `das_diff`, not `das`."""
    actions, frames = _sb360_fixture()
    out, _report = run_tracking_features(
        actions, frames, identity="roster", roster=ROSTER,
        families=["add_pre_shot_gk_position", "add_das"],
    )
    assert out["defending_gk_player_id"].notna().any(), "the keeper WAS named (non-vacuity)"
    assert out["das_diff"].isna().all(), "DAS stays NaN on velocity-less frames even though the keeper is named"
    assert (out["das_source"] == "unscoreable_frame").all(), (
        "add_das must self-degrade (RAN and honestly NaN'd), not be skipped -- the real non-vacuity"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/tracking/test_run_tracking_features.py -v`
Expected: FAIL — `ImportError: cannot import name 'run_tracking_features'`.

- [ ] **Step 3: Implement `run_tracking_features`** in `_run_features.py`; export it + `TrackingFeaturesReport` from `__init__.py`.

Follow `scripts/_sb_battery.py::run_add_star_battery` as the structural precedent (per-family try/skip loop), but: resolve keeper identity first and apply BOTH placement helpers (`add_defending_gk_player_id` always; `apply_keeper_identities_to_frames` only when `identity == "roster"` — the R1 frame bridge), pre-link + shared cache, inject models per the routing table, and return the enriched actions + report. Never mutate inputs (the helpers return copies).

- [ ] **Step 4: Run to verify pass + the pandas-3 leg**

Run: `python -m pytest tests/tracking/test_run_tracking_features.py -v`
Expected: PASS.
Run (pandas-3): `.venv312/Scripts/python.exe -m pytest tests/tracking/test_run_tracking_features.py -v`
Expected: PASS (verifies no pandas-3 CoW / dtype surprise).

---

### Task 7: Re-point `scripts/_sb_battery.py` at the producer's routing (single-source, audit byte-identical)

**Files:**
- Modify: `scripts/_sb_battery.py` (its model-routing adapters delegate to the library's routing)
- Modify: `tests/sb360/_registry.py` (audit scope note for the two new public functions)
- Test: existing `tests/scripts/test_sb_battery.py`, `tests/sb360/test_registry_surface.py`, and the committed `tests/sb360/_entries/*` round-trip must stay green.

**Interfaces:**
- The model-routing knowledge (which family needs `xt` / `xg_column` / the GK prerequisite) is single-sourced from the library (Task 6's `FAMILY_MODEL_REQUIREMENTS` or equivalent). `scripts/_sb_battery.py`'s `with_xt` / `with_xt_keyword` / `defensive_credit` / `pre_shot_gk_*` adapters reference the library's routing instead of re-encoding it; the audit-synthesized inputs (`audit_xt()`, `audit_xg=0.12`, the fixed `visible_area` polygon) STAY in `_sb_battery` (audit concern, not library concern).
- `tests/sb360/_calls.py` and `_registry._adapters()` stay pointed at `scripts/_sb_battery.py` unchanged (the `tests → scripts` layering invariant pinned by `tests/scripts/test_sb_battery.py` is untouched). The committed `_entries/*` round-trip stays byte-identical (the audit's per-column verdicts do not move — this is an output-preserving refactor).
- Audit scope: neither `run_tracking_features` (an orchestrator; not `add_*`; correctness proven by composition-equivalence in Task 6) nor `resolve_keeper_identities` (returns an identity mapping, not action-grain feature columns) gets an SB360 verdict. `audited_surface()` picks up neither automatically (not `add_*`, not in `BOUNDARY_ENTRY_POINTS`). **Add a one-line SCOPE NOTE** to `tests/sb360/_registry.py::audited_surface`'s docstring recording this deliberate exclusion with its reason (mirroring the existing FOV-companion scope note), so the exclusion is documented, not silent.

- [ ] **Step 1: Write / extend the failing test**

Add to `tests/scripts/test_sb_battery.py` (or a new `tests/scripts/test_sb_battery_single_source.py`) an assertion that `_sb_battery`'s routing derives from the library (import the library routing and assert the adapter set matches), plus a scope-note presence assertion in `tests/sb360/test_registry_surface.py`:

```python
def test_audited_surface_documents_the_new_producer_exclusion():
    from tests.sb360 import _registry
    doc = _registry.audited_surface.__doc__ or ""
    assert "run_tracking_features" in doc and "resolve_keeper_identities" in doc, (
        "audited_surface must document why the producer + resolver are outside the audit"
    )
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/sb360/test_registry_surface.py::test_audited_surface_documents_the_new_producer_exclusion -v`
Expected: FAIL (the docstring does not yet mention them).

- [ ] **Step 3: Implement the re-point + scope note**

- Re-point the model-routing adapters in `_sb_battery.py` at the library routing (single-source); keep the audit-synthesized inputs local. Verify the audit's committed `_entries/*` round-trip is byte-identical.
- Add the scope note to `audited_surface`'s docstring.

- [ ] **Step 4: Run the full SB360 + scripts suites**

Run: `python -m pytest tests/sb360/ tests/scripts/test_sb_battery.py -m "not e2e" -v`
Expected: PASS (byte-identical audit; layering intact; scope note present).

---

### Task 8: Commit-prep — ADR-054 pointer, new ADR, CHANGELOG/TODO/version, C4 prose

**Files:**
- Modify: `docs/superpowers/adrs/ADR-054-sb360-degradation-and-statsbomb-port.md` (the stale `_defending_goal` note, after line 121)
- Create: `docs/superpowers/adrs/ADR-0XX-sb360-first-class-provider.md` (number confirmed at commit)
- Modify: `CHANGELOG.md`, `TODO.md`, `pyproject.toml`, `silly_kicks/__init__.py`, `uv.lock` (via `uv lock`)
- Modify: `docs/c4/architecture.dsl` (prose: add `run_tracking_features` / `resolve_keeper_identities` to the `tracking` container description — NO aggregator-count change)
- Modify: `CLAUDE.md` (a durable-contract bullet for the resolver single-source + the call convention)

**Interfaces:** none (documentation + version).

- [ ] **Step 1: Fold in the ADR-054 pointer** — append after `ADR-054-...md:121`:

```
  `TODO.md` rather than assumed covered. -> RESOLVED by ADR-055 (4.77.0): the fork was deleted and
  replaced by the canonical `GoalMap`; no TODO row ever existed.
```
(Match the surrounding bullet's indentation exactly.)

- [ ] **Step 2: Confirm the next-free numbers against `main`**

Run: `git fetch origin && git log origin/main --oneline -5`
Read the newest `pyproject.toml` version, the newest `PR-Sxxx` in `CHANGELOG.md`, and the newest `ADR-0xx` in `docs/superpowers/adrs/`. Assign the next free version / `PR-Sxxx` / `ADR-0xx`. **Do not hardcode before this step.** (Current tip is 4.99.0; the next MINOR is the likely version, but confirm — another session may have taken it.)

- [ ] **Step 3: Write the new ADR** — `docs/superpowers/adrs/ADR-0XX-sb360-first-class-provider.md`

Use `docs/superpowers/adrs/ADR-TEMPLATE.md`. Record: the single keeper-identity resolver (native delegates to TF-13; roster/event ladder is the SB360 new work); the mapping return shape + `keeper_id_source` vocabulary + conflict diagnostic; the canonical call convention (required model may be 3rd positional — the 108-site reason); `run_tracking_features` as an orchestrator (not a new aggregator); the retrain/Hyrum trigger (SB360 GK features go honest-NaN → values); Component 2 already-shipped note. Cross-reference ADR-053/054/055/037/057/058/062/063 and TF-13.

- [ ] **Step 4: Bump version (5 sites) + CHANGELOG + TODO + C4 prose + CLAUDE.md**

- `pyproject.toml`, `silly_kicks/__init__.py` → the confirmed version; `uv lock` to update `uv.lock` (never hand-edit).
- `CHANGELOG.md`: a `## [x.y.z] — YYYY-MM-DD` entry, bold theme, `(PR-Sxxx, ADR-0xx)` in the first sentence, bullet detail (keeper-identity resolver, call convention, producer; note Component 2 was already 4.79.0).
- `TODO.md`: replace `**Current (unreleased branch)**` with this release; remove completed items.
- `docs/c4/architecture.dsl`: extend the `tracking` container description prose to mention the producer + resolver (do NOT change the "33 action-coupled aggregators" count — no new aggregator). Regenerate `architecture.html` via the `mad-scientist-skills:c4` skill if the DSL changed.
- `CLAUDE.md`: add a durable-contract bullet — "ONE keeper-identity resolver (`resolve_keeper_identities`); native path DELEGATES to `*_gk_from_frames`, roster/event ladder is the SB360 new work" + "canonical `add_*` call shape (frames never keyword-only; required fitted model may be 3rd positional)".

- [ ] **Step 5: Full suite + lint at CI scope**

Run: `python -m pytest tests/ -m "not e2e" -v --tb=short`
Run: `python -m ruff check silly_kicks/ tests/ scripts/ && python -m ruff format --check silly_kicks/ tests/ scripts/`
Run: `python -m pyright`
Run (pandas-3 spot-check): `.venv312/Scripts/python.exe -m pytest tests/tracking/test_keeper_identity_roster.py tests/tracking/test_run_tracking_features.py -v`
Expected: all green. **STOP** — the single commit is the user's, on their explicit approval (do NOT commit).

---

## Self-Review

**Spec coverage:**
- Component 1 (keeper identity) → Tasks 1–4. ✓ (constants/types/report, roster path + traps, native delegation + exports, the two placement helpers + the R1 frame bridge, purity/id-dtype/driver-helper).
- Component 2 (dtype) → **no task by design** (already shipped 4.79.0; verified during planning). Documented in the plan header + spec. ✓
- Component 3 (call convention) → Task 5 (gate red-first + `add_pre_shot_gk_angle` fix; required-model-positional refinement, 108-site reason). ✓
- Component 4 (producer) → Task 6 (producer + threading + report + trap-2) and Task 7 (audit re-point single-source). ✓
- Component 5 (ADR-054 pointer) → Task 8 Step 1 (folded into the single commit). ✓
- Review findings (round 1): A1/A2 (single-source coordination + ADR-037 driver-side) — Global Constraints + Task 3 delegation. A3/F1 (mapping return) — Global Constraints + Task 1. F2 (conflict diagnostic) — Task 1 report + Task 2. F6 (composition baseline includes resolver + bridge) — Task 6 Step 1. F7 (keystone red-first) — Tasks 1–3 all TDD-red-first. N1 (native delegation) — Task 3, asserted by `test_native_path_delegates_and_does_not_reimplement`. N2 (tuple returns) — Task 1 + Task 6 signatures.
- Plan-review findings (round 1): P1 (add_das self-degrades — trap-2 uses `das_diff` + `das_source`) Task 6. P2 (concrete fixture) Task 6. P3 (roster-applicability guard) Task 2. P4 (TypeAlias) Task 1. P5 (positional-`xt` red-first confirm) Task 5. P6 (spec goalkick wording) spec.
- Plan-review finding (round 2) **R1 (identity→frame bridge)** — the load-bearing one: two placement helpers (Task 4), the producer applies the frame bridge on the roster path (Task 6 §1), and the keystone carries a `pre_shot_gk_x.notna().any()` non-vacuity assertion PLUS a dedicated control-and-treatment test (`test_bridge_unlocks_pre_shot_gk_position_the_R1_deliverable`, Task 4). ✓

**Placeholder scan:** Task 6's fixture (`_sb360_fixture`) and the composition baseline (now via the public `add_defending_gk_player_id` + `apply_keeper_identities_to_frames` helpers) are CONCRETE (P2 + R1 — no `...` stubs; one SB360 fixture via the known-working `snapshot_to_tracking_frames` serves all producer tests). The `GOALKICK` type name in Task 2 is flagged to confirm against `spadlconfig` (do not guess) — a verify-don't-guess note, not a logic gap. No TBD/TODO.

**Type consistency:** `KeeperIdentity(gk_id, source, conflict)`, `KeeperIdentityMap`, `KeeperIdentityReport`, `resolve_keeper_identities(...) -> tuple[KeeperIdentityMap, KeeperIdentityReport]`, `run_tracking_features(...) -> tuple[pd.DataFrame, TrackingFeaturesReport]`, `KEEPER_ID_SOURCE_VALUES` — used consistently across Tasks 1, 3, 6.

## Global sequencing

Tasks are linear: 1 → 2 → 3 (resolver complete + exported) → 4 (contracts + driver helper) → 5 (call convention) → 6 (producer, depends on 3 + 5) → 7 (audit re-point, depends on 6) → 8 (commit-prep). No task depends on a later one.
