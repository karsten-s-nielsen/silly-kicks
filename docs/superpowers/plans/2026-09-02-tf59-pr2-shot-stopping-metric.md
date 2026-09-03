# TF-59 PR2 — Shot-Stopping Metric (Goals Prevented / GSAA) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a new event-only `silly_kicks/shot_stopping/` package computing **Goals Prevented / GSAA** per `(goalkeeper, match)` from an injected per-shot Post-Shot xG, consuming PR1's resolved `defending_gk_player_id` — and its **authoritative team** — as columns.

**Architecture:** A pure metric package mirroring `silly_kicks/restdefense/` (frozen `Params` dataclass, `_columns.py` schema, `_compute.py` orchestrator returning `tuple[pd.DataFrame, ShotStoppingReport]`, flat `__all__`, **no `add_*` action-coupled aggregator**). Event-only — reads SPADL `actions` + an injected `psxg_column` + the PR1-stamped `defending_gk_player_id` / `defending_gk_team_id`; **never imports `silly_kicks.tracking`**. One small additive change to PR1's `keeper_identity` (Task 1) makes the keeper's team authoritative from the resolver rather than inferred. Additive — no VAEP/tracking retrain, no re-materialize.

**Tech Stack:** Python, pandas, `silly_kicks.spadl.config` (action-type / result ids), `silly_kicks.id_compat` (ADR-019), `silly_kicks.keeper_identity` (the PR1 resolver). No tracking; no new runtime dependency.

**Spec:** `docs/superpowers/specs/2026-09-01-tf59-gk-shot-stopping-and-keeper-appearance-resolver-design.md` (§6 is PR2; §2 goals/non-goals; §9 bookkeeping). PR1 (the resolver + port) shipped as **4.106.0 / PR-S177 / ADR-084** and is on `main`. **Owner decision 2026-09-02:** the keeper's `team_id` is resolved UPSTREAM (gold-standard, authoritative from `keeper_map`), option (a) — emitted on both paths.

## Global Constraints

Copied from the spec + CLAUDE.md; every task's requirements implicitly include this section.

- **Event-only.** `silly_kicks/shot_stopping/**` must NOT import `silly_kicks.tracking` (any submodule). Allowed silly-kicks deps: `silly_kicks.spadl` (config/schema), `silly_kicks.id_compat`, and `silly_kicks.keeper_identity` (a top-level module, not tracking). Enforced by a new AST import-allowlist test (Task 4).
- **No `add_*` aggregator.** The public surface is `compute_shot_stopping` (a `compute_*`), not an `add_*`. C4 action-coupled aggregator count stays **33**.
- **Additive / no retrain.** No VAEP/tracking retrain, no re-materialize. **ONE deliberate change to shipped PR1 behaviour (Task 1, owner-approved):** `add_defending_gk_player_id` now ALSO emits an authoritative `defending_gk_team_id` column on BOTH paths (the defending team it already resolves from `keeper_map` to pick the keeper) — this AMENDS ADR-084's "byte-identical when `appearances` omitted" contract (the omit path gains one additive column; existing column VALUES are unchanged). `add_defending_gk_player_id` is new in 4.106.0, so real Hyrum exposure is minimal.
- **Injected PSxG (port pattern).** silly-kicks ships no xG/PSxG. `psxg_column` is injected; a missing column raises the canonical "ships no xG/PSxG model" message (the `xg_column` idiom from `vaep/labels.py`).
- **`id_compat` for all id ops (ADR-019).** No raw `==`/`!=` on ids; use `canonical_id`/`canonical_id_series`/`same_id`; group/join on canonicalised keys.
- **Period-relative time (ADR-017).** `time_seconds` is per-period; never treat it as an absolute clock.
- **Own goals by RESULT (ADR-018).** Own goals are `bad_touch`+`owngoal`, so the shot-class `type_id` gate already excludes them (no separate own-goal mask — PLAN-03). `shot_blocked` is nullable `"boolean"` (ADR-046): exclude iff literally `True`, treat `pd.NA` as not-blocked.
- **Honest coverage (ADR-042 / ADR-027).** A shot with `pd.NA` `defending_gk_player_id` is **not** dropped-silently nor misattributed: excluded from per-keeper rows but **counted** in the returned `ShotStoppingReport`. A genuinely-unresolvable value is `pd.NA`, never a fabricated id.
- **Resolver fixtures (empirically pinned against shipped PR1 4.106.0).** Any fixture that calls `resolve_keeper_identities` (roster path) MUST (1) include a `type_name` column — the resolver reads it for the goal-kick override (`keeper_identity.py:760`, `== "goalkick"`), so omitting it raises `KeyError('type_name')` (PLAN-09); and (2) have every team whose keeper must resolve ACT at least once per `(game, period)` — the event-only roster path SEEDS the map only from teams present in the actions (roster is a lookup, not a seed), so a defending team that never acts stamps all-NA (PLAN-08). Compute-only fixtures (hand-stamped `defending_gk_*`, no resolver call) are exempt.
- **CI-faithful gates.** `python -m pytest tests/ -m "not e2e"`; `python -m ruff check silly_kicks/ tests/ scripts/` + `ruff format --check`; `pyright` bare. New `@e2e` tests skip without data/token.
- **Delivery.** One feature branch off `main` (no worktrees). ONE coherent, fully-tested commit (no micro-commits). Version / PR-S / ADR numbers assigned ONLY at commit-prep after `git fetch && git merge origin/main`. **No `git commit`/`git push` without explicit per-commit owner approval.** This plan + the ADR + TODO grooming ride PR2's commit.

---

### Task 1: Emit authoritative `defending_gk_team_id` from `add_defending_gk_player_id`

**Files:**
- Modify: `silly_kicks/keeper_identity.py` — `add_defending_gk_player_id` emits `defending_gk_team_id`; the gold-standard (owner ruling 2026-09-03) reads it from the `keeper_map` VALUE, so `KeeperIdentity` gains a raw `team_id` field populated by `resolve_keeper_identities` on both the roster + native paths (see the refinement note under Interfaces).
- Create: `tests/keeper_identity/test_defending_gk_team_id.py`
- Modify: any existing test asserting `add_defending_gk_player_id`'s omit-path COLUMN SET / byte-identity (the omit path now carries `defending_gk_team_id`) — search `tests/keeper_identity/` + `tests/tracking/test_keeper_identity*.py` for column-set / byte-identical assertions and update them (values of existing columns are unchanged; only the column set grows by one). Verified consumer-safe: `tracking/_run_features.py:280` + `tests/tracking/test_run_tracking_features.py:77` compare both-sides, unaffected.
- Modify: `tests/tracking/conftest_id_dtype.py` — `add_defending_gk_player_id` is in `NON_LINKED_AGGREGATORS` (:276) with a justification that says verbatim **"Its SOLE output is the id column `defending_gk_player_id`"** (:279), which becomes FALSE once a 2nd id column is emitted. This file matches NO code-search glob (exactly the doc-drift the exemption discipline exists to prevent), so it is an EXPLICIT plan-listed edit (see Step 6).

**Interfaces:**
- Produces: `add_defending_gk_player_id(actions, keeper_map, *, appearances=None)` now emits **`defending_gk_team_id`** (dtype `object`) on BOTH paths — the defending (opponent) team it resolves from `keeper_map` to select the keeper; `pd.NA` where the defending team / keeper is unresolvable. Consumed by Task 3's `compute_shot_stopping`.
- **Gold-standard refinement (owner ruling 2026-09-03, ADR-085 §6):** `KeeperIdentity` gains a raw `team_id` field (additive, defaulted `pd.NA`), populated by `resolve_keeper_identities` on both the roster (seed loop + goal-kick override) and native (from the frames' non-ball rows) paths. `add_defending_gk_player_id` reads the opponent's team from that map **VALUE** (not recovered from the actions), so the `keeper_map` is self-sufficient and the defending team resolves even for an opponent that never appears in the actions (a frame-seeded map). The one `KeeperIdentity._fields` pin (`tests/keeper_identity/test_promotion_imports.py`) is updated to include `team_id`; a non-vacuity test proves the map-value read moves the result off NA.

**Why (design rationale, owner-approved option (a)):** the keeper's team is a FACT from the roster/appearance resolution, not an inference. `add_defending_gk_player_id` already derives "the single OTHER team present in `keeper_map` for that `(game, period)`" to pick the defending keeper — that IS the keeper's team. Emitting it makes team + identity come from ONE authoritative resolution (guaranteed consistent), eliminating the actions-based "opponent" inference and its 2-team assumption.

- [ ] **Step 1: Read the current `add_defending_gk_player_id` body** in `silly_kicks/keeper_identity.py` and locate where it resolves the defending/opponent `team_id` per action (both the coarse-`keeper_map` path and the `appearances` path — both select a keeper for the opponent team of the acting team). This resolved opponent team is what Step 3 emits.

- [ ] **Step 2: Write the failing tests** `tests/keeper_identity/test_defending_gk_team_id.py`:

```python
from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import same_id
from silly_kicks.keeper_identity import add_defending_gk_player_id, resolve_keeper_identities
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_FAIL = spadlconfig.result_id["fail"]


def _actions() -> pd.DataFrame:
    # Team 10 shoots (defended by team 20's keeper 88); team 20 shoots (defended by team 10's keeper 99).
    # BOTH teams act, so the event-only roster resolver SEEDS both into keeper_map. type_name is REQUIRED:
    # the roster resolver reads actions["type_name"] for the goal-kick override (keeper_identity.py:760,
    # `== "goalkick"`); omitting it raises KeyError('type_name') -- PLAN-09.
    return pd.DataFrame(
        [
            {"game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 100.0, "team_id": 10,
             "player_id": 1, "type_id": _SHOT, "type_name": "shot", "result_id": _FAIL},
            {"game_id": 1, "action_id": 1, "period_id": 1, "time_seconds": 200.0, "team_id": 20,
             "player_id": 2, "type_id": _SHOT, "type_name": "shot", "result_id": _FAIL},
        ]
    )


def _keeper_map():
    kmap, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 99, 20: 88})
    return kmap


def test_defending_gk_team_id_is_the_opponent_team_coarse_path():
    out = add_defending_gk_player_id(_actions(), _keeper_map())
    assert "defending_gk_team_id" in out.columns
    t10 = out[out["team_id"] == 10].iloc[0]   # team-10 shot -> defended by team 20 (keeper 88)
    t20 = out[out["team_id"] == 20].iloc[0]   # team-20 shot -> defended by team 10 (keeper 99)
    assert same_id(t10["defending_gk_player_id"], 88) and same_id(t10["defending_gk_team_id"], 20)
    assert same_id(t20["defending_gk_player_id"], 99) and same_id(t20["defending_gk_team_id"], 10)


def test_defending_gk_team_id_matches_keeper_on_appearance_path():
    # The appearance path must ALSO carry defending_gk_team_id. Build a minimal appearance table via the
    # public port builder (keeper 88 for team 20, keeper 99 for team 10, whole period).
    from silly_kicks.keeper_identity import KeeperSegment, build_keeper_appearances_from_segments

    ap = pd.concat(
        [
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 88, "starting_xi", 1, 0.0, 1, float("inf"))], [1], game_id=1
            ),
            build_keeper_appearances_from_segments(
                [KeeperSegment(10, 99, "starting_xi", 1, 0.0, 1, float("inf"))], [1], game_id=1
            ),
        ],
        ignore_index=True,
    )
    out = add_defending_gk_player_id(_actions(), _keeper_map(), appearances=ap)
    t10 = out[out["team_id"] == 10].iloc[0]
    assert same_id(t10["defending_gk_player_id"], 88) and same_id(t10["defending_gk_team_id"], 20)


def test_team_known_but_keeper_unresolved():
    # roster names team 10's keeper but NOT team 20's. Team 20 is still IDENTIFIABLE (present in the
    # actions), so a team-10 shot -> defending_gk_team_id = 20 (KNOWN) while defending_gk_player_id = NA
    # (team 20's keeper unresolved). The team's NA-ness is NOT tied to the keeper's (PLAN-07 rule).
    kmap, _ = resolve_keeper_identities(_actions(), identity="roster", roster={10: 99})
    out = add_defending_gk_player_id(_actions(), kmap)
    t10 = out[out["team_id"] == 10].iloc[0]  # defended by team 20, whose keeper is unresolved
    assert same_id(t10["defending_gk_team_id"], 20)
    assert pd.isna(t10["defending_gk_player_id"])


def test_team_id_na_when_no_opponent_in_period():
    # A (game, period) with only ONE team in the actions -> no distinct opponent -> BOTH
    # defending_gk_team_id and defending_gk_player_id are pd.NA (never fabricated).
    one_team = pd.DataFrame(  # only team 30 acts -> no opponent seeded (the intended NA case)
        [{"game_id": 2, "action_id": 0, "period_id": 1, "time_seconds": 10.0, "team_id": 30,
          "player_id": 5, "type_id": _SHOT, "type_name": "shot", "result_id": _FAIL}]
    )
    kmap, _ = resolve_keeper_identities(one_team, identity="roster", roster={30: 7})
    r = add_defending_gk_player_id(one_team, kmap).iloc[0]
    assert pd.isna(r["defending_gk_team_id"]) and pd.isna(r["defending_gk_player_id"])
```

> The `resolve_keeper_identities(..., identity="roster", roster={team: gk})` signature + the `KeeperSegment` / `build_keeper_appearances_from_segments` port builder are the real PR1 surface (verify against `silly_kicks/keeper_identity.py`). If the roster-applicability guard raises for the one-team map in `test_team_id_na_when_opponent_unresolvable`, adjust the fixture so the map is built but the opponent is simply absent for one team (the intended NA case), not a raise.

- [ ] **Step 3: Run to verify RED.** Run: `python -m pytest tests/keeper_identity/test_defending_gk_team_id.py -q`. Expected: FAIL because `defending_gk_team_id` is not yet emitted — **NOT** a `KeyError('type_name')` and **NOT** all-NA stamping: every resolve-calling fixture includes `type_name` (PLAN-09) and has BOTH opposing teams acting so the map is seeded (PLAN-08). If you see `KeyError('type_name')` or an all-NA `defending_gk_player_id`, the fixture — not the implementation — is wrong.

- [ ] **Step 4: Implement** — in `add_defending_gk_player_id`, capture the opponent `team_id` already resolved per action (the resolver derives it around `keeper_identity.py:503/529` to look up the keeper) and assign it to a new output column `defending_gk_team_id` (dtype `object`). Emit it on BOTH the coarse-map and appearance paths. Keep the column PURE (added to the returned copy, never mutating `actions`). Do NOT change any existing column's values. **NA rule (deliberate — PLAN-07):** `defending_gk_team_id` is the resolved OPPONENT TEAM and is NA only where that opponent team is unidentifiable for the action's `(game, period)` (no distinct opponent in `keeper_map`) — **INDEPENDENT of the keeper**. A row can carry a KNOWN `defending_gk_team_id` with an NA `defending_gk_player_id` (opponent team identified, its keeper unresolved). Do NOT tie the team's NA-ness to the keeper's.

- [ ] **Step 5: Update the affected existing tests** — the omit-path column-set / byte-identity assertions now include `defending_gk_team_id`. Add a one-line comment at each: `# ADR-085 amendment: add_defending_gk_player_id now stamps the authoritative defending_gk_team_id on both paths.`

- [ ] **Step 6: Correct the id-dtype exemption comment, then run the keeper_identity gates (SHOULD-FIX PLAN-01).** `add_defending_gk_player_id` is EXEMPTED from the id-dtype-invariance sweep (`NON_LINKED_AGGREGATORS`, `tests/tracking/conftest_id_dtype.py:276`) — it takes a `keeper_map`, not `frames`, so the gate reaches no action-vs-frame id comparison and `test_id_dtype_invariance.py` verifies **nothing** about the new column (it passes vacuously). Update that entry's justification: replace **"Its SOLE output is the id column `defending_gk_player_id`, which `_is_id_col` excludes..."** with **"Its output id columns are `defending_gk_player_id` AND `defending_gk_team_id`, both excluded by `_is_id_col` from the value comparison..."** — the AGGREGATORS exemption still holds (both are id columns, so an AGGREGATORS entry would still be vacuous). The new column's dtype-safety is real and ALREADY exercised: it comes from the SAME opponent lookup that routes team ids through `canonical_id` against the canonical `keeper_map` keys (covered by `test_keeper_placement_helpers.py` + `test_keeper_identity_contracts.py::test_roster_keys_match_across_id_dtypes_via_id_compat`) and downstream by compute's `canonical_id_series` (Task 3); it adds no `*_id` scalar PARAMETER, so the id-scalar registry is unaffected. Run: `python -m pytest tests/keeper_identity/ tests/tracking/test_keeper_identity_native.py tests/tracking/test_keeper_identity_roster.py tests/tracking/test_keeper_identity_contracts.py tests/tracking/test_keeper_placement_helpers.py tests/test_add_star_purity.py tests/test_enrichment_nan_safety.py tests/tracking/test_id_dtype_invariance.py -q`. Expected: PASS (purity holds — column added to a copy; NaN inputs → NA).

- [ ] **Step 7: Commit** — do NOT commit.

---

### Task 2: `shot_stopping` scaffolding — schema, params, report, `__init__`

**Files:**
- Create: `silly_kicks/shot_stopping/__init__.py`
- Create: `silly_kicks/shot_stopping/_columns.py`
- Create: `silly_kicks/shot_stopping/_config.py`
- Create: `silly_kicks/shot_stopping/_report.py`
- Create: `tests/shot_stopping/__init__.py`
- Create: `tests/shot_stopping/test_columns.py`
- Create: `tests/shot_stopping/test_config.py`

**Interfaces:**
- Produces: `SHOT_STOPPING_COLUMNS: dict[str,str]`, `SS_KEYS`, the metric-column name constants, `ShotStoppingParams` (frozen; `.default(force_universal=)`, `.for_provider(provider)`, `.is_default()`), `ShotStoppingReport` (frozen; conserving). Consumed by Task 3's `_compute.py`.

- [ ] **Step 1: Write `_columns.py`.**

```python
"""Shot-stopping metric output columns (TF-59 PR2). Single source for the schema every gate iterates."""

from __future__ import annotations

#: Grain: one output row per (game_id, defending keeper player_id).
SS_KEYS = ["game_id", "player_id"]

#: The keeper's team, carried onto each output row (authoritative, from defending_gk_team_id).
SS_TEAM_ID = "team_id"

# Metric column names (spec §6.1). Counts -> Int64; PSxG / goals-prevented -> float64.
SS_SHOTS_FACED = "shots_faced"
SS_GOALS_CONCEDED = "goals_conceded"
SS_PSXG_FACED = "psxg_faced"
SS_GOALS_PREVENTED = "goals_prevented"  # == GSAA: sum(psxg_faced) - goals_conceded
SS_SHOTS_FACED_EXCL_PEN = "shots_faced_excl_penalties"
SS_GOALS_CONCEDED_EXCL_PEN = "goals_conceded_excl_penalties"
SS_PSXG_FACED_EXCL_PEN = "psxg_faced_excl_penalties"
SS_GOALS_PREVENTED_EXCL_PEN = "goals_prevented_excl_penalties"

#: The derived metric columns (documented in feature_glossary; the sample keys + team are structural).
SHOT_STOPPING_METRIC_COLUMNS = [
    SS_SHOTS_FACED,
    SS_GOALS_CONCEDED,
    SS_PSXG_FACED,
    SS_GOALS_PREVENTED,
    SS_SHOTS_FACED_EXCL_PEN,
    SS_GOALS_CONCEDED_EXCL_PEN,
    SS_PSXG_FACED_EXCL_PEN,
    SS_GOALS_PREVENTED_EXCL_PEN,
]

#: Full output column order + dtype (counts nullable Int64; PSxG/GP float64; keys/team object-tolerant).
SHOT_STOPPING_COLUMNS: dict[str, str] = {
    "game_id": "object",
    "player_id": "object",
    SS_TEAM_ID: "object",
    SS_SHOTS_FACED: "Int64",
    SS_GOALS_CONCEDED: "Int64",
    SS_PSXG_FACED: "float64",
    SS_GOALS_PREVENTED: "float64",
    SS_SHOTS_FACED_EXCL_PEN: "Int64",
    SS_GOALS_CONCEDED_EXCL_PEN: "Int64",
    SS_PSXG_FACED_EXCL_PEN: "float64",
    SS_GOALS_PREVENTED_EXCL_PEN: "float64",
}
```

- [ ] **Step 2: Write `_config.py`** (mirror `RestDefenseParams`; the one genuine structural knob is the shootout period, excluded per spec §6.2).

```python
"""ShotStoppingParams -- frozen params for the TF-59 GK shot-stopping metric (PR2).

Mirrors ``restdefense.RestDefenseParams``: a frozen dataclass with ``.default`` / ``.for_provider`` /
``.is_default`` and an EMPTY per-provider override map until an ADR-009 apply-gate clears. The metric
has no *calibratable* parameter (GP/GSAA is deterministic over an injected PSxG); the sole structural
knob is which period id is the penalty shootout (excluded).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ShotStoppingParams:
    """Parameters for the shot-stopping metric.

    Attributes
    ----------
    shootout_period_id: The period id treated as the penalty shootout and EXCLUDED entirely (spec §6.2).

    Examples
    --------
    >>> from silly_kicks.shot_stopping import ShotStoppingParams
    >>> ShotStoppingParams().shootout_period_id
    5
    """

    shootout_period_id: int = 5
    _is_universal_default: bool = field(default=False, compare=False, repr=False)

    @classmethod
    def default(cls, *, force_universal: bool = False) -> ShotStoppingParams:
        return cls(_is_universal_default=not force_universal)

    @classmethod
    def for_provider(cls, provider: str) -> ShotStoppingParams:
        return dataclasses.replace(cls(), **_PROVIDER_SHOT_STOPPING_PARAMS.get(provider, {}))

    def is_default(self) -> bool:
        return self._is_universal_default


#: EMPTY until an ADR-009 apply-gate clears (a per-provider tune is a separate gated PR, never this cycle).
_PROVIDER_SHOT_STOPPING_PARAMS: dict[str, dict] = {}
```

- [ ] **Step 3: Write `_report.py`** (mirror `RestDefenseReport`).

```python
"""ShotStoppingReport -- attribution-coverage census for compute_shot_stopping (TF-59 PR2).

Field names mirror ``RestDefenseReport`` / ``GkdvReport``. A shot faced with a resolved defending
keeper is ATTRIBUTED; one with a ``pd.NA`` ``defending_gk_player_id`` is UNATTRIBUTED (surfaced here,
never silently dropped nor misattributed -- ADR-042). Conservation
(``n_shots_attributed + n_shots_unattributed == n_shots_faced``) is asserted by a CI gate (Task 3).
"""

from __future__ import annotations

from dataclasses import dataclass

from ._config import ShotStoppingParams


@dataclass(frozen=True)
class ShotStoppingReport:
    """Per-``compute_shot_stopping`` attribution census over the on-target-shots-faced population.

    Examples
    --------
    >>> from silly_kicks.shot_stopping import ShotStoppingParams, ShotStoppingReport
    >>> r = ShotStoppingReport(ShotStoppingParams(), 20, 18, 2)
    >>> r.n_shots_attributed + r.n_shots_unattributed == r.n_shots_faced
    True
    """

    params: ShotStoppingParams
    n_shots_faced: int
    n_shots_attributed: int
    n_shots_unattributed: int
```

- [ ] **Step 4: Write `__init__.py`** (flat `__all__`, alphabetised).

```python
"""silly-kicks GK shot-stopping metrics -- Goals Prevented / GSAA (TF-59 PR2).

Event-only, per-(goalkeeper, match): from SPADL ``actions`` + an INJECTED per-shot Post-Shot xG
(``psxg_column``; silly-kicks ships no xG model) + the PR1-stamped ``defending_gk_player_id`` /
``defending_gk_team_id`` columns, compute Goals Prevented (== GSAA = sum(PSxG faced) - goals conceded),
reported with and without in-play penalties. Own goals / blocked shots / the penalty shootout are
excluded (spec §6.2).

Hexagonal / event-only: imports ``silly_kicks.spadl`` (config) + ``silly_kicks.id_compat`` +
``silly_kicks.keeper_identity`` ONLY; NEVER ``silly_kicks.tracking`` (pinned by
``tests/shot_stopping/test_import_allowlist.py``). NOTHING imports ``shot_stopping``. Additive -- no
VAEP/tracking retrain.

See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

from ._columns import SHOT_STOPPING_COLUMNS, SHOT_STOPPING_METRIC_COLUMNS, SS_KEYS
from ._compute import compute_shot_stopping
from ._config import ShotStoppingParams
from ._report import ShotStoppingReport

__all__ = [
    "SHOT_STOPPING_COLUMNS",
    "SHOT_STOPPING_METRIC_COLUMNS",
    "SS_KEYS",
    "ShotStoppingParams",
    "ShotStoppingReport",
    "compute_shot_stopping",
]
```

> NOTE: `__init__.py` imports `_compute` (Task 3). If executing strictly task-by-task, temporarily stub `_compute.py` with `def compute_shot_stopping(*a, **k): raise NotImplementedError` so Task-2 tests import cleanly; Task 3 replaces it.

- [ ] **Step 5: Write `tests/shot_stopping/test_columns.py` + `test_config.py`.**

```python
# tests/shot_stopping/test_columns.py
from __future__ import annotations

from silly_kicks.shot_stopping import SHOT_STOPPING_COLUMNS, SHOT_STOPPING_METRIC_COLUMNS, SS_KEYS


def test_keys_and_metric_columns():
    assert SS_KEYS == ["game_id", "player_id"]
    assert len(SHOT_STOPPING_METRIC_COLUMNS) == 8
    assert list(SHOT_STOPPING_COLUMNS)[:3] == ["game_id", "player_id", "team_id"]
    for c in SHOT_STOPPING_METRIC_COLUMNS:
        assert c in SHOT_STOPPING_COLUMNS
    assert SHOT_STOPPING_COLUMNS["shots_faced"] == "Int64"
    assert SHOT_STOPPING_COLUMNS["goals_prevented"] == "float64"
```

```python
# tests/shot_stopping/test_config.py
from __future__ import annotations

from silly_kicks.shot_stopping import ShotStoppingParams


def test_default_and_flag():
    assert ShotStoppingParams().shootout_period_id == 5
    assert ShotStoppingParams.default().is_default() is True
    assert ShotStoppingParams.default(force_universal=True).is_default() is False
    assert ShotStoppingParams().is_default() is False


def test_for_provider_returns_base_for_unlisted():
    assert ShotStoppingParams.for_provider("statsbomb") == ShotStoppingParams()
```

- [ ] **Step 6: Run.** Run: `python -m pytest tests/shot_stopping/test_columns.py tests/shot_stopping/test_config.py -q`. Expected: PASS (with the `_compute` stub).

- [ ] **Step 7: Commit** — do NOT commit.

---

### Task 3: The compute core — `compute_shot_stopping`

**Files:**
- Create: `silly_kicks/shot_stopping/_compute.py`
- Create: `tests/shot_stopping/test_compute.py`

**Interfaces:**
- Consumes: `SHOT_STOPPING_COLUMNS` + the metric-column constants, `ShotStoppingParams`, `ShotStoppingReport` (Task 2); `silly_kicks.spadl.config` (`actiontype_id`, `result_id`); `silly_kicks.id_compat.canonical_id_series`; the `defending_gk_team_id` column (Task 1).
- Produces: `compute_shot_stopping(actions, *, psxg_column, defending_gk_column="defending_gk_player_id", defending_team_column="defending_gk_team_id", params=_DEFAULT_PARAMS) -> tuple[pd.DataFrame, ShotStoppingReport]` — one row per `(game_id, defending keeper player_id)`, columns exactly `SHOT_STOPPING_COLUMNS`, plus the conserving report.

**Metric definition (spec §6.2):**
- A **shot faced** row = `type_id ∈ {shot, shot_penalty, shot_freekick}` AND `psxg_column` non-null (PSxG-presence IS the on-target gate) AND not known-blocked (`~shot_blocked.eq(True).fillna(False)`) AND `period_id != shootout_period_id`. **Own goals are excluded by construction** — they are `bad_touch`+`owngoal` (ADR-018), so the shot-class gate already drops them (no separate mask; PLAN-03). The defending keeper is the row's `defending_gk_column`.
- **Goal conceded** = a shot-faced row with `result_id == success`.
- **Goals Prevented (= GSAA)** = `sum(psxg_faced) - goals_conceded` per `(game, keeper)`.
- **Penalty split**: `_excl_penalties` companions drop `type_id == shot_penalty` rows.
- **team_id** (the keeper's team) = the AUTHORITATIVE `defending_team_column` stamped by `add_defending_gk_player_id` (the defending team the resolver used) — read directly, constant per keeper, **never inferred from the actions**.
- **Attribution census**: shot-faced rows with `pd.NA` keeper are excluded from grouping but counted in the report.

- [ ] **Step 1: Write `tests/shot_stopping/test_compute.py` (failing tests first).**

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.shot_stopping import SHOT_STOPPING_COLUMNS, ShotStoppingReport, compute_shot_stopping
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_PEN = spadlconfig.actiontype_id["shot_penalty"]
_BAD_TOUCH = spadlconfig.actiontype_id["bad_touch"]
_PASS = spadlconfig.actiontype_id["pass"]
_SUCCESS = spadlconfig.result_id["success"]
_FAIL = spadlconfig.result_id["fail"]
_OWNGOAL = spadlconfig.result_id["owngoal"]


def _row(gid, pid, tid, tyid, resid, psxg, dgk, dgk_team, blocked=pd.NA):
    return {
        "game_id": gid, "period_id": pid, "team_id": tid, "type_id": tyid, "result_id": resid,
        "psxg": psxg, "defending_gk_player_id": dgk, "defending_gk_team_id": dgk_team,
        "shot_blocked": blocked,
    }


def _actions(rows) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["shot_blocked"] = df["shot_blocked"].astype("boolean")
    df["defending_gk_player_id"] = df["defending_gk_player_id"].astype("object")
    df["defending_gk_team_id"] = df["defending_gk_team_id"].astype("object")
    return df


def test_gsaa_exact_over_known_psxg():
    # Team 10 shoots at keeper 99 (team 20). 3 on-target: psxg .2/.5/.8 (one goal, .5). GP = 1.5 - 1 = 0.5.
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.2, 99, 20),
        _row(1, 1, 10, _SHOT, _SUCCESS, 0.5, 99, 20),
        _row(1, 1, 10, _SHOT, _FAIL, 0.8, 99, 20),
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert list(out.columns) == list(SHOT_STOPPING_COLUMNS)
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 3
    assert r["goals_conceded"] == 1
    assert r["psxg_faced"] == pytest.approx(1.5)
    assert r["goals_prevented"] == pytest.approx(0.5)  # GSAA
    assert r["team_id"] == 20  # AUTHORITATIVE, from defending_gk_team_id (not inferred)
    assert isinstance(rep, ShotStoppingReport)
    assert (rep.n_shots_faced, rep.n_shots_attributed, rep.n_shots_unattributed) == (3, 3, 0)


def test_team_id_comes_from_resolver_not_opponent_inference():
    # A shot row carrying a NOISY / wrong team_id (shooter mislabeled) must NOT change the keeper's team:
    # team_id comes from defending_gk_team_id, not "the other team in the actions".
    rows = [
        _row(1, 1, 777, _SHOT, _FAIL, 0.3, 99, 20),   # shooter team_id is garbage (777); keeper team = 20
    ]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["team_id"] == 20


def test_on_target_gate_is_psxg_presence():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, np.nan, 99, 20),   # off target (psxg NaN) -> excluded
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),      # on target
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["shots_faced"] == 1
    assert rep.n_shots_faced == 1


def test_blocked_and_owngoal_and_shootout_excluded():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.4, 99, 20, blocked=True),   # blocked -> excluded
        _row(1, 1, 10, _BAD_TOUCH, _OWNGOAL, 0.9, 99, 20),        # own goal (bad_touch) -> excluded by is_shot
        _row(1, 5, 10, _PEN, _SUCCESS, 0.75, 99, 20),             # shootout (period 5) -> excluded
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),                # the only counted shot
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert out[out["player_id"] == 99].iloc[0]["shots_faced"] == 1
    assert rep.n_shots_faced == 1


def test_penalty_split():
    rows = [
        _row(1, 1, 10, _PEN, _SUCCESS, 0.79, 99, 20),   # in-play penalty, scored
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),      # open-play save
    ]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 2 and r["goals_conceded"] == 1
    assert r["psxg_faced"] == pytest.approx(1.09)
    assert r["goals_prevented"] == pytest.approx(0.09)
    assert r["shots_faced_excl_penalties"] == 1
    assert r["goals_conceded_excl_penalties"] == 0
    assert r["psxg_faced_excl_penalties"] == pytest.approx(0.3)
    assert r["goals_prevented_excl_penalties"] == pytest.approx(0.3)


def test_unattributed_shot_counted_not_dropped():
    rows = [
        _row(1, 1, 10, _SHOT, _FAIL, 0.6, pd.NA, pd.NA),   # unattributed (no keeper / no team)
        _row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20),         # attributed
    ]
    out, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert set(out["player_id"]) == {99}
    assert (rep.n_shots_faced, rep.n_shots_attributed, rep.n_shots_unattributed) == (2, 1, 1)


def test_mid_match_gk_change_attributes_per_keeper():
    rows = [
        _row(1, 1, 10, _SHOT, _SUCCESS, 0.7, 99, 20),   # period 1 -> keeper 99 (team 20)
        _row(1, 2, 10, _SHOT, _FAIL, 0.4, 98, 20),      # period 2 -> keeper 98 (team 20, post-change)
    ]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert set(out["player_id"]) == {98, 99}
    assert out[out["player_id"] == 99].iloc[0]["goals_conceded"] == 1
    assert out[out["player_id"] == 98].iloc[0]["goals_conceded"] == 0
    assert set(out["team_id"]) == {20}  # both keepers on team 20


def test_keeper_faced_only_a_penalty_has_zero_excl_companions():
    # PLAN-04: a keeper who faced ONLY an in-play penalty -> the _excl_penalties companions take the
    # fillna path (0 shots / 0 goals / 0.0 psxg / 0.0 GP), NOT NA.
    rows = [_row(1, 1, 10, _PEN, _SUCCESS, 0.79, 99, 20)]
    out, _ = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    r = out[out["player_id"] == 99].iloc[0]
    assert r["shots_faced"] == 1 and r["goals_conceded"] == 1
    assert r["shots_faced_excl_penalties"] == 0 and r["goals_conceded_excl_penalties"] == 0
    assert r["psxg_faced_excl_penalties"] == pytest.approx(0.0)
    assert r["goals_prevented_excl_penalties"] == pytest.approx(0.0)


def test_appearances_to_compute_chain_flips_at_the_sub():
    # PLAN-05 / spec §6.4: the FULL PR1->PR2 chain. A half-time GK change (team 20: keeper 99 in period 1,
    # keeper 98 in period 2) via a PR1 appearance table -> add_defending_gk_player_id stamps the per-action
    # keeper + team -> compute attributes each period's shot to the right keeper (both on team 20).
    # The implementer verifies the exact appearance-interval resolution against the real PR1 surface.
    from silly_kicks.keeper_identity import (
        KeeperSegment,
        add_defending_gk_player_id,
        build_keeper_appearances_from_segments,
        resolve_keeper_identities,
    )

    acts = pd.DataFrame(
        [
            # team-10 shots (defended by team 20). type_name REQUIRED (roster resolver, PLAN-09).
            {"game_id": 1, "action_id": 0, "period_id": 1, "time_seconds": 100.0, "team_id": 10,
             "player_id": 1, "type_id": _SHOT, "type_name": "shot", "result_id": _SUCCESS,
             "psxg": 0.7, "shot_blocked": pd.NA},
            {"game_id": 1, "action_id": 1, "period_id": 2, "time_seconds": 100.0, "team_id": 10,
             "player_id": 1, "type_id": _SHOT, "type_name": "shot", "result_id": _FAIL,
             "psxg": 0.4, "shot_blocked": pd.NA},
            # team 20 (the DEFENDING team) must ACT in EACH period so the event-only roster resolver
            # SEEDS it into keeper_map -- roster is a lookup, not a seed; a team that never acts is never
            # mapped, so the sub-flip would stamp all-NA and never resolve (PLAN-08).
            {"game_id": 1, "action_id": 2, "period_id": 1, "time_seconds": 50.0, "team_id": 20,
             "player_id": 9, "type_id": _PASS, "type_name": "pass", "result_id": _SUCCESS,
             "psxg": np.nan, "shot_blocked": pd.NA},
            {"game_id": 1, "action_id": 3, "period_id": 2, "time_seconds": 50.0, "team_id": 20,
             "player_id": 9, "type_id": _PASS, "type_name": "pass", "result_id": _SUCCESS,
             "psxg": np.nan, "shot_blocked": pd.NA},
        ]
    )
    acts["shot_blocked"] = acts["shot_blocked"].astype("boolean")
    kmap, _ = resolve_keeper_identities(acts, identity="roster", roster={10: 1, 20: 99})
    ap = pd.concat(
        [
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 99, "starting_xi", 1, 0.0, 1, float("inf"))], [1, 2], game_id=1),
            build_keeper_appearances_from_segments(
                [KeeperSegment(20, 98, "sub_events", 2, 0.0, 2, float("inf"))], [1, 2], game_id=1),
            build_keeper_appearances_from_segments(
                [KeeperSegment(10, 1, "starting_xi", 1, 0.0, 2, float("inf"))], [1, 2], game_id=1),
        ],
        ignore_index=True,
    )
    stamped = add_defending_gk_player_id(acts, kmap, appearances=ap)
    out, _ = compute_shot_stopping(stamped, psxg_column="psxg")
    assert set(out["player_id"]) == {98, 99}
    assert out[out["player_id"] == 99].iloc[0]["goals_conceded"] == 1  # period-1 keeper conceded
    assert out[out["player_id"] == 98].iloc[0]["goals_conceded"] == 0
    assert set(out["team_id"]) == {20}


def test_report_conserves():
    rows = [_row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20), _row(1, 1, 10, _SHOT, _FAIL, 0.6, pd.NA, pd.NA)]
    _, rep = compute_shot_stopping(_actions(rows), psxg_column="psxg")
    assert rep.n_shots_attributed + rep.n_shots_unattributed == rep.n_shots_faced


def test_missing_psxg_column_raises_with_canonical_message():
    rows = [_row(1, 1, 10, _SHOT, _FAIL, 0.3, 99, 20)]
    with pytest.raises(KeyError, match="ships no.*xG/PSxG"):
        compute_shot_stopping(_actions(rows), psxg_column="post_shot_xg")
```

- [ ] **Step 2: Run to verify RED.** Run: `python -m pytest tests/shot_stopping/test_compute.py -q`. Expected: FAIL.

- [ ] **Step 3: Write `_compute.py`.**

```python
"""compute_shot_stopping -- Goals Prevented / GSAA per (goalkeeper, match) (TF-59 PR2, spec §6).

Event-only. Consumes SPADL ``actions`` + an INJECTED ``psxg_column`` + the PR1-stamped
``defending_gk_column`` / ``defending_team_column`` (keeper_identity.add_defending_gk_player_id).
PURE -- never mutates ``actions``. See NOTICE for full bibliographic citations.
"""

from __future__ import annotations

import pandas as pd

from silly_kicks.id_compat import canonical_id_series
from silly_kicks.spadl import config as spadlconfig

from ._columns import (
    SHOT_STOPPING_COLUMNS,
    SS_GOALS_CONCEDED,
    SS_GOALS_CONCEDED_EXCL_PEN,
    SS_GOALS_PREVENTED,
    SS_GOALS_PREVENTED_EXCL_PEN,
    SS_PSXG_FACED,
    SS_PSXG_FACED_EXCL_PEN,
    SS_SHOTS_FACED,
    SS_SHOTS_FACED_EXCL_PEN,
    SS_TEAM_ID,
)
from ._config import ShotStoppingParams
from ._report import ShotStoppingReport

_DEFAULT_PARAMS = ShotStoppingParams()  # module-level singleton (avoids a B008 call-in-default)

_SHOT_TYPE_IDS = frozenset(
    spadlconfig.actiontype_id[n] for n in ("shot", "shot_penalty", "shot_freekick")
)
_PENALTY_TYPE_ID = spadlconfig.actiontype_id["shot_penalty"]
_SUCCESS = spadlconfig.result_id["success"]


def compute_shot_stopping(
    actions: pd.DataFrame,
    *,
    psxg_column: str,
    defending_gk_column: str = "defending_gk_player_id",
    defending_team_column: str = "defending_gk_team_id",
    params: ShotStoppingParams = _DEFAULT_PARAMS,
) -> tuple[pd.DataFrame, ShotStoppingReport]:
    """Goals Prevented / GSAA per (game_id, defending keeper player_id). Returns (samples, report).

    Examples
    --------
    See ``tests/shot_stopping/test_compute.py`` for a worked analytic fixture (injected PSxG ->
    exact GP/GSAA, penalty split, own-goal / blocked / shootout exclusion, attribution census).
    """
    if psxg_column not in actions.columns:
        raise KeyError(
            f"psxg_column {psxg_column!r} is not in actions -- silly-kicks ships no xG/PSxG model. "
            "Inject a Post-Shot xG column (port pattern; cf. xg_column in vaep/labels.py, ADR-085)."
        )
    for col in (defending_gk_column, defending_team_column):
        if col not in actions.columns:
            raise KeyError(
                f"{col!r} is not in actions -- stamp it first with "
                "silly_kicks.keeper_identity.add_defending_gk_player_id (ADR-084/085)."
            )

    psxg = pd.to_numeric(actions[psxg_column], errors="coerce")
    is_shot = actions["type_id"].isin(_SHOT_TYPE_IDS)
    on_target = psxg.notna()                                        # PSxG presence IS the on-target gate
    not_blocked = ~actions["shot_blocked"].eq(True).fillna(False)   # NA -> not blocked (ADR-046)
    not_shootout = actions["period_id"].ne(params.shootout_period_id)
    # Own goals are bad_touch+owngoal (ADR-018), so is_shot ALREADY excludes them -- no separate mask
    # (PLAN-03: a not_owngoal mask over shot-class rows is dead code).
    faced_mask = is_shot & on_target & not_blocked & not_shootout

    faced = pd.DataFrame(
        {
            "game_id": actions["game_id"],
            "keeper": canonical_id_series(actions[defending_gk_column]),
            "team": canonical_id_series(actions[defending_team_column]),  # AUTHORITATIVE (from the resolver)
            "psxg": psxg,
            "is_goal": (actions["result_id"] == _SUCCESS),
            "is_penalty": (actions["type_id"] == _PENALTY_TYPE_ID),
        }
    )[faced_mask.to_numpy()].reset_index(drop=True)

    n_faced = int(len(faced))
    attributed = faced[faced["keeper"].notna()]
    n_attr = int(len(attributed))
    report = ShotStoppingReport(
        params=params, n_shots_faced=n_faced, n_shots_attributed=n_attr,
        n_shots_unattributed=n_faced - n_attr,
    )

    empty = pd.DataFrame({c: pd.Series(dtype=t) for c, t in SHOT_STOPPING_COLUMNS.items()})
    if attributed.empty:
        return empty, report

    def _agg(frame: pd.DataFrame) -> pd.DataFrame:
        g = frame.groupby(["game_id", "keeper"], dropna=True, sort=False)
        out = pd.DataFrame(
            {
                "shots_faced": g.size(),
                "goals_conceded": g["is_goal"].sum(),
                "psxg_faced": g["psxg"].sum(min_count=1),
                "team": g["team"].first(),
            }
        )
        out["goals_prevented"] = out["psxg_faced"] - out["goals_conceded"]
        return out.reset_index()

    full = _agg(attributed).rename(
        columns={
            "shots_faced": SS_SHOTS_FACED, "goals_conceded": SS_GOALS_CONCEDED,
            "psxg_faced": SS_PSXG_FACED, "goals_prevented": SS_GOALS_PREVENTED,
        }
    )
    excl = _agg(attributed[~attributed["is_penalty"]]).rename(
        columns={
            "shots_faced": SS_SHOTS_FACED_EXCL_PEN, "goals_conceded": SS_GOALS_CONCEDED_EXCL_PEN,
            "psxg_faced": SS_PSXG_FACED_EXCL_PEN, "goals_prevented": SS_GOALS_PREVENTED_EXCL_PEN,
        }
    ).drop(columns=["team"])

    merged = full.merge(excl, on=["game_id", "keeper"], how="left")
    merged[SS_TEAM_ID] = merged["team"]  # AUTHORITATIVE keeper team (from defending_gk_team_id)
    out = merged.rename(columns={"keeper": "player_id"}).reindex(columns=list(SHOT_STOPPING_COLUMNS))
    # A keeper who faced only penalties has no open-play rows -> the excl companions are NA from the left
    # merge; a keeper with 0 non-penalty shots has 0 excl-penalty shots, GP 0.0.
    for col, fill in (
        (SS_SHOTS_FACED_EXCL_PEN, 0), (SS_GOALS_CONCEDED_EXCL_PEN, 0),
        (SS_PSXG_FACED_EXCL_PEN, 0.0), (SS_GOALS_PREVENTED_EXCL_PEN, 0.0),
    ):
        out[col] = out[col].fillna(fill)
    return out.astype(SHOT_STOPPING_COLUMNS), report
```

> Implementer notes: (1) `canonical_id_series` returns object dtype with `pd.NA` for missing — grouping with `dropna=True` drops unattributed rows cleanly. (2) `g["is_goal"].sum()` on a boolean Series yields an int count. (3) The empty frame is built column-by-column with declared dtypes (never `.astype` on an all-object empty). (4) `g["team"].first()` carries the constant per-keeper authoritative team. Verify against `test_compute.py`.

- [ ] **Step 4: Run to verify GREEN.** Run: `python -m pytest tests/shot_stopping/test_compute.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** — do NOT commit.

---

### Task 4: Cross-cutting gates — import allowlist + purity

**Files:**
- Create: `tests/shot_stopping/test_import_allowlist.py`
- Create: `tests/shot_stopping/test_purity.py`

- [ ] **Step 1: Write `test_import_allowlist.py`** — event-only: `shot_stopping/**` must NOT import `silly_kicks.tracking` (any submodule), and nothing imports `shot_stopping`. Model on the **`providers/appearances` allowlist** (bans `silly_kicks.tracking` outright), with planted-violation meta-tests.

```python
"""shot_stopping is EVENT-ONLY: never imports silly_kicks.tracking; nothing imports shot_stopping.

Mirrors tests/providers/test_appearances_import_allowlist.py (AST module-level). Allowed silly-kicks
deps: silly_kicks.spadl (config), silly_kicks.id_compat, silly_kicks.keeper_identity (top-level, not
tracking). Each detector carries a planted-violation meta-test.
"""

from __future__ import annotations

import ast
import pathlib

import silly_kicks
import silly_kicks.shot_stopping  # must import cleanly

ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent
SHOT_STOPPING = ROOT / "shot_stopping"
_BANNED_PREFIX = "silly_kicks.tracking"


def _imported_modules(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
        elif isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
    return mods


def _is_banned(m: str) -> bool:
    return m == _BANNED_PREFIX or m.startswith(_BANNED_PREFIX + ".")


def _imports_shot_stopping(path: pathlib.Path) -> bool:
    return any(
        m == "silly_kicks.shot_stopping" or m.startswith("silly_kicks.shot_stopping.")
        for m in _imported_modules(path)
    )


def test_shot_stopping_never_imports_tracking():
    offenders = {
        py.relative_to(SHOT_STOPPING).as_posix(): [m for m in _imported_modules(py) if _is_banned(m)]
        for py in sorted(SHOT_STOPPING.rglob("*.py"))
    }
    offenders = {k: v for k, v in offenders.items() if v}
    assert not offenders, f"{offenders}: shot_stopping is event-only -- must NEVER import silly_kicks.tracking."


def test_nothing_imports_shot_stopping():
    offenders = [
        py.relative_to(ROOT).as_posix()
        for py in sorted(ROOT.rglob("*.py"))
        if not py.is_relative_to(SHOT_STOPPING) and _imports_shot_stopping(py)
    ]
    assert not offenders, f"{offenders}: nothing in silly_kicks should import shot_stopping (a leaf metric)."


def test_public_surface_exists():
    from silly_kicks.shot_stopping import compute_shot_stopping  # noqa: F401


def test_banned_detector_fires_on_planted_violation(tmp_path):
    planted = tmp_path / "_p.py"
    planted.write_text("from silly_kicks.tracking import add_das\n", encoding="utf-8")
    assert any(_is_banned(m) for m in _imported_modules(planted))
    planted.write_text("from silly_kicks.keeper_identity import add_defending_gk_player_id\n", encoding="utf-8")
    assert not any(_is_banned(m) for m in _imported_modules(planted))
```

- [ ] **Step 2: Write `test_purity.py`** — mirror `tests/restdefense/test_purity.py`.

```python
"""Purity gate for shot_stopping (ADR-033 discipline): compute_shot_stopping never mutates inputs."""

from __future__ import annotations

import pandas as pd

from silly_kicks.shot_stopping import compute_shot_stopping
from silly_kicks.spadl import config as spadlconfig

_SHOT = spadlconfig.actiontype_id["shot"]
_FAIL = spadlconfig.result_id["fail"]


def _actions() -> pd.DataFrame:
    df = pd.DataFrame(
        [{"game_id": 1, "period_id": 1, "team_id": 10, "type_id": _SHOT, "result_id": _FAIL,
          "psxg": 0.3, "defending_gk_player_id": 99, "defending_gk_team_id": 20, "shot_blocked": pd.NA}]
    )
    df["shot_blocked"] = df["shot_blocked"].astype("boolean")
    df["defending_gk_player_id"] = df["defending_gk_player_id"].astype("object")
    df["defending_gk_team_id"] = df["defending_gk_team_id"].astype("object")
    return df


def test_compute_is_pure():
    actions = _actions()
    before = actions.copy()
    out, _ = compute_shot_stopping(actions, psxg_column="psxg")
    pd.testing.assert_frame_equal(actions, before)
    assert out is not actions
```

- [ ] **Step 3: Run.** Run: `python -m pytest tests/shot_stopping/ -q`. Expected: PASS.

- [ ] **Step 4: Commit** — do NOT commit.

---

### Task 5: feature_glossary 6th leg + entries + NOTICE

**Files:**
- Modify: `tests/invariants/glossary_emitted_columns.py` (add a `_shot_stopping_columns()` leg + wire it into `emitted_columns()` + update the docstring's leg enumeration to SIX).
- Modify: `silly_kicks/feature_glossary.py` (8 `FeatureColumn` entries + the home-module + attribution constants).
- Modify: `NOTICE` (the GSAA / PSxG entry).

- [ ] **Step 1: Add the 6th glossary leg** in `tests/invariants/glossary_emitted_columns.py`:

```python
def _shot_stopping_columns() -> set[str]:
    """Derived shot-stopping metric columns emitted by compute_shot_stopping (TF-59 PR2).

    compute_shot_stopping is a ``compute_*`` (not add_*/xfns), so name-shape discovery misses it; this
    leg runs it on a tiny fixture and returns the DERIVED metric columns (the keys game_id/player_id
    + the structural team_id are not features)."""
    import pandas as pd

    from silly_kicks.shot_stopping import SHOT_STOPPING_METRIC_COLUMNS, compute_shot_stopping
    from silly_kicks.spadl import config as spadlconfig

    actions = pd.DataFrame(
        [{"game_id": 1, "period_id": 1, "team_id": 10, "type_id": spadlconfig.actiontype_id["shot"],
          "result_id": spadlconfig.result_id["fail"], "psxg": 0.3, "defending_gk_player_id": 99,
          "defending_gk_team_id": 20, "shot_blocked": pd.NA}]
    )
    actions["shot_blocked"] = actions["shot_blocked"].astype("boolean")
    actions["defending_gk_player_id"] = actions["defending_gk_player_id"].astype("object")
    actions["defending_gk_team_id"] = actions["defending_gk_team_id"].astype("object")
    samples, _ = compute_shot_stopping(actions, psxg_column="psxg")
    return set(SHOT_STOPPING_METRIC_COLUMNS) & set(samples.columns)
```

Add `| _shot_stopping_columns()` to the `raw = (...)` union in `emitted_columns()`, and update the module docstring to enumerate a **6th** leg.

- [ ] **Step 2: Add the 8 `FeatureColumn` entries** to `silly_kicks/feature_glossary.py` (units from the closed vocabulary: counts → `"count"`, PSxG/GP → `"xG"`).

```python
_M_SHOT_STOPPING = "silly_kicks.shot_stopping._compute"  # TF-59 PR2 GK shot-stopping (GP / GSAA)
_A_GSAA = "Goals Saved Above Expected (PSxG-based GSAA)"  # verbatim token; must appear in NOTICE

# -- TF-59 PR2 GK shot-stopping (shot_stopping._compute) -------------------------------------
FeatureColumn(name="shots_faced", definition=(
    "On-target, unblocked, in-play shots faced by this keeper in the match (own goals, blocked "
    "shots and the penalty shootout excluded); PSxG-presence is the on-target gate."),
    unit="count", emitting_module=_M_SHOT_STOPPING, attribution=None, higher_is_better=None),
FeatureColumn(name="goals_conceded", definition=(
    "Goals scored on the on-target shots this keeper faced (successful shot-class actions)."),
    unit="count", emitting_module=_M_SHOT_STOPPING, attribution=None, higher_is_better=False),
FeatureColumn(name="psxg_faced", definition=(
    "Sum of injected Post-Shot xG over the on-target shots this keeper faced -- the expected goals "
    "conceded given shot quality."),
    unit="xG", emitting_module=_M_SHOT_STOPPING, attribution=_A_GSAA, higher_is_better=None),
FeatureColumn(name="goals_prevented", definition=(
    "Goals Prevented == GSAA: sum(PSxG faced) minus goals conceded -- goals saved above the "
    "post-shot expectation. Positive = better than an average keeper on the same shots."),
    unit="xG", emitting_module=_M_SHOT_STOPPING, attribution=_A_GSAA, higher_is_better=True),
FeatureColumn(name="shots_faced_excl_penalties", definition=(
    "As shots_faced but excluding in-play penalties (shot_penalty rows)."),
    unit="count", emitting_module=_M_SHOT_STOPPING, attribution=None, higher_is_better=None),
FeatureColumn(name="goals_conceded_excl_penalties", definition=(
    "As goals_conceded but excluding in-play penalties."),
    unit="count", emitting_module=_M_SHOT_STOPPING, attribution=None, higher_is_better=False),
FeatureColumn(name="psxg_faced_excl_penalties", definition=(
    "As psxg_faced but excluding in-play penalties."),
    unit="xG", emitting_module=_M_SHOT_STOPPING, attribution=_A_GSAA, higher_is_better=None),
FeatureColumn(name="goals_prevented_excl_penalties", definition=(
    "As goals_prevented (GSAA) but excluding in-play penalties -- open-play + free-kick shot-stopping."),
    unit="xG", emitting_module=_M_SHOT_STOPPING, attribution=_A_GSAA, higher_is_better=True),
```

- [ ] **Step 3: Add the NOTICE entry** (the `_A_GSAA` token must appear verbatim) under "Mathematical / Methodological References":

```
- Goals Saved Above Expected (PSxG-based GSAA). Post-Shot xG (PSxG) is a
  widely-used shot-quality model (StatsBomb Post-Shot xG; American Soccer
  Analysis GSAA is the shots-faced aggregation).
  Used by: silly_kicks.shot_stopping._compute (TF-59 PR2).
  Goals Prevented == GSAA = sum(PSxG faced) - goals conceded, per (goalkeeper,
  match). silly-kicks ships NO xG/PSxG model -- PSxG is INJECTED (port pattern,
  cf. xg_column in vaep/labels.py). The formula is standard; own goals, blocked
  shots and the penalty shootout are excluded, penalties reported both ways.
```

- [ ] **Step 4: Run the coverage + linkage gates.** Run: `python -m pytest tests/test_feature_glossary_coverage.py tests/test_feature_glossary_notice_linkage.py -q`. Expected: PASS.

- [ ] **Step 5: Commit** — do NOT commit.

---

### Task 6: C4 — add the `shot_stopping` container + re-render

**Files:**
- Modify: `docs/c4/architecture.dsl` (add a `shot_stopping` container + relationships; bump the `glossary` container's feature-column count).
- Modify: `docs/c4/architecture.html` (regenerated via Graphviz `dot`).
- Modify: `tests/test_c4_feature_column_count.py` if it pins the feature-column total.

- [ ] **Step 1: Add the container line** to `docs/c4/architecture.dsl` (beside `restdefense`; description ≤ 200 chars — verify the char count):

```
shot_stopping = container "silly_kicks.shot_stopping" "TF-59 GK shot-stopping: Goals Prevented / GSAA from an INJECTED per-shot Post-Shot xG + the resolved defending keeper; own goals / blocked / shootout excluded. ADR-085." "Python" "Library"
```

(Count check at authoring time: `len(description) <= 200`. Trim if needed, e.g. drop "GK".)

- [ ] **Step 2: Add relationships** (event-only — depends on `spadl` config + `keeper_identity`; analyst uses it):

```
analyst -> shot_stopping "Computes Goals Prevented / GSAA from an injected per-shot Post-Shot xG via" "compute_shot_stopping()"
shot_stopping -> spadl "Reads SPADL action-type / result ids from" "Python import"
shot_stopping -> keeper_identity "Reads the resolved defending keeper id + team stamped by" "add_defending_gk_player_id()"
```

(Note: a `keeper_identity` container/module must exist in the DSL for the third relationship to resolve — verify it was added in PR1's C4 update; if `keeper_identity` is a top-level module NOT modelled as a container, drop that relationship line and rely on the container-level dependency being implicit, OR model `keeper_identity` as a container. Confirm against the current `architecture.dsl`.)

- [ ] **Step 3: Re-render `architecture.html` via Graphviz `dot`** (never Smetana — CLAUDE.md C4 rule): `structurizr.war export -format plantuml/c4plantuml` → `c4_assemble.py --inject-wrap-width` → `plantuml.jar -graphvizdot "C:/Users/Karsten/.claude/tools/graphviz/dot.exe" -tsvg *.puml` → `c4_assemble.py --svg-dir`. Confirm `-testdot` reports "Installation seems OK".

- [ ] **Step 4: Bump the feature-column count in the DSL** (Task 5 added 8 glossary entries). `tests/test_c4_feature_column_count.py` **DERIVES** `len(FEATURE_GLOSSARY)` and asserts the DSL matches — so ONLY the `glossary` container description in `architecture.dsl` needs editing (its literal `"...all 368 derived feature columns..."` → the new total, expected 376; verify with `len(silly_kicks.feature_glossary.FEATURE_GLOSSARY)`). **No edit to the test itself** (it computes the number). Keep the description ≤ 200 chars.

- [ ] **Step 5: Run the C4 gates.** Run: `python -m pytest tests/test_c4_dsl_description_cap.py tests/test_c4_aggregator_count.py tests/test_c4_feature_column_count.py -q`. Expected: PASS (subpackage-has-container; every description ≤ 200; aggregator count 33; feature-column count matches).

- [ ] **Step 6: Commit** — do NOT commit.

---

### Task 7: Commit-prep gate — version, CHANGELOG, ADR-085, TODO, CLAUDE.md — STOP for approval

**Files:**
- Modify: `silly_kicks/_version.py` (bump to the re-derived NEXT-FREE; expected `4.107.0`).
- Modify: `CHANGELOG.md` (new top entry, expected `[4.107.0]`, PR-Snnn / ADR-085).
- Create: `docs/superpowers/adrs/ADR-085-tf59-shot-stopping-metric.md`.
- Modify: `TODO.md` (remove the shipped TF-59 On-Deck row; **keep TF-59b (CSE)**; replace the "Current (unreleased branch)" block).
- Modify: `CLAUDE.md` (add the `shot_stopping` durable-contract bullet; amend the ADR-084 keeper-identity bullet — `add_defending_gk_player_id` now stamps `defending_gk_team_id` on both paths, amending the byte-identity contract).

- [ ] **Step 1: Full CI-faithful gate run** BEFORE touching numbers. `python -m pytest tests/ -m "not e2e" -p no:randomly -q`; `python -m ruff check silly_kicks/ tests/ scripts/`; `python -m ruff format --check silly_kicks/ tests/ scripts/`; `python -m pyright`. All green.

- [ ] **Step 2: `git fetch && git merge origin/main`; re-derive NEXT-FREE** version / PR-S / ADR (do NOT hardcode; expected 4.107.0 / PR-S178 / ADR-085 but re-derive). Resolve BOM/CRLF traps.

- [ ] **Step 3: Bump `silly_kicks/_version.py`**; `git grep -F <old-version>` clean outside CHANGELOG/history; `uv lock`.

- [ ] **Step 4: Write ADR-085** (`docs/superpowers/adrs/ADR-085-tf59-shot-stopping-metric.md`) recording: the event-only `shot_stopping` package; PSxG injected via the port pattern; PSxG-presence as the on-target gate; own-goal (ADR-018) / blocked (ADR-046) / shootout exclusions; per-(GK, match) grain; the `ShotStoppingReport` attribution census (ADR-042); consuming PR1's `defending_gk_player_id` **+ the new authoritative `defending_gk_team_id`** (the gold-standard team-from-resolver decision, option (a), **amending ADR-084's byte-identity-when-omitted contract**). Alternatives: caller-injected keeper + opponent-inferred team (both rejected — §10 + the 2026-09-02 decision). Use `docs/superpowers/adrs/ADR-TEMPLATE.md`.

- [ ] **Step 5: CHANGELOG entry** (`[4.107.0]`, PR-Snnn / ADR-085) — the metric + port + exclusions; the `add_defending_gk_player_id` `defending_gk_team_id` addition (additive, amends ADR-084 byte-identity); additive/no-retrain.

- [ ] **Step 6: TODO grooming** — replace the "Current (unreleased branch)" block with the 4.107.0 summary; **remove the TF-59 On-Deck row** (PR1 + PR2 both shipped); **keep TF-59b (CSE)**; `tests/test_todo_md_format.py` passes.

- [ ] **Step 7: CLAUDE.md** — add a `shot_stopping` durable-contract bullet (event-only GP/GSAA; injected PSxG; consumes `defending_gk_player_id`/`defending_gk_team_id`; own-goal/blocked/shootout exclusions; report-based attribution coverage; no `add_*`, C4 count 33; ADR-085). Amend the ADR-084 keeper-identity bullet to note `add_defending_gk_player_id` now stamps the authoritative `defending_gk_team_id` on both paths (byte-identity contract amended).

- [ ] **Step 8: STOP — present the full diff + file list + intended commit message and WAIT for explicit owner approval.** No `git add`/`git commit`/`git push` without it. On approval: one coherent commit, then push → PR → CI, per the owner's separate go-aheads.

---

## Self-Review (author checklist)

**Spec coverage (§6):** §6.1 package/API → Tasks 2–3 ✅; §6.2 metric definitions → Task 3 (all exclusions + penalty split + GP=GSAA tested) ✅; §6.3 edge cases → Task 3 (`ShotStoppingReport` census; per-keeper zero-rows caller-driven, documented) ✅; §6.4 validation → Task 3 + Task 5 (glossary) + Task 4 (purity) ✅. §9 bookkeeping → Tasks 5 (glossary + NOTICE) + 6 (C4) + 7 (ADR/CHANGELOG/TODO/version) ✅. **Team authority (owner decision 2026-09-02, option a)** → Task 1 (upstream `defending_gk_team_id`) + Task 3 (reads it; `test_team_id_comes_from_resolver_not_opponent_inference` proves it is not the actions inference) ✅.

**Placeholder scan:** the only intentional deferred tokens are the version/PR-S/ADR numbers (Task 7, re-derived at commit-prep) and the char-count trim on the C4 description (Task 6). No "TBD"/"handle edge cases". Task 1's appearance-path + NA tests reference existing PR1 fixtures/signatures the implementer verifies against the real `keeper_identity.py` (Step 1 reads it first).

**Type/name consistency:** `compute_shot_stopping(actions, *, psxg_column, defending_gk_column="defending_gk_player_id", defending_team_column="defending_gk_team_id", params=...)` and the column constants (`SS_*`, `SHOT_STOPPING_COLUMNS`, `SHOT_STOPPING_METRIC_COLUMNS`) are used identically across Tasks 2/3/4/5. `ShotStoppingReport` fields match between `_report.py` (Task 2) and `_compute.py` (Task 3). `defending_gk_team_id` is produced in Task 1 and consumed in Task 3.

**Independent review (2026-09-02) incorporated — round 1:** SHOULD-FIX PLAN-01 (id-dtype exemption comment + file list — Task 1 files/Step 6), CONSIDER PLAN-02 (C4 count wording — Task 6 Step 4), PLAN-03 (dead own-goal mask removed — Task 3 compute + Global Constraints), PLAN-04 (penalty-only test — Task 3), PLAN-05 (appearances→compute chain test — Task 3), PLAN-07 (deliberate team NA rule + split tests — Task 1). **Round 2 (empirical PR1 repro):** PLAN-09 (BLOCKING — the roster resolver reads `type_name` at `keeper_identity.py:760`; `type_name` added to every resolve-calling fixture + Step-3 wording) and PLAN-08 (the chain fixture's defending team must ACT to be seeded; team-20 actions added). Both were FIXTURE defects (design/compute/bookkeeping sound); the durable rule is now a Global Constraint. The reviews' CONFIRMED-CORRECT items (`type_id` over `type_name` for the metric gate; `"xG"` unit; C4 desc ≤ 200; DSL elements resolve; the resolver derives the opponent team at `:503/529`; omit-path change consumer-safe) validate the decisions below.

**Open risks (not blocking):**
1. **`type_id` vs `type_name`** for the shot gate: `type_id` (always in `SPADL_COLUMNS`, base int64, 1:1 map) avoids an `add_names` dependency; spec §6.2 wording says `type_name`. Equivalent; reviewer-confirmed correct.
2. **Unit** for GP/PSxG is `"xG"` (no `"goals"` token in the closed vocabulary). Reviewer-confirmed no `"goals"` token exists; `"xG"` defensible.
3. **`ShotStoppingParams`** carries only `shootout_period_id` — thin but house-consistent (spec §6.1 lists `_config.py`).
4. **ADR-084 byte-identity amendment (Task 1):** `add_defending_gk_player_id` now emits `defending_gk_team_id` on the omit path too — deliberate, owner-approved. Reviewer-verified consumer-safe (`_run_features.py:280` + `test_run_tracking_features.py:77` compare both-sides; new in 4.106.0). Task 1 Steps 5–6 update the affected tests + the exemption comment.
5. **Spec §6.1 signature divergence (PLAN-06, conscious):** the design doc §6.1 still shows `-> pd.DataFrame` with no `defending_team_column`; the plan returns `tuple[DataFrame, ShotStoppingReport]` (§6.3's "returned report", restdefense precedent) + adds `defending_team_column`. Per the repo convention (specs are the frozen design record; amendments live in ADRs), **ADR-085 records the delta** — the design doc is deliberately NOT retro-amended.
