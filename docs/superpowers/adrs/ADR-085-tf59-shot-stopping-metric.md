# ADR-085: TF-59 shot-stopping metric (Goals Prevented / GSAA) + authoritative defending-team stamp

| Field | Value |
|---|---|
| **Date** | 2026-09-02 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

silly-kicks has an extensive goalkeeper stack — xT-GK, `GkCompletionModel`, `GkRetentionModel`, GKDV, ghost-GK — but all of it is positioning/distribution; there is **no shot-stopping metric**. TF-59 PR2 fills that gap with the gold-standard, PSxG-based **Goals Prevented** / **GSAA**, event-only, per-`(goalkeeper, match)`. PR1 (ADR-084, 4.106.0) shipped the prerequisite keeper-appearance resolver + the interval-granular `add_defending_gk_player_id`. This PR consumes that resolution and adds the valuation math.

Two design questions shaped the implementation. First, **which keeper faced each shot** is already resolved by PR1 (`defending_gk_player_id`), so PR2 should read a column, not re-derive identity. Second, the metric reports per keeper *team*, and the keeper's team must be a **fact from the resolver** — the opponent `add_defending_gk_player_id` already derives from `keeper_map` to pick the keeper — not a separate inference from the shot actions (which fails on data-quality edges and decouples team from identity). The owner ruled (2026-09-02) for the authoritative, upstream team (option a). A follow-up owner ruling (2026-09-03) took the **gold-standard** form of that: the raw defending team is stored in the `keeper_map` VALUE (`KeeperIdentity.team_id`) rather than recovered from the actions, so it is available even when the opponent never acts (see decision 6).

## Decision

Ship a new **event-only `silly_kicks/shot_stopping/`** package plus a small additive change to PR1's resolver:

1. **`compute_shot_stopping(actions, *, psxg_column, defending_gk_column="defending_gk_player_id", defending_team_column="defending_gk_team_id", params=…) -> (samples, ShotStoppingReport)`** — one row per `(game_id, defending keeper player_id)`. A `compute_*`, **not** an `add_*` action-coupled aggregator (C4 count stays 33). Mirrors `restdefense/` (frozen `ShotStoppingParams`, `_columns.py` schema, `_report.py` report sidecar).
2. **PSxG injected** (port pattern; silly-kicks ships no xG/PSxG, exactly like `xg_column` in `vaep/labels.py`). **PSxG-presence IS the on-target gate** (PSxG exists only for on-target shots). A missing `psxg_column` raises the canonical "ships no xG/PSxG model" message.
3. **Exclusions:** own goals are `bad_touch`+`owngoal` (ADR-018), so the shot-class `type_id` gate already drops them (no separate mask); blocked shots via `shot_blocked` (ADR-046, `pd.NA` treated as not-blocked); the penalty shootout (`period_id == shootout_period_id`, default 5). **Goals Prevented ≡ GSAA = Σ PSxG(faced) − goals conceded**; penalties reported with and without in-play penalties.
4. **Honest attribution census** — a shot with an NA defending keeper is excluded from per-keeper rows but **counted** in `ShotStoppingReport` (`n_shots_faced` / `n_shots_attributed` / `n_shots_unattributed`, conserving), never dropped-silent nor misattributed (ADR-042 / ADR-027).
5. **`add_defending_gk_player_id` now emits the authoritative `defending_gk_team_id`** — the opponent team it already resolves from `keeper_map` to select the keeper — on BOTH paths. This **amends ADR-084's "byte-identical when `appearances` omitted" contract** (the omit path gains one additive column; existing column VALUES are unchanged). `add_defending_gk_player_id` is new in 4.106.0, so real Hyrum exposure is minimal.
6. **`KeeperIdentity` gains a raw `team_id` field** (gold-standard; owner ruled 2026-09-03). Defaulted `pd.NA` for back-compat, it is populated by `resolve_keeper_identities` on BOTH paths — roster (seed loop + goal-kick override) and native (recovered from the frames' non-ball rows). `add_defending_gk_player_id` reads the opponent's team straight from that map **VALUE**, so the `keeper_map` is **self-sufficient**: the defending team resolves even for an opponent that never appears in the actions (a frame-seeded map) — the case a from-actions recovery returned NA for.
7. **Output ids are RAW** (`player_id` from the raw `defending_gk_player_id`; `team_id` from the opponent's `KeeperIdentity.team_id`, the resolver's own raw provider representation), consistent with `defending_gk_player_id` and joinable via `id_compat` (ADR-019). Keepers are **GROUPED on their CANONICAL id** (`canonical_id_series`), not the raw id, so a keeper whose id appears in mixed dtypes across a match is not fragmented into two split-GSAA rows; the raw id is emitted via `.first()`. Byte-identical for a single-dtype provider; the id-op stays ADR-019-compliant.

## Alternatives considered

| Option | Why rejected |
|---|---|
| Caller injects `defending_keeper_id` + team (no library resolution) | Pushes sub/red-card correctness onto every caller; unvalidatable. Gold standard means the library owns identity (spec §10). |
| Team inferred as "the game's other team" from the shot actions | An inference, not a fact; fails on ≠2-team edges; decouples team from the keeper-identity source of truth. Owner ruled for the upstream authoritative team (a). |
| Canonical (string) output ids | Inconsistent with `defending_gk_player_id`'s raw ids and breaks natural `== <int>` consumer joins/selections. Raw + `id_compat` is the house discipline. |
| Team emitted only on the appearance path (preserve the ADR-084 byte-identity contract) | The keeper's team is authoritative on BOTH paths; emitting it only on one leaves the coarse path without it. Owner accepted the additive-column amendment (breaking changes not a concern). |
| Recover the raw defending team from the ACTIONS (not the map value) | Returns NA for an opponent that never appears in the actions (a frame-seeded map), even when its identity is known upstream — the `keeper_map` key is canonical, so the raw team is otherwise lost. The gold-standard stores the raw team in the map VALUE (`KeeperIdentity.team_id`) so it is always available, symmetric with the raw `gk_id` already stored there. |

## Consequences

### Positive
- The missing GK pillar: gold-standard per-`(keeper, match)` Goals Prevented / GSAA, attributable through substitutions / red cards via PR1's interval resolution.
- Team + identity come from ONE authoritative resolution (guaranteed consistent), both read from the `keeper_map` value; the map is **self-sufficient** (no from-actions recovery), so the defending team resolves even for an opponent absent from the actions (a frame-seeded map). The `KeeperIdentity` value now stores its own team symmetrically with its keeper id.

### Negative
- **Amends ADR-084's byte-identity-when-omitted contract:** `add_defending_gk_player_id` now stamps `defending_gk_team_id` on the omit path too. Additive (existing values unchanged); the id-dtype exemption comment in `tests/tracking/conftest_id_dtype.py` is corrected to name both id columns.
- **`KeeperIdentity` gains a public `team_id` field** (additive, defaulted `pd.NA`). Existing keyword/positional constructors are unaffected, but `KeeperIdentity._fields` changes and any exact-arity 3-tuple unpacking would break (none in-tree; the one `._fields` pin in `tests/keeper_identity/test_promotion_imports.py` is updated to include `team_id`). A hand-built map that OMITS `team_id` reads NA for `defending_gk_team_id` — the resolver always populates it, so only a synthetic map hits this.

### Neutral
- **Additive — no VAEP/tracking retrain, no re-materialize.** No `add_*` aggregator (C4 aggregator count stays 33); the 8 metric columns are documented in `feature_glossary` (a 6th `compute_*` leg) + `NOTICE`; a `shot_stopping` C4 container is added and `architecture.html` re-rendered via Graphviz `dot`.

## Related

- **Spec:** `docs/superpowers/specs/2026-09-01-tf59-gk-shot-stopping-and-keeper-appearance-resolver-design.md` (§6).
- **Plan:** `docs/superpowers/plans/2026-09-02-tf59-pr2-shot-stopping-metric.md`.
- **ADRs:** builds on and **amends ADR-084** (the omit-path byte-identity contract); consumes ADR-078 (single keeper resolver); ADR-018 (own goals by result), ADR-046 (`shot_blocked`), ADR-017 (period-relative time), ADR-042 (honest coverage), ADR-027 (NaN never a sentinel), ADR-019 (`id_compat`), ADR-009 (calibration apply-gate: `_PROVIDER_SHOT_STOPPING_PARAMS` empty), ADR-005 (NOTICE academic-attribution discipline).
