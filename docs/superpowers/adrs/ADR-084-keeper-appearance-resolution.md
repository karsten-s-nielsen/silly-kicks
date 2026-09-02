# ADR-084: Keeper-appearance resolution — promote the keeper-identity resolver, un-foreclose substitution resolution, and inject a per-provider appearance-interval port

| Field | Value |
|---|---|
| **Date** | 2026-09-02 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

ADR-078 established exactly ONE keeper-identity resolver (`tracking.resolve_keeper_identities`, single-source per ADR-055) but deliberately **foreclosed substitution-event resolution**: its `roster` rung names only the *starter*, and per-minute correctness through a mid-match GK change (a substitution or a red card) was left a documented follow-up. TF-59 (GK shot-stopping — Goals-Prevented / GSAA) is a *gold-standard* metric, and gold-standard means the library owns "which keeper faced this shot" — including a shot in the 78th minute after the starter was subbed at 60'. Measured multi-provider data (spec §3) shows this is not hypothetical: real GK substitutions and red cards occur across all four in-scope providers, each in a different native encoding.

Two constraints shape the solution. First, the resolver lived in `silly_kicks/tracking/_keeper_identity.py`, and TF-59's shot-stopping metric is **event-only** — it reads SPADL actions + an injected PSxG column and needs no tracking frames. Dragging `tracking/`'s heavy import chain (numba, pitch-control, accessible-space) into an event-only metric is exactly the coupling the `id_compat` promotion (4.53.0, ADR-019/043) already rejected once. Second, the four providers carry appearance/substitution data in four incompatible shapes — StatsBomb `Starting XI` + `Substitution` events, Sportec/DFL `PlayingPosition="TW"` subs + `PlayerBecomesGoalkeeper` emergency keepers, Gradient Sports roster + `gameEventType=="SUB"`, SkillCorner `players[].playing_time.by_period[]` intervals — so a single resolver cannot parse them; the resolver needs a *normalized* interval table injected into it.

## Decision

Ship keeper-appearance resolution as five coordinated pieces, **amending ADR-078** to un-foreclose substitution resolution: (1) **promote** `tracking/_keeper_identity.py` → the public top-level module `silly_kicks/keeper_identity.py` — a breaking import-path move with **no shim** (fail-loud at import, per the ADR-019 `id_compat`/ADR-043 clean-break precedent); the `native` identity path still **lazy-delegates** to `tracking._gk_resolve` so the promotion adds no runtime edge for existing consumers. (2) An **event-only enumeration path** on the roster resolver (`frames=None`), so keeper identity is resolvable from events alone. (3) An injected **`KeeperAppearances` port** — `validate_keeper_appearances` / `KEEPER_APPEARANCE_COLUMNS`, one row per keeper on-pitch interval, keyed per `(game, period, team)`, PERIOD-RELATIVE times (ADR-017), `object` ids (string-tolerant; DFL/SkillCorner ids are strings) — plus a shared **per-period builder** `build_keeper_appearances_from_segments` that decomposes each keeper's `(from_period,from_time)→(to_period,to_time)` segment into one row per period it spans. (4) **Interval-granular resolution** in `add_defending_gk_player_id(actions, keeper_map, *, appearances=)`: when an appearance table is supplied, each action's defending keeper is the opponent keeper whose `[start,end)` interval covers the action's `time_seconds`, so attribution flips exactly at the sub minute, and a `defending_gk_source` provenance column (the ADR-054 source-column pattern) records the appearance-vs-map cross-check — **byte-identical, and with NO extra column, when `appearances` is omitted**. (5) **Four provider extractors** in `providers/<p>/appearances.py` (StatsBomb / Sportec / GradientSports / SkillCorner), each emitting one row per `(keeper, period)` through the shared builder and depending **only** on the port + `id_compat` + pandas/stdlib — never `silly_kicks.tracking` (CI-enforced by an AST module-level import-allowlist gate, spec §5.6).

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. Per-shot `defending_keeper_id` injected by the caller (no library resolution) | Zero library work | Pushes sub/red-card correctness onto every caller; unvalidatable | Gold standard means the library owns identity, not the caller (spec §10) |
| B. Team-grain shot-stopping as the headline (skip keeper identity) | No interval resolution needed | A degradation — per-keeper *is* the metric | Team-grain is only an explicit fallback when no identity resolves |
| C. Frame-presence derivation of SkillCorner intervals | Reuses tracking frames | SkillCorner is broadcast tracking (~19.6% GK detection); keeper absence ≠ subbed off | Intervals come from the `by_period` metadata, not frame presence |
| D. Keep the resolver in `tracking/`; TF-59 imports tracking | No import-path break | An event-only metric drags in tracking's heavy chain; the resolver is a cross-cutting concern | The `id_compat` promotion precedent — a shared seam does not belong inside one heavy consumer |
| E (chosen). Promote the resolver + event-only path + injected appearance port + interval resolution + four extractors | Event-only metric stays light; one resolver; per-provider parsing isolated behind a normalized port; additive to trained models | A public import-path break for existing `tracking.resolve_keeper_identities` consumers | — |

## Consequences

### Positive

- Gold-standard per-keeper shot-stopping becomes attributable through substitutions and red cards, for all four providers, at the exact sub minute.
- An event-only TF-59 metric consumes the resolver + extractors without importing `silly_kicks.tracking` — the promotion's whole point, enforced by `tests/providers/test_appearances_import_allowlist.py` (AST module-level; a 5th `appearances.py` fails the completeness meta-assertion until added deliberately).
- Per-provider parsing complexity is isolated behind ONE normalized port; the interval-resolution consumer and the shared builder are provider-agnostic.

### Negative

- **Breaking public import-path change (intended, no shim):** consumers of `silly_kicks.tracking.resolve_keeper_identities` / `KeeperIdentity*` migrate to `silly_kicks.keeper_identity`. The in-repo migration (spec §5.6) covers `tracking/{features,_run_features,__init__}.py`, `scripts/{build_tf19_instrument_responsiveness,_sb_roster,_sb_battery}.py`, `docs/PRIVATE_CONSUMERS.md`, and the affected tests; a stale external pin fails loud at import (never silently).
- Starter identification is irreducibly provider-specific and must be maintained per provider: StatsBomb `Starting XI` goalkeeper lineup entry; Sportec `gk_player_ids` disambiguated by earliest keeper action; Gradient Sports earliest on-ball action among non-subbed-on roster GKs; SkillCorner `player_role.acronym=="GK"` / `name=="Goalkeeper"` (which also catches sub keepers).

### Neutral

- **Additive to trained models — no VAEP/tracking retrain, no re-materialize.** The `native` / `roster` resolver outputs are byte-identical; the appearance/event-only paths are opt-in, and `add_defending_gk_player_id` is byte-identical (and column-identical) when `appearances` is omitted.
- A reduced provider export lacking interval data (e.g. the SkillCorner peggy44 schema with no `match_periods`/`by_period`) yields an **honest empty** appearance frame — interval resolution then falls back to the coarse keeper map, never a fabricated interval.
- **C4:** PR1 adds the promoted top-level `keeper_identity` module + two new provider subpackages (`providers/gradientsports`, `providers/skillcorner`) for their extractors; the modeled-surface impact is evaluated by the C4 completeness gate at commit-prep (re-render via Graphviz `dot`, never Smetana, if the gate requires it).

## Related

- **Specs:** `docs/superpowers/specs/2026-09-01-tf59-gk-shot-stopping-and-keeper-appearance-resolver-design.md`
- **Plan:** `docs/superpowers/plans/2026-09-01-tf59-pr1-keeper-appearance-resolver.md`
- **ADRs:** amends **ADR-078** (un-forecloses substitution-event keeper resolution); builds on ADR-055 (single-source keeper-identity resolver), ADR-019/ADR-043 (`id_compat` public-seam promotion precedent — a shared seam does not live inside one heavy consumer), ADR-017 (period-relative time base), ADR-054 (source-column provenance pattern), and ADR-005 (NOTICE academic-attribution discipline for the appearance extractors).
