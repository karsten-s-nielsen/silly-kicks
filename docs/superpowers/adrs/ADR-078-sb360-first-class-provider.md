# ADR-078: SB360 as a first-class tracking-feature provider — one keeper-identity resolver, a canonical call convention, and a provider-agnostic producer

| Field | Value |
|---|---|
| **Date** | 2026-08-28 |
| **Status** | Accepted |
| **Deciders** | Karsten Nielsen |

## Context

StatsBomb-360 freeze-frames are the library's most information-rich tracking source and its most awkward to consume. They are **anonymous**: each player row carries only actor-relative flags (`teammate` / `actor` / `keeper`) and no player identity, and `snapshot_to_tracking_frames` numbers the rows (a row number does not recur across frames). So every GK-domain aggregator that keys on *which* keeper (`add_pre_shot_gk_position` requires a `defending_gk_player_id` column; `add_xt_gk` / `add_ghost_gk` need the resolved keeper) has nothing to key on — the SB360 GK features were silently NaN. Separately, the 33 tracking `add_*` aggregators do **not** share one call shape (six need a fitted `ExpectedThreat`; `add_defensive_credit` needs an `xg_column` the library does not ship; `add_pre_shot_gk_angle` took `frames` keyword-only while its sibling took it positionally), so every consumer re-derived per-aggregator handling — the `ADAPTER_MAP` in `scripts/_sb_battery.py` was the only copy.

The keeper identity **exists outside the frame**, in two authoritative places: the roster/lineup (`{team_id: gk_id}`, naming the *defending* keeper whom no event names) and the goal-kick actor event (the SPADL `type_name == "goalkick"` action's `player_id`, the *acting* keeper, already in `actions` and more authoritative for that row — it beats a stale roster after a substitution). Both are driver-side, raw-JSON concerns; `providers/statsbomb` stays pure-shaping (ADR-054).

Three traps constrain any resolution (the ADR-062 D6 lessons): the synthetic `{0,1}` team fallback breaks the action↔frame join → every direction-dependent feature goes honest-NaN; identity ⟂ velocity (SB360 is velocity-less, so DAS/accessible-space stay NaN regardless of identity — ADR-063, naming the keeper does not make a velocity-constitutive metric score); and a mid-match GK substitution means the roster names only the *starter*.

## Decision

Make SB360 a **first-class, single-sourced tracking-feature provider** with four work components: (1) ONE keeper-identity resolver, `tracking.resolve_keeper_identities`, whose `identity="native"` path DELEGATES to the existing TF-13 `defending_gk_from_frames` / `acting_gk_from_frames` and whose `identity="roster"` path is the SB360 event-›roster-›derivation-›NA ladder — it returns a PURE `{(game, period, team) → KeeperIdentity(gk_id, source, conflict)}` map plus a conserving report, applied by two pure placement helpers (`add_defending_gk_player_id` on the action grain, `apply_keeper_identities_to_frames` as the identity→frame bridge); (2) a canonical `add_*` call convention pinned by a gate — `frames` is never keyword-only, a required fitted model may be the 3rd positional argument, and the single deviation `add_pre_shot_gk_angle` is fixed; (3) a provider-agnostic producer `tracking.run_tracking_features` that resolves keeper identity once, bridges it onto both grains, pre-links + shares one pitch-control cache, injects the caller-supplied `xt`/`xg_column`, and returns the enriched actions plus a report; (4) the ADR-054 `_defending_goal` stale-note pointer. Everything is dependency-inverted: the library consumes injected artifacts (roster dict, fitted `ExpectedThreat`, `xg_column`); the driver builds them.

## Alternatives considered

| Option | Pros | Cons | Why rejected |
|---|---|---|---|
| A. A gkdv-local keeper resolver (the in-flight TF-19 plan) | Local to its consumer | A second native-identity path (violates ADR-055 single-source); ADR-037 confines gkdv to `tracking._das`, so it could not serve the tracking GK families | The resolver must live in `tracking/` to serve `add_pre_shot_gk_*`; gkdv reaches it driver-side |
| B. Re-implement frame-based identity in the native path | Self-contained | A second copy of what `*_gk_from_frames` already does (ADR-055 fork) | Native path **delegates** to TF-13; only the SB360 roster/event ladder is new |
| C. Force every `add_*` to keyword-only `xt` (the spec's first "everything keyword-only") | Total uniformity | 108 call sites across 24 files churned for a type-guarded arg with no correctness payoff (a wrong 3rd-positional `ExpectedThreat` fails immediately on `xt.rate`) | YAGNI; the convention admits a required model as 3rd positional |
| D (chosen). One resolver (native delegates) + map-and-helpers + producer + a required-model-positional convention | Single-source; hexagonal; no gkdv fork; minimal signature churn | The producer's native `defending_gk_player_id` (map consensus) can differ from `add_pre_shot_gk_context` (event-primary) — a documented follow-up, not a regression | — |

## Consequences

### Positive

- SB360 GK-domain features (`add_pre_shot_gk_*`, `add_xt_gk`, `add_ghost_gk`) actually score once the roster is injected — the R1 identity→frame bridge stamps the resolved real id onto the anonymous keeper rows so `add_pre_shot_gk_position`'s `frame.player_id == defending_gk_player_id` match succeeds.
- One keeper-identity seam serves both the tracking GK families and (driver-side) the in-flight TF-19 GKDV cycle, which drops its planned gkdv-local resolver.
- One canonical call convention and one provider-agnostic producer; `scripts/_sb_battery.py`'s model-routing is single-sourced from the library's `FAMILY_MODEL_REQUIREMENTS`, so the audit and the producer can never disagree.
- Provider-agnostic where the concern is (the aggregators are); sportec/skillcorner/metrica get the producer for free.

### Negative

- **Retrain / Hyrum trigger (real, intended):** resolving keeper identity turns SB360 GK-domain features from honest-NaN into VALUES — a retrain trigger for any SB360 GK-feature consumer, and a Hyrum change for the normalised `add_pre_shot_gk_angle` signature. Non-GK / non-SB360 output is unchanged.
- The native-path `defending_gk_player_id` (a per-`(game, period, team)` consensus of `defending_gk_from_frames`) can differ from the established `spadl.utils.add_pre_shot_gk_context` (event-primary keeper + a frame fallback) where the event-primary keeper wins or at a mid-period sub. This is **not** a regression — the producer is a new entry point and does not change `add_pre_shot_gk_context` — and reconciling the two native paths is a documented follow-up.

### Neutral

- **Component 2 (deterministic snapshot id dtypes) carries NO work** — it already shipped in 4.79.0 (ADR-057/058); `_snapshot.py::_cast_to_declared_schema` casts ids to the declared nullable `Int64`, guarded by `test_snapshot_id_dtype_across_pandas.py`. Retained here only so the original survey's finding stays traceable.
- **C4 unchanged count:** `run_tracking_features` is an orchestrator (a `run_*`), not a new action-coupled aggregator — the "33 aggregators" count is unchanged; the C4 `tracking` container description gains the producer + resolver surface (prose only).
- Substitution-event keeper resolution (per-minute correctness beyond the roster starter + the goal-kick rung) and non-keeper SB360 identity remain foreclosed / follow-ups.

## Related

- **Specs:** `docs/superpowers/specs/2026-08-28-sb360-first-class-provider-design.md`
- **Plan:** `docs/superpowers/plans/2026-08-28-sb360-first-class-provider.md`
- **ADRs:** extends ADR-053 (SB360 coverage audit), ADR-054 (`providers/statsbomb` pure-shaping port + the `_defending_goal` stale note this ADR closes), ADR-055 (single-source discipline), ADR-037 (gkdv import confinement), ADR-057/058 (pandas-span + nullable id schema — Component 2), ADR-062 (opt-in FOV companions + derive-what-you-can-honestly identity), ADR-063 (velocity-availability honest-NaN), and TF-13 (`acting_gk_from_frames` / `defending_gk_from_frames`).
