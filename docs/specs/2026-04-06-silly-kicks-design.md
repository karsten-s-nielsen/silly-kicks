# silly-kicks: Maintained Successor to socceraction

**Date:** 2026-04-06
**Status:** Draft
**Repo:** `https://github.com/karsten-s-nielsen/silly-kicks` (private)
**Local folder:** `D:\Development\karstenskyt__silly-kicks`

## Summary

`silly-kicks` is an independently maintained successor to
[ML-KULeuven/socceraction](https://github.com/ML-KULeuven/socceraction) — the
Python library for converting raw football event data into the SPADL action
representation and valuing actions via VAEP. The upstream project is effectively
unmaintained (last substantive commit Nov 2024, README explicitly states feature
requests will not be accepted, Issue #37 on keeper event definitions open 5+
years with zero response). No other maintained fork or alternative exists on
PyPI or GitHub.

**Name inspiration:** Monty Python's *Ministry of Silly Walks* — a government
ministry that classifies and values different types of walks (movements). One
permutation away: "kicks" instead of "walks," quietly signaling the football
domain to anyone in the know.

## Project Identity

| Surface | Form |
|---------|------|
| GitHub repo | `karsten-s-nielsen/silly-kicks` |
| PyPI package | `silly-kicks` |
| Python import | `import silly_kicks` |
| Submodules | `silly_kicks.spadl`, `silly_kicks.vaep`, `silly_kicks.atomic` |
| License | MIT (preserved from upstream, original copyright retained) |
| Python | `>=3.10` |

### Attribution

README opens with clear provenance:

> *Originally developed as socceraction by Tom Decroos and Pieter Robberechts at
> KU Leuven. This project is an independently maintained successor, built under
> the MIT license. The original BibTeX citations are preserved below.*

The original VAEP and SPADL paper citations are preserved and prominently placed.

### Audience

The library is independent and general-purpose — not a luxury-lakehouse-specific
tool. Target audience: any soccer analytics practitioner who needs a maintained,
production-grade SPADL+VAEP implementation. Immediately consumed by
luxury-lakehouse, but designed for community adoption.

## Scope

### What ships in `silly-kicks`

- `silly_kicks.spadl` — SPADL schema definition, provider converters (StatsBomb,
  Wyscout, Opta, kloppy), action type / result / bodypart config
- `silly_kicks.vaep` — VAEP feature extraction, label generation, valuation
  formula, model interface
- `silly_kicks.atomic` — atomic (continuous) SPADL representation (carried
  forward from socceraction)

### What stays out (remains in luxury-lakehouse)

- Spark/Delta integration, `applyInPandas` patterns
- The `_TYPE_KEY_OVERRIDES` workaround (becomes unnecessary once silly-kicks
  fixes the root cause)
- HF Hub publishing, MLflow, platform-specific integrations
- Any I/O, storage, or orchestration concerns

## Architectural Constraints

### Hexagonal Architecture (Spark-Friendliness Contract)

The library is a pure domain core. It has no awareness of Spark, Delta, HF Hub,
or any infrastructure. Downstream consumers provide the adapters.

**1. Pure pandas/numpy interface, zero I/O.**
Every public function takes pandas DataFrames (or scalars) in and returns pandas
DataFrames out. No file reads, no network calls, no database access, no global
state mutation.

**2. Lightweight, deferrable imports.**
The package must be importable inside a Spark UDF closure without pulling in
heavy dependencies. Core dependencies limited to: `pandas`, `numpy`. No eager
imports of large frameworks at package level. Import cost target: <50ms on a
Spark executor.

**3. Stateless functions, no singletons.**
All core functions are pure:
- `convert_to_actions(df, home_team_id)` — pure function
- `add_names(df)` — pure lookup
- `gamestates(df, nb_prev_actions=3)` — pure windowing
- Feature functions — pure transforms
- `value(actions, p_scores, p_concedes)` — pure arithmetic

No module-level caches, no `_instance` patterns, no lazy initialization that
mutates global state.

**4. Bounded-input compatible.**
All functions work correctly on per-game DataFrames (~1,600 rows). No function
requires or assumes access to the full dataset.

**5. Serialization-safe configuration.**
Configuration (action type mappings, feature function lists, column schemas)
exposed as frozen dataclasses or plain dicts — never closures, lambdas, or
objects with `__dict__` state. Safe for Spark UDF closure capture.

### Integration Pattern (luxury-lakehouse)

The current integration pattern — lazy import inside UDF body, call pure
functions on per-game pandas DataFrames, pass model bytes separately — continues
to work as-is. The library provides the "inside" of the hexagon; luxury-lakehouse
provides the Spark/Delta/HF adapters on the "outside."

## The "Nothing Left Behind" Guarantee

### Problem

socceraction silently drops events via a `non_action` sentinel pattern. Every
converter maps unrecognized events to `non_action` (type ID 20) and then filters
them out — no log, no count, no warning. This has caused repeated re-ingestion
in luxury-lakehouse when dropped events were discovered after the fact.

### Silent data loss in socceraction v1.5.3

| Event category | StatsBomb | Wyscout | Opta | Kloppy |
|---|---|---|---|---|
| `keeper_claim` | Produced | **Never produced** | Produced | Produced |
| `keeper_punch` | Produced | **Never produced** | Produced | Produced |
| `keeper_pick_up` | **Never produced** | **Never produced** | Produced | Produced |
| Aerial duels | **Dropped** | **Dropped** | No mapping | **Dropped** |
| Loose-ball duels | **Dropped** | **Dropped** | **Dropped** | **Dropped** |
| Blocks / deflections | **Dropped** | N/A | **Dropped** | **Dropped** |
| Fifty-fifty contests | **Dropped** | N/A | N/A | **Dropped** |
| Keeper saves after goals | Kept | **Dropped** | Kept | Kept |
| Injury clearance passes | **Dropped** | N/A | N/A | N/A |
| Fouls won (fouled player) | N/A | N/A | **Dropped** | N/A |

### Principle: Every source event has an explicit, audited fate

For each provider, `silly-kicks` maintains a **complete mapping registry** —
every source event type is explicitly listed with one of three outcomes:

1. **Mapped** — converts to a specific SPADL action type (with full sub-type
   routing)
2. **Intentionally excluded** — documented reason why (e.g., "Substitution is a
   non-on-ball meta-event")
3. **Unrecognized** — logged with event type name and count so the consumer
   knows something new appeared in the data

No event type silently falls through to a catch-all. The `else -> non_action`
pattern is replaced with an explicit registry.

### Conversion audit report

Every call to `convert_to_actions()` can optionally return a conversion summary:

- Total source events in
- Total SPADL actions out
- Count per mapped type
- Count per intentionally-excluded type
- Count and names of any unrecognized types (zero in a healthy run)

When a data provider adds a new event type, pipelines flag it immediately rather
than silently dropping rows.

### Vocabulary extension (conservative)

Where the original 23-type SPADL vocabulary has gaps that affect real analytics,
`silly-kicks` extends it — with academic justification and backward compatibility
(original type IDs 0-22 preserved). Immediate candidates:

- Consistent `keeper_claim` / `keeper_punch` / `keeper_pick_up` across **all**
  providers (D44, Issue #37)
- Aerial duels (currently universally dropped — relevant for GK analytics, set
  piece analysis)

Extensions are additive (new type IDs > 22), never redefining existing IDs.
Specific new action types are decided during Phase 3 triage based on what the
audited codebase reveals and what downstream analytics actually need.

## Phased Approach

### Phase 1: Clean Import (mechanical, zero behavioral changes)

1. Clone upstream socceraction into `karstenskyt__silly-kicks`
2. Strip `.git` history (fresh repo, not a GitHub fork)
3. Rename package: `socceraction/` -> `silly_kicks/`, update all internal imports
4. Modern project scaffolding:
   - `pyproject.toml` (hatchling build system)
   - Ruff config (E, W, F, I, N, UP, B, S, RUF — matching luxury-lakehouse)
   - Pyright basic mode
   - GitHub Actions CI
   - Line length: 120
5. Attribution: LICENSE (preserve MIT + original copyright), README with
   provenance and BibTeX citations
6. **Gate: all existing upstream tests pass under the new package name**

### Phase 2: Modernize & Audit (structural quality, zero behavioral changes)

1. Run mad scientist audit suite (architecture, security, optimization,
   documentation)
2. Apply findings: type annotations, Python 3.10+ idioms, clean up
   "HERE BE DRAGONS" code
3. Fix or drop the `pandera`/`multimethod` dependency chain
4. Enforce hexagonal architecture contract (verify: no global state, no I/O)
5. **Gate: all existing tests still pass, behavior identical to upstream**

### Phase 3: Fix Known Issues (behavioral changes, each with tests)

1. Triage upstream issues + PRs: review all 21 open issues and 2 human PRs on
   ML-KULeuven/socceraction — still relevant? Already fixed? Worth fixing?
2. D44: Wyscout `keeper_claim` / `keeper_punch` / `keeper_pick_up`
   differentiation
3. Issue #37: consistent keeper event definitions across all providers
4. "Nothing Left Behind" guarantee: replace `non_action` black hole with
   explicit mapping registries + conversion audit reports
5. Provider-specific fixes: each fix gets its own commit with tests

### Phase 4: Integrate (back in luxury-lakehouse)

1. Publish `silly-kicks` to PyPI
2. Update luxury-lakehouse `pyproject.toml`:
   `socceraction==1.5.3` -> `silly-kicks>=0.1.0,<1.0`
3. Update imports: `socceraction.spadl` -> `silly_kicks.spadl`, etc.
4. Remove workarounds (`_TYPE_KEY_OVERRIDES`, defensive coercions no longer
   needed)
5. Re-run full pipeline to verify end-to-end

## Upstream Issues & PRs to Triage (Phase 3)

The following will be reviewed against the `silly-kicks` codebase after Phase 2
modernization. Items may already be resolved by Phase 2 cleanup or may no longer
apply.

### Open Issues (21 on upstream)

- **#37** — Create consistent keeper event definitions in SPADL (5+ years open,
  filed by maintainer, zero comments) — **directly relevant, high priority**
- **#914** — Cannot import name 'overload' from 'multimethod' — **likely
  resolved by Phase 2 dependency cleanup**
- Remaining issues to be triaged during Phase 3 against the modernized codebase

### Open Human PRs (2 on upstream)

- **#948** — WhoScored dict parser (Jul 2025, no review) — evaluate relevance
- **#946** — pandas fillna deprecation fix (Jul 2025, no review) — likely
  already addressed by Phase 2 modernization

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| Upstream becomes active again | MIT license, clear attribution, offer to upstream fixes |
| Community confusion (two packages) | README explains relationship, PyPI description is clear |
| Maintenance burden of carrying a fork | Scope is bounded (~3K LOC), luxury-lakehouse CI catches regressions |
| Breaking changes for luxury-lakehouse | Phase 4 integration is a controlled migration with full pipeline re-run |
| New provider event types appear | "Nothing Left Behind" guarantee flags unrecognized types immediately |

## Success Criteria

- [ ] `silly-kicks` published on PyPI with all existing socceraction tests passing
- [ ] Zero silent data loss: every provider event type has an explicit fate
- [ ] Consistent keeper event production across all providers
- [ ] luxury-lakehouse migrated with full pipeline verification
- [ ] No re-ingestion surprises: conversion audit reports catch new event types
