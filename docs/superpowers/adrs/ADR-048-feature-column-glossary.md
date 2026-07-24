# ADR-048: Feature-column glossary + `describe_level`

| Field | Value |
|---|---|
| **Date** | 2026-07-24 |
| **Status** | Accepted |
| **Deciders** | Karsten S. Nielsen |

## Context

silly-kicks emits ~340 *derived* feature columns (across `add_*` aggregators, `*_xfns` VAEP
transformers, atomic mirrors, spadl enrichers, and vaep features), but nothing documented what each
column means, its unit, where its logic lives, or which methodology it implements. A downstream
wordalisation / reporting layer needs that metadata to be machine-readable and *trustworthy*, and
nothing forced a newly-added column to be documented — the classic silent-rot surface.

## Decision

Ship a **pure Python registry** — `silly_kicks/feature_glossary.py` — of `FeatureColumn(name,
definition, unit, emitting_module, attribution, higher_is_better)` records keyed by exact base column
name, made **complete-by-construction** by an inspection-based CI coverage gate, plus a generic
direction-aware `describe_level(z, *, higher_is_better=True)` z-score→verbal-band helper in a separate
`silly_kicks/reporting.py`.

Key choices:
- **Registry, not a data file** (house idiom: schemas are plain Python dicts). `unit` is a closed
  `Literal` vocabulary (typo/ad-hoc unit fails type-check). JSON export (`glossary_to_json` pure /
  `dump_glossary` thin writer) carries a `GLOSSARY_SCHEMA_VERSION`.
- **`emitting_module` = the metric's HOME/computation module** (`_packing`, `_obso`, …), NOT the
  `features.py` monolith where every public tracking producer is *defined* (`__module__` is uniformly
  `…features`, so run-and-diff attribution is impossible). The gate checks importability + not-`.features`
  (with a conscious `_FEATURES_HOMED_ALLOWLIST` escape); beyond that the field is documentation, not
  gate-verified. This is the honest resolution of the impossibility surfaced in spec review 2.
- **Coverage enumerated by INSPECTION with an `__all__`-less fallback** (`discover_public_column_producers`,
  mirroring `conftest_id_scalar._public_names`), NOT `__all__` alone — the id-scalar registry already paid
  for that lesson. The load-bearing completeness dependency is the `add_*`/`*_xfns` NAME SHAPE, with vaep's
  `fs.*` the recorded exception (enumerated by the default-list run-and-diff leg). A red-first `__all__`-less
  meta-test proves the fallback is live.
- **`attribution` ↔ NOTICE hard gate**: every non-None token appears verbatim in `NOTICE` (enforces
  ADR-005, makes the field trustworthy).
- **Base schema columns excluded** (`SPADL_COLUMNS` / `ATOMIC_SPADL_COLUMNS` / `TRACKING_FRAMES_COLUMNS` +
  linkage provenance): the glossary is the *derived* surface only.
- **Direction lives on the glossary** (`higher_is_better: bool | None`); `describe_level` consumes it and
  flips internally, so a lower-is-better metric is not mislabelled. The "which columns are auto-describable"
  policy + raw→z-score normalisation seam belong to the future wordalisation layer (§ non-goals).

## Alternatives considered

| Option | Why rejected |
|---|---|
| JSON/YAML data-file source of truth | House idiom is Python dicts; registry is type-checkable, parse-free, JSON-exportable. |
| Co-located per-module metadata aggregated centrally | Lower drift, but forces `feature_glossary.py` to import every producer module (heavy coupling); deferred, drift mitigated by typed `unit` + importable-`emitting_module` + review. |
| `describe_level` co-located in `feature_glossary.py` | Generic transform, different responsibility/cadence; mis-homes it. |
| `attribution`→NOTICE soft convention | "A count nothing checks" is the silent-guard failure mode. |
| Enumerate coverage from `__all__` | Reintroduces the id-scalar `__all__` blind spot. |
| `emitting_module` == run-and-diff-attributed module | Impossible against the monolithic `features.py`; downgraded to importable + not-`.features`. |

## Consequences

### Positive
- Every derived feature column is discoverable, machine-readable, and NOTICE-linked; a new
  `add_*`/`*_xfns` column with no entry fails CI (anti-rot), and a stale entry fails too.
- The 171 combinatorial VAEP one-hots are *generated* from the spadlconfig vocabularies (`_onehot_entries`),
  not hand-written; the ~170 unique features are hand-authored with home modules + citations.
- **Additive documentation surface → no VAEP retrain, C4 count unchanged (31).**

### Negative / limitations
- `emitting_module` correctness beyond importable + not-`.features` relies on review (documentation, not
  gate-verified) — the monolith makes free attribution impossible.
- Per-leg non-vacuity anchors catch a fully-stubbed harness leg, NOT a partial one (would need a second
  independent enumeration) — recorded in the harness docstring.
- A public column-producer named outside the `add_*`/`*_xfns` shape and not in the recorded exception set is
  invisible to discovery — a documented limitation, not dressed as anti-rot.

## References
- Spec: `docs/superpowers/specs/2026-07-24-feature-glossary-and-xfns-pc-cache-design.md`
- Plan: `docs/superpowers/plans/2026-07-24-feature-glossary-and-xfns-pc-cache.md`
- ADR-005 (academic attribution), ADR-019 (id-compat / inspection-enumeration lesson), ADR-003 (NaN-safe).
