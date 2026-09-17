# ADR-098: Uniform public metric-family output-contract registry (SK-EXPORT)

- **Status:** Accepted
- **Date:** 2026-09-17
- **Spec:** `docs/superpowers/specs/2026-09-17-sk-export-metric-output-contract-design.md`
- **Plan:** `docs/superpowers/plans/2026-09-17-sk-export-metric-output-contract-plan.md`

## Context

silly-kicks ships eight analyst-facing metric families. A downstream consumer (the luxury-lakehouse
adoption cycle) guards each gold mart's column contract with a schema-drift parity test that imports
the package's authoritative column/grain constants and asserts the mart equals them. That guard was
writable for only three families — `team_metrics`, `shot_stopping`, `restdefense` export both a
`*_METRIC_COLUMNS` and a grain `*_KEYS`. The other four kept them private in `_columns.py`, absent from
`__all__`, and the consumer will not pin a private module path (`docs/PRIVATE_CONSUMERS.md`: path pins
fail silently on rename).

A naive re-export would still leave three inconsistencies that block a *generic* consumer loop:
`gk_decision` has no `*_KEYS`; `MATCH_OUTCOME_METRIC_COLUMNS` is a `dict` while every other family's is
a list/tuple; per-package prefixes (`TEAM_KPI_`/`SS_`/`RD_`/…) are not derivable from the package name.

## Decision

Expose ONE uniform, discoverable, enforced output contract:

- **`silly_kicks.metric_contracts.METRIC_CONTRACTS`** — a plain `dict[str, MetricContract]` (the house
  "schemas are plain Python dicts" convention, cf. `SPADL_COLUMNS`), where `MetricContract` is a
  `TypedDict` (`keys`, `metric_columns`, `columns`, `column_types`). A `TypedDict` value is a plain dict
  at runtime — no new runtime class — with pyright-checked field keys. This is the canonical consumer
  surface: one loop, zero per-package name/type knowledge. Like `feature_glossary`, the registry
  **imports no metric package** — the values are hardcoded literals, so each family stays a pure
  dependency-free LEAF (its `test_import_allowlist.py` contract intact); `tests/test_metric_contracts.py`'s
  round-trip test CI-verifies the literals equal the packages' constants. This is **Option B**, chosen
  over runtime-importing precisely to preserve the six leaf-isolation contracts (runtime-sourcing would
  add an inbound edge to every metric family; the round-trip test already delivers the anti-drift a
  runtime import would).
- **Per-package constants stay public + are the mirror's source of truth:** re-export the four missing
  (`match_outcome`, `gk_decision`, `territory`, `duels`) into `__all__`; keep the domain prefixes
  (uniformity is provided by the registry, so renaming is churn — Chesterton's Fence).
- **`GK_DECISION_KEYS = ("game_id", "keeper")`** — the `summarize_gk_decision` grain (it groups
  `["keeper", "game_id"]`), NOT `player_id`.
- **`MATCH_OUTCOME_METRIC_COLUMNS` stays a `dict`** — the registry coerces (`tuple(...)` names +
  `column_types = dict(...)`). No package constant changes type → the release is purely additive.
- **`column_types` is populated for the five families whose `*_COLUMNS` is a typed `dict[str, str]`**
  (team_metrics/shot_stopping/territory/duels/match_outcome) and `None` for the two that are not
  (gk_decision/restdefense).
- **Enforcement** (`tests/test_metric_contracts.py`): per-entry self-consistency, `column_types`
  coverage, round-trip to the package constants, completeness-by-enumeration (a new family exporting a
  `*_METRIC_COLUMNS` must register; the ADR-056 three-bucket shape, `xsuccess` exempt as a VAEP rating
  method with no column set, `_UNDERIVABLE` empty), and output-faithfulness (keys/metric ⊆ the REAL
  declared output — the non-vacuous cross-check a plain round-trip cannot give).

## Consequences

- **Purely additive.** New module + re-exports + one new `GK_DECISION_KEYS` + a registry. No
  compute/behavior change; every numeric output byte-identical → no retrain, no re-materialize, no
  golden regen, C4-free.
- A consumer gets a uniform contract from one import; the four-family asymmetry cannot silently reappear
  (completeness gate).
- `xsuccess` is intentionally outside the registry; gaining a `*_METRIC_COLUMNS` later would fail the
  completeness gate and force a decision.
