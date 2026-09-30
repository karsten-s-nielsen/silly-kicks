# How-to: authoring a metric

> Class-2 procedural runbook (same tier as `docs/context`, sibling to it under `docs/howto`). The
> durable RULES live in `AGENTS.md` §Key conventions; the WHY/measurement lives in
> `docs/context/conventions-core.md`. This file is the **checklist** — the one place that says, in
> order, what a new metric must wire so a session stops rediscovering it. It CITES the enforcing ADRs
> and tests; it does not copy their bodies (they are the source of truth and drift if duplicated).

A "metric" here is any new derived quantity that ships columns: an action-coupled enricher (`add_*`)
or a standalone aggregator (`compute_*`). Work the steps top-to-bottom; each names the gate that
reds if you skip it.

## 0. `add_*` vs `compute_*` — pick the shape first

- **`add_*`** — action-coupled: takes a SPADL/tracking frame, returns a NEW frame with extra
  columns (one row per input action). It ENRICHES an existing table. Pure (pandas in, pandas out,
  zero mutation). This is the enricher family gated by ADR-033 (purity) and, for tracking, by the
  liveness + call-convention registries (steps 5–6).
- **`compute_*`** — standalone aggregator: takes inputs, returns a fresh summary table at its own
  grain (e.g. per-keeper, per-team). It does not enrich the caller's frame.
- **Aggregator bookkeeping.** An action-coupled `add_*` aggregator that adds a new modelled subpackage
  changes the C4 `tracking` aggregator COUNT — the documented number in `AGENTS.md`'s TF table and the
  C4 model must move together (step 8; ADR-048 C4-pinning, `docs/context/conventions-core.md`).

Everything below assumes you know which shape you are building.

## 1. Package layout

A column-emitting metric family is a package `silly_kicks/<family>/` with focused modules, not one
file. The minimal exemplar is `silly_kicks/shot_stopping/`:

```
<family>/__init__.py     # the PUBLIC surface — __all__ exports the compute fn + the column/grain constants
<family>/_config.py      # params / config dataclass
<family>/_compute.py      # the pure engine (or _engine/_numba for a fused-njit kernel, cf. _das_engine/_das_numba)
<family>/_columns.py     # *_METRIC_COLUMNS + grain *_KEYS (+ column_types when the columns are a typed dict)
<family>/_report.py      # rendering / summary
```

Add family-specific modules as the compute demands — `territory/` adds `_counterfactual.py` + `_hull.py`,
`gk_decision/` adds `_optionset.py`/`_reconstruct.py`/`_value.py`, `duels/` adds `_extract.py`. A tracking
`add_*` feature instead lives as a single `silly_kicks/tracking/_<family>.py` module in the TF table
(`AGENTS.md`). Private modules (`_*.py`) can have downstream pins — check `docs/PRIVATE_CONSUMERS.md`
before renaming one; path pins fail SILENTLY (ADR-019 note, `docs/context/conventions-core.md`).

## 2. Tests mirror the package + a corpus-driver validity test

- Unit tests mirror the package path: `tests/<family>/test_*.py` (tracking → `tests/tracking/`).
- Real-data validation lands as a `scripts/validate_<family>.py` corpus driver + a
  `tests/scripts/test_*` guard; see `docs/howto/corpus-drivers-runbook.md` for the `_driver.py` seam
  and `docs/howto/construct-validity.md` for the GO/NO-GO gate shape.
- **Fixture placement:** shared fixtures under `tests/fixtures/` or the family's `tests/<family>/`
  conftest. A committed fixture consumed by a generator must be reproduced BYTE-FOR-BYTE by that
  generator before any reshape (ADR-056; `docs/context/corpus-drivers.md`). A checksummed fixture
  needs `.gitattributes binary`.

## 3. Register the output contract (ADR-098)

Every column-emitting family single-sources its contract in `silly_kicks.metric_contracts`:

- Export `<FAMILY>_METRIC_COLUMNS` + a grain `<FAMILY>_KEYS` publicly in the family `__init__.__all__`.
- Add a `MetricContract` entry to `METRIC_CONTRACTS` (a plain `dict` valued by a `TypedDict` with
  fields `keys` / `metric_columns` / `columns` / `column_types`). Populate `column_types` only when the
  family's `*_COLUMNS` is a typed `dict[str, str]`; else `None`.
- The registry imports NO metric package (values are hardcoded literals, keeping each family a pure
  leaf); `tests/test_metric_contracts.py` round-trips the literals against the package constants and
  enforces completeness-by-enumeration — **a new column-emitting family that does not register reds
  this test** (`_UNDERIVABLE` asserted empty; `xsuccess` is the one documented exemption).

## 4. `add_*` → purity registry (ADR-033)

Every public `add_*` must NOT mutate any caller-supplied DataFrame/Series/ndarray and MUST return a
NEW object. Register it in `PURITY_ENTRIES` (key `"<package>:<add_name>"`) in
`tests/test_add_star_purity.py`. Two meta-assertions pin the gate surface to the public export
(`__all__` UNION `.features.__all__`), so an unregistered public `add_*` reds CI.

**Contributor contract (the real backstop):** any `add_*` that CONDITIONALLY adds columns (a
present/absent branch) MUST register **≥2 purity variants** — one per branch. The AST heuristic only
nudges toward the one known bug shape; it is not a proof. Also decorate NaN-tolerant enrichers with
`@nan_safe_enrichment` (`silly_kicks._nan_safety`; ADR-003, gated by
`tests/test_enrichment_nan_safety.py`).

## 5. Tracking `add_*` → liveness gate + call-convention registry

- **Liveness (non-NaN AND non-constant):** register in `tests/tracking/test_aggregator_column_liveness.py`.
  Every column the `add_*` adds must be non-null on the multi-domain fixture, and every float metric
  column with ≥2 observations must carry >1 distinct value. A by-design constant goes in
  `STRUCTURAL_CONSTANTS` WITH a justification + dedicated invariant test — never a silent exclusion
  (the registry is currently empty). A meta-assertion pins the gate to `tracking.__all__`, so a new
  `add_*` that wires no liveness entry reds CI.
- **Call convention (ADR-078):** a frame-consuming `add_*` has ONE call shape (`frames`
  positional-or-keyword; optional kwargs keyword-only) — register in
  `tests/tracking/test_call_convention_registry.py`.

## 6. Liveness FIXTURE precondition (verbatim lesson)

Anchor this, from `docs/context/conventions-core.md`:

> **A liveness gate's FIXTURE needs its own precondition test (ADR-032 idiom, re-learned 4.74.0).**

A gate is only as good as the rows it scores. Pin the fixture's validity SEPARATELY (in-domain, no
constant feature, enough rows) and assert only what the guarded thing actually does — two
statistically equivalent models once scored AUC 1.0 and 0.0 on a degenerate fixture whose `vx`/`vy`
were zeroed.

## 7. `*_xfns` factory decision (leak guard)

If the feature reads its own action's `result_id` or any POST-CONTACT outcome, it ships **no
`*_xfns` factory at all**, or stays out of every default xfn list (ADR-030/047/049). Opting a leaky
factory into a default list is a HybridVAEP-class correctness break, not a tuning choice. Enforced by
an auto-discovering absence guard anchored on the transformer NAME (`docs/context/conventions-core.md`).

## 8. Glossary + attribution + C4

- **`feature_glossary`:** every derived column gets a `FeatureColumn(name, definition, unit,
  emitting_module, attribution, higher_is_better)` record in `silly_kicks/feature_glossary.py`;
  `emitting_module` is the metric's HOME compute module, not `features.py`. Coverage is CI-gated
  (ADR-048) — a new `add_*`/`*_xfns` that documents no columns reds the coverage gate. Direction lives
  with `describe_level` in `silly_kicks/reporting.py`.
- **`NOTICE`:** a feature implementing a published methodology gets a `NOTICE` entry, cross-linked
  from the docstring (`See NOTICE for full bibliographic citations.`; ADR-005).
- **C4:** a new action-coupled `add_*` aggregator or subpackage changes the documented count — update
  `docs/c4/architecture.{dsl,html}` (regen via `mad-scientist-skills:c4`) and the `AGENTS.md` TF-table
  count together (ADR-048).

## 9. Value-changing vs additive → version, Hyrum, re-materialize

- **Additive** (new columns, no existing byte changes): purely additive release — no retrain, no
  re-materialize, C4-free EXCEPT a count bump.
- **Value-changing** (any existing output byte moves): bump the version in the SINGLE source
  `silly_kicks/_version.py` (ADR-079 — that one file; `uv.lock` follows from `uv lock`, never
  hand-edited), record a **Hyrum notice** for downstream consumers (a changed column/return/log/path
  is an API break even when the public API did not; `CLAUDE.md` glossary), and flag the **lakehouse
  re-materialize** owed for any persisted mart the change touches.

## Anchor: silent-null defects share one shape (verbatim lesson)

From `docs/context/conventions-core.md` — the reason every band and counterfactual in step 5/6 needs
a two-sided, non-vacuous assertion:

> **Every band needs a test from BOTH sides, and every counterfactual needs a non-vacuity assertion
> that it actually moved something.** ... **Four silent-null defects in this codebase share exactly
> one shape** — the kloppy tracking y-inversion (ADR-031), the fabricated `flat_zones` grid origin
> (ADR-036/PR-S113), the identity-keyed `PitchControlCache` served to a moved-GK counterfactual
> (ADR-043), and a mirrored external-provider event frame — each produced a plausible number from a
> computation that had not happened.

Assert the failing side too (a mutation that SHOULD move the number out of the band), and assert the
counterfactual measurably differs from its factual twin.
