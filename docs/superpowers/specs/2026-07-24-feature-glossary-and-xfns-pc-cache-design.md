# Feature-Column Glossary + `describe_level`, and TF-7 xfns Pitch-Control Cache — Design

| Field | Value |
|---|---|
| **Date** | 2026-07-24 |
| **Status** | Draft (rev 3 — folds in the analysis-session second review: emitting_module semantics + gate) |
| **Deciders** | Karsten S. Nielsen |
| **Version / PR-S / ADR** | assigned at commit-prep (not pre-claimed — [[no-version-number-until-commit-prep]]) |

## Context

Two independent, bounded sub-features ship together in **one branch / commit / PR** (owner-directed
scope call — see the delivery note in Cross-cutting for the bisectability trade-off):

1. **Feature-column glossary + `describe_level`** — a machine-readable catalogue of every *derived*
   feature column silly-kicks emits (name, definition, unit, emitting module, attribution, direction),
   enforced by an auto-enumerating CI gate, plus a generic `describe_level(z, *, higher_is_better=True)`
   z-score→verbal-band helper that seeds the reporting/wordalisation layer.
2. **TF-7 cross-family pitch-control cache in the VAEP `*_xfns` path** — the `add_*` aggregators already
   share a `PitchControlCache` (ADR-008, 3.25.0), but the `*_xfns` VAEP transformers do not, so a
   multi-family xfn list recomputes the same per-frame surfaces once per family. Extend the cache to the
   xfns path via caller-injection (the seam already exists in `add_obso`/`xcross_attempt_xfns`).

The two are unrelated in mechanism; they share a cycle only for delivery convenience. Two self-contained
sub-designs with a shared cross-cutting section.

## Goals / Non-goals

**Goals.**
- Every derived feature column has a discoverable, machine-readable definition + unit + provenance + direction.
- A CI gate makes the glossary complete-by-construction: a new `add_*`/`*_xfns` column with no entry fails
  CI, enumerated **by inspection with an `__all__`-less fallback** (the robustness pattern the id-scalar
  registry paid for) so a future `__all__`-less producer module cannot slip through — with the name-shape
  convention as the actual completeness dependency for this surface (see §1.3).
- Each cited methodology resolves to a real `NOTICE` entry (ADR-005, enforced).
- `describe_level` gives report/wordalisation consumers one canonical, NaN-safe, **direction-aware** z-bucketing.
- One shared `PitchControlCache` reused across all pitch-control-consuming `*_xfns` in a single VAEP pass,
  **value-identical** to today, with the cache-wiring **completeness-gated** so a future PC-consuming family
  cannot silently miss it.

**Non-goals (v1).**
- Documenting base schema columns (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`, `TRACKING_FRAMES_COLUMNS`) —
  those are the converter contract, gated elsewhere; the glossary covers *derived* columns only.
- Documenting non-default method variants exhaustively (e.g. `pitch_control_at_target__voronoi`); v1
  documents the default-config surface the default xfn lists / default aggregator calls emit.
- A wordalisation/reporting narrative engine. Only the `describe_level` primitive is in scope; the policy
  of *which* columns are auto-describable and how *raw* columns are normalised to z-scores before
  `describe_level` (the range/normalisation seam, §1.6) belong to that future layer.
- **Exhaustive** per-column direction authoring. `FeatureColumn.higher_is_better` is filled where a column
  has an unambiguous direction and left `None` (perspective-dependent / not asserted) otherwise; `describe_level`
  is correct *given* a direction, and the wordalisation layer (v2) owns the "only auto-describe known-direction
  columns" policy.

---

## Sub-design #1 — feature-column glossary + `describe_level`

### 1.1 The registry (`silly_kicks/feature_glossary.py`, new top-level public module)

Mirrors the existing top-level public utility modules (`silly_kicks/id_compat.py`, `silly_kicks/reflection.py`).

```python
from dataclasses import dataclass
from typing import Literal

Unit = Literal[
    "metres", "m^2", "m/s", "seconds", "degrees",
    "probability", "count", "xT", "xG", "ratio", "dimensionless",
]   # closed vocabulary -- a typo or ad-hoc unit fails type-check, not just review (drift guard)

@dataclass(frozen=True)
class FeatureColumn:
    name: str                       # exact base/semantic emitted column name (the glossary key)
    definition: str                 # one-sentence, first-time-reader-interpretable definition
    unit: Unit                      # closed vocabulary above
    emitting_module: str            # metric's home/computation module (where the logic lives), e.g. "silly_kicks.tracking._packing"
    attribution: str | None = None  # None = house-original; else a citation TOKEN present verbatim in NOTICE
    higher_is_better: bool | None = None  # direction for describe_level; None = perspective-dependent / not asserted

FEATURE_GLOSSARY: dict[str, FeatureColumn]  # single source of truth, keyed by FeatureColumn.name
GLOSSARY_SCHEMA_VERSION = "1.0"             # frozen once dump_glossary output has external consumers (Hyrum)
```

Accessors (all pure):
- `glossary_entry(name) -> FeatureColumn` — raises `KeyError` on an unknown column.
- `undocumented_columns(cols) -> set[str]` — the caller's columns with no entry (shared by the gate + ad-hoc callers).
- `glossary_to_json() -> str` — **pure**; emits `{"schema_version": GLOSSARY_SCHEMA_VERSION, "columns": {...}}`.
- `dump_glossary(path) -> None` — a **thin writer** over `glossary_to_json()` (the only impure symbol in the module).

`emitting_module` is a **string**, not an import — the registry stays a pure data structure with no
import-time dependency on producer modules (keeps import light, avoids cycles). It names the metric's
**home/computation module** — where a reader finds the logic (`_packing`, `_obso`, …), or `features.py`
for a metric with no separate compute module — **not** necessarily where the public producer is *defined*:
every public tracking `add_*`/`*_xfns` is defined in the monolithic `features.py` (`add_obso`, `obso_xfns`,
`add_packing` all report `__module__ == "silly_kicks.tracking.features"`), so run-and-diff cannot distinguish
home modules. `emitting_module` is therefore **authored provenance**, gate-checked only for **importability**
(no dead references, §1.3 assertion 3), not for being the *right* home — that relies on review, like
`definition`. The residual drift risk is bounded by: typed `unit`, and the coverage gate (a column rename
fires assertions 1–2). Fully co-locating the metadata next to each producer is the lower-drift long-term
shape but reintroduces heavy import coupling; deferred (Alt 1f), not adopted for v1.

**Column naming — base semantic name.** `add_*` aggregators emit stable exact names (documented verbatim).
`*_xfns` emit one column per gamestate slot (slot marker appended); the glossary keys on the **base semantic
name** (slot marker stripped), so one entry covers the `add_*` column and its xfns per-slot siblings.
Method-parameterised families key on the `<feature>__<method>` emitted name (ADR-005 §8), at the default method.

### 1.2 `describe_level` (`silly_kicks/reporting.py`, new small public module)

Deliberately **not** in `feature_glossary.py`: it is a generic z-score→label transform that never reads
`FEATURE_GLOSSARY`; different responsibility + change cadence; a caller bucketing a non-feature z-score
should not import from a "feature glossary" module. `reporting.py` is the correctly-named seed of the
reporting/wordalisation layer.

```python
def describe_level(z, *, higher_is_better=True):
    # z: scalar float | ndarray | pd.Series ; higher_is_better: bool
    # Effective score = z if higher_is_better else -z, then the staircase (upper-open bands):
    #   >= 1.5  -> "outstanding"
    #   >= 1.0  -> "excellent"
    #   >= 0.5  -> "good"
    #   >= -0.5 -> "average"
    #   >= -1.0 -> "below average"
    #   else    -> "poor"
    #   NaN     -> "unknown"      (ADR-003 NaN-safe; NaN never raises, never mislabels)
```

- **Direction-aware:** `higher_is_better=False` flips the sign internally, so a lower-is-better metric
  (turnovers, times-beaten) at high z is correctly "poor", not "outstanding". The caller supplies direction
  (from `FeatureColumn.higher_is_better`, or explicitly).
- Scalar in → `str`; ndarray/Series in → object ndarray/Series (Series index preserved).
- Vectorised (`np.select`-style, no `apply`); NaN handled explicitly before the comparisons.
- Boundaries `>=` (a value exactly at a threshold takes the higher band); pinned by unit tests both sides.
- **Cutoff note:** these are the owner-specified coach-facing *descriptive* bands (provisional, adjustable),
  read as relative-to-cohort, not absolute quality — e.g. `z≥0.5`→"good" is ≈69th percentile ("above average").
  A docstring states this so a consumer does not over-read the words.

### 1.3 Coverage CI gate (`tests/test_feature_glossary_coverage.py`) — the load-bearing part

Makes the glossary complete-by-construction, enumerated **by inspection, not `__all__` alone**.

**Producer discovery** reuses the id-scalar registry's proven idiom (`tests/invariants/conftest_id_scalar.py`
`_public_names`/`discover_public_id_scalar_functions`): walk each package's public modules; take a module's
surface from `__all__` **when declared, else infer public callables via `vars(mod)` filtered on
`__module__ == mod.__name__`** (so an `__all__`-less producer module still contributes, and imported names
do not). Filter to producer functions by name shape (`add_*` / `*_xfns`). Packages: `tracking`,
`atomic.tracking`, `spadl`, `atomic.spadl`, `vaep`.

The `__all__`-less fallback is **defensive** here, not a fix for a live blind spot: all current
`add_*`/`*_xfns` are re-exported in their package `__all__`, so the fallback catches nothing new today
(unlike the id-scalar case, where 13 callables sat in `__all__`-less submodules). It is kept as
Chesterton's-fence robustness against a future `__all__`-less producer module. The load-bearing dependency
this creates is instead the **name-shape convention**: discovery finds a producer only if it is named
`add_*`/`*_xfns`. That holds for all public `tracking`/`spadl`/`atomic` column-producers; `vaep`'s `fs.*`
feature functions (`fs.actiontype_onehot` … `fs.goalscore`) do **not** match, so they are enumerated by the
separate default-list invocation below (the one explicit exception). A guard test pins that exception set
(`_NON_CONFORMING_PRODUCERS`, initially the vaep default lists) so a new non-conforming public column-producer
must be consciously added — the name-shape convention is thereby itself guarded, not an unspoken assumption.

**Emitted-column harnesses (run-and-diff).** Each producer family is invoked at its default config on a
fixture and its added columns collected (emitted − input). This is **five distinct run-and-diff harnesses
with distinct fixtures**, not one — tracking `add_*` reuses the liveness fixture; `*_xfns` need a gamestates
fixture; spadl/`atomic.spadl` enrichers need a SPADL/atomic fixture; vaep needs the default vaep xfn lists
(`vaep.base.xfns_default`, `vaep.hybrid.hybrid_xfns_default`) on a gamestates fixture. This is real harness
work + CI-time cost, called out here so it is not discovered mid-implementation. Per-slot xfn columns
normalise to the base name.

**Assertions:**
1. **No undocumented columns** — every emitted base column has a `FEATURE_GLOSSARY` entry.
2. **No stale entries** — every entry maps to a real emitted base column.
3. **`emitting_module` importable** — every entry's `emitting_module` imports cleanly (no dead references,
   catches typos). It is NOT asserted equal to the run-and-diff-attributed module: the monolithic
   `features.py` means every tracking producer's `__module__` is `features`, so an equality check would
   force the whole tracking catalogue to `features.py` and destroy the field's provenance value (see §1.1).
   Home-module correctness relies on review; no run-and-diff *attribution* is therefore needed (the vaep leg
   can keep its default-list invocation, no per-function runs).
4. **Meta / anti-rot** — the discovered producer set (by inspection, per above) is fully accounted for; a
   newly-exported producer neither registered nor documented fails CI. A red-first meta-test plants an
   `__all__`-less module with a producer and asserts discovery finds it (mirroring
   `test_discovery_sees_a_module_that_declares_no___all__`) — proving the defensive fallback is live.

Runs on **all legs** (behavioural-contract), bounded to default-config columns for speed.

### 1.4 `attribution` ↔ `NOTICE` gate (`tests/test_feature_glossary_notice_linkage.py`)

For every `FeatureColumn` with `attribution is not None`, assert the token appears **verbatim in `NOTICE`**
(substring presence — robust to prose format; both sides authored here). One-directional (`attribution ⊆
NOTICE`). Enforces ADR-005, makes the field trustworthy for wordalisation, and surfaces any methodology
feature whose citation is missing from NOTICE (bounded ADR-005 hygiene, in-scope for v1).

### 1.5 Authoring

~150 `FeatureColumn` entries, authored in per-module batches; the coverage gate drives completeness (author
until green). `unit` from the closed vocabulary; `definition` meets the UI-interpretability bar; `attribution`
reuses the citation token already in each feature's NOTICE entry/docstring; `higher_is_better` filled where
unambiguous, else `None`.

### 1.6 Deferred seam (named, not built)

`describe_level` consumes z-scores, but the glossary carries `unit`, not a "already-normalised vs raw" flag —
so a consumer cannot tell which columns are directly `describe_level`-able vs need normalising first. That
range/normalisation seam belongs to the wordalisation layer (out of v1 scope); named here so it is not a surprise.

---

## Sub-design #2 — TF-7 cross-family pitch-control cache in `*_xfns`

### 2.1 Mechanism — caller-injected, `compute_features` untouched

Add `pitch_control_cache: PitchControlCache | None = None` to each pitch-control-consuming `*_xfns` factory
that lacks it. Each factory's inner helper threads the cache into its aggregator call (the aggregators already
accept it — the public `add_obso` takes `pitch_control_cache` at `features.py:5286` and falls back to a fresh
`PitchControlCache()` internally; `_obso.py` holds only compute internals — `compute_pass_obso` at `_obso.py:289`
also carries the param but is not the public seam):

```python
def obso_xfns(..., xt=None, pitch_control_cache=None):
    def _helper(actions, frames):
        return add_obso(actions, frames, xt=xt, pitch_control_cache=pitch_control_cache)  # was: no cache
    ...
```

The caller builds **one** `PitchControlCache()` and passes it to every factory; the lakehouse slots this into
its existing "pre-build once, pass to all" pipeline. `compute_features` is unchanged.

**Concurrency + lifetime.** `PitchControlCache` is a **mutable memo**; a single shared instance is safe only
**single-threaded within one `compute_features` pass** (the lakehouse's per-match unit of work) — not shared
across threads/processes; a parallel executor gives each worker its own cache. The caller owns lifetime: the
cache is **match-scoped and discarded after the pass**, so peak memory is one match's canonical surfaces, not
a run's.

### 2.2 Families in scope + completeness gate

The pitch-control-consuming aggregators CLAUDE.md names — `pitch_control_xfns`, `obso_xfns`,
`cover_shadow_xfns`, `gk_influence_xfns`, `player_influence_xfns`, `space_creation_xfns` — plus `pausa_xfns`
and `xshot_occurrence_xfns` **iff they build/consume a `PitchControlSurface`** (confirmed during
implementation). `xcross_attempt_xfns` already has the param — bring to the identical shape. Same for the
atomic mirrors (`atomic_pitch_control_xfns` + any atomic PC-consumers).

**Completeness gate** (`tests/tracking/test_pitch_control_xfns_cache_wiring.py`): enumerate every `*_xfns`
factory whose aggregator accepts `pitch_control_cache` and assert the factory itself accepts and threads it —
so a future PC-consuming family cannot be added without the cache seam (mirrors #1's complete-by-construction
philosophy; the perf guard alone would miss an omitted family silently).

### 2.3 Value-identity + no retrain

`PitchControlCache` memoises **canonical** per-frame surfaces keyed on frame identity + params; sharing one
instance across families yields byte-identical surfaces, computed once. `None` default ⇒ own local cache ⇒
**byte-identical to today**; default xfn lists stay cache-`None`. → **no value change, no VAEP retrain, no Hyrum.**

---

## Cross-cutting — testing / delivery / C4 / retrain / ADR / NOTICE

**Testing.**
- #1: coverage gate (§1.3, incl. the `__all__`-less meta-test) + NOTICE-linkage gate (§1.4) + `describe_level`
  unit tests (every band boundary both sides, **`higher_is_better=False` flip**, NaN→"unknown",
  scalar/ndarray/Series dtype + Series index) + Public-API `Examples` for the new public functions.
- **#1 roundtrip e2e** (`tests/test_feature_glossary_roundtrip.py`): `dump_glossary` → reload JSON → every entry
  survives (incl. `schema_version`) → `describe_level` over **one `higher_is_better=True` and one `False`**
  real glossary column, feeding each entry's own direction, asserting the expected bands (so the
  glossary-direction → `describe_level` flip is exercised through real entries, not just the True path).
  Proves the foundation is consumable, not just internally consistent.
- #2: **value-identity test using ≥2 families with divergent PC params sharing one cache instance**, asserting
  exact (`np.array_equal`) equality — this exercises the mis-keying failure mode (same params would pass
  trivially); the once-per-unique-frame **structural call-count** perf guard; the **cache-wiring completeness
  gate** (§2.2); and confirmation the ADR-020 dup-`action_id` retrofit still holds with the cache threaded.

**Delivery / bisectability.** The two sub-features are independent; bundled in one PR they become one squash
commit on `main` (repo is squash-only), i.e. not independently revertable. Owner-directed as one PR/commit;
the accepted cost is explicit here. (Two independently-revertable commits on `main` would require two PRs.)

**C4.** No new action-coupled aggregator — count stays **31**. `feature_glossary.py` + `reporting.py` are new
documentation/utility components; regenerate `docs/c4/architecture.{dsl,html}` (200-char box-description cap).

**Retrain.** None. #1 is additive documentation; #2 is value-identical.

**ADR.** #1 introduces a repo-wide CI-enforced convention (every derived feature column documented +
NOTICE-linked, inspection-enumerated) with a downstream consumer (wordalisation) → **its own new ADR**. #2 is
an **ADR-008 amendment**. ADRs drafted during `/final-review`; numbers at commit-prep.

**NOTICE.** No new entries unless the §1.4 gate surfaces a documented methodology feature whose citation is
missing — those get added (bounded ADR-005 hygiene).

---

## Alternatives considered

| # | Option | Rejected because |
|---|--------|------------------|
| 1a | Glossary as a JSON/YAML data-file source of truth | House idiom is plain Python dicts; a Python registry is type-checkable, import-parse-free, JSON export is one line. |
| 1b | Markdown table as source | Parsing markdown as a data source is fragile. |
| 1c | Coverage scope = action-context / aggregators-only | Owner chose the full derived surface. |
| 1d | `describe_level` co-located in `feature_glossary.py` | Generic transform, different responsibility/cadence; mis-homes it and muddies the gate's registry scope. |
| 1e | `attribution`→NOTICE as a soft convention | "A count nothing routinely checks" is the silent-guard failure mode; a hard gate is near-zero cost. |
| 1f | **Co-located per-module metadata aggregated centrally** | Lower long-term drift (definition lives with the metric), but forces `feature_glossary.py` to import every producer module (heavy import coupling, contradicts the pure-registry decision). Deferred; drift mitigated instead by typed `unit`, the importable-`emitting_module` check, and review. Revisit if drift bites. |
| 1g | Enumerate the coverage surface from `__all__` | Reintroduces the id-scalar `__all__` blind spot (35 `__all__`-less modules already caught once); inspection with the `__all__`-less fallback is mandatory. |
| 2a | Framework-level cache injection (`compute_features` threads one cache) | Extends the frame-aware xfn contract + touches VAEP core; only wins for ad-hoc default-list users, not the perf-critical lakehouse path. Caller-injection mirrors the existing `add_*`/`links` pattern. |

## Open items resolved at implementation (not blockers)

- The exact gamestate-slot marker `*_xfns` append (the gate's base-name normalisation is written against it).
- Whether `pausa_xfns` / `xshot_occurrence_xfns` genuinely build a `PitchControlSurface` (add the param only if so).
- Final `Unit` vocabulary (extend the closed `Literal` only as authoring requires).
