# SK-EXPORT — Uniform Metric-Family Output-Contract Registry — Design Spec

- **Date:** 2026-09-17
- **Status:** Draft — pending independent (lakehouse) review
- **Author:** silly-kicks maintainer session (Claude Opus 4.8)
- **Target release:** next free minor — assigned at commit-prep (a concurrent silly-kicks session may release first; do NOT hardcode a number). Purely additive — no breaking change (see §7).
- **Decision:** next free ADR (a public convention with a downstream consumer → ADR-worthy)
- **Scope decision (owner-set):** gold standard / best practice; breaking changes acceptable because the only known consumer (luxury-lakehouse) is absorbing this in the same cycle and will re-pin + adjust.

## 1. Motivation

The luxury-lakehouse 4.118.0 adoption cycle materializes 8 new metric families, each as a gold mart whose column contract is guarded by a **schema-drift parity test**: import the sk package's authoritative column/grain constants and assert the mart equals them. Today that guard is only writable for 3 of the column-emitting families — `team_metrics`, `shot_stopping`, `restdefense` export both a metric-columns constant and a grain-keys constant; the other four keep them private in `_columns.py`, absent from `__all__`. Pinning a private module path is forbidden (`docs/PRIVATE_CONSUMERS.md`: path pins fail *silently* on rename).

A naive re-export (add the four missing constants to `__all__`) unblocks the consumer but leaves three inconsistencies that keep a *generic* parity guard impossible:

1. **`gk_decision` has no `*_KEYS`** at all — its output is per-decision, not per-`(game_id, keeper)`.
2. **`MATCH_OUTCOME_METRIC_COLUMNS` is a `dict`**; every other family's is a list/tuple of names.
3. **Prefixes are not derivable** from the package name (`TEAM_KPI_`, `SS_`, `RD_`, `MATCH_OUTCOME_`, …), so a consumer cannot loop packages and construct the constant name.

The gold-standard fix is a single, uniform, **discoverable, type-consistent, rot-proof** output contract exposed on every column-emitting metric family, so the consumer's parity guard is ONE generic loop with zero per-package name/type knowledge.

## 2. Current state (verified against main @ 2ca0ef4)

| package | metric-cols constant | type | grain-keys constant | in `__all__`? |
|---|---|---|---|---|
| team_metrics | `TEAM_KPI_METRIC_COLUMNS` (44) | list | `TEAM_KPI_KEYS` | yes |
| shot_stopping | `SHOT_STOPPING_METRIC_COLUMNS` (8) | list | `SS_KEYS` | yes |
| restdefense | `RD_METRIC_COLUMNS` (16) / `RD_LAYER1/2_COLUMNS` | list | `RD_SAMPLE_KEYS` | yes |
| match_outcome | `MATCH_OUTCOME_METRIC_COLUMNS` (5) | **dict** | `MATCH_OUTCOME_KEYS` | **no** |
| gk_decision | `GK_DECISION_METRIC_COLUMNS` (5) | tuple | **none** | **no** |
| territory | `TERRITORY_METRIC_COLUMNS` (12) | list | `TERRITORY_KEYS` | metric yes / keys **no** |
| duels | `DUEL_METRIC_COLUMNS` (6) | list | `DUEL_KEYS` | metric yes / keys **no** |
| xsuccess | — (none) | — | — | n/a (see §8) |

## 3. Design

### 3.1 The registry — the gold-standard consumer surface

New public module `silly_kicks/metric_contracts.py` (top-level, mirrors `feature_glossary.py` — it imports **no** metric package; the values are hardcoded literals kept honest by the §3.3 round-trip test, so each metric family stays a pure dependency-free leaf and its `test_import_allowlist.py` contract is untouched):

```python
from typing import TypedDict, Mapping

class MetricContract(TypedDict):
    keys: tuple[str, ...]            # mart grain
    metric_columns: tuple[str, ...]  # value column NAMES
    columns: tuple[str, ...]         # full ordered output NAMES (keys + metric + provenance)
    column_types: Mapping[str, str] | None  # name -> dtype where the package's *_COLUMNS is a typed dict (5: team_metrics/shot_stopping/territory/duels/match_outcome), else None (gk_decision/restdefense)

METRIC_CONTRACTS: dict[str, MetricContract] = { ... }  # one entry per column-emitting family
```

Consumer parity guard becomes:
```python
from silly_kicks.metric_contracts import METRIC_CONTRACTS
for family, c in METRIC_CONTRACTS.items():
    assert set(mart_columns(family)) == set(c["metric_columns"]) | set(c["keys"])
```
No per-package names, no type branching, `gk_decision` included.

**Why `TypedDict`, not a dataclass:** the house precedent is "schemas are plain Python dicts" (`SPADL_COLUMNS`, `ATOMIC_SPADL_COLUMNS`; pandera/multimethod were deliberately removed). A `TypedDict` value IS a plain dict at runtime — zero new runtime class, JSON-serializable, identical in kind to `SPADL_COLUMNS` — while giving pyright-checked field keys, the only real benefit a frozen dataclass would have offered. Strictly dominates both pure options.

### 3.2 Per-package normalization (the registry is built from these; direct callers keep working)

- **Re-export the missing constants** to each package's `__all__`, mirroring `team_metrics/__init__.py:19`: `match_outcome` (`MATCH_OUTCOME_METRIC_COLUMNS`, `MATCH_OUTCOME_KEYS`, `MATCH_OUTCOME_COLUMNS`), `gk_decision` (`GK_DECISION_METRIC_COLUMNS`, `GK_DECISION_KEYS`, `GK_DECISION_SAMPLE_COLUMNS`), `territory` (`TERRITORY_KEYS`), `duels` (`DUEL_KEYS`).
- **`MATCH_OUTCOME_METRIC_COLUMNS` stays a `dict`, no package break** (review SKEXP-SPEC-02). The registry entry's `metric_columns` is a hardcoded literal EQUAL to that dict's names (insertion-ordered) and its `column_types` mirrors `MATCH_OUTCOME_COLUMNS`; the package constant's type is untouched. Same Chesterton's-Fence logic as the domain prefixes — the registry supplies uniformity, so no per-package constant is churned, and the dict is *more* cohesive than a name-tuple + a separate types mapping. No `team_metrics` change either (`TEAM_KPI_METRIC_COLUMNS` is already `list[str]`).
- **Add `GK_DECISION_KEYS = ("game_id", "keeper")`** to `gk_decision/_columns.py` — the **summarize/mart grain**. VERIFIED against the package (review SKEXP-SPEC-01): `summarize_gk_decision` groups by `["keeper", "game_id"]` (`_compute.py:122`) and the samples carry `keeper`/`keeper_raw`, **not** `player_id` (`_columns.py:42-43`). (The earlier draft's `("game_id", "player_id")` was WRONG — corrected here.) Documented in the constant's comment: *compute → per-decision samples; summarize → per-`(game_id, keeper)`; `GK_DECISION_KEYS` reflects the summarize/mart grain.*
- **Keep the domain prefixes** (`TEAM_KPI_`, `SS_`, `RD_`) unchanged. The registry supplies uniformity, so renaming public constants is churn without payoff (Chesterton's Fence) — and the registry is exactly what removes the consumer's need for derivable prefixes.

### 3.3 Enforcement — complete-by-enumeration, so the asymmetry cannot reappear

New `tests/test_metric_contracts.py`, adopting the repo's registry-completeness idiom (ADR-056 three-bucket shape; same family as the `add_*`/`PURITY_ENTRIES`/id-scalar registries):

- **Self-consistency, every entry:** `metric_columns` non-empty; `keys`, `metric_columns`, `columns` are all `tuple[str, ...]`; `set(metric_columns) ⊆ set(columns)`; `set(keys) ⊆ set(columns)`; `column_types` is `None` or `Mapping[str,str]` whose keys ⊆ `columns`.
- **Round-trip to the package:** each entry's `metric_columns`/`keys` equal the package's public `*_METRIC_COLUMNS`/`*_KEYS` (so the registry cannot drift from the source constants).
- **Output-faithfulness (closes the vacuity gap — review SKEXP-SPEC-01 / CONSIDER):** the round-trip above only proves registry == constant; it passes vacuously if the *constant itself* is wrong (exactly the `player_id`-vs-`keeper` trap). So additionally assert each family's `keys` ⊆ the actual grain of its compute/summarize output and `metric_columns` ⊆ its actual sample output — run `compute_*` (and `summarize_*` where the mart grain is the summarized one, i.e. gk_decision) on the committed minimal fixture and subset-check against the REAL output columns, not just the declared constant. Where a package declares an authoritative full-output column constant (`*_COLUMNS`/`*_SAMPLE_COLUMNS`), that leg may use it instead of running compute; gk_decision's grain lives only in `summarize_gk_decision`, so that leg runs summarize.
- **Completeness (the anti-rot property):** structurally enumerate silly-kicks metric-family packages (a package under `silly_kicks/` exporting a public `*_METRIC_COLUMNS`), assert the set EQUALS `METRIC_CONTRACTS.keys() ∪ _EXEMPT`. `_EXEMPT = {"xsuccess": "<reason>"}` (§8). `_UNDERIVABLE` asserted **empty**. A new metric family fails CI until it registers or is exempted-with-reason.
- **`__all__` presence:** every constant the registry references is in its package's `__all__` (guards the re-export itself).

Watch each new assertion fail first (`[guards-that-cannot-fail]`).

## 4. Files

- New: `silly_kicks/metric_contracts.py`, `tests/test_metric_contracts.py`.
- Modify `__init__.py` (re-export + `__all__`): `match_outcome`, `gk_decision`, `territory`, `duels`.
- Modify `_columns.py`: `gk_decision` only (add `GK_DECISION_KEYS = ("game_id", "keeper")`). `match_outcome/_columns.py` is NOT touched (the dict stays; registry coerces — SKEXP-SPEC-02). No other `_columns.py` logic touched.
- `silly_kicks/_version.py` → next free minor (assigned at commit-prep — see header). `CHANGELOG.md` (SK-EXPORT `Added` entry). `docs/superpowers/adrs/ADR-NNN-*.md` (next free ADR). A one-line **Key conventions** entry in `CLAUDE.md` (the export invariant + the registry as the canonical consumer surface).

## 5. Testing

- `tests/test_metric_contracts.py` (§3.3).
- `python -c "from silly_kicks.metric_contracts import METRIC_CONTRACTS; ...; print('ok')"` smoke.
- The request's one-liner: `python -c "from silly_kicks.match_outcome import MATCH_OUTCOME_METRIC_COLUMNS, MATCH_OUTCOME_KEYS; from silly_kicks.gk_decision import GK_DECISION_METRIC_COLUMNS, GK_DECISION_KEYS; from silly_kicks.territory import TERRITORY_KEYS; from silly_kicks.duels import DUEL_KEYS; print('ok')"`.
- CI-faithful full suite: `python -m pytest tests/ -m "not e2e"`, `ruff check silly_kicks/ tests/ scripts/`, `ruff format --check …`, `pyright`. Confirm no `__all__`/doctest/public-API-example regression (the `MATCH_OUTCOME_METRIC_COLUMNS` type change may touch a doctest or the public-API-example gate).

## 6. Non-goals / out of scope

- **No compute/behavior/logic change.** Re-export + one new key tuple (`GK_DECISION_KEYS`) + a registry. `V`, the metrics, and every numeric output are byte-identical → **no retrain, no re-materialize.**
- **No renaming** of the existing domain-prefixed constants (`TEAM_KPI_*`, `SS_*`, `RD_*`).
- **No new metric.** Registry describes existing output only.

## 7. Consumer impact — purely additive (no breaking change)

Per SKEXP-SPEC-02 the registry coerces `match_outcome`'s dict, so no package constant changes type. The release is **purely additive**:
- **New public module** `silly_kicks.metric_contracts`.
- **New public constants** re-exported on 4 packages + one new `GK_DECISION_KEYS`.
- No type change, no rename, no behavior change → **no `Changed (BREAKING)` CHANGELOG entry**; a plain `Added` entry.

## 8. Carve-out: xsuccess

`xsuccess` emits no mart column-set constant. TF-61 (ADR-095) is a VAEP *rating method* (`VAEP.rate_adjusted` + `vaep.adjusted.adjusted_value`) adding `xsuccess`/`vaep_adjusted_value` columns onto `fct_action_values` — no `feature_glossary` growth, no `*_xfns`, no package column constant. It is therefore **not** in `METRIC_CONTRACTS`; it sits in `_EXEMPT` with that reason, so its absence is asserted-intentional, never accidental. The registry population is the **7** column-emitting families.

## 9. Open items (for the lakehouse review)

- **OI-1 — RESOLVED** to `GK_DECISION_KEYS = ("game_id", "keeper")` (verified: `summarize_gk_decision` groups `["keeper","game_id"]`, `_compute.py:122`; review SKEXP-SPEC-01). The lakehouse `fct_gk_decision` grain must key on `keeper` (not `player_id`) — confirm on their side.
- **OI-2 — `columns` ordering guarantee.** Do consumers rely on `columns` order for DDL, or only the set? If order-load-bearing, the registry must promise a stable order and a test must pin it (Hyrum). Default assumption: set-based parity; order is best-effort.
- **OI-3 — `column_types` coverage.** Only `match_outcome` (and possibly `team_metrics`) carry a typed mapping today; all others get `column_types=None`. Confirm the lakehouse DDL derivation tolerates `None` (falls back to its own type inference) for the untyped families.

## 10. Commit / approval gate

Implementation on a single feature branch `feat/sk-export-metric-output-contract` off `main`. One coherent, fully-tested commit. **No `git commit` / `git push` / tag / publish without explicit owner approval for that specific action** — approval of this spec is not commit authority, and 4.119.0 is not tagged before CI is green. On release, reply to the lakehouse with the ACTUAL released version + tag to pin (`silly-kicks[das,ghost-gk,parse-dfl]==<version>`) — the number is assigned at commit-prep, not before (a concurrent sk session may release first).
