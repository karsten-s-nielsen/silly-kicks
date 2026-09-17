# SK-EXPORT — Uniform Metric-Family Output-Contract Registry — Implementation Plan

> **For agentic workers:** implement task-by-task, TDD red-first. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Give every column-emitting metric family a uniform, discoverable, enforced public output contract — a central `silly_kicks.metric_contracts.METRIC_CONTRACTS` registry plus the per-package constant re-exports it is built from — so a consumer's schema-drift parity guard is ONE generic loop. Purely additive; no compute/behavior change; no retrain, no re-materialize.

**Architecture:** New top-level `silly_kicks/metric_contracts.py` (mirrors `feature_glossary.py`). Registry is a plain `dict[str, MetricContract]` where `MetricContract` is a `TypedDict` (house "schemas are plain dicts" precedent + pyright-checked keys, zero runtime class). Per **Option B** it **imports no metric package** — the values are hardcoded literals kept honest by the round-trip test (Task 3), so each metric family stays a pure leaf and its `test_import_allowlist.py` contract is untouched. The four missing constants are still made public (Task 1) for the test + direct callers. `match_outcome`'s dict is NOT changed; the registry's `match_outcome` literal mirrors its names + types.

**Tech Stack:** Python 3.10, pytest, ruff, pyright.

**Spec:** `docs/superpowers/specs/2026-09-17-sk-export-metric-output-contract-design.md` (all §). Read it.

## Global Constraints

- Line length 120; ML-naming ruff per-file-ignores do not apply here.
- **Branch:** single `feat/sk-export-metric-output-contract` off `main`. One coherent, fully-tested commit.
- **Approval gates:** no `git commit` / `git push` / tag / `pypi` publish without explicit owner "yes" for that specific action. Approval of this plan is NOT commit authority. **Do not tag before CI is green.**
- **Version number is assigned at commit-prep only** — a concurrent silly-kicks session may release first, so `_version.py` and the CHANGELOG/ADR numbers are filled in as the LAST step, re-deriving next-free then (spec header; `[no-version-number-until-commit-prep]`).
- **Additive invariant:** every numeric output byte-identical → NO retrain, NO re-materialize, NO golden regen. A test or introspection that shows any `compute_*` output changed is a STOP-and-report.
- **Watch every new assertion fail first** (`[guards-that-cannot-fail]`) — especially the output-faithfulness leg (Task 3 Step 4).

## File Structure

- New: `silly_kicks/metric_contracts.py`, `tests/test_metric_contracts.py`.
- Modify `__init__.py` (re-export + `__all__`): `silly_kicks/match_outcome/__init__.py`, `silly_kicks/gk_decision/__init__.py`, `silly_kicks/territory/__init__.py`, `silly_kicks/duels/__init__.py`.
- Modify `silly_kicks/gk_decision/_columns.py` (add `GK_DECISION_KEYS`). **No other `_columns.py` touched** (match_outcome dict stays — SKEXP-SPEC-02).
- Docs (Task 4): `CHANGELOG.md`, `docs/superpowers/adrs/ADR-NNN-*.md`, `CLAUDE.md` (one Key-conventions line), `silly_kicks/_version.py`.

---

### Task 1: Leaf constants — `GK_DECISION_KEYS` + the four re-exports

**Files:** `silly_kicks/gk_decision/_columns.py`, and `__init__.py` for `match_outcome`/`gk_decision`/`territory`/`duels`.

**Interfaces — after this task, all import:** `MATCH_OUTCOME_METRIC_COLUMNS`, `MATCH_OUTCOME_KEYS`, `MATCH_OUTCOME_COLUMNS` (match_outcome); `GK_DECISION_METRIC_COLUMNS`, `GK_DECISION_KEYS`, `GK_DECISION_SAMPLE_COLUMNS` (gk_decision); `TERRITORY_KEYS` (territory); `DUEL_KEYS` (duels).

- [ ] **Step 1 (red): add the import-smoke test** in `tests/test_metric_contracts.py` (created here, expanded in Task 3):
```python
def test_public_metric_constants_importable():
    from silly_kicks.match_outcome import MATCH_OUTCOME_METRIC_COLUMNS, MATCH_OUTCOME_KEYS
    from silly_kicks.gk_decision import GK_DECISION_METRIC_COLUMNS, GK_DECISION_KEYS
    from silly_kicks.territory import TERRITORY_KEYS
    from silly_kicks.duels import DUEL_KEYS
```
Run: `python -m pytest tests/test_metric_contracts.py -q` → FAIL (ImportError).

- [ ] **Step 2: add `GK_DECISION_KEYS`** to `gk_decision/_columns.py`:
```python
#: Summarize / mart grain (summarize_gk_decision groups ["keeper", "game_id"] @ _compute.py:122).
#: compute -> per-decision samples; summarize -> per-(game_id, keeper).
GK_DECISION_KEYS: tuple[str, ...] = ("game_id", "keeper")
```
(Value verified against the package — NOT `player_id`.)

- [ ] **Step 3: re-export** in each `__init__.py`, mirroring `team_metrics/__init__.py:19` (`from ._columns import …` + add to `__all__`, keeping `__all__` sorted/grouped as that file does): match_outcome (`MATCH_OUTCOME_METRIC_COLUMNS, MATCH_OUTCOME_KEYS, MATCH_OUTCOME_COLUMNS`), gk_decision (`GK_DECISION_METRIC_COLUMNS, GK_DECISION_KEYS, GK_DECISION_SAMPLE_COLUMNS`), territory (`TERRITORY_KEYS`), duels (`DUEL_KEYS`).

- [ ] **Step 4 (green):** Run: `python -m pytest tests/test_metric_contracts.py::test_public_metric_constants_importable -q` → PASS. Also the spec §5 one-liner smoke.

---

### Task 2: The registry module `silly_kicks/metric_contracts.py`

**Files:** `silly_kicks/metric_contracts.py`.

**Interfaces:**
- Produces: `MetricContract` (`TypedDict`: `keys`, `metric_columns`, `columns` — all `tuple[str, ...]`; `column_types: Mapping[str, str] | None`) and `METRIC_CONTRACTS: dict[str, MetricContract]` with 7 entries (`team_metrics`, `match_outcome`, `shot_stopping`, `gk_decision`, `territory`, `duels`, `restdefense`).

- [ ] **Step 1: build the registry as hardcoded literals** (Option B — `metric_contracts` imports NO metric package; generate the literals once from the packages and paste them, kept honest by the Task 3 round-trip test). Each literal EQUALS the package constant named below; `columns` EQUALS the `<PKG>_COLUMNS` NAMES (for the 5 dict families, the dict's keys). **`column_types` is populated wherever the package's full-columns constant is a typed `dict[str,str]` — VERIFIED 5 families — and `None` only for the 2 that are not (SKEXP-PLAN-06):**
  - **Typed-dict families → `column_types=dict(<PKG>_COLUMNS)`:** `team_metrics` (`TEAM_KPI_COLUMNS` @ `_columns.py:87`), `shot_stopping` (`SHOT_STOPPING_COLUMNS` :34), `territory` (`TERRITORY_COLUMNS` :54), `duels` (`DUEL_COLUMNS` :34), `match_outcome` (`MATCH_OUTCOME_COLUMNS` :22). For each: `metric_columns=tuple(<PKG>_METRIC_COLUMNS)`, `keys=tuple(<PKG>_KEYS)`, `columns=tuple(<PKG>_COLUMNS)`, `column_types=dict(<PKG>_COLUMNS)`. (match_outcome's `MATCH_OUTCOME_METRIC_COLUMNS` is itself a dict → `metric_columns=tuple(MATCH_OUTCOME_METRIC_COLUMNS)`; the package is NOT changed — SKEXP-SPEC-02.)
  - **`gk_decision` → `column_types=None`** (`GK_DECISION_SAMPLE_COLUMNS` @ `_columns.py:38` is a `tuple`, untyped): `keys=tuple(GK_DECISION_KEYS)`, `metric_columns=tuple(GK_DECISION_METRIC_COLUMNS)`, `columns=tuple(GK_DECISION_SAMPLE_COLUMNS)`.
  - **`restdefense` → `column_types=None`** (`RD_METRIC_COLUMNS` @ `_columns.py:53` is a name list): `metric_columns=tuple(RD_METRIC_COLUMNS)`, `keys=tuple(RD_SAMPLE_KEYS)`, `columns=tuple(RD_SAMPLE_KEYS)+tuple(RD_METRIC_COLUMNS)`.

- [ ] **Step 2: `__all__ = ["MetricContract", "METRIC_CONTRACTS"]`**; module docstring names the consumer contract + points to the enforcement test. Add a doctest-free literal Examples block if the public-API-example gate requires one (mirror another top-level module).

- [ ] **Step 3:** Run: `python -c "from silly_kicks.metric_contracts import METRIC_CONTRACTS; print(len(METRIC_CONTRACTS))"` → `7`.

---

### Task 3: Enforcement test (red-first, complete-by-enumeration)

**Files:** `tests/test_metric_contracts.py` (expand).

- [ ] **Step 1 — self-consistency (every entry):** `metric_columns` non-empty; `keys`/`metric_columns`/`columns` are `tuple` of `str`; `set(metric_columns) ⊆ set(columns)`; `set(keys) ⊆ set(columns)`; `column_types` is `None` or a `Mapping[str,str]` whose keys ⊆ `columns`. **`column_types` coverage (SKEXP-PLAN-06):** `column_types is not None` for exactly the 5 families whose `*_COLUMNS` is a typed `dict` (team_metrics/shot_stopping/territory/duels/match_outcome), `None` for the 2 whose full-columns is a tuple/list (gk_decision/restdefense); where non-None, `dict(column_types) == dict(<PKG>_COLUMNS)` and `set(column_types) == set(columns)`.

- [ ] **Step 2 — round-trip:** each entry equals the package's public constants (`metric_columns == tuple(pkg.<X>_METRIC_COLUMNS)` modulo the match_outcome coercion; `keys == tuple(pkg.<X>_KEYS)`).

- [ ] **Step 3 — completeness (ADR-056 three-bucket):**
  - `derived = {pkg for pkg under silly_kicks/ whose public __all__ exports a "*_METRIC_COLUMNS" name}` (structural scan via `pkgutil`/`importlib`).
  - `assert set(METRIC_CONTRACTS) == derived` — every `*_METRIC_COLUMNS`-exporting package is registered.
  - `_EXEMPT = {"xsuccess": "VAEP rating method; emits no mart column-set (TF-61/ADR-095)"}`; assert each exempt pkg exists AND exports NO `*_METRIC_COLUMNS` (so it is correctly outside `derived`, and gaining one later would break the equality → force a decision). `_UNDERIVABLE = frozenset()` asserted empty.

- [ ] **Step 4 — output-faithfulness (closes the vacuity gap; watch it FAIL first):** for each family assert `set(keys) ⊆ declared-output` and `set(metric_columns) ⊆ declared-output`, where `declared-output` is the package's authoritative full-output constant (`*_COLUMNS` / `GK_DECISION_SAMPLE_COLUMNS`), NOT the registry's own `columns`. **Non-vacuity proof:** temporarily set the gk_decision entry's `keys` to `("game_id","player_id")` and confirm THIS test FAILS (`player_id ∉ GK_DECISION_SAMPLE_COLUMNS`); revert. Additionally, run `summarize_gk_decision` on a 2-row hand-built samples fixture (`keeper`,`game_id` + the 5 metric cols) and assert its output columns ⊇ `GK_DECISION_KEYS` (proves `keeper` is the real groupby grain, not just a sample column).

- [ ] **Step 5 — `__all__` presence:** every constant the registry references is in its package's `__all__`.

- [ ] **Step 6 — non-vacuity (watch-red), then green (SKEXP-PLAN-08).** The registry is fully built in Task 2, so completeness / round-trip / `column_types`-coverage would pass immediately — land each RED first via a temporary deliberate break, observe the failure, revert (repo landed-RED idiom): (a) drop one `METRIC_CONTRACTS` entry → completeness fails; (b) corrupt one `metric_columns` → round-trip fails; (c) `None` a typed `column_types` → coverage fails; (d) the Step-4 `player_id` flip → output-faithfulness fails. Revert all four. Then Run: `python -m pytest tests/test_metric_contracts.py -q` → PASS.

---

### Task 4: Docs + version + verification + commit gate

**Files:** `CHANGELOG.md`, `docs/superpowers/adrs/ADR-NNN-*.md`, `CLAUDE.md`, `silly_kicks/_version.py`.

- [ ] **Step 1: ADR** (next free number, re-derived now) — the uniform metric-output-contract convention + the registry as the canonical consumer surface + the `TypedDict`/plain-dict decision + the match_outcome coerce-not-break decision. `CHANGELOG.md` **Added** entry (SK-EXPORT; explicitly "no behavior change, no retrain"). One-line `CLAUDE.md` Key-conventions entry (the export invariant + `metric_contracts` as the enforced registry).

- [ ] **Step 2: assert additivity** — introspect that no `compute_*`/`summarize_*` output changed (a `git diff` shows only `__init__.py` re-exports, the one `GK_DECISION_KEYS` addition, the new module, the new test, and docs). No golden touched.

- [ ] **Step 3: full CI-faithful suite** (mirror `ci.yml`):
```bash
python -m pytest tests/ -m "not e2e"
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
python -m pytest --doctest-modules silly_kicks/ --ignore-glob="*/_[!_]*.py"   # metric_contracts is public
```
Expected: all green; capture pytest exit code directly (not via `| tail`).

- [ ] **Step 4: version bump — LAST, at commit-prep.** Re-derive next-free minor NOW (a concurrent sk session may have shipped); set `silly_kicks/_version.py` `__version__`; fill the ADR/CHANGELOG numbers to match. Re-run `python -m pytest tests/test_version_single_source.py -q`.

- [ ] **Step 5: STOP — request explicit owner approval to commit.** Show the diff / file list. On approval: create `feat/sk-export-metric-output-contract`, one commit (message: `feat(metric-contracts): uniform public output-contract registry + re-exports — silly-kicks <ver> (SK-EXPORT, ADR-NNN)`, `Co-Authored-By` trailer). Do NOT push/tag/publish without their separate "yes"; never tag before CI is green; owner publishes.

- [ ] **Step 6:** On release, reply to the lakehouse with the released version + tag to pin (`silly-kicks[das,ghost-gk,parse-dfl]==<version>`).

---

## Self-Review

1. **Spec coverage:** §3.1 registry → Task 2; §3.2 re-exports/`GK_DECISION_KEYS`/match_outcome-coerce → Tasks 1+2; §3.3 enforcement (self-consistency/round-trip/completeness/output-faithfulness/`__all__`) → Task 3; §4 files + §ADR/version → Task 4; OI-1 RESOLVED = `("game_id","keeper")` (Task 1 Step 2). ✓
2. **Review findings applied:** SKEXP-SPEC-01 → `GK_DECISION_KEYS=("game_id","keeper")` + Task 3 Step 4 non-vacuity proof (the exact `player_id` trap fails); SKEXP-SPEC-02 → match_outcome dict untouched, coerced at the registry (Task 2 Step 1), release purely additive. ✓
3. **TDD / additivity:** every task red-first; the output-faithfulness assertion is watched failing on the wrong tuple; additivity asserted (Task 4 Step 2); no golden/retrain. ✓
4. **Discipline:** single feature branch; one fully-tested commit; commit/push/tag/publish all owner-gated; version number assigned only at commit-prep. ✓
