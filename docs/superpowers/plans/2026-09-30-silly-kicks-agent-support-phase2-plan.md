# silly-kicks Agent-Support — Phase 2 (MCP) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans — inline, batching all tasks to a SINGLE approval-gated commit (see Global Constraints). Do NOT use per-task-commit cadence. Steps use checkbox (`- [ ]`) syntax. TDD throughout: failing test → minimal impl → green.

**Goal:** Ship a small MCP server exposing three runnable read-only tripwire/verdict tools (`check_orientation`, `diagnose_provider`, `validate_construct_validity`) as pure adapters over existing silly-kicks compute.

**Architecture:** `silly_kicks/mcp/server.py` (FastMCP, stdio) owns no logic. Each tool composes a **fail-loud read-only load seam** (`silly_kicks/mcp/_load.py`) with a **pure compute seam** (bound from existing lib fns, or extracted from scripts leaving their CLI unchanged). Load failure RAISES; it is never a verdict (ADR-052 D14).

**Tech Stack:** Python; `mcp[cli]>=1,<2` (FastMCP) as an optional extra; `pytest`; existing `silly_kicks` + `scripts/` + `calibration_runs/` code; `id_compat`; `pining_source` loader.

**Spec:** `D:\Development\_handoffs\silly-kicks-agent-support\2026-09-30-silly-kicks-agent-support-phase2-mcp-design.md` (APPROVED, review r2). Executors read both.

## Global Constraints

- **Edit `silly-kicks` only with per-repo approval, on a feature branch off the default branch (`karsten/agent-support-phase2`) — NEVER a worktree.**
- **One fully-tested commit for all of Phase 2, gated on Karsten's explicit approval immediately before commit. No micro-commits / no per-task commits (overrides writing-plans default). Never commit or push without that explicit approval.** Tasks end at *verify* (tests green).
- **Prerequisite:** Phase 1 (docs + shims) landed.
- **Read-only invariant:** every bound compute seam returns a dict with NO `require_clean_tree` gate, NO `run_commit`, NO artifact write. The server never calls a driver's `run()`/`main()`.
- **Fail-loud load:** the load seam RAISES on tokenless, failed, OR empty/degenerate resolution (tokenless pining can return empty refs + exit 0). A tool MUST NOT return `OK`/`UNORIENTED`/`GO` on a bad load.
- **`mcp[cli]` is an optional extra pinned `<2`**; the server import is guarded so the core lib + non-MCP consumers are unaffected.
- **All ids via `silly_kicks.id_compat`** (never raw `str()`/`==`).
- **Refactored scripts' CLI behavior is byte-identical** to pre-refactor (regression-gated).
- **No new analysis behavior** beyond exposing existing compute as importable pure fns.
- **D1/D2 SIGNED OFF by Karsten 2026-09-30:** **D1** = bounded/held-out `corpus_ref`; **D2** = ship ALL THREE `validate_*` families (`validate_gk_decision`, `validate_territorial_defense`, `validate_xtgk_possession_value`). **D3** (exact seam signatures/paths) confirmed against live HEAD as each task's first step.

## Scope sign-off gate — SATISFIED (Karsten, 2026-09-30)

Spec §10 listed D1 + D2 as open. Both are now signed off (recorded here + in the Review log), so pre-settling is no longer a concern:
- **D1 — corpus scope:** bounded/held-out `corpus_ref` only (fast tripwire; stays clear of the deferred heavy corpus wrappers).
- **D2 — families shipped:** ALL THREE (`validate_gk_decision`, `validate_territorial_defense`, `validate_xtgk_possession_value`). Task 5 has one seam-extraction + regression sub-task per family behind the same tool signature. No deferral of the other two.

## File Structure

Create:
- `silly_kicks/mcp/__init__.py`, `silly_kicks/mcp/server.py` (FastMCP app + `@tool` registrations + `python -m` entry).
- `silly_kicks/mcp/_load.py` (fail-loud read-only load seam: `load_match`, `load_corpus`).
- `silly_kicks/mcp/_validate_seams.py` (`metric_family` → extracted `validate()` dispatch table).
- `tests/mcp/conftest.py` (shared fixtures — see Fixtures note).
- `tests/mcp/test_load_seam.py`, `test_check_orientation.py`, `test_diagnose_provider.py`, `test_validate_construct_validity.py`, `test_registration.py`.
- `.mcp.json` (Claude registration); `docs/howto/mcp.md` (other-client registration).

Modify (compute-seam extraction — CLI unchanged):
- `scripts/validate_gk_decision.py`, `scripts/validate_territorial_defense.py`, `scripts/validate_xtgk_possession_value.py` — each: extract `validate(loaded, gate) -> verdict_dict`; `main()` calls it and still writes its memo/artifact (CLI unchanged).
- `calibration_runs/diag_gs_frames.py` + `calibration_runs/diag_idsse_coords.py` (ids/coords probes) — extract `probe(loaded, aspect) -> dict`; `__main__` calls it.
- `pyproject.toml` — add `[project.optional-dependencies] mcp = ["mcp[cli]>=1,<2"]`.

Bind (no change, import only): `scripts/measure_rc4_orientation.measure()` (`:57`); `silly_kicks` `GkClampDiagnosis` / `validate_gk_position_clamp` (keeper); `silly_kicks` `detect_input_convention` (convention).

**Fixtures note (P2-PLAN-03):** `tests/mcp/conftest.py` defines the fixtures the tests reference. Reuse the EXISTING RC4 fixtures — `unoriented_pining_ref` points at the unoriented case in `tests/tracking/_probe_fixtures.py` / `tests/tracking/test_measure_rc4_orientation.py` (do not invent a new one). `fixture_ref` = an existing per-provider match fixture. `dirty_tree` = a fixture that dirties the working tree (writes a tracked temp file) and exposes `no_new_artifacts()` (asserts no `metrics.json`/artifact appeared). `bounded_corpus_ref` = an existing small/held-out corpus fixture; `empty_ref`/`empty_corpus_ref` = a ref that resolves empty. Literal test bodies assert the **spec-defined contract**; exact loader signatures are read from live HEAD per D3 (each task Step 1).

---

### Task 1: MCP package scaffold + optional extra

**Files:** Create `silly_kicks/mcp/__init__.py`, `silly_kicks/mcp/server.py`; Modify `pyproject.toml`; Test `tests/mcp/test_registration.py`.

**Produces:** `silly_kicks.mcp.server:app` (FastMCP) exposing tools registered by later tasks; `python -m silly_kicks.mcp.server` runs stdio.

- [ ] **Step 1:** Confirm the `mcp[cli]` FastMCP import path at the pinned version; add `[project.optional-dependencies] mcp = ["mcp[cli]>=1,<2"]`; install the extra in the repo venv.
- [ ] **Step 2 (failing test):**
```python
# tests/mcp/test_registration.py
def test_server_exposes_expected_tools():
    from silly_kicks.mcp import server
    names = {t.name for t in server.app.list_tools()}  # adapt to FastMCP API at HEAD
    assert names == {"check_orientation", "diagnose_provider", "validate_construct_validity"}
```
- [ ] **Step 3:** Run → FAIL (module/app missing).
- [ ] **Step 4:** Create `server.py`: guarded `from mcp.server.fastmcp import FastMCP`; `app = FastMCP("silly-kicks")`; `if __name__ == "__main__": app.run()`. (Tools added in Tasks 3–5.)
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_registration.py -v` — FAIL until Tasks 3–5 register the tools (keep as the running integration assertion; do not weaken). No commit.

### Task 2: Fail-loud read-only load seam

**Files:** Create `silly_kicks/mcp/_load.py`; Test `tests/mcp/test_load_seam.py`.

**Produces:** `load_match(match_ref, provider=None) -> LoadedMatch` and `load_corpus(corpus_ref) -> LoadedCorpus` — read-only, RAISE on tokenless/failed/empty resolution.

- [ ] **Step 1:** Read `pining_source` load entry + `PINING_FOR_THE_DATA_TOKEN` handling at HEAD; identify the loaded-data type `measure()` consumes.
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_load_seam.py
import pytest
from silly_kicks.mcp import _load

def test_raises_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception):
        _load.load_match("any-ref")

def test_raises_on_empty_resolution(empty_ref):
    with pytest.raises(Exception):     # empty refs + exit 0 must still RAISE
        _load.load_match(empty_ref)

def test_ok_on_fixture(fixture_ref):
    assert _load.load_match(fixture_ref) is not None
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Implement `_load`: resolve ref via `pining_source`; **raise** if token absent, load throws, or resolution is empty/degenerate (do NOT catch-and-return). No artifact write, no clean-tree gate.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_load_seam.py -v` → PASS. No commit.

### Task 3: `check_orientation` tool

**Files:** Modify `silly_kicks/mcp/server.py`; Test `tests/mcp/test_check_orientation.py`.

**Consumes:** `_load.load_match`; `measure_rc4_orientation.measure()` (`:57`).
**Produces:** tool `check_orientation(match_ref, provider=None) -> dict`.

- [ ] **Step 1:** Confirm `measure()` signature + return keys at HEAD (`:57`; never loads, never catches).
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_check_orientation.py
import pytest
from silly_kicks.mcp import server, _load
from scripts import measure_rc4_orientation as m  # adapt import path at HEAD

def test_verdict_shape(fixture_ref):
    r = server.check_orientation(fixture_ref)
    assert set(r) >= {"match_ref","direction_resolved","geometry_agrees","verdict","detail"}
    assert r["verdict"] in {"OK","UNORIENTED","MISMATCH"}

def test_rc4_unoriented_fixture(unoriented_pining_ref):
    assert server.check_orientation(unoriented_pining_ref)["verdict"] == "UNORIENTED"

def test_thin_wrapper_equality(fixture_ref):        # guards server drift
    loaded = _load.load_match(fixture_ref)
    seam = m.measure(loaded)
    r = server.check_orientation(fixture_ref)
    for k in ("direction_resolved","geometry_agrees","verdict"):
        assert r[k] == seam[k]                      # tool passes seam through, no recompute

def test_fail_loud_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception):
        server.check_orientation("any-ref")         # MUST NOT return OK/UNORIENTED

def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.check_orientation(fixture_ref)
    assert dirty_tree.no_new_artifacts()
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Implement `@app.tool check_orientation`: `loaded = _load.load_match(...)` → `measure(loaded)` → shape dict. No catch around the load.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_check_orientation.py -v` → PASS. No commit.

### Task 4: `diagnose_provider` tool + ids/coords probe extraction

**Files:** Modify `silly_kicks/mcp/server.py`, `calibration_runs/diag_gs_frames.py`, `calibration_runs/diag_idsse_coords.py`; Test `tests/mcp/test_diagnose_provider.py`, `tests/scripts/test_diag_probe_regression.py`.

**Consumes:** `_load.load_match`; extracted `probe()` (ids/coords); lib `GkClampDiagnosis`/`validate_gk_position_clamp` (keeper); lib `detect_input_convention` (convention).
**Produces:** tool `diagnose_provider(provider, match_ref, aspect) -> dict`.

- [ ] **Step 1:** Read the `diag_*.py` bodies (ids/coords) + the lib `GkClampDiagnosis`/`detect_input_convention` signatures at HEAD.
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_diagnose_provider.py
import pytest
from silly_kicks.mcp import server, _load

@pytest.mark.parametrize("aspect", ["ids","coords","keeper","convention"])
def test_aspect_shape(fixture_ref, aspect):
    r = server.diagnose_provider("skillcorner", fixture_ref, aspect)
    assert set(r) >= {"provider","match_ref","aspect","findings","flags"}

def test_thin_wrapper_equality_ids(fixture_ref):     # guards server drift for an extracted seam
    loaded = _load.load_match(fixture_ref)
    from calibration_runs import diag_gs_frames as d  # adapt at HEAD
    assert server.diagnose_provider("skillcorner", fixture_ref, "ids")["findings"] == d.probe(loaded, "ids")

def test_fail_loud_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception):
        server.diagnose_provider("skillcorner","any-ref","ids")

def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.diagnose_provider("skillcorner", fixture_ref, "coords")
    assert dirty_tree.no_new_artifacts()
```
```python
# tests/scripts/test_diag_probe_regression.py
def test_diag_gs_frames_cli_output_unchanged(...):   # main() output/artifact == pre-refactor golden
    ...
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Extract a pure `probe(loaded, aspect)->dict` from the ids/coords `diag_*.py` (their `__main__` now calls it, output unchanged). Implement the tool: `load_match` → dispatch aspect → {ids/coords: `probe`; keeper: `GkClampDiagnosis`/`validate_gk_position_clamp`; convention: `detect_input_convention`} → findings dict.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_diagnose_provider.py tests/scripts/test_diag_probe_regression.py -v` → PASS. No commit.

### Task 5: `validate_construct_validity` tool + all three `validate_*` seams (D2 = all three)

**Files:** Modify `silly_kicks/mcp/server.py`, `silly_kicks/mcp/_validate_seams.py`, `scripts/validate_gk_decision.py`, `scripts/validate_territorial_defense.py`, `scripts/validate_xtgk_possession_value.py`; Test `tests/mcp/test_validate_construct_validity.py`, `tests/scripts/test_validate_gk_decision_regression.py`, `tests/scripts/test_validate_territorial_defense_regression.py`, `tests/scripts/test_validate_xtgk_possession_value_regression.py`.

**Consumes:** `_load.load_corpus`; the three extracted `validate(loaded, gate)->verdict_dict` seams.
**Produces:** tool `validate_construct_validity(metric_family, corpus_ref, gate) -> dict` dispatching on `metric_family`.

- [ ] **Step 1:** D1 (bounded) + D2 (all three families) are signed off (Scope sign-off gate). Read `validate_gk_decision.py`, `validate_territorial_defense.py`, `validate_xtgk_possession_value.py`; in each, separate the pure compute from the artifact/clean-tree wrapper; confirm the shared `metrics.json` verdict shape.
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_validate_construct_validity.py
import pytest
from silly_kicks.mcp import server, _load, _validate_seams as vs

FAMILIES = ["gk_decision", "territorial_defense", "xtgk_possession_value"]

@pytest.mark.parametrize("family", FAMILIES)
def test_verdict_shape(bounded_corpus_ref, family):
    r = server.validate_construct_validity(family, bounded_corpus_ref, "predictive")
    assert set(r) >= {"metric_family","gate","verdict","value","threshold","caveats"}
    assert r["verdict"] in {"GO","NO_GO"}

@pytest.mark.parametrize("family", FAMILIES)
def test_thin_wrapper_equality(bounded_corpus_ref, family):   # guards server drift
    loaded = _load.load_corpus(bounded_corpus_ref)
    r = server.validate_construct_validity(family, bounded_corpus_ref, "predictive")
    seam = vs.SEAMS[family](loaded, "predictive")             # dispatch table to the 3 extracted validate() fns
    for k in ("verdict","value","threshold"):
        assert r[k] == seam[k]

@pytest.mark.parametrize("family", FAMILIES)
def test_fail_loud_on_empty_corpus(empty_corpus_ref, family):
    with pytest.raises(Exception):
        server.validate_construct_validity(family, empty_corpus_ref, "predictive")

@pytest.mark.parametrize("family", FAMILIES)
def test_read_only_on_dirty_tree(dirty_tree, bounded_corpus_ref, family):
    # validate() extracted from clean-tree-gated scripts — the exact case this assertion guards
    server.validate_construct_validity(family, bounded_corpus_ref, "predictive")
    assert dirty_tree.no_new_artifacts()
```
```python
# tests/scripts/test_validate_<family>_regression.py   (one per family)
def test_cli_output_unchanged(...):                   # main() output/artifact == pre-refactor golden
    ...
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** In each of the three scripts extract `validate(loaded, gate)->verdict_dict` (`main()` calls it, still writes its memo). Register the three in `silly_kicks/mcp/_validate_seams.py` as `SEAMS[family]`. Implement the tool: `load_corpus(bounded)` → `SEAMS[metric_family]` → GO/NO-GO dict.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_validate_construct_validity.py tests/scripts/test_validate_gk_decision_regression.py tests/scripts/test_validate_territorial_defense_regression.py tests/scripts/test_validate_xtgk_possession_value_regression.py -v` → PASS. No commit.

**Acceptance:** all three families return the verdict shape and pass thin-wrapper + fail-loud + dirty-tree; each script's CLI output is unchanged.

### Task 6: Registration + docs

**Files:** Create `.mcp.json`, `docs/howto/mcp.md`; re-run `tests/mcp/test_registration.py`.

- [ ] **Step 1:** Write `.mcp.json`: `{"mcpServers":{"silly-kicks":{"command":"python","args":["-m","silly_kicks.mcp.server"],"cwd":"."}}}`.
- [ ] **Step 2:** Write `docs/howto/mcp.md`: the optional-extra install, the `.mcp.json` block, and manual registration for other MCP clients (Codex/Cursor) pointing at the same `python -m silly_kicks.mcp.server`.
- [ ] **Step 3 (verify):** `.mcp.json` valid JSON targeting `python -m silly_kicks.mcp.server`; `pytest tests/mcp/test_registration.py -v` → PASS (all 3 tools registered). No commit.

### Final: single Phase-2 commit (approval-gated)

- [ ] **Step 1:** On the feature branch, run the full Phase-2 suite: `pytest tests/mcp tests/scripts -v` → all green (incl. load-seam fail-loud, thin-wrapper equality, dirty-tree read-only across all three validate families, RC4, CLI-regression ×5).
- [ ] **Step 2:** Show Karsten the full file list + diff for all of Phase 2. **STOP.**
- [ ] **Step 3:** Only on Karsten's explicit "yes to this commit", create ONE commit (all Phase-2 files) on the feature branch. Do not push unless separately approved.

---

## Self-Review

**Spec coverage:** server §3 → Task 1; fail-loud load seam §4a → Task 2 (+ per-tool fail-loud tests in 3/4/5); the 3 tools §5 → Tasks 3/4/5 (`validate_construct_validity` covers all three families per D2); compute-seam extraction §4b + lib-reuse (keeper/convention) → Task 4; CLI-regression → Tasks 4/5 (×3 validate families); registration §6 → Task 6. **Spec §7 tests, each present per tool:** verdict-shape, **thin-wrapper equality (distinct from verdict-shape)**, **dirty-tree read-only (Tasks 3, 4, AND 5 — all three validate families)**, fail-loud, RC4 tripwire (Task 3), CLI-regression. D3 confirm-at-HEAD in each Step 1.

**Scope:** D1 (bounded) + D2 (all three families) SIGNED OFF by Karsten 2026-09-30 — recorded in Global Constraints + the Scope sign-off gate + the Review log. No unapproved deferral.

**Placeholder scan:** test bodies assert the spec contract (verdict keys/enums, RAISES, seam-equality); fixtures in `tests/mcp/conftest.py` reuse the existing RC4 fixtures; loader signatures resolved from HEAD per D3. No vague steps.

**Type/name consistency:** `_load.load_match`/`load_corpus`, `_validate_seams.SEAMS`, `server.check_orientation`/`diagnose_provider`/`validate_construct_validity`, and the verdict dict keys are referenced identically across Tasks 2–6 and match spec §5.

## Review log

- **2026-09-30** — independent review r1: REQUEST CHANGES. Applied: **P2-PLAN-01** (D1/D2 no longer settled — added the Scope sign-off gate + Global-Constraints wording; Task 5 Step 1 confirms sign-off); **P2-PLAN-02** (added thin-wrapper equality per tool + dirty-tree read-only to Tasks 4 AND 5, not just Task 3); **P2-PLAN-03** (added `tests/mcp/conftest.py`, `unoriented_pining_ref` reuses `tests/tracking/_probe_fixtures.py`, specified `dirty_tree`/`bounded_corpus_ref` sources). Report: `D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase2-plan.md`.
- **2026-09-30 — D1/D2 sign-off (Karsten):** D1 = bounded/held-out `corpus_ref`; D2 = ship ALL THREE `validate_*` families. Scope sign-off gate satisfied; Task 5 expanded to `gk_decision` + `territorial_defense` + `xtgk_possession_value` (one seam-extraction + regression sub-task each, `_validate_seams.SEAMS` dispatch).
- **2026-09-30, round 2** — independent re-review: **APPROVE** — all three r1 findings resolved (scope cut removed, all three families ship), no new findings, nothing regressed vs `3ca609f`. Report: `D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase2-plan-r2.md`.
