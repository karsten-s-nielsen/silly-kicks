# silly-kicks Agent-Support — Phase 2 (MCP) Implementation Plan (Rev 7)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans — inline, batching all tasks to a SINGLE approval-gated commit (Global Constraints). No per-task-commit cadence. Steps use checkbox (`- [ ]`). TDD: failing test → minimal impl → green.

**Goal:** Ship a small MCP server exposing three read-only tools (`check_orientation`, `diagnose_provider`, `validate_construct_validity`) as **pure adapters that BIND existing lib functions or READ committed memos** — no library/`scripts/` change.

**Architecture:** `silly_kicks/mcp/server.py` (FastMCP, stdio) owns no logic beyond arg-normalization + one geometry-vs-label diff. Each tool composes a **fail-loud read-only seam** (match-load `_load.py`, or memo-read `_memo.py`) with **bound** compute seams. `check_orientation` binds `measure()` (raw metrics) + `orient_frames_to_ltr_by_geometry` (geometry on a label-stripped copy). Load/read failure RAISES; never a verdict (ADR-052 D14).

**Tech Stack:** Python; `mcp[cli]>=1,<2` (FastMCP) optional extra via `uv sync --extra mcp` (uv.lock committed); `pytest`; `silly_kicks` lib + committed `docs/research/*` memos; `id_compat`; `pining_source` loader.

**Spec:** `docs/superpowers/specs/2026-09-30-silly-kicks-agent-support-phase2-mcp-design.md` (Rev 5 — vs live HEAD `4049db9`). Executors read both.

## Why Rev 7 (read first)

Rev 6 read the per-period home-GK-median-x anchor directly but REIMPLEMENTED only the home-GK branch (r6 PLAN-11), dropping the away-GK fallback (`direction.py:284-293`), the period-5 exclusion (`:278`), and the no-anchor skip (`:287-292`): a period with no tracked home GK gave `median==NaN`, `NaN<52.5==False` → forced "rtl" → false MISMATCH vs an "ltr" label. Broadcast tracking loses the GK routinely, so it false-fires on real data (a GK-present `fixture_ref` wouldn't catch it). Rev 7 BINDS the real function instead of reimplementing its anchor: run `orient_frames_to_ltr_by_geometry(frames, home_team_id=…, on_missing_home="warn", copy=True)` read-only, read the per-period decision from INPUT-vs-OUTPUT coord reflection, and EXCLUDE periods the function could not anchor (its no-anchor warning) + period 5. The away-GK fallback / period-5 / skip all come from the function. Adds an untracked-home-GK test. Rev 5's `_json_safe`-all-aspects (PLAN-10) + built fixtures + xtgk fail-loud (PLAN-08) + Karsten's keep-as-defensive-check (PLAN-08) stand. `compute_attacking_direction` dropped (circular).

## Global Constraints

- **Feature branch `karsten/agent-support-phase2` (off `main`) — NEVER a worktree.**
- **One fully-tested commit, gated on Karsten's explicit approval immediately before commit. No micro/per-task commits. Never commit or push without approval.** Tasks end at *verify*.
- **Prerequisite:** Phase 1 landed (`main` `4049db9`).
- **Read-only invariant:** every bound seam is read-only — no `require_clean_tree`, no `run_commit` write, no artifact write. Never call a driver `run()`/`main()`.
- **Fail-loud seams:** match-load RAISES on tokenless/failed/empty; memo-read RAISES on absent/unreadable/empty/`run_tree_dirty: true`/missing-provenance (never `KeyError`).
- **`mcp[cli]` optional extra pinned `<2`**, guarded import; install via `uv sync --extra mcp`, commit `uv.lock`.
- **All ids via `silly_kicks.id_compat`.**
- **No library or `scripts/` behavior change.**
- **D3 confirm-at-HEAD** is each task's Step 1.

## File Structure

Create: `silly_kicks/mcp/{__init__.py, server.py, _load.py, _memo.py}`; `tests/mcp/{conftest.py, test_registration.py, test_load_seam.py, test_memo_seam.py, test_check_orientation.py, test_diagnose_provider.py, test_validate_construct_validity.py}`; `.mcp.json`; `docs/howto/mcp.md`.
Modify: `pyproject.toml` (+ `mcp` extra); `uv.lock` (via `uv sync`).
Bind (no change): `scripts/measure_rc4_orientation.measure()` (`:57`); `silly_kicks.tracking.orient_frames_to_ltr_by_geometry` (`direction.py:175`); `validate_gk_position_clamp`/`GkClampDiagnosis`; `detect_input_convention`/`DetectionResult`; `validate_id_dtypes`/`IdDtypeDiagnosis`. Read (no change): the memos.

**Fixtures note (BUILD, do not reuse — r4 PLAN-05):** no reusable unoriented fixture exists (`tests/scripts/test_measure_rc4_orientation.py` stubs `measure()`; `probe_frames()` has no `team_attacking_direction`). `conftest.py` BUILDS:
- `fixture_ref` — a REAL loaded multi-period match (a public-provider fixture / loadable public match with genuine coords + labels), NOT a hand-synthesized one — this is the geometry OK side, so it must exercise real per-period coords+labels (the non-vacuity guard against Rev 5). D3/build sources a committed real multi-period frame asset or a public pining match.
- `unoriented_ref` — a `LoadedMatch` whose `frames["team_attacking_direction"]` is all-NaN (RC4: `unlabelled_fraction == 1.0` → `UNORIENTED`).
- `mislabeled_ref` — a CONSTRUCTED `LoadedMatch` where one period's coords contradict its stored `team_attacking_direction` label (the function reflects that period; its label disagrees) → MISMATCH. Synthetic is acceptable HERE (it proves the mechanism; consistent adapters don't ship such a frame).
- `untracked_home_gk_ref` — a CONSTRUCTED `LoadedMatch` with a period whose HOME GK rows are absent but the AWAY GK is present (exercises the bound function's away-GK fallback), plus a period with NEITHER GK tracked (no-anchor → excluded). Guards r6 SPEC-10: no false MISMATCH from a NaN home-GK median.
- `empty_ref` — resolves empty (match-load raises).
- `dirty_tree` — dirties the tree + `no_new_artifacts()`.
- Memo tests point `read_validity_memo(..., research_root=tmp)` at a tmp dir with absent / `run_tree_dirty: true` / provenance-less memos, plus real `docs/research` for the present-clean case.
Loader/seam signatures + the `home_team_id`-from-`LoadedMatch` sourcing read from live HEAD per D3.

---

### Task 1: MCP package scaffold + optional extra

**Files:** Create `silly_kicks/mcp/__init__.py`, `server.py`; Modify `pyproject.toml`, `uv.lock`; Test `tests/mcp/test_registration.py`.

- [ ] **Step 1 (D3):** Confirm `from mcp.server.fastmcp import FastMCP`; add `mcp = ["mcp[cli]>=1,<2"]`; `uv sync --extra mcp` (commit uv.lock).
- [ ] **Step 2 (failing test):**
```python
# tests/mcp/test_registration.py
import pytest; pytest.importorskip("mcp")
def test_server_exposes_expected_tools():
    from silly_kicks.mcp import server
    assert server.tool_names() == {"check_orientation","diagnose_provider","validate_construct_validity"}
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** `server.py`: guarded `from mcp.server.fastmcp import FastMCP`; `app = FastMCP("silly-kicks")`; `tool_names()`; `if __name__ == "__main__": app.run()`.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_registration.py -v` — FAIL until Tasks 3–5 register tools. No commit.

### Task 2: Fail-loud read-only seams (match-load + memo-read)

**Files:** Create `silly_kicks/mcp/_load.py`, `_memo.py`; Test `tests/mcp/test_load_seam.py`, `test_memo_seam.py`.

- [ ] **Step 1 (D3):** Read `scripts/_loader_pining.py` `load_match:305`/`pining_source:358`/`LoadedMatch:281`/token`:73`; read the three memos + confirm gk_decision + territorial carry `run_commit`/`run_tree_dirty`/`verdicts`, xtgk `gate.json` does NOT.
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_load_seam.py
import pytest; from silly_kicks.mcp import _load
def test_raises_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception): _load.load_match("any-ref")
def test_raises_on_empty_resolution(empty_ref):
    with pytest.raises(Exception): _load.load_match(empty_ref)
def test_ok_on_fixture(fixture_ref):
    assert _load.load_match(fixture_ref) is not None
```
```python
# tests/mcp/test_memo_seam.py
import json, pytest; from silly_kicks.mcp import _memo
def test_reads_recorded_memo():
    v = _memo.read_validity_memo("gk_decision")
    assert "verdicts" in v and v["run_tree_dirty"] is False
def test_raises_on_absent_memo(tmp_path):
    with pytest.raises(Exception): _memo.read_validity_memo("gk_decision", research_root=tmp_path)
def test_raises_on_dirty_provenance(tmp_path):
    d = tmp_path/"gk_decision_construct_validity"; d.mkdir()
    (d/"metrics.json").write_text(json.dumps({"run_commit":"x","run_tree_dirty":True,"verdicts":{}}))
    with pytest.raises(Exception): _memo.read_validity_memo("gk_decision", research_root=tmp_path)
def test_raises_on_missing_provenance(tmp_path):
    d = tmp_path/"xtgk_possession_value"; d.mkdir()
    (d/"gate.json").write_text(json.dumps({"wc2022":{"authorising":True}}))
    with pytest.raises(Exception): _memo.read_validity_memo("xtgk_possession_value", research_root=tmp_path)
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** `_load.load_match`: resolve via `pining_source`/`load_match`; **raise** on token-absent/throw/empty (no catch-and-return; no artifact/clean-tree). `_memo.read_validity_memo`: `_FAMILY_MEMOS = {"gk_decision":"gk_decision_construct_validity/metrics.json","territorial_defense":"territorial_defense_construct_validity/metrics.json","xtgk_possession_value":"xtgk_possession_value/gate.json"}`; read JSON under `research_root` (default `docs/research`); **raise** on absent/unreadable/empty, `run_tree_dirty is True`, or missing `run_commit`/`run_tree_dirty` — clear "regenerate `<memo>` with provenance" message. Provide `recorded_verdict(memo)` → `memo["verdicts"]`.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_load_seam.py tests/mcp/test_memo_seam.py -v` → PASS. No commit.

### Task 3: `check_orientation` tool (measure + label-stripped geometry)

**Files:** Modify `server.py`; Test `tests/mcp/test_check_orientation.py`. **Consumes:** `_load.load_match`; `measure()` (`:57`); `orient_frames_to_ltr_by_geometry` (`direction.py:175`).

- [ ] **Step 1 (D3):** Confirm `measure()` keys; confirm `orient_frames_to_ltr_by_geometry(frames, *, home_team_id, on_missing_home, copy)` (`direction.py:175`) reflects a mis-oriented period's x (`x→105-x`, `:294-305`) with the away-GK fallback (`:284-293`), period-5 skip (`:278`), and no-anchor warn+skip (`:287-292`); capture the exact no-anchor warning text (to exclude those periods); resolve `home_team_id` from the loaded match. Verify: REAL `fixture_ref` → OK; constructed `mislabeled_ref` → MISMATCH; `untracked_home_gk_ref` → no false MISMATCH. BIND the function — do NOT reimplement its anchor (r6 SPEC-10) or read its uniform label output (r5 SPEC-08).
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_check_orientation.py
import pytest
from silly_kicks.mcp import server, _load
from scripts import measure_rc4_orientation as m
def test_verdict_shape(fixture_ref):
    r = server.check_orientation(fixture_ref)
    assert set(r) >= {"match_ref","provider","direction_resolved","geometry_agrees","verdict","measure"}
    assert r["verdict"] in {"OK","UNORIENTED","MISMATCH"}
def test_ok_on_correct(fixture_ref):                 # non-vacuity: correct side
    r = server.check_orientation(fixture_ref)
    assert r["verdict"] == "OK" and r["geometry_agrees"] is True
def test_mismatch_on_mislabeled(mislabeled_ref):     # non-vacuity: wrong side
    r = server.check_orientation(mislabeled_ref)
    assert r["verdict"] == "MISMATCH" and r["geometry_agrees"] is False
def test_no_false_mismatch_untracked_home_gk(untracked_home_gk_ref):   # r6 SPEC-10
    assert server.check_orientation(untracked_home_gk_ref)["verdict"] != "MISMATCH"
def test_unoriented(unoriented_ref):
    assert server.check_orientation(unoriented_ref)["verdict"] == "UNORIENTED"
def test_thin_wrapper_passthrough(fixture_ref):
    loaded = _load.load_match(fixture_ref)
    assert server.check_orientation(fixture_ref)["measure"] == m.measure(loaded)
def test_fail_loud_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception): server.check_orientation("any-ref")
def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.check_orientation(fixture_ref); assert dirty_tree.no_new_artifacts()
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** `@app.tool check_orientation`: `loaded=load_match(...)`; `raw=measure(loaded)`; `direction_resolved = raw["unlabelled_fraction"] < 1.0`. Geometry — BIND the function (do NOT reimplement its anchor): under `warnings.catch_warnings(record=True)`, `out = orient_frames_to_ltr_by_geometry(loaded.frames, home_team_id=…, on_missing_home="warn", copy=True)`; collect the periods named in its no-anchor warnings (`no_anchor`). For each period P in {1,2,3,4} with a present home label AND P ∉ `no_anchor`: `reflected = not np.allclose(loaded.frames.loc[P,"x"], out.loc[P,"x"], equal_nan=True)`; geometric home direction = `"rtl" if reflected else "ltr"`; `geometry_agrees = all(geometric == stored home label)` over those eligible periods (True iff `direction_resolved` and all eligible agree; no eligible period ⇒ True). `verdict = "UNORIENTED" if not direction_resolved else "MISMATCH" if not geometry_agrees else "OK"`. Return `{match_ref, provider, direction_resolved, geometry_agrees, verdict, measure: raw}`. `home_team_id` from the loaded match. Ids via `id_compat`. No catch around the load.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_check_orientation.py -v` → PASS (both non-vacuity sides). No commit.

### Task 4: `diagnose_provider` tool (lib-only aspects, JSON-safe all aspects)

**Files:** Modify `server.py`; Test `tests/mcp/test_diagnose_provider.py`. **Consumes:** `_load.load_match`; lib `validate_gk_position_clamp`/`GkClampDiagnosis`, `detect_input_convention`/`DetectionResult`, `validate_id_dtypes`/`IdDtypeDiagnosis`.

- [ ] **Step 1 (D3):** Confirm `GkClampDiagnosis` tuple-keyed dicts; `DetectionResult.diagnostics` numpy scalars; `validate_id_dtypes(..., on_mismatch="warn")` + `IdDtypeDiagnosis` fields.
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_diagnose_provider.py
import json, pytest
from silly_kicks.mcp import server, _load
from silly_kicks.mcp.server import _json_safe   # shared serializer
ASPECTS = ["keeper","convention","id_dtype"]
@pytest.mark.parametrize("aspect", ASPECTS)
def test_aspect_shape(fixture_ref, aspect):
    r = server.diagnose_provider("skillcorner", fixture_ref, aspect)
    assert set(r) >= {"provider","match_ref","aspect","findings","flags"}
@pytest.mark.parametrize("aspect", ASPECTS)
def test_findings_json_serializable(fixture_ref, aspect):       # tuple keys + numpy scalars
    json.dumps(server.diagnose_provider("skillcorner", fixture_ref, aspect)["findings"])
def test_thin_wrapper_equality_id_dtype(fixture_ref):
    from silly_kicks.tracking import validate_id_dtypes
    loaded = _load.load_match(fixture_ref)
    diag = validate_id_dtypes(loaded.actions, loaded.frames, on_mismatch="warn")
    assert server.diagnose_provider("skillcorner", fixture_ref, "id_dtype")["findings"] == _json_safe(diag)
def test_fail_loud_without_token(monkeypatch):
    monkeypatch.delenv("PINING_FOR_THE_DATA_TOKEN", raising=False)
    with pytest.raises(Exception): server.diagnose_provider("skillcorner","any-ref","keeper")
def test_read_only_on_dirty_tree(dirty_tree, fixture_ref):
    server.diagnose_provider("skillcorner", fixture_ref, "convention"); assert dirty_tree.no_new_artifacts()
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** Add `_json_safe(obj)`: `dataclasses.asdict` if a dataclass, then recurse — stringify TUPLE dict keys (`"|".join(map(str,k))`), coerce numpy (`np.floating→float`, `np.integer→int`, `np.bool_→bool`, `np.ndarray→list`). `@app.tool diagnose_provider`: `load_match` → dispatch {keeper: `validate_gk_position_clamp`; convention: `detect_input_convention`; id_dtype: `validate_id_dtypes(..., on_mismatch="warn")`} → `findings = _json_safe(diag)`; `flags` from diagnosis booleans. Ids via `id_compat`. No catch around load.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_diagnose_provider.py -v` → PASS (round-trip all 3 aspects). No commit.

### Task 5: `validate_construct_validity` tool (memo reader; xtgk fail-loud — Karsten)

**Files:** Modify `server.py`; Test `tests/mcp/test_validate_construct_validity.py`. **Consumes:** `_memo.read_validity_memo`/`recorded_verdict`.

- [ ] **Step 1 (D3):** Confirm gk_decision + territorial carry provenance + `verdicts`; xtgk `gate.json` does NOT (→ fail-loud; regen deferred, Karsten 2026-09-30).
- [ ] **Step 2 (failing tests):**
```python
# tests/mcp/test_validate_construct_validity.py
import pytest
from silly_kicks.mcp import server, _memo
PROVENANCED = ["gk_decision","territorial_defense"]
@pytest.mark.parametrize("family", PROVENANCED)
def test_verdict_shape(family):
    r = server.validate_construct_validity(family)
    assert set(r) >= {"metric_family","memo_path","run_commit","run_tree_dirty","run_commit_is_head","verdict"}
    assert r["run_tree_dirty"] is False
@pytest.mark.parametrize("family", PROVENANCED)
def test_thin_wrapper_equality(family):
    assert server.validate_construct_validity(family)["verdict"] == _memo.recorded_verdict(_memo.read_validity_memo(family))
def test_xtgk_raises_until_provenance():
    with pytest.raises(Exception): server.validate_construct_validity("xtgk_possession_value")
def test_fail_loud_on_absent_memo(tmp_path):
    with pytest.raises(Exception): server.validate_construct_validity("gk_decision", research_root=tmp_path)
def test_read_only_on_dirty_tree(dirty_tree):
    server.validate_construct_validity("gk_decision"); assert dirty_tree.no_new_artifacts()
```
- [ ] **Step 3:** Run → FAIL.
- [ ] **Step 4:** `@app.tool validate_construct_validity(metric_family, *, research_root=None)`: `memo = read_validity_memo(...)` (raises on absent/dirty/missing-provenance); return `{metric_family, memo_path, run_commit, run_tree_dirty, run_commit_is_head: <memo run_commit == current HEAD>, verdict: recorded_verdict(memo)}`. No corpus load, no clean-tree.
- [ ] **Step 5 (verify):** `pytest tests/mcp/test_validate_construct_validity.py -v` → PASS. No commit.

**Acceptance:** gk_decision + territorial surface the recorded verdict + provenance and pass thin-wrapper + dirty-tree; xtgk RAISES until regenerated; all three dispatchable.

### Task 6: Registration + docs

**Files:** Create `.mcp.json`, `docs/howto/mcp.md`; re-run `tests/mcp/test_registration.py`.

- [ ] **Step 1:** `.mcp.json`: `{"mcpServers":{"silly-kicks":{"command":"python","args":["-m","silly_kicks.mcp.server"],"cwd":"."}}}`.
- [ ] **Step 2:** `docs/howto/mcp.md`: `uv sync --extra mcp` install, the `.mcp.json` block, manual registration for other MCP clients, a one-line description of each tool, and the xtgk-memo-needs-provenance note.
- [ ] **Step 3 (verify):** `.mcp.json` valid JSON; `pytest tests/mcp/test_registration.py -v` → PASS. No commit.

### Final: single Phase-2 commit (approval-gated)

- [ ] **Step 1:** `pytest tests/mcp -v` → all green (fail-loud match-load + memo-read incl. missing-provenance; thin-wrapper equality ×3; geometry non-vacuity BOTH sides; JSON round-trip ×3 aspects; dirty-tree ×3; UNORIENTED; registration). `ruff check silly_kicks/ tests/` + `pyright` on the new package.
- [ ] **Step 2:** Show Karsten the full file list + diff. **STOP.**
- [ ] **Step 3:** Only on explicit "yes", ONE commit (all Phase-2 files) on `karsten/agent-support-phase2`. Do not push unless separately approved.

---

## Self-Review

**Spec coverage (Rev 7):** server §3 → T1; fail-loud seams §4a → T2; tools §5 → T3/4/5; geometry via BOUND `orient_frames_to_ltr_by_geometry` (reflection input-vs-output, no-anchor periods excluded) §4b/§5.1 → T3 (OK real / MISMATCH constructed / untracked-home-GK); `_json_safe` all aspects §4b/§7 → T4 (round-trip ×3); xtgk fail-loud §4b/§5.3 → T5 + T2; registration §6 → T6. **§7 tests present:** verdict-shape, geometry non-vacuity (OK + MISMATCH), thin-wrapper, JSON round-trip ×3, dirty-tree ×3, fail-loud (token + memo absent/dirty/missing-provenance), UNORIENTED, registration. No CLI-regression.

**Scope:** all three families dispatchable (D2); `geometry_agrees` restored + non-vacuous; `coords` + xtgk regen + live `validate_*` DEFERRED (spec §11); xtgk fail-loud CONFIRMED (Karsten). Nothing silently cut.

**Placeholder scan:** test bodies assert the Rev-5 contract (verdict enum, RAISES incl. missing-provenance, passthrough/`_json_safe` equality, `json.dumps`, both non-vacuity sides). Fixtures are BUILT in conftest (unoriented all-NaN label; mislabeled wrong-label); no "reuse a non-existent fixture" instruction.

**Type/name consistency:** `_load.load_match`, `_memo.read_validity_memo`/`recorded_verdict`/`_FAMILY_MEMOS`, `server.{check_orientation,diagnose_provider,validate_construct_validity,tool_names,_json_safe}`, verdict keys — referenced identically across T1–6 and match spec §4/§5.

## Review log

- **Rev 1/2** — r1 REQUEST CHANGES → r2 APPROVE (vs `3ca609f`).
- **Rev 3** — zero-extraction lib binds + memo reader (D3 HEAD).
- **Rev 4** — restored `geometry_agrees` + 2nd bind; xtgk fail-loud; keeper JSON-safe; RC4 path.
- **Rev 5, 2026-09-30** — applied r4: strip-label geometry diff (later shown still wrong), `_json_safe` all aspects (PLAN-10), fixtures built, xtgk fail-loud (PLAN-08).
- **Rev 6, 2026-09-30** — read the per-period home-GK-median-x anchor (fixed r5 vacuity) but reimplemented only the home-GK branch.
- **Rev 7, 2026-09-30** — applies r6 REQUEST CHANGES (PLAN-11): the Rev-6 reimplementation dropped the away-GK fallback / period-5 / no-anchor skip, forcing "rtl" on a NaN home-GK median (false MISMATCH on untracked-GK frames — routine in broadcast tracking). Rev 7 BINDS `orient_frames_to_ltr_by_geometry` read-only and reads the per-period decision from input-vs-output coord reflection, excluding no-anchor periods (via its warning) + period 5 — inheriting fallback/period-5/skip; adds an untracked-home-GK test. Fourth geometry correction owned — lesson: BIND the seam, don't reimplement it. Awaiting independent re-review. Reports: `2026-09-30-silly-kicks-agent-support-phase2-plan{,-r2,-r3,-r4,-r5,-r6}.md`.
