# silly-kicks Agent-Support — Phase 2 (MCP) Design Spec

**Date:** 2026-09-30
**Status:** REVISED (Rev 7, 2026-09-30) applying the r6 re-review (REQUEST CHANGES). Rev 7 stops REIMPLEMENTING the geometric anchor and instead BINDS the real function: `geometry_agrees` runs `orient_frames_to_ltr_by_geometry` read-only and reads the per-period decision from INPUT-vs-OUTPUT coord reflection. Rev 6 copied only the home-GK branch (`direction.py:281-283`) and dropped the away-GK fallback (`:284-293`), the period-5 exclusion, and the no-anchor skip — so a period with no tracked home GK gave `median==NaN`, `NaN<52.5==False` → forced "rtl" → false MISMATCH (broadcast tracking loses the GK routinely). Binding the function inherits fallback/period-5/skip. Karsten kept `geometry_agrees` as a DEFENSIVE regression check (2026-09-30): OK-side tested on a REAL match, MISMATCH-side on a CONSTRUCTED inconsistent frame, plus an untracked-home-GK test. Rev 5's JSON-safety (all aspects) + built fixtures + xtgk fail-loud stand. Awaiting independent re-review. Not yet implemented.
**Target repo:** `silly-kicks` (this tree).
**Parent spec:** `2026-09-29-silly-kicks-agent-support-neutral-layout-design.md` (APPROVED, r2). Details §5 (Phase 2); Approach A + neutral-core/thin-veneer inherited.
**Prerequisite:** Phase 1 landed on `main` (`4049db9`, PR #263).

---

## 0. Evolution (D3 investigation + r3/r4 re-reviews)

Rev 2 (vs `3ca609f`) was never executed. D3 confirm-at-HEAD (`4049db9`) + two re-review rounds established the real seams:

1. **`check_orientation`** — `scripts/measure_rc4_orientation.py:57 measure(loaded)` returns raw metrics (`unlabelled_fraction`, flip stats, `orientation_warnings`); `direction_resolved` + `OK`/`UNORIENTED` derive from `unlabelled_fraction`. **`geometry_agrees`/`MISMATCH`** BIND `orient_frames_to_ltr_by_geometry` (`direction.py:175`, run read-only `copy=True`, `on_missing_home="warn"`) rather than reimplementing its anchor: the function reflects a period's coords (`x→105-x`) iff that period is mis-oriented, using the home-GK-median-x anchor with the away-GK FALLBACK (`:284-293`), the period-5 exclusion (`:278`), and the no-anchor SKIP+warn (`:287-292`). The tool detects the per-period decision by INPUT-vs-OUTPUT coord comparison: a reflected period ⇒ home was attacking RTL in the STORED frame, an unreflected eligible period ⇒ LTR; compare to the stored per-period `team_attacking_direction` home label. Periods the function could not anchor (its no-anchor warning) and period 5 are EXCLUDED from the comparison — do NOT read them as a direction (the Rev-6 reimplementation forced "rtl" on a NaN home-GK median and false-fired; r6 SPEC-10). **What NOT to use (r5, executed):** the function's own label output is UNIFORM home=ltr/away=rtl (`:307-310`) after flipping coords — it carries no per-period direction, so diffing it is vacuous. `compute_attacking_direction:137` is flag-derived → circular → not used. Because consistent adapters ship coord-label-consistent frames, `MISMATCH` is a DEFENSIVE future-regression check (Karsten kept it 2026-09-30).
2. **`diagnose_provider`** — `calibration_runs/diag_*.py` (Rev-2 ids/coords source) does not exist. `id_dtype`→`validate_id_dtypes` is the ADR-019 trap tripwire and the full id restore. `keeper`→`GkClampDiagnosis`, `convention`→`detect_input_convention` bind lib. `coords` has no seam → deferred.
3. **`validate_construct_validity`** — heterogeneous owner-corpus batteries → a memo READER is the read-only form. The xtgk memo `docs/research/xtgk_possession_value/gate.json` has no `run_commit`/`run_tree_dirty`/`verdicts` (and `xtgk_v2_construct_validity/` has no `metrics.json`); regenerating it is an owner-corpus run (deferred). The reader fails loud on missing-provenance; xtgk raises until regenerated (Karsten confirmed 2026-09-30).

## 1. Context

The orientation NULL-direction no-op (fixed 4×) and the `str()`-on-float id trap (4×) "look healthy," so only a runnable tripwire catches them (parent spec §1/§10). MCP is an open protocol; any client uses the server.

## 2. Goals / Non-Goals

**Goals**
- Three runnable tools: `check_orientation`, `diagnose_provider`, `validate_construct_validity`.
- Server is a **pure adapter** — each tool composes a **fail-loud read-only seam** (match-load or memo-read) with **bound** compute seams. The only server-side computation is comparing two bound seams' outputs (the geometry-vs-label diff, §4b); it derives no new metric.
- Core library stays dependency-light: `mcp[cli]` is an **optional extra**.
- Cross-tool: one neutral server, per-client registration.

**Non-Goals**
- **No compute-seam extraction and no library/`scripts/` behavior change.** Tools BIND existing public lib functions and READ existing committed memos.
- **No new metric/analysis behavior** — with one narrow, explicit exception (r3 P2-SPEC-06): `check_orientation` runs `orient_frames_to_ltr_by_geometry` read-only and compares the adapter's per-period `team_attacking_direction` label against the function's per-period reflect/no-reflect decision (observed input-vs-output). Binding the function's decision + comparing to a label, not a new metric or a reimplementation of its anchor.
- No heavy corpus-scale wrappers or **live `validate_*` corpus runs** (incl. regenerating the xtgk memo) — deferred (§11).
- No non-Claude activation shims this phase (deferred).

## 3. Server design

- **`silly_kicks/mcp/server.py`** — `FastMCP`, stdio, `python -m silly_kicks.mcp.server`.
- **Dependency:** `[project.optional-dependencies] mcp = ["mcp[cli]>=1,<2"]` (pin `<2`). Guarded import; installed via `uv sync --extra mcp`; `uv.lock` committed.
- **Server owns no logic** beyond arg-normalization (ids via `silly_kicks.id_compat`) and the §4b geometry-vs-label diff. Returns JSON-serializable dicts mirroring the seam shapes (Hyrum-safe).
- **Read-only invariant:** every bound seam is read-only — no `require_clean_tree`, no `run_commit` write, no artifact write. Never calls a driver `run()`/`main()`.

## 4. Seams

### 4a. Fail-loud read-only seams

- **Match-load** (`check_orientation`, `diagnose_provider`) — `silly_kicks/mcp/_load.py::load_match(match_ref, provider=None) -> LoadedMatch` over `scripts/_loader_pining.py` (`load_match:305`/`pining_source:358`/`LoadedMatch(match_id,actions,frames):281`/token `:73`). **RAISES** on tokenless / thrown / empty-degenerate resolution (never a verdict on a bad load; ADR-052 D14). Compute seams receive loaded data, stay pure.
- **Memo-read** (`validate_construct_validity`) — `silly_kicks/mcp/_memo.py::read_validity_memo(metric_family, *, research_root=None) -> dict`. **RAISES** on absent / unreadable / empty / invalid-JSON, on `run_tree_dirty: true`, and on a memo MISSING `run_commit`/`run_tree_dirty` (untrustworthy provenance — the xtgk gate.json case), with a clear "regenerate `<memo>` with provenance" message (never a `KeyError`).

### 4b. Bound compute seams (no extraction) + JSON-safe serialization

- `check_orientation` — `measure(loaded) -> dict` (`measure_rc4_orientation.py:57`; raw metrics) → `direction_resolved = unlabelled_fraction < 1.0`. Plus a bind of `orient_frames_to_ltr_by_geometry(loaded.frames, home_team_id=…, on_missing_home="warn", copy=True)` (`direction.py:175`): per period, `reflected = (output x ≠ input x)` (the function only reflects `x→105-x` or leaves unchanged); geometric home direction = `"rtl"` if reflected else `"ltr"`; `geometry_agrees = (geometric direction == stored per-period home label)` over eligible periods (present label; period ∈ {1,2,3,4}; the function did NOT emit a no-anchor warning for it — excluded periods are neither agree nor disagree). Needs `is_goalkeeper`/`team_id` (on frames) + `home_team_id` from the loaded match (D3). The away-GK fallback / period-5 / no-anchor handling all come from the bound function, not reimplemented (§0.1).
- `diagnose_provider` aspect → lib bind:
  - `keeper` → `validate_gk_position_clamp(...) -> GkClampDiagnosis` (`utils.py:878`; dataclass `schema.py:404`).
  - `convention` → `detect_input_convention(...) -> DetectionResult` (`orientation.py:323`; result `:282`).
  - `id_dtype` → `validate_id_dtypes(actions, frames, on_mismatch="warn") -> IdDtypeDiagnosis` (`utils.py:1148`) — the ADR-019 `str()`-on-float tripwire; `"warn"` so a mismatch is a RETURNED finding, never a crash.
  - Dropped/deferred: `ids` folded into `id_dtype`; `coords` (no seam, §11).
- **JSON-safe serialization (r4 SPEC-09):** findings are serialized through a shared `_json_safe(obj)` that `dataclasses.asdict`s, then recursively (a) stringifies TUPLE dict keys (`GkClampDiagnosis.clamped_units`/`ceiling_by_unit`/`pileup_by_unit` keyed `(game_id, team_id)` → `"<game_id>|<team_id>"`) and (b) coerces NUMPY scalars to Python (`DetectionResult.diagnostics` carries numpy group-means; `np.floating→float`, `np.integer→int`, `np.bool_→bool`, `np.ndarray→list`). Applies to ALL aspects — a `json.dumps(findings)` round-trip test per aspect guards it.
- `validate_construct_validity` — the memo-read seam (§4a) is the whole compute. Family → memo:
  - `gk_decision` → `docs/research/gk_decision_construct_validity/metrics.json` (provenance + `verdicts`).
  - `territorial_defense` → `docs/research/territorial_defense_construct_validity/metrics.json` (provenance + `verdicts`).
  - `xtgk_possession_value` → `docs/research/xtgk_possession_value/gate.json` — provenance-less, so the reader RAISES (fail-loud) until regenerated (owner-corpus run, §11). Dispatchable.

**Test-dir placement:** all seams are lib binds or memo reads → tool tests live in `tests/mcp/`; no `tests/scripts/` regression.

## 5. The three tools

Each tool: **fail-loud seam → bound compute → JSON dict**. Failure raises; never a verdict.

1. **`check_orientation(match_ref: str, provider: str | None = None) -> dict`**
   - Load (fail loud). `raw = measure(loaded)`; `direction_resolved = raw["unlabelled_fraction"] < 1.0`. `geometry_agrees` from binding `orient_frames_to_ltr_by_geometry` (per-period reflect/no-reflect) vs the stored per-period label, over eligible periods only (§4b).
   - `verdict`: `"UNORIENTED"` if not `direction_resolved`; `"MISMATCH"` if `direction_resolved` and not `geometry_agrees`; else `"OK"`.
   - Returns `{match_ref, provider, direction_resolved, geometry_agrees, verdict: "OK"|"UNORIENTED"|"MISMATCH", measure: <raw dict>}`.

2. **`diagnose_provider(provider: str, match_ref: str, aspect: "keeper"|"convention"|"id_dtype") -> dict`**
   - Load (fail loud), dispatch to the bound lib diagnostic (§4b). Returns `{provider, match_ref, aspect, findings: _json_safe(<diagnosis>), flags: [str]}` — `flags` from the diagnosis boolean (`GkClampDiagnosis.clamped`→`"gk_clamped"`; `IdDtypeDiagnosis.has_mismatch`→`"id_dtype_mismatch"`; `DetectionResult.convention is None`→`"convention_ambiguous"`). `findings` is JSON-serializable for every aspect.

3. **`validate_construct_validity(metric_family: "gk_decision"|"territorial_defense"|"xtgk_possession_value") -> dict`**
   - Read the memo (fail loud on absent/dirty/missing-provenance, §4a). For the two provenance-bearing families: `{metric_family, memo_path, run_commit, run_tree_dirty: false, run_commit_is_head: bool, verdict: <recorded verdict block>}`. Surfaces the recorded result; no re-run, no synthesised GO/NO_GO. `xtgk_possession_value` RAISES until its memo carries provenance. `run_commit_is_head` is an informational staleness hint (ADR-056 declare-not-enforce). No `corpus_ref`, no `gate` param.

## 6. Registration

- Claude: `.mcp.json` → `{ "mcpServers": { "silly-kicks": { "command":"python","args":["-m","silly_kicks.mcp.server"],"cwd":"." } } }`.
- Other clients: `docs/howto/mcp.md`. Identical server.

## 7. Testing (`tests/mcp/`)

- **Fail-loud seam (ADR-052 D14):** match-load on tokenless/failed/empty → RAISES; memo-read on absent/dirty/missing-provenance → RAISES (incl. the committed xtgk gate.json).
- **Verdict-shape:** `check_orientation` `verdict ∈ {OK,UNORIENTED,MISMATCH}` + `geometry_agrees`; `diagnose_provider` per aspect; `validate_construct_validity` for the two provenance-bearing families (xtgk asserted to RAISE).
- **Geometry NON-VACUITY (r5 SPEC-08):** OK side on a REAL loaded multi-period match → `verdict=="OK"`, `geometry_agrees is True`. MISMATCH side on a CONSTRUCTED frame where a period's coords contradict its stored label → `verdict=="MISMATCH"`, `geometry_agrees is False`.
- **Untracked-home-GK (r6 SPEC-10):** a match with a period whose HOME GK is absent but the AWAY GK is present → the bound function's away-GK fallback resolves it, no false MISMATCH; and a period with NEITHER GK tracked → that period is EXCLUDED (no-anchor), no false MISMATCH. Guards the Rev-6 forced-"rtl"-on-NaN bug.
- **Thin-wrapper equality:** `check_orientation.measure` == `measure(loaded)`; each `diagnose_provider.findings` == `_json_safe(<directly-bound diagnosis>)`; `validate_construct_validity.verdict` == the memo's recorded verdict block.
- **JSON round-trip (r4 SPEC-09):** `json.dumps(diagnose_provider(...)["findings"])` succeeds for EACH aspect (keeper/convention/id_dtype).
- **Dirty-tree read-only:** each tool writes no artifact / trips no clean-tree gate.
- **RC4 tripwire:** `check_orientation` → `UNORIENTED` on an unoriented fixture.
- **Registration:** exactly `{check_orientation, diagnose_provider, validate_construct_validity}`.

## 8. Rollout & approval gates

Separate PR, feature branch `karsten/agent-support-phase2` (no worktree), one fully-tested commit gated on explicit approval; push a separate gate. Build order: `check_orientation`, `diagnose_provider`, `validate_construct_validity`. Independent re-review of this spec + plan before coding; nothing deferred/cut without explicit approval.

## 9. Risks & mitigations

- ADR-052 D14 silent tripwire → fail-loud seams + tests.
- Vacuous or incomplete geometry check → BIND `orient_frames_to_ltr_by_geometry` (inherits away-GK fallback / period-5 / no-anchor skip), read per-period reflection input-vs-output, exclude no-anchor periods; REAL-match OK side + constructed MISMATCH side + untracked-home-GK test (§7). MISMATCH is a defensive future-regression check.
- JSON-unsafe findings (tuple keys, numpy scalars) → `_json_safe` all aspects + per-aspect round-trip.
- Server drift → thin-wrapper equality.
- Clean-tree/artifact side effect → read-only invariant + dirty-tree test.
- `mcp` extra absent → guarded import.
- xtgk memo missing provenance → reader raises, clear regen message; dispatchable.
- `id_dtype` uses `on_mismatch="warn"`.
- Geometry bind needs `home_team_id` + a label-stripped copy → D3 build step.
- No reusable unoriented fixture at HEAD → the tests BUILD unoriented + mislabeled fixtures (§7; a synthetic match or monkeypatched loader returning a `LoadedMatch`).

## 10. Decisions

- **D1 (corpus scope):** SUPERSEDED — Tool 3 reads the recorded memo.
- **D2 (all three families):** HELD — all dispatchable; xtgk raises until provenance regen.
- **xtgk disposition (r4 SPEC-07, Karsten 2026-09-30):** SHIP FAIL-LOUD — regenerating gate.json is a deferred owner-corpus run.
- **Rev-7 contracts:** (a) `check_orientation` → OK/UNORIENTED/MISMATCH, `geometry_agrees` by BINDING `orient_frames_to_ltr_by_geometry` read-only (per-period input-vs-output reflection vs the stored per-period label; away-GK fallback / period-5 / no-anchor skip inherited from the function, not reimplemented; no-anchor periods excluded), a defensive regression check Karsten kept 2026-09-30, raw `measure()` pass-through; (b) `diagnose_provider` → lib-only `{keeper, convention, id_dtype}`, JSON-safe all aspects, `coords` deferred; (c) `validate_construct_validity` → memo reader, xtgk fail-loud.
- **D3 — seam signatures/paths** confirmed vs `4049db9`; re-confirmed per task at build.

## 11. Deferred

- Regenerating the xtgk memo with provenance + live `validate_*` corpus runs + heavy corpus wrappers (owner corpus).
- `diagnose_provider` `coords` aspect (no seam at HEAD).
- `.cursor/rules`/non-Claude shims; conventions-reviewer subagent; howto CI link-check.

## 12. Review log

- **Rev 2** — r1 REQUEST CHANGES → r2 APPROVE (vs `3ca609f`).
- **Rev 3** — D3 HEAD investigation → zero-extraction lib binds + memo reader. **Error owned:** claimed `geometry_agrees` unbacked; it is backed.
- **Rev 4** — restored `geometry_agrees` + 2nd bind; xtgk fail-loud; keeper JSON-safe; RC4 path.
- **Rev 5, 2026-09-30** — applied r4: strip-label geometry diff (later shown still wrong), `_json_safe` all aspects (SPEC-09), fixtures built, xtgk fail-loud (SPEC-07).
- **Rev 6, 2026-09-30** — read the per-period home-GK-median-x anchor directly (fixed r5 vacuity), but reimplemented only the home-GK branch.
- **Rev 7, 2026-09-30** — applies r6 REQUEST CHANGES (SPEC-10): the Rev-6 reimplementation dropped the away-GK fallback / period-5 / no-anchor skip, forcing "rtl" on a NaN home-GK median (false MISMATCH where broadcast tracking loses the GK). Rev 7 BINDS `orient_frames_to_ltr_by_geometry` read-only and reads the per-period decision from input-vs-output coord reflection, excluding no-anchor periods — inheriting the function's fallback/period-5/skip rather than reimplementing them; adds an untracked-home-GK test. **Fourth geometry correction owned** (Rev 3 unbacked, Rev 4/5 vacuous, Rev 6 incomplete reimpl) — lesson: BIND the seam, do not reimplement it. Awaiting independent re-review. Reports: `2026-09-30-silly-kicks-agent-support-phase2-mcp-spec{,-r2,-r3,-r4,-r5,-r6}.md`.
