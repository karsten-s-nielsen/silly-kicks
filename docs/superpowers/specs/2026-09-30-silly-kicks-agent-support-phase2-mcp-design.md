# silly-kicks Agent-Support — Phase 2 (MCP) Design Spec

**Date:** 2026-09-30
**Status:** APPROVED 2026-09-30 (independent review, round 2). Rev 2 applies findings P2-SPEC-01..03 + an empty-resolution hardening. Not yet implemented. Authoring session holds `silly-kicks` READ-ONLY — nothing written to it, nothing committed.
**Target repo:** `silly-kicks` (`D:\Development\karstenskyt__silly-kicks`)
**Parent spec:** `2026-09-29-silly-kicks-agent-support-neutral-layout-design.md` (APPROVED, review r2). This spec details that spec's **§5 (Phase 2)**; Approach A and the neutral-core / thin-veneer principle are inherited, not re-litigated.
**Prerequisite:** Phase 1 (docs + shims) landed and in use (parent spec §5 build trigger). Phase 2 is a separate PR.

---

## 1. Context

Phase 2 adds the capability layer the parent spec deferred: a **small MCP server** exposing three checks that a document cannot enforce because they must *execute*. Evidence (parent spec §1/§10): the orientation defect class was fixed 4× (+1 adjacent fixture fix) and the `str()`-on-float id trap recurred 4× — both are runtime failures that "look healthy," so only a runnable tripwire catches them. The server is the cross-tool capability (MCP is an open protocol; any MCP client uses it, not only Claude).

## 2. Goals / Non-Goals

**Goals**
- Three runnable tools: `check_orientation`, `diagnose_provider`, `validate_construct_validity`.
- Server is a **pure adapter** — no analysis logic of its own; each tool composes a **read-only load seam** (fail-loud) with a **pure compute seam** (§4).
- Keep the core library dependency-light: `mcp[cli]` is an **optional extra**.
- Cross-tool: one neutral server, per-client registration.

**Non-Goals**
- No heavy corpus-scale wrappers (`run_calibration`/`load_corpus`/`compute_metric_on_corpus`) — deferred (§11).
- No new metric/analysis behavior. The only library change is **compute-seam extraction** (§4): exposing existing compute as importable pure functions, current CLI behavior unchanged.
- No non-Claude activation shims in this phase (deferred).

## 3. Server design

- **`silly_kicks/mcp/server.py`** — a `FastMCP` server (from `mcp[cli]`), stdio transport, module-runnable via `python -m silly_kicks.mcp.server`.
- **Dependency:** `[project.optional-dependencies] mcp = ["mcp[cli]>=1,<2"]` (pin `<2`: mcp 2.x moved FastMCP). Core install stays server-free; the server import is guarded.
- **The server owns no logic.** Each `@tool` function: normalizes args (ids via `silly_kicks.id_compat` — never raw `str()`/`==`), runs the **load seam** then the **pure compute seam** (§4), and returns a JSON-serializable dict mirroring the existing `docs/research/*/metrics.json` verdict shape (Hyrum-safe).
- **Read-only invariant (load-bearing):** every bound seam returns a dict with **no `require_clean_tree` gate, no `run_commit`, no artifact write** — a tripwire is called mid-work on a dirty tree. The server never calls a driver's `run()`/`main()`.

## 4. Seams (P2-SPEC-01 + P2-SPEC-02 + P2-SPEC-03)

Each tool is **two seams composed**, never one:

### 4a. Read-only load seam — MUST FAIL LOUD
Tools take **ref strings** (`match_ref`, `corpus_ref`); the bound compute seams consume **loaded data**. A read-only load seam resolves `ref → LoadedMatch/LoadedCorpus` via the existing `pining_source` load path (env `PINING_FOR_THE_DATA_TOKEN`; bounded ref).

**Critical (ADR-052 D14):** `measure_rc4_orientation.measure()` (`:57`) deliberately **does not load and does not catch** — loading-inside-compute + `except` was removed precisely because it made the check "structurally unable to fail": a tokenless run caught the load error and wrote a healthy-looking artifact (a false negative on the top defect class). Therefore:
- The load seam is **separate** from the pure compute seam.
- On a tokenless, failed, OR **empty/degenerate** load (tokenless pining can return empty refs with exit 0 for some providers — not a thrown exception), the tool **RAISES** (surfaces the error to the caller) — it MUST NOT return an `OK`/`UNORIENTED`/`GO` verdict. Silent degradation is the exact failure the tripwire exists to catch.
- Compute seams receive already-loaded data and stay pure (no load, no catch-and-continue).

### 4b. Pure compute seams — per tool (add a returns-a-dict fn; leave `run()`/`main()` calling it, CLI unchanged)
- **`check_orientation`** — no extraction needed; bind `measure_rc4_orientation.measure(loaded) -> dict` (`:57`), distinct from `run()` (`:95`) / `main()` (`:194`).
- **`validate_construct_validity`** — extract a pure `validate(loaded, gate) -> verdict_dict` from each targeted `validate_*.py` (`validate_territorial_defense.py`, `validate_gk_decision.py`, `validate_xtgk_possession_value.py`); each script's `main()` calls it and still writes its memo/artifact as today.
- **`diagnose_provider`** — aspect → seam mapping (reuse existing LIB diagnostics where they exist; do NOT reinvent):
  - `ids`, `coords` → extract a pure `probe()` from the `calibration_runs/diag_*.py` one-offs (`diag_gs_frames.py`, `diag_idsse_coords.py`, `diag_idsse_carrier.py`, `diag_idsse_actor_dist.py`).
  - `keeper` → bind the existing lib diagnostic `GkClampDiagnosis` / `validate_gk_position_clamp` (in `silly_kicks/`) — no extraction.
  - `convention` → bind the existing lib `detect_input_convention` (in `silly_kicks/`) — no extraction.

**Refactor acceptance:** existing CLI entry points produce byte-identical output/artifacts to pre-refactor (regression); each pure compute seam returns a dict with no side effects on a dirty tree.

**Test-dir placement (P2-SPEC-03):** seams extracted from `scripts/` + `calibration_runs/` are tested under **`tests/scripts/`**; seams bound from the `silly_kicks/` lib (`keeper`/`convention`) are tested in their existing **lib test dirs**.

## 5. The three tools

Each tool: **load (fail-loud) → pure compute → JSON dict**. Load failure raises; it is never a verdict.

1. **`check_orientation(match_ref: str, provider: str | None = None) -> dict`**
   - Load `match_ref` (fail loud), then `measure()`. Returns `{match_ref, provider, direction_resolved: bool, geometry_agrees: bool, verdict: "OK"|"UNORIENTED"|"MISMATCH", detail: str}`. Catches the NULL-`team_attacking_direction` silent no-op (RC4 pining case) — and, per §4a, a tokenless load raises rather than returning `OK`.

2. **`diagnose_provider(provider: str, match_ref: str, aspect: "ids"|"coords"|"keeper"|"convention") -> dict`**
   - Load `match_ref` (fail loud), dispatch on `aspect` to the mapped seam (§4b). Returns `{provider, match_ref, aspect, findings: {...}, flags: [str]}`.

3. **`validate_construct_validity(metric_family: str, corpus_ref: str, gate: "predictive"|"discriminating"|"responsiveness") -> dict`**
   - Load `corpus_ref` (fail loud), then `validate()`. Returns GO/NO-GO in the existing `metrics.json` shape: `{metric_family, gate, verdict: "GO"|"NO_GO", value, threshold, caveats: [str]}`.
   - **`corpus_ref` is a BOUNDED reference** (small/held-out corpus id), not a full-season run — keeps it a fast tripwire, clear of the deferred heavy corpus wrappers. (Open decision D1.)

## 6. Registration (neutral server, per-client)

- Claude Code: `.mcp.json` → `{ "mcpServers": { "silly-kicks": { "command": "python", "args": ["-m","silly_kicks.mcp.server"], "cwd": "." } } }` (resolved against the repo venv).
- Other MCP clients (Codex/Cursor/…): documented manual registration in `docs/howto/mcp.md`. Identical server.

## 7. Testing

- **Load-seam fail-loud test (P2-SPEC-01 regression):** each tool, on a tokenless, failed, OR **empty/degenerate** load (empty-ref resolution with exit 0), **RAISES** — it must NOT return `OK`/`UNORIENTED`/`GO`. This guards the ADR-052 D14 anti-pattern directly.
- **Seam unit tests:** each pure compute seam returns the expected dict on a fixture (script/calibration seams in `tests/scripts/`, lib seams in their lib test dirs); plus a **CLI-output regression** that each refactored script's output/artifact is unchanged post-refactor.
- **`tests/mcp/`:**
  - Per-tool verdict-shape assertions on existing per-provider fixtures.
  - **Thin-wrapper equality:** tool output == the bound compute seam's result on the same loaded fixture (guards server drift).
  - **Dirty-tree read-only assertion:** each tool runs on a deliberately dirty tree without tripping `require_clean_tree` or writing an artifact.
  - **RC4 tripwire regression:** `check_orientation` returns `UNORIENTED` on the known unoriented-pining fixture.

## 8. Rollout & approval gates

- Separate PR from Phase 1, on its own **feature branch off the default branch** (no worktree), **one fully-tested commit**, gated on Karsten's explicit approval immediately before commit; push a separate gate.
- Build order: `check_orientation` first (highest evidence + no compute-seam extraction), then `diagnose_provider`, then `validate_construct_validity`.
- Independent-session review of this spec **and** the Phase 2 plan before coding. Nothing deferred/cut without explicit approval.

## 9. Risks & mitigations

- **ADR-052 D14 anti-pattern (silent tripwire):** a tool that loads-inside-compute + catches would return a healthy-looking verdict on a tokenless/failed load → separate fail-loud load seam (§4a) + the load-seam fail-loud test (§7).
- Seam refactor changes CLI behavior → CLI-output regression test per refactored script.
- Server drifts from lib → thin-wrapper equality test.
- Tool trips clean-tree gate / writes stray artifact mid-session → pure read-only compute-seam invariant (§3) + dirty-tree assertion test.
- `mcp` optional extra not installed → guarded server import; core lib + non-MCP consumers unaffected.
- `validate_construct_validity` scope creep toward full-corpus runs → bounded `corpus_ref` (D1) + heavy wrappers stay deferred.

## 10. Open decisions (for review / Karsten sign-off)

- **D1 — `validate_construct_validity` corpus scope:** **DECIDED (Karsten, 2026-09-30): bounded/held-out `corpus_ref` only** (keeps it a tripwire; stays clear of the deferred heavy corpus wrappers).
- **D2 — `validate_*` coverage:** **DECIDED (Karsten, 2026-09-30): ship ALL THREE families** (`validate_gk_decision`, `validate_territorial_defense`, `validate_xtgk_possession_value`) behind the same tool signature.
- **D3 — exact seam names/paths** (compute seams AND the `pining_source` load entry) confirmed at execution time against live HEAD (the plan's first step per tool).

## 11. Deferred (unchanged from parent spec §9, minus what Phase 2 now includes)

- Heavy corpus MCP wrappers: `run_calibration`, `load_corpus`, `compute_metric_on_corpus`.
- `.cursor/rules` and other non-Claude activation shims.
- A conventions-reviewer subagent.
- CI link-check that howto references resolve.

## 12. Review log

- **Rev 2, 2026-09-30** — independent review (`D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase2-mcp-spec.md`, verdict REQUEST CHANGES). Applied: **P2-SPEC-01** (§4a/§5/§7/§9 — separate read-only load seam that FAILS LOUD on tokenless/failed load; compute seams stay pure per ADR-052 D14; new load-seam fail-loud test); **P2-SPEC-02** (§4b — `diagnose_provider` keeper/convention bind existing lib diagnostics `GkClampDiagnosis`/`validate_gk_position_clamp` + `detect_input_convention`, only ids/coords extract from `diag_*.py`); **P2-SPEC-03** (§4/§7 — script/calibration seams tested under `tests/scripts/`, lib seams in lib test dirs).
- **Round 2, 2026-09-30** — independent re-review: **APPROVE**, all three findings resolved, no new. Post-r2 hardening (reviewer heads-up, not a finding): §4a/§7 fail-loud now explicitly covers **empty/degenerate load resolution** (tokenless pining can return empty refs + exit 0 for some providers), not only thrown exceptions. Report: `D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase2-mcp-spec-r2.md`.
- **D1/D2 sign-off (Karsten, 2026-09-30):** D1 = bounded/held-out `corpus_ref`; D2 = ship all three `validate_*` families. §10 updated from open → decided.
