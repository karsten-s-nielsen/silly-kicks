# silly-kicks — Tool-Agnostic Agent-Support Layout (Design Spec)

**Date:** 2026-09-29
**Target repo:** `silly-kicks` (`D:\Development\karstenskyt__silly-kicks`)
**Status:** APPROVED 2026-09-30 (independent review, round 2 — verdict APPROVE). Rev 2 applied findings AS-SPEC-01..04. Not yet implemented. No files written to the target repo by the authoring session (it holds `silly-kicks` READ-ONLY).
**Author:** coursework/DL session (spawned research grounded in the target repo's own history).

---

## 1. Problem

`silly-kicks` sessions "constantly rediscover how to do certain things." A read-only audit of the repo's own history (`docs/context`, `docs/research`, `scripts`, `calibration_runs`, `git log`) shows the pain is **not** a documentation gap — the repo already has a mature 2-tier doc architecture (class-1 `AGENTS.md` terse rules / class-2 `docs/context` topic guides, per `docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md`). The pain has two distinct shapes:

1. **Missing procedural docs.** Some recurring procedures have no single canonical home, so each session reassembles them:
   - No "how to add a metric" checklist — the contract is scattered across `AGENTS.md` §Key conventions and `docs/context/conventions-core.md`.
   - Construct-validity is re-explained across ~15 mentions (a mention-count, not a dir-count): there are **7** `docs/research/*` dirs literally named `*validity` (incl. `gk_decision_construct_validity`, `territorial_defense_construct_validity`, `xtgk_v2_construct_validity`).
   - No generic corpus-driver runbook — `scripts/README_calibration.md` is a complete runbook for TF-24 only; the other **~35** `build_*`/`validate_*`/`measure_*` scripts have none.

2. **Runtime tripwires a document cannot enforce.** The highest-evidence defects recur because catching them requires *executing a check*, not reading a rule:
   - **Orientation defect class fixed 4 times** (ADR-028 RC1–RC5 across `2b162f0`, `89dd9af`, `94e05d1`, `4b15365`), plus **1 adjacent fixture-coherence fix** (`9e5c8d3` = mirror-registry reference-scene coherence, not a production orientation defect); the pining loader shipped unoriented SkillCorner frames for a whole corpus while "looking healthy" because `team_attacking_direction` was NULL and the flip silently no-opped (`docs/research/adr028_rc4_orientation/README.md`).
   - **`str()`-on-a-float-id trap recurred 4 times** (latest in `_elastic_sync.py`), degrading a keyed lookup to a constant while an A/B audit read "identical → works" because both legs degraded the same way (`docs/context/id-compat.md`).
   - `calibration_runs/diag_*.py` re-derive the same low-level provider probes ad hoc, never promoted to a reusable diagnostic.

## 2. Goals / Non-Goals

**Goals**
- Give each of three recurring areas a single canonical, auto-triggered source: **metric authoring**, **provider/data ingest + id-compat (incl. orientation)**, **calibration & validation**.
- Make the orientation and id-dtype tripwires **runnable**, not just documented.
- Keep the whole setup **tool-agnostic**: nothing load-bearing lives in a vendor-specific folder.

**Non-Goals**
- No rewrite of existing `AGENTS.md` / `docs/context` content (it is solid; only surface it more discoverably).
- No third documentation tier.
- No heavy corpus-scale MCP wrappers in the first cut (see Deferred).
- No behavior change to any existing metric or script (except minimal read-only import seams, §5).

## 3. Design principle

**Separate CONTENT/CAPABILITY (neutral) from ACTIVATION (per-tool veneer).**
- Content → `docs/` (markdown). Capability/compute → `scripts/` (CLI, exists) + a small MCP server (open protocol).
- Activation → thin per-tool shims that point at the neutral core and own nothing: `.claude/` for Claude Code; `AGENTS.md` pointer already covers the Codex/Cursor/Aider baseline; `.cursor/rules` etc. added later only if needed.
- Rule: **nothing load-bearing in `.claude/`.**

## 4. Phase 1 — neutral docs + thin shims (one PR, no runtime change)

New `docs/howto/` subdir (procedural runbooks; same class-2 tier as `docs/context`, kept distinct from topic references):

- **`docs/howto/authoring-a-metric.md`** — one checklist consolidating the scattered contract (cite ADRs, do not copy their text):
  1. `add_*` (action-coupled, enriches) vs `compute_*` (standalone aggregator).
  2. Package layout (`<family>/__init__ + _config + _compute/_engine + _report/_probe`).
  3. Tests mirror package + a `tests/scripts/test_*_validity` corpus-driver test; fixture placement.
  4. Register in `metric_contracts` (ADR-098): `METRIC_CONTRACTS`/`*_METRIC_COLUMNS`/grain `*_KEYS`.
  5. `add_*` → `PURITY_ENTRIES` (ADR-033; ≥2 variants for conditional-column adders).
  6. tracking `add_*` → liveness gate (non-NaN AND non-constant) + call-convention registry.
  7. `*_xfns` factory decision (leak-risk absence guard).
  8. `feature_glossary` count bump; `NOTICE` entry; C4 update.
  9. value-changing vs additive → version + Hyrum notice + lakehouse re-materialize.
  - Anchors two hard-won lessons verbatim from `conventions-core.md`: "four silent-null defects share one shape → non-vacuity assertions"; "a liveness gate's fixture needs its own precondition test."

- **`docs/howto/construct-validity.md`** — centralize the gate re-explained ~15×: the three gates (predictive / discriminating / responsiveness) defined once; the GO/NO-GO verdict shape matching `docs/research/*/metrics.json`; the "state what the number does NOT measure" caveat pattern (from `pass_risk_calibration`); how to add a `validate_*.py`; links to exemplar memos.

- **`docs/howto/corpus-drivers-runbook.md`** — the missing generic sibling to `README_calibration.md`: the `scripts/_driver.py` `for_each` seam (resume-before-load, `.excluded.json`, `assert_conservation`/`_require_injective`, `require_clean_tree`); how to build a new `build_*`/`validate_*`/`measure_*` on it instead of re-solving resume/cache (a problem independently half-solved 4× per `_driver.py`'s own docstring); memo-landing convention (`docs/research/<topic>/README.md` + `metrics.json`, `run_commit`, `run_tree_dirty:false`); env (`PINING_FOR_THE_DATA_TOKEN`). Points at `_driver.py`'s docstring rather than duplicating it.

Shims:
- `.claude/skills/{authoring-a-metric,construct-validity,corpus-drivers}/SKILL.md` — trigger `description` + body ≈ "read `docs/howto/X.md`".
- `AGENTS.md` — 3 one-line pointers under a "How-to runbooks" heading. **Must fit the class-1 byte budget** (`tests/test_agents_md_budget.py`: byte ceiling + ≤600 chars/bullet + completeness) — keep terse or the budget test reds.
- One command `.claude/commands/new-metric.md` → authoring skill. (No other commands.)

**Phase-1 validation:** docs accurate against current ADRs/`AGENTS.md` at author time; shim `description`s fire on the intended tasks; the `AGENTS.md` budget test stays green. No runtime code change.

## 5. Phase 2 — small MCP server (separate PR, gated on Phase 1 use)

**Server:** `silly_kicks/mcp/server.py` — importable package module, stdio transport. `mcp[cli]` added as an **optional extra** `[project.optional-dependencies] mcp = ["mcp[cli]>=1,<2"]` (pin `<2`: mcp 2.x moved FastMCP; the core pure-fn lib stays dependency-light and the server is opt-in). The server owns **no logic** — each tool imports and calls an existing `silly_kicks` function / script entry, returns JSON-serializable dicts mirroring existing `metrics.json` shapes (Hyrum-safe), and routes all ids through `id_compat` (no reintroduced `str()`-on-float trap).

**Each tool binds the READ-ONLY / pure seam** — the function that returns a dict with **no `require_clean_tree` gate, no `run_commit`, and no artifact write** — never the side-effecting `run()`/`main()` of a driver. This is load-bearing: a tripwire is called mid-work on a **dirty** tree (the normal in-session state — exactly when you want it), so binding to a clean-tree-gated `run()` would make the tool unusable and could emit stray `metrics.json`. `check_orientation` already has such a seam: `measure_rc4_orientation.measure() -> dict` (`measure_rc4_orientation.py:57`), NOT `run()` (:95) / `main()` (:194).

**Tools (3, runtime checks a doc cannot enforce):**
1. **`check_orientation(match_ref, provider?)`** — binds `measure_rc4_orientation.measure()`; returns `{direction_resolved, geometry_agrees, verdict}`. Standing tripwire for the orientation defect class (incl. the NULL-direction silent no-op).
2. **`diagnose_provider(provider, match_ref, aspect="ids|coords|keeper|convention")`** — promotes the `calibration_runs/diag_*.py` one-offs (player-to-ball distance, duplicate-frame identity, actor distribution, id-dtype) to one callable probe.
3. **`validate_construct_validity(metric_family, corpus_ref, gate=…)`** — wraps the `validate_*.py` family; returns GO/NO-GO JSON in the existing memo shape.

**Flagged code touch (AS-SPEC-01):** Phase 2 must **ensure a pure read-only seam exists for each of the 3 tools**. `check_orientation` already has `measure()`. `validate_construct_validity` and `diagnose_provider` wrap clean-tree-gated, artifact-writing drivers (`validate_*.py`, the `diag_*` scripts), so Phase 2 includes a minimal extraction/confirmation of a pure `measure()`-style entry from each (return a dict; strip the clean-tree gate + artifact write from the callable path) — no behavior change to the existing CLI entry, just an added seam. This is the one code touch beyond the server; called out for review.

**Registration (neutral server, per-client config):** Claude via `.mcp.json` → `python -m silly_kicks.mcp.server` (repo venv); other clients via a documented manual registration in `docs/howto/mcp.md`. Identical server.

**Testing:** `tests/mcp/` per-tool verdict-shape assertions on existing per-provider fixtures; a thin-wrapper equality test (tool output == the underlying pure seam's result on the same fixture, guarding server drift); a read-only assertion (each tool runs on a deliberately dirty tree without tripping `require_clean_tree` or writing an artifact); a tripwire regression (`check_orientation` must flag the known unoriented-pining RC4 case).

**Build order:** start with `check_orientation` (highest evidence); add the other two on demand.

## 6. Rollout & approval gates

- Phase 1 and Phase 2 are separate PRs, each on its own **feature branch off the default branch** (no worktrees), each **one fully-tested commit** (no micro-commits).
- Gates (owner = Karsten): per-repo approval before any write to `silly-kicks`; explicit approval for each commit; independent-session review of this spec **and** the implementation plan before coding; nothing deferred or cut without explicit approval.

## 7. Risks & mitigations

- Docs drift from ADRs → cite ADRs, don't copy; optional CI link-check (deferred).
- MCP server drifts from lib → thin-wrapper equality test.
- MCP tool trips `require_clean_tree` / writes stray artifacts mid-session → each tool binds a pure read-only seam (§5) + the dirty-tree assertion test.
- Core dependency bloat → `mcp` is an opt-in extra.
- Import-seam scope creep → seam is minimal (return-a-dict path only), called out explicitly for review.

## 8. Success criteria

- Each of the three areas has one canonical, auto-triggered source.
- Orientation + id-dtype tripwires are runnable, not just documented.
- A new contributor or agent reaches the procedure without rediscovery.

## 9. Deferred (parked for explicit decision, NOT dropped)

- Heavy corpus MCP wrappers: `run_calibration`, `load_corpus`, `compute_metric_on_corpus`.
- `.cursor/rules` and other non-Claude activation shims.
- A conventions-reviewer subagent (checks a diff against `AGENTS.md` + `docs/context`).
- CI link-check that howto references resolve.

## 10. Evidence appendix (from the read-only history audit; verified by independent review vs `silly-kicks` @ `3ca609f`)

- Orientation defect class fixed 4× (`2b162f0` RC1 detection, `89dd9af` RC2+3+5 / 4.71.0, `94e05d1` RC4 / 4.73.0, `4b15365` pining-loader follow-up) **+ 1 adjacent fixture fix** (`9e5c8d3` mirror-registry reference-scene coherence — NOT a production orientation defect). RC4 memo: `docs/research/adr028_rc4_orientation/README.md`.
- id `str()`-on-float trap recurred 4× (ADR-019 origin; `validate_shot_goalmouth_sb.py`; pitch-control decomposition; `_elastic_sync.py`).
- Corpus-driver resume/cache independently half-solved 4× before unification (`scripts/_driver.py` docstring).
- Construct-validity: **7** `docs/research/*` dirs named `*validity`; the gate re-explained across ~15 mentions (mention-count, not dir-count).
- ~35 `build_*`/`validate_*`/`measure_*` scripts lack a generic runbook (only TF-24 has `README_calibration.md`).
- Provider geometry facts needing standalone memos: `docs/research/gs_keeper_clamp/findings.md` (GK clamped at 27.5m), `docs/research/gs_input_convention/README.md`, `docs/research/bug_kloppy_tracking_y_inverted.md` (ADR-031).
- Already good (reuse, don't duplicate): `AGENTS.md` §Key conventions; the class-1/class-2 split; `scripts/README_calibration.md`; `scripts/_driver.py` docstring; `conventions-core.md`'s silent-null-defects + liveness-fixture lessons.

## 11. Review log

- **Rev 2, 2026-09-30** — independent review (`D:\Development\_reviews\2026-09-29-silly-kicks-agent-support-neutral-layout-spec.md`, verdict REQUEST CHANGES). Applied: AS-SPEC-01 (§5 — tools bind pure read-only seams, named `measure()`; pure-seam extraction added to the flagged code touch; dirty-tree test); AS-SPEC-02 (§4 — `AGENTS.md` byte-budget gate); AS-SPEC-03 (§1/§10 — orientation 4 defect fixes + 1 adjacent fixture fix; ~20→~35 scripts); AS-SPEC-04 (§1/§10 — 7 `*validity` dirs vs ~15 mentions). Phase 1 was assessed clean.
- **Round 2, 2026-09-30** — independent re-review: **APPROVE** — all four r1 findings resolved and verified vs `3ca609f`, no new findings. Report: `D:\Development\_reviews\2026-09-29-silly-kicks-agent-support-neutral-layout-spec-r2.md`.
