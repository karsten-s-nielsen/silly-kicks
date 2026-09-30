# silly-kicks Agent-Support — Phase 1 Implementation Plan (docs + shims)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans — inline, batching all tasks to a SINGLE approval-gated commit (see Global Constraints). Do NOT use per-task-commit cadence. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give silly-kicks three canonical, auto-triggered procedural sources (metric authoring, provider/id-compat, calibration/validity) plus thin tool-agnostic activation — with no runtime behavior change.

**Architecture:** Neutral core (`docs/howto/*` markdown) holds the substance; `.claude/` shims are thin pointers that own nothing (`AGENTS.md` pointer covers the non-Claude baseline). Slots into the existing class-1 (`AGENTS.md`) / class-2 (`docs/context`) doc tiers — no third tier. Phase 1 is docs + shims only; the MCP server is Phase 2 (separate plan).

**Tech Stack:** Markdown; Claude Code skills/commands (`.claude/skills`, `.claude/commands`); existing `pytest` gate `tests/test_agents_md_budget.py`.

**Spec:** `D:\Development\_handoffs\silly-kicks-agent-support\2026-09-29-silly-kicks-agent-support-neutral-layout-design.md` (APPROVED, independent review round 2). Executors read both.

## Global Constraints

- **Target repo `silly-kicks` is edited only with per-repo approval, on a feature branch off the default branch (e.g. `karsten/agent-support-phase1`) — NEVER a worktree or parallel checkout.**
- **One fully-tested commit for all of Phase 1, gated on Karsten's explicit approval immediately before the commit. No micro-commits, no per-task commits (this overrides the writing-plans default). Never commit or push without that explicit approval.** Tasks below end at *verify*.
- **Nothing load-bearing in `.claude/`** — substance lives in `docs/howto/`; shims only point.
- **`AGENTS.md` additions must keep `tests/test_agents_md_budget.py` green** (byte ceiling + ≤600 chars/bullet + completeness).
- **Docs cite ADRs/source files, they do not copy their text.**
- **No behavior change** to any metric or script in Phase 1.
- Slot into the class-1/class-2 doc arch (per `docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md`); no third tier.

## File Structure

Create:
- `docs/howto/authoring-a-metric.md` — the metric-authoring checklist.
- `docs/howto/construct-validity.md` — the validity-gate reference.
- `docs/howto/corpus-drivers-runbook.md` — generic corpus-driver runbook.
- `.claude/skills/authoring-a-metric/SKILL.md`, `.claude/skills/construct-validity/SKILL.md`, `.claude/skills/corpus-drivers/SKILL.md` — thin trigger shims.
- `.claude/commands/new-metric.md` — thin command → authoring skill.

Modify:
- `AGENTS.md` — add a "How-to runbooks" section with 3 one-line pointers.

Note on doc tasks: the deliverable prose is authored **at execution time by reading the cited source files** (ADRs, `conventions-core.md`, `_driver.py` docstring) — the plan fixes the exact structure, required facts, sources to cite, and acceptance criteria; it does not hardcode prose second-hand from the spec's summary (that would risk drift from the source of truth). This is intentional, not a placeholder.

---

### Task 1: `docs/howto/authoring-a-metric.md`

**Files:**
- Create: `docs/howto/authoring-a-metric.md`
- Read to author + cite: `AGENTS.md` §Key conventions; `docs/context/conventions-core.md`; ADR-098 (metric_contracts), ADR-033 (add_* purity); enforcing tests `tests/test_metric_contracts.py`, `tests/test_add_star_purity.py`, `tests/tracking/test_aggregator_column_liveness.py`, the call-convention registry test.

**Produces:** a canonical checklist referenced by the `authoring-a-metric` skill (Task 4) and the `AGENTS.md` pointer (Task 5).

- [ ] **Step 1:** Read the sources above; confirm each referenced ADR/test exists at the current HEAD.
- [ ] **Step 2:** Author the doc with exactly these sections, each citing its enforcing ADR/test:
  1. `add_*` (action-coupled, enriches) vs `compute_*` (standalone aggregator) — and the aggregator-count bookkeeping.
  2. Package layout (`<family>/__init__ + _config + _compute/_engine + _report/_probe`).
  3. Tests mirror the package + a `tests/scripts/test_*_validity` corpus-driver test; fixture placement.
  4. Register in `metric_contracts` (ADR-098): `METRIC_CONTRACTS` / `*_METRIC_COLUMNS` / grain `*_KEYS`.
  5. `add_*` → `PURITY_ENTRIES` (ADR-033; ≥2 variants for conditional-column adders).
  6. tracking `add_*` → liveness gate (non-NaN AND non-constant) + call-convention registry.
  7. `*_xfns` factory decision (leak-risk absence guard).
  8. `feature_glossary` count bump; `NOTICE` entry; C4 update.
  9. value-changing vs additive → version + Hyrum notice + lakehouse re-materialize.
- [ ] **Step 3:** Quote verbatim the two `conventions-core.md` lessons: "four silent-null defects share one shape → non-vacuity assertions" and "a liveness gate's fixture needs its own precondition test."
- [ ] **Step 4 (verify):** Every one of the 9 steps is present and cites a real ADR/test file; the two lessons are quoted; all intra-repo links resolve (`grep` each linked path exists). No commit.

**Acceptance:** 9/9 steps present + cited; 2 lessons quoted; links resolve.

---

### Task 2: `docs/howto/construct-validity.md`

**Files:**
- Create: `docs/howto/construct-validity.md`
- Read to author + cite: `docs/research/{gk_decision_construct_validity, territorial_defense_construct_validity, xtgk_v2_construct_validity}/`, `docs/research/pass_risk_calibration/README.md`; the `validate_*.py` family; a real `docs/research/*/metrics.json` for the verdict shape.

**Produces:** canonical validity reference for the `construct-validity` skill (Task 4).

- [ ] **Step 1:** Read the sources; open one real `metrics.json` to copy its exact GO/NO-GO field shape.
- [ ] **Step 2:** Author with these sections: the three gates (predictive / discriminating / responsiveness) defined once; the GO/NO-GO verdict shape (matching the real `metrics.json`); the "state what the number does NOT measure" caveat pattern (cite `pass_risk_calibration`); how to add a `validate_*.py`; links to the 3 exemplar memos.
- [ ] **Step 3 (verify):** 3 gates defined; verdict shape matches an actual `metrics.json`; links resolve. No commit.

**Acceptance:** 3 gates + verdict-shape match + resolving links.

---

### Task 3: `docs/howto/corpus-drivers-runbook.md`

**Files:**
- Create: `docs/howto/corpus-drivers-runbook.md`
- Read to author + cite: `scripts/_driver.py` (docstring), `scripts/README_calibration.md` (as the format template), `docs/context/corpus-drivers.md`.

**Produces:** the generic driver runbook referenced by the `corpus-drivers` skill (Task 4).

- [ ] **Step 1:** Read the `_driver.py` docstring and `README_calibration.md`.
- [ ] **Step 2:** Author with: the `for_each(items, key=, work=, shard_root=, token_inputs=)` seam (resume-before-load, `.excluded.json`, `assert_conservation`/`_require_injective`, `require_clean_tree`); how to build a new `build_*`/`validate_*`/`measure_*` on it (do NOT re-solve resume/cache); memo-landing convention (`docs/research/<topic>/README.md` + `metrics.json`, `run_commit`, `run_tree_dirty:false`); env (`PINING_FOR_THE_DATA_TOKEN`, sources `pining`/`databricks`). Link to `_driver.py`'s docstring rather than duplicating it.
- [ ] **Step 3 (verify):** covers resume/`.excluded.json`/conservation/clean-tree + memo convention + env; links resolve. No commit.

**Acceptance:** all four topics covered; points at `_driver.py`; links resolve.

---

### Task 4: `.claude` shims (3 skills + 1 command)

**Files:**
- Create: `.claude/skills/authoring-a-metric/SKILL.md`, `.claude/skills/construct-validity/SKILL.md`, `.claude/skills/corpus-drivers/SKILL.md`, `.claude/commands/new-metric.md`

**Consumes:** the three docs from Tasks 1–3 (referenced by relative path).

- [ ] **Step 1:** For each skill write frontmatter `name` + a trigger `description` naming the task phrases (e.g. authoring-a-metric: "adding/creating a new metric, pure-fn add_*/compute_*, metric_contracts, PURITY_ENTRIES, liveness gate"), and a body of a few lines: "Read `docs/howto/<X>.md` and follow it." Nothing load-bearing beyond the pointer.
- [ ] **Step 2:** Write `.claude/commands/new-metric.md` (frontmatter `description`; body invokes/points at the authoring skill + doc).
- [ ] **Step 3 (verify):** each `SKILL.md` has valid frontmatter (`name`, `description`); the `description` contains the intended trigger phrases (manual spot-check — firing isn't unit-testable); body only points at its doc; the referenced `docs/howto/*.md` paths exist. No commit.

**Acceptance:** 3 skills + 1 command, valid frontmatter, doc paths resolve, no substance duplicated.

---

### Task 5: `AGENTS.md` pointers + budget gate

**Files:**
- Modify: `AGENTS.md` (add a "How-to runbooks" section with 3 one-line pointers to the Task 1–3 docs)
- Gate: `tests/test_agents_md_budget.py`

**Consumes:** the three doc paths (Tasks 1–3).

- [ ] **Step 1:** Add a terse "How-to runbooks" section: 3 bullets, each ≤600 chars, one linking each howto with a 6–10 word purpose.
- [ ] **Step 2 (verify):** Run `pytest tests/test_agents_md_budget.py -v` → PASS (byte ceiling + per-bullet limit + completeness). If it reds, tighten the bullets until green.
- [ ] **Step 3 (verify):** the 3 linked paths resolve. No commit.

**Acceptance:** budget test green; 3 pointers resolve.

---

### Final: single Phase-1 commit (approval-gated)

- [ ] **Step 1:** On the feature branch, run the relevant gate(s): `pytest tests/test_agents_md_budget.py -v` (green) + a link-resolution check over the new `docs/howto/*` and shims.
- [ ] **Step 2:** Show Karsten the full file list + diff for all of Phase 1. **STOP.**
- [ ] **Step 3:** Only on Karsten's explicit "yes to this commit", create ONE commit (all Phase-1 files) on the feature branch. Do not push unless separately approved.

**Acceptance:** all gates green; single coherent commit made only after explicit approval.

---

## Self-Review

**Spec coverage (Phase 1 items):** 3 howto docs → Tasks 1–3 ✓; 3 skill shims + `/new-metric` → Task 4 ✓; `AGENTS.md` pointers + budget gate → Task 5 ✓; single-commit/branch governance → Global Constraints + Final ✓. Phase 2 (MCP server + 3 tools + `tests/mcp/`) is intentionally OUT of this plan — separate plan on trigger (spec §5). No spec Phase-1 requirement is unaddressed.

**Placeholder scan:** doc tasks specify concrete sections + sources + acceptance (not "write appropriate docs"); prose is authored from cited sources at execution time by design (noted above). No "TBD"/"handle edge cases"/vague steps.

**Type/name consistency:** doc filenames, skill names, and `AGENTS.md` pointer targets all reference the same three paths `docs/howto/{authoring-a-metric,construct-validity,corpus-drivers}.md` across Tasks 1–5. Consistent.

## Review log

- **2026-09-30** — independent review: **APPROVE** (2 CONSIDER, no blocking). Applied AS-PLAN-01 (header → `executing-plans`, single approval-gated commit; dropped the per-task subagent cadence that conflicted with the one-commit rule) and AS-PLAN-02 (Task 4 Step 3 trigger-phrase spot-check). Report: `D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase1-plan.md`.
- **2026-09-30, round 2** — independent re-review: **APPROVE** — both r1 CONSIDERs applied, no new findings, nothing regressed vs `3ca609f`. Report: `D:\Development\_reviews\2026-09-30-silly-kicks-agent-support-phase1-plan-r2.md`.
