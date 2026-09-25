# AGENTS.md restructure — cut the always-loaded instruction surface

**Date:** 2026-09-24
**Status:** brainstorming-approved; spec review r1 REQUEST CHANGES → revised → r2 **APPROVE**
(SPEC-01..08 resolved). Implementation plan `docs/superpowers/plans/2026-09-24-agents-md-restructure.md`
r2 APPROVE. Implemented on `docs/agents-md-restructure`; pending independent `/review-impl`.
**Branch:** `docs/agents-md-restructure` (off `main` @ 77286f4 / 4.126.0)
**Release:** none — no package change, no version bump, no PyPI tag (see §7)

## 1. Goal

The repo's always-loaded instruction file is **332 lines / ~230,840 bytes (~225 KB)**. It is
loaded into the context of every session, and it has grown by accreting per-module narrative,
measured numbers, and historical defect stories directly into the rules. This costs reasoning
capacity in every session and gets worse each release.

Restructure it so the always-loaded surface holds only **class-1** content — the current,
terse, enforceable invariants a change must not break, each with a pointer to its detail — and
move **class-2** content — the WHY, the history, the measurements, the mechanism walk-throughs
— into an on-demand `docs/context/` tree that is read only when relevant. Rename the canonical
file to `AGENTS.md` (the cross-tool convention) and keep `CLAUDE.md` as a one-line importer so
Claude Code still loads it. Ship a CI gate so the always-loaded surface **cannot re-bloat**.

Target: **~80 % cut** of the always-loaded surface (~230 KB → ~40–45 KB), class-1 = rule +
pointer only.

## 2. Context and precondition

- **`@import` verified empirically** in the running Claude Code version (**2.1.280**). A
  fresh nested `claude -p` session reading a test-dir `CLAUDE.md` containing `@sub/imported.md`
  returned the imported file's canary token. So `CLAUDE.md` = `@AGENTS.md` resolves — the
  rename is viable. (`recency ≠ compatibility` was the owner's flagged concern; cleared by
  test, not assumption.)
- **Chosen store: hybrid B+C.** Reuse the existing `docs/superpowers/adrs/` as the deep-WHY
  store where a rule maps cleanly to one ADR (no re-narration, single source), and add a
  per-domain `docs/context/*.md` tree for the current-state narrative not faithfully captured
  in a single ADR (the Architecture subsystem map detail, and rules that span several ADRs or
  carry later amendments).
- **Multi-domain ADR routing rule (SPEC-05 fix).** An ADR that spans two context domains
  (measured: **ADR-043** touches both `id-compat.md` and `gkdv.md`; **ADR-090** touches both
  `tracking-metrics.md` and `event-metrics.md`) is NOT single-sourced to one file. Rule: split
  its WHY by DOMAIN — the id-compat facet of ADR-043 lives in `id-compat.md`, the gkdv facet in
  `gkdv.md` — and each class-1 rule points to the file for ITS domain. A short one-line
  cross-link (`see also docs/context/<other>.md`) connects the halves; the ADR itself remains
  the shared canonical record both cite. "Single source" in the hybrid means "no re-narration
  of the ADR", not "one file per ADR".
- **`@import` is NOT the mechanism for class-2.** Class-2 is on-demand precisely because it is
  NOT auto-loaded; it is reached by `Read` when relevant, via pointers. `@import` is used only
  for the `CLAUDE.md` → `AGENTS.md` redirect (which IS meant to auto-load).
- **Scope:** this cycle restructures the **repo** file only. The shared global
  `~/.claude/CLAUDE.md` is an **approved phase-2 follow-up** (owner-approved 2026-09-24),
  applying the proven partition + a global anti-bloat rule after the repo pilots the pattern.
  Not deferred-unapproved — recorded and owner-sanctioned.

## 3. Design

### 3.1 `AGENTS.md` target shape (class-1, always-load, budgeted)

All sections terse:

- Title + hexagonal invariant (1–2 lines).
- **Architecture MAP** — each subsystem is one line: `**name** (path/): purpose; entry points;
  → docs/context/<d>.md`, plus its hard invariants as ≤ 1-line each with ADR refs. The TF
  table is kept verbatim (it is navigation, not narrative).
- **Where history lives** table — kept; extended with a `docs/context/` row.
- **Key conventions** — each bullet = a terse imperative RULE + `(ADR-XXX; docs/context/<d>.md)`.
  Zero embedded history / measurement / war-story.
- **Testing** — the CI-shape invariant rules terse + `→ docs/context/ci.md` for the war-stories.
- **Open Items** (pointer to `TODO.md`), **Dependencies** (the list + extras; the
  ruthless-floor / xgboost-bound rationale → `docs/context/`).
- A short **self-documenting meta-rule** (see §3.5) at the top.

Worked example — the ADR-019 id-compat bullet, **~1000 words → ~4 lines**:

> **Dtype-safe id comparisons repo-wide — never raw `==`/`!=` on ids, never merge id keys
> unaligned, never `astype(str)` an id used as a dict-key/join-token.** Use
> `silly_kicks.id_compat` (`ids_equal`/`ids_differ`/`ids_match`/`same_id`/`align_join_keys`/
> `restore_id_dtype`/`canonical_id[_series]`). Gated by `PUBLIC_ID_SCALAR_ENTRIES` + the add_*
> dtype-invariance gate. (ADR-019/043; docs/context/id-compat.md)

The WHY (the `str()`-on-float-id traps, the deleted AST lint, the ADR-027 case) → `id-compat.md`.

**Budget — two distinct numbers (SPEC-CONSIDER fix):**
- **TARGET** (achieves §1's ~80 % cut): landed `AGENTS.md` ≤ **~45 KB**. Asserted directly by
  the test AND checked by the impl reviewer, so the ceiling can never be set so loosely that
  the cut is not actually achieved.
- **CEILING** (anti-regression headroom): landed + ~15 % (landed ~40 KB → ceiling ~**46 KB**;
  `40 × 1.15 = 46`). A new invariant is ~1 line; hitting the ceiling forces moving narrative to
  `docs/context/`.

Aggressiveness: class-1 = rule + pointer only.

### 3.2 `docs/context/` file split (class-2, on-demand)

Directory: `docs/context/`. Seventeen per-domain files; each `AGENTS.md` line points to exactly
one:

| file | holds (source subsystems / ADRs) |
|---|---|
| `providers.md` | converter per-provider contracts (GS/SkillCorner/sportec/metrica), restart-coords ADR-025, Providers parse-ports ADR-031/054 |
| `tracking-features.md` | TF-family narratives, GK identity, metrica contract, preprocess |
| `orientation.md` | reflection / goal-map / direction / action-LTR cluster — ADR-028/029/031/035/041/045/051/055 |
| `id-compat.md` | ADR-019/043 |
| `xt.md` | xT SK-xT-1/2/SER/COUNTS + xt-in-VAEP — ADR-021/022/100/102 |
| `xt-gk.md` | xT-GK v1 ADR-024 + v2 ADR-036 |
| `gkdv.md` | GKDV + TF-19 probes — ADR-043/075/082 |
| `rest-defense.md` | restdefense L1/L2/L3 — ADR-080/081/083/087/089 |
| `gk-metrics.md` | shot-stopping ADR-085, gk-decision ADR-092, keeper-identity ADR-078/084/085 |
| `event-metrics.md` | territory ADR-086/099, duels, match-outcome ADR-097, win-probability ADR-101, xsuccess / vaep-adjusted ADR-095, expected-passing ADR-090 |
| `tracking-metrics.md` | territorial-defense ADR-090, positioning ADR-104 |
| `vaep.md` | VAEP + atomic + causal — ADR-015/018 |
| `trained-models.md` | artifact / serialization / chirality / hub-publish — ADR-011/016/040/044/050/076/088 + velocity-keyed variants ADR-067 |
| `velocity-fov.md` | velocity-availability / FOV / GK-clamp — ADR-054/063/077/083 |
| `corpus-drivers.md` | driver seam / provenance / input-contract / order-insensitivity / fixture-gen — ADR-052/056/037/065 |
| `ci.md` | testing internals — sharding ADR-074, pandas-span ADR-057, build-backend, doctests, slow ADR-023, gate-behaviour |
| `conventions-core.md` | cross-cutting leftovers — purity ADR-033, nan-safety ADR-003, block-detection ADR-046, warnings, rescan ADR-068/073, geometry ADR-050, glossary/C4/metric_contracts ADR-048/098, detect-input-convention ADR-059 |

`docs/context/` is **not** byte-gated — it is on-demand and its size never enters auto-context.

### 3.3 Partition criteria

Applied per line of the current file:

- **Class-1 (stays, terse):** the enforceable INVARIANT (an imperative a change must satisfy),
  the current API/contract surface (entry points, public names, gate names), the ADR ref.
  Test: *does violating it break build/correctness, and must a contributor hold it in-context
  to avoid breaking it?*
- **Class-2 (→ `docs/context/`):** the WHY, measured numbers, the historical defect/story,
  mechanism walk-through, "shipped 4.X.0 / broke Y", superseded-but-recorded reasoning.
  Test: *is it explaining / justifying / historicizing rather than commanding?*
- **Edge rule (load-bearing):** a bullet carrying a CURRENT rule + a LATER amendment/correction
  → class-1 states the **current consolidated rule** (post-amendment); the amendment history
  goes class-2. Class-1 never points at a stale ADR **as** the rule.

### 3.4 Migration safety — no invariant dropped

This is the load-bearing correctness property (the standing scope rule: nothing dropped,
deferred, or softened without the human's approval).

1. **Enumerate** every current bullet/subsystem as a numbered inventory (the checklist lives in
   the implementation plan, produced by the writing-plans skill).
2. **Map** each item → (class-1 line kept) + (class-2 target file). No item unmapped.
3. **Verify mechanically — PER-INVARIANT, not per-token (SPEC-01 BLOCKING fix).** A token-⊆
   check is insufficient: `ADR-\d+` tokens recur (e.g. ADR-027 appears ×5), so a dropped
   imperative would pass the ⊆ check while its token survives on an unrelated rule elsewhere.
   Instead:
   - **Snapshot the inventory from the OLD file at a pinned commit** — `CLAUDE.md` @ `77286f4`
     (the branch base), NOT the new file (self-referential/circular). This snapshot is a
     committed test fixture (e.g. `tests/fixtures/agents_md_invariant_inventory.json`) built
     ONCE from the old file; it is the frozen source of truth. The old file remains recoverable
     via `git show 77286f4:CLAUDE.md`, so the fixture is auditable against it.
   - **One keyed entry per invariant**, each carrying a set of REQUIRED, DISTINCTIVE tokens that
     uniquely identify that invariant — the public gate/contract/symbol names it names (e.g.
     `PUBLIC_ID_SCALAR_ENTRIES`, `resolve_defended_goals`, `zero_velocity_if_unavailable`), not
     just its ADR ref. The check asserts, per entry, that ALL its required tokens appear in
     `AGENTS.md`. A dropped invariant loses its distinctive symbol and FAILS even when its ADR
     token recurs elsewhere.
   - Entries whose invariant is DELIBERATELY dropped (owner-approved, step 4) carry an explicit
     `dropped_with_owner_approval` marker + reason in the fixture, so the enumeration stays
     complete-by-construction and a silent loss cannot masquerade as an approved drop.
4. **No silent drop/defer** — if a bullet is judged genuinely obsolete, it is SURFACED to the
   owner, never deleted unilaterally; an approved drop is recorded in the inventory fixture
   (step 3) with its reason.

### 3.5 Anti-bloat gate ("does not build up again")

Gate the always-loaded surface only (`AGENTS.md`); `docs/context/` stays ungated. One new test
`tests/test_agents_md_budget.py` (repo pattern: `test_todo_md_format`/`test_notice_md_format`;
cheap + version-invariant → runs every CI leg, not `slow`-marked):

1. **TARGET + CEILING byte checks** — `AGENTS.md` ≤ TARGET (~45 KB, the §3.1 80 %-cut bar) and
   ≤ CEILING (landed + ~15 %, anti-regression). The target assertion is what makes the cut real;
   the ceiling is the re-bloat brake.
2. **Per-bullet structural lint — CHAR cap, not line count (SPEC-02 fix).** A "≤ N lines" cap is
   VACUOUS here: the current bullets are single physical lines (119 of 332 exceed 200 chars, one
   is 8254 chars), so a newline-count cap never fires on the real bloat vector. The lint caps
   each Key-conventions bullet at **≤ 600 characters** (≈ 4 wrapped lines) AND requires it to
   carry an `ADR-\d+` ref or a `docs/context/` pointer.
3. **`CLAUDE.md` redirect assertion (COULD-NOT-VERIFY fix — Hyrum on an external tool feature).**
   The test asserts `CLAUDE.md` contains exactly the `@AGENTS.md` import line (plus the comment),
   so a future edit that silently empties or repoints the redirect FAILS CI. This guards the one
   `@import` dependency the design rests on, independent of any CC-version behaviour.
4. **Self-documenting meta-rule** in `AGENTS.md`: "holds current terse rules + pointers ONLY;
   rationale/history/measurement → `docs/context/<domain>.md`; a bullet is ≤ 600 chars + carries
   an ADR/docs pointer; enforced by `tests/test_agents_md_budget.py`."

### 3.6 Rename + reference sweep

- `git mv CLAUDE.md AGENTS.md`; new `CLAUDE.md` = `@AGENTS.md` plus a human-readable comment
  line: *"Canonical instructions: AGENTS.md — this file imports it for Claude Code."*
- **Reference sweep — enumerate every reference to the renamed file** (the discipline from the
  spec `docs/superpowers/specs/2026-08-03-adr051-closeout-and-artifact-validity-design.md` §7.1;
  NOT "ADR-051 §7.1" — SPEC-04 fix, ADR-051 is the orientation defect class and has no §7).
  `grep -rn "CLAUDE.md"` across the tree. **Measured scope (SPEC-03 fix):** ~214 tracked files
  reference the string, ALL in prose/comments — **ZERO functional file-opens**, so the rename
  breaks no code path (a positive). Because `CLAUDE.md` SURVIVES as the `@AGENTS.md` shim, every
  such reference still resolves, so most repoints are cosmetic. Three buckets:
  - **LIVE canonical-content refs** → repoint to `AGENTS.md` (the small set that means "the
    current instruction surface": active top-level docs, the new `docs/context/` pointers,
    any test that reads the content by name).
  - **HISTORICAL refs** (past `docs/superpowers/specs/`, `plans/`, `adrs/`, `CHANGELOG.md`,
    research reports) → **LEAVE AS-IS.** They correctly reference `CLAUDE.md` as it was at their
    time; rewriting them falsifies history.
  - **CC-mechanism refs** → keep (`CLAUDE.md` is still the loader entry).

  The full per-file classification is a plan deliverable.

## 4. Review, branch, commit

- **Branch:** one feature branch `docs/agents-md-restructure` off `main`.
- **Review (load-bearing → INDEPENDENT session):** `/review-spec` on this design doc, then
  `/review-impl` on the branch. The author delivers, STOPS, and hands off a frozen tree; the
  author does not run their own review.
- **Commit:** ONE coherent commit (rename + partition + `docs/context/` + budget test + ref
  updates + TODO summary), full suite green before proposing; committed only on explicit
  per-commit approval.
- **TODO.md:** replace the "Last updated" top summary with this cycle's summary (the standing
  TODO-maintenance discipline; applies even though there is no version bump).

## 5. Testing

- New `tests/test_agents_md_budget.py`: TARGET + CEILING byte checks (§3.1); per-bullet CHAR cap
  (≤ 600) + ADR/docs-pointer requirement (§3.5.2); `CLAUDE.md`-holds-`@AGENTS.md` redirect
  assertion (§3.5.3); and the PER-INVARIANT completeness check against the pinned inventory
  fixture snapshotted from `CLAUDE.md` @ `77286f4` (§3.4.3).
- The existing `tests/test_notice_md_format.py` / `test_todo_md_format.py` / any test that
  reads `CLAUDE.md` by name must be updated by the reference sweep (LIVE bucket) and stay green.
- Full suite green (`-m "not e2e"`) before the commit is proposed.

## 6. Non-goals

- Global `~/.claude/CLAUDE.md` restructure (approved **phase-2** follow-up, not this cycle).
- Any package/BEHAVIOUR change — `silly_kicks/` code is behaviourally untouched. (Task 6's LIVE-repoint edits two COMMENT-only doc-pointer references in `silly_kicks/` — `territorial_defense/_compute.py`, `tracking/_kernels.py` — zero logic change; that is not a behaviour change.)
- Byte-gating `docs/context/` (on-demand; size does not hit auto-context).
- A version bump, a CHANGELOG version entry, or a PyPI release.

## 7. Risks

- **A tool/skill that literally `Read`s `CLAUDE.md`** now gets the `@AGENTS.md` redirect bytes,
  not the rules (CC expands `@import` only in the context-loader, not on a `Read`). Mitigated by
  the comment line pointing a reader to `AGENTS.md`; accepted.
- **Dropping an invariant during the move.** Mitigated by §3.4 (enumerate/map/verify + the
  mechanical completeness check + no-silent-drop).
- **`@import` edge cases** (nested imports, path resolution). Only one level is used
  (`CLAUDE.md` → `AGENTS.md`, same dir); verified working.

## 8. Acceptance criteria

- `AGENTS.md` exists, is class-1 only, ≤ TARGET (~45 KB) AND ≤ CEILING; `CLAUDE.md` =
  `@AGENTS.md` + comment (redirect assertion passes).
- `docs/context/` holds the 17 files; every `AGENTS.md` pointer resolves to a real file.
- The PER-INVARIANT completeness check passes: every keyed invariant in the pinned inventory
  fixture (snapshotted from `CLAUDE.md` @ `77286f4`) has all its distinctive tokens present in
  `AGENTS.md`, EXCEPT entries marked `dropped_with_owner_approval`.
- Every Key-conventions bullet ≤ 600 chars and carries an ADR/docs pointer (per-bullet lint).
- `tests/test_agents_md_budget.py` passes (target + ceiling + char-cap lint + redirect +
  per-invariant completeness).
- Reference sweep complete: LIVE canonical-content refs repointed to `AGENTS.md`; HISTORICAL
  refs left as-is; CC-mechanism refs kept; suite green.
- `TODO.md` top summary replaced.
- No version bump, no CHANGELOG version entry.
