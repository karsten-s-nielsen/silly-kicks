# AGENTS.md restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this
> plan task-by-task (INLINE only — subagents are banned in this repo per owner rule 2026-09-23).
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Partition the 230 KB always-loaded repo `CLAUDE.md` into a terse class-1 `AGENTS.md`
(always-load) plus an on-demand `docs/context/` tree, rename `CLAUDE.md` → `AGENTS.md` behind a
one-line `@AGENTS.md` importer, and ship a CI gate that prevents re-bloat and proves no invariant
was dropped.

**Architecture:** `AGENTS.md` holds only current terse rules + pointers (rule + `ADR-XXX` +
`docs/context/<d>.md`). The WHY/history/measurement moves to 17 per-domain `docs/context/*.md`
files read on demand. A per-invariant inventory fixture, snapshotted ONCE from `CLAUDE.md`
@ `77286f4`, is the migration-safety oracle: the budget test asserts every keyed invariant's
distinctive symbols survive in `AGENTS.md`. `@import` is used ONLY for the `CLAUDE.md` shim.

**Tech Stack:** Markdown; Python + pytest (one new test, repo `test_*_md_format` pattern); git.

**Spec:** `docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md` (r2 APPROVE).

**Status:** plan review r2 APPROVE; implemented inline on `docs/agents-md-restructure` (all gates green — budget 10/10, ruff clean, pyright baseline, suite 9539 passed); pending independent `/review-impl` before the single owner-gated commit.

## Global Constraints

- **Single feature branch** `docs/agents-md-restructure` (already created off `main` @ `77286f4`).
- **ONE commit, at the very end, owner-gated.** No per-task commits, no micro-commits. Every task
  below ends by RUNNING its test, never by committing. The final commit is proposed to the owner
  only after the full suite is green and the independent `/review-impl` has cleared.
- **No version bump, no CHANGELOG version entry, no PyPI tag** (`AGENTS.md`/`CLAUDE.md`/`docs/`
  are not in the wheel; `packages = ["silly_kicks"]`).
- **Class-1 rule format:** terse imperative + `(ADR-XXX; docs/context/<d>.md)`; ≤ 600 chars/bullet.
- **Class-2 rule (§3.3 edge):** a bullet carrying a later amendment → class-1 states the CURRENT
  consolidated rule; history goes to `docs/context/`. Class-1 never cites a stale ADR AS the rule.
- **Nothing dropped/deferred/softened without owner approval** (the standing scope rule); an
  approved drop is recorded in the inventory fixture with its reason.
- **Reference sweep buckets:** LIVE canonical-content ref → repoint to `AGENTS.md`; HISTORICAL ref
  (past specs/plans/adrs/CHANGELOG/research) → LEAVE AS-IS; CC-mechanism ref → keep.
- **Budget numbers:** TARGET ≤ 45 KB (asserted, enforces the 80 % cut); CEILING = landed + 15 %
  (anti-regression, set once landed is measured).
- **Command environment (PLAN-r1 fix 2):** run pytest via the repo's pinned 3.10 venv —
  `./.venv/Scripts/python.exe -m pytest ...` (NEVER bare `python`/`python -m pytest`, which
  resolves to system 3.14). The full-suite run sets `SILLY_KICKS_ASSERT_INVARIANTS=1` (CI sets it,
  `ci.yml`; "green" without it ≠ CI green). Run ruff + pyright via system Python `python -m ruff`
  / `python -m pyright` (per repo venv policy). `git grep` (tracked-only), never `grep -rn .`
  (walks `.venv/`).

---

## File structure

- Create `tests/fixtures/agents_md_invariant_inventory.json` — the migration-safety oracle.
- Create `tests/test_agents_md_budget.py` — the anti-bloat + completeness gate.
- Create `docs/context/*.md` — 17 per-domain class-2 files (spec §3.2 table).
- Rename `CLAUDE.md` → `AGENTS.md` (`git mv`), then edit `AGENTS.md` down to class-1.
- Create new `CLAUDE.md` — the `@AGENTS.md` shim + comment.
- Modify `TODO.md` — top summary.
- Modify the LIVE-bucket files the reference sweep identifies.

---

## Task 1: Invariant inventory fixture (the migration-safety oracle)

**Files:**
- Create: `tests/fixtures/agents_md_invariant_inventory.json`

**Interfaces:**
- Produces: a JSON array. Each entry:
  ```json
  {
    "key": "id-compat",
    "domain": "id-compat.md",
    "adr_refs": ["ADR-019", "ADR-043"],
    "required_tokens": ["id_compat", "PUBLIC_ID_SCALAR_ENTRIES", "canonical_id"],
    "dropped_with_owner_approval": null
  }
  ```
  `required_tokens` = DISTINCTIVE symbols (gate/contract/public names) that uniquely identify the
  invariant — NOT bare `ADR-\d+` tokens (those recur). `dropped_with_owner_approval` is `null`
  unless the owner approved dropping this invariant, in which case it is `{"reason": "..."}`.

- [ ] **Step 1: Dump the pinned old file**

Run: `git show 77286f4:CLAUDE.md > /tmp/claude_md_77286f4.md` (read-only source of truth).

- [ ] **Step 2: Enumerate every class-1 invariant from the old file into the fixture**

Walk `/tmp/claude_md_77286f4.md` top to bottom. For EACH Architecture-subsystem invariant and
EACH Key-conventions / Testing bullet, add one entry: `key` (stable slug), `domain` (its
`docs/context/*.md` target), `adr_refs`, and `required_tokens` = the distinctive public symbols
that bullet names (e.g. `resolve_defended_goals`, `zero_velocity_if_unavailable`,
`PUBLIC_ID_SCALAR_ENTRIES`, `group_rows`, `_serve_positions_core`, `pressure_on_target`, …). Pick
tokens that are unique to that invariant so its loss is detectable. `dropped_with_owner_approval`
= `null` for all (no drops planned; if one is proposed later it is surfaced to the owner first).

- [ ] **Step 3: Validate the fixture against its source inline (no persistent test here)**

The persistent oracle-sanity test lives in the Task 2 module (avoids clobbering it — PLAN-r1
fix 4). Validate the fixture NOW with a throwaway one-liner so a fabricated token is caught before
Task 2:

Run:
```bash
git show 77286f4:CLAUDE.md > /tmp/claude_md_77286f4.md
./.venv/Scripts/python.exe - <<'PY'
import json
old = open("/tmp/claude_md_77286f4.md", encoding="utf-8").read()
inv = json.load(open("tests/fixtures/agents_md_invariant_inventory.json", encoding="utf-8"))
bad = {e["key"]: [t for t in e["required_tokens"] if t not in old] for e in inv}
bad = {k: v for k, v in bad.items() if v}
print("FABRICATED TOKENS:", bad) if bad else print("OK", len(inv), "entries")
PY
```
Expected: `OK <n> entries` (every token present in the pinned old file — oracle is real).

- [ ] **Step 4: Do NOT commit** (single commit at end).

---

## Task 2: The anti-bloat + completeness gate test (write RED)

**Files:**
- Create/extend: `tests/test_agents_md_budget.py`

**Interfaces:**
- Consumes: `tests/fixtures/agents_md_invariant_inventory.json` (Task 1).
- Produces: the CI gate asserting target/ceiling/char-cap/redirect/completeness/pointer-resolve.

- [ ] **Step 1: Write the full test module**

```python
import json, re, subprocess
from pathlib import Path

_PIN = "77286f4"
_INV = Path("tests/fixtures/agents_md_invariant_inventory.json")
_AGENTS = Path("AGENTS.md")
_CLAUDE = Path("CLAUDE.md")
_CONTEXT = Path("docs/context")

_TARGET_BYTES = 45 * 1024          # §3.1 TARGET — enforces the ~80% cut
_CEILING_BYTES = 46 * 1024         # §3.1 CEILING — landed+15%; RESET in Task 8 from measured landed
_MAX_BULLET_CHARS = 600            # §3.5.2
_CONTEXT_MIN_BYTES = 100 * 1024    # PLAN-r1 fix 3 — the moved WHY must land, not vanish

def _agents_text() -> str:
    return _AGENTS.read_text(encoding="utf-8")

def _inventory() -> list[dict]:
    return json.loads(_INV.read_text(encoding="utf-8"))

def _old_claude_md() -> str:
    # encoding="utf-8" is load-bearing on Windows: git show emits UTF-8 (em-dashes, →),
    # subprocess text mode defaults to cp1252 there and raises UnicodeDecodeError.
    return subprocess.run(
        ["git", "show", f"{_PIN}:CLAUDE.md"],
        capture_output=True, encoding="utf-8", errors="replace", check=True,
    ).stdout

def test_inventory_tokens_exist_in_pinned_source():
    # oracle-sanity: the fixture is snapshotted from the OLD file, not fabricated (SPEC-01)
    old = _old_claude_md()
    inv = _inventory()
    assert inv, "inventory is empty"
    bad = {e["key"]: [t for t in e["required_tokens"] if t not in old] for e in inv}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"fixture tokens absent from CLAUDE.md@{_PIN} (fabricated oracle): {bad}"

def test_agents_md_exists():
    assert _AGENTS.is_file(), "AGENTS.md missing"

def test_target_size():
    n = _AGENTS.stat().st_size
    assert n <= _TARGET_BYTES, f"AGENTS.md {n} B > TARGET {_TARGET_BYTES} B (80% cut not achieved)"

def test_ceiling_size():
    n = _AGENTS.stat().st_size
    assert n <= _CEILING_BYTES, f"AGENTS.md {n} B > CEILING {_CEILING_BYTES} B (re-bloat)"

def test_claude_md_is_the_import_shim():
    # spec §3.6 "exactly" (PLAN-r1 fix 6): the only non-comment, non-blank line is @AGENTS.md
    lines = _CLAUDE.read_text(encoding="utf-8").splitlines()
    content = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("<!--")]
    assert content == ["@AGENTS.md"], f"CLAUDE.md must be exactly the @AGENTS.md import (+comments); got {content}"
    assert _CLAUDE.stat().st_size < 512, "CLAUDE.md shim should be tiny (import + comment only)"

def test_context_bulk_conservation():
    # PLAN-r1 fix 3: the narrative removed from CLAUDE.md must reappear in docs/context/,
    # not be silently dropped. A stub tree of tiny files fails this floor.
    total = sum(p.stat().st_size for p in _CONTEXT.glob("*.md"))
    assert total >= _CONTEXT_MIN_BYTES, (
        f"docs/context/ total {total} B < floor {_CONTEXT_MIN_BYTES} B — moved WHY appears dropped"
    )

def test_context_domain_tokens_landed():
    # PLAN-r1 fix 3 (per-block guard): each non-dropped invariant's WHY landed in ITS domain file.
    # A dropped block loses ALL of its distinctive tokens AND its ADR refs from the file → fails.
    inv = _inventory()
    orphaned = {}
    for e in inv:
        if e.get("dropped_with_owner_approval"):
            continue
        dom = _CONTEXT / e["domain"]
        body = dom.read_text(encoding="utf-8") if dom.is_file() else ""
        anchors = list(e["required_tokens"]) + list(e.get("adr_refs", []))
        if not any(a in body for a in anchors):
            orphaned[e["key"]] = e["domain"]
    assert not orphaned, f"invariant WHY not found in its docs/context domain file (dropped?): {orphaned}"

def _key_convention_bullets(text: str) -> list[str]:
    # bullets under '## Key conventions' (top-level '- ' items until the next '## ')
    lines = text.splitlines()
    out, in_sec, cur = [], False, None
    for ln in lines:
        if ln.startswith("## "):
            if cur is not None:
                out.append(cur); cur = None
            in_sec = ln.strip() == "## Key conventions"
            continue
        if not in_sec:
            continue
        if ln.startswith("- "):
            if cur is not None:
                out.append(cur)
            cur = ln
        elif cur is not None and (ln.startswith("  ") or ln.strip() == ""):
            cur += "\n" + ln
    if cur is not None:
        out.append(cur)
    return out

def test_per_bullet_char_cap_and_pointer():
    bullets = _key_convention_bullets(_agents_text())
    assert bullets, "no Key-conventions bullets parsed"
    too_long = [b[:60] for b in bullets if len(b) > _MAX_BULLET_CHARS]
    assert not too_long, f"bullets exceed {_MAX_BULLET_CHARS} chars: {too_long}"
    no_ptr = [
        b[:60] for b in bullets
        if not re.search(r"ADR-\d+", b) and "docs/context/" not in b
    ]
    assert not no_ptr, f"bullets missing ADR/docs pointer: {no_ptr}"

def test_context_pointers_resolve():
    txt = _agents_text()
    refs = set(re.findall(r"docs/context/([A-Za-z0-9_-]+\.md)", txt))
    missing = [r for r in refs if not (_CONTEXT / r).is_file()]
    assert not missing, f"AGENTS.md points to missing docs/context files: {missing}"

def test_per_invariant_completeness():
    txt = _agents_text()
    inv = json.loads(_INV.read_text(encoding="utf-8"))
    lost = {}
    for e in inv:
        if e.get("dropped_with_owner_approval"):
            continue
        absent = [t for t in e["required_tokens"] if t not in txt]
        if absent:
            lost[e["key"]] = absent
    assert not lost, f"invariants lost in AGENTS.md (tokens absent): {lost}"
```

- [ ] **Step 2: Run the module RED**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agents_md_budget.py -v`
Expected: `test_inventory_tokens_exist_in_pinned_source` PASSES; every `AGENTS.md`- and
`docs/context/`-dependent test FAILS (neither exists yet — they are built in Tasks 3–5). This is
the intended red state.

- [ ] **Step 3: Do NOT commit.**

---

## Task 3: Create the 17 `docs/context/` class-2 files

**Files:**
- Create: `docs/context/{providers,tracking-features,orientation,id-compat,xt,xt-gk,gkdv,rest-defense,gk-metrics,event-metrics,tracking-metrics,vaep,trained-models,velocity-fov,corpus-drivers,ci,conventions-core}.md`

**Interfaces:**
- Produces: 17 files, each holding the class-2 WHY/history/measurement for its domain (spec §3.2
  table), organized under `##`/`###` headings keyed to the invariants that point at it.

- [ ] **Step 1: For each domain file, extract its class-2 content from `/tmp/claude_md_77286f4.md`**

Per the spec §3.2 table, MOVE (cut, not summarize) the rationale/history/measurement of each
subsystem/bullet into its domain file. Apply §3.3: keep the imperative RULE for class-1 (Task 4);
move everything explanatory here. For a multi-domain ADR (§2 rule: ADR-043 → `id-compat.md` +
`gkdv.md`; ADR-090 → `tracking-metrics.md` + `event-metrics.md`), split the WHY by domain and add
a one-line `see also docs/context/<other>.md` cross-link in each half.

- [ ] **Step 2: Verify content landed, not just files created (PLAN-r1 fix 3)**

Run: `for f in docs/context/*.md; do test -s "$f" || echo "EMPTY: $f"; done` → no output; confirm
the 17 filenames exactly match the spec table.
Then run the conservation guards (a stub tree FAILS these):
`./.venv/Scripts/python.exe -m pytest tests/test_agents_md_budget.py::test_context_bulk_conservation tests/test_agents_md_budget.py::test_context_domain_tokens_landed -v`
Expected: BOTH PASS — total `docs/context/` ≥ 100 KB (the moved WHY landed) AND every non-dropped
invariant's tokens/ADR refs appear in its own domain file (no block silently dropped).

- [ ] **Step 3: Do NOT commit.**

---

## Task 4: Build `AGENTS.md` (rename + class-1 partition)

**Files:**
- Rename: `CLAUDE.md` → `AGENTS.md` (`git mv`, preserves history)
- Modify: `AGENTS.md` (reduce to class-1)

**Interfaces:**
- Consumes: the inventory fixture (Task 1) and the `docs/context/` files (Task 3).
- Produces: `AGENTS.md` — class-1 only, ≤ TARGET, every invariant's distinctive tokens present,
  every bullet ≤ 600 chars with an ADR/docs pointer, every `docs/context/` pointer resolving.

- [ ] **Step 1: Rename**

Run: `git mv CLAUDE.md AGENTS.md`

- [ ] **Step 2: Add the self-documenting meta-rule at the top of `AGENTS.md`**

```markdown
> **Maintaining this file:** AGENTS.md holds CURRENT terse rules + pointers ONLY.
> Rationale / history / measurement lives in `docs/context/<domain>.md`. A Key-conventions
> bullet is ≤ 600 chars and carries an `ADR-XXX` ref or a `docs/context/` pointer.
> Enforced by `tests/test_agents_md_budget.py`. Canonical filename is AGENTS.md; `CLAUDE.md`
> is a one-line importer for Claude Code.
```

- [ ] **Step 3: Partition every section to class-1**

For each Architecture subsystem: collapse the paragraph to one MAP line (`**name** (path/):
purpose; entry points; → docs/context/<d>.md`) + its hard invariants as ≤1-line each with ADR
refs. Keep the TF table verbatim. For each Key-conventions/Testing bullet: collapse to the terse
imperative RULE + `(ADR-XXX; docs/context/<d>.md)`, ≤ 600 chars. Keep the "Where history lives"
table (add a `docs/context/` row) and the Open Items / Dependencies pointers. Every distinctive
symbol named in the inventory fixture MUST survive in the class-1 text (that is what the
completeness test checks).

- [ ] **Step 4: Measure landed size**

Run: `wc -c AGENTS.md`
Record the byte count; it must be ≤ `_TARGET_BYTES` (45 KB). If over, tighten further (move more
narrative to `docs/context/`).

- [ ] **Step 5: Run the partition tests**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agents_md_budget.py -k "not claude_md and not ceiling" -v`
Expected: `test_agents_md_exists`, `test_target_size`, `test_per_bullet_char_cap_and_pointer`,
`test_context_pointers_resolve`, `test_per_invariant_completeness`, `test_context_bulk_conservation`,
`test_context_domain_tokens_landed`, `test_inventory_tokens_exist_in_pinned_source` all PASS.
(`test_claude_md_is_the_import_shim` still fails — CLAUDE.md is gone until Task 5; `test_ceiling_size`
is finalized in Task 8.)

- [ ] **Step 6: Do NOT commit.**

---

## Task 5: Create the `CLAUDE.md` `@import` shim

**Files:**
- Create: `CLAUDE.md`

- [ ] **Step 1: Write the shim**

```markdown
<!-- Canonical instructions live in AGENTS.md (the cross-tool convention). -->
<!-- This file exists only so Claude Code loads them via @import. -->
@AGENTS.md
```

- [ ] **Step 2: Run the redirect test**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agents_md_budget.py::test_claude_md_is_the_import_shim -v`
Expected: PASS.

- [ ] **Step 3: Do NOT commit.**

---

## Task 6: Reference sweep (LIVE repoint / HISTORICAL leave / CC-mechanism keep)

**Files:**
- Modify: the LIVE-bucket files identified below.

- [ ] **Step 1: Enumerate every reference**

Run (tracked files only — PLAN-r1 fix 5, never `grep -rn .` which walks `.venv/`):
`git grep -n "CLAUDE.md" -- '*.py' '*.md' '*.yml' '*.yaml' '*.toml' | grep -v "2026-09-24-agents-md-restructure"`
Produce a classified list: for each hit, tag LIVE / HISTORICAL / CC-mechanism.

- [ ] **Step 2: Classify (buckets from spec §3.6)**

- **LIVE** (means the current canonical instruction content): repoint to `AGENTS.md`. Examples:
  a test that reads `CLAUDE.md` by name for content; the README/CONTRIBUTING if they describe
  "the instructions file"; `docs/context/` cross-refs.
- **HISTORICAL** (past `docs/superpowers/specs/`, `plans/`, `adrs/`, `CHANGELOG.md`, `docs/research/`):
  LEAVE AS-IS — they correctly name `CLAUDE.md` as of their time.
- **CC-mechanism** (the loader entry, e.g. hooks/tooling referring to the file CC reads): keep —
  `CLAUDE.md` still exists as the shim.

- [ ] **Step 3: Repoint only the LIVE bucket; leave the rest**

Edit each LIVE hit to reference `AGENTS.md`. Do NOT touch HISTORICAL or CC-mechanism hits.

- [ ] **Step 4: Verify no LIVE content-reference remains stale**

Re-run `git grep -n "CLAUDE.md"`; confirm every remaining hit is HISTORICAL or CC-mechanism (or the
shim/new spec/plan). Record the final classification in the commit message body.

- [ ] **Step 5: Do NOT commit.**

---

## Task 7: TODO.md top summary

**Files:**
- Modify: `TODO.md`

- [ ] **Step 1: Replace the "Last updated" / current-summary block**

Replace it wholesale (repo convention: no history kept in the summary) with a one-block summary of
this cycle: the AGENTS.md restructure — class-1/class-2 partition, `docs/context/` tree, anti-bloat
gate, no version bump. Remove any completed items shipped elsewhere. Keep the document's shape.

- [ ] **Step 2: Run the TODO format guard**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_todo_md_format.py -v`
Expected: PASS.

- [ ] **Step 3: Do NOT commit.**

---

## Task 8: Finalize ceiling, full green, stop for review + single commit

**Files:**
- Modify: `tests/test_agents_md_budget.py` (`_CEILING_BYTES`)

- [ ] **Step 1: Set the CEILING from measured landed size**

Set `_CEILING_BYTES = ceil(landed_bytes * 1.15)` using the `wc -c AGENTS.md` value from Task 4
(must still be ≤ 45 KB target). This makes the ceiling the anti-regression brake above the actual
landed size.

- [ ] **Step 2: Run the full budget gate**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_agents_md_budget.py -v`
Expected: ALL PASS (oracle-sanity, exists, target, ceiling, redirect, char-cap+pointer,
pointers-resolve, per-invariant completeness, bulk conservation, domain-tokens landed).

- [ ] **Step 3: Lint + type-check + full suite, at CI scope (PLAN-r1 fix 2, 6)**

Run (ruff + pyright via SYSTEM python per repo venv policy; pyright IS run — repo config scopes
`tests/`, cheap):
```bash
python -m ruff check silly_kicks/ tests/ scripts/
python -m ruff format --check silly_kicks/ tests/ scripts/
python -m pyright
```
Run the full suite via the PINNED 3.10 venv WITH the CI invariant flag:
```bash
SILLY_KICKS_ASSERT_INVARIANTS=1 ./.venv/Scripts/python.exe -m pytest tests/ -m "not e2e" -q
```
Expected: ruff clean; pyright at the `77286f4` baseline (no NEW findings — the only `silly_kicks/`
edits are two comment-only doc-pointer repoints, which change no types); suite green. "Green"
without the env var ≠ CI green.

- [ ] **Step 4: STOP — hand off for independent `/review-impl`**

Deliver the FROZEN working tree (do NOT edit during review — freeze-tree-on-handoff). The owner
starts an INDEPENDENT session for `/review-impl`. The author does not run their own review.

- [ ] **Step 5: After review clears — propose the SINGLE commit (SEPARATE owner go)**

Show the full file list + diff summary. On explicit owner approval, ONE commit:
`docs: restructure CLAUDE.md → AGENTS.md class-1 + docs/context/ on-demand tree + anti-bloat gate`
including the spec, this plan, the fixture, the test, the 17 context files, `AGENTS.md`,
`CLAUDE.md` shim, LIVE-repointed refs, and `TODO.md`. Trailer `Co-Authored-By: Claude Opus 4.8
<noreply@anthropic.com>`; NO `Claude-Session` trailer. No version bump, no CHANGELOG version entry.

- [ ] **Step 6: Push — SEPARATE owner go (PLAN-r1 fix 1)**

`git push origin docs/agents-md-restructure` only after its own explicit approval.

- [ ] **Step 7: Open the PR — SEPARATE owner go (PLAN-r1 fix 1)**

`gh pr create --base main ...` only after its own explicit approval (a PR is outward-facing; the
commit approval does not cover it).

- [ ] **Step 8: Merge — SEPARATE owner go, CI green**

Merge only on a distinct owner go AND CI green (`--merge` is not required here — no 2-commit
provenance; owner picks merge mode).

---

## Self-review

- **Spec coverage:** §3.1 shape → T4; §3.2 17 files → T3; §3.3 partition criteria → T4 step 3;
  §3.4 per-invariant safety → T1 (fixture) + T2 (`test_per_invariant_completeness`); §3.5 gate
  (target/ceiling/char-cap/redirect/meta-rule) → T2 + T4 step 2 + T5 + T8; §3.6 rename + sweep →
  T4 step 1 + T5 + T6; §4 review/commit → T8; §5 TODO → T7. All spec sections mapped.
- **Placeholder scan:** none — test code and fixture schema are concrete; content-movement steps
  are procedural against §3.2/§3.3 (the class-2 text is the old file's content, moved, not invented).
- **Type consistency:** `_TARGET_BYTES`/`_CEILING_BYTES`/`_MAX_BULLET_CHARS`, `_INV`, `_PIN`,
  `_key_convention_bullets`, `_agents_text` used consistently across Task 1/2/8. Fixture schema
  (`key`/`domain`/`adr_refs`/`required_tokens`/`dropped_with_owner_approval`) identical in T1 and
  the T2 completeness test.
