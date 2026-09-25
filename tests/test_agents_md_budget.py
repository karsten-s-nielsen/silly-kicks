"""Anti-bloat + migration-safety gate for the always-loaded instruction surface.

`AGENTS.md` is the class-1 (always-loaded) instruction file; `docs/context/*.md` is the
on-demand class-2 store; `CLAUDE.md` is a one-line `@AGENTS.md` importer for Claude Code.

This gate keeps the always-loaded surface small and, crucially, proves no invariant was silently
dropped when the old 230 KB `CLAUDE.md` was partitioned:

- TARGET / CEILING byte checks on `AGENTS.md` (the ~80% cut + the re-bloat brake).
- Per-bullet CHAR cap + mandatory ADR/docs pointer (bullets are single physical lines, so a
  line-count cap would be vacuous).
- `CLAUDE.md` is exactly the `@AGENTS.md` shim.
- PER-INVARIANT completeness against a fixture snapshotted ONCE from `CLAUDE.md` @ 77286f4:
  every keyed invariant's DISTINCTIVE symbols must survive in `AGENTS.md`.
- Conservation: the moved WHY landed in `docs/context/` (bulk floor + per-block anchor).

See docs/superpowers/specs/2026-09-24-agents-md-restructure-design.md.
"""

import json
import re
from pathlib import Path

_PIN = "77286f4"
_INV = Path("tests/fixtures/agents_md_invariant_inventory.json")
_OLD_SNAPSHOT = Path("tests/fixtures/claude_md_at_77286f4.md")
_AGENTS = Path("AGENTS.md")
_CLAUDE = Path("CLAUDE.md")
_CONTEXT = Path("docs/context")

_TARGET_BYTES = 45 * 1024  # §3.1 TARGET — enforces the ~80% cut
_CEILING_BYTES = 27750  # §3.1 CEILING — landed (24131) + 15% (anti-regression brake)
_MAX_BULLET_CHARS = 600  # §3.5.2
_CONTEXT_MIN_BYTES = 100 * 1024  # the moved WHY must land, not vanish


def _agents_text() -> str:
    return _AGENTS.read_text(encoding="utf-8")


def _inventory() -> list[dict]:
    return json.loads(_INV.read_text(encoding="utf-8"))


def _old_claude_md() -> str:
    # The pinned snapshot of CLAUDE.md @ 77286f4, committed as a fixture so the oracle-sanity
    # check does NOT depend on git history depth -- CI shallow-clones (fetch-depth 1), so
    # `git show 77286f4:CLAUDE.md` returns exit 128 there. Verifiable locally that the fixture
    # equals the pinned blob (modulo line endings): `git show 77286f4:CLAUDE.md`.
    return _OLD_SNAPSHOT.read_text(encoding="utf-8")


def test_inventory_tokens_exist_in_pinned_source():
    # oracle-sanity: the fixture is snapshotted from the OLD file, not fabricated.
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
    # spec §3.6 "exactly": the only non-comment, non-blank line is @AGENTS.md.
    lines = _CLAUDE.read_text(encoding="utf-8").splitlines()
    content = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("<!--")]
    assert content == ["@AGENTS.md"], f"CLAUDE.md must be exactly the @AGENTS.md import (+comments); got {content}"
    assert _CLAUDE.stat().st_size < 512, "CLAUDE.md shim should be tiny (import + comment only)"


def test_context_bulk_conservation():
    # the narrative removed from CLAUDE.md must reappear in docs/context/, not be dropped.
    total = sum(p.stat().st_size for p in _CONTEXT.glob("*.md"))
    assert total >= _CONTEXT_MIN_BYTES, (
        f"docs/context/ total {total} B < floor {_CONTEXT_MIN_BYTES} B — moved WHY appears dropped"
    )


def test_context_domain_tokens_landed():
    # per-block guard: each non-dropped invariant's WHY landed in ITS domain file. A dropped block
    # loses ALL of its distinctive tokens AND its ADR refs from the file → fails.
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
    # top-level '- ' items under '## Key conventions', each with its continuation lines.
    out: list[str] = []
    in_sec = False
    cur: str | None = None
    for ln in text.splitlines():
        if ln.startswith("## "):
            if cur is not None:
                out.append(cur)
                cur = None
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
    no_ptr = [b[:60] for b in bullets if not re.search(r"ADR-\d+", b) and "docs/context/" not in b]
    assert not no_ptr, f"bullets missing ADR/docs pointer: {no_ptr}"


def test_context_pointers_resolve():
    refs = set(re.findall(r"docs/context/([A-Za-z0-9_-]+\.md)", _agents_text()))
    missing = [r for r in refs if not (_CONTEXT / r).is_file()]
    assert not missing, f"AGENTS.md points to missing docs/context files: {missing}"


def test_per_invariant_completeness():
    txt = _agents_text()
    lost = {}
    for e in _inventory():
        if e.get("dropped_with_owner_approval"):
            continue
        absent = [t for t in e["required_tokens"] if t not in txt]
        if absent:
            lost[e["key"]] = absent
    assert not lost, f"invariants lost in AGENTS.md (tokens absent): {lost}"
