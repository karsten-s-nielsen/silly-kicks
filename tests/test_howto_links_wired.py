"""Link-check guard: references in docs/howto/*.md + docs/context/*.md resolve.

Covers four reference kinds, scanned across the class-2 runbooks (docs/howto) and the class-1.5
per-domain narrative (docs/context):

- markdown links ``](path#anchor)`` — HARD: target file must exist; a ``.md`` anchor must be a real heading slug.
- ADR refs ``ADR-NNN`` — HARD: ``docs/superpowers/adrs/ADR-NNN-*.md`` must exist.
- code paths ``silly_kicks|tests|scripts/.../x.py`` — HARD: the file must exist; a ``:NNN`` line
  suffix beyond EOF is SOFT (warned, non-failing — line numbers drift, paths do not).
- bare doc-path prose mentions ``docs/....md`` — HARD: the file must exist (the dominant ref kind here).
"""

from __future__ import annotations

import pathlib
import re

_REPO = pathlib.Path(__file__).resolve().parent.parent
_DIRS = [_REPO / "docs/howto", _REPO / "docs/context"]
_MD_LINK = re.compile(r"\]\((?!https?://)([^)\s#]+)(?:#([^)\s]+))?\)")
_ADR = re.compile(r"\bADR-(\d{3})\b")
_CODE = re.compile(r"\b((?:silly_kicks|tests|scripts)/[\w/]+\.py)(?::(\d+))?")
_DOCP = re.compile(r"\b(docs/[\w./-]+\.md)\b")  # bare doc-path prose mentions (the dominant ref kind here)


def _slug(h: str) -> str:  # GitHub-style heading slug
    return re.sub(r"[^\w\- ]", "", h.strip().lower()).replace(" ", "-")


def _headings(md: str) -> set[str]:
    return {_slug(m.group(1)) for m in re.finditer(r"^#{1,6}\s+(.*)$", md, re.M)}


def _collect(dirs: list[pathlib.Path] | None = None):
    md_files = [p for d in (dirs or _DIRS) for p in d.glob("*.md")]
    hard_errs: list[tuple[str, str, str]] = []
    soft_warns: list[tuple[str, str, str]] = []
    counts = {"md": 0, "adr": 0, "code": 0, "docp": 0}
    for p in md_files:
        txt = p.read_text(encoding="utf-8")
        for m in _MD_LINK.finditer(txt):
            counts["md"] += 1
            target = (p.parent / m.group(1)).resolve()
            if not target.exists():
                target = (_REPO / m.group(1)).resolve()
            if not target.exists():
                hard_errs.append((str(p), m.group(0), "md-link missing"))
                continue
            if m.group(2) and target.suffix == ".md":  # anchor
                if _slug(m.group(2)) not in _headings(target.read_text(encoding="utf-8")):
                    hard_errs.append((str(p), m.group(0), "md anchor missing"))
        for m in _ADR.finditer(txt):
            counts["adr"] += 1
            if not list((_REPO / "docs/superpowers/adrs").glob(f"ADR-{m.group(1)}-*.md")):
                hard_errs.append((str(p), m.group(0), "ADR file missing"))
        for m in _CODE.finditer(txt):
            counts["code"] += 1
            f = _REPO / m.group(1)
            if not f.exists():
                hard_errs.append((str(p), m.group(0), "code path missing"))
            elif m.group(2) and len(f.read_text(encoding="utf-8").splitlines()) < int(m.group(2)):
                soft_warns.append((str(p), m.group(0), "line beyond EOF (SOFT)"))
        for m in _DOCP.finditer(txt):
            counts["docp"] += 1
            if not (_REPO / m.group(1)).exists():
                hard_errs.append((str(p), m.group(0), "doc path missing"))
    return hard_errs, soft_warns, counts


def test_howto_context_references_resolve():
    hard, soft, _ = _collect()
    assert not hard, "unresolved references:\n" + "\n".join(map(str, hard))
    if soft:  # SOFT line-suffix warnings are surfaced, non-failing
        import warnings

        warnings.warn("line-suffix refs beyond EOF: " + "; ".join(str(s) for s in soft), stacklevel=2)


def test_linkcheck_non_vacuity_floor():
    # measured live @50fab4c: ADR 465 / code 120 / docp 39 (md-links 0 -> no md floor). Floors pinned
    # ~50% below so routine doc edits don't trip, but a wholesale regex break (0 matches) fails loud.
    _, _, counts = _collect()
    assert counts["adr"] >= 200 and counts["code"] >= 60 and counts["docp"] >= 20, counts


def test_linkcheck_precondition_catches_a_broken_ref(tmp_path):
    # a fixture md with one broken md-link, one bad ADR, one missing code path, one missing doc path
    # -> all four reported. Proves the guard is not vacuous (a precondition test for the liveness gate).
    bad = tmp_path / "howto"
    bad.mkdir()
    (bad / "x.md").write_text(
        "[a](does_not_exist.md) ADR-999 silly_kicks/nope_xyz.py docs/nope_abc.md", encoding="utf-8"
    )
    hard, _, _ = _collect(dirs=[bad])
    assert len(hard) == 4  # md-link + ADR + code-path + doc-path all reported
