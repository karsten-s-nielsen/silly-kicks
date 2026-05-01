"""Loose structural check on the NOTICE file (PR-S20 establishes the canonical record).

Asserts the file exists, has the two top-level sections (Third-Party Libraries +
Mathematical/Methodological References), cites all 5 PR-S20 reference authors, and
that README.md links to it. Catches accidental drift in low-cost structural ways
without enforcing per-function/NOTICE coupling (which would be too brittle).
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTICE_PATH = REPO_ROOT / "NOTICE"


def test_notice_exists():
    assert NOTICE_PATH.exists(), "NOTICE file missing at repo root"


def test_notice_has_third_party_libraries_section():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    assert "Third-Party Libraries" in text


def test_notice_has_mathematical_references_section():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    assert "Mathematical" in text and "References" in text


def test_notice_cites_all_pr_s20_authors():
    text = NOTICE_PATH.read_text(encoding="utf-8")
    for author in ["Lucey", "Anzer", "Spearman", "Power", "Pollard"]:
        assert author in text, f"missing PR-S20 reference author: {author}"


def test_readme_links_to_notice():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert "NOTICE" in readme
