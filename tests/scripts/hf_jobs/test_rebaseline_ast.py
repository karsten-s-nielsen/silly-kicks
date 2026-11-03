"""Rebaseline code delta = an AST compare of each input module at two endpoints, docstrings removed
(HFU spec 14.3, F9): a docstring-only commit is not a code delta; a body change is."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from scripts.hf_jobs import adopt as A

V1 = '"""Module doc."""\n\n\ndef f(x):\n    """Old doc."""\n    return x + 1\n'
DOC_ONLY = '"""Module doc, reworded."""\n\n\ndef f(x):\n    """New doc, longer."""\n\n    return x + 1\n'
BODY = '"""Module doc."""\n\n\ndef f(x):\n    """Old doc."""\n    return x + 2\n'


_GIT = shutil.which("git") or "git"


def _git(root: Path, *args: str) -> str:
    p = subprocess.run([_GIT, *args], cwd=root, check=True, capture_output=True, text=True)  # noqa: S603 -- fixed argv
    return p.stdout.strip()


def _repo(tmp_path: Path, second: str) -> tuple[Path, str, str]:
    root = tmp_path / "r"
    (root / "pkg").mkdir(parents=True)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    (root / "pkg" / "m.py").write_text(V1, encoding="utf-8")
    (root / "pkg" / "other.py").write_text("X = 1\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "v1")
    base = _git(root, "rev-parse", "HEAD")
    (root / "pkg" / "m.py").write_text(second, encoding="utf-8")
    _git(root, "commit", "-qam", "v2")
    return root, base, _git(root, "rev-parse", "HEAD")


def test_docstring_only_commit_is_not_code_delta(tmp_path):
    root, base, head = _repo(tmp_path, DOC_ONLY)
    assert A._code_delta(["pkg/m.py", "pkg/other.py"], base, head, cwd=root) == []


def test_body_change_is_code_delta(tmp_path):
    root, base, head = _repo(tmp_path, BODY)
    assert A._code_delta(["pkg/m.py", "pkg/other.py"], base, head, cwd=root) == ["pkg/m.py"]


def test_a_module_missing_at_one_endpoint_is_a_code_delta(tmp_path):
    root, base, head = _repo(tmp_path, BODY)
    (root / "pkg" / "new.py").write_text("Y = 2\n", encoding="utf-8")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "v3")
    head = _git(root, "rev-parse", "HEAD")
    assert A._code_delta(["pkg/new.py"], base, head, cwd=root) == ["pkg/new.py"]
