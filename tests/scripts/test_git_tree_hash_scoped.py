"""Scoped code identity (spec 2026-11-08 tracking-batch W0.4, R11).

Every test builds its own throwaway git repo, so nothing here depends on this checkout's state.
"""

from __future__ import annotations

import os
import pathlib
import subprocess

import pytest

from scripts._provenance import (
    CODE_IDENTITY_EXCLUDE,
    CODE_IDENTITY_PATHS,
    ProvenanceError,
    git_tree_hash,
)


def _git(repo: pathlib.Path, *args: str) -> str:
    argv = ["git", "-C", str(repo), *args]
    return subprocess.run(argv, check=True, capture_output=True, text=True).stdout.strip()  # noqa: S603


def _write(repo: pathlib.Path, rel: str, text: str) -> None:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode())


@pytest.fixture
def repo(tmp_path, monkeypatch) -> pathlib.Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "core.autocrlf", "false")
    for rel, text in {
        "silly_kicks/__init__.py": "VALUE = 1\n",
        "silly_kicks/_version.py": '__version__ = "1.0.0"\n',
        "silly_kicks/core.py": "def f():\n    return 1\n",
        "scripts/s.py": "X = 1\n",
        "docs/x.md": "doc\n",
        "pyproject.toml": "[project]\nname = 'x'\n",
        "uv.lock": "lock\n",
        ".gitignore": "__pycache__/\n*.pyc\n",
    }.items():
        _write(r, rel, text)
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    monkeypatch.chdir(r)
    return r


def test_identity_constants():
    assert CODE_IDENTITY_PATHS == ("silly_kicks", "scripts", "pyproject.toml", "uv.lock")
    assert CODE_IDENTITY_EXCLUDE == ("silly_kicks/_version.py",)


def test_scoped_hash_ignores_docs(repo):
    before = git_tree_hash(paths=CODE_IDENTITY_PATHS)
    _write(repo, "docs/x.md", "changed\n")
    assert git_tree_hash(paths=CODE_IDENTITY_PATHS) == before


def test_scoped_hash_sees_untracked_code(repo):
    before = git_tree_hash(paths=CODE_IDENTITY_PATHS)
    _write(repo, "scripts/new.py", "Y = 2\n")
    assert git_tree_hash(paths=CODE_IDENTITY_PATHS) != before


def test_scoped_hash_excludes_version_file(repo):
    before = git_tree_hash(paths=CODE_IDENTITY_PATHS)
    _write(repo, "silly_kicks/_version.py", '__version__ = "9.9.9"\n')
    assert git_tree_hash(paths=CODE_IDENTITY_PATHS) == before


def test_scoped_hash_sees_code_edits(repo):
    before = git_tree_hash(paths=CODE_IDENTITY_PATHS)
    _write(repo, "silly_kicks/core.py", "def f():\n    return 2\n")
    assert git_tree_hash(paths=CODE_IDENTITY_PATHS) != before


def test_scoped_hash_git_failure_raises(repo, monkeypatch):
    monkeypatch.setenv("GIT_DIR", str(repo / "no-such-git-dir"))
    with pytest.raises(ProvenanceError):
        git_tree_hash(paths=CODE_IDENTITY_PATHS)


def test_tree_hash_unscoped_unchanged(repo):
    # the BASE_SHA algorithm, run here: read-tree HEAD into a throwaway index, add -A, write-tree
    _write(repo, "scripts/untracked.py", "Z = 3\n")
    env = {**os.environ, "GIT_INDEX_FILE": str(repo.parent / "probe-index")}
    for args in (["read-tree", "HEAD"], ["add", "-A"]):
        subprocess.run(["git", *args], check=True, capture_output=True, env=env)  # noqa: S603, S607
    probe = subprocess.run(["git", "write-tree"], check=True, capture_output=True, text=True, env=env)  # noqa: S607
    want = probe.stdout.strip()
    assert git_tree_hash() == want
