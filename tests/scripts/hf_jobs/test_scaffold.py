"""The scaffolder (ADR-128 spec 5.8): a scaffolded trainer or driver passes the born-ready gate's
predicates on day one, is registered between the markers, and its generated parity test passes."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.hf_jobs import scaffold as S
from tests.scripts import _corpus_load_rules as clr

REPO = Path(__file__).resolve().parents[3]
_GIT = shutil.which("git") or "git"


def _declares_pipeline(tree) -> bool:  # the gate's predicate (tests/scripts/test_compute_contract.py)
    return any(
        isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "PIPELINE" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and clr._call_name(n.value) == "Pipeline"
        for n in tree.body
    )


def _copy(tmp: Path) -> Path:
    root = tmp / "repo"
    shutil.copytree(REPO / "scripts", root / "scripts", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (root / "tests" / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "tests" / "__init__.py", root / "tests" / "__init__.py")
    shutil.copy(REPO / "tests" / "scripts" / "_parity.py", root / "tests" / "scripts" / "_parity.py")
    (root / "docs" / "artifacts").mkdir(parents=True)
    shutil.copy(REPO / "docs" / "artifacts" / "manifest.toml", root / "docs" / "artifacts" / "manifest.toml")
    shutil.copy(REPO / "pyproject.toml", root / "pyproject.toml")
    # the repo's ignore rules: an untracked __pycache__/ would otherwise move the dirty-tree code identity
    # between the serial and the fanned run (decision P9 digests the working tree)
    shutil.copy(REPO / ".gitignore", root / ".gitignore")
    # a git checkout: the stage CLI's code identity must agree across the fanned processes (decision P9)
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "copy"]):
        subprocess.run(  # noqa: S603 -- fixed argv
            [_GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=root, check=True, capture_output=True
        )
    return root


def _py(root: Path, code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=False)  # noqa: S603


def test_the_real_tree_carries_every_scaffold_marker_once():
    reg = (REPO / "scripts/hf_jobs/registry.py").read_text(encoding="utf-8")
    toys = (REPO / "scripts/_toy_corpora.py").read_text(encoding="utf-8")
    assert reg.count(S.MARKERS["trainer"]) == 1 and reg.count(S.MARKERS["driver"]) == 1
    assert toys.count(S.MARKERS["toy_params"]) == 1 and toys.count(S.MARKERS["toy_builders"]) == 1


@pytest.mark.parametrize("kind", ["trainer", "driver"])
def test_scaffold_writes_a_born_ready_module_and_registers_it(tmp_path, kind):
    root = _copy(tmp_path)
    manifest_before = (root / "docs/artifacts/manifest.toml").read_text(encoding="utf-8")
    assert S.main([kind, "toy_born_ready", "--root", str(root)]) == 0
    module = "train_toy_born_ready" if kind == "trainer" else "build_toy_born_ready"
    tree = ast.parse((root / "scripts" / f"{module}.py").read_text(encoding="utf-8"))
    assert _declares_pipeline(tree)
    p = _py(
        root, f"import scripts.{module} as m; m.PIPELINE.validate(); assert m.PIPELINE.module == 'scripts.{module}'"
    )
    assert p.returncode == 0, p.stderr
    p = _py(
        root,
        "from scripts.hf_jobs.registry import DRIVERS, REGISTRY;"
        " from scripts._toy_corpora import TOY_PARAMS, TOY_BUILDERS;"
        f" m = 'scripts.{module}'; assert m in TOY_PARAMS and m in TOY_BUILDERS;"
        " assert any(s.pipeline == m for s in REGISTRY.values()) or any(d.module == m for d in DRIVERS.values())",
    )
    assert p.returncode == 0, p.stderr
    parity = (root / "tests/scripts/test_parity_toy_born_ready.py").read_text(encoding="utf-8")
    assert f'"scripts.{module}"' in parity
    manifest_after = (root / "docs/artifacts/manifest.toml").read_text(encoding="utf-8")
    if kind == "trainer":
        assert (
            manifest_after.startswith(manifest_before) and "# [[artifacts]]" in manifest_after[len(manifest_before) :]
        )
        p = _py(root, "from scripts.hf_jobs import manifest as M; M.load()")  # the stub stays commented out
        assert p.returncode == 0, p.stderr
    else:
        assert manifest_after == manifest_before


def test_scaffold_refuses_an_existing_name(tmp_path):
    root = _copy(tmp_path)
    S.main(["trainer", "toy_born_ready", "--root", str(root)])
    with pytest.raises(SystemExit, match=r"exists|registered"):
        S.main(["trainer", "toy_born_ready", "--root", str(root)])
    with pytest.raises(SystemExit, match=r"exists|registered"):  # train_ghost_gk.py exists AND is registered
        S.main(["trainer", "ghost_gk", "--root", str(root)])
    (root / "scripts" / "train_xsuccess.py").unlink()  # the module is gone; its registry entry still names it
    with pytest.raises(SystemExit, match="already registered"):
        S.main(["trainer", "xsuccess", "--root", str(root)])


@pytest.mark.slow  # slow: measured 18 s (trainer) + 17 s (driver) on the Windows dev box; runs the executor (ADR-023)
@pytest.mark.parametrize("kind", ["trainer", "driver"])
def test_the_generated_parity_test_passes(tmp_path, kind):
    root = _copy(tmp_path)
    S.main([kind, "toy_born_ready", "--root", str(root)])
    p = subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, "-m", "pytest", "tests/scripts/test_parity_toy_born_ready.py", "-q", "-p", "no:randomly",
         "-p", "no:cacheprovider", "--basetemp", str(tmp_path / "bt")],
        cwd=root, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-2000:]
