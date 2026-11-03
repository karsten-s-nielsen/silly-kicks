"""Every optuna study under ``scripts/`` reports each trial (HFU spec 15.3): a ``.optimize(`` call passes
``callbacks=``, so a direct optuna study is never blind (launch-4 finding F21)."""

from __future__ import annotations

import ast
import pathlib

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"


def _optimize_calls() -> list[tuple[str, int, bool]]:
    out = []
    for path in sorted(_SCRIPTS.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "optimize":
                out.append((path.name, node.lineno, any(k.arg == "callbacks" for k in node.keywords)))
    return out


def test_every_optuna_study_has_a_progress_callback():
    calls = _optimize_calls()
    assert "train_xsuccess.py" in {c[0] for c in calls}, calls  # non-vacuity: the gate sees the known study
    assert [c for c in calls if not c[2]] == []
