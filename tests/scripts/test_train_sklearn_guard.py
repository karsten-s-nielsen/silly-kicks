"""VSSOT-IMPL-03: the ghost-GK trainer refuses a scikit-learn outside the supported fit range.

HistGradientBoosting produces different trees across sklearn versions (measured on the ghost re-fit:
same corpus/commit/pandas under 1.7.2 vs 1.9.0 -> different weights, different sha256). The ``[train]``
extra pins ``scikit-learn>=1.9,<2`` but marker-gates it to ``python_version >= '3.11'`` (sklearn 1.9
dropped Python 3.10), so on py3.10 ``pip install .[train]`` silently resolves an older sklearn. This
guard makes that footgun LOUD at the ghost trainer's fit-time entry point instead of shipping weights
that silently disagree with the bundled artifacts.

Scope is the GHOST trainer only, and that is correct, not a shortcut: the logistic bundles
(gk_completion / gk_retention / receiver) fit a convex problem whose solution is version-stable, and
their own load-time provenance checks compare the MAJOR version only (``_gk_completion.py``:
``sv.split(".")[0] != ...``), so 1.7 vs 1.9 is a non-issue for them. Only ghost is sensitive to the
MINOR sklearn version. The guard lives at the TRAINER (scripts/), never in ``GhostGkModel.fit()``: the
non-slow library unit tests fit toy models on whatever sklearn the CI leg resolved (1.7.2 on py3.10),
and a library-level raise would redden the py3.10 legs; the ghost trainer's own smokes are all
``@slow`` (primary 3.12 leg, sklearn 1.9.0), so this guard passes there.
"""

from __future__ import annotations

import ast
import pathlib

import _train_guard
import pytest
import sklearn

_REPO = pathlib.Path(__file__).resolve().parents[2]
_GHOST_TRAINER = _REPO / "scripts" / "train_ghost_gk.py"


def test_raises_below_the_supported_floor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sklearn, "__version__", "1.7.2")
    with pytest.raises(RuntimeError, match="supported training range"):
        _train_guard.require_training_sklearn()


def test_raises_at_or_above_the_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sklearn, "__version__", "2.0.0")
    with pytest.raises(RuntimeError, match="supported training range"):
        _train_guard.require_training_sklearn()


def test_passes_across_the_supported_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inclusive floor, whole 1.x above it, exclusive ceiling at 2.0.
    for ok in ("1.9.0", "1.9.3", "1.10.0", "1.99.0"):
        monkeypatch.setattr(sklearn, "__version__", ok)
        assert _train_guard.require_training_sklearn() == ok


def test_supports_training_bool_matches_the_range(monkeypatch: pytest.MonkeyPatch) -> None:
    # The skip-companion the ghost smokes use must agree with the raising guard on the SAME range.
    for below in ("1.7.2", "1.8.0"):
        monkeypatch.setattr(sklearn, "__version__", below)
        assert _train_guard.sklearn_supports_training() is False
    for ok in ("1.9.0", "1.10.0"):
        monkeypatch.setattr(sklearn, "__version__", ok)
        assert _train_guard.sklearn_supports_training() is True
    monkeypatch.setattr(sklearn, "__version__", "2.0.0")
    assert _train_guard.sklearn_supports_training() is False


def test_ghost_trainer_main_calls_the_guard() -> None:
    """Anti-vacuity: the guard must be WIRED into the trainer, not merely defined.

    A defined-but-uncalled guard would pass the behaviour tests above while shipping the footgun.
    """
    tree = ast.parse(_GHOST_TRAINER.read_text(encoding="utf-8"))
    called = {
        (n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
    }
    assert "require_training_sklearn" in called, (
        "train_ghost_gk.py must call require_training_sklearn() at its fit-time entry point"
    )
