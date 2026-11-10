"""Every numba dispatch site routes through ``silly_kicks._numba_dispatch`` (HFU spec §13.2 census).

For each kernel family: the explicit switch ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1`` selects the numpy reference and
never loads a numba kernel; with numba hidden, the family raises ``NumbaUnavailableError`` instead of falling back
silently. Explicit engine arguments: an explicit numba request requires numba, an explicit numpy request never
imports it.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from silly_kicks import _numba_dispatch as nd

ROOT = pathlib.Path(__file__).resolve().parents[1] / "silly_kicks"

#: The only modules allowed to import numba: the dispatch module and the kernel modules.
KERNEL_MODULES = frozenset(
    {
        "_numba_dispatch.py",
        "tracking/_ghost_gk_numba.py",
        "tracking/_das_numba.py",
        "tracking/pitch_control/_numba_kernels.py",
        "tracking/_ball_carrier_numba.py",
        "coordination/_kernels/_numba.py",
        "xtgk/_turnover.py",
    }
)


@pytest.fixture
def no_switches(monkeypatch):
    for fam in nd.FAMILIES:
        monkeypatch.delenv(f"SILLY_KICKS_{fam}_FORCE_NUMPY", raising=False)


@pytest.fixture
def hidden_numba(monkeypatch, no_switches):
    """Numba cannot be imported, and every cached probe is reset."""
    from silly_kicks.tracking import _ghost_gk

    monkeypatch.setitem(sys.modules, "numba", None)
    monkeypatch.setattr(nd, "_NUMBA", None)
    monkeypatch.setattr(_ghost_gk, "_HAS_GHOST_NUMBA", None)


def _imports_numba(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split(".")[0] == "numba" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "numba" and node.level == 0:
            return True
    return False


def test_no_numba_import_outside_kernels():
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in KERNEL_MODULES:
            continue
        if _imports_numba(ast.parse(path.read_text(encoding="utf-8"))):
            offenders.append(rel)
    assert offenders == []


# ---- explicit engine arguments ------------------------------------------------------------------


# ---- every family through its PUBLIC entry (plan Task 11), and no silent kernel-import fallback --------------------


class _KernelCalledError(AssertionError):
    pass


def _refuse(*args, **kwargs):
    raise _KernelCalledError("a numba kernel ran under the FORCE_NUMPY switch")


@pytest.fixture(scope="module")
def ghost_model():
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, GhostGkModel

    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 26)), columns=GHOST_GK_FEATURE_NAMES)
    X["phase"] = rng.integers(0, 3, n).astype(float)
    m = GhostGkModel(n_estimators=20)
    m.fit(X, pd.DataFrame({"gk_x": rng.uniform(2, 20, n), "gk_y": rng.uniform(25, 45, n)}))
    return m, X.iloc[:3]


def _entry_ghost(ghost_model):
    m, X = ghost_model
    return m.predict_mean(X)


def _entry_das(_):
    from silly_kicks.tracking._das_engine import compute_das
    from silly_kicks.tracking._das_pack import pack_frames
    from silly_kicks.tracking._das_params import DAS_PARAMS
    from tests.tracking._das_helpers import golden_frames

    return compute_das(pack_frames(golden_frames("S01"), attacking_direction_col="dir"), DAS_PARAMS, engine="auto")


def _entry_coordination(_):
    import dataclasses

    from silly_kicks.coordination import compute_team_coordination
    from silly_kicks.coordination._config import CoordinationParams
    from tests.coordination._fixtures import make_coordination_match

    params = dataclasses.replace(CoordinationParams(), n_surrogates=3, welch_segment_s=20.0)
    return compute_team_coordination(make_coordination_match(seconds=60.0, hz=10.0, provider="sportec"), params=params)


def _entry_pitch_control(method):
    def entry(_):
        from silly_kicks.tracking.features import pitch_control_at_target
        from tests.tracking.pitch_control.test_pitch_control_at_target import _AWAY, _HOME, _action, _frame

        fr = _frame(
            100,
            controllers_xy=[(80, 20), (79, 21), (81, 19)],
            opponents_xy=[(25, 48), (26, 47), (24, 49)],
            controller_team=_HOME,
            opp_team=_AWAY,
        )
        return pitch_control_at_target(_action(_HOME, 80.0, 20.0), fr, method=method)

    return entry


def _entry_ball_carrier(_):
    from silly_kicks.tracking._ball_carrier import infer_ball_carrier
    from tests.tracking.test_ball_carrier import _make_carrier_frame

    frames = _make_carrier_frame(
        ball_x=50.0,
        ball_y=34.0,
        players=[
            dict(pid=10, tid=1, x=51.0, y=34.0, vx=0.0, vy=0.0),
            dict(pid=20, tid=2, x=53.0, y=34.0, vx=0.0, vy=0.0),
        ],
    )
    return infer_ball_carrier(frames)


def _entry_xtgk(_):
    from silly_kicks.xtgk._turnover import EmpiricalTurnoverValue
    from tests.xtgk.test_turnover_perf import _churny_corpus

    return EmpiricalTurnoverValue()._opp_first_shot_after_turnover(_churny_corpus(), "xg", window_seconds=None)


#: family -> (public entry, the numba kernels it must NOT call under the switch, as (module, attribute)).
ENTRIES = {
    "GHOST": (_entry_ghost, [("silly_kicks.tracking._ghost_gk", "_leaf_values_numba")]),
    "DAS": (_entry_das, [("silly_kicks.tracking._das_numba", "compute_das_numba")]),
    "COORDINATION": (_entry_coordination, "coordination"),
    "PITCH_CONTROL": (
        _entry_pitch_control("spearman"),
        [
            ("silly_kicks.tracking.pitch_control._numba_kernels", "tti_numba"),
            ("silly_kicks.tracking.pitch_control._numba_kernels", "influence_numba"),
        ],
    ),
    "PITCH_CONTROL_FB": (
        _entry_pitch_control("fernandez_bornn"),
        [("silly_kicks.tracking.pitch_control._numba_kernels", "gaussian_influence_numba")],
    ),
    "BALL_CARRIER": (_entry_ball_carrier, [("silly_kicks.tracking._ball_carrier_numba", "_carrier_loop_numba")]),
    "XTGK": (_entry_xtgk, [("silly_kicks.xtgk._turnover", "_SCAN_NUMBA")]),
}


def _spy_kernels(monkeypatch, kernels):
    import importlib

    if kernels == "coordination":  # every @njit kernel the coordination modules bind by name
        for mod_name in (
            "silly_kicks.coordination._kernels._numba",
            "silly_kicks.coordination._kernels._cluster",
            "silly_kicks.coordination._kernels._surrogates",
        ):
            mod = importlib.import_module(mod_name)
            for attr in [a for a in vars(mod) if a.endswith("_nb")]:
                monkeypatch.setattr(mod, attr, _refuse)
        return
    for mod_name, attr in kernels:
        monkeypatch.setattr(importlib.import_module(mod_name), attr, _refuse)


def test_every_family_has_a_public_entry():
    assert {k.removesuffix("_FB") for k in ENTRIES} == set(nd.FAMILIES)


@pytest.mark.parametrize("family", sorted(ENTRIES))
def test_public_entry_under_switch_never_calls_numba(monkeypatch, no_switches, ghost_model, family):
    from silly_kicks.tracking import _ghost_gk

    entry, kernels = ENTRIES[family]
    monkeypatch.setenv(f"SILLY_KICKS_{family.removesuffix('_FB')}_FORCE_NUMPY", "1")
    monkeypatch.setattr(_ghost_gk, "_HAS_GHOST_NUMBA", None)
    _spy_kernels(monkeypatch, kernels)
    assert entry(ghost_model) is not None  # ran on the numpy reference; a kernel call would raise _KernelCalledError


@pytest.mark.parametrize("family", sorted(ENTRIES))
def test_public_entry_with_numba_hidden_raises(ghost_model, hidden_numba, family):
    entry, _ = ENTRIES[family]
    with pytest.raises(nd.NumbaUnavailableError):
        entry(ghost_model)


#: Kernel modules whose import-time `except ImportError` only records availability; their dispatch raises when the
#: flag is False (coordination `use_numba()`, DAS `compute_das`), so the flag can never become a silent fallback.
_AVAILABILITY_FLAG_MODULES = frozenset({"coordination/_kernels/_numba.py", "tracking/_das_numba.py"})


def _imports_a_kernel(node: ast.AST) -> bool:
    for sub in ast.walk(node):
        if isinstance(sub, ast.Import) and any(
            a.name.split(".")[0] == "numba" or a.name.endswith(("_numba", "_numba_kernels")) for a in sub.names
        ):
            return True
        if isinstance(sub, ast.ImportFrom):
            mod = sub.module or ""
            if (mod.split(".")[0] == "numba" and sub.level == 0) or mod.endswith(("_numba", "_numba_kernels")):
                return True
    return False


def _reraises(handler: ast.ExceptHandler) -> bool:
    """The handler ends in an unconditional raise (it may build the message first)."""
    return bool(handler.body) and isinstance(handler.body[-1], ast.Raise)


def _catches_import_error(handler: ast.ExceptHandler) -> bool:
    """The handler catches an import failure: ImportError / ModuleNotFoundError, or a broad Exception /
    BaseException / bare except: (A impl r7 note)."""
    t = handler.type
    if t is None:
        return True
    names = list(t.elts) if isinstance(t, ast.Tuple) else [t]
    caught = ("ImportError", "ModuleNotFoundError", "Exception", "BaseException")
    return any(isinstance(n, ast.Name) and n.id in caught for n in names)


def test_no_silent_kernel_import_fallback():
    """`try: <import numba or a kernel module> except ImportError: <fallback>` is the F1 pattern. Only a
    handler that RE-RAISES is allowed (plus the two availability-flag kernel modules above)."""
    offenders = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT).as_posix()
        if rel in _AVAILABILITY_FLAG_MODULES:
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Try) or not any(_imports_a_kernel(s) for s in node.body):
                continue
            for h in node.handlers:
                if _catches_import_error(h) and not _reraises(h):
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


def test_fallback_gate_catches_the_old_pattern():
    """The gate is not vacuous: the pre-Change-6 `_ball_carrier` import is flagged."""
    old = ast.parse(
        "try:\n    from ._ball_carrier_numba import _carrier_loop_numba\n    _HAS_NUMBA = True\n"
        "except ImportError:\n    _HAS_NUMBA = False\n"
    )
    tries = [n for n in ast.walk(old) if isinstance(n, ast.Try)]
    assert tries and _imports_a_kernel(tries[0].body[0]) and _catches_import_error(tries[0].handlers[0])
    assert not _reraises(tries[0].handlers[0])
    for clause in ("except:", "except Exception:", "except (OSError, BaseException):"):
        broad = ast.parse(f"try:\n    import numba\n{clause}\n    x = None\n")
        handler = next(n for n in ast.walk(broad) if isinstance(n, ast.Try)).handlers[0]
        assert _catches_import_error(handler) and not _reraises(handler), clause
