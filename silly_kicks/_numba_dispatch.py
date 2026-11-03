"""The one numba dispatch point for every accelerated kernel family (HFU spec §13.2, ADR-076 amendment 2026-11-06).

numba is a core dependency. A family's numpy reference kernel runs only when its explicit switch
``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1`` asks for it (the parity tests use the switches). Otherwise the numba kernel
runs, and a missing numba raises :class:`NumbaUnavailableError`. It never falls back silently: the old optional
extra hid a 12x slowdown in HF training jobs twice, because ``[train]`` never carried numba.

numba is imported lazily, on the first accelerated call, so ``import silly_kicks`` stays light. Any error other
than ``ImportError`` (for example the numba cache-locator ``RuntimeError``) propagates unchanged.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

__all__ = ["FAMILIES", "NumbaUnavailableError", "import_kernels", "numba_version", "require_numba", "use_numba"]

#: The kernel families (the 2026-11-06 census). Each has one switch, ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY``.
FAMILIES: tuple[str, ...] = ("GHOST", "DAS", "COORDINATION", "PITCH_CONTROL", "BALL_CARRIER", "XTGK")

_NUMBA: ModuleType | None = None  # the imported numba module, once probed


class NumbaUnavailableError(ImportError):
    """numba is a core dependency of silly-kicks but cannot be imported: the installation is broken."""


def _import_numba() -> ModuleType:
    global _NUMBA
    if _NUMBA is None:
        try:
            import numba
        except ImportError as exc:
            raise NumbaUnavailableError(
                "numba is a required dependency of silly-kicks but cannot be imported "
                f"({exc}). Repair the environment: pip install --force-reinstall silly-kicks"
            ) from exc
        _NUMBA = numba
    return _NUMBA


def use_numba(family: str) -> bool:
    """True when the family's numba kernel must run; False only under ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1``.

    Raises :class:`NumbaUnavailableError` when numba cannot be imported, and ``ValueError`` for an unknown family.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown numba kernel family {family!r}; expected one of {FAMILIES}")
    if os.environ.get(f"SILLY_KICKS_{family}_FORCE_NUMPY") == "1":
        return False
    _import_numba()
    return True


def require_numba() -> None:
    """For an explicit numba request (``engine="numba"``, ``kde_backend="cpu-numba"``): import numba or raise
    :class:`NumbaUnavailableError`. The explicit argument is the more specific request, so the switches are ignored.
    """
    _import_numba()


def import_kernels(module: str) -> ModuleType:
    """Import a numba kernel module (absolute dotted name) after the dispatch chose numba.

    The kernel modules raise ``ImportError`` at import when numba is missing; that becomes
    :class:`NumbaUnavailableError` here, so no caller can turn it into a silent numpy fallback.
    """
    try:
        return importlib.import_module(module)
    except NumbaUnavailableError:
        raise
    except ImportError as exc:
        raise NumbaUnavailableError(
            f"numba kernel module {module} cannot be imported ({exc}). numba is a required dependency of "
            "silly-kicks. Repair the environment: pip install --force-reinstall silly-kicks"
        ) from exc


def numba_version() -> str:
    """numba's version for log lines, or ``"absent"``. Never raises."""
    try:
        return str(_import_numba().__version__)
    except ImportError:
        return "absent"
