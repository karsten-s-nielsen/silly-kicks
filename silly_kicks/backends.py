"""Which compute backend each accelerated kernel family uses: the public face of the numba dispatch.

silly-kicks runs each accelerated kernel family (ghost GK, DAS, coordination, pitch control, ball carrier, xT-GK) on
its numba kernel. A family runs its numpy reference kernel only when ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1`` asks for
it, and a missing numba raises :class:`NumbaUnavailableError` rather than falling back. This module lets a consumer
ask which backend a family will use, without importing the private dispatch module
(spec 2026-11-08 tracking-batch section 4.1).

- :func:`active_backend` has the same rules as the dispatch: it raises when numba is missing for an unforced
  family, and for an unknown family.
- :func:`backend_report` is a diagnostic for log lines and environment guards: it never raises for a missing numba.
"""

from __future__ import annotations

from typing import Literal

from silly_kicks import _numba_dispatch
from silly_kicks._numba_dispatch import FAMILIES, NumbaUnavailableError

__all__ = ["FAMILIES", "NumbaUnavailableError", "active_backend", "backend_report"]


def active_backend(family: str) -> Literal["numba", "numpy"]:
    """The backend that ``family``'s kernels run on: ``"numba"``, or ``"numpy"`` when forced.

    Raises ``ValueError`` for a family not in :data:`FAMILIES`, and :class:`NumbaUnavailableError` when numba
    cannot be imported and the family is not forced to numpy (the same rules as the kernels themselves).

    Examples
    --------
    >>> import os
    >>> from unittest import mock
    >>> from silly_kicks.backends import active_backend
    >>> active_backend("DAS")
    'numba'
    >>> with mock.patch.dict(os.environ, {"SILLY_KICKS_DAS_FORCE_NUMPY": "1"}):  # restored on exit
    ...     active_backend("DAS")
    'numpy'
    """
    return "numba" if _numba_dispatch.use_numba(family) else "numpy"


def backend_report() -> dict[str, str]:
    """Each family's backend, plus ``"numba_version"``, for a log line or an environment guard. Never raises for a
    missing numba.

    Values: ``"numba"``; ``"numpy"`` (forced by ``SILLY_KICKS_<FAMILY>_FORCE_NUMPY=1``); ``"unavailable"`` (numba
    cannot be imported, so the family's kernels would raise). ``"numba_version"`` is the version string or
    ``"unavailable"``. An error other than a missing numba (for example a numba cache failure) propagates.

    Examples
    --------
    >>> from silly_kicks.backends import FAMILIES, backend_report
    >>> report = backend_report()
    >>> sorted(report) == sorted([*FAMILIES, "numba_version"])
    True
    >>> report["PITCH_CONTROL"]
    'numba'
    """
    try:
        version: str | None = str(_numba_dispatch._import_numba().__version__)
    except NumbaUnavailableError:
        version = None
    report: dict[str, str] = {}
    for family in FAMILIES:
        if _numba_dispatch._forced_numpy(family):  # the dispatch's own rule (review A-04)
            report[family] = "numpy"
        else:
            report[family] = "numba" if version is not None else "unavailable"
    report["numba_version"] = version if version is not None else "unavailable"
    return report
