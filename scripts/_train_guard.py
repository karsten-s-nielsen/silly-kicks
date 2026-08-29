"""Fail-loud fit-time guard: refuse to fit a BUNDLED model under an unsupported scikit-learn.

VSSOT-IMPL-03. The ``[train]`` extra pins ``scikit-learn>=1.9,<2`` because HistGradientBoosting
produces different trees across sklearn versions (measured on the ghost re-fit: same
corpus/commit/pandas under 1.7.2 vs 1.9.0 -> different weights, different sha256). That floor is
marker-gated to ``python_version >= '3.11'`` (sklearn 1.9 dropped Python 3.10), so on py3.10
``pip install .[train]`` silently resolves an older sklearn. Without this guard a maintainer could
fit and SHIP ghost weights on py3.10/sklearn<=1.7.2 that silently disagree with the bundled
artifacts -- guarded only by a comment. This raises instead.

Called at the TRAINER entry point (``scripts/train_ghost_gk.py`` ``main()``), never in
``GhostGkModel.fit()``: the non-slow library unit tests fit toy models on whatever sklearn the CI leg
resolved, and a library-level raise would redden the py3.10 legs. The range mirrors
``pyproject.toml``'s ``[train]`` ``scikit-learn>=1.9,<2`` -- keep them in lockstep.

Only the ghost trainer needs this: the logistic bundles (gk_completion / gk_retention / receiver) fit
a convex problem whose solution is version-stable, and their own load-time checks compare the MAJOR
version only, so 1.7 vs 1.9 is a non-issue for them.
"""

from __future__ import annotations

_MIN: tuple[int, int] = (1, 9)  # inclusive floor; matches pyproject [train] `scikit-learn>=1.9`
_MAX_MAJOR: int = 2  # exclusive ceiling; matches `,<2`


def _major_minor(version: str) -> tuple[int, int]:
    parts = version.split(".")
    return int(parts[0]), (int(parts[1]) if len(parts) > 1 else 0)


def _in_supported_range(version: str) -> bool:
    major, minor = _major_minor(version)
    return (major, minor) >= _MIN and major < _MAX_MAJOR


def sklearn_supports_training() -> bool:
    """True iff the installed scikit-learn is in the supported fit range ``[1.9, 2)``.

    The read-only companion of :func:`require_training_sklearn`, for a test to SKIP the ghost-trainer
    smokes (which fit real models) when the environment's sklearn is out of range -- so they run and
    pass where sklearn is supported (CI's 3.12 primary leg) and skip, rather than fail, on a stale
    dev env (e.g. sklearn 1.8.0) or a future 2.0. On CI's primary leg this is always True, so it never
    reduces real coverage.
    """
    import sklearn

    return _in_supported_range(sklearn.__version__)


def require_training_sklearn() -> str:
    """Return the installed scikit-learn version, or raise if it is outside the supported fit range.

    Raises ``RuntimeError`` with an actionable message when ``sklearn.__version__`` is not in
    ``[1.9, 2)``. Bundled ghost weights are fit on that range; fitting outside it produces trees that
    silently disagree with what ships. Install ``silly-kicks[train]`` on Python>=3.11.
    """
    import sklearn

    version = sklearn.__version__
    if not _in_supported_range(version):
        raise RuntimeError(
            f"scikit-learn {version} is outside the supported training range "
            f">={_MIN[0]}.{_MIN[1]},<{_MAX_MAJOR} -- refusing to fit a bundled artifact. Bundled ghost "
            f"weights are fit on this range (HistGradientBoosting trees differ across sklearn "
            f"versions), so weights fit outside it silently disagree with what ships. Install "
            f"`silly-kicks[train]` on Python>=3.11."
        )
    return version
