"""Standalone undiscounted value iteration for the xT fixed point.

Extracted byte-identically from the legacy ExpectedThreat.__solve: raw-diff convergence
(NOT abs) + per-iteration heatmaps. Correct because iteration starts at xT=0 under a
monotone non-negative operator (gs, p_move, T >= 0) -> iterates increase from below ->
raw-diff == abs-diff. Do NOT "fix" the stop condition. See ADR-021.
"""

import numpy as np
import numpy.typing as npt


def value_iteration(
    p_scoring: npt.NDArray[np.float64],
    p_shot: npt.NDArray[np.float64],
    p_move: npt.NDArray[np.float64],
    transition: npt.NDArray[np.float64],
    *,
    eps: float = 1e-5,
    max_iter: int | None = None,
) -> tuple[npt.NDArray[np.float64], list[npt.NDArray[np.float64]]]:
    """Solve xT by value iteration.

    ``max_iter=None`` (default) reproduces the legacy unbounded loop exactly; a non-None bound
    is an opt-in safety cap for direct callers on arbitrary (e.g. degenerate) matrices.

    Returns
    -------
    (xT, heatmaps)
        The final ``(w, l)`` value surface and the per-iteration snapshots (heatmaps[0] is the
        initial zeros). The heatmaps list backs ``ExpectedThreat.heatmaps``.

    Examples
    --------
    Solve xT from precomputed probability surfaces::

        from silly_kicks.xthreat import value_iteration

        xT, heatmaps = value_iteration(p_scoring, p_shot, p_move, transition, eps=1e-5)
    """
    w, l = p_scoring.shape
    gs = p_scoring * p_shot
    xT = np.zeros((w, l), dtype=np.float64)
    heatmaps: list[npt.NDArray[np.float64]] = [xT.copy()]
    diff = np.ones((w, l), dtype=np.float64)
    it = 0
    while np.any(diff > eps):
        if max_iter is not None and it >= max_iter:
            break
        total_payoff = (transition @ xT.ravel()).reshape(w, l)
        newxT = gs + (p_move * total_payoff)
        diff = newxT - xT
        xT = newxT
        heatmaps.append(xT.copy())
        it += 1
    return xT, heatmaps
