"""Rung-3a cross-team dependence: the Dixon-Coles low-score correction (TF-53 spec §4/§5).

The Dixon & Coles (1997) tau reweights the four low-score joint cells (0-0, 0-1, 1-0, 1-1) of two
otherwise-independent marginals, with a single fitted correlation ``rho``. tau was derived for Poisson
marginals; here it is applied to the Poisson-binomial marginals using each team's mean goals as lambda
/ mu (the standard practitioner extension), and rho is fit empirically (scripts/train_...). ``rho`` is
served from a bundled JSON artifact with a fail-closed load (SHA-256 + plausible range + training_commit).
"""

from __future__ import annotations

import functools
import hashlib
import json
import pathlib
from dataclasses import dataclass

import numpy as np

from ._config import MatchOutcomeParams

_WEIGHTS_DIR = pathlib.Path(__file__).resolve().parent / "weights"
_RHO_ABS_MAX = 1.0  # plausible range; a fitted DC rho is small, but bound generously and fail-closed


class MatchOutcomeIntegrityError(RuntimeError):
    """A bundled dependence artifact failed its fail-closed load (SHA / range / provenance).

    Examples
    --------
    Raised by :meth:`DependenceModel.load` / :meth:`DependenceModel.bundled` on a bad artifact::

        from silly_kicks.match_outcome import DependenceModel, MatchOutcomeIntegrityError

        try:
            model = DependenceModel.bundled()
        except MatchOutcomeIntegrityError:
            ...  # no bundled rho, or a tampered / out-of-range artifact -- fail closed, never silent
    """


def dixon_coles_tau(i: int, j: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score tau; 1.0 outside the four low cells.

    Examples
    --------
    >>> from silly_kicks.match_outcome import dixon_coles_tau
    >>> dixon_coles_tau(1, 1, 1.5, 1.2, 0.0)  # rho=0 -> no correction anywhere
    1.0
    >>> dixon_coles_tau(2, 0, 1.5, 1.2, 0.1)  # outside the low block
    1.0
    """
    if i == 0 and j == 0:
        return 1.0 - lam * mu * rho
    if i == 0 and j == 1:
        return 1.0 + lam * rho
    if i == 1 and j == 0:
        return 1.0 + mu * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def apply_dependence(home_pmf: np.ndarray, away_pmf: np.ndarray, *, rho: float) -> np.ndarray:
    """Joint scoreline with the Dixon-Coles low-score correction; renormalized to sum 1.

    ``rho == 0`` returns the independent outer product (to floating tolerance).

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks.match_outcome import apply_dependence, goal_count_pmf
    >>> home, away = goal_count_pmf([0.5, 0.3]), goal_count_pmf([0.4])
    >>> joint = apply_dependence(home, away, rho=0.0)  # rho=0 -> independent
    >>> bool(np.allclose(joint, np.outer(home, away)))
    True
    """
    home = np.asarray(home_pmf, dtype="float64")
    away = np.asarray(away_pmf, dtype="float64")
    joint = np.outer(home, away)
    lam = float((np.arange(home.shape[0]) * home).sum())
    mu = float((np.arange(away.shape[0]) * away).sum())
    for i in (0, 1):
        for j in (0, 1):
            if i < joint.shape[0] and j < joint.shape[1]:
                joint[i, j] *= dixon_coles_tau(i, j, lam, mu, rho)
    joint = np.clip(joint, 0.0, None)  # extreme rho can push a low cell negative -> clip then renormalize
    total = joint.sum()
    return joint / total if total > 0 else joint


@dataclass(frozen=True)
class DependenceModel:
    """A fitted Dixon-Coles rho served from a pickle-free JSON artifact (fail-closed load).

    Examples
    --------
    >>> import numpy as np
    >>> from silly_kicks.match_outcome import DependenceModel, goal_count_pmf
    >>> model = DependenceModel(rho=0.05, training_commit="abc1234")
    >>> joint = model.apply(goal_count_pmf([0.5, 0.3]), goal_count_pmf([0.4]))
    >>> bool(abs(joint.sum() - 1.0) < 1e-12)
    True
    """

    rho: float
    training_commit: str

    @classmethod
    def load(cls, weights_dir: pathlib.Path) -> DependenceModel:
        """Load + verify a bundled artifact; raise :class:`MatchOutcomeIntegrityError` on any failure.

        Examples
        --------
        Load a fitted rho from an on-disk ``weights/`` directory (``model.json`` + ``SHA256SUMS``)::

            import pathlib
            from silly_kicks.match_outcome import DependenceModel

            model = DependenceModel.load(pathlib.Path("silly_kicks/match_outcome/weights"))
            model.rho  # the fitted Dixon-Coles correlation
        """
        model_path = weights_dir / "model.json"
        sums_path = weights_dir / "SHA256SUMS"
        if not model_path.exists() or not sums_path.exists():
            raise MatchOutcomeIntegrityError(f"no dependence artifact under {weights_dir}")
        actual = hashlib.sha256(model_path.read_bytes()).hexdigest()
        expected = _sha_for("model.json", sums_path)
        if expected is None or actual != expected:
            raise MatchOutcomeIntegrityError("model.json SHA-256 does not match SHA256SUMS")
        data = json.loads(model_path.read_text(encoding="utf-8"))
        rho = float(data.get("rho", float("nan")))
        if not np.isfinite(rho) or abs(rho) >= _RHO_ABS_MAX:
            raise MatchOutcomeIntegrityError(f"rho {rho!r} out of plausible range (|rho| < {_RHO_ABS_MAX})")
        tc = data.get("training_commit")
        if not tc:
            raise MatchOutcomeIntegrityError("artifact missing training_commit provenance")
        return cls(rho=rho, training_commit=str(tc))

    @classmethod
    def bundled(cls) -> DependenceModel:
        """Serve the bundled rho (``weights/``); raises fail-closed if absent (never silent).

        Examples
        --------
        Serve the package's bundled fit (raises :class:`MatchOutcomeIntegrityError` if not present)::

            from silly_kicks.match_outcome import DependenceModel

            rho = DependenceModel.bundled().rho  # the shipped Dixon-Coles correlation
        """
        return _load_bundled()  # cached: read + SHA the artifact once, not per team per game

    def apply(self, home_pmf: np.ndarray, away_pmf: np.ndarray) -> np.ndarray:
        """Apply this model's rho to two goal PMFs (the :func:`apply_dependence` joint).

        Examples
        --------
        >>> import numpy as np
        >>> from silly_kicks.match_outcome import DependenceModel, goal_count_pmf
        >>> joint = DependenceModel(rho=0.05, training_commit="abc1234").apply(
        ...     goal_count_pmf([0.6, 0.4]), goal_count_pmf([0.3])
        ... )
        >>> bool(abs(joint.sum() - 1.0) < 1e-12)
        True
        """
        return apply_dependence(home_pmf, away_pmf, rho=self.rho)


def _sha_for(name: str, sums_path: pathlib.Path) -> str | None:
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == name:
            return parts[0]
    return None


def resolve_rho(params: MatchOutcomeParams) -> float:
    """The rho for a ``team_dependence='dixon_coles'`` run -- the bundled fit (fail-closed).

    Examples
    --------
    Resolve the rho a ``dixon_coles`` run will use (raises fail-closed if no artifact is bundled)::

        from silly_kicks.match_outcome import MatchOutcomeParams
        from silly_kicks.match_outcome._dependence import resolve_rho

        rho = resolve_rho(MatchOutcomeParams(team_dependence="dixon_coles"))
    """
    return DependenceModel.bundled().rho


@functools.cache
def _load_bundled() -> DependenceModel:
    """Load the bundled artifact ONCE (read + SHA), then serve from cache.

    ``functools.cache`` does not cache exceptions, so a fail-closed absence (no ``weights/``) re-raises
    on every call -- the fail-closed contract is preserved. Cleared implicitly per process; bundled
    weights are immutable within a run.
    """
    return DependenceModel.load(_WEIGHTS_DIR)
