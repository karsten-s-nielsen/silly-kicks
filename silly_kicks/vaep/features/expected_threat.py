"""xT-as-a-VAEP-feature transformer factory (``xt__<method>``).

Wraps a *fitted, caller-supplied* ``ExpectedThreat`` (see NOTICE for citations) as a
frame-free VAEP feature. Train/serve consistency is the caller's responsibility: fit +
freeze the grid on the VAEP training corpus (or a disjoint exogenous corpus) and reuse
the identical object at serve time (mirrors FrozenXt / ADR-009). NaN for non-move /
failed-move actions, matching ``ExpectedThreat.rate``. Opt-in: NOT in any default xfn
list -- adding it to a caller's xfns is a deliberate, self-triggered VAEP retrain.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from silly_kicks.vaep.feature_framework import Actions, Features, FeatureTransfomer, simple

if TYPE_CHECKING:  # ExpectedThreat is only duck-typed at runtime (model.xT/.method/.rate)
    from silly_kicks.xthreat import ExpectedThreat

__all__ = ["xt_xfns"]


def xt_xfns(*, model: ExpectedThreat | str | None = None) -> list[FeatureTransfomer]:
    """Factory: one frame-free transformer emitting ``xt__<model.method>_a{0,1,2}``.

    Parameters
    ----------
    model : ExpectedThreat
        A fitted xT model. ``str`` (a future bundled variant name) and ``None`` raise.

    Returns
    -------
    list[FeatureTransfomer]
        A one-element list holding the transformer.

    Raises
    ------
    ValueError, NotImplementedError, NotFittedError
        See :func:`silly_kicks.xthreat.require_fitted_xt`.

    Examples
    --------
    Opt in to xT as a VAEP feature::

        from silly_kicks.vaep import VAEP, features as fs
        from silly_kicks.vaep.features import xt_xfns

        v = VAEP(xfns=fs.xfns_default + xt_xfns(model=frozen_xt))
    """
    # Lazy import (ADR-041): a MODULE-level `from silly_kicks.xthreat import ...` closes a
    # real cycle -- xthreat/_grid imports spadl.config, and spadl/__init__ imports
    # tracking, which imports vaep.feature_framework, which re-enters this module while
    # xthreat is still partially initialized. Function-local keeps the single-sourced
    # guard without the edge (same idiom as tracking/_xt_gk.py's frozen cell-indexer).
    from silly_kicks.xthreat import require_fitted_xt

    require_fitted_xt(model, caller="xt_xfns")
    col = f"xt__{model.method}"  # type: ignore[union-attr]

    def _xt(actions: Actions) -> Features:
        return pd.DataFrame({col: model.rate(actions)}, index=actions.index)  # type: ignore[union-attr]

    transformer = simple(_xt)
    transformer.__name__ = col
    return [transformer]
