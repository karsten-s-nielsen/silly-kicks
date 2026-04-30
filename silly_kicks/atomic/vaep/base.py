"""Implements the Atomic-VAEP framework.

Attributes
----------
xfns_default : list(callable)
    The default VAEP features.

"""

from collections.abc import Callable

from silly_kicks.atomic.spadl.utils import add_names
from silly_kicks.vaep.base import VAEP

from . import features as fs
from . import formula as vaep
from . import labels as lab

xfns_default = [
    fs.actiontype,
    fs.actiontype_onehot,
    fs.bodypart,
    fs.bodypart_onehot,
    fs.time,
    fs.team,
    fs.time_delta,
    fs.location,
    fs.polar,
    fs.movement_polar,
    fs.direction,
    fs.goalscore,
]


class AtomicVAEP(VAEP):
    """
    An implementation of the VAEP framework for atomic actions.

    In contrast to the original VAEP framework [1]_ this extension
    distinguishes the contribution of the player who initiates the action
    (e.g., gives the pass) and the player who completes the action (e.g.,
    receives the pass) [2]_.

    Parameters
    ----------
    xfns : list
        List of feature transformers (see :mod:`silly_kicks.atomic.vaep.features`)
        used to describe the game states. Uses :attr:`~silly_kicks.vaep.base.xfns_default`
        if None.
    nb_prev_actions : int, default=3
        Number of previous actions used to decscribe the game state.

    See Also
    --------
    :class:`silly_kicks.vaep.VAEP` : Implementation of the original VAEP framework.

    References
    ----------
    .. [1] Tom Decroos, Lotte Bransen, Jan Van Haaren, and Jesse Davis.
        "Actions speak louder than goals: Valuing player actions in soccer." In
        Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
        Discovery & Data Mining, pp. 1851-1861. 2019.
    .. [2] Tom Decroos, Pieter Robberechts and Jesse Davis.
        "Introducing Atomic-SPADL: A New Way to Represent Event Stream Data".
        DTAI Sports Analytics Blog.
        https://dtai.cs.kuleuven.be/sports/blog/introducing-atomic-spadl:-a-new-way-to-represent-event-stream-data
        May 2020.

    Examples
    --------
    Train an AtomicVAEP model on an atomic-SPADL stream::

        import pandas as pd
        from silly_kicks.atomic.vaep import AtomicVAEP

        v = AtomicVAEP()
        # Compute features + labels per game on the atomic stream:
        X_list, y_list = [], []
        for _, game in games.iterrows():
            game_atomic = atomic[atomic["game_id"] == game.game_id]
            X_list.append(v.compute_features(game, game_atomic))
            y_list.append(v.compute_labels(game, game_atomic))
        X, y = pd.concat(X_list), pd.concat(y_list)
        v.fit(X, y)
        ratings = v.rate(game, game_atomic)
    """

    _add_names = staticmethod(add_names)
    _lab = lab
    _fs = fs
    _vaep = vaep

    def __init__(
        self,
        xfns: list[fs.FeatureTransfomer] | None = None,
        yfns: list[Callable] | None = None,
        nb_prev_actions: int = 3,
    ) -> None:
        xfns = list(xfns_default) if xfns is None else xfns
        super().__init__(xfns, yfns, nb_prev_actions)
