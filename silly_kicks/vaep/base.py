"""Implements the VAEP framework.

Attributes
----------
xfns_default : list(callable)
    The default VAEP features.

"""

import math
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import brier_score_loss, roc_auc_score

from silly_kicks.spadl.utils import add_names

from . import features as fs
from . import formula as vaep
from . import labels as lab
from .learners import _LEARNER_REGISTRY

# Type alias for the frames_convention kwarg on compute_features (ADR-006).
_FramesConvention = Literal["absolute_frame", "ltr"]

xfns_default = [
    fs.actiontype_onehot,
    fs.result_onehot,
    fs.actiontype_result_onehot,
    fs.bodypart_onehot,
    fs.time,
    fs.startlocation,
    fs.endlocation,
    fs.startpolar,
    fs.endpolar,
    fs.movement,
    fs.team,
    fs.time_delta,
    fs.space_delta,
    fs.goalscore,
]


class VAEP:
    """
    An implementation of the VAEP framework.

    VAEP (Valuing Actions by Estimating Probabilities) [1]_ defines the
    problem of valuing a soccer player's contributions within a match as
    a binary classification problem and rates actions by estimating its effect
    on the short-term probablities that a team will both score and concede.

    Parameters
    ----------
    xfns : list
        List of feature transformers (see :mod:`silly_kicks.vaep.features`)
        used to describe the game states. Uses :attr:`~silly_kicks.vaep.base.xfns_default`
        if None.
    nb_prev_actions : int, default=3  # noqa: DAR103
        Number of previous actions used to decscribe the game state.


    References
    ----------
    .. [1] Tom Decroos, Lotte Bransen, Jan Van Haaren, and Jesse Davis.
        "Actions speak louder than goals: Valuing player actions in soccer." In
        Proceedings of the 25th ACM SIGKDD International Conference on Knowledge
        Discovery & Data Mining, pp. 1851-1861. 2019.

    Examples
    --------
    Train a VAEP model and rate actions for a single game::

        import pandas as pd
        from silly_kicks.vaep import VAEP

        v = VAEP()
        # Compute features + labels across many games:
        X_list, y_list = [], []
        for _, game in games.iterrows():
            game_actions = actions[actions["game_id"] == game.game_id]
            X_list.append(v.compute_features(game, game_actions))
            y_list.append(v.compute_labels(game, game_actions))
        X, y = pd.concat(X_list), pd.concat(y_list)
        v.fit(X, y, learner="xgboost")

        # Rate one game's actions:
        ratings = v.rate(game, game_actions)
        # ratings has columns: offensive_value / defensive_value / vaep_value
    """

    _add_names = staticmethod(add_names)
    _fs = fs
    _lab = lab
    _vaep = vaep

    def __init__(
        self,
        xfns: list[fs.FeatureTransfomer] | None = None,
        yfns: list[Callable] | None = None,
        nb_prev_actions: int = 3,
    ) -> None:
        self.__models: dict[str, Any] = {}
        self.xfns = list(xfns_default) if xfns is None else xfns
        self.yfns = yfns if yfns is not None else [self._lab.scores, self._lab.concedes]
        self.nb_prev_actions = nb_prev_actions

    def _feature_columns(self) -> list[str]:
        """Return cached feature column names."""
        if not hasattr(self, "_cached_feature_cols"):
            self._cached_feature_cols = self._fs.feature_column_names(self.xfns, self.nb_prev_actions)
        return self._cached_feature_cols

    def compute_features(
        self,
        game: pd.Series,
        game_actions: fs.Actions,
        *,
        frames: pd.DataFrame | None = None,
        frames_convention: "_FramesConvention" = "absolute_frame",
    ) -> pd.DataFrame:
        """
        Transform actions to the feature-based representation of game states.

        Parameters
        ----------
        game : pd.Series
            The SPADL representation of a single game.
        game_actions : pd.DataFrame
            The actions performed during ``game`` in the SPADL representation.
            **Must already be in canonical SPADL LTR convention**, which is what
            every silly-kicks 3.0.0+ converter produces by default. Prior to
            3.0.0 (ADR-006), this method applied an additional
            ``play_left_to_right`` call on top of converter output, which
            silently double-mirrored away-team rows for absolute-frame providers
            (Sportec / Metrica / kloppy gateway / Opta) and accidentally
            cancelled the converter's mirror for possession-perspective providers
            (StatsBomb / Wyscout). The dual-mirror was removed in 3.0.0.
        frames : pd.DataFrame, optional
            Long-form tracking frames matching TRACKING_FRAMES_COLUMNS. Required
            when any xfn in self.xfns is marked frame-aware (via @frame_aware);
            ignored otherwise.
        frames_convention : {"absolute_frame", "ltr"}, default "absolute_frame"
            Coordinate convention of the input ``frames``. Tracking adapters
            ship absolute-frame frames by default (with per-row
            ``team_attacking_direction``); this method normalises them to SPADL
            LTR via :func:`silly_kicks.tracking.utils.play_left_to_right` before
            running frame-aware xfns. Pass ``"ltr"`` to skip normalisation when
            the caller has already done it (e.g. via the tracking adapter's
            ``output_convention="ltr"`` opt-in). Ignored when ``frames is None``.

        Returns
        -------
        features : pd.DataFrame
            Returns the feature-based representation of each game state in the game.

        Raises
        ------
        ValueError
            If self.xfns contains a frame-aware xfn but frames is None, or if
            ``frames_convention`` is not one of the documented values.

        Examples
        --------
        Compute the feature representation for one game (no tracking)::

            X = v.compute_features(game, game_actions)
            # X has one row per game state with the columns specified by ``v.xfns``.

        Compute with tracking-aware xfns appended (e.g. PR-S20 default xfns)::

            from silly_kicks.tracking.features import tracking_default_xfns
            from silly_kicks.vaep.hybrid import HybridVAEP, hybrid_xfns_default
            v = HybridVAEP(xfns=hybrid_xfns_default + tracking_default_xfns)
            X = v.compute_features(game, game_actions, frames=match_frames)
        """
        from .feature_framework import is_frame_aware

        if frames_convention not in ("absolute_frame", "ltr"):
            raise ValueError(
                f"compute_features: frames_convention must be 'absolute_frame' or 'ltr', got {frames_convention!r}"
            )

        game_actions_with_names = self._add_names(game_actions)  # type: ignore
        gamestates = self._fs.gamestates(game_actions_with_names, self.nb_prev_actions)
        # NOTE: pre-3.0.0 silly-kicks applied self._fs.play_left_to_right here
        # on top of converter output. ADR-006 removed it -- converter output is
        # already canonical SPADL LTR, so the second mirror inverted away-team
        # rows. Removed in PR-S22 / 3.0.0.

        if frames is not None and frames_convention == "absolute_frame":
            from silly_kicks.tracking.utils import play_left_to_right as _track_ltr

            frames = _track_ltr(frames, game.home_team_id)

        feats = []
        for fn in self.xfns:
            if is_frame_aware(fn):
                if frames is None:
                    raise ValueError(f"{fn.__name__} requires frames; pass frames= to compute_features")
                # Frame-aware xfns are 2-arg (states, frames) callables, but
                # self.xfns is typed as list[FeatureTransfomer] (1-arg) for
                # backwards compat with non-frame-aware xfns. The is_frame_aware
                # branch above guarantees this xfn was decorated with @frame_aware.
                feats.append(fn(gamestates, frames))  # type: ignore[call-arg]
            else:
                feats.append(fn(gamestates))
        return pd.concat(feats, axis=1)  # type: ignore[reportReturnType]

    def compute_labels(
        self,
        game: pd.Series,
        game_actions: fs.Actions,  # pylint: disable=W0613
    ) -> pd.DataFrame:
        """
        Compute the labels for each game state in the given game.

        Parameters
        ----------
        game : pd.Series
            The SPADL representation of a single game.
        game_actions : pd.DataFrame
            The actions performed during `game` in the SPADL representation.

        Returns
        -------
        labels : pd.DataFrame
            Returns the labels of each game state in the game.

        Examples
        --------
        Compute the label representation (scores / concedes binaries) for one game::

            y = v.compute_labels(game, game_actions)
            # y has columns: scores / concedes (next-N-action lookahead).
        """
        game_actions_with_names = self._add_names(game_actions)  # type: ignore
        return pd.concat([fn(game_actions_with_names) for fn in self.yfns], axis=1)  # type: ignore[reportReturnType]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        learner: str = "xgboost",
        val_size: float = 0.25,
        tree_params: dict[str, Any] | None = None,
        fit_params: dict[str, Any] | None = None,
        random_state: int | None = None,
    ) -> "VAEP":
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : pd.DataFrame
            Feature representation of the game states.
        y : pd.DataFrame
            Scoring and conceding labels for each game state.
        learner : string, default='xgboost'  # noqa: DAR103
            Gradient boosting implementation which should be used to learn the
            model. The supported learners are 'xgboost', 'catboost' and 'lightgbm'.
        val_size : float, default=0.25  # noqa: DAR103
            Percentage of the dataset that will be used as the validation set
            for early stopping. When zero, no validation data will be used.
        tree_params : dict
            Parameters passed to the constructor of the learner.
        fit_params : dict
            Parameters passed to the fit method of the learner.

        Raises
        ------
        ValueError
            If one of the features is missing in the provided dataframe.

        Returns
        -------
        self
            Fitted VAEP model.

        Examples
        --------
        Fit a VAEP model with xgboost (default) on accumulated features + labels::

            v.fit(X, y, learner="xgboost", val_size=0.25)
            # ``random_state`` controls the train/val split deterministically.
        """
        nb_states = len(X)
        rng = np.random.default_rng(random_state)
        idx = rng.permutation(nb_states)
        # fmt: off
        train_idx = idx[:math.floor(nb_states * (1 - val_size))]
        val_idx = idx[(math.floor(nb_states * (1 - val_size)) + 1):]
        # fmt: on

        # filter feature columns
        cols = self._feature_columns()
        if not set(cols).issubset(set(X.columns)):
            missing_cols = " and ".join(set(cols).difference(X.columns))
            raise ValueError(f"{missing_cols} are not available in the features dataframe")

        # split train and validation data
        X_train, y_train = X.iloc[train_idx][cols], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx][cols], y.iloc[val_idx]

        # train classifiers F(X) = Y
        fit_fn = _LEARNER_REGISTRY.get(learner)
        if fit_fn is None:
            raise ValueError(f"Unsupported learner: {learner!r}. Available: {list(_LEARNER_REGISTRY)}")
        for col in list(y.columns):
            eval_set = [(X_val, y_val[col])] if val_size > 0 else None
            self.__models[col] = fit_fn(X_train, y_train[col], eval_set, tree_params, fit_params)
        return self

    def _estimate_probabilities(self, X: pd.DataFrame) -> pd.DataFrame:
        # filter feature columns
        cols = self._feature_columns()
        if not set(cols).issubset(set(X.columns)):
            missing_cols = " and ".join(set(cols).difference(X.columns))
            raise ValueError(f"{missing_cols} are not available in the features dataframe")

        Y_hat = pd.DataFrame()
        for col in self.__models:
            Y_hat[col] = self.__models[col].predict_proba(X[cols])[:, 1]
        return Y_hat

    def rate(
        self,
        game: pd.Series,
        game_actions: fs.Actions,
        game_states: fs.Features | None = None,
        *,
        frames: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Compute the VAEP rating for the given game states.

        Parameters
        ----------
        game : pd.Series
            The SPADL representation of a single game.
        game_actions : pd.DataFrame
            The actions performed during `game` in the SPADL representation.
        game_states : pd.DataFrame, default=None
            DataFrame with the game state representation of each action. If
            `None`, these will be computed on-the-fly.
        frames : pd.DataFrame, optional
            When ``game_states`` is None and self.xfns contains frame-aware xfns,
            frames must be supplied; passed through to compute_features.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        Returns
        -------
        ratings : pd.DataFrame
            Returns the VAEP rating for each given action, as well as the
            offensive and defensive value of each action.

        Examples
        --------
        Rate one game's actions after fitting (no tracking)::

            ratings = v.rate(game, game_actions)
            ratings[["offensive_value", "defensive_value", "vaep_value"]].head()

        Rate with tracking frames (when self.xfns includes frame-aware xfns)::

            ratings = v.rate(game, game_actions, frames=match_frames)
        """
        if not self.__models:
            raise NotFittedError()

        game_actions_with_names = self._add_names(game_actions)  # type: ignore
        if game_states is None:
            game_states = self.compute_features(game, game_actions, frames=frames)

        y_hat = self._estimate_probabilities(game_states)
        p_scores, p_concedes = y_hat.iloc[:, 0], y_hat.iloc[:, 1]
        vaep_values = self._vaep.value(game_actions_with_names, p_scores, p_concedes)
        return vaep_values

    def score(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Evaluate the fit of the model on the given test data and labels.

        Parameters
        ----------
        X : pd.DataFrame
            Feature representation of the game states.
        y : pd.DataFrame
            Scoring and conceding labels for each game state.

        Raises
        ------
        NotFittedError
            If the model is not fitted yet.

        Returns
        -------
        score : dict
            The Brier and AUROC scores for both binary classification problems.

        Examples
        --------
        Evaluate fit quality on held-out data::

            metrics = v.score(X_test, y_test)
            # metrics["scores"]["brier"] / metrics["scores"]["auroc"], same for "concedes".
        """
        if not self.__models:
            raise NotFittedError()

        y_hat = self._estimate_probabilities(X)

        scores: dict[str, dict[str, float]] = {}
        for col in self.__models:
            scores[col] = {}
            scores[col]["brier"] = brier_score_loss(y[col], y_hat[col])
            scores[col]["auroc"] = roc_auc_score(y[col], y_hat[col])  # type: ignore[reportArgumentType]

        return scores
