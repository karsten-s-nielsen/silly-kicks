"""``WinProbabilityModel`` -- interval-hazard GLM + self-contained Markov-chain serve.

The hazard is a logistic on ``(score_diff, minutes_remaining, base_strength, home, man_advantage)``
fitted per-minute-interval-goal indicator (sklearn at fit-time ONLY); the serve is pure-numpy
``sigmoid(x @ beta + b)`` fed into the ``_chain`` forward Markov chain. Pickle-free JSON + SHA256,
fail-closed ``load`` (SHA -> feature-contract probe). Chirality (ADR-011) is N/A: this model has no
geometric/coordinate features, so there is no mirror symmetry to verify -- the behavioral load guard is
the feature-contract probe.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.id_compat import same_id

from ._chain import outcome_table
from ._config import WinProbabilityParams
from ._state import _FOUL, _OWNGOAL, _PERIOD_OFFSET_MIN, _RED, _SHOT_TYPE_IDS, _SUCCESS

_HAZARD_FEATURES = ("score_diff", "minutes_remaining", "base_strength", "home", "man_advantage")

# Fixed probe states for the feature-contract fingerprint (order = _HAZARD_FEATURES).
_PROBE_STATES: tuple[tuple[float, float, float, int, int], ...] = (
    (0, 90.0, 0.0, 1, 0),
    (1, 15.0, 0.3, 0, -1),
    (-2, 45.0, -0.5, 1, 1),
    (3, 5.0, 0.0, 0, 0),
)
_WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
_MIN_MATCHES_FOR_ISOTONIC = 10


class WinProbabilityIntegrityError(RuntimeError):
    """Raised when a loaded artifact fails integrity (SHA) or the feature-contract probe.

    Examples
    --------
    >>> from silly_kicks.win_probability import WinProbabilityIntegrityError
    >>> issubclass(WinProbabilityIntegrityError, RuntimeError)
    True
    """


def _team_goal_minutes(g: pd.DataFrame, teams: list) -> dict:
    """Absolute-minute goal timeline per team (id-based, ADR-018 own-goal-by-result)."""
    out: dict = {t: [] for t in teams}
    is_scored = (g["type_id"].isin(_SHOT_TYPE_IDS) & (g["result_id"] == _SUCCESS)).to_numpy()
    is_og = (g["result_id"] == _OWNGOAL).to_numpy()
    team_col = g["team_id"].to_numpy()
    periods = g["period_id"].to_numpy()
    times = g["time_seconds"].to_numpy()
    for i in range(len(g)):
        minute = _PERIOD_OFFSET_MIN.get(int(periods[i]), 0) + float(times[i]) / 60.0
        t = team_col[i]
        if bool(is_scored[i]):
            out.setdefault(t, []).append(minute)
        elif bool(is_og[i]):
            opp = [x for x in teams if not same_id(x, t)]
            if len(opp) == 1:
                out.setdefault(opp[0], []).append(minute)
    return out


def _team_red_minutes(g: pd.DataFrame, teams: list) -> dict:
    out: dict = {t: [] for t in teams}
    reds = ((g["type_id"] == _FOUL) & (g["result_id"] == _RED)).to_numpy()
    team_col = g["team_id"].to_numpy()
    periods = g["period_id"].to_numpy()
    times = g["time_seconds"].to_numpy()
    for i in range(len(g)):
        if bool(reds[i]):
            minute = _PERIOD_OFFSET_MIN.get(int(periods[i]), 0) + float(times[i]) / 60.0
            out.setdefault(team_col[i], []).append(minute)
    return out


def _build_interval_training(actions, games, strength_column, params):
    """Per-(match, minute-interval, team) rows: X (n,5), y (n,), groups (game ids for OOF)."""
    from ._state import _home_team_map

    home_map = _home_team_map(actions, games)
    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    groups: list = []
    for game_id, g in actions.groupby("game_id", sort=False):
        home_id = home_map.get(game_id)
        teams = list(pd.unique(g["team_id"].dropna()))
        if len(teams) != 2:
            continue
        goal_min = _team_goal_minutes(g, teams)
        red_min = _team_red_minutes(g, teams)
        last_period = int(g["period_id"].max())
        last_time = float(g.loc[g["period_id"] == last_period, "time_seconds"].max())
        final_min = max(_PERIOD_OFFSET_MIN.get(last_period, 90) + last_time / 60.0, params.regulation_minutes)
        strength = {t: 0.0 for t in teams}
        if strength_column is not None and strength_column in g.columns:
            for t in teams:
                vals = g.loc[g["team_id"] == t, strength_column].dropna()
                strength[t] = float(vals.iloc[0]) if len(vals) else 0.0
        step = params.interval_minutes
        n_int = int(np.ceil(final_min / step))
        for k in range(n_int):
            m0 = k * step
            for t in teams:
                opp = next(x for x in teams if not same_id(x, t))
                own_before = sum(1 for mm in goal_min.get(t, []) if mm < m0)
                opp_before = sum(1 for mm in goal_min.get(opp, []) if mm < m0)
                own_red = sum(1 for mm in red_min.get(t, []) if mm < m0)
                opp_red = sum(1 for mm in red_min.get(opp, []) if mm < m0)
                scored = any(m0 <= mm < m0 + step for mm in goal_min.get(t, []))
                rows_x.append(
                    [
                        own_before - opp_before,
                        final_min - m0,
                        strength[t],
                        1.0 if (home_id is not None and same_id(t, home_id)) else 0.0,
                        opp_red - own_red,
                    ]
                )
                rows_y.append(1 if scored else 0)
                groups.append(game_id)
    return np.asarray(rows_x, dtype="float64"), np.asarray(rows_y, dtype="int64"), np.asarray(groups, dtype=object)


class WinProbabilityModel:
    """In-game win-probability model: interval-hazard GLM + Markov-chain serve.

    Examples
    --------
    Fit on a corpus of SPADL actions (a real ``actions`` frame + a ``(game_id, home_team_id)`` frame
    are required, so this is an illustrative block), then serve a game state::

        from silly_kicks.win_probability import WinProbabilityModel

        model = WinProbabilityModel().fit(actions, games=games)
        p_win, p_draw, p_loss = model.predict_outcome(score_diff=0, minutes_remaining=45.0, home=True)
    """

    def __init__(self, *, params: WinProbabilityParams | None = None) -> None:
        self.params = params or WinProbabilityParams.default()
        self._beta: np.ndarray | None = None
        self._intercept: float | None = None
        self._isotonic: tuple[np.ndarray, np.ndarray] | None = None  # (x, y) breakpoints
        self._fitted = False

    # ---- hazard (pure-numpy serve) ----
    def _feature_row(self, *, score_diff, minutes_remaining, base_strength, home, man_advantage) -> np.ndarray:
        return np.array(
            [
                float(score_diff),
                float(minutes_remaining),
                float(base_strength),
                1.0 if home else 0.0,
                float(man_advantage),
            ],
            dtype="float64",
        )

    def _hazard(self, *, score_diff, minutes_remaining, base_strength, home, man_advantage) -> float:
        if not self._fitted or self._beta is None:
            raise WinProbabilityIntegrityError("model not fitted")
        x = self._feature_row(
            score_diff=score_diff,
            minutes_remaining=minutes_remaining,
            base_strength=base_strength,
            home=home,
            man_advantage=man_advantage,
        )
        z = float(x @ self._beta + self._intercept)
        return 1.0 / (1.0 + np.exp(-z))

    def _hazard_pair(self, base_strength, home, man_advantage):
        """(hazard_home, hazard_away) for the chain -- acting team is the '+1' direction."""

        def hz_home(d, m):
            return self._hazard(
                score_diff=d, minutes_remaining=m, base_strength=base_strength, home=home, man_advantage=man_advantage
            )

        def hz_away(d, m):
            return self._hazard(
                score_diff=-d,
                minutes_remaining=m,
                base_strength=-base_strength,
                home=(not home),
                man_advantage=-man_advantage,
            )

        return hz_home, hz_away

    def _n_intervals(self, minutes_remaining: float) -> int:
        return max(int(np.ceil(minutes_remaining / self.params.interval_minutes)), 0)

    def _apply_isotonic(self, p_win: float) -> float:
        if self._isotonic is None:
            return p_win
        xs, ys = self._isotonic
        return float(np.interp(p_win, xs, ys))

    def outcome_table(self, *, base_strength, home, man_advantage, n_intervals):
        """Backward-DP ``(Pwin, Pdraw, Ploss)`` planes for a fixed (strength, home, man_advantage).

        Examples
        --------
        One per-match table gives O(1) per-action lookups (a fitted model is required, so this is an
        illustrative block)::

            Pw, Pd, Pl = model.outcome_table(base_strength=0.0, home=True, man_advantage=0, n_intervals=90)
            p_win_at_level = Pw[:, 90]  # P(win) by current score_diff with a full match remaining
        """
        hz_home, hz_away = self._hazard_pair(base_strength, home, man_advantage)
        return outcome_table(hz_home, hz_away, n_intervals=n_intervals, K=self.params.lattice_pad)

    def predict_outcome(self, *, score_diff, minutes_remaining, base_strength=0.0, home=True, man_advantage=0):
        """``(p_win, p_draw, p_loss)`` from the acting team's perspective, isotonic-recalibrated.

        Examples
        --------
        Serve one game state on a fitted model (illustrative -- requires a fitted model)::

            p_win, p_draw, p_loss = model.predict_outcome(
                score_diff=1, minutes_remaining=15.0, base_strength=0.2, home=True, man_advantage=0
            )
        """
        n = self._n_intervals(minutes_remaining)
        Pw, Pd, Pl = self.outcome_table(
            base_strength=base_strength, home=home, man_advantage=man_advantage, n_intervals=n
        )
        i = int(score_diff) + self.params.lattice_pad
        w, draw, loss = float(Pw[i, n]), float(Pd[i, n]), float(Pl[i, n])
        w_cal = self._apply_isotonic(w)
        rem = max(1.0 - w_cal, 0.0)
        dl = draw + loss
        d_cal = rem * (draw / dl) if dl > 0 else rem / 2.0
        return w_cal, d_cal, rem - d_cal

    # ---- fit ----
    def fit(self, actions, *, games=None, strength_column=None):
        """Fit the interval-hazard logistic (sklearn at fit-time). Isotonic recalibration if enough matches.

        Examples
        --------
        Fit on a SPADL corpus (illustrative -- requires a real ``actions`` frame + a games frame)::

            model = WinProbabilityModel().fit(actions, games=games, strength_column="home_supremacy")
        """
        from sklearn.linear_model import LogisticRegression

        X, y, groups = _build_interval_training(actions, games, strength_column, self.params)
        if X.shape[0] == 0 or len(np.unique(y)) < 2:
            raise ValueError("fit: training data has < 2 label classes (need scored and unscored intervals)")
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        self._beta = np.asarray(clf.coef_).reshape(-1).astype("float64")
        self._intercept = float(np.asarray(clf.intercept_).reshape(-1)[0])
        self._fitted = True
        n_matches = len(np.unique(groups))
        if n_matches >= _MIN_MATCHES_FOR_ISOTONIC:
            self._fit_isotonic(actions, games, strength_column, groups)
        return self

    def _fit_isotonic(self, actions, games, strength_column, groups):  # pragma: no cover - corpus-only
        """OOF win-prob -> match outcome isotonic recalibration (corpus path; skipped on small fixtures)."""
        # Left as the corpus trainer's responsibility (Task 13); small fixtures set _isotonic=None.
        self._isotonic = None

    # ---- coherence certification (TF63-SPEC-09 fail-closed bundle-time gate) ----
    def certify_coherence(self, *, params: WinProbabilityParams | None = None) -> None:
        """Raise if the fitted hazard yields negative leverage anywhere (P(win) non-monotone in score_diff).

        Examples
        --------
        The fail-closed bundle-time coherence gate (TF63-SPEC-09; illustrative -- requires a fitted
        model)::

            model.certify_coherence()  # raises WinProbabilityIntegrityError on a non-monotone fit
        """
        p = params or self.params
        n = p.regulation_minutes // p.interval_minutes
        for base_strength in (-0.5, 0.0, 0.5):
            for home in (True, False):
                for man_advantage in (-1, 0, 1):
                    Pw, _, _ = self.outcome_table(
                        base_strength=base_strength, home=home, man_advantage=man_advantage, n_intervals=n
                    )
                    diffs = Pw[1:, :] - Pw[:-1, :]
                    if (diffs < -p.leverage_nonneg_atol).any():
                        raise WinProbabilityIntegrityError(
                            "certify_coherence: negative leverage (P(win) non-monotone in score_diff) at "
                            f"base_strength={base_strength}, home={home}, man_advantage={man_advantage}"
                        )

    # ---- feature-contract probe ----
    def _probe_hazards(self) -> list[float]:
        return [
            self._hazard(
                score_diff=s[0], minutes_remaining=s[1], base_strength=s[2], home=bool(s[3]), man_advantage=s[4]
            )
            for s in _PROBE_STATES
        ]

    # ---- serialization ----
    def save(self, directory) -> None:
        """Write the pickle-free artifact (``model.json`` + ``SHA256SUMS``) to ``directory``.

        Examples
        --------
        Persist a fitted model (illustrative -- requires a fitted model + a writable directory)::

            model.save("silly_kicks/win_probability/weights")
        """
        if self._beta is None or self._intercept is None:
            raise WinProbabilityIntegrityError("save: model is not fitted")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        model = {
            "feature_names": list(_HAZARD_FEATURES),
            "beta": self._beta.tolist(),
            "intercept": self._intercept,
            "isotonic": None if self._isotonic is None else [self._isotonic[0].tolist(), self._isotonic[1].tolist()],
            "params": {
                "interval_minutes": self.params.interval_minutes,
                "regulation_minutes": self.params.regulation_minutes,
                "lattice_pad": self.params.lattice_pad,
                "ece_max": self.params.ece_max,
                "slope_tol": self.params.slope_tol,
                "leverage_nonneg_atol": self.params.leverage_nonneg_atol,
            },
            "feature_contract": {
                "probe_states": [list(s) for s in _PROBE_STATES],
                "probe_hazards": self._probe_hazards(),
            },
        }
        model_bytes = json.dumps(model, sort_keys=True, indent=2).encode("utf-8")
        (directory / "model.json").write_bytes(model_bytes)
        (directory / "SHA256SUMS").write_text(
            f"{hashlib.sha256(model_bytes).hexdigest()}  model.json\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory) -> WinProbabilityModel:
        """Load an artifact directory, fail-closed on SHA256 or feature-contract mismatch.

        Examples
        --------
        Round-trip a saved model (illustrative -- requires an artifact directory)::

            model = WinProbabilityModel.load("silly_kicks/win_probability/weights")
        """
        directory = Path(directory)
        model_bytes = (directory / "model.json").read_bytes()
        # 1. integrity: SHA256SUMS
        sums = (directory / "SHA256SUMS").read_text(encoding="utf-8")
        recorded = {line.split()[1]: line.split()[0] for line in sums.strip().splitlines()}
        actual = hashlib.sha256(model_bytes).hexdigest()
        if recorded.get("model.json") != actual:
            raise WinProbabilityIntegrityError("load: model.json SHA256 mismatch (tampered or corrupt)")
        model = json.loads(model_bytes)
        params = WinProbabilityParams(**model["params"])
        obj = cls(params=params)
        obj._beta = np.asarray(model["beta"], dtype="float64")
        obj._intercept = float(model["intercept"])
        iso = model.get("isotonic")
        obj._isotonic = None if iso is None else (np.asarray(iso[0]), np.asarray(iso[1]))
        obj._fitted = True
        # 2. feature-contract probe (chirality N/A: no geometric features)
        fc = model["feature_contract"]
        expected = np.asarray(fc["probe_hazards"], dtype="float64")
        got = np.asarray(obj._probe_hazards(), dtype="float64")
        if not np.allclose(got, expected, atol=1e-6, rtol=0.0, equal_nan=True):
            raise WinProbabilityIntegrityError("load: feature-contract probe mismatch")
        return obj

    @classmethod
    def bundled(cls) -> WinProbabilityModel:
        """Load the wheel-bundled public-corpus default (Task 13 fills weights/).

        Examples
        --------
        The public default model, `functools.cache`'d (illustrative -- requires the bundled weights)::

            from silly_kicks.win_probability import WinProbabilityModel

            model = WinProbabilityModel.bundled()
        """
        return _load_bundled()


@functools.cache
def _load_bundled() -> WinProbabilityModel:
    return WinProbabilityModel.load(_WEIGHTS_DIR)
