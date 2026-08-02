"""TF-16 xShotOccurrence (xS): P(a shot is attempted within ~1 s of a frame).

Implements the xS sub-model of Pipping, Feng & Sabin (2026), arXiv:2512.00203
("Beyond Expected Goals: A Probabilistic Framework for Shot Occurrences in
Soccer"). Only xS is implemented; the paper's xG and xG+ are out of scope
(silly-kicks values goals/threat via VAEP and xthreat). See NOTICE.

Ships UNTRAINED in PR-S75 (code + synthetic CI fixture + real-provider
extraction tests); maintainer training run + bundled/Hub weights follow.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from silly_kicks._nan_safety import nan_safe_enrichment
from silly_kicks.id_compat import align_join_keys, ids_match
from silly_kicks.tracking import _geometry as _geo
from silly_kicks.tracking._ball_carrier import (
    DEFAULT_CARRIER_PARAMS,
    derive_team_in_possession,
    infer_ball_carrier,
)
from silly_kicks.tracking._occurrence_labels import _build_occurrence_labels
from silly_kicks.tracking.utils import link_actions_to_frames

# Goal geometry (goal-relative coords: defended goal at x=0, centre y=34).
GOAL_WIDTH = 7.32
GOAL_Y_CENTRE = 34.0
GOAL_Y_MIN = GOAL_Y_CENTRE - GOAL_WIDTH / 2.0  # 30.34
GOAL_Y_MAX = GOAL_Y_CENTRE + GOAL_WIDTH / 2.0  # 37.66
DEFENDER_RADIUS = 0.375  # 75 cm diameter (paper Appendix A)


def _open_goal_fraction(ball: tuple[float, float], defenders: np.ndarray) -> float:
    """Unobstructed share of the goal mouth from the ball (paper Appendix A).

    Each defender between the ball and the goal line is a circle of radius
    ``DEFENDER_RADIUS``; the ball->defender tangent pair projects an obstructed
    interval onto the goal line. Intervals are UNIONed (overlaps not double
    counted). The GK is not passed in (excluded as an occluder).

    Parameters
    ----------
    ball : (x, y)
        Ball position in goal-relative coords (goal line at x=0).
    defenders : np.ndarray
        Shape (n, 2) of non-GK defender (x, y) in goal-relative coords.

    Returns
    -------
    float
        Open fraction in [0, 1]; NaN if the ball position is NaN/behind goal.

    Examples
    --------
    >>> import numpy as np
    >>> _open_goal_fraction((20.0, 34.0), np.empty((0, 2)))
    1.0
    """
    bx, by = ball
    if math.isnan(bx) or math.isnan(by):
        return float("nan")
    if bx <= 0:
        return float("nan")  # ball on/behind goal line --- undefined

    intervals: list[tuple[float, float]] = []
    for dx, dy in defenders:
        if math.isnan(dx) or math.isnan(dy):
            continue
        # Only defenders strictly between ball and goal line cast a shadow.
        if dx >= bx or dx <= 0:
            continue
        d_bd = math.hypot(dx - bx, dy - by)
        if d_bd <= DEFENDER_RADIUS:
            # Ball essentially on the defender --- full obstruction of the mouth.
            intervals.append((GOAL_Y_MIN, GOAL_Y_MAX))
            continue
        half = math.asin(DEFENDER_RADIUS / d_bd)
        base = math.atan2(dy - by, dx - bx)  # ball -> defender bearing
        ys: list[float] = []
        for ang in (base - half, base + half):
            cos_a = math.cos(ang)
            if abs(cos_a) < 1e-12:
                continue
            t = (0.0 - bx) / cos_a  # param to reach x=0 along the tangent ray
            if t <= 0:
                continue
            ys.append(by + t * math.sin(ang))
        if len(ys) < 2:
            continue
        lo, hi = sorted(ys)
        lo = max(lo, GOAL_Y_MIN)
        hi = min(hi, GOAL_Y_MAX)
        if hi > lo:
            intervals.append((lo, hi))

    if not intervals:
        return 1.0
    intervals.sort()
    merged_len = 0.0
    cur_lo, cur_hi = intervals[0]
    for lo, hi in intervals[1:]:
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged_len += cur_hi - cur_lo
            cur_lo, cur_hi = lo, hi
    merged_len += cur_hi - cur_lo
    return max(0.0, 1.0 - merged_len / GOAL_WIDTH)


XShotFeatureSet = Literal["faithful", "extended"]

_BALL_FEATURES = ["r", "theta", "z", "speed", "openGoal"]
_GK_FEATURES = ["GK_r", "GK_theta"]
# Interleave Dist/Angle per the data-dictionary order: Dist_0, Angle_0, Dist_1, ...
_DEF_INTERLEAVED = [c for k in range(5) for c in (f"DefDist_{k}", f"DefAngle_{k}")]
_OFF_INTERLEAVED = [c for k in range(5) for c in (f"OffDist_{k}", f"OffAngle_{k}")]
XSHOT_FEATURE_NAMES_FAITHFUL = _BALL_FEATURES + _GK_FEATURES + _DEF_INTERLEAVED + _OFF_INTERLEAVED
# 5 + 2 + 10 + 10 = 27


def _nearest_k(ball_xy: tuple[float, float], pts: np.ndarray, k: int = 5):
    """Return (dist[k], bearing[k]) of the k nearest pts to ball, NaN-padded.

    Bearing is the angle of (point - ball) in goal-relative coords.
    """
    dist = np.full(k, np.nan)
    ang = np.full(k, np.nan)
    if len(pts) == 0:
        return dist, ang
    bx, by = ball_xy
    d = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
    order = np.argsort(d)[:k]
    for i, j in enumerate(order):
        dist[i] = d[j]
        ang[i] = math.atan2(pts[j, 1] - by, pts[j, 0] - bx)
    return dist, ang


def extract_xshot_features(
    frame_data: pd.DataFrame,
    *,
    gk_team_id: int | str,
    goal_x: float,
    feature_set: XShotFeatureSet = "faithful",
) -> pd.DataFrame:
    """Extract xS features from one frame (goal-relative). Returns a 1-row frame.

    The defending team is ``gk_team_id`` (the team whose goal is being attacked);
    ``goal_x`` is the absolute x of that defended goal (0.0 or 105.0).

    ``feature_set="extended"`` is not implemented in PR-S75 (raises
    NotImplementedError) --- only ``"faithful"`` (the paper's 27 features) ships
    here; see the TF-16 spec.

    Examples
    --------
    >>> # row = extract_xshot_features(frame, gk_team_id=1, goal_x=0.0)
    >>> # row.shape == (1, 27)

    See NOTICE for full bibliographic citations.
    """
    if feature_set != "faithful":
        raise NotImplementedError(
            "xShotOccurrence feature_set='extended' is not implemented in this "
            "release; only 'faithful' (paper Appendix A) is available. See the "
            "TF-16 weights/TF-19 follow-up."
        )

    # Goal-relative ONCE, for every consumer below: the 180-degree point reflection
    # (x, y) -> (105 - x, 68 - y) when the defended goal is at high x. Do NOT reintroduce a
    # per-site helper. `gx` was x-only while y was read at FOUR independent sites (ball, defenders,
    # attackers, keeper), so before PR 5 every bearing negated between the two goal ends while every
    # radial stayed byte-identical -- and a per-site fix leaves "no call site can be missed" an
    # assertion rather than a property. `assign` returns a NEW frame (ADR-033: never mutate the
    # caller's).
    fd = frame_data.assign(
        x=frame_data["x"].map(lambda v: _geo.to_goal_relative_x(float(v), goal_x=goal_x)),
        y=frame_data["y"].map(lambda v: _geo.to_goal_relative_y(float(v), goal_x=goal_x)),
    )

    is_ball = fd["is_ball"].astype(bool)
    is_gk = fd["is_goalkeeper"].astype(bool)
    ball = fd[is_ball]
    players = fd[~is_ball]
    players_is_gk = is_gk[~is_ball]

    if len(ball) > 0:
        bx = float(ball["x"].iloc[0])
        by = float(ball["y"].iloc[0])
        bvx = float(ball["vx"].iloc[0]) if "vx" in ball.columns else np.nan
        bvy = float(ball["vy"].iloc[0]) if "vy" in ball.columns else np.nan
        bz = float(ball["z"].iloc[0]) if "z" in ball.columns else np.nan
    else:
        bx = by = bvx = bvy = bz = np.nan

    r = math.hypot(bx, by - _geo.GOAL_Y) if not math.isnan(bx) else np.nan
    theta = math.atan2(by - _geo.GOAL_Y, bx) if not math.isnan(bx) else np.nan
    speed = math.hypot(bvx, bvy) if not math.isnan(bvx) else np.nan

    # ADR-019: dtype-safe team-id identity (Int64(2) vs "2" must match). One computed
    # mask, three consumers; ids_match resolves NA to False, so NaN-team rows land in
    # ``attacking`` via the plain complement -- byte-matching the legacy ``!=``.
    is_gk_team = ids_match(players["team_id"], gk_team_id)
    defending = players[is_gk_team & (~players_is_gk)]
    attacking = players[~is_gk_team]
    gk_rows = players[is_gk_team & players_is_gk]

    def_xy = (
        np.column_stack(
            [
                defending["x"].to_numpy(dtype=float),
                defending["y"].to_numpy(dtype=float),
            ]
        )
        if len(defending)
        else np.empty((0, 2))
    )
    atk_xy = (
        np.column_stack(
            [
                attacking["x"].to_numpy(dtype=float),
                attacking["y"].to_numpy(dtype=float),
            ]
        )
        if len(attacking)
        else np.empty((0, 2))
    )

    open_goal = _open_goal_fraction((bx, by), def_xy)

    if len(gk_rows) > 0:
        gkx = float(gk_rows["x"].iloc[0])
        gky = float(gk_rows["y"].iloc[0])
        gk_r = math.hypot(gkx, gky - _geo.GOAL_Y)
        gk_theta = math.atan2(gky - _geo.GOAL_Y, gkx)
    else:
        gk_r = gk_theta = np.nan

    ddist, dang = _nearest_k((bx, by), def_xy)
    odist, oang = _nearest_k((bx, by), atk_xy)

    values: dict[str, float] = {
        "r": r,
        "theta": theta,
        "z": bz,
        "speed": speed,
        "openGoal": open_goal,
        "GK_r": gk_r,
        "GK_theta": gk_theta,
    }
    for k in range(5):
        values[f"DefDist_{k}"] = ddist[k]
        values[f"DefAngle_{k}"] = dang[k]
        values[f"OffDist_{k}"] = odist[k]
        values[f"OffAngle_{k}"] = oang[k]

    return pd.DataFrame(
        [[values[c] for c in XSHOT_FEATURE_NAMES_FAITHFUL]],
        columns=XSHOT_FEATURE_NAMES_FAITHFUL,
    )


def build_xshot_labels(
    frames_index: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    horizon_seconds: float = 1.0,
) -> pd.Series:
    """Per-row xS label: 1 iff a same-team shot occurs in [t, t+horizon] same period.

    Compares the shot action's own ``time_seconds`` directly against each frame
    row's window --- NO linkage step (avoids the +/-tolerance link smear at the 1 s
    horizon). Robust to non-contiguous ``frame_id`` (uses time, not frame index).

    Parameters
    ----------
    frames_index : pd.DataFrame
        One row per scored frame slot; columns ``game_id``, ``period_id``,
        ``time_seconds``, ``team_in_possession``.
    shots : pd.DataFrame
        Shot actions; columns ``game_id``, ``period_id``, ``team_id``,
        ``time_seconds``.

    Returns
    -------
    pd.Series
        int (0/1), aligned to ``frames_index.index``.

    Examples
    --------
    >>> # y = build_xshot_labels(fidx, shots, horizon_seconds=1.0)
    """
    # Thin wrapper over the shared occurrence helper (M-3). frames carry the possessing team
    # as `team_in_possession`; shots carry the acting team as `team_id`.
    y = _build_occurrence_labels(frames_index, shots, horizon=horizon_seconds, frame_team_col="team_in_possession")
    return pd.Series(y, index=frames_index.index)


_DEFAULT_CARRIER_PARAMS = DEFAULT_CARRIER_PARAMS  # shared constant (anti-drift; PR-S80 L1)
_DEFAULT_SHOT_TYPES = ("shot", "shot_penalty", "shot_freekick")
_HF_REPO_ID = "silly-kicks/xshot-occurrence-v1"
_MODEL_VERSION = "1.0.0"
_XSHOT_WEIGHTS_ROOT = Path(__file__).parent / "_xshot_weights"
_VARIANT_CACHE: dict = {}  # P3: memoize bundled loads (default-list serve perf)
# The bundled "default" IS the public arm (its metadata records shipped_variant="public").
# "public" was a stale alias reserved at 4.9.0 for a public Hub artifact never created; PR-S118
# added "sc_extended" alongside it in the from_hub branch, so from_variant("public") fell
# through to the Hub and served the RESTRICTED sc_extended artifact under the "public" key.
# Resolve the alias BEFORE the cache so the request maps to the reproducible bundled model
# (spec 2026-07-20 §8).
_VARIANT_ALIASES = {"public": "default"}
_HUB_VARIANTS = frozenset({"sc_extended"})
_INT_PARAMS = ("n_estimators", "max_depth", "min_child_weight")


class IntegrityError(Exception):
    """Raised when a model artifact fails SHA-256 verification."""


def _chirality_block(model: XShotOccurrenceModel) -> dict:
    """Behavioral chirality fingerprint (ADR-037): the model's own extractor + predict on
    the canonical y-asymmetric probe frame. Emitted into save() metadata; a y-mirrored
    artifact cannot reproduce it (the 4.18.0-weights class of bug)."""
    from silly_kicks.tracking._chirality import chirality_fingerprint

    def _predict(frame):
        feats = extract_xshot_features(frame, gk_team_id="B", goal_x=105.0)
        return model.predict_proba(feats)

    return chirality_fingerprint(_predict)


def _feature_contract_block() -> dict:
    """Feature contract (ADR-050): this model's FEATURE VECTOR on the fixed probe frame, plus the
    geometry constants its extractor consumes.

    Model-independent by design -- unlike ``_chirality_block`` it takes no model, because it
    fingerprints the EXTRACTOR, not the fitted weights. Chirality catches a y-mirror in the output;
    this catches a geometry change that shifts an input feature, which the output fingerprint can
    absorb for any feature carrying little weight.

    xS declares ``goal_width`` and nothing else: it consumes NO penalty-area constant (its only
    geometry constants are the goal mouth, which drives ``openGoal``). Declaring a constant this
    extractor does not read would make ``load()`` raise on a change that provably moves none of its
    features -- a guard firing when nothing happened, which is how ``legacy_override`` becomes
    reflex.
    """
    from silly_kicks.tracking._feature_contract import contract_probe_frame, feature_contract

    def _vec():
        return (
            extract_xshot_features(contract_probe_frame(), gk_team_id="B", goal_x=105.0).iloc[0].to_numpy(dtype=float)
        )

    return feature_contract(_vec, constants={"goal_width": GOAL_WIDTH})


def _pinned_params(overrides: dict | None) -> dict:
    """Deterministic XGBoost params (matches the calibration house standard).

    Optuna FloatRange feeds floats; XGBoost wants ints for n_estimators/max_depth/
    min_child_weight --- round those so the search space can be all-float.
    """
    base = {
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.3,
        "tree_method": "hist",
        "n_jobs": 1,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "random_state": 42,
        "eval_metric": "logloss",
        "verbosity": 0,
    }
    if overrides:
        base.update(overrides)
    for k in _INT_PARAMS:
        if base.get(k) is not None:
            base[k] = round(float(base[k]))
    return base


class XShotOccurrenceModel:
    """xS classifier: pinned-deterministic XGBoost over snapshot frame features.

    Serialization is pickle-free (xgboost native booster JSON + metadata.json +
    SHA256SUMS). ``carrier_params`` are recorded so inference can resolve
    possession identically to training (R3). See NOTICE.

    Examples
    --------
    >>> # m = XShotOccurrenceModel().fit(X, y)
    >>> # p = m.predict_proba(X)
    """

    def __init__(self, *, feature_set: XShotFeatureSet = "faithful", params: dict | None = None) -> None:
        if feature_set != "faithful":
            raise NotImplementedError("Only feature_set='faithful' is implemented.")
        self.feature_set: XShotFeatureSet = feature_set
        self._params = _pinned_params(params)
        self._booster = None  # xgboost.Booster after fit/load
        self.carrier_params: dict = dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds: float = 1.0
        self.shot_types: list[str] = ["shot", "shot_penalty", "shot_freekick"]
        # Provenance (set by the trainer before save(); recorded in metadata — N5).
        self.shipped_variant: str | None = None
        self.provider_list: list | None = None

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        *,
        carrier_params: dict | None = None,
        horizon_seconds: float = 1.0,
    ) -> XShotOccurrenceModel:
        """Fit the classifier. Records ``carrier_params`` into the model (R3).

        Examples
        --------
        >>> # XShotOccurrenceModel().fit(X, y)
        """
        import xgboost as xgb

        if int(xgb.__version__.split(".")[0]) < 2:
            raise RuntimeError("xShotOccurrence requires xgboost>=2.0 (calibrated base_score / intercept estimation).")
        self.carrier_params = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
        self.horizon_seconds = horizon_seconds
        params = dict(self._params)
        # N4: anchor calibration to the train positive rate explicitly so the M2 "log-loss keeps
        # P(shot) calibrated without scale_pos_weight" claim does not silently depend on xgboost's
        # auto-intercept behaviour surviving a future pin move.
        params["base_score"] = float(np.asarray(labels, dtype=float).mean())
        clf = xgb.XGBClassifier(**params)
        clf.fit(features.to_numpy(dtype=float), labels.to_numpy(dtype=int))
        booster = clf.get_booster()
        booster.feature_names = list(features.columns)
        self._booster = booster
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return P(shot) per row, shape (n,).

        Examples
        --------
        >>> # p = model.predict_proba(X)
        """
        if self._booster is None:
            raise RuntimeError("Model not fitted/loaded.")
        import xgboost as xgb

        dm = xgb.DMatrix(features.to_numpy(dtype=float), feature_names=list(features.columns))
        return np.asarray(self._booster.predict(dm), dtype=float)

    def save(self, path: Path) -> None:
        """Serialize to booster JSON + metadata.json + SHA256SUMS (no pickle).

        Examples
        --------
        >>> # model.save(Path("models/xshot_occurrence_v1"))
        """
        if self._booster is None:
            raise RuntimeError("Model not fitted.")
        import platform

        import xgboost as xgb

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path / "model.json"))
        metadata = {
            "feature_names": XSHOT_FEATURE_NAMES_FAITHFUL,
            "feature_set": self.feature_set,
            "horizon_seconds": self.horizon_seconds,
            "shot_types": self.shot_types,
            "carrier_params": self.carrier_params,
            "params": self._params,
            "version": _MODEL_VERSION,
            # Coordinate/units template + reproducibility (PR-S80 §6 / M4, L2, N5).
            "pitch_length": _geo.PITCH_LENGTH,
            "pitch_width": _geo.PITCH_WIDTH,
            "geometry_version": _geo.GEOMETRY_VERSION,
            "xgboost_version": xgb.__version__,
            "training_platform": platform.platform(),
            "shipped_variant": self.shipped_variant,
            "provider_list": self.provider_list,
            "chirality": _chirality_block(self),
            "feature_contract": _feature_contract_block(),
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), newline="\n")
        with open(path / "SHA256SUMS", "w", newline="\n") as f:
            for fname in ["model.json", "metadata.json"]:
                raw = (path / fname).read_bytes()
                if fname.endswith(".json"):
                    raw = raw.replace(b"\r\n", b"\n")
                f.write(f"{hashlib.sha256(raw).hexdigest()}  {fname}\n")

    @classmethod
    def load(cls, path: Path, *, legacy_override: bool = False) -> XShotOccurrenceModel:
        """Load from a local directory, verifying SHA-256. Requires xgboost.

        A behavioral chirality fingerprint is enforced (ADR-037 § 9, TF-19 PR-2): a
        pre-PR-2 artifact with no fingerprint is REFUSED unless ``legacy_override=True``
        (which warns), and an output/probe-frame mismatch raises. See ``_chirality``.

        Examples
        --------
        >>> # model = XShotOccurrenceModel.load(Path("models/xshot_occurrence_v1"))
        """
        path = Path(path)
        sums = path / "SHA256SUMS"
        if not sums.exists():
            raise IntegrityError(f"SHA256SUMS not found in {path}")
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            expected, fname = line.split("  ", 1)
            raw = (path / fname).read_bytes()
            if fname.endswith(".json"):
                raw = raw.replace(b"\r\n", b"\n")
            if hashlib.sha256(raw).hexdigest() != expected:
                raise IntegrityError(f"Integrity check failed for {fname}")
        meta = json.loads((path / "metadata.json").read_text())
        # Coordinate-change guard (PR-S80 §6, M4): a pitch-dimension/unit mismatch genuinely skews
        # every goal-relative feature -> FAIL CLOSED (raise). A geometry_version change at identical
        # dims is the translation-invariant case (e.g. the TF-38 origin shift) -> warn only (a
        # warnings.warn is invisible in a swallowed-stdout Spark/batch serve, so we never rely on it
        # for the case that actually skews).
        rec_len = meta.get("pitch_length")
        rec_wid = meta.get("pitch_width")
        if rec_len is not None and (rec_len != _geo.PITCH_LENGTH or rec_wid != _geo.PITCH_WIDTH):
            raise IntegrityError(
                f"Pitch-dimension mismatch: model trained on {rec_len}x{rec_wid} m, library is "
                f"{_geo.PITCH_LENGTH}x{_geo.PITCH_WIDTH} m. Goal-relative features would be skewed; "
                "refusing to load (retrain required)."
            )
        rec_geo = meta.get("geometry_version")
        if rec_geo is not None and rec_geo != _geo.GEOMETRY_VERSION:
            warnings.warn(
                f"geometry_version mismatch (model={rec_geo}, library={_geo.GEOMETRY_VERSION}) at "
                "identical pitch dimensions -- treated as translation-invariant. Verify if a "
                "non-translation coordinate change occurred.",
                stacklevel=2,
            )
        model = cls(feature_set=meta.get("feature_set", "faithful"), params=meta.get("params"))
        model.carrier_params = meta.get("carrier_params", dict(_DEFAULT_CARRIER_PARAMS))
        model.horizon_seconds = meta.get("horizon_seconds", 1.0)
        model.shot_types = meta.get("shot_types", model.shot_types)
        model.shipped_variant = meta.get("shipped_variant")
        model.provider_list = meta.get("provider_list")
        model._booster = load_xgb_booster_base_score_safe(path / "model.json")

        from silly_kicks.tracking._chirality import verify_chirality

        verify_chirality(
            _chirality_block(model),
            meta.get("chirality"),
            legacy_override=legacy_override,
            model_name="xShotOccurrence",
        )

        from silly_kicks.tracking._feature_contract import verify_feature_contract

        verify_feature_contract(
            _feature_contract_block(),
            meta.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="xShotOccurrence",
            error_cls=IntegrityError,
        )
        return model

    @classmethod
    def from_variant(cls, variant: str = "default") -> XShotOccurrenceModel:
        """Load a bundled variant by name (memoized). ``"public"`` aliases to the bundled
        ``"default"`` (which IS the public arm); only ``"sc_extended"`` falls through to the Hub.

        ``"default"`` is bundled in the wheel and SHA-256 verified on first load, then cached —
        an immutable, inference-only instance is safe to share across calls (P3: avoids reloading
        + re-verifying per call on the default-xfns serve path).

        Examples
        --------
        >>> # XShotOccurrenceModel.from_variant("default")
        """
        variant = _VARIANT_ALIASES.get(variant, variant)
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        weights_dir = _XSHOT_WEIGHTS_ROOT / variant
        if (weights_dir / "SHA256SUMS").exists():
            model = cls.load(weights_dir)
        elif variant in _HUB_VARIANTS:  # the Hub-hosted variants
            model = cls.from_hub(_HF_REPO_ID)
        else:
            raise FileNotFoundError(
                f"No bundled xShotOccurrence weights for variant {variant!r} at {weights_dir}. "
                "Train via scripts/train_xshot_occurrence.py, or use from_hub()."
            )
        _VARIANT_CACHE[variant] = model
        return model

    @classmethod
    def from_hub(cls, repo_id: str = _HF_REPO_ID) -> XShotOccurrenceModel:
        """Download published weights from HuggingFace Hub and load.

        Requires ``pip install silly-kicks[xshot]``.

        Examples
        --------
        >>> # model = XShotOccurrenceModel.from_hub()
        """
        try:
            from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError("xShotOccurrence Hub weights require: pip install silly-kicks[xshot]") from None
        local_dir = snapshot_download(repo_id=repo_id)
        return cls.load(Path(local_dir))


def _resolve_model(model: XShotOccurrenceModel | str | None) -> XShotOccurrenceModel:
    if isinstance(model, XShotOccurrenceModel):
        return model
    if model is None or isinstance(model, str):
        return XShotOccurrenceModel.from_variant(model or "default")  # raises until weights ship
    raise TypeError(f"Unsupported model type: {type(model)!r}")


def load_xgb_booster_base_score_safe(model_json_path: Path):
    """Load an xgboost Booster, defensively normalizing a bracketed ``base_score`` (TF-19 PR-2).

    xgboost 3.x serializes ``learner.learner_model_param.base_score`` as a bracketed STRING
    (e.g. ``"[2.19E-1]"``). xgboost 2.x cannot parse that and silently drops to the ``0.5``
    default -> a mis-served intercept (the whole margin is offset, predictions are wrong). The
    library supports ``xgboost>=2.0`` across the 2.x/3.x boundary, so ``load()`` normalizes the
    bracketed form to a scalar in-memory before handing it to the Booster. A scalar base_score
    (2.x-native and already-fixed bundled weights) passes straight through unchanged.
    """
    import xgboost as xgb

    booster = xgb.Booster()
    # xgboost writes model.json as UTF-8; read it as such (a booster with non-ASCII feature names
    # would raise UnicodeDecodeError under a non-UTF-8 platform default, e.g. cp1252 on Windows).
    raw = json.loads(Path(model_json_path).read_text(encoding="utf-8"))
    bs = raw.get("learner", {}).get("learner_model_param", {}).get("base_score")
    if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
        raw["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
        booster.load_model(bytearray(json.dumps(raw).encode()))
    else:
        booster.load_model(str(model_json_path))
    return booster


# Moved byte-identically to _gk_resolve.defended_goal_x (TF-48); shim keeps all
# internal `_defended_goal_x(...)` call sites working unchanged.
from silly_kicks.tracking._gk_resolve import defended_goal_x as _defended_goal_x  # noqa: E402

_ATTACKING_THIRD_M = 35.0  # final third = within 35 m of the attacked goal line


def _ball_in_attacking_third(ball_x: float, goal_x: float) -> bool:
    """True if the ball is in the third nearest ``goal_x`` (the attacked goal)."""
    if math.isnan(ball_x):
        return False
    return abs(ball_x - goal_x) <= _ATTACKING_THIRD_M


def prepare_xshot_training_data(
    frames: pd.DataFrame,
    shots: pd.DataFrame,
    *,
    home_team_id: int | str,  # reserved for symmetry with add_*/compute_*
    feature_set: XShotFeatureSet = "faithful",
    horizon_seconds: float = 1.0,
    attacking_third_only: bool = True,
    shot_types: tuple[str, ...] | None = None,
    carrier_params: dict | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build (features, labels, groups) for one match's frames -- the shared,
    public train/serve-parity entry point (spec §3.5).

    Applies the paper's data-curation domain filter (spec §4): keep only frames
    where (a) ``ball_state == "alive"``, (b) a possession team is resolvable via
    ``infer_ball_carrier`` -> ``derive_team_in_possession`` (carrier params from
    ``carrier_params``, default the library default), and (c) when
    ``attacking_third_only`` (default True) the ball is in the in-possession
    team's attacking third. Labels come from :func:`build_xshot_labels` over
    ``shots`` filtered to ``shot_types`` (default {shot, shot_penalty,
    shot_freekick}). ``groups`` is ``game_id`` for match-stratified GroupKFold.

    Always returns the **faithful** class distribution (no subsampling): this is the
    train/serve-parity entry point, so contaminating it with a class-balance change
    would silently distort any downstream CV's eval folds + base-rate baselines. For
    wall-clock/memory control on large corpora, apply :func:`subsample_negatives` to a
    **training split only** (never the held-out/eval fold). [PR-S80: moved out of here.]

    Parameters
    ----------
    frames : pd.DataFrame
        Canonical tracking frames for one match.
    shots : pd.DataFrame
        Shot actions; columns ``game_id``/``period_id``/``team_id``/
        ``time_seconds`` (+ optionally ``type_name`` for shot-type filtering).
    home_team_id : int | str
        Accepted for call-site symmetry with the ``add_*``/``compute_*`` surfaces.

    Returns
    -------
    (features, labels, groups)
        ``features`` is a (n, 27) DataFrame; ``labels`` int 0/1; ``groups`` the
        per-row ``game_id``.

    Examples
    --------
    >>> # X, y, groups = prepare_xshot_training_data(frames, shots, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    if feature_set != "faithful":
        raise NotImplementedError("Only feature_set='faithful' is implemented.")
    cp: dict = dict(carrier_params) if carrier_params else dict(_DEFAULT_CARRIER_PARAMS)
    types = _DEFAULT_SHOT_TYPES if shot_types is None else tuple(shot_types)

    work = frames
    if "ball_state" in frames.columns:
        work = frames[frames["ball_state"].astype(str) == "alive"]
    if len(work) == 0:
        empty = pd.DataFrame(columns=XSHOT_FEATURE_NAMES_FAITHFUL)
        return empty, np.zeros(0, dtype=int), np.zeros(0)

    carrier = infer_ball_carrier(work, **cp)
    poss = derive_team_in_possession(work, carrier)
    goal_map = _defended_goal_x(work)

    feat_rows: list[pd.DataFrame] = []
    labels_idx: list[dict] = []
    for (gid, pid, _fid), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        outfield = grp[~grp["is_ball"].astype(bool)]
        others = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not others:
            continue
        def_team = others[0]
        goal_x = goal_map.get((gid, pid, def_team))
        if goal_x is None:
            continue
        if attacking_third_only:
            ball = grp[grp["is_ball"].astype(bool)]
            ball_x = float(ball["x"].iloc[0]) if len(ball) else float("nan")
            if not _ball_in_attacking_third(ball_x, goal_x):
                continue
        feat_rows.append(extract_xshot_features(grp, gk_team_id=def_team, goal_x=goal_x))
        labels_idx.append(
            {
                "game_id": gid,
                "period_id": pid,
                "time_seconds": float(grp["time_seconds"].iloc[0]),
                "team_in_possession": tip,
            }
        )

    if not feat_rows:
        empty = pd.DataFrame(columns=XSHOT_FEATURE_NAMES_FAITHFUL)
        return empty, np.zeros(0, dtype=int), np.zeros(0)

    features = pd.concat(feat_rows, ignore_index=True)[XSHOT_FEATURE_NAMES_FAITHFUL]
    fidx = pd.DataFrame(labels_idx)

    shots_f = shots
    if "type_name" in shots.columns:
        shots_f = shots[shots["type_name"].isin(types)]
    elif not types:
        shots_f = shots.iloc[0:0]  # empty type set + no type column -> no positives
    labels = build_xshot_labels(fidx, shots_f, horizon_seconds=horizon_seconds).to_numpy()
    groups = fidx["game_id"].to_numpy()
    return features, labels, groups


def subsample_negatives(
    features: pd.DataFrame,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    fraction: float,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Drop ``fraction`` of the negative-label rows (deterministic given ``seed``).

    **TRAIN-ONLY.** Apply this to a *training* split for wall-clock/memory control on large
    corpora. NEVER apply it to a held-out / evaluation fold: subsampling the eval set fakes its
    class balance and distorts log-loss / PR-AUC / Brier *and* the ``positive_rate`` /
    ``base_rate_brier`` baselines those gates compare against. ``positive`` rows are always kept.

    Returns a new ``(features, labels, groups)`` with the negatives thinned; a no-op (returns the
    inputs unchanged) when ``fraction`` is falsy/<=0 or there is nothing to drop.

    Examples
    --------
    >>> # Xtr, ytr, gtr = subsample_negatives(Xtr, ytr, gtr, fraction=0.5, seed=0)  # TRAIN fold only
    """
    if not fraction or fraction <= 0.0:
        return features, labels, groups
    labels = np.asarray(labels)
    neg_idx = np.flatnonzero(labels == 0)
    n_drop = round(len(neg_idx) * float(fraction))
    if n_drop <= 0:
        return features, labels, groups
    rng = np.random.default_rng(seed)
    drop = rng.choice(neg_idx, size=n_drop, replace=False)
    keep = np.ones(len(labels), dtype=bool)
    keep[drop] = False
    return features.iloc[keep].reset_index(drop=True), labels[keep], np.asarray(groups)[keep]


def compute_xshot_occurrence(
    frames: pd.DataFrame,
    *,
    model: XShotOccurrenceModel | str | None = None,
    home_team_id: int | str | None = None,  # unused (goal resolved GK-based); kept for call symmetry
    pitch_control_cache=None,  # reserved for 'extended' (not used by 'faithful')
    link_frame_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Add an ``xshot_occurrence`` column (P(shot within ~1 s)) per in-possession frame.

    Possession + defended goal are resolved with the carrier params stored in the
    model's metadata (R3), so serve-time selection matches training. Rows whose
    state is undefined get NaN. ``pitch_control_cache`` is accepted for forward
    compat with the (deferred) 'extended' variant but is valid only for canonical
    frames --- counterfactual callers must omit it. See NOTICE / spec §7.

    Examples
    --------
    >>> # out = compute_xshot_occurrence(frames, model=m, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    m = _resolve_model(model)
    out = frames.copy()
    out["xshot_occurrence"] = np.nan

    # N-A: carrier inference + possession MUST run on the FULL contiguous frames.
    # infer_ball_carrier has a CROSS-FRAME dependency (gamma hysteresis carries the
    # incumbent carrier across consecutive frames within a (game, period) segment).
    # Running it on the scattered link_frame_ids subset would yield a different
    # carrier -> different team_in_possession, AND diverge from the trainer (which
    # runs it on full frames) -> train/serve skew (R3). So: carrier on full frames;
    # restrict ONLY the per-frame extract + batched predict (the expensive part).
    carrier = infer_ball_carrier(frames, **m.carrier_params)
    poss = derive_team_in_possession(frames, carrier)
    goal_map = _defended_goal_x(frames)

    # Pass 1: build ONE feature row per target frame (extraction stays per-frame;
    # the XGBoost predict must NOT be --- P1). Restrict to link_frame_ids HERE.
    feat_rows: list[pd.DataFrame] = []
    keys: list[tuple] = []  # (gid, pid, frame_id, team_in_possession)
    n_groups = 0
    n_skipped_goal = 0  # N1: count frames dropped for a missing goal map entry
    for (gid, pid, frame_id), grp in poss.groupby(["game_id", "period_id", "frame_id"], dropna=False):
        if link_frame_ids is not None and int(str(frame_id)) not in link_frame_ids:
            continue
        n_groups += 1
        tip = grp["team_in_possession"].iloc[0]
        if pd.isna(tip):
            continue
        # Defending team = a non-ball team that is NOT in possession. Exclude the
        # ball row (sentinel team_id) so it is never mistaken for a defending team.
        outfield = grp[~grp["is_ball"].astype(bool)]
        teams = [t for t in outfield["team_id"].dropna().unique() if t != tip]
        if not teams:
            continue
        def_team = teams[0]
        goal_x = goal_map.get((gid, pid, def_team))
        if goal_x is None:
            n_skipped_goal += 1
            continue
        feat_rows.append(extract_xshot_features(grp, gk_team_id=def_team, goal_x=goal_x))
        keys.append((gid, pid, frame_id, tip))

    # N1: surface coverage loss rather than dropping silently.
    if n_skipped_goal and n_groups and n_skipped_goal / n_groups > 0.05:
        warnings.warn(
            f"xshot_occurrence: {n_skipped_goal}/{n_groups} frame-groups skipped "
            f"(no defended-goal resolution); possible GK-identification gap.",
            stacklevel=2,
        )

    if not feat_rows:
        return out

    # Pass 2: ONE batched predict over the stacked matrix, scatter back.
    feature_matrix = pd.concat(feat_rows, ignore_index=True)
    probs = m.predict_proba(feature_matrix)
    key_df = pd.DataFrame(keys, columns=["game_id", "period_id", "frame_id", "team_id"])
    key_df["__p"] = probs
    # N-B: join on TEMPORARY string keys so we never mutate out["game_id"]/["team_id"]
    # dtypes (preserve the TRACKING_FRAMES schema --- "add one column, change nothing else").
    out["__gid"] = out["game_id"].astype(str)
    out["__tid"] = out["team_id"].astype(str)
    key_df["__gid"] = key_df["game_id"].astype(str)
    key_df["__tid"] = key_df["team_id"].astype(str)
    key_df = key_df.drop(columns=["game_id", "team_id"])
    out = out.merge(key_df, on=["__gid", "period_id", "frame_id", "__tid"], how="left")
    out["xshot_occurrence"] = out["__p"]
    return out.drop(columns=["__p", "__gid", "__tid"])


@nan_safe_enrichment
def add_xshot_occurrence(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: XShotOccurrenceModel | str | None = None,
    links: pd.DataFrame | None = None,
    home_team_id: int | str | None = None,  # unused (goal resolved GK-based); kept for call symmetry
    pitch_control_cache=None,
) -> pd.DataFrame:
    """Enrich SPADL actions with an ``xshot_occurrence`` column (xS at the linked frame).

    xS is the **possessing team's** shot probability: an action by the team in
    possession at its linked frame receives a value; a defensive action by the
    non-possessing team at the same frame receives NaN by design (S1). NaN
    identifiers route to NaN output (ADR-003). ``links`` skips internal linking.

    Examples
    --------
    >>> # out = add_xshot_occurrence(actions, frames, model=m, home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    m = _resolve_model(model)
    out = actions.copy()
    pointers = links if links is not None else link_actions_to_frames(actions, frames)[0]

    link_frame_ids = None
    if "frame_id" in pointers.columns:
        link_frame_ids = {int(f) for f in pointers["frame_id"].dropna().astype(int).tolist()}

    if "xshot_occurrence" in frames.columns and frames["xshot_occurrence"].notna().any():
        scored = frames
    else:
        scored = compute_xshot_occurrence(frames, model=m, home_team_id=home_team_id, link_frame_ids=link_frame_ids)

    # Map each action to the xS at its linked frame + its own team.
    xcol = scored[scored["xshot_occurrence"].notna()][
        ["game_id", "period_id", "frame_id", "team_id", "xshot_occurrence"]
    ].copy()
    linked = pointers.merge(actions[["action_id", "game_id", "period_id", "team_id"]], on="action_id", how="left")
    # P2: align BOTH game_id AND team_id dtypes (provider asymmetry: int64 vs object).
    #
    # Routed through align_join_keys, NOT a raw .astype(str). The bare cast was an ADR-019
    # violation that silently emptied this merge: an int64 team_id 1 stringifies to "1" while a
    # float64 1.0 stringifies to "1.0", so the keys never matched and `add_xshot_occurrence`
    # returned all-NaN while `compute_xshot_occurrence` on the same frames returned real values.
    # `canonical_id_series` collapses the integral float; `str()` does not. The structural twin
    # `_xcross_attempt.py:799` already used this seam -- the two have been reconciled.
    merge_keys = ["game_id", "period_id", "frame_id", "team_id"]
    linked, xcol = align_join_keys(linked, xcol, merge_keys)
    merged = linked.merge(xcol, on=merge_keys, how="left")
    deduped = merged.drop_duplicates(subset=["action_id"], keep="first")
    col = deduped.set_index("action_id")["xshot_occurrence"]
    out = out.merge(col.rename("xshot_occurrence"), left_on="action_id", right_index=True, how="left")
    return out


def xshot_occurrence_xfns(
    *,
    model: XShotOccurrenceModel | str | None = None,
    home_team_id: int | str | None = None,  # unused (goal resolved GK-based); kept for call symmetry
    pitch_control_cache=None,
) -> list:
    """Factory returning a FrameAwareTransformer emitting xshot_occurrence_a0/_a1/_a2.

    NOT added to any default/union xfn list until weights ship (spec §9 / D1).

    Examples
    --------
    >>> # xfns = xshot_occurrence_xfns(home_team_id=1)

    See NOTICE for full bibliographic citations.
    """
    cols = ["xshot_occurrence_a0", "xshot_occurrence_a1", "xshot_occurrence_a2"]

    def _transformer(states, frames):
        out = pd.DataFrame(index=states[0].index)
        if frames is None:
            for c in cols:
                out[c] = np.nan
            return out
        m = _resolve_model(model)
        slot_pointers = []
        link_frame_ids: set[int] = set()
        for slot in states[:3]:
            ptr = link_actions_to_frames(slot, frames)[0]
            slot_pointers.append(ptr)
            if "frame_id" in ptr.columns:
                link_frame_ids |= {int(f) for f in ptr["frame_id"].dropna().astype(int).tolist()}
        scored = compute_xshot_occurrence(frames, model=m, home_team_id=home_team_id, link_frame_ids=link_frame_ids)
        for i, (slot, ptr) in enumerate(zip(states[:3], slot_pointers, strict=False)):
            enriched = add_xshot_occurrence(slot, scored, model=m, home_team_id=home_team_id, links=ptr)
            out[cols[i]] = enriched["xshot_occurrence"].to_numpy() if "xshot_occurrence" in enriched else np.nan
        return out

    _transformer._frame_aware = True  # type: ignore[attr-defined]
    _transformer.__name__ = "xshot_occurrence_xfn"
    return [_transformer]
