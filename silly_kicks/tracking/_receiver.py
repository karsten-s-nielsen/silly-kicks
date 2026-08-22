"""Expected-receiver model — candidate feature extraction + geometric-proxy baseline.

Infers the INTENDED receiver of a pass from PRE-PASS state only (positions, and — owner variant —
the ball's release-velocity direction + candidate closing speed). It NEVER reads the pass end/loss
location: the pass-event angle is origin->end and is banned, so the only leakage-free release
direction is the ball's velocity vector at the release frame, which velocity-less SB360 freeze frames
do not carry. Hence:

- ``feature_set="public"`` (bundled default) = POSITIONS ONLY (``ball_dist``, ``lane_pressure``,
  ``space``) -- leakage-free on any provider.
- ``feature_set="owner"`` ADDS the velocity-derived ``release_dir_align`` + ``closing_speed`` and
  RAISES if the frame carries no velocity.

See NOTICE for full bibliographic citations (Power et al. 2017).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from silly_kicks.id_compat import canonical_id, ids_match

_WEIGHTS_ROOT = Path(__file__).parent / "_receiver_weights"
_VARIANT_CACHE: dict[str, ReceiverModel] = {}


class IntegrityError(Exception):
    """Raised when a bundled receiver artifact fails an integrity guard (SHA / chirality / contract).

    Examples
    --------
    Raised by ``ReceiverModel.load`` on a tampered or mis-served bundle::

        try:
            ReceiverModel.load(path)
        except IntegrityError:
            ...  # re-fit; do not serve
    """


_PUBLIC_COLS = ["ball_dist", "lane_pressure", "space"]
_OWNER_EXTRA_COLS = ["release_dir_align", "closing_speed"]


@dataclass(frozen=True)
class ReceiverParams:
    """Parameters for candidate feature extraction + the geometric proxy.

    Examples
    --------
    >>> ReceiverParams().proxy_cone_deg
    45.0
    """

    lane_half_width_m: float = 3.0  # corridor half-width for lane_pressure
    proxy_cone_deg: float = 45.0  # forward cone for the geometric proxy
    pressure_scale_m: float = 4.0  # exp decay of a defender's lane-pressure contribution


def _acting_attacks_rtl(frame: pd.DataFrame, team_id) -> bool:
    """Does the acting team attack RIGHT-TO-LEFT in this frame's convention? (ADR-028)

    Real tracking frames (e.g. Gradient Sports) are home-attacks-right, and each player row carries
    ``team_attacking_direction``. SB360 snapshot frames and the synthetic probe frames carry no such
    column (or stamp ``"ltr"`` on both teams because a freeze-frame IS already action-LTR), so an absent
    column -> ``False`` -> no reflection. ``is_ball`` via ``to_numpy(dtype=bool)`` NOT ``.astype(bool)``
    on a possibly-object column (ADR-019).
    """
    if "team_attacking_direction" not in frame.columns:
        return False
    prow = frame[ids_match(frame["team_id"], team_id) & ~frame["is_ball"].to_numpy(dtype=bool)]
    return (not prow.empty) and str(prow["team_attacking_direction"].iloc[0]) == "rtl"


def _passer_xy(action_row: pd.Series, frame: pd.DataFrame) -> np.ndarray:
    """The passer position in the FRAME'S coordinate convention.

    ``start_x``/``start_y`` are SPADL action-LTR (the acting team attacks x=105); the frame players are
    in the frame convention. For an away-team action on a home-attacks-right frame these differ by a
    180-degree point reflection (ADR-028), so the action passer is reprojected into the frame -- keeping
    passer and frame-derived positions in ONE coordinate system. Byte-identical on aligned frames
    (SB360 ``ltr``, home GS, the probe frames): the model trains and serves on one convention.
    """
    x, y = float(action_row["start_x"]), float(action_row["start_y"])
    if _acting_attacks_rtl(frame, action_row["team_id"]):
        x, y = 105.0 - x, 68.0 - y
    return np.array([x, y], dtype=np.float64)


def _split_players(frame: pd.DataFrame, action_team, passer_id):
    non_ball = frame[~frame["is_ball"].to_numpy(dtype=bool)]
    is_team = ids_match(non_ball["team_id"], action_team)
    is_passer = ids_match(non_ball["player_id"], passer_id)
    teammates = non_ball[is_team & ~is_passer]
    opp_outfield = non_ball[~is_team & ~non_ball["is_goalkeeper"].astype(bool)]
    return teammates, opp_outfield


def _ball_release_dir(frame: pd.DataFrame) -> np.ndarray:
    """Unit vector of the ball's release velocity -- the only leakage-free release direction.

    Raises if the frame carries no velocity (SB360 freeze frames): the caller must not silently
    substitute the outcome-selected pass-event angle.
    """
    if "vx" not in frame.columns or "vy" not in frame.columns:
        raise KeyError("release direction needs ball velocity (vx/vy) -- absent on this frame set")
    ball = frame[frame["is_ball"].to_numpy(dtype=bool)]
    if ball.empty:
        raise ValueError("frame has no ball row -- cannot derive the release direction")
    v = np.array([float(ball["vx"].iloc[0]), float(ball["vy"].iloc[0])], dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("ball release velocity is ~0 -- no release direction")
    return v / n


def _lane_pressure(passer: np.ndarray, target: np.ndarray, def_pos: np.ndarray, p: ReceiverParams) -> float:
    """Sum of exp-decayed perpendicular distances for defenders inside the passer->target corridor."""
    if def_pos.shape[0] == 0:
        return 0.0
    seg = target - passer
    seg_len = float(np.linalg.norm(seg))
    if seg_len < 1e-9:
        return 0.0
    u = seg / seg_len
    rel = def_pos - passer
    t = rel @ u  # projection along the segment
    perp = np.linalg.norm(rel - np.outer(t, u), axis=1)
    in_corridor = (t >= 0.0) & (t <= seg_len) & (perp <= p.lane_half_width_m)
    if not in_corridor.any():
        return 0.0
    return float(np.exp(-perp[in_corridor] / p.pressure_scale_m).sum())


def receiver_candidate_features(
    action_row: pd.Series,
    frame: pd.DataFrame,
    *,
    params: ReceiverParams | None = None,
    feature_set: str = "public",
) -> pd.DataFrame:
    """One row per passing-team teammate: pre-pass candidate features for intended-receiver scoring.

    Pure (does not mutate ``frame``). ``feature_set="public"`` is positions-only; ``"owner"`` adds the
    velocity-derived features and raises on a velocity-less frame. The action passer (SPADL action-LTR)
    is reprojected into the frame convention via the frame's ``team_attacking_direction`` (ADR-028), so
    passer and frame-derived positions share ONE coordinate system regardless of provider -- an away-team
    action on a home-attacks-right frame is not silently scored across two frames.

    Examples
    --------
    One row per passing-team teammate; the public set is positions-only::

        feats = receiver_candidate_features(action_row, frame, feature_set="public")
        # columns: candidate_id, ball_dist, lane_pressure, space
    """
    if feature_set not in ("public", "owner"):
        raise ValueError(f"feature_set must be 'public' or 'owner', got {feature_set!r}")
    p = params or ReceiverParams()
    passer = _passer_xy(action_row, frame)
    teammates, opp_outfield = _split_players(frame, action_row["team_id"], action_row["player_id"])
    def_pos = opp_outfield[["x", "y"]].to_numpy(dtype=np.float64)

    if feature_set == "owner":
        ball_dir = _ball_release_dir(frame)
        if "vx" not in teammates.columns or "vy" not in teammates.columns:
            raise KeyError("owner feature_set needs candidate velocity (vx/vy)")

    rows = []
    for _, tm in teammates.iterrows():
        cand = np.array([float(tm["x"]), float(tm["y"])], dtype=np.float64)
        ray = cand - passer
        ray_len = float(np.linalg.norm(ray))
        space = float(np.linalg.norm(def_pos - cand, axis=1).min()) if def_pos.shape[0] else np.inf
        rec = {
            "candidate_id": canonical_id(tm["player_id"]),
            "ball_dist": ray_len,
            "lane_pressure": _lane_pressure(passer, cand, def_pos, p),
            "space": space,
        }
        if feature_set == "owner":
            u_ray = ray / ray_len if ray_len > 1e-9 else np.zeros(2)
            vel = np.array([float(tm["vx"]), float(tm["vy"])], dtype=np.float64)
            rec["release_dir_align"] = float(u_ray @ ball_dir)
            rec["closing_speed"] = float(vel @ u_ray)
        rows.append(rec)

    cols = ["candidate_id"] + _PUBLIC_COLS + (_OWNER_EXTRA_COLS if feature_set == "owner" else [])
    return pd.DataFrame(rows, columns=cols)


def geometric_proxy_receiver(
    action_row: pd.Series,
    frame: pd.DataFrame,
    *,
    params: ReceiverParams | None = None,
):
    """Baseline / fallback: the teammate best aligned with the ball's release-velocity ray.

    Requires ball velocity (GS-only). Returns the canonical candidate id, or ``None`` if no teammate
    falls within the forward cone.

    Examples
    --------
    The teammate best aligned with the ball's release-velocity ray::

        rid = geometric_proxy_receiver(action_row, frame)
    """
    p = params or ReceiverParams()
    ball_dir = _ball_release_dir(frame)  # raises on a velocity-less frame
    passer = _passer_xy(action_row, frame)
    teammates, _ = _split_players(frame, action_row["team_id"], action_row["player_id"])
    cos_min = float(np.cos(np.deg2rad(p.proxy_cone_deg)))
    best_id, best_align = None, cos_min
    for _, tm in teammates.iterrows():
        ray = np.array([float(tm["x"]) - passer[0], float(tm["y"]) - passer[1]], dtype=np.float64)
        n = float(np.linalg.norm(ray))
        if n < 1e-9:
            continue
        align = float((ray / n) @ ball_dir)
        if align >= best_align:
            best_id, best_align = canonical_id(tm["player_id"]), align
    return best_id


def variant_key_for_provider(provider: str) -> str:
    """Provider -> bundled receiver variant key. SkillCorner/GS get an owner variant if one is
    bundled; everything else the public default (resolved against what actually ships).

    Examples
    --------
    >>> variant_key_for_provider("gradientsports")
    'gs_owner'
    >>> variant_key_for_provider("statsbomb")
    'default'
    """
    return {"gradientsports": "gs_owner", "skillcorner": "skillcorner"}.get(provider, "default")


class ReceiverModel:
    """Per-candidate binary logistic P(intended receiver). sklearn at fit; pure-numpy at serve.

    ``feature_set`` fixes the feature columns (``"public"`` positions-only, ``"owner"`` + velocity),
    and is recorded in the artifact so a bundle can never be served with the wrong feature vector.

    Examples
    --------
    Fit on completed passes (observed receiver = label), then rank a pass's candidates::

        model = ReceiverModel("public").fit(candidate_features, labels)
        ranked = model.rank(action_row, frame)  # candidate_id -> P(intended), highest first
    """

    VERSION = "1.0.0"

    def __init__(self, feature_set: str = "public") -> None:
        if feature_set not in ("public", "owner"):
            raise ValueError(f"feature_set must be 'public' or 'owner', got {feature_set!r}")
        self.feature_set = feature_set
        self.feature_names = list(_PUBLIC_COLS + (_OWNER_EXTRA_COLS if feature_set == "owner" else []))
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.shipped_variant: str | None = None
        self.params = ReceiverParams()

    # ---- fit ----
    def fit(self, features: pd.DataFrame, labels) -> ReceiverModel:
        """Fit the per-candidate logistic on candidate features + binary intended-receiver labels.

        Examples
        --------
        Train on completed-pass candidate rows (label 1 = the observed receiver)::

            model = ReceiverModel("public").fit(candidate_features, labels)
        """
        from sklearn.linear_model import LogisticRegression

        X = features[self.feature_names].to_numpy(float)
        y = np.asarray(labels, dtype=int)
        # Sanitize non-finite to NaN so nanmean/nanstd IGNORE them (symmetric with predict_candidates,
        # which fills non-finite with the stored mean). `space` is +inf for a candidate with no opponent
        # outfield players (L7); an inf poisons np.nanmean to inf and sklearn then raises on the NaN.
        X = np.where(np.isfinite(X), X, np.nan)
        mean = np.nanmean(X, axis=0)
        std_raw = np.nanstd(X, axis=0)
        std = np.where(std_raw > 1e-9, std_raw, 1.0)
        Xf = np.where(np.isfinite(X), X, mean[None, :])
        Xs = (Xf - mean) / std
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs").fit(Xs, y)
        self._coef = clf.coef_[0].astype(float)
        self._intercept = float(clf.intercept_[0])
        self._mean, self._std = mean, std
        return self

    # ---- serve (pure numpy) ----
    def predict_candidates(self, features: pd.DataFrame) -> np.ndarray:
        """Per-row P(intended) for a candidate-feature frame (pure-numpy sigmoid, no sklearn).

        Examples
        --------
        Score each teammate row of a candidate-feature frame::

            probs = model.predict_candidates(candidate_features)
        """
        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("ReceiverModel not fitted/loaded.")
        X = features[self.feature_names].to_numpy(float)
        Xf = np.where(np.isfinite(X), X, self._mean[None, :])
        Xs = (Xf - self._mean) / self._std
        return 1.0 / (1.0 + np.exp(-(Xs @ self._coef + self._intercept)))

    def rank(self, action_row: pd.Series, frame: pd.DataFrame) -> pd.Series:
        """Candidate id -> P(intended), highest first. ``argmax`` = the intended receiver.

        Examples
        --------
        The intended receiver is the top-ranked candidate::

            ranked = model.rank(action_row, frame)
            intended = ranked.index[0]
        """
        feats = receiver_candidate_features(action_row, frame, params=self.params, feature_set=self.feature_set)
        if feats.empty:
            return pd.Series(dtype=float)
        p = self.predict_candidates(feats)
        return pd.Series(p, index=feats["candidate_id"].to_numpy()).sort_values(ascending=False)

    # ---- chirality / contract probes ----
    def _probe_action(self, frame: pd.DataFrame) -> pd.Series:
        # Derive the probe passer's team from a real OUTFIELD row, NEVER the ball. A ball belongs to no
        # team, so on the feature-contract probe frame its team_id is None; a None team makes every
        # `ids_match(team_id, None)` False -> zero teammates -> zero candidate rows -> an EMPTY contract
        # fingerprint, i.e. the ADR-011 feature-value prong is silently inert. The first non-ball row is
        # the attacking team on both probe frames, so this is byte-identical for chirality and non-empty
        # for the contract.
        non_ball = frame[~frame["is_ball"].to_numpy(dtype=bool)]
        ball = frame[frame["is_ball"].to_numpy(dtype=bool)].iloc[0]
        return pd.Series(
            {
                "player_id": "__probe_passer__",
                "team_id": non_ball["team_id"].iloc[0],
                "start_x": ball["x"],
                "start_y": ball["y"],
            }
        )

    def _chirality_predict(self, frame: pd.DataFrame) -> np.ndarray:
        # NOTE: every receiver feature is reflection-INVARIANT under a CONSISTENT whole-scene reflection
        # (ball_dist/space/lane_pressure are distances; the owner release_dir_align/closing_speed are dot
        # products of vectors that both flip). So a y-mirror in the extractor cannot move this fingerprint,
        # and the probe here keeps passer + frame in ONE convention (the passer is the frame's ball), so it
        # does NOT exercise the ADR-028 action-vs-frame MIX that a real away-team serve produces. That mix
        # is handled at extraction time by `_passer_xy` reprojecting the action passer into the frame; it
        # is NOT what this guard protects. Chirality here is a coarse predict-drift check (atol=1e-3); the
        # fine-grained feature-value guard is the feature contract.
        feats = receiver_candidate_features(
            self._probe_action(frame), frame, params=self.params, feature_set=self.feature_set
        )
        return self.predict_candidates(feats)

    def _contract_extract(self) -> np.ndarray:
        from silly_kicks.tracking._feature_contract import contract_probe_frame

        frame = contract_probe_frame()
        feats = receiver_candidate_features(
            self._probe_action(frame), frame, params=self.params, feature_set=self.feature_set
        )
        return feats[self.feature_names].to_numpy(float).ravel()

    def _contract_constants(self) -> dict[str, float]:
        # Only constants LOAD-BEARING on the served extractor (ADR-050): lane_half_width_m + pressure_scale_m
        # both shape `lane_pressure`. proxy_cone_deg is DELIBERATELY absent -- it drives only the geometric
        # proxy baseline, never a served feature, so declaring it would spuriously invalidate a bundle on a
        # proxy-only change. It still round-trips losslessly via the serialized `params` block.
        return {
            "lane_half_width_m": self.params.lane_half_width_m,
            "pressure_scale_m": self.params.pressure_scale_m,
        }

    # ---- serialization (pickle-free JSON envelope) ----
    def to_dict(self) -> dict:
        """Pickle-free JSON envelope (coef/mean/std + feature_set + chirality + feature-contract blocks).

        Examples
        --------
        The serialisable payload embedded in a bundle::

            payload = model.to_dict()
        """
        from silly_kicks.tracking._chirality import chirality_fingerprint
        from silly_kicks.tracking._feature_contract import feature_contract

        if self._coef is None or self._mean is None or self._std is None:
            raise RuntimeError("ReceiverModel not fitted/loaded; nothing to serialize.")
        return {
            "version": self.VERSION,
            "feature_set": self.feature_set,
            "feature_names": self.feature_names,
            "coef": self._coef.tolist(),
            "intercept": self._intercept,
            "mean": self._mean.tolist(),
            "std": self._std.tolist(),
            "shipped_variant": self.shipped_variant,
            "params": asdict(self.params),
            "chirality": chirality_fingerprint(self._chirality_predict),
            "feature_contract": feature_contract(self._contract_extract, constants=self._contract_constants()),
        }

    @classmethod
    def from_dict(cls, d: dict) -> ReceiverModel:
        """Rebuild a model from its :meth:`to_dict` payload (no integrity checks -- see :meth:`load`).

        Examples
        --------
        Round-trip through the JSON payload::

            model = ReceiverModel.from_dict(payload)
        """
        m = cls(feature_set=d.get("feature_set", "public"))
        m.feature_names = list(d["feature_names"])
        m._coef = np.asarray(d["coef"], dtype=float)
        m._intercept = float(d["intercept"])
        m._mean = np.asarray(d["mean"], dtype=float)
        m._std = np.asarray(d["std"], dtype=float)
        m.shipped_variant = d.get("shipped_variant")
        # Restore params (M2): without this a non-default-params bundle reloads with default params, so
        # verify_feature_contract compares default constants against the stored ones and RAISES (a bundle
        # that cannot load), and rank()/proxy would extract with the wrong params (train/serve skew).
        if d.get("params"):
            m.params = ReceiverParams(**d["params"])
        return m

    @staticmethod
    def _sha(path: Path) -> str:
        text = (path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def save(self, path: Path | str) -> None:
        """Write ``model.json`` + ``SHA256SUMS`` to a bundle directory.

        Examples
        --------
        Persist a fitted model::

            model.save("silly_kicks/tracking/_receiver_weights/default")
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.json").write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        (path / "SHA256SUMS").write_text(f"{self._sha(path)}  model.json\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str, *, legacy_override: bool = False) -> ReceiverModel:
        """Load a bundle, verifying SHA + chirality + feature contract (raises :class:`IntegrityError`).

        Examples
        --------
        Load a bundled variant directory::

            model = ReceiverModel.load("silly_kicks/tracking/_receiver_weights/default")
        """
        from silly_kicks.tracking._chirality import verify_chirality
        from silly_kicks.tracking._feature_contract import verify_feature_contract

        path = Path(path)
        want = (path / "SHA256SUMS").read_text(encoding="utf-8").split()[0]
        if want != cls._sha(path):
            raise IntegrityError(f"ReceiverModel integrity check failed at {path}")
        d = json.loads((path / "model.json").read_text(encoding="utf-8"))
        m = cls.from_dict(d)
        verify_chirality(
            chirality_fingerprint_of(m),
            d.get("chirality"),
            legacy_override=legacy_override,
            model_name="ReceiverModel",
            error_cls=IntegrityError,
        )
        verify_feature_contract(
            m._feature_contract_block(),
            d.get("feature_contract"),
            legacy_override=legacy_override,
            model_name="ReceiverModel",
            error_cls=IntegrityError,
        )
        return m

    def _feature_contract_block(self) -> dict:
        from silly_kicks.tracking._feature_contract import feature_contract

        return feature_contract(self._contract_extract, constants=self._contract_constants())

    @classmethod
    def from_variant(cls, variant: str = "default") -> ReceiverModel:
        """Load a bundled variant by key (cached); ``variant_key_for_provider`` maps a provider to one.

        Examples
        --------
        Serve the variant a provider should use::

            model = ReceiverModel.from_variant(variant_key_for_provider("statsbomb"))
        """
        if variant in _VARIANT_CACHE:
            return _VARIANT_CACHE[variant]
        wdir = _WEIGHTS_ROOT / variant
        if not (wdir / "SHA256SUMS").exists():
            raise FileNotFoundError(
                f"No bundled receiver weights for {variant!r} at {wdir}. Train via scripts/train_receiver_model.py."
            )
        m = cls.load(wdir)
        _VARIANT_CACHE[variant] = m
        return m


def chirality_fingerprint_of(model: ReceiverModel) -> dict:
    """The behavioural chirality fingerprint used at load to detect a y-mirror-mis-served artifact.

    Examples
    --------
    The fingerprint block embedded in a saved bundle::

        fp = chirality_fingerprint_of(model)  # {version, frame_sha256, outputs}
    """
    from silly_kicks.tracking._chirality import chirality_fingerprint

    return chirality_fingerprint(model._chirality_predict)


def _link_and_index(actions, frames, links):
    from silly_kicks.tracking import link_actions_to_frames

    if links is None:
        links, _ = link_actions_to_frames(actions, frames)
    # Canonicalize the action_id key on BOTH sides (ADR-019): a caller-supplied `links` whose action_id
    # dtype differs from `actions` (e.g. links str, actions int64) would otherwise miss every lookup and
    # return an all-NA receiver column with no error -- the exact silent id-dtype miss ADR-019 forbids.
    frame_of = {
        canonical_id(aid): fid
        for aid, fid in zip(links["action_id"].to_numpy(), links["frame_id"].to_numpy(), strict=True)
    }
    by_frame = {canonical_id(fid): g for fid, g in frames.groupby("frame_id")}
    return frame_of, by_frame


def resolve_intended_receiver(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: ReceiverModel | None = None,
    links: pd.DataFrame | None = None,
) -> pd.Series:
    """Intended-receiver ``player_id`` per action (``model=None`` -> the geometric proxy).

    Pure (no input mutation). An action whose frame is unlinked, or whose inference yields nothing, gets
    ``pd.NA``. NOTE: ``model=None`` uses the geometric proxy, which REQUIRES ball velocity (the only
    leakage-free release direction) and RAISES on a velocity-less frame set (e.g. SB360 freeze frames) --
    it does not silently substitute the outcome-selected pass-event angle. A public ``model`` is
    positions-only and velocity-free.

    Examples
    --------
    Intended-receiver id per action (``model=None`` uses the geometric proxy)::

        receivers = resolve_intended_receiver(actions, frames, model=model)
    """
    frame_of, by_frame = _link_and_index(actions, frames, links)
    out: dict = {}
    for _, act in actions.iterrows():
        fr = by_frame.get(canonical_id(frame_of.get(canonical_id(act["action_id"]))))
        if fr is None:
            out[act["action_id"]] = pd.NA
            continue
        if model is None:
            rid = geometric_proxy_receiver(act, fr)
            out[act["action_id"]] = pd.NA if rid is None else rid  # normalize proxy miss to pd.NA (L6)
        else:
            ranked = model.rank(act, fr)
            out[act["action_id"]] = ranked.index[0] if not ranked.empty else pd.NA
    return pd.Series(out, name="intended_receiver_id")


def intended_receiver_positions(
    actions: pd.DataFrame,
    frames: pd.DataFrame,
    *,
    model: ReceiverModel | None = None,
    links: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """``(action_id, x, y, source)`` -- the intended receiver's frame position, for de-leaking the
    failed-pass target. ``source`` in ``{intended_receiver, geometric_proxy}``. Pure; NaN on a miss.

    Examples
    --------
    The intended receiver's frame position, for de-leaking the failed-pass target::

        pos = intended_receiver_positions(actions, frames, model=model)
        # columns: action_id, receiver_id, x, y, source
    """
    frame_of, by_frame = _link_and_index(actions, frames, links)
    source = "geometric_proxy" if model is None else "intended_receiver"
    rows = []
    for _, act in actions.iterrows():
        aid = act["action_id"]
        fr = by_frame.get(canonical_id(frame_of.get(canonical_id(aid))))
        x = y = np.nan
        rid = pd.NA
        if fr is not None:
            if model is None:
                rid = geometric_proxy_receiver(act, fr)
                if rid is None:
                    rid = pd.NA  # normalize proxy miss to pd.NA (L6)
            else:
                ranked = model.rank(act, fr)
                rid = ranked.index[0] if not ranked.empty else pd.NA
            if rid is not pd.NA:
                match = fr[ids_match(fr["player_id"], rid) & ~fr["is_ball"].to_numpy(dtype=bool)]
                if not match.empty:
                    x, y = float(match["x"].iloc[0]), float(match["y"].iloc[0])
        rows.append({"action_id": aid, "receiver_id": rid, "x": x, "y": y, "source": source})
    return pd.DataFrame(rows, columns=["action_id", "receiver_id", "x", "y", "source"])
