"""TF-79 contact scorer (scorer spec 2026-11-15 r7, "scorer v2"; ADR-134 amendments).

The credit step of the ball-contact detector, learned: for each merged kinematic peak, the ``K_OPTIONS`` nearest
players are the credit options. A gradient-boosted tree ensemble gives each option the probability that the peak is a
real touch by that player, from the peak's kinematics and the option's control features (what the ball and the player
do in the ``CONTROL_WINDOW_S`` before and after the peak). A first-touch decoding (temporal non-maximum suppression
over ``decode_window_s``) keeps, of the arrival and the control peak of one reception, the more probable one.

Fitted with scikit-learn and exported to plain tree arrays; served with numpy only; parameters only, pickle-free. The
artifact carries the ADR-050 feature contract on a committed trajectory probe, so a change to a constant or to the
extractor makes ``load()`` refuse.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from silly_kicks.tracking._ball_contact import (
    BallContactParams,
    _candidate_peaks,
    _merged_peaks,
)
from silly_kicks.tracking._feature_contract import feature_contract, verify_feature_contract

__all__ = [
    "CONTROL_FEATURE_NAMES",
    "CONTROL_WINDOW_S",
    "DECODE_WINDOW_S",
    "FEATURE_NAMES",
    "GAP_CAP_M",
    "HGB_SETTINGS",
    "K_OPTIONS",
    "ContactScorer",
    "ContactScorerIntegrityError",
    "best_options",
    "contact_scorer_probe_frames",
    "control_features",
    "credit_options",
    "decide_options",
    "feature_contract_for",
    "option_rows",
    "option_table",
    "resolve_scorer",
]

#: The credit options per peak: the K nearest players (scorer spec 2.1; the receiver is among the 3 nearest on
#: 83.9 % of PFF receptions on the development half).
K_OPTIONS = 3
#: The cap of the ``gap`` feature, and its value when no next player exists (m).
GAP_CAP_M = 5.0
#: The half-window (s) of the control features (scorer spec r7 2.2): the ball's and the option's behaviour in the
#: ``CONTROL_WINDOW_S`` before and after the peak.
CONTROL_WINDOW_S = 0.4
#: The first-touch decoding window (s) written into ``model.json`` at fit (scorer spec r7 2.1). Serving reads the
#: value in ``model.json``, never this constant: the model file is the single source of truth.
DECODE_WINDOW_S = 0.3
#: The peak and option features of v1 (scorer spec 2.2), computed from the merged peaks.
_PEAK_FEATURE_NAMES: tuple[str, ...] = ("dv", "speed_in", "speed_out", "turn", "dist", "rank", "gap", "approach", "z")
#: The control features (scorer spec r7 2.2), computed from the frames around each peak.
CONTROL_FEATURE_NAMES: tuple[str, ...] = (
    "after_ball_speed",
    "after_rel_disp",
    "after_dist_mean",
    "after_dist_end",
    "before_dist_min",
    "before_rel_disp",
    "speed_drop",
)
#: The scorer's inputs, in model order (16).
FEATURE_NAMES: tuple[str, ...] = (*_PEAK_FEATURE_NAMES, *CONTROL_FEATURE_NAMES)
#: The model settings (scorer spec r7 2.3): fixed, not tuned; no early stopping and a fixed seed make a fit
#: deterministic.
HGB_SETTINGS: dict[str, Any] = {
    "max_iter": 200,
    "learning_rate": 0.1,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 0.0,
    "early_stopping": False,
    "random_state": 0,
}
#: The exported arrays of one tree (no per-sample statistic such as ``count`` or ``gain``).
_TREE_KEYS = ("feature", "threshold", "left", "right", "missing_left", "is_leaf", "value")
#: The numpy serving must equal scikit-learn's ``predict_proba`` within this at fit (scorer spec r7 2.3).
_PARITY_TOL = 1e-12
SCORER_VERSION = "contact-scorer-2"
_FILES = ("model.json",)
_DEFAULT_DIR = Path(__file__).resolve().parent / "_contact_scorer_weights" / "default"


class ContactScorerIntegrityError(RuntimeError):
    """A contact-scorer artifact failed its checksum or its feature contract."""


def option_rows(merged: pd.DataFrame) -> pd.DataFrame:
    """One row per (merged peak, option) with the v1 features of scorer spec 2.2 and the ids (``peak_id``,
    ``period_id``, ``frame_id``, ``time_seconds``, ``player_id``). ``merged`` carries the option columns of
    :func:`._ball_contact._candidate_peaks` with ``k_options = K_OPTIONS``. A non-measured or missing height is a
    missing value (NaN; scorer spec r7 2.2). :func:`control_features` adds the control features."""
    k = K_OPTIONS
    z = np.where(
        (merged["z_source"].astype(object) == "measured").to_numpy() & np.isfinite(merged["z"].to_numpy(dtype=float)),
        merged["z"].to_numpy(dtype=float),
        np.nan,
    )
    parts = []
    for r in range(k):
        has = merged[f"opt{r}_player_id"].notna().to_numpy()
        if not has.any():
            continue
        dist = merged[f"opt{r}_dist"].to_numpy(dtype=float)
        nxt = merged[f"opt{r + 1}_dist"].to_numpy(dtype=float)
        gap = np.where(np.isfinite(nxt), np.minimum(nxt - dist, GAP_CAP_M), GAP_CAP_M)
        part = pd.DataFrame(
            {
                "peak_id": merged["peak_id"].to_numpy(),
                "period_id": merged["period_id"].to_numpy(),
                "frame_id": merged["frame_id"].to_numpy(),
                "time_seconds": merged["time_seconds"].to_numpy(dtype=float),
                "player_id": merged[f"opt{r}_player_id"].to_numpy(dtype=object),
                "dv": merged["dv"].to_numpy(dtype=float),
                "speed_in": merged["speed_in"].to_numpy(dtype=float),
                "speed_out": merged["speed_out"].to_numpy(dtype=float),
                "turn": merged["turn"].to_numpy(dtype=float),
                "dist": dist,
                "rank": float(r),
                "gap": gap,
                "approach": merged[f"opt{r}_approach"].to_numpy(dtype=float),
                "z": z,
            }
        )
        parts.append(part[has])
    cols = ["peak_id", "period_id", "frame_id", "time_seconds", "player_id", *_PEAK_FEATURE_NAMES]
    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True).sort_values(["peak_id", "rank"], kind="stable")[cols]


def _path_speed(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """The summed horizontal steps over the elapsed time; missing with fewer than 2 samples, no elapsed time, or a
    non-finite ball sample in the set (scorer spec r7 2.2)."""
    if len(t) < 2 or not (np.isfinite(x).all() and np.isfinite(y).all()):
        return float("nan")
    elapsed = float(t[-1] - t[0])
    if elapsed <= 0.0:
        return float("nan")
    return float(np.hypot(np.diff(x), np.diff(y)).sum() / elapsed)


def _rel_rate(bx: np.ndarray, by: np.ndarray, p: np.ndarray, t: np.ndarray) -> float:
    """|(ball_last - ball_first) - (option_last - option_first)| / elapsed, between the first and the last sample at
    which both are finite; missing with fewer than 2 such samples or no elapsed time (scorer spec r7 2.2)."""
    ok = np.flatnonzero(np.isfinite(bx) & np.isfinite(by) & np.isfinite(p[:, 0]) & np.isfinite(p[:, 1]))
    if len(ok) < 2:
        return float("nan")
    i0, i1 = int(ok[0]), int(ok[-1])
    elapsed = float(t[i1] - t[i0])
    if elapsed <= 0.0:
        return float("nan")
    dx = (bx[i1] - bx[i0]) - (p[i1, 0] - p[i0, 0])
    dy = (by[i1] - by[i0]) - (p[i1, 1] - p[i0, 1])
    return float(np.hypot(dx, dy) / elapsed)


def _option_control(
    k: int, t0: float, track: np.ndarray, tt: np.ndarray, bx: np.ndarray, by: np.ndarray, c: float
) -> list[float]:
    """The seven control features of one option at the deduped ball sample ``k`` (scorer spec r7 2.2)."""
    hi = int(np.searchsorted(tt, t0 + c + 1e-9, side="right"))
    lo = int(np.searchsorted(tt, t0 - c - 1e-9, side="left"))
    after = np.arange(k, max(hi, k + 1))  # k, then every sample with t < t_s <= t + C
    before = np.arange(lo, k + 1)  # every sample with t - C <= t_s <= t, k included
    with np.errstate(invalid="ignore"):
        d_after = np.hypot(bx[after] - track[after, 0], by[after] - track[after, 1])
        d_before = np.hypot(bx[before] - track[before, 0], by[before] - track[before, 1])
    tail = d_after[1:][np.isfinite(d_after[1:])]  # the after samples except k, with a distance
    head = d_before[np.isfinite(d_before)]
    s_after = _path_speed(bx[after], by[after], tt[after])
    s_before = _path_speed(bx[before], by[before], tt[before])
    return [
        s_after,
        _rel_rate(bx[after], by[after], track[after], tt[after]),
        float(tail.mean()) if len(tail) else float("nan"),
        float(tail[-1]) if len(tail) else float("nan"),
        float(head.min()) if len(head) else float("nan"),
        _rel_rate(bx[before], by[before], track[before], tt[before]),
        s_before - s_after if np.isfinite(s_before) and np.isfinite(s_after) else float("nan"),
    ]


def control_features(opts: pd.DataFrame, frames: pd.DataFrame, params: BallContactParams) -> pd.DataFrame:
    """``opts`` (from :func:`option_rows`) with the seven control features of scorer spec r7 2.2 appended, read from
    the same deduped ball samples as the peak search (``_ball_samples``, ADR-125) and each option's frame rows; a
    distance needs a finite ball and a finite option at that sample. Positions only; no provider velocity column."""
    from silly_kicks._frame_index import group_rows
    from silly_kicks.id_compat import canonical_id
    from silly_kicks.tracking._ball_contact import _ball_samples

    out = np.full((len(opts), len(CONTROL_FEATURE_NAMES)), np.nan)
    if len(opts):
        c = float(CONTROL_WINDOW_S)
        is_ball = frames["is_ball"].to_numpy(dtype=bool)
        ball_by_period = group_rows(frames.loc[is_ball], "period_id")
        players = frames.loc[~is_ball & frames["player_id"].notna().to_numpy()]
        players_by_period = group_rows(players, "period_id")
        o_period = opts["period_id"].to_numpy()
        o_frame = opts["frame_id"].to_numpy()
        o_time = opts["time_seconds"].to_numpy(dtype=float)
        o_player = [str(canonical_id(v)) for v in opts["player_id"]]
        for period in pd.unique(o_period):
            seg = ball_by_period.get(period)
            if seg is None:
                continue
            fid, tt, bx, by, keep, _fp, _span = _ball_samples(seg.sort_values("frame_id"), params)
            fid, tt, bx, by = fid[keep], tt[keep], bx[keep], by[keep]
            pos = {int(f): i for i, f in enumerate(fid)}
            tracks: dict[str, np.ndarray] = {}
            pl = players_by_period.get(period)
            if pl is not None:
                for pid, g in pl.groupby("player_id", sort=False, observed=True):
                    idx = np.array([pos.get(int(f), -1) for f in g["frame_id"].to_numpy()], dtype=np.int64)
                    hit = idx >= 0
                    track = np.full((len(fid), 2), np.nan)
                    track[idx[hit]] = g[["x", "y"]].to_numpy(dtype=float)[hit]
                    tracks[str(canonical_id(pid))] = track
            for i in np.flatnonzero(o_period == period):
                k = pos.get(int(o_frame[i]))
                track = tracks.get(o_player[i])
                if k is None or track is None:
                    continue
                out[i] = _option_control(k, float(o_time[i]), track, tt, bx, by, c)
    return opts.assign(**{name: out[:, j] for j, name in enumerate(CONTROL_FEATURE_NAMES)})


def best_options(opts: pd.DataFrame, prob: np.ndarray) -> pd.DataFrame:
    """Per peak, the option with the highest probability (ties: the nearer rank): ``peak_id``, ``period_id``,
    ``time_seconds``, ``player_id``, ``prob``, ``z``."""
    if len(opts) == 0:
        return pd.DataFrame(columns=["peak_id", "period_id", "time_seconds", "player_id", "prob", "z"])
    o = opts.assign(prob=np.asarray(prob, dtype=float))
    top = o.sort_values(["peak_id", "prob", "rank"], ascending=[True, False, True], kind="stable")
    return top.drop_duplicates("peak_id")[["peak_id", "period_id", "time_seconds", "player_id", "prob", "z"]]


def _decode(best: pd.DataFrame, window_s: float) -> np.ndarray:
    """The first-touch decoding (scorer spec r7 2.1 step 7), aligned to the rows of ``best``: a peak survives when no
    other peak of its period within +-``window_s`` (inclusive) has a higher best-option probability, and no peak with
    an equal one lies earlier in that window. Only peaks with options are in ``best``, so a peak without options
    neither suppresses nor is suppressed."""
    keep = np.ones(len(best), dtype=bool)
    if len(best) == 0:
        return keep
    period = best["period_id"].to_numpy()
    t_all = best["time_seconds"].to_numpy(dtype=float)
    p_all = best["prob"].to_numpy(dtype=float)
    for per in pd.unique(period):
        rows = np.flatnonzero(period == per)
        rows = rows[np.argsort(t_all[rows], kind="stable")]
        t, p = t_all[rows], p_all[rows]
        lo = np.searchsorted(t, t - window_s - 1e-9, side="left")
        hi = np.searchsorted(t, t + window_s + 1e-9, side="right")
        for j in range(len(rows)):
            keep[rows[j]] = lo[j] + int(np.argmax(p[lo[j] : hi[j]])) == j  # the first maximum of the window
    return keep


def decide_options(
    opts: pd.DataFrame, prob: np.ndarray, params: BallContactParams, *, decode_window_s: float
) -> pd.DataFrame:
    """THE credit decision of the scorer (scorer spec r7 2.1 steps 6-8), shared by the detector and the scorer
    driver: per peak, the best option; the first-touch decoding over ``decode_window_s`` (the loaded scorer's value,
    from its ``model.json``); ``credited`` when the peak survives the decoding, its probability is at least
    ``min_contact_probability`` and the ball is not measured above ``contact_max_z_m`` (``z`` is NaN unless
    measured)."""
    b = best_options(opts, prob)
    z = b["z"].to_numpy(dtype=float)
    ok = (b["prob"].to_numpy(dtype=float) >= params.min_contact_probability) & ~(
        np.isfinite(z) & (z > params.contact_max_z_m)
    )
    return b.assign(credited=ok & _decode(b, float(decode_window_s)))


def credit_options(
    opts: pd.DataFrame, prob: np.ndarray, params: BallContactParams, *, decode_window_s: float
) -> pd.DataFrame:
    """:func:`decide_options` as the contact columns the next-touch measure reads: ``period_id``, ``time_seconds``,
    ``player_id`` (None when unresolved) and ``contact_source``."""
    b = decide_options(opts, prob, params, decode_window_s=decode_window_s)
    ok = b["credited"].to_numpy(dtype=bool)
    return pd.DataFrame(
        {
            "period_id": b["period_id"].to_numpy(),
            "time_seconds": b["time_seconds"].to_numpy(dtype=float),
            "player_id": pd.array(np.where(ok, b["player_id"].to_numpy(dtype=object), None), dtype=object),
            "contact_source": np.where(ok, "velocity_break", "unresolved"),
        }
    )


def option_table(frames: pd.DataFrame, params: BallContactParams) -> pd.DataFrame:
    """The scored path from frames: the threshold-free peaks with ``K_OPTIONS`` options, the threshold and the merge
    of ``params``, :func:`option_rows`, then :func:`control_features`. The detector, the scorer driver and the
    contract probe share it."""
    opts = option_rows(_merged_peaks(_candidate_peaks(frames, params, k_options=K_OPTIONS), params))
    return control_features(opts, frames, params)


def contact_scorer_probe_frames() -> pd.DataFrame:
    """The committed trajectory probe of the feature contract (scorer spec 2.3): 25 Hz, a ball at rest kicked along
    +x at 15 m/s at 1.0 s and deflected towards +y at 1.6 s, four static players of two teams, measured z."""
    fps = 25.0
    legs = [
        (0.0, 1.0, (60.0, 34.0), (60.0, 34.0)),
        (1.0, 1.6, (60.0, 34.0), (69.0, 34.0)),
        (1.6, 2.2, (69.0, 34.0), (69.0, 40.0)),
    ]
    players = {1: ("A", 59.6, 34.3), 2: ("B", 69.4, 34.4), 3: ("B", 66.0, 30.0), 4: ("A", 72.0, 38.0)}
    rows = []
    for k in range(round(legs[-1][1] * fps) + 1):
        t = k / fps
        x = y = float("nan")
        for t0, t1, (x0, y0), (x1, y1) in legs:
            if t0 - 1e-9 <= t <= t1 + 1e-9:
                f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
                x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                break
        base = {"game_id": 1, "period_id": 1, "frame_id": k, "time_seconds": t, "frame_rate": fps}
        rows.append(
            {
                **base,
                "player_id": pd.NA,
                "team_id": pd.NA,
                "is_ball": True,
                "x": x,
                "y": y,
                "z": 0.1,
                "z_source": "measured",
            }
        )
        for pid, (team, px, py) in players.items():
            rows.append(
                {
                    **base,
                    "player_id": pid,
                    "team_id": team,
                    "is_ball": False,
                    "x": px,
                    "y": py,
                    "z": 0.0,
                    "z_source": "absent",
                }
            )
    out = pd.DataFrame(rows)
    out["player_id"] = out["player_id"].astype("Int64")
    return out


def feature_contract_for(params: BallContactParams) -> dict:
    """The ADR-050 contract of the scorer's extractor under ``params``: the feature matrix on the trajectory probe
    and the five declared feature-affecting constants (``K_OPTIONS``, ``GAP_CAP_M``, ``CONTROL_WINDOW_S`` and the two
    detector fields). The decoding window is not one: it changes no feature, and ``model.json`` holds it.

    The fingerprint covers the threshold-free extractor: only the two declared fields are taken from ``params``,
    every other field keeps its default and the pre-filter is 0, so a caller's ``min_velocity_change_ms`` (a
    pre-filter, not a feature) can neither change the contract at save nor fail it at load."""
    probe = contact_scorer_probe_frames()
    canonical = BallContactParams(
        kinematic_window_s=params.kinematic_window_s,
        min_candidate_speed_ms=params.min_candidate_speed_ms,
        min_velocity_change_ms=0.0,
    )
    return feature_contract(
        lambda: option_table(probe, canonical)[list(FEATURE_NAMES)].to_numpy(dtype=float),
        constants={
            "K_OPTIONS": float(K_OPTIONS),
            "GAP_CAP_M": float(GAP_CAP_M),
            "CONTROL_WINDOW_S": float(CONTROL_WINDOW_S),
            "kinematic_window_s": float(params.kinematic_window_s),
            "min_candidate_speed_ms": float(params.min_candidate_speed_ms),
        },
        probe=probe,
    )


def checksums(files: dict[str, bytes]) -> bytes:
    """``SHA256SUMS`` over ``files`` (name -> bytes), in name order."""
    return ("\n".join(f"{hashlib.sha256(b).hexdigest()}  {n}" for n, b in sorted(files.items())) + "\n").encode()


def _write_checksums(d: Path) -> None:
    (d / "SHA256SUMS").write_bytes(checksums({f: (d / f).read_bytes() for f in _FILES}))


def _verify_checksums(d: Path) -> None:
    sums = d / "SHA256SUMS"
    if not sums.exists():
        raise ContactScorerIntegrityError(f"{d}: no SHA256SUMS")
    listed = {}
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, _, name = line.partition("  ")
        listed[name] = digest
    if "model.json" not in listed:
        raise ContactScorerIntegrityError(f"{d}: SHA256SUMS does not cover model.json")
    for name, digest in listed.items():
        if hashlib.sha256((d / name).read_bytes()).hexdigest() != digest:
            raise ContactScorerIntegrityError(f"{d / name}: checksum differs from SHA256SUMS")


def _export_trees(clf: Any) -> tuple[list[dict[str, list]], float]:
    """The fitted ``HistGradientBoostingClassifier`` as plain arrays (scorer spec r7 2.3): per tree the
    ``_TREE_KEYS`` arrays, without any per-sample statistic (``count``, ``gain``), plus the baseline raw score. It reads
    scikit-learn's tree nodes, an internal, at fit time only; :meth:`ContactScorer.fit` asserts that the numpy serving
    of the result equals ``predict_proba``, so a release that changes the internals fails at the fit."""
    trees: list[dict[str, list]] = []
    for per_iteration in clf._predictors:
        if len(per_iteration) != 1:
            raise ContactScorerIntegrityError("the contact scorer is a binary model: one tree per iteration")
        nodes = per_iteration[0].nodes
        if "is_categorical" in (nodes.dtype.names or ()) and bool(np.any(nodes["is_categorical"])):
            raise ContactScorerIntegrityError("the contact scorer has no categorical feature")
        trees.append(
            {
                "feature": nodes["feature_idx"].astype(np.int64).tolist(),
                "threshold": nodes["num_threshold"].astype(np.float64).tolist(),
                "left": nodes["left"].astype(np.int64).tolist(),
                "right": nodes["right"].astype(np.int64).tolist(),
                "missing_left": nodes["missing_go_to_left"].astype(bool).tolist(),
                "is_leaf": nodes["is_leaf"].astype(bool).tolist(),
                "value": nodes["value"].astype(np.float64).tolist(),
            }
        )
    return trees, float(np.ravel(clf._baseline_prediction)[0])


def _tree_arrays(trees: list[dict[str, list]]) -> list[tuple[np.ndarray, ...]]:
    """The serving form of the exported trees; refuses a tree whose arrays differ in length."""
    out = []
    for i, tree in enumerate(trees):
        if set(tree) != set(_TREE_KEYS) or len({len(tree[k]) for k in _TREE_KEYS}) != 1:
            raise ContactScorerIntegrityError(f"tree {i} is malformed (keys {sorted(tree)})")
        out.append(
            (
                np.asarray(tree["feature"], dtype=np.int64),
                np.asarray(tree["threshold"], dtype=np.float64),
                np.asarray(tree["left"], dtype=np.int64),
                np.asarray(tree["right"], dtype=np.int64),
                np.asarray(tree["missing_left"], dtype=bool),
                np.asarray(tree["is_leaf"], dtype=bool),
                np.asarray(tree["value"], dtype=np.float64),
            )
        )
    return out


def _check_decode_window(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
        raise ContactScorerIntegrityError(f"decode_window_s must be a finite number >= 0, not {value!r}")
    return float(value)


class ContactScorer:
    """Gradient-boosted P(the peak is a real touch by this option) over :data:`FEATURE_NAMES` (scorer spec r7 2.3).
    scikit-learn at fit (the :data:`HGB_SETTINGS`), exported to tree arrays; numpy at serve. A missing value takes the
    branch the fit chose for it. ``decode_window_s`` is the first-touch decoding window the scorer was selected with;
    ``model.json`` holds it."""

    def __init__(self) -> None:
        self.trees: list[dict[str, list]] | None = None
        self.baseline: float = 0.0
        self.decode_window_s: float = DECODE_WINDOW_S
        self.contract: dict | None = None
        self.sklearn_version: str | None = None
        self._arrays: list[tuple[np.ndarray, ...]] = []

    def fit(self, features: pd.DataFrame, labels) -> ContactScorer:
        """Fit on ``features[FEATURE_NAMES]`` (missing values native) and 0/1 ``labels``, export the trees, and
        refuse the export unless its numpy serving equals ``predict_proba`` on the training rows within 1e-12."""
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier

        x = features[list(FEATURE_NAMES)].to_numpy(dtype=float)
        clf = HistGradientBoostingClassifier(**HGB_SETTINGS).fit(x, np.asarray(labels, dtype=int))
        trees, baseline = _export_trees(clf)
        self.trees, self.baseline, self._arrays = trees, baseline, _tree_arrays(trees)
        self.decode_window_s = DECODE_WINDOW_S
        gap = float(np.max(np.abs(self._serve(x) - clf.predict_proba(x)[:, 1]))) if len(x) else 0.0
        if not gap <= _PARITY_TOL:
            raise ContactScorerIntegrityError(
                f"the exported trees fail the parity check against scikit-learn {sklearn.__version__} "
                f"(max |diff| {gap:.3g} > {_PARITY_TOL})"
            )
        self.sklearn_version = sklearn.__version__
        return self

    def _serve(self, x: np.ndarray) -> np.ndarray:
        raw = np.full(len(x), self.baseline, dtype=np.float64)
        rows = np.arange(len(x))
        for feat, thr, left, right, miss, leaf, value in self._arrays:
            node = np.zeros(len(x), dtype=np.int64)
            active = rows[~leaf[node]]
            while len(active):
                nd = node[active]
                v = x[active, feat[nd]]
                go_left = np.where(np.isnan(v), miss[nd], v <= thr[nd])
                node[active] = np.where(go_left, left[nd], right[nd])
                active = active[~leaf[node[active]]]
            raw += value[node]
        with np.errstate(over="ignore"):
            return 1.0 / (1.0 + np.exp(-raw))

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """P(real touch by the option) per row."""
        if self.trees is None:
            raise RuntimeError("ContactScorer is not fitted")
        return self._serve(features[list(FEATURE_NAMES)].to_numpy(dtype=float))

    def check_params(self, params: BallContactParams) -> None:
        """Refuse detector fields that differ from the ones the scorer's features were computed with."""
        declared = (self.contract or {}).get("constants") or {}
        for field in ("kinematic_window_s", "min_candidate_speed_ms"):
            if field in declared and float(getattr(params, field)) != float(declared[field]):
                raise ValueError(
                    f"the contact scorer was trained with {field}={declared[field]}, not {getattr(params, field)}; "
                    "its features would leave the training distribution"
                )

    def to_dict(self) -> dict[str, Any]:
        if self.trees is None:
            raise RuntimeError("ContactScorer is not fitted")
        d: dict[str, Any] = {
            "version": SCORER_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "trees": self.trees,
            "baseline": self.baseline,
            "decode_window_s": self.decode_window_s,
        }
        if self.contract is not None:
            d["feature_contract"] = self.contract
        return d

    @classmethod
    def from_dict(cls, d: dict) -> ContactScorer:
        if d.get("version") != SCORER_VERSION:
            raise ContactScorerIntegrityError(f"model version {d.get('version')!r} is not {SCORER_VERSION!r}")
        if list(d["feature_names"]) != list(FEATURE_NAMES):
            raise ContactScorerIntegrityError(f"feature names {d['feature_names']} differ from {list(FEATURE_NAMES)}")
        m = cls()
        m.decode_window_s = _check_decode_window(d.get("decode_window_s"))
        m.trees = [dict(t) for t in d["trees"]]
        m.baseline = float(d["baseline"])
        m._arrays = _tree_arrays(m.trees)
        m.contract = d.get("feature_contract")
        return m

    def save(self, d, *, params: BallContactParams | None = None) -> None:
        """Write ``model.json`` (with the feature contract under ``params``, default ``BallContactParams()``) and
        ``SHA256SUMS`` to the directory ``d``."""
        d = Path(d)
        d.mkdir(parents=True, exist_ok=True)
        self.contract = feature_contract_for(params or BallContactParams())
        (d / "model.json").write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        _write_checksums(d)

    @classmethod
    def load(cls, d) -> ContactScorer:
        """Load and verify: the checksums, the model fields (version, names, trees, ``decode_window_s``), then the
        feature contract (constants first; a mismatch raises :class:`ContactScorerIntegrityError`; a missing contract
        warns)."""
        d = Path(d)
        if not (d / "model.json").exists():
            raise FileNotFoundError(f"no contact scorer at {d}")
        _verify_checksums(d)
        m = cls.from_dict(json.loads((d / "model.json").read_text(encoding="utf-8")))
        stored = m.contract
        declared = (stored or {}).get("constants") or {}
        params = BallContactParams(
            kinematic_window_s=float(declared.get("kinematic_window_s", BallContactParams().kinematic_window_s)),
            min_candidate_speed_ms=float(
                declared.get("min_candidate_speed_ms", BallContactParams().min_candidate_speed_ms)
            ),
        )
        verify_feature_contract(
            feature_contract_for(params),
            stored,
            legacy_override=False,
            model_name="contact scorer",
            error_cls=ContactScorerIntegrityError,
        )
        return m


def resolve_scorer(spec: str) -> ContactScorer:
    """``"default"`` (the bundled weights) or a directory path, loaded once per resolved path and cached."""
    return _load_cached(str((_DEFAULT_DIR if spec == "default" else Path(spec)).resolve()))


@functools.lru_cache(maxsize=8)
def _load_cached(path: str) -> ContactScorer:
    return ContactScorer.load(Path(path))
