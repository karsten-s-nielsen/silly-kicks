"""TF-46 Arm 1 Part 2 pre-flight runs (1) and (14): the reference truth model (spec sections 5.2, 5.6, 16.2).

Run (1) characterises the tier-1 event-time semantics per data profile x restart type -- the median and
dispersion of ``t_event - K`` (``K`` = the kinematic release on the undegraded track), the native bracket
quantiles ``a`` / ``b`` (p99.5 / p0.5 of ``t_event - K``, clipped to 2.0 s; the share outside), the
decision-time flag (|median| > 1.0 s), the +/-50 % sensitivity of every K threshold, the GS set-piece codes and
the DFL location-field characterisation (which DFL position field is independent truth and which a tracking
copy). Run (14) is the K gate: on held-out matches, the share of rows with ``|t_event - K - median| <=
tau_event`` (the reduce calibrates tau_event and centres on the median on the OTHER matches -- LOMO).

An ADR-128 stage-contract pipeline (``scripts._stages``): ``score`` is an ItemStage over the staged units
(resume-before-load), ``report`` reduces the done units in sorted key order to the committed tier-named aggregates
(ADR-116: counts and derived statistics only) with a provenance sidecar (ADR-037). Usage::

    python -m scripts.tf46p2_truth --dataset tf46p2 --data-root <ROOT> --store <DIR> --run-all \\
        --out docs/research/tf46_part2
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so `from scripts._X` resolves

from scripts._stages import ItemStage, Pipeline, Stage, main
from scripts._tf46p2_pipeline import (
    BASE_PARAMS,
    REFERENCE_PROVIDERS,
    done_results,
    part2_load,
    part2_refs,
    records_frame,
    report_fileset,
    split_records,
)

_RUN = "truth"
_SCHEMA = "tf46p2-truth-5"

#: Run (1) K sensitivity: every release threshold at +/-50 % (spec section 16.1), one at a time.
K_SENSITIVITY_KEYS: tuple[str, ...] = (
    "k_rest_m",
    "k_rest_s",
    "k_release_disp_m",
    "k_release_rest_sd_mult",
    "k_release_speed_ms",
    "k_release_frames",
    "sc_freekick_vkick_ms",  # v_kick: the strike bar (spec amendment 2026-10-22)
    "throwin_release_speed_ms",
    "throwin_release_dist_m",
    "throwin_release_window_s",
)

#: The DFL "tracking copy" bar (spec section 5.2): a field whose median distance to the tracked actor is within
#: this is a tracking copy for that event type, never independent location truth.
COPY_TOL_M = 0.1


def truth_unit(m) -> pd.DataFrame:
    """One staged match -> its run (1) / (14) records: ``score`` (the K-gate score rows), ``delta`` (per restart,
    ``t_event - K``), ``sens`` (``t_event - K`` under each +/-50 % threshold), ``location`` (DFL field distances to
    the tracked actor) and ``setpiece`` (the GS set-piece code counts). Empty for a non-reference provider."""
    from scripts._tf46p2_score import kick_episode_rows, kick_truth, location_diffs, native_restarts, score_k_gate
    from silly_kicks.tracking._gk_resolve import resolve_defended_goals
    from silly_kicks.tracking._restart_validation import sensitivity_values

    if m.provider not in REFERENCE_PROVIDERS:
        return records_frame({})
    rs = native_restarts(m.actions)
    goal_map = resolve_defended_goals(m.frames)  # once per match (ADR-055)
    k = kick_truth(rs, m.frames, goal_map=goal_map)
    from scripts._tf46p2_pipeline import profile_key

    key = profile_key(m.profile)
    deltas = [
        {"data_profile": key, "restart_type": r.base, "match_id": m.match_id, "delta": float(r.t_event - kk)}
        for r, kk in zip(rs.itertuples(index=False), k, strict=True)
        if np.isfinite(kk)
    ]
    sens = []
    for tkey in K_SENSITIVITY_KEYS:
        for label, value in zip(("lo", "hi"), sensitivity_values(tkey), strict=True):
            kv = kick_truth(rs, m.frames, thresholds={tkey: value}, goal_map=goal_map)
            for r, kk in zip(rs.itertuples(index=False), kv, strict=True):
                sens.append(
                    {
                        "data_profile": key,
                        "restart_type": r.base,
                        "key": tkey,
                        "side": label,
                        "value": float(value),
                        "fired": bool(np.isfinite(kk)),
                        "delta": float(r.t_event - kk) if np.isfinite(kk) else None,
                    }
                )
    setpiece = []
    if m.provider == "gradientsports" and not m.native.empty and "setpieceType" in m.native.columns:
        for code, n in m.native["setpieceType"].dropna().astype(str).value_counts().sort_index().items():
            setpiece.append({"data_profile": key, "setpiece_type": code, "n": int(n)})
    return records_frame(
        {
            "score": score_k_gate(m, rs, k),
            "delta": deltas,
            "sens": sens,
            "location": location_diffs(m),
            "setpiece": setpiece,
            "episodes": [
                {"data_profile": key, **row}
                for row in kick_episode_rows(rs, m.frames, goal_map=goal_map, actions=m.actions)
            ],
        }
    )


def characterise(delta: pd.DataFrame) -> pd.DataFrame:
    """Run (1) characterisation per data profile x restart type: n, the median offset, the dispersion (IQR),
    ``a`` = p99.5 and ``b`` = -p0.5 of ``t_event - K`` (each clipped to [0, 2.0] s), the share of rows outside
    ``[-b, a]`` (the signal fails if > 2 %, spec 16.1) and the decision-time flag (|median| > 1.0 s)."""
    from silly_kicks.tracking._restart_validation import RATIFIED

    cols = [
        "data_profile",
        "restart_type",
        "n",
        "median_s",
        "iqr_s",
        "a_s",
        "b_s",
        "outside_share",
        "outside_fails",
        "decision_time",
    ]
    if delta.empty:
        return pd.DataFrame(columns=cols)
    clip = RATIFIED["bracket_clip_s"]
    rows = []
    for (dp, rt), g in delta.groupby(["data_profile", "restart_type"], sort=True):
        d = g["delta"].astype(float).to_numpy()
        a = float(np.clip(np.quantile(d, 0.995), 0.0, clip))
        b = float(np.clip(-np.quantile(d, 0.005), 0.0, clip))
        outside = float(((d > a) | (d < -b)).mean())
        med = float(np.median(d))
        rows.append(
            {
                "data_profile": dp,
                "restart_type": rt,
                "n": len(d),
                "median_s": med,
                "iqr_s": float(np.quantile(d, 0.75) - np.quantile(d, 0.25)),
                "a_s": a,
                "b_s": b,
                "outside_share": outside,
                "outside_fails": bool(outside > RATIFIED["bracket_outlier_share"]),
                "decision_time": bool(abs(med) > RATIFIED["tau_event_decision_flag_s"]),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def sensitivity_table(sens: pd.DataFrame) -> pd.DataFrame:
    """Run (1) K sensitivity per profile x type x threshold x side: the fire rate and the median offset."""
    cols = ["data_profile", "restart_type", "key", "side", "value", "n", "fire_rate", "median_s"]
    if sens.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (dp, rt, key, side), g in sens.groupby(["data_profile", "restart_type", "key", "side"], sort=True):
        d = pd.to_numeric(g["delta"], errors="coerce").dropna().to_numpy()
        rows.append(
            {
                "data_profile": dp,
                "restart_type": rt,
                "key": key,
                "side": side,
                "value": float(g["value"].iloc[0]),
                "n": len(g),
                "fire_rate": float(g["fired"].astype(bool).mean()),
                "median_s": float(np.median(d)) if len(d) else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def location_table(loc: pd.DataFrame) -> pd.DataFrame:
    """Run (1) DFL location-field characterisation per profile x event type x field: n, the median distance to
    the tracked actor and ``is_copy`` (<= ``COPY_TOL_M``: a tracking copy, never truth; spec 5.2 / B-29)."""
    cols = ["data_profile", "event_type", "field", "n", "median_abs_diff_m", "is_copy"]
    if loc.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for (dp, et, f), g in loc.groupby(["data_profile", "event_type", "field"], sort=True):
        med = float(np.median(g["diff_m"].astype(float)))
        rows.append(
            {
                "data_profile": dp,
                "event_type": et,
                "field": f,
                "n": len(g),
                "median_abs_diff_m": med,
                "is_copy": bool(med <= COPY_TOL_M),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def _score_work(item, ctx) -> pd.DataFrame:
    return truth_unit(item)


def _report_work(key, ctx):
    from scripts._tf46p2_score import (
        aggregate_scores,
        kick_boundary_table,
        kick_episode_table,
        kick_strike_guard_table,
        scores_frame,
    )

    rec = split_records(done_results(ctx, "score"))
    scores = scores_frame(rec["score"].to_dict("records")) if "score" in rec else scores_frame([])
    setpiece = rec.get("setpiece", pd.DataFrame(columns=["data_profile", "setpiece_type", "n"]))
    if not setpiece.empty:
        setpiece = setpiece.groupby(["data_profile", "setpiece_type"], as_index=False)["n"].sum()
    summary = ctx.summary("score")
    tables = {
        _RUN: aggregate_scores(scores),
        f"{_RUN}_characterisation": characterise(rec.get("delta", pd.DataFrame())),
        f"{_RUN}_sensitivity": sensitivity_table(rec.get("sens", pd.DataFrame())),
        f"{_RUN}_location": location_table(rec.get("location", pd.DataFrame())),
        f"{_RUN}_setpiece_codes": setpiece,
        f"{_RUN}_kick_episodes": kick_episode_table(rec.get("episodes", pd.DataFrame())),
        f"{_RUN}_strike_guards": kick_strike_guard_table(rec.get("episodes", pd.DataFrame())),
        f"{_RUN}_law_boundary": kick_boundary_table(rec.get("episodes", pd.DataFrame())),
    }
    return report_fileset(
        _RUN,
        tables,
        ctx,
        {
            "schema": _SCHEMA,
            "n_attempted": summary["n_units"],
            "n_failed": summary["n_failed"],
            "n_excluded": summary["n_excluded"],
        },
    )


PIPELINE = Pipeline(
    name="tf46p2_truth",
    module="scripts.tf46p2_truth",
    params=BASE_PARAMS,
    stages=(
        ItemStage(
            name="score",
            refs=part2_refs,
            key=lambda ref: ref.stem,
            load=part2_load,
            work=_score_work,
            declare=lambda p: {"schema": _SCHEMA},
            params=("max_matches",),
            uses_dataset=True,
            max_failures=0,
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("score",),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="report",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
