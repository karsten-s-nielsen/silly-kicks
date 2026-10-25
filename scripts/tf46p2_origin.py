"""TF-46 Arm 1 Part 2 pre-flight runs (10) and (13): the origin gate and the goal-kick replacement (spec sections
10, 16.2).

* Run (13) origin: ``add_restart_coordinates`` with every section 10.1 gated tier passing, scored per restart type
  x origin tier against the section 10.2 truth (corner: the 1 m corner arc; throw-in: the touchline y and the
  native x; goal kick: the native location; free kick: the native execution spot -- GS has GEOMETRY truth only,
  C-43). The reduce applies the fixed bar (median <= 2 m AND share within 5 m). A gated origin the Law-region
  tripwire reverts is a FAILED origin of its tier. A DFL type run (1) marks a tracking copy has no location truth
  (B-29; needs ``--upstream``).
* Run (10) goal-kick replacement (the CL-6 ``distrust_native_origin`` retirement condition, section 10.3): per goal
  kick, the error of the gated replacement origin AND of the native origin against the DETECTED restarting keeper
  at the kick -- reported (``origin_goalkick_keeper``), never a gate; the ladder stays until it passes.

Transfer rows need the run (9) profiles (``--upstream``, C-03). An ADR-128 stage-contract pipeline. Usage::

    python -m scripts.tf46p2_origin --dataset tf46p2 --data-root <ROOT> --store <DIR> --run-all \\
        --out docs/research/tf46_part2 --upstream docs/research/tf46_part2
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
    UPSTREAM_PARAM,
    done_results,
    part2_load,
    part2_refs,
    profile_key,
    read_upstream_or_none,
    records_frame,
    report_fileset,
    split_records,
    transfer_runs,
    upstream_profiles,
)
from scripts.tf46p2_subtype_merge import copy_excluded_bases


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


def _i(v) -> int:
    """pandas Scalar -> int; the untyped arg sidesteps the Scalar stub friction."""
    return int(v)


_RUN = "origin"
_SCHEMA = "tf46p2-origin-2"


def goalkick_keeper_rows(m, run, restarts: pd.DataFrame) -> list[dict]:
    """Run (10): per goal kick, the distance of the gated replacement origin and of the native origin to the
    restarting team's DETECTED keeper at the kick (action frame), on this run's tracking."""
    from scripts._tf46p2_score import PASS_ALL
    from silly_kicks.spadl.utils import add_restart_coordinates
    from silly_kicks.tracking._gk_geometry import _keeper_xy_in_bracket, _resolve_flip, _truthy_bool

    gks = restarts[restarts["base"] == "goal_kick"]
    if gks.empty or run.frames.empty or "is_goalkeeper" not in run.frames.columns:
        return []
    enr = add_restart_coordinates(run.actions, frames=run.frames, data_profile=run.profile, gate_table=PASS_ALL)
    by_id = {aid: i for i, aid in enumerate(enr["action_id"].to_numpy())}
    fr = run.frames
    det = fr[fr["is_goalkeeper"].astype(bool) & ~fr["is_ball"].astype(bool)]
    if "visibility" in det.columns:
        det = det[_truthy_bool(det["visibility"])]
    gk_by_period = {
        int(p): (g.sort_values("time_seconds").reset_index(drop=True), np.sort(g["time_seconds"].to_numpy(float)))
        for p, g in det.groupby("period_id", sort=True)
    }
    flip = _resolve_flip(run.actions, fr, None) if "team_attacking_direction" in fr.columns else None
    pos = {aid: i for i, aid in enumerate(run.actions["action_id"].to_numpy())}
    rows = []
    for r in gks.itertuples(index=False):
        i, j = by_id.get(r.restart_id), pos.get(r.restart_id)
        if i is None or j is None or flip is None or pd.isna(flip[j]):
            continue
        t = float(run.actions["t_kick"].iloc[j]) if np.isfinite(float(run.actions["t_kick"].iloc[j])) else _f(r.t_event)
        kxy = _keeper_xy_in_bracket(gk_by_period, _i(r.period_id), r.team_id, t - 1.0, t + 1.0, t)
        if kxy is None:
            continue
        if flip[j]:
            kxy = (105.0 - kxy[0], 68.0 - kxy[1])
        e = enr.iloc[i]
        rep_err = (
            float(np.hypot(float(e["enriched_start_x"]) - kxy[0], float(e["enriched_start_y"]) - kxy[1]))
            if pd.notna(e.get("origin_evidence"))
            else None
        )
        nat_err = float(np.hypot(_f(r.x) - kxy[0], _f(r.y) - kxy[1])) if np.isfinite(r.x) and np.isfinite(r.y) else None
        rows.append(
            {
                "data_profile": profile_key(run.profile),
                "tier": run.tier,
                "replacement_source": e["start_coord_source"],
                "replacement_err_m": rep_err,
                "native_err_m": nat_err,
            }
        )
    return rows


def origin_unit(m, profiles: dict, excluded: frozenset[str]) -> pd.DataFrame:
    """One reference match -> run (13) score rows + run (10) goal-kick keeper records (reference + transfer)."""
    from scripts._tf46p2_score import native_restarts, provider_base, run_pipeline, score_origin

    if m.provider not in REFERENCE_PROVIDERS:
        return records_frame({})
    rs = native_restarts(m.actions)
    if provider_base(m.provider) == "idsse" and excluded:
        rs = rs[~rs["base"].isin(sorted(excluded))].reset_index(drop=True)
    ref = run_pipeline(m)
    rows, keeper = score_origin(m, ref, rs), goalkick_keeper_rows(m, ref, rs)
    for _pk, _target, _deg, tr in transfer_runs(m, profiles):
        rows += score_origin(m, tr, rs)
        keeper += goalkick_keeper_rows(m, tr, rs)
    return records_frame({"score": rows, "goalkick_keeper": keeper})


def _score_work(item, ctx) -> pd.DataFrame:
    location = read_upstream_or_none(ctx, "truth_location", "truth")
    return origin_unit(item, upstream_profiles(ctx), copy_excluded_bases(location))


def _report_work(key, ctx):
    from scripts._tf46p2_score import aggregate_scores, scores_frame

    rec = split_records(done_results(ctx, "score"))
    scores = scores_frame(rec["score"].to_dict("records")) if "score" in rec else scores_frame([])
    kp = rec.get("goalkick_keeper", pd.DataFrame())
    if not kp.empty:
        kp_tab = kp.groupby(["data_profile", "tier", "replacement_source"], as_index=False, dropna=False).agg(
            n=("native_err_m", "size"),
            replacement_median_m=(
                "replacement_err_m",
                lambda v: (
                    float(np.nanmedian(pd.to_numeric(v, errors="coerce")))
                    if pd.to_numeric(v, errors="coerce").notna().any()
                    else float("nan")
                ),
            ),
            native_median_m=(
                "native_err_m",
                lambda v: (
                    float(np.nanmedian(pd.to_numeric(v, errors="coerce")))
                    if pd.to_numeric(v, errors="coerce").notna().any()
                    else float("nan")
                ),
            ),
        )
    else:
        kp_tab = pd.DataFrame(
            columns=["data_profile", "tier", "replacement_source", "n", "replacement_median_m", "native_median_m"]
        )
    summary = ctx.summary("score")
    return report_fileset(
        _RUN,
        {_RUN: aggregate_scores(scores), f"{_RUN}_goalkick_keeper": kp_tab},
        ctx,
        {
            "schema": _SCHEMA,
            "n_attempted": summary["n_units"],
            "n_failed": summary["n_failed"],
            "n_excluded": summary["n_excluded"],
            "upstream": ctx.params.get("upstream"),
            "transfer_scored": bool(ctx.params.get("upstream")),
        },
    )


PIPELINE = Pipeline(
    name="tf46p2_origin",
    module="scripts.tf46p2_origin",
    params=(*BASE_PARAMS, UPSTREAM_PARAM),
    stages=(
        ItemStage(
            name="score",
            refs=part2_refs,
            key=lambda ref: ref.stem,
            load=part2_load,
            work=_score_work,
            declare=lambda p: {"schema": _SCHEMA, "upstream": p.get("upstream")},
            params=("max_matches", "upstream"),
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
