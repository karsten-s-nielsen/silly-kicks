"""TF-46 Arm 1 Part 2 pre-flight run (12): the missed-penalty detector incl. shoot-outs (spec sections 12, 16.2).

FIRST the shoot-out frame check (spec 16.2 run 12): per match, the period-5 tracking frames and the period-5
``shot_penalty`` rows -- a shoot-out penalty with no tracking cannot be detected, and is reported, never silently
missing. Then ``detect_penalties`` (every cell passing) is scored against the native penalties (DFL ``Penalty`` /
PFF, SPADL ``shot_penalty`` with the section 25.2 result): detection PRECISION pooled over outcomes (the
``in_play`` / ``shoot_out`` strata pool only under the section 15.3 licence, applied by ``derive_gate_table``),
the goal / missed attribution, and the saved / off-target split. Reported with it: recall, the attribution and
rebound distribution, and the +/-50 % sensitivity of every penalty constant (spec 16.1). The GS admission for
``penalty`` is the GS reference rows; transfer rows (the SkillCorner goal-time semantics on degraded tracking)
need ``--upstream`` (run 9 profiles, C-03).

An ADR-128 stage-contract pipeline. Usage::

    python -m scripts.tf46p2_penalty --dataset tf46p2 --data-root <ROOT> --store <DIR> --run-all \\
        --out docs/research/tf46_part2 --upstream docs/research/tf46_part2
"""

from __future__ import annotations

import sys
from pathlib import Path

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
    records_frame,
    report_fileset,
    split_records,
    transfer_runs,
    upstream_profiles,
)

_RUN = "penalty"
_SCHEMA = "tf46p2-penalty-2"

#: The penalty constants run (12) evaluates at +/-50 % (spec section 16.1 "Penalty sensitivity").
PENALTY_SENSITIVITY_KEYS: tuple[str, ...] = (
    "penalty_spot_r_m",
    "penalty_rest_s",
    "penalty_position_eval_s",
    "penalty_d_gk_m",
    "penalty_gk_y_tol_m",
    "penalty_e_enc_m",
    "penalty_enc_disk_m",
    "penalty_enc_count_k",
    "penalty_taker_near_m",
    "penalty_W_s",
    "penalty_W_rb_s",
    "penalty_rb_live_speed_ms",
)


def shootout_check(m) -> dict:
    """The run (12) FIRST check: period-5 frames and period-5 native penalties per match."""
    from silly_kicks.spadl import config as spadlconfig

    pen = (
        m.actions[m.actions["type_id"] == spadlconfig.actiontype_id["shot_penalty"]]
        if not m.actions.empty
        else m.actions
    )
    return {
        "data_profile": profile_key(m.profile),
        "match_id": m.match_id,
        "n_period5_frames": int((m.frames["period_id"] == 5).sum()) if not m.frames.empty else 0,
        "n_period5_penalties": int((pen["period_id"] == 5).sum()) if len(pen) else 0,
        "n_penalties": len(pen),
    }


def penalty_unit(m, profiles: dict) -> pd.DataFrame:
    """One reference match -> run (12) records: the shoot-out check, the score rows (reference / admission /
    transfer), the detection distribution and the constant sensitivity."""
    from scripts._tf46p2_score import PASS_ALL, run_pipeline, score_penalty
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking._penalty_detector import detect_penalties
    from silly_kicks.tracking._restart_validation import sensitivity_values

    if m.provider not in REFERENCE_PROVIDERS:
        return records_frame({})
    ref = run_pipeline(m)
    rows = score_penalty(m, ref)
    det = detect_penalties(
        m.actions,
        ref.frames,
        provider=ref.provider,
        goal_map=ref.goal_map,
        contacts=ref.contacts,
        data_profile=ref.profile,
        gate_table=PASS_ALL,
        pitch_length=m.pitch_length,
    )
    pk = profile_key(m.profile)
    dist = [
        {"data_profile": pk, "stratum": d.stratum, "outcome": d.outcome, "attribution": d.attribution}
        for d in det.itertuples(index=False)
    ]
    n_native = (
        int((m.actions["type_id"] == spadlconfig.actiontype_id["shot_penalty"]).sum()) if not m.actions.empty else 0
    )
    sens = []
    for key in PENALTY_SENSITIVITY_KEYS:
        for side, value in zip(("lo", "hi"), sensitivity_values(key), strict=True):
            d2 = detect_penalties(
                m.actions,
                ref.frames,
                provider=ref.provider,
                goal_map=ref.goal_map,
                contacts=ref.contacts,
                data_profile=ref.profile,
                gate_table=PASS_ALL,
                constants={key: value},
                pitch_length=m.pitch_length,
            )
            sens.append(
                {
                    "data_profile": pk,
                    "key": key,
                    "side": side,
                    "value": float(value),
                    "n_detected": len(d2),
                    "n_native": n_native,
                }
            )
    for _pk, _target, _deg, tr in transfer_runs(m, profiles):
        rows += score_penalty(m, tr)
    return records_frame(
        {
            "shootout": [shootout_check(m)],
            "score": rows,
            "detections": dist,
            "sens": sens,
            "recall": [{"data_profile": pk, "match_id": m.match_id, "n_detected": len(det), "n_native": n_native}],
        }
    )


def _score_work(item, ctx) -> pd.DataFrame:
    return penalty_unit(item, upstream_profiles(ctx))


def _report_work(key, ctx):
    from scripts._tf46p2_score import aggregate_scores, scores_frame

    rec = split_records(done_results(ctx, "score"))
    scores = scores_frame(rec["score"].to_dict("records")) if "score" in rec else scores_frame([])
    det = rec.get("detections", pd.DataFrame(columns=["data_profile", "stratum", "outcome", "attribution"]))
    dist = (
        det.fillna("withheld").groupby(["data_profile", "stratum", "outcome", "attribution"], as_index=False).size()
        if not det.empty
        else pd.DataFrame(columns=["data_profile", "stratum", "outcome", "attribution", "size"])
    )
    sens = rec.get("sens", pd.DataFrame())
    sens_tab = (
        sens.groupby(["data_profile", "key", "side", "value"], as_index=False)[["n_detected", "n_native"]].sum()
        if not sens.empty
        else pd.DataFrame(columns=["data_profile", "key", "side", "value", "n_detected", "n_native"])
    )
    recall = rec.get("recall", pd.DataFrame())
    recall_tab = (
        recall.groupby("data_profile", as_index=False)[["n_detected", "n_native"]].sum()
        if not recall.empty
        else pd.DataFrame(columns=["data_profile", "n_detected", "n_native"])
    )
    shoot = rec.get("shootout", pd.DataFrame())
    shoot_tab = (
        shoot.groupby("data_profile", as_index=False)[["n_period5_frames", "n_period5_penalties", "n_penalties"]].sum()
        if not shoot.empty
        else pd.DataFrame(columns=["data_profile", "n_period5_frames", "n_period5_penalties", "n_penalties"])
    )
    summary = ctx.summary("score")
    return report_fileset(
        _RUN,
        {
            _RUN: aggregate_scores(scores),
            f"{_RUN}_shootout_check": shoot_tab,
            f"{_RUN}_distribution": dist,
            f"{_RUN}_sensitivity": sens_tab,
            f"{_RUN}_recall": recall_tab,
        },
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
    name="tf46p2_penalty",
    module="scripts.tf46p2_penalty",
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
