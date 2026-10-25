"""TF-46 Arm 1 Part 2 pre-flight runs (6) and (7): the subtype gate and the merge gate (spec sections 11.2, 11.4,
16.2).

* Run (6) subtype: ``refine_restart_types``' tracking rule ("the delivery's next touch enters the penalty area")
  per data profile x direction, scored for PRECISION against the profile's truth (IDSSE: the event-coded location
  of the next touch; GS: the PFF ``possessionEventType`` label) and against the per-cell majority-class baseline
  (the reduce writes the bootstrap lower bound of precision - baseline; the cell passes only above 0). Reported
  separately: the agreement of the native DFL label (``play_penalty_box``) with the location truth (the IDSSE
  label-vs-location mismatch). A DFL event type that run (1) marks as a tracking COPY has no location truth and is
  excluded (B-29; needs ``--upstream``).
* Run (7) merge: the tier-2 simulation rebuilds the SkillCorner derived-free-kick + shot shape from native restarts
  whose own event is the shot (and keeps a free kick + a separate shot as the negative population), then runs the
  restart pipeline as production does (``resolve_kicks`` -> ``refine_restart_types``); precision per form
  (``events_first`` / ``tracking``). The GS admission for ``merge`` scores the same on GS's own tracking.

Transfer rows (SkillCorner targets) need the run (9) degradation profiles: ``--upstream`` (C-03). An ADR-128
stage-contract pipeline. Usage::

    python -m scripts.tf46p2_subtype_merge --dataset tf46p2 --data-root <ROOT> --store <DIR> --run-all \\
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
    read_upstream_or_none,
    records_frame,
    report_fileset,
    split_records,
    transfer_runs,
    upstream_profiles,
)


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


_RUN = "subtype_merge"
_SCHEMA = "tf46p2-subtype-merge-2"

#: DFL restart event type -> the restart base whose subtype truth it carries (spec section 5.2).
_DFL_RESTART_BASE = {"CornerKick": "corner", "FreeKick": "free_kick"}


def copy_excluded_bases(location: pd.DataFrame | None) -> frozenset[str]:
    """The restart bases with NO location truth: run (1) marked their DFL ``position`` field a tracking copy (B-29;
    such a type is never scored against a tracking-derived value)."""
    if location is None or location.empty:
        return frozenset()
    cp = location[(location["field"] == "position") & location["is_copy"].astype(bool)]
    return frozenset(_DFL_RESTART_BASE[e] for e in cp["event_type"] if e in _DFL_RESTART_BASE)


def native_label_agreement(m, restarts: pd.DataFrame, truth: dict) -> list[dict]:
    """IDSSE: the native DFL label (``play_penalty_box`` on the restart event) vs the location truth -- reported
    separately (the label-vs-location meaning mismatch, spec section 11.2), never a gate."""
    if m.native.empty or "play_penalty_box" not in m.native.columns or "timestamp_seconds" not in m.native.columns:
        return []
    nat = m.native.set_index(m.native["timestamp_seconds"].round(3))
    rows = []
    for r in restarts.itertuples(index=False):
        if r.restart_id not in truth or round(_f(r.t_event), 3) not in nat.index:
            continue
        lab = nat.loc[[round(_f(r.t_event), 3)], "play_penalty_box"].iloc[0]
        if lab is None or pd.isna(lab):
            continue
        native = "crossed" if str(lab).lower() in ("true", "1", "yes") else "short"
        rows.append({"restart_type": r.base, "native": native, "location": truth[r.restart_id]})
    return rows


def subtype_merge_unit(m, profiles: dict, excluded: frozenset[str]) -> pd.DataFrame:
    """One reference match -> run (6) / (7) score rows (reference + admission + transfer) and the native-label
    agreement records. Empty for a non-reference provider."""
    from scripts._tf46p2_score import (
        _crossed_truth_idsse,
        native_restarts,
        provider_base,
        run_pipeline,
        score_merge,
        score_subtype,
    )

    if m.provider not in REFERENCE_PROVIDERS:
        return records_frame({})
    rs = native_restarts(m.actions)
    if provider_base(m.provider) == "idsse" and excluded:
        rs = rs[~rs["base"].isin(sorted(excluded))].reset_index(drop=True)
    ref = run_pipeline(m)
    rows = [*score_subtype(m, ref, rs), *score_merge(m, m.profile, m.frames, tier="reference")]
    labels = native_label_agreement(m, rs, _crossed_truth_idsse(m, rs)) if provider_base(m.provider) == "idsse" else []
    for _pk, target, deg, tr in transfer_runs(m, profiles):
        rows += score_subtype(m, tr, rs)
        rows += score_merge(m, target, deg)
    return records_frame({"score": rows, "labels": labels})


def _score_work(item, ctx) -> pd.DataFrame:
    location = read_upstream_or_none(ctx, "truth_location", "truth")
    return subtype_merge_unit(item, upstream_profiles(ctx), copy_excluded_bases(location))


def _report_work(key, ctx):
    from scripts._tf46p2_score import aggregate_scores, scores_frame

    rec = split_records(done_results(ctx, "score"))
    scores = scores_frame(rec["score"].to_dict("records")) if "score" in rec else scores_frame([])
    labels = rec.get("labels", pd.DataFrame(columns=["restart_type", "native", "location"]))
    label_tab = (
        labels.assign(agree=labels["native"] == labels["location"])
        .groupby("restart_type", as_index=False)
        .agg(n=("agree", "size"), agreement=("agree", "mean"))
        if not labels.empty
        else pd.DataFrame(columns=["restart_type", "n", "agreement"])
    )
    summary = ctx.summary("score")
    return report_fileset(
        _RUN,
        {_RUN: aggregate_scores(scores), f"{_RUN}_native_label_agreement": label_tab},
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
    name="tf46p2_subtype_merge",
    module="scripts.tf46p2_subtype_merge",
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
