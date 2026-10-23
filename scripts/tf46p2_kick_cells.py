"""TF-46 Arm 1 Part 2 pre-flight runs (2), (3), (4), (5), (9), (11) + the GS base admission (spec sections 5.3, 6,
7, 9, 14.1, 16.2, 16.3).

* Run (9) -- ``profile`` (an ItemStage over the SkillCorner units): each match's section 16.3 statistics
  (``measure_profile_stats``), its camera-footprint gate share (``footprint_profile_gate``), its GI / tracking
  pairing at the ratified constants and at each pairing constant's +/-50 % value (``assess_gi_pairing``), and its
  camera-follow / off-camera re-detection model contributions. ``calibrate`` (one unit) pools them per SkillCorner
  data profile into the degradation parameters, the camera-follow model (only where the footprint gate passes)
  and the re-detection model -- and runs ``assert_known_paired`` FIRST: an unpaired known-paired opendata match
  stops the run before any cell is scored (C-11).
* Runs (2)-(5), (11) -- ``score`` (an ItemStage over the reference units): the K truth, the reference
  ``first_contact`` (IDSSE) and ``z_overhead`` (gate on vs off), then per SkillCorner profile the TRANSFER run on
  tracking degraded to that profile (``kick_time``, ``taker``, ``first_contact``), the run (3) ``ball_onset``
  report and the degraded section 16.3 statistics.
* ``report``: the fidelity check per profile (a failed statistic withholds exactly its dependent signals, the
  section 16.3 dependency map), the GS admission per signal, and the committed aggregates.

An ADR-128 stage-contract pipeline; the per-signal scoring is ``scripts._tf46p2_score``. Usage::

    python -m scripts.tf46p2_kick_cells --dataset tf46p2 --data-root <ROOT> --store <DIR> --run-all \\
        --out docs/research/tf46_part2 [--upstream docs/research/tf46_part2]
"""

from __future__ import annotations

import json
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
    degradation_params,
    degrade_to,
    done_results,
    part2_load,
    part2_refs,
    profile_key,
    read_upstream_or_none,
    records_frame,
    report_fileset,
    scores_profile_keys,
    split_records,
    unit_seed,
)


def _f(v) -> float:
    """pandas Scalar -> float; the untyped arg sidesteps the Scalar stub friction."""
    return float(v)


_RUN = "kick_cells"
_SCHEMA = "tf46p2-kick-cells-4"


# ---------------------------------------------------------------------------
# Run (9): the SkillCorner target profiles
# ---------------------------------------------------------------------------


def _sc_restart_table(resolved: pd.DataFrame) -> pd.DataFrame:
    """The SkillCorner restart table the event phase reads: the resolved restart rows with their bracket, the
    stoppage s0 = ``kick_bracket_lo`` on a derived row (spec section 5.3, C-PLAN-24 source (i))."""
    from scripts._tf46p2_common import stoppage_time

    rr = resolved[resolved["kick_bracket_lo"].notna()] if "kick_bracket_lo" in resolved.columns else resolved.iloc[:0]
    if rr.empty:
        return pd.DataFrame(columns=["restart_id", "period_id", "t_kick", "kick_bracket_hi", "s0", "s0_source"])
    tab = pd.DataFrame(
        {
            "restart_id": rr["action_id"].to_numpy(),
            "period_id": rr["period_id"].to_numpy(),
            "t_kick": rr["t_kick"].astype(float).to_numpy(),
            "kick_bracket_lo": rr["kick_bracket_lo"].astype(float).to_numpy(),
            "kick_bracket_hi": rr["kick_bracket_hi"].astype(float).to_numpy(),
            "is_skillcorner_derived": (rr.get("action_provenance", pd.Series(index=rr.index)) == "derived").to_numpy(),
        }
    )
    s0 = stoppage_time(tab, None, resolved)
    return tab.merge(s0, on="restart_id", how="left")


def profile_unit(m) -> pd.DataFrame:
    """One SkillCorner match -> its run (9) records (``stats``, ``footprint``, ``pairing``, ``camera``,
    ``redetection``). Empty for a reference provider."""
    from scripts._tf46p2_common import (
        PAIRING_PROX_MS,
        PAIRING_SENSITIVITY_KEYS,
        event_phase,
        fit_camera_follow,
        fit_redetection_error,
        footprint_profile_gate,
        gi_pairing_report,
        measure_profile_stats,
    )
    from scripts._tf46p2_score import run_pipeline
    from silly_kicks.tracking._restart_frame import assess_gi_pairing, gi_direction_check, pairing_proximity_profile
    from silly_kicks.tracking._restart_validation import sensitivity_values

    if m.provider != "skillcorner" or m.frames.empty:
        return records_frame({})
    key = profile_key(m.profile)
    run = run_pipeline(m)
    stats = measure_profile_stats(m.frames, run.actions, run.contacts)
    stats = stats.assign(data_profile=key, match_id=m.match_id)
    stats["samples"] = stats["samples"].map(lambda v: list(v) if isinstance(v, (tuple, list)) else [])
    foot = footprint_profile_gate(m.frames, m.footprint, m.profile) if not m.footprint.empty else None
    direction = gi_direction_check(m.actions, run.goal_map, flips=m.flips)  # the staged applied flip (R.2)
    base = assess_gi_pairing(m.actions, m.frames, provider="skillcorner", direction=direction)
    sens = {}
    for k in PAIRING_SENSITIVITY_KEYS:
        for v in sensitivity_values(k):
            sens[f"{k}={v}"] = assess_gi_pairing(
                m.actions, m.frames, provider="skillcorner", direction=direction, constants={k: v}
            )
    pairing = gi_pairing_report(base, m.profile, is_opendata=m.is_opendata, sensitivity=sens)
    # OD-6 amendment 2026-10-23: the match share and its time-shift negative controls at each run (2) distance;
    # `calibrate` derives the bar per data profile from them (pairing_bar_table) and guards the adopted bar.
    prox = pairing_proximity_profile(m.actions, m.frames, prox_ms=PAIRING_PROX_MS).assign(
        data_profile=key, match_id=m.match_id, is_opendata=bool(m.is_opendata)
    )
    prox = prox.drop(columns=["game_id"])
    mismatched = set(direction.mismatched_periods)
    gi_direction = [
        {
            "data_profile": key,
            "match_id": m.match_id,
            "period_id": int(pid),
            "agrees": (gid, pid) not in mismatched,
            "resolved": True,
        }
        for gid, pid in direction.checked_periods
    ] + [
        # no recorded flip -> never compared; the corpus guard counts these too (a lost record must not pass)
        {"data_profile": key, "match_id": m.match_id, "period_id": int(pid), "agrees": True, "resolved": False}
        for _gid, pid in direction.unresolved_periods
    ]
    camera = []
    if not m.footprint.empty:
        ball = m.frames[m.frames["is_ball"].to_numpy(dtype=bool)][["frame_id", "time_seconds", "x", "y"]]
        fp = m.footprint.merge(ball, on="frame_id", how="inner").sort_values(["period_id", "time_seconds"])
        fp = fp.reset_index(drop=True)
        phase = event_phase(fp[["period_id", "time_seconds"]], _sc_restart_table(run.actions))
        model = fit_camera_follow(
            fp.assign(game_id=str(m.match_id)), fp[["x", "y"]], phase, calibration_matches=[str(m.match_id)]
        )
        camera = [{"data_profile": key, "match_id": m.match_id, "model": json.dumps(model.to_dict())}]
    red = fit_redetection_error(m.frames.assign(game_id=str(m.match_id)), calibration_matches=[str(m.match_id)])
    return records_frame(
        {
            "stats": stats,
            "footprint": [] if foot is None else [{**foot, "match_id": m.match_id}],
            "pairing": pairing,
            "pairing_prox": prox,
            "gi_direction": gi_direction,
            "camera": camera,
            "redetection": [{"data_profile": key, "match_id": m.match_id, "model": json.dumps(red.to_dict())}],
            "profile": [{"data_profile": key, **vars(m.profile)}],
        }
    )


def _merge_camera(rows: pd.DataFrame, min_events: int = 30) -> dict | None:
    """Pool the per-match camera-follow contributions of ONE profile into one model (pooling recomputed)."""
    from scripts._tf46p2_common import CameraFollowModel, _pool_cells, _pooling_record

    if rows.empty:
        return None
    cells: dict = {}
    mids: list[str] = []
    for payload, mid in zip(rows["model"], rows["match_id"], strict=True):
        part = CameraFollowModel.from_dict(json.loads(payload))
        mids.append(str(mid))
        for k, seqs in part.sequences.items():
            cells.setdefault(k, []).extend(seqs)
    model = CameraFollowModel(
        _pool_cells(cells, min_events), _pooling_record(cells, min_events), tuple(sorted(mids)), min_events=min_events
    )
    return model.to_dict()


def _merge_redetection(rows: pd.DataFrame) -> dict | None:
    from scripts._tf46p2_common import RedetectionModel

    if rows.empty:
        return None
    jump: dict[str, list[float]] = {}
    angle: dict[str, list[float]] = {}
    mids = []
    for payload, mid in zip(rows["model"], rows["match_id"], strict=True):
        part = RedetectionModel.from_dict(json.loads(payload))
        mids.append(str(mid))
        for b, v in part.jump_m.items():
            jump.setdefault(b, []).extend(v)
        for b, v in part.angle_rad.items():
            angle.setdefault(b, []).extend(v)
    return RedetectionModel(jump, angle, tuple(sorted(mids))).to_dict()


def _calibrate_work(key, ctx) -> dict:
    """Pool run (9) per SkillCorner data profile; assert the known-paired calibration BEFORE any scoring."""
    from scripts._tf46p2_common import (
        assert_gi_direction_corpus,
        assert_known_paired,
        assert_pairing_bar_holds,
        assert_pairing_lag_holds,
        pairing_bar_table,
        pairing_lag_table,
    )
    from silly_kicks.tracking._restart_validation import RATIFIED

    rec = split_records(done_results(ctx, "profile"))
    pairing = rec.get("pairing", pd.DataFrame())
    assert_known_paired(pairing)  # C-11: an unpaired opendata match stops the run before a cell is scored
    # OD-6 amendment 2026-10-23: each data profile re-derives the bar from its own time-shift negatives; a bar
    # above the adopted pair_prox_share returns it to the owner with that profile's data (stop, never adjust).
    bar_table = pairing_bar_table(rec.get("pairing_prox", pd.DataFrame()), pairing)
    assert_pairing_bar_holds(bar_table)
    # second R.2 stop ruling 2026-10-23: a profile-wide median offset outside the lag tolerance returns the rule
    assert_pairing_lag_holds(pairing)
    lag_table = pairing_lag_table(pairing)
    # Spec 4.3 corpus guard: a systematic GI-direction mismatch (> 1 % of periods, or both periods of one match)
    # stops the run before any cell is scored -- a frame or flip bug, not noise.
    assert_gi_direction_corpus(rec.get("gi_direction", pd.DataFrame(columns=["match_id", "period_id", "agrees"])))
    profiles: dict[str, dict] = {}
    stats = rec.get("stats", pd.DataFrame())
    foot = rec.get("footprint", pd.DataFrame())
    for _, prow in rec.get("profile", pd.DataFrame()).drop_duplicates("data_profile").iterrows():
        pk = str(prow["data_profile"])
        st = stats[stats["data_profile"] == pk] if not stats.empty else stats
        fp = foot[foot["data_profile"] == pk] if not foot.empty else foot
        share = float(np.nanmean(fp["footprint_inside_share"].astype(float))) if len(fp) else float("nan")
        passed = bool(np.isfinite(share) and share >= RATIFIED["footprint_inside_share"])
        cam = rec.get("camera", pd.DataFrame())
        red = rec.get("redetection", pd.DataFrame())
        profiles[pk] = {
            "profile": {c: prow[c] for c in ("provider", "tracking_release", "gi_release", "competition_season")},
            "params": degradation_params(st),
            "footprint_share": share,
            "footprint_passed": passed,
            "camera": _merge_camera(cam[cam["data_profile"] == pk]) if passed and not cam.empty else None,
            "redetection": _merge_redetection(red[red["data_profile"] == pk]) if not red.empty else None,
            "match_ids": sorted(str(x) for x in st["match_id"].unique()) if not st.empty else [],
        }
    return {
        "profiles": profiles,
        "pairing_bar": bar_table.to_dict("records"),
        "pairing_lag": lag_table.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Runs (2)-(5), (11): reference + transfer scoring
# ---------------------------------------------------------------------------


def _bracket_ab(ctx, m) -> dict | None:
    """The run (1) a / b per restart base for this match's profile (C-03), or None without ``--upstream``."""
    char = read_upstream_or_none(ctx, "truth_characterisation", "truth")
    if char is None:
        return None
    pk = profile_key(m.profile)
    sub = char[char["data_profile"] == pk]
    return {str(r.restart_type): (_f(r.a_s), _f(r.b_s)) for r in sub.itertuples(index=False)} or None


def score_unit(m, calibration: dict, bracket_ab: dict | None) -> pd.DataFrame:
    """One reference match -> its score rows + the run (3) onset records + the degraded section 16.3 stats."""
    from scripts._tf46p2_common import measure_profile_stats
    from scripts._tf46p2_pipeline import restart_windows
    from scripts._tf46p2_score import (
        first_touch_truth,
        kick_truth,
        native_restarts,
        run_pipeline,
        score_first_contact,
        score_kick_time_and_taker,
        score_z_overhead,
    )
    from silly_kicks.tracking._restart_validation import RATIFIED, DataProfile

    if m.provider not in REFERENCE_PROVIDERS:
        return records_frame({})
    rs = native_restarts(m.actions)
    ref = run_pipeline(m, bracket_ab=bracket_ab)
    k = kick_truth(rs, m.frames, goal_map=ref.goal_map)  # the match's GoalMap, built once in run_pipeline
    truth = first_touch_truth(m, rs)
    rows = [*score_z_overhead(m, rs, truth), *score_first_contact(m, ref, rs, truth)]
    # The section 5.3 admission arm: the SkillCorner rules on this provider's OWN undegraded tracking, scored
    # against its own truth (kick_time via tf79_contact vs K; the kick-contact taker vs the native player).
    admission = run_pipeline(m, target=m.profile, degraded=m.frames, tier="reference")
    rows += score_kick_time_and_taker(m, admission, rs, k)
    onset, degraded_stats = [], []
    windows = restart_windows(m.actions, m.frames, m.native)
    for pk, prof in sorted(calibration["profiles"].items()):
        target = DataProfile(**prof["profile"])
        deg = degrade_to(
            m.frames,
            prof["params"],
            seed=unit_seed(m.match_id, pk),
            camera=prof["camera"],
            redetection=prof["redetection"],
            restarts=windows,
        )
        tr = run_pipeline(m, target=target, degraded=deg)
        rows += score_kick_time_and_taker(m, tr, rs, k)
        rows += score_first_contact(m, tr, rs, truth)
        by_id = {aid: i for i, aid in enumerate(tr.actions["action_id"].to_numpy())}
        for r, kk in zip(rs.itertuples(index=False), k, strict=True):
            i = by_id.get(r.restart_id)
            if i is None or r.base == "penalty":
                continue
            row = tr.actions.iloc[i]
            t_on, t_kick = float(row["t_onset"]), float(row["t_kick"])
            onset.append(
                {
                    "data_profile": pk,
                    "restart_type": r.base,
                    "subtype": r.restart_type,
                    "onset_fired": bool(np.isfinite(t_on)),
                    "onset_within_tau": bool(
                        np.isfinite(t_on) and np.isfinite(kk) and abs(t_on - kk) <= RATIFIED["tau_detector_ceiling_s"]
                    ),
                    "composed_err_s": abs(t_kick - _f(r.t_event)) if np.isfinite(t_kick) else None,
                }
            )
        st = measure_profile_stats(deg, tr.actions, tr.contacts).assign(data_profile=pk, match_id=m.match_id)
        st["samples"] = st["samples"].map(lambda v: list(v) if isinstance(v, (tuple, list)) else [])
        degraded_stats.append(st)
    return records_frame(
        {
            "score": rows,
            "onset": onset,
            "degraded_stats": pd.concat(degraded_stats, ignore_index=True) if degraded_stats else pd.DataFrame(),
        }
    )


def _score_work(item, ctx) -> pd.DataFrame:
    return score_unit(item, ctx.upstream("calibrate"), _bracket_ab(ctx, item))


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _pooled_stats(stats: pd.DataFrame) -> pd.DataFrame:
    """Pool per-match section 16.3 statistics of ONE profile: the n-weighted mean value per (region, statistic)
    and the concatenated distribution samples (the target and the degraded side share this rule)."""
    if stats.empty:
        return pd.DataFrame(columns=["region", "statistic", "value", "samples"])
    rows = []
    for (reg, stat), g in stats.groupby(["region", "statistic"], sort=True):
        v = g["value"].astype(float).to_numpy()
        n = g["n"].astype(float).to_numpy()
        ok = np.isfinite(v) & (n > 0)
        val = float(np.average(v[ok], weights=n[ok])) if ok.any() else float("nan")
        samples = [x for s in g["samples"] for x in (s or [])]
        rows.append({"region": reg, "statistic": stat, "value": val, "samples": samples})
    return pd.DataFrame(rows)


def fidelity_and_withheld(target: pd.DataFrame, degraded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run (9) fidelity per SkillCorner profile (spec section 16.3): the pooled degraded statistics vs the pooled
    target statistics, and the signals a failed statistic withholds (the dependency map; nothing else)."""
    from scripts._tf46p2_common import fidelity_report
    from silly_kicks.tracking._restart_validation import FIDELITY_DEPENDENCIES

    fid_rows, withheld = [], []
    profiles = sorted(set(target.get("data_profile", pd.Series(dtype=object))))
    for pk in profiles:
        t = _pooled_stats(target[target["data_profile"] == pk])
        d = _pooled_stats(degraded[degraded["data_profile"] == pk]) if not degraded.empty else degraded
        rep = fidelity_report(d, t) if not d.empty else pd.DataFrame()
        if rep.empty:
            continue
        rep = rep.assign(data_profile=pk)
        fid_rows.append(rep)
        failed = set(rep.loc[~rep["passed"].astype(bool), "statistic"])
        for sig, deps in sorted(FIDELITY_DEPENDENCIES.items()):
            hit = sorted(failed & set(deps))
            if hit:
                withheld.append({"data_profile": pk, "signal": sig, "failed_statistics": ",".join(hit)})
    fid = pd.concat(fid_rows, ignore_index=True) if fid_rows else pd.DataFrame()
    return fid, pd.DataFrame(withheld, columns=["data_profile", "signal", "failed_statistics"])


def _report_work(key, ctx):
    from scripts._tf46p2_common import PAIRING_BAR_COLUMNS
    from scripts._tf46p2_score import aggregate_scores, gs_admission_table, score_row, scores_frame

    cal = ctx.upstream("calibrate")
    prof_rec = split_records(done_results(ctx, "profile"))
    rec = split_records(done_results(ctx, "score"))
    scores = scores_frame(rec["score"].to_dict("records")) if "score" in rec else scores_frame([])
    fid, withheld = fidelity_and_withheld(
        prof_rec.get("stats", pd.DataFrame()), rec.get("degraded_stats", pd.DataFrame())
    )
    if not withheld.empty and not scores.empty:
        pk_of = scores_profile_keys(scores)
        drop = np.zeros(len(scores), dtype=bool)
        for w in withheld.itertuples(index=False):
            drop |= (
                (pk_of == w.data_profile).to_numpy()
                & (scores["signal"] == w.signal).to_numpy()
                & (scores["tier"] == "transfer").to_numpy()
            )
        scores = scores[~drop]
    from silly_kicks.tracking._restart_validation import DataProfile

    foot_rows = []
    for _pk, prof in sorted(cal["profiles"].items()):
        for mid in prof["match_ids"]:
            foot_rows.append(
                score_row(
                    DataProfile(**prof["profile"]),
                    "*",
                    "footprint",
                    "-",
                    base="native",
                    tier="reference",
                    match_id=mid,
                    ok=True,
                    footprint_inside_share=prof["footprint_share"],
                )
            )
    all_scores = pd.concat([scores, scores_frame(foot_rows)], ignore_index=True) if foot_rows else scores
    onset = rec.get("onset", pd.DataFrame())
    onset_tab = (
        onset.groupby(["data_profile", "restart_type", "subtype"], as_index=False).agg(
            n=("onset_fired", "size"), onset_rate=("onset_fired", "mean"), within_tau=("onset_within_tau", "mean")
        )
        if not onset.empty
        else pd.DataFrame(columns=["data_profile", "restart_type", "subtype", "n", "onset_rate", "within_tau"])
    )
    profiles_tab = pd.DataFrame(
        [
            {
                "data_profile": pk,
                "footprint_share": p["footprint_share"],
                "footprint_passed": p["footprint_passed"],
                "visibility_model": "camera_follow" if p["footprint_passed"] else "approximation",
                "params": json.dumps(p["params"], sort_keys=True),
                "camera": json.dumps(p["camera"]),
                "redetection": json.dumps(p["redetection"]),
                "n_matches": len(p["match_ids"]),
            }
            for pk, p in sorted(cal["profiles"].items())
        ],
        columns=[
            "data_profile",
            "footprint_share",
            "footprint_passed",
            "visibility_model",
            "params",
            "camera",
            "redetection",
            "n_matches",
        ],
    )
    admission = gs_admission_table(all_scores)
    summary = ctx.summary("score")
    tables = {
        _RUN: aggregate_scores(all_scores),
        f"{_RUN}_fidelity": fid,
        f"{_RUN}_withheld": withheld,
        f"{_RUN}_profiles": profiles_tab,
        f"{_RUN}_onset": onset_tab,
        f"{_RUN}_gs_admission": admission,
        "gi_pairing": prof_rec.get("pairing", pd.DataFrame()),
        "gi_pairing_bar": pd.DataFrame(cal.get("pairing_bar", []), columns=list(PAIRING_BAR_COLUMNS)),
        "gi_pairing_lag": pd.DataFrame(cal.get("pairing_lag", []), columns=["data_profile", "alignment_offset", "n"]),
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
            "upstream": ctx.params.get("upstream"),
        },
    )


PIPELINE = Pipeline(
    name="tf46p2_kick_cells",
    module="scripts.tf46p2_kick_cells",
    params=(*BASE_PARAMS, UPSTREAM_PARAM),
    stages=(
        ItemStage(
            name="profile",
            refs=part2_refs,
            key=lambda ref: ref.stem,
            load=part2_load,
            work=lambda item, ctx: profile_unit(item),
            declare=lambda p: {"schema": _SCHEMA, "stage": "profile"},
            params=("max_matches",),
            uses_dataset=True,
            max_failures=0,
        ),
        Stage(
            name="calibrate",
            units=lambda ctx: ["only"],
            work=_calibrate_work,
            needs=("profile",),
            reduce=lambda r, ctx: r["only"],
        ),
        ItemStage(
            name="score",
            refs=part2_refs,
            key=lambda ref: ref.stem,
            load=part2_load,
            work=_score_work,
            needs=("calibrate",),
            declare=lambda p: {"schema": _SCHEMA, "stage": "score", "upstream": p.get("upstream")},
            params=("max_matches", "upstream"),
            uses_dataset=True,
            max_failures=0,
        ),
        Stage(
            name="report",
            units=lambda ctx: ["only"],
            work=_report_work,
            needs=("profile", "calibrate", "score"),
            reduce=lambda r, ctx: r["only"],
        ),
    ),
    output="report",
)

if __name__ == "__main__":
    raise SystemExit(main(PIPELINE))
