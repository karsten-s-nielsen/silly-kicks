"""Shared ADR-128 stage plumbing for the TF-46 Arm 1 Part 2 pre-flight pipelines (spec section 16.2).

The five corpus walkers (``tf46p2_truth`` / ``_kick_cells`` / ``_subtype_merge`` / ``_penalty`` / ``_origin``) are
stage-contract pipelines over a staged ``flat_tc3`` dataset carrying the Part 2 extras
(``scripts._driver_sources.PART2_EXTRA_FILES``). This module holds what they share: the parameters, the unit
refs and loader (resume-before-load, the dataset read gate), the mixed-record unit frame, the run-9 degradation
profile and its application, and the report ``FileSet`` with its provenance sidecar (ADR-037: top-level
``run_commit`` / ``run_tree_dirty``). The per-signal scoring lives in ``scripts._tf46p2_score``.

ASCII-only (the driver ASCII gate). Imports no pining loader (the ADR-128 I1 gate): on the DGX the same pipeline
runs over a local ``--data-root`` (ADR-128 ruling 11: the DGX is not a second path).
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from scripts._stages import FileSet, Param

#: The parameters every TF-46 Part 2 walker takes.
BASE_PARAMS: tuple[Param, ...] = (
    Param("dataset", str, "tf46p2", help="the staged TF-46 Part 2 dataset (flat_tc3 + PART2_EXTRA_FILES)"),
    Param("max_matches", int, None, help="cap the corpus to its first N units (sorted keys); None = all"),
)

#: The upstream run-root parameter (spec section 16.2 C-03 inter-run flow): run (1) gives the bracket a / b, run
#: (9) the degradation profiles. Absent -> the walker scores the reference tier only and records that.
UPSTREAM_PARAM = Param("upstream", str, None, help="run-root of the committed upstream aggregates (read_upstream)")

REFERENCE_PROVIDERS = frozenset({"sportec", "idsse", "gradientsports"})


def part2_refs(ctx) -> list:
    """The staged units (``staged_part2_refs``), sorted, capped by ``--max-matches``."""
    from scripts._driver_sources import staged_part2_refs

    refs = sorted(staged_part2_refs(ctx.data_root, ctx.params["dataset"]), key=lambda r: r.stem)
    cap = ctx.params.get("max_matches")
    return refs[: int(cap)] if cap is not None else refs


def staging_generation(ctx) -> list[str]:
    """The sorted staged ``shards/<generation>/`` ids of the run's units (directory names only; no file is read)."""
    return sorted({str(r.files[0]).replace("\\", "/").split("/")[1] for r in part2_refs(ctx) if r.files})


def part2_load(ref, ctx):
    """Load ONE unit (resume-before-load: the executor calls this only for a pending unit)."""
    from scripts._driver_sources import dataset_dir
    from scripts._tf46p2_score import load_part2_match

    return load_part2_match(ref, dataset_dir(ctx.data_root, ctx.params["dataset"]))


def unit_seed(match_id: str, salt: str = "") -> int:
    """A deterministic per-unit seed (never wall-clock, never a hash-ordered set): sha256 of the match id."""
    return int(hashlib.sha256(f"{salt}|{match_id}".encode()).hexdigest()[:8], 16)


# ---------------------------------------------------------------------------
# Mixed-record unit frames: one tidy DataFrame per unit, a `record` column naming each row's table
# ---------------------------------------------------------------------------


def records_frame(tables: Mapping[str, pd.DataFrame | list[dict]]) -> pd.DataFrame:
    """Stack named tables into ONE unit frame with a ``record`` column (a unit result is one DataFrame).
    Columns are the union; every value column is stored as a JSON string so a mixed-type column never fails
    ``_stages.assert_storable`` and the round trip is exact."""
    parts = []
    for name in sorted(tables):
        t = tables[name]
        # A list of dicts is serialised AS IS: building a DataFrame first would let pandas 3 infer a `str` column
        # and turn `None` into `NaN` before the JSON (C2 r2 C-24). A DataFrame input is serialised row by row.
        rows = list(t) if isinstance(t, list) else t.to_dict("records")
        if not rows:
            continue
        payload = [json.dumps(r, sort_keys=True, default=_json_default) for r in rows]
        parts.append(pd.DataFrame({"record": name, "payload": payload}))
    if not parts:
        return pd.DataFrame({"record": pd.Series(dtype=object), "payload": pd.Series(dtype=object)})
    return pd.concat(parts, ignore_index=True)


def _json_default(v: Any):
    if v is pd.NA or v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (tuple, set, np.ndarray)):
        return list(v)
    return str(v)


def split_records(frames: Sequence[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The inverse of ``records_frame`` over the done unit frames, in the given (sorted key) order."""
    by: dict[str, list[dict]] = {}
    for f in frames:
        if f is None or f.empty:
            continue
        for rec, payload in zip(f["record"], f["payload"], strict=True):
            by.setdefault(str(rec), []).append(json.loads(payload))
    return {k: _records_to_frame(v) for k, v in sorted(by.items())}


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    """Rows -> a frame whose columns keep the JSON values exactly, under pandas 2 AND 3. A column holding any
    string stays ``object`` (pandas 3 would infer its ``str`` dtype and turn ``None`` into ``NaN``); every other
    column is inferred as before (numbers, booleans, ``None`` -> ``NaN`` in a numeric column, lists as objects)."""
    cols = list(dict.fromkeys(c for r in records for c in r))
    data = {}
    for c in cols:
        values = [r.get(c) for r in records]
        data[c] = pd.Series(values, dtype=object) if any(isinstance(v, str) for v in values) else pd.Series(values)
    return pd.DataFrame(data, columns=cols)


def done_results(ctx, stage: str) -> list[pd.DataFrame]:
    """An ItemStage's done unit results in sorted key order (the reduce reads only this run's units)."""
    from scripts._stages import read_result, sorted_units, unit_states

    st = ctx.pipeline.stage(stage)
    gen = ctx.generation(stage)
    states = unit_states(ctx, st, sorted_units(ctx, st), verify=False)
    return [read_result(ctx.store, f"{gen}/{k}") for k, v in states.items() if v == "done"]


# ---------------------------------------------------------------------------
# Report: the committed tier-named aggregate + its provenance sidecar (ADR-037, ADR-116)
# ---------------------------------------------------------------------------


def report_fileset(run: str, tables: Mapping[str, pd.DataFrame], ctx, extra: Mapping[str, Any]) -> FileSet:
    """``<name>.parquet`` per table + ``<run>.json`` (top-level ``run_commit`` / ``run_tree_dirty`` and the run
    counters). Deterministic bytes (sorted keys, ``index=False``)."""
    files: dict[str, bytes] = {}
    for name, table in sorted(tables.items()):
        buf = io.BytesIO()
        table.reset_index(drop=True).to_parquet(buf, index=False)
        files[f"{name}.parquet"] = buf.getvalue()
    prov = ctx.prov
    side = {
        "run": run,
        "tables": {name: len(t) for name, t in sorted(tables.items())},
        "run_commit": prov.get("commit"),
        "run_tree_dirty": bool(prov.get("dirty", False)),
        "run_tree_state": prov.get("tree_state"),
        # the staged dataset generation(s) the units came from (second R.2 stop ruling 2026-10-23: a reused
        # staging is recorded in every run's provenance)
        "staging_generation": staging_generation(ctx),
        **dict(extra),
    }
    files[f"{run}.json"] = json.dumps(side, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    return FileSet(files)


# ---------------------------------------------------------------------------
# Run (9): the degradation profile per SkillCorner data profile (spec sections 5.3, 16.3)
# ---------------------------------------------------------------------------


def profile_key(profile) -> str:
    from scripts._tf46p2_common import _profile_key

    return _profile_key(profile)


def scores_profile_keys(scores: pd.DataFrame) -> pd.Series:
    """The ``profile_key`` of every score / aggregate row (the same 4-part key the run tables carry)."""
    from silly_kicks.tracking._restart_validation import DataProfile

    def _one(r):
        gi = r["gi_release"]
        return profile_key(
            DataProfile(
                r["provider"], r["tracking_release"], None if gi is None or pd.isna(gi) else gi, r["competition_season"]
            )
        )

    return scores.apply(_one, axis=1) if len(scores) else pd.Series(dtype=object)


def degradation_params(stats: pd.DataFrame) -> dict[str, Any]:
    """The ``degrade`` parameters of ONE SkillCorner profile from its pooled ``measure_profile_stats`` rows: the
    median frame rate, the per-region gap-length sample (pooled) and the per-region median rest noise SD."""
    out: dict[str, Any] = {"frame_rate": float("nan"), "gap_lengths": {}, "noise_sd": {}}
    if stats.empty:
        return out
    fr = stats[stats["statistic"] == "frame_rate"]["value"].astype(float)
    out["frame_rate"] = float(np.nanmedian(fr)) if len(fr) else float("nan")
    for reg, g in stats[stats["statistic"] == "gap_distribution"].groupby("region", sort=True):
        sample: list[float] = []
        for s in g["samples"]:
            sample.extend(float(v) for v in (s or []))
        out["gap_lengths"][str(reg)] = sorted(sample)
    for reg, g in stats[stats["statistic"] == "noise_sd"].groupby("region", sort=True):
        v = g["value"].astype(float)
        out["noise_sd"][str(reg)] = float(np.nanmedian(v)) if np.isfinite(v).any() else 0.0
    return out


def degrade_to(
    frames: pd.DataFrame,
    params: Mapping[str, Any],
    *,
    seed: int,
    camera: Mapping[str, Any] | None = None,
    redetection: Mapping[str, Any] | None = None,
    restarts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Tier-2 degradation of reference tracking to a SkillCorner profile (spec section 5.3): the ball statistics
    (``degrade``: frame rate, per-region gaps, noise), the carried detection flag as ``visibility`` (ball detection
    is measured on the flag, never on extrapolated positions), the estimated ``z_source``, the camera-follow
    visibility (a profile whose footprint gate passed) and the off-camera position replay. Seeded."""
    from scripts._tf46p2_common import (
        CameraFollowModel,
        RedetectionModel,
        apply_camera_follow,
        build_footprint_chain,
        degrade,
        replay_offcamera_error,
    )

    p = dict(params)
    if not np.isfinite(float(p.get("frame_rate", float("nan")))):
        p["frame_rate"] = 0.0
    out = degrade(frames, p, seed=seed)
    if "detected" in out.columns:
        ball = out["is_ball"].to_numpy(dtype=bool)
        det = out["detected"].astype("boolean").fillna(True).to_numpy(dtype=bool)
        vis = out["visibility"].astype(object).to_numpy() if "visibility" in out.columns else np.full(len(out), True)
        vis = np.where(ball, det, vis)
        out["visibility"] = pd.array(
            [bool(v) if v is not None and not pd.isna(v) else pd.NA for v in vis], dtype="boolean"
        )
        out = out.drop(columns=["detected"])
    out["z_source"] = "estimated"
    if camera is not None and restarts is not None and len(restarts):
        model = CameraFollowModel.from_dict(camera)
        timeline = out[out["is_ball"].to_numpy(dtype=bool)][["period_id", "frame_id", "time_seconds", "x", "y"]]
        chain = build_footprint_chain(model, timeline, restarts, seed=seed)
        out = apply_camera_follow(model, out, chain)
    if redetection is not None:
        out = replay_offcamera_error(RedetectionModel.from_dict(redetection), out, seed=seed)
    return out.reset_index(drop=True)


def restart_windows(actions: pd.DataFrame, frames: pd.DataFrame, native: pd.DataFrame) -> pd.DataFrame:
    """The restart table the camera-follow chain and the event phase read (spec section 5.3 / C-PLAN-24):
    ``restart_id``, ``period_id``, ``t_kick``, ``kick_bracket_hi`` and ``s0`` with its source."""
    from scripts._tf46p2_common import stoppage_time
    from scripts._tf46p2_score import native_restarts

    rs = native_restarts(actions)
    if rs.empty:
        return pd.DataFrame(columns=["restart_id", "period_id", "t_kick", "kick_bracket_hi", "s0", "s0_source"])
    tab = pd.DataFrame(
        {
            "restart_id": rs["restart_id"],
            "period_id": rs["period_id"],
            "t_kick": rs["t_event"],
            "kick_bracket_hi": rs["t_event"],
        }
    )
    events = actions.rename(columns={"time_seconds": "time_seconds"})
    s0 = stoppage_time(tab, frames, events)
    return tab.merge(s0, on="restart_id", how="left") if "restart_id" in s0.columns else tab


def read_upstream_or_none(ctx, table: str, run: str):
    """The upstream ``<table>.parquet`` of run ``run``, provenance-checked against that run's ``<run>.json``
    sidecar (C-03: an absent artifact or a different ``run_commit`` refuses the run); ``None`` without
    ``--upstream``. Repo-committed research files, not a staged dataset (the dataset read gate does not apply)."""
    from pathlib import Path

    from scripts._tf46p2_common import read_upstream

    root = ctx.params.get("upstream")
    if not root:
        return None
    _agg, _side = read_upstream(Path(root), run, expect_commit=ctx.prov.get("commit"))
    path = Path(root) / f"{table}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"upstream table missing: {path}")
    return pd.read_parquet(path)


def upstream_profiles(ctx) -> dict:
    """The run (9) SkillCorner degradation profiles from the upstream ``kick_cells`` report (C-03); ``{}`` without
    ``--upstream`` (the walker then scores the reference tier only and records that)."""
    tab = read_upstream_or_none(ctx, "kick_cells_profiles", "kick_cells")
    if tab is None:
        return {}
    out = {}
    for r in tab.itertuples(index=False):
        prof = str(r.data_profile).split("|")
        out[str(r.data_profile)] = {
            "profile": {
                "provider": prof[0],
                "tracking_release": prof[1],
                "gi_release": None if prof[2] in ("", "None") else prof[2],
                "competition_season": prof[3],
            }
            if len(prof) == 4
            else None,
            "params": json.loads(str(r.params)),
            "camera": json.loads(str(r.camera)),
            "redetection": json.loads(str(r.redetection)),
        }
    return out


def transfer_runs(m, profiles: Mapping[str, Mapping[str, Any]]):
    """For each SkillCorner target profile: ``(profile_key, target, degraded_frames, run)`` -- the reference match
    degraded to the target (``degrade_to``, seeded per unit x profile) and the SkillCorner restart pipeline run on
    it (``_tf46p2_score.run_pipeline``). Sorted by profile key (deterministic)."""
    from scripts._tf46p2_score import run_pipeline
    from silly_kicks.tracking._restart_validation import DataProfile

    windows = restart_windows(m.actions, m.frames, m.native)
    for pk, prof in sorted(profiles.items()):
        if prof.get("profile") is None:
            continue
        target = DataProfile(**prof["profile"])
        deg = degrade_to(
            m.frames,
            prof["params"],
            seed=unit_seed(m.match_id, pk),
            camera=prof.get("camera"),
            redetection=prof.get("redetection"),
            restarts=windows,
        )
        yield pk, target, deg, run_pipeline(m, target=target, degraded=deg)
