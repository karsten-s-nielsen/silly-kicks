"""One-off empirical probe for silly-kicks PR-S20 (action_context features).

Task 0 of PR-S20: pull slim, lakehouse-derived per-provider action +
tracking-frame slices for use by the cross-provider parity test in
Task 13. The actual per-feature distribution baselines (e.g.
``nearest_defender_distance_p50``) get backfilled in Task 13 once
``add_action_context`` exists; this probe emits the SLICES + the
SKELETON baselines JSON.

Sources
-------
- ``soccer_analytics.dev_gold.fct_action_values`` (raw provider-ingested
  fields only; ``data_source`` is the provider column).
- ``soccer_analytics.dev_gold.fct_tracking_frames`` (raw provider-native
  fields including ``speed_ms`` --- per PR-S19 baselines, ``speed_ms`` is
  provider-native on all 3 lakehouse providers, so pulling it honors the
  consumer principle).

Hybrid speed policy (ADR-004 invariant 7)
-----------------------------------------
Provider-native ``speed_ms`` IS pulled from the lakehouse and threaded
through the silly_kicks adapters as ``speed_native``. The sportec adapter
preserves it (``speed_source = "native"``); ``_derive_speed`` only fills
NaN rows. This is correct: the lakehouse stores provider-native speed,
not a recomputed value.

Lakehouse-as-CONSUMER (raw fields only)
---------------------------------------
NOT pulled (lakehouse-COMPUTED): ``vaep_value, offensive_value,
defensive_value, pitch_control_value, voronoi_area, distance_to_ball,
acceleration_ms2``. silly_kicks recomputes these from raw fields when
needed.

Provider mapping
----------------
- silly_kicks ``sportec``  <-> lakehouse ``idsse``  (both tables)
- silly_kicks ``metrica``  <-> lakehouse ``metrica`` (both tables)
- silly_kicks ``skillcorner`` <-> lakehouse ``skillcorner`` (tracking ONLY;
  the lakehouse ``fct_action_values`` mart has no skillcorner actions, so
  the probe synthesizes 10 pseudo-actions sampled from ball trajectory).

Join key
--------
``fct_action_values.match_id`` is a bigint surrogate; ``fct_tracking_frames.match_id``
is the provider-native string. They share rows via ``match_key`` (bigint,
present in both tables). All cross-table joins use ``match_key``.

Outputs
-------
1. ``tests/datasets/tracking/action_context_slim/{sportec,metrica,skillcorner}_slim.parquet``
   --- one parquet per provider with a ``__kind`` column distinguishing
   ``"action"`` vs ``"frame"`` rows.
2. ``tests/datasets/tracking/empirical_action_context_baselines.json``
   --- provenance metadata + per-provider stat slots set to None
   (filled in by Task 13).

PFF
---
Intentionally absent from the lakehouse probe (PFF data is not
redistributable per PR-S18). The JSON includes a ``pff`` provider entry
with all stats null + a marker pointing to the synthetic baseline computed
in Task 13 from ``tests/datasets/tracking/pff/medium_halftime.parquet``.

Usage::

    python scripts/probe_action_context_baselines.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# UTF-8 stdout for Windows consoles (per PR-S19 pattern + user CLAUDE.md).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = REPO_ROOT / "tests" / "datasets" / "tracking"
SLIM_DIR = TRACKING_DIR / "action_context_slim"
OUTPUT_JSON = TRACKING_DIR / "empirical_action_context_baselines.json"

LH_TABLE_ACTIONS = "soccer_analytics.dev_gold.fct_action_values"
LH_TABLE_FRAMES = "soccer_analytics.dev_gold.fct_tracking_frames"

# (silly_kicks_name, lakehouse_data_source_for_actions, lakehouse_source_provider_for_frames).
# SkillCorner has no rows in fct_action_values --- handled via
# pseudo-action synthesis from ball trajectory in _process_skillcorner.
PROVIDER_PAIRS: list[tuple[str, str | None, str]] = [
    ("sportec", "idsse", "idsse"),
    ("metrica", "metrica", "metrica"),
    ("skillcorner", None, "skillcorner"),  # actions synthesized
]

# Raw provider-native fields ONLY for fct_tracking_frames. ``speed_ms`` is
# provider-native (per PR-S19 baselines: speed_native_supplied=True for all
# 3 lakehouse providers); pulling it honors the consumer principle.
TRACKING_RAW_COLS = [
    "match_key",
    "match_id",
    "period",
    "frame",
    "timestamp_seconds",
    "frame_rate",
    "player_id",
    "team",
    "team_id",
    "source_provider",
    "is_goalkeeper",
    "x",
    "y",
    "ball_x",
    "ball_y",
    "speed_ms",
]

# Raw fields ONLY for fct_action_values (raw provider-ingested fields).
ACTION_RAW_COLS = [
    "match_key",
    "match_id",
    "period",
    "time_seconds",
    "action_id",
    "team_id",
    "player_id",
    "start_x",
    "start_y",
    "end_x",
    "end_y",
    "action_type",
    "action_result",
    "bodypart",
    "data_source",
]

N_ACTIONS_TARGET = 10  # ~10 actions per provider, stratified across periods
TOLERANCE_SECONDS_FRAME_WINDOW = 2.0  # +/- 2s window of frames around each action
LINK_TOLERANCE_SECONDS = 0.2  # silly_kicks default tolerance for the linkage sanity check

# Lakehouse 120x80 -> silly_kicks 105x68 m unit conversion.
X_SCALE = 105.0 / 120.0
Y_SCALE = 68.0 / 80.0


def _databricks_connect():
    """Open a Databricks SQL connection from env vars; None on missing config.

    Honors ``DATABRICKS_HOST`` priority over ``DATABRICKS_SERVER_HOSTNAME``
    on this machine (memory: reference_lakehouse_tracking_traps).
    """
    try:
        from databricks import sql  # type: ignore[import-not-found]
    except ImportError:
        print("  [warn] databricks-sql-connector not installed; cannot probe.")
        return None

    raw_host = os.environ.get("DATABRICKS_HOST") or os.environ.get(
        "DATABRICKS_SERVER_HOSTNAME",
        "",
    )
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path and not http_path.startswith("/"):
        http_path = "/" + http_path
    if http_path.startswith("//"):
        http_path = http_path[1:]
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
        print("  [warn] DATABRICKS_HOST/HTTP_PATH/TOKEN not all set; cannot probe.")
        return None
    try:
        return sql.connect(
            server_hostname=server_hostname,
            http_path=http_path,
            access_token=token,
        )
    except Exception as e:
        print(f"  [warn] databricks connect failed: {e}")
        return None


def _show_distinct_data_sources(cur) -> set[str]:
    cur.execute(f"SELECT DISTINCT data_source FROM {LH_TABLE_ACTIONS}")
    return {r[0] for r in cur.fetchall() if r[0] is not None}


def _show_distinct_source_providers(cur) -> set[str]:
    cur.execute(f"SELECT DISTINCT source_provider FROM {LH_TABLE_FRAMES}")
    return {r[0] for r in cur.fetchall() if r[0] is not None}


def _find_match_key_with_both(
    cur,
    actions_data_source: str,
    tracking_source_provider: str,
    min_actions: int = 50,
) -> int | None:
    """Find one ``match_key`` present in BOTH tables for the given provider.

    Uses ``ORDER BY match_key LIMIT 1`` (NOT raw ``LIMIT 1`` --- natural
    row order is unstable per memory:reference_lakehouse_tracking_traps).
    The two tables are joined via ``match_key`` (bigint), NOT ``match_id``
    --- ``match_id`` differs between the tables (action_values has a bigint
    surrogate, tracking_frames has the provider-native string).
    """
    sql = f"""
        WITH a AS (
          SELECT match_key, COUNT(*) AS n_actions
          FROM {LH_TABLE_ACTIONS}
          WHERE data_source = '{actions_data_source}'
          GROUP BY match_key HAVING n_actions >= {min_actions}
        ),
        t AS (
          SELECT DISTINCT match_key
          FROM {LH_TABLE_FRAMES}
          WHERE source_provider = '{tracking_source_provider}'
        )
        SELECT a.match_key
        FROM a JOIN t USING (match_key)
        ORDER BY a.match_key LIMIT 1
    """
    cur.execute(sql)
    r = cur.fetchone()
    return None if r is None else int(r[0])


def _find_skillcorner_match_key(cur) -> int | None:
    """SkillCorner has no rows in fct_action_values; pick a tracking match_key."""
    cur.execute(
        f"""
        SELECT match_key, COUNT(*) AS n_frames
        FROM {LH_TABLE_FRAMES}
        WHERE source_provider = 'skillcorner'
        GROUP BY match_key
        ORDER BY match_key LIMIT 1
        """,
    )
    r = cur.fetchone()
    return None if r is None else int(r[0])


def _pull_actions(
    cur,
    actions_data_source: str,
    match_key: int,
    n_target: int = N_ACTIONS_TARGET,
) -> pd.DataFrame:
    """Pull a stratified-by-period sample of actions for one match.

    5 actions per period (period 1 + 2) = 10 total. Uses ROW_NUMBER() for
    deterministic stratification.
    """
    per_period = max(1, n_target // 2)
    col_list = ", ".join(ACTION_RAW_COLS)
    sql = f"""
        WITH ranked AS (
          SELECT {col_list},
                 ROW_NUMBER() OVER (PARTITION BY period ORDER BY time_seconds) AS rn
          FROM {LH_TABLE_ACTIONS}
          WHERE data_source = '{actions_data_source}'
            AND match_key = {match_key}
            AND period IN (1, 2)
        )
        SELECT {col_list}
        FROM ranked WHERE rn <= {per_period}
        ORDER BY period, time_seconds
        LIMIT {n_target}
    """
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame.from_records(rows, columns=cols)


def _pull_frames_around_actions(
    cur,
    tracking_source_provider: str,
    match_key: int,
    action_times: list[tuple[int, float]],
    tolerance: float = TOLERANCE_SECONDS_FRAME_WINDOW,
) -> pd.DataFrame:
    """Pull tracking frames within +/-tolerance seconds of each action's time_seconds.

    Joins on ``match_key`` (bigint, NOT match_id which differs per table).
    """
    if not action_times:
        return pd.DataFrame(columns=TRACKING_RAW_COLS)
    col_list = ", ".join(TRACKING_RAW_COLS)
    window_clauses = " OR ".join(
        f"(period = {p} AND timestamp_seconds BETWEEN {t - tolerance} AND {t + tolerance})" for p, t in action_times
    )
    sql = f"""
        SELECT {col_list}
        FROM {LH_TABLE_FRAMES}
        WHERE source_provider = '{tracking_source_provider}'
          AND match_key = {match_key}
          AND ({window_clauses})
        ORDER BY period, frame, player_id
    """
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame.from_records(rows, columns=cols)


def _pull_skillcorner_frames_window(
    cur,
    match_key: int,
    n_target_frames: int = 250,
) -> pd.DataFrame:
    """Contiguous frame window for SkillCorner --- ~250 frames @ 10 Hz = 25s.

    Stratified across both periods. Used when actions can't be pulled from
    fct_action_values (SkillCorner has no entries there).
    """
    cur.execute(
        f"""
        SELECT period, MIN(frame) AS min_f
        FROM {LH_TABLE_FRAMES}
        WHERE match_key = {match_key} AND source_provider = 'skillcorner'
        GROUP BY period ORDER BY period
        """,
    )
    period_rows = cur.fetchall()
    if not period_rows:
        return pd.DataFrame()
    n_per_period = max(1, n_target_frames // max(len(period_rows), 1))
    where_clauses: list[str] = []
    for r in period_rows:
        period_id = int(r[0])
        min_f = int(r[1])
        max_f = min_f + n_per_period
        where_clauses.append(
            f"(period = {period_id} AND frame BETWEEN {min_f} AND {max_f})",
        )
    where_sql = " OR ".join(where_clauses)
    col_list = ", ".join(TRACKING_RAW_COLS)
    cur.execute(
        f"""
        SELECT {col_list}
        FROM {LH_TABLE_FRAMES}
        WHERE source_provider = 'skillcorner' AND match_key = {match_key}
          AND ({where_sql})
        ORDER BY period, frame, player_id
        """,
    )
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return pd.DataFrame.from_records(rows, columns=cols)


def _synthesize_skillcorner_actions(frames_df: pd.DataFrame) -> pd.DataFrame:
    """Synthesize ~10 pseudo-actions from SkillCorner ball trajectories.

    Sample 5 (period, frame) pairs per period using quantile striding. Each
    pseudo-action takes the ball position at that frame as start, and the
    ball position 1s later as end (or +5m if no future frame). Player/team
    IDs are placeholders (1, 1).

    Output columns mirror ``ACTION_RAW_COLS`` so downstream conversion treats
    the rows uniformly with real-action rows.
    """
    if frames_df.empty:
        return pd.DataFrame(columns=ACTION_RAW_COLS)
    fr = (
        frames_df.drop_duplicates(["period", "frame"])[
            ["match_key", "match_id", "period", "frame", "timestamp_seconds", "ball_x", "ball_y"]
        ]
        .dropna(subset=["ball_x", "ball_y"])
        .sort_values(["period", "frame"])
        .reset_index(drop=True)
    )
    if fr.empty:
        return pd.DataFrame(columns=ACTION_RAW_COLS)
    rows: list[dict[str, Any]] = []
    action_id_counter = 1
    for period_id, period_df in fr.groupby("period"):
        period_df = period_df.reset_index(drop=True)
        if len(period_df) == 0:
            continue
        n_samples = min(5, len(period_df))
        idxs = np.linspace(0, len(period_df) - 1, n_samples, dtype=int)
        for i in idxs:
            start_row = period_df.iloc[i]
            t_target = float(start_row["timestamp_seconds"]) + 1.0
            future = period_df[period_df["timestamp_seconds"] >= t_target]
            if len(future) > 0:
                end_row = future.iloc[0]
                end_x = float(end_row["ball_x"])
                end_y = float(end_row["ball_y"])
            else:
                end_x = float(start_row["ball_x"]) + 5.0
                end_y = float(start_row["ball_y"])
            rows.append(
                {
                    "match_key": int(start_row["match_key"]),
                    "match_id": str(start_row["match_id"]),
                    "period": int(period_id),
                    "time_seconds": float(start_row["timestamp_seconds"]),
                    "action_id": int(action_id_counter),
                    "team_id": 1,
                    "player_id": 1,
                    "start_x": float(start_row["ball_x"]),
                    "start_y": float(start_row["ball_y"]),
                    "end_x": end_x,
                    "end_y": end_y,
                    "action_type": "pass",
                    "action_result": "success",
                    "bodypart": "foot",
                    "data_source": "skillcorner_synthetic_from_ball_trajectory",
                },
            )
            action_id_counter += 1
    return pd.DataFrame(rows, columns=ACTION_RAW_COLS)


def _convert_actions_to_silly_kicks(
    raw_actions: pd.DataFrame,
    silly_kicks_name: str,
    frame_game_id: str | None = None,
) -> pd.DataFrame:
    """Lakehouse 120x80 actions -> silly_kicks 105x68 m action rows.

    - Unit-converts start_x/start_y/end_x/end_y.
    - Renames ``period`` -> ``period_id``.
    - Casts identifiers to nullable Int64 (player_id/team_id are nullable
      because action rows can have NaN player_id on some events).
    - Stamps ``game_id`` to ``frame_game_id`` (the matching tracking
      ``match_id`` string emitted by the lakehouse adapter / silly_kicks
      adapter). Without this harmonization, action ``game_id`` would be
      bigint (match_key) while frame ``game_id`` is a provider string
      (e.g. 'J03WOH'), and pandas concat + parquet write would fail with
      a dtype clash.
    - Stamps ``source_provider`` = silly_kicks_name.
    """
    if len(raw_actions) == 0:
        return pd.DataFrame()
    a = raw_actions.copy()
    a["start_x"] = pd.to_numeric(a["start_x"], errors="coerce") * X_SCALE
    a["start_y"] = pd.to_numeric(a["start_y"], errors="coerce") * Y_SCALE
    a["end_x"] = pd.to_numeric(a["end_x"], errors="coerce") * X_SCALE
    a["end_y"] = pd.to_numeric(a["end_y"], errors="coerce") * Y_SCALE
    a = a.rename(columns={"period": "period_id"})
    if frame_game_id is not None:
        a["game_id"] = str(frame_game_id)
    else:
        # Fallback: lakehouse match_id (string in fct_tracking_frames; bigint
        # in fct_action_values). Stringify either way.
        if "match_id" in a.columns:
            a["game_id"] = a["match_id"].astype(str)
        else:
            a["game_id"] = a["match_key"].astype(str)
    a["action_id"] = pd.to_numeric(a["action_id"], errors="coerce").astype("Int64")
    a["period_id"] = pd.to_numeric(a["period_id"], errors="coerce").astype("Int64")
    # team_id and player_id stay as their native dtype --- for sportec /
    # metrica / skillcorner these are provider-native strings (e.g.
    # 'DFL-CLU-00000P') that match the tracking-side identifiers exactly.
    # Force-casting to Int64 wipes them to NaN and breaks _resolve_action_frame_context's
    # opposite-team filter. NaN entries (rare; some action types lack a player) are
    # preserved as None / NaN in object dtype.
    if "team_id" in a.columns:
        a["team_id"] = a["team_id"].where(a["team_id"].notna(), other=None)
    if "player_id" in a.columns:
        a["player_id"] = a["player_id"].where(a["player_id"].notna(), other=None)
    a["time_seconds"] = pd.to_numeric(a["time_seconds"], errors="coerce").astype("float64")
    a["source_provider"] = silly_kicks_name
    return a


def _convert_lakehouse_to_silly_kicks_frames(
    raw_frames: pd.DataFrame,
    silly_kicks_name: str,
) -> pd.DataFrame:
    """Wide-form lakehouse rows -> silly_kicks long-form 19-col tracking schema.

    Hybrid speed policy (ADR-004 #7): ``speed_ms`` from the lakehouse is
    provider-native (per PR-S19 baselines), so the adapter threads it
    through as ``speed_native``. Any rows where the provider didn't supply
    speed (NaN) are filled by ``_derive_speed`` and tagged
    ``speed_source = "derived"``.

    Sportec/idsse routes through the native sportec adapter; Metrica /
    SkillCorner route through the kloppy gateway (the matching silly_kicks
    pattern --- see ADR-004).
    """
    sys.path.insert(0, str(REPO_ROOT))
    from silly_kicks.tracking.utils import _derive_speed
    from tests.datasets.tracking._lakehouse_adapter import (
        lakehouse_to_kloppy_dataset,
        lakehouse_to_sportec_input,
    )

    if silly_kicks_name == "sportec":
        sportec_input = lakehouse_to_sportec_input(raw_frames)
        sportec_input = sportec_input.dropna(subset=["x_centered", "y_centered"])
        if sportec_input.empty:
            return pd.DataFrame()
        from silly_kicks.tracking.sportec import convert_to_frames as sportec_convert

        home_team_str = str(
            sportec_input.loc[~sportec_input["is_ball"], "team_id"].dropna().iloc[0],
        )
        frames, _ = sportec_convert(
            sportec_input,
            home_team_id=home_team_str,
            home_team_start_left=True,
        )
        # Backfill any remaining NaN speed via finite differences.
        frames = _derive_speed(frames)
        return frames

    # Metrica / SkillCorner via kloppy gateway.
    from kloppy.domain import Provider  # type: ignore[reportMissingImports]

    provider_enum = {
        "metrica": Provider.METRICA,
        "skillcorner": Provider.SKILLCORNER,
    }[silly_kicks_name]
    ds = lakehouse_to_kloppy_dataset(raw_frames, provider_enum)
    if len(ds.records) == 0:
        return pd.DataFrame()
    from silly_kicks.tracking.kloppy import convert_to_frames as kloppy_convert

    frames, _ = kloppy_convert(ds)
    frames = _derive_speed(frames)
    return frames


def _build_combined_slim(
    actions_silly: pd.DataFrame,
    frames_silly: pd.DataFrame,
) -> pd.DataFrame:
    """Stack actions + frames into a single long-form parquet with __kind discriminator.

    Where action and frame DataFrames share a column with conflicting dtypes
    (e.g. ``team_id`` is Int64 on actions but object/string on frames in the
    sportec/kloppy adapters), coerce both sides to ``object`` BEFORE concat
    so pyarrow can serialize the resulting column without a dtype clash on
    parquet write.
    """
    a = actions_silly.copy()
    a["__kind"] = "action"
    f = frames_silly.copy()
    f["__kind"] = "frame"

    # Align dtypes on shared columns where the action side is Int64-nullable
    # and the frame side is object (frame adapters emit string identifiers
    # for sportec/kloppy outputs). Cast action side to object so concat
    # produces a clean object column that pyarrow can write.
    for col in ("team_id", "player_id", "game_id"):
        if col in a.columns and col in f.columns:
            if pd.api.types.is_object_dtype(f[col]) and not pd.api.types.is_object_dtype(a[col]):
                # Stringify action-side values; keep NaN as None for parquet null.
                a[col] = a[col].astype("object").where(a[col].notna(), other=None)
                a[col] = a[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
            elif pd.api.types.is_object_dtype(a[col]) and not pd.api.types.is_object_dtype(f[col]):
                f[col] = f[col].astype("object").where(f[col].notna(), other=None)
                f[col] = f[col].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))

    out = pd.concat([a, f], ignore_index=True, sort=False)
    return out


def _verify_link_rate(
    combined: pd.DataFrame,
    tolerance_seconds: float = LINK_TOLERANCE_SECONDS,
) -> tuple[float, int, int]:
    """Run link_actions_to_frames over the slim parquet's split. Returns
    (link_rate, n_in, n_linked)."""
    sys.path.insert(0, str(REPO_ROOT))
    from silly_kicks.tracking.utils import link_actions_to_frames

    actions = combined[combined["__kind"] == "action"][["action_id", "period_id", "time_seconds"]].copy()
    actions = actions.dropna(subset=["action_id", "period_id", "time_seconds"])
    actions["action_id"] = pd.to_numeric(actions["action_id"], errors="coerce").astype("int64")
    actions["period_id"] = pd.to_numeric(actions["period_id"], errors="coerce").astype("int64")

    frames = combined[combined["__kind"] == "frame"][
        ["period_id", "frame_id", "time_seconds", "source_provider"]
    ].copy()
    frames = frames.dropna(subset=["period_id", "frame_id", "time_seconds"])
    frames["period_id"] = pd.to_numeric(frames["period_id"], errors="coerce").astype("int64")

    if len(actions) == 0 or len(frames) == 0:
        return 0.0, len(actions), 0
    _pointers, report = link_actions_to_frames(
        actions,
        frames,
        tolerance_seconds=tolerance_seconds,
    )
    return float(report.link_rate), report.n_actions_in, report.n_actions_linked


def _process_lakehouse_provider(
    cur,
    silly_kicks_name: str,
    lh_action_src: str,
    lh_track_src: str,
) -> dict[str, Any]:
    """Pull + convert + write parquet for a provider with both actions and frames."""
    print(f"\n[probe] === provider {silly_kicks_name} ===")
    match_key = _find_match_key_with_both(cur, lh_action_src, lh_track_src, min_actions=50)
    if match_key is None:
        print("  [skip] no match_key present in both tables for this provider")
        return {"status": "skipped", "reason": "no overlapping match"}
    print(f"  match_key = {match_key}")

    actions_raw = _pull_actions(cur, lh_action_src, match_key)
    print(f"  pulled {len(actions_raw)} action rows")
    if actions_raw.empty:
        return {"status": "skipped", "reason": "no actions"}

    action_times = [
        (int(p), float(t))
        for p, t in zip(actions_raw["period"], actions_raw["time_seconds"], strict=False)
        if pd.notna(p) and pd.notna(t)
    ]
    frames_raw = _pull_frames_around_actions(
        cur,
        lh_track_src,
        match_key,
        action_times,
    )
    print(f"  pulled {len(frames_raw)} frame rows (raw lakehouse wide-form)")
    if frames_raw.empty:
        return {"status": "skipped", "reason": "no frames in window"}

    return _convert_and_write(silly_kicks_name, actions_raw, frames_raw, match_key)


def _process_skillcorner(cur) -> dict[str, Any]:
    """SkillCorner: no fct_action_values rows, so synthesize from ball track."""
    print("\n[probe] === provider skillcorner ===")
    match_key = _find_skillcorner_match_key(cur)
    if match_key is None:
        return {"status": "skipped", "reason": "no skillcorner tracking match"}
    print(f"  match_key = {match_key} (skillcorner --- actions synthesized from ball track)")

    frames_raw = _pull_skillcorner_frames_window(cur, match_key)
    print(f"  pulled {len(frames_raw)} raw frame rows")
    if frames_raw.empty:
        return {"status": "skipped", "reason": "no frames"}

    actions_raw = _synthesize_skillcorner_actions(frames_raw)
    print(f"  synthesized {len(actions_raw)} pseudo-actions from ball trajectory")
    if actions_raw.empty:
        return {"status": "skipped", "reason": "no synthesizable ball-track actions"}

    return _convert_and_write("skillcorner", actions_raw, frames_raw, match_key)


def _convert_and_write(
    silly_kicks_name: str,
    actions_raw: pd.DataFrame,
    frames_raw: pd.DataFrame,
    match_key: int,
) -> dict[str, Any]:
    """Shared conversion + parquet emit for any provider."""
    frames_silly = _convert_lakehouse_to_silly_kicks_frames(frames_raw, silly_kicks_name)
    print(f"  converted -> {len(frames_silly)} silly-kicks frame rows")
    # Use the frames' canonical game_id (provider-native string) for actions
    # too, so concat + parquet write don't run into a bigint-vs-string clash.
    frame_game_id: str | None = None
    if not frames_silly.empty and "game_id" in frames_silly.columns:
        frame_game_id = str(frames_silly["game_id"].iloc[0])
    actions_silly = _convert_actions_to_silly_kicks(
        actions_raw,
        silly_kicks_name,
        frame_game_id=frame_game_id,
    )
    print(f"  converted -> {len(actions_silly)} silly-kicks action rows")
    if frames_silly.empty or actions_silly.empty:
        return {"status": "skipped", "reason": "empty after conversion", "match_key": match_key}

    combined = _build_combined_slim(actions_silly, frames_silly)
    out_path = SLIM_DIR / f"{silly_kicks_name}_slim.parquet"
    combined.to_parquet(out_path, index=False)
    size_kb = out_path.stat().st_size / 1024.0
    n_actions = int((combined["__kind"] == "action").sum())
    n_frame_rows = int((combined["__kind"] == "frame").sum())
    print(
        f"  wrote {out_path.name}: {len(combined)} rows, {size_kb:.0f} KB (actions={n_actions}, frames={n_frame_rows})",
    )

    # Verify linkage rate.
    link_rate, n_in, n_linked = _verify_link_rate(combined)
    print(f"  link_rate (0.2s tolerance) = {link_rate:.3f} ({n_linked}/{n_in})")

    # Speed source breakdown.
    fr = combined[combined["__kind"] == "frame"]
    n_speed_pop = int(pd.to_numeric(fr["speed"], errors="coerce").notna().sum())
    n_speed_native = int((fr["speed_source"] == "native").sum())
    n_speed_derived = int((fr["speed_source"] == "derived").sum())
    print(
        f"  speed: {n_speed_pop} non-NaN rows | native={n_speed_native} derived={n_speed_derived}",
    )

    # Off-pitch sanity (silly-kicks 105x68).
    x = pd.to_numeric(fr["x"], errors="coerce")
    y = pd.to_numeric(fr["y"], errors="coerce")
    off_x = float(((x < 0) | (x > 105)).mean()) if x.notna().any() else 0.0
    off_y = float(((y < 0) | (y > 68)).mean()) if y.notna().any() else 0.0
    print(f"  off_pitch_x_rate={off_x:.4f} off_pitch_y_rate={off_y:.4f}")

    return {
        "status": "ok",
        "match_key": match_key,
        "n_actions": n_actions,
        "n_frame_rows": n_frame_rows,
        "link_rate": link_rate,
        "speed_rows_native": n_speed_native,
        "speed_rows_derived": n_speed_derived,
        "speed_rows_total_non_nan": n_speed_pop,
        "off_pitch_x_rate": off_x,
        "off_pitch_y_rate": off_y,
    }


def _build_skeleton_baselines(
    per_provider_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the skeleton baselines JSON (per-feature stats null; backfilled in Task 13)."""
    null_stats = {
        "nearest_defender_distance_p25": None,
        "nearest_defender_distance_p50": None,
        "nearest_defender_distance_p75": None,
        "nearest_defender_distance_p99": None,
        "actor_speed_p25": None,
        "actor_speed_p50": None,
        "actor_speed_p75": None,
        "actor_speed_p99": None,
        "receiver_zone_density_p25": None,
        "receiver_zone_density_p50": None,
        "receiver_zone_density_p75": None,
        "receiver_zone_density_p99": None,
        "defenders_in_triangle_to_goal_p25": None,
        "defenders_in_triangle_to_goal_p50": None,
        "defenders_in_triangle_to_goal_p75": None,
        "defenders_in_triangle_to_goal_p99": None,
    }

    providers_block: dict[str, dict[str, Any]] = {}
    for sk_name in ("sportec", "metrica", "skillcorner"):
        s = per_provider_summary.get(sk_name, {})
        entry: dict[str, Any] = {
            "slim_slice": f"action_context_slim/{sk_name}_slim.parquet",
            "match_key_sampled": s.get("match_key"),
            "n_actions_sampled": s.get("n_actions"),
            "n_frame_rows_sampled": s.get("n_frame_rows"),
            "linkage_rate_within_0p2s": s.get("link_rate"),
            "probe_status": s.get("status", "skipped"),
            "speed_rows_native": s.get("speed_rows_native"),
            "speed_rows_derived": s.get("speed_rows_derived"),
            "off_pitch_x_rate": s.get("off_pitch_x_rate"),
            "off_pitch_y_rate": s.get("off_pitch_y_rate"),
        }
        if sk_name == "skillcorner":
            entry["actions_source"] = "synthetic_from_ball_trajectory_lakehouse_has_no_skillcorner_actions"
        entry.update(null_stats)
        entry["note"] = "distribution stats backfilled in Task 13 from this slim slice"
        providers_block[sk_name] = entry

    # PFF: forward reference to Task 13 synthetic baseline.
    pff_entry: dict[str, Any] = {
        "source": "synthetic_pff_medium_halftime",
        "note": (
            "PFF-license: no lakehouse probe; baselines computed in Task 13 "
            "from tests/datasets/tracking/pff/medium_halftime.parquet using "
            "Task-7 features"
        ),
    }
    pff_entry.update(null_stats)
    providers_block["pff"] = pff_entry

    return {
        "probe_run_date": "2026-05-01",
        "probe_run_source_lakehouse_tables": [
            LH_TABLE_ACTIONS,
            LH_TABLE_FRAMES,
        ],
        "lakehouse_units_source": "statsbomb_120x80",
        "silly_kicks_units_target": "metric_105x68",
        "lakehouse_consumer_principle": (
            "raw provider-native fields only; lakehouse-COMPUTED values "
            "(vaep_value, offensive_value, defensive_value, "
            "pitch_control_value, voronoi_area, distance_to_ball, "
            "acceleration_ms2) NOT pulled. speed_ms IS pulled because it is "
            "provider-native per PR-S19 baselines (speed_native_supplied=True)."
        ),
        "tolerance_seconds_around_actions": TOLERANCE_SECONDS_FRAME_WINDOW,
        "linkage_tolerance_seconds": LINK_TOLERANCE_SECONDS,
        "n_actions_per_provider_target": N_ACTIONS_TARGET,
        "providers": providers_block,
    }


def main() -> int:
    print("[probe] PR-S20 Task 0 --- action_context slim slices + skeleton baselines")
    print(f"[probe] Outputs: {SLIM_DIR}/, {OUTPUT_JSON}")
    SLIM_DIR.mkdir(parents=True, exist_ok=True)

    conn = _databricks_connect()
    if conn is None:
        print("[probe] FATAL: cannot connect to lakehouse; aborting.")
        return 1

    per_provider_summary: dict[str, dict[str, Any]] = {}

    try:
        with conn.cursor() as cur:
            print("\n[probe] verifying provider names in the lakehouse...")
            distinct_action_sources = _show_distinct_data_sources(cur)
            print(f"  fct_action_values.data_source: {sorted(distinct_action_sources)}")
            distinct_tracking_providers = _show_distinct_source_providers(cur)
            print(f"  fct_tracking_frames.source_provider: {sorted(distinct_tracking_providers)}")

            for silly_kicks_name, lh_action_src, lh_track_src in PROVIDER_PAIRS:
                if silly_kicks_name == "skillcorner" or lh_action_src is None:
                    info = _process_skillcorner(cur)
                else:
                    if lh_action_src not in distinct_action_sources:
                        print(f"\n[probe] === provider {silly_kicks_name} ===")
                        print(f"  [skip] data_source '{lh_action_src}' absent from fct_action_values")
                        info = {"status": "skipped", "reason": f"data_source '{lh_action_src}' absent"}
                    elif lh_track_src not in distinct_tracking_providers:
                        print(f"\n[probe] === provider {silly_kicks_name} ===")
                        print(
                            f"  [skip] source_provider '{lh_track_src}' absent from fct_tracking_frames",
                        )
                        info = {"status": "skipped", "reason": f"source_provider '{lh_track_src}' absent"}
                    else:
                        try:
                            info = _process_lakehouse_provider(
                                cur,
                                silly_kicks_name,
                                lh_action_src,
                                lh_track_src,
                            )
                        except Exception as e:
                            print(f"  [error] {silly_kicks_name}: {type(e).__name__}: {e}")
                            info = {"status": "error", "reason": str(e)}
                per_provider_summary[silly_kicks_name] = info
    finally:
        conn.close()

    print("\n[probe] writing skeleton baselines JSON")
    out_json = _build_skeleton_baselines(per_provider_summary)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"[probe] wrote {OUTPUT_JSON}")

    print("\n[probe] summary:")
    for sk_name, summary in per_provider_summary.items():
        print(f"  {sk_name}: {summary}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
