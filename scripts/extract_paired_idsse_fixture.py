"""Extract paired IDSSE events + tracking fixture for match J03WMX.

Pulls tracking frames from the lakehouse for time windows around shot
and pass events in the existing ``per_period_match.parquet`` events
fixture.  The paired fixture enables testing:

- Bug #7: ``_derive_end_coordinates`` (pass-class end_x/end_y derivation)
- Bug #2: ``defending_gk_from_frames`` fallback (GK features NULL when
  events-based lookback finds no keeper_save)

Run once on a developer machine with Databricks env vars set:

    python scripts/extract_paired_idsse_fixture.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EVENTS_PATH = REPO_ROOT / "tests" / "datasets" / "idsse" / "per_period_match.parquet"
OUT_PATH = REPO_ROOT / "tests" / "datasets" / "idsse" / "paired_tracking.parquet"

# Time windows (period, start_ts, end_ts) -- each covers a shot + surrounding events.
# P1 first shot at ts=102.3 (preceded by 2 passes at ts=95.3, 99.0)
# P2 first shot at ts=636.8 (preceded by passes/tackles at ts=627-635)
TIME_WINDOWS = [
    (1, 90.0, 107.0),  # P1: 17s around first shot, covers 4 events
    (2, 624.0, 640.0),  # P2: 16s around first shot, covers 8 events
]

KEPT_COLUMNS = [
    "match_id",
    "period",
    "frame",
    "timestamp_seconds",
    "player_id",
    "team",
    "team_id",
    "is_goalkeeper",
    "frame_rate",
    "x",
    "y",
    "ball_x",
    "ball_y",
    "speed_ms",
    "source_provider",
]


def _databricks_connect():
    try:
        from databricks import sql  # type: ignore[import-not-found]
    except ImportError:
        return None
    raw_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get("DATABRICKS_HOST", "")
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path.startswith("//"):
        http_path = http_path[1:]
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
        return None
    return sql.connect(
        server_hostname=server_hostname,
        http_path=http_path,
        access_token=token,
    )


def main() -> int:
    conn = _databricks_connect()
    if conn is None:
        print("[error] No Databricks connectivity. Set DATABRICKS_HOST/TOKEN/HTTP_PATH.")
        return 1

    col_list = ", ".join(KEPT_COLUMNS)
    frames: list[pd.DataFrame] = []

    try:
        with conn.cursor() as cur:
            for period, ts_start, ts_end in TIME_WINDOWS:
                print(f"[extract] P{period} ts={ts_start:.0f}-{ts_end:.0f}...")
                cur.execute(
                    f"""
                    SELECT {col_list}
                    FROM soccer_analytics.dev_gold.fct_tracking_frames
                    WHERE source_provider = 'idsse'
                      AND match_id = 'J03WMX'
                      AND period = {period}
                      AND timestamp_seconds BETWEEN {ts_start} AND {ts_end}
                    ORDER BY frame, player_id
                    """  # noqa: S608 -- one-shot extraction script, not user-facing
                )
                cols = [d[0] for d in cur.description]  # type: ignore[reportOptionalIterable]
                rows = cur.fetchall()
                df = pd.DataFrame.from_records(rows, columns=cols)
                print(f"  {len(df)} rows, {df['frame'].nunique()} frames, {df['player_id'].nunique()} players")
                gk_count = df["is_goalkeeper"].sum() if "is_goalkeeper" in df.columns else 0
                print(f"  GK rows: {gk_count}")
                frames.append(df)
    finally:
        conn.close()

    if not frames:
        print("[error] No data extracted.")
        return 1

    combined = pd.concat(frames, ignore_index=True)

    # Keep original match_id (J03WMX) -- DFL DataHub free-sample license
    # permits non-commercial redistribution (same as per_period_match.parquet).
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(OUT_PATH, index=False)
    size_kb = OUT_PATH.stat().st_size / 1024.0
    print(f"\n[done] wrote {OUT_PATH.name}: {len(combined)} rows, {size_kb:.0f} KB")
    print(f"  Periods: {sorted(combined['period'].unique())}")
    print(f"  Players: {combined['player_id'].nunique()}")
    print(f"  Frames:  {combined['frame'].nunique()}")
    print(f"  GK rows: {combined['is_goalkeeper'].sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
