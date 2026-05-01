"""Build lakehouse-derived CI fixtures for the kloppy-gateway providers.

Pulls a small slice of real wide-form tracking rows from the lakehouse
mart (``soccer_analytics.dev_gold.fct_tracking_frames``) for Metrica +
SkillCorner + Sportec/IDSSE and writes them as committed parquet
fixtures under ``tests/datasets/tracking/<provider>/lakehouse_derived.parquet``.

CI tests load these parquets and rebuild a kloppy ``TrackingDataset``
on the fly via the same ``_lakehouse_to_kloppy_dataset`` helper used by
the e2e sweep, then exercise the gateway / sportec adapter against the
real-distribution data.

PFF is intentionally absent --- its data is not redistributable per the
license memorialized in PR-S18 (memory:reference_pff_data_local).
PFF CI coverage uses the synthetic ``realistic.parquet`` driven by the
empirical baseline JSON.

Run once on a developer machine with Databricks env vars set:

    python scripts/build_lakehouse_ci_fixtures.py

The fixtures are slim (~50 KB each) --- 1 match, ~1500 rows, 10 columns.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKING_DIR = REPO_ROOT / "tests" / "datasets" / "tracking"

# Slim per-provider sample. ~22 players * 70 frames = ~1500 rows; trim
# columns to the canonical wide-form set the CI consumers need.
N_FRAMES_PER_PROVIDER = 70
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
    raw_host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get(
        "DATABRICKS_HOST",
        "",
    )
    server_hostname = raw_host.removeprefix("https://").removeprefix("http://").rstrip("/")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH", "")
    if http_path.startswith("//"):
        http_path = http_path[1:]
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if not (server_hostname and http_path and token):
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


def _pull_slim_slice(provider_raw: str) -> pd.DataFrame | None:
    conn = _databricks_connect()
    if conn is None:
        print(f"  [skip] {provider_raw}: no Databricks connectivity")
        return None
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT match_id
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                GROUP BY match_id
                LIMIT 1
                """
            )
            r = cur.fetchone()
            if r is None:
                return None
            match_id_value = r[0]
            cur.execute(
                f"""
                SELECT min(frame) AS min_f
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                  AND match_id = '{match_id_value}'
                  AND period = 1
                """
            )
            f_row = cur.fetchone()
            if f_row is None or f_row[0] is None:
                return None
            min_f = int(f_row[0])
            max_f = min_f + N_FRAMES_PER_PROVIDER
            col_list = ", ".join(KEPT_COLUMNS)
            cur.execute(
                f"""
                SELECT {col_list}
                FROM soccer_analytics.dev_gold.fct_tracking_frames
                WHERE source_provider = '{provider_raw}'
                  AND match_id = '{match_id_value}'
                  AND period = 1
                  AND frame BETWEEN {min_f} AND {max_f}
                """
            )
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
    finally:
        conn.close()
    return pd.DataFrame.from_records(rows, columns=cols)


def main() -> int:
    targets = [
        ("metrica", "metrica"),
        ("skillcorner", "skillcorner"),
        ("idsse", "sportec"),  # lakehouse=idsse -> local-fixture-name=sportec
    ]
    for raw_name, local_name in targets:
        print(f"[lakehouse->fixture] {raw_name} -> {local_name}")
        df = _pull_slim_slice(raw_name)
        if df is None or len(df) == 0:
            print(f"  [skip] {raw_name}: no rows")
            continue
        # Anonymize match_id to prevent license risk on identifying real
        # match metadata; the structural distribution (positions, NaN
        # patterns, ball trajectory) is what the CI test actually exercises.
        df["match_id"] = f"{local_name}_lh_synth_match_001"
        out_path = TRACKING_DIR / local_name / "lakehouse_derived.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        size_kb = out_path.stat().st_size / 1024.0
        print(f"  wrote {out_path.name}: {len(df)} rows, {size_kb:.0f} KB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
