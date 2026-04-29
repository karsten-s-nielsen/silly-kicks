"""Build production-shape fixtures for sportec + metrica converter tests.

Two extractors with two source strategies:

1. IDSSE (Sportec/DFL): pulls a subset of ``bronze.idsse_events`` from the
   Databricks lakehouse via ``databricks-sql-connector`` using env-var auth.
   Source match: ``idsse_J03WMX`` (known to contain throwOut + punt
   qualifiers — public DFL competition identifier, no PII). Subset:
   ~200-400 representative rows including all keeper-relevant rows.

2. Metrica: subsets the already-vendored kloppy fixture
   ``tests/datasets/kloppy/metrica_events.json`` (Metrica Sample Game 2,
   3,620 events) and converts it to the bronze shape expected by
   ``silly_kicks.spadl.metrica.convert_to_actions``. No network required.

Both extractors write their output to
``tests/datasets/{provider}/sample_match.parquet`` as small (~30 KB
compressed) files small enough to commit.

Security: this script reads ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` /
``DATABRICKS_HTTP_PATH`` from environment variables and never echoes the
values to stdout / stderr. Errors on missing creds report only the
variable NAMES that are missing.

Usage::

    # Both extractors (default).
    python scripts/extract_provider_fixtures.py

    # Single provider.
    python scripts/extract_provider_fixtures.py --provider idsse
    python scripts/extract_provider_fixtures.py --provider metrica
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_IDSSE_OUT = _REPO_ROOT / "tests" / "datasets" / "idsse" / "sample_match.parquet"
_METRICA_OUT = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
_METRICA_KLOPPY_SOURCE = _REPO_ROOT / "tests" / "datasets" / "kloppy" / "metrica_events.json"

# IDSSE source match — known to contain throwOut + punt qualifiers
# (verified via direct probe against bronze.idsse_events 2026-04-29).
# DFL competition identifier — public, no PII.
_IDSSE_SOURCE_MATCH_ID = "idsse_J03WMX"

# Maximum rows in the committed fixture.
_IDSSE_MAX_ROWS = 400


def _extract_idsse(out_path: Path) -> None:
    """Pull a subset of bronze.idsse_events for the source match.

    Reads ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` / ``DATABRICKS_HTTP_PATH``
    from env. NEVER echoes those values to stdout / stderr — only their
    presence/absence on the missing-vars error path.
    """
    try:
        from databricks import sql as dbsql
    except ImportError:
        print(
            "ERROR: databricks-sql-connector not installed. Install with: uv pip install databricks-sql-connector",
            file=sys.stderr,
        )
        sys.exit(1)

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")
    http_path = os.environ.get("DATABRICKS_HTTP_PATH")
    missing = [
        n
        for n, v in (
            ("DATABRICKS_HOST", host),
            ("DATABRICKS_TOKEN", token),
            ("DATABRICKS_HTTP_PATH", http_path),
        )
        if not v
    ]
    if missing:
        print(f"ERROR: missing env vars: {missing}", file=sys.stderr)
        sys.exit(1)

    # Query: keep all GK-relevant Play rows + a stratified sample of other rows.
    # ORDER BY (period, timestamp_seconds, event_id) for deterministic output.
    # Fully qualified table name: catalog.schema.table (Unity Catalog).
    # The S608 SQL-injection rule is suppressed in pyproject.toml's
    # per-file-ignores for this file: interpolated values are module-level
    # constants (no user input), not a real injection vector.
    query = f"""
    WITH source AS (
      SELECT * FROM soccer_analytics.bronze.idsse_events WHERE match_id = '{_IDSSE_SOURCE_MATCH_ID}'
    ),
    gk_rows AS (
      SELECT * FROM source
      WHERE event_type = 'Play' AND play_goal_keeper_action IS NOT NULL
    ),
    other_rows AS (
      SELECT * FROM source
      WHERE NOT (event_type = 'Play' AND play_goal_keeper_action IS NOT NULL)
      ORDER BY rand(42)
      LIMIT {_IDSSE_MAX_ROWS - 100}
    )
    SELECT * FROM gk_rows
    UNION ALL
    SELECT * FROM other_rows
    ORDER BY period, timestamp_seconds, event_id
    """

    print("Connecting to Databricks (host/token/path read from env)...")
    try:
        with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                df = cur.fetchall_arrow().to_pandas()
    except Exception as exc:
        # databricks-sql exception messages contain the SQL error text and
        # query ID — useful for debugging, never include the auth token.
        print(f"ERROR: Databricks query failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Pulled {len(df)} rows from {_IDSSE_SOURCE_MATCH_ID}")
    print(f"event_type counts: {df['event_type'].value_counts().to_dict()}")
    if "play_goal_keeper_action" in df.columns:
        gk_counts = df.loc[df["play_goal_keeper_action"].notna(), "play_goal_keeper_action"].value_counts().to_dict()
        print(f"play_goal_keeper_action counts: {gk_counts}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({size_kb:.1f} KB, {len(df)} rows)")


def _extract_metrica(out_path: Path) -> None:
    """Subset the kloppy-vendored Metrica fixture and convert to bronze shape.

    Source: ``tests/datasets/kloppy/metrica_events.json`` (Metrica Sample
    Game 2, public open-data, CC-BY-NC). Subsets the first ~300 events with
    deterministic ordering and converts to bronze schema.
    """
    if not _METRICA_KLOPPY_SOURCE.exists():
        print(f"ERROR: kloppy source not found at {_METRICA_KLOPPY_SOURCE}", file=sys.stderr)
        sys.exit(1)

    with open(_METRICA_KLOPPY_SOURCE, encoding="utf-8") as f:
        raw = json.load(f)

    events_raw = raw.get("data", raw if isinstance(raw, list) else [])

    subset = []
    for ev in events_raw:
        if len(subset) >= 300:
            break
        if not isinstance(ev, dict):
            continue
        subset.append(ev)

    bronze_rows = []
    for i, ev in enumerate(subset):
        typ_raw = ev.get("type")
        typ = typ_raw.get("name") if isinstance(typ_raw, dict) else typ_raw

        sub = ev.get("subtypes") or ev.get("subtype")
        if isinstance(sub, list):
            sub = sub[0].get("name") if (sub and isinstance(sub[0], dict)) else (sub[0] if sub else None)
        elif isinstance(sub, dict):
            sub = sub.get("name")

        period_raw = ev.get("period")
        period = period_raw.get("id") if isinstance(period_raw, dict) else (period_raw or 1)

        start_obj = ev.get("start") or {}
        end_obj = ev.get("end") or {}
        start_time = start_obj.get("time", 0.0) if isinstance(start_obj, dict) else 0.0
        end_time = end_obj.get("time", start_time) if isinstance(end_obj, dict) else start_time

        # kloppy metrica_events.json uses "from" for the event actor and
        # "to" for the target — NOT "player". Each is a dict with "id" and
        # "name" keys.
        from_obj = ev.get("from") or {}
        player = from_obj.get("id") if isinstance(from_obj, dict) else from_obj

        team_obj = ev.get("team") or {}
        team = team_obj.get("id") if isinstance(team_obj, dict) else team_obj

        # Skip rows with no actor — they're informational events (substitutions,
        # set-piece markers, etc.) that the SPADL converter would drop anyway,
        # and they break the cross-provider parity test's GK-detection heuristic.
        if not player:
            continue

        # Defensive coord extraction: keys may exist with None values for
        # off-pitch / non-coord events (substitutions, kickoffs, etc.).
        # Fall back to a neutral on-pitch default in those cases.
        def _coord(obj: object, key: str, default: float) -> float:
            if not isinstance(obj, dict):
                return default
            v = obj.get(key)
            return float(v) if v is not None else default

        start_x = _coord(start_obj, "x", 50.0)
        start_y = _coord(start_obj, "y", 34.0)
        end_x = _coord(end_obj, "x", start_x)
        end_y = _coord(end_obj, "y", start_y)

        bronze_rows.append(
            {
                "match_id": "metrica_sample_game_2",
                "event_id": i,
                "type": str(typ).upper() if typ else "GENERIC",
                "subtype": str(sub).upper() if sub else None,
                "period": int(period),
                "start_time_s": float(start_time),
                "end_time_s": float(end_time),
                "player": str(player) if player else None,
                "team": str(team) if team else None,
                "start_x": float(start_x),
                "start_y": float(start_y),
                "end_x": float(end_x),
                "end_y": float(end_y),
            }
        )

    df = pd.DataFrame(bronze_rows)
    print(f"Subsetted {len(df)} Metrica events from kloppy fixture")
    print(f"type counts: {df['type'].value_counts().to_dict()}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)
    size_kb = out_path.stat().st_size / 1024
    print(f"Wrote {out_path.name} ({size_kb:.1f} KB, {len(df)} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=["idsse", "metrica", "all"],
        default="all",
        help="Which provider's fixture to (re)build.",
    )
    args = parser.parse_args()

    if args.provider in ("idsse", "all"):
        _extract_idsse(_IDSSE_OUT)
    if args.provider in ("metrica", "all"):
        _extract_metrica(_METRICA_OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
