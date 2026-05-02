"""Build production-shape fixtures for sportec + metrica converter tests.

Two extractors per provider, each with two variants (``--variant``):

1. IDSSE (Sportec/DFL): pulls from ``bronze.idsse_events`` on Databricks
   via ``databricks-sql-connector`` using env-var auth. Source match:
   ``idsse_J03WMX`` (known to contain throwOut + punt qualifiers — public
   DFL competition identifier, no PII).

   - ``--variant default`` -> ~200-400 representative rows including all
     keeper-relevant rows (``sample_match.parquet``; contract-test
     fixture).
   - ``--variant per_period`` -> full match (``per_period_match.parquet``)
     used by ``tests/invariants/test_direction_of_play.py`` for the
     per-(team, period) orientation invariant added in PR-S23.

2. Metrica: two source strategies depending on variant.

   - ``--variant default`` -> subsets the kloppy-vendored fixture
     ``tests/datasets/kloppy/metrica_events.json`` (Metrica Sample Game
     2, period 1 only), converts to bronze shape. No network required.
   - ``--variant per_period`` -> pulls Metrica Sample Game 1 from
     ``bronze.metrica_events`` on Databricks (Sample Game 1 has both
     periods; kloppy ships only Sample Game 2). Applies the 0-1 -> 0-105
     coordinate rescale + column projection that matches the
     silly-kicks-input shape produced by the lakehouse adapter
     ``adapt_metrica_events_for_silly_kicks``.

Output paths:

- ``--variant default``  -> ``tests/datasets/{provider}/sample_match.parquet``
- ``--variant per_period`` -> ``tests/datasets/{provider}/per_period_match.parquet``

Security: this script reads ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` /
``DATABRICKS_HTTP_PATH`` from environment variables and never echoes the
values to stdout / stderr. Errors on missing creds report only the
variable NAMES that are missing.

Usage::

    # Both extractors, default contract-test fixtures.
    python scripts/extract_provider_fixtures.py

    # Single provider, default variant.
    python scripts/extract_provider_fixtures.py --provider idsse

    # Per-period invariant fixtures (PR-S23 / silly-kicks 3.0.1).
    python scripts/extract_provider_fixtures.py --provider idsse --variant per_period
    python scripts/extract_provider_fixtures.py --provider metrica --variant per_period
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
_IDSSE_PER_PERIOD_OUT = _REPO_ROOT / "tests" / "datasets" / "idsse" / "per_period_match.parquet"
_METRICA_OUT = _REPO_ROOT / "tests" / "datasets" / "metrica" / "sample_match.parquet"
_METRICA_PER_PERIOD_OUT = _REPO_ROOT / "tests" / "datasets" / "metrica" / "per_period_match.parquet"
_METRICA_KLOPPY_SOURCE = _REPO_ROOT / "tests" / "datasets" / "kloppy" / "metrica_events.json"

# IDSSE source match — known to contain throwOut + punt qualifiers
# (verified via direct probe against bronze.idsse_events 2026-04-29).
# DFL competition identifier — public, no PII.
_IDSSE_SOURCE_MATCH_ID = "idsse_J03WMX"

# Metrica source match for the per_period variant: Sample Game 1 has
# both periods (Sample Game 2 ships in kloppy but is period-1-only).
_METRICA_PER_PERIOD_SOURCE_MATCH_ID = "Sample_Game_1"

# Metrica standard pitch dimensions -- Sample Game 1 raw bronze ships
# coords in 0-1 normalized form; rescale to silly-kicks-input
# (0-105 m by 0-68 m) at extraction time.
_METRICA_PITCH_LENGTH_M = 105.0
_METRICA_PITCH_WIDTH_M = 68.0

# Maximum rows in the contract-test (default-variant) IDSSE fixture.
# The per-period variant skips this cap to preserve full-match shot density.
_IDSSE_MAX_ROWS = 400


def _extract_idsse(out_path: Path, *, variant: str = "default") -> None:
    """Pull a subset of bronze.idsse_events for the source match.

    ``variant='default'`` keeps the existing ~400-row stratified subset
    (used by contract tests). ``variant='per_period'`` skips the row cap
    so the full match is preserved with per-period shot density intact —
    required by the per-(team, period) orientation invariant added in
    PR-S23 (silly-kicks 3.0.1).

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

    # Query: per variant.
    # ORDER BY (period, timestamp_seconds, event_id) for deterministic output.
    # Fully qualified table name: catalog.schema.table (Unity Catalog).
    # The S608 SQL-injection rule is suppressed in pyproject.toml's
    # per-file-ignores for this file: interpolated values are module-level
    # constants (no user input), not a real injection vector.
    if variant == "per_period":
        # Full match — preserve per-period shot density for the
        # tests/invariants/test_direction_of_play.py per-(team, period)
        # invariant added in PR-S23 (silly-kicks 3.0.1).
        query = f"""
        SELECT * FROM soccer_analytics.bronze.idsse_events
        WHERE match_id = '{_IDSSE_SOURCE_MATCH_ID}'
        ORDER BY period, timestamp_seconds, event_id
        """
    else:
        # Default contract-test fixture: keep all GK-relevant Play rows +
        # a stratified sample of other rows (~400 rows total).
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


def _extract_metrica_per_period(out_path: Path) -> None:
    """Pull Metrica Sample Game 1 from bronze.metrica_events for the per-(team, period) invariant.

    Sample Game 1 has both periods (Sample Game 2 ships in kloppy but is
    period-1-only). Applies the same 0-1 -> 0-105 / 0-68 coordinate rescale
    + column projection that the lakehouse adapter
    ``adapt_metrica_events_for_silly_kicks`` performs, so the output
    matches the silly-kicks-input bronze schema (identical to
    ``sample_match.parquet``).

    Reads ``DATABRICKS_HOST`` / ``DATABRICKS_TOKEN`` / ``DATABRICKS_HTTP_PATH``
    from env (same security posture as ``_extract_idsse``).
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

    query = f"""
    SELECT * FROM soccer_analytics.bronze.metrica_events
    WHERE match_id = '{_METRICA_PER_PERIOD_SOURCE_MATCH_ID}'
    ORDER BY period, start_time_s, event_id
    """

    print("Connecting to Databricks (host/token/path read from env)...")
    try:
        with dbsql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            with conn.cursor() as cur:
                cur.execute(query)
                src = cur.fetchall_arrow().to_pandas()
    except Exception as exc:
        print(f"ERROR: Databricks query failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"Pulled {len(src)} rows from {_METRICA_PER_PERIOD_SOURCE_MATCH_ID}")

    # Project to silly-kicks-input bronze shape + apply Metrica's standard
    # 0-1 -> 0-105 / 0-68 rescale (equivalent to lakehouse's
    # adapt_metrica_events_for_silly_kicks).
    df = pd.DataFrame(
        {
            "match_id": src["match_id"].astype(str),
            "event_id": src["event_id"],
            "type": src["type"],
            "subtype": src["subtype"],
            "period": src["period"].astype("int64"),
            "start_time_s": src["start_time_s"],
            "end_time_s": src["end_time_s"],
            "player": src["player"],
            "team": src["team"],
            "start_x": src["start_x"].astype(float) * _METRICA_PITCH_LENGTH_M,
            "start_y": src["start_y"].astype(float) * _METRICA_PITCH_WIDTH_M,
            "end_x": src["end_x"].astype(float) * _METRICA_PITCH_LENGTH_M,
            "end_y": src["end_y"].astype(float) * _METRICA_PITCH_WIDTH_M,
        }
    )
    print(f"type counts: {df['type'].value_counts().to_dict()}")

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
    parser.add_argument(
        "--variant",
        choices=["default", "per_period"],
        default="default",
        help=(
            "'default' produces the small ~300-row contract-test fixture "
            "(sample_match.parquet); 'per_period' produces a full-match "
            "fixture (per_period_match.parquet) used by "
            "tests/invariants/test_direction_of_play.py for per-(team, "
            "period) orientation assertions (PR-S23 / silly-kicks 3.0.1)."
        ),
    )
    args = parser.parse_args()

    if args.provider in ("idsse", "all"):
        if args.variant == "per_period":
            _extract_idsse(_IDSSE_PER_PERIOD_OUT, variant="per_period")
        else:
            _extract_idsse(_IDSSE_OUT, variant="default")
    if args.provider in ("metrica", "all"):
        if args.variant == "per_period":
            _extract_metrica_per_period(_METRICA_PER_PERIOD_OUT)
        else:
            _extract_metrica(_METRICA_OUT)

    return 0


if __name__ == "__main__":
    sys.exit(main())
