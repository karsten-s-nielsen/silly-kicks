"""Databricks bronze loader for the TF-24 calibration harness.

Reads soccer_analytics.bronze.{provider}_{tracking,events} (id col: match_id), runs the CURRENT
silly-kicks converters (so calibration reflects current output), and yields the uniform
(provider, match_id, actions, frames, home_team_id) tuple. Operator-scale / fallback path + the
bronze.spadl_actions xT-corpus source (IDSSE is public on pining; this is not its only source).

Env: DATABRICKS_HOST, DATABRICKS_HTTP_PATH, DATABRICKS_TOKEN.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pandas as pd

# Allowlist of known bronze providers — the ONLY values interpolated into a table name (M5).
# match_id is ALWAYS parameterized; provider is validated against this set, never free-form.
_ALLOWED_PROVIDERS = frozenset({"idsse", "skillcorner", "gradientsports", "metrica", "sportec", "statsbomb", "wyscout"})
_FETCH_BATCH = 50_000  # L4: stream big tracking pulls in batches, not one giant fetchall


def _connect():
    try:
        import databricks.sql as dbsql
    except ImportError as exc:  # actionable hint
        raise RuntimeError(
            "databricks-sql-connector is required for the Databricks loader: pip install databricks-sql-connector"
        ) from exc
    return dbsql.connect(
        server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
    )


def _table(provider: str, kind: str) -> str:
    """Build a fully-qualified bronze table name from an ALLOWLISTED provider (M5)."""
    if provider not in _ALLOWED_PROVIDERS:
        raise ValueError(f"provider {provider!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    return f"soccer_analytics.bronze.{provider}_{kind}"


def _query_param(cursor, sql: str, params=None) -> pd.DataFrame:
    """Execute a PARAMETERIZED query and batch-fetch into a DataFrame (L4)."""
    cursor.execute(sql, params or {})
    cols = [d[0] for d in cursor.description]
    rows = []
    while True:
        batch = cursor.fetchmany(_FETCH_BATCH)
        if not batch:
            break
        rows.extend(batch)
    return pd.DataFrame(rows, columns=cols)


def load_matches(
    *, providers: list[str], match_ids: dict[str, list[str]] | None = None
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) from bronze."""
    conn = _connect()
    try:
        cur = conn.cursor()
        for provider in providers:
            t_tracking, t_events = _table(provider, "tracking"), _table(provider, "events")
            ids = match_ids.get(provider) if match_ids else None
            if ids is None:
                # Table name is allowlist-validated (_table); no user input interpolated.
                ids = [
                    r[0]
                    for r in _query_param(cur, f"SELECT DISTINCT match_id FROM {t_tracking}").itertuples(index=False)  # noqa: S608
                ]
            for mid in ids:
                # Table from allowlist; match_id PARAMETERIZED (M5) — never f-string-interpolated.
                raw_frames = _query_param(cur, f"SELECT * FROM {t_tracking} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                raw_events = _query_param(cur, f"SELECT * FROM {t_events} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                actions, frames, home = _convert(provider, raw_events, raw_frames)
                yield provider, str(mid), actions, frames, home
        cur.close()
    finally:
        conn.close()


def _convert(provider: str, raw_events: pd.DataFrame, raw_frames: pd.DataFrame):
    """Convert bronze rows to (actions, frames, home_team_id) via silly-kicks converters.

    Bronze tables are the lakehouse's PARSED provider output (already in silly-kicks-converter
    input shape). For Sportec/IDSSE this is the ``tracking.sportec.convert_to_frames`` input shape;
    for the SPADL side it is the ``spadl.sportec.convert_to_actions`` input shape. The home team +
    starting direction come from ``bronze.tracking_player_metadata`` (NOT events.iloc[0]).
    """
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    # IDSSE/sportec bronze is the most common operator path; others can be added as needed.
    if provider in ("idsse", "sportec"):
        from silly_kicks.spadl import sportec as sportec_spadl
        from silly_kicks.tracking import sportec as sportec_tracking

        home_team_id = str(raw_frames["team_id"].dropna().mode().iloc[0])  # placeholder; see NOTE
        home_start_left = True
        frames, _r = sportec_tracking.convert_to_frames(
            raw_frames, home_team_id=home_team_id, home_team_start_left=home_start_left
        )
        actions, _r2 = sportec_spadl.convert_to_actions(
            raw_events, home_team_id=home_team_id, home_team_start_left=home_start_left
        )
        return actions, derive_velocities(smooth_frames(frames)), home_team_id

    raise NotImplementedError(
        f"Databricks _convert for provider {provider!r} not yet wired; pining is the primary "
        "source for skillcorner/idsse/gradientsports. Add the bronze->converter mapping here for "
        "operator-scale runs of this provider (see tracking_player_metadata for home_team_id)."
    )
