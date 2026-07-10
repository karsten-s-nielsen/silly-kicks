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

# Fixed, fully-qualified gold mart for the owner-gated xT NLL cross-check (read-only).
# A module constant — never interpolated from caller input (mirrors the _ALLOWED_PROVIDERS discipline).
_ACTION_VALUES_TABLE = "soccer_analytics.dev_gold.fct_action_values"
# Only the passes-NLL columns; period/action_id deliberately omitted (unused — keeps the ~8.8M-row pull lean).
_ACTION_VALUES_COLUMNS = "match_id, start_x, start_y, end_x, end_y, action_type, action_result"


def _connect():
    try:
        import databricks.sql as dbsql  # type: ignore[import-not-found]
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


def fetch_action_values(*, max_matches: int | None = None) -> pd.DataFrame:
    """Read the gold action-values mart for the owner-gated xT held-out-NLL cross-check.

    Pulls only the columns the passes transition-NLL path needs. SPADL-id shaping is
    ``shape_action_values``. Read-only.

    Parameters
    ----------
    max_matches : int | None
        When set, restrict to the first ``max_matches`` distinct ``match_id`` (deterministic
        ``ORDER BY match_id`` — lexicographically biased, a smoke aid only). ``None`` reads all.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        if max_matches is not None:
            ids = [
                r[0]
                for r in _query_param(
                    cur,
                    f"SELECT DISTINCT match_id FROM {_ACTION_VALUES_TABLE} ORDER BY match_id LIMIT %(n)s",  # noqa: S608
                    {"n": int(max_matches)},
                ).itertuples(index=False)
            ]
            if not ids:
                return pd.DataFrame(columns=_ACTION_VALUES_COLUMNS.replace(" ", "").split(","))
            placeholders = ", ".join(f"%(m{i})s" for i in range(len(ids)))
            params = {f"m{i}": v for i, v in enumerate(ids)}
            sql = f"SELECT {_ACTION_VALUES_COLUMNS} FROM {_ACTION_VALUES_TABLE} WHERE match_id IN ({placeholders})"  # noqa: S608
            return _query_param(cur, sql, params)
        return _query_param(cur, f"SELECT {_ACTION_VALUES_COLUMNS} FROM {_ACTION_VALUES_TABLE}")  # noqa: S608
    finally:
        conn.close()


def shape_action_values(df: pd.DataFrame) -> pd.DataFrame:
    """Map the gold action-values mart to the SPADL-id columns the xthreat NLL path expects.

    Pure, NaN-tolerant: ``action_type`` / ``action_result`` strings -> nullable-int ``type_id`` /
    ``result_id`` codes (unmapped -> <NA>; ``Int64`` is deliberate per ADR-019, avoiding a float id
    column — the caller drops the <NA> rows after its coverage guard so the ids reach the masks
    NA-free); ``match_id`` -> ``game_id`` (the holdout_split key). Coverage + NA-drop are the
    caller's job, not this function's.
    """
    import silly_kicks.spadl.config as spadlconfig  # function-local: keep module import cheap

    out = df.copy()
    out["type_id"] = out["action_type"].map(spadlconfig.actiontype_id).astype("Int64")
    out["result_id"] = out["action_result"].map(spadlconfig.result_id).astype("Int64")
    out["game_id"] = out["match_id"]
    return out


def fetch_idsse_events() -> pd.DataFrame:
    """Read all bronze IDSSE event rows (native sportec-converter input shape) for the owner-gated
    play_evaluation e2e. Read-only; the 7 public IDSSE matches are ~10.5k events. Table name comes
    from the allowlist-validated ``_table`` (idsse is in ``_ALLOWED_PROVIDERS``) -- no user input.
    """
    conn = _connect()
    try:
        cur = conn.cursor()
        return _query_param(cur, f"SELECT * FROM {_table('idsse', 'events')}")  # noqa: S608
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
    from silly_kicks.tracking.utils import filter_extratime_frames

    # IDSSE/sportec bronze is the most common operator path; others can be added as needed.
    if provider in ("idsse", "sportec"):
        from silly_kicks.spadl import sportec as sportec_spadl
        from silly_kicks.tracking import sportec as sportec_tracking

        home_team_id = str(raw_frames["team_id"].dropna().mode().iloc[0])  # placeholder; see NOTE
        home_start_left = True
        # Calibration only --- AC-1 production sources home_team_start_left_extratime via
        # MatchMeta (lakehouse Phase A) and NEVER filters ET. Sportec (tracking + events)
        # RAISES on ET-without-flag in silly-kicks 4.0.0 (ADR-010), so drop ET here.
        raw_frames = filter_extratime_frames(raw_frames, label=f"{provider} tracking")
        raw_events = filter_extratime_frames(raw_events, label=f"{provider} events")
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


# --- xT-GK v2 gate cohort loader (ADR-036 §Part 1b) ----------------------------------------------
# Base SPADL from bronze.spadl_actions (result_id/team_id/possession_id_heuristic), bridged to the
# gold surrogate match_key via dim_matches, LEFT JOIN pressure + frame-present from fct_action_context
# and the calibrated xG reward from fct_shot_xg. All three keyed on (match_key, action_id); the
# action_id join is exact (coord-alignment verified 0.0). provider is allowlist-validated; the value
# is ALSO passed as a bound parameter, never free-form interpolated.
_XTGK_ACTIONS_SQL = """
WITH s AS (SELECT * FROM soccer_analytics.bronze.spadl_actions WHERE data_source = %(ds)s),
     d AS (SELECT match_key, native_match_id FROM soccer_analytics.dev_gold.dim_matches WHERE provider = %(ds)s)
SELECT
  s.game_id, s.period_id, s.action_id, s.time_seconds, s.team_id, s.player_id,
  s.type_id, s.result_id, s.bodypart_id, s.start_x, s.start_y, s.end_x, s.end_y,
  s.possession_id_heuristic AS possession_id, s.data_source,
  c.pressure_on_actor__bekkers_pi, c.pressure_on_actor__andrienko_oval,
  CASE WHEN c.team_shape_n_outfield_players_defending IS NOT NULL THEN true ELSE false END AS frame_present,
  x.xg, x.ood_flag, x.xg_ci_low, x.xg_ci_high
FROM s
LEFT JOIN d ON s.match_id_native = d.native_match_id
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c ON c.match_key = d.match_key AND c.action_id = s.action_id
LEFT JOIN soccer_analytics.dev_gold.fct_shot_xg x ON x.match_key = d.match_key AND x.action_id = s.action_id
ORDER BY s.game_id, s.period_id, s.action_id
"""
_XTGK_SHOTXG_SQL = (
    "SELECT data_source, xg, ood_flag, xg_ci_low, xg_ci_high "
    "FROM soccer_analytics.dev_gold.fct_shot_xg WHERE data_source = %(ds)s"
)


def load_xtgk_cohort(data_source: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Owner-run loader for the xT-GK v2 gate (ADR-036 §Part 1b): returns (actions, shot_xg).

    ``actions`` = attack-LTR SPADL for the cohort with ``xg`` (fct_shot_xg), pressure (both
    ``pressure_on_actor__bekkers_pi`` [the pinned measure] and ``__andrienko_oval`` [comparison]),
    ``frame_present``, ``possession_id``, and per-shot ``ood_flag``/``xg_ci_low``/``xg_ci_high``.
    ``shot_xg`` = the per-shot fct_shot_xg slice for the OOD-rate report. Read-only; provider
    allowlist-validated + bound-parameterized.
    """
    if data_source not in _ALLOWED_PROVIDERS:
        raise ValueError(f"data_source {data_source!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    conn = _connect()
    try:
        cur = conn.cursor()
        actions = _query_param(cur, _XTGK_ACTIONS_SQL, {"ds": data_source})
        shot_xg = _query_param(cur, _XTGK_SHOTXG_SQL, {"ds": data_source})
    finally:
        conn.close()
    # dtypes: nullable gold columns arrive object/None -> coerce the numerics the gate reads.
    numeric = (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "pressure_on_actor__bekkers_pi",
        "pressure_on_actor__andrienko_oval",
        "xg",
        "time_seconds",
    )
    for col in numeric:
        actions[col] = pd.to_numeric(actions[col], errors="coerce")
    for col in ("type_id", "result_id"):
        actions[col] = pd.to_numeric(actions[col], errors="coerce").astype("int64")
    actions["frame_present"] = actions["frame_present"].astype(bool)
    return actions, shot_xg


# --- xT-GK v2 retention (rho) cohort loader (ADR-036 §Part 3, marts-native) -----------------------
# Tracking-frames deprecated: features come from the gold action marts. fct_action_values supplies
# the base SPADL (geometry/type/result/possession) + the materialized GK-distribution flag
# (gk_was_distributing); fct_action_context supplies pressure. Keyed on (match_key, action_id).
_RETENTION_SQL = """
WITH v AS (SELECT * FROM soccer_analytics.dev_gold.fct_action_values WHERE data_source = %(ds)s)
SELECT
  v.match_key AS game_id, v.period AS period_id, v.action_id, v.time_seconds,
  v.team_id, v.player_id, v.start_x, v.start_y, v.end_x, v.end_y,
  v.action_type, v.action_result, v.possession_id, v.gk_was_distributing, v.data_source,
  c.pressure_on_actor__bekkers_pi AS pressure
FROM v
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c
  ON c.match_key = v.match_key AND c.action_id = v.action_id
ORDER BY v.match_key, v.period, v.time_seconds, v.action_id
"""


def load_retention_cohort(data_source: str) -> pd.DataFrame:
    """Full attack-LTR action stream for the rho retention trainer (marts-native; NO tracking frames).

    Maps the gold string ``action_type``/``action_result`` to SPADL ``type_id``/``result_id``
    (unmapped -> -1, harmless: not a shot/move), carries ``gk_was_distributing`` (the GK-distribution
    domain) + ``pressure``. Sorted by (game_id, period_id, time_seconds, action_id).
    """
    import silly_kicks.spadl.config as spadlconfig

    if data_source not in _ALLOWED_PROVIDERS:
        raise ValueError(f"data_source {data_source!r} not in allowlist {sorted(_ALLOWED_PROVIDERS)}")
    conn = _connect()
    try:
        df = _query_param(conn.cursor(), _RETENTION_SQL, {"ds": data_source})
    finally:
        conn.close()
    df["type_id"] = df["action_type"].map(spadlconfig.actiontype_id).fillna(-1).astype("int64")
    df["result_id"] = df["action_result"].map(spadlconfig.result_id).fillna(-1).astype("int64")
    for col in ("start_x", "start_y", "end_x", "end_y", "pressure", "time_seconds"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["gk_was_distributing"] = df["gk_was_distributing"].fillna(False).astype(bool)
    # retains() requires a finite, non-decreasing clock within (game_id, period_id); drop NaN-time
    # rows (rare/anomalous, unlabelable) and re-sort stably in pandas.
    df = df[df["time_seconds"].notna()].copy()
    return df.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="stable").reset_index(drop=True)
