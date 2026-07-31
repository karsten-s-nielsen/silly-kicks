"""Databricks bronze loader for the TF-24 calibration harness.

Reads soccer_analytics.bronze.{provider}_{tracking,events} (id col: match_id), runs the CURRENT
silly-kicks converters (so calibration reflects current output), and yields the uniform
(provider, match_id, actions, frames, home_team_id) tuple. Operator-scale / fallback path + the
bronze.spadl_actions xT-corpus source (IDSSE is public on pining; this is not its only source).

Env: DATABRICKS_HTTP_PATH (always). Auth: DATABRICKS_TOKEN (PAT) if set, else OAuth U2M via a
databricks-sdk profile (DATABRICKS_CONFIG_PROFILE, default OAUTH; authenticate once with
`databricks auth login --profile OAUTH`). DATABRICKS_HOST is needed only on the PAT path -- the
OAuth profile carries its own host. (The workspace moved off PATs; the loader keeps PAT support
for CI and legacy setups.)
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pandas as pd

from silly_kicks.xtgk import apply_resolved_gk_geometry

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
    http_path = os.environ["DATABRICKS_HTTP_PATH"]
    # Auth precedence. `DATABRICKS_AUTH` selects the branch explicitly; UNSET preserves the historic
    # behaviour (a non-empty token wins), so CI and legacy setups are untouched -- this is an opt-in
    # override, not a behaviour change. An empty token string is NOT a usable PAT -> OAuth branch.
    #
    # The override exists because the historic rule takes the PAT branch on ANY non-empty token,
    # so an unusable one pre-empts the working OAuth fallback below it and the resulting error
    # names the WORKSPACE rather than the environment variable that chose the branch. Measured on
    # the maintainer's machine: a dead 36-char `dapi` PAT in `DATABRICKS_TOKEN`, no
    # `DATABRICKS_CONFIG_PROFILE`, every Databricks-backed driver failing for a reason none of them
    # reported.
    auth = os.environ.get("DATABRICKS_AUTH", "").strip().lower()
    if auth not in ("", "pat", "oauth"):
        # Refused rather than ignored: falling through on a typo would reinstate exactly the silent
        # precedence this variable exists to end, one line further down.
        raise RuntimeError(f"DATABRICKS_AUTH must be 'pat', 'oauth' or unset; got {auth!r}")
    token = os.environ.get("DATABRICKS_TOKEN")
    if auth == "pat" and not token:
        raise RuntimeError("DATABRICKS_AUTH=pat but DATABRICKS_TOKEN is unset or empty")
    if token and auth != "oauth":
        try:
            return dbsql.connect(
                server_hostname=os.environ["DATABRICKS_HOST"].replace("https://", ""),
                http_path=http_path,
                access_token=token,
            )
        except Exception as exc:
            # Name the precedence AND both of its causes. This branch fails for a stale PAT and for
            # an expired short-lived bearer alike -- the lakehouse deliberately puts a ~299 s minted
            # OAuth bearer in this same variable -- so a message naming only one mis-diagnoses half
            # the cases.
            raise RuntimeError(
                "Databricks PAT auth failed. A non-empty DATABRICKS_TOKEN took priority over the "
                "OAuth profile; the token may be a STALE PAT or an EXPIRED short-lived bearer. "
                "Unset DATABRICKS_TOKEN, re-mint it, or set DATABRICKS_AUTH=oauth."
            ) from exc
    try:
        from databricks.sdk.core import Config  # type: ignore[import-not-found]
    except ImportError as exc:  # actionable hint
        raise RuntimeError(
            "No DATABRICKS_TOKEN set and databricks-sdk is required for OAuth auth: "
            "pip install databricks-sdk, then `databricks auth login --profile OAUTH`"
        ) from exc
    cfg = Config(profile=os.environ.get("DATABRICKS_CONFIG_PROFILE", "OAUTH"))
    return dbsql.connect(
        server_hostname=cfg.host.replace("https://", ""),
        http_path=http_path,
        credentials_provider=lambda: cfg.authenticate,
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
    *,
    providers: list[str],
    match_ids: dict[str, list[str]] | None = None,
    tracking_limit: int | None = None,
    max_per_provider: int | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) from bronze.

    ``tracking_limit`` and ``max_per_provider`` mirror `_loader_pining.load_matches`, and that
    parity is a bug fix rather than symmetry for its own sake: `calibrate_tracking_defaults`
    chooses between the two loaders at runtime and calls whichever it picked with ONE kwarg set, so
    every ``--source databricks`` (and ``--source auto``) invocation died on
    ``TypeError: load_matches() got an unexpected keyword argument 'tracking_limit'`` before
    reading a row -- the driver's databricks path could not run at all. Dropping the kwargs at the
    call site instead would have made the two memory bounds silently inert on the loader that most
    needs them.

    ``tracking_limit`` caps frames per match POST-QUERY (the pining path's IDSSE behaviour), so it
    bounds memory downstream, not the query. ``max_per_provider`` caps the number of matches after
    any ``match_ids`` selection.
    """
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
            if max_per_provider is not None:
                ids = list(ids)[:max_per_provider]
            for mid in ids:
                # Table from allowlist; match_id PARAMETERIZED (M5) — never f-string-interpolated.
                raw_frames = _query_param(cur, f"SELECT * FROM {t_tracking} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                raw_events = _query_param(cur, f"SELECT * FROM {t_events} WHERE match_id = %(mid)s", {"mid": mid})  # noqa: S608
                actions, frames, home = _convert(provider, raw_events, raw_frames)
                if tracking_limit is not None:
                    frames = frames.head(tracking_limit)
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


def _native_team_id(raw_events: pd.DataFrame, column: str, provider: str) -> str:
    """Read a match-level native team id (DFL-CLU) off the bronze EVENTS table.

    These are per-match constants stamped on every event row. Fails loud with an actionable
    message rather than raising a bare ``KeyError`` / ``IndexError`` deep inside the converter,
    because an absent/empty value silently mis-orients the whole match.
    """
    if column not in raw_events.columns:
        raise ValueError(
            f"{provider} bronze events missing {column!r}; cannot resolve the native team id "
            "required to orient tracking frames and derive direction of play."
        )
    values = raw_events[column].dropna()
    if values.empty:
        raise ValueError(f"{provider} bronze events column {column!r} is entirely null.")
    return str(values.iloc[0])


def _convert(provider: str, raw_events: pd.DataFrame, raw_frames: pd.DataFrame):
    """Convert bronze rows to (actions, frames, home_team_id) via silly-kicks converters.

    Bronze tables are the lakehouse's **RAW** parsed provider output --- NOT converter input
    shape. For Sportec/IDSSE the ``providers.sportec`` parse-port shapers (ADR-031 T3) bridge
    the gap: ``shape_tracking_to_native`` (bronze ``match_id/period/frame/x/y/s/ball_status``
    + denormalized ``ball_*`` -> the 14 ``tracking.sportec.EXPECTED_INPUT_COLUMNS``, synthesizing
    the ball rows bronze does not carry) and ``shape_events_to_native`` (resolves set-piece /
    foul team + player from the DFL qualifier columns). Both are mandatory --- the converters
    raise on the raw bronze shape. This mirrors ``scripts/_loader_pining.py::_build_idsse``.

    Identifier domains are ASYMMETRIC across the two bronze tables, and the converters must be
    fed the domain each one actually carries:

    * ``bronze.idsse_tracking.team_id``  -> DFL-CLU id (``"DFL-CLU-000008"``)
    * ``bronze.idsse_events.team``       -> ``"home"`` / ``"away"`` / ``"unknown"``
      (there is no ``team_id`` on events; the CLU ids live in ``home/away_team_id_native``)

    So ``convert_to_frames`` gets the native CLU id while ``convert_to_actions`` gets the literal
    ``"home"``. Passing the CLU id to the events converter would match ZERO rows (it compares
    against ``team``, copied verbatim into ``actions.team_id``) and mirror every home action as
    away.

    Returned ``home_team_id`` (the 5th ``load_matches`` tuple element) is the **TRACKING /
    frames-domain DFL-CLU id**. Its consumers (``scripts/calibrate_tracking_defaults.py`` ->
    ``silly_kicks/calibration/_features.py`` -> ``add_defensive_line`` / ``add_team_shape``)
    resolve attacking direction from the FRAMES, whose ``team_id`` is CLU. ``actions.team_id`` is
    remapped ``"home"``/``"away"`` -> CLU below so the ADR-028 per-action LTR re-projection (which
    joins actions to frames on ``team_id``) aligns; without that remap the join matches nothing
    and away-team tracking geometry stays 180 degrees wrong.

    Direction of play is DERIVED from the DFL ``<KickOff>`` rows (authoritative source XML), never
    hard-coded --- including extra time, so ET frames are kept rather than dropped (ADR-010's
    ET-without-flag raise is satisfied by supplying the flag). ``..._extratime`` returns ``None``
    for a match with no ET, which the converters accept as "no ET present".
    """
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    # IDSSE/sportec bronze is the most common operator path; others can be added as needed.
    if provider in ("idsse", "sportec"):
        from silly_kicks.providers.sportec import (
            derive_idsse_home_team_start_left,
            derive_idsse_home_team_start_left_extratime,
            shape_events_to_native,
            shape_tracking_to_native,
        )
        from silly_kicks.spadl import sportec as sportec_spadl
        from silly_kicks.tracking import sportec as sportec_tracking

        home_team_id_native = _native_team_id(raw_events, "home_team_id_native", provider)
        away_team_id_native = _native_team_id(raw_events, "away_team_id_native", provider)

        # Events: shape -> derive direction-of-play from the DFL KickOff -> native SPADL.
        native_evt = shape_events_to_native(raw_events)
        home_start_left = derive_idsse_home_team_start_left(native_evt, home_team_id_native)
        home_start_left_et = derive_idsse_home_team_start_left_extratime(native_evt, home_team_id_native)
        actions, _r2 = sportec_spadl.convert_to_actions(
            native_evt,
            home_team_id="home",  # native_evt.team carries the 'home'/'away' label, NOT the CLU id
            home_team_start_left=home_start_left,
            home_team_start_left_extratime=home_start_left_et,
        )
        # Re-namespace the echoed 'home'/'away' action team_id onto the CLU ids the frames use, so
        # the ADR-028 action<->frame join resolves ('unknown' rows keep their label, as on pining).
        actions["team_id"] = (
            actions["team_id"]
            .map({"home": home_team_id_native, "away": away_team_id_native})
            .fillna(actions["team_id"])
        )

        # Tracking: shape (adds the synthetic ball rows) -> native frames, CLU-keyed.
        frames, _r = sportec_tracking.convert_to_frames(
            shape_tracking_to_native(raw_frames),
            home_team_id=home_team_id_native,  # native_trk.team_id carries the DFL CLU id
            home_team_start_left=home_start_left,
            home_team_start_left_extratime=home_start_left_et,
            output_convention="absolute_frame",
        )
        return actions, derive_velocities(smooth_frames(frames)), home_team_id_native

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
  c.is_gk_distribution, c.xt_gk, c.player_key,
  c.xt_gk_origin_x, c.xt_gk_origin_y, c.xt_gk_dest_x, c.xt_gk_dest_y,
  c.xt_gk_origin_source, c.xt_gk_dest_source,
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
    ``frame_present``, ``possession_id``, per-shot ``ood_flag``/``xg_ci_low``/``xg_ci_high``, and (for the
    SP5 construct-validity harness, PR-S112) the GK-distribution domain marker ``is_gk_distribution`` +
    the stored v1 composite ``xt_gk`` (NaN where v1 didn't score; the deep-zone gate ignores both).
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
        "xt_gk_origin_x",
        "xt_gk_origin_y",
        "xt_gk_dest_x",
        "xt_gk_dest_y",
        "pressure_on_actor__bekkers_pi",
        "pressure_on_actor__andrienko_oval",
        "xg",
        "time_seconds",
        "xt_gk",  # stored v1 composite; NaN preserved where v1 didn't score (a baseline, not a gate)
    )
    for col in numeric:
        actions[col] = pd.to_numeric(actions[col], errors="coerce")
    for col in ("type_id", "result_id"):
        actions[col] = pd.to_numeric(actions[col], errors="coerce").astype("int64")
    actions["frame_present"] = actions["frame_present"].astype(bool)
    actions["is_gk_distribution"] = actions["is_gk_distribution"].astype("boolean").fillna(False).astype(bool)
    # ADR-036 amendment (4.46.0): the GK-distribution domain's canonical coords are NOT trustworthy
    # (GS ~60% NaN goal-kick origins; SkillCorner's native origin is the broadcast BALL, not the
    # keeper). Inject the resolved keeper geometry v1 already computes and the lakehouse persists.
    # Doing it HERE means all consumers (validate_xtgk_v2 / keeper_discrimination / kappa_sweep /
    # validate_xtgk_possession_value) inherit it, and rho features are necessarily built from the
    # resolved frame -- the actions-vs-features coherence hazard is closed by construction.
    actions = apply_resolved_gk_geometry(actions)
    return actions, shot_xg


# --- xT-GK v2 retention (rho) cohort loader (ADR-036 §Part 3, marts-native) -----------------------
# Tracking-frames deprecated: features come from the gold action marts. fct_action_values supplies
# the base SPADL (geometry/type/result/possession); fct_action_context supplies pressure AND the
# GK-distribution domain flag (is_gk_distribution = tracking.gk_distribution_mask, resolve_gk="robust").
# Keyed on (match_key, action_id). is_gk_distribution is a HARD dependency -- permanently materialized by
# lakehouse F1 (silly-kicks >= 4.44.0), selected unconditionally; a missing column surfaces as a
# column-named Databricks error, not a cryptic fault. pressure = bekkers_pi, pinned in PR-S109.
_RETENTION_SQL = """
WITH v AS (SELECT * FROM soccer_analytics.dev_gold.fct_action_values WHERE data_source = %(ds)s)
SELECT
  v.match_key AS game_id, v.period AS period_id, v.action_id, v.time_seconds,
  v.team_id, v.player_id, v.start_x, v.start_y, v.end_x, v.end_y,
  v.action_type, v.action_result, v.possession_id, v.data_source,
  c.pressure_on_actor__bekkers_pi AS pressure,
  c.is_gk_distribution,
  c.xt_gk_origin_x, c.xt_gk_origin_y, c.xt_gk_dest_x, c.xt_gk_dest_y,
  c.xt_gk_origin_source, c.xt_gk_dest_source
FROM v
LEFT JOIN soccer_analytics.dev_gold.fct_action_context c
  ON c.match_key = v.match_key AND c.action_id = v.action_id
ORDER BY v.match_key, v.period, v.time_seconds, v.action_id
"""


def load_retention_cohort(data_source: str) -> pd.DataFrame:
    """Full attack-LTR action stream for the rho retention trainer (marts-native; NO tracking frames).

    Requires ``fct_action_context.is_gk_distribution`` (lakehouse F1; silly-kicks >= 4.44.0) -- a HARD
    dependency, unconditionally selected. Maps the gold string ``action_type``/``action_result`` to SPADL
    ``type_id``/``result_id`` (unmapped -> -1, harmless: not a shot/move), carries ``pressure`` (bekkers_pi)
    + the ``is_gk_distribution`` GK-distribution domain flag (NULLs coalesced to False). Sorted by
    (game_id, period_id, time_seconds, action_id).
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
    for col in (
        "start_x",
        "start_y",
        "end_x",
        "end_y",
        "xt_gk_origin_x",
        "xt_gk_origin_y",
        "xt_gk_dest_x",
        "xt_gk_dest_y",
        "pressure",
        "time_seconds",
    ):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # object/None (Databricks nullable bool) -> nullable boolean -> False-coalesced -> numpy bool (no
    # FutureWarning downcast). NULLs (LEFT-JOIN misses / F1 gaps) -> False (excluded from the rho domain).
    df["is_gk_distribution"] = df["is_gk_distribution"].astype("boolean").fillna(False).astype(bool)
    # ADR-036 amendment (4.46.0): rho MUST train on the same resolved geometry the metric scores on.
    # Pre-fix, the SkillCorner variant was trained on 1181 goal-kicks whose origin was the broadcast
    # BALL, not the keeper -- contaminated weights, not merely missing data.
    df = apply_resolved_gk_geometry(df)
    # retains() requires a finite, non-decreasing clock within (game_id, period_id); drop NaN-time
    # rows (rare/anomalous, unlabelable) and re-sort stably in pandas.
    df = df[df["time_seconds"].notna()].copy()
    return df.sort_values(["game_id", "period_id", "time_seconds", "action_id"], kind="stable").reset_index(drop=True)


def resolve_retention_model(provider: str, weights_dir: str | None = None):
    """Load the rho model for ``provider``, or from an explicit artifact dir.

    ``weights_dir`` exists for the ADR-036 two-leg SP5 re-run: leg 1 scores the CORRECTED cohort
    under the PRE-FIX rho, leg 2 under the retrained one, so the delta is attributable between
    "the origins moved" and "rho moved".
    """
    from pathlib import Path

    from silly_kicks.xtgk._retention import GkRetentionModel, variant_key_for_provider

    if weights_dir:
        return GkRetentionModel.load(Path(weights_dir))
    return GkRetentionModel.from_variant(variant_key_for_provider(provider))
