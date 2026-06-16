"""pining-for-the-data loader for the TF-24 calibration harness.

Provider-agnostic fetch from the gated mock provider API (two-step Bearer -> 302 -> presigned
S3). Serves SkillCorner (public), IDSSE (public), Gradient Sports (owner). The artifact formats +
key names differ per provider, so conversion dispatches on provider.

No local paths, no committed data — token from PINING_FOR_THE_DATA_TOKEN (owner) or the public
default; base URL from PINING_API_URL.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from pathlib import Path

import pandas as pd


def _apply_et_direction(frames: pd.DataFrame, et_value, *, label: str):
    """Resolve the extra-time start-direction for a per-period-absolute converter.

    Per-period-absolute converters (Gradient Sports, Sportec, Metrica --- tracking
    and events) RAISE on ET (period 3/4) without ``home_team_start_left_extratime``
    (silly-kicks 4.0.0, ADR-010). This loader stays correct AND crash-free for
    calibration sampling:

    * ``et_value`` present (from ``homeTeamStartLeftExtraTime``) -> pass it through
      (returns ``bool`` + ET frames untouched).
    * ``et_value`` None but ET periods (3/4) present -> drop the ET frames with a
      warning, via the public :func:`silly_kicks.tracking.filter_extratime_frames`.
      Calibration samples regular time; never guess the ET orientation, never crash.
    * No ET periods -> no-op (param stays ``None``).

    **Calibration only** --- AC-1 production sources ``home_team_start_left_extratime``
    via ``MatchMeta`` (lakehouse Phase A) and NEVER filters ET.

    Returns ``(frames, et_param)``.
    """
    from silly_kicks.tracking.utils import filter_extratime_frames

    et_param = bool(et_value) if et_value is not None else None
    if et_param is None:
        frames = filter_extratime_frames(frames, label=label)
    return frames, et_param


_DEFAULT_BASE_URL = "https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1"
_PUBLIC_TOKEN = "test-token-pining-for-the-data"  # noqa: S105  # documented PUBLIC token, not a secret


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # surface the 302 as an HTTPError so we can read Location ourselves


def _base_url() -> str:
    return os.environ.get("PINING_API_URL", _DEFAULT_BASE_URL).rstrip("/")


def _resolve_token(token: str | None) -> str:
    # Owner token enables GS; otherwise the public token (SkillCorner + IDSSE).
    return token or os.environ.get("PINING_FOR_THE_DATA_TOKEN") or _PUBLIC_TOKEN


def _list_matches(provider: str, token: str, base_url: str) -> list[dict]:
    """GET /{provider}/matches -> the matches list (id + artifacts map)."""
    req = urllib.request.Request(  # noqa: S310
        f"{base_url}/{provider}/matches", headers={"Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return json.loads(resp.read()).get("matches", [])


def _download_to_temp(
    provider: str,
    match_id: str,
    artifact_key: str,
    token: str,
    base_url: str,
    dest_dir: Path,
    *,
    use_cache: bool = False,
) -> Path:
    """Two-step: bearer GET -> 302 Location -> presigned GET (no bearer) -> stream to a file.

    Streams so the ~419 MB IDSSE tracking.xml never sits fully in memory. When ``use_cache`` and a
    non-empty ``dest`` already exists, returns it WITHOUT re-fetching (the cache hit — re-runs over
    the same corpus skip every download). Writes go to a ``.partial`` sibling that is atomically
    renamed into place only on a complete stream, so a crashed/retried download never leaves a
    corrupt cache entry (a partial is simply re-fetched on the next attempt).
    """
    dest = dest_dir / f"{provider}_{match_id}_{artifact_key}"
    if use_cache and dest.exists() and dest.stat().st_size > 0:
        return dest  # cache hit — no network
    opener = urllib.request.build_opener(_NoRedirect)
    url = f"{base_url}/{provider}/matches/{match_id}/{artifact_key}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})  # noqa: S310
    location = None
    try:
        opener.open(req, timeout=60)
    except urllib.error.HTTPError as exc:
        if exc.code in (301, 302, 303, 307):
            location = exc.headers.get("Location")
        else:
            raise
    if not location:
        raise RuntimeError(f"pining {provider}/{match_id}/{artifact_key}: expected a 302 redirect")
    partial = dest.with_name(dest.name + ".partial")
    with urllib.request.urlopen(location, timeout=600) as resp, open(partial, "wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
    partial.replace(dest)  # atomic — no partial cache entry on a mid-stream crash
    return dest


def _artifact_key(artifacts: dict, *, suffix: str) -> str:
    """Resolve an artifact KEY by filename suffix (SkillCorner keys are match-id-prefixed)."""
    for key, filename in artifacts.items():
        if str(filename).endswith(suffix):
            return key
    raise KeyError(f"no artifact ending with {suffix!r} in {sorted(artifacts)}")


def load_matches(
    *,
    providers: list[str],
    match_ids: dict[str, list[str]] | None = None,
    token: str | None = None,
    tracking_limit: int | None = None,
    max_per_provider: int | None = None,
    cache_dir: str | Path | None = None,
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) for each requested match.

    ``tracking_limit`` caps frames loaded per match (passed to the kloppy parser for SkillCorner;
    applied post-parse to the first N frames for the IDSSE DFL parse-port path) — essential for the
    ~419 MB IDSSE tracking file in dev/e2e loops. ``max_per_provider`` caps the NUMBER of
    matches loaded per provider (after any ``match_ids`` selection) — bounds total memory for the
    TF-24 sweep on a local machine (loading all matches at full depth can OOM; see calibrate CLI
    ``--max-matches-per-provider``). ``cache_dir`` (when set) persists every downloaded artifact
    under ``cache_dir/{provider}/{match_id}/`` and reuses it on subsequent runs over the same corpus
    — the network is paid once, not per re-run (the large IDSSE/GS tracking files dominate the load).
    """
    tok, base_url = _resolve_token(token), _base_url()
    for provider in providers:
        manifest = {m["id"]: m for m in _list_matches(provider, tok, base_url)}
        wanted = (match_ids.get(provider) if match_ids else None) or list(manifest)
        if max_per_provider is not None:
            wanted = wanted[:max_per_provider]
        for match_id in wanted:
            artifacts = manifest[match_id]["artifacts"]
            actions, frames, home = _build_match_with_retry(
                provider, match_id, artifacts, tok, base_url, tracking_limit, cache_dir=cache_dir
            )
            yield provider, match_id, actions, frames, home


def _build_match_with_retry(
    provider,
    match_id,
    artifacts,
    tok,
    base_url,
    tracking_limit,
    *,
    cache_dir=None,
    attempts: int = 3,
    backoff: float = 3.0,
):
    """Download + build one match, retrying transient network/IO failures with a fresh temp dir.

    The pining fetch (Bearer -> 302 -> presigned S3) and kloppy's subsequent file reads can blip
    transiently — an empty/partial download surfaces as ``kloppy ... InputNotFoundError``, an S3 or
    DNS hiccup as ``urllib``/``OSError``. The TF-24 sweep re-downloads ~140 matches across its four
    fold-loads (2 phases x Stage 1 + Stage 2); a single un-retried blip would crash a whole stage,
    losing hours of Stage-2 enrichment. Retry with a fresh temp dir + linear backoff, then fail loud
    only if a match is genuinely unfetchable after ``attempts`` tries.
    """
    import tempfile

    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            if cache_dir is not None:
                # Persistent per-match cache dir (NOT a temp dir) — artifacts survive the call so
                # subsequent runs over the same corpus skip every download (cache hit in
                # _download_to_temp). The kloppy/build parse still re-runs; only the network is saved.
                dl_dir = Path(cache_dir) / provider / str(match_id)
                dl_dir.mkdir(parents=True, exist_ok=True)
                paths = _download_artifacts(provider, match_id, artifacts, tok, base_url, dl_dir, use_cache=True)
                return _build_match(provider, match_id, paths, tracking_limit)
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                paths = _download_artifacts(provider, match_id, artifacts, tok, base_url, tmp_dir)
                return _build_match(provider, match_id, paths, tracking_limit)
        except Exception as exc:  # transient network/IO (any source) — retried, then re-raised loud
            last_exc = exc
            if attempt < attempts:
                warnings.warn(
                    f"{provider} match {match_id}: load attempt {attempt}/{attempts} failed "
                    f"({type(exc).__name__}: {exc}); retrying in {backoff * attempt:.0f}s",
                    UserWarning,
                    stacklevel=2,
                )
                time.sleep(backoff * attempt)
    raise RuntimeError(
        f"{provider} match {match_id}: failed to load after {attempts} attempts (last error above)"
    ) from last_exc


def _download_artifacts(
    provider, match_id, artifacts, token, base_url, tmp_dir, *, use_cache: bool = False
) -> dict[str, Path]:
    """Download the artifacts each provider needs, keyed by a NORMALISED role name."""
    if provider == "idsse":
        roles = {"events": "events", "metadata": "metadata", "tracking": "tracking"}
    elif provider == "gradientsports":
        roles = {"events": "events", "metadata": "metadata", "roster": "roster", "tracking": "tracking"}
    elif provider == "skillcorner":
        roles = {
            "events": _artifact_key(artifacts, suffix="_dynamic_events.csv"),
            "metadata": _artifact_key(artifacts, suffix="_match.json"),
            "tracking": _artifact_key(artifacts, suffix="_tracking_extrapolated.jsonl"),
        }
    else:
        raise ValueError(f"unknown pining provider {provider!r}")
    out: dict[str, Path] = {}
    for role, key in roles.items():
        artifact_key = key if key in artifacts else role
        out[role] = _download_to_temp(provider, match_id, artifact_key, token, base_url, tmp_dir, use_cache=use_cache)
    return out


def _build_match(provider, match_id, paths, tracking_limit):
    """Provider dispatch: parse the downloaded artifacts into (actions, frames, home_team_id)."""
    if provider == "idsse":
        return _build_idsse(paths, match_id, tracking_limit)
    if provider == "skillcorner":
        return _build_skillcorner(paths, match_id, tracking_limit)
    if provider == "gradientsports":
        return _build_gradientsports(paths, tracking_limit)
    raise ValueError(f"unknown pining provider {provider!r}")


def _preprocess(frames: pd.DataFrame) -> pd.DataFrame:
    """Derive velocities (vx/vy) so bekkers_pi + velocity-aware carrier inference work."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    return derive_velocities(smooth_frames(frames))


def build_skillcorner_frames(paths, tracking_limit):
    """Preprocessed silly-kicks frames from SkillCorner artifacts (tracking ds path).

    The single SkillCorner frame-construction path: kloppy load -> convert_to_frames
    -> _preprocess (smooth + velocities), yielding SPADL-bounds (0-105/0-68) frames.
    Reused by both _build_skillcorner (calibration) and the TF-27 GK-roster e2e.
    """
    from kloppy import skillcorner  # type: ignore[import-not-found]

    from silly_kicks.tracking import kloppy as tracking_kloppy

    ds = skillcorner.load(
        meta_data=str(paths["metadata"]),
        raw_data=str(paths["tracking"]),
        limit=tracking_limit,
        include_empty_frames=False,
    )
    frames, _report = tracking_kloppy.convert_to_frames(ds)
    return _preprocess(frames)


def _build_skillcorner(paths, match_id, tracking_limit):
    """SkillCorner: kloppy tracking + silly-kicks SkillCorner events converter."""
    frames = build_skillcorner_frames(paths, tracking_limit)
    # Events: SkillCorner dynamic-events CSV + match.json -> silly-kicks SkillCorner SPADL converter.
    from silly_kicks.spadl import skillcorner as sk_spadl

    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    home_team_id = str(meta["home_team"]["id"])  # authoritative; matches kloppy tracking team ids
    raw_events = pd.read_csv(paths["events"], low_memory=False)
    actions, _evt_report = sk_spadl.convert_to_actions(raw_events, meta)
    return actions, frames, home_team_id


def _build_idsse(paths, match_id, tracking_limit):
    """IDSSE (DFL/Sportec XML) via the silly-kicks DFL parse+shape port (ADR-031 T3).

    Single-sources the DFL parser: ``providers.sportec.parse_dfl_*`` (bytes -> RAW bronze) ->
    ``shape_*_to_native`` -> the NATIVE silly-kicks ``spadl.sportec`` / ``tracking.sportec``
    converters. This replaces the former kloppy event + loader-local kloppy tracking path
    (``_kloppy_tracking_to_frames``), which produced y-INVERTED frames (ADR-031 / the
    kloppy-tracking-y bug). ``home_team_start_left`` is derived from the DFL ``<KickOff>``
    events (authoritative). Tracking frames are emitted ``absolute_frame`` (matching the prior
    harness convention) then preprocessed (smooth + velocities) consumer-side.
    """
    from silly_kicks.providers.sportec import (
        derive_idsse_home_team_start_left,
        derive_idsse_home_team_start_left_extratime,
        parse_dfl_events,
        parse_dfl_match_info,
        parse_dfl_tracking,
        shape_events_to_native,
        shape_tracking_to_native,
    )
    from silly_kicks.spadl import sportec as sportec_spadl
    from silly_kicks.tracking import sportec as sportec_tracking

    bare_id = str(match_id).removeprefix("DFL-MAT-")  # parser expects the bare DFL MatchId
    mi = parse_dfl_match_info(str(paths["metadata"]))

    # Events: parse -> shape -> derive direction-of-play from the DFL KickOff -> native SPADL.
    native_evt = shape_events_to_native(parse_dfl_events(str(paths["events"]), match_info=mi, match_id=bare_id))
    hsl = derive_idsse_home_team_start_left(native_evt, mi.home_team_id)
    hsl_et = derive_idsse_home_team_start_left_extratime(native_evt, mi.home_team_id)
    actions, _evt_report = sportec_spadl.convert_to_actions(
        native_evt,
        home_team_id="home",  # native_evt.team carries the 'home'/'away' label, not the CLU id
        home_team_start_left=hsl,
        home_team_start_left_extratime=hsl_et,
    )
    # The native converter echoes the 'home'/'away' label as the SPADL team_id, but the tracking
    # frames key team by the DFL CLU id. The ADR-028 per-action LTR re-projection joins actions to
    # frames on team_id, so they MUST share a namespace -- the retired kloppy path used CLU on both
    # sides. Remap the action team_id back to the CLU id so away-team tracking geometry re-projects
    # correctly ('unknown'-team rows keep their label and produce NaN geometry, as before).
    actions["team_id"] = (
        actions["team_id"].map({"home": mi.home_team_id, "away": mi.away_team_id}).fillna(actions["team_id"])
    )

    # Tracking: parse -> shape -> native frames. The port parses the FULL positions XML; honour
    # the dev-loop ``tracking_limit`` by capping to the first N distinct frames AFTER parse (the
    # whole-file parse is the cost of single-sourcing the DFL parser; the kloppy path capped at
    # read time). game_id flows from match_id through both native converters, so the prior
    # game_id-None stamp is no longer needed.
    native_trk = shape_tracking_to_native(parse_dfl_tracking(str(paths["tracking"]), match_info=mi, match_id=bare_id))
    if tracking_limit:
        keep = sorted(native_trk["frame_id"].unique())[:tracking_limit]
        native_trk = native_trk[native_trk["frame_id"].isin(keep)].reset_index(drop=True)
    frames, _trk_report = sportec_tracking.convert_to_frames(
        native_trk,
        home_team_id=mi.home_team_id,  # native_trk.team_id carries the DFL CLU id
        home_team_start_left=hsl,
        home_team_start_left_extratime=hsl_et,
        output_convention="absolute_frame",
    )
    return actions, _preprocess(frames), mi.home_team_id


_GS_EVENT_FIELD_MAP = {
    # possessionEvents.<key> -> flat column
    "possession_event_type": "possessionEventType",
    "body_type": "bodyType",
    "ball_height_type": "ballHeightType",
    "pass_outcome_type": "passOutcomeType",
    "pass_type": "passType",
    "incompletion_reason_type": "incompletionReasonType",
    "cross_outcome_type": "crossOutcomeType",
    "cross_type": "crossType",
    "cross_zone_type": "crossZoneType",
    "shot_outcome_type": "shotOutcomeType",
    "nonEvent": "nonEvent",
    "shot_type": "shotType",
    "shot_nature_type": "shotNatureType",
    "shot_initial_height_type": "shotInitialHeightType",
    "save_height_type": "saveHeightType",
    "save_rebound_type": "saveReboundType",
    "carry_type": "carryType",
    "ball_carry_outcome": "ballCarryOutcome",
    "carry_intent": "carryIntent",
    "carry_defender_player_id": "carryDefenderPlayerId",
    "challenge_type": "challengeType",
    "challenge_outcome_type": "challengeOutcomeType",
    "challenger_player_id": "challengerPlayerId",
    "challenge_winner_player_id": "challengeWinnerPlayerId",
    "tackle_attempt_type": "tackleAttemptType",
    "clearance_outcome_type": "clearanceOutcomeType",
    "rebound_outcome_type": "reboundOutcomeType",
    "keeper_touch_type": "keeperTouchType",
    "touch_outcome_type": "touchOutcomeType",
    "touch_type": "touchType",
}


def _gs_flatten_events(events_json: list[dict], roster: pd.DataFrame) -> pd.DataFrame:
    """Flatten raw GS gameEvents/possessionEvents JSON -> spadl.gradientsports EXPECTED_INPUT_COLUMNS.

    Ports tests/spadl/test_gradientsports.py::_load_synthetic_events, with the real roster used to
    fill challenger/winner team ids (player_id -> team_id).
    """
    pid_to_team = dict(zip(roster["player_id"], roster["team_id"], strict=False))
    rows = []
    for ev in events_json:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        f0 = ev.get("fouls") or {}
        ball = (ev.get("ball") or [{}])[0] if ev.get("ball") else {}
        row = {
            "game_id": ev["gameId"],
            "event_id": ev["gameEventId"],
            "possession_event_id": ev.get("possessionEventId"),
            "period_id": ge.get("period"),
            "time_seconds": ge.get("startGameClock"),
            "team_id": ge.get("teamId"),
            "player_id": ge.get("playerId"),
            "game_event_type": ge.get("gameEventType"),
            "set_piece_type": ge.get("setpieceType"),
            "ball_x": ball.get("x"),
            "ball_y": ball.get("y"),
            "foul_type": f0.get("foulType"),
            "on_field_offense_type": f0.get("onFieldOffenseType"),
            "final_offense_type": f0.get("finalOffenseType"),
            "on_field_foul_outcome_type": f0.get("onFieldFoulOutcomeType"),
            "final_foul_outcome_type": f0.get("finalFoulOutcomeType"),
            "challenger_team_id": None,
            "challenge_winner_team_id": None,
        }
        for col, key in _GS_EVENT_FIELD_MAP.items():
            row[col] = pe.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)

    def _team_for(pid):
        if pid is None or pd.isna(pid):
            return pd.NA
        return pid_to_team.get(int(pid), pd.NA)

    df["challenger_team_id"] = df["challenger_player_id"].map(_team_for)
    df["challenge_winner_team_id"] = df["challenge_winner_player_id"].map(_team_for)
    for col in (
        "possession_event_id",
        "player_id",
        "team_id",
        "carry_defender_player_id",
        "challenger_player_id",
        "challenger_team_id",
        "challenge_winner_player_id",
        "challenge_winner_team_id",
    ):
        df[col] = df[col].astype("Int64")
    df["game_id"] = df["game_id"].astype("int64")
    df["event_id"] = df["event_id"].astype("int64")
    df["period_id"] = df["period_id"].astype("int64")
    df["time_seconds"] = df["time_seconds"].astype("float64")
    df["ball_x"] = df["ball_x"].astype("float64")
    df["ball_y"] = df["ball_y"].astype("float64")
    return df


def _dedupe_gs_frame_records(frames_json: list[dict]) -> list[dict]:
    """Drop duplicate Gradient Sports frame records, keep-first per ``(period, frameNum)``.

    Some GS tracking exports ship the SAME ``(period, frameNum)`` record multiple times — observed
    up to 16 content-divergent copies of a single frame (overlapping data chunks). Left in, each
    duplicate fans out one row per entity at that frame key, so an action linked to such a frame
    sees N x the players + N ball rows. That crashes ``bekkers_pi`` (a 3-D ``ball_pos`` broadcast
    error) and silently inflates the inputs to pitch-control / DAS / team-shape. Keeping the first
    occurrence restores the ADR-004 contract of one row per ``(period, frame, player)``.
    """
    seen: set[tuple] = set()
    out: list[dict] = []
    for fr in frames_json:
        key = (fr["period"], fr["frameNum"])
        if key in seen:
            continue
        seen.add(key)
        out.append(fr)
    return out


def _build_gradientsports(paths, tracking_limit=None):
    """Gradient Sports: flatten JSONL tracking + roster -> add_gradientsports_player_ids -> frames;
    flatten gameEvents JSON -> SPADL via spadl.gradientsports. Ports the PR-A e2e + GS SPADL test.
    """
    import bz2

    from silly_kicks.spadl import gradientsports as gs_spadl
    from silly_kicks.tracking.gradientsports import add_gradientsports_player_ids, convert_to_frames

    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    meta = meta[0] if isinstance(meta, list) else meta  # GS metadata is a 1-element list (PR-A)
    home_team_id = int(meta["homeTeam"]["id"])
    away_team_id = int(meta["awayTeam"]["id"])
    home_start_left = bool(meta.get("homeTeamStartLeft", True))
    home_start_left_et = meta.get("homeTeamStartLeftExtraTime")

    with open(paths["roster"], encoding="utf-8") as fh:
        roster_raw = json.load(fh)
    roster = pd.DataFrame(
        {
            "team_id": [int(r["team"]["id"]) for r in roster_raw],
            "shirt_number": [str(r["shirtNumber"]) for r in roster_raw],
            "player_id": [int(r["player"]["id"]) for r in roster_raw],
            "position_group_type": [r.get("positionGroupType") for r in roster_raw],
        }
    )

    raw = Path(paths["tracking"]).read_bytes()
    text = bz2.decompress(raw).decode("utf-8") if raw[:2] == b"BZ" else raw.decode("utf-8")
    frames_json = [json.loads(line) for line in text.splitlines() if line.strip()]
    frames_json = _dedupe_gs_frame_records(frames_json)  # GS ships some (period, frame) records 2-16x
    if tracking_limit:
        frames_json = frames_json[:tracking_limit]
    game_id = int(meta.get("id", meta.get("gameId", 0)) or 0)
    rows = []
    for fr in frames_json:
        base = dict(
            game_id=game_id,
            period_id=int(fr["period"]),
            frame_id=int(fr["frameNum"]),
            time_seconds=float(fr.get("periodGameClockTime", 0.0)),
            frame_rate=29.97,
            z=0.0,
            speed_native=float("nan"),
            ball_state="alive",
        )
        for side, key in (("home", "homePlayers"), ("away", "awayPlayers")):
            for p in fr.get(key, []):
                rows.append(
                    {
                        **base,
                        "team_side": side,
                        "jersey_number": str(p["jerseyNum"]),
                        "is_ball": False,
                        "x_centered": float(p["x"]),
                        "y_centered": float(p["y"]),
                    }
                )
        for b in fr.get("balls", []):
            ball_z = b.get("z")
            rows.append(
                {
                    **base,
                    "team_side": None,
                    "jersey_number": None,
                    "is_ball": True,
                    "x_centered": float(b["x"]),
                    "y_centered": float(b["y"]),
                    # ball z IS in the raw GS feed (probe 2026-06-10: present on 100% of
                    # ball records); the old base z=0.0 silently flattened it. Players
                    # keep z=0.0 (no z in GS player records). TF-48 depends on real ball z.
                    "z": float(ball_z) if ball_z is not None else float("nan"),
                }
            )
    jersey_frames = pd.DataFrame(rows)
    resolved, _rep = add_gradientsports_player_ids(
        jersey_frames, roster, home_team_id=home_team_id, away_team_id=away_team_id
    )
    # Extra time needs the ET start direction; the GS converter raises without it.
    resolved, home_start_left_et = _apply_et_direction(resolved, home_start_left_et, label=f"gradientsports {game_id}")
    frames, _r = convert_to_frames(
        resolved,
        home_team_id=home_team_id,
        home_team_start_left=home_start_left,
        home_team_start_left_extratime=home_start_left_et,
    )

    with open(paths["events"], encoding="utf-8") as fh:
        events_json = json.load(fh)
    events_df = _gs_flatten_events(events_json, roster)
    # The events converter is per-period-absolute too (raises on ET without the flag).
    # Apply the same resolution as tracking so actions + frames stay ET-consistent.
    events_df, _ = _apply_et_direction(events_df, home_start_left_et, label=f"gradientsports {game_id} events")
    actions, _r2 = gs_spadl.convert_to_actions(
        events_df,
        home_team_id=home_team_id,
        home_team_start_left=home_start_left,
        home_team_start_left_extratime=home_start_left_et,
    )
    return actions, _preprocess(frames), str(home_team_id)
