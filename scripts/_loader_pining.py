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
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

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
    provider: str, match_id: str, artifact_key: str, token: str, base_url: str, dest_dir: Path
) -> Path:
    """Two-step: bearer GET -> 302 Location -> presigned GET (no bearer) -> stream to a temp file.

    Streams so the ~419 MB IDSSE tracking.xml never sits fully in memory.
    """
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
    dest = dest_dir / f"{provider}_{match_id}_{artifact_key}"
    with urllib.request.urlopen(location, timeout=600) as resp, open(dest, "wb") as fh:  # noqa: S310
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            fh.write(chunk)
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
) -> Iterator[tuple[str, str, pd.DataFrame, pd.DataFrame, object]]:
    """Yield (provider, match_id, actions, frames, home_team_id) for each requested match.

    ``tracking_limit`` caps frames loaded per match (passed to the kloppy parsers) — essential for
    the ~419 MB IDSSE tracking file in dev/e2e loops.
    """
    import tempfile

    tok, base_url = _resolve_token(token), _base_url()
    for provider in providers:
        manifest = {m["id"]: m for m in _list_matches(provider, tok, base_url)}
        wanted = (match_ids.get(provider) if match_ids else None) or list(manifest)
        for match_id in wanted:
            artifacts = manifest[match_id]["artifacts"]
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                paths = _download_artifacts(provider, match_id, artifacts, tok, base_url, tmp_dir)
                actions, frames, home = _build_match(provider, match_id, paths, tracking_limit)
            yield provider, match_id, actions, frames, home


def _download_artifacts(provider, match_id, artifacts, token, base_url, tmp_dir) -> dict[str, Path]:
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
        out[role] = _download_to_temp(provider, match_id, artifact_key, token, base_url, tmp_dir)
    return out


def _build_match(provider, match_id, paths, tracking_limit):
    """Provider dispatch: parse the downloaded artifacts into (actions, frames, home_team_id)."""
    if provider == "idsse":
        return _build_idsse(paths, tracking_limit)
    if provider == "skillcorner":
        return _build_skillcorner(paths, match_id, tracking_limit)
    if provider == "gradientsports":
        return _build_gradientsports(paths, tracking_limit)
    raise ValueError(f"unknown pining provider {provider!r}")


def _preprocess(frames: pd.DataFrame) -> pd.DataFrame:
    """Derive velocities (vx/vy) so bekkers_pi + velocity-aware carrier inference work."""
    from silly_kicks.tracking.preprocess import derive_velocities, smooth_frames

    return derive_velocities(smooth_frames(frames))


def _build_skillcorner(paths, match_id, tracking_limit):
    """SkillCorner: kloppy tracking + silly-kicks SkillCorner events converter."""
    from kloppy import skillcorner

    from silly_kicks.tracking import kloppy as tracking_kloppy

    ds = skillcorner.load(
        meta_data=str(paths["metadata"]),
        raw_data=str(paths["tracking"]),
        limit=tracking_limit,
        include_empty_frames=False,
    )
    frames, _report = tracking_kloppy.convert_to_frames(ds)
    # Events: SkillCorner dynamic-events CSV + match.json -> silly-kicks SkillCorner SPADL converter.
    from silly_kicks.spadl import skillcorner as sk_spadl

    with open(paths["metadata"], encoding="utf-8") as fh:
        meta = json.load(fh)
    home_team_id = str(meta["home_team"]["id"])  # authoritative; matches kloppy tracking team ids
    raw_events = pd.read_csv(paths["events"], low_memory=False)
    actions, _evt_report = sk_spadl.convert_to_actions(raw_events, meta)
    return actions, _preprocess(frames), home_team_id


def _build_idsse(paths, tracking_limit):
    """IDSSE (DFL/Sportec XML): kloppy events (-> SPADL via kloppy gateway) + kloppy-parsed
    tracking mapped to the silly-kicks frames schema (the silly-kicks tracking kloppy gateway
    refuses Sportec by ADR-004, so the loader maps the kloppy TrackingDataset directly)."""
    from kloppy import sportec

    from silly_kicks.spadl import kloppy as spadl_kloppy

    ev = sportec.load_event(event_data=str(paths["events"]), meta_data=str(paths["metadata"]))
    actions, _evt_report = spadl_kloppy.convert_to_actions(ev)
    tr = sportec.load_tracking(meta_data=str(paths["metadata"]), raw_data=str(paths["tracking"]), limit=tracking_limit)
    frames, home_team_id = _kloppy_tracking_to_frames(tr)
    return actions, _preprocess(frames), home_team_id


def _kloppy_tracking_to_frames(dataset):
    """Map a kloppy sportec TrackingDataset -> silly-kicks TRACKING_FRAMES_COLUMNS (loader-local;
    avoids the ADR-004 gateway block for Sportec while still using kloppy's DFL XML parser)."""
    from kloppy.domain import Dimension, MetricPitchDimensions, Orientation

    transformed = dataset.transform(
        to_pitch_dimensions=MetricPitchDimensions(
            x_dim=Dimension(0, 105.0),
            y_dim=Dimension(0, 68.0),
            standardized=False,
            pitch_length=105.0,
            pitch_width=68.0,
        ),
        to_orientation=Orientation.HOME_AWAY,
    )
    home_team = transformed.metadata.teams[0]
    home_team_id = str(home_team.team_id)
    frame_rate = float(transformed.metadata.frame_rate or 25.0)
    game_id = str(transformed.metadata.game_id or "idsse")
    rows: list[dict] = []
    for period_frame in transformed.records:
        fid = int(period_frame.frame_id)
        pid = int(period_frame.period.id) if period_frame.period else 1
        t_s = float(period_frame.timestamp.total_seconds()) if period_frame.timestamp is not None else fid / frame_rate
        for player, pdata in period_frame.players_data.items():
            if pdata.coordinates is None:
                continue
            tid = str(player.team.team_id)
            rows.append(
                {
                    "game_id": game_id,
                    "period_id": pid,
                    "frame_id": fid,
                    "time_seconds": t_s,
                    "frame_rate": frame_rate,
                    "player_id": str(player.player_id),
                    "team_id": tid,
                    "is_ball": False,
                    "is_goalkeeper": str(player.starting_position or "") == "Goalkeeper",
                    "x": float(pdata.coordinates.x),
                    "y": float(pdata.coordinates.y),
                    "z": 0.0,
                    "speed": pdata.speed if pdata.speed is not None else float("nan"),
                    "speed_source": "native" if pdata.speed is not None else None,
                    "ball_state": "alive",
                    "team_attacking_direction": "ltr" if tid == home_team_id else "rtl",
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "sportec",
                }
            )
        if period_frame.ball_coordinates is not None:
            rows.append(
                {
                    "game_id": game_id,
                    "period_id": pid,
                    "frame_id": fid,
                    "time_seconds": t_s,
                    "frame_rate": frame_rate,
                    "player_id": None,
                    "team_id": None,
                    "is_ball": True,
                    "is_goalkeeper": False,
                    "x": float(period_frame.ball_coordinates.x),
                    "y": float(period_frame.ball_coordinates.y),
                    "z": 0.0,
                    "speed": float("nan"),
                    "speed_source": None,
                    "ball_state": "alive",
                    "team_attacking_direction": None,
                    "confidence": None,
                    "visibility": None,
                    "source_provider": "sportec",
                }
            )
    return pd.DataFrame(rows), home_team_id


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
            rows.append(
                {
                    **base,
                    "team_side": None,
                    "jersey_number": None,
                    "is_ball": True,
                    "x_centered": float(b["x"]),
                    "y_centered": float(b["y"]),
                }
            )
    jersey_frames = pd.DataFrame(rows)
    resolved, _rep = add_gradientsports_player_ids(
        jersey_frames, roster, home_team_id=home_team_id, away_team_id=away_team_id
    )
    frames, _r = convert_to_frames(resolved, home_team_id=home_team_id, home_team_start_left=home_start_left)

    with open(paths["events"], encoding="utf-8") as fh:
        events_json = json.load(fh)
    events_df = _gs_flatten_events(events_json, roster)
    actions, _r2 = gs_spadl.convert_to_actions(
        events_df, home_team_id=home_team_id, home_team_start_left=home_start_left
    )
    return actions, _preprocess(frames), str(home_team_id)
