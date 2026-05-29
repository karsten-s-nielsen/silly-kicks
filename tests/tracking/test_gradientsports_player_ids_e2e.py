"""TF-24 PR-A e2e: real GS data proves carrier accuracy 0.0 -> nonzero (the bug fix).

Sources GS WC2022 (owner-tier / private) from the pining-for-the-data mock provider API.
Env-gated (skips in CI and whenever the owner token is absent). NO local paths; commits
nothing GS-derived (licence gate)."""

from __future__ import annotations

import bz2
import json
import os
import urllib.error
import urllib.request

import pandas as pd
import pytest

# API base is the deployed mock-provider URL (NOT a secret; includes the /v1 stage); the OWNER
# token is read from the env (never hardcoded -- it is the gated secret).
_API = os.environ.get("PINING_API_URL", "https://ozqgk9a3ji.execute-api.us-east-1.amazonaws.com/v1")
_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")  # owner-tier bearer; GS is private
_PROVIDER = "gradientsports"


def _get_json(path: str) -> object:
    req = urllib.request.Request(f"{_API}{path}", headers={"Authorization": f"Bearer {_TOKEN}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _fetch_artifact(match_id: str, artifact: str) -> bytes:
    """Two-step gated download (mirrors scripts/verify_gradient_load.py): GET the API with the
    bearer -> 302 -> GET the presigned S3 URL WITHOUT the bearer (S3 rejects double-auth)."""
    path = f"/{_PROVIDER}/matches/{match_id}/{artifact}"

    class _NoFollow(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    api_req = urllib.request.Request(f"{_API}{path}", headers={"Authorization": f"Bearer {_TOKEN}"})
    try:
        with urllib.request.build_opener(_NoFollow).open(api_req, timeout=30) as resp:
            return resp.read()  # direct 200 (unlikely for the 302 path)
    except urllib.error.HTTPError as e:
        if e.code != 302:
            raise
        location = e.headers["Location"]
    with urllib.request.urlopen(urllib.request.Request(location), timeout=120) as resp:
        return resp.read()


def _read_jsonl(raw: bytes) -> list[dict]:
    text = bz2.decompress(raw).decode("utf-8") if raw[:2] == b"BZ" else raw.decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


@pytest.mark.e2e
@pytest.mark.skipif(not _TOKEN, reason="PINING_FOR_THE_DATA_TOKEN not set")
def test_real_gs_carrier_accuracy_nonzero():
    from silly_kicks.tracking import infer_ball_carrier, link_actions_to_frames
    from silly_kicks.tracking.gradientsports import add_gradientsports_player_ids, convert_to_frames

    matches = _get_json(f"/{_PROVIDER}/matches")["matches"]  # type: ignore[index]
    match_id = str(matches[0]["id"])

    meta = json.loads(_fetch_artifact(match_id, "metadata"))[0]  # metadata artifact is a 1-element list
    roster_raw = json.loads(_fetch_artifact(match_id, "roster"))
    events_raw = json.loads(_fetch_artifact(match_id, "events"))
    frames_raw = _read_jsonl(_fetch_artifact(match_id, "tracking"))[:3000]  # slice for runtime

    home_team_id = int(meta["homeTeam"]["id"])
    away_team_id = int(meta["awayTeam"]["id"])
    roster = pd.DataFrame(
        {
            "team_id": [int(r["team"]["id"]) for r in roster_raw],
            "shirt_number": [str(r["shirtNumber"]) for r in roster_raw],
            "player_id": [int(r["player"]["id"]) for r in roster_raw],
            "position_group_type": [r.get("positionGroupType") for r in roster_raw],
        }
    )

    # flatten tracking JSONL -> jersey-keyed long form (homePlayers/awayPlayers/balls)
    rows = []
    for fr in frames_raw:
        base = dict(
            game_id=int(match_id),
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

    resolved, report = add_gradientsports_player_ids(
        jersey_frames, roster, home_team_id=home_team_id, away_team_id=away_team_id
    )
    assert report.n_matched > 0
    frames, _ = convert_to_frames(
        resolved,
        home_team_id=home_team_id,
        home_team_start_left=bool(meta.get("homeTeamStartLeft", True)),
        output_convention="ltr",
    )
    carrier = infer_ball_carrier(frames)
    assert len(carrier.dropna(subset=["ball_carrier_player_id"])) > 0
    # supporting: resolved carrier ids live in the events int player.id space
    assert carrier["ball_carrier_player_id"].dropna().isin(roster["player_id"]).mean() > 0

    # --- LOAD-BEARING proof (C3): real events<->tracking carrier join via RAW on-ball event
    # actor ids (gameEvents.playerId IS the events int player_id space). No SPADL conversion.
    on_ball_types = {"PA", "CR", "SH", "BC"}  # actor == ball carrier for these
    acts = []
    for ev in events_raw:
        ge = ev.get("gameEvents") or {}
        pe = ev.get("possessionEvents") or {}
        pid, tid, per, t = ge.get("playerId"), ge.get("teamId"), ge.get("period"), ge.get("startGameClock")
        if ge.get("gameEventType") != "OTB" or pe.get("possessionEventType") not in on_ball_types:
            continue
        if pid is None or per is None or t is None:
            continue
        acts.append(
            dict(
                action_id=len(acts),
                game_id=int(match_id),
                period_id=int(per),
                time_seconds=float(t),
                team_id=int(tid) if tid is not None else 0,
                player_id=int(pid),
                type_name="pass",
            )
        )
    actions = pd.DataFrame(acts)
    fmax = frames.groupby("period_id")["time_seconds"].max().to_dict()
    actions = actions[actions.apply(lambda r: r["time_seconds"] <= fmax.get(r["period_id"], -1), axis=1)]
    assert len(actions) > 0, "no on-ball events fell inside the loaded tracking window"

    pointers, _ = link_actions_to_frames(actions, frames)
    linked = (
        actions.merge(pointers[["action_id", "frame_id"]], on="action_id")
        .merge(
            carrier[["game_id", "period_id", "frame_id", "ball_carrier_player_id"]],
            on=["game_id", "period_id", "frame_id"],
            how="left",
        )
        .dropna(subset=["ball_carrier_player_id"])
    )
    accuracy = (linked["player_id"] == linked["ball_carrier_player_id"]).mean()
    assert accuracy > 0.0, f"GS carrier accuracy {accuracy} -- the 0.0 regression is NOT fixed"
