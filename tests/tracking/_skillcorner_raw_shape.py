"""Hand-built SkillCorner RAW tracking records (TF-46 Arm 1 Part 2, spec section 17.7, Step 0 shape).

Encodes the observed opendata / owner-tier record structure -- ``ball_data`` (x/y/z/is_detected),
``player_data`` (player_id/x/y/is_detected), ``possession`` (player_id/group), and
``image_corners_projection`` (eight corner keys) -- INCLUDING the frame-0 null case. No copied provider data
beyond the key structure (the exact keys were read from the committed public slice and the owner-tier parquet
schema). The loader ``scripts._loader_pining._skillcorner_bronze`` consumes a list of these dicts.
"""

from __future__ import annotations

_FOOTPRINT = {
    "x_top_left": -52.5,
    "y_top_left": 34.0,
    "x_top_right": 52.5,
    "y_top_right": 34.0,
    "x_bottom_right": 52.5,
    "y_bottom_right": -34.0,
    "x_bottom_left": -52.5,
    "y_bottom_left": -34.0,
}


def raw_record(
    frame: int,
    period: int,
    ts: float,
    *,
    ball_detected: object = True,
    poss_player_id: object = 366,
    poss_group: object = "attack",
    footprint: bool = True,
    players: list[dict] | None = None,
) -> dict:
    """One tracking record. ``players`` defaults to GK 366 + outfielder 367, both detected."""
    return {
        "frame": frame,
        "period": period,
        "timestamp": ts,
        "ball_data": {"x": 1.0, "y": 2.0, "z": 0.5, "is_detected": ball_detected},
        "possession": {"player_id": poss_player_id, "group": poss_group},
        "image_corners_projection": dict(_FOOTPRINT) if footprint else None,
        "player_data": players
        if players is not None
        else [
            {"player_id": 366, "x": -10.0, "y": 0.0, "is_detected": True},
            {"player_id": 367, "x": 10.0, "y": 5.0, "is_detected": True},
        ],
    }


def null_frame(frame: int = 0, period: int = 1) -> dict:
    """The opendata frame-0 null case: null possession, null footprint, undetected ball."""
    return {
        "frame": frame,
        "period": period,
        "timestamp": 0.0,
        "ball_data": {"x": None, "y": None, "z": None, "is_detected": None},
        "possession": None,
        "image_corners_projection": None,
        "player_data": [{"player_id": 366, "x": -10.0, "y": 0.0, "is_detected": False}],
    }


def meta() -> dict:
    """Minimal match.json: a GK (366) + outfielder (367) on team 10, a 105x68 pitch."""
    return {
        "id": "m1",
        "home_team": {"id": 10},
        "pitch_length": 105.0,
        "pitch_width": 68.0,
        "players": [
            {"id": 366, "team_id": 10, "player_role": {"acronym": "GK"}},
            {"id": 367, "team_id": 10, "player_role": {"acronym": "CB"}},
        ],
    }
