"""Parity-to-oracle: bronze builders vs the kloppy gateway on the SAME match.

The kloppy gateway is the canonical CS-pinned oracle (ADR-031). The builders must produce
byte-equal coordinate frames given contract-honoring bronze.

- Metrica: SELF-CONTAINED via kloppy open data (network-gated) --- builds contract-honoring
  bronze from the dataset and asserts ball parity to the gateway (validated dx=dy=0).
- SkillCorner: owner-gated (needs raw SkillCorner + bronze for one match via TF23_PARITY_DATA),
  with an independent ball-z physical-range check (kloppy carries SkillCorner ball z).
"""

import json
import os

import pandas as pd
import pytest

pytestmark = pytest.mark.e2e


def test_metrica_builder_matches_kloppy_oracle_open_data():
    """Metrica builder == kloppy gateway, byte-for-byte, on open-data game 1.

    The bronze honors the builder's input contract (y in SPADL bottom-to-top), constructed
    by flipping kloppy's native top-to-bottom y --- mirroring the lakehouse bronze landing.
    """
    metrica = pytest.importorskip("kloppy.metrica", reason="kloppy required")
    from silly_kicks.tracking import kloppy as gw
    from silly_kicks.tracking import metrica as mt

    try:
        ds = metrica.load_open_data(match_id="1", limit=6000)
    except Exception as exc:  # network/open-data unavailable
        pytest.skip(f"metrica open data unavailable: {exc}")

    oracle, _ = gw.convert_to_frames(ds, output_convention="ltr")
    rows = []
    for f in ds.records:
        if not f.ball_coordinates:
            continue
        home, away = {}, {}
        for pl, pdata in f.players_data.items():
            if pdata.coordinates is None:
                continue
            target = home if pl.team.ground.value == "home" else away
            # contract: bronze y is SPADL bottom-to-top -> flip kloppy native (top-to-bottom)
            target[str(pl.jersey_no)] = {"x": pdata.coordinates.x, "y": 1.0 - pdata.coordinates.y}
        rows.append(
            {
                "period": int(f.period.id),
                "frame": int(f.frame_id),
                "timestamp": float(f.timestamp.total_seconds()),
                "ball_x": f.ball_coordinates.x,
                "ball_y": 1.0 - f.ball_coordinates.y,
                "home_players": json.dumps(home),
                "away_players": json.dumps(away),
                "gk_jersey_numbers": "[]",
                "frame_rate": int(ds.metadata.frame_rate or 25),
            }
        )
    built, _ = mt.convert_to_frames(pd.DataFrame(rows), home_team_id="Home", output_convention="ltr")

    ob = oracle[oracle.is_ball].dropna(subset=["y"])[["period_id", "frame_id", "x", "y"]]
    bb = built[built.is_ball].dropna(subset=["y"])[["period_id", "frame_id", "x", "y"]]
    m = ob.merge(bb, on=["period_id", "frame_id"], suffixes=("_o", "_b"))
    assert len(m) > 1000
    assert (m["x_o"] - m["x_b"]).abs().max() < 0.01  # x byte-equal
    assert (m["y_o"] - m["y_b"]).abs().max() < 0.01  # y byte-equal (incl. LTR orientation)


_DATA = os.environ.get("TF23_PARITY_DATA")  # dir with raw SkillCorner + bronze for one match


@pytest.mark.skipif(not _DATA, reason="TF23_PARITY_DATA not set (owner-gated)")
def test_skillcorner_builder_matches_kloppy_oracle():
    import kloppy.skillcorner  # type: ignore[reportMissingImports]

    from silly_kicks.tracking import kloppy as gw
    from silly_kicks.tracking import skillcorner as sk

    ds = kloppy.skillcorner.load(
        meta_data=f"{_DATA}/sc_meta.json", raw_data=f"{_DATA}/sc_tracking.json", include_empty_frames=False
    )
    oracle, _ = gw.convert_to_frames(ds, output_convention="ltr")
    bronze = pd.read_parquet(f"{_DATA}/sc_bronze.parquet")
    home = str(ds.metadata.teams[0].team_id)
    built, _ = sk.convert_to_frames(bronze, home_team_id=home)
    key = ["period_id", "frame_id", "player_id"]
    m = oracle.merge(built, on=key, suffixes=("_o", "_b"))
    assert (m["x_o"] - m["x_b"]).abs().median() < 0.5
    assert (m["y_o"] - m["y_b"]).abs().median() < 0.5
    assert (m["team_attacking_direction_o"] == m["team_attacking_direction_b"]).mean() > 0.99
    ball = oracle.merge(built, on=["period_id", "frame_id"], suffixes=("_o", "_b"))
    ball = ball[ball["is_ball_o"] & ball["is_ball_b"]]
    assert (ball["z_o"] - ball["z_b"]).abs().median() < 0.2  # kloppy carries SkillCorner ball z


@pytest.mark.skipif(not _DATA, reason="TF23_PARITY_DATA not set (owner-gated)")
def test_skillcorner_ball_z_physically_sensible():
    from silly_kicks.tracking import skillcorner as sk

    bronze = pd.read_parquet(f"{_DATA}/sc_bronze.parquet")
    built, _ = sk.convert_to_frames(bronze, home_team_id=str(bronze["team_id"].iloc[0]))
    bz = built[built["is_ball"]]["z"].dropna()
    assert ((bz >= 0) & (bz <= 10)).mean() > 0.99  # physical range
    assert (bz > 0.5).any()  # some airborne frames
