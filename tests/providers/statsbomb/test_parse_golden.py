"""Golden parity for the StatsBomb 360 port, against a COMMITTED real slice.

Mirrors ``tests/datasets/sportec/idsse_slice/``. StatsBomb open data is redistributable under the
**StatsBomb Public Data License (non-commercial)** -- see ``tests/datasets/statsbomb/README.md``
and ``NOTICE`` -- so unlike the owner-tier GS corpus a real slice can travel.

**Read with stdlib json, never statsbombpy.** That package is installed in some local envs and
declared nowhere in ``pyproject.toml``, so an ``importorskip``-guarded golden gate would be
vacuously green wherever it is absent -- the fixture-that-never-runs shape. This test cannot skip.

Slice: Women's World Cup 2023, match 3893795, 6 freeze-frames including a Goal Keeper event.
Digests in ``SOURCE_SHA``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from silly_kicks.providers.statsbomb import observed_pitch_fraction, shape_snapshots

_SLICE = pathlib.Path(__file__).resolve().parents[2] / "datasets" / "statsbomb" / "three-sixty"


def _frames_raw() -> list[dict]:
    return json.loads((_SLICE / "frames.json").read_text(encoding="utf-8"))


def _actions() -> pd.DataFrame:
    """Minimal SPADL-shaped actions keyed to the slice's event_uuids."""
    raw = _frames_raw()
    return pd.DataFrame(
        {
            "action_id": range(len(raw)),
            "original_event_id": [r["event_uuid"] for r in raw],
            "game_id": 3893795,
            "period_id": 1,
            "time_seconds": [float(i) for i in range(len(raw))],
            "start_x": 52.5,
            "start_y": 34.0,
        }
    )


def test_the_slice_matches_its_recorded_digests():
    """SOURCE_SHA pins the bytes. A silently re-fetched or re-formatted slice changes every
    number downstream of it, and would otherwise do so invisibly."""
    recorded = {}
    for line in (_SLICE / "SOURCE_SHA").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split()
            recorded[name] = digest
    assert recorded, "SOURCE_SHA is empty"
    for name, digest in recorded.items():
        actual = hashlib.sha256((_SLICE / name).read_bytes()).hexdigest()
        assert actual == digest, f"{name} does not match SOURCE_SHA -- the slice was modified"


def test_the_slice_round_trips_through_the_port():
    raw = _frames_raw()
    snapshots, visible_area, report = shape_snapshots(raw, _actions())

    assert report.n_frames == len(raw)
    assert report.join_rate == 1.0, "every committed frame is keyed to an action by construction"

    # Real freeze-frames carry ~19 players; the port must not silently drop any with a location.
    n_locatable = sum(1 for r in raw for p in (r.get("freeze_frame") or []) if isinstance(p.get("location"), list))
    assert len(snapshots) == n_locatable

    assert snapshots["x"].between(0.0, 105.0).all()
    assert snapshots["y"].between(0.0, 68.0).all()
    assert snapshots["is_goalkeeper"].any(), "the slice includes a Goal Keeper event"

    assert len(visible_area) == len(raw), "every record in this slice carries a visible_area"


def test_real_polygons_are_not_degenerate_and_not_clipped_flat():
    """Non-vacuity: a polygon collapsed to one vertex, or clamped onto the pitch rectangle, would
    still pass a shape check. Assert it has real extent and real vertices."""
    _s, visible_area, _r = shape_snapshots(_frames_raw(), _actions())
    for poly in visible_area["polygon"]:
        assert poly.shape[0] >= 3, f"a polygon with {poly.shape[0]} vertices has no area"
        assert np.ptp(poly[:, 0]) > 1.0, "polygon has no x extent -- suspect a flat-list mis-parse"


def test_observed_pitch_fraction_on_real_data_is_a_plausible_camera_view():
    """A broadcast frame sees part of the pitch, not none and not all of it."""
    fracs = [observed_pitch_fraction(r.get("visible_area") or []) for r in _frames_raw()]
    assert all(0.0 < f < 1.0 for f in fracs), f"implausible visible fractions: {fracs}"


def test_players_land_inside_or_near_their_own_visible_area():
    """The alignment the cell-centre-correction decision turns on.

    Freeze-frame players are, by construction, the players the camera SAW -- so they must sit
    within the observed region (a small tolerance absorbs polygon simplification upstream). If
    the polygon and the players were transformed differently, this drifts by ~0.44 m or more.
    """
    from matplotlib.path import Path as MplPath

    snapshots, visible_area, _r = shape_snapshots(_frames_raw(), _actions())
    va = dict(zip(visible_area["action_id"], visible_area["polygon"], strict=True))
    outside = 0
    total = 0
    for action_id, grp in snapshots.groupby("action_id"):
        poly = va.get(action_id)
        if poly is None:
            continue
        path = MplPath(poly)
        pts = grp[["x", "y"]].to_numpy()
        total += len(pts)
        outside += int((~path.contains_points(pts, radius=1.0)).sum())
    assert total > 0, "no players scored -- the test would be vacuous"
    assert outside / total < 0.25, (
        f"{outside}/{total} seen players fall outside their own visible_area -- "
        f"suspect the polygon and the players took different transforms"
    )
