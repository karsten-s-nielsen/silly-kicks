"""The StatsBomb 360 parse port -- shape only, never fetch.

Mirrors ``providers/sportec/parse.py``, which fetches nothing: the caller owns I/O and hands the
port already-loaded payloads.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from silly_kicks.providers.statsbomb import (
    JoinReport,
    acting_side_gk_visible,
    defending_gk_visible,
    observed_pitch_fraction,
    shape_snapshots,
)

#: Freeze-frame players are ACTOR-relative: `teammate`/`actor`/`keeper`, and no player identity.
_FF = [
    {"teammate": True, "actor": True, "keeper": False, "location": [60.0, 40.0]},
    {"teammate": True, "actor": False, "keeper": True, "location": [6.0, 40.0]},
    {"teammate": False, "actor": False, "keeper": True, "location": [114.0, 40.0]},
]


def _actions():
    return pd.DataFrame(
        {
            "action_id": [0],
            "original_event_id": ["uuid-1"],
            "game_id": [1],
            "period_id": [1],
            "time_seconds": [10.0],
            "start_x": [60.0],
            "start_y": [34.0],
        }
    )


def _frames(uuid="uuid-1", visible=None):
    return [{"event_uuid": uuid, "freeze_frame": _FF, "visible_area": visible or []}]


def test_snapshots_carry_the_snapshot_to_tracking_frames_contract():
    snaps, _va, _r = shape_snapshots(_frames(), _actions())
    assert set(snaps.columns) >= {"action_id", "team_id", "player_id", "is_goalkeeper", "x", "y"}
    assert len(snaps) == 3


def test_is_goalkeeper_comes_from_the_keeper_flag():
    snaps, _va, _r = shape_snapshots(_frames(), _actions())
    assert snaps["is_goalkeeper"].tolist() == [False, True, True]


def test_team_id_is_actor_relative_not_a_real_identity():
    """SB360 carries no team identity -- only `teammate` relative to the actor."""
    snaps, _va, _r = shape_snapshots(_frames(), _actions())
    assert snaps["team_id"].nunique() == 2, "teammate/opponent must separate into two groups"


def test_coordinates_are_transformed_to_spadl():
    snaps, _va, _r = shape_snapshots(_frames(), _actions())
    assert snaps["x"].max() <= 105.0 and snaps["y"].max() <= 68.0
    # y is INVERTED by the transform: SB y=40 (middle) -> SPADL y~34
    assert snaps["y"].between(33.0, 35.0).all()


def test_zero_overlap_is_COUNTED_not_silently_empty():
    """Measured on the real corpus: 3 of 22 open matches ship a 360 file whose event_uuids have
    ZERO overlap with their own events file. The script already picked WARN + a rate a consumer
    can exclude on; the port adopts that rather than raising or returning silence."""
    with pytest.warns(UserWarning, match="zero overlap"):
        snaps, _va, report = shape_snapshots(_frames(uuid="no-such-uuid"), _actions())
    assert isinstance(report, JoinReport)
    assert report.n_frames == 1
    assert report.n_mapped == 0
    assert report.join_rate == 0.0
    assert snaps.empty, "no frame joined, so no snapshot rows"


def test_a_healthy_join_reports_rate_one():
    _s, _va, report = shape_snapshots(_frames(), _actions())
    assert report.join_rate == 1.0 and report.n_mapped == 1


def test_visible_area_is_a_per_action_polygon_in_spadl_coords():
    flat = [0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0]
    _s, va, _r = shape_snapshots(_frames(visible=flat), _actions())
    assert len(va) == 1
    poly = va.iloc[0]["polygon"]
    assert poly.shape == (4, 2), "a flat 8-element list is 4 vertices, not 1"
    assert poly[:, 0].max() <= 105.0 + 1e-9


def test_a_beyond_touchline_vertex_SURVIVES():
    """The clip is EVENT semantics. A broadcast camera legitimately sees past the touchline, so
    clipping would silently shrink the observed region -- ADR-038's defect class, and the observed
    region is the entire quantity this column carries."""
    flat = [-5.0, -5.0, 125.0, -5.0, 125.0, 85.0, -5.0, 85.0]
    _s, va, _r = shape_snapshots(_frames(visible=flat), _actions())
    poly = va.iloc[0]["polygon"]
    assert poly[:, 0].min() < 0.0, "a vertex outside the pitch was clipped away"
    assert poly[:, 0].max() > 105.0, "a vertex beyond the far touchline was clipped away"


def test_an_absent_visible_area_yields_no_polygon_row_not_a_fabricated_one():
    _s, va, _r = shape_snapshots(_frames(visible=[]), _actions())
    assert va.empty


def test_the_keeper_predicates_answer_different_questions():
    """Which keeper is "the" keeper depends on the action: shots/crosses want the DEFENDING
    keeper, distribution and saves want the ACTING side's -- on a goal kick the keeper IS the
    actor, so the defending predicate excludes them BY CONSTRUCTION."""
    assert defending_gk_visible(_FF) is True
    assert acting_side_gk_visible(_FF) is True
    only_own = [p for p in _FF if p["teammate"]]
    assert defending_gk_visible(only_own) is False
    assert acting_side_gk_visible(only_own) is True


def test_the_port_does_not_import_statsbombpy_or_any_fetcher():
    """Shape, never fetch -- the caller owns I/O (providers/sportec/parse.py fetches nothing).

    AST over IMPORTS, not a substring scan of the source: the module docstring MENTIONS
    statsbombpy while explaining why it is absent, and a text scan cannot tell a described
    dependency from a committed one. `test_provenance_wiring.py` already learned this -- its
    rev-parse detector is AST-matched on CALLS for the same reason.
    """
    import ast
    import pathlib

    import silly_kicks.providers.statsbomb.parse as mod

    tree = ast.parse(pathlib.Path(mod.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"statsbombpy", "requests", "urllib", "urllib3", "httpx", "aiohttp"}
    assert not (imported & forbidden), f"the port must not fetch; it imports {sorted(imported & forbidden)}"


def test_the_prose_scan_would_have_been_fooled():
    """Non-vacuity for the choice of AST over grep: the docstring DOES contain the token, so a
    substring scan passes only by accident of wording and fails on a correct module."""
    import pathlib

    import silly_kicks.providers.statsbomb.parse as mod

    src = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "statsbombpy" in src, "docstring no longer explains the absence -- update both tests together"


def test_player_id_is_synthetic_and_does_NOT_recur_across_frames():
    """SB360 carries no player identity. snapshot_to_tracking_frames assigns a synthetic
    sequential int over the WHOLE table, so the same physical player gets a different id in every
    freeze-frame. That forecloses per-player aggregation and belongs in the contract, not in a
    downstream surprise."""
    two = [*_frames(), {"event_uuid": "uuid-2", "freeze_frame": _FF, "visible_area": []}]
    actions = pd.DataFrame(
        {
            "action_id": [0, 1],
            "original_event_id": ["uuid-1", "uuid-2"],
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [10.0, 20.0],
            "start_x": [60.0, 60.0],
            "start_y": [34.0, 34.0],
        }
    )
    snaps, _va, _r = shape_snapshots(two, actions)
    per_action = snaps.groupby("action_id")["player_id"].apply(list)
    assert per_action.loc[0] != per_action.loc[1], (
        "player_id recurs across frames -- if that becomes true, the contract note is wrong"
    )
    assert not np.array_equal(np.sort(per_action.loc[0]), np.sort(per_action.loc[1]))


def test_players_and_polygon_share_one_transform():
    """The binding constraint behind applying crc to the polygon.

    Players and the region that bounds them must land in ONE frame. A player placed at a polygon
    VERTEX in native SB coordinates must transform to that same vertex in SPADL; if the polygon
    skipped the cell-centre correction that players receive, it would sit 0.4375 m away
    (fidelity 1) and a player exactly on the boundary would read as outside it.
    """
    corner = [10.0, 10.0]
    flat = [10.0, 10.0, 110.0, 10.0, 110.0, 70.0, 10.0, 70.0]
    frames = [
        {
            "event_uuid": "uuid-1",
            "freeze_frame": [{"teammate": True, "actor": True, "keeper": False, "location": corner}],
            "visible_area": flat,
        }
    ]
    snaps, va, _r = shape_snapshots(frames, _actions())
    player_xy = np.array([snaps.iloc[0]["x"], snaps.iloc[0]["y"]])
    vertex_xy = va.iloc[0]["polygon"][0]
    np.testing.assert_allclose(
        player_xy,
        vertex_xy,
        rtol=0,
        atol=1e-9,
        err_msg="a player AT a polygon vertex must transform to that vertex -- the two paths have diverged",
    )


def test_crc_moves_observed_pitch_fraction_once_the_ratio_is_CLIPPED():
    """ADR-055: the old witness went VACUOUS, and the reason it witnessed is now wrong.

    The retired assertion was ``observed_pitch_fraction(flat) == observed_pitch_fraction(shifted)`` on a
    polygon at x 10-110, y 10-70 -- ENTIRELY INTERIOR to the 120x80 pitch. Under clipping that
    test keeps passing (measured: 0.625 both ways, delta exactly 0.0) while the property it
    witnesses becomes FALSE, because for any polygon crossing the touchline the 0.4375 m
    translation changes the CLIPPED intersection area. A guard that still passes while the
    reasoning under it dissolves is worse than no guard.

    So this asserts BOTH sides. ADR-054 D5's *conclusion* stands -- apply crc -- but its
    supporting argument is no longer "crc is invisible here"; it is **alignment**: players reach
    SPADL through the same ``sb_xy_to_spadl`` WITH crc, so omitting it on the polygon would
    offset the region by 0.4375 m from the players it bounds and put a boundary player outside
    it. That is the reason, and it never depended on the ratio being invariant.
    """
    interior = [10.0, 10.0, 110.0, 10.0, 110.0, 70.0, 10.0, 70.0]
    crossing = [-5.0, 10.0, 110.0, 10.0, 110.0, 70.0, -5.0, 70.0]

    # crc == cell_side(fidelity 1) / 2 == 0.4375, read from the library rather than typed in,
    # so a fidelity change moves the fixture instead of quietly invalidating it.
    from silly_kicks.spadl._sb_coordinates import cell_side

    crc = cell_side(1) / 2

    def shifted(flat):
        return [v - crc for v in flat]

    # Interior: still invariant -- which is exactly why it can no longer carry the argument.
    assert observed_pitch_fraction(interior) == observed_pitch_fraction(shifted(interior))

    # Touchline-crossing: the translation DOES move the clipped ratio. Measured 0.002734375
    # (0.687500000 -> 0.684765625); asserted as a lower bound so float noise cannot satisfy it.
    base = observed_pitch_fraction(crossing)
    moved = observed_pitch_fraction(shifted(crossing))
    assert base != moved
    assert abs(base - moved) > 1e-4, (
        f"crc must measurably move the CLIPPED fraction on a touchline-crossing polygon "
        f"({base} vs {moved}); if it does not, this witness has gone vacuous again"
    )
