"""TF-46 add_corner_windows: TF-79 timing + winner + diagnostics."""

from __future__ import annotations

import pandas as pd

from silly_kicks.corners import CORNER_WINDOW_COLUMNS, add_corner_windows
from tests.tracking._restart_fixtures import GateTableStub

from ._gate_helpers import passing


def _corner(source="spadl_corner", t_kick=100.0, period=1):
    return pd.DataFrame(
        {
            "game_id": ["g1"],
            "corner_id": [0],
            "period_id": [period],
            "t_kick": [t_kick],
            "attacking_team_id": ["tA"],
            "corner_side": pd.Series(["left"], dtype="category"),
            "kick_x": [104.5],
            "kick_y": [1.0],
            "corner_kind": pd.Series(["direct"], dtype="category"),
            "corner_source": pd.Series([source], dtype="category"),
        }
    )


def _frames():
    return pd.DataFrame(
        {
            "frame_id": [5, 5],
            "period_id": [1, 1],
            "time_seconds": [100.5, 100.5],
            "player_id": ["pW", "pX"],
            "team_id": ["tW", "tX"],
        }
    )


def _contacts(frame_id=5, t=100.5, player="pW", period=1):
    return pd.DataFrame(
        {
            "period_id": [period],
            "frame_id": [frame_id],
            "time_seconds": [t],
            "player_id": [player],
            "contact_source": pd.Series(["velocity_break"], dtype="object"),
        }
    )


def _no_contacts():
    return pd.DataFrame(
        {
            "period_id": pd.Series([], dtype="int64"),
            "frame_id": pd.Series([], dtype="int64"),
            "time_seconds": pd.Series([], dtype="float64"),
            "player_id": pd.Series([], dtype="object"),
            "contact_source": pd.Series([], dtype="object"),
        }
    )


def test_target_uses_first_contact_plus_lead_when_resolved():
    # LIVE path: a passing first-contact gate is injected (spec section 9.2).
    out = add_corner_windows(_corner(), _frames(), _contacts(), **passing())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "first_contact"
    assert r["first_contact_time"] == 100.5
    assert r["t_target"] == min(100.5 + 1.0, 100.0 + 2.0)  # 101.5
    assert r["t_init"] == 98.0
    assert r["t_outcome"] == 106.0


def test_winner_team_resolved_from_frames():
    out = add_corner_windows(_corner(), _frames(), _contacts(player="pW"), **passing())
    r = out.iloc[0]
    assert r["winner_player_id"] == "pW"
    assert r["winner_team_id"] == "tW"
    assert r["winner_source"] == "tf79_contact"


def test_winner_withheld_without_a_gate_entry():
    # WITHHELD path (CL-3 fail-closed): the commit-2 empty table withholds the winner.
    out = add_corner_windows(_corner(), _frames(), _contacts(player="pW"))
    r = out.iloc[0]
    assert r["winner_source"] == "withheld"
    assert pd.isna(r["winner_player_id"]) and pd.isna(r["winner_team_id"])
    assert pd.isna(r["first_contact_time"])
    assert r["target_snapshot_source"] == "kick_plus_2s"  # fallback snapshot


def test_fallback_to_kick_plus_2s_when_no_contact():
    out = add_corner_windows(_corner(), _frames(), _no_contacts())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "kick_plus_2s"
    assert r["t_target"] == 102.0
    assert pd.isna(r["winner_team_id"])  # no GI fallback
    assert pd.isna(r["winner_player_id"])
    assert pd.isna(r["first_contact_time"])


def test_sb360_no_contacts_winner_is_na_source_sb360_event():
    out = add_corner_windows(_corner(source="sb360_event"), _frames(), _no_contacts())
    r = out.iloc[0]
    assert r["target_snapshot_source"] == "sb360_event"
    assert pd.isna(r["winner_team_id"])
    assert pd.isna(r["first_contact_time"])


def test_first_possession_diagnostic_column_emitted():
    out = add_corner_windows(_corner(), _frames(), _contacts())
    assert "first_possession_time" in out.columns


def test_gi_param_is_deprecated_and_ignored():
    # TF-46 Arm 1 Part 2 (spec section 9.4): gi= is deprecated + ignored; first_possession_time no longer
    # comes from an exact-float t_kick join.
    import pytest

    gi = pd.DataFrame({"game_id": ["g1"], "period_id": [1], "time_seconds": [100.0]})
    with pytest.warns(DeprecationWarning, match="deprecated and ignored"):
        out = add_corner_windows(_corner(), _frames(), _contacts(), gi=gi)
    assert out.iloc[0]["first_possession_time"] != 100.0 or pd.isna(out.iloc[0]["first_possession_time"])


def test_first_possession_time_from_gi_next_possession_time():
    corner = _corner()
    corner["gi_next_possession_time"] = 42.0
    corner["corner_source"] = "skillcorner_marker"
    out = add_corner_windows(corner, _frames(), _contacts())
    assert out.iloc[0]["first_possession_time"] == 42.0
    assert out.iloc[0]["first_possession_status"] == "gi"


def test_all_window_columns_present_and_typed():
    out = add_corner_windows(_corner(), _frames(), _contacts())
    for col in CORNER_WINDOW_COLUMNS:
        assert col in out.columns


def test_pure_does_not_mutate_inputs():
    corners, frames, contacts = _corner(), _frames(), _contacts()
    cb, fb, kb = corners.copy(deep=True), frames.copy(deep=True), contacts.copy(deep=True)
    out = add_corner_windows(corners, frames, contacts)
    pd.testing.assert_frame_equal(corners, cb)
    pd.testing.assert_frame_equal(frames, fb)
    pd.testing.assert_frame_equal(contacts, kb)
    assert out is not corners


def test_nan_t_kick_keeps_row_with_nan_window():
    # A derived (events-only) corner has a NaN t_kick (spec section 9.4): the row is KEPT with NaN
    # window outputs and no target source (ADR-042), never dropped and never crashed.
    out = add_corner_windows(_corner(t_kick=float("nan")), _frames(), _no_contacts())
    assert len(out) == 1
    r = out.iloc[0]
    assert pd.isna(r["t_init"]) and pd.isna(r["t_target"]) and pd.isna(r["t_outcome"])
    assert pd.isna(r["target_snapshot_source"])


# ---- spec section 9 rules: gated provenance, contested cells, the geographic rule, event-coded passthrough ----


class _OnlyCell(GateTableStub):
    """A gate table passing only ONE first-contact cell state."""

    def __init__(self, cell_state):
        self.cell_state = cell_state

    def get(self, key, default=None):
        from silly_kicks.tracking._restart_validation import GateEntry

        if key[2] == "first_contact" and key[3] == self.cell_state:
            return GateEntry("transfer", "idsse", 0.9, 0.82, 0.8, 50, 5, "pooled_taker_known")
        return default


def _pos_frames(rows):
    """frames rows: (frame_id, time, player, team, x, y)."""
    return pd.DataFrame(rows, columns=["frame_id", "time_seconds", "player_id", "team_id", "x", "y"]).assign(
        period_id=1
    )


def _contacts_m(rows):
    """contacts rows: (frame_id, time, player, margin)."""
    df = pd.DataFrame(rows, columns=["frame_id", "time_seconds", "player_id", "attribution_margin"])
    return df.assign(period_id=1, contact_source="velocity_break")


def _known_taker_corner():
    c = _corner()
    c["taker_player_id"] = pd.array(["pT"], dtype="object")
    c["taker_known"] = pd.array([True], dtype="boolean")
    return c


def test_live_winner_records_tier_gate_version_cell_resolution_and_margin():
    from silly_kicks.corners._windows import GATE_TABLE_VERSION
    from silly_kicks.tracking._restart_validation import DataProfile

    frames = _pos_frames([(5, 100.5, "pW", "tW", 95.0, 30.0)])
    out = add_corner_windows(
        _known_taker_corner(),
        frames,
        _contacts_m([(5, 100.5, "pW", 1.2)]),
        data_profile=DataProfile("skillcorner", "r1", "g1", "cs1"),
        gate_table=_OnlyCell("uncontested_taker_known"),
    )
    r = out.iloc[0]
    assert r["winner_source"] == "tf79_contact" and r["first_contact_evidence"] == "transfer"
    assert r["first_contact_gate_version"] == GATE_TABLE_VERSION
    assert r["first_contact_cell_resolution"] == "pooled_taker_known"
    assert r["first_contact_attribution_margin"] == 1.2


def test_contested_cell_from_the_margin_both_sides():
    from silly_kicks.tracking._restart_validation import DataProfile

    prof = DataProfile("skillcorner", "r1", "g1", "cs1")
    frames = _pos_frames([(5, 100.5, "pW", "tW", 95.0, 30.0)])
    for margin, state in ((0.75, "contested_taker_known"), (0.76, "uncontested_taker_known")):
        ok = add_corner_windows(
            _known_taker_corner(),
            frames,
            _contacts_m([(5, 100.5, "pW", margin)]),
            data_profile=prof,
            gate_table=_OnlyCell(state),
        )
        assert ok.iloc[0]["winner_source"] == "tf79_contact", (margin, state)
        other = "uncontested_taker_known" if state.startswith("contested") else "contested_taker_known"
        withheld = add_corner_windows(
            _known_taker_corner(),
            frames,
            _contacts_m([(5, 100.5, "pW", margin)]),
            data_profile=prof,
            gate_table=_OnlyCell(other),
        )
        assert withheld.iloc[0]["winner_source"] == "withheld"


def test_nan_margin_fails_closed_to_contested():
    from silly_kicks.corners._windows import first_contact_cell_state

    assert first_contact_cell_state(float("nan"), True) == "contested_taker_known"
    assert first_contact_cell_state(2.0, False) == "uncontested_taker_unknown"


def test_unknown_taker_window_starts_after_the_last_in_region_contact():
    # pA touches OUTSIDE the corner region first, then pB touches INSIDE it (within 10 m of the flag), then pC
    # outside. The geographic rule starts the window after pB's in-region touch -> pC is the first contact.
    frames = _pos_frames(
        [
            (5, 100.5, "pA", "tA", 90.0, 20.0),
            (6, 101.0, "pB", "tA", 100.0, 3.0),
            (7, 101.5, "pC", "tW", 92.0, 30.0),
        ]
    )
    contacts = _contacts_m([(5, 100.5, "pA", 1.0), (6, 101.0, "pB", 1.0), (7, 101.5, "pC", 1.0)])
    out = add_corner_windows(_corner(), frames, contacts, **passing())
    assert out.iloc[0]["winner_player_id"] == "pC"


def test_known_taker_takes_the_earliest_non_taker_out_of_region_contact():
    frames = _pos_frames([(5, 100.5, "pT", "tA", 90.0, 20.0), (6, 101.0, "pA", "tA", 90.0, 25.0)])
    contacts = _contacts_m([(5, 100.5, "pT", 1.0), (6, 101.0, "pA", 1.0)])
    out = add_corner_windows(_known_taker_corner(), frames, contacts, **passing())
    assert out.iloc[0]["winner_player_id"] == "pA"


def test_event_coded_first_touch_is_native_passthrough_not_gated():
    c = _corner(source="spadl_corner")
    c["event_first_touch_time"] = [101.2]
    c["event_first_touch_player_id"] = pd.array(["pE"], dtype="object")
    c["event_first_touch_team_id"] = pd.array(["tW"], dtype="object")
    out = add_corner_windows(c, _frames(), _no_contacts())  # no data_profile: the gate is never consulted
    r = out.iloc[0]
    assert r["winner_source"] == "event_coded" and r["first_contact_evidence"] == "reference"
    assert (r["winner_player_id"], r["winner_team_id"], r["first_contact_time"]) == ("pE", "tW", 101.2)


def test_an_na_taker_known_is_an_unknown_taker_never_a_crash():
    # CI (C2 push): a corner whose taker_known is <NA> (e.g. a synthesized SB360 anchor) crashed `bool(<NA>)`; it is
    # an UNKNOWN taker -> the geographic rule and the taker_unknown cell (fail closed).
    c = _corner()
    c["taker_player_id"] = pd.array([pd.NA], dtype="object")
    c["taker_known"] = pd.array([pd.NA], dtype="boolean")
    out = add_corner_windows(c, _pos_frames([(5, 100.5, "pW", "tW", 95.0, 30.0)]), _contacts_m([(5, 100.5, "pW", 1.2)]))
    assert len(out) == 1 and out.iloc[0]["winner_source"] == "withheld"
