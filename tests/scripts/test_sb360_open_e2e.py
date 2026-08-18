"""Full-chain SB360 open-data E2E (owner/network-gated).

The committed 6-frame golden slice already exercises the wiring in CI (test_loader_statsbomb.py,
test_validate_sb360_licensed_corpus.py). This is the FULL-MATCH backstop -- it pulls a real open-360
match via statsbombpy and runs the exact production chain
(``flatten_events`` -> ``convert_to_actions`` -> ``shape_snapshots`` -> ``snapshot_to_tracking_frames``
-> ``add_action_context(..., visible_area=...)``), asserting the companions behave honestly on a full
match's worth of partial visibility. Licensed matches can never be in CI, so this open-data run is the
closest committed proxy for the licensed path. Marked e2e: deselected in the normal suite (network).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.e2e


def test_full_open_360_chain_companions_are_honest():
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    from scripts._sb_raw import flatten_events
    from silly_kicks.providers.statsbomb import shape_snapshots
    from silly_kicks.spadl.statsbomb import convert_to_actions
    from silly_kicks.tracking import REGION_OBSERVATION_SOURCE_VALUES, snapshot_to_tracking_frames
    from silly_kicks.tracking.features import add_action_context

    # Women's World Cup 2023, a full open-360 competition (72:107).
    try:
        matches = sb.matches(competition_id=72, season_id=107, fmt="dict")
    except Exception as exc:  # network / availability
        pytest.skip(f"StatsBomb open-data unavailable: {exc}")
    match_id = next(iter(matches))
    home = int(matches[match_id]["home_team"]["home_team_id"])

    # statsbombpy returns events as a dict keyed by id, but 360 frames as a LIST of records.
    events = list(sb.events(match_id=match_id, fmt="dict").values())
    frames_raw = list(sb.frames(match_id=match_id, fmt="dict"))

    flat = flatten_events(events, match_id)
    actions, _report = convert_to_actions(flat, home_team_id=home)
    snapshots, visible_area, _join = shape_snapshots(frames_raw, actions)
    frames, _links = snapshot_to_tracking_frames(snapshots, actions)

    base = add_action_context(actions, frames)
    withva = add_action_context(actions, frames, visible_area=visible_area)

    # Primary columns are byte-identical with and without visible_area (the additive guarantee).
    for c in ["nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"]:
        assert base[c].equals(withva[c]), f"{c} changed when visible_area was supplied"

    allowed_sources = set(REGION_OBSERVATION_SOURCE_VALUES) | {"unlinked"}
    seen_sources: set[str] = set()
    for feat in ["nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"]:
        frac = withva[f"{feat}_observed_fraction"]
        src = withva[f"{feat}_observed_source"]
        assert set(src) <= allowed_sources, f"{feat} source outside the closed set: {set(src) - allowed_sources}"
        # fractions are in [0, 1] or NaN -- never fabricated outside the band
        finite = frac.dropna()
        assert ((finite >= 0.0) & (finite <= 1.0)).all(), f"{feat} fraction out of [0,1]"
        seen_sources |= set(src)

    # At least one partially-observed action on a real match (a fraction strictly inside the band).
    any_partial = any(
        (
            (withva[f"{feat}_observed_fraction"].dropna() > 0.0) & (withva[f"{feat}_observed_fraction"].dropna() < 1.0)
        ).any()
        for feat in ["nearest_defender_distance", "receiver_zone_density", "defenders_in_triangle_to_goal"]
    )
    assert any_partial, "no partially-observed action on a full open match -- the coverage feature is a rubber stamp"

    # The honest-degradation vocabulary is genuinely exercised (goalkick coverage ~32.6% + per-frame
    # visible_area gaps make unlinked/no_polygon near-certain on a real match).
    assert seen_sources & {"no_polygon", "unlinked", "degenerate_region"}, (
        f"only 'observed' appeared on a real match -- degradation never exercised: {seen_sources}"
    )
