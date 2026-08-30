"""e2e METHOD gate for rest defense (TF-60, ADR-080, spec §16.2).

Asserts the METHOD (non-empty output, drop-conservation reconciles, GK metrics populate, FOV
companions degrade honestly) -- NEVER metric values -- so it stays compatible with the
reported-not-gated applied-results convention. Two real-data legs, each fixture/network-gated:

  * a real linked TRACKING match (committed lakehouse-derived slim; native ``is_goalkeeper`` +
    ``link_actions_to_frames`` coverage), and
  * a real SB360 match (statsbombpy open data; ``resolve_keeper_identities(identity="roster")`` +
    ``apply_keeper_identities_to_frames`` bridge, then FOV-partial companions on genuinely cropped
    advanced-ball frames).

Synthetic fixtures cannot exercise these integration seams (real linking, roster bridging, real FOV
crop), which is why this is an ``@e2e`` full-match backstop, deselected in the normal suite.
"""

from __future__ import annotations

import pytest

from silly_kicks.restdefense import RD_LAYER1_COLUMNS, RestDefenseParams
from silly_kicks.restdefense._compute import compute_rest_defense, summarize_rest_defense

pytestmark = pytest.mark.e2e

# A permissive committed-forward gate so the METHOD assertions always have scored rows (this gate
# asserts the pipeline runs, not that the ball was advanced).
_PARAMS = RestDefenseParams(min_ball_advance_m=0.0)
_GK_COLS = ("rd_gk_line_height", "rd_gk_to_line_distance")


def _assert_method(samples, report):
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    assert len(resolved) > 0, "no resolved rest-defense samples on a real match"
    assert report.n_frames_scored + sum(report.drop_reasons.values()) == report.n_frames_in
    for c in RD_LAYER1_COLUMNS:
        assert c in samples.columns
    for c in _GK_COLS:
        assert resolved[c].notna().any(), f"{c} all-NaN on a real match -- GK identity did not resolve"


def test_e2e_linked_tracking_method():
    """Real linked continuous-tracking match: native GK, real link coverage, both rollup grains."""
    try:
        from tests.tracking._provider_inputs import (
            N_ACTIONS_PER_PROVIDER,
            load_provider_frames,
            synthesize_actions,
        )

        frames = load_provider_frames("sportec")
    except Exception as exc:  # slim not committed / import unavailable
        pytest.skip(f"lakehouse-derived tracking slim unavailable: {exc}")

    assert frames["is_goalkeeper"].astype("boolean").fillna(False).any(), "fixture has no native keeper rows"
    actions = synthesize_actions(frames, n_actions=N_ACTIONS_PER_PROVIDER)

    samples, report = compute_rest_defense(actions, frames, params=_PARAMS)
    _assert_method(samples, report)

    # Both rollup grains reduce without error and stay one-row-per-group.
    per_poss = summarize_rest_defense(samples, by="possession")
    per_match = summarize_rest_defense(samples, by="match")
    assert len(per_poss) >= 1 and len(per_match) >= 1


def test_e2e_sb360_open_method():
    """Real SB360 match: roster keeper bridge + FOV-partial companions on a cropped advanced frame."""
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    from scripts._sb_raw import flatten_events
    from silly_kicks.providers.statsbomb import shape_snapshots
    from silly_kicks.spadl.statsbomb import convert_to_actions
    from silly_kicks.tracking import (
        apply_keeper_identities_to_frames,
        resolve_keeper_identities,
        snapshot_to_tracking_frames,
    )

    # Women's World Cup 2023, a full open-360 competition (72:107).
    try:
        matches = sb.matches(competition_id=72, season_id=107, fmt="dict")
        match_id = next(iter(matches))
        home = int(matches[match_id]["home_team"]["home_team_id"])
        events = list(sb.events(match_id=match_id, fmt="dict").values())
        frames_raw = list(sb.frames(match_id=match_id, fmt="dict"))
        lineups = sb.lineups(match_id=match_id, fmt="dict")
    except Exception as exc:  # network / availability
        pytest.skip(f"StatsBomb open-data unavailable: {exc}")

    flat = flatten_events(events, match_id)
    actions, _report = convert_to_actions(flat, home_team_id=home)
    snapshots, visible_area, _join = shape_snapshots(frames_raw, actions)
    frames, _links = snapshot_to_tracking_frames(snapshots, actions)

    # Roster keeper bridge (ADR-078): the anonymous freeze-frames need the resolved keeper id.
    roster = {}
    for _team_name, lu in lineups.items():
        team_id = int(lu["team_id"]) if "team_id" in lu else None
        for player in lu.get("lineup", []):
            positions = player.get("positions") or []
            if any("Goalkeeper" in (p.get("position") or "") for p in positions) and team_id is not None:
                roster[team_id] = int(player["player_id"])
                break
    keeper_map, _kr = resolve_keeper_identities(actions, frames, identity="roster", roster=roster)
    frames = apply_keeper_identities_to_frames(frames, keeper_map)

    samples, report = compute_rest_defense(actions, frames, visible_area=visible_area, params=_PARAMS)
    _assert_method(samples, report)

    # FOV companions degrade honestly on a real cropped match: at least one action's count region is
    # partially observed (a fraction strictly inside the band) -- the SB360 coverage signal.
    resolved = samples[samples["rd_geometry_source"] == "resolved"]
    frac = resolved["rd_num_superiority_observed_fraction"].dropna()
    assert ((frac > 0.0) & (frac < 1.0)).any(), (
        "no partially-observed rest-defense count on a full SB360 match -- the FOV companion is a rubber stamp"
    )
