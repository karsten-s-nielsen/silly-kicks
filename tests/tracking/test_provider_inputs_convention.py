"""The shared fixture builder must emit ACTION-LTR actions (ADR-028, spec D4/D5).

``synthesize_actions`` stamped ``start_x``/``start_y`` from the actor's RAW frame position. Frames
are frame-LTR (home attacks +x), so away-team actions carried frame-convention coordinates while
being labelled SPADL actions -- which are action-LTR (the ACTING team attacks +x). Measured before
the fix: 9/10 actions equalled the raw frame position exactly and 0/10 equalled the point
reflection.

Two consequences, and the second is why this file exists: an ADR-028 passer defect was
UNEXPRESSIBLE on this fixture (raw ``start_x`` was accidentally correct), and a CORRECT
implementation was wrong on it for away rows.
"""

from __future__ import annotations

import pandas as pd
import pytest

from silly_kicks.tracking._action_orientation import FIELD_LENGTH, FIELD_WIDTH
from tests.tracking._provider_inputs import load_provider_frames, synthesize_actions

PROVIDERS = ["sportec", "metrica", "skillcorner", "gradientsports"]


def _direction_map(frames):
    """(period_id, str(team_id)) -> attacking direction, from non-ball rows."""
    players = frames[~frames["is_ball"].astype(bool)].copy()
    players = players[players["team_attacking_direction"].notna()]
    players["_team"] = players["team_id"].astype(str)
    out = {}
    for (period, team), grp in players.groupby(["period_id", "_team"]):
        out[(period, str(team))] = grp["team_attacking_direction"].iloc[0]
    return out


@pytest.mark.parametrize("provider", PROVIDERS)
def test_synthesized_actions_are_action_ltr(provider):
    """An RTL-attacking team's action coords must be the POINT REFLECTION of its frame position.

    Non-vacuity: the reflection assertion is meaningless unless at least one RTL action exists, so
    that is asserted first. Before D5 lands, gradientsports has none -- both its teams are labelled
    "ltr" -- which is why D5 is a prerequisite rather than a nicety.
    """
    frames = load_provider_frames(provider)
    actions = synthesize_actions(frames)
    directions = _direction_map(frames)

    rtl_rows = [
        row for _, row in actions.iterrows() if directions.get((row["period_id"], str(row["team_id"]))) == "rtl"
    ]
    assert rtl_rows, f"{provider}: no RTL action in the fixture -- the check would be vacuous"

    checked = 0
    for row in rtl_rows:
        own = frames[
            (frames["period_id"] == row["period_id"]) & (frames["player_id"].astype(str) == str(row["player_id"]))
        ]
        if own.empty:
            continue
        # Match against ANY of the actor's frame rows in the period, not the nearest-in-time one.
        # The keeper_save action's clock is deliberately shifted by the builder
        # (`save_time = min(gk_row.time, last.time - 2.0)`), so nearest-in-time resolves to a
        # DIFFERENT row than the one its coordinates came from -- which would make this assertion
        # fail on correct code for every provider whose only away action is that keeper_save.
        reflected_x = FIELD_LENGTH - own["x"].to_numpy(dtype=float)
        reflected_y = FIELD_WIDTH - own["y"].to_numpy(dtype=float)
        dist = ((reflected_x - float(row["start_x"])) ** 2 + (reflected_y - float(row["start_y"])) ** 2) ** 0.5
        assert dist.min() == pytest.approx(0.0, abs=1e-6), (
            f"{provider} action {row['action_id']}: start "
            f"({row['start_x']:.3f}, {row['start_y']:.3f}) matches no point reflection of the "
            f"actor's frame positions (closest {dist.min():.3f} m)"
        )
        checked += 1
    assert checked, f"{provider}: no RTL action resolved to a frame row -- vacuous"


def test_balance_teams_defaults_off_and_is_byte_identical():
    """The default MUST NOT move any existing baseline (spec D4)."""
    from pandas.testing import assert_frame_equal

    frames = load_provider_frames("sportec")
    assert_frame_equal(synthesize_actions(frames), synthesize_actions(frames, balance_teams=False))


def test_balance_teams_true_produces_both_teams():
    """Opt-in gives a usable away population; the 9:1 default cannot gate orientation.

    The default picks the first-listed player per frame, which is team-blind and lands ~9:1 on
    whichever team sorts first -- an artifact of frame row order, not of the data.
    """
    frames = load_provider_frames("sportec")
    balanced = synthesize_actions(frames, balance_teams=True)
    counts = balanced["team_id"].astype(str).value_counts()
    assert len(counts) >= 2, f"expected both teams, got {counts.to_dict()}"
    assert counts.min() >= 3, f"minority team too small to gate on: {counts.to_dict()}"


def test_metrica_baseline_matches_the_corrected_fixture():
    """The convention fix (D4) must move metrica ONLY, and in the right DIRECTION.

    Pre-fix, metrica action 1 emitted ``nearest_defender_distance`` 1.029355 m against a true
    18.328738 m: the anchor was in frame convention while the defenders had been reprojected to
    action-LTR. The wrong value looked MORE plausible than the right one -- tight marking rather
    than an absurdity -- which is why it survived. So this pins the magnitude, not just that
    something changed.

    Measured blast radius on the full tracking+invariants suite: metrica alone (1 failed / 3176
    passed); sportec, skillcorner and pff are byte-unchanged, because their single away action is
    the keeper_save, whose context output is already all-NaN.
    """
    import pathlib

    import pandas as pd

    from silly_kicks.tracking.features import nearest_defender_distance

    # __file__-anchored, matching the repo idiom. A CWD-relative path works only when pytest runs
    # from the repo root and breaks silently otherwise.
    repo = pathlib.Path(__file__).resolve().parents[2]
    frames = load_provider_frames("metrica")
    actions = synthesize_actions(frames)
    live = nearest_defender_distance(actions, frames)
    committed = pd.read_parquet(
        repo / "tests" / "datasets" / "tracking" / "action_context_slim" / "metrica_expected.parquet"
    ).set_index("action_id")["nearest_defender_distance"]

    merged = pd.DataFrame({"live": live.to_numpy()}, index=actions["action_id"]).join(committed)
    both = merged.dropna()
    assert len(both) >= 8, f"too few comparable rows: {len(both)}"
    assert (both["live"] - both["nearest_defender_distance"]).abs().max() < 1e-9, "baseline is stale"
    assert both["live"].max() > 5.0, (
        "post-fix distances still look like the pre-fix collapse (max was 1.17 m); the reflection may not be applied"
    )


def test_gradientsports_labels_both_directions():
    """`_provider_inputs.py:71` hardcoded "ltr" for BOTH teams, so no GS fixture could exercise
    any orientation path at all. Derived from geometry: team 100's keeper sits at x~20.5 and
    team 200's at x~60.5 in both periods (outfield medians 32.5 / 72.5), so 100 attacks ltr and
    200 attacks rtl. This synthetic fixture does NOT swap ends at half-time.
    """
    frames = load_provider_frames("gradientsports")
    players = frames[~frames["is_ball"].astype(bool)].copy()
    players["_team"] = players["team_id"].astype(str)
    by_team = {
        team: sorted(grp["team_attacking_direction"].dropna().unique()) for team, grp in players.groupby("_team")
    }
    assert by_team == {"100": ["ltr"], "200": ["rtl"]}, by_team


#: Providers whose slim slice never has BOTH teams' keepers in the same period, so the check below
#: has nothing to compare. Note the distinction, which a first draft of this gate got wrong: these
#: slices DO carry keeper rows (skillcorner 252 of them), just one-sided -- SkillCorner detects a
#: keeper in ~19.6% of frames (ADR-038), so a short slice routinely sees one team's and not the
#: other's. "one-sided keeper coverage" and "keeper coverage that agrees" are different states and
#: must not report the same way, which is why the test SKIPS with the measured counts rather than
#: passing.
_ONE_SIDED_KEEPERS = frozenset({"metrica", "skillcorner"})


@pytest.mark.parametrize(
    "provider",
    [
        pytest.param(
            "sportec",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "MEASURED DEFECT in the committed sportec_slim.parquet: its direction LABELS "
                    "and its keeper POSITIONS are exact opposites, in both periods. "
                    "team_attacking_direction says DFL-CLU-00000P attacks +x (so it defends x=0), "
                    "while that team's keeper mean x is 98.1 (p1) and 77.0 (p2). Confirmed "
                    "independently by orient_frames_to_ltr_by_geometry (ADR-035), which MIRRORS "
                    "this slice's positions. Strict xfail so repairing the slice is forced to "
                    "delete this marker rather than leaving a stale exemption behind. Repair also "
                    "moves sportec_expected.parquet and the lakehouse-parity goldens, so it is a "
                    "separate change from ADR-055."
                ),
            ),
        ),
        "gradientsports",
        *sorted(_ONE_SIDED_KEEPERS),
    ],
)
def test_direction_labels_agree_with_keeper_geometry(provider):
    """A team labelled ``ltr`` attacks +x, so it DEFENDS x=0 and its keeper stands at LOW x.

    This is not a restatement of ``test_synthesized_actions_are_action_ltr``: that one checks the
    ACTION builder against the labels and would pass on a slice whose positions are mirrored,
    because both sides read the same labels. This one checks the labels against the POSITIONS,
    which is the only pairing that can see a slice reflected without its labels being swapped
    (ADR-045: a 180-degree point reflection swaps direction labels -- omitting that is the
    recorded defect class).

    It became load-bearing with ADR-055: ``resolve_defended_goals`` derives the goal end from
    keeper geometry, so on a slice where the two disagree, a position-derived map and a
    label-derived one are opposites and any consumer mixing them scores a mirrored scene.
    """
    frames = load_provider_frames(provider)
    players = frames[~frames["is_ball"].astype(bool)]
    keepers = players[players["is_goalkeeper"].astype(bool)]

    checked = 0
    one_sided = 0
    for period, grp in players.groupby("period_id"):
        labels = {
            str(team): (g["team_attacking_direction"].dropna().unique().tolist() or [None])[0]
            for team, g in grp.groupby(grp["team_id"].astype(str))
        }
        ltr = [t for t, d in labels.items() if d == "ltr"]
        rtl = [t for t, d in labels.items() if d == "rtl"]
        if len(ltr) != 1 or len(rtl) != 1:
            continue
        gk = keepers[keepers["period_id"] == period]
        x_ltr = gk[gk["team_id"].astype(str) == ltr[0]]["x"].mean()
        x_rtl = gk[gk["team_id"].astype(str) == rtl[0]]["x"].mean()
        if pd.isna(x_ltr) or pd.isna(x_rtl):
            one_sided += 1
            continue
        checked += 1
        assert x_ltr < x_rtl, (
            f"{provider} period {period}: the team labelled 'ltr' ({ltr[0]}) attacks +x, so its "
            f"keeper should sit at LOWER x than the 'rtl' team's -- measured {x_ltr:.1f} vs "
            f"{x_rtl:.1f}. Labels and positions disagree: this slice is reflected relative to "
            "its own direction labels."
        )

    if checked == 0:
        assert provider in _ONE_SIDED_KEEPERS, (
            f"{provider}: no period offered BOTH teams' keepers ({one_sided} one-sided), so this "
            "gate compared nothing. That is expected only for the providers listed in "
            f"_ONE_SIDED_KEEPERS ({sorted(_ONE_SIDED_KEEPERS)}) -- for any other provider it "
            "means keeper coverage regressed and the gate has gone vacuous."
        )
        pytest.skip(
            f"{provider}: {one_sided} period(s) with one-sided keeper coverage, 0 comparable "
            f"({len(keepers)} keeper rows present). Geometry-vs-label check inexpressible here."
        )
