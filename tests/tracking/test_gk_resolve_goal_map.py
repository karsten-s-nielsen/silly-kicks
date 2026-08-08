"""Goal-end map extraction (TF-48 prerequisite; spec section 5.1).

``resolve_defended_goals`` is THE single implementation of the rule. The ``defended_goal_x``
shim that ``_xshot_occurrence`` re-imported is gone -- the identity test that pinned it is
retired with it, because there is no second name left to keep in step.
"""

import pandas as pd
import pytest

from silly_kicks.tracking._gk_resolve import resolve_defended_goals


def _frames(gk_x_a=5.0, gk_x_b=100.0, with_gk=True):
    rows = []
    for pid, team, gk, x in [
        (1, "A", True, gk_x_a),
        (2, "A", False, 40.0),
        (3, "B", True, gk_x_b),
        (4, "B", False, 60.0),
    ]:
        if not with_gk and gk:
            continue
        rows.append(
            dict(
                game_id=1,
                period_id=1,
                frame_id=0,
                time_seconds=0.0,
                player_id=pid,
                team_id=team,
                is_ball=False,
                is_goalkeeper=gk,
                x=x,
                y=34.0,
                z=0.0,
            )
        )
    rows.append(
        dict(
            game_id=1,
            period_id=1,
            frame_id=0,
            time_seconds=0.0,
            player_id=None,
            team_id=None,
            is_ball=True,
            is_goalkeeper=False,
            x=50.0,
            y=34.0,
            z=0.0,
        )
    )
    return pd.DataFrame(rows)


def test_gk_based_resolution():
    gm = resolve_defended_goals(_frames())
    assert gm.get(1, 1, "A") == 0.0
    assert gm.get(1, 1, "B") == 105.0
    assert gm.n_guessed == 0


def test_outfield_fallback_when_no_gk_is_a_GUESS_not_a_resolution():
    """N1 coverage is retained -- but as ``guessed``, so consuming it is explicit."""
    gm = resolve_defended_goals(_frames(with_gk=False))
    assert gm.get(1, 1, "A") is None, "a guess must not answer a strict lookup"
    assert gm.get(1, 1, "A", allow_guess=True) == 0.0
    assert gm.get(1, 1, "B", allow_guess=True) == 105.0
    assert gm.n_resolved == 0 and gm.n_guessed == 2


def _frames_numeric_ids():
    """Same scene with INT ids, so the canonical-key contract is expressible.

    The string-id fixture above cannot show it: canonical_id('A') == 'A', so a raw-tuple lookup
    would accidentally hit and the witness below would pass for the wrong reason.
    """
    f = _frames()
    f["team_id"] = f["team_id"].map({"A": 1, "B": 2})
    return f


# ---------------------------------------------------------------------------
# The seam's own contract (ADR-055). Task 2 Step 2 of the plan.
# ---------------------------------------------------------------------------


def test_keys_are_canonical_STRINGS_and_lookups_accept_any_id_dtype():
    """The mappings are keyed by ``canonical_id``, which returns STRINGS.

    This is the fact the next test's failure mode falls out of, so it is asserted first and
    directly rather than inferred from a lookup succeeding.
    """
    gm = resolve_defended_goals(_frames_numeric_ids())
    assert all(isinstance(k[2], str) for k in gm.resolved), dict(gm.resolved)
    assert gm.get(1, 1, 1) == gm.get("1", "1", "1") == gm.get(1.0, 1.0, 1.0) == 0.0


def test_a_RAW_TUPLE_lookup_against_the_MAPPING_misses():
    """Witness for the rule CLAUDE.md states: never hold the mappings as a plain dict.

    Not hypothetical -- it shipped exactly this way in
    ``scripts/validate_shot_goalmouth_sb.py``, which scanned ``goal_map.items()`` comparing
    ``(k[0], k[1]) == key`` with RAW ids and would have returned NaN for every row, silently.
    The accessor canonicalizes on the way in; the raw tuple does not.
    """
    gm = resolve_defended_goals(_frames_numeric_ids())
    assert dict(gm.resolved).get((1, 1, 1)) is None, "int tuple must MISS -- that is the hazard"
    assert gm.get(1, 1, 1) == 0.0, "the accessor must hit"


def test_an_NA_team_is_UNRESOLVED_and_in_NEITHER_mapping():
    """The ladder's third state. ``canonical_id(pd.NA) is pd.NA`` -- never ``None``.

    Testing ``is None`` here would assert nothing, because a missing key also reads as None; the
    key must be absent from BOTH mappings and present in ``unresolved``.
    """
    f = _frames()
    f.loc[f["team_id"] == "B", "team_id"] = pd.NA
    gm = resolve_defended_goals(f)

    na_keys = [k for k in gm.unresolved if k[2] is pd.NA]
    assert na_keys, f"no NA-team key landed in unresolved: {sorted(gm.unresolved)}"
    for k in na_keys:
        assert k not in gm.resolved and k not in gm.guessed, f"{k} is in a mapping as well"
    assert gm.get(1, 1, pd.NA, allow_guess=True) is None


def test_an_all_NaN_x_group_is_unresolved_through_BOTH_mappings():
    """Neither rung of the ladder can answer, so the answer is "no answer".

    The pre-seam code returned 105.0 here, because ``nan < 52.5`` is False -- a confident wrong
    end. Asserting through ``allow_guess=True`` is the point: the guess rung must not rescue it.
    """
    f = _frames()
    f.loc[f["team_id"] == "A", "x"] = float("nan")
    gm = resolve_defended_goals(f)
    assert gm.get(1, 1, "A") is None
    assert gm.get(1, 1, "A", allow_guess=True) is None, "the outfield rung is also all-NaN here"
    assert any(k[2] == "A" for k in gm.unresolved)


def test_attacked_goal_reads_the_OPPONENTS_entry():
    """Not ``105.0 - get(...)``. The arithmetic identity is a second implementation of the rule."""
    gm = resolve_defended_goals(_frames())
    assert gm.get(1, 1, "A") == 0.0
    assert gm.attacked_goal(1, 1, "A") == 105.0
    assert gm.attacked_goal(1, 1, "B") == 0.0


def test_attacked_goal_REFUSES_a_degenerate_map_where_a_count_check_would_pass():
    """Both teams mapped to the SAME end: exactly one opponent exists, so a count-only guard
    passes and the answer would say a team attacks the goal it defends.

    This is the guard the SB360 ``gk_absent`` verdict rests on, so it is asserted directly rather
    than through a consumer. Non-vacuity: the same fixture with the ends DISTINCT resolves fine,
    which is the line above.
    """
    f = _frames(gk_x_a=100.0, gk_x_b=101.0)  # both keepers at the high end
    gm = resolve_defended_goals(f)
    assert gm.get(1, 1, "A") == gm.get(1, 1, "B") == 105.0, "fixture is not degenerate"
    assert gm.attacked_goal(1, 1, "A") is None
    assert gm.attacked_goal(1, 1, "B") is None


def test_attacked_goal_is_None_when_the_period_has_no_single_opponent():
    """One team in the period -> no opponent entry to read, so no answer."""
    f = _frames()
    f = f[(f["team_id"] == "A") | f["is_ball"].astype(bool)]
    gm = resolve_defended_goals(f)
    assert gm.get(1, 1, "A") == 0.0
    assert gm.attacked_goal(1, 1, "A") is None


def test_ends_in_period_returns_a_COPY_keyed_canonically():
    gm = resolve_defended_goals(_frames())
    ends = gm.ends_in_period(1, 1)
    assert ends == {"A": 0.0, "B": 105.0}
    ends["A"] = 999.0
    assert gm.get(1, 1, "A") == 0.0, "ends_in_period handed out its internal index"


def test_the_ball_row_is_excluded_by_is_ball_not_by_a_team_name():
    """A provider whose ball row carries a team label must not become a third 'team'."""
    f = _frames()
    f.loc[f["is_ball"].astype(bool), "team_id"] = "A"
    gm = resolve_defended_goals(f)
    assert set(k[2] for k in gm.resolved) == {"A", "B"}
    assert gm.get(1, 1, "A") == 0.0, "the ball at x=50 must not drag team A's mean"


def test_a_STRING_is_ball_column_still_excludes_the_ball():
    """`_truthy_bool`, not `.astype(bool)` -- `pd.Series(["false"]).astype(bool)` is True."""
    f = _frames()
    f["is_ball"] = f["is_ball"].map({True: "true", False: "false"})
    f["is_goalkeeper"] = f["is_goalkeeper"].map({True: "true", False: "false"})
    gm = resolve_defended_goals(f)
    assert gm.get(1, 1, "A") == 0.0
    assert gm.get(1, 1, "B") == 105.0
    assert gm.n_resolved == 2, "a string is_ball collapsed the player set"


def test_the_map_is_FROZEN_and_its_mappings_are_not_writable():
    """A consumer that mutates a shared map would silently re-key every other consumer."""
    import dataclasses

    gm = resolve_defended_goals(_frames())
    with pytest.raises(dataclasses.FrozenInstanceError):
        gm.resolved = {}  # type: ignore[misc]
    with pytest.raises(TypeError):
        gm.resolved[("1", "1", "A")] = 999.0  # type: ignore[index]


def test_periods_resolve_INDEPENDENTLY():
    """The period term the ten forks structurally lacked -- asserted on the seam itself."""
    f1 = _frames()
    f2 = _frames(gk_x_a=100.0, gk_x_b=5.0)  # ends swapped after half time
    f2["period_id"] = 2
    gm = resolve_defended_goals(pd.concat([f1, f2], ignore_index=True))
    assert gm.get(1, 1, "A") == 0.0
    assert gm.get(1, 2, "A") == 105.0, "period 2 answered with period 1's end"
    assert gm.attacked_goal(1, 1, "A") == 105.0
    assert gm.attacked_goal(1, 2, "A") == 0.0
