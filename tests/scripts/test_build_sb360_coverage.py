"""Unit tests for the SB360 real-data coverage driver.

No ``__init__.py`` in this directory -- it would shadow the ``scripts`` namespace package.
Everything here runs offline; the network-gated smoke is marked ``e2e``.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib

import pytest

import scripts.build_sb360_coverage as mod


def test_competition_ids_are_resolved_and_name_asserted():
    """Prose verification does not survive an upstream renumber."""
    catalogue = [
        {
            "competition_id": 44,
            "season_id": 107,
            "competition_name": "Major League Soccer",
            "season_name": "2023",
            "competition_gender": "male",
        }
    ]
    got = mod.resolve_competition(44, 107, catalogue=catalogue, expect_name="Major League Soccer")
    assert got["competition_gender"] == "male"

    with pytest.raises(mod.CompetitionMismatchError, match=r"expected"):
        mod.resolve_competition(44, 107, catalogue=catalogue, expect_name="Bundesliga")


def test_missing_competition_raises_rather_than_sampling_silently():
    with pytest.raises(mod.CompetitionMismatchError, match=r"not found"):
        mod.resolve_competition(999, 1, catalogue=[], expect_name="x")


def test_observed_pitch_fraction_is_normalised_by_the_STATSBOMB_pitch():
    """`visible_area` arrives in StatsBomb's 120x80 frame, NOT SPADL's 105x68.

    Dividing by 105*68 gives ~1.34 for a fully-visible frame -- a "fraction" above 1, reported
    to a club.
    """
    full = [0.0, 0.0, 120.0, 0.0, 120.0, 80.0, 0.0, 80.0]
    assert mod.observed_pitch_fraction(full) == pytest.approx(1.0)

    half = [0.0, 0.0, 60.0, 0.0, 60.0, 80.0, 0.0, 80.0]
    assert mod.observed_pitch_fraction(half) == pytest.approx(0.5)

    # ADR-055: NaN, not 0.0. "Nothing was published" and "the camera saw none of the pitch"
    # are different findings, and averaging the second in for every event lacking a 360 record
    # is what the n_with_polygon denominator now prevents.
    assert math.isnan(mod.observed_pitch_fraction([]))
    assert math.isnan(mod.observed_pitch_fraction([1.0, 2.0]))
    # An ODD-length flat list is malformed provider data. It must report unusable, not crash:
    # this function runs per-event across a corpus, so a raise would kill a multi-hour pass.
    # (The retired `visible_fraction` raised IndexError here.)
    assert math.isnan(mod.observed_pitch_fraction([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]))


def test_defending_keeper_is_keeper_AND_NOT_teammate():
    """`keeper` alone answers 'a keeper is visible', which is a different question.

    Freeze-frame flags are relative to the ACTOR, so the DEFENDING keeper -- the one the spec
    asks about -- is the keeper who is not a teammate.
    """
    players = [
        {"keeper": True, "teammate": True},  # the attacking side's own keeper
        {"keeper": False, "teammate": False},
        {"keeper": False, "teammate": True},
    ]
    assert mod.defending_gk_visible(players) is False

    players.append({"keeper": True, "teammate": False})
    assert mod.defending_gk_visible(players) is True


def test_acting_side_keeper_is_the_relevant_one_for_distribution_and_saves():
    """WHICH keeper matters depends on the action, and using only one reads 0% by construction.

    On a goal kick or a save the keeper IS the actor, so `keeper AND NOT teammate` excludes them
    however good the coverage is. Measured on MLS 2023 match 3877060 before the fix: `goalkick`
    and `keeper_save` both read exactly 0.000 -- which would have told a club its goal-kick
    coverage was nil. After: both read 1.000 on the acting-side rate.
    """
    acting_keeper = [{"keeper": True, "teammate": True}, {"keeper": False, "teammate": False}]
    assert mod.acting_side_gk_visible(acting_keeper) is True
    assert mod.defending_gk_visible(acting_keeper) is False, (
        "the defending rate MUST be False here -- that is the definitional artefact this pair "
        "of metrics exists to separate, not a bug to paper over"
    )

    opposing_keeper = [{"keeper": True, "teammate": False}]
    assert mod.defending_gk_visible(opposing_keeper) is True
    assert mod.acting_side_gk_visible(opposing_keeper) is False


def test_gk_domain_types_are_SPADL_names_not_statsbomb_ones():
    """The constant is matched against REAL SPADL actions from the converter.

    StatsBomb has no `Cross` type (it is `pass_cross == True`), no `Goal Kick` type (it is
    `pass_type == "Goal Kick"`), and its keeper type is `"Goal Keeper"`. Matching SPADL names
    against StatsBomb's taxonomy would silently reduce `is_gk_domain` to "pass or shot".
    """
    import silly_kicks.spadl as spadl

    valid = set(spadl.actiontypes_df()["type_name"])
    unknown = set(mod.GK_DOMAIN_TYPES) - valid
    assert not unknown, f"GK_DOMAIN_TYPES contains non-SPADL names: {sorted(unknown)}"
    assert "goalkick" in mod.GK_DOMAIN_TYPES, "goal kicks are the spec's named GK-domain event"


def test_partition_naming_no_ids_for_a_cell_drops_it():
    """An empty list and an absent key are BOTH falsy; conflating them makes a worker load the
    entire unsliced manifest (ADR-052's measured defect)."""
    assert mod._ids_for_cell({"44:107": []}, 44, 107) == (True, None)
    assert mod._ids_for_cell({"44:107": [1, 2]}, 44, 107) == (False, [1, 2])
    assert mod._ids_for_cell({"72:107": [1]}, 44, 107) == (True, None)
    # Unpartitioned caller: do NOT drop, take the default slice.
    assert mod._ids_for_cell(None, 44, 107) == (False, None)


def test_default_shard_root_is_covered_by_the_anchored_gitignore_glob():
    """An un-ignored shard root is a provenance defect, not untidiness.

    `.gitignore:90` is `/*_shards/` -- ANCHORED to the repo root on purpose, so an unanchored
    glob cannot silence intentionally-tracked paths at depth. A nested shard dir therefore is
    NOT ignored, dirties the tree, and makes the NEXT artifact-writing run refuse on its
    predecessor's own scratch. The documented operator response is `--allow-dirty`, which
    stamps `run_tree_dirty: true` onto a run whose CODE was clean -- laundering the fact
    ADR-052 wired the gate to preserve.

    Asserted structurally rather than by shelling to git, so it holds in a bare checkout.
    """
    root = pathlib.PurePosixPath(mod.DEFAULT_SHARD_ROOT)
    assert len(root.parts) == 1, (
        f"default shard root {mod.DEFAULT_SHARD_ROOT!r} is nested; the anchored /*_shards/ glob "
        f"only covers TOP-LEVEL directories"
    )
    assert root.name.endswith("_shards"), (
        f"default shard root {mod.DEFAULT_SHARD_ROOT!r} does not end in '_shards', so the glob does not match it"
    )


def test_driver_offers_allow_dirty_and_calls_require_clean_tree_from_main():
    """ADR-037. Enforcement belongs in main(), not in the work function."""
    src = inspect.getsource(mod)
    tree = ast.parse(src)
    main_fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "main")
    called = {n.func.id for n in ast.walk(main_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "require_clean_tree" in called
    assert "--allow-dirty" in src


def test_driver_never_shells_out_to_rev_parse():
    """Matched on CALLS, so prose describing the defect is not mistaken for committing it."""
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            flat = ast.dump(node)
            assert "rev-parse" not in flat, "driver shells out to git rev-parse"


def test_statsbombpy_is_imported_lazily():
    """`--help` and these unit tests must not need the optional dependency."""
    tree = ast.parse(inspect.getsource(mod))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and node.col_offset == 0:
            names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
            # Match the PACKAGE, not the substring. `silly_kicks.providers.statsbomb` is the
            # in-repo parse port -- a first-party module with no optional dependency behind it --
            # and a "statsbomb" substring test flags it the moment it exists. Only `statsbombpy`
            # is the optional dep that `--help` must not require.
            roots = {n.split(".")[0] for n in names if n}
            assert "statsbombpy" not in roots, "statsbombpy is imported at module level; --help would require it"


@pytest.mark.e2e
def test_measure_match_against_real_open_360():
    """Deselected from the normal suite: network + slow. Self-skips without statsbombpy."""
    pytest.importorskip("statsbombpy")
    from statsbombpy import sb  # type: ignore[import-not-found]

    catalogue = mod._load_catalogue()
    row = mod.resolve_competition(44, 107, catalogue=catalogue, expect_name="Major League Soccer")
    assert row["competition_gender"] == "male"

    matches = {m["match_id"]: m for m in mod._values(sb.matches(competition_id=44, season_id=107, fmt="dict"))}
    match_id = sorted(matches)[0]
    home = matches[match_id]["home_team"]
    home_id = home.get("home_team_id", home.get("id")) if isinstance(home, dict) else home

    out = mod.measure_match((44, 107, match_id, home_id))
    assert len(out) > 0, "no freeze-frames returned -- 360 may be absent for this match"
    assert {
        "n_events",
        "n_defending_gk_visible",
        "defending_gk_visible_rate",
        "n_acting_side_gk_visible",
        "acting_side_gk_visible_rate",
    } <= set(out.columns)

    # BOTH keeper rates must be live, or one of them is a column of zeros nobody notices.
    gk = out[out["is_gk_domain"]]
    assert gk["defending_gk_visible_rate"].max() > 0.0, "no defending keeper seen on any GK action"
    assert gk["acting_side_gk_visible_rate"].max() > 0.0, "no acting-side keeper seen either"
    assert out["mean_visible_pitch_fraction"].between(0.0, 1.0).all()

    types = set(out["action_type"])
    assert types != {"unmapped"}, "event_uuid -> SPADL type join resolved nothing"
    assert "pass" in types, f"no SPADL passes -- converter output looks wrong: {sorted(types)}"
    # NON-VACUOUS GK check: `is_gk_domain.any()` passes on any match with a shot. `goalkick`
    # can ONLY appear if the converter ran, since StatsBomb encodes it as a pass sub-type.
    assert "goalkick" in types, f"no SPADL goalkick -- converter did not run: {sorted(types)}"
