"""TF-19 sign-off package: the GKDV arm-values pass feeding the §6.1 ICC power leg."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scripts.build_gkdv_arm_values as mod  # bare import: tests/scripts/ has NO __init__.py


def _frames():
    """Two teams, one ball row, one frame."""
    rows = [
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 5, "is_ball": False, "x": 20.0},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 6, "is_ball": False, "x": 80.0},
        {"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": None, "is_ball": True, "x": 50.0},
        {"game_id": 1, "period_id": 2, "frame_id": 99, "team_id": 5, "is_ball": False, "x": 30.0},
    ]
    return pd.DataFrame(rows)


def test_frame_slice_selects_exactly_one_game_period_frame():
    got = mod._frame_slice(_frames(), 1, 1, 10)
    assert len(got) == 3
    assert set(got["frame_id"]) == {10}
    assert set(got["period_id"]) == {1}


def test_attacking_team_is_the_non_defending_outfield_team():
    sl = mod._frame_slice(_frames(), 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=6) == 5
    assert mod._attacking_team_id(sl, defending_team_id=5) == 6


def test_attacking_team_resolution_is_dtype_safe():
    """ADR-019: the frame `team_id` may be nullable Int64 while the provenance carries a plain
    int -- a raw `!=` would silently keep BOTH teams and pick the defender as the attacker."""
    f = _frames()
    f["team_id"] = f["team_id"].astype("Int64")
    sl = mod._frame_slice(f, 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=6) == 5


def test_attacking_team_is_None_when_only_the_defenders_are_present():
    """Returned rather than guessed: the caller skips the frame instead of scoring a wrong team."""
    only_def = pd.DataFrame([{"game_id": 1, "period_id": 1, "frame_id": 10, "team_id": 6, "is_ball": False, "x": 1.0}])
    assert mod._attacking_team_id(only_def, defending_team_id=6) is None


def test_the_ball_row_is_never_mistaken_for_a_team():
    """The ball carries a null team; including it would make it the 'attacker' on a one-team frame."""
    sl = mod._frame_slice(_frames(), 1, 1, 10)
    assert mod._attacking_team_id(sl, defending_team_id=5) != pytest.approx(float("nan"), nan_ok=True)
    assert mod._attacking_team_id(sl, defending_team_id=5) == 6


def test_threat_arm_is_refused_not_silently_defaulted(monkeypatch, capsys):
    """The threat arm needs a fitted ExpectedThreat and NONE can be loaded: the class exposes only
    fit/interpolator/rate (no save/load anywhere in the package) and `FrozenXt` wraps an
    already-fitted in-memory model. Defaulting to `xt=None` would have persisted structural zeros
    -- `compute_threat_pc` returned 0.0 for None before this cycle's guard -- and an ICC on a
    constant column is degenerate while looking like a measurement. So it refuses, loudly."""
    import sys

    for arm in ("threat", "both"):
        monkeypatch.setattr(sys, "argv", ["build_gkdv_arm_values.py", "--out", "x", "--arm", arm])
        with pytest.raises(SystemExit) as excinfo:
            mod.main()
        assert "fitted ExpectedThreat" in str(excinfo.value)


def test_expected_threat_really_has_no_loader():
    """Pins the FACT the refusal rests on. If serialization is ever added, this goes red and the
    refusal above should be revisited rather than left as folklore."""
    from silly_kicks.xthreat import ExpectedThreat

    assert not hasattr(ExpectedThreat, "load")
    assert not hasattr(ExpectedThreat, "save")


def test_only_the_DEFENDING_keeper_is_credited():
    """The serving seam writes a row for BOTH teams' keepers, but `build_ghost_frames` substitutes
    only the DEFENDING one. A naive pass-through therefore credits each frame's delta to two
    keepers, one of whom never moved.

    MEASURED on real GS data before this filter existed: 4448 rows from 2224 scored frames, both
    rows per frame carrying an IDENTICAL arm_value under different keeper_keys. That is
    keeper-INDEPENDENT noise, and it compresses between-keeper variance toward zero -- the same
    mechanism that made xT-GK v2 read "keeper-flat" on fabricated origins (ADR-036/PR-S113).
    """
    from silly_kicks.id_compat import ids_equal

    # The two leading DROPPED rows are load-bearing: filtering them leaves a NON-CONTIGUOUS index,
    # which is what real provenance looks like. An all-scored fixture keeps the index at 0..n-1 and
    # hides the mask-alignment crash this reproduces (measured: IndexingError on the real run).
    provenance = pd.DataFrame(
        {
            "game_id": [1, 1, 1, 1, 1, 1],
            "period_id": [1, 1, 1, 1, 1, 1],
            "frame_id": [9, 9, 10, 10, 11, 11],
            "gk_team_id": [5, 6, 5, 6, 5, 6],
            "defending_team_id": [6, 6, 6, 6, 6, 6],
            "player_id": [940, 11241, 940, 11241, 940, 11241],
            "drop_reason": ["ball_row_missing", "ball_row_missing", None, None, None, None],
        }
    )
    # Non-vacuity: BOTH keepers must be present, or the selection has nothing to select from.
    assert provenance["player_id"].nunique() == 2

    scored = provenance[provenance["drop_reason"].isna()].reset_index(drop=True)
    keep = np.asarray(ids_equal(scored["gk_team_id"], scored["defending_team_id"]), dtype=bool)
    selected = scored[keep]

    assert len(selected) == 2, "expected exactly one row per frame"
    assert selected.groupby(["game_id", "period_id", "frame_id"]).size().max() == 1
    assert set(selected["player_id"]) == {11241}, "the ATTACKING team's keeper must not be credited"


def test_defending_keeper_selection_is_dtype_safe():
    """ADR-019: `gk_team_id` and `defending_team_id` may arrive on different dtypes; a raw `==`
    would select NOTHING and silently produce an empty arm-values table."""
    from silly_kicks.id_compat import ids_equal

    scored = pd.DataFrame({"gk_team_id": pd.array([5, 6], dtype="Int64"), "defending_team_id": ["6", "6"]})
    assert int(ids_equal(scored["gk_team_id"], scored["defending_team_id"]).sum()) == 1


def _driver_sources():
    """Every PUBLIC driver in ``scripts/``, DERIVED rather than listed.

    A hand-maintained list rots: a new driver is added, nobody remembers to register it, and the
    gate silently covers one file fewer than it claims. Underscore-prefixed modules are excluded
    because they are imported helpers whose docstrings argparse never prints -- and two of them
    (`_loader_*`) legitimately carry non-ASCII.
    """
    import pathlib

    d = pathlib.Path(__file__).resolve().parents[2] / "scripts"
    return sorted(p for p in d.glob("*.py") if not p.name.startswith("_"))


def _non_ascii(path) -> list[str]:
    return sorted({c for c in path.read_text(encoding="utf-8") if ord(c) > 127})


# PRE-EXISTING debt, pinned EXACTLY (see the both-ways test below), not waved through. Listed
# rather than repaired all at once because a blanket edit of eighteen unrelated drivers is not one
# PR's to make -- but a silent narrowing of the gate to "only the files I happened to touch" would
# have hidden the rest entirely, so the list is exact and shrinks as drivers are touched.
#
# TWO CORRECTIONS, both measured, to what this comment used to assert ("every one of these drivers
# dies with UnicodeEncodeError on `--help` from a cp1252 console"):
#   1. argparse prints the module docstring only when the parser is handed it. Measured across the
#      list: **4 of 18** pass `__doc__` to `ArgumentParser`; the other 14 never print it, so no
#      console encoding can make their `--help` raise.
#   2. NON-ASCII is not NON-cp1252. Measured: an em dash (U+2014) encodes fine in cp1252 and raises
#      on cp437 and ascii; U+0394 raises on all three; U+2265 raises on cp1252 but encodes on cp437.
#      `calibrate_xt_bandwidth` held em dashes ONLY, so it never had the failure on the console the
#      story named -- it was reachable on cp437 or a redirected ascii stdout.
# The gate stays: ASCII-only source is the rule that holds under EVERY console encoding, which is
# the property worth having. Only the justification was wrong, and a gate defended by a claim its
# own population falsifies is one an inconvenienced contributor deletes.
#
# The `calibrate_*` pair also carried a "not modifiable from this cycle" note that is no longer
# true: the owner ruled (2026-07-29) that the corpus-driver cycle has NO isolation exclusions, and
# `calibrate_xt_bandwidth` was cleaned when that cycle migrated it. `calibrate_tracking_defaults`
# follows when its own migration lands.
_KNOWN_NON_ASCII_DRIVERS = frozenset(
    {
        "build_worldcup_fixture",
        "calibrate_tracking_defaults",
        "download_skillcorner_sample",
        "extract_paired_idsse_fixture",
        "extract_provider_fixtures",
        "gen_ghost_gk_kde_golden",
        "probe_preprocess_baseline",
        "regenerate_action_context_baselines",
        "regenerate_gs_et_native_gk",
        "train_ghost_gk",
        "train_gk_completion",
        "train_gk_retention",
        "validate_xs_probe",
        "validate_xtgk_possession_value",
        "validate_xtgk_v2",
        "xtgk_v2_kappa_sweep",
        "xtgk_v2_keeper_discrimination",
    }
)


@pytest.mark.parametrize("src", _driver_sources(), ids=lambda p: p.stem)
def test_driver_source_is_ascii_so_help_works_on_windows(src):
    """A driver whose `--help` prints its module docstring dies with UnicodeEncodeError the moment
    that docstring holds a character the console cannot encode -- measured: U+0394 in a delta
    description, on cp1252.

    ASCII-only is the rule rather than cp1252-safe because which characters survive depends on the
    console: an em dash encodes on cp1252 and raises on cp437, U+2265 does the opposite (both
    measured; see `_KNOWN_NON_ASCII_DRIVERS`). ASCII is the only set that holds under all of them,
    and it needs no per-file reasoning about which parser prints what."""
    if src.stem in _KNOWN_NON_ASCII_DRIVERS:
        pytest.skip("pre-existing debt, pinned exactly by test_the_known_offender_list_is_EXACT")
    assert not _non_ascii(src), f"non-ASCII in {src.name} can break --help on a non-UTF-8 console: {_non_ascii(src)}"


def test_the_known_offender_list_is_EXACT():
    """Fails in BOTH directions, which is the only thing that stops a debt list from becoming a
    dumping ground: a NEW offender cannot join silently, and a FIXED one must be removed from the
    list. A one-sided `actual <= known` would let the gate rot into permanent permission."""
    actual = {p.stem for p in _driver_sources() if _non_ascii(p)}
    assert actual == set(_KNOWN_NON_ASCII_DRIVERS), (
        f"new offenders: {sorted(actual - _KNOWN_NON_ASCII_DRIVERS)}; "
        f"fixed, remove from the list: {sorted(_KNOWN_NON_ASCII_DRIVERS - actual)}"
    )


def test_the_ascii_gate_actually_covers_the_drivers_it_claims_to():
    """Meta-assertion: a derived list that silently resolved to nothing would make every
    parametrised case above vacuous -- and the drivers THIS cycle ships must be actively checked,
    never skipped as debt."""
    names = {p.stem for p in _driver_sources()}
    mine = {"build_gkdv_arm_values", "build_layer2_spells", "run_signoff_power"}
    assert mine <= names
    assert not (mine & _KNOWN_NON_ASCII_DRIVERS), "this cycle's own drivers must not be on the debt list"
    assert not any(n.startswith("_") for n in names)


def test_corpus_manifest_AGGREGATES_every_partition(tmp_path):
    """MEASURED defect this replaces: with 8 parallel workers all writing one shared
    `arm_values_manifest.json`, the last writer won -- `totals` described a SINGLE partition
    (n_matches: 8) while `arms_written` covered all 64. The data was never wrong; the artifact
    misdescribed its own SCOPE, which for a provenance-bearing file is the same class of defect as
    a false commit SHA.

    `drop_reasons` cannot be recomputed from the shard table (it holds only SCORED rows), so
    aggregation must read the per-worker manifests.
    """
    import json

    for i in range(3):
        (tmp_path / f"manifest_p{i}.json").write_text(
            json.dumps(
                {
                    "n_frames_in": 100,
                    "n_frames_scored": 10,
                    "n_matches": 8,
                    "drop_reasons": {"no_possession": 60, "ball_far_from_attacked_goal": 30},
                    "partition": f"p{i}",
                }
            ),
            encoding="utf-8",
        )

    got = mod._aggregate_manifests(tmp_path)
    assert got["n_matches"] == 24, "must SUM partitions, not report the last writer's 8"
    assert got["n_frames_in"] == 300
    assert got["n_frames_scored"] == 30
    assert got["drop_reasons"] == {"no_possession": 180, "ball_far_from_attacked_goal": 90}
    assert got["n_partitions"] == 3
    assert got["partitions"] == ["p0", "p1", "p2"]
    assert got["conservation_holds"] is True


def test_conservation_is_reported_FALSE_when_it_genuinely_fails(tmp_path):
    """The other side. A conservation flag that only ever reads True is decoration -- it must be
    able to say no, or a run that lost frames would look healthy."""
    import json

    (tmp_path / "manifest_p0.json").write_text(
        json.dumps({"n_frames_in": 100, "n_frames_scored": 10, "n_matches": 1, "drop_reasons": {"x": 5}}),
        encoding="utf-8",
    )
    got = mod._aggregate_manifests(tmp_path)
    assert got["conservation_holds"] is False, "10 + 5 != 100 must be reported as a failure"


def test_aggregate_of_an_empty_dir_is_empty_not_a_crash(tmp_path):
    got = mod._aggregate_manifests(tmp_path)
    assert got["n_partitions"] == 0
    assert got["n_matches"] == 0
