"""The Stage-1 gate is pre-registered: invariance >= 99.9%, and the argmax must not move.

Pre-registered means the threshold was fixed in the design (spec D5) BEFORE any corrected-geometry
data was scored, so this file exists to pin it against later adjustment. A gate whose threshold
moves after seeing the result is not a gate.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

from scripts.check_stage1_argmax import (
    augment_metrics,
    build_fold,
    compare_assignments,
    invariance_verdict,
    load_neighbours,
    moved_beyond_noise,
    reflect_frames,
    require_velocity,
)


def test_at_threshold_passes():
    assert invariance_verdict(same=9990, total=10000) == "stands"


def test_below_threshold_requires_sweep():
    assert invariance_verdict(same=9989, total=10000) == "sweep"


def test_perfect_invariance_passes():
    assert invariance_verdict(same=10, total=10) == "stands"


def test_zero_rows_is_an_error_not_a_pass():
    """An empty comparison must never read as 'stands' -- that is a silent no-op gate."""
    with pytest.raises(ValueError):
        invariance_verdict(same=0, total=0)


def test_the_reflection_negates_velocity():
    """Prong 1 is STRUCTURALLY BLIND to this (beta=0 kills the velocity term), and prong 2 -- whose
    neighbours have beta != 0 -- is what gets corrupted. So it must be tested directly."""
    src = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "frame_id": [1, 1],
            "player_id": ["a", "b"],
            "team_id": ["H", "A"],
            "x": [10.0, 95.0],
            "y": [20.0, 48.0],
            "vx": [1.5, -2.0],
            "vy": [-0.5, 3.0],
        }
    )
    out = reflect_frames(src)
    assert list(out["x"]) == [95.0, 10.0]
    assert list(out["y"]) == [48.0, 20.0]
    assert list(out["vx"]) == [-1.5, 2.0], "velocities were not negated -- not a reflection"
    assert list(out["vy"]) == [0.5, -3.0]


def test_the_reflection_leaves_the_source_untouched():
    """The checker scores BOTH legs; a reflection that mutated its input would make the factual leg
    the reflected one and the comparison vacuous."""
    src = pd.DataFrame(
        {
            "game_id": [1],
            "period_id": [1],
            "frame_id": [1],
            "player_id": ["a"],
            "team_id": ["H"],
            "x": [10.0],
            "y": [20.0],
            "vx": [1.5],
            "vy": [-0.5],
        }
    )
    before = src.copy(deep=True)
    reflect_frames(src)
    pd.testing.assert_frame_equal(src, before)


def test_missing_velocity_columns_raise_rather_than_zero_silently():
    """`_ball_carrier.py` substitutes pvx=0.0 when vx/vy are absent, which makes beta inert and
    every neighbour score identically -- an argmax that 'cannot move' for the wrong reason."""
    with pytest.raises(ValueError, match="vx"):
        require_velocity(pd.DataFrame({"x": [1.0], "y": [2.0]}))


def test_all_nan_velocity_is_rejected_too():
    """Present-but-empty is the same failure with a different shape: the columns exist, so a
    `in frames.columns` check passes, and beta is still inert."""
    with pytest.raises(ValueError, match=r"vx|vy"):
        require_velocity(pd.DataFrame({"x": [1.0], "y": [2.0], "vx": [float("nan")], "vy": [float("nan")]}))


def test_velocity_present_is_accepted():
    """Non-vacuity partner: the guard must accept a frame that HAS velocity, or the two tests above
    would pass against a function that rejects everything."""
    require_velocity(pd.DataFrame({"x": [1.0], "y": [2.0], "vx": [0.5], "vy": [0.5]}))


# --------------------------------------------------------------------------------------------
# compare_assignments -- the no-carrier convention is a STATED choice, so both branches are pinned


def _series(pairs):
    idx = pd.MultiIndex.from_tuples([(str(g), "1", str(f)) for g, f in enumerate(range(len(pairs)))])
    return pd.Series([p for p in pairs], index=idx)


def test_no_carrier_frames_are_EXCLUDED_by_default():
    """Counting `no-carrier == no-carrier` as agreement inflates the fraction by however many
    dead-ball frames the corpus holds. The claim under test is about carrier CHOICE, and a frame
    with no carrier expresses none."""
    a = _series(["p1", "nan", "p3"])
    b = _series(["p1", "nan", "p9"])
    out = compare_assignments(a, b, count_no_carrier_as_agreement=False)
    assert out["n_frames"] == 2, "the both-no-carrier frame must leave the denominator"
    assert out["n_same"] == 1
    assert out["n_no_carrier"] == 1


def test_no_carrier_frames_CAN_be_counted_as_agreement():
    """The opposite convention is defensible; silence is not. Pinning both directions is what makes
    the recorded `no_carrier_convention` field meaningful rather than decorative."""
    a = _series(["p1", "nan", "p3"])
    b = _series(["p1", "nan", "p9"])
    out = compare_assignments(a, b, count_no_carrier_as_agreement=True)
    assert out["n_frames"] == 3
    assert out["n_same"] == 2, "the both-no-carrier frame now counts as agreement"


def test_a_disagreement_is_actually_detected():
    """Non-vacuity: the two tests above would pass against a function that always reports agreement
    on the frames it keeps."""
    a = _series(["p1", "p2"])
    b = _series(["p1", "p9"])
    out = compare_assignments(a, b, count_no_carrier_as_agreement=False)
    assert (out["n_frames"], out["n_same"]) == (2, 1)


# --------------------------------------------------------------------------------------------
# build_fold -- a fabricated home_team_id would mis-orient geometry inside the objective


def _write_corpus(tmp_path):
    gen = tmp_path / "shards" / "tok"
    gen.mkdir(parents=True)
    acts = tmp_path / "_actions"
    acts.mkdir()
    frames = pd.DataFrame({"game_id": ["g1"], "x": [1.0], "y": [2.0]})
    frames.to_parquet(gen / "skillcorner__1886347.parquet")
    pd.DataFrame({"action_id": [1]}).to_parquet(acts / "skillcorner__1886347.parquet")
    (tmp_path / "home_teams.json").write_text(json.dumps({"g1": "4177"}))
    return gen, acts, tmp_path / "home_teams.json"


def test_build_fold_keys_home_by_GAME_ID_not_match_id(tmp_path):
    """SkillCorner's `game_id` is a kloppy hash unrelated to its match id; keying by the filename
    would find no home team for every SkillCorner match while looking populated."""
    gen, acts, home = _write_corpus(tmp_path)
    fold = build_fold(sorted(gen.glob("*.parquet")), actions_dir=acts, home_teams=home)
    assert list(fold) == ["skillcorner"]
    (_actions, _frames, home_id) = fold["skillcorner"][0]
    assert home_id == "4177"


def test_build_fold_SKIPS_a_match_with_no_actions_rather_than_defaulting(tmp_path):
    gen, acts, home = _write_corpus(tmp_path)
    (acts / "skillcorner__1886347.parquet").unlink()
    fold = build_fold(sorted(gen.glob("*.parquet")), actions_dir=acts, home_teams=home)
    assert fold == {}, "a match without actions must be skipped, not scored on a fabricated fold"


def test_build_fold_SKIPS_a_match_with_no_home_id_rather_than_defaulting(tmp_path):
    """A fabricated `home_team_id` would silently mis-orient one match's geometry inside an
    objective whose whole purpose here is to detect geometry-driven change."""
    gen, acts, home = _write_corpus(tmp_path)
    home.write_text(json.dumps({"some-other-game": "999"}))
    fold = build_fold(sorted(gen.glob("*.parquet")), actions_dir=acts, home_teams=home)
    assert fold == {}


# --------------------------------------------------------------------------------------------
# load_neighbours -- exercised against the REAL store, because its failure modes are all shape


_STORE = pathlib.Path("calibration_runs/balanced_confirm_tol3/s1.db")


@pytest.mark.skipif(not _STORE.is_file(), reason="prior Optuna store not present (gitignored)")
def test_neighbours_exclude_the_optimum_itself():
    """The optimum is not its own neighbour; including it would guarantee a tie and make
    `argmax_moved` structurally unable to report movement."""
    opt = json.loads((_STORE.parent / "carrier_best.json").read_text())
    optimum = {"beta": float(opt["beta"]), "gamma": float(opt["gamma"])}
    nbs = load_neighbours(_STORE, optimum=optimum, k=4)
    assert len(nbs) == 4
    for nb in nbs:
        assert not (abs(nb["beta"] - optimum["beta"]) < 1e-12 and abs(nb["gamma"] - optimum["gamma"]) < 1e-12)


@pytest.mark.skipif(not _STORE.is_file(), reason="prior Optuna store not present (gitignored)")
def test_neighbours_are_the_NEAREST_in_normalised_space():
    """Un-normalised distance would let the wider parameter dominate, so the neighbour set would
    probe one axis only -- a sensitivity test that cannot see the other dimension."""
    opt = json.loads((_STORE.parent / "carrier_best.json").read_text())
    optimum = {"beta": float(opt["beta"]), "gamma": float(opt["gamma"])}
    near = load_neighbours(_STORE, optimum=optimum, k=3)
    far = load_neighbours(_STORE, optimum=optimum, k=30)
    assert [(n["beta"], n["gamma"]) for n in near] == [(f["beta"], f["gamma"]) for f in far[:3]], (
        "k=3 must be the first three of k=30 -- otherwise the ordering is not by distance"
    )


# --------------------------------------------------------------------------------------------
# moved_beyond_noise -- the argmax rule. Mirrors `_diagnostics.tf25_gate_fires`, so it is pinned
# from BOTH sides plus the nan case, exactly as that gate is.


def test_a_gain_larger_than_the_SE_counts_as_moved():
    assert moved_beyond_noise(recorded=0.500, best_alternative=0.510, se=0.005) is True


def test_a_gain_smaller_than_the_SE_does_NOT_count():
    """The measured case: the margin was 1.005e-4 while between-fold spread was ~1e-2. A bare
    boolean on the raw difference would have called that a move."""
    assert moved_beyond_noise(recorded=0.537868, best_alternative=0.537968, se=0.01) is False


def test_a_gain_exactly_at_the_SE_does_NOT_count():
    """Strict `>`, matching `tf25_gate_fires`: a tie leaves the recorded point winning, the
    conservative reading for a confirmation whose purpose is to avoid an unnecessary sweep."""
    # Operands chosen binary-exact ON PURPOSE: 0.51 - 0.5 == 0.010000000000000009, so the
    # obvious numbers would have tested float representation instead of the rule.
    assert moved_beyond_noise(recorded=0.5, best_alternative=0.75, se=0.25) is False


def test_a_nan_SE_can_never_justify_moved():
    """A single fold has no SE. `tf25_gate_fires` refuses a provider-specific default on a nan SE
    for the same reason: an unmeasurable spread is not evidence of a difference."""
    assert moved_beyond_noise(recorded=0.5, best_alternative=0.9, se=float("nan")) is False


def test_a_worse_alternative_is_never_moved():
    assert moved_beyond_noise(recorded=0.6, best_alternative=0.5, se=0.001) is False


def test_the_shard_key_does_not_repeat_the_separator():
    """`join_key` REJECTS a component containing `__`, because ("a__b","c") and ("a","b__c") both
    join to "a__b__c" -- two distinct items silently sharing one shard. Passing the full stem as the
    match component tripped exactly that guard on the first real run."""
    from scripts._driver import join_key
    from scripts.check_stage1_argmax import shard_key

    key = shard_key(pathlib.Path("/x/y/gradientsports__10502.parquet"))
    assert key == ("gradientsports", "10502")
    assert join_key(key)  # must not raise


def test_the_shard_key_survives_a_match_id_containing_the_separator():
    """Partition on the FIRST separator only: a provider prefix is stripped, the rest is kept whole
    rather than truncated at a second `__`."""
    from scripts.check_stage1_argmax import shard_key

    assert shard_key(pathlib.Path("/x/skillcorner__a__b.parquet")) == ("skillcorner", "a__b")


# --------------------------------------------------------------------------------------------
# augment_metrics -- the confirmation `out` dict is AUGMENTED (never replaced), and the recommendation
# artifact + fold-stability diagnostic ride alongside. The critical property is NON-LOSSINESS.


def _summary():
    # shipped incumbent + two candidates, all within noise -> keep incumbent
    return {
        "shipped_point": {
            "mean": 0.80,
            "se": 0.01,
            "per_fold": [0.79, 0.80, 0.81],
            "params": {"beta": 0.0, "gamma": 0.25},
        },
        "recorded_optimum": {
            "mean": 0.8001,
            "se": 0.01,
            "per_fold": [0.7901, 0.8001, 0.8101],
            "params": {"beta": 1.9e-4, "gamma": 0.221},
        },
        "nb0": {
            "mean": 0.8002,
            "se": 0.01,
            "per_fold": [0.7902, 0.8002, 0.8102],
            "params": {"beta": 0.1, "gamma": 0.3},
        },
    }


def _base_out(summary):
    # a realistic confirmation `out`, carrying the Prong-1 invariance result + metadata that MUST survive
    return {
        "invariance": {"shipped_point": {"invariance_fraction": 0.9999, "verdict": "stands"}},
        "invariance_threshold": 0.999,
        "cv_scheme": "GroupKFold(5)",
        "objective": "CarrierAccuracyObjective.carrier_accuracy",
        "argmax_moved": False,
        "points": summary,
        "run_commit": "abc123",
        "run_tree_dirty": False,
    }


def test_augment_metrics_is_non_lossy_and_adds_selection_no_tolerance_m():
    base = _base_out(_summary())
    out, selected = augment_metrics(base, provenance={"commit": "abc123", "dirty": False}, min_effect_size=0.01)
    # NON-LOSSY (F1 regression guard): the invariance prong + all metadata survive
    assert out["invariance"] == base["invariance"]
    assert out["cv_scheme"] == "GroupKFold(5)" and out["objective"] == base["objective"]
    assert out["argmax_moved"] is False and out["run_commit"] == "abc123"
    # ADDED blocks
    assert out["selection"]["moved"] is False
    assert out["fold_stability"]["verdict"] == "no_discriminating_evidence"
    assert "fold_winners" in out["fold_stability"]  # §3.4 per-fold ranks
    assert "fold_to_point_var_ratio" in out["fold_stability"]  # §3.4 variance ratio
    # selection artifact: {beta, gamma} + provenance, NO tolerance_m
    assert selected["beta"] == 0.0 and selected["gamma"] == 0.25 and "tolerance_m" not in selected
    assert selected["run_commit"] == "abc123" and selected["run_tree_dirty"] is False


def test_fold_stability_verdict_flips_on_a_discriminating_fold_set():
    # non-vacuity (spec §6): a wide-separation, low-SE candidate MOVES the verdict off the incumbent
    summary = _summary()
    summary["nb0"] = {"mean": 0.91, "se": 0.005, "per_fold": [0.90, 0.91, 0.92], "params": {"beta": 0.1, "gamma": 0.3}}
    out, selected = augment_metrics(
        _base_out(summary), provenance={"commit": "abc", "dirty": False}, min_effect_size=0.01
    )
    assert out["selection"]["moved"] is True
    assert out["fold_stability"]["verdict"] == "moved"
    assert selected["beta"] == 0.1 and selected["gamma"] == 0.3  # the artifact follows the move
