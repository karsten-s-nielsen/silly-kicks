import pytest

from silly_kicks.calibration._selection import (
    PointScore,
    Selection,
    build_selection_artifact,
    select_recommended_point,
)


def _pt(label, per_fold, beta=0.0, gamma=0.25):
    per_fold = tuple(per_fold)
    return PointScore(
        label=label,
        params={"beta": beta, "gamma": gamma},
        per_fold=per_fold,
        mean=sum(per_fold) / len(per_fold),
    )


_INC = _pt("shipped", [0.79, 0.80, 0.81])  # mean 0.80


def test_incumbent_kept_when_no_candidate_clears():
    cand = _pt("c", [0.8001, 0.8001, 0.8001])  # gain 1e-4 < δ
    sel = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)
    assert sel.moved is False and sel.selected is _INC


def test_incumbent_replaced_when_candidate_clears_both():
    cand = _pt("c", [0.84, 0.85, 0.86])  # gain 0.05 > δ; consistent diff -> clears paired SE
    sel = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is cand
    assert sel.effect_size == pytest.approx(0.05)


def test_gain_exactly_at_delta_is_kept_strict():
    # gain lands EXACTLY on δ via exact binary fractions -> strict '>' keeps; a regression to '>=' MOVES.
    # (0.79/0.80/0.81 literals compute gain 0.00999...898 < δ, so they never exercise the boundary.)
    inc = _pt("shipped", [0.5, 0.5, 0.5])  # mean 0.5
    cand = _pt("c", [0.75, 0.75, 0.75])  # mean 0.75; gain exactly 0.25 == δ
    sel = select_recommended_point(incumbent=inc, candidates=[cand], min_effect_size=0.25)
    assert sel.moved is False


def test_effect_size_floor_is_load_bearing_both_sides():
    # SAME candidate, two δ: high δ keeps (δ blocks); low δ moves. The MOVE proves the paired-SE bar
    # cleared, so δ was the sole blocker -- the real from-both-sides "floor is load-bearing" assertion.
    cand = _pt("c", [0.792, 0.803, 0.815])  # gain ~0.00333, tiny consistent diff -> clears paired SE
    assert select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01).moved is False
    moved = select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.001)
    assert moved.moved is True and moved.selected is cand


def test_paired_se_exactly_zero_is_recorded_on_the_moved_branch():
    # exactly-representable folds -> per-fold diff is exactly 0.25 -> paired_se == 0.0 (no float noise).
    inc = _pt("shipped", [0.5, 0.625, 0.75])  # mean 0.625
    cand = _pt("c", [0.75, 0.875, 1.0])  # mean 0.875; diffs all exactly 0.25
    sel = select_recommended_point(incumbent=inc, candidates=[cand], min_effect_size=0.1)  # gain 0.25 > δ
    assert sel.moved is True
    assert sel.paired_se == 0.0  # recorded on the moved branch -> the exact-zero SE path is exercised


def test_two_candidates_both_clear_best_gain_wins():
    c1 = _pt("c1", [0.83, 0.84, 0.85])  # gain 0.04
    c2 = _pt("c2", [0.87, 0.88, 0.89])  # gain 0.08
    sel = select_recommended_point(incumbent=_INC, candidates=[c1, c2], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is c2


def test_recorded_optimum_is_a_candidate_distinct_from_incumbent():
    recorded = _pt("recorded_optimum", [0.84, 0.85, 0.86], beta=1.9e-4, gamma=0.221)
    neighbour = _pt("nb0", [0.805, 0.81, 0.815], beta=0.1, gamma=0.3)
    sel = select_recommended_point(incumbent=_INC, candidates=[recorded, neighbour], min_effect_size=0.01)
    assert sel.moved is True and sel.selected is recorded


def test_fold_length_mismatch_raises():
    cand = _pt("c", [0.85, 0.85])  # 2 folds vs incumbent's 3
    with pytest.raises(ValueError, match="per_fold length"):
        select_recommended_point(incumbent=_INC, candidates=[cand], min_effect_size=0.01)


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="unknown policy"):
        select_recommended_point(incumbent=_INC, candidates=[], min_effect_size=0.01, policy="argmax")


def _sel(beta=0.0, gamma=0.25, extra_params=None):
    params = {"beta": beta, "gamma": gamma}
    if extra_params:
        params.update(extra_params)
    pt = PointScore(label="shipped", params=params, per_fold=(0.8, 0.8, 0.8), mean=0.8)
    return Selection(
        selected=pt,
        incumbent=pt,
        moved=False,
        reason="kept",
        best_candidate=None,
        effect_size=None,
        paired_se=None,
    )


def test_artifact_carries_beta_gamma_and_provenance_no_tolerance_m():
    # the selected point carries a stray tolerance_m; the artifact must NOT surface it
    art = build_selection_artifact(
        _sel(extra_params={"tolerance_m": 8.0}),
        provenance={"commit": "abc123", "dirty": False},
    )
    assert art["beta"] == 0.0 and art["gamma"] == 0.25
    assert "tolerance_m" not in art
    assert art["run_commit"] == "abc123"
    assert art["run_tree_dirty"] is False


def test_artifact_from_a_dirty_tree_carries_true_and_would_fail_the_output_gate():
    # other side: the structural gate asserts `run_tree_dirty is False`, so a dirty artifact is rejected
    art = build_selection_artifact(_sel(), provenance={"commit": "abc123", "dirty": True})
    assert art["run_tree_dirty"] is True
