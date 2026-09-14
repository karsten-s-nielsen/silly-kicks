import dataclasses

import pytest

from silly_kicks.gk_decision import GkDecisionParams, GkDecisionReport
from silly_kicks.gk_decision._columns import GK_DECISION_DROP_REASONS, OPTION_SET_SOURCE_VALUES


def test_params_defaults_and_provider():
    assert GkDecisionParams().min_options == 3
    assert GkDecisionParams().value_fn == "completion_progression"
    assert GkDecisionParams().reachability_min_xpass == 0.85
    assert GkDecisionParams().fov_radius_m == 10.0
    assert GkDecisionParams().fov_min_observed_fraction == 0.7
    assert GkDecisionParams().receiver_exclusion_m == 5.0
    assert GkDecisionParams.default().is_default() is True
    assert GkDecisionParams().is_default() is False
    # empty override map (ADR-009): every provider resolves to base
    assert GkDecisionParams.for_provider("skillcorner") == GkDecisionParams()


def test_params_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        GkDecisionParams().min_options = 5  # type: ignore[misc]


def test_report_conserves():
    # incl. the reconstruction drops (PR2): n_no_frame / n_fov_cropped default 0 (native path).
    r = GkDecisionReport(
        GkDecisionParams(),
        n_decisions_in=15,
        n_scored=6,
        n_too_few_options=2,
        n_no_unique_chosen=1,
        n_chosen_unvalued=1,
        n_no_frame=3,
        n_fov_cropped=2,
    )
    assert (
        r.n_scored + r.n_too_few_options + r.n_no_unique_chosen + r.n_chosen_unvalued + r.n_no_frame + r.n_fov_cropped
        == r.n_decisions_in
    )
    # native path: the two reconstruction counters default to 0 (Phase-1 byte-identical shape)
    native = GkDecisionReport(
        GkDecisionParams(), n_decisions_in=6, n_scored=4, n_too_few_options=1, n_no_unique_chosen=1, n_chosen_unvalued=0
    )
    assert native.n_no_frame == 0 and native.n_fov_cropped == 0


def test_vocab_closed():
    assert OPTION_SET_SOURCE_VALUES == ("native", "reconstructed")
    assert set(GK_DECISION_DROP_REASONS) == {
        "too_few_options",
        "no_unique_chosen",
        "chosen_unvalued",
        "no_frame",
        "fov_cropped",
    }
