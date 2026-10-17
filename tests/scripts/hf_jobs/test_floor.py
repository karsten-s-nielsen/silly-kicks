import pytest

from scripts.hf_jobs.floor import evaluate_spec, read_pair, regression_check


def test_lower_better_regression_beyond_delta_blocks():
    ok, d = regression_check(0.02, 0.05, direction="lower_better", delta=0.01)
    assert ok is False and d["regressed"] is True


def test_lower_better_within_delta_admits():
    ok, _ = regression_check(0.02, 0.025, direction="lower_better", delta=0.01)
    assert ok is True


def test_lower_better_improvement_admits():
    ok, _ = regression_check(0.02, 0.01, direction="lower_better", delta=0.01)
    assert ok is True


def test_higher_better_regression_beyond_delta_blocks():
    ok, _ = regression_check(0.80, 0.76, direction="higher_better", delta=0.01)
    assert ok is False


def test_higher_better_improvement_admits():
    ok, _ = regression_check(0.80, 0.83, direction="higher_better", delta=0.01)
    assert ok is True


def test_nan_is_fail_closed():
    ok, _ = regression_check(0.80, float("nan"), direction="higher_better", delta=0.01)
    assert ok is False


def test_read_pair_and_evaluate_spec():
    from scripts.hf_jobs import registry as reg

    spec = reg.REGISTRY["win_probability"]  # primary_metric="ece", lower_better
    run = {"committed_ece": 0.02, "fresh_ece": 0.05}
    assert read_pair(run, "ece") == (0.02, 0.05)
    ok, _ = evaluate_spec(spec, run)
    assert ok is False


def test_missing_pair_key_blocks():
    from scripts.hf_jobs import registry as reg

    spec = reg.REGISTRY["win_probability"]
    with pytest.raises(KeyError):
        evaluate_spec(spec, {"fresh_ece": 0.02})  # committed_ece absent -> cannot silently pass


def test_direction_polarity_per_trainer():
    # Pins each registry metric_direction: a worsened fresh value must BLOCK, an improved one must ADMIT.
    # The same-model feasibility test (committed == fresh) cannot catch an inverted direction; this does.
    from scripts.hf_jobs import registry as reg

    for spec in reg.REGISTRY.values():
        base, m, d = 0.5, spec.primary_metric, spec.delta
        worse = base + 2 * d if spec.metric_direction == "lower_better" else base - 2 * d
        better = base - 2 * d if spec.metric_direction == "lower_better" else base + 2 * d
        ok_worse, _ = evaluate_spec(spec, {f"committed_{m}": base, f"fresh_{m}": worse})
        ok_better, _ = evaluate_spec(spec, {f"committed_{m}": base, f"fresh_{m}": better})
        assert ok_worse is False and ok_better is True, f"{spec.name}: direction polarity wrong"
