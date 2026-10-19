import pytest

from scripts.hf_jobs import registry as reg
from scripts.hf_jobs.floor import certify, evaluate_spec, read_pair, regression_check


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


def test_owner_certification_is_cv_parity():
    spec = reg.REGISTRY["ghost_gk"]
    ok, d = certify(
        spec, "owner", fresh_metrics={"cv_mae_euclidean_mean": 1.15}, committed_metrics={"cv_mae_euclidean_mean": 1.13}
    )
    assert ok and d["basis"] == "cv_parity"
    bad, _ = certify(
        spec, "owner", fresh_metrics={"cv_mae_euclidean_mean": 1.30}, committed_metrics={"cv_mae_euclidean_mean": 1.13}
    )
    assert not bad


def test_public_certification_needs_the_headtohead_pair():
    spec = reg.REGISTRY["win_probability"]
    with pytest.raises(KeyError):
        certify(spec, "public", fresh_metrics={}, committed_metrics={}, headtohead={})


def test_owner_cv_metric_reads_the_shipped_candidate_from_each_file():
    """xshot/xcross sc_extended: the dotted path resolves ``{shipped_variant}`` per file, and a
    worsened fresh PR-AUC beyond the delta BLOCKS (higher_better)."""
    spec = reg.REGISTRY["xshot_occurrence"]

    def metrics(variant, pr_auc):
        return {"shipped_variant": variant, "candidates": {variant: {"metrics": {"pr_auc": pr_auc}}}}

    ok, d = certify(spec, "sc_extended", fresh_metrics=metrics("v", 0.405), committed_metrics=metrics("v", 0.41))
    assert ok and d["metric"] == "candidates.{shipped_variant}.metrics.pr_auc"
    bad, _ = certify(spec, "sc_extended", fresh_metrics=metrics("v", 0.38), committed_metrics=metrics("v", 0.41))
    assert not bad
    with pytest.raises(KeyError):  # a fresh file without the committed shipped candidate cannot silently pass
        certify(
            spec,
            "sc_extended",
            fresh_metrics={"shipped_variant": "v", "candidates": {}},
            committed_metrics=metrics("v", 0.41),
        )


def test_every_owner_certified_trainer_names_a_cv_metric():
    for spec in reg.REGISTRY.values():
        tiers = spec.tier_certification or {"-": spec.certification}
        if "owner" in tiers.values():
            assert spec.cv_metric, f"{spec.name}: owner-certified without a cv_metric"


def test_an_unknown_basis_is_refused():
    import dataclasses

    spec = dataclasses.replace(reg.REGISTRY["ghost_gk"], certification="rebaseline")
    with pytest.raises(ValueError, match="identity"):
        certify(spec, "owner", fresh_metrics={}, committed_metrics={})
