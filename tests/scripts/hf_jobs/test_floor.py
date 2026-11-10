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


# ---- HFU spec 14.4: public = in-sample parity (the head-to-head) AND two-sided CV parity ------------------


def _h2h(spec, *, ok: bool = True) -> dict:
    m, d = spec.primary_metric, spec.delta
    worse = 0.5 + 2 * d if spec.metric_direction == "lower_better" else 0.5 - 2 * d
    return {f"committed_{m}": 0.5, f"fresh_{m}": 0.5 if ok else worse}


@pytest.mark.parametrize(("fresh", "ok"), [(0.0068, True), (0.0065, True), (0.0069, False), (0.0061, False)])
def test_public_cv_parity_relative_5pct(fresh, ok):
    spec = reg.REGISTRY["win_probability"]
    got, d = certify(
        spec, "public", fresh_metrics={"ece": fresh}, committed_metrics={"ece": 0.0065}, headtohead=_h2h(spec)
    )
    assert got is ok
    assert d["in_sample_parity"]["basis"] == "in_sample_parity" and d["cv_parity"]["metric"] == "ece"
    assert d["cv_parity"]["rel_tol"] == 0.05


def test_public_needs_both_parities():
    spec = reg.REGISTRY["pass_completion"]
    good, bad = {"ece": 0.0065}, {"ece": 0.0100}
    assert certify(spec, "public", fresh_metrics=good, committed_metrics=good, headtohead=_h2h(spec))[0]
    assert not certify(spec, "public", fresh_metrics=bad, committed_metrics=good, headtohead=_h2h(spec))[0]
    assert not certify(spec, "public", fresh_metrics=good, committed_metrics=good, headtohead=_h2h(spec, ok=False))[0]


@pytest.mark.parametrize(("offset", "ok"), [(5e-7, True), (-5e-7, True), (2e-6, False), (-2e-6, False)])
def test_rho_parity_two_sided(offset, ok):
    spec = reg.REGISTRY["match_outcome_dependence"]
    rho = 0.021614208683848977
    got, d = certify(
        spec, "public", fresh_metrics={"rho": rho + offset}, committed_metrics={"rho": rho}, headtohead=_h2h(spec)
    )
    assert got is ok and d["cv_parity"]["abs_tol"] == 1e-6


def test_public_cv_parity_nan_is_fail_closed():
    spec = reg.REGISTRY["xsuccess"]
    got, _ = certify(
        spec, "public", fresh_metrics={"brier": float("nan")}, committed_metrics={"brier": 0.084}, headtohead=_h2h(spec)
    )
    assert not got


def test_every_public_certified_trainer_names_a_cv_metric():
    want = {
        "win_probability": "ece",
        "pass_completion": "ece",
        "xsuccess": "brier",
        "match_outcome_dependence": "rho",
        "xshot_occurrence": "candidates.{shipped_variant}.metrics.pr_auc",
        "xcross_attempt": "candidates.{shipped_variant}.metrics.pr_auc",
    }
    public = {
        s.name: s.cv_metric
        for s in reg.REGISTRY.values()
        if "public" in (s.tier_certification or {"-": s.certification}).values()
    }
    assert public == want


_XC_FRESH = {"shipped_variant": "public", "candidates": {"public": {"metrics": {"pr_auc": 0.08673616941158695}}}}
_XC_COMMITTED = {"shipped_variant": "public", "candidates": {"public": {"metrics": {"pr_auc": 0.0872974779264555}}}}
_XC_H2H = {"committed_pr_auc": 0.1753334038608238, "fresh_pr_auc": 0.15699130250937446}  # launch 6


def test_public_certification_without_the_in_sample_gate():
    """HFU spec 17.3 (F31): with ``in_sample_gate=False`` a public run passes on CV parity alone; the head-to-head is
    recorded, not gated. The launch-6 xCross figures fail with the gate and pass without it."""
    spec = reg.REGISTRY["xcross_attempt"]
    kw = {"fresh_metrics": _XC_FRESH, "committed_metrics": _XC_COMMITTED, "headtohead": _XC_H2H}
    assert not certify(spec, "public", **kw)[0]
    ok, d = certify(spec, "public", **kw, in_sample_gate=False)
    assert ok and d["in_sample_parity"]["gated"] is False and d["in_sample_parity"]["regressed"] is True


def test_only_the_xcross_default_build_drops_the_in_sample_gate():
    off = {
        (name, b.variant)
        for name, spec in reg.REGISTRY.items()
        if isinstance(spec, reg.TrainerSpec)
        for b in spec.builds
        if not b.in_sample_gate
    }
    assert off == {("xcross_attempt", "default")}
