"""TF-19 sign-off package: plasmode ATT power + the FIREWALL (spec §5.1, §5.4)."""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.causal.power import InjectionSpec, att_power_curve


def _spells(n: int = 400, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "Z": rng.integers(0, 2, size=n),
        "X": rng.normal(size=(n, 3)),
        "clusters": np.repeat(np.arange(n // 20), 20),
    }


def test_firewall_refuses_a_bare_outcome_array():
    """THE gate this cycle rests on. A call-count spy on `estimate_att` would be VACUOUS -- the
    harness always calls it -- so the guard is provenance, and this is its RED side."""
    s = _spells()
    with pytest.raises(ValueError, match="not an InjectionSpec"):
        att_power_curve(
            Z=s["Z"],
            injection=np.zeros(len(s["Z"])),
            X=s["X"],
            clusters=s["clusters"],
            sizes=(200,),
            n_replicates=2,
            rng_seed=0,
        )


class _FullyDuckTypedFake:
    """A COMPLETE lookalike: every attribute `att_power_curve` touches, including `true_effect`.

    Completeness is the point. An incomplete fake makes the firewall test pass for an INCIDENTAL
    reason -- measured: with the guard removed, an earlier fake carrying no `true_effect` ran the
    whole estimation loop on its own outcomes and only died assembling the return dict
    (`AttributeError: '_Fake' object has no attribute 'true_effect'`). The breach had already
    happened; the crash was luck. With this fake, removing the guard makes the call SUCCEED
    silently, so the test can only pass because the guard stopped it.
    """

    base_rate, relative_effect, true_effect, stamp = 0.15, 0.0, 0.0, "not-the-stamp"

    def draw(self, Z, rng):  # noqa: N803 -- mirrors InjectionSpec.draw exactly
        return np.zeros(len(Z))


def test_firewall_refuses_a_stamp_lookalike():
    """Duck-typing is not enough: a class exposing the whole surface is still not an InjectionSpec."""
    s = _spells()
    with pytest.raises(ValueError, match="not an InjectionSpec"):
        att_power_curve(
            Z=s["Z"],
            injection=_FullyDuckTypedFake(),
            X=s["X"],
            clusters=s["clusters"],
            sizes=(200,),
            n_replicates=2,
            rng_seed=0,
        )


def test_the_lookalike_would_otherwise_run_to_completion():
    """Non-vacuity for the test above: the fake is complete enough that ONLY the guard stops it.
    If this ever fails, the fake has drifted and the firewall test has gone incidental."""
    s = _spells()
    fake = _FullyDuckTypedFake()
    for attr in ("base_rate", "relative_effect", "true_effect", "draw"):
        assert hasattr(fake, attr), f"fake is incomplete: missing {attr}"
    assert fake.draw(s["Z"], np.random.default_rng(0)).shape == s["Z"].shape


def test_a_fresh_outcome_is_drawn_PER_REPLICATE_not_once():
    """Spec §5.4: "Per replicate: ... inject a treatment effect." Freezing one realisation and
    reusing it makes every replicate the same dataset reordered, which understates variance and
    turns the power estimate into a single-draw accident."""
    s = _spells()
    out = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(400,),
        n_replicates=30,
        rng_seed=0,
    )
    assert out["n_distinct_outcome_draws"][400] == 30


def test_power_rises_with_the_injected_effect_both_sides():
    s = _spells(n=800)
    lo = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.0),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(800,),
        n_replicates=60,
        rng_seed=0,
    )
    hi = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.60),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(800,),
        n_replicates=60,
        rng_seed=0,
    )
    assert lo["power_by_size"][800] <= 0.20
    assert hi["power_by_size"][800] >= 0.60


def test_matched_n_is_the_MATCHED_count_not_the_subsample_size():
    """`N_MIN_MATCHED` is spec-defined as "the smallest matched-n bin at which power >= 0.80".
    Recording `idx.size` would record the SUBSAMPLE SIZE -- identically the input, since the
    resampler truncates to exactly `size`. ATT's focal set is the TREATED units only, so a correct
    matched-n is strictly smaller than the subsample."""
    s = _spells()
    out = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(200, 400),
        n_replicates=10,
        rng_seed=0,
    )
    assert out["matched_n_by_size"][400] < 400, "matched_n is echoing the subsample size"
    assert out["matched_n_by_size"][200] < out["matched_n_by_size"][400]


def _rare_treatment(n: int = 800, n_treated_clusters: int = 8, seed: int = 0) -> dict:
    """A design whose treatment is RARE -- one treated row in each of a few clusters.

    This is the shape Layer 2 actually has: "the keeper is advanced past the penalty-area line at
    spell entry" is uncommon, so a small cluster resample frequently contains no treated unit at
    all. Constructed deterministically rather than drawn, so the degenerate counts below are a
    property of the design and not of a lucky seed.
    """
    rng = np.random.default_rng(seed)
    clusters = np.repeat(np.arange(n // 20), 20)
    Z = np.zeros(n, dtype=int)
    Z[np.arange(n_treated_clusters) * 20] = 1  # one treated row in each of the first k clusters
    return {"Z": Z, "X": rng.normal(size=(n, 3)), "clusters": clusters}


def test_a_single_class_resample_is_COUNTED_not_a_crash():
    """MEASURED REGRESSION: this raised out of sklearn ("needs samples of at least 2 classes") and
    killed a corpus run -- after the 8h corpus pass that fed it, none of which was persisted."""
    s = _rare_treatment()
    out = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(100, 800),
        n_replicates=40,
        rng_seed=0,
    )
    assert out["n_degenerate_by_size"][100] > 0, "no positivity failure was exercised at n=100"
    # The OTHER side: the full-corpus size carries every treated cluster, so nothing is degenerate
    # there. Without this the test would pass just as well if the counter fired unconditionally.
    assert out["n_degenerate_by_size"][800] == 0


def test_a_healthy_design_records_zero_degenerate():
    """Non-vacuity for the counter itself: the ordinary balanced design must never register one."""
    s = _spells()
    out = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.20),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(200, 400),
        n_replicates=10,
        rng_seed=0,
    )
    assert out["n_degenerate_by_size"] == {200: 0, 400: 0}


def test_an_inestimable_design_reports_power_ZERO_not_a_nan():
    """Degenerate replicates count in the DENOMINATOR.

    Excluding them would condition on estimability and inflate the curve exactly where the design
    is weakest -- and an all-degenerate size would divide by zero, surfacing as a nan power that
    reads like a missing measurement rather than a design that cannot be estimated at all.
    """
    s = _rare_treatment(n_treated_clusters=0)  # no treated unit anywhere
    out = att_power_curve(
        Z=s["Z"],
        injection=InjectionSpec(base_rate=0.15, relative_effect=0.60),
        X=s["X"],
        clusters=s["clusters"],
        sizes=(200,),
        n_replicates=15,
        rng_seed=0,
    )
    assert out["power_by_size"][200] == 0.0
    assert out["n_degenerate_by_size"][200] == 15
    assert out["matched_n_by_size"][200] == 0  # np.mean([]) would be nan


def test_cluster_resampling_never_splits_a_cluster():
    """Cluster-preserving resampling is the ATT-side analogue of the match-blocked ICC null."""
    from silly_kicks.causal.power import _resample_clusters

    clusters = np.repeat(np.arange(10), 20)
    rng = np.random.default_rng(0)
    idx = _resample_clusters(clusters, np.unique(clusters), 100, rng)
    picked = clusters[idx]
    # every cluster present is present in full (the final cluster may be truncated by the cap)
    counts = np.bincount(picked, minlength=10)
    assert set(counts[counts > 0]) <= {20, int(counts[counts > 0].min())}
    assert len(idx) == 100
