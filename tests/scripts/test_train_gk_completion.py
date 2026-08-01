"""The GK-completion bundling guard is keyed on SERVED PREDICTIONS, not on parameter deltas.

A max-over-arrays comparison of `coef`/`intercept`/`mean`/`std` is wrong in BOTH directions, and
both are measured below. `mean` and `std` are raw-feature statistics in METRES, so a translation --
exactly what a coordinate or geometry correction produces -- moves them while every served
probability stays identical; and a change confined to standardisation moves served probabilities
while the coefficients sit byte-still. The house answer to "test behaviour, not parameters" already
exists twice (`_chirality.py` fingerprints model output on a fixed probe, `_feature_contract.py`
fingerprints the feature vector on one); this follows it.
"""

from __future__ import annotations

import re
import sys

import numpy as np
import pandas as pd
import pytest

import scripts.train_gk_completion as mod


def _weights():
    rng = np.random.default_rng(0)
    return (
        {"coef": rng.normal(size=4), "intercept": 0.1, "mean": np.zeros(4), "std": np.ones(4)},
        rng.normal(size=(64, 4)),
    )


def test_a_pure_translation_is_NOT_a_retrain():
    """The measured defect. A geometry correction translates the raw features, so `mean` moves by
    metres while the coefficients move by ~3e-17. Standardisation absorbs it exactly, so
    `(x - mean) / std` -- and every served probability -- is unchanged. A guard keyed on ANY array
    moving calls that a retrain and stamps the artifact "retrained on X"."""
    old, probe = _weights()
    shifted = {**old, "mean": old["mean"] + 5.0}

    assert not mod.predictions_moved(old, shifted, probe_old=probe, probe_new=probe + 5.0)


def test_a_real_retrain_IS_detected():
    """Non-vacuity: the guard must reject a translation, not reject everything. Without this half,
    `return False` passes the test above."""
    old, probe = _weights()
    new = {**old, "coef": old["coef"] + 0.5}

    assert mod.predictions_moved(old, new, probe_old=probe, probe_new=probe)


def test_a_STANDARDISATION_ONLY_change_IS_detected():
    """The direction a coefficients-only fix would get wrong: the coefficients are byte-identical
    and the served probabilities still move."""
    old, probe = _weights()
    new = {**old, "std": old["std"] * 2.0}

    assert mod.predictions_moved(old, new, probe_old=probe, probe_new=probe)


def test_a_SINGLE_probe_cannot_express_the_property():
    """Pins WHY the signature takes two probes -- the one-probe form was drafted first and measured
    to fail. Serving both models on one array asks whether two functions agree on an input; the
    question is whether each model behaves the same on the coordinates IT sees."""
    old, probe = _weights()
    shifted = {**old, "mean": old["mean"] + 5.0}

    # The one-probe form: the same array to both. It reports movement where there is none.
    assert mod.predictions_moved(old, shifted, probe_old=probe, probe_new=probe)


def test_mode_and_reason_are_REQUIRED(monkeypatch):
    """No default on either: a weights change must never be reachable by accident, and must never
    ship without a stated cause.

    Three-argument `setattr`, not the two-argument string form: pytest 9.1 removed string targets.
    """
    monkeypatch.setattr(sys, "argv", ["train_gk_completion.py"])

    with pytest.raises(SystemExit):
        mod.main()


def test_retrain_REFUSES_without_a_declared_feature_space():
    """Neither default is safe -- `unchanged` and `moved` need opposite instruments -- so the
    operator declares which question is being asked."""
    args = _args(mode="retrain", feature_space=None)

    with pytest.raises(SystemExit, match="feature-space"):
        mod._validate_bundling_args(args)


def test_retrain_REFUSES_a_moved_feature_space_with_no_probe():
    """The motivating case, and the one the artifact format cannot currently serve: the weights
    directory stores coef/intercept/mean/std but NOT a design matrix. Refusing names why;
    defaulting to `probe_old = X_all` would silently answer the wrong question and stamp the
    artifact `retrained` for a change that moved nothing behavioural."""
    args = _args(mode="retrain", feature_space="moved", probe_old=None)

    with pytest.raises(SystemExit, match="probe-old"):
        mod._validate_bundling_args(args)


def test_a_REBUNDLE_needs_neither_declaration():
    """The other side. `rebundle` asks a parameter question (did the fresh fit reproduce the
    committed weights), which the artifact CAN answer, so it must not be blocked by a control that
    exists for the retrain path."""
    mod._validate_bundling_args(_args(mode="rebundle", feature_space=None))


def test_a_retrain_that_did_not_MOVE_the_model_is_refused():
    """The guard's whole purpose, exercised through the wiring rather than the pure function: a
    retrain reproducing the committed behaviour means the input change never reached the model, so
    shipping it as "retrained on X" would be a false claim."""
    old, probe = _weights()
    committed = _fake_model(old)

    with pytest.raises(SystemExit, match="never reached the model"):
        mod._assert_retrain_moved_predictions(
            _args(mode="retrain", feature_space="unchanged"), committed, _fake_model(old), probe, list("abcd")
        )


def test_a_retrain_that_DID_move_the_model_passes():
    """Non-vacuity for the wiring: it must accept a genuine retrain, not refuse everything."""
    old, probe = _weights()

    mod._assert_retrain_moved_predictions(
        _args(mode="retrain", feature_space="unchanged"),
        _fake_model(old),
        _fake_model({**old, "coef": old["coef"] + 0.5}),
        probe,
        list("abcd"),
    )


def test_the_superseded_coefficients_are_recorded():
    """A weights change is reviewable later only if what it replaced is written down -- the
    artifact keeps no history and git shows the npz as a binary blob. `None` on a first bundle."""
    old, _probe = _weights()

    assert mod._superseded_coef(None, list("abcd")) is None
    recorded = mod._superseded_coef(_fake_model(old), list("abcd"))
    assert recorded is not None
    assert list(recorded) == list("abcd")
    assert recorded["a"] == pytest.approx(float(old["coef"][0]))


# ---------------------------------------------------------------------------


def _args(**kw):
    import argparse

    return argparse.Namespace(**{"mode": "rebundle", "reason": "test", "feature_space": None, "probe_old": None, **kw})


def _fake_model(w: dict):
    """Duck-typed stand-in for `GkCompletionModel`: the guard reads four attributes and no more."""
    import types

    return types.SimpleNamespace(_coef=w["coef"], _intercept=w["intercept"], _mean=w["mean"], _std=w["std"])


# ---------------------------------------------------------------------------
# The REBUNDLE gate stays keyed on PARAMETERS -- a decision reversed by measurement (ADR-052)
# ---------------------------------------------------------------------------


def test_a_rebundle_across_a_MOVED_feature_space_must_still_abort():
    """The cycle plan called the re-bundle's parameter check a "mirror defect" and specified
    re-keying it onto served predictions. MEASURED, and the premise does not hold.

    The claim was that a geometry correction moves `mean` by metres while leaving "every served
    probability identical", so the abort is spurious. That comparison is *committed-on-OLD-features
    vs fresh-on-NEW-features* -- 1.7e-16, genuinely identical. But a re-bundle **ships the COMMITTED
    weights**, and production then serves them on the **NEW** features. That comparison is what
    actually ships, and it moves by **0.72** in probability.

    So the abort is correct: after a feature-space move the right action is `--mode retrain`, which
    ships the fresh fit and whose two-probe guard handles the moved case properly. The fence stays
    up, and this test is why.
    """
    old, X_old = _weights()
    X_new = X_old + 5.0  # the extractor moved
    committed, fresh = _fake_model(old), _fake_model({**old, "mean": old["mean"] + 5.0})

    # What the plan measured: each model on the coordinates IT saw. Identical -- and irrelevant.
    assert not mod.predictions_moved(_as_dict(committed), _as_dict(fresh), probe_old=X_old, probe_new=X_new)

    # What a re-bundle actually ships: the COMMITTED weights against the NEW features.
    served_if_rebundled = mod.predictions_moved(_as_dict(committed), _as_dict(fresh), probe_old=X_new, probe_new=X_new)
    assert served_if_rebundled, "if these agreed, re-bundling across a moved feature space would be safe"


def _as_dict(m):
    return {"coef": m._coef, "intercept": m._intercept, "mean": m._mean, "std": m._std}


# ---------------------------------------------------------------------------
# The guard against a REAL model, not a dict (spec §11.2)
# ---------------------------------------------------------------------------
#
# Every test above builds its weights from `_weights()` / `_fake_model()` -- plain dicts and a
# SimpleNamespace. That is right for exercising `predictions_moved`'s algebra, but it means the
# trainer's own bridge to the artifact, `_as_weights`, is never run against a `GkCompletionModel`.
# `_as_weights` reads four PRIVATE attributes (`_coef`, `_intercept`, `_mean`, `_std`); a rename or a
# serialisation change that dropped one would sail past every test in this file while breaking the
# bundling gate on the real path. These three close that, and the third is what makes the second
# mean anything.


def test_as_weights_matches_the_real_artifact_contract():
    """`_as_weights` against a bundled model, with the shapes tied to the feature list.

    The `len(feats)` assertions are the point: a model and a feature list that disagree would make
    `_superseded_coef`'s `zip(..., strict=True)` raise mid-run, after the corpus work is paid for.
    """
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES, GkCompletionModel

    n = len(GK_COMPLETION_FEATURE_NAMES)
    w = mod._as_weights(GkCompletionModel.from_variant("default"))

    assert set(w) == {"coef", "intercept", "mean", "std"}
    assert w["coef"].shape == (n,), f"coef/feature-list disagreement: {w['coef'].shape} vs {n}"
    assert w["mean"].shape == (n,)
    assert w["std"].shape == (n,)
    assert np.isfinite(w["intercept"])
    assert np.isfinite(w["coef"]).all() and np.isfinite(w["mean"]).all()
    assert (w["std"] > 0).all(), "a zero std would divide-by-zero at serve time"


def test_a_save_load_round_trip_serves_identically(tmp_path):
    """A byte round-trip must report NOTHING moved -- the guard's zero point on the real path."""
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES, GkCompletionModel

    original = GkCompletionModel.from_variant("default")
    original.save(tmp_path / "rt")
    reloaded = GkCompletionModel.load(tmp_path / "rt")

    probe = np.random.default_rng(0).normal(size=(64, len(GK_COMPLETION_FEATURE_NAMES)))
    w_before, w_after = mod._as_weights(original), mod._as_weights(reloaded)

    for key in ("coef", "mean", "std"):
        np.testing.assert_allclose(w_after[key], w_before[key], atol=0, rtol=0)
    assert w_after["intercept"] == w_before["intercept"]
    assert not mod.predictions_moved(w_before, w_after, probe_old=probe, probe_new=probe)


def test_the_round_trip_check_would_NOTICE_a_dropped_parameter(tmp_path):
    """Non-vacuity for the test above, which passes identically on a guard that always says False.

    A serialisation bug does not politely drop an entire array -- it perturbs one. So perturb one,
    at a magnitude small enough that a slack comparison would wave it through, and require the guard
    to catch it. Without this, `test_a_save_load_round_trip_serves_identically` is evidence of
    nothing.
    """
    from silly_kicks.tracking._gk_completion import GK_COMPLETION_FEATURE_NAMES, GkCompletionModel

    original = GkCompletionModel.from_variant("default")
    probe = np.random.default_rng(0).normal(size=(64, len(GK_COMPLETION_FEATURE_NAMES)))

    intact = mod._as_weights(original)
    damaged = {**intact, "coef": intact["coef"].copy()}
    damaged["coef"][0] += 0.01

    assert mod.predictions_moved(intact, damaged, probe_old=probe, probe_new=probe), (
        "a 0.01 perturbation of one coefficient went undetected -- the round-trip test above is vacuous"
    )


# ---------------------------------------------------------------------------
# The rebundle drift abort is ACTIONABLE (spec §11.2)
# ---------------------------------------------------------------------------


def test_rebundle_accepts_drift_within_tolerance():
    """The tolerance exists for float / tracking_limit noise, so it must still admit it."""
    base, _ = _weights()
    nudged = {**base, "coef": base["coef"] + 0.9 * mod._CORPUS_IDENTITY_ATOL}
    assert mod._assert_rebundle_reproduces(_fake_model(nudged), _fake_model(base)) is None


def test_rebundle_drift_beyond_tolerance_names_the_remedy():
    """The other side -- and it checks the MESSAGE, because that is the whole improvement.

    The bare `np.testing.assert_allclose` this replaced already failed on drift; what it did not do
    was tell the operator that the answer is a retrain with a vintage-matched probe. Asserting only
    "it raised" would pass on the old behaviour and prove nothing.
    """
    base, _ = _weights()
    moved = {**base, "mean": base["mean"] + 5.0}  # a geometry correction, in metres

    with pytest.raises(SystemExit) as excinfo:
        mod._assert_rebundle_reproduces(_fake_model(moved), _fake_model(base))

    msg = str(excinfo.value)
    assert "--mode retrain" in msg and "--feature-space moved" in msg and "--probe-old" in msg
    assert "mean" in msg, "the message should name which parameter drifted worst"
    assert "do not widen the tolerance" in msg


def test_rebundle_abort_reports_the_ACTUAL_drift_of_every_parameter():
    """`assert_allclose` short-circuits on `coef` and never reports `mean`/`std`.

    Diagnosing a feature-space move needs the shape of the drift across all four -- a translation
    moves `mean` while leaving `coef` still, which is the signature that tells you which mode to
    re-run in.

    The first version of this test asserted only that each ``|param|`` label APPEARED in the message.
    That was vacuous: the message emits all four labels unconditionally, so it passed for any
    implementation that produced the text at all, including one computing the numbers wrongly. It now
    asserts the reported VALUES, which is the part a reader acts on, and that the named `worst` is
    genuinely the largest.
    """
    base, _ = _weights()
    moved = {**base, "coef": base["coef"] + 3.0, "mean": base["mean"] + 5.0}

    with pytest.raises(SystemExit) as excinfo:
        mod._assert_rebundle_reproduces(_fake_model(moved), _fake_model(base))

    msg = str(excinfo.value)
    expected = {"coef": 3.0, "intercept": 0.0, "mean": 5.0, "std": 0.0}
    for param, value in expected.items():
        assert re.search(rf"max \|{param}\| drift = {value:.6g}\b", msg), (
            f"{param} drift not reported as {value:.6g}; message was:\n{msg}"
        )
    # `mean` moved furthest, so it must be the one named -- not merely mentioned.
    assert "exceeded by 'mean'" in msg, "the abort names the wrong worst-drifting parameter"


@pytest.mark.parametrize("param", ["coef", "intercept", "mean", "std"])
def test_rebundle_gates_on_EVERY_served_parameter_in_isolation(param):
    """Drift each of the four served parameters ALONE; every one must abort.

    The existing drift tests move `coef` and `mean` (together or singly), so a guard that silently
    stopped checking `intercept` or `std` would keep them green -- and `_assert_rebundle_reproduces`
    picks its `worst` key from a collection, which is exactly the shape a well-meaning edit narrows.
    `predictions_moved` cannot substitute here: it is the RETRAIN guard, and standardisation makes
    `mean`/`std` drift partly self-cancelling in served space, which is why the rebundle path is
    deliberately kept on parameters (see the `_CORPUS_IDENTITY_ATOL` block comment).

    Each parameter is perturbed well beyond `_CORPUS_IDENTITY_ATOL` so the test is about COVERAGE,
    not about where the threshold sits.
    """
    base, _ = _weights()
    delta = 10 * mod._CORPUS_IDENTITY_ATOL
    drifted = dict(base)
    drifted[param] = base[param] + delta  # works for the float intercept and the arrays alike

    with pytest.raises(SystemExit) as excinfo:
        mod._assert_rebundle_reproduces(_fake_model(drifted), _fake_model(base))

    msg = str(excinfo.value)
    assert f"|{param}|" in msg, f"abort message does not report {param} drift"
    assert f"exceeded by {param!r}" in msg, f"{param} drifted alone but was not named as the worst"


@pytest.mark.parametrize("side", ["one_sided", "both_sided"])
@pytest.mark.parametrize("param", ["coef", "intercept", "mean", "std"])
def test_rebundle_REFUSES_non_finite_drift_in_any_position(param, side):
    """A NaN anywhere must abort, on EITHER side -- part regression guard, part strengthening.

    ONE_SIDED is the measured regression. `max(drift, key=...)` is ORDER-DEPENDENT under NaN: every
    comparison against NaN is False, so `max` keeps whichever key it was already holding and a NaN in
    any position but the first is discarded. Measured on the first version of
    `_assert_rebundle_reproduces`: a fresh-fit NaN in `intercept`, `mean` or `std` against finite
    committed weights was ACCEPTED, while the four `np.testing.assert_allclose` calls it replaced
    rejected it in all four positions. Only `coef` aborted, and only because it is first in the dict.

    BOTH_SIDED is the other half of the band, and it is where this guard is STRICTER than what it
    replaced rather than merely equal to it: `assert_allclose` defaults to `equal_nan=True`, so a NaN
    at the same index on BOTH sides was ACCEPTED by the original in all four positions (measured).
    That is precisely the case the abort message names -- a committed artifact carrying NaN met by a
    fresh fit that reproduces it -- and `GkCompletionModel.load` has no finiteness check, so nothing
    upstream rules it out. Without this leg the strengthening is unpinned and a refactor back toward
    `assert_allclose` semantics stays green.

    Either way it is the worst direction for a bundling gate: NaN weights are a degenerate fit, and
    accepting them ships the committed model while reporting that the fresh fit reproduced it.
    """
    base, _ = _weights()

    def _nan_like():  # a FRESH array per side -- the two legs must not secretly share one object
        return np.nan if param == "intercept" else np.full_like(np.asarray(base[param], dtype=float), np.nan)

    bad = {**base, param: _nan_like()}
    committed = {**base, param: _nan_like()} if side == "both_sided" else dict(base)

    if side == "both_sided":
        # Non-vacuity: pin that this leg tests a STRENGTHENING, not a restatement of the one-sided
        # leg. The predecessor ACCEPTED exactly this input; if that stops being true, the docstring
        # above and the block comment in `_assert_rebundle_reproduces` are stale.
        np.testing.assert_allclose(
            np.atleast_1d(bad[param]), np.atleast_1d(committed[param]), atol=mod._CORPUS_IDENTITY_ATOL
        )

    with pytest.raises(SystemExit) as excinfo:
        mod._assert_rebundle_reproduces(_fake_model(bad), _fake_model(committed))
    assert "non-finite" in str(excinfo.value)
    assert param in str(excinfo.value)


def test_rebundle_REFUSES_a_changed_feature_COUNT_with_a_readable_error():
    """A feature-set change must not surface as a numpy broadcast error mid-corpus-pass.

    Measured before the fix: `ValueError: operands could not be broadcast together with shapes
    (8,) (4,)`, raised from the raw subtraction and naming numpy rather than the thing that changed.
    No tolerance can reconcile a different input space, so this is its own abort with its own remedy.
    """
    base, _ = _weights()
    wider = {"coef": np.ones(8), "intercept": base["intercept"], "mean": np.zeros(8), "std": np.ones(8)}

    with pytest.raises(SystemExit) as excinfo:
        mod._assert_rebundle_reproduces(_fake_model(wider), _fake_model(base))
    msg = str(excinfo.value)
    assert "FEATURE SET changed" in msg
    assert "(8,)" in msg and "(4,)" in msg


def test_rebundle_REFUSES_an_empty_parameter_array():
    """(0,) == (0,) passes the shape check, then `np.max` raises a bare numpy ValueError.

    A degenerate artifact is not "reproduces the committed model", and the operator needs to be told
    which parameter and which side rather than reading a numpy traceback at the end of a corpus pass.
    """
    base, _ = _weights()
    empty = {"coef": np.array([]), "intercept": base["intercept"], "mean": np.array([]), "std": np.array([])}

    with pytest.raises(SystemExit, match="EMPTY"):
        mod._assert_rebundle_reproduces(_fake_model(empty), _fake_model(empty))


def test_rebundle_REFUSES_a_None_intercept():
    """`intercept` is scalar so it has no shape to check -- but it needs the same None guard.

    Without it, `abs(None - x)` raises `TypeError` and names neither the parameter nor the side. The
    three ARRAY parameters are covered by the existing `assert ... is not None` narrowing; intercept
    was exempt from both that and the shape loop.
    """
    base, _ = _weights()
    holed = {**base, "intercept": None}

    with pytest.raises(SystemExit, match="intercept"):
        mod._assert_rebundle_reproduces(_fake_model(holed), _fake_model(base))


@pytest.mark.parametrize("param", ["coef", "mean", "std"])
def test_rebundle_REFUSES_an_INF_not_only_a_nan(param):
    """The non-finite guard is about NON-FINITE, not about NaN specifically.

    The parameterized NaN test varies the POSITION but never the VALUE, so an implementation that
    special-cased `np.isnan` instead of `np.isfinite` would stay green. A one-sided `inf` yields an
    `inf` drift; `assert_allclose` would also have rejected this, so unlike the NaN case this is
    parity rather than strengthening -- which is precisely why it needs its own assertion.
    """
    base, _ = _weights()
    bad = {**base, param: np.full_like(np.asarray(base[param], dtype=float), np.inf)}

    with pytest.raises(SystemExit, match="non-finite"):
        mod._assert_rebundle_reproduces(_fake_model(bad), _fake_model(base))


@pytest.mark.parametrize(
    ("n_probe", "why"),
    [
        (1, "a 1-row probe BROADCASTS -- numpy answers silently instead of raising"),
        (17, "a short probe that happens not to broadcast would raise, but with a numpy message"),
        (200, "a long probe -- same problem, opposite direction"),
    ],
)
def test_retrain_REFUSES_a_probe_whose_ROW_COUNT_differs(n_probe, why, tmp_path):
    """Row alignment is asserted EXPLICITLY, not left to numpy's broadcast error.

    Measured, and this is why the explicit check exists: `predictions_moved` compares element-wise,
    so it was believed a mismatched probe "raises rather than answering". True for 1666-vs-3491 (the
    shape this cycle hit), FALSE for a 1-row probe, which numpy broadcasts against any corpus and
    answers silently. ADR-052's follow-up says to persist a fixed probe SAMPLE beside the weights --
    a one-row sample is exactly the shape that slips through.
    """
    base, x_all = _weights()  # x_all is (64, 4)
    probe = pd.DataFrame(np.random.default_rng(1).normal(size=(n_probe, 4)), columns=list("abcd"))
    path = tmp_path / "probe.parquet"
    probe.to_parquet(path)

    args = _args(mode="retrain", feature_space="moved", probe_old=str(path))
    with pytest.raises(SystemExit, match="rows but this run"):
        mod._assert_retrain_moved_predictions(
            args,
            _fake_model(base),
            _fake_model({**base, "coef": base["coef"] + 0.5}),
            pd.DataFrame(x_all, columns=list("abcd")),
            list("abcd"),
        )


def test_retrain_ACCEPTS_a_row_ALIGNED_probe(tmp_path):
    """The other side: the check must not reject the legitimate case it was added around.

    This supplies a REAL probe FILE whose row count matches, rather than `probe_old=None`. The first
    version passed `None`, which takes the `else X_all` fallback -- so the counts matched by
    construction and the aligned-probe path, the one the guard actually sits on, was never executed.
    An acceptance half that cannot exercise the accepted path is not an acceptance half.
    """
    base, x_all = _weights()
    x_df = pd.DataFrame(x_all, columns=list("abcd"))

    probe = pd.DataFrame(np.random.default_rng(7).normal(size=x_all.shape), columns=list("abcd"))
    path = tmp_path / "aligned.parquet"
    probe.to_parquet(path)
    args = _args(mode="retrain", feature_space="moved", probe_old=str(path))

    mod._assert_retrain_moved_predictions(
        args, _fake_model(base), _fake_model({**base, "coef": base["coef"] + 0.5}), x_df, list("abcd")
    )
