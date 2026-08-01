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

import sys

import numpy as np
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
