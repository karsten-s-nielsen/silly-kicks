"""Feature-contract gates (ADR-050).

The contract records what a model's feature extractor produces on a fixed probe frame, plus the
geometry constants that extractor consumes, so a later library change that silently alters trained
features makes ``load()`` raise instead of serving skewed values.
"""

from __future__ import annotations

import warnings

import pytest

import silly_kicks.spadl.config as spadlconfig


@pytest.mark.parametrize("name", ["MissingFeatureContractWarning", "UnverifiableFeatureContractWarning"])
def test_warning_category_is_registered_on_every_public_surface(name):
    """The value of a named category is a STABLE import path. Registering it in one list and
    forgetting the others is the classic failure."""
    import silly_kicks.tracking as T
    import silly_kicks.tracking._warnings as W

    cls = getattr(W, name)
    assert issubclass(cls, UserWarning)
    assert name in W.__all__
    assert name in T.__all__
    assert getattr(T, name) is cls


def test_the_warning_can_be_escalated_to_an_error_by_category():
    """A batch consumer sets filterwarnings('error', ...) and gets fail-closed semantics without
    silly-kicks changing its default."""
    from silly_kicks.tracking import MissingFeatureContractWarning

    with warnings.catch_warnings():
        warnings.simplefilter("error", MissingFeatureContractWarning)
        with pytest.raises(MissingFeatureContractWarning):
            warnings.warn("x", MissingFeatureContractWarning, stacklevel=2)


def test_the_two_categories_are_independent():
    """Neither may subclass the other. If Unverifiable were a subclass of Missing, escalating
    Missing would ALSO escalate probe changes -- silently undoing the warn-and-skip decision that
    lets a probe be extended without bricking every not-yet-re-saved artifact. A subclass
    relationship is exactly how someone would 'tidy' these two later."""
    from silly_kicks.tracking import (
        MissingFeatureContractWarning,
        UnverifiableFeatureContractWarning,
    )

    assert not issubclass(UnverifiableFeatureContractWarning, MissingFeatureContractWarning)
    assert not issubclass(MissingFeatureContractWarning, UnverifiableFeatureContractWarning)


# --------------------------------------------------------------------------------------------
# The probe
# --------------------------------------------------------------------------------------------


def test_probe_is_deterministic():
    """The probe hash is what decides whether a stored fingerprint is comparable at all. If the
    frame varied between calls, every load would see a 'probe changed' warning and skip the
    fingerprint check -- the guard would degrade to nothing without ever failing."""
    import pandas as pd

    from silly_kicks.tracking._feature_contract import contract_probe_frame

    a, b = contract_probe_frame(), contract_probe_frame()
    pd.testing.assert_frame_equal(a, b)
    assert a is not b, "must return a fresh frame; a shared object could be mutated by a caller"


def test_probe_is_nan_free_for_all_three_extractors():
    """A NaN feature is a feature the contract cannot gate at all."""
    import numpy as np

    from silly_kicks.tracking._feature_contract import contract_probe_frame
    from silly_kicks.tracking._ghost_gk import GHOST_GK_FEATURE_NAMES, extract_ghost_gk_features
    from silly_kicks.tracking._xcross_attempt import extract_xcross_features
    from silly_kicks.tracking._xshot_occurrence import extract_xshot_features

    f = contract_probe_frame()

    g = extract_ghost_gk_features(
        f,
        gk_team_id="B",
        goal_x=105.0,
        score_diff=1,
        phase=0,
        ball_carrier_team_id="A",
        prev_defensive_line_x=90.0,
        prev_defending_centroid_x=94.0,
        dt=0.04,
    )
    assert not [n for n in GHOST_GK_FEATURE_NAMES if not np.isfinite(float(g[n].iloc[0]))]

    s = extract_xshot_features(f, gk_team_id="B", goal_x=105.0).iloc[0]
    assert not [c for c in s.index if not np.isfinite(float(s[c]))]

    c = extract_xcross_features(f, gk_team_id="B", goal_x=105.0, carrier_player_id="A2", score_differential=1.0).iloc[0]
    assert not [col for col in c.index if not np.isfinite(float(c[col]))]


def test_probe_makes_the_box_constant_load_bearing(monkeypatch):
    """The single most load-bearing gate here. Asserted THROUGH the real extractor: re-implementing
    ghost's predicate inside the test would pass no matter what ``extract_ghost_gk_features``
    actually does, including if its ``<`` became ``<=``.

    Measured: 0 at the shipped 40.3, 1 after flipping to 40.32."""
    import silly_kicks.tracking._ghost_gk as gg
    from silly_kicks.tracking._feature_contract import contract_probe_frame

    kw = {
        "gk_team_id": "B",
        "goal_x": 105.0,
        "score_diff": 1,
        "phase": 0,
        "ball_carrier_team_id": "A",
        "prev_defensive_line_x": 90.0,
        "prev_defending_centroid_x": 94.0,
        "dt": 0.04,
    }
    before = int(gg.extract_ghost_gk_features(contract_probe_frame(), **kw)["attackers_in_box"].iloc[0])

    # Flip the CANONICAL constant: ghost's own `_PENALTY_AREA_*` were deleted when its predicate
    # and declaration migrated onto `spadlconfig` (ADR-050 §6). Direction is reversed accordingly --
    # the baseline is now the 20.16 box, and shrinking it to 20.15 moves the probe the other way.
    monkeypatch.setattr(spadlconfig, "penalty_area_half_width", 40.3 / 2.0)
    after = int(gg.extract_ghost_gk_features(contract_probe_frame(), **kw)["attackers_in_box"].iloc[0])

    assert (before, after) == (1, 0)


# --------------------------------------------------------------------------------------------
# feature_contract / verify_feature_contract
# --------------------------------------------------------------------------------------------


def _fc(**over):
    """A minimal valid contract dict, overridable per-test."""
    return {
        "version": "feature-contract-1",
        "probe_sha256": "abc123",
        "fingerprint": [1.0, 2.0, 3.0],
        "constants": {"penalty_area_half_width": 20.16},
        **over,
    }


class _FakeIntegrityError(Exception):
    pass


def test_round_trip_on_an_unmodified_contract_passes():
    """Guards a real trap: with chirality's equal_nan=False this fails on a vector against ITSELF
    whenever the vector contains a NaN (measured: 3 structurally-NaN ghost features)."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    verify_feature_contract(_fc(), _fc(), legacy_override=False, model_name="m", error_cls=_FakeIntegrityError)


def test_fingerprint_mismatch_raises():
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_FakeIntegrityError, match="feature contract"):
        verify_feature_contract(
            _fc(fingerprint=[1.0, 2.0, 9.0]),
            _fc(),
            legacy_override=False,
            model_name="m",
            error_cls=_FakeIntegrityError,
        )


def test_missing_contract_warns_and_does_not_raise():
    """Asserted BY CATEGORY, not by message text."""
    from silly_kicks.tracking import MissingFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.warns(MissingFeatureContractWarning):
        verify_feature_contract(_fc(), None, legacy_override=False, model_name="m", error_cls=_FakeIntegrityError)


def test_probe_change_warns_and_skips_the_fingerprint():
    """UNVERIFIABLE, not MISSING -- and the distinction is load-bearing, see the test below."""
    from silly_kicks.tracking import UnverifiableFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(
            _fc(probe_sha256="NEW", fingerprint=[9.0, 9.0, 9.0]),
            _fc(),
            legacy_override=False,
            model_name="m",
            error_cls=_FakeIntegrityError,
        )


def test_escalating_the_missing_category_does_not_brick_a_probe_change():
    """THE reason for two categories.

    Warn-and-skip on a probe change exists precisely so that adding a constant -- which REQUIRES
    extending the probe -- does not hard-fail every not-yet-re-saved artifact. A batch consumer
    that escalates the missing-contract category must still get that behaviour. One umbrella
    category would silently undo the design decision.
    """
    import warnings as _w

    from silly_kicks.tracking import MissingFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with _w.catch_warnings():
        _w.simplefilter("ignore")
        _w.filterwarnings("error", category=MissingFeatureContractWarning)
        # a probe change must still merely warn under that filter
        verify_feature_contract(
            _fc(probe_sha256="NEW", fingerprint=[9.0, 9.0, 9.0]),
            _fc(),
            legacy_override=False,
            model_name="m",
            error_cls=_FakeIntegrityError,
        )
        # ...while a genuinely missing contract now raises, which is what escalation bought
        with pytest.raises(MissingFeatureContractWarning):
            verify_feature_contract(_fc(), None, legacy_override=False, model_name="m", error_cls=_FakeIntegrityError)


def test_probe_change_PLUS_constant_change_still_raises():
    """The two nets must not cancel. Skipping the fingerprint must NOT skip constants."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_FakeIntegrityError, match="constant"):
        verify_feature_contract(
            _fc(probe_sha256="NEW", constants={"penalty_area_half_width": 20.15}),
            _fc(),
            legacy_override=False,
            model_name="m",
            error_cls=_FakeIntegrityError,
        )


def test_constant_change_alone_raises_even_when_the_fingerprint_matches():
    """A sub-probe-resolution change (20.16 -> 20.161) moves no feature, so only the declared
    constant can catch it. This is where the constants prong is isolated."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    with pytest.raises(_FakeIntegrityError, match="constant"):
        verify_feature_contract(
            _fc(constants={"penalty_area_half_width": 20.161}),
            _fc(),
            legacy_override=False,
            model_name="m",
            error_cls=_FakeIntegrityError,
        )


def test_new_constant_keys_are_additive_and_removed_keys_warn():
    from silly_kicks.tracking import UnverifiableFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    verify_feature_contract(  # new key on the library side -> ignored
        _fc(constants={"penalty_area_half_width": 20.16, "new_thing": 1.0}),
        _fc(),
        legacy_override=False,
        model_name="m",
        error_cls=_FakeIntegrityError,
    )
    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(
            _fc(constants={}), _fc(), legacy_override=False, model_name="m", error_cls=_FakeIntegrityError
        )


@pytest.mark.parametrize("kind", ["fingerprint", "constants"])
def test_legacy_override_escapes_EITHER_mismatch_with_a_warning(kind):
    """The constants branch used to fall through SILENTLY while the fingerprint branch warned.
    Parametrized so neither can regress without the other noticing."""
    from silly_kicks.tracking import UnverifiableFeatureContractWarning
    from silly_kicks.tracking._feature_contract import verify_feature_contract

    over = (
        {"fingerprint": [9.0, 9.0, 9.0]} if kind == "fingerprint" else {"constants": {"penalty_area_half_width": 20.15}}
    )
    with pytest.warns(UnverifiableFeatureContractWarning):
        verify_feature_contract(_fc(**over), _fc(), legacy_override=True, model_name="m", error_cls=_FakeIntegrityError)


def test_the_raised_type_is_the_MODELs_own_integrity_error():
    """``error_cls`` exists so a consumer catching ``_ghost_gk.IntegrityError`` catches this too --
    the same reasoning ADR-040 gives for chirality. Asserted, not assumed."""
    from silly_kicks.tracking._feature_contract import verify_feature_contract
    from silly_kicks.tracking._ghost_gk import IntegrityError

    with pytest.raises(IntegrityError):
        verify_feature_contract(
            _fc(fingerprint=[9.0, 9.0, 9.0]),
            _fc(),
            legacy_override=False,
            model_name="GhostGk",
            error_cls=IntegrityError,
        )


def test_builder_refuses_a_non_finite_feature_vector():
    """Enforces the zero-NaN policy at SAVE time by construction, mirroring chirality's own
    non-finite guard -- not merely in a test."""
    import numpy as np

    from silly_kicks.tracking._feature_contract import feature_contract

    with pytest.raises(ValueError, match="non-finite"):
        feature_contract(lambda: np.array([1.0, np.nan]), constants={})


# --------------------------------------------------------------------------------------------
# Wiring into the three models
# --------------------------------------------------------------------------------------------


def test_xshot_records_and_verifies_a_contract(tmp_path, monkeypatch):
    """save() stamps it; load() on the same library passes; a LIBRARY change then raises.

    Mutate the LIBRARY, never the artifact on disk: ``load()`` verifies SHA256SUMS BEFORE parsing
    metadata, so editing metadata.json raises ``IntegrityError("Integrity check failed for
    metadata.json")`` and no domain ``match=`` could ever fire. Mutating the library is also the
    truer test -- the real failure mode is a library change under a fixed artifact; tampering is
    already the SHA check's job.
    """
    import json

    import silly_kicks.tracking._xshot_occurrence as xs_mod
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    m = XShotOccurrenceModel.from_variant("default")
    out = tmp_path / "xs"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert "feature_contract" in meta
    # xS declares GOAL WIDTH, not the penalty area: `_xshot_occurrence.py` contains no
    # penalty-area constant or predicate at all.
    assert meta["feature_contract"]["constants"] == {"goal_width": 7.32}

    XShotOccurrenceModel.load(out)  # round-trip on the unmodified artifact is clean

    # Patching GOAL_WIDTH moves BOTH the declared constant and the fingerprint -- MEASURED:
    # `_open_goal_fraction` reads GOAL_WIDTH directly as the denominator, and on this probe
    # `openGoal` is 0.996352 (not a saturated 1.0), so it shifts to 0.996439. `match="constant"`
    # still pins the CONSTANTS branch because constants are compared FIRST and raise there.
    monkeypatch.setattr(xs_mod, "GOAL_WIDTH", 7.5)
    with pytest.raises(Exception, match="constant"):
        XShotOccurrenceModel.load(out)


def test_xcross_records_and_verifies_a_contract(tmp_path, monkeypatch):
    """Mirror of the xS test; see its docstring for why the library is mutated, not the artifact."""
    import json

    import silly_kicks.tracking._xcross_attempt as xc_mod
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    m = XCrossAttemptModel.from_variant("default")
    out = tmp_path / "xc"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert set(meta["feature_contract"]["constants"]) == {
        "penalty_area_half_width",
        "penalty_area_depth",
        "goal_width",
    }

    XCrossAttemptModel.load(out)

    monkeypatch.setattr(xc_mod, "_BOX_HALF_WIDTH_M", 20.15)
    with pytest.raises(Exception, match="constant"):
        XCrossAttemptModel.load(out)


def test_ghost_records_pitch_dims_and_a_contract(tmp_path, monkeypatch):
    """Mutate the LIBRARY, not the artifact -- see the xS test's docstring.

    Also covers the pitch-dims guard ghost was missing entirely: xS and xCross have recorded
    pitch_length/pitch_width since their first release, ghost did not.
    """
    import json

    from silly_kicks.tracking import _geometry as _geo
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    m = GhostGkModel.from_variant("default")
    out = tmp_path / "ghost"
    m.save(out)
    meta = json.loads((out / "metadata.json").read_text())
    assert meta["pitch_length"] == 105.0 and meta["pitch_width"] == 68.0
    # The divergence this line used to record is CLOSED: ghost was re-fit onto the canonical box,
    # so its declared half-width is 20.16. Recording the divergence WAS the point, and the raise it
    # produced on an unaccompanied flip is what forced the re-fit rather than letting the constant
    # move under trained weights.
    #
    # Deliberately a LITERAL, not `spadlconfig.penalty_area_half_width`. `test_declared_constant_values`
    # already asserts artifact-equals-canonical; if this one also read the canonical source, both
    # would follow a change to `spadlconfig` in silence. One hard-coded pin is what catches "the
    # canonical constant itself moved".
    assert meta["feature_contract"]["constants"]["penalty_area_half_width"] == 20.16
    assert meta["feature_contract"]["constants"]["penalty_area_depth"] == 16.5

    GhostGkModel.load(out)

    monkeypatch.setattr(_geo, "PITCH_LENGTH", 100.0)
    with pytest.raises(Exception, match=r"[Pp]itch"):
        GhostGkModel.load(out)


# --------------------------------------------------------------------------------------------
# The bundled artifacts
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls_path", "weights_dir", "n_constants"),
    [
        ("silly_kicks.tracking._ghost_gk:GhostGkModel", "_ghost_gk_weights", 2),
        ("silly_kicks.tracking._xshot_occurrence:XShotOccurrenceModel", "_xshot_weights", 1),
        ("silly_kicks.tracking._xcross_attempt:XCrossAttemptModel", "_xcross_weights", 3),
    ],
)
def test_every_bundled_artifact_carries_a_verified_contract(cls_path, weights_dir, n_constants, recwarn):
    """Every bundled artifact is stamped, so NO bundled load warns -- which is what lets CI
    escalate the missing-contract category with an empty opt-out list.

    Asserted as ABSENCE of the warning, not merely as "it loaded": a contract that exists but
    cannot be verified warns rather than raising on some paths, and would otherwise pass here.
    """
    import importlib
    import json
    from pathlib import Path

    from silly_kicks.tracking import MissingFeatureContractWarning

    mod_name, cls_name = cls_path.split(":")
    cls = getattr(importlib.import_module(mod_name), cls_name)

    assert cls.from_variant("default") is not None
    assert not [w for w in recwarn if issubclass(w.category, MissingFeatureContractWarning)]

    tracking_file = importlib.import_module("silly_kicks.tracking").__file__
    assert tracking_file is not None
    root = Path(tracking_file).parent
    meta = json.loads((root / weights_dir / "default" / "metadata.json").read_text(encoding="utf-8"))
    assert len(meta["feature_contract"]["constants"]) == n_constants


def test_ghost_pin_is_enforced_by_a_raise_not_by_prose(monkeypatch):
    """Flipping ghost's box constant without re-fitting must RAISE, because that is the exact skew
    the "do not unify before the re-fit" instruction exists to prevent. Before this artifact
    carried a contract, only a docstring said so -- and a docstring is deletable by whoever is
    "finishing the job"."""
    from silly_kicks.tracking._ghost_gk import GhostGkModel

    # ghost has no _VARIANT_CACHE (verified), so from_variant re-runs load() every call.
    # Flip the CANONICAL constant -- ghost's own were deleted at the ADR-050 §6 closure.
    #
    # The value is deliberately one NO artifact has ever been stamped with. Using "the other era's"
    # constant makes this test era-dependent and silently self-disarming: patching to 20.15 while
    # the bundled artifact still declares 20.15 MATCHES, so `load()` succeeds and the test fails
    # `DID NOT RAISE` -- observed while migrating. A never-stamped value diverges from whatever is
    # on disk, before or after the re-fit.
    monkeypatch.setattr(spadlconfig, "penalty_area_half_width", 19.0)
    with pytest.raises(Exception, match=r"constant|feature contract"):
        GhostGkModel.from_variant("default")


def test_xshot_pin_is_enforced_by_a_raise(monkeypatch):
    """The _VARIANT_CACHE clear is load-bearing: xS memoizes bundled loads, so a from_variant call
    earlier in the session returns the cached instance WITHOUT re-running load() -- the raise would
    never fire and this test would pass vacuously."""
    import silly_kicks.tracking._xshot_occurrence as xs_mod
    from silly_kicks.tracking._xshot_occurrence import XShotOccurrenceModel

    monkeypatch.setattr(xs_mod, "_VARIANT_CACHE", {})
    monkeypatch.setattr(xs_mod, "GOAL_WIDTH", 7.5)
    with pytest.raises(Exception, match=r"constant|feature contract"):
        XShotOccurrenceModel.from_variant("default")


def test_xcross_pin_is_enforced_by_a_raise(monkeypatch):
    """See the xS test for why _VARIANT_CACHE must be cleared."""
    import silly_kicks.tracking._xcross_attempt as xc_mod
    from silly_kicks.tracking._xcross_attempt import XCrossAttemptModel

    monkeypatch.setattr(xc_mod, "_VARIANT_CACHE", {})
    monkeypatch.setattr(xc_mod, "_BOX_HALF_WIDTH_M", 20.15)
    with pytest.raises(Exception, match=r"constant|feature contract"):
        XCrossAttemptModel.from_variant("default")


# --------------------------------------------------------------------------------------------
# The CI escalation
# --------------------------------------------------------------------------------------------


def test_the_escalation_is_live_not_decorative():
    """Non-vacuity: if the pyproject filterwarnings line were dropped or misspelled, this whole
    mechanism would be inert and the suite would pass identically. Assert the other side."""
    from silly_kicks.tracking import MissingFeatureContractWarning

    with pytest.raises(MissingFeatureContractWarning):
        warnings.warn("probe", MissingFeatureContractWarning, stacklevel=2)


def test_the_unverifiable_category_is_NOT_escalated():
    """The other half of the same decision, and the one a future 'tidy-up' would break: escalating
    both categories would make any probe extension a hard failure across every artifact.

    NOTE the bare ``warnings.warn`` with NO recording context. ``catch_warnings(record=True)``,
    ``recwarn`` and ``pytest.warns`` all call ``simplefilter("always")``, which OVERRIDES the ini
    filterwarnings config -- so any of them would make this test pass whether or not the category
    is escalated. Asserting that this line simply does not raise is the only form with
    discriminating power.
    """
    from silly_kicks.tracking import UnverifiableFeatureContractWarning

    warnings.warn("probe", UnverifiableFeatureContractWarning, stacklevel=2)
