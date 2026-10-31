"""Test-only helpers for the stoppage column (TF-46 Part 2 spec section 27.19).

Owner decision 2026-10-26 (R-C2 fixtures). A ``ball_state`` reader on a provider without an observed dead-ball state
(SkillCorner, Gradient Sports, Metrica) requires the ``stoppage_state`` column. ``derive_stoppage_state`` builds it,
but it refuses frames thinner than their ``frame_rate``, and most module-test fixtures are a few sparse frames.

Owner decision 2026-10-26 (models). A saved xS / xCross model records the stoppage provenance of its training frames,
and ``load()`` refuses one without it. A toy model fitted on a hand-built feature matrix has no frames, so its test
states the provenance with :func:`with_toy_stoppage_provenance`.

Owner decision 2026-10-26 (interim tests, until the R-C3 retrain commit). The shipped xS / xCross bundles carry no
provenance, so they refuse to load. A test that PINS a shipped bundle asserts the refusal instead of its body
(:func:`bundle_pinned`, :func:`assert_bundle_refused`). A PROPERTY gate (purity, mirror, dtype, SB360 observation,
goal-map consumers, glossary) still tests the real weights: :func:`legacy_bundles` loads them under
``legacy_override=True``. All three FAIL once a bundle loads without the override: the retrain landed, so remove them
and every test runs its body against the retrained bundle again.
"""

from __future__ import annotations

import functools
import warnings

import pandas as pd

from silly_kicks.tracking import STOPPAGE_RULE_VERSION, STOPPAGE_SOURCES

#: The current event stoppage source, read from the library constant (a version bump breaks this loudly).
_EVENTS_SOURCE = STOPPAGE_SOURCES[1]


def mark_all_live(frames: pd.DataFrame) -> pd.DataFrame:
    """Re-export of the canonical helper in :mod:`scripts._toy_corpora`. The toy builder needs the same
    all-live marking, and a scripts module imported by the parity subprocess must not import the tests
    package (see ``scripts/validate_hub_variants``), so the definition lives there. Kept here so existing
    tests still import ``mark_all_live`` from ``tests._stoppage_fixtures``."""
    from scripts._toy_corpora import mark_all_live as _impl

    return _impl(frames)


def toy_stoppage_provenance() -> dict:
    """The stoppage provenance of a toy training set: live event-span rows of the current rule version (a fresh dict
    per call, so a test that edits it cannot leak into another)."""
    return {"stoppage_rule_version": STOPPAGE_RULE_VERSION, "stoppage_sources": [_EVENTS_SOURCE]}


def with_toy_stoppage_provenance(model):
    """``model`` with :func:`toy_stoppage_provenance` recorded (a toy model fitted on a hand-built feature matrix);
    returns the same model, so it chains after ``fit``."""
    model.stoppage_provenance = toy_stoppage_provenance()
    return model


#: The reader-built model classes whose shipped bundles are refused until the R-C3 retrain (owner decision 2026-10-26).
REFUSED_UNTIL_RETRAIN = ("XShotOccurrenceModel", "XCrossAttemptModel")
_RETRAIN_LANDED = "loads without legacy_override: the retrained bundle landed, so remove the interim helper ({})"


def _bundle_module(cls_name: str):
    from silly_kicks.tracking import _xcross_attempt as xc
    from silly_kicks.tracking import _xshot_occurrence as xs

    return {"XShotOccurrenceModel": (xs, xs._XSHOT_WEIGHTS_ROOT), "XCrossAttemptModel": (xc, xc._XCROSS_WEIGHTS_ROOT)}[
        cls_name
    ]


def _bundled_dirs(root) -> list:
    return sorted(d for d in root.iterdir() if (d / "SHA256SUMS").exists())


def assert_bundle_refused(cls_name: str) -> bool:
    """For a test that pins a shipped bundle: ``True`` after asserting that every bundled variant of an xS / xCross
    class is refused at load (the test then stops); ``False`` for any other class (the test runs unchanged).

    Fails when an xS / xCross bundle loads: the retrain landed, so remove the call.
    """
    if cls_name not in REFUSED_UNTIL_RETRAIN:
        return False
    from silly_kicks.tracking._xshot_occurrence import IntegrityError

    mod, root = _bundle_module(cls_name)
    cls = getattr(mod, cls_name)
    dirs = _bundled_dirs(root)
    assert dirs, f"{cls_name}: no bundled variant under {root}"
    for d in dirs:
        try:
            cls.load(d)
        except IntegrityError as e:
            assert "stoppage provenance" in str(e), f"{cls_name}/{d.name} refused for another reason: {e}"
            continue
        raise AssertionError(f"{cls_name}/{d.name} " + _RETRAIN_LANDED.format("assert_bundle_refused"))
    return True


def bundle_pinned(*cls_names: str):
    """Decorator for a test that pins the shipped bundle of the named xS / xCross classes: the test asserts their
    refusal (:func:`assert_bundle_refused`) and does NOT run its body, which pins the old bundle's behaviour. The body
    stays as written, so the retrain commit only removes the decorator."""
    assert cls_names and set(cls_names) <= set(REFUSED_UNTIL_RETRAIN), cls_names

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for name in cls_names:
                assert_bundle_refused(name)

        return wrapper

    return deco


_LEGACY_CACHE: dict[str, dict] = {}


def legacy_bundles(monkeypatch) -> None:
    """For a PROPERTY gate: fill the xS / xCross ``_VARIANT_CACHE`` with the shipped bundles loaded under
    ``legacy_override=True`` (audit use: the gate tests a property of the real weights, not their training domain), for
    this test only. Loads each bundle once per session.

    Fails when a bundle loads without the override: the retrain landed, so remove the fixture use.
    """
    from silly_kicks.tracking import StoppageProvenanceWarning

    for cls_name in REFUSED_UNTIL_RETRAIN:
        mod, root = _bundle_module(cls_name)
        if cls_name not in _LEGACY_CACHE:
            cls = getattr(mod, cls_name)
            cache = {}
            for d in _bundled_dirs(root):
                with warnings.catch_warnings(record=True) as seen:
                    warnings.simplefilter("always")
                    model = cls.load(d, legacy_override=True)
                if not any(issubclass(w.category, StoppageProvenanceWarning) for w in seen):
                    raise AssertionError(f"{cls_name}/{d.name} " + _RETRAIN_LANDED.format("legacy_bundles"))
                cache[d.name] = model
            _LEGACY_CACHE[cls_name] = cache
        monkeypatch.setattr(mod, "_VARIANT_CACHE", dict(_LEGACY_CACHE[cls_name]))
