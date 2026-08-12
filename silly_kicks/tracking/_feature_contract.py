"""Trained-model FEATURE contract (ADR-050).

Sibling of :mod:`._chirality` -- probe, fingerprint, verify-at-load -- with three deliberate policy
differences: a MISSING contract warns rather than raising (pre-contract artifacts are undeclared,
not known-bad); a PROBE change warns and skips the fingerprint comparison ONLY; a fingerprint or
declared-constant mismatch RAISES.

The problem it solves: a model's serving path can be changed by editing a geometry constant far
from the model, with no signal at all. Chirality catches a y-mirror in the model's OUTPUT; it does
not catch a 1 cm penalty-area change that shifts one input feature. This records both the feature
vector on a fixed probe frame AND the constants the extractor consumed, so either kind of drift
makes ``load()`` fail loudly instead of serving skewed values.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Callable

import numpy as np
import pandas as pd

from ._warnings import MissingFeatureContractWarning, UnverifiableFeatureContractWarning

__all__ = [
    "DECLARED_CONSTANT_SOURCES",
    "contract_probe_frame",
    "feature_contract",
    "verify_feature_contract",
]

_CONTRACT_VERSION = "feature-contract-1"

# Tolerance is CHOSEN, not inherited from chirality. Chirality's rtol=1e-2 was sized for a gross
# sign flip on a probability; a feature vector spans metres, counts and radians, where rtol=1e-2 on
# a ~17 m feature is a 0.17 m blind spot -- 17x the 0.01 m change this contract exists to catch.
# The atol is pending a measured DGX-vs-x86 delta; until then every fingerprinted artifact is
# produced on x86 so no cross-platform comparison happens against an unvalidated tolerance.
_CONTRACT_ATOL = 1e-6
_CONTRACT_RTOL = 0.0

#: Module-level geometry constants that ARE declared in some model's feature contract, mapped to
#: the contract key they are declared under. The enumeration gate
#: (tests/tracking/test_geometry_constant_enumeration.py) requires every geometry-named module
#: constant to appear here or in that test's _EXEMPT list, with a reason.
#:
#: Keyed by BARE name, not module-qualified: `_BOX_HALF_WIDTH_M` / `_BOX_DEPTH_M` exist in both
#: `_xcross_attempt.py` and `defensive_credit/_params.py` and are the same quantity from the same
#: canonical source. If a future module gives one of these names a different meaning, this must
#: become module-qualified.
DECLARED_CONSTANT_SOURCES: dict[str, str] = {
    # penalty area
    # xCross's aliases. These are now LOAD-BEARING beyond xCross: ghost's own three entries were
    # pruned when it migrated onto the canonical source (ADR-050 §6), so these are the only module
    # constants left mapping to `penalty_area_half_width` / `penalty_area_depth` -- and models still
    # STAMP both keys, which `test_the_registry_and_the_built_contracts_agree` requires to be
    # covered here. A future cycle migrating xCross the same way would empty the registry and fail
    # that assertion from the other direction; `test_declared_constant_values.py` is what should
    # absorb the responsibility if that happens.
    "_BOX_HALF_WIDTH_M": "penalty_area_half_width",
    "_BOX_DEPTH_M": "penalty_area_depth",
    # goal mouth -- these drive `openGoal`, so a goal-width change skews xS exactly the way a box
    # change skews ghost. Same class, same treatment.
    "GOAL_WIDTH": "goal_width",
    "GOAL_Y_MIN": "goal_width",
    "GOAL_Y_MAX": "goal_width",
    "_GOAL_HALF_WIDTH_M": "goal_width",
}

_BASE = {"game_id": "fc", "period_id": 1, "frame_id": 1, "time_seconds": 10.0, "is_ball": False}


def _player(pid, team, x, y, *, gk=False, vx=0.7, vy=-0.4):
    return dict(_BASE, team_id=team, player_id=pid, x=x, y=y, z=0.0, vx=vx, vy=vy, is_goalkeeper=gk)


def contract_probe_frame() -> pd.DataFrame:
    """One synthetic frame. Team A attacks the goal at x=105; team B defends it and has the keeper.

    EVERY element here is load-bearing -- MEASURED, do not "simplify":

    * **A1 at (90.0, 13.845)** is the discriminating row: gr_x = 15.0 is inside the 16.5 m depth,
      and y = 13.845 is inside ``[13.84, ...]`` but outside ``[13.85, ...]``, so ghost's
      ``attackers_in_box`` is 1 at half-width 20.16 and 0 at 20.15. Being in the y-band ALONE is
      not enough -- of 844 band rows on a real match only 70 were also within depth.
    * **five attackers and five defenders**: xS's nearest-k fills ``DefDist_0..4`` and
      ``OffDist_0..4``; with 4 and 3 the extractor returned 7 NaN features (measured).
    * **A2..A5 sit well outside the box** so the 0-vs-1 discrimination stays clean.
    * **>=3 non-collinear defenders** make ghost's ConvexHull compactness feature finite.
    * **a ball row carrying** ``z``: without it xS's ``z`` feature is NaN (measured).

    Examples
    --------
    >>> frame = contract_probe_frame()
    >>> len(frame), int(frame["is_ball"].sum())
    (12, 1)
    """
    rows = [
        _player("A1", "A", 90.0, 13.845),
        _player("A2", "A", 84.0, 40.0),
        _player("A3", "A", 76.0, 28.0),
        _player("A4", "A", 64.0, 47.0),
        _player("A5", "A", 58.0, 19.0),
        _player("B1", "B", 95.0, 30.0),
        _player("B2", "B", 97.0, 44.0),
        _player("B3", "B", 92.0, 22.0),
        _player("B4", "B", 99.0, 36.0),
        _player("B5", "B", 86.0, 51.0),
        _player("BGK", "B", 103.0, 34.5, gk=True),
        dict(
            _BASE,
            team_id=None,
            player_id=None,
            x=88.0,
            y=20.0,
            z=0.6,
            vx=3.0,
            vy=1.0,
            is_goalkeeper=False,
            is_ball=True,
        ),
    ]
    frame = pd.DataFrame(rows)
    frame["is_ball"] = frame["is_ball"].astype(bool)
    return frame


def feature_contract(extract_on_probe: Callable[[], np.ndarray], *, constants: dict[str, float]) -> dict:
    """Build the contract.

    ``extract_on_probe`` is a ZERO-ARGUMENT closure the model supplies, binding its own extractor
    to :func:`contract_probe_frame`. The three extractors' signatures genuinely do not unify, so
    this module stays extractor-agnostic.

    Raises on a non-finite vector: a NaN feature is one the contract could never gate, so allowing
    it would ship a fingerprint with silent holes in it.
    """
    frame = contract_probe_frame()
    probe_sha = hashlib.sha256(json.dumps(frame.to_dict("records"), sort_keys=True, default=str).encode()).hexdigest()
    values = np.asarray(extract_on_probe(), dtype=float).ravel()
    if not np.all(np.isfinite(values)):
        raise ValueError(f"feature contract produced non-finite values: {values!r}")
    return {
        "version": _CONTRACT_VERSION,
        "probe_sha256": probe_sha,
        "fingerprint": [round(float(v), 10) for v in values],
        "constants": {k: float(v) for k, v in constants.items()},
    }


def verify_feature_contract(
    recomputed: dict,
    stored: dict | None,
    *,
    legacy_override: bool,
    model_name: str,
    error_cls: type[Exception] | None = None,
) -> None:
    """Verify at ``load()``.

    Argument order mirrors :func:`._chirality.verify_chirality` EXACTLY -- recomputed first, stored
    second. Both are plain dicts, so a swap is not a type error: it would make the ``is None``
    branch test the wrong side and silently invert the missing-contract policy.
    """
    err = error_cls or ValueError

    if stored is None:
        warnings.warn(
            f"{model_name}: artifact carries no feature contract, so its extractor cannot be "
            "verified. Loading anyway (pre-contract artifacts are undeclared, not known-bad); "
            "re-save to gain the guard.",
            MissingFeatureContractWarning,
            stacklevel=2,
        )
        return

    # Constants are PROBE-INDEPENDENT, so they are compared FIRST and always: a probe change is no
    # reason to stop comparing 20.16 against 20.15.
    rec_c = dict(recomputed.get("constants") or {})
    sto_c = dict(stored.get("constants") or {})
    removed = sorted(set(sto_c) - set(rec_c))
    if removed:
        warnings.warn(
            f"{model_name}: declared constant(s) {removed} are recorded in the artifact but no "
            "longer declared by the library; cannot compare them.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )
    # EXACT float equality, deliberately -- do NOT add a tolerance here. The change this prong
    # exists to catch is 0.01 m, so any tolerance loose enough to absorb derivation noise is within
    # an order of magnitude of the signal. It is safe today because both sides evaluate the *same
    # expression* over the same doubles, which is IEEE-deterministic (so cross-platform too), and
    # the JSON round trip preserves a float exactly.
    #
    # The one way to trip it spuriously: refactor a model to store a constant directly where it
    # previously derived it (or vice versa), so the two forms differ in the last bit. The fix then
    # is to RE-STAMP the artifact -- not to loosen this comparison.
    changed = {k: (sto_c[k], rec_c[k]) for k in set(rec_c) & set(sto_c) if rec_c[k] != sto_c[k]}
    if changed:
        if not legacy_override:
            raise err(
                f"{model_name}: declared constant mismatch {changed} (artifact value first). The "
                "features this model was trained on were computed with different geometry; "
                "refusing to load. Re-fit, or pass legacy_override=True only if independently "
                "verified."
            )
        # An override that silently swallows a CONSTANT mismatch is the worst branch here: the
        # fingerprint branch below warns, so a reader would reasonably infer this one does too.
        warnings.warn(
            f"{model_name}: declared constant mismatch {changed} suppressed by legacy_override.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )

    if recomputed.get("probe_sha256") != stored.get("probe_sha256"):
        warnings.warn(
            f"{model_name}: cannot verify the fingerprint -- the contract probe changed (stored "
            f"{str(stored.get('probe_sha256', ''))[:8]} vs library "
            f"{str(recomputed.get('probe_sha256', ''))[:8]}). Re-save to regain teeth. The "
            "declared-constant comparison above still applied.",
            UnverifiableFeatureContractWarning,
            stacklevel=2,
        )
        return

    a = np.asarray(recomputed.get("fingerprint", []), dtype=float)
    b = np.asarray(stored.get("fingerprint", []), dtype=float)
    # equal_nan=True is belt-and-braces: the builder already forbids non-finite values, so this can
    # only ever mask a case that cannot be stored. Without it, allclose(v, v) is False for any
    # vector containing a NaN -- a round trip on an unmodified artifact would fail.
    ok = a.shape == b.shape and np.allclose(a, b, atol=_CONTRACT_ATOL, rtol=_CONTRACT_RTOL, equal_nan=True)
    if not ok:
        if legacy_override:
            warnings.warn(
                f"{model_name}: feature contract mismatch overridden by legacy_override.",
                UnverifiableFeatureContractWarning,
                stacklevel=2,
            )
            return
        raise err(
            f"{model_name}: feature contract mismatch -- the library's extractor no longer "
            f"reproduces the features this artifact was trained on (atol={_CONTRACT_ATOL}, "
            f"rtol={_CONTRACT_RTOL}). Refusing to load; re-fit or pass legacy_override=True."
        )
