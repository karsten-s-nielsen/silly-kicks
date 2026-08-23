"""Task 3: the per-candidate ReceiverModel + its ADR-011 bundle (SHA + chirality + feature contract)."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from silly_kicks.tracking import _receiver
from silly_kicks.tracking._receiver import (
    IntegrityError,
    ReceiverModel,
    ReceiverParams,
    variant_key_for_provider,
)

_ATT, _DEF = 1, 2


def _fitted_public() -> ReceiverModel:
    # separable synthetic: the intended receiver is open (low lane_pressure, high space)
    intended = pd.DataFrame({"ball_dist": [15.0] * 15, "lane_pressure": [0.0] * 15, "space": [12.0] * 15})
    others = pd.DataFrame({"ball_dist": [15.0] * 15, "lane_pressure": [1.5] * 15, "space": [4.0] * 15})
    X = pd.concat([intended, others], ignore_index=True)
    y = np.array([1] * 15 + [0] * 15)
    return ReceiverModel("public").fit(X, y)


def _frame() -> pd.DataFrame:
    rows = [
        (True, pd.NA, pd.NA, False, 50.0, 34.0, 10.0, 0.0),
        (False, 9, _ATT, False, 50.0, 34.0, 0.0, 0.0),  # passer
        (False, 10, _ATT, False, 70.0, 34.0, 0.0, 0.0),  # marked (defender interposed)
        (False, 11, _ATT, False, 55.0, 50.0, 0.0, 0.0),  # open
        (False, 12, _ATT, False, 55.0, 20.0, 0.0, 0.0),  # open
        (False, 20, _DEF, False, 60.0, 34.0, 0.0, 0.0),
        (False, 30, _DEF, True, 100.0, 34.0, 0.0, 0.0),
    ]
    df = pd.DataFrame(rows, columns=["is_ball", "player_id", "team_id", "is_goalkeeper", "x", "y", "vx", "vy"])
    df["game_id"], df["period_id"], df["frame_id"] = 1, 1, 100
    return df.astype({"player_id": "Int64", "team_id": "Int64"})


def _action() -> pd.Series:
    return pd.Series({"player_id": 9, "team_id": _ATT, "start_x": 50.0, "start_y": 34.0})


def test_rank_prefers_open_teammates_over_the_marked_one():
    ranked = _fitted_public().rank(_action(), _frame())
    assert ranked["11"] > ranked["10"] and ranked["12"] > ranked["10"]  # open beats marked
    assert ranked.index[0] != "10"  # the argmax intended receiver is not the marked teammate


def test_save_load_round_trip_bit_for_bit(tmp_path):
    m = _fitted_public()
    m.save(tmp_path)
    loaded = ReceiverModel.load(tmp_path)
    X = pd.DataFrame({"ball_dist": [15.0, 15.0], "lane_pressure": [0.0, 2.0], "space": [12.0, 3.0]})
    np.testing.assert_array_equal(loaded.predict_candidates(X), m.predict_candidates(X))
    assert loaded.feature_set == "public"


def _resave_with_edit(path, edit_fn):
    d = json.loads((path / "model.json").read_text(encoding="utf-8"))
    edit_fn(d)
    (path / "model.json").write_text(json.dumps(d, indent=2), encoding="utf-8")
    text = (path / "model.json").read_text(encoding="utf-8").replace("\r\n", "\n")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (path / "SHA256SUMS").write_text(f"{sha}  model.json\n", encoding="utf-8")


def test_load_raises_on_chirality_mismatch(tmp_path):
    _fitted_public().save(tmp_path)
    # corrupt the stored chirality outputs (+ fix SHA) -> recomputed != stored -> raise
    _resave_with_edit(tmp_path, lambda d: d["chirality"].__setitem__("outputs", [0.99, 0.01]))
    with pytest.raises(IntegrityError):
        ReceiverModel.load(tmp_path)


def test_load_raises_on_feature_contract_constant_mismatch(tmp_path):
    _fitted_public().save(tmp_path)
    _resave_with_edit(tmp_path, lambda d: d["feature_contract"]["constants"].__setitem__("lane_half_width_m", 999.0))
    with pytest.raises(IntegrityError):
        ReceiverModel.load(tmp_path)


def test_feature_contract_fingerprint_is_non_empty_and_has_teeth(tmp_path):
    """H1 (review): the ADR-011 feature-VALUE prong must not be vacuous. If the probe passer's team is
    taken from the ball (team_id None on the contract probe) it has zero teammates -> an EMPTY
    fingerprint -> ``allclose([], [])`` is True and the prong never fires. Assert the stored fingerprint
    is non-empty AND that a perturbed one is rejected at load."""
    _fitted_public().save(tmp_path)
    d = json.loads((tmp_path / "model.json").read_text(encoding="utf-8"))
    assert d["feature_contract"]["fingerprint"], "contract fingerprint is EMPTY -- the feature-value prong is vacuous"
    _resave_with_edit(
        tmp_path,
        lambda d: d["feature_contract"].__setitem__(
            "fingerprint", [float(v) + 1.0 for v in d["feature_contract"]["fingerprint"]]
        ),
    )
    with pytest.raises(IntegrityError):
        ReceiverModel.load(tmp_path)


def test_non_default_params_round_trip_and_load(tmp_path):
    """M2 (review): a model fit with non-default ``ReceiverParams`` must save AND load. Without restoring
    ``params`` in ``from_dict``, the reloaded model recomputes the contract with DEFAULT constants and
    ``verify_feature_contract`` raises -- a bundle that cannot load. Params must survive the round trip."""
    m = _fitted_public()
    m.params = ReceiverParams(lane_half_width_m=5.0, pressure_scale_m=6.0)
    m.save(tmp_path)
    loaded = ReceiverModel.load(tmp_path)  # must NOT raise
    assert loaded.params.lane_half_width_m == 5.0 and loaded.params.pressure_scale_m == 6.0


def test_bundled_default_variant_loads_and_is_positions_only():
    """D1 (ADR-066): the SHIPPED ``default`` bundle loads through the full ADR-011 guard chain (SHA +
    chirality + feature contract) and is the positions-only public model -- the always-default variant."""
    _receiver._VARIANT_CACHE.clear()
    m = ReceiverModel.from_variant("default")  # raises IntegrityError if the shipped bundle is bad
    assert m.feature_set == "public"
    assert m.feature_names == ["ball_dist", "lane_pressure", "space"]
    assert variant_key_for_provider("statsbomb") == "default"


def test_from_variant_degrades_unbundled_provider_keys_to_default():
    """D2 (ADR-066): ``variant_key_for_provider`` can return ``gs_owner``/``skillcorner``, but no owner
    variant earned bundling -- ``from_variant`` DEGRADES those keys to the shipped ``default`` (warns),
    while an UNKNOWN key still RAISES so a typo is never silently served the default."""
    _receiver._VARIANT_CACHE.clear()
    default = ReceiverModel.from_variant("default").to_dict()
    for key in ("gs_owner", "skillcorner"):
        _receiver._VARIANT_CACHE.clear()
        with pytest.warns(UserWarning, match="resolved against what actually ships"):
            degraded = ReceiverModel.from_variant(key)
        assert degraded.to_dict() == default  # byte-identical to the shipped default bundle
    _receiver._VARIANT_CACHE.clear()
    with pytest.raises(FileNotFoundError):
        ReceiverModel.from_variant("bogus_typo")
