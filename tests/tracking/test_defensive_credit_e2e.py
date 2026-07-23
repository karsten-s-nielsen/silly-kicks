"""Owner-gated e2e: TF-51 defensive-credit family on a real GS match with a fitted xT + injected xG.

SCOPE (P-8): this is a PLUMBING / SANITY smoke -- it asserts the family runs end-to-end on real data
and produces sane sign/magnitude distributions. It is NOT xG or xT accuracy validation: the xG is a
crude distance heuristic and xT is fit on the single match. Accuracy validation is the owner-run
SkillCorner cross-check (spec section 12), not this test.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

_MATCH = "10502"
_TOKEN = os.environ.get("PINING_FOR_THE_DATA_TOKEN")
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _TOKEN, reason="owner-tier Gradient Sports data (PINING_FOR_THE_DATA_TOKEN)"),
]


def _load_loader():
    spec = importlib.util.spec_from_file_location(
        "_loader_pining", str(Path(__file__).parents[2] / "scripts" / "_loader_pining.py")
    )
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _fit_xt(actions):
    from silly_kicks.xthreat import ExpectedThreat

    return ExpectedThreat().fit(actions)


def test_defensive_credit_family_on_real_gs_match():
    from silly_kicks.spadl import config as spadlconfig
    from silly_kicks.tracking import (
        DEFENSIVE_CREDIT_RULES,
        add_defensive_credit,
        compute_bravery,
        compute_defensive_credits,
    )

    L = _load_loader()
    _prov, _m, actions, frames, _home = next(
        iter(L.load_matches(providers=["gradientsports"], match_ids={"gradientsports": [_MATCH]}))
    )
    # injected xG proxy (no lakehouse xG here): a crude shot-distance heuristic -- SANITY only, P-8.
    shot = actions["type_id"] == spadlconfig.actiontype_id["shot"]
    dist = np.hypot(105.0 - actions["start_x"], 34.0 - actions["start_y"])
    actions = actions.copy()
    actions["xg"] = np.where(shot, np.clip(0.4 * np.exp(-dist / 12.0), 0.0, 1.0), np.nan)
    xt = _fit_xt(actions)

    long = compute_defensive_credits(actions, frames, xg_column="xg", xt=xt)
    assert not long.empty, "expected some defensive credit rows on a real match"
    assert set(long["rule"]).issubset(set(DEFENSIVE_CREDIT_RULES))
    assert long["signed_value"].abs().max() <= 1.0  # sane magnitudes (xG in [0,1], xT small)

    agg = add_defensive_credit(actions, frames, xg_column="xg", xt=xt)  # P-2: no home_team_id; on-target via TF-48
    assert (agg["n_defensive_credits"].fillna(0) >= 0).all()
    assert np.isfinite(agg["defensive_credit_net"].to_numpy()).all()  # always finite

    brav = compute_bravery(actions)
    assert (brav["bravery_pct_known_domain"].dropna().between(0.0, 1.0)).all()
    assert brav["n_set_piece_crosses_faced"].sum() >= 0  # set-piece gap exposed
