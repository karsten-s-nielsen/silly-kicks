"""The cv fold unit carries the per-provider reference (corpus widening spec 5.7, plan ruling R8).

``cv_unit`` computes each provider's training-fold rate from the fold's training positions ``tr`` (the rows behind
``train_rate = y[tr].mean()``) and the arm's row-aligned providers, normalised to licensing identities
(``sportec`` -> ``idsse``). It persists ``providers.npy`` and ``train_rate_by_provider``; ``cv_skill`` builds
``skill["by_provider"]`` from them.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts._corpus_identity import canonical_providers, provider_identity
from scripts._study_shared import cv_skill, cv_unit, cv_unit_fileset, read_cv_unit


def _arm():
    rng = np.random.default_rng(3)
    providers = np.array(["sportec"] * 40 + ["skillcorner"] * 60 + ["gradientsports"] * 50)
    groups = np.array([str(i // 10) for i in range(150)])  # bare game ids
    y = (rng.random(150) < np.where(providers == "sportec", 0.4, 0.1)).astype(int)
    return y, groups, providers


def _plan(groups):
    games = sorted(set(groups), key=int)
    out = []
    for k in range(3):
        held = {g for i, g in enumerate(games) if i % 3 == k}
        te = np.flatnonzero(np.isin(groups, list(held)))
        tr = np.flatnonzero(~np.isin(groups, list(held)))
        out.append((k, tr, te))
    return out


def _unit(k, tr, te, y, groups, providers):
    p = np.full(len(te), 0.2)
    return cv_unit(((0.1, 0.2), p, float(y[tr].mean())), float(y.mean()), tr, te, y, groups, providers)


def test_provider_identity_normalises_and_refuses_empty():
    assert provider_identity("sportec") == "idsse"
    assert provider_identity("skillcorner") == "skillcorner"
    with pytest.raises(ValueError, match="empty"):
        provider_identity("")
    assert canonical_providers(["sportec", "skillcorner"]) == ["idsse", "skillcorner"]


def test_pooled_rate_equals_train_rate():
    y, groups, providers = _arm()
    for k, tr, te in _plan(groups):
        u = _unit(k, tr, te, y, groups, providers)
        assert u["train_rate"] == float(y[tr].mean())
        for prov, rate in u["train_rate_by_provider"].items():
            raw = {"idsse": "sportec"}.get(prov, prov)
            assert rate == float(y[tr][providers[tr] == raw].mean())
        n = sum(
            (providers[tr] == {"idsse": "sportec"}.get(p, p)).sum() * r for p, r in u["train_rate_by_provider"].items()
        )
        assert n / len(tr) == pytest.approx(u["train_rate"])  # the provider rates recompose the pooled rate


def test_by_provider_keys_are_identities():
    y, groups, providers = _arm()
    units = [_unit(k, tr, te, y, groups, providers) for k, tr, te in _plan(groups)]
    assert set(units[0]["providers"]) <= {"idsse", "skillcorner", "gradientsports"}
    skill = cv_skill(units, nested=True)
    assert skill is not None
    assert set(skill["by_provider"]) == {"idsse", "skillcorner", "gradientsports"}  # never sportec, never a game id


def test_empty_provider_refuses_the_unit():
    y, groups, providers = _arm()
    bad = providers.copy()
    bad[5] = ""
    k, tr, te = _plan(groups)[0]
    with pytest.raises(ValueError, match="empty"):
        _unit(k, tr, te, y, groups, bad)


def test_unit_round_trips_providers_and_rates():
    y, groups, providers = _arm()
    k, tr, te = _plan(groups)[1]
    u = _unit(k, tr, te, y, groups, providers)
    back = read_cv_unit(cv_unit_fileset(u))
    assert back["train_rate_by_provider"] == u["train_rate_by_provider"]
    assert back["providers"].tolist() == u["providers"].tolist()
    assert "providers.npy" in cv_unit_fileset(u).files


def test_skipped_fold_round_trips():
    y, _groups, _providers = _arm()
    u = cv_unit(None, float(y.mean()), None, None, None, None, None)
    back = read_cv_unit(cv_unit_fileset(u))
    assert back["fold"] is None and back["train_rate_by_provider"] == {} and len(back["providers"]) == 0
