"""The per-provider forecast-skill gate (corpus widening spec 5.7, ADR-132; spec test 15 core).

Fixtures use the REAL shapes: bare game-id ``groups`` (never ``provider__game``) and a separate, row-aligned provider
column whose values are licensing identities (``idsse``, never the converter name ``sportec``).
"""

from __future__ import annotations

import numpy as np
import pytest

from silly_kicks.tracking._artifact_integrity import forecast_skill, provider_skill, skill_gates

_FIELDS = {"n_matches", "n_oof_rows", "n_positive", "row_share", "brier_skill", "log_loss_skill", "pr_auc_lift"}


def _corpus(seed: int = 0):
    """Two providers with different base rates, 5 gate folds, bare game ids; a per-row true probability."""
    rng = np.random.default_rng(seed)
    rows = []
    for prov, games, lo, hi in (("skillcorner", range(1000, 1020), 0.02, 0.4), ("idsse", range(1, 11), 0.10, 0.7)):
        for g in games:
            for _ in range(120):
                rows.append((prov, str(g), hi if rng.random() < 0.2 else lo))
    providers = np.array([r[0] for r in rows])
    groups = np.array([r[1] for r in rows])
    q = np.array([r[2] for r in rows])
    y = (rng.random(len(q)) < q).astype(int)
    uniq = sorted(set(groups))
    fold_of = {g: i % 5 for i, g in enumerate(uniq)}
    fold = np.array([fold_of[g] for g in groups])
    # each row's provider-own training-fold rate, and the pooled one
    r_prov = np.empty(len(y))
    r_pool = np.empty(len(y))
    for k in range(5):
        tr = fold != k
        r_pool[fold == k] = y[tr].mean()
        for prov in ("skillcorner", "idsse"):
            sel = (fold == k) & (providers == prov)
            r_prov[sel] = y[tr & (providers == prov)].mean()
    return y, q, r_prov, r_pool, providers, groups, fold


def _metrics(by_provider) -> dict:
    skill = {"brier_skill": 0.1, "log_loss_skill": 0.1, "pr_auc_lift": 0.1, "bootstrap": {}}
    if by_provider is not None:
        skill["by_provider"] = by_provider
    return {"n_usable_folds": 5, "skill": skill}


def test_every_provider_skilled_passes():
    y, q, r_prov, _r_pool, providers, groups, fold = _corpus()
    by = provider_skill(y, q, r_prov, providers, groups, fold)
    assert set(by) == {"skillcorner", "idsse"}
    for v in by.values():
        assert set(v) == _FIELDS
        assert v["brier_skill"] > 0 and v["log_loss_skill"] > 0
    g = skill_gates(_metrics(by))
    assert g["provider_brier_skill_gt_0"] and g["provider_log_loss_skill_gt_0"]


def test_reported_fields_count_the_provider_rows():
    y, q, r_prov, _r_pool, providers, groups, fold = _corpus()
    by = provider_skill(y, q, r_prov, providers, groups, fold)
    assert by["idsse"]["n_matches"] == 10 and by["skillcorner"]["n_matches"] == 20
    assert by["idsse"]["n_oof_rows"] == 1200 and by["skillcorner"]["n_oof_rows"] == 2400
    assert by["idsse"]["n_positive"] == int(y[providers == "idsse"].sum())
    assert by["idsse"]["row_share"] == pytest.approx(1200 / 3600)


def test_one_provider_unskilled_fails():
    y, q, r_prov, r_pool, providers, groups, fold = _corpus()
    p = q.copy()
    idsse = providers == "idsse"
    p[idsse] = r_pool[idsse]  # the IDSSE rows get only the pooled rate: no skill against IDSSE's own rate
    assert forecast_skill(y, p, r_pool, fold, groups, nested=True)["brier_skill"] > 0  # pooled still positive
    by = provider_skill(y, p, r_prov, providers, groups, fold)
    assert by["skillcorner"]["brier_skill"] > 0 and by["idsse"]["brier_skill"] <= 0
    g = skill_gates(_metrics(by))
    assert not g["provider_brier_skill_gt_0"]


def test_provider_rate_forecast_has_zero_skill():
    # A model that only recognises the provider: it beats the pooled reference, but earns no provider skill.
    y, _q, r_prov, r_pool, providers, groups, fold = _corpus()
    assert forecast_skill(y, r_prov, r_pool, fold, groups, nested=True)["brier_skill"] > 0
    by = provider_skill(y, r_prov, r_prov, providers, groups, fold)
    for v in by.values():
        assert v["brier_skill"] == 0.0 and v["log_loss_skill"] == 0.0
    g = skill_gates(_metrics(by))
    assert not g["provider_brier_skill_gt_0"] and not g["provider_log_loss_skill_gt_0"]


def test_nan_provider_skill_fails():
    y, q, r_prov, _r_pool, providers, groups, fold = _corpus()
    r = r_prov.copy()
    r[(providers == "idsse") & (fold == 0)] = np.nan  # no IDSSE training rows in gate fold 0
    by = provider_skill(y, q, r, providers, groups, fold)
    assert np.isnan(by["idsse"]["brier_skill"]) and np.isnan(by["idsse"]["log_loss_skill"])
    g = skill_gates(_metrics(by))
    assert not g["provider_brier_skill_gt_0"] and not g["provider_log_loss_skill_gt_0"]


def test_zero_reference_loss_is_nan():
    y = np.zeros(6, dtype=int)
    by = provider_skill(
        y, np.full(6, 0.1), np.zeros(6), np.array(["idsse"] * 6), np.array(list("aabbcc")), np.arange(6) % 2
    )
    assert np.isnan(by["idsse"]["brier_skill"])


def test_missing_by_provider_fails():
    for by in (None, {}):
        g = skill_gates(_metrics(by))
        assert not g["provider_brier_skill_gt_0"] and not g["provider_log_loss_skill_gt_0"]
    g = skill_gates({"n_usable_folds": 5})
    assert not g["provider_brier_skill_gt_0"] and not g["provider_log_loss_skill_gt_0"]


def test_by_provider_keys_are_the_given_identities():
    y, q, r_prov, _r_pool, providers, groups, fold = _corpus()
    by = provider_skill(y, q, r_prov, providers, groups, fold)
    assert set(by) == set(providers)
    assert not set(by) & set(groups)
