"""PR-S112: construct_validity_scores real-rho path -- GK-domain restriction + injected retention + v1_stored."""

import numpy as np
import pytest

from scripts.validate_xtgk_v2 import construct_validity_scores


class _FakeRho:
    def predict_proba(self, features):
        return np.full(len(features), 0.6)


def _cohort(*, with_domain: bool, with_v1: bool):
    # possession-parity split needs both even+odd possession_ids; a mix of GK-dist True/False rows.
    from tests.xtgk.conftest import mixed_shot_and_shotless_cohort

    a = mixed_shot_and_shotless_cohort().reset_index(drop=True)
    test_rows = a.index[(a["possession_id"] % 2 == 1)]  # the eval (odd-possession) split
    assert len(test_rows) >= 2, "fixture must have >=2 test-split rows"
    if with_domain:
        a["is_gk_distribution"] = np.arange(len(a)) % 3 != 0
        # GUARANTEE both a True and a False row IN THE TEST split -> non-vacuous 0 < n_test_gk < n_test
        a.loc[test_rows[0], "is_gk_distribution"] = True
        a.loc[test_rows[1], "is_gk_distribution"] = False
    if with_v1:
        a["xt_gk"] = np.linspace(-0.02, 0.05, len(a))
        if with_domain:
            # NULL xt_gk on an odd-possession + is_gk_distribution=True row (test_rows[0]) so the null lands
            # IN test_gk and actually drops from the v1 denominator (the coverage path). NOT index[:2] (train).
            a.loc[test_rows[0], "xt_gk"] = np.nan
    return a


def test_real_rho_gk_domain_restriction_and_v1_stored():
    a = _cohort(with_domain=True, with_v1=True)
    # ADR-036 amendment: this fixture is a RAW cohort (no xt_gk_origin_* columns), so its
    # GK-distribution rows are legitimately unattested. Assert the metric says so out loud rather
    # than silencing it -- an unflagged raw cohort is the exact defect 4.46.0 fixes.
    with pytest.warns(UserWarning, match="apply_resolved_gk_geometry"):
        s = construct_validity_scores(a, xg_column="xg", pressure_column="pressure", retention=_FakeRho())
    n_test = int((a["possession_id"] % 2 == 1).sum())
    assert 0 < s["n_test_gk"] < n_test  # restriction applied (non-vacuous)
    assert np.isfinite(s["xt_gk_v2"]["auc"])  # real features built + scored, no crash
    assert s["v1_stored"]["n"] < s["n_test_gk"]  # STRICT: the null xt_gk row is dropped from v1's denominator
    assert "v2_on_v1_rows" in s  # apples-to-apples v2-vs-v1 number reported
    assert "v1_composite" not in s  # old key gone
    assert "lift" in s  # lift over max(raw, dest, v1_stored)


def test_domain_fallback_when_column_absent():
    a = _cohort(with_domain=False, with_v1=False)
    s = construct_validity_scores(a, xg_column="xg", pressure_column="pressure")  # stub path, no domain col
    assert s["n_test_gk"] == int((a["possession_id"] % 2 == 1).sum())  # falls back to all-test
    for k in ("xt_gk_v2", "raw_completion", "destination_xt", "v1_stored", "lift"):
        assert k in s
