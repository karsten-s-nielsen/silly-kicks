"""CI-safe unit slice for the trained-artifact integrity gate (ADR-126).

The real-corpus audit (re-extract + re-fit per family) is the opt-in/e2e Task-6 run; these units
give the gate its teeth in the fast suite: the reproduction tripwire and the verdict composition.
Numbers follow the skillcorner-1925299 scale (8394 rows, 21 shots, 10fps) so the physically-derived
band inputs are consistent with the n_rows used by the reproduction check.
"""

from __future__ import annotations

from scripts import validate_trained_artifact_integrity as V


def _shipped():
    return {
        "n_rows": 8394,
        "n_positive": 129,  # 0.0154 * 8394
        "candidates": {
            "public": {
                "metrics": {
                    "pr_auc": 0.0820,
                    "positive_rate": 0.0154,
                    "brier": 0.0140,
                    "base_rate_brier": 0.0152,
                    "n_usable_folds": 5,
                }
            }
        },
    }


def test_reproduces_passes_on_match_fails_on_perturbation():
    shipped = _shipped()
    fresh_ok = {
        "n_rows": 8394,
        "n_positive": 129,
        "candidates": {"public": {"metrics": {"pr_auc": 0.0831, "positive_rate": 0.0154}}},
    }
    fresh_bad = {
        "n_rows": 8394,
        "n_positive": 1864,  # 0.2220 * 8394 -> the committed defect
        "candidates": {"public": {"metrics": {"pr_auc": 0.3418, "positive_rate": 0.2220}}},
    }
    assert V.reproduces(shipped, fresh_ok, rel_tol=0.05) is True
    assert V.reproduces(shipped, fresh_bad, rel_tol=0.05) is False  # committed defect vs correct re-fit


def test_verdict_passes_corrected_fails_defect():
    shipped = _shipped()
    # a faithful corrected re-fit: reproduces + plausible base rate + lift + calibration + folds
    fresh_ok = dict(_shipped(), n_events=21, fps=10.0, horizon_s=1.0)
    v_ok = V.verdict("xshot_occurrence", shipped, fresh_ok, rel_tol=0.05)
    assert v_ok["reproduced"] and v_ok["base_rate_ok"] and v_ok["lift_ok"]
    assert v_ok["calibration_ok"] and v_ok["enough_folds"]
    assert V.is_go(v_ok) is True

    # the committed defect: 22% base rate -> base_rate_ok False, reproduced False -> NO-GO
    defect = {
        "n_rows": 8394,
        "n_positive": 1864,
        "n_events": 21,
        "fps": 10.0,
        "horizon_s": 1.0,
        "candidates": {
            "public": {
                "metrics": {
                    "pr_auc": 0.3418,
                    "positive_rate": 0.2220,
                    "brier": 0.17,
                    "base_rate_brier": 0.1727,
                    "n_usable_folds": 5,
                }
            }
        },
    }
    v_bad = V.verdict("xshot_occurrence", shipped, defect, rel_tol=0.05)
    assert v_bad["base_rate_ok"] is False
    assert v_bad["reproduced"] is False
    assert V.is_go(v_bad) is False
