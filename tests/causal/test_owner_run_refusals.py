"""Task 6b Step 4b (R11): the owner-run refusal branches must demonstrably FIRE.

Layer discipline -- each fixture constructs its condition at the layer the guard defends:
(a) ``_gated_probe_matches`` (trainer, spec M6) at the meta.json-dict layer; (b)/(c)
``_entanglement_gate`` / ``analyze`` (shot-arm runner, R10) at the opportunity-frame layer.
Both script modules are loaded in-process via the importlib pattern the causal e2e uses."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from silly_kicks.causal import SHOT_ARM_CONFOUNDERS, shot_arm_config

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(stem):
    spec = importlib.util.spec_from_file_location(f"_{stem}", _SCRIPTS / f"{stem}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load("train_xcross_attempt")


@pytest.fixture(scope="module")
def shot_runner():
    return _load("validate_xshot_causal")


# --- (a) trainer M6: held-out gated statistic (meta.json-dict layer) ----------------------


def _meta(membership: dict) -> dict:
    """Planted probe meta.json dict: match_id -> in_training_folds flag."""
    return {
        "home_team_id": "5",
        "probe_matches": [["gradientsports", mid] for mid in membership],
        "probe_providers": ["gradientsports"],
        "match_groups": {mid: [f"g_{mid}"] for mid in membership},
        "in_training_folds": dict(membership),
    }


def test_refusal_admitted_and_all_probe_matches_in_training(trainer):
    """(a) provider admitted to training + every probe match in the training folds -> refuse."""
    with pytest.raises(SystemExit, match=r"[Hh]eld-out"):
        trainer._gated_probe_matches(_meta({"m1": True, "m2": True}), True)


def test_refusal_admitted_and_missing_provenance(trainer):
    """(a) pre-plan probe sample (no probe_matches recorded) + admitted -> refuse, never guess."""
    with pytest.raises(SystemExit, match="provenance"):
        trainer._gated_probe_matches({"home_team_id": "5"}, True)


def test_refusal_admitted_unknown_membership_counts_as_in_training(trainer):
    """(a) membership unrecorded for a match -> conservative in-training -> refusal fires."""
    meta = _meta({"m1": True})
    meta["in_training_folds"] = {}
    with pytest.raises(SystemExit, match=r"[Hh]eld-out"):
        trainer._gated_probe_matches(meta, True)


def test_healthy_admitted_returns_only_held_out_matches(trainer):
    """(c) healthy: admitted with one held-out match -> the gated set is exactly that match."""
    meta = _meta({"m1": True, "m2": False})
    assert trainer._gated_probe_matches(meta, True) == [["gradientsports", "m2"]]


def test_healthy_not_admitted_returns_all_matches(trainer):
    """(c) healthy: provider NOT admitted (public candidate shipped) -> all matches held-out."""
    meta = _meta({"m1": False, "m2": False})
    assert trainer._gated_probe_matches(meta, False) == meta["probe_matches"]


# --- (b)/(c) shot-arm runner R10: control-conversion floor (opportunity-frame layer) -------

_GK_BLOCK = list(shot_arm_config({}).gk_block)


def _opp(n=200, control_conversions=40, seed=0):
    """Planted shot-arm opportunity frame: the registered confounder + GK-block columns,
    10 game clusters, alternating Z, and an EXACT control-conversion count."""
    rng = np.random.default_rng(seed)
    cols = list(SHOT_ARM_CONFOUNDERS) + _GK_BLOCK
    opp = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    opp["game_id"] = np.repeat([f"g{i}" for i in range(10)], n // 10)
    z = (np.arange(n) % 2 == 0).astype(int)
    y = np.zeros(n, dtype=int)
    control_idx = np.where(z == 0)[0]
    treated_idx = np.where(z == 1)[0]
    assert control_conversions <= len(control_idx)  # fixture validity
    y[control_idx[:control_conversions]] = 1
    y[treated_idx[: len(treated_idx) // 3]] = 1  # treated conversions: healthy Y variation
    opp["Z"], opp["Y"] = z, y
    return opp


def test_shot_arm_floor_is_registered_at_30(shot_runner):
    assert shot_runner.SHOT_ARM_MIN_CONTROL_CONVERSIONS == 30


def test_refusal_control_conversions_below_floor(shot_runner):
    """(b) one below the floor -> not measurable; rate + count reported (R10)."""
    floor = shot_runner.SHOT_ARM_MIN_CONTROL_CONVERSIONS
    gate = shot_runner._entanglement_gate(_opp(control_conversions=floor - 1))
    assert gate["measurable"] is False
    assert gate["control_conversions"] == floor - 1
    assert 0.0 <= gate["control_y_rate"] <= 1.0


def test_floor_boundary_exactly_at_floor_is_measurable(shot_runner):
    floor = shot_runner.SHOT_ARM_MIN_CONTROL_CONVERSIONS
    assert shot_runner._entanglement_gate(_opp(control_conversions=floor))["measurable"] is True


def test_refusal_analyze_refuses_the_entanglement_verdict(shot_runner):
    """(b) end-to-end through analyze(): the verdict is REFUSED (degenerate), the numbers
    are still reported (reported-not-gated house style)."""
    m = shot_runner.analyze(_opp(control_conversions=5), seed=0, n_seeds=5)
    assert m["status"] == "ok"
    assert m["entanglement"] == "degenerate"
    assert m["entanglement_refused"] is True
    assert "SHOT_ARM_MIN_CONTROL_CONVERSIONS" in m["entanglement_refusal_reason"]
    assert m["control_conversions"] == 5  # R10: count reported alongside the refusal
    assert 0.0 <= m["control_y_rate"] <= 1.0


def test_healthy_shot_arm_emits_verdict_and_both_bands(shot_runner):
    """(c) healthy: verdict emitted, cluster band gates, row band reported."""
    m = shot_runner.analyze(_opp(control_conversions=40), seed=0, n_seeds=5)
    assert m["status"] == "ok"
    assert m["entanglement_refused"] is False
    assert m["entanglement"] in ("clears", "inside_band")
    assert np.isfinite(m["placebo_band_p95_row"])
    assert np.isfinite(m["placebo_band_p95_cluster"])
    assert m["placebo_band_p95"] == m["placebo_band_p95_cluster"]  # the GATE reads the cluster band
    assert m["placebo_n_clusters"] == 10
    # the emitted verdict is a legal regate_verdict entanglement input (spec SS3.5)
    from silly_kicks.tracking._model_eval import regate_verdict

    assert regate_verdict(arm="shot", probe_verdict="fail", entanglement=m["entanglement"]) == "gated_clean_fail"


def test_healthy_refusal_report_renders(shot_runner, tmp_path):
    """The refusal state must survive the artifact write (report + metrics)."""
    m = shot_runner.analyze(_opp(control_conversions=5), seed=0, n_seeds=5)
    shot_runner._write(tmp_path, m)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "REFUSED" in report
    assert "SHOT_ARM_MIN_CONTROL_CONVERSIONS" in report
