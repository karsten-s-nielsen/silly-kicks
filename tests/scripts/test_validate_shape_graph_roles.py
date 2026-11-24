"""W5.7: the real-data role gate driver (spec 2026-11-08 tracking-batch section 4.5 W5.7, owner ruling R15; plan
Task 11A). The real run is Task 15 step 7b; these tests pin the scoring, the null, the gate and the arms."""

from __future__ import annotations

import ast
import json
import math

import numpy as np
import pandas as pd
import pytest

from scripts import validate_shape_graph_roles as R
from scripts._frame_reflection import reflect_frames
from tests._prod_fixtures import make_production_frames

#: The player_role.acronym census of the 20 public SkillCorner matches (plan Task 11A step 1, 2026-11-11, pining
#: match.json only): acronym -> players over the 20 rosters.
ROLE_CENSUS = {
    "AM": 27,
    "CB": 11,
    "CF": 104,
    "DM": 7,
    "GK": 40,
    "LB": 36,
    "LCB": 44,
    "LDM": 13,
    "LF": 5,
    "LM": 40,
    "LW": 62,
    "LWB": 9,
    "RB": 43,
    "RCB": 45,
    "RDM": 17,
    "RF": 5,
    "RM": 40,
    "RW": 65,
    "RWB": 10,
    "SUB": 96,
}


# ---- the truth table and the population -------------------------------------------------------------------------


def test_role_table_covers_census():
    outfield = set(ROLE_CENSUS) - {"GK", "SUB"}
    assert outfield <= set(R.ROLE_VERTICAL_GROUP)
    assert set(R.ROLE_VERTICAL_GROUP.values()) == {"back", "mid", "fwd"}
    assert R.role_side("LCB") == "L" and R.role_side("RW") == "R" and R.role_side("CF") is None


def test_unknown_acronym_raises():
    with pytest.raises(ValueError, match="XYZ"):
        R.role_group("XYZ")


def _player(pid, acr, start: str | None = "00:00", team=1):
    return {"id": pid, "team_id": team, "player_role": {"acronym": acr}, "start_time": start}


def test_population_predicate():
    roster = [
        _player(1, "LB"),
        _player(2, "GK"),
        _player(3, "SUB"),
        _player(4, "CF", start=None),
        _player(5, "RW"),  # no labelled row
        _player(6, "RM", start="61:30"),  # a substitute who came on with a real role
    ]
    assert R.scored_population(roster, labelled={"1", "2", "3", "4", "6"}) == frozenset({"1", "6"})


# ---- the modal labels ---------------------------------------------------------------------------------------------


def test_modal_vertical_mode_then_group():
    # B 40 %, DM 30 %, M 30 %: the five-level mode is B (back), although the groups would give mid 60 %
    assert R.modal_vertical(["B"] * 4 + ["DM"] * 3 + ["M"] * 3) == "back"


def test_modal_side_side_then_mode():
    # sides first: L-side {L, LC} = 2, R-side {R, R} = 2, a tie the first-seen L wins; a mode-first rule gives R
    assert R.modal_side(["L", "LC", "R", "R", "C"]) == "L"
    assert R.modal_side(["C", "RC", "R"]) == "R"


def test_tie_first_seen_wins():
    assert R.modal_vertical(["F", "B"]) == "fwd"
    assert R.modal_side(["R", "L"]) == "R"
    assert R.is_tied(["F", "B"]) and not R.is_tied(["F", "F", "B"])


# ---- the block permutation null -----------------------------------------------------------------------------------


def _scored(truth_a, pred_a, truth_b, pred_b) -> pd.DataFrame:
    rows = [("m1", "t1", t, p) for t, p in zip(truth_a, pred_a, strict=True)]
    rows += [("m2", "t2", t, p) for t, p in zip(truth_b, pred_b, strict=True)]
    return pd.DataFrame(rows, columns=["match_id", "team_id", "truth", "pred"])


def test_block_null_never_crosses_blocks():
    # each block holds one truth value, so a within-block shuffle can never change the agreement (a cross-block
    # shuffle would mix "back" into the "fwd" block)
    scored = _scored(["back"] * 5, ["back"] * 5, ["fwd"] * 5, ["fwd"] * 5)
    out = R.block_permutation_null(scored, n_draws=200, seed=1)
    assert out["agreement"] == 1.0 and out["null_p95"] == 1.0 and out["p"] == 1.0 and out["n"] == 10


def test_block_null_seeded_reproducible():
    rng = np.random.default_rng(0)
    truth = rng.choice(["back", "mid", "fwd"], 40)
    pred = np.where(rng.random(40) < 0.7, truth, rng.choice(["back", "mid", "fwd"], 40))
    scored = _scored(truth[:20], pred[:20], truth[20:], pred[20:])
    assert R.block_permutation_null(scored, n_draws=500, seed=7) == R.block_permutation_null(
        scored, n_draws=500, seed=7
    )


def test_p_value_formula():
    rng = np.random.default_rng(3)
    truth = rng.choice(["back", "mid", "fwd"], 30)
    pred = np.where(rng.random(30) < 0.6, truth, "mid")
    scored = _scored(truth[:15], pred[:15], truth[15:], pred[15:])
    null = R.null_agreements(scored, n_draws=300, seed=11)
    obs = float(np.mean(scored["truth"] == scored["pred"]))
    out = R.block_permutation_null(scored, n_draws=300, seed=11)
    assert out["p"] == (1 + int(np.sum(null >= obs))) / (1 + 300)
    assert out["null_p95"] == float(np.percentile(null, 95))


def test_wilson_interval_known_values():
    low, high = R.wilson(8, 10)
    assert math.isclose(low, 0.4902, abs_tol=1e-4) and math.isclose(high, 0.9433, abs_tol=1e-4)
    low0, high0 = R.wilson(0, 10)
    assert low0 == 0.0 and math.isclose(high0, 0.2775, abs_tol=1e-4)


# ---- the gate -------------------------------------------------------------------------------------------------------


def _metrics(**override) -> dict:
    cells = {cell: {"agreement": 0.8, "null_p95": 0.5, "baseline_agreement": 0.7} for cell in R.CELLS}
    metrics = {"cells": cells, "mirror": {"n_differing_team_frames": 0}}
    for path, value in override.items():
        cell, field = path.split("__")
        if cell == "mirror":
            metrics["mirror"][field] = value
        else:
            metrics["cells"][cell][field] = value
    return metrics


def test_gate_each_rule_fails_alone():
    assert R.gate(_metrics()) == []
    assert R.gate(_metrics(vertical_home__null_p95=0.85)) == ["rule1_beats_null:vertical_home"]
    assert R.gate(_metrics(lateral_away__baseline_agreement=0.9)) == ["rule2_no_regression:lateral_away"]
    assert R.gate(_metrics(mirror__n_differing_team_frames=2)) == ["rule3_mirror_invariance"]


# ---- the arms -------------------------------------------------------------------------------------------------------


def test_reflect_frames_exact():
    frames = make_production_frames(n_frames=3, n_periods=1)
    once = reflect_frames(frames)
    assert once["x"].dtype == np.float64 and once["y"].dtype == np.float64
    twice = reflect_frames(once)
    np.testing.assert_array_equal(twice["x"].to_numpy(), frames["x"].to_numpy(dtype=np.float64))
    np.testing.assert_array_equal(twice["y"].to_numpy(), frames["y"].to_numpy(dtype=np.float64))
    swapped = {"ltr": "rtl", "rtl": "ltr"}
    for a, b in zip(frames["team_attacking_direction"], once["team_attacking_direction"], strict=True):
        if pd.isna(a):
            assert pd.isna(b)  # a missing label stays missing (None or NaN by pandas major)
        else:
            assert b == swapped.get(a, a)


def test_mirror_report_names_team_frame():
    keys = ["game_id", "period_id", "frame_id", "team_id"]
    plain = pd.DataFrame(
        {"game_id": [1, 1], "period_id": [1, 1], "frame_id": [10, 10], "team_id": [5, 5], "player_id": [1, 2]}
    ).assign(vertical=["B", "M"], horizontal=["L", "C"], position_label=["LB", "CM"])
    mirrored = plain.copy()
    mirrored.loc[1, "horizontal"] = "R"
    report = R.mirror_report(plain, mirrored, faces_differ=lambda key: key == (1, 1, 10, 5))
    assert report == [{"key": [1, 1, 10, 5], "faces_differ": True}]
    assert R.mirror_report(plain, plain.copy(), faces_differ=lambda key: False) == []
    assert keys == R.TEAM_FRAME_KEYS


def test_baseline_arm_uses_v4139_copy_with_frame_direction(monkeypatch):
    from scripts import _shape_graph_v4139_baseline as baseline

    frames = make_production_frames(n_frames=2, n_periods=1)
    seen: list[float] = []
    real = baseline.infer_positions

    def spy(sg, positions, direction):
        seen.append(direction)
        return real(sg, positions, direction)

    monkeypatch.setattr(baseline, "infer_positions", spy)
    base = R.baseline_labels(frames)
    assert sorted(set(seen)) == [-1.0, 1.0]  # ltr -> +1, rtl -> -1
    assert list(base.columns) == [*R.TEAM_FRAME_KEYS, "player_id", "vertical", "horizontal"]
    assert base.groupby(R.TEAM_FRAME_KEYS, sort=False, observed=True).ngroups == len(seen) > 0


def test_baseline_copy_is_standalone():
    """The 4.139 baseline arm (ruling C5) imports no live helper, so a later edit of the live shape graph cannot move
    it; only the role gate imports it."""
    from tests.scripts._script_population import iter_package_modules, iter_script_modules

    tree = iter_script_modules()["scripts._shape_graph_v4139_baseline"]
    live = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and (n.module or "").startswith("silly_kicks")]
    assert [(n.module, sorted(a.name for a in n.names)) for n in live] == [
        ("silly_kicks.tracking._shape_graph", ["PositionLabel", "ShapeGraph"])
    ]
    importers = [
        name
        for name, t in {**iter_package_modules(), **iter_script_modules()}.items()
        if "_shape_graph_v4139_baseline" in ast.unparse(t) and name != "scripts._shape_graph_v4139_baseline"
    ]
    assert importers == ["scripts.validate_shape_graph_roles"]


def test_frame_id_is_raw_frame():
    from tests._skillcorner_sample import SAMPLE_DIR, available_matches, find_artifact

    mids = available_matches(("_match.json", "_tracking_extrapolated.jsonl"))
    if not mids:
        pytest.skip(f"no SkillCorner sample under {SAMPLE_DIR}")
    from scripts._loader_pining import build_skillcorner_frames

    d = SAMPLE_DIR / mids[0]
    meta, tracking = find_artifact(d, "_match.json"), find_artifact(d, "_tracking_extrapolated.jsonl")
    assert meta is not None and tracking is not None  # available_matches checked both
    paths = {"metadata": meta, "tracking": tracking}
    frames, _report = build_skillcorner_frames(paths, mids[0], 300, with_velocity=False)
    raw = [json.loads(line) for line in tracking.read_text(encoding="utf-8").splitlines() if line.strip()]
    raw_frames = {int(r["frame"]) for r in raw if r.get("player_data")}
    assert set(frames["frame_id"].astype(int)) <= raw_frames and len(set(frames["frame_id"])) > 100


def test_driver_registered():
    from tests.scripts.test_compute_contract import _LOCAL_ONLY
    from tests.scripts.test_provenance_wiring import ARTIFACT_DRIVERS

    assert "validate_shape_graph_roles" in ARTIFACT_DRIVERS
    assert "validate_shape_graph_roles" in _LOCAL_ONLY
    assert R.PIPELINE.module == "scripts.validate_shape_graph_roles"
