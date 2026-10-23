"""TF-46 Arm 1 Part 2 (spec section 18.2.1): the training provenance contract.

Covers: the five per-dimension rules; ``training_exclusion_mask == provenance_masks(...)["result"]``; the
action-level exclusion composition (xsuccess / expected_passing exclude result-OR-location; xtgk excludes
result); the frame-level occurrence censoring (xshot / xcross censor instead of dropping, with a count); the
ghost restart censor (CL-5); and the ``TRAINING_READS`` declaration gate (ADR-056 content predicate).
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

import silly_kicks.spadl.config as cfg
from silly_kicks.spadl._training_exclusion import (
    DIMENSIONS,
    provenance_masks,
    training_exclusion_mask,
    uncertainty_interval,
)
from silly_kicks.tracking._ghost_gk import (
    _SET_PIECE_DECAY_SECONDS,
    build_restart_phase_censor_intervals,
    ghost_restart_censor_mask,
)
from silly_kicks.tracking._occurrence_labels import occurrence_labels_and_censor

_SHOT_PENALTY = cfg.actiontype_id["shot_penalty"]
_PASS = cfg.actiontype_id["pass"]
_CORNER_CROSSED = cfg.actiontype_id["corner_crossed"]


def _provenance_fixture() -> pd.DataFrame:
    """One row per unvalidated kind plus a clean row (index 0)."""
    n = 10
    base = {
        "game_id": ["g"] * n,
        "period_id": [1] * n,
        "team_id": ["tA"] * n,
        "type_id": [_PASS] * n,
        "result_source": ["native"] * n,
        "restart_evidence": [pd.NA] * n,
        "repair_kind": [pd.NA] * n,
        "action_provenance": ["original"] * n,
        "kick_time_source": [pd.NA] * n,
        "t_kick": [np.nan] * n,
        "type_source": ["native"] * n,
        "start_x": [30.0] * n,
        "origin_evidence": [pd.NA] * n,
        "penalty_evidence": [pd.NA] * n,
        "time_seconds": [float(10 * i) for i in range(n)],
        "kick_bracket_lo": [np.nan] * n,
        "kick_bracket_hi": [np.nan] * n,
    }
    a = pd.DataFrame(base)
    a.loc[1, "result_source"] = "inferred"  # result
    a.loc[2, "restart_evidence"] = "chain_ambiguous"  # result
    a.loc[3, "repair_kind"] = "missing_goal"  # result AND time
    # derived restart, no gated t_kick (time); kick bracket present
    a.loc[4, ["action_provenance", "kick_time_source", "kick_bracket_lo", "kick_bracket_hi"]] = [
        "derived",
        "withheld",
        100.0,
        110.0,
    ]
    # derived restart WITH gated t_kick (time-valid)
    a.loc[5, ["action_provenance", "kick_time_source", "t_kick"]] = ["derived", "measured", 55.0]
    a.loc[6, "type_source"] = "default"  # type
    a.loc[7, "start_x"] = np.nan  # location
    a.loc[8, ["start_x", "origin_evidence"]] = [np.nan, "stationary_window"]  # location-valid (gated origin)
    # a derived penalty with withheld kick time but a native shot_penalty type -> time-valid (gated penalty)
    a.loc[9, ["type_id", "type_source", "action_provenance", "kick_time_source"]] = [
        _SHOT_PENALTY,
        "native",
        "derived",
        "withheld",
    ]
    return a


def test_provenance_masks_per_dimension():
    a = _provenance_fixture()
    pm = provenance_masks(a)
    assert set(pm) == set(DIMENSIONS)
    assert pm["result"].to_numpy().tolist() == [False, True, True, True, False, False, False, False, False, False]
    # time: repair row (3) and ungated derived restart (4); NOT the gated derived (5) nor the gated penalty (9)
    assert pm["time"].to_numpy().tolist() == [False, False, False, True, True, False, False, False, False, False]
    assert pm["type"].to_numpy()[6] and int(pm["type"].sum()) == 1
    assert pm["location"].to_numpy()[7] and not pm["location"].to_numpy()[8]
    assert int(pm["location"].sum()) == 1
    assert not pm["existence"].any()


def test_uncertainty_interval():
    a = _provenance_fixture()
    ui = uncertainty_interval(a)
    # ungated derived restart (4): its kick bracket
    assert ui.loc[4, "t_lo"] == 100.0 and ui.loc[4, "t_hi"] == 110.0
    # repair row (3): no kick bracket -> row time on both ends
    assert ui.loc[3, "t_lo"] == 30.0 and ui.loc[3, "t_hi"] == 30.0
    # a time-validated row is NaN on both ends
    assert np.isnan(ui.loc[0, "t_lo"]) and np.isnan(ui.loc[5, "t_hi"])


def test_training_exclusion_mask_equals_result_dimension():
    a = _provenance_fixture()
    pd.testing.assert_series_equal(training_exclusion_mask(a), provenance_masks(a)["result"], check_names=False)


def test_action_level_exclusion_composition():
    a = _provenance_fixture()
    pm = provenance_masks(a)
    # xsuccess / expected_passing: result OR location
    excl = (pm["result"] | pm["location"]).to_numpy()
    assert excl[1] and excl[7] and not excl[0] and not excl[5]
    # a location-unvalidated row is dropped only by the result-OR-location rule, never by result alone
    assert pm["location"].to_numpy()[7] and not pm["result"].to_numpy()[7]
    # xtgk: result only -- the location-unvalidated row survives
    assert not pm["result"].to_numpy()[7]


def test_tolerant_of_missing_columns():
    a = pd.DataFrame({"type_id": [_PASS, _PASS], "start_x": [10.0, 20.0]})
    pm = provenance_masks(a)
    assert not any(m.any() for m in pm.values())
    assert not training_exclusion_mask(a).any()


def _frames_index(times, team="tA", period=1):
    return pd.DataFrame(
        {
            "game_id": ["g"] * len(times),
            "period_id": [period] * len(times),
            "time_seconds": [float(t) for t in times],
            "team_in_possession": [team] * len(times),
        }
    )


def test_occurrence_censor_validated_labels_and_uncertain_censors():
    frames = _frames_index([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    events = pd.DataFrame(
        {
            "game_id": ["g", "g"],
            "period_id": [1, 1],
            "team_id": ["tA", "tA"],
            "type_id": [cfg.actiontype_id["shot"], cfg.actiontype_id["shot"]],
            "time_seconds": [1.5, 100.0],  # second shot has an unvalidated time -> censor window
            "type_source": ["native", "native"],
            "result_source": ["native", "native"],
            "action_provenance": ["original", "derived"],
            "kick_time_source": [pd.NA, "withheld"],
            "t_kick": [np.nan, np.nan],
            "kick_bracket_lo": [np.nan, 2.0],
            "kick_bracket_hi": [np.nan, 4.0],
        }
    )
    y, censor = occurrence_labels_and_censor(
        frames, events, horizon=1.0, frame_team_col="team_in_possession", reads_type=False
    )
    # validated shot at 1.5 labels the frames at t=1.0 (1.0..2.0 covers 1.5) and nothing else positive
    assert y[1] == 1
    # the uncertain shot (bracket 2..4) censors team frames in [2 - 1, 4] = [1, 4]
    assert censor.tolist() == [False, True, True, True, True, False]


def test_build_xshot_and_xcross_return_censor():
    from silly_kicks.tracking._xcross_attempt import build_xcross_labels
    from silly_kicks.tracking._xshot_occurrence import build_xshot_labels

    frames = _frames_index([0.0, 1.0, 2.0, 3.0])
    shots = pd.DataFrame(
        {
            "game_id": ["g"],
            "period_id": [1],
            "team_id": ["tA"],
            "type_id": [cfg.actiontype_id["shot"]],
            "time_seconds": [100.0],
            "type_source": ["native"],
            "action_provenance": ["derived"],
            "kick_time_source": ["withheld"],
            "t_kick": [np.nan],
            "kick_bracket_lo": [0.0],
            "kick_bracket_hi": [3.0],
        }
    )
    ys, cs = build_xshot_labels(frames, shots, horizon_seconds=1.0, return_censor=True)
    assert isinstance(ys, pd.Series) and cs.any()  # the ungated shot censors its window
    crosses = shots.assign(type_id=[_CORNER_CROSSED])
    yc, cc = build_xcross_labels(
        frames, crosses, horizon_seconds=1.0, cross_types=("corner_crossed",), return_censor=True
    )
    assert isinstance(yc, pd.Series) and cc.any()


def test_ghost_restart_censor_intervals_and_mask():
    actions = pd.DataFrame(
        {
            "game_id": ["g", "g", "g"],
            "period_id": [1, 1, 1],
            "team_id": ["tA", "tA", "tA"],
            "type_id": [cfg.actiontype_id["corner_crossed"], cfg.actiontype_id["goalkick"], _PASS],
            "type_source": ["native", "native", "native"],
            "result_source": ["native", "native", "native"],
            "action_provenance": ["derived", "derived", "original"],
            "kick_time_source": ["withheld", "measured", pd.NA],
            "t_kick": [np.nan, 50.0, np.nan],
            "time_seconds": [20.0, 50.0, 70.0],
            "kick_bracket_lo": [18.0, 48.0, np.nan],
            "kick_bracket_hi": [22.0, 52.0, np.nan],
        }
    )
    iv = build_restart_phase_censor_intervals(actions)
    # only the ungated set-piece (row 0) censors; the gated goal kick (row 1) does not
    assert len(iv) == 1
    assert iv.iloc[0]["t_lo"] == 18.0
    assert iv.iloc[0]["t_hi"] == 22.0 + _SET_PIECE_DECAY_SECONDS  # set piece extends by the decay
    meta = pd.DataFrame({"period_id": [1, 1, 1], "frame_id": [1, 2, 3]})
    gframes = pd.DataFrame({"period_id": [1, 1, 1], "frame_id": [1, 2, 3], "time_seconds": [19.0, 40.0, 70.0]})
    mask = ghost_restart_censor_mask(meta, gframes, actions)
    assert mask.tolist() == [True, False, False]  # frame at 19.0 is in [18, 32]


def test_ghost_restart_censor_intervals_empty_without_actions():
    assert build_restart_phase_censor_intervals(None).empty


# --- declaration gate (ADR-056 content predicate over the module source) ---

_REPO = Path(__file__).resolve().parents[1]
#: The registered training entry points -> their source file (relative to the repo root).
_ENTRY_POINTS = {
    "silly_kicks.xsuccess._model": "silly_kicks/xsuccess/_model.py",
    "silly_kicks.expected_passing._model": "silly_kicks/expected_passing/_model.py",
    "silly_kicks.xtgk._markov": "silly_kicks/xtgk/_markov.py",
    "silly_kicks.xtgk._turnover": "silly_kicks/xtgk/_turnover.py",
    "silly_kicks.tracking._xshot_occurrence": "silly_kicks/tracking/_xshot_occurrence.py",
    "silly_kicks.tracking._xcross_attempt": "silly_kicks/tracking/_xcross_attempt.py",
    "scripts.train_ghost_gk": "scripts/train_ghost_gk.py",
    "scripts.train_ghost_outfield": "scripts/train_ghost_outfield.py",
}
#: Reviewed exceptions (a declared dim that the content predicate cannot see). Kept EMPTY unless justified.
_DECLARATION_EXCEPTIONS: dict[str, set[str]] = {}


def _derive_reads(src: str) -> set[str]:
    """Dimensions triggered by the provenance API the module's label / feature builders call."""
    dims: set[str] = set()
    if "training_exclusion_mask" in src and "provenance_masks" not in src:
        dims.add("result")
    if "provenance_masks" in src:
        for d in DIMENSIONS:
            if re.search(rf'\["{d}"\]', src):
                dims.add(d)
    if "occurrence_labels_and_censor" in src:
        dims.add("time")
        if "reads_type=True" in src:
            dims.add("type")
    if "build_restart_phase_censor_intervals" in src or "ghost_restart_censor_mask" in src:
        dims.add("time")
    return dims


def test_training_reads_declaration_gate():
    # Population is DERIVED (grep for the declaration) and asserted EXACTLY against the registry.
    found = {
        str(p.relative_to(_REPO)).replace("\\", "/")
        for base in ("silly_kicks", "scripts")
        for p in (_REPO / base).rglob("*.py")
        if "TRAINING_READS = frozenset" in p.read_text(encoding="utf-8")
    }
    assert found == set(_ENTRY_POINTS.values()), f"TRAINING_READS population drifted: {found}"

    assert not _DECLARATION_EXCEPTIONS, "reviewed exception list must stay empty unless justified"
    for mod_name, rel in _ENTRY_POINTS.items():
        mod = importlib.import_module(mod_name)
        declared = set(mod.TRAINING_READS)
        assert declared and declared <= set(DIMENSIONS)
        derived = _derive_reads((_REPO / rel).read_text(encoding="utf-8"))
        assert derived == declared, f"{mod_name}: derived {derived} != declared {declared}"
