"""Commit-2 gate on the committed DAS corpus artifacts (das-native spec sections 4 and 7.2;
combined-cycle-completion spec 12 D1/D2).

Two §4.2 speed targets are ADJUSTED to the measured reality (combined-cycle Phase B amendment 4, owner
decision 2026-10-05), surfaced and recorded, never silently relaxed:
* ``add_das_speedup`` >= 6x (not the aspirational 10x): add_das is end-to-end, diluted by the
  ``link_actions_to_frames`` the old path also ran; the raw engine speedup is 15x (``ref_over_numba_serial``).
* ``prange_efficiency@16`` >= 0.55 (not 0.6): the shipped ADR-107 cost constant is already 0.49, and the
  corpus measured 0.576 -- better than the constant, below the original aspiration.
The other three targets keep their original das-native §4.2 values.
"""

import json
import math
import re
from pathlib import Path

import pytest

_DIR = Path("docs/research/das_native_parity")
_M = json.loads((_DIR / "metrics.json").read_text(encoding="utf-8"))
_P = json.loads((_DIR / "performance.json").read_text(encoding="utf-8"))
_PROVIDERS = sorted(_M["providers"])
# das-native spec section 4.2 -- copied; add_das/prange adjusted to measured reality (amendment 4), the rest verbatim.
_TARGETS = {"ref_over_numba_serial": 10.0, "ref_over_numpy": 2.0, "add_das_speedup": 6.0, "das_xfns_speedup": 50.0}
_PRANGE_EFFICIENCY_AT_16 = 0.55


def test_provenance_is_clean_and_single_commit():
    for doc in (_M, _P):
        assert doc["run_tree_dirty"] is False
        assert re.fullmatch(r"[0-9a-f]{40}", doc["run_commit"])
    assert _M["run_commit"] == _P["run_commit"]
    assert _M["n_failed"] == 0
    assert _M["commit_consistent"] is True and _M["commits_seen"] == [_M["run_commit"]]


def test_population_is_the_pre_registered_one():
    # Completeness itself is enforced by the reduce, which refuses an unaccounted key (B r2 C7);
    # this pins WHICH population was reduced (the Task 17 Step 5 owner-token counts).
    assert _M["population"]["listed_per_provider"] == {"gradientsports": 64, "idsse": 7, "skillcorner": 909}


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_parity_inside_the_golden_bounds_in_all_four_cells(provider):
    p = _M["providers"][provider]
    counts = p["n_outside_golden_bound"]
    for eng in ("numpy", "numba"):
        for grain in ("team", "player"):
            assert counts[eng][grain] == {"das": 0, "as": 0}, (eng, grain)
    for grain in ("team", "player"):  # no vacuous numba cell (B r2 CCC-PLAN-23)
        for out in ("das", "as"):
            assert p["numba_compared"][grain][out] > 0, (grain, out)
            assert math.isfinite(p["numba_vs_numpy_max_abs"][grain][out]), (grain, out)


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_zero_finite_mask_mismatches_and_counts_recorded(provider):
    p = _M["providers"][provider]
    assert p["finite_mask_mismatches"] == {"team": 0, "player": 0}
    assert set(p["finite_counts"]) == {"team", "player"}
    assert p["d_key_frames"] >= 0 and p["direction"]["n_compared"] > 0


def test_the_d_key_figure_is_recorded():
    # The D-KEY figure (frames whose (game, frame_id) recurs in another period) is recorded per provider.
    # MEASURED on the full owner corpus (amendment 4): 0 of 980 matches reuse a frame_id across periods, so
    # the collision-free key changed nothing HERE -- a real, recorded result, not a gap. The keying code
    # path itself is exercised by tests/tracking/test_das_pack.py (D-KEY) and test_das_divergences.py.
    assert all(_M["providers"][p]["d_key_frames"] >= 0 for p in _PROVIDERS)
    assert sum(_M["providers"][p]["d_key_frames"] for p in _PROVIDERS) == 0


@pytest.mark.parametrize("provider", _PROVIDERS)
def test_quadrature_shift_is_recorded(provider):
    q = _M["providers"][provider]["quadrature_shift_das"]
    assert all(math.isfinite(q[k]) for k in ("median", "p90", "max"))


@pytest.mark.parametrize("name", sorted(_TARGETS))
def test_speed_target(name):
    assert _P["summary"][name] >= _TARGETS[name], f"{name} = {_P['summary'][name]:.2f} < {_TARGETS[name]}"


def test_prange_efficiency_at_16_threads():
    assert _P["summary"]["prange_efficiency"]["16"] >= _PRANGE_EFFICIENCY_AT_16


def test_benchmark_ran_on_a_quiet_box():
    c = _P["contention"]
    assert c["foreign_cpu_fraction"] is not None and c["foreign_cpu_fraction"] < c["max"]


def test_both_paths_ran_in_one_pandas_major():
    majors = {r[k].split(".")[0] for r in _P["per_match"] for k in ("pandas_new", "pandas_old") if k in r}
    assert majors == {"2"}


def test_every_path_run_that_did_not_fit_is_recorded():
    # Phase B amendment F7: a path run above the memory ceiling is a recorded RESULT (the old path cannot
    # score a GS match within it). Each speedup still rests on at least one finished pair, and the new path
    # never hits the ceiling.
    s = _P["summary"]
    assert s["speedup_n"]["add_das"] > 0 and s["speedup_n"]["das_xfns"] > 0
    assert s["new_path_over_memory"] == {}
    for r in _P["per_match"]:
        rec = r.get("old_path_over_memory")
        if rec is not None:
            assert rec["limit_gib"] == _P["path_memory_limit_gib"]
            assert rec["phase"] in {"load", "add_das", "das_xfns"}


def test_adr108_quotes_the_artifact():
    adr = next(Path("docs/superpowers/adrs").glob("ADR-108-*.md")).read_text(encoding="utf-8")
    assert "COMMIT-2 PLACEHOLDER" not in adr
    for provider in _PROVIDERS:
        assert f"{_M['providers'][provider]['quadrature_shift_das']['median']:.3g}" in adr
