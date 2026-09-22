"""Local reduce-path + smoke tests for the TF-56 positioning-gap validity driver (AMENDED 2026-09-22).

The [[feedback_test_trainer_locally_before_dgx]] gate: the driver's pooled reduce (the GO/NO-GO
composite the metric's ship-status depends on) is exercised LOCALLY, on synthetic shards + one tiny
real-fixture _measure_match, BEFORE any DGX corpus run. The battery is the AMENDED predictive gate
(corr(gap_t, conceded_threat_{t+dt})) -- the dose/responsiveness battery was retired. Frame-level
correctness is the owner run's job; provenance wiring is pinned by tests/scripts/test_provenance_wiring.py.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scripts.build_tf56_positioning_validity as drv


def _shard(*, n: int, gap_fn, conceded_fn, feasible: int, n_units: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        unit = i % n_units
        gap = float(gap_fn(i, unit))
        rows.append(
            {
                "game_id": str(unit),
                "period_id": 1,
                "frame_id": i,
                "team_id": str(10 + unit),
                "positioning_gap": gap,
                "conceded_threat_5s": float(conceded_fn(i, unit, gap)),
                "conceded_threat_10s": float(conceded_fn(i, unit, gap)),
                "n_feasible_proposals": feasible,
                "sa_converged": False,
            }
        )
    return pd.DataFrame(rows, columns=drv._SHARD_COLUMNS)


def test_source_is_ascii():
    src = Path(drv.__file__).read_text(encoding="utf-8")
    assert src.isascii(), "driver source must be ASCII (Windows --help)"


def test_shard_schema_declaration_matches_the_built_shard():
    """The declared _SHARD_COLUMNS must equal the keys _measure_match actually builds (4.77.1 trap)."""
    # AST: the dict literal built inside _measure_match's row loop must carry exactly _SHARD_COLUMNS.
    tree = ast.parse(Path(drv.__file__).read_text(encoding="utf-8"))
    built: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict) and any(
            isinstance(k, ast.Constant) and k.value == "positioning_gap" for k in node.keys if k is not None
        ):
            built = {str(k.value) for k in node.keys if isinstance(k, ast.Constant)}
            break
    assert built == set(drv._SHARD_COLUMNS), f"built {sorted(built)} != declared {sorted(drv._SHARD_COLUMNS)}"


def test_reduce_go_on_a_predictive_discriminating_corpus():
    rng = np.random.default_rng(0)
    # gap varies across units + noise; conceded = 0.5*gap + noise -> positive, significant correlation.
    shards = [
        _shard(
            n=300,
            gap_fn=lambda i, u: 1.0 + u + rng.normal(0.0, 0.3),
            conceded_fn=lambda i, u, gap: 0.5 * gap + rng.normal(0.0, 0.2),
            feasible=10,
        )
    ]
    verdict = drv.reduce_positioning_verdicts(drv.pool_positioning_shards(shards))
    assert verdict["predictive"] == "predictive"
    assert verdict["predictive_r"] > 0.0 and verdict["predictive_p"] < 0.05
    assert verdict["discriminating"] is True
    assert verdict["non_degenerate"] is True
    assert verdict["composite"] == "go"


def test_reduce_no_go_when_not_predictive():
    rng = np.random.default_rng(1)
    # conceded independent of gap -> no correlation -> not_predictive -> no_go.
    shards = [
        _shard(
            n=300,
            gap_fn=lambda i, u: 1.0 + u + rng.normal(0.0, 0.3),
            conceded_fn=lambda i, u, gap: rng.normal(0.0, 1.0),
            feasible=10,
        )
    ]
    verdict = drv.reduce_positioning_verdicts(drv.pool_positioning_shards(shards))
    assert verdict["predictive"] == "not_predictive"
    assert verdict["composite"] == "no_go"


def test_reduce_no_go_when_not_discriminating():
    # constant gap across units -> unit_gap_std 0 -> not discriminating -> no_go (even if predictive).
    shards = [_shard(n=300, gap_fn=lambda i, u: 2.0, conceded_fn=lambda i, u, gap: 1.0, feasible=10)]
    verdict = drv.reduce_positioning_verdicts(drv.pool_positioning_shards(shards))
    assert verdict["discriminating"] is False
    assert verdict["composite"] == "no_go"


def test_reduce_arm_unscoreable_on_a_thin_corpus():
    # below MIN_DOMAIN_FRAMES -> the predictive verdict short-circuits arm_unscoreable -> composite no_go.
    rng = np.random.default_rng(2)
    shards = [
        _shard(
            n=10,
            gap_fn=lambda i, u: 1.0 + u + rng.normal(0.0, 0.3),
            conceded_fn=lambda i, u, gap: 0.5 * gap,
            feasible=10,
        )
    ]
    verdict = drv.reduce_positioning_verdicts(drv.pool_positioning_shards(shards))
    assert verdict["predictive"] == "arm_unscoreable"
    assert verdict["composite"] == "no_go"


def test_reduce_empty_corpus_is_no_go():
    verdict = drv.reduce_positioning_verdicts(drv.pool_positioning_shards([]))
    assert verdict["composite"] == "no_go"
    assert verdict["n_domain"] == 0


class _FakeXt:
    """Minimal xt stub: ``rate`` returns the injected per-action ``v`` column (the conceded value)."""

    def rate(self, actions):
        return actions["v"].to_numpy(dtype=float)


def test_conceded_threat_windows_select_attacking_actions_only():
    xt = _FakeXt()
    actions = pd.DataFrame(
        {
            "period_id": [1, 1, 1, 1, 2],
            "time_seconds": [100.0, 101.0, 103.0, 120.0, 102.0],
            "team_id": [2, 2, 2, 1, 2],  # team 2 attacks; team 1 defends
            "v": [0.9, 0.3, 0.2, 0.5, 0.7],
        }
    )
    # STRICTLY-FUTURE window (t0, t0+w) period 1 defending=1: t=101 (0.3) + t=103 (0.2) attack; t=100 is
    # AT t0 (the decision instant -> EXCLUDED), t=120 is out of window, the defender + period 2 excluded.
    val = drv._conceded_threat(actions, xt, period_id=1, t0=100.0, window_s=10.0, defending_team_id=1)
    assert val == pytest.approx(0.5)
    # the frame-instant action at t0 is excluded (strictly-future), never leaked into the conceded sum.
    assert drv._conceded_threat(actions, xt, period_id=1, t0=100.0, window_s=0.5, defending_team_id=1) == 0.0
    # no attacking action in the window -> a real 0.0 (nothing conceded), never NaN.
    assert drv._conceded_threat(actions, xt, period_id=1, t0=200.0, window_s=10.0, defending_team_id=1) == 0.0
    # unknown frame time -> NaN (the pairing is undefined), never a fabricated 0.
    assert np.isnan(drv._conceded_threat(actions, xt, period_id=1, t0=float("nan"), window_s=10.0, defending_team_id=1))
    # empty actions -> 0.0 (finite; the empty-corpus branch runs before the t0 check).
    assert drv._conceded_threat(pd.DataFrame(), xt, period_id=1, t0=100.0, window_s=10.0, defending_team_id=1) == 0.0
    # only-positive xT progression summed (negatives dropped); actions strictly after t0.
    neg = pd.DataFrame({"period_id": [1, 1], "time_seconds": [101.0, 102.0], "team_id": [2, 2], "v": [0.4, -0.9]})
    neg_val = drv._conceded_threat(neg, xt, period_id=1, t0=100.0, window_s=10.0, defending_team_id=1)
    assert neg_val == pytest.approx(0.4)


def test_xt_fingerprint_discriminates_surfaces_for_cache_safety():
    """SHOULD-FIX 09: the shard cache key must change with the xT SURFACE, not just its file path.

    Shards are resumable across runs; token_inputs carries _xt_fingerprint(xt) so a re-fit producing a
    DIFFERENT surface yields a different generation digest -> stale cross-xT shards are never pooled
    into the composite verdict. (test_injected_xt_roundtrips proves save/load fidelity, NOT this.)
    """
    from silly_kicks.xthreat import ExpectedThreat

    a = ExpectedThreat(l=16, w=12)
    a.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    b = ExpectedThreat(l=16, w=12)
    b.xT = np.tile(np.linspace(0.0, 2.0, 16), (12, 1))  # a DIFFERENT surface
    assert drv._xt_fingerprint(a) == drv._xt_fingerprint(a)  # stable
    assert drv._xt_fingerprint(a) != drv._xt_fingerprint(b)  # discriminates surfaces


def _positioning_scene():
    """A tiny in-domain two-team velocity frame (the conftest geometry) + a toy xt + tiny params."""
    from silly_kicks.positioning import PositioningParams, SAParams
    from silly_kicks.xthreat import ExpectedThreat
    from tests.tracking._gk_test_helpers import _make_two_team_frame

    xt = ExpectedThreat(l=16, w=12)
    xt.xT = np.tile(np.linspace(0.0, 1.0, 16), (12, 1))
    frame = _make_two_team_frame(
        home_positions=[(12.0, 30.0), (12.0, 38.0), (40.0, 34.0), (46.0, 30.0)],
        away_positions=[(16.0, 30.0), (16.0, 38.0), (24.0, 34.0), (34.0, 44.0)],
        home_gk_pos=(3.0, 34.0),
        away_gk_pos=(100.0, 34.0),
        home_velocities=[(-3.0, 0.0), (-3.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
        away_velocities=[(-2.0, 0.0), (-2.0, 0.0), (-2.0, 0.0), (-2.0, 0.0)],
    )
    ball = frame["is_ball"].astype(bool)
    frame.loc[ball, "x"] = 26.0
    frame.loc[ball, "y"] = 34.0
    frame["ball_state"] = "alive"
    frame["team_in_possession"] = 2
    frame["speed_source"] = "derived"
    frame["speed"] = np.hypot(frame["vx"].astype(float), frame["vy"].astype(float))
    if "time_seconds" not in frame.columns:
        frame["time_seconds"] = 100.0
    params = PositioningParams(sa=SAParams(num_iterations=25, patience=25))
    return frame, xt, params


def test_measure_match_smoke_produces_a_valid_shard():
    frame, xt, params = _positioning_scene()
    # team 2 attacks; a couple of team-2 move actions in the frame's window feed the conceded leg.
    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [101.0, 104.0],
            "team_id": [2, 2],
            "type_id": [0, 0],
            "start_x": [60.0, 70.0],
            "start_y": [34.0, 34.0],
            "end_x": [70.0, 85.0],
            "end_y": [34.0, 34.0],
        }
    )
    item = ("synthetic", "m1", actions, frame, 1)
    shard, counts = drv._measure_match(item, xt=xt, params=params)
    assert list(shard.columns) == drv._SHARD_COLUMNS
    assert counts["n_matches"] == 1
    assert counts["n_frames_in"] == counts["n_frames_scored"] + counts["n_frames_dropped"]  # conservation
    assert counts["n_frames_scored"] == 1
    row = shard.iloc[0]
    assert float(row["positioning_gap"]) >= 0.0
    assert np.isfinite(float(row["conceded_threat_5s"]))  # the paired conceded-threat leg
    assert np.isfinite(float(row["conceded_threat_10s"]))


def test_injected_xt_roundtrips_and_scores_identically(tmp_path, spadl_actions):
    """PARALLEL-COHERENCE proof: a saved+loaded xT scores a shard BYTE-IDENTICALLY to the fit xT.

    The parallel protocol fits xT once (--fit-only --xt-path), then workers LOAD it. This is only
    coherent if load(save(xt)) reproduces the exact scoring surface -- otherwise each worker's shard
    would sit on a different xT and the pooled predictive correlation would be incoherent. Guards the
    ADR-100 (SK-XT-SER) round-trip AT the driver's use site, not just in the xthreat unit tests.
    """
    from silly_kicks.xthreat import ExpectedThreat

    frame, _toy_xt, params = _positioning_scene()
    xt = drv._fit_corpus_xt([("p", "m1", spadl_actions, pd.DataFrame(), 1)])  # a real fitted xT
    p = str(tmp_path / "xt.json")
    xt.save(p)
    loaded = ExpectedThreat.load(p)
    assert np.array_equal(np.asarray(loaded.xT, dtype=float), np.asarray(xt.xT, dtype=float))

    actions = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "time_seconds": [101.0, 104.0],
            "team_id": [2, 2],
            "type_id": [0, 0],
            "start_x": [60.0, 70.0],
            "start_y": [34.0, 34.0],
            "end_x": [70.0, 85.0],
            "end_y": [34.0, 34.0],
        }
    )
    item = ("synthetic", "m1", actions, frame, 1)
    s_fit, _ = drv._measure_match(item, xt=xt, params=params)
    s_loaded, _ = drv._measure_match(item, xt=loaded, params=params)
    pd.testing.assert_frame_equal(s_fit, s_loaded)  # injected xT == fit xT -> workers are coherent


def test_default_providers_resolve_an_idsse_worker_slice():
    """The --providers default must use the pining API/cache tokens so a worker resolves non-empty.

    REGRESSION: the default was "skillcorner,sportec,gradientsports", but the API/cache key for
    DFL/Sportec is ``idsse``. A worker ``--match-ids-json {"idsse":[...]}`` then hit
    ``providers_for_slice([...sportec...], {"idsse":...}) == []`` -> load_matches yielded 0 -> every
    idsse worker wrote an empty shard (found by the DGX 3-step protocol probe).
    """
    from scripts._partition import providers_for_slice

    assert "sportec" not in drv._DEFAULT_PROVIDERS  # idsse is the DFL/Sportec API+cache key
    providers = [str(p) for p in drv._DEFAULT_PROVIDERS.split(",")]  # str(): erase LiteralString -> list[str]
    kept = providers_for_slice(providers, {"idsse": ["X"]})
    assert kept == ["idsse"]  # a worker's idsse slice resolves non-empty (was [] under the sportec default)


def test_input_contract_declares_the_driver_and_symbols():
    contract = drv.input_contract()
    assert contract["driver"] == "build_tf56_positioning_validity"


def test_fit_corpus_xt_produces_a_fitted_grid(spadl_actions):
    """The DGX pass-1 within-corpus xT fit runs on raw loader SPADL (id-based; no add_names)."""
    items = [("p", "m1", spadl_actions, pd.DataFrame(), 1)]
    xt = drv._fit_corpus_xt(items)
    grid = np.asarray(xt.xT, dtype=float)
    assert grid.ndim == 2
    assert np.isfinite(grid).all()
    assert float(np.abs(grid).sum()) > 0.0  # a non-degenerate fit (near-goal xT is nonzero)
