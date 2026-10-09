"""TF-58 shared coordination driver modules: pure, self-contained pieces the D1/D2/D3 drivers compose.

Pins the params codegen (byte-reproduces commit-1, deterministic, applies MOVED selections), the broadcast-occlusion
simulator, the pre-registered threshold constants, the shared corpus plumbing (``_coordination_corpus``: the CLI
surface, the resumable source wiring, the stage timer, and ``match_tables``), and the ``_coordination_hypotheses``
reducers -- ``boundary_f1_by_gap`` (D1) plus the seven pre-registered hypotheses H1-H7 and ``gated_pass`` (§8.5),
each tested from both sides of its threshold.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import _coordination_corpus as corpus  # noqa: E402
import _coordination_hypotheses as hyp  # noqa: E402
import _coordination_occlusion as occ  # noqa: E402
import _coordination_params_codegen as cg  # noqa: E402
import _coordination_thresholds as thr  # noqa: E402

import silly_kicks.spadl.config as spadlconfig  # noqa: E402
from silly_kicks.coordination import CoordinationParams  # noqa: E402
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match  # noqa: E402


# --------------------------------------------------------------------------- codegen
def test_codegen_reproduces_the_committed_derivation_module():
    # Commit 2: the committed module is BASE_SOURCE="derivation", rendered from the authoritative DGX derivation +
    # calibration artifacts; the codegen must reproduce it byte-for-byte from those inputs (the generator-
    # reproduces-its-output rule). The empty-input (interim) render is still covered by
    # test_codegen_deterministic_and_sorted + test_interim_base_matches_R4_table below.
    repo = Path(__file__).resolve().parents[2]
    art = repo / "docs" / "research" / "tf58_team_coordination"
    derivation = json.loads((art / "derivation.json").read_text(encoding="utf-8"))
    calibration = json.loads((art / "calibration.json").read_text(encoding="utf-8"))
    committed = (repo / cg.GENERATED_PATH).read_text(encoding="utf-8")
    assert cg.render_generated_params(derivation, calibration) == committed


def test_codegen_deterministic_and_sorted():
    a = cg.render_generated_params(None, None)
    b = cg.render_generated_params(None, None)
    assert a == b
    # keys inside BASE_COORDINATION_PARAMS are emitted sorted
    base_block = a.split("BASE_COORDINATION_PARAMS", 1)[1]
    top_keys = [line.strip().strip('"').split('"')[0] for line in base_block.splitlines() if line.startswith('    "')]
    assert top_keys == sorted(top_keys)


def test_interim_base_matches_R4_table():
    b = cg.INTERIM_BASE
    assert b["butterworth_cutoff_hz"] == 0.4
    assert (b["band_low_cpm"], b["band_high_cpm"]) == (0.22, 0.83)
    assert b["welch_segment_s"] == 400.0
    assert b["possession_gap_s"] == 2.0
    assert set(b["min_observed_fraction"].values()) == {0.5}
    assert set(b["vc_epsilon"].values()) == {0.0}
    assert set(b["min_shift_s"].values()) == {60.0}


def test_codegen_with_derivation_writes_pooled_base():
    pooled = {**dict(cg.INTERIM_BASE), "welch_segment_s": 120.0}
    derivation = {"pooled": pooled, "providers": {"skillcorner": pooled}}
    text = cg.render_generated_params(derivation, None)
    assert 'BASE_SOURCE: str = "derivation"' in text
    assert '"welch_segment_s": 120.0' in text
    ns: dict = {}
    exec(compile(text, "<gen>", "exec"), ns)  # noqa: S102
    assert ns["BASE_SOURCE"] == "derivation"
    assert ns["BASE_COORDINATION_PARAMS"]["welch_segment_s"] == 120.0
    assert ns["PROVIDER_COORDINATION_PARAMS"]["skillcorner"]["welch_segment_s"] == 120.0


def test_codegen_applies_only_moved_selections_to_base_and_providers():
    pooled = dict(cg.INTERIM_BASE)
    derivation = {"pooled": pooled, "providers": {"idsse": dict(pooled)}}
    calibration = {"possession_gap_s": {"multiplier": 1.0, "offset": 0.5}}  # 2.0 -> 2.5
    ns: dict = {}
    exec(compile(cg.render_generated_params(derivation, calibration), "<gen>", "exec"), ns)  # noqa: S102
    assert ns["BASE_COORDINATION_PARAMS"]["possession_gap_s"] == 2.5
    assert ns["PROVIDER_COORDINATION_PARAMS"]["idsse"]["possession_gap_s"] == 2.5
    assert ns["BASE_COORDINATION_PARAMS"]["welch_segment_s"] == 400.0  # untouched


def _artifacts():
    pooled = {
        **dict(cg.INTERIM_BASE),
        "welch_segment_s": 120.0,
        "min_shift_s": {**cg.INTERIM_BASE["min_shift_s"], "spread": 45.0},
    }
    derivation = {"pooled": pooled, "providers": {"idsse": {**pooled, "butterworth_cutoff_hz": 0.6}}}
    calibration = {
        "confirmation": {"gate_cleared": True},
        "moved_multipliers": {"possession_gap_s": {"multiplier": 1.0, "offset": 0.5}},
    }
    return derivation, calibration


def _fields(params) -> dict:
    return {
        f.name: (
            dict(getattr(params, f.name)) if isinstance(getattr(params, f.name), Mapping) else getattr(params, f.name)
        )
        for f in dataclasses.fields(params)
        if f.compare
    }


def test_params_from_artifacts_equal_for_provider_after_regeneration():
    # M-5 artifact handoff: D3 computes from derivation.json + calibration.json, never the in-package module. The values
    # must be EXACTLY what `for_provider` returns once commit 2 commits the regenerated module -- proven by installing
    # the rendered text as the generated module in a fresh interpreter and reloading the config.
    import json
    import subprocess

    derivation, calibration = _artifacts()
    text = cg.render_generated_params(derivation, cg.calibration_moves(calibration))
    probe = (
        "import importlib, json, sys, dataclasses\n"
        "from collections.abc import Mapping\n"
        "import silly_kicks.coordination._provider_params_generated as g\n"
        "exec(sys.stdin.read(), g.__dict__)\n"
        "import silly_kicks.coordination._config as c\n"
        "importlib.reload(c)\n"
        "out = {}\n"
        "for prov in ('idsse', 'skillcorner'):\n"
        "    p = c.CoordinationParams.for_provider(prov)\n"
        "    vals = {f.name: getattr(p, f.name) for f in dataclasses.fields(p) if f.compare}\n"
        "    out[prov] = {k: (dict(v) if isinstance(v, Mapping) else v) for k, v in vals.items()}\n"
        "print(json.dumps(out, sort_keys=True, default=str))\n"
    )
    done = subprocess.run(  # noqa: S603 -- fixed argv
        [sys.executable, "-c", probe], input=text, capture_output=True, text=True, check=True
    )
    regenerated = json.loads(done.stdout)
    for prov in ("idsse", "skillcorner"):
        handed = json.loads(
            json.dumps(_fields(cg.params_from_artifacts(prov, derivation, calibration)), sort_keys=True, default=str)
        )
        assert handed == regenerated[prov], prov
    assert regenerated["idsse"]["butterworth_cutoff_hz"] == 0.6 and regenerated["idsse"]["possession_gap_s"] == 2.5
    assert regenerated["skillcorner"]["welch_segment_s"] == 120.0  # an unlisted provider gets the pooled base


def test_calibration_moves_apply_only_when_the_confirm_gate_cleared():
    derivation, calibration = _artifacts()
    held = {**calibration, "confirmation": {"gate_cleared": False}}
    assert cg.calibration_moves(held) is None and cg.calibration_moves(None) is None
    assert cg.params_from_artifacts("idsse", derivation, held).possession_gap_s == 2.0  # Tier-B value stands
    assert cg.params_from_artifacts("idsse", derivation, calibration).possession_gap_s == 2.5


def test_params_from_artifacts_refuses_without_a_derivation():
    with pytest.raises(SystemExit, match="derivation"):
        cg.params_from_artifacts("idsse", None, None)


# --------------------------------------------------------------------------- occlusion
def _mini_match(width_seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(width_seed)
    rows = []
    for fid in range(50):
        t = fid / 10.0
        bx = 20.0 + 60.0 * (fid / 49.0)  # ball sweeps 20 -> 80
        rows.append(
            {
                "game_id": 1,
                "period_id": 1,
                "frame_id": fid,
                "time_seconds": t,
                "player_id": None,
                "team_id": None,
                "is_ball": True,
                "is_goalkeeper": False,
                "x": bx,
                "y": 34.0,
            }
        )
        for team, base in ((1, 5.0), (2, 60.0)):
            for k in range(11):
                rows.append(
                    {
                        "game_id": 1,
                        "period_id": 1,
                        "frame_id": fid,
                        "time_seconds": t,
                        "player_id": team * 100 + k,
                        "team_id": team,
                        "is_ball": False,
                        "is_goalkeeper": (k == 0),
                        "x": base + k * 4.0 + rng.normal(0, 0.1),
                        "y": 6.0 + k * 5.0,
                    }
                )
    return pd.DataFrame(rows)


def test_fov_mask_clamps_to_pitch():
    f = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": 0,
            "time_seconds": 0.0,
            "player_id": [None, 1, 2],
            "team_id": [None, 1, 1],
            "is_ball": [True, False, False],
            "is_goalkeeper": False,
            "x": [2.0, 5.0, 45.0],
            "y": 34.0,
        }
    )
    m = occ.fov_mask(f, width_m=40.0)  # ball x=2 clamps window to [0, 40]
    assert bool(m[0]) is True  # ball always visible
    assert bool(m[1]) is True  # x=5 inside [0,40]
    assert bool(m[2]) is False  # x=45 outside


def test_calibrate_width_hits_target_rate():
    frames = [_mini_match(s) for s in range(2)]
    w = occ.calibrate_width(frames, target_outfield=0.6, tol=0.01)
    of, _gk = occ.detection_rates(frames[0], occ.fov_mask(frames[0], width_m=w))
    assert abs(of - 0.6) <= 0.05
    assert 5.0 <= w <= 105.0


def test_occlusion_interpolates_masked_positions_and_flags_visibility():
    f = _mini_match(3)
    out = occ.simulate_broadcast_occlusion(f, width_m=30.0)
    assert out["x"].notna().all()  # masked players still carry an (extrapolated) position
    vis = out.loc[~out["is_ball"], "visibility"]
    assert vis.notna().any() and set(pd.unique(vis.dropna())) <= {True, False}
    assert out.loc[out["is_ball"], "visibility"].isna().all()  # ball rows carry no per-player flag


# --------------------------------------------------------------------------- thresholds
def test_thresholds_match_spec_table():
    assert (thr.H1_P, thr.H1_LONGITUDINAL_MEAN_WITHIN_DEG) == (0.01, 30.0)
    assert (thr.H2_POSITIVE_SHARE, thr.H2_MEDIAN_ABS_LAG_S) == (0.70, 1.0)
    assert thr.H3_P == 0.05
    assert (thr.H4_BELOW_CEIL_SHARE, thr.H4_CEIL_CPM, thr.H4_P) == (0.95, 1.0, 0.01)
    assert (thr.H5_P, thr.H5_TOST_MARGIN, thr.H5_TOST_ALPHA) == (0.01, 0.05, 0.05)
    assert thr.H7_BC_THRESHOLD == pytest.approx(5.0 / 9.0)
    assert (thr.H7_BIMODAL_SHARE, thr.H7_SWITCH_WINDOW_S, thr.H7_SURROGATE_PERCENTILE) == (0.50, 10.0, 95.0)
    assert thr.STOPPAGE_LEG_MIN_S == 25.0
    assert thr.GATED_HYPOTHESES == ("H1", "H2", "H3", "H4", "H5", "H7")
    assert "H6" not in thr.GATED_HYPOTHESES


def test_a09_reliability_power_constants_match_spec():
    # Pre-registered A-09 per-cell power thresholds (spec §8.5 / §8.2, owner batch-3). Fixed BEFORE scoring so
    # ``min`` is never a post-hoc free parameter across the ~2000 cells.
    assert thr.RELIABILITY_MIN_N_GROUPS == 30
    assert thr.RELIABILITY_MAX_CI_HALFWIDTH == 0.25
    assert thr.CIRCULAR_RELIABILITY_MIN_RBAR == 0.10
    assert thr.OCCLUSION_MIN_MATCHES_PER_BIN == 5
    # the closed unmeasurable-reason vocabulary (C.7 schema)
    assert thr.UNMEASURABLE_REASONS == ("n<min", "ci_too_wide", "Rbar->0")


# --------------------------------------------------------------------------- corpus plumbing (_coordination_corpus)
def test_add_common_args_defaults_to_the_full_tf58_corpus():
    ap = argparse.ArgumentParser()
    corpus.add_common_args(ap)
    args = ap.parse_args([])
    assert args.providers == corpus.TF58_PROVIDERS == ("skillcorner", "gradientsports", "idsse")
    assert args.list_matches is False and args.allow_dirty is False
    assert (args.out, args.max_matches, args.match_ids_json) == (None, None, None)


def test_providers_arg_splits_a_comma_list():
    ap = argparse.ArgumentParser()
    corpus.add_common_args(ap)
    assert ap.parse_args(["--providers", "idsse, gradientsports"]).providers == ("idsse", "gradientsports")


def test_corpus_source_slices_providers_and_wraps_pining(monkeypatch):
    seen = {}

    def fake_pining_source(providers, *, match_ids, max_per_provider, cache_dir, token):
        seen.update(providers=providers, match_ids=match_ids, max_per_provider=max_per_provider, cache=cache_dir)
        return ["ref"], (lambda r: r)

    monkeypatch.setattr("scripts._loader_pining.pining_source", fake_pining_source)
    monkeypatch.setattr("scripts._loader_pining.resolve_cache_dir", lambda d: d or "CACHE")
    # a slice pinning only IDSSE ids must narrow the provider list to IDSSE (the _wanted_for_provider trap)
    monkeypatch.setattr("scripts._partition.providers_for_slice", lambda provs, mids: ["idsse"] if mids else provs)

    args = SimpleNamespace(
        providers=("skillcorner", "idsse"), match_ids_json=None, max_matches=7, cache_dir=None, token="T"
    )
    refs, load = corpus.corpus_source(args)
    assert refs == ["ref"] and callable(load)
    assert seen["providers"] == ["skillcorner", "idsse"] and seen["max_per_provider"] == 7 and seen["cache"] == "CACHE"


def test_stage_timer_accumulates_and_exposes_a_manifest_block():
    timer = corpus.StageTimer()
    with timer("load"):
        pass
    with timer("load"):
        pass
    with timer("signals"):
        pass
    d = timer.as_dict()
    assert set(d) == {"load", "signals"} and all(v >= 0.0 for v in d.values())
    assert timer.manifest() == {"stage_seconds": d}


def _synthetic_loaded(**kwargs):
    frames = make_coordination_match(seconds=450.0, provider="sportec", **kwargs)
    actions = make_coordination_actions(frames)
    return SimpleNamespace(provider="sportec", match_id="m1", frames=frames, actions=actions)


def test_match_tables_returns_a_long_frame_tagged_by_table():
    loaded = _synthetic_loaded()
    out = corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0)
    assert list(out.columns[:4]) == ["provider", "match_id", "variant", "table"]
    assert (out["provider"] == "sportec").all() and (out["match_id"] == "m1").all()
    assert (out["variant"] == "base").all()
    tables = set(out["table"])
    assert tables <= set(corpus.RESULT_TABLES)
    assert {"pair", "spectral", "windows"} <= tables  # the families reachable on this fixture


def test_match_tables_reuses_signals_across_variants_and_is_non_vacuous():
    loaded = _synthetic_loaded()
    p = CoordinationParams()
    loose = dataclasses.replace(p, vc_epsilon={k: 5.0 for k in p.vc_epsilon})  # post-preparation change only
    single = corpus.match_tables(loaded, p, n_surrogates=0).reset_index(drop=True)
    both = corpus.match_tables(loaded, p, n_surrogates=0, variants={"base": p, "loose": loose})
    assert set(both["variant"]) == {"base", "loose"}
    base_block = both[both["variant"] == "base"].reset_index(drop=True)
    # REUSE correctness: the reused-signals "base" variant equals the standalone build byte-for-byte.
    pd.testing.assert_frame_equal(base_block, single)
    # NON-VACUITY: the loose vector-coding epsilon actually moves the output off the base.
    loose_block = both[both["variant"] == "loose"].reset_index(drop=True)
    assert not base_block.drop(columns="variant").equals(loose_block.drop(columns="variant"))


def test_match_tables_records_the_per_stage_breakdown():
    # R8: windows, signal preparation, every family, the table combine and the melt/switch-event bookkeeping each get
    # their own counter in the driver's StageTimer (they accumulate across variants and matches).
    loaded = _synthetic_loaded()
    timer = corpus.StageTimer()
    corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0, include_switch_events=True, timer=timer)
    stages = timer.as_dict()
    families = (
        "relative_phase",
        "cross_correlation",
        "vector_coding",
        "coherence",
        "spectral",
        "cluster_phase",
        "relative_stretch",
        "combine",
    )
    assert {"windows", "signals", "melt", "switch_events", *(f"family.{f}" for f in families)} <= set(stages)
    assert all(v >= 0.0 for v in stages.values())


def test_match_tables_output_does_not_depend_on_the_timer():
    loaded = _synthetic_loaded()
    plain = corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0)
    timed = corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0, timer=corpus.StageTimer())
    pd.testing.assert_frame_equal(plain, timed)


def test_match_tables_refuses_a_frameless_match():
    loaded = SimpleNamespace(provider="skillcorner", match_id="x", frames=None, actions=None)
    with pytest.raises(ValueError, match="tracking frames"):
        corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0)


def test_possession_change_times_finds_attacking_transitions():
    windows = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "window_kind": "possession",
            "window_id": [0, 1, 2, 3],
            "start_time_s": [0.0, 10.0, 20.0, 30.0],
            "attacking_team_id": [1, 1, 2, 1],  # changes at 20 s and 30 s
        }
    )
    assert corpus.possession_change_times(windows)["time"].tolist() == [20.0, 30.0]


def test_match_tables_emits_switch_events_for_h7():
    loaded = _synthetic_loaded()
    out = corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0, include_switch_events=True)
    assert {"rsi_switch_times", "possession_changes"} <= set(out["table"])  # H7 inputs emitted
    sw = out[out["table"] == "rsi_switch_times"]
    assert {"match_id", "time"} <= set(sw.columns)
    off = corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0)  # default omits them
    assert "rsi_switch_times" not in set(off["table"])


# --------------------------------------------------------------------------- boundary F1 (_coordination_hypotheses)
def _two_possession_match(*, gap_lo=7.0, gap_hi=9.0, switch=15.0, seconds=30.0, hz=10.0):
    """Team A (near the ball at x~=20) possesses [0, switch) with one ball-dead NA gap [gap_lo, gap_hi); team B
    (near x~=85) possesses [switch, seconds). One within-possession gap + one team-change boundary -- so the
    boundary F1 peaks at the smallest gap that bridges the NA gap without merging the two possessions."""
    n = round(seconds * hz)
    t = np.arange(n) / hz
    frame_id = np.arange(n)
    dead = (t >= gap_lo) & (t < gap_hi)
    ball_x = np.where(t < switch, 20.0, 85.0)
    rows: list[dict[str, object]] = []

    def push(pid, team, is_ball, is_gk, x, direction):
        for i in range(n):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "frame_id": int(frame_id[i]),
                    "time_seconds": float(t[i]),
                    "frame_rate": hz,
                    "player_id": pid,
                    "team_id": team,
                    "is_ball": is_ball,
                    "is_goalkeeper": is_gk,
                    "x": float(x[i]) if hasattr(x, "__len__") else float(x),
                    "y": 34.0,
                    "z": 0.0,
                    "speed": 0.0,
                    "speed_source": "unavailable",
                    "ball_state": "dead" if (is_ball and dead[i]) else "alive",
                    "team_attacking_direction": direction,
                    "visibility": None,
                    "source_provider": "sportec",
                    "is_goalkeeper_source": "provided",
                }
            )

    push(np.nan, np.nan, True, False, ball_x, "")
    for k in range(5):
        push(100 + k, 1, False, False, np.full(n, 18.0 + k), "ltr")  # team A clustered near x=20
        push(200 + k, 2, False, False, np.full(n, 83.0 + k), "rtl")  # team B clustered near x=85
    df = pd.DataFrame(rows)
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    for c in ("ball_state", "source_provider", "is_goalkeeper_source"):
        df[c] = df[c].astype("category")
    return df


def _two_possession_actions():
    # action times aligned to the ball-move times so the event A->B boundary lands on the SAME frame as the
    # frames-only boundary (t=0 avoids an uncovered -1 prefix; t=15 == the possession switch).
    rows = [
        (0.0, 1, "pass"),
        (8.0, 1, "pass"),
        (15.0, 2, "pass"),
        (22.0, 2, "pass"),
    ]
    out = []
    for i, (t, team, name) in enumerate(rows):
        out.append(
            {
                "game_id": 1,
                "period_id": 1,
                "action_id": i,
                "time_seconds": t,
                "team_id": team,
                "player_id": team * 100,
                "type_id": spadlconfig.actiontypes.index(name),
                "result_id": spadlconfig.results.index("success"),
                "start_x": 52.5,
                "start_y": 34.0,
                "end_x": 60.0,
                "end_y": 34.0,
                "bodypart_id": spadlconfig.bodyparts.index("foot"),
            }
        )
    return pd.DataFrame(out)


def test_frame_possession_ids_maps_each_frame_to_its_covering_window():
    frames = pd.DataFrame(
        {
            "game_id": 1,
            "period_id": 1,
            "frame_id": [0, 1, 2, 3],
            "time_seconds": [0.0, 1.0, 2.0, 3.0],
        }
    )
    windows = pd.DataFrame(
        {
            "game_id": [1, 1],
            "period_id": [1, 1],
            "window_kind": ["possession", "possession"],
            "start_time_s": [0.0, 2.0],
            "end_time_s": [2.0, 4.0],
        }
    )
    assert hyp._frame_possession_ids(frames, windows).tolist() == [0, 0, 1, 1]


def _h1_pair(n_total, n_x_gt_y, circ_mean_deg):
    rows = []
    for i in range(n_total):
        rx, ry = (0.9, 0.1) if i < n_x_gt_y else (0.1, 0.9)
        for sig, r in (("centroid_x", rx), ("centroid_y", ry)):
            rows.append(
                {
                    "game_id": 1,
                    "period_id": i,
                    "window_id": 0,
                    "window_kind": "period",
                    "level": "team_team",
                    "signal_a": sig,
                    "signal_b": sig,
                    "team_a_id": 1,
                    "team_b_id": 2,
                    "coord_rp_resultant_length": r,
                    "coord_rp_mean_deg": circ_mean_deg if sig == "centroid_x" else 0.0,
                }
            )
    return pd.DataFrame(rows)


def test_h1_centroid_phase_stability_both_sides():
    assert hyp.h1_centroid_phase_stability(_h1_pair(20, 20, 25.0))["pass"] is True
    assert hyp.h1_centroid_phase_stability(_h1_pair(20, 12, 25.0))["pass"] is False  # sign test not significant
    assert hyp.h1_centroid_phase_stability(_h1_pair(20, 20, 35.0))["pass"] is False  # circular mean outside +-30


def _h2_pair(n, pos_share, abs_lag):
    n_pos = round(pos_share * n)
    rows = [
        {
            "game_id": 1,
            "period_id": i,
            "window_id": 0,
            "window_kind": "period",
            "level": "team_team",
            "signal_a": "spread",
            "signal_b": "spread",
            "team_a_id": 1,
            "team_b_id": 2,
            "coord_xc_r_at_max": 0.5 if i < n_pos else -0.5,
            "coord_xc_lag_s": abs_lag,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_h2_spread_xcorr_both_sides():
    assert hyp.h2_spread_xcorr(_h2_pair(20, 0.75, 0.9))["pass"] is True
    assert hyp.h2_spread_xcorr(_h2_pair(20, 0.65, 0.9))["pass"] is False  # positive share below 0.70
    assert hyp.h2_spread_xcorr(_h2_pair(20, 0.75, 1.1))["pass"] is False  # median |lag| above 1.0


def _h3(separated, *, signal_phase=1, team: object = 1):
    # phases are 1..n_phases (spec 7.6: phase k holds the samples in ((k-1)/n, k/n]); Moura's early third is phase 1.
    # The separated signal is planted in `signal_phase`; every other phase is neutral.
    pp, win = [], []
    for i in range(20):
        term = "shot" if i < 10 else "tackle"
        for phase in (1, 2, 3):
            anti = (0.6 if term == "shot" else 0.4) if separated and phase == signal_phase else 0.5
            pp.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "window_id": i,
                    "window_kind": "possession",
                    "level": "team_team",
                    "signal_a": "spread",
                    "signal_b": "spread",
                    "phase_index": phase,
                    "team_a_id": team,
                    "coord_vc_pct_anti_phase": anti,
                    "coord_vc_pct_a_phase": anti,
                    "coord_vc_pct_b_phase": 0.0,
                }
            )
        win.append(
            {
                "game_id": 1,
                "period_id": 1,
                "window_id": i,
                "window_kind": "possession",
                "terminal_action": term,
                "attacking_team_id": team,
            }
        )
    return pd.DataFrame(pp), pd.DataFrame(win)


def test_h3_early_third_by_terminal_both_sides():
    assert hyp.h3_early_third_by_terminal(*_h3(True))["pass"] is True
    assert hyp.h3_early_third_by_terminal(*_h3(False))["pass"] is False


def test_h3_reads_the_early_third_which_is_phase_one():
    # round-2 finding: H3 selected phase_index == 0, which no row has (phases are 1-based), so H3 could never be
    # evaluated on real data. The early third is phase 1; the same signal planted in phase 2 must not count.
    assert hyp.h3_early_third_by_terminal(*_h3(True, signal_phase=1))["pass"] is True
    assert hyp.h3_early_third_by_terminal(*_h3(True, signal_phase=2))["pass"] is False


def test_h3_identifies_the_attacking_team_by_string_id():
    # review A-11: ids compare through id_compat, never pd.to_numeric (an IDSSE DFL-CLU-... id coerces to NaN, so the
    # attacking team was never team A and the attack-phase fraction read team B's column)
    assert hyp.h3_early_third_by_terminal(*_h3(True, team="DFL-CLU-00000A"))["pass"] is True


def _h4_spectral(all_below, first_gt, *, fast_other_signal=False, spread_above=False):
    # spec 8.5 H4 is Moura 2013's: the median frequency of team AREA and SPREAD (convex_hull_area, spread)
    rows = []
    for i in range(20):
        for signal in ("convex_hull_area", "spread"):
            f1 = 0.5 if (all_below or i % 2 == 0) else 1.5
            if spread_above and signal == "spread":
                f1 = 1.5
            f2 = f1 - 0.1 if first_gt else f1 + 0.1
            for period, f in ((1, f1), (2, max(f2, 0.05))):
                rows.append(
                    {
                        "game_id": i,
                        "period_id": period,
                        "window_kind": "period",
                        "team_id": 1,
                        "signal": signal,
                        "coord_median_freq_cpm": f,
                    }
                )
        if fast_other_signal:  # a signal Moura never measured, oscillating far above the ceiling
            for period in (1, 2):
                rows.append(
                    {
                        "game_id": i,
                        "period_id": period,
                        "window_kind": "period",
                        "team_id": 1,
                        "signal": "stretch_x",
                        "coord_median_freq_cpm": 3.0,
                    }
                )
    return pd.DataFrame(rows)


def test_h4_median_frequency_both_sides():
    assert hyp.h4_median_frequency(_h4_spectral(True, True))["pass"] is True
    assert hyp.h4_median_frequency(_h4_spectral(False, True))["pass"] is False  # not enough below the ceiling
    assert hyp.h4_median_frequency(_h4_spectral(True, False))["pass"] is False  # second half not lower


def test_h4_reads_only_mouras_area_and_spread():
    # review A-13: H4 pooled every team signal (+ the possession series); a fast signal Moura never measured must
    # not flip it, and BOTH of Moura's signals must hold (the paper reports area and spread separately)
    assert hyp.h4_median_frequency(_h4_spectral(True, True, fast_other_signal=True))["pass"] is True
    assert hyp.h4_median_frequency(_h4_spectral(True, True, spread_above=True))["pass"] is False


def _h5(x_gt_y_count, n_halves, tost_diff, seed=0, ids=lambda t: t):
    rng = np.random.default_rng(seed)
    ct = []
    for i in range(n_halves):
        rx, ry = (0.8, 0.2) if i < x_gt_y_count else (0.2, 0.8)
        for axis, r in (("x", rx), ("y", ry)):
            ct.append(
                {
                    "game_id": 1,
                    "period_id": i,
                    "window_id": 0,
                    "window_kind": "period",
                    "team_id": 1,
                    "axis": axis,
                    "coord_rho_group_mean": r,
                }
            )
    win = []
    for t in range(1, 11):
        d = tost_diff + rng.normal(0, 0.005)
        for wid, att, team, rho in ((t * 2, ids(t), ids(t), 0.5 + d), (t * 2 + 1, ids(999), ids(t), 0.5)):
            win.append(
                {"game_id": 1, "period_id": 1, "window_id": wid, "window_kind": "possession", "attacking_team_id": att}
            )
            ct.append(
                {
                    "game_id": 1,
                    "period_id": 1,
                    "window_id": wid,
                    "window_kind": "possession",
                    "team_id": team,
                    "axis": "x",
                    "coord_rho_group_mean": rho,
                }
            )
    return pd.DataFrame(ct), pd.DataFrame(win)


def test_h5_rho_group_both_sides():
    assert hyp.h5_rho_group(*_h5(20, 20, 0.01))["pass"] is True
    assert hyp.h5_rho_group(*_h5(12, 20, 0.01))["pass"] is False  # x not longitudinally > y
    assert hyp.h5_rho_group(*_h5(20, 20, 0.08))["pass"] is False  # possession NOT equivalent (TOST fails)


def test_h5_splits_possession_by_string_team_id():
    # review A-11: with IDSSE-style string ids every row fell "out of possession" (NaN == NaN is False), so the TOST
    # had no in-possession mean; through id_compat the in/out split is real again, on both sides of the margin
    ids = lambda t: f"DFL-CLU-{t:06d}"  # noqa: E731
    assert hyp.h5_rho_group(*_h5(20, 20, 0.01, ids=ids))["pass"] is True
    assert hyp.h5_rho_group(*_h5(20, 20, 0.08, ids=ids))["pass"] is False


def test_h6_dyad_is_descriptive():
    pair = pd.DataFrame(
        [
            {"level": "dyad", "axis": "x", "coord_rp_pct_near_in_phase": 0.5},
            {"level": "dyad", "axis": "y", "coord_rp_pct_near_in_phase": 0.3},
        ]
    )
    out = hyp.h6_dyad_near_in_phase(pair)
    assert out["pass"] is None and len(out["quartiles_x"]) == 3


def _h7_rsi(bc_share):
    rows = [
        {
            "game_id": 1,
            "period_id": i,
            "window_kind": "period",
            "axis": "x",
            "coord_rsi_bimodality_coefficient": 0.7 if i < round(bc_share * 20) else 0.3,
        }
        for i in range(20)
    ]
    return pd.DataFrame(rows)


def _h7_events(coupled, seed=0):
    # IRREGULAR change times: a constant time-shift then decouples the switches (a periodic grid would just re-phase
    # them, leaving the surrogate share bimodal). Coupled switches sit 2 s after each change (within the 10 s window).
    rng = np.random.default_rng(seed)
    changes = np.sort(rng.uniform(0.0, 600.0, 30))
    switches = changes + 2.0 if coupled else np.sort(rng.uniform(0.0, 600.0, 30))
    return (
        pd.DataFrame({"match_id": "m", "time": switches}),
        pd.DataFrame({"match_id": "m", "time": changes}),
    )


def test_h7_rsi_both_sides():
    sw_coupled, changes = _h7_events(coupled=True)
    sw_random, _ = _h7_events(coupled=False)
    assert hyp.h7_rsi(_h7_rsi(0.6), sw_coupled, changes, seed=0)["pass"] is True
    assert hyp.h7_rsi(_h7_rsi(0.4), sw_coupled, changes, seed=0)["pass"] is False  # BC share below 0.50
    assert hyp.h7_rsi(_h7_rsi(0.6), sw_random, changes, seed=0)["pass"] is False  # switches below the surrogate


def _switches(rows):
    return pd.DataFrame(rows, columns=["match_id", "game_id", "period_id", "axis", "time"])


def _changes(rows):
    return pd.DataFrame(rows, columns=["match_id", "game_id", "period_id", "time"])


def test_h7_never_matches_a_switch_to_another_periods_change():
    # review A-12: time is period-relative -- a second-half switch at 600 s is NOT "after" a first-half change at
    # 595 s; switches match only their own match-half's possession changes
    sw = _switches([("m", "m", 2, "x", 600.0)])
    obs, _ = hyp._switch_vs_change_surrogate(sw, _changes([("m", "m", 1, 595.0)]), seed=0, n_surrogates=9)
    assert obs == 0.0
    obs_same, _ = hyp._switch_vs_change_surrogate(sw, _changes([("m", "m", 2, 595.0)]), seed=0, n_surrogates=9)
    assert obs_same == 1.0  # the other side: the same half's change does count


def test_h7_share_is_the_share_of_switches_pooled_over_every_switch():
    # spec 8.5 H7: "share of switches within 10 s after a possession change" -- pooled over the switches (3 of 4 here),
    # not a mean of per-match shares (which would weigh a one-switch match like a three-switch one: 0.5)
    sw = _switches(
        [("a", "a", 1, "x", 100.0), ("a", "a", 1, "x", 200.0), ("a", "a", 1, "y", 300.0), ("b", "b", 1, "x", 50.0)]
    )
    ch = _changes([("a", "a", 1, 95.0), ("a", "a", 1, 195.0), ("a", "a", 1, 295.0), ("b", "b", 1, 400.0)])
    obs, surro = hyp._switch_vs_change_surrogate(sw, ch, seed=0, n_surrogates=9)
    assert obs == pytest.approx(0.75)
    assert surro.size == 9


def test_h7_uses_one_preregistered_seed_and_k_in_both_drivers():
    # review A-41 / A-53: D2's gate and D3's report ran H7 with different seeds (0 vs 20260926) and K = 200 against the
    # spec's 199 convention -- one pre-registered seed and K now, and no driver passes its own
    import ast
    from pathlib import Path

    import _coordination_thresholds as thr

    assert thr.H7_N_SURROGATES == 199
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    for driver in ("calibrate_coordination", "validate_team_coordination"):
        tree = ast.parse((scripts / f"{driver}.py").read_text(encoding="utf-8"))
        calls = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "evaluate_hypotheses"
        ]
        assert calls, driver
        for call in calls:
            assert not [kw for kw in call.keywords if kw.arg == "seed"], (driver, call.lineno)


def test_gated_pass_ignores_h6():
    good: dict[str, dict[str, bool | None]] = {h: {"pass": True} for h in ("H1", "H2", "H3", "H4", "H5", "H7")}
    good["H6"] = {"pass": None}
    assert hyp.gated_pass(good) is True
    assert hyp.gated_pass({**good, "H2": {"pass": False}}) is False
    assert hyp.gated_pass({**good, "H6": {"pass": False}}) is True  # H6 is descriptive, never gates


def test_boundary_f1_by_gap_prefers_true_gap():
    frames = _two_possession_match(gap_lo=7.0, gap_hi=9.0)  # a 2 s within-possession NA gap
    actions = _two_possession_actions()
    curve = hyp.boundary_f1_by_gap(frames, actions, [1.0, 3.0, 5.0])
    # a too-small gap splits team A's possession at the NA gap -> a false boundary -> lower F1;
    # gaps that bridge it (>= 2 s) recover the single true A->B boundary.
    assert curve[3.0] > curve[1.0]
    assert curve[3.0] == curve[5.0]  # both bridge the 2 s gap; nothing else to merge
    from derive_coordination_params import possession_gap_argmax

    assert possession_gap_argmax(curve) == 3.0  # the smallest bridging gap on this grid (ties -> smaller)


def test_boundary_f1_by_gap_infers_the_carrier_once(monkeypatch):
    # F3 (speed): ball-carrier inference is gap-invariant. boundary_f1_by_gap must infer it ONCE and
    # thread carrier= into the per-gap possession rebuild -- NOT re-infer (and re-_pre_index_frames,
    # ~79-87% of pass-b) per candidate gap.
    import silly_kicks.tracking as tracking

    real = tracking.infer_ball_carrier
    calls = {"n": 0}

    def _counting(frames, *a, **k):
        calls["n"] += 1
        return real(frames, *a, **k)

    monkeypatch.setattr(tracking, "infer_ball_carrier", _counting)
    frames = _two_possession_match(gap_lo=7.0, gap_hi=9.0)
    actions = _two_possession_actions()
    curve = hyp.boundary_f1_by_gap(frames, actions, [1.0, 3.0, 5.0])
    assert calls["n"] == 1, f"infer_ball_carrier called {calls['n']}x over 3 gaps (gap-invariant -> expect 1)"
    # byte-identity backstop: the curve is unchanged vs the per-gap version
    assert curve[3.0] > curve[1.0] and curve[3.0] == curve[5.0]


def test_provider_bootstrap_se_byte_identical_after_index_resample():
    # F2 (speed): the match-resample bootstrap drops the 1000x pd.concat for per-match row-index arrays
    # + a single .iloc per draw. Row order within a draw is identical (drawn-match order, rows in situ)
    # -> byte-identical SE (the F2 refactor is unchanged). The seed entropy is input_contract()["digest"], which
    # moved with the commit-2 derivation module + OBJECTIVE_VERSION, so the goldens are re-baselined to the
    # current digest (originally captured from the pre-refactor concat version @ 9b99003).
    from derive_coordination_params import provider_bootstrap_se

    u = pd.DataFrame({"match": ["a", "a", "b", "c", "c", "c", "d"], "value": [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 0.5]})
    se = provider_bootstrap_se(
        u, lambda df: float(df["value"].mean()), seed_key=("relative_phase", "band_low_cpm"), n_boot=1000
    )
    assert se == pytest.approx(1.0089459531423133, rel=0.0, abs=0.0)
    # order-sensitive reducer -> proves per-draw row order is preserved (ADR-105)
    se2 = provider_bootstrap_se(
        u,
        lambda df: float((df["value"].to_numpy() * np.arange(1, len(df) + 1)).sum() / len(df)),
        seed_key=("spectral", "x"),
        n_boot=500,
    )
    assert se2 == pytest.approx(6.696609165950613, rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    "driver",
    [
        "derive_coordination_params",
        "calibrate_coordination",
        "validate_team_coordination",
        "validate_coordination_numerics",
    ],
)
def test_driver_input_contract_imports_under_python_dash_m(driver):
    # `python -m scripts.<driver>` puts the repo root on sys.path, never scripts/: the contract import must not assume
    # the scripts directory (a bare `from _input_contract import ...` raised ModuleNotFoundError there).
    repo = Path(__file__).resolve().parents[2]
    code = f"import scripts.{driver} as d; print(d.input_contract()['driver'])"
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], cwd=repo, capture_output=True, text=True, timeout=300
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == driver


@pytest.mark.parametrize("work", ["match_tables", "metrics_match", "numerics_match"])
def test_every_corpus_pass_refuses_a_discarded_detection_flag(work):
    # spec 8.3 / ADR-069 Layer 2 as concretised (ADR-111): the per-match work of D2 (match_tables), D3 and the
    # numerics gate refuses a detection-aware match whose `visibility` was discarded, before any signal is computed;
    # for_each records the match as failed and every combine refuses a failed key (B-1), so no artifact covers it.
    import validate_coordination_numerics as nf
    import validate_team_coordination as d3

    frames = make_coordination_match(seconds=60.0, provider="skillcorner")
    frames["visibility"] = None
    loaded = SimpleNamespace(provider="skillcorner", match_id="x", frames=frames, actions=None)
    call = {
        "match_tables": lambda: corpus.match_tables(loaded, CoordinationParams(), n_surrogates=0),
        "metrics_match": lambda: d3.metrics_match(loaded),
        "numerics_match": lambda: nf.numerics_match(loaded),
    }[work]
    with pytest.raises(ValueError, match=r"tracking\.skillcorner"):
        call()


# --------------------------------------------------------------------------- occlusion per-bin CI (ADR-112 follow-up)
def _occ_ci(values, weights, matches):
    from scripts.derive_coordination_params import _bootstrap_weighted_median_ci

    return _bootstrap_weighted_median_ci(
        np.asarray(values, dtype=float), np.asarray(weights, dtype=float), np.asarray(matches, dtype=object)
    )


def test_occlusion_ci_is_match_grain_not_row_grain():
    # the per-bin CI resamples MATCHES, not rows: a bin whose rows all come from ONE match has no between-match
    # variation to bootstrap -> NaN, however many rows it holds; two matches -> a finite, ordered CI.
    rng = np.random.default_rng(0)
    vals = rng.normal(size=400)
    w = np.ones(400)
    assert np.all(np.isnan(_occ_ci(vals, w, ["m1"] * 400)))
    lo, hi = _occ_ci(vals, w, (["m1"] * 200) + (["m2"] * 200))
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi


def test_occlusion_ci_fewer_than_two_values_is_nan():
    assert np.all(np.isnan(_occ_ci([1.0], [1.0], ["m1"])))
    assert np.all(np.isnan(_occ_ci([np.nan, np.nan], [1.0, 1.0], ["m1", "m2"])))


def test_occlusion_ci_caps_every_draw_sort(monkeypatch):
    # the D1-reduce hotspot bound: a million-row bin must never sort a million rows 400x. Every weighted_quantile the
    # bootstrap calls sees at most _OCCLUSION_CI_MAX_ROWS rows (the per-match cap balances clusters so even resampling
    # an uneven cluster repeatedly cannot blow the bound), while the CI still comes back finite.
    import scripts.derive_coordination_params as derive

    n = 600_000
    rng = np.random.default_rng(1)
    vals = rng.normal(size=n)
    w = np.ones(n)
    matches = rng.integers(0, 37, size=n)  # 37 uneven clusters
    seen: list[int] = []
    real = derive.weighted_quantile

    def _spy(v, q, ww):
        seen.append(int(np.asarray(v).size))
        return real(v, q, ww)

    monkeypatch.setattr(derive, "weighted_quantile", _spy)
    lo, hi = derive._bootstrap_weighted_median_ci(vals, w, matches.astype(object))
    cap = derive._OCCLUSION_CI_MAX_ROWS
    assert seen, "bootstrap never scored a draw"
    assert max(seen) <= cap, f"a draw sorted {max(seen)} > cap {cap}"
    assert np.isfinite(lo) and np.isfinite(hi)


def test_occlusion_ci_is_seeded_deterministic():
    rng = np.random.default_rng(2)
    vals = rng.normal(size=5000)
    w = rng.uniform(0.5, 1.5, size=5000)
    matches = rng.integers(0, 8, size=5000)
    assert _occ_ci(vals, w, matches) == _occ_ci(vals, w, matches)
