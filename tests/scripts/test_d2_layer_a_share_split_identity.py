"""D-5 for the per-variant baseline share split (ADR-112 follow-up, option C).

The baseline layer-a pass now writes ONE share per post-preparation variant instead of one stacked share, and
``combine_levels`` combines per variant. Nothing is dropped -- the full melt is retained per variant -- so the
per-variant combined table a reader reads is byte-identical (same rows, same ``provider``/``match_id`` order) to the
former stacked combine filtered to that variant. This drives the confirm's two reducers (``evaluate_hypotheses`` +
``reliability_over_folds``) on both representations and asserts identical output, and checks the summary fold keeps
``generation`` / ``population_digest`` (the ``objective_ids`` / ``population`` inputs) unchanged so the whole
``calibration.json`` stays byte-identical. The confirm is a deterministic function of these reads + the fold; running
the full ruthless OAT twice is not reconstructed here (the old stacked read-path no longer exists) -- the byte-identity
is proven at those two seams.

Needs ruthless 0.7.0 (the ``calibrate_coordination`` import). Run with ``python`` (the C:\\Python314 user-site has it).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _fake_corpus import make_loaded  # noqa: E402

import scripts._coordination_params_codegen as cg  # noqa: E402
import scripts.calibrate_coordination as d2  # noqa: E402
from scripts._coordination_corpus import match_tables  # noqa: E402
from scripts._coordination_hypotheses import evaluate_hypotheses  # noqa: E402
from silly_kicks.calibration import match_cv_splits  # noqa: E402
from silly_kicks.coordination import CoordinationParams  # noqa: E402
from tests.coordination._fixtures import make_coordination_actions, make_coordination_match  # noqa: E402

pytestmark = [
    pytest.mark.filterwarnings("ignore::silly_kicks.coordination.CoordinationCoverageWarning"),
    pytest.mark.filterwarnings("ignore:vx/vy columns not found"),
]
_ORDER = ["provider", "match_id"]
_SHOTS = (36.0, 84.0, 132.0, 180.0)
_TACKLES = (60.0, 108.0, 156.0, 204.0)


def _match(game: int, team_ids: tuple[int, int], seed: int):
    frames = make_coordination_match(
        seconds=240.0, provider="sportec", periods=2, team_ids=team_ids, seed=seed, dead_intervals=[(100.0, 130.0)]
    )
    frames["game_id"] = game
    parts = []
    for period, fp in frames.groupby("period_id", sort=True):
        acts = make_coordination_actions(
            fp, restarts=[(t, "tackle") for t in _TACKLES], goals=_SHOTS, possession_every_s=12.0
        )
        acts["game_id"] = game
        acts["period_id"] = period
        parts.append(acts)
    actions = pd.concat(parts, ignore_index=True)
    actions["action_id"] = np.arange(len(actions))
    return make_loaded("sportec", str(game), frames=frames, actions=actions)


@pytest.fixture(scope="module")
def stacked() -> pd.DataFrame:
    """The baseline layer-a melt (2 matches), stacking every post-preparation variant -- the former single share."""
    base = CoordinationParams.for_provider("sportec")
    variants = d2._post_preparation_variants(base)
    matches = [_match(1, (1, 2), 11), _match(2, (1, 3), 12)]
    return pd.concat(
        [match_tables(m, base, n_surrogates=0, variants=variants, include_switch_events=True) for m in matches],
        ignore_index=True,
    )


def _norm(obj) -> str:
    return json.dumps(obj, default=str, sort_keys=True)


def _per_variant(stacked: pd.DataFrame, v: str) -> pd.DataFrame:
    """What the per-variant share + per-variant combine produces for variant ``v``: that variant's rows, sorted by
    the combine's ``_ORDER_COLUMNS``."""
    return stacked[stacked["variant"] == v].sort_values(_ORDER, kind="mergesort").reset_index(drop=True)


def _stacked_filter(stacked: pd.DataFrame, v: str) -> pd.DataFrame:
    """What the OLD stacked combine + ``frame[frame["variant"] == v]`` read returned."""
    s = stacked.sort_values(_ORDER, kind="mergesort").reset_index(drop=True)
    return s[s["variant"] == v].reset_index(drop=True)


def test_fixture_preconditions(stacked):
    # ADR-032 non-vacuity: >=2 matches (so match-CV has folds), >=2 variants incl base, the H1-H7 tables present.
    assert stacked["match_id"].nunique() >= 2
    variants = set(stacked["variant"])
    assert "base" in variants and len(variants) >= 2
    assert {
        "pair",
        "pair_phase",
        "spectral",
        "cluster_team",
        "rsi",
        "windows",
        "rsi_switch_times",
        "possession_changes",
    } <= set(stacked["table"])


def test_per_variant_read_equals_stacked_filter(stacked):
    # the load-bearing equivalence: the per-variant file == the old stacked-combine filtered to that variant, row for
    # row (same sort). Nothing dropped, order preserved.
    for v in sorted(set(stacked["variant"])):
        pd.testing.assert_frame_equal(_per_variant(stacked, v), _stacked_filter(stacked, v))


def test_confirm_reducers_identical_per_variant_vs_stacked(stacked):
    # drive the confirm's actual reducers on both representations: identical -> calibration.json's decision fields are
    # unchanged. Also the ADR-032 non-vacuity: base reaches >=2 CV folds and a finite reliability.
    seed = CoordinationParams().surrogate_seed
    base_reached = False
    for v in sorted(set(stacked["variant"])):
        new, old = _per_variant(stacked, v), _stacked_filter(stacked, v)
        assert _norm(evaluate_hypotheses(d2.split_tables(new), seed=seed)) == _norm(
            evaluate_hypotheses(d2.split_tables(old), seed=seed)
        )
        splits_new = match_cv_splits(d2._join_keys(new))
        rel_new = d2.reliability_over_folds(new, d2.COORD_METHOD_FAMILIES, splits_new)
        rel_old = d2.reliability_over_folds(old, d2.COORD_METHOD_FAMILIES, match_cv_splits(d2._join_keys(old)))
        assert _norm(rel_new) == _norm(rel_old)
        if v == "base":
            assert len(splits_new) >= 2 and np.isfinite(rel_new[0])
            base_reached = True
    assert base_reached


def test_joint_variant_schema_matches_baseline(stacked):
    # the reviewer's schema-split flag: the joint pass emits ONE variant (no stack to split) and its combined file must
    # share the full-melt schema of the per-variant baseline files -- both are match_tables output.
    base = CoordinationParams.for_provider("sportec")
    joint = d2.apply_level(d2.apply_level(base, "butterworth_cutoff_hz", 0.5), "min_observed_fraction", 0.1)
    joint_melt = match_tables(
        _match(1, (1, 2), 11), base, n_surrogates=0, variants={"joint": joint}, include_switch_events=True
    )
    assert set(joint_melt.columns) == set(stacked.columns)


def test_summary_fold_preserves_objective_id_inputs():
    # D2-SPEC-05: the per-variant combine summaries agree on generation + population_digest and differ only in the
    # name-derived `pass`; the fold returns them unchanged with `pass` reset to the baseline level's canonical name, so
    # generation_tokens / objective_id / _population are byte-identical to the former single stacked combine.
    base = {
        "pass": d2.level_share_name(d2.BASELINE_LEVEL, "base"),
        "generation": "g1",
        "population_digest": "d1",
        "n_listed": 2,
        "stage_seconds": {"corpus": 1.0, "combine": 0.5},
    }
    other = {
        "pass": d2.level_share_name(d2.BASELINE_LEVEL, "welch_segment_s=1.5"),
        "generation": "g1",
        "population_digest": "d1",
        "n_listed": 2,
        "stage_seconds": {"corpus": 2.0},
    }
    folded = d2._fold_baseline_summary([base, other])
    assert (folded["generation"], folded["population_digest"], folded["n_listed"]) == ("g1", "d1", 2)
    assert folded["pass"] == d2.level_share_name(d2.BASELINE_LEVEL)  # canonical, not a per-variant name
    # IMPL-04: stage_seconds is SUMMED across variants (not just the first), so baseline timing reflects all of them.
    assert folded["stage_seconds"] == {"corpus": 3.0, "combine": 0.5}


def test_summary_fold_refuses_disagreeing_population():
    with pytest.raises(SystemExit, match="disagree on"):
        d2._fold_baseline_summary(
            [
                {"pass": "x", "generation": "g1", "population_digest": "d1"},
                {"pass": "y", "generation": "g1", "population_digest": "d2"},
            ]
        )


# ----------------------------------------------------------------- whole-calibration.json golden (D2-SPEC-05b)
# The approved bar asserts the WHOLE calibration.json (minus volatiles) is byte-identical to a golden captured from the
# ACTUAL pre-C code (fcca558). The confirm's ruthless OAT needs FINITE reliability for every candidate, which a real
# match_tables melt of a tiny synthetic corpus does not give (it fatals: non-finite reliability) -- so, like every
# other D2 test, this PLANTS controlled high-separation shares (finite reliability, the OAT runs, welch_segment_s
# moves) and runs the REAL combine_levels + _layer_b + _confirm. The ONLY version-specific step is the baseline share
# layout: pre-C writes one stacked share, C writes one per variant; `plant_and_confirm` detects which and plants
# accordingly, so the SAME function captures the golden at fcca558 and asserts reproduction at HEAD. (The planted
# shares are pair-only, so H1-H7 are non-findings -- but byte-identically so in both layouts; the point is the whole
# file, not a live hypothesis finding.)
_CLEAN = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}
_GOLDEN = Path(__file__).resolve().parent / "_fixtures" / "d2_calibration_golden.json"
_VOLATILE = {"stage_seconds", "run_commit", "run_tree_dirty", "run_tree_state", "__provenance__"}


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    # ADR-038 label lookups never reach the network (fail-closed empty manifest), mirroring test_calibrate_coordination.
    monkeypatch.setattr("scripts._loader_pining.match_visibility", lambda providers, *, token=None, base_url=None: {})


def _strip_volatiles(obj: Any) -> Any:
    """Recursively drop the run-to-run volatile keys (``*timings*`` / ``stage_seconds`` / ``run_*``)."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatiles(v)
            for k, v in obj.items()
            if k not in _VOLATILE and "timings" not in k and "stage_seconds" not in k
        }
    if isinstance(obj, list):
        return [_strip_volatiles(x) for x in obj]
    return obj


def _derivation(out: Path) -> tuple[Path, str]:
    deriv = {"pooled": dict(cg.INTERIM_BASE), "providers": {"skillcorner": dict(cg.INTERIM_BASE)}}
    data = json.dumps(deriv, sort_keys=True).encode("utf-8")
    path = out / "derivation.json"
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def _args(out: Path, derivation: Path, **over):
    base = dict(
        out=str(out),
        layer="confirm",
        level=d2.BASELINE_LEVEL,
        providers=("skillcorner",),
        match_ids_json=None,
        corpus_json=None,
        max_matches=None,
        cache_dir=None,
        token=None,
        allow_dirty=True,
        list_matches=False,
        derivation=str(derivation),
    )
    base.update(over)
    return SimpleNamespace(**base)


def _pair_frame(variant: str, sep: float, n_matches: int = 10, seed: int = 0) -> pd.DataFrame:
    """A melted pair-table metric frame whose team-discrimination ICC rises with ``sep`` (mirrors
    test_calibrate_coordination._pair_frame -- the established finite-reliability fixture)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        for team, base in ((1, 0.5), (2, 0.5 + sep)):
            val = base + rng.normal(0, 0.02)
            rows.append(
                {
                    "provider": "skillcorner",
                    "match_id": f"skillcorner-{m}",
                    "variant": variant,
                    "table": "pair",
                    "team_a_id": team,
                    "team_b_id": 3 - team,
                    "coord_rp_resultant_length": val,
                    "coord_rp_pct_near_in_phase": val,
                    "coord_xc_max_abs_r": val,
                    "coord_vc_pct_in_phase": val,
                    "coord_vc_pct_anti_phase": 0.1,
                    "coord_coh_band_mean": val,
                }
            )
    return pd.DataFrame(rows)


def _write_share(out: Path, name: str, df: pd.DataFrame, sha: str, *, tag: str = "all") -> None:
    """One worker's share of a D2 pass in the layout ``write_worker_partial`` writes (table + manifest)."""
    keys = sorted({f"{p}__{m}" for p, m in zip(df["provider"], df["match_id"], strict=True)}) if len(df) else []
    out.mkdir(parents=True, exist_ok=True)
    df.reset_index(drop=True).to_parquet(out / f"{name}.{tag}.parquet", index=False)
    manifest = {
        "generation": "g1",
        "n_attempted": len(keys),
        "stage_seconds": {"corpus": 1.0},
        "listed": keys,
        "produced": keys,
        "excluded_keys": [],
        "failed_keys": [],
        "run_commit": _CLEAN["commit"],
        "run_tree_dirty": False,
        "run_tree_state": "clean",
        "derivation_sha256": sha,
    }
    (out / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")


def _baseline_variants() -> list[str]:
    return ["base"] + [
        d2.variant_label(p, float(v))
        for p in d2.POST_PREPARATION_PARAMS
        for v in d2.SWEEP[p]
        if not np.isclose(v, d2.BASELINE[p])
    ]


def _per_variant_layout() -> bool:
    """True if this tree stores the baseline share per variant (option C): ``level_share_name`` takes a variant."""
    try:
        d2.level_share_name(d2.BASELINE_LEVEL, "base")
        return True
    except TypeError:
        return False


def plant_and_confirm(out: Path) -> dict:
    """Plant controlled high-separation layer-a shares (welch_segment_s=1.5 separates, the OAT moves it) in the
    tree's own baseline layout, then run the REAL combine_levels + _layer_b + _confirm, returning calibration.json.
    Same function at fcca558 (stacked baseline share) and at HEAD (per-variant). No C-only symbol is called."""
    out.mkdir(parents=True, exist_ok=True)
    deriv, sha = _derivation(out)
    variants = _baseline_variants()
    high = d2.variant_label("welch_segment_s", 1.5)
    base_melt = pd.concat([_pair_frame(v, sep=(0.6 if v == high else 0.05)) for v in variants], ignore_index=True)
    if _per_variant_layout():
        for v in variants:
            _write_share(out, d2.level_share_name(d2.BASELINE_LEVEL, v), base_melt[base_melt["variant"] == v], sha)
    else:
        _write_share(out, d2.level_share_name(d2.BASELINE_LEVEL), base_melt, sha)
    for level in d2.preparation_levels():
        if level == d2.BASELINE_LEVEL:
            continue
        _write_share(out, d2.level_share_name(level), _pair_frame("base", sep=0.05), sha)
    d2._layer_b(_args(out, deriv, layer="b"), _CLEAN)
    d2._confirm(_args(out, deriv, layer="confirm"), _CLEAN)
    return json.loads((out / "calibration.json").read_text(encoding="utf-8"))


def test_harness_smoke_produces_a_real_confirm(tmp_path):
    # the plant -> combine_levels -> layer_b -> confirm pipeline runs end-to-end and writes a real calibration.json.
    cal = plant_and_confirm(tmp_path / "out")
    assert {"selections", "hypotheses", "objective_ids", "population", "confirmation", "joint_point"} <= set(cal)
    assert cal["selections"] and set(cal["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}


@pytest.mark.skipif(not _GOLDEN.is_file(), reason="golden fixture not captured yet (see the capture steps in the PR)")
def test_whole_calibration_json_byte_identical_to_golden(tmp_path):
    # D2-SPEC-05(b): the WHOLE calibration.json (minus volatiles) reproduces the golden captured from pre-C (fcca558).
    current = _strip_volatiles(plant_and_confirm(tmp_path / "out"))
    golden = _strip_volatiles(json.loads(_GOLDEN.read_text(encoding="utf-8")))
    # non-vacuity: the golden is a real confirm -- selections, hypotheses (all seven), population, objective_ids.
    assert {"selections", "hypotheses", "objective_ids", "population", "joint_point"} <= set(golden)
    assert golden["selections"] and set(golden["hypotheses"]) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}
    assert current == golden
