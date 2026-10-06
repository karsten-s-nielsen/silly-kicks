"""TF-58 Task 22 (D2, §8.4): the gated Tier-C calibration driver.

All fixtures are synthetic layer-a shards written under ``tmp_path``; nothing touches the corpus. The ruthless
OAT/confirm grid, the reliability objective, the ADR-060 selection, the confirmation gate, the worker-share combine
(B-1), the artifact handoff (M-5) and the provenance wiring are exercised directly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import calibrate_coordination as d
from ruthless import GridSearchStrategy, InProcessBackend
from ruthless.errors import FatalEvaluationError
from ruthless.result import Candidate

import scripts._coordination_params_codegen as cg
from scripts._provenance import store_path_for

_CLEAN = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}
_DIRTY = {"commit": "0" * 40, "dirty": True, "tree_state": "dirty"}
_REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _private_manifest(monkeypatch):
    """ADR-038 label lookups never reach the network here: the stub manifest lists nothing, so every match is
    private (fail-closed)."""
    monkeypatch.setattr("scripts._loader_pining.match_visibility", lambda providers, *, token=None, base_url=None: {})


# --------------------------------------------------------------------------- constants (C26)
def test_levels_are_native_floats_and_include_baseline():
    assert d.levels_are_floats()
    for param, baseline in d.BASELINE.items():
        assert any(np.isclose(v, baseline) for v in d.SWEEP[param]), f"{param} sweep omits its baseline"


def test_level_counts_match_spec():
    counts = {p: len(v) for p, v in d.SWEEP.items()}
    assert counts == {
        "butterworth_cutoff_hz": 7,
        "max_detection_gap_s": 5,
        "min_observed_fraction": 6,
        "welch_segment_s": 5,
        "vc_epsilon": 5,
    }


def test_preparation_levels_are_the_eleven_layer_a_passes():
    # plan Task 22: 7 + 5 - 1 shared baseline = 11 layer-a passes; post-preparation levels ride the baseline pass.
    levels = d.preparation_levels()
    assert len(levels) == 11 and levels[0] == d.BASELINE_LEVEL
    assert all(d.canonical_level(level) == level for level in levels)


def test_layer_a_level_is_canonical_and_refuses_anything_but_a_preparation_level():
    assert d.canonical_level("butterworth_cutoff_hz=0.50") == "butterworth_cutoff_hz=0.5"
    with pytest.raises(SystemExit, match="variants of the baseline"):
        d.canonical_level("welch_segment_s=1.5")  # a post-preparation level is a variant, never its own pass
    with pytest.raises(SystemExit, match="not an off-baseline level"):
        d.canonical_level("butterworth_cutoff_hz=0.6")  # off the pre-registered grid
    with pytest.raises(SystemExit, match="not an off-baseline level"):
        d.canonical_level("butterworth_cutoff_hz=1.0")  # the baseline is the 'baseline' pass


class _ConstObjective:
    def __init__(self):
        self.calls = 0

    def evaluate(self, candidate: Candidate) -> dict[str, float]:
        self.calls += 1
        return {"reliability": 0.5, "reliability_se": 0.01, "fold_00": 0.5}


def test_oat_config_constructs_and_enumerates_1_plus_sum_levels_minus_1(tmp_path):
    cfg = d.oat_config(str(tmp_path / "oat.db"), "obj-1")
    obj = _ConstObjective()
    result = GridSearchStrategy(cfg).run(obj, backend=InProcessBackend())
    expected = 1 + sum(len(v) - 1 for v in d.SWEEP.values())  # baseline + each param's off-baseline levels
    assert expected == 24
    assert result.diagnostics["n_unique"] == expected


# --------------------------------------------------------------------------- reliability objective
def _pair_frame(provider, n_matches, variant, sep, seed=0):
    """A melted pair-table metric frame whose team-discrimination ICC rises with ``sep`` (team means split)."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_matches):
        for team, base in ((1, 0.5), (2, 0.5 + sep)):
            val = base + rng.normal(0, 0.02)
            rows.append(
                {
                    "provider": provider,
                    "match_id": f"{provider}-{m}",
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


def _write_level(out: Path, level_key: str, frames: list[pd.DataFrame]):
    """The COMBINED per-level table(s) a reader reads. The baseline is stored PER VARIANT (ADR-112 follow-up, option
    C): one combined file per variant present in the frames; a non-baseline level is a single file."""
    df = pd.concat(frames, ignore_index=True)
    if level_key == d.BASELINE_LEVEL:
        for variant in sorted(df["variant"].unique()):
            path = d.level_combined_path(out, d.BASELINE_LEVEL, variant)
            path.parent.mkdir(parents=True, exist_ok=True)
            df[df["variant"] == variant].to_parquet(path, index=False)
    else:
        path = d.level_combined_path(out, level_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)


def _candidate(param, value):
    params = dict(d.BASELINE)
    if param is not None:
        params[param] = float(value)
    return Candidate(id="c", params=params, program=None)


def test_objective_reads_the_deviating_level(tmp_path):
    # baseline level: low team separation; the cutoff=0.5 preparation level: high separation -> higher ICC.
    _write_level(tmp_path, d.BASELINE_LEVEL, [_pair_frame("skillcorner", 10, "base", sep=0.02)])
    _write_level(tmp_path, "butterworth_cutoff_hz=0.5", [_pair_frame("skillcorner", 10, "base", sep=0.6)])
    obj = d.CoordinationReliabilityObjective(tmp_path)
    base_rel = obj.evaluate(_candidate(None, 0.0))["reliability"]
    dev_rel = obj.evaluate(_candidate("butterworth_cutoff_hz", 0.5))["reliability"]
    assert np.isfinite(base_rel) and np.isfinite(dev_rel)
    assert dev_rel > base_rel  # the deviating level's shards -- higher separation -> higher ICC


def test_objective_reads_a_post_preparation_variant(tmp_path):
    # a post-preparation level is a variant inside the BASELINE file, not its own file.
    base = _pair_frame("skillcorner", 10, "base", sep=0.02)
    wide = _pair_frame("skillcorner", 10, d.variant_label("welch_segment_s", 1.5), sep=0.6)
    _write_level(tmp_path, d.BASELINE_LEVEL, [base, wide])
    obj = d.CoordinationReliabilityObjective(tmp_path)
    assert (
        obj.evaluate(_candidate("welch_segment_s", 1.5))["reliability"]
        > obj.evaluate(_candidate(None, 0.0))["reliability"]
    )


def test_missing_shard_raises_fatal(tmp_path):
    _write_level(tmp_path, d.BASELINE_LEVEL, [_pair_frame("skillcorner", 10, "base", sep=0.3)])
    obj = d.CoordinationReliabilityObjective(tmp_path)
    with pytest.raises(FatalEvaluationError, match="layer-a shards missing"):
        obj.evaluate(_candidate("butterworth_cutoff_hz", 0.5))  # no file for this preparation level


# --------------------------------------------------------------------------- selection (ADR-060)
def _history(param, moved_gain):
    base = SimpleNamespace(
        candidate=SimpleNamespace(params=dict(d.BASELINE)),
        metrics={"reliability": 0.5, **{f"fold_{i:02d}": 0.5 for i in range(5)}},
    )
    hist = [base]
    for value in d.SWEEP[param]:
        if np.isclose(value, d.BASELINE[param]):
            continue
        params = dict(d.BASELINE)
        params[param] = float(value)
        gain = moved_gain if np.isclose(value, d.SWEEP[param][-1]) else 0.0
        hist.append(
            SimpleNamespace(
                candidate=SimpleNamespace(params=params),
                metrics={"reliability": 0.5 + gain, **{f"fold_{i:02d}": 0.5 + gain for i in range(5)}},
            )
        )
    return hist


def test_selection_moves_only_beyond_noise_floor():
    moved = d.select_per_parameter(_history("welch_segment_s", moved_gain=0.20))["welch_segment_s"]
    assert moved.moved is True  # a clear, tight gain clears the effect floor and the paired SE
    stuck = d.select_per_parameter(_history("welch_segment_s", moved_gain=0.0005))["welch_segment_s"]
    assert stuck.moved is False  # a gain below MIN_EFFECT_SIZE does not move


# --------------------------------------------------------------------------- confirmation gate
def test_confirmation_gate_requires_both_reliability_and_hypotheses():
    all_pass = {h: {"pass": True} for h in ("H1", "H2", "H3", "H4", "H5", "H7")}
    one_fail = {**all_pass, "H4": {"pass": False}}
    assert d._confirm_gate(True, all_pass) is True
    assert d._confirm_gate(True, one_fail) is False  # hypotheses fail
    assert d._confirm_gate(False, all_pass) is False  # not moved
    assert d._confirm_gate(False, one_fail) is False


# --------------------------------------------------------------------------- worker shares + artifacts (B-1, M-5)
def _derivation_file(tmp_path: Path, derivation: dict | None = None) -> tuple[Path, dict, str]:
    """D1's derivation.json as the DGX chain hands it to D2 (path, content, sha256 of its bytes)."""
    derivation = derivation or {"pooled": dict(cg.INTERIM_BASE), "providers": {"skillcorner": dict(cg.INTERIM_BASE)}}
    data = json.dumps(derivation, sort_keys=True).encode("utf-8")
    path = tmp_path / "derivation.json"
    path.write_bytes(data)
    return path, derivation, hashlib.sha256(data).hexdigest()


def _write_share(dest: Path, name: str, df: pd.DataFrame, *, tag="all", generation="g1", **extra):
    """One worker's share of a D2 pass, in the layout ``write_worker_partial`` writes (table + manifest)."""
    keys = sorted({f"{p}__{m}" for p, m in zip(df["provider"], df["match_id"], strict=True)}) if len(df) else []
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / f"{name}.{tag}.parquet", index=False)
    manifest = {
        "generation": generation,
        "n_attempted": len(keys),
        "stage_seconds": {"corpus": 1.0},
        "listed": keys,
        "produced": keys,
        "excluded_keys": [],
        "failed_keys": [],
        "run_commit": _CLEAN["commit"],
        "run_tree_dirty": False,
        "run_tree_state": "clean",
        **extra,
    }
    (dest / f"manifest_{name}.{tag}.json").write_text(json.dumps(manifest), encoding="utf-8")


def _level_frame(level: str, high_variant: str | None, n_matches: int = 10) -> pd.DataFrame:
    """One layer-a level's metric table: the baseline carries every post-preparation variant (spec 8.4)."""
    if level != d.BASELINE_LEVEL:
        return _pair_frame("skillcorner", n_matches, "base", sep=(0.6 if level == high_variant else 0.05))
    variants = ["base"] + [
        d.variant_label(p, float(v))
        for p in d.POST_PREPARATION_PARAMS
        for v in d.SWEEP[p]
        if not np.isclose(v, d.BASELINE[p])
    ]
    return pd.concat(
        [_pair_frame("skillcorner", n_matches, var, sep=(0.6 if var == high_variant else 0.05)) for var in variants],
        ignore_index=True,
    )


def _plant_share(out: Path, level: str, frame: pd.DataFrame, *, tag: str = "all", **extra):
    """Write a level's worker share(s) in the production layout: the baseline splits its stacked variants into
    PER-VARIANT shares (ADR-112 follow-up, option C); a non-baseline level is a single share."""
    if level == d.BASELINE_LEVEL:
        for variant in sorted(frame["variant"].unique()):
            _write_share(
                out, d.level_share_name(d.BASELINE_LEVEL, variant), frame[frame["variant"] == variant], tag=tag, **extra
            )
    else:
        _write_share(out, d.level_share_name(level), frame, tag=tag, **extra)


def _plant_all_levels(out: Path, sha: str, high_variant: str | None = None):
    """Plant one worker's share of every layer-a level the OAT enumerates (baseline + the preparation levels)."""
    for level in d.preparation_levels():
        _plant_share(out, level, _level_frame(level, high_variant), derivation_sha256=sha)


def _args(out, derivation=None, **over):
    base = dict(
        out=str(out),
        layer="confirm",
        level=d.BASELINE_LEVEL,
        providers=("skillcorner",),
        match_ids_json=None,
        corpus_json=None,
        max_matches=None,
        cache_dir=None,
        token=None,
        allow_dirty=True,
        list_matches=False,
        derivation=None if derivation is None else str(derivation),
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_layer_b_combines_every_worker_share_of_every_level(tmp_path):
    out = tmp_path / "out"
    path, _derivation, sha = _derivation_file(tmp_path)
    corpus = tmp_path / "corpus.json"
    corpus.write_text(json.dumps({"skillcorner": [f"skillcorner-{m}" for m in range(10)]}), encoding="utf-8")
    for level in d.preparation_levels():
        frame = _level_frame(level, None)
        first = frame["match_id"].isin([f"skillcorner-{m}" for m in range(5)])
        _plant_share(out, level, frame[first], tag="w0", derivation_sha256=sha)
        _plant_share(out, level, frame[~first], tag="w1", derivation_sha256=sha)
    args = _args(out, path, corpus_json=str(corpus), match_ids_json=str(tmp_path / "w0.json"))
    d._layer_b(args, _CLEAN)
    oat = json.loads((out / "oat.json").read_text(encoding="utf-8"))
    assert oat["derivation_sha256"] == sha and set(oat["joint_point"]) == set(d.BASELINE)
    assert all(p["n_workers"] == 2 and p["n_listed"] == 10 for p in oat["population"].values())
    assert len(pd.read_parquet(d.level_combined_path(out, d.BASELINE_LEVEL, "base"))["match_id"].unique()) == 10
    # B-1: a level one worker never ran is refused, never scored on half the corpus
    (out / f"manifest_{d.level_share_name('butterworth_cutoff_hz=0.5')}.w1.json").unlink()
    with pytest.raises(SystemExit, match="miss 5 corpus key"):
        d._layer_b(args, _CLEAN)


def test_layer_b_refuses_levels_computed_from_different_derivations(tmp_path):
    out = tmp_path / "out"
    _plant_all_levels(out, "a" * 64)
    stray = d.level_share_name("max_detection_gap_s=2.0")
    _write_share(out, stray, _level_frame("x", None), derivation_sha256="b" * 64)  # computed from another derivation
    with pytest.raises(SystemExit, match="different derivations"):
        d._layer_b(_args(out), _CLEAN)


# --------------------------------------------------------------------------- C27 objective identity + stores
def test_objective_id_changes_when_any_generation_changes():
    a = d.objective_id_for(["layer_a__baseline.parquet", "layer_a__x.parquet"])
    b = d.objective_id_for(["layer_a__baseline.parquet", "layer_a__y.parquet"])
    assert a != b and a.startswith("calibrate_coordination:")
    assert d.objective_id_for(["b", "a"]) == d.objective_id_for(["a", "b"])  # order-insensitive


def test_objective_id_and_contract_track_the_objective_DEFINITION(monkeypatch):
    # A-33: the C27 store id and the input_contract must move when the objective's DEFINITION changes -- the columns it
    # reads (_FAMILY_COLUMNS) or its arithmetic (the fold scheme / ICC weighting, carried by OBJECTIVE_VERSION) -- or an
    # objective edit silently resumes a store scored under the old definition (the generation tokens would not notice).
    toks = ["layer_a__baseline.parquet"]
    base_id = d.objective_id_for(toks)
    base_ic = d.input_contract()["digest"]
    assert base_id.startswith("calibrate_coordination:")

    monkeypatch.setattr(d, "OBJECTIVE_VERSION", d.OBJECTIVE_VERSION + "-edit")
    assert d.objective_id_for(toks) != base_id
    assert d.input_contract()["digest"] != base_ic
    monkeypatch.undo()

    edited = {**d._FAMILY_COLUMNS, "relative_phase": ("pair", "team_a_id", ("coord_rp_resultant_length",))}
    monkeypatch.setattr(d, "_FAMILY_COLUMNS", edited)
    assert d.objective_id_for(toks) != base_id  # an edit to which columns define reliability moves the id
    assert d.input_contract()["digest"] != base_ic


def test_oat_id_moves_when_one_level_is_regenerated_or_its_population_changes(tmp_path):
    # M-3: the id is built from the shard generations the combine ACTUALLY read (and the corpus they cover), so a
    # single regenerated level -- or the same generation grown by more matches -- can never resume stale scores.
    out = tmp_path / "out"
    _plant_all_levels(out, "a" * 64)
    summaries, _sha = d.combine_levels(_args(out))
    first = d.objective_id_for(d.generation_tokens(summaries))
    assert d.objective_id_for(d.generation_tokens(d.combine_levels(_args(out))[0])) == first  # stable on a rerun
    level = d.level_share_name("butterworth_cutoff_hz=0.5")
    _write_share(out, level, _level_frame("x", None), generation="g2", derivation_sha256="a" * 64)
    regenerated = d.objective_id_for(d.generation_tokens(d.combine_levels(_args(out))[0]))
    _write_share(out, level, _level_frame("x", None, n_matches=11), generation="g2", derivation_sha256="a" * 64)
    grown = d.objective_id_for(d.generation_tokens(d.combine_levels(_args(out))[0]))
    assert len({first, regenerated, grown}) == 3


def test_dirty_tree_oat_id_never_resumes(tmp_path):
    # C4 (the one D21 identity rule): a dirty-tree id carries a per-call nonce and opens a FRESH store.
    clean = d.objective_id_for(["t"], prov=_CLEAN)
    one, two = d.objective_id_for(["t"], prov=_DIRTY), d.objective_id_for(["t"], prov=_DIRTY)
    assert clean == d.objective_id_for(["t"]) and one != two and one.startswith(clean)
    assert store_path_for(tmp_path / "grid_oat.db", clean) == str(tmp_path / "grid_oat.db")
    assert store_path_for(tmp_path / "grid_oat.db", one) != store_path_for(tmp_path / "grid_oat.db", two)


def test_oat_and_confirm_use_different_store_paths(tmp_path):
    store = d.oat_config(str(tmp_path / "grid_oat.db"), "id").store
    assert store is not None
    oat = store.path
    # _joint_point_score builds a grid_confirm.db store; assert the two paths differ.
    assert oat.endswith("grid_oat.db")
    assert str(tmp_path / "grid_confirm.db") != oat


def test_resume_from_store_skips_evaluated_points(tmp_path):
    cfg = d.oat_config(str(tmp_path / "resume.db"), "same-id")
    GridSearchStrategy(cfg).run(_ConstObjective(), backend=InProcessBackend())
    second = GridSearchStrategy(cfg).run(_ConstObjective(), backend=InProcessBackend())
    assert second.diagnostics["n_from_store"] == second.diagnostics["n_unique"]  # everything replayed from the store


# --------------------------------------------------------------------------- confirm: artifact + fallback
def test_calibration_artifact_fields_and_fallback(tmp_path):
    out = tmp_path / "out"
    path, derivation, sha = _derivation_file(tmp_path)
    _plant_all_levels(out, sha, high_variant=None)  # no level clearly beats baseline -> no move -> fallback
    committed = (_REPO / cg.GENERATED_PATH).read_bytes()

    d._confirm(_args(out, path), _CLEAN)

    art = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert set(art) >= {
        "selections",
        "joint_point",
        "confirmation",
        "hypotheses",
        "objective_ids",
        "moved_multipliers",
        "fallback_reason",
        "input_contract",
        "run_commit",
        "derivation_sha256",
        "population",
        "stage_seconds",
        "corpus_visibility",
    }
    assert art["corpus_visibility"] == "sc_extended"  # ADR-038: SkillCorner only, nothing public
    assert art["confirmation"]["gate_cleared"] is False and art["fallback_reason"]  # Tier-B stands, reason recorded
    assert art["derivation_sha256"] == sha and set(art["population"]) == set(d.preparation_levels())
    # the final module lands in --out: on a fallback it is exactly D1's module (the Tier-B values stand) ...
    assert (out / d.GENERATED_ARTIFACT).read_text(encoding="utf-8") == cg.render_generated_params(derivation, None)
    # ... and the package module is never touched (M-5: the DGX chain stays on the clean commit-1 tree).
    assert (_REPO / cg.GENERATED_PATH).read_bytes() == committed


def test_confirm_regenerates_params_when_gate_clears(tmp_path, monkeypatch):
    out = tmp_path / "out"
    path, derivation, sha = _derivation_file(tmp_path)
    _plant_all_levels(out, sha, high_variant=d.variant_label("welch_segment_s", 1.5))  # welch=1.5 clearly wins
    # unit-isolate the reliability leg of the confirm gate from the hypothesis leg
    monkeypatch.setattr(d, "gated_pass", lambda results: True)

    d._confirm(_args(out, path), _CLEAN)

    art = json.loads((out / "calibration.json").read_text(encoding="utf-8"))
    assert art["confirmation"]["gate_cleared"] is True
    assert art["moved_multipliers"] == {"welch_segment_s": {"multiplier": 1.5}}
    generated = (out / d.GENERATED_ARTIFACT).read_text(encoding="utf-8")
    assert generated == cg.render_generated_params(derivation, art["moved_multipliers"])
    assert generated != cg.render_generated_params(derivation, None)  # non-vacuity: the move reached the module


def test_confirm_refuses_a_derivation_other_than_the_one_the_levels_used(tmp_path):
    out = tmp_path / "out"
    path, _derivation, _sha = _derivation_file(tmp_path)
    _plant_all_levels(out, "e" * 64)  # the shares were computed from another derivation.json
    with pytest.raises(SystemExit, match="different derivation"):
        d._confirm(_args(out, path), _CLEAN)
    assert not (out / "calibration.json").exists()


def test_confirm_requires_the_derivation(tmp_path):
    out = tmp_path / "out"
    _plant_all_levels(out, "e" * 64)
    with pytest.raises(SystemExit, match="--derivation"):
        d._confirm(_args(out), _CLEAN)


def test_refuses_dirty_tree_without_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts._provenance.git_provenance",
        lambda: {"commit": "abc", "dirty": True, "tree_state": "dirty", "dirty_files": ["x.py"]},
    )
    monkeypatch.setattr("sys.argv", ["calibrate_coordination.py", "--layer", "confirm", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="DIRTY"):
        d.main()


def test_input_contract_declared_and_written():
    ic = d.input_contract()
    assert ic["driver"] == "calibrate_coordination" and len(ic["digest"]) == 64


# --------------------------------------------------------------------------- joint source (IMPL-04)
def test_joint_source_picks_the_right_precomputed_file_and_variant(tmp_path):
    base = _pair_frame("skillcorner", 4, "base", sep=0.1)
    welch = _pair_frame("skillcorner", 4, d.variant_label("welch_segment_s", 1.5), sep=0.6)
    _write_level(tmp_path, d.BASELINE_LEVEL, [base, welch])
    _write_level(tmp_path, "butterworth_cutoff_hz=0.5", [_pair_frame("skillcorner", 4, "base", sep=0.5)])

    f0, v0, s0 = d._joint_source(tmp_path, dict(d.BASELINE))  # nothing moved -> baseline base
    assert v0 == "base" and (f0["variant"] == "base").all() and s0 is None
    f1, v1, _ = d._joint_source(tmp_path, {**d.BASELINE, "welch_segment_s": 1.5})  # 1 post-prep -> baseline variant
    assert v1 == d.variant_label("welch_segment_s", 1.5) and (f1["variant"] == v1).all()
    f2, v2, _ = d._joint_source(tmp_path, {**d.BASELINE, "butterworth_cutoff_hz": 0.5})  # 1 prep -> its level file
    assert v2 == "base" and len(f2) == len(base) and (f2["variant"] == "base").all()


_MULTI = {**d.BASELINE, "butterworth_cutoff_hz": 0.5, "welch_segment_s": 1.5}  # prep + post-prep -> a joint pass


def _one_match_corpus(monkeypatch):
    from _fake_corpus import SpyLoader, make_loaded, make_ref

    # a detection-aware match carries its `visibility` flag (the corpus passes' Layer-2 pre-flight checks it)
    frames = pd.DataFrame({"x": [1.0], "is_ball": [False], "visibility": [True]})
    loaded = make_loaded("skillcorner", "m1", frames=frames, actions=None)
    loader = SpyLoader({("skillcorner", "m1"): loaded})
    monkeypatch.setattr(d, "corpus_source", lambda args: ([make_ref("skillcorner", "m1")], loader))


def _write_oat(out: Path, joint: dict, sha: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "oat.json").write_text(json.dumps({"joint_point": joint, "derivation_sha256": sha}), encoding="utf-8")


def test_layer_joint_runs_the_joint_prep_pass_and_confirm_combines_it(tmp_path, monkeypatch):
    out = tmp_path / "out"
    path, _derivation, sha = _derivation_file(tmp_path)
    _write_oat(out, _MULTI, sha)
    _one_match_corpus(monkeypatch)
    joint_frame = _pair_frame("skillcorner", 4, "joint", sep=0.6)
    monkeypatch.setattr(d, "match_tables", lambda loaded, params, **kw: joint_frame.copy())

    d._layer_joint(_args(out, path, layer="joint"), _CLEAN)
    frame, variant, summary = d._joint_source(out, _MULTI, args=_args(out, path))
    assert variant == "joint" and len(frame) == len(joint_frame) and (frame["variant"] == "joint").all()
    assert summary is not None and summary["consistent"]["joint"] == _MULTI
    assert d.level_combined_path(out, "joint").is_file()  # the joint-prep combined table was written


def test_layer_joint_is_a_no_op_when_the_joint_is_precomputed(tmp_path, monkeypatch):
    out = tmp_path / "out"
    path, _derivation, sha = _derivation_file(tmp_path)
    _write_oat(out, {**d.BASELINE, "welch_segment_s": 1.5}, sha)  # one move: precomputed in layer a
    _one_match_corpus(monkeypatch)
    monkeypatch.setattr(d, "match_tables", lambda *a, **kw: pytest.fail("no joint pass is needed"))
    assert d._layer_joint(_args(out, path, layer="joint"), _CLEAN) is None
    assert not list(out.glob("manifest_joint_prep.*.json"))


def test_joint_source_refuses_shares_computed_for_another_joint(tmp_path, monkeypatch):
    out = tmp_path / "out"
    path, _derivation, sha = _derivation_file(tmp_path)
    _write_oat(out, _MULTI, sha)
    _one_match_corpus(monkeypatch)
    monkeypatch.setattr(d, "match_tables", lambda *a, **kw: _pair_frame("skillcorner", 4, "joint", sep=0.6))
    d._layer_joint(_args(out, path, layer="joint"), _CLEAN)
    other = {**_MULTI, "welch_segment_s": 2.0}
    with pytest.raises(SystemExit, match="another joint"):
        d._joint_source(out, other, args=_args(out, path))


def test_layer_joint_refuses_a_derivation_other_than_the_oat_used(tmp_path, monkeypatch):
    out = tmp_path / "out"
    path, _derivation, _sha = _derivation_file(tmp_path)
    _write_oat(out, _MULTI, "f" * 64)
    _one_match_corpus(monkeypatch)
    with pytest.raises(SystemExit, match="different derivation"):
        d._layer_joint(_args(out, path, layer="joint"), _CLEAN)


def test_layer_joint_requires_the_oat_selection(tmp_path):
    path, _derivation, _sha = _derivation_file(tmp_path)
    with pytest.raises(SystemExit, match="--layer b first"):
        d._layer_joint(_args(tmp_path / "out", path, layer="joint"), _CLEAN)


# --------------------------------------------------------------------------- layer a: params + per-stage timing (R8)
def _timer_spy(seen: list, frame: pd.DataFrame):
    """A ``match_tables`` stand-in that records the params + ``timer`` it is handed and books one family stage."""

    def fake(loaded, params, *, timer=None, **kw):
        seen.append((params, timer))
        if timer is not None:
            with timer("family.cluster_phase"):
                pass
        return frame.copy()

    return fake


def test_layer_a_manifest_carries_the_per_stage_breakdown(tmp_path, monkeypatch):
    # R8 (plan L2770): layer a threads its StageTimer into match_tables, so the level's worker manifest carries the
    # per-stage breakdown beside the corpus wall clock.
    path, _derivation, _sha = _derivation_file(tmp_path)
    _one_match_corpus(monkeypatch)
    seen: list = []
    monkeypatch.setattr(d, "match_tables", _timer_spy(seen, _pair_frame("skillcorner", 4, "base", sep=0.1)))
    d._layer_a(_args(tmp_path, path, layer="a"), _CLEAN)
    assert seen and all(t is not None for _p, t in seen)
    name = d.level_share_name(d.BASELINE_LEVEL, "base")  # the baseline is written per variant (option C)
    manifest = json.loads((tmp_path / f"manifest_{name}.all.json").read_text(encoding="utf-8"))
    assert {"corpus", "family.cluster_phase"} <= set(manifest["stage_seconds"])


def test_layer_a_computes_with_the_derivation_artifact_and_tokens_its_digest(tmp_path, monkeypatch):
    # M-5 + M-4: layer a computes with the derivation.json it is handed (never the in-package module), and the
    # artifact's digest is in the shard token -- a different derivation is a different shard generation.
    pooled = {**dict(cg.INTERIM_BASE), "welch_segment_s": 120.0}
    path, derivation, sha = _derivation_file(tmp_path, {"pooled": pooled, "providers": {}})
    _one_match_corpus(monkeypatch)
    seen: list = []
    monkeypatch.setattr(d, "match_tables", _timer_spy(seen, _pair_frame("skillcorner", 4, "base", sep=0.1)))
    level = "butterworth_cutoff_hz=0.5"
    d._layer_a(_args(tmp_path / "one", path, layer="a", level=level), _CLEAN)
    params = seen[0][0]
    assert params.welch_segment_s == 120.0  # the artifact's value, not the committed module's
    expected = cg.params_from_artifacts("skillcorner", derivation, None)
    assert params.butterworth_cutoff_hz == pytest.approx(expected.butterworth_cutoff_hz * 0.5)  # the level applied
    name = d.level_share_name(level)
    first = json.loads((tmp_path / "one" / f"manifest_{name}.all.json").read_text(encoding="utf-8"))
    assert first["derivation_sha256"] == sha
    (tmp_path / "x").mkdir()
    other, _d2, _s2 = _derivation_file(tmp_path / "x")  # another derivation.json (the interim base)
    d._layer_a(_args(tmp_path / "one", other, layer="a", level=level), _CLEAN)
    second = json.loads((tmp_path / "one" / f"manifest_{name}.all.json").read_text(encoding="utf-8"))
    assert second["generation"] != first["generation"]  # the derivation digest is in the shard token


def test_layer_a_requires_the_derivation(tmp_path, monkeypatch):
    _one_match_corpus(monkeypatch)
    with pytest.raises(SystemExit, match="--derivation"):
        d._layer_a(_args(tmp_path, layer="a"), _CLEAN)


def test_joint_prep_pass_manifest_carries_the_per_stage_breakdown(tmp_path, monkeypatch):
    # R8: the joint preparation pass is a corpus pass like any other -- its manifest carries the corpus wall clock
    # and match_tables' per-stage breakdown, not an empty timer.
    out = tmp_path / "out"
    path, _derivation, sha = _derivation_file(tmp_path)
    _write_oat(out, _MULTI, sha)
    _one_match_corpus(monkeypatch)
    seen: list = []
    monkeypatch.setattr(d, "match_tables", _timer_spy(seen, _pair_frame("skillcorner", 4, "joint", sep=0.6)))
    d._layer_joint(_args(out, path, layer="joint"), _CLEAN)
    assert seen and all(t is not None for _p, t in seen)
    manifest = json.loads((out / "manifest_joint_prep.all.json").read_text(encoding="utf-8"))
    assert {"corpus", "family.cluster_phase"} <= set(manifest["stage_seconds"])


def test_fallback_reason_names_the_actual_cause():
    # A-56: the fallback reason must distinguish WHY Tier-B stands, not blanket "gate not cleared".
    passing = {"H1": {"pass": True}}
    failing = {"H1": {"pass": False}}
    assert d._fallback_reason(True, True, passing, gate=True) is None  # gate cleared
    assert "no OAT level improved" in (d._fallback_reason(False, False, failing, gate=False) or "")  # nothing moved
    assert "joint candidate did not beat" in (d._fallback_reason(True, False, passing, gate=False) or "")  # joint lost
    assert "hypotheses gate did not pass" in (
        d._fallback_reason(True, True, failing, gate=False) or ""
    )  # hypotheses failed
