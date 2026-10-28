"""Spec r4 D18 (c): the xShot / xCross tiers that read one dataset share one store and one extraction, so every reader
of the extract output keeps only the rows (or units) of the union of its tier's arms FIRST -- and then sees exactly
what the public-only dataset gave it. `prep` (both trainers) and the xcross `probe_comparison` cohort do this; both
declare `arms_json`, so two tiers keep separate results there.

The fixtures are the committed toy trees: the SHARED dataset is the whole toy tree; the PUBLIC-ONLY dataset is the same
tree with every non-public match removed (the same files, so the same extraction). The public arm is chosen so the
restriction is never vacuous: it leaves out the first two SkillCorner matches, which the comparison cohort
(`PROBE_KEEP = 2` SkillCorner matches in match order) would otherwise take.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

from scripts import _stages as S
from scripts._toy_corpora import TOY_BUILDERS, TOY_PARAMS
from scripts.hf_jobs import arm_policy
from tests.scripts._parity import artifact, run_serial
from tests.scripts._stage_harness import PROV, params_from_args

XSHOT, XCROSS = "scripts.train_xshot_occurrence", "scripts.train_xcross_attempt"
_DATASET = {XSHOT: "toyxs", XCROSS: "toyxc"}
#: The public matches of the fixture: SkillCorner 102-105 and Gradient Sports 106 (in-job arm keys).
_PUBLIC = [arm_policy._injob_key("skillcorner", "skillcorner", str(g)) for g in range(102, 106)] + [
    arm_policy._injob_key("gradientsports", "gradientsports", "106")
]
PUBLIC_ARMS = json.dumps({"arms": {"public": _PUBLIC}, "ship_order": ["public"]}, sort_keys=True)


def _args(module: str, arms_json: str) -> list[str]:
    args = list(TOY_PARAMS[module])
    args[args.index("--arms-json") + 1] = arms_json
    return args


def _public_only(data_root: pathlib.Path, dataset: str) -> None:
    """Remove every non-public match (shard, actions, home team) from a toy flat dataset."""
    root = data_root / dataset
    keep = set(_PUBLIC)
    home_path = root / "home_teams.json"
    home = json.loads(home_path.read_text(encoding="utf-8"))
    for shard in sorted(root.glob("shards/*/*.parquet")):
        provider, _, game = shard.stem.partition("__")
        if arm_policy._injob_key(provider, provider, game) not in keep:  # toy: source = stem provider
            shard.unlink()
            (root / "_actions" / shard.name).unlink(missing_ok=True)
            home.pop(game, None)
    home_path.write_text(json.dumps(home, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _world(module: str, tmp_path: pathlib.Path, *, public_only: bool) -> pathlib.Path:
    data = TOY_BUILDERS[module](tmp_path / ("public" if public_only else "shared"))
    if public_only:
        _public_only(data, _DATASET[module])
    return data


def _run(module: str, data: pathlib.Path, store: pathlib.Path, arms_json: str, *stages: str):
    import importlib

    pipeline = importlib.import_module(module).PIPELINE
    params = params_from_args(pipeline, _args(module, arms_json))
    ctx = S.make_context(pipeline, params, PROV, store=S.LocalStore(store), data_root=data)
    for name in ("extract", *stages):
        st = ctx.pipeline.stage(name)
        refs = st.ref_map(ctx) if isinstance(st, S.ItemStage) else {}
        for k in S.sorted_units(ctx, st):
            S.run_unit(ctx, st, k, worker="w0", ref=refs.get(k))
        S.reduce_stage(ctx, name)
    return ctx


@pytest.mark.slow  # slow: two extraction passes over the toy tree in-process (ADR-023)
@pytest.mark.parametrize("module", [XSHOT, XCROSS])
def test_prep_on_the_shared_corpus_with_the_public_arm_equals_prep_on_the_public_only_corpus(module, tmp_path):
    want = _run(module, _world(module, tmp_path, public_only=True), tmp_path / "s-pub", PUBLIC_ARMS, "prep")
    got = _run(module, _world(module, tmp_path, public_only=False), tmp_path / "s-shr", PUBLIC_ARMS, "prep")
    assert S._encode(got.upstream("prep")) == S._encode(want.upstream("prep"))  # byte-identical study inputs
    # The other side of the band: with EVERY arm, the shared corpus gives `prep` more matches.
    every_arm = TOY_PARAMS[module][TOY_PARAMS[module].index("--arms-json") + 1]
    every = _run(module, _world(module, tmp_path / "all", public_only=False), tmp_path / "s-all", every_arm, "prep")
    assert S._encode(every.upstream("prep")) != S._encode(want.upstream("prep"))


@pytest.mark.slow  # slow: two extraction passes over the toy tree in-process (ADR-023)
def test_the_public_tiers_comparison_cohort_equals_the_public_only_one(tmp_path):
    want = _run(XCROSS, _world(XCROSS, tmp_path, public_only=True), tmp_path / "s-pub", PUBLIC_ARMS, "probe_comparison")
    got = _run(XCROSS, _world(XCROSS, tmp_path, public_only=False), tmp_path / "s-shr", PUBLIC_ARMS, "probe_comparison")
    cohort = got.upstream("probe_comparison")
    assert isinstance(cohort, S.FileSet) and cohort.files  # non-vacuous: a cohort was written
    assert S._encode(cohort) == S._encode(want.upstream("probe_comparison"))
    meta = cohort.files["meta.json"]  # the cohort's provenance names its matches
    assert b"skillcorner-102" in meta and b"skillcorner-100" not in meta  # the first, non-public matches are skipped


@pytest.mark.slow  # slow: four full toy pipeline runs (ADR-023)
@pytest.mark.parametrize("module", [XSHOT, XCROSS])
def test_the_public_bundle_from_the_shared_dataset_equals_the_public_only_bundle(module, tmp_path):
    outs = {}
    for name, public_only in (("pub", True), ("shr", False)):
        data = _world(module, tmp_path / name, public_only=public_only)
        run_serial(
            module,
            tmp_path / name / "s",
            data,
            _args(module, PUBLIC_ARMS),
            tmp_path / name / "out",
            env=dict(os.environ),
        )
        outs[name] = artifact(tmp_path / name / "out")
    assert outs["pub"] and outs["shr"] == outs["pub"]


@pytest.mark.slow  # slow: one extraction pass over the toy tree in-process (ADR-023)
@pytest.mark.parametrize(("module", "stage"), [(XSHOT, "prep"), (XCROSS, "prep"), (XCROSS, "probe_comparison")])
def test_an_arm_key_that_names_no_unit_is_refused_before_any_row_is_cut(module, stage, tmp_path):
    """Fail closed (HFU R4-10, 2026-10-30): the real DFL staging names a shard by one provider (``idsse__<game>``)
    and its rows by another (``source_provider`` ``sportec``), so a key built from the stem named no unit and the
    row filter dropped that match silently. A reader of the shared extraction refuses such a key, by name."""
    from scripts.hf_jobs.arms import ArmKeyMismatchError

    stray = "idsse__idsse-103"  # the stem-built key: no staged unit has it
    arms = json.dumps({"arms": {"public": [*_PUBLIC[:1], stray, *_PUBLIC[2:]]}, "ship_order": ["public"]})
    ctx = _run(module, _world(module, tmp_path, public_only=False), tmp_path / "s", arms)
    with pytest.raises(ArmKeyMismatchError, match=stray):
        ctx.pipeline.stage(stage).work("only", ctx)
