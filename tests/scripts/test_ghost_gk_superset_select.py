"""Spec r4 D18 (a): ghost_gk extracts ONE superset per unit and each variant `select`s from it -- byte for byte the old
per-variant extract.

The superset `extract` runs `prepare_ghost_gk_training_data` once (faithful features, the widest variant grid) and
keeps every row BEFORE the keeper-detection filter, with the keeper visibility beside it. A per-variant `select` takes
over the old extract tail: the variant grid filter, the SkillCorner selection-bias sums, the keeper filter (fail-closed
on an all-null visibility), the `position_only` projection and `_pack`. The oracle is the `2474b73` `_extract_work`
kept verbatim (`tests/scripts/_ghost_gk_extract_reference.py`); `select` always reads the superset unit BACK FROM THE
STORE. Fixture units (A R4-SPEC-01, B R4S-03) beside the toy units: a game whose rows all leave the default grid, an
all-null keeper visibility, an unmapped unit, and a unit whose games lose every keeper frame.
"""

from __future__ import annotations

import json
from typing import cast

import pandas as pd
import pytest

from scripts import _stages as S
from scripts import _toy_corpora as TC
from scripts import train_ghost_gk as T
from scripts._driver import _merge_counters
from scripts._item_outcome import ItemExcluded
from scripts.hf_jobs.registry import REGISTRY
from tests.scripts import _ghost_gk_extract_reference as REF
from tests.scripts._stage_harness import PROV, params_from_args

_DATASET = "toygk"
#: Every committed variant's own build args (the campaign's), on the fixture dataset.
VARIANTS = {b.variant: list(b.run_args) for b in REGISTRY["ghost_gk"].builds}


def _stage_unit(root, home_teams: dict, provider: str, game: str, frames: pd.DataFrame, *, mapped: bool = True) -> None:
    """One extra staged flat unit (the `build_flat_corpus` layout); ``mapped=False`` leaves it out of the home map."""
    kind = TC.PROVIDER_ID_KIND[provider]
    stem = f"{provider}__{game}"
    TC.staged_frames(frames, kind).to_parquet(root / "shards" / TC.TOY_GENERATION / f"{stem}.parquet", index=False)
    n = int(frames["frame_id"].nunique())
    actions = TC.staged_actions(TC._game_actions(game, n_frames=n, home=1, away=2), kind)
    actions.to_parquet(root / "_actions" / f"{stem}.parquet", index=False)
    if mapped:
        home_teams[game] = TC.staged_home(1, kind)


def _keepers_beyond_the_default_grid(game: str) -> pd.DataFrame:
    """Both keepers about 40 m from their own goal: outside the default 30 m grid, inside the 52.5 m superset."""
    f = TC._game_frames(game, "gradientsports", n_frames=8, seed=1, dt=1.0)
    gk = f["is_goalkeeper"].to_numpy(dtype=bool) & ~f["is_ball"].to_numpy(dtype=bool)
    home = gk & (f["team_id"].astype("string") == "1").fillna(False).to_numpy()
    f.loc[home, "x"] = f.loc[home, "x"] + 35.0
    f.loc[gk & ~home, "x"] = f.loc[gk & ~home, "x"] - 35.0
    return f


def _skillcorner(game: str, visibility) -> pd.DataFrame:
    f = TC._game_frames(game, "skillcorner", n_frames=6, seed=1, dt=1.0, keeper_visibility=True)
    f["visibility"] = pd.array(visibility(f), dtype="boolean")
    return f


def _all_null(f: pd.DataFrame) -> list:
    """Every visibility flag null: a detection-aware provider whose flag was discarded (fail-closed)."""
    return [pd.NA] * len(f)


def _keepers_undetected(f: pd.DataFrame):
    """Every keeper row undetected, every other row detected: every keeper frame is lost to the filter."""
    return ~f["is_goalkeeper"].to_numpy(dtype=bool)


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    base = tmp_path_factory.mktemp("gkss") / "data"
    root = TC.build_ghost_corpus(base / _DATASET, n_games=4, n_frames=10, heavy_frames=None)
    home = json.loads((root / "home_teams.json").read_text(encoding="utf-8"))
    _stage_unit(root, home, "gradientsports", "300", _keepers_beyond_the_default_grid("300"))
    _stage_unit(root, home, "skillcorner", "301", _skillcorner("301", _all_null))
    _stage_unit(root, home, "skillcorner", "302", _skillcorner("302", _keepers_undetected))
    _stage_unit(
        root,
        home,
        "gradientsports",
        "303",
        TC._game_frames("303", "gradientsports", n_frames=4, seed=1, dt=1.0),
        mapped=False,
    )
    (root / "home_teams.json").write_text(json.dumps(home, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return base


def _ctx(data_root, store, variant: str, *extra: str):
    args = ["--dataset", _DATASET, *VARIANTS[variant], *extra]
    return S.make_context(
        T.PIPELINE, params_from_args(T.PIPELINE, args), PROV, store=S.LocalStore(store), data_root=data_root
    )


def _old(ref, ctx):
    """The oracle: the `2474b73` per-variant extract of one unit."""
    return REF._extract_work(T._extract_load(ref, ctx), ctx)


def _new(ref, ctx):
    """The change: the variant's `select` over the superset unit, read back from the store."""
    return T._select_work(T._select_load(ref, ctx), ctx)


def _outcome(fn, ref, ctx, bias: dict):
    """``fn(ref, ctx)``'s outcome as a comparable tuple: a frame (with the unit's bias counters), no result, an
    exclusion (reason, details) or an error (type, message)."""
    try:
        out = fn(ref, ctx)
    except ItemExcluded as exc:
        return ("excluded", exc.reason, dict(exc.details or {}))
    except Exception as exc:
        return ("error", type(exc).__name__, str(exc))
    if out is None:
        return ("none", dict(bias))
    return ("frame", out, dict(bias))


@pytest.fixture(scope="module")
def superset(data_root, tmp_path_factory):
    """ONE store whose superset `extract` ran over every unit (shared by every variant: one generation)."""
    store = tmp_path_factory.mktemp("gkss-store")
    ctx = _ctx(data_root, store, "default")
    st = ctx.pipeline.stage("extract")
    assert isinstance(st, S.ItemStage)
    refs = st.ref_map(ctx)
    outcomes = {k: S.run_unit(ctx, st, k, worker="w0", ref=refs[k]) for k in S.sorted_units(ctx, st)}
    return store, outcomes


def test_the_extract_generation_is_shared_and_the_select_generation_is_per_variant(data_root, tmp_path):
    tokens = {v: _ctx(data_root, tmp_path, v).tokens for v in VARIANTS}
    assert len({t["extract"] for t in tokens.values()}) == 1  # one extraction for all five variants
    sel = {v: t["select"] for v, t in tokens.items()}
    assert sel["default"] == sel["full"]  # same feature set and grid: one selection (E11)
    assert len({sel[v] for v in ("default", "position_only", "sweeper", "sweeper_position_only")}) == 4
    rows = {s["name"]: s for s in S.plan(_ctx(data_root, tmp_path, "default"))["stages"]}
    assert rows["select"]["counters"] and not rows["extract"]["counters"]  # the ONE counters-bearing stage


def test_a_variant_grid_outside_the_superset_grid_refuses_by_name(data_root, tmp_path):
    with pytest.raises(ValueError, match="superset"):
        _ctx(data_root, tmp_path, "sweeper", "--grid-x-max", "60")


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_select_reproduces_the_old_extract_for_every_unit(data_root, superset, variant):
    store, extracted = superset
    ctx = _ctx(data_root, store, variant)
    ext = ctx.pipeline.stage("extract")
    assert isinstance(ext, S.ItemStage)
    refs = ext.ref_map(ctx)
    kinds: dict[str, str] = {}
    for k in S.sorted_units(ctx, ctx.pipeline.stage("extract")):
        old = _outcome(_old, refs[k], ctx, REF._UNIT_BIAS)
        new = _outcome(_new, refs[k], ctx, T._UNIT_BIAS)
        kinds[k] = old[0]
        assert new[0] == old[0], (k, new[:1], old[:3])
        if old[0] == "frame":
            old_f = cast("tuple[str, pd.DataFrame, dict]", old)
            new_f = cast("tuple[str, pd.DataFrame, dict]", new)
            assert list(new_f[1].columns) == list(old_f[1].columns), k
            assert dict(new_f[1].dtypes) == dict(old_f[1].dtypes), k
            assert new_f[1].equals(old_f[1]), k
            assert S._encode(new_f[1]) == S._encode(old_f[1]), k  # the stored bytes
            assert new_f[2] == old_f[2], k  # the selection-bias counters
        else:
            assert new == old, k
    # Non-vacuity: every fixture class occurs, and the grid really moved rows between the two grids.
    assert kinds["gradientsports__303"] == "excluded" and extracted["gradientsports__303"] == "excluded"
    assert kinds["skillcorner__301"] == "error"  # all-null visibility: fail-closed, now in `select`
    assert extracted["skillcorner__301"] == "done"  # the superset keeps the rows; the keeper filter is `select`'s
    assert kinds["skillcorner__302"] == "none"  # every keeper frame lost
    wide = VARIANTS[variant].count("52.5") > 0
    assert kinds["gradientsports__300"] == ("frame" if wide else "none")
    assert sum(v == "frame" for v in kinds.values()) >= 3


def test_select_records_the_exclusion_extract_recorded(data_root, superset):
    store, _ = superset
    ctx = _ctx(data_root, store, "default")
    st = ctx.pipeline.stage("select")
    assert isinstance(st, S.ItemStage)
    key = "gradientsports__303"
    assert S.run_unit(ctx, st, key, worker="w0", ref=st.ref_map(ctx)[key]) == "excluded"
    old = json.loads(store.joinpath(ctx.generation("extract"), f"{key}.excluded.json").read_text(encoding="utf-8"))
    new = json.loads(store.joinpath(ctx.generation("select"), f"{key}.excluded.json").read_text(encoding="utf-8"))
    assert new == old == {"reason": "no home_team_id in mapping", "details": {"game_ids": ["303"]}}


@pytest.mark.slow  # slow: the toy ghost_gk corpus through two extraction paths in-process (ADR-023)
@pytest.mark.parametrize("variant", ["default", "sweeper_position_only"])
def test_the_cache_and_the_bias_counters_equal_the_old_path(tmp_path, variant):
    """The END of the change: `cache` (all that `cv`, `fit` and `importance` read) and the bias counters (all that
    `assemble` reads besides them) are byte-identical to the old path on the committed toy tree, so the bundle is."""
    data_root = TC.TOY_BUILDERS["scripts.train_ghost_gk"](tmp_path / "data")
    ctx = _ctx(data_root, tmp_path / "new", variant)
    for name in ("validate", "extract", "select", "cache"):
        st = ctx.pipeline.stage(name)
        refs = st.ref_map(ctx) if isinstance(st, S.ItemStage) else {}
        for k in S.sorted_units(ctx, st):
            assert S.run_unit(ctx, st, k, worker="w0", ref=refs.get(k)) in ("done", "excluded")
        S.reduce_stage(ctx, name)
    new_counters, _ = S.item_counters(ctx, "select")

    old_store = S.LocalStore(tmp_path / "old")
    st = ctx.pipeline.stage("extract")
    assert isinstance(st, S.ItemStage)
    refs = st.ref_map(ctx)
    parts, old_counters = [], {}
    for k in S.sorted_units(ctx, st):
        try:
            out = REF._extract_work(T._extract_load(refs[k], ctx), ctx)
        except ItemExcluded:
            continue
        S.write_result(old_store, f"old/{k}", out)
        parts.append(S.read_result(old_store, f"old/{k}"))  # the old cache read each unit back from the store
        _merge_counters(old_counters, dict(REF._UNIT_BIAS), k)
    old_cache = T._combine_packs([p for p in parts if len(p)])
    # `2474b73`'s `cache` was a stage of the same shape: its unit result was written, the reduce read it back and
    # wrote `_output`, and `cv` / `fit` read that. Replay those store round trips: on pandas 3 an object-string
    # column (the mixed-provider `_game_id`) reads back as `str`, so the in-memory frame is not what any reader saw.
    S.write_result(old_store, "old/cache/only", old_cache)
    S.write_result(old_store, "old/cache/_output", S.read_result(old_store, "old/cache/only"))
    stored = f"{ctx.generation('cache')}/_output.parquet"
    assert ctx.store.read_bytes(stored) == old_store.read_bytes("old/cache/_output.parquet")  # the stored bytes
    assert S._encode(ctx.upstream("cache")) == S._encode(S.read_result(old_store, "old/cache/_output"))
    assert new_counters == old_counters and new_counters.get("bias_depth_detected_n")  # non-vacuous bias sums
