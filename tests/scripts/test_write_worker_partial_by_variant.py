"""Direct test of ``write_worker_partial_by_variant`` (IMPL-01).

The per-variant share writer replaces the single stacked write in the D2 baseline layer-a pass (ADR-112 follow-up,
option C). Drives the ACTUAL function on a synthetic multi-variant per-match shard set -- the shape a baseline
layer-a pass produces -- and proves: one variant per share; the UNION of the per-variant shares == the stacked share
``write_worker_partial`` would have written (no row/col loss); each per-variant manifest carries the SAME completeness
keys as the stacked manifest (so ``combine_workers``' B-1 proof is unchanged); and a BOUNDED working set -- one
variant is concatenated at a time, never the stacked total (the plan Task 2 memory bound). Pure pandas, no ruthless.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scripts._coordination_corpus import (  # noqa: E402
    StageTimer,
    write_worker_partial,
    write_worker_partial_by_variant,
)
from scripts._driver import shard_path  # noqa: E402

_CLEAN = {"commit": "0" * 40, "dirty": False, "tree_state": "clean"}
_ORDER = ["provider", "match_id"]
_VARIANTS = ["base", "welch_segment_s=1.5", "vc_epsilon=0.5"]  # raw variant values (the `variant` column)
_ROWS_PER = 4
_ONE_VARIANT_ROWS = 2 * _ROWS_PER  # 2 matches x _ROWS_PER rows
_STACKED_ROWS = _ONE_VARIANT_ROWS * len(_VARIANTS)


def _nm(variant: str) -> str:
    """A filesystem-safe share name per variant (mirrors level_share_name's mangling)."""
    return "b__" + variant.replace("=", "__").replace(".", "p")


def _per_match(provider: str, match: str) -> pd.DataFrame:
    rows = [
        {"provider": provider, "match_id": match, "variant": v, "table": "pair", "value": float(i)}
        for v in _VARIANTS
        for i in range(_ROWS_PER)
    ]
    return pd.DataFrame(rows)


def _fake_res(shard_dir: Path):
    """A for_each-style result over 2 matches, with each match's per-variant shard written where the writer reads it."""
    keys = ["skillcorner__1", "skillcorner__2"]
    for key, match in zip(keys, ("1", "2"), strict=True):
        path = shard_path(shard_dir, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        _per_match("skillcorner", match).to_parquet(path, index=False)
    return SimpleNamespace(
        shard_dir=shard_dir,
        keys=list(keys),
        shard_keys=list(keys),
        exclusions=set(),
        failures=set(),
        manifest=lambda: {"generation": "g1", "n_attempted": len(keys)},
    )


def _read_share(dest: Path, name: str) -> pd.DataFrame:
    return pd.read_parquet(dest / f"{name}.all.parquet")


def _manifest(dest: Path, name: str) -> dict:
    return json.loads((dest / f"manifest_{name}.all.json").read_text(encoding="utf-8"))


def _write_by_variant(dest: Path, res) -> None:
    write_worker_partial_by_variant(dest, _nm, res, _CLEAN, StageTimer(), tag="all", variants=_VARIANTS)


def test_each_per_variant_share_holds_exactly_one_variant(tmp_path):
    _write_by_variant(tmp_path, _fake_res(tmp_path / "shards"))
    for v in _VARIANTS:
        df = _read_share(tmp_path, _nm(v))
        assert set(df["variant"]) == {v}
        assert len(df) == _ONE_VARIANT_ROWS


def test_union_of_per_variant_shares_equals_the_stacked_share(tmp_path):
    res = _fake_res(tmp_path / "shards")
    stacked = write_worker_partial(tmp_path, "stacked", res, _CLEAN, StageTimer(), tag="all")
    _write_by_variant(tmp_path, res)
    union = pd.concat([_read_share(tmp_path, _nm(v)) for v in _VARIANTS], ignore_index=True)
    sort_cols = [*_ORDER, "variant", "value"]
    a = union.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    b = stacked.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    assert len(a) == _STACKED_ROWS
    pd.testing.assert_frame_equal(a[sorted(a.columns)], b[sorted(b.columns)])


def test_per_variant_manifests_match_the_stacked_completeness(tmp_path):
    res = _fake_res(tmp_path / "shards")
    write_worker_partial(tmp_path, "stacked", res, _CLEAN, StageTimer(), tag="all")
    _write_by_variant(tmp_path, res)
    stacked_m = _manifest(tmp_path, "stacked")
    for v in _VARIANTS:
        m = _manifest(tmp_path, _nm(v))
        for key in ("listed", "produced", "excluded_keys", "failed_keys", "generation"):
            assert m[key] == stacked_m[key], (v, key)


def test_bounded_working_set_never_concatenates_all_variants(tmp_path, monkeypatch):
    # the memory bound (plan Task 2): each pd.concat the writer makes holds ONE variant's rows, never the stacked total.
    res = _fake_res(tmp_path / "shards")
    real_concat = pd.concat
    seen: list[int] = []

    def _spy(objs, *a, **k):
        objs = list(objs)
        seen.append(sum(len(o) for o in objs))
        return real_concat(objs, *a, **k)

    monkeypatch.setattr("scripts._coordination_corpus.pd.concat", _spy)
    _write_by_variant(tmp_path, res)
    assert seen, "the writer never concatenated anything"
    assert max(seen) <= _ONE_VARIANT_ROWS < _STACKED_ROWS
