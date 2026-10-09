"""Reduce-memory architecture guards (rev 3): the categorical combine stays categorical-safe + actually shrinks.

- Task 2b (anti-rot): EVERY ``.groupby`` / ``.pivot_table`` in a reduce module passes ``observed=`` -- under a
  categorical group key the ``observed=`` default flips (pd2 False -> empty groups / pd3 True) and diverges.
  ``observed=True`` is a no-op on object dtype, so the invariant is safe everywhere. A new un-``observed=`` groupby
  fails here. (merge / sort_values / set_index / astype(str) categorical-safety is covered dynamically by the D-5
  byte-identity harness, which drives the full ``build_report`` object-vs-sorted-categorical.)
- Task 5 (reduce-memory guard): ``combine_workers(categorical=True)`` returns ``category`` string columns and a
  combined table strictly smaller than the object path -- the OOM fix, proven to discriminate.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pandas as pd

_SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from scripts._coordination_corpus import (  # noqa: E402
    _ORDER_COLUMNS,
    StageTimer,
    combine_workers,
    write_worker_partial,
)
from scripts._driver import for_each  # noqa: E402


def _coordination_scripts() -> list[pathlib.Path]:
    # DERIVED population (ADR-056 anti-rot): every coordination script, globbed -- NOT a hand-kept tuple, so a
    # NEW coordination reduce module is auto-gated rather than silently escaping (re-review A RM-IMPL-04).
    return sorted(_SCRIPTS.glob("*coordination*.py"))


def _groupby_pivot_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"groupby", "pivot_table"}
        ):
            yield node


def _observed_is_true(node: ast.Call) -> bool:
    # assert the VALUE is True, not mere presence: an explicit observed=False must fail (re-review B RM-IMPL-03).
    for kw in node.keywords:
        if kw.arg == "observed":
            return isinstance(kw.value, ast.Constant) and kw.value.value is True
    return False


def test_every_coordination_groupby_pivot_sets_observed_true():
    mods = _coordination_scripts()
    assert len(mods) >= 5, f"derived coordination-script set looks wrong: {[p.name for p in mods]}"
    bad: list[str] = []
    for p in mods:
        for node in _groupby_pivot_calls(ast.parse(p.read_text(encoding="utf-8"))):
            if not _observed_is_true(node):
                bad.append(f"{p.name}:{node.lineno}")
    assert not bad, f"groupby/pivot_table without observed=True (categorical-unsafe under the rev-3 combine): {bad}"


def test_the_observed_gate_can_fail():
    # not vacuous: a MISSING observed= AND an explicit observed=False are both rejected; only =True passes.
    for src in ("x = df.groupby('a').sum()", "x = df.groupby('a', observed=False).sum()"):
        assert not _observed_is_true(next(_groupby_pivot_calls(ast.parse(src))))
    assert _observed_is_true(next(_groupby_pivot_calls(ast.parse("x = df.groupby('a', observed=True).sum()"))))


def _share(dest, tag):
    def work(ref):
        p, m = ref
        n = 300
        return pd.DataFrame(
            {"provider": [p] * n, "match_id": [m] * n, "table": ["pair"] * n, "level": ["dyad"] * n, "value": range(n)}
        )

    res = for_each(
        [("skillcorner", "m1"), ("skillcorner", "m2")],
        key=lambda r: r,
        work=work,
        shard_root=dest / "shards",
        token_inputs={"pass": "u"},
        tag=tag,
        label="match",
    )
    write_worker_partial(dest, "u", res, {"commit": "x", "dirty": False, "tree_state": "clean"}, StageTimer(), tag=tag)


def test_categorical_combine_shrinks_and_is_categorical(tmp_path):
    _share(tmp_path, "all")
    obj, _ = combine_workers(tmp_path, "u", expected=None)
    cat, _ = combine_workers(tmp_path, "u", expected=None, categorical=True)
    assert isinstance(cat["provider"].dtype, pd.CategoricalDtype)
    assert isinstance(cat["table"].dtype, pd.CategoricalDtype)
    obj_mem = int(obj.memory_usage(deep=True).sum())
    cat_mem = int(cat.memory_usage(deep=True).sum())
    assert cat_mem < obj_mem, f"categorical combine did not shrink: cat={cat_mem} obj={obj_mem}"
    pd.testing.assert_frame_equal(cat, obj, check_dtype=False, check_categorical=False)


# --- Task 5 non-groupby categorical-op safety (gold-standard hybrid: boundary invariant + coverage + load-bearing
# plant, RM-SPEC-11). The ORDER risk for the reduce's `sort_values`/`set_index` on a categorical key collapses to ONE
# invariant -- categorical columns carry SORTED unordered categories -- which makes THOSE ops order-match the object
# (lexicographic) path. We gate that invariant structurally HERE (universally-safe, precise, no per-call-site AST
# noise), prove the combine's sort keys are categorical so the D-5 byte-identity harness exercises the order-sensitive
# path non-vacuously, and plant the hazard to prove the invariant is load-bearing. The OTHER ops are NOT covered by
# the sort-order invariant: `merge` on categorical keys can reorder even with sorted categories (B TM-IMPL-06, pd
# 2.3.3), and `astype(str)`/`isin`/`map` are elementwise. All of those -- merge `_window_join:102`, astype(str)
# `_team_key:110`, isin/map -- are backstopped by D-5, which runs the ACTUAL op on the real reduce object-vs-sorted-
# categorical and asserts identical output (the merge feeds order-insensitive aggregates, so its reorder is benign and
# D-5 green confirms it). An AST gate can do none of this -- it can't see a column's runtime dtype.
def test_combine_output_categoricals_are_sorted_and_cover_the_sort_keys(tmp_path):
    _share(tmp_path, "all")
    cat, _ = combine_workers(tmp_path, "u", expected=None, categorical=True)
    # coverage: the combine's canonical sort keys (_ORDER_COLUMNS, the :464 sort_values) come back CATEGORICAL, so the
    # order-sensitive sort runs on the categorical dtype -- D-5's byte-identity genuinely exercises it (not vacuously).
    for c in _ORDER_COLUMNS:
        if c in cat.columns:
            assert isinstance(cat[c].dtype, pd.CategoricalDtype), f"combine sort key {c!r} is not categorical"
    # invariant: every categorical column is UNORDERED with SORTED categories -> `sort_values`/`set_index` order
    # matches the object/lexicographic path (append-order categories would diverge -- RM-R3-02 / ADR-112 D-1).
    # (`merge`/`isin`/`map`/`astype(str)` are NOT covered by this order invariant -- D-5-backstopped; TM-IMPL-06.)
    cat_cols = [c for c in cat.columns if isinstance(cat[c].dtype, pd.CategoricalDtype)]
    assert cat_cols, "combine returned no categorical columns"
    for c in cat_cols:
        cats = list(cat[c].cat.categories)
        assert cat[c].cat.ordered is False, f"{c!r} is an ORDERED categorical"
        assert cats == sorted(cats), f"{c!r} categories not sorted (would diverge sort order): {cats[:5]}"


def test_sorted_categories_are_load_bearing_for_sort_order():
    # the invariant above is not cosmetic: a categorical sort key orders by CATEGORY order, not value. The combine
    # pins CategoricalDtype(categories=sorted(union)); `union_categoricals`' default append-order would NOT. Plant it:
    # same values, two category orders -> sorted matches the object order, append-order DIVERGES.
    vals = ["skillcorner", "gradientsports", "idsse", "gradientsports", "skillcorner"]
    obj_order = pd.Series(vals).sort_values(kind="mergesort").tolist()
    sorted_cat = pd.Series(vals, dtype=pd.CategoricalDtype(sorted(set(vals)), ordered=False))
    append_cat = pd.Series(vals, dtype=pd.CategoricalDtype(["skillcorner", "gradientsports", "idsse"], ordered=False))
    assert sorted_cat.sort_values(kind="mergesort").tolist() == obj_order  # sorted categories == object order
    assert append_cat.sort_values(kind="mergesort").tolist() != obj_order  # append-order categories DIVERGE (hazard)
