"""The W9 link population, derived from the static call graph (spec 2026-11-08 tracking-batch §4.9 W9.6; ADR-056).

The population is every function that links actions to frames, directly or through a member:

- **seed:** every caller of ``link_actions_to_frames``, ``link_actions_to_frames_elastic``, or
  ``_resolve_action_frame_context``; every function with a ``links`` parameter; every ``*_xfns`` factory and the
  closures it defines; every function that ``lift_to_states`` wraps; minus the link machinery (the two linkers and
  the ``LINK_MACHINERY`` link-identity functions, which bind and remap ``links`` for a member);
- **fixpoint:** every caller of a member is a member, except that a public ``silly_kicks`` function and a
  ``scripts/`` entry point are members that are not expanded further (the stop rule).

The registry (``link_population_registry.py``) asserts the derived set exactly. The derivation reads the shared AST
seam (``tests/scripts/_script_population.py``): one parse, one call graph, no second walker (R12).
"""

from __future__ import annotations

import ast
import functools
import importlib
import inspect
import pathlib
import traceback
from collections.abc import Callable, Mapping
from typing import Literal, cast

import pandas as pd
import pytest

import silly_kicks
from tests.scripts._script_population import CallGraph, call_graph, iter_package_modules, iter_script_modules

#: The three link entry points (spec W9.6 seed).
LINKERS = (
    "silly_kicks.tracking.utils:link_actions_to_frames",
    "silly_kicks.tracking.utils:link_actions_to_frames_elastic",
    "silly_kicks.tracking.utils:_resolve_action_frame_context",
)
#: Link machinery beside the linkers (owner ruling 2026-11-11): the link-identity functions bind and remap ``links``.
#: They are not link consumers, so the seed subtracts them BY NAME, as it subtracts ``LINKERS[:2]``. Another
#: ``_link_identity`` function with a ``links`` parameter still enters the population. Their own tests cover them
#: (``tests/tracking/test_link_identity.py``).
LINK_MACHINERY = tuple(
    f"silly_kicks.tracking._link_identity:{name}"
    for name in (
        "resolve_link_positions",
        "remap_links",
        "_remap",
        "_verify_positions",
        "_legacy_positions",
        "require_links_bound",
    )
)
#: scripts/ entry points are leaves of the fixpoint (spec W9.6 stop rule).
SCRIPT_ENTRY_POINTS = frozenset({"main", "cli", "_main", "run"})
PACKAGE_ROOT = pathlib.Path(silly_kicks.__file__).resolve().parent


@functools.cache
def graph() -> CallGraph:
    """The call graph over ``silly_kicks/`` and ``scripts/`` (cached per session)."""
    return call_graph({**iter_package_modules(), **iter_script_modules()})


def is_public(qualname: str) -> bool:
    """Public ``silly_kicks`` API: a top-level function, class, or module-level name (an xfn list such as
    ``das_xfns``) without a leading underscore, or a public or dunder method of a public class. A closure is never
    public."""
    module, _, name = qualname.partition(":")
    parts = name.split(".")
    if not module.startswith("silly_kicks") or "<locals>" in parts or parts[0].startswith("_"):
        return False
    return all(not p.startswith("_") or (p.startswith("__") and p.endswith("__")) for p in parts[1:])


def is_script_entry(qualname: str) -> bool:
    module, _, name = qualname.partition(":")
    return module.startswith("scripts") and name in SCRIPT_ENTRY_POINTS


def _params(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    a = fn.args
    return [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]


def _lift_targets(g: CallGraph) -> set[str]:
    """Every function passed as the first argument of a ``lift_to_states(...)`` call."""
    out: set[str] = set()
    trees = {**iter_package_modules(), **iter_script_modules()}
    for module, tree in trees.items():
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Name)):
                continue
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name == "lift_to_states":
                target = g.resolve(module, node.args[0].id)
                if target is not None:
                    out.add(target)
    return out


def seed() -> frozenset[str]:
    """The W9.6 seed."""
    g = graph()
    fns = g.functions()
    out: set[str] = set()
    for linker in LINKERS:
        out |= {c for c in g.callers(linker) if c in fns}
    for q, fn in fns.items():
        name = q.partition(":")[2]
        if "links" in _params(fn):
            out.add(q)
        top = name.split(".")[0]
        if top.endswith("_xfns"):
            out.add(q)  # the factory and every closure it defines
    out |= {q for q in _lift_targets(g) if q in fns}
    return frozenset(out - set(LINKERS[:2]) - set(LINK_MACHINERY))


@functools.cache
def derive_population() -> frozenset[str]:
    """The W9.6 population: the seed, closed under callers, stopping at public API and scripts entry points."""
    g = graph()
    fns = g.functions()
    pop = set(seed())
    todo = list(pop)
    while todo:
        q = todo.pop()
        if is_public(q) or is_script_entry(q):
            continue
        for c in g.callers(q):
            # a caller is a function, or a public module-level name (a default xfn list), which is a leaf
            if c not in pop and (c in fns or (g.is_global(c) and is_public(c))):
                pop.add(c)
                todo.append(c)
    return frozenset(pop)


def derive_coverage() -> dict[str, str]:
    """private package member -> one public package member that reaches it (the lexically first)."""
    g = graph()
    pop = derive_population()
    public = sorted(q for q in pop if is_public(q))
    reach = {p: g.closure([p]) for p in public}
    out: dict[str, str] = {}
    for q in sorted(pop):
        if q.startswith("silly_kicks") and not is_public(q):
            hit = next((p for p in public if q in reach[p]), None)
            if hit is not None:
                out[q] = hit
    return out


def resolve_object(member: str) -> Callable:
    """The live object of a member ``module:name`` (a dotted name walks into a class: ``Class.method``)."""
    module, _, name = member.partition(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return cast(Callable, obj)


def accepts_links(member: str) -> bool:
    """True when the member's own call takes ``links`` (an adapter's member decides inside the adapter)."""
    obj = resolve_object(member)
    return callable(obj) and "links" in inspect.signature(obj).parameters


Route = Literal["battery", "generic", "adapter"]


def _registered_aggregators() -> frozenset[str]:
    from scripts._sb_battery import registered_add_star_aggregators

    return frozenset(registered_add_star_aggregators())


def route(member: str) -> Route:
    """``battery`` when the member is a registered ``add_*`` aggregator that ``silly_kicks.tracking`` exports as this
    very function; ``generic`` when it fits the ADR-078 call shape; else ``adapter``."""
    import silly_kicks.tracking as tracking
    from tests.invariants.link_population_registry import PREREQUISITE_ADAPTERS

    if member in PREREQUISITE_ADAPTERS:
        return "adapter"
    name = member.partition(":")[2]
    fn = resolve_object(member)
    if name in _registered_aggregators():
        exported = getattr(tracking, name, None)
        if exported is not None and (exported.__module__, exported.__qualname__) == (fn.__module__, fn.__qualname__):
            return "battery"
    if callable(fn) and not isinstance(fn, type) and _fits_adr078(fn):
        return "generic"
    return "adapter"


def _fits_adr078(fn: Callable) -> bool:
    """ADR-078: ``(actions, frames, *, ...)`` -- ``frames`` positional-or-keyword right after ``actions``, every
    further parameter keyword-only or defaulted."""
    params = list(inspect.signature(fn).parameters.values())
    if len(params) < 2 or [p.name for p in params[:2]] != ["actions", "frames"]:
        return False
    if params[1].kind not in (params[1].POSITIONAL_ONLY, params[1].POSITIONAL_OR_KEYWORD):
        return False
    # every further parameter is optional: a required keyword (a model, an xT surface) needs an adapter
    return all(p.default is not p.empty or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD) for p in params[2:])


def nonconforming_members() -> frozenset[str]:
    """Public population members that are neither battery aggregators nor ADR-078 shaped."""
    from tests.invariants.link_population_registry import EXEMPT_POOLED, PREREQUISITE_ADAPTERS

    skip = set(EXEMPT_POOLED) | set(PREREQUISITE_ADAPTERS)
    return frozenset(m for m in derive_population() if is_public(m) and m not in skip and route(m) == "adapter")


def classify_member_raise(frames: list[traceback.FrameSummary]) -> traceback.FrameSummary:
    """The deepest stack frame under ``silly_kicks/`` (a resolved-path test, A4-PLAN-71). With none, the raise is a
    scaffolding defect: ``pytest.fail`` (``Failed``, never ``AssertionError``)."""
    hits = [f for f in frames if pathlib.Path(f.filename).resolve().is_relative_to(PACKAGE_ROOT)]
    if not hits:
        pytest.fail(f"the raise never entered silly_kicks/: {[f.filename for f in frames]}")
    return hits[-1]


def split_by_game(
    actions: pd.DataFrame, frames: pd.DataFrame
) -> tuple[tuple[pd.DataFrame, pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]:
    """The two games of a two-game input, each as ``(actions, frames)`` with a fresh index, in first-appearance
    order of ``actions['game_id']``."""
    from silly_kicks.id_compat import canonical_id_series

    a_keys = canonical_id_series(actions["game_id"])
    f_keys = canonical_id_series(frames["game_id"])
    order = list(dict.fromkeys(a_keys.tolist()))
    if len(order) != 2:
        raise ValueError(f"split_by_game needs exactly two games, got {order}")
    parts = []
    for k in order:
        parts.append(
            (
                actions[(a_keys == k).to_numpy()].reset_index(drop=True),
                frames[(f_keys == k).to_numpy()].reset_index(drop=True),
            )
        )
    return parts[0], parts[1]


def _home_team_id(actions: pd.DataFrame):
    from tests._prod_fixtures import HOME_TEAM_ID

    return HOME_TEAM_ID  # one home team for both games and each part (A3-PLAN-68)


def resolve_call(
    member: str, *, member_frame: bool = False
) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame | None], object]:
    """A ready call ``(actions, frames, links) -> result``; it never calls the member. An atomic member
    (``silly_kicks.atomic.*``) receives ``convert_to_atomic(actions)``, its documented input. With ``member_frame``
    the caller passes the member's own frame (``member_actions``), and the call does not convert it again."""
    call = _resolve_call(member)
    if member_frame or not member.startswith("silly_kicks.atomic."):
        return call
    return lambda actions, frames, links: call(member_actions(member, actions), frames, links)


def member_actions(member: str, actions: pd.DataFrame) -> pd.DataFrame:
    """The actions a member reads: ``convert_to_atomic(actions)`` for an atomic member, else ``actions``. A caller
    that supplies ``links`` builds them on THIS frame (W9.5: links bind to exactly the actions passed)."""
    if not member.startswith("silly_kicks.atomic."):
        return actions
    from silly_kicks.atomic.spadl import convert_to_atomic
    from silly_kicks.id_compat import canonical_id_series

    # injected analytics are not atom-derivable; thread them through the conversion (the atomic contract)
    native = [c for c in ("shot_blocked", "cross_blocked") if c in actions.columns]
    # a converter call converts ONE game (atomic action_id is per game, chronological within it)
    keys = canonical_id_series(actions["game_id"])
    parts = [
        convert_to_atomic(actions[(keys == k).to_numpy()].reset_index(drop=True), preserve_native=native)
        for k in dict.fromkeys(keys.tolist())
    ]
    return pd.concat(parts, ignore_index=True)


def _resolve_call(member: str) -> Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame | None], object]:
    from scripts._sb_battery import call_aggregator
    from tests.invariants.link_population_registry import ADAPTERS

    kind = route(member)
    name = member.partition(":")[2]
    if kind == "battery":
        return lambda a, f, links: call_aggregator(name, a, f, links, _home_team_id(a))
    fn = resolve_object(member)
    if kind == "generic":
        takes_links = accepts_links(member)
        return lambda a, f, links: fn(a, f, links=links) if takes_links else fn(a, f)
    return ADAPTERS[member]


def call_member(member: str, actions: pd.DataFrame, frames: pd.DataFrame, *, links: pd.DataFrame | None = None):
    return resolve_call(member)(actions, frames, links)


Comparator = Literal["per_action", "per_key", "linker", "pooled"]


def comparator_for(member: str) -> Comparator:
    from tests.invariants.link_population_registry import EXEMPT_POOLED, PER_KEY

    if member in EXEMPT_POOLED:
        return "pooled"
    if member in LINKERS:
        return "linker"
    if member in PER_KEY:
        return "per_key"
    return "per_action"


def _assert_values_equal(whole, parts: list) -> None:
    import numpy as np

    from scripts._bitwise import assert_bits_equal

    if isinstance(whole, tuple):  # (table, report): the table half is per action; a report is a summary
        _assert_values_equal(whole[0], [p[0] for p in parts])
    elif isinstance(whole, dict):  # a per-key mapping: the whole equals the union of the parts
        merged: dict = {}
        for p in parts:
            merged.update(p)
        assert set(whole) == set(merged), (sorted(map(str, whole)), sorted(map(str, merged)))
        for k in whole:
            assert_bits_equal(whole[k], merged[k])
    elif isinstance(whole, np.ndarray):
        assert_bits_equal(whole, np.concatenate(parts))
    elif isinstance(whole, (pd.DataFrame, pd.Series)):
        # Each part against its own slice of the whole, in game order. Never a concat of the parts: pandas
        # turns an object column of pd.NA/None into nan when it concatenates all-NA entries, a difference the
        # member never made.
        assert len(whole) == sum(len(p) for p in parts), (len(whole), [len(p) for p in parts])
        lo = 0
        for p in parts:
            if isinstance(whole, pd.DataFrame):
                assert list(whole.columns) == list(p.columns), (list(whole.columns), list(p.columns))
            assert_bits_equal(whole.iloc[lo : lo + len(p)].reset_index(drop=True), p.reset_index(drop=True))
            lo += len(p)
    else:
        assert_bits_equal(whole, parts)


#: The pre-existing pointer columns (the base linker output, spec W9.2).
POINTER_COLUMNS = ["action_id", "frame_id", "time_offset_seconds", "n_candidate_frames", "link_quality_score"]


def _multiset(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[POINTER_COLUMNS].sort_values(POINTER_COLUMNS, kind="stable").reset_index(drop=True)


def _assert_linker_split(pointers: pd.DataFrame, parts: list[pd.DataFrame], actions: pd.DataFrame) -> None:
    """Pointer rows per game equal the per-game call. With ``action_row`` (W9.2), game b's rows are offset by game a's
    length; without it (the base output), the multiset over both games equals the two calls' multisets together."""
    from silly_kicks.id_compat import canonical_id_series

    if "action_row" not in pointers.columns:
        _assert_values_equal(_multiset(pointers), [_multiset(pd.concat(parts, ignore_index=True))])
        return
    keys = canonical_id_series(actions["game_id"])
    n_a = int((keys == keys.iloc[0]).sum())
    rows = pointers["action_row"].to_numpy()
    for k, (lo, hi) in enumerate(((0, n_a), (n_a, len(actions)))):
        sub = pointers[(rows >= lo) & (rows < hi)]
        _assert_values_equal(_multiset(sub), [_multiset(parts[k])])
        assert sorted((sub["action_row"] - lo).tolist()) == sorted(parts[k]["action_row"].tolist())


class GameSplitComparator:
    """One comparator per output type (spec W9.6). ``assert_split_equal`` raises only ``AssertionError``."""

    def __init__(self, kind: Comparator) -> None:
        self.kind = kind

    def assert_split_equal(self, whole, parts: list, actions: pd.DataFrame) -> None:
        if self.kind == "per_action":
            _assert_values_equal(whole, parts)
        elif self.kind == "per_key":
            from silly_kicks.id_compat import canonical_id_series

            frame = whole[0] if isinstance(whole, tuple) else whole
            keys = canonical_id_series(frame["game_id"]).to_numpy()
            games = list(dict.fromkeys(canonical_id_series(actions["game_id"]).tolist()))
            for game, part in zip(games, parts, strict=True):
                part_frame = part[0] if isinstance(part, tuple) else part
                _assert_values_equal(frame[keys == game].reset_index(drop=True), [part_frame])
        elif self.kind == "linker":
            _assert_linker_split(whole[0], [p[0] for p in parts], actions)
        else:
            raise AssertionError(f"{self.kind}: pooled outputs are exempt from the split property")


COMPARATORS: Mapping[str, GameSplitComparator] = {
    k: GameSplitComparator(k) for k in ("per_action", "per_key", "linker", "pooled")
}  # type: ignore[assignment]


def public_non_exempt() -> tuple[str, ...]:
    from tests.invariants.link_population_registry import EXEMPT_POOLED

    return tuple(sorted(m for m in derive_population() if is_public(m) and m not in EXEMPT_POOLED))
