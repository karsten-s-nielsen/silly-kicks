"""Runtime cross-check of the W9 population (spec 2026-11-08 tracking-batch W9.6).

The static call graph cannot see every dispatch: xfns run through lists and closures. This module patches both linkers
and ``_resolve_action_frame_context`` with a recorder, runs the fixtures through every public population member, and
asserts that every observed caller maps to a population member. A caller is identified by ``(co_filename,
co_firstlineno)`` of the calling frame's code object -- never by a qualified name: Python 3.10 has no
``co_qualname``, and generic closure names collide. For a decorated function, ``co_firstlineno`` is the first
decorator's line, so the AST map keys each node by its first decorator line when it has one.
"""

from __future__ import annotations

import contextlib
import functools
import pathlib
import sys
from collections.abc import Iterator

import silly_kicks.tracking.utils as tracking_utils
from tests._prod_fixtures import make_single_game_inputs
from tests.invariants._link_population import LINKERS, graph, public_non_exempt, resolve_call
from tests.invariants.link_population_registry import LINK_POPULATION, LINKS_REQUIRED
from tests.scripts._script_population import REPO

_SCOPES = (REPO / "silly_kicks", REPO / "scripts")
_LINKER_NAMES = tuple(q.partition(":")[2] for q in LINKERS)


def _module_file(module: str) -> pathlib.Path:
    base = REPO / pathlib.Path(*module.split("."))
    return (base / "__init__.py") if (base / "__init__.py").is_file() else base.with_suffix(".py")


def node_keys(members: frozenset[str] | set[str]) -> dict[tuple[str, int], str]:
    """``(resolved file, first decorator line or def line) -> member`` for every function member."""
    g = graph()
    out: dict[tuple[str, int], str] = {}
    for member in members:
        node = g.definition(member)
        if node is None:
            continue  # a module-level xfn list: it has no code object of its own
        line = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        out[(str(_module_file(member.partition(":")[0]).resolve()), line)] = member
    return out


def _in_scope(filename: str, scopes: tuple[pathlib.Path, ...]) -> bool:
    path = pathlib.Path(filename).resolve()
    return any(path.is_relative_to(s) for s in scopes)


@contextlib.contextmanager
def recording(scopes: tuple[pathlib.Path, ...] = _SCOPES) -> Iterator[set[tuple[str, int]]]:
    """Patch every binding of the three link entry points in every loaded module; yield the observed callers
    (resolved file, ``co_firstlineno``) under ``silly_kicks/`` or ``scripts/``, the linkers' own calls excluded."""
    seen: set[tuple[str, int]] = set()
    originals = {name: getattr(tracking_utils, name) for name in _LINKER_NAMES}
    linker_codes = {fn.__code__ for fn in originals.values()}

    def wrap(fn):
        @functools.wraps(fn)
        def recorder(*args, **kwargs):
            caller = sys._getframe(1).f_code
            if caller not in linker_codes and _in_scope(caller.co_filename, scopes):
                seen.add((str(pathlib.Path(caller.co_filename).resolve()), caller.co_firstlineno))
            return fn(*args, **kwargs)

        return recorder

    wrappers = {name: wrap(fn) for name, fn in originals.items()}
    patched: list[tuple[object, str, object]] = []
    for module in list(sys.modules.values()):
        mod_file = getattr(module, "__file__", None)
        if not mod_file or not _in_scope(mod_file, scopes):
            continue
        for name, fn in originals.items():
            for attr, value in list(vars(module).items()):
                if value is fn:
                    patched.append((module, attr, value))
                    setattr(module, attr, wrappers[name])
    try:
        yield seen
    finally:
        for module, attr, value in patched:
            setattr(module, attr, value)


def unexplained(seen: set[tuple[str, int]], members: frozenset[str] | set[str]) -> set[tuple[str, int]]:
    return seen - set(node_keys(members))


def _run_all_members(actions, frames) -> None:
    pointers = tracking_utils.link_actions_to_frames(actions, frames, on_low_coverage="ignore")[0]
    for member in public_non_exempt():
        # a member without a links=None call shape gets the caller's pointers (LINKS_REQUIRED)
        resolve_call(member)(actions, frames, pointers if member in LINKS_REQUIRED else None)


def test_every_observed_caller_is_a_population_member():
    actions, frames = make_single_game_inputs()
    with recording() as seen:
        _run_all_members(actions, frames)
    assert seen, "non-vacuity: the recorder saw no link call"
    assert unexplained(seen, LINK_POPULATION) == set()


def test_recorder_catches_planted_xfn(tmp_path, monkeypatch):
    """A statically unreferenced xfn that calls the linker: the check must flag it. The probe module lives in
    ``tmp_path`` (never inside the package, which other tests parse), and the recorder scope includes it."""
    planted = tmp_path / "planted_link_xfn_probe.py"
    monkeypatch.syspath_prepend(str(tmp_path))
    planted.write_text(
        "from silly_kicks.tracking.utils import link_actions_to_frames\n\n\n"
        "def planted_xfn(actions, frames):\n"
        "    return link_actions_to_frames(actions, frames, on_low_coverage='ignore')[0]\n",
        encoding="utf-8",
    )
    try:
        import importlib

        probe = importlib.import_module("planted_link_xfn_probe")
        actions, frames = make_single_game_inputs()
        with recording(scopes=(*_SCOPES, tmp_path.resolve())) as seen:
            probe.planted_xfn(actions, frames)
        assert unexplained(seen, LINK_POPULATION) == {(str(planted.resolve()), 4)}
    finally:
        sys.modules.pop("planted_link_xfn_probe", None)
