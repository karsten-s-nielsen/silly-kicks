"""The shared AST seam's module walkers and call graph (spec 2026-11-08 tracking-batch, R12).

One parse of ``silly_kicks/`` and ``scripts/`` serves every gate that derives a population from code: the
``group_rows`` caller set (ADR-073) and the W9 link population.
"""

from __future__ import annotations

import ast

from tests._scale_guarded import group_rows_callers
from tests.scripts._script_population import call_graph, iter_package_modules, iter_script_modules

#: ``group_rows_callers()`` at 0552f58 (its package .py files equal BASE_SHA bb0c601), captured before it moved onto the
#: shared walkers. A later task that
#: adds a ``group_rows`` caller adds its name here (and to ``SCALE_GUARDED``) on purpose.
FROZEN_GROUP_ROWS_CALLERS = frozenset(
    {
        "scripts._loader_databricks.load_matches",
        "scripts.validate_territorial_defense._measure_match",
        "silly_kicks.causal._confounders._pressure_at_entry",
        "silly_kicks.causal.opportunities.build_opportunities",
        "silly_kicks.coordination._signals._build_coordination_signals",
        "silly_kicks.gkdv._probe.paired_vector_controls",
        "silly_kicks.match_outcome._compute.compute_match_outcome",
        "silly_kicks.restdefense._compute._score_samples",
        "silly_kicks.restdefense._counting.count_goalside_by_sample",
        "silly_kicks.restdefense._probe.paired_vector_controls",
        "silly_kicks.spadl._skillcorner_inference.infer_defensive_actions",
        "silly_kicks.team_metrics._buildup.compute_buildup_kpis",
        "silly_kicks.team_metrics._pressing.compute_pressing_kpis",
        "silly_kicks.team_metrics._progression.compute_progression_kpis",
        "silly_kicks.territorial_defense._arms.arm_a_threat_suppressed_batch",
        "silly_kicks.territorial_defense._compute._distinct_defenders",
        "silly_kicks.territorial_defense._compute._score_arm_a",
        "silly_kicks.territorial_defense._compute._score_arm_b",
        "silly_kicks.territorial_defense._compute.compute_territorial_defense",
        "silly_kicks.territorial_defense._engine.classify_arm_a_domain",
        "silly_kicks.tracking._ball_contact.detect_ball_contacts",
        "silly_kicks.tracking._elastic_sync._build_frame_lookups",
        "silly_kicks.tracking._elastic_sync._detect_candidate_frames",
        "silly_kicks.tracking._gk_identification.derive_goalkeepers",
        "silly_kicks.tracking._line_breaking.detect_line_breaking",
        "silly_kicks.tracking._link_identity.__init__",
        "silly_kicks.tracking._off_ball_runs._off_ball_runs_kernel",
        "silly_kicks.tracking._run_values.detect_off_ball_runs",
        "silly_kicks.tracking._shape_graph._team_frame_graphs",
        "silly_kicks.tracking.defensive_credit._line_break_signal.precompute_line_break_between_lines",
        "silly_kicks.tracking.defensive_credit._orchestration._credit_rows_one_game",
        "silly_kicks.tracking.pitch_control._cache.warm",
        "silly_kicks.tracking.pitch_control._dispatch.build_pitch_control_index",
        "silly_kicks.tracking.utils._count_candidates_within_tolerance",
        "silly_kicks.tracking.utils.link_actions_to_frames",
    }
)


def test_group_rows_callers_unchanged():
    assert group_rows_callers() == FROZEN_GROUP_ROWS_CALLERS


def test_module_walkers_key_by_dotted_name():
    pkg = iter_package_modules()
    assert "silly_kicks.tracking._shape_graph" in pkg
    assert "silly_kicks.tracking" in pkg  # an __init__.py is keyed by its import name
    scripts = iter_script_modules()
    assert "scripts._driver" in scripts and "scripts.hf_jobs.campaign" in scripts


def test_call_graph_sees_factory_closures():
    g = call_graph({"m": ast.parse("def f():\n    def inner():\n        g()\n    return inner\ndef g(): pass\n")})
    assert "m:g" in g.callees("m:f.<locals>.inner") and "m:f" in g.callers("m:g")


def test_call_graph_follows_dynamic_kernel_import():
    trees = {
        "k": ast.parse("def kern(): pass\n"),
        "m": ast.parse("def f():\n    ks = import_kernels('k')\n    return ks.kern()\n"),
    }
    assert "k:kern" in call_graph(trees).callees("m:f")


def test_call_graph_resolves_imports_and_reexports():
    trees = {
        "p": ast.parse("from p._impl import work\n"),
        "p._impl": ast.parse("LIMIT = 3\ndef work():\n    return _helper() + LIMIT\ndef _helper(): return 1\n"),
        "q": ast.parse(
            "from p import work\nimport p._impl as impl\n"
            "def a():\n    return work()\ndef b():\n    return impl._helper()\n"
        ),
    }
    trees_pkg = {"p"}
    g = call_graph(trees, packages=trees_pkg)
    assert "p._impl:work" in g.callees("q:a")
    assert "p._impl:_helper" in g.callees("q:b")
    assert "p._impl:_helper" in g.callees("p._impl:work")
    assert g.referenced_globals("p._impl:work") == {"p._impl:LIMIT"}


def test_call_graph_relative_import():
    trees = {
        "p": ast.parse(""),
        "p.a": ast.parse("from .b import h\ndef f():\n    return h()\n"),
        "p.b": ast.parse("def h(): pass\n"),
    }
    assert "p.b:h" in call_graph(trees, packages={"p"}).callees("p.a:f")


def test_call_graph_function_reference_is_an_edge():
    # A function passed as a value (an xfn list, a dispatch table, a callback) is reachable.
    g = call_graph({"m": ast.parse("def f(): pass\ndef g():\n    return [f]\n")})
    assert "m:f" in g.callees("m:g")


def test_call_graph_self_method():
    src = "class C:\n    def a(self):\n        return self.b()\n    def b(self): pass\n"
    assert "m:C.b" in call_graph({"m": ast.parse(src)}).callees("m:C.a")
