"""Enforcement for the SK-EXPORT uniform metric-family output-contract registry.

Guards the ``silly_kicks.metric_contracts.METRIC_CONTRACTS`` surface five ways: per-entry
self-consistency, ``column_types`` coverage (non-None iff the package declares a typed ``dict``
full-columns mapping), round-trip to the package constants, completeness-by-enumeration (the
ADR-056 three-bucket shape, so a new metric family cannot silently escape), and output-faithfulness
(keys/metric are subsets of the package's REAL declared output -- the non-vacuous cross-check that a
plain round-trip cannot give, since it would pass against a wrong constant).
"""

from __future__ import annotations

import ast
import importlib
import pathlib

import pandas as pd

import silly_kicks
from silly_kicks.metric_contracts import METRIC_CONTRACTS, MetricContract

# family -> (module, KEYS attr, METRIC_COLUMNS attr, full-columns attr | None)
# The full-columns attr is the package's authoritative declared output constant; None where the
# package has no dedicated one (restdefense: full output is keys + metric).
_PKG: dict[str, tuple[str, str, str, str | None]] = {
    "team_metrics": ("silly_kicks.team_metrics", "TEAM_KPI_KEYS", "TEAM_KPI_METRIC_COLUMNS", "TEAM_KPI_COLUMNS"),
    "match_outcome": (
        "silly_kicks.match_outcome",
        "MATCH_OUTCOME_KEYS",
        "MATCH_OUTCOME_METRIC_COLUMNS",
        "MATCH_OUTCOME_COLUMNS",
    ),
    "shot_stopping": ("silly_kicks.shot_stopping", "SS_KEYS", "SHOT_STOPPING_METRIC_COLUMNS", "SHOT_STOPPING_COLUMNS"),
    "gk_decision": (
        "silly_kicks.gk_decision",
        "GK_DECISION_KEYS",
        "GK_DECISION_METRIC_COLUMNS",
        "GK_DECISION_SAMPLE_COLUMNS",
    ),
    "territory": ("silly_kicks.territory", "TERRITORY_KEYS", "TERRITORY_METRIC_COLUMNS", "TERRITORY_COLUMNS"),
    "duels": ("silly_kicks.duels", "DUEL_KEYS", "DUEL_METRIC_COLUMNS", "DUEL_COLUMNS"),
    "restdefense": ("silly_kicks.restdefense", "RD_SAMPLE_KEYS", "RD_METRIC_COLUMNS", None),
    "positioning": (
        "silly_kicks.positioning",
        "POSITIONING_KEYS",
        "POSITIONING_METRIC_COLUMNS",
        "POSITIONING_SAMPLE_COLUMNS",
    ),
    # TF-58 coordination: seven families exported from ONE package (one _PKG row per exported constant).
    "coordination_pair": (
        "silly_kicks.coordination",
        "COORDINATION_PAIR_KEYS",
        "COORDINATION_PAIR_METRIC_COLUMNS",
        "COORDINATION_PAIR_COLUMNS",
    ),
    "coordination_pair_phase": (
        "silly_kicks.coordination",
        "COORDINATION_PAIR_PHASE_KEYS",
        "COORDINATION_PAIR_PHASE_METRIC_COLUMNS",
        "COORDINATION_PAIR_PHASE_COLUMNS",
    ),
    "coordination_spectral": (
        "silly_kicks.coordination",
        "COORDINATION_SPECTRAL_KEYS",
        "COORDINATION_SPECTRAL_METRIC_COLUMNS",
        "COORDINATION_SPECTRAL_COLUMNS",
    ),
    "coordination_cluster_team": (
        "silly_kicks.coordination",
        "COORDINATION_CLUSTER_TEAM_KEYS",
        "COORDINATION_CLUSTER_TEAM_METRIC_COLUMNS",
        "COORDINATION_CLUSTER_TEAM_COLUMNS",
    ),
    "coordination_cluster_player": (
        "silly_kicks.coordination",
        "COORDINATION_CLUSTER_PLAYER_KEYS",
        "COORDINATION_CLUSTER_PLAYER_METRIC_COLUMNS",
        "COORDINATION_CLUSTER_PLAYER_COLUMNS",
    ),
    "coordination_team_sync": (
        "silly_kicks.coordination",
        "COORDINATION_TEAM_SYNC_KEYS",
        "COORDINATION_TEAM_SYNC_METRIC_COLUMNS",
        "COORDINATION_TEAM_SYNC_COLUMNS",
    ),
    "coordination_rsi": (
        "silly_kicks.coordination",
        "COORDINATION_RSI_KEYS",
        "COORDINATION_RSI_METRIC_COLUMNS",
        "COORDINATION_RSI_COLUMNS",
    ),
}

#: xsuccess is a VAEP rating method (TF-61) with no mart column-set -> deliberately not in the registry.
#: win_probability emits per-ACTION feature-grain columns (p_win/.../win_prob_leverage), NOT a
#: per-(entity, match) mart family -> exports WIN_PROBABILITY_COLUMNS (not *_METRIC_COLUMNS), so it is
#: out of the derived enrollment set; recorded here for the decision (TF-63/ADR-101).
_EXEMPT: dict[str, str] = {
    "xsuccess": "VAEP rating method; emits no mart column-set (TF-61/ADR-095)",
    "win_probability": "per-action feature-grain output; not a per-(entity,match) mart family (TF-63/ADR-101)",
}
#: nothing is un-derivable-and-un-enrolled.
_UNDERIVABLE: frozenset[str] = frozenset()


def _attr(family: str, which: int):
    """The named constant object (``which`` in {1, 2, 3}); the named attr must be non-None."""
    attr = _PKG[family][which]
    assert attr is not None, (family, which)
    return getattr(importlib.import_module(_PKG[family][0]), attr)


def _full_columns(family: str):
    """The package's full-columns constant, or ``None`` where it has none (restdefense)."""
    attr = _PKG[family][3]
    return getattr(importlib.import_module(_PKG[family][0]), attr) if attr is not None else None


def _pkg_all(init_path: pathlib.Path) -> list[str]:
    """The string literals in a package ``__init__``'s ``__all__`` (AST; no import)."""
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, ast.List):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _metric_constants_exported(root: pathlib.Path) -> set[tuple[str, str]]:
    """Every ``(package, constant)`` where ``constant`` is a ``*_METRIC_COLUMNS`` name in that package's
    ``__all__`` (ADR-098 amendment: completeness is keyed PER exported constant, so a package exporting several
    -- coordination's seven families -- is accounted for one family per constant, not once per package)."""
    out: set[tuple[str, str]] = set()
    for sub in root.iterdir():
        init = sub / "__init__.py"
        if sub.is_dir() and init.exists():
            out.update((sub.name, n) for n in _pkg_all(init) if n.endswith("_METRIC_COLUMNS"))
    return out


def _completeness_diff(registered: set[tuple[str, str]], derived: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return registered ^ derived


def _registry_gap(library_families, pkg_families) -> set[str]:
    """Families in the LIBRARY registry (``METRIC_CONTRACTS``) but not this test's ``_PKG`` table, or the reverse.

    ``_PKG`` is what the exported-constant derivation is checked against; this ties the library registry to it, so
    a family dropped from ``METRIC_CONTRACTS`` (ADR-098: "registers or the gate fails") cannot pass unnoticed."""
    return set(library_families) ^ set(pkg_families)


def test_public_metric_constants_importable():
    """Task 1 smoke: the four previously-private constants are now importable."""
    from silly_kicks.duels import DUEL_KEYS  # noqa: F401
    from silly_kicks.gk_decision import GK_DECISION_KEYS, GK_DECISION_METRIC_COLUMNS  # noqa: F401
    from silly_kicks.match_outcome import MATCH_OUTCOME_KEYS, MATCH_OUTCOME_METRIC_COLUMNS  # noqa: F401
    from silly_kicks.territory import TERRITORY_KEYS  # noqa: F401


def test_registry_self_consistency():
    for family, c in METRIC_CONTRACTS.items():
        assert set(c) == set(MetricContract.__annotations__), family
        assert c["metric_columns"], f"{family}: empty metric_columns"
        for field in ("keys", "metric_columns", "columns"):
            assert isinstance(c[field], tuple) and all(isinstance(x, str) for x in c[field]), (family, field)
        cols = set(c["columns"])
        assert set(c["metric_columns"]) <= cols, family
        assert set(c["keys"]) <= cols, family
        ct = c["column_types"]
        assert ct is None or (
            isinstance(ct, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in ct.items())
        )
        if ct is not None:
            assert set(ct) <= cols, family


def test_column_types_coverage():
    """SKEXP-PLAN-06: column_types is non-None IFF the pkg's full-columns constant is a typed dict."""
    for family, c in METRIC_CONTRACTS.items():
        full = _full_columns(family)  # the full-columns constant (or None for restdefense)
        expect_typed = isinstance(full, dict)
        assert (c["column_types"] is not None) == expect_typed, (
            f"{family}: column_types {'set' if c['column_types'] is not None else 'None'} but "
            f"full-columns is {type(full).__name__}"
        )
        if c["column_types"] is not None:
            assert isinstance(full, dict), family
            assert dict(c["column_types"]) == dict(full), family
            assert set(c["column_types"]) == set(c["columns"]), family


def test_round_trip_to_package_constants():
    for family, c in METRIC_CONTRACTS.items():
        assert c["keys"] == tuple(_attr(family, 1)), family
        assert c["metric_columns"] == tuple(_attr(family, 2)), family


def test_completeness_is_keyed_per_exported_constant_planted(tmp_path):
    """ADR-098 amendment: the derived population is keyed PER exported ``*_METRIC_COLUMNS`` constant, so a
    single package exporting several families is one enrollment per constant -- coordination's seven families
    from one package cannot collapse to a single entry the way a per-package derivation would."""
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text('__all__ = ["A_KEYS", "A_METRIC_COLUMNS", "B_METRIC_COLUMNS"]\n', encoding="utf-8")
    derived = _metric_constants_exported(tmp_path)
    assert derived == {("pkg", "A_METRIC_COLUMNS"), ("pkg", "B_METRIC_COLUMNS")}
    # a registry naming only ONE of the package's two constants is INCOMPLETE by exactly the other.
    assert _completeness_diff({("pkg", "A_METRIC_COLUMNS")}, derived) == {("pkg", "B_METRIC_COLUMNS")}


def test_library_registry_gap_detects_a_dropped_family_planted():
    """Planted violation: a library registry missing one family is reported by exactly that family."""
    dropped = next(iter(_PKG))
    library = {f: c for f, c in METRIC_CONTRACTS.items() if f != dropped}
    assert _registry_gap(library, _PKG) == {dropped}


def test_completeness_three_bucket():
    root = pathlib.Path(silly_kicks.__file__).parent
    assert not _registry_gap(METRIC_CONTRACTS, _PKG), _registry_gap(METRIC_CONTRACTS, _PKG)  # ADR-098 library leg
    derived = _metric_constants_exported(root)
    registered = {(module.rsplit(".", 1)[-1], metric_attr) for module, _k, metric_attr, _f in _PKG.values()}
    assert not _completeness_diff(registered, derived), _completeness_diff(registered, derived)
    assert len(registered) == len(_PKG)  # one family per exported constant; no two families share a key
    for name in _EXEMPT:
        init = root / name / "__init__.py"
        assert init.exists(), f"exempt package missing on disk: {name}"
        assert not any(n.endswith("_METRIC_COLUMNS") for n in _pkg_all(init)), (
            f"{name} now exports *_METRIC_COLUMNS -> it must be registered in METRIC_CONTRACTS, not exempt"
        )
    assert _UNDERIVABLE == frozenset()


def test_output_faithfulness():
    """keys/metric are subsets of the package's REAL declared output (not just the mirrored constant)."""
    for family, c in METRIC_CONTRACTS.items():
        full = _full_columns(family)
        if full is None:  # restdefense: no dedicated output constant -> declared output IS keys+metric
            declared = set(c["keys"]) | set(c["metric_columns"])
        else:
            declared = set(full)  # dict -> its keys; tuple -> its names
        assert set(c["keys"]) <= declared, f"{family}: keys not in declared output {declared - set(c['keys'])}"
        assert set(c["metric_columns"]) <= declared, family


def test_gk_decision_keys_are_the_real_summarize_grain():
    """gk_decision's grain lives only in summarize_gk_decision -> prove keys are its groupby output."""
    from silly_kicks.gk_decision import GK_DECISION_KEYS, GK_DECISION_SAMPLE_COLUMNS, summarize_gk_decision

    base = {
        "game_id": 100,
        "period_id": 1,
        "decision_id": 0,
        "keeper": 7,
        "keeper_raw": 7,
        "team_id": 55,
        "decision_value": 0.1,
        "chosen_ev": 0.5,
        "best_ev": 0.6,
        "sel_efficiency": 0.8,
        "decision_pct": 0.5,
        "n_options": 3,
        "option_set_source": "native",
    }
    samples = pd.DataFrame([{**base, "decision_id": i} for i in range(2)], columns=list(GK_DECISION_SAMPLE_COLUMNS))
    out = summarize_gk_decision(samples)
    assert set(GK_DECISION_KEYS) <= set(out.columns), set(GK_DECISION_KEYS) - set(out.columns)


def test_referenced_constants_are_in_package_all():
    for family, (module, keys_attr, metric_attr, full_attr) in _PKG.items():
        exported = set(importlib.import_module(module).__all__)
        for attr in (keys_attr, metric_attr, full_attr):
            if attr is not None:
                assert attr in exported, f"{family}: {attr} not in {module}.__all__"
