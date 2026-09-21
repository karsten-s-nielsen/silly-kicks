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


def _packages_exporting_metric_columns() -> set[str]:
    root = pathlib.Path(silly_kicks.__file__).parent
    out: set[str] = set()
    for sub in root.iterdir():
        init = sub / "__init__.py"
        if sub.is_dir() and init.exists() and any(n.endswith("_METRIC_COLUMNS") for n in _pkg_all(init)):
            out.add(sub.name)
    return out


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


def test_completeness_three_bucket():
    derived = _packages_exporting_metric_columns()
    assert set(METRIC_CONTRACTS) == derived, set(METRIC_CONTRACTS) ^ derived
    root = pathlib.Path(silly_kicks.__file__).parent
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
