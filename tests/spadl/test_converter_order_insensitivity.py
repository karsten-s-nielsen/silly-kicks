"""Order-insensitivity gate for every SPADL ``convert_to_actions``.

Two complementary gates, per the chronological-``action_id``-invariant design
(``docs/superpowers/specs/2026-08-20-chronological-action-id-invariant-design.md``):

* **Permutation-invariance (primary, §3a).** Convert a native input and a
  timestamp-block permutation of it; drop ``action_id``; assert the two outputs
  are equal as multisets of full SPADL-field rows (canonical-content sort on the
  DISCRETE columns only, then a float-tolerant compare). This catches EVERY
  order-dependent derivation -- end coords, dribbles, skillcorner's
  ``same_team_next``/``is_short`` shifts, wyscout's event logic -- not just the
  index numbering. A non-vacuity assertion pins that the permutation is not a
  no-op.
* **Index-chronology (complementary, §3b).** Within every
  ``(game_id, period_id)`` group, ``action_id``-sorted ``time_seconds`` is
  non-decreasing (finite rows). A cheap direct check of the OUTPUT invariant;
  necessary but not sufficient (it cannot see shift-derived corruption).

RED-FIRST (ADR-051): the observed order-dependent converters are marked
``xfail(strict=True)`` via the two ``_*_BROKEN`` sets below, so CI is green while
each known-broken case is an expected-fail. Task 3 fixes them; a strict xfail
that starts passing FAILS, so a fixed converter cannot keep a stale marker.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.spadl._converter_cases import CONVERTER_CASES, discover_converter_modules

# ---------------------------------------------------------------------------
# Observed order-dependent converters (Task 2 Step 4 discovery; see the module
# docstring). A broken converter fails the permutation gate; whether it ALSO
# fails index-chronology depends on whether it sorts before assigning
# ``action_id`` (skillcorner sorts, so it fails permutation but NOT chronology).
# Each set is applied via strict xfail, so it must match the observed failures
# exactly -- Task 3 removes each name as it lands the fix.
# ---------------------------------------------------------------------------
# Permutation gate: converters whose OUTPUT CONTENT changes with input order.
# gradientsports/metrica/sportec derive end coords + dribbles + action_id over
# raw event order; skillcorner computes same_team_next/is_short by positional
# .shift() BEFORE its time-sort (so type_id/result_id flip -- verified); wyscout
# converts events->actions (incl. duel->take-on collapse) over raw order; opta
# runs _fix_recoveries + _fix_unintentional_ball_touches by file-adjacent .shift()
# BEFORE its time-sort (recovery flips dribble<->non_action; a deflected pass's
# end flips -- verified), so its time-sort does NOT make it order-insensitive.
_PERMUTATION_BROKEN: frozenset[str] = frozenset()
# Index-chronology gate: converters whose action_id is not chronological. Same
# set MINUS skillcorner, which sorts (period_id, time_seconds) before assigning
# action_id -- so its action_id IS chronological even though its shift-derived
# fields are corrupt (exactly the case §3b cannot see; the permutation gate does).
_CHRONOLOGY_BROKEN: frozenset[str] = frozenset()

_XFAIL_REASON = "order-insensitivity fix pending -- Task 3 (chronological action_id invariant)"


def _params(broken: frozenset[str]) -> list:
    out = []
    for name in sorted(CONVERTER_CASES):
        marks = (pytest.mark.xfail(strict=True, reason=_XFAIL_REASON),) if name in broken else ()
        out.append(pytest.param(name, marks=marks, id=name))
    return out


def _discrete_columns(df: pd.DataFrame) -> list[str]:
    """Every non-float column -- the alignment key (L2: never sort on floats)."""
    return [c for c in df.columns if not pd.api.types.is_float_dtype(df[c])]


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Canonical content order: stable-sort by the DISCRETE columns only.

    Float columns (coordinates, time) are deliberately excluded from the sort
    key -- last-bit differences would reorder near-equal rows differently across
    the two runs and misalign the row-wise compare (L2). They are compared with a
    tolerance AFTER alignment.
    """
    df = df.reset_index(drop=True)
    discrete = _discrete_columns(df)
    if discrete:
        key = pd.DataFrame({c: df[c].astype(str) for c in discrete})
        order = key.sort_values(list(discrete), kind="mergesort").index
        df = df.iloc[order].reset_index(drop=True)
    return df


@pytest.mark.parametrize("name", _params(_PERMUTATION_BROKEN))
def test_converter_is_order_insensitive(name: str) -> None:
    case = CONVERTER_CASES[name]
    inp = case.build_input()
    permuted = case.permute(inp)

    # Non-vacuity: the permutation must actually reorder the input, else the gate
    # would prove nothing.
    assert case.signature(permuted) != case.signature(inp), (
        f"{name}: permutation is a no-op -- gate would be vacuous (needs a multi-timestamp input)"
    )

    a = _canonical(case.run(inp).drop(columns=["action_id"]))
    b = _canonical(case.run(permuted).drop(columns=["action_id"]))
    pd.testing.assert_frame_equal(a, b, check_exact=False, atol=1e-9, rtol=1e-9, check_dtype=False)


@pytest.mark.parametrize("name", _params(_CHRONOLOGY_BROKEN))
def test_converter_action_id_is_chronological(name: str) -> None:
    case = CONVERTER_CASES[name]
    out = case.run(case.build_input())

    ts_all = out["time_seconds"].to_numpy(dtype="float64")
    finite = out.loc[np.isfinite(ts_all)]
    for (gid, pid), grp in finite.groupby(["game_id", "period_id"], dropna=False, sort=False):
        ts = grp.sort_values("action_id", kind="mergesort")["time_seconds"].to_numpy(dtype="float64")
        diffs = np.diff(ts)
        assert (diffs >= -1e-9).all(), (
            f"{name}: action_id order is not chronological within "
            f"(game_id={gid!r}, period_id={pid!r}); time_seconds={ts.tolist()}"
        )


def test_every_converter_has_a_case() -> None:
    """Anti-rot meta-assertion (both directions): every ``convert_to_actions`` in
    ``silly_kicks/spadl/`` has a registered case, and no case is orphaned."""
    modules = discover_converter_modules()
    registered = set(CONVERTER_CASES)
    assert modules == registered, (
        f"converter modules {sorted(modules)} != registered CONVERTER_CASES {sorted(registered)}; "
        f"a new converter must carry an order-insensitivity case"
    )
